"""AutoML-as-a-Service: REST API поверх automl_core.

Эндпоинты
---------
GET  /health                    — проверка живости (для балансировщика Octavia / k8s probe)
POST /train                     — загрузить CSV (multipart: file, target, time_budget, strategy)
GET  /jobs                      — список задач обучения
GET  /jobs/{job_id}             — статус, метрика CV, лидерборд
POST /predict/{job_id}          — JSON {"records": [{...}, ...]} -> предсказания
GET  /metrics                   — простые метрики в формате Prometheus

Хранилище моделей — каталог MODEL_DIR (в OpenStack это том Cinder). Если заданы S3_ENDPOINT /
S3_BUCKET, артефакты дополнительно выгружаются в S3-совместимое хранилище
(OpenStack Swift s3api, Ceph RGW, Yandex Object Storage, VK Cloud S3).

Запуск: uvicorn service.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import io
import json
import os
import threading
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route

from automl_core import AutoML

MODEL_DIR = os.environ.get("MODEL_DIR", "./models")
MAX_WORKERS = int(os.environ.get("MAX_TRAIN_WORKERS", "1"))
MAX_BUDGET = float(os.environ.get("MAX_TIME_BUDGET", "3600"))
os.makedirs(MODEL_DIR, exist_ok=True)

# Обучение идёт в дочерних процессах: оно не блокирует API (GIL), а OpenMP-потоки
# HistGradientBoosting не конкурируют с event loop. TRAIN_THREADS — лимит BLAS/OpenMP на задачу.
TRAIN_THREADS = int(os.environ.get("TRAIN_THREADS", str(os.cpu_count() or 1)))
_pool = ProcessPoolExecutor(max_workers=MAX_WORKERS)
_jobs: dict[str, dict] = {}
_models: dict[str, AutoML] = {}
_lock = threading.Lock()
_counters = {"train_requests": 0, "predict_requests": 0, "predicted_rows": 0}


# ---------------------------------------------------------------- storage
def _s3_upload(local_path: str, key: str):
    endpoint, bucket = os.environ.get("S3_ENDPOINT"), os.environ.get("S3_BUCKET")
    if not (endpoint and bucket):
        return None
    import boto3  # опциональная зависимость
    boto3.client("s3", endpoint_url=endpoint).upload_file(local_path, bucket, key)
    return f"s3://{bucket}/{key}"


def _meta_path(job_id):
    return os.path.join(MODEL_DIR, f"{job_id}.json")


def _save_meta(job_id):
    with open(_meta_path(job_id), "w") as f:
        json.dump(_jobs[job_id], f, ensure_ascii=False, default=str)


def _load_existing():
    """После рестарта контейнера поднимаем метаданные задач с тома."""
    for fn in os.listdir(MODEL_DIR):
        if fn.endswith(".json"):
            with open(os.path.join(MODEL_DIR, fn)) as f:
                meta = json.load(f)
            if meta.get("status") == "running":
                meta["status"] = "failed"
                meta["error"] = "interrupted by restart"
            _jobs[meta["job_id"]] = meta


def _get_model(job_id) -> AutoML:
    if job_id not in _models:
        _models[job_id] = AutoML.load(os.path.join(MODEL_DIR, f"{job_id}.joblib"))
    return _models[job_id]


# ---------------------------------------------------------------- training
def _train_worker(job_id, df, target, budget, strategy, model_dir, threads):
    """Выполняется в дочернем процессе; модель кладёт на диск, возвращает метаданные."""
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=threads):
        y = df.pop(target)
        model = AutoML(time_budget=budget, strategy=strategy).fit(df, y)
    path = os.path.join(model_dir, f"{job_id}.joblib")
    model.save(path)
    return {"result": model.describe(), "features": list(df.columns),
            "leaderboard": model.leaderboard_.head(10).to_dict(orient="records"),
            "path": path}


def _on_done(job_id, fut):
    try:
        out = fut.result()
        with _lock:
            _jobs[job_id].update(
                status="succeeded", finished_at=time.time(), result=out["result"],
                features=out["features"], leaderboard=out["leaderboard"],
                artifact=_s3_upload(out["path"], f"models/{job_id}.joblib") or out["path"])
    except Exception as e:
        with _lock:
            _jobs[job_id].update(status="failed", error=f"{e}\n{traceback.format_exc()}")
    _save_meta(job_id)


# ---------------------------------------------------------------- handlers
async def health(_):
    return JSONResponse({"status": "ok", "jobs": len(_jobs)})


async def train(request: Request):
    form = await request.form()
    upload = form.get("file")
    target = form.get("target")
    if upload is None or not target:
        return JSONResponse({"error": "нужны поля file (CSV) и target"}, status_code=400)
    df = pd.read_csv(io.BytesIO(await upload.read()))
    if target not in df.columns:
        return JSONResponse({"error": f"колонка {target!r} не найдена"}, status_code=400)
    budget = min(float(form.get("time_budget", 60)), MAX_BUDGET)
    strategy = form.get("strategy", "ensemble")
    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {"job_id": job_id, "status": "running", "target": target,
                     "rows": len(df), "columns": df.shape[1], "time_budget": budget,
                     "strategy": strategy, "created_at": time.time()}
    _counters["train_requests"] += 1
    _save_meta(job_id)
    fut = _pool.submit(_train_worker, job_id, df, target, budget, strategy,
                       MODEL_DIR, TRAIN_THREADS)
    fut.add_done_callback(lambda f: _on_done(job_id, f))
    return JSONResponse({"job_id": job_id, "status": "running"}, status_code=202)


async def list_jobs(_):
    return JSONResponse([{k: j.get(k) for k in ("job_id", "status", "target", "rows",
                                                "strategy", "time_budget")}
                         for j in _jobs.values()])


async def get_job(request: Request):
    job = _jobs.get(request.path_params["job_id"])
    if job is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(json.loads(json.dumps(job, default=str)))


async def predict(request: Request):
    job_id = request.path_params["job_id"]
    job = _jobs.get(job_id)
    if job is None or job["status"] != "succeeded":
        return JSONResponse({"error": "модель не готова"}, status_code=409)
    body = await request.json()
    X = pd.DataFrame(body["records"])[job["features"]]
    model = _get_model(job_id)
    out = {"predictions": model.predict(X).tolist()}
    if model.task_ != "regression":
        proba = model.predict_proba(X)
        out["classes"] = model.label_encoder_.classes_.tolist()
        out["probabilities"] = proba.round(5).tolist()
    _counters["predict_requests"] += 1
    _counters["predicted_rows"] += len(X)
    return JSONResponse(json.loads(json.dumps(out, default=str)))


async def metrics(_):
    lines = [f"automl_{k}_total {v}" for k, v in _counters.items()]
    for s in ("running", "succeeded", "failed"):
        lines.append(f'automl_jobs{{status="{s}"}} '
                     f'{sum(j["status"] == s for j in _jobs.values())}')
    return PlainTextResponse("\n".join(lines) + "\n")


_load_existing()
app = Starlette(routes=[
    Route("/health", health),
    Route("/train", train, methods=["POST"]),
    Route("/jobs", list_jobs),
    Route("/jobs/{job_id}", get_job),
    Route("/predict/{job_id}", predict, methods=["POST"]),
    Route("/metrics", metrics),
])
