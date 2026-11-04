"""Единый запуск open-source AutoML-фреймворков на тех же разбиениях, что и benchmark/.

Выводит метрики в том же формате, что benchmark/run_benchmark.py, чтобы результаты
можно было дописать в общую таблицу (benchmark/results/oss.csv).

    pip install flaml[automl] autogluon.tabular h2o lightautoml   # по необходимости
    python -m oss.run_oss --framework flaml --dataset credit_synth --budget 60
    python -m oss.run_oss --framework autogluon --dataset credit_synth --budget 60
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, log_loss, mean_squared_error, r2_score,
                             roc_auc_score)

from benchmark.run_benchmark import split
from automl_core.engine import infer_task


def metrics(task, y, proba=None, pred=None, classes=None):
    if task == "regression":
        return {"rmse": float(np.sqrt(mean_squared_error(y, pred))), "r2": float(r2_score(y, pred))}
    yi = pd.Series(y).map({c: i for i, c in enumerate(classes)}).to_numpy()
    out = {"accuracy": float(accuracy_score(yi, proba.argmax(1))),
           "logloss": float(log_loss(yi, np.clip(proba, 1e-7, 1), labels=np.arange(len(classes))))}
    out["roc_auc"] = float(roc_auc_score(yi, proba[:, 1]) if task == "binary"
                           else roc_auc_score(yi, proba, multi_class="ovr"))
    return out


def run_flaml(Xtr, ytr, Xte, task, budget):
    from flaml import AutoML
    m = AutoML()
    m.fit(Xtr, ytr, task="regression" if task == "regression" else "classification",
          time_budget=budget, metric="rmse" if task == "regression" else
          ("roc_auc" if task == "binary" else "log_loss"), verbose=0, seed=0)
    if task == "regression":
        return None, m.predict(Xte), None, m.best_estimator
    return m.predict_proba(Xte), None, list(m.classes_), m.best_estimator


def run_autogluon(Xtr, ytr, Xte, task, budget):
    from autogluon.tabular import TabularPredictor
    df = Xtr.copy(); df["__y"] = ytr.values
    p = TabularPredictor(label="__y", problem_type=task, verbosity=0,
                         path=f"/tmp/ag_{int(time.time())}").fit(
        df, time_limit=budget, presets="medium_quality")
    best = p.model_best
    if task == "regression":
        return None, p.predict(Xte).to_numpy(), None, best
    pr = p.predict_proba(Xte)
    return pr.to_numpy(), None, list(pr.columns), best


def run_h2o(Xtr, ytr, Xte, task, budget):
    import h2o
    from h2o.automl import H2OAutoML
    h2o.init(nthreads=-1, verbose=False)
    tr = h2o.H2OFrame(pd.concat([Xtr, ytr.rename("__y")], axis=1))
    if task != "regression":
        tr["__y"] = tr["__y"].asfactor()
    aml = H2OAutoML(max_runtime_secs=int(budget), seed=0, verbosity=None)
    aml.train(y="__y", training_frame=tr)
    pr = aml.leader.predict(h2o.H2OFrame(Xte)).as_data_frame()
    if task == "regression":
        return None, pr["predict"].to_numpy(), None, aml.leader.model_id
    cls = [c for c in pr.columns if c != "predict"]
    return pr[cls].to_numpy(), None, [type(ytr.iloc[0])(c.lstrip("p")) for c in cls], aml.leader.model_id


def run_lama(Xtr, ytr, Xte, task, budget):
    from lightautoml.automl.presets.tabular_presets import TabularAutoML
    from lightautoml.tasks import Task
    df = Xtr.copy(); df["__y"] = ytr.values
    name = {"binary": "binary", "multiclass": "multiclass", "regression": "reg"}[task]
    m = TabularAutoML(task=Task(name), timeout=budget, reader_params={"random_state": 0})
    m.fit_predict(df, roles={"target": "__y"}, verbose=0)
    p = m.predict(Xte).data
    if task == "regression":
        return None, p[:, 0], None, "LAMA"
    if task == "binary":
        return np.c_[1 - p[:, 0], p[:, 0]], None, sorted(ytr.unique()), "LAMA"
    return p, None, sorted(ytr.unique()), "LAMA"


RUNNERS = {"flaml": run_flaml, "autogluon": run_autogluon, "h2o": run_h2o, "lightautoml": run_lama}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--framework", choices=RUNNERS, required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--budget", type=float, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="benchmark/results/oss.csv")
    a = ap.parse_args()
    Xtr, Xte, ytr, yte = split(a.dataset, a.seed)
    task = infer_task(ytr)
    t0 = time.time()
    proba, pred, classes, best = RUNNERS[a.framework](Xtr, ytr, Xte, task, a.budget)
    res = {"dataset": a.dataset, "strategy": a.framework, "budget_s": a.budget, "seed": a.seed,
           "wall_time_s": round(time.time() - t0, 1), "members": str(best),
           **metrics(task, yte, proba, pred, classes)}
    print(json.dumps(res, ensure_ascii=False))
    pd.DataFrame([res]).to_csv(a.out, mode="a", header=not os.path.exists(a.out), index=False)
