"""VK Cloud ML Platform: FEDOT (ИТМО) в JupyterHub + логирование в MLflow,
затем публикация модели через MLflow Deploy (REST API в Docker).

В ноутбуке JupyterHub VK Cloud MLflow уже настроен (MLFLOW_TRACKING_URI в окружении).
    pip install fedot mlflow
    python fedot_mlflow.py --train credit_synth_train.csv --target target --timeout 10
"""
import argparse

import mlflow
import pandas as pd
from fedot.api.main import Fedot
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ap = argparse.ArgumentParser()
ap.add_argument("--train", required=True)
ap.add_argument("--target", required=True)
ap.add_argument("--timeout", type=float, default=10, help="минуты")
a = ap.parse_args()

df = pd.read_csv(a.train)
tr, te = train_test_split(df, test_size=0.25, random_state=0, stratify=df[a.target])
mlflow.set_experiment("automl-fedot")
with mlflow.start_run(run_name="fedot-credit"):
    model = Fedot(problem="classification", metric="roc_auc", timeout=a.timeout,
                  preset="best_quality", n_jobs=-1, seed=0)
    model.fit(features=tr.drop(columns=[a.target]), target=tr[a.target])
    proba = model.predict_proba(features=te.drop(columns=[a.target]))
    auc = roc_auc_score(te[a.target], proba if proba.ndim == 1 else proba[:, -1])
    mlflow.log_params({"framework": "FEDOT", "timeout_min": a.timeout})
    mlflow.log_metric("holdout_auc", auc)
    model.current_pipeline.show(save_path="pipeline.png")
    mlflow.log_artifact("pipeline.png")
    model.current_pipeline.save(path="fedot_pipeline", create_subdir=False)
    mlflow.log_artifacts("fedot_pipeline", artifact_path="model")
    print("holdout ROC-AUC:", round(auc, 4))
    print("pipeline:", model.current_pipeline.descriptive_id)
