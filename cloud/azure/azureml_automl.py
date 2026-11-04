"""Azure Machine Learning Automated ML (Python SDK v2, пакет azure-ai-ml).

SDK v1 (AutoMLConfig) больше не поддерживается — используйте только v2.

Требования: pip install azure-ai-ml azure-identity mltable; az login;
рабочая область Azure ML и вычислительный кластер (например, Standard_DS3_v2, 0..4 узла).

    python cloud/azure/azureml_automl.py --subscription <id> --resource-group rg-ml \
        --workspace ws-ml --compute cpu-cluster --csv data/credit_synth_train.csv --target target
"""
import argparse
import os
import shutil
import tempfile

from azure.ai.ml import Input, MLClient, automl
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

ap = argparse.ArgumentParser()
ap.add_argument("--subscription", required=True)
ap.add_argument("--resource-group", required=True)
ap.add_argument("--workspace", required=True)
ap.add_argument("--compute", default="cpu-cluster")
ap.add_argument("--csv", required=True)
ap.add_argument("--target", required=True)
ap.add_argument("--task", default="classification", choices=["classification", "regression"])
ap.add_argument("--metric", default="AUC_weighted")
ap.add_argument("--timeout-minutes", type=int, default=30)
ap.add_argument("--max-trials", type=int, default=20)
a = ap.parse_args()

ml = MLClient(DefaultAzureCredential(), a.subscription, a.resource_group, a.workspace)

# MLTable — табличный ассет Azure ML: каталог с CSV и файлом MLTable
tmp = tempfile.mkdtemp()
shutil.copy(a.csv, os.path.join(tmp, "train.csv"))
with open(os.path.join(tmp, "MLTable"), "w") as f:
    f.write("paths:\n  - file: ./train.csv\ntransformations:\n"
            "  - read_delimited:\n      delimiter: ','\n      encoding: 'utf8'\n"
            "      header: all_files_same_headers\n")
train = Input(type=AssetTypes.MLTABLE, path=tmp)

common = dict(compute=a.compute, experiment_name="automl-coursework",
              training_data=train, target_column_name=a.target, n_cross_validations=5,
              enable_model_explainability=True)
job = (automl.classification(primary_metric=a.metric, **common) if a.task == "classification"
       else automl.regression(primary_metric="normalized_root_mean_squared_error", **common))
job.set_limits(timeout_minutes=a.timeout_minutes, trial_timeout_minutes=10,
               max_trials=a.max_trials, max_concurrent_trials=4,
               enable_early_termination=True)
job.set_training(enable_stack_ensemble=True, enable_vote_ensemble=True,
                 enable_onnx_compatible_models=True)
job.set_featurization(mode="auto")

returned = ml.jobs.create_or_update(job)
print("Studio URL:", returned.services["Studio"].endpoint)
ml.jobs.stream(returned.name)

# Лучший дочерний запуск (через MLflow)
import mlflow
mlflow.set_tracking_uri(ml.workspaces.get(a.workspace).mlflow_tracking_uri)
parent = mlflow.get_run(returned.name)
best_id = parent.data.tags.get("automl_best_child_run_id")
best = mlflow.get_run(best_id)
print("BEST run:", best_id, {k: v for k, v in best.data.metrics.items()
                              if k in ("AUC_weighted", "accuracy", "log_loss",
                                       "normalized_root_mean_squared_error", "r2_score")})
print("Развертывание: см. cloud/azure/README.md (managed online endpoint из лучшей модели).")
