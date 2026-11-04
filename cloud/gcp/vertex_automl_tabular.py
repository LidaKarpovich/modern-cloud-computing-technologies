"""Google Cloud Vertex AI AutoML Tabular.

Требования: pip install google-cloud-aiplatform; gcloud auth application-default login;
включённый API aiplatform.googleapis.com; бакет GCS в том же регионе.

    python cloud/gcp/vertex_automl_tabular.py --project my-proj --region europe-west4 \
        --bucket gs://my-automl-bucket --csv data/credit_synth_train.csv --target target \
        --type classification --budget-milli-node-hours 1000

Минимальный бюджет — 1000 milli node hours (1 узло-час). Для полного контроля над шагами
(feature engineering, тюнинг, дистилляция) используйте Tabular Workflow for End-to-End AutoML
(google_cloud_pipeline_components), см. README.
"""
import argparse
import time

from google.cloud import aiplatform, storage

ap = argparse.ArgumentParser()
ap.add_argument("--project", required=True)
ap.add_argument("--region", default="europe-west4")
ap.add_argument("--bucket", required=True, help="gs://bucket")
ap.add_argument("--csv", required=True)
ap.add_argument("--target", required=True)
ap.add_argument("--type", default="classification", choices=["classification", "regression"])
ap.add_argument("--objective", default=None,
                help="maximize-au-roc | minimize-log-loss | minimize-rmse ...")
ap.add_argument("--budget-milli-node-hours", type=int, default=1000)
ap.add_argument("--deploy", action="store_true")
a = ap.parse_args()

aiplatform.init(project=a.project, location=a.region, staging_bucket=a.bucket)
name = f"automl-{int(time.time())}"
bucket_name = a.bucket.replace("gs://", "").split("/")[0]
blob = f"{name}/train.csv"
storage.Client(project=a.project).bucket(bucket_name).blob(blob).upload_from_filename(a.csv)

ds = aiplatform.TabularDataset.create(display_name=name, gcs_source=[f"gs://{bucket_name}/{blob}"])
job = aiplatform.AutoMLTabularTrainingJob(
    display_name=name, optimization_prediction_type=a.type,
    optimization_objective=a.objective)
model = job.run(dataset=ds, target_column=a.target,
                budget_milli_node_hours=a.budget_milli_node_hours,
                training_fraction_split=0.8, validation_fraction_split=0.1,
                test_fraction_split=0.1, model_display_name=f"{name}-model",
                disable_early_stopping=False)

for ev in model.list_model_evaluations():
    m = ev.to_dict()["metrics"]
    keys = ["auRoc", "auPrc", "logLoss", "rootMeanSquaredError", "rSquared"]
    print({k: m[k] for k in keys if k in m})

if a.deploy:
    ep = model.deploy(machine_type="n1-standard-2", min_replica_count=1, max_replica_count=1)
    import pandas as pd
    sample = pd.read_csv(a.csv).drop(columns=[a.target]).head(3).astype(str)
    print(ep.predict(instances=sample.to_dict(orient="records")))
    if input("Undeploy и удалить endpoint? [Y/n] ").lower() != "n":
        ep.undeploy_all()
        ep.delete()
