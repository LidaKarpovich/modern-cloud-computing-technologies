"""Amazon SageMaker Autopilot (AutoML API v2) — табличная классификация/регрессия.

Требования: pip install boto3 sagemaker pandas; IAM-роль с AmazonSageMakerFullAccess и доступом к S3.

    python cloud/aws/sagemaker_autopilot.py --csv data/credit_synth_train.csv --target target \
        --bucket my-automl-bucket --role arn:aws:iam::<acc>:role/SageMakerExecRole \
        --mode ENSEMBLING --max-minutes 30 --deploy

Режимы: ENSEMBLING (AutoGluon-стек внутри Autopilot, быстрый), HYPERPARAMETER_TUNING
(HPO по XGBoost/LightGBM/MLP и др.), AUTO (выбор по размеру данных).
ВНИМАНИЕ: задание и endpoint тарифицируются; в конце скрипт предлагает удалить endpoint.
"""
import argparse
import time

import boto3

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--target", required=True)
ap.add_argument("--bucket", required=True)
ap.add_argument("--role", required=True)
ap.add_argument("--region", default="eu-central-1")
ap.add_argument("--problem", default=None,
                help="BinaryClassification | MulticlassClassification | Regression (None = авто)")
ap.add_argument("--metric", default=None, help="AUC | F1 | Accuracy | MSE | RMSE ...")
ap.add_argument("--mode", default="ENSEMBLING", choices=["ENSEMBLING", "HYPERPARAMETER_TUNING", "AUTO"])
ap.add_argument("--max-minutes", type=int, default=30)
ap.add_argument("--max-candidates", type=int, default=50)
ap.add_argument("--deploy", action="store_true")
a = ap.parse_args()

session = boto3.Session(region_name=a.region)
s3, sm = session.client("s3"), session.client("sagemaker")
job = f"automl-{int(time.time())}"
key = f"{job}/input/train.csv"
s3.upload_file(a.csv, a.bucket, key)
print(f"uploaded s3://{a.bucket}/{key}")

tab_cfg = {
    "TargetAttributeName": a.target,
    "Mode": a.mode,
    "CompletionCriteria": {
        "MaxAutoMLJobRuntimeInSeconds": a.max_minutes * 60,
        "MaxCandidates": a.max_candidates,
    },
    "GenerateCandidateDefinitionsOnly": False,
}
if a.problem:
    tab_cfg["ProblemType"] = a.problem

kwargs = dict(
    AutoMLJobName=job,
    AutoMLJobInputDataConfig=[{
        "ChannelType": "training", "ContentType": "text/csv;header=present",
        "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix",
                                        "S3Uri": f"s3://{a.bucket}/{job}/input/"}}}],
    OutputDataConfig={"S3OutputPath": f"s3://{a.bucket}/{job}/output/"},
    AutoMLProblemTypeConfig={"TabularJobConfig": tab_cfg},
    RoleArn=a.role,
    DataSplitConfig={"ValidationFraction": 0.2},
)
if a.metric:
    kwargs["AutoMLJobObjective"] = {"MetricName": a.metric}
sm.create_auto_ml_job_v2(**kwargs)
print("started", job)

t0 = time.time()
while True:
    d = sm.describe_auto_ml_job_v2(AutoMLJobName=job)
    st, sec = d["AutoMLJobStatus"], d.get("AutoMLJobSecondaryStatus")
    print(f"[{(time.time() - t0) / 60:5.1f} min] {st} / {sec}")
    if st in ("Completed", "Failed", "Stopped"):
        break
    time.sleep(60)
if st != "Completed":
    raise SystemExit(d.get("FailureReason"))

# Лидерборд кандидатов
cands = sm.list_candidates_for_auto_ml_job(AutoMLJobName=job, SortBy="FinalObjectiveMetricValue",
                                           SortOrder="Descending", MaxResults=10)["Candidates"]
print("\nTop candidates:")
for c in cands:
    m = c.get("FinalAutoMLJobObjectiveMetric", {})
    print(f"  {c['CandidateName'][:60]:<60} {m.get('MetricName')}={m.get('Value')}")
best = d["BestCandidate"]
print("\nBEST:", best["CandidateName"], best["FinalAutoMLJobObjectiveMetric"])
print("Explainability/insights report:", d.get("AutoMLJobArtifacts", {}))

if a.deploy:
    model_name = f"{job}-model"
    sm.create_model(ModelName=model_name, Containers=best["InferenceContainers"],
                    ExecutionRoleArn=a.role)
    sm.create_endpoint_config(EndpointConfigName=f"{job}-cfg", ProductionVariants=[{
        "VariantName": "main", "ModelName": model_name,
        "InstanceType": "ml.m5.large", "InitialInstanceCount": 1}])
    sm.create_endpoint(EndpointName=f"{job}-ep", EndpointConfigName=f"{job}-cfg")
    sm.get_waiter("endpoint_in_service").wait(EndpointName=f"{job}-ep")
    print("endpoint ready:", f"{job}-ep")

    import pandas as pd
    rt = session.client("sagemaker-runtime")
    sample = pd.read_csv(a.csv).drop(columns=[a.target]).head(5)
    t1 = time.time()
    resp = rt.invoke_endpoint(EndpointName=f"{job}-ep", ContentType="text/csv",
                              Body=sample.to_csv(index=False, header=False))
    print("prediction:", resp["Body"].read().decode(), f"latency={1000 * (time.time() - t1):.0f}ms")
    if input("Удалить endpoint (рекомендуется, он тарифицируется)? [Y/n] ").lower() != "n":
        sm.delete_endpoint(EndpointName=f"{job}-ep")
        sm.delete_endpoint_config(EndpointConfigName=f"{job}-cfg")
        sm.delete_model(ModelName=model_name)
        print("deleted")
