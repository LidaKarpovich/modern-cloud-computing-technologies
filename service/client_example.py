"""Пример работы с AutoML-сервисом: обучение на CSV и предсказание.

    python service/client_example.py --url http://<floating-ip>:8000 --csv data.csv --target label
"""
import argparse
import time

import pandas as pd
import requests

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8000")
ap.add_argument("--csv", required=True)
ap.add_argument("--target", required=True)
ap.add_argument("--budget", type=float, default=60)
ap.add_argument("--strategy", default="ensemble")
a = ap.parse_args()

with open(a.csv, "rb") as f:
    r = requests.post(f"{a.url}/train", files={"file": f},
                      data={"target": a.target, "time_budget": a.budget,
                            "strategy": a.strategy})
r.raise_for_status()
job_id = r.json()["job_id"]
print("job:", job_id)

while (job := requests.get(f"{a.url}/jobs/{job_id}").json())["status"] == "running":
    time.sleep(5)
print("status:", job["status"])
if job["status"] != "succeeded":
    raise SystemExit(job.get("error"))
print("CV:", job["result"]["metric"], round(job["result"]["cv_score"], 4),
      "| trials:", job["result"]["n_trials"])
for m in job["result"]["members"]:
    print(f"  {m['family']:<14} weight={m['weight']}")

sample = pd.read_csv(a.csv).drop(columns=[a.target]).head(5)
records = sample.astype(object).where(sample.notna(), None).to_dict(orient="records")  # NaN -> null
pred = requests.post(f"{a.url}/predict/{job_id}", json={"records": records}).json()
print("predictions:", pred["predictions"])
