"""LightAutoML (Sber AI Lab) — запуск внутри Yandex DataSphere Job (или любой ВМ)."""
import argparse
import json
import time

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task

ap = argparse.ArgumentParser()
ap.add_argument("--train", required=True)
ap.add_argument("--target", required=True)
ap.add_argument("--timeout", type=int, default=600)
ap.add_argument("--cpu", type=int, default=8)
ap.add_argument("--model", default="model.pkl")
ap.add_argument("--report", default="report.json")
a = ap.parse_args()

df = pd.read_csv(a.train)
tr, te = train_test_split(df, test_size=0.25, random_state=0, stratify=df[a.target])
automl = TabularAutoML(task=Task("binary", metric="auc"), timeout=a.timeout, cpu_limit=a.cpu,
                       reader_params={"n_jobs": a.cpu, "cv": 5, "random_state": 0})
t0 = time.time()
oof = automl.fit_predict(tr, roles={"target": a.target}, verbose=1)
fit_time = time.time() - t0
pred = automl.predict(te).data[:, 0]
report = {"framework": "LightAutoML", "fit_time_s": round(fit_time, 1),
          "oof_auc": float(roc_auc_score(tr[a.target], oof.data[:, 0])),
          "holdout_auc": float(roc_auc_score(te[a.target], pred)),
          "structure": automl.create_model_str_desc()}
print(json.dumps(report, ensure_ascii=False, indent=2))
joblib.dump(automl, a.model)
json.dump(report, open(a.report, "w"), ensure_ascii=False, indent=2)
