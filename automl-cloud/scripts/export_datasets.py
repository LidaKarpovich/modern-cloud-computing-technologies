"""Выгрузка тех же train/test-разбиений в CSV (колонка цели — `target`) для запуска
в облачных AutoML-сервисах (SageMaker, Vertex AI, Azure ML, DataSphere, ModelArts, VK Cloud).

    python -m scripts.export_datasets --out data
"""
import argparse
import os

from benchmark.datasets import DATASETS
from benchmark.run_benchmark import split

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
for name in DATASETS:
    Xtr, Xte, ytr, yte = split(name, a.seed)
    Xtr.assign(target=ytr.values).to_csv(f"{a.out}/{name}_train.csv", index=False)
    Xte.assign(target=yte.values).to_csv(f"{a.out}/{name}_test.csv", index=False)
    print(f"{name}: train={len(Xtr)} test={len(Xte)} -> {a.out}/{name}_*.csv")
