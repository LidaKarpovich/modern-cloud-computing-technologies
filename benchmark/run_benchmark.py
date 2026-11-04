"""Бенчмарк стратегий мини-AutoML против ручной базовой линии.

Пример:
    python -m benchmark.run_benchmark --budget 30 --seeds 0 1 2 --out benchmark/results
    python -m benchmark.run_benchmark --curve credit_synth --budgets 10 30 60 120
"""
from __future__ import annotations

import argparse
import json
import os
import time

import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from automl_core import AutoML, evaluate_holdout
from automl_core.engine import build_preprocessor, infer_task
from benchmark.datasets import DATASETS

STRATEGIES = ["default", "random", "halving", "ensemble"]


class Baseline:
    """Что сделал бы инженер за 5 минут: стандартизация + логистическая регрессия / Ridge."""
    def __init__(self):
        self.members_ = None

    def fit(self, X, y):
        from sklearn.preprocessing import LabelEncoder
        self.task_ = infer_task(pd.Series(y))
        self.label_encoder_ = LabelEncoder().fit(y) if self.task_ != "regression" else None
        est = Ridge() if self.task_ == "regression" else LogisticRegression(max_iter=2000)
        self.pipe = Pipeline([("prep", build_preprocessor(X, scale=True)), ("model", est)])
        yy = y if self.label_encoder_ is None else self.label_encoder_.transform(y)
        self.pipe.fit(X, yy)
        return self

    def predict(self, X):
        return self.pipe.predict(X)

    def predict_proba(self, X):
        return self.pipe.predict_proba(X)


def split(name, seed):
    X, y = DATASETS[name]()
    strat = None if infer_task(y) == "regression" else y
    return train_test_split(X, y, test_size=0.25, random_state=seed, stratify=strat)


def run_one(name, strategy, budget, seed):
    Xtr, Xte, ytr, yte = split(name, seed)
    t0 = time.time()
    if strategy == "baseline":
        m = Baseline().fit(Xtr, ytr)
        info = {"n_trials": 1, "members": "linear"}
    else:
        m = AutoML(time_budget=budget, strategy=strategy, seed=seed).fit(Xtr, ytr)
        d = m.describe()
        info = {"n_trials": d["n_trials"], "cv_score": d["cv_score"],
                "members": "+".join(f"{x['family']}:{x['weight']}" for x in d["members"])}
    wall = time.time() - t0
    res = {"dataset": name, "strategy": strategy, "budget_s": budget, "seed": seed,
           "n_train": len(Xtr), "n_features": Xtr.shape[1], "task": m.task_,
           "wall_time_s": round(wall, 1), **info, **evaluate_holdout(m, Xte, yte)}
    print(json.dumps({k: v for k, v in res.items() if k != "members"}, ensure_ascii=False))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=30)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--strategies", nargs="+", default=["baseline"] + STRATEGIES)
    ap.add_argument("--curve", help="датасет для кривой «бюджет → качество»")
    ap.add_argument("--budgets", type=float, nargs="+", default=[10, 30, 60, 120])
    ap.add_argument("--out", default="benchmark/results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = []
    if a.curve:
        for b in a.budgets:
            for s in ["random", "halving", "ensemble"]:
                for seed in a.seeds:
                    rows.append(run_one(a.curve, s, b, seed))
        fname = f"curve_{a.curve}.csv"
    else:
        for name in a.datasets:
            for s in a.strategies:
                for seed in a.seeds:
                    rows.append(run_one(name, s, a.budget, seed))
        fname = "benchmark.csv"
    pd.DataFrame(rows).to_csv(os.path.join(a.out, fname), index=False)
    print("saved", os.path.join(a.out, fname))


if __name__ == "__main__":
    main()
