"""Один trial Katib: обучает HistGradientBoosting и печатает метрику в формате 'auc=<value>'
(Katib StdOut metrics collector парсит строки <metric-name>=<value>)."""
import argparse

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

from automl_core.engine import build_preprocessor
from benchmark.run_benchmark import split

ap = argparse.ArgumentParser()
ap.add_argument("--lr", type=float, default=0.1)
ap.add_argument("--leaves", type=int, default=31)
ap.add_argument("--l2", type=float, default=0.0)
a = ap.parse_args()

Xtr, Xva, ytr, yva = split("credit_synth", 0)
model = make_pipeline(build_preprocessor(Xtr, scale=False),
                      HistGradientBoostingClassifier(learning_rate=a.lr, max_leaf_nodes=a.leaves,
                                                     l2_regularization=a.l2, random_state=0))
model.fit(Xtr, ytr)
print(f"auc={roc_auc_score(yva, model.predict_proba(Xva)[:, 1]):.6f}")
