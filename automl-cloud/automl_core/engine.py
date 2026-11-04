"""Мини-AutoML на scikit-learn.

Реализует четыре стратегии, которые воспроизводят ключевые идеи промышленных AutoML-систем:

* ``default``  — каждая модель из пространства с параметрами по умолчанию, выбирается лучшая
                 по кросс-валидации (аналог «перебора алгоритмов» без HPO);
* ``random``   — случайный поиск по семействам моделей и гиперпараметрам в пределах
                 бюджета времени (базовая линия HPO, Bergstra & Bengio, 2012);
* ``halving``  — successive halving / Hyperband: много конфигураций оцениваются на малой
                 доле данных, лучшие 1/eta переходят на следующий уровень (Li et al., 2017);
* ``ensemble`` — случайный поиск + жадный отбор ансамбля по out-of-fold предсказаниям
                 (Caruana et al., 2004) — так устроены финальные модели AutoGluon, auto-sklearn,
                 Azure AutoML, SageMaker Autopilot (ensembling mode).

Использование::

    from automl_core import AutoML
    m = AutoML(time_budget=60, strategy="ensemble").fit(X_train, y_train)
    m.predict(X_test); m.leaderboard_
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, log_loss, mean_squared_error,
                             r2_score, roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from . import search_space as ss

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------ helpers
def infer_task(y: pd.Series) -> str:
    if pd.api.types.is_float_dtype(y) and y.nunique() > 20:
        return "regression"
    return "binary" if y.nunique() == 2 else "multiclass"


def build_preprocessor(X: pd.DataFrame, scale: bool) -> ColumnTransformer:
    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    transformers = [("num", Pipeline(num_steps), num)]
    if cat:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10,
                                     sparse_output=False)),
        ]), cat))
    return ColumnTransformer(transformers)


def score(task: str, y_true, pred) -> float:
    """Чем больше — тем лучше. pred: вероятности (классификация) или значения (регрессия)."""
    if task == "regression":
        return -float(np.sqrt(mean_squared_error(y_true, pred)))
    if task == "binary":
        return float(roc_auc_score(y_true, pred[:, 1]))
    return -float(log_loss(y_true, np.clip(pred, 1e-7, 1 - 1e-7), labels=np.arange(pred.shape[1])))


METRIC_NAME = {"regression": "-RMSE", "binary": "ROC-AUC", "multiclass": "-LogLoss"}


@dataclass
class Trial:
    family: str
    params: dict
    score: float
    fit_time: float
    data_fraction: float = 1.0
    oof: np.ndarray | None = field(default=None, repr=False)


# ------------------------------------------------------------------ AutoML
class AutoML:
    def __init__(self, time_budget: float = 60, strategy: str = "ensemble",
                 task: str = "auto", cv: int = 5, seed: int = 0,
                 eta: int = 3, ensemble_size: int = 25, verbose: bool = False):
        assert strategy in {"default", "random", "halving", "ensemble"}
        self.time_budget, self.strategy, self.task = time_budget, strategy, task
        self.cv, self.seed, self.eta = cv, seed, eta
        self.ensemble_size, self.verbose = ensemble_size, verbose

    # ---------------------------------------------------------- internals
    def _pipeline(self, family, params):
        return Pipeline([
            ("prep", build_preprocessor(self._X, scale=family in ss.NEEDS_SCALING)),
            ("model", ss.make_estimator(family, params, self.task_, self.seed)),
        ])

    def _splitter(self, k):
        cls = KFold if self.task_ == "regression" else StratifiedKFold
        return cls(n_splits=k, shuffle=True, random_state=self.seed)

    def _evaluate(self, family, params, idx=None, k=None) -> Trial:
        X, y = (self._X, self._y) if idx is None else (self._X.iloc[idx], self._y[idx])
        k = k or self.cv
        method = "predict" if self.task_ == "regression" else "predict_proba"
        t0 = time.time()
        try:
            oof = cross_val_predict(self._pipeline(family, params), X, y,
                                    cv=self._splitter(k), method=method, n_jobs=1)
            s = score(self.task_, y, oof)
        except Exception as e:  # некорректная конфигурация — штрафуем, но не падаем
            if self.verbose:
                print(f"  ! {family}: {e}")
            oof, s = None, -np.inf
        tr = Trial(family, params, s, time.time() - t0,
                   1.0 if idx is None else len(idx) / len(self._y),
                   oof if idx is None else None)
        self.trials_.append(tr)
        if self.verbose:
            print(f"  {family:<14} frac={tr.data_fraction:.2f} score={s:.4f} ({tr.fit_time:.1f}s)")
        return tr

    def _time_left(self):
        return self.time_budget - (time.time() - self._t0)

    # ---------------------------------------------------------- strategies
    def _run_default(self):
        for fam in self._families:
            if self._time_left() <= 0:
                break
            self._evaluate(fam, ss.default_params(fam, self.task_))

    def _run_random(self):
        self._run_default()            # «тёплый старт» — дефолты всех семейств
        i = 0
        while self._time_left() > 0:
            fam = self._families[i % len(self._families)]
            self._evaluate(fam, ss.sample(fam, self._rng, self.task_))
            i += 1

    def _run_halving(self):
        n = len(self._y)
        eta = self.eta
        min_frac = max(1 / eta ** 2, min(1.0, 200 / n))   # не меньше ~200 строк на нижнем уровне
        while self._time_left() > 0:
            # один «бракет» Hyperband: eta^2 конфигураций -> eta -> 1
            configs = [(f, ss.sample(f, self._rng, self.task_))
                       for f in self._rng.choice(self._families, size=eta ** 2)]
            frac = min_frac
            while configs and self._time_left() > 0:
                if frac >= 1.0:
                    for f, p in configs:
                        if self._time_left() <= 0:
                            break
                        self._evaluate(f, p)
                    break
                idx = self._rng.choice(n, size=int(frac * n), replace=False)
                res = []
                for f, p in configs:
                    if self._time_left() <= 0:
                        break
                    res.append((self._evaluate(f, p, idx=idx, k=3).score, f, p))
                res.sort(key=lambda r: r[0], reverse=True)
                configs = [(f, p) for _, f, p in res[: max(1, len(res) // eta)]]
                frac = min(1.0, frac * eta)

    def _select_ensemble(self):
        """Жадный отбор ансамбля с возвращением (Caruana, 2004) по OOF-предсказаниям."""
        pool = sorted([t for t in self.trials_ if t.oof is not None and np.isfinite(t.score)],
                      key=lambda t: t.score, reverse=True)[:20]
        counts = np.zeros(len(pool), dtype=int)
        current = None
        for _ in range(self.ensemble_size):
            best_j, best_s = None, -np.inf
            for j, t in enumerate(pool):
                cand = t.oof if current is None else (current * counts.sum() + t.oof) / (counts.sum() + 1)
                s = score(self.task_, self._y, cand)
                if s > best_s:
                    best_j, best_s = j, s
            counts[best_j] += 1
            t = pool[best_j]
            current = t.oof.copy() if current is None else \
                (current * (counts.sum() - 1) + t.oof) / counts.sum()
        self.ensemble_score_ = score(self.task_, self._y, current)
        return [(pool[j], c / counts.sum()) for j, c in enumerate(counts) if c > 0]

    # ---------------------------------------------------------- public API
    def fit(self, X: pd.DataFrame, y):
        self._t0 = time.time()
        self._rng = np.random.default_rng(self.seed)
        X = pd.DataFrame(X).reset_index(drop=True)
        y = pd.Series(np.asarray(y))
        self.task_ = infer_task(y) if self.task == "auto" else self.task
        self.label_encoder_ = None
        if self.task_ != "regression":
            self.label_encoder_ = LabelEncoder().fit(y)
            y = pd.Series(self.label_encoder_.transform(y))
        self._X, self._y = X, y.to_numpy()
        self._families = ss.families_for(len(y))
        self.trials_: list[Trial] = []

        {"default": self._run_default, "random": self._run_random,
         "halving": self._run_halving, "ensemble": self._run_random}[self.strategy]()
        self.search_time_ = time.time() - self._t0

        full = [t for t in self.trials_ if t.data_fraction == 1.0 and np.isfinite(t.score)]
        if not full:   # бюджет кончился раньше, чем что-то дошло до 100% данных
            best = max(self.trials_, key=lambda t: t.score)
            full = [self._evaluate(best.family, best.params)]
        if self.strategy == "ensemble":
            members = self._select_ensemble()
        else:
            members = [(max(full, key=lambda t: t.score), 1.0)]
        self.members_ = [(self._pipeline(t.family, t.params).fit(X, self._y), w, t)
                         for t, w in members]
        self.best_score_cv_ = (self.ensemble_score_ if self.strategy == "ensemble"
                               else members[0][0].score)
        self.total_time_ = time.time() - self._t0
        del self._X
        return self

    def predict_proba(self, X):
        assert self.task_ != "regression"
        return sum(w * m.predict_proba(pd.DataFrame(X)) for m, w, _ in self.members_)

    def predict(self, X):
        if self.task_ == "regression":
            return sum(w * m.predict(pd.DataFrame(X)) for m, w, _ in self.members_)
        return self.label_encoder_.inverse_transform(self.predict_proba(X).argmax(1))

    @property
    def leaderboard_(self) -> pd.DataFrame:
        rows = [{"family": t.family, "data_fraction": round(t.data_fraction, 3),
                 METRIC_NAME[self.task_]: t.score, "fit_time_s": round(t.fit_time, 2),
                 "params": t.params} for t in self.trials_]
        return pd.DataFrame(rows).sort_values(["data_fraction", METRIC_NAME[self.task_]],
                                              ascending=False).reset_index(drop=True)

    def describe(self) -> dict:
        return {"task": self.task_, "strategy": self.strategy,
                "metric": METRIC_NAME[self.task_], "cv_score": self.best_score_cv_,
                "n_trials": len(self.trials_), "search_time_s": round(self.search_time_, 1),
                "members": [{"family": t.family, "weight": round(w, 3), "params": t.params}
                            for _, w, t in self.members_]}

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "AutoML":
        return joblib.load(path)


def evaluate_holdout(model: AutoML, X_test, y_test) -> dict:
    """Метрики на отложенной выборке (не участвовавшей в поиске)."""
    if model.task_ == "regression":
        p = model.predict(X_test)
        return {"rmse": float(np.sqrt(mean_squared_error(y_test, p))),
                "r2": float(r2_score(y_test, p))}
    proba = model.predict_proba(X_test)
    y_enc = model.label_encoder_.transform(y_test)
    out = {"accuracy": float(accuracy_score(y_enc, proba.argmax(1))),
           "logloss": float(log_loss(y_enc, np.clip(proba, 1e-7, 1),
                                     labels=np.arange(proba.shape[1])))}
    out["roc_auc"] = float(roc_auc_score(y_enc, proba[:, 1]) if model.task_ == "binary"
                           else roc_auc_score(y_enc, proba, multi_class="ovr"))
    return out
