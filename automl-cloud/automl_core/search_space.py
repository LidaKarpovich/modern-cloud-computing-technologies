"""Пространство поиска мини-AutoML: семейства моделей и распределения гиперпараметров.

Каждая запись: имя -> (фабрика оценщика, функция сэмплирования гиперпараметров, флаги).
Сэмплирование делается через numpy.random.Generator, чтобы поиск был воспроизводимым.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                              HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR


def _loguniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


# ---------------------------------------------------------------- samplers
def s_linear(rng, task):
    if task == "regression":
        return {"alpha": _loguniform(rng, 1e-3, 1e3)}
    return {"C": _loguniform(rng, 1e-3, 1e3), "max_iter": 2000}


def s_rf(rng, task):
    return {
        "n_estimators": int(rng.choice([100, 200, 300])),
        "max_features": float(rng.uniform(0.2, 1.0)),
        "min_samples_leaf": int(rng.integers(1, 10)),
        "n_jobs": -1,
    }


def s_hgb(rng, task):
    return {
        "learning_rate": _loguniform(rng, 0.01, 0.3),
        "max_leaf_nodes": int(rng.choice([15, 31, 63, 127])),
        "min_samples_leaf": int(rng.integers(5, 60)),
        "l2_regularization": _loguniform(rng, 1e-6, 10.0),
        "max_iter": int(rng.choice([100, 200, 400])),
        "early_stopping": True,
    }


def s_knn(rng, task):
    return {"n_neighbors": int(rng.integers(3, 40)),
            "weights": str(rng.choice(["uniform", "distance"]))}


def s_svm(rng, task):
    p = {"C": _loguniform(rng, 1e-2, 1e3), "gamma": _loguniform(rng, 1e-4, 1.0)}
    if task != "regression":
        p["probability"] = True
    return p


def s_mlp(rng, task):
    width = int(rng.choice([32, 64, 128, 256]))
    depth = int(rng.integers(1, 3))
    return {"hidden_layer_sizes": tuple([width] * depth),
            "alpha": _loguniform(rng, 1e-6, 1e-1),
            "learning_rate_init": _loguniform(rng, 1e-4, 1e-2),
            "max_iter": 300, "early_stopping": True}


# name: (classifier, regressor, sampler, max_rows_for_model)
SPACE = {
    "linear": (LogisticRegression, Ridge, s_linear, None),
    "random_forest": (RandomForestClassifier, RandomForestRegressor, s_rf, None),
    "extra_trees": (ExtraTreesClassifier, ExtraTreesRegressor, s_rf, None),
    "hist_gbm": (HistGradientBoostingClassifier, HistGradientBoostingRegressor, s_hgb, None),
    "knn": (KNeighborsClassifier, KNeighborsRegressor, s_knn, 50_000),
    "svm": (SVC, SVR, s_svm, 5_000),          # O(n^2) — только на небольших данных
    "mlp": (MLPClassifier, MLPRegressor, s_mlp, None),
}

# Модели, которым нужна стандартизация признаков
NEEDS_SCALING = {"linear", "knn", "svm", "mlp"}


def make_estimator(family: str, params: dict, task: str, seed: int):
    clf, reg, _, _ = SPACE[family]
    cls = reg if task == "regression" else clf
    params = dict(params)
    if "random_state" in cls().get_params():
        params["random_state"] = seed
    return cls(**params)


def sample(family: str, rng, task: str) -> dict:
    return SPACE[family][2](rng, task)


def default_params(family: str, task: str) -> dict:
    """Параметры «по умолчанию» (как если бы человек взял модель из коробки)."""
    if family == "linear" and task != "regression":
        return {"max_iter": 2000}
    if family == "svm" and task != "regression":
        return {"probability": True}
    if family in {"random_forest", "extra_trees"}:
        return {"n_jobs": -1}
    return {}


def families_for(n_rows: int) -> list[str]:
    return [f for f, (_, _, _, lim) in SPACE.items() if lim is None or n_rows <= lim]
