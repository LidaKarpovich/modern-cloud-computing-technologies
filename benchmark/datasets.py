"""Наборы данных для бенчмарка. Все доступны офлайн (встроены в scikit-learn или синтетические),
поэтому эксперимент воспроизводим без доступа к интернету."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import datasets as skd


def _frame(bunch):
    X = pd.DataFrame(bunch.data, columns=getattr(bunch, "feature_names", None))
    X.columns = [str(c) for c in X.columns]
    return X, pd.Series(bunch.target)


def breast_cancer():
    return _frame(skd.load_breast_cancer())


def digits():
    return _frame(skd.load_digits())


def wine():
    return _frame(skd.load_wine())


def diabetes():
    X, y = _frame(skd.load_diabetes())
    return X, y.astype(float)


def synthetic_credit(n=20_000, seed=0):
    """«Кредитный скоринг»: смешанные типы, категориальные признаки, пропуски, дисбаланс 85/15."""
    rng = np.random.default_rng(seed)
    Xn, y = skd.make_classification(n_samples=n, n_features=20, n_informative=8,
                                    n_redundant=4, weights=[0.85], flip_y=0.03,
                                    class_sep=0.8, random_state=seed)
    X = pd.DataFrame(Xn, columns=[f"num_{i}" for i in range(20)])
    for i, c in enumerate(["region", "channel", "product"]):
        bins = [4, 6, 12][i]
        X[c] = pd.qcut(X[f"num_{i}"] + rng.normal(0, 0.5, n), bins,
                       labels=[f"{c}_{k}" for k in range(bins)]).astype(str)
    X = X.drop(columns=["num_0", "num_1", "num_2"])
    mask = rng.random(X.shape) < 0.05                      # 5 % пропусков
    X = X.mask(mask)
    return X, pd.Series(y)


def synthetic_friedman(n=5_000, seed=0):
    """Нелинейная регрессия Friedman #1 (10 признаков, 5 информативных)."""
    X, y = skd.make_friedman1(n_samples=n, n_features=10, noise=1.0, random_state=seed)
    return pd.DataFrame(X, columns=[f"x{i}" for i in range(10)]), pd.Series(y)


DATASETS = {
    "breast_cancer": breast_cancer,      # 569 x 30, binary
    "wine": wine,                        # 178 x 13, 3 класса
    "digits": digits,                    # 1797 x 64, 10 классов
    "diabetes": diabetes,                # 442 x 10, регрессия
    "credit_synth": synthetic_credit,    # 20000 x 20, binary, cat + NaN
    "friedman_synth": synthetic_friedman,  # 5000 x 10, регрессия
}
