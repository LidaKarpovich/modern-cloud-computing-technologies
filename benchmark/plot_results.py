"""Сводные таблицы и графики по результатам бенчмарка.

    python -m benchmark.plot_results --dir benchmark/results
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ORDER = ["baseline", "default", "random", "halving", "ensemble"]
LABEL = {"baseline": "Baseline (линейная)", "default": "Default (перебор алгоритмов)",
         "random": "Random search", "halving": "Successive halving", "ensemble": "Random + ансамбль"}
COLORS = {"baseline": "#9aa0a6", "default": "#7aa6c2", "random": "#4f7cac",
          "halving": "#d08c3c", "ensemble": "#2f8f5b"}


def primary(df):
    """Основная метрика: ROC-AUC (классификация, ↑) или RMSE (регрессия, ↓)."""
    df = df.copy()
    df["metric"] = df["roc_auc"].where(df["task"] != "regression", df["rmse"])
    df["metric_name"] = df["task"].map(lambda t: "RMSE ↓" if t == "regression" else "ROC-AUC ↑")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="benchmark/results")
    a = ap.parse_args()
    df = primary(pd.read_csv(os.path.join(a.dir, "benchmark.csv")))

    cols = ["metric", "accuracy", "logloss", "rmse", "r2", "n_trials", "wall_time_s"]
    summ = (df.groupby(["dataset", "metric_name", "strategy"])[cols]
              .agg(["mean", "std"]).round(4))
    summ.to_csv(os.path.join(a.dir, "summary.csv"))

    # широкая таблица: датасет x стратегия (mean ± std основной метрики)
    g = df.groupby(["dataset", "metric_name", "strategy"])["metric"]
    wide = (g.mean().round(4).astype(str) + " ± " + g.std().round(4).astype(str)).unstack()[ORDER]
    wide.to_csv(os.path.join(a.dir, "summary_wide.csv"))
    print(wide.to_string())

    # относительное улучшение над baseline (ошибка = 1-AUC или RMSE)
    base = df[df.strategy == "baseline"].groupby("dataset")["metric"].mean()
    err = df.assign(err=lambda d: d["metric"].where(d.task == "regression", 1 - d["metric"]))
    base_err = err[err.strategy == "baseline"].groupby("dataset")["err"].mean()
    rel = (err.groupby(["dataset", "strategy"])["err"].mean().unstack()[ORDER]
           .div(base_err, axis=0).sub(1).mul(100).round(1))
    rel.to_csv(os.path.join(a.dir, "relative_error_change_pct.csv"))
    print("\nИзменение ошибки относительно baseline, %:\n", rel.to_string())

    # --- рисунок 1: панели по датасетам
    # wine решается идеально всеми методами (ROC-AUC = 1.0) — на графике не информативен
    ds = [d for d in df.dataset.unique().tolist() if d != "wine"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    ax = axes.ravel()[-1]
    r = rel.drop(index="wine", errors="ignore")[["default", "random", "halving", "ensemble"]]
    yy = range(len(r))
    for k, s in enumerate(r.columns):
        ax.barh([y + (k - 1.5) * 0.2 for y in yy], r[s], height=0.2, color=COLORS[s], label=s)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(list(yy), r.index, fontsize=9)
    ax.set_title("Изменение ошибки к baseline, % (меньше — лучше)", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    for ax, d in zip(axes.ravel()[:-1], ds):
        sub = df[df.dataset == d]
        m = sub.groupby("strategy")["metric"].agg(["mean", "std"]).reindex(ORDER)
        ax.bar(range(len(ORDER)), m["mean"], yerr=m["std"], capsize=3,
               color=[COLORS[s] for s in ORDER])
        lo, hi = (m["mean"] - m["std"].fillna(0)).min(), (m["mean"] + m["std"].fillna(0)).max()
        pad = (hi - lo) * 0.6 + 1e-4
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(f"{d}  ({sub.metric_name.iloc[0]})", fontsize=11)
        ax.set_xticks(range(len(ORDER)), ["base", "default", "random", "halving", "ens"],
                      fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("Качество на отложенной выборке: mean ± std по 3 сидам, бюджет 30 c", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(a.dir, "fig_benchmark.png"), dpi=150)

    # --- рисунок 2: кривая бюджет -> качество
    cpath = os.path.join(a.dir, "curve_credit_synth.csv")
    if os.path.exists(cpath):
        c = pd.read_csv(cpath)
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for s in ["random", "halving", "ensemble"]:
            sub = c[c.strategy == s].groupby("budget_s")["roc_auc"].mean()
            ax.plot(sub.index, sub.values, marker="o", label=LABEL[s], color=COLORS[s])
        b = df[(df.dataset == "credit_synth") & (df.strategy == "baseline")]["roc_auc"].mean()
        ax.axhline(b, ls="--", color=COLORS["baseline"], label=LABEL["baseline"])
        ax.set_xscale("log")
        ax.set_xticks(sorted(c.budget_s.unique()), [f"{int(x)}" for x in sorted(c.budget_s.unique())])
        ax.set_xlabel("Бюджет времени поиска, с (2 vCPU)")
        ax.set_ylabel("ROC-AUC на отложенной выборке")
        ax.set_title("credit_synth (20 000 строк): бюджет → качество")
        ax.grid(alpha=0.3); ax.legend(fontsize=9)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(a.dir, "fig_budget_curve.png"), dpi=150)
        c.groupby(["budget_s", "strategy"])[["roc_auc", "n_trials"]].mean().round(4) \
         .to_csv(os.path.join(a.dir, "curve_summary.csv"))
        print("\n", c.groupby(["budget_s", "strategy"])[["roc_auc", "n_trials"]].mean().round(4))


if __name__ == "__main__":
    main()
