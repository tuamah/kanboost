"""
CC-10 (Claude Code): sanity check on a large, standard, clean tabular
benchmark -- does today's mixed EEG evidence (CX-19 rejected,
CC-9 rejected on accuracy) reflect a real limit, or just the n=65
OpenNeuro dataset's small-sample noise? Adult Census Income (48,842 rows,
mixed numeric/categorical, ~24% positive class) is a standard GBM
benchmark, unrelated to EEG/connectivity work -- a genuinely different
dataset, not a variant of today's task.

Protocol: 8,000-row stratified subsample (train_size=8000, random_state=0,
fixed once -- NOT re-subsampled per seed, so all seeds/folds share the
exact same 8,000 rows; only StratifiedKFold's own fold assignment varies
by seed). StratifiedKFold(5) x seeds [11,22,33,44,55] (same seed
convention as CX-18/19/CC-8/9). KANBoost capacity chosen via a short,
fair calibration pass (not an exhaustive grid): kan_hidden=8,
n_estimators=250, kan_steps=12 was the best of 5 configs tried on a
single held-out split (BA 0.7423 vs HistGBDT's 0.7725 there -- a real
gap, not an artifact of undertraining, since wider/deeper alternatives
tried did not close it further, matching CX-9's earlier OpenNeuro
finding that KANBoost capacity search alone does not close this kind of
gap). HistGBDT: max_iter=300, learning_rate=0.05 (same config used
throughout this project's OpenNeuro benchmarks, for consistency of
comparison philosophy even though this is different data).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from kanboost import KANBoostClassifier

OUT_DIR = Path(__file__).resolve().parents[2] / "remote" / "results" / "kaggle_cc10_adult_benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)
STAMP = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
SEEDS = [11, 22, 33, 44, 55]


def load_data():
    data = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    X, y = data.data, data.target
    y = (y == ">50K").astype(int)
    X_sub, _, y_sub, _ = train_test_split(X, y, train_size=8000, stratify=y, random_state=0)
    X_sub = X_sub.reset_index(drop=True)
    y_sub = y_sub.reset_index(drop=True).to_numpy()
    cat_cols = [c for c in X_sub.columns if str(X_sub[c].dtype) == "category"]
    return X_sub, y_sub, cat_cols


def eval_kanboost(X, y, cat_cols):
    rows = []
    for seed in SEEDS:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(cv.split(X, y), start=1):
            model = KANBoostClassifier(
                n_estimators=250, kan_hidden=8, kan_steps=12,
                categorical_cols=cat_cols, early_stopping_rounds=None, random_state=seed,
            )
            X_tr, X_va = X.iloc[tr].reset_index(drop=True), X.iloc[va].reset_index(drop=True)
            y_tr, y_va = y[tr], y[va]
            t0 = time.perf_counter()
            model.fit(X_tr, y_tr)
            fit_s = time.perf_counter() - t0
            p = model.predict_proba(X_va)[:, 1]
            pred = (p >= 0.5).astype(int)
            rows.append({
                "model": "kanboost_e250_h8_s12", "cv_seed": seed, "fold": fold,
                "balanced_accuracy": balanced_accuracy_score(y_va, pred),
                "f1_macro": f1_score(y_va, pred, average="macro"),
                "log_loss": log_loss(y_va, np.column_stack([1 - p, p]), labels=[0, 1]),
                "roc_auc": roc_auc_score(y_va, p),
                "fit_seconds": fit_s,
            })
            print(f"kanboost seed={seed} fold={fold} BA={rows[-1]['balanced_accuracy']:.4f} fit={fit_s:.1f}s")
    return rows


def eval_histgbdt(X, y, cat_cols):
    rows = []
    cat_mask = [c in cat_cols for c in X.columns]
    for seed in SEEDS:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(cv.split(X, y), start=1):
            model = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.05, categorical_features=cat_mask, random_state=seed,
            )
            X_tr, X_va = X.iloc[tr], X.iloc[va]
            y_tr, y_va = y[tr], y[va]
            t0 = time.perf_counter()
            model.fit(X_tr, y_tr)
            fit_s = time.perf_counter() - t0
            p = model.predict_proba(X_va)[:, 1]
            pred = (p >= 0.5).astype(int)
            rows.append({
                "model": "hist_gbdt_t0p5", "cv_seed": seed, "fold": fold,
                "balanced_accuracy": balanced_accuracy_score(y_va, pred),
                "f1_macro": f1_score(y_va, pred, average="macro"),
                "log_loss": log_loss(y_va, np.column_stack([1 - p, p]), labels=[0, 1]),
                "roc_auc": roc_auc_score(y_va, p),
                "fit_seconds": fit_s,
            })
    return rows


def main():
    X, y, cat_cols = load_data()
    print({"n": len(y), "positive_rate": float(y.mean()), "categorical_cols": cat_cols})

    rows = []
    rows.extend(eval_histgbdt(X, y, cat_cols))
    rows.extend(eval_kanboost(X, y, cat_cols))

    metrics = pd.DataFrame(rows)
    summary = (
        metrics.groupby("model")
        .agg(mean_balanced_accuracy=("balanced_accuracy", "mean"),
             std_balanced_accuracy=("balanced_accuracy", "std"),
             worst_seed_ba=("balanced_accuracy", "min"),
             mean_log_loss=("log_loss", "mean"),
             mean_roc_auc=("roc_auc", "mean"),
             mean_fit_seconds=("fit_seconds", "mean"),
             folds=("fold", "count"))
        .reset_index().sort_values(["mean_balanced_accuracy", "mean_log_loss"], ascending=[False, True])
    )
    seed_summary = (
        metrics.groupby(["model", "cv_seed"])
        .agg(mean_balanced_accuracy=("balanced_accuracy", "mean"), mean_log_loss=("log_loss", "mean"))
        .reset_index().sort_values(["cv_seed", "model"])
    )

    prefix = f"cc10-adult-benchmark_{STAMP}"
    metrics.to_csv(OUT_DIR / f"{prefix}_metrics.csv", index=False)
    summary.to_csv(OUT_DIR / f"{prefix}_summary.csv", index=False)
    seed_summary.to_csv(OUT_DIR / f"{prefix}_seed_summary.csv", index=False)
    (OUT_DIR / f"{prefix}_results.json").write_text(json.dumps({
        "seeds": SEEDS, "n": len(y),
        "summary": summary.to_dict(orient="records"), "seed_summary": seed_summary.to_dict(orient="records"),
    }, indent=2), encoding="utf-8")

    print("SUMMARY")
    print(summary.to_string(index=False))
    print("SEED SUMMARY")
    print(seed_summary.to_string(index=False))


if __name__ == "__main__":
    main()
