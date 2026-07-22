"""
CX-19 (Codex proposal): split-robust OpenNeuro KANBoost via fold-local
rank-aggregation feature selection, paired with CX-18's accepted
inner-OOF global threshold. Targets the worst-case-split weakness CX-18
left open (cv_seed=44 -> 0.670 BA vs HistGBDT's 0.701 on that split).

Reuses CX-18's exact scaffold (add_eeg_features, clean_prefix, kan_model,
inner_thresholds, eval_outer, 5 seeds x 5 outer StratifiedKFold folds) so
results are directly comparable under CLAUDE.md rule 10 (same dataset,
folds, seeds, model config). Only the feature selector changes: plain
SelectKBest(f_classif, k=80) -> RankAggregationSelector(k=80), which
bootstrap-resamples the fold-local training rows, ranks features by three
independent leakage-safe criteria per resample (ANOVA F, mutual
information, |logistic coefficient| after scaling), and averages ranks
before taking the top-k. All resampling/ranking happens strictly inside
whatever training rows the pipeline is given (outer train fold, or inner
train fold during threshold tuning) -- no different fit-boundary than the
SelectKBest baseline already had.

Simplification vs Codex's original proposal text: does not include
"KANBoost single-feature gain" as a 4th ranking criterion (would require
one KANBoost fit per feature per resample -- too expensive for this
round's budget on top of the model that's already being fit). Documented
here rather than silently dropped, matching this project's standing
practice for scoped-down implementations (see CC-6b's note on skipping
literal weight-transplant).
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # E:/project/kanboost, use working tree kanboost

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import (
    SelectKBest, VarianceThreshold, f_classif, mutual_info_classif,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import check_random_state

import kanboost
from kanboost import KANBoostClassifier


PLATFORM = "kaggle" if Path("/kaggle/working").exists() else "local"
OUT_DIR = Path("/kaggle/working" if PLATFORM == "kaggle" else "remote/results/kaggle_cx19_openneuro_rank_aggregation")
OUT_DIR.mkdir(parents=True, exist_ok=True)
STAMP = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
DATASET_NAME = "ds004504_large_clean_features_eeg_rank_aggregation"
PREFIX = f"cx19-openneuro-rank-aggregation_{DATASET_NAME}_{STAMP}"
FEATURE_FILE = Path(__file__).with_name("openneuro_large_clean_features.csv")
if not FEATURE_FILE.exists():
    matches = sorted(Path("/kaggle/input").glob("**/openneuro_large_clean_features.csv"))
    if matches:
        FEATURE_FILE = matches[0]
if not FEATURE_FILE.exists():
    raise FileNotFoundError("openneuro_large_clean_features.csv not found")

EXCLUDED_COLUMNS = {"participant_id", "target", "group", "feature_seconds"}
BANDS = ["delta", "theta", "alpha", "beta", "gamma_low"]
CHANNELS = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"]
REGIONS = {
    "frontal": ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"],
    "central": ["C3", "C4", "Cz"],
    "temporal": ["T3", "T4", "T5", "T6"],
    "parietal": ["P3", "P4", "Pz"],
    "occipital": ["O1", "O2"],
}
PAIRS = [("Fp1", "Fp2"), ("F3", "F4"), ("C3", "C4"), ("P3", "P4"), ("O1", "O2"), ("F7", "F8"), ("T3", "T4"), ("T5", "T6")]
SEEDS = [11, 22, 33, 44, 55]
THRESHOLDS = np.round(np.arange(0.35, 0.6001, 0.01), 3)


def add_eeg_features(df):
    out = df.copy()
    eps = 1e-6
    for band in BANDS:
        rel_cols = [f"{band}_rel_{ch}" for ch in CHANNELS if f"{band}_rel_{ch}" in out.columns]
        if rel_cols:
            out[f"{band}_rel_channel_mean"] = out[rel_cols].mean(axis=1)
            out[f"{band}_rel_channel_std"] = out[rel_cols].std(axis=1)
            out[f"{band}_rel_channel_maxmin"] = out[rel_cols].max(axis=1) - out[rel_cols].min(axis=1)
        for region, channels in REGIONS.items():
            cols = [f"{band}_rel_{ch}" for ch in channels if f"{band}_rel_{ch}" in out.columns]
            if cols:
                out[f"{band}_{region}_rel_mean"] = out[cols].mean(axis=1)
        for left, right in PAIRS:
            lcol = f"{band}_rel_{left}"
            rcol = f"{band}_rel_{right}"
            if lcol in out.columns and rcol in out.columns:
                out[f"{band}_asym_{left}_{right}"] = (out[lcol] - out[rcol]) / (out[lcol].abs() + out[rcol].abs() + eps)
    mean_rel = {band: out[f"{band}_mean_rel_power"].astype(float) for band in BANDS if f"{band}_mean_rel_power" in out.columns}
    if {"delta", "theta", "alpha", "beta"}.issubset(mean_rel):
        out["delta_alpha_ratio_derived"] = mean_rel["delta"] / (mean_rel["alpha"] + eps)
        out["theta_alpha_ratio_derived"] = mean_rel["theta"] / (mean_rel["alpha"] + eps)
        out["beta_alpha_ratio_derived"] = mean_rel["beta"] / (mean_rel["alpha"] + eps)
        out["slow_fast_ratio_derived"] = (mean_rel["delta"] + mean_rel["theta"]) / (mean_rel["beta"] + mean_rel.get("gamma_low", 0.0) + eps)
        out["alpha_slow_ratio_derived"] = mean_rel["alpha"] / (mean_rel["delta"] + mean_rel["theta"] + eps)
        matrix = np.column_stack([mean_rel[b].to_numpy(dtype=float) for b in mean_rel])
        matrix = np.clip(matrix, 0.0, None)
        probs = matrix / (matrix.sum(axis=1, keepdims=True) + eps)
        out["band_entropy"] = -(probs * np.log(probs + eps)).sum(axis=1)
    return out


def clean_prefix():
    return [SimpleImputer(strategy="median"), VarianceThreshold(threshold=0.0)]


class RankAggregationSelector(BaseEstimator, TransformerMixin):
    """Fold-local, bootstrap rank-aggregation feature selector (CX-19).

    Bootstrap-resamples the rows given to `fit` (never anything outside
    that call's data -- caller controls the leakage boundary exactly like
    SelectKBest would), ranks features per resample by three criteria,
    averages ranks, keeps the top `k`.
    """

    def __init__(self, k=80, n_resamples=15, random_state=0):
        self.k = k
        self.n_resamples = n_resamples
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        rng = check_random_state(self.random_state)
        n, p = X.shape
        rank_sum = np.zeros(p, dtype=float)
        n_ranks = 0

        for _ in range(self.n_resamples):
            for _retry in range(5):
                idx = rng.randint(0, n, size=n)
                yb = y[idx]
                if len(np.unique(yb)) >= 2:
                    break
            else:
                continue
            Xb = X[idx]

            f_scores, _ = f_classif(Xb, yb)
            f_scores = np.nan_to_num(f_scores, nan=-np.inf)
            rank_sum += _to_rank(-f_scores)
            n_ranks += 1

            mi = mutual_info_classif(Xb, yb, random_state=rng.randint(0, 2**31 - 1))
            rank_sum += _to_rank(-mi)
            n_ranks += 1

            Xb_scaled = StandardScaler().fit_transform(Xb)
            clf = LogisticRegression(penalty="l2", C=1.0, max_iter=1000)
            clf.fit(Xb_scaled, yb)
            coef = np.abs(clf.coef_).ravel()
            rank_sum += _to_rank(-coef)
            n_ranks += 1

        avg_rank = rank_sum / max(n_ranks, 1)
        self.selected_idx_ = np.argsort(avg_rank)[: self.k]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, self.selected_idx_]


def _to_rank(values):
    """Lower value -> rank 0 (best)."""
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values))
    return ranks


def kan_model(seed):
    return KANBoostClassifier(
        n_estimators=80,
        learning_rate=0.1,
        kan_hidden=3,
        kan_steps=6,
        gam=False,
        early_stopping_rounds=None,
        random_state=seed,
    )


def select_factory_baseline(seed):
    return make_pipeline(*clean_prefix(), SelectKBest(f_classif, k=80), kan_model(seed))


def select_factory_rank_agg(seed):
    return make_pipeline(*clean_prefix(), RankAggregationSelector(k=80, n_resamples=15, random_state=seed), kan_model(seed))


def threshold_score(y_true, p, threshold):
    pred = (p >= threshold).astype(int)
    return balanced_accuracy_score(y_true, pred), f1_score(y_true, pred, average="macro")


def choose_threshold(y_true, p):
    best = None
    for threshold in THRESHOLDS:
        ba, f1 = threshold_score(y_true, p, threshold)
        key = (ba, f1, -abs(float(threshold) - 0.5))
        if best is None or key > best[0]:
            best = (key, float(threshold))
    return best[1]


def inner_thresholds(X_train, y_train, seed, factory):
    p_oof = np.zeros(len(y_train), dtype=float)
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed + 1000)
    for inner_tr, inner_va in inner.split(X_train, y_train):
        model = factory(seed)
        model.fit(X_train[inner_tr], y_train[inner_tr])
        p_inner = model.predict_proba(X_train[inner_va])[:, 1]
        p_oof[inner_va] = p_inner
    return choose_threshold(y_train, p_oof)


def eval_outer(model_name, X, y, factory, use_inner_threshold):
    rows = []
    for seed in SEEDS:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(cv.split(X, y), start=1):
            threshold = 0.5
            threshold_s = 0.0
            if use_inner_threshold:
                t_thr = time.perf_counter()
                threshold = inner_thresholds(X[tr], y[tr], seed + fold, factory)
                threshold_s = time.perf_counter() - t_thr
            model = factory(seed)
            t0 = time.perf_counter()
            model.fit(X[tr], y[tr])
            fit_s = time.perf_counter() - t0
            t1 = time.perf_counter()
            p = model.predict_proba(X[va])[:, 1]
            pred_s = time.perf_counter() - t1
            pred = (p >= threshold).astype(int)
            rows.append({
                "platform": PLATFORM,
                "timestamp_utc": STAMP,
                "dataset": DATASET_NAME,
                "model": model_name,
                "cv_seed": seed,
                "fold": fold,
                "threshold": threshold,
                "threshold_seconds": threshold_s,
                "balanced_accuracy": balanced_accuracy_score(y[va], pred),
                "f1_macro": f1_score(y[va], pred, average="macro"),
                "log_loss": log_loss(y[va], np.column_stack([1.0 - p, p]), labels=[0, 1]),
                "roc_auc": roc_auc_score(y[va], p),
                "fit_seconds": fit_s,
                "predict_seconds": pred_s,
            })
    return rows


def eval_hist_raw(X, y):
    rows = []
    for seed in SEEDS:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(cv.split(X, y), start=1):
            model = make_pipeline(*clean_prefix(), HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed))
            t0 = time.perf_counter()
            model.fit(X[tr], y[tr])
            fit_s = time.perf_counter() - t0
            t1 = time.perf_counter()
            p = model.predict_proba(X[va])[:, 1]
            pred_s = time.perf_counter() - t1
            pred = (p >= 0.5).astype(int)
            rows.append({
                "platform": PLATFORM,
                "timestamp_utc": STAMP,
                "dataset": DATASET_NAME,
                "model": "hist_gbdt_raw_t0p5",
                "cv_seed": seed,
                "fold": fold,
                "threshold": 0.5,
                "threshold_seconds": 0.0,
                "balanced_accuracy": balanced_accuracy_score(y[va], pred),
                "f1_macro": f1_score(y[va], pred, average="macro"),
                "log_loss": log_loss(y[va], np.column_stack([1.0 - p, p]), labels=[0, 1]),
                "roc_auc": roc_auc_score(y[va], p),
                "fit_seconds": fit_s,
                "predict_seconds": pred_s,
            })
    return rows


def main():
    features = pd.read_csv(FEATURE_FILE)
    keep = np.isin(features["target"].to_numpy(), ["AD", "Control"])
    features = features.loc[keep].reset_index(drop=True)
    y = LabelEncoder().fit_transform(features["target"].to_numpy())
    derived = add_eeg_features(features)
    derived_cols = [c for c in derived.columns if c not in EXCLUDED_COLUMNS]
    X_derived = derived[derived_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    raw_cols = [c for c in features.columns if c not in EXCLUDED_COLUMNS]
    raw = features[raw_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)

    rows = []
    rows.extend(eval_hist_raw(raw, y))
    rows.extend(eval_outer("kanboost_select80_baseline_t0p5", X_derived, y, select_factory_baseline, use_inner_threshold=False))
    rows.extend(eval_outer("kanboost_select80_baseline_inner_global", X_derived, y, select_factory_baseline, use_inner_threshold=True))
    rows.extend(eval_outer("kanboost_rankagg80_t0p5", X_derived, y, select_factory_rank_agg, use_inner_threshold=False))
    rows.extend(eval_outer("kanboost_rankagg80_inner_global", X_derived, y, select_factory_rank_agg, use_inner_threshold=True))

    metrics = pd.DataFrame(rows)
    summary = (
        metrics.groupby("model")
        .agg(
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            std_balanced_accuracy=("balanced_accuracy", "std"),
            worst_seed_ba=("balanced_accuracy", "min"),
            mean_f1_macro=("f1_macro", "mean"),
            mean_log_loss=("log_loss", "mean"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_fit_seconds=("fit_seconds", "mean"),
            folds=("fold", "count"),
        )
        .reset_index()
        .sort_values(["mean_balanced_accuracy", "mean_log_loss"], ascending=[False, True])
    )
    seed_summary = (
        metrics.groupby(["model", "cv_seed"])
        .agg(
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_log_loss=("log_loss", "mean"),
            mean_roc_auc=("roc_auc", "mean"),
        )
        .reset_index()
        .sort_values(["cv_seed", "model"])
    )

    metrics_path = OUT_DIR / f"{PREFIX}_metrics.csv"
    summary_path = OUT_DIR / f"{PREFIX}_summary.csv"
    seed_summary_path = OUT_DIR / f"{PREFIX}_seed_summary.csv"
    results_path = OUT_DIR / f"{PREFIX}_results.json"
    env_path = OUT_DIR / f"{PREFIX}_environment.txt"
    metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    seed_summary.to_csv(seed_summary_path, index=False)
    results_path.write_text(
        json.dumps(
            {
                "platform": PLATFORM,
                "kanboost_version": kanboost.__version__,
                "kanboost_source": "working tree (research/als-solve-perf, includes CX-13/CX-20)",
                "dataset": DATASET_NAME,
                "feature_file": str(FEATURE_FILE),
                "seeds": SEEDS,
                "summary": summary.to_dict(orient="records"),
                "seed_summary": seed_summary.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    env_path.write_text(
        "\n".join([
            f"platform={platform.platform()}",
            f"python={platform.python_version()}",
            f"numpy={np.__version__}",
            f"kanboost_version={kanboost.__version__}",
            f"pid={os.getpid()}",
        ]),
        encoding="utf-8",
    )
    print("SUMMARY")
    print(summary.to_string(index=False))
    print("SEED SUMMARY")
    print(seed_summary.to_string(index=False))
    print("wrote", metrics_path)


if __name__ == "__main__":
    main()
