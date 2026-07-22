"""
CC-9 follow-up: apply CX-18's exact accepted inner-OOF global threshold
tuning to CC-9's combined (band-power + PLV/PLI) feature arm, and compare
against CX-18's actual accepted best (kanboost_select80_inner_global_fine:
mean BA 0.7183, mean log loss 0.5757, mean ROC AUC 0.7812) -- the real
bar this project uses for accuracy decisions, not the fixed-0.5-threshold
number CC-9's first pass was mistakenly gated against.

Reuses the already-extracted feature table from the full CC-9 run
(remote/results/kaggle_cc9_openneuro_connectivity/cc9_openneuro_
connectivity_features_20260722-161726.csv) -- no raw EEG re-extraction
needed, so this runs in minutes, not the ~45 min the original extraction
took.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder

from kanboost import KANBoostClassifier

FEATURES_PATH = Path(__file__).resolve().parents[2] / "remote" / "results" / "kaggle_cc9_openneuro_connectivity" / "cc9_openneuro_connectivity_features_20260722-161726.csv"
OUT_DIR = Path(__file__).resolve().parents[2] / "remote" / "results" / "kaggle_cc9_openneuro_connectivity"
STAMP = time.strftime("%Y%m%d-%H%M%S", time.gmtime())

REGIONS = {
    "frontal": ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"],
    "central": ["C3", "C4", "Cz"],
    "temporal": ["T3", "T4", "T5", "T6"],
    "parietal": ["P3", "P4", "Pz"],
    "occipital": ["O1", "O2"],
}
CHANNELS = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"]
BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma_low"]
PAIRS = [("Fp1", "Fp2"), ("F3", "F4"), ("C3", "C4"), ("P3", "P4"), ("O1", "O2"), ("F7", "F8"), ("T3", "T4"), ("T5", "T6")]
SEEDS = [11, 22, 33, 44, 55]
THRESHOLDS = np.round(np.arange(0.35, 0.6001, 0.01), 3)
EXCLUDED_COLUMNS = {"participant_id", "target", "group", "feature_seconds", "n_channels", "duration_seconds"}


def add_eeg_ratio_features(df):
    out = df.copy()
    eps = 1e-6
    for band in BAND_NAMES:
        rel_cols = [f"{band}_rel_{ch}" for ch in CHANNELS if f"{band}_rel_{ch}" in out.columns]
        if rel_cols:
            out[f"{band}_rel_channel_mean"] = out[rel_cols].mean(axis=1)
            out[f"{band}_rel_channel_std"] = out[rel_cols].std(axis=1)
            out[f"{band}_rel_channel_maxmin"] = out[rel_cols].max(axis=1) - out[rel_cols].min(axis=1)
        for region, chans in REGIONS.items():
            cols = [f"{band}_rel_{ch}" for ch in chans if f"{band}_rel_{ch}" in out.columns]
            if cols:
                out[f"{band}_{region}_rel_mean"] = out[cols].mean(axis=1)
        for left, right in PAIRS:
            lcol, rcol = f"{band}_rel_{left}", f"{band}_rel_{right}"
            if lcol in out.columns and rcol in out.columns:
                out[f"{band}_asym_{left}_{right}"] = (out[lcol] - out[rcol]) / (out[lcol].abs() + out[rcol].abs() + eps)
    mean_rel = {b: out[f"{b}_mean_rel_power"].astype(float) for b in BAND_NAMES if f"{b}_mean_rel_power" in out.columns}
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


def kan_model(seed):
    return KANBoostClassifier(
        n_estimators=80, learning_rate=0.1, kan_hidden=3, kan_steps=6,
        gam=False, early_stopping_rounds=None, random_state=seed,
    )


def factory(seed, k):
    return make_pipeline(*clean_prefix(), SelectKBest(f_classif, k=k), kan_model(seed))


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


def inner_thresholds(X_train, y_train, seed, k):
    p_oof = np.zeros(len(y_train), dtype=float)
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed + 1000)
    for inner_tr, inner_va in inner.split(X_train, y_train):
        model = factory(seed, min(k, X_train.shape[1]))
        model.fit(X_train[inner_tr], y_train[inner_tr])
        p_oof[inner_va] = model.predict_proba(X_train[inner_va])[:, 1]
    return choose_threshold(y_train, p_oof)


def eval_outer_inner_threshold(model_name, X, y, k):
    rows = []
    for seed in SEEDS:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(cv.split(X, y), start=1):
            threshold = inner_thresholds(X[tr], y[tr], seed + fold, k)
            model = factory(seed, min(k, X.shape[1]))
            t0 = time.perf_counter()
            model.fit(X[tr], y[tr])
            fit_s = time.perf_counter() - t0
            p = model.predict_proba(X[va])[:, 1]
            pred = (p >= threshold).astype(int)
            rows.append({
                "model": model_name, "cv_seed": seed, "fold": fold, "threshold": threshold,
                "balanced_accuracy": balanced_accuracy_score(y[va], pred),
                "f1_macro": f1_score(y[va], pred, average="macro"),
                "log_loss": log_loss(y[va], np.column_stack([1 - p, p]), labels=[0, 1]),
                "roc_auc": roc_auc_score(y[va], p),
                "fit_seconds": fit_s,
            })
    return rows


def eval_hist_raw(X, y):
    rows = []
    for seed in SEEDS:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(cv.split(X, y), start=1):
            model = make_pipeline(*clean_prefix(), HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed))
            model.fit(X[tr], y[tr])
            p = model.predict_proba(X[va])[:, 1]
            pred = (p >= 0.5).astype(int)
            rows.append({
                "model": "hist_gbdt_raw_t0p5", "cv_seed": seed, "fold": fold, "threshold": 0.5,
                "balanced_accuracy": balanced_accuracy_score(y[va], pred),
                "f1_macro": f1_score(y[va], pred, average="macro"),
                "log_loss": log_loss(y[va], np.column_stack([1 - p, p]), labels=[0, 1]),
                "roc_auc": roc_auc_score(y[va], p),
                "fit_seconds": 0.0,
            })
    return rows


def main():
    features = pd.read_csv(FEATURES_PATH)
    y = LabelEncoder().fit_transform(features["target"].to_numpy())
    raw_cols = [c for c in features.columns if c not in EXCLUDED_COLUMNS]
    raw = features[raw_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)

    derived = add_eeg_ratio_features(features)
    connectivity_cols = [c for c in features.columns if "_plv_" in c or "_pli_" in c]
    bandpower_derived_cols = [c for c in derived.columns if c not in EXCLUDED_COLUMNS and c not in connectivity_cols]
    X_combined = derived[bandpower_derived_cols + connectivity_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    X_bandpower_only = derived[bandpower_derived_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)

    print({"n_subjects": len(y), "n_bandpower_derived": len(bandpower_derived_cols),
           "n_connectivity": len(connectivity_cols), "n_combined": X_combined.shape[1]})

    rows = []
    rows.extend(eval_hist_raw(raw, y))
    rows.extend(eval_outer_inner_threshold("kanboost_bandpower_inner_global", X_bandpower_only, y, 80))
    rows.extend(eval_outer_inner_threshold("kanboost_combined_inner_global", X_combined, y, 80))

    metrics = pd.DataFrame(rows)
    summary = (
        metrics.groupby("model")
        .agg(mean_balanced_accuracy=("balanced_accuracy", "mean"),
             std_balanced_accuracy=("balanced_accuracy", "std"),
             mean_log_loss=("log_loss", "mean"),
             mean_roc_auc=("roc_auc", "mean"),
             mean_threshold=("threshold", "mean"),
             mean_fit_seconds=("fit_seconds", "mean"),
             folds=("fold", "count"))
        .reset_index().sort_values(["mean_balanced_accuracy", "mean_log_loss"], ascending=[False, True])
    )
    seed_summary = (
        metrics.groupby(["model", "cv_seed"])
        .agg(mean_balanced_accuracy=("balanced_accuracy", "mean"), mean_log_loss=("log_loss", "mean"))
        .reset_index().sort_values(["cv_seed", "model"])
    )

    prefix = f"cc9b-openneuro-connectivity-threshold_{STAMP}"
    metrics.to_csv(OUT_DIR / f"{prefix}_metrics.csv", index=False)
    summary.to_csv(OUT_DIR / f"{prefix}_summary.csv", index=False)
    seed_summary.to_csv(OUT_DIR / f"{prefix}_seed_summary.csv", index=False)
    (OUT_DIR / f"{prefix}_results.json").write_text(json.dumps({
        "seeds": SEEDS, "n_bandpower_derived": len(bandpower_derived_cols), "n_connectivity": len(connectivity_cols),
        "summary": summary.to_dict(orient="records"), "seed_summary": seed_summary.to_dict(orient="records"),
        "cx18_accepted_baseline": {"mean_balanced_accuracy": 0.718262, "mean_log_loss": 0.575735, "mean_roc_auc": 0.781238},
    }, indent=2), encoding="utf-8")

    print("SUMMARY")
    print(summary.to_string(index=False))
    print("SEED SUMMARY")
    print(seed_summary.to_string(index=False))
    print("CX-18 accepted baseline for comparison: BA=0.7183 log_loss=0.5757 roc_auc=0.7812")


if __name__ == "__main__":
    main()
