"""
CC-11 bugfix rerun: the first pass gave HistGBDT the full 196-feature
derived pool with NO feature selection, while KANBoost used
SelectKBest(k=80) internally -- an unfair, inconsistent comparison that
likely explains HistGBDT's implausible sub-chance result (BA 0.486, log
loss 1.228, both worse than a naive base-rate predictor -- a classic
overfitting-on-too-many-features-too-few-rows signature: 196 features vs
~138 training rows per fold).

Fix: both models now get the identical SelectKBest(f_classif, k=80)
selection, fit fold-locally inside the same pipeline structure. Reuses
the already-extracted feature table -- no raw EEG reprocessing needed.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder

from kanboost import KANBoostClassifier

FEATURES_PATH = Path(__file__).resolve().parents[2] / "remote" / "results" / "kaggle_cc11_covid_eeg" / "cc11_covid_eeg_features_20260722-201317.csv"
OUT_DIR = Path(__file__).resolve().parents[2] / "remote" / "results" / "kaggle_cc11_covid_eeg"
STAMP = time.strftime("%Y%m%d-%H%M%S", time.gmtime())

BANDS = {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0), "gamma_low": (30.0, 45.0)}
CHANNELS = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
            "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"]
REGIONS = {
    "frontal": ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"],
    "central": ["C3", "C4", "Cz"],
    "temporal": ["T3", "T4", "T5", "T6"],
    "parietal": ["P3", "P4", "Pz"],
    "occipital": ["O1", "O2"],
}
PAIRS = [("Fp1", "Fp2"), ("F3", "F4"), ("C3", "C4"), ("P3", "P4"), ("O1", "O2"), ("F7", "F8"), ("T3", "T4"), ("T5", "T6")]
SEEDS = [11, 22, 33, 44, 55]
EXCLUDED_COLUMNS = {"participant_id", "target", "n_channels", "duration_seconds", "feature_seconds"}


def add_eeg_ratio_features(df):
    out = df.copy()
    eps = 1e-6
    band_names = list(BANDS)
    for band in band_names:
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
    mean_rel = {b: out[f"{b}_mean_rel_power"].astype(float) for b in band_names if f"{b}_mean_rel_power" in out.columns}
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


def eval_model(model_name, X, y, estimator_factory, k=80):
    rows = []
    for seed in SEEDS:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(cv.split(X, y), start=1):
            model = make_pipeline(*clean_prefix(), SelectKBest(f_classif, k=min(k, X.shape[1])), estimator_factory(seed))
            t0 = time.perf_counter()
            model.fit(X[tr], y[tr])
            fit_s = time.perf_counter() - t0
            p = model.predict_proba(X[va])[:, 1]
            pred = (p >= 0.5).astype(int)
            rows.append({
                "model": model_name, "cv_seed": seed, "fold": fold,
                "balanced_accuracy": balanced_accuracy_score(y[va], pred),
                "f1_macro": f1_score(y[va], pred, average="macro"),
                "log_loss": log_loss(y[va], np.column_stack([1 - p, p]), labels=[0, 1]),
                "roc_auc": roc_auc_score(y[va], p),
                "fit_seconds": fit_s,
            })
    return rows


def main():
    features = pd.read_csv(FEATURES_PATH)
    y = LabelEncoder().fit_transform(features["target"].to_numpy())
    derived = add_eeg_ratio_features(features)
    feat_cols = [c for c in derived.columns if c not in EXCLUDED_COLUMNS]
    X = derived[feat_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    print({"n_subjects": len(y), "n_features": len(feat_cols), "positive_rate": float(y.mean())})

    rows = []
    rows.extend(eval_model("hist_gbdt_select80", X, y,
                            lambda seed: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed)))
    rows.extend(eval_model("kanboost_e80_h3_s6_select80", X, y,
                            lambda seed: KANBoostClassifier(n_estimators=80, kan_hidden=3, kan_steps=6,
                                                             gam=False, early_stopping_rounds=None, random_state=seed)))

    metrics = pd.DataFrame(rows)
    summary = (metrics.groupby("model")
        .agg(mean_balanced_accuracy=("balanced_accuracy", "mean"), std_balanced_accuracy=("balanced_accuracy", "std"),
             worst_seed_ba=("balanced_accuracy", "min"), mean_log_loss=("log_loss", "mean"),
             mean_roc_auc=("roc_auc", "mean"), mean_fit_seconds=("fit_seconds", "mean"), folds=("fold", "count"))
        .reset_index().sort_values(["mean_balanced_accuracy", "mean_log_loss"], ascending=[False, True]))
    seed_summary = (metrics.groupby(["model", "cv_seed"])
        .agg(mean_balanced_accuracy=("balanced_accuracy", "mean"), mean_log_loss=("log_loss", "mean"))
        .reset_index().sort_values(["cv_seed", "model"]))

    prefix = f"cc11b-covid-eeg-fixed_{STAMP}"
    metrics.to_csv(OUT_DIR / f"{prefix}_metrics.csv", index=False)
    summary.to_csv(OUT_DIR / f"{prefix}_summary.csv", index=False)
    seed_summary.to_csv(OUT_DIR / f"{prefix}_seed_summary.csv", index=False)
    (OUT_DIR / f"{prefix}_results.json").write_text(json.dumps({
        "seeds": SEEDS, "n": len(y), "n_features": len(feat_cols),
        "summary": summary.to_dict(orient="records"), "seed_summary": seed_summary.to_dict(orient="records"),
    }, indent=2), encoding="utf-8")

    print("SUMMARY")
    print(summary.to_string(index=False))
    print("SEED SUMMARY")
    print(seed_summary.to_string(index=False))


if __name__ == "__main__":
    main()
