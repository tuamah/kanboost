"""
CC-8 (Claude Code, literature-motivated): leave-one-subject-out (LOSO)
validation on the same OpenNeuro ds004504 large-clean-features data used
by CX-6..CX-19.

Motivation: recent published work on this exact dataset (e.g. the AD/FTD
functional-connectivity ensemble study, and small-EEG-sample deep-learning
robustness studies from 2025-2026) consistently uses LOSO, not repeated
k-fold, as the validation standard for n~65-88-subject EEG classification
-- specifically because k-fold's *fold composition* is itself a source of
variance at this sample size, which is exactly the failure mode CX-19
targets (cv_seed=44 dropping to 0.670 BA). LOSO removes fold-composition
variance entirely (every subject is held out exactly once, deterministically)
so it isolates whether CX-18/19's instability is a fold-composition
artifact or a real model weakness -- and is worth reporting as the
literature-comparable number regardless.

This does NOT touch kanboost/core -- pure benchmark/evaluation script
using the working tree's KANBoostClassifier (includes CX-13/CX-20;
prediction-only speed change, exact parity, does not affect these
results).

Not a replacement for CX-18/19's k-fold numbers (different protocol,
rule 10 applies -- not directly comparable), but a complementary,
literature-grounded robustness check on the same data.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder

import kanboost
from kanboost import KANBoostClassifier

OUT_DIR = Path("remote/results/kaggle_cx19_openneuro_rank_aggregation")
OUT_DIR.mkdir(parents=True, exist_ok=True)
STAMP = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
FEATURE_FILE = Path(__file__).with_name("openneuro_large_clean_features.csv")

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
SEEDS = [11, 22, 33]


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


def kan_model(seed):
    return KANBoostClassifier(
        n_estimators=80, learning_rate=0.1, kan_hidden=3, kan_steps=6,
        gam=False, early_stopping_rounds=None, random_state=seed,
    )


def kan_factory(seed):
    return make_pipeline(*clean_prefix(), SelectKBest(f_classif, k=80), kan_model(seed))


def hist_factory(seed):
    return make_pipeline(*clean_prefix(), HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed))


def loso_eval(model_name, X, y, factory, seed):
    loo = LeaveOneOut()
    p = np.zeros(len(y))
    t_fit = 0.0
    for tr, va in loo.split(X):
        model = factory(seed)
        t0 = time.perf_counter()
        model.fit(X[tr], y[tr])
        t_fit += time.perf_counter() - t0
        p[va] = model.predict_proba(X[va])[:, 1]
    pred = (p >= 0.5).astype(int)
    return {
        "model": model_name,
        "seed": seed,
        "n": len(y),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "f1_macro": f1_score(y, pred, average="macro"),
        "log_loss": log_loss(y, np.column_stack([1 - p, p]), labels=[0, 1]),
        "roc_auc": roc_auc_score(y, p),
        "total_fit_seconds": t_fit,
    }


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
    print(f"n={len(y)} (AD={int((y==0).sum() if False else (features['target']=='AD').sum())}, Control={(features['target']=='Control').sum()})")
    for seed in SEEDS:
        print(f"HistGBDT LOSO seed={seed} ...")
        rows.append(loso_eval("hist_gbdt_raw_loso", raw, y, hist_factory, seed))
        print(f"KANBoost select80 LOSO seed={seed} ...")
        rows.append(loso_eval("kanboost_select80_loso", X_derived, y, kan_factory, seed))

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    out_path = OUT_DIR / f"cc8-openneuro-loso_{STAMP}.csv"
    df.to_csv(out_path, index=False)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
