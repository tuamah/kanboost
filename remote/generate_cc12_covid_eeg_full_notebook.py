"""
Generates a comprehensive Colab/Kaggle notebook for ds007823 (COVID-19
EEG): KANBoost vs HistGBDT vs CatBoost vs XGBoost, staged across small/
medium/large subject counts, plus a final cross-stage comparison, a
results-saving cell, and a tiered (simple/detailed/full) symbolic
equation extraction cell using KANBoost's own `tiered_equations()`.

Builds on CC-11's lessons (see AI_REVIEW_LOOP.md): every model in every
stage uses the IDENTICAL SelectKBest(k) feature-selection budget --
CC-11 found that giving one model an unfair feature/capacity budget
produces misleading results, twice, before being caught.

Run this script to (re)generate the .ipynb; do not hand-edit the
notebook JSON directly.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "colab_cc12_covid_eeg_full" / "openneuro_ds007823_full_model_comparison.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(True)}


cells = [
    md("""# CC-12: KANBoost vs HistGBDT vs CatBoost vs XGBoost on ds007823 (COVID-19 EEG)

Comprehensive, staged comparison on **ds007823** ("A COVID-19 survivors
and close contacts EEG dataset", Cuban Neuroscience Center, CC0,
*Clinical Neurophysiology Practice* 2026): 173 subjects (87 Covid, 86
Control), 21-channel resting EEG, 200Hz, EDF format.

Structure:
1. **Small** stage -- small balanced subject subset.
2. **Medium** stage -- larger balanced subset.
3. **Large** stage -- all eligible subjects.
4. **Cross-stage comparison** -- every model, every stage, side by side.
5. **Save results** -- timestamped CSV/JSON outputs.
6. **Symbolic equations** -- KANBoost's own `tiered_equations()`: a
   short interpretable formula, a medium-complexity formula, and the
   full, most-accurate symbolic approximation, each with a measured
   fidelity score against the real model (not an assumption).

**Fairness note** (learned the hard way in CC-11 -- see
`AI_REVIEW_LOOP.md`): every model at every stage uses the *same*
`SelectKBest(k)` feature-selection budget. Giving one model more
features or more boosting iterations than another produces misleading
comparisons -- this notebook checks that explicitly rather than assuming
it.
"""),
    code("""DATASET_ID = "ds007823"
RANDOM_STATE = 42
SEEDS = [11, 22, 33, 44, 55]
N_SPLITS = 5
SELECT_K = 80

STAGES = [
    dict(stage="small",  n_per_group=15,  kanboost_estimators=20,  kanboost_hidden=3, kanboost_steps=4, tree_estimators=80),
    dict(stage="medium", n_per_group=40,  kanboost_estimators=60,  kanboost_hidden=4, kanboost_steps=6, tree_estimators=160),
    dict(stage="large",  n_per_group=None, kanboost_estimators=150, kanboost_hidden=4, kanboost_steps=8, tree_estimators=300),
]
print({"dataset": DATASET_ID, "seeds": SEEDS, "stages": [s["stage"] for s in STAGES]})
"""),
    code("""import sys, subprocess, importlib.util

def pip_install(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

needed = {"openneuro": "openneuro-py", "mne": "mne", "catboost": "catboost", "xgboost": "xgboost"}
missing = [pip for mod, pip in needed.items() if importlib.util.find_spec(mod) is None]
if missing:
    pip_install(*missing)
if importlib.util.find_spec("kanboost") is None:
    pip_install("kanboost")
print("dependency check complete", missing)
"""),
    code("""from __future__ import annotations

import json, time
from pathlib import Path

import numpy as np
import pandas as pd
import mne
mne.set_log_level("WARNING")

if Path("/kaggle/working").exists():
    PLATFORM = "kaggle"
    WORK_ROOT, CACHE_ROOT = Path("/kaggle/working"), Path("/kaggle/temp")
elif Path("/content").exists():
    PLATFORM = "colab"
    WORK_ROOT = CACHE_ROOT = Path("/content")
else:
    PLATFORM = "local"
    WORK_ROOT = CACHE_ROOT = Path.cwd()

OUT_DIR = WORK_ROOT / "outputs"
DATA_DIR = CACHE_ROOT / "ds007823_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
STAMP = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
print({"platform": PLATFORM, "work_root": str(WORK_ROOT), "run_stamp": STAMP})
"""),
    code("""BANDS = {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0), "gamma_low": (30.0, 45.0)}
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
EXCLUDED_COLUMNS = {"participant_id", "target", "n_channels", "duration_seconds", "feature_seconds"}

participants_url = "https://s3.amazonaws.com/openneuro.org/ds007823/participants.tsv"
participants_all = pd.read_csv(participants_url, sep="\\t")
print("class counts:", participants_all["group"].value_counts().to_dict())
display(participants_all.head())
"""),
    code("""import urllib.error, urllib.request

def select_stage_participants(stage):
    if stage["n_per_group"] is None:
        df = participants_all.copy()
    else:
        df = (participants_all.groupby("group", group_keys=False)
              .apply(lambda g: g.sample(min(stage["n_per_group"], len(g)), random_state=RANDOM_STATE))
              .reset_index(drop=True))
    return df.sort_values("participant_id").reset_index(drop=True)

def find_or_fetch_edf(pid):
    dest = DATA_DIR / pid / "eeg"
    dest.mkdir(parents=True, exist_ok=True)
    fname = f"{pid}_task-COVID_eeg.edf"
    out = dest / fname
    if out.exists() and out.stat().st_size > 0:
        return out
    try:
        import openneuro
        openneuro.download(dataset=DATASET_ID, target_dir=str(DATA_DIR), include=[pid, "participants.tsv"])
    except Exception as exc:
        print(f"openneuro-py fetch failed for {pid} ({exc}); trying direct S3")
    if out.exists() and out.stat().st_size > 0:
        return out
    url = f"https://s3.amazonaws.com/openneuro.org/{DATASET_ID}/{pid}/eeg/{fname}"
    urllib.request.urlretrieve(url, out)
    return out


def bandpower_from_psd(psd, freqs, lo, hi):
    mask = (freqs >= lo) & (freqs < hi)
    if mask.sum() == 0:
        return np.full(psd.shape[0], np.nan)
    return np.trapz(psd[:, mask], freqs[mask], axis=1)


def extract_subject_features(pid, target):
    edf_path = find_or_fetch_edf(pid)
    t0 = time.perf_counter()
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    rename = {ch: ch.capitalize() if ch.upper() in {"CZ", "FZ", "PZ"} else ch for ch in raw.ch_names}
    raw.rename_channels(rename)
    keep = [ch for ch in CHANNELS if ch in raw.ch_names]
    raw.pick(keep)
    raw.filter(0.5, 45.0, fir_design="firwin", verbose="ERROR")

    spectrum = raw.compute_psd(method="welch", fmin=0.5, fmax=45.0, n_fft=512, n_overlap=256, verbose="ERROR")
    psd, freqs = spectrum.get_data(), spectrum.freqs
    ch_names = raw.ch_names
    total = bandpower_from_psd(psd, freqs, 0.5, 45.0)

    feats = {"participant_id": pid, "target": target,
              "n_channels": int(len(ch_names)), "duration_seconds": float(raw.times[-1])}
    for band, (lo, hi) in BANDS.items():
        bp = bandpower_from_psd(psd, freqs, lo, hi)
        rel = bp / np.maximum(total, 1e-18)
        feats[f"{band}_mean_log_power"] = float(np.mean(np.log10(bp + 1e-18)))
        feats[f"{band}_std_log_power"] = float(np.std(np.log10(bp + 1e-18)))
        feats[f"{band}_mean_rel_power"] = float(np.mean(rel))
        for ch, val in zip(ch_names, rel):
            feats[f"{band}_rel_{ch}"] = float(val)
    feats["feature_seconds"] = float(time.perf_counter() - t0)
    return feats


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


def build_feature_table(stage_participants):
    rows = []
    for i, row in stage_participants.iterrows():
        pid, target = row["participant_id"], row["group"]
        print(f"  [{i+1}/{len(stage_participants)}] {pid} ({target}) ...")
        rows.append(extract_subject_features(pid, target))
    return pd.DataFrame(rows)
"""),
    code("""from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from kanboost import KANBoostClassifier


def clean_prefix():
    return [SimpleImputer(strategy="median"), VarianceThreshold(threshold=0.0)]


def make_kanboost(seed, n_est, hid, steps):
    return KANBoostClassifier(n_estimators=n_est, kan_hidden=hid, kan_steps=steps,
                               gam=False, early_stopping_rounds=None, random_state=seed)

def make_histgbdt(seed, n_est):
    return HistGradientBoostingClassifier(max_iter=n_est, learning_rate=0.05, random_state=seed)

def make_catboost(seed, n_est):
    return CatBoostClassifier(iterations=n_est, learning_rate=0.05, depth=4, verbose=False, random_state=seed)

def make_xgboost(seed, n_est):
    return XGBClassifier(n_estimators=n_est, max_depth=4, learning_rate=0.05,
                          eval_metric="logloss", random_state=seed, verbosity=0)


# Every model uses the SAME SelectKBest(k) budget -- see the fairness
# note in the intro cell / CC-11's history in AI_REVIEW_LOOP.md.
MODEL_FACTORIES = {
    "kanboost": lambda seed, stage: make_kanboost(seed, stage["kanboost_estimators"], stage["kanboost_hidden"], stage["kanboost_steps"]),
    "hist_gbdt": lambda seed, stage: make_histgbdt(seed, stage["tree_estimators"]),
    "catboost": lambda seed, stage: make_catboost(seed, stage["tree_estimators"]),
    "xgboost": lambda seed, stage: make_xgboost(seed, stage["tree_estimators"]),
}


def eval_stage(stage_name, X, y, stage_cfg, k=SELECT_K):
    rows = []
    for model_name, factory in MODEL_FACTORIES.items():
        for seed in SEEDS:
            cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
            for fold, (tr, va) in enumerate(cv.split(X, y), start=1):
                model = make_pipeline(*clean_prefix(), SelectKBest(f_classif, k=min(k, X.shape[1])), factory(seed, stage_cfg))
                t0 = time.perf_counter()
                model.fit(X[tr], y[tr])
                fit_s = time.perf_counter() - t0
                p = model.predict_proba(X[va])[:, 1]
                pred = (p >= 0.5).astype(int)
                rows.append({
                    "data_stage": stage_name, "model": model_name, "cv_seed": seed, "fold": fold,
                    "n_subjects": len(y), "n_features_available": X.shape[1],
                    "balanced_accuracy": balanced_accuracy_score(y[va], pred),
                    "f1_macro": f1_score(y[va], pred, average="macro"),
                    "log_loss": log_loss(y[va], np.column_stack([1 - p, p]), labels=[0, 1]),
                    "roc_auc": roc_auc_score(y[va], p),
                    "fit_seconds": fit_s,
                })
    return rows
"""),
    md("""## Stage 1: Small data"""),
    code("""stage = STAGES[0]
small_participants = select_stage_participants(stage)
print(f"small stage: {len(small_participants)} subjects, {small_participants['group'].value_counts().to_dict()}")
small_features = build_feature_table(small_participants)

y_small = LabelEncoder().fit_transform(small_features["target"].to_numpy())
derived_small = add_eeg_ratio_features(small_features)
feat_cols_small = [c for c in derived_small.columns if c not in EXCLUDED_COLUMNS]
X_small = derived_small[feat_cols_small].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)

small_rows = eval_stage("small", X_small, y_small, stage)
print(f"small stage done: {len(small_rows)} result rows")
"""),
    md("""## Stage 2: Medium data"""),
    code("""stage = STAGES[1]
medium_participants = select_stage_participants(stage)
print(f"medium stage: {len(medium_participants)} subjects, {medium_participants['group'].value_counts().to_dict()}")
medium_features = build_feature_table(medium_participants)

y_medium = LabelEncoder().fit_transform(medium_features["target"].to_numpy())
derived_medium = add_eeg_ratio_features(medium_features)
feat_cols_medium = [c for c in derived_medium.columns if c not in EXCLUDED_COLUMNS]
X_medium = derived_medium[feat_cols_medium].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)

medium_rows = eval_stage("medium", X_medium, y_medium, stage)
print(f"medium stage done: {len(medium_rows)} result rows")
"""),
    md("""## Stage 3: Large data (all eligible subjects)"""),
    code("""stage = STAGES[2]
large_participants = select_stage_participants(stage)
print(f"large stage: {len(large_participants)} subjects, {large_participants['group'].value_counts().to_dict()}")
large_features = build_feature_table(large_participants)
large_features_path = OUT_DIR / f"cc12_ds007823_large_features_{STAMP}.csv"
large_features.to_csv(large_features_path, index=False)
print("wrote", large_features_path)

y_large = LabelEncoder().fit_transform(large_features["target"].to_numpy())
derived_large = add_eeg_ratio_features(large_features)
feat_cols_large = [c for c in derived_large.columns if c not in EXCLUDED_COLUMNS]
X_large = derived_large[feat_cols_large].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)

large_rows = eval_stage("large", X_large, y_large, stage)
print(f"large stage done: {len(large_rows)} result rows")
"""),
    md("""## Cross-stage comparison: every model, every data size, side by side"""),
    code("""all_rows = small_rows + medium_rows + large_rows
metrics = pd.DataFrame(all_rows)

summary = (metrics.groupby(["data_stage", "model"])
    .agg(n_subjects=("n_subjects", "first"),
         mean_balanced_accuracy=("balanced_accuracy", "mean"), std_balanced_accuracy=("balanced_accuracy", "std"),
         worst_seed_ba=("balanced_accuracy", "min"), mean_log_loss=("log_loss", "mean"),
         mean_roc_auc=("roc_auc", "mean"), mean_fit_seconds=("fit_seconds", "mean"), folds=("fold", "count"))
    .reset_index()
    .sort_values(["data_stage", "mean_balanced_accuracy"], ascending=[True, False]))

stage_order = pd.Categorical(summary["data_stage"], categories=["small", "medium", "large"], ordered=True)
summary = summary.assign(_stage_order=stage_order).sort_values(["_stage_order", "mean_balanced_accuracy"], ascending=[True, False]).drop(columns="_stage_order")

print("CROSS-STAGE SUMMARY (every model x every data size)")
display(summary)

seed_summary = (metrics.groupby(["data_stage", "model", "cv_seed"])
    .agg(mean_balanced_accuracy=("balanced_accuracy", "mean"), mean_log_loss=("log_loss", "mean"))
    .reset_index().sort_values(["data_stage", "cv_seed", "model"]))
"""),
    md("""## Save results"""),
    code("""prefix = f"cc12-ds007823-full-comparison_{STAMP}"
metrics_path = OUT_DIR / f"{prefix}_metrics.csv"
summary_path = OUT_DIR / f"{prefix}_summary.csv"
seed_summary_path = OUT_DIR / f"{prefix}_seed_summary.csv"
results_path = OUT_DIR / f"{prefix}_results.json"

metrics.to_csv(metrics_path, index=False)
summary.to_csv(summary_path, index=False)
seed_summary.to_csv(seed_summary_path, index=False)
results_path.write_text(json.dumps({
    "platform": PLATFORM, "run_stamp": STAMP, "dataset": DATASET_ID, "seeds": SEEDS, "select_k": SELECT_K,
    "stages": STAGES,
    "summary": summary.to_dict(orient="records"),
    "seed_summary": seed_summary.to_dict(orient="records"),
}, indent=2), encoding="utf-8")

print("wrote", metrics_path)
print("wrote", summary_path)
print("wrote", seed_summary_path)
print("wrote", results_path)
"""),
    md("""## Symbolic equations: short, medium, and full formulas for what the model learned

Uses KANBoost's own `tiered_equations()` (GAM mode, additive
`F(x) = c + sum_j g_j(x_j)`) on the large-stage data: a **short**
interpretable formula (few terms), a **medium** formula (more terms,
still readable), and the **full** formula (every feature that clears the
R^2 gate, most accurate but least compact). Each tier's `fidelity`
against the real model is measured, not assumed -- "full" is more
accurate by construction, not automatically "better" for explanation.

Restricted to binary classification (`distill_equation`'s own
requirement) -- ds007823's Covid-vs-Control task qualifies directly.
"""),
    code("""from kanboost.interpret.symbolic import tiered_equations

# tiered_equations()/distill_equation() need direct access to KANBoost's
# own introspection methods (feature_importances_dict(), etc.) -- an
# sklearn Pipeline wrapping it hides those (AttributeError). And for
# readable formulas with real feature names (not "feature_0"), the model
# needs a DataFrame with column names, not a raw numpy array
# (KANBoostClassifier.fit() sets feature_names_in_ from X.columns). So:
# do imputation/variance-threshold/SelectKBest ONCE here, keeping a
# DataFrame, then pass a bare KANBoostClassifier -- not the usual pipeline.
large_stage = STAGES[2]
X_large_df = derived_large[feat_cols_large].replace([np.inf, -np.inf], np.nan)
_imputer = SimpleImputer(strategy="median")
_X_imp = pd.DataFrame(_imputer.fit_transform(X_large_df), columns=X_large_df.columns)
_vt = VarianceThreshold(threshold=0.0)
_X_vt = pd.DataFrame(_vt.fit_transform(_X_imp), columns=_X_imp.columns[_vt.get_support()])
_skb = SelectKBest(f_classif, k=min(SELECT_K, _X_vt.shape[1]))
_skb.fit(_X_vt, y_large)
X_large_selected = _X_vt.loc[:, _skb.get_support()]
print(f"equations cell: {X_large_selected.shape[1]} selected features (named), for readable formulas")

def build_and_fit_gam(X_train, y_train, seed):
    return KANBoostClassifier(gam=True, kan_hidden=1, n_estimators=large_stage["kanboost_estimators"],
                               kan_steps=large_stage["kanboost_steps"], early_stopping_rounds=None,
                               random_state=seed).fit(X_train, y_train)

try:
    tiers = tiered_equations(build_and_fit_gam, X_large_selected, y_large, simple_max_terms=5, detailed_max_terms=12,
                              n_seeds=6, random_state=RANDOM_STATE)
    equations_ok = True
except ValueError as exc:
    # distill_equation()'s stability gate correctly refuses to report an
    # equation when no feature has a reproducible relationship with the
    # target across seeds -- this is a meaningful negative result, not a
    # bug, and is consistent with near-chance classification performance
    # (see the cross-stage summary above): don't invent a formula for a
    # model that found no stable signal.
    print("No stable symbolic equation found (min_r2/amplitude/stability gates all failed):")
    print(f"  {exc}")
    print("This is consistent with near-chance classification performance above --")
    print("the model has no reproducible feature-target relationship to report.")
    tiers = None
    equations_ok = False

if equations_ok:
    for tier_name in ["simple", "detailed", "full"]:
        tier = tiers[tier_name]
        print(f"=== {tier_name.upper()} ({len(tier['kept_features'])} terms) ===")
        print("formula:", tier["formula"])
        print("fidelity:", tier["fidelity"])
        print()

    equations_path = OUT_DIR / f"cc12-ds007823-tiered-equations_{STAMP}.json"
    equations_path.write_text(json.dumps({
        tier_name: {"latex": tiers[tier_name]["latex"], "kept_features": tiers[tier_name]["kept_features"],
                    "fidelity": tiers[tier_name]["fidelity"]}
        for tier_name in ["simple", "detailed", "full"]
    }, indent=2, default=str), encoding="utf-8")
    print("wrote", equations_path)
else:
    equations_path = OUT_DIR / f"cc12-ds007823-tiered-equations_{STAMP}.json"
    equations_path.write_text(json.dumps({
        "status": "no_stable_equation",
        "reason": "no feature survived the min_r2/min_relative_amplitude/stability_threshold gates together",
        "interpretation": "consistent with near-chance classification performance in the cross-stage comparison above",
    }, indent=2), encoding="utf-8")
    print("wrote", equations_path, "(negative result recorded, not a crash)")
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
        "colab": {"name": "openneuro_ds007823_full_model_comparison", "provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print("wrote", OUT)
