"""
Generates a standalone Colab/Kaggle notebook for CC-11: KANBoost vs
HistGBDT on OpenNeuro ds007823 (COVID-19 survivors vs close-contact
controls EEG, Cuban Neuroscience Center, 2026). Mirrors the cell-based
approach in generate_openneuro_sequential_notebook.py, but for the new
CC-11 dataset/task instead of ds004504.

Run this script to (re)generate the .ipynb; do not hand-edit the
notebook JSON directly, since this generator is the source of truth.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COLAB_OUT = ROOT / "colab_cc11_covid_eeg" / "openneuro_ds007823_covid_eeg_benchmark.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(True)}


cells = [
    md("""# CC-11: KANBoost vs HistGBDT on OpenNeuro ds007823 (COVID-19 EEG)

Tests whether KANBoost's demonstrated edge over HistGBDT (seen on the
small, n=65 ds004504 Alzheimer's/FTD EEG task) is specific to that
dataset, or holds on a different, larger (n=173), more recently
published (2026) EEG classification task: COVID-19 survivors vs
close-contact controls.

Dataset: **ds007823**, "A COVID-19 survivors and close contacts EEG
dataset" (Cuban Neuroscience Center), CC0 license, published in
*Clinical Neurophysiology Practice* 2026. 173 subjects (87 Covid, 86
Control), 21-channel resting EEG at 200Hz, EDF format.

Protocol: same band-power feature family as the ds004504 pipeline (5
bands x per-channel relative power + derived ratios/asymmetry/entropy),
`StratifiedKFold(5)` x seeds `[11,22,33,44,55]`, `SelectKBest(k=80)`,
`KANBoostClassifier(n_estimators=80, kan_hidden=3, kan_steps=6)` vs
`HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05)`.
"""),
    code("""KANBOOST_COMMIT = "11f4b0fd95ac40aa475702a81d5efbb749cee55b"  # or pip install kanboost==1.2.4
DATASET_ID = "ds007823"
RANDOM_STATE = 42
SEEDS = [11, 22, 33, 44, 55]
"""),
    code("""import sys, subprocess, importlib.util

def pip_install(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

needed = {"openneuro": "openneuro-py", "mne": "mne"}
missing = [pip for mod, pip in needed.items() if importlib.util.find_spec(mod) is None]
if missing:
    pip_install(*missing)
if importlib.util.find_spec("kanboost") is None:
    pip_install(f"git+https://github.com/tuamah/kanboost.git@{KANBOOST_COMMIT}")
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
participants = pd.read_csv(participants_url, sep="\\t")
print("class counts:", participants["group"].value_counts().to_dict())
display(participants.head())
"""),
    code("""import urllib.error, urllib.request

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
"""),
    code("""def add_eeg_ratio_features(df):
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
"""),
    code("""rows = []
for i, row in participants.iterrows():
    pid, target = row["participant_id"], row["group"]
    print(f"[{i+1}/{len(participants)}] {pid} ({target}) ...")
    feats = extract_subject_features(pid, target)
    rows.append(feats)

features = pd.DataFrame(rows)
features_path = OUT_DIR / f"cc11_covid_eeg_features_{STAMP}.csv"
features.to_csv(features_path, index=False)
print("wrote", features_path)
features.head()
"""),
    code("""from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from kanboost import KANBoostClassifier

def clean_prefix():
    return [SimpleImputer(strategy="median"), VarianceThreshold(threshold=0.0)]

def eval_kanboost(X, y, n_est=80, hid=3, steps=6, k=80):
    rows = []
    for seed in SEEDS:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(cv.split(X, y), start=1):
            model = make_pipeline(*clean_prefix(), SelectKBest(f_classif, k=min(k, X.shape[1])),
                                   KANBoostClassifier(n_estimators=n_est, kan_hidden=hid, kan_steps=steps,
                                                       gam=False, early_stopping_rounds=None, random_state=seed))
            t0 = time.perf_counter()
            model.fit(X[tr], y[tr])
            fit_s = time.perf_counter() - t0
            p = model.predict_proba(X[va])[:, 1]
            pred = (p >= 0.5).astype(int)
            rows.append({"model": f"kanboost_e{n_est}_h{hid}_s{steps}", "cv_seed": seed, "fold": fold,
                         "balanced_accuracy": balanced_accuracy_score(y[va], pred),
                         "f1_macro": f1_score(y[va], pred, average="macro"),
                         "log_loss": log_loss(y[va], np.column_stack([1 - p, p]), labels=[0, 1]),
                         "roc_auc": roc_auc_score(y[va], p), "fit_seconds": fit_s})
    return rows

def eval_histgbdt(X, y, k=80):
    # Must match eval_kanboost's SelectKBest(k) exactly -- giving HistGBDT
    # the full unselected feature pool while KANBoost gets a selected
    # subset is an unfair comparison (196 features vs ~138 training
    # rows/fold caused HistGBDT to badly overfit in an earlier version of
    # this notebook; see AI_REVIEW_LOOP.md CC-11).
    rows = []
    for seed in SEEDS:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(cv.split(X, y), start=1):
            model = make_pipeline(*clean_prefix(), SelectKBest(f_classif, k=min(k, X.shape[1])),
                                   HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed))
            model.fit(X[tr], y[tr])
            p = model.predict_proba(X[va])[:, 1]
            pred = (p >= 0.5).astype(int)
            rows.append({"model": "hist_gbdt_select80_t0p5", "cv_seed": seed, "fold": fold,
                         "balanced_accuracy": balanced_accuracy_score(y[va], pred),
                         "f1_macro": f1_score(y[va], pred, average="macro"),
                         "log_loss": log_loss(y[va], np.column_stack([1 - p, p]), labels=[0, 1]),
                         "roc_auc": roc_auc_score(y[va], p), "fit_seconds": 0.0})
    return rows
"""),
    code("""y = LabelEncoder().fit_transform(features["target"].to_numpy())
derived = add_eeg_ratio_features(features)
feat_cols = [c for c in derived.columns if c not in EXCLUDED_COLUMNS]
X = derived[feat_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
print({"n_subjects": len(y), "n_features": len(feat_cols), "positive_rate": float(y.mean())})

metric_rows = []
metric_rows.extend(eval_histgbdt(X, y))
metric_rows.extend(eval_kanboost(X, y))

metrics = pd.DataFrame(metric_rows)
summary = (metrics.groupby("model")
    .agg(mean_balanced_accuracy=("balanced_accuracy", "mean"), std_balanced_accuracy=("balanced_accuracy", "std"),
         worst_seed_ba=("balanced_accuracy", "min"), mean_log_loss=("log_loss", "mean"),
         mean_roc_auc=("roc_auc", "mean"), mean_fit_seconds=("fit_seconds", "mean"), folds=("fold", "count"))
    .reset_index().sort_values(["mean_balanced_accuracy", "mean_log_loss"], ascending=[False, True]))
seed_summary = (metrics.groupby(["model", "cv_seed"])
    .agg(mean_balanced_accuracy=("balanced_accuracy", "mean"), mean_log_loss=("log_loss", "mean"))
    .reset_index().sort_values(["cv_seed", "model"]))

prefix = f"cc11-covid-eeg-benchmark_{STAMP}"
metrics.to_csv(OUT_DIR / f"{prefix}_metrics.csv", index=False)
summary.to_csv(OUT_DIR / f"{prefix}_summary.csv", index=False)
seed_summary.to_csv(OUT_DIR / f"{prefix}_seed_summary.csv", index=False)
(OUT_DIR / f"{prefix}_results.json").write_text(json.dumps({
    "seeds": SEEDS, "n": len(y), "n_features": len(feat_cols),
    "summary": summary.to_dict(orient="records"), "seed_summary": seed_summary.to_dict(orient="records"),
}, indent=2), encoding="utf-8")

print("SUMMARY"); print(summary.to_string(index=False))
print("SEED SUMMARY"); print(seed_summary.to_string(index=False))
display(summary)
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
        "colab": {"name": "openneuro_ds007823_covid_eeg_benchmark", "provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

COLAB_OUT.parent.mkdir(parents=True, exist_ok=True)
COLAB_OUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print("wrote", COLAB_OUT)
