"""
CC-9 (Claude Code, literature-motivated): EEG functional-connectivity
features (PLV/PLI) for the OpenNeuro ds004504 AD-vs-Control benchmark.

Competitor/gap: recent published work on this exact task (2025-2026, e.g.
the AD/FTD functional-connectivity ensemble study and EEG phase-
synchronization papers found in this round's literature search) uses
phase-locking value (PLV) and phase-lag index (PLI) between channel pairs
as a feature family complementary to -- and in some studies stronger
than -- band-power features, which is all CX-12/14/18 have used so far.

Hypothesis: adding PLV/PLI connectivity features (aggregated by region
pair, plus global summary stats, per band) to the existing derived
band-power feature set improves KANBoost's balanced accuracy/log loss on
the same CX-18 benchmark protocol, either alone or combined with the
existing features.

Scope: model-quality benchmark only, using the SAME evaluation scaffold
as CX-18 (add_eeg_features-style band-power derivation, SelectKBest,
KANBoostClassifier h3/e80/s6, StratifiedKFold(5) x seeds [11,22,33,44,55],
HistGBDT raw baseline) for direct comparability under rule 10. No
kanboost/core change -- this is a feature-engineering-only experiment.

Acceptance gate (self-imposed, since this is a new Claude Code proposal,
not yet reviewed by Codex/ChatGPT):
- Exceeds CX-18's baseline (mean BA 0.7183, mean log loss 0.5757, mean
  ROC AUC 0.7812) by a real margin: mean BA >=0.735 (+0.017 over CX-18)
  or matches BA while cutting log loss by >=0.03, using the *same*
  select-80 pipeline so only the feature pool differs.
- Must not require raw-EEG access beyond what CX-12's pipeline already
  uses (same DATASET_ID/tag, same AD/Control subject set, same
  filter/resample settings) -- so results stay comparable.
- Reject if connectivity features add no measurable signal over the
  existing band-power pool (BA improvement <0.01) given the added
  extraction cost (Hilbert transform is O(n log n) per channel per band,
  non-trivial but not prohibitive).

PLV/PLI implementation validated against synthetic signals before this
script was written (fixed-lag same-frequency sinusoids -> PLV=PLI=1.0;
independent-frequency/noise -> both ~0; zero-lag identical signal -> PLV=1
but PLI=0 exactly, confirming PLI's documented zero-lag insensitivity
that distinguishes it from PLV in the literature).
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

PLATFORM = "kaggle" if Path("/kaggle/working").exists() else "local"
WORK_ROOT = Path("/kaggle/working") if PLATFORM == "kaggle" else Path(__file__).resolve().parent
CACHE_ROOT = Path("/kaggle/temp") if PLATFORM == "kaggle" else WORK_ROOT / "cache"
OUT_DIR = WORK_ROOT / "outputs" if PLATFORM == "kaggle" else WORK_ROOT.parent / "results" / "kaggle_cc9_openneuro_connectivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
STAMP = time.strftime("%Y%m%d-%H%M%S", time.gmtime())

DATASET_ID = "ds004504"
DATASET_VERSION = "1.0.7"
DATA_DIR = CACHE_ROOT / "openneuro_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

BANDS = {
    "delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0), "gamma_low": (30.0, 45.0),
}
REGIONS = {
    "frontal": ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"],
    "central": ["C3", "C4", "Cz"],
    "temporal": ["T3", "T4", "T5", "T6"],
    "parietal": ["P3", "P4", "Pz"],
    "occipital": ["O1", "O2"],
}


def pip_install(*pkgs):
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])


def ensure_deps():
    import importlib.util
    needed = {"openneuro": "openneuro-py", "mne": "mne"}
    missing = [pip for mod, pip in needed.items() if importlib.util.find_spec(mod) is None]
    if missing:
        pip_install(*missing)
    wheel = Path(__file__).with_name("kanboost-1.2.4-py3-none-any.whl")
    if not wheel.exists():
        matches = sorted(Path("/kaggle/input").glob("**/kanboost-1.2.4-py3-none-any.whl"))
        if matches:
            wheel = matches[0]
    if wheel.exists():
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "--no-deps", str(wheel)])
    else:
        # Fall back to the working tree (includes CX-13/CX-20; prediction-only
        # speed change, exact parity, does not affect these results).
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def direct_download_subject(pid, dest_dir):
    sub = pid.replace("sub-", "")
    dest = dest_dir / f"sub-{sub}" / "eeg"
    dest.mkdir(parents=True, exist_ok=True)
    base_url = f"https://s3.amazonaws.com/openneuro.org/{DATASET_ID}/sub-{sub}/eeg"
    fname = f"sub-{sub}_task-eyesclosed_eeg.set"
    out = dest / fname
    if out.exists() and out.stat().st_size > 0:
        return out
    url = f"{base_url}/{fname}"
    try:
        urllib.request.urlretrieve(url, out)
    except urllib.error.HTTPError as exc:
        raise FileNotFoundError(f"{url}: {exc}")
    return out


def find_or_fetch_set_path(pid):
    sub = pid.replace("sub-", "")
    pattern = f"sub-{sub}_task-eyesclosed_eeg.set"
    matches = sorted(DATA_DIR.rglob(pattern))
    if matches:
        return matches[0]
    try:
        import openneuro
        includes = [f"sub-{sub}", "participants.tsv", "dataset_description.json"]
        openneuro.download(dataset=DATASET_ID, tag=DATASET_VERSION, target_dir=str(DATA_DIR), include=includes)
    except Exception as exc:
        print(f"openneuro-py fetch failed for {pid} ({exc}); trying direct S3", file=sys.stderr)
    matches = sorted(DATA_DIR.rglob(pattern))
    if matches:
        return matches[0]
    return direct_download_subject(pid, DATA_DIR)


def bandpower_from_psd(psd, freqs, lo, hi):
    mask = (freqs >= lo) & (freqs < hi)
    if mask.sum() == 0:
        return np.full(psd.shape[0], np.nan)
    return np.trapz(psd[:, mask], freqs[mask], axis=1)


def plv_pli_matrix(phases):
    """phases: (n_channels, n_samples) -> (plv, pli), each (n_channels, n_channels)."""
    n_ch = phases.shape[0]
    plv = np.eye(n_ch)
    pli = np.eye(n_ch)
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            diff = phases[i] - phases[j]
            v = np.abs(np.mean(np.exp(1j * diff)))
            l = np.abs(np.mean(np.sign(np.sin(diff))))
            plv[i, j] = plv[j, i] = v
            pli[i, j] = pli[j, i] = l
    return plv, pli


def aggregate_connectivity(matrix, ch_names, band, metric):
    idx = {name: i for i, name in enumerate(ch_names)}
    feats = {}
    region_names = list(REGIONS)
    all_pairs_vals = []
    n = len(ch_names)
    for i in range(n):
        for j in range(i + 1, n):
            all_pairs_vals.append(matrix[i, j])
    feats[f"{band}_{metric}_global_mean"] = float(np.mean(all_pairs_vals))
    feats[f"{band}_{metric}_global_std"] = float(np.std(all_pairs_vals))
    for a in range(len(region_names)):
        for b in range(a + 1, len(region_names)):
            r1, r2 = region_names[a], region_names[b]
            idx1 = [idx[ch] for ch in REGIONS[r1] if ch in idx]
            idx2 = [idx[ch] for ch in REGIONS[r2] if ch in idx]
            if not idx1 or not idx2:
                continue
            vals = [matrix[i, j] for i in idx1 for j in idx2]
            feats[f"{band}_{metric}_{r1}_{r2}"] = float(np.mean(vals))
    return feats


def extract_subject_features(pid, target, group, from_mne, hilbert):
    set_path = find_or_fetch_set_path(pid)
    t0 = time.perf_counter()
    raw = from_mne.io.read_raw_eeglab(set_path, preload=True, verbose="ERROR")
    raw.pick_types(eeg=True)
    raw.filter(0.5, 45.0, fir_design="firwin", verbose="ERROR")
    raw.resample(250, verbose="ERROR")
    ch_names = raw.ch_names

    spectrum = raw.compute_psd(method="welch", fmin=0.5, fmax=45.0, n_fft=512, n_overlap=256, verbose="ERROR")
    psd, freqs = spectrum.get_data(), spectrum.freqs
    total = bandpower_from_psd(psd, freqs, 0.5, 45.0)

    feats = {
        "participant_id": pid, "target": target, "group": group,
        "n_channels": int(len(ch_names)), "duration_seconds": float(raw.times[-1]),
    }
    for band, (lo, hi) in BANDS.items():
        bp = bandpower_from_psd(psd, freqs, lo, hi)
        rel = bp / np.maximum(total, 1e-18)
        feats[f"{band}_mean_log_power"] = float(np.mean(np.log10(bp + 1e-18)))
        feats[f"{band}_std_log_power"] = float(np.std(np.log10(bp + 1e-18)))
        feats[f"{band}_mean_rel_power"] = float(np.mean(rel))
        for ch, val in zip(ch_names, rel):
            feats[f"{band}_rel_{ch.replace(' ', '_')}"] = float(val)

    for band, (lo, hi) in BANDS.items():
        band_raw = raw.copy().filter(lo, hi, fir_design="firwin", verbose="ERROR")
        data = band_raw.get_data()  # (n_channels, n_samples)
        analytic = hilbert(data, axis=1)
        phases = np.angle(analytic)
        plv_mat, pli_mat = plv_pli_matrix(phases)
        feats.update(aggregate_connectivity(plv_mat, ch_names, band, "plv"))
        feats.update(aggregate_connectivity(pli_mat, ch_names, band, "pli"))

    feats["feature_seconds"] = float(time.perf_counter() - t0)
    return feats


def build_feature_table():
    import mne
    mne.set_log_level("WARNING")
    from scipy.signal import hilbert

    participants_url = "https://raw.githubusercontent.com/OpenNeuroDatasets/ds004504/master/participants.tsv"
    participants = pd.read_csv(participants_url, sep="\t")
    participants["participant_id"] = participants["participant_id"].astype(str)
    participants = participants[participants["Group"].isin(["A", "C"])].copy()
    participants["target"] = participants["Group"].map({"A": "AD", "C": "Control"})
    participants = participants.sort_values("participant_id").reset_index(drop=True)

    rows = []
    for i, row in participants.iterrows():
        print(f"[{i+1}/{len(participants)}] extracting {row['participant_id']} ({row['target']}) ...")
        t0 = time.perf_counter()
        feats = extract_subject_features(row["participant_id"], row["target"], row["Group"], mne, hilbert)
        rows.append(feats)
        print(f"  done in {time.perf_counter()-t0:.1f}s")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Evaluation scaffold -- mirrors CX-18's exact protocol for comparability
# (rule 10): same add_eeg_features-style band-power ratios, same
# SelectKBest(f_classif, k=80), same KANBoostClassifier h3/e80/s6, same
# StratifiedKFold(5) x seeds [11,22,33,44,55], same HistGBDT raw baseline.
# ---------------------------------------------------------------------------

CHANNELS = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"]
BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma_low"]
PAIRS = [("Fp1", "Fp2"), ("F3", "F4"), ("C3", "C4"), ("P3", "P4"), ("O1", "O2"), ("F7", "F8"), ("T3", "T4"), ("T5", "T6")]
SEEDS = [11, 22, 33, 44, 55]
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
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.impute import SimpleImputer
    return [SimpleImputer(strategy="median"), VarianceThreshold(threshold=0.0)]


def kan_model(seed):
    from kanboost import KANBoostClassifier
    return KANBoostClassifier(
        n_estimators=80, learning_rate=0.1, kan_hidden=3, kan_steps=6,
        gam=False, early_stopping_rounds=None, random_state=seed,
    )


def factory(seed, k):
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.pipeline import make_pipeline
    return make_pipeline(*clean_prefix(), SelectKBest(f_classif, k=k), kan_model(seed))


def eval_outer(model_name, X, y, seed_seq, k):
    from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    rows = []
    for seed in seed_seq:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(cv.split(X, y), start=1):
            model = factory(seed, min(k, X.shape[1]))
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


def eval_hist_raw(X, y, seed_seq):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    rows = []
    for seed in seed_seq:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(cv.split(X, y), start=1):
            model = make_pipeline(*clean_prefix(), HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed))
            model.fit(X[tr], y[tr])
            p = model.predict_proba(X[va])[:, 1]
            pred = (p >= 0.5).astype(int)
            rows.append({
                "model": "hist_gbdt_raw_t0p5", "cv_seed": seed, "fold": fold,
                "balanced_accuracy": balanced_accuracy_score(y[va], pred),
                "f1_macro": f1_score(y[va], pred, average="macro"),
                "log_loss": log_loss(y[va], np.column_stack([1 - p, p]), labels=[0, 1]),
                "roc_auc": roc_auc_score(y[va], p),
                "fit_seconds": 0.0,
            })
    return rows


def main():
    ensure_deps()
    from sklearn.preprocessing import LabelEncoder

    features_path = OUT_DIR / f"cc9_openneuro_connectivity_features_{STAMP}.csv"
    features = build_feature_table()
    features.to_csv(features_path, index=False)
    print("wrote", features_path)

    y = LabelEncoder().fit_transform(features["target"].to_numpy())
    raw_cols = [c for c in features.columns if c not in EXCLUDED_COLUMNS]
    raw = features[raw_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)

    derived = add_eeg_ratio_features(features)
    connectivity_cols = [c for c in features.columns if "_plv_" in c or "_pli_" in c]
    bandpower_derived_cols = [c for c in derived.columns if c not in EXCLUDED_COLUMNS and c not in connectivity_cols]

    X_bandpower_only = derived[bandpower_derived_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    X_connectivity_only = features[connectivity_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    X_combined = derived[bandpower_derived_cols + connectivity_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)

    print({"n_subjects": len(y), "n_bandpower_derived": len(bandpower_derived_cols),
           "n_connectivity": len(connectivity_cols), "n_combined": X_combined.shape[1]})

    rows = []
    rows.extend(eval_hist_raw(raw, y, SEEDS))
    rows.extend(eval_outer("kanboost_bandpower_select80", X_bandpower_only, y, SEEDS, 80))
    rows.extend(eval_outer("kanboost_connectivity_only", X_connectivity_only, y, SEEDS, min(80, X_connectivity_only.shape[1])))
    rows.extend(eval_outer("kanboost_combined_select80", X_combined, y, SEEDS, 80))

    metrics = pd.DataFrame(rows)
    summary = (
        metrics.groupby("model")
        .agg(mean_balanced_accuracy=("balanced_accuracy", "mean"),
             std_balanced_accuracy=("balanced_accuracy", "std"),
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

    prefix = f"cc9-openneuro-connectivity_{STAMP}"
    metrics.to_csv(OUT_DIR / f"{prefix}_metrics.csv", index=False)
    summary.to_csv(OUT_DIR / f"{prefix}_summary.csv", index=False)
    seed_summary.to_csv(OUT_DIR / f"{prefix}_seed_summary.csv", index=False)
    (OUT_DIR / f"{prefix}_results.json").write_text(json.dumps({
        "platform": PLATFORM, "seeds": SEEDS,
        "n_bandpower_derived": len(bandpower_derived_cols), "n_connectivity": len(connectivity_cols),
        "summary": summary.to_dict(orient="records"), "seed_summary": seed_summary.to_dict(orient="records"),
    }, indent=2), encoding="utf-8")

    print("SUMMARY")
    print(summary.to_string(index=False))
    print("SEED SUMMARY")
    print(seed_summary.to_string(index=False))


if __name__ == "__main__":
    main()
