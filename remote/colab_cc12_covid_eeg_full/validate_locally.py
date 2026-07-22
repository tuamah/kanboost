"""
Local validation harness for the CC-12 notebook's logic, before treating
it as delivered. Reuses the CC-11 EDF cache (already downloaded, no new
network cost) and runs only the small stage (30 subjects) plus the
equations cell, to catch bugs cheaply before trusting the full notebook
on Colab.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mne
mne.set_log_level("WARNING")

DATASET_ID = "ds007823"
RANDOM_STATE = 42
SEEDS = [11, 22]  # trimmed for a fast validation pass, not the real 5
N_SPLITS = 5
SELECT_K = 80
CACHE_DATA_DIR = Path(__file__).resolve().parents[1] / "kaggle_cc11_covid_eeg" / "cache" / "eeg_data"

STAGE = dict(stage="small", n_per_group=15, kanboost_estimators=20, kanboost_hidden=3, kanboost_steps=4, tree_estimators=80)

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
EXCLUDED_COLUMNS = {"participant_id", "target", "n_channels", "duration_seconds", "feature_seconds"}

participants_all = pd.read_csv(Path(__file__).resolve().parents[1] / "kaggle_cc11_covid_eeg" / "participants.tsv", sep="\t")


def select_stage_participants(stage):
    df = (participants_all.groupby("group", group_keys=False)
          .apply(lambda g: g.sample(min(stage["n_per_group"], len(g)), random_state=RANDOM_STATE))
          .reset_index(drop=True))
    return df.sort_values("participant_id").reset_index(drop=True)


def find_or_fetch_edf(pid):
    out = CACHE_DATA_DIR / pid / "eeg" / f"{pid}_task-COVID_eeg.edf"
    assert out.exists(), f"expected cached file missing: {out}"
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
        rows.append(extract_subject_features(row["participant_id"], row["group"]))
    return pd.DataFrame(rows)


from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from kanboost import KANBoostClassifier
from kanboost.interpret.symbolic import tiered_equations


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
                    "balanced_accuracy": balanced_accuracy_score(y[va], pred),
                    "log_loss": log_loss(y[va], np.column_stack([1 - p, p]), labels=[0, 1]),
                    "roc_auc": roc_auc_score(y[va], p),
                    "fit_seconds": fit_s,
                })
    return rows


def main():
    print("=== extracting small-stage features (reusing CC-11 cache, no download) ===")
    stage_participants = select_stage_participants(STAGE)
    print(f"n={len(stage_participants)}, class counts: {stage_participants['group'].value_counts().to_dict()}")
    features = build_feature_table(stage_participants)
    print("feature table shape:", features.shape)

    y = LabelEncoder().fit_transform(features["target"].to_numpy())
    derived = add_eeg_ratio_features(features)
    feat_cols = [c for c in derived.columns if c not in EXCLUDED_COLUMNS]
    X = derived[feat_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    print("X shape:", X.shape, "n_features:", len(feat_cols))

    print("=== running eval_stage for all 4 models (trimmed seeds) ===")
    rows = eval_stage("small_validation", X, y, STAGE)
    metrics = pd.DataFrame(rows)
    summary = metrics.groupby("model").agg(
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        mean_log_loss=("log_loss", "mean"),
        mean_roc_auc=("roc_auc", "mean"),
        mean_fit_seconds=("fit_seconds", "mean"),
        folds=("fold", "count"),
    ).reset_index()
    print(summary.to_string(index=False))
    assert set(summary["model"]) == {"kanboost", "hist_gbdt", "catboost", "xgboost"}, "missing a model!"
    assert not metrics[["balanced_accuracy", "log_loss", "roc_auc"]].isna().any().any(), "NaN in metrics!"
    print("all 4 models ran successfully, no NaN -- eval_stage logic OK")

    print("\\n=== testing tiered_equations() (GAM mode) ===")
    # tiered_equations()/distill_equation() need direct access to KANBoost's
    # own introspection methods (feature_importances_dict(), etc.) -- an
    # sklearn Pipeline wrapping it hides those (AttributeError). And for
    # readable formulas with real feature names (not "feature_0"), the
    # model needs a DataFrame with column names, not a raw numpy array
    # (KANBoostClassifier.fit() sets feature_names_in_ from X.columns).
    # So: do imputation/variance-threshold/SelectKBest ONCE up front here,
    # keeping a DataFrame, then pass a bare KANBoostClassifier.
    X_df = derived[feat_cols].replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X_df), columns=X_df.columns)
    vt = VarianceThreshold(threshold=0.0)
    X_vt = pd.DataFrame(vt.fit_transform(X_imp), columns=X_imp.columns[vt.get_support()])
    skb = SelectKBest(f_classif, k=min(SELECT_K, X_vt.shape[1]))
    skb.fit(X_vt, y)
    X_selected = X_vt.loc[:, skb.get_support()]
    print(f"equations cell: {X_selected.shape[1]} selected features (named), for readable formulas")

    def build_and_fit_gam(X_train, y_train, seed):
        return KANBoostClassifier(gam=True, kan_hidden=1, n_estimators=STAGE["kanboost_estimators"],
                                   kan_steps=STAGE["kanboost_steps"], early_stopping_rounds=None,
                                   random_state=seed).fit(X_train, y_train)

    t0 = time.perf_counter()
    tiers = tiered_equations(build_and_fit_gam, X_selected, y, simple_max_terms=5, detailed_max_terms=12,
                              n_seeds=3, random_state=RANDOM_STATE)
    print(f"tiered_equations() completed in {time.perf_counter()-t0:.1f}s")
    for tier_name in ["simple", "detailed", "full"]:
        tier = tiers[tier_name]
        print(f"--- {tier_name} ({len(tier['kept_features'])} terms) ---")
        print("formula:", str(tier["formula"])[:200])
        print("fidelity:", tier["fidelity"])
    print("\\ntiered_equations() cell logic OK")


if __name__ == "__main__":
    main()
