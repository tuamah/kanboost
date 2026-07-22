"""
Generates a comprehensive Colab/Kaggle notebook for ARGO (post-ischemic
VT ablation EGM dataset, PhysioNet, open access): KANBoost vs HistGBDT
vs CatBoost vs XGBoost, staged across small/medium/large patient
subsets, with cardiac-electrophysiology-informed feature engineering,
patient-grouped cross-validation, a cross-stage comparison, a
results-saving cell, and a tiered symbolic-equations cell.

See AI_REVIEW_LOOP.md's CC-13 entry for the three real bugs this feature
design already had to catch and fix against actual downloaded data
(sentinel-window handling, window-length leakage in raw-count features,
window_duration_ms itself as an explicit leaked feature -- the last one
producing apparent near-perfect model accuracy that was mostly this
artifact, not real EGM signal) plus a patient-grouped-CV fold-count bug
(the small stage's 3 patients can't support 5 GroupKFold splits) -- all
fixes are baked into the cell code below, not left for the notebook's
user to rediscover.

Run this script to (re)generate the .ipynb; do not hand-edit the
notebook JSON directly.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "colab_cc13_argo_vt" / "argo_vt_ablation_model_comparison.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(True)}


cells = [
    md("""# CC-13: KANBoost vs HistGBDT vs CatBoost vs XGBoost on ARGO (VT ablation EGM)

**ARGO** ("Ablation Reinforcement by computer-aided Guidance and
Optimization", PhysioNet, open access, CC BY-NC-SA 4.0 -- **non-commercial
use only**): 9 post-ischemic ventricular tachycardia patients, 1962
annotated 2.5s segments. Each segment has 15 channels at 1000Hz --
1 bipolar + 2 unipolar intracardiac electrogram (EGM) channels from the
mapping catheter, plus a standard 12-lead surface ECG (I, II, III, aVL,
aVR, aVF, V1-V6) -- labeled by 3-expert-annotator consensus as
**Physiological**, **AVP** (abnormal ventricular potential -- an
ablation target), or **Unknown**.

**Task framing**: binary **AVP vs Physiological** classification. Unknown
is excluded from training (see the cleaning cell) -- it represents
segments even 3 independent expert electrophysiologists could not agree
on, i.e. an inherently unreliable label for a clean classification
target, not a real third category worth predicting.

**Clinical grounding for the feature set** (see the feature-extraction
cell for the full rationale): bipolar EGM peak-to-peak amplitude
(standard voltage-mapping scar metric), fractionation rate (re-entry
substrate marker), a late-potential proxy (delayed-conduction marker,
a primary ablation target in its own right), bipolar/unipolar amplitude
ratio (near-field vs far-field discrimination), slew rate (conduction
velocity proxy), and ejection fraction (real prognostic covariate, not
just demographics).

**Patient-grouped, stratified, multi-seed CV**: record counts per
patient range from 46 to 839 -- `StratifiedGroupKFold` (patient as the
group) is used everywhere so no patient's segments ever appear in both
train and validation, which plain `StratifiedKFold` would allow and
which would leak patient-specific EGM morphology across the split.

**Fairness note** (learned in CC-11/CC-12 -- see `AI_REVIEW_LOOP.md`):
every model at every stage uses the identical `SelectKBest(k)`
feature-selection budget.
"""),
    code("""RANDOM_STATE = 42
SEEDS = [11, 22, 33, 44, 55]
N_SPLITS = 5
SELECT_K = 40  # ARGO has ~74 candidate features, not ~200 like the EEG notebooks

# Patients ordered by record count (ascending) so small/medium/large
# stages are a real, monotonically increasing, patient-grouped
# progression rather than an arbitrary subset.
PATIENT_ORDER_BY_SIZE = ["Pt5", "Pt9", "Pt7", "Pt3", "Pt2", "Pt8", "Pt1", "Pt4", "Pt6"]
STAGES = [
    dict(stage="small",  patients=PATIENT_ORDER_BY_SIZE[:3], kanboost_estimators=40,  kanboost_hidden=3, kanboost_steps=6, tree_estimators=100),
    dict(stage="medium", patients=PATIENT_ORDER_BY_SIZE[:6], kanboost_estimators=100, kanboost_hidden=4, kanboost_steps=8, tree_estimators=200),
    dict(stage="large",  patients=PATIENT_ORDER_BY_SIZE,      kanboost_estimators=150, kanboost_hidden=4, kanboost_steps=8, tree_estimators=300),
]
print({"stages": [(s["stage"], len(s["patients"])) for s in STAGES]})
"""),
    code("""import sys, subprocess, importlib.util

def pip_install(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

needed = {"wfdb": "wfdb", "catboost": "catboost", "xgboost": "xgboost", "requests": "requests"}
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
import wfdb

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
DATA_DIR = CACHE_ROOT / "argo_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
STAMP = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
print({"platform": PLATFORM, "work_root": str(WORK_ROOT), "run_stamp": STAMP})
"""),
    md("""## Download ARGO (open access, no credentialing -- CC BY-NC-SA 4.0)

Skips the 3D electroanatomical mesh files and MATLAB duplicate (not
needed for per-segment EGM/ECG classification). Low concurrency + a
shared session + retries -- an earlier naive high-concurrency attempt
triggered PhysioNet connection resets (SSL errors, not a real protocol
issue) -- see `AI_REVIEW_LOOP.md`'s CC-13 entry.
"""),
    code("""import concurrent.futures
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://physionet.org/files/argo/1.0.0"
SUFFIXES = [".dat", ".hea", ".annotation_consensus"]  # individual annotator files not needed -- consensus is ground truth
TOP_LEVEL_FILES = ["Additional_subject_data.csv", "RECORDS"]
MAX_WORKERS = 6

def _make_session():
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS, max_retries=retry))
    return s

def _fetch(session, url, out_path):
    if out_path.exists() and out_path.stat().st_size > 0:
        return True
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            r = session.get(url, timeout=30)
            r.raise_for_status()
            out_path.write_bytes(r.content)
            return True
        except Exception:
            if attempt == 3:
                return False
            time.sleep(1.5 * (attempt + 1))
    return False

def download_argo():
    session = _make_session()
    for fname in TOP_LEVEL_FILES:
        if not _fetch(session, f"{BASE_URL}/{fname}", DATA_DIR / fname):
            raise RuntimeError(f"Could not fetch {fname} -- check network/PhysioNet availability before retrying.")
    records = (DATA_DIR / "RECORDS").read_text().strip().splitlines()
    jobs = [(f"{BASE_URL}/{rec}{suf}", DATA_DIR / f"{rec}{suf}") for rec in records for suf in SUFFIXES]
    print(f"{len(records)} records, {len(jobs)} files to fetch")
    ok, failed = 0, []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_fetch, session, url, out): url for url, out in jobs}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            if fut.result():
                ok += 1
            else:
                failed.append(futures[fut])
            if i % 500 == 0:
                print(f"  {i}/{len(jobs)} ({ok} ok, {len(failed)} failed)")
    print(f"done: {ok}/{len(jobs)} ok, {len(failed)} failed")

    # Verify on disk, not just via the in-memory ok/failed counters -- a
    # Colab disconnect mid-download can leave files partially written or
    # the loop interrupted entirely without raising here. This cell is
    # safe to just re-run if it reports missing files: _fetch() skips
    # any file that already exists with nonzero size, so a rerun only
    # fetches what's still missing.
    missing = [rec for rec in records if not all((DATA_DIR / f"{rec}{suf}").exists() for suf in SUFFIXES)]
    if missing:
        raise RuntimeError(
            f"{len(missing)}/{len(records)} records are missing files on disk after download "
            f"(e.g. {missing[0]}). This usually means the runtime disconnected mid-download. "
            f"Just re-run this cell -- already-downloaded files are skipped, so it will only "
            f"fetch what's still missing."
        )
    print(f"verified: all {len(records)} records have their files on disk.")
    return records

records = download_argo()
"""),
    md("""## Cardiac-electrophysiology-informed feature extraction

Channel layout is **positional**, not name-based (catheter electrode
naming varies by mapping point -- e.g. "20A_19-20" vs "M1-M2" -- but the
structure is always fixed): channel 0 = bipolar EGM, channels 1-2 =
unipolar EGM pair, channels 3-14 = 12-lead ECG in a fixed order.

**Three fixes baked in from real-data validation** (not theoretical --
each was caught by actually running this against downloaded records and
noticing something wrong, not by inspection):
1. Physiological/Unknown labels use an out-of-range sentinel sample
   index (ARGO's convention for "no window specifically delineated" --
   only AVP segments get a precise onset/offset). Falls back to the
   whole record in that case, instead of a degenerate 1-sample window.
2. Fractionation and duration are reported as **rates/proportions**
   (deflections/second, fraction-of-window-above-threshold), not raw
   counts -- raw counts were almost entirely confounded with window
   length itself (AVP's short delineated window vs Physiological/
   Unknown's whole-record fallback), which would have let the model
   learn the annotation convention instead of real EGM morphology.
3. `window_duration_ms` itself is deliberately **excluded** as a
   feature. It was included in an earlier pass and `tiered_equations()`
   immediately found a 1-variable formula using only that column with
   AUC=0.96 -- since only AVP segments get a short delineated window at
   all, window length is nearly a direct proxy for the label itself
   (the same leakage as fix 2, just re-introduced as an explicit
   feature). Every model's apparent near-perfect accuracy in that pass
   was mostly this artifact, not real EGM signal.
"""),
    code("""EGM_BIPOLAR_IDX = 0
EGM_UNIPOLAR_IDX = [1, 2]
ECG_IDX = list(range(3, 15))
ECG_LEAD_NAMES = ["I", "II", "III", "aVL", "aVR", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def load_record_and_label(rec_path):
    full = str(DATA_DIR / rec_path)
    record = wfdb.rdrecord(full)
    ann = wfdb.rdann(full, "annotation_consensus")
    label = ann.aux_note[0] if ann.aux_note else None
    if len(ann.sample) >= 2:
        onset, offset = int(ann.sample[0]), int(ann.sample[-1])
    elif len(ann.sample) == 1:
        sample = int(ann.sample[0])
        if sample < record.sig_len:
            onset, offset = sample, record.sig_len
        else:
            onset, offset = 0, record.sig_len  # out-of-range sentinel -- see markdown above
    else:
        onset, offset = 0, record.sig_len
    onset = max(0, min(onset, record.sig_len - 1))
    offset = max(onset + 1, min(offset, record.sig_len))
    return record, label, onset, offset


def _egm_channel_features(x, fs, prefix):
    if len(x) < 4:
        return {f"{prefix}_{k}": np.nan for k in
                ["p2p", "rms", "duration_frac", "fractionation_rate", "dom_freq",
                 "spectral_entropy", "max_slew", "late_ratio"]}
    duration_s = len(x) / fs
    p2p = float(np.max(x) - np.min(x))
    rms = float(np.sqrt(np.mean(x ** 2)))
    feats = {f"{prefix}_p2p": p2p, f"{prefix}_rms": rms}

    thresh = 0.5 * p2p
    above = np.abs(x - np.mean(x)) > thresh
    feats[f"{prefix}_duration_frac"] = float(np.mean(above))

    dx = np.diff(x)
    sign_changes = np.sum(np.diff(np.sign(dx)) != 0)
    feats[f"{prefix}_fractionation_rate"] = float(sign_changes) / duration_s

    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    spectrum = np.abs(np.fft.rfft(x - np.mean(x)))
    if spectrum.sum() > 0:
        feats[f"{prefix}_dom_freq"] = float(freqs[np.argmax(spectrum)])
        p = spectrum / spectrum.sum()
        p = p[p > 0]
        feats[f"{prefix}_spectral_entropy"] = float(-(p * np.log(p)).sum())
    else:
        feats[f"{prefix}_dom_freq"] = 0.0
        feats[f"{prefix}_spectral_entropy"] = 0.0

    feats[f"{prefix}_max_slew"] = float(np.max(np.abs(dx)) * fs)

    split = int(len(x) * 0.8)
    early_rms = float(np.sqrt(np.mean(x[:split] ** 2))) if split > 0 else np.nan
    late_rms = float(np.sqrt(np.mean(x[split:] ** 2))) if split < len(x) else np.nan
    feats[f"{prefix}_late_ratio"] = late_rms / (early_rms + 1e-9) if early_rms == early_rms else np.nan
    return feats


def extract_features(record, onset, offset):
    fs = record.fs
    sig = record.p_signal
    feats = {}

    bip_whole = sig[:, EGM_BIPOLAR_IDX]
    bip_window = sig[onset:offset, EGM_BIPOLAR_IDX]
    feats.update(_egm_channel_features(bip_whole, fs, "bip_whole"))
    feats.update(_egm_channel_features(bip_window, fs, "bip_window"))

    uni_p2p_whole = []
    for i, uidx in enumerate(EGM_UNIPOLAR_IDX):
        uni_whole = sig[:, uidx]
        uni_window = sig[onset:offset, uidx]
        feats.update(_egm_channel_features(uni_whole, fs, f"uni{i}_whole"))
        feats.update(_egm_channel_features(uni_window, fs, f"uni{i}_window"))
        uni_p2p_whole.append(float(np.max(uni_whole) - np.min(uni_whole)))

    # Whole-record amplitudes only, deliberately -- using the window
    # here would reintroduce the same selection confound as
    # window_duration_ms (see below): a tightly-cropped AVP window's
    # amplitude vs a whole Physiological/Unknown record's amplitude
    # aren't comparable quantities.
    bip_p2p_whole = feats["bip_whole_p2p"]
    max_uni_p2p_whole = max(uni_p2p_whole) if uni_p2p_whole else np.nan
    feats["bipolar_unipolar_ratio_whole"] = bip_p2p_whole / (max_uni_p2p_whole + 1e-9)

    for lead_i, lead_name in zip(ECG_IDX, ECG_LEAD_NAMES):
        x = sig[:, lead_i]
        feats[f"ecg_{lead_name}_p2p"] = float(np.max(x) - np.min(x))
        feats[f"ecg_{lead_name}_rms"] = float(np.sqrt(np.mean(x ** 2)))

    # window_duration_ms is deliberately NOT included as a feature: only
    # AVP segments get a precisely delineated window at all (see
    # load_record_and_label) -- Physiological/Unknown always fall back
    # to the whole record. So window length is almost a direct proxy for
    # the label itself (a labeling-protocol artifact), not real EGM
    # signal. Confirmed empirically: tiered_equations() found a
    # 1-variable formula using only window_duration_ms with AUC=0.96,
    # i.e. every model was mostly learning the annotation protocol, not
    # cardiac electrophysiology. Caught during validation -- see
    # AI_REVIEW_LOOP.md's CC-13 entry.
    return feats
"""),
    md("""## Extract features for all 1962 records"""),
    code("""rows = []
t0 = time.perf_counter()
for i, rec in enumerate(records):
    rec_rel = rec.replace("ARGODataset_Folder/", "")
    record, label, onset, offset = load_record_and_label(rec_rel)
    feats = extract_features(record, onset, offset)
    feats["patient"] = rec_rel.split("/")[0]
    feats["record_id"] = rec_rel.split("/")[1]
    feats["label"] = label
    rows.append(feats)
    if (i + 1) % 300 == 0:
        print(f"  {i+1}/{len(records)} done, {time.perf_counter()-t0:.1f}s")

all_features = pd.DataFrame(rows)
print(f"extracted {len(all_features)} records in {time.perf_counter()-t0:.1f}s")
print("label distribution:", all_features["label"].value_counts().to_dict())

features_path = OUT_DIR / f"cc13_argo_features_{STAMP}.csv"
all_features.to_csv(features_path, index=False)
print("wrote", features_path)
"""),
    md("""## Cleaning: exclude Unknown, merge demographics, guard against degenerate columns -- and a deeper leakage fix

**Why Unknown is excluded from training, not treated as a third class**:
it represents segments where 3 independent expert electrophysiologists
could not reach consensus -- an inherently unreliable ground-truth label,
not a real physiological category with a learnable signature. Training a
3-class model on it would mean asking the model to predict annotator
*disagreement*, not a clinical state.

**A deeper leakage issue than the two already fixed in the feature-
extraction cell above** (caught by noticing every model scored
suspiciously close to perfect, then tracing it with `tiered_equations()`
-- see `AI_REVIEW_LOOP.md`'s CC-13 entry for the full story): only AVP
segments get a precisely delineated annotation window at all; the
*existence* of a short window is therefore itself a near-perfect proxy
for the label, and this leaks into **every** `*_window_*` feature, even
the rate/proportion-normalized ones from the fix above -- a tightly
cropped window around one specific deflection and a whole, mostly-quiet
2.5s recording produce systematically different rate statistics *purely
from being different-length excerpts of different content*, independent
of any real tissue difference. More fundamentally: at real prediction
time on a new, unlabeled 2.5s recording, no window has been delineated
yet -- that delineation *is* what a real detector would need to produce,
not something it can be handed as an input. So `*_window_*` columns are
kept in the saved feature table for reference/future work, but **excluded
from the primary training feature set** below -- only `*_whole_*` and
ECG features (identically computed for every record regardless of label)
are used for classification.
"""),
    code("""demographics = pd.read_csv(DATA_DIR / "Additional_subject_data.csv", sep=";")
demographics.columns = ["patient_num", "sex", "age", "ejection_fraction", "n_points"]
demographics["patient"] = "Pt" + demographics["patient_num"].str.replace("P", "", regex=False)

clean = all_features[all_features["label"].isin(["P", "A"])].copy()
clean["target"] = (clean["label"] == "A").astype(int)  # 1 = AVP, 0 = Physiological
clean = clean.merge(demographics[["patient", "sex", "age", "ejection_fraction"]], on="patient", how="left")
clean["sex_female"] = (clean["sex"] == "F").astype(int)

excluded_cols = {"patient", "record_id", "label", "target", "sex"}
window_leak_cols = [c for c in clean.columns if "_window_" in c]
print(f"excluding {len(window_leak_cols)} window-based columns from training (selection-confound risk, see markdown above): {window_leak_cols[:5]}...")
feat_cols_all = [c for c in clean.columns if c not in excluded_cols and c not in window_leak_cols]

const_cols = [c for c in feat_cols_all if clean[c].nunique(dropna=True) <= 1]
if const_cols:
    print(f"dropping {len(const_cols)} constant columns: {const_cols}")
    feat_cols_all = [c for c in feat_cols_all if c not in const_cols]

print(f"clean shape: {clean.shape}, AVP={int(clean['target'].sum())}, Physiological={int((1-clean['target']).sum())}")
print(f"usable (whole-record-only) feature count: {len(feat_cols_all)}")
print("records per patient (post-cleaning):", clean["patient"].value_counts().to_dict())
"""),
    md("""## Model factories and patient-grouped, stratified, multi-seed evaluation"""),
    code("""from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
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

MODEL_FACTORIES = {
    "kanboost": lambda seed, stage: make_kanboost(seed, stage["kanboost_estimators"], stage["kanboost_hidden"], stage["kanboost_steps"]),
    "hist_gbdt": lambda seed, stage: make_histgbdt(seed, stage["tree_estimators"]),
    "catboost": lambda seed, stage: make_catboost(seed, stage["tree_estimators"]),
    "xgboost": lambda seed, stage: make_xgboost(seed, stage["tree_estimators"]),
}


def eval_stage(stage_name, X, y, groups, stage_cfg, k=SELECT_K):
    # n_splits can't exceed the number of available groups (patients) --
    # the small stage has only 3 patients, so requesting 5 splits
    # produces empty validation folds (caught during validation on real
    # data -- see AI_REVIEW_LOOP.md's CC-13 entry).
    n_splits_stage = min(N_SPLITS, len(set(groups)))
    rows = []
    for model_name, factory in MODEL_FACTORIES.items():
        for seed in SEEDS:
            cv = StratifiedGroupKFold(n_splits=n_splits_stage, shuffle=True, random_state=seed)
            for fold, (tr, va) in enumerate(cv.split(X, y, groups=groups), start=1):
                model = make_pipeline(*clean_prefix(), SelectKBest(f_classif, k=min(k, X.shape[1])), factory(seed, stage_cfg))
                t0 = time.perf_counter()
                model.fit(X[tr], y[tr])
                fit_s = time.perf_counter() - t0
                p = model.predict_proba(X[va])[:, 1]
                pred = (p >= 0.5).astype(int)
                rows.append({
                    "data_stage": stage_name, "model": model_name, "cv_seed": seed, "fold": fold,
                    "n_records": len(y), "n_patients": len(set(groups)),
                    "balanced_accuracy": balanced_accuracy_score(y[va], pred),
                    "f1_macro": f1_score(y[va], pred, average="macro"),
                    "log_loss": log_loss(y[va], np.column_stack([1 - p, p]), labels=[0, 1]),
                    "roc_auc": roc_auc_score(y[va], p),
                    "fit_seconds": fit_s,
                })
    return rows
"""),
    md("""## Stage 1: Small (3 smallest patients)"""),
    code("""stage = STAGES[0]
small = clean[clean["patient"].isin(stage["patients"])].reset_index(drop=True)
print(f"small stage: {len(small)} records from {stage['patients']}, "
      f"AVP={int(small['target'].sum())}, Physiological={int((1-small['target']).sum())}")

X_small = small[feat_cols_all].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
y_small = small["target"].to_numpy()
groups_small = small["patient"].to_numpy()

small_rows = eval_stage("small", X_small, y_small, groups_small, stage)
print(f"small stage done: {len(small_rows)} result rows")
"""),
    md("""## Stage 2: Medium (6 patients)"""),
    code("""stage = STAGES[1]
medium = clean[clean["patient"].isin(stage["patients"])].reset_index(drop=True)
print(f"medium stage: {len(medium)} records from {stage['patients']}, "
      f"AVP={int(medium['target'].sum())}, Physiological={int((1-medium['target']).sum())}")

X_medium = medium[feat_cols_all].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
y_medium = medium["target"].to_numpy()
groups_medium = medium["patient"].to_numpy()

medium_rows = eval_stage("medium", X_medium, y_medium, groups_medium, stage)
print(f"medium stage done: {len(medium_rows)} result rows")
"""),
    md("""## Stage 3: Large (all 9 patients)"""),
    code("""stage = STAGES[2]
large = clean[clean["patient"].isin(stage["patients"])].reset_index(drop=True)
print(f"large stage: {len(large)} records from all {len(stage['patients'])} patients, "
      f"AVP={int(large['target'].sum())}, Physiological={int((1-large['target']).sum())}")

X_large = large[feat_cols_all].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
y_large = large["target"].to_numpy()
groups_large = large["patient"].to_numpy()

large_rows = eval_stage("large", X_large, y_large, groups_large, stage)
print(f"large stage done: {len(large_rows)} result rows")
"""),
    md("""## Cross-stage comparison"""),
    code("""all_rows = small_rows + medium_rows + large_rows
metrics = pd.DataFrame(all_rows)

summary = (metrics.groupby(["data_stage", "model"])
    .agg(n_records=("n_records", "first"), n_patients=("n_patients", "first"),
         mean_balanced_accuracy=("balanced_accuracy", "mean"), std_balanced_accuracy=("balanced_accuracy", "std"),
         worst_seed_ba=("balanced_accuracy", "min"), mean_log_loss=("log_loss", "mean"),
         mean_roc_auc=("roc_auc", "mean"), mean_fit_seconds=("fit_seconds", "mean"), folds=("fold", "count"))
    .reset_index())
stage_order = pd.Categorical(summary["data_stage"], categories=["small", "medium", "large"], ordered=True)
summary = summary.assign(_o=stage_order).sort_values(["_o", "mean_balanced_accuracy"], ascending=[True, False]).drop(columns="_o")

print("CROSS-STAGE SUMMARY (every model x every patient-group size)")
display(summary)

seed_summary = (metrics.groupby(["data_stage", "model", "cv_seed"])
    .agg(mean_balanced_accuracy=("balanced_accuracy", "mean"), mean_log_loss=("log_loss", "mean"))
    .reset_index().sort_values(["data_stage", "cv_seed", "model"]))
"""),
    md("""## Save results"""),
    code("""prefix = f"cc13-argo-full-comparison_{STAMP}"
metrics_path = OUT_DIR / f"{prefix}_metrics.csv"
summary_path = OUT_DIR / f"{prefix}_summary.csv"
seed_summary_path = OUT_DIR / f"{prefix}_seed_summary.csv"
results_path = OUT_DIR / f"{prefix}_results.json"

metrics.to_csv(metrics_path, index=False)
summary.to_csv(summary_path, index=False)
seed_summary.to_csv(seed_summary_path, index=False)
results_path.write_text(json.dumps({
    "platform": PLATFORM, "run_stamp": STAMP, "seeds": SEEDS, "select_k": SELECT_K,
    "stages": [{"stage": s["stage"], "patients": s["patients"]} for s in STAGES],
    "summary": summary.to_dict(orient="records"), "seed_summary": seed_summary.to_dict(orient="records"),
}, indent=2), encoding="utf-8")

print("wrote", metrics_path); print("wrote", summary_path)
print("wrote", seed_summary_path); print("wrote", results_path)
"""),
    md("""## Symbolic equations: short, medium, and full formulas for what distinguishes AVP from Physiological

Uses KANBoost's own `tiered_equations()` on the large-stage data (all 9
patients). As in CC-12: needs a bare `KANBoostClassifier` (not wrapped in
a `Pipeline`) and a DataFrame with real column names for readable
formulas, so feature selection is done once up front here rather than
inside a pipeline.
"""),
    code("""from kanboost.interpret.symbolic import tiered_equations

large_stage = STAGES[2]
X_large_df = large[feat_cols_all].replace([np.inf, -np.inf], np.nan)
_imputer = SimpleImputer(strategy="median")
_X_imp = pd.DataFrame(_imputer.fit_transform(X_large_df), columns=X_large_df.columns)
_vt = VarianceThreshold(threshold=0.0)
_X_vt = pd.DataFrame(_vt.fit_transform(_X_imp), columns=_X_imp.columns[_vt.get_support()])
_skb = SelectKBest(f_classif, k=min(SELECT_K, _X_vt.shape[1]))
_skb.fit(_X_vt, y_large)
X_large_selected = _X_vt.loc[:, _skb.get_support()]
print(f"equations cell: {X_large_selected.shape[1]} selected features (named)")

def build_and_fit_gam(X_train, y_train, seed):
    return KANBoostClassifier(gam=True, kan_hidden=1, n_estimators=large_stage["kanboost_estimators"],
                               kan_steps=large_stage["kanboost_steps"], early_stopping_rounds=None,
                               random_state=seed).fit(X_train, y_train)

try:
    tiers = tiered_equations(build_and_fit_gam, X_large_selected, y_large, simple_max_terms=5, detailed_max_terms=12,
                              n_seeds=6, random_state=RANDOM_STATE)
    equations_ok = True
except ValueError as exc:
    print("No stable symbolic equation found (min_r2/amplitude/stability gates all failed):")
    print(f"  {exc}")
    print("This would mean no feature has a reproducible relationship with AVP-vs-Physiological across seeds.")
    tiers = None
    equations_ok = False

if equations_ok:
    for tier_name in ["simple", "detailed", "full"]:
        tier = tiers[tier_name]
        print(f"=== {tier_name.upper()} ({len(tier['kept_features'])} terms) ===")
        print("formula:", tier["formula"])
        print("fidelity:", tier["fidelity"])
        print()
    equations_path = OUT_DIR / f"cc13-argo-tiered-equations_{STAMP}.json"
    equations_path.write_text(json.dumps({
        tier_name: {"latex": tiers[tier_name]["latex"], "kept_features": tiers[tier_name]["kept_features"],
                    "fidelity": tiers[tier_name]["fidelity"]}
        for tier_name in ["simple", "detailed", "full"]
    }, indent=2, default=str), encoding="utf-8")
    print("wrote", equations_path)
else:
    equations_path = OUT_DIR / f"cc13-argo-tiered-equations_{STAMP}.json"
    equations_path.write_text(json.dumps({
        "status": "no_stable_equation",
        "reason": "no feature survived the min_r2/min_relative_amplitude/stability_threshold gates together",
    }, indent=2), encoding="utf-8")
    print("wrote", equations_path, "(negative result recorded, not a crash)")
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
        "colab": {"name": "argo_vt_ablation_model_comparison", "provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print("wrote", OUT)
