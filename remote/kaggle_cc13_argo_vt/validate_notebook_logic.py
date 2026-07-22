"""
Validates the CC-13 notebook's cleaning + model-comparison + equations
logic end-to-end, reusing the already-extracted full feature table
(argo_features_full.csv) so this doesn't need to re-run raw signal
extraction. Trimmed to the small stage + 2 seeds for speed.
"""
import time

import numpy as np
import pandas as pd

RANDOM_STATE = 42
SEEDS = [11, 22]  # trimmed for validation speed
N_SPLITS = 5
SELECT_K = 40
PATIENT_ORDER_BY_SIZE = ["Pt5", "Pt9", "Pt7", "Pt3", "Pt2", "Pt8", "Pt1", "Pt4", "Pt6"]
STAGE = dict(stage="small", patients=PATIENT_ORDER_BY_SIZE[:3], kanboost_estimators=40, kanboost_hidden=3, kanboost_steps=6, tree_estimators=100)

all_features = pd.read_csv("../results/kaggle_cc13_argo_vt/argo_features_full.csv")
demographics = pd.read_csv("cache/argo_data/Additional_subject_data.csv", sep=";")
demographics.columns = ["patient_num", "sex", "age", "ejection_fraction", "n_points"]
demographics["patient"] = "Pt" + demographics["patient_num"].str.replace("P", "", regex=False)

clean = all_features[all_features["label"].isin(["P", "A"])].copy()
clean["target"] = (clean["label"] == "A").astype(int)
clean = clean.merge(demographics[["patient", "sex", "age", "ejection_fraction"]], on="patient", how="left")
clean["sex_female"] = (clean["sex"] == "F").astype(int)

excluded_cols = {"patient", "record_id", "label", "target", "sex"}
# window-based features excluded: only AVP gets a delineated window at
# all, so window existence/rate-statistics leak the label even after
# rate-normalization (deeper issue caught after the first "too good to
# be true" run -- see AI_REVIEW_LOOP.md's CC-13 entry). Whole-record
# features are computed identically for every record regardless of label.
window_leak_cols = [c for c in clean.columns if "_window_" in c]
print(f"excluding {len(window_leak_cols)} window-based columns from training")
feat_cols_all = [c for c in clean.columns if c not in excluded_cols and c not in window_leak_cols]
const_cols = [c for c in feat_cols_all if clean[c].nunique(dropna=True) <= 1]
if const_cols:
    print(f"dropping {len(const_cols)} constant columns: {const_cols}")
    feat_cols_all = [c for c in feat_cols_all if c not in const_cols]

print(f"clean shape: {clean.shape}, AVP={int(clean['target'].sum())}, Physiological={int((1-clean['target']).sum())}")
print(f"usable feature count: {len(feat_cols_all)}")
assert not clean[feat_cols_all].isna().all(axis=None), "all-NaN after cleaning!"

from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
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


def eval_stage(stage_name, X, y, groups, stage_cfg, k=SELECT_K):
    # n_splits can't exceed the number of available groups (patients) --
    # a stage with only 3 patients can give at most 3 group-disjoint
    # folds; requesting 5 produces empty validation folds (caught during
    # validation on real data, see AI_REVIEW_LOOP.md's CC-13 entry).
    n_splits_stage = min(N_SPLITS, len(set(groups)))
    rows = []
    for model_name, factory in MODEL_FACTORIES.items():
        for seed in SEEDS:
            cv = StratifiedGroupKFold(n_splits=n_splits_stage, shuffle=True, random_state=seed)
            for fold, (tr, va) in enumerate(cv.split(X, y, groups=groups), start=1):
                # verify no patient leakage across this fold
                assert not (set(groups[tr]) & set(groups[va])), "PATIENT LEAKAGE DETECTED"
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


print("\n=== running eval_stage (small stage, GroupKFold leakage-checked) ===")
small = clean[clean["patient"].isin(STAGE["patients"])].reset_index(drop=True)
print(f"small stage: {len(small)} records from {STAGE['patients']}, patients present: {sorted(small['patient'].unique())}")
X_small = small[feat_cols_all].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
y_small = small["target"].to_numpy()
groups_small = small["patient"].to_numpy()

rows = eval_stage("small_validation", X_small, y_small, groups_small, STAGE)
metrics = pd.DataFrame(rows)
summary = metrics.groupby("model").agg(
    mean_balanced_accuracy=("balanced_accuracy", "mean"),
    mean_log_loss=("log_loss", "mean"),
    mean_roc_auc=("roc_auc", "mean"),
    mean_fit_seconds=("fit_seconds", "mean"),
    folds=("fold", "count"),
).reset_index()
print(summary.to_string(index=False))
assert set(summary["model"]) == {"kanboost", "hist_gbdt", "catboost", "xgboost"}
assert not metrics[["balanced_accuracy", "log_loss", "roc_auc"]].isna().any().any()
print("all 4 models ran, no patient leakage, no NaN -- eval_stage logic OK")

print("\n=== testing tiered_equations() ===")
X_df = clean[feat_cols_all].replace([np.inf, -np.inf], np.nan)
imputer = SimpleImputer(strategy="median")
X_imp = pd.DataFrame(imputer.fit_transform(X_df), columns=X_df.columns)
vt = VarianceThreshold(threshold=0.0)
X_vt = pd.DataFrame(vt.fit_transform(X_imp), columns=X_imp.columns[vt.get_support()])
skb = SelectKBest(f_classif, k=min(SELECT_K, X_vt.shape[1]))
skb.fit(X_vt, clean["target"].to_numpy())
X_selected = X_vt.loc[:, skb.get_support()]
print(f"equations cell: {X_selected.shape[1]} selected features (named)")

def build_and_fit_gam(X_train, y_train, seed):
    return KANBoostClassifier(gam=True, kan_hidden=1, n_estimators=STAGE["kanboost_estimators"],
                               kan_steps=STAGE["kanboost_steps"], early_stopping_rounds=None,
                               random_state=seed).fit(X_train, y_train)

t0 = time.perf_counter()
try:
    tiers = tiered_equations(build_and_fit_gam, X_selected, clean["target"].to_numpy(),
                              simple_max_terms=5, detailed_max_terms=12, n_seeds=3, random_state=RANDOM_STATE)
    print(f"tiered_equations() completed in {time.perf_counter()-t0:.1f}s")
    for tier_name in ["simple", "detailed", "full"]:
        tier = tiers[tier_name]
        print(f"--- {tier_name} ({len(tier['kept_features'])} terms) ---")
        print("formula:", str(tier["formula"])[:200])
        print("fidelity:", tier["fidelity"])
except ValueError as exc:
    print(f"No stable equation (this is a valid outcome, not a crash): {exc}")

print("\nCC-13 notebook logic validated OK")
