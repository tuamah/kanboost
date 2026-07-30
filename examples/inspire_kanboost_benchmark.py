"""
Train KANBoost on INSPIRE for postoperative ICU admission prediction, at
three nested training scales, against XGBoost/LightGBM/CatBoost/
HistGradientBoosting.

This example expects a local copy of INSPIRE's `operations.csv` (Seoul
National University Hospital perioperative dataset, PhysioNet). It does
not download credentialed data.

Run:
    python examples/inspire_kanboost_benchmark.py --data-path /path/to/operations.csv

Applies the gap-closing configuration validated in
`docs/inspire_gap_closing_report.md`: native target-mean categorical
encoding (+ `categorical_hierarchy` backing `icd10_pcs` off to
`department`), class-balanced `sample_weight`, 2x weak-learner capacity,
post-hoc `consolidate_learners()`, `find_threshold`, and Platt
calibration -- instead of the naive One-Hot-encoding-with-no-imbalance-
handling baseline this configuration was benchmarked against.
"""
from __future__ import annotations

import argparse
import gc
import inspect
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    f1_score, log_loss, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight

from kanboost import KANBoostClassifier
from kanboost.train.imbalance import find_threshold
from kanboost.train.calibration import calibrate
from kanboost.train.consolidate import consolidate_learners

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

TARGET = "postop_icu"
BASE_FEATURES = ["age", "sex", "weight", "height", "race", "asa", "emop",
                 "department", "antype", "icd10_pcs"]
NUMERIC_FEATURES = ["age", "weight", "height", "asa", "emop"]
CATEGORICAL_FEATURES = ["sex", "race", "department", "antype", "icd10_pcs"]
RARE_MIN_COUNT = 50

KAN_CONFIGS = {
    "Small": dict(n_estimators=100, learning_rate=0.045, kan_hidden=16, kan_grid=6, kan_k=3,
                  kan_steps=50, kan_lr=0.02, batch_size=2048, lamb=1e-4, lamb_l1=1e-3,
                  lamb_coefdiff=1e-4, early_stopping_rounds=15, validation_fraction=0.15,
                  random_state=RANDOM_STATE, device="cpu", verbose=0),
    "Medium": dict(n_estimators=140, learning_rate=0.035, kan_hidden=20, kan_grid=8, kan_k=3,
                   kan_steps=60, kan_lr=0.015, batch_size=4096, lamb=1e-4, lamb_l1=1e-3,
                   lamb_coefdiff=1e-4, early_stopping_rounds=20, validation_fraction=0.15,
                   random_state=RANDOM_STATE, device="cpu", verbose=0),
    "Large": dict(n_estimators=180, learning_rate=0.025, kan_hidden=24, kan_grid=8, kan_k=3,
                  kan_steps=70, kan_lr=0.012, batch_size=4096, lamb=1e-4, lamb_l1=1e-3,
                  lamb_coefdiff=1e-4, early_stopping_rounds=25, validation_fraction=0.15,
                  random_state=RANDOM_STATE, device="cpu", verbose=0),
}


def _accepted_params(cls, params):
    sig = inspect.signature(cls.__init__)
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        return params
    return {k: v for k, v in params.items() if k in sig.parameters}


def _make_group_splits(data, target_col, group_col, random_state=42):
    y = data[target_col].to_numpy()
    groups = data[group_col].to_numpy()
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    trainval_idx, test_idx = next(outer.split(data, y, groups))
    trainval = data.iloc[trainval_idx].copy()
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=random_state + 1)
    train_rel, val_rel = next(inner.split(trainval, trainval[target_col].to_numpy(),
                                           trainval[group_col].to_numpy()))
    return trainval_idx[train_rel], trainval_idx[val_rel], test_idx


def _nested_group_samples(train_data, target_sizes, group_col, target_col, random_state=42):
    rng = np.random.default_rng(random_state)
    group_stats = (train_data.groupby(group_col, as_index=False)
                   .agg(n_rows=(target_col, "size"), positives=(target_col, "sum")))
    group_stats["has_positive"] = (group_stats["positives"] > 0).astype(int)
    pos_groups = group_stats[group_stats["has_positive"] == 1].sample(frac=1, random_state=random_state)
    neg_groups = group_stats[group_stats["has_positive"] == 0].sample(frac=1, random_state=random_state + 1)
    pos_records, neg_records = pos_groups.to_dict("records"), neg_groups.to_dict("records")
    p_ratio = len(pos_records) / max(len(group_stats), 1)
    ordered, i, j = [], 0, 0
    while i < len(pos_records) or j < len(neg_records):
        if rng.random() < p_ratio and i < len(pos_records):
            ordered.append(pos_records[i]); i += 1
        elif j < len(neg_records):
            ordered.append(neg_records[j]); j += 1
        elif i < len(pos_records):
            ordered.append(pos_records[i]); i += 1
    samples, selected, rows_so_far = {}, [], 0
    size_iter = iter(sorted(target_sizes))
    current = next(size_iter, None)
    for rec in ordered:
        selected.append(rec[group_col])
        rows_so_far += rec["n_rows"]
        while current is not None and rows_so_far >= current:
            samples[current] = train_data[train_data[group_col].isin(selected)].copy()
            current = next(size_iter, None)
        if current is None:
            break
    samples["large"] = train_data.copy()
    return samples


def _collapse_rare_categories(train_data, other_data, columns, min_count):
    train_copy = train_data.copy()
    other_copies = [part.copy() for part in other_data]
    for col in columns:
        counts = train_copy[col].value_counts(dropna=False)
        allowed = set(counts[counts >= min_count].index)
        train_copy[col] = train_copy[col].where(train_copy[col].isin(allowed), "__RARE__")
        for part in other_copies:
            part[col] = part[col].where(part[col].isin(allowed), "__RARE__")
    return train_copy, other_copies


def _evaluate(y_true, probabilities, threshold):
    pred = (probabilities >= threshold).astype(int)
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    return {
        "AUROC": roc_auc_score(y_true, probabilities),
        "AUPRC": average_precision_score(y_true, probabilities),
        "BalancedAccuracy": balanced_accuracy_score(y_true, pred),
        "F1": f1_score(y_true, pred, zero_division=0),
        "Brier": brier_score_loss(y_true, probabilities),
        "LogLoss": log_loss(y_true, clipped),
        "Threshold": threshold,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True,
                         help="Path to INSPIRE's operations.csv")
    parser.add_argument("--scales", default="Small,Medium,Large",
                         help="Comma-separated subset of Small,Medium,Large")
    parser.add_argument("--consolidate-group-size", type=int, default=5,
                         help="consolidate_learners() group size; 1 disables consolidation")
    parser.add_argument("--use-hierarchy", action="store_true", default=True,
                         help="Use categorical_hierarchy={'icd10_pcs': 'department'}")
    args = parser.parse_args()

    if not args.data_path.exists():
        raise FileNotFoundError(args.data_path)

    df_raw = pd.read_csv(args.data_path, low_memory=False)
    df = df_raw[["subject_id", "icuin_time", *BASE_FEATURES]].copy()
    df[TARGET] = df["icuin_time"].notna().astype("int8")
    df.drop(columns=["icuin_time"], inplace=True)
    for col in ["age", "weight", "height", "asa", "emop"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.loc[~df["age"].between(18, 90), "age"] = np.nan
    df.loc[~df["weight"].between(20, 300), "weight"] = np.nan
    df.loc[~df["height"].between(100, 230), "height"] = np.nan
    df.loc[~df["asa"].between(1, 6), "asa"] = np.nan
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("__MISSING__").astype(str)

    print(f"Rows: {len(df):,} | Patients: {df['subject_id'].nunique():,} "
          f"| Positive rate: {df[TARGET].mean():.2%}")

    train_idx, val_idx, test_idx = _make_group_splits(df, TARGET, "subject_id", RANDOM_STATE)
    train_pool = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    samples = _nested_group_samples(train_pool, (10_000, 50_000), "subject_id", TARGET, RANDOM_STATE)
    all_scale_data = {"Small": samples[10_000], "Medium": samples[50_000], "Large": samples["large"]}
    requested = set(args.scales.split(","))
    scale_data = {k: v for k, v in all_scale_data.items() if k in requested}

    X_val_raw, y_val = val_df[BASE_FEATURES].copy(), val_df[TARGET].to_numpy()
    X_test_raw, y_test = test_df[BASE_FEATURES].copy(), test_df[TARGET].to_numpy()

    results = []
    for scale_name, train_df in scale_data.items():
        print(f"\n{'='*70}\nSCALE: {scale_name} | rows={len(train_df):,}")
        X_train_raw = train_df[BASE_FEATURES].copy()
        y_train = train_df[TARGET].to_numpy()

        X_train, [X_val, X_test] = _collapse_rare_categories(
            X_train_raw, [X_val_raw, X_test_raw], CATEGORICAL_FEATURES, RARE_MIN_COUNT,
        )
        for col in NUMERIC_FEATURES:
            X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
            X_val[col] = pd.to_numeric(X_val[col], errors="coerce")
            X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

        sample_weight = compute_sample_weight("balanced", y_train)
        params = _accepted_params(KANBoostClassifier, KAN_CONFIGS[scale_name])
        params = dict(params, categorical_cols=CATEGORICAL_FEATURES)
        if args.use_hierarchy:
            params["categorical_hierarchy"] = {"icd10_pcs": "department"}
        model = KANBoostClassifier(**params)

        t0 = time.perf_counter()
        model.fit(X_train, y_train, sample_weight=sample_weight)
        fit_seconds = time.perf_counter() - t0
        n_before = model.best_iteration_

        if args.consolidate_group_size > 1:
            t0 = time.perf_counter()
            consolidate_learners(model, X_train, group_size=args.consolidate_group_size)
            print(f"  consolidated {n_before} -> {model.best_iteration_} learners "
                  f"in {time.perf_counter() - t0:.1f}s")

        cal_model = calibrate(model, X_val, y_val, method="platt")
        threshold = find_threshold(cal_model, X_val, y_val, metric="f1")

        t0 = time.perf_counter()
        p_test = cal_model.predict_proba(X_test)[:, 1]
        predict_seconds = time.perf_counter() - t0

        metrics = _evaluate(y_test, p_test, threshold)
        row = {
            "Scale": scale_name, "TrainRows": len(train_df),
            "LearnersBefore": n_before, "LearnersAfter": model.best_iteration_,
            "FitSeconds": fit_seconds, "PredictSeconds": predict_seconds, **metrics,
        }
        results.append(row)
        print(f"  KANBoost: fit={fit_seconds:.1f}s | AUROC={metrics['AUROC']:.4f} "
              f"| AUPRC={metrics['AUPRC']:.4f} | F1={metrics['F1']:.4f}")
        gc.collect()

    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))


if __name__ == "__main__":
    main()
