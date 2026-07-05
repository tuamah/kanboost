"""
KANBoost vs. a plain tree-boosting baseline (sklearn's
HistGradientBoosting*) on three standard UCI-style datasets:

- Adult Income (classification, ~48K rows, mixed numeric/categorical)
- California Housing (regression, ~20K rows, numeric only)
- Breast Cancer Wisconsin (classification, ~570 rows, numeric only --
  the small-data regime, where KANBoost's smaller learner capacity per
  round is less of a handicap than on the two larger datasets)

This is a sanity-floor check, not a claim of parity with tuned
CatBoost/XGBoost/LightGBM: the goal is "KANBoost reaches competent
accuracy on real, non-toy data", plus a demonstration of a feature no
tree-boosting library has -- a hard monotonic constraint -- on Housing's
`MedInc` (median income), which should economically drive price upward.

Run: python examples/benchmark_uci.py
"""
import time

import numpy as np
from sklearn.datasets import fetch_california_housing, fetch_openml, load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from kanboost import KANBoostClassifier, KANBoostRegressor

RANDOM_STATE = 42


def run_adult_income():
    print("\n=== Adult Income (classification) ===")
    data = fetch_openml("adult", version=2, as_frame=True)
    df = data.frame.dropna()
    y = (df["class"] == ">50K").astype(int).values
    X = df.drop(columns=["class", "fnlwgt"])
    categorical_cols = [c for c in X.columns if str(X[c].dtype) == "category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    if len(X_train) > 10_000:
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train, train_size=10_000, random_state=RANDOM_STATE, stratify=y_train
        )

    kan_model = KANBoostClassifier(
        n_estimators=60, kan_hidden=1, kan_steps=15, learning_rate=0.3,
        early_stopping_rounds=10, validation_fraction=0.15, batch_size=2048,
        categorical_cols=categorical_cols, random_state=RANDOM_STATE,
    )
    t0 = time.time()
    kan_model.fit(X_train, y_train)
    kan_time = time.time() - t0
    kan_proba = kan_model.predict_proba(X_test)[:, 1]
    kan_auc = roc_auc_score(y_test, kan_proba)
    kan_acc = accuracy_score(y_test, kan_model.predict(X_test))

    X_train_enc = X_train.copy()
    X_test_enc = X_test.copy()
    for c in categorical_cols:
        X_train_enc[c] = X_train_enc[c].cat.codes
        X_test_enc[c] = X_test_enc[c].cat.codes
    baseline = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    t0 = time.time()
    baseline.fit(X_train_enc, y_train)
    baseline_time = time.time() - t0
    base_proba = baseline.predict_proba(X_test_enc)[:, 1]
    base_auc = roc_auc_score(y_test, base_proba)
    base_acc = accuracy_score(y_test, baseline.predict(X_test_enc))

    print(f"| Model | AUC | Accuracy | Train time |")
    print(f"|---|---|---|---|")
    print(f"| KANBoostClassifier (10K train rows) | {kan_auc:.4f} | {kan_acc:.4f} | {kan_time:.1f}s |")
    print(f"| HistGradientBoostingClassifier (full train) | {base_auc:.4f} | {base_acc:.4f} | {baseline_time:.1f}s |")
    print(f"Pass criterion (AUC > 0.85): {'PASS' if kan_auc > 0.85 else 'FAIL'}")
    return dict(kan_auc=kan_auc, kan_acc=kan_acc, kan_time=kan_time,
                base_auc=base_auc, base_acc=base_acc, base_time=baseline_time)


def run_california_housing():
    print("\n=== California Housing (regression) ===")
    data = fetch_california_housing(as_frame=True)
    X, y = data.data, data.target.values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    kan_model = KANBoostRegressor(
        n_estimators=60, kan_hidden=1, kan_steps=15, learning_rate=0.3,
        early_stopping_rounds=10, validation_fraction=0.15, batch_size=2048,
        random_state=RANDOM_STATE,
    )
    t0 = time.time()
    kan_model.fit(X_train, y_train)
    kan_time = time.time() - t0
    kan_pred = kan_model.predict(X_test)
    kan_rmse = float(np.sqrt(mean_squared_error(y_test, kan_pred)))
    kan_r2 = r2_score(y_test, kan_pred)

    baseline = HistGradientBoostingRegressor(random_state=RANDOM_STATE)
    t0 = time.time()
    baseline.fit(X_train, y_train)
    baseline_time = time.time() - t0
    base_pred = baseline.predict(X_test)
    base_rmse = float(np.sqrt(mean_squared_error(y_test, base_pred)))
    base_r2 = r2_score(y_test, base_pred)

    print(f"| Model | RMSE | R^2 | Train time |")
    print(f"|---|---|---|---|")
    print(f"| KANBoostRegressor | {kan_rmse:.4f} | {kan_r2:.4f} | {kan_time:.1f}s |")
    print(f"| HistGradientBoostingRegressor | {base_rmse:.4f} | {base_r2:.4f} | {baseline_time:.1f}s |")
    print(f"Pass criterion (R^2 > 0.6): {'PASS' if kan_r2 > 0.6 else 'FAIL'}")

    print("\n--- Monotonic constraint demo: MedInc (median income) -> price, should only increase ---")
    mono_model = KANBoostRegressor(
        n_estimators=40, kan_hidden=1, kan_steps=15, learning_rate=0.3,
        early_stopping_rounds=10, validation_fraction=0.15,
        gam=True, monotone_constraints={"MedInc": 1}, random_state=RANDOM_STATE,
    )
    mono_model.fit(X_train, y_train)
    deriv = mono_model.predict_derivative(X_test, "MedInc")
    print(f"min(d price / d MedInc) over test set = {deriv.min():.6f} "
          f"(must be >= 0 for a hard monotonic guarantee)")

    return dict(kan_rmse=kan_rmse, kan_r2=kan_r2, kan_time=kan_time,
                base_rmse=base_rmse, base_r2=base_r2, base_time=baseline_time)


def run_breast_cancer():
    print("\n=== Breast Cancer Wisconsin (classification, small data) ===")
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target.values  # 1 = benign, 0 = malignant

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    kan_model = KANBoostClassifier(
        n_estimators=60, kan_hidden=1, kan_steps=15, learning_rate=0.3,
        early_stopping_rounds=10, validation_fraction=0.15, random_state=RANDOM_STATE,
    )
    t0 = time.time()
    kan_model.fit(X_train, y_train)
    kan_time = time.time() - t0
    kan_proba = kan_model.predict_proba(X_test)[:, 1]
    kan_auc = roc_auc_score(y_test, kan_proba)
    kan_acc = accuracy_score(y_test, kan_model.predict(X_test))

    baseline = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    t0 = time.time()
    baseline.fit(X_train, y_train)
    baseline_time = time.time() - t0
    base_proba = baseline.predict_proba(X_test)[:, 1]
    base_auc = roc_auc_score(y_test, base_proba)
    base_acc = accuracy_score(y_test, baseline.predict(X_test))

    print(f"| Model | AUC | Accuracy | Train time |")
    print(f"|---|---|---|---|")
    print(f"| KANBoostClassifier | {kan_auc:.4f} | {kan_acc:.4f} | {kan_time:.1f}s |")
    print(f"| HistGradientBoostingClassifier | {base_auc:.4f} | {base_acc:.4f} | {baseline_time:.1f}s |")
    print(f"Pass criterion (AUC > 0.95): {'PASS' if kan_auc > 0.95 else 'FAIL'}")
    return dict(kan_auc=kan_auc, kan_acc=kan_acc, kan_time=kan_time,
                base_auc=base_auc, base_acc=base_acc, base_time=baseline_time)


if __name__ == "__main__":
    adult_results = run_adult_income()
    housing_results = run_california_housing()
    breast_cancer_results = run_breast_cancer()
