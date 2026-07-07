"""
Tests for kanboost.calibration (calibrate() / CalibratedKANBoost).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, accuracy_score

from kanboost import KANBoostClassifier
from kanboost.calibration import calibrate, CalibratedKANBoost


def _breast_cancer_splits(cal_size=0.5, seed=0):
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.5, random_state=seed, stratify=y
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_rest, y_rest, test_size=cal_size, random_state=seed, stratify=y_rest
    )
    return X_train, y_train, X_cal, y_cal, X_test, y_test


def _fit_base_model(X_train, y_train):
    model = KANBoostClassifier(
        n_estimators=30, kan_steps=10, early_stopping_rounds=None, random_state=0,
    )
    model.fit(X_train, y_train)
    return model


def test_platt_preserves_auc_and_improves_calibration():
    X_train, y_train, X_cal, y_cal, X_test, y_test = _breast_cancer_splits()
    model = _fit_base_model(X_train, y_train)

    raw_proba = model.predict_proba(X_test)[:, 1]
    cal_model = calibrate(model, X_cal, y_cal, method="platt")
    cal_proba = cal_model.predict_proba(X_test)[:, 1]

    assert abs(roc_auc_score(y_test, raw_proba) - roc_auc_score(y_test, cal_proba)) < 1e-9
    assert brier_score_loss(y_test, cal_proba) < brier_score_loss(y_test, raw_proba)
    assert log_loss(y_test, cal_proba) < log_loss(y_test, raw_proba)


def test_probabilities_are_valid():
    X_train, y_train, X_cal, y_cal, X_test, y_test = _breast_cancer_splits()
    model = _fit_base_model(X_train, y_train)
    cal_model = calibrate(model, X_cal, y_cal, method="platt")
    proba = cal_model.predict_proba(X_test)

    assert proba.shape == (len(X_test), 2)
    assert np.all(proba >= 0) and np.all(proba <= 1)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_isotonic_runs_and_improves_brier():
    X_train, y_train, X_cal, y_cal, X_test, y_test = _breast_cancer_splits(cal_size=0.6)
    model = _fit_base_model(X_train, y_train)

    raw_proba = model.predict_proba(X_test)[:, 1]
    cal_model = calibrate(model, X_cal, y_cal, method="isotonic")
    cal_proba = cal_model.predict_proba(X_test)[:, 1]

    assert brier_score_loss(y_test, cal_proba) < brier_score_loss(y_test, raw_proba)


def test_unknown_method_rejected():
    X_train, y_train, X_cal, y_cal, X_test, y_test = _breast_cancer_splits()
    model = _fit_base_model(X_train, y_train)
    try:
        calibrate(model, X_cal, y_cal, method="bogus")
        raise AssertionError("unknown method was not rejected")
    except ValueError:
        pass


def test_multiclass_rows_sum_to_one_and_accuracy_reasonable():
    X, y = load_iris(return_X_y=True)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    X_train, X_rest, y_train, y_rest = train_test_split(
        X_df, y, test_size=0.5, random_state=0, stratify=y
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, random_state=0, stratify=y_rest
    )

    model = KANBoostClassifier(
        n_estimators=10, kan_steps=8, early_stopping_rounds=None, random_state=0,
    )
    model.fit(X_train, y_train)

    raw_acc = accuracy_score(y_test, model.predict(X_test))
    cal_model = calibrate(model, X_cal, y_cal, method="platt")
    proba = cal_model.predict_proba(X_test)
    cal_acc = accuracy_score(y_test, cal_model.predict(X_test))

    assert np.allclose(proba.sum(axis=1), 1.0)
    assert cal_acc >= raw_acc - 0.1  # renormalization can shift argmax near boundaries


def test_save_load_roundtrip(tmp_path):
    X_train, y_train, X_cal, y_cal, X_test, y_test = _breast_cancer_splits()
    model = _fit_base_model(X_train, y_train)
    cal_model = calibrate(model, X_cal, y_cal, method="platt")

    path = str(tmp_path / "model.pt")
    cal_model.save(path)
    loaded = CalibratedKANBoost.load(path)

    assert np.allclose(loaded.predict_proba(X_test), cal_model.predict_proba(X_test))
    assert np.array_equal(loaded.predict(X_test), cal_model.predict(X_test))


def test_calibrate_requires_classifier():
    from kanboost import KANBoostRegressor
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=100, n_features=3, random_state=0)
    X_df = pd.DataFrame(X, columns=["a", "b", "c"])
    model = KANBoostRegressor(n_estimators=5, kan_steps=5, early_stopping_rounds=None, random_state=0)
    model.fit(X_df, y)
    try:
        calibrate(model, X_df, y)
        raise AssertionError("calibrate() on a regressor was not rejected")
    except ValueError:
        pass


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    test_platt_preserves_auc_and_improves_calibration()
    test_probabilities_are_valid()
    test_isotonic_runs_and_improves_brier()
    test_unknown_method_rejected()
    test_multiclass_rows_sum_to_one_and_accuracy_reasonable()
    with tempfile.TemporaryDirectory() as d:
        test_save_load_roundtrip(Path(d))
    test_calibrate_requires_classifier()
    print("All calibration tests passed.")
