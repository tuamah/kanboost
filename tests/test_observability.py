"""
Tests for kanboost.observability -- verifies the observability layer
works without modifying (or depending on private internals surviving
unchanged in) _base.py/classifier.py/regressor.py beyond `_fit_learner`
existing and `verbose` being a settable attribute.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.datasets import make_classification, make_regression

from kanboost import KANBoostClassifier, KANBoostRegressor
from kanboost.observability import (
    time_predict,
    memory_snapshot,
    gpu_utilization_flag,
    capture_boosting_rounds,
)


def test_time_predict_returns_metrics():
    X, y = make_classification(n_samples=100, n_features=4, random_state=0)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    model = KANBoostClassifier(n_estimators=3, kan_steps=3, early_stopping_rounds=None, random_state=0)
    model.fit(X_df, y)

    preds, metrics = time_predict(model, X_df, method="predict_proba")
    assert preds.shape == (100, 2)
    assert metrics.elapsed_seconds >= 0
    assert metrics.n_samples == 100
    assert metrics.samples_per_second > 0
    assert metrics.device == "cpu"


def test_memory_snapshot_returns_a_value():
    snap = memory_snapshot()
    # Must resolve to a real number on every supported platform (Windows
    # ctypes path, Unix `resource` fallback, or psutil) -- None here would
    # mean every fallback silently failed.
    assert snap.rss_mb is not None
    assert snap.rss_mb > 0


def test_gpu_utilization_flag_shape():
    import torch

    X, y = make_regression(n_samples=50, n_features=3, random_state=0)
    X_df = pd.DataFrame(X, columns=["a", "b", "c"])
    model = KANBoostRegressor(n_estimators=2, kan_steps=2, random_state=0)
    model.fit(X_df, y)

    info = gpu_utilization_flag(model)
    assert info["cuda_available"] == torch.cuda.is_available()
    assert info["model_device"] == str(model.device_)
    assert info["model_on_gpu"] == str(model.device_).startswith("cuda")


def test_capture_boosting_rounds_binary_classifier():
    X, y = make_classification(n_samples=150, n_features=5, random_state=2)
    X_val, y_val = X[:30], y[:30]
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    X_val_df = pd.DataFrame(X_val, columns=[f"f{i}" for i in range(5)])

    model = KANBoostClassifier(n_estimators=5, kan_steps=3, early_stopping_rounds=None, random_state=0)
    with capture_boosting_rounds(model) as rounds:
        model.fit(X_df, y, eval_set=(X_val_df, y_val))

    assert len(rounds) == 5
    assert all(r.elapsed_seconds >= 0 for r in rounds)
    assert all(r.loss is not None for r in rounds)  # eval_set was given

    # verbose and _fit_learner must be restored to their original state
    assert model.verbose is False
    assert model._fit_learner.__func__ is type(model)._fit_learner


def test_capture_boosting_rounds_multiclass_and_regressor():
    X, y = make_classification(
        n_samples=150, n_features=5, n_informative=4, n_redundant=0,
        n_classes=3, n_clusters_per_class=1, random_state=1,
    )
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    model = KANBoostClassifier(n_estimators=3, kan_steps=3, early_stopping_rounds=None, random_state=0)
    with capture_boosting_rounds(model) as rounds:
        model.fit(X_df, y)
    # one-vs-rest: 3 classes x 3 estimators each
    assert len(rounds) == 9

    Xr, yr = make_regression(n_samples=100, n_features=3, random_state=0)
    Xr_df = pd.DataFrame(Xr, columns=["a", "b", "c"])
    reg = KANBoostRegressor(n_estimators=4, kan_steps=3, early_stopping_rounds=None, random_state=0)
    with capture_boosting_rounds(reg) as rounds_reg:
        reg.fit(Xr_df, yr)
    assert len(rounds_reg) == 4
    # no eval_set -> no val_loss to parse
    assert all(r.loss is None for r in rounds_reg)


def test_capture_boosting_rounds_restores_state_on_exception():
    X, y = make_regression(n_samples=50, n_features=3, random_state=0)
    X_df = pd.DataFrame(X, columns=["a", "b", "c"])
    model = KANBoostRegressor(n_estimators=2, kan_steps=2, random_state=0)

    try:
        with capture_boosting_rounds(model):
            raise ValueError("boom")
    except ValueError:
        pass

    assert model.verbose is False
    assert model._fit_learner.__func__ is type(model)._fit_learner


if __name__ == "__main__":
    test_time_predict_returns_metrics()
    test_memory_snapshot_returns_a_value()
    test_gpu_utilization_flag_shape()
    test_capture_boosting_rounds_binary_classifier()
    test_capture_boosting_rounds_multiclass_and_regressor()
    test_capture_boosting_rounds_restores_state_on_exception()
    print("All observability tests passed.")
