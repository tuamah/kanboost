"""
Tests for kanboost.pipeline (KANBoostPipeline).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from kanboost.core.config import KANBoostConfig
from kanboost.pipeline import KANBoostPipeline, PipelineResult


def _binary_splits(seed=0):
    X, y = make_classification(n_samples=400, n_features=8, random_state=seed)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    X_tr, X_cal, y_tr, y_cal = train_test_split(X_tr, y_tr, test_size=0.3, stratify=y_tr, random_state=seed)
    return X_tr, X_cal, X_te, y_tr, y_cal, y_te


def test_fit_returns_a_fitted_model():
    X_tr, X_cal, X_te, y_tr, y_cal, y_te = _binary_splits()
    config = KANBoostConfig.from_flat(n_estimators=15, kan_steps=10, random_state=0)
    pipeline = KANBoostPipeline(config, task="classification")
    result = pipeline.fit(X_tr, y_tr)

    assert isinstance(result, PipelineResult)
    assert result.calibrated_model is None
    assert result.symbolic_model is None
    preds = result.model.predict_proba(X_te)
    assert preds.shape == (len(X_te), 2)


def test_fast_uses_fast_fit():
    X_tr, X_cal, X_te, y_tr, y_cal, y_te = _binary_splits()
    config = KANBoostConfig.from_flat(n_estimators=15, kan_steps=10, random_state=0)
    pipeline = KANBoostPipeline(config, task="classification", fast=True)
    result = pipeline.fit(X_tr, y_tr)
    assert result.model.predict_proba(X_te).shape == (len(X_te), 2)


def test_calibrate_requires_cal_data():
    X_tr, X_cal, X_te, y_tr, y_cal, y_te = _binary_splits()
    config = KANBoostConfig.from_flat(n_estimators=10, kan_steps=8, random_state=0)
    pipeline = KANBoostPipeline(config, task="classification", calibrate=True)
    try:
        pipeline.fit(X_tr, y_tr)
        raise AssertionError("calibrate=True without X_cal/y_cal was not rejected")
    except ValueError:
        pass


def test_calibrate_produces_calibrated_model():
    X_tr, X_cal, X_te, y_tr, y_cal, y_te = _binary_splits()
    config = KANBoostConfig.from_flat(n_estimators=15, kan_steps=10, random_state=0)
    pipeline = KANBoostPipeline(config, task="classification", calibrate=True)
    result = pipeline.fit(X_tr, y_tr, X_cal=X_cal, y_cal=y_cal)

    assert result.calibrated_model is not None
    cal_proba = result.calibrated_model.predict_proba(X_te)
    assert cal_proba.shape == (len(X_te), 2)


def test_export_symbolic_requires_gam_true():
    X_tr, X_cal, X_te, y_tr, y_cal, y_te = _binary_splits()
    config = KANBoostConfig.from_flat(n_estimators=10, kan_steps=8, random_state=0, gam=False)
    pipeline = KANBoostPipeline(config, task="classification", export_symbolic=True)
    try:
        pipeline.fit(X_tr, y_tr)
        raise AssertionError("export_symbolic=True with gam=False was not rejected")
    except ValueError:
        pass


def test_export_symbolic_produces_symbolic_model():
    X_tr, X_cal, X_te, y_tr, y_cal, y_te = _binary_splits()
    config = KANBoostConfig.from_flat(
        n_estimators=15, kan_steps=10, kan_hidden=1, gam=True, random_state=0,
    )
    pipeline = KANBoostPipeline(config, task="classification", export_symbolic=True)
    result = pipeline.fit(X_tr, y_tr)

    assert result.symbolic_model is not None
    raw_score = result.symbolic_model.predict(X_te)
    assert raw_score.shape == (len(X_te),)


def test_regression_task():
    X, y = make_regression(n_samples=300, n_features=5, random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
    config = KANBoostConfig.from_flat(n_estimators=10, kan_steps=8, random_state=0)
    pipeline = KANBoostPipeline(config, task="regression")
    result = pipeline.fit(X_tr, y_tr)
    preds = result.model.predict(X_te)
    assert preds.shape == (len(X_te),)


def test_invalid_task_rejected():
    config = KANBoostConfig.from_flat(n_estimators=10)
    try:
        KANBoostPipeline(config, task="not_a_real_task")
        raise AssertionError("invalid task was not rejected")
    except ValueError:
        pass


if __name__ == "__main__":
    test_fit_returns_a_fitted_model()
    test_fast_uses_fast_fit()
    test_calibrate_requires_cal_data()
    test_calibrate_produces_calibrated_model()
    test_export_symbolic_requires_gam_true()
    test_export_symbolic_produces_symbolic_model()
    test_regression_task()
    test_invalid_task_rejected()
    print("All pipeline tests passed.")
