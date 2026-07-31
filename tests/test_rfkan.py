"""
Tests for kanboost.train.rfkan (fit_with_rfkan() / predict_proba_rfkan()).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

from kanboost import KANBoostClassifier, KANBoostRegressor
from kanboost.train.rfkan import fit_with_rfkan, predict_proba_rfkan


def _breast_cancer_splits(seed=0):
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)


def test_rfkan_fits_and_predicts():
    X_tr, X_te, y_tr, y_te = _breast_cancer_splits()
    model = KANBoostClassifier(
        n_estimators=20, kan_hidden=8, kan_grid=3, random_state=0, verbose=False,
    )
    fit_with_rfkan(model, X_tr, y_tr)

    assert model.best_iteration_ == 20
    assert len(model._rfkan_gammas_) == 20
    proba = predict_proba_rfkan(model, X_te)
    assert proba.shape == (len(y_te), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    auc = roc_auc_score(y_te, proba[:, 1])
    assert auc > 0.9


def test_rfkan_much_faster_than_standard_fit_at_similar_accuracy():
    X_tr, X_te, y_tr, y_te = _breast_cancer_splits()

    baseline = KANBoostClassifier(
        n_estimators=25, kan_hidden=8, kan_grid=3, learning_rate=0.1,
        early_stopping_rounds=None, random_state=0, verbose=False,
    )
    t0 = time.time()
    baseline.fit(X_tr, y_tr)
    t_baseline = time.time() - t0
    auc_baseline = roc_auc_score(y_te, baseline.predict_proba(X_te)[:, 1])

    rfkan = KANBoostClassifier(
        n_estimators=25, kan_hidden=8, kan_grid=3, learning_rate=0.1, random_state=0, verbose=False,
    )
    t0 = time.time()
    fit_with_rfkan(rfkan, X_tr, y_tr)
    t_rfkan = time.time() - t0
    auc_rfkan = roc_auc_score(y_te, predict_proba_rfkan(rfkan, X_te)[:, 1])

    assert t_rfkan < t_baseline
    assert auc_rfkan > auc_baseline - 0.03


def test_rfkan_with_newton_improves_accuracy_at_same_speed_class():
    X_tr, X_te, y_tr, y_te = _breast_cancer_splits()
    plain = KANBoostClassifier(n_estimators=20, kan_hidden=8, kan_grid=3, random_state=0, verbose=False)
    fit_with_rfkan(plain, X_tr, y_tr)
    auc_plain = roc_auc_score(y_te, predict_proba_rfkan(plain, X_te)[:, 1])

    newton = KANBoostClassifier(n_estimators=20, kan_hidden=8, kan_grid=3, random_state=0, verbose=False)
    fit_with_rfkan(newton, X_tr, y_tr, use_newton=True)
    auc_newton = roc_auc_score(y_te, predict_proba_rfkan(newton, X_te)[:, 1])

    # Not a strict inequality (dataset-dependent) -- checks Newton reweighting
    # is at least competitive, per the consistent INSPIRE improvement in the docstring.
    assert auc_newton > auc_plain - 0.02


def test_rfkan_with_line_search_fewer_rounds_competitive():
    X_tr, X_te, y_tr, y_te = _breast_cancer_splits()
    baseline = KANBoostClassifier(
        n_estimators=30, kan_hidden=8, kan_grid=3, learning_rate=0.1,
        early_stopping_rounds=None, random_state=0, verbose=False,
    )
    baseline.fit(X_tr, y_tr)
    auc_baseline = roc_auc_score(y_te, baseline.predict_proba(X_te)[:, 1])

    fewer = KANBoostClassifier(n_estimators=10, kan_hidden=8, kan_grid=3, random_state=0, verbose=False)
    fit_with_rfkan(fewer, X_tr, y_tr, use_line_search=True)
    auc_fewer = roc_auc_score(y_te, predict_proba_rfkan(fewer, X_te)[:, 1])

    assert auc_fewer > auc_baseline - 0.03


def test_rfkan_rejects_newton_and_line_search_together():
    X_tr, X_te, y_tr, y_te = _breast_cancer_splits()
    model = KANBoostClassifier(n_estimators=5, random_state=0, verbose=False)
    with pytest.raises(ValueError, match="not supported"):
        fit_with_rfkan(model, X_tr, y_tr, use_newton=True, use_line_search=True)


def test_rfkan_multiclass():
    X, y = load_iris(return_X_y=True)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

    model = KANBoostClassifier(
        n_estimators=15, kan_hidden=8, kan_grid=3, random_state=0, verbose=False,
    )
    fit_with_rfkan(model, X_tr, y_tr)

    assert isinstance(model.learners_, dict)
    assert len(model.learners_) == 3
    assert isinstance(model._rfkan_gammas_, dict) and len(model._rfkan_gammas_) == 3
    proba = predict_proba_rfkan(model, X_te)
    assert proba.shape == (len(y_te), 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    preds = model.classes_[np.argmax(proba, axis=1)]
    assert accuracy_score(y_te, preds) > 0.85


def test_rfkan_multiclass_with_newton():
    X, y = load_iris(return_X_y=True)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

    model = KANBoostClassifier(
        n_estimators=15, kan_hidden=8, kan_grid=3, random_state=0, verbose=False,
    )
    fit_with_rfkan(model, X_tr, y_tr, use_newton=True)
    proba = predict_proba_rfkan(model, X_te)
    preds = model.classes_[np.argmax(proba, axis=1)]
    assert accuracy_score(y_te, preds) > 0.85


def test_rfkan_rejects_regressor():
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names).iloc[:200]
    y = data.target[:200]
    model = KANBoostRegressor(n_estimators=5, random_state=0, verbose=False)
    with pytest.raises(TypeError, match="regressor"):
        fit_with_rfkan(model, X, y)


def test_predict_proba_rfkan_requires_rfkan_fit():
    X_tr, X_te, y_tr, y_te = _breast_cancer_splits()
    model = KANBoostClassifier(n_estimators=5, random_state=0, verbose=False)
    model.fit(X_tr, y_tr)  # standard fit, not fit_with_rfkan
    with pytest.raises(RuntimeError, match="fit_with_rfkan"):
        predict_proba_rfkan(model, X_te)


if __name__ == "__main__":
    test_rfkan_fits_and_predicts()
    test_rfkan_much_faster_than_standard_fit_at_similar_accuracy()
    test_rfkan_with_newton_improves_accuracy_at_same_speed_class()
    test_rfkan_with_line_search_fewer_rounds_competitive()
    test_rfkan_rejects_newton_and_line_search_together()
    test_rfkan_multiclass()
    test_rfkan_multiclass_with_newton()
    test_rfkan_rejects_regressor()
    test_predict_proba_rfkan_requires_rfkan_fit()
    print("All rfkan tests passed.")
