"""
Tests for kanboost.train.linesearch (fit_with_line_search()).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from kanboost import KANBoostClassifier, KANBoostRegressor
from kanboost.train.linesearch import fit_with_line_search, predict_proba_line_search


def _breast_cancer_splits(seed=0):
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)


def test_line_search_fits_and_predicts():
    X_tr, X_te, y_tr, y_te = _breast_cancer_splits()
    model = KANBoostClassifier(
        n_estimators=15, kan_hidden=8, kan_grid=3, random_state=0, verbose=False,
    )
    fit_with_line_search(model, X_tr, y_tr)

    assert model.best_iteration_ == 15
    assert len(model._line_search_gammas_) == 15
    proba = predict_proba_line_search(model, X_te)
    assert proba.shape == (len(y_te), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    auc = roc_auc_score(y_te, proba[:, 1])
    assert auc > 0.9


def test_line_search_fewer_rounds_competitive_with_more_baseline_rounds():
    X_tr, X_te, y_tr, y_te = _breast_cancer_splits()

    baseline = KANBoostClassifier(
        n_estimators=30, kan_hidden=8, kan_grid=3, learning_rate=0.1,
        early_stopping_rounds=None, random_state=0, verbose=False,
    )
    baseline.fit(X_tr, y_tr)
    auc_baseline = roc_auc_score(y_te, baseline.predict_proba(X_te)[:, 1])

    fewer = KANBoostClassifier(
        n_estimators=10, kan_hidden=8, kan_grid=3, random_state=0, verbose=False,
    )
    fit_with_line_search(fewer, X_tr, y_tr)
    auc_fewer = roc_auc_score(y_te, predict_proba_line_search(fewer, X_te)[:, 1])

    # Not a strict inequality assertion (this is a lossy, dataset-dependent
    # technique) -- just checks the fewer-round line-search model is in the
    # same ballpark as a 3x-larger fixed-shrinkage ensemble, per the measured
    # INSPIRE results in the module docstring.
    assert auc_fewer > auc_baseline - 0.03


def test_line_search_rejects_multiclass():
    X, y = load_iris(return_X_y=True)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    model = KANBoostClassifier(n_estimators=5, random_state=0, verbose=False)
    with pytest.raises(ValueError, match="binary"):
        fit_with_line_search(model, X, y)


def test_line_search_rejects_regressor():
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names).iloc[:200]
    y = data.target[:200]
    model = KANBoostRegressor(n_estimators=5, random_state=0, verbose=False)
    with pytest.raises(TypeError, match="regressor"):
        fit_with_line_search(model, X, y)


def test_line_search_respects_sample_weight():
    X_tr, X_te, y_tr, y_te = _breast_cancer_splits()
    sw = np.where(y_tr == 1, 2.0, 1.0)
    model = KANBoostClassifier(n_estimators=10, kan_hidden=8, kan_grid=3, random_state=0, verbose=False)
    fit_with_line_search(model, X_tr, y_tr, sample_weight=sw)
    proba = predict_proba_line_search(model, X_te)
    assert np.all(np.isfinite(proba))


if __name__ == "__main__":
    test_line_search_fits_and_predicts()
    test_line_search_fewer_rounds_competitive_with_more_baseline_rounds()
    test_line_search_rejects_multiclass()
    test_line_search_rejects_regressor()
    test_line_search_respects_sample_weight()
    print("All linesearch tests passed.")
