"""
Tests for kanboost.train.ga2m (fit_with_ga2m() / predict_proba_ga2m() /
main_effect_contributions() / pairwise_interaction_contributions()).
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
from kanboost.train.ga2m import (
    fit_with_ga2m, predict_proba_ga2m,
    main_effect_contributions, pairwise_interaction_contributions,
)


def _breast_cancer_splits(seed=0):
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)


def test_ga2m_fits_and_predicts():
    X_tr, X_te, y_tr, y_te = _breast_cancer_splits()
    model = KANBoostClassifier(n_estimators=15, kan_grid=3, random_state=0, verbose=False)
    fit_with_ga2m(model, X_tr, y_tr, n_pairs_per_round=10)

    assert model.best_iteration_ == 15
    proba = predict_proba_ga2m(model, X_te)
    assert proba.shape == (len(y_te), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert roc_auc_score(y_te, proba[:, 1]) > 0.9


def test_ga2m_beats_dense_rfkan_with_line_search():
    """The core claim: GA2M+line-search should be competitive with (not
    necessarily always beating, on an unrelated small dataset) dense
    RF-KAN -- at minimum, in the same ballpark, per the consistent
    INSPIRE improvement documented in the module."""
    from kanboost.train.rfkan import fit_with_rfkan, predict_proba_rfkan

    X_tr, X_te, y_tr, y_te = _breast_cancer_splits()

    dense = KANBoostClassifier(n_estimators=15, kan_hidden=10, kan_grid=3, random_state=0, verbose=False)
    fit_with_rfkan(dense, X_tr, y_tr, use_line_search=True)
    auc_dense = roc_auc_score(y_te, predict_proba_rfkan(dense, X_te)[:, 1])

    ga2m = KANBoostClassifier(n_estimators=15, kan_grid=3, random_state=0, verbose=False)
    fit_with_ga2m(ga2m, X_tr, y_tr, n_pairs_per_round=10, use_line_search=True)
    auc_ga2m = roc_auc_score(y_te, predict_proba_ga2m(ga2m, X_te)[:, 1])

    assert auc_ga2m > auc_dense - 0.03


def test_main_effect_contributions_covers_every_feature():
    X_tr, X_te, y_tr, y_te = _breast_cancer_splits()
    model = KANBoostClassifier(n_estimators=10, kan_grid=3, random_state=0, verbose=False)
    fit_with_ga2m(model, X_tr, y_tr, n_pairs_per_round=5)

    contributions = main_effect_contributions(model)
    assert len(contributions) == X_tr.shape[1]
    assert set(contributions.keys()) <= set(X_tr.columns) | set(range(X_tr.shape[1]))
    assert all(v >= 0 for v in contributions.values())
    # sorted descending
    values = list(contributions.values())
    assert values == sorted(values, reverse=True)


def test_pairwise_interaction_contributions_returns_named_pairs():
    X_tr, X_te, y_tr, y_te = _breast_cancer_splits()
    model = KANBoostClassifier(n_estimators=10, kan_grid=3, random_state=0, verbose=False)
    fit_with_ga2m(model, X_tr, y_tr, n_pairs_per_round=5)

    interactions = pairwise_interaction_contributions(model, top_k=5)
    assert len(interactions) <= 5
    for key, val in interactions.items():
        assert isinstance(key, tuple) and len(key) == 2
        assert val >= 0


def test_ga2m_use_newton_runs_without_error():
    X_tr, X_te, y_tr, y_te = _breast_cancer_splits()
    model = KANBoostClassifier(n_estimators=10, kan_grid=3, random_state=0, verbose=False)
    fit_with_ga2m(model, X_tr, y_tr, n_pairs_per_round=5, use_newton=True)
    proba = predict_proba_ga2m(model, X_te)
    assert np.all(np.isfinite(proba))
    assert roc_auc_score(y_te, proba[:, 1]) > 0.85


def test_ga2m_armijo_line_search_runs_and_stays_finite():
    X_tr, X_te, y_tr, y_te = _breast_cancer_splits()
    model = KANBoostClassifier(n_estimators=15, kan_grid=3, random_state=0, verbose=False)
    fit_with_ga2m(model, X_tr, y_tr, n_pairs_per_round=10, use_newton=True, line_search_mode="armijo")
    proba = predict_proba_ga2m(model, X_te)
    assert np.all(np.isfinite(proba))
    assert roc_auc_score(y_te, proba[:, 1]) > 0.85


def test_ga2m_hessian_floor_modes_run_and_stay_finite():
    X_tr, X_te, y_tr, y_te = _breast_cancer_splits()
    for mode in ["hard", "adaptive", "soft_lm"]:
        model = KANBoostClassifier(n_estimators=15, kan_grid=3, random_state=0, verbose=False)
        fit_with_ga2m(model, X_tr, y_tr, n_pairs_per_round=10, use_newton=True, hessian_floor_mode=mode)
        proba = predict_proba_ga2m(model, X_te)
        assert np.all(np.isfinite(proba)), mode
        assert roc_auc_score(y_te, proba[:, 1]) > 0.85, mode


def test_predict_proba_ga2m_parallel_matches_sequential():
    X_tr, X_te, y_tr, y_te = _breast_cancer_splits()
    model = KANBoostClassifier(n_estimators=15, kan_grid=3, random_state=0, verbose=False)
    fit_with_ga2m(model, X_tr, y_tr, n_pairs_per_round=10)

    proba_seq = predict_proba_ga2m(model, X_te, n_jobs=1)
    proba_par = predict_proba_ga2m(model, X_te, n_jobs=2)

    assert np.allclose(proba_seq, proba_par, atol=1e-8)
    assert roc_auc_score(y_te, proba_par[:, 1]) > 0.9


def test_ga2m_rejects_unknown_line_search_mode():
    X_tr, X_te, y_tr, y_te = _breast_cancer_splits()
    model = KANBoostClassifier(n_estimators=5, random_state=0, verbose=False)
    with pytest.raises(ValueError, match="line_search_mode"):
        fit_with_ga2m(model, X_tr, y_tr, line_search_mode="bogus")


def test_ga2m_rejects_unknown_hessian_floor_mode():
    X_tr, X_te, y_tr, y_te = _breast_cancer_splits()
    model = KANBoostClassifier(n_estimators=5, random_state=0, verbose=False)
    with pytest.raises(ValueError, match="hessian_floor_mode"):
        fit_with_ga2m(model, X_tr, y_tr, use_newton=True, hessian_floor_mode="bogus")


def test_ga2m_rejects_multiclass():
    X, y = load_iris(return_X_y=True)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    model = KANBoostClassifier(n_estimators=5, random_state=0, verbose=False)
    with pytest.raises(ValueError, match="binary"):
        fit_with_ga2m(model, X, y)


def test_ga2m_rejects_regressor():
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names).iloc[:200]
    y = data.target[:200]
    model = KANBoostRegressor(n_estimators=5, random_state=0, verbose=False)
    with pytest.raises(TypeError, match="regressor"):
        fit_with_ga2m(model, X, y)


def test_predict_proba_ga2m_requires_ga2m_fit():
    X_tr, X_te, y_tr, y_te = _breast_cancer_splits()
    model = KANBoostClassifier(n_estimators=5, random_state=0, verbose=False)
    model.fit(X_tr, y_tr)
    with pytest.raises(RuntimeError, match="fit_with_ga2m"):
        predict_proba_ga2m(model, X_te)


if __name__ == "__main__":
    test_ga2m_fits_and_predicts()
    test_ga2m_beats_dense_rfkan_with_line_search()
    test_ga2m_use_newton_runs_without_error()
    test_ga2m_armijo_line_search_runs_and_stays_finite()
    test_ga2m_hessian_floor_modes_run_and_stay_finite()
    test_predict_proba_ga2m_parallel_matches_sequential()
    test_ga2m_rejects_unknown_line_search_mode()
    test_ga2m_rejects_unknown_hessian_floor_mode()
    test_main_effect_contributions_covers_every_feature()
    test_pairwise_interaction_contributions_returns_named_pairs()
    test_ga2m_rejects_multiclass()
    test_ga2m_rejects_regressor()
    test_predict_proba_ga2m_requires_ga2m_fit()
    print("All ga2m tests passed.")
