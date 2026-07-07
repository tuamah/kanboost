"""
Integration test between kanboost and its sibling tuning package,
kantun. Skipped entirely if kantun isn't installed (matching the
optional-dependency test pattern used for fastapi/streamlit elsewhere
in this suite) -- kantun is deliberately NOT a kanboost dependency (see
kanboost/__init__.py's docstring: tuning lives in a separate package so
kanboost's own dependency footprint stays minimal).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from kanboost import KANBoostClassifier, KANBoostRegressor

kantun = pytest.importorskip("kantun", reason="kantun not installed -- sibling package, optional")
from kantun import KantunSearch  # noqa: E402


def test_kantun_random_search_on_kanboost_classifier():
    X, y = make_classification(n_samples=200, n_features=5, random_state=0)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])

    param_space = {"n_estimators": [5, 8], "kan_hidden": [1, 3], "kan_steps": [3]}
    search = KantunSearch(
        KANBoostClassifier, param_space,
        n_iter=3, cv=2, scoring="auc", random_state=0, verbose=False,
    )
    search.fit(X_df, y)

    assert set(search.best_params_) == set(param_space)
    for k, v in search.best_params_.items():
        assert v in param_space[k]
    assert 0 <= search.best_score_ <= 1
    assert search.best_estimator_.predict_proba(X_df).shape == (200, 2)

    df = search.results_dataframe()
    assert list(df["mean_score"]) == sorted(df["mean_score"], reverse=True)
    for row in search.cv_results_:
        assert set(row) >= {"params", "mean_score", "std_score", "seconds"}


def test_kantun_random_search_on_kanboost_regressor():
    X, y = make_regression(n_samples=200, n_features=4, noise=0.1, random_state=0)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])

    search = KantunSearch(
        KANBoostRegressor, {"n_estimators": [5, 8], "kan_steps": [3]},
        n_iter=2, cv=2, scoring="neg_mse", random_state=0, verbose=False,
    )
    search.fit(X_df, y)

    assert search.best_estimator_ is not None
    preds = search.best_estimator_.predict(X_df)
    assert preds.shape == (200,)


def test_kantun_halving_on_kanboost_classifier():
    X, y = make_classification(n_samples=300, n_features=5, random_state=0)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])

    search = KantunSearch(
        KANBoostClassifier,
        {"n_estimators": [5, 10], "kan_hidden": [1, 3], "kan_steps": [3]},
        search_type="halving", n_iter=4, cv=2, scoring="auc",
        random_state=0, verbose=False, halving_factor=2, min_resource=20,
    )
    search.fit(X_df, y)
    assert search.best_estimator_ is not None
    assert all("rung" in r for r in search.cv_results_)


def test_kantun_parallel_on_kanboost_classifier():
    X, y = make_classification(n_samples=150, n_features=5, random_state=0)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])

    search = KantunSearch(
        KANBoostClassifier, {"n_estimators": [5, 8], "kan_steps": [3]},
        n_iter=2, cv=2, scoring="auc", random_state=0, verbose=False, n_jobs=2,
    )
    search.fit(X_df, y)
    assert search.best_estimator_ is not None
