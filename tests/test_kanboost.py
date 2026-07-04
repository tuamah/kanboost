"""
Basic sanity tests for KANBoostClassifier / KANBoostRegressor.
Run with: pytest tests/
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from kanboost import KANBoostClassifier, KANBoostRegressor


def test_classifier_fits_and_predicts():
    X, y = make_classification(n_samples=300, n_features=5, random_state=0)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])

    model = KANBoostClassifier(
        n_estimators=5, kan_steps=3, early_stopping_rounds=None, random_state=0
    )
    model.fit(X_df, y)

    proba = model.predict_proba(X_df)
    assert proba.shape == (300, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)

    preds = model.predict(X_df)
    assert set(np.unique(preds)).issubset({0, 1})


def test_classifier_with_categorical():
    X, y = make_classification(n_samples=200, n_features=4, random_state=1)
    X_df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    X_df["region"] = np.random.choice(["north", "south", "east"], size=200)

    model = KANBoostClassifier(
        n_estimators=3, kan_steps=3, early_stopping_rounds=None,
        categorical_cols=["region"], random_state=0,
    )
    model.fit(X_df, y)
    proba = model.predict_proba(X_df)
    assert proba.shape[0] == 200


def test_classifier_early_stopping():
    X, y = make_classification(n_samples=400, n_features=5, random_state=2)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    X_train, X_val = X_df.iloc[:300], X_df.iloc[300:]
    y_train, y_val = y[:300], y[300:]

    model = KANBoostClassifier(
        n_estimators=20, kan_steps=3, early_stopping_rounds=3, random_state=0
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    assert model.best_iteration_ <= 20
    assert len(model.learners_) >= model.best_iteration_


def test_feature_importances_sum_to_one():
    X, y = make_classification(n_samples=200, n_features=4, random_state=3)
    X_df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    model = KANBoostClassifier(
        n_estimators=4, kan_steps=3, early_stopping_rounds=None, random_state=0
    )
    model.fit(X_df, y)
    importances = model.feature_importances()
    assert importances.shape == (4,)
    assert np.isclose(importances.sum(), 1.0, atol=1e-3)


def test_regressor_fits_and_predicts():
    X, y = make_regression(n_samples=200, n_features=4, noise=0.1, random_state=0)
    X_df = pd.DataFrame(X, columns=["a", "b", "c", "d"])

    model = KANBoostRegressor(
        n_estimators=5, kan_steps=3, early_stopping_rounds=None, random_state=0
    )
    model.fit(X_df, y)
    preds = model.predict(X_df)
    assert preds.shape == (200,)


def test_sklearn_compatibility():
    """KANBoost estimators must work with sklearn's clone/CV machinery."""
    from sklearn.base import clone, is_classifier
    from sklearn.model_selection import cross_val_score

    X, y = make_classification(n_samples=200, n_features=4, random_state=5)
    X_df = pd.DataFrame(X, columns=["a", "b", "c", "d"])

    m = KANBoostClassifier(n_estimators=3, kan_steps=3, early_stopping_rounds=None)
    assert is_classifier(m)
    m2 = clone(m)
    assert m2.get_params() == m.get_params()

    scores = cross_val_score(m, X_df, y, cv=2, scoring="roc_auc")
    assert not np.isnan(scores).any()


def test_input_validation():
    X, y = make_classification(n_samples=100, n_features=5, random_state=6)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])

    # NaN in y must be rejected
    y_bad = y.astype(float).copy(); y_bad[0] = np.nan
    try:
        KANBoostClassifier(n_estimators=2, kan_steps=2).fit(X_df, y_bad)
        raise AssertionError("NaN y was not rejected")
    except ValueError:
        pass

    # invalid hyperparameters must be rejected
    try:
        KANBoostClassifier(learning_rate=5).fit(X_df, y)
        raise AssertionError("invalid learning_rate was not rejected")
    except ValueError:
        pass

    # predicting before fitting must be rejected
    try:
        KANBoostClassifier().predict(X_df)
        raise AssertionError("predict before fit was not rejected")
    except RuntimeError:
        pass


if __name__ == "__main__":
    test_classifier_fits_and_predicts()
    test_classifier_with_categorical()
    test_classifier_early_stopping()
    test_feature_importances_sum_to_one()
    test_regressor_fits_and_predicts()
    test_sklearn_compatibility()
    test_input_validation()
    print("All tests passed.")


