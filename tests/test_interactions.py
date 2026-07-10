"""
Tests for kanboost.interactions (friedman_h() / check_additive_sufficiency()).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.datasets import load_breast_cancer

from kanboost import KANBoostClassifier
from kanboost.interactions import friedman_h, check_additive_sufficiency


class _FakeInteractionModel(BaseEstimator, RegressorMixin):
    """predict() = a*b + 0.1*c -- an exact, known interaction between a
    and b, and none between a/c or b/c. Used to sanity-check the
    H-statistic formula itself, independent of any real model's own
    fitting noise (verified during development: a real RandomForest
    showed a spuriously elevated H of 0.5+ for genuinely non-interacting
    pairs, purely from partial-dependence estimation noise -- this
    fake, exact model has no such noise)."""

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def predict(self, X):
        X = np.asarray(X)
        return X[:, 0] * X[:, 1] + 0.1 * X[:, 2]


def _known_interaction_data(n=500, seed=0):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "a": rng.uniform(-1, 1, n),
        "b": rng.uniform(-1, 1, n),
        "c": rng.uniform(-1, 1, n),
    })


def test_friedman_h_detects_known_interaction_and_rejects_known_non_interaction():
    X = _known_interaction_data()
    model = _FakeInteractionModel().fit(X)

    result = friedman_h(model, X, sample_size=300)
    h_by_pair = {frozenset(k): v for k, v in result["pairwise"].items()}

    assert h_by_pair[frozenset(("a", "b"))] > 0.9   # true interaction
    assert h_by_pair[frozenset(("a", "c"))] < 0.1    # no interaction
    assert h_by_pair[frozenset(("b", "c"))] < 0.1    # no interaction

    # ranked is sorted descending by H
    ranked_h = [h for _, _, h in result["ranked"]]
    assert ranked_h == sorted(ranked_h, reverse=True)
    assert result["ranked"][0][2] > 0.9


def test_friedman_h_requires_dataframe():
    model = _FakeInteractionModel().fit(_known_interaction_data())
    try:
        friedman_h(model, np.zeros((10, 3)))
        raise AssertionError("non-DataFrame X was not rejected")
    except TypeError:
        pass


def test_friedman_h_requires_at_least_two_features():
    X = _known_interaction_data()
    model = _FakeInteractionModel().fit(X)
    try:
        friedman_h(model, X, features=["a"])
        raise AssertionError("a single feature was not rejected")
    except ValueError:
        pass


def _small_breast_cancer():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=[c.replace(" ", "_") for c in data.feature_names])
    return X, data.target


def test_check_additive_sufficiency_requires_gam_true():
    X, y = _small_breast_cancer()
    model = KANBoostClassifier(
        n_estimators=10, kan_steps=8, kan_hidden=1, gam=False,
        early_stopping_rounds=None, random_state=0, verbose=False,
    )
    model.fit(X, y)
    try:
        check_additive_sufficiency(model, X, y)
        raise AssertionError("a gam=False model was not rejected")
    except ValueError:
        pass


def test_check_additive_sufficiency_rejects_top_n_below_2():
    X, y = _small_breast_cancer()
    model = KANBoostClassifier(
        n_estimators=5, kan_steps=5, kan_hidden=1, gam=True,
        early_stopping_rounds=None, random_state=0, verbose=False,
    )
    model.fit(X, y)
    for bad in (0, 1):
        try:
            check_additive_sufficiency(model, X, y, top_n=bad)
            raise AssertionError(f"top_n={bad} was not rejected")
        except ValueError:
            pass


def test_check_additive_sufficiency_filters_transformed_feature_names():
    # A real bug this guards against: feature_importances_dict() can
    # return names not present as raw X columns (e.g. a "<col>_missing"
    # indicator from categorical/missing-value encoding) -- these used
    # to reach friedman_h() unfiltered and crash with a bare KeyError
    # inside partial_dependence() instead of being skipped.
    X, y = _small_breast_cancer()
    model = KANBoostClassifier(
        n_estimators=5, kan_steps=5, kan_hidden=1, gam=True,
        early_stopping_rounds=None, random_state=0, verbose=False,
    )
    model.fit(X, y)

    real_importances = model.feature_importances_dict()
    top_real = list(real_importances.keys())[:3]

    def fake_importances():
        # Interleave two names that don't exist in X with real ones,
        # highest "importance" first, matching feature_importances_dict()'s
        # own descending convention.
        return {
            "not_a_real_column": 999.0,
            top_real[0]: real_importances[top_real[0]],
            "another_fake_column": 998.0,
            top_real[1]: real_importances[top_real[1]],
            top_real[2]: real_importances[top_real[2]],
        }

    model.feature_importances_dict = fake_importances
    result = check_additive_sufficiency(
        model, X, y, top_n=2, grid_resolution=6, sample_size=60,
    )
    checked = set(result["features_checked"])
    assert checked == {top_real[0], top_real[1]}
    assert "not_a_real_column" not in checked


def test_check_additive_sufficiency_returns_well_formed_report():
    X, y = _small_breast_cancer()
    model = KANBoostClassifier(
        n_estimators=10, kan_steps=8, kan_hidden=1, gam=True,
        early_stopping_rounds=None, random_state=0, verbose=False,
    )
    model.fit(X, y)

    result = check_additive_sufficiency(
        model, X, y, top_n=3, threshold=0.1, grid_resolution=6, sample_size=60,
    )

    assert result["verdict"] in {"additive_sufficient", "interactions_detected"}
    assert len(result["features_checked"]) == 3
    assert len(result["pairwise"]) == 3  # C(3,2)

    # exceeds_threshold must be exactly h_flexible > threshold, and the
    # overall verdict must be consistent with the per-pair flags -- not
    # two independently-computed values that could drift apart.
    for row in result["pairwise"]:
        assert row["exceeds_threshold"] == (row["h_flexible"] > result["threshold"])
    assert (result["verdict"] == "interactions_detected") == any(
        row["exceeds_threshold"] for row in result["pairwise"]
    )

    # ranked descending by h_flexible
    flex_values = [row["h_flexible"] for row in result["pairwise"]]
    assert flex_values == sorted(flex_values, reverse=True)

    # the refit counterpart is genuinely gam=False, and a real fitted model
    assert result["flexible_model"].get_params()["gam"] is False
    assert result["flexible_model"] is not model


if __name__ == "__main__":
    test_friedman_h_detects_known_interaction_and_rejects_known_non_interaction()
    test_friedman_h_requires_dataframe()
    test_friedman_h_requires_at_least_two_features()
    test_check_additive_sufficiency_requires_gam_true()
    test_check_additive_sufficiency_rejects_top_n_below_2()
    test_check_additive_sufficiency_filters_transformed_feature_names()
    test_check_additive_sufficiency_returns_well_formed_report()
    print("All interactions tests passed.")
