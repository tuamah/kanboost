"""
Parity + coverage tests for the CX-13/CX-20 ensemble-level forward-basis
cache in `_BaseKANBoost._raw_score_chain` (kanboost/core/base.py).

`_reference_raw_score_chain` reproduces the pre-cache per-learner loop
verbatim, so it serves as the parity oracle: any divergence between the
cached path (now the production `_raw_score_chain`) and this reference is
caught here rather than only in external Kaggle benchmarks.
"""
import os
import sys
import tempfile
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sklearn.datasets import make_classification, make_regression

from kanboost import KANBoostClassifier, KANBoostRegressor


def _reference_raw_score_chain(self, X_t, learners, init_pred, best_iteration):
    F = np.full(X_t.shape[0], init_pred)
    for learner in learners[:best_iteration]:
        with torch.no_grad():
            F += self.learning_rate * learner(X_t).cpu().numpy().flatten()
    return F


def _predict_with_reference_path(model, X):
    """Predict using the pre-CX-13 per-learner loop instead of the cache."""
    patched = model.__class__.__new__(model.__class__)
    patched.__dict__.update(model.__dict__)
    patched._raw_score_chain = types.MethodType(_reference_raw_score_chain, patched)
    return patched


def test_regressor_predict_parity_small_and_wide():
    for n_features, kan_hidden, n_estimators in [(8, 3, 10), (16, 6, 15)]:
        X, y = make_regression(n_samples=200, n_features=n_features, random_state=0)
        model = KANBoostRegressor(
            n_estimators=n_estimators, kan_hidden=kan_hidden, kan_steps=3,
            early_stopping_rounds=None, random_state=0,
        )
        model.fit(X, y)

        cached = model.predict(X)
        reference = _predict_with_reference_path(model, X).predict(X)

        assert np.max(np.abs(cached - reference)) <= 1e-10


def test_classifier_predict_proba_parity_binary():
    X, y = make_classification(n_samples=200, n_features=8, random_state=1)
    model = KANBoostClassifier(
        n_estimators=10, kan_hidden=3, kan_steps=3,
        early_stopping_rounds=None, random_state=0,
    )
    model.fit(X, y)

    cached_proba = model.predict_proba(X)
    reference_proba = _predict_with_reference_path(model, X).predict_proba(X)

    assert np.max(np.abs(cached_proba - reference_proba)) <= 1e-10
    assert np.array_equal(model.predict(X), _predict_with_reference_path(model, X).predict(X))


def test_classifier_predict_proba_parity_multiclass():
    X, y = make_classification(
        n_samples=240, n_features=8, n_classes=3, n_informative=6, random_state=2
    )
    model = KANBoostClassifier(
        n_estimators=8, kan_hidden=3, kan_steps=3,
        early_stopping_rounds=None, random_state=0,
    )
    model.fit(X, y)

    cached_proba = model.predict_proba(X)
    reference_proba = _predict_with_reference_path(model, X).predict_proba(X)

    assert cached_proba.shape[1] == 3
    assert np.max(np.abs(cached_proba - reference_proba)) <= 1e-10


def test_gam_identity_output_parity():
    X, y = make_regression(n_samples=150, n_features=6, random_state=3)
    model = KANBoostRegressor(
        n_estimators=12, kan_hidden=1, gam=True, kan_steps=3,
        early_stopping_rounds=None, random_state=0,
    )
    model.fit(X, y)

    cached = model.predict(X)
    reference = _predict_with_reference_path(model, X).predict(X)

    assert np.max(np.abs(cached - reference)) <= 1e-10


def test_early_stopped_model_parity():
    """`best_iteration_ < len(learners_)` is exactly the state early
    stopping leaves behind (see `_boost_chain`). Truncate it directly
    after a normal fit instead of depending on early stopping actually
    triggering on synthetic data, which is fixture-dependent and flaky.
    """
    X, y = make_regression(n_samples=200, n_features=6, random_state=4)
    model = KANBoostRegressor(
        n_estimators=20, kan_hidden=3, kan_steps=3,
        early_stopping_rounds=None, random_state=0,
    )
    model.fit(X, y)
    assert model.best_iteration_ == len(model.learners_)

    model.best_iteration_ = len(model.learners_) - 7
    reference_model = _predict_with_reference_path(model, X)

    cached = model.predict(X)
    reference = reference_model.predict(X)

    assert np.max(np.abs(cached - reference)) <= 1e-10


def test_save_load_roundtrip_predicts_identically():
    X, y = make_regression(n_samples=150, n_features=5, random_state=5)
    model = KANBoostRegressor(
        n_estimators=8, kan_hidden=3, kan_steps=3,
        early_stopping_rounds=None, random_state=0,
    )
    model.fit(X, y)
    before = model.predict(X)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.pkl")
        model.save(path)
        loaded = KANBoostRegressor.load(path)
        after = loaded.predict(X)

    assert np.max(np.abs(before - after)) <= 1e-10


def test_fallback_path_used_when_layer0_knots_differ():
    """Perturbing one learner's layer-0 knots must still give correct
    (reference-matching) output -- the mismatch should route through the
    original per-learner loop, not silently reuse a mismatched cached basis.
    """
    X, y = make_regression(n_samples=150, n_features=6, random_state=6)
    model = KANBoostRegressor(
        n_estimators=6, kan_hidden=3, kan_steps=3,
        early_stopping_rounds=None, random_state=0,
    )
    model.fit(X, y)

    reference_before = _predict_with_reference_path(model, X).predict(X)

    perturbed_knots = model.learners_[0].layers[0].knots + 1e-3
    model.learners_[0].layers[0].knots = perturbed_knots

    fallback_cached = model.predict(X)
    reference_after = _predict_with_reference_path(model, X).predict(X)

    assert np.max(np.abs(fallback_cached - reference_after)) <= 1e-10
    assert not np.allclose(reference_after, reference_before)


def test_cached_path_is_not_slower_on_a_wide_ensemble():
    X, y = make_regression(n_samples=1000, n_features=32, random_state=7)
    model = KANBoostRegressor(
        n_estimators=60, kan_hidden=16, kan_steps=3,
        early_stopping_rounds=None, random_state=0,
    )
    model.fit(X, y)
    reference_model = _predict_with_reference_path(model, X)

    model.predict(X)
    reference_model.predict(X)

    t0 = time.perf_counter()
    for _ in range(3):
        model.predict(X)
    cached_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(3):
        reference_model.predict(X)
    reference_time = time.perf_counter() - t0

    assert cached_time <= reference_time * 1.5, (
        f"cached path ({cached_time:.3f}s) should not be meaningfully "
        f"slower than the reference loop ({reference_time:.3f}s)"
    )
