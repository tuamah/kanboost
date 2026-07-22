"""
Tests for kanboost.editing (consolidate() / EditableGAM).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from kanboost import KANBoostClassifier, KANBoostRegressor
from kanboost.interpret.editing import consolidate


def test_consolidate_requires_gam():
    X, y = make_regression(n_samples=80, n_features=3, random_state=0)
    X_df = pd.DataFrame(X, columns=["a", "b", "c"])
    model = KANBoostRegressor(n_estimators=3, kan_steps=3, early_stopping_rounds=None, random_state=0)
    model.fit(X_df, y)
    try:
        consolidate(model)
        raise AssertionError("consolidate() on a non-gam model was not rejected")
    except ValueError:
        pass


def test_consolidate_does_not_trigger_std_degrees_of_freedom_warning():
    # Real bug this guards against: the baseline probe used to be
    # evaluated with a batch of exactly 1 row (torch.zeros((1, n))),
    # which made pykan's own internal activation-scale bookkeeping
    # compute torch.std() over a size-1 dimension -- warning on every
    # single consolidate() call (and everything built on it: symbolic
    # export, the dashboard, etc.). Verified separately that duplicating
    # the all-zero row to a batch of 2 leaves the actual value bit-for-bit
    # identical; only the warning disappears.
    import warnings

    X, y = make_regression(n_samples=80, n_features=3, random_state=0)
    X_df = pd.DataFrame(X, columns=["a", "b", "c"])
    model = KANBoostRegressor(
        n_estimators=5, kan_steps=8, kan_hidden=1, gam=True,
        early_stopping_rounds=None, random_state=0,
    )
    model.fit(X_df, y)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        consolidate(model)

    dof_warnings = [w for w in caught if "degrees of freedom" in str(w.message)]
    assert not dof_warnings


def test_consolidate_regressor_parity():
    X, y = make_regression(n_samples=200, n_features=4, noise=0.1, random_state=0)
    X_df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    model = KANBoostRegressor(
        n_estimators=10, kan_steps=8, kan_hidden=1, gam=True,
        early_stopping_rounds=None, random_state=0,
    )
    model.fit(X_df, y)

    gam = consolidate(model, resolution=300, grid=15, k=3)
    assert gam.max_consolidation_error() < 0.05

    orig = model.predict(X_df)
    edited = gam.predict(X_df)
    assert np.max(np.abs(orig - edited)) < 0.05

    # raw array input must match DataFrame input
    assert np.allclose(gam.predict(X_df.values), gam.predict(X_df))


def test_consolidate_binary_classifier_parity():
    X, y = make_classification(n_samples=200, n_features=5, random_state=0)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    model = KANBoostClassifier(
        n_estimators=8, kan_steps=8, kan_hidden=1, gam=True,
        early_stopping_rounds=None, random_state=0,
    )
    model.fit(X_df, y)

    gam = consolidate(model, resolution=300, grid=15, k=3)
    orig = model.predict_proba(X_df)[:, 1]
    edited = gam.predict_proba(X_df)[:, 1]
    assert np.max(np.abs(orig - edited)) < 0.05


def test_consolidate_multiclass_returns_dict():
    X, y = make_classification(
        n_samples=200, n_features=5, n_informative=4, n_redundant=0,
        n_classes=3, n_clusters_per_class=1, random_state=1,
    )
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    model = KANBoostClassifier(
        n_estimators=6, kan_steps=6, kan_hidden=1, gam=True,
        early_stopping_rounds=None, random_state=0,
    )
    model.fit(X_df, y)

    gams = consolidate(model)
    assert isinstance(gams, dict)
    assert set(gams.keys()) == set(model.classes_)
    for g in gams.values():
        assert g.max_consolidation_error() < 0.1


def test_edit_locality_and_reset():
    X, y = make_regression(n_samples=150, n_features=4, noise=0.1, random_state=0)
    X_df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    model = KANBoostRegressor(
        n_estimators=8, kan_steps=6, kan_hidden=1, gam=True,
        early_stopping_rounds=None, random_state=0,
    )
    model.fit(X_df, y)
    gam = consolidate(model)

    before = {f: gam.curves[f].copy() for f in gam.feature_names}
    gam.set_offset("a", (-0.5, 0.5), delta=3.0)
    assert not np.allclose(gam.curves["a"], before["a"])
    for f in gam.feature_names:
        if f != "a":
            assert np.allclose(gam.curves[f], before[f])

    gam.reset("a")
    assert np.allclose(gam.curves["a"], before["a"])


def test_diff_reports_metric_change():
    X, y = make_regression(n_samples=150, n_features=3, noise=0.1, random_state=0)
    X_df = pd.DataFrame(X, columns=["a", "b", "c"])
    model = KANBoostRegressor(
        n_estimators=6, kan_steps=6, kan_hidden=1, gam=True,
        early_stopping_rounds=None, random_state=0,
    )
    model.fit(X_df, y)
    gam = consolidate(model)

    d0 = gam.diff(X_df, y)
    assert d0["mean_abs_score_delta"] == 0.0  # no edits yet

    gam.set_offset("a", (-1.0, 1.0), delta=5.0)
    d1 = gam.diff(X_df, y)
    assert d1["per_feature_max_delta"]["a"] > 0
    assert d1["per_feature_max_delta"]["b"] == 0
    assert d1["mean_abs_score_delta"] > 0
    assert d1["metric"] == "rmse"
    assert d1["metric_after"] > d1["metric_before"]  # a +5 offset should hurt RMSE


def test_enforce_monotone_holds_between_knots():
    X, y = make_regression(n_samples=150, n_features=3, noise=0.1, random_state=0)
    X_df = pd.DataFrame(X, columns=["a", "b", "c"])
    model = KANBoostRegressor(
        n_estimators=8, kan_steps=6, kan_hidden=1, gam=True,
        early_stopping_rounds=None, random_state=0,
    )
    model.fit(X_df, y)
    gam = consolidate(model)

    rng = np.random.RandomState(0)
    gam.curves["b"] = gam.curves["b"] + rng.normal(scale=5.0, size=len(gam.curves["b"]))
    gam._invalidate("b")
    gam.enforce_monotone("b", sign=1)

    fine_x = np.linspace(-1, 1, 1000)
    fine_curve = gam.curve_at("b", fine_x)
    assert np.diff(fine_curve).min() >= -1e-3  # monotone even between the resolution grid's sample points

    try:
        gam.enforce_monotone("b", sign=2)
        raise AssertionError("invalid sign was not rejected")
    except ValueError:
        pass


def test_enforce_monotone_decreasing():
    """sign=-1 must produce a genuinely non-increasing curve (not crash,
    and not accidentally enforce the opposite direction)."""
    X, y = make_regression(n_samples=150, n_features=3, noise=0.1, random_state=0)
    X_df = pd.DataFrame(X, columns=["a", "b", "c"])
    model = KANBoostRegressor(
        n_estimators=8, kan_steps=6, kan_hidden=1, gam=True,
        early_stopping_rounds=None, random_state=0,
    )
    model.fit(X_df, y)
    gam = consolidate(model)

    rng = np.random.RandomState(0)
    gam.curves["c"] = gam.curves["c"] + rng.normal(scale=5.0, size=len(gam.curves["c"]))
    gam._invalidate("c")
    gam.enforce_monotone("c", sign=-1)

    fine_x = np.linspace(-1, 1, 1000)
    fine_curve = gam.curve_at("c", fine_x)
    assert np.diff(fine_curve).max() <= 1e-3  # non-increasing: no positive jumps, even between knots


def test_save_load_roundtrip_preserves_edits(tmp_path):
    X, y = make_regression(n_samples=150, n_features=3, noise=0.1, random_state=0)
    X_df = pd.DataFrame(X, columns=["a", "b", "c"])
    model = KANBoostRegressor(
        n_estimators=8, kan_steps=6, kan_hidden=1, gam=True,
        early_stopping_rounds=None, random_state=0,
    )
    model.fit(X_df, y)
    gam = consolidate(model)

    gam.set_offset("a", (-0.5, 0.5), delta=2.0)
    rng = np.random.RandomState(0)
    gam.curves["b"] = gam.curves["b"] + rng.normal(scale=5.0, size=len(gam.curves["b"]))
    gam._invalidate("b")
    gam.enforce_monotone("b", sign=1)

    path = str(tmp_path / "editable.pt")
    gam.save(path)
    loaded = type(gam).load(path)

    assert np.allclose(loaded.predict(X_df), gam.predict(X_df), atol=1e-3)

    fine_x = np.linspace(-1, 1, 1000)
    loaded_curve = loaded.curve_at("b", fine_x)
    assert np.diff(loaded_curve).min() >= -1e-3  # monotonicity survives the round-trip


def test_consolidated_predict_is_much_faster_than_the_ensemble():
    """consolidate() isn't just an editing tool -- for a gam=True model,
    EditableGAM.predict() is also a fast, high-fidelity predict path:
    one spline evaluation per feature instead of n_estimators full KAN
    forward passes. This is the documented answer to KANBoost's slow
    prediction time for GAM-mode models specifically (see README's
    "Honest limitations")."""
    import time

    X, y = make_regression(n_samples=1000, n_features=6, noise=0.1, random_state=0)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    # n_estimators=200, not 40: CX-13/CX-20's cached-forward path (see
    # kanboost/core/base.py::_raw_score_chain) made the *ensemble* GAM
    # predict path itself faster too, so the two paths' gap only stays a
    # clear >=3x at a large-enough ensemble -- consolidate()'s one-
    # spline-per-feature cost is ~constant in n_estimators, while the
    # ensemble path (even cached) still does real per-learner work.
    model = KANBoostRegressor(
        n_estimators=200, kan_steps=8, kan_hidden=1, gam=True,
        early_stopping_rounds=None, random_state=0,
    )
    model.fit(X_df, y)
    gam = consolidate(model)

    model.predict(X_df.iloc[:5])  # warm up (lazy CUDA init etc.)
    gam.predict(X_df.iloc[:5])

    t0 = time.time()
    model.predict(X_df)
    ensemble_time = time.time() - t0

    t0 = time.time()
    gam.predict(X_df)
    gam_time = time.time() - t0

    assert gam.max_consolidation_error() < 1e-3  # fidelity cost is negligible
    assert gam_time < ensemble_time / 3  # a real, large speedup, not a fluke


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    test_consolidate_requires_gam()
    test_consolidate_does_not_trigger_std_degrees_of_freedom_warning()
    test_consolidate_regressor_parity()
    test_consolidate_binary_classifier_parity()
    test_consolidate_multiclass_returns_dict()
    test_edit_locality_and_reset()
    test_diff_reports_metric_change()
    test_enforce_monotone_holds_between_knots()
    test_enforce_monotone_decreasing()
    with tempfile.TemporaryDirectory() as d:
        test_save_load_roundtrip_preserves_edits(Path(d))
    test_consolidated_predict_is_much_faster_than_the_ensemble()
    print("All editing tests passed.")
