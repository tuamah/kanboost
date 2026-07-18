"""
Direct tests for kanboost.core.kan (DeepKAN) — the numpy/scipy replacement for
pykan. These test the KAN interface itself, independent of the boosting loop
(see test_kanboost.py for ensemble-level integration coverage).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pytest

from kanboost.core.kan import KAN, fit_params, SYMBOLIC_LIB
from kanboost.core.regressor import KANBoostRegressor
from kanboost.interpret.editing import consolidate
from kanboost.interpret.symbolic import export_symbolic


def _r2(y, pred):
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def test_gam_exact():
    """A single closed-form P-spline solve should recover an additive
    function almost exactly (this is the whole point of DeepKAN's GAM mode)."""
    rng = np.random.RandomState(0)
    n = 500
    X = rng.uniform(-1, 1, (n, 2)).astype(np.float32)
    y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2

    kan = KAN(width=[2, 1, 1], grid=10, k=3, seed=1)
    kan.fix_symbolic(1, 0, 0, "x")

    X_t = torch.tensor(X)
    dataset = {"train_input": X_t, "train_label": torch.tensor(y, dtype=torch.float32)}
    kan.fit(dataset, steps=20, lamb=0.0001, lamb_l1=0.5, lamb_coefdiff=1.0)

    pred = kan(X_t).cpu().numpy().flatten()
    assert _r2(y, pred) > 0.99


def test_interface():
    """Compatibility surface the boosting loop and interpret pipeline rely on."""
    kan = KAN(width=[3, 1, 1], grid=5, k=3, seed=1)
    kan.fix_symbolic(1, 0, 0, "x")

    rng = np.random.RandomState(0)
    X = rng.uniform(-1, 1, (50, 3)).astype(np.float32)
    X_t = torch.tensor(X)
    y = torch.tensor(rng.randn(50), dtype=torch.float32)
    kan.fit({"train_input": X_t, "train_label": y}, steps=5)

    out = kan(X_t)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (50, 1)
    assert kan.act_fun[0].coef.shape[0] == 3
    assert kan.width[0][0] == 3


def test_state_dict_roundtrip():
    rng = np.random.RandomState(0)
    n = 200
    X = rng.uniform(-1, 1, (n, 2)).astype(np.float32)
    y = (X[:, 0] * 2 - X[:, 1]).astype(np.float32)

    kan = KAN(width=[2, 1, 1], grid=5, k=3, seed=1)
    kan.fix_symbolic(1, 0, 0, "x")
    X_t = torch.tensor(X)
    kan.fit({"train_input": X_t, "train_label": torch.tensor(y)}, steps=10)
    before = kan(X_t).cpu().numpy()

    sd = kan.state_dict()
    fresh = KAN(width=[2, 1, 1], grid=5, k=3, seed=1)
    fresh.fix_symbolic(1, 0, 0, "x")
    fresh.load_state_dict(sd)
    after = fresh(X_t).cpu().numpy()

    assert np.allclose(before, after)


def test_als_nongam():
    """Non-GAM (multi-hidden-unit) ALS should meaningfully fit an
    interaction term it cannot express additively."""
    rng = np.random.RandomState(0)
    n = 300
    X = rng.uniform(-1, 1, (n, 2)).astype(np.float32)
    y = (X[:, 0] * X[:, 1]).astype(np.float32)

    kan = KAN(width=[2, 3, 1], grid=3, k=3, seed=1)
    X_t = torch.tensor(X)
    kan.fit({"train_input": X_t, "train_label": torch.tensor(y)}, steps=10)

    pred = kan(X_t).cpu().numpy().flatten()
    assert _r2(y, pred) > 0.8


def test_als_nongam_wide_input():
    """Regression test for a real bug found during review: `_solve_layer_single`
    (ALS step 2, updating layer-0 given layer-1) used to fit each input
    feature's spline INDEPENDENTLY in a loop, each one trying alone to
    reconstruct the entire per-hidden-unit target -- summing n_in
    independently-fit reconstructions overshoots by ~n_in x. This was
    invisible with the 2-3 input features every other DeepKAN test uses (the
    overshoot wasn't catastrophic enough to fail their assertions), but with
    real ~30-feature data it drove the loss so far above the trivial
    zero-baseline that ALS's rollback-to-best-state kept reverting to an
    all-zero layer-1, silently making every non-GAM learner predict a
    constant. Pin the failure mode at a wide-input scale directly (not just
    via the slow breast_cancer integration test in test_accel.py)."""
    rng = np.random.RandomState(0)
    n = 400
    n_features = 12
    X = rng.uniform(-1, 1, (n, n_features)).astype(np.float32)
    true_coefs = rng.randn(n_features)
    y = (X @ true_coefs).astype(np.float32)

    kan = KAN(width=[n_features, 3, 1], grid=3, k=3, seed=1)
    X_t = torch.tensor(X)
    kan.fit({"train_input": X_t, "train_label": torch.tensor(y)}, steps=10)

    pred = kan(X_t).cpu().numpy().flatten()
    assert pred.std() > 1e-3, "predictions must not collapse to a constant"
    assert _r2(y, pred) > 0.3


def test_als_solve_layer_weighted_matches_manual_wls():
    """Regression test for a real bug found during review: the ALS solver's
    weighted least-squares RHS was accidentally BᵀW(Wt) instead of BᵀWt,
    silently squaring the weights. This unit-tests `_solve_layer` directly
    (bypassing ALS's outer nonlinear iteration, which is not a reliable
    optimizer and would confound any end-to-end comparison) against a
    manual weighted-least-squares solve on the same design matrix."""
    from kanboost.core.kan.network import DeepKAN
    from kanboost.core.kan.bspline import _b_basis_1d

    rng = np.random.RandomState(0)
    n = 200
    x_in = rng.uniform(-1, 1, (n, 1)).astype(float)
    target = (2.0 * x_in[:, 0] + 0.3 * rng.randn(n)).astype(float)
    w = rng.uniform(0.1, 5.0, n)

    kan = DeepKAN(width=[1, 1, 1], grid=4, k=3, seed=0)
    layer = kan.layers[0]
    kan._solve_layer(layer, x_in, target, w, lam_smooth=0.0, lam_ridge=1e-6)
    fitted = layer.forward(x_in).flatten()

    # Manual weighted least squares on the identical design matrix.
    B = _b_basis_1d(x_in[:, 0], layer.knots[0], layer.k)
    Bw = B * w[:, np.newaxis]
    c_manual = np.linalg.solve(Bw.T @ B + 1e-6 * np.eye(B.shape[1]), Bw.T @ target)
    fitted_manual = B @ c_manual

    assert np.allclose(fitted, fitted_manual, atol=1e-4)


def test_monotone():
    rng = np.random.RandomState(0)
    n = 300
    X = rng.uniform(-1, 1, (n, 1)).astype(np.float32)
    y = (X[:, 0] ** 3 + 0.05 * rng.randn(n)).astype(np.float32)

    kan = KAN(width=[1, 1, 1], grid=8, k=3, seed=1)
    kan.fix_symbolic(1, 0, 0, "x")
    X_t = torch.tensor(X)
    kan.fit(
        {"train_input": X_t, "train_label": torch.tensor(y)},
        steps=10, monotone_signs=[1],
    )

    x_fine = np.linspace(-1, 1, 200).astype(np.float32).reshape(-1, 1)
    pred = kan(torch.tensor(x_fine)).cpu().numpy().flatten()
    assert np.all(np.diff(pred) >= -1e-6)


def test_gam_hidden_gt1_raises():
    """gam=True requires exactly one hidden unit per feature-sum term —
    fix_symbolic(1,0,0,'x') on a wider hidden layer is a model
    misspecification, not a valid GAM, and must fail loudly."""
    kan = KAN(width=[2, 3, 1], grid=3, k=3, seed=1)
    with pytest.raises(NotImplementedError):
        kan.fix_symbolic(1, 0, 0, "x")


def test_numba_basis_matches_scipy_fallback():
    """Regression test for the numba-accelerated B-spline kernel (~6.5x
    faster than scipy's design_matrix on this project's typical shapes,
    since scipy pays sparse-matrix construction overhead for a result that's
    immediately densified). Must match the scipy fallback to machine
    precision across grid/order combinations, or a bug in the hand-written
    Cox-de-Boor recursion could silently corrupt every fit."""
    from kanboost.core.kan.bspline import _b_basis_1d, _b_basis_1d_scipy, extend_grid, build_grid, _NUMBA

    if not _NUMBA:
        pytest.skip("numba not installed -- scipy fallback path is exercised by every other test")

    rng = np.random.RandomState(0)
    for grid, k in [(2, 3), (5, 3), (8, 3), (3, 2), (10, 3)]:
        knots = extend_grid(build_grid(grid, k), k_extend=k)[0]
        x = rng.uniform(-1.5, 1.5, 60)
        x_clip = np.clip(x, knots[k], knots[-k - 1])
        got = _b_basis_1d(x, knots, k)
        want = _b_basis_1d_scipy(x_clip, knots, k)
        assert np.abs(got - want).max() < 1e-10, f"grid={grid} k={k}"


def test_fit_params():
    rng = np.random.RandomState(0)
    x = np.linspace(-1, 1, 200)
    y = 2 * np.sin(3 * x + 1) - 0.5 + 0.01 * rng.randn(200)

    params, r2 = fit_params(x, y, SYMBOLIC_LIB["sin"][0], verbose=False)
    assert r2 > 0.99


def test_editing_pipeline():
    """The interpret pipeline (consolidate -> export_symbolic) must work
    unchanged against DeepKAN-trained GAM ensembles, and should recover
    x^2 as the closed form for a feature whose true shape is x^2."""
    rng = np.random.RandomState(0)
    n = 400
    X = rng.uniform(-1, 1, (n, 2))
    y = X[:, 0] ** 2 + 0.1 * rng.randn(n)
    import pandas as pd
    X_df = pd.DataFrame(X, columns=["a", "b"])

    model = KANBoostRegressor(
        n_estimators=20, kan_steps=20, kan_hidden=1, kan_grid=8,
        gam=True, early_stopping_rounds=None, random_state=0,
    )
    model.fit(X_df, y)

    gam = consolidate(model)
    assert set(gam.feature_names) == {"a", "b"}

    sym = export_symbolic(model, features=["a"], min_r2=0.9, allow_periodic=False)
    assert sym.terms["a"]["kind"] == "symbolic"
    assert sym.terms["a"]["candidate"] == "x^2"
