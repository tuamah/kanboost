"""
B-spline primitives, fit_params, and SYMBOLIC_LIB — numpy/scipy drop-in replacements
for kan.spline and kan.utils, preserving the exact same public signatures so
editing.py and symbolic.py work unchanged after their import lines are updated.
"""

from __future__ import annotations

import numpy as np
import scipy.interpolate
import scipy.linalg
import scipy.optimize
import sympy

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False


def _np(t) -> np.ndarray:
    if _TORCH and isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _like(arr: np.ndarray, ref):
    """Return arr as the same type (torch.Tensor or numpy) as ref."""
    if _TORCH and isinstance(ref, torch.Tensor):
        return torch.tensor(arr, dtype=ref.dtype, device=ref.device)
    return arr


# ---------------------------------------------------------------------------
# Internal 1-D B-spline basis
# ---------------------------------------------------------------------------

def _b_basis_1d(x: np.ndarray, knots: np.ndarray, k: int) -> np.ndarray:
    """B-spline design matrix for one spline.

    x     : (n,) — evaluation points; clipped to valid support
    knots : (G+2k+1,) — extended knot vector (all-distinct, uniform)
    k     : spline order
    Returns (n, G+k) numpy array
    """
    x = np.clip(x, knots[k], knots[-k - 1])
    return scipy.interpolate.BSpline.design_matrix(x, knots, k).toarray()


def _b_basis_deriv_1d(x: np.ndarray, knots: np.ndarray, k: int) -> np.ndarray:
    """Exact derivative of the B-spline basis via the recurrence
    d/dx B_{k,i} = k*(B_{k-1,i}/(t_{i+k}-t_i) - B_{k-1,i+1}/(t_{i+k+1}-t_{i+1})).
    Returns (n, G+k).
    """
    if k == 0:
        return np.zeros((len(x), len(knots) - 1))
    B_km1 = _b_basis_1d(x, knots, k - 1)           # (n, G+k+1-1) = (n, G+k+1) where G+k = K
    K = len(knots) - k - 1                          # number of order-k bases
    result = np.zeros((len(x), K))
    for i in range(K):
        d_lo = knots[i + k] - knots[i]
        d_hi = knots[i + k + 1] - knots[i + 1]
        t1 = k * B_km1[:, i] / d_lo if d_lo > 1e-14 else np.zeros(len(x))
        t2 = k * B_km1[:, i + 1] / d_hi if d_hi > 1e-14 else np.zeros(len(x))
        result[:, i] = t1 - t2
    return result


# ---------------------------------------------------------------------------
# Public grid helpers (pykan-compatible tensor signatures)
# ---------------------------------------------------------------------------

def diff_matrix(K: int, order: int = 2) -> np.ndarray:
    """P-spline difference matrix D ∈ R^{(K-order)×K}.  Penalty = λ·DᵀD."""
    return np.diff(np.eye(K), n=order, axis=0)


def build_grid(num_intervals: int, k: int, lo: float = -1.0, hi: float = 1.0) -> np.ndarray:
    """Uniform base grid (NOT extended), shape (1, num_intervals + 1)."""
    return np.linspace(lo, hi, num_intervals + 1)[np.newaxis, :]


def extend_grid(grid, k_extend: int = 0):
    """Extend grid by k_extend steps on both sides.
    Accepts and returns the same type as input (torch.Tensor or numpy).
    Grid shape: (in_dim, n_points) — same as pykan.
    """
    is_torch = _TORCH and isinstance(grid, torch.Tensor)
    g = _np(grid).copy().astype(float)
    h = (g[:, [-1]] - g[:, [0]]) / (g.shape[1] - 1)
    for _ in range(k_extend):
        g = np.concatenate([g[:, [0]] - h, g], axis=1)
        g = np.concatenate([g, g[:, [-1]] + h], axis=1)
    return _like(g, grid) if is_torch else g


# ---------------------------------------------------------------------------
# coef2curve / curve2coef (pykan-compatible torch tensor signatures)
# ---------------------------------------------------------------------------

def coef2curve(x_eval, grid, coef, k: int, device=None):
    """Evaluate B-spline curves.

    x_eval : (batch, in_dim)
    grid   : (in_dim, G+2k+1)
    coef   : (in_dim, out_dim, G+k)
    k      : spline order
    Returns : (batch, in_dim, out_dim) — same type as x_eval
    """
    is_torch = _TORCH and isinstance(x_eval, torch.Tensor)
    x = _np(x_eval)
    g = _np(grid)
    c = _np(coef)
    batch, in_dim = x.shape
    out_dim = c.shape[1]
    result = np.zeros((batch, in_dim, out_dim))
    for i in range(in_dim):
        B = _b_basis_1d(x[:, i], g[i], k)  # (batch, K)
        result[:, i, :] = B @ c[i].T        # (batch, out_dim)
    if is_torch:
        return torch.tensor(result, dtype=x_eval.dtype, device=x_eval.device)
    return result


def curve2coef(x_eval, y_eval, grid, k: int):
    """Fit B-spline coefficients from curve samples via least squares.

    x_eval : (batch, in_dim)
    y_eval : (batch, in_dim, out_dim)
    grid   : (in_dim, G+2k+1)
    k      : spline order
    Returns : (in_dim, out_dim, G+k) — same type as x_eval
    """
    is_torch = _TORCH and isinstance(x_eval, torch.Tensor)
    x = _np(x_eval)
    y = _np(y_eval)
    g = _np(grid)
    in_dim = x.shape[1]
    out_dim = y.shape[2]
    K = g.shape[1] - k - 1
    coef = np.zeros((in_dim, out_dim, K))
    for i in range(in_dim):
        B = _b_basis_1d(x[:, i], g[i], k)       # (batch, K)
        for o in range(out_dim):
            coef[i, o], _, _, _ = np.linalg.lstsq(B, y[:, i, o], rcond=None)
    if is_torch:
        return torch.tensor(coef, dtype=x_eval.dtype, device=x_eval.device)
    return coef


# ---------------------------------------------------------------------------
# fit_params (pykan-compatible signature)
# ---------------------------------------------------------------------------

def _vectorized_grid_search(x, y, fun, a_vals, b_vals, y_mean, ss_tot):
    """Evaluate y ≈ c*fun(a*x+b)+d over the full (a, b) grid in one shot via
    broadcasting, instead of a nested Python double loop — that loop measured
    at ~0.86s per fit_params call (a 101x101 grid), which multiplied across
    every feature/candidate/class in symbolic_report() blew past the
    dashboard's 60s render timeout for multiclass GAM models.
    """
    A = a_vals[:, np.newaxis, np.newaxis]      # (G, 1, 1)
    Bv = b_vals[np.newaxis, :, np.newaxis]      # (1, G, 1)
    X = x[np.newaxis, np.newaxis, :]            # (1, 1, n)
    with np.errstate(all="ignore"):
        Z = np.asarray(fun(A * X + Bv), dtype=float)   # (G, G, n)
        zm = Z.mean(axis=2, keepdims=True)
        var_z = ((Z - zm) ** 2).sum(axis=2)             # (G, G)
        cov = ((Z - zm) * (y - y_mean)).sum(axis=2)     # (G, G)
        c = np.where(var_z > 1e-12, cov / np.where(var_z > 1e-12, var_z, 1.0), 0.0)
        d = y_mean - c * zm[:, :, 0]
        ss_res = ((y - c[:, :, np.newaxis] * Z - d[:, :, np.newaxis]) ** 2).sum(axis=2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.zeros_like(ss_res)
    r2 = np.where(np.isfinite(r2), r2, -np.inf)
    idx = np.unravel_index(np.argmax(r2), r2.shape)
    return a_vals[idx[0]], b_vals[idx[1]], float(c[idx]), float(d[idx]), float(r2[idx])


def fit_params(
    x, y, fun,
    a_range=(-10, 10), b_range=(-10, 10),
    grid_number=101, iteration=3,
    verbose=True, device="cpu",
):
    """Fit y ≈ c*fun(a*x + b) + d via coarse-to-fine grid search + Nelder-Mead polish.

    Returns (np.array([a, b, c, d]), r2).  Accepts torch tensors.
    """
    x = _np(x).flatten().astype(float)
    y = _np(y).flatten().astype(float)
    y_mean = y.mean()
    ss_tot = float(((y - y_mean) ** 2).sum())

    a_lo, a_hi = a_range
    b_lo, b_hi = b_range
    best_a, best_b, best_c, best_d, best_r2 = 1.0, 0.0, 1.0, 0.0, -np.inf

    for pass_i in range(max(iteration, 1)):
        # Only the first pass needs full resolution over the whole range;
        # each refinement pass already searches a much narrower window, so a
        # coarser grid there still improves precision at a fraction of the cost.
        g = grid_number if pass_i == 0 else max(grid_number // 5, 11)
        a_vals = np.linspace(a_lo, a_hi, g)
        b_vals = np.linspace(b_lo, b_hi, g)
        a, b, c, d, r2 = _vectorized_grid_search(x, y, fun, a_vals, b_vals, y_mean, ss_tot)
        if r2 > best_r2:
            best_a, best_b, best_c, best_d, best_r2 = a, b, c, d, r2
        # Narrow the search window around the current best for the next pass
        a_step = (a_hi - a_lo) / (g - 1)
        b_step = (b_hi - b_lo) / (g - 1)
        a_lo, a_hi = best_a - 2 * a_step, best_a + 2 * a_step
        b_lo, b_hi = best_b - 2 * b_step, best_b + 2 * b_step

    best_params = np.array([best_a, best_b, best_c, best_d])

    def _neg_r2(params):
        a0, b0, c0, d0 = params
        try:
            z = np.asarray(fun(a0 * x + b0), dtype=float)
            if not np.all(np.isfinite(z)):
                return 1.0
            ss_res = float(((y - c0 * z - d0) ** 2).sum())
            return ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        except Exception:
            return 1.0

    res = scipy.optimize.minimize(
        _neg_r2, best_params, method="Nelder-Mead",
        options={"maxiter": 300, "xatol": 1e-5, "fatol": 1e-5},
    )
    if res.success and (1.0 - res.fun) > best_r2:
        best_r2 = 1.0 - res.fun
        best_params = res.x

    return best_params, float(best_r2)


# ---------------------------------------------------------------------------
# SYMBOLIC_LIB  (pykan-compatible 4-tuple format)
# ---------------------------------------------------------------------------
# Each entry: (numpy_fn, sympy_fn, complexity_int, safe_numpy_fn)
# symbolic.py uses [0] (numpy) and [1] (sympy) only.

def _safe_log(x):
    return np.log(np.abs(x) + 1e-8)

def _safe_sqrt(x):
    return np.sqrt(np.abs(x))

def _safe_recip(x):
    return 1.0 / (x + np.sign(x + 1e-8) * 1e-6)

def _safe_tan(x):
    return np.tan(np.clip(x, -1.55, 1.55))

def _safe_arcsin(x):
    return np.arcsin(np.clip(x, -1.0, 1.0))

def _safe_exp(x):
    return np.exp(np.clip(x, -100, 100))

def _gaussian(x):
    return np.exp(-x ** 2)

def _sigmoid(x):
    return 1.0 / (1.0 + _safe_exp(-x))

SYMBOLIC_LIB = {
    "x":        (lambda x: x,              lambda x: x,                      1, lambda x: x),
    "x^2":      (lambda x: x ** 2,         lambda x: x ** 2,                 2, lambda x: x ** 2),
    "x^3":      (lambda x: x ** 3,         lambda x: x ** 3,                 3, lambda x: x ** 3),
    "x^4":      (lambda x: x ** 4,         lambda x: x ** 4,                 3, lambda x: x ** 4),
    "1/x":      (_safe_recip,              lambda x: 1 / x,                  2, _safe_recip),
    "sqrt":     (_safe_sqrt,               sympy.sqrt,                       2, _safe_sqrt),
    "exp":      (_safe_exp,                sympy.exp,                        2, _safe_exp),
    "log":      (_safe_log,                sympy.log,                        2, _safe_log),
    "abs":      (np.abs,                   sympy.Abs,                        2, np.abs),
    "sin":      (np.sin,                   sympy.sin,                        2, np.sin),
    "cos":      (np.cos,                   sympy.cos,                        2, np.cos),
    "tan":      (_safe_tan,                sympy.tan,                        2, _safe_tan),
    "tanh":     (np.tanh,                  sympy.tanh,                       2, np.tanh),
    "sgn":      (np.sign,                  sympy.sign,                       1, np.sign),
    "arcsin":   (_safe_arcsin,             sympy.asin,                       2, _safe_arcsin),
    "arctan":   (np.arctan,                sympy.atan,                       2, np.arctan),
    "gaussian": (_gaussian,                lambda x: sympy.exp(-x ** 2),     2, _gaussian),
    "sigmoid":  (_sigmoid,                 lambda x: 1 / (1 + sympy.exp(-x)), 2, _sigmoid),
}
