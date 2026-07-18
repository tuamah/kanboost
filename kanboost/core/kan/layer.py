"""
KANLayer — one layer of B-spline connections.
Internals are pure numpy; no torch required here.
"""

from __future__ import annotations

import numpy as np
from kanboost.core.kan.bspline import _b_basis_1d, _b_basis_deriv_1d


class KANLayer:
    """B-spline layer: n_in → n_out, one spline per (in, out) edge.

    coef shape: (n_in, n_out, K)  — identical to pykan act_fun[l].coef axis order,
    so feature_importances() and _apply_monotone_projection replacements work unchanged.
    """

    def __init__(self, n_in: int, n_out: int, knots: np.ndarray, k: int) -> None:
        """
        knots : (n_in, G+2k+1) — per-feature extended knot vectors (same for each
                feature since inputs are pre-scaled to [-1, 1], but kept per-feature
                to match pykan's grid interface).
        """
        self.n_in = n_in
        self.n_out = n_out
        self.k = k
        self.knots = knots                          # (n_in, G+2k+1)
        K = knots.shape[1] - k - 1
        self.coef = np.zeros((n_in, n_out, K))     # initialised to zero

    # ------------------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        """x : (n, n_in) → (n, n_out)."""
        n = x.shape[0]
        out = np.zeros((n, self.n_out))
        for i in range(self.n_in):
            B = _b_basis_1d(x[:, i], self.knots[i], self.k)   # (n, K)
            out += B @ self.coef[i].T                           # (n, n_out)
        return out

    def postacts(self, x: np.ndarray) -> np.ndarray:
        """Per-edge output values. x : (n, n_in) → (n, n_out, n_in).

        Replaces pykan's 4-tuple unpack `_, _, postacts, _ = act_fun[0](X_t)`.
        The sum over axis 1 gives per-feature contributions.
        """
        n = x.shape[0]
        out = np.zeros((n, self.n_out, self.n_in))
        for i in range(self.n_in):
            B = _b_basis_1d(x[:, i], self.knots[i], self.k)   # (n, K)
            for o in range(self.n_out):
                out[:, o, i] = B @ self.coef[i, o]
        return out

    def deriv(self, x: np.ndarray) -> np.ndarray:
        """Analytic first derivative. x : (n, n_in) → (n, n_in, n_out)."""
        n = x.shape[0]
        out = np.zeros((n, self.n_in, self.n_out))
        for i in range(self.n_in):
            dB = _b_basis_deriv_1d(x[:, i], self.knots[i], self.k)  # (n, K)
            out[:, i, :] = dB @ self.coef[i].T                        # (n, n_out)
        return out
