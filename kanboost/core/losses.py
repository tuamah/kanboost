"""
Loss objects for the shared boosting loop in `_base.py`.

Each loss defines:
- `init_pred(y, sample_weight)`: the ensemble's starting raw score
- `negative_gradient(y, F)`: the pseudo-residual each weak learner is fit to
- `val_loss(y_val, F_val)`: the metric used for early stopping
- `improvement_eps`: minimum improvement to reset the early-stopping counter

Keeping these as small objects (rather than branching on estimator type
inside the boosting loop) lets `_BaseKANBoost._boost_chain` be shared
verbatim by both the classifier (always `LogisticLoss`, once per
one-vs-rest chain) and the regressor (`SquaredLoss` or `QuantileLoss`).
"""

from __future__ import annotations

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _weighted_quantile(y: np.ndarray, alpha: float, sample_weight=None) -> float:
    y = np.asarray(y, dtype=float)
    if sample_weight is None:
        return float(np.quantile(y, alpha))
    order = np.argsort(y)
    y_sorted = y[order]
    w_sorted = np.asarray(sample_weight, dtype=float)[order]
    cum_w = np.cumsum(w_sorted) - 0.5 * w_sorted
    cum_w /= w_sorted.sum()
    return float(np.interp(alpha, cum_w, y_sorted))


class LogisticLoss:
    """Binary logloss; used for every one-vs-rest chain in `KANBoostClassifier`."""

    improvement_eps = 1e-5

    def init_pred(self, y: np.ndarray, sample_weight=None) -> float:
        p = float(np.clip(np.average(y, weights=sample_weight), 1e-6, 1 - 1e-6))
        return float(np.log(p / (1 - p)))

    def negative_gradient(self, y: np.ndarray, F: np.ndarray) -> np.ndarray:
        return y - _sigmoid(F)

    def val_loss(self, y_val: np.ndarray, F_val: np.ndarray) -> float:
        prob = np.clip(_sigmoid(F_val), 1e-7, 1 - 1e-7)
        return -float(np.mean(y_val * np.log(prob) + (1 - y_val) * np.log(1 - prob)))


class SquaredLoss:
    """Mean squared error; the default `KANBoostRegressor` objective."""

    improvement_eps = 1e-6

    def init_pred(self, y: np.ndarray, sample_weight=None) -> float:
        return float(np.average(y, weights=sample_weight))

    def negative_gradient(self, y: np.ndarray, F: np.ndarray) -> np.ndarray:
        return y - F

    def val_loss(self, y_val: np.ndarray, F_val: np.ndarray) -> float:
        return float(np.mean((y_val - F_val) ** 2))


class QuantileLoss:
    """Pinball loss for quantile regression (`KANBoostRegressor(objective="quantile", alpha=...)`)."""

    improvement_eps = 1e-6

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def init_pred(self, y: np.ndarray, sample_weight=None) -> float:
        return _weighted_quantile(y, self.alpha, sample_weight)

    def negative_gradient(self, y: np.ndarray, F: np.ndarray) -> np.ndarray:
        # Each weak learner is fit (by MSE) to this +alpha/-(1-alpha) step
        # function, i.e. the sign of the pinball loss's subgradient -- the
        # same trick classic GBM quantile regression uses.
        return np.where(y > F, self.alpha, self.alpha - 1.0)

    def val_loss(self, y_val: np.ndarray, F_val: np.ndarray) -> float:
        diff = y_val - F_val
        return float(np.mean(np.maximum(self.alpha * diff, (self.alpha - 1) * diff)))
