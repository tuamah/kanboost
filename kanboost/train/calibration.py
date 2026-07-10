"""
kanboost.calibration -- post-hoc probability calibration for a fitted
KANBoostClassifier.

Three independent benchmarks this session (see README.md's "Honest
limitations" / Benchmarks section) found the same pattern: KANBoost's
`predict_proba` *ranking* (ROC AUC/PR AUC) is competitive with or ahead
of tuned tree ensembles, but its raw probability *values* are
comparatively miscalibrated -- worst Brier score and log-loss across all
three runs, with the F1-optimal decision threshold sitting around
0.40-0.42 rather than 0.5. That's a systematic shift/skew in the raw
score, not a shape-based miscalibration, which is exactly what Platt
scaling (a 2-parameter logistic fit on the raw score) corrects with the
least variance -- hence the default here.

Additive: no changes to `_base.py`/`classifier.py`. `calibrate()` wraps
an already-fitted `KANBoostClassifier` and reads its raw scores via the
same `_raw_score_chain`/`_transform_X` methods `predict_proba` itself
uses -- it does not need retraining or any new hooks in the core model.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


def _raw_scores(model, X):
    """Per-chain raw scores (log-odds, pre-sigmoid) for every class in
    `model.classes_`, shape (n_samples, n_classes) -- always 2 columns
    for binary (both classes' scores, positive class first per
    `classes_[1]`), even though `_raw_score_chain` only computes one
    chain for the binary case."""
    model._check_fitted()
    X_t = model._transform_X(X)
    if len(model.classes_) == 2:
        pos_score = model._raw_score_chain(
            X_t, model.learners_, model.init_pred_, model.best_iteration_
        )
        return np.column_stack([-pos_score, pos_score])
    return np.column_stack([
        model._raw_score_chain(X_t, model.learners_[c], model.init_pred_[c], model.best_iteration_[c])
        for c in model.classes_
    ])


def _calibration_path(path: str) -> str:
    # os.path.splitext, not a bare rsplit(".", 1) -- a directory
    # component containing a dot (e.g. "out.v2/model") must not be
    # mistaken for a file extension.
    root, _ext = os.path.splitext(path)
    return root + "_calibration.pt"


def _fit_one_calibrator(raw_score: np.ndarray, y_binary: np.ndarray, method: str):
    if method == "platt":
        # "Classic" Platt scaling is unregularized; sklearn's LogisticRegression
        # defaults to L2 with C=1.0. A large C effectively turns that off --
        # deliberate here, since calibration sets are often small and the
        # 2-parameter fit is already low-variance, so there's little need for
        # extra shrinkage, and it keeps the fit closer to the textbook method.
        calibrator = LogisticRegression(C=1e6)
        calibrator.fit(raw_score.reshape(-1, 1), y_binary)
        return calibrator
    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_score, y_binary)
        return calibrator
    raise ValueError(f"Unknown method {method!r}; expected 'platt' or 'isotonic'")


def _apply_one_calibrator(calibrator, raw_score: np.ndarray, method: str) -> np.ndarray:
    if method == "platt":
        return calibrator.predict_proba(raw_score.reshape(-1, 1))[:, 1]
    return calibrator.predict(raw_score)


def calibrate(model, X_cal, y_cal, method: str = "platt") -> "CalibratedKANBoost":
    """Fit a post-hoc calibration map for a fitted `KANBoostClassifier`,
    using `X_cal`/`y_cal` -- data the model was **not** trained on (a
    held-out calibration split, e.g. carved out the same way `eval_set`
    is). Returns a `CalibratedKANBoost` wrapping the original model;
    the original model itself is untouched.

    `method="platt"` (default): a 2-parameter logistic fit on the raw
    score. Recommended for KANBoost specifically -- the measured
    miscalibration is a systematic shift (the optimal decision threshold
    sits around 0.40-0.42, not 0.5), which is exactly what a 2-parameter
    monotone rescaling fixes, and it needs far less calibration data
    than isotonic regression to avoid overfitting. Being strictly
    monotone, it also leaves ROC AUC/PR AUC exactly unchanged.
    `method="isotonic"`: a free-form monotone fit; more flexible, but
    needs a larger `X_cal` (order of 1000+ rows) to not overfit, and can
    produce flat/zero-gradient regions.

    Multiclass: each one-vs-rest chain is calibrated independently
    (mirroring how the base model's own `predict_proba` computes each
    chain independently), then rows are renormalized to sum to 1.
    """
    if not hasattr(model, "classes_"):
        raise ValueError("calibrate() requires a fitted classifier (no classes_ found).")
    y_cal = np.asarray(y_cal)
    raw = _raw_scores(model, X_cal)  # (n, n_classes)

    calibrators = []
    for j, c in enumerate(model.classes_):
        calibrators.append(_fit_one_calibrator(raw[:, j], (y_cal == c).astype(float), method))

    return CalibratedKANBoost(model, calibrators, method)


class CalibratedKANBoost:
    """A fitted `KANBoostClassifier` plus a post-hoc calibration map.
    Build one via `calibrate(model, X_cal, y_cal)`, not directly.

    Edit-then-calibrate ordering: if you also use `kanboost.editing`,
    calibrate *after* any `EditableGAM` edits are finalized, not before
    -- an edit changes the model's raw scores, which would silently
    stale an already-fitted calibration map (nothing here detects that;
    recalibrate if you edit the model afterward).
    """

    def __init__(self, model, calibrators: list, method: str):
        self.model = model
        self.calibrators = calibrators
        self.method = method

    @property
    def classes_(self):
        return self.model.classes_

    def predict_proba(self, X) -> np.ndarray:
        raw = _raw_scores(self.model, X)
        calibrated = np.column_stack([
            _apply_one_calibrator(cal, raw[:, j], self.method)
            for j, cal in enumerate(self.calibrators)
        ])
        totals = calibrated.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0  # avoid 0/0 in the (pathological) all-zero row case
        return calibrated / totals

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        """Same semantics as the base model's `.predict()`: `threshold`
        only applies to the binary case; multiclass always uses argmax."""
        proba = self.predict_proba(X)
        if len(self.classes_) == 2:
            return np.where(proba[:, 1] >= threshold, self.classes_[1], self.classes_[0])
        return self.classes_[np.argmax(proba, axis=1)]

    def save(self, path: str) -> None:
        torch.save({"method": self.method, "calibrators": self.calibrators}, _calibration_path(path))
        self.model.save(path)

    @classmethod
    def load(cls, path: str, device: str | None = None) -> "CalibratedKANBoost":
        """Load a base model saved via `save()` plus its sibling
        `..._calibration.pt` calibration-map file (both written by
        `save()`). Only load files your own code saved -- like the base
        model's own `load()`, this unpickles arbitrary Python objects
        (`weights_only=False`), which is unsafe for untrusted input."""
        from ..core.classifier import KANBoostClassifier

        model = KANBoostClassifier.load(path, device=device)
        payload = torch.load(_calibration_path(path), weights_only=False)
        return cls(model, payload["calibrators"], payload["method"])
