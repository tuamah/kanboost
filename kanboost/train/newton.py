"""
kanboost.train.newton -- Newton-step (second-order) boosting for
KANBoostClassifier (binary only), instead of the standard first-order
`_boost_chain` (each weak learner fit directly to the raw pseudo-residual
`y - p`).

GB-KAN (Mohr & Frochte, ICAART 2026) -- an independently published
KAN-based boosting framework -- explicitly lists "second-order boosting"
as unsolved future work for this model family. This module implements it
for kanboost: each round reweights the fitting problem using the
logistic loss's second derivative (the Hessian diagonal
`h_i = p_i(1-p_i)`), fitting the weak learner to the Newton target
`z_i = (y_i - p_i) / h_i` with sample weight `h_i` -- exactly how
XGBoost derives its leaf values, adapted to kanboost's closed-form ALS
weak learner instead of a tree split search.

Measured on INSPIRE (`postop_icu`; see
`docs/inspire_kanboost_evaluation.md` §9), same round count as a normal
fixed-`learning_rate` fit, all three scales tested:

    Small  (10K rows, 100 rounds):  AUROC 0.9225->0.9246, AUPRC 0.6707->0.6879
    Medium (50K rows, 140 rounds):  AUROC 0.9389->0.9411, AUPRC 0.7560->0.7686
    Large  (78K rows, 180 rounds):  AUROC 0.9413->0.9434, AUPRC 0.7643->0.7757

Accuracy improved consistently at every scale. Fit time, however, did
NOT move consistently -- neutral at Small, ~33% slower at Medium, ~19%
faster at Large -- ALS convergence (number of internal sweeps before its
own early-stop tolerance triggers) appears to depend on how the
reweighted target's curvature interacts with the dataset, not on a
fixed multiplier. Treat this as an accuracy-oriented option, not a
speed-oriented one.

**Tested and explicitly rejected**: combining this with
`kanboost.train.linesearch.fit_with_line_search()`'s per-round line
search. At every scale tested, the combination underperformed EITHER
technique used alone (e.g. Large, 50 rounds: combined AUROC 0.9400 vs.
0.9434 for Newton alone at 180 rounds, or 0.9432 for line search alone
at 50 rounds). Likely cause: the line search there optimizes the step
size against the *original* logistic loss, while the learner was fit
against the Newton-*reweighted* target -- an inconsistency, not a
line-search bug per se. A theoretically consistent combination would
need to line-search against the same reweighted quadratic objective
the learner was fit to; that redesign is not implemented here. Use one
or the other, not both, until this is resolved.
"""

from __future__ import annotations

import numpy as np
import torch

from kanboost.core.losses import LogisticLoss, _sigmoid

_MIN_HESSIAN = 1e-3  # floor on p(1-p); keeps Newton targets bounded as p -> 0/1


def fit_with_newton_boosting(model, X, y, sample_weight=None, min_hessian: float = _MIN_HESSIAN):
    """Fit `model` (an unfitted binary `KANBoostClassifier`) using
    Newton-step (second-order) boosting instead of its standard
    first-order `_boost_chain`.

    `model.n_estimators` and `model.learning_rate` are used as-is (same
    meaning as a normal `fit()` call -- this is a drop-in accuracy
    improvement at the same round count, not a round-reduction
    technique like `fit_with_line_search`).

    `min_hessian`: floor applied to `p*(1-p)` before dividing to form
    the Newton target, to avoid exploding targets for samples the
    ensemble is already confident about. Default matches what was
    measured.

    Returns `model`, fitted in place (same convention as `model.fit()`).
    """
    if not hasattr(model, "predict_proba"):
        raise TypeError("fit_with_newton_boosting requires a KANBoostClassifier, not a regressor.")

    X, y_arr, X_arr = model._prepare_fit(X, y)
    model.classes_ = np.unique(y_arr)
    if len(model.classes_) != 2:
        raise ValueError(
            "fit_with_newton_boosting only supports binary classification; "
            f"got {len(model.classes_)} classes. Multiclass is not implemented."
        )
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=float).ravel()
    else:
        sample_weight = np.ones(len(y_arr))

    y_bin = (y_arr == model.classes_[1]).astype(float)
    X_t = torch.tensor(X_arr, dtype=torch.float32, device=model.device_)
    n_features = X_arr.shape[1]

    loss = LogisticLoss()
    init_pred = loss.init_pred(y_bin, sample_weight)
    F = np.full(len(y_bin), init_pred)
    learners = []

    for t in range(model.n_estimators):
        p = _sigmoid(F)
        hess = np.clip(p * (1 - p), min_hessian, None)
        target = (y_bin - p) / hess
        fit_weight = hess * sample_weight

        learner = model._new_learner(n_features, seed_offset=t)
        update = model._fit_learner(learner, X_t, target, sample_weight=fit_weight, seed_offset=t)
        F += model.learning_rate * update
        learners.append(learner)

    model.learners_ = learners
    model.init_pred_ = init_pred
    model.best_iteration_ = len(learners)
    return model
