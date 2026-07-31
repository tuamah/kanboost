"""
kanboost.train.newton -- Newton-step (second-order) boosting for
KANBoostClassifier (binary and multiclass), instead of the standard
first-order `_boost_chain` (each weak learner fit directly to the raw
pseudo-residual `y - p`).

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
fixed-`learning_rate` fit, all three scales tested (binary):

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

**Multiclass** (3+ classes): one binary Newton-step chain per class,
one-vs-rest, mirroring exactly how `KANBoostClassifier.fit()` handles
multiclass itself (`seed_base = i * n_estimators` per chain,
`learners_`/`init_pred_`/`best_iteration_` keyed by class label). Each
chain is fit independently -- this is `n_classes` times the binary cost,
same as standard multiclass kanboost. Measured on Digits (sklearn, 10
classes, 30 rounds): standard-ALS multiclass fit 29.5s/94.8% accuracy
vs. Newton-boosted 62.4s/97.4% -- accuracy improves further, but so
does fit time, since the reweighting's cost compounds across all
`n_classes` chains on this engine (see `kanboost.train.rfkan` for a
combination where that compounding cost disappears).

**Hessian floor options** (`hessian_floor_mode`): two alternatives to
the default hard `clip(p*(1-p), min_hessian, None)` were tested in
`kanboost.train.ga2m`'s closed-form engine (see
`docs/inspire_kanboost_evaluation.md` §12) and found accuracy-neutral
there while being 7-12% faster at Medium/Large scale: `"adaptive"`
(floor scales with the round's own mean hessian instead of a fixed
constant) and `"soft_lm"` (additive damping instead of a hard clip).
Exposed here with the same options for consistency -- same reweighting
formula, not independently re-measured in this ALS-based chain, but
expected to generalize since only the floor/damping arithmetic changes,
not the solver. The line-search consistency fix that WAS a clear,
consistent win in the GA2M engine (Armijo backtracking from `gamma=1`,
"Variant A") is not exposed here: this module has no per-round line
search step at all (it uses `model.learning_rate` directly), so there
is nothing to make "consistent" -- see `kanboost.train.ga2m`'s
`line_search_mode="armijo"` for that fix where it applies.

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


def _newton_target(y_bin, F, sample_weight, min_hessian, hessian_floor_mode, rel_floor_frac, damping):
    p = _sigmoid(F)
    hess_raw = p * (1 - p)
    if hessian_floor_mode == "hard":
        hess = np.clip(hess_raw, min_hessian, None)
    elif hessian_floor_mode == "adaptive":
        floor = max(1e-4, rel_floor_frac * float(np.mean(hess_raw)))
        hess = np.clip(hess_raw, floor, None)
    elif hessian_floor_mode == "soft_lm":
        hess = hess_raw + damping
    else:
        raise ValueError(f"Unknown hessian_floor_mode: {hessian_floor_mode!r}")
    target = (y_bin - p) / hess
    fit_weight = hess * sample_weight
    return target, fit_weight


def _fit_newton_chain(model, X_t, y_bin, n_features, sample_weight, min_hessian, seed_base,
                       hessian_floor_mode="hard", rel_floor_frac=0.05, damping=1e-3):
    loss = LogisticLoss()
    init_pred = loss.init_pred(y_bin, sample_weight)
    F = np.full(len(y_bin), init_pred)
    learners = []

    for t in range(model.n_estimators):
        target, fit_weight = _newton_target(
            y_bin, F, sample_weight, min_hessian, hessian_floor_mode, rel_floor_frac, damping,
        )

        learner = model._new_learner(n_features, seed_offset=seed_base + t)
        update = model._fit_learner(learner, X_t, target, sample_weight=fit_weight, seed_offset=seed_base + t)
        F += model.learning_rate * update
        learners.append(learner)

    return learners, init_pred, len(learners)


def fit_with_newton_boosting(model, X, y, sample_weight=None, min_hessian: float = _MIN_HESSIAN,
                              hessian_floor_mode: str = "hard", rel_floor_frac: float = 0.05,
                              damping: float = 1e-3):
    """Fit `model` (an unfitted `KANBoostClassifier`, binary or
    multiclass) using Newton-step (second-order) boosting instead of
    its standard first-order `_boost_chain`.

    `model.n_estimators` and `model.learning_rate` are used as-is (same
    meaning as a normal `fit()` call -- this is a drop-in accuracy
    improvement at the same round count, not a round-reduction
    technique like `fit_with_line_search`).

    `min_hessian`: floor applied to `p*(1-p)` before dividing to form
    the Newton target, to avoid exploding targets for samples the
    ensemble is already confident about. Default matches what was
    measured. Only used when `hessian_floor_mode="hard"` (the default).

    `hessian_floor_mode`: `"hard"` (default) clips at `min_hessian`.
    `"adaptive"` floors at `max(1e-4, rel_floor_frac * mean(hess))`
    instead (scales with the round's own average confidence).
    `"soft_lm"` adds `damping` directly instead of clipping. See module
    docstring -- both alternatives were accuracy-neutral and 7-12%
    faster than the hard floor when measured in `kanboost.train.ga2m`'s
    engine; not independently re-measured in this ALS-based chain.

    Returns `model`, fitted in place (same convention as `model.fit()`).
    Multiclass models populate `learners_`/`init_pred_`/`best_iteration_`
    as dicts keyed by class label, exactly like a standard multiclass
    `fit()` -- the base `predict_proba()`/`predict()`/`save()`/`load()`
    all work unmodified.
    """
    if not hasattr(model, "predict_proba"):
        raise TypeError("fit_with_newton_boosting requires a KANBoostClassifier, not a regressor.")

    X, y_arr, X_arr = model._prepare_fit(X, y)
    model.classes_ = np.unique(y_arr)
    if len(model.classes_) < 2:
        raise ValueError(f"KANBoostClassifier needs at least 2 classes; got {model.classes_}.")
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=float).ravel()
    else:
        sample_weight = np.ones(len(y_arr))

    X_t = torch.tensor(X_arr, dtype=torch.float32, device=model.device_)
    n_features = X_arr.shape[1]

    if len(model.classes_) == 2:
        y_bin = (y_arr == model.classes_[1]).astype(float)
        model.learners_, model.init_pred_, model.best_iteration_ = _fit_newton_chain(
            model, X_t, y_bin, n_features, sample_weight, min_hessian, seed_base=0,
            hessian_floor_mode=hessian_floor_mode, rel_floor_frac=rel_floor_frac, damping=damping,
        )
    else:
        model.learners_ = {}
        model.init_pred_ = {}
        model.best_iteration_ = {}
        for i, c in enumerate(model.classes_):
            y_bin = (y_arr == c).astype(float)
            learners, init_pred, best_iteration = _fit_newton_chain(
                model, X_t, y_bin, n_features, sample_weight, min_hessian,
                seed_base=i * model.n_estimators,
                hessian_floor_mode=hessian_floor_mode, rel_floor_frac=rel_floor_frac, damping=damping,
            )
            model.learners_[c] = learners
            model.init_pred_[c] = init_pred
            model.best_iteration_[c] = best_iteration

    return model
