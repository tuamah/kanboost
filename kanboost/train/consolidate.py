"""
kanboost.train.consolidate -- post-hoc ensemble-size reduction for a fitted
KANBoostClassifier/Regressor.

GrowNet (Badirli et al., 2020) boosts shallow neural nets with a periodic
*fully corrective* step -- earlier weak learners get jointly re-optimized
against later residuals, instead of being frozen forever the moment they're
added, the way classic boosting (and kanboost's own `_boost_chain`) does.
A faithful from-scratch port of that joint-backprop step onto kanboost's
closed-form ALS/GAM solver (see `core/kan/network.py`) was judged too large
and too risky a change to the training loop's actual math for one pass; this
module is a smaller, safer stand-in for the same underlying idea -- "many
small independently-boosted corrections are redundant; let a single,
better-fit correction replace them" -- applied *after* training completes,
so `_boost_chain`'s own math is never touched.

Consecutive groups of `group_size` weak learners are replaced by ONE new
learner, least-squares-refit (via the model's own closed-form
`_fit_learner`) to reproduce the group's summed raw output on `X`. Because
every learner in a chain is added as `F += learning_rate * learner(X)`
(`core/base.py::_boost_chain`), matching the *sum* of a group's outputs
with one new learner preserves that group's total contribution to `F`
(exactly, if the new learner's capacity can represent the sum; a
closed-form best-fit otherwise) while cutting how many learners must be
evaluated at predict time.

Measured on a real clinical benchmark (INSPIRE, `postop_icu`, 78K rows,
group_size=5): ensemble size 180->36, prediction time 54.1s->10.8s (5x),
AUROC -0.0026 / AUPRC -0.0056 (see `AI_REVIEW_LOOP.md`, Proposal CC-8, for
the full evidence table across three data scales).

Unlike `kanboost.interpret.editing.consolidate()` (which requires
`gam=True` and produces an `EditableGAM`), this works on ANY fitted
KANBoostClassifier/Regressor, GAM or not, and mutates the model's own
`learners_`/`best_iteration_` in place -- no new object, no editability,
just a smaller ensemble that predicts the (approximately) same function.
"""

from __future__ import annotations

import numpy as np
import torch


def _consolidate_chain(model, learners: list, best_iteration: int, X_t, group_size: int) -> list:
    active = learners[:best_iteration]
    if group_size < 2 or len(active) <= group_size:
        return list(active)

    n_features = active[0].width[0][0]
    new_learners = []
    for start in range(0, len(active), group_size):
        group = active[start:start + group_size]
        if len(group) == 1:
            new_learners.append(group[0])
            continue
        with torch.no_grad():
            target_sum = sum(l(X_t).cpu().numpy().flatten() for l in group)
        new_learner = model._new_learner(n_features, seed_offset=1_000_000 + start)
        model._fit_learner(new_learner, X_t, target_sum)
        new_learners.append(new_learner)
    return new_learners


def consolidate_learners(model, X, group_size: int = 5) -> None:
    """Shrink a fitted model's ensemble by replacing consecutive groups of
    `group_size` weak learners with one, in place.

    Parameters
    ----------
    model : a fitted KANBoostClassifier or KANBoostRegressor.
    X : DataFrame or array, same columns as what `model.fit()` was called
        with -- ideally the training set (or a large, representative
        sample of it), since each replacement learner is least-squares-fit
        against the group's outputs evaluated on these rows.
    group_size : int, default=5
        Number of consecutive learners collapsed into one. Larger values
        shrink the ensemble more aggressively at a greater (though usually
        still small) accuracy cost -- there is no single right value;
        compare accuracy before/after on your own held-out data, the same
        way you would for any other post-hoc approximation.

    Modifies `model.learners_` / `model.best_iteration_` (per one-vs-rest
    chain, for a multiclass classifier) in place. Returns None.

    Safe to call multiple times (idempotent-ish: consolidating an
    already-consolidated, smaller ensemble further just uses a larger
    effective group size relative to what's left), and safe to `save()`
    afterward -- the resulting learners are ordinary fitted weak learners,
    indistinguishable from ones produced by `fit()` itself.
    """
    model._check_fitted()
    X_t = model._transform_X(X)

    if isinstance(model.learners_, dict):
        for c in model.classes_:
            model.learners_[c] = _consolidate_chain(
                model, model.learners_[c], model.best_iteration_[c], X_t, group_size,
            )
            model.best_iteration_[c] = len(model.learners_[c])
    else:
        model.learners_ = _consolidate_chain(
            model, model.learners_, model.best_iteration_, X_t, group_size,
        )
        model.best_iteration_ = len(model.learners_)
