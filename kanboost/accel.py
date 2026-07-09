"""
Opt-in accelerated training for KANBoostClassifier / KANBoostRegressor.

Profiling `fit()` (see the accompanying benchmark in `tests/test_accel.py`)
shows the dominant per-round cost is each weak learner's from-scratch Adam
loop over `kan_steps` optimizer steps -- constructing a fresh `KAN(...)`
every round and re-running the full step budget on it, even though
consecutive rounds' learners share identical architecture and are fitting
increasingly similar residuals late in the ensemble.

`fast_fit()` warm-starts round t+1's learner from round t's fitted
`state_dict` (same `width`/`grid`/`k`, only the seed differs, so shapes
always match) and only spends the model's full `kan_steps` budget on the
very first round of each boosting chain; every later round gets
`warm_start_steps` (default: a quarter of `kan_steps`, floor 3).

This is implemented by temporarily monkey-patching the *instance's*
`_new_learner`/`_fit_learner`/`_boost_chain` bound methods for the
duration of one `fit()` call, then restoring the originals -- so it goes
through the exact same `_fit_learner` / `_fit_learner_custom_loop` /
`_apply_monotone_projection` machinery as a normal `fit()` (monotone
constraints are enforced identically; only how each learner's weights are
*initialized* changes), with zero edits to `_base.py`, `classifier.py`, or
`regressor.py`.
"""

from __future__ import annotations

import torch


def fast_fit(model, X, y, eval_set: tuple | None = None, sample_weight=None,
             warm_start_steps: int | None = None):
    """Fit `model` (an unfitted `KANBoostClassifier`/`KANBoostRegressor`)
    the same way `model.fit(X, y, eval_set=eval_set, sample_weight=sample_weight)`
    would, but warm-starting every round after the first from the previous
    round's learner weights so fewer optimizer steps are needed overall.

    warm_start_steps : int, optional
        Optimizer steps used for every round after the chain's first.
        Defaults to `max(model.kan_steps // 4, 3)`.

    Returns `model` (fitted in place, same as `model.fit(...)`).

    Because each round now starts from a nearby point in weight space
    rather than a fresh random init, this trades a small amount of
    per-round independence (classic boosting fits each learner to the
    *current* residual from scratch) for speed -- always compare accuracy
    against a normal `fit()` on your own data before relying on this for
    anything but a quick iteration loop.
    """
    if warm_start_steps is None:
        warm_start_steps = max(model.kan_steps // 4, 3)
    else:
        warm_start_steps = max(warm_start_steps, 1)

    full_kan_steps = model.kan_steps
    original_new_learner = model._new_learner
    original_fit_learner = model._fit_learner
    original_boost_chain = model._boost_chain

    state = {"round": 0, "prev": None}

    def warm_new_learner(n_features, seed_offset):
        learner = original_new_learner(n_features, seed_offset)
        if state["prev"] is not None:
            with torch.no_grad():
                learner.load_state_dict(state["prev"].state_dict())
        state["round"] += 1
        model.kan_steps = full_kan_steps if state["round"] <= 1 else warm_start_steps
        return learner

    def warm_fit_learner(learner, X_t, residual, sample_weight=None, seed_offset=0):
        update = original_fit_learner(
            learner, X_t, residual, sample_weight=sample_weight, seed_offset=seed_offset
        )
        state["prev"] = learner
        return update

    def warm_boost_chain(*args, **kwargs):
        # Each call is a new one-vs-rest chain (multiclass) -- must not
        # warm-start one class's first learner from another class's last.
        state["round"] = 0
        state["prev"] = None
        model.kan_steps = full_kan_steps
        try:
            return original_boost_chain(*args, **kwargs)
        finally:
            model.kan_steps = full_kan_steps

    model._new_learner = warm_new_learner
    model._fit_learner = warm_fit_learner
    model._boost_chain = warm_boost_chain
    try:
        model.fit(X, y, eval_set=eval_set, sample_weight=sample_weight)
    finally:
        # Remove the shadowing instance attributes entirely (rather than
        # reassigning the originals) so attribute lookup falls back to the
        # class's own bound methods -- leaving *any* callable in
        # `model.__dict__` breaks `model.save()`, which pickles
        # `self.__dict__` wholesale (`_base.py`'s `_freeze`) and can't
        # pickle a closure-holding local function.
        for name in ("_new_learner", "_fit_learner", "_boost_chain"):
            model.__dict__.pop(name, None)
        model.kan_steps = full_kan_steps
    return model
