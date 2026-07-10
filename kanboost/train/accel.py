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

Implemented via the explicit `_learner_init_hook`/`_boost_chain_start_hook`
extension points on `_BaseKANBoost` (see `kanboost/core/base.py`) -- an
earlier version of this module monkey-patched the *instance's*
`_new_learner`/`_fit_learner`/`_boost_chain` bound methods wholesale for
the duration of one `fit()` call, which was fragile to any future change
in those methods' internals; these two hooks are a small, documented
contract instead. Either way, training still goes through the exact same
`_fit_learner`/`_fit_learner_custom_loop`/`_apply_monotone_projection`
machinery as a normal `fit()` (monotone constraints are enforced
identically; only how each learner's weights are *initialized* changes).
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
    state = {"round": 0, "prev": None}

    def learner_init_hook(learner):
        if state["prev"] is not None:
            with torch.no_grad():
                learner.load_state_dict(state["prev"].state_dict())
        state["round"] += 1
        model.kan_steps = full_kan_steps if state["round"] <= 1 else warm_start_steps
        # `learner` is the same mutable nn.Module object `_fit_learner`
        # will go on to train in place -- capturing the reference here
        # (not after training) is equivalent, since by the time the next
        # round's hook call reads `state["prev"]`, this round's
        # `_fit_learner` call has already mutated its parameters in place.
        state["prev"] = learner

    def boost_chain_start_hook():
        # Each call is a new one-vs-rest chain (multiclass) -- must not
        # warm-start one class's first learner from another class's last.
        state["round"] = 0
        state["prev"] = None
        model.kan_steps = full_kan_steps

    model._learner_init_hook = learner_init_hook
    model._boost_chain_start_hook = boost_chain_start_hook
    try:
        model.fit(X, y, eval_set=eval_set, sample_weight=sample_weight)
    finally:
        # Reset to None (the required default -- see core/base.py) rather
        # than leaving a closure behind: model.save() pickles
        # self.__dict__ wholesale, and a closure isn't picklable.
        model._learner_init_hook = None
        model._boost_chain_start_hook = None
        model.kan_steps = full_kan_steps
    return model
