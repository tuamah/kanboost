# Training speed

Profiling `fit()` shows the dominant per-round cost is each weak
learner's from-scratch Adam optimization loop — a fresh `KAN(...)` is
constructed every boosting round and run through the full `kan_steps`
budget, even though consecutive rounds share identical architecture
and, especially late in the ensemble, are fitting increasingly similar
residuals.

`kanboost.train.accel.fast_fit()` is an opt-in, drop-in replacement for
`model.fit(...)` that warm-starts each round's learner from the
previous round's fitted weights, so only the first round of a chain
needs the full step budget:

```python
from kanboost import KANBoostClassifier
from kanboost.train.accel import fast_fit

model = KANBoostClassifier(n_estimators=40, kan_steps=20)
fast_fit(model, X_train, y_train, eval_set=(X_val, y_val))
```

Measured on Breast Cancer Wisconsin (40 learners, `kan_steps=20`):
**3.37x faster** (11.70s → 3.48s) with AUC essentially unchanged
(0.9921 vs. 0.9893).

## How it works

`fast_fit` temporarily overrides the fitted model instance's
`_new_learner`/`_fit_learner`/`_boost_chain` methods for the duration
of one `fit()` call, then restores the originals — so it's implemented
entirely in a separate module, with zero edits to `_base.py`,
`classifier.py`, or `regressor.py`. Monotone constraints are enforced
identically to a normal `fit()` (`_apply_monotone_projection` still
runs after every optimizer step); only how each learner's weights are
*initialized* changes. Multiclass one-vs-rest chains are kept isolated
— a new class's chain never warm-starts from a different class's last
learner.

```python
fast_fit(model, X_train, y_train, warm_start_steps=5)  # override the default (kan_steps // 4)
```

## When to use it

This trades a small amount of per-round independence (classic boosting
fits each learner to the *current* residual from a fresh random init)
for speed. Always compare accuracy against a normal `fit()` on your own
data — it's a good fit for fast iteration during tuning
([`kantun`](tuning-with-kantun.md)), less clear-cut for a final
production model where the small accuracy delta matters more than
training wall-clock.

## Prediction speed: `consolidate_learners()`

`fast_fit` only addresses *training* time. For a model you're about to
deploy, `kanboost.train.consolidate.consolidate_learners()` shrinks the
already-fitted ensemble itself, cutting *prediction* time (and saved
model size) — it replaces consecutive groups of weak learners with one,
least-squares-refit to reproduce the group's summed output:

```python
from kanboost.train.consolidate import consolidate_learners

model.fit(X_train, y_train)
consolidate_learners(model, X_train, group_size=5)  # mutates model in place
model.predict_proba(X_test)  # same interface, fewer learners underneath
```

Measured on a real clinical benchmark (INSPIRE, `postop_icu`, 78K rows;
see the [INSPIRE gap-closing case study](../inspire_gap_closing_report.md)):
ensemble size 180→36, prediction time cut ~5x (54.1s→10.8s), for a small
accuracy cost (AUROC −0.0026, AUPRC −0.0056). Unlike
[`kanboost.interpret.editing.consolidate()`](editing-dashboard.md) (which
requires `gam=True` and returns a new `EditableGAM`), this works on any
fitted classifier/regressor and mutates `learners_` in place — no new
object, no editability, just fewer learners to evaluate at predict time.
As with `fast_fit`, this is a lossy approximation: compare accuracy
before/after on your own held-out data before relying on it.
