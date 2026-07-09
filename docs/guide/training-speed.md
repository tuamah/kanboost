# Training speed

Profiling `fit()` shows the dominant per-round cost is each weak
learner's from-scratch Adam optimization loop — a fresh `KAN(...)` is
constructed every boosting round and run through the full `kan_steps`
budget, even though consecutive rounds share identical architecture
and, especially late in the ensemble, are fitting increasingly similar
residuals.

`kanboost.accel.fast_fit()` is an opt-in, drop-in replacement for
`model.fit(...)` that warm-starts each round's learner from the
previous round's fitted weights, so only the first round of a chain
needs the full step budget:

```python
from kanboost import KANBoostClassifier
from kanboost.accel import fast_fit

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
