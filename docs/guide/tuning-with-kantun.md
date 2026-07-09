# Tuning with kantun

Hyperparameter tuning lives in a separate sibling package,
[`kantun`](https://github.com/tuamah/kantun), so kanboost's own
dependency footprint stays minimal and kantun stays usable for tuning
other model types too. Install it separately:

```bash
pip install kantun
```

## Quickstart

```python
from kantun import KantunSearch
from kanboost import KANBoostClassifier

param_space = {
    "n_estimators": [30, 60, 100],
    "learning_rate": [0.1, 0.2, 0.3],
    "kan_hidden": [3, 4, 6],
    "kan_grid": [2, 3],
}

search = KantunSearch(KANBoostClassifier, param_space, n_iter=10, cv=3, scoring="auc")
search.fit(X, y)

print(search.best_params_, search.best_score_)
best_model = search.best_estimator_          # ready to use
results_df = search.results_dataframe()       # sorted leaderboard
```

`KantunSearch` auto-detects that `KANBoostClassifier.fit()` accepts
`eval_set` and passes it automatically on each fold so early stopping
kicks in during the search itself.

## Search types

- `search_type="random"` (default): samples `n_iter` random combinations
- `search_type="grid"`: tries every combination in `param_distributions`
- `search_type="halving"`: **successive halving** — starts every
  candidate on a small, stratified subsample of each fold's *training*
  data (held-out validation data is always full and untouched), keeps
  the top `1/halving_factor` by score, and grows the training subsample
  by `halving_factor` each round until a round trains on the full data.
  Training-set size is the resource halved (not, say, `n_estimators`),
  since it's the only resource meaningful for *any* estimator — kantun
  tunes arbitrary sklearn-compatible models, not just KANBoost.

  ```python
  search = KantunSearch(
      KANBoostClassifier, param_space, search_type="halving",
      n_iter=20, cv=3, halving_factor=3, min_resource=50,
  )
  ```

## Speeding up an expensive search

KANBoost trains roughly 10-20x slower than a tree ensemble per weak
learner, which makes an exhaustive search expensive. Two independent
knobs help, usable together or separately:

**`n_jobs`** — evaluate multiple param combos concurrently. Uses
threads, not processes: safe for CUDA device selection, and PyTorch
releases the GIL during tensor ops so real overlap still happens.

```python
search = KantunSearch(KANBoostClassifier, param_space, n_jobs=4)
```

**`prune=True`** — abandon a combo after its *first* CV fold if that
fold's score already falls more than `prune_margin` standard deviations
(of the current best combo's own fold spread) below the running best —
skips the remaining `cv - 1` folds for combos that are essentially never
going to become the best. A pruned combo's single-fold score is
recorded in `cv_results_` (`"pruned": True`) but never becomes
`best_params_`/`best_score_`. Off by default; the first combo evaluated
is never pruned (there's nothing to compare against yet).

```python
search = KantunSearch(KANBoostClassifier, param_space, prune=True, prune_margin=1.0)
```

## Supported scoring

- Classification: `"auc"` (default), `"f1"`, `"accuracy"`, or a callable
  `scorer(y_true, y_pred, y_prob, labels) -> float`.
- Regression: `"neg_mse"` (default), `"neg_mae"`, or a callable
  `scorer(y_true, y_pred) -> float`.

## Continuous parameter ranges

`param_distributions` values can be a callable `sampler(rng) -> value`
instead of a list, for `search_type="random"`/`"halving"`:

```python
param_space = {
    "n_estimators": [30, 60, 100],
    "kan_lr": lambda rng: 10 ** rng.uniform(-3, -1),   # log-uniform
}
search = KantunSearch(KANBoostClassifier, param_space, n_iter=20)
```

## Time budget and skipping the refit

```python
search = KantunSearch(
    KANBoostClassifier, param_space,
    time_budget_s=600,   # stop after ~10 minutes wall-clock
    refit=False,          # skip the final full-dataset refit
)
```

`time_budget_s` is checked between combos (or halving rungs), never
mid-fit, and always lets at least one combo/rung finish.

## Works with any sklearn-style estimator, not just KANBoost

```python
from sklearn.ensemble import RandomForestClassifier
from kantun import KantunSearch

search = KantunSearch(
    RandomForestClassifier,
    {"n_estimators": [50, 100], "max_depth": [3, 5, None]},
    n_iter=5, cv=3, scoring="f1",
    use_eval_set=False,   # RandomForestClassifier.fit() has no eval_set kwarg
)
search.fit(X, y)
```
