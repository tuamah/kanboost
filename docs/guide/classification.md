# Classification & regression

## `KANBoostClassifier`

Binary and multiclass (one-vs-rest, `n_classes` independent binary
boosting chains combined via softmax at prediction time).

```python
from kanboost import KANBoostClassifier

model = KANBoostClassifier(
    n_estimators=100,
    learning_rate=0.2,
    kan_hidden=4,
    kan_grid=3,
    categorical_cols=["region", "plan_type"],  # optional, auto-encoded
    early_stopping_rounds=10,
    device="cuda",  # or None to auto-detect, "cpu" to force CPU
    batch_size=2048,  # optional: mini-batch training for large datasets
)
model.fit(X_train, y_train, eval_set=(X_val, y_val), sample_weight=None)

probs = model.predict_proba(X_val)  # (n, 2) binary, (n, n_classes) multiclass
labels = model.predict(X_val)
```

`validation_fraction` carves out an internal train/validation split
automatically (stratified) if you don't want to build `eval_set`
yourself:

```python
model = KANBoostClassifier(validation_fraction=0.15, early_stopping_rounds=10)
model.fit(X_train, y_train)  # no eval_set needed
```

## `KANBoostRegressor`

```python
from kanboost import KANBoostRegressor

model = KANBoostRegressor(objective="squared_error")  # or objective="quantile", alpha=0.9
model.fit(X_train, y_train, eval_set=(X_val, y_val))
preds = model.predict(X_val)
report = model.evaluate(X_val, y_val)  # MSE, RMSE, MAE, R^2 (+ pinball for quantile)
```

## Persistence

```python
model.save("model.pt")
loaded = KANBoostClassifier.load("model.pt")  # device=... to override where it loads
```

Learners are stored as `state_dict`s (pykan's `KAN` isn't directly
picklable) and rebuilt on load.

## Benchmarks

Preliminary results on a real-world telecom churn dataset (100K rows,
10 numeric features used, 8K-row sample for the KANBoost run due to
current training-speed limits):

| Model | Test AUC | Notes |
|---|---|---|
| CatBoost (tuned, full data, ~100 columns) | **0.6992** | production baseline |
| KANBoostClassifier (this repo, 10 features, 8K sample) | 0.64 | early prototype, untuned |
| Plain KAN (no boosting) | 0.65 | single model, same features |
| Plain MLP | 0.59–0.62 | same features |

Standard UCI-style datasets, KANBoost vs. sklearn's
`HistGradientBoosting*` (untuned defaults) as a sanity floor — see
[`examples/benchmark_uci.py`](https://github.com/tuamah/kanboost/blob/main/examples/benchmark_uci.py),
reproducible in one run (`kan_hidden=1`, `n_estimators=60`,
`kan_steps=15`, `batch_size=2048`):

| Dataset | Metric | KANBoost | HistGradientBoosting | KANBoost train time |
|---|---|---|---|---|
| Adult Income (10K-row train sample, 48K total) | AUC | 0.884 | 0.919 | ~17s |
| California Housing (full, 20.6K rows) | R² | 0.639 | 0.836 | ~13s |
| Breast Cancer Wisconsin (full, 569 rows) | AUC | **0.9954** | 0.9931 | ~11s |

A separate, fully independent real-world test (an NFL Draft prediction
dataset, ~2.8K rows, 80 engineered features after preprocessing, 5-fold
CV, `n_estimators=300`, `early_stopping_rounds=30`,
`validation_fraction=0.15`), comparing against tuned CatBoost rather than
untuned HistGradientBoosting:

| Model | Mean CV AUC | OOF AUC | Time per fold |
|---|---|---|---|
| CatBoost (tuned) | **0.83880** | 0.81961 | 2.4–7.8s |
| KANBoost | 0.83153 | **0.83002** | 84–90s |

KANBoost trailed on 4 of 5 individual folds by under 0.5 points, and
actually edged CatBoost out on OOF AUC (the metric computed on all
pooled out-of-fold predictions at once, rather than averaged per-fold) —
at roughly 17–20x the training time. Consistent with the UCI results
above: KANBoost's accuracy is competitive, not the reason to reach for
it; the ~20x slowdown is real and dataset-independent so far.

On the small-data end (Breast Cancer, 569 rows), KANBoost's smaller
per-round learner capacity stops being a handicap and it edges out the
tree baseline — the two larger datasets show the more typical pattern of
tree boosting ahead on both accuracy and speed.

Also in that script: a `monotone_constraints={"MedInc": 1}` model on
California Housing, verified via `predict_derivative` to have a
non-negative derivative (min ≈ +0.50) on the *held-out test set* — a
hard structural guarantee tree-boosting libraries can't offer.

A separate, more rigorous cross-validated benchmark on Breast Cancer
Wisconsin (mean ± std over folds, tuned KANBoost vs. tuned tree
ensembles and a scaled logistic regression/MLP baseline) surfaces a
genuine, previously undocumented finding — a calibration gap, not just
a speed one:

| Model | ROC AUC | F1 @ 0.5 | F1 @ best per-fold threshold | Brier score | Fit time |
|---|---|---|---|---|---|
| LogReg (scaled) | 0.9954 | 0.9616 | **0.9823** | **0.0201** | 0.02s |
| **KANBoost (tuned)** | **0.9940** | 0.9476 | 0.9706 | 0.0578 | 30.4s |
| LightGBM | 0.9936 | 0.9551 | 0.9680 | 0.0246 | 0.14s |
| XGBoost | 0.9934 | **0.9614** | 0.9673 | 0.0228 | 0.28s |
| HistGradientBoosting | 0.9937 | 0.9518 | 0.9619 | 0.0300 | 0.33s |
| RandomForest | 0.9881 | 0.9406 | 0.9579 | 0.0331 | 1.49s |
| MLP (scaled) | 0.9751 | 0.8951 | 0.9345 | 0.1151 | 0.05s |

KANBoost's *ranking* ability (ROC AUC, PR AUC) is genuinely excellent
here — second only to logistic regression, ahead of every tree
ensemble. But its raw probabilities are comparatively **miscalibrated**:
worst Brier score of the group, and the per-fold F1-optimal decision
threshold averaged **0.405**, not 0.5 — a real, systematic skew, not
noise. At the default 0.5 cutoff its F1 looks mediocre; using each
fold's own optimal threshold instead, F1 and MCC both jump to
second-best overall, ahead of LightGBM, XGBoost, and
HistGradientBoosting.

!!! tip "Don't threshold at 0.5 without checking"
    Tune the decision threshold on a validation set (or apply a
    post-hoc calibration step like Platt scaling/isotonic regression)
    the way you would for any model with a known calibration gap.
    KANBoost's probability *ranking* can be trusted at face value; its
    probability *values* currently can't be, out of the box.

Also notable, and not previously measured: **prediction time**, not just
fit time, is markedly slower here too (~0.99s vs. 0.006–0.12s for the
tree ensembles) — roughly two orders of magnitude, distinct from the
already-documented training-speed gap.

**Read these tables honestly**: KANBoost does not consistently beat tuned
tree boosting on accuracy or speed. The value proposition is
interpretability and structural guarantees (monotonicity, exact additive
decomposition, analytic derivatives) that trees and MLPs
can't provide even in principle — not raw predictive performance.
