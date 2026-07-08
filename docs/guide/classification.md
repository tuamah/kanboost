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

A second, independent run on the same dataset with a stricter
methodology — the decision threshold picked on a held-out *validation*
split, then applied once to a separate *test* split (rather than the
per-fold-optimal-on-test-itself threshold above, which is a slightly
more optimistic setup) — both confirms and strengthens this picture:

| Model | Test ROC AUC | Test Brier | Threshold (from val) | Test accuracy @ that threshold | Test F1 | Test MCC |
|---|---|---|---|---|---|---|
| RandomForest | **0.9983** | 0.0274 | 0.440 | 0.9649 | 0.9512 | 0.9245 |
| **KANBoost (tuned)** | 0.9980 | **0.0578** | 0.415 | **0.9825** | **0.9756** | **0.9626** |
| LogReg (scaled) | 0.9954 | 0.0222 | 0.730 | 0.9649 | 0.9500 | 0.9258 |
| HistGradientBoosting | 0.9940 | 0.0296 | 0.235 | 0.9737 | 0.9630 | 0.9442 |
| LightGBM | 0.9940 | **0.0198** | 0.270 | **0.9825** | **0.9756** | **0.9626** |
| XGBoost | 0.9931 | 0.0199 | 0.280 | **0.9825** | **0.9756** | **0.9626** |
| MLP (scaled) | 0.9894 | 0.0980 | 0.555 | 0.9386 | 0.9136 | 0.8675 |

Brier score replicates the calibration gap independently — still
clearly the worst of the group, confirming it's a real, repeatable
property of KANBoost's raw probability outputs, not an artifact of one
experimental setup. But with an honestly-selected (validation-derived,
not test-leaked) threshold, KANBoost's classification metrics
(accuracy/F1/MCC) come out in an exact three-way tie for **best in the
entire comparison**, matching LightGBM and XGBoost and ahead of
RandomForest, LogReg, HistGradientBoosting, and MLP — while its ROC AUC
is second only to RandomForest. Threshold calibration isn't a marginal
tweak here; it's the difference between mediocre and top-tier
classification performance for KANBoost specifically.

A **third**, independent CV run on the same dataset (8 models including
CatBoost, mean ± std over folds, log-loss added alongside Brier)
confirms the pattern again, and sharpens it:

| Model | ROC AUC | PR AUC | F1 @ 0.5 | Log loss | Brier |
|---|---|---|---|---|---|
| **KANBoost (tuned)** | **0.9960** | **0.9948** | 0.9309 | **0.3628** | **0.0971** |
| LogReg (scaled) | 0.9951 | 0.9939 | **0.9620** | 0.0796 | 0.0213 |
| CatBoost | 0.9947 | 0.9934 | 0.9474 | 0.0929 | 0.0257 |
| XGBoost | 0.9935 | 0.9914 | 0.9499 | 0.0967 | 0.0272 |
| HistGradientBoosting | 0.9928 | 0.9909 | 0.9464 | 0.1292 | 0.0285 |
| LightGBM | 0.9916 | 0.9898 | 0.9490 | 0.1501 | 0.0306 |
| RandomForest | 0.9904 | 0.9889 | 0.9430 | 0.1253 | 0.0336 |
| MLP (scaled) | 0.9855 | 0.9831 | 0.9284 | 0.2218 | 0.0580 |

This time KANBoost's ROC AUC and PR AUC are the **highest of all 8
models** — including CatBoost and LogReg. But log loss (which, unlike
Brier, penalizes confidently-wrong probabilities heavily) is nearly 2x
worse than the next-worst model (MLP) and 4-5x worse than the tree
ensembles — the starkest evidence yet, across a third independent
methodology, that KANBoost's ranking and its raw probability confidence
are two very different things. The practical guidance stands regardless
of which of these three runs you look at: trust the ranking, calibrate
or threshold-tune before trusting the raw probability values.

A fourth benchmark, this time spanning **three separate datasets** (not
repeated runs of one) with **Wilcoxon signed-rank significance testing**
(5-fold, `KANBoost` vs. each of 7 other models, per dataset) rather than
just comparing means:

| Dataset (rows) | KANBoost ROC AUC | Best of the other 7 | KANBoost rank | Brier (KANBoost) | Fit time (KANBoost) |
|---|---|---|---|---|---|
| Heart-Statlog (270) | **0.9181** | LogReg 0.9053 | **1st of 8** | 0.1253 | 9.8s |
| Breast Cancer Wisconsin (569) | 0.9945 | LogReg 0.9946 (~tied) | 1st–2nd of 8 | 0.0563 | 13.4s |
| Diabetes / Pima (768) | 0.8290 | CatBoost 0.8402 | 5th of 8 | 0.1659 | 12.2s |

On Heart-Statlog specifically, KANBoost's ROC AUC beat **every one of
the other 7 models on every one of the 5 folds** — `p=0.0625` against
all seven, the smallest p-value obtainable with 5 paired folds (i.e.
maximally significant given the sample size). Brier score again lands
in the worse half of the pack on the two datasets where it's not
best-in-class, consistent with the calibration gap documented above.

!!! note "An emerging pattern worth taking seriously, not yet proven"
    Across all four Breast-Cancer-family runs plus this new
    multi-dataset one, KANBoost's *relative* standing tracks dataset
    size — decisive win on the smallest set (270 rows), a near-tie for
    best on a mid-small one (569 rows), and merely mid-pack on the
    largest of the three (768 rows). This lines up with the small-data
    observation already made about Breast Cancer vs. Adult Income/
    California Housing earlier in this page: fewer rows seem to blunt
    tree ensembles' usual edge more than they blunt KANBoost's. Three
    datasets and one 8-model comparison is not enough to call this a
    rule — it's a pattern to test further, not a guarantee.

**Read these tables honestly**: KANBoost does not consistently beat tuned
tree boosting on accuracy or speed. The value proposition is
interpretability and structural guarantees (monotonicity, exact additive
decomposition, analytic derivatives) that trees and MLPs
can't provide even in principle — not raw predictive performance.
