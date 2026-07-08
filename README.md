# KANBoost

**Gradient boosting with Kolmogorov-Arnold Network (KAN) learners** — an
interpretable, from-scratch alternative to tree-based boosting frameworks
(XGBoost, LightGBM, CatBoost).

Instead of decision trees as weak learners, KANBoost fits a sequence of
small, shallow [KAN](https://arxiv.org/abs/2404.19756) networks to the
pseudo-residuals of the previous stage, following the classic Friedman
(2001) gradient boosting recipe. Because each KAN edge is a learnable
univariate spline rather than an opaque weight, the resulting ensemble
exposes per-feature shape functions that are directly inspectable —
closer to a Generalized Additive Model than a black box.

> **Status: early-stage research project.** This is *not* a drop-in
> replacement for CatBoost/XGBoost in production. See
> [Benchmarks](#benchmarks) and [Honest limitations](#honest-limitations)
> below before using this for anything important.

## Why this exists

As of mid-2026, there is no widely-used, pip-installable library that
combines KAN with gradient boosting. A closely related idea was
published as **GB-KAN** (ICAART 2026), but no public code accompanies
that paper. KANBoost is an independent, from-scratch open-source
implementation of the same general idea, plus:

- automatic handling of categorical features (smoothed target-mean
  encoding, done fold-safe), instead of requiring manual one-hot encoding
- automatic handling of missing values (median imputation + optional
  missing-indicator columns), instead of requiring you to impute first
- built-in early stopping on a validation set
- approximate feature importances derived from learned spline coefficients

## Features

- **Binary and multiclass classification** (`KANBoostClassifier`,
  one-vs-rest for 3+ classes) and **regression** (`KANBoostRegressor`,
  squared-error or quantile/pinball loss)
- **GPU support** — `device="cuda"` (or `device=None` to auto-detect),
  falls back to CPU
- **Model persistence** — `model.save(path)` / `KANBoostClassifier.load(path)`
- **`sample_weight`** support in `fit()`
- **`validation_fraction`** — automatic internal train/validation split
  for early stopping when you don't have a separate `eval_set` handy
- **`batch_size`** — mini-batch training for larger datasets
- **Interpretability**: `model.feature_importances()` /
  `feature_importances_dict()`, `model.plot_feature(name)` for a
  partial-dependence-style curve of a single feature's learned response,
  and `model.feature_contributions(X)` for native per-sample,
  per-feature attribution (not a post-hoc method like SHAP)
- **Hard monotonic constraints** — `monotone_constraints={"feature": 1|-1}`
  (requires `gam=True`), enforced by projecting each edge's B-spline
  control points onto the monotone cone every step — a real guarantee,
  not a penalty
- **GAM mode** (`gam=True`) — fixes each learner's output edge to
  identity, making the ensemble an exact additive model
  `F(x) = c + sum_j g_j(x_j)`; combine with `model.symbolic_report(X)`
  to fit closed-form functions (`sin`, `x^2`, `tanh`, ...) to each
  feature's learned shape function
- **`model.predict_derivative(X, feature)`** — analytic, exact derivative
  curves (trees have none; MLPs only give pointwise autograd gradients)
- **`model.refine(X, new_grid)`** / **`model.prune(X, threshold)`** —
  near-losslessly re-express a fitted ensemble on a finer spline grid, or
  zero out dead edges post-hoc, without retraining from scratch
- **`model.feature_interaction(X)`** — native structural interaction
  scores read off the trained weights (`kan_hidden > 1`)
- **`lamb`/`lamb_l1`/`lamb_coefdiff`** — tunable smoothness/sparsity
  regularization on the learned splines (pykan's own regularizers)
- **`kanboost.editing.consolidate(model)`** (requires `gam=True`) — collapse
  a fitted ensemble's per-feature shape functions (each currently a sum of
  splines across every boosting round) into one editable spline per
  feature, wrapped in an `EditableGAM`: shift/pin a region
  (`set_offset`/`set_values`), re-enforce hard monotonicity after an edit
  (`enforce_monotone`, same guarantee as `monotone_constraints`), inspect
  the effect (`diff`), and predict/save/load exactly like the original
  model. See [Editable models](#editable-models-human-in-the-loop) below.
- **`kanboost.symbolic.export_symbolic(model)`** (requires `gam=True`) —
  a real, executable symbolic formula: `sympy` expression, LaTeX,
  standalone numpy predict function, and a per-feature fidelity report
  (closed-form R^2 + amplitude, with a numeric fallback for features no
  candidate fits well). See
  [Symbolic formula export](#symbolic-formula-export-optional-additive)
  below.
- **`kanboost.calibration.calibrate(model, X_cal, y_cal)`** — post-hoc
  Platt/isotonic probability calibration for `KANBoostClassifier`; fixes
  a real, benchmark-confirmed miscalibration gap without retraining. See
  [Calibration](#calibration-optional-additive) below.
- Automatic categorical encoding and missing-value handling, no manual
  preprocessing required

## Install

```bash
pip install kanboost
```

Add `pip install kanboost[api]` if you also want the optional FastAPI
serving layer (see [Serving & observability](#serving--observability-optional-additive)).

Or from source:

```bash
git clone https://github.com/tuamah/kanboost.git
cd kanboost
pip install -e .
```

## Quickstart

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from kanboost import KANBoostClassifier

df = pd.read_csv("your_data.csv")
X = df.drop(columns=["target"])
y = df["target"].values  # binary or multiclass; NaN in X is handled automatically

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

model = KANBoostClassifier(
    n_estimators=100,
    learning_rate=0.2,
    kan_hidden=4,
    kan_grid=3,
    categorical_cols=["region", "plan_type"],  # optional
    early_stopping_rounds=10,
    device="cuda",  # or None to auto-detect, "cpu" to force CPU
)
model.fit(X_train, y_train, eval_set=(X_val, y_val))  # sample_weight=... optional

probs = model.predict_proba(X_val)  # shape (n, 2) binary, (n, n_classes) multiclass
importances = model.feature_importances_dict()

model.save("model.pt")
loaded = KANBoostClassifier.load("model.pt")  # device=... to override where it loads

model.plot_feature("region")  # matplotlib partial-dependence plot
```

## Serving & observability (optional, additive)

These live in their own modules and never modify or depend on private
training/inference internals beyond `model.verbose`/`model._fit_learner`
existing -- nothing here changes how `fit`/`predict` behave.

**Observability** (`kanboost.observability`, no extra install needed):

```python
from kanboost.observability import (
    time_predict, memory_snapshot, gpu_utilization_flag, capture_boosting_rounds,
)

preds, metrics = time_predict(model, X_val, method="predict_proba")
print(metrics.elapsed_seconds, metrics.samples_per_second, metrics.device)

print(memory_snapshot())        # process RSS + CUDA allocator stats
print(gpu_utilization_flag(model))  # cuda_available, device_name, model_on_gpu

with capture_boosting_rounds(model) as rounds:
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
for r in rounds:
    print(r.round, r.elapsed_seconds, r.loss, r.gpu_allocated_mb)
```

**Logging** (`kanboost.logging_utils`, stdlib only):

```python
from kanboost.logging_utils import get_logger, log_boosting_rounds

logger = get_logger("my_experiment")  # respects KANBOOST_LOG_LEVEL env var
log_boosting_rounds(rounds, logger=logger, model_name="churn_v3")
```

**Serving** (`kanboost.serving`, needs `pip install kanboost[api]`):

```python
from kanboost.serving import create_app

app = create_app("model.pt")  # auto-detects classifier vs. regressor
# uvicorn.run(app, host="0.0.0.0", port=8000)
```

or as a uvicorn target directly:

```bash
KANBOOST_MODEL_PATH=model.pt uvicorn kanboost.serving:app
```

Endpoints: `GET /health`, `POST /predict` (`{"records": [{"col": val, ...}]}`),
and `POST /predict_proba` (classifiers only).

## Editable models (human-in-the-loop)

`kanboost.editing.consolidate(model)` collapses a fitted `gam=True`
ensemble's per-feature shape function -- currently a sum of splines
across every boosting round -- into one editable spline per feature.
This is conceptually similar to Microsoft's [GAM Changer](https://github.com/interpretml/gam-changer),
an editing tool for EBM (Explainable Boosting Machine): both let a
domain expert directly reshape a model's per-feature curves. The
difference is what happens *after* an edit. EBM's shape functions are
piecewise-constant bins, so checking monotonicity there is just
comparing adjacent bins -- there's no notion of smoothness or
between-point behavior to verify, because there's no continuous curve
in the first place. KANBoost's feature is a genuine continuous
B-spline, so `enforce_monotone` re-derives a provably monotone
coefficient sequence after an edit -- guaranteed for *every* point on
the curve, not just at the sampled locations used to build it -- the
same variation-diminishing projection `monotone_constraints` uses
during training, not a best-effort correction.

```python
from kanboost.editing import consolidate

model = KANBoostRegressor(gam=True, kan_hidden=1, n_estimators=50)
model.fit(X_train, y_train)

gam = consolidate(model)  # multiclass classifier -> {class_label: EditableGAM}
print(gam.max_consolidation_error())  # worst per-feature fit error (call with feature=... for one feature)

gam.set_offset("age", x_range=(-0.2, 0.3), delta=0.5)   # shift a region
gam.set_values("region", x_range=(0.6, 1.0), value=0.0)  # pin a region flat
gam.enforce_monotone("income", sign=1)  # re-derive a provably monotone curve

report = gam.diff(X_val, y_val)  # per-feature deltas + before/after metric
gam.predict(X_val)                # exact, same interface as the original model
gam.save("edited_model.pt")
```

`consolidate()` is also, incidentally, a fast predict path for `gam=True`
models: one B-spline evaluation per feature instead of `n_estimators`
full KAN forward passes. Measured **~30-50x faster prediction** than the
original ensemble (varies by hardware/model size; 1000-row, 6-feature,
40-estimator model), with a consolidation fidelity cost around 1e-6
(`gam.max_consolidation_error()`)
-- see [Honest limitations](#honest-limitations) for KANBoost's
predict-time gap against tree ensembles, which this closes for GAM-mode
models specifically.

## Symbolic formula export (optional, additive)

`model.symbolic_report(X)` / `kanboost.experimental.symbolic_export`
give a human-readable summary of each feature's best-fitting named
function. `kanboost.symbolic.export_symbolic` goes further: an actual
executable formula.

```python
from kanboost.symbolic import export_symbolic

model = KANBoostRegressor(gam=True, kan_hidden=1, n_estimators=50)
model.fit(X_train, y_train)

sym = export_symbolic(model, min_r2=0.85)  # multiclass classifier -> {class_label: SymbolicModel}

print(sym.to_sympy())          # a real sympy expression
print(sym.to_latex())          # ready to paste into a paper
print(sym.fidelity_report())   # {feature: {"kind", "r2", "candidate", "amplitude"}}
print(sym.symbolic_fraction()) # fraction of features that got a true closed form

sym.predict(X_test)            # standalone -- no torch/pykan needed at call time
sym.save("formula.pt")
```

Each feature's exact aggregated shape function (the same one
`plot_feature`/`symbolic_report` use) gets one closed-form candidate
fit (`c * fun(a*x + b) + d`, pykan's own `SYMBOLIC_LIB`) if some
candidate clears `min_r2`; otherwise it's kept as a numeric
(spline-interpolated) term instead of forced into a misleading formula.
`fidelity_report()`'s `amplitude` field matters alongside `r2` --  a
near-flat, unimportant feature can still score a deceptively high R^2
by fitting its own tiny wiggles, so check amplitude against other
features' before treating a high-R^2 term as meaningful. Because this
refits every feature's spline as a parametric approximation,
`sym.predict()` is a *lossy* approximation of the original model
(unlike `EditableGAM.predict`, which is exact) -- `fidelity_report()`
tells you how lossy, per feature.

For a quick, ranked summary instead of the full formula:

```python
from kanboost.symbolic import explain

for entry in explain(model, top_features=5, symbolic=True, simplify=True):
    print(entry["feature"], entry["importance"], entry["formula"])
```

`explain()` ranks features by `model.feature_importances_dict()` and
attaches each top feature's symbolic term (`simplify=True` runs
`sympy.simplify()` on it -- cheap here, since it's one term, not the
whole model). `symbolic=False` skips formula extraction entirely, for
a plain top-`N` importance ranking. Multiclass: uses each feature's term
from `model.classes_[0]`'s chain as a representative formula (one-vs-rest
chains can fit a feature differently per class) -- call
`export_symbolic(model)` directly and index by class for a true
per-class formula.

## Calibration (optional, additive)

Three independent benchmarks (see [Benchmarks](#benchmarks)) found the
same pattern: KANBoost's `predict_proba` *ranking* (ROC AUC/PR AUC) is
competitive with or ahead of tuned tree ensembles, but its raw
probability *values* are comparatively miscalibrated -- worst Brier
score and log-loss in all three runs, with the F1-optimal decision
threshold sitting around 0.40-0.42 rather than 0.5. `kanboost.calibration`
fixes this post-hoc, without retraining:

```python
from kanboost.calibration import calibrate

model = KANBoostClassifier(n_estimators=100)
model.fit(X_train, y_train)

# X_cal/y_cal must be held out -- not used in model.fit()
cal_model = calibrate(model, X_cal, y_cal, method="platt")  # or method="isotonic"

cal_model.predict_proba(X_test)  # calibrated probabilities
cal_model.predict(X_test)         # same threshold semantics as the base model
cal_model.save("calibrated_model.pt")
loaded = CalibratedKANBoost.load("calibrated_model.pt")
```

`method="platt"` (default) fits a 2-parameter logistic rescaling of the
raw score -- the right fix for a systematic shift like KANBoost's, needs
little calibration data, and being strictly monotone, leaves ROC AUC/PR
AUC exactly unchanged (verified in tests). `method="isotonic"` is more
flexible but needs a larger `X_cal` (order of 1000+ rows) to avoid
overfitting. On a held-out Breast Cancer split: Brier score
0.090 -> 0.030, log-loss 0.344 -> 0.119, ROC AUC unchanged, with Platt.

Multiclass: each one-vs-rest chain is calibrated independently, then
rows are renormalized to sum to 1. If you also use `kanboost.editing`,
calibrate *after* finalizing any `EditableGAM` edits, not before -- an
edit changes the model's raw scores and would silently stale an
already-fitted calibration map.

## Experimental utilities (optional, additive)

`kanboost.experimental` is a small toolkit of convenience functions
built entirely on the public methods above -- nothing here needs core
changes, and `suggest_constraints` in particular is a heuristic, not a
guarantee: always confirm with `audit_monotonicity` on a model actually
fit with the suggested constraints.

```python
from kanboost.experimental import (
    suggest_constraints, audit_monotonicity, symbolic_export,
    predict_interval, explain_row, dashboard_html,
)

# suggest which features look monotone in the raw data (advisory only)
constraints = suggest_constraints(X_train, y_train)

model = KANBoostRegressor(gam=True, kan_hidden=1, monotone_constraints=constraints)
model.fit(X_train, y_train)

# verify the constraint actually held on held-out data (not just training data)
print(audit_monotonicity(model, X_test))

print(symbolic_export(model, X_test))                  # compact human-readable summary
print(explain_row(model, X_test, row_index=0))          # top feature contributions for one row
predict_interval([model_seed0, model_seed1], X_test)     # mean/lower/upper/std across models
dashboard_html(model, X_test, y_test, path="report.html")  # one static HTML report
```

## Interactive dashboard (optional, additive)

`dashboard_html` above is a zero-dependency static snapshot -- good for
sharing or archiving in CI. `kanboost.dashboard` is a live, local
Streamlit app for actually exploring one of your own fitted models:
feature importances, `plot_feature` curves, `symbolic_report` (GAM
mode), `feature_interaction`, per-row `explain_row`, and -- for a
single-chain `gam=True` model (regressor or binary classifier; not yet
multiclass) -- a panel to live-edit shape functions via
`kanboost.editing.EditableGAM` (`set_offset`, `enforce_monotone`, `diff`,
`save`), with the before/after curve redrawn immediately. Requires
`pip install kanboost[dashboard]`.

```python
from kanboost.dashboard import launch

launch("model.pt")                      # opens a local browser tab
launch("model.pt", data_path="X.csv")   # preload a dataset to explore
```

or from the command line: `python -m kanboost.dashboard model.pt X.csv`

This runs a local server for one person exploring one model, not a
hosted multi-tenant service -- see [Serving](#serving--observability-optional-additive)
for that.

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
[`examples/benchmark_uci.py`](./examples/benchmark_uci.py), reproducible
in one run (`kan_hidden=1`, `n_estimators=60`, `kan_steps=15`,
`batch_size=2048`):

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

**Practical takeaway**: don't threshold KANBoost's `predict_proba` output
at the default 0.5 for classification metrics — tune the decision
threshold on a validation set (or apply a post-hoc calibration step like
Platt scaling/isotonic regression) the way you would for any model with
a known calibration gap. Its probability *ranking* can be trusted at
face value; its probability *values* currently can't be, out of the box.

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

**An emerging pattern worth taking seriously, not yet proven**: across
all four Breast-Cancer-family runs plus this new multi-dataset one,
KANBoost's *relative* standing tracks dataset size — decisive win on
the smallest set (270 rows), a near-tie for best on a mid-small one
(569 rows), and merely mid-pack on the largest of the three (768 rows).
This lines up with the small-data observation already made about
Breast Cancer vs. Adult Income/California Housing earlier in this
section: fewer rows seem to blunt tree ensembles' usual edge more than
they blunt KANBoost's. Three datasets and one 8-model comparison is not
enough to call this a rule -- it's a pattern to test further, not a
guarantee.

**Read these tables honestly**: KANBoost does not consistently beat tuned
tree boosting on accuracy or speed. The value proposition is
interpretability and structural guarantees (monotonicity, exact additive
decomposition, analytic derivatives) that trees and MLPs
can't provide even in principle — not raw predictive performance.

## Honest limitations

- **Speed**: each weak learner is a full KAN forward/backward pass in
  pure PyTorch. This is currently far slower per-iteration than a
  histogram-based tree split in XGBoost/CatBoost/LightGBM -- and, per
  the Breast Cancer cross-validated benchmark above, prediction time is
  markedly slower too, not just training.
- **Probability calibration**: `predict_proba`'s *ranking* (AUC) is
  competitive with or ahead of tuned tree ensembles, but the raw
  probability *values* are comparatively miscalibrated out of the box
  (worst Brier score in the Breast Cancer benchmark above, with the
  per-fold F1-optimal threshold sitting well below the default 0.5).
  Use `kanboost.calibration.calibrate()` (see
  [Calibration](#calibration-optional-additive) above) or tune the
  decision threshold yourself before relying on classification metrics
  at the default cutoff.
- **Prediction speed for `gam=True` models specifically has a fix**:
  `kanboost.editing.consolidate()` (see
  [Editable models](#editable-models-human-in-the-loop) above) is also
  a ~30-50x-faster predict path for GAM-mode models, at negligible fidelity
  cost -- it doesn't help non-GAM models or training speed.
- **Tuning**: hyperparameters (`kan_grid`, `kan_hidden`, `kan_steps`,
  `learning_rate`) interact in ways that are not yet well understood;
  expect to need real tuning for your dataset.
- **Categorical encoding** is a simple smoothed target-mean encoder, not
  CatBoost's ordered boosting scheme — it can leak on small folds if not
  used carefully.
- **Monotonic constraints require `gam=True` and `kan_hidden=1`** — the
  guarantee only holds for a pure additive ensemble; it can't be made
  sound through a hidden layer that mixes features.
- Multiclass classification is one-vs-rest (independent binary chains
  combined via softmax), not a single joint softmax objective, and no
  user-pluggable custom loss functions yet (only squared-error/quantile
  for regression, logloss for classification).

## Roadmap

See [`ROADMAP.md`](./ROADMAP.md) for the full project plan, including
planned speed optimizations (FastKAN-style RBF basis, `torch.compile`),
symbolic-formula extraction for the full ensemble, and benchmark
expansion to standard UCI datasets.

## Contributing

Issues and PRs welcome, especially:
- speed optimizations for the per-iteration KAN fit
- better categorical encoding
- benchmark results on additional public datasets

## License

MIT — see [`LICENSE`](./LICENSE).

## Citation / related work

If you use this, please also cite the KAN paper and, where relevant,
the GB-KAN paper this project is conceptually closest to:

```
Liu, Z., Wang, Y., Vaidya, S., et al. (2024). KAN: Kolmogorov-Arnold
Networks. arXiv:2404.19756.

[GB-KAN authors] (2026). Gradient Boosting with Interpretable
Kolmogorov-Arnold Networks. ICAART 2026.
```
