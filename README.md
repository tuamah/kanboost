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

**Read these tables honestly**: KANBoost does not consistently beat tuned
tree boosting on accuracy or speed. The value proposition is
interpretability and structural guarantees (monotonicity, exact additive
decomposition, analytic derivatives) that trees and MLPs
can't provide even in principle — not raw predictive performance.

## Honest limitations

- **Speed**: each weak learner is a full KAN forward/backward pass in
  pure PyTorch. This is currently far slower per-iteration than a
  histogram-based tree split in XGBoost/CatBoost/LightGBM.
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
