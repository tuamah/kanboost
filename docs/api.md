# API reference

This is a hand-written summary of the public surface; see the
[Guide](guide/classification.md) for full usage examples of each.

## Core

- `kanboost.KANBoostClassifier(...)` — binary/multiclass classification
- `kanboost.KANBoostRegressor(...)` — regression (squared-error or quantile)
- `kanboost.classification_report_dict(...)` / `print_classification_report(...)`

Both estimators share:

| Method | Description |
|---|---|
| `.fit(X, y, eval_set=None, sample_weight=None)` | Fit the boosted ensemble |
| `.predict(X)` | Point predictions |
| `.predict_proba(X)` (classifier only) | Class probabilities |
| `.save(path)` / `.load(path)` (classmethod) | Persistence |
| `.feature_importances()` / `.feature_importances_dict()` | Approximate importances |
| `.plot_feature(name)` | Partial-dependence-style plot |
| `.feature_contributions(X)` | Native per-sample attribution |
| `.feature_interaction(X)` | Structural interaction scores (`kan_hidden > 1`) |
| `.predict_derivative(X, feature)` | Analytic derivative curve |
| `.symbolic_report(X)` | Closed-form shape-function fits (GAM mode) |
| `.refine(X, new_grid)` / `.prune(X, threshold)` | Post-hoc model surgery |

Key constructor parameters: `n_estimators`, `learning_rate`, `kan_hidden`,
`kan_grid`, `kan_k`, `kan_steps`, `kan_lr`, `early_stopping_rounds`,
`validation_fraction`, `categorical_cols`, `device`, `batch_size`, `gam`,
`monotone_constraints`, `lamb`/`lamb_l1`/`lamb_coefdiff`.

## Optional, additive modules

| Module | Extra | Purpose |
|---|---|---|
| `kanboost.observability` | none | Timing, memory, GPU, per-round metrics |
| `kanboost.logging_utils` | none | stdlib logging wrapper |
| `kanboost.serving` | `kanboost[api]` | FastAPI serving layer |
| `kanboost.editing` | none | `consolidate()` / `EditableGAM` |
| `kanboost.symbolic` | `sympy` (core dependency) | `export_symbolic()` / `SymbolicModel` |
| `kanboost.calibration` | none | `calibrate()` / `CalibratedKANBoost` (Platt/isotonic) |
| `kanboost.experimental` | none (needs `scipy`) | `suggest_constraints`, `audit_monotonicity`, `symbolic_export`, `predict_interval`, `explain_row`, `dashboard_html` |
| `kanboost.dashboard` | `kanboost[dashboard]` | Interactive Streamlit app |
| `kanboost.mlhub` | `kanboost[mlhub]` | Push/pull a model to a MinIO-backed object store |
| `kanboost.mlflow_utils` | `kanboost[mlflow]` | Log a training run to MLflow |

See the [Guide](guide/interpretability.md) for each module's full API
with examples.

## `kantun` (separate package)

`kantun.KantunSearch(model_cls, param_distributions, ...)` — see
[Tuning with kantun](guide/tuning-with-kantun.md).
