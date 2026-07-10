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

As of v1.0.0, these live under `kanboost.core`/`kanboost.interpret`/
`kanboost.train`/`kanboost.ops`/`kanboost.registry` (see
[MIGRATION.md](https://github.com/tuamah/kanboost/blob/main/MIGRATION.md)
for the full old -> new path mapping). Each subpackage re-exports its
own contents one level flat.

| Module | Extra | Purpose |
|---|---|---|
| `kanboost.ops.observability` | none | Timing, memory, GPU, per-round metrics |
| `kanboost.ops.logging_utils` | none | stdlib logging wrapper |
| `kanboost.ops.serving` | `kanboost[api]` | FastAPI serving layer |
| `kanboost.interpret.editing` | none | `consolidate()` / `EditableGAM` |
| `kanboost.interpret.symbolic` | `sympy`, `scipy` (both core dependencies) | `export_symbolic()` / `explain()` / `symbolic_summary()` / `refit_constants()` / `refit_constants_from_model()` / `formula_fidelity()` / `stability_across_seeds()` / `stability_across_sample_sizes()` / `distill_equation()` / `SymbolicModel` |
| `kanboost.train.calibration` | none | `calibrate()` / `CalibratedKANBoost` (Platt/isotonic) |
| `kanboost.interpret.experimental` | none (needs `scipy`) | `suggest_constraints`, `audit_monotonicity`, `symbolic_export`, `predict_interval`, `explain_row`, `dashboard_html` |
| `kanboost.ops.dashboard` | `kanboost[dashboard]` | Interactive Streamlit app |
| `kanboost.registry.mlhub` | `kanboost[mlhub]` | Push/pull a model to a MinIO-backed object store |
| `kanboost.registry.local` | none | `LocalRegistry` — a local, versioned model registry |
| `kanboost.ops.mlflow_utils` | `kanboost[mlflow]` | Log a training run to MLflow |
| `kanboost.train.imbalance` | none | `balanced_weights()` / `find_threshold()` for imbalanced targets |
| `kanboost.train.accel` | none | `fast_fit()` — warm-started, ~3x faster opt-in training |
| `kanboost.interpret.interactions` | `scikit-learn`, `scipy` (both core dependencies) | `friedman_h()` / `check_additive_sufficiency()` — verify whether `gam=True`'s additive assumption actually holds for your data |
| `kanboost.pipeline` | none | `KANBoostPipeline` — train -> optional calibrate -> optional symbolic export as one call |
| `kanboost.core.config` | none | `KANBoostConfig`/`KANConfig`/`BoostConfig` — typed, grouped hyperparameters |

See the [Guide](guide/interpretability.md) for each module's full API
with examples.

## `kantun` (separate package)

`kantun.KantunSearch(model_cls, param_distributions, ...)` — see
[Tuning with kantun](guide/tuning-with-kantun.md).
