# Migrating to KANBoost v1.0.0

v1.0.0 is a deliberate breaking release: a full package restructure
into `core`/`interpret`/`train`/`ops`/`registry` subpackages (an "ML
System Design" layout: typed config, a pipeline orchestrator, a local
model registry), plus new features on top. This is not a drop-in
upgrade for code that imports anything beyond the top level.

## What's unchanged

```python
from kanboost import KANBoostClassifier, KANBoostRegressor
```

This top-level import — the one most code actually uses — is exactly
the same as every prior version. If that's the only thing you import
from `kanboost`, **you have nothing to change**.

`kantun`'s `KantunSearch` (the separate hyperparameter-tuning package)
also needs no changes: it only ever imports `KANBoostClassifier` at the
top level and works entirely through the standard flat constructor
kwargs / `get_params()` / `set_params()`, all of which are unchanged.

## What breaks: submodule import paths

Every module beyond the top-level re-export moved into a subpackage.
Update any import that goes deeper than `from kanboost import ...`:

| Old (< v1.0.0) | New (>= v1.0.0) |
|---|---|
| `kanboost.symbolic` | `kanboost.interpret.symbolic` |
| `kanboost.editing` | `kanboost.interpret.editing` |
| `kanboost.interactions` | `kanboost.interpret.interactions` |
| `kanboost.experimental` | `kanboost.interpret.experimental` |
| `kanboost.accel` | `kanboost.train.accel` |
| `kanboost.calibration` | `kanboost.train.calibration` |
| `kanboost.imbalance` | `kanboost.train.imbalance` |
| `kanboost.metrics` | `kanboost.train.metrics` |
| `kanboost.serving` | `kanboost.ops.serving` |
| `kanboost.dashboard` | `kanboost.ops.dashboard` |
| `kanboost.observability` | `kanboost.ops.observability` |
| `kanboost.logging_utils` | `kanboost.ops.logging_utils` |
| `kanboost.mlflow_utils` | `kanboost.ops.mlflow_utils` |
| `kanboost.mlhub` | `kanboost.registry.mlhub` |
| `kanboost._base`, `kanboost.classifier`, `kanboost.regressor`, `kanboost.losses`, `kanboost.encoders` | `kanboost.core.base`/`.classifier`/`.regressor`/`.losses`/`.encoders` |

Each subpackage also re-exports its own public API one level flat, so
you can often shorten the import instead of just relocating it:

```python
# both work; the second is shorter
from kanboost.train.accel import fast_fit
from kanboost.train import fast_fit
```

`uvicorn`/CLI invocations also move:

```bash
# before
uvicorn kanboost.serving:app
python -m kanboost.dashboard model.pt

# after
uvicorn kanboost.ops.serving:app
python -m kanboost.ops.dashboard model.pt
```

## What breaks: nothing, for `model.save()`/`.load()`

This *looked* like it would break — `save()` pickles
`self.preprocessor_` by its module path, and that path changed — but
`load()` transparently handles a file saved with any older kanboost
version: it detects the resulting `ModuleNotFoundError` and retries
once with the old flat module paths aliased to their new locations.
**You do not need to re-save or convert existing model files.** New
saves use `format_version: 2`; old (`format_version: 1`) files keep
loading exactly as before.

## What's new (not a migration, additive)

- **`kanboost.config`** — `KANBoostConfig`/`KANConfig`/`BoostConfig`,
  typed dataclasses grouping the ~19 constructor kwargs by concern.
  The flat kwarg constructor (`KANBoostClassifier(n_estimators=..., kan_hidden=..., ...)`)
  is unaffected and still the primary way to build a model; config
  objects are an alternative, not a replacement:
  ```python
  from kanboost import KANBoostConfig
  cfg = KANBoostConfig.from_flat(n_estimators=100, kan_hidden=1, gam=True)
  cfg.to_flat()  # back to the flat kwargs, e.g. to build the estimator
  ```
- **`kanboost.pipeline.KANBoostPipeline`** — sequences
  train -> optional calibrate -> optional symbolic export as one call:
  ```python
  from kanboost import KANBoostPipeline, KANBoostConfig
  cfg = KANBoostConfig.from_flat(n_estimators=100, kan_hidden=1, gam=True, random_state=0)
  result = KANBoostPipeline(cfg, task="classification", fast=True, calibrate=True).fit(
      X_train, y_train, X_cal=X_cal, y_cal=y_cal,
  )
  result.model            # the fitted KANBoostClassifier
  result.calibrated_model # the calibrated wrapper, if calibrate=True
  ```
- **`kanboost.registry.LocalRegistry`** — a local, versioned model
  registry on top of `save()`/`load()`:
  ```python
  from kanboost.registry import LocalRegistry
  reg = LocalRegistry("./models")
  version = reg.register(model, "churn", tags={"stage": "prod"})
  reg.get("churn")  # loads the latest version
  reg.push("churn", bucket="prod-models")  # delegates to kanboost.registry.mlhub
  ```
