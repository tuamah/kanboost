# MLOps integration

Two small, independent modules for pushing a trained model or its
metrics somewhere other than the local filesystem. Neither is required
by, or imports, the other.

## `kanboost.mlhub`

Push/pull a saved model (`model.save(path)`'s output) to/from a
MinIO-backed object store behind a FastAPI gateway (the shape this
targets: `POST /api/minio/buckets/{bucket}/upload`,
`GET .../download/{key}`, `GET .../objects`).

Verified end-to-end against a live server: authenticate via
`X-API-Key` (not `Authorization: Bearer`, which is a different scheme
reserved for session tokens from a login endpoint -- using Bearer with
an API key 401s with "Invalid or expired token" rather than "not
authenticated", which is what revealed it was the wrong scheme, not a
bad key). Bucket creation's request body field is `name`, confirmed via
a 422 response that spelled out the exact mismatch after an initial
guess (`bucket`) was wrong. A model pushed, pulled back under a new
name, and reloaded produced byte-identical predictions to the original.

```python
from kanboost.registry.mlhub import push_model, pull_model, list_models, ensure_bucket

model.save("model.pt")
ensure_bucket("kanboost-models", api_key="...")   # or set MLHUB_API_KEY
push_model("model.pt", "kanboost-models", api_key="...")

pull_model("kanboost-models", "model.pt", "downloaded.pt", api_key="...")
list_models("kanboost-models", api_key="...")
```

Requires `pip install kanboost[mlhub]` (`requests`). `base_url`
defaults to `https://mlhub.dev`; override via `base_url=` or
`MLHUB_BASE_URL` for a different deployment. If your platform's
endpoint shapes differ (field names, response format), the module's
own docstring in `kanboost/mlhub.py` explains what to check and adjust
-- this was itself discovered by iterating against a real 401 then a
real 422, not guessed once and assumed correct.

## `kanboost.mlflow_utils`

Log a training run's hyperparameters (`model.get_params()`, since
KANBoost estimators are already scikit-learn `BaseEstimator`s),
evaluation metrics (`model.evaluate()`), and optionally the saved model
file, to an MLflow tracking server -- via the standard `mlflow` client,
not a platform's own read-only REST wrapper. Many self-hosted platforms
only expose `GET`/`DELETE` on MLflow experiments/runs/models through
their own API layer, with no way to *create* a run that way; real
logging has to go through MLflow's own client talking to the tracking
server directly.

```python
from kanboost.ops.mlflow_utils import log_training_run

run_id = log_training_run(
    model, X_test, y_test,
    tracking_uri="https://mlflow.your-host.example/",  # or MLFLOW_TRACKING_URI
    experiment_name="kanboost",
    extra_metrics={"brier": 0.03},  # e.g. from kanboost.train.calibration
    extra_params={"dataset": "breast_cancer"},
    save_model_path="model.pt",
)
```

Requires `pip install kanboost[mlflow]`. Verified end-to-end against a
local sqlite-backed tracking store (params/metrics/artifact all logged
and read back correctly, for both a classifier and a regressor).
Talking to a specific remote tracking server may need its own auth
(`MLFLOW_TRACKING_USERNAME`/`MLFLOW_TRACKING_PASSWORD`, or whatever
your deployment requires) layered on top of `tracking_uri`.
