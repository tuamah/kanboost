"""
kanboost.mlflow_utils -- optional MLflow experiment tracking for
KANBoost training runs.

Uses the standard `mlflow` Python client pointed directly at a tracking
server (e.g. `MLFLOW_TRACKING_URI=https://mlflow.mlhub.dev/`), not a
platform's own read-only REST wrapper around MLflow (a MEAP/MLHub-style
platform's `/api/mlflow/*` endpoints, for example, only expose
GET/DELETE on experiments/runs/models -- there's no way to create a run
through them, so real logging has to go through MLflow's own client
talking to the tracking server directly).

Additive: `mlflow` is only imported inside `log_training_run`, so
importing kanboost (or even this module) never requires it.
"""

from __future__ import annotations

import os


def log_training_run(
    model,
    X_test,
    y_test,
    run_name: str | None = None,
    tracking_uri: str | None = None,
    experiment_name: str = "kanboost",
    extra_metrics: dict | None = None,
    extra_params: dict | None = None,
    save_model_path: str | None = None,
) -> str:
    """Log a fitted KANBoost model's hyperparameters, evaluation
    metrics, and (optionally) its saved model file as one MLflow run.

    `tracking_uri` defaults to the `MLFLOW_TRACKING_URI` environment
    variable if not given. Hyperparameters come from `model.get_params()`
    (KANBoost estimators are scikit-learn `BaseEstimator`s already).
    Metrics come from `model.evaluate(X_test, y_test, verbose=False)` --
    the same dict `KANBoostClassifier`/`KANBoostRegressor` already
    compute and print, just logged instead of printed. `extra_metrics`/
    `extra_params` are merged in on top (e.g. Brier score/log-loss from
    `kanboost.calibration`, or a dataset name).

    If `save_model_path` is given, `model.save(save_model_path)` is
    called and the resulting file is logged as an MLflow artifact.

    Returns the MLflow run ID.
    """
    import mlflow

    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)

    metrics = model.evaluate(X_test, y_test, verbose=False)
    if extra_metrics:
        metrics.update(extra_metrics)

    params = model.get_params()
    if extra_params:
        params.update(extra_params)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({k: v for k, v in params.items() if v is not None})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})

        if save_model_path:
            model.save(save_model_path)
            mlflow.log_artifact(save_model_path)

        return run.info.run_id
