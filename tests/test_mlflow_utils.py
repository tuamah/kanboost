"""
Tests for kanboost.mlflow_utils. Uses a local sqlite-backed MLflow
tracking store (no live server) so this is fully offline. Skipped
entirely if `mlflow` isn't installed, since it's an optional extra.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

mlflow = pytest.importorskip("mlflow", reason="mlflow not installed -- kanboost.mlflow_utils is optional")

from kanboost import KANBoostClassifier, KANBoostRegressor
from kanboost.ops.mlflow_utils import log_training_run


@pytest.fixture
def tracking_uri(tmp_path):
    db_path = str(tmp_path / "mlflow_test.db").replace("\\", "/")
    return f"sqlite:///{db_path}"


def test_log_training_run_classifier(tracking_uri, tmp_path):
    X, y = make_classification(n_samples=200, n_features=5, random_state=0)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    X_train, X_test = X_df.iloc[:150], X_df.iloc[150:]
    y_train, y_test = y[:150], y[150:]

    model = KANBoostClassifier(n_estimators=5, kan_steps=5, early_stopping_rounds=None, random_state=0)
    model.fit(X_train, y_train)

    save_path = str(tmp_path / "model.pt")
    run_id = log_training_run(
        model, X_test, y_test,
        run_name="test_run", tracking_uri=tracking_uri, experiment_name="kanboost_test_clf",
        extra_metrics={"brier": 0.05, "log_loss": 0.2},
        extra_params={"dataset": "synthetic"},
        save_model_path=save_path,
    )
    assert run_id

    mlflow.set_tracking_uri(tracking_uri)
    run = mlflow.get_run(run_id)

    # hyperparameters from get_params() (non-None ones) plus the extra
    assert run.data.params["n_estimators"] == "5"
    assert run.data.params["kan_steps"] == "5"
    assert run.data.params["dataset"] == "synthetic"

    # metrics from model.evaluate() plus the extras merged in
    assert "accuracy" in run.data.metrics
    assert "auc" in run.data.metrics or "roc_auc" in run.data.metrics
    assert run.data.metrics["brier"] == 0.05
    assert run.data.metrics["log_loss"] == 0.2

    # artifact (the saved model file) was logged
    artifacts = mlflow.artifacts.list_artifacts(run_id=run_id)
    assert any(a.path == "model.pt" for a in artifacts)
    assert os.path.exists(save_path)  # save_model_path itself was also written locally


def test_log_training_run_regressor(tracking_uri):
    X, y = make_regression(n_samples=150, n_features=4, noise=0.1, random_state=0)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    X_train, X_test = X_df.iloc[:100], X_df.iloc[100:]
    y_train, y_test = y[:100], y[100:]

    model = KANBoostRegressor(n_estimators=5, kan_steps=5, early_stopping_rounds=None, random_state=0)
    model.fit(X_train, y_train)

    run_id = log_training_run(
        model, X_test, y_test,
        tracking_uri=tracking_uri, experiment_name="kanboost_test_reg",
    )
    mlflow.set_tracking_uri(tracking_uri)
    run = mlflow.get_run(run_id)
    assert "mse" in run.data.metrics or "rmse" in run.data.metrics


def test_log_training_run_without_save_path_logs_no_artifact(tracking_uri):
    X, y = make_classification(n_samples=100, n_features=4, random_state=0)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    model = KANBoostClassifier(n_estimators=3, kan_steps=3, early_stopping_rounds=None, random_state=0)
    model.fit(X_df.iloc[:80], y[:80])

    run_id = log_training_run(
        model, X_df.iloc[80:], y[80:],
        tracking_uri=tracking_uri, experiment_name="kanboost_test_no_artifact",
    )
    mlflow.set_tracking_uri(tracking_uri)
    artifacts = mlflow.artifacts.list_artifacts(run_id=run_id)
    assert artifacts == []


if __name__ == "__main__":
    print("Run via pytest -- uses tmp_path fixtures.")
