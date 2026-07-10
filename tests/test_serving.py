"""
Tests for kanboost.serving. Skipped entirely if fastapi/httpx aren't
installed, since the API is an optional extra (`pip install kanboost[api]`).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from kanboost import KANBoostClassifier, KANBoostRegressor

fastapi_testclient = pytest.importorskip(
    "fastapi.testclient", reason="fastapi/httpx not installed -- kanboost[api] is optional"
)
from kanboost.ops.serving import create_app  # noqa: E402


@pytest.fixture
def saved_classifier_path(tmp_path):
    X, y = make_classification(n_samples=100, n_features=4, random_state=0)
    X_df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    model = KANBoostClassifier(n_estimators=3, kan_steps=3, early_stopping_rounds=None, random_state=0)
    model.fit(X_df, y)
    path = str(tmp_path / "clf.pt")
    model.save(path)
    return path


@pytest.fixture
def saved_regressor_path(tmp_path):
    X, y = make_regression(n_samples=100, n_features=3, random_state=0)
    X_df = pd.DataFrame(X, columns=["x", "y", "z"])
    model = KANBoostRegressor(n_estimators=3, kan_steps=3, early_stopping_rounds=None, random_state=0)
    model.fit(X_df, y)
    path = str(tmp_path / "reg.pt")
    model.save(path)
    return path


def test_classifier_endpoints(saved_classifier_path):
    app = create_app(saved_classifier_path)
    client = fastapi_testclient.TestClient(app)

    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_class"] == "KANBoostClassifier"
    assert "cuda_available" in body

    records = [{"a": 0.1, "b": -0.2, "c": 0.3, "d": 0.0}] * 3
    r = client.post("/predict", json={"records": records})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 3

    r = client.post("/predict_proba", json={"records": records})
    assert r.status_code == 200
    body = r.json()
    assert len(body["probabilities"]) == 3
    assert len(body["probabilities"][0]) == 2  # binary
    assert body["classes"] == [0.0, 1.0]

    r = client.post("/predict", json={"records": []})
    assert r.status_code == 400


def test_regressor_endpoints(saved_regressor_path):
    app = create_app(saved_regressor_path)
    client = fastapi_testclient.TestClient(app)

    records = [{"x": 0.1, "y": 0.2, "z": 0.3}]
    r = client.post("/predict", json={"records": records})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 1

    # regressor has no predict_proba, so the route must not exist
    r = client.post("/predict_proba", json={"records": records})
    assert r.status_code == 404


def test_predict_rejects_unknown_columns(saved_classifier_path):
    app = create_app(saved_classifier_path)
    client = fastapi_testclient.TestClient(app)
    r = client.post("/predict", json={"records": [{"totally_unknown_col": 1}]})
    assert r.status_code == 400
