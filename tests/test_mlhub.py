"""
Tests for kanboost.mlhub. No live server involved -- requests.post/get
are mocked. Skipped entirely if `requests` isn't installed.
"""
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

pytest.importorskip("requests", reason="requests not installed -- kanboost.mlhub is optional")

from kanboost.mlhub import push_model, pull_model, list_models, ensure_bucket


def test_resolve_api_key_requires_a_key(tmp_path, monkeypatch):
    monkeypatch.delenv("MLHUB_API_KEY", raising=False)
    model_file = tmp_path / "model.pt"
    model_file.write_bytes(b"x")  # file must exist so the ValueError (not FileNotFoundError) is what's tested
    with pytest.raises(ValueError):
        push_model(str(model_file), "bucket", api_key=None)


def test_push_model_sends_bearer_and_multipart(tmp_path):
    model_file = tmp_path / "model.pt"
    model_file.write_bytes(b"fake-model-bytes")

    mock_response = MagicMock()
    mock_response.json.return_value = {"key": "model.pt", "status": "ok"}
    mock_response.raise_for_status.return_value = None

    with patch("requests.post", return_value=mock_response) as mock_post:
        result = push_model(str(model_file), "kanboost-models", api_key="test-key")

    assert result == {"key": "model.pt", "status": "ok"}
    args, kwargs = mock_post.call_args
    assert args[0] == "https://mlhub.dev/api/minio/buckets/kanboost-models/upload"
    assert kwargs["headers"] == {"X-API-Key": "test-key"}
    assert "file" in kwargs["files"]


def test_push_model_uses_env_var_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("MLHUB_API_KEY", "env-key")
    model_file = tmp_path / "model.pt"
    model_file.write_bytes(b"x")

    mock_response = MagicMock()
    mock_response.json.return_value = {}
    mock_response.raise_for_status.return_value = None

    with patch("requests.post", return_value=mock_response) as mock_post:
        push_model(str(model_file), "bucket")

    _, kwargs = mock_post.call_args
    assert kwargs["headers"] == {"X-API-Key": "env-key"}


def test_pull_model_writes_streamed_content(tmp_path):
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]

    dest = str(tmp_path / "downloaded.pt")
    with patch("requests.get", return_value=mock_response) as mock_get:
        result = pull_model("bucket", "model.pt", dest, api_key="k")

    assert result == dest
    with open(dest, "rb") as f:
        assert f.read() == b"chunk1chunk2"
    args, kwargs = mock_get.call_args
    assert args[0] == "https://mlhub.dev/api/minio/buckets/bucket/download/model.pt"


def test_list_models_returns_json():
    mock_response = MagicMock()
    mock_response.json.return_value = [{"key": "a.pt"}, {"key": "b.pt"}]
    mock_response.raise_for_status.return_value = None

    with patch("requests.get", return_value=mock_response):
        result = list_models("bucket", api_key="k")

    assert result == [{"key": "a.pt"}, {"key": "b.pt"}]


def test_ensure_bucket_sends_name_field():
    """Regression test: the server's create-bucket endpoint expects a
    `name` field in the request body, not `bucket` -- confirmed via a
    live 422 response (`"loc":["body","name"],"msg":"Field required"`)
    after the original guess (`bucket`) was wrong."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    with patch("requests.post", return_value=mock_response) as mock_post:
        ensure_bucket("kanboost-models", api_key="k")

    _, kwargs = mock_post.call_args
    assert kwargs["json"] == {"name": "kanboost-models"}


def test_ensure_bucket_ignores_conflict():
    mock_response = MagicMock()
    mock_response.status_code = 409
    with patch("requests.post", return_value=mock_response):
        ensure_bucket("bucket", api_key="k")  # must not raise


def test_ensure_bucket_raises_on_real_error():
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.raise_for_status.side_effect = Exception("server error")
    with patch("requests.post", return_value=mock_response):
        with pytest.raises(Exception):
            ensure_bucket("bucket", api_key="k")


def test_custom_base_url_respected(tmp_path):
    model_file = tmp_path / "model.pt"
    model_file.write_bytes(b"x")
    mock_response = MagicMock()
    mock_response.json.return_value = {}
    mock_response.raise_for_status.return_value = None

    with patch("requests.post", return_value=mock_response) as mock_post:
        push_model(str(model_file), "bucket", api_key="k", base_url="https://custom.example.com/")

    args, _ = mock_post.call_args
    assert args[0] == "https://custom.example.com/api/minio/buckets/bucket/upload"
