"""
kanboost.mlhub -- optional integration with a MEAP/MLHub-style platform
(a self-hosted "Enterprise AI OS" exposing a MinIO-backed object store
behind a FastAPI gateway: `POST /api/minio/buckets/{bucket}/upload`,
`GET .../download/{key}`, `GET .../objects`, etc.).

Lets you push/pull a saved KANBoost model (`.pt`, from `model.save(path)`)
to/from the platform's object storage instead of keeping it only local --
useful for picking up training on a second machine, or serving from a
model that was trained elsewhere.

Additive: `requests` is only imported inside the functions here, so
importing kanboost (or even this module) never requires it.

Authentication is `X-API-Key: <key>` (confirmed against a live server --
API keys created via `/api/keys` don't authenticate as `Authorization:
Bearer`, which is a *different* scheme reserved for session tokens from
`/api/auth/login`; using Bearer with an API key 401s with "Invalid or
expired token" rather than "not authenticated", which is what gave this
away). The multipart upload field name (`file`) and response shape are
still unverified against a live upload call -- if `push_model` 404s or
422s, that's the most likely mismatch; compare against your platform's
actual `POST /api/minio/buckets/{bucket}/upload` behavior (e.g. via its
Swagger "Try it out") and adjust `_UPLOAD_FIELD_NAME` or `push_model`'s
response parsing accordingly.
"""

from __future__ import annotations

import os

_UPLOAD_FIELD_NAME = "file"  # adjust if your server expects a different multipart field name


def _headers(api_key: str) -> dict:
    # Confirmed against a live server: this platform's API keys (from
    # /api/keys) authenticate via X-API-Key, not `Authorization: Bearer`
    # -- Bearer is a *different* auth scheme reserved for session tokens
    # from /api/auth/login (it 401s with "Invalid or expired token" for
    # an API key, not "not authenticated", confirming it's a recognized
    # scheme expecting a different kind of token).
    return {"X-API-Key": api_key}


def _base_url(base_url: str | None) -> str:
    return (base_url or os.environ.get("MLHUB_BASE_URL") or "https://mlhub.dev").rstrip("/")


def _resolve_api_key(api_key: str | None) -> str:
    key = api_key or os.environ.get("MLHUB_API_KEY")
    if not key:
        raise ValueError(
            "No API key given -- pass api_key=... or set the MLHUB_API_KEY "
            "environment variable. Never hardcode a key into source you commit."
        )
    return key


def push_model(path: str, bucket: str, key: str | None = None, api_key: str | None = None, base_url: str | None = None) -> dict:
    """Upload a saved model file (`model.save(path)`'s output, or any
    file) to `bucket` on the platform's object store.

    `key` is the object name within the bucket; defaults to the local
    file's basename. Returns the server's JSON response.
    """
    import requests

    key = key or os.path.basename(path)
    url = f"{_base_url(base_url)}/api/minio/buckets/{bucket}/upload"

    with open(path, "rb") as f:
        response = requests.post(
            url,
            headers=_headers(_resolve_api_key(api_key)),
            files={_UPLOAD_FIELD_NAME: (key, f)},
        )
    response.raise_for_status()
    return response.json()


def pull_model(bucket: str, key: str, dest: str, api_key: str | None = None, base_url: str | None = None) -> str:
    """Download object `key` from `bucket` to local path `dest`. Returns `dest`."""
    import requests

    url = f"{_base_url(base_url)}/api/minio/buckets/{bucket}/download/{key}"
    response = requests.get(url, headers=_headers(_resolve_api_key(api_key)), stream=True)
    response.raise_for_status()

    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    return dest


def list_models(bucket: str, api_key: str | None = None, base_url: str | None = None) -> list:
    """List objects in `bucket`."""
    import requests

    url = f"{_base_url(base_url)}/api/minio/buckets/{bucket}/objects"
    response = requests.get(url, headers=_headers(_resolve_api_key(api_key)))
    response.raise_for_status()
    return response.json()


def ensure_bucket(bucket: str, api_key: str | None = None, base_url: str | None = None) -> None:
    """Create `bucket` if it doesn't already exist (ignores a 409/"already
    exists"-style conflict; re-raises any other error)."""
    import requests

    url = f"{_base_url(base_url)}/api/minio/buckets"
    response = requests.post(
        url, headers=_headers(_resolve_api_key(api_key)), json={"bucket": bucket},
    )
    if response.status_code >= 400 and response.status_code != 409:
        response.raise_for_status()
