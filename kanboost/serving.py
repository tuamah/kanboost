"""
kanboost.serving -- optional FastAPI wrapper around a saved KANBoost model.

Additive: FastAPI/pydantic are only imported inside `create_app`, so
importing kanboost (or even this module, as long as you don't call
`create_app`/set `KANBOOST_MODEL_PATH`) never requires them. Install
with `pip install kanboost[api]`.

Usage
-----
Programmatic:
    from kanboost.serving import create_app
    app = create_app("model.pt")
    # then run with any ASGI server, e.g. uvicorn.run(app, ...)

As a uvicorn target (reads the model path from an env var so the module
is importable without arguments):
    KANBOOST_MODEL_PATH=model.pt uvicorn kanboost.serving:app
"""

import os
from typing import Any

# Deliberately no `from __future__ import annotations` in this file:
# FastAPI/pydantic need to resolve `PredictRequest`'s annotation to the
# real class object, and PEP 563's string annotations can't be resolved
# for a name that's only ever bound inside a function's local scope
# (which is exactly what would happen if this model were defined inside
# create_app() instead of at module level, as it is below).

import numpy as np
import pandas as pd
import torch

from .observability import gpu_utilization_flag, time_predict

try:
    from pydantic import BaseModel

    class PredictRequest(BaseModel):
        records: list[dict[str, Any]]
except ImportError:
    PredictRequest = None  # only used inside create_app, which raises its own clear error


def _load_any(path: str, device: str | None = None):
    """Load a saved model, auto-detecting KANBoostClassifier vs.
    KANBoostRegressor from the file's own `class_name` metadata."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    class_name = payload.get("class_name")
    if class_name == "KANBoostClassifier":
        from .classifier import KANBoostClassifier
        return KANBoostClassifier.load(path, device=device)
    if class_name == "KANBoostRegressor":
        from .regressor import KANBoostRegressor
        return KANBoostRegressor.load(path, device=device)
    raise ValueError(
        f"{path!r} has unrecognized class_name {class_name!r}; expected "
        "'KANBoostClassifier' or 'KANBoostRegressor'."
    )


def create_app(model_path: str, device: str | None = None):
    """Build a FastAPI app serving one fitted model loaded from `model_path`.

    Endpoints:
      GET  /health         -- status, model class, device/GPU info
      POST /predict         -- {"records": [{"col": val, ...}, ...]} -> predictions
      POST /predict_proba   -- classifiers only; same input -> class probabilities
    """
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:
        raise ImportError(
            "kanboost.serving requires fastapi and pydantic; "
            "install with `pip install kanboost[api]`."
        ) from exc
    if PredictRequest is None:
        raise ImportError(
            "kanboost.serving requires pydantic; "
            "install with `pip install kanboost[api]`."
        )

    model = _load_any(model_path, device=device)
    is_classifier = hasattr(model, "predict_proba")

    app = FastAPI(title="KANBoost model server")

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model_class": type(model).__name__,
            **gpu_utilization_flag(model),
        }

    @app.post("/predict")
    def predict(req: PredictRequest):
        if not req.records:
            raise HTTPException(status_code=400, detail="records must be non-empty")
        X = pd.DataFrame(req.records)
        try:
            preds, metrics = time_predict(model, X, method="predict")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "predictions": np.asarray(preds).tolist(),
            "elapsed_seconds": metrics.elapsed_seconds,
        }

    if is_classifier:
        @app.post("/predict_proba")
        def predict_proba(req: PredictRequest):
            if not req.records:
                raise HTTPException(status_code=400, detail="records must be non-empty")
            X = pd.DataFrame(req.records)
            try:
                proba, metrics = time_predict(model, X, method="predict_proba")
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return {
                "probabilities": np.asarray(proba).tolist(),
                "classes": np.asarray(model.classes_).tolist(),
                "elapsed_seconds": metrics.elapsed_seconds,
            }

    return app


# Module-level `app`, only built if KANBOOST_MODEL_PATH is set at import
# time -- so `import kanboost.serving` alone never requires a model to
# exist or fastapi to be installed; only running it as a uvicorn target does.
app = None
if os.environ.get("KANBOOST_MODEL_PATH"):
    app = create_app(
        os.environ["KANBOOST_MODEL_PATH"],
        device=os.environ.get("KANBOOST_DEVICE"),
    )
