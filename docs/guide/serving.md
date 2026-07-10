# Serving & observability

These live in their own modules and never modify or depend on private
training/inference internals beyond `model.verbose`/`model._fit_learner`
existing — nothing here changes how `fit`/`predict` behave.

## Observability

`kanboost.ops.observability` — no extra install needed:

```python
from kanboost.ops.observability import (
    time_predict, memory_snapshot, gpu_utilization_flag, capture_boosting_rounds,
)

preds, metrics = time_predict(model, X_val, method="predict_proba")
print(metrics.elapsed_seconds, metrics.samples_per_second, metrics.device)

print(memory_snapshot())        # process RSS + CUDA allocator stats
print(gpu_utilization_flag(model))  # cuda_available, device_name, model_on_gpu

with capture_boosting_rounds(model) as rounds:
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
for r in rounds:
    print(r.round, r.elapsed_seconds, r.loss, r.gpu_allocated_mb)
```

## Logging

`kanboost.ops.logging_utils` — stdlib only:

```python
from kanboost.ops.logging_utils import get_logger, log_boosting_rounds

logger = get_logger("my_experiment")  # respects KANBOOST_LOG_LEVEL env var
log_boosting_rounds(rounds, logger=logger, model_name="churn_v3")
```

## Serving

`kanboost.ops.serving` — needs `pip install kanboost[api]`:

```python
from kanboost.ops.serving import create_app

app = create_app("model.pt")  # auto-detects classifier vs. regressor
# uvicorn.run(app, host="0.0.0.0", port=8000)
```

or as a uvicorn target directly:

```bash
KANBOOST_MODEL_PATH=model.pt uvicorn kanboost.ops.serving:app
```

Endpoints: `GET /health`, `POST /predict` (`{"records": [{"col": val, ...}]}`),
and `POST /predict_proba` (classifiers only).

This is a service you'd run for other programs to call. If you want a
local, human-facing tool for exploring a model instead, see the
[interactive dashboard](editing-dashboard.md).
