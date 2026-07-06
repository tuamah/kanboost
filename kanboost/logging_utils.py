"""
kanboost.logging_utils -- a thin, opt-in wrapper around the stdlib
`logging` module. Nothing in the training/inference core imports or
requires this; it exists for callers who want structured log lines
(e.g. piped to a file or a log aggregator) instead of `verbose=True`'s
raw `print()` output, and pairs naturally with
`kanboost.observability.capture_boosting_rounds` for per-round logging.
"""

from __future__ import annotations

import logging
import os

_DEFAULT_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"


def get_logger(name: str = "kanboost", level: str | int | None = None) -> logging.Logger:
    """Return a configured logger. Idempotent: calling this again with
    the same `name` returns the same logger without adding duplicate
    handlers.

    `level` defaults to the `KANBOOST_LOG_LEVEL` env var (e.g. "DEBUG",
    "INFO"), falling back to "INFO" if unset.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    resolved_level = level or os.environ.get("KANBOOST_LOG_LEVEL", "INFO")
    logger.setLevel(resolved_level)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def log_boosting_rounds(rounds, logger: logging.Logger | None = None, model_name: str = "model") -> None:
    """Log one line per `kanboost.observability.RoundMetric` (as produced
    by `capture_boosting_rounds`), e.g.:

        with capture_boosting_rounds(model) as rounds:
            model.fit(X, y, eval_set=(X_val, y_val))
        log_boosting_rounds(rounds, model_name="churn_classifier")
    """
    logger = logger or get_logger()
    for r in rounds:
        parts = [f"round={r.round}", f"elapsed={r.elapsed_seconds:.3f}s"]
        if r.loss is not None:
            parts.append(f"val_loss={r.loss:.5f}")
        if r.gpu_allocated_mb is not None:
            parts.append(f"gpu_mb={r.gpu_allocated_mb:.1f}")
        logger.info(f"[{model_name}] " + " ".join(parts))
