"""
kanboost.pipeline -- orchestrates train -> (optional) calibrate ->
(optional) symbolic export as one call.

A thin coordinator, not a new abstraction hierarchy: every stage below
is the *existing*, unchanged function from kanboost.train/kanboost.interpret
(fast_fit, calibrate, export_symbolic) -- this class just sequences them
and carries the results forward, so a caller doesn't have to wire the
plumbing (which model to calibrate, which model to export) by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .core.config import KANBoostConfig
from .core.classifier import KANBoostClassifier
from .core.regressor import KANBoostRegressor


@dataclass
class PipelineResult:
    model: Any
    calibrated_model: Any = None
    symbolic_model: Any = None
    metrics: dict = field(default_factory=dict)


class KANBoostPipeline:
    def __init__(self, config: KANBoostConfig, task: str = "classification", fast: bool = False,
                 calibrate: bool = False, calibration_method: str = "platt", export_symbolic: bool = False):
        if task not in ("classification", "regression"):
            raise ValueError(f"task must be 'classification' or 'regression'; got {task!r}")
        self.config = config
        self.task = task
        self.fast = fast
        self.do_calibrate = calibrate
        self.calibration_method = calibration_method
        self.do_export_symbolic = export_symbolic
        self.result_: PipelineResult | None = None

    def _build_model(self):
        flat = self.config.to_flat()
        if self.task == "classification":
            # objective/alpha are regressor-only kwargs KANBoostConfig
            # always carries (for round-tripping a regressor's config);
            # KANBoostClassifier.__init__ doesn't accept them at all.
            flat.pop("objective", None)
            flat.pop("alpha", None)
            return KANBoostClassifier(**flat)
        return KANBoostRegressor(**flat)

    def fit(self, X_train, y_train, X_cal=None, y_cal=None, eval_set=None, sample_weight=None) -> PipelineResult:
        model = self._build_model()
        if self.fast:
            from .train.accel import fast_fit
            fast_fit(model, X_train, y_train, eval_set=eval_set, sample_weight=sample_weight)
        else:
            model.fit(X_train, y_train, eval_set=eval_set, sample_weight=sample_weight)

        calibrated_model = None
        if self.do_calibrate:
            if X_cal is None or y_cal is None:
                raise ValueError("calibrate=True requires X_cal and y_cal")
            from .train.calibration import calibrate as calibrate_fn
            calibrated_model = calibrate_fn(model, X_cal, y_cal, method=self.calibration_method)

        symbolic_model = None
        if self.do_export_symbolic:
            if not self.config.gam:
                raise ValueError("export_symbolic=True requires the config's gam=True")
            from .interpret.symbolic import export_symbolic
            symbolic_model = export_symbolic(model)

        self.result_ = PipelineResult(model=model, calibrated_model=calibrated_model, symbolic_model=symbolic_model)
        return self.result_
