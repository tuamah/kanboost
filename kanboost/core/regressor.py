"""
KANBoostRegressor: regression via gradient boosting with shallow KAN
learners.

`objective="squared_error"` (default) minimizes (optionally weighted)
squared error. `objective="quantile"` minimizes pinball loss at the
given `alpha`, producing a conditional quantile estimate instead of a
conditional mean.
"""

from __future__ import annotations

import numpy as np
import torch

from sklearn.base import RegressorMixin

from .base import _BaseKANBoost, _validate_Xy, _validate_sample_weight
from .losses import SquaredLoss, QuantileLoss


class KANBoostRegressor(RegressorMixin, _BaseKANBoost):
    """Gradient-boosted KAN ensemble for regression.

    All parameters are identical to KANBoostClassifier (see its
    docstring) except:

    objective : {"squared_error", "quantile"}, default="squared_error"
        "squared_error" fits the conditional mean; "quantile" fits the
        conditional `alpha`-quantile (pinball loss).
    alpha : float, default=0.5
        Target quantile when `objective="quantile"`; ignored otherwise.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        kan_hidden: int = 3,
        kan_grid: int = 2,
        kan_k: int = 3,
        kan_steps: int = 20,
        kan_lr: float = 0.02,
        early_stopping_rounds: int | None = 10,
        validation_fraction: float | None = None,
        categorical_cols=None,
        random_state: int = 42,
        verbose: bool = False,
        device: str | None = None,
        batch_size: int | None = None,
        gam: bool = False,
        monotone_constraints: dict | None = None,
        lamb: float = 0.0,
        lamb_l1: float = 1.0,
        lamb_coefdiff: float = 0.0,
        objective: str = "squared_error",
        alpha: float = 0.5,
    ):
        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            kan_hidden=kan_hidden,
            kan_grid=kan_grid,
            kan_k=kan_k,
            kan_steps=kan_steps,
            kan_lr=kan_lr,
            early_stopping_rounds=early_stopping_rounds,
            validation_fraction=validation_fraction,
            categorical_cols=categorical_cols,
            random_state=random_state,
            verbose=verbose,
            device=device,
            batch_size=batch_size,
            gam=gam,
            monotone_constraints=monotone_constraints,
            lamb=lamb,
            lamb_l1=lamb_l1,
            lamb_coefdiff=lamb_coefdiff,
        )
        self.objective = objective
        self.alpha = alpha

    def _make_loss(self):
        if self.objective == "squared_error":
            return SquaredLoss()
        if self.objective == "quantile":
            if not (0 < self.alpha < 1):
                raise ValueError("alpha must be in (0, 1) for objective='quantile'")
            return QuantileLoss(self.alpha)
        raise ValueError(
            f"Unknown objective {self.objective!r}; expected "
            f"'squared_error' or 'quantile'"
        )

    def fit(self, X, y, eval_set: tuple | None = None, sample_weight=None):
        """Fit the boosted ensemble.

        Parameters
        ----------
        X : DataFrame or array of shape (n_samples, n_features)
        y : array of shape (n_samples,), continuous target
        eval_set : (X_val, y_val) tuple, optional
            Validation data for early stopping.
        sample_weight : array of shape (n_samples,), optional
            Per-sample weights used when fitting each weak learner and
            when computing the initial prediction. Not applied to
            categorical target-mean encoding or to eval_set's loss.
        """
        if (eval_set is None and self.validation_fraction is not None
                and self.early_stopping_rounds is not None):
            X, X_eval, y, y_eval, sample_weight = self._internal_split(
                X, y, sample_weight, stratify=False
            )
            eval_set = (X_eval, y_eval)

        X, y, X_arr = self._prepare_fit(X, y)
        sample_weight = _validate_sample_weight(sample_weight, y)
        loss = self._make_loss()

        X_t = torch.tensor(X_arr, dtype=torch.float32, device=self.device_)
        n_features = X_arr.shape[1]

        X_val_t = y_val = None
        if eval_set is not None:
            X_val_df, y_val = eval_set
            X_val_df, y_val = _validate_Xy(X_val_df, y_val)
            X_val_arr = self.preprocessor_.transform(X_val_df)
            X_val_t = torch.tensor(X_val_arr, dtype=torch.float32, device=self.device_)

        self.learners_, self.init_pred_, self.best_iteration_ = self._boost_chain(
            X_t, y, loss, n_features, X_val_t, y_val, sample_weight, seed_base=0,
        )
        return self

    def predict(self, X) -> np.ndarray:
        """Return continuous predictions of shape (n_samples,)."""
        X_t = self._transform_X(X)
        return self._raw_score_chain(X_t, self.learners_, self.init_pred_, self.best_iteration_)

    def evaluate(self, X, y, verbose: bool = True) -> dict:
        """Predict on X and report MSE, RMSE, MAE, and R^2 against y
        (plus mean pinball loss when `objective="quantile"`)."""
        from sklearn.metrics import (
            mean_squared_error, mean_absolute_error, r2_score,
        )

        y = np.asarray(y, dtype=float).ravel()
        y_pred = self.predict(X)
        mse = float(mean_squared_error(y, y_pred))
        report = {
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(y, y_pred)),
            "r2": float(r2_score(y, y_pred)),
        }
        if self.objective == "quantile":
            diff = y - y_pred
            report["pinball"] = float(np.mean(
                np.maximum(self.alpha * diff, (self.alpha - 1) * diff)
            ))
        if verbose:
            for k, v in report.items():
                print(f"{k.upper():5s}: {v:.5f}")
        return report
