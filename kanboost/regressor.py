"""
KANBoostRegressor: regression via gradient boosting with shallow KAN
learners, minimizing squared error (residual = y - F(x) directly).
"""

from __future__ import annotations

import numpy as np
import torch

from sklearn.base import RegressorMixin

from ._base import _BaseKANBoost, _validate_Xy


class KANBoostRegressor(RegressorMixin, _BaseKANBoost):
    """Gradient-boosted KAN ensemble for regression (squared-error loss).

    Parameters are identical to KANBoostClassifier; see its docstring.
    """


    def fit(self, X, y, eval_set: tuple | None = None):
        """Fit the boosted ensemble.

        Parameters
        ----------
        X : DataFrame or array of shape (n_samples, n_features)
        y : array of shape (n_samples,), continuous target
        eval_set : (X_val, y_val) tuple, optional
            Validation data for early stopping on MSE.
        """
        X, y, X_arr = self._prepare_fit(X, y)

        X_t = torch.tensor(X_arr, dtype=torch.float32)
        n_features = X_arr.shape[1]

        self.init_pred_ = float(y.mean())
        F = np.full(len(y), self.init_pred_)

        X_val_t = y_val = F_val = None
        if eval_set is not None:
            X_val_df, y_val = eval_set
            X_val_df, y_val = _validate_Xy(X_val_df, y_val)
            X_val_arr = self.preprocessor_.transform(X_val_df)
            X_val_t = torch.tensor(X_val_arr, dtype=torch.float32)
            F_val = np.full(len(y_val), self.init_pred_)

        best_val_loss = np.inf
        rounds_since_best = 0
        self.learners_ = []
        self.best_iteration_ = None

        for t in range(self.n_estimators):
            residual = y - F

            learner = self._new_learner(n_features, seed_offset=t)
            update = self._fit_learner(learner, X_t, residual)
            F += self.learning_rate * update
            self.learners_.append(learner)

            if X_val_t is not None:
                with torch.no_grad():
                    F_val += self.learning_rate * learner(X_val_t).numpy().flatten()
                val_loss = float(np.mean((y_val - F_val) ** 2))

                if self.verbose:
                    print(f"[{t + 1}/{self.n_estimators}] val_mse={val_loss:.5f}")

                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    rounds_since_best = 0
                    self.best_iteration_ = t + 1
                else:
                    rounds_since_best += 1
                    if (self.early_stopping_rounds is not None
                            and rounds_since_best >= self.early_stopping_rounds):
                        if self.verbose:
                            print(f"Early stopping at iteration {t + 1}")
                        break

        if self.best_iteration_ is None:
            self.best_iteration_ = len(self.learners_)
        return self

    def predict(self, X) -> np.ndarray:
        """Return continuous predictions of shape (n_samples,)."""
        X_t = self._transform_X(X)
        F = np.full(X_t.shape[0], self.init_pred_)
        for learner in self.learners_[: self.best_iteration_]:
            with torch.no_grad():
                F += self.learning_rate * learner(X_t).numpy().flatten()
        return F

    def evaluate(self, X, y, verbose: bool = True) -> dict:
        """Predict on X and report MSE, RMSE, MAE, and R^2 against y."""
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
        if verbose:
            for k, v in report.items():
                print(f"{k.upper():5s}: {v:.5f}")
        return report
