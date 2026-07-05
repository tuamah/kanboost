"""
kanboost._base -- shared machinery for KANBoostClassifier / KANBoostRegressor.

Contains:
- input validation
- the shared boosting loop skeleton
- sklearn-compatible get_params / set_params (so KANBoost estimators work
  inside sklearn Pipelines, GridSearchCV, and kantun without adapters)
- feature importances (shared by both estimators)
- suppression of pykan's noisy checkpoint logging
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from kan import KAN
from sklearn.base import BaseEstimator

from .encoders import TabularPreprocessor


@contextlib.contextmanager
def _suppress_pykan_noise():
    """pykan prints checkpoint messages to stdout and tqdm progress bars
    to stderr on every instantiation/fit; silence both."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        yield


def _validate_Xy(X, y):
    """Validate and normalize training inputs. Returns (X_df, y_arr)."""
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(np.asarray(X))
        X.columns = [f"f{i}" for i in range(X.shape[1])]

    y = np.asarray(y, dtype=float).ravel()

    if len(X) != len(y):
        raise ValueError(
            f"X and y have inconsistent lengths: {len(X)} vs {len(y)}"
        )
    if len(X) == 0:
        raise ValueError("X is empty.")
    if np.isnan(y).any():
        raise ValueError(
            "y contains NaN values. Remove or impute them before fitting."
        )
    return X, y


class _BaseKANBoost(BaseEstimator):
    """Shared base for the boosting estimators. Not part of the public API.

    Inherits sklearn's BaseEstimator, which provides get_params/set_params,
    __sklearn_tags__, and full compatibility with sklearn Pipelines,
    GridSearchCV, cross_val_score, and clone().
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
        categorical_cols=None,
        random_state: int = 42,
        verbose: bool = False,
        device: str | None = None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.kan_hidden = kan_hidden
        self.kan_grid = kan_grid
        self.kan_k = kan_k
        self.kan_steps = kan_steps
        self.kan_lr = kan_lr
        self.early_stopping_rounds = early_stopping_rounds
        self.categorical_cols = categorical_cols
        self.random_state = random_state
        self.verbose = verbose
        self.device = device

        # fitted state
        self.preprocessor_ = None
        self.learners_ = []
        self.init_pred_ = None
        self.best_iteration_ = None
        self.feature_names_in_ = None
        self.device_ = None

    # ------------------------------------------------------------------
    # NOTE: get_params/set_params/__sklearn_tags__ come from BaseEstimator.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # shared internals
    # ------------------------------------------------------------------
    def _validate_hyperparams(self):
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")
        if not (0 < self.learning_rate <= 1):
            raise ValueError("learning_rate must be in (0, 1]")
        if self.kan_hidden < 1 or self.kan_grid < 1 or self.kan_steps < 1:
            raise ValueError("kan_hidden, kan_grid, kan_steps must be >= 1")

    def _resolve_device(self) -> torch.device:
        """`device=None` auto-selects cuda when available, else cpu."""
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _prepare_fit(self, X, y):
        """Common preamble: seeds, validation, preprocessing."""
        self._validate_hyperparams()
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self.device_ = self._resolve_device()

        X, y = _validate_Xy(X, y)
        self.feature_names_in_ = list(X.columns)

        self.preprocessor_ = TabularPreprocessor(
            categorical_cols=self.categorical_cols or []
        )
        X_arr = self.preprocessor_.fit_transform(X, y)

        if np.isnan(X_arr).any():
            raise ValueError(
                "Preprocessed X contains NaN values. KANBoost does not yet "
                "handle missing values natively -- impute before fitting."
            )
        return X, y, X_arr

    def _new_learner(self, n_features: int, seed_offset: int) -> KAN:
        with _suppress_pykan_noise():
            return KAN(
                width=[n_features, self.kan_hidden, 1],
                grid=self.kan_grid,
                k=self.kan_k,
                seed=self.random_state + seed_offset,
                device=str(self.device_),
            )

    def _fit_learner(self, learner: KAN, X_t: torch.Tensor, residual: np.ndarray):
        r_t = torch.tensor(residual, dtype=torch.float32, device=self.device_).unsqueeze(1)
        dataset = {
            "train_input": X_t, "train_label": r_t,
            "test_input": X_t, "test_label": r_t,
        }
        with _suppress_pykan_noise():
            learner.fit(
                dataset, opt="Adam", steps=self.kan_steps, lr=self.kan_lr,
                loss_fn=nn.MSELoss(),
            )
        with torch.no_grad():
            return learner(X_t).cpu().numpy().flatten()

    def _check_fitted(self):
        if not self.learners_:
            raise RuntimeError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call fit() before predicting."
            )

    def _transform_X(self, X) -> torch.Tensor:
        self._check_fitted()
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(np.asarray(X), columns=self.feature_names_in_)
        X_arr = self.preprocessor_.transform(X)
        return torch.tensor(X_arr, dtype=torch.float32, device=self.device_)

    # ------------------------------------------------------------------
    def feature_importances(self) -> np.ndarray:
        """
        Approximate per-feature importance: L2 norm of each learner's
        first-layer spline coefficients per input dimension, summed over
        the ensemble and normalized to 1. A rough analogue of GBDT
        'gain' importance.
        """
        self._check_fitted()
        n_features = self.learners_[0].width[0][0]
        importances = np.zeros(n_features)
        for learner in self.learners_[: self.best_iteration_]:
            coef = learner.act_fun[0].coef.detach().cpu().numpy()
            importances += np.linalg.norm(coef, axis=(1, 2))
        total = importances.sum()
        return importances / total if total > 0 else importances

    def feature_importances_dict(self) -> dict:
        """Feature importances keyed by input column name, sorted desc."""
        imps = self.feature_importances()
        # preprocessor may reorder: numeric cols first, then categorical
        ordered_names = (
            list(self.preprocessor_.numeric_cols_)
            + list(self.preprocessor_.categorical_cols)
        )
        pairs = sorted(zip(ordered_names, imps), key=lambda kv: -kv[1])
        return {name: float(v) for name, v in pairs}
