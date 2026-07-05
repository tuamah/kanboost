"""
kanboost._base -- shared machinery for KANBoostClassifier / KANBoostRegressor.

Contains:
- input validation
- the shared boosting loop skeleton
- sklearn-compatible get_params / set_params (so KANBoost estimators work
  inside sklearn Pipelines, GridSearchCV, and kantun without adapters)
- feature importances (shared by both estimators)
- model persistence (save/load)
- partial-dependence-style spline plotting
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


def _validate_sample_weight(sample_weight, y):
    if sample_weight is None:
        return None
    sample_weight = np.asarray(sample_weight, dtype=float).ravel()
    if len(sample_weight) != len(y):
        raise ValueError(
            f"sample_weight has {len(sample_weight)} entries; expected {len(y)}"
        )
    if (sample_weight < 0).any():
        raise ValueError("sample_weight must be non-negative")
    if sample_weight.sum() <= 0:
        raise ValueError("sample_weight must have at least one positive entry")
    return sample_weight


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
        return X, y, X_arr

    def _new_learner(self, n_features: int, seed_offset: int) -> KAN:
        with _suppress_pykan_noise():
            return KAN(
                width=[n_features, self.kan_hidden, 1],
                grid=self.kan_grid,
                k=self.kan_k,
                seed=self.random_state + seed_offset,
                device=str(self.device_),
                auto_save=False,
            )

    def _fit_learner(
        self,
        learner: KAN,
        X_t: torch.Tensor,
        residual: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ):
        r_t = torch.tensor(residual, dtype=torch.float32, device=self.device_).unsqueeze(1)
        dataset = {
            "train_input": X_t, "train_label": r_t,
            "test_input": X_t, "test_label": r_t,
        }
        if sample_weight is not None:
            # w_t is indexed positionally against the full training set; this only
            # stays aligned with pred/target because we always call fit() with the
            # default full-batch (batch=-1). Do not pass batch= without also
            # slicing w_t to match the sampled batch.
            w_t = torch.tensor(sample_weight, dtype=torch.float32, device=self.device_).unsqueeze(1)
            loss_fn = lambda pred, target: torch.mean(w_t * (pred - target) ** 2)
        else:
            loss_fn = nn.MSELoss()
        with _suppress_pykan_noise():
            learner.fit(
                dataset, opt="Adam", steps=self.kan_steps, lr=self.kan_lr,
                loss_fn=loss_fn, update_grid=False,
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

    def _raw_score_chain(self, X_t, learners, init_pred, best_iteration) -> np.ndarray:
        """Sum of a single boosting chain's contributions on already-transformed input."""
        F = np.full(X_t.shape[0], init_pred)
        for learner in learners[:best_iteration]:
            with torch.no_grad():
                F += self.learning_rate * learner(X_t).cpu().numpy().flatten()
        return F

    # ------------------------------------------------------------------
    def feature_importances(self) -> np.ndarray:
        """
        Approximate per-feature importance: L2 norm of each learner's
        first-layer spline coefficients per input dimension, summed over
        the ensemble and normalized to 1. A rough analogue of GBDT
        'gain' importance. For multiclass classifiers, importances are
        summed across all one-vs-rest chains before normalizing.
        """
        self._check_fitted()
        if isinstance(self.learners_, dict):
            chains = list(self.learners_.values())
            best_iterations = list(self.best_iteration_.values())
        else:
            chains = [self.learners_]
            best_iterations = [self.best_iteration_]

        n_features = chains[0][0].width[0][0]
        importances = np.zeros(n_features)
        for learners, best_iteration in zip(chains, best_iterations):
            for learner in learners[:best_iteration]:
                coef = learner.act_fun[0].coef.detach().cpu().numpy()
                importances += np.linalg.norm(coef, axis=(1, 2))
        total = importances.sum()
        return importances / total if total > 0 else importances

    def feature_importances_dict(self) -> dict:
        """Feature importances keyed by transformed column name, sorted desc."""
        imps = self.feature_importances()
        ordered_names = self.preprocessor_.transformed_feature_names()
        pairs = sorted(zip(ordered_names, imps), key=lambda kv: -kv[1])
        return {name: float(v) for name, v in pairs}

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Serialize this fitted model to `path` (a single file).

        pykan's KAN modules can't be pickled directly (they hold
        non-picklable closures internally), so each learner is stored as
        a state_dict and rebuilt from the model's hyperparameters on load.
        """
        self._check_fitted()

        def _freeze(obj):
            if isinstance(obj, KAN):
                return {"__kan_state_dict__": obj.state_dict()}
            if isinstance(obj, list):
                return [_freeze(o) for o in obj]
            if isinstance(obj, dict):
                return {k: _freeze(v) for k, v in obj.items()}
            return obj

        payload = {
            "format_version": 1,
            "class_name": type(self).__name__,
            "params": self.get_params(),
            "kan_arch": {
                "width": [self.preprocessor_.output_width, self.kan_hidden, 1],
                "grid": self.kan_grid,
                "k": self.kan_k,
            },
            "state": _freeze(dict(self.__dict__)),
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, device: str | None = None):
        """Load a model previously saved with `.save(path)`.

        `device` overrides the device the model is restored onto (default:
        auto-detect cuda, else cpu -- independent of what device it was
        trained/saved on).
        """
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("class_name") != cls.__name__:
            raise ValueError(
                f"{path!r} was saved from {payload.get('class_name')!r}, "
                f"not {cls.__name__!r}."
            )

        params = dict(payload["params"])
        if device is not None:
            params["device"] = device
        obj = cls(**params)
        obj.device_ = obj._resolve_device()
        arch = payload["kan_arch"]

        def _thaw(obj_):
            if isinstance(obj_, dict) and "__kan_state_dict__" in obj_:
                with _suppress_pykan_noise():
                    learner = KAN(
                        width=arch["width"], grid=arch["grid"], k=arch["k"],
                        seed=0, device=str(obj.device_), auto_save=False,
                    )
                    learner.load_state_dict(obj_["__kan_state_dict__"])
                return learner
            if isinstance(obj_, list):
                return [_thaw(o) for o in obj_]
            if isinstance(obj_, dict):
                return {k: _thaw(v) for k, v in obj_.items()}
            return obj_

        state = _thaw(payload["state"])
        state.pop("device", None)
        state.pop("device_", None)
        obj.__dict__.update(state)
        return obj

    # ------------------------------------------------------------------
    # interpretability
    # ------------------------------------------------------------------
    def plot_feature(self, feature_name: str, resolution: int = 200, ax=None):
        """Partial-dependence-style plot of the ensemble's response to one
        feature, with every other (scaled) feature held at its robust-scaled
        median (0.0). Plots the raw boosted score -- log-odds for
        KANBoostClassifier, the predicted target for KANBoostRegressor.

        Exact for `kan_hidden=1` (a pure additive/GAM-style ensemble, since
        input features don't mix before the output layer); an approximation
        when `kan_hidden > 1` lets the hidden layer combine features.

        For a multiclass classifier, one curve per class is drawn.
        """
        import matplotlib.pyplot as plt

        self._check_fitted()
        names = self.preprocessor_.transformed_feature_names()
        if feature_name not in names:
            raise ValueError(f"Unknown feature {feature_name!r}; known features: {names}")
        col = names.index(feature_name)

        grid = np.linspace(-1, 1, resolution)
        X_probe = np.zeros((resolution, len(names)))
        X_probe[:, col] = grid
        X_t = torch.tensor(X_probe, dtype=torch.float32, device=self.device_)

        plot_ax = ax if ax is not None else plt.gca()
        if isinstance(self.learners_, dict):
            for c in self.classes_:
                score = self._raw_score_chain(
                    X_t, self.learners_[c], self.init_pred_[c], self.best_iteration_[c]
                )
                plot_ax.plot(grid, score, label=f"class {c}")
            plot_ax.legend()
        else:
            score = self._raw_score_chain(X_t, self.learners_, self.init_pred_, self.best_iteration_)
            plot_ax.plot(grid, score)

        plot_ax.set_xlabel(f"{feature_name} (scaled)")
        plot_ax.set_ylabel("raw score")
        plot_ax.set_title(f"Partial dependence: {feature_name}")
        return plot_ax
