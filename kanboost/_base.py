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
- KAN-native interpretability: analytic derivatives, symbolic extraction
  (GAM mode), monotonic constraints, pruning, grid refinement, feature
  interaction scores
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
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.kan_hidden = kan_hidden
        self.kan_grid = kan_grid
        self.kan_k = kan_k
        self.kan_steps = kan_steps
        self.kan_lr = kan_lr
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.categorical_cols = categorical_cols
        self.random_state = random_state
        self.verbose = verbose
        self.device = device
        self.batch_size = batch_size
        self.gam = gam
        self.monotone_constraints = monotone_constraints
        self.lamb = lamb
        self.lamb_l1 = lamb_l1
        self.lamb_coefdiff = lamb_coefdiff

        # fitted state
        self.preprocessor_ = None
        self.learners_ = []
        self.init_pred_ = None
        self.best_iteration_ = None
        self.feature_names_in_ = None
        self.device_ = None
        self.monotone_signs_ = None

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
        if self.validation_fraction is not None and not (0 < self.validation_fraction < 1):
            raise ValueError("validation_fraction must be in (0, 1)")
        if self.batch_size is not None and self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.monotone_constraints:
            if not self.gam:
                raise ValueError(
                    "monotone_constraints requires gam=True -- monotonicity through "
                    "a hidden layer's own spline can't be guaranteed edge-wise; GAM "
                    "mode (kan_hidden=1, identity output layer) makes the constraint sound."
                )
            if self.kan_hidden != 1:
                raise ValueError("monotone_constraints requires kan_hidden=1 (GAM mode).")
            if any(v not in (1, -1) for v in self.monotone_constraints.values()):
                raise ValueError("monotone_constraints values must be 1 (increasing) or -1 (decreasing)")

    def _resolve_device(self) -> torch.device:
        """`device=None` auto-selects cuda when available, else cpu."""
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _resolve_monotone_signs(self) -> np.ndarray:
        names = self.preprocessor_.transformed_feature_names()
        signs = np.zeros(len(names))
        if self.monotone_constraints:
            for col, sign in self.monotone_constraints.items():
                if col not in names:
                    raise ValueError(
                        f"monotone_constraints references unknown feature {col!r}; "
                        f"known features: {names}"
                    )
                signs[names.index(col)] = sign
        return signs

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
        self.monotone_signs_ = self._resolve_monotone_signs()
        return X, y, X_arr

    def _internal_split(self, X, y, sample_weight, stratify: bool):
        """Carve out a validation slice from (X, y) for early stopping when
        the caller didn't pass an explicit `eval_set`. Splitting happens
        before preprocessing is fit, so the held-out rows never leak into
        the scaler/encoder statistics -- the same guarantee an external
        eval_set would give.
        """
        from sklearn.model_selection import train_test_split

        X, y = _validate_Xy(X, y)
        sample_weight = _validate_sample_weight(sample_weight, y)
        indices = np.arange(len(y))
        idx_train, idx_val = train_test_split(
            indices, test_size=self.validation_fraction,
            random_state=self.random_state, stratify=y if stratify else None,
        )
        X_train = X.iloc[idx_train].reset_index(drop=True)
        X_val = X.iloc[idx_val].reset_index(drop=True)
        y_train, y_val = y[idx_train], y[idx_val]

        sw_train = None
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float).ravel()
            sw_train = sample_weight[idx_train]

        return X_train, X_val, y_train, y_val, sw_train

    def _learner_kan_kwargs(self, n_features: int, seed_offset: int) -> dict:
        """`KAN(...)` constructor kwargs, shared by `_new_learner` and
        `load()` so a saved model's learners are rebuilt identically
        (monotone/GAM settings included) before `load_state_dict`."""
        kwargs = dict(
            width=[n_features, self.kan_hidden, 1],
            grid=self.kan_grid,
            k=self.kan_k,
            seed=self.random_state + seed_offset,
            device=str(self.device_),
            auto_save=False,
        )
        if self.monotone_constraints:
            # Freeze the SiLU "base" branch at exactly zero (it isn't monotone
            # itself) and freeze the spline branch's sign, so the only thing
            # `_apply_monotone_projection` needs to keep monotone is `coef`.
            kwargs.update(scale_base_mu=0.0, scale_base_sigma=0.0, sb_trainable=False, sp_trainable=False)
        return kwargs

    def _new_learner(self, n_features: int, seed_offset: int) -> KAN:
        with _suppress_pykan_noise():
            learner = KAN(**self._learner_kan_kwargs(n_features, seed_offset))
            if self.gam:
                # Fix the single output edge to the identity function, so the
                # whole learner reduces to an exact sum of per-feature spline
                # shape functions: F(x) = sum_j g_j(x_j). Required for both
                # exact GAM-mode attribution and sound monotonic constraints.
                learner.fix_symbolic(1, 0, 0, "x", fit_params_bool=False, verbose=False, log_history=False)
            return learner

    def _apply_monotone_projection(self, learner: KAN) -> None:
        """Project the first layer's B-spline control points (`coef`) onto
        the monotone cone for each constrained feature. Sorted control
        points guarantee a monotone curve (the B-spline variation-diminishing
        property), so this is a hard projection, not a penalty."""
        if not self.monotone_constraints:
            return
        with torch.no_grad():
            coef = learner.act_fun[0].coef
            for j, sign in enumerate(self.monotone_signs_):
                if sign > 0:
                    coef.data[j] = torch.cummax(coef.data[j], dim=-1).values
                elif sign < 0:
                    coef.data[j] = -torch.cummax(-coef.data[j], dim=-1).values

    def _fit_learner(
        self,
        learner: KAN,
        X_t: torch.Tensor,
        residual: np.ndarray,
        sample_weight: np.ndarray | None = None,
        seed_offset: int = 0,
    ):
        r_t = torch.tensor(residual, dtype=torch.float32, device=self.device_).unsqueeze(1)
        w_t = None
        if sample_weight is not None:
            w_t = torch.tensor(sample_weight, dtype=torch.float32, device=self.device_).unsqueeze(1)

        n = X_t.shape[0]
        needs_custom_loop = bool(self.monotone_constraints) or (
            self.batch_size is not None and self.batch_size < n
        )
        if needs_custom_loop:
            batch_size = self.batch_size if (self.batch_size is not None and self.batch_size < n) else n
            self._fit_learner_custom_loop(learner, X_t, r_t, w_t, seed_offset, batch_size)
        else:
            dataset = {
                "train_input": X_t, "train_label": r_t,
                "test_input": X_t, "test_label": r_t,
            }
            if w_t is not None:
                loss_fn = lambda pred, target: torch.mean(w_t * (pred - target) ** 2)
            else:
                loss_fn = nn.MSELoss()
            with _suppress_pykan_noise():
                learner.fit(
                    dataset, opt="Adam", steps=self.kan_steps, lr=self.kan_lr,
                    loss_fn=loss_fn, update_grid=False,
                    lamb=self.lamb, lamb_l1=self.lamb_l1, lamb_coefdiff=self.lamb_coefdiff,
                )
        with torch.no_grad():
            return learner(X_t).cpu().numpy().flatten()

    def _fit_learner_custom_loop(self, learner: KAN, X_t, r_t, w_t, seed_offset: int, batch_size: int):
        """Plain Adam loop over `learner(x_batch)`, bypassing pykan's own
        `.fit()`. Used whenever `batch_size` requires sampling (so
        `sample_weight` stays trivially aligned with the sampled batch,
        unlike pykan's own `batch=` option) and whenever
        `monotone_constraints` is set (so `_apply_monotone_projection` can
        run after every optimizer step -- pykan's `.fit()` doesn't expose a
        per-step hook). Note: `lamb`/`lamb_l1`/`lamb_coefdiff` regularization
        only applies on the pykan `.fit()` path, not here.

        `seed_offset` (the same per-learner offset `_new_learner` uses)
        must vary across calls -- otherwise every weak learner in the
        ensemble would draw the identical sequence of mini-batches, and
        rows never selected by that one sequence would never contribute
        a gradient to any learner.
        """
        n = X_t.shape[0]
        rng = np.random.RandomState(self.random_state + seed_offset)
        optimizer = torch.optim.Adam(learner.parameters(), lr=self.kan_lr)
        with _suppress_pykan_noise():
            for _ in range(self.kan_steps):
                if batch_size < n:
                    idx = torch.as_tensor(
                        rng.choice(n, size=batch_size, replace=False), device=self.device_
                    )
                else:
                    idx = slice(None)
                pred = learner(X_t[idx])
                target = r_t[idx]
                if w_t is not None:
                    loss = torch.mean(w_t[idx] * (pred - target) ** 2)
                else:
                    loss = torch.mean((pred - target) ** 2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self._apply_monotone_projection(learner)

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

    def _all_chains(self):
        """(learners, best_iteration) pairs for every chain -- one for
        regressors/binary classifiers, one per class for multiclass."""
        if isinstance(self.learners_, dict):
            return list(zip(self.learners_.values(), self.best_iteration_.values()))
        return [(self.learners_, self.best_iteration_)]

    def _boost_chain(self, X_t, y, loss, n_features, X_val_t, y_val, sample_weight, seed_base):
        """Train one boosting chain against `loss` (see `losses.py`).

        Shared by KANBoostRegressor (one chain) and KANBoostClassifier
        (one chain per one-vs-rest class). Returns (learners, init_pred,
        best_iteration).
        """
        init_pred = loss.init_pred(y, sample_weight)
        F = np.full(len(y), init_pred)
        F_val = np.full(len(y_val), init_pred) if X_val_t is not None else None

        best_val_loss = np.inf
        rounds_since_best = 0
        learners = []
        best_iteration = None

        for t in range(self.n_estimators):
            residual = loss.negative_gradient(y, F)

            learner = self._new_learner(n_features, seed_offset=seed_base + t)
            update = self._fit_learner(
                learner, X_t, residual, sample_weight=sample_weight, seed_offset=seed_base + t,
            )
            F += self.learning_rate * update
            learners.append(learner)

            if X_val_t is not None:
                with torch.no_grad():
                    F_val += self.learning_rate * learner(X_val_t).cpu().numpy().flatten()
                val_loss = loss.val_loss(y_val, F_val)

                if self.verbose:
                    print(f"[{t + 1}/{self.n_estimators}] val_loss={val_loss:.5f}")

                if val_loss < best_val_loss - loss.improvement_eps:
                    best_val_loss = val_loss
                    rounds_since_best = 0
                    best_iteration = t + 1
                else:
                    rounds_since_best += 1
                    if (self.early_stopping_rounds is not None
                            and rounds_since_best >= self.early_stopping_rounds):
                        if self.verbose:
                            print(f"Early stopping at iteration {t + 1}")
                        break
            elif self.verbose and (t + 1) % max(1, self.n_estimators // 10) == 0:
                print(f"[{t + 1}/{self.n_estimators}] "
                      f"train residual std={residual.std():.4f}")

        if best_iteration is None:
            best_iteration = len(learners)
        return learners, init_pred, best_iteration

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
        chains = self._all_chains()
        n_features = chains[0][0][0].width[0][0]
        importances = np.zeros(n_features)
        for learners, best_iteration in chains:
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
        n_features = arch["width"][0]

        def _thaw(obj_):
            if isinstance(obj_, dict) and "__kan_state_dict__" in obj_:
                with _suppress_pykan_noise():
                    learner = KAN(**obj._learner_kan_kwargs(n_features, seed_offset=0))
                    if obj.gam:
                        # symbolic_fun's `funs`/`funs_name` are plain Python
                        # lists, not Parameters, so they aren't in the saved
                        # state_dict -- re-apply fix_symbolic before loading
                        # the (serialized) mask/affine Parameters, or the
                        # thawed learner's output edge silently reverts to
                        # whatever the default symbolic function is.
                        learner.fix_symbolic(1, 0, 0, "x", fit_params_bool=False, verbose=False, log_history=False)
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

    def feature_contributions(self, X) -> np.ndarray | dict:
        """Per-sample, per-feature attribution, computed natively from each
        weak learner's first KAN layer rather than a post-hoc method like
        SHAP.

        Each learner's first layer produces one spline output per
        (feature, hidden-unit) edge; summing those edges over hidden units
        gives that learner's additive per-feature contribution to its
        hidden representation. This sum reconstructs the hidden
        representation exactly when `kan_hidden=1` (there is only one
        hidden unit, so no cross-feature mixing happens before it); with
        `gam=True` the output layer is also fixed to identity, so
        contributions sum *exactly* to the raw score. Otherwise the output
        layer's own spline is generally nonlinear, so contributions are
        informative but won't sum exactly to `predict_proba`/`predict`.

        Returns an (n_samples, n_transformed_features) array (columns
        ordered per `preprocessor_.transformed_feature_names()`), or --
        for a multiclass classifier -- a dict `{class_label: array}`.
        """
        self._check_fitted()
        X_t = self._transform_X(X)

        def _chain_contributions(learners, best_iteration):
            n_features = learners[0].width[0][0]
            contrib = np.zeros((X_t.shape[0], n_features))
            for learner in learners[:best_iteration]:
                with torch.no_grad():
                    _, _, postacts, _ = learner.act_fun[0](X_t)
                contrib += self.learning_rate * postacts.sum(dim=1).cpu().numpy()
            return contrib

        if isinstance(self.learners_, dict):
            return {
                c: _chain_contributions(self.learners_[c], self.best_iteration_[c])
                for c in self.classes_
            }
        return _chain_contributions(self.learners_, self.best_iteration_)

    def predict_derivative(self, X, feature_name: str) -> np.ndarray | dict:
        """Analytic per-sample derivative of the raw boosted score with
        respect to one (scaled) feature, via autograd through the whole
        ensemble.

        Trees have zero/undefined derivatives (piecewise-constant); a
        generic MLP only gives a pointwise autograd gradient. Because every
        KAN edge is a smooth spline, this derivative is an exact, globally
        defined function you can evaluate anywhere -- not a finite-difference
        approximation.

        Returns an (n_samples,) array, or -- for a multiclass classifier --
        a dict `{class_label: array}`.
        """
        self._check_fitted()
        names = self.preprocessor_.transformed_feature_names()
        if feature_name not in names:
            raise ValueError(f"Unknown feature {feature_name!r}; known features: {names}")
        col = names.index(feature_name)

        X_t = self._transform_X(X).clone().requires_grad_(True)

        def _chain_derivative(learners, init_pred, best_iteration):
            F = X_t.new_full((X_t.shape[0],), float(init_pred))
            for learner in learners[:best_iteration]:
                F = F + self.learning_rate * learner(X_t).flatten()
            grad, = torch.autograd.grad(F.sum(), X_t, retain_graph=False)
            return grad[:, col].detach().cpu().numpy()

        if isinstance(self.learners_, dict):
            return {
                c: _chain_derivative(self.learners_[c], self.init_pred_[c], self.best_iteration_[c])
                for c in self.classes_
            }
        return _chain_derivative(self.learners_, self.init_pred_, self.best_iteration_)

    def symbolic_report(self, X, top_k: int = 3) -> dict:
        """GAM-mode only (`gam=True`): fit a small library of closed-form
        functions (x, x^2, x^3, sin, cos, exp, log, sqrt, tanh, abs) to each
        feature's aggregated shape function `g_j`, sampled from the exact
        partial-dependence curve `plot_feature` already computes. Returns
        the top-`k` candidates per feature by R^2, as
        `{feature: [(name, r2), ...]}` -- or, for a multiclass classifier,
        `{class_label: {feature: [(name, r2), ...]}}` (one report per
        one-vs-rest chain; the chains fit different binary targets, so
        their shape functions must be scored separately, not summed).

        Because `gam=True` fixes the output layer to identity, each chain
        is exactly `F(x) = c + sum_j g_j(x_j)`, so this is one honest curve
        fit per feature -- not pykan's `auto_symbolic`, which only handles
        a single network and doesn't aggregate an ensemble. Candidate R^2
        values can be high for more than one candidate on a smooth curve
        (e.g. `sin` and `tanh` both fitting a monotone S-shape well) --
        treat this as a shortlist to inspect, not a single right answer.
        """
        if not self.gam:
            raise RuntimeError("symbolic_report requires gam=True.")
        self._check_fitted()
        from kan.utils import fit_params, SYMBOLIC_LIB

        candidates = ["x", "x^2", "x^3", "sin", "cos", "exp", "log", "sqrt", "tanh", "abs"]
        candidates = [c for c in candidates if c in SYMBOLIC_LIB]

        names = self.preprocessor_.transformed_feature_names()
        grid = torch.linspace(-1, 1, 200, device=self.device_)

        def _chain_report(learners, best_iteration):
            report = {}
            for j, name in enumerate(names):
                X_probe = torch.zeros((200, len(names)), device=self.device_)
                X_probe[:, j] = grid
                y = torch.zeros(200, device=self.device_)
                for learner in learners[:best_iteration]:
                    with torch.no_grad():
                        y = y + self.learning_rate * learner(X_probe).flatten()

                scored = []
                for name_c in candidates:
                    fun = SYMBOLIC_LIB[name_c][0]
                    try:
                        _, r2 = fit_params(grid, y, fun, verbose=False, device=str(self.device_))
                        scored.append((name_c, float(r2)))
                    except Exception:
                        continue
                scored.sort(key=lambda kv: -kv[1])
                report[name] = scored[:top_k]
            return report

        if isinstance(self.learners_, dict):
            return {
                c: _chain_report(self.learners_[c], self.best_iteration_[c])
                for c in self.classes_
            }
        return _chain_report(self.learners_, self.best_iteration_)

    def prune(self, X, threshold: float = 3e-2) -> None:
        """Zero out edges (and their input KAN-layer connections) whose
        contribution is below `threshold`, in place, for every learner in
        the ensemble -- shrinks saved model size and sparsifies
        `feature_contributions`. Uses pykan's own activation-based pruning
        (`KAN.prune_edge`), which trees have no equivalent of: a fitted
        tree's structure is fixed, but a fitted KAN's dead spline edges can
        be identified and removed post-hoc without retraining.
        """
        self._check_fitted()
        X_t = self._transform_X(X)
        with _suppress_pykan_noise():
            for learners, best_iteration in self._all_chains():
                for learner in learners[:best_iteration]:
                    learner.get_act(X_t)
                    learner.attribute(plot=False)
                    learner.prune_edge(threshold, log_history=False)

    def refine(self, X, new_grid: int) -> None:
        """Re-express every learner in the ensemble on a finer spline grid
        (`new_grid` intervals), in place, without retraining from scratch --
        pykan's `KAN.refine` least-squares-fits new, finer control points
        against the current curve sampled at `X`, so this is a near-lossless
        resolution upgrade (not bit-exact), not a full retrain. Fitted
        decision trees have no equivalent operation; refining a KAN ensemble
        this way is only possible because each edge is a continuous spline.
        """
        self._check_fitted()
        X_t = self._transform_X(X)

        def _refine_chain(learners, best_iteration):
            refined = []
            with _suppress_pykan_noise():
                for learner in learners[:best_iteration]:
                    learner.get_act(X_t)
                    new_learner = learner.refine(new_grid)
                    new_learner.auto_save = False
                    refined.append(new_learner)
            return refined

        if isinstance(self.learners_, dict):
            for c in self.classes_:
                self.learners_[c] = _refine_chain(self.learners_[c], self.best_iteration_[c])
                self.best_iteration_[c] = len(self.learners_[c])
        else:
            self.learners_ = _refine_chain(self.learners_, self.best_iteration_)
            self.best_iteration_ = len(self.learners_)
        self.kan_grid = new_grid

    def feature_interaction(self, X, top_k: int = 10, neuron_th: float = 1e-2, feature_th: float = 1e-2) -> dict:
        """Native structural feature-interaction scores: how much pairs of
        input features jointly drive a learner's output, traced back
        through the hidden layer via pykan's `KAN.feature_interaction`
        (layer index 1 -- the output neuron -- not layer 0, whose "groups"
        are just individual input features attributed to themselves). Only
        meaningful when `kan_hidden > 1` (with `kan_hidden=1` there is only
        one hidden unit, so every feature trivially "interacts" with itself).
        This is a structural score read directly off the trained weights,
        not a post-hoc perturbation method like SHAP interaction values.

        `neuron_th`/`feature_th` are pykan's own activation-magnitude
        thresholds for deciding a hidden neuron/feature is "active" -- lower
        them if interactions you expect aren't showing up.

        Returns the top-`top_k` feature pairs by aggregate interaction
        count, as `{(feature_a, feature_b): count}`.
        """
        if self.kan_hidden == 1:
            raise RuntimeError(
                "feature_interaction requires kan_hidden > 1 (with kan_hidden=1 "
                "there is only one hidden unit, so interaction scores are trivial)."
            )
        self._check_fitted()
        X_t = self._transform_X(X)
        names = self.preprocessor_.transformed_feature_names()

        import itertools

        counts: dict = {}
        with _suppress_pykan_noise():
            for learners, best_iteration in self._all_chains():
                for learner in learners[:best_iteration]:
                    learner.get_act(X_t)
                    # keys are tuples of feature indices that jointly drive the
                    # output neuron (arbitrary length, not just pairs)
                    active_groups = learner.feature_interaction(1, neuron_th=neuron_th, feature_th=feature_th)
                    for group, weight in active_groups.items():
                        for i, j in itertools.combinations(sorted(group), 2):
                            key = (names[i], names[j])
                            counts[key] = counts.get(key, 0) + weight

        return dict(sorted(counts.items(), key=lambda kv: -kv[1])[:top_k])
