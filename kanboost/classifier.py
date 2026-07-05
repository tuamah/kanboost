"""
KANBoostClassifier: binary classification via gradient boosting with
shallow KAN networks as weak learners.

Follows the classic Friedman (2001) gradient boosting recipe:

    F_0(x)      = log-odds of the base rate
    for t = 1..T:
        r_t     = pseudo-residuals = y - sigmoid(F_{t-1}(x))     (logloss)
        f_t     = a small KAN fit to (X, r_t)
        F_t(x)  = F_{t-1}(x) + learning_rate * f_t(x)
    prediction  = sigmoid(F_T(x))

Each weak learner is a small, shallow KAN so it plays the same structural
role a shallow decision tree plays in XGBoost/CatBoost: a cheap,
high-bias/low-variance component that is only useful in aggregate.
"""

from __future__ import annotations

import numpy as np
import torch

from sklearn.base import ClassifierMixin

from ._base import _BaseKANBoost, _validate_Xy


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class KANBoostClassifier(ClassifierMixin, _BaseKANBoost):
    """Gradient-boosted KAN ensemble for binary classification.

    Parameters
    ----------
    n_estimators : int, default=100
        Maximum number of boosting iterations (weak learners).
    learning_rate : float, default=0.1
        Shrinkage applied to each learner's contribution.
    kan_hidden : int, default=3
        Hidden-layer width of each weak KAN learner.
    kan_grid : int, default=2
        Number of B-spline grid intervals per edge function.
    kan_k : int, default=3
        B-spline polynomial degree.
    kan_steps : int, default=20
        Optimizer steps used to fit each weak learner.
    kan_lr : float, default=0.02
        Learning rate of each weak learner's inner optimizer.
    early_stopping_rounds : int or None, default=10
        Stop if validation logloss hasn't improved for this many rounds.
        Requires eval_set in fit(). None disables early stopping.
    categorical_cols : list of str, optional
        Column names to target-mean encode automatically.
    random_state : int, default=42
    verbose : bool, default=False
    device : str or None, default=None
        Torch device to train and predict on, e.g. "cpu", "cuda", "cuda:0".
        None auto-selects "cuda" when available, else "cpu".
    """


    def fit(self, X, y, eval_set: tuple | None = None):
        """Fit the boosted ensemble.

        Parameters
        ----------
        X : DataFrame or array of shape (n_samples, n_features)
        y : array of shape (n_samples,) with values in {0, 1}
        eval_set : (X_val, y_val) tuple, optional
            Validation data for early stopping.
        """
        X, y, X_arr = self._prepare_fit(X, y)

        classes = np.unique(y)
        if not np.array_equal(classes, [0, 1]) and not np.array_equal(classes, [0]) \
                and not np.array_equal(classes, [1]):
            raise ValueError(
                f"KANBoostClassifier supports binary targets in {{0, 1}}; "
                f"got classes {classes}. Multiclass is on the roadmap."
            )
        self.classes_ = np.array([0, 1])

        X_t = torch.tensor(X_arr, dtype=torch.float32, device=self.device_)
        n_features = X_arr.shape[1]

        p = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
        self.init_pred_ = float(np.log(p / (1 - p)))
        F = np.full(len(y), self.init_pred_)

        X_val_t = y_val = F_val = None
        if eval_set is not None:
            X_val_df, y_val = eval_set
            X_val_df, y_val = _validate_Xy(X_val_df, y_val)
            X_val_arr = self.preprocessor_.transform(X_val_df)
            X_val_t = torch.tensor(X_val_arr, dtype=torch.float32, device=self.device_)
            F_val = np.full(len(y_val), self.init_pred_)

        best_val_loss = np.inf
        rounds_since_best = 0
        self.learners_ = []
        self.best_iteration_ = None

        for t in range(self.n_estimators):
            residual = y - _sigmoid(F)

            learner = self._new_learner(n_features, seed_offset=t)
            update = self._fit_learner(learner, X_t, residual)
            F += self.learning_rate * update
            self.learners_.append(learner)

            if X_val_t is not None:
                with torch.no_grad():
                    F_val += self.learning_rate * learner(X_val_t).cpu().numpy().flatten()
                val_prob = np.clip(_sigmoid(F_val), 1e-7, 1 - 1e-7)
                val_loss = -float(np.mean(
                    y_val * np.log(val_prob) + (1 - y_val) * np.log(1 - val_prob)
                ))

                if self.verbose:
                    print(f"[{t + 1}/{self.n_estimators}] val_logloss={val_loss:.5f}")

                if val_loss < best_val_loss - 1e-5:
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
            elif self.verbose and (t + 1) % max(1, self.n_estimators // 10) == 0:
                print(f"[{t + 1}/{self.n_estimators}] "
                      f"train residual std={residual.std():.4f}")

        if self.best_iteration_ is None:
            self.best_iteration_ = len(self.learners_)
        return self

    # ------------------------------------------------------------------
    def _raw_score(self, X) -> np.ndarray:
        X_t = self._transform_X(X)
        F = np.full(X_t.shape[0], self.init_pred_)
        for learner in self.learners_[: self.best_iteration_]:
            with torch.no_grad():
                F += self.learning_rate * learner(X_t).cpu().numpy().flatten()
        return F

    def predict_proba(self, X) -> np.ndarray:
        """Return array of shape (n_samples, 2): P(class 0), P(class 1)."""
        prob_pos = _sigmoid(self._raw_score(X))
        return np.vstack([1 - prob_pos, prob_pos]).T

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        """Return hard 0/1 predictions at the given probability threshold."""
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def evaluate(self, X, y, threshold: float = 0.5, verbose: bool = True) -> dict:
        """Predict on X and report confusion matrix, accuracy, precision,
        recall, F1, and ROC-AUC against y. Returns the metrics dict."""
        from .metrics import classification_report_dict, print_classification_report

        y_prob = self.predict_proba(X)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        report = classification_report_dict(y, y_pred, y_prob)
        if verbose:
            print_classification_report(report)
        return report
