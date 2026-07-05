"""
KANBoostClassifier: classification via gradient boosting with shallow KAN
networks as weak learners.

Binary targets follow the classic Friedman (2001) gradient boosting
recipe:

    F_0(x)      = log-odds of the base rate
    for t = 1..T:
        r_t     = pseudo-residuals = y - sigmoid(F_{t-1}(x))     (logloss)
        f_t     = a small KAN fit to (X, r_t)
        F_t(x)  = F_{t-1}(x) + learning_rate * f_t(x)
    prediction  = sigmoid(F_T(x))

Targets with more than two classes are handled one-vs-rest: one
independent binary boosting chain per class, combined via softmax over
the chains' raw scores at prediction time.

Each weak learner is a small, shallow KAN so it plays the same structural
role a shallow decision tree plays in XGBoost/CatBoost: a cheap,
high-bias/low-variance component that is only useful in aggregate.
"""

from __future__ import annotations

import numpy as np
import torch

from sklearn.base import ClassifierMixin

from ._base import _BaseKANBoost, _validate_Xy, _validate_sample_weight
from .losses import LogisticLoss, _sigmoid


class KANBoostClassifier(ClassifierMixin, _BaseKANBoost):
    """Gradient-boosted KAN ensemble for classification (binary or
    multiclass, one-vs-rest).

    Parameters
    ----------
    n_estimators : int, default=100
        Maximum number of boosting iterations (weak learners) per chain.
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
        Requires `eval_set` in `fit()`, or `validation_fraction` to carve
        one out automatically. None disables early stopping.
    validation_fraction : float or None, default=None
        If `early_stopping_rounds` is set and `fit()` is called without
        `eval_set`, this fraction of the (pre-preprocessing) training
        rows is held out -- stratified by class -- as an internal
        validation split.
    categorical_cols : list of str, optional
        Column names to target-mean encode automatically.
    random_state : int, default=42
    verbose : bool, default=False
    device : str or None, default=None
        Torch device to train and predict on, e.g. "cpu", "cuda", "cuda:0".
        None auto-selects "cuda" when available, else "cpu".
    batch_size : int or None, default=None
        If set and smaller than the training set, each weak learner is
        fit with mini-batch Adam instead of full-batch.
    gam : bool, default=False
        If True, fixes each weak learner's output edge to the identity
        function, so the whole ensemble reduces to an exact additive model
        `F(x) = c + sum_j g_j(x_j)`. Required for `monotone_constraints`
        and for `symbolic_report()`; also makes `feature_contributions()`
        exact (it otherwise ignores the output layer's own nonlinearity).
    monotone_constraints : dict or None, default=None
        `{feature_name: 1 or -1}` to force the ensemble's dependence on
        that (transformed) feature to be non-decreasing (`1`) or
        non-increasing (`-1`). Requires `gam=True` and `kan_hidden=1` --
        enforced via a hard projection onto the first layer's B-spline
        control points after every optimizer step (the "variation
        diminishing" property: sorted control points guarantee a monotone
        curve), not a soft penalty.
    lamb, lamb_l1, lamb_coefdiff : float, default=0.0, 1.0, 0.0
        pykan's own regularization strengths (`KAN.fit`'s `lamb`,
        `lamb_l1`, `lamb_coefdiff`), passed straight through. Only applied
        on the full-batch path -- have no effect when `batch_size` or
        `monotone_constraints` forces the custom Adam loop.
    """

    def fit(self, X, y, eval_set: tuple | None = None, sample_weight=None):
        """Fit the boosted ensemble.

        Parameters
        ----------
        X : DataFrame or array of shape (n_samples, n_features)
        y : array of shape (n_samples,) -- 2 or more distinct classes
        eval_set : (X_val, y_val) tuple, optional
            Validation data for early stopping.
        sample_weight : array of shape (n_samples,), optional
            Per-sample weights used when fitting each weak learner and
            when computing the initial log-odds. Not applied to
            categorical target-mean encoding or to eval_set's loss.
        """
        if (eval_set is None and self.validation_fraction is not None
                and self.early_stopping_rounds is not None):
            X, X_eval, y, y_eval, sample_weight = self._internal_split(
                X, y, sample_weight, stratify=True
            )
            eval_set = (X_eval, y_eval)

        X, y, X_arr = self._prepare_fit(X, y)
        sample_weight = _validate_sample_weight(sample_weight, y)

        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError(
                f"KANBoostClassifier needs at least 2 classes; got {self.classes_}."
            )

        X_t = torch.tensor(X_arr, dtype=torch.float32, device=self.device_)
        n_features = X_arr.shape[1]

        X_val_t = y_val = None
        if eval_set is not None:
            X_val_df, y_val = eval_set
            X_val_df, y_val = _validate_Xy(X_val_df, y_val)
            X_val_arr = self.preprocessor_.transform(X_val_df)
            X_val_t = torch.tensor(X_val_arr, dtype=torch.float32, device=self.device_)

        loss = LogisticLoss()
        if len(self.classes_) == 2:
            y_bin = (y == self.classes_[1]).astype(float)
            y_val_bin = (y_val == self.classes_[1]).astype(float) if y_val is not None else None
            self.learners_, self.init_pred_, self.best_iteration_ = self._boost_chain(
                X_t, y_bin, loss, n_features, X_val_t, y_val_bin, sample_weight, seed_base=0,
            )
        else:
            self.learners_ = {}
            self.init_pred_ = {}
            self.best_iteration_ = {}
            for i, c in enumerate(self.classes_):
                y_bin = (y == c).astype(float)
                y_val_bin = (y_val == c).astype(float) if y_val is not None else None
                learners, init_pred, best_iteration = self._boost_chain(
                    X_t, y_bin, loss, n_features, X_val_t, y_val_bin, sample_weight,
                    seed_base=i * self.n_estimators,
                )
                self.learners_[c] = learners
                self.init_pred_[c] = init_pred
                self.best_iteration_[c] = best_iteration

        return self

    # ------------------------------------------------------------------
    def predict_proba(self, X) -> np.ndarray:
        """Return array of shape (n_samples, n_classes)."""
        X_t = self._transform_X(X)
        if len(self.classes_) == 2:
            prob_pos = _sigmoid(self._raw_score_chain(
                X_t, self.learners_, self.init_pred_, self.best_iteration_
            ))
            return np.vstack([1 - prob_pos, prob_pos]).T

        raw = np.column_stack([
            self._raw_score_chain(
                X_t, self.learners_[c], self.init_pred_[c], self.best_iteration_[c]
            )
            for c in self.classes_
        ])
        raw = raw - raw.max(axis=1, keepdims=True)
        exp = np.exp(raw)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        """Return hard class predictions.

        `threshold` only applies to the binary case (probability of the
        positive class, `classes_[1]`); multiclass always uses argmax.
        """
        proba = self.predict_proba(X)
        if len(self.classes_) == 2:
            return np.where(proba[:, 1] >= threshold, self.classes_[1], self.classes_[0])
        return self.classes_[np.argmax(proba, axis=1)]

    def evaluate(self, X, y, threshold: float = 0.5, verbose: bool = True) -> dict:
        """Predict on X and report classification metrics against y."""
        from .metrics import classification_report_dict, print_classification_report

        proba = self.predict_proba(X)
        if len(self.classes_) == 2:
            y_prob = proba[:, 1]
            y_pred = np.where(y_prob >= threshold, self.classes_[1], self.classes_[0])
        else:
            y_prob = proba
            y_pred = self.classes_[np.argmax(proba, axis=1)]

        report = classification_report_dict(y, y_pred, y_prob, labels=self.classes_)
        if verbose:
            print_classification_report(report, class_names=[str(c) for c in self.classes_])
        return report
