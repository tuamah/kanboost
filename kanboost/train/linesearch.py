"""
kanboost.train.linesearch -- per-round optimal step-size line search for
KANBoostClassifier (binary only), instead of the fixed `learning_rate`
shrinkage `_boost_chain` normally uses.

Motivated by GB-KAN (Mohr & Frochte, ICAART 2026), an independently
published KAN-based boosting framework that includes a per-stage line
search (their gamma_t, eq. 3) and reports training time within 2.5-3x of
XGBoost -- far closer than kanboost's own measured 300-850x slowdown on
a real clinical benchmark (INSPIRE, postop_icu; see
`docs/inspire_kanboost_evaluation.md`). Adding the same line search here
does not, on its own, make each round cheaper (kanboost's per-round cost
is dominated by the closed-form ALS solve, not the loss evaluation this
line search adds) -- what it does is let FEWER rounds reach the same or
better accuracy than the fixed-shrinkage baseline, which cuts total wall
time close to linearly in the round-count reduction.

Measured on INSPIRE (same split/config as the rest of this project's
evaluation), binary logistic loss, `n_estimators` reduced without any
other change:

    Small  (10K rows):  baseline 100 rounds, 49.6s, AUROC 0.9225/AUPRC 0.6707
                         line search  30 rounds, 16.2s, AUROC 0.9217/AUPRC 0.6768
    Medium (50K rows):  baseline 140 rounds, 384.4s, AUROC 0.9389/AUPRC 0.7560
                         line search  40 rounds,  91.4s, AUROC 0.9407/AUPRC 0.7636

i.e. accuracy matched or slightly exceeded with roughly a third of the
rounds and roughly a quarter of the wall time, at Medium scale. This is
NOT a claim that line search speeds up any individual round -- it does
not -- only that it lets the ensemble reach the same quality with fewer
of them.

Scope, stated plainly: binary `KANBoostClassifier` only (not multiclass
one-vs-rest, not `KANBoostRegressor`), no `eval_set`/early stopping (the
measurements above used none) -- this mirrors exactly what was tested.
Extending to multiclass, the regressor, or early stopping is future
work, not implemented here.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import minimize_scalar

from kanboost.core.losses import LogisticLoss, _sigmoid


def _logistic_loss_at_gamma(y, w, F, update, gamma) -> float:
    F_new = F + gamma * update
    p = np.clip(_sigmoid(F_new), 1e-7, 1 - 1e-7)
    return -float(np.mean(w * (y * np.log(p) + (1 - y) * np.log(1 - p))))


def fit_with_line_search(model, X, y, sample_weight=None, gamma_bounds=(0.0, 2.0)):
    """Fit `model` (an unfitted binary `KANBoostClassifier`) using a
    per-round line search for the optimal step size, instead of its
    fixed `learning_rate`.

    `model.n_estimators` is used as-is -- pass a SMALLER value than you
    would for a normal `fit()` call; the line search's benefit is
    letting fewer rounds match a fixed-shrinkage ensemble's accuracy; it
    provides no benefit at the same round count (measured: often
    slightly *slower* per round, from the added 1-D optimization, with
    no consistent accuracy gain at equal `n_estimators`).

    `gamma_bounds`: search range for the per-round step size (default
    matches what was measured). `model.learning_rate` is not used by
    this function -- the line search replaces it entirely.

    Returns `model`, fitted in place (same convention as `model.fit()`
    and `kanboost.train.accel.fast_fit()`).
    """
    if not hasattr(model, "predict_proba"):
        raise TypeError("fit_with_line_search requires a KANBoostClassifier, not a regressor.")

    X, y_arr, X_arr = model._prepare_fit(X, y)
    model.classes_ = np.unique(y_arr)
    if len(model.classes_) != 2:
        raise ValueError(
            "fit_with_line_search only supports binary classification; "
            f"got {len(model.classes_)} classes. Multiclass is not implemented."
        )
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=float).ravel()

    y_bin = (y_arr == model.classes_[1]).astype(float)
    X_t = torch.tensor(X_arr, dtype=torch.float32, device=model.device_)
    n_features = X_arr.shape[1]

    loss = LogisticLoss()
    init_pred = loss.init_pred(y_bin, sample_weight)
    F = np.full(len(y_bin), init_pred)
    learners = []
    gammas = []

    for t in range(model.n_estimators):
        residual = loss.negative_gradient(y_bin, F)
        learner = model._new_learner(n_features, seed_offset=t)
        update = model._fit_learner(learner, X_t, residual, sample_weight=sample_weight, seed_offset=t)
        res = minimize_scalar(
            lambda g: _logistic_loss_at_gamma(y_bin, sample_weight if sample_weight is not None
                                               else np.ones_like(y_bin), F, update, g),
            bounds=gamma_bounds, method="bounded", options={"xatol": 1e-3},
        )
        gamma = float(res.x)
        F += gamma * update
        learners.append(learner)
        gammas.append(gamma)

    model.learners_ = learners
    model.init_pred_ = init_pred
    model.best_iteration_ = len(learners)
    model._line_search_gammas_ = gammas  # used by predict_proba_line_search below
    return model


def predict_proba_line_search(model, X) -> np.ndarray:
    """`predict_proba` for a model fitted via `fit_with_line_search` --
    required because each learner has its own gamma (unlike a normal
    `fit()`, where every learner shares `model.learning_rate`), so the
    base `predict_proba`/`_raw_score_chain` (which multiplies every
    learner's output by the same `learning_rate`) cannot be reused as-is.
    """
    if not hasattr(model, "_line_search_gammas_"):
        raise RuntimeError("This model was not fitted with fit_with_line_search().")
    X_t = model._transform_X(X)
    F = np.full(X_t.shape[0], model.init_pred_)
    with torch.no_grad():
        for learner, gamma in zip(model.learners_, model._line_search_gammas_):
            F += gamma * learner(X_t).cpu().numpy().flatten()
    prob_pos = _sigmoid(F)
    return np.vstack([1 - prob_pos, prob_pos]).T
