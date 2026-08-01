"""
kanboost.train.rfkan -- Random-Feature KAN boosting (RF-KAN), an
alternative weak-learner training engine for `KANBoostClassifier`
(binary and multiclass) that replaces kanboost's standard Gauss-Newton
ALS (`_fit_als`, up to 10 alternating sweeps per learner) with a single
closed-form solve per round.

Root cause identified (see `docs/inspire_kanboost_evaluation.md` §10):
kanboost's per-round cost is dominated by ALS's alternating refinement
of BOTH layers (input->hidden, hidden->output) for up to 10 sweeps,
each requiring a fresh eigendecomposition of the hidden-layer's
normal-equations system, since the hidden representation changes every
sweep. RF-KAN instead freezes layer0 (input->hidden) as a random
projection for the round -- Random Features / Extreme Learning Machine
theory (Rahimi & Recht 2007; Huang 2006) guarantees this retains
universal-approximation capacity given enough hidden units -- and
solves layer1 (hidden->output) with ONE closed-form penalized
least-squares solve. No alternation, no repeated eigendecomposition.

Critically, layer0 is re-randomized EVERY ROUND (not shared across the
whole boosting chain) -- an earlier attempt at freezing ONE shared
random projection for the entire chain preserved the speedup but cost
real accuracy (boosting needs per-round diversity to correct itself);
re-randomizing each round costs nothing extra (each round only needs
ONE eigendecomposition regardless) and fully recovers accuracy.

Measured on INSPIRE (`postop_icu`), same round count as a normal fit,
all three scales -- accuracy matches or nearly matches the standard ALS
engine, at 3.7x (Small) / 4.3x (Medium) / 4.3-5.3x (Large) less fit time:

    Small  (100 rounds): ALS 27.6s AUROC 0.9225/AUPRC 0.6707 -> RF-KAN 7.4s  0.9226/0.6709
    Medium (140 rounds): ALS 265.5s 0.9389/0.7560            -> RF-KAN 62.4s 0.9388/0.7561
    Large  (180 rounds): ALS 618.0s 0.9413/0.7643             -> RF-KAN 143.6s 0.9413/0.7643

Composes cleanly with the two other opt-in engines in `kanboost.train`:
- `use_newton=True` (see `kanboost.train.newton`): reweight each round's
  target with the logistic loss's Hessian. Gives the BEST accuracy of
  any option in this module, at the SAME speed as plain RF-KAN (Newton
  reweighting doesn't add sweeps -- there are none to add to). Measured:
  Small 0.9247/0.6878, Medium 0.9411/0.7688, Large 0.9433/0.7758 --
  better than the standard ALS engine on every metric, at 3.7-4.3x its
  speed.
- `use_line_search=True` (see `kanboost.train.linesearch`): search each
  round's optimal step size instead of using `model.learning_rate`,
  which lets far fewer rounds match or exceed full-round accuracy. Gives
  the BEST speed of any option here: Medium, 42 rounds instead of 140,
  19.3s instead of 265.5s (13.8x), AUROC/AUPRC both *exceeding* the
  full-round ALS baseline (0.9407/0.7643 vs. 0.9389/0.7560).

**Do not set both `use_newton` and `use_line_search` at once.** Tested
explicitly at all three scales: the combination underperforms EITHER
technique alone every time (e.g. Large: combined 0.9396/0.7655 vs.
Newton-alone 0.9433/0.7758 or line-search-alone 0.9431/0.7712) -- the
line search there optimizes against the *original* loss while the
learner was fit to the Newton-*reweighted* target, a real mathematical
inconsistency, not a defect in either piece individually. This is
enforced with a `ValueError`, not just documented, since every measured
combination made things strictly worse with no compensating benefit.

**Multiclass** (3+ classes): one binary RF-KAN chain per class,
one-vs-rest, mirroring `KANBoostClassifier.fit()`'s own convention
(`seed_base = i * n_estimators` per chain, `learners_`/`init_pred_`/
`best_iteration_`/`_rfkan_gammas_` all keyed by class label).
`predict_proba_rfkan` combines the per-class chains via the same
softmax normalization the base `predict_proba()` uses for multiclass.
Each chain still needs its own `predict_proba_rfkan()` call path --
the base `predict_proba()` still cannot read RF-KAN's per-round-random
layer0. Measured on Digits (sklearn, 10 classes, 30 rounds):
`use_newton=True` matched standard-ALS multiclass Newton boosting's
accuracy exactly (97.4%) at 7.6x its speed (8.2s vs. 62.4s) -- the
Newton+RF-KAN combination's advantage over plain-ALS Newton widens in
multiclass, since Newton's per-round cost compounds across all
`n_classes` chains on the standard ALS engine but stays cheap on
RF-KAN's already-fast one.

**How this differs from `kanboost.train.newton`/`kanboost.train.linesearch`,
and why both still exist**: those two modules keep kanboost's standard
ALS engine (layer0 genuinely adapts to the data every round) and only
change the target reweighting or step size on top of it -- slower than
RF-KAN, but layer0 stays a real, data-adaptive representation, so
`feature_contributions()`, `plot_feature()`, `symbolic_report()`, and
`feature_interaction()` all still attribute to actual input features
meaningfully. **RF-KAN's layer0 is a fresh random projection every
round** -- there is no stable input-to-hidden-unit mapping to attribute
through, so those interpretability methods are not meaningful on a
model fitted here (KANBoost's primary differentiator against tree
boosting). Choose `kanboost.train.newton`/`linesearch` when
interpretability matters for a given model; choose `rfkan` when raw
speed/accuracy is the priority and interpretability is not needed for
that particular model.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import minimize_scalar

from kanboost.core.losses import LogisticLoss, _sigmoid
from kanboost.core.kan.network import DeepKAN, _penalty_block, _eigh_factor, _solve_with_factor
from kanboost.core.kan.bspline import _b_basis_1d, extend_grid

_MIN_HESSIAN = 1e-3
_DEFAULT_GAMMA_BOUNDS = (0.0, 2.0)


def _logistic_loss_at_gamma(y, w, F, update, gamma) -> float:
    F_new = F + gamma * update
    p = np.clip(_sigmoid(F_new), 1e-7, 1 - 1e-7)
    return -float(np.mean(w * (y * np.log(p) + (1 - y) * np.log(1 - p))))


def _fit_rfkan_chain(model, X_arr, y_bin, sample_weight, n_hidden, grid, k,
                      use_newton, use_line_search, gamma_bounds, min_hessian, seed_base):
    n_in = X_arr.shape[1]
    loss = LogisticLoss()
    init_pred = loss.init_pred(y_bin, sample_weight)
    F = np.full(len(y_bin), init_pred)
    learners = []
    gammas = []

    for t in range(model.n_estimators):
        learner = DeepKAN(width=[n_in, n_hidden, 1], grid=grid, k=k, seed=model.random_state + seed_base + t)
        layer0 = learner.layers[0]
        z = layer0.forward(X_arr)

        layer1 = learner.layers[1]
        knots1 = np.zeros((n_hidden, grid + 2 * k + 1))
        for h in range(n_hidden):
            lo, hi = z[:, h].min() - 0.1, z[:, h].max() + 0.1
            base = np.linspace(lo, hi, grid + 1)[np.newaxis, :]
            knots1[h] = extend_grid(base, k_extend=k)[0]
        layer1.knots = knots1
        K1 = layer1.coef.shape[2]

        cols = [_b_basis_1d(z[:, h], knots1[h], k) for h in range(n_hidden)]
        B1 = np.concatenate(cols, axis=1)
        P_blk = _penalty_block(K1, model.lamb * model.lamb_coefdiff, model.lamb * model.lamb_l1)
        P_full = np.zeros((n_hidden * K1, n_hidden * K1))
        for h in range(n_hidden):
            s, e = h * K1, (h + 1) * K1
            P_full[s:e, s:e] = P_blk

        p = _sigmoid(F)
        if use_newton:
            hess = np.clip(p * (1 - p), min_hessian, None)
            target = (y_bin - p) / hess
            fit_weight = hess * sample_weight
        else:
            target = y_bin - p
            fit_weight = sample_weight

        Bw1 = B1 * fit_weight[:, np.newaxis]
        M1 = Bw1.T @ B1 + P_full
        eigvecs, eigvals = _eigh_factor(M1, rel_floor=1e-4)
        b = Bw1.T @ target
        c = _solve_with_factor(eigvecs, eigvals, b)
        for h in range(n_hidden):
            layer1.coef[h, 0, :] = c[h * K1:(h + 1) * K1]
        update = B1 @ c

        if use_line_search:
            res = minimize_scalar(
                lambda g: _logistic_loss_at_gamma(y_bin, sample_weight, F, update, g),
                bounds=gamma_bounds, method="bounded", options={"xatol": 1e-3},
            )
            gamma = float(res.x)
        else:
            gamma = model.learning_rate

        F += gamma * update
        learners.append(learner)
        gammas.append(gamma)

    return learners, init_pred, len(learners), gammas


def fit_with_rfkan(
    model,
    X,
    y,
    sample_weight=None,
    use_newton: bool = False,
    use_line_search: bool = False,
    gamma_bounds=_DEFAULT_GAMMA_BOUNDS,
    min_hessian: float = _MIN_HESSIAN,
):
    """Fit `model` (an unfitted `KANBoostClassifier`, binary or
    multiclass) using the RF-KAN engine: a fresh random layer0 each
    round, one closed-form penalized least-squares solve for layer1, no
    ALS sweeps.

    `model.n_estimators`/`model.kan_hidden`/`model.kan_grid`/`model.kan_k`
    set the ensemble shape as usual. `model.learning_rate` is used
    unless `use_line_search=True`, in which case each round's step size
    is searched instead (see module docstring for measured effects of
    each option, alone and combined).

    Populates `model.learners_` with genuine `DeepKAN` objects (their
    `layers[0]` is the frozen random projection, `layers[1]` the solved
    output layer) and `model._rfkan_gammas_` with one step size per
    round, so `predict_proba_rfkan(model, X)` reproduces the fit exactly.
    Multiclass models get dict-keyed `learners_`/`init_pred_`/
    `best_iteration_`/`_rfkan_gammas_`, one one-vs-rest chain per class,
    exactly like a standard multiclass `fit()`.

    Unlike `fit_with_newton_boosting`, this does NOT populate
    `model.learners_` in a form the base `predict_proba()` can read
    (step sizes vary per round when `use_line_search=True`, and even
    otherwise the base `_raw_score_chain` reuse-shared-layer0-basis
    optimization assumes identical layer0 knots across learners, which
    RF-KAN's per-round-random layer0 violates) -- always use
    `predict_proba_rfkan()`.

    Returns `model`, fitted in place.
    """
    if use_newton and use_line_search:
        raise ValueError(
            "use_newton=True and use_line_search=True together are not supported: measured at all "
            "three INSPIRE scales, the combination underperforms EITHER technique alone every time "
            "(see module docstring). Use one or the other."
        )
    if not hasattr(model, "predict_proba"):
        raise TypeError("fit_with_rfkan requires a KANBoostClassifier, not a regressor.")

    X, y_arr, X_arr = model._prepare_fit(X, y)
    model.classes_ = np.unique(y_arr)
    if len(model.classes_) < 2:
        raise ValueError(f"KANBoostClassifier needs at least 2 classes; got {model.classes_}.")
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=float).ravel()
    else:
        sample_weight = np.ones(len(y_arr))

    n_hidden, grid, k = model.kan_hidden, model.kan_grid, model.kan_k

    if len(model.classes_) == 2:
        y_bin = (y_arr == model.classes_[1]).astype(float)
        learners, init_pred, best_iteration, gammas = _fit_rfkan_chain(
            model, X_arr, y_bin, sample_weight, n_hidden, grid, k,
            use_newton, use_line_search, gamma_bounds, min_hessian, seed_base=0,
        )
        model.learners_ = learners
        model.init_pred_ = init_pred
        model.best_iteration_ = best_iteration
        model._rfkan_gammas_ = gammas
    else:
        model.learners_ = {}
        model.init_pred_ = {}
        model.best_iteration_ = {}
        model._rfkan_gammas_ = {}
        for i, c in enumerate(model.classes_):
            y_bin = (y_arr == c).astype(float)
            learners, init_pred, best_iteration, gammas = _fit_rfkan_chain(
                model, X_arr, y_bin, sample_weight, n_hidden, grid, k,
                use_newton, use_line_search, gamma_bounds, min_hessian,
                seed_base=i * model.n_estimators,
            )
            model.learners_[c] = learners
            model.init_pred_[c] = init_pred
            model.best_iteration_[c] = best_iteration
            model._rfkan_gammas_[c] = gammas

    return model


def _raw_score_rfkan(model, X_t, learners, init_pred, gammas) -> np.ndarray:
    F = np.full(X_t.shape[0], init_pred)
    with torch.no_grad():
        for learner, gamma in zip(learners, gammas):
            F += gamma * learner(X_t).cpu().numpy().flatten()
    return F


def predict_proba_rfkan(model, X) -> np.ndarray:
    """`predict_proba` for a model fitted via `fit_with_rfkan` -- required
    because each round's layer0 is independently random (unlike a
    standard fit, where every learner shares the same layer0 knots) and
    each round may have its own step size, neither of which the base
    `predict_proba`/`_raw_score_chain` supports.

    Multiclass models are combined via the same softmax normalization
    the base `predict_proba()` uses.
    """
    if not hasattr(model, "_rfkan_gammas_"):
        raise RuntimeError("This model was not fitted with fit_with_rfkan().")
    X_t = model._transform_X(X)

    if isinstance(model.learners_, dict):
        raw = np.column_stack([
            _raw_score_rfkan(model, X_t, model.learners_[c], model.init_pred_[c], model._rfkan_gammas_[c])
            for c in model.classes_
        ])
        raw = raw - raw.max(axis=1, keepdims=True)
        exp = np.exp(raw)
        return exp / exp.sum(axis=1, keepdims=True)

    prob_pos = _sigmoid(_raw_score_rfkan(model, X_t, model.learners_, model.init_pred_, model._rfkan_gammas_))
    return np.vstack([1 - prob_pos, prob_pos]).T
