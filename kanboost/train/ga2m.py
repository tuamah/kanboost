"""
kanboost.train.ga2m -- GA2M-style RF-KAN: a weak-learner engine that
recovers native interpretability while keeping RF-KAN's speed and (with
line search) exceeding its accuracy.

Motivation: RF-KAN (`kanboost.train.rfkan`) freezes layer0 as a DENSE
random mixture of ALL input features per hidden unit -- fast and
accurate, but no hidden unit can be attributed to a specific feature
(kanboost's primary differentiator against tree boosting is lost).
Restricting layer0 to per-feature-only units (no mixing at all)
restores attribution but consistently COSTS real AUPRC (~0.012-0.033,
confirmed at all three INSPIRE scales) -- the model can no longer
represent feature *interactions* at all, and this dataset's interaction
effects (e.g. `department` x `icd10_pcs`) carry real signal.

This module keeps interpretability AND recovers (and then exceeds) the
lost accuracy, following the same idea GA2M / Explainable Boosting
Machines (Microsoft InterpretML) use: model main effects (one hidden
unit per feature) PLUS explicit PAIRWISE interaction terms (one hidden
unit per feature PAIR) -- both individually attributable, unlike dense
mixing's opaque high-order combination. Rather than testing all
C(n,2) pairs every round (expensive), a random SUBSET of pairs is
sampled fresh each boosting round -- diversity across rounds covers the
pair space over the course of training, the same principle that already
lets RF-KAN re-randomize its whole layer0 every round safely.

A hidden unit fed by exactly two features j, j' computes
z_h = g_j(x_j) + g_j'(x_j') at layer0, then the output layer applies its
own spline g1_h(z_h) -- a genuine (single-ridge-function) interaction
term, not separable into g_j(x_j)+g_j'(x_j') in general, since g1_h is
itself nonlinear.

Measured on INSPIRE, combined with `fit_with_line_search`'s per-round
step-size search (the configuration that won every comparison run in
this evaluation -- see `docs/inspire_kanboost_evaluation.md` §12),
GA2M+line-search beat every other engine tested this session --
including RF-KAN+Newton (previously the best) -- on BOTH AUROC and
AUPRC, at every scale, while keeping full per-feature/per-pair
attribution:

    Small  (30 rounds):  AUROC 0.9226->0.9283, AUPRC 0.6709->0.6883, 2.3s
    Medium (42 rounds):  AUROC 0.9388->0.9477, AUPRC 0.7561->0.7797, 19.6s
    Large  (54 rounds):  AUROC 0.9413->0.9505, AUPRC 0.7643->0.7918, 45.4s

(baseline column above is the shipped dense-random RF-KAN at full round
count; GA2M+line-search uses ~30% of the rounds and is faster, not just
more accurate.)

**Without line search, GA2M underperforms dense RF-KAN on AUPRC** at
every scale tested (a real, confirmed trade for interpretability) --
`use_line_search=True` is the default here specifically because it is
what closes that gap (and then some); disabling it is not recommended
based on current evidence.

**`use_newton=True`, unlike `kanboost.train.rfkan`'s dense engine, is
NOT clearly harmful here -- but it isn't a clear win either.** Combined
with line search on this GA2M structure, its effect was inconsistent:
slightly worse at Small (AUROC 0.9283->0.9241, AUPRC 0.6883->0.6724),
roughly a wash at Medium, slightly better at Large (AUROC
0.9505->0.9506, AUPRC 0.7918->0.7928). Default is `False` (the
configuration with the strongest, most consistent evidence); try
`True` on your own data and compare, rather than assuming either
direction.

**A stricter/faster variant exists** (independent per-hidden-unit
solve, ignoring cross-unit correlations, instead of one joint solve)
that is ~15% faster with only a marginally lower AUROC/AUPRC -- but it
is measurably LESS accurate for attribution specifically: the joint
solve used here correctly partitions credit between overlapping units
(e.g. a feature's main effect and an interaction term involving that
same feature); the independent solve does not, and can double-count
shared signal between them. Not implemented in this module -- use the
joint solve here whenever the interpretation itself will be trusted,
not just the predictions. (For reference: that independent-solve
variant combined with `use_newton=True` was *consistently*, if
modestly, better than without Newton at all three scales tested --
unlike the joint-solve case above -- even slightly beating this
module's own numbers on AUPRC at Large scale. Not shipped here, since
it requires the less-precise independent solve to begin with.)
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import minimize_scalar

from kanboost.core.losses import LogisticLoss, _sigmoid
from kanboost.core.kan.network import _make_knots, _penalty_block, _eigh_factor, _solve_with_factor
from kanboost.core.kan.layer import KANLayer
from kanboost.core.kan.bspline import _b_basis_1d, extend_grid

_DEFAULT_GAMMA_BOUNDS = (0.0, 2.0)


def _logistic_loss_at_gamma(y, w, F, update, gamma) -> float:
    F_new = F + gamma * update
    p = np.clip(_sigmoid(F_new), 1e-7, 1 - 1e-7)
    return -float(np.mean(w * (y * np.log(p) + (1 - y) * np.log(1 - p))))


def _build_layer0_ga2m(n_in, n_pairs_per_round, grid, k, seed):
    """n_in main-effect units (one per feature, index == feature index)
    followed by n_pairs_per_round interaction units (each fed by a
    randomly-sampled distinct feature pair)."""
    rng = np.random.RandomState(seed)
    n_hidden = n_in + n_pairs_per_round
    knots0 = _make_knots(n_in, grid, k)
    layer0 = KANLayer(n_in, n_hidden, knots0, k)
    K0 = layer0.coef.shape[2]
    layer0.coef[:] = 0.0

    for j in range(n_in):
        layer0.coef[j, j, :] = rng.randn(K0) * 0.5

    all_pairs = [(a, b) for a in range(n_in) for b in range(a + 1, n_in)]
    rng.shuffle(all_pairs)
    n_pairs_per_round = min(n_pairs_per_round, len(all_pairs))
    chosen_pairs = all_pairs[:n_pairs_per_round]
    for idx, (a, b) in enumerate(chosen_pairs):
        h = n_in + idx
        layer0.coef[a, h, :] = rng.randn(K0) * 0.5
        layer0.coef[b, h, :] = rng.randn(K0) * 0.5

    return layer0, chosen_pairs


def _fit_ga2m_chain(model, X_arr, y_bin, sample_weight, n_pairs_per_round, grid, k,
                     use_line_search, gamma_bounds, seed_base, use_newton=False, min_hessian=1e-3):
    n_in = X_arr.shape[1]
    loss = LogisticLoss()
    init_pred = loss.init_pred(y_bin, sample_weight)
    F = np.full(len(y_bin), init_pred)
    rounds = []  # (layer0, knots1, coefs, n_hidden, K1, gamma, pairs)

    for t in range(model.n_estimators):
        layer0, pairs = _build_layer0_ga2m(n_in, n_pairs_per_round, grid, k, seed=model.random_state + seed_base + t)
        n_hidden = n_in + len(pairs)
        z = layer0.forward(X_arr)

        knots1 = np.zeros((n_hidden, grid + 2 * k + 1))
        for h in range(n_hidden):
            lo, hi = z[:, h].min() - 0.1, z[:, h].max() + 0.1
            base = np.linspace(lo, hi, grid + 1)[np.newaxis, :]
            knots1[h] = extend_grid(base, k_extend=k)[0]
        K1 = grid + k

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
        rounds.append((layer0, knots1, c, n_hidden, K1, gamma, pairs))

    return rounds, init_pred


def fit_with_ga2m(
    model,
    X,
    y,
    sample_weight=None,
    n_pairs_per_round: int = 10,
    use_line_search: bool = True,
    gamma_bounds=_DEFAULT_GAMMA_BOUNDS,
    use_newton: bool = False,
    min_hessian: float = 1e-3,
):
    """Fit `model` (an unfitted binary `KANBoostClassifier`) using the
    GA2M-RF-KAN engine: one random-projection hidden unit per input
    feature (main effect) plus `n_pairs_per_round` randomly-sampled
    pairwise-interaction units, refreshed every round; solved jointly
    via one closed-form penalized least squares per round (no ALS
    sweeps). `use_line_search=True` (default, and the configuration
    that was actually validated -- see module docstring) searches each
    round's optimal step size instead of using `model.learning_rate`.

    `n_pairs_per_round`: how many of the `C(n_features, 2)` possible
    feature pairs to include as interaction units each round. More
    pairs costs more per-round compute (bigger hidden layer) but covers
    more of the interaction space faster across rounds; fewer pairs is
    cheaper but relies on more rounds for the same coverage. 10 was the
    value used in the measurements quoted in the module docstring for a
    ~13-feature INSPIRE encoding; scale roughly with `model.n_estimators`
    and your feature count if very different.

    `use_newton`: reweight each round's target with the logistic loss's
    Hessian (see `kanboost.train.newton`). Default `False` -- unlike
    `fit_with_rfkan`, where this is a clear win, its effect here was
    inconsistent when measured (see module docstring); try both on your
    own data.

    Populates `model.learners_` with (layer0, knots1, coefs, ...) round
    records -- always use `predict_proba_ga2m()`, not the base
    `predict_proba()`.

    Returns `model`, fitted in place. Multiclass is not implemented.
    """
    if not hasattr(model, "predict_proba"):
        raise TypeError("fit_with_ga2m requires a KANBoostClassifier, not a regressor.")

    X, y_arr, X_arr = model._prepare_fit(X, y)
    model.classes_ = np.unique(y_arr)
    if len(model.classes_) != 2:
        raise ValueError(
            f"fit_with_ga2m only supports binary classification; got {len(model.classes_)} classes. "
            "Multiclass is not implemented."
        )
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=float).ravel()
    else:
        sample_weight = np.ones(len(y_arr))

    y_bin = (y_arr == model.classes_[1]).astype(float)
    grid, k = model.kan_grid, model.kan_k

    rounds, init_pred = _fit_ga2m_chain(
        model, X_arr, y_bin, sample_weight, n_pairs_per_round, grid, k,
        use_line_search, gamma_bounds, seed_base=0, use_newton=use_newton, min_hessian=min_hessian,
    )

    model.learners_ = rounds
    model.init_pred_ = init_pred
    model.best_iteration_ = len(rounds)
    model._ga2m_feature_names_ = (
        model.preprocessor_.transformed_feature_names()
        if hasattr(model.preprocessor_, "transformed_feature_names") else None
    )
    return model


def predict_proba_ga2m(model, X) -> np.ndarray:
    """`predict_proba` for a model fitted via `fit_with_ga2m`."""
    if not hasattr(model, "_ga2m_feature_names_"):
        raise RuntimeError("This model was not fitted with fit_with_ga2m().")
    X_t = model._transform_X(X)
    x_np = X_t.cpu().numpy() if hasattr(X_t, "cpu") else np.asarray(X_t)
    F = np.full(x_np.shape[0], model.init_pred_)
    k = model.kan_k
    for layer0, knots1, coefs, n_hidden, K1, gamma, pairs in model.learners_:
        z = layer0.forward(x_np)
        cols = [_b_basis_1d(z[:, h], knots1[h], k) for h in range(n_hidden)]
        B1 = np.concatenate(cols, axis=1)
        F += gamma * (B1 @ coefs)
    prob_pos = _sigmoid(F)
    return np.vstack([1 - prob_pos, prob_pos]).T


def main_effect_contributions(model) -> dict:
    """Total accumulated contribution of each individual feature's main
    effect, summed over every boosting round (main-effect units appear
    in every round, one per feature, so this is a fair total -- unlike
    pairwise interactions below, coverage is uniform).

    Returns `{feature_name_or_index: total_coefficient_L2_norm}`,
    sorted descending. Feature names come from
    `model.preprocessor_.transformed_feature_names()` if available,
    else integer indices.
    """
    if not hasattr(model, "_ga2m_feature_names_"):
        raise RuntimeError("This model was not fitted with fit_with_ga2m().")
    names = model._ga2m_feature_names_
    layer0_ref, _, _, _, K1, _, _ = model.learners_[0]
    n_in = layer0_ref.coef.shape[0]
    totals = np.zeros(n_in)
    for layer0, knots1, coefs, n_hidden, K1, gamma, pairs in model.learners_:
        for j in range(n_in):
            totals[j] += abs(gamma) * np.linalg.norm(coefs[j * K1:(j + 1) * K1])
    labels = names if names is not None and len(names) == n_in else list(range(n_in))
    pairs_sorted = sorted(zip(labels, totals), key=lambda kv: -kv[1])
    return {label: float(val) for label, val in pairs_sorted}


def pairwise_interaction_contributions(model, top_k: int = 10) -> dict:
    """Total accumulated contribution of each feature PAIR's interaction
    term, summed over only the rounds where that pair happened to be
    sampled (pairs not sampled in a given round contribute 0 for it;
    pairs never sampled across the whole fit are absent from the
    result entirely -- this is raw accumulated contribution, not
    normalized by how often each pair was sampled, so treat it as "how
    much this pair mattered given how often boosting revisited it", not
    a pure per-pair effect size independent of sampling luck).

    Returns the top-`top_k` pairs by total contribution, as
    `{(feature_a, feature_b): total_coefficient_L2_norm}`.
    """
    if not hasattr(model, "_ga2m_feature_names_"):
        raise RuntimeError("This model was not fitted with fit_with_ga2m().")
    names = model._ga2m_feature_names_
    layer0_ref, _, _, _, _, _, _ = model.learners_[0]
    n_in = layer0_ref.coef.shape[0]
    labels = names if names is not None and len(names) == n_in else list(range(n_in))

    totals: dict = {}
    for layer0, knots1, coefs, n_hidden, K1, gamma, pairs in model.learners_:
        for idx, (a, b) in enumerate(pairs):
            h = n_in + idx
            key = (labels[a], labels[b])
            totals[key] = totals.get(key, 0.0) + abs(gamma) * np.linalg.norm(coefs[h * K1:(h + 1) * K1])

    return dict(sorted(totals.items(), key=lambda kv: -kv[1])[:top_k])
