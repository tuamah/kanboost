"""
kanboost.interactions -- Friedman's H-statistic (Friedman & Popescu,
2008) for pairwise feature interaction strength, computed from partial
dependence.

Model-agnostic by design: works on ANY fitted estimator with
`predict_proba` or `predict` (not just KANBoost) via
`sklearn.inspection.partial_dependence`. This is deliberate -- the
question this answers is "does the underlying data/target relationship
have real interactions the model is capturing", not "does this
specific model's architecture support them". Running this on a
`gam=True` KANBoostClassifier will show near-zero H for every pair (the
architecture is a strict `F(x) = c + sum_j g_j(x_j)`, so it literally
cannot represent an interaction, regardless of whether one exists in
the data) -- to check whether `gam=True` is leaving real interaction
signal on the table, run this on a flexible model fit to the same data
instead (`gam=False`, a higher `kan_hidden`, or a tree ensemble).

Caveat verified empirically during development: tree ensembles
(RandomForest/XGBoost) can show a *spuriously* elevated H (0.5+) for
feature pairs with NO true interaction, purely from partial-dependence
estimation noise on a moderate sample -- trees split on combinations of
features even when fit to an additive target, and 2-way PD surfaces
from a small forest are not smooth. Sanity-checked against a
hand-written model with a known-exact formula (`a*b + 0.1*c`): H(a,b)
came out to 0.9996 (true interaction) and H(a,c)/H(b,c) to ~0.02 (no
interaction) -- so the *formula* here is correct; a real model's H
values should be read with that estimation-noise caveat in mind,
especially with few samples or a small ensemble.
"""

from __future__ import annotations

import itertools

import numpy as np


def friedman_h(model, X, features=None, grid_resolution: int = 20,
                sample_size: int = 300, random_state: int = 0) -> dict:
    """Friedman's H-statistic for every pair of features (or the given
    `features` subset), measuring how much of each pair's joint partial
    dependence is *not* explained by the sum of their individual
    partial dependences:

        H_jk^2 = sum_i [PD_jk(x_j^i, x_k^i) - PD_j(x_j^i) - PD_k(x_k^i)]^2
                 / sum_i PD_jk(x_j^i, x_k^i)^2

    (all three PD functions centered to mean zero over the sample
    first, per Friedman & Popescu's own convention, and evaluated at
    each sampled row's *own* feature values, not on an arbitrary grid).
    H close to 0 means the pair's joint effect on the model's output is
    (near-)additive; close to 1 means it's almost entirely interaction.

    `model` needs `predict_proba` (binary classifier -- uses class 1's
    probability) or `predict` (regressor / anything else). `X` must be
    a pandas DataFrame with named columns.

    `sample_size` subsamples `X` before computing partial dependence
    (Monte Carlo averaging over the complement features is the
    dominant cost) -- the full row count isn't needed for a stable
    H-statistic estimate. `grid_resolution` controls how finely each
    feature's partial dependence is evaluated before interpolating back
    onto the sample's own values.

    Returns `{"pairwise": {(f_j, f_k): H_jk}, "ranked": [(f_j, f_k, H_jk), ...]}`
    (`ranked` sorted descending by H -- the pairs to look at first).
    Requires `scipy` and `scikit-learn` (both core dependencies).

    For a **multiclass** classifier, `result["average"][0]` (what this
    function reads) is class 0's probability, not a "positive class" in
    any meaningful sense -- interactions are measured against an
    arbitrary reference class. For a genuinely per-class picture, call
    this once per class using a small wrapper whose `predict_proba`
    returns just that class's column.
    """
    import pandas as pd
    from scipy.interpolate import RegularGridInterpolator
    from sklearn.inspection import partial_dependence

    if not isinstance(X, pd.DataFrame):
        raise TypeError("friedman_h() requires X as a pandas DataFrame with named columns")

    feature_names = list(features) if features is not None else list(X.columns)
    if len(feature_names) < 2:
        raise ValueError("friedman_h() needs at least 2 features to form a pair")

    rng = np.random.RandomState(random_state)
    if len(X) > sample_size:
        idx = rng.choice(len(X), size=sample_size, replace=False)
        X_sample = X.iloc[idx].reset_index(drop=True)
    else:
        X_sample = X.reset_index(drop=True)

    # sklearn's response_method only accepts "predict_proba",
    # "decision_function", or "auto" -- "auto" resolves to predict()
    # for a regressor (or predict_proba() if that's all a classifier
    # without predict_proba exposes).
    response_method = "predict_proba" if hasattr(model, "predict_proba") else "auto"

    def pd_1way(feature):
        result = partial_dependence(
            model, X_sample, [feature], kind="average",
            grid_resolution=grid_resolution, response_method=response_method,
        )
        return result["grid_values"][0], np.asarray(result["average"][0])

    def pd_2way(f_j, f_k):
        result = partial_dependence(
            model, X_sample, [f_j, f_k], kind="average",
            grid_resolution=grid_resolution, response_method=response_method,
        )
        return result["grid_values"][0], result["grid_values"][1], np.asarray(result["average"][0])

    # Cache each feature's 1-way PD once -- reused across every pair it
    # appears in, rather than recomputing per pair.
    pd_1way_cache = {f: pd_1way(f) for f in feature_names}

    pairwise = {}
    for f_j, f_k in itertools.combinations(feature_names, 2):
        grid_j, pd_j_vals = pd_1way_cache[f_j]
        grid_k, pd_k_vals = pd_1way_cache[f_k]
        grid_j2, grid_k2, pd_jk_vals = pd_2way(f_j, f_k)

        xj = X_sample[f_j].to_numpy(dtype=np.float64)
        xk = X_sample[f_k].to_numpy(dtype=np.float64)

        pd_j_at_x = np.interp(xj, grid_j, pd_j_vals)  # np.interp clamps outside the grid range
        pd_k_at_x = np.interp(xk, grid_k, pd_k_vals)

        # sklearn's default percentiles=(0.05, 0.95) leave ~10% of
        # sample values outside the PD grid. np.interp above clamps to
        # the grid edge for those; RegularGridInterpolator would
        # instead *extrapolate* linearly if given the same out-of-range
        # points, injecting a spurious mismatch between the 1-way and
        # 2-way terms that looks like "interaction" but is really just
        # inconsistent boundary handling. Clip to match np.interp's
        # clamping behavior so both PD estimates agree at the edges.
        xj_clipped = np.clip(xj, grid_j2.min(), grid_j2.max())
        xk_clipped = np.clip(xk, grid_k2.min(), grid_k2.max())
        interp = RegularGridInterpolator((grid_j2, grid_k2), pd_jk_vals, bounds_error=False, fill_value=None)
        pd_jk_at_x = interp(np.column_stack([xj_clipped, xk_clipped]))

        pd_j_c = pd_j_at_x - pd_j_at_x.mean()
        pd_k_c = pd_k_at_x - pd_k_at_x.mean()
        pd_jk_c = pd_jk_at_x - pd_jk_at_x.mean()

        numerator = np.sum((pd_jk_c - pd_j_c - pd_k_c) ** 2)
        denominator = np.sum(pd_jk_c ** 2)
        h = float(np.sqrt(numerator / denominator)) if denominator > 1e-12 else 0.0
        pairwise[(f_j, f_k)] = h

    ranked = sorted(((j, k, h) for (j, k), h in pairwise.items()), key=lambda t: t[2], reverse=True)
    return {"pairwise": pairwise, "ranked": ranked}


def check_additive_sufficiency(model, X, y, top_n: int = 6, threshold: float = 0.1,
                                grid_resolution: int = 8, sample_size: int = 150,
                                random_state: int = 0) -> dict:
    """Is `model`'s `gam=True` additive assumption actually adequate for
    *this* data, or is it leaving real pairwise interaction signal on
    the table?

    `friedman_h()` on a `gam=True` model alone can't answer this -- the
    architecture is *forced* additive, so its own H is always near-zero
    noise regardless of what the data actually looks like (verified:
    0.013-0.040 on breast_cancer). The only way to see what's actually
    being missed is to fit a *flexible* counterpart (same
    hyperparameters, `gam=False`) on the same data and measure its H
    instead -- a real interaction shows up there even though `gam=True`
    can't represent it.

    This does exactly that: refits a `gam=False` version of `model`
    (same class, same other hyperparameters, `random_state`) on `X, y`,
    runs `friedman_h()` on both for the `top_n` most important features
    (by `model.feature_importances_dict()`), and returns a verdict.

    `threshold` is compared against the *flexible* model's H for each
    pair (not `gam=True`'s, which is only there for context/noise-floor
    comparison). The default `0.1` is calibrated against what was
    measured on breast_cancer during development: `gam=True` sits at
    0.01-0.04 (pure noise), a real-but-modest interaction there reached
    0.06-0.10, and the synthetic sanity check (a hand-written function
    with an exact multiplicative interaction) reached 0.99 -- so `0.1`
    sits just above breast_cancer's own (modest) signal, flagging
    clearly stronger interactions than that while not firing on noise.
    Adjust it for your own data's scale if needed; this default is a
    starting point, not a universal cutoff.

    Only meaningful on a model that was actually fit with `gam=True`
    (raises `ValueError` otherwise -- there's nothing being "left out"
    to check if the model wasn't additive to begin with).

    Returns
    -------
    dict with:
        "verdict": "additive_sufficient" or "interactions_detected"
        "threshold": the threshold used
        "features_checked": the top_n feature names
        "pairwise": list of dicts, most-interactive-first --
            `{"feature_j", "feature_k", "h_gam", "h_flexible", "exceeds_threshold"}`
        "flexible_model": the fitted `gam=False` counterpart (for further inspection)
    """
    if not model.get_params().get("gam", False):
        raise ValueError(
            "check_additive_sufficiency() only makes sense on a model fit with "
            "gam=True -- there's no additive assumption to check otherwise."
        )
    if top_n < 2:
        raise ValueError(f"top_n must be >= 2 (need at least a pair); got {top_n}")

    importances = model.feature_importances_dict()
    # feature_importances_dict() can include transformed names not
    # present in raw X (e.g. a "<col>_missing" indicator, or a
    # categorical column reordered by encoding) -- friedman_h() needs
    # actual X columns, so filter to those first rather than crashing
    # inside partial_dependence() with a bare KeyError on a name X
    # doesn't have.
    top_features = [f for f in importances if f in X.columns][:top_n]
    if len(top_features) < 2:
        raise ValueError(
            "fewer than 2 of model's important features are present as raw columns "
            "in X (feature_importances_dict() can include transformed names, e.g. "
            "'<col>_missing' indicators, that aren't in the original X) -- pass "
            "`features=` explicitly to friedman_h() with real X column names instead."
        )

    flexible_params = dict(model.get_params())
    flexible_params["gam"] = False
    flexible_params["random_state"] = random_state
    flexible_model = type(model)(**flexible_params)
    flexible_model.fit(X, y)

    h_gam = friedman_h(model, X, features=top_features,
                        grid_resolution=grid_resolution, sample_size=sample_size, random_state=random_state)
    h_flex = friedman_h(flexible_model, X, features=top_features,
                         grid_resolution=grid_resolution, sample_size=sample_size, random_state=random_state)

    pairwise = []
    for (f_j, f_k), h_flexible in h_flex["pairwise"].items():
        pairwise.append({
            "feature_j": f_j,
            "feature_k": f_k,
            "h_gam": h_gam["pairwise"][(f_j, f_k)],
            "h_flexible": h_flexible,
            "exceeds_threshold": h_flexible > threshold,
        })
    pairwise.sort(key=lambda row: row["h_flexible"], reverse=True)

    verdict = "interactions_detected" if any(row["exceeds_threshold"] for row in pairwise) else "additive_sufficient"

    return {
        "verdict": verdict,
        "threshold": threshold,
        "features_checked": top_features,
        "pairwise": pairwise,
        "flexible_model": flexible_model,
    }
