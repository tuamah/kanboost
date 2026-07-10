"""
kanboost.symbolic -- fidelity-aware symbolic formula export for a fitted
`gam=True` KANBoost model: an actual `sympy` expression, LaTeX string,
and standalone (no torch/pykan dependency) numpy predict function --
not just a text summary (see `kanboost.experimental.symbolic_export`
for that lighter-weight report).

Fits one closed-form candidate function per feature to the *exact*
aggregated shape function `g_j`, the same one `symbolic_report`/
`plot_feature` use -- but where they only fit and report a name/R^2,
this reconstructs the actual fitted expression (`c * fun(a*x + b) + d`,
pykan's own `SYMBOLIC_LIB` convention) and combines every feature's term
into one total formula `F(x) = intercept + sum_j term_j(x_j)`.

Reuses `kanboost.editing.consolidate()` for the per-feature curves and
intercept rather than re-deriving curve sampling from scratch: an
earlier version of `consolidate()` had a real bug where naively summing
per-feature partial-dependence probes double-counted every other
feature's zero-point contribution (see `editing.py`'s docstring) --
fitting symbolic terms on top of that same, now-corrected, centered
representation avoids reintroducing that class of bug here.

Fidelity-aware: a feature whose best candidate's R^2 falls below
`min_r2` is *not* forced into a misleading formula -- it's kept as a
numeric (spline-interpolated) term instead, clearly flagged as such in
`fidelity_report()`/`to_latex()`, so the exported formula never silently
claims a closed form that doesn't actually fit.
"""

from __future__ import annotations

import numpy as np
import sympy
import torch

from .editing import consolidate


_CANDIDATES = ["x", "x^2", "x^3", "sin", "cos", "exp", "log", "sqrt", "tanh", "abs"]

# Rough complexity ranking used by `parsimony_margin` (export_symbolic()):
# affine < power/abs < bounded-sigmoidal < periodic/root < unbounded-singular.
# Order-independent of _CANDIDATES' own list order -- only relative rank
# between any two candidates matters.
_CANDIDATE_COMPLEXITY = {
    "x": 1, "abs": 2, "x^2": 2, "x^3": 3, "tanh": 3,
    "sqrt": 4, "sin": 4, "cos": 4, "log": 5, "exp": 5,
}


def export_symbolic(model, min_r2: float = 0.8, resolution: int = 200, grid: int = 10, k: int = 3,
                     features=None, parsimony_margin: float = 0.0, allow_periodic: bool = True):
    """Fit one closed-form symbolic term per feature (falling back to a
    numeric spline term where no candidate reaches `min_r2`) and combine
    them into a `SymbolicModel`: `intercept + sum_j term_j(x_j)`.

    For a multiclass classifier, returns `{class_label: SymbolicModel}`
    (one independent formula per one-vs-rest chain), matching
    `consolidate()`'s own convention.

    `X` is not needed -- like `consolidate()`, this samples each
    feature's shape function directly from the trained ensemble on
    [-1, 1] (the model's internal scaled range), not from data.

    `features`, if given, restricts the (relatively expensive)
    candidate-fitting search to just those feature names -- every other
    feature is kept as a numeric term directly, with no candidates
    tried. Useful when only a handful of features' formulas are
    actually needed (see `explain()`, which uses this to avoid fitting
    candidates for features outside `top_features`).

    `parsimony_margin` (default 0.0, i.e. off -- pure best-R^2 selection,
    the original behavior): when > 0, a more complex candidate (by the
    fixed ranking `x < abs/x^2 < x^3/tanh < sqrt/sin/cos < log/exp`) only
    replaces a simpler one already found if it improves R^2 by more than
    this margin. Guards against, e.g., `sin` or `cos` being selected over
    a plain `x`/`tanh` term purely because it fits an extra 0.001 R^2 by
    matching noise -- see the `min_r2`-vs-`amplitude` warning in
    `fidelity_report()`'s docstring for the same class of pitfall this
    addresses from a different angle.

    `allow_periodic` (default True): set False to drop `sin`/`cos` from
    the candidate library entirely, rather than merely deprioritizing
    them via `parsimony_margin`. Useful for domains (e.g. clinical risk
    scores) where a periodic term is implausible on its face regardless
    of how well it happens to fit a bounded [-1, 1] curve -- `sin`/`cos`/
    `tanh` can look visually near-identical over that narrow a domain
    (observed directly on real data this project benchmarked), so a
    high R^2 for `sin` there is not evidence of a genuinely periodic
    relationship.
    """
    from kan.utils import fit_params, SYMBOLIC_LIB

    candidates = [c for c in _CANDIDATES if c in SYMBOLIC_LIB]
    if not allow_periodic:
        candidates = [c for c in candidates if c not in ("sin", "cos")]
    feature_set = set(features) if features is not None else None

    consolidated = consolidate(model, resolution=resolution, grid=grid, k=k)
    if isinstance(consolidated, dict):
        return {
            c: SymbolicModel._from_editable(gam, candidates, min_r2, feature_set, parsimony_margin)
            for c, gam in consolidated.items()
        }
    return SymbolicModel._from_editable(consolidated, candidates, min_r2, feature_set, parsimony_margin)


def explain(model, top_features: int = 5, symbolic: bool = True, simplify: bool = True, min_r2: float = 0.8) -> list:
    """High-level convenience report: rank features by
    `model.feature_importances_dict()` (already handles multiclass by
    summing importances across every one-vs-rest chain into one
    ranking, same as that method's own docstring), and for the top
    `top_features`, attach each one's symbolic term if `symbolic=True`.

    Returns a list of dicts, most important feature first:
    `{"feature", "importance", "kind", "r2", "amplitude", "formula"}`
    (`"formula"` is a `sympy` expression, or `None` if `symbolic=False`).
    `simplify=True` runs `sympy.simplify()` on each formula (cheap here
    -- these are single-feature terms, not the whole model).

    For a multiclass classifier, `symbolic=True` uses each top
    feature's term from its *first* class's chain (`model.classes_[0]`)
    -- one-vs-rest chains can fit a feature differently per class, so
    this is a representative formula, not a claim that it's identical
    across classes. Call `export_symbolic(model)` directly and index by
    class for a per-class formula.
    """
    importances = model.feature_importances_dict()
    top = list(importances.items())[:top_features]

    # Only fit candidates for the top features actually being reported --
    # export_symbolic's `features=` skips the expensive search for
    # everything else (still exact/spline-numeric there, just unused).
    top_names = [name for name, _ in top]
    sym = export_symbolic(model, min_r2=min_r2, features=top_names) if symbolic else None
    if isinstance(sym, dict):
        sym = sym[model.classes_[0]]

    report = []
    for name, importance in top:
        entry = {"feature": name, "importance": importance}
        if symbolic:
            term = sym.terms[name]
            entry["kind"] = term["kind"]
            entry["r2"] = term["r2"]
            entry["amplitude"] = term["amplitude"]
            entry["formula"] = sym.term_sympy(name, simplify=simplify)
        else:
            entry.update({"kind": None, "r2": None, "amplitude": None, "formula": None})
        report.append(entry)
    return report


def symbolic_summary(model, min_r2: float = 0.8, top_n: int | None = None,
                      min_amplitude: float | None = None, allow_periodic: bool = True) -> dict:
    """One-call convenience report: the model's most valuable features
    (ranked by `amplitude` -- how much each feature's term actually
    moves the prediction, not just how well a candidate happened to fit
    it), each one's individual closed-form equation, and finally the
    whole model's combined equation.

    Unlike `explain()` (which only fits candidates for its
    `top_features` count, for speed, leaving the rest as opaque
    `g_<feature>(x)` placeholders in the full formula), this defaults to
    running the candidate search over *every* feature, so `full_formula`
    only has a placeholder for a feature whose best candidate genuinely
    falls below `min_r2` (check `ranked_terms[i]["kind"]`), not for
    every feature outside some top-N cutoff. Pass `top_n` to restrict
    both the ranking and the candidate search to that many features (by
    `model.feature_importances_dict()`) if the full search is too slow
    for your model -- with `top_n` set, `full_formula` goes back to
    having placeholders for every feature outside that cutoff, same as
    `explain()`.

    `min_amplitude`, if given, drops any term whose `amplitude` falls
    below it from *both* `ranked_terms` and `full_formula` -- a
    low-amplitude term barely moves the prediction regardless of its
    R^2 (see the warning in `fidelity_report()`'s docstring), so leaving
    it in the final equation adds length without adding real signal.
    `full_formula` in that case is rebuilt as `intercept + sum` over only
    the retained terms, not `sym.to_sympy()`'s full model.

    `allow_periodic=False` drops `sin`/`cos` from the candidate library
    entirely -- see `export_symbolic()`'s docstring for why.

    Returns
    -------
    dict with:
        "ranked_terms": list of dicts, most-amplitude-first --
            `{"feature", "kind", "r2", "candidate", "amplitude", "formula"}`
            (`"formula"` is a `sympy` expression for that one feature's term)
        "full_formula": `sympy` expression, `intercept + sum_j term_j(x_j)`
        "full_latex": str, `sym.to_latex()`
        "model": the underlying `SymbolicModel` (for `.predict()`/`.save()`/etc.)

    For a multiclass classifier, uses `model.classes_[0]`'s chain (same
    convention as `explain()`) -- call `export_symbolic(model)` directly
    and index by class for a per-class summary.
    """
    features = None
    if top_n is not None:
        if top_n < 1:
            raise ValueError(f"top_n must be >= 1 (or None for every feature); got {top_n}")
        importances = model.feature_importances_dict()
        features = list(importances.keys())[:top_n]

    sym = export_symbolic(model, min_r2=min_r2, features=features, allow_periodic=allow_periodic)
    if isinstance(sym, dict):
        sym = sym[model.classes_[0]]

    fidelity = sym.fidelity_report()
    candidate_names = features if features is not None else list(fidelity.keys())
    if min_amplitude is not None:
        candidate_names = [n for n in candidate_names if fidelity[n]["amplitude"] >= min_amplitude]
    ranked_names = sorted(candidate_names, key=lambda name: fidelity[name]["amplitude"], reverse=True)

    ranked_terms = [
        {
            "feature": name,
            "kind": fidelity[name]["kind"],
            "r2": fidelity[name]["r2"],
            "candidate": fidelity[name]["candidate"],
            "amplitude": fidelity[name]["amplitude"],
            "formula": sym.term_sympy(name),
        }
        for name in ranked_names
    ]

    if min_amplitude is not None:
        full_formula = sympy.Float(sym.intercept)
        for name in ranked_names:
            full_formula += sym.term_sympy(name)
    else:
        full_formula = sym.to_sympy()

    return {
        "ranked_terms": ranked_terms,
        "full_formula": full_formula,
        "full_latex": sympy.latex(full_formula),
        "model": sym,
    }


def refit_constants(sym: "SymbolicModel", X_scaled, target) -> "SymbolicModel":
    """Jointly re-optimize every symbolic term's `(a, b, c, d)` and the
    intercept against a real target array (e.g. the trained ensemble's
    own raw score -- not the 0/1 label), instead of each term's default
    fit: independently, to that one feature's *isolated* marginal curve
    from `consolidate()`.

    This addresses a real limitation of the default export: two terms
    that each fit their own curve well can still combine into a worse
    joint approximation than a joint refit would give, since nothing
    about the default fit ever looks at more than one feature at a time.
    Only "symbolic" (closed-form) terms are refit; "numeric" (spline
    fallback) terms have no closed-form parameters and are left as-is,
    their fixed contribution held constant during optimization.

    `X_scaled` : array of shape (n_samples, n_features), already through
        `sym.preprocessor.transform(X)` -- see `refit_constants_from_model()`
        for a convenience wrapper that does this (and derives `target`)
        for you from a fitted binary classifier or regressor.
    `target` : array of shape (n_samples,), the value the *combined*
        equation should approximate.

    Returns a new `SymbolicModel` (does not mutate `sym`). Each term's
    `r2`/`amplitude` fields are left as the *original* per-curve fit's
    values -- they describe how well the term matched its own isolated
    curve, not the post-refit joint fit, and are kept only as
    provenance. Compare `formula_fidelity()` before/after refitting to
    judge whether it actually helped.
    """
    from scipy.optimize import minimize
    import copy

    X_scaled = np.asarray(X_scaled, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    symbolic_names = [n for n, t in sym.terms.items() if t["kind"] == "symbolic"]
    if not symbolic_names:
        return sym  # nothing with free parameters to refit

    col_idx = {name: sym.feature_names.index(name) for name in symbolic_names}
    funcs = {name: _NUMPY_FUN[sym.terms[name]["candidate"]] for name in symbolic_names}

    # Numeric terms have no free parameters -- precompute their fixed
    # contribution once, held constant throughout the optimization.
    numeric_contrib = np.zeros(len(X_scaled))
    for name, term in sym.terms.items():
        if term["kind"] != "symbolic":
            j = sym.feature_names.index(name)
            numeric_contrib += np.interp(X_scaled[:, j], term["x_grid"], term["y_grid"])

    x0 = [sym.intercept]
    for name in symbolic_names:
        p = sym.terms[name]["params"]
        x0 += [p["a"], p["b"], p["c"], p["d"]]
    x0 = np.array(x0, dtype=np.float64)

    def predict(params):
        pred = np.full(len(X_scaled), params[0]) + numeric_contrib
        for i, name in enumerate(symbolic_names):
            a, b, c, d = params[1 + 4 * i: 5 + 4 * i]
            x = X_scaled[:, col_idx[name]]
            pred += c * funcs[name](a * x + b) + d
        return pred

    def objective(params):
        return float(np.mean((predict(params) - target) ** 2))

    result = minimize(objective, x0, method="L-BFGS-B")

    new_terms = copy.deepcopy(sym.terms)
    for i, name in enumerate(symbolic_names):
        a, b, c, d = result.x[1 + 4 * i: 5 + 4 * i]
        new_terms[name]["params"] = {"a": float(a), "b": float(b), "c": float(c), "d": float(d)}

    return SymbolicModel(
        sym.feature_names, float(result.x[0]), new_terms,
        preprocessor=sym.preprocessor, feature_names_in_=sym.feature_names_in_,
    )


def refit_constants_from_model(model, sym: "SymbolicModel", X) -> "SymbolicModel":
    """Convenience wrapper around `refit_constants()`: derives the
    target raw score and scaled feature matrix directly from a fitted
    binary `KANBoostClassifier` (target = `logit(predict_proba)`) or a
    `KANBoostRegressor` (target = `predict(X)`), from `X` (raw, unscaled
    input matching what `model.fit()` was called with).

    Not supported for multiclass classifiers here -- `sym` is already
    one class chain's `SymbolicModel` by the time you have it (from
    `export_symbolic(model)[class_label]`), so compute that chain's own
    raw score (`log(p / (1 - p))` for that one-vs-rest probability) and
    call `refit_constants()` directly instead.
    """
    if hasattr(model, "classes_") and len(model.classes_) != 2:
        raise ValueError(
            "refit_constants_from_model() only supports binary classifiers "
            "or regressors -- for a multiclass one-vs-rest chain, compute "
            "that chain's raw score yourself and call refit_constants() directly."
        )

    X_scaled = sym.preprocessor.transform(X)
    if hasattr(model, "classes_"):
        proba = np.clip(model.predict_proba(X)[:, 1], 1e-7, 1 - 1e-7)
        target = np.log(proba / (1 - proba))
    else:
        target = model.predict(X)
    return refit_constants(sym, X_scaled, target)


def formula_fidelity(model, sym: "SymbolicModel", X, y=None) -> dict:
    """How closely the extracted symbolic formula tracks the real
    trained ensemble on `X`.

    Returns `{"max_abs_error", "mean_abs_error"}` always (comparing the
    equation's raw score against the model's own raw score -- `logit`
    of `predict_proba` for a binary classifier, `predict()` otherwise),
    plus `{"auc_model", "auc_equation"}` when `model` is a binary
    classifier *and* `y` (the true labels for `X`) is given -- these let
    you check the equation retains the model's ranking quality, e.g.
    `auc_equation >= auc_model - 0.005`, not just that its raw scores
    are numerically close.
    """
    from sklearn.metrics import roc_auc_score

    X_scaled = sym.preprocessor.transform(X)
    equation_score = sym.predict_scaled(X_scaled)

    is_binary_clf = hasattr(model, "classes_") and len(model.classes_) == 2
    if is_binary_clf:
        proba = np.clip(model.predict_proba(X)[:, 1], 1e-7, 1 - 1e-7)
        model_score = np.log(proba / (1 - proba))
    else:
        model_score = model.predict(X)

    abs_error = np.abs(np.asarray(model_score) - equation_score)
    result = {
        "max_abs_error": float(abs_error.max()),
        "mean_abs_error": float(abs_error.mean()),
    }
    if is_binary_clf and y is not None:
        y_bin = (np.asarray(y) == model.classes_[1]).astype(int)
        result["auc_model"] = float(roc_auc_score(y_bin, model_score))
        result["auc_equation"] = float(roc_auc_score(y_bin, equation_score))
    return result


def stability_across_seeds(build_and_fit, X, y, n_seeds: int = 5, min_r2: float = 0.8,
                            top_n: int | None = None, test_size: float = 0.2, random_state: int = 0,
                            allow_periodic: bool = True):
    """Train `n_seeds` independent models via `build_and_fit(X_train,
    y_train, seed) -> fitted_model` (a factory you provide, e.g.
    `lambda Xt, yt, s: KANBoostClassifier(gam=True, kan_hidden=1, random_state=s).fit(Xt, yt)`),
    extract `symbolic_summary()` on each, and report two things a single
    extraction can't show:

    - **candidate-function stability**: for each feature, the fraction
      of seeds whose extracted formula picked the *same* (modal)
      candidate function -- a feature whose shape genuinely differs
      run-to-run (a real property of boosting's stochastic training,
      not a bug) shows low agreement here, rather than silently
      presenting one seed's formula as if it were the only possible one.
    - **fidelity per seed**: `formula_fidelity()` on each seed's own
      held-out split, including `auc_model`/`auc_equation` for binary
      classifiers -- so "does the equation retain ranking quality" is a
      measured number per seed, not assumed from a single run.

    Returns `{"candidate_stability": DataFrame, "fidelity_per_seed": DataFrame}`.
    Requires `pandas`.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    candidate_records = []
    fidelity_rows = []

    for seed in range(n_seeds):
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state + seed, stratify=y,
            )
        except ValueError:
            # a class too rare to stratify at this test_size -- fall
            # back to a plain random split rather than failing the seed.
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state + seed,
            )

        model = build_and_fit(X_train, y_train, random_state + seed)
        result = symbolic_summary(model, min_r2=min_r2, top_n=top_n, allow_periodic=allow_periodic)
        sym = result["model"]

        for term in result["ranked_terms"]:
            # pandas' value_counts() silently drops None, which would
            # either crash a numeric-fallback-only feature's modal-agreement
            # lookup (empty value_counts -> IndexError) or, worse, report
            # perfect agreement on a feature that's numeric in most seeds
            # and symbolic in just one (the None rows vanish from the
            # denominator) -- the exact opposite of the instability this
            # function exists to expose. "numeric" is a valid, meaningful
            # candidate label in its own right.
            candidate = term["candidate"] or "numeric"
            candidate_records.append({"seed": seed, "feature": term["feature"], "candidate": candidate})

        fid = formula_fidelity(model, sym, X_test, y_test)
        fidelity_rows.append({"seed": seed, **fid})

    candidates_df = pd.DataFrame(candidate_records)
    # Divide by n_seeds, not by how many seeds a feature happened to
    # appear in (candidates_df.groupby(...).size(), which can be < n_seeds
    # when top_n restricts the ranking and a feature's importance rank
    # shifts seed to seed) -- otherwise a feature that only ever made the
    # top_n cut once, and picked the same candidate that one time, scores
    # a false 1.0 "agreement" instead of the 1/n_seeds it actually earned.
    modal_agreement = (
        candidates_df.groupby("feature")["candidate"]
        .agg(lambda s: s.value_counts().iloc[0] / n_seeds)
        .rename("modal_agreement")
    )
    modal_candidate = (
        candidates_df.groupby("feature")["candidate"]
        .agg(lambda s: s.value_counts().idxmax())
        .rename("modal_candidate")
    )
    stability = pd.concat([modal_agreement, modal_candidate], axis=1).reset_index()

    return {
        "candidate_stability": stability.sort_values("modal_agreement"),
        "fidelity_per_seed": pd.DataFrame(fidelity_rows),
    }


def distill_equation(build_and_fit, X, y, top_n: int = 6, min_r2: float = 0.98,
                      min_relative_amplitude: float = 0.03, stability_threshold: float = 0.7,
                      n_seeds: int = 10, allow_periodic: bool = False,
                      test_size: float = 0.2, random_state: int = 0) -> dict:
    """One-call pipeline combining every symbolic-quality safeguard in
    this module into a single distilled equation, instead of running
    `symbolic_summary()`, amplitude pruning, `stability_across_seeds()`,
    and `refit_constants_from_model()` separately and reconciling them
    by hand.

    `build_and_fit(X_train, y_train, seed) -> fitted_model` is a
    factory you provide (must produce a `gam=True` model -- e.g.
    `lambda Xt, yt, s: KANBoostClassifier(gam=True, kan_hidden=1, random_state=s).fit(Xt, yt)`).

    Pipeline, in order:

    1. Fit a *reference* model on one held-out 80/20 (`test_size`)
       split and extract its `symbolic_summary()` (`min_r2`,
       `allow_periodic`, restricted to the `top_n` most important
       features).
    2. Run `stability_across_seeds()` (`n_seeds` independent refits) to
       get each feature's modal-candidate agreement rate.
    3. Keep a feature only if **all three** hold: it got a genuine
       closed-form term (not a numeric fallback -- i.e. it already
       cleared `min_r2`), its `amplitude` is at least
       `min_relative_amplitude` of the *total* amplitude across all
       `top_n` terms (a term contributing a negligible share of the
       model's total additive range, regardless of its own R^2), and
       its modal-candidate agreement across seeds is at least
       `stability_threshold`.
    4. Jointly refit the surviving terms' constants
       (`refit_constants_from_model()`) against the reference model's
       real raw score -- the intercept and every kept term's `(a,b,c,d)`
       are re-optimized together, not copied from step 1's per-term fit.
    5. Report fidelity (`formula_fidelity()`) of the *final, pruned,
       refit* equation against the reference model on its own held-out
       test split, including `auc_model`/`auc_equation` for a binary
       classifier.

    Not supported for multiclass classifiers -- raises `ValueError`
    immediately (before any of the `n_seeds + 1` model fits below) for
    the same reason `refit_constants_from_model()` doesn't support them:
    `sym` would already be one class chain, not the whole model.

    **Scope note**: step 3's per-term criteria are independent filters,
    not a causal per-term ablation test (removing one term and
    re-measuring the actual AUC drop it alone causes) -- that would need
    re-fitting/re-evaluating once per surviving term per seed, which
    this does not do. `min_relative_amplitude` is a fast, correlational
    proxy for "does this term matter enough to keep", not a substitute
    for a true leave-one-term-out significance test.

    Returns
    -------
    dict with:
        "formula": final `sympy` expression (pruned + jointly refit)
        "latex": str
        "kept_features": list of feature names that survived all three gates
        "dropped_features": {"numeric_fallback": [...], "low_relative_amplitude": [...], "unstable": [...]}
        "candidate_stability": DataFrame from `stability_across_seeds()`
        "fidelity": dict from `formula_fidelity()` on the reference model's held-out split
        "reference_model": the model fit on the reference train split
        "reference_symbolic_model": the final, pruned + refit `SymbolicModel`

    Raises `ValueError` if no feature survives all three gates -- there's
    no equation to return in that case, and silently returning an
    empty/intercept-only formula would be misleading.
    """
    import copy
    from sklearn.model_selection import train_test_split

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
        )

    reference_model = build_and_fit(X_train, y_train, random_state)
    if hasattr(reference_model, "classes_") and len(reference_model.classes_) > 2:
        raise ValueError(
            "distill_equation() only supports binary classifiers or regressors -- "
            "for a multiclass one-vs-rest chain, call export_symbolic(model) directly "
            "and distill each class's chain by hand."
        )

    summary = symbolic_summary(reference_model, min_r2=min_r2, top_n=top_n, allow_periodic=allow_periodic)
    sym = summary["model"]

    total_amplitude = sum(t["amplitude"] for t in summary["ranked_terms"])
    relative_amplitude = {
        t["feature"]: (t["amplitude"] / total_amplitude if total_amplitude > 1e-12 else 0.0)
        for t in summary["ranked_terms"]
    }

    # random_state + 1 (not random_state) for the stability sweep -- the
    # reference model above already used random_state, and
    # stability_across_seeds' own seed loop starts at its random_state +
    # 0, so reusing random_state here would silently refit the exact
    # same split+seed as the reference (wasted work, and that model's
    # own candidate choice would vote in its own agreement score).
    stability = stability_across_seeds(
        build_and_fit, X, y, n_seeds=n_seeds, min_r2=min_r2, top_n=top_n,
        test_size=test_size, random_state=random_state + 1, allow_periodic=allow_periodic,
    )
    agreement = dict(zip(stability["candidate_stability"]["feature"], stability["candidate_stability"]["modal_agreement"]))

    kept_features = []
    dropped = {"numeric_fallback": [], "low_relative_amplitude": [], "unstable": []}
    for term in summary["ranked_terms"]:
        name = term["feature"]
        if term["kind"] != "symbolic":
            dropped["numeric_fallback"].append(name)
        elif relative_amplitude[name] < min_relative_amplitude:
            dropped["low_relative_amplitude"].append(name)
        elif agreement.get(name, 0.0) < stability_threshold:
            dropped["unstable"].append(name)
        else:
            kept_features.append(name)

    if not kept_features:
        raise ValueError(
            "distill_equation() found no feature that survived min_r2, "
            f"min_relative_amplitude={min_relative_amplitude}, and "
            f"stability_threshold={stability_threshold} together -- "
            f"dropped: {dropped}. Loosen one of these thresholds."
        )

    pruned_terms = {name: copy.deepcopy(sym.terms[name]) for name in kept_features}
    pruned_sym = SymbolicModel(
        kept_features, sym.intercept, pruned_terms,
        preprocessor=sym.preprocessor, feature_names_in_=sym.feature_names_in_,
    )
    refit_sym = refit_constants_from_model(reference_model, pruned_sym, X_train)
    fidelity = formula_fidelity(reference_model, refit_sym, X_test, y_test)

    return {
        "formula": refit_sym.to_sympy(),
        "latex": refit_sym.to_latex(),
        "kept_features": kept_features,
        "dropped_features": dropped,
        "candidate_stability": stability["candidate_stability"],
        "fidelity": fidelity,
        "reference_model": reference_model,
        "reference_symbolic_model": refit_sym,
    }


class SymbolicModel:
    """A fitted, fidelity-aware symbolic export. Build one via
    `export_symbolic(model)`, not directly."""

    def __init__(self, feature_names, intercept, terms, preprocessor=None, feature_names_in_=None):
        self.feature_names = list(feature_names)
        self.intercept = float(intercept)
        self.terms = terms  # {name: {"kind", "r2", "sympy_expr" or None, "x_grid", "y_grid"}}
        self.preprocessor = preprocessor
        self.feature_names_in_ = feature_names_in_
        self._symbols = _make_unique_symbols(self.feature_names)

    @classmethod
    def _from_editable(cls, gam, candidates, min_r2, feature_set=None, parsimony_margin=0.0):
        from kan.utils import fit_params, SYMBOLIC_LIB

        terms = {}
        x_t = torch.tensor(gam.x_grid, dtype=torch.float32)
        for name in gam.feature_names:
            amplitude = float(gam.curves[name].max() - gam.curves[name].min())

            if feature_set is not None and name not in feature_set:
                # Skip the (relatively expensive) candidate search for
                # features nobody asked for; keep the exact spline curve
                # as a numeric term instead.
                terms[name] = {
                    "kind": "numeric", "r2": float("nan"), "candidate": None,
                    "x_grid": gam.x_grid.copy(), "y_grid": gam.curves[name].copy(),
                    "amplitude": amplitude,
                }
                continue

            y_t = torch.tensor(gam.curves[name], dtype=torch.float32)
            best = None
            for cand in candidates:
                fun = SYMBOLIC_LIB[cand][0]
                try:
                    params, r2 = fit_params(x_t, y_t, fun, verbose=False)
                except Exception:
                    continue
                r2 = float(r2)
                params_list = [float(p) for p in params]
                if best is None:
                    best = (cand, r2, params_list)
                elif (parsimony_margin > 0
                        and _CANDIDATE_COMPLEXITY.get(cand, 0) > _CANDIDATE_COMPLEXITY.get(best[0], 0)):
                    if r2 > best[1] + parsimony_margin:
                        best = (cand, r2, params_list)
                elif r2 > best[1]:
                    best = (cand, r2, params_list)

            if best is not None and best[1] >= min_r2:
                cand, r2, (a, b, c, d) = best
                terms[name] = {
                    "kind": "symbolic", "r2": r2, "candidate": cand,
                    "params": {"a": a, "b": b, "c": c, "d": d}, "amplitude": amplitude,
                }
            else:
                terms[name] = {
                    "kind": "numeric",
                    "r2": best[1] if best is not None else float("nan"),
                    "candidate": best[0] if best is not None else None,
                    "x_grid": gam.x_grid.copy(), "y_grid": gam.curves[name].copy(),
                    "amplitude": amplitude,
                }

        return cls(gam.feature_names, gam.intercept, terms,
                    preprocessor=gam.preprocessor, feature_names_in_=gam.feature_names_in_)

    # ------------------------------------------------------------------
    def to_sympy(self):
        """The full model as one `sympy` expression, `intercept + sum_j
        term_j(x_j)`. Numeric (non-symbolic) terms appear as an opaque
        function symbol `g_<feature>(x)` -- there's no closed form for
        those; see `fidelity_report()` for which features that affects."""
        from kan.utils import SYMBOLIC_LIB

        expr = sympy.Float(self.intercept)
        for name, term in self.terms.items():
            x = self._symbols[name]
            if term["kind"] == "symbolic":
                sympy_fun = SYMBOLIC_LIB[term["candidate"]][1]
                p = term["params"]
                expr += p["c"] * sympy_fun(p["a"] * x + p["b"]) + p["d"]
            else:
                expr += sympy.Function(f"g_{_safe_symbol_name(name)}")(x)
        return expr

    def to_latex(self) -> str:
        return sympy.latex(self.to_sympy())

    def term_sympy(self, feature: str, simplify: bool = False):
        """Just one feature's term as a `sympy` expression (not the
        whole model) -- `c * fun(a*x + b) + d`, or an opaque
        `g_<feature>(x)` symbol if that feature fell back to a numeric
        term. `simplify=True` runs `sympy.simplify()` on it (can be slow
        on complex expressions; each single term here is cheap)."""
        from kan.utils import SYMBOLIC_LIB

        term = self.terms[feature]
        x = self._symbols[feature]
        if term["kind"] == "symbolic":
            sympy_fun = SYMBOLIC_LIB[term["candidate"]][1]
            p = term["params"]
            expr = p["c"] * sympy_fun(p["a"] * x + p["b"]) + p["d"]
        else:
            expr = sympy.Function(f"g_{_safe_symbol_name(feature)}")(x)
        return sympy.simplify(expr) if simplify else expr

    def fidelity_report(self) -> dict:
        """`{feature: {"kind": "symbolic"|"numeric", "r2": float,
        "candidate": name_or_None, "amplitude": float}}` -- what
        fraction of the model is a true closed form vs. a numeric
        fallback, and how well each feature's chosen candidate actually
        fit.

        `amplitude` (the term's max-min range on [-1, 1]) matters
        alongside `r2`: a high R^2 does not by itself mean a feature is
        important, only that its curve's *shape* -- however small --
        was well matched by some candidate. A near-flat curve can score
        a deceptively high R^2 by fitting its own tiny wiggles; check
        `amplitude` against the other features' to judge whether a term
        actually contributes much to the prediction.
        """
        return {
            name: {"kind": t["kind"], "r2": t["r2"], "candidate": t["candidate"], "amplitude": t["amplitude"]}
            for name, t in self.terms.items()
        }

    def symbolic_fraction(self) -> float:
        """Fraction of features (by count, not by importance) that got
        a genuine closed-form term rather than a numeric fallback."""
        if not self.terms:
            return 0.0
        return sum(t["kind"] == "symbolic" for t in self.terms.values()) / len(self.terms)

    # ------------------------------------------------------------------
    def _term_value_scaled(self, name: str, x: np.ndarray) -> np.ndarray:
        # Evaluated via plain numpy (not torch/sympy) so predict_scaled()
        # never needs torch/pykan at call time, even for symbolic terms.
        term = self.terms[name]
        if term["kind"] == "symbolic":
            p = term["params"]
            fn = _NUMPY_FUN[term["candidate"]]
            return p["c"] * fn(p["a"] * np.asarray(x, dtype=np.float64) + p["b"]) + p["d"]
        return np.interp(np.asarray(x, dtype=np.float64), term["x_grid"], term["y_grid"])

    def predict_scaled(self, X_scaled) -> np.ndarray:
        """Predict from already-scaled feature values (the model's
        internal [-1, 1] range, e.g. `model.preprocessor_.transform(X)`)
        -- pure numpy, no torch/pykan/sympy needed at call time. Column
        order must match `self.feature_names`.

        Note: for a `"log"` or `"sqrt"` term, this and `to_sympy()`'s
        formula only agree where the fitted argument `a*x + b` is
        non-negative (the domain candidates were fit on, [-1, 1]) --
        outside that range this uses `log(|x|+eps)`/`sqrt(|x|)` to stay
        finite, while the symbolic formula would give NaN/complex. Only
        reachable with inputs outside the training range; within-range
        predictions are unaffected.
        """
        X_scaled = np.asarray(X_scaled, dtype=np.float64)
        score = np.full(X_scaled.shape[0], self.intercept)
        for j, name in enumerate(self.feature_names):
            score += self._term_value_scaled(name, X_scaled[:, j])
        return score

    def predict(self, X) -> np.ndarray:
        """Predict from raw (unscaled) input -- requires the
        preprocessor this `SymbolicModel` was exported with (i.e. built
        via `export_symbolic(model)`, not reconstructed by hand)."""
        import pandas as pd

        if self.preprocessor is None:
            raise RuntimeError(
                "This SymbolicModel has no preprocessor attached; use "
                "predict_scaled() with already-scaled arrays instead."
            )
        if not isinstance(X, pd.DataFrame):
            if self.feature_names_in_ is None:
                raise RuntimeError(
                    "Raw array input requires the original fit-time column "
                    "order; pass a DataFrame with column names instead."
                )
            X = pd.DataFrame(np.asarray(X), columns=self.feature_names_in_)
        X_scaled = self.preprocessor.transform(X)
        return self.predict_scaled(X_scaled)

    def save(self, path: str) -> None:
        torch.save({
            "feature_names": self.feature_names,
            "intercept": self.intercept,
            "terms": self.terms,
            "preprocessor": self.preprocessor,
            "feature_names_in_": self.feature_names_in_,
        }, path)

    @classmethod
    def load(cls, path: str) -> "SymbolicModel":
        payload = torch.load(path, weights_only=False)
        return cls(
            payload["feature_names"], payload["intercept"], payload["terms"],
            preprocessor=payload["preprocessor"], feature_names_in_=payload["feature_names_in_"],
        )


def _safe_symbol_name(name: str) -> str:
    """sympy Symbol names can't contain most punctuation; make feature
    names like "mean radius" or "a-b" round-trippable."""
    return "".join(ch if ch.isalnum() else "_" for ch in str(name))


def _make_unique_symbols(feature_names) -> dict:
    """`{original_name: sympy.Symbol}`, guarding against two distinct
    feature names sanitizing to the same symbol name (e.g. "a b" and
    "a-b" both become "a_b") -- sympy interns Symbols by name, so an
    unguarded collision would silently conflate two different features
    into one symbol throughout to_sympy()/term_sympy(). Colliding names
    beyond the first get a numeric suffix."""
    symbols = {}
    seen = {}
    for name in feature_names:
        base = _safe_symbol_name(name)
        if base in seen:
            seen[base] += 1
            base = f"{base}_{seen[base]}"
        else:
            seen[base] = 0
        symbols[name] = sympy.Symbol(base)
    return symbols


_NUMPY_FUN = {
    "x": lambda x: x,
    "x^2": lambda x: x ** 2,
    "x^3": lambda x: x ** 3,
    "sin": np.sin,
    "cos": np.cos,
    "exp": np.exp,
    "log": lambda x: np.log(np.abs(x) + 1e-8),
    "sqrt": lambda x: np.sqrt(np.abs(x)),
    "tanh": np.tanh,
    "abs": np.abs,
}
