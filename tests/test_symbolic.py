"""
Tests for kanboost.symbolic (export_symbolic() / SymbolicModel).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import sympy
from sklearn.datasets import make_classification

from kanboost import KANBoostClassifier, KANBoostRegressor
from kanboost.interpret.symbolic import (
    export_symbolic, explain, symbolic_summary, SymbolicModel,
    refit_constants, refit_constants_from_model, formula_fidelity, stability_across_seeds,
    stability_across_sample_sizes, distill_equation, tiered_equations,
)


def _known_function_data(n=800, seed=0):
    rng = np.random.RandomState(seed)
    x1 = rng.uniform(-1, 1, n)
    x2 = rng.uniform(-1, 1, n)
    x3 = rng.uniform(-1, 1, n)  # near-noise feature: no real relationship
    y = 3 * np.sin(2 * x1) + 2 * (x2 ** 2) + rng.normal(scale=0.05, size=n)
    X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    return X, y


def test_requires_gam():
    X, y = _known_function_data()
    model = KANBoostRegressor(n_estimators=5, kan_steps=5, early_stopping_rounds=None, random_state=0)
    model.fit(X, y)
    try:
        export_symbolic(model)
        raise AssertionError("export_symbolic() on a non-gam model was not rejected")
    except ValueError:
        pass


def test_fidelity_report_amplitude_flags_negligible_term():
    """x3 has no real relationship to y -- its curve should have a much
    smaller amplitude than x1/x2, even if a candidate happens to fit its
    small residual wiggle with a deceptively high R^2 (this is exactly
    why fidelity_report() reports amplitude alongside r2)."""
    X, y = _known_function_data()
    model = KANBoostRegressor(
        n_estimators=30, kan_steps=15, kan_hidden=1, gam=True,
        early_stopping_rounds=None, random_state=0,
    )
    model.fit(X, y)
    sym = export_symbolic(model, min_r2=0.85)
    report = sym.fidelity_report()

    assert report["x1"]["amplitude"] > report["x3"]["amplitude"] * 5
    assert report["x2"]["amplitude"] > report["x3"]["amplitude"] * 3
    for name in sym.feature_names:
        assert "r2" in report[name] and "kind" in report[name] and "amplitude" in report[name]


def test_predict_is_a_reasonable_approximation():
    X, y = _known_function_data()
    model = KANBoostRegressor(
        n_estimators=30, kan_steps=15, kan_hidden=1, gam=True,
        early_stopping_rounds=None, random_state=0,
    )
    model.fit(X, y)
    sym = export_symbolic(model, min_r2=0.85)

    orig_pred = model.predict(X)
    sym_pred = sym.predict(X)
    # A lossy approximation (each feature's spline is itself replaced by a
    # fitted closed-form candidate) -- not exact like EditableGAM.predict,
    # but should track the real predictions reasonably closely.
    assert np.corrcoef(orig_pred, sym_pred)[0, 1] > 0.98
    assert np.mean(np.abs(orig_pred - sym_pred)) < 0.5 * np.std(y)

    # raw array input must match DataFrame input
    assert np.allclose(sym.predict(X.values), sym.predict(X))


def test_to_sympy_and_to_latex():
    X, y = _known_function_data()
    model = KANBoostRegressor(
        n_estimators=20, kan_steps=10, kan_hidden=1, gam=True,
        early_stopping_rounds=None, random_state=0,
    )
    model.fit(X, y)
    sym = export_symbolic(model, min_r2=0.85)

    expr = sym.to_sympy()
    assert isinstance(expr, sympy.Expr)
    latex = sym.to_latex()
    assert isinstance(latex, str) and len(latex) > 0


def test_numeric_fallback_for_low_r2():
    """A feature with a genuinely irregular/non-parametric shape should
    fall back to a numeric term when min_r2 is set high enough that no
    candidate clears it."""
    rng = np.random.RandomState(0)
    n = 600
    x1 = rng.uniform(-1, 1, n)
    x2 = rng.uniform(-1, 1, n)
    # a jagged, non-smooth function no SYMBOLIC_LIB candidate fits well
    y = np.sign(np.sin(15 * x1)) * 2.0 + 0.1 * x2 + rng.normal(scale=0.02, size=n)
    X = pd.DataFrame({"x1": x1, "x2": x2})

    model = KANBoostRegressor(
        n_estimators=25, kan_steps=15, kan_hidden=1, gam=True,
        early_stopping_rounds=None, random_state=0,
    )
    model.fit(X, y)
    sym = export_symbolic(model, min_r2=0.999)  # deliberately strict
    report = sym.fidelity_report()
    assert any(t["kind"] == "numeric" for t in report.values())
    assert sym.symbolic_fraction() < 1.0

    # predict must still work (numeric terms use spline interpolation)
    preds = sym.predict(X)
    assert preds.shape == (len(X),)
    assert np.isfinite(preds).all()


def test_binary_classifier():
    X, y = _known_function_data()
    y_bin = (y > np.median(y)).astype(int)
    model = KANBoostClassifier(
        n_estimators=20, kan_steps=12, kan_hidden=1, gam=True,
        early_stopping_rounds=None, random_state=0,
    )
    model.fit(X, y_bin)
    sym = export_symbolic(model, min_r2=0.8)
    assert isinstance(sym, SymbolicModel)
    raw_score = sym.predict(X)
    assert raw_score.shape == (len(X),)


def test_multiclass_returns_dict():
    X, y = make_classification(
        n_samples=300, n_features=4, n_informative=4, n_redundant=0,
        n_classes=3, n_clusters_per_class=1, random_state=1,
    )
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    model = KANBoostClassifier(
        n_estimators=10, kan_steps=8, kan_hidden=1, gam=True,
        early_stopping_rounds=None, random_state=0,
    )
    model.fit(X_df, y)
    sym = export_symbolic(model, min_r2=0.8)
    assert isinstance(sym, dict)
    assert set(sym.keys()) == set(model.classes_)
    for m in sym.values():
        assert isinstance(m, SymbolicModel)


def test_save_load_roundtrip(tmp_path):
    X, y = _known_function_data()
    model = KANBoostRegressor(
        n_estimators=20, kan_steps=10, kan_hidden=1, gam=True,
        early_stopping_rounds=None, random_state=0,
    )
    model.fit(X, y)
    sym = export_symbolic(model, min_r2=0.85)

    path = str(tmp_path / "symbolic.pt")
    sym.save(path)
    loaded = SymbolicModel.load(path)

    assert np.allclose(loaded.predict(X), sym.predict(X))
    assert loaded.fidelity_report() == sym.fidelity_report()


def test_explain_ranks_by_importance_and_attaches_formulas():
    X, y = _known_function_data()
    model = KANBoostRegressor(
        n_estimators=25, kan_steps=12, kan_hidden=1, gam=True,
        early_stopping_rounds=None, random_state=0,
    )
    model.fit(X, y)

    report = explain(model, top_features=2, symbolic=True, simplify=True)
    assert len(report) == 2
    importances = model.feature_importances_dict()
    # sorted by importance, most important first
    assert report[0]["importance"] >= report[1]["importance"]
    assert report[0]["feature"] in importances and report[1]["feature"] in importances
    for entry in report:
        assert entry["kind"] in ("symbolic", "numeric")
        assert isinstance(entry["formula"], sympy.Expr)
        assert entry["amplitude"] is not None


def test_explain_symbolic_false_omits_formulas():
    X, y = _known_function_data()
    model = KANBoostRegressor(
        n_estimators=10, kan_steps=8, kan_hidden=1, gam=True,
        early_stopping_rounds=None, random_state=0,
    )
    model.fit(X, y)

    report = explain(model, top_features=3, symbolic=False)
    assert len(report) == 3
    assert all(entry["formula"] is None and entry["kind"] is None for entry in report)


def test_explain_multiclass_uses_first_class_chain():
    X, y = make_classification(
        n_samples=300, n_features=6, n_informative=5, n_redundant=0,
        n_classes=3, n_clusters_per_class=1, random_state=1,
    )
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    model = KANBoostClassifier(
        n_estimators=10, kan_steps=8, kan_hidden=1, gam=True,
        early_stopping_rounds=None, random_state=0,
    )
    model.fit(X_df, y)

    report = explain(model, top_features=3, symbolic=True)
    assert len(report) == 3
    for entry in report:
        assert isinstance(entry["formula"], sympy.Expr)

    # explicitly verify the documented provenance: each formula must
    # equal classes_[0]'s chain's term, not some other class's
    sym_per_class = export_symbolic(model, min_r2=0.8)
    first_class_sym = sym_per_class[model.classes_[0]]
    for entry in report:
        expected = first_class_sym.term_sympy(entry["feature"], simplify=False)
        assert sympy.simplify(entry["formula"] - expected) == 0 or entry["formula"] == expected


def test_symbolic_summary_ranks_by_amplitude_not_dict_order():
    X, y = _known_function_data()
    model = KANBoostRegressor(
        n_estimators=30, kan_steps=15, kan_hidden=1, gam=True,
        early_stopping_rounds=None, random_state=0,
    )
    model.fit(X, y)
    result = symbolic_summary(model, min_r2=0.85)

    assert set(result.keys()) == {"ranked_terms", "full_formula", "full_latex", "model"}
    assert len(result["ranked_terms"]) == 3  # x1, x2, x3
    amplitudes = [t["amplitude"] for t in result["ranked_terms"]]
    assert amplitudes == sorted(amplitudes, reverse=True)
    # x3 is near-noise (see test_fidelity_report_amplitude_flags_negligible_term)
    # -- it must rank last, not just be present.
    assert result["ranked_terms"][-1]["feature"] == "x3"
    for t in result["ranked_terms"]:
        assert isinstance(t["formula"], sympy.Expr)
    assert isinstance(result["full_formula"], sympy.Expr)
    assert isinstance(result["full_latex"], str) and len(result["full_latex"]) > 0
    assert isinstance(result["model"], SymbolicModel)


def test_symbolic_summary_top_n_restricts_ranked_terms_not_just_candidate_search():
    X, y = _known_function_data()
    model = KANBoostRegressor(
        n_estimators=30, kan_steps=15, kan_hidden=1, gam=True,
        early_stopping_rounds=None, random_state=0,
    )
    model.fit(X, y)
    result = symbolic_summary(model, min_r2=0.85, top_n=2)

    # A real bug this exact test guards against: top_n restricting only
    # the (expensive) candidate search, while ranked_terms still listed
    # every feature in the model regardless of top_n.
    assert len(result["ranked_terms"]) == 2
    names = {t["feature"] for t in result["ranked_terms"]}
    assert names == {"x1", "x2"}  # the two features feature_importances_dict() ranks highest


def test_symbolic_summary_rejects_non_positive_top_n():
    X, y = _known_function_data()
    model = KANBoostRegressor(
        n_estimators=10, kan_steps=8, kan_hidden=1, gam=True,
        early_stopping_rounds=None, random_state=0,
    )
    model.fit(X, y)
    for bad in (0, -1):
        try:
            symbolic_summary(model, top_n=bad)
            raise AssertionError(f"top_n={bad} was not rejected")
        except ValueError:
            pass


def test_parsimony_margin_prefers_simpler_candidate():
    X, y = _known_function_data()
    model = KANBoostRegressor(
        n_estimators=30, kan_steps=15, kan_hidden=1, gam=True,
        early_stopping_rounds=None, random_state=0,
    )
    model.fit(X, y)

    sym_default = export_symbolic(model, min_r2=0.8, parsimony_margin=0.0)
    sym_parsimonious = export_symbolic(model, min_r2=0.8, parsimony_margin=0.2)

    from kanboost.interpret.symbolic import _CANDIDATE_COMPLEXITY

    for name, term in sym_parsimonious.terms.items():
        if term["kind"] != "symbolic" or sym_default.terms[name]["kind"] != "symbolic":
            continue
        default_complexity = _CANDIDATE_COMPLEXITY[sym_default.terms[name]["candidate"]]
        parsimonious_complexity = _CANDIDATE_COMPLEXITY[term["candidate"]]
        # A large margin should never end up with a *more* complex
        # candidate than the unmargined default chose for the same term.
        assert parsimonious_complexity <= default_complexity


def test_min_amplitude_prunes_ranked_terms_and_full_formula():
    X, y = _known_function_data()
    model = KANBoostRegressor(
        n_estimators=30, kan_steps=15, kan_hidden=1, gam=True,
        early_stopping_rounds=None, random_state=0,
    )
    model.fit(X, y)

    full = symbolic_summary(model, min_r2=0.85)
    amplitudes = sorted((t["amplitude"] for t in full["ranked_terms"]), reverse=True)
    threshold = amplitudes[1]  # keep only the single largest-amplitude term

    pruned = symbolic_summary(model, min_r2=0.85, min_amplitude=threshold)
    assert len(pruned["ranked_terms"]) < len(full["ranked_terms"])
    assert all(t["amplitude"] >= threshold for t in pruned["ranked_terms"])
    # the pruned formula must not just be filtered in ranked_terms --
    # it must actually drop the term from the equation itself.
    assert len(pruned["full_formula"].free_symbols) == len(pruned["ranked_terms"])


def test_refit_constants_improves_or_preserves_fidelity():
    X, y = _known_function_data()
    y_bin = (y > np.median(y)).astype(int)
    X_train, X_test = X.iloc[:600], X.iloc[600:]
    y_train, y_test = y_bin[:600], y_bin[600:]

    model = KANBoostClassifier(
        n_estimators=20, kan_steps=12, kan_hidden=1, gam=True,
        early_stopping_rounds=None, random_state=0,
    )
    model.fit(X_train, y_train)

    sym = export_symbolic(model, min_r2=0.8)
    fid_before = formula_fidelity(model, sym, X_test, y_test)

    sym_refit = refit_constants_from_model(model, sym, X_train)
    fid_after = formula_fidelity(model, sym_refit, X_test, y_test)

    # Refitting jointly against the real target should not make the
    # held-out fit meaningfully worse.
    assert fid_after["mean_abs_error"] <= fid_before["mean_abs_error"] + 1e-6
    assert isinstance(sym_refit, SymbolicModel)
    assert sym_refit is not sym  # returns a new object, doesn't mutate


def test_refit_constants_from_model_rejects_multiclass():
    Xm, ym = make_classification(
        n_samples=300, n_features=4, n_informative=4, n_redundant=0,
        n_classes=3, n_clusters_per_class=1, random_state=1,
    )
    X_df = pd.DataFrame(Xm, columns=[f"f{i}" for i in range(4)])
    model = KANBoostClassifier(
        n_estimators=10, kan_steps=8, kan_hidden=1, gam=True,
        early_stopping_rounds=None, random_state=0,
    )
    model.fit(X_df, ym)
    sym = export_symbolic(model, min_r2=0.8)[model.classes_[0]]
    try:
        refit_constants_from_model(model, sym, X_df)
        raise AssertionError("multiclass model was not rejected")
    except ValueError:
        pass


def test_formula_fidelity_reports_auc_only_for_binary_with_labels():
    X, y = _known_function_data()
    y_bin = (y > np.median(y)).astype(int)
    model = KANBoostClassifier(
        n_estimators=15, kan_steps=10, kan_hidden=1, gam=True,
        early_stopping_rounds=None, random_state=0,
    )
    model.fit(X, y_bin)
    sym = export_symbolic(model, min_r2=0.8)

    fid_no_y = formula_fidelity(model, sym, X)
    assert "auc_model" not in fid_no_y
    assert "max_abs_error" in fid_no_y

    fid_with_y = formula_fidelity(model, sym, X, y_bin)
    assert 0.0 <= fid_with_y["auc_model"] <= 1.0
    assert 0.0 <= fid_with_y["auc_equation"] <= 1.0

    # regressor: never reports AUC, with or without a "y" passed
    reg = KANBoostRegressor(
        n_estimators=15, kan_steps=10, kan_hidden=1, gam=True,
        early_stopping_rounds=None, random_state=0,
    )
    reg.fit(X, y)
    reg_sym = export_symbolic(reg, min_r2=0.8)
    fid_reg = formula_fidelity(reg, reg_sym, X, y)
    assert "auc_model" not in fid_reg


def test_stability_across_seeds_reports_candidate_agreement():
    X, y = _known_function_data()
    y_bin = (y > np.median(y)).astype(int)

    def build_and_fit(X_train, y_train, seed):
        m = KANBoostClassifier(
            n_estimators=15, kan_steps=10, kan_hidden=1, gam=True,
            early_stopping_rounds=None, random_state=seed,
        )
        m.fit(X_train, y_train)
        return m

    report = stability_across_seeds(build_and_fit, X, y_bin, n_seeds=3, min_r2=0.8, top_n=2)

    assert set(report.keys()) == {"candidate_stability", "fidelity_per_seed"}
    assert len(report["fidelity_per_seed"]) == 3
    assert set(report["fidelity_per_seed"].columns) >= {
        "seed", "max_abs_error", "mean_abs_error", "auc_model", "auc_equation",
    }
    assert (report["candidate_stability"]["modal_agreement"] <= 1.0).all()
    assert (report["candidate_stability"]["modal_agreement"] > 0.0).all()


def test_stability_across_sample_sizes_reports_agreement_and_stabilized_at():
    X, y = _known_function_data(n=1000)
    y_bin = (y > np.median(y)).astype(int)

    def build_and_fit(X_train, y_train, seed):
        m = KANBoostClassifier(
            n_estimators=15, kan_steps=10, kan_hidden=1, gam=True,
            early_stopping_rounds=None, random_state=seed,
        )
        m.fit(X_train, y_train)
        return m

    report = stability_across_sample_sizes(
        build_and_fit, X, y_bin, sample_sizes=[100, 300, 600], min_r2=0.8, top_n=2,
    )

    assert set(report.keys()) == {
        "candidates_by_size", "agreement_with_largest", "fidelity_by_size", "stabilized_at",
    }
    assert set(report["candidates_by_size"].keys()) == {100, 300, 600}
    assert set(report["agreement_with_largest"].keys()) == {100, 300, 600}
    assert report["agreement_with_largest"][600] == 1.0  # the largest always agrees with itself
    assert all(0.0 <= v <= 1.0 for v in report["agreement_with_largest"].values())
    assert len(report["fidelity_by_size"]) == 3
    assert set(report["fidelity_by_size"].columns) >= {
        "sample_size", "max_abs_error", "mean_abs_error", "auc_model", "auc_equation",
    }
    assert report["stabilized_at"] in (100, 300, 600)


def test_stability_across_sample_sizes_rejects_unsorted_sizes():
    X, y = _known_function_data()
    y_bin = (y > np.median(y)).astype(int)

    def build_and_fit(X_train, y_train, seed):
        m = KANBoostClassifier(
            n_estimators=5, kan_steps=5, kan_hidden=1, gam=True,
            early_stopping_rounds=None, random_state=seed,
        )
        m.fit(X_train, y_train)
        return m

    try:
        stability_across_sample_sizes(build_and_fit, X, y_bin, sample_sizes=[300, 100])
        raise AssertionError("unsorted sample_sizes was not rejected")
    except ValueError:
        pass


def test_stability_across_seeds_handles_numeric_fallback_feature():
    # `candidate=None` in a "numeric"-kind term happens when every
    # candidate's fit_params() call raises (e.g. a NaN curve -- verified
    # separately: fit_params raises ValueError on any candidate given an
    # all-NaN y) -- reachable even with top_n=None (so the term isn't
    # structurally excluded from ranked_terms). Reproducing that
    # end-to-end from real training data is not reliably constructible
    # (fit_params turns out to fit *something* with near-1.0 R^2 even
    # for a genuinely unrelated feature, since sin/cos/tanh have enough
    # free parameters to track small wiggles in a 200-point curve), so
    # this monkeypatches symbolic_summary() to return that exact shape
    # directly, isolating the test to stability_across_seeds' own
    # candidate-recording logic. A real bug this guards against:
    # pandas' value_counts() silently drops None, which either crashed
    # the modal-agreement lookup (empty value_counts -> IndexError) or,
    # for a feature numeric in most seeds and symbolic in one, reported
    # a false modal_agreement of 1.0 (the None rows vanished from the
    # denominator) -- exactly backwards for a stability check.
    from unittest.mock import patch

    def fake_symbolic_summary(model, min_r2=0.8, top_n=None, allow_periodic=True):
        return {
            "ranked_terms": [
                {"feature": "x1", "kind": "numeric", "r2": float("nan"),
                 "candidate": None, "amplitude": 0.1, "formula": sympy.Symbol("x1")},
            ],
            "full_formula": sympy.Symbol("x1"),
            "full_latex": "x1",
            "model": None,
        }

    X, y = _known_function_data()
    y_bin = (y > np.median(y)).astype(int)

    def build_and_fit(X_train, y_train, seed):
        m = KANBoostClassifier(
            n_estimators=10, kan_steps=8, kan_hidden=1, gam=True,
            early_stopping_rounds=None, random_state=seed,
        )
        m.fit(X_train, y_train)
        return m

    with patch("kanboost.interpret.symbolic.symbolic_summary", side_effect=fake_symbolic_summary), \
         patch("kanboost.interpret.symbolic.formula_fidelity", return_value={"max_abs_error": 0.0, "mean_abs_error": 0.0}):
        report = stability_across_seeds(build_and_fit, X, y_bin, n_seeds=3, min_r2=0.8)

    stability = report["candidate_stability"]
    assert len(stability) == 1
    assert stability.iloc[0]["modal_candidate"] == "numeric"
    assert stability.iloc[0]["modal_agreement"] == 1.0


def test_allow_periodic_false_excludes_sin_and_cos():
    # sin/tanh/cos can look nearly identical over the fitted [-1, 1]
    # domain (documented elsewhere in this module), so which one wins
    # the unconstrained R^2 race isn't guaranteed to be literally "sin"
    # even when sin is the true generator -- only that *some* feature
    # picks a periodic candidate by default is asserted here, not which
    # specific feature.
    X, y = _known_function_data()  # y = 3*sin(2*x1) + 2*x2^2 + noise
    model = KANBoostRegressor(
        n_estimators=30, kan_steps=15, kan_hidden=1, gam=True,
        early_stopping_rounds=None, random_state=0,
    )
    model.fit(X, y)

    sym_default = export_symbolic(model, min_r2=0.8)
    default_candidates = {t["candidate"] for t in sym_default.terms.values() if t["kind"] == "symbolic"}
    assert default_candidates & {"sin", "cos"}  # sanity: periodic really is preferred by default here

    sym_no_periodic = export_symbolic(model, min_r2=0.8, allow_periodic=False)
    candidates = {t["candidate"] for t in sym_no_periodic.terms.values() if t["kind"] == "symbolic"}
    assert "sin" not in candidates and "cos" not in candidates


def test_distill_equation_returns_well_formed_report():
    X, y = _known_function_data()
    y_bin = (y > np.median(y)).astype(int)

    def build_and_fit(X_train, y_train, seed):
        m = KANBoostClassifier(
            n_estimators=15, kan_steps=10, kan_hidden=1, gam=True,
            early_stopping_rounds=None, random_state=seed,
        )
        m.fit(X_train, y_train)
        return m

    result = distill_equation(
        build_and_fit, X, y_bin, top_n=3, min_r2=0.8, min_relative_amplitude=0.0,
        stability_threshold=0.0, n_seeds=3, allow_periodic=False,
    )

    assert set(result.keys()) == {
        "formula", "latex", "kept_features", "dropped_features",
        "candidate_stability", "fidelity", "reference_model", "reference_symbolic_model",
    }
    assert len(result["kept_features"]) >= 1
    assert isinstance(result["formula"], sympy.Expr)
    # with allow_periodic=False, the final refit formula must not contain sin/cos
    formula_funcs = {str(f.func) for f in result["formula"].atoms(sympy.Function)}
    assert "sin" not in formula_funcs and "cos" not in formula_funcs
    assert "auc_model" in result["fidelity"]
    assert result["reference_symbolic_model"].feature_names == result["kept_features"]


def test_distill_equation_max_terms_caps_and_keeps_highest_amplitude():
    X, y = _known_function_data()
    y_bin = (y > np.median(y)).astype(int)

    def build_and_fit(X_train, y_train, seed):
        m = KANBoostClassifier(
            n_estimators=15, kan_steps=10, kan_hidden=1, gam=True,
            early_stopping_rounds=None, random_state=seed,
        )
        m.fit(X_train, y_train)
        return m

    kwargs = dict(
        top_n=4, min_r2=0.8, min_relative_amplitude=0.0,
        stability_threshold=0.0, n_seeds=3, allow_periodic=False,
    )
    uncapped = distill_equation(build_and_fit, X, y_bin, **kwargs)
    assert len(uncapped["kept_features"]) > 2  # sanity: the cap below is real

    capped = distill_equation(build_and_fit, X, y_bin, max_terms=2, **kwargs)
    assert len(capped["kept_features"]) == 2
    # the two kept are the two highest-amplitude of the uncapped result,
    # in the same (amplitude-descending) order.
    assert capped["kept_features"] == uncapped["kept_features"][:2]


def test_distill_equation_raises_when_nothing_survives_gates():
    X, y = _known_function_data()
    y_bin = (y > np.median(y)).astype(int)

    def build_and_fit(X_train, y_train, seed):
        m = KANBoostClassifier(
            n_estimators=10, kan_steps=8, kan_hidden=1, gam=True,
            early_stopping_rounds=None, random_state=seed,
        )
        m.fit(X_train, y_train)
        return m

    try:
        # an impossible relative-amplitude bar -- no term can carry more
        # than 100% of the total amplitude budget.
        distill_equation(
            build_and_fit, X, y_bin, top_n=3, min_r2=0.8,
            min_relative_amplitude=1.5, n_seeds=2,
        )
        raise AssertionError("an impossible threshold combination was not rejected")
    except ValueError:
        pass


def test_tiered_equations_returns_three_tiers_with_increasing_fidelity():
    X, y = _known_function_data()
    y_bin = (y > np.median(y)).astype(int)

    def build_and_fit(X_train, y_train, seed):
        m = KANBoostClassifier(
            n_estimators=15, kan_steps=10, kan_hidden=1, gam=True,
            early_stopping_rounds=None, random_state=seed,
        )
        m.fit(X_train, y_train)
        return m

    report = tiered_equations(
        build_and_fit, X, y_bin, simple_max_terms=2, detailed_max_terms=3,
        min_r2=0.8, min_relative_amplitude=0.0, stability_threshold=0.0,
        n_seeds=3, allow_periodic=False,
    )

    assert set(report.keys()) == {"simple", "detailed", "full"}
    for tier_name in ("simple", "detailed", "full"):
        tier = report[tier_name]
        assert isinstance(tier["formula"], sympy.Expr)
        assert "auc_model" in tier["fidelity"]
        assert "auc_equation" in tier["fidelity"]
    assert "dropped_low_r2" in report["full"]

    assert len(report["simple"]["kept_features"]) <= 2
    assert len(report["detailed"]["kept_features"]) <= 3
    # full has no term cap -- it must never end up with fewer terms than
    # detailed on the same underlying model/data.
    assert len(report["full"]["kept_features"]) >= len(report["detailed"]["kept_features"])


def test_symbolic_summary_multiclass_uses_first_class_chain():
    X, y = make_classification(
        n_samples=300, n_features=4, n_informative=4, n_redundant=0,
        n_classes=3, n_clusters_per_class=1, random_state=1,
    )
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    model = KANBoostClassifier(
        n_estimators=10, kan_steps=8, kan_hidden=1, gam=True,
        early_stopping_rounds=None, random_state=0,
    )
    model.fit(X_df, y)
    result = symbolic_summary(model, min_r2=0.8)

    expected_sym = export_symbolic(model, min_r2=0.8)[model.classes_[0]]
    assert result["full_latex"] == expected_sym.to_latex()


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    test_requires_gam()
    test_fidelity_report_amplitude_flags_negligible_term()
    test_predict_is_a_reasonable_approximation()
    test_to_sympy_and_to_latex()
    test_numeric_fallback_for_low_r2()
    test_binary_classifier()
    test_multiclass_returns_dict()
    with tempfile.TemporaryDirectory() as d:
        test_save_load_roundtrip(Path(d))
    test_explain_ranks_by_importance_and_attaches_formulas()
    test_explain_symbolic_false_omits_formulas()
    test_explain_multiclass_uses_first_class_chain()
    test_symbolic_summary_ranks_by_amplitude_not_dict_order()
    test_symbolic_summary_top_n_restricts_ranked_terms_not_just_candidate_search()
    test_symbolic_summary_rejects_non_positive_top_n()
    test_symbolic_summary_multiclass_uses_first_class_chain()
    test_parsimony_margin_prefers_simpler_candidate()
    test_min_amplitude_prunes_ranked_terms_and_full_formula()
    test_refit_constants_improves_or_preserves_fidelity()
    test_refit_constants_from_model_rejects_multiclass()
    test_formula_fidelity_reports_auc_only_for_binary_with_labels()
    test_stability_across_seeds_reports_candidate_agreement()
    test_stability_across_seeds_handles_numeric_fallback_feature()
    test_allow_periodic_false_excludes_sin_and_cos()
    test_distill_equation_returns_well_formed_report()
    test_distill_equation_raises_when_nothing_survives_gates()
    print("All symbolic tests passed.")
