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
from kanboost.symbolic import export_symbolic, explain, symbolic_summary, SymbolicModel


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
    print("All symbolic tests passed.")
