"""Tests for kanboost.experimental."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

from kanboost import KANBoostRegressor
from kanboost.experimental import (
    suggest_constraints,
    audit_monotonicity,
    symbolic_export,
    predict_interval,
    explain_row,
    dashboard_html,
)


def _monotone_vs_nonmonotone_data(n=500, seed=0):
    rng = np.random.RandomState(seed)
    income = rng.uniform(-1, 1, n)
    quad = rng.uniform(-1, 1, n)
    other = rng.uniform(-1, 1, n)
    noise = rng.normal(scale=0.5, size=n)
    y = 3.0 * income + quad ** 2 + 0.5 * other + noise
    X = pd.DataFrame({"income": income, "quad": quad, "other": other})
    return X, y


def test_suggest_constraints_distinguishes_monotone_from_quadratic():
    X, y = _monotone_vs_nonmonotone_data()
    constraints = suggest_constraints(X, y)
    assert constraints.get("income") == 1
    assert "quad" not in constraints


def test_suggest_constraints_detects_decreasing():
    rng = np.random.RandomState(1)
    n = 500
    x = rng.uniform(-1, 1, n)
    y = -2.0 * x + rng.normal(scale=0.3, size=n)
    X = pd.DataFrame({"x": x})
    constraints = suggest_constraints(X, y)
    assert constraints.get("x") == -1


def _fit_constrained_regressor(X, y, constraints):
    model = KANBoostRegressor(
        n_estimators=10, kan_steps=8, kan_hidden=1, gam=True,
        monotone_constraints=constraints, early_stopping_rounds=None, random_state=0,
    )
    model.fit(X, y)
    return model


def test_audit_monotonicity_passes_for_constrained_feature():
    X, y = make_regression(n_samples=200, n_features=3, noise=0.1, random_state=0)
    X_df = pd.DataFrame(X, columns=["a", "b", "c"])
    model = _fit_constrained_regressor(X_df, y, {"a": 1})
    report = audit_monotonicity(model, X_df)
    assert report["a"]["passed"] is True
    assert report["a"]["violation_rate"] == 0.0


def test_audit_monotonicity_flags_violation():
    rng = np.random.RandomState(0)
    n = 300
    x = rng.uniform(-1, 1, n)
    y = x ** 2 + rng.normal(scale=0.05, size=n)  # genuinely non-monotone in x
    X_df = pd.DataFrame({"x": x})
    model = KANBoostRegressor(
        n_estimators=10, kan_steps=8, kan_hidden=1, gam=True,
        early_stopping_rounds=None, random_state=0,
    )
    model.fit(X_df, y)
    # falsely assert x is increasing; a quadratic relationship must violate this
    report = audit_monotonicity(model, X_df, constraints={"x": 1})
    assert report["x"]["passed"] is False
    assert report["x"]["violation_rate"] > 0


def test_symbolic_export_binary_and_multiclass():
    from sklearn.datasets import make_classification
    from kanboost import KANBoostClassifier

    X, y = make_classification(n_samples=200, n_features=4, random_state=0)
    X_df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    model = KANBoostClassifier(
        n_estimators=5, kan_steps=5, kan_hidden=1, gam=True,
        early_stopping_rounds=None, random_state=0,
    )
    model.fit(X_df, y)
    summary = symbolic_export(model, X_df, top_k=1, min_r2=0.0)
    assert "score ~=" in summary

    Xm, ym = make_classification(
        n_samples=200, n_features=4, n_informative=4, n_redundant=0,
        n_classes=3, n_clusters_per_class=1, random_state=1,
    )
    Xm_df = pd.DataFrame(Xm, columns=["a", "b", "c", "d"])
    model_mc = KANBoostClassifier(
        n_estimators=3, kan_steps=5, kan_hidden=1, gam=True,
        early_stopping_rounds=None, random_state=0,
    )
    model_mc.fit(Xm_df, ym)
    summary_mc = symbolic_export(model_mc, Xm_df, top_k=1, min_r2=0.0)
    for c in model_mc.classes_:
        assert f"class_{c} ~=" in summary_mc


def test_predict_interval_shape_and_validation():
    X, y = make_regression(n_samples=100, n_features=3, noise=0.1, random_state=0)
    X_df = pd.DataFrame(X, columns=["a", "b", "c"])
    models = [
        KANBoostRegressor(n_estimators=4, kan_steps=4, random_state=seed, early_stopping_rounds=None).fit(X_df, y)
        for seed in (0, 1, 2)
    ]
    interval = predict_interval(models, X_df, level=0.9)
    assert set(interval.keys()) == {"mean", "lower", "upper", "std"}
    assert interval["lower"].shape == (100,)
    assert np.all(interval["lower"] <= interval["upper"])

    try:
        predict_interval(models[0], X_df)
        raise AssertionError("non-list models was not rejected")
    except TypeError:
        pass


def test_explain_row_binary_and_multiclass():
    from sklearn.datasets import make_classification
    from kanboost import KANBoostClassifier

    X, y = make_classification(n_samples=100, n_features=4, random_state=0)
    X_df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    model = KANBoostClassifier(n_estimators=3, kan_steps=3, early_stopping_rounds=None, random_state=0)
    model.fit(X_df, y)
    top = explain_row(model, X_df, row_index=0, top_k=2)
    assert len(top) == 2
    assert all("feature" in t and "contribution" in t for t in top)

    Xm, ym = make_classification(
        n_samples=100, n_features=4, n_informative=4, n_redundant=0,
        n_classes=3, n_clusters_per_class=1, random_state=1,
    )
    Xm_df = pd.DataFrame(Xm, columns=["a", "b", "c", "d"])
    model_mc = KANBoostClassifier(n_estimators=3, kan_steps=3, early_stopping_rounds=None, random_state=0)
    model_mc.fit(Xm_df, ym)
    top_mc = explain_row(model_mc, Xm_df, row_index=0, top_k=2)
    assert len(top_mc) == 2


def test_dashboard_html_handles_numpy_types(tmp_path):
    X, y = make_regression(n_samples=100, n_features=3, noise=0.1, random_state=0)
    X_df = pd.DataFrame(X, columns=["a", "b", "c"])
    model = _fit_constrained_regressor(X_df, y, {"a": 1})

    path = str(tmp_path / "report.html")
    out = dashboard_html(model, X_df, y, path=path)
    content = open(out, encoding="utf-8").read()
    assert "KANBoost Explainability Report" in content

    # the embedded JSON blob must itself be valid (regression test for the
    # np.float32/np.int64-vs-json.dumps crash)
    start = content.index("<pre>") + len("<pre>")
    end = content.index("</pre>")
    parsed = json.loads(content[start:end])
    assert parsed["model"] == "KANBoostRegressor"


def test_json_safe_converts_numpy_types_directly():
    """Direct unit test for _json_safe -- the dashboard test above
    exercises a real model's output, but nothing in that pipeline
    actually happens to produce numpy scalars/arrays, so it wouldn't
    catch _json_safe being removed or broken. This does."""
    from kanboost.experimental import _json_safe

    payload = {
        "a": np.float32(1.5),
        "b": np.int64(3),
        "c": np.arange(3),
        "d": [np.float32(2.0), {"e": np.int32(4)}],
    }
    safe = _json_safe(payload)
    dumped = json.dumps(safe)  # must not raise
    reloaded = json.loads(dumped)
    assert reloaded["a"] == 1.5
    assert reloaded["b"] == 3
    assert reloaded["c"] == [0, 1, 2]
    assert reloaded["d"] == [2.0, {"e": 4}]


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    test_suggest_constraints_distinguishes_monotone_from_quadratic()
    test_suggest_constraints_detects_decreasing()
    test_audit_monotonicity_passes_for_constrained_feature()
    test_audit_monotonicity_flags_violation()
    test_symbolic_export_binary_and_multiclass()
    test_predict_interval_shape_and_validation()
    test_explain_row_binary_and_multiclass()
    with tempfile.TemporaryDirectory() as d:
        test_dashboard_html_handles_numpy_types(Path(d))
    test_json_safe_converts_numpy_types_directly()
    print("All experimental tests passed.")
