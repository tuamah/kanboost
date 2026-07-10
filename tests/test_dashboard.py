"""
Tests for kanboost.dashboard. Skipped entirely if streamlit isn't
installed, since the dashboard is an optional extra
(`pip install kanboost[dashboard]`).
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from kanboost import KANBoostClassifier, KANBoostRegressor
from kanboost.ops.dashboard import (
    explain_row_dataframe,
    importances_dataframe,
    interaction_dataframe,
    launch,
    model_overview,
)

st_testing = pytest.importorskip(
    "streamlit.testing.v1", reason="streamlit not installed -- kanboost[dashboard] is optional"
)

_DASHBOARD_APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kanboost", "ops", "_dashboard_app.py")


# ------------------------------------------------------------------
# plain helper functions (no streamlit dependency)
# ------------------------------------------------------------------
def test_model_overview_regressor():
    X, y = make_regression(n_samples=100, n_features=3, noise=0.1, random_state=0)
    X_df = pd.DataFrame(X, columns=["a", "b", "c"])
    model = KANBoostRegressor(n_estimators=3, kan_steps=3, early_stopping_rounds=None, random_state=0)
    model.fit(X_df, y)

    info = model_overview(model)
    assert info["class"] == "KANBoostRegressor"
    assert info["n_features"] == 3
    assert info["feature_names"] == ["a", "b", "c"]
    assert info["is_classifier"] is False
    assert info["is_multiclass"] is False


def test_model_overview_multiclass():
    X, y = make_classification(
        n_samples=150, n_features=4, n_informative=4, n_redundant=0,
        n_classes=3, n_clusters_per_class=1, random_state=1,
    )
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    model = KANBoostClassifier(n_estimators=3, kan_steps=3, early_stopping_rounds=None, random_state=0)
    model.fit(X_df, y)

    info = model_overview(model)
    assert info["is_classifier"] is True
    assert info["is_multiclass"] is True


def test_importances_and_interaction_and_explain_dataframes():
    X, y = make_regression(n_samples=150, n_features=4, noise=0.1, random_state=0)
    X_df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    model = KANBoostRegressor(
        n_estimators=5, kan_steps=5, kan_hidden=3, early_stopping_rounds=None, random_state=0,
    )
    model.fit(X_df, y)

    imps = importances_dataframe(model)
    assert list(imps.columns) == ["feature", "importance"]
    assert len(imps) == 4

    inter = interaction_dataframe(model, X_df, top_k=5)
    assert "feature_pair" in inter.columns and "count" in inter.columns

    exp = explain_row_dataframe(model, X_df, row_index=0, top_k=2)
    assert len(exp) == 2


def test_launch_rejects_missing_files():
    try:
        launch("no_such_model_file.pt")
        raise AssertionError("missing model file was not rejected")
    except FileNotFoundError:
        pass


# ------------------------------------------------------------------
# full app smoke tests, via streamlit's AppTest (no real browser needed)
# ------------------------------------------------------------------
def _save_model_and_data(tmp_path, model, X_df, y):
    model_path = str(tmp_path / "model.pt")
    model.save(model_path)
    data_path = str(tmp_path / "data.csv")
    X_df.assign(target=y).to_csv(data_path, index=False)
    return model_path, data_path


def test_app_runs_gam_with_data_and_edit_tab(tmp_path, monkeypatch):
    X, y = make_regression(n_samples=100, n_features=3, noise=0.1, random_state=0)
    X_df = pd.DataFrame(X, columns=["a", "b", "c"])
    model = KANBoostRegressor(
        n_estimators=4, kan_steps=4, kan_hidden=1, gam=True,
        early_stopping_rounds=None, random_state=0,
    )
    model.fit(X_df, y)
    model_path, data_path = _save_model_and_data(tmp_path, model, X_df, y)

    monkeypatch.setenv("KANBOOST_MODEL_PATH", model_path)
    monkeypatch.setenv("KANBOOST_DATA_PATH", data_path)

    at = st_testing.AppTest.from_file(_DASHBOARD_APP_PATH)
    at.run(timeout=60)
    assert not at.exception
    assert len(at.tabs) == 5  # Overview, Shape functions, Explain a row, Interactions, Edit

    apply_btn = next(b for b in at.button if b.label == "Apply offset")
    apply_btn.click().run(timeout=60)
    assert not at.exception

    enforce_btn = next(b for b in at.button if b.label == "Enforce increasing")
    enforce_btn.click().run(timeout=60)
    assert not at.exception

    save_path = str(tmp_path / "edited.pt")
    at.text_input[0].set_value(save_path).run(timeout=60)
    save_btn = next(b for b in at.button if b.label == "Save")
    save_btn.click().run(timeout=60)
    assert not at.exception

    from kanboost.interpret.editing import EditableGAM
    from kanboost.ops.serving import _load_any

    loaded = EditableGAM.load(save_path)
    assert loaded.feature_names == ["a", "b", "c"]

    # Not just "did it save something" -- the edit made through the UI
    # must actually be present in the saved file (guards against a
    # curves/_coef_cache desync that would save silently unmodified,
    # which would produce no exception and correct feature_names too).
    pristine_model = _load_any(model_path)
    assert not (loaded.predict(X_df) == pristine_model.predict(X_df)).all()


def test_app_runs_non_gam_without_edit_tab(tmp_path, monkeypatch):
    X, y = make_regression(n_samples=80, n_features=3, noise=0.1, random_state=0)
    X_df = pd.DataFrame(X, columns=["a", "b", "c"])
    model = KANBoostRegressor(
        n_estimators=3, kan_steps=3, kan_hidden=3, gam=False,
        early_stopping_rounds=None, random_state=0,
    )
    model.fit(X_df, y)
    model_path = str(tmp_path / "model.pt")
    model.save(model_path)

    monkeypatch.setenv("KANBOOST_MODEL_PATH", model_path)
    monkeypatch.delenv("KANBOOST_DATA_PATH", raising=False)

    at = st_testing.AppTest.from_file(_DASHBOARD_APP_PATH)
    at.run(timeout=60)
    assert not at.exception
    assert len(at.tabs) == 4  # no Edit tab: gam=False


def test_app_runs_multiclass_without_edit_tab(tmp_path, monkeypatch):
    X, y = make_classification(
        n_samples=150, n_features=4, n_informative=4, n_redundant=0,
        n_classes=3, n_clusters_per_class=1, random_state=1,
    )
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    model = KANBoostClassifier(
        n_estimators=3, kan_steps=3, kan_hidden=1, gam=True,
        early_stopping_rounds=None, random_state=0,
    )
    model.fit(X_df, y)
    model_path, data_path = _save_model_and_data(tmp_path, model, X_df, y)

    monkeypatch.setenv("KANBOOST_MODEL_PATH", model_path)
    monkeypatch.setenv("KANBOOST_DATA_PATH", data_path)

    at = st_testing.AppTest.from_file(_DASHBOARD_APP_PATH)
    at.run(timeout=60)
    assert not at.exception
    assert len(at.tabs) == 4  # no Edit tab: multiclass, even though gam=True
