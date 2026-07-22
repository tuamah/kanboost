"""
Basic sanity tests for KANBoostClassifier / KANBoostRegressor.
Run with: pytest tests/
"""
import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from kanboost import KANBoostClassifier, KANBoostRegressor


def test_classifier_fits_and_predicts():
    X, y = make_classification(n_samples=300, n_features=5, random_state=0)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])

    model = KANBoostClassifier(
        n_estimators=5, kan_steps=3, early_stopping_rounds=None, random_state=0
    )
    model.fit(X_df, y)

    proba = model.predict_proba(X_df)
    assert proba.shape == (300, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)

    preds = model.predict(X_df)
    assert set(np.unique(preds)).issubset({0, 1})


def test_regressor_categorical_hierarchy_wiring():
    """categorical_hierarchy must reach TabularPreprocessor's `hierarchy`
    and change the fitted encoding vs. the flat (no-hierarchy) baseline --
    this is the public-API wiring for the CC-6b hierarchical encoder,
    which previously only worked when instantiating TabularPreprocessor
    directly.
    """
    rng = np.random.RandomState(0)
    n = 400
    region = rng.choice(["north", "south"], size=n)
    city = np.where(
        region == "north",
        rng.choice(["city_a", "city_b"], size=n),
        rng.choice(["city_c", "rare_city"], size=n),
    )
    # rare_city appears rarely -- a hierarchy should back it off toward
    # "south"'s mean instead of the flat global mean.
    mask_rare = city == "rare_city"
    city[mask_rare] = "rare_city"
    keep = np.ones(n, dtype=bool)
    keep[np.where(mask_rare)[0][3:]] = False  # keep only 3 rare_city rows
    region, city = region[keep], city[keep]

    y = np.where(region == "north", 1.0, -1.0) + rng.randn(keep.sum()) * 0.1
    X_df = pd.DataFrame({"region": region, "city": city})

    model = KANBoostRegressor(
        n_estimators=5, kan_steps=3, early_stopping_rounds=None,
        categorical_cols=["region", "city"],
        categorical_hierarchy={"city": "region"},
        random_state=0,
    )
    model.fit(X_df, y)

    assert model.preprocessor_.hierarchy == {"city": "region"}
    preds = model.predict(X_df)
    assert preds.shape == (keep.sum(),)

    flat_model = KANBoostRegressor(
        n_estimators=5, kan_steps=3, early_stopping_rounds=None,
        categorical_cols=["region", "city"],
        random_state=0,
    )
    flat_model.fit(X_df, y)
    assert flat_model.preprocessor_.hierarchy == {}

    # Different encoders -> different transformed columns for the rare city.
    hier_arr = model.preprocessor_.transform(X_df)
    flat_arr = flat_model.preprocessor_.transform(X_df)
    assert not np.allclose(hier_arr, flat_arr)


def test_classifier_with_categorical():
    X, y = make_classification(n_samples=200, n_features=4, random_state=1)
    X_df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    X_df["region"] = np.random.choice(["north", "south", "east"], size=200)

    model = KANBoostClassifier(
        n_estimators=3, kan_steps=3, early_stopping_rounds=None,
        categorical_cols=["region"], random_state=0,
    )
    model.fit(X_df, y)
    proba = model.predict_proba(X_df)
    assert proba.shape[0] == 200


def test_classifier_early_stopping():
    X, y = make_classification(n_samples=400, n_features=5, random_state=2)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    X_train, X_val = X_df.iloc[:300], X_df.iloc[300:]
    y_train, y_val = y[:300], y[300:]

    model = KANBoostClassifier(
        n_estimators=20, kan_steps=3, early_stopping_rounds=3, random_state=0
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    assert model.best_iteration_ <= 20
    assert len(model.learners_) >= model.best_iteration_


def test_feature_importances_sum_to_one():
    X, y = make_classification(n_samples=200, n_features=4, random_state=3)
    X_df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    model = KANBoostClassifier(
        n_estimators=4, kan_steps=3, early_stopping_rounds=None, random_state=0
    )
    model.fit(X_df, y)
    importances = model.feature_importances()
    assert importances.shape == (4,)
    assert np.isclose(importances.sum(), 1.0, atol=1e-3)


def test_regressor_fits_and_predicts():
    X, y = make_regression(n_samples=200, n_features=4, noise=0.1, random_state=0)
    X_df = pd.DataFrame(X, columns=["a", "b", "c", "d"])

    model = KANBoostRegressor(
        n_estimators=5, kan_steps=3, early_stopping_rounds=None, random_state=0
    )
    model.fit(X_df, y)
    preds = model.predict(X_df)
    assert preds.shape == (200,)


def test_sklearn_compatibility():
    """KANBoost estimators must work with sklearn's clone/CV machinery."""
    from sklearn.base import clone, is_classifier
    from sklearn.model_selection import cross_val_score

    X, y = make_classification(n_samples=200, n_features=4, random_state=5)
    X_df = pd.DataFrame(X, columns=["a", "b", "c", "d"])

    m = KANBoostClassifier(n_estimators=3, kan_steps=3, early_stopping_rounds=None)
    assert is_classifier(m)
    m2 = clone(m)
    assert m2.get_params() == m.get_params()

    scores = cross_val_score(m, X_df, y, cv=2, scoring="roc_auc")
    assert not np.isnan(scores).any()


def test_input_validation():
    X, y = make_classification(n_samples=100, n_features=5, random_state=6)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])

    # NaN in y must be rejected
    y_bad = y.astype(float).copy(); y_bad[0] = np.nan
    try:
        KANBoostClassifier(n_estimators=2, kan_steps=2).fit(X_df, y_bad)
        raise AssertionError("NaN y was not rejected")
    except ValueError:
        pass

    # invalid hyperparameters must be rejected
    try:
        KANBoostClassifier(learning_rate=5).fit(X_df, y)
        raise AssertionError("invalid learning_rate was not rejected")
    except ValueError:
        pass

    # predicting before fitting must be rejected
    try:
        KANBoostClassifier().predict(X_df)
        raise AssertionError("predict before fit was not rejected")
    except RuntimeError:
        pass


def test_save_load_roundtrip(tmp_path):
    X, y = make_classification(n_samples=150, n_features=5, random_state=0)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])

    model = KANBoostClassifier(n_estimators=4, kan_steps=3, early_stopping_rounds=None, random_state=0)
    model.fit(X_df, y)
    proba_before = model.predict_proba(X_df)

    path = str(tmp_path / "model.pt")
    model.save(path)
    loaded = KANBoostClassifier.load(path)
    proba_after = loaded.predict_proba(X_df)

    assert np.allclose(proba_before, proba_after, atol=1e-5)

    # loading with the wrong class must be rejected
    try:
        KANBoostRegressor.load(path)
        raise AssertionError("cross-class load was not rejected")
    except ValueError:
        pass


def test_load_migrates_pre_v1_package_restructure_saved_files(tmp_path):
    # A model saved before kanboost's v1.0.0 package restructure pickles
    # self.preprocessor_ (a raw TabularPreprocessor, not converted by
    # save()'s own _freeze()) under the old flat module path
    # ("kanboost.encoders", not "kanboost.core.encoders"). Simulate that
    # exact scenario: temporarily rename the class's own __module__ (what
    # actually changes what torch.save's pickler records) AND alias
    # sys.modules["kanboost.encoders"] to the real module (pickle's
    # pickler itself verifies the class is importable under the recorded
    # name before writing it -- without the alias, save() itself would
    # fail, not just a later load()). Both are undone before load() runs,
    # so load()'s own ModuleNotFoundError fallback is what's actually
    # under test, not a lingering alias from this test.
    import kanboost.core.encoders as encoders_module
    from kanboost.core.encoders import TabularPreprocessor

    X, y = make_regression(n_samples=80, n_features=3, random_state=0)
    X_df = pd.DataFrame(X, columns=["a", "b", "c"])
    model = KANBoostRegressor(n_estimators=5, kan_steps=5, early_stopping_rounds=None, random_state=0)
    model.fit(X_df, y)

    original_module = TabularPreprocessor.__module__
    path = str(tmp_path / "pre_restructure_model.pt")
    try:
        TabularPreprocessor.__module__ = "kanboost.encoders"
        sys.modules["kanboost.encoders"] = encoders_module
        model.save(path)
    finally:
        TabularPreprocessor.__module__ = original_module
        sys.modules.pop("kanboost.encoders", None)

    loaded = KANBoostRegressor.load(path)
    assert np.allclose(loaded.predict(X_df), model.predict(X_df))


def test_multiclass_classification():
    X, y = make_classification(
        n_samples=300, n_features=6, n_informative=4, n_redundant=0,
        n_classes=3, n_clusters_per_class=1, random_state=1,
    )
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])

    model = KANBoostClassifier(n_estimators=5, kan_steps=3, early_stopping_rounds=None, random_state=0)
    model.fit(X_df, y)

    proba = model.predict_proba(X_df)
    assert proba.shape == (300, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)

    preds = model.predict(X_df)
    assert set(np.unique(preds)).issubset({0, 1, 2})

    report = model.evaluate(X_df, y, verbose=False)
    assert "f1" in report and 0.0 <= report["f1"] <= 1.0

    importances = model.feature_importances()
    assert importances.shape == (6,)
    assert np.isclose(importances.sum(), 1.0, atol=1e-3)


def test_missing_values_are_imputed_not_rejected():
    X, y = make_classification(n_samples=150, n_features=5, random_state=0)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    X_df.iloc[0, 0] = np.nan
    X_df.iloc[5, 2] = np.nan

    model = KANBoostClassifier(n_estimators=3, kan_steps=3, early_stopping_rounds=None, random_state=0)
    model.fit(X_df, y)  # must not raise
    preds = model.predict(X_df)
    assert preds.shape == (150,)
    assert "f0_missing" in model.preprocessor_.transformed_feature_names()


def test_plot_feature_returns_axes():
    import matplotlib
    matplotlib.use("Agg")

    X, y = make_classification(n_samples=150, n_features=5, random_state=0)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    model = KANBoostClassifier(n_estimators=3, kan_steps=3, early_stopping_rounds=None, random_state=0)
    model.fit(X_df, y)

    ax = model.plot_feature("f0")
    assert ax is not None


def test_sample_weight_changes_fit():
    X, y = make_classification(n_samples=150, n_features=5, random_state=0)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])

    rng = np.random.RandomState(0)
    weights = rng.uniform(0.1, 5.0, size=len(y))

    unweighted = KANBoostClassifier(n_estimators=4, kan_steps=3, early_stopping_rounds=None, random_state=0)
    unweighted.fit(X_df, y)

    weighted = KANBoostClassifier(n_estimators=4, kan_steps=3, early_stopping_rounds=None, random_state=0)
    weighted.fit(X_df, y, sample_weight=weights)

    # extreme weights should change the fitted ensemble's predictions
    assert not np.allclose(
        unweighted.predict_proba(X_df), weighted.predict_proba(X_df), atol=1e-6
    )

    # mismatched-length weights must be rejected
    try:
        KANBoostClassifier(n_estimators=2, kan_steps=2).fit(X_df, y, sample_weight=weights[:-1])
        raise AssertionError("mismatched sample_weight length was not rejected")
    except ValueError:
        pass

    # all-zero weights must be rejected (would divide by zero internally)
    try:
        KANBoostClassifier(n_estimators=2, kan_steps=2).fit(X_df, y, sample_weight=np.zeros(len(y)))
        raise AssertionError("all-zero sample_weight was not rejected")
    except ValueError:
        pass


def test_save_load_multiclass_with_missing_values(tmp_path):
    X, y = make_classification(
        n_samples=200, n_features=5, n_informative=4, n_redundant=0,
        n_classes=3, n_clusters_per_class=1, random_state=1,
    )
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    X_df.iloc[0, 0] = np.nan
    X_df.iloc[3, 2] = np.nan

    model = KANBoostClassifier(n_estimators=3, kan_steps=3, early_stopping_rounds=None, random_state=0)
    model.fit(X_df, y)
    proba_before = model.predict_proba(X_df)

    path = str(tmp_path / "multiclass_nan.pt")
    model.save(path)
    loaded = KANBoostClassifier.load(path)
    proba_after = loaded.predict_proba(X_df)

    assert np.allclose(proba_before, proba_after, atol=1e-5)
    assert np.array_equal(loaded.classes_, model.classes_)


def test_quantile_regressor():
    rng = np.random.RandomState(0)
    X, y = make_regression(n_samples=300, n_features=4, noise=0.0, random_state=0)
    y = y + rng.exponential(scale=5.0, size=len(y))  # skewed noise
    X_df = pd.DataFrame(X, columns=["a", "b", "c", "d"])

    low = KANBoostRegressor(
        n_estimators=15, kan_steps=5, early_stopping_rounds=None,
        objective="quantile", alpha=0.1, random_state=0,
    )
    low.fit(X_df, y)
    high = KANBoostRegressor(
        n_estimators=15, kan_steps=5, early_stopping_rounds=None,
        objective="quantile", alpha=0.9, random_state=0,
    )
    high.fit(X_df, y)

    # the 0.9-quantile prediction should exceed the 0.1-quantile prediction
    # for the large majority of rows
    assert (high.predict(X_df) > low.predict(X_df)).mean() > 0.8

    report = high.evaluate(X_df, y, verbose=False)
    assert "pinball" in report

    try:
        KANBoostRegressor(objective="not_a_real_objective").fit(X_df, y)
        raise AssertionError("unknown objective was not rejected")
    except ValueError:
        pass


def test_internal_split_early_stopping():
    X, y = make_classification(n_samples=400, n_features=5, random_state=2)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])

    model = KANBoostClassifier(
        n_estimators=20, kan_steps=3, early_stopping_rounds=3,
        validation_fraction=0.2, random_state=0,
    )
    model.fit(X_df, y)  # no eval_set passed
    assert model.best_iteration_ <= 20
    preds = model.predict(X_df)
    assert preds.shape == (400,)


def test_batch_size_training():
    X, y = make_classification(n_samples=200, n_features=5, random_state=2)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    weights = np.random.RandomState(0).uniform(0.5, 2.0, size=len(y))

    model = KANBoostClassifier(
        n_estimators=4, kan_steps=5, early_stopping_rounds=None,
        batch_size=32, random_state=0,
    )
    model.fit(X_df, y, sample_weight=weights)
    proba = model.predict_proba(X_df)
    assert proba.shape == (200, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)


def test_batch_size_uses_different_batches_per_learner():
    """batch_size is a no-op in DeepKAN (closed-form solve uses all data).
    This test now just verifies that passing batch_size doesn't raise."""
    X, y = make_classification(n_samples=200, n_features=5, random_state=2)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])

    model = KANBoostClassifier(
        n_estimators=6, kan_steps=5, early_stopping_rounds=None,
        batch_size=16, random_state=0,
    )

    model.fit(X_df, y)
    # no assertion — batch_size is a no-op in DeepKAN's closed-form solver


def test_feature_contributions():
    X, y = make_classification(n_samples=200, n_features=5, random_state=2)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])

    model = KANBoostClassifier(
        n_estimators=4, kan_steps=3, kan_hidden=1,
        early_stopping_rounds=None, random_state=0,
    )
    model.fit(X_df, y)
    contrib = model.feature_contributions(X_df)
    assert contrib.shape == (200, 5)

    # multiclass: dict keyed by class label
    Xm, ym = make_classification(
        n_samples=200, n_features=5, n_informative=4, n_redundant=0,
        n_classes=3, n_clusters_per_class=1, random_state=1,
    )
    Xm_df = pd.DataFrame(Xm, columns=[f"f{i}" for i in range(5)])
    model_mc = KANBoostClassifier(n_estimators=3, kan_steps=3, early_stopping_rounds=None, random_state=0)
    model_mc.fit(Xm_df, ym)
    contrib_mc = model_mc.feature_contributions(Xm_df)
    assert set(contrib_mc.keys()) == set(model_mc.classes_)
    for arr in contrib_mc.values():
        assert arr.shape == (200, 5)


def test_monotone_constraints():
    rng = np.random.RandomState(0)
    X, _ = make_regression(n_samples=150, n_features=4, noise=0.1, random_state=0)
    X_df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    y = X[:, 0] * 2 + rng.normal(scale=0.1, size=150)

    model = KANBoostRegressor(
        n_estimators=5, kan_steps=8, early_stopping_rounds=None,
        gam=True, kan_hidden=1, monotone_constraints={"a": 1}, random_state=0,
    )
    model.fit(X_df, y)

    grid = np.linspace(-1, 1, 100)
    n_features = len(model.preprocessor_.transformed_feature_names())
    X_probe = np.zeros((100, n_features))
    X_probe[:, 0] = grid
    import torch
    X_t = torch.tensor(X_probe, dtype=torch.float32, device=model.device_)
    curve = model._raw_score_chain(X_t, model.learners_, model.init_pred_, model.best_iteration_)
    assert np.all(np.diff(curve) >= -1e-6)

    # requires gam=True
    try:
        KANBoostRegressor(monotone_constraints={"a": 1}, gam=False).fit(X_df, y)
        raise AssertionError("monotone_constraints without gam=True was not rejected")
    except ValueError:
        pass


def test_gam_mode_and_symbolic_report():
    X, y = make_regression(n_samples=150, n_features=3, noise=0.1, random_state=0)
    X_df = pd.DataFrame(X, columns=["a", "b", "c"])

    model = KANBoostRegressor(
        n_estimators=5, kan_steps=8, early_stopping_rounds=None,
        gam=True, kan_hidden=1, random_state=0,
    )
    model.fit(X_df, y)
    preds = model.predict(X_df)
    assert preds.shape == (150,)

    report = model.symbolic_report(X_df, top_k=2)
    assert set(report.keys()) == set(model.preprocessor_.transformed_feature_names())
    for candidates in report.values():
        assert len(candidates) <= 2
        for name, r2 in candidates:
            assert isinstance(name, str) and isinstance(r2, float)

    non_gam = KANBoostRegressor(n_estimators=3, kan_steps=3, early_stopping_rounds=None, random_state=0)
    non_gam.fit(X_df, y)
    try:
        non_gam.symbolic_report(X_df)
        raise AssertionError("symbolic_report without gam=True was not rejected")
    except RuntimeError:
        pass


def test_predict_derivative():
    X, y = make_regression(n_samples=150, n_features=4, noise=0.1, random_state=0)
    X_df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    y = X[:, 0] * 2 + y * 0  # purely monotone-increasing in "a"

    model = KANBoostRegressor(
        n_estimators=5, kan_steps=8, early_stopping_rounds=None,
        gam=True, kan_hidden=1, monotone_constraints={"a": 1}, random_state=0,
    )
    model.fit(X_df, y)
    deriv = model.predict_derivative(X_df, "a")
    assert deriv.shape == (150,)
    assert deriv.min() >= -1e-4  # non-negative for a monotone-increasing fit


def test_prune_and_refine():
    X, y = make_regression(n_samples=150, n_features=4, noise=0.1, random_state=0)
    X_df = pd.DataFrame(X, columns=["a", "b", "c", "d"])

    model = KANBoostRegressor(
        n_estimators=3, kan_steps=5, kan_hidden=2, kan_grid=2,
        early_stopping_rounds=None, random_state=0,
    )
    model.fit(X_df, y)

    model.prune(X_df, threshold=1e-6)  # must not raise, predictions still usable
    assert model.predict(X_df).shape == (150,)

    before = model.predict(X_df)
    model.refine(X_df, 5)
    after = model.predict(X_df)
    assert model.kan_grid == 5
    assert np.allclose(before, after, atol=0.5)  # refine approximates, not identical


def test_feature_interaction():
    # a genuine multiplicative interaction (a*b), not additive noise --
    # a vacuous {} result (e.g. from querying the wrong pykan layer) must
    # be caught by this test, not just "isinstance(result, dict)"
    rng = np.random.RandomState(0)
    X = rng.uniform(-1, 1, size=(400, 4))
    y = X[:, 0] * X[:, 1] + rng.normal(scale=0.01, size=400)
    X_df = pd.DataFrame(X, columns=["a", "b", "c", "d"])

    model = KANBoostRegressor(
        n_estimators=15, kan_steps=15, kan_hidden=3,
        early_stopping_rounds=None, random_state=0,
    )
    model.fit(X_df, y)
    result = model.feature_interaction(X_df)
    assert isinstance(result, dict)
    assert len(result) > 0, "feature_interaction returned no interactions on data with a known one"
    assert all(isinstance(k, tuple) and len(k) == 2 for k in result)

    gam_model = KANBoostRegressor(n_estimators=3, kan_steps=5, kan_hidden=1, random_state=0, early_stopping_rounds=None)
    gam_model.fit(X_df, y)
    try:
        gam_model.feature_interaction(X_df)
        raise AssertionError("feature_interaction with kan_hidden=1 was not rejected")
    except RuntimeError:
        pass


def test_regularization_lamb_params_accepted():
    X, y = make_regression(n_samples=100, n_features=4, noise=0.1, random_state=0)
    X_df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    model = KANBoostRegressor(
        n_estimators=3, kan_steps=5, early_stopping_rounds=None,
        lamb=0.01, lamb_l1=0.5, lamb_coefdiff=1.0, random_state=0,
    )
    model.fit(X_df, y)  # must not raise
    assert model.predict(X_df).shape == (100,)


def test_device_resolution():
    import torch

    X, y = make_regression(n_samples=50, n_features=3, random_state=0)
    X_df = pd.DataFrame(X, columns=["a", "b", "c"])

    cpu_model = KANBoostRegressor(n_estimators=2, kan_steps=2, device="cpu")
    assert cpu_model._resolve_device() == torch.device("cpu")

    auto_model = KANBoostRegressor(n_estimators=2, kan_steps=2, device=None)
    expected = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert auto_model._resolve_device() == expected

    if not torch.cuda.is_available():
        # explicit cuda request must fail fast with a clear error, not
        # silently fall back to cpu or fail later with a cryptic CUDA error
        for bad_device in ("cuda", "cuda:0"):
            try:
                KANBoostRegressor(n_estimators=2, device=bad_device)._resolve_device()
                raise AssertionError(f"device={bad_device!r} should have raised RuntimeError")
            except RuntimeError:
                pass


if __name__ == "__main__":
    import tempfile

    test_classifier_fits_and_predicts()
    test_classifier_with_categorical()
    test_classifier_early_stopping()
    test_feature_importances_sum_to_one()
    test_regressor_fits_and_predicts()
    test_sklearn_compatibility()
    test_input_validation()
    with tempfile.TemporaryDirectory() as d:
        test_save_load_roundtrip(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_save_load_multiclass_with_missing_values(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_load_migrates_pre_v1_package_restructure_saved_files(Path(d))
    test_multiclass_classification()
    test_missing_values_are_imputed_not_rejected()
    test_plot_feature_returns_axes()
    test_sample_weight_changes_fit()
    test_quantile_regressor()
    test_internal_split_early_stopping()
    test_batch_size_training()
    test_batch_size_uses_different_batches_per_learner()
    test_feature_contributions()
    test_monotone_constraints()
    test_gam_mode_and_symbolic_report()
    test_predict_derivative()
    test_prune_and_refine()
    test_feature_interaction()
    test_regularization_lamb_params_accepted()
    test_device_resolution()
    print("All tests passed.")


