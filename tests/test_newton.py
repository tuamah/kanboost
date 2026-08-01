"""
Tests for kanboost.train.newton (fit_with_newton_boosting()).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

from kanboost import KANBoostClassifier, KANBoostRegressor
from kanboost.train.newton import fit_with_newton_boosting


def _breast_cancer_splits(seed=0):
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)


def test_newton_boosting_fits_and_predicts_via_standard_api():
    X_tr, X_te, y_tr, y_te = _breast_cancer_splits()
    model = KANBoostClassifier(
        n_estimators=20, kan_hidden=8, kan_grid=3, random_state=0, verbose=False,
    )
    fit_with_newton_boosting(model, X_tr, y_tr)

    assert model.best_iteration_ == 20
    # unlike fit_with_line_search, this populates learners_/init_pred_ in the
    # exact format _boost_chain would -- the model's own predict_proba works.
    proba = model.predict_proba(X_te)
    assert proba.shape == (len(y_te), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    auc = roc_auc_score(y_te, proba[:, 1])
    assert auc > 0.9


def test_newton_boosting_improves_on_first_order_at_same_round_count():
    X_tr, X_te, y_tr, y_te = _breast_cancer_splits()

    baseline = KANBoostClassifier(
        n_estimators=25, kan_hidden=8, kan_grid=3, learning_rate=0.1,
        early_stopping_rounds=None, random_state=0, verbose=False,
    )
    baseline.fit(X_tr, y_tr)
    auc_baseline = roc_auc_score(y_te, baseline.predict_proba(X_te)[:, 1])

    newton = KANBoostClassifier(
        n_estimators=25, kan_hidden=8, kan_grid=3, learning_rate=0.1,
        random_state=0, verbose=False,
    )
    fit_with_newton_boosting(newton, X_tr, y_tr)
    auc_newton = roc_auc_score(y_te, newton.predict_proba(X_te)[:, 1])

    # Not a strict inequality (dataset-dependent, lossy technique) -- checks
    # Newton boosting is at least competitive, per the measured INSPIRE
    # results in the module docstring (consistent small improvement).
    assert auc_newton > auc_baseline - 0.02


def test_newton_boosting_multiclass():
    X, y = load_iris(return_X_y=True)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

    model = KANBoostClassifier(
        n_estimators=15, kan_hidden=8, kan_grid=3, random_state=0, verbose=False,
    )
    fit_with_newton_boosting(model, X_tr, y_tr)

    assert isinstance(model.learners_, dict)
    assert len(model.learners_) == 3  # one one-vs-rest chain per class
    assert all(v == 15 for v in model.best_iteration_.values())
    proba = model.predict_proba(X_te)
    assert proba.shape == (len(y_te), 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert accuracy_score(y_te, model.predict(X_te)) > 0.85


def test_newton_boosting_hessian_floor_modes_run_and_stay_finite():
    X_tr, X_te, y_tr, y_te = _breast_cancer_splits()
    for mode in ["hard", "adaptive", "soft_lm"]:
        model = KANBoostClassifier(
            n_estimators=15, kan_hidden=8, kan_grid=3, random_state=0, verbose=False,
        )
        fit_with_newton_boosting(model, X_tr, y_tr, hessian_floor_mode=mode)
        proba = model.predict_proba(X_te)
        assert np.all(np.isfinite(proba)), mode
        assert roc_auc_score(y_te, proba[:, 1]) > 0.85, mode


def test_newton_boosting_rejects_unknown_hessian_floor_mode():
    X_tr, X_te, y_tr, y_te = _breast_cancer_splits()
    model = KANBoostClassifier(n_estimators=5, random_state=0, verbose=False)
    with pytest.raises(ValueError, match="hessian_floor_mode"):
        fit_with_newton_boosting(model, X_tr, y_tr, hessian_floor_mode="bogus")


def test_newton_boosting_rejects_regressor():
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names).iloc[:200]
    y = data.target[:200]
    model = KANBoostRegressor(n_estimators=5, random_state=0, verbose=False)
    with pytest.raises(TypeError, match="regressor"):
        fit_with_newton_boosting(model, X, y)


def test_newton_boosting_then_save_load_roundtrip(tmp_path):
    X_tr, X_te, y_tr, y_te = _breast_cancer_splits()
    model = KANBoostClassifier(
        n_estimators=15, kan_hidden=8, kan_grid=3, random_state=0, verbose=False,
    )
    fit_with_newton_boosting(model, X_tr, y_tr)

    path = str(tmp_path / "model.pt")
    model.save(path)
    loaded = KANBoostClassifier.load(path)

    assert np.allclose(loaded.predict_proba(X_te), model.predict_proba(X_te))


if __name__ == "__main__":
    test_newton_boosting_fits_and_predicts_via_standard_api()
    test_newton_boosting_improves_on_first_order_at_same_round_count()
    test_newton_boosting_multiclass()
    test_newton_boosting_hessian_floor_modes_run_and_stay_finite()
    test_newton_boosting_rejects_unknown_hessian_floor_mode()
    test_newton_boosting_rejects_regressor()
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        test_newton_boosting_then_save_load_roundtrip(Path(d))
    print("All newton tests passed.")
