"""
Tests for kanboost.train.consolidate (consolidate_learners()).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

from kanboost import KANBoostClassifier, KANBoostRegressor
from kanboost.train.consolidate import consolidate_learners


def _breast_cancer_splits(seed=0):
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)


def test_consolidate_shrinks_ensemble_and_preserves_accuracy():
    X_tr, X_te, y_tr, y_te = _breast_cancer_splits()

    model = KANBoostClassifier(
        n_estimators=30, kan_steps=20, early_stopping_rounds=None, random_state=0, verbose=False,
    )
    model.fit(X_tr, y_tr)
    n_before = model.best_iteration_
    auc_before = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])

    consolidate_learners(model, X_tr, group_size=5)

    assert model.best_iteration_ == -(-n_before // 5)  # ceil(n_before / 5)
    assert len(model.learners_) == model.best_iteration_
    auc_after = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
    # Consolidation is a lossy approximation -- allow a small accuracy cost,
    # not an exact-parity assertion.
    assert auc_after > auc_before - 0.02


def test_consolidate_group_size_larger_than_ensemble_is_noop():
    X_tr, X_te, y_tr, y_te = _breast_cancer_splits()
    model = KANBoostClassifier(
        n_estimators=10, kan_steps=15, early_stopping_rounds=None, random_state=0, verbose=False,
    )
    model.fit(X_tr, y_tr)
    n_before = model.best_iteration_

    consolidate_learners(model, X_tr, group_size=1000)

    assert model.best_iteration_ == n_before


def test_consolidate_multiclass_chains_stay_isolated():
    X, y = load_iris(return_X_y=True)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

    model = KANBoostClassifier(
        n_estimators=15, kan_steps=15, early_stopping_rounds=None, random_state=0, verbose=False,
    )
    model.fit(X_tr, y_tr)

    consolidate_learners(model, X_tr, group_size=5)

    assert len(model.learners_) == 3  # one chain per class
    assert all(v == 3 for v in model.best_iteration_.values())  # ceil(15/5) == 3
    assert accuracy_score(y_te, model.predict(X_te)) > 0.75


def test_consolidate_then_save_load_roundtrip(tmp_path):
    X_tr, X_te, y_tr, y_te = _breast_cancer_splits()
    model = KANBoostClassifier(
        n_estimators=20, kan_steps=15, early_stopping_rounds=None, random_state=0, verbose=False,
    )
    model.fit(X_tr, y_tr)
    consolidate_learners(model, X_tr, group_size=4)

    path = str(tmp_path / "model.pt")
    model.save(path)
    loaded = KANBoostClassifier.load(path)

    assert np.allclose(loaded.predict_proba(X_te), model.predict_proba(X_te))
    assert np.array_equal(loaded.predict(X_te), model.predict(X_te))


def test_consolidate_works_on_regressor():
    from sklearn.datasets import fetch_california_housing

    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names).iloc[:2000]
    y = data.target[:2000]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

    model = KANBoostRegressor(
        n_estimators=15, kan_steps=15, early_stopping_rounds=None, random_state=0, verbose=False,
    )
    model.fit(X_tr, y_tr)
    n_before = model.best_iteration_

    consolidate_learners(model, X_tr, group_size=5)

    assert model.best_iteration_ < n_before
    preds = model.predict(X_te)
    assert np.all(np.isfinite(preds))


if __name__ == "__main__":
    test_consolidate_shrinks_ensemble_and_preserves_accuracy()
    test_consolidate_group_size_larger_than_ensemble_is_noop()
    test_consolidate_multiclass_chains_stay_isolated()
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        test_consolidate_then_save_load_roundtrip(Path(d))
    test_consolidate_works_on_regressor()
    print("All consolidate tests passed.")
