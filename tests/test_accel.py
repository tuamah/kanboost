"""
Tests for kanboost.accel (fast_fit()).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

from kanboost import KANBoostClassifier
from kanboost.train.accel import fast_fit
from kanboost.interpret.experimental import audit_monotonicity


def _breast_cancer_splits(seed=0):
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)


def test_fast_fit_is_faster_and_preserves_accuracy():
    X_tr, X_te, y_tr, y_te = _breast_cancer_splits()

    m_normal = KANBoostClassifier(
        n_estimators=30, kan_steps=20, early_stopping_rounds=None, random_state=0, verbose=False,
    )
    t0 = time.time()
    m_normal.fit(X_tr, y_tr)
    t_normal = time.time() - t0
    auc_normal = roc_auc_score(y_te, m_normal.predict_proba(X_te)[:, 1])

    m_fast = KANBoostClassifier(
        n_estimators=30, kan_steps=20, early_stopping_rounds=None, random_state=0, verbose=False,
    )
    t0 = time.time()
    fast_fit(m_fast, X_tr, y_tr)
    t_fast = time.time() - t0
    auc_fast = roc_auc_score(y_te, m_fast.predict_proba(X_te)[:, 1])

    assert t_fast < t_normal
    assert auc_fast > auc_normal - 0.02


def test_fast_fit_restores_kan_steps_and_hooks():
    X_tr, X_te, y_tr, y_te = _breast_cancer_splits()
    model = KANBoostClassifier(
        n_estimators=10, kan_steps=12, early_stopping_rounds=None, random_state=0, verbose=False,
    )
    assert model._learner_init_hook is None
    assert model._boost_chain_start_hook is None

    fast_fit(model, X_tr, y_tr)

    assert model.kan_steps == 12
    # fast_fit() must clear its hooks after fit() returns -- leaving a
    # closure behind would break model.save()'s pickling later.
    assert model._learner_init_hook is None
    assert model._boost_chain_start_hook is None


def test_fast_fit_respects_monotone_constraints():
    X_tr, X_te, y_tr, y_te = _breast_cancer_splits()
    feat = X_tr.columns[0]
    model = KANBoostClassifier(
        gam=True, kan_hidden=1, n_estimators=20, kan_steps=20,
        early_stopping_rounds=None, random_state=0, verbose=False,
        monotone_constraints={feat: 1},
    )
    fast_fit(model, X_tr, y_tr)
    audit = audit_monotonicity(model, X_tr)
    assert audit[feat]["passed"]
    assert audit[feat]["violation_rate"] == 0.0


def test_fast_fit_multiclass_chains_stay_isolated():
    X, y = load_iris(return_X_y=True)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

    model = KANBoostClassifier(
        n_estimators=12, kan_steps=12, early_stopping_rounds=None, random_state=0, verbose=False,
    )
    fast_fit(model, X_tr, y_tr)

    assert len(model.learners_) == 3
    # every chain must get its own full-length set of learners -- if warm
    # start state leaked across one-vs-rest chains, later chains would
    # warm-start their first learner from a different class's last one.
    assert all(len(v) == 12 for v in model.learners_.values())
    assert accuracy_score(y_te, model.predict(X_te)) > 0.85


def test_fast_fit_then_save_load_roundtrip(tmp_path):
    X_tr, X_te, y_tr, y_te = _breast_cancer_splits()
    model = KANBoostClassifier(
        n_estimators=10, kan_steps=12, early_stopping_rounds=None, random_state=0, verbose=False,
    )
    fast_fit(model, X_tr, y_tr)

    # fast_fit() sets _learner_init_hook/_boost_chain_start_hook to
    # closures for the duration of one fit() call; save() must not still
    # find a closure there afterward, or pickling self.__dict__ (which
    # includes these two hook attributes, always present with a None
    # default -- see core/base.py) fails.
    assert model._learner_init_hook is None
    assert model._boost_chain_start_hook is None

    path = str(tmp_path / "model.pt")
    model.save(path)
    loaded = KANBoostClassifier.load(path)

    assert np.allclose(loaded.predict_proba(X_te), model.predict_proba(X_te))
    assert np.array_equal(loaded.predict(X_te), model.predict(X_te))


def test_fast_fit_custom_warm_start_steps():
    X_tr, X_te, y_tr, y_te = _breast_cancer_splits()
    model = KANBoostClassifier(
        n_estimators=15, kan_steps=20, early_stopping_rounds=None, random_state=0, verbose=False,
    )
    fast_fit(model, X_tr, y_tr, warm_start_steps=5)
    assert model.kan_steps == 20
    auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
    assert auc > 0.9


if __name__ == "__main__":
    test_fast_fit_is_faster_and_preserves_accuracy()
    test_fast_fit_restores_kan_steps_and_hooks()
    test_fast_fit_respects_monotone_constraints()
    test_fast_fit_multiclass_chains_stay_isolated()
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        test_fast_fit_then_save_load_roundtrip(Path(d))
    test_fast_fit_custom_warm_start_steps()
    print("All accel tests passed.")
