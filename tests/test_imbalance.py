"""
Tests for kanboost.imbalance (balanced_weights() / find_threshold()).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from kanboost import KANBoostClassifier
from kanboost.imbalance import balanced_weights, find_threshold


def _imbalanced_splits(seed=0):
    X, y = make_classification(
        n_samples=1200, n_features=10, n_informative=6,
        weights=[0.9, 0.1], flip_y=0.02, random_state=seed,
    )
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=seed)
    return X_tr, y_tr, X_val, y_val, X_te, y_te


def _fit_degenerate_model(X_tr, y_tr, X_val, y_val, sample_weight=None):
    model = KANBoostClassifier(
        gam=True, kan_hidden=1, n_estimators=40, kan_steps=10,
        early_stopping_rounds=10, random_state=0, verbose=False,
    )
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), sample_weight=sample_weight)
    return model


def test_balanced_weights_equalizes_class_mass():
    y = np.array([0] * 90 + [1] * 10)
    w = balanced_weights(y)
    assert abs(w[y == 0][0] * 90 - w[y == 1][0] * 10) < 1e-9
    assert w[y == 0][0] < w[y == 1][0]


def test_baseline_reproduces_degenerate_classifier():
    X_tr, y_tr, X_val, y_val, X_te, y_te = _imbalanced_splits()
    model = _fit_degenerate_model(X_tr, y_tr, X_val, y_val)
    report = model.evaluate(X_te, y_te, verbose=False)

    # The documented bug: high AUC (real signal) but F1=0 at the default
    # threshold, because a well-calibrated model on a 90/10 split puts
    # p < 0.5 almost everywhere.
    assert report["auc"] > 0.75
    assert report["f1"] == 0.0


def test_find_threshold_recovers_nonzero_f1_without_hurting_auc():
    X_tr, y_tr, X_val, y_val, X_te, y_te = _imbalanced_splits()
    model = _fit_degenerate_model(X_tr, y_tr, X_val, y_val)
    baseline = model.evaluate(X_te, y_te, verbose=False)

    t = find_threshold(model, X_val, y_val, metric="f1")
    tuned = model.evaluate(X_te, y_te, threshold=t, verbose=False)

    assert tuned["f1"] > 0.3
    assert tuned["f1"] > baseline["f1"]
    # threshold tuning only changes predict()'s cutoff, not predict_proba(),
    # so AUC (rank-based) must be exactly unchanged.
    assert abs(tuned["auc"] - baseline["auc"]) < 1e-9


def test_find_threshold_youden_runs_and_is_between_zero_and_one():
    X_tr, y_tr, X_val, y_val, X_te, y_te = _imbalanced_splits()
    model = _fit_degenerate_model(X_tr, y_tr, X_val, y_val)
    t = find_threshold(model, X_val, y_val, metric="youden")
    assert 0.0 <= t <= 1.0


def test_find_threshold_rejects_multiclass():
    from sklearn.datasets import load_iris
    import pandas as pd

    X, y = load_iris(return_X_y=True)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    model = KANBoostClassifier(n_estimators=5, kan_steps=5, early_stopping_rounds=None, random_state=0)
    model.fit(X_df, y)
    try:
        find_threshold(model, X_df, y)
        raise AssertionError("find_threshold on a multiclass model was not rejected")
    except ValueError:
        pass


def test_balanced_weights_does_not_regress_auc():
    X_tr, y_tr, X_val, y_val, X_te, y_te = _imbalanced_splits()
    baseline = _fit_degenerate_model(X_tr, y_tr, X_val, y_val)
    weighted = _fit_degenerate_model(X_tr, y_tr, X_val, y_val, sample_weight=balanced_weights(y_tr))

    auc_base = roc_auc_score(y_te, baseline.predict_proba(X_te)[:, 1])
    auc_weighted = roc_auc_score(y_te, weighted.predict_proba(X_te)[:, 1])
    assert auc_weighted > auc_base - 0.05


if __name__ == "__main__":
    test_balanced_weights_equalizes_class_mass()
    test_baseline_reproduces_degenerate_classifier()
    test_find_threshold_recovers_nonzero_f1_without_hurting_auc()
    test_find_threshold_youden_runs_and_is_between_zero_and_one()
    test_find_threshold_rejects_multiclass()
    test_balanced_weights_does_not_regress_auc()
    print("All imbalance tests passed.")
