"""
Tests for kanboost.registry.local (LocalRegistry).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from kanboost import KANBoostClassifier
from kanboost.registry.local import LocalRegistry


def _fitted_model(seed=0):
    X, y = make_classification(n_samples=200, n_features=6, random_state=seed)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed)
    model = KANBoostClassifier(n_estimators=10, kan_steps=8, early_stopping_rounds=None, random_state=seed)
    model.fit(X_tr, y_tr)
    return model, X_te, y_te


def test_register_and_get_latest(tmp_path):
    model, X_te, y_te = _fitted_model()
    reg = LocalRegistry(str(tmp_path))

    version = reg.register(model, "churn", tags={"stage": "dev"})
    assert version == 1

    loaded = reg.get("churn")  # default: latest
    assert np.allclose(loaded.predict_proba(X_te), model.predict_proba(X_te))


def test_multiple_versions_increment_and_get_by_number(tmp_path):
    model1, X_te, _ = _fitted_model(seed=0)
    model2, _, _ = _fitted_model(seed=1)
    reg = LocalRegistry(str(tmp_path))

    v1 = reg.register(model1, "churn")
    v2 = reg.register(model2, "churn")
    assert v1 == 1 and v2 == 2

    loaded_v1 = reg.get("churn", version=1)
    loaded_latest = reg.get("churn")  # should be v2
    assert np.allclose(loaded_v1.predict_proba(X_te), model1.predict_proba(X_te))
    assert np.allclose(loaded_latest.predict_proba(X_te), model2.predict_proba(X_te))


def test_list_versions_and_names(tmp_path):
    model, _, _ = _fitted_model()
    reg = LocalRegistry(str(tmp_path))
    reg.register(model, "churn", tags={"stage": "prod"})
    reg.register(model, "fraud")

    assert set(reg.list()) == {"churn", "fraud"}
    churn_versions = reg.list("churn")
    assert len(churn_versions) == 1
    assert churn_versions[0]["tags"] == {"stage": "prod"}
    assert "config" in churn_versions[0] and "timestamp" in churn_versions[0]


def test_get_unregistered_name_raises(tmp_path):
    reg = LocalRegistry(str(tmp_path))
    try:
        reg.get("nonexistent")
        raise AssertionError("getting an unregistered name was not rejected")
    except ValueError:
        pass


def test_get_nonexistent_version_raises(tmp_path):
    model, _, _ = _fitted_model()
    reg = LocalRegistry(str(tmp_path))
    reg.register(model, "churn")
    try:
        reg.get("churn", version=99)
        raise AssertionError("getting a nonexistent version was not rejected")
    except ValueError:
        pass


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    for test in [
        test_register_and_get_latest,
        test_multiple_versions_increment_and_get_by_number,
        test_list_versions_and_names,
        test_get_unregistered_name_raises,
        test_get_nonexistent_version_raises,
    ]:
        with tempfile.TemporaryDirectory() as d:
            test(Path(d))
    print("All local registry tests passed.")
