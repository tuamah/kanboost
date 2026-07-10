"""
Tests for kanboost.config (KANConfig / BoostConfig / KANBoostConfig).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kanboost.core.config import KANConfig, BoostConfig, KANBoostConfig


def test_from_flat_to_flat_roundtrip():
    flat = dict(
        n_estimators=50, learning_rate=0.2, kan_hidden=1, kan_grid=5, kan_k=3,
        kan_steps=15, kan_lr=0.01, early_stopping_rounds=5, validation_fraction=0.1,
        categorical_cols=["a"], device="cpu", batch_size=32, gam=True,
        monotone_constraints={"a": 1}, lamb=0.1, lamb_l1=0.9, lamb_coefdiff=0.2,
        random_state=7, verbose=True, objective="squared_error", alpha=0.5,
    )
    cfg = KANBoostConfig.from_flat(**flat)
    assert cfg.kan.hidden == 1 and cfg.kan.grid == 5
    assert cfg.boost.n_estimators == 50
    assert cfg.gam is True
    assert cfg.to_flat() == flat


def test_from_flat_uses_defaults_for_missing_kwargs():
    cfg = KANBoostConfig.from_flat(n_estimators=10)
    assert cfg.kan == KANConfig()  # defaults untouched
    assert cfg.boost.n_estimators == 10


def test_to_dict_from_dict_roundtrip():
    cfg = KANBoostConfig.from_flat(kan_hidden=1, gam=True, monotone_constraints={"x": -1})
    d = cfg.to_dict()
    restored = KANBoostConfig.from_dict(d)
    assert restored == cfg


def test_validates_monotone_constraints_requires_gam_and_hidden_1():
    try:
        KANBoostConfig.from_flat(monotone_constraints={"x": 1}, gam=False)
        raise AssertionError("gam=False with monotone_constraints was not rejected")
    except ValueError:
        pass

    try:
        KANBoostConfig.from_flat(monotone_constraints={"x": 1}, gam=True, kan_hidden=2)
        raise AssertionError("kan_hidden!=1 with monotone_constraints was not rejected")
    except ValueError:
        pass

    # valid combination does not raise
    KANBoostConfig.from_flat(monotone_constraints={"x": 1}, gam=True, kan_hidden=1)


def test_validates_basic_ranges():
    for bad_kwargs in [
        {"n_estimators": 0}, {"learning_rate": 0}, {"learning_rate": 1.5},
        {"kan_hidden": 0}, {"validation_fraction": 1.0}, {"batch_size": 0},
    ]:
        try:
            KANBoostConfig.from_flat(**bad_kwargs)
            raise AssertionError(f"{bad_kwargs} was not rejected")
        except ValueError:
            pass


if __name__ == "__main__":
    test_from_flat_to_flat_roundtrip()
    test_from_flat_uses_defaults_for_missing_kwargs()
    test_to_dict_from_dict_roundtrip()
    test_validates_monotone_constraints_requires_gam_and_hidden_1()
    test_validates_basic_ranges()
    print("All config tests passed.")
