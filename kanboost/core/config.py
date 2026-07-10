"""
kanboost.core.config -- typed configuration for KANBoostClassifier/Regressor,
built on stdlib dataclasses (no new dependency).

Internal source of truth for a model's hyperparameters, grouped by
concern (KAN architecture, boosting loop, everything else) instead of
one flat ~19-kwarg constructor. The flat kwarg surface on
KANBoostClassifier/Regressor stays exactly as-is -- required for
sklearn's clone()/get_params()/set_params() and for kantun's
KantunSearch, both of which need a flat namespace -- via
from_flat()/to_flat(). This module is a grouping/validation layer, not
a breaking change to the constructor by itself.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field


@dataclass
class KANConfig:
    hidden: int = 3
    grid: int = 2
    k: int = 3
    steps: int = 20
    lr: float = 0.02
    lamb: float = 0.0
    lamb_l1: float = 1.0
    lamb_coefdiff: float = 0.0


@dataclass
class BoostConfig:
    n_estimators: int = 100
    learning_rate: float = 0.1
    early_stopping_rounds: int | None = 10
    validation_fraction: float | None = None


@dataclass
class KANBoostConfig:
    kan: KANConfig = field(default_factory=KANConfig)
    boost: BoostConfig = field(default_factory=BoostConfig)
    categorical_cols: object = None
    device: str | None = None
    batch_size: int | None = None
    gam: bool = False
    monotone_constraints: dict | None = None
    random_state: int = 42
    verbose: bool = False
    objective: str = "squared_error"  # regressor-only, ignored by the classifier
    alpha: float = 0.5                # regressor-only, ignored by the classifier

    # flat-kwarg-name -> nested-field-name, for from_flat()/to_flat()
    _FLAT_TO_KAN = {
        "kan_hidden": "hidden", "kan_grid": "grid", "kan_k": "k", "kan_steps": "steps",
        "kan_lr": "lr", "lamb": "lamb", "lamb_l1": "lamb_l1", "lamb_coefdiff": "lamb_coefdiff",
    }
    _FLAT_TO_BOOST = {
        "n_estimators": "n_estimators", "learning_rate": "learning_rate",
        "early_stopping_rounds": "early_stopping_rounds", "validation_fraction": "validation_fraction",
    }
    _TOP_LEVEL = [
        "categorical_cols", "device", "batch_size", "gam", "monotone_constraints",
        "random_state", "verbose", "objective", "alpha",
    ]

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        """Same checks as `_BaseKANBoost._validate_hyperparams()` -- kept
        here so a config can be validated at construction time, before
        any model is even instantiated."""
        if self.boost.n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")
        if not (0 < self.boost.learning_rate <= 1):
            raise ValueError("learning_rate must be in (0, 1]")
        if self.kan.hidden < 1 or self.kan.grid < 1 or self.kan.steps < 1:
            raise ValueError("kan_hidden, kan_grid, kan_steps must be >= 1")
        if self.boost.validation_fraction is not None and not (0 < self.boost.validation_fraction < 1):
            raise ValueError("validation_fraction must be in (0, 1)")
        if self.batch_size is not None and self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.monotone_constraints:
            if not self.gam:
                raise ValueError(
                    "monotone_constraints requires gam=True -- monotonicity through "
                    "a hidden layer's own spline can't be guaranteed edge-wise; GAM "
                    "mode (kan_hidden=1, identity output layer) makes the constraint sound."
                )
            if self.kan.hidden != 1:
                raise ValueError("monotone_constraints requires kan_hidden=1 (GAM mode).")
            if any(v not in (1, -1) for v in self.monotone_constraints.values()):
                raise ValueError("monotone_constraints values must be 1 (increasing) or -1 (decreasing)")

    @classmethod
    def from_flat(cls, **kwargs) -> "KANBoostConfig":
        """Build a `KANBoostConfig` from the current flat constructor
        kwargs (`n_estimators=`, `kan_hidden=`, ... exactly as
        `KANBoostClassifier`/`Regressor` accept them today)."""
        kan_kwargs = {v: kwargs[k] for k, v in cls._FLAT_TO_KAN.items() if k in kwargs}
        boost_kwargs = {v: kwargs[k] for k, v in cls._FLAT_TO_BOOST.items() if k in kwargs}
        top_kwargs = {k: kwargs[k] for k in cls._TOP_LEVEL if k in kwargs}
        return cls(kan=KANConfig(**kan_kwargs), boost=BoostConfig(**boost_kwargs), **top_kwargs)

    def to_flat(self) -> dict:
        """Inverse of `from_flat()` -- the flat kwarg dict sklearn's
        `get_params()`/`clone()` and kantun need."""
        flat = {flat_name: getattr(self.kan, nested_name) for flat_name, nested_name in self._FLAT_TO_KAN.items()}
        flat.update({flat_name: getattr(self.boost, nested_name) for flat_name, nested_name in self._FLAT_TO_BOOST.items()})
        flat.update({name: getattr(self, name) for name in self._TOP_LEVEL})
        return flat

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "KANBoostConfig":
        d = dict(d)
        kan = d.pop("kan", None)
        boost = d.pop("boost", None)
        kan = KANConfig(**kan) if isinstance(kan, dict) else (kan or KANConfig())
        boost = BoostConfig(**boost) if isinstance(boost, dict) else (boost or BoostConfig())
        return cls(kan=kan, boost=boost, **d)

    @classmethod
    def from_yaml(cls, path: str) -> "KANBoostConfig":
        import yaml
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_yaml(self, path: str) -> None:
        import yaml
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f)
