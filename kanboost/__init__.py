"""
KANBoost: Gradient Boosting with Kolmogorov-Arnold Network learners.

An interpretable alternative to tree-based gradient boosting (XGBoost,
LightGBM, CatBoost), using shallow KAN networks as weak learners instead
of decision trees.
"""

from .core.classifier import KANBoostClassifier
from .core.regressor import KANBoostRegressor
from .core.config import KANBoostConfig, KANConfig, BoostConfig
from .train.metrics import classification_report_dict, print_classification_report
from .pipeline import KANBoostPipeline, PipelineResult

__version__ = "1.2.1"
__all__ = [
    "KANBoostClassifier",
    "KANBoostRegressor",
    "KANBoostConfig",
    "KANConfig",
    "BoostConfig",
    "KANBoostPipeline",
    "PipelineResult",
    "classification_report_dict",
    "print_classification_report",
]

# Hyperparameter tuning lives in the separate `kantun` package, so that
# kanboost's core dependency footprint stays minimal and kantun can be
# used to tune other model types too. Install with: pip install kantun
