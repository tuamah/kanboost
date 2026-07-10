"""
KANBoost: Gradient Boosting with Kolmogorov-Arnold Network learners.

An interpretable alternative to tree-based gradient boosting (XGBoost,
LightGBM, CatBoost), using shallow KAN networks as weak learners instead
of decision trees.
"""

from .classifier import KANBoostClassifier
from .regressor import KANBoostRegressor
from .metrics import classification_report_dict, print_classification_report

__version__ = "0.0.21"
__all__ = [
    "KANBoostClassifier",
    "KANBoostRegressor",
    "classification_report_dict",
    "print_classification_report",
]

# Hyperparameter tuning lives in the separate `kantun` package, so that
# kanboost's core dependency footprint stays minimal and kantun can be
# used to tune other model types too. Install with: pip install kantun
