"""kanboost.train -- faster/opt-in training, post-hoc calibration,
class-imbalance handling, and evaluation metrics."""

from .accel import fast_fit
from .calibration import calibrate, CalibratedKANBoost
from .imbalance import balanced_weights, find_threshold
from .metrics import classification_report_dict, print_classification_report
