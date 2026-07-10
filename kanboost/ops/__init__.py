"""kanboost.ops -- serving, the interactive dashboard, observability,
logging, and MLflow experiment tracking. Each submodule keeps its heavy
dependency (fastapi/streamlit/mlflow) lazily imported, so importing
`kanboost.ops` itself never requires any of them."""

from .dashboard import launch
from .serving import create_app
from .observability import gpu_utilization_flag, time_predict, capture_boosting_rounds, RoundMetric
from .logging_utils import get_logger, log_boosting_rounds
from .mlflow_utils import log_training_run
