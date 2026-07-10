"""Tests for kanboost.logging_utils."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kanboost.ops.logging_utils import get_logger, log_boosting_rounds
from kanboost.ops.observability import RoundMetric


def test_get_logger_is_idempotent():
    l1 = get_logger("kanboost_test_idempotent")
    l2 = get_logger("kanboost_test_idempotent")
    assert l1 is l2
    assert len(l1.handlers) == 1  # a second call must not add a duplicate handler


def test_log_boosting_rounds_runs_without_error(capsys):
    logger = get_logger("kanboost_test_rounds")
    rounds = [
        RoundMetric(round=0, elapsed_seconds=0.1, gpu_allocated_mb=None, loss=0.5),
        RoundMetric(round=1, elapsed_seconds=0.2, gpu_allocated_mb=12.5, loss=0.4),
    ]
    log_boosting_rounds(rounds, logger=logger, model_name="test_model")
    captured = capsys.readouterr()
    assert "round=0" in captured.err
    assert "val_loss=0.50000" in captured.err
    assert "gpu_mb=12.5" in captured.err


if __name__ == "__main__":
    test_get_logger_is_idempotent()
    print("All logging_utils tests passed.")
