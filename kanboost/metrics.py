"""
Evaluation utilities for KANBoost models.

Kept as plain functions (not baked only into the classifier) so they can
also be reused by the tuning module and by users who just want metrics
on any set of predictions.
"""

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
)


def classification_report_dict(y_true, y_pred, y_prob=None) -> dict:
    """
    Returns a dict with confusion matrix + standard classification
    metrics. If y_prob (probability of the positive class) is given,
    also includes ROC-AUC.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    report = {
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": ["true_negative_row0", "true_positive_row1"],
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None:
        try:
            report["auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            report["auc"] = None
    return report


def print_classification_report(report: dict, class_names=("0", "1")) -> None:
    """Pretty-print a report dict returned by classification_report_dict."""
    cm = np.array(report["confusion_matrix"])
    print("Confusion Matrix:")
    print(f"                 predicted {class_names[0]}   predicted {class_names[1]}")
    print(f"    actual {class_names[0]:>3}      {cm[0, 0]:>10d}      {cm[0, 1]:>10d}")
    print(f"    actual {class_names[1]:>3}      {cm[1, 0]:>10d}      {cm[1, 1]:>10d}")
    print()
    print(f"Accuracy : {report['accuracy']:.4f}")
    print(f"Precision: {report['precision']:.4f}")
    print(f"Recall   : {report['recall']:.4f}")
    print(f"F1 score : {report['f1']:.4f}")
    if report.get("auc") is not None:
        print(f"ROC-AUC  : {report['auc']:.4f}")
