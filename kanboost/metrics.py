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


def classification_report_dict(y_true, y_pred, y_prob=None, labels=None) -> dict:
    """
    Returns a dict with confusion matrix + standard classification
    metrics.

    - Binary (2 classes): `y_prob` is the probability of the positive
      class (`labels[1]` / the larger label); `precision`/`recall`/`f1`
      are for that positive class; `auc` is the standard binary ROC-AUC.
    - Multiclass (3+ classes): `y_prob` is an (n_samples, n_classes)
      array matching the sorted class order; `precision`/`recall`/`f1`
      are macro-averaged; `auc` is one-vs-rest macro ROC-AUC.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = np.unique(y_true)
    is_binary = len(labels) <= 2
    average = "binary" if is_binary else "macro"
    pos_label = labels[-1] if is_binary else 1

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = {
        "labels": list(labels),
        "confusion_matrix": cm.tolist(),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(
            y_true, y_pred, average=average, pos_label=pos_label, zero_division=0
        )),
        "recall": float(recall_score(
            y_true, y_pred, average=average, pos_label=pos_label, zero_division=0
        )),
        "f1": float(f1_score(
            y_true, y_pred, average=average, pos_label=pos_label, zero_division=0
        )),
    }
    if y_prob is not None:
        try:
            if is_binary:
                report["auc"] = float(roc_auc_score(y_true, y_prob))
            else:
                report["auc"] = float(
                    roc_auc_score(y_true, y_prob, multi_class="ovr", labels=labels)
                )
        except ValueError:
            report["auc"] = None
    return report


def print_classification_report(report: dict, class_names=None) -> None:
    """Pretty-print a report dict returned by classification_report_dict."""
    cm = np.array(report["confusion_matrix"])
    labels = report.get("labels", list(range(cm.shape[0])))
    if class_names is None:
        class_names = [str(l) for l in labels]

    print("Confusion Matrix (rows=actual, cols=predicted):")
    header = "".join(f"{name:>12}" for name in class_names)
    print(f"{'':>10}{header}")
    for i, name in enumerate(class_names):
        row = "".join(f"{cm[i, j]:>12d}" for j in range(cm.shape[1]))
        print(f"{name:>10}{row}")
    print()
    print(f"Accuracy : {report['accuracy']:.4f}")
    print(f"Precision: {report['precision']:.4f}")
    print(f"Recall   : {report['recall']:.4f}")
    print(f"F1 score : {report['f1']:.4f}")
    if report.get("auc") is not None:
        print(f"ROC-AUC  : {report['auc']:.4f}")
