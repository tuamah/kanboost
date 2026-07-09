"""
Utilities for imbalanced classification targets.

KANBoostClassifier's `LogisticLoss` and `predict(threshold=0.5)` are both
standard and correct: a well-calibrated model trained on a 90/10 dataset
legitimately outputs `p < 0.5` almost everywhere, so the default threshold
alone can yield a degenerate always-majority-class prediction even though
the underlying scores carry real signal (e.g. a high ROC-AUC). This module
gives two independent, additive knobs to address that -- both compose with
the existing `sample_weight` and `predict(threshold=...)` parameters, with
no changes to the core boosting loop or loss.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_curve


def balanced_weights(y) -> np.ndarray:
    """Inverse-frequency sample weights, analogous to sklearn's
    `class_weight="balanced"`: `weight[i] = n_samples / (n_classes * count[y[i]])`.

    Feeds straight into `KANBoostClassifier.fit(..., sample_weight=...)`;
    rebalances each weak learner's per-round MSE fit and the initial
    log-odds so the minority class isn't drowned out by the majority's
    small residuals.
    """
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    class_weight = {c: n_samples / (n_classes * count) for c, count in zip(classes, counts)}
    return np.array([class_weight[v] for v in y], dtype=float)


def find_threshold(model, X_val, y_val, metric: str = "f1") -> float:
    """Return the classification threshold (for binary `predict`/`evaluate`)
    that maximizes `metric` on `(X_val, y_val)`.

    metric : {"f1", "youden"}
        "f1" scans candidate thresholds from the validation ROC curve and
        picks the one maximizing F1 for the positive class.
        "youden" maximizes `tpr - fpr` (Youden's J statistic), which does
        not require choosing a target metric but ignores class costs.

    Only meaningful for binary classifiers (`len(model.classes_) == 2`).
    """
    if len(model.classes_) != 2:
        raise ValueError("find_threshold only supports binary classifiers.")
    y_val = np.asarray(y_val)
    y_bin = (y_val == model.classes_[1]).astype(int)
    y_prob = model.predict_proba(X_val)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_bin, y_prob)
    # roc_curve's first threshold is +inf (the "predict nothing" point);
    # drop it so we only consider thresholds actually reachable by data.
    fpr, tpr, thresholds = fpr[1:], tpr[1:], thresholds[1:]

    if metric == "youden":
        best_idx = int(np.argmax(tpr - fpr))
    elif metric == "f1":
        f1s = [f1_score(y_bin, (y_prob >= t).astype(int), zero_division=0) for t in thresholds]
        best_idx = int(np.argmax(f1s))
    else:
        raise ValueError(f"Unknown metric {metric!r}; use 'f1' or 'youden'.")

    return float(thresholds[best_idx])
