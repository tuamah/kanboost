# Calibration

Three independent benchmarks (see [Benchmarks](classification.md#benchmarks))
found the same pattern: KANBoost's `predict_proba` *ranking* (ROC
AUC/PR AUC) is competitive with or ahead of tuned tree ensembles, but
its raw probability *values* are comparatively miscalibrated — worst
Brier score and log-loss in all three runs, with the F1-optimal
decision threshold sitting around 0.40–0.42 rather than 0.5.

`kanboost.calibration` fixes this post-hoc, without retraining:

```python
from kanboost.calibration import calibrate, CalibratedKANBoost

model = KANBoostClassifier(n_estimators=100)
model.fit(X_train, y_train)

# X_cal/y_cal must be held out -- not used in model.fit()
cal_model = calibrate(model, X_cal, y_cal, method="platt")  # or method="isotonic"

cal_model.predict_proba(X_test)  # calibrated probabilities
cal_model.predict(X_test)         # same threshold semantics as the base model
cal_model.save("calibrated_model.pt")
loaded = CalibratedKANBoost.load("calibrated_model.pt")
```

## Which method

`method="platt"` (default) fits a 2-parameter logistic rescaling of the
raw score. This is the right fix for a systematic shift like KANBoost's
measured pattern, needs relatively little calibration data, and — being
strictly monotone — leaves ROC AUC/PR AUC exactly unchanged (verified
in tests: identical to within `1e-9`).

`method="isotonic"` fits a free-form monotone map. More flexible for
shape-based miscalibration, but needs a larger `X_cal` (order of
1000+ rows) to avoid overfitting.

Measured on a held-out Breast Cancer Wisconsin split, Platt scaling:

| | Brier | Log loss | ROC AUC |
|---|---|---|---|
| Raw `predict_proba` | 0.090 | 0.344 | 0.9931 |
| Calibrated (Platt) | 0.030 | 0.119 | 0.9931 (unchanged) |

## Multiclass

Each one-vs-rest chain is calibrated independently (mirroring how the
base model's own `predict_proba` computes each chain), then rows are
renormalized to sum to 1. Renormalization can occasionally shift which
class wins the argmax near a decision boundary — tests confirm accuracy
doesn't meaningfully degrade, but it isn't guaranteed to only improve.

## Interacting with editable models

If you also use [`kanboost.editing`](editing-dashboard.md), calibrate
*after* finalizing any `EditableGAM` edits, not before — an edit changes
the model's raw scores and would silently stale an already-fitted
calibration map. There's no code-level coupling between the two
modules; this is an ordering discipline you need to maintain yourself.

## Prediction speed (a separate, unrelated gap)

Calibration doesn't address KANBoost's prediction-*speed* gap against
tree ensembles (a separate, also benchmark-confirmed issue). For
`gam=True` models specifically, [`kanboost.editing.consolidate()`](editing-dashboard.md)
is the fix for that — see its docs for the ~50x measured speedup. For
*training*-time speed, see [`kanboost.accel.fast_fit()`](training-speed.md).
