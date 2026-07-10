# Imbalanced classification

Reproducing [arXiv:2509.16750](https://arxiv.org/abs/2509.16750)'s exact
benchmark methodology (see [Benchmarks](classification.md#benchmarks))
surfaced a real failure mode: `KANBoostClassifier` trained on their
CDC BRFSS "heart" dataset (~90% majority class) with `gam=True`,
`kan_hidden=1` produced `accuracy=0.895` but `F1=precision=recall=0.0` —
an always-predict-majority-class model, despite `AUC=0.869` showing the
model's raw scores carried real discriminative signal.

## Why this happens

`LogisticLoss` and `predict(threshold=0.5)` are both standard and
correct in isolation. The problem is their combination: a
*well-calibrated* model on a 90/10 split legitimately outputs
`p < 0.5` almost everywhere, since `init_pred_` starts at the true
base-rate log-odds (~`-2.2` for 10% positives) and each weak learner's
unweighted MSE fit is dominated by the majority class's small
residuals. `predict()`'s default `threshold=0.5` then reads every one
of those sub-0.5 scores as "negative" — F1 = 0 despite a genuinely
useful ranking underneath.

`kanboost.train.imbalance` gives two independent, composable fixes — neither
touches `LogisticLoss` itself, since it isn't wrong.

## `find_threshold` — fix the decision boundary, not the model

The verified-effective fix. Scans the validation ROC curve for the
threshold that maximizes F1 (or Youden's J), then use it with
`predict(threshold=...)`/`evaluate(threshold=...)`:

```python
from kanboost.train.imbalance import find_threshold

model = KANBoostClassifier(gam=True, kan_hidden=1, early_stopping_rounds=10)
model.fit(X_train, y_train, eval_set=(X_val, y_val))

t = find_threshold(model, X_val, y_val, metric="f1")
model.evaluate(X_test, y_test, threshold=t)
```

Because this only changes `predict()`'s cutoff (not `predict_proba()`),
ROC AUC is unaffected. On a synthetic 90/10 reproduction of the bug:

| | Threshold | F1 | Precision | Recall | AUC |
|---|---|---|---|---|---|
| Default (`0.5`) | 0.500 | 0.000 | 0.000 | 0.000 | 0.857 |
| `find_threshold` | 0.241 | 0.545 | 0.540 | 0.551 | 0.857 (unchanged) |

## `balanced_weights` — rebalance training itself

Inverse-frequency sample weights, analogous to scikit-learn's
`class_weight="balanced"`. Feeds into the existing `sample_weight`
parameter — no core changes:

```python
from kanboost.train.imbalance import balanced_weights

model.fit(X_train, y_train, eval_set=(X_val, y_val),
          sample_weight=balanced_weights(y_train))
```

On its own this shifted the minority class's residuals to matter more
per round, but was a much weaker fix than threshold tuning in testing
(F1 0.0 → 0.04 alone vs. 0.0 → 0.545 for `find_threshold`) — use it
*together* with `find_threshold` rather than as a substitute for it.

## Recommendation

Always check `find_threshold` first — it's free (no retraining) and
was the dominant fix in every test. Add `balanced_weights` on top if
the minority class still underperforms after threshold tuning.
