<!--
DRAFT ONLY — not published anywhere. Review and edit before posting
(e.g. to Show HN, a personal blog, or linking from the docs site).
-->

# We reproduced a published KAN benchmark exactly — then found (and fixed) a real bug it exposed

[KANBoost](https://github.com/tuamah/kanboost) is a gradient boosting
library that uses shallow Kolmogorov-Arnold Networks instead of
decision trees as weak learners — same Friedman-style boosting loop as
XGBoost/LightGBM/CatBoost, but each learner is an interpretable spline
network instead of a tree.

To get an honest, third-party point of comparison, we reproduced
[arXiv:2509.16750](https://arxiv.org/abs/2509.16750)'s exact benchmark
methodology — same 6 datasets, same preprocessing, same train/test
split seed — and compared KANBoost's numbers directly against their
published KAAM/Logistic-KAN/LR/RF results. ([Full Colab
notebook](examples/benchmark_vs_kaam_paper.ipynb).)

On most of their datasets, KANBoost matched or beat their reported
numbers. On one — their CDC BRFSS "heart" dataset, ~90% majority class
— something else happened: `accuracy=0.895`, `ROC AUC=0.869`, but
`F1 = precision = recall = 0.0`.

## That's not a modeling failure — it's a threshold bug

A 0.869 AUC means the model's raw scores rank positive vs. negative
cases correctly most of the time. F1=0 at the same time means
`predict()`'s default 0.5 cutoff called *every single row* negative.
Both can be true at once: a well-calibrated model trained on a 90/10
split legitimately outputs `p < 0.5` almost everywhere, because the
ensemble's starting point is the true base-rate log-odds (~-2.2 for a
10% positive rate), and each round's weak learner is fit by unweighted
MSE — dominated by the 90% majority's small residuals.

We shipped the fix as `kanboost.imbalance`:

```python
from kanboost.imbalance import find_threshold

model.fit(X_train, y_train, eval_set=(X_val, y_val))
t = find_threshold(model, X_val, y_val, metric="f1")
model.evaluate(X_test, y_test, threshold=t)
```

On a synthetic reproduction of the exact failure mode: F1 went from
**0.000 → 0.545**, with ROC AUC unchanged (0.857 both ways, since
threshold tuning only moves the cutoff, not the scores). We also added
`balanced_weights()` for inverse-frequency `sample_weight` — useful
too, but noticeably weaker alone (F1 0.000 → 0.040) than fixing the
threshold. Full writeup: [Imbalanced classification
guide](https://tuamah.github.io/kanboost/guide/imbalance/).

## Also new this round

- **`kanboost.accel.fast_fit()`** — warm-starts each boosting round's
  learner from the previous round's weights instead of training from
  scratch every time. 3.37x faster on Breast Cancer Wisconsin with
  accuracy essentially unchanged (0.9921 → 0.9893 AUC), monotone
  constraints still enforced exactly.
- **`kantun`** (the hyperparameter-tuning sibling package) now
  supports continuous/log-uniform parameter sampling, custom scoring
  functions, and a wall-clock `time_budget_s` for expensive searches.

Repo: https://github.com/tuamah/kanboost · Docs:
https://tuamah.github.io/kanboost/ · `pip install kanboost`
