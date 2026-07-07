# KANBoost

**Gradient boosting with Kolmogorov-Arnold Network (KAN) learners** — an
interpretable, from-scratch alternative to tree-based boosting frameworks
(XGBoost, LightGBM, CatBoost).

Instead of decision trees as weak learners, KANBoost fits a sequence of
small, shallow [KAN](https://arxiv.org/abs/2404.19756) networks to the
pseudo-residuals of the previous stage, following the classic Friedman
(2001) gradient boosting recipe. Because each KAN edge is a learnable
univariate spline rather than an opaque weight, the resulting ensemble
exposes per-feature shape functions that are directly inspectable —
closer to a Generalized Additive Model than a black box.

!!! warning "Status"
    Beta. This is *not* a drop-in replacement for CatBoost/XGBoost in
    production. See [Benchmarks](guide/classification.md#benchmarks) and
    the [Roadmap](roadmap.md)'s "Honest limitations" before using this
    for anything important.

## Why this exists

As of mid-2026, there is no widely-used, pip-installable library that
combines KAN with gradient boosting. A closely related idea was
published as **GB-KAN** (ICAART 2026), but no public code accompanies
that paper. KANBoost is an independent, from-scratch open-source
implementation of the same general idea, plus:

- automatic handling of categorical features (smoothed target-mean
  encoding, done fold-safe), instead of requiring manual one-hot encoding
- automatic handling of missing values (median imputation + optional
  missing-indicator columns), instead of requiring you to impute first
- built-in early stopping on a validation set
- approximate feature importances derived from learned spline coefficients

## What you get

- **Binary and multiclass classification** and **regression**
  (squared-error or quantile/pinball loss)
- **GPU support**, model persistence, `sample_weight`, mini-batch
  training for larger datasets
- **KAN-native interpretability**: exact per-feature shape functions,
  hard monotonic constraints, symbolic-formula extraction, analytic
  derivatives, post-hoc refine/prune -- see
  [Interpretability](guide/interpretability.md)
- **Editable models**: collapse a fitted model into directly editable
  shape functions with provable monotonicity after an edit -- see
  [Editable models & dashboard](guide/editing-dashboard.md)
- **A live local dashboard** and an **optional FastAPI serving layer**
  -- see [Editable models & dashboard](guide/editing-dashboard.md) and
  [Serving & observability](guide/serving.md)
- **Hyperparameter tuning** via the sibling [`kantun`](https://github.com/tuamah/kantun)
  package -- see [Tuning with kantun](guide/tuning-with-kantun.md)

## Quickstart

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from kanboost import KANBoostClassifier

df = pd.read_csv("your_data.csv")
X = df.drop(columns=["target"])
y = df["target"].values  # binary or multiclass; NaN in X is handled automatically

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

model = KANBoostClassifier(
    n_estimators=100,
    learning_rate=0.2,
    kan_hidden=4,
    kan_grid=3,
    categorical_cols=["region", "plan_type"],  # optional
    early_stopping_rounds=10,
    device="cuda",  # or None to auto-detect, "cpu" to force CPU
)
model.fit(X_train, y_train, eval_set=(X_val, y_val))  # sample_weight=... optional

probs = model.predict_proba(X_val)  # shape (n, 2) binary, (n, n_classes) multiclass
importances = model.feature_importances_dict()

model.save("model.pt")
loaded = KANBoostClassifier.load("model.pt")  # device=... to override where it loads

model.plot_feature("region")  # matplotlib partial-dependence plot
```

See [Installation](installation.md) to get started, or the
[Guide](guide/classification.md) for the full API.
