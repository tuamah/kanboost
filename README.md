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

> **Status: early-stage research project.** This is *not* a drop-in
> replacement for CatBoost/XGBoost in production. See
> [Benchmarks](#benchmarks) and [Honest limitations](#honest-limitations)
> below before using this for anything important.

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

## Features

- **Binary and multiclass classification** (`KANBoostClassifier`,
  one-vs-rest for 3+ classes) and **regression** (`KANBoostRegressor`)
- **GPU support** — `device="cuda"` (or `device=None` to auto-detect),
  falls back to CPU
- **Model persistence** — `model.save(path)` / `KANBoostClassifier.load(path)`
- **`sample_weight`** support in `fit()`
- **Interpretability**: `model.feature_importances()` /
  `feature_importances_dict()`, and `model.plot_feature(name)` for a
  partial-dependence-style curve of a single feature's learned response
- Automatic categorical encoding and missing-value handling, no manual
  preprocessing required

## Install

```bash
pip install kanboost
```

Or from source:

```bash
git clone https://github.com/tuamah/kanboost.git
cd kanboost
pip install -e .
```

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

## Benchmarks

Preliminary results on a real-world telecom churn dataset (100K rows,
10 numeric features used, 8K-row sample for the KANBoost run due to
current training-speed limits):

| Model | Test AUC | Notes |
|---|---|---|
| CatBoost (tuned, full data, ~100 columns) | **0.6992** | production baseline |
| KANBoostClassifier (this repo, 10 features, 8K sample) | 0.64 | early prototype, untuned |
| Plain KAN (no boosting) | 0.65 | single model, same features |
| Plain MLP | 0.59–0.62 | same features |

**Read this table honestly**: KANBoost does not yet beat CatBoost on
this dataset. The goal of this repo, at this stage, is to establish a
working, extensible implementation and an honest baseline — not to claim
state-of-the-art results.

## Honest limitations

- **Speed**: each weak learner is a full KAN forward/backward pass in
  pure PyTorch. This is currently far slower per-iteration than a
  histogram-based tree split in XGBoost/CatBoost/LightGBM.
- **Tuning**: hyperparameters (`kan_grid`, `kan_hidden`, `kan_steps`,
  `learning_rate`) interact in ways that are not yet well understood;
  expect to need real tuning for your dataset.
- **Categorical encoding** is a simple smoothed target-mean encoder, not
  CatBoost's ordered boosting scheme — it can leak on small folds if not
  used carefully.
- **No monotonic constraints or custom loss functions** yet, and
  multiclass classification is one-vs-rest (independent binary chains
  combined via softmax), not a single joint softmax objective.

## Roadmap

See [`ROADMAP.md`](./ROADMAP.md) for the full project plan, including
planned speed optimizations (FastKAN-style RBF basis, `torch.compile`),
symbolic-formula extraction for the full ensemble, and benchmark
expansion to standard UCI datasets.

## Contributing

Issues and PRs welcome, especially:
- speed optimizations for the per-iteration KAN fit
- better categorical encoding
- benchmark results on additional public datasets

## License

MIT — see [`LICENSE`](./LICENSE).

## Citation / related work

If you use this, please also cite the KAN paper and, where relevant,
the GB-KAN paper this project is conceptually closest to:

```
Liu, Z., Wang, Y., Vaidya, S., et al. (2024). KAN: Kolmogorov-Arnold
Networks. arXiv:2404.19756.

[GB-KAN authors] (2026). Gradient Boosting with Interpretable
Kolmogorov-Arnold Networks. ICAART 2026.
```
