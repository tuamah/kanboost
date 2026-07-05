# KANBoost Roadmap

**Goal:** an interpretable gradient boosting library that uses KAN
(Kolmogorov-Arnold Network) learners instead of decision trees, aiming
for accuracy competitive with CatBoost/XGBoost/LightGBM while exposing
inspectable per-feature spline shape functions.

## Shipped

**v0.0.1 — initial release**
- Binary classification (`KANBoostClassifier`) and regression (`KANBoostRegressor`)
- Automatic categorical encoding (smoothed target-mean, fold-safe)
- Early stopping on an explicit `eval_set`
- Approximate feature importances from learned spline coefficients

**v0.0.2 — GPU support**
- `device` parameter (`"cpu"`, `"cuda"`, `"cuda:0"`, or `None` to auto-detect)

**v0.0.3 — multiclass, persistence, missing values, interpretability, weighting**
- Multiclass classification (one-vs-rest binary chains + softmax)
- `save()` / `load()` model persistence (learners stored as `state_dict`s,
  since pykan's `KAN` isn't directly picklable)
- Automatic missing-value handling (median imputation + optional
  `<col>_missing` indicator columns)
- `plot_feature(name)` — partial-dependence-style spline plot
- `sample_weight` support in `fit()`

**v0.0.4 — objectives, training scale, native attribution**
- Shared boosting loop (`_BaseKANBoost._boost_chain`) driven by pluggable
  loss objects (`kanboost/losses.py`), replacing near-duplicate
  classifier/regressor loops
- `KANBoostRegressor(objective="quantile", alpha=...)` — pinball-loss
  quantile regression
- `validation_fraction` — internal train/validation split for early
  stopping when no explicit `eval_set` is given (split happens before
  preprocessing is fit, so no leakage)
- `batch_size` — mini-batch Adam training per weak learner, for datasets
  where full-batch is too slow or doesn't fit in memory
- `feature_contributions(X)` — native per-sample, per-feature attribution
  from each learner's first KAN layer (exact reconstruction of the
  hidden representation when `kan_hidden=1`; see its docstring for the
  precise guarantee)

## Deferred (with reasons)

- **Monotonic constraints** — no sound way to constrain a KAN with a
  hidden layer (the output layer's own spline can undo any per-feature
  monotonicity); a penalty-based approximation would be misleading
  rather than a real guarantee. Revisit only alongside a documented
  "pure additive" (`kan_hidden=1`, identity output layer) mode.
- **Symbolic formula extraction** (`pykan.auto_symbolic`) — fragile and
  slow in practice, and formula count explodes with more than a handful
  of estimators. Worth a standalone spike against a `kan_hidden=1`,
  small-`n_estimators` model, not a general-purpose feature.
- **`torch.compile` / ONNX export / FastKAN backend** — pykan's `KAN`
  modules don't trace or compile cleanly out of the box; would need
  upstream changes or a from-scratch spline layer.
- **Multi-GPU** — each weak learner is tiny; the bottleneck is the
  number of sequential boosting rounds, not per-learner compute, so
  multi-GPU wouldn't help without changing the training loop's
  architecture.
- **Benchmark suite vs. XGBoost/LightGBM/CatBoost on standard UCI
  datasets, and a docs site** — legitimate next steps, but they're
  ongoing measurement/writing efforts rather than library features;
  tracked separately from this code roadmap.
- **`kantun` integration test** — depends on the sibling `kantun`
  package; add once both repos' APIs have settled rather than pinning
  kanboost's test suite to kantun's release cadence.
- **CLI** — the sklearn-style Python API already covers the realistic
  usage patterns; a CLI wouldn't add much for a model-fitting library.

## Honest limitations (see also `README.md`)

- **Speed**: each weak learner is a full KAN forward/backward pass in
  pure PyTorch — slower per-iteration than a histogram-based tree split.
  `batch_size` helps on large datasets but doesn't close this gap.
- **Multiclass is one-vs-rest**, not a single joint softmax objective —
  `n_classes` independent binary chains, `n_classes` times the training cost.
- **Categorical encoding** is a simple smoothed target-mean encoder, not
  CatBoost's ordered boosting scheme.
