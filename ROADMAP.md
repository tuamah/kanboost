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

**v0.0.5 — KAN-native structural guarantees (monotonicity, GAM/symbolic, derivatives, model surgery)**
- `gam=True` — fixes each learner's output edge to the identity function
  (`fix_symbolic`), so the ensemble is an exact additive model
  `F(x) = c + sum_j g_j(x_j)`. The earlier "deferred" reasoning
  (monotonicity/symbolic extraction can't work through a hidden layer's
  own nonlinear output spline) turned out to be a scoping problem, not a
  fundamental one: constrain the mode, not just the edges.
- `monotone_constraints={"feature": 1|-1}` (requires `gam=True`,
  `kan_hidden=1`) — hard monotonicity via projecting the first layer's
  B-spline control points onto the monotone cone after every optimizer
  step (the variation-diminishing property of B-splines), with the SiLU
  base branch and the spline's sign both frozen so nothing can undo it.
  Verified on California Housing's `MedInc` in
  `examples/benchmark_uci.py` — derivative stays non-negative on held-out
  test data, not just training data.
- `symbolic_report(X)` — in GAM mode, fits a small closed-form function
  library (`sin`, `cos`, `exp`, `x^2`, `tanh`, ...) to each feature's
  exact aggregated shape function, one fit per feature (not per learner,
  avoiding the earlier fragility/explosion concern).
- `predict_derivative(X, feature)` — analytic derivative curves via
  autograd through the whole ensemble; exact and globally defined, unlike
  a tree's zero/undefined derivative or an MLP's pointwise gradient.
- `refine(X, new_grid)` / `prune(X, threshold)` — post-hoc, in-place
  resolution upgrade or dead-edge removal on a fitted ensemble, wrapping
  pykan's `KAN.refine`/`prune_edge`. No retraining from scratch, and no
  equivalent operation exists for a fitted decision tree.
- `feature_interaction(X)` — native structural interaction scores
  (`kan_hidden > 1` only) via pykan's own attribution machinery.
- `lamb`/`lamb_l1`/`lamb_coefdiff` — pykan's spline smoothness/sparsity
  regularizers, plumbed through (full-batch path only).
- `examples/benchmark_uci.py` — Adult Income and California Housing vs.
  `HistGradientBoosting*` as an honest sanity floor (results in README).

## Deferred (with reasons)

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
