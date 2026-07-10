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

**v0.0.6 — device fail-fast fix, observability, logging, optional serving API**
- `_resolve_device` now fails fast with a clear `RuntimeError` when
  `device="cuda"`/`"cuda:0"` is requested but CUDA isn't available,
  instead of silently degrading or failing later with a cryptic CUDA
  error deep inside training. Still resolves arbitrary device strings
  (`"cuda:0"`, `"mps"`) via `torch.device(...)`, unlike a naive
  `if device == "cuda"` check that would only handle the exact string.
- `kanboost/observability.py` (new, additive -- no changes to
  `_base.py`/`classifier.py`/`regressor.py`): `time_predict`,
  `memory_snapshot`, `gpu_utilization_flag`, and
  `capture_boosting_rounds` (per-round timing/loss/GPU-memory, via a
  temporary instance-level wrap of `_fit_learner` plus parsing the
  existing `verbose=True` log lines -- restored automatically even if
  `fit()` raises).
- `kanboost/logging_utils.py` (new) -- a thin, opt-in wrapper around the
  stdlib `logging` module, pairing with `capture_boosting_rounds` for
  structured per-round log lines instead of raw `print()`.
- `kanboost/serving.py` (new, optional: `pip install kanboost[api]`) --
  a FastAPI wrapper (`create_app(model_path)`) with `/health`,
  `/predict`, `/predict_proba` endpoints, auto-detecting classifier vs.
  regressor from a saved model's own metadata.

**v0.0.7 — docs-only release**
- Added an independent real-world benchmark (NFL Draft prediction
  dataset, 5-fold CV vs. tuned CatBoost) to the README. No code changes.

**v0.0.8 — editable models**
- `kanboost/editing.py` (new): `consolidate(model)` collapses a fitted
  `gam=True` ensemble's per-feature shape function -- a sum of splines
  across every boosting round -- into a single spline per feature,
  wrapped in an `EditableGAM`. Positioned against Microsoft's GAM
  Changer (an editing tool for EBM): EBM's piecewise-constant bins give
  no way to verify an edit preserves monotonicity/smoothness; here,
  `enforce_monotone` re-derives a provably monotone coefficient sequence
  after an edit, reusing the same variation-diminishing projection
  `monotone_constraints` already uses during training.
- Consolidation correctness note: naively sampling each feature's curve
  by zeroing every other (scaled) feature captures `g_j(x_j) + sum_{i!=j}
  g_i(0)`, not `g_j(x_j)` alone -- summing those probes directly would
  double-count every other feature's zero-point contribution `(n-1)`
  times over. Fixed by centering each probe against the ensemble's score
  at the all-zero input (the standard GAM identifiability convention,
  `g_j(0) := 0`, with the removed constant folded into the intercept).
  Caught via a predict-parity check before shipping, not assumed correct
  from the per-feature curve fit alone (which was already accurate and
  would not have surfaced this).

**v0.0.9 — experimental utilities**
- `kanboost/experimental.py` (new, additive): `suggest_constraints`
  (Spearman correlation + quantile-binned bin-mean consistency, advisory
  only -- not a guarantee), `audit_monotonicity` (verifies
  `predict_derivative`'s sign actually matches `monotone_constraints` on
  given data -- catches a constraint that was requested but silently not
  enforced, e.g. by a custom weak-learner backend that bypasses the
  training-time projection), `symbolic_export` (compact text summary
  over `symbolic_report`), `predict_interval` (mean/quantile spread
  across a list of independently fitted models -- a convenience wrapper,
  not a replacement for `objective="quantile"`'s calibrated conditional
  quantiles), `explain_row` (top feature contributions for one row via
  `feature_contributions`), and `dashboard_html` (one static HTML report
  combining several of the above).
- Two bugs found and fixed before shipping (both surfaced by testing
  against real fitted models, not assumed from the code alone):
  `suggest_constraints`'s original point-to-point consistency check gave
  a false negative on a genuinely strongly-monotone feature (Spearman
  0.881) once other noisy signal was mixed in; switched to checking
  consistency of quantile-binned means instead, which is far less
  sensitive to point-level noise. `dashboard_html` crashed on
  `json.dumps` whenever a `np.float32`/`np.int64`/`np.ndarray` value
  reached it (only `np.float64` happens to satisfy Python's `float`
  duck-typing); fixed with a recursive numpy-to-plain-Python converter
  applied before serializing.

**v0.0.10 — interactive dashboard**
- `kanboost/dashboard.py` + `kanboost/_dashboard_app.py` (new, optional:
  `pip install kanboost[dashboard]`): a local Streamlit app
  (`launch(model_path, data_path=None)`) for exploring one fitted model
  -- feature importances, `plot_feature`, `symbolic_report` (GAM mode),
  `feature_interaction`, `explain_row`, and, for a single-chain
  `gam=True` model, a live editing panel wired to
  `kanboost.editing.EditableGAM` (`set_offset`, `enforce_monotone`,
  `diff`, `save`), with the curve redrawn immediately after each edit.
  Chosen over Dash/Gradio/a hand-written Flask+JS app because Streamlit
  renders matplotlib figures natively (`plot_feature` needed no
  changes) and every widget interaction is a plain server-side Python
  callback, which a stateful editing panel needs and Gradio's
  input-to-output function model doesn't fit well.
- Verified end-to-end (not just import-tested) via Streamlit's
  `AppTest` harness: full script execution, button clicks
  (`set_offset`/`enforce_monotone`/`save`) round-tripping through a real
  `EditableGAM`, and a saved edited model reloading correctly -- across
  a GAM regressor with data, a non-GAM model (edit tab correctly absent),
  and a multiclass GAM classifier (edit tab correctly absent too, since
  `consolidate()` returns a dict of independent per-class models in that
  case, out of scope for v1's single-chain editing panel).
- The existing static `kanboost.experimental.dashboard_html` is kept
  as-is (not superseded) -- a zero-dependency shareable snapshot and a
  live local tool serve genuinely different purposes.
- Deferred to a later release: drag-on-canvas curve editing (vs. slider
  + offset), a multiclass editing panel, a what-if prediction
  playground, and an in-app "export static report" button wrapping
  `dashboard_html`.

**v0.0.11 — kantun integration test, Beta, docs site**
- `tests/test_kantun_integration.py` (new, skipped if `kantun` isn't
  installed -- same optional-dependency test pattern already used for
  fastapi/streamlit): a real `KantunSearch(KANBoostClassifier, ...).fit`
  and `KantunSearch(KANBoostRegressor, ...).fit`, plus dedicated cases
  for kantun's new `search_type="halving"` and `n_jobs` parallelism
  (both added in kantun v0.0.3, alongside fold-1 pruning) tuning
  KANBoost specifically. `kantun` is still not a `kanboost` dependency
  -- the whole point of splitting the two packages was keeping
  kanboost's footprint minimal, so this stays a sibling-package test,
  not a hard coupling.
- The `kantun` integration test was deferred pending "both APIs
  settling" -- kantun's public surface hasn't changed since its initial
  release (only additive: `n_jobs`, `prune`, `search_type="halving"`,
  none breaking), and kanboost's has been additive-only since v0.0.4.
  That condition has been met.
- `Development Status :: 4 - Beta` (from `3 - Alpha`): justified by the
  breadth of what's shipped and tested (37+ features across
  interpretability, persistence, serving, editing, and now a verified
  sibling-package integration), not by this one test in isolation.
- `mkdocs.yml` + `docs/` (new, optional: `pip install kanboost[docs]`) --
  a docs site (mkdocs-material) covering installation, classification/
  regression + benchmarks, interpretability, editable models/dashboard,
  serving/observability, and tuning with kantun (including the new
  `search_type="halving"`/`n_jobs`/`prune` options); deployed to GitHub
  Pages on push to `main` via `.github/workflows/docs.yml`, which also
  runs `mkdocs build --strict` on every PR touching docs so a broken
  link/nav entry fails CI instead of silently shipping.
- Two real bugs in kantun's new code, found and fixed before shipping
  (both via testing against real fitted models, not assumed correct
  from the code alone): successive halving's stratified train
  subsampling crashed (`ValueError: test_size ... < n_classes`) when a
  rung's resource count, computed from one fold's size, landed just
  below another fold's actual size (`StratifiedKFold` doesn't guarantee
  perfectly equal folds) -- fixed by treating "not enough left over to
  stratify the discard" the same as "at capacity: use the full fold".
  Separately, halving's `best_score_`/`best_params_` were being updated
  from *every* rung, including subsampled ones -- a low-resource score
  isn't comparable to a full CV mean, any more than a single pruned
  fold's score is (concretely observed: a combo's rung-2 score, on 144
  of 204 training rows, briefly outranked the eventual winner's true
  full-data score). Fixed so only the final, full-training-data rung's
  scores can win `best_score_`/`best_params_`.

**v0.0.12 — docs-only release**
- Added a rigorous cross-validated Breast Cancer Wisconsin benchmark
  (tuned KANBoost vs. tuned tree ensembles, LogReg, MLP; mean ± std over
  folds) to the README and docs site, surfacing a genuine finding not
  previously documented: KANBoost's `predict_proba` *ranking* (ROC
  AUC/PR AUC) is competitive with or ahead of tuned tree ensembles, but
  its raw probability *values* are comparatively miscalibrated out of
  the box (worst Brier score of the group; per-fold F1-optimal
  threshold averaged 0.405, not 0.5). Added a corresponding "Honest
  limitations" entry and a docs tip: tune the decision threshold or
  apply post-hoc calibration before relying on classification metrics
  at the default 0.5 cutoff. Also newly documented: prediction time
  (not just fit time) is markedly slower than tree ensembles on this
  benchmark. No code changes.

**v0.0.13 — docs-only release**
- Added a second, independent Breast Cancer Wisconsin benchmark with a
  stricter methodology (decision threshold selected on a held-out
  *validation* split, then applied once to a separate *test* split,
  rather than the per-fold-optimal-on-test-itself threshold from
  v0.0.12's benchmark). Both confirms the calibration finding (Brier
  score still clearly the worst of the group) and strengthens the
  practical takeaway: with an honestly-selected threshold, KANBoost's
  accuracy/F1/MCC come out in an exact three-way tie for best in the
  comparison, matching LightGBM and XGBoost. No code changes.
- Added a third, independent Breast Cancer Wisconsin CV run (8 models
  including CatBoost, log-loss added alongside Brier). KANBoost's ROC
  AUC/PR AUC come out highest of all 8 models this time; its log loss
  is roughly 2x worse than the next-worst model and 4-5x worse than the
  tree ensembles -- the starkest confirmation yet, across a third
  independent methodology, of the ranking-vs-calibration gap. No code
  changes.

**v0.0.14 — post-hoc calibration, documented predict-speed fix for GAM mode**
- `kanboost/calibration.py` (new): `calibrate(model, X_cal, y_cal, method="platt"|"isotonic")`
  / `CalibratedKANBoost` -- post-hoc Platt scaling (default) or isotonic
  regression on a fitted `KANBoostClassifier`'s raw scores, addressing
  the calibration gap documented in v0.0.12/v0.0.13's three benchmarks.
  Platt is the default because the measured miscalibration is a
  systematic shift (F1-optimal threshold ~0.40-0.42, not 0.5) -- exactly
  what a 2-parameter monotone rescaling fixes, with less data and less
  overfitting risk than isotonic, and (being strictly monotone) zero
  effect on ROC AUC/PR AUC. Multiclass: each one-vs-rest chain
  calibrated independently, then rows renormalized to sum to 1. No
  changes to `_base.py`/`classifier.py` -- reads raw scores through the
  same `_raw_score_chain`/`_transform_X` `predict_proba` itself uses.
  Verified end-to-end on a real held-out Breast Cancer split: Brier
  0.090 -> 0.030, log-loss 0.344 -> 0.119, ROC AUC unchanged to `1e-9`.
- Documented (no new code): `kanboost.editing.consolidate()` is also a
  fast predict path for `gam=True` models -- one spline evaluation per
  feature instead of `n_estimators` full KAN forward passes. Measured
  ~30-50x faster prediction on a 1000-row/6-feature/40-estimator model, at
  ~1e-6 consolidation fidelity cost. This is the answer to the
  prediction-speed gap for GAM-mode models specifically; it does not
  help non-GAM models or training speed.
- Fit-time speed: deliberately left deferred. The boosting loop is
  inherently sequential and each round's cost is `kan_steps` full
  KAN forward/backward passes; cutting `kan_steps`/`kan_grid` defaults
  trades accuracy for speed (a tuning knob users already have, not a
  free optimization), and every from-scratch-backend alternative
  considered this session risks repeating the earlier rejected RBF
  backend's mistake (silently not enforcing `monotone_constraints`).
  No new safe, meaningfully-impactful fit-time speedup was found.

**v0.0.15 — docs-only release: multi-dataset, statistically-tested benchmark**
- Added a fourth benchmark, this time spanning three separate datasets
  (Heart-Statlog 270 rows, Breast Cancer Wisconsin 569 rows, Diabetes/
  Pima 768 rows) with Wilcoxon signed-rank significance testing
  (KANBoost vs. each of 7 other models, 5-fold paired) rather than just
  comparing means. On Heart-Statlog, KANBoost's ROC AUC beat every one
  of the other 7 models on every one of the 5 folds (p=0.0625 against
  all seven -- the smallest p-value obtainable at that sample size).
  Brier score again lands in the worse half of the pack on the two
  datasets where KANBoost isn't best-in-class, consistent with the
  calibration gap. Documented, explicitly labeled not-yet-proven, an
  emerging pattern: KANBoost's relative standing across all four
  Breast-Cancer-family-plus-this run tracks dataset size -- better
  relative to tree ensembles on smaller datasets. No code changes.

**v0.0.16 — executable symbolic formula export**
- `kanboost/symbolic.py` (new): `export_symbolic(model, min_r2=0.8)` /
  `SymbolicModel` -- a real `sympy` expression, LaTeX string, and
  standalone (no torch/pykan needed at call time) numpy predict
  function for a fitted `gam=True` model, not just the existing
  human-readable text summary (`symbolic_report`/
  `kanboost.experimental.symbolic_export`). Fits one closed-form
  candidate (`c * fun(a*x + b) + d`, pykan's `SYMBOLIC_LIB`) per
  feature to the exact aggregated shape function; features below
  `min_r2` fall back to a numeric (spline-interpolated) term instead of
  a misleading forced formula. Multiclass: `{class_label: SymbolicModel}`.
- Reuses `kanboost.editing.consolidate()` for per-feature curves and
  the intercept, rather than re-deriving curve sampling from scratch --
  deliberately avoiding a repeat of `consolidate()`'s own earlier
  double-counting bug (v0.0.8) by building on its already-fixed,
  already-tested centering logic instead of a fresh implementation.
- A real, non-obvious finding surfaced while building this: R^2 alone
  doesn't mean a fitted term is meaningful. A near-noise synthetic
  feature (no real relationship to the target) scored R^2=0.996 with a
  `cos` fit purely by matching its own tiny residual wiggle, while its
  curve's amplitude (~0.13) was over 20x smaller than a genuinely
  important feature's (~3.0) in the same model. Added an `amplitude`
  field to `fidelity_report()` specifically so this can't be missed,
  and documented it as a required check, not an edge case.
- Added `sympy` as an explicit core dependency (previously only
  available transitively via torch's own dependency chain).
- `explain(model, top_features=5, symbolic=True, simplify=True)`: a
  high-level convenience wrapper ranking features by
  `model.feature_importances_dict()` and attaching each top feature's
  symbolic term (`SymbolicModel.term_sympy()`, a new method extracting
  one feature's term instead of the whole model's expression).
  `export_symbolic()` gained a `features=` filter so `explain()` only
  runs the (relatively expensive) candidate search for the features
  actually being reported, not every feature in the model.
- Independent review (this session) on the whole module: APPROVE WITH
  NITS, all addressed before shipping -- guarded against a real
  `sympy.Symbol` name collision (two feature names that sanitize to the
  same symbol, e.g. `"a b"` and `"a-b"`, would otherwise silently
  conflate into one variable in `to_sympy()`), documented a narrow
  `predict_scaled()`-vs-`to_sympy()` divergence for `log`/`sqrt` terms
  on out-of-training-range inputs (unreachable for in-range data, since
  a candidate that goes negative on its fitting domain gets `NaN` R^2
  and is never selected), aligned `explain()`'s `min_r2` default with
  `export_symbolic()`'s (was inconsistently 0.85 vs 0.8), and
  strengthened a test that named a specific behavior (multiclass
  `explain()` using `classes_[0]`'s chain) without actually asserting it.
- Separately, verified and documented a real risk in pykan's own
  `learner.auto_symbolic()` when called directly by a user on
  `model.learners_`: it mutates each weak learner in place, replacing
  its spline with an approximate symbolic function -- silently changing
  `model.predict()`'s output from that point on, with no guarantee the
  approximation is close (an edge with R^2=0.0001 was still forced into
  a formula in one observed case). `export_symbolic()`/`explain()` never
  touch the original model for exactly this reason.
- 11 new tests (8 for `export_symbolic`/`SymbolicModel`, 3 for
  `explain()`). Bumps version to 0.0.16.

**v0.0.17 — MLOps integration (mlhub, mlflow_utils)**
- `kanboost/mlhub.py` (new, `pip install kanboost[mlhub]`):
  `push_model`/`pull_model`/`list_models`/`ensure_bucket` for a
  MinIO-backed object store behind a FastAPI gateway. **Verified
  end-to-end against a live server**, not just written to a guessed
  spec: authentication is `X-API-Key` (an earlier guess of
  `Authorization: Bearer` 401'd with "Invalid or expired token" -- a
  recognized-but-wrong scheme, which is what revealed it, not a bad
  key); bucket creation's request body field is `name`, not `bucket`
  (found via a 422 response that spelled out the exact mismatch). A
  model pushed, pulled back under a new name, and reloaded produced
  byte-identical predictions to the original.
- `kanboost/mlflow_utils.py` (new, `pip install kanboost[mlflow]`):
  `log_training_run(model, X_test, y_test, ...)` logs a fitted model's
  hyperparameters (`model.get_params()`), evaluation metrics
  (`model.evaluate()`), and optionally the saved model file, as one
  MLflow run -- via the standard `mlflow` client pointed at a tracking
  server, not a platform's own read-only REST wrapper (many self-hosted
  platforms only expose `GET`/`DELETE` on MLflow experiments/runs
  through their own API, with no way to *create* a run that way).
  Verified end-to-end against a local sqlite-backed tracking store.
- Both modules follow the same additive pattern as everything else this
  session: new files, heavy dependencies (`requests`, `mlflow`) lazily
  imported inside functions, zero changes to `_base.py`/`classifier.py`/
  `regressor.py`.
- 12 new tests (9 for `mlhub`, all `requests.post`/`get` mocked, no live
  server involved in the test suite itself; 3 for `mlflow_utils`,
  offline sqlite-backed).

**v0.0.18 — imbalanced classification, real fit-time speedup, kantun gaps**
- `kanboost/imbalance.py` (new): `balanced_weights(y)` / `find_threshold(model, X_val, y_val, metric="f1"|"youden")`.
  Root-caused and reproduced the exact degenerate-classifier bug
  surfaced by the arXiv:2509.16750 benchmark (AUC=0.87 but F1=0 on a
  90/10 split) on a synthetic dataset: `LogisticLoss` and
  `predict(threshold=0.5)` are each correct in isolation, but a
  well-calibrated model on a heavily imbalanced split legitimately
  outputs `p<0.5` almost everywhere, so the default threshold alone
  reads every score as negative. `find_threshold` was the dominant fix
  (F1 0.0 -> 0.545, AUC unchanged to `1e-9` since it only moves the
  cutoff, not the scores); `balanced_weights` alone was much weaker
  (F1 0.0 -> 0.04) and is documented as a complement, not a
  substitute. No changes to `losses.py`/`_base.py` -- deliberately did
  not "fix" `LogisticLoss` itself, since it isn't wrong.
- `kanboost/accel.py` (new): `fast_fit(model, X, y, ...)`, revisiting
  v0.0.14's deferred fit-time speedup with a profiler-driven approach
  instead of a new backend. Warm-starts each boosting round's learner
  from the previous round's `state_dict` (architecture is identical
  round-to-round; only the seed differs), so only each chain's first
  round needs the full `kan_steps` budget. Implemented as a temporary
  monkey-patch of the model instance's `_new_learner`/`_fit_learner`/
  `_boost_chain` bound methods, restored in a `finally` block --
  routes through the exact same `_fit_learner_custom_loop`/
  `_apply_monotone_projection` machinery as a normal `fit()`, so
  monotone constraints are enforced identically (verified: 0%
  violation rate with `monotone_constraints` set). Multiclass chains
  verified isolated (a new class's chain never warm-starts from a
  different class's last learner). Measured 3.37x faster on Breast
  Cancer Wisconsin (11.70s -> 3.48s, 40 estimators) with AUC
  essentially unchanged (0.9921 vs 0.9893). Zero changes to
  `_base.py`/`classifier.py`/`regressor.py`.
- `kantun` v0.0.4: `param_distributions` values may now be a callable
  `sampler(rng) -> value` (continuous/log-uniform ranges) for
  `search_type="random"`/`"halving"`; `scoring` may be a callable
  scorer, not just one of the built-in strings; new `time_budget_s`
  wall-clock cap, checked between combos/rungs (never mid-fit, always
  lets at least one combo/rung finish; for halving, promotes the best
  candidate from the last completed rung if the budget is hit before
  the final full-data rung); new `refit=False` to skip the final
  full-dataset refit. Also fixed a stale README install section that
  still said "once published to PyPI" for a package that had already
  been on PyPI since v0.0.3.
- New docs pages: `guide/imbalance.md`, `guide/training-speed.md`;
  updated `guide/tuning-with-kantun.md` and `docs/api.md`.
- Independent review (this session) caught a real bug before shipping:
  `fast_fit()`'s restore step reassigned the original bound methods
  onto the instance instead of removing the shadowing attributes,
  leaving `_new_learner`/`_fit_learner`/`_boost_chain` permanently in
  `model.__dict__` -- which broke `model.save()` (`_base.py`'s
  `_freeze` pickles `self.__dict__` wholesale, and a closure-holding
  local function can't be pickled). Empirically reproduced, then fixed
  by popping the three shadowing keys in the `finally` block instead
  of reassigning them, so attribute lookup falls back to the class.
- 6 new tests for `kanboost.imbalance`, 6 for `kanboost.accel`
  (including a `fast_fit -> save -> load -> predict_proba` parity
  regression test for the bug above), 6 new for `kantun` (callable
  param sampling, callable scoring, time budget on both flat and
  halving search, `refit=False`).

**v0.0.19 — `kanboost.symbolic.symbolic_summary()`, a one-call full-formula report**
- Motivated by hands-on exploration of `heart_model` (the
  arXiv:2509.16750 benchmark's `heart` dataset, `gam=True`,
  `kan_hidden=1`) via `export_symbolic`/`explain` in a notebook: getting
  a complete, amplitude-ranked report required several manual steps
  (rank by importance, fit candidates, re-fit unrestricted for the full
  formula, sort by amplitude). `symbolic_summary(model, min_r2=0.8,
  top_n=None)` does all of it in one call, returning `{"ranked_terms",
  "full_formula", "full_latex", "model"}`.
- Ranks by **amplitude** (how much a term actually moves the
  prediction), not `feature_importances_dict()`'s importance -- the
  same distinction `fidelity_report()`'s docstring has warned about
  since v0.0.16 (a high R^2 doesn't mean a term matters). Unlike
  `explain()`, which only fits candidates for its `top_features` count
  (leaving the rest as opaque `g_<feature>(x)` placeholders in the
  implied full formula), `symbolic_summary()` defaults to fitting
  candidates for *every* feature, so `full_formula` is a genuine closed
  form for the whole model, not a partial one.
- A real bug caught and fixed before landing: the first implementation
  restricted `top_n` to only the (expensive) candidate search, while
  `ranked_terms` still listed every feature in the model regardless of
  `top_n` -- caught by writing
  `test_symbolic_summary_top_n_restricts_ranked_terms_not_just_candidate_search`
  and seeing it fail (30 terms returned instead of the requested 5)
  before fixing `ranked_names` to filter to `features` when `top_n` is
  set, not just sort `fidelity_report()`'s full key set.
- Verified end-to-end on the real `heart` benchmark data via the
  updated example notebook: 21/21 features got a genuine closed-form
  term (`symbolic_fraction() == 1.0`, no numeric fallback), ranked by
  amplitude in an order clinically consistent with known heart-disease
  risk factors (`Stroke` history ranked highest, `NoDocbcCost` lowest),
  and the fitted curves visually overlaid the real per-feature curves
  from `kanboost.editing.consolidate()` almost exactly (R^2 > 0.9998
  for all 5 spot-checked features).
- Independent review (this session) before shipping: APPROVE WITH NITS,
  all addressed -- `top_n` values below 1 now raise `ValueError` instead
  of silently slicing (`top_n=-1` previously meant "all but one
  feature", which is surprising); softened an overclaim in the
  docstring/docs that `full_formula` never has a `g_<feature>(x)`
  placeholder (true only when every feature's best candidate clears
  `min_r2` and `top_n=None` -- a feature that genuinely fits nothing
  above `min_r2` still falls back correctly, by design); documented
  that `full_formula` reverts to per-cutoff placeholders once `top_n`
  is set, same as `explain()`.
- 4 new tests (adds a `top_n<1` rejection test to the 3 above). Zero
  changes to `_base.py`/`classifier.py`/`regressor.py` -- purely
  additive to the existing `kanboost/symbolic.py` module.

**v0.0.20 -- leaner, jointly-optimized, verified-stable symbolic equations**
- Motivated by a hands-on critique (this session) of `symbolic_summary()`'s
  output: too many terms, disproportionate `sin`/`cos` selection,
  over-trusting R^2 alone when choosing a candidate, low-amplitude
  features left in the final equation, constants fit independently per
  term rather than jointly, no measured guarantee the equation retains
  the model's ranking quality, and no visibility into whether a formula
  is stable across random seeds. Four additive fixes, one per concern:
  - **`export_symbolic(..., parsimony_margin=0.0)`**: a more complex
    candidate (fixed ranking, roughly
    `x < abs/x^2 < x^3/tanh < sqrt/sin/cos < log/exp`) only replaces a
    simpler one already found if it improves R^2 by more than the
    margin -- default `0.0` preserves the exact prior behavior.
    Empirically, `parsimony_margin=0.05` changed the selected candidate
    on 27 of 30 breast-cancer terms versus the unmargined default,
    confirming most default R^2 "wins" from a more complex candidate
    were marginal.
  - **`symbolic_summary(..., min_amplitude=None)`**: drops any term
    below the given amplitude from *both* `ranked_terms` and
    `full_formula` (rebuilt from only the retained terms, not
    `sym.to_sympy()`'s full model) -- a low-amplitude term barely moves
    the prediction regardless of its R^2.
  - **`refit_constants(sym, X_scaled, target)` /
    `refit_constants_from_model(model, sym, X)`**: jointly re-optimizes
    every symbolic term's `(a, b, c, d)` and the intercept (via
    `scipy.optimize.minimize`, L-BFGS-B) against the real trained
    model's own raw score, instead of each term's default fit --
    independently, to that one feature's isolated marginal curve from
    `consolidate()`. Numeric (spline-fallback) terms have no closed-form
    parameters and are left as-is. Verified on real data (Breast
    Cancer): mean absolute fidelity error dropped from 0.0054 to 0.0047
    after refitting, with AUC unchanged (0.9917), on a held-out split.
  - **`formula_fidelity(model, sym, X, y=None)` /
    `stability_across_seeds(build_and_fit, X, y, n_seeds=5, ...)`**: the
    first reports `max_abs_error`/`mean_abs_error` always, plus
    `auc_model`/`auc_equation` for a binary classifier when `y` is
    given, so "does the equation retain ranking quality" is a measured
    number, not an assumption; the second trains several independent
    models and reports each feature's modal-candidate agreement rate
    across seeds alongside per-seed fidelity, since boosting's
    stochastic training means a different seed can genuinely pick a
    different candidate function for the same feature -- a single
    extraction was never "the" formula to begin with.
- A structurally impossible request was caught and declined during
  scoping, not built: a proposed target equation included a genuine
  feature-interaction term (`C3 * A3`). `gam=True`'s additive
  decomposition (`F(x) = c + sum_j g_j(x_j)`) cannot represent a
  cross-feature term by construction -- that constraint is exactly what
  makes `monotone_constraints` and `consolidate()`'s exact centering
  possible in the first place. Getting a genuine interaction term would
  require either abandoning pure `gam=True` (`kan_hidden > 1`, losing
  the additive separation the current extraction depends on) or a
  wholly separate black-box symbolic-regression distiller (e.g.
  PySR/genetic programming against the ensemble's raw predictions) --
  flagged as a candidate future experiment, not folded into this
  release.
- Independent review (this session) caught a real blocking bug before
  shipping: `stability_across_seeds()` recorded a numeric-fallback
  term's `candidate` as `None`, and pandas' `value_counts()` silently
  drops `None` -- a feature that's numeric-fallback in *every* seed
  crashed (`IndexError`, empty `value_counts()`), and a feature numeric
  in most seeds but symbolic in one reported a false `modal_agreement`
  of 1.0 (the `None` rows vanished from the denominator) -- the exact
  opposite of the instability the function exists to expose. `candidate`
  is `None` only when every candidate's `fit_params()` call raises (an
  all-NaN curve, verified separately) -- deliberately fitting a
  min-R^2-forced numeric fallback in the regression test still leaves a
  real (if rejected) `candidate` name, not `None`, so
  `test_stability_across_seeds_handles_numeric_fallback_feature` instead
  monkeypatches `symbolic_summary()` to isolate the test to
  `stability_across_seeds()`'s own candidate-recording logic. Fixed by
  recording `"numeric"` instead of `None`.
- 7 new tests. Zero changes to `_base.py`/`classifier.py`/`regressor.py`
  -- purely additive to the existing `kanboost/symbolic.py` module.

## Deferred (with reasons)

- **`torch.compile` / ONNX export / FastKAN backend** — pykan's `KAN`
  modules don't trace or compile cleanly out of the box; would need
  upstream changes or a from-scratch spline layer.
- **Multi-GPU** — each weak learner is tiny; the bottleneck is the
  number of sequential boosting rounds, not per-learner compute, so
  multi-GPU wouldn't help without changing the training loop's
  architecture.
- **Benchmark suite vs. XGBoost/LightGBM/CatBoost on standard UCI
  datasets** — a legitimate next step, but an ongoing measurement effort
  rather than a library feature; tracked separately from this code
  roadmap. (The docs site itself shipped in v0.0.11 -- see below.)
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
