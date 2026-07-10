# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

KANBoost is a from-scratch gradient boosting library that uses shallow
Kolmogorov-Arnold Networks (KAN, via the `pykan` package) as weak learners
instead of decision trees, following the classic Friedman (2001) boosting
recipe. It's a research-stage, pip-installable sklearn-compatible library
(`KANBoostClassifier` / `KANBoostRegressor`), not a production drop-in
replacement for XGBoost/LightGBM/CatBoost — see the README's "Benchmarks"
and "Honest limitations" sections before assuming a change should chase
tree-boosting parity.

## Commands

```bash
pip install -e .              # install for development
pip install -e ".[api,dashboard,mlhub,mlflow,docs]"  # + optional extras as needed

pytest tests/ -q               # run the full suite (mirrors .github/workflows/tests.yml)
pytest tests/test_kanboost.py -q                       # one test file
pytest tests/test_kanboost.py::test_name -q            # one test

mkdocs serve                   # preview docs locally (requires the `docs` extra)
```

There is no configured linter/formatter (no ruff/flake8/black/mypy config
in this repo) — match the existing code style rather than introducing a
new tool. CI (`.github/workflows/tests.yml`) runs `pytest tests/ -q` on
Python 3.10 and 3.12 for every push/PR to `main`; that's the bar a change
needs to clear. `.github/workflows/publish.yml` builds and publishes to
PyPI via Trusted Publishing on GitHub release — bump `version` in
`pyproject.toml` (and `__version__` in `kanboost/__init__.py`) together
when cutting a release.

## Architecture

### Core boosting loop (`kanboost/_base.py`, `classifier.py`, `regressor.py`)

`_BaseKANBoost` (in `_base.py`) is the shared, private base class for both
public estimators; `KANBoostClassifier` and `KANBoostRegressor` subclass it
alongside sklearn's `ClassifierMixin`/`RegressorMixin`, making both
sklearn-compatible (`get_params`/`set_params`, `clone()`, `Pipeline`,
`GridSearchCV`). The fit loop, common to both:

1. `TabularPreprocessor` (`encoders.py`) fits on the training data:
   numeric columns get median imputation (+ optional `<col>_missing`
   indicator), outlier clipping, `RobustScaler`, then clip to `[-1, 1]`
   (KAN's spline grid range); categorical columns (`categorical_cols=`)
   get smoothed target-mean encoding computed only on the training fold.
2. `F_0` initializes to the base rate (log-odds for classification, mean
   for regression).
3. Each boosting round fits one small `pykan.KAN` instance
   (`_fit_learner`/`_fit_learner_custom_loop` in `_base.py`) to the
   pseudo-residuals of the previous stage, then adds
   `learning_rate * f_t(x)` to the running prediction.
4. If `early_stopping_rounds` is set (with an explicit `eval_set` or an
   auto-carved `validation_fraction` split), the loop stops once
   validation loss stalls.
5. Multiclass classification is **one-vs-rest**: `n_classes` independent
   binary boosting chains, combined via softmax at predict time — not a
   single joint objective. This is why multiclass training cost scales
   linearly with class count.
6. Losses live in `losses.py` (`LogisticLoss`, `SquaredLoss`,
   `QuantileLoss`); which one is used depends on estimator/task
   (`objective=` for the regressor).

`_base.py` also owns everything that reads the trained KAN weights
directly rather than treating them as an opaque model: `feature_importances`,
`predict_derivative` (analytic derivatives — a real advantage over
tree/MLP models, which only have pointwise autograd gradients or none),
`monotone_constraints` enforcement (projects B-spline control points onto
the monotone cone every step — a hard guarantee, only valid when
`gam=True` and `kan_hidden=1`), `refine`/`prune` (re-express a fitted
ensemble on a different grid or zero out dead edges), `feature_interaction`,
and `save`/`load` (persists to a single file via `torch.save`).

`gam=True` fixes each learner's output edge to identity, making the whole
ensemble an exact additive model `F(x) = c + sum_j g_j(x_j)`. Most of the
"interpretability" modules below (`editing`, `symbolic`, structural
monotonicity) only work in this mode — check for a `gam=True` requirement
before assuming a feature applies to a general model.

### Everything else is additive, one-directional, and optional

A strong convention in this codebase: modules outside `_base.py`/
`classifier.py`/`regressor.py`/`encoders.py`/`losses.py` only ever *consume*
the public model API (`fit`, `predict`, `predict_proba`,
`feature_importances_dict`, `model.verbose`, `model._fit_learner`, etc.) —
they never get imported by core training/inference code, and adding one
should never require touching the boosting loop. Follow this pattern for
new functionality: put it in its own module, have it accept a fitted
model as an argument, and gate any new hard dependency behind a
`pyproject.toml` optional-dependency extra (see `api`, `dashboard`,
`mlhub`, `mlflow`, `docs`) rather than adding it to the base
`dependencies` list. Each of these has a matching `tests/test_<module>.py`:

- **`editing.py`** — `consolidate(model)` (requires `gam=True`) collapses
  a fitted ensemble's per-feature shape function into one `EditableGAM`
  spline per feature that a human can directly reshape
  (`set_offset`/`set_values`/`enforce_monotone`), and which also predicts
  ~30-50x faster than replaying every boosting round.
- **`symbolic.py`** — `export_symbolic(model)` (requires `gam=True`) fits
  a closed-form (`sympy`) function per feature from pykan's symbolic
  library; `explain()`/`symbolic_summary()` are convenience wrappers.
- **`calibration.py`** — `calibrate(model, X_cal, y_cal)` post-hoc
  Platt/isotonic-rescales `predict_proba` output; fixes a real,
  benchmark-documented miscalibration gap (see README "Calibration")
  without retraining. Must be re-run after any `EditableGAM` edit.
- **`imbalance.py`** — `find_threshold`/`balanced_weights` for skewed
  targets, where the default `threshold=0.5` can degenerate to
  always-majority-class predictions.
- **`accel.py`** — `fast_fit(model, X, y)`, an opt-in warm-started
  training path (~3x faster, same accuracy).
- **`experimental.py`** — small heuristics built purely on the public API
  (`suggest_constraints`, `audit_monotonicity`, `predict_interval`,
  `explain_row`, `dashboard_html`). `suggest_constraints` is advisory only
  — always verify with `audit_monotonicity` on a model actually trained
  with the suggested constraint.
- **`observability.py`** / **`logging_utils.py`** — timing/memory/GPU
  introspection and stdlib logging; no extra dependency.
- **`serving.py`** — FastAPI app factory (`create_app`, extra: `api`);
  `dashboard.py`/`_dashboard_app.py` — local Streamlit app (extra:
  `dashboard`).
- **`mlhub.py`** — push/pull a saved model to a MinIO-backed store behind
  a FastAPI gateway (extra: `mlhub`); **`mlflow_utils.py`** — log a run's
  params/metrics/model to MLflow (extra: `mlflow`). Independent of each
  other.
- **`metrics.py`** — `classification_report_dict`/`print_classification_report`,
  the two names re-exported from the top-level `kanboost` package
  alongside the two estimators.

Hyperparameter tuning intentionally lives in a **separate** package
(`kantun`, not in this repo) so kanboost's core dependency footprint stays
minimal — don't add tuning logic here.

### Docs

`docs/` (mkdocs-material, `mkdocs.yml` nav) has one guide per optional
module. `docs/roadmap.md` deliberately just points at `ROADMAP.md` at the
repo root rather than duplicating it, to avoid the two going out of sync
— update `ROADMAP.md`, not `docs/roadmap.md`, when roadmap status changes.
`README.md`'s benchmark tables are real, reproducible results (several
independent runs, methodology noted inline) — don't edit numbers there
without re-running the corresponding script/notebook in `examples/`.
