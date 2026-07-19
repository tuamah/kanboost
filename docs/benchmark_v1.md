# Benchmark v1 — DeepKAN Baseline (v1.2.2)

This page documents the official v1.2.2 baseline: the default training
behavior, the accuracy/speed numbers behind it, and the engineering
decisions (including rejected ones) that led here. Any future
optimization should be measured against these numbers, not against
"feels faster."

## Default training behavior (frozen as of v1.2.2)

| Setting | Value |
|---|---|
| Non-GAM solver | `eigh` (truncated eigenvalue pseudo-inverse) |
| ALS internal convergence tolerance | `1e-3` |
| Early stopping | enabled automatically when `validation_fraction` is set |
| B-spline evaluation | `numba`-accelerated when installed (`pip install kanboost[accel]`), scipy fallback otherwise |
| GAM solver | closed-form P-spline solve (unchanged, unaffected by any of the above) |

**Important, easy-to-miss default:** `early_stopping_rounds` defaults to
`10`, but it does nothing unless `validation_fraction` (or an explicit
`eval_set`) is also set — there is no internal validation split without
it. Set `validation_fraction=0.15` (or similar) explicitly if you want
early stopping active; its absence is not "no early stopping," it's
"early stopping silently disabled."

## 1. Experimental setup

- **Python:** 3.10.11
- **OS:** Windows 10 (Windows-10-10.0.26200-SP0)
- **kanboost:** v1.2.2, `numba` accelerator installed
- **Datasets:**
  - Friedman #1, 500 samples, 5 features (PMLB `649_fri_c0_500_5`)
  - Friedman #1, 1000 samples, 10 features — 5 relevant + 5 pure noise (PMLB `595_fri_c0_1000_10`)
  - UCI Concrete Compressive Strength, 1030 samples, 8 features
  - California Housing, subsampled to 3000 of 20,640 rows, 8 features (full-size run is impractically slow at `kan_hidden=64` — see Engineering Decisions)
  - Diabetes, 442 samples, 10 features
- **Preprocessing:** kanboost's built-in `TabularPreprocessor` (automatic scaling/encoding); no external preprocessing applied
- **Metrics:** R², RMSE, wall-clock fit time; Train R² / Validation R² / Overfit gap via a separate 80/20 diagnostic split
- **CV strategy:** 5-fold `KFold(shuffle=True, random_state=42)` for the reported R²/RMSE/time; a single 80/20 `train_test_split(random_state=42)` for the overfitting diagnostic (kept separate from the CV numbers on purpose — CV is the accuracy judgment, the 80/20 split is only for detecting train/val gaps)
- **KANBoost config:** `n_estimators=60, kan_hidden=64, kan_grid=3, learning_rate=0.1, gam=False` (non-GAM); `n_estimators=30, kan_hidden=1, gam=True` (GAM). Tree ensembles: `n_estimators=300, max_depth=4, learning_rate=0.05` (XGBoost/LightGBM/CatBoost defaults used throughout this benchmark, not separately tuned)

`kan_hidden=64, kan_grid=3` was arrived at via a systematic diagnostic
sweep on Friedman-1000 specifically (see
[Engineering Decisions](#4-engineering-decisions)) — it is the single
config used across all five datasets below for a fair, honest
comparison, **not** a per-dataset-optimal config. Concrete specifically
does measurably better with `kan_grid=8` (0.909 vs 0.893); this is
intentionally not special-cased here.

## 2. Accuracy results (5-fold CV, R²)

| Dataset | XGBoost | LightGBM | CatBoost | **KANBoost** | KANBoost-GAM |
|---|---|---|---|---|---|
| Friedman-500 | 0.904 | 0.903 | 0.932 | **0.935** | 0.891 |
| Friedman-1000 | 0.915 | 0.911 | 0.934 | **0.919** | 0.871 |
| Concrete | 0.931 | 0.917 | 0.914 | 0.887 | 0.841 |
| California Housing (n=3000) | 0.782 | 0.779 | 0.769 | 0.710 | 0.653 |
| Diabetes (n=442) | 0.385 | 0.398 | 0.462 | 0.322 | 0.426 |

**Reading:** KANBoost matches or beats XGBoost/LightGBM on the two
Friedman datasets (best model overall on Friedman-500) using a single
non-tuned-per-dataset config. It trails on Concrete (grid=3 is not this
dataset's optimum), California Housing, and Diabetes.

### Overfitting diagnostic (Train R² / Val R², 80/20 split)

| Model | Diabetes gap | California Housing gap |
|---|---|---|
| XGBoost | 0.584 | 0.143 |
| LightGBM | 0.472 | 0.112 |
| CatBoost | 0.329 | 0.068 |
| **KANBoost** | **0.170** | **0.043** |
| KANBoost-GAM | 0.021 | -0.008 |

**Important correction to an early misdiagnosis:** KANBoost's lower
absolute R² on Diabetes is *not* classic overfitting — its train/val
gap is smaller than every tree ensemble's, in some cases dramatically
so (XGBoost's gap is 3.4x KANBoost's). The low absolute score reflects
a different limitation (see Engineering Decisions — most of this gap
turned out to be a benchmark-harness bug: early stopping was silently
disabled).

## 3. Speed results

### B-spline evaluation kernel

| | Time (18,000 calls, representative shape) |
|---|---|
| scipy (`BSpline.design_matrix` + `.toarray()`) | 2.75s |
| numba JIT dense Cox-de-Boor | **0.42s (6.5x faster)** |

A hand-written pure-numpy Cox-de-Boor recursion was tried first and was
*slower* than scipy (Python/numpy loop overhead outweighs skipping
scipy's sparse-matrix construction). Only a compiled (JIT) kernel wins.

### Full training time, cumulative optimization chain (Friedman-500, `n_estimators=60`, 5-fold CV)

| Stage | Time | Cumulative speedup |
|---|---|---|
| Baseline (v1.2.1) | 34.1s | 1x |
| + eigh factorization hoisted (once per sweep, not per output/hidden-unit) | 23.4s | 1.5x |
| + layer-0 basis matrix cached within one fit | 13.4s | 2.5x |
| + layer-1 design matrix reused within one sweep | 11.2s | 3.0x |
| + init-loop redundancy removed | 9.3s | 3.7x |
| + basis/eigendecomposition cached across the whole boosting chain | 8.8s | 3.9x |
| + numba B-spline kernel | **3.7s** | **9.2x** |

Friedman-1000 and Concrete showed 6.8x and 7.4x respectively over the
same chain.

### ALS convergence tolerance fix (separate, later fix — measured at `kan_hidden=64`)

| | California Housing (n=3000) | Friedman-1000 |
|---|---|---|
| Tolerance `1e-6` (never triggers in practice) | 0.710 R², ~193s | 0.919 R², ~89-96s |
| Tolerance `1e-3` (adopted default) | **0.710 R², 121.9s** | **0.927 R², 50.7s** |

~1.6-1.9x additional speedup, **no accuracy cost** — confirmed via full
5-fold CV (not a single split) on both datasets. This was one existing,
already-tested mechanism (per-learner ALS early-break on training-loss
convergence) whose threshold had simply never been loosened from an
overly strict default.

**Honest caveat:** the 9.2x chain above and the 1.6-1.9x tolerance fix
were measured in separate rounds, at different `kan_hidden` values
(mostly 8 for the former, 64 for the latter) — they are not simply
multiplicative, and the exact combined number at `kan_hidden=64` with
every optimization stacked has not been separately re-measured
end-to-end. Treat "up to ~9x" and "up to ~1.9x on top of that" as two
distinct, independently-verified findings, not a single ~17x claim.

## 4. Engineering decisions

### Adopted

- **Three-tier basis caching** (within-sweep, within-fit, across-chain) — pure performance, bit-identical R² at every step, verified before each tier was trusted.
- **Optional numba B-spline kernel** — 6.5x per-call, opt-in via `kanboost[accel]`, scipy fallback preserves zero-hard-dependency installs.
- **ALS tolerance `1e-6` → `1e-3`** — the existing early-break mechanism never fired at `1e-6`; loosening it is a one-constant change with a measured, safe win on two very different datasets.
- **`kan_hidden`/`kan_grid` diagnostic sweep methodology** — established that no single config generalizes across dataset shapes; documented here rather than automated into a preset (see below).

### Rejected — Cholesky instead of `eigh`

Cholesky is ~4-10x cheaper than `eigh` at the same matrix size and was
a natural target once profiling showed `eigh` at ~30% of fit time at
`kan_hidden=64`. **Rejected as numerically unsafe.** ALS's
normal-equations matrix routinely has condition numbers up to
`~1e16-1e17` (near-duplicate/collinear hidden-unit columns). A small
diagonal jitter large enough for `cho_factor` to succeed is not
equivalent to `eigh`'s relative eigenvalue-truncation floor — these are
different regularizers (uniform Tikhonov shift vs. selective
truncation of only the smallest eigenvalues). On a synthetic
near-singular test case (condition number ~3.7e17), Cholesky-with-jitter
returned a solution norm of **6.4 million** against `eigh`'s correct
**148** — silently, badly wrong, not merely slower-but-correct. Making
the jitter track the true spectral radius safely would need a cheap
max-eigenvalue estimate (e.g. power iteration), which erodes most of
the expected win. Left as `eigh`.

### Rejected as a default — cross-round warm-starting + capped ALS sweeps

`kanboost.train.accel.fast_fit()` (pre-existing) plus a reduced sweep
budget gave a 3.2x speedup on California Housing with negligible
accuracy cost (-0.0012 R²). The *same* configuration cost a real 0.04
R² on Friedman-1000 (confirmed via 5-fold CV, not a single split — the
gap far exceeds fold-level noise). Mechanism: warm-starting biases a
new learner toward the *previous* round's residual shape, which is
actively wrong when the residual's structure changes quickly
round-to-round (e.g. Friedman's `sin(π·x1·x2)` term). **Kept opt-in
only** (`fast_fit`, already existed) — not promoted to a default.

### Rejected — named preset/mode API (`preset='balanced'/'accuracy'`, `mode='fast'/'accurate'`)

Proposed twice this development cycle, rejected both times. The
underlying parameters (`kan_hidden`, `kan_grid`, `kan_steps`, `fast_fit()`)
already exist as explicit, documented arguments — a preset adds a
lookup table, precedence rules, and a permanent naming commitment for
numbers tuned on a single dataset, without adding any real capability.
Documentation of measured tradeoffs (this page) is preferred over
bundled convenience API.

### Not pursued (evidence doesn't support it yet)

Hessian/second-order boosting, a C++/CUDA compute engine, and new
basis families (Fourier/RBF/rational) were all proposed at various
points and set aside: for squared-error regression, Hessian boosting is
mathematically identical to the existing first-order gradient (the
loss's second derivative is a constant 1), so it cannot change anything
for KANBoost's default regression objective. A C++ engine's realistic
ceiling (the remaining, non-cacheable per-sweep basis evaluation) was
estimated as a small fraction of total time — not worth the permanent
cross-platform build/packaging burden for a currently pure-Python
project, especially once `numba` delivered most of the available win
without one.

## What v1 means going forward

Any future change to KANBoost's training path must answer one
question: **does it produce a measured improvement over this baseline
on at least the five datasets above (via 5-fold CV, not a single
split)?** "Looks like it should be faster/more accurate" is not
sufficient — see this page's own Cholesky and warm-starting examples
above for why a single-dataset or non-CV measurement is not enough.

The next reasonable phase (not started) is **adaptive efficiency**:
reducing time without changing results (smarter ALS scheduling,
selective solver calls) and improving UX (data-size-aware
configuration guidance — as documentation, per the API-surface decision
above, not a new bundled parameter) — plus benchmarking on a wider
range of real-world tabular datasets (medical, physical sciences)
before drawing further general conclusions.
