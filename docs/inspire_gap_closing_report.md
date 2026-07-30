# Closing the KANBoost-vs-Trees Gap on INSPIRE (postop_icu)

**Date**: 2026-07-30
**Dataset**: INSPIRE (Seoul National University Hospital perioperative dataset), local copy at `INSPIRE/operations.csv` (130,960 rows, 99,886 patients).
**Task**: binary classification, `postop_icu` = whether `icuin_time` is non-null (11.4% positive rate).
**Features**: `age, sex, weight, height, race, asa, emop, department, antype, icd10_pcs` (5 numeric, 5 categorical).
**Split**: group-aware (by `subject_id`) 60/20/20 via `StratifiedGroupKFold`, fixed across all scales. Three nested training scales (Small ⊂ Medium ⊂ Large by patient): 10,000 / 50,000 / 78,496 rows.
**Hardware**: CPU-only for this run (torch build installed here is `2.11.0+cpu`; an NVIDIA GPU is present but not usable by this torch build — KANBoost's `device="cuda"` path was not exercised).
**kanboost version**: 1.3.0 (installed editable from `G:\My Drive\project\kanboost`, includes the 2026-07-22 out-of-fold target-encoding leakage fix and the `categorical_hierarchy` feature).

## 1. Baseline reproduction (no fixes)

Original notebook design: One-Hot encoding (`min_frequency=50`) for KANBoost's categorical features, Ordinal encoding for the tree models, no class weighting, threshold chosen by scanning balanced accuracy.

| Scale | Model | AUROC | AUPRC | F1 | Fit time |
|---|---|---:|---:|---:|---:|
| Small | KANBoost | 0.9104 | 0.6255 | 0.5467 | 40.0s |
| Small | Best tree (HistGB) | 0.9329 | 0.7167 | — | 2.6s |
| Medium | KANBoost | 0.9176 | 0.6413 | 0.5796 | 612.8s |
| Medium | Best tree (LightGBM) | 0.9535 | 0.8081 | 0.6461 | 1.4s |
| Large | KANBoost | 0.9189 | 0.6406 | 0.5711 | 2295.0s |
| Large | Best tree (LightGBM) | 0.9562 | 0.8160 | 0.6545 | 2.7s |

Gap widened with scale (AUROC gap 0.023 → 0.036 → 0.037), and KANBoost's training time grew far faster than the trees' (7-850x slower depending on scale).

## 2. Root-cause diagnosis

Read directly from kanboost's own source and docs (not assumed from generic KAN literature):

- `kanboost/core/kan/network.py`: as of v1.2.1 (2026-07-18), KANBoost's weak learner (`DeepKAN`) is **not** a torch/Adam network — it's a closed-form numpy/scipy ALS (non-GAM) / P-spline (GAM) solver. The `opt="Adam"`/`kan_steps` arguments passed by `classifier.py` are accepted for API compatibility but ignored by the solver (`steps` becomes an ALS sweep cap, capped at 10).
- `kanboost/core/kan/bspline.py`: the B-spline basis evaluation is JIT-compiled via numba (measured 6.5x faster than the scipy fallback), and `_boost_chain`/`_fit_als` share a `basis_cache` across the whole boosting chain (layer-0 design matrix + its eigendecomposition computed once, reused every round) — so the classic "PolyKAN/FastKAN-style kernel fusion" opportunity that generic KAN literature targets is **already implemented** here, just not under those names.
- `kanboost/docs/guide/classification.md` / `calibration.md` / `imbalance.md`: three independent benchmarks documented by the project itself already characterize the exact failure modes reproduced here — weak-learner capacity degrading with dataset size, systematic probability miscalibration (worst Brier/log-loss, F1-optimal threshold ~0.40-0.42 not 0.5), and a specific imbalanced-classification failure mode (init_pred_ starts at the true base-rate log-odds; unweighted MSE per round is dominated by the majority class).
- One-Hot encoding was the actual encoding used for KANBoost in the notebook that produced the first three_scale_results — but `kanboost`'s own `TabularPreprocessor` (`categorical_cols=...`) does smoothed, K-fold out-of-fold target-mean encoding instead, which is far narrower (fewer columns) and better matched to a spline-based model than sparse one-hot columns, especially for the high-cardinality `icd10_pcs` column.

## 3. Fixes adopted (validated empirically on this dataset, not assumed from literature)

Five fixes, layered incrementally and measured at each step on the Small scale before being confirmed at Medium/Large:

1. **Native target-mean categorical encoding** (`categorical_cols=CATEGORICAL_FEATURES`) instead of manual One-Hot.
2. **Class-balanced `sample_weight`** (`sklearn.utils.class_weight.compute_sample_weight("balanced", y_train)`).
3. **2x weak-learner capacity** (`kan_hidden`, `kan_grid` both doubled per scale).
4. **`kanboost.train.imbalance.find_threshold(cal_model, X_val, y_val, metric="f1")`** instead of a manual balanced-accuracy threshold scan — per the project's own docs, the dominant fix for the imbalance-driven F1=0 failure mode.
5. **`kanboost.train.calibration.calibrate(model, X_val, y_val, method="platt")`** — post-hoc, monotone (AUROC/AUPRC unchanged), corrects the documented systematic miscalibration.

### Result: all three scales, adopted 5 fixes vs. baseline vs. best tree

| Scale | Model | AUROC | AUPRC | F1 | Fit time |
|---|---|---:|---:|---:|---:|
| Small | KANBoost baseline | 0.9104 | 0.6255 | 0.5467 | 40.0s |
| Small | **KANBoost, 5 fixes** | **0.9221** | **0.6690** | **0.6202** | 40.2s |
| Small | Best tree | 0.9329 | 0.7167 | — | 2.6s |
| Medium | KANBoost baseline | 0.9176 | 0.6413 | 0.5796 | 612.8s |
| Medium | **KANBoost, 5 fixes** | **0.9388** | **0.7557** | **0.6797** | **389.9s** |
| Medium | Best tree | 0.9535 | 0.8081 | 0.6461 | 1.4s |
| Large | KANBoost baseline | 0.9189 | 0.6406 | 0.5711 | 2295.0s |
| Large | **KANBoost, 5 fixes** | **0.9413** | **0.7636** | **0.6831** | **1104.9s** |
| Large | Best tree | 0.9562 | 0.8160 | 0.6545 | 2.7s |

The 5 fixes **both** narrowed the accuracy gap **and** cut training time, at the two larger scales (1.6x faster at Medium, 2.1x faster at Large) — the opposite of the documented "capacity vs. speed" tradeoff, because the encoding fix shrinks the feature count enough to more than offset the doubled kan_hidden/kan_grid cost.

## 4. Three further techniques tested, one adopted

Requested explicitly as a 3-part experiment (RBF basis / "kernel fusion" / GrowNet-style consolidation), tested on real data rather than assumed from literature:

| Technique | Result | Verdict |
|---|---|---|
| **RBF basis instead of B-spline** (monkeypatched `_b_basis_1d`/`_b_basis_deriv_1d`, FastKAN-style) | **2.2x SLOWER** on Small (88.8s vs 40.2s), slightly worse accuracy | **Rejected.** kanboost's B-spline is already numba-JIT'd; the naive-numpy RBF replacement added overhead instead of removing it. The literature's claimed speedup doesn't transfer to an already-optimized baseline. |
| "Kernel fusion" (PolyKAN-style) | N/A — no separate lever exists | kanboost's actual bottleneck is a closed-form ALS solver, not a torch/CUDA training loop; there is no fused-kernel opportunity of the kind PolyKAN targets in this architecture. |
| **Post-hoc consolidation** (GrowNet-inspired: replace consecutive groups of 5 learners with one, least-squares-refit via `model._fit_learner` to reproduce the group's summed output) | Learners cut 5x (100→20 / 140→28 / 180→36); **prediction time cut 4-8x** (Medium 57.8s→7.2s, Large 54.1s→14.3s); accuracy cost minimal (AUROC −0.002, AUPRC −0.004 to −0.006) | **Adopted.** |
| **`categorical_hierarchy={"icd10_pcs": "department"}`** (kanboost v1.3.0 feature, 2026-07-22, not previously tried) | Small/Medium: no change. **Large: training 1.8x faster (1104.9s → 612.3s) for near-identical accuracy** (AUROC −0.0003, AUPRC −0.0019) | **Adopted** (helps convergence speed on the largest scale; neutral elsewhere). |

## 5. Final combined configuration ("ULTIMATE") vs. best tree, Large scale

All five adopted fixes + `categorical_hierarchy` + post-hoc consolidation (group size 5), evaluated once on Large:

| Model | Fit+consolidate time | Predict time | Learners | AUROC | AUPRC | F1 |
|---|---:|---:|---:|---:|---:|---:|
| KANBoost baseline | 2295.0s | (slow, one-hot 255-col input) | 180 | 0.9189 | 0.6406 | 0.5711 |
| KANBoost, 5 fixes only | 1104.9s | 54.1s | 180 | 0.9413 | 0.7636 | 0.6831 |
| **KANBoost ULTIMATE** | **879.7s** | **10.8s** | **36** | 0.9387 | 0.7580 | **0.6789** |
| Best tree (LightGBM) | 2.7s | ~0.1s | — | 0.9562 | 0.8160 | 0.6545 |

**vs. baseline**: 2.6x faster to train, far fewer learners to predict with, AUROC +0.020, AUPRC +0.117, F1 +0.108.
**Remaining gap to best tree**: AUROC 0.0175 (down from 0.0373), AUPRC 0.0580 (down from 0.1754) — both roughly halved or better. **F1 (0.679) now exceeds the best tree's (0.655).**

**Tradeoff, stated honestly**: ULTIMATE trades a small amount of AUROC/AUPRC (vs. the 5-fixes-only variant) for a much smaller, much faster-to-predict model. Use "5 fixes only" if peak ranking accuracy is the sole priority; use ULTIMATE if repeated/deployed inference matters (the more realistic scenario for a clinical postop_icu predictor).

## 6. What was NOT done, and why

- **No GPU run**: this machine's torch build has no CUDA support (`torch.version.cuda is None`), despite an NVIDIA GPU + driver being present. All numbers above are CPU-only; installing a CUDA-enabled torch build was out of scope for this exercise but would likely help the trees marginally and KANBoost more (per kanboost's own device-selection code path).
- **No true GrowNet joint-corrective boosting**: a faithful implementation (joint backprop across multiple weak learners against a fully corrective objective) was judged too large a change to build safely, correctly, and quickly against kanboost's closed-form ALS solver in this session. The post-hoc consolidation step adopted here is an explicitly-labeled, safer stand-in that captures the "reduce redundant small corrections" idea without touching the training loop's math.
- **This work has not been through kanboost's own review pipeline** (see `AI_REVIEW_LOOP.md`: ChatGPT hypothesis → Claude Code implementation → Codex independent review → ChatGPT scientific judgment → user merge approval). All code here lives in a personal scratchpad directory outside the `kanboost/` package, uses monkeypatching and private (`_`-prefixed) methods not covered by the test suite, and has not been added to `kanboost/tests/`.

## 7. Reproducing this report

Scripts (scratchpad, not part of the `kanboost` package):
- `inspire_three_scales.py` — baseline (one-hot, no fixes), all 3 scales vs. trees.
- `inspire_kanboost_improved.py` — 5 fixes (encoding + balance + capacity), later extended with `find_threshold`/`calibrate`.
- `inspire_kanboost_hierarchy.py` — 5 fixes + `categorical_hierarchy`.
- `inspire_kanboost_v2.py` — RBF-basis toggle + post-hoc consolidation (used to isolate each technique).
- `inspire_kanboost_ultimate.py` — final combined configuration (Large only).

Raw result CSVs are alongside these scripts in the same scratchpad directory.
