# Closing the KANBoost-vs-Trees Gap on INSPIRE (postop_icu)

> **This is the chronological engineering log** (every fix tried, in order, with code). For the audited, scientifically defensible summary — including the leakage audit, mixed-code robustness analysis, and a fair (threshold-tuned-on-both-sides) tree comparison — see **[`inspire_kanboost_evaluation.md`](inspire_kanboost_evaluation.md)**.

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

## 6. External benchmark comparison and a leakage audit

Before treating the numbers above as competitive with the literature, they were checked against published work on the same task and dataset, and the unusually high AUPRC was independently audited for leakage rather than assumed innocent.

### 6.1 The only directly comparable published benchmark

The INSPIRE descriptor paper itself (Nature Scientific Data) only validates a **30-day mortality** task (best model: gradient boosting, AUROC 0.944, 1.21% event rate) — it does not benchmark ICU admission at all. The one paper found that benchmarks the same `postop_icu` task on the same dataset is **"Comparison of large language models and conventional machine learning in postoperative outcome prediction"** (*Korean J Anesthesiol*, `10.4097/kja.25646`), which compares XGBoost against GPT-4o/Llama-3-70B/OpenBioLLM-70B:

| Model | AUROC | AUPRC | Event rate | Test n |
|---|---:|---:|---:|---:|
| XGBoost (that paper) | 0.851 | 0.208 | 5.62% | 8,099 |
| GPT-4o / Llama-3-70B / OpenBioLLM-70B | 0.70–0.76 | 0.11–0.14 | 5.62% | 8,099 |
| **This report's LightGBM (Large)** | 0.956 | 0.816 | 11.41% | 26,267 |
| **This report's KANBoost, 5 fixes (Large)** | 0.941 | 0.764 | 11.41% | 26,267 |

Two things worth noting on their own: (1) even frontier LLMs lose to plain XGBoost on this structured task — independent confirmation that gradient-boosted trees (and, by extension, KANBoost's boosting-of-splines design) are the right family for this kind of clinical tabular problem, not large language models; (2) this report's numbers are substantially higher on every metric — large enough to demand scrutiny before drawing any "we beat the literature" conclusion.

### 6.2 Why the event rate differs: confirmed from PhysioNet's own documentation

PhysioNet's INSPIRE page states explicitly: *"Only ICU admissions within 24 hours of surgery were included in the `icuin_time` and `icuout_time` columns to focus on ICU admissions immediately after surgery."* Measuring the gap between `orout_time` (OR exit) and `icuin_time` directly on the local data confirms this: median gap is exactly 0 hours, essentially 100% of ICU admissions occur within an hour of leaving the OR. There is **no field anywhere in the schema distinguishing "planned/routine" from "unplanned/unanticipated" ICU admission** — `postop_icu` as defined here (`icuin_time.notna()`, 11.41% positive) is structurally an "immediate-transfer" flag, not a "complication occurred" flag.

The one data-grounded correction available is excluding cardiopulmonary bypass cases (`cpbon_time` not null — on-pump cardiac surgery is essentially always routed to ICU as routine care, not as a complication): this drops the positive rate from 11.41% to 9.69%, narrowing but not closing the gap to the published paper's 5.62% — the remainder is most likely additional cohort-construction criteria in that specific study (e.g., excluding whole departments/procedure classes) that aren't recoverable from this table alone.

### 6.3 Leakage audit: is `icd10_pcs` leaking outcome information?

The unusually high AUPRC (0.76–0.82 here vs. 0.11–0.21 in the published mortality/ICU papers) was checked directly rather than dismissed. A feature-ablation run (LightGBM, same group-aware split, Large scale) isolates exactly where the signal comes from:

| Feature set | AUROC | AUPRC |
|---|---:|---:|
| `department + emop + asa + age` only (no `icd10_pcs`) | 0.914 | 0.603 |
| + `antype` | 0.918 | 0.618 |
| + full demographics (still no `icd10_pcs`) | 0.921 | 0.630 |
| **Full feature set (incl. `icd10_pcs`)** | **0.956** | **0.818** |
| **`icd10_pcs` alone** (only feature) | 0.923 | **0.728** |

`icd10_pcs` (the procedure code) alone very nearly reproduces the full model's performance. Digging further: of 2,253 unique procedure codes, **162 out of 486 codes with ≥20 occurrences (33%) have a perfectly deterministic ICU rate — exactly 0% or exactly 100%**. Example: code `00B00` (n=1,659) has a 95.8% ICU rate; several others sit at exactly 0%.

**Verdict: this is not classical data leakage (no post-outcome information reaches the model), but a task-definition mismatch worth stating plainly.** `icd10_pcs` is known at surgical planning time, not assigned retroactively from what happened during/after surgery — so nothing here violates the "features available before or at planning" design constraint the original notebook set out. But because `icuin_time` (per §6.2) only captures the *immediate, routine, protocol-driven* OR→ICU transfer — not a downstream complication — a large share of `postop_icu` is close to a **deterministic function of procedure type** (certain procedures are *always* or *never* followed by immediate ICU transfer as a matter of institutional care pathway, not patient-specific risk). That is a fundamentally easier statistical target than "will this patient die within 30 days" or "will this patient unexpectedly need ICU after an uneventful-seeming operation" — which is exactly why this report's AUPRC is not comparable to the mortality-prediction literature's AUPRC, and only partially comparable to the ekja.org ICU-admission paper (whose narrower, rarer cohort likely excludes more of the "deterministic by procedure type" cases this report's broader definition retains).

### 6.4 What this means for the report's conclusions

- The **relative comparisons in this report are still valid**: KANBoost vs. baseline, KANBoost vs. trees, and the effect of each individual fix were all measured on the *same* label and *same* splits throughout, so the gap-closing conclusions (§3–§5) hold regardless of how "easy" the underlying task turns out to be.
- The **absolute numbers should not be quoted as beating the published literature** — the tasks are not equivalent. A fair one-line summary: *on this dataset's broader "immediate postoperative ICU transfer" definition (which is partly a deterministic function of procedure type), KANBoost with the fixes above reaches AUROC/AUPRC competitive with strong tree ensembles; on the published paper's narrower, rarer "ICU admission" definition, both this project's models and the published XGBoost baseline would be expected to score substantially lower — a hypothesis current CPB-exclusion experiments (see the repository's benchmark scripts) are testing directly.*
- Recommended before any external claim of matching or beating a published benchmark: patient-grouped split (already done here), a temporal (not random) split if the goal is deployment realism, and reporting the full metric suite (AUROC/AUPRC/F1/Brier/calibration, already done here) *plus* a feature-ablation/leakage check like §6.3 as standard practice — not an afterthought.

## 7. Beyond deterministic procedure codes: mixed-code robustness analysis

§6.3 showed that many procedure codes are perfectly deterministic w.r.t. `postop_icu` (institutional protocol, not patient-specific risk). That alone doesn't tell us whether the model's overall accuracy is *just* a lookup over those deterministic codes, or whether it retains genuine discriminative signal once that shortcut is unavailable. This section tests that directly, following the same methodology as §6.3 (LightGBM, same group-aware split, Large scale).

### 7.1 Performance by procedure-code category

Every test-row's procedure code is tagged by its **train-set-only** ICU rate (no leakage from test), into four tiers:

| Code tier | n (test) | Positive rate | AUROC | AUPRC |
|---|---:|---:|---:|---:|
| Deterministic-negative (≤2% ICU rate in train) | 13,101 | 0.60% | 0.851 | 0.074 |
| Deterministic-positive (≥98% ICU rate in train) | 420 | 98.57% | 0.535 | 0.984 |
| **Mixed-risk (2–98% ICU rate in train)** | **10,860** | **20.92%** | **0.9155** | **0.7993** |
| Rare (<20 occurrences in train) | 1,886 | 12.94% | 0.909 | 0.662 |

The deterministic tiers behave exactly as expected for a near-constant target (AUROC near chance in the ≥98% tier, since there's almost nothing left to rank). The finding that matters: **within the mixed-risk tier — the codes that carry no deterministic shortcut at all — the model still reaches AUROC 0.9155 and AUPRC 0.7993**, both close to the full-dataset numbers (0.956 / 0.818).

### 7.2 Full-data ablation without `icd10_pcs`

Repeated from §6.3 for direct comparison: with `icd10_pcs` removed entirely, AUROC = **0.9209**, AUPRC = **0.6295** (vs. 0.956 / 0.818 with it). A large, genuine signal survives even with no procedure-code information at all — coming from `department`, `asa`, `emop`, and demographics alone.

### 7.3 Mixed-codes-only retraining (the decisive test)

The strongest test: **remove every deterministic code from both training and test**, retrain from scratch on only the mixed-risk subset (train n=32,322, test n=10,860, test positive rate 20.92%), then check whether `icd10_pcs` still helps *within* this subset:

| Configuration | AUROC | AUPRC |
|---|---:|---:|
| Mixed-codes-only, **with** `icd10_pcs` | **0.9163** | **0.8012** |
| Mixed-codes-only, **without** `icd10_pcs` | 0.8541 | 0.5908 |

Two things follow from this table. First, restricting to mixed-risk codes barely moves performance relative to evaluating the full model on that same subset (§7.1: 0.9155/0.7993) — the model isn't relying on deterministic codes leaking into training in some indirect way. Second, `icd10_pcs` still adds a real, substantial improvement (AUROC +0.062, AUPRC +0.210) *even after every deterministic code has been removed* — so the procedure code is not merely a protocol-lookup key; at the fidelity level of "which specific procedure, among procedures with genuinely uncertain ICU outcomes," it carries real, additional clinical-risk signal beyond department/demographics.

### 7.4 Conclusion

Not "there is no issue here" — the more precise, defensible statement is:

> The original results should be interpreted as **protocol-aware ICU pathway prediction** (§6.4), since a meaningful share of the label is institutionally near-deterministic given procedure type. However, the mixed-code robustness analysis shows the model does **not** depend solely on deterministic protocol lookups: restricted entirely to procedures with genuinely uncertain (2–98%) historical ICU rates, it still achieves AUROC 0.9163 / AUPRC 0.8012 with `icd10_pcs`, dropping to 0.8541 / 0.5908 without it — evidence of real clinical-risk signal captured within the non-deterministic procedure space, not just a lookup table over institutional routing rules.

Combined with the group-aware split (no patient leakage across folds), the ablation study (§6.3/§7.2), calibration and Brier-score reporting (§3), and the CPB-exclusion sensitivity check (§6.2), this gives a reasonably rigorous verification bundle before any external claim about this dataset: **split integrity + feature ablation + mixed-code-only retraining + calibration + a documented sensitivity analysis on the label definition itself.**

**Caveat, stated plainly**: §6.3 and §7 use LightGBM, not KANBoost, as the diagnostic tool — chosen deliberately because it fits in seconds, making many ablation variants (stratified tiers, mixed-only retrain, with/without `icd10_pcs`) practical to run at all on CPU-only hardware. Throughout this report KANBoost and the tree ensembles track each other closely on every metric measured on the same data (§1–§5), which is suggestive but not proof that KANBoost's *specific* fitted function relies on the same non-deterministic procedure-code signal in the same way. Repeating §7.1/§7.3 with KANBoost itself (not just LightGBM) is flagged as the natural next step for full rigor, not yet done as of this writing.

## 8. What was NOT done, and why

- **No GPU run**: this machine's torch build has no CUDA support (`torch.version.cuda is None`), despite an NVIDIA GPU + driver being present. All numbers above are CPU-only; installing a CUDA-enabled torch build was out of scope for this exercise but would likely help the trees marginally and KANBoost more (per kanboost's own device-selection code path).
- **No true GrowNet joint-corrective boosting**: a faithful implementation (joint backprop across multiple weak learners against a fully corrective objective) was judged too large a change to build safely, correctly, and quickly against kanboost's closed-form ALS solver in this session. The post-hoc consolidation step adopted here is an explicitly-labeled, safer stand-in that captures the "reduce redundant small corrections" idea without touching the training loop's math.
- **This work has not been through kanboost's own review pipeline** (see `AI_REVIEW_LOOP.md`: ChatGPT hypothesis → Claude Code implementation → Codex independent review → ChatGPT scientific judgment → user merge approval). All code here lives in a personal scratchpad directory outside the `kanboost/` package, uses monkeypatching and private (`_`-prefixed) methods not covered by the test suite, and has not been added to `kanboost/tests/`.

## 9. Reproducing this report

Scripts (scratchpad, not part of the `kanboost` package):
- `inspire_three_scales.py` — baseline (one-hot, no fixes), all 3 scales vs. trees.
- `inspire_kanboost_improved.py` — 5 fixes (encoding + balance + capacity), later extended with `find_threshold`/`calibrate`.
- `inspire_kanboost_hierarchy.py` — 5 fixes + `categorical_hierarchy`.
- `inspire_kanboost_v2.py` — RBF-basis toggle + post-hoc consolidation (used to isolate each technique).
- `inspire_kanboost_ultimate.py` — final combined configuration (Large only).
- `inspire_kanboost_ultimate_v2.py` — final combined configuration with an `--exclude-cpb`-style toggle (§6.2), run across all three scales for both label definitions.
- `leakage_audit.py` — the feature-ablation/leakage check in §6.3.

Raw result CSVs are alongside these scripts in the same scratchpad directory. The reusable, package-side pieces from this work (`consolidate_learners()`, the reproducible `examples/inspire_kanboost_benchmark.py`) are the ones actually shipped in `kanboost` 1.4.0 — see `AI_REVIEW_LOOP.md`, Proposal CC-12.

## 10. Continued past 1.4.0: line search, Newton boosting, RF-KAN, GA2M, and a C++ accelerator (through 1.12.0)

Everything above stopped at `kanboost` 1.4.0. Work continued well past this point in the same session and is tracked in full, proposal-by-proposal, in `AI_REVIEW_LOOP.md` (CC-13 through CC-21) and summarized scientifically in `inspire_kanboost_evaluation.md` (§8-§16); this section is a short chronological pointer, not a duplicate of that detail.

In order: `fit_with_line_search()` (CC-13, per-round step-size search, 3.1-4.2x fewer-round speedup); `fit_with_newton_boosting()` (CC-14, second-order reweighting, closing a gap GB-KAN's paper lists as unsolved); `fit_with_rfkan()` (CC-15, rebuilt random-projection weak-learner engine, 3.7-5.3x faster fit at matching accuracy, at the cost of native interpretability); multiclass support for both (CC-16); `fit_with_ga2m()` (CC-17, GA2M-style main-effect + pairwise-interaction structure, recovering and then exceeding RF-KAN's accuracy while keeping full attribution — the single best-performing engine found in this evaluation); a rejected third-order loss-weighting experiment (CC-18); three accepted second-order-weighting refinements — consistent Armijo line search, adaptive/soft-LM hessian floors (CC-19); parallel (`n_jobs`) prediction across boosting rounds (CC-20, a real but scale-dependent ~2.1x in a repeated-call benchmark, ranging from a slowdown to 1.4x under a realistic cold-start pattern); and finally an optional C++ (pybind11) prediction accelerator (CC-21, a consistent ~2.3-2.7x across all three scales, the most stable speedup found in this whole investigation).

**Cumulative result, this report's original baseline vs. the final `kanboost` 1.12.0 configuration (GA2M + Newton + Armijo line search, C++ backend), Large scale**:

| | Fit time | Predict time | AUROC | AUPRC |
|---|---:|---:|---:|---:|
| §1 baseline (this report, v1.3.0, one-hot, no fixes) | 2295.0s | (slow, one-hot 255-col input) | 0.9189 | 0.6406 |
| §5 ULTIMATE (this report, v1.3.0/1.4.0) | 879.7s | 10.8s | 0.9387 | 0.7580 |
| **Final (v1.12.0, `inspire_kanboost_evaluation.md` §16.1)** | **37.5s** | **9.49s** | **0.9504** | **0.7942** |
| Best tree (XGBoost, unchanged throughout) | ~4.1s | ~0.39s | 0.9561 | 0.8156 |

Fit time dropped a further ~23x beyond this report's own "ULTIMATE" configuration (879.7s → 37.5s), predict time held roughly flat while the *model itself* got both faster-fitting and more accurate (AUROC +0.012, AUPRC +0.036 over ULTIMATE) — the remaining gap to the best tree narrowed to AUROC 0.0057 and AUPRC 0.0214, the closest this evaluation ever got. See `inspire_kanboost_evaluation.md` §16.1 for the full three-scale table and the tree-comparison caveats (threshold tuning, calibration) that still apply.
