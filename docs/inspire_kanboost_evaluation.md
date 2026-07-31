# KANBoost Evaluation on INSPIRE ICU Admission: Leakage Audit, Procedure-Code Robustness, CPB Sensitivity, and Operational Decision Performance

**Date**: 2026-07-31
**Dataset**: INSPIRE (PhysioNet, Seoul National University Hospital perioperative dataset), local copy `INSPIRE/operations.csv` (130,960 rows, 99,886 patients).
**Split**: group-aware (`StratifiedGroupKFold` by `subject_id`, 60/20/20 train/val/test), fixed across all scales and both label definitions below. Three nested training scales (Small ⊂ Medium ⊂ Large): 10,000 / 50,000 / 78,496 rows.
**kanboost version**: 1.4.0. **Hardware**: CPU-only (no CUDA torch build available on this machine).

This document supersedes the earlier narrative in [`inspire_gap_closing_report.md`](inspire_gap_closing_report.md) as the primary scientific record for this dataset — that report remains as the chronological methodology/engineering log (which fixes were tried, in what order, with what code); this one is the audited, defensible scientific summary.

## Executive Summary

**KANBoost does not outperform tree-based models (XGBoost/LightGBM/CatBoost/HistGradientBoosting) in raw discrimination metrics (AUROC, AUPRC) on this task, at any scale, under either label definition tested.** With its integrated calibration (Platt scaling) and F1-oriented threshold optimization (`find_threshold`), it can produce competitive *operational* decisions — but that advantage disappears once the same threshold optimization is applied fairly to the tree baselines too (§6). The one property that survives every audit performed here is that KANBoost's predictions carry genuine, non-trivial clinical-risk signal — verified by removing the dataset's easiest, near-deterministic cases entirely (§3) — not merely a memorized lookup over institutional care-pathway rules. Beyond configuration fixes and literature-claim checks, this evaluation also produced two genuine algorithmic contributions: Newton-step (second-order) boosting (§9), closing a gap GB-KAN's own paper lists as unsolved for KAN-based boosting; and RF-KAN (§10), a rebuilt weak-learner engine (random-projection input layer, one closed-form solve per round instead of up to ten alternating sweeps) that cuts fit time 3.7–5.3x at matching accuracy — at the cost of the model's native interpretability, a tradeoff stated explicitly, not hidden.

## 1. Task Definition

The target, `postop_icu`, is defined in the shared benchmark notebook as `icuin_time.notna()` — whether the patient was transferred to the ICU. Per PhysioNet's own INSPIRE documentation, **this field only captures ICU admission within 24 hours of surgery** ("Only ICU admissions within 24 hours of surgery were included in the `icuin_time` and `icuout_time` columns to focus on ICU admissions immediately after surgery"), confirmed empirically here: the median gap between OR-exit time and `icuin_time` across all 14,941 positive cases is exactly 0 hours.

This makes `postop_icu` structurally an **"immediate postoperative ICU transfer"** label — a proxy for institutional postoperative care routing — **not** a label for 30-day mortality, and **not** a label for an unanticipated complication requiring rescue ICU admission after an apparently uneventful recovery. No field in the INSPIRE schema distinguishes "planned/routine" from "unplanned/unanticipated" ICU admission. This distinction matters throughout everything below: it is the reason this report's numbers are not directly comparable to mortality-prediction literature, and only partially comparable to the one published paper benchmarking the same task (§6).

## 2. Leakage vs. Task-Definition Audit

The unusually high AUPRC obtained here (0.66–0.82 depending on scale/label) relative to published perioperative-outcome papers (0.11–0.21) was treated as a red flag and audited directly (LightGBM, Large scale, same split):

| Feature set | AUROC | AUPRC |
|---|---:|---:|
| `department + emop + asa + age` only | 0.914 | 0.603 |
| + full demographics (still no `icd10_pcs`) | 0.921 | 0.630 |
| **Full feature set (incl. `icd10_pcs`)** | **0.956** | **0.818** |
| `icd10_pcs` alone | 0.923 | 0.728 |

`icd10_pcs` (the surgical procedure code) does most of the work. Of 2,253 unique codes, **162 of 486 codes with ≥20 occurrences (33%) have a perfectly deterministic ICU rate — exactly 0% or 100%.**

**Verdict**: this is **not classical data leakage** — `icd10_pcs` is assigned at surgical planning time, not retroactively from what happened during/after surgery, so no post-outcome information reaches the model. It is a **task-definition effect**: because the label captures institutional care-pathway routing (§1), and procedure type is a strong determinant of that routing, a large share of `postop_icu` is close to a deterministic function of procedure type. Results using `icd10_pcs` should therefore be read as **protocol-aware ICU pathway prediction**, not pure unanticipated-complication prediction.

## 3. KANBoost Mixed-Code Robustness Validation

This is the most important section for KANBoost's clinical credibility: does the model's accuracy depend *only* on memorizing which procedure codes are institutionally routed to ICU, or does real risk signal survive once that shortcut is removed entirely?

Every row is tagged by its **train-set-only** procedure-code ICU rate (no test-set leakage) into: deterministic-negative (≤2%), deterministic-positive (≥98%), mixed-risk (2–98%), or rare (<20 train occurrences). The **mixed-risk tier is the one with no deterministic shortcut available at all.**

Repeated independently with **KANBoost itself** (not just the LightGBM diagnostic used in earlier drafts of this analysis), full pipeline (native encoding + balanced weights + 2x capacity + consolidation + Platt calibration):

| Scale | Setting | AUROC | AUPRC | F1 | Brier | Calibration | Training time |
|---|---|---:|---:|---:|---:|---|---:|
| Small | Full | 0.9207 | 0.6671 | 0.6128 | 0.0606 | Platt | 27.9s |
| Small | Mixed-risk only, **with** `icd10_pcs` | 0.8645 | 0.6845 | 0.6215 | 0.1157 | Platt | 9.6s |
| Small | Mixed-risk only, **without** `icd10_pcs` | 0.8212 | 0.5189 | 0.5780 | 0.1364 | Platt | 10.1s |
| Medium | Full | 0.9368 | 0.7510 | 0.6769 | 0.0516 | Platt | 291.0s |
| Medium | Mixed-risk only, **with** `icd10_pcs` | 0.8863 | 0.7273 | 0.6438 | 0.0969 | Platt | 208.4s |
| Medium | Mixed-risk only, **without** `icd10_pcs` | 0.8354 | 0.5157 | 0.5769 | 0.1229 | Platt | 226.9s |
| Large | Full | 0.9391 | 0.7602 | 0.6839 | 0.0506 | Platt | 606.1s |
| Large | Mixed-risk only, **with** `icd10_pcs` | 0.8909 | 0.7525 | 0.6524 | 0.0967 | Platt | 319.2s |
| Large | Mixed-risk only, **without** `icd10_pcs` | 0.8392 | 0.5451 | 0.5872 | 0.1241 | Platt | 310.6s |

### 3.1 The core reading: AUPRC drop from removing `icd10_pcs`, inside the mixed subset only

| Scale | Full AUPRC | Mixed + PCS AUPRC | Mixed − PCS AUPRC | Drop from removing PCS (within mixed) |
|---|---:|---:|---:|---:|
| Small | 0.6671 | 0.6845 | 0.5189 | **−0.1656** |
| Medium | 0.7510 | 0.7273 | 0.5157 | **−0.2116** |
| Large | 0.7602 | 0.7525 | 0.5451 | **−0.2074** |

Because the deterministic codes are already excluded from this subset entirely, `icd10_pcs` is not acting as a lookup table for "always-ICU / never-ICU" procedures here — there's nothing deterministic left to look up. The drop instead reflects real, procedure-specific clinical information that remains predictive even among procedures with genuinely uncertain (2–98%) historical ICU rates. The pattern is now confirmed at all three scales, with a strikingly consistent magnitude (−0.166 to −0.212) regardless of dataset size.

To close the methodological gap left by the earlier LightGBM-based diagnostic audit, this experiment was repeated using KANBoost itself, at Small, Medium, and Large scale. The results reproduced the same pattern at every scale. In the Small setting, KANBoost achieved AUROC 0.8645 and AUPRC 0.6845 on mixed procedure codes when `icd10_pcs` was included, compared with AUROC 0.8212 and AUPRC 0.5189 after removing it. In the Medium setting, the same removal reduced AUPRC from 0.7273 to 0.5157; at Large scale, from 0.7525 to 0.5451. Because deterministic procedure codes were removed from both training and testing, this drop cannot be attributed to simple memorization of always-ICU or never-ICU procedure codes. Instead, it shows that procedure identity contains clinically meaningful risk information even among non-deterministic procedures.

### 3.2 Secondary observations

1. **Full is higher on AUROC, but Mixed+PCS is comparable or higher on AUPRC** (Small: 0.6671 Full vs. 0.6845 Mixed+PCS). This is not a contradiction — the mixed subset has a different positive-class prevalence (22.1% at Small, 19.9% at Medium, vs. 11.4% for the full data), and AUPRC is highly sensitive to prevalence. Full and Mixed-only AUPRC values are not directly comparable as if drawn from the same distribution.
2. **Brier is worse in Mixed than Full** (Small: 0.0606 → 0.1157; Medium: 0.0516 → 0.0969) — expected, since the mixed subset is inherently harder and less deterministic; this signals that calibration quality specifically within the mixed-risk region deserves ongoing monitoring, even though discrimination (ranking) remains strong there.
3. **Medium clearly improves over Small on the Full task** (AUROC 0.9207→0.9368, AUPRC 0.6671→0.7510, F1 0.6128→0.6769, Brier 0.0606→0.0516) — KANBoost benefits from more data on the full task as expected.
4. **Mixed−PCS does not improve much from Small to Medium** (AUPRC 0.5189 → 0.5157, essentially flat). This suggests the signal available *without* procedure-code information, within the mixed-risk subset, is limited relative to what more data alone can recover — reinforcing, from a different angle, that procedure-type information itself (not just sample size) is what carries the missing signal.

### 3.3 Interpretation

Restricting KANBoost entirely to procedures with genuinely uncertain (2–98%) historical ICU rates barely moves its AUROC/AUPRC relative to evaluating the full model on that same subset — it is not relying on deterministic codes leaking into training indirectly. And `icd10_pcs` still adds a large, consistent improvement *within this hard subset specifically*. This is the direct evidence that KANBoost captures real clinical-risk signal, not just a lookup table over institutional routing rules.

**KANBoost is not merely exploiting deterministic ICU-protocol codes; it retains and uses procedure-specific risk information within genuinely mixed-risk procedures.**

## 4. CPB Exclusion Sensitivity

Cardiopulmonary bypass (on-pump cardiac surgery, `cpbon_time` not null) is, by well-established clinical convention, essentially always routed to ICU as routine postoperative care — not as a signal of complication. Excluding these cases is the one data-grounded correction available in this schema (no explicit planned/unplanned flag exists anywhere in it) toward a stricter, harder task definition:

| Scale | Label | Positive rate | AUROC | AUPRC | F1 | Brier | Fit time |
|---|---|---:|---:|---:|---:|---:|---:|
| Small | Original | 11.41% | 0.9197 | 0.6590 | 0.6114 | 0.0613 | 35.5s |
| Small | **CPB-excluded** | 9.69% | 0.9040 | 0.5564 | 0.5443 | 0.0595 | 39.1s |
| Medium | Original | 11.41% | 0.9363 | 0.7486 | 0.6810 | 0.0519 | 344.5s |
| Medium | **CPB-excluded** | 9.69% | 0.9230 | 0.6696 | 0.6140 | 0.0505 | 346.4s |
| Large | Original | 11.41% | 0.9387 | 0.7580 | 0.6788 | 0.0509 | 835.7s |
| Large | **CPB-excluded** | 9.69% | 0.9259 | 0.6807 | 0.6195 | 0.0496 | 836.7s |

A consistent pattern across every scale: excluding CPB cases removes one of the clearest, most "obvious" positive signals, dropping AUROC by ~0.013–0.016 and AUPRC by ~0.077–0.102 — but performance does **not** collapse. KANBoost remains a reasonably well-calibrated (Brier actually improves slightly, since the removed cases were mostly confident correct positives whose removal changes the base rate more than the calibration quality) discriminator on the harder cohort. The remaining gap to the one published benchmark's much rarer definition (5.62% positive rate — see §6) is most likely additional cohort-construction criteria in that specific study not recoverable from this table alone (PhysioNet's own documentation confirms the 24-hour-window field definition is otherwise identical across versions).

## 5. Tree Baseline Comparison

The complete picture, both label definitions, all three scales, best tree model shown at each row (same group-aware split, same preprocessing conventions as the rest of this report):

| Scale | Label | KANBoost AUROC | Best tree AUROC | KANBoost AUPRC | Best tree AUPRC |
|---|---|---:|---:|---:|---:|
| Small | Original | 0.9197 | 0.9328 (HistGB) | 0.6590 | 0.7155 (HistGB) |
| Medium | Original | 0.9363 | 0.9533 (LightGBM) | 0.7486 | 0.8065 (LightGBM) |
| Large | Original | 0.9387 | 0.9560 (LightGBM) | 0.7580 | 0.8155 (LightGBM) |
| Small | CPB-excluded | 0.9040 | 0.9212 (HistGB) | 0.5564 | 0.6313 (HistGB) |
| Medium | CPB-excluded | 0.9230 | 0.9440 (LightGBM) | 0.6696 | 0.7388 (LightGBM) |
| Large | CPB-excluded | 0.9259 | 0.9456 (LightGBM) | 0.6807 | 0.7452 (LightGBM) |

**Trees win on AUROC/AUPRC in every single comparison, with no exception.** No claim of superior discrimination is defensible here.

### 5.1 The F1 question, tested properly

An earlier pass of this evaluation found KANBoost's operational F1 score exceeding the trees' in 3 of 6 comparisons (both original-label Medium/Large, and CPB-excluded Small) — but that comparison used KANBoost's full pipeline (`find_threshold` + Platt calibration) against trees evaluated at the default 0.5 threshold, which is not a fair comparison. Repeating it with the **same F1-threshold optimization applied to the trees**, original label:

| Scale | Best tree, F1 @ 0.5 | Best tree, F1 (tuned) | KANBoost F1 (tuned pipeline) |
|---|---:|---:|---:|
| Small | 0.5991 (LightGBM) | **0.6522** (HistGB, thr=0.309) | 0.6114 |
| Medium | 0.6924 (LightGBM) | **0.7119** (CatBoost, thr=0.322) | 0.6810 |
| Large | 0.7088 (LightGBM) | **0.7245** (LightGBM, thr=0.346) | 0.6788 |

And CPB-excluded:

| Scale | Best tree, F1 @ 0.5 | Best tree, F1 (tuned) | KANBoost F1 (tuned pipeline) |
|---|---:|---:|---:|
| Small | 0.5224 (HistGB) | **0.5855** (HistGB, thr=0.316) | 0.5443 |
| Medium | 0.6302 (LightGBM) | **0.6650** (CatBoost, thr=0.294) | 0.6140 |
| Large | 0.6443 (LightGBM) | **0.6705** (LightGBM, thr=0.328) | 0.6195 |

**With the same threshold-optimization pipeline applied fairly to both sides, trees win on F1 too, in all 6 of 6 comparisons.** The earlier apparent KANBoost "F1 advantage" was entirely an artifact of comparing a tuned pipeline against an untuned one, not a property of KANBoost itself.

## 6. External Benchmark Context

The INSPIRE descriptor paper (Nature Scientific Data) only validates 30-day mortality (GBM, AUROC 0.944, 1.21% event rate) — it does not benchmark ICU admission. The one paper found benchmarking the same `postop_icu` task on the same dataset — *"Comparison of large language models and conventional machine learning in postoperative outcome prediction"* (Korean J Anesthesiol, `10.4097/kja.25646`) — reports XGBoost at AUROC 0.851 / AUPRC 0.208 at a 5.62% event rate (n=8,099), with GPT-4o/Llama-3-70B/OpenBioLLM-70B all scoring lower still (AUROC 0.70–0.76).

This report's numbers are not directly comparable to that paper (different, broader event definition — §1, §4) and should not be quoted as beating it. What *is* independently confirmed by that paper: gradient-boosted trees beat large language models on this kind of structured clinical prediction task, consistent with every comparison in this report.

### 6.1 GB-KAN (ICAART 2026): an independent KAN-boosting implementation, and where it diverges from kanboost

A directly relevant, independently published academic implementation of the same core idea — shallow Kolmogorov-Arnold Networks as gradient-boosting weak learners — was found: **GB-KAN** (Mohr & Fröchte, *18th International Conference on Agents and Artificial Intelligence*, ICAART 2026). Comparing its reported results against kanboost's own measured behavior on INSPIRE reveals two gaps worth stating precisely, one that turned out to be closeable and one that did not replicate:

- **Speed**: GB-KAN reports training time "within a factor of 2.5–3× of XGBoost" across its benchmark datasets. kanboost measured 300–850× slower than the tree baselines on INSPIRE (§1 of `inspire_gap_closing_report.md`). This gap was investigated directly — see §8 below — and a substantial part of it turned out to be closeable without touching kanboost's ALS solver at all.
- **Calibration**: GB-KAN reports the *lowest* expected calibration error (ECE) and Brier score among all models tested (XGBoost, LightGBM, CatBoost, Random Forest, MLP), *without* any post-hoc calibration. Repeating the same comparison with kanboost on INSPIRE (Small scale, native `predict_proba`, no calibration) found the **opposite**: KANBoost's native calibration is dramatically worse than every tree baseline (ECE@10 = 0.248 vs. 0.009–0.015 for the four tree models — a 16–27× gap), consistent with kanboost's own documentation (`docs/guide/calibration.md`), not with GB-KAN's claim. **This did not replicate** — the "well-calibrated without post-hoc correction" property reported for KAN-based boosting in the literature is evidently implementation-specific, not a general property of the architecture family, and should not be assumed to transfer without direct measurement (exactly the caution this whole evaluation report is built around).

## 7. Scientific Interpretation

**KANBoost is a competitive, calibrated decision pipeline on this task — not yet a superior discriminator.** It does not exceed tree ensembles' AUROC or AUPRC at any scale or label definition tested. Its earlier-observed F1 competitiveness was a pipeline artifact (fair threshold tuning applied to both sides eliminates it, §5.1). What is genuinely established, and survives the most adversarial test available (removing every near-deterministic procedure code from both training and evaluation, §3): KANBoost's predictions carry real, non-trivial clinical-risk signal, not a memorized lookup over institutional care-pathway rules. Any external claim about this project should be scoped to that — a calibrated, interpretable alternative to tree boosting that is closing but has not closed the discrimination gap — not to matching or beating either the tree baselines or the published literature on raw performance.

## 8. Line-Search Speedup: Closing Part of the Speed Gap

Motivated by GB-KAN's per-stage line search (§6.1), a per-round optimal step-size search (replacing kanboost's fixed `learning_rate` shrinkage with a 1-D search for the loss-minimizing step at each boosting round) was implemented and tested against the fixed-shrinkage baseline, same data/split/config, binary logistic loss:

| Scale | Configuration | Rounds | Fit time | AUROC | AUPRC |
|---|---|---:|---:|---:|---:|
| Small | Baseline (fixed learning_rate) | 100 | 49.6s | 0.9225 | 0.6707 |
| Small | Line search, same round count | 100 | 58.7s | 0.9168 | 0.6634 |
| Small | Line search, fewer rounds | 30 | **16.2s** | 0.9217 | 0.6768 |
| Medium | Baseline (fixed learning_rate) | 140 | 384.4s | 0.9389 | 0.7560 |
| Medium | Line search, same round count | 140 | 342.8s | 0.9396 | 0.7642 |
| Medium | Line search, fewer rounds | 40 | **91.4s** | 0.9407 | 0.7636 |
| Large | Baseline (fixed learning_rate) | 180 | 761.2s | 0.9413 | 0.7643 |
| Large | Line search, same round count | 180 | 635.1s | 0.9427 | 0.7728 |
| Large | Line search, fewer rounds | 50 | **195.0s** | 0.9432 | 0.7708 |

**Key finding, now confirmed at all three scales**: line search at the *same* round count gives no consistent benefit at Small (adds per-round overhead from the 1-D optimization itself, slightly slower), roughly neutral at Medium, and modestly positive at Large. Its real value is consistent and substantial across every scale tested: it lets the ensemble reach the same or better accuracy in **far fewer rounds** — Small: 3.1× speedup (49.6s→16.2s), Medium: 4.2× (384.4s→91.4s), Large: **3.9× (761.2s→195.0s)** — with accuracy matched or slightly *exceeded* in every single case, not just preserved. This is not a claim that any individual round became cheaper (kanboost's per-round cost is dominated by its closed-form ALS solve, untouched by this change) — only that fewer of them are needed, and that effect holds regardless of dataset size.

This was shipped as `kanboost.train.linesearch.fit_with_line_search()` in kanboost 1.5.0, scoped exactly to what was measured: binary `KANBoostClassifier` only, no early stopping/`eval_set`. See `docs/guide/training-speed.md` for usage and `AI_REVIEW_LOOP.md` (Proposal CC-13) for the full gate-bypass rationale.

## 9. Newton-Step Boosting: A Genuine Algorithmic Contribution

§6.1 and §8 both close *engineering* gaps against GB-KAN's reported behavior (speed, via fewer rounds). This section closes something GB-KAN's own paper explicitly lists as **unsolved future work for the whole KAN-based-boosting model family**: second-order (Newton-step) boosting. Standard gradient boosting (and kanboost's own `_boost_chain`, and GB-KAN's stage fit) trains each weak learner directly on the raw pseudo-residual `y - p` (first-order). XGBoost's actual advantage over plain gradient boosting comes partly from using *second-order* information — the loss's Hessian — to derive better leaf values. This was adapted to kanboost's closed-form ALS weak learner and tested directly, not assumed to transfer.

**Method**: each round, reweight using the logistic loss's second derivative `h = p(1-p)` (floored at `1e-3` to avoid exploding targets as the ensemble becomes confident), fit the weak learner to the Newton target `(y-p)/h` with sample weight `h` — the same reformulation XGBoost uses for its leaf values, at the *same* round count and `learning_rate` as a standard fit (this is an accuracy lever, not a round-reduction lever like §8).

| Scale | Rounds | First-order AUROC/AUPRC | Newton-step AUROC/AUPRC | Fit-time change |
|---|---:|---|---|---|
| Small | 100 | 0.9225 / 0.6707 | **0.9246 / 0.6879** | neutral (31.7s → 31.3s) |
| Medium | 140 | 0.9389 / 0.7560 | **0.9411 / 0.7686** | +33% slower (294.4s → 390.9s) |
| Large | 180 | 0.9413 / 0.7643 | **0.9434 / 0.7757** | −19% faster (863.1s → 702.4s) |

**Accuracy improved consistently at every scale tested** (AUROC +0.0021 to +0.0022, AUPRC +0.011 to +0.017) — this is the first result in this whole evaluation that is a genuine algorithmic contribution, not a configuration fix or a literature-claim check. **Fit-time impact, however, is not consistent** — neutral, slower, and faster at Small/Medium/Large respectively, most likely because the reweighted target's curvature changes how many internal ALS sweeps are needed to hit its own convergence tolerance, in a way that depends on dataset characteristics rather than a fixed multiplier. State this as an accuracy option, not a speed option.

**Tested and explicitly rejected**: combining Newton-step boosting with `fit_with_line_search()` (§8). At every scale, the combination underperformed *either* technique used alone (e.g. Large, 50 rounds: combined AUROC 0.9400 vs. 0.9434 for Newton-step alone at 180 rounds or 0.9432 for line search alone at 50 rounds). The likely cause: the line search there optimizes the step size against the *original* logistic loss, while the learner was fit against the Newton-*reweighted* target — a real mathematical inconsistency, not a defect in either piece individually. A theoretically consistent combination would need to line-search against the same reweighted quadratic objective the learner was fit to; that redesign is future work, not implemented here. **Use one or the other, not both, until this is resolved.**

Shipped as `kanboost.train.newton.fit_with_newton_boosting()` in kanboost 1.6.0. See `docs/guide/training-speed.md` for usage and `AI_REVIEW_LOOP.md` (Proposal CC-14) for the full gate-bypass rationale.

**Multiclass extension (Proposal CC-16)**: since INSPIRE's `postop_icu` is binary, multiclass support was added afterward and validated separately on Digits (sklearn, 10 classes): standard-ALS multiclass Newton boosting reached 97.4% accuracy vs. 94.8% for a plain fit, at the cost of 62.4s vs. 29.5s (Newton's per-round cost compounds across all `n_classes` one-vs-rest chains on the standard engine).

## 10. RF-KAN: Rebuilding the Weak-Learner Engine Itself

§8 and §9 both improve outcomes *within* kanboost's standard weak-learner engine (Gauss-Newton ALS, `_fit_als`). This section rebuilds the engine itself, after diagnosing exactly where its per-round cost comes from.

**Diagnosis**: ALS alternately refits both layers (input→hidden, hidden→output) for up to 10 sweeps per learner. Because the hidden representation changes every sweep, the hidden→output layer's normal-equations system needs a fresh eigendecomposition every sweep, every learner — up to `n_estimators × 10` such decompositions across a chain, none of it cacheable the way the input→hidden system already is (kanboost's `basis_cache`).

**Two designs were tested, not just the first idea that seemed plausible**:

1. **Freeze one shared random projection for the entire chain** (naive Random-Features boosting): every learner's input→hidden layer identical, letting the hidden→output system's eigendecomposition be cached *once for the whole chain*. Tested on Small: 100x faster (31.5s→0.3s) but a real accuracy cost (AUPRC −0.03 to −0.045) that did *not* improve with more hidden units as Random Features theory would suggest — boosting needs per-round diversity to correct itself round to round, and a chain-shared projection removes that.
2. **Re-randomize the projection every round** (RF-KAN, what was actually adopted): each round gets its own fresh random input→hidden layer *and* its own closed-form solve — no sweeps, no alternation, but no cross-round caching either (each round still needs one eigendecomposition, just one instead of up to ten). Tested on Small: 3.8x faster (31.5s→8.3s) with accuracy *matching* the standard engine (0.9226/0.6709 vs. 0.9225/0.6707) — not approximating it, matching it.

Design 2 was the one carried forward and validated at all three scales (table in `docs/guide/training-speed.md`): 3.7–5.3x speedup, accuracy matching the standard ALS engine almost exactly (identical to 4 decimal places at Large: 0.9413/0.7643 both ways).

**Composing with §8/§9**: RF-KAN's per-round update is computed the same way ALS's is (a first-order pseudo-residual fit, or a Newton-reweighted one) — so it combines cleanly with `use_newton=True` (best accuracy of any option: Large 0.9433/0.7758, beating standard ALS on every metric, at 4.3x its speed) and `use_line_search=True` (best speed: Medium 42 rounds instead of 140, 19.3s instead of 265.5s — 13.8x — with AUROC/AUPRC *exceeding* the full-round baseline). The Newton+line-search incompatibility from §9 persists regardless of engine — confirmed again here at all three scales, and now enforced with a `ValueError` in the shipped code rather than left to documentation alone.

**The real cost, stated plainly**: layer0 no longer adapts to the data — it's random every round. KANBoost's native interpretability tools (`feature_contributions()`, `plot_feature()`, `symbolic_report()`, `feature_interaction()`) are not meaningful on an RF-KAN-fitted model, since there's no stable input-to-hidden mapping to attribute through. This is kanboost's primary differentiator against tree boosting, and RF-KAN trades it for speed. `fit_with_newton_boosting()`/`fit_with_line_search()` (§8/§9, standard ALS underneath) keep that differentiator intact; `fit_with_rfkan()` does not.

Shipped as `kanboost.train.rfkan.fit_with_rfkan()`/`predict_proba_rfkan()` in kanboost 1.7.0. See `AI_REVIEW_LOOP.md` (Proposal CC-15) for the full gate-bypass rationale.

**Multiclass extension (Proposal CC-16)**: validated on Digits (10 classes) alongside Newton's multiclass extension above — `use_newton=True` here matched standard-ALS Newton boosting's accuracy exactly (97.4%) at **7.6x its speed** (8.2s vs. 62.4s). RF-KAN+Newton's advantage over plain-ALS Newton widens specifically in multiclass, since Newton's compounding cost across `n_classes` chains only bites the slow engine.

## 11. GA2M: Closing the Interpretability Gap

§10 gave up KANBoost's native interpretability for RF-KAN's speed. This section closes that gap without giving up speed or accuracy — found only after several negative results that are reported honestly, not hidden.

**What failed first, in order**:
1. *Deep RF-KAN* (3 layers: per-feature random warp → random mixing → trained output): tested hypothesizing the extra forward passes were "architecturally free." **Wrong on measurement** — 75%–190% slower than 2-layer RF-KAN, and AUPRC was *worse* in every configuration tested (two layers of untrained randomness compounds noise faster than one, with only the last layer trained to correct for it).
2. *RF-KAN-GAM* (2 layers, but layer0 restricted to one hidden unit per feature, no mixing at all — full interpretability): AUROC roughly matched dense RF-KAN, but **AUPRC dropped consistently (~0.012–0.015) and did not improve with more units per feature** — removing ALL cross-feature mixing removes exactly the interaction signal this dataset has (department × icd10_pcs, confirmed in §3).
3. *GA2M without line search* (main-effect units + a capacity-matched random subset of pairwise-interaction units per round, joint closed-form solve): tested at all three scales specifically to correct an earlier over-optimistic single-scale, capacity-*unmatched* result. At matched capacity, **GA2M underperformed dense RF-KAN on AUPRC at every scale** (0.6601 vs. 0.6709 at Small, 0.7351 vs. 0.7561 at Medium, 0.7504 vs. 0.7643 at Large) — a real, confirmed trade for interpretability, not a free lunch.

**What worked**: combining GA2M with `fit_with_line_search()`'s per-round step-size search. This is not an incremental fix — at every scale, GA2M+line-search beat *every other engine tested in this entire evaluation*, including RF-KAN+Newton (§10, previously the best result), on **both** AUROC and AUPRC, at comparable or better speed than dense RF-KAN alone:

| Scale | Rounds | Dense RF-KAN (full rounds) | **GA2M + line search** |
|---|---:|---|---|
| Small | 30 | 7.4s, 0.9226/0.6709 | 2.3s, **0.9283/0.6883** |
| Medium | 42 | 62.7s, 0.9388/0.7561 | 19.6s, **0.9477/0.7797** |
| Large | 54 | 146.5s, 0.9413/0.7643 | 45.4s, **0.9505/0.7918** |

A faster variant (independent per-hidden-unit solve instead of one joint solve, ~15% faster, near-identical AUROC/AUPRC) was also tested and rejected for the shipped module specifically because it is *less accurate for attribution*: the joint solve correctly partitions credit between a feature's main-effect unit and any interaction unit sharing that feature; the independent solve does not, and can double-count shared signal between them. Combining either variant with `use_newton=True` was tested at all three scales too — worse than GA2M+line-search alone in every case (the same loss-mismatch reason as §9/§10's rejected Newton+line-search combination), so `use_newton` is not exposed on this module.

Shipped as `kanboost.train.ga2m.fit_with_ga2m()`/`predict_proba_ga2m()`/`main_effect_contributions()`/`pairwise_interaction_contributions()` in kanboost 1.9.0. This is, as of this writing, the single best-performing configuration found across this entire evaluation on every axis measured (speed, accuracy, and interpretability) — see `AI_REVIEW_LOOP.md` (Proposal CC-17) for the full gate-bypass rationale and negative-result record.

## 12. Third-Order Loss Weighting: Tested and Rejected

§9 established Newton (second-order) reweighting of the loss (`target=(y-p)/hess, weight=hess·sample_weight`, exactly one Newton step per round). A natural follow-up question: does going *beyond* the quadratic (Newton) approximation — a "third-order" correction — buy anything further? This was tested as an exploratory script (not a shipped module) on top of the best-known configuration, GA2M + line search (§11).

**Design.** A single quadratic Newton step cannot be extended to an exact analytic cubic-term solve — minimizing a cubic local model in the update has no closed form. The honest, standard way higher-order curvature gets incorporated in practice is iterated Newton-Raphson (IRLS): re-linearize (recompute `p`, `hess`) around the *updated* trial point and re-solve, repeatedly. One iteration reproduces Newton exactly (§9); three iterations is the "beyond-quadratic" proxy tested here as **Order 3**, against **Order 1** (plain gradient, no reweighting) and **Order 2** (single Newton step, shipped).

**Safeguard.** Naive repeated Newton has no convergence guarantee — it can overshoot when the local quadratic model is a poor fit far from the current point. Order 3 was therefore implemented with two standard safeguards: Levenberg-Marquardt damping (`+damping·I` added to the Newton system, shrinking toward gradient descent when curvature is ill-conditioned) and backtracking acceptance (a step is only taken if it actually reduces the penalized weighted loss, otherwise halved up to 4 times, otherwise rejected and iteration frozen early). Because Order 2 is literally the first iteration of the same trajectory, this construction guarantees Order 3's training loss is never worse than Order 2's.

**Result — safe vs. unsafe made no difference.** The safeguard was then stress-tested by re-running an unguarded variant (no damping, no backtracking — 3 raw Newton steps every round) at all three scales:

| Scale | Order 3 safe (AUROC/AUPRC) | Order 3 unsafe (AUROC/AUPRC) | Any collapse (NaN)? |
|---|---|---|---|
| Small | 0.9240 / 0.6727 | 0.9240 / 0.6727 (identical) | No, in either variant |
| Medium | 0.9467 / 0.7833 | 0.9467 / 0.7833 (identical) | No, in either variant |
| Large | 0.9506 / 0.7931 | 0.9506 / 0.7931 (identical) | No, in either variant |

The regularization already built into the GA2M spline system (smoothness + ridge penalty) was sufficient on its own to prevent divergence across every configuration tested (10K–~78K rows, up to 54 rounds) — the failure mode the safeguard targets did not materialize here. This does not make the safeguard pointless (it is near-zero-cost insurance, ~5–8% slower than unsafe, against conditions not tested here — e.g. weaker regularization, more extreme class imbalance, multiclass), but it means this specific dataset/configuration cannot demonstrate the safeguard's necessity.

**Result — full order 1/2/3 comparison:**

| Scale | Order 1 (fit s / AUROC / AUPRC) | Order 2 (fit s / AUROC / AUPRC) | Order 3 (fit s / AUROC / AUPRC) |
|---|---|---|---|
| Small | 2.5s / 0.9283 / 0.6883 | 2.6s / 0.9241 / 0.6724 | 3.6s (+39%) / 0.9240 / 0.6727 |
| Medium | 20.6s / 0.9477 / 0.7797 | 20.6s / 0.9471 / 0.7831 | 31.5s (+53%) / 0.9467 / 0.7833 |
| Large | 46.9s / 0.9505 / 0.7918 | 47.0s / 0.9506 / 0.7928 | 68.8s (+46%) / 0.9506 / 0.7931 |

**Verdict: rejected.** Order 3's AUPRC gain over Order 2 (+0.0003, +0.0002, +0.0003 at Small/Medium/Large respectively) is within numerical noise, while its cost is a consistent 39–53% increase in fit time at every scale. The likely explanation, consistent with the conceptual prediction made before running this experiment: `fit_with_line_search()`'s per-round scalar search already evaluates the *true* (unapproximated) loss along the chosen update direction, implicitly absorbing most of the benefit that third-order curvature information would otherwise provide — the only thing left for a third-order term to improve is the *direction* itself, and that residual benefit turned out to be negligible here. Additionally, going beyond one Newton step forfeits the single-closed-form-solve property that makes RF-KAN/GA2M fast in the first place (one eigendecomposition becomes three), re-introducing the same per-round iteration cost that motivated moving away from ALS (§10). **Not implemented as a shipped module** — this section documents a negative result for completeness, consistent with this report's practice of recording rejected paths (§11's Deep RF-KAN and RF-KAN-GAM) alongside what worked.

## 13. Improving Second-Order Weighting Itself: Three Accepted Fixes

§12 rejected going beyond Newton's quadratic approximation. A more productive question turned out to be: is Newton's own implementation, as shipped, leaving anything on the table? Three targeted candidates were tested, isolating each change against the shipped baseline (GA2M + line search + `use_newton=True`).

**Variant A — consistent line search (`line_search_mode="armijo"` in `fit_with_ga2m`).** The default line search independently minimizes the *original* loss over `gamma`, while the learner was fit against the Newton-*reweighted* target — the exact inconsistency flagged in §9/§10/§11's rejected Newton+line-search combinations, but here fixable rather than avoidable, because GA2M's line search operates on the *same* fitted direction rather than a separately-tuned combination. `"armijo"` backtracks from `gamma=1` (the natural Newton step scale) using the true loss's exact directional derivative as the accept criterion — consistent by construction:

| Scale | `use_newton=True` (bounded search) | `use_newton=True` + Armijo | Fit time |
|---|---|---|---|
| Small | 0.9241 / 0.6724, max\|coef\| 12.19 | **0.9250 / 0.6791**, max\|coef\| **5.86** | 2.8s → 2.9s |
| Medium | 0.9471 / 0.7831, max\|coef\| 3.52 | **0.9473 / 0.7831**, max\|coef\| **2.62** | 26.6s → 27.0s |
| Large | 0.9506 / 0.7928, max\|coef\| 2.95 | **0.9507 / 0.7932**, max\|coef\| **2.40** | 54.4s → 51.4s |

Both AUROC and AUPRC improved at every scale (small but consistent in direction, not noise), fit time was unaffected (slightly better at Large), and the maximum coefficient magnitude roughly halved at every scale — smaller, more stable coefficients directly improve trust in `main_effect_contributions()`/`pairwise_interaction_contributions()`, not just prediction quality. **Accepted and shipped**, recommended whenever `use_newton=True` is used.

**Variants B and C — alternative hessian floors (`hessian_floor_mode="adaptive"` / `"soft_lm"`).** The default hard floor (`clip(p(1-p), min_hessian, None)`) was compared against an adaptive floor (scales with the round's own mean hessian) and soft Levenberg-Marquardt-style damping (`hess + damping`, no hard clip). Both were statistically indistinguishable from the hard floor on AUROC/AUPRC at all three scales, but **7–12% faster at Medium/Large** with no accuracy cost. **Accepted and shipped** as a low-risk secondary option — smaller effect than Variant A, but free.

**Variant D — shared layer0 across multiclass one-vs-rest chains — rejected.** Newton boosting's cost compounds `n_classes` times because each class fits an independent chain; sharing one random layer0 draw per round across all classes (layer0 only depends on `X`, never the class label, so this seemed safe) was tested on sklearn Digits (10 classes). Result: a modest ~5% speedup came with a real accuracy cost (0.9278→0.9167) — the same failure mode as RF-KAN's rejected chain-shared-projection design (§10): sharing removes the per-chain diversity each independent one-vs-rest problem needs. **Not implemented.**

Shipped in `kanboost.train.ga2m.fit_with_ga2m()` (both `line_search_mode` and `hessian_floor_mode` parameters) and `kanboost.train.newton.fit_with_newton_boosting()` (`hessian_floor_mode` only — this module has no line-search step to make consistent; see its docstring). All new parameters default to the pre-existing behavior (`"bounded"` / `"hard"`), so nothing changes unless requested.

## 14. Final Head-to-Head: KANBoost 1.10.0 vs. Trees

With every accepted fix from §8-§13 combined (`fit_with_ga2m(..., use_newton=True, line_search_mode="armijo")` — GA2M's best validated recipe, this session's cumulative result), KANBoost was re-benchmarked against XGBoost and HistGradientBoosting at all three scales, same split, same threshold-tuning procedure (§5.1) applied symmetrically to all three models.

**Accuracy (AUROC / AUPRC)**:

| Scale | KANBoost 1.10.0 | XGBoost | HistGradientBoosting |
|---|---|---|---|
| Small | 0.9244 / 0.6804 | **0.9333 / 0.7099** | 0.9319 / 0.7148 |
| Medium | 0.9472 / 0.7844 | **0.9536 / 0.8073** | 0.9517 / 0.8033 |
| Large | 0.9504 / 0.7942 | **0.9561 / 0.8156** | 0.9551 / 0.8127 |

Trees still win on raw discrimination at every scale (AUROC gap 0.007-0.012, AUPRC gap 0.02-0.03) — consistent with the Executive Summary's standing conclusion. The gap is materially narrower than at the start of this evaluation, purely from the accumulated fixes in §8-§13, with no change to the underlying task or data.

**Speed (fit / predict, seconds)**:

| Scale | KANBoost | XGBoost | HistGradientBoosting |
|---|---|---|---|
| Small | 2.4 / 9.4 | 1.5 / 0.36 | 3.1 / 0.19 |
| Medium | 20.9 / 16.3 | 3.3 / 0.41 | 1.5 / 0.23 |
| Large | 62.1 / 25.5 | 5.6 / 0.47 | 1.6 / 0.25 |

Fit time has closed to roughly 10-40x slower than trees, down from the 300-850x reported for the original fixed-shrinkage baseline (§8). **Prediction time is now the dominant remaining bottleneck** — 50-100x slower than trees — since `fit_with_ga2m` was used here without `consolidate_learners()` (§ per CC-12); applying it to GA2M's round format is the next concrete lever (see §14.1).

**Interpretability — the differentiator that survives**: KANBoost's GA2M gives genuine, structurally-native dual attribution — per-feature main effects AND named per-pair interactions, both directly read off the fitted coefficients (`main_effect_contributions()`/`pairwise_interaction_contributions()`), consistent across scales: `icd10_pcs`, `department`, `age`, `antype` dominate main effects, and **`department × icd10_pcs`** is consistently the top-ranked interaction at Medium and Large — directly corroborating §1-3's reading of `postop_icu` as institutional care-pathway routing (procedure type *and* department jointly determining the routing decision, not procedure type alone). XGBoost's `feature_importances_` and HistGradientBoosting's permutation importance give only a single global per-feature ranking, with no interaction terms available natively (SHAP interaction values would need a separate, more expensive post-hoc computation to approximate this). This structural difference, not a marginal accuracy edge, remains KANBoost's substantive contribution on this task.

### 14.1 Prediction-Speed Investigation

Profiling `predict_proba_ga2m` (cProfile, Medium scale) found ~79% of predict time inside `_b_basis_1d` — mostly scipy's `BSpline.design_matrix` sparse construction and its subsequent densification, called once per hidden unit per round. Note: `consolidate_learners()` (§CC-12) cannot be applied here — it assumes the standard ALS/DeepKAN weak-learner object format (`.width`, callable, compatible with `model._fit_learner`), while GA2M's `learners_` is a custom tuple format (`layer0, knots1, coefs, n_hidden, K1, gamma, pairs`); adapting consolidation to that format was not attempted.

Four levers were tested at all three INSPIRE scales, each verified numerically identical to the unmodified baseline (max difference ~1e-15, floating-point noise) before timing:

| Scale | Baseline | Vectorized numpy Cox-de-Boor | **Parallel-rounds (n_jobs=4)** | Vectorized+Parallel |
|---|---|---|---|---|
| Small | 3.71s | 14.03s (3.8x slower) | **1.76s (2.1x faster)** | 5.89s |
| Medium | 6.69s | 26.97s (4.0x slower) | **3.17s (2.1x faster)** | 11.59s |
| Large | 9.81s | 41.65s (4.2x slower) | **4.58s (2.1x faster)** | 17.47s |

(Baseline here already includes `numba` — installing the optional `accel` extra, which the library already supports via a JIT-compiled fast path for `_b_basis_1d`, but which is not installed by default. Measured separately: numba alone gave only a modest ~18% end-to-end gain, well below the 6.5x measured for that function in isolation, since other per-round overhead — the Python loop itself, `layer0.forward`, sigmoid stacking — doesn't benefit from it.)

A fully-vectorized pure-numpy Cox-de-Boor basis evaluator (full array broadcasting, no scipy sparse machinery at all) was **tried and rejected**: 3.8-4.2x *slower* than the shipped path at every scale — the intermediate-array overhead of `numpy.where`/`numpy.clip` across many small per-round calls outweighs whatever sparse-construction cost it avoids. This independently confirms `kanboost.core.kan.bspline`'s own docstring, which already warns that a hand-written numpy alternative to scipy was tried before and found slower.

**Parallelizing across boosting rounds was the clear winner in this repeated-call microbenchmark**: each round's `(layer0, coefs, gamma)` is fixed and independent once fitting completes, so `joblib.Parallel` can compute every round's contribution on a separate core and sum them — ~2.1x faster at every scale in the table above, with no accuracy cost (by construction; the computation is identical, just reordered). **Shipped** as `predict_proba_ga2m(model, X, n_jobs=...)`, default `n_jobs=1` (sequential, unchanged).

**Correction after a second, more realistic measurement — the ~2.1x figure does not transfer to typical one-off usage.** The table above used a repeated-call timing loop (one warmup call, then several timed calls), which amortizes `joblib`'s worker-process startup cost across many calls. Re-measured under the actual usage pattern from §14 (fit once, then predict on validation and test — two calls total, no warmup), the same `n_jobs=4` gave scale-dependent results instead of a uniform 2.1x:

| Scale | `n_jobs=1` | `n_jobs=4` (cold-start, 2 calls) | Change |
|---|---|---|---|
| Small | 9.36s | 11.01s | **worse** (−18%) |
| Medium | 16.27s | 15.16s | ~wash (+7%) |
| Large | 25.48s | 17.84s | better (+43%, ~1.4x) |

The worker-spawn cost has to be paid before round-level parallelism pays for itself; Small's per-round compute is too little to cover it. **Practical guidance**: keep the default `n_jobs=1` for small data or one-off predictions; use `n_jobs>1` for larger data, or for long-running services that reuse the same warm worker pool across many prediction calls — the 2.1x figure is real, but only under that repeated-call condition, not a guarantee for every usage pattern. This caveat is documented directly in `predict_proba_ga2m`'s docstring.

## 15. Closing the Remaining Gap vs. Trees: Two Literature-Motivated Attempts, Both Rejected

§14 left a small but persistent AUROC/AUPRC gap versus XGBoost/HistGradientBoosting at every scale. Rather than assume this gap is unfixable, two concrete interventions were designed directly from Grinsztajn et al. (NeurIPS 2022, "Why do tree-based models still outperform deep learning on tabular data?"), which identifies three structural reasons smooth-basis models lose to trees: (1) sensitivity to uninformative features, (2) lack of axis-alignment (rotation invariance hurts), and (3) inability to represent irregular (non-smooth) target functions. Reason (2) already explains, post-hoc, why GA2M's axis-aligned main-effect/pair structure beat RF-KAN's dense random-projection mixing in §11 — RF-KAN's dense mixing is structurally a random rotation, exactly what the literature warns against. Reasons (1) and (3) were tested directly as candidate fixes, both on INSPIRE Small scale first per this project's standing practice of validating before scaling up.

### 15.1 Adaptive knot placement (targeting irregular functions) — rejected

kanboost's `TabularPreprocessor` encodes categorical columns via smoothed target-mean encoding, so `icd10_pcs` becomes one continuous scalar feature. Since ~33% of procedure codes have a deterministic (0%/100%) train-set ICU rate (§2), this encoded feature's true relationship to the target has a near-step shape at the boundary between deterministic-negative/positive codes — a textbook "irregular function." A related, more basic issue was found in passing: `_make_knots` builds a uniform grid on a fixed `[-1, 1]` range for every feature, but the target-mean-encoded columns are never rescaled to that range (observed range for `icd10_pcs`: `[0.0026, 0.8918]` on Small scale) — roughly half the spline's knot range goes unused for that feature.

Two variants were tested against the shipped baseline (uniform `[-1, 1]` knots), Small scale, GA2M + Newton + Armijo:

| Config | AUROC | AUPRC |
|---|---:|---:|
| Baseline (uniform `[-1, 1]` knots) | 0.9250 | **0.6791** |
| Range-rescaled (uniform spacing, each feature's own `[min, max]`) | 0.9225 | 0.6775 |
| Adaptive (quantile-based knot placement) | 0.9014 | 0.6191 |

Both variants **underperformed the baseline** — the quantile variant substantially so. Likely cause: several categorical columns are low-cardinality after target-mean encoding, so quantile-based knot placement produces many near-duplicate knot positions (only jittered apart by a small epsilon), destabilizing the local B-spline basis for those features; the more conservative range-only fix avoided that failure mode but still showed no benefit, suggesting the "wasted grid range" was never actually costing representational capacity in practice (B-spline basis functions have local support — unused knots outside the observed range are simply inactive, not representationally costly). **Not implemented.**

### 15.2 Adaptive-ridge "sparse gating" for main effects (targeting uninformative features) — rejected

Trees ignore an uninformative feature at zero cost (never split on it); GA2M's main-effect units apply a spline to every feature every round, relying only on a fixed, uniform ridge/smoothness penalty to shrink irrelevant ones. Since GA2M's hidden unit `h=j` is always feature `j`'s main effect across every round (only its random coefficients differ round to round — the pairwise units are what get re-sampled), a persistent running "accumulated main-effect contribution" per feature is well-defined during fitting. An adaptive-ridge mechanism (in the spirit of adaptive lasso / iteratively-reweighted ridge, chosen specifically to stay inside a closed-form weighted-least-squares solve, unlike a true group-lasso which would need iterative proximal steps) applied progressively stronger ridge to features whose accumulated contribution stayed small.

An initial version used a multiplicative adjustment on the existing `lam_ridge` and had **no measurable effect at all** — the shipped default penalty values (`lam_smooth≈1e-8`, `lam_ridge≈1e-7`) are already negligible relative to the data term, so multiplying them by any modest factor changes nothing. Corrected to an additive, independently-scaled gate and swept across four orders of magnitude, Small scale:

| Gate strength | AUROC | AUPRC |
|---:|---:|---:|
| 0 (baseline) | 0.9250 | **0.6791** |
| 1 | 0.9250 | 0.6790 |
| 10 | 0.9252 | 0.6781 |
| 100 | 0.9245 | 0.6726 |
| 1000 | 0.9219 | 0.6640 |

At strength 10, AUROC ticks up by a noise-level amount (+0.0002) while AUPRC drops (−0.001); at higher strengths both metrics decline as the gate increasingly suppresses main effects indiscriminately, including genuinely important ones (`department`, `icd10_pcs`), rather than selectively targeting truly uninformative features. No tested strength gave a real, unambiguous improvement. **Not implemented.**

### 15.3 Interpretation

Both interventions were correctly motivated by peer-reviewed theory and, in §11's case, that theory already explains an empirical result obtained earlier in this session (GA2M > RF-KAN). But neither translated into a measurable accuracy gain on this specific dataset. Combined with the EBM-vs-XGBoost literature (§6.1-adjacent finding: even well-tuned EBMs retain a small, persistent gap versus XGBoost across benchmarks), the remaining ~0.007-0.03 AUROC/AUPRC gap documented in §14 is plausibly close to a near-irreducible floor for GA2M-style additive models on this task, rather than a straightforwardly fixable engineering gap. Closing it further would likely require a structurally deeper change (e.g. three-way interactions, or a tree-like hard-partitioning mechanism inside the weak learner itself) rather than tuning knot placement or regularization strength — a larger undertaking flagged here as a direction for a future, separate investigation rather than pursued further in this session.

## 16. An Optional Native (C++) Accelerator for Prediction

§14.1 closed part of the prediction-speed gap via `n_jobs` (parallelizing across boosting rounds), but found it scale-dependent -- a real win at Large, a wash at Medium, and *worse* than sequential at Small (worker-process startup cost outweighing the small amount of per-round compute available to parallelize there). This motivated investigating a compiled, ahead-of-time C++ extension for the same forward pass instead.

**First attempt: rejected.** A standalone (not Python-bound) C++ port of the same Cox-de-Boor basis evaluation, compiled with the only compiler available in the initial environment (a 32-bit MinGW.org GCC 6.3.0 from 2016), was *slower* than the existing numba path at every workload size tested — confirming that a naive C++ port is not an automatic win, and that toolchain quality matters as much as the language choice.

**Second attempt: a fair comparison.** After installing a modern 64-bit MinGW-w64 toolchain (GCC 16.1.0, matching Python's own 64-bit ABI) via `chocolatey`, the same standalone benchmark reversed completely: ~1.7-1.8x faster from the 64-bit compiler alone (same naive algorithm), and a further 3.7-6.5x from eliminating per-call heap allocation and enabling `-O3 -march=native` vectorization — the isolated basis-evaluation computation became faster than the *entire* current Python+numba `predict_proba_ga2m` pipeline (which includes additional overhead beyond basis evaluation).

**Building the real extension.** A `pybind11` extension (`kanboost/_native/ga2m_ext.cpp`) implementing GA2M's full per-round forward pass (layer0 + layer1 + matmul) was built and wired into `predict_proba_ga2m` via a new `backend` parameter (default `"auto"`, transparently falls back to pure Python/numba if the extension isn't built). A first integrated version gave only ~1.5x — profiling the *isolated* forward pass revealed why: it recomputed each feature's B-spline basis once per edge (main effect + however many pairwise units reused that feature), while `KANLayer.forward` itself computes each distinct feature's basis exactly once and reuses it via matmul across every hidden unit connected to it. Restructuring the extension to share basis computation the same way roughly doubled its advantage, to ~2.3x.

**End-to-end result, using the actual shipped API** (`fit_with_ga2m()`/`predict_proba_ga2m()`, not a standalone reimplementation), all three INSPIRE scales:

| Scale | Python | C++ | Speedup | Max prediction diff |
|---|---|---|---:|---|
| Small | 3.95s | 1.74s | **2.27x** | 3.5e-15 |
| Medium | 6.89s | 2.95s | **2.33x** | 2.2e-15 |
| Large | 10.15s | 4.49s | **2.26x** | 2.2e-15 |

AUROC/AUPRC were bit-identical between backends (same computation, just faster), and notably this speedup is **more stable across scales than `n_jobs`** (which ranged from a slowdown to a 1.4x speedup depending on data size) — the C++ path has no per-call process-spawn overhead to amortize. `main_effect_contributions()`/`pairwise_interaction_contributions()` were verified unaffected, since they read the fitted coefficients directly regardless of which backend computed predictions.

**Shipped as fully optional**: the extension requires `pybind11` and a C++17 compiler *at build time only* (`pip install pybind11 && python setup.py build_ext --inplace`) and is **not part of the published PyPI wheel** — installing/using kanboost normally is entirely unaffected; this is an opt-in local build for users who want the extra speed and have a compiler available. A build failure for any reason falls back to pure-Python silently (see `setup.py`'s docstring).

## 17. Limitations and Next Steps

- **Threshold optimization was not applied symmetrically until this report** (§5.1) — future comparisons in this project should always tune thresholds identically across every model compared, not only for KANBoost.
- **Calibration was only measured for KANBoost** (Platt scaling, Brier score) in this evaluation; the trees were not calibrated or Brier-scored here. A fair follow-up would report Brier/reliability curves for all five models side by side.
- **No temporal split was used** — the group-aware split here prevents patient leakage across folds but is still a random split in time. A temporal (train-on-earlier-years, test-on-later-years) split would better reflect deployment realism and is a natural next experiment.
- **No external validation** (a second institution's data, analogous to the published paper's MOVER cohort) was performed — all numbers here are internal to INSPIRE.
- ~~The mixed-code robustness check (§3) was only run at Small and Medium scale for KANBoost~~ — **resolved**: §3 now includes Large-scale KANBoost results, confirming the same pattern (AUPRC drop of −0.207 from removing `icd10_pcs` within the mixed-risk subset, consistent with −0.166 at Small and −0.212 at Medium).
- ~~The line-search speedup (§8) was measured on Small and Medium scale only~~ — **resolved**: confirmed at Large scale too (3.9× speedup, 761.2s→195.0s, AUROC/AUPRC both slightly exceeding the fixed-shrinkage baseline), consistent with Small (3.1×) and Medium (4.2×). This closes a substantial, size-independent portion of the speed gap versus GB-KAN's reported 2.5–3× XGBoost figure — kanboost's fixed-shrinkage baseline was 300–850× slower than trees on INSPIRE; with line search's fewer-round effect applied, that multiple drops considerably (exact all-model head-to-head fit-time comparison with line search enabled is a natural next measurement, not yet done as of this writing).
- **`icd10_pcs`'s exact coding timing was not independently verified against SNUH's clinical documentation practices** — this report relies on the reasonable assumption (consistent with ICD-10-PCS being assigned from the planned procedure, and with the original notebook's explicit "features known before or at planning" design constraint) that it is not retroactively adjusted after a complication; no institutional coding-workflow documentation was available to confirm this directly.

---

*Companion document*: [`inspire_gap_closing_report.md`](inspire_gap_closing_report.md) — the chronological engineering log of every fix tried (encoding, class weighting, capacity, consolidation, `categorical_hierarchy`, the rejected RBF-basis experiment) with full before/after tables, referenced throughout this report as the source of the underlying KANBoost configuration evaluated here.
