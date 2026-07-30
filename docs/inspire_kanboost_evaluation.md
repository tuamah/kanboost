# KANBoost Evaluation on INSPIRE ICU Admission: Leakage Audit, Procedure-Code Robustness, CPB Sensitivity, and Operational Decision Performance

**Date**: 2026-07-31
**Dataset**: INSPIRE (PhysioNet, Seoul National University Hospital perioperative dataset), local copy `INSPIRE/operations.csv` (130,960 rows, 99,886 patients).
**Split**: group-aware (`StratifiedGroupKFold` by `subject_id`, 60/20/20 train/val/test), fixed across all scales and both label definitions below. Three nested training scales (Small ⊂ Medium ⊂ Large): 10,000 / 50,000 / 78,496 rows.
**kanboost version**: 1.4.0. **Hardware**: CPU-only (no CUDA torch build available on this machine).

This document supersedes the earlier narrative in [`inspire_gap_closing_report.md`](inspire_gap_closing_report.md) as the primary scientific record for this dataset — that report remains as the chronological methodology/engineering log (which fixes were tried, in what order, with what code); this one is the audited, defensible scientific summary.

## Executive Summary

**KANBoost does not outperform tree-based models (XGBoost/LightGBM/CatBoost/HistGradientBoosting) in raw discrimination metrics (AUROC, AUPRC) on this task, at any scale, under either label definition tested.** With its integrated calibration (Platt scaling) and F1-oriented threshold optimization (`find_threshold`), it can produce competitive *operational* decisions — but that advantage disappears once the same threshold optimization is applied fairly to the tree baselines too (§6). The one property that survives every audit performed here is that KANBoost's predictions carry genuine, non-trivial clinical-risk signal — verified by removing the dataset's easiest, near-deterministic cases entirely (§4) — not merely a memorized lookup over institutional care-pathway rules.

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

## 3. Mixed-Code Robustness Analysis

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

**Interpretation**: restricting KANBoost entirely to procedures with genuinely uncertain (2–98%) historical ICU rates barely moves its AUROC/AUPRC relative to evaluating the full model on that same subset — it is not relying on deterministic codes leaking into training indirectly. And `icd10_pcs` still adds a large, consistent improvement *within this hard subset specifically* — Medium scale: AUROC +0.051, AUPRC +0.212 versus removing it. This is the direct evidence that KANBoost captures real clinical-risk signal, not just a lookup table over institutional routing rules. (Brier score is visibly worse in the mixed-only subsets — expected, since removing near-constant-target rows removes exactly the cases a calibrated model finds easiest to get right, leaving proportionally harder, more genuinely uncertain cases.)

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

## 7. Scientific Interpretation

**KANBoost is a competitive, calibrated decision pipeline on this task — not yet a superior discriminator.** It does not exceed tree ensembles' AUROC or AUPRC at any scale or label definition tested. Its earlier-observed F1 competitiveness was a pipeline artifact (fair threshold tuning applied to both sides eliminates it, §5.1). What is genuinely established, and survives the most adversarial test available (removing every near-deterministic procedure code from both training and evaluation, §3): KANBoost's predictions carry real, non-trivial clinical-risk signal, not a memorized lookup over institutional care-pathway rules. Any external claim about this project should be scoped to that — a calibrated, interpretable alternative to tree boosting that is closing but has not closed the discrimination gap — not to matching or beating either the tree baselines or the published literature on raw performance.

## 8. Limitations and Next Steps

- **Threshold optimization was not applied symmetrically until this report** (§5.1) — future comparisons in this project should always tune thresholds identically across every model compared, not only for KANBoost.
- **Calibration was only measured for KANBoost** (Platt scaling, Brier score) in this evaluation; the trees were not calibrated or Brier-scored here. A fair follow-up would report Brier/reliability curves for all five models side by side.
- **No temporal split was used** — the group-aware split here prevents patient leakage across folds but is still a random split in time. A temporal (train-on-earlier-years, test-on-later-years) split would better reflect deployment realism and is a natural next experiment.
- **No external validation** (a second institution's data, analogous to the published paper's MOVER cohort) was performed — all numbers here are internal to INSPIRE.
- **The mixed-code robustness check (§3) was only run at Small and Medium scale** for KANBoost specifically (CPU-only hardware made the Large-scale version impractical to add within this session); the LightGBM version of the same check (§2) was run at Large scale. Extending §3 to Large scale for KANBoost is the most direct remaining gap to close.
- **`icd10_pcs`'s exact coding timing was not independently verified against SNUH's clinical documentation practices** — this report relies on the reasonable assumption (consistent with ICD-10-PCS being assigned from the planned procedure, and with the original notebook's explicit "features known before or at planning" design constraint) that it is not retroactively adjusted after a complication; no institutional coding-workflow documentation was available to confirm this directly.

---

*Companion document*: [`inspire_gap_closing_report.md`](inspire_gap_closing_report.md) — the chronological engineering log of every fix tried (encoding, class weighting, capacity, consolidation, `categorical_hierarchy`, the rejected RBF-basis experiment) with full before/after tables, referenced throughout this report as the source of the underlying KANBoost configuration evaluated here.
