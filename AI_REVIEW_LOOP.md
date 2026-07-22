# Depth-3 KAN Research Loop

Governed by the protocol in `CLAUDE.md` (KANBoost AI Review Protocol) --
read that file first; this document is the live ledger it requires under
rule 14.

Branch: `research/depth3-kan` (isolated from `main`; the stable depth-2
`kanboost/core/kan/` implementation is untouched throughout this work).

Pipeline (see `CLAUDE.md`): ChatGPT (hypothesis/protocol) -> Claude Code
(implement + test) -> Codex (independent code/results review, no edits) ->
ChatGPT (scientific judgment) -> User (merge approval).

Loop: Hypothesis -> Implementation -> Tests -> Evidence -> Codex Review -> ChatGPT Judgment -> User Approval -> Next Experiment.

All experiment code lives in the scratchpad
(`.../scratchpad/depth3_prototype.py`, `depth2_prototype.py`,
`depth_experiment_v2.py`, `depth_residual_prototype.py`,
`depth_residual_prototype.py`, `depth_experiment_residual.py`), never in
`kanboost/`, until/unless a proposal is explicitly approved for promotion.

## Proposal naming convention

**Added 2026-07-22 (user-requested), after Claude and Codex independently
numbered two unrelated proposals "Proposal 6"** (categorical
target-encoding leakage vs. OpenNeuro quality gap vs HistGBDT). To prevent
this recurring:

- Every new proposal gets a **model-prefixed ID**: `CC-<n>` for a
  proposal originated by Claude Code, `CX-<n>` for one originated by
  Codex. A proposal raised jointly/by the user directly may use `JT-<n>`.
- `<n>` is a running integer PER PREFIX, not shared -- i.e. `CC-7` and
  `CX-7` can coexist without collision, since the prefix already
  disambiguates.
- Before opening a new heading (`### Proposal CC-n -- ...` /
  `### Proposal CX-n -- ...`), grep this file for the exact ID first to
  confirm it is unused.
- Existing collided entries were retagged in place: the categorical
  leakage fix is now **CC-6**, the OpenNeuro HistGBDT quality-gap
  proposal is now **CX-6**. Older, already-resolved proposals (1-5, 7)
  were not retagged retroactively since they never collided -- only
  rename an old heading if a real collision is later found there too.

---

## Codex Review Package (current snapshot)

Refreshed after every significant change, per user request, so Codex can
pick up full context on demand without re-deriving it. This section is the
single source of truth for "what changed and why, right now" -- read it
before diffing anything.

**Active shared goal (user, 2026-07-22)**: Claude Code and Codex should
work in coordination toward the highest realistically measured KANBoost
accuracy and speed. Treat this as an applied-science optimization loop:
study where strong competing models outperform KANBoost, define the exact
gap, propose a mathematically and computationally plausible way to close it,
agree on acceptance criteria, then test with controlled local and remote
evidence. Separate model-quality evidence from speed evidence, and do not
promote any change that improves one by hiding a regression in the other.

**Required proposal shape for this goal**:
- Competitor/gap: what CatBoost, XGBoost, LightGBM, logistic/GAM, neural KAN,
  or a domain-specific pipeline currently does better.
- Hypothesis: what change may close the gap and why it is plausible
  mathematically/computationally.
- Scope: training speed, prediction speed, memory, accuracy, calibration,
  generalization, robustness, interpretability, or benchmark infrastructure.
- Acceptance gate: exact metric, tolerance, runtime target, dataset/folds,
  seeds, and failure/rejection rule before implementation.
- Evidence: Claude local result, Codex local result, and Kaggle/Colab remote
  result side-by-side whenever feasible; include commands, files changed,
  uncertainty, and final accept/reject/inconclusive decision. If one local
  side cannot run because of missing dependencies or weak hardware, state
  that explicitly and still compare the available local evidence against
  Kaggle/Colab.
- Double-experiment requirement: for every agreed speed or resource-use
  proposal, Claude Code and Codex should each run the same protocol
  independently when feasible, record attributed local results, then one
  model sends the exact same experiment to Kaggle/Colab. The final decision
  must compare Claude local vs Codex local vs remote results, including
  notebook/kernel name, dataset, timestamped output path, hardware notes,
  runtime, metrics, parity tolerances, and any skipped/failed side. A remote
  result confirms transferability; it does not erase local discrepancies.

**User operating approval (2026-07-22)**: for the next active work window,
the user wants Claude Code and Codex to proceed autonomously through the
coordination loop: propose, discuss in this ledger, run agreed experiments,
and report accumulated evidence for user review. This approval covers
research proposals, local tests, Kaggle/Colab experiment runs, and result
recording. It does **not** override the standing gates against merging to
`main`, publishing a public API change, committing unrelated work, or
touching credentials/secrets without an explicit task-specific instruction.

**Current decision matrix (refresh 2026-07-22, Codex)**:
- **Accepted / strongest speed evidence**: Proposal 2, GAM training
  basis/system cache (`gam=True`, `kan_hidden=1`). Claude local measured
  +114.7% to +151.8% speedup with exact R2 parity; Codex Kaggle measured
  +160.8% to +237.7% with exact R2 parity. This is the strongest proven
  training-speed improvement and is pending only the normal review/merge
  gate, not more proof-of-speed.
- **Accepted / strongest prediction-speed evidence**: CX-13 v2
  ensemble-level forward basis cache reference. Kaggle measured 2.2x-3.0x
  speedups for non-GAM `predict`/`predict_proba`, 15.3x for a p32/e100 GAM
  prediction case, and exact public-output parity (`max_abs_diff=0.0`,
  classifier classes identical). This is ready for a production
  implementation pass, but because AGENTS currently defines Claude as
  primary implementer and Codex as read-only reviewer, Codex should either
  hand the implementation spec to Claude or wait for explicit permission to
  edit core prediction code.
- **Accepted by user for later Colab validation**: Proposal 5,
  per-call basis reuse for interpretability methods
  (`feature_contributions`, `predict_derivative`). Claude local found
  very large reporting/dashboard speedups with exact parity; Codex Kaggle
  first confirmed the idea, and the concrete `1.2.4` wheel rerun later
  confirmed the release-candidate artifact itself: `feature_contributions`
  26.6x to 87.7x faster than an uncached reference, GAM
  `predict_derivative` 5.5x to 10.8x faster, with exact/near-exact parity
  (0 to ~3e-15). User approved preparing a trial version for their own
  Colab/external reruns; Codex bumped local metadata to `1.2.4`, built
  `dist/kanboost-1.2.4-py3-none-any.whl`, and validated that wheel on
  Kaggle. Keep this separate from OpenNeuro model-quality work; it
  accelerates analysis/reporting, not training accuracy.
- **Accepted / strongest accuracy evidence so far**: CX-14 refined the
  CX-12 EEG-derived feature win. Kaggle large-stage OpenNeuro result:
  `kanboost_eeg_select80_h3_e80_s6` balanced accuracy 0.777, macro F1
  0.773, log loss 0.568, ROC AUC 0.785, beating raw HistGBDT's 0.749
  balanced accuracy and 0.712 log loss, and passing the stronger >=0.769
  accuracy target. CX-15 repeated-seed validation tempers that claim:
  across 25 folds, the same KANBoost arm averages 0.689 balanced accuracy
  vs raw HistGBDT 0.676, with much better log loss (0.576 vs 0.773). Thus
  CX-14 is the best single-split frontier, while CX-15 is the more robust
  scientific claim: small balanced-accuracy lead, strong calibration lead.
  CX-16 stability selection is only a partial/inconclusive robustness
  improvement (+0.008 repeated balanced accuracy over ordinary SelectKBest,
  not the required +0.02).
  CX-17 inner-CV thresholding was high-value but narrowly inconclusive
  (0.714 vs 0.716 gate). CX-18 fine inner-threshold refinement is now
  accepted as the best repeated-CV balanced-accuracy result: 0.718 BA,
  0.576 log loss over 25 folds, beating raw HistGBDT 0.676/0.773.
- **Rejected speed paths**: Proposal 1 batched layer-0 ALS solve is
  hardware-dependent and unreliable; Proposal 3 non-GAM prediction cache
  fails the 2x gate and is not worth promotion; broad C++/compiled paths
  remain gated until profiling shows a real bottleneck after Proposal 2.
- **Rejected quality tweak**: Proposal 7 inner-CV threshold tuning is not
  a fix for KANBoost balanced accuracy on OpenNeuro; it hurt the KANBoost
  project arm and should not be rerun unless the model/protocol changes.
  CX-8 probability blending is also rejected for the accuracy frontier:
  it improved log loss but failed to beat HistGBDT balanced accuracy.
  CX-9 wider/deeper KANBoost capacity search is rejected: increasing
  hidden width, estimators, or ALS steps did not beat HistGBDT on
  OpenNeuro. CX-10 fold-local feature selection is rejected as a final
  accuracy fix: it improved KANBoost balanced accuracy from 0.702 to
  0.734, but still missed HistGBDT's 0.749 gate.
- **Open / needs consolidation**: Proposal 6's original OpenNeuro raw
  feature quality gap is no longer the frontier after CX-12; KANBoost can
  beat HistGBDT when the EEG feature representation is made more
  physiologically structured. Still missing: Claude local side, Codex local
  side with full dependencies, and repeated/resampled stability evidence
  around the winning CX-12 arm.
- **Protocol state**: OpenNeuro notebooks now pin KANBoost commit
  `11f4b0fd95ac40aa475702a81d5efbb749cee55b`, use fold-local
  `SimpleImputer -> VarianceThreshold` cleaning, keep raw OpenNeuro data
  under `/kaggle/temp`, and write timestamped outputs under
  `/kaggle/working/outputs`.
- **Next best work**: for accuracy, do not run another broad grid. The
  useful next evidence is stability: repeat CX-14's winning arm across
  multiple CV seeds/repeated stratified folds and compare against raw
  HistGBDT and raw+derived HistGBDT. CX-15 has now done this and accepted
  stability with caveats. For speed, the next best production work is to
  implement CX-13's cached-forward path in core prediction code with the
  exact float32 rounding invariant from v2, then rerun the same Kaggle
  parity/speed gate.
- **Updated next best work after CX-18**: for accuracy, consolidate rather
  than search: fold the CX-18 accepted protocol into the OpenNeuro
  notebook/report as a benchmark-only decision rule and, if time permits,
  run one final grouped/subject-level sanity check if the dataset metadata
  supports it. For speed, proceed with CX-13 production implementation via
  Claude or explicit user permission for Codex to edit core code.

**[Codex Correction, 2026-07-22] Goal is NOT complete**:
The previous synthesis below is only a checkpoint of the best evidence so
far, not the end of the user's objective. The user explicitly corrected
Codex: we have not yet reached the highest possible accuracy or highest
possible speed. Continue the Claude/Codex loop. Treat Proposal 2 and
Proposal 5 as strong accepted wins, Proposal 6 as promising but not the
final quality frontier, and keep searching for new high-value accuracy and
speed gaps before any final closeout.

Immediate renewed target:
- Accuracy: beat or clearly match the best OpenNeuro large-stage competitor
  on balanced accuracy, not only stay within 0.05, while preserving
  KANBoost's log-loss/calibration advantage.
- Speed: after Proposal 2 and Proposal 5, profile the remaining production
  bottlenecks for non-GAM training/prediction and real benchmark workflows;
  propose only changes with clear parity gates and expected transferability
  to Kaggle/Colab.
- Coordination: Claude and Codex should add new proposals with acceptance
  criteria before implementation, then record accepted/rejected/inconclusive
  outcomes. Do not call the overall goal complete until no plausible
  high-value next experiment remains.

**[Codex Proposal CX-8, 2026-07-22] OpenNeuro probability-calibrated
stacking / residual blend to beat the large-stage balanced-accuracy leader**

Competitor/gap: on the pinned OpenNeuro large-stage Kaggle run,
`hist_gbdt_clean` leads balanced accuracy (0.749) while
`kanboost_project_clean` is lower (0.702) but has better log loss (0.617
vs 0.712). This suggests the models carry complementary signal:
HistGBDT's thresholded decisions are stronger, KANBoost's probabilities
are better calibrated.

Hypothesis: a leakage-safe, fold-local meta-model or convex probability
blend using KANBoost + HistGBDT (+ optionally logistic/RBF-KAN) can improve
balanced accuracy beyond either single model while preserving KANBoost's
calibration advantage. The simplest first test is not a broad ensemble
rewrite: use outer 5-fold CV, generate inner-CV out-of-fold probabilities
on each outer train fold, learn either (a) one logistic meta-classifier on
probabilities, or (b) a scalar convex weight grid optimizing balanced
accuracy with log-loss tie-break, then evaluate once on the held-out outer
fold.

Scope: model quality / calibration on real OpenNeuro EEG features. Not a
training-speed proposal.

Acceptance gate:
- Primary: large-stage mean balanced accuracy must beat the current best
  single model by >=0.02 absolute, i.e. >=0.769 against HistGBDT's 0.749,
  OR reach >=0.749 while improving mean log loss by >=0.05 vs HistGBDT.
- Secondary: macro F1 must not drop by more than 0.02 from the best single
  model; no fold may use validation labels in preprocessing, feature
  selection, thresholding, or meta-model training.
- Rejection: if the blend only matches KANBoost's log loss while balanced
  accuracy remains below HistGBDT, or if it needs test-fold threshold
  tuning, reject as not closing the real accuracy frontier.

Protocol:
- Start from the already exported pinned large feature CSV under
  `remote/results/kaggle_openneuro_clean_pinned_codex/outputs/`.
- Reuse the exact fold-local cleaning (`SimpleImputer ->
  VarianceThreshold`) and outer `StratifiedKFold(random_state=42)`.
- First run as a features-only Kaggle/Codex experiment to avoid re-downloading
  EEG. If it passes, Claude can decide whether to fold it into the full
  OpenNeuro notebook. Record as CX-8 and do not confuse it with CC-6/CX-6.

**[Codex Review, 2026-07-22] CX-8 implementation prepared; Kaggle push
temporarily failed**: Codex implemented the features-only Kaggle script
under `remote/kaggle_cx8_openneuro_blend/cx8_openneuro_blend.py` with
metadata in `remote/kaggle_cx8_openneuro_blend/kernel-metadata.json`.
The script passed local `py_compile` under bundled Python. It attaches the
existing Kaggle Dataset
`tuamamhamza/openneuro-ds004504-large-clean-features`, installs pinned
KANBoost commit `11f4b0fd95ac40aa475702a81d5efbb749cee55b`, and evaluates
single models plus leakage-safe inner-CV probability blends. First attempt
to push kernel `tuamamhamza/cx8-openneuro-kanboost-blend` failed during
Kaggle API token introspection with
`ssl.SSLEOFError: UNEXPECTED_EOF_WHILE_READING`; this is an external API
transport failure, not a script compile failure. Next action: retry
`kaggle kernels push -p remote/kaggle_cx8_openneuro_blend` when Kaggle API
connectivity stabilizes, then download outputs and accept/reject CX-8 by
the gate above.

**[Codex Review, 2026-07-22] CX-8 Kaggle run complete -- REJECT for
balanced-accuracy frontier**: Retry succeeded; kernel
`tuamamhamza/cx8-openneuro-kanboost-blend` completed on Kaggle. Outputs
downloaded under `remote/results/kaggle_cx8_openneuro_blend/` with prefix
`kaggle_cx8-openneuro-blend_ds004504_large_clean_features_blend_20260722-041433`.

Summary:

| model | balanced accuracy | macro F1 | log loss | ROC AUC |
|---|---:|---:|---:|---:|
| `hist_gbdt_clean` | 0.749 | 0.751 | 0.712 | 0.770 |
| `kanboost_project_clean` | 0.702 | 0.691 | 0.617 | 0.747 |
| `logreg_clean` | 0.700 | 0.697 | 0.906 | 0.740 |
| `blend_kanboost_hist_gbdt_inner_balacc` | 0.699 | 0.695 | 0.609 | 0.708 |
| `blend_mean_kan_hist_logreg` | 0.685 | 0.683 | 0.569 | 0.760 |
| `blend_kanboost_logreg_inner_balacc` | 0.665 | 0.663 | 0.591 | 0.726 |

Decision: reject CX-8 as an accuracy-frontier fix. It confirms that
KANBoost probabilities can improve calibration/log loss in blends, but the
leakage-safe inner-CV blend does not beat the HistGBDT balanced-accuracy
leader and does not satisfy the predefined >=0.769 gate. Do not promote
this blend as a model-quality improvement. A future OpenNeuro accuracy
proposal must target feature representation/model capacity or a stronger
single KANBoost variant, not simple probability blending.

**[Codex Proposal CX-9, 2026-07-22] OpenNeuro KANBoost capacity/regularity
search on large clean features**

Competitor/gap: CX-8 shows simple post-hoc probability blending is not
enough. The best KANBoost arm remains `kanboost_project_clean`
(`kan_hidden=3`, `n_estimators=40`, `kan_steps=8`) at 0.702 balanced
accuracy, while HistGBDT reaches 0.749. KANBoost's better log loss suggests
the probability surface is useful, but the 0.5 decision boundary and/or
base learner capacity is not extracting enough class-separating signal.

Hypothesis: a very small KANBoost capacity/regularity search may recover
balanced accuracy without losing calibration: vary `kan_hidden`,
`n_estimators`, `kan_steps`, learning rate, and GAM/non-GAM shape under
the exact same fold-local cleaning. Because n=65 is small, the search must
be tiny and predefined to avoid overfitting-by-experiment.

Scope: model quality and runtime on OpenNeuro large clean features. Not a
core algorithm change unless one configuration robustly beats the current
frontier.

Candidate grid:
- `kan_hidden`: 2, 3, 5, 8
- `n_estimators`: 40, 80
- `kan_steps`: 6, 8
- `learning_rate`: 0.05, 0.1
- include one GAM reference (`kan_hidden=1`, `gam=True`, 80 estimators)

Acceptance gate:
- Accept only if a KANBoost configuration reaches mean balanced accuracy
  >=0.769 (>=0.02 above HistGBDT 0.749) with macro F1 >=0.731 and log loss
  <=0.712, OR reaches balanced accuracy >=0.749 with log loss <=0.617.
- Reject if the best KANBoost arm remains below 0.749 balanced accuracy,
  or if the apparent win is dominated by one fold with much worse log loss.
- Record fit/predict time; do not call it a speed improvement unless it is
  also faster than the current `kanboost_project_clean` at comparable
  quality.

Protocol: run features-only Kaggle on the existing
`openneuro-ds004504-large-clean-features` dataset. Keep the same outer
5-fold splits and no validation-label leakage. This is a search result,
not a final production default, unless Claude/Codex both later confirm it
under the double-experiment rule.

**[Codex Review, 2026-07-22] CX-9 Kaggle run submitted**:
Codex implemented the script under
`remote/kaggle_cx9_openneuro_kanboost_search/cx9_openneuro_kanboost_search.py`
with metadata in
`remote/kaggle_cx9_openneuro_kanboost_search/kernel-metadata.json`.
Local `py_compile` passed under bundled Python. Kernel
`tuamamhamza/cx9-openneuro-kanboost-search` version 1 was pushed
successfully. Status/results pending.

**[Codex Review, 2026-07-22] CX-9 Kaggle run complete -- REJECT**:
Kernel `tuamamhamza/cx9-openneuro-kanboost-search` completed. Outputs were
downloaded under `remote/results/kaggle_cx9_openneuro_kanboost_search/`
with prefix
`kaggle_cx9-openneuro-kanboost-search_ds004504_large_clean_features_kanboost_search_20260722-042249`.

Top summary:

| model | balanced accuracy | macro F1 | log loss | mean fit |
|---|---:|---:|---:|---:|
| `hist_gbdt_clean` | 0.749 | 0.751 | 0.712 | 0.131s |
| `kanboost_nongam_h3_e40_s8_lr0p1` | 0.702 | 0.691 | 0.617 | 0.723s |
| `kanboost_nongam_h3_e40_s6_lr0p1` | 0.702 | 0.691 | 0.618 | 0.649s |
| `kanboost_nongam_h2_e80_s6_lr0p1` | 0.693 | 0.690 | 0.608 | 0.984s |
| `kanboost_nongam_h5_e40_s6_lr0p05` | 0.676 | 0.671 | 0.629 | 0.899s |

Decision: reject CX-9 as an accuracy-frontier fix. The predefined grid did
not produce any KANBoost configuration reaching HistGBDT's 0.749 balanced
accuracy, and the best arm is the already-known project configuration. The
search does reinforce a useful constraint: simply increasing hidden width,
estimators, or ALS steps tends to worsen balanced accuracy or runtime on
this small n/high p OpenNeuro feature table. The next accuracy proposal
should reduce noise or change feature representation, not broaden
KANBoost capacity.

**[Codex Proposal CX-10, 2026-07-22] Fold-local feature selection for
non-GAM KANBoost on OpenNeuro large features**

Competitor/gap: CX-9 suggests KANBoost is not under-capacity; it is more
likely over-exposed to noisy/high-dimensional EEG summary features
(~113 features, n=65). Tree models can ignore weak features by splits,
while KANBoost's smooth weak learners may spend capacity fitting noisy
columns.

Hypothesis: adding fold-local feature selection before non-GAM KANBoost
will improve balanced accuracy by reducing noisy columns while preserving
calibration. This differs from the earlier `kanboost_gam_select20_clean`,
which tested GAM/select20 and was weak; CX-10 tests the winning non-GAM
shape with selected feature counts.

Scope: model quality/runtime on OpenNeuro large clean features. No core API
change.

Candidate grid:
- `SelectKBest(f_classif)` with `k` in {10, 20, 40, 80}
- KANBoost non-GAM configs: current winner h3/e40/s8/lr0.1 and compact
  h2/e80/s6/lr0.1 from CX-9's low-log-loss runner-up.
- Baselines: HistGBDT and current full-feature KANBoost h3/e40/s8/lr0.1.

Acceptance gate:
- Accept only if selected-feature KANBoost reaches >=0.749 balanced
  accuracy with log loss <=0.617, or >=0.769 balanced accuracy with log
  loss <=0.712.
- Reject if selected KANBoost remains below HistGBDT or improves only log
  loss while balanced accuracy falls.

Protocol: features-only Kaggle, same dataset and outer folds. Selection
must live inside the sklearn Pipeline after imputation/variance filtering,
so it is fit only on each training fold.

**[Codex Review, 2026-07-22] CX-10 implementation prepared; Kaggle push
temporarily failed**: Codex implemented the script under
`remote/kaggle_cx10_openneuro_feature_select/cx10_openneuro_feature_select.py`
with metadata in
`remote/kaggle_cx10_openneuro_feature_select/kernel-metadata.json`.
Local `py_compile` passed under bundled Python. First push attempt for
kernel `tuamamhamza/cx10-openneuro-feature-select` failed during Kaggle API
authentication with `ConnectionAbortedError(10053)` / connection aborted
by the host machine. This is a transport/API failure; the experiment is
ready for retry with
`kaggle kernels push -p remote/kaggle_cx10_openneuro_feature_select`.

**[Codex Review, 2026-07-22] CX-10 second Kaggle push retry failed**:
Codex rechecked the kernel metadata and reran local `py_compile`
successfully, then retried
`kaggle kernels push -p remote/kaggle_cx10_openneuro_feature_select`.
The second push also failed during Kaggle API authentication/token
introspection with `ssl.SSLEOFError: UNEXPECTED_EOF_WHILE_READING`. Treat
CX-10 as implementation-ready but remote-run-pending due to repeated
external Kaggle API transport failures. Do not mark CX-10 accepted or
rejected until the kernel actually runs.

**[Codex Review, 2026-07-22] CX-10 Kaggle run complete -- REJECT as final
accuracy-frontier fix, keep as useful signal**: A later retry succeeded;
kernel `tuamamhamza/cx10-openneuro-feature-select` completed on Kaggle.
Outputs were downloaded under
`remote/results/kaggle_cx10_openneuro_feature_select/` with prefix
`kaggle_cx10-openneuro-feature-select_ds004504_large_clean_features_select_20260722-044600`.

Top summary:

| model | balanced accuracy | macro F1 | log loss | mean fit |
|---|---:|---:|---:|---:|
| `hist_gbdt_clean` | 0.749 | 0.751 | 0.712 | 0.153s |
| `kanboost_select80_h2_e80_s6_lr0p1` | 0.734 | 0.727 | 0.598 | 0.731s |
| `kanboost_select40_h3_e40_s8_lr0p1` | 0.718 | 0.716 | 0.601 | 0.385s |
| `kanboost_select80_h3_e40_s8_lr0p1` | 0.717 | 0.709 | 0.606 | 0.534s |
| `kanboost_full_h3_e40_s8_lr0p1` | 0.702 | 0.691 | 0.617 | 1.168s |

Decision: reject CX-10 under the predefined gate because the best selected
KANBoost arm did not reach HistGBDT's 0.749 balanced accuracy and did not
meet the >=0.769 stronger gate. However, it is not a dead end: selecting
80 fold-local features with the compact h2/e80/s6 model improved balanced
accuracy by +0.032 over the full-feature KANBoost and improved log loss
from 0.617 to 0.598. The next OpenNeuro quality proposal should treat this
as evidence that denoising/representation matters more than raw capacity.

**[Codex Proposal CX-11, 2026-07-22] Post-Proposal-2/5 production speed
profiling before the next optimization**

Competitor/gap: Proposal 2 solved the GAM training hot path and Proposal 5
solved repeated interpretability calls. Proposal 3 showed that simple
layer0 prediction caching is not enough for non-GAM serving speed. We do
not yet have a fresh profile of the current production code after these
wins plus Claude's current `encoders.py` work.

Hypothesis: the next speed win should come from measured residual
bottlenecks, not from guessing. Likely candidates are non-GAM training
inside `_fit_learner` / `DeepKAN.fit`, B-spline basis construction in
layer solves, or Python-level per-learner prediction loops. A narrow
profile can rank them and prevent another low-yield Proposal 3-style
attempt.

Scope: benchmark/profiling infrastructure first; no production code change
until a single bottleneck accounts for enough wall time to justify a
proposal.

Protocol:
- Use synthetic regression/classification workloads with p=8 and p=32,
  `n_train` in {2000, 10000}, `n_estimators` in {40, 100}, non-GAM
  `kan_hidden` in {3, 16}.
- Capture fit, predict, predict_proba, `feature_contributions`, and GAM
  derivative timings under the current worktree, recording
  `kanboost.__version__`, git diff summary, CPU/platform, and cProfile top
  functions.
- If local Python dependencies remain unavailable, prepare the same script
  as a Kaggle/Colab run and mark local Codex side skipped.

Acceptance gate for a follow-up optimization proposal:
- Only open a new speed implementation proposal if one bottleneck is >=25%
  of wall time and has a plausible exact-parity optimization expected to
  save >=15% end-to-end on at least two problem sizes.
- Reject any candidate that only improves a microbenchmark while total
  workflow speed changes by <15%, or changes predictions beyond 1e-10 for
  regression/raw scores and exact class/proba tolerance gates for
  classifiers.

**[Codex Review, 2026-07-22] CX-11 implementation pushed to Kaggle**:
Codex prepared a production-speed profiling script under
`remote/kaggle_cx11_production_speed_profile/cx11_kanboost_production_speed_profile.py`
with metadata in
`remote/kaggle_cx11_production_speed_profile/kernel-metadata.json`.
Local `py_compile` passed under the bundled Python. Kernel
`tuamamhamza/cx11-kanboost-production-speed-profile` version 1 was pushed
successfully. It installs the `kanboost 1.2.4` wheel dataset, profiles
regression and classification fit/predict paths on representative
synthetic workloads, records cProfile cumulative-time tables, and writes
timestamped `metrics.csv`, `results.json`, `profiles.txt`, and
`environment.txt`. Status at push time: running; results pending.

**[Codex Review, 2026-07-22] CX-11 Kaggle run complete -- opens CX-13
serving/train-basis speed hypothesis**: Kernel
`tuamamhamza/cx11-kanboost-production-speed-profile` completed on Kaggle.
Outputs were downloaded under
`remote/results/kaggle_cx11_production_speed_profile/` with prefix
`kaggle_cx11-kanboost-production-speed-profile_synthetic_prod_profile_20260722-045323`.

Representative timings from `kanboost 1.2.4` wheel:

| case | task | fit | predict/proba | feature contributions |
|---|---:|---:|---:|---:|
| p8 h3 e40 n=2000 | regression | 3.436s | 0.050s | 0.008s |
| p32 h16 e100 n=2000 | regression | 13.612s | 0.565s | 0.052s |
| p8 h3 e40 n=10000 | regression | 2.307s | 0.134s | 0.011s |
| p32 GAM h1 e100 n=10000 | regression | 5.716s | 0.919s | 0.118s |
| p32 h16 e100 n=2000 | classification | 7.978s | 0.552s proba | n/a |

Profile evidence: non-GAM h16/e100 training spends most time inside
`network.py:_fit_als` and B-spline basis construction
(`bspline.py:_b_basis_1d_numba`, ~3.5-5.3s in the p32 h16 profiles).
For GAM large p32/e100, repeated forward calls dominate and almost all
that time is also B-spline basis construction (~4.4s of 5.7s). Proposal 5
already makes interpretability fast; the remaining speed frontier is
serving/forward/training basis reuse, not feature-contribution code.

Decision: CX-11 is accepted as profiling evidence, not as a production
change. Open a follow-up speed proposal only if it reuses basis safely
across learners/calls and proves exact parity. The likely candidate is
CX-13: ensemble-level prediction/forward basis cache for shared layer-0
knots, plus possibly a tighter training-forward cache where the same
`X_t` is forwarded repeatedly.

**[Codex Proposal CX-12, 2026-07-22] EEG-aware derived feature
representation before KANBoost selection**

Competitor/gap: CX-9 rejected raw capacity expansion and CX-10 showed
that fold-local feature selection helps KANBoost (+0.032 balanced
accuracy) but still does not catch HistGBDT. The large OpenNeuro feature
table has structured EEG band/channel columns; HistGBDT can discover
piecewise channel interactions, while KANBoost may need cleaner,
lower-noise coordinates that expose physiologically meaningful contrasts.

Hypothesis: deterministic, label-free EEG feature expansion can improve
KANBoost's decision boundary without leakage: per-band regional means,
within-band channel dispersion, left/right asymmetry, and band-power ratios
should expose signal in fewer smoother coordinates. Applying fold-local
`SelectKBest` after this expansion may retain KANBoost's log-loss advantage
while raising balanced accuracy.

Scope: OpenNeuro model-quality experiment only; no production code/API
change. The expansion is an experiment script/notebook transformation.

Candidate arms:
- Baselines: HistGBDT on raw clean features; KANBoost raw h3/e40/s8; best
  CX-10 selected arm h2/e80/s6 select80.
- New arms: KANBoost on raw+EEG-derived features with `k` in {40, 80, 120}
  and h2/e80/s6 plus h3/e40/s8.

Acceptance gate:
- Accept if any KANBoost derived-feature arm reaches >=0.749 balanced
  accuracy with log loss <=0.598, or >=0.769 balanced accuracy with log
  loss <=0.712.
- Reject if derived features only improve log loss/ROC AUC while balanced
  accuracy remains below HistGBDT, or if the improvement is driven by any
  label-aware preprocessing outside the CV fold.

Protocol: features-only Kaggle using the existing
`tuamamhamza/openneuro-ds004504-large-clean-features` dataset. Derived
features are computed row-wise without labels; imputation, variance
filtering, and `SelectKBest` remain inside each sklearn Pipeline fold.

**[Codex Review, 2026-07-22] CX-12 Kaggle run complete -- ACCEPT as
OpenNeuro accuracy improvement**: Kernel
`tuamamhamza/cx12-openneuro-eeg-derived-features` completed on Kaggle.
Outputs were downloaded under
`remote/results/kaggle_cx12_openneuro_eeg_features/` with prefix
`kaggle_cx12-openneuro-eeg-features_ds004504_large_clean_features_eeg_derived_20260722-045725`.
The experiment expanded the raw 115 feature columns to 201 row-wise,
label-free EEG features, then kept imputation/variance filtering and
`SelectKBest` inside the CV pipeline.

Top summary:

| model | balanced accuracy | macro F1 | log loss | ROC AUC | mean fit |
|---|---:|---:|---:|---:|---:|
| `kanboost_eeg_select80_h3_e40_s8` | 0.765 | 0.758 | 0.595 | 0.780 | 0.501s |
| `hist_gbdt_raw` | 0.749 | 0.751 | 0.712 | 0.770 | 0.128s |
| `kanboost_eeg_select80_h2_e80_s6` | 0.746 | 0.739 | 0.580 | 0.813 | 0.660s |
| `kanboost_raw_select80_h2_e80_s6` | 0.734 | 0.727 | 0.598 | 0.766 | 0.646s |
| `kanboost_raw_h3_e40_s8` | 0.702 | 0.691 | 0.617 | 0.747 | 1.060s |

Decision: accept CX-12 under the predefined first gate. The best KANBoost
derived-feature arm beats HistGBDT balanced accuracy by +0.015 and improves
log loss by -0.117. It also improves the raw KANBoost project arm by
+0.063 balanced accuracy and -0.022 log loss. This is the first evidence
in the loop that KANBoost can beat the OpenNeuro large-stage balanced
accuracy leader without leakage or threshold tuning. Still, it narrowly
misses the stronger >=0.769 gate, so the next accuracy experiment should
stabilize and refine around `select80,h3,e40,s8`, not restart from broad
capacity expansion.

**[Codex Proposal CX-13, 2026-07-22] Ensemble-level forward basis cache for
prediction speed**

Competitor/gap: CX-11 shows prediction/proba time becomes nontrivial for
wide ensembles (`p=32,h=16,e=100`: ~0.55s per 1000 rows), while
interpretability is already fast after Proposal 5. `_raw_score_chain`
currently loops over learners and each learner recomputes B-spline bases
inside `KANLayer.forward`. With `update_grid=False` during fit and shared
`kan_grid/kan_k`, layer-0 knots are expected to be identical across
learners in a chain.

Hypothesis: an internal ensemble forward path can precompute layer-0
B-spline bases once per input batch/chain, reuse them across learners, and
sum learner outputs exactly. For GAM/identity output this is especially
direct; for non-GAM it can still reuse layer-0 bases before evaluating each
learner's layer-1 basis on its hidden activations.

Scope: prediction/serving speed only; no public API change. Training
behavior and learned parameters must remain unchanged.

Acceptance gate:
- Exact/near-exact parity: regression raw predictions max abs diff <=1e-10,
  classifier probabilities <=1e-10, classes identical.
- Speed: >=15% end-to-end `predict`/`predict_proba` speedup on at least
  two CX-11-style cases, including one wide `p=32,h=16,e=100` case.
- Reject if knot equality does not hold generally, if the optimization
  adds fragile assumptions about saved models, or if speedup is isolated to
  a microbenchmark while full public `predict` remains below +15%.

Protocol: implement in a scratch/branch-local path first, run local parity
where dependencies allow, then validate with the CX-11 Kaggle profile
script modified to compare production vs cached-forward reference.

**[Codex Review, 2026-07-22] CX-13 v1 Kaggle run complete -- promising
speed, parity failed strict gate**: Kernel
`tuamamhamza/cx13-kanboost-predict-forward-cache` version 1 completed.
Outputs were downloaded under
`remote/results/kaggle_cx13_predict_forward_cache/` with prefix
`kaggle_cx13-kanboost-predict-forward-cache_synthetic_predict_cache_20260722-051354`.

Summary:

| case | task | production | cached ref | speedup | max diff |
|---|---:|---:|---:|---:|---:|
| p8 h3 e40 n=2000 | regression | 0.042s | 0.015s | 2.79x | 2.8e-7 |
| p32 h16 e100 n=2000 | regression | 0.459s | 0.203s | 2.26x | 2.8e-7 |
| p8 h3 e40 n=10000 | regression | 0.112s | 0.039s | 2.88x | 3.5e-7 |
| p32 GAM h1 e100 n=10000 | regression | 0.762s | 0.043s | 17.73x | 5.3e-7 |
| p8 h3 e40 n=2000 | classification | 0.042s | 0.015s | 2.81x | 6.5e-9 |
| p32 h16 e100 n=2000 | classification | 0.464s | 0.201s | 2.31x | 8.1e-9 |
| p8 h3 e40 n=10000 | classification | 0.112s | 0.039s | 2.88x | 8.1e-9 |

Decision: do not accept v1 yet despite strong 2.3x-17.7x speedups. The
predeclared parity gate was <=1e-10 for raw/proba values, and v1 misses it
because the cached reference computes learner outputs in numpy float64
while production `DeepKAN.__call__` rounds through a torch float32 tensor
when public predict passes `X_t` as `torch.float32`. Codex patched the
experiment to cast each learner output to float32 before accumulation and
pushed kernel version 2. Treat CX-13 as pending until v2 confirms parity.

**[Codex Review, 2026-07-22] CX-13 v2 Kaggle run complete -- ACCEPT as
prediction-speed implementation candidate**: Kernel
`tuamamhamza/cx13-kanboost-predict-forward-cache` version 2 completed.
Outputs were downloaded under
`remote/results/kaggle_cx13_predict_forward_cache/` with prefix
`kaggle_cx13-kanboost-predict-forward-cache_synthetic_predict_cache_20260722-051631`.
The only semantic change from v1 was to cast each cached learner output to
`float32` before adding it to the ensemble score, matching production
`DeepKAN.__call__` when public prediction passes `torch.float32` input.

Summary:

| case | task | production | cached ref | speedup | max diff |
|---|---:|---:|---:|---:|---:|
| p8 h3 e40 n=2000 | regression | 0.060s | 0.021s | 2.81x | 0.0 |
| p32 h16 e100 n=2000 | regression | 0.681s | 0.308s | 2.21x | 0.0 |
| p8 h3 e40 n=10000 | regression | 0.166s | 0.057s | 2.91x | 0.0 |
| p32 GAM h1 e100 n=10000 | regression | 1.134s | 0.074s | 15.30x | 0.0 |
| p8 h3 e40 n=2000 | classification | 0.059s | 0.022s | 2.76x | 0.0 |
| p32 h16 e100 n=2000 | classification | 0.677s | 0.305s | 2.22x | 0.0 |
| p8 h3 e40 n=10000 | classification | 0.169s | 0.057s | 2.97x | 0.0 |

Decision: accept CX-13 v2 as the next production speed candidate. It
passes the parity and >=15% end-to-end prediction-speed gates by a wide
margin, and all tested chains had equal layer-0 knots. Production
implementation requirements for Claude/Codex:
- Guard the optimized path behind a layer-0 knot equality check; fall back
  to the existing `_raw_score_chain` if any learner differs.
- Preserve the float32 per-learner rounding invariant before adding each
  learner's contribution.
- Cover regression, binary classification, multiclass fallback behavior,
  GAM identity output, saved/loaded models, and early-stopped
  `best_iteration`.
- Rerun the CX-13 v2 benchmark after implementation and require exact
  public-output parity plus >=15% speedup on at least two cases.

**[Codex Implementation Package for Claude, 2026-07-22] CX-13 production
cached-forward path**

Ownership: Claude Code is the primary implementer under `AGENTS.md`.
Codex should review the implementation and rerun/compare evidence unless
the user explicitly assigns Codex to edit core code.

Target files likely involved:
- `kanboost/core/base.py`: add an optimized internal path used by
  `_raw_score_chain` when safe.
- `kanboost/core/kan/layer.py` or `kanboost/core/kan/network.py`: only if
  a small helper is needed to evaluate layer-0 from precomputed bases.
- Tests under the repo's existing test layout, if present; otherwise a
  focused new regression/parity test file.

Implementation sketch:
1. In `_raw_score_chain`, before the learner loop, check whether active
   learners are non-empty and every active learner has identical layer-0
   `n_in`, `n_out`, `k`, and `knots`.
2. If the check fails, keep the current slow loop exactly.
3. If it passes, transform `X_t` to numpy once, precompute
   `_b_basis_1d(X[:, j], knots[j], k)` for every layer-0 feature once per
   public prediction call.
4. For each learner, build its hidden activation `z` by multiplying the
   cached basis list by that learner's layer-0 coefficients.
5. If `learner._output_identity` is true, output `z.sum(axis=1) +
   learner._intercept`; otherwise call the existing layer-1 forward on
   `z`.
6. Cast each learner output to `np.float32` before adding
   `learning_rate * out` to the ensemble score; this is required for exact
   parity with production `DeepKAN.__call__` and is the difference between
   CX-13 v1 failing parity and CX-13 v2 passing.
7. Preserve `best_iteration` slicing exactly and avoid changing any public
   API, model state, training behavior, serialization format, or
   multiclass semantics.

Minimum tests/gates before promotion:
- Regression public `predict`: max abs diff exactly 0.0 or <=1e-10 vs old
  path on p8/h3/e40 and p32/h16/e100.
- Binary classifier public `predict_proba`: max abs diff <=1e-10 and
  `predict` classes identical.
- GAM identity output path.
- Early-stopped model where `best_iteration < len(learners)`.
- Save/load round-trip still predicts identically.
- Fallback path: manually perturb one learner's layer-0 knots and verify
  the old path is used or parity still holds.
- Re-run `remote/kaggle_cx13_predict_forward_cache` or a production
  version of it after implementation and require >=15% public prediction
  speedup on at least two cases.

Codex review notes after re-reading current core:
- The current production hot path is exactly
  `kanboost/core/base.py:_raw_score_chain`, which loops over active
  learners and calls `learner(X_t)` for each; `KANLayer.forward` then
  recomputes `_b_basis_1d` for every feature/learner.
- A minimal implementation should live in `_BaseKANBoost` rather than
  changing public `KANLayer.forward` semantics. This keeps the optimization
  internal to ensemble prediction and reduces surface-area risk.
- Be careful with multiclass: `_raw_score_chain` is called once per
  one-vs-rest chain, so the cache should be per-chain/per-call, not shared
  globally across classes unless keyed by the exact active learner list and
  transformed input.
- Do not cache across public prediction calls unless invalidation is
  solved; CX-13's evidence is for per-call reuse only.

**[Codex Proposal CX-14, 2026-07-22] Narrow refinement around the CX-12
OpenNeuro winner**

Competitor/gap: CX-12 produced the first accepted OpenNeuro accuracy win:
KANBoost derived-feature select80 h3/e40/s8 reached 0.765 balanced
accuracy. It beats raw HistGBDT but narrowly misses the stronger 0.769
target, and we have not yet checked whether HistGBDT also benefits from
the same EEG-derived feature space.

Hypothesis: the winning point is sensitive to selected feature count and
moderate estimator/step regularity, not broad hidden width. A narrow grid
around k=80 and h3 should either push KANBoost past 0.769 or show that the
current win is the local optimum. A derived-feature HistGBDT baseline
checks whether the feature engineering helped KANBoost specifically or
simply improved every model.

Scope: model-quality benchmark only; no production code/API change.

Candidate grid:
- Feature space: raw+EEG-derived from CX-12.
- `SelectKBest(f_classif)` k in {60, 70, 80, 90, 100}.
- KANBoost configs: h3/e40/s8 (winner), h3/e60/s8, h3/e80/s6, h4/e40/s6.
- Baselines: raw HistGBDT, derived-feature HistGBDT.

Acceptance gate:
- Accept as stronger accuracy frontier if any KANBoost arm reaches >=0.769
  balanced accuracy with log loss <=0.598, or beats the best derived
  HistGBDT balanced accuracy while keeping log loss at least 0.05 lower.
- Reject if the best arm remains <=0.765 or if derived HistGBDT becomes
  the true leader.

Protocol: same Kaggle dataset, folds, label-free derived features, and
fold-local preprocessing/selection as CX-12.

**[Codex Review, 2026-07-22] CX-14 implementation pushed to Kaggle**:
Codex prepared
`remote/kaggle_cx14_openneuro_eeg_refine/cx14_openneuro_eeg_refine.py`
and `remote/kaggle_cx14_openneuro_eeg_refine/kernel-metadata.json`.
Local `py_compile` passed. Kernel `tuamamhamza/cx14-openneuro-eeg-refine`
version 1 was pushed successfully. Status/results pending.

**[Codex Review, 2026-07-22] CX-14 Kaggle run complete -- ACCEPT as new
accuracy frontier**: Kernel `tuamamhamza/cx14-openneuro-eeg-refine`
completed. Outputs were downloaded under
`remote/results/kaggle_cx14_openneuro_eeg_refine/` with prefix
`kaggle_cx14-openneuro-eeg-refine_ds004504_large_clean_features_eeg_refine_20260722-050311`.

Top summary:

| model | balanced accuracy | macro F1 | log loss | ROC AUC | mean fit |
|---|---:|---:|---:|---:|---:|
| `kanboost_eeg_select80_h3_e80_s6` | 0.777 | 0.773 | 0.568 | 0.785 | 0.906s |
| `kanboost_eeg_select80_h3_e60_s8` | 0.765 | 0.758 | 0.576 | 0.785 | 0.791s |
| `kanboost_eeg_select80_h3_e40_s8` | 0.765 | 0.758 | 0.595 | 0.780 | 0.553s |
| `kanboost_eeg_select70_h3_e80_s6` | 0.763 | 0.758 | 0.570 | 0.803 | 0.839s |
| `hist_gbdt_raw` | 0.749 | 0.751 | 0.712 | 0.770 | 0.137s |
| `hist_gbdt_eeg_derived` | 0.667 | 0.661 | 0.702 | 0.779 | 0.166s |

Decision: accept CX-14 under the stronger predefined gate. The winning
KANBoost arm reaches >=0.769 balanced accuracy and improves log loss by
0.145 absolute vs raw HistGBDT. It also shows that the EEG-derived feature
space is not a universal tree-model shortcut: HistGBDT on raw+derived
features drops to 0.667 balanced accuracy, while KANBoost rises to 0.777.
This supports the hypothesis that the derived smooth EEG coordinates are
especially compatible with KANBoost's spline learners. Remaining
uncertainty: n is small and fold variance is high (`std_balanced_accuracy`
0.115), so this should be followed by repeated-seed stability evidence
before claiming a robust final benchmark frontier.

**[Codex Proposal CX-15, 2026-07-22] Repeated-seed stability check for the
CX-14 OpenNeuro winner**

Competitor/gap: CX-14 passes the accuracy target, but OpenNeuro large
AD-vs-Control has small n and high fold variance. A single
`StratifiedKFold(random_state=42)` result can overstate a frontier if one
split is favorable.

Hypothesis: if the CX-14 improvement is real, the winning
`kanboost_eeg_select80_h3_e80_s6` arm should remain above raw HistGBDT on
mean balanced accuracy and log loss across repeated CV seeds, not only one
split. The exact mean may drop, but the ranking and calibration advantage
should persist.

Scope: benchmark confidence only; no production code/API change.

Protocol:
- Use the same raw+EEG-derived feature builder from CX-14.
- Compare exactly three arms: raw HistGBDT, derived HistGBDT, and
  `kanboost_eeg_select80_h3_e80_s6`.
- Run 5-fold `StratifiedKFold` across seeds {11, 22, 33, 44, 55}; aggregate
  25 validation folds.
- Keep imputation/variance filtering/selection inside each fold pipeline.

Acceptance gate:
- Accept stability if KANBoost's repeated-seed mean balanced accuracy is
  >= raw HistGBDT by >=0.01 and mean log loss is <= raw HistGBDT by >=0.05.
- Mark inconclusive if KANBoost remains better in log loss but balanced
  accuracy lead shrinks below 0.01.
- Reject stability if raw or derived HistGBDT overtakes KANBoost on
  repeated mean balanced accuracy.

**[Codex Review, 2026-07-22] CX-15 implementation pushed to Kaggle**:
Codex prepared
`remote/kaggle_cx15_openneuro_eeg_stability/cx15_openneuro_eeg_stability.py`
and `remote/kaggle_cx15_openneuro_eeg_stability/kernel-metadata.json`.
Local `py_compile` passed. Kernel
`tuamamhamza/cx15-openneuro-eeg-stability` version 1 was pushed
successfully. Status/results pending.

**[Codex Review, 2026-07-22] CX-15 Kaggle run complete -- ACCEPT
stability with important caveat**: Kernel
`tuamamhamza/cx15-openneuro-eeg-stability` completed. Outputs were
downloaded under `remote/results/kaggle_cx15_openneuro_eeg_stability/`
with prefix
`kaggle_cx15-openneuro-eeg-stability_ds004504_large_clean_features_eeg_stability_20260722-051114`.

Repeated 5-fold CV over seeds {11, 22, 33, 44, 55}, 25 folds total:

| model | balanced accuracy | macro F1 | log loss | ROC AUC | folds |
|---|---:|---:|---:|---:|---:|
| `kanboost_eeg_select80_h3_e80_s6` | 0.689 | 0.681 | 0.576 | 0.781 | 25 |
| `hist_gbdt_raw` | 0.676 | 0.667 | 0.773 | 0.731 | 25 |
| `hist_gbdt_eeg_derived` | 0.633 | 0.621 | 0.819 | 0.720 | 25 |

Seed-level balanced accuracy:

| seed | KANBoost | raw HistGBDT | derived HistGBDT |
|---:|---:|---:|---:|
| 11 | 0.709 | 0.663 | 0.576 |
| 22 | 0.751 | 0.649 | 0.683 |
| 33 | 0.696 | 0.686 | 0.635 |
| 44 | 0.610 | 0.701 | 0.635 |
| 55 | 0.677 | 0.679 | 0.634 |

Decision: accept CX-15 under the predefined stability gate, but with a
strong caveat. KANBoost's repeated mean balanced accuracy leads raw
HistGBDT by +0.013 and log loss by -0.198, satisfying the acceptance gate.
However, the seed-level table shows the 0.777 CX-14 single-split result is
split-sensitive: KANBoost loses balanced accuracy to raw HistGBDT on seed
44 and essentially ties on seed 55. The robust claim should be:
EEG-derived KANBoost improves calibration strongly and has a small repeated
balanced-accuracy lead, while the high 0.777 number is a best-split
frontier that needs either more data, repeated-CV model selection, or a
more stable feature protocol before being treated as the definitive final
accuracy.

**[Codex Proposal CX-16, 2026-07-22] Fold-local stability selection for
OpenNeuro EEG-derived features**

Competitor/gap: CX-15 shows KANBoost keeps a repeated-CV advantage, but
the balanced-accuracy lead is small and split-sensitive. The likely weak
point is unstable feature selection in a tiny-n/high-p table: ordinary
`SelectKBest(f_classif, k=80)` can choose different noisy EEG-derived
features depending on one fold's labels.

Hypothesis: replacing one-shot `SelectKBest` with fold-local stability
selection should reduce split sensitivity. Within each outer training
fold only, repeatedly rank features on inner stratified splits using
ANOVA F statistics after imputation/variance filtering, average ranks
or selection frequency, then train KANBoost on the most stable 80
features. This keeps selection label-aware but leakage-safe because the
held-out outer fold is never used in ranking.

Scope: model-quality robustness on OpenNeuro; no production code/API
change.

Protocol:
- Same raw+EEG-derived feature builder as CX-14/CX-15.
- Repeated outer 5-fold CV seeds {11, 22, 33, 44, 55}.
- Compare raw HistGBDT, derived HistGBDT, current
  `SelectKBest80 + KANBoost h3/e80/s6`, and
  `StabilitySelect80 + KANBoost h3/e80/s6`.
- All imputation, variance filtering, and stability ranking fit only on
  each outer training fold.

Acceptance gate:
- Accept if stability selection improves repeated mean balanced accuracy
  by >=0.02 over CX-15's ordinary KANBoost 0.689 while keeping log loss
  <=0.58.
- Mark inconclusive if log loss remains strong but balanced accuracy gain
  is <0.02.
- Reject if stability selection underperforms ordinary SelectKBest or
  raw HistGBDT on repeated mean balanced accuracy.

**[Codex Review, 2026-07-22] CX-16 implementation pushed to Kaggle**:
Codex prepared
`remote/kaggle_cx16_openneuro_stability_select/cx16_openneuro_stability_select.py`
and `remote/kaggle_cx16_openneuro_stability_select/kernel-metadata.json`.
Local `py_compile` passed. Kernel
`tuamamhamza/cx16-openneuro-stability-select` version 1 was pushed
successfully. Status/results pending.

**[Codex Review, 2026-07-22] CX-16 Kaggle run complete --
INCONCLUSIVE/PARTIAL improvement**: Kernel
`tuamamhamza/cx16-openneuro-stability-select` completed. Outputs were
downloaded under `remote/results/kaggle_cx16_openneuro_stability_select/`
with prefix
`kaggle_cx16-openneuro-stability-select_ds004504_large_clean_features_eeg_stability_select_20260722-052355`.

Repeated 5-fold CV over seeds {11, 22, 33, 44, 55}, 25 folds total:

| model | balanced accuracy | macro F1 | log loss | ROC AUC | folds |
|---|---:|---:|---:|---:|---:|
| `kanboost_stability80_h3_e80_s6` | 0.696 | 0.689 | 0.571 | 0.791 | 25 |
| `kanboost_select80_h3_e80_s6` | 0.689 | 0.681 | 0.576 | 0.781 | 25 |
| `hist_gbdt_raw` | 0.676 | 0.667 | 0.773 | 0.731 | 25 |
| `hist_gbdt_eeg_derived` | 0.633 | 0.621 | 0.819 | 0.720 | 25 |

Decision: mark CX-16 inconclusive/partial, not accepted as a new robust
accuracy fix. Stability selection improves KANBoost repeated mean balanced
accuracy by +0.008 and log loss by -0.005 over ordinary SelectKBest, and
it improves the weak seed-44 case from 0.610 to 0.648. But the predefined
acceptance gate required +0.02 balanced-accuracy improvement over CX-15's
0.689, so it does not clear the bar. The useful finding is directional:
fold-local stable feature selection reduces some split sensitivity and
preserves calibration, but a stronger protocol is needed before claiming
another accuracy frontier.

**[Codex Proposal CX-17, 2026-07-22] Leakage-safe inner-CV thresholding for
EEG-derived KANBoost probabilities**

Competitor/gap: CX-15/CX-16 show KANBoost's probabilities are much better
calibrated than HistGBDT on repeated OpenNeuro CV, but balanced accuracy
lead remains small and split-sensitive. The public classifier uses a fixed
0.5 threshold; for tiny, noisy biomedical data this may leave balanced
accuracy on the table even when ranking/calibration is useful.

Hypothesis: tuning a scalar decision threshold using inner-CV
out-of-fold probabilities inside each outer training fold can improve
balanced accuracy without changing fitted probabilities or using outer
validation labels. This revisits thresholding only because the feature
space/model changed materially after CX-12/CX-14; older Proposal 7 on the
raw/project arm remains rejected.

Scope: benchmark decision rule only; no production default threshold or
API change.

Protocol:
- Same raw+EEG-derived feature builder and repeated outer seeds
  {11, 22, 33, 44, 55}.
- Compare raw HistGBDT with 0.5 threshold, KANBoost SelectKBest80 at 0.5,
  KANBoost SelectKBest80 with inner-CV threshold, KANBoost
  StabilitySelect80 at 0.5, and KANBoost StabilitySelect80 with inner-CV
  threshold.
- For each outer fold and thresholded KANBoost arm, generate inner
  3-fold OOF probabilities on the outer-training split only, choose the
  threshold from grid {0.20, 0.225, ..., 0.80} maximizing balanced
  accuracy with macro-F1 tie break, refit on the full outer train, then
  apply that threshold once to the outer validation fold.
- Log loss/ROC AUC are computed from probabilities and should not change
  except for refit randomness already controlled by seed.

Acceptance gate:
- Accept if either thresholded KANBoost arm reaches repeated mean balanced
  accuracy >=0.716 (>= +0.02 over CX-16's 0.696 best) while keeping mean
  log loss <=0.58.
- Mark inconclusive if thresholding improves balanced accuracy by
  0.005-0.02 without hurting log loss.
- Reject if thresholded arms do not beat their corresponding 0.5-threshold
  arms or if gains depend on any outer validation label leakage.

**[Codex Review, 2026-07-22] CX-17 implementation pushed to Kaggle**:
Codex prepared
`remote/kaggle_cx17_openneuro_inner_threshold/cx17_openneuro_inner_threshold.py`
and `remote/kaggle_cx17_openneuro_inner_threshold/kernel-metadata.json`.
Local `py_compile` passed. Kernel
`tuamamhamza/cx17-openneuro-inner-threshold` version 1 was pushed
successfully. Status/results pending.

**[Codex Review, 2026-07-22] CX-17 Kaggle run complete --
INCONCLUSIVE, best repeated-CV BA so far but misses gate**: Kernel
`tuamamhamza/cx17-openneuro-inner-threshold` completed. Outputs were
downloaded under `remote/results/kaggle_cx17_openneuro_inner_threshold/`
with prefix
`kaggle_cx17-openneuro-inner-threshold_ds004504_large_clean_features_eeg_inner_threshold_20260722-053116`.

Repeated 5-fold CV over seeds {11, 22, 33, 44, 55}, 25 folds total:

| model | balanced accuracy | macro F1 | log loss | ROC AUC | mean threshold |
|---|---:|---:|---:|---:|---:|
| `kanboost_select80_inner_threshold` | 0.714 | 0.705 | 0.576 | 0.781 | 0.465 |
| `kanboost_stability80_t0p5` | 0.696 | 0.689 | 0.571 | 0.791 | 0.500 |
| `kanboost_select80_t0p5` | 0.689 | 0.681 | 0.576 | 0.781 | 0.500 |
| `kanboost_stability80_inner_threshold` | 0.686 | 0.676 | 0.571 | 0.791 | 0.464 |
| `hist_gbdt_raw_t0p5` | 0.676 | 0.667 | 0.773 | 0.731 | 0.500 |

Decision: mark CX-17 inconclusive/high-value, not accepted. The best arm
improves repeated balanced accuracy from CX-15's ordinary SelectKBest
0.689 to 0.714 and keeps log loss at 0.576, but the predefined acceptance
gate was >=0.716, so it narrowly misses by 0.002. It does beat raw HistGBDT
by +0.039 balanced accuracy and -0.198 log loss. The thresholded SelectKBest
arm improves every seed relative to its 0.5 counterpart, including the weak
seed-44 split (0.610 -> 0.659), so the signal is real enough to justify a
small follow-up. StabilitySelect plus thresholding performed worse than
StabilitySelect at 0.5, so do not combine those two as-is.

**[Codex Proposal CX-18, 2026-07-22] Fine inner-threshold refinement for
the CX-17 SelectKBest arm**

Competitor/gap: CX-17 missed the repeated-CV acceptance gate by only
0.002 balanced accuracy. Its selected thresholds cluster below 0.5
(mean 0.465), and every seed improved over the 0.5 SelectKBest baseline.
The failure mode is not lack of signal; it is whether the threshold
selection rule is too coarse/noisy for tiny inner folds.

Hypothesis: a narrower threshold grid and a stability-aware threshold
selector can recover the missing 0.002-0.01 repeated balanced accuracy
without affecting probabilities/log loss. Two candidates are worth a
small test: (a) finer grid 0.35-0.60 step 0.01, and (b) choose the median
of per-inner-fold best thresholds rather than the single OOF global best,
reducing sensitivity to one inner split.

Scope: benchmark decision-rule robustness only; no production default
threshold/API change.

Protocol:
- Test only the winning `SelectKBest80 + KANBoost h3/e80/s6` arm plus raw
  HistGBDT baseline.
- Same repeated outer seeds {11, 22, 33, 44, 55}.
- Compare: 0.5 threshold, CX-17 OOF global threshold with fine grid, and
  median-of-inner-fold-best threshold with fine grid.

Acceptance gate:
- Accept if either refined threshold rule reaches repeated mean balanced
  accuracy >=0.716 and log loss <=0.58.
- Mark inconclusive if it stays between 0.714 and 0.716.
- Reject if it drops below CX-17's 0.714 or improves only one seed while
  hurting the repeated mean.

**[Codex Review, 2026-07-22] CX-18 implementation pushed to Kaggle**:
Codex prepared
`remote/kaggle_cx18_openneuro_threshold_refine/cx18_openneuro_threshold_refine.py`
and `remote/kaggle_cx18_openneuro_threshold_refine/kernel-metadata.json`.
Local `py_compile` passed. Kernel
`tuamamhamza/cx18-openneuro-threshold-refine` version 1 was pushed
successfully. Multiple Codex status checks initially showed
`KernelWorkerStatus.RUNNING`; results were later downloaded and evaluated
below.

**[Codex Review, 2026-07-22] CX-18 Kaggle run complete -- ACCEPT as best
repeated-CV OpenNeuro accuracy protocol so far**: Kernel
`tuamamhamza/cx18-openneuro-threshold-refine` completed. Outputs were
downloaded under `remote/results/kaggle_cx18_openneuro_threshold_refine/`
with prefix
`kaggle_cx18-openneuro-threshold-refine_ds004504_large_clean_features_eeg_threshold_refine_20260722-053836`.

Repeated 5-fold CV over seeds {11, 22, 33, 44, 55}, 25 folds total:

| model | balanced accuracy | macro F1 | log loss | ROC AUC | mean threshold |
|---|---:|---:|---:|---:|---:|
| `kanboost_select80_inner_global_fine` | 0.718 | 0.711 | 0.576 | 0.781 | 0.486 |
| `kanboost_select80_inner_median_fine` | 0.698 | 0.687 | 0.576 | 0.781 | 0.471 |
| `kanboost_select80_t0p5` | 0.689 | 0.681 | 0.576 | 0.781 | 0.500 |
| `hist_gbdt_raw_t0p5` | 0.676 | 0.667 | 0.773 | 0.731 | 0.500 |

Decision: accept CX-18 under the predefined gate. The fine-grid inner-OOF
global threshold reaches repeated mean balanced accuracy 0.718 (>=0.716)
with log loss 0.576 (<=0.58), improving +0.030 over fixed-threshold
KANBoost and +0.043 over raw HistGBDT while preserving the same
probabilities/calibration. The median-of-inner-fold thresholds are worse,
so the accepted decision rule is specifically the inner-OOF global
fine-grid threshold, not median thresholding. Remaining caveat: this is a
benchmark decision-rule protocol, not a default production classifier
threshold/API change unless the user explicitly approves such an API.

**[Codex Review, 2026-07-22] CX-18 standalone notebook prepared and pushed**:
To make CX-18 reviewable outside the scratch script, Codex generated a
standalone benchmark notebook at
`remote/kaggle_cx18_openneuro_threshold_refine_notebook/openneuro_ds004504_cx18_threshold_benchmark.ipynb`
with metadata in
`remote/kaggle_cx18_openneuro_threshold_refine_notebook/kernel-metadata.json`.
The notebook wraps the accepted CX-18 script with protocol notes and uses
the same Kaggle datasets (`openneuro-ds004504-large-clean-features` and
`kanboost-1-2-4-proposal5-wheel`). Kernel
`tuamamhamza/openneuro-ds004504-cx18-threshold-benchmark` version 1 was
pushed successfully. First Codex status check failed during Kaggle API
token introspection with `RemoteDisconnected`; this is an external
transport/auth-introspection failure after a successful push, not evidence
that the notebook failed. Status/results pending until a later status check
or output download succeeds. This notebook is the recommended artifact for
Claude to merge into the larger OpenNeuro sequential benchmark notebook; it
is benchmark-only and must not alter KANBoost's default classifier
threshold/API.

**[Codex Review, 2026-07-22] CX-18 notebook v1 failed, v2 patched and
pushed**: Kaggle log for
`tuamamhamza/openneuro-ds004504-cx18-threshold-benchmark` v1 showed
`NameError: name '__file__' is not defined` in the notebook environment.
Codex patched the notebook to use
`HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()`
for locating optional local files before falling back to `/kaggle/input`,
then pushed kernel version 2 successfully. Repeated Codex status checks
showed `KernelWorkerStatus.RUNNING`; status/results pending.

**[Codex Review, 2026-07-22] CX-18 notebook v2 complete -- PASS, but goal
remains open**: Kaggle kernel
`tuamamhamza/openneuro-ds004504-cx18-threshold-benchmark` version 2 completed
successfully and Codex downloaded outputs under
`remote/results/kaggle_cx18_notebook/` with prefix
`kaggle_cx18-openneuro-threshold-refine_ds004504_large_clean_features_eeg_threshold_refine_20260722-055234`.

Notebook summary:

| model | balanced accuracy | macro F1 | log loss | ROC AUC | folds |
|---|---:|---:|---:|---:|---:|
| `kanboost_select80_inner_global_fine` | 0.718 | 0.711 | 0.576 | 0.781 | 25 |
| `kanboost_select80_inner_median_fine` | 0.698 | 0.687 | 0.576 | 0.781 | 25 |
| `kanboost_select80_t0p5` | 0.689 | 0.681 | 0.576 | 0.781 | 25 |
| `hist_gbdt_raw_t0p5` | 0.676 | 0.667 | 0.773 | 0.731 | 25 |

Decision: the standalone notebook reproduces the accepted CX-18 result and
is suitable for Claude to fold into the larger OpenNeuro sequential
benchmark/report as a benchmark-only decision-rule cell. It does not close
the user's full goal. Seed-level evidence still shows instability:
`cv_seed=44` has `kanboost_select80_inner_global_fine` balanced accuracy
0.670 versus raw HistGBDT 0.701. Therefore the next accuracy work should
target cross-split robustness and representation, not another broad capacity
grid or a claim that CX-18 is final.

**[Codex Proposal CX-19, 2026-07-22] Split-robust OpenNeuro KANBoost via
fold-local rank aggregation and conservative decision calibration**

Competitor/gap: CX-18 beats raw HistGBDT on repeated mean balanced accuracy
and log loss, but it is not uniformly stronger. The weakest observed split
is `cv_seed=44`, where KANBoost drops to 0.670 BA while HistGBDT reaches
0.701. This is now the largest accuracy weakness: not average probability
quality, but decision robustness under small-n subject splits.

Hypothesis: the current SelectKBest-80 representation is brittle because
small outer folds can choose feature sets that overemphasize unstable EEG
features. A fold-local rank aggregation selector can reduce variance:
inside each outer train fold, repeatedly resample the train subjects,
rank features by several leakage-safe criteria (ANOVA F, mutual
information, absolute logistic coefficient after scaling, and KANBoost
single-feature gain if feasible), aggregate ranks, then train the same
KANBoost configuration on the top-k stable features. Pair it with the
accepted CX-18 inner-OOF global fine threshold. This targets robustness
without changing the KANBoost estimator API.

Scope: model quality / robustness / benchmark protocol on OpenNeuro. Not a
core speed change.

Acceptance gate:
- Primary: repeated 25-fold BA must reach >=0.730, OR improve the worst
  seed BA by >=0.03 while keeping mean BA >=0.718.
- Calibration: mean log loss must stay <=0.590 and ROC AUC must not drop
  by more than 0.02 from CX-18.
- Fairness: all ranking, thresholding, imputation, variance filtering, and
  scaling must be fit only inside each outer train fold or its inner folds.
- Rejection: reject if the improvement appears only in one seed while mean
  BA remains <=0.718, if log loss worsens beyond 0.590, or if runtime grows
  more than 3x without a BA gain of at least +0.02.

Protocol: implement as a features-only Kaggle experiment against
`openneuro-ds004504-large-clean-features`, compare directly to the CX-18
arms using the same seeds `[11, 22, 33, 44, 55]` and 5 outer folds per
seed. Claude local and Codex local should record whether dependencies allow
the full run; Kaggle is the transferability gate.

**[Codex Proposal CX-20, 2026-07-22] Production prediction forward-cache
implementation and validation for CX-13**

Competitor/gap: after Proposal 2 (GAM training cache) and Proposal 5
(interpretability cache), the largest proven speed gap is ordinary
`predict` / `predict_proba` runtime. CX-13 v2 shows exact-parity reference
speedups of 2.2x-3.0x for non-GAM prediction and 15.3x for a wide GAM
prediction case on Kaggle, but this is not yet integrated into production
core code.

Hypothesis: implement an internal ensemble-level prediction helper that
precomputes layer-0 B-spline bases once per predict call and reuses them
across all learners whose layer-0 knots match. Preserve the public API and
fall back to the existing learner-by-learner path if knots differ. The
critical numerical invariant from CX-13 v2 is to cast each learner output
to `np.float32` before adding `learning_rate * out` to the running score.

Scope: prediction speed and benchmark/runtime quality. No training-quality
or public-API change.

Acceptance gate:
- Exact public-output parity on regression predictions and classification
  probabilities/classes. Target max absolute difference: 0.0 where feasible,
  otherwise <=1e-10 with a documented float-path reason.
- Must preserve `best_iteration`, early-stopped ensembles, binary and
  multiclass behavior, save/load behavior, and fallback behavior when
  layer-0 knots differ.
- Speed gate: >=2x faster on at least one non-GAM medium/large prediction
  benchmark and no more than 5% slower on any tested prediction case.
- Double-experiment evidence: Claude local test, Codex local test if
  dependencies allow, then Kaggle/Colab rerun using the same benchmark
  shapes as CX-13 v2. Until this passes, CX-13 remains an accepted
  reference/proposal, not a production speed feature.

Ownership: Claude Code is still the primary implementer under `AGENTS.md`;
Codex should review the patch and may prepare independent validation
scripts/notebooks. Codex should not edit `kanboost/core/` unless the user
explicitly changes Codex's role for this implementation pass.

**[Codex Synthesis, 2026-07-22] Checkpoint: what can and cannot be achieved
from evidence so far**:
- **Ready for user trial / release-candidate packaging**: Proposal 5 as
  `kanboost 1.2.4` wheel. This is the clearest user-facing win: it does
  not change model training or predictions, only repeated interpretability
  calls. External Kaggle validation of the exact wheel shows
  `feature_contributions` 26.6x-87.7x faster and GAM
  `predict_derivative` 5.5x-10.8x faster with parity from 0 to ~3e-15.
  Best next action is user/Claude Colab rerun with
  `dist/kanboost-1.2.4-py3-none-any.whl`, then explicit commit/push/release
  approval if the user accepts the artifact.
- **Ready for merge review, not more speed proof**: Proposal 2 GAM training
  cache. It has exact parity, large local speedups, and larger Kaggle
  speedups. It should be reviewed for code simplicity and invariants, not
  re-benchmarked again unless its implementation changes.
- **Scientifically encouraging but not a training-speed claim**: Proposal 6
  OpenNeuro. The full pinned Kaggle rerun closes the predefined large-stage
  quality gap to HistGBDT (0.047 <= 0.05) and ties medium-stage best
  balanced accuracy. KANBoost still loses raw large-stage balanced accuracy
  to HistGBDT and RBF-KAN, but keeps better log loss than HistGBDT and far
  better log loss than RBF-KAN. Claim this as "competitive quality with
  better calibration on this protocol", not as speed superiority or a
  universal OpenNeuro win.
- **Do not spend more cycles without a new bottleneck profile**: Proposal 1
  batched ALS, Proposal 3 non-GAM layer0 prediction cache, Proposal 7
  threshold tuning, broad C++ backend, Cholesky-for-eigh, default
  cross-fold parallelism on this 8 GB machine, and depth-3/bottleneck
  reshaping. Each has a recorded failure mode: hardware fragility,
  insufficient speedup, accuracy harm, numerical risk, memory pressure, or
  weak generalization.
- **Main remaining uncertainty**: the double-experiment ideal is still not
  fully satisfied for Proposal 6 because Codex local cannot run the full ML
  stack here (`sklearn` missing) and Claude local evidence for the exact
  pinned OpenNeuro notebook is not recorded in this snapshot. That
  limitation should be reported honestly; it does not erase the Kaggle
  evidence, but it prevents calling Proposal 6 a fully local+remote-closed
  package.

**[Codex Review, 2026-07-22] Proposal 6 pinned Kaggle rerun submitted**:
Codex created an isolated Kaggle kernel copy so it would not clobber the
shared `openneuro-ds004504-kanboost-clean-benchmark` resource:
`tuamamhamza/openneuro-ds004504-kanboost-clean-pinned-codex`, source under
`remote/kaggle_openneuro_clean_pinned_codex/`. It runs the hardened
OpenNeuro notebook with pinned KANBoost commit
`11f4b0fd95ac40aa475702a81d5efbb749cee55b`, fold-local cleaning, raw EEG
cache under `/kaggle/temp`, and timestamped outputs under
`/kaggle/working/outputs`.

Local Codex side for this exact protocol is currently skipped, not
successful: the bundled local Python does not have `sklearn` installed
(`ModuleNotFoundError: No module named 'sklearn'`). Per the
double-experiment rule, do not count this as a complete local-vs-remote
package.

**[Codex Review, 2026-07-22] Proposal 6 pinned Kaggle rerun complete**:
Kernel `tuamamhamza/openneuro-ds004504-kanboost-clean-pinned-codex`
completed. Outputs were downloaded to
`remote/results/kaggle_openneuro_clean_pinned_codex/outputs/` with prefix
`kaggle_openneuro_ds004504_model_benchmark_ds004504_v1_0_7_eeg_20260721-231436`.
Manifest confirms `kanboost_commit =
11f4b0fd95ac40aa475702a81d5efbb749cee55b`, `platform = kaggle`, all stages
completed (`small`, `medium`, `large`), and raw EEG did not appear in the
downloaded output file list.

Pinned full-notebook stage summary:

| stage | best baseline | best baseline bal acc | best KANBoost arm | KANBoost bal acc | gap | KANBoost log loss | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| small | `catboost_clean` | 0.750 | `kanboost_gam_cached_clean` / `kanboost_gam_select20_clean` | 0.625 | -0.125 | 0.667 / 0.662 | gap open at tiny n |
| medium | `catboost_clean` / `xgboost_clean` / `kanboost_gam_cached_clean` | 0.800 | `kanboost_gam_cached_clean` | 0.800 | 0.000 | 0.568 | gap closed |
| large | `hist_gbdt_clean` | 0.749 | `kanboost_project_clean` | 0.702 | -0.047 | 0.617 | within <=0.05 gate |

Large-stage details:
- `hist_gbdt_clean`: balanced accuracy 0.749, macro F1 0.751, log loss
  0.712, ROC AUC 0.770, mean fit 0.138s.
- `kanboost_project_clean`: balanced accuracy 0.702, macro F1 0.691, log
  loss 0.617, ROC AUC 0.747, mean fit 0.685s.
- `rbf_kan_torch_clean`: balanced accuracy 0.732 but log loss 1.494, so it
  beats KANBoost on balanced accuracy but is much worse calibrated.

Decision: Proposal 6's large-stage predefined quality gap is now closed by
the full pinned Kaggle notebook (0.047 <= 0.05) and medium-stage is tied,
but the claim remains "remote-confirmed / local-incomplete" rather than a
complete double-experiment package because Codex local could not run and no
Claude local result for this exact pinned protocol is recorded here.

**Branch**: `research/depth3-kan`. **Production code touched**: none
(`kanboost/core/kan/network.py`'s only diff is a pre-existing, unrelated
documentation comment from an earlier, already-committed-adjacent session;
not part of this research work).

**Files this research work owns** (all scratchpad, paths relative to
`.../scratchpad/`):
- `depth3_prototype.py` -- standalone depth-3 KAN (n_in->H1->H2->1), 3-block
  Gauss-Seidel ALS with rollback. Fixed once (see Round 1 below): the
  linearized-step denominator was summing over all samples instead of
  per-sample, which starved the middle layer of training.
- `depth2_prototype.py` -- standalone depth-2 KAN using the *identical*
  fixed step formula, built specifically so depth-3 vs depth-2 comparisons
  aren't confounded by an optimizer difference.
- `depth_experiment_v2.py` -- Round 1 CV harness (joint ALS, parameter- and
  seed-matched depth-3 vs depth-2).
- `depth_residual_prototype.py` -- Round 2: `ResidualDepth3Correction`
  (trained on the residual of a frozen, pretrained `DeepKAN2`), block order
  output->middle->input, **per-block** (not per-sweep) loss check with
  rollback. `ResidualModel` composes `F(x) = F_base(x) + dF(x)`.
- `depth_experiment_residual.py` -- Round 2 CV harness.
- `depth_experiment_residual_v2.py` -- Round 2 CV harness at the
  ~12,500-param scale (A config).
- `depth_experiment_bottleneck.py` -- B (uncached bottleneck) vs C (cached
  bottleneck) CV harness.

**PRIORITY REVIEW ASK (user-requested, review this specifically before
closing the path)**: the `cache_basis` logic in
`depth_residual_prototype.py` -- the reuse-validity reasoning at the input
block (lines ~194-208 in the current version): `z1c`/`B1` are reused
because C0 (which determines z1) is untouched between the top of a round
and the input block itself; `z2` is still recomputed fresh via
`B1 @ (updated C1)` since C1 changed in the middle block just before.
Please verify this reasoning directly against the code (not just the
comment), and check `verify_cache_equivalence()`'s 5 gates are sufficient
to catch a violation if the reasoning were wrong (same-rounds-executed,
rollback-flag-identity, per-round relative loss diff <=1e-9, per-round
parameter diff <1e-8, final-prediction diff <1e-10). This is currently a
scratchpad-only flag (`cache_basis=False` default), NOT wired into any
public API -- confirm it stays that way pending your review.

**Other review points**:
1. `depth3_prototype.py` / `depth2_prototype.py`: is `_row_normalized_step`
   (per-sample Gauss-Newton with LM damping) correctly derived and
   correctly applied identically on both sides of the comparison?
2. `depth_residual_prototype.py`: is the per-block rollback logic in
   `ResidualDepth3Correction.fit` actually monotone (each block's `self.C*`
   only commits if `full_loss()` did not increase)? Is composing
   `F_base + dF` on frozen-base residuals implemented without leakage
   (base is fit on `X_tr,y_tr` only, never sees `X_va`)?
3. All harnesses: seed/fold parity between arms (same `KFold(random_state=
   42)`, same per-fold `seed=fold` passed to both models) -- this is what
   makes the paired-diff statistics valid.
4. Any silent numerical issue eigh's `rel_floor=1e-4` clipping could be
   masking, beyond what `cond`/`n_floored` already surface.
5. Known, deliberately-unfixed latent bug (Fable's finding, not part of
   this round's scope): `_clamp_to_span` clamps every column to column 0's
   knot span instead of each column's own span. Affects both arms of every
   comparison identically, so it doesn't invalidate any A/B/C result above,
   but flag it as a real, separate bug worth its own fix + parity check.

**Evidence produced so far**: see "Round 1", "Round 2", and "Round 2
refinement" sections below. Headline: Round 1 (joint from-scratch 3-block
ALS) does NOT hold up at scale and is NOT recommended for promotion.
**Round 2 (residual correction, H1=28/H2=25) passes decisively at every
scale tested (2,520 / 3,096 / 12,390 params) and is the adopted
architecture.** A follow-up bottleneck-reshaping experiment (H1=66/H2=8)
was rejected (no accuracy gain, worse stability); its `cache_basis`
optimization is numerically proven safe (6/6 equivalence tests, at-scale
B-vs-C diff ~1e-15) but gives only a 3-11% speedup, kept scratchpad-only.
Profiling shows the frozen depth-2 base dominates wall-clock (68% of a
9.5s fit) -- the correction itself is only 25%, which caps the payoff of
any further correction-side optimizer work per Amdahl's law. Next queued
work (adaptive backtracking, dynamic middle-layer damping, internal early
stopping) targets stability on the ALREADY-ADOPTED shape, not a new
architecture search, and is paused pending Codex's cache_basis review.

**Known limitations / not yet done**: no literal depth-2-weight-splitting
init for the Round 2 correction (documented simplification); no C++ port
(explicitly gated on Python proof-of-benefit per the original plan).

---

## [Claude Code] New proposal from competitive-gap research

Per the user's request to research gaps vs other models (CatBoost,
XGBoost, LightGBM) and vs KANBoost's own strengths, tied to `AGENTS.md`'s
Goal (item 2: "study the best competing model... state what gap KANBoost
is trying to close").

**Researched gaps** (web search, sourced): (1) CatBoost's native
categorical handling via "ordered" target statistics -- only uses prior
rows in a random permutation to compute each category's target-mean,
specifically to prevent target leakage, per [CatBoost's own docs](https://catboost.ai/docs/en/features/categorical-features)
and [GeeksforGeeks' explainer](https://www.geeksforgeeks.org/machine-learning/catboosts-categorical-encoding-one-hot-vs-target-encoding/).
(2) XGBoost/LightGBM's native missing-value handling learns a per-split
default direction rather than a single global imputed value, per
[XGBoost's own paper](https://arxiv.org/pdf/1603.02754) and its
sparsity-aware split-finding algorithm. (3) GPU acceleration is standard
across all three competitors (CatBoost reported up to 40x GPU speedup);
KANBoost has zero GPU path (confirmed earlier this session:
`torch.cuda.is_available()` is False on this machine and the actual
math is pure NumPy/SciPy CPU-bound regardless).

### Proposal CC-6 -- fix categorical target-encoding leakage (model quality / generalization gap vs CatBoost)

**[Retag note, Claude Code, 2026-07-22]**: retagged from "Proposal 6" to
**CC-6** to disambiguate from Codex's unrelated, independently-numbered
"Proposal 6" (OpenNeuro quality gap, now retagged **CX-6**). See "##
Proposal naming convention" below.

**Hypothesis**: `TabularPreprocessor`'s categorical encoding (smoothed
target-mean, `kanboost/core/encoders.py`) computes each category's
target-mean using ALL rows (including the row being encoded) in one
`fit()` pass -- unlike CatBoost's "ordered" scheme, this is
self-referential: a row's own `y` value contributes to its own encoded
feature value. Smoothing reduces but does not eliminate this. This should
manifest as spurious in-sample "signal" on high-cardinality categorical
columns that does not hold out-of-fold, i.e. real target leakage.

**[Claude Code] Confirmed empirically** (`proposal6_categorical_leakage.py`):
a purely-noise categorical column (independent random assignment, zero
true relationship to `y`) shows INCREASING spurious in-sample correlation
with `y` as cardinality grows -- exactly where an ordered scheme matters
most:

| n_categories | in-sample (leaking) corr | out-of-fold (leakage-free) corr |
|---|---|---|
| 10 | +0.088 | +0.032 |
| 30 | +0.107 | -0.030 |
| 50 | +0.178 | +0.023 |
| 100 | **+0.215** | -0.012 |

The out-of-fold version (fit the encoder per-CV-fold on training rows
only, encode the held-out fold with that mapping -- a K-fold
approximation of CatBoost's permutation-based ordered scheme) stays near
zero regardless of cardinality, as it correctly should for pure noise.
This is a genuine, real, sourced-and-measured gap, not a theoretical
concern.

**Proposed fix**: replace (or add as an option alongside) the current
single-pass global target-mean encoding with an ordered/out-of-fold
scheme -- e.g., K-fold target encoding during `fit()` (each row encoded
using only OTHER folds' statistics) or a permutation-based ordered scheme
matching CatBoost's own approach more closely. Ties to Goal priority 1
(preserve/improve validation quality) and priority 2 (close a specific,
named competitive gap).

**Acceptance criteria (proposed, for discussion)**:
- On the synthetic noise-categorical test above: out-of-fold-style
  encoding must not show correlation growing with cardinality (should
  stay statistically indistinguishable from zero at every tested
  cardinality).
- On at least one real dataset with genuine categorical columns and
  meaningful cardinality: validation R2/accuracy does not regress, and
  the train/validation gap attributable to categorical columns
  measurably shrinks vs the current encoding.
- No change to numeric-column handling (missing-value imputation +
  indicator columns), scope is categorical encoding only.
- Must not silently change `TabularPreprocessor`'s public behavior for
  existing callers without an explicit, documented decision -- per
  `CLAUDE.md` rule 8, no public API change without explicit user approval.

**Status: proposed, not yet implemented.** Per the Goal's mutual-agreement
requirement, awaiting Codex's independent read/reaction before
implementation begins, and per the double-experiment protocol, expect
independent testing from both models plus a remote (Kaggle or Colab)
check before any promotion decision. This is one candidate among several
to discuss over the agreed multi-hour window.

**Naming collision found (flag for Codex)**: Codex independently used
"Proposal 6" for a DIFFERENT, unrelated proposal (OpenNeuro real-data
quality gap vs HistGBDT, see the "Proposal 6 pinned Kaggle rerun" entries
elsewhere in this file, already closed/decided). These two "Proposal 6"s
are NOT the same thing -- this one (categorical target-encoding leakage)
should probably be renumbered (e.g. this file's own later section calls
it "Proposal 6" again at line ~1132 for yet another OpenNeuro-quality
item). Recommend Codex and Claude agree on a single shared numbering
scheme going forward to avoid this ambiguity recurring.

**[Claude Code] IMPLEMENTED, user-approved 2026-07-22**: per explicit
user instruction ("نفّذ إصلاح التسريب الفئوي"), implemented the
out-of-fold fix in `kanboost/core/encoders.py`:
- `fit()`/`transform()` are unchanged in behavior for disjoint data
  (train-fit -> val/test-transform remains correct and was never the
  leaking path).
- `fit_transform()` (the ONLY call site that previously re-encoded the
  same rows used to compute the mapping -- confirmed via
  `grep -rn "fit_transform\|\.transform("` across `kanboost/`, single
  call site at `base.py:235`, all other 6 call sites are `transform()`
  on disjoint val/test/interpretation data) now additionally computes
  each fitting row's categorical encoding out-of-fold via K-fold
  (`cv_folds=5` default, new constructor param, `random_state=42`
  default), backing off to the same global-mean smoothing as before
  within each fold. The stored `cat_maps_` (fit on ALL rows) is
  unchanged and still used for `transform()` on new data, which is
  correct there.
- Re-ran the exact noise-categorical decisive test
  (`scratchpad/proposal6_verify_fix.py`) against the NEW
  `fit_transform()` path directly (previously the test measured a
  standalone OOF reimplementation, not the production code path):
  corr(cat_encoded, y) = +0.032 / -0.029 / +0.024 / -0.011 for
  n_categories = 10/30/50/100 -- flat near zero, no growth with
  cardinality, confirming the leak is closed in the actual production
  method callers use.
- Full `pytest tests/` run completed: **165 passed, 0 failed** (1076.6s).
  No regressions from the `fit_transform()` OOF change.
- User approved commit+push directly; committed and pushed to
  `research/als-solve-perf`.

**[Retag note, Claude Code, 2026-07-22]**: retagged from "Proposal 6b" to
**CC-6b** per the new naming convention (see "## Proposal naming
convention" near the top of this file).

**[Claude Code] Proposal CC-6b -- hierarchical (backoff-to-parent) encoding,
tested on controlled synthetic data, decisive result.** Per the user's
request to research hierarchical structure in categorical tabular data:
found a real, established technique -- hierarchical/backoff target
encoding with partial pooling (sourced: [category_encoders' hierarchical
target encoding](https://github.com/scikit-learn-contrib/categorical-encoding/issues/136),
[Bayesian partial-pooling/shrinkage](https://vasishth.github.io/IntroBayesSMLP2021/slides/04HLMAdditionalNotesShrinkage.pdf)) --
instead of smoothing a sparse fine-grained category (e.g. city) toward
the GLOBAL mean (current KANBoost/CatBoost-style behavior), smooth toward
its PARENT category's mean (e.g. region), which itself has far more
samples and is a much better prior for a sparse child category.

**Synthetic test** (`proposal6b_hierarchical_encoding.py`): built a
2-level region->city hierarchy (8 regions x 25 cities) with a real region-
level effect, city-level effects correlated with their region, and
Zipf-like city popularity (most cities have very few samples) -- both
encodings computed out-of-fold (Proposal 6's leakage fix already applied
to both arms, isolating this as a separate question: given leakage is
fixed, does backing off to the parent instead of the global mean help?).
Measured MSE between each encoding and the TRUE per-city effect:

| seed | all-rows MSE improvement | sparse-city-rows MSE improvement | corr(encoding,y) flat->hier |
|---|---|---|---|
| 0 | +89.3% | +92.9% | 0.825 -> 0.893 |
| 1 | +95.2% | +95.8% | 0.886 -> 0.938 |
| 2 | +91.8% | +89.3% | 0.876 -> 0.927 |

**Decisive, consistent, large improvement (89-95% MSE reduction) across 3
random seeds, largest exactly where it should matter most (sparse
categories).** This is on top of Proposal 6's leakage fix, not a
replacement for it -- both encodings here are already out-of-fold/
leakage-free; this isolates the SEPARATE backoff-target question.

**Next step**: validate on a real, recent dataset with genuine
hierarchical categorical structure. Found via web search: [Florida Real
Estate: Sold Properties Dataset 2026](https://www.kaggle.com/datasets/kanchana1990/florida-real-estate-sold-dataset-2026)
(10,893 verified closed residential sales, ZIP-code-level with
city/county structure, published 2026) -- a natural ZIP -> city hierarchy
with genuine real-world sparsity. Downloading via the already-working
Kaggle API access to test the same comparison on real, current data next.

**[Claude Code] Real-data result (Florida Real Estate 2026):** the
dataset has no explicit city/county column, only 5-digit `zip`. Used the
genuine, real US postal hierarchy instead of an invented one: zip5 (fine,
867 distinct values, 18.6% seen <=3 times) nested inside zip3 (coarse, 25
distinct values, far denser). Target = log(lastSoldPrice), same
out-of-fold flat-vs-hierarchical comparison as the synthetic test
(n=10,891 after dropping null zip/price rows).

| subset | flat MSE (backoff-to-global) | hier MSE (backoff-to-zip3) | improvement |
|---|---|---|---|
| all rows | 0.75946 | 0.73024 | +3.85% |
| sparse zip5 rows only (n=319, <=3 samples) | 1.27003 | 1.16663 | +8.14% |

Direction and pattern match the synthetic result exactly (backoff helps,
and helps most on sparse fine-grained categories) but the MAGNITUDE is
far smaller (+3.9%/+8.1% vs +89-95% on synthetic) -- expected, since real
ZIP-code price variation is far less cleanly nested than the synthetic
generator's region-dominates-city construction, and this dataset only
has 2 hierarchy levels with modest (not extreme-Zipf) sparsity. Script:
`scratchpad/proposal6b_real_florida.py` (not checked into the repo).

Conclusion: CC-6b's real-data effect is smaller but directionally
confirmed and still positive even in the harder subset where it matters
(sparse categories).

**[Claude Code] IMPLEMENTED, user-approved 2026-07-22**: added an
optional `hierarchy: dict = None` constructor parameter to
`TabularPreprocessor` (`kanboost/core/encoders.py`) -- `{fine_col:
coarse_col}` backs a sparse fine category off to its parent's smoothed
mean instead of the flat global mean, for columns named as keys only;
any categorical column not in `hierarchy` is completely unaffected
(additive-only change, no behavior change for existing callers who don't
pass it). `_smoothed_map()` was generalized to take a per-row `prior`
array instead of a hardcoded global-mean scalar (global mean is just the
default prior when no hierarchy applies). `fit()`, `transform()` (with a
coarse-level fallback for unseen fine categories), and `fit_transform()`'s
OOF loop were all extended to support this.

Verification (`scratchpad/cc6b_verify_production.py`), using the actual
production `fit_transform()`, not a standalone prototype:
1. Flat path (no `hierarchy` passed) reproduces CC-6's exact leakage-fix
   correlations unchanged (+0.032/-0.029/+0.024/-0.011 for
   n_categories=10/30/50/100) -- confirms zero behavior change for
   existing callers.
2. Hierarchical path on the same synthetic region/city data reproduces
   the earlier prototype's result almost exactly, now via production
   code: +89.3%/+95.2%/+91.8% MSE improvement across 3 seeds (prototype
   was +89.3-95.8%, effectively identical).

Full `pytest tests/` re-run: **164 passed, 1 failed**
(`test_dashboard.py::test_app_runs_gam_with_data_and_edit_tab`, a
RuntimeError). Investigated: this test uses only numeric columns
(`a`,`b`,`c`, no categorical data at all, so it cannot be touched by an
encoders.py change) and passes cleanly (`1 passed`) when run alone in
8s. The test drives a Streamlit `AppTest` with `timeout=60` per
interaction; the failure is consistent with that timeout getting
squeezed under the full 165-test suite's machine load, not a real
regression. Confirmed unrelated and user-approved; committed as
`ec882d8` and pushed to `research/als-solve-perf`.

`hierarchy` is opt-in only -- no existing caller (`base.py:235`) passes
it, so production behavior for current users is unchanged until/unless
`base.py` is wired to accept and forward a hierarchy spec, which is a
separate, not-yet-made decision (would need a way for the end user to
declare which categorical columns nest under which parent).

---

## [Claude Code] CC-7 -- GAM additivity vs missingness x feature interactions

Continuing the "research more gaps" request. Refines the ledger's earlier
"Secondary, lower-priority finding" about missing-value handling: the gap
is not really about median-imputation quality, it's structural.

**Hypothesis**: KANBoost's missing-value mitigation (median impute +
`_missing` indicator column, `kanboost/core/encoders.py`) is a FLAT
additive feature. In GAM mode (`gam=True, kan_hidden=1`) the whole model
is a strictly additive GAM: `f(x) = sum_j g_j(x_j)`. This means the
missingness indicator can only shift the prediction by a constant -- it
structurally CANNOT change how another feature affects `y` depending on
whether a value was missing (no way to represent an interaction between
"is missing" and another feature). A tree model (XGBoost/LightGBM) can
trivially represent this via conditional splits.

**Decisive synthetic test** (`scratchpad/cc7_missing_value_gap.py`): `x1
~ U(-1,1)`; `x2` MNAR-missing more often when `x1<0`; `y = x1` when `x2`
present, `y = -x1` when `x2` missing (a textbook sign-flip interaction
between missingness and `x1`).

| model | val R2 |
|---|---:|
| KANBoost GAM (`gam=True`, additive) | 0.5289 |
| KANBoost non-GAM (`gam=False`, `kan_hidden=8`) | 0.9455 |
| Plain sklearn decision tree (depth=4) | 0.7779 |

**Confirmed decisively**: the purely additive GAM path caps out well
below both a model that can combine features nonlinearly (KANBoost
non-GAM, hidden units let the missingness indicator interact with `x1`)
and even a shallow tree. This is not a bug or a missing-value-handling
defect to "fix" -- it is the well-known, inherent expressiveness limit of
additive GAMs (no interaction terms), which KANBoost's GAM mode
deliberately trades away for exact closed-form solves, monotonicity
guarantees, and interpretability. Non-GAM KANBoost already closes this
specific gap when interactions matter.

**Conclusion / no action recommended**: this is a genuine, sourced,
structural finding worth documenting (users should know GAM mode can miss
missingness-driven interactions and should reach for non-GAM KANBoost --
or engineer an explicit `x1 * x2_missing` interaction feature -- if they
suspect this pattern), but there is no code fix to propose: it is not a
bug, and "add interaction terms to GAM mode" would defeat the entire
point of GAM mode (a deliberate, documented trade-off, not an oversight).
Recommend adding a one-line note to `TabularPreprocessor`'s or GAM mode's
docstring about this trade-off; no other production change proposed.

**Secondary, lower-priority finding (not yet investigated further)**: the
missing-value gap vs XGBoost/LightGBM's per-split learned default
direction is real but smaller in impact, since KANBoost already has a
reasonable mitigation (median imputation + a `_missing` indicator column
per feature) rather than a total absence of missing-value handling. Lower
priority than the categorical-leakage finding above.

---

## Codex speed-review proposals for Claude to test next

Added by Codex as a read-oriented review handoff, not as an implementation
decision. These are hypotheses for Claude Code to test under the existing
protocol before any production promotion. Preserve the current stable
depth-2 implementation unless the evidence below clears its predefined
acceptance gate.

**Shared protocol for all proposals**:
- Measure before and after on the current production baseline, using the
  same folds/seeds/datasets/preprocessing/hyperparameters and isolated
  hardware conditions.
- Report exact commands, runtime, peak memory if relevant, train/validation
  metrics, prediction parity or R2/AUC deltas, numerical tolerances,
  failures, and uncertainty.
- A speed-only refactor must be prediction-equivalent within floating-point
  tolerance. A training-speed refactor must also preserve validation quality
  within the stated gate.
- Stop after each proposal's evidence and request Codex + ChatGPT review
  before promoting anything into the public API.

**Status (Claude, this pass)**: evaluating these proposals per user
request ("try and evaluate"). Starting with Proposal 1 (closing out
already-in-progress work), since the ledger's own rigorous 8.9% figure is
BELOW Codex's stated >=10% gate at one hidden width -- this needs a
multi-width closeout, not just the single n=2000/kan_hidden=64 measurement
already on record, before deciding keep-vs-revert per Codex's own decision
rule.

**Proposal 1 CLOSEOUT RESULT (`proposal1_closeout.py`, warmed-up paired
same-process comparison, 5-fold CV, kan_hidden in {16, 64, 128})**:

| kan_hidden | orig | patched | speedup | R2 diff | passes >=10% gate? |
|---|---|---|---|---|---|
| 16 | 1.61s | 1.41s | +14.6% | +6.4e-14 | YES |
| 64 | 10.66s | 9.78s | +8.9% | 0 (exact) | NO |
| 128 | 30.57s | 31.15s | **-1.9% (SLOWER)** | 0 (exact) | **NO, regresses** |

R2 is bit-identical or ~1e-14 across all three (correctness never in
question -- the earlier `verify_cache_equivalence`-style checks hold at
every width). But **the speedup shrinks and then REVERSES as kan_hidden
grows** -- passing only at the least-important width (16, where fits are
already fast) and actively regressing at 128 (where speed matters most in
practice). Likely cause: the batching removes Python-loop overhead, which
is a large relative cost only when the loop is short (small kan_hidden);
at larger widths, the batched version's own temporary-array construction
(the `(n, n_hidden)` einsum/broadcast intermediates) starts to cost more
than it saves, while eigh/layer1-solve increasingly dominate the sweep
regardless.

**Per Codex's own explicit decision rule ("if speedup is below 10% or
parity fails, revert the working-tree change and record the rejection")**:
this fails at the two more realistic/commonly-used widths (64, 128) and
only passes at the least consequential one (16). **Recommendation:
REVERT the `network.py` change** -- it does not reliably clear its own
acceptance gate across realistic hidden widths, despite passing cleanly
at the single width (64, n=2000) originally spot-checked before this
wider sweep. This is exactly the kind of result the project's standing
discipline exists to catch: an initially-promising single-point
measurement that does not survive a more thorough test across the actual
range of use.

**HOLD (superseded below) -- not reverted at the time, per user
instruction.** The edit remained on `research/als-solve-perf`,
uncommitted, while awaiting remote-hardware verification, since the
regression at larger `kan_hidden` locally might have been machine-specific
(6-core Windows, this particular OpenBLAS build/thread behavior, this
memory bandwidth/cache hierarchy).

**Remote verification on Kaggle (per Codex's documented remote-execution
path, `remote/kaggle_speed_bench/` protocol) -- RESULT.** Kernel
`tuamamhamza/kanboost-proposal1-verify`, self-contained script embedding
the project's actual `bspline.py`/`layer.py`/`network.py` (base64-encoded,
mirrored at the real `kanboost/core/kan/` import path so the files'
own unmodified absolute imports resolve with zero text rewriting -- no
transcription risk), same warmed-up paired methodology as the local test.
Remote environment: Linux 6.12, Python 3.12.13, NumPy 2.0.2, **4 CPUs**
(vs local's 6). Outputs: `kaggle_kanboost-proposal1-verify_synthetic_p8_
20260721-204402_{results.json,metrics.csv,environment.txt,run_log.txt}`
under `remote/results/kaggle/`.

| kan_hidden | Kaggle (4-core Linux) | Local (6-core Windows) |
|---|---|---|
| 16 | +6.2% (fails >=10%) | +14.6% (passes) |
| 64 | **+19.1% (passes)** | +8.9% (fails) |
| 128 | +6.2% (fails) | -1.9% (fails, regresses) |
| 256 | +3.7% (fails) | not tested locally |

R2 is bit-identical (0.0 diff) on every width, on both machines --
correctness was never in question anywhere.

**This confirms the user's hardware-dependence hypothesis directly, but
sharpens the conclusion rather than rescuing the change: which width
clears the gate DIFFERS by machine (16 locally, 64 on Kaggle), and no
width clears it on BOTH machines simultaneously.** The benefit is real but
unpredictable in advance -- a user cannot know, without benchmarking their
own specific hardware and `kan_hidden` choice, whether this change helps,
hurts, or does nothing. That is a strictly weaker property than "a
reliable >=10% win," which is what Codex's acceptance gate requires.

**FINAL DECISION (Proposal 1, both machines evaluated): REVERTED.** Per
Codex's own decision rule, applied now with two-hardware evidence instead
of one: no configuration reliably clears the stated bar across realistic
`kan_hidden` values on either tested machine. `kanboost/core/kan/
network.py`'s per-hidden-unit layer0-update loop has been restored to its
original, pre-batching form (a `ponytail:` comment documents what was
tried and why, pointing back to this section). **Full test suite rerun
after the revert: 165 passed, 0 failed** (480s); the separately-validated
Proposal 2 GAM-caching fix is untouched by this revert and remains in
place. This closes Proposal 1.

**Remote-execution infrastructure notes (for future proposals)**: the
Kaggle push/status/output loop works from this session using Codex's
documented bundled-Python-runtime path; a self-contained kernel script can
embed the project's real source files verbatim via base64 (avoids
transcription risk) and must mirror the REAL import path
(`kanboost/core/kan/...`) rather than renaming the package, or the files'
own absolute imports fail (`ModuleNotFoundError`) -- this cost one wasted
push/run cycle before being caught. A `{{`/`}}`-brace template-escaping
bug (assuming `.format()` semantics while using `.replace()`) caused a
second wasted cycle (`TypeError: unhashable type: 'dict'` from a doubled
dict literal) -- always run a full local execution of the generated
script, not just a syntax check, before pushing. A stray local
`__pycache__` directory left in the kernel folder from a local smoke-test
run caused one `kaggle kernels push` API error ("error occurred while
saving the entity changes") -- clean the kernel folder to only its
intended source files before every push.

### Proposal 1 -- close and verify the batched layer0 ALS solve

**Hypothesis**: The current working-tree change in
`kanboost/core/kan/network.py` that replaces the per-hidden-unit layer0
update loop with one batched solve is a safe production-speed improvement.
The ledger already records an isolated equivalence check and an end-to-end
speedup around 1.16x, but this should be closed formally before relying on
it.

**Files to inspect/test**:
- `kanboost/core/kan/network.py`, around the layer0 update inside
  `DeepKAN._fit_als`.
- `tests/test_deepkan.py`, `tests/test_kanboost.py`, and any existing
  scratchpad equivalence script used for this exact change.

**Acceptance criteria**:
- Coefficients/predictions match the original per-hidden loop to <=1e-10
  on representative hidden widths, including `kan_hidden` in {16, 64, 128}
  if runtime permits.
- End-to-end fit speed improves by >=10% on at least one real
  KANBoostRegressor or KANBoostClassifier workload with no validation
  metric degradation beyond CV noise.
- No increase in condition-number warnings, rollback behavior, or memory
  large enough to matter.

**Decision rule**: If the change clears the gates, propose it as a narrow
production refactor. If speedup is below 10% or parity fails, revert the
working-tree change and record the rejection.

**Proposal 4 RESULT -- REJECTED, per its own decision rule, no
implementation needed.** Profiled a real 20x `predict()` call on a fitted
30-estimator, kan_hidden=64 model: total 6.750s, of which `torch.tensor`
calls sum to 0.031s (**0.46% of total** -- nowhere near the >=10% gate).
The actual cost is B-spline basis evaluation (`_b_basis_1d`/
`_b_basis_1d_numba`, ~76% of total) and `KANLayer.forward`, not the
torch/numpy boundary. Per Codex's own stated rule ("if the conversion cost
is negligible compared with B-spline/eigh work, record the negative
result and do not refactor"), this is closed without touching
`base.py`/`network.py`'s torch boundary at all.

### Proposal 2 -- add GAM training basis/system cache across boosting rounds

**Hypothesis**: Non-GAM ALS already reuses layer0 basis/system information
across a boosting chain via `basis_cache`, but GAM mode rebuilds the same
stacked design matrix, penalty matrix, weighted normal equations, and
eigendecomposition for every learner even when `X`, `grid`, `k`,
`lamb*`, monotone constraints, and sample weights are unchanged. A
GAM-specific cache should reduce `gam=True` training time without changing
predictions.

**Files to inspect/test**:
- `kanboost/core/base.py`, `_boost_chain` and `_fit_learner`, where the
  existing `basis_cache` is created and passed.
- `kanboost/core/kan/network.py`, `_solve_gam`.
- GAM/monotone tests in `tests/test_kanboost.py` and symbolic/editing
  tests that rely on exact GAM behavior.

**Acceptance criteria**:
- For unconstrained `gam=True`, predictions and stored coefficients match
  the current path to <=1e-10 or a justified floating-point tolerance.
- For monotone-constrained `gam=True`, the projection path remains correct:
  monotonicity tests still pass and validation quality does not regress.
- Training wall-clock improves by >=15% on a realistic GAM workload, or the
  change is rejected as not worth added cache complexity.
- Cache keys must include enough information to avoid stale reuse when
  sample weights, regularization, grid/order, feature count, or monotone
  settings differ.

**Decision rule**: Promote only if the cache is pure performance with tight
parity and a measured >=15% GAM training win.

**Proposal 2 -- IMPLEMENTED and equivalence-verified, speed measurement in
progress.** Confirmed Codex's exact hypothesis by reading the code: GAM
layer knots are fixed at construction (data-independent -- verified two
freshly-constructed models with different seeds have identical
`layer.knots`), so for a GAM boosting chain every learner shares the same
X, knots, grid/k, lamb*, and sample_weight -- only the residual target `r`
changes each round. `_solve_gam` was rebuilding the full design matrix B,
penalty P_full, and normal-equations matrix A from scratch every single
learner, AND re-solving via `_solve_normal` (a fresh `_eigh_factor` every
time) despite `_solve_normal`'s own docstring saying "for many RHS against
the same M, factor once ... and reuse." `base.py`'s `basis_cache` dict was
already being created per boosting chain and passed into every learner's
`fit()` call (for GAM and non-GAM alike) -- it just wasn't being read by
`_solve_gam` at all.

**Fix**: `_solve_gam` now accepts `basis_cache`, and `fit()`'s dispatcher
passes it through (previously only forwarded to `_fit_als`). On a cache
miss, builds B/P_full/A and factors A once via `_eigh_factor`, storing
`(B, P_full, Bw, A, eigvecs, eigvals)`; on a hit, reuses the factorization
and only recomputes `b = Bw.T @ r` (cheap) plus the solve via
`_solve_with_factor`. Monotone projection (downstream of the solve, acting
on the resulting coefficients) is untouched by this change either way.

**Verification**: two learners run back-to-back sharing one `basis_cache`
dict give predictions **bit-identical (0.0 max diff, not just <1e-10)**
to two independent no-cache fits, both for the unconstrained path and the
monotone-constrained path. Cache confirmed to hold exactly 1 entry after 2
learners (proving actual reuse, not just correctness by coincidence) --
since caching only skips *recomputing the same deterministic operations*,
not approximating them, bit-identical output is the expected and correct
result here (stronger than the `<=1e-10` gate Codex asked for).

**[Claude Code] Real bug found and fixed by Claude Code**: moving `P_blk`'s
computation inside the cache-miss branch left it undefined
(`UnboundLocalError`) on a cache HIT, because the monotone-constrained
re-solve path (further down in `_solve_gam`, the unconstrained-features-
plus-intercept re-fit against the constrained-feature residual) also
needs `P_blk` regardless of whether the main system was cached --
`tests/test_accel.py::test_fast_fit_respects_monotone_constraints` failed
with this exact error on the first full-suite run. Claude Code fixed it
by always (re)computing `P_blk` unconditionally (it's a cheap K x K
matrix, independent of n/X, so there's no cost to not caching it) instead
of gating it behind the cache branch -- resolved without needing to ask
Codex/ChatGPT. Re-verified: bit-identical predictions on the monotone path
across a cache-miss-then-hit pair of learners (0.0 diff), cache confirmed
to hold exactly 1 entry after 2 learners. This is exactly the kind of
defect the project's "run the real test suite, not just a hand-picked
check" discipline exists to catch.

**[Claude Code] Full test suite: 165 passed, 0 failed** -- confirmed
twice (once right after the `P_blk` fix, once again after Proposal 1 was
separately reverted, to make sure the two changes don't interact).

**[Claude Code] Real end-to-end speedup measurement -- DECISIVE PASS.**
Warmed-up, paired, same-process comparison (`proposal2_speed.py`):
current on-disk cached behavior vs a monkeypatched uncached path (forces
`basis_cache=None` on every `_solve_gam` call, reproducing the pre-fix
behavior) -- same data/seeds, `gam=True`, `kan_hidden=1`, across
`n_estimators` in {50, 150, 300, 600} to check the benefit scales with
chain length rather than being a fixed-size artifact.

| n_estimators | no-cache | cached | speedup | R2 diff |
|---|---|---|---|---|
| 50 | 0.315s | 0.147s | **+114.7%** | 0.0 (exact) |
| 150 | 0.852s | 0.365s | **+133.4%** | 0.0 (exact) |
| 300 | 1.678s | 0.667s | **+151.8%** | 0.0 (exact) |
| 600 | 3.292s | 1.350s | **+143.9%** | 0.0 (exact) |

**Clears Codex's >=15% GAM-training-speedup gate by a wide, consistent
margin at every chain length tested (2.1x-2.5x faster)**, with R2
bit-identical (not just within tolerance) in every case -- correctness was
never in question, and unlike Proposal 1, the benefit does not degrade or
reverse at any tested size; if anything it strengthens with more
estimators (as expected, since the one-time system build/factorization
cost is amortized over more learners).

**[Claude Code] Proposal 2: ACCEPTED for promotion, pending Codex read-only
review and ChatGPT sign-off per the standard pipeline** -- the code is
already applied to `kanboost/core/kan/network.py` on
`research/als-solve-perf` (uncommitted), full test suite green, equivalence
bit-identical, speedup decisive and stable across chain lengths. This is
the strongest, most unambiguous result of any of the five proposals so
far. No merge to `main` without explicit user approval, per `CLAUDE.md`
rule 8.

**[Claude Code] Committed and pushed**: `research/als-solve-perf` @
`11f4b0fd95ac40aa475702a81d5efbb749cee55b`, pushed to
`https://github.com/tuamah/kanboost` (public, verified reachable via raw
content fetch). Commit message documents both the Proposal 2 GAM-cache
addition and the Proposal 1 revert in one changeset (same file). Still no
merge to `main` -- this is a pushed research branch only, per explicit
user instruction to "push an update," which the user distinguished from
merging.

**[User -- relaying external review] Critical scope note for the
OpenNeuro benchmark notebook, before extending Proposal 2 testing there**:
Proposal 2's cache only accelerates the `gam=True, kan_hidden=1` path. The
OpenNeuro notebook's existing `kanboost_project` model uses `kan_hidden=3,
gam=False` (the non-GAM ALS path) -- it never exercises this cache at all,
so it cannot be used to demonstrate this speedup. If/when Proposal 2 is
tested on the OpenNeuro notebook specifically:
- Add a separate model `kanboost_gam_cached` = `KANBoostClassifier(
  n_estimators=stage["kanboost_estimators"], kan_steps=stage["kanboost_steps"],
  kan_hidden=1, gam=True, early_stopping_rounds=None, random_state=RANDOM_STATE)`.
- Ideally add an internal `uncached` comparison variant on the SAME Kaggle
  runtime -- same data/folds/seed/n_estimators, only cache on/off -- for a
  fair, isolated speed measurement.
- Keep two separate tables always: model-quality/accuracy across all
  models (existing), and cached-vs-uncached speed for KANBoost-GAM alone.
  Never conflate "is KANBoost accurate on OpenNeuro" with "does the GAM
  cache make training faster" -- they are unrelated questions answered by
  the same notebook.

Not yet implemented in the OpenNeuro notebook -- recorded here so it isn't
lost before that extension happens.

**[Claude Code] Standalone Kaggle verification of Proposal 2 -- COMPLETE,
cross-hardware CONFIRMED.** Kernel `tuamamhamza/kanboost-proposal2-gam-
cache` (`remote/kaggle_proposal2/`), `pip install git+https://github.com/
tuamah/kanboost.git@11f4b0fd95ac40aa475702a81d5efbb749cee55b` (the exact
pushed commit, verified installed correctly by grepping the installed
`network.py` for `basis_cache` before trusting the run -- not assumed).
Same synthetic data/protocol as the local measurement (warmed-up, paired,
same-process, `gam=True`, `kan_hidden=1`, `n_estimators` in {50, 150, 300,
600}). Outputs: `kaggle_kanboost-proposal2-gam-cache_synthetic_p8_
20260721-212729_{metrics.csv,results.json,environment.txt,run_log.txt}`
under `remote/results/kaggle_proposal2/`.

| n_estimators | Kaggle speedup | local speedup | R2 diff (both) |
|---|---|---|---|
| 50 | **+160.8%** | +114.7% | 0.0 (exact) |
| 150 | **+218.3%** | +133.4% | 0.0 (exact) |
| 300 | **+226.1%** | +151.8% | 0.0 (exact) |
| 600 | **+237.7%** | +143.9% | 0.0 (exact) |

**Even stronger on Kaggle's 4-core Linux runner than locally (2.6x-3.4x
vs 2.1x-2.5x), growing monotonically with chain length on BOTH machines,
bit-identical R2 in every single case on both.** This is the cross-
hardware confirmation Proposal 1 failed to achieve -- Proposal 2 passes
decisively on two different machines, unlike Proposal 1 which only passed
inconsistently at different, non-overlapping widths per machine. Per the
external review's own framing: this makes Proposal 2 the strongest
promotion candidate of any speed proposal tested this session.

**Status: Proposal 2 fully verified (equivalence + local speedup + cross-
hardware Kaggle speedup), pushed to `research/als-solve-perf`, awaiting
Codex read-only review and ChatGPT sign-off before any merge to `main`.**
Next planned step (not yet started, per the external review's explicit
guidance above): extend to the OpenNeuro notebook with a dedicated
`kanboost_gam_cached` model and an internal cached-vs-uncached arm, kept
in a separate table from cross-model accuracy comparisons.

**[Codex Review] Remaining speed bottlenecks to test after Proposal 2**:
Proposal 2 is currently the strongest proven speed improvement because it
removes repeated invariant work in the `gam=True, kan_hidden=1` training
path and has local bit-identical results with 2.1x-2.5x speedup. After its
Kaggle verification, the next speed work should be prioritized as follows:

1. **Prediction/ensemble evaluation path**: likely the largest remaining
   project-level speed target once GAM training is cached. `_raw_score_chain`
   still loops over learners one by one and each learner independently
   evaluates B-spline bases. This affects serving, repeated CV scoring,
   dashboards, and any `predict_proba`-heavy workflow. Test as Proposal 3,
   but first profile where time goes after Proposal 2 lands. Acceptance
   should stay strict: prediction parity <=1e-9, saved/load parity, binary
   + multiclass + regression coverage, and >=2x prediction speedup on
   40-100 estimator workloads before adding a compiled/private predict path.
2. **Feature-extraction pipeline for OpenNeuro/EEG benchmarks**: on real EEG
   runs, MNE loading/filtering/PSD extraction can dominate wall-clock more
   than KANBoost itself. This is not a KANBoost core bottleneck, so do not
   use it to justify model-code changes; instead cache extracted
   subject-level features by `(dataset, subject, max_seconds, preprocessing
   params)` in the notebook/results workflow. Acceptance: exact feature-file
   reuse across reruns, no train/test leakage, and clear separation from
   model fit timing.
3. **Non-GAM training ALS internals**: still worth profiling, but previous
   Proposal 1 showed this area is hardware- and width-sensitive. Do not
   promote another non-GAM training refactor unless it is warmed-up,
   same-process, multi-width, and Kaggle-verified with stable speedups across
   realistic `kan_hidden` values. Prediction/serving work has a cleaner
   expected payoff than more layer0-solve tweaks right now.
4. **Torch <-> NumPy boundary**: already profiled as too small for Proposal 4
   (about 0.46% of a real predict workload), so keep it closed unless a new
   profile after Proposal 2 materially changes the breakdown. Avoid CPU path
   refactors here without fresh evidence.

For Claude Code: do not mix these questions. Proposal 2 Kaggle runs should
measure `KANBoost-GAM cached` vs `KANBoost-GAM uncached`; OpenNeuro model
quality tables should remain separate from cached-vs-uncached timing; and
feature-extraction timing should be reported separately from model fit and
prediction timing.

**[Claude Code] Proposal 3 -- investigated, real precision bug found and
fixed, speedup measured (falls short of gate so far).**

**Real caching opportunity confirmed by code inspection**: layer0's knots
are fixed at construction (grid-based, not data-adaptive) and are
identical across every learner in a boosting chain, since layer0 never
gets rebuilt from `z` (unlike layer1, whose knots genuinely differ per
learner -- verified empirically: `layer0.knots` identical across all
learners, `layer1.knots` differ). So the layer0 B-spline basis matrix
(built from X and the shared knots) can be computed ONCE per prediction
call and reused across every learner, instead of re-derived per learner.

**[Claude Code] Real bug found and fixed by Claude Code (not Codex)**:
first parity check against the current per-learner `_raw_score_chain`
failed at ~1e-8 (above the stated <=1e-9 gate), even at `n_estimators=1`
(so not ensemble-accumulated noise). Root cause isolated via a direct
single-learner comparison: `DeepKAN.__call__` casts each learner's output
to `x.dtype` (float32, from `_transform_X`) before `_raw_score_chain` sums
it -- production already rounds every learner's contribution to float32,
and the cached prototype was initially staying in float64 throughout,
which is a genuine (if tiny) precision *change*, not a bug in the cached
math itself. Fixed by casting each learner's cached-path output to
float32 (matching `.cpu().numpy()`'s own dtype) exactly once, no
redundant re-cast. **Result: EXACT 0.0 parity** (not just <1e-9) across
n_estimators in {1,2,5,10,20}, kan_hidden in {16,32}, and both GAM and
non-GAM modes. Resolved without needing to ask Codex/ChatGPT.

**Speed measurement (non-GAM, kan_hidden=16)** -- Codex's gate: >=2x
(100%) prediction speedup:

| n_estimators | original | cached | speedup |
|---|---|---|---|
| 40 | 170.36ms | 119.71ms | +42.3% |
| 100 | 356.87ms | 235.42ms | +51.6% |
| 200 | 684.75ms | 481.18ms | +42.3% |

**Real, consistent speedup (~1.4-1.5x) but falls short of the stated >=2x
gate at kan_hidden=16.**

**[Claude Code] Scaling test at larger kan_hidden -- speedup SHRINKS, not
grows, as kan_hidden increases:**

| kan_hidden | n_estimators=40 | n_estimators=100 |
|---|---|---|
| 16 | +42.3% | +51.6% |
| 32 | +18.6% | +20.2% |
| 64 | +10.8% | +13.2% |

**Root cause**: the cache only accelerates layer0's basis evaluation
(shared across learners); layer1's cost (which depends on `kan_hidden`)
does NOT benefit from this cache and grows with `kan_hidden`, so it
dominates an ever-larger share of total prediction time as `kan_hidden`
increases -- the exact parameter that matters most for model capacity is
also the one that erodes this optimization's relative benefit. **[Claude
Code] local verdict: FAILS the >=2x gate at every tested width (16/32/64),
and the gap worsens (not improves) at more realistic/larger hidden sizes.
Recommend rejection, pending Codex's independent local result and the
double-experiment Kaggle check per `AGENTS.md`'s new protocol before this
is finalized.**

### Proposal 3 -- compiled or cached prediction path for non-GAM ensembles

**Hypothesis**: Serving-time prediction for non-GAM models is still slow
because `_raw_score_chain` loops over every weak learner and each learner
re-evaluates B-spline bases independently. GAM has a documented fast path
via `consolidate()`, but `gam=False` does not. A prediction-only cache or
compiled ensemble representation may reduce serving latency substantially
without touching training.

**Files to inspect/test**:
- `kanboost/core/base.py`, `_raw_score_chain`, `predict`, and
  `predict_proba`.
- `kanboost/core/kan/layer.py`, `KANLayer.forward`.
- `kanboost/interpret/editing.py`, only as a reference for the successful
  GAM consolidation pattern; do not force that additive assumption onto
  non-GAM models.

**Candidate approaches**:
- Per-call basis cache keyed by `(input column values, knots, k)` so
  repeated learner calls on the same transformed `X` can reuse basis
  evaluations where grids match.
- A separate `compile_predict()` / private compiled-chain object that
  pre-packs learner coefficients and knots for faster batch inference.
- Keep this prediction-only; do not change training semantics.

**Acceptance criteria**:
- `predict` / `predict_proba` parity with current models to <=1e-9 on
  binary, multiclass, regression, and saved-loaded models.
- At least 2x prediction speedup on a non-GAM ensemble with enough learners
  to represent serving use (for example 40-100 estimators), measured after
  warmup.
- Memory overhead is reported and remains reasonable for large `n_estimators`.

**Decision rule**: Prefer an opt-in/private compiled path first if it adds
state or memory. Do not complicate the main training loop for prediction
speed.

**[Codex Review] Proposal 3 first remote profiling -- layer0-only cache
REJECTED as insufficient.** Codex ran an isolated Kaggle script, not a
production-code change: `tuamamhamza/kanboost-proposal3-predict-profile`
version 2, source under `remote/kaggle_proposal3_predict/`, installing the
verified Proposal 2 commit
`11f4b0fd95ac40aa475702a81d5efbb749cee55b`. Outputs downloaded under
`remote/results/kaggle_proposal3/` with prefix
`kaggle_kanboost-proposal3-predict-profile_synthetic_friedman_20260721-214747`.

Prototype paths tested against current `model.predict()` for non-GAM
`KANBoostRegressor` on synthetic Friedman data:
- `layer0_cached`: cache/reuse the first-layer B-spline bases per feature
  across learners, then loop learners for each learner's layer1.
- `layer0_stacked`: compute all learners' layer0 activations in larger BLAS
  calls, then loop learners for layer1.

| n_train | n_pred | n_estimators | kan_hidden | baseline predict | layer0_cached | cached speedup | layer0_stacked | stacked speedup | max diff |
|---|---|---|---|---|---|---|---|---|---|
| 2000 | 2000 | 40 | 16 | 0.204s | 0.137s | 1.49x | 0.240s | 0.85x | 6.6e-7 |
| 2000 | 5000 | 80 | 16 | 0.987s | 0.706s | 1.40x | 1.100s | 0.90x | 6.4e-7 |
| 3000 | 5000 | 80 | 32 | 1.903s | 1.602s | 1.19x | 2.104s | 0.90x | 7.3e-7 |

Decision: do **not** promote either layer0-only prototype. It misses the
predefined >=2x serving-speed gate, and the prototype parity is around
1e-7 rather than <=1e-9 because it bypasses the current torch-float32
roundtrip and sums in NumPy. The result is still useful: the main prediction
bottleneck after caching layer0 basis is the per-learner layer1 evaluation,
whose inputs `z` differ per learner and cannot be shared by a simple feature
basis cache. A future Proposal 3 variant would need a genuinely compiled
whole-chain representation or a mathematically different approximation with
its own accuracy gate; simple layer0 basis reuse is not enough.

Reconciliation with Claude Code's local result above: Claude's later local
prototype explicitly cast each cached-path learner output to float32 and
therefore achieved exact 0.0 parity with production. Codex's Kaggle prototype
did not include that cast, which explains its ~1e-7 max diff. This does not
change the decision because both independent runs agree on the important
speed conclusion: simple layer0 reuse is a real but insufficient ~1.2x-1.5x
win, below the >=2x Proposal 3 gate and shrinking as `kan_hidden` grows.

### Proposal 4 -- remove unnecessary Torch-to-NumPy round trips on CPU

**Hypothesis**: DeepKAN is NumPy/SciPy internally, but the outer estimator
still constructs torch tensors and converts each learner's output back to
NumPy. On CPU this may add overhead in training, prediction, and
interpretability calls.

**Files to inspect/test**:
- `kanboost/core/base.py`, `_transform_X`, `_fit_learner`,
  `_raw_score_chain`, `feature_contributions`, and `predict_derivative`.
- `kanboost/core/kan/network.py`, `DeepKAN.__call__` and `_forward_np`.

**Acceptance criteria**:
- A NumPy-native internal path gives >=10% speedup in prediction or fit on
  CPU workloads, with parity <=1e-9.
- CUDA/device behavior remains backward-compatible for callers that pass a
  torch device, or the NumPy path is guarded to CPU-only.
- Save/load and public API behavior remain unchanged.

**Decision rule**: Adopt only if the measured overhead is real. If the
conversion cost is negligible compared with B-spline/eigh work, record the
negative result and do not refactor.

**[Claude Code] Proposal 5 -- extraordinary result, biggest win of the
whole session.** Code inspection first: `feature_contributions()` touches
ONLY layer0 (`learner.act_fun[0].postacts(x_np)`, summed over the hidden-
unit axis) -- no layer1 involvement at all, unlike Proposal 3's full
prediction path. So the shared-layer0-basis cache applies with zero
dilution from a per-learner-varying layer1 cost.

**Parity** (`proposal5_interpret_cache.py`, cached `feature_contributions`
vs current): max diff 3.7e-16 to 1.8e-15 across n_estimators in
{10,20,50}, kan_hidden in {16,32} -- machine-precision, not just within
the 1e-10 gate.

**Speed** (Codex's gate: >=20% speedup):

| kan_hidden | n_estimators=40 | n_estimators=100 |
|---|---|---|
| 16 | +2172% (21.7x) | +2763% (27.6x) |
| 32 | +5147% (51.5x) | +4340% (43.4x) |
| 64 | **+12909% (129x)** | +11448% (114x) |

**The speedup GROWS with kan_hidden (opposite of Proposal 3)** -- makes
sense given the mechanism: `postacts` redundantly re-evaluates the same
B-spline basis once per learner regardless of `kan_hidden`, but does
`n_in x kan_hidden` matmuls against it; the cached version reuses one
basis build across the whole ensemble and only does the (cheap) matmul
work per learner, so the ORIGINAL cost grows with `kan_hidden` while the
CACHED cost barely does. **This clears the >=20% gate by two to three
orders of magnitude at every tested configuration -- the strongest,
clearest result of any proposal this session, well beyond even Proposal
2's 2-3x.**

**[Claude Code] `predict_derivative` (GAM mode) confirms the same
pattern**: GAM mode's derivative is also pure layer0
(`dF/dx_col = g_col'(x_col)`, no layer1 chain-rule term). Cached version
(shared `_b_basis_deriv_1d` build once, reused per learner): **exact 0.0
parity**, n_estimators=100 -- 30.77ms -> 3.51ms, **+777.5% speedup (8.8x)**.
Non-GAM mode's derivative needs a real layer1 chain-rule term (per the
`else` branch in `predict_derivative_analytic`) so it will dilute like
Proposal 3 -- not yet measured, lower priority given GAM mode alone
already clears the gate overwhelmingly and both `feature_contributions`
and GAM `predict_derivative` are commonly-used interpretability paths.

**[Claude Code] Proposal 5 local verdict: ACCEPT, by a very wide margin.**
Both tested interpretability paths (feature_contributions;
predict_derivative in GAM mode) clear the >=20%/1e-10 gates by 1-2+ orders
of magnitude, with parity at or near machine precision. Per the double-
experiment protocol, awaiting Codex's independent local result and a
Kaggle cross-hardware check before final promotion, but this is the
clearest ACCEPT of any proposal evaluated this session.

**[Claude Code] APPLIED to `kanboost/core/base.py`, version bumped to
1.2.3, committed and pushed.** `research/als-solve-perf` @
`6a4a5d39c52dd8ca4f42e5e726c2b715d1434a02`. Full test suite: 165 passed,
0 failed (738s). Cache added to `feature_contributions()` and
`predict_derivative()` (GAM-mode branch only; non-GAM keeps its original
per-learner call, documented as separate/not-yet-implemented since it
needs a real per-learner layer1 chain-rule term). Still no merge to
`main` -- pushed research branch only, per `CLAUDE.md` rule 8.

**[Claude Code] Colab notebook built and ready**:
`remote/colab_proposal5/kanboost_proposal5_interpret_cache.ipynb` -- 7
cells, installs kanboost from the exact pushed commit
`6a4a5d39c52dd8ca4f42e5e726c2b715d1434a02`, runs the same
feature_contributions (kan_hidden in {16,32,64}, n_estimators in {40,100})
and predict_derivative-GAM (n_estimators in {40,100}) parity+speed
protocol as the local measurement, writes the standard timestamped output
files (`colab_kanboost-proposal5-interpret-cache_synthetic_p8_<stamp>_
{results.json,metrics.csv,environment.txt,run_log.txt}`). Validated
locally: valid nbformat structure, all code cells syntax-checked clean.

**Not yet run** -- per `AGENTS.md`'s Colab workflow, this requires the
user to open it in Google Colab (colab.research.google.com, upload from
`remote/colab_proposal5/`) and run all cells themselves, since Colab
execution needs the user's own Google session; Claude Code cannot drive
it directly the way Kaggle's CLI allows. Awaiting that run, plus Codex's
independent local result and Kaggle run for Proposal 3, before final
promotion decisions on either proposal.

### Proposal 5 -- per-call basis reuse for interpretability methods

**Hypothesis**: `feature_contributions()` and `predict_derivative()` repeat
first-layer basis/derivative work learner-by-learner. A per-call cache may
speed dashboards and reports without changing model training or prediction.

**Files to inspect/test**:
- `kanboost/core/base.py`, `feature_contributions` and
  `predict_derivative`.
- `kanboost/core/kan/layer.py`, `postacts` and `deriv`.
- Dashboard/experimental tests that call these interpretability methods.

**Acceptance criteria**:
- Exact output parity to <=1e-10 for contributions and derivatives.
- >=20% speedup on a dashboard-like workload with many learners and at
  least several hundred rows.
- No stale cache across different `X` calls; cache lifetime should be one
  method call unless there is a stronger invalidation story.

**Decision rule**: Good candidate for a narrow internal refactor if the
dashboard/report speedup is measurable; otherwise leave the simpler loop.

**[Codex Review] Proposal 5 remote profiling -- PASSES as a narrow
interpretability-speed candidate.** Codex ran an isolated Kaggle script, not
a production-code change: `tuamamhamza/kanboost-proposal5-interpret-profile`
version 2, source under `remote/kaggle_proposal5_interpret/`, installing
commit `11f4b0fd95ac40aa475702a81d5efbb749cee55b`. Outputs downloaded under
`remote/results/kaggle_proposal5/` with prefix
`kaggle_kanboost-proposal5-interpret-profile_synthetic_friedman_20260721-215902`.

Prototype paths:
- `feature_contributions_cached`: compute each feature's layer0 B-spline
  basis once per method call and reuse it across learners.
- `predict_derivative_cached`: compute layer0 bases once and compute the
  selected feature's derivative basis once, instead of recomputing it per
  hidden unit/per learner.
- The v1 prototype showed ~1e-7 parity noise from dtype mismatch. v2 casts
  transformed inputs to float32 to match production's `_transform_X` /
  `DeepKAN.__call__` behavior before evaluating bases.

| n_train | n_eval | n_estimators | kan_hidden | contributions speedup | contrib max diff | derivative speedup | derivative max diff |
|---|---|---|---|---|---|---|---|
| 2000 | 2000 | 40 | 16 | 6.55x | 1.1e-15 | 2.16x | 2.7e-15 |
| 2000 | 5000 | 80 | 16 | 6.68x | 1.8e-15 | 2.15x | 4.4e-15 |
| 3000 | 5000 | 80 | 32 | 7.29x | 3.6e-15 | 2.41x | 4.4e-15 |

Decision: Proposal 5 clears its >=20% dashboard/report speed gate by a wide
margin with effectively exact parity. This does **not** speed ordinary
training or prediction, but it is valuable for dashboards, reports,
monotonicity/derivative analysis, and repeated interpretability workflows.
Recommendation for Claude Code: implement as a narrow internal refactor in
`feature_contributions()` and `predict_derivative()` only, with one-call
cache lifetime, no public API change, tests for regression/binary/multiclass
and save/load behavior, and parity gates at <=1e-10. Keep it separate from
Proposal 3 serving-speed work, which failed its >=2x gate.

### Proposal CX-6 -- close real-data model-quality gaps before optimizing more speed

**[Retag note, Claude Code, 2026-07-22]**: this proposal was originally
labeled "Proposal 6" by Codex, independently of Claude's own unrelated
"Proposal 6" (categorical target-encoding leakage, see the "Proposal 6 --
fix categorical target-encoding leakage" section elsewhere in this file).
Per the user's instruction, retagged as **CX-6** (Codex-originated) to
disambiguate; the categorical-leakage one is retagged **CC-6**
(Claude-Code-originated). See the new "## Proposal naming convention"
section for the scheme going forward.

**USER FINAL DECISION (2026-07-22): CX-6 ACCEPTED.** The user approved
this proposal's conclusion as final: KANBoost's OpenNeuro `ds004504`
large-stage quality gap vs `hist_gbdt_clean` is closed within the
predefined <=0.05 gate (0.047), medium-stage is tied, and KANBoost
retains better log loss/calibration than both HistGBDT and RBF-KAN at
large scale. Per the synthesis above, this is accepted as "competitive
quality with better calibration on this protocol" -- not a claim of
universal OpenNeuro superiority or a speed result. No further local
double-experiment closure is required; the Kaggle pinned rerun stands as
sufficient evidence for this decision.

**[Codex Review] Hypothesis**: The next highest-value accuracy question is
not another low-level speed cache; it is why strong baselines beat KANBoost
on real OpenNeuro EEG features in the first small smoke run. On
`ds004504_v1_0_7_eeg_small` (8 subjects, 115 features), CatBoost and
RBF-KAN reached mean balanced accuracy 0.75 while `kanboost_project`
(`kan_hidden=3`, `gam=False`) reached 0.625. Because the dataset is tiny and
high-dimensional, the likely gap is not "KAN cannot model the signal"; it is
sample-efficiency/regularization under p >> fold-size pressure.

**Competitor/gap**:
- CatBoost/RBF-KAN are ahead on balanced accuracy/F1 in the small OpenNeuro
  smoke run.
- KANBoost non-GAM has more flexible basis structure than the data can
  reliably support in 6-train/2-valid folds.
- KANBoost-GAM was not tested there yet, and Proposal 2's `gam=True` cache
  means GAM variants are now cheap enough to include.

**Experiments to run under the double-experiment protocol**:
1. Add `kanboost_gam_cached` (`gam=True`, `kan_hidden=1`) to the OpenNeuro
   model-quality table. This tests whether the more biased/interpretable
   additive model generalizes better than non-GAM KANBoost on low-n EEG.
2. Add a separate `kanboost_gam_cached_vs_uncached` timing table only for
   Proposal 2; do not mix this with accuracy.
3. Test a leakage-safe feature-selection wrapper inside each CV fold:
   `SimpleImputer -> SelectKBest(mutual_info_classif or f_classif, k in
   {10, 20, 40}) -> KANBoostClassifier`, comparing GAM and non-GAM. The
   selector must be fit on the training fold only.
4. Test stronger regularization/low-capacity KANBoost variants:
   `kan_hidden in {1,2,3}`, `kan_grid in {2,3}`, and modest `lamb`/
   `lamb_coefdiff` only if the code path supports it without changing public
   API semantics.

**Acceptance gates**:
- Quality: on medium/large OpenNeuro stages, KANBoost variant should either
  match the best baseline within 0.05 mean balanced accuracy or improve over
  current `kanboost_project` by >=0.05 without worsening macro F1/log loss
  materially. Small-stage results alone are only smoke evidence.
- Speed: any added KANBoost quality variant must report fit/predict time
  separately; slow variants must justify their quality gain.
- Leakage: all feature selection, scaling, imputation, and label encoding
  must occur inside the CV fold or be provably unsupervised and prefit only
  on training data. Subject-level split remains mandatory.
- Evidence: Claude local, Codex local if feasible, and Kaggle/Colab remote
  results must be recorded side-by-side. If local cannot run because the
  machine lacks dependencies/resources, record that and use Kaggle as the
  decisive hardware-neutral run.

**Decision rule**: Promote/tune only variants that improve real-data
validation quality under the leakage-safe protocol. If CatBoost remains
ahead after medium/large stages, record the gap honestly and treat KANBoost's
advantage there as interpretability/calibrated smooth structure rather than
raw accuracy.

**[Codex Review, 2026-07-22] Kaggle small-feature smoke result --
INCONCLUSIVE / not accepted as final quality evidence.**

Remote run:
- Kernel: `tuamamhamza/kanboost-proposal6-openneuro-gap-smoke`.
- Dataset: `tuamamhamza/openneuro-ds004504-small-features`.
- Input feature file:
  `remote/kaggle_dataset_openneuro_small_features/openneuro_small_features.csv`.
- Output prefix:
  `remote/results/kaggle_proposal6/kaggle_kanboost-proposal6-openneuro-gap-smoke_ds004504_small_features_20260721-221202`.
- Environment: Kaggle Linux 6.12, Python 3.12.13, NumPy 2.0.2.
- Local sides: Claude local result for this exact Proposal 6 protocol is
  not yet recorded; Codex local could not be used as decisive evidence
  because this session's bundled local Python lacks the needed sklearn/
  model-stack dependencies. Per the double-experiment rule, this is remote
  smoke evidence only, not a full acceptance package.
- Commands used (through the bundled Kaggle CLI):
  `python.exe -X utf8 -m kaggle datasets create -p remote\kaggle_dataset_openneuro_small_features -q`;
  `python.exe -X utf8 -m kaggle kernels push -p remote\kaggle_proposal6_openneuro_gap`;
  `python.exe -X utf8 -m kaggle kernels status tuamamhamza/kanboost-proposal6-openneuro-gap-smoke`;
  `python.exe -X utf8 -m kaggle kernels output tuamamhamza/kanboost-proposal6-openneuro-gap-smoke -p remote\results\kaggle_proposal6 -o`.

Summary table:

| model | balanced acc | F1 macro | log loss | mean fit | mean predict | AUC |
|---|---:|---:|---:|---:|---:|---:|
| catboost | 0.750 | 0.667 | 0.683 | 0.107s | 0.0045s | 0.750 |
| kanboost_gam_select20 | 0.625 | 0.583 | 0.653 | 0.054s | 0.0242s | 0.750 |
| kanboost_gam | 0.625 | 0.500 | 0.663 | 0.168s | 0.0673s | 1.000 |
| kanboost_nongam_h3 | 0.500 | 0.333 | 0.657 | 0.821s | 0.0641s | 0.750 |
| xgboost | 0.500 | 0.333 | 0.693 | 0.032s | 0.0039s | 0.500 |
| kanboost_nongam_select20 | 0.375 | 0.250 | 0.703 | 0.131s | 0.0203s | 0.750 |

Interpretation:
- CatBoost remains the best small-smoke model by balanced accuracy and F1.
- `kanboost_gam_select20` is the best KANBoost variant in this run: it
  matches plain GAM balanced accuracy (0.625), improves F1/log loss, and is
  faster to fit/predict than the full 115-feature GAM. This supports the
  low-capacity + leakage-safe feature-selection hypothesis, but does not
  clear the predefined medium/large quality gate.
- Non-GAM KANBoost remains weak on this tiny p >> n setting in this smoke:
  `kanboost_nongam_h3` is at chance-level balanced accuracy, and
  `kanboost_nongam_select20` is worse. This strengthens the current
  diagnosis that sample efficiency/regularization is the active real-data
  gap, not another ordinary training-speed cache.
- The selector emitted sklearn warnings that features `[0 1]` were constant
  (`invalid value encountered in divide`). These likely correspond to
  metadata-like constant columns in the small feature table, so the next
  OpenNeuro notebook must drop constant/non-biological columns inside the
  fold-safe preprocessing path before feature selection and model fitting.
- Because n=8 subjects gives folds of only 6 train / 2 validation examples,
  one fold can swing the mean dramatically. This run is useful for
  debugging the protocol and ranking next variants, but it is not enough to
  accept or reject Proposal 6 scientifically.

Next required Proposal 6 step:
- Extend the OpenNeuro sequential notebook to medium/large stages with a
  cleaned feature set, subject-level CV, and separate tables for model
  quality vs `KANBoost-GAM cached/uncached` timing. Record Claude local,
  Codex local if feasible, and Kaggle/Colab remote results side-by-side
  before deciding whether KANBoost can close the CatBoost/RBF-KAN real-data
  accuracy gap.

**[Codex Review, 2026-07-22] Proposal 6 clean-feature smoke -- 
INCONCLUSIVE but materially improves the next hypothesis.**

Reason for this follow-up: the previous smoke emitted `SelectKBest`
warnings for constant features. Codex created a separate Kaggle script,
not modifying the original Proposal 6 kernel, to add a fold-local
`VarianceThreshold(threshold=0.0)` after median imputation and before any
supervised `SelectKBest`. This keeps the cleaning leakage-safe and tests
whether constant columns were distorting the tiny-feature smoke.

Remote run:
- Kernel: `tuamamhamza/kanboost-proposal6-clean-openneuro-smoke`.
- Version 1 failed because the Kaggle input CSV was not found at the
  assumed path. Fixed by including `openneuro_small_features.csv` in the
  kernel folder and adding a `/kaggle/input/**/openneuro_small_features.csv`
  fallback search.
- Version 2 completed.
- Dataset/source: `tuamamhamza/openneuro-ds004504-small-features`, with
  in-kernel CSV fallback.
- Output prefix:
  `remote/results/kaggle_proposal6_clean/kaggle_kanboost-proposal6-clean-openneuro-smoke_ds004504_small_features_cleaned_constant_filter_20260721-222538`.
- Environment: Kaggle Linux, Python 3.12.13, NumPy 2.0.2.
- Cleaned features: 115 raw model features, 3 globally constant, 112 after
  fold-local `VarianceThreshold` for full-feature models; selected models
  then used 10 or 20 features inside the fold.
- Commands used: `kaggle kernels push -p remote\kaggle_proposal6_clean_smoke`;
  `kaggle kernels status tuamamhamza/kanboost-proposal6-clean-openneuro-smoke`;
  `kaggle kernels output tuamamhamza/kanboost-proposal6-clean-openneuro-smoke -p remote\results\kaggle_proposal6_clean -o`.
- Local sides: still not a complete double-experiment package. Claude local
  for this exact clean smoke is not recorded; Codex local remains limited by
  local sklearn/model-stack availability. Treat this as remote protocol
  evidence and hypothesis-sharpening only.

Summary table:

| model | balanced acc | F1 macro | log loss | mean fit | mean predict | AUC | mean features |
|---|---:|---:|---:|---:|---:|---:|---:|
| catboost_clean | 0.750 | 0.667 | 0.683 | 0.096s | 0.0044s | 0.750 | 112 |
| kanboost_nongam_clean | 0.750 | 0.667 | 0.666 | 0.272s | 0.0614s | 1.000 | 112 |
| kanboost_gam_clean_select20 | 0.625 | 0.583 | 0.653 | 0.051s | 0.0235s | 0.750 | 20 |
| kanboost_gam_clean | 0.625 | 0.500 | 0.663 | 0.616s | 0.0677s | 1.000 | 112 |
| kanboost_gam_clean_select10 | 0.500 | 0.417 | 0.644 | 0.035s | 0.0161s | 0.750 | 10 |
| xgboost_clean | 0.500 | 0.333 | 0.693 | 0.027s | 0.0038s | 0.500 | 112 |
| kanboost_nongam_clean_select20 | 0.375 | 0.250 | 0.703 | 0.122s | 0.0210s | 0.750 | 20 |

Interpretation:
- Cleaning constant columns removes the previous sklearn warnings and changes
  the small-smoke ranking: full-feature `kanboost_nongam_clean` now ties
  CatBoost on balanced accuracy/F1 and has slightly better log loss, but is
  slower at prediction. This is not enough for a scientific claim because
  n=8 makes fold variance enormous, but it is strong evidence that the
  medium/large OpenNeuro notebook must include leakage-safe constant-feature
  filtering before any model comparison.
- `kanboost_gam_clean_select20` remains the best low-cost KANBoost variant
  by runtime/log-loss tradeoff among feature-selected arms, but does not
  close the balanced-accuracy gap on the tiny smoke.
- `SelectKBest(k=10)` is too aggressive in this smoke; it lowers balanced
  accuracy to 0.50 despite good log loss. Keep k=20 and full-feature arms
  for the next run; optionally add k=40 only on medium/large where folds are
  large enough.
- The main real-data gap is now more precise: KANBoost may be competitive
  with CatBoost after basic fold-safe feature hygiene, but this must be
  tested on medium/large subject counts before acceptance. The next
  OpenNeuro sequential notebook should add `VarianceThreshold` inside every
  model pipeline and report feature counts per fold.

Decision: keep Proposal 6 open. This clean smoke is positive enough to
justify the medium/large remote run, but it does not satisfy the predefined
quality gate by itself and does not replace the required double-experiment
comparison.

**[Codex Review, 2026-07-22] OpenNeuro clean medium/large notebook protocol
prepared.**

Codex updated the OpenNeuro sequential notebook generator and regenerated
both notebook targets:
- `remote/generate_openneuro_sequential_notebook.py`
- `remote/kaggle_openneuro_bench/openneuro_ds004504_model_benchmark.ipynb`
- `remote/colab_openneuro_bench/openneuro_ds004504_model_benchmark.ipynb`

Protocol changes:
- Add fold-local `SimpleImputer(strategy="median") ->
  VarianceThreshold(threshold=0.0)` to every model pipeline, including
  CatBoost/XGBoost/RBF-KAN/KANBoost, so the constant-feature issue found in
  the small smoke cannot recur silently.
- Add `n_features_after_fold_transform` to per-fold metrics and
  `mean_features_after_fold_transform` to summaries.
- Rename quality arms with `_clean` suffix so new results are not confused
  with the earlier raw-feature small run.
- Add KANBoost variants required by Proposal 6:
  `kanboost_project_clean` (`kan_hidden=3`, non-GAM),
  `kanboost_gam_cached_clean` (`gam=True`, `kan_hidden=1`), and
  `kanboost_gam_select20_clean`.
- Keep stage sequencing (`small -> medium -> large`) and timestamped output
  naming unchanged.

Validation performed:
- `py_compile remote/generate_openneuro_sequential_notebook.py` succeeded.
- Regeneration succeeded (`cells=16`).
- Both regenerated notebooks parse as JSON.
- `rg` confirmed `VarianceThreshold`, `kanboost_gam_cached_clean`, and
  `n_features_after_fold_transform` are present in both Kaggle and Colab
  notebooks.

Remote execution plan:
- To avoid clobbering the older shared OpenNeuro Kaggle kernel, Codex
  prepared a separate private Kaggle kernel folder:
  `remote/kaggle_openneuro_clean_bench`.
- Kernel id:
  `tuamamhamza/openneuro-ds004504-kanboost-clean-benchmark`.
- This run is intended to provide the next remote evidence package for
  Proposal 6 on the cleaned protocol. It still does not replace Claude local
  and Codex local results under the double-experiment rule.

**[Codex Review, 2026-07-22] OpenNeuro clean Kaggle run completed -- medium
passes, large does NOT close the best-baseline gap yet.**

Remote run:
- Kernel: `tuamamhamza/openneuro-ds004504-kanboost-clean-benchmark`.
- Timestamp: `20260721-223240`.
- Outputs:
  `remote/results/kaggle_openneuro_clean/outputs/kaggle_openneuro_ds004504_model_benchmark_ds004504_v1_0_7_eeg_*_20260721-223240_*`.
- Stages completed: small, medium, large.
- Subjects: small n=8, medium n=20, large n=65.
- Features: 115 raw, 112 after fold-local cleaning for small/medium,
  113 after fold-local cleaning for large.
- Local sides: still incomplete under the double-experiment protocol;
  Claude local and Codex local for this exact clean notebook are not yet
  recorded. Treat this as remote evidence, not the final three-way package.

Stage summaries by balanced accuracy:

| stage | best model | best bal acc | best F1 | best log loss | best fit/predict | best KANBoost arm | KANBoost bal acc | KANBoost gap |
|---|---|---:|---:|---:|---|---|---:|---:|
| small | catboost_clean | 0.750 | 0.667 | 0.683 | 0.101s / 0.0045s | kanboost_gam_cached_clean / kanboost_gam_select20_clean | 0.625 | -0.125 |
| medium | catboost_clean / xgboost_clean / kanboost_gam_cached_clean | 0.800 | 0.793 | 0.558 / 0.472 / 0.568 | KANBoost 0.923s / 0.095s | kanboost_gam_cached_clean | 0.800 | 0.000 |
| large | hist_gbdt_clean | 0.749 | 0.751 | 0.712 | 0.137s / 0.010s | kanboost_project_clean | 0.682 | -0.067 |

Full large-stage ordering:
- `hist_gbdt_clean`: balanced acc 0.749, F1 0.751, log loss 0.712.
- `rbf_kan_torch_clean`: balanced acc 0.732, F1 0.730, log loss 1.494.
- `logreg_clean`: balanced acc 0.700, F1 0.697, log loss 0.906.
- `kanboost_project_clean`: balanced acc 0.682, F1 0.675, log loss 0.619.
- `random_forest_clean`: balanced acc 0.664, F1 0.662, log loss 0.596.
- `xgboost_clean`: balanced acc 0.652, F1 0.646, log loss 0.788.
- `catboost_clean`: balanced acc 0.640, F1 0.634, log loss 0.877.
- `kanboost_gam_select20_clean`: balanced acc 0.625, F1 0.619, log loss 1.041.
- `kanboost_gam_cached_clean`: balanced acc 0.549, F1 0.541, log loss 0.772.

Interpretation:
- The clean protocol matters: constant-column filtering is now present in
  every model pipeline and the notebook reports the post-transform feature
  count, so the earlier `SelectKBest` warning is closed.
- Medium-stage evidence is genuinely encouraging: `kanboost_gam_cached_clean`
  ties the best balanced accuracy/F1 at 0.80 and is within normal model
  ranking noise, although it is slower than CatBoost/XGBoost.
- Large-stage evidence does not satisfy Proposal 6's quality gate. The best
  KANBoost arm (`kanboost_project_clean`, non-GAM) is 0.067 balanced-accuracy
  behind the best baseline (`hist_gbdt_clean`), missing the predefined
  "within 0.05" gate. Therefore KANBoost has NOT yet closed the real-data
  accuracy gap on the largest OpenNeuro stage.
- The best KANBoost large-stage arm is non-GAM, not GAM. Proposal 2's GAM
  cache helps make GAM cheap enough to test, but GAM itself is not the
  winning quality shape here.
- KANBoost's large-stage log loss (0.619) is better than the best-balanced
  HistGBDT log loss (0.712), so the remaining gap is not a total failure:
  KANBoost may be better calibrated/smoother while losing balanced accuracy.
  The next quality work should examine thresholding/calibration or
  class-balanced decision thresholds, not only raw model capacity.
- Operational issue found: because `DATA_DIR` was under `/kaggle/working`,
  `kaggle kernels output` downloaded raw EEG files as outputs. Codex patched
  the notebook generator after this run so future Kaggle notebooks use
  `/kaggle/temp/openneuro_data` for raw/cache data while keeping only
  `/kaggle/working/outputs` as final outputs.

Decision: Proposal 6 remains open/inconclusive, but the large-stage result
rejects the current clean KANBoost variants as a complete accuracy fix. The
next candidate must target the remaining large-stage gap directly: decision
threshold calibration, class-balanced validation thresholds, richer EEG
features, or a KANBoost configuration that improves balanced accuracy
without sacrificing the currently favorable log-loss behavior.

**[Codex Review, 2026-07-22] Proposal 7 -- inner-CV threshold tuning on
OpenNeuro large features: REJECT threshold tuning; keep 0.5 for KANBoost
unless a stronger calibration protocol is proposed.**

Hypothesis: KANBoost's large-stage log loss was better than the
best-balanced HistGBDT while balanced accuracy lagged. Maybe the probability
ranking/calibration was acceptable but the default 0.5 decision threshold
was poorly matched to balanced accuracy. Test a leakage-safe threshold:
inside each outer training fold, build inner out-of-fold probabilities,
select the threshold maximizing balanced accuracy, then apply that threshold
to the untouched outer validation fold.

Remote run:
- Kernel: `tuamamhamza/kanboost-proposal7-openneuro-threshold`.
- Dataset: `tuamamhamza/openneuro-ds004504-large-clean-features`.
- Timestamp: `20260721-230125`.
- Outputs:
  `remote/results/kaggle_proposal7_threshold/kaggle_kanboost-proposal7-openneuro-threshold_ds004504_large_clean_features_threshold_20260721-230125_*`.
- Version 1 failed because Kaggle script kernels do not include sidecar CSV
  files. Fixed by creating the small Kaggle Dataset above and linking it in
  `kernel-metadata.json`; version 2 completed.
- Protocol: outer 5-fold StratifiedKFold, inner 3-fold OOF threshold
  selection on the outer-train subjects only, no raw EEG re-download.
- Local sides: not a complete double-experiment package; this is Codex
  remote evidence only.

Summary:

| model | decision | bal acc | F1 | log loss | AUC | mean threshold |
|---|---|---:|---:|---:|---:|---:|
| hist_gbdt_clean | default 0.5 | 0.749 | 0.751 | 0.712 | 0.770 | 0.500 |
| hist_gbdt_clean | inner threshold | 0.749 | 0.751 | 0.712 | 0.770 | 0.475 |
| kanboost_project_clean | default 0.5 | 0.702 | 0.691 | 0.617 | 0.747 | 0.500 |
| kanboost_project_clean | inner threshold | 0.571 | 0.531 | 0.617 | 0.747 | 0.422 |
| kanboost_gam_cached_clean | default 0.5 | 0.549 | 0.541 | 0.772 | 0.556 | 0.500 |
| kanboost_gam_cached_clean | inner threshold | 0.550 | 0.532 | 0.772 | 0.556 | 0.506 |
| xgboost_clean | default 0.5 | 0.652 | 0.646 | 0.788 | 0.727 | 0.500 |
| xgboost_clean | inner threshold | 0.661 | 0.657 | 0.788 | 0.727 | 0.434 |
| catboost_clean | default 0.5 | 0.640 | 0.634 | 0.877 | 0.727 | 0.500 |
| catboost_clean | inner threshold | 0.652 | 0.633 | 0.877 | 0.727 | 0.331 |

Interpretation:
- For `kanboost_project_clean`, inner-CV thresholding is actively harmful:
  balanced accuracy drops from 0.702 to 0.571. The threshold selected from
  the outer training fold does not generalize to validation, probably because
  n=52 inside each outer train fold is too small/noisy for threshold search.
- `kanboost_gam_cached_clean` is unchanged and still weak on large features.
- HistGBDT is unchanged; XGBoost/CatBoost gain only small amounts and remain
  below HistGBDT.
- The default `kanboost_project_clean` rerun here scores 0.702, closer to
  HistGBDT's 0.749 than the full notebook's 0.682. This run installs a fixed
  Git commit (`11f4b0fd95ac40aa475702a81d5efbb749cee55b`) whereas the full
  notebook installed `kanboost` by package name. Therefore, do not treat the
  0.702 vs 0.682 difference as a scientific improvement until the full
  notebook is rerun with a pinned KANBoost commit/version.

Decision:
- Reject Proposal 7's threshold-tuning idea as a fix for KANBoost balanced
  accuracy on OpenNeuro large.
- Keep default 0.5 threshold for now.
- Next protocol hardening: pin the exact KANBoost commit/version in all
  OpenNeuro notebooks before comparing model-quality numbers, then rerun the
  clean large benchmark or a features-only equivalent. If the pinned default
  `kanboost_project_clean` reproducibly stays around 0.70, it is within
  ~0.05 of HistGBDT while retaining better log loss, making the remaining
  claim much stronger.

**[Codex Review, 2026-07-22] Protocol hardening applied after Proposal 7.**

Codex updated the OpenNeuro notebook generator so future OpenNeuro runs pin
KANBoost to the same exact commit used by the successful remote speed/quality
experiments:
`11f4b0fd95ac40aa475702a81d5efbb749cee55b`.

Files regenerated/updated:
- `remote/generate_openneuro_sequential_notebook.py`
- `remote/kaggle_openneuro_bench/openneuro_ds004504_model_benchmark.ipynb`
- `remote/colab_openneuro_bench/openneuro_ds004504_model_benchmark.ipynb`
- `remote/kaggle_openneuro_clean_bench/openneuro_ds004504_model_benchmark.ipynb`

Validation:
- `py_compile remote/generate_openneuro_sequential_notebook.py` succeeded.
- All three notebooks parse as JSON.
- `rg` confirms each notebook contains:
  `KANBOOST_COMMIT = "11f4b0fd95ac40aa475702a81d5efbb749cee55b"`,
  `pip_install(f"git+https://github.com/tuamah/kanboost.git@{KANBOOST_COMMIT}")`,
  `kanboost_commit` in stage/all-stage/manifest outputs,
  `/kaggle/temp` for raw OpenNeuro data cache, and
  `kanboost_gam_cached_clean` for Proposal 6 quality testing.

Decision: any new OpenNeuro model-quality conclusion should use these pinned
notebooks or a features-only script that installs the same commit. Older
OpenNeuro notebook results that installed `kanboost` by package name remain
useful context but are weaker evidence for precise model ranking.

**[Codex Review, 2026-07-22] Pinned large-feature evidence extracted from
Proposal 7 default arms.**

Although Proposal 7 rejected threshold tuning, its `default_0p5` rows are a
valid pinned-commit, features-only rerun on the OpenNeuro large feature
table. This partially satisfies the protocol-hardening step above without
rerunning raw EEG extraction:
- Same large feature CSV from the clean OpenNeuro notebook.
- Same outer 5-fold StratifiedKFold seed.
- Same fold-local cleaning pipeline.
- KANBoost installed from pinned commit
  `11f4b0fd95ac40aa475702a81d5efbb749cee55b`.

Pinned large-feature default-threshold ranking:

| model | bal acc | F1 | log loss | AUC | mean fit | mean predict |
|---|---:|---:|---:|---:|---:|---:|
| hist_gbdt_clean | 0.749 | 0.751 | 0.712 | 0.770 | 0.126s | 0.0051s |
| kanboost_project_clean | 0.702 | 0.691 | 0.617 | 0.747 | 0.694s | 0.0672s |
| logreg_clean | 0.700 | 0.697 | 0.906 | 0.740 | 0.0127s | 0.0018s |
| random_forest_clean | 0.681 | 0.678 | 0.596 | 0.765 | 0.614s | 0.0698s |
| xgboost_clean | 0.652 | 0.646 | 0.788 | 0.727 | 0.107s | 0.0024s |
| catboost_clean | 0.640 | 0.634 | 0.877 | 0.727 | 0.432s | 0.0026s |
| kanboost_gam_cached_clean | 0.549 | 0.541 | 0.772 | 0.556 | 0.259s | 0.0767s |

Interpretation:
- With the pinned commit, `kanboost_project_clean` is 0.047 balanced-accuracy
  behind `hist_gbdt_clean`, which is within Proposal 6's predefined
  "within 0.05 of best baseline" gate on large features.
- KANBoost also has better log loss than HistGBDT (0.617 vs 0.712), matching
  the earlier observation that KANBoost's probability estimates are
  competitive even when raw balanced accuracy lags.
- KANBoost is still slower than HistGBDT on fit and prediction in this
  feature-level benchmark, so the claim is model-quality competitiveness,
  not speed superiority.
- The GAM KANBoost arm remains weak on large features; the winning KANBoost
  shape is still the non-GAM `kan_hidden=3` project arm.

Decision: update Proposal 6 status from "large-stage quality gap not closed"
to "large-stage quality gap is plausibly closed under the pinned-commit
features-only rerun, but not yet final." It still needs Claude local/Codex
local comparison or a rerun of the pinned notebook before being promoted as
the final OpenNeuro conclusion under the double-experiment rule.

**User note (2026-07-22)**: Proposal 5 is accepted by the user for later
Colab testing. Keep Proposal 5 separate from OpenNeuro model-quality work:
it is an interpretability/reporting-speed improvement, not a training or
accuracy fix.

**[Codex Review, 2026-07-22] Proposal 5 trial version bump prepared**:
Per the user's request to prepare the approved first-layer interpretability
cache for self-testing, Codex bumped local package metadata from `1.2.3` to
`1.2.4` in `pyproject.toml` and `kanboost/__init__.py`. Scope is version
metadata only; the already-applied Proposal 5 code path in
`kanboost/core/base.py` is unchanged by this Codex edit. Verification:
`rg -n "version =|__version__" pyproject.toml kanboost/__init__.py`
shows both values at `1.2.4`. Full import tests were not run in this local
shell: no `python` executable is on PATH, and the bundled Codex Python used
for checks lacks project ML dependencies such as `sklearn`. Do not treat
this as published to PyPI/GitHub or merged; it is a local release-candidate
version bump ready for explicit commit/push/publish approval and external
reruns.

**[Codex Review, 2026-07-22] Proposal 5 `1.2.4` wheel + Colab path
prepared**: Codex built a local wheel with
`python.exe -X utf8 -m pip wheel . --no-deps --no-build-isolation -w dist`
after the isolated build failed because network access was blocked while
fetching `setuptools`. Artifact:
`dist/kanboost-1.2.4-py3-none-any.whl`, SHA256
`ec8f82de72fca58af349cdb7d5ba66328a61b1c19213cd1346ae38fce0ad701c`.
Updated `remote/colab_proposal5/kanboost_proposal5_interpret_cache.ipynb`
so the default install path is: upload that wheel in Colab, then run
`pip install --force-reinstall ./kanboost-1.2.4-py3-none-any.whl`. The
notebook records `kanboost_source='wheel-1.2.4-rc'` in its environment
JSON. Verification: notebook JSON loads successfully under bundled Python.
This gives the user a concrete self-test artifact without publishing or
pushing a release.

**[Codex Review, 2026-07-22] Proposal 5 `1.2.4` wheel Kaggle rerun --
PASS**: Codex created a Kaggle Dataset for the wheel
(`tuamamhamza/kanboost-1-2-4-proposal5-wheel`) and an isolated kernel
`tuamamhamza/kanboost-proposal5-wheel124-interpret-profile`, source under
`remote/kaggle_proposal5_wheel124/`. Version 1 failed because Kaggle did
not copy the `.whl` file into `/kaggle/src`; fixed by attaching the wheel
as a Dataset. Version 2 completed but used the old prototype-vs-production
comparison, which is not the right speed interpretation after the cache is
already inside production `1.2.4`. Version 3 corrected the protocol:
production `1.2.4` methods are compared against explicit uncached
references.

Outputs downloaded under `remote/results/kaggle_proposal5_wheel124/` with
prefix
`kaggle_kanboost-proposal5-wheel124-interpret-profile_synthetic_friedman_20260721-234540`.
Environment confirms `kanboost_version=1.2.4`,
`install_source=wheel-1.2.4-rc`, and
`wheel=kanboost-1.2.4-py3-none-any.whl`.

Wheel `1.2.4` corrected Kaggle results:

| benchmark | n_train | n_eval | n_estimators | kan_hidden | speedup | max abs diff |
|---|---:|---:|---:|---:|---:|---:|
| `feature_contributions` | 2000 | 2000 | 40 | 16 | 26.62x | 8.9e-16 |
| `feature_contributions` | 2000 | 5000 | 80 | 16 | 46.44x | 1.6e-15 |
| `feature_contributions` | 3000 | 5000 | 80 | 32 | 87.68x | 3.1e-15 |
| `predict_derivative_gam` | 2000 | 2000 | 40 | 1 | 5.52x | 0.0 |
| `predict_derivative_gam` | 2000 | 5000 | 80 | 1 | 10.81x | 0.0 |

Decision: Proposal 5 remains accepted and is now validated specifically as
the `1.2.4` wheel artifact the user can test in Colab. The improved
feature-contribution speedups are larger than the earlier prototype-profile
numbers because the corrected wheel protocol compares production `1.2.4`
against an uncached reference, not against another cached prototype.

### Proposals explicitly not recommended for another round now

- Do not retry Cholesky as a drop-in replacement for `eigh`; prior evidence
  showed silent numerical failure on near-singular ALS systems.
- Do not promote cross-fold parallelism as a default on this 8 GB Windows
  machine; previous tests missed the >=20% speed gate and violated the
  user's RAM reserve rule, including one real `ArrayMemoryError`.
- Do not pursue a broad C++ backend unless a newly profiled hot loop is
  genuinely loop-bound and not already BLAS/Numba dominated.
- Do not reopen depth-3 residual correction or bottleneck reshaping for
  speed; the current ledger closed those paths for production.
- Do not make `fast_fit()` the default. It remains opt-in because prior CV
  showed a material accuracy loss on Friedman-1000 despite a strong speedup
  on California Housing.

### Remote Kaggle execution path for Claude

**Purpose**: run heavier benchmarks remotely so the local 8 GB Windows
machine is not the limiting resource, while still preserving the project's
evidence standard: exact commands, reproducible scripts, downloaded output
files, and `AI_REVIEW_LOOP.md` updates after each run.

**Current local Kaggle setup (verified 2026-07-21)**:
- Kaggle API access token is stored locally at
  `C:\Users\tom-G4\.kaggle\access_token`.
- Do not print, copy, or commit the token. If it appears in logs or chat,
  revoke it in Kaggle and generate a new one.
- Kaggle CLI is installed in Codex's bundled Python runtime, not in the
  system `PATH`.
- The token and CLI are both required:
  - the token is the credential/permission;
  - the Kaggle CLI is the client that sends commands to Kaggle using that
    token automatically.
  Do not pass the token on the command line. The CLI reads it from
  `C:\Users\tom-G4\.kaggle\access_token`.
- Use this executable form from the repo root:

```powershell
& 'C:\Users\tom-G4\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m kaggle competitions list
```

This command was tested successfully after allowing network access to
`api.kaggle.com`.

**Important execution detail for Claude/Codex**:
- Plain `kaggle ...` is not available on `PATH` in this environment.
- Plain `python ...` is also not available on `PATH`.
- Always call Kaggle through the bundled Python executable:

```powershell
& 'C:\Users\tom-G4\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m kaggle <subcommand>
```

- Network access to Kaggle is blocked by the sandbox unless the command is
  run with approval/escalation. If a Kaggle command fails with
  `WinError 10013` or cannot reach `api.kaggle.com`, rerun the same command
  with the approved Kaggle CLI prefix and network escalation.
- Verified authentication command:

```powershell
& 'C:\Users\tom-G4\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m kaggle competitions list
```

- Verified owner/kernel namespace from `kernels list --mine`:
  `tuamamhamza`.

**Recommended remote-run workflow**:
1. Create a dedicated folder such as `remote/kaggle_speed_bench/`.
2. Put the Kaggle kernel source there (`.py` or `.ipynb`) plus
   `kernel-metadata.json`.
3. The remote script should install or vendor the exact project state being
   tested, run one proposal at a time, and write machine-readable outputs:
   `kaggle_<notebook>_<dataset>_<YYYYMMDD-HHMMSS>_results.json`,
   `kaggle_<notebook>_<dataset>_<YYYYMMDD-HHMMSS>_metrics.csv`,
   `kaggle_<notebook>_<dataset>_<YYYYMMDD-HHMMSS>_profile.txt`,
   `kaggle_<notebook>_<dataset>_<YYYYMMDD-HHMMSS>_environment.txt`, and
   `kaggle_<notebook>_<dataset>_<YYYYMMDD-HHMMSS>_run_log.txt`.
   Use a single timestamp per run so all files from the same run sort
   together.
4. Push/run the kernel from that folder:

```powershell
& 'C:\Users\tom-G4\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m kaggle kernels push -p remote/kaggle_speed_bench
```

5. Download the latest outputs after completion:

```powershell
& 'C:\Users\tom-G4\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m kaggle kernels output <owner/kernel-slug> -p remote/results/kaggle -o
```

6. Record the downloaded metrics, logs, hardware/runtime details, and any
   failures back into this ledger before drawing conclusions.

**Rules for remote evidence**:
- Treat Kaggle as a different hardware environment, not as a direct
  replacement for local timing. Compare remote before/after runs only
  against other remote runs with the same kernel image, resource type,
  seeds, folds, and data.
- Prefer CPU runs for the current DeepKAN/KANBoost speed questions unless
  a proposal explicitly uses GPU-accelerated libraries; the production
  bottlenecks are mostly NumPy/SciPy/BLAS/B-spline/eigh work.
- Do not rely on notebook screenshots. Every run must emit files that can
  be downloaded and reviewed locally.
- Do not commit Kaggle credentials, downloaded private data, or large output
  artifacts unless the user explicitly approves.
- Keep Kaggle outputs under `remote/results/kaggle/` (or, if a shared
  output folder is used, every filename must start with `kaggle_`) so they
  cannot be confused with Colab outputs.
- Each Kaggle result filename must include the notebook/kernel slug, dataset
  identifier, and run timestamp. Example:
  `kaggle_kanboost-speed-bench_friedman1000_20260721-153012_metrics.csv`.

### Remote Google Colab execution path for Claude

**Purpose**: keep a second remote-execution route available for interactive
experiments, GPU checks, or manual notebook runs. Colab is useful when the
user wants to inspect a notebook live, but it is less automation-friendly
than Kaggle because runs are normally controlled through the browser rather
than a simple push/output CLI loop.

**Recommended Colab workflow**:
1. Create a notebook under a dedicated folder such as
   `remote/colab_speed_bench/kanboost_remote_bench.ipynb`.
2. The first notebook cells must be fully self-contained:
   - print runtime information (`python`, `platform`, CPU/GPU, RAM);
   - clone or upload the exact project state to test;
   - install pinned dependencies;
   - set seeds and benchmark parameters explicitly.
3. The notebook must write the same reviewable outputs as Kaggle:
   `colab_<notebook>_<dataset>_<YYYYMMDD-HHMMSS>_results.json`,
   `colab_<notebook>_<dataset>_<YYYYMMDD-HHMMSS>_metrics.csv`,
   `colab_<notebook>_<dataset>_<YYYYMMDD-HHMMSS>_profile.txt`,
   `colab_<notebook>_<dataset>_<YYYYMMDD-HHMMSS>_environment.txt`, and
   `colab_<notebook>_<dataset>_<YYYYMMDD-HHMMSS>_run_log.txt`.
   Use a single timestamp per run so all files from the same run sort
   together.
4. The user can open the notebook in Google Colab and run all cells.
5. After the run, download the output files into the repo, preferably under
   `remote/results/colab/`, then record the evidence here.

**Colab-specific rules**:
- Do not treat a Colab result as directly comparable to local Windows or
  Kaggle timing. Compare Colab before/after only against Colab before/after
  with the same runtime type.
- Prefer CPU runtime for current DeepKAN/KANBoost speed questions unless the
  experiment explicitly targets GPU acceleration. The known production hot
  spots are mostly NumPy/SciPy/BLAS/B-spline/eigh work, so GPU availability
  alone does not guarantee a fair or useful speedup.
- Do not rely on screenshots or notebook cell outputs as the only evidence.
  Persist files and bring them back into the repo for Codex/ChatGPT review.
- Avoid storing Kaggle, GitHub, Google Drive, or other credentials in the
  notebook. Use Colab secrets or manual upload when credentials are
  unavoidable, and never commit them.
- If the notebook is shared, remember that Colab shares code and saved cell
  outputs, not the active VM state. Keep setup cells complete enough for a
  fresh runtime.
- Keep Colab outputs under `remote/results/colab/` (or, if a shared output
  folder is used, every filename must start with `colab_`) so they cannot
  be confused with Kaggle outputs.
- Each Colab result filename must include the notebook name, dataset
  identifier, and run timestamp. Example:
  `colab_kanboost_remote_bench_friedman1000_20260721-153012_results.json`.

**Decision guidance**:
- Use Kaggle first for automated benchmark loops and downloadable outputs.
- Use Colab when manual inspection, GPU experiments, or quick interactive
  iteration is more important than automation.

### Remote execution smoke test -- Kaggle

**Status (2026-07-21)**: Kaggle remote execution path verified with a small
NumPy-only synthetic regression smoke test.

**Kernel**: `tuamamhamza/kanboost-remote-smoke`
(`https://www.kaggle.com/code/tuamamhamza/kanboost-remote-smoke`).

**Local source files**:
- `remote/kaggle_speed_bench/kernel-metadata.json`
- `remote/kaggle_speed_bench/kanboost_remote_smoke.py`

**Downloaded Kaggle outputs**: stored under `remote/results/kaggle/` with
the shared run prefix
`kaggle_kanboost-remote-smoke_synthetic_regression_n200_p8_20260721-202434`.
Files downloaded:
- `..._results.json`
- `..._metrics.csv`
- `..._profile.txt`
- `..._environment.txt`
- `..._run_log.txt`
- `kanboost-remote-smoke.log`

**Remote environment evidence**:
- Python: 3.12.13
- Platform: Linux 6.12.90+ x86_64
- NumPy: 2.0.2
- scikit-learn: not required for this smoke test

**Smoke-test result**:
- Dataset: `synthetic_regression_n200_p8`
- CV: 5 folds, seed 42
- Mean R2: 0.9944519576
- Std R2: 0.0012157334
- Overall R2: 0.9945982602
- Overall MSE: 0.0456375755
- Script elapsed time inside Kaggle: 0.1107762420 seconds

**Interpretation**: This is not a KANBoost performance benchmark. It only
verifies the remote execution loop end-to-end: push kernel, run on Kaggle,
download timestamped result files, read logs locally. The next remote run
can replace this smoke script with the first real speed proposal benchmark.

### Kaggle remote-run runbook -- exact working sequence

This is the concrete sequence verified in the smoke test above. Claude can
reuse it for future experiments by replacing the smoke script with a real
benchmark script and keeping the same artifact discipline.

**0. Credential setup (already done locally)**:
- The Kaggle API token was saved to:
  `C:\Users\tom-G4\.kaggle\access_token`
- The token is not passed in commands and must never be printed or
  committed. Kaggle CLI reads this file automatically.
- The Kaggle CLI package is installed in the bundled Codex Python runtime.

**1. Verify account/API connectivity**:

```powershell
& 'C:\Users\tom-G4\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m kaggle competitions list
```

Expected: a competitions table. In the verified run this required network
approval because sandboxed network access raised `WinError 10013`.

**2. Discover the Kaggle owner namespace**:

```powershell
& 'C:\Users\tom-G4\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m kaggle kernels list --mine
```

Verified owner namespace: `tuamamhamza`.

**3. Prepare a kernel folder in the repo**:

Smoke-test folder used:
`remote/kaggle_speed_bench/`

Required files:
- `kernel-metadata.json`
- one Python script or notebook referenced by `code_file`

Verified smoke-test metadata:

```json
{
  "id": "tuamamhamza/kanboost-remote-smoke",
  "title": "kanboost remote smoke",
  "code_file": "kanboost_remote_smoke.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": false,
  "enable_internet": false,
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": []
}
```

**4. Write the experiment script so Kaggle emits downloadable files**:
- Save outputs to `/kaggle/working` when running on Kaggle.
- Use the required filename convention:
  `kaggle_<notebook>_<dataset>_<YYYYMMDD-HHMMSS>_<artifact>.<ext>`
- Use one timestamp for all artifacts from the same run.
- Required artifacts:
  `_results.json`, `_metrics.csv`, `_profile.txt`, `_environment.txt`,
  `_run_log.txt`.
- Print the JSON summary to stdout as an extra check; Kaggle also downloads
  `<kernel>.log`.

**5. Optional local syntax/smoke check before upload**:

```powershell
& 'C:\Users\tom-G4\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' remote\kaggle_speed_bench\kanboost_remote_smoke.py
```

Note: local outputs are not Kaggle evidence. If local validation writes
files, move them to `remote/results/local_validation/` so they do not mix
with real Kaggle outputs.

**6. Push/run the Kaggle kernel**:

```powershell
& 'C:\Users\tom-G4\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m kaggle kernels push -p remote\kaggle_speed_bench
```

Verified smoke-test response:
`Kernel version 1 successfully pushed. Please check progress at
https://www.kaggle.com/code/tuamamhamza/kanboost-remote-smoke`

**7. Check execution status**:

```powershell
& 'C:\Users\tom-G4\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m kaggle kernels status tuamamhamza/kanboost-remote-smoke
```

Verified smoke-test status:
`KernelWorkerStatus.COMPLETE`

**8. Download Kaggle outputs into the dedicated results folder**:

```powershell
& 'C:\Users\tom-G4\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m kaggle kernels output tuamamhamza/kanboost-remote-smoke -p remote\results\kaggle -o
```

Verified downloaded files:
- `kaggle_kanboost-remote-smoke_synthetic_regression_n200_p8_20260721-202434_results.json`
- `kaggle_kanboost-remote-smoke_synthetic_regression_n200_p8_20260721-202434_metrics.csv`
- `kaggle_kanboost-remote-smoke_synthetic_regression_n200_p8_20260721-202434_profile.txt`
- `kaggle_kanboost-remote-smoke_synthetic_regression_n200_p8_20260721-202434_environment.txt`
- `kaggle_kanboost-remote-smoke_synthetic_regression_n200_p8_20260721-202434_run_log.txt`
- `kanboost-remote-smoke.log`

**9. Read and record evidence locally**:
- Read `*_results.json` for summary metrics.
- Read `*_metrics.csv` for fold-level metrics.
- Read `*_environment.txt` for remote environment details.
- Read `<kernel>.log` for stdout/stderr and conversion/runtime messages.
- Summarize the result in this ledger before interpreting it.

**10. For the next real benchmark**:
- Keep the same folder/result discipline.
- Change `id`, `title`, `code_file`, notebook name, and dataset identifier
  if the experiment changes materially.
- Compare only Kaggle-before vs Kaggle-after runs with the same kernel
  image/resource type. Do not compare Kaggle timing directly against local
  Windows timing.

### Remote benchmark notebook -- OpenNeuro ds004504

**Status (Kaggle small-stage smoke executed successfully)**: created a
Colab/Kaggle-ready notebook for benchmarking KANBoost against CatBoost,
XGBoost, classical baselines, and a second KAN-like PyTorch RBF-KAN model on
OpenNeuro `ds004504` version `1.0.7`. Kaggle kernel
`tuamamhamza/openneuro-ds004504-kanboost-benchmark` version 7 completed on
2026-07-21 UTC with `RUN_STAGE_MEDIUM=False` and `RUN_STAGE_LARGE=False`
(small stage only).

**Dataset**:
- Source: `https://openneuro.org/datasets/ds004504/versions/1.0.7`
- Task: resting-state eyes-closed EEG.
- Primary benchmark task in notebook: AD vs cognitively normal control
  (`Group` A vs C), excluding FTD by default. A three-class mode is
  available but should be interpreted cautiously because the dataset is
  small.

**Notebook files**:
- Colab path:
  `remote/colab_openneuro_bench/openneuro_ds004504_model_benchmark.ipynb`
- Kaggle path:
  `remote/kaggle_openneuro_bench/openneuro_ds004504_model_benchmark.ipynb`
- Kaggle metadata:
  `remote/kaggle_openneuro_bench/kernel-metadata.json`
- Kaggle kernel id:
  `tuamamhamza/openneuro-ds004504-kanboost-benchmark`

**Model comparison in notebook**:
- Logistic regression baseline.
- HistGradientBoosting baseline.
- RandomForest baseline.
- CatBoost.
- XGBoost.
- Project model: `KANBoostClassifier`.
- Second KAN-like comparator: `RBFKANClassifier`, a compact PyTorch
  RBF-expanded KAN-style classifier implemented inside the notebook.

**Feature protocol**:
- Download selected BIDS EEGLAB files from OpenNeuro. In Kaggle version 7,
  `openneuro-py` reported completion but did not materialize the subject
  `.set` files, so the notebook now has a direct public S3 fallback for the
  expected BIDS EEGLAB files under
  `https://s3.amazonaws.com/openneuro.org/ds004504/sub-*/eeg/`. This fallback
  was used in the successful small smoke run; record this caveat when
  comparing to strict historical-snapshot runs.
- Read EEG with MNE.
- Filter 0.5-45 Hz, resample to 250 Hz.
- In smoke mode, crop recordings to 120 seconds and use a small balanced
  subject subset.
- Extract subject-level band-power features only. No random window-level
  split is used, preventing same-subject leakage across folds.
- Demographics are excluded by default. If enabled later, treat that as a
  separate confound-aware experiment.

**Metrics/output protocol**:
- Subject-level stratified CV.
- Metrics: accuracy, balanced accuracy, macro F1, log loss, ROC AUC for
  binary mode, fit time, prediction time.
- Output filename convention follows the remote-run rules:
  `<platform>_openneuro_ds004504_model_benchmark_ds004504_v1_0_7_eeg_<stage>_<YYYYMMDD-HHMMSS>_<artifact>.<ext>`.
- Artifacts for each stage include `features.csv`, `metrics.csv`,
  `summary.csv`, `results.json`, `profile.txt`, and `run_log.txt`.
- A final cross-stage comparison is also saved with the synthetic stage
  name `all_stages`, including combined `summary.csv`, `metrics.csv`,
  `results.json`, `environment.txt`, and `run_log.txt`.
- At the very end, the notebook writes a complete delivery bundle:
  - `<platform>_openneuro_ds004504_model_benchmark_ds004504_v1_0_7_eeg_<YYYYMMDD-HHMMSS>_complete_manifest.json`
  - `<platform>_openneuro_ds004504_model_benchmark_ds004504_v1_0_7_eeg_<YYYYMMDD-HHMMSS>_complete_results.zip`
  The manifest lists every generated result file and byte size. The zip
  includes the stage outputs, `all_stages` outputs, and the manifest, so
  Claude/Codex can download one archive and review the full run.

**Sequential sample-size protocol**:
- The notebook now runs as three explicit cells/stages:
  1. `small`: balanced mini subset (`n_per_group=4`), 60-second EEG crop.
  2. `medium`: larger balanced subset (`n_per_group=10`), 180-second EEG
     crop.
  3. `large`: all eligible subjects for the selected task, full recording by
     default (`max_seconds=None`; Claude may cap it if remote runtime is too
     high, but must record the cap).
- Each stage trains and compares all listed models independently, writes its
  own timestamped files, and displays a per-stage summary before the next
  stage starts.
- This makes the evidence sequential: first verify the pipeline on small
  data, then observe scaling on medium data, then run the main large
  benchmark.

**Kaggle small-stage result (version 7, timestamp `20260721-210717`)**:
- Run configuration: `AD_vs_Control`, 8 subjects total (`n_per_group=4`),
  60-second crop, 115 extracted features, 4-fold subject-level stratified CV.
- Runtime profile: feature extraction 11.71s, total model fit time 6.91s,
  total prediction time 0.43s, stage elapsed 28.54s.
- Best mean balanced accuracy: CatBoost and RBF-KAN tied at 0.75
  (std 0.289, macro F1 0.667). CatBoost was much faster
  (`mean_fit_seconds=0.101`) than RBF-KAN (`0.931`).
- Project `KANBoostClassifier`: mean balanced accuracy 0.625, macro F1 0.50,
  mean log loss 0.656, mean ROC AUC 1.0, mean fit time 0.379s. Treat the AUC
  carefully because the validation folds contain only 2 subjects each.
- Other baselines: logistic regression balanced accuracy 0.625; HistGBDT,
  RandomForest, and XGBoost were at 0.50 on this tiny smoke run.
- Downloaded result files live under `remote/results/kaggle/outputs/`, with
  prefix
  `kaggle_openneuro_ds004504_model_benchmark_ds004504_v1_0_7_eeg_20260721-210717`.
  The final bundle is
  `kaggle_openneuro_ds004504_model_benchmark_ds004504_v1_0_7_eeg_20260721-210717_complete_results.zip`,
  and the manifest is the matching `_complete_manifest.json`.
- Interpretation: this confirms the remote Kaggle execution, data download
  fallback, feature extraction, model comparison, timestamped saving, and
  result retrieval loop. It is NOT a scientific model-quality conclusion yet;
  the next valid evidence step is the medium stage, then large/all-subject
  stage, on the same Kaggle runtime.

**Before interpreting results**:
- First remote run should keep the default `small -> medium -> large`
  sequence only if the platform runtime budget is sufficient. For a pure
  smoke check, Claude may set `RUN_STAGE_MEDIUM=False` and
  `RUN_STAGE_LARGE=False`, but must record that only the small stage ran.
- Only after the small stage succeeds should medium/large results be trusted.
- Compare remote before/after runs only within the same platform/runtime.
- Do not treat this notebook as evidence until the timestamped output files
  and the final `complete_results.zip` / `complete_manifest.json` are
  downloaded and summarized in this ledger.

---

## Round 2 refinement: bottleneck shape (H2 << H1) + basis caching

Per ChatGPT's 4-axis review (residual-only -- already true; shrink the
correction to a real bottleneck; better training; reduce JVP/ALS cost via
caching) and Fable's implementation plan.

**Fable's judgment on the proposed `cache_basis` change** (before any run):
signed off as a provable exact reuse, not an approximation -- z1c/B1 are
untouched between the top of a round and the input block (C0, which
determines z1, is only written by that same block, at its end), so reusing
them is valid; z2 is still recomputed fresh via `B1 @ (updated C1)` since
C1 *did* change in the middle block. Expected floating-point drift ~1e-13
to 1e-16 from summation-order differences (GEMM vs accumulation loop), not
bit-identical -- gates below account for this. Fable also flagged a
pre-existing, unrelated latent bug: `_clamp_to_span` clamps every column to
column 0's knot span instead of each column's own span after
`_rebuild_knots` diverges per-column spans. Ruling: **do not fix now** --
it affects both cached and uncached paths identically, so today's A/B/C
comparison stays fair; scheduled as a separate follow-up with its own
before/after parity check.

**cache_basis equivalence test -- small config (H1=16, H2=8, n=500,
10 rounds)**: ALL 5 gates pass. Same rounds/blocks executed (30/30). Zero
rollback-flag flips. Max relative per-round loss diff = 3.86e-13 (gate:
<=1e-9). Max per-round parameter diff: C0=1.50e-12, C1=9.28e-13,
C2=7.25e-12 (gate: <1e-8, each). Max final-prediction diff = 1.04e-12
(gate: <1e-10). `cache_basis=True` confirmed to be an exact reuse at this
scale.

**Extended equivalence sweep** (per user request, before trusting any
A/B/C accuracy result): `verify_cache_equivalence` run across 5 additional
seed/size/shape combinations (n=300-800, H1=8-66, H2=4-8, n_rounds=10-15,
including the actual bottleneck shape H1=66/H2=8 at n=500) -- ALL 5 pass
every gate: zero rollback-flag flips in any run, max relative per-round
loss diff 1.5e-13 to 2.5e-13, max per-round parameter diff 4.7e-13 to
1.1e-11, max final-prediction diff 4.0e-13 to 6.5e-13. Combined with the
original small-config test, `cache_basis=True` is confirmed exact-reuse
safe across 6 independent configurations, not just one.

**B vs C at the real bottleneck scale (H1=66, H2=8, H_base=200 -> 12,396
params, matching A's 12,390) -- RESULT**, same seeds/folds/data as A:

*At-scale equivalence (B uncached vs C cached, 20 fold-pairs across both
functions)*: max |R2_va(B)-R2_va(C)| = 7.8e-15 (function 1) and 4.6e-15
(function 2) -- confirms exact reuse at real scale, not just the unit
tests above. Every per-fold R2, rollback_frac, and cond number is
identical between B and C to floating-point precision.

*Caching speedup (C vs B, same architecture)*: **+11.3% faster on function
1, +3.2% faster on function 2** -- real but modest, well short of a
dramatic win. Consistent with Fable's prediction that the skipped basis
rebuild is one piece of a round dominated by other costs (eigh solves, the
H1xH2 JVP loop).

*Bottleneck shape accuracy (B/C vs A, at matched ~12,390-12,396 params)*:

| function | R2_val A (H1=28,H2=25) | R2_val B/C (H1=66,H2=8) | delta | mean rollback_frac A vs B |
|---|---|---|---|---|
| exp(sin+x^2) | 0.990+-0.003 | 0.986+-0.005 | **-0.004 (worse)** | ~0.00 vs 0.117 |
| sin(radial) | 0.957+-0.016 | 0.960+-0.012 | +0.003 (roughly tied) | ~0.00 vs 0.030 |

**The aggressive bottleneck (H2=8, H1=66) does NOT clearly beat A's milder
shape (H1=28, H2=25) on accuracy** -- one function is slightly worse, the
other is a tie within noise. It also introduces real optimization
instability A didn't have: individual folds hit rollback_frac up to 0.37
and 0.43 (function 1, folds 3 and 9) -- above Fable's flagged 20%
backtracking-trigger threshold, even though the *mean* (11.7%) stays under
it. This is consistent with Fable's warning that H1=66 from only 5 inputs
creates strong z1-column collinearity in the middle-layer system.

**Verdict on this refinement round**: caching (`cache_basis=True`) is
numerically proven safe and gives a real, modest speedup (3-11%) -- kept
as a scratchpad-only optional flag, NOT promoted to any API (per
instruction, and per Fable's own bar: this is below the 5-10% "worth API
surface" threshold on one of the two functions). The aggressive bottleneck
reshaping is NOT adopted -- A's original shape (H1=28, H2=25) remains the
best-evidenced Round 2 configuration; it has no accuracy advantage here
and is measurably more rollback-unstable on some folds. No change to the
Round 2 architecture already validated at 3 scales (2,520 / 3,096 /
12,390 params).

`cache_basis` remains scratchpad-only, not wired into any API, per the
explicit instruction.

**FINAL STATUS (user-confirmed)**:
- `cache_basis`: **partially accepted** -- numerically proven safe (6/6
  equivalence configs pass, at-scale B-vs-C diff ~1e-15), but its 3-11%
  speedup is not enough to justify production integration now. Stays in
  `scratchpad` as an experimental flag. Awaiting Codex's read-only review
  of the cache_basis logic specifically (reuse-validity reasoning in
  `depth_residual_prototype.py` lines ~194-208) before this sub-path is
  considered closed.
- H1=66/H2=8 aggressive bottleneck: **rejected architecturally** -- did
  not improve accuracy (tied-to-worse) and raised rollback_frac to 43% on
  individual folds.
- **Adopted architecture stays H1=28, H2=25** (the original Round 2
  shape), validated at 2,520 / 3,096 / 12,390 params.
- **Key finding**: shrinking H2 alone is not the lever -- growing H1 to
  compensate for the parameter budget introduced collinearity among z1
  columns (only 5 raw inputs feeding 66 hidden units) and instability, not
  a net win. Reshaping the bottleneck is not the next productive move.

**Profiling (per the requested next step, before any further optimizer
change)**: single fit at the adopted config (H_base=200, H1=28, H2=25,
n=4000) profiled with `cProfile`. Total 9.51s. **The frozen depth-2 base's
own fit is 6.45s (68% of total) -- the dominant cost, not the depth-3
correction.** The correction's own `fit()` is 2.41s (25%), split
roughly evenly between `eigh` factorization (32 calls, 2.23s total,
shared with the base's internal ALS) and B-spline basis/derivative
evaluation (`_b_basis_1d`/`_b_basis_deriv_1d`, ~2.8s+1.0s, shared across
both). **Implication for the next-step plan (adaptive backtracking,
dynamic middle-layer damping, internal early stopping in the correction):
Amdahl's law caps their payoff at roughly the correction's 25% share of
wall-clock** -- these are legitimate stability/robustness improvements
worth pursuing on their own merits (rollback reduction, convergence
quality), but should not be sold as a wall-clock win over the current
9.5s/fit, since ~7/10 of that time is the already-stable, unmodified
depth-2 base.

**Next step queued (not yet implemented)**: adaptive backtracking
triggered when a block's rollback rate rises (matches Fable's ~20%
threshold guidance), dynamic damping for the middle block specifically,
and internal early-stopping refinement within the correction network --
all targeting stability/convergence quality on the ALREADY-ADOPTED H1=28,
H2=25 shape, not a new architecture search. Awaiting Codex's cache_basis
review before further work resumes.

---

## Round 2 optimizer refinement (sequenced): profiling -> early stopping -> damping -> backtracking

Per the user's exact ordering: measure first, then only add what a
measured cost justifies (burden of proof on complexity, per `CLAUDE.md`
rule 4).

**Step 1 -- fine-grained per-block profiling (adopted config, H1=28,
H2=25, H_base=200, n=4000)**: instrumented `ResidualDepth3Correction.fit`
with an opt-in `profile=True` flag (`time.perf_counter` around each named
section; verified with `verify_cache_equivalence` before and after adding
it -- identical diffs, confirming the instrumentation itself changes
nothing). Correction-only time: 2.322s of the 9.51s total single-fit time.
Breakdown:

| section | time | % of correction time |
|---|---|---|
| rollback_loss_checks (3x `full_loss()`/round) | 0.974s | **41.9%** |
| input_jvp (JVP loop, layer0 update) | 0.764s | **32.9%** |
| setup (per-round knot rebuild + joint design) | 0.217s | 9.3% |
| middle_solve | 0.091s | 3.9% |
| middle_basis_deriv | 0.091s | 3.9% |
| middle_eigh | 0.085s | 3.7% |
| output_eigh | 0.068s | 2.9% |
| input_solve | 0.028s | 1.2% |
| output_solve | 0.004s | 0.2% |

**The single biggest cost is NOT any optimization block -- it's the
monotonicity-check forward passes (rollback_loss_checks, 42%), followed by
the JVP computation (33%).** Both scale directly with the number of
rounds executed, which is exactly what early stopping targets. This
sharpens the priority: early stopping first is well-justified by the
actual profile, not just intuition.

**Step 2 -- early stopping**: added an opt-in `early_stopping` mode to
`fit()` (default False, preserves the exact prior behavior -- reverified
via `verify_cache_equivalence`, identical diffs after this change too):
stops once round-end loss's relative improvement is below `es_rel_tol=
1e-4` for `es_patience=3` consecutive rounds, never before
`es_min_rounds=5`. Best-so-far parameters are already guaranteed by the
existing per-block rollback mechanism (no separate "restore best" step
needed). Comparison running now: fixed-10-rounds vs fixed-15-rounds vs
early-stopping(cap 15), same 10-fold/seeds/data as the adopted A
configuration (`depth_experiment_earlystop.py`, output
`earlystop_results.txt`). Acceptance gate: time reduction >=20%, R2 diff
<0.002, no increase in fold-to-fold variance.

**Step 2 result -- early stopping REJECTED**: added an opt-in
`early_stopping` mode (default False, verified via `verify_cache_
equivalence` to leave default behavior byte-identical). Compared
fixed-10 vs fixed-15 vs early-stopping(cap 15) on 10-fold CV, both
functions. **Early stopping never triggered -- every fold ran the full 15
rounds (mean_n_rounds=15.0 exactly, matching fixed-15), because the
correction had not plateaued: R2 kept improving meaningfully from round 10
to 15 (+0.0019 to +0.0047), so the "3 consecutive rounds below 1e-4
relative improvement" condition was never met within this cap.**
Consequently early-stopping gave zero additional speedup over fixed-15,
which itself is 5.8-15.4% SLOWER than fixed-10 for a small accuracy gain.
**Verdict: REJECTED against the >=20% time-reduction criterion** -- not a
flaw in the early-stopping mechanism, but evidence that fixed-10 is
already the best cost/benefit point in the range tested; a 30-round probe
to find where the plateau actually is was explicitly declined by the user
as not worth the added runtime given fixed-10 already wins on cost/benefit.
**`n_rounds=10` stays the baseline`, early_stopping stays an inert,
scratchpad-only opt-in flag (default False).**

**Step 3 -- dynamic LM damping (middle block only)**: added an opt-in
`dynamic_damping` mode (default False, verified via `verify_cache_
equivalence` to leave default behavior byte-identical): starts from
`lam_ridge_mid=1e-3`; after each round's middle-block outcome, halves the
middle-block ridge on a clean (non-rolled-back) update or multiplies it by
5 on a rollback, clamped to [1e-8, 1e2], applied to the next round's
middle-block system. Comparison running now against the `n_rounds=10`
baseline, same seeds/folds/architecture (`depth_experiment_damping.py`,
output `damping_results.txt`). Acceptance (either sufficient): R2 improves
by >=0.005 with time increase <=10%, OR rollback/condition-number clearly
improves without losing accuracy.

**Step 3 result -- dynamic damping REJECTED, decisively, on both
acceptance paths**: R2_diff = +0.0000 on both functions (rollback pattern
was identical between arms -- the SAME blocks succeeded/failed regardless
of lambda, so the optimization outcome was insensitive to damping).
Time change: -1.8% to -3.5% (trivial, not a real speedup). Condition
number got measurably WORSE, not better: cond(M1) jumped from ~3.7e7 to
~1.87e10 (~500x) on both functions, because most rounds succeed already
(rollback_frac stays 0-10%), so lambda halves almost every round and
never gets multiplied back up, collapsing toward the 1e-8 floor
(lam_mid_final=9.77e-07). Still under Fable's 1e12 danger gate, but a
clear negative trend with zero offsetting benefit. **Neither acceptance
path (R2 +0.005, or improved rollback/conditioning) was met -- rejected.**
Confirms the diagnosis: instability was never a real problem in H1=28/
H2=25 (rollback already low from the start), so there was nothing for
adaptive damping to fix.

**Step 4 -- adaptive backtracking: NOT IMPLEMENTED, by design.** Its own
stated acceptance precondition (meaningful rollback rate to justify it)
does not hold for the adopted configuration -- implementing and testing it
would add complexity with no realistic hypothesis behind it. Skipped per
explicit user decision, not merely deferred.

---

## FINAL ADOPTED CONFIGURATION (Round 2, closed)

```
Residual depth-3 correction: F(x) = F_depth2(x) + dF(x)
H1 = 28, H2 = 25
rounds = 10 (fixed)
lam_ridge_mid = 1e-3 (fixed)
```

**Documented sub-decisions**:
- early stopping: REJECTED (never triggers within a reasonable round cap;
  fixed-10 already the best cost/benefit point).
- dynamic damping: REJECTED (zero accuracy change, worse conditioning,
  trivial time change).
- adaptive backtracking: NOT IMPLEMENTED (no rollback problem to justify
  it; skipped, not deferred).
- aggressive bottleneck (H1=66/H2=8): REJECTED (no accuracy gain, 43% peak
  rollback on some folds).
- `cache_basis`: numerically proven safe (6/6 equivalence configs, at-scale
  B-vs-C diff ~1e-15) but only 3-11% speedup -- kept scratchpad-only, not
  promoted, pending Codex's review of the reuse-validity logic.

**Evidence base for this configuration**: passes decisively at 3 parameter
scales (2,520 / 3,096 / 12,390), all on synthetic composed test functions
(`exp(sin(pi*x1)+x2^2)`, `sin(pi*(x1^2+x2^2))`).

**Next step (per user decision): NOT further optimizer tuning -- validate
this exact finalized configuration on diverse REAL datasets**, not just
the synthetic composed functions used for architecture search so far.

---

## Real-data validation of the finalized Round 2 configuration

Fixed config applied as-is (no re-tuning): H1=28, H2=25, rounds=10,
lam_ridge_mid=1e-3, H_base=64 (chosen consistent with prior benchmark_v1
depth-2 hidden-size conventions). Compared against a plain depth-2
baseline at a param-matched width per dataset, 5-fold CV, standardized
X/y, same protocol as the synthetic-function rounds.

**Datasets** (continuity with `docs/benchmark_v1.md`'s own real-data
benchmarks): California Housing (sklearn, subsampled to 3,000 rows --
documented limitation, same as the earlier notebook, for CV wall-clock
feasibility), Diabetes (sklearn, 442 rows, 10 features), Friedman-500
(`649_fri_c0_500_5` via pmlb, 500 rows, 5 features), Friedman-1000
(`595_fri_c0_1000_10` via pmlb, 1000 rows, 10 features).

**RESULT -- decisive and NEGATIVE. This reverses the entire program's
practical verdict.**

| dataset | R2_val residual | R2_val depth2 | paired diff | mean-2SE | folds+ | overfit gap (resid vs d2) |
|---|---|---|---|---|---|---|
| California Housing (n=3000, 8 feat) | 0.694+-0.028 | 0.689+-0.030 | +0.005 | -0.0141 (tie) | 3/5 | 0.093 vs 0.050 |
| Diabetes (n=442, 10 feat) | -0.163+-0.146 | -0.002+-0.045 | **-0.162** | -0.278 | **0/5** | 1.147 vs 0.920 |
| Friedman-500-5 (n=500, 5 feat) | 0.807+-0.036 | 0.854+-0.024 | **-0.047** | -0.068 | **0/5** | 0.164 vs 0.094 |
| Friedman-1000-10 (n=1000, 10 feat) | 0.754+-0.023 | 0.800+-0.016 | **-0.046** | -0.064 | **0/5** | 0.225 vs 0.148 |

**The residual depth-3 correction LOSES clearly to plain, param-matched
depth-2 on 3 of 4 real datasets (0/5 folds positive on each), and only
ties (not beats) on the 4th (California Housing, mean-2SE negative).**
This is a complete reversal from every synthetic-function result in this
document, where the same architecture won decisively and consistently.

**Diagnosis**: the residual model's overfit_gap (train R2 minus val R2) is
systematically WORSE than plain depth-2's on every single dataset -- 1.5x
to 12x larger. Train R2 is very high everywhere (0.96-0.98) while
validation R2 is much lower, the classic overfitting signature. **The
synthetic test functions (`exp(sin(pi*x1)+x2^2)`, `sin(pi*(x1^2+x2^2))`)
were specifically constructed as clean nested compositions -- exactly the
structure depth-3 is theoretically suited to exploit. Real datasets do not
reliably have that structure, so the correction's extra capacity fits
noise in the residual instead of recovering genuine unmodeled signal.**
The entire architecture search in this document (Round 1, Round 2, and
all its refinements) was validated ONLY against synthetic functions
handpicked to favor this hypothesis -- this is the first test against
data not constructed for that purpose, and it fails.

**Revised verdict on the whole depth-3 program**: the "FINAL ADOPTED
CONFIGURATION" above is **not currently recommended for production/general
use** -- its validated advantage appears specific to nested-composition
synthetic functions and does not transfer to general real-world regression
data in this test. This does not retroactively invalidate the synthetic-
function evidence (the effect there is real and was rigorously verified),
but it reframes the practical conclusion: depth-3 (in this residual-
correction form) is not yet evidence of a general improvement over
depth-2 for KANBoost's typical real-data use case, only for a narrow,
specific class of functions.

**Open per rule 15 -- stop here for ChatGPT scientific review and user
decision** before any further action. Candidate next steps for that
review to weigh (not decided): (a) close the depth-3 program as
"validated only for known-nested-composition problems, not general use,"
(b) investigate whether stronger regularization on the correction
(specifically for real, noisier data) recovers competitiveness, (c) test
on a real dataset believed to actually have nested-composition structure
before generalizing the negative conclusion to "depth-3 never helps on
real data."

**ChatGPT's scientific ruling**: Depth-3 residual REJECTED as a general/
production model. Its success is confined to clean nested-composition
functions; the real-data failure is systematic overfitting, not random
noise or a minor tuning gap -- explicitly advises against a broad new
hyperparameter search (risk of selective post-hoc fitting to this data).
One last low-complexity test before closing the path for good, since it
directly distinguishes "correction is too strong" (fixable by shrinkage)
from "correction fits the wrong thing" (shrinkage won't help):

```
F(x) = F_depth2(x) + alpha * dF_depth3(x),  alpha in {0, 0.05, 0.1, 0.25, 0.5, 1.0}
```

alpha selected via an INNER validation split of the training fold only
(never touching the outer/held-out fold) -- alpha=0 reduces automatically
to plain depth-2, so this can never do worse than depth-2 on the fold used
to CHOOSE alpha, but can still lose on the truly held-out fold if alpha
selection doesn't generalize.

**Acceptance (all required)**: beats depth-2 on >=3/4 datasets; mean R2
gain >=+0.01; overfit-gap increase <=+0.01; alpha selection uses no outer-
fold data (verified: `select_alpha_inner` only ever sees `X_tr`/`y_tr`,
split further into an 80/20 inner train/val -- the outer `X_va`/`y_va` is
untouched until final scoring after alpha is fixed).

**Procedure implemented** (`depth_experiment_shrinkage.py`): per outer
fold, (1) inner 80/20 split of outer-train, fit base+correction on inner-
train, pick the alpha maximizing inner-val R2 (including alpha=0 as the
depth-2-equivalent floor); (2) refit base+correction fresh on the FULL
outer-train data; (3) apply the alpha chosen in step 1 to this refit
model; (4) score on the outer-held-out fold. Same 4 real datasets as
above, same standardization/protocol.

**RESULT -- FAILS all three acceptance criteria simultaneously.**

| dataset | mean_diff | overfit-gap increase | folds+ | chosen alphas (per fold) |
|---|---|---|---|---|
| California Housing | **+0.0132** | +0.0151 | 5/5 | 0.5, 0.5, 0.5, 0.5, 0.25 |
| Diabetes | +0.0027 (noisy, n=442, both models near R2~0) | +0.0123 | 3/5 | 0.1, 0.25, 0.25, 0.25, 0.5 |
| Friedman-500-5 | -0.0105 | +0.0051 | 1/5 | 0.1, 0.0, 0.0, 0.1, 0.0 |
| Friedman-1000-10 | -0.0047 | -0.0010 | 2/5 | 0.5, 0.05, 0.05, 0.1, 0.1 |

**Overall**: 2/4 datasets beat depth-2 (need >=3/4) -- FAIL. Mean R2 gain
+0.0002 across datasets (need >=+0.01) -- FAIL. Max overfit-gap increase
+0.0151 (need <=+0.01) -- FAIL.

**Key diagnostic detail**: even where the inner validation correctly
selected alpha=0 or near-0 (Friedman-500-5, 3 of 5 folds chose alpha=0.0,
i.e. pure depth-2), the outer-fold result still occasionally underperformed
plain depth-2 slightly -- meaning shrinkage's inner-validation alpha choice
itself does not always generalize perfectly to the held-out fold, on top
of the correction not helping in the first place on this dataset. This
confirms ChatGPT's diagnosis directly: the real-data failure is not simply
"correction strength too high" (which uniform shrinkage would have fixed
cleanly) -- it is dataset-dependent structural mismatch. California
Housing is the one dataset where shrinkage (alpha~0.5) gave a genuine,
consistent (5/5 folds) improvement, but even there the overfit-gap
increase (+0.0151) breaches the safety criterion, so it does not count as
a passing case under the predefined rule.

---

## FINAL CLOSURE

**Per the user's pre-committed decision rule, this fails the acceptance
gate, so depth-3 (residual-correction form) is CLOSED as a general or
production option for KANBoost.**

Documented final status:
- **Validated capability**: depth-3 residual correction (H1=28, H2=25,
  rounds=10, lam_ridge_mid=1e-3) reliably and significantly beats
  parameter-matched depth-2 on functions with clean, nested-composition
  structure (verified at 3 parameter scales: 2,520 / 3,096 / 12,390) --
  this remains true and rigorously verified, it is not retracted.
- **Not validated / rejected for general real-world regression data**:
  fails on 3 of 4 diverse real datasets, with systematic overfitting (not
  random noise) as the root cause; a shrinkage/blending remedy
  (`alpha in {0,...,1}` via leakage-free inner validation) does not
  rescue it to the required bar either.
- **Classification going forward**: depth-3 residual correction is a
  narrow, specialized architecture applicable when the target function is
  independently known/suspected to have nested-composition structure --
  it is NOT a general-purpose upgrade over depth-2 and should not be
  offered as a default or recommended option in KANBoost.
- No code from this research path is promoted to `kanboost/core/`. The
  stable depth-2 implementation is unaffected and remains the production
  default, exactly as it was before this entire investigation began.
- `cache_basis` and all other scratchpad optimizer refinements
  (early stopping, dynamic damping) are also closed/rejected as documented
  above and are not carried forward.

This closes the depth-3 KAN research program opened at the top of this
document. `research/depth3-kan` branch retained for the record; no merge
to `main`.

---

# ALS-solve performance: numpy batching vs C++ extension

New, separate research question (branch `research/als-solve-perf`, off
`main` -- `research/depth3-kan` never diverged from `main` with any
commits, so this is a clean split). Triggered by the user wanting to try
building a C++/pybind11 extension for KANBoost (MSVC 14.51 confirmed
installed via Visual Studio 2022 Build Tools; `pybind11` not yet
installed). Per the burden-of-proof rule, measured the cheapest in-
language alternative FIRST, before any native build.

**Target**: the per-hidden-unit layer0-update loop in `DeepKAN._fit_als`
(`kanboost/core/kan/network.py` lines ~545-560) -- for each of
`n_hidden` hidden units, computes a linearized target and calls
`_solve_with_factor` separately, even though all units share the same
pre-factored eigendecomposition (`eigvecs0`/`eigvals0`). This is exactly
the "solve the same system for many right-hand-sides" pattern the
production code's own comment on `_eigh_factor` identifies as the reason
eigh is factored once and reused -- but the per-unit solves themselves
were still a Python loop, one small matmul at a time.

**Numpy-only batching fix (no C++, no new dependency)**: replaced the
per-unit loop with one vectorized pass -- all `n_hidden` linearized
targets built into a single `(n, n_hidden)` matrix, one `Bw0.T @ T`
matmul, one batched solve `eigvecs0 @ ((eigvecs0.T @ RHS) / eigvals0[:,
None])` instead of `n_hidden` separate small ones. Implemented as
`_fit_als_batched` in the scratchpad (`als_batch_solve_prototype.py`,
`als_batch_endtoend.py`) -- monkeypatches `DeepKAN._fit_als` at runtime
for testing; `kanboost/core/` is NOT modified on disk.

**Verification**: coefficients from the batched version vs the original
loop, on the REAL `KANLayer`/`DeepKAN` classes (not a synthetic proxy),
match to max abs diff 1.8e-13 to 4.4e-13 across n_hidden in {16, 64, 128}
-- far under the 1e-9 gate.

**Speedup**:
- Isolated layer0-update step (same real classes): 1.44x-1.61x faster
  across n_hidden in {16, 64, 128}.
- **End-to-end real `KANBoostRegressor` fit** (n=2000, 8 features,
  n_estimators=30, kan_hidden=64): **20.19s -> 17.31s, 1.166x speedup
  (16.6% faster), R2 diff +2.6e-13 (bit-identical for practical
  purposes)**. Smaller than the isolated-step speedup, as expected --
  Amdahl's law, since this loop is only part of a sweep's total cost
  (eigh factorization, the layer1 `_solve_layer` call, and knot rebuilding
  all take real time too).

**C++/pybind11 comparison -- built and measured.** Environment: MSVC
14.51.36246 (VS 2022/"18" BuildTools), `pybind11` 3.0.4 installed,
compiled directly via `cl.exe` (`/O2 /EHsc /std:c++17 /LD`) after
`vcvarsall.bat x64`, no CMake needed for this single-file test
(`als_batch_cpp.cpp`, `als_batch_cpp_compare.py`). Implementation: plain
C++ loops (no BLAS linked) for the same two pieces -- linearized-target
construction (`compute_targets`) and the batched eigh-factored solve
(`batched_eigh_solve`) -- deliberately NOT reimplementing the raw matmuls
elsewhere (`Bw0.T @ T` stays in numpy) to isolate a fair test of "does
hand-written native code beat numpy's BLAS-backed version for this
specific, modest-sized linear algebra," not a rebuild of BLAS itself.

**Verification**: C++ vs numpy-batched coefficients match to 2.4e-14 to
4.2e-14 across n_hidden in {16, 64, 128, 256} -- both are correct.

**Speed result -- DECISIVE, and in the OPPOSITE direction from a naive
"C++ is faster" assumption**:

| n_hidden | numpy (batched) | C++ (no BLAS) | C++/numpy |
|---|---|---|---|
| 16 | 1.58ms | 1.59ms | 0.99x (tie) |
| 64 | 7.12ms | 11.20ms | 0.64x (C++ slower) |
| 128 | 15.03ms | 23.08ms | 0.65x (C++ slower) |
| 256 | 27.87ms | 66.58ms | **0.42x (C++ 2.4x SLOWER)** |

**The disadvantage widens as problem size grows** -- exactly the regime
where BLAS threading/vectorization matters most. Root cause: numpy's
matmuls are backed by multi-threaded OpenBLAS (6 threads on this machine,
confirmed earlier via `threadpoolctl`); the hand-written C++ loops are
single-threaded with no BLAS linked, so they cannot compete once the
matrices are large enough for threading/SIMD to pay off. This is the
same lesson as the earlier scipy-vs-numpy `eigh` driver test in this
document: don't hand-roll what an optimized BLAS backend already does
well.

**FINAL DECISION: no C++ extension for this target.** The numpy-only
batching fix (16.6% end-to-end speedup, verified safe, zero new
dependencies) is both simpler AND faster than the C++ alternative that
was actually built and measured -- not assumed. Per the user's own
instruction ("try both, let the practical result decide"), this is not a
judgment call, it's a measured outcome. C++/pybind11 remains available as
infrastructure (MSVC + pybind11 now confirmed working end-to-end on this
machine) for a future target that is NOT dominated by BLAS-backed linear
algebra (e.g., a genuinely loop-bound, non-matmul computation), should one
arise -- but is not adopted here.

**Follow-up test (user request): does OpenMP multithreading -- matching
OpenBLAS's own thread count (6, confirmed via `threadpoolctl` earlier in
this document) -- close the gap?** Added `#pragma omp parallel for` to
every independent outer loop in both `compute_targets` and
`batched_eigh_solve` (each iteration writes disjoint output, verified
race-free), `omp_set_num_threads(6)`, recompiled with `/openmp`. Result:

| n_hidden | numpy | C++ (serial) | C++ (6-thread OpenMP) |
|---|---|---|---|
| 16 | 1.20ms | 1.59ms | 1.23ms (0.97x) |
| 64 | 7.14ms | 11.20ms | 9.15ms (0.78x) |
| 128 | 13.91ms | 23.08ms | 22.72ms (0.61x) |
| 256 | 29.49ms | 66.58ms | 63.62ms (0.46x) |

Threading helps modestly (mainly at n_hidden 64-128) but **does not change
the verdict** -- C++ stays slower than numpy at every non-trivial size,
and the gap is still large at n_hidden=256 (still ~2x slower). Likely
cause: per-call OpenMP thread-team spawn/teardown overhead competes with
or exceeds the parallelization gain at these problem sizes, whereas
OpenBLAS uses an already-warm, persistent thread pool. **Numerical
equivalence unchanged** (2.4e-14 to 4.2e-14 vs numpy, same as the serial
version).

**FINAL DECISION (reconfirmed with threading): no C++ extension for this
target**, even with multithreading matched to OpenBLAS's own thread count.
The numpy-only batching fix remains simpler, dependency-free, and faster
than every C++ variant actually built and measured (serial and
multithreaded alike).

**APPLIED to `kanboost/core/kan/network.py`** (per explicit user
instruction), on branch `research/als-solve-perf`, NOT committed/merged.
Replaced the per-hidden-unit loop (former lines ~545-560) with the
verified batched version -- same file, same class, same method, only the
inner loop body changed; rollback/convergence/basis-cache logic untouched.

**Full existing test suite**: 165 passed, 0 failed (only pre-existing
matplotlib deprecation warnings, unrelated). ~12 minutes wall-clock.

**Corrected speed measurement -- the first two post-edit timings (43.97s,
then 25.34s on a rerun) were noisy and NOT directly comparable to each
other or to the earlier scratchpad number**, because each was a fresh
Python process paying its own numba JIT/disk-cache warm-up cost, and the
first ran immediately after the 12-minute test suite (likely residual
system load/disk contention). R2 was identical (0.9016) in both, so
correctness was never in question -- only the raw wall-clock numbers were
unreliable for a speedup ratio.

**Rigorous, same-process, warmed-up paired comparison** (`paired_
comparison_final.py`: a throwaway 2-estimator warm-up fit first to absorb
JIT/cache cost equally, THEN both timed arms -- the original per-unit loop
reconstructed exactly as it was pre-edit, vs the current on-disk patched
`network.py` -- run back-to-back in the same process):

```
original loop (reconstructed):  18.62s
current network.py (patched):   17.10s
speedup: 1.089x (8.9% faster)    R2 diff: +2.6e-13 (bit-identical)
```

**This 8.9% figure supersedes the earlier 16.6% estimate** -- that number
came from a comparison without a separate warm-up run, so the first
("baseline") timed arm absorbed disproportionately more of the one-time
JIT cost, inflating the apparent speedup. The 8.9% number, measured with
both arms starting from an equally warm cache, is the trustworthy one.
Still a real, safe, zero-new-dependency, zero-accuracy-cost win -- just
smaller than first estimated. Documenting the correction itself, per this
project's standing rule to never let an unverified number stand once a
more rigorous measurement is available.

**Status: implemented and verified on `research/als-solve-perf`, not
committed/merged.** Awaiting Codex's read-only review and ChatGPT's
sign-off before any merge to `main`, per `CLAUDE.md`'s pipeline.

**Deferred, per Fable's ruling**: line search / step-damping / decreasing
step_ratio (per-block rollback already enforces monotonicity; only add a
minimal 2-halving backtrack if `rollback_frac` exceeds ~20% in the
bottleneck's actual runs -- not yet observed). `lam_ridge_mid` stays fixed
at 1e-3, not pre-scaled with the bottleneck (no principled scaling law,
watch `cond_M1`/rollback logs instead -- no new hyperparameter surface).
float32 stays deferred (unverified). Cholesky stays closed (prior
rejection for numerical safety on ill-conditioned matrices still holds).
Time-matched secondary comparison (arm D: shrink H1 until wall-clock
matches A; arm E: plain depth-2 at A's wall-clock) queued as a follow-up
after B/C lands, not blocking.

**Round 2 ~12,500-param scale check -- RESULT (A config: H_base=200,
H1=28, H2=25, 12,390 params vs plain depth-2 H=350, 12,600 params,
n=4000/10-fold)**:

| function | R2_val residual | R2_val depth2 | paired diff | mean-2SE | folds+ | overfit gap (resid vs d2) | time (resid vs d2) |
|---|---|---|---|---|---|---|---|
| exp(sin+x^2) | 0.990+-0.003 | 0.961+-0.009 | +0.029 | **+0.0236** | **10/10** | 0.0029 vs 0.0047 | 8.6s vs 23.4s |
| sin(radial) | 0.957+-0.016 | 0.920+-0.025 | +0.037 | **+0.0255** | **10/10** | 0.0131 vs 0.0124 | 8.6s vs 24.0s |

**This is the strongest result across either round so far**: 10/10 folds
positive on BOTH functions (Round 1's joint-ALS got 4/10 on function 2 at
this exact scale and REVERSED sign). Overfit gap tied-or-better. Residual
model is ~2.7x FASTER than the param-matched depth-2 baseline, consistent
with the 2,520/3,096-param results. **Conclusion: Round 2 (depth-2-
pretrained + residual depth-3 correction + per-block monotonic rollback)
is scale-robust where Round 1 (joint, from-scratch 3-block ALS) was not.**
This is now the only depth-3 architecture with evidence at three different
scales (2,520 / 3,096 / 12,390 params), passing decisively at every one.

---

# Auto Resource Planner -- Pre-Implementation Evaluation

Separate hypothesis from the depth-3 KAN work above, same governance
(`CLAUDE.md`/`AGENTS.md`). Scratchpad-only:
`resource_planner.py` (detection + planning logic),
`resource_planner_benchmark.py` (3-mode proof-of-benefit harness).
`kanboost/core/` untouched; no new dependencies added to KANBoost itself
(`psutil` was already installed in this environment; `threadpoolctl` is
already a transitive dependency used for BLAS thread inspection).

**Hypothesis**: parallelizing fold execution with a resource-aware plan
(workers x BLAS-threads-per-worker chosen from real CPU/RAM/GPU detection)
reduces CV wall-clock time by >=20% without changing `n_splits`, without
oversubscription, without unsafe RAM use (peak < 75% available), without
accuracy drift beyond 1e-8 in mean R^2, and without new mandatory
dependencies.

**Detected resources on this machine**: 6 physical cores, 6 logical cores
(no hyperthreading), 8.36 GB total RAM, **2.39 GB available** (machine is
under real memory pressure right now -- other applications are using
~5.97 GB). BLAS: OpenBLAS (bundled `scipy-openblas`), default 6 threads,
no env var caps set. GPU: none usable -- `torch.cuda.is_available()` is
False (CPU-only torch build installed), no `cupy`, no `nvidia-smi` --
planner correctly reports `backend: cpu` with no warnings needed for that
fallback.

**Peak-RAM estimation approach**: analytic formula from real matrix
dimensions (joint design B: n_train x H*K; penalized system M: H*K x H*K;
LAPACK `syevd` workspace: empirically ~3x M's footprint) -- e.g. H=350,
K=6, n_train~3600 -> ~201.6 MB/fold. Cross-check against measured RSS
(`measure_peak_ram_fit`, background-thread RSS sampler) is implemented but
not yet reported here -- pending benchmark completion.

**Planner output for the medium workload** (n_splits=10, H=350): workers=6
(= physical_cores, since RAM allows up to 8 concurrent folds at ~201.6
MB/fold but cores cap it at 6), blas_threads_per_worker=1, no
oversubscription (6x1=6 <= 6 physical cores), projected peak RAM=1.21 GB
(< 75% of 2.39 GB available), zero warnings.

**Status**: first-pass 3-mode benchmark (serial / manual-parallel-3-workers
/ auto) launched on small (H=86, n=1000, 5-fold) and medium (H=350,
n=4000, 10-fold) workloads, per rule 10 run alone (no concurrent unrelated
job). Still running.

**ChatGPT review of the first pass (before results returned)**: the
analytic 201.6 MB/fold estimate only covers the solver's own matrices --
it omits per-worker-process Python/SciPy baseline memory, Windows
`spawn`-related data duplication, numba/import cache overhead, and
real-time available-RAM drift. On an 8 GB machine with only 2.39 GB
currently available, defaulting to `6 workers x 1 BLAS thread` is
premature even though the pure-matrix math clears the 70%/75% thresholds.

**Machine-specific safety rule (explicit, from the user)**: reserve
2-2.5 GB for the OS/other processes; use at most ~5.5-6 GB for training on
this machine. `6x1` is only adopted as a recommendation if measurement
shows: peak system RAM < ~6 GB, zero swap activity, genuinely faster wall
time than the alternatives, and no fold fails or degrades. Prior
expectation, to be confirmed or falsified by measurement, not assumed:
`2 workers x 3 BLAS threads` or `3 workers x 2 BLAS threads` is more likely
the balanced choice on this hardware.

**First-pass result (CONFIRMS the risk directly, not just theoretically)**:

Small CV (H=86, n=1000, 5-fold): 1x6 serial = 4.49s. 3-workers-1-blas =
11.05s (0.41x -- SLOWER, process-spawn/numba-reimport overhead dominates
at this size). 5-workers-1-blas (auto's choice at this scale) = 12.07s
(0.37x -- also slower). R^2 identical to 5e-15 in all configs (numerically
safe where it ran). **Parallelizing small folds is net negative here.**

Medium CV (H=350, n=4000, 10-fold): 1x6 serial = 229.12s. 3-workers-1-blas
= 202.06s (0.88x time, only **~12% faster -- below the 20% acceptance
bar**). The **auto plan's 6-workers-1-blas config CRASHED**:
`numpy._core._exceptions._ArrayMemoryError: Unable to allocate 57.7 MiB
for an array with shape (3600, 2100)` inside a worker process, mid-run.
This is a real out-of-memory failure, not a projection -- it happened
despite the analytic estimate (1.21 GB projected total, well under the
1.79 GB / 75%-of-available threshold) saying it should be safe. Confirms
ChatGPT's diagnosis exactly: the analytic estimate omits per-worker
Python/SciPy/numba baseline overhead and Windows `spawn` duplication, and
real available RAM was ~2.39 GB with other applications actively
fluctuating that number -- the naive planner is unsafe as currently
specified.

**Decision so far**: `6x1` (or any auto-chosen config derived only from
the matrix-only analytic estimate) is REJECTED as a default -- it failed
outright, not just underperformed. `3x1` gave a real but sub-threshold
gain (12% < 20%) on the one workload it completed. Per rule 12 (reject
when criteria aren't met, even if partially promising), this pushes
toward likely overall rejection of the auto-parallelism hypothesis at this
machine's scale, pending the rigorous 2x3/3x2 measurements below, which
have not yet been tried.

**Revised benchmark result** (`resource_planner_benchmark_v2.py`, fixed
4-config sweep 1x6/2x3/3x2/6x1, system-wide peak RAM via continuous
background sampling, each worker's OS-tracked `peak_wset`, page faults,
swap sin/sout):

Small CV (H=86, n=1000, 5-fold): every parallel config is SLOWER than
serial -- 2x3 -77.8%, 3x2 -72.8%, 6x1 -86.6%. Process-spawn/numba-reimport
overhead dominates completely at this workload size. R^2 identical to
~1e-15 across all configs (numerically safe, just pointless here).

Medium CV (H=350, n=4000, 10-fold):

| config | wall time | vs serial | system peak RAM used | system RAM left at peak |
|---|---|---|---|---|
| 1x6 (current) | 239.0s | -- | 6.16 GB | 2.20 GB (meets the 2-2.5GB reserve rule) |
| 2x3 | 201.4s | +15.8% | 7.24 GB | **1.12 GB (violates reserve rule)** |
| 3x2 | 194.5s | +18.6% | 6.77 GB | **1.59 GB (violates reserve rule)** |
| 6x1 | 192.2s | +19.6% | 7.97 GB | **0.39 GB (severe violation)** |

No config reaches the predefined 20% speed threshold (best: 6x1 at
19.6%), and every parallel config exceeds the user's explicit ~6 GB
peak-RAM / 2-2.5 GB reserve safety rule -- 6x1 leaves only 390 MB free
system-wide, which is the same configuration that produced an outright
`ArrayMemoryError` crash in the first-pass run; it survived this run but
with almost no margin. R^2 identical to ~1e-15 in every config (accuracy
criterion is not the blocker here -- safety and the speed threshold are).
Zero swap activity recorded in any run.

**Final verdict: REJECT the auto-parallel-fold hypothesis on this
machine, per rule 12** -- no configuration satisfies both the >=20%
wall-clock criterion and the RAM-safety criterion simultaneously. The
configs that get closest to 20% (3x2, 6x1) do so only by consuming RAM
margins the user explicitly ruled unsafe on this 8 GB machine, and 6x1
specifically already caused a real crash once. **The current serial
configuration (1 worker, BLAS using all physical cores) remains the
correct default and is NOT changed.** No promotion to `kanboost/core/`,
no scratchpad code adopted as a running default -- this is a negative
result, recorded so the question isn't re-litigated without new hardware
or a materially different workload profile.

**Status: Auto Resource Planner evaluation CLOSED (rejected).** Awaiting
Codex read-only review and ChatGPT sign-off on the rejection reasoning
before considering this fully settled, per the standard pipeline.

---

## Round 1 — Hypothesis: does a 3rd KAN layer help at all?

**Hypothesis**: A depth-3 KAN (n_in -> H1 -> H2 -> 1), trained end-to-end via
3-block Gauss-Newton ALS, beats a parameter-matched depth-2 KAN on nested-
composition functions where the classical depth-2 Kolmogorov-Arnold form is
not the natural representation.

**Test functions**: f1 = exp(sin(pi*x1)+x2^2); f2 = sin(pi*(x1^2+x2^2)) (radial).
Both on 5 uniform inputs in [-1,1], normalized targets.

**Result (2520 params, n=1000, 5-fold CV, naive 3-block ALS with the
original linearized-step formula)**: REJECTED. depth-3 lost by 0.17-0.21 R^2
on both functions, on train AND validation (ruling out overfitting as the
explanation). Condition numbers were healthy throughout (< 1e8).

**ChatGPT review of Round 1**: found a real bug — the linearized-step
denominator summed over ALL samples (`norm_sq = sum_m slope[m]^2`), which
shrinks the effective per-sample correction by ~1/n per sweep. The middle
layer was never meaningfully training; z2 stayed near its random
initialization. This is *not* evidence against depth-3 — it's evidence of
an under-trained optimizer.

**Fix applied** (both to the depth-3 prototype AND to a matched depth-2
prototype built with the identical optimizer, for fairness): per-sample
(row-wise) Gauss-Newton normalization with LM damping, near-linear C1 init,
z-clamping into knot spans, added diagnostics (step-magnitude ratio,
dead-derivative fraction, rollback count).

**Result after fix (2520 params, n=1000, 5-fold)**: reversed — depth-3 now
beats depth-2 by +0.016 to +0.07 R^2, train R^2 also higher (gap closed and
flipped), 4/5 folds positive both functions. Borderline significance at
n=5 folds (mean-2SE straddled zero on f1, negative on f2 due to one
high-variance fold).

**Result after fix (2520 params, n=4000, 10-fold — reduces fold-variance
noise at the SAME param budget)**:

| function | R2_val depth3 | R2_val depth2 | paired diff | mean-2SE | folds+ | time (depth3 vs depth2) |
|---|---|---|---|---|---|---|
| exp(sin+x^2) | 0.962+-0.013 | 0.946+-0.014 | +0.016 | +0.0053 | 8/10 | 0.76s vs 1.35s |
| sin(radial) | 0.882+-0.024 | 0.811+-0.089 | +0.071 | +0.0024 | 7/10 | 0.76s vs 1.30s |

Overfit gap for depth-3 is equal-or-lower than depth-2's in both cases
(0.005 vs 0.007; 0.015 vs 0.015). depth-3 is also ~1.7x *faster* than
depth-2 at equal params (fewer, cheaper per-edge basis evals). Zero
rollbacks, healthy step-ratio decay, zero dead derivatives.

**Status**: the medium-scale retest (~34,560 params: depth-3 H1=72,H2=74 vs
depth-2 H=960, n=4000/10-fold) was launched in the background, ran for
~32 minutes with zero output, and was killed after diagnosis. Root cause,
confirmed by direct measurement: `np.linalg.eigh` on the depth-2 baseline's
layer-1 system (H*K x H*K = 960*6 = 5760x5760, recomputed every sweep
because the knots/design shift each sweep in this ALS scheme) costs **~61
seconds per call**. At 10 sweeps x 10 folds x 2 functions = 200 calls, that
line alone is ~3.4 hours -- not a hang, a genuine O(n^3) cost wall at this
matrix size on this machine. This is a real, reportable constraint (rule
11), not a bug: eigh is the correct, numerically-safe choice (Round 1's own
earlier finding rejected Cholesky as silently wrong on ill-conditioned
matrices), so the fix is scope (smaller H at this budget, or accept the
runtime) rather than the solver.

**ChatGPT's diagnosis and decision**: the bottleneck is not depth-3 itself,
it's the *depth-2 matched baseline* at large H (the single wide layer's
M1 system is H*K x H*K, growing faster than depth-3's split H1/H2
matrices for the same total parameter count). Ranked fix options given:
(1) reduce H with closer matching (~320-480) to stay under the eigh wall,
(2) reuse the spectral factorization when the matrix doesn't change much
between sweeps, (3) iterative solver (CG/LSMR) instead of full
eigendecomposition if only a solve is needed, not the spectrum itself, (4)
exploit structure (block/low-rank/Woodbury/randomized eigh), (5) fewer
sweeps or internal early stopping. Conclusion: the next fair comparison
should be at a computationally feasible size or matched time budget, not
at 5,760x5,760 with repeated full eigh.

**Action taken**: measured eigh cost directly at H=320/350/400 before
committing (1.1s/1.8s/2.3s per call respectively -- confirms option (1) is
sufficient without needing solver surgery yet). Selected H=350 (depth-2,
12,600 params) matched against depth-3 H1=40,H2=46 (12,516 params, 0.7%
apart) -- same n=4000/10-fold protocol as the small-scale result. Total
estimated eigh cost ~6 min across the full grid, tractable. Launched in
background (`depth_experiment_v2.py`, output `medium_scale_results_v2.txt`).
Options (2)-(5) are noted for later if a genuinely large-H comparison is
ever needed, per the burden-of-proof rule (no solver change without a
measured need).

**Result (~12,500 params: depth-3 H1=40,H2=46 [12,516] vs depth-2 H=350
[12,600]; n=4000, 10-fold)**:

| function | R2_val depth3 | R2_val depth2 | paired diff | mean-2SE | folds+ | overfit gap (d3 vs d2) | time (d3 vs d2) |
|---|---|---|---|---|---|---|---|
| exp(sin+x^2) | 0.976+-0.007 | 0.961+-0.009 | +0.0147 | +0.0071 | 9/10 | 0.0043 vs 0.0047 | **3.2s vs 23.9s** |
| sin(radial) | 0.912+-0.024 | 0.920+-0.025 | **-0.0083** | **-0.0240** | **4/10** | 0.0201 vs 0.0124 | **3.1s vs 22.5s** |

**This is a real, reportable reversal (rule 12: reject when criteria fail,
even if a related result looked promising)**: the joint-ALS depth-3
advantage found at 2,520 params (+0.016 to +0.071) DILUTES on function 1
(down to +0.015, now below the +0.02 raw bar though still positive at 2SE)
and REVERSES on function 2 (-0.008 mean, 4/10 folds positive, mean-2SE
clearly negative, depth-3's overfit gap now *worse* than depth-2's: 0.020
vs 0.012). **Verdict: joint-ALS depth-3 (Round 1) does not hold up as a
scale-robust advantage — its accuracy edge is not consistent as parameter
budget grows, and it fails its own predefined acceptance criteria on one
of the two required functions at this scale.**

One robust, unconfounded finding survives regardless of the accuracy
question: depth-3's wall-clock time does not grow with H the way depth-2's
does (3.1-3.2s vs 22.5-23.9s at this scale — the eigh-cost wall hits
depth-2's single wide layer, not depth-3's split H1/H2 layers, at equal
total parameters). This is an architectural property, not a tuning
artifact, and is worth keeping in the record independent of the accuracy
verdict above.

**Status: Round 1 (joint, from-scratch 3-block ALS) is NOT recommended for
promotion** -- it wins narrowly and inconsistently at small scale, and
loses/reverses at medium scale. Round 2 (residual correction) remains the
only architecture that has decisively passed its acceptance criteria on
both functions at every scale tested so far (2,520-equivalent and 3,096
params); it has not yet been tested at the ~12,500 scale specifically.
Recommendation for the next round: re-run Round 2 at the same ~12,500-param
budget used here, to check whether ITS advantage is scale-robust where
Round 1's was not, before requesting Codex/ChatGPT sign-off on anything.

---

## Round 2 — ChatGPT's diagnosis: the problem is HOW depth-3 is trained, not whether depth helps

**ChatGPT's restated hypothesis**: A jointly-trained, randomly-initialized
3-block ALS system is intrinsically harder to identify/optimize than a
2-block system (the middle layer has no anchor — many (C1, C2) pairs give
the same z2 up to a smooth reparametrization, and joint ALS has to resolve
this from scratch every fold/seed). The fix is not more sweeps or more
parameters, it's a training PROCEDURE change:

1. Replace joint 3-block ALS with block coordinate descent + explicit line
   search; never accept a block update that raises the loss (stricter than
   the current whole-sweep rollback — checked per block).
2. Initialize from a converged depth-2 fit, splitting its hidden layer into
   two, instead of random init.
3. Freeze two blocks and train the third on a rotating schedule: output,
   then middle, then input.
4. Add a residual/boosting structure: `F(x) = F_depth2(x) + dF_depth3(x)`,
   so depth-3 only has to explain what depth-2 couldn't, instead of
   learning the whole function from a random start.
5. Stronger regularization specifically on the middle layer (named as the
   primary source of non-identifiability/instability).
6. Add a shallow asymmetric control model `d -> h -> h/2 -> 1` (fewer
   params, no big symmetric H1~H2) as an additional experimental arm.
7. Log per block: loss, condition number, update magnitude, train/val gap.

ChatGPT's recommended best bet: **depth-2 pretrained + residual depth-3
correction + monotonic block updates with rollback** — lower risk than the
from-scratch joint 3-block ALS, because the correction only has to model a
residual (typically smaller/smoother), and depth-2's already-validated fit
is never put at risk (it's frozen).

**Note for the review record**: Round 1's post-fix results (above) already
show a joint, from-scratch 3-block ALS beating depth-2 at matched params,
with healthy diagnostics and no instability — i.e., the "joint ALS can't be
identified from scratch" concern was not what caused Round 1's original
failure (that was the 1/n step-shrinkage bug, now fixed). Round 2 is
therefore evaluated as a genuinely separate, second experimental arm
(residual-correction architecture) to compare against the already-working
joint-ALS depth-3, not as a required fix for a still-broken baseline.
Both arms will be reported side by side.

**Planned implementation** (scratchpad only, `depth_residual_prototype.py`):
- `DeepKAN2` (existing prototype) trained to convergence on y -> frozen.
- A small depth-3-shaped correction network trained on residual
  `e = y - F_depth2(x)`, using the same row-normalized ALS step, with
  block order output -> middle -> input and a per-block loss check +
  rollback (not just per-sweep).
- Total prediction: `F(x) = F_depth2(x) + dF(x)`.
- Fair comparison: total parameter budget (depth2-base + correction) held
  equal to a single plain depth-2 model of the same total size, same
  n/folds/seeds as Round 1.
- Additional arm: shallow asymmetric `d -> h -> h/2 -> 1` correction size.

**Implementation note (documented simplification)**: the correction's C1
init is NOT literally split from the trained depth-2 base's weights (point
2 in ChatGPT's review) — it uses the same near-linear random init as the
Round 1 joint-ALS prototype. Full weight-transplant surgery is out of
scope for this round; can be added later if these numbers justify chasing
further gains.

**Result (3096 total params: depth2-base H=50 [1800] + correction H1c=16,
H2c=8 [1296]; matched plain depth-2 H=86 [3096]; n=1000, 5-fold)**:

| function | R2_val residual | R2_val depth2 | paired diff | mean-2SE | folds+ | overfit gap (resid vs d2) |
|---|---|---|---|---|---|---|
| exp(sin+x^2) | 0.958+-0.006 | 0.939+-0.010 | +0.019 | **+0.0057** | **5/5** | 0.028 vs 0.027 |
| sin(radial) | 0.831+-0.045 | 0.766+-0.055 | +0.065 | -0.0268 | 3/5 | 0.113 vs 0.104 |

Function 1: cleanest result of either round so far — all 5 folds positive,
mean-2SE clearly positive, overfit gap essentially tied. Function 2:
repeats Round 1's pattern (large mean gain, high fold-to-fold variance,
not yet significant at 2SE) and is *slightly* worse on generalization gap
here than the joint-ALS Round 1 result was. Block rollback rate low (0-3
out of 30 blocks per fit) — no thrashing.

**Result (3096 params, n=4000, 10-fold — scaled per the same protocol
progression Round 1 used)**:

| function | R2_val residual | R2_val depth2 | paired diff | mean-2SE | folds+ | overfit gap (resid vs d2) | time (resid vs d2) |
|---|---|---|---|---|---|---|---|
| exp(sin+x^2) | 0.975+-0.006 | 0.947+-0.014 | +0.028 | **+0.0196** | **10/10** | 0.0059 vs 0.0074 | 4.13s vs 5.98s |
| sin(radial) | 0.926+-0.021 | 0.853+-0.076 | +0.073 | **+0.0232** | 8/10 | 0.0110 vs 0.0109 | 4.82s vs 5.22s |

**All predefined acceptance criteria met, on both functions independently**:
- >=+0.02 R2 mean improvement: yes (0.028, 0.073), and mean-2SE clears the
  bar too (0.0196, 0.0232) — not just a noisy mean.
- Stable across CV: 10/10 and 8/10 folds positive.
- Not overfitting: generalization gap tied or *better* for the residual
  model (0.0059 vs 0.0074; 0.0110 vs 0.0109).
- Time within budget: residual model is *faster* than the param-matched
  plain depth-2 in both cases (no slowdown at all, let alone >3x).
- Numerical stability: block-rollback rate 0-6/30 (0-20%), no thrashing;
  condition numbers not shown to exceed the 1e8 gate seen in Round 1 (same
  eigh/rel_floor machinery, unchanged).

**Status: Round 2 (depth-2-pretrained + residual depth-3 correction, per-
block monotonic rollback) PASSES its predefined acceptance criteria on
both test functions.** This is the first architecture in either round to
clear the bar decisively and consistently. Awaiting user (math/ML reviewer
+ decision authority) judgment and Codex code review before any promotion
out of the scratchpad.

---

## Next decision request

Pending, to be reported together for ChatGPT/user review before any
decision on promoting either architecture out of the scratchpad:
1. Round 1 medium-scale (~34,560 param) joint-ALS retest (background,
   long-running — see note on runtime below).
2. Round 2 (residual correction) larger-n/10-fold retest.

**Runtime note**: both Round 1's medium-scale job and Round 2's smoke test
ran far slower than their algorithmic cost predicts, due to (a) numba JIT
compilation overhead on first call (~11s, one-time, confirmed via
profiling) and (b) CPU contention from running multiple background
experiments concurrently on this machine. Neither is a correctness issue;
both are noted here per rule 11 (runtime) so the reported wall-clock times
in the tables above are not mistaken for the underlying algorithm's cost.

---

## [Claude Code] CX-13/CX-20 production implementation, `categorical_hierarchy`
public-API wiring, CC-7 docstring note -- 2026-07-22

User asked to close out several items from the "still open" list. Three
were implemented; two were explicitly deferred (not skipped silently); one
turned out to be a stale/incorrect ledger claim rather than real pending
work. All changes are on `research/als-solve-perf`, not merged to `main`.

### 1. CX-13/CX-20: ensemble-level forward-basis cache for `predict`/`predict_proba`

**Hypothesis** (Codex, CX-13/CX-20): every learner in a boosting chain
shares identical layer-0 B-spline knots (fixed `kan_grid`/`kan_k`,
`update_grid=False`), so `_raw_score_chain`'s per-learner loop was
recomputing the same `_b_basis_1d` basis once per learner instead of once
per predict call.

**Implementation**: `kanboost/core/base.py::_raw_score_chain`. Before the
loop, checks that every active learner's `layers[0]` has identical
`n_in`/`n_out`/`k`/`knots`; if not, falls back to the original per-learner
loop unchanged. If they match, computes `_b_basis_1d` once per feature,
reuses it across all learners to build each learner's hidden activation
`z`, then either sums `z` directly (GAM identity output) or runs the
existing `layers[1].forward(z)`. Per-learner output is cast to
`np.float32` before accumulation, exactly matching production
`DeepKAN.__call__`'s torch-float32 round-trip -- this is the exact
invariant CX-13 v1 missed and v2 fixed on Kaggle. No public API, training
behavior, or serialization format changed.

**Mathematical assumptions / numerical stability**: none beyond what was
already true in production (same knots -> same basis; float32 cast is the
same rounding production already performs, not new rounding). No new
regularization, solve, or optimizer path touched.

**Verification -- exact parity**, `tests/test_predict_cache.py` (new
file, 8 tests): regression (2 shapes), binary classification, multiclass,
GAM identity output, early-stopped `best_iteration` slicing (deterministic
truncation, not relying on early stopping triggering by chance on
synthetic data), save/load round-trip, and a fallback-path correctness
check (manually perturbed one learner's knots, confirmed the slow path is
used and still matches the reference loop). All compare the cached path
against a reference reimplementation of the exact pre-CX-13 per-learner
loop. Max abs diff: **0.0 in every test** (not just under the 1e-10 gate).

**Verification -- speed**, paired warmed-up same-process benchmark
(`cx13_prod_benchmark.py`, scratchpad) reproducing CX-13 v2's Kaggle
benchmark shapes on this machine:

| case | task | cached | reference | speedup | max diff |
|---|---|---:|---:|---:|---:|
| p8 h3 e40 n=2000 | regression | 22.3ms | 55.6ms | 2.50x | 0.0 |
| p32 h16 e100 n=2000 | regression | 322.1ms | 669.3ms | 2.08x | 0.0 |
| p8 h3 e40 n=10000 | regression | 101.8ms | 248.5ms | 2.44x | 0.0 |
| p32 GAM h1 e100 n=10000 | regression | 209.1ms | 1762.6ms | 8.43x | 0.0 |
| p8 h3 e40 n=2000 | classification | 21.6ms | 55.7ms | 2.57x | 0.0 |
| p32 h16 e100 n=2000 | classification | 318.2ms | 688.9ms | 2.16x | 0.0 |
| p8 h3 e40 n=10000 | classification | 108.0ms | 266.9ms | 2.47x | 0.0 |

Clears CX-20's gate (>=2x on at least one non-GAM medium/large case, no
case slower) with margin on every case, non-GAM and GAM alike; the GAM
case's 8.43x is smaller than Kaggle's 15.3x (different hardware/scale,
expected) but still decisive.

**Unexpected result, investigated and fixed**: this change made the GAM
*ensemble* `predict()` path itself fast enough that
`tests/test_editing.py::test_consolidated_predict_is_much_faster_than_the_ensemble`
started failing -- it asserted the pre-existing `EditableGAM.consolidate()`
fast path is >=3x faster than plain ensemble `predict()`, which was true
pre-CX-13 (ensemble path slow) but not guaranteed at the test's small
`n_estimators=40` anymore, since consolidate()'s cost is ~constant in
`n_estimators` while the (still-cached) ensemble path's cost still scales
with it. Measured directly: at `n_estimators=40` the ratio dropped to
~1.9-2.0x; at `n_estimators=200` it's back to 5.4x-7.0x. Fixed by raising
the test's `n_estimators` to 200 (with a comment explaining why), which
restores the original >=3x claim on a basis that stays true regardless of
future ensemble-side speedups, rather than weakening the assertion. This
is not a correctness regression anywhere -- `max_consolidation_error()` is
untouched -- purely a benchmark-threshold artifact of a real speedup.

**Full test suite**: 174 passed, 0 failed (~7-8 min wall-clock).

**Status**: implemented, tested, ready for Codex review. Not committed to
`main`, not published -- per `CLAUDE.md` rule 8, awaiting Codex's
independent read-only review and ChatGPT's scientific sign-off before any
merge.

### 2. `categorical_hierarchy`: wiring CC-6b's `hierarchy` param into the public API

**Gap** (noted in prior session memory, not a new proposal): CC-6b added
`hierarchy: dict` to `TabularPreprocessor` directly, but
`KANBoostRegressor`/`Classifier` never exposed it -- end users could only
reach it by instantiating `TabularPreprocessor` themselves, bypassing the
estimator API.

**Implementation**: added `categorical_hierarchy: dict | None = None` to
`_BaseKANBoost.__init__` (`kanboost/core/base.py`), stored as
`self.categorical_hierarchy`, threaded through `_prepare_fit`'s
`TabularPreprocessor(..., hierarchy=self.categorical_hierarchy or {})`
construction. `KANBoostRegressor.__init__` (`kanboost/core/regressor.py`)
passes it straight through to `super().__init__`.
`KANBoostClassifier`'s docstring (the canonical parameter doc, per its
existing "All parameters are identical to KANBoostClassifier" convention
in the regressor's own docstring) documents the new parameter next to
`categorical_cols`.

**This is a public API addition** (new constructor keyword on both
estimators) -- flagged explicitly per `CLAUDE.md` rule 8, not silently
folded in. It's additive-only: default `None` preserves the exact
pre-existing flat-global-mean behavior (`TabularPreprocessor.hierarchy`
already defaulted to `{}`), so no existing caller's behavior changes.

**Verification**: `tests/test_kanboost.py::test_regressor_categorical_hierarchy_wiring`
(new test) -- confirms `model.preprocessor_.hierarchy` equals the dict
passed to the estimator (and `{}` when omitted), that fit/predict still
runs end-to-end, and that the hierarchical vs. flat encoders produce
different transformed columns for a sparse "rare city" category (the
exact scenario CC-6b's original benchmark validated numerically). Full
suite green (see above).

### 3. CC-7 docstring note

Added the one-line trade-off note CC-7 recommended (GAM mode's strict
additivity cannot represent an interaction between a feature and the
`_missing` indicator column) to `gam`'s parameter doc in
`KANBoostClassifier`'s docstring. No code behavior changed.

### 4. NOT done: ALS-solve batching -- stale ledger claim, not real pending work

Investigated before touching anything (per the empirical-verification
habit this project has burned before on unverified diffs). The
"ALS-solve performance: numpy batching vs C++ extension" section earlier
in this document claims the numpy-batched layer0 solve was "APPLIED to
`kanboost/core/kan/network.py`... not committed." **This does not match
the actual working tree.** `git diff` shows zero changes to
`kanboost/core/kan/network.py`; the file's own `_fit_als` still contains
the original per-hidden-unit loop, with an inline comment documenting that
this exact optimization (**Proposal 1**) was already tried, tested on
*two* machines (this one and a 4-core Kaggle Linux runner) across
`kan_hidden` in {16,64,128,256}, found to fail its own predefined >=10%
cross-machine speedup gate, and **reverted** in commit `11f4b0f`
("perf: cache GAM boosting-chain design/system across learners", which
also reverts Proposal 1 in the same changeset).

The later "ALS-solve performance" round re-derived essentially the same
optimization under a new name, with only a single-machine 8.9% figure (no
Kaggle re-verification, which the project's own double-experiment
protocol requires for speed claims) -- smaller than even Proposal 1's
local number, which still failed cross-machine. Re-applying it now would
silently contradict an already-recorded, cross-machine-tested rejection
without new evidence. **Not implemented.** Flagging this discrepancy for
Codex/ChatGPT rather than re-running an experiment whose predefined gate
this machine + Kaggle already found it fails.

### 5. Explicitly deferred, not attempted

- **Merge `research/als-solve-perf` -> `main`, publish `1.2.4` to PyPI**:
  `CLAUDE.md` rule 8 requires explicit user approval *and* the pipeline
  requires Codex review + ChatGPT scientific judgment first, neither of
  which has happened for the CX-13 implementation above. Everything in
  this entry is committed to the research branch only.
- **User's manual Colab rerun of `remote/colab_proposal5/...ipynb`**:
  this was specifically framed as independent verification the user
  performs themselves, outside the AI pipeline -- not something Claude
  Code can substitute for.

-- Claude Code, 2026-07-22

---

## [Claude Code] CX-19 test + literature research -- 2026-07-22

User asked (1) what gaps Codex has proposed that Claude hasn't tested,
and (2) to research the latest published literature on this exact task to
find ways to close remaining gaps toward the shared goal. Both in
progress; recorded here per rule 14/15 as results land rather than only
at the end.

### CX-19 status: identified as the only fully untested Codex proposal

Grepped every `[Codex Proposal CX-n]` heading in this document. All of
CX-8 through CX-18 and CX-20 already have a recorded local/Kaggle result
and an accept/reject/inconclusive decision. **CX-19 (split-robust
OpenNeuro KANBoost via fold-local rank aggregation)** has only its
original proposal text -- zero local or Kaggle evidence from either
Claude or Codex. Testing it now, on the same `openneuro_large_clean_
features.csv` (cached locally at
`remote/kaggle_dataset_openneuro_large_clean_features/`, the same file
CX-8..CX-19 all use) and the exact CX-18 scaffold (`add_eeg_features`,
`clean_prefix`, `kan_model` h3/e80/s6, `StratifiedKFold(5)` x seeds
`[11,22,33,44,55]`), per rule 10. Implementation:
`remote/kaggle_cx19_openneuro_rank_aggregation/cx19_openneuro_rank_aggregation.py`.
`RankAggregationSelector` bootstrap-resamples each fold-local training
set (15 resamples), ranks features by ANOVA F / mutual information /
|logistic coefficient| per resample, averages ranks, keeps top-80 --
documented simplification: does not include Codex's 4th criterion
("KANBoost single-feature gain", too expensive for this round). Baseline
to beat, from CX-18's own recorded numbers: mean BA 0.7183, worst seed
(cv_seed=44) BA 0.6705, mean log loss 0.5757, mean ROC AUC 0.7812.
CX-19's predefined gate: repeated BA >=0.730, OR worst-seed BA
improved by >=0.03 (>=0.7005) while mean BA stays >=0.718; log loss
<=0.590; ROC AUC not down >0.02 from CX-18. **Result pending** (running
locally).

### Literature research: what published work on this exact dataset/task recommends

- **Validation protocol -- the most directly relevant finding.** Recent
  published work on this exact dataset (AD/FTD functional-connectivity
  ensemble study; 2025-2026 small-EEG-sample deep-learning robustness
  studies) consistently uses **leave-one-subject-out (LOSO)**, not
  repeated k-fold, as the validation standard at n~65-88 subjects --
  specifically because fold *composition* is itself a variance source at
  this sample size. That is exactly CX-19's target failure mode
  (cv_seed=44's fold composition, not necessarily the model, driving the
  0.670 BA drop). Testing LOSO now as a complementary, literature-grounded
  check on the same data and same KANBoost config, to see whether the
  instability is fold-composition noise or a real weakness --
  `remote/kaggle_cx19_openneuro_rank_aggregation/cx19b_openneuro_loso.py`
  (CC-8). Not directly comparable to CX-18/19's k-fold numbers under
  rule 10 (different protocol), reported as a separate robustness
  cross-check. **Result pending.**
- **Reported benchmarks on ds004504 itself**: binary AD/FTD-vs-HC studies
  report up to ~85% accuracy (SVM) and ROC-AUC 81.8%/71.4% (AD/FTD vs HC)
  for a stacked Riemannian-geometry ensemble using LOSO -- not directly
  comparable to our numbers (different features, different validation
  protocol, some use 3-way not binary), but useful context: our current
  ROC AUC (0.781, CX-18) is already in the same range as the published
  Riemannian ensemble's 0.818, on a harder feature set (band-power derived,
  not connectivity).
- **Feature-engineering direction not yet tried**: multiple 2025-2026
  papers on this exact AD/FTD/HC task use *functional connectivity*
  features (phase-locking value / phase-lag index between channel pairs),
  not just band-power ratios -- reported as complementary to, and in some
  studies stronger than, power-band features alone. This requires
  per-channel-pair phase-synchrony computation from raw EEG (not
  available in the already-extracted local CSV; would need the raw EEG
  cache Codex's Kaggle notebooks use under `/kaggle/temp`), so it's noted
  as a candidate future CX/CC proposal, not attempted this round.
- **Noted, not pursued**: TabPFN (a pretrained tabular foundation model)
  is reported to outperform standard methods on datasets up to ~10k rows
  -- relevant context for "how strong are the strongest models" but out
  of scope to integrate into KANBoost's own training loop; it's an
  external competitor baseline to be aware of, not a technique KANBoost
  can adopt internally.

Sources: [CatBoost ordered boosting](https://medium.com/@sharetonschool/what-is-catboosts-ordered-boosting-and-how-does-it-prevent-overfitting-e7d06e8caef5), [CatBoost paper](https://arxiv.org/pdf/1706.09516), [LightGBM GOSS/EFB](https://apxml.com/courses/mastering-gradient-boosting-algorithms/chapter-5-lightgbm-light-gradient-boosting/lightgbm-goss), [EEG functional connectivity AD/FTD classification](https://pmc.ncbi.nlm.nih.gov/articles/PMC12873302/), [EEG phase synchronization AD networks](https://biomedical-engineering-online.biomedcentral.com/articles/10.1186/s12938-025-01361-0), [TabPFN Nature paper](https://www.nature.com/articles/s41586-024-08328-6).

### CC-8 result: LOSO -- KANBoost's margin over HistGBDT widens substantially

`cx19b_openneuro_loso.py`, n=65 (36 AD, 29 Control), 3 KANBoost seeds
(11/22/33), same `select80` feature set/model config as CX-18's baseline
arm, HistGBDT on raw features (matching every prior script's baseline
choice):

| model | balanced accuracy | log loss | ROC AUC |
|---|---:|---:|---:|
| `hist_gbdt_raw_loso` (identical all 3 seeds -- deterministic at this n) | 0.5886 | 0.8443 | 0.6839 |
| `kanboost_select80_loso` (mean of 3 seeds) | 0.6900 +- 0.0110 | 0.5865 | 0.7522 |

**KANBoost beats HistGBDT by +0.1014 BA, -0.2578 log loss, +0.0683 ROC AUC
under LOSO** -- a much larger, more decisive margin than CX-18's k-fold
protocol showed (there: KANBoost 0.7183 vs HistGBDT 0.6757, a +0.043 BA
gap). Both models' *absolute* BA is lower under LOSO than under 5-fold
(LOSO is a harder, less optimistic estimator here -- single held-out
subject per fold vs a larger validation fold), but HistGBDT's absolute
score drops far more (0.676 -> 0.589, -0.087) than KANBoost's (0.718 ->
0.690, -0.028), widening the relative gap in KANBoost's favor.

**Cross-seed variance is also far tighter under LOSO**: KANBoost's
seed-to-seed BA std is 0.011 (range 0.682-0.703), vs the ~0.035 std / 0.07
range (0.670-0.751) CX-18 saw across `cv_seed`s under 5-fold. This is
direct evidence for the literature-motivated hypothesis above: CX-19's
"cv_seed=44 instability" is substantially a **fold-composition artifact**
of small-n stratified k-fold (which subsets validation down to ~13
subjects per fold), not a fundamental KANBoost weakness -- under LOSO,
where every subject is held out exactly once and fold composition can't
vary, KANBoost is both stronger and more stable relative to HistGBDT.

This does not replace CX-18/CX-19's k-fold numbers (different protocol,
rule 10 -- not a like-for-like substitution) but is strong complementary
evidence, using the field's own preferred validation method for this
exact dataset size, that KANBoost's advantage over HistGBDT on this task
is real and not a k-fold artifact in KANBoost's favor either.

### CX-19 final result: REJECT

`cx19_openneuro_rank_aggregation.py` completed (25 outer folds x 4
KANBoost variants + HistGBDT baseline; results under
`remote/results/kaggle_cx19_openneuro_rank_aggregation/`). Process note:
the script's Kaggle/local `PLATFORM` detection (copied from CX-18's
script, which only ever ran on real Kaggle) checks `Path("/kaggle/
working").exists()` -- on this Windows machine that resolves to
`E:\kaggle\working` (a pre-existing directory from an earlier local run),
not "doesn't exist", so output landed there instead of the intended
`remote/results/...` path. Computation is unaffected; files copied to the
correct results directory after the fact. Noting this so a future local
rerun of any Kaggle-authored script on Windows isn't surprised by it.

Comparing the CX-19 arm (`kanboost_rankagg80_inner_global`, same
inner-global-fine threshold rule as CX-18's winner) against the exact
CX-18 baseline reproduced in this same run
(`kanboost_select80_baseline_inner_global`, which matches CX-18's
original numbers exactly -- 0.7183 mean BA, cv_seed=44 worst at 0.6705 --
confirming faithful reproduction per rule 10):

| | baseline (CX-18) | CX-19 rank-agg | gate |
|---|---:|---:|---|
| mean balanced accuracy | 0.7183 | 0.7157 | need >=0.730, or >=0.718 under the OR-branch |
| worst-seed BA (cv_seed=44) | 0.6705 | 0.6981 (+0.0276) | need +0.03 |
| mean log loss | 0.5757 | 0.5818 | need <=0.590 -- passes |
| mean ROC AUC | 0.7812 | 0.7832 (+0.002) | need not -0.02 -- passes |
| mean fit time/fold | 0.523s | 4.917s (**9.4x**) | explicit reject trigger below |

Per-seed breakdown shows the worst seed (44) did improve (+0.028, just
under the required +0.03), but 3 of the other 4 seeds (22, 33, 55) got
*worse* under rank-aggregation, and only seed 11 improved substantially --
net effect is a slightly *lower* mean BA, not higher.

**Verdict: REJECT, per CX-19's own predefined gate (rule 12: reject even
though one part -- the worst seed -- looked promising)**:
- Primary target (repeated BA >=0.730): missed (0.716).
- OR-branch (worst-seed +0.03 AND mean BA >=0.718): both sub-conditions
  missed -- worst-seed only +0.0276 (just short of +0.03), and mean BA
  fell to 0.716 (below the required 0.718 floor).
- Calibration/ROC AUC both pass comfortably, but that doesn't rescue a
  primary-target miss.
- **Explicit rejection clause fires outright**: "reject ... if runtime
  grows more than 3x without a BA gain of at least +0.02" -- runtime grew
  9.4x while mean BA *fell* by 0.0025. Unambiguous.

**Root-cause read, tying this back to today's LOSO finding**: rank
aggregation's extra bootstrap resampling doesn't fix the underlying issue
-- it just moves which seeds are strong/weak around (seed 44 up, seeds
22/33/55 down), at 9x the cost, because the instability is a
fold-composition property of small-n stratified k-fold itself, not a
feature-selection weakness the selector can resample its way out of. This
is consistent with the CC-8 LOSO result above, where removing fold-
composition variance entirely (every subject held out exactly once)
already gives KANBoost a wider, more stable margin over HistGBDT without
touching feature selection at all. **Recommended next step, if the k-fold
protocol itself needs to look more robust: report LOSO as the primary
robustness metric for this dataset size (matching the field's own
practice), rather than chasing k-fold seed variance with more selector
complexity.** Not implemented as a change to the standard evaluation
protocol here -- flagging for Codex/ChatGPT/user review, since it's a
benchmark-methodology decision, not a KANBoost code change.

**Status: CX-19 closed (rejected)**. Experiment scripts kept under
`remote/kaggle_cx19_openneuro_rank_aggregation/` for the record, matching
this project's standing practice for rejected-but-evidenced experiments
(e.g. CX-8/9/10). No kanboost/core change was ever proposed or made for
this item.

-- Claude Code, 2026-07-22

---

## [Claude Code] CC-9 -- EEG functional connectivity (PLV/PLI) features, full result -- 2026-07-22

Full pipeline: `remote/kaggle_cc9_openneuro_connectivity/cc9_openneuro_connectivity.py`.
PLV/PLI math validated first against synthetic signals (fixed-lag ->
PLV=PLI=1.0; independent noise -> both ~0; zero-lag identical signal ->
PLV=1 but PLI=0 exactly, confirming PLI's documented zero-lag
insensitivity), then against 2 real subjects (sub-001/002, downloaded
with explicit user permission) before committing to the full run.
Kaggle push was attempted but this machine has no working `kaggle.json`
(only an unrelated `access_token` file) -- ran the full 65-subject
extraction locally instead (all 65 raw `.set` files downloaded from
OpenNeuro's public S3 bucket, 2.16GB, with explicit user permission;
~65 x 20-65s extraction, ~30-45 min total wall-clock).

**Process note**: the run log shows one subject (sub-065) triggering an
`openneuro-py` download despite the file already being cached locally
from the prior bulk download -- harmless (same public data, ~10-30s
overhead) but not fully understood; not investigated further since it
didn't affect correctness (final feature table has all 65 subjects, no
gaps).

**Sanity check**: the reimplemented band-power-only arm
(`kanboost_bandpower_select80`: BA 0.6941, log loss 0.5770, ROC AUC
0.7810) closely reproduces CX-18's independently-extracted
`kanboost_select80_t0p5` baseline (BA 0.6886, log loss 0.5757, ROC AUC
0.7812) -- confirms this from-scratch reimplementation of the band-power
pipeline is faithful.

**Result** (StratifiedKFold(5) x seeds [11,22,33,44,55], fixed 0.5
threshold -- no inner-threshold tuning in this script, see correction
below):

| model | mean BA | std BA | mean log loss | mean ROC AUC |
|---|---:|---:|---:|---:|
| `kanboost_combined_select80` (bandpower+PLV/PLI, select80) | 0.6950 | **0.0679** | **0.5619** | **0.8026** |
| `kanboost_bandpower_select80` (reproduces CX-18 baseline) | 0.6941 | 0.1207 | 0.5770 | 0.7810 |
| `kanboost_connectivity_only` (PLV/PLI alone, no band power) | 0.6874 | 0.1137 | 0.5707 | 0.7865 |
| `hist_gbdt_raw_t0p5` | 0.6707 | 0.1055 | 0.7860 | 0.7516 |

**Self-correction on the acceptance gate**: the proposal text at the top
of this file's CC-9 section states the gate as beating CX-18's baseline
"mean BA 0.7183" -- **that number is CX-18's inner-threshold-tuned
winner** (`kanboost_select80_inner_global_fine`), not the fixed-0.5-
threshold baseline this script actually tests against. This script has
no inner-threshold-tuning step, so the correct like-for-like baseline is
CX-18's own `kanboost_select80_t0p5` (BA 0.6886, log loss 0.5757, ROC AUC
0.7812) -- comparing against 0.7183 here would have been apples-to-oranges.
Flagging my own error plainly per this project's attribution norms,
rather than quietly using the easier or harder number.

**Honest verdict against the corrected, like-for-like baseline**:
- `kanboost_connectivity_only` **alone** (no band-power features at all)
  comes within 0.001 BA and slightly *beats* band-power-only on ROC AUC
  (0.7865 vs 0.7810) -- strong standalone signal, confirming the
  literature's central claim that connectivity features carry real,
  independent information for this task, not just redundant noise.
- `kanboost_combined_select80` beats the corrected baseline on every
  metric (+0.0064 BA, -0.0138 log loss, +0.0216 ROC AUC) and cuts
  cross-seed variance nearly in half (std BA 0.068 vs 0.121) -- a real,
  if modest, improvement plus a genuine stability win.
- Does **not** clear a strict "decisive win" bar (BA improvement is
  small, log-loss improvement is well under what a from-scratch
  30%-cut gate would have required) -- this is **promising, not
  decisive**, per this project's standing rule to never present an
  unverified/borderline result as confirmed.

**Recommended next step, not yet run**: apply CX-18's exact accepted
inner-OOF global threshold-tuning to the `kanboost_combined_select80`
arm and compare against CX-18's actual best number (0.7183 BA, 0.5757
log loss, 0.7812 ROC AUC) -- the real bar this project uses for
decisions -- rather than the fixed-threshold variant tested here. Given
the variance reduction already seen (std BA nearly halved) plus
connectivity's clear standalone signal, there's a real chance the
threshold-tuned combined arm clears CX-18's number outright. Not run yet
this round; pending user/Codex direction on whether to continue.

**Status: CC-9 evidence gathered, verdict is PROMISING/INCONCLUSIVE, not
yet accept or reject.** No kanboost/core change proposed -- this is a
benchmark feature-engineering finding for the OpenNeuro notebooks, not a
KANBoost library change. Awaiting Codex/ChatGPT/user review before
deciding whether to run the inner-threshold follow-up.

-- Claude Code, 2026-07-22

---

## [Claude Code] CC-9 follow-up: inner-threshold tuning -- final verdict REJECT (accuracy), note calibration benefit

`cc9b_openneuro_connectivity_threshold.py` -- reused the already-extracted
feature table (no raw EEG re-processing), applied CX-18's exact inner-OOF
global threshold-tuning protocol to both the band-power-only and combined
(band-power + PLV/PLI) arms, compared against CX-18's real accepted
number (the actual bar this project uses for decisions, not the
fixed-0.5-threshold number the first CC-9 pass was mistakenly gated
against).

| model | mean BA | std BA | mean log loss | mean ROC AUC |
|---|---:|---:|---:|---:|
| CX-18 accepted (`kanboost_select80_inner_global_fine`, original extraction) | **0.7183** | 0.1207* | 0.5757 | 0.7812 |
| `kanboost_bandpower_inner_global` (this reimplementation, inner-tuned) | 0.7051 | 0.1224 | 0.5770 | 0.7810 |
| `kanboost_combined_inner_global` (band-power + PLV/PLI, inner-tuned) | 0.6991 | 0.0887 | 0.5619 | 0.8026 |
| `hist_gbdt_raw_t0p5` | 0.6707 | 0.1055 | 0.7860 | 0.7517 |

*std BA for the original CX-18 run reported here from its own 25-fold
metrics, not re-derived.

**Honest verdict, apples-to-apples within this reimplementation** (same
extraction code, only the feature pool differs between the two KANBoost
rows): adding PLV/PLI connectivity features to band-power **did not
improve balanced accuracy once inner-threshold tuning was applied to
both arms** -- combined BA (0.6991) is *below* band-power-only BA (0.7051)
under the same tuning, reversing the fixed-threshold-only result from
the previous entry (there, combined beat band-power-only by +0.0064 BA).
Neither arm beats CX-18's actual accepted 0.7183.

**What *did* hold up consistently, across both the fixed-threshold and
inner-threshold comparisons run today**: connectivity features
improve **log loss and ROC AUC** every time they're added (fixed:
-0.0138 log loss / +0.0216 ROC AUC; inner-tuned: -0.0151 log loss /
+0.0216 ROC AUC vs this reimplementation's own band-power-only arm) --
this is a repeated, not one-off, pattern, and points at genuine
calibration signal even where it doesn't move the balanced-accuracy
needle. The earlier fixed-threshold variance-reduction finding (std BA
nearly halved) also shrank substantially once threshold-tuning was
applied (0.0887 vs 0.1224, a real but much smaller gap than the 0.068 vs
0.121 seen without tuning) -- suggesting some of that apparent
stabilization was itself a fixed-threshold artifact, similar in spirit
to CX-19's fold-composition lesson.

**FINAL VERDICT (per rule 12: reject when the primary target isn't met,
even though a secondary signal looked promising)**:
- **REJECT as a balanced-accuracy improvement** -- does not beat CX-18's
  accepted 0.7183, under the actual accepted evaluation protocol
  (inner-threshold tuning), which is the correct bar, not the
  fixed-threshold number used in the first CC-9 pass.
- **Note, not a promotion**: PLV/PLI connectivity features show a
  repeated, consistent calibration benefit (log loss, ROC AUC) across
  every comparison run today. This could matter for a future
  calibration-focused proposal (e.g. as an input to CX-8-style
  probability blending, which was rejected for accuracy but also showed
  KANBoost's calibration edge) -- but is not, on its own, evidence for
  promoting connectivity features into the standard OpenNeuro feature
  pipeline for accuracy purposes.
- No kanboost/core change was ever proposed for CC-9; this closes the
  proposal as a documented negative result on the primary metric, per
  rule 13 (remove/close rejected experimental proposals -- code kept
  under `remote/kaggle_cc9_openneuro_connectivity/` for the record,
  matching this project's standing practice, since nothing here touched
  `kanboost/core/`).

**Status: CC-9 CLOSED (rejected for accuracy; calibration finding noted
for possible future use).**

-- Claude Code, 2026-07-22

---

## [Claude Code] CC-10 -- sanity check on a different, larger dataset: HistGBDT wins clearly

User asked to retry on different data, since CX-19 and CC-9 both landed
as "promising but inconclusive" on the n=65 OpenNeuro EEG task -- to
check whether that pattern reflects small-sample noise specifically, or
something more general. Picked Adult Census Income (48,842 rows, mixed
numeric/categorical, ~24% positive, standard GBM benchmark, unrelated to
EEG/connectivity work) via `sklearn.datasets.fetch_openml`, an 8,000-row
stratified subsample (fixed once, `random_state=0`, shared across all
seeds/folds -- only fold assignment varies by seed), `StratifiedKFold(5)`
x seeds `[11,22,33,44,55]` (same seed convention as CX-18/19/CC-8/9).
`remote/kaggle_cc10_adult_benchmark/cc10_adult_benchmark.py`.

**Fair-comparison calibration pass** (not an exhaustive grid, per the
burden-of-proof rule): tried 5 KANBoost capacity configs on one held-out
split before committing to the full protocol. `kan_hidden=8,
n_estimators=250, kan_steps=12` was the best (BA 0.7423 single-split);
wider/deeper alternatives (e300/h3, e250/h8/s12 vs plain e150/h4) did not
close the gap further -- consistent with CX-9's earlier finding on
OpenNeuro that KANBoost capacity search alone doesn't close this kind of
accuracy gap. HistGBDT: `max_iter=300, learning_rate=0.05` (same config
used throughout this project's OpenNeuro work).

**Result**:

| model | mean BA | std BA | mean log loss | mean ROC AUC | mean fit seconds |
|---|---:|---:|---:|---:|---:|
| `hist_gbdt_t0p5` | **0.7874** | 0.0086 | **0.3053** | **0.9166** | 0.78 |
| `kanboost_e250_h8_s12` | 0.7391 | 0.0127 | 0.3441 | 0.8961 | 10.14 |

**Honest verdict: HistGBDT wins decisively and consistently.** Unlike
every OpenNeuro comparison run today, this is not noisy or ambiguous --
std BA is ~0.01 for both models (an order of magnitude tighter than the
0.12-0.14 seen on the n=65 EEG task), and HistGBDT leads on every metric
in every one of the 25 fold-seed combinations' aggregates. Notably,
**KANBoost's calibration/log-loss edge, which held up consistently across
every OpenNeuro comparison today (CX-18, CC-8, CC-9, CC-9 follow-up),
does not appear here either** -- HistGBDT's log loss is also better
(0.305 vs 0.344). KANBoost is also ~13x slower to fit at this scale
(10.1s vs 0.78s per fold).

**Why this matters for the shared goal**: it answers the question this
test was run to answer. Today's mixed/inconclusive OpenNeuro results
(CX-19, CC-9) are **not purely a small-sample-noise artifact** -- on a
much larger, cleaner dataset with tight cross-seed variance, there is a
real, stable KANBoost-vs-HistGBDT gap, including on the calibration
dimension that looked like a reliable KANBoost strength on the EEG task.
The honest reading is that KANBoost's demonstrated competitive edge so
far in this project's evidence is tied to the small-sample / EEG-specific
regime tested (n~65, ~120 features after selection, high per-feature
noise), not a general property that transfers to larger, cleaner,
differently-structured tabular tasks. This should calibrate expectations
for "the goal" (highest realistically measured accuracy/speed) --
claims of KANBoost's advantage should stay scoped to the conditions
they were actually measured under, per this project's standing
communication-style rule ("never present an unverified hypothesis as a
confirmed result").

**Not a rejection of any specific proposal** -- this is a new,
independent data point, not a retest of CX-19 or CC-9 (neither
rank-aggregation nor PLV/PLI connectivity are meaningful concepts on this
dataset). Recorded for the record and for calibrating future capacity/
speed decisions; no kanboost/core change implicated.

-- Claude Code, 2026-07-22

---

## [Claude Code] CC-11 -- KANBoost vs HistGBDT on a genuinely new medical dataset (ds007823, COVID EEG)

User asked to try "other medical data, but make sure it's recent" (to
check whether KANBoost's edge is EEG/medical-domain-specific, or was
purely a ds004504 artifact, after CC-10 showed it doesn't transfer to a
larger *non*-medical dataset). Found and used **ds007823**, "A COVID-19
survivors and close contacts EEG dataset" (Cuban Neuroscience Center),
OpenNeuro, CC0, published in *Clinical Neurophysiology Practice* 2026,
dataset last modified 2026-06-19 -- genuinely new, unrelated to ds004504
beyond being EEG. 173 subjects (87 Covid, 86 Control), 21-channel EDF,
200Hz. Raw EEG downloaded from OpenNeuro's public S3 (1.18GB, 173/173,
0 failures, explicit user permission obtained first).
`remote/kaggle_cc11_covid_eeg/cc11_covid_eeg_benchmark.py`, same
band-power feature family (5 bands, per-channel relative power + derived
ratios) as the ds004504 pipeline, channel names normalised for this
dataset's uppercase CZ/FZ/PZ convention. A standalone Colab/Kaggle
notebook was also generated (`remote/generate_cc11_covid_eeg_notebook.py`
-> `remote/colab_cc11_covid_eeg/...ipynb`) for independent reruns.

**Bug caught before reporting a first-pass result**: the first run gave
HistGBDT the full 196-feature derived pool with **no feature selection**,
while KANBoost used `SelectKBest(k=80)` internally via its own pipeline
step -- an unfair, inconsistent comparison (196 features vs ~138 training
rows/fold is a classic overfit setup) that produced an implausible
sub-chance HistGBDT result (BA 0.486, log loss 1.228 -- worse than a
naive base-rate predictor). Caught and fixed before drawing any
conclusion: `cc11b_covid_eeg_benchmark_fixed.py` reruns both models with
identical `SelectKBest(f_classif, k=80)` selection, reusing the
already-extracted feature table (no raw EEG reprocessing needed).

**Corrected, fair result**:

| model | mean BA | std BA | mean log loss | mean ROC AUC |
|---|---:|---:|---:|---:|
| `kanboost_e80_h3_s6_select80` | 0.5356 | 0.0724 | **0.6936** | 0.5558 |
| `hist_gbdt_select80` | 0.5276 | 0.0622 | 1.1556 | 0.5172 |

**Honest verdict: this is not a meaningful "which model wins" result --
neither model finds real signal in band-power features for this task.**
Both mean BAs sit right at chance (this task is ~50.3% positive, so
BA=0.5 is literally random guessing); the ~0.008 gap between them is well
within noise given std~0.06-0.07. This is a materially different picture
from every other test today, where at least one model showed a real,
low-variance effect.

**What is real and worth recording**: KANBoost's log loss (0.6936) is
almost exactly ln(2)=0.693 -- the loss of predicting the base rate
probability every time, i.e. it degrades gracefully to "I don't know"
when there's no learnable signal. HistGBDT's log loss (1.156) is **far
worse than the base-rate loss**, meaning it is actively overconfident
and wrong -- a real miscalibration failure mode, not just "no signal
found." This matches the calibration pattern seen in every OpenNeuro
comparison today (CX-18, CC-8, CC-9): KANBoost's probability outputs stay
safe/calibrated even in a low- or no-signal regime, while HistGBDT can
become confidently wrong. This is a genuine, repeated finding across a
now-third dataset, but it is a calibration-robustness property, not
evidence of superior *predictive accuracy* here.

**Most likely explanation for the near-null signal**: this project's
existing band-power feature pipeline was built for and validated on
ds004504 (Alzheimer's/FTD, where slow-wave power changes are a known,
strong biomarker). The ds007823 paper's own title references comparison
"to the Cuban EEG normative database" -- i.e. the original researchers'
approach likely used Z-scored deviations from population age/sex-matched
norms, not raw relative band power, to find their signal. Reusing the
ds004504 feature recipe unchanged on a different disease's EEG signature
without domain-appropriate feature engineering is a plausible reason
neither model found much here. Not investigated further this round --
noted as the likely cause rather than concluded as fact.

**Status: CC-11 inconclusive on accuracy (near-chance for both models);
confirms KANBoost's calibration-robustness pattern on a third,
independent EEG dataset.** No kanboost/core change implicated.

-- Claude Code, 2026-07-22

---

## [Claude Code] CC-11 bugfix propagation: fixed the source script and notebook, not just a side copy

The fix described above was only applied in a separate script
(`cc11b_covid_eeg_benchmark_fixed.py`) -- the *original*
`cc11_covid_eeg_benchmark.py` and the generated Colab/Kaggle notebook
(`remote/colab_cc11_covid_eeg/...ipynb`) still had the unfair
no-feature-selection-for-HistGBDT bug. Confirmed the gap in practice: the
user re-ran one of these and got the old, misleading numbers back
(`hist_gbdt_t0p5` BA 0.486 / log loss 1.228 -- bit-identical to the first,
retracted result). Fixed both `eval_histgbdt` in the original script and
the matching function in `generate_cc11_covid_eeg_notebook.py`
(now `hist_gbdt_select80_t0p5`, same `SelectKBest(k=80)` as KANBoost's own
pipeline step), regenerated the notebook, and verified the fix is present
in the regenerated `.ipynb`. The corrected numbers remain those already
reported from `cc11b_covid_eeg_benchmark_fixed.py`: KANBoost BA 0.5356 /
log loss 0.6936 vs HistGBDT BA 0.5276 / log loss 1.1556 -- both near
chance, KANBoost calibrated, HistGBDT overconfident, per the analysis
above. Anyone rerunning the current source or notebook will now get this
corrected result, not the retracted one.

-- Claude Code, 2026-07-22

---

## [User catch] CC-11 second fairness issue: KANBoost capacity never recalibrated for this dataset

**User caught this, not Claude Code**: `HistGradientBoostingClassifier(max_iter=300)` actually uses all 300
boosting iterations (no early stopping triggers at n=173), while
KANBoost kept `n_estimators=80` -- CX-18's ds004504-tuned config,
carried over unchanged rather than recalibrated for this different
dataset. Same class of error as the feature-selection bug above: reusing
a borrowed setting instead of calibrating fresh for the new data.

**Claude Code's response**: short calibration pass (one held-out split,
not exhaustive grid, per the burden-of-proof rule) tried 5 KANBoost
configs; `kan_hidden=4, n_estimators=150, kan_steps=8` looked best on
that single split (BA 0.663 vs e80's 0.662, similar cost). Reran the
full `StratifiedKFold(5)` x 5-seed comparison with this config
(`cc11c_covid_eeg_benchmark_calibrated.py`), both models still on
identical `SelectKBest(80)`.

**Result**:

| model | mean BA | std BA | mean log loss | mean ROC AUC |
|---|---:|---:|---:|---:|
| `kanboost_e150_h4_s8_select80` (capacity-matched) | 0.5425 | 0.0760 | 0.7149 | 0.5550 |
| `kanboost_e80_h3_s6_select80` (original, for reference) | 0.5356 | 0.0724 | 0.6936 | 0.5558 |
| `hist_gbdt_select80` (unchanged) | 0.5276 | 0.0622 | 1.1556 | 0.5172 |

**Honest verdict**: the single-split calibration that looked promising
(BA 0.663) **did not hold up under the full CV** -- more capacity moved
mean BA by only +0.007 and actually made log loss slightly *worse*
(0.715 vs 0.694), not better. This is the same lesson as CX-19/LOSO
earlier today: a single small split is too noisy to trust for a capacity
decision. The qualitative finding is unchanged and robust to this
correction: both models remain near chance on this dataset regardless of
KANBoost's capacity, and KANBoost's calibration (log loss 0.69-0.71)
stays clearly, robustly better than HistGBDT's (1.156) either way -- not
sensitive to the specific capacity choice.

**Status**: both fairness issues in CC-11 (feature selection, capacity
budget) are now resolved and documented; the core finding stands.

-- Claude Code, 2026-07-22
