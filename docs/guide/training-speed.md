# Training speed

Profiling `fit()` shows the dominant per-round cost is each weak
learner's from-scratch Adam optimization loop — a fresh `KAN(...)` is
constructed every boosting round and run through the full `kan_steps`
budget, even though consecutive rounds share identical architecture
and, especially late in the ensemble, are fitting increasingly similar
residuals.

`kanboost.train.accel.fast_fit()` is an opt-in, drop-in replacement for
`model.fit(...)` that warm-starts each round's learner from the
previous round's fitted weights, so only the first round of a chain
needs the full step budget:

```python
from kanboost import KANBoostClassifier
from kanboost.train.accel import fast_fit

model = KANBoostClassifier(n_estimators=40, kan_steps=20)
fast_fit(model, X_train, y_train, eval_set=(X_val, y_val))
```

Measured on Breast Cancer Wisconsin (40 learners, `kan_steps=20`):
**3.37x faster** (11.70s → 3.48s) with AUC essentially unchanged
(0.9921 vs. 0.9893).

## How it works

`fast_fit` temporarily overrides the fitted model instance's
`_new_learner`/`_fit_learner`/`_boost_chain` methods for the duration
of one `fit()` call, then restores the originals — so it's implemented
entirely in a separate module, with zero edits to `_base.py`,
`classifier.py`, or `regressor.py`. Monotone constraints are enforced
identically to a normal `fit()` (`_apply_monotone_projection` still
runs after every optimizer step); only how each learner's weights are
*initialized* changes. Multiclass one-vs-rest chains are kept isolated
— a new class's chain never warm-starts from a different class's last
learner.

```python
fast_fit(model, X_train, y_train, warm_start_steps=5)  # override the default (kan_steps // 4)
```

## When to use it

This trades a small amount of per-round independence (classic boosting
fits each learner to the *current* residual from a fresh random init)
for speed. Always compare accuracy against a normal `fit()` on your own
data — it's a good fit for fast iteration during tuning
([`kantun`](tuning-with-kantun.md)), less clear-cut for a final
production model where the small accuracy delta matters more than
training wall-clock.

## Prediction speed: `consolidate_learners()`

`fast_fit` only addresses *training* time. For a model you're about to
deploy, `kanboost.train.consolidate.consolidate_learners()` shrinks the
already-fitted ensemble itself, cutting *prediction* time (and saved
model size) — it replaces consecutive groups of weak learners with one,
least-squares-refit to reproduce the group's summed output:

```python
from kanboost.train.consolidate import consolidate_learners

model.fit(X_train, y_train)
consolidate_learners(model, X_train, group_size=5)  # mutates model in place
model.predict_proba(X_test)  # same interface, fewer learners underneath
```

Measured on a real clinical benchmark (INSPIRE, `postop_icu`, 78K rows;
see the [INSPIRE gap-closing case study](../inspire_gap_closing_report.md)):
ensemble size 180→36, prediction time cut ~5x (54.1s→10.8s), for a small
accuracy cost (AUROC −0.0026, AUPRC −0.0056). Unlike
[`kanboost.interpret.editing.consolidate()`](editing-dashboard.md) (which
requires `gam=True` and returns a new `EditableGAM`), this works on any
fitted classifier/regressor and mutates `learners_` in place — no new
object, no editability, just fewer learners to evaluate at predict time.
As with `fast_fit`, this is a lossy approximation: compare accuracy
before/after on your own held-out data before relying on it.

## Training speed via fewer rounds: `fit_with_line_search()`

`kanboost.train.linesearch.fit_with_line_search()` replaces the fixed
`learning_rate` shrinkage with a per-round line search for the
loss-minimizing step size, inspired by
[GB-KAN](https://www.scitepress.org/Papers/2026/142468/142468.pdf) (an
independently published KAN-based boosting framework). It does **not**
make any individual round cheaper — the benefit is that fewer rounds
are needed to reach the same accuracy, which cuts total wall time
close to linearly in the round-count reduction:

```python
from kanboost import KANBoostClassifier
from kanboost.train.linesearch import fit_with_line_search, predict_proba_line_search

model = KANBoostClassifier(n_estimators=40)  # a fraction of what a
                                              # fixed-shrinkage fit() would need
fit_with_line_search(model, X_train, y_train)
proba = predict_proba_line_search(model, X_test)  # NOT model.predict_proba --
                                                    # each round has its own
                                                    # step size, not a shared
                                                    # learning_rate
```

Measured on INSPIRE (`postop_icu`, Medium scale, 50K rows): 40
line-search rounds matched *and slightly exceeded* 140 rounds of a
fixed-shrinkage baseline (AUROC 0.9407 vs. 0.9389, AUPRC 0.7636 vs.
0.7560) in **91.4s vs. 384.4s — a 4.2x wall-clock speedup**. At the
*same* round count, line search gives no consistent benefit (the 1-D
search itself adds a small per-round cost) — its value is specifically
in letting you use fewer rounds, not in improving each one.

**Scope**: binary `KANBoostClassifier` only (not multiclass, not
`KANBoostRegressor`), no `eval_set`/early stopping. Use
`predict_proba_line_search()`, not the base model's own
`predict_proba()` — each learner has its own step size here, unlike a
normal `fit()` where every learner shares `learning_rate`.

## Accuracy at the same round count: `fit_with_newton_boosting()`

`kanboost.train.newton.fit_with_newton_boosting()` is a different lever
on the same underlying idea GB-KAN's paper flags as unsolved for
KAN-based boosting: **second-order (Newton-step) boosting**. Instead of
fitting each weak learner directly to the raw pseudo-residual `y - p`,
it reweights using the logistic loss's second derivative
(`h = p*(1-p)`), fitting the learner to the Newton target
`(y - p) / h` with sample weight `h` — the same reformulation XGBoost
uses to derive its leaf values.

```python
from kanboost import KANBoostClassifier
from kanboost.train.newton import fit_with_newton_boosting

model = KANBoostClassifier(n_estimators=140)  # same round count as a normal fit()
fit_with_newton_boosting(model, X_train, y_train)
model.predict_proba(X_test)  # works via the standard API -- no special
                              # predict function needed, unlike
                              # fit_with_line_search
```

Measured on INSPIRE at the *same* round count as a normal fit, all
three scales: AUROC/AUPRC both improved consistently (Small: 0.9225→0.9246
/ 0.6707→0.6879; Medium: 0.9389→0.9411 / 0.7560→0.7686; Large:
0.9413→0.9434 / 0.7643→0.7757). **Fit time did not move consistently**
(neutral at Small, ~33% slower at Medium, ~19% faster at Large) — this
is an accuracy-oriented option, not a speed-oriented one; don't expect
a predictable time change.

**Tested and rejected**: combining this with `fit_with_line_search()`.
At every scale, the combination underperformed either technique alone
— the line search there optimizes against the *original* loss while
the learner was fit to the Newton-*reweighted* target, an
inconsistency, not a bug in either piece individually. Use one or the
other, not both.

**Scope**: binary and multiclass `KANBoostClassifier` (not
`KANBoostRegressor`). Multiclass fits one one-vs-rest Newton chain per
class, exactly like a standard multiclass `fit()`. Measured on Digits
(10 classes, sklearn): standard-ALS multiclass fit 29.5s/94.8% accuracy
vs. Newton-boosted multiclass 62.4s/97.4% -- accuracy improves further
in multiclass, but so does the per-round cost, since Newton reweighting
compounds across `n_classes` independent chains.

## A different weak-learner engine entirely: `fit_with_rfkan()`

Everything above keeps kanboost's standard weak-learner engine (Gauss-Newton
ALS, `_fit_als`) and changes only the target reweighting or step size.
`kanboost.train.rfkan.fit_with_rfkan()` replaces the engine itself: ALS
alternately refits both layers for up to 10 sweeps per learner, and
profiling traced most of the per-round cost to that alternation (a
fresh eigendecomposition every sweep, since the hidden representation
changes each time). RF-KAN instead freezes the input→hidden layer as a
fresh **random projection each round** (Random Features / ELM: Rahimi &
Recht 2007, Huang 2006) and solves the hidden→output layer with ONE
closed-form penalized least-squares solve — no alternation, no repeated
eigendecomposition.

```python
from kanboost import KANBoostClassifier
from kanboost.train.rfkan import fit_with_rfkan, predict_proba_rfkan

model = KANBoostClassifier(n_estimators=140)  # same round count as a normal fit
fit_with_rfkan(model, X_train, y_train)
proba = predict_proba_rfkan(model, X_test)  # NOT model.predict_proba --
                                             # each round's layer0 is
                                             # independently random
```

Measured on INSPIRE, same round count as a normal fit, all three scales
— accuracy matches (not just approximates) the standard ALS engine, at
3.7–5.3x less fit time:

| Scale | ALS (standard) | RF-KAN | Speedup |
|---|---|---|---:|
| Small | 27.6s, AUROC 0.9225/AUPRC 0.6707 | 7.4s, 0.9226/0.6709 | 3.7x |
| Medium | 265.5s, 0.9389/0.7560 | 62.4s, 0.9388/0.7561 | 4.3x |
| Large | 618.0s, 0.9413/0.7643 | 143.6s, 0.9413/0.7643 | 4.3x |

It composes with the two options above — `use_newton=True` gives the
**best accuracy** of any option in this guide, at the *same* speed as
plain RF-KAN (Small 0.9247/0.6878, Medium 0.9411/0.7688, Large
0.9433/0.7758 — better than standard ALS on every metric, 3.7–4.3x
faster); `use_line_search=True` gives the **best speed**, since RF-KAN's
speedup and line search's round-count reduction compound (Medium: 42
rounds instead of 140, 19.3s instead of 265.5s — 13.8x — with AUROC/AUPRC
both *exceeding* the full-round baseline: 0.9407/0.7643 vs. 0.9389/0.7560).

**`use_newton=True` and `use_line_search=True` together are rejected
with a `ValueError`** — measured at all three scales, the combination
is worse than either alone every time, for the same reason noted above
(line search optimizing against the original loss vs. a
Newton-reweighted fit target).

**The real tradeoff, stated plainly**: layer0 is a fresh random
projection every round here, not a representation that adapts to the
data the way standard ALS's does. This means KANBoost's native
interpretability tools — `feature_contributions()`, `plot_feature()`,
`symbolic_report()`, `feature_interaction()` — are **not meaningful**
on a model fitted this way, since there's no stable input-to-hidden
mapping to attribute through. That's kanboost's primary differentiator
against tree boosting, and RF-KAN gives it up for speed. Use
`fit_with_newton_boosting()`/`fit_with_line_search()` (which keep
standard ALS, interpretability intact) when that matters for a given
model; use `fit_with_rfkan()` when raw speed/accuracy is the priority
and you don't need to interpret that particular model afterward.

**Scope**: binary and multiclass `KANBoostClassifier` (not
`KANBoostRegressor`), same one-vs-rest convention as
`fit_with_newton_boosting()`. Measured on Digits (10 classes):
`use_newton=True` matched the best multiclass accuracy measured in this
guide (97.4%, tied with standard-ALS Newton boosting) at **7.6x its
speed** (8.2s vs. 62.4s) — the RF-KAN+Newton combination's advantage
over plain ALS widens in multiclass, since Newton's added cost compounds
across `n_classes` independent chains on the standard engine but not on
RF-KAN's already-cheap one.

## Speed, accuracy, AND interpretability together: `fit_with_ga2m()`

Every option above trades interpretability for speed/accuracy (`fit_with_rfkan`) or keeps interpretability but gives up some of RF-KAN's benefit. `kanboost.train.ga2m.fit_with_ga2m()` is the one engine in this guide that does not force that tradeoff: it restructures WHICH features feed each hidden unit (one per input feature -- a "main effect" -- plus a random subset of feature PAIRS re-sampled every round -- an "interaction") instead of RF-KAN's dense random mixing of all features per unit. Every hidden unit is attributable to exactly one feature or one named pair, so `main_effect_contributions()`/`pairwise_interaction_contributions()` give genuine, per-feature/per-pair attribution (GA2M / Explainable Boosting Machine style) -- unlike RF-KAN, where a unit's dense random mixture cannot be attributed to anything.

```python
from kanboost import KANBoostClassifier
from kanboost.train.ga2m import (
    fit_with_ga2m, predict_proba_ga2m,
    main_effect_contributions, pairwise_interaction_contributions,
)

model = KANBoostClassifier(n_estimators=42)  # a fraction of a normal fit's rounds --
                                              # line search is on by default here
fit_with_ga2m(model, X_train, y_train, n_pairs_per_round=10)
proba = predict_proba_ga2m(model, X_test)

main_effect_contributions(model)              # {feature: total contribution}, sorted desc
pairwise_interaction_contributions(model)     # {(feature_a, feature_b): total contribution}, sorted desc
```

**This is the best-performing engine measured in this entire guide** — line search is on by default (`use_line_search=True`) because the combination is what was actually validated: at every INSPIRE scale tested, GA2M+line-search beat every other option here, including `fit_with_rfkan(use_newton=True)` (previously the best), on **both** AUROC and AUPRC, at comparable or better speed:

| Scale | Rounds | Dense RF-KAN (full rounds) | **GA2M + line search** | Speedup |
|---|---:|---|---|---:|
| Small | 30 | 7.4s, 0.9226/0.6709 | 2.3s, **0.9283/0.6883** | 3.2x |
| Medium | 42 | 62.7s, 0.9388/0.7561 | 19.6s, **0.9477/0.7797** | 3.2x |
| Large | 54 | 146.5s, 0.9413/0.7643 | 45.4s, **0.9505/0.7918** | 3.2x |

**Without line search, GA2M underperforms dense RF-KAN on AUPRC** at every scale tested (a real, confirmed cost of restricting mixing to main effects + pairs instead of dense mixing) — line search is specifically what recovers this and then exceeds it, which is why it defaults on here unlike `fit_with_rfkan`.

A faster variant (independent per-hidden-unit solve instead of one joint solve, ~15% faster) was tested and intentionally **not** implemented here: it is measurably *less accurate for attribution specifically* — the joint solve used here correctly partitions credit between overlapping units (e.g. a feature's main effect and an interaction term involving that same feature); an independent solve does not, and can double-count shared signal between them. Use this module, not a faster-but-approximate variant, whenever the interpretation itself will be trusted, not just the predictions.

**Scope**: binary `KANBoostClassifier` only (not multiclass, not `KANBoostRegressor`).
