# Symbolic formula export

`model.symbolic_report(X)` and `kanboost.experimental.symbolic_export`
(see [Interpretability](interpretability.md)) give a human-readable
summary of each feature's best-fitting named function.
`kanboost.symbolic.export_symbolic` goes further: an actual executable
formula.

```python
from kanboost.symbolic import export_symbolic

model = KANBoostRegressor(gam=True, kan_hidden=1, n_estimators=50)
model.fit(X_train, y_train)

sym = export_symbolic(model, min_r2=0.85)  # multiclass classifier -> {class_label: SymbolicModel}

print(sym.to_sympy())          # a real sympy expression
print(sym.to_latex())          # ready to paste into a paper
print(sym.fidelity_report())   # {feature: {"kind", "r2", "candidate", "amplitude"}}
print(sym.symbolic_fraction()) # fraction of features that got a true closed form

sym.predict(X_test)            # standalone -- no torch/pykan needed at call time
sym.save("formula.pt")
```

## How it fits each feature

Each feature's exact aggregated shape function — the same one
`plot_feature`/`symbolic_report` use, and reusing
[`kanboost.editing.consolidate()`](editing-dashboard.md)'s already-correct,
already-tested per-feature centering rather than re-deriving curve
sampling from scratch — gets one closed-form candidate fit
(`c * fun(a*x + b) + d`, pykan's own `SYMBOLIC_LIB`: `x`, `x^2`, `x^3`,
`sin`, `cos`, `exp`, `log`, `sqrt`, `tanh`, `abs`) if some candidate
clears `min_r2`. If nothing does, that feature is kept as a numeric
(spline-interpolated) term instead of forced into a misleading formula
— `fidelity_report()` flags which features that happened to, so the
exported formula never silently claims a closed form that doesn't
actually fit.

!!! warning "R² alone doesn't mean a term matters"
    `fidelity_report()`'s `amplitude` field (the term's max-min range)
    matters alongside `r2`. A near-flat, unimportant feature can still
    score a deceptively high R² by fitting its own tiny wiggles —
    check `amplitude` against the other features' before treating a
    high-R² term as meaningful. This was caught empirically while
    building this feature: a genuinely irrelevant (near-noise) feature
    in a test case scored R²=0.996 with a `cos` fit, while its
    amplitude (~0.13) was more than 20x smaller than a real feature's
    (~3.0) in the same model.

## Predict is a lossy approximation

Because this refits every feature's spline as a parametric
approximation, `sym.predict()` is a *lossy* approximation of the
original model — unlike [`EditableGAM.predict`](editing-dashboard.md),
which is exact. `fidelity_report()` tells you how lossy, per feature;
low-`r2` (numeric-fallback) features contribute the same spline-exact
value they would in the original ensemble, so most of the approximation
error concentrates in the features that *did* get a symbolic fit.

## Multiclass

`export_symbolic` returns `{class_label: SymbolicModel}` for a
multiclass classifier — one independent formula per one-vs-rest chain,
matching `consolidate()`'s own convention.

## Quick ranked summary: `explain()`

For a top-`N` feature report instead of the full formula:

```python
from kanboost.symbolic import explain

for entry in explain(model, top_features=5, symbolic=True, simplify=True):
    print(entry["feature"], entry["importance"], entry["formula"])
```

Ranks features by `model.feature_importances_dict()` and attaches each
top feature's symbolic term (`simplify=True` runs `sympy.simplify()` on
it — cheap here, since it's one term, not the whole model).
`symbolic=False` skips formula extraction for a plain top-`N`
importance ranking. For a multiclass model, `explain()` uses each
feature's term from `model.classes_[0]`'s chain as a representative
formula — one-vs-rest chains can fit a feature differently per class,
so call `export_symbolic(model)` directly and index by class if you
need a true per-class formula.

## One-call report: `symbolic_summary()`

`explain()` ranks by *importance* and only fits candidates for its
top-`N`, so the full model formula it implies still has opaque
`g_<feature>(x)` placeholders for everything else. `symbolic_summary()`
instead ranks by `amplitude` — how much a feature's term actually moves
the prediction, which is the metric [the warning above](#r-alone-doesnt-mean-a-term-matters)
says to check, not raw importance — and by default fits candidates for
*every* feature, so `full_formula` only gets a placeholder for a
feature whose best candidate genuinely falls below `min_r2` (check each
term's `"kind"`), not for every feature outside some top-N cutoff:

```python
from kanboost.symbolic import symbolic_summary

result = symbolic_summary(model, min_r2=0.8)  # top_n=None -> every feature

for term in result["ranked_terms"]:  # most-amplitude-first
    print(term["feature"], term["candidate"], term["amplitude"], term["formula"])

print(result["full_formula"])   # sympy expression, the whole model
print(result["full_latex"])     # ready to paste into a paper
result["model"].predict(X_test) # the underlying SymbolicModel, same API as export_symbolic()'s
```

Pass `top_n=k` to restrict both the ranking *and* the candidate search
to the `k` most important features (by `feature_importances_dict()`) —
useful when the full per-feature candidate search is too slow for a
model with many features. With `top_n` set, `full_formula` goes back to
having a `g_<feature>(x)` placeholder for every feature outside that
cutoff, same as `explain()`. Multiclass: uses `model.classes_[0]`'s chain,
same convention as `explain()`.

Pass `min_amplitude=t` to drop any term whose `amplitude` falls below
`t` from *both* `ranked_terms` and `full_formula` — a low-amplitude term
barely moves the prediction regardless of its R², so leaving it in the
final equation adds length without adding real signal:

```python
result = symbolic_summary(model, min_r2=0.8, min_amplitude=0.05)
len(result["ranked_terms"])  # fewer terms than an unpruned call
```

## Leaner, more honest equations: parsimony, joint refitting, and stability

Four sharp, real limitations of the default export, and what addresses each:

**1. Candidate selection over-trusts R² alone.** Two candidates that fit
almost equally well (say `x^3` at R²=0.94 and `sin` at R²=0.95) pick the
higher-R² one even when it's periodic/more complex for a negligible
gain. `parsimony_margin` (default `0.0`, i.e. off) makes `export_symbolic()`
only prefer a more complex candidate (by a fixed ranking, roughly
`x < abs/x² < x³/tanh < sqrt/sin/cos < log/exp`) when it improves R² by
more than the margin:

```python
sym = export_symbolic(model, min_r2=0.8, parsimony_margin=0.02)
```

**2. Each term's constants are fit in isolation, never jointly.**
`export_symbolic()`'s default fits each feature's `(a, b, c, d)` to that
one feature's own isolated marginal curve — nothing about the default
fit ever looks at more than one feature at a time, so the *combined*
equation can be a worse joint approximation than a joint refit would
give. `refit_constants_from_model()` re-optimizes every symbolic term's
constants together against the real trained model's own raw score:

```python
from kanboost.symbolic import refit_constants_from_model, formula_fidelity

sym = symbolic_summary(model, min_r2=0.8)["model"]
sym_refit = refit_constants_from_model(model, sym, X_train)  # new SymbolicModel, doesn't mutate sym

formula_fidelity(model, sym, X_test, y_test)        # before
formula_fidelity(model, sym_refit, X_test, y_test)  # after -- compare
```

Only supports binary classifiers and regressors directly (a multiclass
one-vs-rest chain needs its own raw score computed by hand and passed
to the lower-level `refit_constants(sym, X_scaled, target)`).

**3. "Does the equation still rank like the model?" was never measured.**
`formula_fidelity(model, sym, X, y=None)` returns `max_abs_error`/
`mean_abs_error` always, plus `auc_model`/`auc_equation` when `model` is
a binary classifier and `y` is given — so `auc_equation >= auc_model - 0.005`
is a number you can check, not an assumption.

**4. A formula from one seed isn't *the* formula.** Boosting is
stochastic — a different seed can genuinely pick a different candidate
function for the same feature. `stability_across_seeds()` trains several
independent models and reports how often each feature's modal candidate
was chosen, alongside per-seed `formula_fidelity()`:

```python
from kanboost.symbolic import stability_across_seeds

def build_and_fit(X_train, y_train, seed):
    m = KANBoostClassifier(gam=True, kan_hidden=1, random_state=seed)
    m.fit(X_train, y_train)
    return m

report = stability_across_seeds(build_and_fit, X, y, n_seeds=5, top_n=5)
report["candidate_stability"]   # per-feature modal_candidate + modal_agreement
report["fidelity_per_seed"]     # auc_model/auc_equation per seed
```
