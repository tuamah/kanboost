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
