# Interpretability

## Feature importances & attribution

```python
importances = model.feature_importances_dict()

model.plot_feature("region")  # matplotlib partial-dependence-style plot

contributions = model.feature_contributions(X)  # native per-sample, per-feature attribution
```

`feature_contributions` is a native attribution method (read directly
off the first KAN layer's activations), not a post-hoc approximation
like SHAP.

## GAM mode: exact additive decomposition

```python
model = KANBoostRegressor(gam=True, kan_hidden=1)
model.fit(X_train, y_train)
```

Fixes each learner's output edge to the identity function, so the whole
ensemble reduces to an exact additive model `F(x) = c + sum_j g_j(x_j)`.
This is what makes hard monotonic constraints, symbolic-formula
extraction, and [editable models](editing-dashboard.md) all possible.

## Hard monotonic constraints

```python
model = KANBoostRegressor(
    gam=True, kan_hidden=1,
    monotone_constraints={"income": 1, "age": -1},
)
```

Requires `gam=True` and `kan_hidden=1`. Enforced by projecting each
edge's B-spline control points onto the monotone cone after every
optimizer step (the variation-diminishing property of B-splines) — a
real structural guarantee, not a soft penalty. Verified on California
Housing's `MedInc` in `examples/benchmark_uci.py`: the derivative stays
non-negative on *held-out test data*, not just training data.

## Symbolic-formula extraction

```python
report = model.symbolic_report(X)  # in GAM mode
```

Fits a small closed-form function library (`sin`, `cos`, `exp`, `x^2`,
`tanh`, ...) to each feature's exact aggregated shape function — one fit
per feature, not per learner, avoiding fragility from fitting a
non-convex symbolic regression per boosting round.

## Analytic derivatives

```python
d = model.predict_derivative(X, "income")
```

Exact, globally-defined derivative curves via autograd through the
whole ensemble — unlike a decision tree's zero/undefined derivative, or
an MLP's pointwise gradient.

## Post-hoc model surgery

```python
model.refine(X, new_grid=10)   # near-losslessly re-express on a finer spline grid
model.prune(X, threshold=0.01)  # zero out dead edges
```

No retraining from scratch, and no equivalent operation exists for a
fitted decision tree.

## Structural feature interaction

```python
scores = model.feature_interaction(X)  # kan_hidden > 1 only
```

Native structural interaction scores read directly off the trained
weights via pykan's own attribution machinery.

## Regularization

```python
model = KANBoostRegressor(lamb=0.01, lamb_l1=1.0, lamb_coefdiff=0.0)
```

Tunable smoothness/sparsity regularization on the learned splines
(pykan's own regularizers). Full-batch path only.

## Advisory tooling: `kanboost.experimental`

Convenience functions built entirely on the public methods above:

```python
from kanboost.experimental import (
    suggest_constraints, audit_monotonicity, symbolic_export,
    predict_interval, explain_row, dashboard_html,
)

# suggest which features look monotone in the raw data (advisory only)
constraints = suggest_constraints(X_train, y_train)

model = KANBoostRegressor(gam=True, kan_hidden=1, monotone_constraints=constraints)
model.fit(X_train, y_train)

# verify the constraint actually held on held-out data (not just training data)
print(audit_monotonicity(model, X_test))

print(symbolic_export(model, X_test))                  # compact human-readable summary
print(explain_row(model, X_test, row_index=0))          # top feature contributions for one row
predict_interval([model_seed0, model_seed1], X_test)     # mean/lower/upper/std across models
dashboard_html(model, X_test, y_test, path="report.html")  # one static HTML report
```

`suggest_constraints` is a heuristic, not a guarantee — always confirm
with `audit_monotonicity` on a model actually fit with the suggested
constraints.
