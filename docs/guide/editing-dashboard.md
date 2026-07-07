# Editable models & dashboard

## Editable models (human-in-the-loop)

`kanboost.editing.consolidate(model)` collapses a fitted `gam=True`
ensemble's per-feature shape function — currently a sum of splines
across every boosting round — into one editable spline per feature.

This is conceptually similar to Microsoft's [GAM Changer](https://github.com/interpretml/gam-changer),
an editing tool for EBM (Explainable Boosting Machine): both let a
domain expert directly reshape a model's per-feature curves. The
difference is what happens *after* an edit. EBM's shape functions are
piecewise-constant bins, so checking monotonicity there is just
comparing adjacent bins — there's no notion of smoothness or
between-point behavior to verify, because there's no continuous curve
in the first place. KANBoost's feature is a genuine continuous
B-spline, so `enforce_monotone` re-derives a provably monotone
coefficient sequence after an edit — guaranteed for *every* point on
the curve, not just at the sampled locations used to build it — the
same variation-diminishing projection `monotone_constraints` uses
during training, not a best-effort correction.

```python
from kanboost.editing import consolidate

model = KANBoostRegressor(gam=True, kan_hidden=1, n_estimators=50)
model.fit(X_train, y_train)

gam = consolidate(model)  # multiclass classifier -> {class_label: EditableGAM}
print(gam.max_consolidation_error())  # worst per-feature fit error (call with feature=... for one feature)

gam.set_offset("age", x_range=(-0.2, 0.3), delta=0.5)   # shift a region
gam.set_values("region", x_range=(0.6, 1.0), value=0.0)  # pin a region flat
gam.enforce_monotone("income", sign=1)  # re-derive a provably monotone curve

report = gam.diff(X_val, y_val)  # per-feature deltas + before/after metric
gam.predict(X_val)                # exact, same interface as the original model
gam.save("edited_model.pt")
```

!!! tip "Also a fast predict path"
    `consolidate()` doubles as a fast predictor for `gam=True` models:
    one B-spline evaluation per feature instead of `n_estimators` full
    KAN forward passes. Measured **~30-50x faster prediction** (varies
    by hardware/model size; 1000-row, 6-feature, 40-estimator model),
    with `max_consolidation_error()` around 1e-6. See [Calibration](calibration.md)
    for KANBoost's known prediction-speed gap against tree ensembles --
    this closes it for GAM-mode models specifically (not non-GAM models,
    and not training speed).

## Interactive dashboard

`kanboost.experimental.dashboard_html` (see [Interpretability](interpretability.md))
is a zero-dependency static snapshot — good for sharing or archiving in
CI. `kanboost.dashboard` is a live, local Streamlit app for actually
exploring one of your own fitted models: feature importances,
`plot_feature` curves, `symbolic_report` (GAM mode), `feature_interaction`,
per-row `explain_row`, and — for a single-chain `gam=True` model
(regressor or binary classifier; not yet multiclass) — a panel to
live-edit shape functions via `kanboost.editing.EditableGAM`
(`set_offset`, `enforce_monotone`, `diff`, `save`), with the before/after
curve redrawn immediately.

Requires `pip install kanboost[dashboard]`.

```python
from kanboost.dashboard import launch

launch("model.pt")                      # opens a local browser tab
launch("model.pt", data_path="X.csv")   # preload a dataset to explore
```

or from the command line:

```bash
python -m kanboost.dashboard model.pt X.csv
```

This runs a local server for one person exploring one model, not a
hosted multi-tenant service — see [Serving & observability](serving.md)
for that.
