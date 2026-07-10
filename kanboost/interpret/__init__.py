"""kanboost.interpret -- symbolic export, editable models, interaction
detection, and other interpretability tooling for a fitted `gam=True`
KANBoost model."""

from .symbolic import (
    export_symbolic, explain, symbolic_summary, SymbolicModel,
    refit_constants, refit_constants_from_model, formula_fidelity,
    stability_across_seeds, stability_across_sample_sizes, distill_equation,
)
from .editing import consolidate, EditableGAM
from .interactions import friedman_h, check_additive_sufficiency
from .experimental import (
    suggest_constraints, audit_monotonicity, symbolic_export,
    predict_interval, explain_row, dashboard_html,
)
