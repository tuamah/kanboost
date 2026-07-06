"""
kanboost.dashboard -- optional interactive Streamlit dashboard for a
saved KANBoost model.

Additive: streamlit is only imported inside `launch()`, so importing
kanboost (or even this module) never requires it. Install with
`pip install kanboost[dashboard]`.

Usage
-----
Programmatic:
    from kanboost.dashboard import launch
    launch("model.pt")                      # opens a local browser tab
    launch("model.pt", data_path="X.csv")    # preload a dataset to explore

As a script:
    python -m kanboost.dashboard model.pt [data.csv]

This launches a *local* Streamlit server for one user exploring one of
their own fitted models -- not a hosted multi-tenant service (see
`kanboost.serving` for that). For a zero-dependency, shareable static
snapshot instead of a live tool, see `kanboost.experimental.dashboard_html`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def launch(model_path: str, data_path: str | None = None, port: int = 8501) -> None:
    """Launch the dashboard against a model saved via `.save(path)`, and
    (optionally) a CSV dataset to explore it with. Blocks until the
    server is stopped (Ctrl-C), same as running `streamlit run` directly.
    """
    try:
        from streamlit.web import bootstrap
    except ImportError as exc:
        raise ImportError(
            "kanboost.dashboard requires streamlit; "
            "install with `pip install kanboost[dashboard]`."
        ) from exc

    if not Path(model_path).exists():
        raise FileNotFoundError(f"No such model file: {model_path!r}")
    if data_path is not None and not Path(data_path).exists():
        raise FileNotFoundError(f"No such data file: {data_path!r}")

    os.environ["KANBOOST_MODEL_PATH"] = str(model_path)
    if data_path is not None:
        os.environ["KANBOOST_DATA_PATH"] = str(data_path)
    else:
        os.environ.pop("KANBOOST_DATA_PATH", None)

    app_path = str(Path(__file__).parent / "_dashboard_app.py")
    flag_options = {"server.port": port, "server.headless": True}
    bootstrap.run(app_path, False, [], flag_options)


# ----------------------------------------------------------------------
# Data-assembly helpers, kept free of any streamlit import so they're
# testable without a running dashboard/browser. `_dashboard_app.py`
# (the actual Streamlit script) calls these; it's the only file here
# that imports streamlit.
# ----------------------------------------------------------------------

def model_overview(model) -> dict:
    """Small dict of model metadata for the Overview panel."""
    is_classifier = hasattr(model, "classes_")
    return {
        "class": type(model).__name__,
        "device": str(getattr(model, "device_", "unknown")),
        "gam_mode": bool(getattr(model, "gam", False)),
        "monotone_constraints": dict(getattr(model, "monotone_constraints", {}) or {}),
        "is_classifier": is_classifier,
        "is_multiclass": is_classifier and len(getattr(model, "classes_", [])) > 2,
        "n_features": len(model.preprocessor_.transformed_feature_names()),
        "feature_names": model.preprocessor_.transformed_feature_names(),
    }


def importances_dataframe(model):
    """`feature_importances_dict()` as a two-column DataFrame, sorted
    descending (the dict is already sorted, this just makes it
    plottable)."""
    import pandas as pd

    imps = model.feature_importances_dict()
    return pd.DataFrame({"feature": list(imps.keys()), "importance": list(imps.values())})


def interaction_dataframe(model, X, top_k: int = 10):
    """`feature_interaction(X)` as a DataFrame; raises the same
    `RuntimeError` `feature_interaction` does for `kan_hidden=1`."""
    import pandas as pd

    result = model.feature_interaction(X, top_k=top_k)
    return pd.DataFrame(
        [{"feature_pair": " & ".join(map(str, k)), "count": v} for k, v in result.items()]
    )


def explain_row_dataframe(model, X, row_index: int = 0, top_k: int = 8):
    """`kanboost.experimental.explain_row` as a DataFrame."""
    import pandas as pd
    from .experimental import explain_row

    rows = explain_row(model, X, row_index=row_index, top_k=top_k)
    return pd.DataFrame(rows)


def _main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m kanboost.dashboard <model_path> [data_path]")
        sys.exit(1)
    model_path = sys.argv[1]
    data_path = sys.argv[2] if len(sys.argv) > 2 else None
    launch(model_path, data_path)


if __name__ == "__main__":
    _main()
