"""
The actual Streamlit script for `kanboost.dashboard`. Never import this
module directly -- it's executed by Streamlit itself (via
`kanboost.dashboard.launch()`), reads its model/data paths from the
`KANBOOST_MODEL_PATH`/`KANBOOST_DATA_PATH` env vars `launch()` sets, and
assumes streamlit is already installed (that's `launch()`'s job to
check, with a clear error, before ever getting here).
"""

import os

import matplotlib
matplotlib.use("Agg")  # headless: no GUI backend, avoids main-thread warnings/failures
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from kanboost.dashboard import (
    explain_row_dataframe,
    importances_dataframe,
    interaction_dataframe,
    model_overview,
)
from kanboost.editing import EditableGAM, consolidate
from kanboost.experimental import audit_monotonicity
from kanboost.serving import _load_any

st.set_page_config(page_title="KANBoost Dashboard", layout="wide")


@st.cache_resource
def _load_model(path: str, mtime: float):
    """`mtime` is part of the cache key purely so re-running `launch()`
    against a re-saved model at the same path invalidates the cache."""
    return _load_any(path)


def _load_data(path: str | None):
    if path is None:
        return None
    return pd.read_csv(path)


model_path = os.environ.get("KANBOOST_MODEL_PATH")
if not model_path:
    st.error("No model path set. Launch this via `kanboost.dashboard.launch(model_path)`.")
    st.stop()

model = _load_model(model_path, os.path.getmtime(model_path))
info = model_overview(model)

data_path = os.environ.get("KANBOOST_DATA_PATH")
data = _load_data(data_path)

st.title("KANBoost Dashboard")
st.caption(f"{info['class']} -- {info['n_features']} features -- device: {info['device']}"
           + (" -- GAM mode" if info["gam_mode"] else ""))

if data is None:
    uploaded = st.sidebar.file_uploader("Upload a CSV to explore (optional)", type=["csv"])
    if uploaded is not None:
        data = pd.read_csv(uploaded)
if data is None:
    st.sidebar.info(
        "No dataset loaded -- feature-importance/shape-function panels still work "
        "(they don't need data), but row explanations, interaction scores, and the "
        "edit panel's fidelity/metric diff need one. Pass `data_path=` to `launch()` "
        "or upload a CSV."
    )

tab_names = ["Overview", "Shape functions", "Explain a row", "Interactions"]
if info["gam_mode"] and not info["is_multiclass"]:
    tab_names.append("Edit (experimental)")
tabs = st.tabs(tab_names)

# ------------------------------------------------------------------
with tabs[0]:
    st.subheader("Model")
    st.json(info)
    st.subheader("Feature importances")
    st.bar_chart(importances_dataframe(model).set_index("feature"))
    if info["monotone_constraints"] and data is not None:
        st.subheader("Monotonicity audit")
        st.json(audit_monotonicity(model, data, info["monotone_constraints"]))

# ------------------------------------------------------------------
with tabs[1]:
    feature = st.selectbox("Feature", info["feature_names"], key="shape_feature")
    if info["is_multiclass"]:
        st.info("Multiclass model: one curve per class, drawn on the same axes.")
    fig = model.plot_feature(feature)
    st.pyplot(fig.figure)
    if info["gam_mode"]:
        try:
            report = model.symbolic_report(data if data is not None else pd.DataFrame(
                {f: [0.0] for f in info["feature_names"]}
            ))
            per_feature = report.get(feature) if not info["is_multiclass"] else None
            if per_feature:
                st.caption("Closest symbolic fits (GAM mode): " +
                           ", ".join(f"{name} (R2={r2:.3f})" for name, r2 in per_feature))
        except Exception as exc:
            st.caption(f"Symbolic report unavailable: {exc}")

# ------------------------------------------------------------------
with tabs[2]:
    if data is None:
        st.warning("Load a dataset (sidebar) to explain a row.")
    else:
        row_index = st.number_input("Row index", min_value=0, max_value=len(data) - 1, value=0)
        st.dataframe(explain_row_dataframe(model, data, row_index=int(row_index)))

# ------------------------------------------------------------------
with tabs[3]:
    if data is None:
        st.warning("Load a dataset (sidebar) to compute feature interactions.")
    elif info["n_features"] < 2:
        st.info("Need at least 2 features.")
    else:
        try:
            st.dataframe(interaction_dataframe(model, data))
        except RuntimeError as exc:
            st.info(str(exc))  # kan_hidden=1 models raise this; it's expected, not a bug

# ------------------------------------------------------------------
if "Edit (experimental)" in tab_names:
    with tabs[4]:
        # Includes mtime so re-saving the model at the same path mid-session
        # (e.g. from another script) invalidates the stale EditableGAM here
        # too, matching _load_model's cache key above.
        state_key = f"editable_gam::{model_path}::{os.path.getmtime(model_path)}"
        if state_key not in st.session_state:
            st.session_state[state_key] = consolidate(model)
        gam: EditableGAM = st.session_state[state_key]

        st.caption(
            "Curve edits below are in the model's internal *scaled* feature "
            "units ([-1, 1]), not raw units -- see `plot_feature`'s x-axis for "
            "the same convention."
        )
        edit_feature = st.selectbox("Feature to edit", gam.feature_names, key="edit_feature")

        col1, col2 = st.columns(2)
        with col1:
            lo, hi = st.slider("Region (scaled x)", -1.0, 1.0, (-0.5, 0.5), key="edit_range")
            delta = st.number_input("Offset (raw score units)", value=0.0, key="edit_delta")
            if st.button("Apply offset"):
                gam.set_offset(edit_feature, (lo, hi), delta)
        with col2:
            if st.button("Enforce increasing"):
                gam.enforce_monotone(edit_feature, sign=1)
            if st.button("Enforce decreasing"):
                gam.enforce_monotone(edit_feature, sign=-1)
            if st.button("Reset this feature"):
                gam.reset(edit_feature)

        fig2, ax2 = plt.subplots()
        ax2.plot(gam.x_grid, gam._original_curves[edit_feature], label="original", alpha=0.6)
        ax2.plot(gam.x_grid, gam.curves[edit_feature], label="edited")
        ax2.set_xlabel(f"{edit_feature} (scaled)")
        ax2.set_ylabel("curve contribution")
        ax2.legend()
        st.pyplot(fig2)

        if data is not None:
            st.subheader("Effect of edits so far")
            y_col = st.selectbox(
                "Target column in the loaded data (optional, for a metric diff)",
                [None] + list(data.columns), key="edit_y_col",
            )
            # gam.diff() runs two full raw_score passes over `data` -- gated
            # behind a button rather than recomputed on every rerun (every
            # widget interaction on this page), which would be wasteful on
            # a large dataset.
            if st.button("Compute diff"):
                y_data = data[y_col] if y_col else None
                st.json(gam.diff(data, y_data))

        save_path = st.text_input(
            "Save edited model to",
            value="edited_model.pt",
            help="Relative paths are resolved against the server process's "
                 "working directory, not your browser -- use an absolute "
                 "path if you're unsure where that is.",
        )
        if st.button("Save"):
            gam.save(save_path)
            st.success(f"Saved to {save_path}")
