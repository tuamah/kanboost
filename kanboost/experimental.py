"""
kanboost.experimental -- small, additive utilities built entirely on
KANBoost's existing public API (`predict_derivative`, `symbolic_report`,
`feature_contributions`, `feature_importances_dict`, `monotone_constraints`).
No changes to `_base.py`/`classifier.py`/`regressor.py` are needed for
anything here.

These close a specific gap: KANBoost already lets you *impose* a
monotonic constraint or *inspect* a shape function one call at a time,
but has no single call to (a) suggest which features are worth
constraining from the data itself, (b) verify a constraint actually
holds on new data, or (c) produce one glance-able report combining
several of the existing interpretability methods. Everything here is
advisory tooling around already-shipped guarantees, not a new guarantee
of its own -- `suggest_constraints` in particular is a heuristic, not a
proof; always confirm with `audit_monotonicity` on a model actually
fit with the suggested constraints.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def suggest_constraints(
    X, y, min_abs_corr: float = 0.25, min_consistency: float = 0.55, n_bins: int = 10,
) -> dict:
    """Suggest a `monotone_constraints` dict from raw training data, as a
    starting point for a `gam=True, kan_hidden=1` model -- **advisory
    only**, not a guarantee; always verify with `audit_monotonicity` on
    a model actually fit with the suggested constraints before trusting
    them.

    For each numeric column: computes the Spearman rank correlation with
    `y` (skips columns with |correlation| < `min_abs_corr`), then checks
    whether binning the column into `n_bins` quantile bins gives
    consistently increasing (or decreasing) bin-mean `y` -- more robust
    to noise than checking consecutive raw values, since it looks at the
    trend of local averages rather than point-to-point differences.
    Suggests `+1`/`-1` only if that consistency is >= `min_consistency`.
    Columns with fewer than 3 quantile bins after deduplication (e.g.
    binary/near-constant columns) are always skipped -- a monotonicity
    constraint is low-value on a feature with only a couple of distinct
    values, and there isn't enough resolution to judge consistency
    reliably anyway.
    """
    X = pd.DataFrame(X)
    y = np.asarray(y, dtype=float)
    out = {}

    for col in X.columns:
        s = pd.to_numeric(X[col], errors="coerce")
        ok = s.notna()
        if ok.sum() < 20 or s[ok].nunique() < 2:
            continue

        corr, _ = spearmanr(s[ok], y[ok])
        if np.isnan(corr) or abs(corr) < min_abs_corr:
            continue

        n_available_bins = min(n_bins, max(ok.sum() // 5, 1))
        try:
            bins = pd.qcut(s[ok], q=n_available_bins, duplicates="drop")
        except ValueError:
            continue
        bin_means = pd.Series(y[ok]).groupby(bins, observed=True).mean()
        if len(bin_means) < 3:
            # Binary/few-unique features fall through here (fewer than 3
            # quantile bins) and are silently skipped -- monotonicity
            # constraints are low-value on a feature with only a couple
            # of distinct values anyway, so this isn't worth a warning.
            continue

        diffs = np.diff(bin_means.to_numpy())
        consistency = np.mean(diffs >= 0) if corr > 0 else np.mean(diffs <= 0)
        if consistency >= min_consistency:
            out[col] = 1 if corr > 0 else -1

    return out


def audit_monotonicity(model, X, constraints: dict | None = None, tol: float = 1e-5) -> dict:
    """Verify whether `model.predict_derivative` actually has the sign
    `constraints` (defaulting to `model.monotone_constraints`) claims,
    on `X` -- catches a constraint that was requested but silently not
    enforced (e.g. a custom weak-learner backend that ignores
    `monotone_constraints` -- see `ROADMAP.md`), or one that holds on
    training data but not on `X`.

    Returns `{feature: {"constraint", "passed", "violation_rate",
    "min_derivative", "max_derivative"}}`.
    """
    constraints = constraints or getattr(model, "monotone_constraints", {}) or {}
    report = {}

    for feature, sign in constraints.items():
        d = np.asarray(model.predict_derivative(X, feature), dtype=float)
        bad = d < -tol if sign == 1 else d > tol

        report[feature] = {
            "constraint": "increasing" if sign == 1 else "decreasing",
            "passed": bool(not bad.any()),
            "violation_rate": float(bad.mean()),
            "min_derivative": float(d.min()),
            "max_derivative": float(d.max()),
        }

    return report


def symbolic_export(model, X, top_k: int = 1, min_r2: float = 0.8) -> str:
    """A compact, human-readable symbolic summary from
    `model.symbolic_report()`: for each feature, the single best-fitting
    named function (`sin`, `x^2`, `tanh`, ...) if its R^2 clears
    `min_r2`, else the feature is omitted (its shape isn't well
    described by any candidate -- the underlying spline is still exact,
    this is just a lossy human-readable approximation of it).

    For an actual executable/exportable formula (a real `sympy`
    expression, LaTeX, a standalone numpy predict function, and a
    fidelity report with per-feature amplitude, not just this text
    summary), see `kanboost.symbolic.export_symbolic` instead.
    """
    raw = model.symbolic_report(X, top_k=top_k)

    def one_chain(report, prefix="score"):
        terms = []
        for feature, candidates in report.items():
            if not candidates:
                continue
            name, r2 = candidates[0]
            if r2 >= min_r2:
                terms.append(f"{name}({feature})  # R2={r2:.3f}")
        return f"{prefix} ~= " + " + ".join(terms) if terms else f"{prefix}: no stable symbolic terms"

    lines = []
    if isinstance(raw, dict) and raw and isinstance(next(iter(raw.values())), dict):
        for cls, rep in raw.items():
            lines.append(one_chain(rep, prefix=f"class_{cls}"))
    else:
        lines.append(one_chain(raw))

    return "\n".join(lines)


def predict_interval(models, X, level: float = 0.90) -> dict:
    """A prediction interval from a list of independently fitted models
    (e.g. several random seeds), as `{"mean", "lower", "upper", "std"}`.

    This is a convenience wrapper around bagging-style variance, not a
    replacement for `KANBoostRegressor(objective="quantile", alpha=...)`,
    which fits calibrated conditional quantiles directly and is the
    better choice if you only need one specific quantile rather than an
    ensemble-variance-based spread.
    """
    if not isinstance(models, (list, tuple)):
        raise TypeError("models must be a list of fitted models")

    preds = np.vstack([np.asarray(m.predict(X), dtype=float) for m in models])
    alpha = (1 - level) / 2

    return {
        "mean": preds.mean(axis=0),
        "lower": np.quantile(preds, alpha, axis=0),
        "upper": np.quantile(preds, 1 - alpha, axis=0),
        "std": preds.std(axis=0),
    }


def explain_row(model, X, row_index: int = 0, top_k: int = 8) -> list:
    """Top-`top_k` feature contributions (by |value|) for one row of
    `X`, via `model.feature_contributions` -- works for both a single
    chain (regressor/binary classifier) and a multiclass classifier
    (uses the first class's chain; call per-class explicitly for more)."""
    contrib = model.feature_contributions(pd.DataFrame(X).iloc[[row_index]])

    if isinstance(contrib, dict):
        contrib = next(iter(contrib.values()))

    names = model.preprocessor_.transformed_feature_names()
    vals = contrib[0]
    order = np.argsort(np.abs(vals))[::-1][:top_k]

    return [
        {"feature": names[i], "contribution": float(vals[i])}
        for i in order
    ]


def _json_safe(obj):
    """Recursively convert numpy scalars/arrays to plain Python types so
    `json.dumps` doesn't choke on them (it accepts `float`/`int`/`list`
    but not `np.float32`/`np.int64`/`np.ndarray`)."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def dashboard_html(model, X, y=None, path: str = "kanboost_report.html") -> str:
    """Write a small, self-contained (static, no server) explainability
    report combining feature importances, a monotonicity audit (if the
    model has `monotone_constraints`), and one row's explanation. Returns
    `path`. See `symbolic_export`/`audit_monotonicity` for the pieces
    used individually.
    """
    from pathlib import Path

    X = pd.DataFrame(X)
    importances = model.feature_importances_dict()
    constraints = getattr(model, "monotone_constraints", {}) or {}

    data = {
        "model": type(model).__name__,
        "features": list(X.columns),
        "top_importances": dict(list(importances.items())[:15]),
        "constraints_audit": audit_monotonicity(model, X, constraints) if constraints else {},
        "first_row_explanation": explain_row(model, X, 0),
    }

    if y is not None and hasattr(model, "evaluate"):
        try:
            data["metrics"] = model.evaluate(X, y, verbose=False)
        except Exception:
            data["metrics"] = "evaluation failed"

    html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>KANBoost Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 32px; line-height: 1.5; }}
pre {{ background: #f6f8fa; padding: 16px; border-radius: 8px; overflow:auto; }}
h1, h2 {{ margin-bottom: 8px; }}
</style>
</head>
<body>
<h1>KANBoost Explainability Report</h1>
<h2>Summary</h2>
<pre>{json.dumps(_json_safe(data), indent=2, ensure_ascii=False)}</pre>
</body>
</html>
"""
    Path(path).write_text(html, encoding="utf-8")
    return path
