"""
kanboost.symbolic -- fidelity-aware symbolic formula export for a fitted
`gam=True` KANBoost model: an actual `sympy` expression, LaTeX string,
and standalone (no torch/pykan dependency) numpy predict function --
not just a text summary (see `kanboost.experimental.symbolic_export`
for that lighter-weight report).

Fits one closed-form candidate function per feature to the *exact*
aggregated shape function `g_j`, the same one `symbolic_report`/
`plot_feature` use -- but where they only fit and report a name/R^2,
this reconstructs the actual fitted expression (`c * fun(a*x + b) + d`,
pykan's own `SYMBOLIC_LIB` convention) and combines every feature's term
into one total formula `F(x) = intercept + sum_j term_j(x_j)`.

Reuses `kanboost.editing.consolidate()` for the per-feature curves and
intercept rather than re-deriving curve sampling from scratch: an
earlier version of `consolidate()` had a real bug where naively summing
per-feature partial-dependence probes double-counted every other
feature's zero-point contribution (see `editing.py`'s docstring) --
fitting symbolic terms on top of that same, now-corrected, centered
representation avoids reintroducing that class of bug here.

Fidelity-aware: a feature whose best candidate's R^2 falls below
`min_r2` is *not* forced into a misleading formula -- it's kept as a
numeric (spline-interpolated) term instead, clearly flagged as such in
`fidelity_report()`/`to_latex()`, so the exported formula never silently
claims a closed form that doesn't actually fit.
"""

from __future__ import annotations

import numpy as np
import sympy
import torch

from .editing import consolidate


_CANDIDATES = ["x", "x^2", "x^3", "sin", "cos", "exp", "log", "sqrt", "tanh", "abs"]


def export_symbolic(model, min_r2: float = 0.8, resolution: int = 200, grid: int = 10, k: int = 3, features=None):
    """Fit one closed-form symbolic term per feature (falling back to a
    numeric spline term where no candidate reaches `min_r2`) and combine
    them into a `SymbolicModel`: `intercept + sum_j term_j(x_j)`.

    For a multiclass classifier, returns `{class_label: SymbolicModel}`
    (one independent formula per one-vs-rest chain), matching
    `consolidate()`'s own convention.

    `X` is not needed -- like `consolidate()`, this samples each
    feature's shape function directly from the trained ensemble on
    [-1, 1] (the model's internal scaled range), not from data.

    `features`, if given, restricts the (relatively expensive)
    candidate-fitting search to just those feature names -- every other
    feature is kept as a numeric term directly, with no candidates
    tried. Useful when only a handful of features' formulas are
    actually needed (see `explain()`, which uses this to avoid fitting
    candidates for features outside `top_features`).
    """
    from kan.utils import fit_params, SYMBOLIC_LIB

    candidates = [c for c in _CANDIDATES if c in SYMBOLIC_LIB]
    feature_set = set(features) if features is not None else None

    consolidated = consolidate(model, resolution=resolution, grid=grid, k=k)
    if isinstance(consolidated, dict):
        return {c: SymbolicModel._from_editable(gam, candidates, min_r2, feature_set) for c, gam in consolidated.items()}
    return SymbolicModel._from_editable(consolidated, candidates, min_r2, feature_set)


def explain(model, top_features: int = 5, symbolic: bool = True, simplify: bool = True, min_r2: float = 0.8) -> list:
    """High-level convenience report: rank features by
    `model.feature_importances_dict()` (already handles multiclass by
    summing importances across every one-vs-rest chain into one
    ranking, same as that method's own docstring), and for the top
    `top_features`, attach each one's symbolic term if `symbolic=True`.

    Returns a list of dicts, most important feature first:
    `{"feature", "importance", "kind", "r2", "amplitude", "formula"}`
    (`"formula"` is a `sympy` expression, or `None` if `symbolic=False`).
    `simplify=True` runs `sympy.simplify()` on each formula (cheap here
    -- these are single-feature terms, not the whole model).

    For a multiclass classifier, `symbolic=True` uses each top
    feature's term from its *first* class's chain (`model.classes_[0]`)
    -- one-vs-rest chains can fit a feature differently per class, so
    this is a representative formula, not a claim that it's identical
    across classes. Call `export_symbolic(model)` directly and index by
    class for a per-class formula.
    """
    importances = model.feature_importances_dict()
    top = list(importances.items())[:top_features]

    # Only fit candidates for the top features actually being reported --
    # export_symbolic's `features=` skips the expensive search for
    # everything else (still exact/spline-numeric there, just unused).
    top_names = [name for name, _ in top]
    sym = export_symbolic(model, min_r2=min_r2, features=top_names) if symbolic else None
    if isinstance(sym, dict):
        sym = sym[model.classes_[0]]

    report = []
    for name, importance in top:
        entry = {"feature": name, "importance": importance}
        if symbolic:
            term = sym.terms[name]
            entry["kind"] = term["kind"]
            entry["r2"] = term["r2"]
            entry["amplitude"] = term["amplitude"]
            entry["formula"] = sym.term_sympy(name, simplify=simplify)
        else:
            entry.update({"kind": None, "r2": None, "amplitude": None, "formula": None})
        report.append(entry)
    return report


class SymbolicModel:
    """A fitted, fidelity-aware symbolic export. Build one via
    `export_symbolic(model)`, not directly."""

    def __init__(self, feature_names, intercept, terms, preprocessor=None, feature_names_in_=None):
        self.feature_names = list(feature_names)
        self.intercept = float(intercept)
        self.terms = terms  # {name: {"kind", "r2", "sympy_expr" or None, "x_grid", "y_grid"}}
        self.preprocessor = preprocessor
        self.feature_names_in_ = feature_names_in_
        self._symbols = _make_unique_symbols(self.feature_names)

    @classmethod
    def _from_editable(cls, gam, candidates, min_r2, feature_set=None):
        from kan.utils import fit_params, SYMBOLIC_LIB

        terms = {}
        x_t = torch.tensor(gam.x_grid, dtype=torch.float32)
        for name in gam.feature_names:
            amplitude = float(gam.curves[name].max() - gam.curves[name].min())

            if feature_set is not None and name not in feature_set:
                # Skip the (relatively expensive) candidate search for
                # features nobody asked for; keep the exact spline curve
                # as a numeric term instead.
                terms[name] = {
                    "kind": "numeric", "r2": float("nan"), "candidate": None,
                    "x_grid": gam.x_grid.copy(), "y_grid": gam.curves[name].copy(),
                    "amplitude": amplitude,
                }
                continue

            y_t = torch.tensor(gam.curves[name], dtype=torch.float32)
            best = None
            for cand in candidates:
                fun = SYMBOLIC_LIB[cand][0]
                try:
                    params, r2 = fit_params(x_t, y_t, fun, verbose=False)
                except Exception:
                    continue
                r2 = float(r2)
                if best is None or r2 > best[1]:
                    best = (cand, r2, [float(p) for p in params])

            if best is not None and best[1] >= min_r2:
                cand, r2, (a, b, c, d) = best
                terms[name] = {
                    "kind": "symbolic", "r2": r2, "candidate": cand,
                    "params": {"a": a, "b": b, "c": c, "d": d}, "amplitude": amplitude,
                }
            else:
                terms[name] = {
                    "kind": "numeric",
                    "r2": best[1] if best is not None else float("nan"),
                    "candidate": best[0] if best is not None else None,
                    "x_grid": gam.x_grid.copy(), "y_grid": gam.curves[name].copy(),
                    "amplitude": amplitude,
                }

        return cls(gam.feature_names, gam.intercept, terms,
                    preprocessor=gam.preprocessor, feature_names_in_=gam.feature_names_in_)

    # ------------------------------------------------------------------
    def to_sympy(self):
        """The full model as one `sympy` expression, `intercept + sum_j
        term_j(x_j)`. Numeric (non-symbolic) terms appear as an opaque
        function symbol `g_<feature>(x)` -- there's no closed form for
        those; see `fidelity_report()` for which features that affects."""
        from kan.utils import SYMBOLIC_LIB

        expr = sympy.Float(self.intercept)
        for name, term in self.terms.items():
            x = self._symbols[name]
            if term["kind"] == "symbolic":
                sympy_fun = SYMBOLIC_LIB[term["candidate"]][1]
                p = term["params"]
                expr += p["c"] * sympy_fun(p["a"] * x + p["b"]) + p["d"]
            else:
                expr += sympy.Function(f"g_{_safe_symbol_name(name)}")(x)
        return expr

    def to_latex(self) -> str:
        return sympy.latex(self.to_sympy())

    def term_sympy(self, feature: str, simplify: bool = False):
        """Just one feature's term as a `sympy` expression (not the
        whole model) -- `c * fun(a*x + b) + d`, or an opaque
        `g_<feature>(x)` symbol if that feature fell back to a numeric
        term. `simplify=True` runs `sympy.simplify()` on it (can be slow
        on complex expressions; each single term here is cheap)."""
        from kan.utils import SYMBOLIC_LIB

        term = self.terms[feature]
        x = self._symbols[feature]
        if term["kind"] == "symbolic":
            sympy_fun = SYMBOLIC_LIB[term["candidate"]][1]
            p = term["params"]
            expr = p["c"] * sympy_fun(p["a"] * x + p["b"]) + p["d"]
        else:
            expr = sympy.Function(f"g_{_safe_symbol_name(feature)}")(x)
        return sympy.simplify(expr) if simplify else expr

    def fidelity_report(self) -> dict:
        """`{feature: {"kind": "symbolic"|"numeric", "r2": float,
        "candidate": name_or_None, "amplitude": float}}` -- what
        fraction of the model is a true closed form vs. a numeric
        fallback, and how well each feature's chosen candidate actually
        fit.

        `amplitude` (the term's max-min range on [-1, 1]) matters
        alongside `r2`: a high R^2 does not by itself mean a feature is
        important, only that its curve's *shape* -- however small --
        was well matched by some candidate. A near-flat curve can score
        a deceptively high R^2 by fitting its own tiny wiggles; check
        `amplitude` against the other features' to judge whether a term
        actually contributes much to the prediction.
        """
        return {
            name: {"kind": t["kind"], "r2": t["r2"], "candidate": t["candidate"], "amplitude": t["amplitude"]}
            for name, t in self.terms.items()
        }

    def symbolic_fraction(self) -> float:
        """Fraction of features (by count, not by importance) that got
        a genuine closed-form term rather than a numeric fallback."""
        if not self.terms:
            return 0.0
        return sum(t["kind"] == "symbolic" for t in self.terms.values()) / len(self.terms)

    # ------------------------------------------------------------------
    def _term_value_scaled(self, name: str, x: np.ndarray) -> np.ndarray:
        # Evaluated via plain numpy (not torch/sympy) so predict_scaled()
        # never needs torch/pykan at call time, even for symbolic terms.
        term = self.terms[name]
        if term["kind"] == "symbolic":
            p = term["params"]
            fn = _NUMPY_FUN[term["candidate"]]
            return p["c"] * fn(p["a"] * np.asarray(x, dtype=np.float64) + p["b"]) + p["d"]
        return np.interp(np.asarray(x, dtype=np.float64), term["x_grid"], term["y_grid"])

    def predict_scaled(self, X_scaled) -> np.ndarray:
        """Predict from already-scaled feature values (the model's
        internal [-1, 1] range, e.g. `model.preprocessor_.transform(X)`)
        -- pure numpy, no torch/pykan/sympy needed at call time. Column
        order must match `self.feature_names`.

        Note: for a `"log"` or `"sqrt"` term, this and `to_sympy()`'s
        formula only agree where the fitted argument `a*x + b` is
        non-negative (the domain candidates were fit on, [-1, 1]) --
        outside that range this uses `log(|x|+eps)`/`sqrt(|x|)` to stay
        finite, while the symbolic formula would give NaN/complex. Only
        reachable with inputs outside the training range; within-range
        predictions are unaffected.
        """
        X_scaled = np.asarray(X_scaled, dtype=np.float64)
        score = np.full(X_scaled.shape[0], self.intercept)
        for j, name in enumerate(self.feature_names):
            score += self._term_value_scaled(name, X_scaled[:, j])
        return score

    def predict(self, X) -> np.ndarray:
        """Predict from raw (unscaled) input -- requires the
        preprocessor this `SymbolicModel` was exported with (i.e. built
        via `export_symbolic(model)`, not reconstructed by hand)."""
        import pandas as pd

        if self.preprocessor is None:
            raise RuntimeError(
                "This SymbolicModel has no preprocessor attached; use "
                "predict_scaled() with already-scaled arrays instead."
            )
        if not isinstance(X, pd.DataFrame):
            if self.feature_names_in_ is None:
                raise RuntimeError(
                    "Raw array input requires the original fit-time column "
                    "order; pass a DataFrame with column names instead."
                )
            X = pd.DataFrame(np.asarray(X), columns=self.feature_names_in_)
        X_scaled = self.preprocessor.transform(X)
        return self.predict_scaled(X_scaled)

    def save(self, path: str) -> None:
        torch.save({
            "feature_names": self.feature_names,
            "intercept": self.intercept,
            "terms": self.terms,
            "preprocessor": self.preprocessor,
            "feature_names_in_": self.feature_names_in_,
        }, path)

    @classmethod
    def load(cls, path: str) -> "SymbolicModel":
        payload = torch.load(path, weights_only=False)
        return cls(
            payload["feature_names"], payload["intercept"], payload["terms"],
            preprocessor=payload["preprocessor"], feature_names_in_=payload["feature_names_in_"],
        )


def _safe_symbol_name(name: str) -> str:
    """sympy Symbol names can't contain most punctuation; make feature
    names like "mean radius" or "a-b" round-trippable."""
    return "".join(ch if ch.isalnum() else "_" for ch in str(name))


def _make_unique_symbols(feature_names) -> dict:
    """`{original_name: sympy.Symbol}`, guarding against two distinct
    feature names sanitizing to the same symbol name (e.g. "a b" and
    "a-b" both become "a_b") -- sympy interns Symbols by name, so an
    unguarded collision would silently conflate two different features
    into one symbol throughout to_sympy()/term_sympy(). Colliding names
    beyond the first get a numeric suffix."""
    symbols = {}
    seen = {}
    for name in feature_names:
        base = _safe_symbol_name(name)
        if base in seen:
            seen[base] += 1
            base = f"{base}_{seen[base]}"
        else:
            seen[base] = 0
        symbols[name] = sympy.Symbol(base)
    return symbols


_NUMPY_FUN = {
    "x": lambda x: x,
    "x^2": lambda x: x ** 2,
    "x^3": lambda x: x ** 3,
    "sin": np.sin,
    "cos": np.cos,
    "exp": np.exp,
    "log": lambda x: np.log(np.abs(x) + 1e-8),
    "sqrt": lambda x: np.sqrt(np.abs(x)),
    "tanh": np.tanh,
    "abs": np.abs,
}
