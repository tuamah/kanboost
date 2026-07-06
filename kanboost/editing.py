"""
kanboost.editing -- consolidate a fitted GAM-mode ensemble into a single,
directly editable additive model: one B-spline curve per feature instead
of the sum of curves across every boosting round.

Positioned against Microsoft's GAM Changer (https://github.com/interpretml/
gam-changer), a similar editable-GAM tool built for EBM: EBM's shape
functions are piecewise-constant bins, so an edit there has no way to
guarantee monotonicity or smoothness survives it. Here each feature's
curve is a genuine B-spline, and `enforce_monotone` re-derives a
coefficient sequence that is provably monotone (the same
variation-diminishing projection kanboost's own training loop already
uses for `monotone_constraints`), not just a visual nudge -- an edit
followed by `enforce_monotone` gives the same hard guarantee training
does, not a best-effort one.

Requires the source model to have been fit with `gam=True` -- that's
what makes it an *exact* additive ensemble `F(x) = c + sum_j g_j(x_j)`
in the first place, so there's nothing lost by consolidating each g_j
into one spline.
"""

from __future__ import annotations

import numpy as np
import torch

from kan.spline import coef2curve, curve2coef, extend_grid


def _build_base_grid(n_intervals: int) -> torch.Tensor:
    """A single-feature (in_dim=1) uniform grid over [-1, 1] with
    `n_intervals` intervals, shape (1, n_intervals + 1)."""
    return torch.linspace(-1, 1, n_intervals + 1, dtype=torch.float32).unsqueeze(0)


def consolidate(model, resolution: int = 200, grid: int = 10, k: int = 3):
    """Consolidate a fitted `gam=True` ensemble into an `EditableGAM` --
    or, for a multiclass classifier, a dict `{class_label: EditableGAM}`
    (one per one-vs-rest chain, since each chain is its own independent
    additive model).

    `resolution` is how finely each feature's aggregated shape function
    is sampled before being refit as a single spline; `grid`/`k` control
    the resolution/order of that new spline. Higher `resolution` and
    `grid` reduce consolidation error (see `EditableGAM.max_consolidation_error`)
    at the cost of a slightly heavier object.
    """
    if not getattr(model, "gam", False):
        raise ValueError(
            "consolidate() requires a model fitted with gam=True (an exact "
            "additive ensemble F(x) = c + sum_j g_j(x_j)); otherwise the "
            "output layer's own nonlinearity means no single per-feature "
            "spline can reproduce the ensemble."
        )
    model._check_fitted()
    names = model.preprocessor_.transformed_feature_names()
    is_classifier = hasattr(model, "classes_")

    def _consolidate_chain(learners, init_pred, best_iteration):
        gam_model = EditableGAM._from_chain(
            model, learners, init_pred, best_iteration, names, resolution, grid, k,
        )
        gam_model.preprocessor = model.preprocessor_
        gam_model.is_classifier = is_classifier
        gam_model.feature_names_in_ = model.feature_names_in_
        return gam_model

    if isinstance(model.learners_, dict):
        return {
            c: _consolidate_chain(model.learners_[c], model.init_pred_[c], model.best_iteration_[c])
            for c in model.classes_
        }
    return _consolidate_chain(model.learners_, model.init_pred_, model.best_iteration_)


class EditableGAM:
    """One editable additive model: `raw_score(x) = intercept + sum_j
    curve_j(x_j)`, where each `curve_j` is a single B-spline. Build one
    via `consolidate(model)`, not by constructing this class directly.
    """

    def __init__(self, feature_names, x_grid, curves, intercept, spline_grid, spline_k):
        self.feature_names = list(feature_names)
        self.x_grid = np.asarray(x_grid, dtype=np.float64)
        self.curves = {name: np.asarray(c, dtype=np.float64).copy() for name, c in curves.items()}
        self._original_curves = {name: c.copy() for name, c in self.curves.items()}
        self.intercept = float(intercept)
        self.spline_grid = spline_grid
        self.spline_k = spline_k
        self._coef_cache: dict = {}
        self.preprocessor = None
        self.is_classifier = False
        self.feature_names_in_ = None

    @classmethod
    def _from_chain(cls, model, learners, init_pred, best_iteration, names, resolution, grid, k):
        """Sample each feature's curve by holding every other (scaled)
        feature at 0 -- the same partial-dependence probe `plot_feature`
        uses. That probe's raw value at x_j is g_j(x_j) + sum_{i!=j}
        g_i(0), not g_j(x_j) alone: every other feature's contribution
        at its own zero point leaks into every probe. Summing n such
        probes directly would double-count that shared baseline (n-1)
        times over. Centering each probe by its value at x_j=0 (which
        equals sum_i g_i(0) for every feature, since that's exactly the
        all-zero input) removes the leak: `curve_j(x) := probe_j(x) -
        probe_j(0)`, with the removed constant folded into `intercept`.
        This is also the standard GAM identifiability convention
        (g_j(0) = 0, i.e. each shape function is centered).
        """
        device = model.device_
        x_grid_t = torch.linspace(-1, 1, resolution, device=device)
        n_features = len(names)

        def _score(X_batch):
            y = torch.zeros(X_batch.shape[0], device=device)
            for learner in learners[:best_iteration]:
                with torch.no_grad():
                    y = y + model.learning_rate * learner(X_batch).flatten()
            return y

        # Evaluate the shared baseline at the exact all-zero input, rather
        # than reading it off the nearest grid point (which may not be
        # exactly 0 for an even `resolution`).
        baseline = float(_score(torch.zeros((1, n_features), dtype=torch.float32, device=device))[0])

        curves = {}
        for j, name in enumerate(names):
            X_probe = torch.zeros((resolution, n_features), dtype=torch.float32, device=device)
            X_probe[:, j] = x_grid_t
            y = _score(X_probe).cpu().numpy()
            curves[name] = y - baseline  # center: curve_j(0) == 0
        consolidated_intercept = init_pred + baseline
        return cls(names, x_grid_t.cpu().numpy(), curves, consolidated_intercept, grid, k)

    # ------------------------------------------------------------------
    # spline machinery (all CPU/numpy-facing; this object is meant to be
    # cheap to inspect and edit, not to train on a GPU)
    # ------------------------------------------------------------------
    def _coef(self, feature: str) -> torch.Tensor:
        if feature not in self._coef_cache:
            base_grid = _build_base_grid(self.spline_grid)
            ext_grid = extend_grid(base_grid, k_extend=self.spline_k)
            x_eval = torch.tensor(self.x_grid, dtype=torch.float32).unsqueeze(1)  # (resolution, 1)
            y_eval = torch.tensor(self.curves[feature], dtype=torch.float32).view(-1, 1, 1)
            coef = curve2coef(x_eval, y_eval, ext_grid, self.spline_k)
            self._coef_cache[feature] = (ext_grid, coef)
        return self._coef_cache[feature]

    def _invalidate(self, feature: str) -> None:
        self._coef_cache.pop(feature, None)

    def curve_at(self, feature: str, x: np.ndarray) -> np.ndarray:
        """Evaluate one feature's current (possibly edited) curve at
        arbitrary scaled-feature values `x` (same [-1, 1] range the
        underlying ensemble trains on)."""
        ext_grid, coef = self._coef(feature)
        x_t = torch.tensor(np.asarray(x, dtype=np.float64), dtype=torch.float32).view(-1, 1)
        y = coef2curve(x_t, ext_grid, coef, self.spline_k)
        return y.view(-1).numpy()

    # ------------------------------------------------------------------
    # prediction
    # ------------------------------------------------------------------
    def _transform(self, X) -> np.ndarray:
        import pandas as pd

        if self.preprocessor is None:
            raise RuntimeError(
                "This EditableGAM has no preprocessor attached (only happens if "
                "it wasn't built via consolidate()); pass already-scaled arrays "
                "to raw_score() column-by-column instead."
            )
        if not isinstance(X, pd.DataFrame):
            # Use the model's original fit-time column order (not
            # numeric_cols_ + categorical_cols, which would silently
            # misassign columns for any raw layout that interleaves
            # numeric and categorical columns) -- same convention
            # `_base.py`'s `_transform_X` uses for the same reason.
            if self.feature_names_in_ is None:
                raise RuntimeError(
                    "Raw array input requires the original fit-time column "
                    "order, but this EditableGAM has none recorded; pass a "
                    "DataFrame with column names instead."
                )
            X = pd.DataFrame(np.asarray(X), columns=self.feature_names_in_)
        return self.preprocessor.transform(X)

    def raw_score(self, X) -> np.ndarray:
        """Additive raw score `intercept + sum_j curve_j(x_j)` for raw
        (unscaled) input `X`."""
        X_arr = self._transform(X)
        score = np.full(X_arr.shape[0], self.intercept)
        for j, name in enumerate(self.feature_names):
            score += self.curve_at(name, X_arr[:, j])
        return score

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        """Regressor chain: the raw score itself. Classifier chain
        (binary only -- see module docstring for multiclass): 0/1 label
        from `sigmoid(raw_score) >= threshold`."""
        score = self.raw_score(X)
        if not self.is_classifier:
            return score
        return (1.0 / (1.0 + np.exp(-score)) >= threshold).astype(float)

    def predict_proba(self, X) -> np.ndarray:
        """Binary classifier chain only: array of shape (n, 2)."""
        if not self.is_classifier:
            raise AttributeError("predict_proba is only defined for a classifier chain.")
        prob_pos = 1.0 / (1.0 + np.exp(-self.raw_score(X)))
        return np.vstack([1 - prob_pos, prob_pos]).T

    # ------------------------------------------------------------------
    # editing
    # ------------------------------------------------------------------
    def set_offset(self, feature: str, x_range: tuple, delta: float) -> None:
        """Additively shift `feature`'s curve by `delta` (in raw-score
        units) within the scaled-feature interval `x_range = (lo, hi)`
        (both in [-1, 1]), tapering to zero outside it. Invalidates the
        cached spline fit for `feature`, so the next `predict`/`raw_score`
        call refits it from the edited curve."""
        lo, hi = x_range
        mask = (self.x_grid >= lo) & (self.x_grid <= hi)
        self.curves[feature] = self.curves[feature].copy()
        self.curves[feature][mask] += delta
        self._invalidate(feature)

    def set_values(self, feature: str, x_range: tuple, value: float) -> None:
        """Pin `feature`'s curve to a constant `value` within the scaled
        interval `x_range`, e.g. to flatten out a region you believe is
        overfit or unreliable."""
        lo, hi = x_range
        mask = (self.x_grid >= lo) & (self.x_grid <= hi)
        self.curves[feature] = self.curves[feature].copy()
        self.curves[feature][mask] = value
        self._invalidate(feature)

    def enforce_monotone(self, feature: str, sign: int) -> None:
        """Force `feature`'s curve to be non-decreasing (`sign=1`) or
        non-increasing (`sign=-1`), and refit that feature's spline
        coefficients with the same cumulative-max projection kanboost's
        training loop uses for `monotone_constraints` -- so the guarantee
        holds for `curve_at`/`predict` between sample points too, not
        just at the `resolution` grid points used to build the curve.
        """
        if sign not in (1, -1):
            raise ValueError("sign must be 1 (increasing) or -1 (decreasing)")
        curve = self.curves[feature]
        if sign == 1:
            curve = np.maximum.accumulate(curve)
        else:
            # A forward running minimum is itself non-increasing (each new
            # min, over a growing prefix, can only fall or stay put) -- no
            # reversal needed. (Reversing first, as an earlier version of
            # this method did, produces a running *maximum* in reverse,
            # which is non-decreasing -- the wrong direction -- and also
            # leaves a negative-stride view that crashes downstream code.)
            curve = np.minimum.accumulate(curve)
        self.curves[feature] = curve
        self._invalidate(feature)

        # Belt-and-suspenders: also project the refit coefficients onto
        # the monotone cone (variation-diminishing property of B-splines),
        # in case least-squares refitting introduced a tiny overshoot
        # between knots. `self.curves` is the single source of truth
        # everything else (predict, save/load) derives from, so the
        # projected coefficients are immediately re-sampled back into
        # `self.curves` rather than left to live only in the cache --
        # otherwise this correction would silently vanish on the next
        # cache invalidation (e.g. a save/load round-trip).
        ext_grid, coef = self._coef(feature)
        coef_np = coef.detach().numpy()
        if sign == 1:
            coef_np = np.maximum.accumulate(coef_np, axis=-1)
        else:
            coef_np = np.minimum.accumulate(coef_np, axis=-1)
        projected_coef = torch.tensor(coef_np, dtype=torch.float32)
        self._coef_cache[feature] = (ext_grid, projected_coef)

        x_t = torch.tensor(self.x_grid, dtype=torch.float32).view(-1, 1)
        self.curves[feature] = coef2curve(x_t, ext_grid, projected_coef, self.spline_k).view(-1).numpy()
        self._invalidate(feature)  # curves changed again; force a clean refit from them next access

    def reset(self, feature: str | None = None) -> None:
        """Undo edits. With `feature=None`, resets every feature."""
        targets = [feature] if feature is not None else list(self.curves)
        for name in targets:
            self.curves[name] = self._original_curves[name].copy()
            self._invalidate(name)

    # ------------------------------------------------------------------
    # inspection
    # ------------------------------------------------------------------
    def max_consolidation_error(self, feature: str | None = None) -> float:
        """Max absolute difference between the *original* ensemble's
        sampled curve and this object's spline fit to it, before any
        edits -- i.e. how much fidelity consolidation itself cost you.
        Compare against `feature`'s curve range to judge whether it's
        negligible; raise `resolution`/`grid` in `consolidate()` if not."""
        names = [feature] if feature is not None else self.feature_names
        errors = []
        for name in names:
            fitted = self.curve_at(name, self.x_grid)
            errors.append(np.max(np.abs(fitted - self._original_curves[name])))
        return float(max(errors))

    def diff(self, X=None, y=None) -> dict:
        """Summarize edits made so far: per-feature max curve deviation
        from the original consolidated (pre-edit) curves, and -- if `X`
        is given -- the resulting mean absolute change in `raw_score(X)`.
        Pass `y` too (classifier: 0/1 labels; regressor: continuous) to
        additionally report AUC (classifier) or RMSE (regressor) before
        vs. after the edits, evaluated on `X`.
        """
        report = {
            "per_feature_max_delta": {
                name: float(np.max(np.abs(self.curves[name] - self._original_curves[name])))
                for name in self.feature_names
            }
        }
        if X is None:
            return report

        edited_curves = self.curves
        self.curves = {k: v.copy() for k, v in self._original_curves.items()}
        self._coef_cache = {}
        score_before = self.raw_score(X)
        self.curves = edited_curves
        self._coef_cache = {}
        score_after = self.raw_score(X)

        report["mean_abs_score_delta"] = float(np.mean(np.abs(score_after - score_before)))

        if y is not None:
            y = np.asarray(y)
            if self.is_classifier:
                from sklearn.metrics import roc_auc_score

                sigmoid = lambda z: 1.0 / (1.0 + np.exp(-z))
                report["metric"] = "auc"
                report["metric_before"] = float(roc_auc_score(y, sigmoid(score_before)))
                report["metric_after"] = float(roc_auc_score(y, sigmoid(score_after)))
            else:
                from sklearn.metrics import mean_squared_error

                report["metric"] = "rmse"
                report["metric_before"] = float(np.sqrt(mean_squared_error(y, score_before)))
                report["metric_after"] = float(np.sqrt(mean_squared_error(y, score_after)))
        return report

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save({
            "feature_names": self.feature_names,
            "x_grid": self.x_grid,
            "curves": self.curves,
            "original_curves": self._original_curves,
            "intercept": self.intercept,
            "spline_grid": self.spline_grid,
            "spline_k": self.spline_k,
            "is_classifier": self.is_classifier,
            "preprocessor": self.preprocessor,
            "feature_names_in_": self.feature_names_in_,
        }, path)

    @classmethod
    def load(cls, path: str) -> "EditableGAM":
        payload = torch.load(path, weights_only=False)
        obj = cls(
            payload["feature_names"], payload["x_grid"], payload["curves"],
            payload["intercept"], payload["spline_grid"], payload["spline_k"],
        )
        obj._original_curves = payload["original_curves"]
        obj.is_classifier = payload["is_classifier"]
        obj.preprocessor = payload["preprocessor"]
        obj.feature_names_in_ = payload.get("feature_names_in_")
        return obj
