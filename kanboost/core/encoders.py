"""
Preprocessing utilities for KANBoost.

KAN is sensitive to input scale (splines are defined on a bounded grid),
so numeric features must be scaled/clipped, and categorical features
must be encoded to numbers before reaching the model. This module
automates both steps so the end user can pass a raw-ish DataFrame,
similar to how CatBoost handles categorical columns internally.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler


class TabularPreprocessor:
    """
    Automatic preprocessing for mixed numeric/categorical tabular data.

    - Numeric columns: median-imputed (missing values get an extra
      "<col>_missing" indicator column, added only for columns that had
      at least one missing value during fit), outlier-clipped (1st/99th
      percentile), RobustScaler'd, then clipped to [-1, 1] (KAN's default
      spline grid range).
    - Categorical columns: target-mean encoding (smoothed toward the
      global mean, or toward a parent category's smoothed mean if
      `hierarchy` names one -- see below). `fit()`/`transform()` compute
      and apply this mapping on disjoint data (e.g. fit on train,
      transform held-out val/test), which is leakage-free.
      `fit_transform()` additionally encodes each *fitting* row
      out-of-fold (K-fold, similar in spirit to CatBoost's ordered
      target statistics), since encoding a row with a mapping that
      includes its own label would otherwise leak target information
      into the very rows being trained on.
    - Optional `hierarchy`: `{fine_col: coarse_col}` -- backs a sparse
      fine-grained category (e.g. `city`) off to its parent's smoothed
      mean (e.g. `region`) instead of the flat global mean, which is a
      better prior when many fine categories have few samples. Only
      applies to columns named as keys; any categorical column not in
      `hierarchy` keeps the original flat-global-mean backoff.
    """

    def __init__(
        self,
        categorical_cols=None,
        smoothing: float = 10.0,
        add_missing_indicator: bool = True,
        cv_folds: int = 5,
        random_state: int = 42,
        hierarchy: dict = None,
    ):
        self.categorical_cols = categorical_cols or []
        self.smoothing = smoothing
        self.add_missing_indicator = add_missing_indicator
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.hierarchy = hierarchy or {}
        self.numeric_cols_ = None
        self.scaler_ = None
        self.clip_low_ = None
        self.clip_high_ = None
        self.numeric_medians_ = None
        self.missing_indicator_cols_ = []
        self.cat_maps_ = {}
        self.coarse_maps_ = {}
        self.global_mean_ = None

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.numeric_cols_ = [c for c in X.columns if c not in self.categorical_cols]
        self.global_mean_ = float(np.mean(y))

        # --- numeric ---
        if self.numeric_cols_:
            X_num = X[self.numeric_cols_].to_numpy(dtype=float)

            if self.add_missing_indicator:
                self.missing_indicator_cols_ = [
                    col for col, has_na in
                    zip(self.numeric_cols_, np.isnan(X_num).any(axis=0))
                    if has_na
                ]

            self.numeric_medians_ = np.nanmedian(X_num, axis=0)
            # a column that is entirely NaN has an undefined median; fall back to 0
            self.numeric_medians_ = np.nan_to_num(self.numeric_medians_, nan=0.0)
            X_num = np.where(np.isnan(X_num), self.numeric_medians_, X_num)

            self.clip_low_ = np.percentile(X_num, 1, axis=0)
            self.clip_high_ = np.percentile(X_num, 99, axis=0)
            X_num = np.clip(X_num, self.clip_low_, self.clip_high_)
            self.scaler_ = RobustScaler()
            self.scaler_.fit(X_num)

        # --- categorical: smoothed target-mean encoding ---
        for col in self.categorical_cols:
            if col in self.hierarchy:
                coarse_col = self.hierarchy[col]
                coarse_values = X[coarse_col].astype(str).values
                coarse_map = self._smoothed_map(
                    coarse_values, y, np.full(len(y), self.global_mean_)
                )
                self.coarse_maps_[col] = (coarse_col, coarse_map)
                fine_prior = np.array(
                    [coarse_map.get(v, self.global_mean_) for v in coarse_values]
                )
            else:
                fine_prior = np.full(len(y), self.global_mean_)
            self.cat_maps_[col] = self._smoothed_map(
                X[col].astype(str).values, y, fine_prior
            )

        return self

    def _smoothed_map(self, col_values, y, prior_by_row) -> dict:
        """Smooth each category's target-mean toward `prior_by_row`
        (per-row backoff target -- the flat global mean, or a parent
        category's smoothed mean when `hierarchy` applies)."""
        stats = (
            pd.DataFrame({"col": col_values, "y": y, "prior": prior_by_row})
            .groupby("col")
            .agg(mean=("y", "mean"), count=("y", "count"), prior=("prior", "first"))
        )
        smoothed = (
            stats["mean"] * stats["count"] + stats["prior"] * self.smoothing
        ) / (stats["count"] + self.smoothing)
        return smoothed.to_dict()

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        parts = []
        if self.numeric_cols_:
            X_num = X[self.numeric_cols_].to_numpy(dtype=float)
            nan_mask = np.isnan(X_num)
            X_num = np.where(nan_mask, self.numeric_medians_, X_num)
            X_num = np.clip(X_num, self.clip_low_, self.clip_high_)
            X_num = self.scaler_.transform(X_num)
            X_num = np.clip(X_num, -1, 1)
            parts.append(X_num)

            if self.missing_indicator_cols_:
                idx = [self.numeric_cols_.index(c) for c in self.missing_indicator_cols_]
                parts.append(nan_mask[:, idx].astype(float))

        for col in self.categorical_cols:
            mapping = self.cat_maps_[col]
            encoded = X[col].astype(str).map(mapping)
            if col in self.hierarchy:
                coarse_col, coarse_map = self.coarse_maps_[col]
                fallback = X[coarse_col].astype(str).map(coarse_map).fillna(self.global_mean_)
                encoded = encoded.fillna(fallback)
            else:
                encoded = encoded.fillna(self.global_mean_)
            parts.append(encoded.to_numpy(dtype=float).reshape(-1, 1))

        return np.hstack(parts) if parts else np.empty((len(X), 0))

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        X_arr = self.transform(X)

        if self.categorical_cols:
            n = len(X)
            n_splits = max(2, min(self.cv_folds, n))
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            cat_start = len(self.numeric_cols_ or []) + len(self.missing_indicator_cols_)
            for j, col in enumerate(self.categorical_cols):
                col_values = X[col].astype(str).values
                is_hier = col in self.hierarchy
                if is_hier:
                    coarse_values = X[self.hierarchy[col]].astype(str).values

                oof = np.empty(n, dtype=float)
                for tr_idx, va_idx in cv.split(X):
                    if is_hier:
                        coarse_map_fold = self._smoothed_map(
                            coarse_values[tr_idx], y[tr_idx],
                            np.full(len(tr_idx), self.global_mean_),
                        )
                        fine_prior_tr = np.array(
                            [coarse_map_fold.get(v, self.global_mean_) for v in coarse_values[tr_idx]]
                        )
                        mapping = self._smoothed_map(col_values[tr_idx], y[tr_idx], fine_prior_tr)
                        fallback_va = np.array(
                            [coarse_map_fold.get(v, self.global_mean_) for v in coarse_values[va_idx]]
                        )
                        oof[va_idx] = [
                            mapping.get(v, fallback_va[k])
                            for k, v in enumerate(col_values[va_idx])
                        ]
                    else:
                        mapping = self._smoothed_map(
                            col_values[tr_idx], y[tr_idx], np.full(len(tr_idx), self.global_mean_)
                        )
                        oof[va_idx] = np.array(
                            [mapping.get(v, self.global_mean_) for v in col_values[va_idx]]
                        )
                X_arr[:, cat_start + j] = oof

        return X_arr

    @property
    def output_width(self) -> int:
        """Number of columns produced by transform()."""
        return (
            len(self.numeric_cols_ or [])
            + len(self.missing_indicator_cols_)
            + len(self.categorical_cols)
        )

    def transformed_feature_names(self) -> list:
        """Column names in the exact order transform() emits them."""
        return (
            list(self.numeric_cols_ or [])
            + [f"{c}_missing" for c in self.missing_indicator_cols_]
            + list(self.categorical_cols)
        )
