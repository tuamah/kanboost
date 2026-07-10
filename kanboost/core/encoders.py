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
from sklearn.preprocessing import RobustScaler


class TabularPreprocessor:
    """
    Automatic preprocessing for mixed numeric/categorical tabular data.

    - Numeric columns: median-imputed (missing values get an extra
      "<col>_missing" indicator column, added only for columns that had
      at least one missing value during fit), outlier-clipped (1st/99th
      percentile), RobustScaler'd, then clipped to [-1, 1] (KAN's default
      spline grid range).
    - Categorical columns: target-mean encoding (smoothed), computed on
      the training fold only, to avoid leakage -- similar in spirit to
      CatBoost's ordered target statistics (simplified, non-ordered
      version). Missing/unseen categories fall back to the global mean.
    """

    def __init__(
        self,
        categorical_cols=None,
        smoothing: float = 10.0,
        add_missing_indicator: bool = True,
    ):
        self.categorical_cols = categorical_cols or []
        self.smoothing = smoothing
        self.add_missing_indicator = add_missing_indicator
        self.numeric_cols_ = None
        self.scaler_ = None
        self.clip_low_ = None
        self.clip_high_ = None
        self.numeric_medians_ = None
        self.missing_indicator_cols_ = []
        self.cat_maps_ = {}
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
            stats = (
                pd.DataFrame({col: X[col].astype(str), "y": y})
                .groupby(col)["y"]
                .agg(["mean", "count"])
            )
            smoothed = (
                stats["mean"] * stats["count"] + self.global_mean_ * self.smoothing
            ) / (stats["count"] + self.smoothing)
            self.cat_maps_[col] = smoothed.to_dict()

        return self

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
            encoded = X[col].astype(str).map(mapping).fillna(self.global_mean_)
            parts.append(encoded.to_numpy(dtype=float).reshape(-1, 1))

        return np.hstack(parts) if parts else np.empty((len(X), 0))

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)

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
