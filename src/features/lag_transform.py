"""
src/features/lag_transform.py

Generates volatility-scaled causal lags and first-order momentum features.
Second-order derivatives (acceleration) are purged.
"""

from typing import Dict
import pandas as pd


class Layer14LagTransformEngine:
    """
    Applies statistically validated causal lags and first-order momentum.
    Lags are clamped to [1, 2] to preserve signal freshness.
    """

    @staticmethod
    def apply_volatility_scaled_lags(X: pd.DataFrame, optimal_lags: Dict[str, int], eps: float = 1e-8) -> pd.DataFrame:
        X_transformed = pd.DataFrame(index=X.index)

        for col, lag in optimal_lags.items():
            if col not in X.columns:
                continue

            safe_lag = min(max(1, int(lag)), 2)
            feature_series = X[col]

            # 1. Causal Lagged Z-Score
            feature_lagged = feature_series.shift(safe_lag)
            roll_mean = feature_lagged.rolling(20).mean()
            roll_std = feature_lagged.rolling(20).std(ddof=1)
            z_scaled = (feature_lagged - roll_mean) / (roll_std + eps)
            X_transformed[f"{col}_zscaled_lag{safe_lag}"] = z_scaled

            # 2. First-order momentum only (Acceleration is omitted)
            momentum = (feature_series - feature_lagged) / (feature_lagged.abs() + eps)
            X_transformed[f"{col}_momentum"] = momentum

        return X_transformed