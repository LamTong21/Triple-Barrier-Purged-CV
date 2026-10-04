"""
src/data_pipeline/audit.py

Validates physical price geometry and constructs a causal Synthetic Total Return Index
to prevent corporate action and dividend look-ahead leakage.
"""

from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd


class Layer1DataIntegrity:
    """
    Audits candlestick geometric bounds, removes duplicate timestamps,
    computes causal price adjustments, and eliminates zero-variance columns.
    """

    @staticmethod
    def run_audit(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Executes physical data cleaning and constructs forward-chained total returns.

        Parameters
        ----------
        df : pd.DataFrame
            Raw OHLCV DataFrame with DatetimeIndex.

        Returns
        -------
        Tuple[pd.DataFrame, Dict[str, Any]]
            Cleaned DataFrame and an audit report dictionary.
        """
        df = df.copy()
        initial_len = len(df)

        # 1. Deduplicate index and enforce strictly monotonic timeline
        if df.index.duplicated().any():
            df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()

        # 2. Geometric envelope audit: high >= max(open, close), low <= min(open, close)
        max_oc = df[["open", "close"]].max(axis=1)
        min_oc = df[["open", "close"]].min(axis=1)
        df["high"] = np.maximum(df["high"], max_oc)
        df["low"] = np.minimum(df["low"], min_oc)
        df = df[(df["low"] > 0) & (df["volume"] >= 0)]

        # 3. Fill default corporate action factors if missing
        if "split_factor" not in df.columns:
            df["split_factor"] = 1.0
        if "dividend" not in df.columns:
            df["dividend"] = 0.0

        # 4. Leakage-free Synthetic Total Return Index
        # Calculated causally strictly relying on information at bar t and t-1
        daily_ret = ((df["close"] + df["dividend"]) / (df["close"].shift(1) * df["split_factor"])) - 1.0
        df["close_adj"] = 1000.0 * (1.0 + daily_ret.fillna(0.0)).cumprod()

        ratio = df["close_adj"] / df["close"]
        for col in ["open", "high", "low"]:
            df[f"{col}_adj"] = df[col] * ratio

        # 5. Drop flat zero-variance features
        dropped_zero_var = []
        for col in df.columns:
            if col not in ["split_factor", "dividend", "target_label"]:
                if pd.api.types.is_numeric_dtype(df[col]):
                    if df[col].std(ddof=0) < 1e-8 or df[col].nunique() <= 1:
                        dropped_zero_var.append(col)

        if dropped_zero_var:
            df.drop(columns=dropped_zero_var, inplace=True)

        audit_report = {
            "initial_bars": initial_len,
            "clean_bars": len(df),
            "dropped_flat_cols": dropped_zero_var,
        }

        return df, audit_report