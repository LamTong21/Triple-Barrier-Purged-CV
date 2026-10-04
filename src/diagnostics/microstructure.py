"""
src/diagnostics/microstructure.py

Limit Order Book (L2) microstructure diagnostics:
- Evaluates micro-price deviation predictive correlation against forward returns.
- Validates bid-ask spread stationarity using the Augmented Dickey-Fuller test.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.tsa.stattools import adfuller


class Layer7cMicrostructureDiagnostics:
    """
    Assesses predictive causality and stability of order book depth feeds.
    """

    @staticmethod
    def run(df: pd.DataFrame) -> Dict[str, Any]:
        """Tests microstructure features against price dynamics."""
        r = df["log_return"].fillna(0.0)

        if "micro_mid_deviation" not in df.columns or "l2_spread" not in df.columns:
            return {
                "has_l2_orderbook": False,
                "micro_deviation_sig": False,
                "micro_dev_predictive_power": 0.0,
                "l2_spread_stationary": False,
            }

        micro_dev = df["micro_mid_deviation"].fillna(0.0)
        l2_spread = df["l2_spread"].fillna(0.0)

        dev_corr, dev_pval = spearmanr(micro_dev.shift(1).fillna(0.0), r)
        is_dev_sig = bool(dev_pval < 0.05 and abs(dev_corr) > 0.05)

        try:
            spread_adf_p = adfuller(l2_spread, autolag="AIC")[1]
            is_spread_stat = bool(spread_adf_p < 0.05)
        except Exception:
            is_spread_stat = False

        return {
            "has_l2_orderbook": True,
            "micro_deviation_sig": is_dev_sig,
            "micro_dev_predictive_power": float(dev_corr),
            "l2_spread_stationary": is_spread_stat,
        }