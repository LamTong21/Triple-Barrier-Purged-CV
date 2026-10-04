"""
src/diagnostics/fractional.py

Fractional Differentiation for memory preservation and strict stationarity.
Implements Marcos López de Prado's expanding window fractional differentiation.
"""

import warnings
from typing import Dict, Any
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning


class Layer5FractionalIntegration:
    """
    Determines minimum fractional differentiation order d* that achieves stationarity
    (ADF p < 0.05) while preserving maximum long-memory correlation with original prices.
    """

    @staticmethod
    def get_weights(d: float, size: int, threshold: float = 1e-4) -> np.ndarray:
        """Computes binomial expansion weights for fractional differencing."""
        w = [1.0]
        for k in range(1, size):
            w_k = -w[-1] / k * (d - k + 1)
            if abs(w_k) < threshold:
                break
            w.append(w_k)
        return np.array(w[::-1])

    @classmethod
    def fractionally_diff(cls, series: pd.Series, d: float, threshold: float = 1e-4) -> pd.Series:
        """Applies fractional differencing using finite dot product convolutions."""
        weights = cls.get_weights(d, len(series), threshold)
        width = len(weights)
        vals = series.values
        res = [np.dot(weights, vals[i - width : i]) for i in range(width, len(vals))]
        return pd.Series(res, index=series.index[width:], name=f"frac_diff_{d:.2f}")

    @classmethod
    def find_optimal_d(cls, series: pd.Series, d_step: float = 0.05) -> Dict[str, Any]:
        """
        Scans grid d in [0.0, 1.0] to find the minimum d* passing stationarity tests.
        """
        s = series.dropna()
        if len(s) < 50:
            return {"optimal_d": 1.0, "price_stationary": False, "memory_retention_corr": 0.0}

        adf_raw_p = adfuller(s, autolag="AIC")[1]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InterpolationWarning)
            try:
                _, kpss_raw_p, _, _ = kpss(s, regression="c", nlags="auto")
            except Exception:
                kpss_raw_p = 0.0

        best_d = 1.0
        best_corr = 0.0

        for d in np.arange(0.0, 1.05, d_step):
            fd = cls.fractionally_diff(s, d)
            if len(fd) < 50:
                continue
            adf_p = adfuller(fd.dropna(), autolag="AIC")[1]
            if adf_p < 0.05:
                best_d = float(round(d, 2))
                aligned = pd.concat([s, fd], axis=1).dropna()
                best_corr = float(np.corrcoef(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1])
                break

        return {
            "price_stationary": bool(adf_raw_p < 0.05 and kpss_raw_p > 0.05),
            "optimal_d": best_d,
            "memory_retention_corr": best_corr,
        }