"""
src/diagnostics/memory.py

Long-range and short-range serial dependence diagnostics:
- Detrended Fluctuation Analysis (DFA) for Hurst Exponent estimation.
- Lo-MacKinlay Heteroskedasticity-Consistent Variance Ratio Test.
- Multi-scale memory profiling via Autocorrelation (ACF).
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf


class Layer4MemoryDependence:
    """
    Measures persistence, mean reversion, and serial autocorrelation across time horizons.
    """

    @staticmethod
    def compute_hurst_dfa(series: pd.Series) -> float:
        """
        Estimates the Hurst exponent using linear Detrended Fluctuation Analysis (DFA).

        Parameters
        ----------
        series : pd.Series
            Input time series (e.g., log returns).

        Returns
        -------
        float
            Estimated Hurst exponent H.
        """
        s = series.dropna().values
        y = np.cumsum(s - np.mean(s))
        n = len(y)
        if n < 40:
            return 0.5

        scales = np.floor(np.logspace(np.log10(10), np.log10(n // 4), num=15)).astype(int)
        scales = np.unique(scales)
        fluctuations = []

        for s_len in scales:
            num_segments = n // s_len
            segment_fluct = []
            for i in range(num_segments):
                seg = y[i * s_len : (i + 1) * s_len]
                x = np.arange(s_len)
                poly = np.polyfit(x, seg, 1)
                trend = np.polyval(poly, x)
                segment_fluct.append(np.sqrt(np.mean((seg - trend) ** 2)))
            fluctuations.append(np.mean(segment_fluct))

        poly_hurst = np.polyfit(np.log(scales), np.log(fluctuations), 1)
        return float(poly_hurst[0])

    @staticmethod
    def lo_mackinlay_vr(prices: pd.Series, k: int) -> Dict[str, float]:
        """
        Lo-MacKinlay Heteroskedasticity-Consistent Variance Ratio Test.

        Parameters
        ----------
        prices : pd.Series
            Price series.
        k : int
            Holding horizon (lag factor).

        Returns
        -------
        Dict[str, float]
            Variance ratio statistic, z-score, and p-value.
        """
        p = np.log(prices.dropna().values)
        t = len(p)
        if t <= k + 2:
            return {"vr": 1.0, "z_stat": 0.0, "p_value": 1.0}

        r1 = p[1:] - p[:-1]
        mu = (p[-1] - p[0]) / (t - 1)
        var_1 = np.sum((r1 - mu) ** 2) / (t - 2)

        rk = p[k:] - p[:-k]
        m = k * (t - k) * (1.0 - (k / (t - 1.0)))
        var_k = np.sum((rk - k * mu) ** 2) / m
        vr = var_k / var_1

        delta = np.zeros(k - 1)
        denom = (np.sum((r1 - mu) ** 2)) ** 2
        for j in range(1, k):
            num = np.sum(((r1[j:] - mu) ** 2) * ((r1[:-j] - mu) ** 2))
            delta[j - 1] = ((2.0 * (k - j) / k) ** 2) * (num / denom)

        phi_k = np.sum(delta)
        z_star = (vr - 1.0) / np.sqrt(phi_k) if phi_k > 0 else 0.0
        p_val = 2.0 * (1.0 - stats.norm.cdf(abs(z_star)))

        return {"vr": float(vr), "z_stat": float(z_star), "p_value": float(p_val)}

    @classmethod
    def multi_scale_memory(cls, prices: pd.Series, returns: pd.Series) -> Dict[str, Any]:
        """
        Constructs multi-scale memory profile and dynamic lag candidate set.
        """
        r = returns.dropna()
        if len(r) < 50:
            return {
                "dynamic_lags": [1, 3, 5],
                "hurst": 0.5,
                "short_term_mean_reverting": False,
                "long_term_trending": False,
            }

        max_lag = min(30, len(r) // 3)
        acf_vals = acf(r, nlags=max_lag, fft=True)
        conf_interval = 1.96 / np.sqrt(len(r))
        sig_lags = [i for i, val in enumerate(acf_vals[1:], 1) if abs(val) > conf_interval]

        if not sig_lags:
            sig_lags = [1, 3, 5]

        vr_short = cls.lo_mackinlay_vr(prices, k=3)
        vr_long = cls.lo_mackinlay_vr(prices, k=20)

        return {
            "dynamic_lags": sorted(list(set(sig_lags[:5] + [1, 5, 20]))),
            "hurst": cls.compute_hurst_dfa(returns),
            "short_term_mean_reverting": bool(vr_short["vr"] < 1.0 and vr_short["p_value"] < 0.05),
            "long_term_trending": bool(vr_long["vr"] > 1.0 and vr_long["p_value"] < 0.05),
            "vr_short_stat": vr_short,
            "vr_long_stat": vr_long,
        }