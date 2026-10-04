"""
src/diagnostics/nonlinearity.py

Nonlinear dynamics and statistical complexity diagnostics:
- Brock-Dechert-Scheinkman (BDS) test for general nonlinear dependence.
- Permutation Entropy (Bandt & Pompe) for multi-scale dynamical complexity.
- Volume-return Spearman rank correlation.
"""

import math
from typing import Dict, Any
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import bds
from sklearn.feature_selection import mutual_info_regression


class Layer7bVolumeDiagnostics:
    """Evaluates volume-price feedback and trading volume significance."""

    @staticmethod
    def run(df: pd.DataFrame) -> Dict[str, Any]:
        v = df["volume"].dropna()
        r_abs = np.log(df["close_adj"] / df["close_adj"].shift(1)).abs().dropna()

        aligned = pd.concat([v, r_abs], axis=1).dropna()
        if aligned.empty or len(aligned) < 30:
            return {"is_volume_significant": False, "vol_ret_corr": 0.0}

        corr, p_val = stats.spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])

        return {
            "vol_ret_corr": float(corr),
            "is_volume_significant": bool(p_val < 0.05 and corr > 0.1),
        }


class Layer8NonlinearDependence:
    """Evaluates BDS nonlinearity test on linear AR(1) residuals."""

    @staticmethod
    def run(returns: pd.Series) -> Dict[str, Any]:
        r = returns.dropna()
        if len(r) < 50:
            return {
                "bds_stat_dim2": 0.0,
                "bds_pvalue_dim2": 1.0,
                "is_nonlinear_dependent": False,
                "mutual_information_lag1": 0.0,
            }

        ar_res = AutoReg(r.values, lags=1).fit()
        resid = ar_res.resid
        resid_std = (resid - np.nanmean(resid)) / (np.nanstd(resid) + 1e-8)

        bds_stat, p_val = bds(resid_std, max_dim=2, epsilon=None)

        x_lag = r.shift(1).dropna()
        y_curr = r.iloc[1:]
        mi = mutual_info_regression(x_lag.values.reshape(-1, 1), y_curr.values, random_state=42)[0]

        return {
            "bds_stat_dim2": float(bds_stat),
            "bds_pvalue_dim2": float(p_val),
            "is_nonlinear_dependent": bool(p_val < 0.05),
            "mutual_information_lag1": float(mi),
        }


class Layer8bComplexityDiagnostics:
    """Computes Bandt & Pompe Permutation Entropy (order m=3, tau=1)."""

    @staticmethod
    def run(returns: pd.Series) -> Dict[str, Any]:
        r = returns.dropna().values
        if len(r) < 30:
            return {"is_high_complexity": False, "permutation_entropy": 1.0}

        m, tau = 3, 1
        n = len(r) - (m - 1) * tau
        patterns = np.array([r[i : i + m * tau : tau] for i in range(n)])
        ranks = np.argsort(patterns, axis=1)
        _, counts = np.unique(ranks, axis=0, return_counts=True)
        probs = counts / counts.sum()

        pe = -np.sum(probs * np.log2(probs + 1e-8)) / np.log2(math.factorial(m))

        return {
            "permutation_entropy": float(pe),
            "is_high_complexity": bool(pe > 0.85),
        }