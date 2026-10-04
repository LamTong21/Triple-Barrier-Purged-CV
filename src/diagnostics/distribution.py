"""
src/diagnostics/distribution.py

Distributional characteristics and normality testing on financial return series.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from scipy import stats


class Layer3DistributionalDiagnostics:
    """
    Evaluates empirical return distribution moments (mean, variance, skewness, kurtosis)
    and executes the Jarque-Bera asymptotic test for normality.
    """

    @staticmethod
    def run(returns: pd.Series) -> Dict[str, Any]:
        """
        Executes distribution diagnostics.

        Parameters
        ----------
        returns : pd.Series
            Univariate return series.

        Returns
        -------
        Dict[str, Any]
            Statistical moments and normality test flags.
        """
        r = returns.dropna()
        if len(r) < 30:
            return {
                "mean": 0.0,
                "std": 0.0,
                "skewness": 0.0,
                "kurtosis": 3.0,
                "is_leptokurtic": False,
                "jarque_bera_stat": 0.0,
                "jarque_bera_pvalue": 1.0,
                "reject_normality": False,
            }

        skew = float(stats.skew(r, bias=False))
        kurt = float(stats.kurtosis(r, fisher=False, bias=False))
        jb_stat, jb_p = stats.jarque_bera(r)

        return {
            "mean": float(np.mean(r)),
            "std": float(np.std(r, ddof=1)),
            "skewness": skew,
            "kurtosis": kurt,
            "is_leptokurtic": bool(kurt > 3.0),
            "jarque_bera_stat": float(jb_stat),
            "jarque_bera_pvalue": float(jb_p),
            "reject_normality": bool(jb_p < 0.05),
        }