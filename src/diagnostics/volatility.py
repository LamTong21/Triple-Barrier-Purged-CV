"""
src/diagnostics/volatility.py

Volatility dynamics and jump-diffusion econometric diagnostics:
- Engle's ARCH LM test for volatility clustering.
- Leverage effect correlation and asymmetric volatility checks.
- Barndorff-Nielsen & Shephard (BNS) realized bipower variation jump test.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import het_arch


class Layer6VolatilityDynamics:
    """
    Evaluates ARCH effects, volatility clustering, and asymmetry.
    """

    @staticmethod
    def estimate(df: pd.DataFrame) -> Dict[str, Any]:
        """Runs ARCH LM and returns leverage correlation."""
        c = df["close_adj"]
        ret = np.log(c / c.shift(1)).dropna()
        if len(ret) < 30:
            return {
                "arch_lm_stat": 0.0,
                "arch_lm_pvalue": 1.0,
                "has_arch_effect": False,
                "leverage_effect_corr": 0.0,
                "has_asymmetric_vol": False,
            }

        lm_stat, lm_p, _, _ = het_arch(ret, nlags=5)
        ret_lag = ret.shift(1).dropna()
        vol_proxy = (ret ** 2).iloc[1:]
        aligned_df = pd.concat([ret_lag, vol_proxy], axis=1).dropna()
        leverage_corr = aligned_df.iloc[:, 0].corr(aligned_df.iloc[:, 1])

        return {
            "arch_lm_stat": float(lm_stat),
            "arch_lm_pvalue": float(lm_p),
            "has_arch_effect": bool(lm_p < 0.05),
            "leverage_effect_corr": float(leverage_corr),
            "has_asymmetric_vol": bool(leverage_corr < -0.1),
        }


class Layer6bVolatilityJumpDiagnostics:
    """
    Barndorff-Nielsen & Shephard (BNS) Realized Bipower Variation Jump Test.
    Decomposes quadratic variation into continuous diffusion vs discontinuous jumps.
    """

    @staticmethod
    def run(df: pd.DataFrame, window: int = 20) -> Dict[str, Any]:
        """Evaluates jump presence and mean jump proportion."""
        r = df["log_return"].dropna()
        if len(r) < window * 2:
            return {"has_volatility_jumps": False, "mean_jump_ratio": 0.0}

        rv = (r ** 2).rolling(window).sum()
        abs_r = r.abs()
        bv = (np.pi / 2.0) * (abs_r * abs_r.shift(1)).rolling(window).sum()

        jump_ratio = np.maximum(rv - bv, 0.0) / (rv + 1e-8)
        significant_jumps = (jump_ratio > 0.25).sum()

        return {
            "has_volatility_jumps": bool(significant_jumps > (len(r) * 0.05)),
            "mean_jump_ratio": float(jump_ratio.mean()),
        }