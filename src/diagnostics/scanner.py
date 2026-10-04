"""
src/diagnostics/scanner.py

Stage 2 Diagnostic Scanner: executes the econometric test battery on training data
and packages parameters and conditional flags into a routing payload.
"""

from typing import Dict, Any
import pandas as pd

from src.diagnostics.memory import Layer4MemoryDependence
from src.diagnostics.spectral import Layer4bSpectralCycleDiagnostics
from src.diagnostics.fractional import Layer5FractionalIntegration
from src.diagnostics.volatility import Layer6VolatilityDynamics, Layer6bVolatilityJumpDiagnostics
from src.diagnostics.nonlinearity import Layer7bVolumeDiagnostics, Layer8NonlinearDependence, Layer8bComplexityDiagnostics
from src.diagnostics.microstructure import Layer7cMicrostructureDiagnostics


class Stage2SingleAssetDGPScanner:
    """
    Centralized scanner orchestrating econometric hypothesis testing.
    Outputs routing configurations to guide downstream feature transformations.
    """

    @staticmethod
    def execute(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Executes diagnostic suites and packages routing payload.

        Parameters
        ----------
        df : pd.DataFrame
            Single asset training partition.

        Returns
        -------
        Dict[str, Any]
            Routing payload containing dynamic lags, optimal d*, and hypothesis flags.
        """
        returns = df["log_return"].dropna()
        prices = df["close_adj"].dropna()

        mem_stats = Layer4MemoryDependence.multi_scale_memory(prices, returns)
        cycle_stats = Layer4bSpectralCycleDiagnostics.run(prices)
        frac_stats = Layer5FractionalIntegration.find_optimal_d(prices)
        vol_stats = Layer6VolatilityDynamics.estimate(df)
        jump_stats = Layer6bVolatilityJumpDiagnostics.run(df)
        volu_stats = Layer7bVolumeDiagnostics.run(df)
        nonlin_stats = Layer8NonlinearDependence.run(returns)
        complex_stats = Layer8bComplexityDiagnostics.run(returns)
        micro_stats = Layer7cMicrostructureDiagnostics.run(df)

        routing_payload = {
            "dynamic_lags": mem_stats["dynamic_lags"],
            "optimal_d": frac_stats["optimal_d"],
            "dominant_cycle": cycle_stats["dominant_cycle_len"],
            "flags": {
                "has_long_trend": mem_stats["long_term_trending"],
                "has_short_reversion": mem_stats["short_term_mean_reverting"],
                "has_vol_clustering": vol_stats["has_arch_effect"],
                "has_asymmetric_vol": vol_stats["has_asymmetric_vol"],
                "is_volume_significant": volu_stats["is_volume_significant"],
                "is_nonlinear": nonlin_stats["is_nonlinear_dependent"],
                "has_vol_jumps": jump_stats["has_volatility_jumps"],
                "is_high_complexity": complex_stats["is_high_complexity"],
                "has_l2_orderbook": micro_stats["has_l2_orderbook"],
                "micro_deviation_sig": micro_stats["micro_deviation_sig"],
            },
        }

        return routing_payload