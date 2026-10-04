"""
src/features/routers.py

Orchestrates all stateless strategies and enforces a global Point-in-Time Shift-Lock.
"""

from typing import Sequence
import pandas as pd

from src.features.technical_strategies import (
    BaselineStrategy,
    TrendMomentumStrategy,
    OscillatorMeanReversionStrategy,
    VolatilityDynamicsStrategy,
    KinematicDynamicsStrategy,
    WaveletMultiResolutionStrategy,
    IntradayShadowPressureStrategy,
)
from src.features.microstructure_strategies import LiquidityMicrostructureStrategy
from src.features.orderflow_divergence import (
    VolumePriceDivergenceStrategy,
    DirectionalPressureStrategy,
    SignedVolatilityDivergenceStrategy,
    MarketContextStrategy,
)


class StatelessFeatureRouter:
    """
    Executes modular feature builders and applies a mandatory 1-bar backward shift.
    Guarantees that features at index 't' only contain market information up to 't-1'.
    """

    def __init__(self, default_lags: Sequence[int] = (1, 3, 5, 10, 20), enforce_shift: bool = True):
        self.enforce_shift = enforce_shift
        self.strategies = {
            "Baseline": BaselineStrategy(dynamic_lags=default_lags),
            "Trend": TrendMomentumStrategy(dynamic_lags=default_lags),
            "Oscillator": OscillatorMeanReversionStrategy(dynamic_lags=default_lags),
            "Volatility": VolatilityDynamicsStrategy(dynamic_lags=default_lags, has_asymmetric_vol=True),
            "Kinematics": KinematicDynamicsStrategy(),
            "Wavelet": WaveletMultiResolutionStrategy(),
            "Intraday": IntradayShadowPressureStrategy(),
            "FlowDivergence": VolumePriceDivergenceStrategy(window=20),
            "Directional": DirectionalPressureStrategy(window=14),
            "SignedVol": SignedVolatilityDivergenceStrategy(window=20),
            "Liquidity": LiquidityMicrostructureStrategy(),
            "MarketContext": MarketContextStrategy(window_short=20, window_long=60),
        }

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_frames = [strategy.construct(df) for strategy in self.strategies.values()]
        X_features = pd.concat(feature_frames, axis=1)

        # MANDATORY POINT-IN-TIME LOCK
        if self.enforce_shift:
            X_features = X_features.shift(1)

        return pd.concat([df, X_features], axis=1)