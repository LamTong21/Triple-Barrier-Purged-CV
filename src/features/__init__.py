"""
src/features/__init__.py

Feature engineering, dynamic transformations, and consensus feature selection:
- BaseFeatureStrategy: Abstract interface for feature generation blocks.
- Technical, Microstructure, and Order Flow divergence feature strategies.
- StatelessFeatureRouter: Point-in-time shifted feature calculation.
- Layer14LagTransformEngine: Volatility-scaled causal lags and first-order momentum.
- Stage4ConsensusSelectionRouter: Hierarchical clustered consensus selection.
- EconometricsFeaturePipeline: Scikit-learn compliant feature selection pipeline.
"""

from src.features.base_strategy import BaseFeatureStrategy
from src.features.technical_strategies import (
    BaselineStrategy,
    TrendMomentumStrategy,
    OscillatorMeanReversionStrategy,
    FractionalMemoryStrategy,
    VolatilityDynamicsStrategy,
    VolatilityJumpDiffusionStrategy,
    MultiScaleComplexityStrategy,
    WaveletMultiResolutionStrategy,
    IntradayShadowPressureStrategy,
    KinematicDynamicsStrategy,
)
from src.features.microstructure_strategies import (
    L2MicrostructureStrategy,
    OrderFlowToxicityStrategy,
    LiquidityMicrostructureStrategy,
)
from src.features.orderflow_divergence import (
    VolumePriceDivergenceStrategy,
    DirectionalPressureStrategy,
    SignedVolatilityDivergenceStrategy,
    MarketContextStrategy,
)
from src.features.routers import StatelessFeatureRouter
from src.features.lag_transform import Layer14LagTransformEngine
from src.features.consensus_selection import Stage4ConsensusSelectionRouter
from src.features.pipeline import EconometricsFeaturePipeline

__all__ = [
    "BaseFeatureStrategy",
    "BaselineStrategy",
    "TrendMomentumStrategy",
    "OscillatorMeanReversionStrategy",
    "FractionalMemoryStrategy",
    "VolatilityDynamicsStrategy",
    "VolatilityJumpDiffusionStrategy",
    "MultiScaleComplexityStrategy",
    "WaveletMultiResolutionStrategy",
    "IntradayShadowPressureStrategy",
    "KinematicDynamicsStrategy",
    "L2MicrostructureStrategy",
    "OrderFlowToxicityStrategy",
    "LiquidityMicrostructureStrategy",
    "VolumePriceDivergenceStrategy",
    "DirectionalPressureStrategy",
    "SignedVolatilityDivergenceStrategy",
    "MarketContextStrategy",
    "StatelessFeatureRouter",
    "Layer14LagTransformEngine",
    "Stage4ConsensusSelectionRouter",
    "EconometricsFeaturePipeline",
]