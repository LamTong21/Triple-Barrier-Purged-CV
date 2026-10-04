"""
src/__init__.py

Core Python package for quantitative financial dynamics research:
- Causal bar geometry auditing and Triple Barrier labeling.
- Data-driven DGP diagnostic tests and spectral analysis.
- Multi-scale kinematic, microstructure, and order flow feature engineering.
- Purged & Embargo Cross-Validation with Bounded Platt Scaling calibration.
- Portfolio PnL auditing and concurrent holding resolution.
"""

from src.data_pipeline.audit import Layer1DataIntegrity
from src.data_pipeline.topology import Layer2MicrostructureTopologies
from src.data_pipeline.labeling import Layer0TripleBarrier
from src.features.pipeline import EconometricsFeaturePipeline
from src.modeling.splitters import TimeBasedPurgedTimeSeriesSplit
from src.modeling.calibrator import TimeSeriesCalibrator
from src.backtest.engine import PortfolioBacktestEngine

__version__ = "0.7.2"
__author__ = "Research Team"
__all__ = [
    "Layer1DataIntegrity",
    "Layer2MicrostructureTopologies",
    "Layer0TripleBarrier",
    "EconometricsFeaturePipeline",
    "TimeBasedPurgedTimeSeriesSplit",
    "TimeSeriesCalibrator",
    "PortfolioBacktestEngine",
]