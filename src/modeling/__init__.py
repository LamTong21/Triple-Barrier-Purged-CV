"""
src/modeling/__init__.py

Time-series cross-validation, probability calibration, dynamic thresholding,
and outer loop modeling routines:
- TimeBasedPurgedTimeSeriesSplit: Purged & Embargo CV (López de Prado standard).
- TimeSeriesCalibrator: Bounded Platt Scaling via L-BFGS-B optimization.
- DynamicThresholdEngine: Rolling percentile thresholding and Z-score bet sizing.
- PurgedCVTrainer: End-to-end execution trainer for out-of-sample inference.
"""

from src.modeling.splitters import TimeBasedPurgedTimeSeriesSplit
from src.modeling.calibrator import TimeSeriesCalibrator
from src.modeling.thresholding import DynamicThresholdEngine
from src.modeling.trainer import PurgedCVTrainer

__all__ = [
    "TimeBasedPurgedTimeSeriesSplit",
    "TimeSeriesCalibrator",
    "DynamicThresholdEngine",
    "PurgedCVTrainer",
]