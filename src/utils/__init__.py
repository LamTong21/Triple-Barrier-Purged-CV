"""
src/utils/__init__.py

Utility tools for statistical reporting, probability calibration diagnostics,
and publication-grade financial plotting:
- plot_equity_and_drawdown: Visualizes cumulative equity and underwater drawdown.
- plot_probability_calibration_curve: Generates binned calibration diagnostic plots.
- evaluate_brier_decomposition: Computes Murphy (1973) Resolution and Reliability.
- print_classification_and_confusion_matrix: Displays precision, recall, and 2x2 matrix.
"""

from src.utils.plotting import (
    plot_equity_and_drawdown,
    plot_probability_calibration_curve,
)
from src.utils.metrics_logger import (
    evaluate_brier_decomposition,
    print_classification_and_confusion_matrix,
)

__all__ = [
    "plot_equity_and_drawdown",
    "plot_probability_calibration_curve",
    "evaluate_brier_decomposition",
    "print_classification_and_confusion_matrix",
]