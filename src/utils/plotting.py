"""
src/utils/plotting.py

Generates publication-quality charts for portfolio performance, drawdowns,
and probability calibration analysis.
"""

from typing import Optional, Sequence
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve


def plot_equity_and_drawdown(
    df_executed: pd.DataFrame,
    fig_path: str = "portfolio_performance.png",
    dpi: int = 300,
    fold_col: Optional[str] = "fold",
) -> None:
    """
    Plots a dual-panel figure displaying the cumulative equity curve and underwater drawdown.

    Parameters
    ----------
    df_executed : pd.DataFrame
        Trade log with DatetimeIndex containing 'equity_curve' and 'drawdown'.
    fig_path : str, default='portfolio_performance.png'
        File destination path for the saved image.
    dpi : int, default=300
        Resolution of the saved chart.
    fold_col : Optional[str], default='fold'
        Column name indicating cross-validation folds to delineate with vertical lines.
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]}
    )

    # Panel 1: Equity Curve
    ax1.plot(
        df_executed.index,
        df_executed["equity_curve"],
        color="#1f77b4",
        linewidth=1.8,
        label="Model Portfolio (Dynamic Sizing)",
    )
    ax1.axhline(1.0, color="gray", linestyle="--", alpha=0.6)
    ax1.set_title(
        "Out-of-Sample Performance Curve (Purged & Embargo CV)",
        fontsize=13,
        fontweight="bold",
    )
    ax1.set_ylabel("Portfolio Value (Base = 1.0)", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="upper left")

    # Delineate Cross-Validation Folds
    if fold_col and fold_col in df_executed.columns:
        for f in df_executed[fold_col].unique():
            fold_start = df_executed[df_executed[fold_col] == f].index[0]
            ax1.axvline(fold_start, color="darkred", linestyle=":", alpha=0.4)
            ax1.text(
                fold_start,
                ax1.get_ylim()[1] * 0.97,
                f" F{f}",
                color="darkred",
                fontsize=9,
                fontweight="bold",
            )

    # Panel 2: Underwater Drawdown
    ax2.fill_between(
        df_executed.index,
        df_executed["drawdown"] * 100,
        0,
        color="#d62728",
        alpha=0.4,
        label="Underwater Drawdown",
    )
    ax2.plot(df_executed.index, df_executed["drawdown"] * 100, color="#d62728", linewidth=1.0)
    ax2.set_ylabel("Drawdown (%)", fontsize=11)
    ax2.set_xlabel("Out-of-Sample Evaluation Window", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="lower left")

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.tight_layout()
    plt.savefig(fig_path, dpi=dpi)
    plt.close()
    print(f"[✓] Equity and Drawdown chart exported to {fig_path}")


def plot_probability_calibration_curve(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    n_bins: int = 5,
    strategy: str = "quantile",
    fig_path: str = "calibration_curve.png",
    dpi: int = 300,
) -> None:
    """
    Plots empirical calibration reliability diagram vs perfectly calibrated diagonal.

    Parameters
    ----------
    y_true : Sequence[int]
        Binary ground truth labels (0 or 1).
    y_prob : Sequence[float]
        Predicted probabilities for the positive class.
    n_bins : int, default=5
        Number of probability bins.
    strategy : str, default='quantile'
        Binning strategy ('quantile' or 'uniform').
    fig_path : str, default='calibration_curve.png'
        File destination path for the saved image.
    dpi : int, default=300
        Resolution of the saved chart.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy)

    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    plt.plot(
        prob_pred,
        prob_true,
        marker="o",
        linewidth=1.8,
        color="#1f77b4",
        label=f"Calibrated Estimator ({strategy})",
    )

    plt.title("Empirical Probability Calibration Curve", fontsize=12, fontweight="bold")
    plt.xlabel("Mean Predicted Probability", fontsize=11)
    plt.ylabel("Empirical Win Rate (Fraction of Positives)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=dpi)
    plt.close()
    print(f"[✓] Calibration curve exported to {fig_path}")