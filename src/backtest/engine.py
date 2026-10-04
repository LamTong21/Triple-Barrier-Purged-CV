"""
src/backtest/engine.py

End-to-end Quantitative Portfolio Backtest Engine:
- Coordinates execution locking, trade friction deductions, and metric computation.
- Generates publication-grade equity curve and underwater drawdown visualizations.
"""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.backtest.execution import ConcurrentTradeResolver
from src.backtest.metrics import FinancialRiskMetrics


class PortfolioBacktestEngine:
    """
    Simulates portfolio execution across out-of-sample predictions.
    """

    def __init__(
        self,
        fee_rate: float = 0.0015,
        slippage: float = 0.0005,
        annual_factor: int = 252,
        risk_free_rate: float = 0.045,
    ):
        """
        Parameters
        ----------
        fee_rate : float, default=0.0015
            One-way brokerage fee and taxes (0.15%).
        slippage : float, default=0.0005
            One-way estimated market execution slippage (0.05%).
        annual_factor : int, default=252
            Annualization factor for trading days.
        risk_free_rate : float, default=0.045
            Annual risk-free benchmark rate (4.5%).
        """
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.cost_per_trade = self.fee_rate + self.slippage
        self.annual_factor = annual_factor
        self.risk_free_rate = risk_free_rate

    def resolve_concurrent_trades(self, df_oos: pd.DataFrame) -> pd.DataFrame:
        """Resolves overlapping positions causally."""
        return ConcurrentTradeResolver.resolve(df_oos, self.cost_per_trade)

    def compute_metrics(self, df_executed: pd.DataFrame, label: str = "Portfolio") -> Dict[str, Any]:
        """Computes comprehensive performance statistics."""
        return FinancialRiskMetrics.calculate(
            df_executed=df_executed,
            label=label,
            annual_factor=self.annual_factor,
            risk_free_rate=self.risk_free_rate,
        )

    def generate_full_report(
        self, df_oos: pd.DataFrame, fig_path: str = "portfolio_performance.png"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Executes complete backtest, compiles fold-by-fold summary, and exports chart.

        Parameters
        ----------
        df_oos : pd.DataFrame
            Out-of-sample predictions DataFrame.
        fig_path : str, default='portfolio_performance.png'
            Path to save generated equity curve image.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            - df_backtest: Detailed trade execution log DataFrame.
            - summary_df: Summary metrics table indexed by partition label.
        """
        df_backtest = self.resolve_concurrent_trades(df_oos)

        # Full portfolio metrics
        overall_metrics = self.compute_metrics(df_backtest, label="Full OOS Portfolio")

        # Fold-by-fold performance audit
        fold_metrics_list = []
        for f in sorted(df_backtest["fold"].unique()):
            sub_df = df_backtest[df_backtest["fold"] == f].copy()
            sub_df["equity_curve"] = (1.0 + sub_df["net_trade_pnl"]).cumprod()
            sub_df["peak_equity"] = sub_df["equity_curve"].cummax()
            sub_df["drawdown"] = (sub_df["equity_curve"] - sub_df["peak_equity"]) / sub_df["peak_equity"]
            fold_metrics_list.append(self.compute_metrics(sub_df, label=f"Fold {f}"))

        summary_df = pd.DataFrame(fold_metrics_list + [overall_metrics]).set_index("Label")
        self.plot_performance(df_backtest, fig_path)

        return df_backtest, summary_df

    def plot_performance(self, df: pd.DataFrame, fig_path: str):
        """Plots publication-grade dual-panel performance figure."""
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]}
        )

        # Panel 1: Equity Curve
        ax1.plot(
            df.index,
            df["equity_curve"],
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

        # Delineate CV folds
        for f in df["fold"].unique():
            fold_start = df[df["fold"] == f].index[0]
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
            df.index,
            df["drawdown"] * 100,
            0,
            color="#d62728",
            alpha=0.4,
            label="Underwater Drawdown",
        )
        ax2.plot(df.index, df["drawdown"] * 100, color="#d62728", linewidth=1.0)
        ax2.set_ylabel("Drawdown (%)", fontsize=11)
        ax2.set_xlabel("Out-of-Sample Evaluation Window", fontsize=11)
        ax2.grid(True, linestyle=":", alpha=0.5)
        ax2.legend(loc="lower left")

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()