"""
src/backtest/metrics.py

Financial risk, return, and performance accounting metrics:
- Sharpe, Sortino, and Calmar Ratios.
- Maximum Drawdown (MDD) and Annualized Volatility.
- Win Rate and Win/Loss Ratio Payoff Asymmetry.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd


class FinancialRiskMetrics:
    """
    Computes standard quantitative finance and risk metrics.
    """

    @staticmethod
    def calculate(
        df_executed: pd.DataFrame,
        label: str = "Portfolio",
        annual_factor: int = 252,
        risk_free_rate: float = 0.045,
    ) -> Dict[str, Any]:
        """
        Calculates financial performance metrics from executed trade logs.

        Parameters
        ----------
        df_executed : pd.DataFrame
            Executed trade DataFrame with 'executed_signal', 'net_trade_pnl',
            'equity_curve', and 'drawdown'.
        label : str, default='Portfolio'
            Identifier label for the dataset or fold.
        annual_factor : int, default=252
            Trading sessions per calendar year.
        risk_free_rate : float, default=0.045
            Annualized risk-free benchmark rate.

        Returns
        -------
        Dict[str, Any]
            Dictionary of computed performance and risk ratios.
        """
        trades = df_executed[df_executed["executed_signal"] == 1]
        n_trades = len(trades)

        if n_trades == 0:
            return {
                "Label": label,
                "Total Trades": 0,
                "Win Rate (%)": 0.0,
                "Profit Factor": np.nan,
                "Win/Loss Ratio": np.nan,
                "Cum. Return (%)": 0.0,
                "CAGR (%)": 0.0,
                "Ann. Volatility (%)": 0.0,
                "Sharpe Ratio": 0.0,
                "Sortino Ratio": 0.0,
                "Max Drawdown (%)": 0.0,
                "Calmar Ratio": np.nan,
                "Avg Bet Size": 0.0,
            }

        pnl = trades["net_trade_pnl"].values
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]

        win_rate = len(wins) / n_trades if n_trades > 0 else 0.0
        profit_factor = (
            np.sum(wins) / np.abs(np.sum(losses))
            if len(losses) > 0 and np.abs(np.sum(losses)) > 0
            else np.nan
        )
        avg_win = np.mean(wins) if len(wins) > 0 else 0.0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.0
        win_loss_ratio = avg_win / np.abs(avg_loss) if avg_loss != 0 else np.nan

        daily_pnl = df_executed["net_trade_pnl"].values
        n_days = len(daily_pnl)
        cumulative_return = df_executed["equity_curve"].iloc[-1] - 1.0

        years = n_days / annual_factor
        cagr = (
            (1.0 + cumulative_return) ** (1.0 / years) - 1.0
            if years > 0 and cumulative_return > -1.0
            else np.nan
        )
        ann_vol = np.std(daily_pnl, ddof=1) * np.sqrt(annual_factor)

        sharpe = (
            (np.mean(daily_pnl) * annual_factor - risk_free_rate) / ann_vol
            if ann_vol > 0
            else 0.0
        )

        downside_returns = daily_pnl[daily_pnl < 0]
        downside_vol = (
            np.std(downside_returns, ddof=1) * np.sqrt(annual_factor)
            if len(downside_returns) > 1
            else 1e-8
        )
        sortino = (
            (np.mean(daily_pnl) * annual_factor - risk_free_rate) / downside_vol
            if downside_vol > 0
            else 0.0
        )

        max_dd = df_executed["drawdown"].min()
        calmar = (
            cagr / np.abs(max_dd)
            if max_dd < 0 and not np.isnan(cagr)
            else np.nan
        )

        return {
            "Label": label,
            "Total Trades": int(n_trades),
            "Win Rate (%)": win_rate * 100.0,
            "Profit Factor": profit_factor,
            "Win/Loss Ratio": win_loss_ratio,
            "Cum. Return (%)": cumulative_return * 100.0,
            "CAGR (%)": cagr * 100.0,
            "Ann. Volatility (%)": ann_vol * 100.0,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Max Drawdown (%)": max_dd * 100.0,
            "Calmar Ratio": calmar,
            "Avg Bet Size": trades["executed_bet"].mean(),
        }