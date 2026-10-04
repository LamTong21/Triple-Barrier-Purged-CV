"""
src/backtest/execution.py

Causal trade execution and concurrent trade conflict resolution.
Prevents holding period overlapping and unphysical portfolio over-leverage
under Single-Asset execution constraints.
"""

import numpy as np
import pandas as pd


class ConcurrentTradeResolver:
    """
    Enforces causal order execution under single-position capacity constraints:
    If a trade is initiated at bar t and exits at barrier_touch_time, any subsequent
    entry signals generated during the open interval (t, barrier_touch_time) are
    strictly discarded.
    """

    @staticmethod
    def resolve(df_oos: pd.DataFrame, cost_per_trade: float) -> pd.DataFrame:
        """
        Filters overlapping concurrent entries and calculates net trade returns.

        Parameters
        ----------
        df_oos : pd.DataFrame
            Out-of-sample prediction logs containing 'pred_signal', 'barrier_return',
            'bet_size', and 'barrier_touch_time'.
        cost_per_trade : float
            One-way transaction cost + slippage rate.

        Returns
        -------
        pd.DataFrame
            DataFrame augmented with 'executed_signal', 'executed_return',
            'net_trade_pnl', 'equity_curve', and 'drawdown'.
        """
        df = df_oos.sort_index().copy()
        df["barrier_touch_time"] = pd.to_datetime(df["barrier_touch_time"])

        n = len(df)
        active_signals = np.zeros(n, dtype=int)
        realized_returns = np.zeros(n, dtype=float)
        realized_bets = np.zeros(n, dtype=float)

        current_unlock_time = pd.Timestamp.min
        timestamps = df.index
        signals = df["pred_signal"].values
        returns = df["barrier_return"].values
        bets = df["bet_size"].values
        touch_times = df["barrier_touch_time"].values

        for i in range(n):
            t_now = timestamps[i]
            # Only permit trade initiation if previous position has been settled
            if signals[i] == 1 and t_now >= current_unlock_time:
                active_signals[i] = 1
                realized_returns[i] = returns[i]
                realized_bets[i] = bets[i]
                current_unlock_time = pd.Timestamp(touch_times[i])

        df["executed_signal"] = active_signals
        df["executed_return"] = realized_returns
        df["executed_bet"] = realized_bets

        # Two-way round-trip transaction friction: 2 * (fee + slippage) * position_size
        df["trade_cost"] = np.where(
            df["executed_signal"] == 1,
            2.0 * cost_per_trade * df["executed_bet"],
            0.0,
        )
        df["net_trade_pnl"] = (df["executed_return"] * df["executed_bet"]) - df["trade_cost"]

        # Cumulative performance curves
        df["equity_curve"] = (1.0 + df["net_trade_pnl"]).cumprod()
        df["peak_equity"] = df["equity_curve"].cummax()
        df["drawdown"] = (df["equity_curve"] - df["peak_equity"]) / df["peak_equity"]

        return df