#!/usr/bin/env python3
"""
scripts/run_portfolio_audit.py

Quantitatively audits Portfolio PnL and resolves concurrent overlapping holding periods.
Incorporates:
- Causal trade execution lockout (ignores overlapping signals during open positions)
- 2-way friction deduction (brokerage fees + market slippage)
- Dynamic Z-Score position sizing
- Economic and risk metric computation (Sharpe, Sortino, Calmar, Max Drawdown, Win/Loss Ratio)
- Generates high-resolution Equity and Underwater Drawdown curves
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def parse_args():
    parser = argparse.ArgumentParser(description="Portfolio PnL Audit & Concurrent Trade Resolution Engine")
    parser.add_argument("--predictions_path", type=str, default="data/processed/df_oos_predictions.csv", help="OOS predictions path")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory for PnL logs")
    parser.add_argument("--fig_path", type=str, default="portfolio_performance.png", help="Path to save performance chart")
    parser.add_argument("--fee_rate", type=float, default=0.0015, help="One-way transaction fee + taxes (0.15%)")
    parser.add_argument("--slippage", type=float, default=0.0005, help="Estimated one-way execution slippage (0.05%)")
    parser.add_argument("--risk_free_rate", type=float, default=0.045, help="Annual risk-free rate (4.5%)")
    parser.add_argument("--annual_factor", type=int, default=252, help="Trading sessions per year")
    return parser.parse_args()


class PortfolioBacktestEngine:
    def __init__(self, fee_rate: float, slippage: float, risk_free_rate: float, annual_factor: int):
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.cost_per_trade = self.fee_rate + self.slippage
        self.risk_free_rate = risk_free_rate
        self.annual_factor = annual_factor

    def resolve_concurrent_trades(self, df_oos: pd.DataFrame) -> pd.DataFrame:
        """
        Enforces causal order execution: locks out subsequent buy signals while a trade is active
        until its empirical barrier_touch_time is reached.
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
            if signals[i] == 1 and t_now >= current_unlock_time:
                active_signals[i] = 1
                realized_returns[i] = returns[i]
                realized_bets[i] = bets[i]
                current_unlock_time = pd.Timestamp(touch_times[i])

        df["executed_signal"] = active_signals
        df["executed_return"] = realized_returns
        df["executed_bet"] = realized_bets

        # Two-way round-trip transaction friction
        df["trade_cost"] = np.where(df["executed_signal"] == 1, 2.0 * self.cost_per_trade * df["executed_bet"], 0.0)
        df["net_trade_pnl"] = (df["executed_return"] * df["executed_bet"]) - df["trade_cost"]

        df["equity_curve"] = (1.0 + df["net_trade_pnl"]).cumprod()
        df["peak_equity"] = df["equity_curve"].cummax()
        df["drawdown"] = (df["equity_curve"] - df["peak_equity"]) / df["peak_equity"]

        return df

    def compute_metrics(self, df_executed: pd.DataFrame, label: str = "Portfolio") -> dict:
        trades = df_executed[df_executed["executed_signal"] == 1]
        n_trades = len(trades)

        if n_trades == 0:
            return {"Label": label, "Total Trades": 0}

        pnl = trades["net_trade_pnl"].values
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]

        win_rate = len(wins) / n_trades if n_trades > 0 else 0.0
        profit_factor = np.sum(wins) / np.abs(np.sum(losses)) if len(losses) > 0 and np.abs(np.sum(losses)) > 0 else np.nan
        avg_win = np.mean(wins) if len(wins) > 0 else 0.0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.0
        win_loss_ratio = avg_win / np.abs(avg_loss) if avg_loss != 0 else np.nan

        daily_pnl = df_executed["net_trade_pnl"].values
        n_days = len(daily_pnl)
        cumulative_return = df_executed["equity_curve"].iloc[-1] - 1.0

        years = n_days / self.annual_factor
        cagr = (1.0 + cumulative_return) ** (1.0 / years) - 1.0 if years > 0 and cumulative_return > -1.0 else np.nan
        ann_vol = np.std(daily_pnl, ddof=1) * np.sqrt(self.annual_factor)

        sharpe = (np.mean(daily_pnl) * self.annual_factor - self.risk_free_rate) / ann_vol if ann_vol > 0 else 0.0
        downside_returns = daily_pnl[daily_pnl < 0]
        downside_vol = np.std(downside_returns, ddof=1) * np.sqrt(self.annual_factor) if len(downside_returns) > 1 else 1e-8
        sortino = (np.mean(daily_pnl) * self.annual_factor - self.risk_free_rate) / downside_vol if downside_vol > 0 else 0.0

        max_dd = df_executed["drawdown"].min()
        calmar = cagr / np.abs(max_dd) if max_dd < 0 and not np.isnan(cagr) else np.nan

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
            "Avg Bet Size": trades["executed_bet"].mean()
        }

    def plot_performance(self, df: pd.DataFrame, fig_path: str):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})

        ax1.plot(df.index, df["equity_curve"], color="#1f77b4", linewidth=1.8, label="Model Portfolio (Dynamic Sizing)")
        ax1.axhline(1.0, color="gray", linestyle="--", alpha=0.6)
        ax1.set_title("Out-of-Sample Performance Curve (Purged & Embargo CV)", fontsize=13, fontweight="bold")
        ax1.set_ylabel("Portfolio Value (Base = 1.0)", fontsize=11)
        ax1.grid(True, linestyle=":", alpha=0.5)
        ax1.legend(loc="upper left")

        for f in df["fold"].unique():
            fold_start = df[df["fold"] == f].index[0]
            ax1.axvline(fold_start, color="darkred", linestyle=":", alpha=0.4)
            ax1.text(fold_start, ax1.get_ylim()[1] * 0.97, f" F{f}", color="darkred", fontsize=9, fontweight="bold")

        ax2.fill_between(df.index, df["drawdown"] * 100, 0, color="#d62728", alpha=0.4, label="Underwater Drawdown")
        ax2.plot(df.index, df["drawdown"] * 100, color="#d62728", linewidth=1.0)
        ax2.set_ylabel("Drawdown (%)", fontsize=11)
        ax2.set_xlabel("Out-of-Sample Evaluation Window", fontsize=11)
        ax2.grid(True, linestyle=":", alpha=0.5)
        ax2.legend(loc="lower left")

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"[✓] Equity and Drawdown chart exported to {fig_path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[*] Reading OOS predictions: {args.predictions_path}")
    df_oos = pd.read_csv(args.predictions_path, index_col="time", parse_dates=True)

    engine = PortfolioBacktestEngine(
        fee_rate=args.fee_rate,
        slippage=args.slippage,
        risk_free_rate=args.risk_free_rate,
        annual_factor=args.annual_factor
    )

    print("[*] Resolving concurrent holding periods & applying fee frictions...")
    df_backtest = engine.resolve_concurrent_trades(df_oos)

    # Save detailed trade execution log
    log_path = os.path.join(args.output_dir, "df_backtest_execution.csv")
    df_backtest.to_csv(log_path)
    print(f"[✓] Execution log saved to {log_path}")

    # Compute fold-by-fold & overall metrics
    overall_metrics = engine.compute_metrics(df_backtest, label="Full OOS Portfolio")
    fold_metrics = []
    for f in sorted(df_backtest["fold"].unique()):
        sub_df = df_backtest[df_backtest["fold"] == f].copy()
        sub_df["equity_curve"] = (1.0 + sub_df["net_trade_pnl"]).cumprod()
        sub_df["peak_equity"] = sub_df["equity_curve"].cummax()
        sub_df["drawdown"] = (sub_df["equity_curve"] - sub_df["peak_equity"]) / sub_df["peak_equity"]
        fold_metrics.append(engine.compute_metrics(sub_df, label=f"Fold {f}"))

    summary_df = pd.DataFrame(fold_metrics + [overall_metrics]).set_index("Label")

    print("\n" + "=" * 90)
    print("FINAL PORTFOLIO AUDIT REPORT (OUT-OF-SAMPLE PnL & RISK METRICS)")
    print("=" * 90)
    print(summary_df.round(2).to_string())

    engine.plot_performance(df_backtest, args.fig_path)


if __name__ == "__main__":
    main()