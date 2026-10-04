"""
src/backtest/__init__.py

Portfolio backtesting, causal trade execution, and financial risk metrics:
- ConcurrentTradeResolver: Resolves overlapping holding horizons.
- PortfolioBacktestEngine: Audits PnL incorporating two-way frictions and sizing.
- FinancialRiskMetrics: Evaluates Sharpe, Sortino, Calmar, and Win/Loss asymmetry.
"""

from src.backtest.execution import ConcurrentTradeResolver
from src.backtest.metrics import FinancialRiskMetrics
from src.backtest.engine import PortfolioBacktestEngine

__all__ = [
    "ConcurrentTradeResolver",
    "FinancialRiskMetrics",
    "PortfolioBacktestEngine",
]