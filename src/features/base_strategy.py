"""
src/features/base_strategy.py

Abstract base class defining the contract for all modular feature construction strategies.
"""

from abc import ABC, abstractmethod
import pandas as pd


class BaseFeatureStrategy(ABC):
    """
    Abstract base strategy. Each feature module generates a set of features
    derived from input OHLCV or L2 order book data.
    """

    @abstractmethod
    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        """
        Constructs and returns transformed feature columns.

        Parameters
        ----------
        df : pd.DataFrame
            Market data containing prices, volumes, and auxiliary series.
        eps : float, default=1e-8
            Small epsilon constant to prevent zero-division errors.

        Returns
        -------
        pd.DataFrame
            DataFrame of newly engineered features with matching DatetimeIndex.
        """
        pass