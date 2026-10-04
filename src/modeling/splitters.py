"""
src/modeling/splitters.py

Purged and Embargoed Time-Series Cross-Validation.
Eliminates look-ahead leakage and auto-correlated label contamination
caused by overlapping Triple Barrier holding horizons (López de Prado standard).
"""

from typing import Iterator, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


class TimeBasedPurgedTimeSeriesSplit:
    """
    Purged and Embargoed Cross-Validation splitter for financial time series.

    Guarantees asymptotic statistical independence between training set S_train
    and testing set S_test by strictly enforcing:
        t_touch(i) < t_start(test) - h_embargo, for all i in S_train.
    """

    def __init__(
        self,
        n_splits: int = 5,
        max_train_size: Optional[int] = None,
        embargo_td: pd.Timedelta = pd.Timedelta(days=5),
    ):
        """
        Parameters
        ----------
        n_splits : int, default=5
            Number of splits / folds.
        max_train_size : Optional[int], default=None
            Maximum number of bars in each rolling training window.
        embargo_td : pd.Timedelta, default=Timedelta(days=5)
            Safety embargo window following the last test bar.
        """
        self.n_splits = n_splits
        self.max_train_size = max_train_size
        self.embargo_td = embargo_td

    def split(
        self, df: pd.DataFrame, touch_time_col: str = "barrier_touch_time"
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generates purged and embargoed train/test integer index arrays.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset with DatetimeIndex and target barrier touch timestamps.
        touch_time_col : str, default='barrier_touch_time'
            Column containing timestamp when trade actually hit TP/SL/Timeout.

        Yields
        ------
        Iterator[Tuple[np.ndarray, np.ndarray]]
            Tuples of (train_indices, test_indices).
        """
        assert isinstance(df.index, pd.DatetimeIndex), "DataFrame index must be a DatetimeIndex."
        assert df.index.is_monotonic_increasing, "DatetimeIndex must be monotonically increasing."
        assert touch_time_col in df.columns, f"Required column '{touch_time_col}' missing from DataFrame."

        tscv = TimeSeriesSplit(n_splits=self.n_splits, max_train_size=self.max_train_size)

        touch_times = pd.to_datetime(df[touch_time_col]).values
        timestamps = df.index.values
        embargo_ns = np.timedelta64(self.embargo_td.to_numpy(), "ns")

        for raw_tr_idx, te_idx in tscv.split(df):
            if len(te_idx) == 0:
                continue

            test_start_time = timestamps[te_idx[0]]
            train_touch_limit = test_start_time - embargo_ns

            # Purge any training observations whose holding horizon overlaps the test interval
            # or infringes the safety embargo buffer
            valid_train_mask = (
                (touch_times[raw_tr_idx] < train_touch_limit)
                & (~pd.isna(touch_times[raw_tr_idx]))
            )
            tr_purged_embargoed = raw_tr_idx[valid_train_mask]

            if len(tr_purged_embargoed) > 0:
                yield tr_purged_embargoed, te_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits