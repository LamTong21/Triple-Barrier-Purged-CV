"""
tests/test_purged_cv.py

Validates that TimeBasedPurgedTimeSeriesSplit strictly eliminates look-ahead
and auto-correlation leakage across temporal folds.
Guarantees: test_start_time - max_train_touch_time > embargo_td.
"""

import numpy as np
import pandas as pd
import pytest

from src.modeling.splitters import TimeBasedPurgedTimeSeriesSplit


@pytest.fixture
def dummy_labeled_data():
    """Generates synthetic time series with overlapping barrier touch timestamps."""
    n_bars = 300
    dates = pd.date_range(start="2020-01-01", periods=n_bars, freq="B")
    
    # Simulate trade holding horizon between 1 and 5 business days
    holding_days = np.random.randint(1, 6, size=n_bars)
    touch_times = [
        dates[min(i + holding_days[i], n_bars - 1)] for i in range(n_bars)
    ]
    
    df = pd.DataFrame(
        {
            "close_adj": np.linspace(100, 150, n_bars) + np.random.randn(n_bars),
            "target_label": np.random.choice([0, 1], size=n_bars),
            "barrier_touch_time": pd.to_datetime(touch_times),
        },
        index=dates,
    )
    return df


def test_purged_cv_embargo_safety_gap(dummy_labeled_data):
    """
    Verifies that for every fold, the temporal separation between the latest
    barrier touch in train and the first bar of test exceeds the embargo threshold.
    """
    embargo_days = 5
    embargo_td = pd.Timedelta(days=embargo_days)
    splitter = TimeBasedPurgedTimeSeriesSplit(
        n_splits=5,
        max_train_size=120,
        embargo_td=embargo_td,
    )

    split_count = 0
    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(dummy_labeled_data, touch_time_col="barrier_touch_time")
    ):
        split_count += 1
        test_start_time = dummy_labeled_data.index[test_idx[0]]
        max_train_touch = dummy_labeled_data["barrier_touch_time"].iloc[train_idx].max()

        gap = test_start_time - max_train_touch
        assert gap > embargo_td, (
            f"Leakage detected in Fold {fold + 1}! "
            f"Observed safety gap {gap} <= Embargo buffer {embargo_td}."
        )

    assert split_count == 5, f"Expected 5 valid splits, got {split_count}."


def test_purged_cv_monotonicity_and_disjointness(dummy_labeled_data):
    """Verifies that test sets across folds are disjoint and monotonic."""
    splitter = TimeBasedPurgedTimeSeriesSplit(n_splits=4, embargo_td=pd.Timedelta(days=3))
    
    prev_test_max = -1
    for _, test_idx in splitter.split(dummy_labeled_data):
        assert len(test_idx) > 0, "Empty test fold generated."
        assert test_idx[0] > prev_test_max, "Test partitions must be strictly forward-chaining."
        prev_test_max = test_idx[-1]