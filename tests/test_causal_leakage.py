"""
tests/test_causal_leakage.py

Ensures Point-in-Time Shift-Lock integrity across the entire feature space.
Features at row t must strictly depend only on information up to t-1.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.routers import StatelessFeatureRouter


@pytest.fixture
def dummy_ohlcv():
    """Generates synthetic daily bar prices with volume and market benchmark."""
    n_bars = 80
    dates = pd.date_range(start="2022-01-01", periods=n_bars, freq="D")
    
    prices = 100.0 + np.cumsum(np.random.randn(n_bars))
    df = pd.DataFrame(
        {
            "open_adj": prices * (1.0 + np.random.uniform(-0.01, 0.01, n_bars)),
            "high_adj": prices * (1.0 + np.random.uniform(0.01, 0.03, n_bars)),
            "low_adj": prices * (1.0 - np.random.uniform(0.01, 0.03, n_bars)),
            "close_adj": prices,
            "volume": np.random.randint(1000, 5000, n_bars).astype(float),
            "mkt_return": np.random.normal(0, 0.01, n_bars),
        },
        index=dates,
    )
    df["log_return"] = np.log(df["close_adj"] / df["close_adj"].shift(1))
    return df


def test_stateless_router_shift_lock_prevents_contemporaneous_leakage(dummy_ohlcv):
    """
    Perturbs data at row t and verifies that features at row t are unchanged,
    proving that features at t do NOT read bar t's price shocks.
    """
    router = StatelessFeatureRouter(default_lags=[1, 3], enforce_shift=True)
    
    # 1. Transform original data
    df_transformed_orig = router.execute(dummy_ohlcv.copy())
    feature_cols = [c for c in df_transformed_orig.columns if c not in dummy_ohlcv.columns]
    assert len(feature_cols) > 0, "No features were generated."

    # 2. Inject shock at bar index t=50
    perturbed_df = dummy_ohlcv.copy()
    shock_idx = 50
    perturbed_df.iloc[shock_idx, perturbed_df.columns.get_loc("close_adj")] *= 2.0
    perturbed_df.iloc[shock_idx, perturbed_df.columns.get_loc("volume")] *= 10.0

    df_transformed_shocked = router.execute(perturbed_df)

    # 3. Features at index t must be strictly identical between original and shocked runs
    # because row t features must only see information through t-1
    orig_vals_at_t = df_transformed_orig.loc[dummy_ohlcv.index[shock_idx], feature_cols]
    shocked_vals_at_t = df_transformed_shocked.loc[dummy_ohlcv.index[shock_idx], feature_cols]

    pd.testing.assert_series_equal(
        orig_vals_at_t,
        shocked_vals_at_t,
        check_exact=False,
        rtol=1e-6,
        obj="Contemporaneous Leakage Check: Features at bar t altered by shocks at bar t!",
    )


def test_first_row_must_be_nan_due_to_shift_lock(dummy_ohlcv):
    """Verifies that the very first observation of shifted features is NaN."""
    router = StatelessFeatureRouter(default_lags=[1, 3], enforce_shift=True)
    df_out = router.execute(dummy_ohlcv.copy())
    feature_cols = [c for c in df_out.columns if c not in dummy_ohlcv.columns]

    first_row_features = df_out.iloc[0][feature_cols]
    assert first_row_features.isna().all(), "First row of shifted features must be NaN."