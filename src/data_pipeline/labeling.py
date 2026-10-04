"""
src/data_pipeline/labeling.py

Long-Only Asymmetric Triple Barrier Meta-Labeling Engine.
Guarantees causal volatility estimation to prevent future leakage across validation folds.
"""

import numpy as np
import pandas as pd


class Layer0TripleBarrier:
    """
    Constructs dynamic Triple Barrier labels:
    - Upper Barrier (Take-Profit): P0 * (1 + pt * sigma)
    - Lower Barrier (Stop-Loss): P0 * (1 - sl * sigma)
    - Vertical Barrier (Time-Out): h bars
    
    Target logic:
    - 1.0: Hits Upper Barrier first.
    - 0.0: Hits Lower Barrier first OR reaches Vertical Barrier (Time-out).
    """

    @staticmethod
    def label(
        df: pd.DataFrame,
        h: int = 5,
        pt: float = 1.0,
        sl: float = 1.0,
        vol_span: int = 20,
        vol_floor: float = 1e-4
    ) -> pd.DataFrame:
        """
        Generates causal labels and timestamps of barrier touches.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 'close_adj' and 'log_return'.
        h : int, default=5
            Vertical barrier holding horizon (bars).
        pt : float, default=1.0
            Take-profit factor.
        sl : float, default=1.0
            Stop-loss factor.
        vol_span : int, default=20
            EWMA window span for local volatility.
        vol_floor : float, default=1e-4
            Minimum allowable volatility floor.

        Returns
        -------
        pd.DataFrame
            DataFrame with 'target_label', 'target_barrier_return', and 'barrier_touch_time'.
        """
        df = df.copy()

        # Causal Volatility: Strictly lagged by 1 bar to prevent forward-looking bias
        ewm_vol = df["log_return"].ewm(span=vol_span, min_periods=2).std().shift(1)
        sigma = ewm_vol.fillna(vol_floor).clip(lower=vol_floor)

        prices = df["close_adj"].values
        sigmas = sigma.values
        timestamps = df.index
        n = len(df)

        meta_target = np.full(n, np.nan)
        ret_barrier = np.full(n, np.nan)
        touch_time = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")

        for i in range(n - h):
            p0 = prices[i]
            if pd.isna(p0) or pd.isna(sigmas[i]):
                continue

            upper_barrier = p0 * (1.0 + pt * sigmas[i])
            lower_barrier = p0 * (1.0 - sl * sigmas[i])

            window_prices = prices[i + 1 : i + h + 1]
            hit_upper = np.where(window_prices >= upper_barrier)[0]
            hit_lower = np.where(window_prices <= lower_barrier)[0]

            first_upper = hit_upper[0] if len(hit_upper) > 0 else np.inf
            first_lower = hit_lower[0] if len(hit_lower) > 0 else np.inf

            if first_upper < first_lower and first_upper < np.inf:
                # Scenario 1: Reached Take-Profit
                idx_hit = int(first_upper)
                meta_target[i] = 1.0
                ret_barrier[i] = (window_prices[idx_hit] / p0) - 1.0
                touch_time[i] = timestamps[i + 1 + idx_hit]
            elif first_lower < first_upper and first_lower < np.inf:
                # Scenario 2: Reached Stop-Loss
                idx_hit = int(first_lower)
                meta_target[i] = 0.0
                ret_barrier[i] = (window_prices[idx_hit] / p0) - 1.0
                touch_time[i] = timestamps[i + 1 + idx_hit]
            else:
                # Scenario 3: Time-Out (Vertical Barrier Hit)
                meta_target[i] = 0.0
                ret_barrier[i] = (window_prices[-1] / p0) - 1.0
                touch_time[i] = timestamps[i + h]

        df["local_volatility"] = sigma
        df["target_label"] = meta_target
        df["target_barrier_return"] = ret_barrier
        df["barrier_touch_time"] = pd.to_datetime(touch_time)

        return df