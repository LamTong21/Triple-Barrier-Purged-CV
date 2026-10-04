"""
src/modeling/thresholding.py

Pure Rolling Percentile Thresholding and Dynamic Bet Sizing Engine.
Eliminates distribution mismatch across regime shifts and scales allocation via local Z-scores.
"""

from typing import Tuple
import numpy as np
import pandas as pd


class DynamicThresholdEngine:
    """
    Computes rolling percentile entry thresholds and Z-score bet sizes.

    Formulas:
        Threshold_t = Percentile_Q({p_tau}_{tau=t-W}^{t-1})
        Signal_t = 1 if p_t >= Threshold_t else 0
        Z_t = (p_t - mu_{p, W}) / sigma_{p, W}
        BetSize_t = clip(0.20 + (Z_t - 0.5) * (0.80 / 1.50), 0.20, 1.0)
    """

    def __init__(self, rolling_window: int = 60, enter_percentile: float = 65.0):
        self.rolling_window = rolling_window
        self.enter_percentile = enter_percentile

    def compute_threshold_and_signals(
        self,
        train_probs: pd.Series,
        test_probs: pd.Series,
    ) -> Tuple[pd.Series, np.ndarray, np.ndarray]:
        """
        Computes dynamic rolling threshold and generates binary trade decisions and bet sizes.

        Parameters
        ----------
        train_probs : pd.Series
            Predicted probability series from the training set.
        test_probs : pd.Series
            Predicted probability series on the test set.

        Returns
        -------
        Tuple[pd.Series, np.ndarray, np.ndarray]
            - oos_dynamic_threshold: Dynamic threshold series for the test window.
            - y_pred_binary: Binary trade decisions (0 or 1).
            - bet_sizes: Dynamic position sizing array.
        """
        # Concatenate lookback buffer from training probabilities to avoid cold start
        probs_extended = pd.concat([
            train_probs.iloc[-self.rolling_window:],
            test_probs,
        ]).sort_index()

        # Dynamic rolling thresholding
        rolling_threshold_series = (
            probs_extended
            .rolling(window=self.rolling_window, min_periods=self.rolling_window // 2)
            .apply(lambda w: np.percentile(w, self.enter_percentile), raw=True)
        )
        oos_dyn_threshold = rolling_threshold_series.loc[test_probs.index]

        # Binary entry signal
        y_pred_binary = (test_probs >= oos_dyn_threshold).astype(int).values

        # Rolling mean and standard deviation for local Z-score calculation
        rolling_mean = (
            probs_extended
            .rolling(window=self.rolling_window, min_periods=self.rolling_window // 2)
            .mean()
            .loc[test_probs.index]
        )
        rolling_std = (
            probs_extended
            .rolling(window=self.rolling_window, min_periods=self.rolling_window // 2)
            .std(ddof=0)
            .loc[test_probs.index]
            + 1e-8
        )

        prob_zscores = ((test_probs - rolling_mean) / rolling_std).values

        # Dynamic Bet Sizing (clamped between 20% and 100%)
        bet_sizes = np.where(
            y_pred_binary == 1,
            np.clip(0.20 + (prob_zscores - 0.5) * (0.80 / 1.50), 0.20, 1.0),
            0.0,
        )

        return oos_dyn_threshold, y_pred_binary, bet_sizes