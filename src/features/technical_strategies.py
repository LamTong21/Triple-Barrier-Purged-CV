"""
src/features/technical_strategies.py

Standard technical, volatility, fractional memory, and kinematic feature strategies.
NOTE: Higher-order derivatives (acceleration/jerk) are purged to prevent variance inflation.
"""

from typing import List, Sequence
import numpy as np
import pandas as pd

from src.features.base_strategy import BaseFeatureStrategy


class BaselineStrategy(BaseFeatureStrategy):
    """
    Computes candle geometry ratios, overnight gaps, short-horizon relative volume,
    and cyclical trigonometric calendar encodings.
    """

    def __init__(self, dynamic_lags: Sequence[int]):
        # Keep only short lags (<= 3) to minimize over-parameterization
        self.lags = sorted(list({k for k in dynamic_lags if 1 <= k <= 3} or {1, 2, 3}))

    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        h, l, c, o = df["high_adj"], df["low_adj"], df["close_adj"], df["open_adj"]
        v = df["volume"]

        candle_range = (h - l) + eps
        feat["geo_body_ratio"] = (c - o) / candle_range
        feat["geo_upper_shadow"] = (h - np.maximum(o, c)) / candle_range
        feat["geo_lower_shadow"] = (np.minimum(o, c) - l) / candle_range
        feat["overnight_gap"] = (o / c.shift(1)) - 1.0

        log_v = np.log(v + eps)
        for k in self.lags:
            roll_vol_mean = v.shift(1).rolling(k * 5).mean()
            feat[f"rvol_{k * 5}"] = v / (roll_vol_mean + eps)
            mu_lv = log_v.shift(1).rolling(k * 5).mean()
            std_lv = log_v.shift(1).rolling(k * 5).std(ddof=1)
            feat[f"vol_shock_{k * 5}"] = (log_v - mu_lv) / (std_lv + eps)

        if isinstance(df.index, pd.DatetimeIndex):
            dow = df.index.dayofweek
            month = df.index.month
            feat["sin_dow"] = np.sin(2 * np.pi * dow / 5.0)
            feat["cos_dow"] = np.cos(2 * np.pi * dow / 5.0)
            feat["sin_month"] = np.sin(2 * np.pi * month / 12.0)
            feat["cos_month"] = np.cos(2 * np.pi * month / 12.0)

        return feat


class TrendMomentumStrategy(BaseFeatureStrategy):
    """
    Computes volatility-standardized multi-horizon momentum and MACD divergence.
    """

    def __init__(self, dynamic_lags: Sequence[int]):
        self.lags = sorted(list({k for k in dynamic_lags if 1 <= k <= 3} or {1, 2, 3}))

    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        p = df["close_adj"]
        vol_20 = df["log_return"].shift(1).rolling(20).std(ddof=1) + eps

        for k in self.lags:
            ret_k = np.log(p / p.shift(k))
            feat[f"ret_{k}"] = ret_k
            feat[f"vol_adj_mom_{k}"] = ret_k / (vol_20 * np.sqrt(k))

        ema_12 = p.ewm(span=12, adjust=False).mean()
        ema_26 = p.ewm(span=26, adjust=False).mean()
        feat["macd_dist"] = (ema_12 / ema_26) - 1.0
        return feat


class OscillatorMeanReversionStrategy(BaseFeatureStrategy):
    """
    Standard mean-reversion metrics: Z-Scores, Bollinger %B, and Wilder's RSI.
    """

    def __init__(self, dynamic_lags: Sequence[int]):
        self.lags = [k for k in dynamic_lags if k <= 20] or [5, 10, 20]

    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        p = df["close_adj"]

        for k in self.lags:
            if k >= 3:
                roll_mean = p.shift(1).rolling(k).mean()
                roll_std = p.shift(1).rolling(k).std(ddof=1)
                feat[f"zscore_{k}"] = (p - roll_mean) / (roll_std + eps)
                upper_band = roll_mean + (2.0 * roll_std)
                lower_band = roll_mean - (2.0 * roll_std)
                feat[f"bollinger_pctB_{k}"] = (p - lower_band) / ((upper_band - lower_band) + eps)

        delta = p.diff()
        gain = delta.where(delta > 0, 0.0).shift(1).rolling(14).mean()
        loss = -delta.where(delta < 0, 0.0).shift(1).rolling(14).mean()
        rs = gain / (loss + eps)
        feat["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
        return feat


class FractionalMemoryStrategy(BaseFeatureStrategy):
    """
    Fractional differentiation memory operator evaluated over rolling lookback windows.
    """

    def __init__(self, optimal_d: float, threshold: float = 1e-4, max_window: int = 100):
        self.d = optimal_d
        self.threshold = threshold
        self.max_window = max_window
        self.weights = self._compute_weights(max_window)
        self.actual_max_len = len(self.weights)

    def _compute_weights(self, size: int) -> np.ndarray:
        w = [1.0]
        for k in range(1, size):
            w_k = -w[-1] / k * (self.d - k + 1)
            if abs(w_k) < self.threshold:
                break
            w.append(w_k)
        return np.array(w)

    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        p = df["close_adj"]

        def apply_frac_diff(x):
            x_rev = x[::-1]
            valid_len = min(len(x_rev), self.actual_max_len)
            return np.dot(self.weights[:valid_len], x_rev[:valid_len])

        feat[f"frac_diff_d{self.d:.2f}"] = p.rolling(self.max_window, min_periods=10).apply(apply_frac_diff, raw=True)
        return feat


class VolatilityDynamicsStrategy(BaseFeatureStrategy):
    """
    Estimates intraday Parkinson, Garman-Klass, Rogers-Satchell, and Yang-Zhang volatilities.
    """

    def __init__(self, dynamic_lags: Sequence[int], has_asymmetric_vol: bool):
        self.lags = dynamic_lags
        self.has_asymmetric = has_asymmetric_vol

    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        h, l, c, o = df["high_adj"], df["low_adj"], df["close_adj"], df["open_adj"]
        r = df["log_return"]

        feat["vol_parkinson"] = np.sqrt((1.0 / (4.0 * np.log(2.0))) * (np.log(h / l) ** 2))
        log_hl = np.log(h / l)
        log_co = np.log(c / o)
        gk_core = 0.5 * (log_hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (log_co ** 2)
        feat["vol_garman_klass"] = np.sqrt(np.maximum(0.0, gk_core))
        rs_core = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
        feat["vol_rogers_satchell"] = np.sqrt(np.maximum(0.0, rs_core))

        window = 20
        o_prev_c = np.log(o / c.shift(1))
        c_prev_o = np.log(c / o)
        var_o = o_prev_c.rolling(window).var()
        var_c = c_prev_o.rolling(window).var()
        var_rs = pd.Series(rs_core, index=df.index).rolling(window).mean()
        k_val = 0.34 / (1.34 + (window + 1.0) / (window - 1.0))
        feat["vol_yang_zhang"] = np.sqrt(np.maximum(0.0, var_o + k_val * var_c + (1.0 - k_val) * var_rs))

        if self.has_asymmetric:
            downside_ret = r.where(r < 0, 0.0)
            feat["vol_downside_dev_20"] = downside_ret.rolling(20).std(ddof=1)
            feat["vol_upside_dev_20"] = r.where(r > 0, 0.0).rolling(20).std(ddof=1)
            feat["vol_semi_variance_ratio_raw"] = feat["vol_downside_dev_20"] / (feat["vol_upside_dev_20"] + eps)

        feat["skewness_20"] = r.rolling(20).skew()
        feat["kurtosis_20"] = r.rolling(20).kurt()
        return feat


class VolatilityJumpDiffusionStrategy(BaseFeatureStrategy):
    """
    Extracts continuous vs jump diffusion components using Bipower Variation.
    """

    def __init__(self, window: int = 20):
        self.window = window

    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        r = df["log_return"].fillna(0.0)

        rv = (r ** 2).rolling(self.window).sum()
        abs_r = r.abs()
        bv = (np.pi / 2.0) * (abs_r * abs_r.shift(1)).rolling(self.window).sum()

        feat["jump_diffusion_component"] = np.maximum(rv - bv, 0.0) / (rv + eps)
        feat["continuous_vol_ratio"] = bv / (rv + eps)
        feat["jump_signed_shock"] = feat["jump_diffusion_component"] * np.sign(r)
        return feat


class MultiScaleComplexityStrategy(BaseFeatureStrategy):
    """
    Measures dynamical chaos and algorithmic complexity over short rolling windows.
    """

    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        r = df["log_return"].fillna(0.0)

        def calc_pe(x):
            if len(x) < 6:
                return np.nan
            m, tau = 3, 1
            n = len(x) - (m - 1) * tau
            patterns = np.array([x[i : i + m * tau : tau] for i in range(n)])
            ranks = np.argsort(patterns, axis=1)
            _, counts = np.unique(ranks, axis=0, return_counts=True)
            probs = counts / counts.sum()
            return -np.sum(probs * np.log2(probs + 1e-8)) / np.log2(6.0)

        feat["chaos_permutation_entropy_20"] = r.rolling(20).apply(calc_pe, raw=True)
        binary_seq = (r > 0).astype(int)
        feat["lz_complexity_proxy_20"] = (binary_seq.diff().abs()).rolling(20).mean()
        return feat


class WaveletMultiResolutionStrategy(BaseFeatureStrategy):
    """
    Haar wavelet decomposition proxy extracting multi-scale energy ratios.
    """

    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        p = df["close_adj"]

        p_s1 = p.shift(1)
        feat["wavelet_detail_d1"] = (p - p_s1) / np.sqrt(2.0)
        feat["wavelet_detail_d2"] = ((p + p_s1) - (p.shift(2) + p.shift(3))) / 2.0
        feat["wavelet_approx_a3"] = p.rolling(8).mean()
        feat["wavelet_energy_ratio"] = (feat["wavelet_detail_d1"] ** 2) / ((feat["wavelet_detail_d2"] ** 2) + eps)
        return feat


class IntradayShadowPressureStrategy(BaseFeatureStrategy):
    """
    Extracts candlestick shadow asymmetry and volume-weighted tail absorption.
    """

    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        h, l, c, o = df["high_adj"], df["low_adj"], df["close_adj"], df["open_adj"]
        v = df["volume"]

        candle_range = (h - l) + eps
        body = (c - o).abs()
        upper_shadow = h - np.maximum(o, c)
        lower_shadow = np.minimum(o, c) - l

        feat["shadow_asymmetry_ratio"] = (upper_shadow - lower_shadow) / candle_range
        feat["buying_tail_power"] = (lower_shadow / candle_range) * v
        feat["selling_tail_power"] = (upper_shadow / candle_range) * v
        feat["body_efficiency_ratio"] = body / candle_range
        return feat


class KinematicDynamicsStrategy(BaseFeatureStrategy):
    """
    Extracts price velocity and volatility squeeze.
    Acceleration and jerk are purged to eliminate white noise amplification.
    """

    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        p = df["close_adj"]
        h, l = df["high_adj"], df["low_adj"]
        v = df["volume"]
        r = df["log_return"].fillna(0.0)

        # 1. First-order velocity
        feat["kinematic_velocity"] = r

        # 2. Bollinger Band vs Keltner Channel squeeze ratio
        roll_std = p.shift(1).rolling(20).std(ddof=1)
        bb_width = 4.0 * roll_std

        p_prev = p.shift(1)
        tr = pd.concat([h - l, (h - p_prev).abs(), (l - p_prev).abs()], axis=1).max(axis=1)
        atr = tr.shift(1).rolling(20).mean()
        kc_width = 3.0 * atr

        feat["kinematic_squeeze_ratio"] = bb_width / (kc_width + eps)

        # 3. Close Location Value weighted by relative volume
        clv = ((p - l) - (h - p)) / (h - l + eps)
        feat["kinematic_clv_vol"] = clv * (v / (v.shift(1).rolling(20).mean() + eps))
        return feat