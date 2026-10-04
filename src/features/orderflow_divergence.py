"""
src/features/orderflow_divergence.py

Volume-price momentum divergence, directional pressure, and exogenous market index alpha.
All indicators enforce strictly causal rolling windows without global look-ahead cumsum.
"""

import numpy as np
import pandas as pd

from src.features.base_strategy import BaseFeatureStrategy


class VolumePriceDivergenceStrategy(BaseFeatureStrategy):
    """
    Computes rolling Chaikin Money Flow (CMF), rolling OBV slope, and VPT divergence.
    """

    def __init__(self, window: int = 20):
        self.window = window

    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        h, l, c = df["high_adj"], df["low_adj"], df["close_adj"]
        v = df["volume"]
        r = df["log_return"].fillna(0.0)

        # 1. Rolling CMF (Leak-free)
        clv = ((c - l) - (h - c)) / ((h - l) + eps)
        vol_money_flow = clv * v
        feat["cmf_20"] = vol_money_flow.rolling(self.window).sum() / (v.rolling(self.window).sum() + eps)
        feat["cmf_slope_5"] = feat["cmf_20"] - feat["cmf_20"].shift(5)

        # 2. Rolling On-Balance Volume (OBV) Momentum
        obv_flow = np.sign(r) * v
        roll_obv = obv_flow.rolling(self.window).sum()
        roll_obv_std = obv_flow.rolling(self.window).std(ddof=1) + eps
        feat["obv_zscore_20"] = roll_obv / roll_obv_std
        feat["obv_slope_5"] = (roll_obv - obv_flow.shift(5).rolling(self.window).sum()) / (
            v.rolling(self.window).mean() + eps
        )

        # 3. Rolling Volume-Price Trend (VPT)
        vpt_flow = r * v
        roll_vpt = vpt_flow.rolling(self.window).sum()
        roll_vpt_std = vpt_flow.rolling(self.window).std(ddof=1) + eps
        feat["vpt_zscore_20"] = roll_vpt / roll_vpt_std

        # 4. Direct CMF vs Price Return divergence
        norm_price_ret = (c / c.shift(self.window)) - 1.0
        feat["cmf_price_divergence"] = feat["cmf_20"] - norm_price_ret

        return feat


class DirectionalPressureStrategy(BaseFeatureStrategy):
    """
    Computes buying vs selling pressure and body directional momentum.
    """

    def __init__(self, window: int = 14):
        self.window = window

    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        h, l, c, o = df["high_adj"], df["low_adj"], df["close_adj"], df["open_adj"]
        v = df["volume"]

        candle_range = (h - l) + eps
        upper_wick = h - np.maximum(o, c)
        lower_wick = np.minimum(o, c) - l
        body = c - o

        buy_strength = (c - l) / candle_range
        sell_strength = (h - c) / candle_range
        feat["net_pressure_ratio"] = buy_strength - sell_strength
        feat["net_pressure_volume"] = feat["net_pressure_ratio"] * np.log(v + 1.0)
        feat["net_pressure_vol_roll"] = feat["net_pressure_volume"].rolling(self.window).mean()

        feat["wick_absorption_direction"] = (lower_wick - upper_wick) / candle_range
        feat["wick_absorption_thrust"] = feat["wick_absorption_direction"] * (
            v / (v.rolling(self.window).mean() + eps)
        )

        body_ratio = body / candle_range
        feat["body_direction_momentum"] = body_ratio.rolling(5).mean()
        return feat


class SignedVolatilityDivergenceStrategy(BaseFeatureStrategy):
    """
    Measures asymmetric upside vs downside semi-variances and directional expansion thrust.
    """

    def __init__(self, window: int = 20):
        self.window = window

    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        r = df["log_return"].fillna(0.0)

        downside_ret = r.where(r < 0, 0.0)
        upside_ret = r.where(r > 0, 0.0)

        downside_var = (downside_ret ** 2).rolling(self.window).mean()
        upside_var = (upside_ret ** 2).rolling(self.window).mean()

        feat["vol_directional_bias"] = (upside_var - downside_var) / (upside_var + downside_var + eps)

        c, o = df["close_adj"], df["open_adj"]
        signed_body = np.sign(c - o)
        daily_range = np.log(df["high_adj"] / df["low_adj"])
        feat["signed_expansion_thrust"] = signed_body * daily_range

        return feat


class MarketContextStrategy(BaseFeatureStrategy):
    """
    Generates alpha and beta metrics relative to the benchmark index (VN-Index).
    Strictly enforces a 1-bar lag on market inputs to eliminate look-ahead leakage.
    """

    def __init__(self, window_short: int = 20, window_long: int = 60):
        self.ws = window_short
        self.wl = window_long

    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        if "mkt_return" not in df.columns:
            return feat

        r_asset = df["log_return"].fillna(0.0)
        r_mkt_lag = df["mkt_return"].shift(1).fillna(0.0)

        feat["mkt_relative_return"] = r_asset - r_mkt_lag
        feat["mkt_rs_cum_20"] = r_asset.rolling(self.ws).sum() - r_mkt_lag.rolling(self.ws).sum()
        feat["mkt_rs_momentum"] = feat["mkt_rs_cum_20"] - feat["mkt_rs_cum_20"].shift(5)

        cov_60 = r_asset.rolling(self.wl).cov(r_mkt_lag)
        var_mkt_60 = r_mkt_lag.rolling(self.wl).var()
        feat["mkt_rolling_beta_60"] = (cov_60 / (var_mkt_60 + eps)).clip(lower=-1.0, upper=3.5)
        feat["mkt_beta_shock"] = feat["mkt_rolling_beta_60"] - feat["mkt_rolling_beta_60"].shift(5)

        vol_asset_20 = r_asset.rolling(self.ws).std(ddof=1)
        vol_mkt_20 = r_mkt_lag.rolling(self.ws).std(ddof=1)
        feat["mkt_vol_ratio_20"] = vol_asset_20 / (vol_mkt_20 + eps)
        feat["mkt_correlation_20"] = r_asset.rolling(self.ws).corr(r_mkt_lag).fillna(0.0)
        feat["mkt_divergence_signed"] = np.sign(r_asset) * np.sign(r_mkt_lag) * (r_asset - r_mkt_lag)

        return feat