"""
src/features/microstructure_strategies.py

Limit Order Book (L2), Order Flow Toxicity (VPIN), and Corwin-Schultz / Amihud liquidity features.
"""

import numpy as np
import pandas as pd

from src.features.base_strategy import BaseFeatureStrategy


class L2MicrostructureStrategy(BaseFeatureStrategy):
    """
    Extracts order book dynamics: micro-price deviation, spread volatility,
    and depth replenishment / cancellation proxies.
    """

    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)

        if "micro_mid_deviation" in df.columns:
            feat["l2_micro_dev_raw"] = df["micro_mid_deviation"]
            feat["l2_micro_dev_diff"] = feat["l2_micro_dev_raw"] - feat["l2_micro_dev_raw"].shift(1)

            if "l2_spread" in df.columns:
                spread = df["l2_spread"]
                feat["l2_spread_zscore"] = (spread - spread.rolling(20).mean()) / (spread.rolling(20).std() + eps)

            b_size = df.get("bid_size_1", df.get("bid_size", pd.Series(0, index=df.index)))
            a_size = df.get("ask_size_1", df.get("ask_size", pd.Series(0, index=df.index)))

            if not (b_size.sum() == 0 and a_size.sum() == 0):
                imbalance = b_size - a_size
                feat["l2_order_imbalance_raw"] = imbalance
                feat["l2_order_imbalance_z"] = (imbalance - imbalance.rolling(20).mean()) / (imbalance.rolling(20).std() + eps)
                feat["l2_depth_total"] = b_size + a_size
                feat["l2_depth_ratio"] = b_size / (a_size + eps)

                delta_b = b_size.diff().fillna(0)
                delta_a = a_size.diff().fillna(0)
                feat["l2_bid_replenishment"] = np.where(delta_b > 0, delta_b, 0)
                feat["l2_ask_replenishment"] = np.where(delta_a > 0, delta_a, 0)
                feat["l2_bid_withdrawal"] = np.where(delta_b < 0, -delta_b, 0)
                feat["l2_ask_withdrawal"] = np.where(delta_a < 0, -delta_a, 0)

        return feat


class OrderFlowToxicityStrategy(BaseFeatureStrategy):
    """
    Computes Order Flow Imbalance (OFI) and Volume-Synchronized Probability of Toxicity (VPIN).
    """

    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)

        if all(col in df.columns for col in ["bid_size_1", "ask_size_1"]):
            order_imbalance = df["bid_size_1"] - df["ask_size_1"]
        elif all(col in df.columns for col in ["bid_size", "ask_size"]):
            order_imbalance = df["bid_size"] - df["ask_size"]
        else:
            c, o, h, l = df["close_adj"], df["open_adj"], df["high_adj"], df["low_adj"]
            v = df["volume"]
            buy_pressure = (c - l) / (h - l + eps)
            sell_pressure = (h - c) / (h - l + eps)
            order_imbalance = (buy_pressure - sell_pressure) * v

        feat["flow_imbalance_proxy"] = order_imbalance
        feat["flow_imbalance_zscore"] = (order_imbalance - order_imbalance.rolling(20).mean()) / (
            order_imbalance.rolling(20).std() + eps
        )

        v_series = df["volume"] if "volume" in df.columns else pd.Series(1, index=df.index)
        vol_bucket = v_series.rolling(10).sum()
        imbalance_bucket = order_imbalance.abs().rolling(10).sum()
        feat["flow_vpin_10"] = imbalance_bucket / (vol_bucket + eps)

        return feat


class LiquidityMicrostructureStrategy(BaseFeatureStrategy):
    """
    Computes Corwin-Schultz bid-ask spreads and bounded Rolling Percentile Amihud Illiquidity.
    """

    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        h, l, c = df["high"], df["low"], df["close"]
        v = df["volume"]
        r = df["log_return"]

        # Corwin-Schultz Spread
        h_prev, l_prev = h.shift(1), l.shift(1)
        gamma = (np.log(h / l) ** 2) + (np.log(h_prev / l_prev) ** 2)
        h_2d = np.maximum(h, h_prev)
        l_2d = np.minimum(l, l_prev)
        beta = np.log(h_2d / l_2d) ** 2
        den = 3.0 - 2.0 * np.sqrt(2.0)
        alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / den - np.sqrt(gamma / den)
        spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
        feat["liq_spread_cs"] = spread.clip(lower=0.0)

        # Amihud Illiquidity bounded in [0, 1] using rolling percentile rank
        dollar_volume = c * v
        amihud_raw = r.abs() / (dollar_volume.replace(0, np.nan) + eps)
        feat["liq_amihud_raw"] = amihud_raw
        feat["liq_amihud_pct_rank"] = (
            amihud_raw.rolling(20)
            .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) == 20 else 0.5, raw=False)
        )
        feat["liq_turnover_ratio_20"] = dollar_volume / (dollar_volume.shift(1).rolling(20).mean() + eps)

        return feat