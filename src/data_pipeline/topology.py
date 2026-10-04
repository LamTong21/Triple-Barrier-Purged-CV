"""
src/data_pipeline/topology.py

Expands financial state space into structural topologies:
- OHLCV geometric logarithmic ranges.
- Multi-Level L2 Order Book Imbalance (OBI) with exponential decay weighting.
- Depth-weighted Micro-Price and direction deviation vectors.
"""

import numpy as np
import pandas as pd


class Layer2MicrostructureTopologies:
    """
    Computes standard bar dynamics and multi-level Limit Order Book (L2) topologies.
    """

    @staticmethod
    def compute(df: pd.DataFrame, max_depth_levels: int = 10) -> pd.DataFrame:
        """
        Transforms clean market data into rich topological feature arrays.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned market data containing adjusted prices and order book feeds.
        max_depth_levels : int, default=10
            Maximum L2 order book levels to scan.

        Returns
        -------
        pd.DataFrame
            DataFrame augmented with topological variables.
        """
        df = df.copy()

        # 1. Standard OHLCV Topologies
        df["log_return"] = np.log(df["close_adj"] / df["close_adj"].shift(1))
        df["overnight_return"] = np.log(df["open_adj"] / df["close_adj"].shift(1))
        df["intraday_return"] = np.log(df["close_adj"] / df["open_adj"])
        df["hl_log_range"] = np.log(df["high_adj"] / df["low_adj"])

        # 2. Multi-Level L2 Order Book Topology
        available_levels = [
            i for i in range(1, max_depth_levels + 1)
            if f"bid_price_{i}" in df.columns and f"ask_price_{i}" in df.columns
        ]

        if len(available_levels) > 1:
            df["mid_price"] = (df["bid_price_1"] + df["ask_price_1"]) / 2.0
            df["l2_spread"] = df["ask_price_1"] - df["bid_price_1"]

            total_weighted_bid_size = np.zeros(len(df))
            total_weighted_ask_size = np.zeros(len(df))
            cross_weighted_bid_price = np.zeros(len(df))
            cross_weighted_ask_price = np.zeros(len(df))

            # Apply exponential decay weight to penalize spoofing at distant levels
            for i in available_levels:
                decay_weight = np.exp(-(i - 1) * 0.5)

                b_size_w = df[f"bid_size_{i}"] * decay_weight
                a_size_w = df[f"ask_size_{i}"] * decay_weight

                total_weighted_bid_size += b_size_w
                total_weighted_ask_size += a_size_w

                cross_weighted_bid_price += df[f"bid_price_{i}"] * b_size_w
                cross_weighted_ask_price += df[f"ask_price_{i}"] * a_size_w

            imbalance_denom = total_weighted_bid_size + total_weighted_ask_size + 1e-8

            # Depth Order Book Imbalance (OBI)
            df["order_imbalance_depth"] = (total_weighted_bid_size - total_weighted_ask_size) / imbalance_denom

            # Depth Micro-Price cross-weighted by opposite queue sizes
            df["micro_price_depth"] = (
                (cross_weighted_bid_price / (total_weighted_bid_size + 1e-8)) * total_weighted_ask_size +
                (cross_weighted_ask_price / (total_weighted_ask_size + 1e-8)) * total_weighted_bid_size
            ) / imbalance_denom

            df["micro_mid_deviation"] = df["micro_price_depth"] - df["mid_price"]

        # Fallback: Top-of-Book (L1) Microstructure
        elif all(col in df.columns for col in ["best_bid", "best_ask", "bid_size", "ask_size"]):
            df["mid_price"] = (df["best_bid"] + df["best_ask"]) / 2.0
            imbalance_denom = df["bid_size"] + df["ask_size"] + 1e-8

            df["micro_price"] = (
                df["best_bid"] * df["ask_size"] + df["best_ask"] * df["bid_size"]
            ) / imbalance_denom
            df["l2_spread"] = df["best_ask"] - df["best_bid"]
            df["micro_mid_deviation"] = df["micro_price"] - df["mid_price"]
            df["order_imbalance_depth"] = (df["bid_size"] - df["ask_size"]) / imbalance_denom

        return df