"""
src/data_pipeline/loader.py

Automates historical market data ingestion across assets, audits bar integrity,
and constructs synthetic portfolio return indexes.
"""

import os
import time
from typing import Dict
import numpy as np
import pandas as pd
import yfinance as yf


class MultiAssetDataPreparer:
    """
    Downloads, audits, and caches daily OHLCV series for target portfolios and benchmarks.
    """

    def __init__(
        self,
        portfolio_path: str = "data/final_portfolio.csv",
        start_date: str = "2018-01-01",
        end_date: str = "2026-06-01",
        cache_dir: str = "data/cache",
    ):
        self.portfolio_path = portfolio_path
        self.start_date = start_date
        self.end_date = end_date
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        if not os.path.exists(self.portfolio_path):
            raise FileNotFoundError(f"Portfolio file not found: {self.portfolio_path}")

        self.df_portfolio = pd.read_csv(self.portfolio_path)
        self.tickers = sorted(self.df_portfolio["Ticker"].unique().tolist())
        print(f"[*] Initialized portfolio: {len(self.tickers)} assets loaded from {self.portfolio_path}")

    def fetch_history(self, symbol: str) -> pd.DataFrame:
        """Fetches OHLCV from cache or remote provider."""
        cache_file = os.path.join(self.cache_dir, f"{symbol}.csv")
        if os.path.exists(cache_file):
            df = pd.read_csv(cache_file)
            df["time"] = pd.to_datetime(df["time"])
            df.set_index("time", inplace=True)
            return df

        try:
            yf_symbol = "VNINDEX.QA" if symbol == "VNINDEX" else f"{symbol}.VN"
            ticker_obj = yf.Ticker(yf_symbol)
            df = ticker_obj.history(start=self.start_date, end=self.end_date, interval="1d")

            if df is not None and not df.empty:
                df.reset_index(inplace=True)
                df.rename(
                    columns={
                        "Date": "time", "Open": "open", "High": "high",
                        "Low": "low", "Close": "close", "Volume": "volume"
                    },
                    inplace=True
                )
                df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
                df.sort_values(by="time", inplace=True)
                df.set_index("time", inplace=True)
                df.to_csv(cache_file)
                return df
        except Exception as err:
            print(f"  [!] Failed to download {symbol}: {err}")

        return pd.DataFrame()

    def clean_and_adjust(self, df: pd.DataFrame, df_mkt: pd.DataFrame) -> pd.DataFrame:
        """Enforces bar boundaries and joins benchmark market returns."""
        df = df.copy()
        df = df[~df.index.duplicated(keep="last")].sort_index()

        max_oc = df[["open", "close"]].max(axis=1)
        min_oc = df[["open", "close"]].min(axis=1)
        df["high"] = np.maximum(df["high"], max_oc)
        df["low"] = np.minimum(df["low"], min_oc)
        df = df[(df["low"] > 0) & (df["volume"] >= 0)]

        if "close_adj" not in df.columns:
            df["close_adj"] = df["close"]
            ratio = df["close_adj"] / df["close"]
            df["open_adj"] = df["open"] * ratio
            df["high_adj"] = df["high"] * ratio
            df["low_adj"] = df["low"] * ratio

        df["log_return"] = np.log(df["close_adj"] / df["close_adj"].shift(1))
        df["overnight_return"] = np.log(df["open_adj"] / df["close_adj"].shift(1))
        df["intraday_return"] = np.log(df["close_adj"] / df["open_adj"])
        df["hl_log_range"] = np.log(df["high_adj"] / df["low_adj"])

        if not df_mkt.empty:
            df = df.join(df_mkt, how="left")
            df[df_mkt.columns] = df[df_mkt.columns].fillna(0.0)

        return df.dropna(subset=["close_adj", "open_adj"])

    def prepare_dataset(self) -> Dict[str, pd.DataFrame]:
        """Runs the entire ingestion loop for assets and market benchmarks."""
        print("[*] Ingesting Market Benchmark (VN-INDEX)...")
        df_mkt = self.fetch_history("VNINDEX")
        if not df_mkt.empty:
            df_mkt.index = pd.to_datetime(df_mkt.index)
            df_mkt.sort_index(inplace=True)
            df_mkt = df_mkt[["close", "volume", "high", "low"]].rename(
                columns={
                    "close": "mkt_close", "volume": "mkt_volume",
                    "high": "mkt_high", "low": "mkt_low"
                }
            )
            df_mkt["mkt_return"] = np.log(df_mkt["mkt_close"] / df_mkt["mkt_close"].shift(1))

        clean_dict = {}
        for idx, ticker in enumerate(self.tickers, 1):
            print(f"[{idx:02d}/{len(self.tickers):02d}] Processing ticker: {ticker}...")
            df_raw = self.fetch_history(ticker)
            if not df_raw.empty and len(df_raw) > 50:
                clean_df = self.clean_and_adjust(df_raw, df_mkt)
                clean_dict[ticker] = clean_df
            else:
                print(f"  [!] Skipped {ticker} due to insufficient trading bars.")
            time.sleep(1)

        print(f"\n[✓] Ingestion completed: {len(clean_dict)}/{len(self.tickers)} series ready.")
        return clean_dict