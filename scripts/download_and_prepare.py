#!/usr/bin/env python3
"""
scripts/download_and_prepare.py

Fetches historical market data, audits bar geometry, calculates causal returns,
labels dataset using Causal Triple Barrier method, and saves cleaned datasets ready for CV.
"""

import argparse
import os
import sys
import pickle
import time
import numpy as np
import pandas as pd
import yfinance as yf

# Append project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_pipeline.audit import Layer1DataIntegrity
from src.data_pipeline.topology import Layer2MicrostructureTopologies
from src.data_pipeline.labeling import Layer0TripleBarrier


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-asset Data Fetcher, Bar Auditor, and Labeler")
    parser.add_argument("--portfolio_path", type=str, default="data/final_portfolio.csv", help="Path to portfolio CSV")
    parser.add_argument("--cache_dir", type=str, default="data/cache", help="Cache directory for downloaded data")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Directory to save processed datasets")
    parser.add_argument("--start_date", type=str, default="2018-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default="2026-06-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--target_horizon", type=int, default=5, help="Triple barrier horizontal barrier (bars)")
    parser.add_argument("--pt", type=float, default=1.0, help="Profit taking multiplier (pt * sigma)")
    parser.add_argument("--sl", type=float, default=1.0, help="Stop loss multiplier (sl * sigma)")
    parser.add_argument("--vol_span", type=int, default=20, help="Span for causal EWMA volatility estimation")
    parser.add_argument("--vol_floor", type=float, default=0.005, help="Minimum volatility floor to prevent zero-width barriers")
    return parser.parse_args()


class MultiAssetDataPreparer:
    def __init__(self, portfolio_path: str, start_date: str, end_date: str, cache_dir: str):
        self.portfolio_path = portfolio_path
        self.start_date = start_date
        self.end_date = end_date
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        if os.path.exists(self.portfolio_path):
            self.df_portfolio = pd.read_csv(self.portfolio_path)
            self.tickers = sorted(self.df_portfolio["Ticker"].unique().tolist())
            print(f"[*] Initialized portfolio: {len(self.tickers)} unique tickers loaded from {self.portfolio_path}")
        else:
            raise FileNotFoundError(f"Portfolio file not found at: {self.portfolio_path}")

    def fetch_history(self, symbol: str) -> pd.DataFrame:
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
            print(f"  [!] Error fetching {symbol} from Yahoo Finance: {err}")

        return pd.DataFrame()

    def clean_and_adjust(self, df: pd.DataFrame, df_mkt: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df[~df.index.duplicated(keep="last")].sort_index()

        # Audit candlestick geometry bounds
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

    def prepare_dataset(self) -> dict[str, pd.DataFrame]:
        print("[*] Fetching Market Benchmark (VN-INDEX)...")
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
            print(f"[{idx:02d}/{len(self.tickers):02d}] Preparing asset data: {ticker}...")
            df_raw = self.fetch_history(ticker)
            if not df_raw.empty and len(df_raw) > 50:
                clean_df = self.clean_and_adjust(df_raw, df_mkt)
                clean_dict[ticker] = clean_df
            else:
                print(f"  [!] Skipped {ticker} (empty or insufficient trading days).")
            time.sleep(1)

        print(f"\n[✓] Data Prep completed: {len(clean_dict)}/{len(self.tickers)} assets ready.")
        return clean_dict


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Fetch & Clean Multi-Asset Datasets
    preparer = MultiAssetDataPreparer(
        portfolio_path=args.portfolio_path,
        start_date=args.start_date,
        end_date=args.end_date,
        cache_dir=args.cache_dir
    )
    clean_price_dict = preparer.prepare_dataset()

    dict_path = os.path.join(args.output_dir, "clean_price_dict.pkl")
    with open(dict_path, "wb") as f:
        pickle.dump(clean_price_dict, f)
    print(f"[✓] Saved asset dictionary to {dict_path}")

    # 2. Build Synthetic Single Asset from Portfolio Weights
    df_port = preparer.df_portfolio.copy()
    min_year = 2018
    base_year = int(df_port["year"].max()) if "year" in df_port.columns else 2023
    
    if "year" in df_port.columns:
        for y in range(min_year, base_year):
            df_slice = df_port.loc[df_port["year"] == base_year].copy()
            df_slice["year"] = y
            df_port = pd.concat([df_port, df_slice], axis="rows").sort_values(by="year")

    df_raw = pd.DataFrame()
    available_years = df_port["year"].unique().tolist() if "year" in df_port.columns else [base_year]

    print("[*] Aggregating weighted synthetic portfolio time series...")
    for year in available_years:
        df_year = pd.DataFrame()
        for tick in df_port["Ticker"].unique().tolist():
            try:
                sub = df_port.loc[(df_port["year"] == year) & (df_port["Ticker"] == tick), "Weight_Final_Constrained"]
                if sub.empty:
                    continue
                weight = sub.item()
            except Exception:
                continue

            if tick in clean_price_dict:
                val = clean_price_dict[tick].copy()
                val = val.loc[val.index.year == year, :]
                if not val.empty:
                    df_year = weight * val if df_year.empty else df_year + (weight * val)

        if not df_year.empty:
            df_raw = pd.concat([df_raw, df_year], axis="rows")

    raw_path = os.path.join(args.output_dir, "df_raw.csv")
    df_raw.to_csv(raw_path)
    print(f"[✓] Saved synthetic portfolio to {raw_path}")

    # 3. Apply Point-in-time Integrity and Causal Triple Barrier Labeling
    print("[*] Performing Layer 1 Data Integrity audit...")
    df_clean, _ = Layer1DataIntegrity.run_audit(df_raw)

    print("[*] Generating Causal Triple Barrier Labels...")
    df_labeled = Layer0TripleBarrier.label(
        df_clean,
        h=args.target_horizon,
        pt=args.pt,
        sl=args.sl,
        vol_span=args.vol_span,
        vol_floor=args.vol_floor
    )

    labeled_path = os.path.join(args.output_dir, "df_labeled.csv")
    df_labeled.to_csv(labeled_path)
    print(f"[✓] Saved labeled dataset to {labeled_path}")

    # 4. Final dropna only on target label and lookback warm-up
    df_ready = df_labeled.dropna(subset=["target_label", "barrier_touch_time"]).copy()
    df_ready = df_ready.dropna().copy()

    ready_path = os.path.join(args.output_dir, "df_ready_for_cv.csv")
    df_ready.to_csv(ready_path)
    print(f"[✓] Ready for Purged CV: {len(df_ready)} rows saved to {ready_path} (NaN count: {df_ready.isna().sum().sum()})")


if __name__ == "__main__":
    main()