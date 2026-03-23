"""
Download QQQ market data and compute 40+ technical indicators.
- 5-min bars: ~60 days (Yahoo Finance limit)
- Daily bars: 5 years of history
"""

import yfinance as yf
import pandas as pd
import ta
from datetime import datetime, timedelta
import os

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all available technical indicators from the 'ta' library."""
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Ensure we have the required columns
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Add ALL indicators from ta library (40+ indicators)
    df = ta.add_all_ta_features(
        df,
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
        fillna=True  # Fill NaN values
    )
    
    return df


def download_and_process(ticker: str, interval: str, period: str, filename: str):
    """Download data and add indicators."""
    
    print(f"\n{'='*60}")
    print(f"Downloading {ticker} - {interval} bars - {period} period")
    print('='*60)
    
    # Download data
    data = yf.download(ticker, period=period, interval=interval, progress=True)
    
    if data.empty:
        print(f"ERROR: No data returned for {ticker}")
        return None
    
    # Flatten multi-level columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    print(f"Downloaded {len(data)} bars")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Add technical indicators
    print("Computing technical indicators...")
    data = add_all_indicators(data)
    
    # Save to CSV
    filepath = os.path.join(DATA_DIR, filename)
    data.to_csv(filepath)
    print(f"Saved to {filepath}")
    print(f"Total columns (price + indicators): {len(data.columns)}")
    
    return data


def main():
    ticker = "QQQ"
    
    # 1. Download 5-minute data (max ~60 days available)
    df_5min = download_and_process(
        ticker=ticker,
        interval="5m",
        period="60d",  # Max available for 5-min
        filename="QQQ_5min_with_indicators.csv"
    )
    
    # 2. Download daily data (5 years)
    df_daily = download_and_process(
        ticker=ticker,
        interval="1d",
        period="5y",
        filename="QQQ_daily_with_indicators.csv"
    )
    
    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE")
    print("="*60)
    
    if df_5min is not None:
        print(f"\n5-Minute Data:")
        print(f"  - Bars: {len(df_5min)}")
        print(f"  - Date range: {df_5min.index[0]} to {df_5min.index[-1]}")
        print(f"  - Columns: {len(df_5min.columns)}")
    
    if df_daily is not None:
        print(f"\nDaily Data:")
        print(f"  - Bars: {len(df_daily)}")
        print(f"  - Date range: {df_daily.index[0]} to {df_daily.index[-1]}")
        print(f"  - Columns: {len(df_daily.columns)}")
    
    # List all indicator columns
    if df_daily is not None:
        base_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        indicator_cols = [c for c in df_daily.columns if c not in base_cols]
        print(f"\nTechnical Indicators ({len(indicator_cols)} total):")
        for i, col in enumerate(indicator_cols, 1):
            print(f"  {i:2d}. {col}")


if __name__ == "__main__":
    main()
