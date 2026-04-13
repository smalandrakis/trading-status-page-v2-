"""
Resample BTC 1-minute data to 5-minute candles

This reduces noise while maintaining sufficient resolution for intraday patterns.
5-min candles are common in algorithmic trading for catching short-term moves.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def resample_to_5min(df):
    """
    Resample 1-minute OHLCV data to 5-minute candles

    Aggregation rules:
    - open: first value in window
    - high: max value in window
    - low: min value in window
    - close: last value in window
    - volume: sum of volume in window
    - trades: sum of trades in window
    """
    print("Resampling to 5-minute candles...")

    resampled = df.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'trades': 'sum'
    })

    # Drop rows with NaN (incomplete candles at boundaries)
    resampled = resampled.dropna()

    return resampled


def main():
    # File paths
    data_dir = Path(__file__).parent / 'data'
    input_file = data_dir / 'btc_1m_12mo.parquet'
    output_file = data_dir / 'btc_5m_12mo.parquet'

    print("="*60)
    print("Resampling BTC Data: 1-min → 5-min")
    print("="*60)

    print(f"\nLoading 1-minute data from: {input_file}")
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df):,} rows (1-minute candles)")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Columns: {list(df.columns)}")

    # Resample to 5-min
    df_5m = resample_to_5min(df)

    print(f"\nResampled to {len(df_5m):,} rows (5-minute candles)")
    print(f"Reduction: {len(df)/len(df_5m):.1f}x fewer candles")
    print(f"Date range: {df_5m.index[0]} to {df_5m.index[-1]}")

    # Verify data quality
    print("\nData Quality Checks:")
    print(f"  Missing values: {df_5m.isna().sum().sum()}")
    print(f"  Zero volume candles: {(df_5m['volume'] == 0).sum()}")

    # Show sample
    print("\nSample data (first 5 rows):")
    print(df_5m.head())

    # Save
    print(f"\nSaving 5-minute data to: {output_file}")
    df_5m.to_parquet(output_file)

    print("\n✓ Done!")
    print(f"  Original: {len(df):,} 1-min candles")
    print(f"  Resampled: {len(df_5m):,} 5-min candles")


if __name__ == '__main__':
    main()
