"""
Create TP/SL based labels for BTC 1-minute data.

Labels each candle based on first-touch logic:
- LONG (1): If price reaches +1% TP before hitting -0.5% SL
- SHORT (0): If price reaches -0.5% SL before hitting +1% TP

Uses vectorized operations for performance on 525K rows.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Constants
TP_PCT = 0.01  # 1% Take Profit
SL_PCT = 0.005  # 0.5% Stop Loss
SCAN_WINDOW = 200  # Look forward 200 bars (200 minutes)

def create_tpsl_labels(df, tp_pct=TP_PCT, sl_pct=SL_PCT, scan_window=SCAN_WINDOW):
    """
    Create TP/SL based labels using first-touch logic.

    Args:
        df: DataFrame with OHLCV data (must have 'close', 'high', 'low' columns)
        tp_pct: Take profit percentage (default 1%)
        sl_pct: Stop loss percentage (default 0.5%)
        scan_window: Number of bars to scan forward (default 200)

    Returns:
        Series with labels: 1 (LONG), 0 (SHORT), NaN (no clear signal)
    """
    labels = pd.Series(index=df.index, dtype=float)

    close_prices = df['close'].values
    high_prices = df['high'].values
    low_prices = df['low'].values
    n = len(df)

    print(f"Processing {n} rows with scan window of {scan_window} bars...")
    print(f"TP threshold: +{tp_pct*100:.1f}%, SL threshold: -{sl_pct*100:.1f}%")

    # Process each bar
    for i in range(n):
        if i % 50000 == 0:
            print(f"  Progress: {i}/{n} ({i/n*100:.1f}%)")

        current_close = close_prices[i]

        # Define TP and SL price levels
        tp_long = current_close * (1 + tp_pct)
        sl_long = current_close * (1 - sl_pct)

        # Scan forward window (but not beyond array end)
        end_idx = min(i + 1, n)  # Start from next bar
        max_scan = min(i + scan_window + 1, n)

        if end_idx >= n:
            # Not enough forward data
            labels.iloc[i] = np.nan
            continue

        # Find first touch of TP or SL in forward window
        tp_hit = False
        sl_hit = False

        for j in range(end_idx, max_scan):
            # Check if TP is hit (high reaches TP threshold)
            if high_prices[j] >= tp_long:
                tp_hit = True
                break

            # Check if SL is hit (low reaches SL threshold)
            if low_prices[j] <= sl_long:
                sl_hit = True
                break

        # Label based on first touch
        if tp_hit and not sl_hit:
            labels.iloc[i] = 1  # LONG signal
        elif sl_hit and not tp_hit:
            labels.iloc[i] = 0  # SHORT signal
        else:
            # Neither hit within window, or both hit simultaneously (rare)
            labels.iloc[i] = np.nan

    print(f"Labeling complete!")
    return labels


def main():
    # File paths
    data_dir = Path(__file__).parent / 'data'
    input_file = data_dir / 'btc_1m_12mo.parquet'
    output_file = data_dir / 'btc_1m_labeled.parquet'

    print(f"Loading data from: {input_file}")
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")

    # Create labels
    print("\nCreating TP/SL labels...")
    df['label'] = create_tpsl_labels(df)

    # Report statistics
    print("\n" + "="*60)
    print("LABEL DISTRIBUTION")
    print("="*60)

    total_rows = len(df)
    long_count = (df['label'] == 1).sum()
    short_count = (df['label'] == 0).sum()
    nan_count = df['label'].isna().sum()

    print(f"Total rows: {total_rows:,}")
    print(f"LONG labels (1):  {long_count:,} ({long_count/total_rows*100:.2f}%)")
    print(f"SHORT labels (0): {short_count:,} ({short_count/total_rows*100:.2f}%)")
    print(f"No label (NaN):   {nan_count:,} ({nan_count/total_rows*100:.2f}%)")

    labeled_count = long_count + short_count
    if labeled_count > 0:
        print(f"\nLabeled data balance:")
        print(f"  LONG:  {long_count/labeled_count*100:.2f}%")
        print(f"  SHORT: {short_count/labeled_count*100:.2f}%")

        # Check for extreme imbalance
        imbalance_ratio = max(long_count, short_count) / labeled_count
        if imbalance_ratio > 0.6:
            print(f"\n⚠️  Warning: Class imbalance detected ({imbalance_ratio*100:.1f}% majority class)")
            print("  Consider using 'scale_pos_weight' in XGBoost training")

    # Save labeled data
    print(f"\nSaving labeled data to: {output_file}")
    df.to_parquet(output_file)
    print("✓ Done!")


if __name__ == '__main__':
    main()
