"""
Create Symmetric TP/SL Labels for Multiple Time Horizons

CORRECTED LOGIC:
- LONG signal (1): Price hits +1.0% TP before -0.5% SL
- SHORT signal (0): Price hits -1.0% TP before +0.5% SL

Both directions have 1% take profit with 0.5% stop loss (2:1 reward/risk).

Time Horizons:
- 2h window: 24 candles (5-min * 24 = 120 min)
- 4h window: 48 candles
- 6h window: 72 candles
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Constants
TP_PCT = 0.01  # 1% Take Profit (both directions)
SL_PCT = 0.005  # 0.5% Stop Loss (both directions)

HORIZONS = {
    '2h': 24,   # 24 * 5min = 2 hours
    '4h': 48,   # 48 * 5min = 4 hours
    '6h': 72    # 72 * 5min = 6 hours
}


def create_symmetric_labels(df, tp_pct=TP_PCT, sl_pct=SL_PCT, horizon_bars=24, horizon_name='2h'):
    """
    Create labels with SYMMETRIC TP/SL for both directions.

    LONG (1):  +1% TP hit before -0.5% SL
    SHORT (0): -1% TP hit before +0.5% SL

    Args:
        df: DataFrame with OHLCV data
        tp_pct: Take profit percentage (1% for both directions)
        sl_pct: Stop loss percentage (0.5% for both directions)
        horizon_bars: Number of bars to scan forward
        horizon_name: Label for this horizon (e.g., '2h', '4h', '6h')

    Returns:
        Series with labels: 1 (LONG), 0 (SHORT), NaN (no clear signal)
    """
    labels = pd.Series(index=df.index, dtype=float)

    close_prices = df['close'].values
    high_prices = df['high'].values
    low_prices = df['low'].values
    n = len(df)

    print(f"\nProcessing {horizon_name} horizon ({horizon_bars} bars = {horizon_bars*5} minutes)...")
    print(f"  TP: ±{tp_pct*100:.1f}%, SL: ±{sl_pct*100:.1f}%")

    long_count = 0
    short_count = 0
    no_signal_count = 0

    for i in range(n):
        if i % 10000 == 0 and i > 0:
            print(f"  Progress: {i:,}/{n:,} ({i/n*100:.1f}%)")

        current_close = close_prices[i]

        # Define TP and SL levels for LONG
        long_tp = current_close * (1 + tp_pct)   # +1%
        long_sl = current_close * (1 - sl_pct)   # -0.5%

        # Define TP and SL levels for SHORT
        short_tp = current_close * (1 - tp_pct)  # -1%
        short_sl = current_close * (1 + sl_pct)  # +0.5%

        # Scan forward window
        end_idx = min(i + horizon_bars + 1, n)

        if i + 1 >= n:
            labels.iloc[i] = np.nan
            no_signal_count += 1
            continue

        # Track first touch for LONG and SHORT
        long_hit = None  # 'TP' or 'SL'
        short_hit = None  # 'TP' or 'SL'
        long_bar = horizon_bars + 1
        short_bar = horizon_bars + 1

        # Scan forward to find first LONG outcome
        for j in range(i + 1, end_idx):
            if high_prices[j] >= long_tp:
                long_hit = 'TP'
                long_bar = j - i
                break
            if low_prices[j] <= long_sl:
                long_hit = 'SL'
                long_bar = j - i
                break

        # Scan forward to find first SHORT outcome
        for j in range(i + 1, end_idx):
            if low_prices[j] <= short_tp:
                short_hit = 'TP'
                short_bar = j - i
                break
            if high_prices[j] >= short_sl:
                short_hit = 'SL'
                short_bar = j - i
                break

        # Label based on which direction has better outcome
        # Prefer direction that hits TP first (sooner and with win)
        if long_hit == 'TP' and short_hit == 'TP':
            # Both directions profitable - choose faster one
            if long_bar <= short_bar:
                labels.iloc[i] = 1  # LONG
                long_count += 1
            else:
                labels.iloc[i] = 0  # SHORT
                short_count += 1
        elif long_hit == 'TP' and short_hit != 'TP':
            # Only LONG profitable
            labels.iloc[i] = 1
            long_count += 1
        elif short_hit == 'TP' and long_hit != 'TP':
            # Only SHORT profitable
            labels.iloc[i] = 0
            short_count += 1
        else:
            # Neither profitable or both hit SL
            labels.iloc[i] = np.nan
            no_signal_count += 1

    print(f"  Labeling complete!")
    print(f"    LONG:  {long_count:,}")
    print(f"    SHORT: {short_count:,}")
    print(f"    No signal: {no_signal_count:,}")

    return labels


def main():
    # File paths
    data_dir = Path(__file__).parent / 'data'
    input_file = data_dir / 'btc_5m_12mo.parquet'

    print("="*60)
    print("Creating Symmetric TP/SL Labels for Multiple Horizons")
    print("="*60)
    print("\nLONG:  +1.0% TP before -0.5% SL")
    print("SHORT: -1.0% TP before +0.5% SL")

    print(f"\nLoading 5-minute data from: {input_file}")
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df):,} rows")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")

    # Create labels for each horizon
    for horizon_name, horizon_bars in HORIZONS.items():
        print(f"\n{'='*60}")
        print(f"Horizon: {horizon_name}")
        print(f"{'='*60}")

        labels = create_symmetric_labels(df, TP_PCT, SL_PCT, horizon_bars, horizon_name)

        # Add labels to dataframe
        col_name = f'label_{horizon_name}'
        df[col_name] = labels

        # Statistics
        total = len(df)
        long_count = (labels == 1).sum()
        short_count = (labels == 0).sum()
        no_signal = labels.isna().sum()
        labeled = long_count + short_count

        print(f"\n  Summary for {horizon_name} horizon:")
        print(f"    Total rows: {total:,}")
        print(f"    LONG (1):   {long_count:,} ({long_count/total*100:.2f}%)")
        print(f"    SHORT (0):  {short_count:,} ({short_count/total*100:.2f}%)")
        print(f"    No signal:  {no_signal:,} ({no_signal/total*100:.2f}%)")

        if labeled > 0:
            print(f"\n    Labeled balance:")
            print(f"      LONG:  {long_count/labeled*100:.2f}%")
            print(f"      SHORT: {short_count/labeled*100:.2f}%")

            balance = max(long_count, short_count) / labeled
            if balance < 0.6:
                print(f"      ✓ Good balance!")
            elif balance < 0.7:
                print(f"      ⚠ Moderate imbalance")
            else:
                print(f"      ⚠⚠ High imbalance - consider scale_pos_weight")

    # Save labeled data
    output_file = data_dir / 'btc_5m_labeled_multi.parquet'
    print(f"\n{'='*60}")
    print(f"Saving labeled data to: {output_file}")
    df.to_parquet(output_file)

    print("\n✓ Done!")
    print(f"  Created {len(HORIZONS)} sets of labels: {list(HORIZONS.keys())}")
    print(f"  Output columns: {[f'label_{h}' for h in HORIZONS.keys()]}")


if __name__ == '__main__':
    main()
