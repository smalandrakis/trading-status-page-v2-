"""
Label Generation for Micro-Movement Trading (0.3% TP / 0.1% SL)

Creates binary labels (LONG, SHORT, NEUTRAL) for three shorter horizons:
- 30min (6 bars of 5min data)
- 1h (12 bars)
- 2h (24 bars)

Quality filter: Only label if movement exceeds 0.4% total range
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

print("="*80)
print("MICRO-MOVEMENT LABEL GENERATION")
print("Configuration: 0.3% TP / 0.1% SL")
print("="*80)

# Load data
BOT_DIR = Path(__file__).parent
data_file = BOT_DIR / 'data' / 'BTC_5min_8years.parquet'

print(f"\nLoading data from {data_file}...")
df = pd.read_parquet(data_file)
df.columns = [c.lower() for c in df.columns]
print(f"Loaded {len(df):,} rows ({df.index[0]} to {df.index[-1]})")

# Parameters
TP_PCT = 0.003  # 0.3%
SL_PCT = 0.001  # 0.1%
QUALITY_THRESHOLD = 0.004  # 0.4% minimum total range

def generate_labels_for_horizon(df, horizon_bars, horizon_name):
    """Generate labels for a specific time horizon"""
    print(f"\nGenerating labels for {horizon_name} horizon ({horizon_bars} bars)...")

    labels = []

    for i in range(len(df) - horizon_bars):
        current_price = df.iloc[i]['close']
        future_prices = df.iloc[i+1:i+1+horizon_bars]['close'].values

        if len(future_prices) < horizon_bars:
            labels.append('NEUTRAL')
            continue

        # Calculate max/min in forward window
        max_price = future_prices.max()
        min_price = future_prices.min()

        # Quality filter: skip if total range < 0.4%
        total_range_pct = (max_price - min_price) / current_price
        if total_range_pct < QUALITY_THRESHOLD:
            labels.append('NEUTRAL')
            continue

        # Check if TP would have been hit for LONG
        max_return = (max_price - current_price) / current_price
        min_return = (min_price - current_price) / current_price

        # LONG: TP hit before SL
        if max_return >= TP_PCT:
            # Check if SL was hit first
            hit_sl_first = False
            for price in future_prices:
                return_pct = (price - current_price) / current_price
                if return_pct <= -SL_PCT:
                    hit_sl_first = True
                    break
                if return_pct >= TP_PCT:
                    break

            if not hit_sl_first:
                labels.append('LONG')
                continue

        # SHORT: TP hit before SL
        if min_return <= -TP_PCT:
            # Check if SL was hit first
            hit_sl_first = False
            for price in future_prices:
                return_pct = (price - current_price) / current_price
                if return_pct >= SL_PCT:
                    hit_sl_first = True
                    break
                if return_pct <= -TP_PCT:
                    break

            if not hit_sl_first:
                labels.append('SHORT')
                continue

        labels.append('NEUTRAL')

        if (i + 1) % 50000 == 0:
            print(f"  Progress: {i+1:,}/{len(df):,} rows...")

    # Add remaining rows as NEUTRAL
    labels.extend(['NEUTRAL'] * horizon_bars)

    return labels

# Generate labels for all horizons
df['label_30min'] = generate_labels_for_horizon(df, 6, '30min')
df['label_1h'] = generate_labels_for_horizon(df, 12, '1h')
df['label_2h'] = generate_labels_for_horizon(df, 24, '2h')

# Statistics
print("\n" + "="*80)
print("LABEL DISTRIBUTION:")
print("="*80)

for horizon in ['30min', '1h', '2h']:
    col = f'label_{horizon}'
    counts = df[col].value_counts()
    total = len(df)

    print(f"\n{horizon.upper()} Horizon:")
    for label in ['LONG', 'SHORT', 'NEUTRAL']:
        count = counts.get(label, 0)
        pct = count / total * 100
        print(f"  {label:8}: {count:,} ({pct:.1f}%)")

# Save
output_file = BOT_DIR / 'data' / 'BTC_5min_with_micro_labels.parquet'
print(f"\nSaving labeled data to {output_file}...")
df.to_parquet(output_file)

print(f"\n✓ Label generation complete!")
print(f"  Output: {output_file}")
print(f"  Total rows: {len(df):,}")
print("="*80)
