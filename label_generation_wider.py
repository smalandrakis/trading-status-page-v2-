"""
Label Generation for Wider Micro-Movement Targets (0.5% TP / 0.15% SL)

Test if wider targets reduce NEUTRAL bias and improve signal quality
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path

print("="*80)
print("LABEL GENERATION - WIDER TARGETS (0.5% TP / 0.15% SL)")
print("="*80)

BOT_DIR = Path(__file__).parent

# Load data
data_file = BOT_DIR / 'data' / 'BTC_5min_8years.parquet'
print(f"\nLoading data from {data_file}...")
df = pd.read_parquet(data_file)
df.columns = [c.lower() for c in df.columns]
print(f"Loaded {len(df):,} rows")

# Wider targets
TP_PCT = 0.005  # 0.5% (vs 0.3% before)
SL_PCT = 0.0015  # 0.15% (vs 0.1% before)
QUALITY_THRESHOLD = 0.006  # 0.6% total range (vs 0.4% before)

# Three horizons (same as before)
horizons = [
    (6, '30min'),   # 6 bars = 30 min
    (12, '1h'),     # 12 bars = 1h
    (24, '2h')      # 24 bars = 2h
]

print(f"\nTarget Parameters:")
print(f"  TP: {TP_PCT*100:.1f}%")
print(f"  SL: {SL_PCT*100:.2f}%")
print(f"  Quality threshold: {QUALITY_THRESHOLD*100:.1f}% (min total range)")
print(f"  TP/SL Ratio: {TP_PCT/SL_PCT:.1f}:1")
print()

def generate_labels_for_horizon(df, horizon_bars, horizon_name):
    """Generate labels for one time horizon"""
    print(f"Generating labels for {horizon_name} ({horizon_bars} bars)...")

    labels = []
    for i in range(len(df) - horizon_bars):
        current_price = df.iloc[i]['close']
        future_prices = df.iloc[i+1:i+1+horizon_bars]['close'].values

        if len(future_prices) == 0:
            labels.append('NEUTRAL')
            continue

        max_price = max(future_prices)
        min_price = min(future_prices)

        max_return = (max_price / current_price) - 1
        min_return = (min_price / current_price) - 1

        # Quality filter - wider threshold
        total_range_pct = (max_price - min_price) / current_price
        if total_range_pct < QUALITY_THRESHOLD:
            labels.append('NEUTRAL')
            continue

        # Check if TP hit before SL for LONG
        hit_sl_first = False
        for price in future_prices:
            return_pct = (price / current_price) - 1
            if return_pct >= TP_PCT:
                if not hit_sl_first:
                    labels.append('LONG')
                    break
            elif return_pct <= -SL_PCT:
                hit_sl_first = True
                labels.append('SHORT')
                break
        else:
            labels.append('NEUTRAL')

    # Add NaN for last horizon_bars rows
    labels.extend([np.nan] * horizon_bars)

    return labels

# Generate labels for each horizon
for horizon_bars, horizon_name in horizons:
    labels = generate_labels_for_horizon(df, horizon_bars, horizon_name)
    df[f'label_{horizon_name}'] = labels

    # Distribution
    value_counts = df[f'label_{horizon_name}'].value_counts()
    total = value_counts.sum()

    print(f"\n{horizon_name} Label Distribution:")
    for label in ['LONG', 'SHORT', 'NEUTRAL']:
        count = value_counts.get(label, 0)
        pct = count / total * 100
        print(f"  {label:8}: {count:7,} ({pct:5.1f}%)")

# Save
output_file = BOT_DIR / 'data' / 'BTC_5min_with_wider_labels.parquet'
df.to_parquet(output_file)
print(f"\n✓ Saved labeled data: {output_file}")
print(f"  Total rows: {len(df):,}")
print(f"  Columns: {', '.join(df.columns[-3:])}")

print("\n" + "="*80)
print("LABEL GENERATION COMPLETE")
print("="*80)

# Quick comparison to original 0.3%/0.1% labels
print("\nComparison to Original 0.3% TP / 0.1% SL:")
print("\nOriginal (0.3%/0.1%):")
print("  30min: 12.5% LONG, 12.7% SHORT, 74.8% NEUTRAL")
print("  1h:    21.0% LONG, 21.0% SHORT, 58.1% NEUTRAL")
print("  2h:    27.4% LONG, 27.1% SHORT, 45.4% NEUTRAL")
print("\nExpected with Wider Targets (0.5%/0.15%):")
print("  Should see LESS neutral bias (more LONG/SHORT signals)")
print("  Better signal quality due to larger movements")
print("="*80)
