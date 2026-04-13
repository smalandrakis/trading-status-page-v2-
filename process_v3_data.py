"""
Process V3 Dataset: Resample + Label + Features

Applies the same pipeline to 4-year dataset:
1. Resample 1-min to 5-min candles
2. Create symmetric TP/SL labels (1%/-0.5%) for 2h/4h/6h horizons
3. Engineer the same 22 selected features used in V2
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# Constants
TP_PCT = 0.01  # 1% Take Profit
SL_PCT = 0.005  # 0.5% Stop Loss
HORIZONS = {
    '2h': 24,  # 24 * 5min = 2h
    '4h': 48,  # 48 * 5min = 4h
    '6h': 72   # 72 * 5min = 6h
}


def resample_to_5min(df):
    """Resample 1-min to 5-min candles"""
    print("\nResampling to 5-min candles...")

    resampled = df.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'trades': 'sum'
    }).dropna()

    print(f"  {len(df):,} rows → {len(resampled):,} rows")
    return resampled


def create_symmetric_labels(df):
    """Create TP/SL labels for all horizons"""
    print("\nCreating symmetric labels...")

    for horizon_name, bars_ahead in HORIZONS.items():
        print(f"  {horizon_name} ({bars_ahead} bars)...")

        labels = []

        for i in range(len(df)):
            current_close = df.iloc[i]['close']

            # Define levels
            long_tp = current_close * (1 + TP_PCT)
            long_sl = current_close * (1 - SL_PCT)
            short_tp = current_close * (1 - TP_PCT)  # Symmetric!
            short_sl = current_close * (1 + SL_PCT)

            # Look ahead
            end_idx = min(i + bars_ahead + 1, len(df))
            future = df.iloc[i+1:end_idx]

            if len(future) == 0:
                labels.append(np.nan)
                continue

            # Check which level hit first
            long_tp_hit = (future['high'] >= long_tp).idxmax() if (future['high'] >= long_tp).any() else None
            long_sl_hit = (future['low'] <= long_sl).idxmax() if (future['low'] <= long_sl).any() else None
            short_tp_hit = (future['low'] <= short_tp).idxmax() if (future['low'] <= short_tp).any() else None
            short_sl_hit = (future['high'] >= short_sl).idxmax() if (future['high'] >= short_sl).any() else None

            # LONG: TP before SL?
            long_winner = False
            if long_tp_hit is not None:
                if long_sl_hit is None or long_tp_hit < long_sl_hit:
                    long_winner = True

            # SHORT: TP before SL?
            short_winner = False
            if short_tp_hit is not None:
                if short_sl_hit is None or short_tp_hit < short_sl_hit:
                    short_winner = True

            # Assign label
            if long_winner and not short_winner:
                labels.append(1)  # LONG
            elif short_winner and not long_winner:
                labels.append(0)  # SHORT
            else:
                labels.append(np.nan)  # Ambiguous

        df[f'label_{horizon_name}'] = labels

        valid = df[f'label_{horizon_name}'].notna().sum()
        long_pct = (df[f'label_{horizon_name}'] == 1).sum() / valid * 100 if valid > 0 else 0
        short_pct = (df[f'label_{horizon_name}'] == 0).sum() / valid * 100 if valid > 0 else 0

        print(f"    Valid: {valid:,} ({valid/len(df)*100:.1f}%)")
        print(f"    LONG: {long_pct:.1f}%, SHORT: {short_pct:.1f}%")

    return df


def engineer_selected_features(df):
    """Engineer only the 22 selected features from V2"""
    print("\nEngineering 22 selected features...")

    # Load selected features
    models_dir = Path(__file__).parent / 'ml_models'
    with open(models_dir / 'selected_features.json', 'r') as f:
        selected_features = json.load(f)

    print(f"  Target: {len(selected_features)} features")

    # Calculate all features (need full set first, then filter)
    # Price-based
    df['returns_1'] = df['close'].pct_change(1)
    df['returns_5'] = df['close'].pct_change(5)
    df['returns_10'] = df['close'].pct_change(10)
    df['returns_20'] = df['close'].pct_change(20)

    # Moving averages
    for period in [12, 24, 48, 96]:
        df[f'sma_{period}'] = df['close'].rolling(period).mean()
        df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        df[f'dist_sma_{period}'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}'] * 100
        df[f'dist_ema_{period}'] = (df['close'] - df[f'ema_{period}']) / df[f'ema_{period}'] * 100

    # Volatility
    for period in [12, 24, 48]:
        df[f'rolling_std_{period}'] = df['returns_1'].rolling(period).std()
        df[f'atr_{period}'] = df[['high', 'low', 'close']].apply(
            lambda x: x['high'] - x['low'], axis=1
        ).rolling(period).mean()
        df[f'atr_{period}_pct'] = df[f'atr_{period}'] / df['close'] * 100

    # Volume
    for period in [12, 24, 48]:
        df[f'volume_ma_{period}'] = df['volume'].rolling(period).mean()
        df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_ma_{period}']

    # RSI
    for period in [14, 21, 28]:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-8)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

    # Trend features
    for hours in [1, 2, 3, 4]:
        bars = hours * 12  # 5-min bars
        df[f'trend_{hours}h'] = df['close'] - df['close'].shift(bars)
        df[f'trend_{hours}h_pct'] = df[f'trend_{hours}h'] / df['close'].shift(bars) * 100

    # Range features
    for hours in [1, 2, 4]:
        bars = hours * 12
        df[f'range_{hours}h'] = df['high'].rolling(bars).max() - df['low'].rolling(bars).min()
        df[f'range_{hours}h_pct'] = df[f'range_{hours}h'] / df['close'] * 100

    # Support/Resistance (most important!)
    for period in [48, 96]:
        df[f'support_{period}'] = df['low'].rolling(period).min()
        df[f'resistance_{period}'] = df['high'].rolling(period).max()
        df[f'dist_to_support_{period}'] = (df['close'] - df[f'support_{period}']) / df['close'] * 100
        df[f'dist_to_resistance_{period}'] = (df[f'resistance_{period}'] - df['close']) / df['close'] * 100

    # ADX proxy (simplified)
    tr = pd.DataFrame({
        'hl': df['high'] - df['low'],
        'hc': (df['high'] - df['close'].shift()).abs(),
        'lc': (df['low'] - df['close'].shift()).abs()
    }).max(axis=1)

    df['atr_14'] = tr.rolling(14).mean()

    plus_dm = df['high'].diff().clip(lower=0)
    minus_dm = -df['low'].diff().clip(upper=0)

    df['dm_plus'] = plus_dm.rolling(14).mean()
    df['dm_minus'] = minus_dm.rolling(14).mean()

    dm_sum = df['dm_plus'] + df['dm_minus'] + 1e-8
    df['dx'] = (df['dm_plus'] - df['dm_minus']).abs() / dm_sum * 100
    df['adx_proxy'] = df['dx'].rolling(14).mean()

    # Time features
    df['hour'] = df.index.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_of_week'] = df.index.dayofweek

    # Bollinger Band position (50 period)
    bb_mid_50 = df['close'].rolling(50).mean()
    bb_std_50 = df['close'].rolling(50).std()
    df['bb_position_50'] = (df['close'] - bb_mid_50) / (2 * bb_std_50 + 1e-8)

    # Distance from previous 4h close
    bars_4h = 48  # 4h in 5-min bars
    df['dist_from_prev_4h'] = (df['close'] - df['close'].shift(bars_4h)) / df['close'].shift(bars_4h) * 100

    # Volume hour median (median volume over the same hour historically)
    df['volume_hour_median'] = df.groupby('hour')['volume'].transform(
        lambda x: x.rolling(window=min(len(x), 20), min_periods=1).median()
    )

    # Filter to selected features only
    available_features = [f for f in selected_features if f in df.columns]
    missing_features = [f for f in selected_features if f not in df.columns]

    if missing_features:
        print(f"  ⚠ Missing {len(missing_features)} features: {missing_features[:5]}")

    print(f"  ✓ Engineered {len(available_features)} features")

    return df, available_features


def main():
    data_dir = Path(__file__).parent / 'data'

    print("="*60)
    print("V3 Data Processing Pipeline")
    print("="*60)

    # Load 4-year dataset
    input_file = data_dir / 'btc_1m_4yr.parquet'
    print(f"\nLoading: {input_file}")
    df = pd.read_parquet(input_file)
    print(f"  {len(df):,} rows ({df.index.min()} to {df.index.max()})")

    # Step 1: Resample
    df_5m = resample_to_5min(df)

    # Step 2: Create labels
    df_5m = create_symmetric_labels(df_5m)

    # Step 3: Engineer features
    df_5m, feature_cols = engineer_selected_features(df_5m)

    # Drop rows with NaN in features or labels
    print("\nCleaning data...")
    initial_rows = len(df_5m)

    # Drop rows with any NaN in features
    df_5m = df_5m.dropna(subset=feature_cols)
    print(f"  After feature NaN: {len(df_5m):,} rows ({len(df_5m)/initial_rows*100:.1f}%)")

    # Keep rows with at least one valid label
    has_label = df_5m[['label_2h', 'label_4h', 'label_6h']].notna().any(axis=1)
    df_5m = df_5m[has_label]
    print(f"  After label filter: {len(df_5m):,} rows ({len(df_5m)/initial_rows*100:.1f}%)")

    # Save
    output_file = data_dir / 'btc_5m_v3_features.parquet'
    print(f"\nSaving to: {output_file}")
    df_5m.to_parquet(output_file)

    # Summary
    print("\n" + "="*60)
    print("V3 DATA SUMMARY")
    print("="*60)
    print(f"  Total rows: {len(df_5m):,}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Date range: {df_5m.index.min()} to {df_5m.index.max()}")
    print(f"  Duration: {(df_5m.index.max() - df_5m.index.min()).days} days")

    for horizon in ['2h', '4h', '6h']:
        valid = df_5m[f'label_{horizon}'].notna().sum()
        long_count = (df_5m[f'label_{horizon}'] == 1).sum()
        short_count = (df_5m[f'label_{horizon}'] == 0).sum()
        print(f"\n  {horizon} labels:")
        print(f"    Valid: {valid:,}")
        print(f"    LONG: {long_count:,} ({long_count/valid*100:.1f}%)")
        print(f"    SHORT: {short_count:,} ({short_count/valid*100:.1f}%)")

    print(f"\n✓ V3 data ready for training!")


if __name__ == '__main__':
    main()
