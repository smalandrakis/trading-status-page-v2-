"""
Feature Engineering Pipeline for BTC TP/SL Model

Applies comprehensive technical indicators to labeled BTC data:
- Moving averages (SMA/EMA)
- Momentum indicators (RSI, MACD, Stochastic)
- Volatility indicators (ATR, Bollinger Bands)
- Volume features
- Lag features
- Rolling window statistics
- Time-based features
"""

import pandas as pd
import numpy as np
from pathlib import Path

def add_moving_averages(df):
    """Add SMA and EMA indicators"""
    periods = [5, 10, 20, 50, 100, 200]

    for period in periods:
        df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

        # Price ratios to moving averages
        df[f'close_to_sma_{period}'] = df['close'] / df[f'sma_{period}']
        df[f'close_to_ema_{period}'] = df['close'] / df[f'ema_{period}']

    return df


def add_momentum_indicators(df):
    """Add RSI, MACD, Stochastic, ROC"""
    # RSI with multiple periods
    for period in [7, 14, 21]:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Stochastic Oscillator
    for period in [14, 21]:
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()
        df[f'stoch_{period}'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df[f'stoch_{period}_ma'] = df[f'stoch_{period}'].rolling(window=3).mean()

    # Rate of Change
    for period in [5, 10, 20]:
        df[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100

    return df


def add_volatility_indicators(df):
    """Add ATR, Bollinger Bands, rolling standard deviations"""
    # ATR (Average True Range)
    for period in [7, 14, 21]:
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df[f'atr_{period}'] = tr.rolling(window=period).mean()
        # Normalize by price
        df[f'atr_{period}_pct'] = df[f'atr_{period}'] / df['close']

    # Bollinger Bands
    for period in [20, 50]:
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        df[f'bb_upper_{period}'] = sma + (2 * std)
        df[f'bb_lower_{period}'] = sma - (2 * std)
        df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / sma
        df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])

    # Rolling standard deviations
    for period in [5, 10, 20]:
        df[f'std_{period}'] = df['close'].pct_change().rolling(window=period).std()

    return df


def add_volume_features(df):
    """Add volume-based features"""
    # Volume moving averages
    for period in [5, 10, 20]:
        df[f'volume_ma_{period}'] = df['volume'].rolling(window=period).mean()
        df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_ma_{period}']

    # Volume change
    df['volume_change'] = df['volume'].pct_change()

    # Volume-weighted features
    df['vwap_20'] = (df['close'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
    df['close_to_vwap'] = df['close'] / df['vwap_20']

    return df


def add_lag_features(df):
    """Add lagged returns and prices"""
    # Lagged returns
    for lag in [1, 2, 3, 5, 10, 20]:
        df[f'return_lag_{lag}'] = df['close'].pct_change(lag)

    # Lagged close prices (normalized)
    for lag in [1, 2, 3, 5]:
        df[f'close_lag_{lag}_ratio'] = df['close'] / df['close'].shift(lag)

    # Lagged volume
    for lag in [1, 2, 3]:
        df[f'volume_lag_{lag}_ratio'] = df['volume'] / df['volume'].shift(lag)

    return df


def add_rolling_windows(df):
    """Add rolling window statistics"""
    # Rolling returns statistics
    for window in [5, 10, 20]:
        returns = df['close'].pct_change()
        df[f'return_mean_{window}'] = returns.rolling(window=window).mean()
        df[f'return_std_{window}'] = returns.rolling(window=window).std()
        df[f'return_min_{window}'] = returns.rolling(window=window).min()
        df[f'return_max_{window}'] = returns.rolling(window=window).max()

    # Rolling high/low
    for window in [5, 10, 20]:
        df[f'high_max_{window}'] = df['high'].rolling(window=window).max()
        df[f'low_min_{window}'] = df['low'].rolling(window=window).min()
        df[f'high_low_range_{window}'] = (df[f'high_max_{window}'] - df[f'low_min_{window}']) / df['close']

    return df


def add_time_features(df):
    """Add cyclical time-based features"""
    # Extract time components
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month

    # Cyclical encoding (sin/cos for hour and day of week)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # Trading session indicators (US market hours)
    df['is_us_trading'] = ((df['hour'] >= 14) & (df['hour'] < 21)).astype(int)  # 9:30 AM - 4:00 PM EST in UTC
    df['is_asia_trading'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)  # Asian session

    return df


def add_pattern_features(df):
    """Add pattern detection features"""
    # Consecutive ups/downs
    price_change = df['close'].diff()
    df['consecutive_ups'] = (price_change > 0).astype(int).groupby((price_change <= 0).cumsum()).cumsum()
    df['consecutive_downs'] = (price_change < 0).astype(int).groupby((price_change >= 0).cumsum()).cumsum()

    # Price momentum (acceleration)
    df['return_1'] = df['close'].pct_change()
    df['return_2'] = df['close'].pct_change(2)
    df['momentum_acceleration'] = df['return_1'] - df['return_2']

    return df


def main():
    # File paths
    data_dir = Path(__file__).parent / 'data'
    input_file = data_dir / 'btc_1m_labeled.parquet'
    output_file = data_dir / 'btc_1m_tpsl_features.parquet'

    print(f"Loading labeled data from: {input_file}")
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")

    # Count initial labeled rows
    initial_labeled = df['label'].notna().sum()
    print(f"Initial labeled rows: {initial_labeled:,}")

    print("\nApplying feature engineering...")

    # Apply all feature engineering functions
    print("  - Moving averages...")
    df = add_moving_averages(df)

    print("  - Momentum indicators...")
    df = add_momentum_indicators(df)

    print("  - Volatility indicators...")
    df = add_volatility_indicators(df)

    print("  - Volume features...")
    df = add_volume_features(df)

    print("  - Lag features...")
    df = add_lag_features(df)

    print("  - Rolling windows...")
    df = add_rolling_windows(df)

    print("  - Time features...")
    df = add_time_features(df)

    print("  - Pattern features...")
    df = add_pattern_features(df)

    print(f"\nTotal features created: {len(df.columns) - 7}")  # Subtract original OHLCV + trades + label

    # Drop rows with NaN in features (warm-up period)
    print("\nRemoving NaN rows (warm-up period for indicators)...")
    before_drop = len(df)
    df = df.dropna()
    after_drop = len(df)
    dropped = before_drop - after_drop

    print(f"  Dropped {dropped:,} rows ({dropped/before_drop*100:.2f}%)")
    print(f"  Remaining: {after_drop:,} rows")

    # Check remaining labeled data
    final_labeled = df['label'].notna().sum()
    long_count = (df['label'] == 1).sum()
    short_count = (df['label'] == 0).sum()

    print(f"\nFinal labeled data: {final_labeled:,}")
    print(f"  LONG:  {long_count:,} ({long_count/final_labeled*100:.2f}%)")
    print(f"  SHORT: {short_count:,} ({short_count/final_labeled*100:.2f}%)")

    # Save engineered dataset
    print(f"\nSaving features to: {output_file}")
    df.to_parquet(output_file)

    # Show sample
    print("\nSample features (first 5 rows):")
    print(df.head())

    print("\n✓ Feature engineering complete!")
    print(f"  Total columns: {len(df.columns)}")
    print(f"  Feature columns: {len(df.columns) - 7}")


if __name__ == '__main__':
    main()
