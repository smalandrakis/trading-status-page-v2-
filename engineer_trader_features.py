"""
Advanced Trader-Focused Feature Engineering

Built with insights from professional trading:
1. Multi-timeframe trend analysis (1h, 2h, 3h, 4h)
2. Previous day/week price levels and distance
3. Volume-hour relationships and anomalies
4. Market session patterns (Asian, European, US)
5. Momentum regimes and volatility states
6. Order flow proxies (volume imbalance, trade intensity)
7. Support/resistance levels
8. Market microstructure signals
"""

import pandas as pd
import numpy as np
from pathlib import Path


def add_multi_timeframe_trends(df):
    """
    Multi-timeframe trend features (1h, 2h, 3h, 4h)

    Professional traders always check higher timeframes.
    """
    print("  - Multi-timeframe trends...")

    periods_5min = {
        '1h': 12,   # 12 * 5min = 1 hour
        '2h': 24,
        '3h': 36,
        '4h': 48
    }

    for name, periods in periods_5min.items():
        # Price change over timeframe
        df[f'trend_{name}_pct'] = df['close'].pct_change(periods) * 100

        # High-low range over timeframe
        df[f'range_{name}'] = (df['high'].rolling(periods).max() - df['low'].rolling(periods).min()) / df['close']

        # Direction consistency (% of positive closes)
        df[f'trend_{name}_consistency'] = df['close'].diff().rolling(periods).apply(lambda x: (x > 0).sum() / len(x))

        # Volume trend
        df[f'vol_trend_{name}'] = df['volume'].pct_change(periods)

    return df


def add_previous_levels(df):
    """
    Previous day close and distance from current price - LIMITED LOOKBACK

    Using shorter lookback to preserve data.
    """
    print("  - Previous day levels (limited lookback)...")

    # Previous 4h close instead of 24h (48 candles = 4 hours)
    df['prev_4h_close'] = df['close'].shift(48)
    df['dist_from_prev_4h'] = (df['close'] - df['prev_4h_close']) / df['prev_4h_close'] * 100

    return df


def add_volume_hour_relationship(df):
    """
    Volume patterns by hour of day

    Crypto has distinct volume patterns across trading sessions.
    """
    print("  - Volume-hour relationships...")

    # Hour of day (0-23)
    df['hour'] = df.index.hour

    # Average volume by hour (computed across all data - this is OK for features)
    volume_by_hour = df.groupby('hour')['volume'].median()
    df['volume_hour_median'] = df['hour'].map(volume_by_hour)

    # Volume relative to hour's typical volume
    df['volume_vs_hour'] = df['volume'] / df['volume_hour_median']

    # Volume surge detection
    df['volume_surge'] = (df['volume_vs_hour'] > 2.0).astype(int)

    # Volume moving averages
    for periods in [12, 24, 48]:  # 1h, 2h, 4h
        df[f'volume_ma_{periods}'] = df['volume'].rolling(periods).mean()
        df[f'volume_ratio_{periods}'] = df['volume'] / df[f'volume_ma_{periods}']

    return df


def add_market_sessions(df):
    """
    Trading session indicators with volume characteristics

    UTC hours:
    - Asian session: 00:00-08:00 UTC (Tokyo, Singapore, Hong Kong)
    - European session: 07:00-16:00 UTC (London, Frankfurt)
    - US session: 13:00-22:00 UTC (NY, Chicago)
    """
    print("  - Market session features...")

    hour = df.index.hour

    # Session indicators
    df['session_asian'] = ((hour >= 0) & (hour < 8)).astype(int)
    df['session_european'] = ((hour >= 7) & (hour < 16)).astype(int)
    df['session_us'] = ((hour >= 13) & (hour < 22)).astype(int)
    df['session_overlap_eu_us'] = ((hour >= 13) & (hour < 16)).astype(int)  # High volume period

    # Session-specific volume
    df['volume_in_overlap'] = df['volume'] * df['session_overlap_eu_us']

    # Time until/since session changes (cyclical)
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)

    # Day of week
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    return df


def add_momentum_regimes(df):
    """
    Momentum and volatility regime classification

    Helps identify different market states.
    """
    print("  - Momentum regimes...")

    # ATR (volatility)
    for periods in [12, 24, 48]:
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df[f'atr_{periods}'] = tr.rolling(periods).mean()
        df[f'atr_{periods}_pct'] = df[f'atr_{periods}'] / df['close'] * 100

    # Volatility regime (low/med/high based on ATR percentiles)
    atr_48 = df['atr_48_pct']
    df['vol_regime_low'] = (atr_48 < atr_48.quantile(0.33)).astype(int)
    df['vol_regime_high'] = (atr_48 > atr_48.quantile(0.67)).astype(int)

    # Momentum strength (RSI-based)
    for periods in [14, 28]:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(periods).mean()
        rs = gain / loss
        df[f'rsi_{periods}'] = 100 - (100 / (1 + rs))

    # Momentum regime
    rsi_14 = df['rsi_14']
    df['momentum_oversold'] = (rsi_14 < 30).astype(int)
    df['momentum_overbought'] = (rsi_14 > 70).astype(int)

    # Trend strength (ADX proxy using directional movement)
    periods = 14
    df['dm_plus'] = (df['high'].diff().clip(lower=0))
    df['dm_minus'] = (-df['low'].diff().clip(lower=0))
    dm_sum = df['dm_plus'] + df['dm_minus'] + 1e-8  # Avoid division by zero
    df['dx'] = (df['dm_plus'] - df['dm_minus']).abs() / dm_sum * 100
    df['adx_proxy'] = df['dx'].rolling(periods).mean()

    return df


def add_order_flow_proxies(df):
    """
    Order flow and trade intensity proxies

    Without full order book data, use volume/trades as proxies.
    """
    print("  - Order flow proxies...")

    # Trade intensity (trades per unit volume)
    df['trade_intensity'] = df['trades'] / (df['volume'] + 1e-8)  # Avoid division by zero

    # Imbalance: up bars vs down bars
    df['price_change'] = df['close'].diff()
    df['up_bar'] = (df['price_change'] > 0).astype(int)
    df['down_bar'] = (df['price_change'] < 0).astype(int)

    # Rolling imbalance (last N bars)
    for periods in [12, 24]:
        df[f'imbalance_{periods}'] = (df['up_bar'].rolling(periods).sum() - df['down_bar'].rolling(periods).sum()) / periods

    # Volume-weighted price change
    df['vwpc'] = df['price_change'] * df['volume']
    df['vwpc_ma_12'] = df['vwpc'].rolling(12).mean()

    # Large move detection (>0.5% in single candle)
    df['large_move'] = (df['price_change'].abs() / df['close'] > 0.005).astype(int)

    return df


def add_support_resistance(df):
    """
    Dynamic support/resistance levels - REDUCED PERIODS

    Use recent highs/lows as key levels (max 8h lookback).
    """
    print("  - Support/resistance levels...")

    for periods in [24, 48, 96]:  # 2h, 4h, 8h (keep at 96 max)
        df[f'resistance_{periods}'] = df['high'].rolling(periods).max()
        df[f'support_{periods}'] = df['low'].rolling(periods).min()

        # Distance to levels
        df[f'dist_to_resistance_{periods}'] = (df[f'resistance_{periods}'] - df['close']) / df['close'] * 100
        df[f'dist_to_support_{periods}'] = (df['close'] - df[f'support_{periods}']) / df['close'] * 100

    return df


def add_price_patterns(df):
    """
    Simple price patterns and structures

    Candlestick and bar patterns.
    """
    print("  - Price patterns...")

    # Candle body and wick ratios
    df['body'] = (df['close'] - df['open']).abs()
    df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
    df['body_ratio'] = df['body'] / (df['high'] - df['low'] + 1e-8)

    # Doji detection (small body relative to range)
    df['is_doji'] = (df['body_ratio'] < 0.1).astype(int)

    # Consecutive bars
    df['consecutive_ups'] = (df['close'] > df['open']).astype(int).groupby((df['close'] <= df['open']).cumsum()).cumsum()
    df['consecutive_downs'] = (df['close'] < df['open']).astype(int).groupby((df['close'] >= df['open']).cumsum()).cumsum()

    # Gap detection (open vs previous close)
    df['gap'] = df['open'] - df['close'].shift()
    df['gap_pct'] = df['gap'] / df['close'].shift() * 100

    return df


def add_statistical_features(df):
    """
    Statistical and mathematical features

    Bollinger Bands, standard deviations, Z-scores.
    """
    print("  - Statistical features...")

    for periods in [20, 50]:
        # Bollinger Bands
        sma = df['close'].rolling(periods).mean()
        std = df['close'].rolling(periods).std()
        df[f'bb_upper_{periods}'] = sma + (2 * std)
        df[f'bb_lower_{periods}'] = sma - (2 * std)
        df[f'bb_position_{periods}'] = (df['close'] - df[f'bb_lower_{periods}']) / (df[f'bb_upper_{periods}'] - df[f'bb_lower_{periods}'])
        df[f'bb_width_{periods}'] = (df[f'bb_upper_{periods}'] - df[f'bb_lower_{periods}']) / sma

    # Z-score of returns
    for periods in [24, 48]:
        returns = df['close'].pct_change()
        mean_return = returns.rolling(periods).mean()
        std_return = returns.rolling(periods).std()
        df[f'return_zscore_{periods}'] = (returns - mean_return) / (std_return + 1e-8)

    return df


def main():
    # File paths
    data_dir = Path(__file__).parent / 'data'
    input_file = data_dir / 'btc_5m_labeled_multi.parquet'
    output_file = data_dir / 'btc_5m_features_advanced.parquet'

    print("="*60)
    print("Advanced Trader-Focused Feature Engineering")
    print("="*60)

    print(f"\nLoading labeled data from: {input_file}")
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df):,} rows with {len(df.columns)} columns")

    initial_cols = len(df.columns)

    print("\nBuilding features...")

    # Apply all feature engineering
    df = add_multi_timeframe_trends(df)
    df = add_previous_levels(df)
    df = add_volume_hour_relationship(df)
    df = add_market_sessions(df)
    df = add_momentum_regimes(df)
    df = add_order_flow_proxies(df)
    df = add_support_resistance(df)
    df = add_price_patterns(df)
    df = add_statistical_features(df)

    new_features = len(df.columns) - initial_cols
    print(f"\n✓ Created {new_features} new features")
    print(f"  Total columns: {len(df.columns)}")

    # Drop NaN rows (from rolling windows)
    print(f"\nChecking NaN counts by column...")
    nan_counts = df.isna().sum().sort_values(ascending=False).head(10)
    print("Top 10 columns with most NaN:")
    for col, count in nan_counts.items():
        print(f"  {col}: {count:,} ({count/len(df)*100:.2f}%)")

    print(f"\nRemoving NaN rows...")
    before = len(df)
    df = df.dropna()
    after = len(df)
    print(f"  Dropped {before - after:,} rows ({(before-after)/before*100:.2f}%)")
    print(f"  Remaining: {after:,} rows")

    # Check remaining labels for each horizon
    print(f"\nRemaining labeled samples:")
    for horizon in ['2h', '4h', '6h']:
        label_col = f'label_{horizon}'
        if label_col in df.columns:
            labeled = df[label_col].notna().sum()
            long = (df[label_col] == 1).sum()
            short = (df[label_col] == 0).sum()
            print(f"  {horizon}: {labeled:,} total ({long:,} LONG, {short:,} SHORT)")

    # Save
    print(f"\nSaving features to: {output_file}")
    df.to_parquet(output_file)

    print("\n✓ Feature engineering complete!")
    print(f"  Final shape: {df.shape}")


if __name__ == '__main__':
    main()
