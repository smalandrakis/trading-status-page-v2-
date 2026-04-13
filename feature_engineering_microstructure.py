"""
Micro-Structure Feature Engineering

Add 10 new features focused on intraday micro-movements:
1. VWAP distance (5-period, 20-period)
2. Tick velocity (price changes per bar)
3. Realized volatility (5-bar rolling)
4. Micro support/resistance (recent highs/lows)
5. Volume profile deviation
6. Intraday session markers
7. Price momentum (1-bar, 3-bar, 5-bar)
8. Bid-ask spread proxy (high-low range)
"""

import pandas as pd
import numpy as np
from pathlib import Path

BOT_DIR = Path(__file__).parent

def calculate_vwap_distance(df, period):
    """VWAP distance over period"""
    vwap = (df['close'] * df['volume']).rolling(period).sum() / df['volume'].rolling(period).sum()
    return (df['close'] - vwap) / vwap

def calculate_tick_velocity(df, period=5):
    """Price changes per bar over period"""
    price_changes = df['close'].diff().abs()
    return price_changes.rolling(period).sum() / period

def calculate_realized_volatility(df, period=5):
    """Realized volatility over period"""
    returns = df['close'].pct_change()
    return returns.rolling(period).std()

def calculate_micro_sr(df, lookback=20):
    """Distance to recent support/resistance"""
    recent_high = df['high'].rolling(lookback).max()
    recent_low = df['low'].rolling(lookback).min()

    dist_to_high = (recent_high - df['close']) / df['close']
    dist_to_low = (df['close'] - recent_low) / df['close']

    return dist_to_high, dist_to_low

def calculate_volume_profile_deviation(df, period=20):
    """Deviation from average volume"""
    avg_volume = df['volume'].rolling(period).mean()
    return (df['volume'] - avg_volume) / avg_volume

def calculate_intraday_session(df):
    """Intraday session markers (Asia/Europe/US open)"""
    hour = df.index.hour

    # BTC trades 24/7 but major activity around:
    # Asia: 00:00-08:00 UTC
    # Europe: 08:00-16:00 UTC
    # US: 13:00-21:00 UTC

    is_asia = ((hour >= 0) & (hour < 8)).astype(int)
    is_europe = ((hour >= 8) & (hour < 16)).astype(int)
    is_us = ((hour >= 13) & (hour < 21)).astype(int)

    return is_asia, is_europe, is_us

def calculate_price_momentum(df, periods=[1, 3, 5]):
    """Price momentum over different periods"""
    momentums = {}
    for p in periods:
        momentums[f'momentum_{p}bar'] = df['close'].pct_change(p)
    return momentums

def calculate_spread_proxy(df, period=5):
    """Bid-ask spread proxy using high-low range"""
    hl_range = (df['high'] - df['low']) / df['close']
    return hl_range.rolling(period).mean()

def add_microstructure_features(df):
    """Add all micro-structure features to dataframe"""
    print("Calculating micro-structure features...")

    # VWAP
    print("  1/10: VWAP distance...")
    df['vwap_dist_5'] = calculate_vwap_distance(df, 5)
    df['vwap_dist_20'] = calculate_vwap_distance(df, 20)

    # Tick velocity
    print("  2/10: Tick velocity...")
    df['tick_velocity'] = calculate_tick_velocity(df, 5)

    # Realized volatility
    print("  3/10: Realized volatility...")
    df['realized_vol_5'] = calculate_realized_volatility(df, 5)

    # Micro S/R
    print("  4/10: Micro support/resistance...")
    dist_high, dist_low = calculate_micro_sr(df, 20)
    df['dist_to_high'] = dist_high
    df['dist_to_low'] = dist_low

    # Volume profile
    print("  5/10: Volume profile deviation...")
    df['volume_deviation'] = calculate_volume_profile_deviation(df, 20)

    # Intraday sessions
    print("  6/10: Intraday session markers...")
    is_asia, is_europe, is_us = calculate_intraday_session(df)
    df['session_asia'] = is_asia
    df['session_europe'] = is_europe
    df['session_us'] = is_us

    # Price momentum
    print("  7/10: Price momentum...")
    momentums = calculate_price_momentum(df, [1, 3, 5])
    for key, val in momentums.items():
        df[key] = val

    # Spread proxy
    print("  8/10: Spread proxy...")
    df['spread_proxy'] = calculate_spread_proxy(df, 5)

    # Additional: Price position in recent range
    print("  9/10: Price position in range...")
    recent_high = df['high'].rolling(20).max()
    recent_low = df['low'].rolling(20).min()
    df['price_position'] = (df['close'] - recent_low) / (recent_high - recent_low)

    # Additional: Volume trend
    print("  10/10: Volume trend...")
    df['volume_trend'] = df['volume'].pct_change(5)

    print("✓ Added 17 micro-structure features")

    return df

# Test on sample
if __name__ == '__main__':
    print("="*80)
    print("MICRO-STRUCTURE FEATURE ENGINEERING TEST")
    print("="*80)

    data_file = BOT_DIR / 'data' / 'BTC_5min_8years.parquet'
    print(f"\nLoading data from {data_file}...")
    df = pd.read_parquet(data_file)
    df.columns = [c.lower() for c in df.columns]

    print(f"Original columns: {df.columns.tolist()}")
    print(f"Original shape: {df.shape}")

    df = add_microstructure_features(df)

    print(f"\nNew columns added:")
    new_cols = ['vwap_dist_5', 'vwap_dist_20', 'tick_velocity', 'realized_vol_5',
                'dist_to_high', 'dist_to_low', 'volume_deviation',
                'session_asia', 'session_europe', 'session_us',
                'momentum_1bar', 'momentum_3bar', 'momentum_5bar',
                'spread_proxy', 'price_position', 'volume_trend']

    for col in new_cols:
        print(f"  {col}: {df[col].notna().sum():,} non-null values")

    print(f"\nFinal shape: {df.shape}")

    # Show sample
    print("\nSample rows (last 5):")
    print(df[new_cols].tail())

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    print("\nNext: Integrate these features into model training pipeline")
