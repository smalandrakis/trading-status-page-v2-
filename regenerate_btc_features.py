"""
Regenerate BTC_features.parquet with all 206 features.
This script processes the existing parquet file and adds all missing features.
"""

import pandas as pd
import numpy as np
import ta
import json
import os
from datetime import datetime

print("=" * 80)
print("REGENERATING BTC_features.parquet WITH ALL 206 FEATURES")
print("=" * 80)
print(f"Started: {datetime.now()}")

# Load existing data (has OHLCV + 84 TA features)
print("\nLoading existing parquet file...")
df = pd.read_parquet('data/BTC_features.parquet')
print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
print(f"Date range: {df.index[0]} to {df.index[-1]}")

# Keep only OHLCV to regenerate all features consistently
ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
df_base = df[ohlcv_cols].copy()
print(f"\nExtracted base OHLCV data: {len(df_base.columns)} columns")

# Load expected features
with open('models_btc/feature_columns.json', 'r') as f:
    expected_features = json.load(f)
print(f"Target: {len(expected_features)} features")

# Step 1: Add all TA features
print("\n" + "=" * 60)
print("STEP 1: Adding TA features...")
print("=" * 60)

df = ta.add_all_ta_features(
    df_base.copy(), 
    open="Open", high="High", low="Low", 
    close="Close", volume="Volume", 
    fillna=True
)
print(f"After TA: {len(df.columns)} columns")

# Step 2: Add time features
print("\n" + "=" * 60)
print("STEP 2: Adding time features...")
print("=" * 60)

df['hour'] = df.index.hour
df['minute'] = df.index.minute
df['time_of_day'] = df['hour'] + df['minute'] / 60
df['is_premarket'] = 0  # BTC trades 24/7
df['is_regular_hours'] = 1
df['is_afterhours'] = 0
df['is_first_hour'] = (df['hour'] == 0).astype(int)
df['is_last_hour'] = (df['hour'] == 23).astype(int)
df['day_of_week'] = df.index.dayofweek
df['is_monday'] = (df['day_of_week'] == 0).astype(int)
df['is_friday'] = (df['day_of_week'] == 4).astype(int)
df['month'] = df.index.month
df['is_month_end'] = df.index.is_month_end.astype(int)
df['is_month_start'] = df.index.is_month_start.astype(int)
df['week_of_year'] = df.index.isocalendar().week.astype(int)

print(f"After time features: {len(df.columns)} columns")

# Step 3: Add price/return features
print("\n" + "=" * 60)
print("STEP 3: Adding price/return features...")
print("=" * 60)

df['return_1bar'] = df['Close'].pct_change()
df['return_5bar'] = df['Close'].pct_change(5)
df['return_10bar'] = df['Close'].pct_change(10)
df['return_20bar'] = df['Close'].pct_change(20)
df['log_return_1bar'] = np.log(df['Close'] / df['Close'].shift(1))

# Price relative to moving averages
for window in [5, 10, 20, 50]:
    ma = df['Close'].rolling(window).mean()
    df[f'price_to_ma{window}'] = df['Close'] / ma - 1

# Volatility features
for window in [5, 10, 20]:
    df[f'volatility_{window}bar'] = df['return_1bar'].rolling(window).std()

# Bar features
df['bar_range'] = (df['High'] - df['Low']) / df['Close']
df['bar_range_ma5'] = df['bar_range'].rolling(5).mean()
df['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
df['gap'] = df['Open'] / df['Close'].shift(1) - 1

print(f"After price features: {len(df.columns)} columns")

# Step 4: Add previous day context features
print("\n" + "=" * 60)
print("STEP 4: Adding previous day context features...")
print("=" * 60)

# 288 bars = 1 day of 5-min bars
df['prev_close'] = df['Close'].shift(288)
df['prev_high'] = df['High'].rolling(288).max().shift(1)
df['prev_low'] = df['Low'].rolling(288).min().shift(1)
df['prev_volume'] = df['Volume'].rolling(288).sum().shift(1)
df['prev_day_return'] = df['Close'].pct_change(288)
df['prev_2day_return'] = df['Close'].pct_change(576)
df['prev_5day_return'] = df['Close'].pct_change(1440)
df['price_vs_prev_close'] = df['Close'] / df['prev_close'] - 1
df['price_vs_prev_high'] = df['Close'] / df['prev_high'] - 1
df['price_vs_prev_low'] = df['Close'] / df['prev_low'] - 1
df['volume_vs_prev'] = df['Volume'] / (df['prev_volume'] / 288 + 1)

print(f"After prev day features: {len(df.columns)} columns")

# Step 5: Add lagged indicator features
print("\n" + "=" * 60)
print("STEP 5: Adding lagged indicator features...")
print("=" * 60)

key_indicators = [
    'momentum_rsi', 'trend_macd', 'trend_macd_signal', 
    'trend_macd_diff', 'volatility_bbp', 'volatility_atr',
    'trend_adx', 'momentum_stoch', 'volume_obv', 'volume_mfi'
]

for indicator in key_indicators:
    if indicator in df.columns:
        for lag in [1, 2, 3, 5, 10, 20, 50]:
            df[f'{indicator}_lag{lag}'] = df[indicator].shift(lag)
        print(f"  Added lags for {indicator}")

print(f"After lagged features: {len(df.columns)} columns")

# Step 6: Add change features
print("\n" + "=" * 60)
print("STEP 6: Adding change features...")
print("=" * 60)

for indicator in ['momentum_rsi', 'trend_macd', 'trend_adx', 'volatility_atr']:
    if indicator in df.columns:
        df[f'{indicator}_change_1'] = df[indicator].diff(1)
        df[f'{indicator}_change_5'] = df[indicator].diff(5)
        print(f"  Added changes for {indicator}")

print(f"After change features: {len(df.columns)} columns")

# Step 7: Handle NaN and inf values
print("\n" + "=" * 60)
print("STEP 7: Handling NaN and inf values...")
print("=" * 60)

nan_before = df.isna().sum().sum()
df = df.fillna(0)
df = df.replace([np.inf, -np.inf], 0)
print(f"Filled {nan_before:,} NaN values")

# Step 8: Verify all expected features are present
print("\n" + "=" * 60)
print("STEP 8: Verifying features...")
print("=" * 60)

available = [f for f in expected_features if f in df.columns]
missing = [f for f in expected_features if f not in df.columns]

print(f"Expected: {len(expected_features)}")
print(f"Available: {len(available)}")
print(f"Missing: {len(missing)}")

if missing:
    print(f"\nStill missing: {missing}")

# Step 9: Save the regenerated parquet
print("\n" + "=" * 60)
print("STEP 9: Saving regenerated parquet...")
print("=" * 60)

# Backup old file
if os.path.exists('data/BTC_features.parquet'):
    os.rename('data/BTC_features.parquet', 'data/BTC_features_backup_89cols.parquet')
    print("Backed up old file to BTC_features_backup_89cols.parquet")

# Save new file
df.to_parquet('data/BTC_features.parquet')
print(f"Saved new parquet: {len(df):,} rows, {len(df.columns)} columns")

# Final verification
print("\n" + "=" * 60)
print("FINAL VERIFICATION")
print("=" * 60)

df_check = pd.read_parquet('data/BTC_features.parquet')
print(f"Verified: {len(df_check):,} rows, {len(df_check.columns)} columns")
print(f"Date range: {df_check.index[0]} to {df_check.index[-1]}")

available_final = [f for f in expected_features if f in df_check.columns]
print(f"Expected features available: {len(available_final)}/{len(expected_features)}")

print(f"\nCompleted: {datetime.now()}")
print("=" * 80)
