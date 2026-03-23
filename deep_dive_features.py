#!/usr/bin/env python3
"""
Deep dive: Column-by-column feature comparison between live and backtest.
Find the exact discrepancy causing signal differences.
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import joblib
import json
import warnings
import ta
warnings.filterwarnings('ignore')

print("="*70)
print("DEEP DIVE: Feature-by-Feature Comparison")
print("="*70)

# Historical end values for cumulative features
CUMULATIVE_HIST_END = {
    "volume_adi": 2691358.22,
    "volume_obv": -631706.40,
    "volume_nvi": 27058487522.49,
    "volume_vpt": -11056.59,
    "volume_em": -114668782648.80,
    "volume_sma_em": -17174542815.06,
    "others_cr": 488.97,
}

BINANCE_API = "https://api.binance.com/api/v3"

def get_binance_klines(symbol='BTCUSDT', interval='5m', limit=1000, start_time=None, end_time=None):
    """Fetch klines from Binance API."""
    url = f"{BINANCE_API}/klines"
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    if start_time:
        params['startTime'] = int(start_time.timestamp() * 1000)
    if end_time:
        params['endTime'] = int(end_time.timestamp() * 1000)
    
    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()
    
    df = pd.DataFrame(data, columns=[
        'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = df[col].astype(float)
    
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]

# 1. Download Binance data (same as live bot)
print("\n1. DOWNLOADING BINANCE DATA")
print("-"*50)

# Target time: 16:40 CET = 15:40 UTC (when the trade triggered)
target_time_utc = datetime(2025, 12, 22, 15, 40, 0)
end_time = target_time_utc + timedelta(minutes=5)
start_time = target_time_utc - timedelta(days=5)  # Need history for features

all_data = []
current_start = start_time

while current_start < end_time:
    chunk = get_binance_klines(
        symbol='BTCUSDT',
        interval='5m',
        limit=1000,
        start_time=current_start,
        end_time=end_time
    )
    if len(chunk) == 0:
        break
    all_data.append(chunk)
    current_start = chunk.index[-1] + timedelta(minutes=5)

binance_data = pd.concat(all_data)
binance_data = binance_data[~binance_data.index.duplicated(keep='last')]
binance_data = binance_data.sort_index()

print(f"Downloaded {len(binance_data)} bars")
print(f"Data range: {binance_data.index[0]} to {binance_data.index[-1]}")

# Find closest bar to target time
closest_idx = binance_data.index.get_indexer([target_time_utc], method='nearest')[0]
target_time_actual = binance_data.index[closest_idx]
print(f"Target bar: {target_time_actual}, Close=${binance_data.iloc[closest_idx]['Close']:.2f}")

# 2. Load models and feature columns
print("\n2. LOADING MODELS")
print("-"*50)

model_dir = "models_btc_v2"
models = {}
model_names = ['2h_0.5pct', '4h_0.5pct', '2h_0.5pct_SHORT', '4h_0.5pct_SHORT']

for name in model_names:
    models[name] = joblib.load(f"{model_dir}/model_{name}.joblib")
    print(f"  Loaded {name}")

with open(f"{model_dir}/feature_columns.json", 'r') as f:
    feature_columns = json.load(f)
print(f"Feature columns: {len(feature_columns)}")

# 3. Calculate features EXACTLY as live bot does
print("\n3. CALCULATING FEATURES (Backtest method)")
print("-"*50)

def add_features_btc_style(df):
    """Replicate btc_ensemble_bot.py add_features() exactly."""
    if len(df) < 50:
        return df
    
    df = df.copy()
    
    df = ta.add_all_ta_features(
        df, open="Open", high="High", low="Low", 
        close="Close", volume="Volume", fillna=True
    )
    
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['time_of_day'] = df['hour'] + df['minute'] / 60
    
    df['is_premarket'] = 0
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
    
    df['return_1bar'] = df['Close'].pct_change()
    df['return_5bar'] = df['Close'].pct_change(5)
    df['return_10bar'] = df['Close'].pct_change(10)
    df['return_20bar'] = df['Close'].pct_change(20)
    df['log_return_1bar'] = np.log(df['Close'] / df['Close'].shift(1))
    
    for window in [5, 10, 20, 50]:
        ma = df['Close'].rolling(window).mean()
        df[f'price_to_ma{window}'] = df['Close'] / ma - 1
    
    for window in [5, 10, 20]:
        df[f'volatility_{window}bar'] = df['return_1bar'].rolling(window).std()
    
    df['bar_range'] = (df['High'] - df['Low']) / df['Close']
    df['bar_range_ma5'] = df['bar_range'].rolling(5).mean()
    df['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['gap'] = df['Open'] / df['Close'].shift(1) - 1
    
    df['trade_date'] = df.index.date
    daily = df.groupby('trade_date').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 
        'Close': 'last', 'Volume': 'sum'
    })
    daily['prev_close'] = daily['Close'].shift(1)
    daily['prev_high'] = daily['High'].shift(1)
    daily['prev_low'] = daily['Low'].shift(1)
    daily['prev_volume'] = daily['Volume'].shift(1)
    daily['prev_day_return'] = daily['Close'].pct_change()
    daily['prev_2day_return'] = daily['Close'].pct_change(2)
    daily['prev_5day_return'] = daily['Close'].pct_change(5)
    
    df = df.merge(
        daily[['prev_close', 'prev_high', 'prev_low', 'prev_volume',
               'prev_day_return', 'prev_2day_return', 'prev_5day_return']], 
        left_on='trade_date', right_index=True, how='left'
    )
    
    df['price_vs_prev_close'] = df['Close'] / df['prev_close'] - 1
    df['price_vs_prev_high'] = df['Close'] / df['prev_high'] - 1
    df['price_vs_prev_low'] = df['Close'] / df['prev_low'] - 1
    df['volume_vs_prev'] = df['Volume'] / (df['prev_volume'] / 288 + 1)
    
    df = df.drop(columns=['trade_date'], errors='ignore')
    
    key_indicators = ['momentum_rsi', 'trend_macd', 'trend_macd_signal', 
                    'trend_macd_diff', 'volatility_bbp', 'volatility_atr',
                    'trend_adx', 'momentum_stoch', 'volume_obv', 'volume_mfi']
    LOOKBACK_PERIODS = [1, 2, 3, 5, 10, 20, 50]
    for indicator in key_indicators:
        if indicator in df.columns:
            for lag in LOOKBACK_PERIODS:
                df[f'{indicator}_lag{lag}'] = df[indicator].shift(lag)
    
    change_indicators = ['momentum_rsi', 'trend_macd', 'trend_adx', 'volatility_atr']
    for indicator in change_indicators:
        if indicator in df.columns:
            df[f'{indicator}_change_1'] = df[indicator].diff(1)
            df[f'{indicator}_change_5'] = df[indicator].diff(5)
    
    df = df.fillna(0)
    df = df.replace([np.inf, -np.inf], 0)
    
    return df

# Calculate features
features_df = add_features_btc_style(binance_data)
print(f"Features calculated: {len(features_df.columns)} columns")

# Get the target row
target_row = features_df.iloc[closest_idx]
print(f"Target bar features extracted")

# 4. Now simulate what LIVE bot does
print("\n4. SIMULATING LIVE BOT FEATURE CALCULATION")
print("-"*50)

# Live bot fetches 500 bars and calculates features
# Let's get exactly 500 bars ending at target time
live_end_time = target_time_utc + timedelta(minutes=5)
live_start_time = live_end_time - timedelta(minutes=500*5)

live_data = get_binance_klines(
    symbol='BTCUSDT',
    interval='5m',
    limit=500,
    end_time=live_end_time
)
print(f"Live simulation: {len(live_data)} bars")
print(f"Live data range: {live_data.index[0]} to {live_data.index[-1]}")

# Calculate features for live simulation
live_features_df = add_features_btc_style(live_data)

# Apply cumulative offsets (as live bot does)
cumulative_offsets = {}
for feat, hist_end in CUMULATIVE_HIST_END.items():
    if feat in live_features_df.columns:
        live_first = live_features_df[feat].iloc[0]
        cumulative_offsets[feat] = hist_end - live_first

for base_feat, offset in cumulative_offsets.items():
    if base_feat in live_features_df.columns:
        live_features_df[base_feat] = live_features_df[base_feat] + offset
    for lag in [1, 2, 3, 5, 10, 20, 50]:
        lag_feat = f'{base_feat}_lag{lag}'
        if lag_feat in live_features_df.columns:
            live_features_df[lag_feat] = live_features_df[lag_feat] + offset

# Get the target row from live simulation (last row)
live_target_row = live_features_df.iloc[-1]
print(f"Live simulation features extracted (last bar: {live_features_df.index[-1]})")

# 5. Compare features
print("\n5. FEATURE COMPARISON AT 15:40 UTC (16:40 CET)")
print("-"*50)

available_features = [f for f in feature_columns if f in features_df.columns and f in live_features_df.columns]
print(f"Comparing {len(available_features)} features")

# Find features with significant differences
differences = []
for feat in available_features:
    bt_val = target_row[feat] if feat in target_row else 0
    live_val = live_target_row[feat] if feat in live_target_row else 0
    
    if bt_val != 0:
        pct_diff = abs(live_val - bt_val) / abs(bt_val) * 100
    else:
        pct_diff = abs(live_val - bt_val) * 100
    
    differences.append({
        'feature': feat,
        'backtest': bt_val,
        'live_sim': live_val,
        'abs_diff': abs(live_val - bt_val),
        'pct_diff': pct_diff
    })

diff_df = pd.DataFrame(differences)
diff_df = diff_df.sort_values('pct_diff', ascending=False)

print("\nTOP 30 FEATURES WITH LARGEST DIFFERENCES:")
print("-"*80)
print(f"{'Feature':<40} {'Backtest':>12} {'Live Sim':>12} {'Diff':>10} {'%Diff':>8}")
print("-"*80)

for _, row in diff_df.head(30).iterrows():
    print(f"{row['feature']:<40} {row['backtest']:>12.4f} {row['live_sim']:>12.4f} {row['abs_diff']:>10.4f} {row['pct_diff']:>7.1f}%")

# 6. Get predictions from both
print("\n6. PREDICTION COMPARISON")
print("-"*50)

# Backtest prediction (without cumulative offsets)
bt_X = target_row[available_features].values.reshape(1, -1)
bt_X = np.nan_to_num(bt_X, nan=0, posinf=0, neginf=0)

# Live simulation prediction (with cumulative offsets)
live_X = live_target_row[available_features].values.reshape(1, -1)
live_X = np.nan_to_num(live_X, nan=0, posinf=0, neginf=0)

print(f"{'Model':<20} {'Backtest':>12} {'Live Sim':>12} {'Diff':>10}")
print("-"*60)

for model_name, model in models.items():
    bt_prob = model.predict_proba(bt_X)[0][1]
    live_prob = model.predict_proba(live_X)[0][1]
    diff = live_prob - bt_prob
    print(f"{model_name:<20} {bt_prob*100:>11.1f}% {live_prob*100:>11.1f}% {diff*100:>+9.1f}%")

# 7. Check cumulative feature impact
print("\n7. CUMULATIVE FEATURE IMPACT ANALYSIS")
print("-"*50)

# Calculate predictions with and without cumulative offsets
features_no_offset = add_features_btc_style(live_data)
features_with_offset = features_no_offset.copy()

# Apply offsets
for base_feat, offset in cumulative_offsets.items():
    if base_feat in features_with_offset.columns:
        features_with_offset[base_feat] = features_with_offset[base_feat] + offset
    for lag in [1, 2, 3, 5, 10, 20, 50]:
        lag_feat = f'{base_feat}_lag{lag}'
        if lag_feat in features_with_offset.columns:
            features_with_offset[lag_feat] = features_with_offset[lag_feat] + offset

no_offset_row = features_no_offset.iloc[-1]
with_offset_row = features_with_offset.iloc[-1]

no_offset_X = no_offset_row[available_features].values.reshape(1, -1)
no_offset_X = np.nan_to_num(no_offset_X, nan=0, posinf=0, neginf=0)

with_offset_X = with_offset_row[available_features].values.reshape(1, -1)
with_offset_X = np.nan_to_num(with_offset_X, nan=0, posinf=0, neginf=0)

print(f"{'Model':<20} {'No Offset':>12} {'With Offset':>12} {'Impact':>10}")
print("-"*60)

for model_name, model in models.items():
    no_offset_prob = model.predict_proba(no_offset_X)[0][1]
    with_offset_prob = model.predict_proba(with_offset_X)[0][1]
    impact = with_offset_prob - no_offset_prob
    print(f"{model_name:<20} {no_offset_prob*100:>11.1f}% {with_offset_prob*100:>11.1f}% {impact*100:>+9.1f}%")

# 8. Check what the ACTUAL live bot logged
print("\n8. ACTUAL LIVE BOT SIGNALS AT 16:40 CET")
print("-"*50)

import re
with open('logs/btc_bot.log', 'r') as f:
    for line in f:
        if '2025-12-22 16:40' in line and 'BTC:' in line and 'Pos:' in line:
            print(line.strip())
            break

print("\n9. SUMMARY")
print("="*70)
print("""
The key differences between backtest and live are:

1. DATA WINDOW: 
   - Backtest uses all historical data (5+ days)
   - Live bot fetches only 500 bars (~1.7 days)
   
2. CUMULATIVE OFFSETS:
   - Live bot applies offsets to volume_adi, volume_obv, etc.
   - These offsets depend on when the bot started
   
3. FEATURE CALCULATION:
   - Both use the same ta.add_all_ta_features()
   - But different data windows affect rolling calculations
""")
