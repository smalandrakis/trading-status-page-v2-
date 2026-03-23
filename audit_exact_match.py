#!/usr/bin/env python3
"""
EXACT MATCH: Replicate live bot's exact data window and offsets at 15:46 UTC.
This should produce identical signals to live.
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import joblib
import json
import warnings
import re
import ta
warnings.filterwarnings('ignore')

print("="*70)
print("EXACT MATCH: Replicating Live Bot at 15:46 UTC (16:46 CET)")
print("="*70)

BINANCE_API = "https://api.binance.com/api/v3"

def get_binance_klines(symbol='BTCUSDT', interval='5m', limit=500, end_time=None):
    """Fetch klines from Binance API."""
    url = f"{BINANCE_API}/klines"
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
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

# Live bot offsets at 15:46:40 UTC (from log)
LIVE_OFFSETS_AT_1546 = {
    "volume_adi": 2691352.38,  # hist_end=2691358.22, live_first=5.84
    "volume_obv": -631751.89,  # hist_end=-631706.40, live_first=45.49
    "volume_nvi": 27058486522.49,  # hist_end=27058487522.49, live_first=1000.00
    "volume_vpt": -11056.59,  # hist_end=-11056.59, live_first=0.00
    "volume_em": -114668782648.80,
    "volume_sma_em": -17174542815.06,
    "others_cr": 488.97,
}

def add_features_with_exact_offsets(df, offsets):
    """Calculate features with EXACT offsets from live bot."""
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
    
    # Apply EXACT offsets from live bot
    for base_feat, offset in offsets.items():
        if base_feat in df.columns:
            df[base_feat] = df[base_feat] + offset
        for lag in [1, 2, 3, 5, 10, 20, 50]:
            lag_feat = f'{base_feat}_lag{lag}'
            if lag_feat in df.columns:
                df[lag_feat] = df[lag_feat] + offset
    
    return df

# Load models
print("\n1. LOADING MODELS")
print("-"*50)

model_dir = "models_btc_v2"
models = {}
model_names = ['2h_0.5pct', '4h_0.5pct', '2h_0.5pct_SHORT', '4h_0.5pct_SHORT']

for name in model_names:
    models[name] = joblib.load(f"{model_dir}/model_{name}.joblib")
    print(f"  Loaded {name}")

with open(f"{model_dir}/feature_columns.json", 'r') as f:
    feature_columns = json.load(f)

# Parse live signals
print("\n2. PARSING LIVE SIGNALS")
print("-"*50)

live_signals = []
with open('logs/btc_bot.log', 'r') as f:
    for line in f:
        if '2025-12-22' in line and 'BTC:' in line and 'Pos:' in line:
            match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - INFO - BTC: \$([0-9.]+) \| Pos: (\d+)/\d+ \| (.+)', line)
            if match:
                timestamp = datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')
                price = float(match.group(2))
                signals_str = match.group(4)
                
                probs = {}
                for part in signals_str.split(' | '):
                    if ':' in part and '%' in part:
                        model, pct = part.split(':')
                        probs[model.strip()] = float(pct.strip().replace('%', '')) / 100
                
                live_signals.append({
                    'timestamp': timestamp,
                    'price': price,
                    **probs
                })

live_df = pd.DataFrame(live_signals)
live_df['bar_time'] = live_df['timestamp'].dt.floor('5min')
live_5min = live_df.groupby('bar_time').last().reset_index()
print(f"Live signals: {len(live_5min)} unique 5-min bars")

# Test specific timestamps
print("\n3. TESTING SPECIFIC TIMESTAMPS")
print("-"*50)

# Test at 16:40 CET = 15:40 UTC (when trade triggered)
test_times_utc = [
    datetime(2025, 12, 22, 14, 40, 0),  # 15:40 CET
    datetime(2025, 12, 22, 15, 40, 0),  # 16:40 CET - trade time
    datetime(2025, 12, 22, 16, 0, 0),   # 17:00 CET
]

for test_time_utc in test_times_utc:
    print(f"\n--- Testing {test_time_utc} UTC ({(test_time_utc + timedelta(hours=1)).strftime('%H:%M')} CET) ---")
    
    # Download 500 bars ending at this time
    data = get_binance_klines(
        symbol='BTCUSDT',
        interval='5m',
        limit=500,
        end_time=test_time_utc + timedelta(minutes=5)
    )
    
    print(f"Data: {len(data)} bars, {data.index[0]} to {data.index[-1]}")
    print(f"Last close: ${data['Close'].iloc[-1]:.2f}")
    
    # Calculate features with exact offsets
    features = add_features_with_exact_offsets(data, LIVE_OFFSETS_AT_1546)
    
    # Get prediction for last bar
    available_features = [f for f in feature_columns if f in features.columns]
    X = features[available_features].iloc[-1:].values
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
    print(f"\nBacktest predictions:")
    for model_name, model in models.items():
        prob = model.predict_proba(X)[0][1]
        print(f"  {model_name}: {prob*100:.0f}%")
    
    # Find matching live signal
    cet_time = test_time_utc + timedelta(hours=1)
    live_match = live_5min[live_5min['bar_time'] == cet_time]
    if len(live_match) > 0:
        print(f"\nLive signals at {cet_time.strftime('%H:%M')} CET:")
        row = live_match.iloc[0]
        for model_name in models.keys():
            if model_name in row:
                print(f"  {model_name}: {row[model_name]*100:.0f}%")

# 4. Full comparison
print("\n" + "="*70)
print("4. FULL COMPARISON WITH EXACT OFFSETS")
print("="*70)

# Download data for full Dec 22
end_time = datetime(2025, 12, 22, 22, 0, 0)  # 23:00 CET
data = get_binance_klines(symbol='BTCUSDT', interval='5m', limit=500, end_time=end_time)
features = add_features_with_exact_offsets(data, LIVE_OFFSETS_AT_1546)

# Generate predictions for each bar
available_features = [f for f in feature_columns if f in features.columns]
dec22 = datetime(2025, 12, 22)
dec22_features = features[features.index.date == dec22.date()]

backtest_signals = []
for idx, row in dec22_features.iterrows():
    X = row[available_features].values.reshape(1, -1)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
    probs = {}
    for model_name, model in models.items():
        probs[model_name] = model.predict_proba(X)[0][1]
    
    backtest_signals.append({
        'timestamp': idx,
        'bar_time': idx + timedelta(hours=1),  # UTC to CET
        'price': row['Close'],
        **probs
    })

backtest_df = pd.DataFrame(backtest_signals)

# Merge with live
comparison = pd.merge(
    live_5min[['bar_time', 'price', '2h_0.5pct', '4h_0.5pct', '2h_0.5pct_SHORT', '4h_0.5pct_SHORT']],
    backtest_df[['bar_time', 'price', '2h_0.5pct', '4h_0.5pct', '2h_0.5pct_SHORT', '4h_0.5pct_SHORT']],
    on='bar_time',
    suffixes=('_live', '_bt'),
    how='inner'
)

print(f"\nMatched bars: {len(comparison)}")

# Statistics
print("\nSTATISTICS:")
for model in ['2h_0.5pct', '4h_0.5pct', '2h_0.5pct_SHORT', '4h_0.5pct_SHORT']:
    live_col = f'{model}_live'
    bt_col = f'{model}_bt'
    diff = (comparison[live_col] - comparison[bt_col]) * 100
    corr = comparison[live_col].corr(comparison[bt_col])
    print(f"  {model}: Avg diff: {diff.mean():+.1f}%, Std: {diff.std():.1f}%, Corr: {corr:.3f}")

# Show sample comparison
print("\nSAMPLE (16:30-17:00 CET - around trade time):")
print("-"*80)
mask = (comparison['bar_time'] >= datetime(2025, 12, 22, 16, 30)) & (comparison['bar_time'] <= datetime(2025, 12, 22, 17, 0))
for _, row in comparison[mask].iterrows():
    time_str = row['bar_time'].strftime('%H:%M')
    print(f"{time_str}: 2h_L: {row['2h_0.5pct_live']*100:.0f}%/{row['2h_0.5pct_bt']*100:.0f}% | 4h_L: {row['4h_0.5pct_live']*100:.0f}%/{row['4h_0.5pct_bt']*100:.0f}% | 2h_S: {row['2h_0.5pct_SHORT_live']*100:.0f}%/{row['2h_0.5pct_SHORT_bt']*100:.0f}% | 4h_S: {row['4h_0.5pct_SHORT_live']*100:.0f}%/{row['4h_0.5pct_SHORT_bt']*100:.0f}%")
