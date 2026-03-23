#!/usr/bin/env python3
"""
FINAL AUDIT: Match live bot EXACTLY
- 300 bars (not 500)
- Features calculated on bar close
- Exact cumulative offsets from live bot log
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
print("FINAL AUDIT: Exact Live Bot Replication")
print("Using 300 bars (same as live bot)")
print("="*70)

BINANCE_API = "https://api.binance.com/api/v3"

# Live bot offsets at 15:46:40 UTC
LIVE_OFFSETS = {
    "volume_adi": 2691352.38,
    "volume_obv": -631751.89,
    "volume_nvi": 27058486522.49,
    "volume_vpt": -11056.59,
    "volume_em": -114668782648.80,
    "volume_sma_em": -17174542815.06,
    "others_cr": 488.97,
}

def get_binance_klines(symbol='BTCUSDT', interval='5m', limit=300, end_time=None):
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

def add_features(df, offsets):
    """Exact replication of btc_ensemble_bot.add_features()"""
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
    
    # Apply offsets
    for base_feat, offset in offsets.items():
        if base_feat in df.columns:
            df[base_feat] = df[base_feat] + offset
        for lag in [1, 2, 3, 5, 10, 20, 50]:
            lag_feat = f'{base_feat}_lag{lag}'
            if lag_feat in df.columns:
                df[lag_feat] = df[lag_feat] + offset
    
    return df

# Load models
model_dir = "models_btc_v2"
models = {}
for name in ['2h_0.5pct', '4h_0.5pct', '2h_0.5pct_SHORT', '4h_0.5pct_SHORT']:
    models[name] = joblib.load(f"{model_dir}/model_{name}.joblib")

with open(f"{model_dir}/feature_columns.json", 'r') as f:
    feature_columns = json.load(f)

# Parse live signals
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
                
                live_signals.append({'timestamp': timestamp, 'price': price, **probs})

live_df = pd.DataFrame(live_signals)
live_df['bar_time'] = live_df['timestamp'].dt.floor('5min')
live_5min = live_df.groupby('bar_time').first().reset_index()  # First signal of each bar

print(f"\nLive signals: {len(live_5min)} bars")

# Test at specific bar times (when new bar was detected)
print("\n" + "="*70)
print("BAR-BY-BAR COMPARISON")
print("="*70)

# Find "New bar" log entries to get exact bar times
new_bar_times = []
with open('logs/btc_bot.log', 'r') as f:
    for line in f:
        if '2025-12-22' in line and 'New bar:' in line:
            match = re.search(r'New bar: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            if match:
                bar_time = datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')
                new_bar_times.append(bar_time)

print(f"Found {len(new_bar_times)} new bar events in log")

# For each new bar, replicate what live bot did
results = []
for bar_time_utc in new_bar_times[-20:]:  # Last 20 bars
    # Download 300 bars ending at this bar time
    end_time = bar_time_utc + timedelta(minutes=5)
    data = get_binance_klines(limit=300, end_time=end_time)
    
    if len(data) < 50:
        continue
    
    # Calculate features
    features = add_features(data, LIVE_OFFSETS)
    
    # Get prediction
    available_features = [f for f in feature_columns if f in features.columns]
    X = features[available_features].iloc[-1:].values
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
    bt_probs = {}
    for model_name, model in models.items():
        bt_probs[model_name] = model.predict_proba(X)[0][1]
    
    # Find matching live signal
    bar_time_cet = bar_time_utc + timedelta(hours=1)
    live_match = live_5min[live_5min['bar_time'] == bar_time_cet]
    
    if len(live_match) > 0:
        live_row = live_match.iloc[0]
        results.append({
            'bar_time': bar_time_cet,
            'price_bt': data['Close'].iloc[-1],
            'price_live': live_row['price'],
            '2h_L_bt': bt_probs['2h_0.5pct'],
            '2h_L_live': live_row.get('2h_0.5pct', 0),
            '4h_L_bt': bt_probs['4h_0.5pct'],
            '4h_L_live': live_row.get('4h_0.5pct', 0),
            '2h_S_bt': bt_probs['2h_0.5pct_SHORT'],
            '2h_S_live': live_row.get('2h_0.5pct_SHORT', 0),
            '4h_S_bt': bt_probs['4h_0.5pct_SHORT'],
            '4h_S_live': live_row.get('4h_0.5pct_SHORT', 0),
        })

results_df = pd.DataFrame(results)

if len(results_df) > 0:
    print(f"\nMatched {len(results_df)} bars")
    print("\nDETAILED COMPARISON:")
    print("-"*110)
    print(f"{'Time':<8} {'Price':<18} {'2h LONG':<16} {'4h LONG':<16} {'2h SHORT':<16} {'4h SHORT':<16}")
    print(f"{'':8} {'BT':>8} {'Live':>8} {'BT':>6} {'Live':>6} {'Δ':>3} {'BT':>6} {'Live':>6} {'Δ':>3} {'BT':>6} {'Live':>6} {'Δ':>3} {'BT':>6} {'Live':>6} {'Δ':>3}")
    print("-"*110)
    
    for _, row in results_df.iterrows():
        t = row['bar_time'].strftime('%H:%M')
        print(f"{t:<8} ${row['price_bt']:<7.0f} ${row['price_live']:<7.0f} "
              f"{row['2h_L_bt']*100:>5.0f}% {row['2h_L_live']*100:>5.0f}% {(row['2h_L_live']-row['2h_L_bt'])*100:>+3.0f} "
              f"{row['4h_L_bt']*100:>5.0f}% {row['4h_L_live']*100:>5.0f}% {(row['4h_L_live']-row['4h_L_bt'])*100:>+3.0f} "
              f"{row['2h_S_bt']*100:>5.0f}% {row['2h_S_live']*100:>5.0f}% {(row['2h_S_live']-row['2h_S_bt'])*100:>+3.0f} "
              f"{row['4h_S_bt']*100:>5.0f}% {row['4h_S_live']*100:>5.0f}% {(row['4h_S_live']-row['4h_S_bt'])*100:>+3.0f}")
    
    print("\n" + "-"*110)
    print("STATISTICS:")
    for model, bt_col, live_col in [
        ('2h_0.5pct', '2h_L_bt', '2h_L_live'),
        ('4h_0.5pct', '4h_L_bt', '4h_L_live'),
        ('2h_0.5pct_SHORT', '2h_S_bt', '2h_S_live'),
        ('4h_0.5pct_SHORT', '4h_S_bt', '4h_S_live'),
    ]:
        diff = (results_df[live_col] - results_df[bt_col]) * 100
        corr = results_df[live_col].corr(results_df[bt_col])
        print(f"  {model}: Avg diff: {diff.mean():+.1f}%, Std: {diff.std():.1f}%, Corr: {corr:.3f}")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
