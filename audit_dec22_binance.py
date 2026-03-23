#!/usr/bin/env python3
"""
Audit Dec 22, 2025 BTC signals using BINANCE data (same source as live bot).
This should give us an accurate comparison.
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
print("AUDIT: Dec 22, 2025 BTC Signals - Using BINANCE Data")
print("="*70)

# Historical end values for cumulative features (EXACT values from btc_ensemble_bot.py)
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
    """Fetch klines from Binance API (same as live bot)."""
    url = f"{BINANCE_API}/klines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
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

# 1. Download BTC data from Binance
print("\n1. DOWNLOADING BINANCE BTCUSDT DATA")
print("-"*50)

# Get enough history for feature calculation (need ~500 bars for indicators)
# Dec 22 = today, so get last 7 days
end_time = datetime(2025, 12, 22, 23, 59, 59)
start_time = end_time - timedelta(days=7)

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
    print(f"  Downloaded {len(chunk)} bars ending at {chunk.index[-1]}")

binance_data = pd.concat(all_data)
binance_data = binance_data[~binance_data.index.duplicated(keep='last')]
binance_data = binance_data.sort_index()

print(f"\nTotal Binance data: {len(binance_data)} bars")
print(f"From: {binance_data.index[0]}")
print(f"To: {binance_data.index[-1]}")

# Filter to Dec 22
dec22 = datetime(2025, 12, 22)
dec22_data = binance_data[binance_data.index.date == dec22.date()]
print(f"Dec 22 bars: {len(dec22_data)}")
print(f"Dec 22 price range: ${dec22_data['Low'].min():.2f} - ${dec22_data['High'].max():.2f}")

# 2. Parse live signals from log
print("\n2. PARSING LIVE SIGNALS FROM LOG")
print("-"*50)

live_signals = []
with open('logs/btc_bot.log', 'r') as f:
    for line in f:
        if '2025-12-22' in line and 'BTC:' in line and 'Pos:' in line:
            match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - INFO - BTC: \$([0-9.]+) \| Pos: (\d+)/\d+ \| (.+)', line)
            if match:
                timestamp = datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')
                price = float(match.group(2))
                positions = int(match.group(3))
                signals_str = match.group(4)
                
                probs = {}
                for part in signals_str.split(' | '):
                    if ':' in part and '%' in part:
                        model, pct = part.split(':')
                        probs[model.strip()] = float(pct.strip().replace('%', '')) / 100
                
                live_signals.append({
                    'timestamp': timestamp,
                    'price': price,
                    'positions': positions,
                    **probs
                })

live_df = pd.DataFrame(live_signals)
print(f"Parsed {len(live_df)} live signal logs from Dec 22")

live_df['bar_time'] = live_df['timestamp'].dt.floor('5min')
live_5min = live_df.groupby('bar_time').last().reset_index()
print(f"Unique 5-min bars with signals: {len(live_5min)}")

# 3. Load models
print("\n3. LOADING MODELS")
print("-"*50)

model_dir = "models_btc_v2"
models = {}
model_names = ['2h_0.5pct', '4h_0.5pct', '2h_0.5pct_SHORT', '4h_0.5pct_SHORT']

for name in model_names:
    try:
        model_path = f"{model_dir}/model_{name}.joblib"
        models[name] = joblib.load(model_path)
        print(f"  Loaded {name}")
    except Exception as e:
        print(f"  Failed to load {name}: {e}")

with open(f"{model_dir}/feature_columns.json", 'r') as f:
    feature_columns = json.load(f)
print(f"Feature columns required: {len(feature_columns)}")

# 4. Calculate features (same as btc_ensemble_bot.py)
print("\n4. CALCULATING FEATURES (Binance data)")
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
    
    # Apply cumulative feature offsets
    cumulative_offsets = {}
    for feat, hist_end in CUMULATIVE_HIST_END.items():
        if feat in df.columns:
            live_first = df[feat].iloc[0]
            cumulative_offsets[feat] = hist_end - live_first
    
    for base_feat, offset in cumulative_offsets.items():
        if base_feat in df.columns:
            df[base_feat] = df[base_feat] + offset
        for lag in [1, 2, 3, 5, 10, 20, 50]:
            lag_feat = f'{base_feat}_lag{lag}'
            if lag_feat in df.columns:
                df[lag_feat] = df[lag_feat] + offset
    
    return df

print("Calculating features...")
features_df = add_features_btc_style(binance_data)
print(f"Features calculated: {len(features_df.columns)} columns")

available_features = [f for f in feature_columns if f in features_df.columns]
print(f"Available features: {len(available_features)}/{len(feature_columns)}")

# 5. Generate backtest signals for Dec 22
print("\n5. GENERATING BACKTEST SIGNALS (Binance data)")
print("-"*50)

backtest_signals = []
dec22_features = features_df[features_df.index.date == dec22.date()]
print(f"Dec 22 bars with features: {len(dec22_features)}")

for idx, row in dec22_features.iterrows():
    try:
        X = row[available_features].values.reshape(1, -1)
        if np.isnan(X).any():
            continue
        
        probs = {}
        for model_name, model in models.items():
            prob = model.predict_proba(X)[0][1]
            probs[model_name] = prob
        
        backtest_signals.append({
            'timestamp': idx,
            'price': row['Close'],
            **probs
        })
    except Exception as e:
        pass

backtest_df = pd.DataFrame(backtest_signals)
print(f"Backtest signals generated: {len(backtest_df)}")

# 6. Compare live vs backtest (Binance)
print("\n6. LIVE vs BACKTEST COMPARISON (Binance data)")
print("-"*50)

if len(backtest_df) > 0 and len(live_5min) > 0:
    backtest_df['bar_time'] = backtest_df['timestamp'].dt.floor('5min')
    
    # Adjust for timezone - Binance is UTC, live logs are local time (CET = UTC+1)
    # Live timestamps are in CET, Binance timestamps are in UTC
    # So live 16:40 CET = Binance 15:40 UTC
    backtest_df['bar_time_cet'] = backtest_df['bar_time'] + timedelta(hours=1)
    
    comparison = pd.merge(
        live_5min[['bar_time', 'price', '2h_0.5pct', '4h_0.5pct', '2h_0.5pct_SHORT', '4h_0.5pct_SHORT']],
        backtest_df[['bar_time_cet', 'price', '2h_0.5pct', '4h_0.5pct', '2h_0.5pct_SHORT', '4h_0.5pct_SHORT']],
        left_on='bar_time',
        right_on='bar_time_cet',
        suffixes=('_live', '_bt'),
        how='inner'
    )
    
    print(f"Matched bars: {len(comparison)}")
    
    if len(comparison) > 0:
        print("\nDETAILED COMPARISON:")
        print("-"*100)
        print(f"{'Time':<8} {'Price L':<10} {'Price BT':<10} {'2h LONG':<18} {'4h LONG':<18} {'2h SHORT':<18} {'4h SHORT':<18}")
        print(f"{'':8} {'':10} {'':10} {'L':>6} {'BT':>6} {'Δ':>4} {'L':>6} {'BT':>6} {'Δ':>4} {'L':>6} {'BT':>6} {'Δ':>4} {'L':>6} {'BT':>6} {'Δ':>4}")
        print("-"*100)
        
        for _, row in comparison.iterrows():
            time_str = row['bar_time'].strftime('%H:%M')
            price_l = row['price_live']
            price_bt = row['price_bt']
            
            def fmt(live, bt):
                l = live * 100 if pd.notna(live) else 0
                b = bt * 100 if pd.notna(bt) else 0
                d = l - b
                return f"{l:>5.0f}% {b:>5.0f}% {d:>+3.0f}"
            
            print(f"{time_str:<8} ${price_l:<9.0f} ${price_bt:<9.0f} {fmt(row['2h_0.5pct_live'], row['2h_0.5pct_bt'])} {fmt(row['4h_0.5pct_live'], row['4h_0.5pct_bt'])} {fmt(row.get('2h_0.5pct_SHORT_live', 0), row.get('2h_0.5pct_SHORT_bt', 0))} {fmt(row.get('4h_0.5pct_SHORT_live', 0), row.get('4h_0.5pct_SHORT_bt', 0))}")
        
        print("\n" + "-"*100)
        print("STATISTICS:")
        for model in ['2h_0.5pct', '4h_0.5pct', '2h_0.5pct_SHORT', '4h_0.5pct_SHORT']:
            live_col = f'{model}_live'
            bt_col = f'{model}_bt'
            if live_col in comparison.columns and bt_col in comparison.columns:
                diff = (comparison[live_col] - comparison[bt_col]) * 100
                corr = comparison[live_col].corr(comparison[bt_col])
                print(f"  {model}:")
                print(f"    Avg diff: {diff.mean():+.1f}%, Std: {diff.std():.1f}%, Correlation: {corr:.3f}")

# 7. Threshold crossing analysis
print("\n7. THRESHOLD CROSSING ANALYSIS (Binance backtest)")
print("-"*50)

THRESHOLD_LONG = 0.65
THRESHOLD_SHORT = 0.65

print(f"Thresholds: LONG >= {THRESHOLD_LONG:.0%}, SHORT >= {THRESHOLD_SHORT:.0%}")

print("\nBACKTEST signals that crossed threshold (Binance data):")
bt_triggered = 0
for _, row in backtest_df.iterrows():
    triggered = []
    if row.get('2h_0.5pct', 0) >= THRESHOLD_LONG:
        triggered.append(f"2h LONG: {row['2h_0.5pct']:.0%}")
    if row.get('4h_0.5pct', 0) >= THRESHOLD_LONG:
        triggered.append(f"4h LONG: {row['4h_0.5pct']:.0%}")
    if row.get('2h_0.5pct_SHORT', 0) >= THRESHOLD_SHORT:
        triggered.append(f"2h SHORT: {row['2h_0.5pct_SHORT']:.0%}")
    if row.get('4h_0.5pct_SHORT', 0) >= THRESHOLD_SHORT:
        triggered.append(f"4h SHORT: {row['4h_0.5pct_SHORT']:.0%}")
    
    if triggered:
        bt_triggered += 1
        # Convert UTC to CET for display
        cet_time = row['timestamp'] + timedelta(hours=1)
        print(f"  {cet_time.strftime('%H:%M')} CET (UTC {row['timestamp'].strftime('%H:%M')}) @ ${row['price']:.0f}: {', '.join(triggered)}")

print(f"\nTotal backtest threshold crossings: {bt_triggered}")

print("\nLIVE signals that crossed threshold:")
live_triggered = 0
for _, row in live_5min.iterrows():
    triggered = []
    if row.get('2h_0.5pct', 0) >= THRESHOLD_LONG:
        triggered.append(f"2h LONG: {row['2h_0.5pct']:.0%}")
    if row.get('4h_0.5pct', 0) >= THRESHOLD_LONG:
        triggered.append(f"4h LONG: {row['4h_0.5pct']:.0%}")
    if row.get('2h_0.5pct_SHORT', 0) >= THRESHOLD_SHORT:
        triggered.append(f"2h SHORT: {row['2h_0.5pct_SHORT']:.0%}")
    if row.get('4h_0.5pct_SHORT', 0) >= THRESHOLD_SHORT:
        triggered.append(f"4h SHORT: {row['4h_0.5pct_SHORT']:.0%}")
    
    if triggered:
        live_triggered += 1
        print(f"  {row['bar_time'].strftime('%H:%M')} CET @ ${row['price']:.0f}: {', '.join(triggered)}")

print(f"\nTotal live threshold crossings: {live_triggered}")

# 8. Summary
print("\n" + "="*70)
print("AUDIT SUMMARY (Using Binance Data)")
print("="*70)
print(f"""
DATA SOURCES:
- Live: Binance BTCUSDT spot (api.binance.com/api/v3)
- Backtest: Binance BTCUSDT spot (same source!)

This should give us ACCURATE comparison since both use the same data source.

SIGNAL COMPARISON:
- Matched bars: {len(comparison) if 'comparison' in dir() else 0}
- Live threshold crossings: {live_triggered}
- Backtest threshold crossings: {bt_triggered}

NOTE: MBT (Micro Bitcoin Futures) tracks CME CF Bitcoin Reference Rate
which is based on spot prices, so BTCUSDT spot is the correct proxy.
""")
