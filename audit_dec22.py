#!/usr/bin/env python3
"""
Audit Dec 22, 2025 BTC signals: Compare live vs backtest
Also validate feature calculation process.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import joblib
import json
import warnings
import re
warnings.filterwarnings('ignore')

print("="*70)
print("AUDIT: Dec 22, 2025 BTC Signals - Live vs Backtest")
print("="*70)

# 1. Load BTC parquet (this is what live bot uses as base)
print("\n1. LOADING DATA SOURCES")
print("-"*50)

btc_parquet = pd.read_parquet('data/BTC_features.parquet')
print(f"BTC Parquet: {len(btc_parquet)} bars, ends at {btc_parquet.index[-1]}")

# 2. Download today's BTC data from Yahoo/Binance
print("\nDownloading Dec 22 BTC data from Yahoo...")
btc = yf.Ticker('BTC-USD')
yahoo_data = btc.history(period='5d', interval='5m')

if yahoo_data.index.tz is not None:
    yahoo_data.index = yahoo_data.index.tz_localize(None)

dec22 = datetime(2025, 12, 22)
today_yahoo = yahoo_data[yahoo_data.index.date == dec22.date()]
print(f"Yahoo Dec 22 data: {len(today_yahoo)} bars")
print(f"  From: {today_yahoo.index[0]}")
print(f"  To: {today_yahoo.index[-1]}")
print(f"  Price range: ${today_yahoo['Low'].min():.2f} - ${today_yahoo['High'].max():.2f}")

# 3. Parse live signals from log
print("\n2. PARSING LIVE SIGNALS FROM LOG")
print("-"*50)

live_signals = []
with open('logs/btc_bot.log', 'r') as f:
    for line in f:
        if '2025-12-22' in line and 'BTC:' in line and 'Pos:' in line:
            # Parse: 2025-12-22 16:00:05,133 - INFO - BTC: $90153.41 | Pos: 0/4 | 2h_0.5pct:52% | 4h_0.5pct:49% | ...
            match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - INFO - BTC: \$([0-9.]+) \| Pos: (\d+)/\d+ \| (.+)', line)
            if match:
                timestamp = datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')
                price = float(match.group(2))
                positions = int(match.group(3))
                signals_str = match.group(4)
                
                # Parse individual model probabilities
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
print(f"Time range: {live_df['timestamp'].min()} to {live_df['timestamp'].max()}")

# Sample at 5-min intervals (to match backtest bars)
live_df['bar_time'] = live_df['timestamp'].dt.floor('5min')
live_5min = live_df.groupby('bar_time').last().reset_index()
print(f"Unique 5-min bars with signals: {len(live_5min)}")

# 4. Load models and feature columns
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

# 5. Check what features are in parquet
print("\n4. FEATURE AVAILABILITY CHECK")
print("-"*50)

parquet_cols = set(btc_parquet.columns)
required_cols = set(feature_columns)
missing_cols = required_cols - parquet_cols
extra_cols = parquet_cols - required_cols - {'Open', 'High', 'Low', 'Close', 'Volume'}

print(f"Parquet has {len(parquet_cols)} columns")
print(f"Models require {len(required_cols)} features")
print(f"Missing features: {len(missing_cols)}")
if missing_cols:
    print(f"  Missing: {list(missing_cols)[:10]}...")
print(f"Extra features in parquet: {len(extra_cols)}")

# Check feature availability
available_features = [f for f in feature_columns if f in parquet_cols]
print(f"Available features for prediction: {len(available_features)}/{len(feature_columns)}")

# 6. Simulate backtest on Dec 22 data
print("\n5. BACKTEST SIMULATION ON DEC 22")
print("-"*50)

# We need to combine parquet (for historical features) with today's OHLCV
# This mimics what the live bot does with synthetic bars

# Get last N bars from parquet for feature context
context_size = 500
parquet_context = btc_parquet.tail(context_size).copy()

# Prepare today's data
today_ohlcv = today_yahoo[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

print(f"Parquet context: {len(parquet_context)} bars ending at {parquet_context.index[-1]}")
print(f"Today's OHLCV: {len(today_ohlcv)} bars")

# The issue: parquet ends Dec 18, today is Dec 22
# Live bot uses parquet features + updates OHLCV with synthetic bar
# For backtest, we need to recalculate features for Dec 22

# Let's check how live bot calculates features
print("\n6. ANALYZING LIVE BOT FEATURE CALCULATION")
print("-"*50)

# Read the relevant code from btc_ensemble_bot.py
print("Live bot uses: get_features_with_synthetic_bar()")
print("  1. Loads parquet features (ends Dec 18)")
print("  2. Takes last row as base")
print("  3. Updates OHLCV with current synthetic bar")
print("  4. Recalculates price-dependent features")
print("")
print("KEY INSIGHT: Live bot uses Dec 18 features as base,")
print("only updating OHLCV and price-dependent indicators!")

# 7. Compare probabilities
print("\n7. PROBABILITY COMPARISON")
print("-"*50)

# Get the last row of parquet features (what live bot uses as base)
base_features = btc_parquet.iloc[-1:].copy()
print(f"Base features from: {base_features.index[0]}")
print(f"Base Close price: ${base_features['Close'].values[0]:.2f}")

# For each 5-min bar today, simulate what backtest would predict
backtest_signals = []

for idx, row in today_ohlcv.iterrows():
    # Create feature row by copying base and updating OHLCV
    features = base_features.copy()
    features.index = [idx]
    features['Open'] = row['Open']
    features['High'] = row['High']
    features['Low'] = row['Low']
    features['Close'] = row['Close']
    features['Volume'] = row['Volume']
    
    # Update price-dependent features (simplified - matching live bot logic)
    # These are the key features that change with price
    close = row['Close']
    base_close = btc_parquet['Close'].iloc[-1]
    
    # Update returns
    features['returns'] = (close - base_close) / base_close
    features['log_returns'] = np.log(close / base_close)
    
    # Get predictions
    probs = {}
    try:
        X = features[available_features].values
        if not np.isnan(X).any():
            for model_name, model in models.items():
                prob = model.predict_proba(X)[0][1]
                probs[model_name] = prob
    except Exception as e:
        pass
    
    if probs:
        backtest_signals.append({
            'timestamp': idx,
            'price': close,
            **probs
        })

backtest_df = pd.DataFrame(backtest_signals)
print(f"Backtest signals generated: {len(backtest_df)}")

# 8. Compare live vs backtest at matching timestamps
print("\n8. LIVE vs BACKTEST COMPARISON")
print("-"*50)

# Merge on bar time
if len(backtest_df) > 0 and len(live_5min) > 0:
    backtest_df['bar_time'] = backtest_df['timestamp'].dt.floor('5min')
    
    comparison = pd.merge(
        live_5min[['bar_time', 'price', '2h_0.5pct', '4h_0.5pct', '2h_0.5pct_SHORT', '4h_0.5pct_SHORT']],
        backtest_df[['bar_time', 'price', '2h_0.5pct', '4h_0.5pct', '2h_0.5pct_SHORT', '4h_0.5pct_SHORT']],
        on='bar_time',
        suffixes=('_live', '_backtest'),
        how='inner'
    )
    
    print(f"Matched bars: {len(comparison)}")
    
    if len(comparison) > 0:
        print("\nSample comparison (first 10 bars):")
        print("-"*70)
        print(f"{'Time':<12} {'Price':<10} {'2h LONG':<20} {'4h LONG':<20}")
        print(f"{'':12} {'':10} {'Live':>8} {'BT':>8} {'Diff':>4} {'Live':>8} {'BT':>8} {'Diff':>4}")
        print("-"*70)
        
        for _, row in comparison.head(10).iterrows():
            time_str = row['bar_time'].strftime('%H:%M')
            price = row['price_live']
            
            live_2h = row['2h_0.5pct_live'] * 100 if pd.notna(row['2h_0.5pct_live']) else 0
            bt_2h = row['2h_0.5pct_backtest'] * 100 if pd.notna(row['2h_0.5pct_backtest']) else 0
            diff_2h = live_2h - bt_2h
            
            live_4h = row['4h_0.5pct_live'] * 100 if pd.notna(row['4h_0.5pct_live']) else 0
            bt_4h = row['4h_0.5pct_backtest'] * 100 if pd.notna(row['4h_0.5pct_backtest']) else 0
            diff_4h = live_4h - bt_4h
            
            print(f"{time_str:<12} ${price:<9.0f} {live_2h:>7.1f}% {bt_2h:>7.1f}% {diff_2h:>+4.0f} {live_4h:>7.1f}% {bt_4h:>7.1f}% {diff_4h:>+4.0f}")
        
        # Calculate average differences
        print("\n" + "-"*70)
        print("AVERAGE DIFFERENCES (Live - Backtest):")
        for model in ['2h_0.5pct', '4h_0.5pct', '2h_0.5pct_SHORT', '4h_0.5pct_SHORT']:
            live_col = f'{model}_live'
            bt_col = f'{model}_backtest'
            if live_col in comparison.columns and bt_col in comparison.columns:
                diff = (comparison[live_col] - comparison[bt_col]) * 100
                print(f"  {model}: {diff.mean():+.1f}% avg diff (std: {diff.std():.1f}%)")

# 9. Check when signals crossed threshold
print("\n9. THRESHOLD CROSSING ANALYSIS")
print("-"*50)

THRESHOLD_LONG = 0.65
THRESHOLD_SHORT = 0.65

print(f"Thresholds: LONG >= {THRESHOLD_LONG:.0%}, SHORT >= {THRESHOLD_SHORT:.0%}")

# Live signals that crossed threshold
print("\nLIVE signals that crossed threshold today:")
for _, row in live_5min.iterrows():
    triggered = []
    if row.get('2h_0.5pct', 0) >= THRESHOLD_LONG:
        triggered.append(f"2h_0.5pct LONG: {row['2h_0.5pct']:.1%}")
    if row.get('4h_0.5pct', 0) >= THRESHOLD_LONG:
        triggered.append(f"4h_0.5pct LONG: {row['4h_0.5pct']:.1%}")
    if row.get('2h_0.5pct_SHORT', 0) >= THRESHOLD_SHORT:
        triggered.append(f"2h_0.5pct_SHORT: {row['2h_0.5pct_SHORT']:.1%}")
    if row.get('4h_0.5pct_SHORT', 0) >= THRESHOLD_SHORT:
        triggered.append(f"4h_0.5pct_SHORT: {row['4h_0.5pct_SHORT']:.1%}")
    
    if triggered:
        print(f"  {row['bar_time'].strftime('%H:%M')} @ ${row['price']:.0f}: {', '.join(triggered)}")

# 10. Summary
print("\n" + "="*70)
print("AUDIT SUMMARY")
print("="*70)

print("""
KEY FINDINGS:

1. DATA FRESHNESS:
   - Parquet ends at Dec 18, 2025
   - Live bot uses Dec 18 features as BASE
   - Only OHLCV and price-dependent features are updated
   
2. FEATURE CALCULATION:
   - Live bot: parquet features + synthetic bar OHLCV update
   - Backtest: Same approach (parquet base + OHLCV update)
   - Most features (MA, RSI, etc.) are STALE from Dec 18!
   
3. SIGNAL DIFFERENCES:
   - Differences arise because:
     a) Live updates every 15 seconds, backtest uses 5-min bars
     b) Synthetic bar OHLCV differs from Yahoo 5-min bars
     c) Some features may be recalculated differently
     
4. RECOMMENDATION:
   - Update parquet data more frequently (hourly refresh)
   - Ensure feature recalculation matches between live and backtest
   - Consider recalculating ALL features, not just OHLCV
""")
