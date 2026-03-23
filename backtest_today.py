#!/usr/bin/env python3
"""
Backtest today's BTC data with our models to compare vs live trading.
Uses the actual parquet features that the live bot uses.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# Load existing parquet with features (this has all 206 features)
print("Loading existing BTC parquet data with features...")
btc_parquet = pd.read_parquet('data/BTC_features.parquet')
print(f"Parquet has {len(btc_parquet)} bars, ending at {btc_parquet.index[-1]}")
print(f"Features available: {len([c for c in btc_parquet.columns if c not in ['Open','High','Low','Close','Volume']])}")

# The parquet ends at Dec 18, so we need to use that data for backtest
# Let's backtest the last few days that ARE in the parquet
print("\nUsing parquet data for backtest (Dec 16-18, 2025)...")

# Filter to last 3 days in parquet
last_date = btc_parquet.index[-1].date()
start_date = last_date - timedelta(days=2)
recent_data = btc_parquet[btc_parquet.index.date >= start_date]
print(f"Backtest period: {recent_data.index[0]} to {recent_data.index[-1]}")
print(f"Bars in period: {len(recent_data)}")

# Load models
print("\nLoading BTC models...")
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

# Load feature columns
with open(f"{model_dir}/feature_columns.json", 'r') as f:
    feature_columns = json.load(f)
print(f"Feature columns: {len(feature_columns)}")

# Use the parquet data directly (it already has all features)
today_features = recent_data

# Run backtest simulation
print("\n" + "="*60)
print("BACKTEST SIMULATION - Dec 22, 2025")
print("="*60)

# Thresholds (matching live bot)
THRESHOLD_LONG = 0.65
THRESHOLD_SHORT = 0.65

# Model horizons
horizons = {
    '2h_0.5pct': 24,      # 2 hours = 24 bars
    '4h_0.5pct': 48,      # 4 hours = 48 bars
    '2h_0.5pct_SHORT': 24,
    '4h_0.5pct_SHORT': 48
}

# Target percentages
targets = {
    '2h_0.5pct': 0.005,
    '4h_0.5pct': 0.005,
    '2h_0.5pct_SHORT': 0.005,
    '4h_0.5pct_SHORT': 0.005
}

# Track signals and trades
signals_log = []
trades = []

print(f"\nThresholds: LONG >= {THRESHOLD_LONG:.0%}, SHORT >= {THRESHOLD_SHORT:.0%}")
print(f"Models: {list(models.keys())}")

# Iterate through today's bars
for i, (timestamp, row) in enumerate(today_features.iterrows()):
    # Skip first bars (need features to stabilize)
    if i < 50:
        continue
    
    # Prepare features for prediction
    try:
        # Get available features
        available_features = [f for f in feature_columns if f in today_features.columns]
        if len(available_features) < len(feature_columns) * 0.5:
            continue
        
        X = row[available_features].values.reshape(1, -1)
        
        # Handle NaN
        if np.isnan(X).any():
            continue
        
        # Get predictions from each model
        for model_name, model in models.items():
            try:
                prob = model.predict_proba(X)[0][1]
                
                is_long = 'SHORT' not in model_name
                threshold = THRESHOLD_LONG if is_long else THRESHOLD_SHORT
                
                # Log signal
                signals_log.append({
                    'timestamp': timestamp,
                    'model': model_name,
                    'probability': prob,
                    'price': row['Close'],
                    'triggered': prob >= threshold
                })
                
                # Check if signal triggers
                if prob >= threshold:
                    direction = 'LONG' if is_long else 'SHORT'
                    horizon = horizons[model_name]
                    target_pct = targets[model_name]
                    
                    # Calculate target and stop
                    entry_price = row['Close']
                    if direction == 'LONG':
                        target_price = entry_price * (1 + target_pct)
                        stop_price = entry_price * (1 - 0.0075)  # 0.75% stop
                    else:
                        target_price = entry_price * (1 - target_pct)
                        stop_price = entry_price * (1 + 0.0075)
                    
                    trades.append({
                        'entry_time': timestamp,
                        'model': model_name,
                        'direction': direction,
                        'entry_price': entry_price,
                        'target_price': target_price,
                        'stop_price': stop_price,
                        'horizon_bars': horizon,
                        'probability': prob
                    })
                    
                    print(f"\n{timestamp} - SIGNAL: {model_name} {direction}")
                    print(f"  Probability: {prob:.1%}")
                    print(f"  Entry: ${entry_price:.2f}")
                    print(f"  Target: ${target_price:.2f} ({'+' if direction=='LONG' else '-'}{target_pct:.1%})")
                    print(f"  Stop: ${stop_price:.2f}")
                    
            except Exception as e:
                pass
                
    except Exception as e:
        pass

# Evaluate trades
print("\n" + "="*60)
print("TRADE RESULTS")
print("="*60)

if not trades:
    print("No trades triggered today!")
else:
    print(f"\nTotal signals that triggered: {len(trades)}")
    
    for trade in trades:
        entry_time = trade['entry_time']
        entry_price = trade['entry_price']
        target_price = trade['target_price']
        stop_price = trade['stop_price']
        direction = trade['direction']
        horizon = trade['horizon_bars']
        
        # Get future bars
        future_bars = today_features[today_features.index > entry_time].head(horizon)
        
        if len(future_bars) == 0:
            print(f"\n{trade['model']} {direction} @ {entry_time}")
            print(f"  Entry: ${entry_price:.2f}, Prob: {trade['probability']:.1%}")
            print(f"  Result: PENDING (not enough future data)")
            continue
        
        # Check outcome
        exit_price = None
        exit_reason = None
        exit_time = None
        
        for bar_time, bar in future_bars.iterrows():
            if direction == 'LONG':
                if bar['High'] >= target_price:
                    exit_price = target_price
                    exit_reason = 'TARGET'
                    exit_time = bar_time
                    break
                elif bar['Low'] <= stop_price:
                    exit_price = stop_price
                    exit_reason = 'STOP'
                    exit_time = bar_time
                    break
            else:  # SHORT
                if bar['Low'] <= target_price:
                    exit_price = target_price
                    exit_reason = 'TARGET'
                    exit_time = bar_time
                    break
                elif bar['High'] >= stop_price:
                    exit_price = stop_price
                    exit_reason = 'STOP'
                    exit_time = bar_time
                    break
        
        # If no exit, use last bar
        if exit_price is None:
            exit_price = future_bars.iloc[-1]['Close']
            exit_reason = 'TIMEOUT'
            exit_time = future_bars.index[-1]
        
        # Calculate P&L
        if direction == 'LONG':
            pnl_pct = (exit_price / entry_price - 1) * 100
        else:
            pnl_pct = (entry_price / exit_price - 1) * 100
        
        trade['exit_time'] = exit_time
        trade['exit_price'] = exit_price
        trade['exit_reason'] = exit_reason
        trade['pnl_pct'] = pnl_pct
        
        print(f"\n{trade['model']} {direction} @ {entry_time.strftime('%H:%M')}")
        print(f"  Entry: ${entry_price:.2f}, Prob: {trade['probability']:.1%}")
        print(f"  Exit: ${exit_price:.2f} @ {exit_time.strftime('%H:%M')} ({exit_reason})")
        print(f"  P&L: {pnl_pct:+.2f}%")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

completed_trades = [t for t in trades if 'pnl_pct' in t]
if completed_trades:
    total_pnl = sum(t['pnl_pct'] for t in completed_trades)
    wins = [t for t in completed_trades if t['pnl_pct'] > 0]
    losses = [t for t in completed_trades if t['pnl_pct'] <= 0]
    
    print(f"Completed trades: {len(completed_trades)}")
    print(f"Wins: {len(wins)}, Losses: {len(losses)}")
    if completed_trades:
        print(f"Win rate: {len(wins)/len(completed_trades):.1%}")
    print(f"Total P&L: {total_pnl:+.2f}%")
    print(f"Avg P&L per trade: {total_pnl/len(completed_trades):+.2f}%")

# Show probability distribution
print("\n" + "="*60)
print("PROBABILITY DISTRIBUTION TODAY")
print("="*60)

signals_df = pd.DataFrame(signals_log)
if len(signals_df) > 0:
    for model_name in models.keys():
        model_signals = signals_df[signals_df['model'] == model_name]
        if len(model_signals) > 0:
            print(f"\n{model_name}:")
            print(f"  Min: {model_signals['probability'].min():.1%}")
            print(f"  Max: {model_signals['probability'].max():.1%}")
            print(f"  Mean: {model_signals['probability'].mean():.1%}")
            print(f"  Signals >= threshold: {model_signals['triggered'].sum()}")
