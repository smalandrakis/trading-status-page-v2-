"""
Validate Optimized LONG/SHORT XGBoost Models (0.5% TP / 0.15% SL)

Uses hyperparameter-optimized dual binary classifiers
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import sys

BOT_DIR = Path(__file__).parent

# Load data
data_file = BOT_DIR / 'data' / 'BTC_5min_8years.parquet'
print("Loading data...")
df = pd.read_parquet(data_file)
df.columns = [c.lower() for c in df.columns]
df = df[df.index >= '2025-01-01']
print(f"Testing on {len(df):,} bars\n")

# Add temporal features
print("Adding temporal features...")
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['prev_close_1'] = df['close'].shift(1)
df['prev_close_5'] = df['close'].shift(5)
df['prev_close_20'] = df['close'].shift(20)
df['price_change_1'] = df['close'].pct_change(1)
df['price_change_5'] = df['close'].pct_change(5)
df['price_change_20'] = df['close'].pct_change(20)

# Load optimized models
print("\nLoading OPTIMIZED XGBoost LONG/SHORT models...")
models_dir = BOT_DIR / 'models'
model_long = joblib.load(models_dir / 'btc_model_long_xgboost_optimized.pkl')
scaler_long = joblib.load(models_dir / 'btc_scaler_long_xgboost_optimized.pkl')
model_short = joblib.load(models_dir / 'btc_model_short_xgboost_optimized.pkl')
scaler_short = joblib.load(models_dir / 'btc_scaler_short_xgboost_optimized.pkl')
feature_names = joblib.load(models_dir / 'btc_features_xgboost_optimized.pkl')

# Load base predictor
sys.path.insert(0, str(BOT_DIR))
from btc_model_package.predictor import BTCPredictor
base_predictor = BTCPredictor()

# Parameters
TP_PCT = 0.005
SL_PCT = 0.0015
COMMISSION = 2.02
SLIPPAGE = 0.0005
BTC_CONTRACT_SIZE = 0.1

LONG_THRESHOLD = 0.50
SHORT_THRESHOLD = 0.50

def calculate_position_size(confidence):
    size = (confidence - 0.45) * 15
    return int(max(3, min(6, size)))

def calculate_features(window, idx):
    # V3 features
    features_dict = base_predictor.calculate_features(window)

    # Temporal features
    row = df.iloc[idx]
    temporal_features = {
        'hour': row['hour'],
        'day_of_week': row['day_of_week'],
        'prev_close_1': row['prev_close_1'],
        'prev_close_5': row['prev_close_5'],
        'prev_close_20': row['prev_close_20'],
        'price_change_1': row['price_change_1'],
        'price_change_5': row['price_change_5'],
        'price_change_20': row['price_change_20']
    }

    features_dict.update(temporal_features)

    # Convert to array
    features_array = np.array([[features_dict.get(f, 0) for f in feature_names]])
    features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)

    return features_array

def predict(window, idx):
    features = calculate_features(window, idx)

    # Scale for each model
    features_long = scaler_long.transform(features)
    features_short = scaler_short.transform(features)

    # Get probabilities
    proba_long = model_long.predict_proba(features_long)[0]
    proba_short = model_short.predict_proba(features_short)[0]

    # Binary classifiers: class 1 = LONG/SHORT
    long_confidence = proba_long[1]
    short_confidence = proba_short[1]

    # Decision logic: take highest confidence signal above threshold
    if long_confidence >= LONG_THRESHOLD and long_confidence > short_confidence:
        return 'LONG', long_confidence
    elif short_confidence >= SHORT_THRESHOLD:
        return 'SHORT', short_confidence
    else:
        return 'NEUTRAL', max(long_confidence, short_confidence)

def simulate_trade(entry_price, direction, size, forward_prices):
    if direction == 'LONG':
        tp = entry_price * (1 + TP_PCT - SLIPPAGE)
        sl = entry_price * (1 - SL_PCT - SLIPPAGE)
    else:
        tp = entry_price * (1 - TP_PCT + SLIPPAGE)
        sl = entry_price * (1 + SL_PCT + SLIPPAGE)

    for price in forward_prices:
        if direction == 'LONG':
            if price >= tp:
                notional = entry_price * BTC_CONTRACT_SIZE * size
                pnl = notional * (TP_PCT - SLIPPAGE) - COMMISSION
                return 'WIN', pnl
            elif price <= sl:
                notional = entry_price * BTC_CONTRACT_SIZE * size
                pnl = -(notional * (SL_PCT + SLIPPAGE) + COMMISSION)
                return 'LOSS', pnl
        else:
            if price <= tp:
                notional = entry_price * BTC_CONTRACT_SIZE * size
                pnl = notional * (TP_PCT - SLIPPAGE) - COMMISSION
                return 'WIN', pnl
            elif price >= sl:
                notional = entry_price * BTC_CONTRACT_SIZE * size
                pnl = -(notional * (SL_PCT + SLIPPAGE) + COMMISSION)
                return 'LOSS', pnl

    return 'TIMEOUT', 0

# Run backtest
print("Running backtest (every 10 bars)...")
trades = []
for i in range(250, len(df) - 48, 10):
    window = df.iloc[i-250:i]
    try:
        signal, confidence = predict(window, i)
    except:
        continue

    if signal == 'NEUTRAL':
        continue

    size = calculate_position_size(confidence)
    entry_price = df.iloc[i]['close']
    forward_prices = df.iloc[i+1:i+49]['close'].values

    outcome, pnl = simulate_trade(entry_price, signal, size, forward_prices)

    trades.append({
        'direction': signal,
        'confidence': confidence,
        'size': size,
        'outcome': outcome,
        'pnl': pnl
    })

    if len(trades) % 100 == 0:
        print(f"  Progress: {len(trades)} trades...")

# Results
if len(trades) == 0:
    print("\nERROR: No trades generated!")
    exit(1)

trades_df = pd.DataFrame(trades)
print(f"\n{'='*80}")
print("OPTIMIZED LONG/SHORT XGBOOST MODELS (0.5% TP / 0.15% SL)")
print('='*80)

total = len(trades_df)
wins = (trades_df['pnl'] > 0).sum()
losses = (trades_df['pnl'] < 0).sum()
timeouts = (trades_df['outcome'] == 'TIMEOUT').sum()
win_rate = wins / total * 100

total_pnl = trades_df['pnl'].sum()
avg_pnl = trades_df['pnl'].mean()
avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if wins > 0 else 0
avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losses > 0 else 0

print(f"\nTotal Trades: {total:,}")
print(f"Wins: {wins} ({win_rate:.1f}%)")
print(f"Losses: {losses} ({losses/total*100:.1f}%)")
print(f"Timeouts: {timeouts} ({timeouts/total*100:.1f}%)")
print(f"\nTotal P&L: ${total_pnl:,.2f}")
print(f"Avg P&L/Trade: ${avg_pnl:+.2f}")
print(f"Avg Win: ${avg_win:+.2f}")
print(f"Avg Loss: ${avg_loss:+.2f}")

# By direction
print(f"\n{'='*80}")
print("PERFORMANCE BY DIRECTION:")
print('='*80)
for direction in ['LONG', 'SHORT']:
    dir_trades = trades_df[trades_df['direction'] == direction]
    if len(dir_trades) == 0:
        continue
    dir_total = len(dir_trades)
    dir_wins = (dir_trades['pnl'] > 0).sum()
    dir_wr = dir_wins / dir_total * 100
    dir_avg_pnl = dir_trades['pnl'].mean()
    dir_total_pnl = dir_trades['pnl'].sum()
    print(f"\n{direction:5}: {dir_total:,} trades ({dir_total/total*100:.1f}%), {dir_wr:.1f}% WR, ${dir_avg_pnl:+.2f} avg, ${dir_total_pnl:+,.2f} total")

print(f"\n{'='*80}")
print("COMPARISON:")
print('='*80)
print("\nBefore Optimization (33 features):")
print("  LONG:  974 trades (25.4%), 48.7% WR, +$34.15 avg")
print("  SHORT: 2,856 trades (74.6%), 37.4% WR, +$13.57 avg")
print("  Total: 40.3% WR, +$18.80/trade, $72,015 total")

print(f"\nAfter Optimization (31 features):")
long_trades = trades_df[trades_df['direction'] == 'LONG']
short_trades = trades_df[trades_df['direction'] == 'SHORT']
if len(long_trades) > 0:
    long_wr = (long_trades['pnl'] > 0).sum() / len(long_trades) * 100
    long_avg = long_trades['pnl'].mean()
    long_total_pnl = long_trades['pnl'].sum()
    print(f"  LONG:  {len(long_trades)} trades ({len(long_trades)/total*100:.1f}%), {long_wr:.1f}% WR, ${long_avg:+.2f} avg, ${long_total_pnl:+,.2f} total")
if len(short_trades) > 0:
    short_wr = (short_trades['pnl'] > 0).sum() / len(short_trades) * 100
    short_avg = short_trades['pnl'].mean()
    short_total_pnl = short_trades['pnl'].sum()
    print(f"  SHORT: {len(short_trades)} trades ({len(short_trades)/total*100:.1f}%), {short_wr:.1f}% WR, ${short_avg:+.2f} avg, ${short_total_pnl:+,.2f} total")
print(f"  Total: {win_rate:.1f}% WR, ${avg_pnl:+.2f}/trade, ${total_pnl:+,.2f} total")

improvement = avg_pnl - 18.80
pnl_improvement = total_pnl - 72015
print(f"\nImprovement:")
print(f"  Avg P&L: ${improvement:+.2f}/trade")
print(f"  Total P&L: ${pnl_improvement:+,.2f}")
print(f"  Optimized params: Lower lr (0.03), shallower depth (4), more regularization")

print(f"\n{'='*80}")
print("DECISION:")
print('='*80)

if avg_pnl >= 18.80 and win_rate >= 40:
    print(f"✓ IMPROVEMENT: {win_rate:.1f}% WR, ${avg_pnl:.2f}/trade")
    print("  → DEPLOY optimized models")
elif avg_pnl >= 15 and total >= 3000:
    print(f"✓ COMPARABLE: {win_rate:.1f}% WR, ${avg_pnl:.2f}/trade")
    print("  → Optimized models perform similarly")
else:
    print(f"⚠ WORSE: {win_rate:.1f}% WR, ${avg_pnl:.2f}/trade")
    print("  → Keep previous models (non-optimized)")

print('='*80)
