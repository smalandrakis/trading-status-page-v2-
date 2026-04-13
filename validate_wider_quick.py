"""
Quick Validation - Wider Target Models (0.5% TP / 0.15% SL)

Tests wider target models vs current 0.3%/0.1% approach
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
import joblib

BOT_DIR = Path(__file__).parent

# Load data
data_file = BOT_DIR / 'data' / 'BTC_5min_8years.parquet'
print("Loading data...")
df = pd.read_parquet(data_file)
df.columns = [c.lower() for c in df.columns]
df = df[df.index >= '2025-01-01']  # Test on 2025-2026
print(f"Testing on {len(df):,} bars\n")

# Load models
print("Loading wider target models...")
models_dir = BOT_DIR / 'models'
model_30min = joblib.load(models_dir / 'btc_model_30min_wider.pkl')
model_1h = joblib.load(models_dir / 'btc_model_1h_wider.pkl')
model_2h = joblib.load(models_dir / 'btc_model_2h_wider.pkl')
scaler_30min = joblib.load(models_dir / 'btc_scaler_30min_wider.pkl')
scaler_1h = joblib.load(models_dir / 'btc_scaler_1h_wider.pkl')
scaler_2h = joblib.load(models_dir / 'btc_scaler_2h_wider.pkl')
feature_names = joblib.load(models_dir / 'btc_features_wider.pkl')

# Load base predictor for features
import sys
sys.path.insert(0, str(BOT_DIR))
from btc_model_package.predictor import BTCPredictor
base_predictor = BTCPredictor()

# Parameters - WIDER targets
TP_PCT = 0.005  # 0.5%
SL_PCT = 0.0015  # 0.15%
COMMISSION = 2.02
SLIPPAGE = 0.0005
BTC_CONTRACT_SIZE = 0.1

# Ensemble weights (favor shorter horizon)
WEIGHTS = [0.5, 0.3, 0.2]
LONG_THRESHOLD = 0.50
SHORT_THRESHOLD = 0.30

def calculate_position_size(confidence):
    size = (confidence - 0.45) * 15
    return int(max(3, min(6, size)))

def calculate_features(window):
    features_dict = base_predictor.calculate_features(window)
    return np.array([[features_dict[f] for f in feature_names]])

def predict(window):
    features = calculate_features(window)

    # Scale and predict
    features_30min = scaler_30min.transform(features)
    features_1h = scaler_1h.transform(features)
    features_2h = scaler_2h.transform(features)

    proba_30min = model_30min.predict_proba(features_30min)[0]
    proba_1h = model_1h.predict_proba(features_1h)[0]
    proba_2h = model_2h.predict_proba(features_2h)[0]

    # Ensemble
    proba = WEIGHTS[0] * proba_30min + WEIGHTS[1] * proba_1h + WEIGHTS[2] * proba_2h

    classes = model_30min.classes_
    long_idx = list(classes).index('LONG')
    short_idx = list(classes).index('SHORT')

    long_prob = proba[long_idx]
    short_prob = proba[short_idx]

    if long_prob >= LONG_THRESHOLD:
        return 'LONG', long_prob
    elif short_prob >= SHORT_THRESHOLD:
        return 'SHORT', short_prob
    else:
        return 'NEUTRAL', max(long_prob, short_prob)

def simulate_trade(entry_price, direction, size, forward_prices):
    if direction == 'LONG':
        tp = entry_price * (1 + TP_PCT - SLIPPAGE)
        sl = entry_price * (1 - SL_PCT - SLIPPAGE)
    else:
        tp = entry_price * (1 - TP_PCT + SLIPPAGE)
        sl = entry_price * (1 + SL_PCT + SLIPPAGE)

    for i, price in enumerate(forward_prices):
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
print("Running backtest on FULL dataset (every 10 bars)...")
trades = []
for i in range(250, len(df) - 48, 10):  # Every 10 bars for more comprehensive test
    window = df.iloc[i-250:i]
    try:
        signal, confidence = predict(window)
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
print("WIDER TARGET MODELS (0.5% TP / 0.15% SL)")
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
    print(f"\n{direction:5}: {dir_total:,} trades, {dir_wr:.1f}% WR, ${dir_avg_pnl:+.2f} avg P&L")

print(f"\n{'='*80}")
print("COMPARISON TO ORIGINAL 0.3%/0.1% APPROACH:")
print('='*80)
print("\nOriginal (0.3% TP / 0.1% SL):")
print("  Win Rate: 38.4%")
print("  Avg P&L: -$0.62/trade")

print(f"\nWider Targets (0.5% TP / 0.15% SL):")
print(f"  Win Rate: {win_rate:.1f}%")
print(f"  Avg P&L: ${avg_pnl:+.2f}/trade")

improvement_wr = win_rate - 38.4
improvement_pnl = avg_pnl - (-0.62)

print(f"\nChange:")
print(f"  WR: {improvement_wr:+.1f} percentage points")
print(f"  Avg P&L: ${improvement_pnl:+.2f}/trade")

print(f"\n{'='*80}")
print("DECISION:")
print('='*80)

if win_rate >= 42 and avg_pnl >= 15:
    print(f"✓ EXCELLENT: {win_rate:.1f}% WR, ${avg_pnl:.2f}/trade")
    print("  → DEPLOY wider target models")
elif win_rate >= 40 and avg_pnl >= 8:
    print(f"✓ GOOD: {win_rate:.1f}% WR, ${avg_pnl:.2f}/trade")
    print("  → Consider deployment")
elif avg_pnl > 5:
    print(f"⚠ MARGINAL: {win_rate:.1f}% WR, ${avg_pnl:.2f}/trade")
    print(f"  Better than 0.3%/0.1% but below target")
else:
    print(f"✗ UNPROFITABLE: {win_rate:.1f}% WR, ${avg_pnl:.2f}/trade")
    print("  → Do NOT deploy - explore other approaches")

print('='*80)
