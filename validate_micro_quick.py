"""
Quick Micro Model Validation (Simplified)

Tests custom models vs V3 baseline on smaller dataset
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
from btc_model_package.micro_predictor import MicroPredictor

BOT_DIR = Path(__file__).parent

# Load data (full 2025-2026 dataset)
data_file = BOT_DIR / 'data' / 'BTC_5min_8years.parquet'
print("Loading data...")
df = pd.read_parquet(data_file)
df.columns = [c.lower() for c in df.columns]
df = df[df.index >= '2025-01-01']  # Full 2025-2026 dataset
print(f"Testing on {len(df):,} bars\n")

# Load micro predictor
print("Loading custom micro-movement models...")
predictor = MicroPredictor()
predictor.LONG_THRESHOLD = 0.25  # Lower thresholds to get signals
predictor.SHORT_THRESHOLD = 0.60  # (SHORT if short_prob > 0.40)
print()

# Parameters
TP_PCT = 0.003  # 0.3%
SL_PCT = 0.001  # 0.1%
COMMISSION = 2.02
SLIPPAGE = 0.0005
BTC_CONTRACT_SIZE = 0.1

def calculate_position_size(confidence):
    size = (confidence - 0.20) * 20  # Adjusted for lower confidence range
    return int(max(3, min(6, size)))

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
                return 'WIN', pnl, i
            elif price <= sl:
                notional = entry_price * BTC_CONTRACT_SIZE * size
                pnl = -(notional * (SL_PCT + SLIPPAGE) + COMMISSION)
                return 'LOSS', pnl, i
        else:
            if price <= tp:
                notional = entry_price * BTC_CONTRACT_SIZE * size
                pnl = notional * (TP_PCT - SLIPPAGE) - COMMISSION
                return 'WIN', pnl, i
            elif price >= sl:
                notional = entry_price * BTC_CONTRACT_SIZE * size
                pnl = -(notional * (SL_PCT + SLIPPAGE) + COMMISSION)
                return 'LOSS', pnl, i

    return 'TIMEOUT', 0, len(forward_prices)

# Run backtest
print("Running backtest...")
trades = []
for i in range(250, len(df) - 48, 50):  # Every 50 bars for speed
    window = df.iloc[i-250:i]
    try:
        signal, confidence, _ = predictor.predict(window)
    except:
        continue

    if signal == 'NEUTRAL':
        continue

    size = calculate_position_size(confidence)
    entry_price = df.iloc[i]['close']
    forward_prices = df.iloc[i+1:i+49]['close'].values

    outcome, pnl, bars_held = simulate_trade(entry_price, signal, size, forward_prices)

    trades.append({
        'direction': signal,
        'confidence': confidence,
        'size': size,
        'outcome': outcome,
        'pnl': pnl
    })

    if len(trades) % 20 == 0 and len(trades) > 0:
        print(f"  Progress: {len(trades)} trades...")

# Results
if len(trades) == 0:
    print("\nERROR: No trades generated!")
    print("Possible issues:")
    print("  - Thresholds too high (try lowering further)")
    print("  - Models predicting only NEUTRAL")
    print("  - Feature calculation errors")
    exit(1)

trades_df = pd.DataFrame(trades)
print(f"\n{'='*80}")
print("CUSTOM MICRO-MOVEMENT MODELS - QUICK VALIDATION")
print('='*80)

total = len(trades_df)
wins = (trades_df['pnl'] > 0).sum()
losses = (trades_df['pnl'] < 0).sum()
timeouts = (trades_df['outcome'] == 'TIMEOUT').sum()
win_rate = wins / total * 100 if total > 0 else 0

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

print(f"\n{'='*80}")
print("COMPARISON VS V3 BASELINE:")
print('='*80)
print("\nV3 Models Baseline:")
print("  Win Rate: 37.6%")
print("  Avg P&L: $0.07/trade")

print(f"\nCustom Models:")
print(f"  Win Rate: {win_rate:.1f}%")
print(f"  Avg P&L: ${avg_pnl:+.2f}/trade")

improvement_wr = win_rate - 37.6
improvement_pnl = avg_pnl - 0.07

print(f"\nImprovement:")
print(f"  WR: {improvement_wr:+.1f} percentage points")
print(f"  Avg P&L: ${improvement_pnl:+.2f}/trade")

print(f"\n{'='*80}")
print("DEPLOYMENT DECISION:")
print('='*80)

if win_rate >= 42 and avg_pnl >= 15:
    print(f"✓ EXCELLENT: {win_rate:.1f}% WR, ${avg_pnl:.2f}/trade")
    print("  → DEPLOY custom models to production bot")
elif win_rate >= 40 and avg_pnl >= 10:
    print(f"✓ GOOD: {win_rate:.1f}% WR, ${avg_pnl:.2f}/trade")
    print("  → DEPLOY custom models")
elif win_rate > 37.6:
    print(f"⚠ IMPROVEMENT: {win_rate:.1f}% WR, ${avg_pnl:.2f}/trade")
    print(f"  Better than V3 baseline but below target")
else:
    print(f"✗ NO IMPROVEMENT: {win_rate:.1f}% WR")
    print("  → Do NOT deploy")

print('='*80)
