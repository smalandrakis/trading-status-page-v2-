"""
Quick Backtest: V3 Models on 0.3%/0.1% Micro-Movement Strategy
Simplified version to complete quickly
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
from btc_model_package.predictor import BTCPredictor

BOT_DIR = Path(__file__).parent

# Load data
data_file = BOT_DIR / 'data' / 'BTC_5min_8years.parquet'
print(f"Loading data...")
df = pd.read_parquet(data_file)
df.columns = [c.lower() for c in df.columns]
df = df[df.index >= '2025-01-01']
print(f"Testing on 2025-2026: {len(df):,} bars\n")

# Load predictor
print("Loading V3 models...")
predictor = BTCPredictor()
predictor.LONG_THRESHOLD = 0.50
predictor.SHORT_THRESHOLD = 0.30
print()

# Parameters
TP_PCT = 0.003  # 0.3%
SL_PCT = 0.001  # 0.1%
COMMISSION = 2.02
SLIPPAGE = 0.0005  # 0.05% per side
BTC_CONTRACT_SIZE = 0.1

def calculate_position_size(confidence):
    """3x-6x sizing"""
    size = (confidence - 0.45) * 15
    return int(max(3, min(6, size)))

def simulate_trade(entry_price, direction, size, forward_prices):
    """Simulate trade"""
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
for i in range(250, len(df) - 48, 100):  # Every 100 bars for speed
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

    if len(trades) % 50 == 0:
        print(f"  Progress: {len(trades)} trades...")

# Results
trades_df = pd.DataFrame(trades)
print(f"\n{'='*80}")
print("BACKTEST RESULTS")
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

# By size
print(f"\n{'-'*80}")
print("BY POSITION SIZE:")
print('-'*80)
for size in sorted(trades_df['size'].unique()):
    size_df = trades_df[trades_df['size'] == size]
    size_wr = (size_df['pnl'] > 0).sum() / len(size_df) * 100
    size_avg = size_df['pnl'].mean()
    print(f"{size}x: {len(size_df)} trades, {size_wr:.1f}% WR, ${size_avg:+.2f} avg P&L")

# By direction
print(f"\n{'-'*80}")
print("BY DIRECTION:")
print('-'*80)
for direction in ['LONG', 'SHORT']:
    dir_df = trades_df[trades_df['direction'] == direction]
    if len(dir_df) > 0:
        dir_wr = (dir_df['pnl'] > 0).sum() / len(dir_df) * 100
        dir_avg = dir_df['pnl'].mean()
        print(f"{direction}: {len(dir_df)} trades, {dir_wr:.1f}% WR, ${dir_avg:+.2f} avg P&L")

print(f"\n{'='*80}")
print("CONCLUSION:")
print('='*80)

if win_rate >= 45:
    print(f"✓ EXCELLENT: {win_rate:.1f}% WR - existing V3 models work great!")
    print(f"  Avg P&L: ${avg_pnl:.2f}/trade ({'profitable' if avg_pnl > 0 else 'UNPROFITABLE'})")
    print(f"  No custom training needed - deploy as-is")
elif win_rate >= 40:
    print(f"✓ GOOD: {win_rate:.1f}% WR - existing V3 models are profitable")
    print(f"  Avg P&L: ${avg_pnl:.2f}/trade")
    print(f"  Optional: Train custom models for improvement")
elif win_rate >= 35:
    print(f"⚠ MARGINAL: {win_rate:.1f}% WR - borderline")
    print(f"  Avg P&L: ${avg_pnl:.2f}/trade - {'viable' if avg_pnl > 5 else 'too tight'}")
    print(f"  Recommend: Train custom models for 0.3%/0.1%")
else:
    print(f"✗ INSUFFICIENT: {win_rate:.1f}% WR below target")
    print(f"  Avg P&L: ${avg_pnl:.2f}/trade")
    print(f"  MUST train custom models for 0.3%/0.1% targets")

print(f"\n{'='*80}\n")
