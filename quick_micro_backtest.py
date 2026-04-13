"""
Quick Backtest: Current V3 Models on 0.3% TP / 0.1% SL Strategy

Test if existing models (trained for 1% TP / 0.5% SL) can work for micro-movements
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

BOT_DIR = Path(__file__).parent
sys.path.insert(0, str(BOT_DIR))

from btc_model_package.predictor import BTCPredictor

print("="*80)
print("QUICK BACKTEST: V3 Models on 0.3%/0.1% Micro-Movement Strategy")
print("="*80)

# Load data
data_file = BOT_DIR / 'data' / 'BTC_5min_8years.parquet'
if not data_file.exists():
    print(f"Data file not found: {data_file}")
    sys.exit(1)

print(f"\nLoading data from {data_file}...")
df = pd.read_parquet(data_file)
print(f"Loaded {len(df):,} rows")
print(f"Columns: {df.columns.tolist()}")
print(f"Index type: {type(df.index)}")

# Normalize column names to lowercase
df.columns = [c.lower() for c in df.columns]

# Use recent data (2025-2026) for quick validation
df = df[df.index >= '2025-01-01']
print(f"Testing on 2025-2026 data: {len(df):,} rows")

# Load predictor
print("\nLoading V3 models...")
predictor = BTCPredictor()
predictor.LONG_THRESHOLD = 0.50  # Micro bot thresholds
predictor.SHORT_THRESHOLD = 0.30

# Micro-movement parameters
TP_PCT = 0.003  # 0.3%
SL_PCT = 0.001  # 0.1%
COMMISSION = 2.02  # per trade round-trip
SLIPPAGE = 0.0005  # 0.05% per side
BTC_CONTRACT_SIZE = 0.1

def calculate_position_size(confidence):
    """Micro-bot sizing: 3x-6x"""
    size = (confidence - 0.45) * 15
    return int(max(3, min(6, size)))

def simulate_trade(entry_price, direction, size, forward_prices):
    """Simulate a single trade"""
    if direction == 'LONG':
        tp_price = entry_price * (1 + TP_PCT - SLIPPAGE)
        sl_price = entry_price * (1 - SL_PCT - SLIPPAGE)
    else:  # SHORT
        tp_price = entry_price * (1 - TP_PCT + SLIPPAGE)
        sl_price = entry_price * (1 + SL_PCT + SLIPPAGE)

    # Check each forward bar
    for i, price in enumerate(forward_prices):
        if direction == 'LONG':
            if price >= tp_price:
                notional = entry_price * BTC_CONTRACT_SIZE * size
                gross_profit = notional * (TP_PCT - SLIPPAGE)
                net_profit = gross_profit - COMMISSION
                return 'WIN', net_profit, i
            elif price <= sl_price:
                notional = entry_price * BTC_CONTRACT_SIZE * size
                gross_loss = notional * (SL_PCT + SLIPPAGE)
                net_loss = -(gross_loss + COMMISSION)
                return 'LOSS', net_loss, i
        else:  # SHORT
            if price <= tp_price:
                notional = entry_price * BTC_CONTRACT_SIZE * size
                gross_profit = notional * (TP_PCT - SLIPPAGE)
                net_profit = gross_profit - COMMISSION
                return 'WIN', net_profit, i
            elif price >= sl_price:
                notional = entry_price * BTC_CONTRACT_SIZE * size
                gross_loss = notional * (SL_PCT + SLIPPAGE)
                net_loss = -(gross_loss + COMMISSION)
                return 'LOSS', net_loss, i

    # Timeout after 48 bars (4 hours)
    return 'TIMEOUT', 0, len(forward_prices)

# Run backtest
print("\n" + "="*80)
print("Running backtest...")
print("="*80)

trades = []
position = None

for i in range(250, len(df) - 48):  # Need 250 bars for features, 48 bars forward
    if position is not None:
        continue  # Already in position

    # Get signal
    window = df.iloc[i-250:i]
    try:
        signal, confidence, _ = predictor.predict(window)
    except Exception as e:
        if i == 250:
            print(f"DEBUG: First prediction error: {e}")
        continue

    if i % 10000 == 0:
        print(f"Progress: {i:,}/{len(df):,} bars, {len(trades)} trades, last signal: {signal} @ {confidence:.1%}")

    if signal == 'NEUTRAL':
        continue

    # Calculate size
    size = calculate_position_size(confidence)

    # Entry
    entry_price = df.iloc[i]['close']
    forward_prices = df.iloc[i+1:i+49]['close'].values

    # Simulate trade
    outcome, pnl, bars_held = simulate_trade(entry_price, signal, size, forward_prices)

    trades.append({
        'timestamp': df.index[i],
        'direction': signal,
        'confidence': confidence,
        'size': size,
        'entry_price': entry_price,
        'outcome': outcome,
        'pnl': pnl,
        'bars_held': bars_held
    })

    if i % 1000 == 0:
        print(f"Progress: {i:,}/{len(df):,} bars, {len(trades)} trades...")

# Results
trades_df = pd.DataFrame(trades)

print("\n" + "="*80)
print("BACKTEST RESULTS")
print("="*80)

if len(trades_df) == 0:
    print("No trades generated!")
    sys.exit(1)

total_trades = len(trades_df)
wins = (trades_df['pnl'] > 0).sum()
losses = (trades_df['pnl'] < 0).sum()
timeouts = (trades_df['outcome'] == 'TIMEOUT').sum()
win_rate = wins / total_trades * 100 if total_trades > 0 else 0

total_pnl = trades_df['pnl'].sum()
avg_pnl = trades_df['pnl'].mean()
avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if wins > 0 else 0
avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losses > 0 else 0

print(f"\nTotal Trades: {total_trades:,}")
print(f"Wins: {wins} ({win_rate:.1f}%)")
print(f"Losses: {losses} ({losses/total_trades*100:.1f}%)")
print(f"Timeouts: {timeouts} ({timeouts/total_trades*100:.1f}%)")
print(f"\nTotal P&L: ${total_pnl:,.2f}")
print(f"Avg P&L/Trade: ${avg_pnl:+.2f}")
print(f"Avg Win: ${avg_win:+.2f}")
print(f"Avg Loss: ${avg_loss:+.2f}")

# By size
print("\n" + "-"*80)
print("BY POSITION SIZE:")
print("-"*80)
for size in sorted(trades_df['size'].unique()):
    size_df = trades_df[trades_df['size'] == size]
    size_wr = (size_df['pnl'] > 0).sum() / len(size_df) * 100
    size_avg = size_df['pnl'].mean()
    print(f"{size}x: {len(size_df)} trades, {size_wr:.1f}% WR, ${size_avg:+.2f} avg P&L")

# By direction
print("\n" + "-"*80)
print("BY DIRECTION:")
print("-"*80)
for direction in ['LONG', 'SHORT']:
    dir_df = trades_df[trades_df['direction'] == direction]
    if len(dir_df) > 0:
        dir_wr = (dir_df['pnl'] > 0).sum() / len(dir_df) * 100
        dir_avg = dir_df['pnl'].mean()
        print(f"{direction}: {len(dir_df)} trades, {dir_wr:.1f}% WR, ${dir_avg:+.2f} avg P&L")

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)

if win_rate >= 45:
    print(f"✓ EXCELLENT: {win_rate:.1f}% WR is great for 0.3%/0.1% strategy")
    print(f"  Expected: ${avg_pnl:.2f}/trade is {'profitable' if avg_pnl > 0 else 'UNPROFITABLE'}")
elif win_rate >= 40:
    print(f"✓ GOOD: {win_rate:.1f}% WR is profitable for 0.3%/0.1% strategy")
    print(f"  Expected: ${avg_pnl:.2f}/trade")
elif win_rate >= 35:
    print(f"⚠ MARGINAL: {win_rate:.1f}% WR is borderline")
    print(f"  Expected: ${avg_pnl:.2f}/trade - {'viable' if avg_pnl > 5 else 'too tight'}")
else:
    print(f"✗ INSUFFICIENT: {win_rate:.1f}% WR is below target")
    print(f"  Expected: ${avg_pnl:.2f}/trade - NOT RECOMMENDED")
    print("\n  RECOMMENDATION: Train custom models for 0.3%/0.1% targets")

print("\n" + "="*80)

# Save results
results_file = BOT_DIR / 'micro_backtest_results.csv'
trades_df.to_csv(results_file, index=False)
print(f"\nResults saved to: {results_file}")
