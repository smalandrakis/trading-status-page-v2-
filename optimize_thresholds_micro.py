"""
Threshold Optimization for Micro Models

Test different threshold combinations to find optimal WR/P&L
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
from btc_model_package.micro_predictor import MicroPredictor

BOT_DIR = Path(__file__).parent

# Load data
data_file = BOT_DIR / 'data' / 'BTC_5min_8years.parquet'
df = pd.read_parquet(data_file)
df.columns = [c.lower() for c in df.columns]
df = df[df.index >= '2025-01-01'][:20000]  # Test on 20K bars for speed
print(f"Testing on {len(df):,} bars\n")

# Parameters
TP_PCT = 0.003
SL_PCT = 0.001
COMMISSION = 2.02
SLIPPAGE = 0.0005
BTC_CONTRACT_SIZE = 0.1

def calculate_position_size(confidence):
    size = (confidence - 0.20) * 20
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

# Test different threshold combinations
threshold_combos = [
    (0.20, 0.65, "Very Low"),
    (0.25, 0.60, "Low (current)"),
    (0.30, 0.55, "Medium-Low"),
    (0.35, 0.50, "Medium"),
    (0.40, 0.45, "Medium-High"),
    (0.45, 0.40, "High"),
]

print("="*100)
print("THRESHOLD OPTIMIZATION")
print("="*100)
print(f"\n{'Long Thresh':<12} {'Short Thresh':<14} {'Label':<15} {'Trades':<8} {'WR':<8} {'Avg P&L':<12} {'Total P&L':<12}")
print("-"*100)

for long_thresh, short_thresh, label in threshold_combos:
    # Load predictor with these thresholds
    predictor = MicroPredictor()
    predictor.LONG_THRESHOLD = long_thresh
    predictor.SHORT_THRESHOLD = short_thresh

    trades = []
    for i in range(250, len(df) - 48, 50):
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

        outcome, pnl = simulate_trade(entry_price, signal, size, forward_prices)

        trades.append({'pnl': pnl})

    if len(trades) == 0:
        print(f"{long_thresh:<12.2f} {short_thresh:<14.2f} {label:<15} {'0':<8} {'-':<8} {'-':<12} {'-':<12}")
        continue

    trades_df = pd.DataFrame(trades)
    total = len(trades_df)
    wins = (trades_df['pnl'] > 0).sum()
    win_rate = wins / total * 100
    avg_pnl = trades_df['pnl'].mean()
    total_pnl = trades_df['pnl'].sum()

    print(f"{long_thresh:<12.2f} {short_thresh:<14.2f} {label:<15} {total:<8} {win_rate:<7.1f}% ${avg_pnl:<11.2f} ${total_pnl:<11.2f}")

print("="*100)
print("\nRECOMMENDATION:")
print("Select thresholds with:")
print("  1. WR > 40%")
print("  2. Avg P&L > $5")
print("  3. At least 50+ trades for statistical significance")
print("="*100)
