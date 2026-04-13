"""
Walk-Forward Test with EXACT SAME settings as verify_exact_configs.py

This verifies that the optimal configurations hold up with proper walk-forward methodology:
- Position sizing: 1x-5x based on confidence
- Asymmetric thresholds: 0.65 LONG / 0.25 SHORT
- Test configs: 2.5/1.0 (Swing) and 1.0/0.5 (HF)

Expected results should match verify_exact_configs.py since both use identical logic.
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

BOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BOT_DIR)

from btc_model_package.predictor import BTCPredictor

# EXACT SAME PARAMETERS as verify_exact_configs.py
DATA_PATH = os.path.join(BOT_DIR, 'data', 'BTC_5min_8years.parquet')
MODEL_DIR = os.path.join(BOT_DIR, 'btc_model_package')

MAX_FORWARD_BARS = 72
STEP_BARS = 12
LOOKBACK = 250
COMMISSION = 2.02
CONTRACT_VALUE = 0.1

TEST_START = '2024-01-01'
TEST_END = '2025-12-01'

# ASYMMETRIC THRESHOLDS (same as bots)
LONG_THRESHOLD = 0.65
SHORT_THRESHOLD = 0.25


def calculate_position_size(confidence):
    """Position sizing: 1x-5x based on confidence"""
    size = (confidence - 0.60) * 20
    return max(1.0, min(5.0, size))


def simulate_trade(df_forward, direction, entry_price, tp_pct, sl_pct, max_bars):
    """Simulate trade with TP/SL - EXACT SAME as verify_exact_configs.py"""
    for i in range(1, min(len(df_forward), max_bars + 1)):
        bar = df_forward.iloc[i]
        high = bar['high']
        low = bar['low']

        if direction == 'LONG':
            tp_price = entry_price * (1 + tp_pct / 100)
            sl_price = entry_price * (1 - sl_pct / 100)

            if low <= sl_price:
                return 'STOP_LOSS', sl_price, i, -sl_pct
            if high >= tp_price:
                return 'TAKE_PROFIT', tp_price, i, tp_pct

        elif direction == 'SHORT':
            tp_price = entry_price * (1 - tp_pct / 100)
            sl_price = entry_price * (1 + sl_pct / 100)

            if high >= sl_price:
                return 'STOP_LOSS', sl_price, i, -sl_pct
            if low <= tp_price:
                return 'TAKE_PROFIT', tp_price, i, tp_pct

    # Timeout
    last_close = df_forward.iloc[min(len(df_forward) - 1, max_bars)]['close']
    if direction == 'LONG':
        pnl_pct = (last_close / entry_price - 1) * 100
    else:
        pnl_pct = (entry_price / last_close - 1) * 100

    return 'TIMEOUT', last_close, min(len(df_forward) - 1, max_bars), pnl_pct


def run_walkforward(df, predictor, config_name, tp_pct, sl_pct):
    """Run walk-forward backtest with specific configuration"""

    # Set thresholds
    predictor.LONG_THRESHOLD = LONG_THRESHOLD
    predictor.SHORT_THRESHOLD = SHORT_THRESHOLD

    trades = []
    start_idx = LOOKBACK
    end_idx = len(df) - MAX_FORWARD_BARS - 1
    last_trade_bar = -999

    print(f"\nTesting {config_name}...")
    print(f"  Test bars: {start_idx} to {end_idx} (step={STEP_BARS})")

    for idx in range(start_idx, end_idx, STEP_BARS):
        # Prevent overlapping trades
        if idx - last_trade_bar < 6:
            continue

        # Get lookback window
        window = df.iloc[idx - LOOKBACK:idx].copy()
        if len(window) < LOOKBACK:
            continue

        # Get prediction
        try:
            signal, confidence, details = predictor.predict(window)
        except Exception as e:
            continue

        if signal == 'NEUTRAL':
            continue

        # Calculate position size
        position_size = calculate_position_size(confidence)

        # Entry
        entry_price = window['close'].iloc[-1]
        forward_data = df.iloc[idx:idx + MAX_FORWARD_BARS + 1]

        if len(forward_data) < 2:
            continue

        # Simulate trade
        exit_reason, exit_price, bars_held, pnl_pct = simulate_trade(
            forward_data, signal, entry_price, tp_pct, sl_pct, MAX_FORWARD_BARS
        )

        # Calculate P&L
        gross_pnl = (pnl_pct / 100) * entry_price * CONTRACT_VALUE * position_size
        commission_cost = COMMISSION * position_size
        net_pnl = gross_pnl - commission_cost

        trades.append({
            'entry_time': window.index[-1],
            'signal': signal,
            'confidence': confidence,
            'position_size': position_size,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'pnl_pct': pnl_pct,
            'gross_pnl': gross_pnl,
            'commission': commission_cost,
            'net_pnl': net_pnl,
            'bars_held': bars_held,
        })

        last_trade_bar = idx + bars_held

    return trades


def analyze_results(trades, config_name):
    """Calculate and print metrics"""
    if not trades:
        print(f"\n{config_name}: NO TRADES")
        return None

    df = pd.DataFrame(trades)

    total = len(df)
    wins = (df['pnl_pct'] > 0).sum()
    losses = (df['pnl_pct'] <= 0).sum()
    wr = wins / total * 100 if total > 0 else 0

    total_pnl = df['net_pnl'].sum()
    avg_pnl = df['net_pnl'].mean()
    avg_size = df['position_size'].mean()

    gross_wins = df.loc[df['pnl_pct'] > 0, 'net_pnl'].sum()
    gross_losses = abs(df.loc[df['pnl_pct'] <= 0, 'net_pnl'].sum())
    pf = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    equity = df['net_pnl'].cumsum()
    peak = equity.cummax()
    drawdown = equity - peak
    max_dd = drawdown.min()

    total_commission = df['commission'].sum()

    # Count by exit reason
    tp_count = (df['exit_reason'] == 'TAKE_PROFIT').sum()
    sl_count = (df['exit_reason'] == 'STOP_LOSS').sum()
    to_count = (df['exit_reason'] == 'TIMEOUT').sum()

    print(f"\n{'='*80}")
    print(f"{config_name}")
    print(f"{'='*80}")
    print(f"Total Trades:     {total:,}")
    print(f"Wins / Losses:    {wins} / {losses}")
    print(f"Win Rate:         {wr:.1f}%")
    print(f"")
    print(f"Total P&L:        ${total_pnl:+,.2f}")
    print(f"Avg P&L/trade:    ${avg_pnl:+.2f}")
    print(f"Avg Position:     {avg_size:.2f}x")
    print(f"")
    print(f"Total Commission: ${total_commission:,.2f}")
    print(f"Profit Factor:    {pf:.2f}")
    print(f"Max Drawdown:     ${max_dd:,.2f}")
    print(f"")
    print(f"Exit Reasons:")
    print(f"  Take Profit:    {tp_count} ({tp_count/total*100:.1f}%)")
    print(f"  Stop Loss:      {sl_count} ({sl_count/total*100:.1f}%)")
    print(f"  Timeout:        {to_count} ({to_count/total*100:.1f}%)")

    return {
        'config': config_name,
        'trades': total,
        'wins': wins,
        'losses': losses,
        'win_rate': wr,
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'avg_size': avg_size,
        'profit_factor': pf,
        'max_dd': max_dd,
        'total_commission': total_commission,
    }


def main():
    print("="*80)
    print("WALK-FORWARD TEST - VERIFY OPTIMAL CONFIGS")
    print("="*80)
    print(f"\nTest Period: {TEST_START} to {TEST_END}")
    print(f"Position Sizing: 1x-5x based on confidence")
    print(f"Thresholds: LONG={LONG_THRESHOLD}, SHORT={SHORT_THRESHOLD}")
    print(f"Commission: ${COMMISSION} per contract")

    # Load data
    print(f"\nLoading data...")
    df_raw = pd.read_parquet(DATA_PATH)
    df_raw.columns = [c.lower() for c in df_raw.columns]
    df = df_raw.loc[TEST_START:TEST_END].copy()
    print(f"Total bars: {len(df):,}")

    # Load predictor
    print(f"\nLoading models...")
    predictor = BTCPredictor(model_dir=MODEL_DIR)

    # Configs to test
    configs = [
        {
            'name': 'Swing Bot (2.5/1.0)',
            'tp': 2.5,
            'sl': 1.0,
        },
        {
            'name': 'HF Bot (1.0/0.5)',
            'tp': 1.0,
            'sl': 0.5,
        },
    ]

    results = []

    for config in configs:
        trades = run_walkforward(
            df, predictor,
            config['name'],
            config['tp'],
            config['sl']
        )

        metrics = analyze_results(trades, config['name'])
        if metrics:
            results.append(metrics)

    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY - WALK-FORWARD RESULTS")
    print("="*80)
    print(f"{'Config':<25} {'Trades':<8} {'WR%':<8} {'Total P&L':<13} {'Avg P&L':<10}")
    print("-"*80)

    for r in results:
        print(f"{r['config']:<25} {r['trades']:<8d} {r['win_rate']:<8.1f} ${r['total_pnl']:<+12,.0f} ${r['avg_pnl']:<+9.2f}")

    print("\n" + "="*80)
    print("COMPARISON TO verify_exact_configs.py")
    print("="*80)
    print("\nExpected (from verify_exact_configs.py):")
    print("  Swing (2.5/1.0): 677 trades, 50.7% WR, $13,194 total")
    print("  HF (1.0/0.5):    734 trades, 45.9% WR, $10,120 total")
    print("\nIf results match closely, the optimal configs are VERIFIED!")


if __name__ == '__main__':
    main()
