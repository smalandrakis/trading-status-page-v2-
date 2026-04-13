"""
VERIFY EXACT TP/SL PERFORMANCE - No filters, just raw results

Test only the 4 configurations we care about:
1. Current Swing: 3.0/1.5
2. Best from optimization: 2.5/1.0
3. Baseline HF: 1.0/0.5
4. Current HF: 1.5/0.3

Let's see which is ACTUALLY best.
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

BOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BOT_DIR)

from btc_model_package.predictor import BTCPredictor

# Exact same parameters as optimize_everything.py
DATA_PATH = os.path.join(BOT_DIR, 'data', 'BTC_5min_8years.parquet')
MODEL_DIR = os.path.join(BOT_DIR, 'btc_model_package')

MAX_FORWARD_BARS = 72
STEP_BARS = 12
LOOKBACK = 250
COMMISSION = 2.02
CONTRACT_VALUE = 0.1

TEST_START = '2024-01-01'
TEST_END = '2025-12-01'

# Position sizing
def calculate_position_size(confidence):
    size = (confidence - 0.60) * 20
    return max(1.0, min(5.0, size))


def simulate_trade(df_forward, direction, entry_price, tp_pct, sl_pct, max_bars):
    """Simulate trade with TP/SL"""
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


def run_backtest(df, predictor, config_name, tp_pct, sl_pct, long_th, short_th):
    """Run backtest with specific configuration"""

    predictor.LONG_THRESHOLD = long_th
    predictor.SHORT_THRESHOLD = short_th

    trades = []
    start_idx = LOOKBACK
    end_idx = len(df) - MAX_FORWARD_BARS - 1
    last_trade_bar = -999

    for idx in range(start_idx, end_idx, STEP_BARS):
        if idx - last_trade_bar < 6:
            continue

        window = df.iloc[idx - LOOKBACK:idx].copy()
        if len(window) < LOOKBACK:
            continue

        try:
            signal, confidence, details = predictor.predict(window)
        except:
            continue

        if signal == 'NEUTRAL':
            continue

        position_size = calculate_position_size(confidence)

        entry_price = window['close'].iloc[-1]
        forward_data = df.iloc[idx:idx + MAX_FORWARD_BARS + 1]

        if len(forward_data) < 2:
            continue

        exit_reason, exit_price, bars_held, pnl_pct = simulate_trade(
            forward_data, signal, entry_price, tp_pct, sl_pct, MAX_FORWARD_BARS
        )

        gross_pnl = (pnl_pct / 100) * entry_price * CONTRACT_VALUE * position_size
        commission_cost = COMMISSION * position_size
        net_pnl = gross_pnl - commission_cost

        trades.append({
            'signal': signal,
            'confidence': confidence,
            'position_size': position_size,
            'entry_price': entry_price,
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
    print("VERIFY EXACT TP/SL PERFORMANCE")
    print("="*80)
    print(f"\nTest Period: {TEST_START} to {TEST_END}")
    print(f"Position Sizing: 1x-5x based on confidence")
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
            'name': 'Current Swing Bot (3.0/1.5)',
            'tp': 3.0,
            'sl': 1.5,
            'long_th': 0.65,
            'short_th': 0.25,
        },
        {
            'name': 'Best from Optimization (2.5/1.0)',
            'tp': 2.5,
            'sl': 1.0,
            'long_th': 0.65,
            'short_th': 0.25,
        },
        {
            'name': 'Baseline (1.0/0.5) - Suggested HF',
            'tp': 1.0,
            'sl': 0.5,
            'long_th': 0.65,
            'short_th': 0.25,
        },
        {
            'name': 'Current HF Bot (1.5/0.3)',
            'tp': 1.5,
            'sl': 0.3,
            'long_th': 0.65,
            'short_th': 0.25,
        },
    ]

    results = []

    for config in configs:
        print(f"\nTesting {config['name']}...")
        trades = run_backtest(
            df, predictor,
            config['name'],
            config['tp'],
            config['sl'],
            config['long_th'],
            config['short_th']
        )

        metrics = analyze_results(trades, config['name'])
        if metrics:
            results.append(metrics)

    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    print(f"{'Config':<35} {'Trades':<8} {'WR%':<8} {'Total P&L':<13} {'Avg P&L':<10}")
    print("-"*80)

    for r in results:
        print(f"{r['config']:<35} {r['trades']:<8d} {r['win_rate']:<8.1f} ${r['total_pnl']:<+12,.0f} ${r['avg_pnl']:<+9.2f}")

    # Rank by total P&L
    results_sorted = sorted(results, key=lambda x: x['total_pnl'], reverse=True)

    print("\n" + "="*80)
    print("RANKING BY TOTAL P&L")
    print("="*80)
    for i, r in enumerate(results_sorted, 1):
        print(f"{i}. {r['config']:<40} ${r['total_pnl']:+,.2f}")

    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)

    best = results_sorted[0]
    print(f"\nBest Overall: {best['config']}")
    print(f"Total P&L: ${best['total_pnl']:+,.2f}")
    print(f"Win Rate: {best['win_rate']:.1f}%")
    print(f"Trades: {best['trades']}")

    # Find best HF (most trades)
    hf_candidates = [r for r in results_sorted if r['trades'] >= 700]
    if hf_candidates:
        best_hf = hf_candidates[0]
        print(f"\nBest High-Frequency: {best_hf['config']}")
        print(f"Total P&L: ${best_hf['total_pnl']:+,.2f}")
        print(f"Win Rate: {best_hf['win_rate']:.1f}%")
        print(f"Trades: {best_hf['trades']}")


if __name__ == '__main__':
    main()
