"""
Test Very High Thresholds (0.80, 0.85, 0.90)
Focus on ultra-selective signals with highest WR
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

BOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BOT_DIR)

from btc_model_package.predictor import BTCPredictor

# Config
DATA_PATH = os.path.join(BOT_DIR, 'data', 'BTC_5min_8years.parquet')
MODEL_DIR = os.path.join(BOT_DIR, 'btc_model_package')

TP_PCT = 1.0
SL_PCT = 0.5
MAX_FORWARD_BARS = 72
STEP_BARS = 12
LOOKBACK = 250
COMMISSION = 2.02
CONTRACT_VALUE = 0.1

TEST_START = '2024-01-01'
TEST_END = '2025-12-01'

# High thresholds to test
HIGH_THRESHOLDS = [
    (0.75, 0.15),
    (0.80, 0.10),
    (0.85, 0.05),
    (0.90, 0.02),
]


def simulate_trade(df_forward, direction, entry_price, tp_pct, sl_pct, max_bars):
    """Simulate trade"""
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


def run_backtest_with_threshold(df, predictor, long_threshold, short_threshold, commission):
    """Run backtest"""
    predictor.LONG_THRESHOLD = long_threshold
    predictor.SHORT_THRESHOLD = short_threshold

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

        entry_price = window['close'].iloc[-1]
        entry_time = window.index[-1]
        forward_data = df.iloc[idx:idx + MAX_FORWARD_BARS + 1]

        if len(forward_data) < 2:
            continue

        exit_reason, exit_price, bars_held, pnl_pct = simulate_trade(
            forward_data, signal, entry_price, TP_PCT, SL_PCT, MAX_FORWARD_BARS
        )

        pnl_dollar = (pnl_pct / 100) * entry_price * CONTRACT_VALUE - commission

        trades.append({
            'entry_time': entry_time,
            'direction': signal,
            'confidence': confidence,
            'entry_price': entry_price,
            'pnl_pct': pnl_pct,
            'pnl_dollar': pnl_dollar,
        })

        last_trade_bar = idx + bars_held

    return trades


def analyze_results(trades):
    """Calculate metrics"""
    if not trades:
        return None

    df = pd.DataFrame(trades)
    total = len(df)
    wins = (df['pnl_pct'] > 0).sum()
    wr = wins / total * 100 if total > 0 else 0

    total_pnl = df['pnl_dollar'].sum()
    avg_pnl = df['pnl_dollar'].mean()

    gross_wins = df.loc[df['pnl_pct'] > 0, 'pnl_dollar'].sum()
    gross_losses = abs(df.loc[df['pnl_pct'] <= 0, 'pnl_dollar'].sum())
    pf = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    equity = df['pnl_dollar'].cumsum()
    peak = equity.cummax()
    drawdown = equity - peak
    max_dd = drawdown.min()

    # Direction breakdown
    long_trades = df[df['direction'] == 'LONG']
    short_trades = df[df['direction'] == 'SHORT']

    return {
        'trades': total,
        'win_rate': wr,
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'profit_factor': pf,
        'max_dd': max_dd,
        'long_count': len(long_trades),
        'short_count': len(short_trades),
        'long_wr': (long_trades['pnl_pct'] > 0).sum() / len(long_trades) * 100 if len(long_trades) > 0 else 0,
        'short_wr': (short_trades['pnl_pct'] > 0).sum() / len(short_trades) * 100 if len(short_trades) > 0 else 0,
    }


def main():
    print("=" * 80)
    print("HIGH THRESHOLD ANALYSIS (0.75 - 0.90)")
    print("=" * 80)

    # Load data
    print(f"\nLoading data...")
    df_raw = pd.read_parquet(DATA_PATH)
    df_raw.columns = [c.lower() for c in df_raw.columns]
    df = df_raw.loc[TEST_START:TEST_END].copy()

    print(f"Test period: {df.index[0]} to {df.index[-1]}")
    print(f"Total bars: {len(df):,}")

    # Load predictor
    print(f"Loading 22-feature predictor...")
    predictor = BTCPredictor(model_dir=MODEL_DIR)
    print(f"✓ {len(predictor.features)} features loaded\n")

    results = []

    for long_th, short_th in HIGH_THRESHOLDS:
        print(f"Testing LONG={long_th:.2f}, SHORT={short_th:.2f}...", end=' ')

        trades = run_backtest_with_threshold(
            df, predictor, long_th, short_th, COMMISSION
        )

        if not trades:
            print("No trades")
            continue

        metrics = analyze_results(trades)
        print(f"{metrics['trades']} trades, {metrics['win_rate']:.1f}% WR, ${metrics['total_pnl']:+,.0f}")

        results.append({
            'LONG_TH': long_th,
            'SHORT_TH': short_th,
            **metrics
        })

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS WITH $2.02 COMMISSION")
    print("=" * 80)
    print(f"{'LONG_TH':<9} {'SHORT_TH':<10} {'Trades':<7} {'WR%':<7} "
          f"{'LONG':<6} {'SHORT':<7} {'Total P&L':<12} {'Avg P&L':<10} {'PF':<6}")
    print("-" * 80)

    for r in results:
        marker = "✓" if r['total_pnl'] > 0 else "✗"
        print(f"{r['LONG_TH']:<9.2f} {r['SHORT_TH']:<10.2f} "
              f"{r['trades']:<7d} {r['win_rate']:<7.1f} "
              f"{r['long_count']:<6d} {r['short_count']:<7d} "
              f"${r['total_pnl']:<+11,.0f} ${r['avg_pnl']:<+9.2f} "
              f"{r['profit_factor']:<6.2f} {marker}")

    # Find best
    if results:
        print("\n" + "=" * 80)
        print("COMPARISON")
        print("=" * 80)

        best_total = max(results, key=lambda x: x['total_pnl'])
        best_wr = max(results, key=lambda x: x['win_rate'])
        best_avg = max(results, key=lambda x: x['avg_pnl'])

        print(f"\nHighest Total P&L: LONG={best_total['LONG_TH']:.2f}, SHORT={best_total['SHORT_TH']:.2f}")
        print(f"  ${best_total['total_pnl']:+,.2f} on {best_total['trades']} trades")

        print(f"\nHighest Win Rate: LONG={best_wr['LONG_TH']:.2f}, SHORT={best_wr['SHORT_TH']:.2f}")
        print(f"  {best_wr['win_rate']:.1f}% WR on {best_wr['trades']} trades")

        print(f"\nHighest Avg P&L: LONG={best_avg['LONG_TH']:.2f}, SHORT={best_avg['SHORT_TH']:.2f}")
        print(f"  ${best_avg['avg_pnl']:+.2f} per trade on {best_avg['trades']} trades")

        print("\n" + "=" * 80)
        print("RECOMMENDATION")
        print("=" * 80)

        # Compare 0.60 baseline with best high threshold
        print("\n0.60/0.30 (baseline): 1,612 trades, 42.7% WR, +$4,430")
        if best_total['total_pnl'] > 4430:
            print(f"{best_total['LONG_TH']:.2f}/{best_total['SHORT_TH']:.2f} (high threshold): "
                  f"{best_total['trades']} trades, {best_total['win_rate']:.1f}% WR, "
                  f"${best_total['total_pnl']:+,.0f} ✓ BETTER")
        else:
            print(f"{best_total['LONG_TH']:.2f}/{best_total['SHORT_TH']:.2f} (high threshold): "
                  f"{best_total['trades']} trades, {best_total['win_rate']:.1f}% WR, "
                  f"${best_total['total_pnl']:+,.0f} - Worse total P&L")


if __name__ == '__main__':
    main()
