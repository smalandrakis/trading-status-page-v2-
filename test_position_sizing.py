"""
Confidence-Based Position Sizing Backtest

Strategy: Scale position size with confidence level
- Higher confidence = Larger position
- Lower confidence = Smaller position or skip

This reduces total trades (lower commission) while capitalizing on strongest signals.
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
COMMISSION = 2.02  # Per contract
CONTRACT_VALUE = 0.1  # 0.1 BTC per MBT contract

TEST_START = '2024-01-01'
TEST_END = '2025-12-01'

# Position sizing strategies to test
SIZING_STRATEGIES = {
    'Fixed_1x': {
        'description': 'Fixed 1 contract (baseline)',
        'thresholds': (0.60, 0.30),
        'size_func': lambda conf: 1.0
    },

    'Tiered_3levels': {
        'description': 'Medium confidence: 1x, High: 2x, Very high: 3x',
        'thresholds': (0.65, 0.25),  # Higher base threshold
        'size_func': lambda conf: (
            1.0 if conf < 0.70 else
            2.0 if conf < 0.75 else
            3.0
        )
    },

    'Tiered_4levels': {
        'description': 'Scaled: 1x/2x/3x/4x based on confidence',
        'thresholds': (0.65, 0.25),
        'size_func': lambda conf: (
            1.0 if conf < 0.70 else
            2.0 if conf < 0.75 else
            3.0 if conf < 0.80 else
            4.0
        )
    },

    'Conservative_highconf': {
        'description': 'Only trade 0.70+, scale 1x/2x/3x',
        'thresholds': (0.70, 0.20),
        'size_func': lambda conf: (
            1.0 if conf < 0.75 else
            2.0 if conf < 0.80 else
            3.0
        )
    },

    'Aggressive_scale': {
        'description': 'Linear scaling: size = (conf - 0.60) * 20',
        'thresholds': (0.65, 0.25),
        'size_func': lambda conf: max(1.0, min(5.0, (conf - 0.60) * 20))
    },

    'Ultra_selective': {
        'description': 'Only 0.75+, size 2-5x',
        'thresholds': (0.75, 0.15),
        'size_func': lambda conf: (
            2.0 if conf < 0.78 else
            3.0 if conf < 0.82 else
            4.0 if conf < 0.85 else
            5.0
        )
    },
}


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


def run_backtest_with_sizing(df, predictor, strategy_name, config):
    """Run backtest with variable position sizing"""

    long_th, short_th = config['thresholds']
    size_func = config['size_func']

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

        # Determine position size based on confidence
        position_size = size_func(confidence)

        entry_price = window['close'].iloc[-1]
        entry_time = window.index[-1]
        forward_data = df.iloc[idx:idx + MAX_FORWARD_BARS + 1]

        if len(forward_data) < 2:
            continue

        exit_reason, exit_price, bars_held, pnl_pct = simulate_trade(
            forward_data, signal, entry_price, TP_PCT, SL_PCT, MAX_FORWARD_BARS
        )

        # Calculate P&L with variable position size
        gross_pnl = (pnl_pct / 100) * entry_price * CONTRACT_VALUE * position_size
        commission_cost = COMMISSION * position_size
        net_pnl = gross_pnl - commission_cost

        trades.append({
            'entry_time': entry_time,
            'direction': signal,
            'confidence': confidence,
            'position_size': position_size,
            'entry_price': entry_price,
            'exit_reason': exit_reason,
            'pnl_pct': pnl_pct,
            'gross_pnl': gross_pnl,
            'commission': commission_cost,
            'net_pnl': net_pnl,
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

    total_pnl = df['net_pnl'].sum()
    total_commission = df['commission'].sum()
    total_gross = df['gross_pnl'].sum()
    avg_pnl = df['net_pnl'].mean()
    avg_size = df['position_size'].mean()

    gross_wins = df.loc[df['pnl_pct'] > 0, 'net_pnl'].sum()
    gross_losses = abs(df.loc[df['pnl_pct'] <= 0, 'net_pnl'].sum())
    pf = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    equity = df['net_pnl'].cumsum()
    peak = equity.cummax()
    drawdown = equity - peak
    max_dd = drawdown.min()

    return {
        'trades': total,
        'win_rate': wr,
        'total_pnl': total_pnl,
        'total_gross': total_gross,
        'total_commission': total_commission,
        'avg_pnl': avg_pnl,
        'avg_size': avg_size,
        'profit_factor': pf,
        'max_dd': max_dd,
    }


def main():
    print("=" * 80)
    print("CONFIDENCE-BASED POSITION SIZING ANALYSIS")
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

    for strategy_name, config in SIZING_STRATEGIES.items():
        print(f"\nTesting: {strategy_name}")
        print(f"  {config['description']}")
        print(f"  Thresholds: LONG={config['thresholds'][0]:.2f}, SHORT={config['thresholds'][1]:.2f}")

        trades = run_backtest_with_sizing(df, predictor, strategy_name, config)

        if not trades:
            print("  No trades generated")
            continue

        metrics = analyze_results(trades)

        print(f"  Trades: {metrics['trades']}")
        print(f"  Avg Size: {metrics['avg_size']:.2f}x")
        print(f"  Win Rate: {metrics['win_rate']:.1f}%")
        print(f"  Total P&L: ${metrics['total_pnl']:+,.2f}")
        print(f"  Avg P&L: ${metrics['avg_pnl']:+.2f}")
        print(f"  Commission: ${metrics['total_commission']:,.2f}")
        print(f"  Max DD: ${metrics['max_dd']:,.2f}")

        results.append({
            'strategy': strategy_name,
            'description': config['description'],
            **metrics
        })

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE - SORTED BY TOTAL P&L")
    print("=" * 80)

    results_sorted = sorted(results, key=lambda x: x['total_pnl'], reverse=True)

    print(f"{'Strategy':<25} {'Trades':<7} {'Avg Size':<9} {'WR%':<7} "
          f"{'Total P&L':<12} {'Avg P&L':<10} {'Commission':<12}")
    print("-" * 80)

    for r in results_sorted:
        marker = "✓" if r['total_pnl'] > 0 else "✗"
        print(f"{r['strategy']:<25} {r['trades']:<7d} {r['avg_size']:<9.2f} "
              f"{r['win_rate']:<7.1f} ${r['total_pnl']:<+11,.0f} "
              f"${r['avg_pnl']:<+9.2f} ${r['total_commission']:<11,.0f} {marker}")

    # Best strategy
    if results_sorted:
        best = results_sorted[0]
        print("\n" + "=" * 80)
        print("BEST STRATEGY")
        print("=" * 80)
        print(f"\n{best['strategy']}: {best['description']}")
        print(f"\n  Trades: {best['trades']}")
        print(f"  Avg Position Size: {best['avg_size']:.2f}x")
        print(f"  Win Rate: {best['win_rate']:.1f}%")
        print(f"  Total P&L: ${best['total_pnl']:+,.2f}")
        print(f"  Avg P&L per trade: ${best['avg_pnl']:+.2f}")
        print(f"  Gross P&L: ${best['total_gross']:+,.2f}")
        print(f"  Total Commission: ${best['total_commission']:,.2f}")
        print(f"  Commission as % of gross: {best['total_commission']/abs(best['total_gross'])*100:.1f}%")
        print(f"  Profit Factor: {best['profit_factor']:.2f}")
        print(f"  Max Drawdown: ${best['max_dd']:,.2f}")

        # Compare to baseline
        baseline = next((r for r in results if r['strategy'] == 'Fixed_1x'), None)
        if baseline and best['strategy'] != 'Fixed_1x':
            improvement = best['total_pnl'] - baseline['total_pnl']
            improvement_pct = (improvement / baseline['total_pnl']) * 100 if baseline['total_pnl'] > 0 else 0

            print("\n" + "-" * 80)
            print("VS BASELINE (Fixed_1x)")
            print("-" * 80)
            print(f"  Baseline P&L: ${baseline['total_pnl']:+,.2f}")
            print(f"  Best Strategy P&L: ${best['total_pnl']:+,.2f}")
            print(f"  Improvement: ${improvement:+,.2f} ({improvement_pct:+.1f}%)")


if __name__ == '__main__':
    main()
