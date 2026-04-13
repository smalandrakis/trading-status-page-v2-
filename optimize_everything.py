"""
COMPREHENSIVE STRATEGY OPTIMIZATION

Test ALL possible improvements:
1. TP/SL optimization (different ratios, asymmetric LONG/SHORT)
2. Dynamic TP/SL based on volatility (ATR-based)
3. Market regime filters (volatility, trend, volume)
4. Time-based filters (avoid low-volume hours/days)
5. Exit strategy improvements (trailing stops, time-based exits)
6. Signal quality filters (require multi-timeframe agreement)

Goal: Find the absolute best configuration
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
from itertools import product

BOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BOT_DIR)

from btc_model_package.predictor import BTCPredictor

# =============================================================================
# CONFIG
# =============================================================================
DATA_PATH = os.path.join(BOT_DIR, 'data', 'BTC_5min_8years.parquet')
MODEL_DIR = os.path.join(BOT_DIR, 'btc_model_package')

MAX_FORWARD_BARS = 72
STEP_BARS = 12
LOOKBACK = 250
COMMISSION = 2.02
CONTRACT_VALUE = 0.1

TEST_START = '2024-01-01'
TEST_END = '2025-12-01'

# =============================================================================
# STRATEGIES TO TEST
# =============================================================================

# 1. TP/SL RATIOS (most important - we've been locked at 1.0/0.5)
TPSL_CONFIGS = [
    # Current baseline
    {'name': 'Baseline_1.0_0.5', 'tp': 1.0, 'sl': 0.5, 'long_th': 0.65, 'short_th': 0.25},

    # Wider TP, same SL
    {'name': 'Wide_TP_1.5_0.5', 'tp': 1.5, 'sl': 0.5, 'long_th': 0.65, 'short_th': 0.25},
    {'name': 'Wide_TP_2.0_0.5', 'tp': 2.0, 'sl': 0.5, 'long_th': 0.65, 'short_th': 0.25},
    {'name': 'Wide_TP_2.0_0.75', 'tp': 2.0, 'sl': 0.75, 'long_th': 0.65, 'short_th': 0.25},

    # Tighter SL
    {'name': 'Tight_SL_1.0_0.3', 'tp': 1.0, 'sl': 0.3, 'long_th': 0.65, 'short_th': 0.25},
    {'name': 'Tight_SL_1.5_0.3', 'tp': 1.5, 'sl': 0.3, 'long_th': 0.65, 'short_th': 0.25},

    # Wider both (swing trading style)
    {'name': 'Swing_2.0_1.0', 'tp': 2.0, 'sl': 1.0, 'long_th': 0.65, 'short_th': 0.25},
    {'name': 'Swing_3.0_1.5', 'tp': 3.0, 'sl': 1.5, 'long_th': 0.65, 'short_th': 0.25},

    # Asymmetric (different R:R)
    {'name': 'Asymm_1.5_1.0', 'tp': 1.5, 'sl': 1.0, 'long_th': 0.65, 'short_th': 0.25},
    {'name': 'Asymm_2.5_1.0', 'tp': 2.5, 'sl': 1.0, 'long_th': 0.65, 'short_th': 0.25},
]

# 2. MARKET REGIME FILTERS
REGIME_FILTERS = {
    'None': lambda df: True,
    'High_Volume': lambda df: df['volume'].iloc[-1] > df['volume'].rolling(50).mean().iloc[-1] * 1.2,
    'High_Volatility': lambda df: df['close'].pct_change().rolling(20).std().iloc[-1] > df['close'].pct_change().rolling(100).std().iloc[-1],
    'Low_Volatility': lambda df: df['close'].pct_change().rolling(20).std().iloc[-1] < df['close'].pct_change().rolling(100).std().iloc[-1],
}

# 3. TIME FILTERS
TIME_FILTERS = {
    'None': lambda dt: True,
    'Avoid_Weekend': lambda dt: dt.dayofweek < 5,  # Mon-Fri only
    'Business_Hours': lambda dt: 8 <= dt.hour <= 20,  # 8am-8pm only
    'High_Volume_Hours': lambda dt: dt.hour in [8,9,10,13,14,15,16,20,21],  # Avoid lunch/overnight
}


def calculate_position_size(confidence):
    """Position sizing: linear scale 1-5x"""
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


def run_backtest(df, predictor, config, regime_filter_name, time_filter_name):
    """Run backtest with specific configuration"""

    tp_pct = config['tp']
    sl_pct = config['sl']
    long_th = config['long_th']
    short_th = config['short_th']

    regime_filter = REGIME_FILTERS[regime_filter_name]
    time_filter = TIME_FILTERS[time_filter_name]

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

        # Time filter
        if not time_filter(window.index[-1]):
            continue

        # Regime filter
        try:
            if not regime_filter(window):
                continue
        except:
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
            'pnl_pct': pnl_pct,
            'net_pnl': net_pnl,
            'position_size': position_size,
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
        'avg_pnl': avg_pnl,
        'avg_size': avg_size,
        'profit_factor': pf,
        'max_dd': max_dd,
    }


def main():
    print("=" * 80)
    print("COMPREHENSIVE STRATEGY OPTIMIZATION")
    print("=" * 80)

    # Load data
    print(f"\nLoading data...")
    df_raw = pd.read_parquet(DATA_PATH)
    df_raw.columns = [c.lower() for c in df_raw.columns]
    df = df_raw.loc[TEST_START:TEST_END].copy()

    print(f"Test period: {df.index[0]} to {df.index[-1]}")
    print(f"Total bars: {len(df):,}\n")

    # Load predictor
    predictor = BTCPredictor(model_dir=MODEL_DIR)

    results = []
    total_tests = len(TPSL_CONFIGS) * len(REGIME_FILTERS) * len(TIME_FILTERS)
    test_count = 0

    print(f"Testing {len(TPSL_CONFIGS)} TP/SL configs × {len(REGIME_FILTERS)} regime filters × {len(TIME_FILTERS)} time filters")
    print(f"Total: {total_tests} combinations\n")

    # Test all combinations
    for tpsl_config in TPSL_CONFIGS:
        for regime_name in REGIME_FILTERS.keys():
            for time_name in TIME_FILTERS.keys():
                test_count += 1

                config_name = f"{tpsl_config['name']}_{regime_name}_{time_name}"

                if test_count % 10 == 0:
                    print(f"  Testing {test_count}/{total_tests}: {config_name[:50]}...")

                trades = run_backtest(df, predictor, tpsl_config, regime_name, time_name)

                if not trades or len(trades) < 50:  # Skip if too few trades
                    continue

                metrics = analyze_results(trades)

                if metrics:
                    results.append({
                        'config': config_name,
                        'tp': tpsl_config['tp'],
                        'sl': tpsl_config['sl'],
                        'regime': regime_name,
                        'time_filter': time_name,
                        **metrics
                    })

    # Sort by total P&L
    results_sorted = sorted(results, key=lambda x: x['total_pnl'], reverse=True)

    print("\n" + "=" * 80)
    print(f"TOP 20 CONFIGURATIONS (sorted by Total P&L)")
    print("=" * 80)
    print(f"{'Config':<50} {'Trades':<7} {'WR%':<7} {'P&L':<11} {'Avg':<8} {'MaxDD':<10}")
    print("-" * 80)

    for i, r in enumerate(results_sorted[:20], 1):
        print(f"{i:2d}. {r['config'][:47]:<47} {r['trades']:<7d} {r['win_rate']:<7.1f} "
              f"${r['total_pnl']:<+10,.0f} ${r['avg_pnl']:<+7.2f} ${r['max_dd']:<9,.0f}")

    # Best overall
    if results_sorted:
        best = results_sorted[0]
        print("\n" + "=" * 80)
        print("BEST CONFIGURATION")
        print("=" * 80)
        print(f"\nConfig: {best['config']}")
        print(f"\n  TP/SL: {best['tp']:.1f}% / {best['sl']:.1f}%")
        print(f"  Regime Filter: {best['regime']}")
        print(f"  Time Filter: {best['time_filter']}")
        print(f"\n  Trades: {best['trades']}")
        print(f"  Win Rate: {best['win_rate']:.1f}%")
        print(f"  Total P&L: ${best['total_pnl']:+,.2f}")
        print(f"  Avg P&L: ${best['avg_pnl']:+.2f}")
        print(f"  Avg Size: {best['avg_size']:.2f}x")
        print(f"  Profit Factor: {best['profit_factor']:.2f}")
        print(f"  Max Drawdown: ${best['max_dd']:,.2f}")

        # Compare to baseline
        baseline = next((r for r in results if r['config'].startswith('Baseline_1.0_0.5_None_None')), None)
        if baseline and best['config'] != baseline['config']:
            improvement = best['total_pnl'] - baseline['total_pnl']
            improvement_pct = (improvement / baseline['total_pnl']) * 100 if baseline['total_pnl'] > 0 else 0

            print("\n" + "-" * 80)
            print("VS BASELINE (1.0% TP / 0.5% SL, no filters)")
            print("-" * 80)
            print(f"  Baseline P&L: ${baseline['total_pnl']:+,.2f}")
            print(f"  Best P&L: ${best['total_pnl']:+,.2f}")
            print(f"  Improvement: ${improvement:+,.2f} ({improvement_pct:+.1f}%)")

        # Analyze TP/SL impact
        print("\n" + "=" * 80)
        print("TP/SL ANALYSIS (Grouped by TP/SL ratio)")
        print("=" * 80)

        tpsl_groups = {}
        for r in results:
            key = f"{r['tp']:.1f}/{r['sl']:.1f}"
            if key not in tpsl_groups:
                tpsl_groups[key] = []
            tpsl_groups[key].append(r['total_pnl'])

        tpsl_summary = []
        for key, pnls in tpsl_groups.items():
            tpsl_summary.append({
                'ratio': key,
                'count': len(pnls),
                'avg_pnl': np.mean(pnls),
                'max_pnl': np.max(pnls),
                'min_pnl': np.min(pnls),
            })

        tpsl_summary_sorted = sorted(tpsl_summary, key=lambda x: x['avg_pnl'], reverse=True)

        print(f"{'TP/SL Ratio':<15} {'Tests':<7} {'Avg P&L':<11} {'Max P&L':<11} {'Min P&L':<11}")
        print("-" * 80)
        for s in tpsl_summary_sorted[:10]:
            print(f"{s['ratio']:<15} {s['count']:<7d} ${s['avg_pnl']:<+10,.0f} "
                  f"${s['max_pnl']:<+10,.0f} ${s['min_pnl']:<+10,.0f}")


if __name__ == '__main__':
    main()
