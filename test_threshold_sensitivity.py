"""
Test V3 Predictor with Different Probability Thresholds

Uses Windsurf's realistic simulation approach (bar-by-bar TP/SL check)
but tests multiple confidence thresholds to find optimal settings.

Tests both 17-feature and 22-feature models.
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
from collections import defaultdict

# Add project dir for imports
BOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BOT_DIR)

from btc_model_package.predictor import BTCPredictor


# =============================================================================
# CONFIG
# =============================================================================
DATA_PATH = os.path.join(BOT_DIR, 'data', 'BTC_5min_8years.parquet')
MODEL_DIR = os.path.join(BOT_DIR, 'btc_model_package')

TP_PCT = 1.0    # Take profit %
SL_PCT = 0.5    # Stop loss %
MAX_FORWARD_BARS = 72  # 6 hours timeout
STEP_BARS = 12  # Evaluate every hour
LOOKBACK = 250

# Test these thresholds
THRESHOLDS_TO_TEST = [
    (0.50, 0.40),  # Less selective
    (0.55, 0.35),  # Current default
    (0.60, 0.30),  # More selective for LONG
    (0.65, 0.25),  # Very selective for LONG
    (0.70, 0.20),  # Extremely selective
]

# Commission scenarios
COMMISSION_SCENARIOS = [
    ('No Commission', 0.0),
    ('IB MBT ($2.02)', 2.02),
]

CONTRACT_VALUE = 0.1  # MBT = 0.1 BTC

# Date range
TEST_START = '2024-01-01'
TEST_END = '2025-12-01'


def simulate_trade(df_forward, direction, entry_price, tp_pct, sl_pct, max_bars):
    """
    Simulate a trade by scanning forward bars for TP or SL hit.
    Uses Windsurf's realistic approach.
    """
    for i in range(1, min(len(df_forward), max_bars + 1)):
        bar = df_forward.iloc[i]
        high = bar['high']
        low = bar['low']

        if direction == 'LONG':
            tp_price = entry_price * (1 + tp_pct / 100)
            sl_price = entry_price * (1 - sl_pct / 100)

            # Check SL first (conservative)
            if low <= sl_price:
                pnl_pct = -sl_pct
                return 'STOP_LOSS', sl_price, i, pnl_pct
            if high >= tp_price:
                pnl_pct = tp_pct
                return 'TAKE_PROFIT', tp_price, i, pnl_pct

        elif direction == 'SHORT':
            tp_price = entry_price * (1 - tp_pct / 100)
            sl_price = entry_price * (1 + sl_pct / 100)

            if high >= sl_price:
                pnl_pct = -sl_pct
                return 'STOP_LOSS', sl_price, i, pnl_pct
            if low <= tp_price:
                pnl_pct = tp_pct
                return 'TAKE_PROFIT', tp_price, i, pnl_pct

    # Timeout
    last_close = df_forward.iloc[min(len(df_forward) - 1, max_bars)]['close']
    if direction == 'LONG':
        pnl_pct = (last_close / entry_price - 1) * 100
    else:
        pnl_pct = (entry_price / last_close - 1) * 100

    return 'TIMEOUT', last_close, min(len(df_forward) - 1, max_bars), pnl_pct


def run_backtest_with_threshold(df, predictor, long_threshold, short_threshold, commission):
    """Run backtest with specific threshold settings"""

    # Set thresholds
    predictor.LONG_THRESHOLD = long_threshold
    predictor.SHORT_THRESHOLD = short_threshold

    trades = []

    start_idx = LOOKBACK
    end_idx = len(df) - MAX_FORWARD_BARS - 1

    last_trade_bar = -999

    for idx in range(start_idx, end_idx, STEP_BARS):
        # Skip if in previous trade
        if idx - last_trade_bar < 6:
            continue

        # Get window
        window = df.iloc[idx - LOOKBACK:idx].copy()

        if len(window) < LOOKBACK:
            continue

        # Predict
        try:
            signal, confidence, details = predictor.predict(window)
        except Exception as e:
            continue

        # Filter
        if signal == 'NEUTRAL':
            continue

        # Simulate trade
        entry_price = window['close'].iloc[-1]
        entry_time = window.index[-1]
        forward_data = df.iloc[idx:idx + MAX_FORWARD_BARS + 1]

        if len(forward_data) < 2:
            continue

        exit_reason, exit_price, bars_held, pnl_pct = simulate_trade(
            forward_data, signal, entry_price, TP_PCT, SL_PCT, MAX_FORWARD_BARS
        )

        # Calculate dollar P&L
        pnl_dollar = (pnl_pct / 100) * entry_price * CONTRACT_VALUE - commission

        trades.append({
            'entry_time': entry_time,
            'direction': signal,
            'confidence': confidence,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'bars_held': bars_held,
            'pnl_pct': pnl_pct,
            'pnl_dollar': pnl_dollar,
        })

        last_trade_bar = idx + bars_held

    return trades


def analyze_results(trades):
    """Calculate metrics from trades"""
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
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    equity = df['pnl_dollar'].cumsum()
    peak = equity.cummax()
    drawdown = equity - peak
    max_dd = drawdown.min()

    return {
        'trades': total,
        'win_rate': wr,
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'profit_factor': profit_factor,
        'max_dd': max_dd,
        'gross_wins': gross_wins,
        'gross_losses': gross_losses,
    }


def main():
    print("=" * 80)
    print("V3 PREDICTOR THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 80)

    # Load data
    print(f"\nLoading data from {DATA_PATH}...")
    df_raw = pd.read_parquet(DATA_PATH)
    df_raw.columns = [c.lower() for c in df_raw.columns]
    df = df_raw.loc[TEST_START:TEST_END].copy()

    print(f"Test period: {df.index[0]} to {df.index[-1]}")
    print(f"Total bars: {len(df):,}")
    print(f"TP: {TP_PCT}% | SL: {SL_PCT}% | Breakeven WR: {SL_PCT/(TP_PCT+SL_PCT)*100:.1f}%")

    # Load predictor
    print(f"\nLoading predictor from {MODEL_DIR}...")
    predictor = BTCPredictor(model_dir=MODEL_DIR)

    print(f"Models loaded: {len(predictor.models)} horizons")
    print(f"Features: {len(predictor.features)}")

    # Test each commission scenario
    for comm_name, commission in COMMISSION_SCENARIOS:
        print("\n" + "=" * 80)
        print(f"SCENARIO: {comm_name}")
        print("=" * 80)

        results_table = []

        for long_th, short_th in THRESHOLDS_TO_TEST:
            print(f"\nTesting LONG_TH={long_th:.2f}, SHORT_TH={short_th:.2f}...", end=' ')

            trades = run_backtest_with_threshold(
                df, predictor, long_th, short_th, commission
            )

            if not trades:
                print("No trades")
                continue

            metrics = analyze_results(trades)

            print(f"{metrics['trades']} trades, {metrics['win_rate']:.1f}% WR, "
                  f"${metrics['total_pnl']:+,.0f} P&L")

            results_table.append({
                'LONG_TH': long_th,
                'SHORT_TH': short_th,
                'Trades': metrics['trades'],
                'WR%': metrics['win_rate'],
                'Total_PnL': metrics['total_pnl'],
                'Avg_PnL': metrics['avg_pnl'],
                'PF': metrics['profit_factor'],
                'Max_DD': metrics['max_dd'],
            })

        # Print summary table
        print("\n" + "-" * 80)
        print("SUMMARY TABLE")
        print("-" * 80)
        print(f"{'LONG_TH':<9} {'SHORT_TH':<10} {'Trades':<7} {'WR%':<7} "
              f"{'Total P&L':<12} {'Avg P&L':<10} {'PF':<6} {'Max DD':<10}")
        print("-" * 80)

        for row in results_table:
            marker = "✓" if row['Total_PnL'] > 0 else "✗"
            print(f"{row['LONG_TH']:<9.2f} {row['SHORT_TH']:<10.2f} "
                  f"{row['Trades']:<7d} {row['WR%']:<7.1f} "
                  f"${row['Total_PnL']:<+11,.0f} ${row['Avg_PnL']:<+9.2f} "
                  f"{row['PF']:<6.2f} ${row['Max_DD']:<9,.0f} {marker}")

        # Find best threshold
        if results_table:
            best = max(results_table, key=lambda x: x['Total_PnL'])
            print("\n" + "-" * 80)
            print(f"BEST THRESHOLD: LONG={best['LONG_TH']:.2f}, SHORT={best['SHORT_TH']:.2f}")
            print(f"  Trades: {best['Trades']}")
            print(f"  Win Rate: {best['WR%']:.1f}%")
            print(f"  Total P&L: ${best['Total_PnL']:+,.2f}")
            print(f"  Avg P&L: ${best['Avg_PnL']:+.2f}")
            print(f"  Profit Factor: {best['PF']:.2f}")
            print(f"  Max DD: ${best['Max_DD']:,.2f}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == '__main__':
    main()
