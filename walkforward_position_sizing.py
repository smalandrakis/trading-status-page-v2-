"""
Walk-Forward Backtest with Confidence-Based Position Sizing

Tests the optimal position sizing strategy found:
- Base thresholds: LONG=0.65, SHORT=0.25
- Position scaling: size = (confidence - 0.60) × 20 (1-5x)

Windsurf: Run this to verify the +$7,047 result
Usage: python3 walkforward_position_sizing.py
"""

import pandas as pd
import numpy as np
import sys
import os
import time
from datetime import datetime
from collections import defaultdict

BOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BOT_DIR)

from btc_model_package.predictor import BTCPredictor

# =============================================================================
# CONFIG
# =============================================================================
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

# OPTIMAL THRESHOLDS
LONG_THRESHOLD = 0.65
SHORT_THRESHOLD = 0.25

# POSITION SIZING
MIN_SIZE = 1.0
MAX_SIZE = 5.0
SCALING_FACTOR = 20


def calculate_position_size(confidence):
    """Calculate position size: (confidence - 0.60) × 20, capped 1-5x"""
    size = (confidence - 0.60) * SCALING_FACTOR
    return max(MIN_SIZE, min(MAX_SIZE, size))


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


def run_walkforward():
    print("=" * 70)
    print("BTC V3 WALK-FORWARD - CONFIDENCE-BASED POSITION SIZING")
    print("=" * 70)

    # Load data
    print(f"\nLoading data from {DATA_PATH}...")
    df_raw = pd.read_parquet(DATA_PATH)
    df_raw.columns = [c.lower() for c in df_raw.columns]
    df = df_raw.loc[TEST_START:TEST_END].copy()

    print(f"Test period: {df.index[0]} to {df.index[-1]}")
    print(f"Total bars: {len(df):,}")
    print(f"TP: {TP_PCT}% | SL: {SL_PCT}% | Timeout: {MAX_FORWARD_BARS} bars")
    print(f"Thresholds: LONG={LONG_THRESHOLD}, SHORT={SHORT_THRESHOLD}")
    print(f"Position sizing: {MIN_SIZE}x - {MAX_SIZE}x (linear scale)")
    print(f"Breakeven WR: {SL_PCT/(TP_PCT+SL_PCT)*100:.1f}%")

    # Load predictor
    print(f"\nLoading predictor from {MODEL_DIR}...")
    predictor = BTCPredictor(model_dir=MODEL_DIR)
    predictor.LONG_THRESHOLD = LONG_THRESHOLD
    predictor.SHORT_THRESHOLD = SHORT_THRESHOLD

    # Walk forward
    trades = []
    eval_count = 0

    start_idx = LOOKBACK
    end_idx = len(df) - MAX_FORWARD_BARS - 1
    total_evals = (end_idx - start_idx) // STEP_BARS

    print(f"\nRunning {total_evals:,} evaluations...")
    print("-" * 70)

    t0 = time.time()
    last_trade_bar = -999

    for idx in range(start_idx, end_idx, STEP_BARS):
        eval_count += 1

        if eval_count % 500 == 0:
            elapsed = time.time() - t0
            pct = eval_count / total_evals * 100
            eta = elapsed / eval_count * (total_evals - eval_count)
            print(f"  [{pct:5.1f}%] {eval_count:,}/{total_evals:,} evals | "
                  f"{len(trades)} trades | elapsed {elapsed:.0f}s | ETA {eta:.0f}s")

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

        # Calculate position size
        position_size = calculate_position_size(confidence)

        entry_price = window['close'].iloc[-1]
        entry_time = window.index[-1]
        forward_data = df.iloc[idx:idx + MAX_FORWARD_BARS + 1]

        if len(forward_data) < 2:
            continue

        exit_reason, exit_price, bars_held, pnl_pct = simulate_trade(
            forward_data, signal, entry_price, TP_PCT, SL_PCT, MAX_FORWARD_BARS
        )

        # P&L with position sizing
        gross_pnl = (pnl_pct / 100) * entry_price * CONTRACT_VALUE * position_size
        commission_cost = COMMISSION * position_size
        net_pnl = gross_pnl - commission_cost

        trades.append({
            'entry_time': entry_time,
            'direction': signal,
            'confidence': confidence,
            'position_size': position_size,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'bars_held': bars_held,
            'pnl_pct': pnl_pct,
            'gross_pnl': gross_pnl,
            'commission': commission_cost,
            'net_pnl': net_pnl,
            'quarter': f"{entry_time.year}-Q{(entry_time.month-1)//3+1}",
            'month': entry_time.strftime('%Y-%m'),
        })

        last_trade_bar = idx + bars_held

    elapsed = time.time() - t0
    print(f"\nCompleted {eval_count:,} evaluations in {elapsed:.1f}s")

    if not trades:
        print("\nNo trades generated!")
        return

    # ==========================================================================
    # RESULTS
    # ==========================================================================
    df_trades = pd.DataFrame(trades)

    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)

    total = len(df_trades)
    wins = (df_trades['pnl_pct'] > 0).sum()
    wr = wins / total * 100

    total_pnl = df_trades['net_pnl'].sum()
    total_gross = df_trades['gross_pnl'].sum()
    total_commission = df_trades['commission'].sum()
    avg_pnl = df_trades['net_pnl'].mean()
    avg_size = df_trades['position_size'].mean()

    avg_winner = df_trades.loc[df_trades['pnl_pct'] > 0, 'net_pnl'].mean() if wins > 0 else 0
    avg_loser = df_trades.loc[df_trades['pnl_pct'] <= 0, 'net_pnl'].mean() if total > wins else 0

    gross_wins = df_trades.loc[df_trades['pnl_pct'] > 0, 'net_pnl'].sum()
    gross_losses = abs(df_trades.loc[df_trades['pnl_pct'] <= 0, 'net_pnl'].sum())
    pf = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    print(f"Total trades:    {total}")
    print(f"Avg position:    {avg_size:.2f}x contracts")
    print(f"Wins / Losses:   {wins} / {total - wins}")
    print(f"Win rate:        {wr:.1f}% (breakeven: {SL_PCT/(TP_PCT+SL_PCT)*100:.1f}%)")
    print(f"Gross P&L:       ${total_gross:+,.2f}")
    print(f"Total commission: ${total_commission:,.2f}")
    print(f"Net P&L:         ${total_pnl:+,.2f}")
    print(f"Avg P&L/trade:   ${avg_pnl:+.2f}")
    print(f"Avg winner:      ${avg_winner:+.2f}")
    print(f"Avg loser:       ${avg_loser:+.2f}")
    print(f"Profit factor:   {pf:.2f}")
    print(f"Commission %:    {total_commission/abs(total_gross)*100:.1f}% of gross")

    # By direction
    print("\n" + "-" * 70)
    print("BY DIRECTION")
    print("-" * 70)
    for direction in ['LONG', 'SHORT']:
        dt = df_trades[df_trades['direction'] == direction]
        if len(dt) == 0:
            continue
        w = (dt['pnl_pct'] > 0).sum()
        wr_d = w / len(dt) * 100
        pnl_d = dt['net_pnl'].sum()
        avg_size_d = dt['position_size'].mean()
        print(f"  {direction:5s}: {len(dt):4d} trades | Avg size {avg_size_d:.2f}x | "
              f"{w}W/{len(dt)-w}L | WR {wr_d:.1f}% | P&L ${pnl_d:+,.2f} | "
              f"Avg ${dt['net_pnl'].mean():+.2f}")

    # By exit reason
    print("\n" + "-" * 70)
    print("BY EXIT REASON")
    print("-" * 70)
    for reason in ['TAKE_PROFIT', 'STOP_LOSS', 'TIMEOUT']:
        dt = df_trades[df_trades['exit_reason'] == reason]
        if len(dt) == 0:
            continue
        pct = len(dt) / total * 100
        pnl_r = dt['net_pnl'].sum()
        print(f"  {reason:15s}: {len(dt):4d} ({pct:.1f}%) | P&L ${pnl_r:+,.2f}")

    # By quarter
    print("\n" + "-" * 70)
    print("BY QUARTER")
    print("-" * 70)
    print(f"  {'Quarter':<10s} {'Trades':>6s} {'Wins':>5s} {'WR%':>6s} "
          f"{'P&L($)':>10s} {'AvgSize':>8s}")
    print(f"  {'-'*10} {'-'*6} {'-'*5} {'-'*6} {'-'*10} {'-'*8}")

    profitable_quarters = 0
    for quarter in sorted(df_trades['quarter'].unique()):
        qt = df_trades[df_trades['quarter'] == quarter]
        w = (qt['pnl_pct'] > 0).sum()
        wr_q = w / len(qt) * 100
        pnl_q = qt['net_pnl'].sum()
        avg_size_q = qt['position_size'].mean()
        marker = "✓" if pnl_q > 0 else "✗"
        if pnl_q > 0:
            profitable_quarters += 1
        print(f"  {quarter:<10s} {len(qt):>6d} {w:>5d} {wr_q:>5.1f}% "
              f"${pnl_q:>+9,.2f} {avg_size_q:>7.2f}x {marker}")

    print(f"\n  Profitable quarters: {profitable_quarters}/{len(df_trades['quarter'].unique())}")

    # Position size distribution
    print("\n" + "-" * 70)
    print("POSITION SIZE DISTRIBUTION")
    print("-" * 70)
    size_bins = [(1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0),
                 (3.0, 3.5), (3.5, 4.0), (4.0, 4.5), (4.5, 5.5)]
    for lo, hi in size_bins:
        st = df_trades[(df_trades['position_size'] >= lo) &
                       (df_trades['position_size'] < hi)]
        if len(st) == 0:
            continue
        w = (st['pnl_pct'] > 0).sum()
        wr_s = w / len(st) * 100
        pnl_s = st['net_pnl'].sum()
        avg_conf = st['confidence'].mean()
        print(f"  [{lo:.1f}-{hi:.1f}x): {len(st):4d} trades | Avg conf {avg_conf:.1%} | "
              f"WR {wr_s:.1f}% | P&L ${pnl_s:+,.2f}")

    # Drawdown
    print("\n" + "-" * 70)
    print("EQUITY CURVE & DRAWDOWN")
    print("-" * 70)
    equity = df_trades['net_pnl'].cumsum()
    peak = equity.cummax()
    drawdown = equity - peak
    max_dd = drawdown.min()
    max_dd_idx = drawdown.idxmin()

    print(f"  Peak equity:      ${peak.max():+,.2f}")
    print(f"  Max drawdown:     ${max_dd:,.2f}")
    print(f"  Final equity:     ${equity.iloc[-1]:+,.2f}")

    # Save
    output_path = os.path.join(BOT_DIR, 'results', 'position_sizing_walkforward.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_trades.to_csv(output_path, index=False)
    print(f"\nTrades saved to: {output_path}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    run_walkforward()
