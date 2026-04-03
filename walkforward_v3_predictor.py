"""
Walk-Forward Backtest for BTC V3 Predictor Models

Tests the BTCPredictor ensemble (2h/4h/6h) on historical 5-min data using
the exact same predict() pipeline the bot uses live.

Walk-forward approach:
- Slide through historical data in steps
- At each step, feed 250 bars to predictor.predict()
- If signal != NEUTRAL and confidence >= threshold, simulate trade
- Trade outcome: scan forward bars for TP=1% or SL=0.5% hit
- Report by quarter and overall

Uses BTC_5min_8years.parquet (2017-2025, 839K bars).
V3 models trained on 2022-2026, so we test on:
- In-sample: 2022-2025 (model saw this data during training)
- We focus on the LAST portion as most representative of live performance.

Run: python3 walkforward_v3_predictor.py
"""

import pandas as pd
import numpy as np
import sys
import os
import time
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
MIN_CONFIDENCE = 0.60  # Optimized for 22-feature model (0.60 LONG / 0.30 SHORT)
MAX_FORWARD_BARS = 72  # 6 hours timeout (72 * 5min)
STEP_BARS = 12  # Evaluate every 12 bars (1 hour) to keep runtime reasonable
LOOKBACK = 250  # Bars needed by predictor
COMMISSION = 2.02  # Round-trip commission per MBT contract ($)
CONTRACT_VALUE = 0.1  # MBT = 0.1 BTC

# Date range for test (V3 models trained on 2022-2026 data)
# Use 2024-2025 as the most recent / representative period
TEST_START = '2024-01-01'
TEST_END = '2025-12-01'


def simulate_trade(df_forward, direction, entry_price, tp_pct, sl_pct, max_bars):
    """
    Simulate a trade by scanning forward bars for TP or SL hit.
    
    Returns: (exit_reason, exit_price, bars_held, pnl_pct)
    """
    for i in range(1, min(len(df_forward), max_bars + 1)):
        bar = df_forward.iloc[i]
        high = bar['high']
        low = bar['low']
        
        if direction == 'LONG':
            tp_price = entry_price * (1 + tp_pct / 100)
            sl_price = entry_price * (1 - sl_pct / 100)
            
            # Check SL first (conservative: assume SL hit before TP on same bar)
            if low <= sl_price:
                pnl_pct = -sl_pct
                return 'STOP_LOSS', sl_price, i, pnl_pct
            if high >= tp_price:
                pnl_pct = tp_pct
                return 'TAKE_PROFIT', tp_price, i, pnl_pct
                
        elif direction == 'SHORT':
            tp_price = entry_price * (1 - tp_pct / 100)
            sl_price = entry_price * (1 + sl_pct / 100)
            
            # Check SL first
            if high >= sl_price:
                pnl_pct = -sl_pct
                return 'STOP_LOSS', sl_price, i, pnl_pct
            if low <= tp_price:
                pnl_pct = tp_pct
                return 'TAKE_PROFIT', tp_price, i, pnl_pct
    
    # Timeout: use close of last bar
    last_close = df_forward.iloc[min(len(df_forward) - 1, max_bars)]['close']
    if direction == 'LONG':
        pnl_pct = (last_close / entry_price - 1) * 100
    else:
        pnl_pct = (entry_price / last_close - 1) * 100
    
    return 'TIMEOUT', last_close, min(len(df_forward) - 1, max_bars), pnl_pct


def run_walkforward():
    print("=" * 70)
    print("BTC V3 PREDICTOR — WALK-FORWARD BACKTEST")
    print("=" * 70)
    
    # Load data
    print(f"\nLoading data from {DATA_PATH}...")
    df_raw = pd.read_parquet(DATA_PATH)
    
    # Rename columns to lowercase (predictor expects lowercase)
    df_raw.columns = [c.lower() for c in df_raw.columns]
    
    # Filter to test range
    df = df_raw.loc[TEST_START:TEST_END].copy()
    print(f"Test period: {df.index[0]} to {df.index[-1]}")
    print(f"Total bars: {len(df):,}")
    print(f"TP: {TP_PCT}% | SL: {SL_PCT}% | Timeout: {MAX_FORWARD_BARS} bars ({MAX_FORWARD_BARS*5/60:.0f}h)")
    print(f"Min confidence: {MIN_CONFIDENCE}")
    print(f"Step: every {STEP_BARS} bars ({STEP_BARS*5} min)")
    print(f"Breakeven WR: {SL_PCT/(TP_PCT+SL_PCT)*100:.1f}%")
    
    # Load predictor
    print(f"\nLoading predictor from {MODEL_DIR}...")
    predictor = BTCPredictor(model_dir=MODEL_DIR)
    predictor.LONG_THRESHOLD = 0.60
    predictor.SHORT_THRESHOLD = 0.30
    print(f"Thresholds: LONG > {predictor.LONG_THRESHOLD}, SHORT < {predictor.SHORT_THRESHOLD}")
    
    # Walk forward
    trades = []
    signals_count = {'LONG': 0, 'SHORT': 0, 'NEUTRAL': 0}
    eval_count = 0
    
    # We need LOOKBACK bars before the first evaluation point
    # and MAX_FORWARD_BARS after the last evaluation point
    start_idx = LOOKBACK
    end_idx = len(df) - MAX_FORWARD_BARS - 1
    
    total_evals = (end_idx - start_idx) // STEP_BARS
    print(f"\nRunning {total_evals:,} evaluations...")
    print("-" * 70)
    
    t0 = time.time()
    last_trade_bar = -999  # Prevent overlapping trades
    
    for idx in range(start_idx, end_idx, STEP_BARS):
        eval_count += 1
        
        # Progress
        if eval_count % 500 == 0:
            elapsed = time.time() - t0
            pct = eval_count / total_evals * 100
            eta = elapsed / eval_count * (total_evals - eval_count)
            print(f"  [{pct:5.1f}%] {eval_count:,}/{total_evals:,} evals | "
                  f"{len(trades)} trades | elapsed {elapsed:.0f}s | ETA {eta:.0f}s")
        
        # Skip if we're still in a previous trade's hold period
        if idx - last_trade_bar < 6:  # Minimum 30 min gap between entries
            continue
        
        # Get 250-bar window
        window = df.iloc[idx - LOOKBACK:idx].copy()
        
        if len(window) < LOOKBACK:
            continue
        
        # Predict
        try:
            signal, confidence, details = predictor.predict(window)
        except Exception as e:
            continue
        
        signals_count[signal] += 1
        
        # Filter
        if signal == 'NEUTRAL':
            continue
        if confidence < MIN_CONFIDENCE:
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
        
        # Calculate dollar P&L (1 MBT contract = 0.1 BTC)
        pnl_dollar = (pnl_pct / 100) * entry_price * CONTRACT_VALUE - COMMISSION
        
        trades.append({
            'entry_time': entry_time,
            'direction': signal,
            'confidence': confidence,
            'avg_prob': details.get('avg_probability', 0),
            'prob_2h': details.get('probabilities', {}).get('2h', 0),
            'prob_4h': details.get('probabilities', {}).get('4h', 0),
            'prob_6h': details.get('probabilities', {}).get('6h', 0),
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'bars_held': bars_held,
            'pnl_pct': pnl_pct,
            'pnl_dollar': pnl_dollar,
            'quarter': f"{entry_time.year}-Q{(entry_time.month-1)//3+1}",
            'month': entry_time.strftime('%Y-%m'),
        })
        
        last_trade_bar = idx + bars_held  # Don't enter during this trade's hold
    
    elapsed = time.time() - t0
    print(f"\nCompleted {eval_count:,} evaluations in {elapsed:.1f}s")
    print(f"Signal distribution: {signals_count}")
    
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
    losses = (df_trades['pnl_pct'] <= 0).sum()
    wr = wins / total * 100
    
    total_pnl = df_trades['pnl_dollar'].sum()
    avg_pnl = df_trades['pnl_dollar'].mean()
    avg_pnl_pct = df_trades['pnl_pct'].mean()
    
    avg_winner = df_trades.loc[df_trades['pnl_pct'] > 0, 'pnl_dollar'].mean() if wins > 0 else 0
    avg_loser = df_trades.loc[df_trades['pnl_pct'] <= 0, 'pnl_dollar'].mean() if losses > 0 else 0
    
    gross_wins = df_trades.loc[df_trades['pnl_pct'] > 0, 'pnl_dollar'].sum()
    gross_losses = abs(df_trades.loc[df_trades['pnl_pct'] <= 0, 'pnl_dollar'].sum())
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')
    
    print(f"Total trades:    {total}")
    print(f"Wins / Losses:   {wins} / {losses}")
    print(f"Win rate:        {wr:.1f}% (breakeven: {SL_PCT/(TP_PCT+SL_PCT)*100:.1f}%)")
    print(f"Total P&L:       ${total_pnl:+,.2f}")
    print(f"Avg P&L/trade:   ${avg_pnl:+.2f} ({avg_pnl_pct:+.3f}%)")
    print(f"Avg winner:      ${avg_winner:+.2f}")
    print(f"Avg loser:       ${avg_loser:+.2f}")
    print(f"Profit factor:   {profit_factor:.2f}")
    
    # By direction
    print("\n" + "-" * 70)
    print("BY DIRECTION")
    print("-" * 70)
    for direction in ['LONG', 'SHORT']:
        dt = df_trades[df_trades['direction'] == direction]
        if len(dt) == 0:
            print(f"  {direction}: 0 trades")
            continue
        w = (dt['pnl_pct'] > 0).sum()
        l = (dt['pnl_pct'] <= 0).sum()
        wr_d = w / len(dt) * 100
        pnl_d = dt['pnl_dollar'].sum()
        print(f"  {direction:5s}: {len(dt):4d} trades | {w}W/{l}L | WR {wr_d:.1f}% | P&L ${pnl_d:+,.2f} | Avg ${dt['pnl_dollar'].mean():+.2f}")
    
    # By exit reason
    print("\n" + "-" * 70)
    print("BY EXIT REASON")
    print("-" * 70)
    for reason in ['TAKE_PROFIT', 'STOP_LOSS', 'TIMEOUT']:
        dt = df_trades[df_trades['exit_reason'] == reason]
        if len(dt) == 0:
            continue
        pct = len(dt) / total * 100
        pnl_r = dt['pnl_dollar'].sum()
        print(f"  {reason:15s}: {len(dt):4d} ({pct:.1f}%) | P&L ${pnl_r:+,.2f} | Avg bars {dt['bars_held'].mean():.1f}")
    
    # By quarter
    print("\n" + "-" * 70)
    print("BY QUARTER (Walk-Forward)")
    print("-" * 70)
    print(f"  {'Quarter':<10s} {'Trades':>6s} {'Wins':>5s} {'WR%':>6s} {'P&L($)':>10s} {'AvgP&L':>8s} {'PF':>6s}")
    print(f"  {'-'*10} {'-'*6} {'-'*5} {'-'*6} {'-'*10} {'-'*8} {'-'*6}")
    
    profitable_quarters = 0
    total_quarters = 0
    
    for quarter in sorted(df_trades['quarter'].unique()):
        qt = df_trades[df_trades['quarter'] == quarter]
        w = (qt['pnl_pct'] > 0).sum()
        l = (qt['pnl_pct'] <= 0).sum()
        wr_q = w / len(qt) * 100 if len(qt) > 0 else 0
        pnl_q = qt['pnl_dollar'].sum()
        avg_q = qt['pnl_dollar'].mean()
        gw = qt.loc[qt['pnl_pct'] > 0, 'pnl_dollar'].sum()
        gl = abs(qt.loc[qt['pnl_pct'] <= 0, 'pnl_dollar'].sum())
        pf_q = gw / gl if gl > 0 else float('inf')
        
        marker = "✓" if pnl_q > 0 else "✗"
        if pnl_q > 0:
            profitable_quarters += 1
        total_quarters += 1
        
        print(f"  {quarter:<10s} {len(qt):>6d} {w:>5d} {wr_q:>5.1f}% ${pnl_q:>+9,.2f} ${avg_q:>+7.2f} {pf_q:>5.2f} {marker}")
    
    print(f"\n  Profitable quarters: {profitable_quarters}/{total_quarters} "
          f"({profitable_quarters/total_quarters*100:.0f}%)")
    
    # By month
    print("\n" + "-" * 70)
    print("BY MONTH")
    print("-" * 70)
    print(f"  {'Month':<10s} {'Trades':>6s} {'WR%':>6s} {'P&L($)':>10s}")
    print(f"  {'-'*10} {'-'*6} {'-'*6} {'-'*10}")
    
    profitable_months = 0
    total_months = 0
    for month in sorted(df_trades['month'].unique()):
        mt = df_trades[df_trades['month'] == month]
        w = (mt['pnl_pct'] > 0).sum()
        wr_m = w / len(mt) * 100 if len(mt) > 0 else 0
        pnl_m = mt['pnl_dollar'].sum()
        marker = "✓" if pnl_m > 0 else "✗"
        if pnl_m > 0:
            profitable_months += 1
        total_months += 1
        print(f"  {month:<10s} {len(mt):>6d} {wr_m:>5.1f}% ${pnl_m:>+9,.2f} {marker}")
    
    print(f"\n  Profitable months: {profitable_months}/{total_months} "
          f"({profitable_months/total_months*100:.0f}%)")
    
    # Confidence analysis
    print("\n" + "-" * 70)
    print("BY CONFIDENCE BAND")
    print("-" * 70)
    bins = [(0.58, 0.62), (0.62, 0.66), (0.66, 0.70), (0.70, 0.80), (0.80, 1.01)]
    for lo, hi in bins:
        ct = df_trades[(df_trades['confidence'] >= lo) & (df_trades['confidence'] < hi)]
        if len(ct) == 0:
            continue
        w = (ct['pnl_pct'] > 0).sum()
        wr_c = w / len(ct) * 100
        pnl_c = ct['pnl_dollar'].sum()
        print(f"  [{lo:.2f}-{hi:.2f}): {len(ct):4d} trades | WR {wr_c:.1f}% | P&L ${pnl_c:+,.2f}")
    
    # Drawdown analysis
    print("\n" + "-" * 70)
    print("EQUITY CURVE & MAX DRAWDOWN")
    print("-" * 70)
    
    equity = df_trades['pnl_dollar'].cumsum()
    peak = equity.cummax()
    drawdown = equity - peak
    max_dd = drawdown.min()
    max_dd_idx = drawdown.idxmin()
    
    print(f"  Peak equity:      ${peak.max():+,.2f}")
    print(f"  Max drawdown:     ${max_dd:,.2f}")
    print(f"  Max DD at trade:  #{max_dd_idx} ({df_trades.iloc[max_dd_idx]['entry_time']})")
    print(f"  Final equity:     ${equity.iloc[-1]:+,.2f}")
    
    # Save trades
    output_path = os.path.join(BOT_DIR, 'results', 'v3_predictor_walkforward.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_trades.to_csv(output_path, index=False)
    print(f"\nTrades saved to: {output_path}")
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    run_walkforward()
