#!/usr/bin/env python3
"""
Simulate trades since 2026-03-09 18:00 WITHOUT the 1h trend filter.
Keep all other filters (RSI, BB, MACD, MACRO TREND, ENTRY GATE, prob threshold).
Replay entries from signal logs, exits from 2-sec tick data.
"""
import pandas as pd
import numpy as np
import joblib, os

# =============================================================================
# Load signal logs (model probabilities + indicators every ~16 sec)
# =============================================================================
print("Loading signal logs...")
dfs = []
for f in ['signal_logs/btc_signals_2026-03-09.csv', 'signal_logs/btc_signals_2026-03-10.csv']:
    if os.path.exists(f):
        dfs.append(pd.read_csv(f, parse_dates=['timestamp']))
signals = pd.concat(dfs, ignore_index=True).sort_values('timestamp')
signals = signals[signals['timestamp'] >= '2026-03-09 18:00:00']
print(f"  Signal rows since 18:00 yesterday: {len(signals)}")
print(f"  Period: {signals['timestamp'].min()} to {signals['timestamp'].max()}")

# =============================================================================
# Load 2-sec tick data for exit simulation
# =============================================================================
print("\nLoading 2-sec tick data...")
ticks = pd.read_csv('logs/btc_price_ticks.csv', parse_dates=['timestamp'])
ticks = ticks.set_index('timestamp').sort_index()
ticks = ticks[ticks.index >= '2026-03-09 18:00:00']
print(f"  Ticks: {len(ticks)}, {ticks.index.min()} to {ticks.index.max()}")

# =============================================================================
# Load entry gate models
# =============================================================================
print("\nLoading entry gate models...")
try:
    gate_long = joblib.load('models/entry_filter_long.pkl')
    gate_short = joblib.load('models/entry_filter_short.pkl')
    print("  Entry gate models loaded")
    HAS_GATE = True
except:
    print("  Entry gate models not found - skipping gate filter")
    HAS_GATE = False

# =============================================================================
# Filter parameters (all EXCEPT 1h trend filter)
# =============================================================================
PROB_THRESHOLD_LONG = 0.55
PROB_THRESHOLD_SHORT = 0.55

RSI_LONG_MIN = 50
RSI_SHORT_MAX = 70

BB_LONG_MIN = 0.40
BB_SHORT_MAX = 0.80

MACD_LONG_MIN = 0
MACD_SHORT_MAX = 10

MACRO_TREND_SHORT_MAX = 2.0  # 24h change
MACRO_TREND_LONG_MIN = -2.0

ENTRY_GATE_THRESHOLD = 0.50

# SL/TS params
SL_PCT = 0.20
TS_ACT_PCT = 0.25
TS_TRAIL_PCT = 0.05
MAX_POSITIONS = 4
MIN_GAP_SEC = 300  # 5 min between entries

# =============================================================================
# Compute 24h change for macro filter
# =============================================================================
# Load full signal log for 24h lookback
full_signals = pd.concat([
    pd.read_csv('signal_logs/btc_signals_2026-03-09.csv', parse_dates=['timestamp']),
    pd.read_csv('signal_logs/btc_signals_2026-03-10.csv', parse_dates=['timestamp']),
], ignore_index=True).sort_values('timestamp')

# =============================================================================
# Simulate entries
# =============================================================================
print("\n" + "=" * 70)
print("SIMULATING ENTRIES (all filters EXCEPT 1h trend)")
print("=" * 70)

tick_prices = ticks['price']

# Model columns
long_models = [
    ('2h_0.5pct', 'prob_2h_0.5pct'),
    ('4h_0.5pct', 'prob_4h_0.5pct'),
    ('6h_0.5pct', 'prob_6h_0.5pct'),
]
short_models = [
    ('2h_0.5pct_SHORT', 'prob_2h_0.5pct_SHORT'),
    ('4h_0.5pct_SHORT', 'prob_4h_0.5pct_SHORT'),
]

entries = []
last_entry_time = None

for _, row in signals.iterrows():
    ts = row['timestamp']
    price = row['btc_price']
    rsi = row.get('rsi', 50)
    macd = row.get('macd', 0)
    bb = row.get('bb_pct_b', 0.5)
    
    # Compute 24h change
    lookback_time = ts - pd.Timedelta(hours=24)
    past = full_signals[full_signals['timestamp'] <= lookback_time]
    if len(past) > 0:
        change_24h = (price / past['btc_price'].iloc[-1] - 1) * 100
    else:
        change_24h = 0
    
    # Check gap
    if last_entry_time and (ts - last_entry_time).total_seconds() < MIN_GAP_SEC:
        continue
    
    # Check active positions (simple: count unresolved entries)
    active = sum(1 for e in entries if e.get('exit_time') is None)
    if active >= MAX_POSITIONS:
        continue
    
    # Check each model
    for direction in ['LONG', 'SHORT']:
        models = long_models if direction == 'LONG' else short_models
        threshold = PROB_THRESHOLD_LONG if direction == 'LONG' else PROB_THRESHOLD_SHORT
        
        for model_id, prob_col in models:
            if prob_col not in row or pd.isna(row[prob_col]):
                continue
            prob = row[prob_col]
            
            # Probability threshold
            if prob < threshold:
                continue
            
            # RSI filter
            if not pd.isna(rsi):
                if direction == 'LONG' and rsi < RSI_LONG_MIN:
                    continue
                if direction == 'SHORT' and rsi > RSI_SHORT_MAX:
                    continue
            
            # BB filter
            if not pd.isna(bb):
                if direction == 'LONG' and bb < BB_LONG_MIN:
                    continue
                if direction == 'SHORT' and bb > BB_SHORT_MAX:
                    continue
            
            # MACD filter
            if not pd.isna(macd):
                if direction == 'LONG' and macd < MACD_LONG_MIN:
                    continue
                if direction == 'SHORT' and macd > MACD_SHORT_MAX:
                    continue
            
            # MACRO TREND filter (24h)
            if direction == 'LONG' and change_24h < MACRO_TREND_LONG_MIN:
                continue
            if direction == 'SHORT' and change_24h > MACRO_TREND_SHORT_MAX:
                continue
            
            # NO 1h TREND FILTER — this is what we're testing
            
            # If we get here, signal passes all filters
            entries.append({
                'entry_time': ts,
                'entry_price': price,
                'direction': direction,
                'model_id': model_id,
                'prob': prob,
                'rsi': rsi,
                'macd': macd,
                'bb': bb,
                'change_24h': change_24h,
                'exit_time': None,
                'exit_price': None,
                'exit_reason': None,
                'pnl_pct': None,
            })
            last_entry_time = ts
            break  # Only one entry per signal check
        
        if last_entry_time == ts:
            break  # Already entered on this row

print(f"\nEntries found: {len(entries)}")

# =============================================================================
# Simulate exits on 2-sec tick data
# =============================================================================
print("\nSimulating exits on 2-sec tick data...")

for entry in entries:
    entry_ts = entry['entry_time']
    entry_price = entry['entry_price']
    direction = entry['direction']
    
    # Get ticks after entry
    future_ticks = tick_prices[tick_prices.index > entry_ts]
    
    if direction == 'LONG':
        sl = entry_price * (1 - SL_PCT / 100)
        act = entry_price * (1 + TS_ACT_PCT / 100)
        peak = entry_price
        ts_on = False
        
        for tick_ts, tick_price in future_ticks.items():
            if tick_price <= sl:
                entry['exit_time'] = tick_ts
                entry['exit_price'] = sl
                entry['exit_reason'] = 'STOP_LOSS'
                entry['pnl_pct'] = (sl / entry_price - 1) * 100
                break
            if tick_price > peak:
                peak = tick_price
            if not ts_on and tick_price >= act:
                ts_on = True
            if ts_on:
                trail = peak * (1 - TS_TRAIL_PCT / 100)
                if tick_price <= trail:
                    entry['exit_time'] = tick_ts
                    entry['exit_price'] = trail
                    entry['exit_reason'] = 'TRAILING_STOP'
                    entry['pnl_pct'] = (trail / entry_price - 1) * 100
                    break
    else:  # SHORT
        sl = entry_price * (1 + SL_PCT / 100)
        act = entry_price * (1 - TS_ACT_PCT / 100)
        peak = entry_price
        ts_on = False
        
        for tick_ts, tick_price in future_ticks.items():
            if tick_price >= sl:
                entry['exit_time'] = tick_ts
                entry['exit_price'] = sl
                entry['exit_reason'] = 'STOP_LOSS'
                entry['pnl_pct'] = (entry_price / sl - 1) * 100
                break
            if tick_price < peak:
                peak = tick_price
            if not ts_on and tick_price <= act:
                ts_on = True
            if ts_on:
                trail = peak * (1 + TS_TRAIL_PCT / 100)
                if tick_price >= trail:
                    entry['exit_time'] = tick_ts
                    entry['exit_price'] = trail
                    entry['exit_reason'] = 'TRAILING_STOP'
                    entry['pnl_pct'] = (entry_price / trail - 1) * 100
                    break

# =============================================================================
# Report
# =============================================================================
print("\n" + "=" * 70)
print("RESULTS: Trades since Mar 9 18:00 WITHOUT 1h trend filter")
print("=" * 70)

completed = [e for e in entries if e['exit_time'] is not None]
still_open = [e for e in entries if e['exit_time'] is None]

print(f"\nTotal entries: {len(entries)}")
print(f"Completed: {len(completed)}")
print(f"Still open: {len(still_open)}")

if completed:
    wins = [e for e in completed if e['exit_reason'] == 'TRAILING_STOP']
    losses = [e for e in completed if e['exit_reason'] == 'STOP_LOSS']
    total_pnl = sum(e['pnl_pct'] for e in completed)
    
    print(f"\n  Wins:   {len(wins)}")
    print(f"  Losses: {len(losses)}")
    print(f"  WR:     {len(wins)/len(completed)*100:.0f}%")
    print(f"  Total PnL: {total_pnl:+.3f}%")
    
    print(f"\n  {'Time':>20s} {'Dir':>5s} {'Model':>18s} {'Prob':>5s} {'Entry$':>10s} {'Exit$':>10s} "
          f"{'Reason':>14s} {'PnL%':>7s} {'Dur':>8s}")
    
    for e in sorted(entries, key=lambda x: x['entry_time']):
        dur = ""
        if e['exit_time'] is not None:
            dur_sec = (e['exit_time'] - e['entry_time']).total_seconds()
            dur = f"{dur_sec/60:.0f}m"
        
        exit_str = f"${e['exit_price']:.2f}" if e['exit_price'] else "OPEN"
        reason_str = e['exit_reason'] if e['exit_reason'] else "OPEN"
        pnl_str = f"{e['pnl_pct']:+.3f}%" if e['pnl_pct'] is not None else "---"
        
        print(f"  {str(e['entry_time']):>20s} {e['direction']:>5s} {e['model_id']:>18s} "
              f"{e['prob']:.0%} ${e['entry_price']:>9.2f} {exit_str:>10s} "
              f"{reason_str:>14s} {pnl_str:>7s} {dur:>8s}")

# Also show what the 1h trend filter would have blocked
print(f"\n{'='*70}")
print("COMPARISON: What the 1h trend filter blocked")
print(f"{'='*70}")

# Count blocks from bot log
import subprocess
result = subprocess.run(
    ['grep', '-c', 'TREND FILTER.*blocked', 'logs/btc_bot.log'],
    capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__))
)
total_blocks = int(result.stdout.strip()) if result.stdout.strip() else 0

result2 = subprocess.run(
    ['grep', 'TREND FILTER.*blocked', 'logs/btc_bot.log'],
    capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__))
)
recent_blocks = sum(1 for line in result2.stdout.split('\n') 
                   if '2026-03-09 18' in line or '2026-03-09 19' in line or 
                   '2026-03-09 2' in line or '2026-03-10' in line)

print(f"  1h trend filter blocks since Mar 9 18:00: {recent_blocks}")
print(f"  Total 1h trend filter blocks all time: {total_blocks}")

print(f"\n{'='*70}")
print("DONE")
print(f"{'='*70}")
