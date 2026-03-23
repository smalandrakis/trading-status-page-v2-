#!/usr/bin/env python3
"""
Simulate proposed ensemble bot filters on all historical trades.

Filters tested:
1. RSI floor for SHORT: block if RSI < 40 (oversold)
2. RSI ceiling for LONG: block if RSI > 70 (overbought)
3. 6h LONG cross-timeframe: require 2h or 4h prob > 40%

Uses BTC_features.parquet for RSI and ensemble bot logs for model probabilities.
"""

import sqlite3
import pandas as pd
import numpy as np
import re
import os
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'trades.db')
PARQUET_PATH = os.path.join(BASE_DIR, 'data', 'BTC_features.parquet')
ARCHIVE_PATH = os.path.join(BASE_DIR, 'data', 'archive', 'BTC_features_archive.parquet')
LOG_PATH = os.path.join(BASE_DIR, 'logs', 'btc_ensemble_bot.log')

# Load trades
conn = sqlite3.connect(DB_PATH)
trades = pd.read_sql("""
    SELECT *, rowid as trade_id FROM trades 
    WHERE model_id NOT LIKE 'legacy%'
    ORDER BY entry_time
""", conn)
conn.close()

print(f"Total trades: {len(trades)}")
print(f"Date range: {trades['entry_time'].iloc[0]} to {trades['entry_time'].iloc[-1]}")

# Load parquet for RSI
pq = None
for path in [ARCHIVE_PATH, PARQUET_PATH]:
    if os.path.exists(path):
        try:
            pq = pd.read_parquet(path)
            print(f"Loaded parquet: {len(pq)} rows from {path}")
            break
        except:
            pass

# Parse entry probabilities from logs
# Format: "BTC: $73720.00 | Pos: 0/4 | 2h_0.5pct:34% | 4h_0.5pct:33% | 6h_0.5pct:29% | 2h_0.5pct_SHORT:65% | 4h_0.5pct_SHORT:58%"
prob_pattern = re.compile(r'(\d+h_\d+\.\d+pct(?:_SHORT)?):(\d+)%')

# Build a time-indexed dict of probabilities from recent logs
print("\nParsing log for model probabilities...")
log_probs = {}
if os.path.exists(LOG_PATH):
    with open(LOG_PATH, 'r') as f:
        for line in f:
            if 'BTC: $' in line and '|' in line:
                try:
                    ts_str = line[:19]
                    probs = dict(prob_pattern.findall(line))
                    probs = {k: int(v) for k, v in probs.items()}
                    if probs:
                        log_probs[ts_str] = probs
                except:
                    pass
    print(f"Parsed {len(log_probs)} probability snapshots from log")


def get_rsi_at_time(entry_time_str):
    """Get RSI from parquet at trade entry time."""
    if pq is None:
        return None
    try:
        ts = pd.Timestamp(entry_time_str)
        # Find nearest parquet bar at or before entry
        idx = pq.index.searchsorted(ts, side='right') - 1
        if idx < 0:
            return None
        row = pq.iloc[idx]
        if 'momentum_rsi' in row.index:
            return row['momentum_rsi']
    except:
        pass
    return None


def get_probs_at_time(entry_time_str):
    """Get model probabilities from log at trade entry time."""
    try:
        ts = pd.Timestamp(entry_time_str)
        # Find the closest log entry within 2 minutes before trade
        best_key = None
        best_diff = timedelta(minutes=2)
        for key in log_probs:
            key_ts = pd.Timestamp(key)
            diff = ts - key_ts
            if timedelta(0) <= diff < best_diff:
                best_diff = diff
                best_key = key
        if best_key:
            return log_probs[best_key]
    except:
        pass
    return {}


# Simulate filters on each trade
print("\n" + "=" * 80)
print("FILTER SIMULATION RESULTS")
print("=" * 80)

results = []
for _, t in trades.iterrows():
    rsi = get_rsi_at_time(t['entry_time'])
    probs = get_probs_at_time(t['entry_time'])
    
    # Determine what each filter would do
    is_win = t['pnl_dollar'] > 0
    direction = t['direction']
    model = t['model_id']
    pnl = t['pnl_dollar']
    
    # Filter 1: RSI floor for SHORT (block if RSI < 40)
    rsi_short_block = (direction == 'SHORT' and rsi is not None and rsi < 40)
    
    # Filter 2: RSI ceiling for LONG (block if RSI > 70)
    rsi_long_block = (direction == 'LONG' and rsi is not None and rsi > 70)
    
    # Filter 3: 6h LONG requires 2h or 4h > 40%
    cross_tf_block = False
    if model == '6h_0.5pct' and direction == 'LONG' and probs:
        p2h = probs.get('2h_0.5pct', None)
        p4h = probs.get('4h_0.5pct', None)
        if p2h is not None and p4h is not None:
            if p2h < 40 and p4h < 40:
                cross_tf_block = True
    
    would_block = rsi_short_block or rsi_long_block or cross_tf_block
    
    results.append({
        'entry_time': t['entry_time'],
        'model': model,
        'direction': direction,
        'pnl': pnl,
        'is_win': is_win,
        'rsi': rsi,
        'probs': probs,
        'rsi_short_block': rsi_short_block,
        'rsi_long_block': rsi_long_block,
        'cross_tf_block': cross_tf_block,
        'would_block': would_block,
    })

df = pd.DataFrame(results)

# Overall stats
print(f"\nTotal trades: {len(df)}")
print(f"Wins: {df['is_win'].sum()} ({100*df['is_win'].mean():.0f}%)")
print(f"Total PnL: ${df['pnl'].sum():.2f}")

# How many trades have RSI data?
has_rsi = df['rsi'].notna().sum()
has_probs = (df['probs'].apply(len) > 0).sum()
print(f"\nTrades with RSI data: {has_rsi}/{len(df)}")
print(f"Trades with prob data: {has_probs}/{len(df)}")

# Filter 1: RSI < 40 SHORT block
print("\n" + "-" * 60)
print("FILTER 1: Block SHORT if RSI < 40")
print("-" * 60)
f1 = df[df['rsi_short_block']]
if len(f1) > 0:
    wins_blocked = f1['is_win'].sum()
    losses_blocked = len(f1) - wins_blocked
    pnl_blocked = f1['pnl'].sum()
    print(f"Would block: {len(f1)} trades ({wins_blocked}W/{losses_blocked}L)")
    print(f"PnL blocked: ${pnl_blocked:.2f}")
    print(f"Net effect: {'POSITIVE' if pnl_blocked < 0 else 'NEGATIVE'} (removing ${abs(pnl_blocked):.2f} of {'losses' if pnl_blocked < 0 else 'wins'})")
    for _, r in f1.iterrows():
        print(f"  {r['entry_time'][:16]} | {r['model']:20s} | RSI={r['rsi']:.1f} | {'WIN' if r['is_win'] else 'LOSS':4s} | ${r['pnl']:+.2f}")
else:
    print("No trades would be blocked (may lack RSI data for older trades)")

# Filter 2: RSI > 70 LONG block
print("\n" + "-" * 60)
print("FILTER 2: Block LONG if RSI > 70")
print("-" * 60)
f2 = df[df['rsi_long_block']]
if len(f2) > 0:
    wins_blocked = f2['is_win'].sum()
    losses_blocked = len(f2) - wins_blocked
    pnl_blocked = f2['pnl'].sum()
    print(f"Would block: {len(f2)} trades ({wins_blocked}W/{losses_blocked}L)")
    print(f"PnL blocked: ${pnl_blocked:.2f}")
    print(f"Net effect: {'POSITIVE' if pnl_blocked < 0 else 'NEGATIVE'} (removing ${abs(pnl_blocked):.2f} of {'losses' if pnl_blocked < 0 else 'wins'})")
    for _, r in f2.iterrows():
        print(f"  {r['entry_time'][:16]} | {r['model']:20s} | RSI={r['rsi']:.1f} | {'WIN' if r['is_win'] else 'LOSS':4s} | ${r['pnl']:+.2f}")
else:
    print("No trades would be blocked (may lack RSI data for older trades)")

# Filter 3: 6h LONG cross-timeframe
print("\n" + "-" * 60)
print("FILTER 3: Block 6h LONG if both 2h < 40% and 4h < 40%")
print("-" * 60)
f3 = df[df['cross_tf_block']]
if len(f3) > 0:
    wins_blocked = f3['is_win'].sum()
    losses_blocked = len(f3) - wins_blocked
    pnl_blocked = f3['pnl'].sum()
    print(f"Would block: {len(f3)} trades ({wins_blocked}W/{losses_blocked}L)")
    print(f"PnL blocked: ${pnl_blocked:.2f}")
    print(f"Net effect: {'POSITIVE' if pnl_blocked < 0 else 'NEGATIVE'} (removing ${abs(pnl_blocked):.2f} of {'losses' if pnl_blocked < 0 else 'wins'})")
    for _, r in f3.iterrows():
        p = r['probs']
        print(f"  {r['entry_time'][:16]} | {r['model']:20s} | 2h={p.get('2h_0.5pct','?')}% 4h={p.get('4h_0.5pct','?')}% | {'WIN' if r['is_win'] else 'LOSS':4s} | ${r['pnl']:+.2f}")
else:
    print("No trades would be blocked (may lack prob data for older trades)")

# Combined effect
print("\n" + "-" * 60)
print("COMBINED: All 3 filters together")
print("-" * 60)
blocked = df[df['would_block']]
kept = df[~df['would_block']]
if len(blocked) > 0:
    print(f"Trades blocked: {len(blocked)}")
    print(f"  Wins blocked: {blocked['is_win'].sum()}")
    print(f"  Losses blocked: {len(blocked) - blocked['is_win'].sum()}")
    print(f"  PnL blocked: ${blocked['pnl'].sum():.2f}")
    print(f"\nTrades kept: {len(kept)}")
    print(f"  Wins: {kept['is_win'].sum()} ({100*kept['is_win'].mean():.0f}%)")
    print(f"  Losses: {len(kept) - kept['is_win'].sum()}")
    print(f"  PnL: ${kept['pnl'].sum():.2f}")
    print(f"\nOriginal WR: {100*df['is_win'].mean():.1f}% → New WR: {100*kept['is_win'].mean():.1f}%")
    print(f"Original PnL: ${df['pnl'].sum():.2f} → New PnL: ${kept['pnl'].sum():.2f}")
    print(f"Net improvement: ${kept['pnl'].sum() - df['pnl'].sum():.2f}")

# Also test RSI thresholds sensitivity
print("\n" + "=" * 80)
print("SENSITIVITY: RSI threshold sweep (SHORT floor / LONG ceiling)")
print("=" * 80)
short_trades = df[(df['direction'] == 'SHORT') & df['rsi'].notna()]
long_trades = df[(df['direction'] == 'LONG') & df['rsi'].notna()]

if len(short_trades) > 0:
    print(f"\nSHORT trades with RSI data: {len(short_trades)}")
    for thresh in [30, 35, 40, 45, 50]:
        blocked = short_trades[short_trades['rsi'] < thresh]
        if len(blocked) > 0:
            w = blocked['is_win'].sum()
            l = len(blocked) - w
            print(f"  RSI < {thresh}: block {len(blocked)}T ({w}W/{l}L, ${blocked['pnl'].sum():+.2f})")

if len(long_trades) > 0:
    print(f"\nLONG trades with RSI data: {len(long_trades)}")
    for thresh in [60, 65, 70, 75, 80]:
        blocked = long_trades[long_trades['rsi'] > thresh]
        if len(blocked) > 0:
            w = blocked['is_win'].sum()
            l = len(blocked) - w
            print(f"  RSI > {thresh}: block {len(blocked)}T ({w}W/{l}L, ${blocked['pnl'].sum():+.2f})")
