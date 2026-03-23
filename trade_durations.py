#!/usr/bin/env python3
"""Check how long trades take to resolve on 2-sec tick data."""
import sqlite3, pandas as pd, numpy as np

conn = sqlite3.connect('trades.db')
trades = pd.read_sql("""
    SELECT entry_time, exit_time, direction, exit_reason, pnl_dollar 
    FROM trades 
    WHERE exit_reason IN ('TRAILING_STOP','STOP_LOSS') 
    ORDER BY entry_time DESC LIMIT 200
""", conn)
conn.close()

trades['entry_time'] = pd.to_datetime(trades['entry_time'], format='ISO8601')
trades['exit_time'] = pd.to_datetime(trades['exit_time'], format='ISO8601')
trades['dur_sec'] = (trades['exit_time'] - trades['entry_time']).dt.total_seconds()
trades['dur_min'] = trades['dur_sec'] / 60

print("ACTUAL TRADE DURATIONS (last 200 trades from DB)")
print(f"Total: {len(trades)} trades\n")

d = trades['dur_min']
print(f"Overall: median {d.median():.1f} min, mean {d.mean():.1f} min\n")

for t, lbl in [(1,'<1 min'),(5,'<5 min'),(15,'<15 min'),(30,'<30 min'),(60,'<1 hr')]:
    n = (d < t).sum()
    print(f"  {lbl:10s}: {n:3d} ({n/len(d)*100:.0f}%)")

print(f"\nBy exit reason:")
for reason in ['STOP_LOSS','TRAILING_STOP']:
    sub = trades[trades['exit_reason']==reason]['dur_min']
    print(f"  {reason:15s}: median {sub.median():.1f} min, mean {sub.mean():.1f} min, n={len(sub)}")

print(f"\nBy direction:")
for dn in ['LONG','SHORT']:
    sub = trades[trades['direction']==dn]['dur_min']
    if len(sub)>0:
        print(f"  {dn:5s}: median {sub.median():.1f} min, mean {sub.mean():.1f} min")

print(f"\nPercentiles (all):")
for p in [10,25,50,75,90,95]:
    print(f"  {p:3d}th: {np.percentile(d,p):.1f} min")

print(f"\nPercentiles — STOP_LOSS only:")
sl = trades[trades['exit_reason']=='STOP_LOSS']['dur_min']
for p in [10,25,50,75,90,95]:
    print(f"  {p:3d}th: {np.percentile(sl,p):.1f} min")

print(f"\nPercentiles — TRAILING_STOP only:")
ts = trades[trades['exit_reason']=='TRAILING_STOP']['dur_min']
for p in [10,25,50,75,90,95]:
    print(f"  {p:3d}th: {np.percentile(ts,p):.1f} min")

# KEY QUESTION: how many resolve within a single 5-min bar?
under5 = (d < 5).sum()
print(f"\n*** TRADES RESOLVING IN <5 MIN: {under5}/{len(d)} ({under5/len(d)*100:.0f}%) ***")
print("These trades would be INVISIBLE to 5-min bar backtesting!")
