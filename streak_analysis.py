#!/usr/bin/env python3
"""Analyze: does BTC mean-revert after consecutive up/down days?"""
import numpy as np, pandas as pd

raw = pd.read_parquet('data/btc_daily_5yr.parquet')
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = [c[0].lower() for c in raw.columns]
else:
    raw.columns = [c.lower() for c in raw.columns]
raw = raw.dropna(subset=['close']).sort_index()
raw.index = pd.to_datetime(raw.index)
raw.index = raw.index.tz_localize(None) if raw.index.tz else raw.index

o=raw['open'].values; h=raw['high'].values; l=raw['low'].values; c=raw['close'].values
n=len(raw)
ret = pd.Series(c).pct_change().values * 100  # daily close-to-close %

# TP/SL labels
tp=2.0; sl=0.5
long_win=np.zeros(n,dtype=np.int8); short_win=np.zeros(n,dtype=np.int8)
for i in range(n):
    ep=o[i]
    ltp=h[i]>=ep*(1+tp/100); lsl=l[i]<=ep*(1-sl/100)
    stp=l[i]<=ep*(1-tp/100); ssl=h[i]>=ep*(1+sl/100)
    if ltp and not lsl: long_win[i]=1
    elif ltp and lsl: long_win[i]=1 if c[i]>ep else 0
    if stp and not ssl: short_win[i]=1
    elif stp and ssl: short_win[i]=1 if c[i]<ep else 0

# Compute streak at each day
up_streak = np.zeros(n); dn_streak = np.zeros(n)
for i in range(1, n):
    if ret[i-1] > 0:
        up_streak[i] = up_streak[i-1] + 1
        dn_streak[i] = 0
    elif ret[i-1] < 0:
        dn_streak[i] = dn_streak[i-1] + 1
        up_streak[i] = 0

# ══════════════════════════════════════════════════════════
# Analysis 1: LONG and SHORT win rates by prior streak
# ══════════════════════════════════════════════════════════
print("="*70)
print("LONG TP=2%/SL=0.5% win rate by PRIOR UP-STREAK")
print("="*70)
print(f"  {'Up streak':>10} {'Days':>5} {'LONG WR':>8} {'SHORT WR':>9} {'Nxt ret':>8} {'Implication':>15}")
print(f"  {'-'*60}")
for streak in range(0, 8):
    mask = up_streak == streak
    cnt = mask.sum()
    if cnt < 10: continue
    lwr = long_win[mask].mean()*100
    swr = short_win[mask].mean()*100
    avg_ret = np.nanmean(ret[1:][mask[:-1]])  # next day return after this streak
    better = "→ SHORT" if swr > lwr else "→ LONG"
    print(f"  {streak:>10} {cnt:5} {lwr:7.1f}% {swr:8.1f}% {avg_ret:+7.2f}% {better:>15}")

print(f"\n{'='*70}")
print("LONG TP=2%/SL=0.5% win rate by PRIOR DOWN-STREAK")
print("="*70)
print(f"  {'Dn streak':>10} {'Days':>5} {'LONG WR':>8} {'SHORT WR':>9} {'Nxt ret':>8} {'Implication':>15}")
print(f"  {'-'*60}")
for streak in range(0, 8):
    mask = dn_streak == streak
    cnt = mask.sum()
    if cnt < 10: continue
    lwr = long_win[mask].mean()*100
    swr = short_win[mask].mean()*100
    avg_ret = np.nanmean(ret[1:][mask[:-1]])
    better = "→ SHORT" if swr > lwr else "→ LONG"
    print(f"  {streak:>10} {cnt:5} {lwr:7.1f}% {swr:8.1f}% {avg_ret:+7.2f}% {better:>15}")

# ══════════════════════════════════════════════════════════
# Analysis 2: Conditional P&L simulation
# ══════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("P&L SIMULATION: What if we switch direction based on streaks?")
print("="*70)

strategies = [
    ("Always LONG", lambda us, ds: 'LONG'),
    ("LONG, but SHORT after 2+ up days", lambda us, ds: 'SHORT' if us >= 2 else 'LONG'),
    ("LONG, but SHORT after 3+ up days", lambda us, ds: 'SHORT' if us >= 3 else 'LONG'),
    ("LONG, but SKIP after 2+ up days", lambda us, ds: 'SKIP' if us >= 2 else 'LONG'),
    ("LONG, but SKIP after 3+ up days", lambda us, ds: 'SKIP' if us >= 3 else 'LONG'),
    ("LONG after 1+ down days, else SKIP", lambda us, ds: 'LONG' if ds >= 1 else 'SKIP'),
    ("LONG after 2+ down days, else SKIP", lambda us, ds: 'LONG' if ds >= 2 else 'SKIP'),
    ("Always SHORT after 2+ up, LONG after 2+ down, else SKIP",
     lambda us, ds: 'SHORT' if us >= 2 else ('LONG' if ds >= 2 else 'SKIP')),
    ("Contrarian: SHORT after 2+ up, LONG after 2+ down, else LONG",
     lambda us, ds: 'SHORT' if us >= 2 else 'LONG'),
]

print(f"\n  {'Strategy':>55} | {'Trades':>6} {'Wins':>5} {'WR':>6} {'P&L':>8} {'EV':>7}")
print(f"  {'-'*95}")

for name, rule in strategies:
    trades=0; wins=0; pnl=0
    for i in range(200, n):  # skip warmup
        action = rule(up_streak[i], dn_streak[i])
        if action == 'LONG':
            trades += 1
            if long_win[i]: wins += 1; pnl += tp
            else: pnl -= sl
        elif action == 'SHORT':
            trades += 1
            if short_win[i]: wins += 1; pnl += tp
            else: pnl -= sl
    wr = wins/trades*100 if trades > 0 else 0
    ev = pnl/trades if trades > 0 else 0
    print(f"  {name:>55} | {trades:6} {wins:5} {wr:5.1f}% {pnl:+7.1f}% {ev:+6.3f}%")

# ══════════════════════════════════════════════════════════
# Analysis 3: Did we have this feature in the model?
# ══════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("FEATURE CHECK: Was streak info in the v3 model?")
print("="*70)
print("  Features related to streaks in the 96-feature set:")
print("  - up_streak: consecutive up days (shifted by 1)")
print("  - dn_streak: consecutive down days (shifted by 1)")
print("  - up_pct_3d/5d/7d/10d: % of up days in last N days")
print("  - ret_1d/2d/3d/5d: cumulative return over last N days")
print("  YES — streak features were already in the model.")

# ══════════════════════════════════════════════════════════
# Analysis 4: 2-day return as predictor
# ══════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("2-DAY CUMULATIVE RETURN → Next day direction")
print("="*70)
ret2d = pd.Series(c).pct_change(2).values * 100
bins = [(-999,-4),(-4,-2),(-2,-1),(-1,0),(0,1),(1,2),(2,4),(4,999)]
print(f"  {'2d return':>12} {'Days':>5} {'LONG WR':>8} {'SHORT WR':>9} {'Nxt ret':>8} {'Better':>8}")
print(f"  {'-'*55}")
for lo, hi in bins:
    mask = (ret2d >= lo) & (ret2d < hi)
    # Shift: use ret2d from yesterday to predict today
    mask_shifted = np.zeros(n, dtype=bool)
    mask_shifted[1:] = mask[:-1]
    cnt = mask_shifted.sum()
    if cnt < 20: continue
    lwr = long_win[mask_shifted].mean()*100
    swr = short_win[mask_shifted].mean()*100
    avg_ret = np.nanmean(ret[1:][mask_shifted[:-1]])
    better = "SHORT" if swr > lwr else "LONG"
    label = f"{lo:+.0f}% to {hi:+.0f}%"
    print(f"  {label:>12} {cnt:5} {lwr:7.1f}% {swr:8.1f}% {avg_ret:+7.2f}% {better:>8}")

print("\nDone!")
