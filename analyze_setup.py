#!/usr/bin/env python3
"""Analyze current SL/TS setup and recommend optimizations."""
import sqlite3
import pandas as pd
import numpy as np

conn = sqlite3.connect('trades.db')
trades = pd.read_sql('SELECT * FROM trades ORDER BY entry_time', conn)
conn.close()

trades['entry_time'] = pd.to_datetime(trades['entry_time'], format='ISO8601')

print("=" * 70)
print("CURRENT SETUP ANALYSIS")
print("=" * 70)

print(f"\nTotal trades: {len(trades)}")
print(f"Period: {trades['entry_time'].min()} to {trades['entry_time'].max()}")
print(f"Total PnL: ${trades['pnl_dollar'].sum():+.2f}")

# Win/Loss stats
w = trades[trades['pnl_dollar'] > 0]
l = trades[trades['pnl_dollar'] <= 0]
print(f"\nWins:   {len(w)} ({len(w)/len(trades)*100:.0f}%), avg=${w['pnl_dollar'].mean():+.2f}, median=${w['pnl_dollar'].median():+.2f}")
print(f"Losses: {len(l)} ({len(l)/len(trades)*100:.0f}%), avg=${l['pnl_dollar'].mean():+.2f}, median=${l['pnl_dollar'].median():+.2f}")
print(f"Avg win / avg loss ratio: {abs(w['pnl_dollar'].mean() / l['pnl_dollar'].mean()):.2f}")

# By exit reason
print("\nBy exit reason:")
for reason in trades['exit_reason'].unique():
    sub = trades[trades['exit_reason'] == reason]
    print(f"  {reason:20s}: {len(sub):3d} trades, avg ${sub['pnl_dollar'].mean():+.2f}, total ${sub['pnl_dollar'].sum():+.2f}")

# By direction
print("\nBy direction:")
for d in ['LONG', 'SHORT']:
    sub = trades[trades['direction'] == d]
    sw = sub[sub['pnl_dollar'] > 0]
    sl = sub[sub['pnl_dollar'] <= 0]
    if len(sub) > 0 and len(sw) > 0 and len(sl) > 0:
        print(f"  {d}: {len(sub)} trades, {len(sw)}W/{len(sl)}L ({len(sw)/len(sub)*100:.0f}%), total ${sub['pnl_dollar'].sum():+.2f}")
        print(f"    Avg win: ${sw['pnl_dollar'].mean():+.2f}, Avg loss: ${sl['pnl_dollar'].mean():+.2f}, R:R={abs(sw['pnl_dollar'].mean()/sl['pnl_dollar'].mean()):.2f}")

# PnL% distribution
print("\nPnL% distribution by exit:")
for reason in ['TRAILING_STOP', 'STOP_LOSS']:
    sub = trades[trades['exit_reason'] == reason]
    if len(sub) > 0:
        print(f"  {reason}: mean={sub['pnl_pct'].mean():+.3f}%, median={sub['pnl_pct'].median():+.3f}%, "
              f"min={sub['pnl_pct'].min():+.3f}%, max={sub['pnl_pct'].max():+.3f}%")

# Recent trends
print("\nRecent trends:")
for cutoff, label in [('2026-03-07', 'Last 3 days'), ('2026-03-03', 'Last week'), ('2026-02-27', 'Last 2 weeks')]:
    recent = trades[trades['entry_time'] >= cutoff]
    if len(recent) > 0:
        rw = (recent['pnl_dollar'] > 0).sum()
        rl = (recent['pnl_dollar'] <= 0).sum()
        print(f"  {label:15s}: {len(recent):3d} trades, {rw}W/{rl}L ({rw/len(recent)*100:.0f}%), ${recent['pnl_dollar'].sum():+.2f}")

# ===============================================================
# KEY ANALYSIS: Avg win size vs avg loss size at different params
# ===============================================================
print("\n" + "=" * 70)
print("THE CORE PROBLEM")
print("=" * 70)

ts_trades = trades[trades['exit_reason'] == 'TRAILING_STOP']
sl_trades = trades[trades['exit_reason'] == 'STOP_LOSS']

if len(ts_trades) > 0 and len(sl_trades) > 0:
    avg_win_pct = ts_trades['pnl_pct'].mean()
    avg_loss_pct = sl_trades['pnl_pct'].mean()
    
    print(f"\n  Avg TRAILING_STOP exit: {avg_win_pct:+.3f}%")
    print(f"  Avg STOP_LOSS exit:    {avg_loss_pct:+.3f}%")
    print(f"  Ratio (win/loss):      {abs(avg_win_pct/avg_loss_pct):.2f}x")
    
    # Break-even WR calculation
    # At 50% WR: expected = 0.5 * avg_win + 0.5 * avg_loss
    expected_50 = 0.5 * avg_win_pct + 0.5 * avg_loss_pct
    print(f"\n  At 50% WR, expected per trade: {expected_50:+.4f}%")
    
    # What WR do we need to break even?
    # WR * avg_win + (1-WR) * avg_loss = 0
    # WR = -avg_loss / (avg_win - avg_loss)
    be_wr = -avg_loss_pct / (avg_win_pct - avg_loss_pct)
    print(f"  Break-even WR needed: {be_wr*100:.1f}%")
    print(f"  Current WR: {len(w)/len(trades)*100:.1f}%")

# ===============================================================
# Simulate different SL/TS configs on the 16-sec data
# ===============================================================
print("\n" + "=" * 70)
print("SL/TS PARAMETER OPTIMIZATION (on 2-month 16-sec data)")
print("=" * 70)

# Load 16-sec data
df16 = pd.read_csv('logs/btc_16sec_from_log.csv', parse_dates=['timestamp'])
df16 = df16.set_index('timestamp').sort_index()
df16 = df16[~df16.index.duplicated(keep='first')]
prices = df16['price'].values
n = len(prices)
print(f"\n  16-sec prices: {n} bars")

configs = [
    # (SL%, TS_activation%, TS_trail%, label)
    (0.30, 0.30, 0.10, 'CURRENT (0.30/0.30/0.10)'),
    (0.20, 0.25, 0.05, 'NEW LIVE (0.20/0.25/0.05)'),
    (0.15, 0.20, 0.05, 'Tight (0.15/0.20/0.05)'),
    (0.20, 0.20, 0.05, 'TightTS (0.20/0.20/0.05)'),
    (0.20, 0.30, 0.05, 'WideAct (0.20/0.30/0.05)'),
    (0.20, 0.25, 0.08, 'WideTrail (0.20/0.25/0.08)'),
    (0.20, 0.25, 0.10, 'WiderTrail (0.20/0.25/0.10)'),
    (0.15, 0.25, 0.08, 'TightSL+WTrail (0.15/0.25/0.08)'),
    (0.25, 0.25, 0.05, 'MedSL (0.25/0.25/0.05)'),
    (0.20, 0.35, 0.08, 'LetRun (0.20/0.35/0.08)'),
    (0.20, 0.40, 0.10, 'BigRun (0.20/0.40/0.10)'),
    (0.15, 0.30, 0.10, 'TightSL+BigRun (0.15/0.30/0.10)'),
]

MAX_BARS = 675  # ~3 hours at 16-sec

print(f"\n  {'Config':>32s} {'LongWR':>7s} {'ShortWR':>8s} {'L_AvgW':>8s} {'L_AvgL':>8s} {'L_RR':>6s} "
      f"{'S_AvgW':>8s} {'S_AvgL':>8s} {'S_RR':>6s} {'L_EV':>8s} {'S_EV':>8s}")

for sl_pct, ts_act, ts_trail, label in configs:
    lw_count = 0; ll_count = 0; lw_pnl = []; ll_pnl = []
    sw_count = 0; sl_count = 0; sw_pnl = []; sl_pnl_list = []
    
    # Sample every 100th bar to keep it fast (~1000 simulations)
    sample_step = max(1, n // 2000)
    
    for i in range(0, n - MAX_BARS, sample_step):
        entry = prices[i]
        
        # LONG
        stop = entry * (1 - sl_pct / 100)
        act = entry * (1 + ts_act / 100)
        pk = entry; ts_on = False; exited = False
        for j in range(i+1, min(i+MAX_BARS+1, n)):
            p = prices[j]
            if p <= stop:
                ll_count += 1
                ll_pnl.append((stop/entry - 1) * 100)
                exited = True; break
            if p > pk: pk = p
            if not ts_on and p >= act: ts_on = True
            if ts_on:
                tr = pk * (1 - ts_trail / 100)
                if p <= tr:
                    lw_count += 1
                    lw_pnl.append((tr/entry - 1) * 100)
                    exited = True; break
        
        # SHORT
        stop = entry * (1 + sl_pct / 100)
        act = entry * (1 - ts_act / 100)
        pk = entry; ts_on = False; exited = False
        for j in range(i+1, min(i+MAX_BARS+1, n)):
            p = prices[j]
            if p >= stop:
                sl_count += 1
                sl_pnl_list.append((entry/stop - 1) * 100)
                exited = True; break
            if p < pk: pk = p
            if not ts_on and p <= act: ts_on = True
            if ts_on:
                tr = pk * (1 + ts_trail / 100)
                if p >= tr:
                    sw_count += 1
                    sw_pnl.append((entry/tr - 1) * 100)
                    exited = True; break
    
    lt = lw_count + ll_count
    st = sw_count + sl_count
    
    if lt > 0 and st > 0 and lw_count > 0 and ll_count > 0 and sw_count > 0 and sl_count > 0:
        l_wr = lw_count / lt * 100
        s_wr = sw_count / st * 100
        l_avg_w = np.mean(lw_pnl)
        l_avg_l = np.mean(ll_pnl)
        s_avg_w = np.mean(sw_pnl)
        s_avg_l = np.mean(sl_pnl_list)
        l_rr = abs(l_avg_w / l_avg_l)
        s_rr = abs(s_avg_w / s_avg_l)
        
        # Expected value per trade
        l_ev = (l_wr/100) * l_avg_w + (1 - l_wr/100) * l_avg_l
        s_ev = (s_wr/100) * s_avg_w + (1 - s_wr/100) * s_avg_l
        
        print(f"  {label:>32s} {l_wr:>5.1f}% {s_wr:>6.1f}% "
              f"{l_avg_w:>+7.3f}% {l_avg_l:>+7.3f}% {l_rr:>5.2f} "
              f"{s_avg_w:>+7.3f}% {s_avg_l:>+7.3f}% {s_rr:>5.2f} "
              f"{l_ev:>+7.4f}% {s_ev:>+7.4f}%")

print("\n  EV = expected value per trade (positive = profitable)")
print("  RR = reward/risk ratio (avg win / avg loss)")
print("  Higher EV + higher RR = better config")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
