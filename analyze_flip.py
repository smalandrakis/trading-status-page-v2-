#!/usr/bin/env python3
"""Analyze: would flipping SHORT signals to LONG be more profitable?"""
import sqlite3
import pandas as pd
import numpy as np

conn = sqlite3.connect('trades.db')
trades = pd.read_sql('SELECT * FROM trades ORDER BY entry_time', conn)
conn.close()

print("=== OVERALL ===")
dirs = ['LONG', 'SHORT']
for d in dirs:
    t = trades[trades['direction'] == d]
    w = (t['exit_reason'] == 'TRAILING_STOP').sum()
    l = (t['exit_reason'] == 'STOP_LOSS').sum()
    n = len(t)
    wr = w / n * 100 if n > 0 else 0
    pnl = t['pnl_dollar'].sum()
    print("  %5s: %dW/%dL (%d%% WR), PnL=$%+.2f, n=%d" % (d, w, l, wr, pnl, n))

total_pnl = trades['pnl_dollar'].sum()
print("  TOTAL PnL: $%+.2f" % total_pnl)

print("\n=== LAST 20 SHORT TRADES ===")
shorts = trades[trades['direction'] == 'SHORT'].tail(20)
for _, r in shorts.iterrows():
    tag = "WIN " if r['exit_reason'] == 'TRAILING_STOP' else "LOSS"
    et = str(r['entry_time'])[:16]
    print("  %s $%.0f->$%.0f %s $%+.2f" % (et, r['entry_price'], r['exit_price'], tag, r['pnl_dollar']))

print("\n=== FLIP ANALYSIS ===")
short_sl = trades[(trades['direction'] == 'SHORT') & (trades['exit_reason'] == 'STOP_LOSS')]
short_ts = trades[(trades['direction'] == 'SHORT') & (trades['exit_reason'] == 'TRAILING_STOP')]
n_sl = len(short_sl)
n_ts = len(short_ts)
n_total = n_sl + n_ts
print("  SHORT SL (price UP = LONG win):   %d" % n_sl)
print("  SHORT TS (price DOWN = LONG loss): %d" % n_ts)
flip_wr = n_sl / n_total * 100 if n_total > 0 else 0
print("  Flipped WR: %d%%" % flip_wr)

# Estimate flipped PnL
sl_loss_total = short_sl['pnl_dollar'].sum()  # negative
ts_win_total = short_ts['pnl_dollar'].sum()   # positive
# If flipped: SL losses become wins (roughly same magnitude), TS wins become losses
print("  Current SHORT PnL: $%+.2f" % (sl_loss_total + ts_win_total))
print("  If flipped (rough): $%+.2f" % (-(sl_loss_total + ts_win_total)))
