#!/usr/bin/env python3
"""Sanity check: does the model add value vs always-LONG baseline?"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

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

# Labels for TP=2%/SL=0.5%
tp=2.0; sl=0.5
lw=np.zeros(n,dtype=np.int8)
for i in range(n):
    ep=o[i]
    tp_hit=h[i]>=ep*(1+tp/100); sl_hit=l[i]<=ep*(1-sl/100)
    if tp_hit and not sl_hit: lw[i]=1
    elif tp_hit and sl_hit: lw[i]=1 if c[i]>ep else 0

be = sl/(tp+sl)*100
print(f"TP={tp}% SL={sl}% | R:R={tp/sl:.0f}:1 | BE={be:.1f}%")
print(f"Overall base rate: {lw.mean()*100:.1f}%\n")

# Walk-forward quarters (same as the test)
quarters = pd.date_range('2022-01-01','2026-04-01',freq='QS')

print("ALWAYS-LONG BASELINE (enter LONG every single day):")
print(f"  {'Quarter':>8} {'BTC':>7} {'Days':>5} {'Wins':>4} {'WR':>6} {'P&L':>8} {'CumPnL':>8}")
print(f"  {'-'*55}")
cum=0; total_t=0; total_w=0
for qi in range(len(quarters)-1):
    ts=quarters[qi]; te=quarters[qi+1]
    mask=(raw.index>=ts)&(raw.index<te)
    idx=np.where(mask)[0]
    if len(idx)==0: continue
    nt=len(idx); w=int(lw[idx].sum()); wr=w/nt*100
    pnl=w*tp-(nt-w)*sl
    btc_ret=(c[idx[-1]]/c[idx[0]]-1)*100
    cum+=pnl; total_t+=nt; total_w+=w
    s="+" if pnl>0 else "-"
    qn=f"{ts.year}-Q{(ts.month-1)//3+1}"
    print(f"  {qn:>8} {btc_ret:+6.1f}% {nt:5} {w:4} {wr:5.1f}% {pnl:+7.2f}% {cum:+7.2f}% {s}")

total_wr=total_w/total_t*100
print(f"\n  TOTAL: {total_t}T {total_w}W {total_wr:.1f}%WR P&L={cum:+.1f}%")
print(f"  Avg EV/trade: {cum/total_t:+.3f}%")

# Compare
print(f"\n{'='*60}")
print("COMPARISON:")
print(f"  Always-LONG:     {total_t}T {total_wr:.1f}%WR P&L={cum:+.1f}%  EV={cum/total_t:+.3f}%")
print(f"  RF model >20%:   1365T  29.8%WR P&L=+335.0%  EV=+0.245%")
print(f"  RF model >25%:   1054T  31.2%WR P&L=+295.5%  EV=+0.280%")
print(f"  RF model >30%:    748T  33.2%WR P&L=+246.0%  EV=+0.329%")
print(f"\nIf model WR ≈ base rate, model adds NO value (just always go LONG).")
print(f"If model WR > base rate on selected days, model IS adding value.")
