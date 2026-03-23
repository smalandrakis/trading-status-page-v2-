"""Quick BTC strategy analysis — fully vectorised, no Python loops."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

ticks = pd.read_csv('logs/btc_price_ticks.csv', parse_dates=['timestamp'])
ticks = ticks.sort_values('timestamp').set_index('timestamp')
ticks.index = pd.to_datetime(ticks.index, utc=True).tz_localize(None)
bars = ticks['price'].resample('5min').agg(Open='first',High='max',Low='min',Close='last').dropna()
bars = bars[bars.index >= '2026-03-01']

close = bars['Close'].values.astype(float)
high  = bars['High'].values.astype(float)
low   = bars['Low'].values.astype(float)
N = len(close)
print(f"Bars: {N}  ({bars.index[0]:%Y-%m-%d} → {bars.index[-1]:%Y-%m-%d})\n")

def sim_vectorised(sl_pct, tp_pct, H):
    """Simulate SL/TP on every bar as entry, vectorised."""
    n = N - H
    ep = close[:n]                    # entry prices
    tp_p = ep * (1 + tp_pct / 100)
    sl_p = ep * (1 - sl_pct / 100)

    # Build future window arrays: shape (n, H)
    # future_highs[i, j] = high[i+1+j]
    idx = np.arange(n)[:, None] + np.arange(1, H+1)[None, :]
    fh = high[idx]   # (n, H)
    fl = low[idx]    # (n, H)

    # First bar that hits TP / SL
    tp_hit_bar = np.where((fh >= tp_p[:, None]).any(axis=1),
                          np.argmax(fh >= tp_p[:, None], axis=1), H)
    sl_hit_bar = np.where((fl <= sl_p[:, None]).any(axis=1),
                          np.argmax(fl <= sl_p[:, None], axis=1), H)

    tp_hit = tp_hit_bar < H
    sl_hit = sl_hit_bar < H
    win  = tp_hit & (~sl_hit | (tp_hit_bar <= sl_hit_bar))
    loss = sl_hit & (~tp_hit | (sl_hit_bar < tp_hit_bar))

    nw, nl = win.sum(), loss.sum()
    total = nw + nl
    wr = nw / total * 100 if total else 0
    ev = wr/100*tp_pct - (1-wr/100)*sl_pct
    return wr, ev, nw, nl, total

# ── 1. RAW MOVE FREQUENCY ────────────────────────────────────────────────────
print("=== How often does BTC move ≥X% in the next N bars? (upside) ===")
for H, lbl in [(6,'30m'),(12,'1h'),(24,'2h'),(48,'4h')]:
    n = N - H
    fh = high[np.arange(n)[:,None]+np.arange(1,H+1)[None,:]].max(axis=1)
    for thr in [0.3, 0.5, 1.0, 1.5]:
        pct = (fh >= close[:n]*(1+thr/100)).mean()*100
        print(f"  >{thr:.1f}% within {lbl}: {pct:.1f}% of bars")
    print()

# ── 2. COIN FLIP (pure random entry, LONG only) ──────────────────────────────
print("=== Coin flip (random LONG entry) ===")
print(f"  {'SL%':>5} {'TP%':>5} {'Horizon':>8} {'WR%':>6} {'EV%':>7} {'Trades':>7}")
for sl, tp in [(0.30,0.50),(0.50,1.00),(0.50,1.50)]:
    for H, lbl in [(6,'30m'),(12,'1h'),(24,'2h'),(48,'4h')]:
        wr,ev,nw,nl,tot = sim_vectorised(sl, tp, H)
        print(f"  {sl:>5.2f} {tp:>5.2f} {lbl:>8} {wr:>6.1f} {ev:>+7.3f} {tot:>7}")
    print()

# ── 3. TREND FOLLOW (enter after >0.15% move, follow direction) ──────────────
print("=== Trend follow (enter LONG after 1-bar >+0.15% move) ===")
# Entry condition: current bar return > 0.15%
ret_1bar = (close[1:] / close[:-1] - 1) * 100
trend_long  = ret_1bar > 0.15   # enter LONG after up bar
trend_short = ret_1bar < -0.15  # enter SHORT after down bar
for sl, tp in [(0.30,0.90),(0.50,1.00)]:
    for H, lbl in [(6,'30m'),(12,'1h'),(24,'2h')]:
        n = len(trend_long) - H
        entries = np.where(trend_long[:n])[0]
        if len(entries) == 0: continue
        ep = close[entries+1]
        tp_p = ep*(1+tp/100); sl_p = ep*(1-sl/100)
        fh = high[entries[:,None]+np.arange(2,H+2)[None,:]]
        fl = low[entries[:,None]+np.arange(2,H+2)[None,:]]
        tp_bar = np.where((fh>=tp_p[:,None]).any(1), np.argmax(fh>=tp_p[:,None],axis=1), H)
        sl_bar = np.where((fl<=sl_p[:,None]).any(1), np.argmax(fl<=sl_p[:,None],axis=1), H)
        win  = (tp_bar<H)&(~(sl_bar<H)|(tp_bar<=sl_bar))
        loss = (sl_bar<H)&(~(tp_bar<H)|(sl_bar<tp_bar))
        nw,nl = win.sum(),loss.sum(); tot=nw+nl
        wr = nw/tot*100 if tot else 0
        ev = wr/100*tp - (1-wr/100)*sl
        print(f"  SL={sl} TP={tp} {lbl:>4} | WR={wr:.1f}% EV={ev:+.3f}% n={tot}")
    print()

print("Done.")
