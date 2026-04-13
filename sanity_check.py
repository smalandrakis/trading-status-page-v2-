#!/usr/bin/env python3
"""Sanity check: verify label building with brute-force spot checks."""
import numpy as np, pandas as pd

df = pd.read_parquet('data/btc_1m_12mo.parquet')
df.index = df.index.tz_localize(None); df = df.sort_index()
p = df['close'].values.astype(np.float64)
n = len(p)

print(f"Data: {n:,} bars, {df.index[0]} → {df.index[-1]}")
print(f"Price: ${p[0]:,.0f} → ${p[-1]:,.0f}\n")

# ── Vectorized labels (same as assessment) ──
MAX_BARS = 240
long_tp_bar = np.full(n, 9999, dtype=np.int32)
long_sl_bar = np.full(n, 9999, dtype=np.int32)
for off in range(1, MAX_BARS + 1):
    idx = np.arange(n) + off; valid = idx < n
    fwd = np.where(valid, p[np.clip(idx, 0, n-1)], np.nan)
    hit = valid & (fwd >= p * 1.01) & (long_tp_bar == 9999); long_tp_bar[hit] = off
    hit = valid & (fwd <= p * 0.995) & (long_sl_bar == 9999); long_sl_bar[hit] = off
y_long_vec = ((long_tp_bar < long_sl_bar) & (long_tp_bar < 9999)).astype(np.int8)

# ── Brute force check on 1000 random bars ──
np.random.seed(42)
check_idx = np.random.choice(n - MAX_BARS, size=1000, replace=False)
mismatches = 0
for i in check_idx:
    entry = p[i]
    tp_price = entry * 1.01
    sl_price = entry * 0.995
    fwd = p[i+1 : i+1+MAX_BARS]
    # Find first TP and SL bar
    tp_bars = np.where(fwd >= tp_price)[0]
    sl_bars = np.where(fwd <= sl_price)[0]
    tp_bar = tp_bars[0] if len(tp_bars) > 0 else 9999
    sl_bar = sl_bars[0] if len(sl_bars) > 0 else 9999
    brute_label = 1 if (tp_bar < sl_bar and tp_bar < 9999) else 0
    if brute_label != y_long_vec[i]:
        mismatches += 1
        print(f"  MISMATCH at bar {i}: brute={brute_label} vec={y_long_vec[i]} "
              f"tp_bar={tp_bar} sl_bar={sl_bar} vec_tp={long_tp_bar[i]} vec_sl={long_sl_bar[i]}")

print(f"Brute-force check: {mismatches}/1000 mismatches")
print(f"Labels are {'CORRECT' if mismatches == 0 else 'WRONG'}\n")

# ── Verify base rates ──
total = n - MAX_BARS  # exclude tail
wins = y_long_vec[:total].sum()
print(f"LONG base rate (240 bars): {wins:,}/{total:,} = {wins/total*100:.2f}%")
print(f"Breakeven needed: 33.3%")
print(f"Gap to breakeven: {33.3 - wins/total*100:.1f}pp\n")

# ── Specific examples: show 5 random winners ──
win_idx = np.where(y_long_vec[:total] == 1)[0]
sample_wins = np.random.choice(win_idx, size=5, replace=False)
print("5 RANDOM WINNING LONG ENTRIES:")
for i in sorted(sample_wins):
    entry = p[i]
    tp_hit = long_tp_bar[i]
    sl_bar = long_sl_bar[i]
    tp_price = p[i + tp_hit]
    print(f"  Bar {i} ({df.index[i]}): entry=${entry:,.0f} → TP hit at bar+{tp_hit} "
          f"(${tp_price:,.0f}, +{(tp_price/entry-1)*100:.2f}%) | SL would hit at bar+{sl_bar}")

# ── Show 5 random losers ──
loss_idx = np.where((y_long_vec[:total] == 0) & (long_sl_bar[:total] < 9999))[0]
sample_loss = np.random.choice(loss_idx, size=5, replace=False)
print("\n5 RANDOM LOSING LONG ENTRIES (SL hit first):")
for i in sorted(sample_loss):
    entry = p[i]
    sl_hit = long_sl_bar[i]
    tp_bar = long_tp_bar[i]
    sl_price = p[i + sl_hit]
    print(f"  Bar {i} ({df.index[i]}): entry=${entry:,.0f} → SL hit at bar+{sl_hit} "
          f"(${sl_price:,.0f}, {(sl_price/entry-1)*100:.2f}%) | TP would hit at bar+{tp_bar}")

# ── Monthly breakdown ──
print("\nMONTHLY BREAKDOWN (240-bar horizon):")
print(f"  {'Month':>7} {'BTC%':>7} | {'LONG base':>9} {'#wins':>6} | {'regime':>10}")
months = pd.date_range(df.index[0].replace(day=1), df.index[-1], freq='MS')
for i in range(len(months)-1):
    ms = months[i]; me = months[i+1]
    mask = (df.index >= ms) & (df.index < me)
    idx_m = np.where(mask)[0]
    if len(idx_m) < 100: continue
    btc_ret = (p[idx_m[-1]] / p[idx_m[0]] - 1) * 100
    lt = len(idx_m); lw = y_long_vec[idx_m].sum()
    regime = "BULL" if btc_ret > 3 else "BEAR" if btc_ret < -3 else "CHOP"
    print(f"  {ms.strftime('%Y-%m'):>7} {btc_ret:>+6.1f}% | {lw/lt*100:8.1f}% {lw:>6,} | {regime:>10}")
