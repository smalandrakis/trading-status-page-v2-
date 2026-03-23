#!/usr/bin/env python3
"""
Proper Data Science Assessment: Can we predict 1%+ BTC moves from 1-min features?

Approach:
1. Find ALL instances where price moved +1% (LONG TP) or -1% (SHORT TP)
   before hitting the -0.5% SL / +0.5% SL, within various horizons
2. Look BACKWARDS at those winning entry points — what did the features look like?
3. Compare feature distributions: winning entries vs losing entries vs all bars
4. Measure individual feature predictive power (AUC, mutual information)
5. Measure combined predictive power (model AUC, calibration)
6. Honest assessment: is there enough signal to trade?
"""

import numpy as np
import pandas as pd
import time
import os
from sklearn.metrics import roc_auc_score, mutual_info_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from scipy import stats

DATA_PATH = 'data/btc_1m_12mo.parquet'

# ══════════════════════════════════════════════════════════════════════
# STEP 1: Load data & compute features
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("LOADING DATA & FEATURES")
print("=" * 70)
df = pd.read_parquet(DATA_PATH)
df.index = df.index.tz_localize(None) if df.index.tz else df.index
df = df.sort_index()
print(f"  {len(df):,} 1-min candles | {df.index[0].date()} → {df.index[-1].date()}")
print(f"  Price: ${df['close'].iloc[0]:,.0f} → ${df['close'].iloc[-1]:,.0f} "
      f"({(df['close'].iloc[-1]/df['close'].iloc[0]-1)*100:+.1f}%)")

c = df['close'].astype(float)
h = df['high'].astype(float)
l = df['low'].astype(float)
o = df['open'].astype(float)
v = df['volume'].astype(float)
r1 = c.pct_change()

feat = pd.DataFrame(index=df.index)

# Returns
for w in [1, 2, 3, 5, 10, 15, 30, 60, 120, 240]:
    feat[f'ret_{w}'] = c.pct_change(w) * 100
# Volatility
for w in [5, 10, 15, 30, 60, 120, 240]:
    feat[f'vol_{w}'] = r1.rolling(w).std() * 100
# Vol ratios
feat['vol_ratio_5_60'] = feat['vol_5'] / feat['vol_60'].replace(0, np.nan)
feat['vol_ratio_15_60'] = feat['vol_15'] / feat['vol_60'].replace(0, np.nan)
feat['vol_ratio_30_120'] = feat['vol_30'] / feat['vol_120'].replace(0, np.nan)
feat['vol_ratio_60_240'] = feat['vol_60'] / feat['vol_240'].replace(0, np.nan)
# Volume ratios
for w in [5, 15, 30, 60, 120]:
    feat[f'vratio_{w}'] = v / v.rolling(w).mean().replace(0, np.nan)
# RSI
for p in [7, 14, 30, 60]:
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(span=p).mean()
    loss = (-delta.clip(upper=0)).ewm(span=p).mean()
    feat[f'rsi_{p}'] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
# Bollinger
for w in [20, 60, 120]:
    ma = c.rolling(w).mean(); std = c.rolling(w).std()
    feat[f'bb_pct_{w}'] = (c - (ma - 2*std)) / (4*std).replace(0, np.nan)
# ATR
for w in [14, 30, 60, 120]:
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    feat[f'atr_pct_{w}'] = tr.rolling(w).mean() / c * 100
# MACD
for fast, slow in [(12, 26), (5, 15), (30, 60)]:
    ema_f = c.ewm(span=fast).mean(); ema_s = c.ewm(span=slow).mean()
    diff = ema_f - ema_s; sig = diff.ewm(span=9).mean()
    feat[f'macd_{fast}_{slow}'] = diff / c * 100
    feat[f'macd_hist_{fast}_{slow}'] = (diff - sig) / c * 100
# Position in range
for w in [60, 120, 240, 480]:
    rh = h.rolling(w).max(); rl = l.rolling(w).min()
    feat[f'pos_high_{w}'] = (c - rh) / c * 100
    feat[f'pos_low_{w}'] = (c - rl) / c * 100
    feat[f'pos_range_{w}'] = (c - rl) / (rh - rl).replace(0, np.nan)
# Bar structure
feat['bar_range'] = (h - l) / c * 100
feat['bar_body'] = (c - o).abs() / c * 100
feat['upper_wick'] = (h - c.clip(lower=o)) / c * 100
feat['lower_wick'] = (c.clip(upper=o) - l) / c * 100
feat['bar_range_ma30'] = feat['bar_range'].rolling(30).mean()
feat['bar_range_ratio'] = feat['bar_range'] / feat['bar_range_ma30'].replace(0, np.nan)
# Distance from MA
for w in [30, 60, 120]:
    ma = c.rolling(w).mean()
    feat[f'dist_ma_{w}'] = (c - ma) / ma * 100
# Correlation
feat['vol_price_corr_30'] = r1.rolling(30).corr(v.pct_change())
feat['vol_price_corr_60'] = r1.rolling(60).corr(v.pct_change())
# ROC
feat['roc_5'] = c.pct_change(5) * 100
feat['roc_15'] = c.pct_change(15) * 100
feat['roc_60'] = c.pct_change(60) * 100
# Consecutive bars
feat['up_count_10'] = sum((r1.shift(i) > 0).astype(int) for i in range(10))
feat['down_count_10'] = 10 - feat['up_count_10']
# Time
hour = df.index.hour + df.index.minute / 60
feat['hour_sin'] = np.sin(2 * np.pi * hour / 24)
feat['hour_cos'] = np.cos(2 * np.pi * hour / 24)
dow = df.index.dayofweek
feat['dow_sin'] = np.sin(2 * np.pi * dow / 7)
feat['dow_cos'] = np.cos(2 * np.pi * dow / 7)
# Regime features
for w in [360, 720, 1440, 4320, 10080]:
    feat[f'trend_slope_{w}'] = c.pct_change(w) * 100
for w in [1440, 4320, 10080]:
    rh = c.rolling(w).max()
    feat[f'drawdown_{w}'] = (c - rh) / rh * 100

feat = feat.replace([np.inf, -np.inf], np.nan)
feature_cols = list(feat.columns)
print(f"  {len(feature_cols)} features computed")

# ══════════════════════════════════════════════════════════════════════
# STEP 2: Find ALL winning trades (1% TP hit before 0.5% SL)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: FINDING ALL 1% MOVES (TP=1.0% / SL=0.5%)")
print("=" * 70)

p_arr = c.values.astype(np.float64)
n = len(p_arr)

TP_PCT = 1.0
SL_PCT = 0.5

for horizon_name, MAX_BARS in [("2h (120 bars)", 120), ("4h (240 bars)", 240), ("8h (480 bars)", 480)]:
    # Vectorized label building
    long_tp = np.full(n, 9999, dtype=np.int32)
    long_sl = np.full(n, 9999, dtype=np.int32)
    short_tp = np.full(n, 9999, dtype=np.int32)
    short_sl = np.full(n, 9999, dtype=np.int32)

    for off in range(1, MAX_BARS + 1):
        idx = np.arange(n) + off
        valid = idx < n
        fwd = np.where(valid, p_arr[np.clip(idx, 0, n-1)], np.nan)
        hit = valid & (fwd >= p_arr * 1.01) & (long_tp == 9999); long_tp[hit] = off
        hit = valid & (fwd <= p_arr * 0.995) & (long_sl == 9999); long_sl[hit] = off
        hit = valid & (fwd <= p_arr * 0.99) & (short_tp == 9999); short_tp[hit] = off
        hit = valid & (fwd >= p_arr * 1.005) & (short_sl == 9999); short_sl[hit] = off

    y_long = ((long_tp < long_sl) & (long_tp < 9999)).astype(np.int8)
    y_short = ((short_tp < short_sl) & (short_tp < 9999)).astype(np.int8)
    y_long_loss = ((long_sl < long_tp) & (long_sl < 9999)).astype(np.int8)
    y_short_loss = ((short_sl < short_tp) & (short_sl < 9999)).astype(np.int8)
    y_long_neither = 1 - y_long - y_long_loss

    long_wins = y_long.sum()
    long_losses = y_long_loss.sum()
    long_neither = y_long_neither.sum()
    short_wins = y_short.sum()
    short_losses = ((short_sl < short_tp) & (short_sl < 9999)).sum()
    short_neither = n - short_wins - short_losses

    print(f"\n  Horizon: {horizon_name}")
    print(f"  ┌─────────────────────────────────────────────┐")
    print(f"  │ LONG trades (buy, TP=+1.0%, SL=-0.5%)      │")
    print(f"  │   Winners (TP hit first):  {long_wins:>7,} ({long_wins/n*100:5.1f}%) │")
    print(f"  │   Losers  (SL hit first):  {long_losses:>7,} ({long_losses/n*100:5.1f}%) │")
    print(f"  │   Neither (timeout):       {long_neither:>7,} ({long_neither/n*100:5.1f}%) │")
    print(f"  │   Win rate if traded ALL:  {long_wins/(long_wins+long_losses)*100:5.1f}%           │")
    print(f"  ├─────────────────────────────────────────────┤")
    print(f"  │ SHORT trades (sell, TP=-1.0%, SL=+0.5%)    │")
    print(f"  │   Winners (TP hit first):  {short_wins:>7,} ({short_wins/n*100:5.1f}%) │")
    print(f"  │   Losers  (SL hit first):  {short_losses:>7,} ({short_losses/n*100:5.1f}%) │")
    print(f"  │   Neither (timeout):       {short_neither:>7,} ({short_neither/n*100:5.1f}%) │")
    print(f"  │   Win rate if traded ALL:  {short_wins/(short_wins+short_losses)*100:5.1f}%           │")
    print(f"  └─────────────────────────────────────────────┘")
    print(f"  Breakeven WR needed: {SL_PCT/(TP_PCT+SL_PCT)*100:.1f}%")

# Use 240-bar horizon for deep analysis
print("\n\n" + "=" * 70)
print("Using 4h (240-bar) horizon for detailed analysis")
print("=" * 70)

MAX_BARS = 240
long_tp = np.full(n, 9999, dtype=np.int32); long_sl = np.full(n, 9999, dtype=np.int32)
short_tp = np.full(n, 9999, dtype=np.int32); short_sl = np.full(n, 9999, dtype=np.int32)
for off in range(1, MAX_BARS + 1):
    idx = np.arange(n) + off; valid = idx < n
    fwd = np.where(valid, p_arr[np.clip(idx, 0, n-1)], np.nan)
    hit = valid & (fwd >= p_arr * 1.01) & (long_tp == 9999); long_tp[hit] = off
    hit = valid & (fwd <= p_arr * 0.995) & (long_sl == 9999); long_sl[hit] = off
    hit = valid & (fwd <= p_arr * 0.99) & (short_tp == 9999); short_tp[hit] = off
    hit = valid & (fwd >= p_arr * 1.005) & (short_sl == 9999); short_sl[hit] = off
y_long = ((long_tp < long_sl) & (long_tp < 9999)).astype(np.int8)
y_short = ((short_tp < short_sl) & (short_tp < 9999)).astype(np.int8)

# ══════════════════════════════════════════════════════════════════════
# STEP 3: Feature analysis — what separates winners from losers?
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: FEATURE ANALYSIS — What predicts winning entries?")
print("=" * 70)

# Clean data
valid_mask = feat.notna().all(axis=1).values
X_all = feat[feature_cols].fillna(0).values

for direction, y_arr, label in [("LONG", y_long, "LONG (TP=+1%, SL=-0.5%)"),
                                  ("SHORT", y_short, "SHORT (TP=-1%, SL=+0.5%)")]:
    print(f"\n  {'─'*60}")
    print(f"  {label}")
    print(f"  {'─'*60}")

    mask = valid_mask & (np.arange(n) < n - MAX_BARS)  # exclude tail
    X = X_all[mask]
    y = y_arr[mask]
    wins = y.sum()
    total = len(y)

    print(f"  Analyzable bars: {total:,}")
    print(f"  Winners: {wins:,} ({wins/total*100:.1f}%)")
    print(f"  Losers/timeout: {total-wins:,} ({(total-wins)/total*100:.1f}%)")

    # Individual feature AUC
    print(f"\n  INDIVIDUAL FEATURE PREDICTIVE POWER (AUC):")
    print(f"  {'Feature':<25} {'AUC':>6} {'Direction':>10} {'Win mean':>10} {'Loss mean':>10} {'Diff':>8}")
    print(f"  {'-'*75}")

    aucs = []
    for i, col in enumerate(feature_cols):
        x_col = X[:, i]
        # Skip if constant
        if x_col.std() == 0:
            continue
        try:
            auc = roc_auc_score(y, x_col)
        except:
            continue
        # Flip if AUC < 0.5 (feature predicts inversely)
        auc_adj = max(auc, 1 - auc)
        direction_str = "↑ wins" if auc > 0.5 else "↓ wins"
        win_mean = x_col[y == 1].mean()
        loss_mean = x_col[y == 0].mean()
        aucs.append((col, auc_adj, direction_str, win_mean, loss_mean))

    aucs.sort(key=lambda x: -x[1])
    for col, auc, dir_str, wm, lm in aucs[:20]:
        diff = wm - lm
        print(f"  {col:<25} {auc:.4f} {dir_str:>10} {wm:>10.4f} {lm:>10.4f} {diff:>+8.4f}")

    best_auc = aucs[0][1] if aucs else 0.5
    print(f"\n  Best single-feature AUC: {best_auc:.4f}")
    print(f"  (0.50 = random, 0.55 = weak, 0.60 = moderate, 0.65+ = strong)")

    # ── Combined model predictive power ──
    print(f"\n  COMBINED MODEL PREDICTIVE POWER:")

    # Time-series CV: 5 folds, expanding window
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)

    model = HistGradientBoostingClassifier(
        max_iter=200, max_depth=4, min_samples_leaf=200,
        learning_rate=0.03, random_state=42, l2_regularization=1.0)

    fold_aucs = []
    fold_details = []
    for fold_i, (tr_idx, te_idx) in enumerate(tscv.split(X)):
        model.fit(X[tr_idx], y[tr_idx])
        probs = model.predict_proba(X[te_idx])[:, 1]
        try:
            fold_auc = roc_auc_score(y[te_idx], probs)
        except:
            continue
        fold_aucs.append(fold_auc)

        # Measure at various thresholds
        base = y[te_idx].mean()
        for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
            sel = probs > thr
            if sel.sum() < 10:
                continue
            wr = y[te_idx][sel].mean()
            lift = wr / base if base > 0 else 0
            fold_details.append({
                'fold': fold_i, 'threshold': thr,
                'selected': sel.sum(), 'wr': wr * 100,
                'base': base * 100, 'lift': lift
            })

    print(f"  5-fold time-series CV AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    print(f"  Fold AUCs: {[f'{a:.4f}' for a in fold_aucs]}")

    # Aggregate threshold analysis across folds
    if fold_details:
        det_df = pd.DataFrame(fold_details)
        print(f"\n  THRESHOLD ANALYSIS (averaged across folds):")
        print(f"  {'Threshold':>9} {'Selected':>8} {'WR%':>7} {'Base%':>7} {'Lift':>6}")
        print(f"  {'-'*42}")
        for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
            sub = det_df[det_df['threshold'] == thr]
            if len(sub) == 0:
                continue
            print(f"  >{thr:7.0%} {sub['selected'].mean():8.0f} {sub['wr'].mean():6.1f}% "
                  f"{sub['base'].mean():6.1f}% {sub['lift'].mean():5.2f}x")

    # ── Statistical significance ──
    print(f"\n  STATISTICAL SIGNIFICANCE:")
    # Chi-squared test: is the model's selection significantly different from random?
    model.fit(X[:len(X)//2], y[:len(X)//2])  # train on first half
    probs_test = model.predict_proba(X[len(X)//2:])[:, 1]
    y_test = y[len(X)//2:]
    base_rate = y_test.mean()

    for thr in [0.5, 0.6, 0.7]:
        sel = probs_test > thr
        if sel.sum() < 10:
            continue
        n_sel = sel.sum()
        n_wins_sel = y_test[sel].sum()
        expected_wins = n_sel * base_rate
        # Binomial test
        from scipy.stats import binomtest
        p_val = binomtest(int(n_wins_sel), int(n_sel), base_rate, alternative='greater').pvalue
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
        print(f"  >{thr:.0%}: {n_sel} selected, {n_wins_sel} wins ({n_wins_sel/n_sel*100:.1f}%) "
              f"vs base {base_rate*100:.1f}%, p={p_val:.4f} {sig}")

# ══════════════════════════════════════════════════════════════════════
# STEP 4: Monthly regime breakdown
# ══════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("STEP 4: MONTHLY BREAKDOWN — Does signal quality change by regime?")
print("=" * 70)

months_idx = pd.date_range(df.index[0].replace(day=1), df.index[-1], freq='MS')
print(f"\n  {'Month':>7} {'BTC%':>7} | {'L_base':>7} {'L_wins':>7} | {'S_base':>7} {'S_wins':>7}")
print(f"  {'-'*55}")

for i in range(len(months_idx) - 1):
    m_start = months_idx[i]; m_end = months_idx[i+1]
    mask = (df.index >= m_start) & (df.index < m_end)
    idx_m = np.where(mask)[0]
    if len(idx_m) < 100: continue

    btc_ret = (c.iloc[idx_m[-1]] / c.iloc[idx_m[0]] - 1) * 100
    lw = y_long[idx_m].sum(); lt = len(idx_m)
    sw = y_short[idx_m].sum()
    print(f"  {m_start.strftime('%Y-%m'):>7} {btc_ret:>+6.1f}% | "
          f"{lw/lt*100:6.1f}% {lw:>6,} | "
          f"{sw/lt*100:6.1f}% {sw:>6,}")

# ══════════════════════════════════════════════════════════════════════
# STEP 5: Final verdict
# ══════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("STEP 5: FINAL VERDICT")
print("=" * 70)
print("""
  This assessment answers:
  1. How many 1%+ winning trades exist in 525K bars?
  2. How many features were tested for predictive power?
  3. What is the individual and combined predictive power?
  4. Is the signal statistically significant?
  5. Is the signal consistent across market regimes?
""")
