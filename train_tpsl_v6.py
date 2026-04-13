#!/usr/bin/env python3
"""
Train TP/SL v6 — Better problem definition + richer features + proper validation.

PROBLEM REDEFINITION:
The issue is NOT that there's no signal — the assessment proved LONG signal is
statistically significant (p<0.0001). The issue is the signal is REGIME-DEPENDENT:
base rate swings from 8.5% to 24.9% across months.

FIX: Instead of predicting "will TP hit?" in absolute terms, we need features that
CAPTURE THE CURRENT REGIME so the model knows what base rate it's working with.

NEW FEATURES (on top of 83 base):
1. Trajectory features: how price moved in last 30/60/120/240 min
   - Max drawdown, max runup, path efficiency, reversal count
2. Microstructure: tick intensity, volume clustering, spread proxies
3. Regime encoding: rolling base rate estimate, volatility regime percentile
4. Cross-timeframe momentum alignment
5. Order flow proxies: buy/sell volume imbalance

VALIDATION:
- Rolling 3-month train, 1-month test, across all 9 testable months
- Report per-month AND aggregate results
- Only deploy if profitable in >= 6/9 months
"""

import numpy as np
import pandas as pd
import time
import os
import pickle
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import binomtest

DATA_PATH = 'data/btc_1m_12mo.parquet'
OUT_DIR = 'models/tpsl_v6'
os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# STEP 1: Load data
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 1: Loading data")
print("=" * 70)
df = pd.read_parquet(DATA_PATH)
df.index = df.index.tz_localize(None) if df.index.tz else df.index
df = df.sort_index()
print(f"  {len(df):,} candles | {df.index[0].date()} → {df.index[-1].date()}")

c = df['close'].astype(float)
h = df['high'].astype(float)
l = df['low'].astype(float)
o = df['open'].astype(float)
v = df['volume'].astype(float)
r1 = c.pct_change()
p_arr = c.values.astype(np.float64)
n = len(p_arr)

# ══════════════════════════════════════════════════════════════════════
# STEP 2: Compute features (base + trajectory + regime)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: Computing features")
print("=" * 70)
t0 = time.time()

feat = pd.DataFrame(index=df.index)

# ── A. BASE FEATURES (same 75 as before) ──
for w in [1, 2, 3, 5, 10, 15, 30, 60, 120, 240]:
    feat[f'ret_{w}'] = c.pct_change(w) * 100
for w in [5, 10, 15, 30, 60, 120, 240]:
    feat[f'vol_{w}'] = r1.rolling(w).std() * 100
feat['vol_ratio_5_60'] = feat['vol_5'] / feat['vol_60'].replace(0, np.nan)
feat['vol_ratio_15_60'] = feat['vol_15'] / feat['vol_60'].replace(0, np.nan)
feat['vol_ratio_30_120'] = feat['vol_30'] / feat['vol_120'].replace(0, np.nan)
feat['vol_ratio_60_240'] = feat['vol_60'] / feat['vol_240'].replace(0, np.nan)
for w in [5, 15, 30, 60, 120]:
    feat[f'vratio_{w}'] = v / v.rolling(w).mean().replace(0, np.nan)
for p_rsi in [7, 14, 30, 60]:
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(span=p_rsi).mean()
    loss = (-delta.clip(upper=0)).ewm(span=p_rsi).mean()
    feat[f'rsi_{p_rsi}'] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
for w in [20, 60, 120]:
    ma = c.rolling(w).mean(); std = c.rolling(w).std()
    feat[f'bb_pct_{w}'] = (c - (ma - 2*std)) / (4*std).replace(0, np.nan)
for w in [14, 30, 60, 120]:
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    feat[f'atr_pct_{w}'] = tr.rolling(w).mean() / c * 100
for fast, slow in [(12, 26), (5, 15), (30, 60)]:
    ema_f = c.ewm(span=fast).mean(); ema_s = c.ewm(span=slow).mean()
    diff = ema_f - ema_s; sig = diff.ewm(span=9).mean()
    feat[f'macd_{fast}_{slow}'] = diff / c * 100
    feat[f'macd_hist_{fast}_{slow}'] = (diff - sig) / c * 100
for w in [60, 120, 240, 480]:
    rh = h.rolling(w).max(); rl = l.rolling(w).min()
    feat[f'pos_high_{w}'] = (c - rh) / c * 100
    feat[f'pos_low_{w}'] = (c - rl) / c * 100
    feat[f'pos_range_{w}'] = (c - rl) / (rh - rl).replace(0, np.nan)
feat['bar_range'] = (h - l) / c * 100
feat['bar_body'] = (c - o).abs() / c * 100
feat['upper_wick'] = (h - c.clip(lower=o)) / c * 100
feat['lower_wick'] = (c.clip(upper=o) - l) / c * 100
feat['bar_range_ma30'] = feat['bar_range'].rolling(30).mean()
feat['bar_range_ratio'] = feat['bar_range'] / feat['bar_range_ma30'].replace(0, np.nan)
for w in [30, 60, 120]:
    ma = c.rolling(w).mean()
    feat[f'dist_ma_{w}'] = (c - ma) / ma * 100
feat['vol_price_corr_30'] = r1.rolling(30).corr(v.pct_change())
feat['vol_price_corr_60'] = r1.rolling(60).corr(v.pct_change())
feat['roc_5'] = c.pct_change(5) * 100
feat['roc_15'] = c.pct_change(15) * 100
feat['roc_60'] = c.pct_change(60) * 100
feat['up_count_10'] = sum((r1.shift(i) > 0).astype(int) for i in range(10))
feat['down_count_10'] = 10 - feat['up_count_10']
hour = df.index.hour + df.index.minute / 60
feat['hour_sin'] = np.sin(2 * np.pi * hour / 24)
feat['hour_cos'] = np.cos(2 * np.pi * hour / 24)
dow = df.index.dayofweek
feat['dow_sin'] = np.sin(2 * np.pi * dow / 7)
feat['dow_cos'] = np.cos(2 * np.pi * dow / 7)
n_base = len(feat.columns)

# ── B. TRAJECTORY FEATURES: How price moved in last N minutes ──
# These capture the "shape" of recent price movement, not just endpoints
for w in [30, 60, 120, 240]:
    # Max drawdown from peak in last w bars
    rolling_high = h.rolling(w).max()
    feat[f'traj_drawdown_{w}'] = (c - rolling_high) / rolling_high * 100

    # Max runup from trough in last w bars
    rolling_low = l.rolling(w).min()
    feat[f'traj_runup_{w}'] = (c - rolling_low) / rolling_low * 100

    # Path efficiency: net move / total absolute moves (1.0 = straight line)
    abs_moves = r1.abs().rolling(w).sum()
    net_move = (c / c.shift(w) - 1).abs()
    feat[f'traj_efficiency_{w}'] = net_move / abs_moves.replace(0, np.nan)

    # Reversal count: how many direction changes in last w bars
    direction_changes = (r1.shift(0).gt(0) != r1.shift(1).gt(0)).astype(float)
    feat[f'traj_reversals_{w}'] = direction_changes.rolling(w).sum()

    # Acceleration: second derivative of price (is the move speeding up?)
    ret_w = c.pct_change(w//2) * 100
    ret_w_prev = c.shift(w//2).pct_change(w//2) * 100
    feat[f'traj_accel_{w}'] = ret_w - ret_w_prev

    # Ratio of time spent going up vs down
    up_bars = (r1 > 0).astype(float).rolling(w).sum()
    feat[f'traj_up_ratio_{w}'] = up_bars / w

    # Volume-weighted direction: are big-volume bars up or down?
    vw_dir = (r1 * v).rolling(w).sum() / v.rolling(w).sum().replace(0, np.nan)
    feat[f'traj_vw_direction_{w}'] = vw_dir * 100

# ── C. MICROSTRUCTURE / ORDER FLOW PROXIES ──
# Buy vs sell volume proxy (close > open = buy, close < open = sell)
buy_vol = v.where(c >= o, 0)
sell_vol = v.where(c < o, 0)
for w in [5, 15, 30, 60, 120]:
    bv = buy_vol.rolling(w).sum()
    sv = sell_vol.rolling(w).sum()
    feat[f'buy_sell_ratio_{w}'] = bv / sv.replace(0, np.nan)
    feat[f'buy_sell_imbalance_{w}'] = (bv - sv) / (bv + sv).replace(0, np.nan)

# Volume clustering: are we in a high-vol or low-vol cluster?
vol_ma = v.rolling(60).mean()
vol_std = v.rolling(60).std()
feat['vol_zscore'] = (v - vol_ma) / vol_std.replace(0, np.nan)
feat['vol_zscore_5'] = feat['vol_zscore'].rolling(5).mean()  # sustained high vol?

# Large bar detection: how many bars in last N were "large" (>2 std)
bar_range_std = feat['bar_range'].rolling(120).std()
bar_range_mean = feat['bar_range'].rolling(120).mean()
large_bar = (feat['bar_range'] > bar_range_mean + 2 * bar_range_std).astype(float)
for w in [10, 30, 60]:
    feat[f'large_bars_{w}'] = large_bar.rolling(w).sum()

# ── D. REGIME ENCODING ──
# Trend direction and strength at multiple timeframes
for w in [360, 720, 1440, 4320, 10080]:  # 6h, 12h, 1d, 3d, 7d
    feat[f'trend_{w}'] = c.pct_change(w) * 100

# Drawdown from rolling high (regime awareness)
for w in [1440, 4320, 10080]:
    rh = c.rolling(w).max()
    feat[f'drawdown_{w}'] = (c - rh) / rh * 100

# Volatility regime: current vs historical percentile
vol_60 = feat['vol_60']
for lookback in [1440, 10080]:
    feat[f'vol_pctile_{lookback}'] = vol_60.rolling(lookback).rank(pct=True)

# ATR regime percentile
atr_60 = feat['atr_pct_60']
for lookback in [1440, 10080]:
    feat[f'atr_pctile_{lookback}'] = atr_60.rolling(lookback).rank(pct=True)

# Mean-reversion potential: distance from VWAP
for w in [60, 240, 1440]:
    vwap = (c * v).rolling(w).sum() / v.rolling(w).sum().replace(0, np.nan)
    feat[f'dist_vwap_{w}'] = (c - vwap) / vwap * 100

# ── E. CROSS-TIMEFRAME MOMENTUM ALIGNMENT ──
# Are all timeframes aligned? (strong signal when they are)
mom_signs = pd.DataFrame(index=df.index)
for w in [5, 15, 60, 240, 1440]:
    mom_signs[f'm_{w}'] = np.sign(c.pct_change(w))

# Count how many timeframes agree on direction
feat['tf_bull_count'] = (mom_signs > 0).sum(axis=1)
feat['tf_bear_count'] = (mom_signs < 0).sum(axis=1)
feat['tf_alignment'] = feat['tf_bull_count'] - feat['tf_bear_count']  # +5 = all bullish

# ── F. SUPPORT/RESISTANCE PROXIMITY ──
# How far from recent local extremes
for w in [60, 240, 1440]:
    recent_high = h.rolling(w).max()
    recent_low = l.rolling(w).min()
    rng = (recent_high - recent_low).replace(0, np.nan)
    feat[f'dist_support_{w}'] = (c - recent_low) / rng  # 0 = at support, 1 = at resistance
    # Number of times price touched the support/resistance zone (within 0.1% of extremes)
    near_low = (c <= recent_low * 1.001).astype(float)
    near_high = (c >= recent_high * 0.999).astype(float)
    feat[f'support_touches_{w}'] = near_low.rolling(w).sum()
    feat[f'resist_touches_{w}'] = near_high.rolling(w).sum()

feat = feat.replace([np.inf, -np.inf], np.nan)
feature_cols = list(feat.columns)
n_new = len(feature_cols) - n_base
print(f"  {n_base} base + {n_new} new = {len(feature_cols)} total features in {time.time()-t0:.1f}s")

# ══════════════════════════════════════════════════════════════════════
# STEP 3: Build labels
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: Building labels (TP=1.0%/SL=0.5%, 240 bars)")
print("=" * 70)
t0 = time.time()

TP_PCT = 1.0; SL_PCT = 0.5; MAX_BARS = 240
long_tp = np.full(n, 9999, dtype=np.int32); long_sl = np.full(n, 9999, dtype=np.int32)
short_tp = np.full(n, 9999, dtype=np.int32); short_sl = np.full(n, 9999, dtype=np.int32)
for off in range(1, MAX_BARS + 1):
    idx = np.arange(n) + off; valid = idx < n
    fwd = np.where(valid, p_arr[np.clip(idx, 0, n-1)], np.nan)
    hit = valid & (fwd >= p_arr * (1 + TP_PCT/100)) & (long_tp == 9999); long_tp[hit] = off
    hit = valid & (fwd <= p_arr * (1 - SL_PCT/100)) & (long_sl == 9999); long_sl[hit] = off
    hit = valid & (fwd <= p_arr * (1 - TP_PCT/100)) & (short_tp == 9999); short_tp[hit] = off
    hit = valid & (fwd >= p_arr * (1 + SL_PCT/100)) & (short_sl == 9999); short_sl[hit] = off
y_long = ((long_tp < long_sl) & (long_tp < 9999)).astype(np.int8)
y_short = ((short_tp < short_sl) & (short_tp < 9999)).astype(np.int8)
print(f"  LONG wins: {y_long.sum():,}/{n:,} ({y_long.mean()*100:.1f}%)")
print(f"  SHORT wins: {y_short.sum():,}/{n:,} ({y_short.mean()*100:.1f}%)")
print(f"  Labels built in {time.time()-t0:.1f}s")

# ══════════════════════════════════════════════════════════════════════
# STEP 4: Walk-forward with rolling 3-month window
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4: Walk-forward (3-month rolling train, 1-month test)")
print("=" * 70)

valid_mask = feat.notna().all(axis=1).values
feat_vals = feat[feature_cols].fillna(0).values
months = pd.date_range(df.index[0].replace(day=1), df.index[-1], freq='MS')
TRAIN_MONTHS = 3

results = []
all_monthly = []

for test_month_idx in range(TRAIN_MONTHS, len(months)):
    test_start = months[test_month_idx]
    test_end = months[test_month_idx + 1] if test_month_idx + 1 < len(months) else df.index[-1]
    train_start = months[test_month_idx - TRAIN_MONTHS]
    train_end = test_start

    tr_mask = (df.index >= train_start) & (df.index < train_end) & valid_mask
    te_mask = (df.index >= test_start) & (df.index < test_end) & valid_mask
    tr_idx = np.where(tr_mask)[0]
    te_idx = np.where(te_mask)[0]
    if len(tr_idx) < 1000 or len(te_idx) < 100:
        continue

    X_tr = feat_vals[tr_idx]; X_te = feat_vals[te_idx]
    te_ret = (p_arr[te_idx[-1]] / p_arr[te_idx[0]] - 1) * 100

    for direction, y_arr in [('LONG', y_long), ('SHORT', y_short)]:
        y_tr = y_arr[tr_idx]; y_te = y_arr[te_idx]
        base_wr = y_te.mean() * 100
        train_base = y_tr.mean() * 100

        model = HistGradientBoostingClassifier(
            max_iter=300, max_depth=5, min_samples_leaf=100,
            learning_rate=0.05, random_state=42, l2_regularization=0.5)
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_te)[:, 1]

        try:
            auc = roc_auc_score(y_te, probs)
        except:
            auc = 0.5

        for thr in [0.30, 0.40, 0.50, 0.60, 0.70]:
            sel = probs > thr
            nt = sel.sum()
            if nt < 5: continue
            w = int(y_te[sel].sum())
            wr = w / nt * 100
            ev = (wr/100 * TP_PCT) - ((100-wr)/100 * SL_PCT)
            be = SL_PCT / (TP_PCT + SL_PCT) * 100

            results.append({
                'month': test_start.strftime('%Y-%m'),
                'dir': direction,
                'thr': thr,
                'trades': nt, 'wins': w, 'wr': wr, 'ev': ev,
                'base_wr': base_wr, 'train_base': train_base,
                'auc': auc, 'btc_ret': te_ret,
            })

    # Print progress
    print(f"  {test_start.strftime('%Y-%m')}: BTC {te_ret:+.1f}%, "
          f"LONG base={y_long[te_idx].mean()*100:.1f}%, "
          f"SHORT base={y_short[te_idx].mean()*100:.1f}%")

# ══════════════════════════════════════════════════════════════════════
# STEP 5: Results
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 5: WALK-FORWARD RESULTS")
print("=" * 70)

res_df = pd.DataFrame(results)
res_df.to_csv(os.path.join(OUT_DIR, 'walkforward_results.csv'), index=False)

# Aggregate by direction and threshold
for direction in ['LONG', 'SHORT']:
    print(f"\n  ── {direction} ──")
    for thr in [0.30, 0.40, 0.50, 0.60, 0.70]:
        sub = res_df[(res_df['dir'] == direction) & (res_df['thr'] == thr)]
        if len(sub) == 0: continue
        total_t = sub['trades'].sum()
        total_w = sub['wins'].sum()
        if total_t == 0: continue
        wr = total_w / total_t * 100
        ev = (wr/100 * TP_PCT) - ((100-wr)/100 * SL_PCT)
        months_prof = (sub['ev'] > 0).sum()
        months_total = len(sub)
        print(f"  >{thr:.0%}: {total_t:>5}T {total_w:>4}W "
              f"WR={wr:5.1f}% EV={ev:+.3f}% "
              f"({months_prof}/{months_total} months profitable)")

# Monthly detail for best strategy
print("\n" + "=" * 70)
print("MONTHLY DETAIL — Best strategies")
print("=" * 70)

for direction in ['LONG', 'SHORT']:
    # Find best threshold by aggregate EV
    best_thr = None; best_ev = -999
    for thr in [0.30, 0.40, 0.50, 0.60, 0.70]:
        sub = res_df[(res_df['dir'] == direction) & (res_df['thr'] == thr)]
        if len(sub) == 0: continue
        total_t = sub['trades'].sum(); total_w = sub['wins'].sum()
        if total_t < 20: continue
        wr = total_w / total_t * 100
        ev = (wr/100 * TP_PCT) - ((100-wr)/100 * SL_PCT)
        if ev > best_ev:
            best_ev = ev; best_thr = thr

    if best_thr is None: continue
    sub = res_df[(res_df['dir'] == direction) & (res_df['thr'] == best_thr)].sort_values('month')
    total_t = sub['trades'].sum(); total_w = sub['wins'].sum()
    wr = total_w / total_t * 100
    ev = (wr/100 * TP_PCT) - ((100-wr)/100 * SL_PCT)

    print(f"\n  {direction} >{best_thr:.0%} — Aggregate: {total_t}T, WR={wr:.1f}%, EV={ev:+.3f}%")
    print(f"  {'Month':>7} {'BTC%':>7} {'Base':>6} | {'#T':>5} {'#W':>4} {'WR':>6} {'EV':>7} {'AUC':>5} |")
    print(f"  {'-'*58}")
    for _, row in sub.iterrows():
        status = "+" if row['ev'] > 0 else "-"
        print(f"  {row['month']:>7} {row['btc_ret']:>+6.1f}% {row['base_wr']:>5.1f}% | "
              f"{row['trades']:>5.0f} {row['wins']:>4.0f} {row['wr']:>5.1f}% {row['ev']:>+6.3f}% "
              f"{row['auc']:>.3f} | {status}")

    # Statistical significance: binomial test on aggregate
    p_val = binomtest(int(total_w), int(total_t), SL_PCT / (TP_PCT + SL_PCT),
                      alternative='greater').pvalue
    print(f"  Binomial test vs breakeven (33.3%): p={p_val:.6f} "
          f"{'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'}")

# ══════════════════════════════════════════════════════════════════════
# STEP 6: Feature importance from best model
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 6: TOP FEATURES (from last fold)")
print("=" * 70)

# Retrain on last 3 months
last_3m_start = months[-TRAIN_MONTHS] if len(months) > TRAIN_MONTHS else months[0]
tr_mask_final = (df.index >= last_3m_start) & valid_mask
tr_idx_final = np.where(tr_mask_final)[0]
X_final = feat_vals[tr_idx_final]

for direction, y_arr in [('LONG', y_long), ('SHORT', y_short)]:
    y_final = y_arr[tr_idx_final]
    model = HistGradientBoostingClassifier(
        max_iter=300, max_depth=5, min_samples_leaf=100,
        learning_rate=0.05, random_state=42, l2_regularization=0.5)
    model.fit(X_final, y_final)

    # Permutation-based importance is more reliable but slow; use built-in for speed
    # Note: HGB doesn't have feature_importances_ by default, so we use a workaround
    # Train AUC to verify model learned something
    probs_train = model.predict_proba(X_final)[:, 1]
    try:
        train_auc = roc_auc_score(y_final, probs_train)
    except:
        train_auc = 0.5

    print(f"\n  {direction} model (last 3mo train AUC: {train_auc:.4f})")
    print(f"  Base rate: {y_final.mean()*100:.1f}%")

    # Get feature importance via permutation (sample for speed)
    np.random.seed(42)
    sample_idx = np.random.choice(len(X_final), size=min(50000, len(X_final)), replace=False)
    X_sample = X_final[sample_idx]
    y_sample = y_final[sample_idx]
    base_score = roc_auc_score(y_sample, model.predict_proba(X_sample)[:, 1])

    importances = []
    for fi in range(len(feature_cols)):
        X_perm = X_sample.copy()
        X_perm[:, fi] = np.random.permutation(X_perm[:, fi])
        perm_score = roc_auc_score(y_sample, model.predict_proba(X_perm)[:, 1])
        imp = base_score - perm_score
        importances.append((feature_cols[fi], imp))

    importances.sort(key=lambda x: -x[1])
    print(f"  {'Feature':<30} {'Importance':>12}")
    print(f"  {'-'*44}")
    for fname, imp in importances[:25]:
        bar = "█" * int(imp * 500) if imp > 0 else ""
        is_new = " ★" if fname not in feature_cols[:n_base] else ""
        print(f"  {fname:<30} {imp:>+10.5f}  {bar}{is_new}")

    # Save model
    with open(os.path.join(OUT_DIR, f'model_{direction.lower()}.pkl'), 'wb') as f:
        pickle.dump(model, f)

with open(os.path.join(OUT_DIR, 'feature_cols.pkl'), 'wb') as f:
    pickle.dump(feature_cols, f)

print(f"\n  Models saved to {OUT_DIR}/")
print("\nDone!")
