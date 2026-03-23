#!/usr/bin/env python3
"""
Train TP/SL v4 — direction-specific models with direct TP/SL label targets.

Instead of predicting "will price go up?", we predict:
  - LONG model:  "will TP=+1.0% hit before SL=-0.5%?"
  - SHORT model: "will TP=-1.0% hit before SL=+0.5%?"

This directly models the trading outcome, not just direction.

Uses 12 months of 1-min Binance data, 75 features, HistGradientBoosting.
"""

import numpy as np
import pandas as pd
import time
import os
import pickle
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report

DATA_PATH = 'data/btc_1m_12mo.parquet'
OUT_DIR = 'models/tpsl_v4'
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

# ══════════════════════════════════════════════════════════════════════
# STEP 2: Compute features (same 75 as v3)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: Computing features (75)")
print("=" * 70)
t0 = time.time()

c = df['close'].astype(float)
h = df['high'].astype(float)
l = df['low'].astype(float)
o = df['open'].astype(float)
v = df['volume'].astype(float)
r1 = c.pct_change()

feat = pd.DataFrame(index=df.index)

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
for p in [7, 14, 30, 60]:
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(span=p).mean()
    loss = (-delta.clip(upper=0)).ewm(span=p).mean()
    feat[f'rsi_{p}'] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
for w in [20, 60, 120]:
    ma = c.rolling(w).mean(); std = c.rolling(w).std()
    feat[f'bb_pct_{w}'] = (c - (ma - 2*std)) / (4*std).replace(0, np.nan)
for w in [14, 30, 60, 120]:
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    feat[f'atr_pct_{w}'] = tr.rolling(w).mean() / c * 100
for fast, slow in [(12,26), (5,15), (30,60)]:
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

feat = feat.replace([np.inf, -np.inf], np.nan)
feature_cols = list(feat.columns)
print(f"  {len(feat):,} rows, {len(feature_cols)} features in {time.time()-t0:.1f}s")

# ══════════════════════════════════════════════════════════════════════
# STEP 3: Build DIRECT TP/SL labels — vectorized (fast)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: Building direct TP/SL labels (vectorized)")
print("=" * 70)
t0 = time.time()

p_arr = c.values.astype(np.float64)
n = len(p_arr)

TP_PCT = 1.0
SL_PCT = 0.5
MAX_BARS_LIST = [60, 120, 240]

def build_labels_vectorized(prices, tp_pct, sl_pct, max_bars):
    """Vectorized TP/SL label builder using forward rolling max/min."""
    n = len(prices)
    # Forward returns matrix: for each bar i, what is the max/min price in next max_bars?
    # We iterate over offsets 1..max_bars and track running cummax/cummin + first-hit bar
    long_tp_bar = np.full(n, 9999, dtype=np.int32)
    long_sl_bar = np.full(n, 9999, dtype=np.int32)
    short_tp_bar = np.full(n, 9999, dtype=np.int32)
    short_sl_bar = np.full(n, 9999, dtype=np.int32)

    tp_up = prices * (1 + tp_pct / 100)    # LONG TP price
    sl_down = prices * (1 - sl_pct / 100)  # LONG SL price
    tp_down = prices * (1 - tp_pct / 100)  # SHORT TP price
    sl_up = prices * (1 + sl_pct / 100)    # SHORT SL price

    for offset in range(1, max_bars + 1):
        # Future price at this offset
        fwd_idx = np.arange(n) + offset
        valid = fwd_idx < n
        fwd_price = np.where(valid, prices[np.clip(fwd_idx, 0, n-1)], np.nan)

        # LONG: first bar where price >= TP
        hit = valid & (fwd_price >= tp_up) & (long_tp_bar == 9999)
        long_tp_bar[hit] = offset
        # LONG: first bar where price <= SL
        hit = valid & (fwd_price <= sl_down) & (long_sl_bar == 9999)
        long_sl_bar[hit] = offset
        # SHORT: first bar where price <= TP
        hit = valid & (fwd_price <= tp_down) & (short_tp_bar == 9999)
        short_tp_bar[hit] = offset
        # SHORT: first bar where price >= SL
        hit = valid & (fwd_price >= sl_up) & (short_sl_bar == 9999)
        short_sl_bar[hit] = offset

    # LONG wins: TP hit before SL
    y_long = ((long_tp_bar < long_sl_bar) & (long_tp_bar < 9999)).astype(np.int8)
    # SHORT wins: TP hit before SL
    y_short = ((short_tp_bar < short_sl_bar) & (short_tp_bar < 9999)).astype(np.int8)
    return y_long, y_short

labels = {}
for mb in MAX_BARS_LIST:
    t1 = time.time()
    yl, ys = build_labels_vectorized(p_arr, TP_PCT, SL_PCT, mb)
    labels[f'long_{mb}'] = yl
    labels[f'short_{mb}'] = ys
    dt = time.time() - t1
    print(f"  horizon={mb}min: LONG {yl.mean()*100:.1f}% wins | SHORT {ys.mean()*100:.1f}% wins  ({dt:.1f}s)")

print(f"  Total label time: {time.time()-t0:.1f}s")

# ══════════════════════════════════════════════════════════════════════
# STEP 4: Train direction-specific models
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4: Training direction-specific models")
print("=" * 70)

# Walk-forward split: 75% train, 25% test
valid = feat.dropna().index
feat_clean = feat.loc[valid]
n_clean = len(feat_clean)
split = int(n_clean * 0.75)
X_tr = feat_clean.iloc[:split][feature_cols].fillna(0).values
X_te = feat_clean.iloc[split:][feature_cols].fillna(0).values
idx_tr = feat_clean.index[:split]
idx_te = feat_clean.index[split:]

# Map indices to positions in original arrays
pos_tr = np.searchsorted(df.index, idx_tr)
pos_te = np.searchsorted(df.index, idx_te)

print(f"  Train: {len(X_tr):,} | Test: {len(X_te):,}")
print(f"  Train: {idx_tr[0].date()} → {idx_tr[-1].date()}")
print(f"  Test:  {idx_te[0].date()} → {idx_te[-1].date()}")

results = []

for horizon in MAX_BARS_LIST:
    for direction in ['long', 'short']:
        key = f'{direction}_{horizon}'
        y_all = labels[key]
        y_tr = y_all[pos_tr]
        y_te = y_all[pos_te]

        base_wr = y_te.mean() * 100
        be = SL_PCT / (TP_PCT + SL_PCT) * 100  # 33.3%

        for model_name, model_cls, params in [
            ('HGB', HistGradientBoostingClassifier,
             dict(max_iter=300, max_depth=5, min_samples_leaf=100, learning_rate=0.05, random_state=42)),
            ('RF', RandomForestClassifier,
             dict(n_estimators=100, max_depth=8, min_samples_leaf=50, n_jobs=-1, random_state=42)),
        ]:
            t0 = time.time()
            m = model_cls(**params)
            m.fit(X_tr, y_tr)
            probs = m.predict_proba(X_te)[:, 1]
            dt = time.time() - t0

            # Get vol_60 for filtering
            vol_60_vals = feat_clean.iloc[split:]['vol_60'].fillna(0).values
            vol_60_median = feat_clean.iloc[:split]['vol_60'].median()

            # Evaluate at multiple thresholds and vol filters
            for thr in [0.55, 0.60, 0.65, 0.70, 0.75]:
                for vol_mult, vol_label in [(1.0, '1.0x'), (1.5, '1.5x'), (2.0, '2.0x'), (2.5, '2.5x')]:
                    mask = (probs > thr) & (vol_60_vals >= vol_60_median * vol_mult)
                    n_trades = mask.sum()
                    if n_trades < 20:
                        continue
                    wins = y_te[mask].sum()
                    losses = n_trades - wins
                    wr = wins / n_trades * 100
                    ev = (wr/100 * TP_PCT) - ((100-wr)/100 * SL_PCT)
                    results.append({
                        'direction': direction.upper(),
                        'horizon': horizon,
                        'model': model_name,
                        'threshold': thr,
                        'vol_filter': vol_label,
                        'trades': n_trades,
                        'wins': wins,
                        'losses': losses,
                        'wr': wr,
                        'ev': ev,
                    })

            print(f"  {key} {model_name}: trained in {dt:.1f}s | base WR={base_wr:.1f}% (BE={be:.1f}%)")

# ══════════════════════════════════════════════════════════════════════
# STEP 5: Results
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 5: Results — all profitable combos")
print("=" * 70)

res_df = pd.DataFrame(results)
profitable = res_df[res_df['ev'] > 0].sort_values('ev', ascending=False)
print(f"\n{len(profitable)} profitable out of {len(res_df)} tested\n")

print("TOP 30 STRATEGIES:")
print(f"  {'#':>3}  {'Dir':>5} {'Hrz':>4} {'Model':>5} {'Thr':>5} {'Vol':>5} | {'Trades':>6} {'Wins':>5} {'Loss':>5} | {'WR%':>6} {'EV%':>7}")
print("-" * 85)

for i, (_, row) in enumerate(profitable.head(30).iterrows()):
    print(f"  {i+1:3d}  {row['direction']:>5} {row['horizon']:4d} {row['model']:>5} {row['threshold']:5.0%} {row['vol_filter']:>5} | {row['trades']:6d} {row['wins']:5.0f} {row['losses']:5.0f} | {row['wr']:5.1f}% {row['ev']:+6.3f}%")

# Best overall
best = profitable.iloc[0]
print(f"\nBEST: {best['direction']} horizon={best['horizon']}min {best['model']} >{best['threshold']:.0%} vol>={best['vol_filter']}")
print(f"  {best['trades']}T, {best['wr']:.1f}%WR, EV={best['ev']:+.3f}%")

# Best SHORT-only
short_prof = profitable[profitable['direction'] == 'SHORT']
if len(short_prof) > 0:
    best_s = short_prof.iloc[0]
    print(f"\nBEST SHORT: horizon={best_s['horizon']}min {best_s['model']} >{best_s['threshold']:.0%} vol>={best_s['vol_filter']}")
    print(f"  {best_s['trades']}T, {best_s['wr']:.1f}%WR, EV={best_s['ev']:+.3f}%")

# Best LONG-only
long_prof = profitable[profitable['direction'] == 'LONG']
if len(long_prof) > 0:
    best_l = long_prof.iloc[0]
    print(f"\nBEST LONG: horizon={best_l['horizon']}min {best_l['model']} >{best_l['threshold']:.0%} vol>={best_l['vol_filter']}")
    print(f"  {best_l['trades']}T, {best_l['wr']:.1f}%WR, EV={best_l['ev']:+.3f}%")
else:
    print("\nNo profitable LONG strategies found.")

# Save best model
print("\nSaving best model...")
# Retrain best model on the correct target
best_key = f"{best['direction'].lower()}_{best['horizon']}"
y_all_best = labels[best_key]
y_tr_best = y_all_best[pos_tr]

if best['model'] == 'HGB':
    best_model = HistGradientBoostingClassifier(
        max_iter=300, max_depth=5, min_samples_leaf=100, learning_rate=0.05, random_state=42)
else:
    best_model = RandomForestClassifier(
        n_estimators=100, max_depth=8, min_samples_leaf=50, n_jobs=-1, random_state=42)

best_model.fit(X_tr, y_tr_best)

with open(os.path.join(OUT_DIR, 'best_model.pkl'), 'wb') as f:
    pickle.dump(best_model, f)
with open(os.path.join(OUT_DIR, 'feature_cols.pkl'), 'wb') as f:
    pickle.dump(feature_cols, f)

vol_60_median = feat_clean.iloc[:split]['vol_60'].median()
config = {
    'direction': best['direction'],
    'horizon': int(best['horizon']),
    'model_type': best['model'],
    'threshold': float(best['threshold']),
    'vol_filter': best['vol_filter'],
    'vol_60_median': float(vol_60_median),
    'wr': float(best['wr']),
    'ev': float(best['ev']),
    'trades': int(best['trades']),
    'tp': TP_PCT,
    'sl': SL_PCT,
}
with open(os.path.join(OUT_DIR, 'config.pkl'), 'wb') as f:
    pickle.dump(config, f)

print(f"  Saved to {OUT_DIR}/")
print(f"  Config: {config}")

# Also save full results
res_df.to_csv(os.path.join(OUT_DIR, 'all_results.csv'), index=False)
print(f"  Full results: {OUT_DIR}/all_results.csv")
print("\nDone!")
