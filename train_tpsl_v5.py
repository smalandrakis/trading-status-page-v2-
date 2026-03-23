#!/usr/bin/env python3
"""
Train TP/SL v5 — Rolling retrain + regime features, walk-forward validated.

Key changes from v4:
1. Rolling 3-month training window (not expanding) — matches current regime
2. Regime features: trend slope, drawdown, volatility regime, trend strength
3. Walk-forward across 9 monthly folds (months 4-12)
4. Also test multiple TP/SL ratios (0.3/0.15, 0.5/0.25, 1.0/0.5)
5. Both LONG and SHORT models per fold
"""

import numpy as np
import pandas as pd
import time
import os
import pickle
from sklearn.ensemble import HistGradientBoostingClassifier

DATA_PATH = 'data/btc_1m_12mo.parquet'
OUT_DIR = 'models/tpsl_v5'
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
# STEP 2: Compute features (75 base + regime features)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: Computing features (75 base + regime)")
print("=" * 70)
t0 = time.time()

c = df['close'].astype(float)
h = df['high'].astype(float)
l = df['low'].astype(float)
o = df['open'].astype(float)
v = df['volume'].astype(float)
r1 = c.pct_change()

feat = pd.DataFrame(index=df.index)

# --- Base 75 features (same as v3/v4) ---
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

# --- NEW: Regime features ---
# Trend slope over different horizons (linear regression slope)
for w in [360, 720, 1440, 4320, 10080]:  # 6h, 12h, 1d, 3d, 7d in minutes
    feat[f'trend_slope_{w}'] = c.pct_change(w) * 100

# Drawdown from rolling high
for w in [1440, 4320, 10080]:  # 1d, 3d, 7d
    rh = c.rolling(w).max()
    feat[f'drawdown_{w}'] = (c - rh) / rh * 100

# Volatility regime: current vol vs long-term vol
feat['vol_regime_60_1440'] = feat['vol_60'] / r1.rolling(1440).std().replace(0, np.nan) / 100
feat['vol_regime_240_10080'] = feat['vol_240'] / r1.rolling(10080).std().replace(0, np.nan) / 100

# Trend strength: ADX-like measure
for w in [60, 240, 1440]:
    plus_dm = (h - h.shift(1)).clip(lower=0)
    minus_dm = (l.shift(1) - l).clip(lower=0)
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(w).mean()
    plus_di = (plus_dm.rolling(w).mean() / atr.replace(0, np.nan)) * 100
    minus_di = (minus_dm.rolling(w).mean() / atr.replace(0, np.nan)) * 100
    feat[f'adx_{w}'] = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100).rolling(w).mean()
    feat[f'di_diff_{w}'] = plus_di - minus_di

# Mean-reversion signal: distance from VWAP-like measure
for w in [60, 240, 1440]:
    vwap = (c * v).rolling(w).sum() / v.rolling(w).sum().replace(0, np.nan)
    feat[f'dist_vwap_{w}'] = (c - vwap) / vwap * 100

# Higher-timeframe momentum (resampled)
c_5m = c.resample('5min').last().dropna()
for w in [12, 48, 288]:  # 1h, 4h, 1d in 5-min bars
    mom = c_5m.pct_change(w) * 100
    feat[f'htf_mom_{w*5}'] = mom.reindex(feat.index, method='ffill')

feat = feat.replace([np.inf, -np.inf], np.nan)
feature_cols = list(feat.columns)
n_base = 75
n_regime = len(feature_cols) - n_base
print(f"  {len(feat):,} rows, {n_base} base + {n_regime} regime = {len(feature_cols)} features in {time.time()-t0:.1f}s")

# ══════════════════════════════════════════════════════════════════════
# STEP 3: Build labels (vectorized, multiple TP/SL configs)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: Building labels (vectorized)")
print("=" * 70)
t0 = time.time()

p_arr = c.values.astype(np.float64)
n = len(p_arr)

CONFIGS = [
    {'tp': 0.3, 'sl': 0.15, 'horizon': 60,  'name': '03_015_60'},
    {'tp': 0.3, 'sl': 0.15, 'horizon': 120, 'name': '03_015_120'},
    {'tp': 0.5, 'sl': 0.25, 'horizon': 120, 'name': '05_025_120'},
    {'tp': 0.5, 'sl': 0.25, 'horizon': 240, 'name': '05_025_240'},
    {'tp': 1.0, 'sl': 0.5,  'horizon': 240, 'name': '10_050_240'},
]

def build_labels_vectorized(prices, tp_pct, sl_pct, max_bars):
    nn = len(prices)
    long_tp = np.full(nn, 9999, dtype=np.int32)
    long_sl = np.full(nn, 9999, dtype=np.int32)
    short_tp = np.full(nn, 9999, dtype=np.int32)
    short_sl = np.full(nn, 9999, dtype=np.int32)
    tp_up = prices * (1 + tp_pct / 100)
    sl_down = prices * (1 - sl_pct / 100)
    tp_down = prices * (1 - tp_pct / 100)
    sl_up = prices * (1 + sl_pct / 100)
    for off in range(1, max_bars + 1):
        idx = np.arange(nn) + off; valid = idx < nn
        fwd = np.where(valid, prices[np.clip(idx, 0, nn-1)], np.nan)
        hit = valid & (fwd >= tp_up) & (long_tp == 9999); long_tp[hit] = off
        hit = valid & (fwd <= sl_down) & (long_sl == 9999); long_sl[hit] = off
        hit = valid & (fwd <= tp_down) & (short_tp == 9999); short_tp[hit] = off
        hit = valid & (fwd >= sl_up) & (short_sl == 9999); short_sl[hit] = off
    y_long = ((long_tp < long_sl) & (long_tp < 9999)).astype(np.int8)
    y_short = ((short_tp < short_sl) & (short_tp < 9999)).astype(np.int8)
    return y_long, y_short

all_labels = {}
for cfg in CONFIGS:
    t1 = time.time()
    yl, ys = build_labels_vectorized(p_arr, cfg['tp'], cfg['sl'], cfg['horizon'])
    all_labels[f"long_{cfg['name']}"] = yl
    all_labels[f"short_{cfg['name']}"] = ys
    dt = time.time() - t1
    print(f"  {cfg['name']}: LONG {yl.mean()*100:.1f}% | SHORT {ys.mean()*100:.1f}%  ({dt:.1f}s)")

print(f"  Total: {time.time()-t0:.1f}s")

# ══════════════════════════════════════════════════════════════════════
# STEP 4: Walk-forward with rolling 3-month train window
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4: Walk-forward (rolling 3-month train, 1-month test)")
print("=" * 70)

# Clean features
valid_mask = feat.notna().all(axis=1)
feat_vals = feat[feature_cols].fillna(0).values
pos_in_orig = np.arange(n)

# Define monthly boundaries
months = pd.date_range(df.index[0].replace(day=1), df.index[-1], freq='MS')
print(f"  {len(months)} month boundaries: {months[0].date()} → {months[-1].date()}")

TRAIN_MONTHS = 3
results = []

for test_month_idx in range(TRAIN_MONTHS, len(months)):
    test_start = months[test_month_idx]
    test_end = months[test_month_idx + 1] if test_month_idx + 1 < len(months) else df.index[-1]
    train_start = months[test_month_idx - TRAIN_MONTHS]
    train_end = test_start

    # Get indices
    tr_mask = (df.index >= train_start) & (df.index < train_end) & valid_mask
    te_mask = (df.index >= test_start) & (df.index < test_end) & valid_mask
    tr_idx = np.where(tr_mask.values)[0]
    te_idx = np.where(te_mask.values)[0]

    if len(tr_idx) < 1000 or len(te_idx) < 100:
        continue

    X_tr = feat_vals[tr_idx]
    X_te = feat_vals[te_idx]

    te_price_start = c.iloc[te_idx[0]]
    te_price_end = c.iloc[te_idx[-1]]
    te_ret = (te_price_end / te_price_start - 1) * 100

    print(f"\n  Test: {test_start.strftime('%Y-%m')} ({len(te_idx):,} bars, BTC {te_ret:+.1f}%)")
    print(f"  Train: {train_start.strftime('%Y-%m')}..{train_end.strftime('%Y-%m')} ({len(tr_idx):,} bars)")

    for cfg in CONFIGS:
        for direction in ['long', 'short']:
            key = f"{direction}_{cfg['name']}"
            y_all = all_labels[key]
            y_tr = y_all[tr_idx]
            y_te = y_all[te_idx]
            base_wr = y_te.mean() * 100
            be = cfg['sl'] / (cfg['tp'] + cfg['sl']) * 100

            m = HistGradientBoostingClassifier(
                max_iter=200, max_depth=4, min_samples_leaf=200,
                learning_rate=0.03, random_state=42, l2_regularization=1.0)
            m.fit(X_tr, y_tr)
            probs = m.predict_proba(X_te)[:, 1]

            for thr in [0.55, 0.60, 0.65, 0.70]:
                mask = probs > thr
                nt = mask.sum()
                if nt < 5:
                    continue
                w = int(y_te[mask].sum())
                wr = w / nt * 100
                ev = (wr/100 * cfg['tp']) - ((100-wr)/100 * cfg['sl'])
                results.append({
                    'test_month': test_start.strftime('%Y-%m'),
                    'direction': direction.upper(),
                    'tp': cfg['tp'], 'sl': cfg['sl'],
                    'horizon': cfg['horizon'],
                    'config': cfg['name'],
                    'threshold': thr,
                    'trades': nt, 'wins': w,
                    'wr': wr, 'ev': ev,
                    'base_wr': base_wr, 'be': be,
                    'btc_ret': te_ret,
                })

    # Print best for this month
    month_res = [r for r in results if r['test_month'] == test_start.strftime('%Y-%m') and r['ev'] > 0]
    if month_res:
        best = max(month_res, key=lambda x: x['ev'])
        print(f"    Best: {best['direction']} TP={best['tp']}/SL={best['sl']} >{best['threshold']:.0%} "
              f"→ {best['trades']}T, {best['wr']:.1f}%WR, EV={best['ev']:+.3f}%")
    else:
        print(f"    No profitable strategy this month")

# ══════════════════════════════════════════════════════════════════════
# STEP 5: Aggregate results
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 5: Aggregate walk-forward results")
print("=" * 70)

res_df = pd.DataFrame(results)
res_df.to_csv(os.path.join(OUT_DIR, 'walkforward_results.csv'), index=False)

# Group by config and threshold, compute aggregate stats
if len(res_df) > 0:
    # For each strategy config, aggregate across all months
    grouped = res_df.groupby(['direction', 'config', 'threshold']).agg(
        total_trades=('trades', 'sum'),
        total_wins=('wins', 'sum'),
        months_tested=('test_month', 'nunique'),
        months_profitable=('ev', lambda x: (x > 0).sum()),
    ).reset_index()
    grouped['wr'] = grouped['total_wins'] / grouped['total_trades'] * 100
    grouped['be'] = res_df.groupby(['direction', 'config', 'threshold'])['be'].first().values
    # Compute EV using the TP/SL from config name
    for idx, row in grouped.iterrows():
        cfg_name = row['config']
        cfg = next(c for c in CONFIGS if c['name'] == cfg_name)
        ev = (row['wr']/100 * cfg['tp']) - ((100-row['wr'])/100 * cfg['sl'])
        grouped.at[idx, 'ev'] = ev

    # Filter: profitable AND consistent (>= 50% of months profitable)
    profitable = grouped[(grouped['ev'] > 0) & (grouped['total_trades'] >= 30)]
    profitable = profitable.sort_values('ev', ascending=False)

    print(f"\nAll profitable strategies (min 30 trades total):")
    print(f"  {'Dir':>5} {'Config':>12} {'Thr':>5} | {'Trades':>6} {'WR':>6} {'EV':>7} | {'Mon':>3} {'Prof':>4}")
    print("-" * 65)
    for _, row in profitable.head(30).iterrows():
        print(f"  {row['direction']:>5} {row['config']:>12} {row['threshold']:5.0%} | "
              f"{row['total_trades']:6.0f} {row['wr']:5.1f}% {row['ev']:+6.3f}% | "
              f"{row['months_tested']:3.0f} {row['months_profitable']:4.0f}")

    # Monthly breakdown for top strategy
    if len(profitable) > 0:
        best = profitable.iloc[0]
        print(f"\n{'='*60}")
        print(f"TOP STRATEGY monthly breakdown: {best['direction']} {best['config']} >{best['threshold']:.0%}")
        print(f"{'='*60}")
        monthly = res_df[
            (res_df['direction'] == best['direction']) &
            (res_df['config'] == best['config']) &
            (res_df['threshold'] == best['threshold'])
        ].sort_values('test_month')
        for _, row in monthly.iterrows():
            status = "✓" if row['ev'] > 0 else "✗"
            print(f"  {row['test_month']}: {row['trades']:4.0f}T {row['wins']:3.0f}W "
                  f"{row['wr']:5.1f}%WR EV={row['ev']:+.3f}% BTC={row['btc_ret']:+.1f}% {status}")

    # Save best model (retrain on most recent 3 months)
    if len(profitable) > 0:
        best_row = profitable.iloc[0]
        cfg_name = best_row['config']
        cfg = next(c for c in CONFIGS if c['name'] == cfg_name)
        direction = best_row['direction'].lower()
        thr = best_row['threshold']

        print(f"\nRetraining best model on last 3 months for deployment...")
        last_3m_start = months[-TRAIN_MONTHS] if len(months) > TRAIN_MONTHS else months[0]
        tr_mask_final = (df.index >= last_3m_start) & valid_mask
        tr_idx_final = np.where(tr_mask_final.values)[0]
        X_final = feat_vals[tr_idx_final]
        y_key = f"{direction}_{cfg_name}"
        y_final = all_labels[y_key][tr_idx_final]

        best_model = HistGradientBoostingClassifier(
            max_iter=200, max_depth=4, min_samples_leaf=200,
            learning_rate=0.03, random_state=42, l2_regularization=1.0)
        best_model.fit(X_final, y_final)

        with open(os.path.join(OUT_DIR, 'best_model.pkl'), 'wb') as f:
            pickle.dump(best_model, f)
        with open(os.path.join(OUT_DIR, 'feature_cols.pkl'), 'wb') as f:
            pickle.dump(feature_cols, f)
        config = {
            'direction': best_row['direction'],
            'tp': cfg['tp'], 'sl': cfg['sl'],
            'horizon': cfg['horizon'],
            'threshold': thr,
            'wr': float(best_row['wr']),
            'ev': float(best_row['ev']),
            'trades': int(best_row['total_trades']),
            'months_tested': int(best_row['months_tested']),
            'months_profitable': int(best_row['months_profitable']),
            'train_window': f'{TRAIN_MONTHS} months',
            'train_start': str(last_3m_start.date()),
            'vol_60_median': float(feat.loc[tr_mask_final, 'vol_60'].median()),
        }
        with open(os.path.join(OUT_DIR, 'config.pkl'), 'wb') as f:
            pickle.dump(config, f)
        print(f"  Saved to {OUT_DIR}/")
        print(f"  Config: {config}")
    else:
        print("\nNo strategy passed walk-forward. Model NOT saved.")
else:
    print("\nNo results generated.")

print("\nDone!")
