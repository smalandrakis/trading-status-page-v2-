#!/usr/bin/env python3
"""
Cross-validate the SHORT night (00-08h) pattern on 5-min bar data.
Data: 97K bars, Jan 1 - Dec 5, 2025 (11 months).
Multiple train/test splits to be thorough.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Load 5-min data
# =============================================================================
print("Loading 5-min bar data...")
df = pd.read_parquet('data/BTC_2025_full_features.parquet')
close = df['Close']
high = df['High']
low = df['Low']
print(f"  Bars: {len(df)}, {df.index.min()} to {df.index.max()}")

# =============================================================================
# Compute same features as 16-sec exploration (scaled for 5-min bars)
# 16-sec: 19 bars = 5 min, 56 bars = 15 min, 225 bars = 1h, 450 bars = 2h
# 5-min:  1 bar = 5 min, 3 bars = 15 min, 12 bars = 1h, 24 bars = 2h
# =============================================================================
print("Computing features...")
ind = pd.DataFrame(index=df.index)

# Returns
for lb, label in [(1, '5m'), (3, '15m'), (12, '1h'), (24, '2h')]:
    ind[f'ret_{label}'] = close.pct_change(lb) * 100

# Volatility
ret1 = close.pct_change() * 100
for w, label in [(3, '15m'), (12, '1h')]:
    ind[f'vol_{label}'] = ret1.rolling(w).std()

# Channel position
for w, label in [(12, '1h'), (24, '2h')]:
    roll_min = close.rolling(w).min(); roll_max = close.rolling(w).max()
    ind[f'chan_{label}'] = (close - roll_min) / (roll_max - roll_min).replace(0, np.nan)

# RSI (use from parquet if available, else compute)
def compute_rsi(s, p):
    d = s.diff(); g = d.clip(lower=0).rolling(p).mean(); l = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - (100 / (1 + g / l.replace(0, np.nan)))

for p, label in [(3, '15m'), (12, '1h')]:
    ind[f'rsi_{label}'] = compute_rsi(close, p)

# Bollinger Bands
for p, label in [(12, '1h')]:
    sma = close.rolling(p).mean(); std = close.rolling(p).std()
    ind[f'bb_{label}'] = (close - (sma - 2*std)) / ((sma + 2*std) - (sma - 2*std)).replace(0, np.nan)

# ATR%
def compute_atr(h, l, c, p):
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(p).mean()

ind[f'atr_pct_1h'] = compute_atr(high, low, close, 12) / close * 100

# MACD
for (f, s, sig, label) in [(3, 12, 3, '15m')]:
    ef = close.ewm(span=f, adjust=False).mean()
    es = close.ewm(span=s, adjust=False).mean()
    ind[f'macd_{label}'] = (ef - es) - (ef - es).ewm(span=sig, adjust=False).mean()

# Speed ratio
ind['speed_ratio'] = (close.diff(1).abs()) / (close.diff(3).abs() / 3).replace(0, np.nan)

ind['hour'] = df.index.hour

# Feature sets (same as 16-sec exploration, mapped to 5-min equivalents)
C_combined = ['ret_5m','ret_15m','ret_1h','vol_15m','vol_1h','chan_1h','chan_2h',
              'rsi_15m','rsi_1h','bb_1h','macd_15m','atr_pct_1h','speed_ratio','hour']
E_minimal = ['ret_1h','ret_2h','vol_1h','chan_1h','rsi_1h','hour']

print(f"  {len(ind.columns)} features")

# =============================================================================
# Simulate targets (SL=0.30%, TS=0.30%/0.10%) on 5-min bars
# Max bars = 72 (6 hours at 5-min)
# =============================================================================
print("\nSimulating targets...")
close_arr = close.values
n = len(close_arr)
MAX_BARS = 72

lw = np.zeros(n, dtype=int)
sw = np.zeros(n, dtype=int)
lp = np.full(n, np.nan)
sp = np.full(n, np.nan)

SL, TS_ACT, TS_TRAIL = 0.30, 0.30, 0.10

for i in range(n - MAX_BARS):
    entry = close_arr[i]
    # LONG
    sl = entry * (1 - SL/100); ta = entry * (1 + TS_ACT/100)
    pk = entry; ts_on = False
    for j in range(i+1, min(i+MAX_BARS+1, n)):
        p = close_arr[j]
        if p <= sl: lp[i] = (sl/entry-1)*100; break
        if p > pk: pk = p
        if not ts_on and p >= ta: ts_on = True
        if ts_on:
            tr = pk * (1 - TS_TRAIL/100)
            if p <= tr: lp[i] = (tr/entry-1)*100; lw[i] = 1; break
    # SHORT
    sl = entry * (1 + SL/100); ta = entry * (1 - TS_ACT/100)
    pk = entry; ts_on = False
    for j in range(i+1, min(i+MAX_BARS+1, n)):
        p = close_arr[j]
        if p >= sl: sp[i] = (entry/sl-1)*100; break
        if p < pk: pk = p
        if not ts_on and p <= ta: ts_on = True
        if ts_on:
            tr = pk * (1 + TS_TRAIL/100)
            if p >= tr: sp[i] = (entry/tr-1)*100; sw[i] = 1; break

lr = (~np.isnan(lp)).sum(); sr = (~np.isnan(sp)).sum()
print(f"  LONG:  {lw.sum()}/{lr} wins ({lw.sum()/lr*100:.1f}%)")
print(f"  SHORT: {sw.sum()}/{sr} wins ({sw.sum()/sr*100:.1f}%)")

# =============================================================================
# Test on MULTIPLE time windows (rolling walk-forward)
# Train 60 days, test 30 days, slide by 30 days
# =============================================================================
print(f"\n{'='*80}")
print("WALK-FORWARD VALIDATION (train 60d, test 30d, slide 30d)")
print(f"{'='*80}")

valid = ind.dropna()
long_s = pd.Series(lw[:len(ind)], index=ind.index).reindex(valid.index).fillna(0).astype(int)
short_s = pd.Series(sw[:len(ind)], index=ind.index).reindex(valid.index).fillna(0).astype(int)
long_pnl_s = pd.Series(lp[:len(ind)], index=ind.index).reindex(valid.index)
short_pnl_s = pd.Series(sp[:len(ind)], index=ind.index).reindex(valid.index)

MIN_GAP = 2  # 10 min gap in 5-min bars

start = valid.index.min()
end = valid.index.max()

window_results = []

fold = 0
train_start = start
while True:
    train_end = train_start + pd.Timedelta(days=60)
    test_start = train_end
    test_end = test_start + pd.Timedelta(days=30)
    
    if test_end > end:
        break
    
    train_mask = np.array((valid.index >= train_start) & (valid.index < train_end))
    test_mask = np.array((valid.index >= test_start) & (valid.index < test_end))
    
    if train_mask.sum() < 1000 or test_mask.sum() < 500:
        train_start += pd.Timedelta(days=30)
        continue
    
    fold += 1
    test_indices = test_mask.nonzero()[0]
    test_data = valid[test_mask]
    test_days = (valid[test_mask].index[-1] - valid[test_mask].index[0]).total_seconds() / 86400
    hours_test = valid[test_mask].index.hour
    
    for direction in ['LONG', 'SHORT']:
        y_s = long_s if direction == 'LONG' else short_s
        pnl_s = long_pnl_s if direction == 'LONG' else short_pnl_s
        
        for feat_name, feat_cols in [('C_combined', C_combined), ('E_minimal', E_minimal)]:
            model = RandomForestClassifier(n_estimators=200, max_depth=6, 
                                          min_samples_leaf=30, random_state=42)
            model.fit(valid[train_mask][feat_cols], y_s[train_mask])
            probs = model.predict_proba(valid[test_mask][feat_cols])[:, 1]
            
            # Test configs that worked on 16-sec data
            configs = []
            
            # Pure thresholds
            for thresh in [0.55, 0.60, 0.65]:
                configs.append((f'pure>{thresh:.0%}', probs >= thresh))
            
            # Night hours 00-08
            hmask = np.array(hours_test < 8)
            for thresh in [0.50, 0.55, 0.60]:
                configs.append((f'00-08h>{thresh:.0%}', (probs >= thresh) & hmask))
            
            # 04-10h
            hmask2 = np.array((hours_test >= 4) & (hours_test < 10))
            for thresh in [0.50, 0.55, 0.60]:
                configs.append((f'04-10h>{thresh:.0%}', (probs >= thresh) & hmask2))
            
            # 08-14h
            hmask3 = np.array((hours_test >= 8) & (hours_test < 14))
            for thresh in [0.50, 0.55, 0.60]:
                configs.append((f'08-14h>{thresh:.0%}', (probs >= thresh) & hmask3))
            
            # Regime: below EMA for short, above for long
            if direction == 'SHORT':
                emask = np.array(test_data['ret_1h'] < 0)
                for thresh in [0.50, 0.55, 0.60]:
                    configs.append((f'trend_dn>{thresh:.0%}', (probs >= thresh) & emask))
            else:
                emask = np.array(test_data['ret_1h'] > 0)
                for thresh in [0.50, 0.55, 0.60]:
                    configs.append((f'trend_up>{thresh:.0%}', (probs >= thresh) & emask))
            
            for config_name, mask in configs:
                indices = np.where(mask)[0]
                taken = []; last = -MIN_GAP
                for idx in indices:
                    if idx - last >= MIN_GAP: taken.append(idx); last = idx
                
                if len(taken) < 5: continue
                
                wins = sum(int(y_s.iloc[test_indices[i]]) for i in taken)
                nt = len(taken)
                wr = wins / nt * 100
                total_pnl = sum(float(pnl_s.iloc[test_indices[i]]) for i in taken 
                               if not np.isnan(pnl_s.iloc[test_indices[i]]))
                
                window_results.append({
                    'fold': fold,
                    'test_period': f"{test_start.strftime('%b %d')}-{test_end.strftime('%b %d')}",
                    'dir': direction,
                    'feat': feat_name,
                    'config': config_name,
                    'trades': nt,
                    'wins': wins,
                    'wr': wr,
                    'pnl': total_pnl,
                    'per_day': nt / test_days,
                })
    
    train_start += pd.Timedelta(days=30)

rdf = pd.DataFrame(window_results)

# =============================================================================
# Summary: which configs work CONSISTENTLY across folds?
# =============================================================================
print(f"\n{'='*80}")
print(f"CONSISTENCY CHECK: Average WR across {fold} folds")
print(f"{'='*80}")

for direction in ['LONG', 'SHORT']:
    subset = rdf[rdf['dir'] == direction]
    
    # Group by config + feature set and average across folds
    grouped = subset.groupby(['feat', 'config']).agg(
        avg_wr=('wr', 'mean'),
        min_wr=('wr', 'min'),
        max_wr=('wr', 'max'),
        avg_pnl=('pnl', 'mean'),
        total_pnl=('pnl', 'sum'),
        folds=('fold', 'count'),
        avg_trades=('trades', 'mean'),
    ).sort_values('avg_wr', ascending=False)
    
    # Only show configs that ran in all folds
    full_folds = grouped[grouped['folds'] == fold]
    
    print(f"\n  {direction} — configs present in all {fold} folds, sorted by avg WR:")
    print(f"  {'Features':>12s} {'Config':>16s} {'Avg WR':>7s} {'Min WR':>7s} {'Max WR':>7s} "
          f"{'Avg PnL':>8s} {'Tot PnL':>8s} {'Avg T':>6s}")
    
    for (feat, config), r in full_folds.head(20).iterrows():
        print(f"  {feat:>12s} {config:>16s} {r['avg_wr']:>5.0f}% {r['min_wr']:>5.0f}% {r['max_wr']:>5.0f}% "
              f"{r['avg_pnl']:>+7.1f}% {r['total_pnl']:>+7.1f}% {r['avg_trades']:>5.0f}")

# Show per-fold details for the best configs
print(f"\n{'='*80}")
print("PER-FOLD DETAIL for top configs")
print(f"{'='*80}")

for direction in ['LONG', 'SHORT']:
    subset = rdf[rdf['dir'] == direction]
    grouped = subset.groupby(['feat', 'config']).agg(avg_wr=('wr', 'mean'), folds=('fold', 'count'))
    best = grouped[grouped['folds'] == fold].sort_values('avg_wr', ascending=False).head(3)
    
    for (feat, config), _ in best.iterrows():
        fold_data = subset[(subset['feat'] == feat) & (subset['config'] == config)]
        print(f"\n  {direction} {feat} {config}:")
        for _, r in fold_data.iterrows():
            print(f"    {r['test_period']:>14s}: {r['trades']:3.0f}t, {r['wr']:.0f}% WR, "
                  f"{r['wins']:.0f}W/{r['trades']-r['wins']:.0f}L, {r['pnl']:+.2f}%")

print(f"\n{'='*80}")
print("DONE")
print(f"{'='*80}")
