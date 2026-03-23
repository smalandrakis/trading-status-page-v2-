#!/usr/bin/env python3
"""
Re-run high WR exploration with 2 months of 16-sec data (Jan 6 - Mar 9).
Extracted from bot log: 138K price points.
Train: Jan 6 - Feb 27 (~52 days)
Test: Feb 27 - Mar 9 (~10 days) — much larger test set than before
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Load 16-sec data from bot log
# =============================================================================
print("Loading 16-sec data from bot log...")
df = pd.read_csv('logs/btc_16sec_from_log.csv', parse_dates=['timestamp'])
df = df.set_index('timestamp').sort_index()
df = df[~df.index.duplicated(keep='first')]  # Remove duplicate timestamps
df = df[df['price'] > 0]
print(f"  Rows: {len(df)}, {df.index.min()} to {df.index.max()}")

# Check for gaps
diffs = df.index.to_series().diff().dt.total_seconds()
big_gaps = diffs[diffs > 120]  # Gaps > 2 min
print(f"  Gaps >2min: {len(big_gaps)}")
if len(big_gaps) > 0:
    print(f"  Largest gap: {big_gaps.max():.0f}s at {big_gaps.idxmax()}")

# Resample to clean 16-sec OHLC (handles any irregular spacing)
bars = df['price'].resample('16s').agg(
    open='first', high='max', low='min', close='last'
).dropna()
print(f"  16-sec bars: {len(bars)}")

close = bars['close']
high = bars['high']
low = bars['low']

# =============================================================================
# Compute indicators
# =============================================================================
print("\nComputing indicators...")

def compute_rsi(s, p):
    d = s.diff(); g = d.clip(lower=0).rolling(p).mean(); l = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - (100 / (1 + g / l.replace(0, np.nan)))

def compute_atr(h, l, c, p):
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(p).mean()

ind = pd.DataFrame(index=bars.index)

# Returns
for lb, label in [(4, '1m'), (19, '5m'), (56, '15m'), (225, '1h'), (450, '2h')]:
    ind[f'ret_{label}'] = close.pct_change(lb) * 100

# Volatility
ret1 = close.pct_change() * 100
for w, label in [(19, '5m'), (56, '15m'), (225, '1h')]:
    ind[f'vol_{label}'] = ret1.rolling(w).std()

# Channel position
for w, label in [(56, '15m'), (225, '1h'), (450, '2h')]:
    roll_min = close.rolling(w).min(); roll_max = close.rolling(w).max()
    ind[f'chan_{label}'] = (close - roll_min) / (roll_max - roll_min).replace(0, np.nan)

# EMA distance
for p, label in [(19, '5m'), (56, '15m'), (225, '1h'), (450, '2h')]:
    ema = close.ewm(span=p, adjust=False).mean()
    ind[f'ema_dist_{label}'] = (close - ema) / close * 100

# RSI
for p, label in [(56, '15m'), (225, '1h')]:
    ind[f'rsi_{label}'] = compute_rsi(close, p)

# Bollinger Bands
for p, label in [(56, '15m'), (225, '1h')]:
    sma = close.rolling(p).mean(); std = close.rolling(p).std()
    ind[f'bb_{label}'] = (close - (sma - 2*std)) / ((sma + 2*std) - (sma - 2*std)).replace(0, np.nan)

# ATR
for p, label in [(56, '15m'), (225, '1h')]:
    ind[f'atr_pct_{label}'] = compute_atr(high, low, close, p) / close * 100

# MACD
for (f, s, sig, label) in [(19, 56, 14, '5m'), (56, 225, 38, '15m')]:
    ef = close.ewm(span=f, adjust=False).mean()
    es = close.ewm(span=s, adjust=False).mean()
    ind[f'macd_{label}'] = (ef - es) - (ef - es).ewm(span=sig, adjust=False).mean()

# Speed
ind['speed_ratio'] = (close.diff(19).abs() / 19) / (close.diff(56).abs() / 56).replace(0, np.nan)

# Consecutive direction
ret_sign = np.sign(close.diff())
ind['consec'] = ret_sign.groupby((ret_sign != ret_sign.shift()).cumsum()).cumcount() + 1
ind['consec'] *= ret_sign

ind['hour'] = bars.index.hour

print(f"  {len(ind.columns)} features")

# =============================================================================
# Simulate targets (SL=0.30%, TS=0.30%/0.10%)
# =============================================================================
print("\nSimulating targets (SL=0.30%, TS=0.30%/0.10%)...")

close_arr = close.values
n = len(close_arr)
lw = np.zeros(n, dtype=int)
sw = np.zeros(n, dtype=int)
lp = np.full(n, np.nan)
sp = np.full(n, np.nan)

SL, TS_ACT, TS_TRAIL, MAX_BARS = 0.30, 0.30, 0.10, 675

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
# Train/test split: Jan 6 - Feb 27 train, Feb 27+ test
# =============================================================================
valid = ind.dropna()
split_date = pd.Timestamp('2026-02-27')
train_mask = np.array(valid.index < split_date)
test_mask = np.array(valid.index >= split_date)

long_s = pd.Series(lw[:len(ind)], index=ind.index).reindex(valid.index).fillna(0).astype(int)
short_s = pd.Series(sw[:len(ind)], index=ind.index).reindex(valid.index).fillna(0).astype(int)
long_pnl_s = pd.Series(lp[:len(ind)], index=ind.index).reindex(valid.index)
short_pnl_s = pd.Series(sp[:len(ind)], index=ind.index).reindex(valid.index)

train_days = (valid[train_mask].index[-1] - valid[train_mask].index[0]).total_seconds() / 86400
test_days = (valid[test_mask].index[-1] - valid[test_mask].index[0]).total_seconds() / 86400

print(f"\nTrain: {train_mask.sum()} bars ({train_days:.0f} days)")
print(f"Test:  {test_mask.sum()} bars ({test_days:.0f} days)")
print(f"Train LONG WR: {long_s[train_mask].mean()*100:.1f}%, SHORT WR: {short_s[train_mask].mean()*100:.1f}%")
print(f"Test  LONG WR: {long_s[test_mask].mean()*100:.1f}%, SHORT WR: {short_s[test_mask].mean()*100:.1f}%")

# =============================================================================
# Feature sets
# =============================================================================
FEAT_SETS = {
    'C_combined': ['ret_5m','ret_15m','ret_1h','vol_15m','vol_1h','chan_1h','chan_2h',
                    'rsi_15m','rsi_1h','bb_1h','macd_15m','atr_pct_1h','speed_ratio','hour'],
    'D_momentum': ['ret_5m','ret_15m','ret_1h','ret_2h','ema_dist_5m','ema_dist_15m',
                    'ema_dist_1h','ema_dist_2h','speed_ratio','vol_1h','chan_1h','hour'],
    'E_minimal': ['ret_1h','ret_2h','vol_1h','chan_1h','rsi_1h','hour'],
}

MIN_GAP = 30  # ~8 min between signals
test_indices = test_mask.nonzero()[0]

# =============================================================================
# Run all 4 strategies
# =============================================================================
def run_backtest(probs, y_s, pnl_s, extra_mask=None):
    """Test multiple thresholds, return list of results."""
    results = []
    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        mask = probs >= thresh
        if extra_mask is not None:
            mask = mask & extra_mask
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
        results.append({'thresh': thresh, 'trades': nt, 'wins': wins, 'wr': wr,
                        'pnl': total_pnl, 'per_day': nt / test_days})
    return results

all_results = []

for direction in ['LONG', 'SHORT']:
    y_s = long_s if direction == 'LONG' else short_s
    pnl_s = long_pnl_s if direction == 'LONG' else short_pnl_s
    test_data = valid[test_mask]
    hours_test = valid[test_mask].index.hour
    
    for feat_name, feat_cols in FEAT_SETS.items():
        model = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=30, random_state=42)
        model.fit(valid[train_mask][feat_cols], y_s[train_mask])
        probs = model.predict_proba(valid[test_mask][feat_cols])[:, 1]
        
        # Strategy 1: Pure threshold
        for r in run_backtest(probs, y_s, pnl_s):
            all_results.append({**r, 'dir': direction, 'feat': feat_name, 'strategy': f'pure>{r["thresh"]:.0%}'})
        
        # Strategy 3: Time filters
        for hours_label, hour_filter in [
            ('08-14h', lambda h: (h >= 8) & (h < 14)),
            ('04-10h', lambda h: (h >= 4) & (h < 10)),
            ('00-08h', lambda h: h < 8),
            ('00-12h', lambda h: h < 12),
        ]:
            hmask = np.array(hour_filter(hours_test))
            for r in run_backtest(probs, y_s, pnl_s, extra_mask=hmask):
                all_results.append({**r, 'dir': direction, 'feat': feat_name, 
                                   'strategy': f'{hours_label}>{r["thresh"]:.0%}'})
        
        # Strategy 4: Regime confluence
        if direction == 'LONG':
            regime_filters = {
                'trend_up_1h': np.array(test_data['ret_1h'] > 0),
                'above_ema_1h': np.array(test_data['ema_dist_1h'] > 0),
                'rsi>50': np.array(test_data['rsi_1h'] > 50),
                'trend+ema': np.array((test_data['ret_1h'] > 0) & (test_data['ema_dist_1h'] > 0)),
                'trend+ema+rsi': np.array((test_data['ret_1h'] > 0) & (test_data['ema_dist_1h'] > 0) & (test_data['rsi_1h'] > 50)),
                'strong_2h': np.array(test_data['ret_2h'] > 0.15),
            }
        else:
            regime_filters = {
                'trend_dn_1h': np.array(test_data['ret_1h'] < 0),
                'below_ema_1h': np.array(test_data['ema_dist_1h'] < 0),
                'rsi<45': np.array(test_data['rsi_1h'] < 45),
                'trend+ema': np.array((test_data['ret_1h'] < 0) & (test_data['ema_dist_1h'] < 0)),
                'trend+ema+rsi': np.array((test_data['ret_1h'] < 0) & (test_data['ema_dist_1h'] < 0) & (test_data['rsi_1h'] < 50)),
                'strong_2h': np.array(test_data['ret_2h'] < -0.15),
            }
        
        for filter_name, fmask in regime_filters.items():
            for r in run_backtest(probs, y_s, pnl_s, extra_mask=fmask):
                all_results.append({**r, 'dir': direction, 'feat': feat_name,
                                   'strategy': f'{filter_name}>{r["thresh"]:.0%}'})

    # Strategy 2: Ensemble agreement
    models = {}
    for feat_name in FEAT_SETS:
        feat_cols = FEAT_SETS[feat_name]
        m = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=30, random_state=42)
        m.fit(valid[train_mask][feat_cols], y_s[train_mask])
        models[feat_name] = m.predict_proba(valid[test_mask][feat_cols])[:, 1]
    
    for min_agree in [2, 3]:
        for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
            agree = np.zeros(test_mask.sum())
            for p in models.values():
                agree += (p >= thresh).astype(int)
            mask = agree >= min_agree
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
            all_results.append({'thresh': thresh, 'trades': nt, 'wins': wins, 'wr': wr,
                               'pnl': total_pnl, 'per_day': nt / test_days,
                               'dir': direction, 'feat': f'{min_agree}/3_ensemble',
                               'strategy': f'{min_agree}/3>{thresh:.0%}'})

# =============================================================================
# Report: filter for WR >= 65%, sort by PnL
# =============================================================================
rdf = pd.DataFrame(all_results)

print(f"\n{'='*90}")
print(f"RESULTS WITH WR >= 65% (trained Jan-Feb, tested Feb 27 - Mar 9, {test_days:.0f} days)")
print(f"{'='*90}")

for direction in ['LONG', 'SHORT']:
    subset = rdf[(rdf['dir'] == direction) & (rdf['wr'] >= 65)].sort_values('pnl', ascending=False)
    
    print(f"\n{'─'*90}")
    print(f"  {direction} — {len(subset)} configs with WR >= 65%")
    print(f"{'─'*90}")
    
    if len(subset) == 0:
        print("  None found")
        continue
    
    print(f"  {'Features':>14s} {'Strategy':>25s} {'Trades':>7s} {'T/day':>6s} "
          f"{'WR':>5s} {'W/L':>8s} {'PnL%':>8s}")
    
    for _, r in subset.head(25).iterrows():
        print(f"  {r['feat']:>14s} {r['strategy']:>25s} {r['trades']:>7.0f} {r['per_day']:>5.1f} "
              f"{r['wr']:>4.0f}% {r['wins']:.0f}W/{r['trades']-r['wins']:.0f}L "
              f"{r['pnl']:>+7.2f}%")

# Also show 70%+ WR
print(f"\n{'='*90}")
print(f"RESULTS WITH WR >= 70%")
print(f"{'='*90}")

for direction in ['LONG', 'SHORT']:
    subset = rdf[(rdf['dir'] == direction) & (rdf['wr'] >= 70)].sort_values('pnl', ascending=False)
    
    print(f"\n  {direction} — {len(subset)} configs with WR >= 70%:")
    if len(subset) == 0:
        print("  None found")
        continue
    
    for _, r in subset.head(15).iterrows():
        print(f"  {r['feat']:>14s} {r['strategy']:>25s} {r['trades']:>4.0f}t ({r['per_day']:.1f}/d) "
              f"{r['wr']:.0f}% {r['wins']:.0f}W/{r['trades']-r['wins']:.0f}L "
              f"{r['pnl']:>+7.2f}%")

# Show 80%+
print(f"\n{'='*90}")
print(f"RESULTS WITH WR >= 80%")
print(f"{'='*90}")

for direction in ['LONG', 'SHORT']:
    subset = rdf[(rdf['dir'] == direction) & (rdf['wr'] >= 80)].sort_values('pnl', ascending=False)
    
    print(f"\n  {direction} — {len(subset)} configs:")
    if len(subset) == 0:
        print("  None found")
        continue
    
    for _, r in subset.iterrows():
        print(f"  {r['feat']:>14s} {r['strategy']:>25s} {r['trades']:>4.0f}t ({r['per_day']:.1f}/d) "
              f"{r['wr']:.0f}% {r['wins']:.0f}W/{r['trades']-r['wins']:.0f}L "
              f"{r['pnl']:>+7.2f}%")

print(f"\n{'='*90}")
print("EXPLORATION COMPLETE")
print(f"{'='*90}")
