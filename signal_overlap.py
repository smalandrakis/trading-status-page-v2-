#!/usr/bin/env python3
"""
Analyze signal overlap between top models from 2-sec tick data.
Check: do they fire at the same time or different times?
Can we deploy all of them for more trades?
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. LOAD DATA + FEATURES (same as ml_2sec_signals.py)
# =============================================================================
print("Loading data and computing features...")

ticks = pd.read_csv('logs/btc_price_ticks.csv', parse_dates=['timestamp'])
ticks = ticks.set_index('timestamp').sort_index()
ticks = ticks[ticks['price'] > 0]
price = ticks['price']

bars = price.resample('16s').agg(
    **{'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
).dropna()

close = bars['close']
high = bars['high']
low = bars['low']

def compute_rsi(s, p):
    d = s.diff()
    g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - (100 / (1 + g / l.replace(0, np.nan)))

def compute_atr(h, l, c, p):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(p).mean()

feat = pd.DataFrame(index=bars.index)

for lb, lbl in [(4, '1m'), (8, '2m'), (19, '5m'), (38, '10m'), (56, '15m'), (113, '30m'), (225, '1h')]:
    feat['ret_' + lbl] = close.pct_change(lb) * 100

ret1 = close.pct_change() * 100
for w, lbl in [(19, '5m'), (56, '15m'), (113, '30m'), (225, '1h')]:
    feat['vol_' + lbl] = ret1.rolling(w).std()

feat['vol_ratio_5m_1h'] = feat['vol_5m'] / feat['vol_1h'].replace(0, np.nan)
feat['vol_ratio_15m_1h'] = feat['vol_15m'] / feat['vol_1h'].replace(0, np.nan)

for w, lbl in [(56, '15m'), (225, '1h'), (450, '2h')]:
    rmin = close.rolling(w).min()
    rmax = close.rolling(w).max()
    rng = (rmax - rmin).replace(0, np.nan)
    feat['chan_' + lbl] = (close - rmin) / rng

for p, lbl in [(19, '5m'), (56, '15m'), (225, '1h')]:
    ema = close.ewm(span=p, adjust=False).mean()
    feat['ema_dist_' + lbl] = (close - ema) / close * 100

for p, lbl in [(56, '15m'), (225, '1h')]:
    feat['rsi_' + lbl] = compute_rsi(close, p)

for p, lbl in [(56, '15m'), (225, '1h')]:
    sma = close.rolling(p).mean()
    std = close.rolling(p).std()
    lower = sma - 2 * std
    upper = sma + 2 * std
    feat['bb_' + lbl] = (close - lower) / (upper - lower).replace(0, np.nan)

for (f, s, sig, lbl) in [(19, 56, 14, '5m_15m'), (56, 225, 38, '15m_1h')]:
    ef = close.ewm(span=f, adjust=False).mean()
    es = close.ewm(span=s, adjust=False).mean()
    macd_line = ef - es
    signal_line = macd_line.ewm(span=sig, adjust=False).mean()
    feat['macd_hist_' + lbl] = macd_line - signal_line

for p, lbl in [(56, '15m'), (225, '1h')]:
    feat['atr_pct_' + lbl] = compute_atr(high, low, close, p) / close * 100

feat['speed_5m_15m'] = (close.diff(19).abs() / 19) / (close.diff(56).abs() / 56).replace(0, np.nan)

ret_sign = np.sign(close.diff())
feat['consec'] = ret_sign.groupby((ret_sign != ret_sign.shift()).cumsum()).cumcount() + 1
feat['consec'] = feat['consec'] * ret_sign

feat['bar_range_pct'] = (high - low) / close * 100
feat['bar_range_ma'] = feat['bar_range_pct'].rolling(56).mean()
feat['bar_range_ratio'] = feat['bar_range_pct'] / feat['bar_range_ma'].replace(0, np.nan)

up_ticks = (close > close.shift(1)).astype(int)
down_ticks = (close < close.shift(1)).astype(int)
for w, lbl in [(19, '5m'), (56, '15m')]:
    feat['tick_imbal_' + lbl] = (up_ticks.rolling(w).sum() - down_ticks.rolling(w).sum()) / w

feat['hour'] = bars.index.hour

# =============================================================================
# 2. LABELS
# =============================================================================
print("Computing labels...")

close_arr = close.values
n = len(close_arr)
MAX_BARS = 675

def simulate_labels(close_arr, n, sl_pct, ts_act_pct, ts_trail_pct, direction):
    labels = np.zeros(n, dtype=int)
    for i in range(n - MAX_BARS):
        entry = close_arr[i]
        if direction == 'LONG':
            sl = entry * (1 - sl_pct / 100)
            act = entry * (1 + ts_act_pct / 100)
            peak = entry
            ts_on = False
            for j in range(i + 1, min(i + MAX_BARS + 1, n)):
                p = close_arr[j]
                if p <= sl: break
                if p > peak: peak = p
                if not ts_on and p >= act: ts_on = True
                if ts_on:
                    trail = peak * (1 - ts_trail_pct / 100)
                    if p <= trail: labels[i] = 1; break
        else:
            sl = entry * (1 + sl_pct / 100)
            act = entry * (1 - ts_act_pct / 100)
            peak = entry
            ts_on = False
            for j in range(i + 1, min(i + MAX_BARS + 1, n)):
                p = close_arr[j]
                if p >= sl: break
                if p < peak: peak = p
                if not ts_on and p <= act: ts_on = True
                if ts_on:
                    trail = peak * (1 + ts_trail_pct / 100)
                    if p >= trail: labels[i] = 1; break
    return labels

# The top configs we want to compare
configs = {
    'L1_full_wide':    {'sl': 0.30, 'ts_act': 0.40, 'ts_trail': 0.10, 'dir': 'LONG',  'feats': 'full',     'thresh': 0.60},
    'L2_full_med':     {'sl': 0.25, 'ts_act': 0.30, 'ts_trail': 0.08, 'dir': 'LONG',  'feats': 'full',     'thresh': 0.65},
    'L3_mom_wide':     {'sl': 0.30, 'ts_act': 0.40, 'ts_trail': 0.10, 'dir': 'LONG',  'feats': 'momentum', 'thresh': 0.60},
    'L4_full_cur':     {'sl': 0.20, 'ts_act': 0.25, 'ts_trail': 0.05, 'dir': 'LONG',  'feats': 'full',     'thresh': 0.70},
    'S1_full_cur':     {'sl': 0.20, 'ts_act': 0.25, 'ts_trail': 0.05, 'dir': 'SHORT', 'feats': 'full',     'thresh': 0.70},
    'S2_micro_tight':  {'sl': 0.15, 'ts_act': 0.20, 'ts_trail': 0.05, 'dir': 'SHORT', 'feats': 'micro',    'thresh': 0.70},
    'S3_mr_cur':       {'sl': 0.20, 'ts_act': 0.25, 'ts_trail': 0.05, 'dir': 'SHORT', 'feats': 'mean_rev', 'thresh': 0.70},
    'S4_full_tight':   {'sl': 0.15, 'ts_act': 0.20, 'ts_trail': 0.05, 'dir': 'SHORT', 'feats': 'full',     'thresh': 0.70},
}

feature_sets = {
    'full': list(feat.columns),
    'momentum': ['ret_1m', 'ret_2m', 'ret_5m', 'ret_10m', 'ret_15m', 'ret_30m', 'ret_1h',
                  'vol_5m', 'vol_15m', 'vol_1h', 'vol_ratio_5m_1h',
                  'ema_dist_5m', 'ema_dist_15m', 'ema_dist_1h', 'speed_5m_15m', 'consec'],
    'mean_rev': ['rsi_15m', 'rsi_1h', 'bb_15m', 'bb_1h', 'chan_15m', 'chan_1h', 'chan_2h',
                 'ema_dist_15m', 'ema_dist_1h', 'vol_ratio_5m_1h', 'bar_range_ratio'],
    'micro': ['ret_1m', 'ret_2m', 'ret_5m', 'vol_5m', 'vol_ratio_5m_1h',
              'tick_imbal_5m', 'tick_imbal_15m', 'speed_5m_15m', 'consec',
              'bar_range_ratio', 'ema_dist_5m'],
}

valid = feat.dropna()
split_idx = int(len(valid) * 0.70)
split_time = valid.index[split_idx]
train_mask = valid.index < split_time
test_mask = valid.index >= split_time

print("Train: %d, Test: %d" % (train_mask.sum(), test_mask.sum()))

# =============================================================================
# 3. TRAIN ALL MODELS AND GET TEST SIGNALS
# =============================================================================
print("\nTraining models and collecting test signals...")

label_cache = {}
signal_arrays = {}

for name, cfg in configs.items():
    direction = cfg['dir']
    sl_key = "%.2f_%.2f_%.2f_%s" % (cfg['sl'], cfg['ts_act'], cfg['ts_trail'], direction)
    
    if sl_key not in label_cache:
        print("  Simulating labels: %s" % sl_key)
        lab = simulate_labels(close_arr, n, cfg['sl'], cfg['ts_act'], cfg['ts_trail'], direction)
        label_cache[sl_key] = pd.Series(lab, index=bars.index)
    
    y = label_cache[sl_key].reindex(valid.index)
    cols = [c for c in feature_sets[cfg['feats']] if c in valid.columns]
    
    model = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=30, random_state=42)
    model.fit(valid[train_mask][cols], y[train_mask])
    
    probs = model.predict_proba(valid[test_mask][cols])[:, 1]
    signals = probs >= cfg['thresh']
    
    y_test = y[test_mask]
    n_sig = signals.sum()
    wins = y_test[signals].sum() if n_sig > 0 else 0
    wr = wins / n_sig * 100 if n_sig > 0 else 0
    
    signal_arrays[name] = signals
    
    avg_win = cfg['ts_act'] - cfg['ts_trail']
    avg_loss = cfg['sl']
    ev = wr/100 * avg_win - (100 - wr)/100 * avg_loss if n_sig > 0 else 0
    
    print("  %s: %dW/%dL (%.1f%% WR), EV=%.4f%%, n=%d" % (
        name, wins, n_sig - wins, wr, ev, n_sig))

# =============================================================================
# 4. OVERLAP ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("SIGNAL OVERLAP ANALYSIS")
print("=" * 70)

test_index = valid.index[test_mask]
names = list(configs.keys())

# Convert to DataFrames
sig_df = pd.DataFrame({n: signal_arrays[n] for n in names}, index=range(test_mask.sum()))

# Pairwise overlap
print("\n--- PAIRWISE OVERLAP (% of signals shared) ---")
print("%18s" % "", end="")
for n in names:
    print(" %6s" % n[:6], end="")
print()

for n1 in names:
    s1 = sig_df[n1]
    print("%18s" % n1, end="")
    for n2 in names:
        s2 = sig_df[n2]
        both = (s1 & s2).sum()
        either = max(s1.sum(), 1)
        pct = both / either * 100
        print(" %5.0f%%" % pct, end="")
    print()

# LONG vs LONG overlap
print("\n--- LONG MODEL OVERLAP ---")
long_names = [n for n in names if n.startswith('L')]
for i, n1 in enumerate(long_names):
    for n2 in long_names[i+1:]:
        s1 = sig_df[n1]
        s2 = sig_df[n2]
        both = (s1 & s2).sum()
        only1 = (s1 & ~s2).sum()
        only2 = (~s1 & s2).sum()
        total = (s1 | s2).sum()
        print("  %s vs %s: %d shared, %d only-%s, %d only-%s, %d total unique signals" % (
            n1, n2, both, only1, n1[:4], only2, n2[:4], total))

print("\n--- SHORT MODEL OVERLAP ---")
short_names = [n for n in names if n.startswith('S')]
for i, n1 in enumerate(short_names):
    for n2 in short_names[i+1:]:
        s1 = sig_df[n1]
        s2 = sig_df[n2]
        both = (s1 & s2).sum()
        only1 = (s1 & ~s2).sum()
        only2 = (~s1 & s2).sum()
        total = (s1 | s2).sum()
        print("  %s vs %s: %d shared, %d only-%s, %d only-%s, %d total unique signals" % (
            n1, n2, both, only1, n1[:4], only2, n2[:4], total))

# =============================================================================
# 5. COMBINED DEPLOYMENT ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("COMBINED DEPLOYMENT ANALYSIS")
print("=" * 70)

# Any LONG signal
any_long = sig_df[long_names].any(axis=1)
all_long = sig_df[long_names].all(axis=1)  # consensus
majority_long = sig_df[long_names].sum(axis=1) >= 2

# Any SHORT signal
any_short = sig_df[short_names].any(axis=1)
all_short = sig_df[short_names].all(axis=1)
majority_short = sig_df[short_names].sum(axis=1) >= 2

# For WR, use the best label (current params for simplicity)
y_long = label_cache.get("0.30_0.40_0.10_LONG", label_cache.get("0.20_0.25_0.05_LONG"))
y_short = label_cache.get("0.20_0.25_0.05_SHORT", label_cache.get("0.15_0.20_0.05_SHORT"))

if y_long is not None:
    y_long_test = y_long.reindex(valid.index)[test_mask].values
if y_short is not None:
    y_short_test = y_short.reindex(valid.index)[test_mask].values

print("\nLONG strategies:")
for label, mask in [("Any model fires", any_long), ("Majority (2+)", majority_long), ("All agree", all_long)]:
    n_sig = mask.sum()
    if n_sig > 0 and y_long is not None:
        wins = y_long_test[mask.values].sum()
        wr = wins / n_sig * 100
        print("  %-20s: %3dW/%3dL (%5.1f%% WR), n=%d" % (label, wins, n_sig - wins, wr, n_sig))
    else:
        print("  %-20s: n=%d" % (label, n_sig))

print("\nSHORT strategies:")
for label, mask in [("Any model fires", any_short), ("Majority (2+)", majority_short), ("All agree", all_short)]:
    n_sig = mask.sum()
    if n_sig > 0 and y_short is not None:
        wins = y_short_test[mask.values].sum()
        wr = wins / n_sig * 100
        print("  %-20s: %3dW/%3dL (%5.1f%% WR), n=%d" % (label, wins, n_sig - wins, wr, n_sig))
    else:
        print("  %-20s: n=%d" % (label, n_sig))

# =============================================================================
# 6. TEMPORAL DISTRIBUTION — how spread out are signals?
# =============================================================================
print("\n" + "=" * 70)
print("TEMPORAL DISTRIBUTION")
print("=" * 70)

test_times = valid.index[test_mask]
for name in names:
    sig = signal_arrays[name]
    sig_times = test_times[sig]
    if len(sig_times) > 1:
        gaps = sig_times.to_series().diff().dropna()
        med_gap = gaps.median().total_seconds() / 60
        print("  %s: %d signals, median gap=%.0f min, min=%.0f min, max=%.0f min" % (
            name, len(sig_times), med_gap,
            gaps.min().total_seconds() / 60,
            gaps.max().total_seconds() / 60))

# Combined
any_signal = sig_df.any(axis=1)
sig_times = test_times[any_signal.values]
if len(sig_times) > 1:
    gaps = sig_times.to_series().diff().dropna()
    med_gap = gaps.median().total_seconds() / 60
    print("\n  ALL COMBINED: %d signals, median gap=%.0f min" % (len(sig_times), med_gap))

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
