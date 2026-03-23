#!/usr/bin/env python3
"""
Build ML models on 2-sec tick data to find quality LONG and SHORT signals.
Uses only tick-derived features. Proper temporal train/test split.
Tests multiple SL/TS configurations for best R:R.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. LOAD AND CLEAN TICK DATA
# =============================================================================
print("=" * 70)
print("STEP 1: Loading 2-sec tick data")
print("=" * 70)

ticks = pd.read_csv('logs/btc_price_ticks.csv', parse_dates=['timestamp'])
ticks = ticks.set_index('timestamp').sort_index()
ticks = ticks[ticks['price'] > 0]

# Remove large gaps (>10 min) — mark them so we don't compute features across gaps
gaps = ticks.index.to_series().diff()
gap_mask = gaps > pd.Timedelta(minutes=10)
print("Total ticks: %d" % len(ticks))
print("Gaps >10min: %d" % gap_mask.sum())
print("Date range: %s to %s" % (ticks.index.min(), ticks.index.max()))

# Resample to clean 2-sec bars
price = ticks['price']

# =============================================================================
# 2. COMPUTE FEATURES FROM RAW TICKS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: Computing features from 2-sec ticks")
print("=" * 70)

# Work with 16-sec bars (8 ticks) for features — matches bot's timeframe
bars = price.resample('16s').agg(
    **{'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
).dropna()
print("16-sec bars: %d" % len(bars))

close = bars['close']
high = bars['high']
low = bars['low']
opn = bars['open']

# Mark bars that span a data gap
bar_gaps = bars.index.to_series().diff() > pd.Timedelta(minutes=10)

feat = pd.DataFrame(index=bars.index)

# --- Price returns at multiple scales ---
for lb, lbl in [(4, '1m'), (8, '2m'), (19, '5m'), (38, '10m'), (56, '15m'), (113, '30m'), (225, '1h')]:
    feat['ret_' + lbl] = close.pct_change(lb) * 100

# --- Volatility (rolling std of returns) ---
ret1 = close.pct_change() * 100
for w, lbl in [(19, '5m'), (56, '15m'), (113, '30m'), (225, '1h')]:
    feat['vol_' + lbl] = ret1.rolling(w).std()

# --- Realized volatility ratio (short/long) ---
feat['vol_ratio_5m_1h'] = feat['vol_5m'] / feat['vol_1h'].replace(0, np.nan)
feat['vol_ratio_15m_1h'] = feat['vol_15m'] / feat['vol_1h'].replace(0, np.nan)

# --- Channel position (where price is in range) ---
for w, lbl in [(56, '15m'), (225, '1h'), (450, '2h')]:
    rmin = close.rolling(w).min()
    rmax = close.rolling(w).max()
    rng = (rmax - rmin).replace(0, np.nan)
    feat['chan_' + lbl] = (close - rmin) / rng

# --- EMA distance ---
for p, lbl in [(19, '5m'), (56, '15m'), (225, '1h')]:
    ema = close.ewm(span=p, adjust=False).mean()
    feat['ema_dist_' + lbl] = (close - ema) / close * 100

# --- RSI ---
def compute_rsi(s, p):
    d = s.diff()
    g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - (100 / (1 + g / l.replace(0, np.nan)))

for p, lbl in [(56, '15m'), (225, '1h')]:
    feat['rsi_' + lbl] = compute_rsi(close, p)

# --- Bollinger Band % ---
for p, lbl in [(56, '15m'), (225, '1h')]:
    sma = close.rolling(p).mean()
    std = close.rolling(p).std()
    lower = sma - 2 * std
    upper = sma + 2 * std
    feat['bb_' + lbl] = (close - lower) / (upper - lower).replace(0, np.nan)

# --- MACD histogram ---
for (f, s, sig, lbl) in [(19, 56, 14, '5m_15m'), (56, 225, 38, '15m_1h')]:
    ef = close.ewm(span=f, adjust=False).mean()
    es = close.ewm(span=s, adjust=False).mean()
    macd_line = ef - es
    signal_line = macd_line.ewm(span=sig, adjust=False).mean()
    feat['macd_hist_' + lbl] = macd_line - signal_line

# --- ATR % ---
def compute_atr(h, l, c, p):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(p).mean()

for p, lbl in [(56, '15m'), (225, '1h')]:
    feat['atr_pct_' + lbl] = compute_atr(high, low, close, p) / close * 100

# --- Speed (magnitude of recent move vs longer move) ---
feat['speed_5m_15m'] = (close.diff(19).abs() / 19) / (close.diff(56).abs() / 56).replace(0, np.nan)

# --- Consecutive direction ---
ret_sign = np.sign(close.diff())
feat['consec'] = ret_sign.groupby((ret_sign != ret_sign.shift()).cumsum()).cumcount() + 1
feat['consec'] = feat['consec'] * ret_sign

# --- Bar range (high-low as % of close) ---
feat['bar_range_pct'] = (high - low) / close * 100
feat['bar_range_ma'] = feat['bar_range_pct'].rolling(56).mean()
feat['bar_range_ratio'] = feat['bar_range_pct'] / feat['bar_range_ma'].replace(0, np.nan)

# --- Microstructure: tick imbalance ---
up_ticks = (close > close.shift(1)).astype(int)
down_ticks = (close < close.shift(1)).astype(int)
for w, lbl in [(19, '5m'), (56, '15m')]:
    feat['tick_imbal_' + lbl] = (up_ticks.rolling(w).sum() - down_ticks.rolling(w).sum()) / w

# --- Hour of day ---
feat['hour'] = bars.index.hour

print("Features computed: %d" % len(feat.columns))
print("Features: %s" % ', '.join(feat.columns))

# =============================================================================
# 3. CREATE LABELS — simulate actual SL/TS outcomes
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: Creating trade outcome labels")
print("=" * 70)

# Test multiple SL/TS configs
configs = [
    {'name': 'tight',   'sl': 0.15, 'ts_act': 0.20, 'ts_trail': 0.05},
    {'name': 'current', 'sl': 0.20, 'ts_act': 0.25, 'ts_trail': 0.05},
    {'name': 'medium',  'sl': 0.25, 'ts_act': 0.30, 'ts_trail': 0.08},
    {'name': 'wide',    'sl': 0.30, 'ts_act': 0.40, 'ts_trail': 0.10},
]

close_arr = close.values
n = len(close_arr)
MAX_BARS = 675  # ~3 hours max hold

def simulate_labels(close_arr, n, sl_pct, ts_act_pct, ts_trail_pct, direction):
    """Returns array of 1 (win via trailing stop) or 0 (loss via stop loss or timeout)."""
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
                if p <= sl:
                    break  # SL hit = loss
                if p > peak:
                    peak = p
                if not ts_on and p >= act:
                    ts_on = True
                if ts_on:
                    trail = peak * (1 - ts_trail_pct / 100)
                    if p <= trail:
                        labels[i] = 1  # TS win
                        break
        else:  # SHORT
            sl = entry * (1 + sl_pct / 100)
            act = entry * (1 - ts_act_pct / 100)
            peak = entry
            ts_on = False
            for j in range(i + 1, min(i + MAX_BARS + 1, n)):
                p = close_arr[j]
                if p >= sl:
                    break
                if p < peak:
                    peak = p
                if not ts_on and p <= act:
                    ts_on = True
                if ts_on:
                    trail = peak * (1 + ts_trail_pct / 100)
                    if p >= trail:
                        labels[i] = 1
                        break
    return labels

# Compute labels for each config + direction
all_labels = {}
for cfg in configs:
    for direction in ['LONG', 'SHORT']:
        key = "%s_%s" % (cfg['name'], direction)
        print("  Simulating %s..." % key)
        labels = simulate_labels(close_arr, n, cfg['sl'], cfg['ts_act'], cfg['ts_trail'], direction)
        all_labels[key] = pd.Series(labels, index=bars.index)
        wr = labels.sum() / max((labels >= 0).sum(), 1) * 100
        # Compute R:R
        avg_win = cfg['ts_act'] - cfg['ts_trail']  # approximate avg win
        avg_loss = cfg['sl']
        rr = avg_win / avg_loss if avg_loss > 0 else 0
        print("    Base WR: %.1f%%, R:R ~%.2f, EV/trade: %.3f%%" % (
            wr, rr, wr/100 * avg_win - (100-wr)/100 * avg_loss))

# =============================================================================
# 4. TRAIN MODELS — temporal split
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4: Training models with temporal split")
print("=" * 70)

# Drop NaN rows
valid = feat.dropna()
print("Valid bars (no NaN): %d" % len(valid))

# Temporal split: first 70% train, last 30% test
split_idx = int(len(valid) * 0.70)
split_time = valid.index[split_idx]
train_mask = valid.index < split_time
test_mask = valid.index >= split_time

print("Train: %d bars (%s to %s)" % (train_mask.sum(), valid.index[0], split_time))
print("Test:  %d bars (%s to %s)" % (test_mask.sum(), split_time, valid.index[-1]))

# Feature sets
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

results = []

for cfg in configs:
    for direction in ['LONG', 'SHORT']:
        label_key = "%s_%s" % (cfg['name'], direction)
        y = all_labels[label_key].reindex(valid.index)
        
        y_train = y[train_mask]
        y_test = y[test_mask]
        
        base_wr_test = y_test.mean() * 100
        
        for fs_name, fs_cols in feature_sets.items():
            cols = [c for c in fs_cols if c in valid.columns]
            X_train = valid[train_mask][cols]
            X_test = valid[test_mask][cols]
            
            # Try GBM and RF
            for model_name, model in [
                ('GBM', GradientBoostingClassifier(n_estimators=200, max_depth=4, 
                    min_samples_leaf=50, learning_rate=0.05, random_state=42)),
                ('RF', RandomForestClassifier(n_estimators=200, max_depth=6,
                    min_samples_leaf=30, random_state=42)),
            ]:
                model.fit(X_train, y_train)
                probs = model.predict_proba(X_test)[:, 1]
                
                # Test at multiple thresholds
                for thresh in [0.50, 0.55, 0.60, 0.65, 0.70]:
                    signals = probs >= thresh
                    n_signals = signals.sum()
                    if n_signals < 10:
                        continue
                    
                    wins = y_test[signals].sum()
                    losses = n_signals - wins
                    wr = wins / n_signals * 100
                    
                    avg_win = cfg['ts_act'] - cfg['ts_trail']
                    avg_loss = cfg['sl']
                    ev = wr/100 * avg_win - (100-wr)/100 * avg_loss
                    rr = avg_win / avg_loss
                    
                    results.append({
                        'config': cfg['name'],
                        'direction': direction,
                        'features': fs_name,
                        'model': model_name,
                        'threshold': thresh,
                        'n_signals': n_signals,
                        'wins': wins,
                        'losses': losses,
                        'wr': wr,
                        'base_wr': base_wr_test,
                        'rr': rr,
                        'ev_per_trade': ev,
                        'total_ev': ev * n_signals,
                        'sl': cfg['sl'],
                        'ts_act': cfg['ts_act'],
                        'ts_trail': cfg['ts_trail'],
                    })

# =============================================================================
# 5. RESULTS — sorted by EV
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5: RESULTS — Best configurations by Expected Value")
print("=" * 70)

rdf = pd.DataFrame(results)

# Filter: only configs with positive EV and WR > 50%
good = rdf[(rdf['ev_per_trade'] > 0) & (rdf['wr'] > 50)].sort_values('total_ev', ascending=False)

print("\n--- TOP 20 CONFIGURATIONS (positive EV, WR>50%%) ---")
if len(good) == 0:
    print("  NO configurations found with positive EV and WR > 50%%!")
    print("\n  Showing best by WR regardless:")
    good = rdf[rdf['n_signals'] >= 10].sort_values('wr', ascending=False).head(20)

for _, r in good.head(20).iterrows():
    print("  %5s %s %-8s %-10s t=%.2f | %3dW/%3dL (%5.1f%% WR) | base=%.1f%% | "
          "EV=%.4f%% | total=%.2f%% | SL=%.2f TS=%.2f/%.2f" % (
        r['direction'], r['model'], r['features'], r['config'], r['threshold'],
        r['wins'], r['losses'], r['wr'], r['base_wr'],
        r['ev_per_trade'], r['total_ev'],
        r['sl'], r['ts_act'], r['ts_trail']))

print("\n--- BEST LONG ---")
long_good = rdf[(rdf['direction'] == 'LONG') & (rdf['n_signals'] >= 10)].sort_values('ev_per_trade', ascending=False)
for _, r in long_good.head(10).iterrows():
    print("  %s %-8s %-10s t=%.2f | %3dW/%3dL (%5.1f%% WR) | EV=%.4f%% | n=%d" % (
        r['model'], r['features'], r['config'], r['threshold'],
        r['wins'], r['losses'], r['wr'], r['ev_per_trade'], r['n_signals']))

print("\n--- BEST SHORT ---")
short_good = rdf[(rdf['direction'] == 'SHORT') & (rdf['n_signals'] >= 10)].sort_values('ev_per_trade', ascending=False)
for _, r in short_good.head(10).iterrows():
    print("  %s %-8s %-10s t=%.2f | %3dW/%3dL (%5.1f%% WR) | EV=%.4f%% | n=%d" % (
        r['model'], r['features'], r['config'], r['threshold'],
        r['wins'], r['losses'], r['wr'], r['ev_per_trade'], r['n_signals']))

# =============================================================================
# 6. FEATURE IMPORTANCE for best models
# =============================================================================
print("\n" + "=" * 70)
print("STEP 6: Feature importance for best LONG and SHORT")
print("=" * 70)

for direction in ['LONG', 'SHORT']:
    best = rdf[(rdf['direction'] == direction) & (rdf['n_signals'] >= 10)].sort_values('ev_per_trade', ascending=False)
    if len(best) == 0:
        print("  No good %s model found" % direction)
        continue
    b = best.iloc[0]
    label_key = "%s_%s" % (b['config'], direction)
    y = all_labels[label_key].reindex(valid.index)
    cols = [c for c in feature_sets[b['features']] if c in valid.columns]
    
    if b['model'] == 'GBM':
        mdl = GradientBoostingClassifier(n_estimators=200, max_depth=4,
            min_samples_leaf=50, learning_rate=0.05, random_state=42)
    else:
        mdl = RandomForestClassifier(n_estimators=200, max_depth=6,
            min_samples_leaf=30, random_state=42)
    
    mdl.fit(valid[train_mask][cols], y[train_mask])
    imp = pd.Series(mdl.feature_importances_, index=cols).sort_values(ascending=False)
    
    print("\n  Best %s: %s %s %s t=%.2f (%.1f%% WR, EV=%.4f%%)" % (
        direction, b['model'], b['features'], b['config'], b['threshold'], b['wr'], b['ev_per_trade']))
    print("  Feature importance:")
    for fname, fval in imp.head(8).items():
        print("    %-20s %.3f" % (fname, fval))

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
