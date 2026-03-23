#!/usr/bin/env python3
"""
Train and save the 2-sec tick ML models for deployment.
Models: L1 (full/wide), L3 (momentum/wide), S1 (full/current), S3 (mean_rev/current), S4 (full/tight)
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

MODELS_DIR = 'models/tick_models'
os.makedirs(MODELS_DIR, exist_ok=True)

# =============================================================================
# LOAD DATA
# =============================================================================
print("Loading 2-sec tick data...")
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

print("  Bars: %d (%s to %s)" % (len(bars), bars.index[0], bars.index[-1]))

# =============================================================================
# FEATURES
# =============================================================================
print("Computing features...")

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

print("  Features: %d" % len(feat.columns))

# =============================================================================
# LABELS
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

# Feature sets
FULL_FEATURES = list(feat.columns)
MOMENTUM_FEATURES = ['ret_1m', 'ret_2m', 'ret_5m', 'ret_10m', 'ret_15m', 'ret_30m', 'ret_1h',
                      'vol_5m', 'vol_15m', 'vol_1h', 'vol_ratio_5m_1h',
                      'ema_dist_5m', 'ema_dist_15m', 'ema_dist_1h', 'speed_5m_15m', 'consec']
MEAN_REV_FEATURES = ['rsi_15m', 'rsi_1h', 'bb_15m', 'bb_1h', 'chan_15m', 'chan_1h', 'chan_2h',
                      'ema_dist_15m', 'ema_dist_1h', 'vol_ratio_5m_1h', 'bar_range_ratio']

valid = feat.dropna()
print("  Valid bars: %d" % len(valid))

# =============================================================================
# TRAIN AND SAVE MODELS
# =============================================================================
models_config = {
    'L1_full_wide': {
        'direction': 'LONG', 'features': FULL_FEATURES, 'threshold': 0.60,
        'sl': 0.30, 'ts_act': 0.40, 'ts_trail': 0.10,
    },
    'L3_mom_wide': {
        'direction': 'LONG', 'features': MOMENTUM_FEATURES, 'threshold': 0.60,
        'sl': 0.30, 'ts_act': 0.40, 'ts_trail': 0.10,
    },
    'S1_full_cur': {
        'direction': 'SHORT', 'features': FULL_FEATURES, 'threshold': 0.70,
        'sl': 0.20, 'ts_act': 0.25, 'ts_trail': 0.05,
    },
    'S3_mr_cur': {
        'direction': 'SHORT', 'features': MEAN_REV_FEATURES, 'threshold': 0.70,
        'sl': 0.20, 'ts_act': 0.25, 'ts_trail': 0.05,
    },
    'S4_full_tight': {
        'direction': 'SHORT', 'features': FULL_FEATURES, 'threshold': 0.70,
        'sl': 0.15, 'ts_act': 0.20, 'ts_trail': 0.05,
    },
}

for name, cfg in models_config.items():
    print("\nTraining %s..." % name)
    direction = cfg['direction']
    
    # Simulate labels
    labels = simulate_labels(close_arr, n, cfg['sl'], cfg['ts_act'], cfg['ts_trail'], direction)
    y = pd.Series(labels, index=bars.index).reindex(valid.index)
    
    cols = [c for c in cfg['features'] if c in valid.columns]
    
    # Train on ALL data (for deployment)
    model = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=30, random_state=42)
    model.fit(valid[cols], y)
    
    # Save model
    model_path = os.path.join(MODELS_DIR, '%s.pkl' % name)
    joblib.dump(model, model_path)
    
    # Save config
    config_path = os.path.join(MODELS_DIR, '%s_config.pkl' % name)
    joblib.dump({
        'name': name,
        'direction': direction,
        'features': cols,
        'threshold': cfg['threshold'],
        'sl': cfg['sl'],
        'ts_act': cfg['ts_act'],
        'ts_trail': cfg['ts_trail'],
    }, config_path)
    
    wr = y.mean() * 100
    print("  Saved: %s (features=%d, base WR=%.1f%%)" % (model_path, len(cols), wr))

# Save feature list for the bot
feature_info = {
    'all_features': FULL_FEATURES,
    'momentum_features': MOMENTUM_FEATURES,
    'mean_rev_features': MEAN_REV_FEATURES,
}
joblib.dump(feature_info, os.path.join(MODELS_DIR, 'feature_info.pkl'))

print("\n" + "=" * 60)
print("ALL MODELS SAVED to %s" % MODELS_DIR)
print("=" * 60)
