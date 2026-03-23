#!/usr/bin/env python3
"""
Push for 70-80% win rate entry models.
Strategy: be very selective — trade less but win more.

Approaches:
1. Higher model probability thresholds (70%, 75%, 80%, 85%)
2. Ensemble agreement: require 2-3 models to agree
3. Confluence filters: model + price regime conditions
4. Asymmetric SL/TP: tighter SL, let winners run further
5. Time-of-day filtering (morning only, etc.)
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Load and prepare data (reuse from explore_models.py)
# =============================================================================
print("Loading and preparing data...")

raw = pd.read_csv('logs/btc_price_ticks.csv', parse_dates=['timestamp'])
raw = raw.set_index('timestamp').sort_index()
raw = raw[raw['price'] > 0]

bars = raw['price'].resample('16s').agg(
    open='first', high='max', low='min', close='last', count='count'
).dropna()

close = bars['close']
high = bars['high']
low = bars['low']

def compute_rsi(s, p):
    d = s.diff(); g = d.clip(lower=0).rolling(p).mean(); l = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - (100 / (1 + g / l.replace(0, np.nan)))

def compute_bb(s, p):
    sma = s.rolling(p).mean(); std = s.rolling(p).std()
    return (s - (sma - 2*std)) / ((sma + 2*std) - (sma - 2*std)).replace(0, np.nan)

def compute_atr(h, l, c, p):
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(p).mean()

# Build all features
ind = pd.DataFrame(index=bars.index)

# Price features
for lb, label in [(4, '1m'), (19, '5m'), (56, '15m'), (225, '1h'), (450, '2h')]:
    ind[f'ret_{label}'] = close.pct_change(lb) * 100

ret1 = close.pct_change() * 100
for w, label in [(19, '5m'), (56, '15m'), (225, '1h')]:
    ind[f'vol_{label}'] = ret1.rolling(w).std()

for w, label in [(56, '15m'), (225, '1h'), (450, '2h')]:
    roll_min = close.rolling(w).min(); roll_max = close.rolling(w).max()
    ind[f'chan_{label}'] = (close - roll_min) / (roll_max - roll_min).replace(0, np.nan)

for p, label in [(19, '5m'), (56, '15m'), (225, '1h'), (450, '2h')]:
    ema = close.ewm(span=p, adjust=False).mean()
    ind[f'ema_dist_{label}'] = (close - ema) / close * 100

ind['speed_5m'] = close.diff(19).abs() / 19
ind['speed_15m'] = close.diff(56).abs() / 56
ind['speed_ratio'] = ind['speed_5m'] / ind['speed_15m'].replace(0, np.nan)

# Indicator features
for p, label in [(19, '5m'), (56, '15m'), (225, '1h')]:
    ind[f'rsi_{label}'] = compute_rsi(close, p)

for p, label in [(56, '15m'), (225, '1h')]:
    ind[f'bb_{label}'] = compute_bb(close, p)

for p, label in [(56, '15m'), (225, '1h')]:
    ind[f'atr_pct_{label}'] = compute_atr(high, low, close, p) / close * 100

# MACD
for (f, s, sig, label) in [(19, 56, 14, '5m'), (56, 225, 38, '15m')]:
    ef = close.ewm(span=f, adjust=False).mean()
    es = close.ewm(span=s, adjust=False).mean()
    ind[f'macd_{label}'] = (ef - es) - (ef - es).ewm(span=sig, adjust=False).mean()

ret_sign = np.sign(close.diff())
ind['consec'] = ret_sign.groupby((ret_sign != ret_sign.shift()).cumsum()).cumcount() + 1
ind['consec'] *= ret_sign

ind['hour'] = bars.index.hour
ind['tick_ratio'] = bars['count'] / bars['count'].rolling(56).mean().replace(0, np.nan)

print(f"Features: {len(ind.columns)} cols, {len(ind)} bars")

# =============================================================================
# Simulate targets with multiple SL/TP configs
# =============================================================================
print("\nSimulating targets...")

close_arr = close.values

def sim_targets(prices, sl_pct, ts_act, ts_trail, max_bars=675):
    n = len(prices)
    lw = np.zeros(n, dtype=int)
    sw = np.zeros(n, dtype=int)
    lp = np.full(n, np.nan)
    sp = np.full(n, np.nan)
    
    for i in range(n - max_bars):
        entry = prices[i]
        
        # LONG
        sl = entry * (1 - sl_pct / 100)
        ta = entry * (1 + ts_act / 100)
        pk = entry; ts_on = False
        for j in range(i+1, min(i+max_bars+1, n)):
            p = prices[j]
            if p <= sl: lp[i] = (sl/entry-1)*100; break
            if p > pk: pk = p
            if not ts_on and p >= ta: ts_on = True
            if ts_on:
                tr = pk * (1 - ts_trail / 100)
                if p <= tr: lp[i] = (tr/entry-1)*100; lw[i] = 1; break
        
        # SHORT
        sl = entry * (1 + sl_pct / 100)
        ta = entry * (1 - ts_act / 100)
        pk = entry; ts_on = False
        for j in range(i+1, min(i+max_bars+1, n)):
            p = prices[j]
            if p >= sl: sp[i] = (entry/sl-1)*100; break
            if p < pk: pk = p
            if not ts_on and p <= ta: ts_on = True
            if ts_on:
                tr = pk * (1 + ts_trail / 100)
                if p >= tr: sp[i] = (entry/tr-1)*100; sw[i] = 1; break
    
    return lw, sw, lp, sp

# Config A: current (SL=0.30%, TS=0.30%/0.10%)
lw_a, sw_a, lp_a, sp_a = sim_targets(close_arr, 0.30, 0.30, 0.10)
# Config B: tighter SL (SL=0.15%, TS=0.25%/0.05%)
lw_b, sw_b, lp_b, sp_b = sim_targets(close_arr, 0.15, 0.25, 0.05)
# Config C: wider TS (SL=0.20%, TS=0.40%/0.10%) — let winners run
lw_c, sw_c, lp_c, sp_c = sim_targets(close_arr, 0.20, 0.40, 0.10)

configs = {
    'SL0.30_TS0.30/0.10': (lw_a, sw_a, lp_a, sp_a),
    'SL0.15_TS0.25/0.05': (lw_b, sw_b, lp_b, sp_b),
    'SL0.20_TS0.40/0.10': (lw_c, sw_c, lp_c, sp_c),
}

for name, (lw, sw, lp, sp) in configs.items():
    lr = (~np.isnan(lp)).sum()
    sr = (~np.isnan(sp)).sum()
    print(f"  {name}: LONG {lw.sum()}/{lr} ({lw.sum()/lr*100:.1f}%), SHORT {sw.sum()}/{sr} ({sw.sum()/sr*100:.1f}%)")

# =============================================================================
# Train/test split
# =============================================================================
valid = ind.dropna()
split_date = valid.index[0] + pd.Timedelta(days=7)
train_mask = np.array(valid.index < split_date)
test_mask = np.array(valid.index >= split_date)

print(f"\nTrain: {train_mask.sum()} bars, Test: {test_mask.sum()} bars")

# =============================================================================
# Strategy 1: Very high thresholds on single models
# =============================================================================
print(f"\n{'='*70}")
print("STRATEGY 1: High thresholds (single model)")
print(f"{'='*70}")

FEAT_SETS = {
    'C_combined': ['ret_5m','ret_15m','ret_1h','vol_15m','vol_1h','chan_1h','chan_2h',
                    'rsi_15m','rsi_1h','bb_1h','macd_15m','atr_pct_1h','speed_ratio','hour'],
    'D_momentum': ['ret_5m','ret_15m','ret_1h','ret_2h','ema_dist_5m','ema_dist_15m',
                    'ema_dist_1h','ema_dist_2h','speed_ratio','vol_1h','chan_1h','hour'],
    'E_minimal': ['ret_1h','ret_2h','vol_1h','chan_1h','rsi_1h','hour'],
}

MIN_GAP = 30  # 8 min gap

for config_name, (lw, sw, lp, sp) in configs.items():
    long_s = pd.Series(lw[:len(ind)], index=ind.index).reindex(valid.index).fillna(0).astype(int)
    short_s = pd.Series(sw[:len(ind)], index=ind.index).reindex(valid.index).fillna(0).astype(int)
    long_pnl_s = pd.Series(lp[:len(ind)], index=ind.index).reindex(valid.index)
    short_pnl_s = pd.Series(sp[:len(ind)], index=ind.index).reindex(valid.index)
    
    for direction in ['LONG', 'SHORT']:
        y_s = long_s if direction == 'LONG' else short_s
        pnl_s = long_pnl_s if direction == 'LONG' else short_pnl_s
        
        for feat_name, feat_cols in FEAT_SETS.items():
            model = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=30, random_state=42)
            model.fit(valid[train_mask][feat_cols], y_s[train_mask])
            probs = model.predict_proba(valid[test_mask][feat_cols])[:, 1]
            
            for thresh in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
                indices = np.where(probs >= thresh)[0]
                taken = []; last = -MIN_GAP
                for idx in indices:
                    if idx - last >= MIN_GAP: taken.append(idx); last = idx
                
                if len(taken) < 5: continue
                
                wins = sum(y_s.iloc[test_mask.nonzero()[0][i]] for i in taken)
                nt = len(taken)
                wr = wins / nt * 100
                total_pnl = sum(pnl_s.iloc[test_mask.nonzero()[0][i]] for i in taken 
                               if not np.isnan(pnl_s.iloc[test_mask.nonzero()[0][i]]))
                days = (valid[test_mask].index[-1] - valid[test_mask].index[0]).total_seconds() / 86400
                
                if wr >= 65:
                    print(f"  {direction:5s} {config_name:22s} {feat_name:14s} >{thresh:.0%}: "
                          f"{nt:3d} trades ({nt/days:.1f}/day), {wr:.0f}% WR, "
                          f"{wins:.0f}W/{nt-wins:.0f}L, {total_pnl:+.2f}%")

# =============================================================================
# Strategy 2: Ensemble agreement (multiple models must agree)
# =============================================================================
print(f"\n{'='*70}")
print("STRATEGY 2: Ensemble agreement (2-3 models must agree)")
print(f"{'='*70}")

for config_name, (lw, sw, lp, sp) in [('SL0.30_TS0.30/0.10', configs['SL0.30_TS0.30/0.10'])]:
    long_s = pd.Series(lw[:len(ind)], index=ind.index).reindex(valid.index).fillna(0).astype(int)
    short_s = pd.Series(sw[:len(ind)], index=ind.index).reindex(valid.index).fillna(0).astype(int)
    long_pnl_s = pd.Series(lp[:len(ind)], index=ind.index).reindex(valid.index)
    short_pnl_s = pd.Series(sp[:len(ind)], index=ind.index).reindex(valid.index)
    
    for direction in ['LONG', 'SHORT']:
        y_s = long_s if direction == 'LONG' else short_s
        pnl_s = long_pnl_s if direction == 'LONG' else short_pnl_s
        
        # Train 3 different models on different feature sets
        models = {}
        for feat_name in ['C_combined', 'D_momentum', 'E_minimal']:
            feat_cols = FEAT_SETS[feat_name]
            m = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=30, random_state=42)
            m.fit(valid[train_mask][feat_cols], y_s[train_mask])
            models[feat_name] = (m, feat_cols)
        
        # Get probabilities from all 3
        probs_all = {}
        for fname, (m, fc) in models.items():
            probs_all[fname] = m.predict_proba(valid[test_mask][fc])[:, 1]
        
        test_indices = test_mask.nonzero()[0]
        
        for min_agree in [2, 3]:
            for thresh in [0.50, 0.55, 0.60, 0.65, 0.70]:
                # Count how many models agree at threshold
                agree_count = np.zeros(len(test_indices))
                for fname in probs_all:
                    agree_count += (probs_all[fname] >= thresh).astype(int)
                
                indices = np.where(agree_count >= min_agree)[0]
                taken = []; last = -MIN_GAP
                for idx in indices:
                    if idx - last >= MIN_GAP: taken.append(idx); last = idx
                
                if len(taken) < 5: continue
                
                wins = sum(y_s.iloc[test_indices[i]] for i in taken)
                nt = len(taken)
                wr = wins / nt * 100
                total_pnl = sum(pnl_s.iloc[test_indices[i]] for i in taken 
                               if not np.isnan(pnl_s.iloc[test_indices[i]]))
                days = (valid[test_mask].index[-1] - valid[test_mask].index[0]).total_seconds() / 86400
                
                if wr >= 65:
                    print(f"  {direction:5s} {min_agree}/3 agree >{thresh:.0%}: "
                          f"{nt:3d} trades ({nt/days:.1f}/day), {wr:.0f}% WR, "
                          f"{wins:.0f}W/{nt-wins:.0f}L, {total_pnl:+.2f}%")

# =============================================================================
# Strategy 3: Time-filtered (only trade best hours)
# =============================================================================
print(f"\n{'='*70}")
print("STRATEGY 3: Time-of-day filter + model")
print(f"{'='*70}")

for config_name, (lw, sw, lp, sp) in [('SL0.30_TS0.30/0.10', configs['SL0.30_TS0.30/0.10'])]:
    long_s = pd.Series(lw[:len(ind)], index=ind.index).reindex(valid.index).fillna(0).astype(int)
    short_s = pd.Series(sw[:len(ind)], index=ind.index).reindex(valid.index).fillna(0).astype(int)
    long_pnl_s = pd.Series(lp[:len(ind)], index=ind.index).reindex(valid.index)
    short_pnl_s = pd.Series(sp[:len(ind)], index=ind.index).reindex(valid.index)
    
    hours_test = valid[test_mask].index.hour
    
    for direction in ['LONG', 'SHORT']:
        y_s = long_s if direction == 'LONG' else short_s
        pnl_s = long_pnl_s if direction == 'LONG' else short_pnl_s
        
        feat_cols = FEAT_SETS['C_combined']
        model = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=30, random_state=42)
        model.fit(valid[train_mask][feat_cols], y_s[train_mask])
        probs = model.predict_proba(valid[test_mask][feat_cols])[:, 1]
        
        test_indices = test_mask.nonzero()[0]
        
        # Try different hour windows
        for hours_label, hour_filter in [
            ('00-08 (night)', lambda h: h < 8),
            ('08-14 (morning)', lambda h: (h >= 8) & (h < 14)),
            ('14-20 (afternoon)', lambda h: (h >= 14) & (h < 20)),
            ('00-12', lambda h: h < 12),
            ('04-10', lambda h: (h >= 4) & (h < 10)),
        ]:
            hour_mask = hour_filter(hours_test)
            
            for thresh in [0.50, 0.55, 0.60, 0.65, 0.70]:
                mask = (probs >= thresh) & np.array(hour_mask)
                indices = np.where(mask)[0]
                taken = []; last = -MIN_GAP
                for idx in indices:
                    if idx - last >= MIN_GAP: taken.append(idx); last = idx
                
                if len(taken) < 5: continue
                
                wins = sum(y_s.iloc[test_indices[i]] for i in taken)
                nt = len(taken)
                wr = wins / nt * 100
                total_pnl = sum(pnl_s.iloc[test_indices[i]] for i in taken 
                               if not np.isnan(pnl_s.iloc[test_indices[i]]))
                days = (valid[test_mask].index[-1] - valid[test_mask].index[0]).total_seconds() / 86400
                
                if wr >= 65:
                    print(f"  {direction:5s} {hours_label:16s} >{thresh:.0%}: "
                          f"{nt:3d} trades ({nt/days:.1f}/day), {wr:.0f}% WR, "
                          f"{wins:.0f}W/{nt-wins:.0f}L, {total_pnl:+.2f}%")

# =============================================================================
# Strategy 4: Confluence — model + regime filters
# =============================================================================
print(f"\n{'='*70}")
print("STRATEGY 4: Model + regime confluence")
print(f"{'='*70}")

for config_name, (lw, sw, lp, sp) in [('SL0.30_TS0.30/0.10', configs['SL0.30_TS0.30/0.10'])]:
    long_s = pd.Series(lw[:len(ind)], index=ind.index).reindex(valid.index).fillna(0).astype(int)
    short_s = pd.Series(sw[:len(ind)], index=ind.index).reindex(valid.index).fillna(0).astype(int)
    long_pnl_s = pd.Series(lp[:len(ind)], index=ind.index).reindex(valid.index)
    short_pnl_s = pd.Series(sp[:len(ind)], index=ind.index).reindex(valid.index)
    
    test_data = valid[test_mask]
    test_indices = test_mask.nonzero()[0]
    
    for direction in ['LONG', 'SHORT']:
        y_s = long_s if direction == 'LONG' else short_s
        pnl_s = long_pnl_s if direction == 'LONG' else short_pnl_s
        
        feat_cols = FEAT_SETS['C_combined']
        model = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=30, random_state=42)
        model.fit(valid[train_mask][feat_cols], y_s[train_mask])
        probs = model.predict_proba(valid[test_mask][feat_cols])[:, 1]
        
        # Regime filters
        if direction == 'LONG':
            regime_filters = {
                'trend_up_1h': test_data['ret_1h'] > 0,
                'trend_up_2h': test_data['ret_2h'] > 0,
                'rsi_ok': test_data['rsi_1h'] > 45,
                'above_ema_1h': test_data['ema_dist_1h'] > 0,
                'low_vol': test_data['vol_1h'] < test_data['vol_1h'].quantile(0.7),
                'chan_upper': test_data['chan_1h'] > 0.5,
                'trend+rsi': (test_data['ret_1h'] > 0) & (test_data['rsi_1h'] > 50),
                'trend+ema+rsi': (test_data['ret_1h'] > 0) & (test_data['ema_dist_1h'] > 0) & (test_data['rsi_1h'] > 50),
                'strong_trend': test_data['ret_2h'] > 0.15,
            }
        else:
            regime_filters = {
                'trend_dn_1h': test_data['ret_1h'] < 0,
                'trend_dn_2h': test_data['ret_2h'] < 0,
                'rsi_low': test_data['rsi_1h'] < 45,
                'below_ema_1h': test_data['ema_dist_1h'] < 0,
                'chan_lower': test_data['chan_1h'] < 0.5,
                'trend+rsi': (test_data['ret_1h'] < 0) & (test_data['rsi_1h'] < 50),
                'trend+ema+rsi': (test_data['ret_1h'] < 0) & (test_data['ema_dist_1h'] < 0) & (test_data['rsi_1h'] < 50),
                'strong_trend': test_data['ret_2h'] < -0.15,
            }
        
        for filter_name, filter_mask in regime_filters.items():
            for thresh in [0.50, 0.55, 0.60, 0.65, 0.70]:
                mask = (probs >= thresh) & np.array(filter_mask)
                indices = np.where(mask)[0]
                taken = []; last = -MIN_GAP
                for idx in indices:
                    if idx - last >= MIN_GAP: taken.append(idx); last = idx
                
                if len(taken) < 5: continue
                
                wins = sum(y_s.iloc[test_indices[i]] for i in taken)
                nt = len(taken)
                wr = wins / nt * 100
                total_pnl = sum(pnl_s.iloc[test_indices[i]] for i in taken 
                               if not np.isnan(pnl_s.iloc[test_indices[i]]))
                days = (valid[test_mask].index[-1] - valid[test_mask].index[0]).total_seconds() / 86400
                
                if wr >= 70:
                    print(f"  {direction:5s} {filter_name:20s} >{thresh:.0%}: "
                          f"{nt:3d} trades ({nt/days:.1f}/day), {wr:.0f}% WR, "
                          f"{wins:.0f}W/{nt-wins:.0f}L, {total_pnl:+.2f}%")

print(f"\n{'='*70}")
print("EXPLORATION COMPLETE")
print(f"{'='*70}")
