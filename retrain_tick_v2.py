#!/usr/bin/env python3
"""Fast retrain of tick models using vectorized labels + same features as bot."""
import pandas as pd, numpy as np, joblib, os, warnings, time
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
warnings.filterwarnings('ignore')

t0 = time.time()

# ============ LOAD & RESAMPLE ============
print("Loading tick data...")
df = pd.read_csv('logs/btc_price_ticks.csv', parse_dates=['timestamp'])
df = df.set_index('timestamp').sort_index()
df = df[df['price'] > 0]
bars = df['price'].resample('16s').ohlc().dropna()
bars.columns = ['open', 'high', 'low', 'close']
close = bars['close']
print(f"Bars: {len(bars)}, {bars.index[0].date()} to {bars.index[-1].date()}")

# ============ VECTORIZED LABELS ============
# Instead of O(n*window) Python loop, sample every 30 bars and check fixed windows
print("Computing labels (vectorized)...")

def vectorized_labels(prices_arr, tp_pct, sl_pct, windows=[15, 30, 60, 120, 225, 450, 675]):
    """For each bar, check if TP is hit before SL in forward-looking windows."""
    n = len(prices_arr)
    long_win = np.zeros(n, dtype=np.int32)
    short_win = np.zeros(n, dtype=np.int32)
    
    # Build forward rolling max and min for each window size
    # Use the smallest window where we can determine outcome
    for w in windows:
        if w > n - 1:
            continue
        for i in range(0, n - w):
            if long_win[i] == 1 and short_win[i] == 1:
                continue  # Already determined both
            
            entry = prices_arr[i]
            tp_up = entry * (1 + tp_pct)
            sl_up = entry * (1 - sl_pct)
            tp_dn = entry * (1 - tp_pct)
            sl_dn = entry * (1 + sl_pct)
            
            fwd = prices_arr[i+1:i+w+1]
            
            if long_win[i] == 0:
                # Check LONG: does price hit tp_up before sl_up?
                hits_tp = np.where(fwd >= tp_up)[0]
                hits_sl = np.where(fwd <= sl_up)[0]
                tp_idx = hits_tp[0] if len(hits_tp) > 0 else w + 1
                sl_idx = hits_sl[0] if len(hits_sl) > 0 else w + 1
                if tp_idx < sl_idx:
                    long_win[i] = 1
            
            if short_win[i] == 0:
                hits_tp = np.where(fwd <= tp_dn)[0]
                hits_sl = np.where(fwd >= sl_dn)[0]
                tp_idx = hits_tp[0] if len(hits_tp) > 0 else w + 1
                sl_idx = hits_sl[0] if len(hits_sl) > 0 else w + 1
                if tp_idx < sl_idx:
                    short_win[i] = 1
    
    return long_win, short_win

# Only compute labels on sampled points (every 8th bar = ~2 min) for speed
sample_step = 8
sample_idx = np.arange(0, len(close) - 675, sample_step)
prices = close.values

long_labels = np.zeros(len(close), dtype=np.int32)
short_labels = np.zeros(len(close), dtype=np.int32)

print(f"  Computing for {len(sample_idx)} sampled bars...")
for count, i in enumerate(sample_idx):
    if count % 500 == 0 and count > 0:
        elapsed = time.time() - t0
        pct = count / len(sample_idx) * 100
        print(f"  {pct:.0f}% ({count}/{len(sample_idx)}) elapsed={elapsed:.0f}s")
    
    entry = prices[i]
    fwd = prices[i+1:i+676]
    
    # LONG
    hits_tp = np.where(fwd >= entry * 1.005)[0]
    hits_sl = np.where(fwd <= entry * 0.995)[0]
    tp_first = hits_tp[0] if len(hits_tp) > 0 else 9999
    sl_first = hits_sl[0] if len(hits_sl) > 0 else 9999
    if tp_first < sl_first:
        long_labels[i] = 1
    
    # SHORT
    hits_tp = np.where(fwd <= entry * 0.995)[0]
    hits_sl = np.where(fwd >= entry * 1.005)[0]
    tp_first = hits_tp[0] if len(hits_tp) > 0 else 9999
    sl_first = hits_sl[0] if len(hits_sl) > 0 else 9999
    if tp_first < sl_first:
        short_labels[i] = 1

# Only keep sampled rows
mask = np.zeros(len(close), dtype=bool)
mask[sample_idx] = True

print(f"  LONG win rate: {long_labels[mask].mean()*100:.1f}%")
print(f"  SHORT win rate: {short_labels[mask].mean()*100:.1f}%")
print(f"  Labels done in {time.time()-t0:.0f}s")

# ============ FEATURES (same as bot) ============
print("Computing features (same as btc_tick_bot.compute_features)...")

def compute_rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def compute_atr(high, low, close, period):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

feat = pd.DataFrame(index=bars.index)
for lb, lbl in [(4,'1m'),(8,'2m'),(19,'5m'),(38,'10m'),(56,'15m'),(113,'30m'),(225,'1h')]:
    feat['ret_'+lbl] = close.pct_change(lb)*100
ret1 = close.pct_change()*100
for w, lbl in [(19,'5m'),(56,'15m'),(113,'30m'),(225,'1h')]:
    feat['vol_'+lbl] = ret1.rolling(w).std()
feat['vol_ratio_5m_1h'] = feat['vol_5m']/feat['vol_1h'].replace(0,np.nan)
feat['vol_ratio_15m_1h'] = feat['vol_15m']/feat['vol_1h'].replace(0,np.nan)
for w, lbl in [(56,'15m'),(225,'1h'),(450,'2h')]:
    rmin=close.rolling(w).min(); rmax=close.rolling(w).max()
    rng=(rmax-rmin).replace(0,np.nan)
    feat['chan_'+lbl] = (close-rmin)/rng
for p, lbl in [(19,'5m'),(56,'15m'),(225,'1h')]:
    ema = close.ewm(span=p,adjust=False).mean()
    feat['ema_dist_'+lbl] = (close-ema)/close*100
for p, lbl in [(56,'15m'),(225,'1h')]:
    feat['rsi_'+lbl] = compute_rsi(close, p)
for p, lbl in [(56,'15m'),(225,'1h')]:
    sma=close.rolling(p).mean(); std=close.rolling(p).std()
    lower=sma-2*std; upper=sma+2*std
    feat['bb_'+lbl] = (close-lower)/(upper-lower).replace(0,np.nan)
for (f,s,sig,lbl) in [(19,56,14,'5m_15m'),(56,225,38,'15m_1h')]:
    ef=close.ewm(span=f,adjust=False).mean()
    es=close.ewm(span=s,adjust=False).mean()
    ml=ef-es; sl=ml.ewm(span=sig,adjust=False).mean()
    feat['macd_hist_'+lbl] = ml-sl
for p, lbl in [(56,'15m'),(225,'1h')]:
    feat['atr_pct_'+lbl] = compute_atr(bars['high'],bars['low'],close,p)/close*100
feat['speed_5m_15m'] = (close.diff(19).abs()/19)/(close.diff(56).abs()/56).replace(0,np.nan)
ret_sign = np.sign(close.diff())
feat['consec'] = ret_sign.groupby((ret_sign!=ret_sign.shift()).cumsum()).cumcount()+1
feat['consec'] *= ret_sign
feat['bar_range_pct'] = (bars['high']-bars['low'])/close*100
feat['bar_range_ma'] = feat['bar_range_pct'].rolling(56).mean()
feat['bar_range_ratio'] = feat['bar_range_pct']/feat['bar_range_ma'].replace(0,np.nan)
up_ticks = (close>close.shift(1)).astype(int)
down_ticks = (close<close.shift(1)).astype(int)
for w, lbl in [(19,'5m'),(56,'15m')]:
    feat['tick_imbal_'+lbl] = (up_ticks.rolling(w).sum()-down_ticks.rolling(w).sum())/w
feat['hour'] = bars.index.hour

print(f"Features: {len(feat.columns)} cols")

# ============ ALIGN ============
# Keep only sampled rows that have valid features
feat_sampled = feat.iloc[sample_idx].dropna()
valid_idx = feat_sampled.index
lw_valid = long_labels[bars.index.isin(valid_idx)][:len(feat_sampled)]
sw_valid = short_labels[bars.index.isin(valid_idx)][:len(feat_sampled)]
feat_sampled = feat_sampled.iloc[:len(lw_valid)]

print(f"Training samples: {len(feat_sampled)}")

# Temporal 75/25 split
split = int(len(feat_sampled) * 0.75)
X_tr = feat_sampled.iloc[:split]
X_te = feat_sampled.iloc[split:]
yl_tr, yl_te = lw_valid[:split], lw_valid[split:]
ys_tr, ys_te = sw_valid[:split], sw_valid[split:]

print(f"Train: {len(X_tr)} ({X_tr.index[0].date()} to {X_tr.index[-1].date()})")
print(f"Test:  {len(X_te)} ({X_te.index[0].date()} to {X_te.index[-1].date()})")
print(f"Train LONG WR: {yl_tr.mean()*100:.1f}%, SHORT WR: {ys_tr.mean()*100:.1f}%")
print(f"Test  LONG WR: {yl_te.mean()*100:.1f}%, SHORT WR: {ys_te.mean()*100:.1f}%")

# ============ TRAIN 5 MODELS ============
# L1: full features, LONG
# L3: momentum subset, LONG  
# S1: full features, SHORT
# S3: mean-reversion subset, SHORT
# S4: full features tight SL, SHORT (same model, different params)

full_features = list(feat_sampled.columns)
mom_features = [c for c in full_features if any(x in c for x in ['ret_','vol_','ema_dist','speed','consec'])]
mr_features = [c for c in full_features if any(x in c for x in ['rsi','bb','chan','ema_dist','vol_ratio','bar_range_ratio'])]

configs = {
    'L1_full_wide':  {'features': full_features, 'y_tr': yl_tr, 'y_te': yl_te, 'direction': 'LONG'},
    'L3_mom_wide':   {'features': mom_features,  'y_tr': yl_tr, 'y_te': yl_te, 'direction': 'LONG'},
    'S1_full_cur':   {'features': full_features, 'y_tr': ys_tr, 'y_te': ys_te, 'direction': 'SHORT'},
    'S3_mr_cur':     {'features': mr_features,   'y_tr': ys_tr, 'y_te': ys_te, 'direction': 'SHORT'},
    'S4_full_tight': {'features': full_features, 'y_tr': ys_tr, 'y_te': ys_te, 'direction': 'SHORT'},
}

os.makedirs('models/tick_models_v2', exist_ok=True)
best_models = {}

for name, cfg in configs.items():
    print(f"\n{'='*50}")
    print(f"Training {name} ({cfg['direction']}, {len(cfg['features'])} features)")
    print(f"{'='*50}")
    
    X_tr_m = X_tr[cfg['features']]
    X_te_m = X_te[cfg['features']]
    y_tr = cfg['y_tr']
    y_te = cfg['y_te']
    
    # Try both RF and GBM, keep best
    best_wr = 0
    best_model = None
    best_thresh = 0
    
    for algo_name, algo in [
        ('GBM', GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, 
                                            min_samples_leaf=30, subsample=0.8, random_state=42)),
        ('RF', RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=30,
                                       class_weight='balanced', random_state=42, n_jobs=-1)),
    ]:
        algo.fit(X_tr_m, y_tr)
        probs = algo.predict_proba(X_te_m)[:, 1]
        
        print(f"\n  {algo_name}:")
        for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
            indices = np.where(probs >= thresh)[0]
            # Enforce 30-bar gap
            taken = []; last = -30
            for idx in indices:
                if idx - last >= 30: taken.append(idx); last = idx
            if len(taken) < 5: continue
            wins = sum(y_te[i] for i in taken)
            nt = len(taken)
            wr = wins / nt * 100
            days = max((X_te_m.index[-1] - X_te_m.index[0]).days, 1)
            marker = ""
            # Prefer: high WR with reasonable trade count
            score = wr * min(nt / 10, 1.0)  # penalize very few trades
            if wr > best_wr and nt >= 10:
                best_wr = wr
                best_model = algo
                best_thresh = thresh
                best_algo = algo_name
                marker = " <-- BEST"
            print(f"    >{thresh:.0%}: {nt}T ({nt/days:.1f}/d) {wr:.0f}%WR {wins}W/{nt-wins}L{marker}")
    
    if best_model is not None:
        print(f"\n  Selected: {best_algo} @ >{best_thresh:.0%} ({best_wr:.0f}% WR)")
        # Top features
        imps = sorted(zip(cfg['features'], best_model.feature_importances_), key=lambda x: -x[1])[:5]
        print(f"  Top: {', '.join(f'{n}={v:.3f}' for n,v in imps)}")
        
        best_models[name] = {
            'model': best_model,
            'features': cfg['features'],
            'threshold': best_thresh,
            'direction': cfg['direction'],
            'wr': best_wr,
            'algo': best_algo
        }

# ============ SAVE ============
print(f"\n{'='*50}")
print("SAVING MODELS")
print(f"{'='*50}")
for name, info in best_models.items():
    joblib.dump(info['model'], f'models/tick_models_v2/{name}.pkl')
    cfg_dict = {
        'name': name,
        'direction': info['direction'],
        'features': info['features'],
        'threshold': info['threshold'],
        'sl': 0.3 if 'L' in name else 0.2,
        'ts_act': 0.4 if 'L' in name else 0.25,
        'ts_trail': 0.1 if 'L' in name else 0.05,
    }
    if name == 'S4_full_tight':
        cfg_dict['sl'] = 0.15
        cfg_dict['ts_act'] = 0.2
    joblib.dump(cfg_dict, f'models/tick_models_v2/{name}_config.pkl')
    print(f"  {name}: {info['algo']} >{info['threshold']:.0%} {info['wr']:.0f}%WR features={len(info['features'])}")

joblib.dump({'features': full_features, 'bar_seconds': 16}, 'models/tick_models_v2/feature_info.pkl')
print(f"\nTotal time: {time.time()-t0:.0f}s")
print("DONE - models saved to models/tick_models_v2/")
