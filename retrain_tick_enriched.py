#!/usr/bin/env python3
"""Retrain tick models with enriched features (price + volume + indicators)."""
import pandas as pd, numpy as np, joblib, os, time, warnings
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
warnings.filterwarnings('ignore')
t0 = time.time()

# ============ LOAD TICK DATA ============
print("Loading tick data...")
df = pd.read_csv('logs/btc_price_ticks.csv', parse_dates=['timestamp'])
df = df.set_index('timestamp').sort_index()
df = df[df['price'] > 0]
bars = df['price'].resample('16s').ohlc().dropna()
bars.columns = ['open', 'high', 'low', 'close']
close = bars['close']
print(f"16-sec bars: {len(bars)}, {bars.index[0]} to {bars.index[-1]}")

# ============ LOAD 5-MIN ENRICHED FEATURES ============
print("Loading 5-min enriched features...")
feat5 = pd.read_parquet('data/BTC_features.parquet')
extra_cols = [
    'Volume', 'volume_obv', 'volume_cmf', 'volume_mfi', 'volume_vwap',
    'volatility_atr', 'volatility_bbp', 'volatility_bbw', 'volatility_kcp',
    'trend_macd_diff', 'trend_adx', 'trend_adx_pos', 'trend_adx_neg', 'trend_cci',
    'trend_aroon_ind', 'trend_stc',
    'momentum_rsi', 'momentum_stoch', 'momentum_wr', 'momentum_ao', 'momentum_roc',
    'momentum_tsi', 'momentum_ppo_hist',
]
extra_cols = [c for c in extra_cols if c in feat5.columns]
feat5_sub = feat5[extra_cols]
feat5_16s = feat5_sub.reindex(bars.index, method='ffill')
print(f"Enriched features: {len(extra_cols)} cols, overlap bars: {feat5_16s.notna().all(axis=1).sum()}")

# ============ PRICE FEATURES (same as bot) ============
print("Computing price features...")

def compute_rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def compute_atr(high, low, close, period):
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
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
    feat['chan_'+lbl]=(close-rmin)/(rmax-rmin).replace(0,np.nan)
for p, lbl in [(19,'5m'),(56,'15m'),(225,'1h')]:
    ema=close.ewm(span=p,adjust=False).mean()
    feat['ema_dist_'+lbl]=(close-ema)/close*100
for p, lbl in [(56,'15m'),(225,'1h')]:
    feat['rsi_'+lbl]=compute_rsi(close,p)
for p, lbl in [(56,'15m'),(225,'1h')]:
    sma=close.rolling(p).mean(); std=close.rolling(p).std()
    lower=sma-2*std; upper=sma+2*std
    feat['bb_'+lbl]=(close-lower)/(upper-lower).replace(0,np.nan)
for (f,s,sig,lbl) in [(19,56,14,'5m_15m'),(56,225,38,'15m_1h')]:
    ef=close.ewm(span=f,adjust=False).mean(); es=close.ewm(span=s,adjust=False).mean()
    ml=ef-es; sl_line=ml.ewm(span=sig,adjust=False).mean()
    feat['macd_hist_'+lbl]=ml-sl_line
for p, lbl in [(56,'15m'),(225,'1h')]:
    feat['atr_pct_'+lbl]=compute_atr(bars['high'],bars['low'],close,p)/close*100
feat['speed_5m_15m']=(close.diff(19).abs()/19)/(close.diff(56).abs()/56).replace(0,np.nan)
ret_sign=np.sign(close.diff())
feat['consec']=ret_sign.groupby((ret_sign!=ret_sign.shift()).cumsum()).cumcount()+1
feat['consec']*=ret_sign
feat['bar_range_pct']=(bars['high']-bars['low'])/close*100
feat['bar_range_ma']=feat['bar_range_pct'].rolling(56).mean()
feat['bar_range_ratio']=feat['bar_range_pct']/feat['bar_range_ma'].replace(0,np.nan)
up_ticks=(close>close.shift(1)).astype(int); dn_ticks=(close<close.shift(1)).astype(int)
for w, lbl in [(19,'5m'),(56,'15m')]:
    feat['tick_imbal_'+lbl]=(up_ticks.rolling(w).sum()-dn_ticks.rolling(w).sum())/w
feat['hour']=bars.index.hour

price_cols = list(feat.columns)
print(f"Price features: {len(price_cols)}")

# Add enriched features
for col in extra_cols:
    feat['ext_'+col] = feat5_16s[col]
all_cols = list(feat.columns)
print(f"Total features: {len(all_cols)}")

# ============ LABELS ============
print("Computing labels (sampled every 8 bars)...")
prices = close.values
sample_step = 8
sample_idx = np.arange(450, len(close)-675, sample_step)
long_labels = np.zeros(len(close), dtype=np.int32)
short_labels = np.zeros(len(close), dtype=np.int32)

for i in sample_idx:
    e = prices[i]; fwd = prices[i+1:i+676]
    tp=np.where(fwd>=e*1.005)[0]; sl=np.where(fwd<=e*0.995)[0]
    if len(tp)>0 and (len(sl)==0 or tp[0]<sl[0]): long_labels[i]=1
    tp=np.where(fwd<=e*0.995)[0]; sl=np.where(fwd>=e*1.005)[0]
    if len(tp)>0 and (len(sl)==0 or tp[0]<sl[0]): short_labels[i]=1

mask = np.zeros(len(close), dtype=bool); mask[sample_idx] = True
print(f"Samples: {mask.sum()}, LONG WR={long_labels[mask].mean()*100:.1f}%, SHORT WR={short_labels[mask].mean()*100:.1f}%")
print(f"Labels done: {time.time()-t0:.0f}s")

# ============ ALIGN ============
feat_sampled = feat.iloc[sample_idx].dropna()
lw = long_labels[bars.index.isin(feat_sampled.index)][:len(feat_sampled)]
sw = short_labels[bars.index.isin(feat_sampled.index)][:len(feat_sampled)]
feat_sampled = feat_sampled.iloc[:len(lw)]
print(f"Valid samples: {len(feat_sampled)}")

# ============ SPLIT ============
split = int(len(feat_sampled)*0.75)
X_tr, X_te = feat_sampled.iloc[:split], feat_sampled.iloc[split:]
yl_tr, yl_te = lw[:split], lw[split:]
ys_tr, ys_te = sw[:split], sw[split:]
print(f"Train: {len(X_tr)} ({X_tr.index[0].date()} to {X_tr.index[-1].date()})")
print(f"Test:  {len(X_te)} ({X_te.index[0].date()} to {X_te.index[-1].date()})")

# ============ TRAIN: PRICE-ONLY vs ENRICHED ============
results = {}
for name, y_tr, y_te, direction in [
    ('L1_full_wide', yl_tr, yl_te, 'LONG'),
    ('L3_mom_wide', yl_tr, yl_te, 'LONG'),
    ('S1_full_cur', ys_tr, ys_te, 'SHORT'),
    ('S3_mr_cur', ys_tr, ys_te, 'SHORT'),
    ('S4_full_tight', ys_tr, ys_te, 'SHORT'),
]:
    print(f"\n{'='*60}")
    print(f"{name} ({direction})")
    print(f"{'='*60}")
    
    for feat_set_name, cols in [('PRICE-ONLY', price_cols), ('ENRICHED', all_cols)]:
        X_tr_m = X_tr[cols]; X_te_m = X_te[cols]
        
        model = GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            min_samples_leaf=30, subsample=0.8, random_state=42
        )
        model.fit(X_tr_m, y_tr)
        probs = model.predict_proba(X_te_m)[:,1]
        
        print(f"\n  {feat_set_name} ({len(cols)} features):")
        best_wr = 0; best_thresh = 0
        for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
            indices = np.where(probs >= thresh)[0]
            taken=[]; last=-30
            for idx in indices:
                if idx-last>=30: taken.append(idx); last=idx
            if len(taken)<5: continue
            wins = sum(y_te[i] for i in taken)
            nt=len(taken); wr=wins/nt*100
            days=max((X_te_m.index[-1]-X_te_m.index[0]).days,1)
            tag = ""
            if wr > best_wr and nt >= 10:
                best_wr = wr; best_thresh = thresh; tag = " ***"
            print(f"    >{thresh:.0%}: {nt}T ({nt/days:.1f}/d) {wr:.0f}%WR {wins}W/{nt-wins}L{tag}")
        
        if feat_set_name == 'ENRICHED' and best_wr > 0:
            # Top features
            imps = sorted(zip(cols, model.feature_importances_), key=lambda x:-x[1])[:8]
            print(f"    Top: {', '.join(f'{n}={v:.3f}' for n,v in imps)}")
            results[name] = {'model': model, 'features': cols, 'threshold': best_thresh,
                             'direction': direction, 'wr': best_wr}

# ============ SAVE ============
print(f"\n{'='*60}")
print("RESULTS & SAVING")
print(f"{'='*60}")
os.makedirs('models/tick_models_v3', exist_ok=True)
for name, info in results.items():
    print(f"  {name}: {info['wr']:.0f}% WR @ >{info['threshold']:.0%}, {len(info['features'])} features")
    joblib.dump(info['model'], f'models/tick_models_v3/{name}.pkl')
    cfg = {'name': name, 'direction': info['direction'], 'features': info['features'],
           'threshold': info['threshold']}
    joblib.dump(cfg, f'models/tick_models_v3/{name}_config.pkl')

joblib.dump({'features': all_cols, 'bar_seconds': 16, 'enriched': True,
             'extra_cols': extra_cols}, 'models/tick_models_v3/feature_info.pkl')
print(f"\nTotal time: {time.time()-t0:.0f}s")
print("DONE")
