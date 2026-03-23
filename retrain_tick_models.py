#!/usr/bin/env python3
"""Retrain tick bot models with all available data (16 days)."""
import pandas as pd, numpy as np, joblib, os, warnings
from numba import njit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
warnings.filterwarnings('ignore')

# ============ LOAD ============
print("Loading...")
df = pd.read_csv('logs/btc_price_ticks.csv', parse_dates=['timestamp'])
df = df.set_index('timestamp').sort_index()
df = df[df['price']>0]
bars = df['price'].resample('16s').ohlc().dropna()
bars.columns = ['open','high','low','close']
close = bars['close']
print(f"16-sec bars: {len(bars)}, {bars.index[0].date()} to {bars.index[-1].date()}")

# ============ FAST LABELS (numba) ============
@njit
def compute_labels(prices, pct, max_look):
    n = len(prices)
    lw = np.zeros(n, dtype=np.int32)
    sw = np.zeros(n, dtype=np.int32)
    for i in range(n-1):
        e = prices[i]
        tp_up = e*(1+pct); sl_up = e*(1-pct)
        tp_dn = e*(1-pct); sl_dn = e*(1+pct)
        for j in range(i+1, min(i+max_look+1, n)):
            if prices[j] >= tp_up: lw[i]=1; break
            if prices[j] <= sl_up: break
        for j in range(i+1, min(i+max_look+1, n)):
            if prices[j] <= tp_dn: sw[i]=1; break
            if prices[j] >= sl_dn: break
    return lw, sw

print("Computing labels (numba-accelerated)...")
lw, sw = compute_labels(close.values, 0.005, 675)
print(f"LONG wins: {lw.sum()}/{len(lw)} ({lw.mean()*100:.1f}%)")
print(f"SHORT wins: {sw.sum()}/{len(sw)} ({sw.mean()*100:.1f}%)")

# ============ FEATURES ============
print("Building features...")
f = pd.DataFrame(index=close.index)
for lb in [1,2,4,8,15,30,60,120]:
    f[f'ret_{lb}'] = close.pct_change(lb)*100
ret1 = close.pct_change()*100
for w in [8,15,30,60]:
    f[f'vol_{w}'] = ret1.rolling(w).std()
for w in [30,60,120]:
    rmin=close.rolling(w).min(); rmax=close.rolling(w).max()
    rng=rmax-rmin
    f[f'chan_pos_{w}'] = (close-rmin)/rng.replace(0,np.nan)
ret_sign = np.sign(close.diff())
f['consec_dir'] = ret_sign.groupby((ret_sign!=ret_sign.shift()).cumsum()).cumcount()+1
f['consec_dir'] *= ret_sign
f['speed_8'] = close.diff(8).abs()/8
f['speed_30'] = close.diff(30).abs()/30
f['speed_ratio'] = f['speed_8']/f['speed_30'].replace(0,np.nan)
f['hour'] = close.index.hour
for span in [8,30,60]:
    ema = close.ewm(span=span).mean()
    f[f'ema_dist_{span}'] = (close-ema)/close*100
f['bar_range_8'] = (bars['high']-bars['low']).rolling(8).mean()/close*100
f['bar_range_30'] = (bars['high']-bars['low']).rolling(30).mean()/close*100

valid = f.dropna()
lw_v = lw[:len(valid)]; sw_v = sw[:len(valid)]
valid = valid.iloc[:len(lw_v)]
print(f"Valid: {len(valid)} bars, features: {len(valid.columns)}")

# ============ SPLIT 75/25 temporal ============
split = int(len(valid)*0.75)
X_tr, X_te = valid.iloc[:split], valid.iloc[split:]
yl_tr, yl_te = lw_v[:split], lw_v[split:]
ys_tr, ys_te = sw_v[:split], sw_v[split:]
print(f"Train: {len(X_tr)} ({X_tr.index[0].date()} to {X_tr.index[-1].date()})")
print(f"Test:  {len(X_te)} ({X_te.index[0].date()} to {X_te.index[-1].date()})")

# ============ TRAIN ============
results = {}
for name, y_tr, y_te in [('L1_full_wide', yl_tr, yl_te), ('L3_mom_wide', yl_tr, yl_te), ('S1_full_cur', ys_tr, ys_te)]:
    print(f"\n=== {name} ===")
    m = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                    min_samples_leaf=40, subsample=0.8, random_state=42)
    m.fit(X_tr, y_tr)
    probs = m.predict_proba(X_te)[:,1]
    
    for thresh in [0.50,0.55,0.60,0.65,0.70,0.75]:
        mask = probs >= thresh
        if mask.sum()==0: continue
        indices = np.where(mask)[0]
        taken=[]; last=-30
        for idx in indices:
            if idx-last>=30: taken.append(idx); last=idx
        if not taken: continue
        wins = sum(y_te[i] for i in taken)
        nt=len(taken); wr=wins/nt*100
        days=max((X_te.index[-1]-X_te.index[0]).days,1)
        tag = " <-- BEST" if wr >= 60 and nt >= 20 else ""
        print(f"  >{thresh:.0%}: {nt}T ({nt/days:.1f}/d) {wr:.0f}%WR {wins}W/{nt-wins}L{tag}")
    
    results[name] = m

    # Top features
    imps = sorted(zip(valid.columns, m.feature_importances_), key=lambda x:-x[1])[:5]
    print(f"  Top feats: {', '.join(f'{n}={v:.3f}' for n,v in imps)}")

# ============ SAVE ============
os.makedirs('models/tick_models_v2', exist_ok=True)
for name, m in results.items():
    joblib.dump(m, f'models/tick_models_v2/{name}.pkl')
    joblib.dump({'features': list(valid.columns), 'version': 'v2_16days'}, f'models/tick_models_v2/{name}_config.pkl')
joblib.dump({'features': list(valid.columns), 'bar_seconds': 16}, 'models/tick_models_v2/feature_info.pkl')
print("\nSaved to models/tick_models_v2/")
print("DONE")
