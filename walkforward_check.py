#!/usr/bin/env python3
"""Quick walk-forward robustness check for v4 LONG model."""
import pandas as pd, numpy as np, time
from sklearn.ensemble import HistGradientBoostingClassifier

df = pd.read_parquet('data/btc_1m_12mo.parquet')
df.index = df.index.tz_localize(None); df = df.sort_index()
p = df['close'].values.astype(np.float64)
n = len(p)
c = df['close'].astype(float); h = df['high'].astype(float)
l = df['low'].astype(float); o = df['open'].astype(float)
v = df['volume'].astype(float); r1 = c.pct_change()

# Features (same 75)
feat = pd.DataFrame(index=df.index)
for w in [1,2,3,5,10,15,30,60,120,240]: feat[f'ret_{w}'] = c.pct_change(w)*100
for w in [5,10,15,30,60,120,240]: feat[f'vol_{w}'] = r1.rolling(w).std()*100
feat['vol_ratio_5_60'] = feat['vol_5']/feat['vol_60'].replace(0,np.nan)
feat['vol_ratio_15_60'] = feat['vol_15']/feat['vol_60'].replace(0,np.nan)
feat['vol_ratio_30_120'] = feat['vol_30']/feat['vol_120'].replace(0,np.nan)
feat['vol_ratio_60_240'] = feat['vol_60']/feat['vol_240'].replace(0,np.nan)
for w in [5,15,30,60,120]: feat[f'vratio_{w}'] = v/v.rolling(w).mean().replace(0,np.nan)
for pp in [7,14,30,60]:
    delta=c.diff(); gain=delta.clip(lower=0).ewm(span=pp).mean()
    loss=(-delta.clip(upper=0)).ewm(span=pp).mean()
    feat[f'rsi_{pp}'] = 100-(100/(1+gain/loss.replace(0,np.nan)))
for w in [20,60,120]:
    ma=c.rolling(w).mean(); std=c.rolling(w).std()
    feat[f'bb_pct_{w}'] = (c-(ma-2*std))/(4*std).replace(0,np.nan)
for w in [14,30,60,120]:
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    feat[f'atr_pct_{w}'] = tr.rolling(w).mean()/c*100
for fast,slow in [(12,26),(5,15),(30,60)]:
    ef=c.ewm(span=fast).mean(); es=c.ewm(span=slow).mean(); d=ef-es; sig=d.ewm(span=9).mean()
    feat[f'macd_{fast}_{slow}'] = d/c*100; feat[f'macd_hist_{fast}_{slow}'] = (d-sig)/c*100
for w in [60,120,240,480]:
    rh=h.rolling(w).max(); rl=l.rolling(w).min()
    feat[f'pos_high_{w}']=(c-rh)/c*100; feat[f'pos_low_{w}']=(c-rl)/c*100
    feat[f'pos_range_{w}']=(c-rl)/(rh-rl).replace(0,np.nan)
feat['bar_range']=(h-l)/c*100; feat['bar_body']=(c-o).abs()/c*100
feat['upper_wick']=(h-c.clip(lower=o))/c*100; feat['lower_wick']=(c.clip(upper=o)-l)/c*100
feat['bar_range_ma30']=feat['bar_range'].rolling(30).mean()
feat['bar_range_ratio']=feat['bar_range']/feat['bar_range_ma30'].replace(0,np.nan)
for w in [30,60,120]: ma=c.rolling(w).mean(); feat[f'dist_ma_{w}']=(c-ma)/ma*100
feat['vol_price_corr_30']=r1.rolling(30).corr(v.pct_change())
feat['vol_price_corr_60']=r1.rolling(60).corr(v.pct_change())
feat['roc_5']=c.pct_change(5)*100; feat['roc_15']=c.pct_change(15)*100; feat['roc_60']=c.pct_change(60)*100
feat['up_count_10']=sum((r1.shift(i)>0).astype(int) for i in range(10))
feat['down_count_10']=10-feat['up_count_10']
hour=df.index.hour+df.index.minute/60
feat['hour_sin']=np.sin(2*np.pi*hour/24); feat['hour_cos']=np.cos(2*np.pi*hour/24)
dow=df.index.dayofweek
feat['dow_sin']=np.sin(2*np.pi*dow/7); feat['dow_cos']=np.cos(2*np.pi*dow/7)
feat = feat.replace([np.inf,-np.inf], np.nan)
feature_cols = list(feat.columns)

# Labels (vectorized)
def build_labels(prices, max_bars):
    nn = len(prices)
    ltp = np.full(nn, 9999, dtype=np.int32)
    lsl = np.full(nn, 9999, dtype=np.int32)
    for off in range(1, max_bars+1):
        idx = np.arange(nn)+off; valid = idx < nn
        fwd = np.where(valid, prices[np.clip(idx,0,nn-1)], np.nan)
        hit = valid & (fwd >= prices*1.01) & (ltp==9999); ltp[hit] = off
        hit = valid & (fwd <= prices*0.995) & (lsl==9999); lsl[hit] = off
    return ((ltp < lsl) & (ltp < 9999)).astype(np.int8)

print("Building labels...")
y_240 = build_labels(p, 240)
print(f"  240min base rate: {y_240.mean()*100:.1f}%")

# Walk-forward: 5 periods, train on expanding window, test on next period
valid_idx = feat.dropna().index
feat_c = feat.loc[valid_idx]
pos_map = np.searchsorted(df.index, valid_idx)
n_c = len(feat_c)
fold_size = n_c // 5

print(f"\nWalk-forward (4 folds, ~{fold_size:,} bars each)")
print(f"{'Fold':>4} {'Train period':>28} {'Test period':>28} | {'Thr':>4} {'#T':>5} {'W':>4} {'WR%':>6} {'EV%':>7}")
print("-" * 100)

results_by_thr = {0.60: [], 0.65: [], 0.70: [], 0.75: []}

for fold in range(4):
    tr_end = (fold + 1) * fold_size
    te_start = tr_end
    te_end = min(te_start + fold_size, n_c)

    X_tr = feat_c.iloc[:tr_end][feature_cols].fillna(0).values
    y_tr = y_240[pos_map[:tr_end]]
    X_te = feat_c.iloc[te_start:te_end][feature_cols].fillna(0).values
    y_te = y_240[pos_map[te_start:te_end]]

    tr_d = f"{feat_c.index[0].strftime('%Y-%m-%d')}..{feat_c.index[tr_end-1].strftime('%Y-%m-%d')}"
    te_d = f"{feat_c.index[te_start].strftime('%Y-%m-%d')}..{feat_c.index[te_end-1].strftime('%Y-%m-%d')}"

    t0 = time.time()
    m = HistGradientBoostingClassifier(max_iter=300, max_depth=5, min_samples_leaf=100, learning_rate=0.05, random_state=42)
    m.fit(X_tr, y_tr)
    probs = m.predict_proba(X_te)[:, 1]
    dt = time.time() - t0

    base_wr = y_te.mean() * 100
    print(f"  Fold {fold+1} train={tr_d} test={te_d} (base={base_wr:.1f}%, {dt:.0f}s)")

    for thr in [0.60, 0.65, 0.70, 0.75]:
        mask = probs > thr
        nt = mask.sum()
        if nt == 0:
            print(f"         >{thr:.0%}: 0 trades")
            continue
        w = int(y_te[mask].sum())
        wr = w / nt * 100
        ev = (wr/100*1.0) - ((100-wr)/100*0.5)
        results_by_thr[thr].append((nt, w))
        print(f"         >{thr:.0%}: {nt:5d}T {w:4d}W {wr:5.1f}% EV={ev:+.3f}%")

print("\n" + "=" * 60)
print("COMBINED ACROSS ALL FOLDS:")
print("=" * 60)
for thr in [0.60, 0.65, 0.70, 0.75]:
    if not results_by_thr[thr]:
        continue
    total_t = sum(x[0] for x in results_by_thr[thr])
    total_w = sum(x[1] for x in results_by_thr[thr])
    wr = total_w / total_t * 100
    ev = (wr/100*1.0) - ((100-wr)/100*0.5)
    print(f"  >{thr:.0%}: {total_t}T, {total_w}W, WR={wr:.1f}%, EV={ev:+.3f}%")
