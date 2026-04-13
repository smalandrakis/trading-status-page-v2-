#!/usr/bin/env python3
"""Check if v7 multi-TF results have look-ahead bias from higher TF bars."""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

df = pd.read_parquet('data/btc_1m_12mo.parquet')
df.index = df.index.tz_localize(None); df = df.sort_index()
ca = df['close'].values.astype(np.float64)
ha = df['high'].values.astype(np.float64)
la = df['low'].values.astype(np.float64)
nn = len(ca)

# SHORT TP=2%/SL=1% labels (top strategy from v7)
lt = np.full(nn,9999,dtype=np.int32); ls = np.full(nn,9999,dtype=np.int32)
for off in range(1,961):
    ix = np.arange(nn)+off; v = ix<nn
    fh = np.where(v, ha[np.clip(ix,0,nn-1)], np.nan)
    fl = np.where(v, la[np.clip(ix,0,nn-1)], np.nan)
    h = v&(fl<=ca*0.98)&(lt==9999); lt[h] = off
    h = v&(fh>=ca*1.01)&(ls==9999); ls[h] = off
y = ((lt<ls)&(lt<9999)).astype(np.int8)
print(f"SHORT tp20_sl10 base: {y.mean()*100:.1f}%")

# Simple feature builder
def make_feats(ohlcv, px):
    c=ohlcv['close'].astype(float); r=c.pct_change()
    f=pd.DataFrame(index=ohlcv.index)
    for w in [1,2,3,5,10,20]: f[f'{px}_ret_{w}']=c.pct_change(w)*100
    for w in [5,10,20]: f[f'{px}_vol_{w}']=r.rolling(w).std()*100
    for p in [7,14]:
        d=c.diff(); g=d.clip(lower=0).ewm(span=p).mean()
        lo=(-d.clip(upper=0)).ewm(span=p).mean()
        f[f'{px}_rsi_{p}']=100-(100/(1+g/lo.replace(0,np.nan)))
    return f.replace([np.inf,-np.inf],np.nan)

# 1-min features (no leakage possible)
f1m = make_feats(df, '1m')
# 4h features (potential leakage)
r4h = df.resample('4h').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
f4h = make_feats(r4h, '4h').reindex(df.index, method='ffill')
# 4h features SHIFTED by 240 bars (guaranteed no leakage)
f4h_safe = f4h.shift(240)

# Train/test split
train = (df.index >= '2025-12-01') & (df.index < '2026-03-01')
test = df.index >= '2026-03-01'

for name, feat_df in [('1m only', f1m), ('4h (raw ffill)', f4h), ('4h (shifted 4h)', f4h_safe)]:
    cols = list(feat_df.columns)
    vm = feat_df.notna().all(axis=1).values
    fv = feat_df[cols].fillna(0).values
    tri = np.where(train & vm)[0]; tei = np.where(test & vm)[0]
    if len(tri) < 100 or len(tei) < 100:
        print(f"{name}: not enough data"); continue
    m = HistGradientBoostingClassifier(max_iter=200, max_depth=4, min_samples_leaf=200,
        learning_rate=0.03, random_state=42, l2_regularization=1.0)
    m.fit(fv[tri], y[tri]); pr = m.predict_proba(fv[tei])[:,1]
    auc = roc_auc_score(y[tei], pr)
    for th in [0.5, 0.7]:
        sel = pr > th; nt = sel.sum()
        w = y[tei][sel].sum() if nt > 0 else 0
        wr = w/nt*100 if nt > 0 else 0
        print(f"  {name:>20}: AUC={auc:.4f}  >{th:.0%}: {nt:>5}T {w:>4}W {wr:5.1f}%WR")

print("\nIf '4h raw' >> '4h shifted', there IS look-ahead bias.")
