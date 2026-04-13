#!/usr/bin/env python3
"""
v7 — Multi-timeframe features (5m/15m/1h/4h) + multiple TP/SL configs + walk-forward.
"""
import numpy as np, pandas as pd, time, os, pickle
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

DATA_PATH = 'data/btc_1m_12mo.parquet'
OUT_DIR = 'models/tpsl_v7'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ──
print("Loading data...")
df = pd.read_parquet(DATA_PATH)
df.index = df.index.tz_localize(None) if df.index.tz else df.index
df = df.sort_index()
print(f"  {len(df):,} 1-min candles")

# ── Resample ──
print("Resampling...")
timeframes = {'1m': df}
for tf, rule in [('5m','5min'),('15m','15min'),('1h','1h'),('4h','4h')]:
    r = df.resample(rule).agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    timeframes[tf] = r
    print(f"  {tf}: {len(r):,} bars")

# ── Feature computation per TF ──
def compute_tf_features(ohlcv, px):
    c=ohlcv['close'].astype(float); h=ohlcv['high'].astype(float)
    l=ohlcv['low'].astype(float); o=ohlcv['open'].astype(float)
    v=ohlcv['volume'].astype(float); r=c.pct_change()
    f=pd.DataFrame(index=ohlcv.index)
    for w in [1,2,3,5,10,20]:
        f[f'{px}_ret_{w}']=c.pct_change(w)*100
    for w in [5,10,20]:
        f[f'{px}_vol_{w}']=r.rolling(w).std()*100
    for p in [7,14]:
        d=c.diff(); g=d.clip(lower=0).ewm(span=p).mean(); lo=(-d.clip(upper=0)).ewm(span=p).mean()
        f[f'{px}_rsi_{p}']=100-(100/(1+g/lo.replace(0,np.nan)))
    ma20=c.rolling(20).mean(); std20=c.rolling(20).std()
    f[f'{px}_bb']=( c-(ma20-2*std20))/(4*std20).replace(0,np.nan)
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    f[f'{px}_atr']=tr.rolling(14).mean()/c*100
    e12=c.ewm(span=12).mean(); e26=c.ewm(span=26).mean(); d=e12-e26; s=d.ewm(span=9).mean()
    f[f'{px}_macd']=d/c*100; f[f'{px}_macd_h']=(d-s)/c*100
    for w in [20,50]:
        rh=h.rolling(w).max(); rl=l.rolling(w).min()
        f[f'{px}_pos_{w}']=(c-rl)/(rh-rl).replace(0,np.nan)
        f[f'{px}_dhi_{w}']=(c-rh)/c*100; f[f'{px}_dlo_{w}']=(c-rl)/c*100
    f[f'{px}_brange']=(h-l)/c*100; f[f'{px}_bbody']=(c-o)/c*100
    f[f'{px}_vr5']=v/v.rolling(5).mean().replace(0,np.nan)
    f[f'{px}_vr20']=v/v.rolling(20).mean().replace(0,np.nan)
    bv=v.where(c>=o,0); sv=v.where(c<o,0)
    f[f'{px}_bsr5']=bv.rolling(5).sum()/sv.rolling(5).sum().replace(0,np.nan)
    for w in [10,20]:
        ma=c.rolling(w).mean(); f[f'{px}_dma_{w}']=(c-ma)/ma*100
    f[f'{px}_up5']=sum((r.shift(i)>0).astype(float) for i in range(5))
    return f.replace([np.inf,-np.inf],np.nan)

TF_SHIFT = {'1m':0, '5m':5, '15m':15, '1h':60, '4h':240}  # shift in 1-min bars

print("Computing features (with TF shifts to prevent look-ahead)...")
t0=time.time()
parts=[]
for tf,data in timeframes.items():
    tfeat=compute_tf_features(data,tf)
    if tf!='1m':
        # Shift by bar duration BEFORE ffill to prevent leakage
        tfeat=tfeat.reindex(df.index,method='ffill').shift(TF_SHIFT[tf])
    parts.append(tfeat)
    print(f"  {tf}: {len(tfeat.columns)} features (shifted {TF_SHIFT[tf]} bars)")

# Time features
tf=pd.DataFrame(index=df.index)
hr=df.index.hour+df.index.minute/60
tf['hour_sin']=np.sin(2*np.pi*hr/24); tf['hour_cos']=np.cos(2*np.pi*hr/24)
dw=df.index.dayofweek
tf['dow_sin']=np.sin(2*np.pi*dw/7); tf['dow_cos']=np.cos(2*np.pi*dw/7)
parts.append(tf)

feat=pd.concat(parts,axis=1).loc[df.index]
feature_cols=list(feat.columns)
print(f"  TOTAL: {len(feature_cols)} features ({time.time()-t0:.1f}s)")

# ── Labels ──
print("\nBuilding labels...")
ca=df['close'].values.astype(np.float64)
ha=df['high'].values.astype(np.float64)
la=df['low'].values.astype(np.float64)
nn=len(ca)

CONFIGS=[
    {'tp':1.0,'sl':0.5,'hz':240,'name':'tp10_sl05'},
    {'tp':1.0,'sl':1.0,'hz':480,'name':'tp10_sl10'},
    {'tp':2.0,'sl':1.0,'hz':960,'name':'tp20_sl10'},
    {'tp':3.0,'sl':0.7,'hz':1440,'name':'tp30_sl07'},
    {'tp':3.0,'sl':1.0,'hz':1440,'name':'tp30_sl10'},
]

all_labels={}
for cfg in CONFIGS:
    lt=np.full(nn,9999,dtype=np.int32); ls=np.full(nn,9999,dtype=np.int32)
    st=np.full(nn,9999,dtype=np.int32); ss=np.full(nn,9999,dtype=np.int32)
    for off in range(1,cfg['hz']+1):
        ix=np.arange(nn)+off; v=ix<nn
        fh=np.where(v,ha[np.clip(ix,0,nn-1)],np.nan)
        fl=np.where(v,la[np.clip(ix,0,nn-1)],np.nan)
        h=v&(fh>=ca*(1+cfg['tp']/100))&(lt==9999); lt[h]=off
        h=v&(fl<=ca*(1-cfg['sl']/100))&(ls==9999); ls[h]=off
        h=v&(fl<=ca*(1-cfg['tp']/100))&(st==9999); st[h]=off
        h=v&(fh>=ca*(1+cfg['sl']/100))&(ss==9999); ss[h]=off
    yl=((lt<ls)&(lt<9999)).astype(np.int8); ys=((st<ss)&(st<9999)).astype(np.int8)
    all_labels[f"L_{cfg['name']}"]=yl; all_labels[f"S_{cfg['name']}"]=ys
    be=cfg['sl']/(cfg['tp']+cfg['sl'])*100
    print(f"  {cfg['name']}: L={yl.mean()*100:.1f}% S={ys.mean()*100:.1f}% BE={be:.1f}%")

# ── Walk-forward ──
print("\nWalk-forward validation...")
vm=feat.notna().all(axis=1).values
fv=feat[feature_cols].fillna(0).values
mos=pd.date_range(df.index[0].replace(day=1),df.index[-1],freq='MS')
TM=3; results=[]

for ti in range(TM,len(mos)):
    ts=mos[ti]; te=mos[ti+1] if ti+1<len(mos) else df.index[-1]
    trs=mos[ti-TM]
    tri=np.where(((df.index>=trs)&(df.index<ts)&vm))[0]
    tei=np.where(((df.index>=ts)&(df.index<te)&vm))[0]
    if len(tri)<1000 or len(tei)<100: continue
    Xtr=fv[tri]; Xte=fv[tei]
    bret=(ca[tei[-1]]/ca[tei[0]]-1)*100
    print(f"  {ts.strftime('%Y-%m')}: BTC {bret:+.1f}%")

    for cfg in CONFIGS:
        for d,pfx in [('LONG','L'),('SHORT','S')]:
            k=f"{pfx}_{cfg['name']}"
            ytr=all_labels[k][tri]; yte=all_labels[k][tei]
            bw=yte.mean()*100; be=cfg['sl']/(cfg['tp']+cfg['sl'])*100
            m=HistGradientBoostingClassifier(max_iter=200,max_depth=4,min_samples_leaf=200,
                learning_rate=0.03,random_state=42,l2_regularization=1.0)
            m.fit(Xtr,ytr)
            pr=m.predict_proba(Xte)[:,1]
            try: auc=roc_auc_score(yte,pr)
            except: auc=0.5
            for th in [0.3,0.4,0.5,0.6,0.7]:
                sel=pr>th; nt=sel.sum()
                if nt<3: continue
                w=int(yte[sel].sum()); wr=w/nt*100
                ev=(wr/100*cfg['tp'])-((100-wr)/100*cfg['sl'])
                results.append({'month':ts.strftime('%Y-%m'),'dir':d,'config':cfg['name'],
                    'tp':cfg['tp'],'sl':cfg['sl'],'thr':th,'trades':nt,'wins':w,
                    'wr':wr,'ev':ev,'base_wr':bw,'be':be,'auc':auc,'btc_ret':bret})

# ── Results ──
print("\n"+"="*70)
print("RESULTS")
print("="*70)
res=pd.DataFrame(results)
res.to_csv(os.path.join(OUT_DIR,'walkforward_results.csv'),index=False)

grp=res.groupby(['dir','config','thr']).agg(
    tt=('trades','sum'),tw=('wins','sum'),mo=('month','nunique'),
    mp=('ev',lambda x:(x>0).sum()),auc=('auc','mean')).reset_index()
grp['wr']=grp['tw']/grp['tt']*100
for i,r in grp.iterrows():
    cfg=next(c for c in CONFIGS if c['name']==r['config'])
    grp.at[i,'ev']=(r['wr']/100*cfg['tp'])-((100-r['wr'])/100*cfg['sl'])
    grp.at[i,'be']=cfg['sl']/(cfg['tp']+cfg['sl'])*100

prof=grp[(grp['ev']>0)&(grp['tt']>=20)].sort_values('ev',ascending=False)
print(f"\nPROFITABLE (min 20T):")
print(f"  {'Dir':>5} {'Config':>12} {'Thr':>4} | {'#T':>5} {'WR':>6} {'BE':>5} {'EV':>7} | {'AUC':>5} {'Mo':>2} {'P':>2}")
print(f"  {'-'*65}")
for _,r in prof.head(25).iterrows():
    print(f"  {r['dir']:>5} {r['config']:>12} {r['thr']:4.0%} | {r['tt']:5.0f} {r['wr']:5.1f}% "
          f"{r['be']:4.1f}% {r['ev']:+6.3f}% | {r['auc']:.3f} {r['mo']:2.0f} {r['mp']:2.0f}")

# Monthly detail for top 3
for rank,(_, br) in enumerate(prof.head(3).iterrows()):
    sub=res[(res['dir']==br['dir'])&(res['config']==br['config'])&(res['thr']==br['thr'])].sort_values('month')
    cfg=next(c for c in CONFIGS if c['name']==br['config'])
    tt=sub['trades'].sum(); tw=sub['wins'].sum(); wr=tw/tt*100
    ev=(wr/100*cfg['tp'])-((100-wr)/100*cfg['sl'])
    print(f"\n  #{rank+1}: {br['dir']} {br['config']} >{br['thr']:.0%} — {tt}T WR={wr:.1f}% EV={ev:+.3f}%")
    for _,r in sub.iterrows():
        s="+" if r['ev']>0 else "-"
        print(f"    {r['month']} BTC={r['btc_ret']:+.1f}% {r['trades']:.0f}T {r['wins']:.0f}W "
              f"{r['wr']:.1f}%WR EV={r['ev']:+.3f}% {s}")

if len(prof)==0:
    print("\nNo profitable strategy found.")
else:
    print(f"\n  Models saved to {OUT_DIR}/")
    # Save feature cols
    with open(os.path.join(OUT_DIR,'feature_cols.pkl'),'wb') as f: pickle.dump(feature_cols,f)

print("\nDone!")
