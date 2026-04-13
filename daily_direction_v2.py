#!/usr/bin/env python3
"""
Daily Direction v2 — realistic TP/SL + direction picker + every day trades.

Key changes from v1:
1. Focus on TP=2%/SL=1% (23% base rate, 33.3% BE) — most realistic
2. Also TP=1.5%/SL=0.7% and TP=1%/SL=0.5% for higher frequency
3. Direction model: each day model picks LONG or SHORT, then apply TP/SL
4. Add intraday features from 1-min data (overnight move, Asian session, etc.)
5. Proper every-day simulation: 1 trade per day, always in market
"""
import numpy as np, pandas as pd, time, os
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score

DATA_PATH = 'data/btc_1m_12mo.parquet'
os.makedirs('models/daily_dir', exist_ok=True)

# ── Load ──
print("Loading...")
df = pd.read_parquet(DATA_PATH)
df.index = df.index.tz_localize(None) if df.index.tz else df.index
df = df.sort_index()

# Daily OHLCV
daily = df.resample('1D').agg({
    'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
daily['ret'] = daily['close'].pct_change()*100
print(f"  {len(daily)} days, {len(df):,} 1-min bars")

# ── Build labels on 1-min data ──
print("\nBuilding labels...")
ca=df['close'].values.astype(np.float64)
ha=df['high'].values.astype(np.float64)
la=df['low'].values.astype(np.float64)
nn=len(ca)

CONFIGS=[
    {'tp':1.0,'sl':0.5,'hz':1440,'name':'tp10_sl05'},  # BE=33.3%
    {'tp':1.5,'sl':0.7,'hz':1440,'name':'tp15_sl07'},  # BE=31.8%
    {'tp':2.0,'sl':1.0,'hz':1440,'name':'tp20_sl10'},  # BE=33.3%
    {'tp':2.0,'sl':0.7,'hz':1440,'name':'tp20_sl07'},  # BE=25.9%
    {'tp':3.0,'sl':1.0,'hz':1440,'name':'tp30_sl10'},  # BE=25.0%
]

# Get daily entry indices
doi=[]
for day in daily.index:
    idx=np.where((df.index>=day)&(df.index<day+pd.Timedelta(days=1)))[0]
    if len(idx)>0: doi.append(idx[0])
doi=np.array(doi)

all_labels={}
for cfg in CONFIGS:
    lw=np.zeros(len(doi),dtype=np.int8); sw=np.zeros(len(doi),dtype=np.int8)
    for di,ei in enumerate(doi):
        ep=ca[ei]; end=min(ei+cfg['hz'],nn)
        ltp=9999;lsl=9999;stp=9999;ssl=9999
        for j in range(ei+1,end):
            b=j-ei
            if ltp==9999 and ha[j]>=ep*(1+cfg['tp']/100): ltp=b
            if lsl==9999 and la[j]<=ep*(1-cfg['sl']/100): lsl=b
            if stp==9999 and la[j]<=ep*(1-cfg['tp']/100): stp=b
            if ssl==9999 and ha[j]>=ep*(1+cfg['sl']/100): ssl=b
            if ltp<9999 and lsl<9999 and stp<9999 and ssl<9999: break
        lw[di]=1 if ltp<lsl else 0
        sw[di]=1 if stp<ssl else 0
    all_labels[f'L_{cfg["name"]}']=lw; all_labels[f'S_{cfg["name"]}']=sw
    be=cfg['sl']/(cfg['tp']+cfg['sl'])*100
    print(f"  {cfg['name']:>12}: L={lw.mean()*100:.1f}% S={sw.mean()*100:.1f}% BE={be:.1f}%")

# Also: pure direction label (next day close > open?)
next_day_up = (daily['close'].values > daily['open'].values).astype(np.int8)
print(f"  Next day UP: {next_day_up.mean()*100:.1f}%")

# ── Features ──
print("\nBuilding features...")
feat=pd.DataFrame(index=daily.index)
c=daily['close'].astype(float); h=daily['high'].astype(float)
l=daily['low'].astype(float); o=daily['open'].astype(float)
v=daily['volume'].astype(float); r=c.pct_change()

# Daily price features (all shifted by 1 to use yesterday's close)
for w in [1,2,3,5,7,14,21,30]:
    feat[f'ret_{w}d']=c.pct_change(w).shift(1)*100
for w in [3,5,7,14,30]:
    feat[f'vol_{w}d']=r.rolling(w).std().shift(1)*100
for w in [3,5,7,14]:
    feat[f'rng_{w}d']=((h-l)/o*100).rolling(w).mean().shift(1)
for p in [7,14]:
    d=c.diff(); g=d.clip(lower=0).ewm(span=p).mean(); lo=(-d.clip(upper=0)).ewm(span=p).mean()
    feat[f'rsi_{p}']=(100-(100/(1+g/lo.replace(0,np.nan)))).shift(1)
ma20=c.rolling(20).mean(); std20=c.rolling(20).std()
feat['bb']=((c-(ma20-2*std20))/(4*std20).replace(0,np.nan)).shift(1)
e12=c.ewm(span=12).mean(); e26=c.ewm(span=26).mean()
md=e12-e26; sg=md.ewm(span=9).mean()
feat['macd']=(md/c*100).shift(1); feat['macd_h']=((md-sg)/c*100).shift(1)
for w in [10,20,30]:
    rh=h.rolling(w).max(); rl=l.rolling(w).min()
    feat[f'pos_{w}d']=((c-rl)/(rh-rl).replace(0,np.nan)).shift(1)
for w in [5,10]:
    feat[f'vr_{w}d']=(v/v.rolling(w).mean().replace(0,np.nan)).shift(1)
for w in [7,14,30]:
    ma=c.rolling(w).mean(); feat[f'dma_{w}d']=((c-ma)/ma*100).shift(1)
feat['prev_body']=((c-o)/o*100).shift(1)
feat['prev_range']=((h-l)/o*100).shift(1)
feat['prev_up_wick']=((h-c.clip(lower=o))/o*100).shift(1)
feat['prev_dn_wick']=((c.clip(upper=o)-l)/o*100).shift(1)
tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
feat['atr7']=(tr.rolling(7).mean()/c*100).shift(1)
feat['atr14']=(tr.rolling(14).mean()/c*100).shift(1)
up=(r>0).astype(float)
feat['up_streak']=up.groupby((up!=up.shift()).cumsum()).cumcount().shift(1)
dn=(r<0).astype(float)
feat['dn_streak']=dn.groupby((dn!=dn.shift()).cumsum()).cumcount().shift(1)
feat['dow_sin']=np.sin(2*np.pi*daily.index.dayofweek/7)
feat['dow_cos']=np.cos(2*np.pi*daily.index.dayofweek/7)

# Intraday features from 1-min: overnight gap, first-hour action
print("  Adding intraday features...")
for di in range(1, len(daily)):
    day = daily.index[di]; prev_day = daily.index[di-1]
    # Overnight gap: today's open vs yesterday's close
    feat.loc[day, 'overnight_gap'] = (daily.loc[day,'open']/daily.loc[prev_day,'close']-1)*100
    # First 60 min of previous day
    prev_start = df.index.searchsorted(prev_day)
    prev_end = min(prev_start + 60, len(df))
    if prev_end > prev_start:
        first_hr = df.iloc[prev_start:prev_end]
        feat.loc[day, 'prev_first_hr_ret'] = (first_hr['close'].iloc[-1]/first_hr['open'].iloc[0]-1)*100
        feat.loc[day, 'prev_first_hr_range'] = (first_hr['high'].max()-first_hr['low'].min())/first_hr['open'].iloc[0]*100
    # Last 60 min of previous day
    prev_day_end = df.index.searchsorted(prev_day + pd.Timedelta(days=1))
    prev_last_start = max(prev_day_end - 60, prev_start)
    if prev_day_end > prev_last_start:
        last_hr = df.iloc[prev_last_start:prev_day_end]
        if len(last_hr) > 0:
            feat.loc[day, 'prev_last_hr_ret'] = (last_hr['close'].iloc[-1]/last_hr['open'].iloc[0]-1)*100

feat=feat.replace([np.inf,-np.inf],np.nan)
fcols=list(feat.columns)
print(f"  {len(fcols)} features")

# ── Walk-forward: 3mo train, 1mo test ──
print("\n" + "="*70)
print("WALK-FORWARD VALIDATION")
print("="*70)

fv=feat[fcols].values; vm=~np.isnan(fv).any(axis=1)
months=pd.date_range(daily.index[0].replace(day=1),daily.index[-1],freq='MS')
TM=3; results=[]

for ti in range(TM, len(months)):
    ts=months[ti]; te=months[ti+1] if ti+1<len(months) else daily.index[-1]+pd.Timedelta(days=1)
    trs=months[ti-TM]
    tri=np.where((daily.index>=trs)&(daily.index<ts)&vm)[0]
    tei=np.where((daily.index>=ts)&(daily.index<te)&vm)[0]
    if len(tri)<20 or len(tei)<5: continue
    Xtr=fv[tri]; Xte=fv[tei]
    bret=(daily['close'].iloc[tei[-1]]/daily['close'].iloc[tei[0]]-1)*100
    mo=ts.strftime('%Y-%m')
    print(f"\n  {mo}: {len(tei)} days, BTC {bret:+.1f}%")

    for cfg in CONFIGS:
        be=cfg['sl']/(cfg['tp']+cfg['sl'])*100

        # === APPROACH A: Predict TP/SL outcome per direction ===
        for d,pfx in [('LONG','L'),('SHORT','S')]:
            ytr=all_labels[f'{pfx}_{cfg["name"]}'][tri]
            yte=all_labels[f'{pfx}_{cfg["name"]}'][tei]
            if ytr.sum()<2: continue
            for mname,model in [
                ('HGB',HistGradientBoostingClassifier(max_iter=100,max_depth=3,
                    min_samples_leaf=5,learning_rate=0.05,random_state=42,l2_regularization=2.0)),
                ('RF',RandomForestClassifier(n_estimators=200,max_depth=5,
                    min_samples_leaf=3,random_state=42,n_jobs=-1))]:
                model.fit(Xtr,ytr); pr=model.predict_proba(Xte)[:,1]
                try: auc=roc_auc_score(yte,pr)
                except: auc=0.5
                for th in [0.15,0.20,0.30,0.40,0.50]:
                    sel=pr>th; nt=sel.sum()
                    if nt<1: continue
                    w=int(yte[sel].sum()); wr=w/nt*100
                    ev=(wr/100*cfg['tp'])-((100-wr)/100*cfg['sl'])
                    results.append({'month':mo,'dir':d,'config':cfg['name'],
                        'tp':cfg['tp'],'sl':cfg['sl'],'model':mname,'thr':th,
                        'trades':nt,'wins':w,'wr':wr,'ev':ev,
                        'base_wr':yte.mean()*100,'be':be,'auc':auc,'btc':bret})

        # === APPROACH B: Direction picker — trade every day ===
        lw_tr=all_labels[f'L_{cfg["name"]}'][tri]; sw_tr=all_labels[f'S_{cfg["name"]}'][tri]
        lw_te=all_labels[f'L_{cfg["name"]}'][tei]; sw_te=all_labels[f'S_{cfg["name"]}'][tei]
        # Target: 1 if LONG is better (TP hit, or both miss but close>open)
        y_dir = np.where(lw_tr==1, 1, np.where(sw_tr==1, 0,
                         (next_day_up[tri]).astype(int)))
        if y_dir.sum()<2 or (1-y_dir).sum()<2: continue
        m=HistGradientBoostingClassifier(max_iter=100,max_depth=3,min_samples_leaf=5,
            learning_rate=0.05,random_state=42,l2_regularization=2.0)
        m.fit(Xtr,y_dir)
        p_long=m.predict_proba(Xte)[:,1]
        # Every day: go LONG if p_long>0.5, else SHORT
        wins=0; nt=len(tei)
        for i in range(nt):
            if p_long[i]>0.5: wins+=lw_te[i]
            else: wins+=sw_te[i]
        wr=wins/nt*100; ev=(wr/100*cfg['tp'])-((100-wr)/100*cfg['sl'])
        results.append({'month':mo,'dir':'PICK','config':cfg['name'],
            'tp':cfg['tp'],'sl':cfg['sl'],'model':'DIR','thr':0.50,
            'trades':nt,'wins':wins,'wr':wr,'ev':ev,
            'base_wr':max(lw_te.mean(),sw_te.mean())*100,'be':be,'auc':0,'btc':bret})

        # Direction picker with confidence filter: only trade when confident
        for conf_th in [0.55, 0.60, 0.65]:
            confident = (p_long > conf_th) | (p_long < (1-conf_th))
            if confident.sum() < 1: continue
            wins2=0; nt2=confident.sum()
            for i in range(len(tei)):
                if not confident[i]: continue
                if p_long[i]>0.5: wins2+=lw_te[i]
                else: wins2+=sw_te[i]
            wr2=wins2/nt2*100; ev2=(wr2/100*cfg['tp'])-((100-wr2)/100*cfg['sl'])
            results.append({'month':mo,'dir':'PICK','config':cfg['name'],
                'tp':cfg['tp'],'sl':cfg['sl'],'model':f'DIR_c{int(conf_th*100)}','thr':conf_th,
                'trades':nt2,'wins':wins2,'wr':wr2,'ev':ev2,
                'base_wr':max(lw_te.mean(),sw_te.mean())*100,'be':be,'auc':0,'btc':bret})

# ── Results ──
print("\n\n"+"="*70)
print("AGGREGATE RESULTS")
print("="*70)
res=pd.DataFrame(results)
res.to_csv('models/daily_dir/walkforward_v2.csv',index=False)

grp=res.groupby(['dir','config','model','thr']).agg(
    tt=('trades','sum'),tw=('wins','sum'),mo=('month','nunique'),
    mp=('ev',lambda x:(x>0).sum()),auc=('auc','mean')).reset_index()
grp['wr']=grp['tw']/grp['tt']*100
for i,r in grp.iterrows():
    cfg=next(c for c in CONFIGS if c['name']==r['config'])
    grp.at[i,'ev']=(r['wr']/100*cfg['tp'])-((100-r['wr'])/100*cfg['sl'])
    grp.at[i,'be']=cfg['sl']/(cfg['tp']+cfg['sl'])*100

prof=grp[(grp['ev']>0)&(grp['tt']>=10)].sort_values('ev',ascending=False)
print(f"\nPROFITABLE (min 10T):")
print(f"  {'Dir':>5} {'Config':>12} {'Model':>8} {'Thr':>4} | {'#T':>4} {'WR':>6} {'BE':>5} "
      f"{'EV':>8} | {'AUC':>5} {'Mo':>2} {'P':>2}")
print(f"  {'-'*72}")
for _,r in prof.head(30).iterrows():
    print(f"  {r['dir']:>5} {r['config']:>12} {r['model']:>8} {r['thr']:4.0%} | "
          f"{r['tt']:4.0f} {r['wr']:5.1f}% {r['be']:4.1f}% {r['ev']:+7.3f}% | "
          f"{r['auc']:.3f} {r['mo']:2.0f} {r['mp']:2.0f}")

# Monthly detail top 5
for rank,(_,br) in enumerate(prof.head(5).iterrows()):
    sub=res[(res['dir']==br['dir'])&(res['config']==br['config'])&
            (res['model']==br['model'])&(res['thr']==br['thr'])].sort_values('month')
    cfg=next(c for c in CONFIGS if c['name']==br['config'])
    tt=sub['trades'].sum(); tw=sub['wins'].sum(); wr=tw/tt*100
    ev=(wr/100*cfg['tp'])-((100-wr)/100*cfg['sl'])
    print(f"\n  #{rank+1}: {br['dir']} {br['config']} {br['model']} >{br['thr']:.0%} — "
          f"{tt}T WR={wr:.1f}% EV={ev:+.3f}%")
    for _,r in sub.iterrows():
        s="+" if r['ev']>0 else "-"
        print(f"    {r['month']} BTC={r['btc']:+.1f}% {r['trades']:.0f}T {r['wins']:.0f}W "
              f"{r['wr']:.1f}%WR EV={r['ev']:+.3f}% {s}")

if len(prof)==0: print("\n  No profitable strategies found.")
print("\nDone!")
