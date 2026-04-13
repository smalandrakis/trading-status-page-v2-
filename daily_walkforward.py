#!/usr/bin/env python3
"""
Walk-forward validation for daily LONG TP=2%/SL=0.5% strategy.
Rolling 1-year train → 3-month test, strictly chronological.
Also tests nearby configs to see which is most robust forward.
"""
import numpy as np, pandas as pd, os, warnings
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')

DATA_PATH = 'data/btc_daily_5yr.parquet'

# ── Load ──
raw = pd.read_parquet(DATA_PATH)
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = [c[0].lower() for c in raw.columns]
else:
    raw.columns = [c.lower() for c in raw.columns]
raw = raw.dropna(subset=['close']).sort_index()
raw.index = pd.to_datetime(raw.index)
raw.index = raw.index.tz_localize(None) if raw.index.tz else raw.index
o=raw['open'].values; h=raw['high'].values; l=raw['low'].values; c=raw['close'].values
n=len(raw)
print(f"{n} days | {raw.index[0].date()} → {raw.index[-1].date()}")

# ── Labels ──
CONFIGS=[
    {'tp':2.0,'sl':0.5,'name':'tp20_sl05'},
    {'tp':3.0,'sl':0.7,'name':'tp30_sl07'},
    {'tp':3.0,'sl':1.0,'name':'tp30_sl10'},
    {'tp':2.0,'sl':0.7,'name':'tp20_sl07'},
]
all_labels={}
for cfg in CONFIGS:
    lw=np.zeros(n,dtype=np.int8)
    for i in range(n):
        ep=o[i]
        tp_hit=h[i]>=ep*(1+cfg['tp']/100)
        sl_hit=l[i]<=ep*(1-cfg['sl']/100)
        if tp_hit and not sl_hit: lw[i]=1
        elif tp_hit and sl_hit: lw[i]=1 if c[i]>ep else 0
    all_labels[cfg['name']]=lw
    be=cfg['sl']/(cfg['tp']+cfg['sl'])*100
    print(f"  {cfg['name']}: base={lw.mean()*100:.1f}% BE={be:.1f}%")

# ── Features ──
feat=pd.DataFrame(index=raw.index)
cc=raw['close'].astype(float); hh=raw['high'].astype(float)
ll=raw['low'].astype(float); oo=raw['open'].astype(float)
vv=raw['volume'].astype(float); rr=cc.pct_change()

for w in [1,2,3,5,7,10,14,21,30,60,90]:
    feat[f'ret_{w}d']=cc.pct_change(w).shift(1)*100
for w in [3,5,7,10,14,21,30]:
    feat[f'vol_{w}d']=rr.rolling(w).std().shift(1)*100
    if w<=7: feat[f'vol_ratio_{w}_30']=(rr.rolling(w).std()/rr.rolling(30).std().replace(0,np.nan)).shift(1)
rng_s=(hh-ll)/oo*100
for w in [1,3,5,7,14,30]:
    feat[f'range_{w}d']=rng_s.rolling(w).mean().shift(1)
feat['range_1d_raw']=rng_s.shift(1)
for p in [3,5,7,14,21]:
    delta=cc.diff(); gain=delta.clip(lower=0).ewm(span=p).mean()
    loss=(-delta.clip(upper=0)).ewm(span=p).mean()
    feat[f'rsi_{p}']=(100-(100/(1+gain/loss.replace(0,np.nan)))).shift(1)
for w in [10,20]:
    ma=cc.rolling(w).mean(); std=cc.rolling(w).std()
    feat[f'bb_{w}']=((cc-(ma-2*std))/(4*std).replace(0,np.nan)).shift(1)
e12=cc.ewm(span=12).mean(); e26=cc.ewm(span=26).mean()
md=e12-e26; sg=md.ewm(span=9).mean()
feat['macd']=(md/cc*100).shift(1); feat['macd_hist']=((md-sg)/cc*100).shift(1)
feat['macd_crossover']=(np.sign(md-sg)-np.sign(md.shift(1)-sg.shift(1))).shift(1)
tr=pd.concat([hh-ll,(hh-cc.shift()).abs(),(ll-cc.shift()).abs()],axis=1).max(axis=1)
for w in [7,14,30]: feat[f'atr_{w}d']=(tr.rolling(w).mean()/cc*100).shift(1)
for w in [7,14,20,30,60]:
    rh=hh.rolling(w).max(); rl=ll.rolling(w).min()
    feat[f'pos_{w}d']=((cc-rl)/(rh-rl).replace(0,np.nan)).shift(1)
    feat[f'dist_high_{w}d']=((cc-rh)/cc*100).shift(1)
    feat[f'dist_low_{w}d']=((cc-rl)/cc*100).shift(1)
for w in [5,7,10,14,20,30,50,100,200]:
    ma=cc.rolling(w).mean(); feat[f'dma_{w}']=((cc-ma)/ma*100).shift(1)
for w in [7,14,30,50]:
    ma=cc.rolling(w).mean(); feat[f'ma_slope_{w}']=(ma.pct_change(5)*100).shift(1)
for w in [5,10,20]: feat[f'vratio_{w}']=(vv/vv.rolling(w).mean().replace(0,np.nan)).shift(1)
feat['vol_trend']=(vv.rolling(5).mean()/vv.rolling(20).mean().replace(0,np.nan)).shift(1)
feat['body']=((cc-oo)/oo*100).shift(1)
feat['upper_wick']=((hh-cc.clip(lower=oo))/oo*100).shift(1)
feat['lower_wick']=((cc.clip(upper=oo)-ll)/oo*100).shift(1)
feat['body_range_ratio']=(((cc-oo).abs())/(hh-ll).replace(0,np.nan)).shift(1)
up=(rr>0).astype(float)
feat['up_streak']=up.groupby((up!=up.shift()).cumsum()).cumcount().shift(1)
dn=(rr<0).astype(float)
feat['dn_streak']=dn.groupby((dn!=dn.shift()).cumsum()).cumcount().shift(1)
feat['dow_sin']=np.sin(2*np.pi*raw.index.dayofweek/7)
feat['dow_cos']=np.cos(2*np.pi*raw.index.dayofweek/7)
feat['month_sin']=np.sin(2*np.pi*raw.index.month/12)
feat['month_cos']=np.cos(2*np.pi*raw.index.month/12)
for w in [3,5,7,10]: feat[f'up_pct_{w}d']=up.rolling(w).mean().shift(1)*100
for w in [5,10,14,30]:
    feat[f'max_up_{w}d']=cc.pct_change(1).rolling(w).max().shift(1)*100
    feat[f'max_dn_{w}d']=cc.pct_change(1).rolling(w).min().shift(1)*100
feat['gap']=((oo/cc.shift(1)-1)*100).shift(1)
feat=feat.replace([np.inf,-np.inf],np.nan)
fcols=list(feat.columns)
fv=feat[fcols].values
vm=~np.isnan(fv).any(axis=1)
print(f"{len(fcols)} features")

# ══════════════════════════════════════════════════════════════════
# WALK-FORWARD: 1yr train → 3mo test, sliding quarterly
# ══════════════════════════════════════════════════════════════════
print("\n"+"="*70)
print("WALK-FORWARD: 1yr train → 3mo test")
print("="*70)

quarters = pd.date_range('2022-01-01', '2026-04-01', freq='QS')
TRAIN_DAYS = 365
results = []

for qi in range(len(quarters)-1):
    test_start = quarters[qi]
    test_end = quarters[qi+1]
    train_start = test_start - pd.Timedelta(days=TRAIN_DAYS)

    tri = np.where((raw.index>=train_start)&(raw.index<test_start)&vm)[0]
    tei = np.where((raw.index>=test_start)&(raw.index<test_end)&vm)[0]
    if len(tri)<60 or len(tei)<10: continue

    Xtr=fv[tri]; Xte=fv[tei]
    btc_ret=(c[tei[-1]]/c[tei[0]]-1)*100
    qname=f"{test_start.strftime('%Y-Q')}{(test_start.month-1)//3+1}"
    print(f"\n  {qname} ({test_start.date()}→{test_end.date()}): "
          f"{len(tri)} train, {len(tei)} test, BTC {btc_ret:+.1f}%")

    for cfg in CONFIGS:
        be=cfg['sl']/(cfg['tp']+cfg['sl'])*100
        ytr=all_labels[cfg['name']][tri]; yte=all_labels[cfg['name']][tei]
        base=yte.mean()*100

        for mname, model in [
            ('RF', RandomForestClassifier(n_estimators=300,max_depth=6,
                min_samples_leaf=5,random_state=42,n_jobs=-1)),
            ('HGB', HistGradientBoostingClassifier(max_iter=200,max_depth=4,
                min_samples_leaf=10,learning_rate=0.05,random_state=42,l2_regularization=2.0)),
        ]:
            model.fit(Xtr,ytr)
            pr=model.predict_proba(Xte)[:,1]
            try: auc=roc_auc_score(yte,pr)
            except: auc=0.5

            for th in [0.15,0.20,0.25,0.30,0.40,0.50]:
                sel=pr>th; nt=sel.sum()
                if nt<1: continue
                w=int(yte[sel].sum()); wr=w/nt*100
                ev=(wr/100*cfg['tp'])-((100-wr)/100*cfg['sl'])
                pnl=w*cfg['tp']-(nt-w)*cfg['sl']
                results.append({
                    'quarter':qname,'test_start':test_start,'dir':'LONG',
                    'config':cfg['name'],'tp':cfg['tp'],'sl':cfg['sl'],
                    'model':mname,'thr':th,
                    'trades':nt,'wins':w,'wr':wr,'ev':ev,'pnl':pnl,
                    'base':base,'be':be,'auc':auc,'btc':btc_ret,
                })

# ══════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════
print("\n\n"+"="*70)
print("WALK-FORWARD AGGREGATE RESULTS")
print("="*70)

res=pd.DataFrame(results)

grp=res.groupby(['config','model','thr']).agg(
    tt=('trades','sum'),tw=('wins','sum'),
    total_pnl=('pnl','sum'),
    nq=('quarter','nunique'),
    qprof=('pnl',lambda x:(x>0).sum()),
    avg_auc=('auc','mean'),
).reset_index()
grp['wr']=grp['tw']/grp['tt']*100
for i,r in grp.iterrows():
    cfg=next(cc for cc in CONFIGS if cc['name']==r['config'])
    grp.at[i,'ev']=(r['wr']/100*cfg['tp'])-((100-r['wr'])/100*cfg['sl'])
    grp.at[i,'be']=cfg['sl']/(cfg['tp']+cfg['sl'])*100

prof=grp[grp['ev']>0].sort_values('total_pnl',ascending=False)
print(f"\nPROFITABLE (sorted by total P&L):")
print(f"  {'Config':>12} {'Mdl':>4} {'Thr':>4} | {'#T':>4} {'WR':>6} {'BE':>5} {'EV':>7} "
      f"{'P&L':>8} | {'AUC':>5} {'Q':>2} {'QP':>2}")
print(f"  {'-'*72}")
for _,r in prof.head(25).iterrows():
    print(f"  {r['config']:>12} {r['model']:>4} {r['thr']:4.0%} | {r['tt']:4.0f} {r['wr']:5.1f}% "
          f"{r['be']:4.1f}% {r['ev']:+6.3f}% {r['total_pnl']:+7.2f}% | "
          f"{r['avg_auc']:.3f} {r['nq']:2.0f} {r['qprof']:2.0f}")

# ── Quarterly detail for top 5 strategies ──
print("\n"+"="*70)
print("QUARTERLY DETAIL — Top strategies")
print("="*70)

for rank,(_,br) in enumerate(prof.head(5).iterrows()):
    sub=res[(res['config']==br['config'])&(res['model']==br['model'])&
            (res['thr']==br['thr'])].sort_values('test_start')
    cfg=next(cc for cc in CONFIGS if cc['name']==br['config'])
    tt=sub['trades'].sum(); tw=sub['wins'].sum()
    wr=tw/tt*100; ev=(wr/100*cfg['tp'])-((100-wr)/100*cfg['sl'])
    tpnl=sub['pnl'].sum()
    print(f"\n  #{rank+1}: LONG {br['config']} {br['model']} >{br['thr']:.0%} — "
          f"{tt}T {tw}W WR={wr:.1f}% P&L={tpnl:+.1f}%")
    cum_pnl=0
    for _,r in sub.iterrows():
        cum_pnl+=r['pnl']
        s="+" if r['pnl']>0 else "-"
        print(f"    {r['quarter']:>8} BTC={r['btc']:+5.1f}% base={r['base']:4.1f}% | "
              f"{r['trades']:.0f}T {r['wins']:.0f}W {r['wr']:5.1f}%WR "
              f"PnL={r['pnl']:+6.2f}% cum={cum_pnl:+7.2f}% {s}")

if len(prof)==0:
    print("\n  No profitable strategies in walk-forward.")

print("\nDone!")
