#!/usr/bin/env python3
"""
Combo Optimizer: Find the best combination of daily LONG strategies
that maximizes trades + P&L with minimal overlap.

For each config+model+threshold combo, we know WHICH days it fires on.
We want to find the union of strategies that:
1. Adds the most unique trading days
2. Maintains positive EV on non-overlapping days
3. Maximizes total P&L
"""
import numpy as np, pandas as pd, os, pickle, warnings, ast
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from itertools import combinations
warnings.filterwarnings('ignore')

DATA_PATH = 'data/btc_daily_5yr.parquet'

# ── Load data ──
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
print(f"{n} days loaded")

# ── Build ALL labels ──
CONFIGS=[
    {'tp':3.0,'sl':1.0,'name':'tp30_sl10'},
    {'tp':3.0,'sl':0.7,'name':'tp30_sl07'},
    {'tp':2.0,'sl':0.7,'name':'tp20_sl07'},
    {'tp':4.0,'sl':1.0,'name':'tp40_sl10'},
    {'tp':5.0,'sl':1.0,'name':'tp50_sl10'},
    {'tp':2.0,'sl':0.5,'name':'tp20_sl05'},
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

# ── Build features (same as v3) ──
feat=pd.DataFrame(index=raw.index)
cc=raw['close'].astype(float); hh=raw['high'].astype(float)
ll=raw['low'].astype(float); oo=raw['open'].astype(float)
vv=raw['volume'].astype(float); rr=cc.pct_change()

for w in [1,2,3,5,7,10,14,21,30,60,90]:
    feat[f'ret_{w}d']=cc.pct_change(w).shift(1)*100
for w in [3,5,7,10,14,21,30]:
    feat[f'vol_{w}d']=rr.rolling(w).std().shift(1)*100
    if w<=7: feat[f'vol_ratio_{w}_30']=(rr.rolling(w).std()/rr.rolling(30).std().replace(0,np.nan)).shift(1)
rng=(hh-ll)/oo*100
for w in [1,3,5,7,14,30]:
    feat[f'range_{w}d']=rng.rolling(w).mean().shift(1)
feat['range_1d_raw']=rng.shift(1)
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
valid_idx=np.where(vm)[0]
print(f"{len(fcols)} features, {len(valid_idx)} valid days")

# ── Run all strategies on ONE seed, capture per-day signals ──
print("\n" + "="*70)
print("Running all strategies, capturing per-day trade signals...")
print("="*70)

SEED=42; TRAIN_FRAC=300/365
rng_s=np.random.RandomState(SEED)
train_mask=np.zeros(n,dtype=bool); test_mask=np.zeros(n,dtype=bool)
for yr in sorted(raw.index.year.unique()):
    yr_valid=[i for i in valid_idx if raw.index[i].year==yr]
    if len(yr_valid)<30: continue
    n_train=int(len(yr_valid)*TRAIN_FRAC)
    rng_s.shuffle(yr_valid)
    for i in yr_valid[:n_train]: train_mask[i]=True
    for i in yr_valid[n_train:]: test_mask[i]=True

tri=np.where(train_mask)[0]; tei=np.where(test_mask)[0]
Xtr=fv[tri]; Xte=fv[tei]
print(f"Train: {len(tri)}, Test: {len(tei)}")

# For each strategy, record which TEST days it fires and whether it wins
strategies=[]
THRESHOLDS=[0.15,0.20,0.25,0.30,0.40,0.50,0.60]

for cfg in CONFIGS:
    ytr=all_labels[cfg['name']][tri]; yte=all_labels[cfg['name']][tei]
    if ytr.sum()<3: continue
    be=cfg['sl']/(cfg['tp']+cfg['sl'])*100

    for mname,model in [
        ('HGB',HistGradientBoostingClassifier(max_iter=200,max_depth=4,min_samples_leaf=10,
            learning_rate=0.05,random_state=42,l2_regularization=2.0)),
        ('RF',RandomForestClassifier(n_estimators=300,max_depth=6,min_samples_leaf=5,
            random_state=42,n_jobs=-1))]:
        model.fit(Xtr,ytr)
        pr=model.predict_proba(Xte)[:,1]
        try: auc=roc_auc_score(yte,pr)
        except: auc=0.5

        for th in THRESHOLDS:
            sel=pr>th
            nt=sel.sum()
            if nt<3: continue
            w=int(yte[sel].sum()); wr=w/nt*100
            ev=(wr/100*cfg['tp'])-((100-wr)/100*cfg['sl'])
            if ev<=0: continue  # only profitable strategies

            # Record which test days fire
            fire_days=set(np.where(sel)[0])  # indices into tei
            win_days=set(np.where(sel & (yte==1))[0])

            strategies.append({
                'name': f"L_{cfg['name']}_{mname}_{int(th*100)}",
                'config': cfg['name'], 'tp': cfg['tp'], 'sl': cfg['sl'],
                'model': mname, 'thr': th,
                'trades': nt, 'wins': w, 'wr': wr, 'ev': ev,
                'be': be, 'auc': auc,
                'fire_days': fire_days, 'win_days': win_days,
                'pnl_per_trade': ev,  # % per trade
            })

print(f"\n{len(strategies)} profitable strategies")

# ── Overlap analysis ──
print("\n" + "="*70)
print("OVERLAP ANALYSIS")
print("="*70)

# Sort by EV descending
strategies.sort(key=lambda x: -x['ev'])

# Show top 15
print(f"\n{'#':>2} {'Strategy':>30} | {'#T':>4} {'WR':>6} {'EV':>7} | Days fired")
print("-"*70)
for i,s in enumerate(strategies[:15]):
    print(f"{i+1:2} {s['name']:>30} | {s['trades']:4} {s['wr']:5.1f}% {s['ev']:+6.3f}% | {len(s['fire_days'])}")

# ── Greedy combo builder ──
print("\n" + "="*70)
print("GREEDY COMBO OPTIMIZER")
print("="*70)
print("Adding strategies one by one, maximizing new unique trades...")

# Greedy: pick strategy that adds most PROFITABLE unique days
used_days=set()
combo=[]
remaining=list(range(len(strategies)))

while remaining:
    best_idx=None; best_new_trades=0; best_new_wins=0; best_new_pnl=0; best_ev=0

    for si in remaining:
        s=strategies[si]
        new_fire = s['fire_days'] - used_days
        new_wins = s['win_days'] & new_fire
        if len(new_fire)==0: continue
        new_wr = len(new_wins)/len(new_fire)*100
        new_ev = (new_wr/100*s['tp']) - ((100-new_wr)/100*s['sl'])
        new_pnl = new_ev * len(new_fire)
        # Pick strategy with best marginal P&L from new days
        if new_pnl > best_new_pnl:
            best_idx=si; best_new_trades=len(new_fire)
            best_new_wins=len(new_wins); best_new_pnl=new_pnl; best_ev=new_ev

    if best_idx is None or best_new_trades < 1 or best_new_pnl <= 0:
        break

    s=strategies[best_idx]
    new_fire = s['fire_days'] - used_days
    new_wins = s['win_days'] & new_fire
    used_days |= new_fire
    combo.append({
        'strategy': s,
        'new_trades': len(new_fire),
        'new_wins': len(new_wins),
        'new_ev': best_ev,
        'cumul_trades': len(used_days),
    })
    remaining.remove(best_idx)

# ── Show combo results ──
print(f"\n{'#':>2} {'Strategy':>35} | {'NewT':>4} {'NewW':>4} {'NewWR':>6} {'NewEV':>7} | "
      f"{'CumT':>4} {'CumPnL':>8}")
print("-"*90)
cumul_pnl=0
for i,c in enumerate(combo):
    s=c['strategy']
    new_wr=c['new_wins']/c['new_trades']*100 if c['new_trades']>0 else 0
    trade_pnl = c['new_wins']*s['tp'] - (c['new_trades']-c['new_wins'])*s['sl']
    cumul_pnl += trade_pnl
    print(f"{i+1:2} {s['name']:>35} | {c['new_trades']:4} {c['new_wins']:4} {new_wr:5.1f}% "
          f"{c['new_ev']:+6.3f}% | {c['cumul_trades']:4} {cumul_pnl:+7.2f}%")

# ── Final combined stats ──
print("\n" + "="*70)
print("FINAL COMBINED STRATEGY")
print("="*70)

total_days=len(used_days)
# Reconstruct all wins from combo
all_wins=set()
all_trades_detail=[]
for c in combo:
    s=c['strategy']
    new_fire=s['fire_days']-set().union(*[cc['strategy']['fire_days']-used_days for cc in combo[:0]] if False else [set()])
    # Simpler: just track by adding
    pass

# Re-simulate the combo properly
used=set(); total_t=0; total_w=0; total_pnl=0
yearly={}
for c in combo:
    s=c['strategy']
    new_fire = s['fire_days'] - used
    new_wins = s['win_days'] & new_fire
    for d in new_fire:
        actual_day_idx = tei[d]
        yr = raw.index[actual_day_idx].year
        won = d in new_wins
        pnl = s['tp'] if won else -s['sl']
        if yr not in yearly: yearly[yr]={'trades':0,'wins':0,'pnl':0}
        yearly[yr]['trades']+=1
        yearly[yr]['wins']+=int(won)
        yearly[yr]['pnl']+=pnl
        total_t+=1; total_w+=int(won); total_pnl+=pnl
    used |= new_fire

total_wr=total_w/total_t*100 if total_t>0 else 0
print(f"\n  Total test trades: {total_t}")
print(f"  Total wins: {total_w}")
print(f"  Win rate: {total_wr:.1f}%")
print(f"  Total P&L: {total_pnl:+.2f}% (sum of individual trade %)")
print(f"  Avg P&L per trade: {total_pnl/total_t:+.3f}%")
print(f"  Trading frequency: ~{total_t/5:.0f} trades/year ({total_t} over ~5yr test set)")
print(f"  Strategies in combo: {len(combo)}")

print(f"\n  Per-year breakdown:")
print(f"  {'Year':>6} {'Trades':>6} {'Wins':>5} {'WR':>6} {'P&L':>8}")
print(f"  {'-'*35}")
for yr in sorted(yearly.keys()):
    y=yearly[yr]
    wr=y['wins']/y['trades']*100 if y['trades']>0 else 0
    print(f"  {yr:>6} {y['trades']:6} {y['wins']:5} {wr:5.1f}% {y['pnl']:+7.2f}%")

print(f"\n  Strategies used:")
for i,c in enumerate(combo):
    s=c['strategy']
    print(f"    {i+1}. {s['name']} — {c['new_trades']}T added, EV={c['new_ev']:+.3f}%")

# ── Also show: what if we just use the SINGLE best high-frequency strategy? ──
print("\n" + "="*70)
print("COMPARISON: Single best strategies")
print("="*70)

for s in sorted(strategies, key=lambda x: -x['trades']*x['ev'])[:5]:
    pnl = s['wins']*s['tp'] - (s['trades']-s['wins'])*s['sl']
    print(f"  {s['name']:>35}: {s['trades']}T {s['wr']:.1f}%WR EV={s['ev']:+.3f}% P&L={pnl:+.2f}%")

print("\nDone!")
