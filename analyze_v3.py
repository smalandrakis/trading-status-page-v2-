#!/usr/bin/env python3
import pandas as pd
res = pd.read_csv('models/daily_v3/results_v3.csv')
CONFIGS=[{'tp':3.0,'sl':1.0,'name':'tp30_sl10'},{'tp':3.0,'sl':0.7,'name':'tp30_sl07'},
    {'tp':2.0,'sl':0.7,'name':'tp20_sl07'},{'tp':4.0,'sl':1.0,'name':'tp40_sl10'},
    {'tp':5.0,'sl':1.0,'name':'tp50_sl10'},{'tp':2.0,'sl':0.5,'name':'tp20_sl05'}]
grp = res.groupby(['dir','config','model','thr']).agg(
    avg_t=('trades','mean'), avg_wr=('wr','mean'), avg_ev=('ev','mean'),
    std_ev=('ev','std'), min_ev=('ev','min'), max_ev=('ev','max'),
    avg_auc=('auc','mean')).reset_index()
for i,r in grp.iterrows():
    cfg=next(c for c in CONFIGS if c['name']==r['config'])
    grp.at[i,'be']=cfg['sl']/(cfg['tp']+cfg['sl'])*100
    grp.at[i,'rr']=cfg['tp']/cfg['sl']
rob = grp[(grp['avg_ev']>0)&(grp['min_ev']>-0.3)].sort_values('avg_ev',ascending=False)
print('ROBUST (avg_ev>0, worst seed >-0.3%):')
for _,r in rob.head(20).iterrows():
    print(f'{r["dir"]:>5} {r["config"]:>12} {r["model"]:>4} >{r["thr"]:.0%} | '
          f'{r["avg_t"]:4.0f}T {r["avg_wr"]:5.1f}%WR BE={r["be"]:4.1f}% '
          f'EV={r["avg_ev"]:+.3f}% std={r["std_ev"]:.3f} min={r["min_ev"]:+.3f} '
          f'AUC={r["avg_auc"]:.3f} {r["rr"]:.0f}:1')
print(f'Total robust: {len(rob)}')
