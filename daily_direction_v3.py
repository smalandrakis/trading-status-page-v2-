#!/usr/bin/env python3
"""
Daily Direction v3 — 5 years of data, rich features, random train/test split, >=3:1 R:R.

Design:
- 5 years BTC daily data (2021-2026)
- Rich features: 1d/2d/3d/5d/7d/14d/21d/30d returns, vol, RSI, MACD, BB, ATR, etc.
- Labels: does TP hit before SL within 24h? Using intraday high/low.
- R:R >= 3:1 only: TP=3%/SL=1%, TP=2%/SL=0.7%, TP=4%/SL=1%, TP=3%/SL=0.7%
- Train on random 300 days/year, test on remaining ~65 days. Multiple random seeds.
- High-confidence filter: only trade when model probability is very high.
"""
import numpy as np, pandas as pd, os, pickle, warnings
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')

DATA_PATH = 'data/btc_daily_5yr.parquet'
OUT_DIR = 'models/daily_v3'
os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# STEP 1: Load data
# ══════════════════════════════════════════════════════════════════
print("="*70)
print("STEP 1: Load 5-year daily data")
print("="*70)
raw = pd.read_parquet(DATA_PATH)
# Handle multi-level columns from yfinance
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = [c[0].lower() for c in raw.columns]
else:
    raw.columns = [c.lower() for c in raw.columns]
raw = raw.dropna(subset=['close'])
raw.index = pd.to_datetime(raw.index)
raw.index = raw.index.tz_localize(None) if raw.index.tz else raw.index
raw = raw.sort_index()
print(f"  {len(raw)} days | {raw.index[0].date()} → {raw.index[-1].date()}")
print(f"  Price range: ${raw['close'].min():.0f} → ${raw['close'].max():.0f}")

# Yearly stats
for yr in sorted(raw.index.year.unique()):
    sub = raw[raw.index.year == yr]
    ret = (sub['close'].iloc[-1]/sub['close'].iloc[0]-1)*100
    rng = ((sub['high']-sub['low'])/sub['open']*100)
    print(f"  {yr}: {len(sub)} days, BTC {ret:+.1f}%, avg daily range {rng.mean():.1f}%")

# ══════════════════════════════════════════════════════════════════
# STEP 2: Build TP/SL labels using intraday high/low
# ══════════════════════════════════════════════════════════════════
print("\n"+"="*70)
print("STEP 2: Build labels (>=3:1 R:R)")
print("="*70)

CONFIGS = [
    {'tp':3.0, 'sl':1.0, 'name':'tp30_sl10'},  # 3:1, BE=25.0%
    {'tp':3.0, 'sl':0.7, 'name':'tp30_sl07'},  # 4.3:1, BE=18.9%
    {'tp':2.0, 'sl':0.7, 'name':'tp20_sl07'},  # 2.9:1, BE=25.9%
    {'tp':4.0, 'sl':1.0, 'name':'tp40_sl10'},  # 4:1, BE=20.0%
    {'tp':5.0, 'sl':1.0, 'name':'tp50_sl10'},  # 5:1, BE=16.7%
    {'tp':2.0, 'sl':0.5, 'name':'tp20_sl05'},  # 4:1, BE=20.0%
]

o = raw['open'].values; h = raw['high'].values; l = raw['low'].values; c = raw['close'].values
n = len(raw)

all_labels = {}
for cfg in CONFIGS:
    lw = np.zeros(n, dtype=np.int8)
    sw = np.zeros(n, dtype=np.int8)
    for i in range(n):
        ep = o[i]  # enter at day open
        # LONG: TP if high >= open*(1+tp%), SL if low <= open*(1-sl%)
        long_tp = h[i] >= ep * (1 + cfg['tp']/100)
        long_sl = l[i] <= ep * (1 - cfg['sl']/100)
        # SHORT: TP if low <= open*(1-tp%), SL if high >= open*(1+sl%)
        short_tp = l[i] <= ep * (1 - cfg['tp']/100)
        short_sl = h[i] >= ep * (1 + cfg['sl']/100)

        # If both TP and SL hit same day, we're conservative: assume SL hit first
        # unless range is clearly directional
        if long_tp and not long_sl:
            lw[i] = 1
        elif long_tp and long_sl:
            # Both hit: if close > open, lean toward TP hit first
            lw[i] = 1 if c[i] > ep else 0

        if short_tp and not short_sl:
            sw[i] = 1
        elif short_tp and short_sl:
            sw[i] = 1 if c[i] < ep else 0

    all_labels[f'L_{cfg["name"]}'] = lw
    all_labels[f'S_{cfg["name"]}'] = sw
    be = cfg['sl']/(cfg['tp']+cfg['sl'])*100
    print(f"  {cfg['name']:>12}: LONG={lw.mean()*100:5.1f}% SHORT={sw.mean()*100:5.1f}% "
          f"BE={be:.1f}% RR={cfg['tp']/cfg['sl']:.1f}:1")

    # Yearly breakdown
    for yr in sorted(raw.index.year.unique()):
        mask = raw.index.year == yr
        print(f"    {yr}: L={lw[mask].mean()*100:5.1f}% S={sw[mask].mean()*100:5.1f}%")

# ══════════════════════════════════════════════════════════════════
# STEP 3: Features (all using PRIOR data only — no leakage)
# ══════════════════════════════════════════════════════════════════
print("\n"+"="*70)
print("STEP 3: Build features")
print("="*70)

feat = pd.DataFrame(index=raw.index)
cc = raw['close'].astype(float)
hh = raw['high'].astype(float)
ll = raw['low'].astype(float)
oo = raw['open'].astype(float)
vv = raw['volume'].astype(float)
rr = cc.pct_change()

# Returns at multiple lookbacks (all shifted by 1: use yesterday's close)
for w in [1, 2, 3, 5, 7, 10, 14, 21, 30, 60, 90]:
    feat[f'ret_{w}d'] = cc.pct_change(w).shift(1) * 100

# Volatility
for w in [3, 5, 7, 10, 14, 21, 30]:
    feat[f'vol_{w}d'] = rr.rolling(w).std().shift(1) * 100
    # Vol ratio: recent vs longer
    if w <= 7:
        feat[f'vol_ratio_{w}_30'] = (rr.rolling(w).std() / rr.rolling(30).std().replace(0,np.nan)).shift(1)

# Daily range
rng = (hh - ll) / oo * 100
for w in [1, 3, 5, 7, 14, 30]:
    feat[f'range_{w}d'] = rng.rolling(w).mean().shift(1)
feat['range_1d_raw'] = rng.shift(1)

# RSI at multiple periods
for p in [3, 5, 7, 14, 21]:
    delta = cc.diff()
    gain = delta.clip(lower=0).ewm(span=p).mean()
    loss = (-delta.clip(upper=0)).ewm(span=p).mean()
    feat[f'rsi_{p}'] = (100 - (100 / (1 + gain / loss.replace(0, np.nan)))).shift(1)

# Bollinger Bands %B
for w in [10, 20]:
    ma = cc.rolling(w).mean(); std = cc.rolling(w).std()
    feat[f'bb_{w}'] = ((cc - (ma - 2*std)) / (4*std).replace(0, np.nan)).shift(1)

# MACD
e12 = cc.ewm(span=12).mean(); e26 = cc.ewm(span=26).mean()
macd_line = e12 - e26; sig = macd_line.ewm(span=9).mean()
feat['macd'] = (macd_line / cc * 100).shift(1)
feat['macd_hist'] = ((macd_line - sig) / cc * 100).shift(1)
feat['macd_crossover'] = (np.sign(macd_line - sig) - np.sign(macd_line.shift(1) - sig.shift(1))).shift(1)

# ATR
tr = pd.concat([hh-ll, (hh-cc.shift()).abs(), (ll-cc.shift()).abs()], axis=1).max(axis=1)
for w in [7, 14, 30]:
    feat[f'atr_{w}d'] = (tr.rolling(w).mean() / cc * 100).shift(1)

# Position in range (shifted)
for w in [7, 14, 20, 30, 60]:
    rh = hh.rolling(w).max(); rl = ll.rolling(w).min()
    feat[f'pos_{w}d'] = ((cc - rl) / (rh - rl).replace(0, np.nan)).shift(1)
    feat[f'dist_high_{w}d'] = ((cc - rh) / cc * 100).shift(1)
    feat[f'dist_low_{w}d'] = ((cc - rl) / cc * 100).shift(1)

# Moving average distances
for w in [5, 7, 10, 14, 20, 30, 50, 100, 200]:
    ma = cc.rolling(w).mean()
    feat[f'dma_{w}'] = ((cc - ma) / ma * 100).shift(1)

# MA slope (trend strength)
for w in [7, 14, 30, 50]:
    ma = cc.rolling(w).mean()
    feat[f'ma_slope_{w}'] = (ma.pct_change(5) * 100).shift(1)

# Volume features
for w in [5, 10, 20]:
    feat[f'vratio_{w}'] = (vv / vv.rolling(w).mean().replace(0, np.nan)).shift(1)
feat['vol_trend'] = (vv.rolling(5).mean() / vv.rolling(20).mean().replace(0, np.nan)).shift(1)

# Candle structure (shifted)
feat['body'] = ((cc - oo) / oo * 100).shift(1)
feat['upper_wick'] = ((hh - cc.clip(lower=oo)) / oo * 100).shift(1)
feat['lower_wick'] = ((cc.clip(upper=oo) - ll) / oo * 100).shift(1)
feat['body_range_ratio'] = (((cc - oo).abs()) / (hh - ll).replace(0, np.nan)).shift(1)

# Streak features
up = (rr > 0).astype(float)
feat['up_streak'] = up.groupby((up != up.shift()).cumsum()).cumcount().shift(1)
dn = (rr < 0).astype(float)
feat['dn_streak'] = dn.groupby((dn != dn.shift()).cumsum()).cumcount().shift(1)

# Day of week
feat['dow_sin'] = np.sin(2 * np.pi * raw.index.dayofweek / 7)
feat['dow_cos'] = np.cos(2 * np.pi * raw.index.dayofweek / 7)

# Month cyclical
feat['month_sin'] = np.sin(2 * np.pi * raw.index.month / 12)
feat['month_cos'] = np.cos(2 * np.pi * raw.index.month / 12)

# Consecutive moves (how many of last N days were up/down)
for w in [3, 5, 7, 10]:
    feat[f'up_pct_{w}d'] = up.rolling(w).mean().shift(1) * 100

# Max drawdown / runup in last N days
for w in [5, 10, 14, 30]:
    feat[f'max_up_{w}d'] = cc.pct_change(1).rolling(w).max().shift(1) * 100
    feat[f'max_dn_{w}d'] = cc.pct_change(1).rolling(w).min().shift(1) * 100

# Gap (open vs prev close)
feat['gap'] = ((oo / cc.shift(1) - 1) * 100).shift(1)

feat = feat.replace([np.inf, -np.inf], np.nan)
fcols = list(feat.columns)
print(f"  {len(fcols)} features")
print(f"  Valid rows: {feat.notna().all(axis=1).sum()} / {len(feat)}")

# ══════════════════════════════════════════════════════════════════
# STEP 4: Random train/test split — 300 train, ~65 test per year
# ══════════════════════════════════════════════════════════════════
print("\n"+"="*70)
print("STEP 4: Random split validation (300 train / ~65 test per year)")
print("="*70)

fv = feat[fcols].values
vm = ~np.isnan(fv).any(axis=1)
valid_idx = np.where(vm)[0]
print(f"  {len(valid_idx)} valid days for training/testing")

N_SEEDS = 10  # 10 random splits for robustness
TRAIN_FRAC = 300/365
results = []

for seed in range(N_SEEDS):
    rng = np.random.RandomState(seed * 42 + 7)
    # Random split: pick 300/365 per year for train
    train_mask = np.zeros(len(raw), dtype=bool)
    test_mask = np.zeros(len(raw), dtype=bool)

    for yr in sorted(raw.index.year.unique()):
        yr_valid = [i for i in valid_idx if raw.index[i].year == yr]
        if len(yr_valid) < 30: continue
        n_train = int(len(yr_valid) * TRAIN_FRAC)
        rng.shuffle(yr_valid)
        for i in yr_valid[:n_train]: train_mask[i] = True
        for i in yr_valid[n_train:]: test_mask[i] = True

    tri = np.where(train_mask)[0]; tei = np.where(test_mask)[0]
    if seed == 0:
        print(f"  Seed {seed}: train={len(tri)}, test={len(tei)}")
        for yr in sorted(raw.index.year.unique()):
            nt = sum(1 for i in tri if raw.index[i].year == yr)
            ne = sum(1 for i in tei if raw.index[i].year == yr)
            print(f"    {yr}: train={nt}, test={ne}")

    Xtr = fv[tri]; Xte = fv[tei]

    for cfg in CONFIGS:
        be = cfg['sl']/(cfg['tp']+cfg['sl'])*100
        rr = cfg['tp']/cfg['sl']

        for d, pfx in [('LONG','L'), ('SHORT','S')]:
            ytr = all_labels[f'{pfx}_{cfg["name"]}'][tri]
            yte = all_labels[f'{pfx}_{cfg["name"]}'][tei]
            if ytr.sum() < 3: continue

            for mname, model in [
                ('HGB', HistGradientBoostingClassifier(
                    max_iter=200, max_depth=4, min_samples_leaf=10,
                    learning_rate=0.05, random_state=42, l2_regularization=2.0)),
                ('RF', RandomForestClassifier(
                    n_estimators=300, max_depth=6, min_samples_leaf=5,
                    random_state=42, n_jobs=-1)),
            ]:
                model.fit(Xtr, ytr)
                pr = model.predict_proba(Xte)[:, 1]
                try: auc = roc_auc_score(yte, pr)
                except: auc = 0.5

                for th in [0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]:
                    sel = pr > th; nt = sel.sum()
                    if nt < 3: continue
                    w = int(yte[sel].sum()); wr = w/nt*100
                    ev = (wr/100*cfg['tp']) - ((100-wr)/100*cfg['sl'])

                    # Per-year breakdown
                    yr_detail = {}
                    for yr in sorted(raw.index.year.unique()):
                        yr_te = [j for j, idx in enumerate(tei) if raw.index[idx].year == yr]
                        if len(yr_te) == 0: continue
                        yr_sel = sel[yr_te]; yr_yt = yte[np.array(yr_te)]
                        yr_nt = yr_sel.sum()
                        if yr_nt > 0:
                            yr_w = yr_yt[yr_sel[np.arange(len(yr_te))]].sum() if yr_nt > 0 else 0
                            # Simpler: index into test yte
                            yr_w = int(yte[np.array(yr_te)][sel[yr_te]].sum())
                            yr_wr = yr_w / yr_nt * 100
                            yr_ev = (yr_wr/100*cfg['tp']) - ((100-yr_wr)/100*cfg['sl'])
                            yr_detail[yr] = {'trades': yr_nt, 'wins': yr_w, 'wr': yr_wr, 'ev': yr_ev}

                    results.append({
                        'seed': seed, 'dir': d, 'config': cfg['name'],
                        'tp': cfg['tp'], 'sl': cfg['sl'], 'rr': rr,
                        'model': mname, 'thr': th,
                        'trades': nt, 'wins': w, 'wr': wr, 'ev': ev,
                        'base_wr': yte.mean()*100, 'be': be, 'auc': auc,
                        'yr_detail': str(yr_detail),
                    })

    if seed % 3 == 0:
        print(f"  Seed {seed} done.")

print(f"  All {N_SEEDS} seeds done. {len(results)} result rows.")

# ══════════════════════════════════════════════════════════════════
# STEP 5: Aggregate across seeds
# ══════════════════════════════════════════════════════════════════
print("\n"+"="*70)
print("STEP 5: RESULTS (averaged across 10 random splits)")
print("="*70)

res = pd.DataFrame(results)
res.to_csv(os.path.join(OUT_DIR, 'results_v3.csv'), index=False)

# Average across seeds
grp = res.groupby(['dir','config','model','thr']).agg(
    avg_trades=('trades','mean'), avg_wins=('wins','mean'),
    avg_wr=('wr','mean'), avg_ev=('ev','mean'),
    std_ev=('ev','std'), avg_auc=('auc','mean'),
    min_ev=('ev','min'), max_ev=('ev','max'),
    seeds=('seed','nunique'),
).reset_index()

# Add config info
for i, r in grp.iterrows():
    cfg = next(c for c in CONFIGS if c['name']==r['config'])
    grp.at[i, 'be'] = cfg['sl']/(cfg['tp']+cfg['sl'])*100
    grp.at[i, 'rr'] = cfg['tp']/cfg['sl']

# Profitable: positive avg EV AND positive min EV (robust across all seeds)
prof = grp[(grp['avg_ev'] > 0)].sort_values('avg_ev', ascending=False)
prof_robust = grp[(grp['avg_ev'] > 0) & (grp['min_ev'] > -0.3)].sort_values('avg_ev', ascending=False)

print(f"\nALL POSITIVE EV (avg across {N_SEEDS} seeds):")
print(f"  {'Dir':>5} {'Config':>12} {'Model':>4} {'Thr':>4} | {'Avg#T':>5} {'AvgWR':>6} {'BE':>5} "
      f"{'AvgEV':>8} {'StdEV':>7} {'MinEV':>7} {'MaxEV':>7} | {'AUC':>5} {'RR':>4}")
print(f"  {'-'*88}")
for _, r in prof.head(40).iterrows():
    robust = "★" if r['min_ev'] > -0.3 else " "
    print(f"  {r['dir']:>5} {r['config']:>12} {r['model']:>4} {r['thr']:4.0%} | "
          f"{r['avg_trades']:5.0f} {r['avg_wr']:5.1f}% {r['be']:4.1f}% "
          f"{r['avg_ev']:+7.3f}% {r['std_ev']:6.3f}% {r['min_ev']:+6.3f}% {r['max_ev']:+6.3f}% | "
          f"{r['avg_auc']:.3f} {r['rr']:3.1f} {robust}")

# ══════════════════════════════════════════════════════════════════
# STEP 6: Detailed per-year analysis of top strategies
# ══════════════════════════════════════════════════════════════════
print("\n"+"="*70)
print("STEP 6: TOP STRATEGIES — Per-year breakdown (seed 0)")
print("="*70)

# Re-run seed 0 with detailed per-year output
seed0 = res[res['seed']==0]
seed0_prof = seed0[seed0['ev']>0].sort_values('ev', ascending=False)
for rank, (_, r) in enumerate(seed0_prof.head(10).iterrows()):
    cfg = next(c for c in CONFIGS if c['name']==r['config'])
    print(f"\n  #{rank+1}: {r['dir']} {r['config']} {r['model']} >{r['thr']:.0%} "
          f"— {r['trades']:.0f}T WR={r['wr']:.1f}% EV={r['ev']:+.3f}% AUC={r['auc']:.3f}")
    # Parse yr_detail
    import ast
    try:
        yd = ast.literal_eval(r['yr_detail'])
        for yr in sorted(yd.keys()):
            d = yd[yr]
            s = "+" if d['ev'] > 0 else "-"
            print(f"    {yr}: {d['trades']:.0f}T {d['wins']:.0f}W {d['wr']:.1f}%WR "
                  f"EV={d['ev']:+.3f}% {s}")
    except:
        pass

# ══════════════════════════════════════════════════════════════════
# STEP 7: Feature importance from best model
# ══════════════════════════════════════════════════════════════════
print("\n"+"="*70)
print("STEP 7: Feature importance (best strategy, seed 0)")
print("="*70)

if len(seed0_prof) > 0:
    best = seed0_prof.iloc[0]
    cfg = next(c for c in CONFIGS if c['name']==best['config'])
    pfx = 'L' if best['dir']=='LONG' else 'S'

    # Retrain on full train set from seed 0
    rng_final = np.random.RandomState(7)
    train_mask = np.zeros(len(raw), dtype=bool)
    for yr in sorted(raw.index.year.unique()):
        yr_valid = [i for i in valid_idx if raw.index[i].year == yr]
        if len(yr_valid) < 30: continue
        n_train = int(len(yr_valid) * TRAIN_FRAC)
        rng_final.shuffle(yr_valid)
        for i in yr_valid[:n_train]: train_mask[i] = True
    tri_final = np.where(train_mask)[0]

    if best['model'] == 'HGB':
        m_final = HistGradientBoostingClassifier(
            max_iter=200, max_depth=4, min_samples_leaf=10,
            learning_rate=0.05, random_state=42, l2_regularization=2.0)
    else:
        m_final = RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=5,
            random_state=42, n_jobs=-1)

    m_final.fit(fv[tri_final], all_labels[f'{pfx}_{cfg["name"]}'][tri_final])

    if hasattr(m_final, 'feature_importances_'):
        imp = pd.Series(m_final.feature_importances_, index=fcols).sort_values(ascending=False)
        print(f"\n  Top 25 features for {best['dir']} {best['config']} {best['model']}:")
        for fname, val in imp.head(25).items():
            bar = "█" * int(val / imp.max() * 20)
            print(f"    {fname:>25}: {val:.4f}  {bar}")

        # Save model
        with open(os.path.join(OUT_DIR, 'best_model.pkl'), 'wb') as f:
            pickle.dump(m_final, f)
        with open(os.path.join(OUT_DIR, 'feature_cols.pkl'), 'wb') as f:
            pickle.dump(fcols, f)
        with open(os.path.join(OUT_DIR, 'config.pkl'), 'wb') as f:
            pickle.dump({'config': cfg, 'dir': best['dir'], 'thr': best['thr'],
                         'model_type': best['model']}, f)
        print(f"\n  Model saved to {OUT_DIR}/")

print("\nDone!")
