"""Train TP/SL model v3 — GBM + richer features + TP/SL grid on 12mo 1-min data.
All vectorized. Prints progress throughout.
"""
import numpy as np, pandas as pd, pickle, time, os, warnings
warnings.filterwarnings('ignore')

MODEL_DIR = 'models/tpsl_v3'
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Step 1: Load ──
print("=" * 70)
print("STEP 1: Loading data")
print("=" * 70)
df = pd.read_parquet('data/btc_1m_12mo.parquet')
print(f"  {len(df):,} candles | {df.index[0].date()} → {df.index[-1].date()}")
print(f"  ${df['close'].iloc[0]:,.0f} → ${df['close'].iloc[-1]:,.0f}")

# ── Step 2: Rich features ──
print("\n" + "=" * 70)
print("STEP 2: Computing features (60+)")
print("=" * 70)
t0 = time.time()
c = df['close']; h = df['high']; l = df['low']; v = df['volume']
r1 = c.pct_change()

feat = pd.DataFrame(index=df.index)

# Returns at multiple timeframes
for w in [1, 2, 3, 5, 10, 15, 30, 60, 120, 240]:
    feat[f'ret_{w}'] = c.pct_change(w) * 100

# Volatility
for w in [5, 10, 15, 30, 60, 120, 240]:
    feat[f'vol_{w}'] = r1.rolling(w).std() * 100

# Vol ratios (short/long)
feat['vol_ratio_5_60'] = feat['vol_5'] / feat['vol_60'].replace(0, np.nan)
feat['vol_ratio_15_60'] = feat['vol_15'] / feat['vol_60'].replace(0, np.nan)
feat['vol_ratio_30_120'] = feat['vol_30'] / feat['vol_120'].replace(0, np.nan)
feat['vol_ratio_60_240'] = feat['vol_60'] / feat['vol_240'].replace(0, np.nan)

# Volume features
for w in [5, 15, 30, 60, 120]:
    feat[f'vratio_{w}'] = v / v.rolling(w).mean().replace(0, np.nan)

# RSI at multiple periods
for p in [7, 14, 30, 60]:
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(span=p).mean()
    loss = (-delta.clip(upper=0)).ewm(span=p).mean()
    feat[f'rsi_{p}'] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

# Bollinger %B
for w in [20, 60, 120]:
    ma = c.rolling(w).mean()
    std = c.rolling(w).std()
    feat[f'bb_pct_{w}'] = (c - (ma - 2*std)) / (4*std).replace(0, np.nan)

# ATR %
for w in [14, 30, 60, 120]:
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    feat[f'atr_pct_{w}'] = tr.rolling(w).mean() / c * 100

# MACD variants
for fast, slow in [(12,26), (5,15), (30,60)]:
    ema_f = c.ewm(span=fast).mean()
    ema_s = c.ewm(span=slow).mean()
    diff = ema_f - ema_s
    sig = diff.ewm(span=9).mean()
    feat[f'macd_{fast}_{slow}'] = diff / c * 100
    feat[f'macd_hist_{fast}_{slow}'] = (diff - sig) / c * 100

# Price position
for w in [60, 120, 240, 480]:
    rh = h.rolling(w).max()
    rl = l.rolling(w).min()
    feat[f'pos_high_{w}'] = (c - rh) / c * 100
    feat[f'pos_low_{w}'] = (c - rl) / c * 100
    feat[f'pos_range_{w}'] = (c - rl) / (rh - rl).replace(0, np.nan)

# Bar microstructure
feat['bar_range'] = (h - l) / c * 100
feat['bar_body'] = (c - df['open']).abs() / c * 100
feat['upper_wick'] = (h - c.clip(lower=df['open'])) / c * 100
feat['lower_wick'] = (c.clip(upper=df['open']) - l) / c * 100
feat['bar_range_ma30'] = feat['bar_range'].rolling(30).mean()
feat['bar_range_ratio'] = feat['bar_range'] / feat['bar_range_ma30'].replace(0, np.nan)

# Trend strength
for w in [30, 60, 120]:
    ma = c.rolling(w).mean()
    feat[f'dist_ma_{w}'] = (c - ma) / ma * 100

# Volume-weighted features
feat['vol_price_corr_30'] = r1.rolling(30).corr(v.pct_change())
feat['vol_price_corr_60'] = r1.rolling(60).corr(v.pct_change())

# Momentum
feat['roc_5'] = c.pct_change(5) * 100
feat['roc_15'] = c.pct_change(15) * 100
feat['roc_60'] = c.pct_change(60) * 100

# Consecutive bars
feat['consec_up'] = (r1 > 0).astype(int)
for i in range(1, 10):
    feat['consec_up'] = feat['consec_up'] * ((r1 > 0).astype(int).shift(i).fillna(0) + feat['consec_up'])
# Simplify: just count of last 10 bars that were up
feat['up_count_10'] = sum((r1.shift(i) > 0).astype(int) for i in range(10))
feat['down_count_10'] = 10 - feat['up_count_10']
feat.drop(columns=['consec_up'], inplace=True)

# Hour of day (cyclical)
hour = df.index.hour + df.index.minute / 60
feat['hour_sin'] = np.sin(2 * np.pi * hour / 24)
feat['hour_cos'] = np.cos(2 * np.pi * hour / 24)

# Day of week
dow = df.index.dayofweek
feat['dow_sin'] = np.sin(2 * np.pi * dow / 7)
feat['dow_cos'] = np.cos(2 * np.pi * dow / 7)

feat = feat.replace([np.inf, -np.inf], np.nan).dropna()
cols = list(feat.columns)
print(f"  {len(feat):,} rows, {len(cols)} features in {time.time()-t0:.1f}s")

# ── Step 3: Labels for multiple TP/SL combos ──
print("\n" + "=" * 70)
print("STEP 3: Building labels for TP/SL grid")
print("=" * 70)
t0 = time.time()
p_arr = df['close'].reindex(feat.index).values
n = len(p_arr)

tpsl_combos = [(0.3, 0.3), (0.5, 0.3), (0.5, 0.5), (0.7, 0.3), (0.7, 0.5), (1.0, 0.5)]
MAX_FWD = 240

# Pre-compute ratio array for all offsets
# Store first hit bars for each TP/SL combo
labels_dict = {}
for tp, sl in tpsl_combos:
    tp_r, sl_r = tp/100, sl/100
    ltp = np.full(n, 9999, dtype=np.int16)
    lsl = np.full(n, 9999, dtype=np.int16)
    stp = np.full(n, 9999, dtype=np.int16)
    ssl_ = np.full(n, 9999, dtype=np.int16)
    
    for j in range(1, MAX_FWD + 1):
        if j >= n: break
        ratio = np.ones(n) * np.nan
        ratio[:n-j] = p_arr[j:] / p_arr[:n-j]
        
        m = (ratio >= 1 + tp_r) & (ltp == 9999); ltp[m] = j
        m = (ratio <= 1 - sl_r) & (lsl == 9999); lsl[m] = j
        m = (ratio <= 1 - tp_r) & (stp == 9999); stp[m] = j
        m = (ratio >= 1 + sl_r) & (ssl_ == 9999); ssl_[m] = j
    
    long_win = ltp < lsl
    long_loss = lsl < ltp
    short_win = stp < ssl_
    short_loss = ssl_ < stp
    
    bl = long_win.sum(); bll = long_loss.sum()
    bs = short_win.sum(); bsl = short_loss.sum()
    be = sl/(tp+sl)*100
    print(f"  TP={tp}%/SL={sl}% | LONG {bl/(bl+bll)*100:.1f}%WR | SHORT {bs/(bs+bsl)*100:.1f}%WR | BE={be:.1f}%")
    
    labels_dict[(tp,sl)] = {'lw': long_win, 'll': long_loss, 'sw': short_win, 'sl': short_loss}

print(f"  Done in {time.time()-t0:.1f}s")

# ── Step 4: Train GBM ──
print("\n" + "=" * 70)
print("STEP 4: Training models")
print("=" * 70)

X = feat.fillna(0).values
split = int(n * 0.75)
X_tr, X_te = X[:split], X[split:]
n_te = n - split

print(f"  Train: {split:,} | Test: {n_te:,}")
print(f"  Train: {feat.index[0].date()} → {feat.index[split-1].date()}")
print(f"  Test:  {feat.index[split].date()} → {feat.index[-1].date()}")

# Train multiple models with different targets
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

models = {}
# Target 1: 5-min forward return direction
N_FWD = 5
fwd5 = np.zeros(n)
fwd5[:n-N_FWD] = (p_arr[N_FWD:] - p_arr[:n-N_FWD]) / p_arr[:n-N_FWD] * 100
y5 = (fwd5 > 0.02).astype(int)

# Target 2: 15-min forward return direction
N_FWD2 = 15
fwd15 = np.zeros(n)
fwd15[:n-N_FWD2] = (p_arr[N_FWD2:] - p_arr[:n-N_FWD2]) / p_arr[:n-N_FWD2] * 100
y15 = (fwd15 > 0.03).astype(int)

# Target 3: 60-min forward return direction
N_FWD3 = 60
fwd60 = np.zeros(n)
fwd60[:n-N_FWD3] = (p_arr[N_FWD3:] - p_arr[:n-N_FWD3]) / p_arr[:n-N_FWD3] * 100
y60 = (fwd60 > 0.05).astype(int)

for name, y, model_cls, params in [
    ('RF_5m', y5, RandomForestClassifier, dict(n_estimators=100, max_depth=8, min_samples_leaf=50, n_jobs=-1, random_state=42)),
    ('HGB_5m', y5, HistGradientBoostingClassifier, dict(max_iter=300, max_depth=5, min_samples_leaf=100, learning_rate=0.05, random_state=42)),
    ('HGB_15m', y15, HistGradientBoostingClassifier, dict(max_iter=300, max_depth=5, min_samples_leaf=100, learning_rate=0.05, random_state=42)),
    ('HGB_60m', y60, HistGradientBoostingClassifier, dict(max_iter=300, max_depth=5, min_samples_leaf=100, learning_rate=0.05, random_state=42)),
]:
    t0 = time.time()
    m = model_cls(**params)
    m.fit(X_tr, y[:split])
    probs = m.predict_proba(X_te)[:, 1]
    models[name] = {'model': m, 'probs': probs}
    print(f"  {name}: trained in {time.time()-t0:.1f}s | OOS mean prob: {probs.mean():.3f}")

# ── Step 5: Mega evaluation ──
print("\n" + "=" * 70)
print("STEP 5: Strategy evaluation (all combos)")
print("=" * 70)

vol_60_idx = cols.index('vol_60')
vol_te = X_te[:, vol_60_idx]
vol_med = np.median(X[:, vol_60_idx])

results = []
for tp, sl in tpsl_combos:
    lab = labels_dict[(tp,sl)]
    lw_te = lab['lw'][split:]
    ll_te = lab['ll'][split:]
    sw_te = lab['sw'][split:]
    sl_te = lab['sl'][split:]
    be = sl/(tp+sl)*100
    
    for mname, mdata in models.items():
        probs = mdata['probs']
        for thr in [0.55, 0.60, 0.65]:
            for vm in [0, 1.5, 2.0, 2.5]:
                vmask = vol_te >= vol_med * vm if vm > 0 else np.ones(n_te, dtype=bool)
                lm = (probs > thr) & vmask
                sm = (probs < (1-thr)) & vmask
                
                w = lw_te[lm].sum() + sw_te[sm].sum()
                lo = ll_te[lm].sum() + sl_te[sm].sum()
                tot = w + lo
                if tot < 30: continue
                wr = w/tot*100
                ev = (wr/100*tp) - ((100-wr)/100*sl)
                vn = f"{vm:.1f}x" if vm > 0 else "none"
                results.append({
                    'tp': tp, 'sl': sl, 'model': mname, 'thr': thr, 'vol': vn,
                    'trades': tot, 'wins': w, 'losses': lo, 'wr': wr, 'ev': ev, 'be': be,
                    'long_w': lw_te[lm].sum(), 'long_l': ll_te[lm].sum(),
                    'short_w': sw_te[sm].sum(), 'short_l': sl_te[sm].sum(),
                })

# Sort by EV
results.sort(key=lambda x: x['ev'], reverse=True)
profitable = [r for r in results if r['ev'] > 0]

print(f"\nTested {len(results)} combos | {len(profitable)} profitable")
print(f"\nTOP 20 STRATEGIES:")
print(f"{'#':>3} {'TP':>4} {'SL':>4} {'Model':>8} {'Thr':>5} {'Vol':>5} | {'Trades':>6} {'LW':>4}+{'SW':>4} {'LL':>4}+{'SL':>4} | {'WR%':>5} {'BE%':>5} {'EV%':>7}")
print("-" * 95)
for i, r in enumerate(results[:20]):
    print(f"{i+1:3d} {r['tp']:4.1f} {r['sl']:4.1f} {r['model']:>8} {r['thr']:5.0%} {r['vol']:>5} | "
          f"{r['trades']:6d} {r['long_w']:4d}+{r['short_w']:4d} {r['long_l']:4d}+{r['short_l']:4d} | "
          f"{r['wr']:5.1f} {r['be']:5.1f} {r['ev']:+6.3f}%")

# Show bottom too (worst strategies)
print(f"\nBOTTOM 5 (worst):")
for i, r in enumerate(results[-5:]):
    print(f"  {r['tp']:.1f}/{r['sl']:.1f} {r['model']:>8} {r['thr']:.0%} {r['vol']:>5} | {r['trades']:6d}T | {r['wr']:.1f}%WR | EV={r['ev']:+.3f}%")

# Save best model
if profitable:
    best = profitable[0]
    print(f"\nBEST: TP={best['tp']}% SL={best['sl']}% {best['model']} >{best['thr']:.0%} vol>={best['vol']}")
    print(f"  {best['trades']}T, {best['wr']:.1f}%WR, EV={best['ev']:+.3f}%")
    print(f"  LONG: {best['long_w']}W/{best['long_l']}L | SHORT: {best['short_w']}W/{best['short_l']}L")
    
    bm = models[best['model']]['model']
    with open(f'{MODEL_DIR}/best_model.pkl', 'wb') as f: pickle.dump(bm, f)
    with open(f'{MODEL_DIR}/feature_cols.pkl', 'wb') as f: pickle.dump(cols, f)
    with open(f'{MODEL_DIR}/config.pkl', 'wb') as f: pickle.dump({
        'tp': best['tp'], 'sl': best['sl'], 'model_name': best['model'],
        'threshold': best['thr'], 'vol_filter': best['vol'],
        'vol_60_median': vol_med, 'wr': best['wr'], 'ev': best['ev'],
    }, f)
    print(f"  Saved to {MODEL_DIR}/")

# Top features for best model
if profitable:
    bm = models[profitable[0]['model']]['model']
    imp = bm.feature_importances_
    print(f"\nTop 10 features ({profitable[0]['model']}):")
    for j in np.argsort(imp)[::-1][:10]:
        print(f"  {cols[j]:25s} {imp[j]:.4f}")

print("\nDONE.")
