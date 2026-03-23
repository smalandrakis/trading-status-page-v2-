"""Train TP/SL model v2 on 12 months of 1-min Binance data.
All vectorized — no slow Python loops. Walk-forward validation.
"""
import numpy as np, pandas as pd, pickle, time, os
from sklearn.ensemble import RandomForestClassifier

TP, SL = 1.0, 0.5
MODEL_DIR = 'models/tpsl_v2'
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Step 1: Load 1-min data ──
print("Step 1: Loading 1-min data...")
t0 = time.time()
df = pd.read_parquet('data/btc_1m_12mo.parquet')
print(f"  {len(df):,} candles, {df.index[0].date()} to {df.index[-1].date()}")
print(f"  Price: ${df['close'].iloc[0]:,.0f} → ${df['close'].iloc[-1]:,.0f}")
print(f"  Loaded in {time.time()-t0:.1f}s")

# ── Step 2: Compute features on 1-min bars ──
print("\nStep 2: Computing features...")
t0 = time.time()
c = df['close']; h = df['high']; l = df['low']; v = df['volume']

feat = pd.DataFrame(index=df.index)

# Returns
for w in [1, 5, 15, 30, 60]:
    feat[f'ret_{w}m'] = c.pct_change(w) * 100

# Volatility (rolling std of 1-min returns)
r1 = c.pct_change()
for w in [5, 15, 30, 60]:
    feat[f'vol_{w}m'] = r1.rolling(w).std()

# Volume ratios
for w in [5, 15, 30, 60]:
    feat[f'vratio_{w}m'] = v / v.rolling(w).mean().replace(0, np.nan)

# RSI
for p in [14, 30]:
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(p).mean()
    loss = (-delta.clip(upper=0)).rolling(p).mean()
    feat[f'rsi_{p}'] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

# Bollinger Band %B
for w in [20, 60]:
    ma = c.rolling(w).mean()
    std = c.rolling(w).std()
    feat[f'bb_pct_{w}'] = (c - (ma - 2*std)) / (4*std).replace(0, np.nan)

# ATR %
for w in [14, 60]:
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    feat[f'atr_pct_{w}'] = tr.rolling(w).mean() / c * 100

# MACD
ema12 = c.ewm(span=12).mean()
ema26 = c.ewm(span=26).mean()
feat['macd_diff'] = ema12 - ema26
feat['macd_hist'] = feat['macd_diff'] - feat['macd_diff'].ewm(span=9).mean()

# Price position (% from rolling high/low)
for w in [60, 240]:
    feat[f'pos_from_high_{w}'] = (c - h.rolling(w).max()) / c * 100
    feat[f'pos_from_low_{w}'] = (c - l.rolling(w).min()) / c * 100

# Bar range
feat['bar_range'] = (h - l) / c * 100
feat['bar_range_ma'] = feat['bar_range'].rolling(30).mean()

# Volume
feat['vol_ma_ratio'] = v / v.rolling(60).mean().replace(0, np.nan)

cols = list(feat.columns)
feat = feat.dropna()
print(f"  {len(feat):,} rows, {len(cols)} features in {time.time()-t0:.1f}s")

# ── Step 3: Build labels (vectorized) ──
print("\nStep 3: Building TP/SL labels (vectorized)...")
t0 = time.time()
p_arr = df['close'].reindex(feat.index).values
n = len(p_arr)
tp_r, sl_r = TP/100, SL/100

# Check forward 240 bars (= 4 hours at 1-min)
MAX_FWD = 240
long_tp = np.full(n, 9999, dtype=np.int32)
long_sl = np.full(n, 9999, dtype=np.int32)
short_tp = np.full(n, 9999, dtype=np.int32)
short_sl = np.full(n, 9999, dtype=np.int32)

for j in range(1, MAX_FWD + 1):
    if j >= n: break
    ratio = np.ones(n) * np.nan
    ratio[:n-j] = p_arr[j:] / p_arr[:n-j]
    
    m = (ratio >= 1 + tp_r) & (long_tp == 9999)
    long_tp[m] = j
    m = (ratio <= 1 - sl_r) & (long_sl == 9999)
    long_sl[m] = j
    m = (ratio <= 1 - tp_r) & (short_tp == 9999)
    short_tp[m] = j
    m = (ratio >= 1 + sl_r) & (short_sl == 9999)
    short_sl[m] = j
    
    if j % 60 == 0:
        print(f"  Forward offset {j}/{MAX_FWD}")

long_win = long_tp < long_sl
long_loss = long_sl < long_tp
short_win = short_tp < short_sl
short_loss = short_sl < short_tp

# Combined label: 1=LONG_WIN, 2=LONG_LOSS, 3=SHORT_WIN, 4=SHORT_LOSS, 0=neither
labels = np.zeros(n, dtype=np.int8)
labels[long_win] = 1
labels[long_loss] = 2
labels[short_win] = 3
labels[short_loss] = 4

bl = long_win.sum(); bll = long_loss.sum()
bs = short_win.sum(); bsl = short_loss.sum()
print(f"  Done in {time.time()-t0:.1f}s")
print(f"  LONG base: {bl}W/{bll}L = {bl/(bl+bll)*100:.1f}% WR")
print(f"  SHORT base: {bs}W/{bsl}L = {bs/(bs+bsl)*100:.1f}% WR")
print(f"  Neither: {(labels==0).sum()}")

# ── Step 4: Forward return target for RF ──
print("\nStep 4: Training RF...")
t0 = time.time()
N_FWD = 5  # 5 bars = 5 min forward on 1-min data
X = feat.fillna(0).values
fwd_ret = np.zeros(n)
fwd_ret[:n-N_FWD] = (p_arr[N_FWD:] - p_arr[:n-N_FWD]) / p_arr[:n-N_FWD] * 100
y_up = (fwd_ret > 0.02).astype(int)

# Walk-forward: train on first 9 months, test on last 3
split = int(n * 0.75)
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y_up[:split], y_up[split:]
labels_te = labels[split:]

print(f"  Train: {split:,} rows ({split/len(feat)*100:.0f}%)")
print(f"  Test:  {n-split:,} rows ({(n-split)/len(feat)*100:.0f}%)")
print(f"  Train dates: {feat.index[0].date()} → {feat.index[split-1].date()}")
print(f"  Test dates:  {feat.index[split].date()} → {feat.index[-1].date()}")

rf = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=50,
                             random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)
probs = rf.predict_proba(X_te)[:, 1]
print(f"  Trained in {time.time()-t0:.1f}s")

# ── Step 5: Evaluate strategies ──
print("\nStep 5: Strategy evaluation (OOS)")
vol_60m_idx = cols.index('vol_60m')
vol_te = X_te[:, vol_60m_idx]
vol_med = np.median(X[:, vol_60m_idx])

long_win_te = labels_te == 1
long_loss_te = labels_te == 2
short_win_te = labels_te == 3
short_loss_te = labels_te == 4

# Base rates in test period
tl = long_win_te.sum() + long_loss_te.sum()
ts = short_win_te.sum() + short_loss_te.sum()
print(f"  Test LONG base: {long_win_te.sum()}W/{long_loss_te.sum()}L = {long_win_te.sum()/tl*100:.1f}%")
print(f"  Test SHORT base: {short_win_te.sum()}W/{short_loss_te.sum()}L = {short_win_te.sum()/ts*100:.1f}%")

print(f"\n  {'Strategy':>30} | {'Trades':>6} | {'LW':>4} {'LL':>4} {'SW':>4} {'SL':>4} | {'WR%':>5} | {'EV%':>7} | OK?")
print("  " + "-" * 85)

results = []
for thr in [0.55, 0.60, 0.65, 0.70]:
    for vm in [0, 1.0, 1.5, 2.0, 2.5]:
        vol_mask = vol_te >= vol_med * vm if vm > 0 else np.ones(len(vol_te), dtype=bool)
        l_mask = (probs > thr) & vol_mask
        s_mask = (probs < (1-thr)) & vol_mask
        
        lw = long_win_te[l_mask].sum()
        ll = long_loss_te[l_mask].sum()
        sw = short_win_te[s_mask].sum()
        sl_ = short_loss_te[s_mask].sum()
        
        tw = lw + sw; tl_ = ll + sl_; tot = tw + tl_
        if tot < 20: continue
        wr = tw/tot*100
        ev = (wr/100*TP) - ((100-wr)/100*SL)
        ok = "YES" if ev > 0 else "no"
        vn = f"{vm:.1f}x" if vm > 0 else "none"
        name = f"RF>{thr:.0%} vol>={vn}"
        results.append({'name': name, 'tot': tot, 'lw': lw, 'll': ll, 'sw': sw, 'sl': sl_, 'wr': wr, 'ev': ev})
        if ev > 0 or vm == 0:
            print(f"  {name:>30} | {tot:6d} | {lw:4d} {ll:4d} {sw:4d} {sl_:4d} | {wr:5.1f} | {ev:+6.3f}% | {ok}")

# Vol-only strategies
for vm in [1.5, 2.0, 2.5]:
    vol_mask = vol_te >= vol_med * vm
    ret5_idx = cols.index('ret_5m')
    ret_vals = X_te[:, ret5_idx]
    l_mask = vol_mask & (ret_vals > 0)
    s_mask = vol_mask & (ret_vals < 0)
    lw = long_win_te[l_mask].sum()
    ll = long_loss_te[l_mask].sum()
    sw = short_win_te[s_mask].sum()
    sl_ = short_loss_te[s_mask].sum()
    tw = lw + sw; tl_ = ll + sl_; tot = tw + tl_
    if tot < 20:
        continue
    wr = tw/tot*100
    ev = (wr/100*TP) - ((100-wr)/100*SL)
    ok = "YES" if ev > 0 else "no"
    name = f"vol>={vm:.1f}x + ret_5m"
    print(f"  {name:>30} | {tot:6d} | {lw:4d} {ll:4d} {sw:4d} {sl_:4d} | {wr:5.1f} | {ev:+6.3f}% | {ok}")

# Top features
print("\nTop 10 features:")
imp = rf.feature_importances_
for j in np.argsort(imp)[::-1][:10]:
    print(f"  {cols[j]:25s} {imp[j]:.4f}")

# Sort and show top 10 profitable
profitable = sorted([r for r in results if r['ev'] > 0], key=lambda x: x['ev'], reverse=True)
if profitable:
    print(f"\nTOP 5 PROFITABLE (out of {len(profitable)}):")
    for i, r in enumerate(profitable[:5]):
        print(f"  {i+1}. {r['name']:>30} | {r['tot']:5d}T | {r['wr']:.1f}%WR | EV={r['ev']:+.3f}%")

# Save model
print("\nSaving model...")
with open(f'{MODEL_DIR}/rf_1m.pkl', 'wb') as f: pickle.dump(rf, f)
with open(f'{MODEL_DIR}/feature_cols.pkl', 'wb') as f: pickle.dump(cols, f)
with open(f'{MODEL_DIR}/vol_stats.pkl', 'wb') as f: pickle.dump({'vol_60m_median': vol_med}, f)
print(f"  Saved to {MODEL_DIR}/")
print("DONE.")
