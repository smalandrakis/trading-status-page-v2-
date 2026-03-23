"""Train TP/SL model v4 — larger dataset, chunked label computation."""
import numpy as np, pandas as pd, pickle, time, os
from sklearn.ensemble import RandomForestClassifier

TP, SL = 1.0, 0.5
MODEL_DIR = 'models/tpsl_v1'
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Step 1: Load data ──
print("Step 1: Loading data...")
t0 = time.time()
feat = pd.read_parquet('data/archive/tick_features_archive.parquet')
bars = pd.read_parquet('data/archive/tick_bars_16sec.parquet')
print(f"  Loaded: {len(feat)} feature rows, {len(bars)} bars in {time.time()-t0:.1f}s")

# Align lengths
n = min(len(feat), len(bars))
feat = feat.iloc[:n]
p = bars['close'].values[:n]
cols = list(feat.columns)
print(f"  Aligned: {n} rows, {len(cols)} features")
print(f"  Date range: {feat.index[0]} → {feat.index[-1]}")

# ── Step 2: Build TP/SL labels (chunked) ──
print("\nStep 2: Building TP/SL labels (chunked)...")
t0 = time.time()
tp_pct, sl_pct = TP / 100, SL / 100
labels = np.zeros(n, dtype=np.int8)  # 0=no result, 1=LONG_TP, 2=SHORT_TP, 3=LONG_SL, 4=SHORT_SL
CHUNK = 10000
MAX_FWD = 600  # ~2.5 hours
for start in range(0, n, CHUNK):
    end = min(start + CHUNK, n)
    for i in range(start, end):
        entry = p[i]
        tp_long = entry * (1 + tp_pct)
        sl_long = entry * (1 - sl_pct)
        tp_short = entry * (1 - tp_pct)
        sl_short = entry * (1 + sl_pct)
        # Track LONG and SHORT outcomes independently
        long_result = 0   # 0=none, 1=TP, 3=SL
        long_bar = MAX_FWD + 1
        short_result = 0  # 0=none, 2=TP, 4=SL
        short_bar = MAX_FWD + 1
        limit = min(i + MAX_FWD, n)
        for j in range(i + 1, limit):
            if p[j] >= tp_long:
                long_result = 1; long_bar = j - i; break
            if p[j] <= sl_long:
                long_result = 3; long_bar = j - i; break
        for j in range(i + 1, limit):
            if p[j] <= tp_short:
                short_result = 2; short_bar = j - i; break
            if p[j] >= sl_short:
                short_result = 4; short_bar = j - i; break
        # Pick whichever outcome hit first
        if long_result and short_result:
            labels[i] = long_result if long_bar <= short_bar else short_result
        elif long_result:
            labels[i] = long_result
        elif short_result:
            labels[i] = short_result
    pct = end / n * 100
    print(f"  Chunk {start//CHUNK + 1}: {start}-{end} ({pct:.0f}%) done")

np.save('data/archive/tpsl_labels.npy', labels)
print(f"  Labels built in {time.time()-t0:.1f}s")
print(f"  LONG_TP={np.sum(labels==1)}, SHORT_TP={np.sum(labels==2)}, "
      f"LONG_SL={np.sum(labels==3)}, SHORT_SL={np.sum(labels==4)}, "
      f"No result={np.sum(labels==0)}")

# ── Step 3: Forward return labels + train ──
print("\nStep 3: Training RF model...")
N_FWD = 20
fwd_ret = np.zeros(n)
for i in range(n - N_FWD):
    fwd_ret[i] = (p[i + N_FWD] - p[i]) / p[i] * 100
y_up = np.where(fwd_ret > 0.05, 1, 0)
X = feat.fillna(0).values

split = int(n * 0.75)
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y_up[:split], y_up[split:]
labels_te = labels[split:]

print(f"  Train: {split} rows ({y_tr.sum()} up = {y_tr.mean()*100:.1f}%)")
print(f"  Test:  {n - split} rows")

t0 = time.time()
rf = RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=30,
                             random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)
print(f"  Trained in {time.time()-t0:.1f}s")

probs = rf.predict_proba(X_te)[:, 1]

# ── Step 4: Evaluate ──
print("\nStep 4: Evaluation (OOS)")
vol_idx = cols.index('vol_1h')
vol_te = X_te[:, vol_idx]
vol_med = np.median(X[:, vol_idx])  # full dataset median

print(f"  vol_1h median: {vol_med:.6f}")
print(f"  {'Thr':>5} {'VolFlt':>7} | {'Trades':>6} | {'Wins':>4} {'Loss':>4} | {'WR%':>5} | {'EV%':>8} | Beat?")
print("  " + "-" * 70)
for thr in [0.55, 0.60, 0.65, 0.70]:
    for vol_min_mult in [0, 1.0, 1.5, 2.0, 2.5]:
        vol_min = vol_med * vol_min_mult if vol_min_mult > 0 else 0
        wins = losses = 0
        for i in range(len(X_te)):
            if vol_min > 0 and vol_te[i] < vol_min:
                continue
            if probs[i] > thr:
                if labels_te[i] == 1: wins += 1
                elif labels_te[i] > 0: losses += 1
            elif probs[i] < (1 - thr):
                if labels_te[i] == 2: wins += 1
                elif labels_te[i] > 0: losses += 1
        tot = wins + losses
        if tot < 5: continue
        wr = wins / tot * 100
        ev = (wr / 100 * TP) - ((100 - wr) / 100 * SL)
        vl = f"{vol_min_mult:.1f}x" if vol_min_mult > 0 else "none"
        beat = "YES ✓" if wr > 33.3 else "no"
        print(f"  {thr:5.0%} {vl:>7} | {tot:6d} | {wins:4d} {losses:4d} | {wr:5.1f} | {ev:+7.3f}% | {beat}")

print(f"\n  Vol-only strategies:")
print(f"  {'VolFlt':>7} {'Lookback':>8} | {'Trades':>6} | {'Wins':>4} {'Loss':>4} | {'WR%':>5} | Beat?")
print("  " + "-" * 65)
for vol_mult in [1.5, 2.0, 2.5]:
    for lb_col in ['ret_1m', 'ret_5m', 'ret_15m']:
        lb_idx = cols.index(lb_col)
        vol_min = vol_med * vol_mult
        wins = losses = 0
        for i in range(len(X_te)):
            if vol_te[i] < vol_min: continue
            ret = X_te[i, lb_idx]
            if ret > 0:
                if labels_te[i] == 1: wins += 1
                elif labels_te[i] > 0: losses += 1
            elif ret < 0:
                if labels_te[i] == 2: wins += 1
                elif labels_te[i] > 0: losses += 1
        tot = wins + losses
        if tot < 5: continue
        wr = wins / tot * 100
        beat = "YES ✓" if wr > 33.3 else "no"
        print(f"  {vol_mult:.1f}x    {lb_col:>8} | {tot:6d} | {wins:4d} {losses:4d} | {wr:5.1f} | {beat}")

print("\nTop 10 features:")
imp = rf.feature_importances_
for j in np.argsort(imp)[::-1][:10]:
    print(f"  {cols[j]:30s} {imp[j]:.4f}")

# ── Step 5: Save model ──
print("\nStep 5: Saving model...")
with open(f'{MODEL_DIR}/rf_fwd_return.pkl', 'wb') as f:
    pickle.dump(rf, f)
with open(f'{MODEL_DIR}/feature_cols.pkl', 'wb') as f:
    pickle.dump(cols, f)
with open(f'{MODEL_DIR}/vol_stats.pkl', 'wb') as f:
    pickle.dump({'vol_1h_median': vol_med}, f)
print(f"  Saved to {MODEL_DIR}/")
print(f"  Model: {rf.n_estimators} trees, {len(cols)} features, trained on {split} rows")
print("DONE.")
