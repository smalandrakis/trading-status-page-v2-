"""Fast vectorized grid search over TP/SL combos + RF thresholds.
Uses numpy vectorization — no slow Python loops for label computation.
Prints progress after each TP/SL combo.
"""
import numpy as np, pandas as pd, time, pickle
from sklearn.ensemble import RandomForestClassifier

print("=" * 70)
print("GRID SEARCH: Best TP/SL + Strategy Config")
print("=" * 70)

# ── Load data ──
t0 = time.time()
feat = pd.read_parquet('data/archive/tick_features_archive.parquet')
bars = pd.read_parquet('data/archive/tick_bars_16sec.parquet')
n = min(len(feat), len(bars))
feat = feat.iloc[:n]
p = bars['close'].values[:n]
cols = list(feat.columns)
X = feat.fillna(0).values
print(f"Loaded {n} bars ({feat.index[0]} → {feat.index[-1]}) in {time.time()-t0:.1f}s")

# ── Train RF once (forward return model, doesn't depend on TP/SL) ──
print("\nTraining RF model (100 trees)...")
t0 = time.time()
N_FWD = 20
fwd_ret = np.zeros(n)
fwd_ret[:-N_FWD] = (p[N_FWD:] - p[:-N_FWD]) / p[:-N_FWD] * 100
y_up = (fwd_ret > 0.05).astype(int)

split = int(n * 0.75)
rf = RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=30,
                             random_state=42, n_jobs=-1)
rf.fit(X[:split], y_up[:split])
probs = rf.predict_proba(X[split:])[:, 1]
print(f"  Done in {time.time()-t0:.1f}s, OOS size={n-split}")

# Vol and return arrays for OOS
vol_idx = cols.index('vol_1h')
vol_oos = X[split:, vol_idx]
vol_med = np.median(X[:, vol_idx])
p_oos = p[split:]
n_oos = len(p_oos)

# Return columns for vol-only strategies
ret_cols = {c: X[split:, cols.index(c)] for c in ['ret_1m', 'ret_5m', 'ret_15m']}

# ── Vectorized label builder ──
def build_labels_fast(prices, tp_pct, sl_pct, max_fwd=400):
    """Vectorized: for each bar, did LONG TP or SL hit first? Same for SHORT.
    Returns long_labels (1=TP, -1=SL, 0=neither), short_labels same."""
    n = len(prices)
    long_lab = np.zeros(n, dtype=np.int8)
    short_lab = np.zeros(n, dtype=np.int8)
    
    # Process in vectorized chunks of forward bars
    # For each offset j=1..max_fwd, check if TP/SL hit
    long_done = np.zeros(n, dtype=bool)
    short_done = np.zeros(n, dtype=bool)
    
    for j in range(1, max_fwd + 1):
        if j >= n:
            break
        # Future price at offset j
        fwd = np.empty(n)
        fwd[:] = np.nan
        fwd[:n-j] = prices[j:]
        
        ratio = fwd / prices  # price[i+j] / price[i]
        
        # LONG: TP if ratio >= 1+tp, SL if ratio <= 1-sl
        mask_ltp = (~long_done) & (ratio >= 1 + tp_pct)
        mask_lsl = (~long_done) & (ratio <= 1 - sl_pct)
        long_lab[mask_ltp] = 1
        long_lab[mask_lsl] = -1
        long_done |= mask_ltp | mask_lsl
        
        # SHORT: TP if ratio <= 1-tp, SL if ratio >= 1+sl
        mask_stp = (~short_done) & (ratio <= 1 - tp_pct)
        mask_ssl = (~short_done) & (ratio >= 1 + sl_pct)
        short_lab[mask_stp] = 1
        short_lab[mask_ssl] = -1
        short_done |= mask_stp | mask_ssl
        
        # Early exit if all done
        if long_done.all() and short_done.all():
            break
    
    return long_lab, short_lab

# ── Grid search ──
print("\n" + "=" * 70)
print("GRID SEARCH RESULTS")
print("=" * 70)

tp_values = [0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
sl_values = [0.15, 0.2, 0.3, 0.4, 0.5]
vol_mults = [0, 1.0, 1.5, 2.0, 2.5]
rf_thresholds = [0.55, 0.60, 0.65]

results = []
total_combos = len(tp_values) * len(sl_values)
combo_i = 0

print(f"\nTesting {total_combos} TP/SL combos x {len(vol_mults)} vol filters x {len(rf_thresholds)} RF thresholds")
print(f"{'TP%':>5} {'SL%':>5} | {'Base LONG':>10} {'Base SHORT':>11} | {'Best Strategy':>40} | {'WR%':>5} {'EV%':>8} {'Trades':>6}")
print("-" * 105)

for tp in tp_values:
    for sl in sl_values:
        combo_i += 1
        t0 = time.time()
        tp_pct = tp / 100
        sl_pct = sl / 100
        breakeven_wr = sl / (tp + sl) * 100
        
        long_lab, short_lab = build_labels_fast(p_oos, tp_pct, sl_pct)
        
        # Base rates (random entry)
        l_total = np.sum(np.abs(long_lab) > 0)
        l_wins = np.sum(long_lab == 1)
        s_total = np.sum(np.abs(short_lab) > 0)
        s_wins = np.sum(short_lab == 1)
        base_l_wr = l_wins / l_total * 100 if l_total else 0
        base_s_wr = s_wins / s_total * 100 if s_total else 0
        
        best_for_combo = None
        
        # Strategy 1: RF + vol filter
        for thr in rf_thresholds:
            for vm in vol_mults:
                vol_min = vol_med * vm if vm > 0 else 0
                vol_mask = vol_oos >= vol_min if vm > 0 else np.ones(n_oos, dtype=bool)
                
                long_mask = vol_mask & (probs > thr)
                short_mask = vol_mask & (probs < (1 - thr))
                
                w = np.sum(long_lab[long_mask] == 1) + np.sum(short_lab[short_mask] == 1)
                l = np.sum(long_lab[long_mask] == -1) + np.sum(short_lab[short_mask] == -1)
                t = w + l
                if t < 10:
                    continue
                wr = w / t * 100
                ev = (wr / 100 * tp) - ((100 - wr) / 100 * sl)
                name = f"RF>{thr:.0%} vol>={vm:.1f}x" if vm > 0 else f"RF>{thr:.0%} no_vol"
                results.append({'tp': tp, 'sl': sl, 'strategy': name, 'trades': t,
                                'wins': w, 'losses': l, 'wr': wr, 'ev': ev, 'be_wr': breakeven_wr})
                if best_for_combo is None or ev > best_for_combo['ev']:
                    best_for_combo = results[-1]
        
        # Strategy 2: Vol + momentum (no ML)
        for vm in [1.5, 2.0, 2.5]:
            vol_min = vol_med * vm
            vol_mask = vol_oos >= vol_min
            for rc_name, rc_vals in ret_cols.items():
                long_mask = vol_mask & (rc_vals > 0)
                short_mask = vol_mask & (rc_vals < 0)
                
                w = np.sum(long_lab[long_mask] == 1) + np.sum(short_lab[short_mask] == 1)
                l = np.sum(long_lab[long_mask] == -1) + np.sum(short_lab[short_mask] == -1)
                t = w + l
                if t < 10:
                    continue
                wr = w / t * 100
                ev = (wr / 100 * tp) - ((100 - wr) / 100 * sl)
                name = f"vol>={vm:.1f}x + {rc_name}"
                results.append({'tp': tp, 'sl': sl, 'strategy': name, 'trades': t,
                                'wins': w, 'losses': l, 'wr': wr, 'ev': ev, 'be_wr': breakeven_wr})
                if best_for_combo is None or ev > best_for_combo['ev']:
                    best_for_combo = results[-1]
        
        # Print best for this TP/SL combo
        if best_for_combo:
            b = best_for_combo
            marker = " <<<" if b['ev'] > 0 else ""
            print(f"{tp:5.1f} {sl:5.1f} | {base_l_wr:8.1f}%L {base_s_wr:9.1f}%S | {b['strategy']:>40} | {b['wr']:5.1f} {b['ev']:+7.3f}% {b['trades']:6d}{marker}")
        else:
            print(f"{tp:5.1f} {sl:5.1f} | {base_l_wr:8.1f}%L {base_s_wr:9.1f}%S | {'(no viable strategy)':>40} |")

# ── Top 20 results ──
print("\n" + "=" * 70)
print("TOP 20 STRATEGIES (sorted by EV%)")
print("=" * 70)
results_sorted = sorted(results, key=lambda x: x['ev'], reverse=True)
print(f"{'#':>3} {'TP%':>5} {'SL%':>5} {'R:R':>5} | {'Strategy':>35} | {'Trades':>6} {'WR%':>6} {'BE%':>5} | {'EV%':>8} | Profitable?")
print("-" * 105)
for i, r in enumerate(results_sorted[:20]):
    rr = r['tp'] / r['sl'] if r['sl'] > 0 else 0
    profit = "YES $$$" if r['ev'] > 0 else "no"
    print(f"{i+1:3d} {r['tp']:5.1f} {r['sl']:5.1f} {rr:5.1f} | {r['strategy']:>35} | {r['trades']:6d} {r['wr']:6.1f} {r['be_wr']:5.1f} | {r['ev']:+7.3f}% | {profit}")

# ── Summary ──
profitable = [r for r in results if r['ev'] > 0]
print(f"\n{'='*70}")
print(f"SUMMARY: {len(profitable)} profitable configs out of {len(results)} tested")
if profitable:
    best = max(profitable, key=lambda x: x['ev'])
    print(f"BEST: TP={best['tp']}% SL={best['sl']}% | {best['strategy']} | "
          f"{best['trades']}T {best['wr']:.1f}%WR EV={best['ev']:+.3f}%")
    
    # Also show best by trade count (more reliable)
    reliable = [r for r in profitable if r['trades'] >= 50]
    if reliable:
        best_r = max(reliable, key=lambda x: x['ev'])
        print(f"BEST (≥50 trades): TP={best_r['tp']}% SL={best_r['sl']}% | {best_r['strategy']} | "
              f"{best_r['trades']}T {best_r['wr']:.1f}%WR EV={best_r['ev']:+.3f}%")
else:
    print("NO profitable configurations found.")
print("=" * 70)
