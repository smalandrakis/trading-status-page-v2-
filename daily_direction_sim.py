#!/usr/bin/env python3
"""
Daily Direction Strategy Simulation
- Resample 1-min data to daily bars
- Features: last 24h movement, multi-day trends, vol, RSI, day-of-week, etc.
- Label: does price hit TP=+5% before SL=-1% within 24h? (LONG)
         does price hit TP=-5% before SL=+1% within 24h? (SHORT)
- Also test: just predict next-day direction (up/down) and enter with TP/SL
- 1 trade per day, walk-forward validation
"""
import numpy as np, pandas as pd, time, os
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score

DATA_PATH = 'data/btc_1m_12mo.parquet'
os.makedirs('models/daily_dir', exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# STEP 1: Load & explore daily volatility
# ══════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 1: Load data & daily stats")
print("=" * 70)
df = pd.read_parquet(DATA_PATH)
df.index = df.index.tz_localize(None) if df.index.tz else df.index
df = df.sort_index()
print(f"  {len(df):,} 1-min candles | {df.index[0].date()} → {df.index[-1].date()}")

# Daily OHLCV
daily = df.resample('1D').agg({
    'open': 'first', 'high': 'max', 'low': 'min',
    'close': 'last', 'volume': 'sum'
}).dropna()
daily['ret'] = daily['close'].pct_change() * 100
daily['range'] = (daily['high'] - daily['low']) / daily['open'] * 100
daily['max_up'] = (daily['high'] / daily['open'] - 1) * 100
daily['max_down'] = (1 - daily['low'] / daily['open']) * 100

print(f"  {len(daily)} trading days")
print(f"  Daily return: mean={daily['ret'].mean():.2f}%, std={daily['ret'].std():.2f}%")
print(f"  Daily range:  mean={daily['range'].mean():.2f}%, max={daily['range'].max():.2f}%")
print(f"  Max intraday up:   mean={daily['max_up'].mean():.2f}%")
print(f"  Max intraday down: mean={daily['max_down'].mean():.2f}%")

# ══════════════════════════════════════════════════════════════════
# STEP 2: Build intraday TP/SL labels using 1-min bars
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: Build TP/SL labels on daily entries")
print("=" * 70)

TP_SL_CONFIGS = [
    {'tp': 5.0, 'sl': 1.0, 'name': 'tp50_sl10'},   # 5:1 R:R, BE=16.7%
    {'tp': 3.0, 'sl': 1.0, 'name': 'tp30_sl10'},   # 3:1 R:R, BE=25.0%
    {'tp': 3.0, 'sl': 0.7, 'name': 'tp30_sl07'},   # 4.3:1 R:R, BE=18.9%
    {'tp': 2.0, 'sl': 1.0, 'name': 'tp20_sl10'},   # 2:1 R:R, BE=33.3%
    {'tp': 5.0, 'sl': 2.0, 'name': 'tp50_sl20'},   # 2.5:1 R:R, BE=28.6%
]
MAX_HOLD_BARS = 1440  # 24 hours in 1-min bars

ca = df['close'].values.astype(np.float64)
ha = df['high'].values.astype(np.float64)
la = df['low'].values.astype(np.float64)
nn = len(ca)

# Get the 1-min index of each day's open
daily_open_indices = []
for day in daily.index:
    day_mask = (df.index >= day) & (df.index < day + pd.Timedelta(days=1))
    idx = np.where(day_mask)[0]
    if len(idx) > 0:
        daily_open_indices.append(idx[0])
daily_open_indices = np.array(daily_open_indices)
print(f"  {len(daily_open_indices)} daily entry points")

all_labels = {}
for cfg in TP_SL_CONFIGS:
    long_win = np.zeros(len(daily_open_indices), dtype=np.int8)
    short_win = np.zeros(len(daily_open_indices), dtype=np.int8)
    long_outcome = []  # for detailed analysis
    short_outcome = []

    for di, entry_idx in enumerate(daily_open_indices):
        entry_price = ca[entry_idx]
        end_idx = min(entry_idx + MAX_HOLD_BARS, nn)

        # Scan forward bar by bar
        long_tp_hit = False; long_sl_hit = False
        short_tp_hit = False; short_sl_hit = False
        long_first_tp = 9999; long_first_sl = 9999
        short_first_tp = 9999; short_first_sl = 9999

        for j in range(entry_idx + 1, end_idx):
            bars_elapsed = j - entry_idx
            # LONG: TP if high >= entry*(1+tp%), SL if low <= entry*(1-sl%)
            if not long_tp_hit and ha[j] >= entry_price * (1 + cfg['tp'] / 100):
                long_tp_hit = True; long_first_tp = bars_elapsed
            if not long_sl_hit and la[j] <= entry_price * (1 - cfg['sl'] / 100):
                long_sl_hit = True; long_first_sl = bars_elapsed
            # SHORT: TP if low <= entry*(1-tp%), SL if high >= entry*(1+sl%)
            if not short_tp_hit and la[j] <= entry_price * (1 - cfg['tp'] / 100):
                short_tp_hit = True; short_first_tp = bars_elapsed
            if not short_sl_hit and ha[j] >= entry_price * (1 + cfg['sl'] / 100):
                short_sl_hit = True; short_first_sl = bars_elapsed
            if long_tp_hit and long_sl_hit and short_tp_hit and short_sl_hit:
                break

        long_win[di] = 1 if (long_first_tp < long_first_sl) else 0
        short_win[di] = 1 if (short_first_tp < short_first_sl) else 0

        # Track outcome type
        if long_first_tp < long_first_sl: long_outcome.append('TP')
        elif long_first_sl < long_first_tp: long_outcome.append('SL')
        else: long_outcome.append('TIMEOUT')

        if short_first_tp < short_first_sl: short_outcome.append('TP')
        elif short_first_sl < short_first_tp: short_outcome.append('SL')
        else: short_outcome.append('TIMEOUT')

    all_labels[f"L_{cfg['name']}"] = long_win
    all_labels[f"S_{cfg['name']}"] = short_win
    be = cfg['sl'] / (cfg['tp'] + cfg['sl']) * 100
    lo = pd.Series(long_outcome); so = pd.Series(short_outcome)
    print(f"  {cfg['name']:>12}: LONG TP={lo.eq('TP').mean()*100:.1f}% SL={lo.eq('SL').mean()*100:.1f}% "
          f"TO={lo.eq('TIMEOUT').mean()*100:.1f}% | SHORT TP={so.eq('TP').mean()*100:.1f}% "
          f"SL={so.eq('SL').mean()*100:.1f}% TO={so.eq('TIMEOUT').mean()*100:.1f}% | BE={be:.1f}%")

# ══════════════════════════════════════════════════════════════════
# STEP 3: Daily features (known at day open)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: Build daily features")
print("=" * 70)

feat = pd.DataFrame(index=daily.index)
c = daily['close'].astype(float)
h = daily['high'].astype(float)
l = daily['low'].astype(float)
o = daily['open'].astype(float)
v = daily['volume'].astype(float)

# Previous day returns (shifted so we use yesterday's data at today's open)
for w in [1, 2, 3, 5, 7, 14, 21, 30]:
    feat[f'ret_{w}d'] = c.pct_change(w).shift(1) * 100

# Volatility (rolling std of daily returns, shifted)
r = c.pct_change()
for w in [3, 5, 7, 14, 30]:
    feat[f'vol_{w}d'] = r.rolling(w).std().shift(1) * 100

# Daily range (shifted)
rng = ((h - l) / o * 100)
for w in [3, 5, 7, 14]:
    feat[f'avg_range_{w}d'] = rng.rolling(w).mean().shift(1)

# RSI (shifted)
for p in [7, 14]:
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(span=p).mean()
    loss = (-delta.clip(upper=0)).ewm(span=p).mean()
    feat[f'rsi_{p}'] = (100 - (100 / (1 + gain / loss.replace(0, np.nan)))).shift(1)

# Bollinger %B (shifted)
ma20 = c.rolling(20).mean(); std20 = c.rolling(20).std()
feat['bb_pctb'] = ((c - (ma20 - 2*std20)) / (4*std20).replace(0, np.nan)).shift(1)

# MACD (shifted)
e12 = c.ewm(span=12).mean(); e26 = c.ewm(span=26).mean()
macd = e12 - e26; sig = macd.ewm(span=9).mean()
feat['macd'] = (macd / c * 100).shift(1)
feat['macd_hist'] = ((macd - sig) / c * 100).shift(1)

# Position in range (shifted)
for w in [10, 20, 30]:
    rh = h.rolling(w).max(); rl = l.rolling(w).min()
    feat[f'pos_{w}d'] = ((c - rl) / (rh - rl).replace(0, np.nan)).shift(1)

# Volume ratio (shifted)
for w in [5, 10]:
    feat[f'vratio_{w}d'] = (v / v.rolling(w).mean().replace(0, np.nan)).shift(1)

# Consecutive up/down days (shifted)
up = (r > 0).astype(float)
feat['up_streak'] = up.groupby((up != up.shift()).cumsum()).cumcount().shift(1)
down = (r < 0).astype(float)
feat['down_streak'] = down.groupby((down != down.shift()).cumsum()).cumcount().shift(1)

# Day of week (cyclical)
feat['dow_sin'] = np.sin(2 * np.pi * daily.index.dayofweek / 7)
feat['dow_cos'] = np.cos(2 * np.pi * daily.index.dayofweek / 7)

# Distance from moving averages (shifted)
for w in [7, 14, 30, 50]:
    ma = c.rolling(w).mean()
    feat[f'dist_ma_{w}d'] = ((c - ma) / ma * 100).shift(1)

# Max up / max down previous day (shifted)
feat['prev_max_up'] = ((h / o - 1) * 100).shift(1)
feat['prev_max_down'] = ((1 - l / o) * 100).shift(1)
feat['prev_body'] = ((c - o) / o * 100).shift(1)

# ATR % (shifted)
tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
for w in [7, 14]:
    feat[f'atr_{w}d'] = (tr.rolling(w).mean() / c * 100).shift(1)

feat = feat.replace([np.inf, -np.inf], np.nan)
feature_cols = list(feat.columns)
print(f"  {len(feature_cols)} daily features (all shifted to prevent leakage)")

# ══════════════════════════════════════════════════════════════════
# STEP 4: Simple baseline — yesterday's direction
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4: Baselines")
print("=" * 70)

for cfg in TP_SL_CONFIGS:
    lw = all_labels[f"L_{cfg['name']}"]
    sw = all_labels[f"S_{cfg['name']}"]
    be = cfg['sl'] / (cfg['tp'] + cfg['sl']) * 100

    # Random (coin flip) — go LONG every day
    l_wr = lw.mean() * 100
    l_ev = (l_wr/100 * cfg['tp']) - ((100-l_wr)/100 * cfg['sl'])
    s_wr = sw.mean() * 100
    s_ev = (s_wr/100 * cfg['tp']) - ((100-s_wr)/100 * cfg['sl'])
    # Best of LONG or SHORT
    best_dir = 'LONG' if l_ev > s_ev else 'SHORT'
    best_wr = l_wr if l_ev > s_ev else s_wr
    best_ev = max(l_ev, s_ev)
    print(f"  {cfg['name']:>12}: Always-{best_dir} WR={best_wr:.1f}% EV={best_ev:+.3f}% "
          f"(BE={be:.1f}%) | LONG={l_wr:.1f}% SHORT={s_wr:.1f}%")

# Momentum baseline: go in direction of yesterday
prev_ret = daily['ret'].shift(1).values
for cfg in TP_SL_CONFIGS:
    lw = all_labels[f"L_{cfg['name']}"]
    sw = all_labels[f"S_{cfg['name']}"]
    valid = ~np.isnan(prev_ret)
    go_long = prev_ret > 0
    go_short = prev_ret <= 0
    wins = 0; trades = 0
    for i in range(len(daily)):
        if not valid[i]: continue
        if go_long[i]: wins += lw[i]; trades += 1
        else: wins += sw[i]; trades += 1
    wr = wins/trades*100 if trades > 0 else 0
    ev = (wr/100*cfg['tp']) - ((100-wr)/100*cfg['sl'])
    print(f"  {cfg['name']:>12}: Momentum WR={wr:.1f}% EV={ev:+.3f}% ({trades}T)")

# Counter-trend baseline: go opposite of yesterday
for cfg in TP_SL_CONFIGS:
    lw = all_labels[f"L_{cfg['name']}"]
    sw = all_labels[f"S_{cfg['name']}"]
    valid = ~np.isnan(prev_ret)
    go_long = prev_ret <= 0  # reversal
    wins = 0; trades = 0
    for i in range(len(daily)):
        if not valid[i]: continue
        if go_long[i]: wins += lw[i]; trades += 1
        else: wins += sw[i]; trades += 1
    wr = wins/trades*100 if trades > 0 else 0
    ev = (wr/100*cfg['tp']) - ((100-wr)/100*cfg['sl'])
    print(f"  {cfg['name']:>12}: Reversal WR={wr:.1f}% EV={ev:+.3f}% ({trades}T)")

# ══════════════════════════════════════════════════════════════════
# STEP 5: ML walk-forward (2-month train, 1-month test)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 5: ML Walk-forward (2-month train, 1-month test)")
print("=" * 70)

feat_vals = feat[feature_cols].values
valid_mask = ~np.isnan(feat_vals).any(axis=1)
months = pd.date_range(daily.index[0].replace(day=1), daily.index[-1], freq='MS')
TRAIN_MONTHS = 3
results = []

for ti in range(TRAIN_MONTHS, len(months)):
    ts = months[ti]
    te = months[ti + 1] if ti + 1 < len(months) else daily.index[-1] + pd.Timedelta(days=1)
    trs = months[ti - TRAIN_MONTHS]

    tr_mask = (daily.index >= trs) & (daily.index < ts) & valid_mask
    te_mask = (daily.index >= ts) & (daily.index < te) & valid_mask
    tr_idx = np.where(tr_mask)[0]; te_idx = np.where(te_mask)[0]
    if len(tr_idx) < 20 or len(te_idx) < 5: continue

    X_tr = feat_vals[tr_idx]; X_te = feat_vals[te_idx]
    month_str = ts.strftime('%Y-%m')
    btc_ret = (daily['close'].iloc[te_idx[-1]] / daily['close'].iloc[te_idx[0]] - 1) * 100
    print(f"\n  {month_str}: {len(te_idx)} days, BTC {btc_ret:+.1f}%")

    for cfg in TP_SL_CONFIGS:
        be = cfg['sl'] / (cfg['tp'] + cfg['sl']) * 100

        for direction in ['LONG', 'SHORT']:
            pfx = 'L' if direction == 'LONG' else 'S'
            y_tr = all_labels[f"{pfx}_{cfg['name']}"][tr_idx]
            y_te = all_labels[f"{pfx}_{cfg['name']}"][te_idx]

            # Skip if no positive labels in train
            if y_tr.sum() < 2: continue

            for model_name, model in [
                ('HGB', HistGradientBoostingClassifier(
                    max_iter=100, max_depth=3, min_samples_leaf=10,
                    learning_rate=0.05, random_state=42, l2_regularization=2.0)),
                ('RF', RandomForestClassifier(
                    n_estimators=100, max_depth=5, min_samples_leaf=5,
                    random_state=42, n_jobs=-1)),
            ]:
                model.fit(X_tr, y_tr)
                probs = model.predict_proba(X_te)[:, 1]
                try: auc = roc_auc_score(y_te, probs)
                except: auc = 0.5

                for thr in [0.20, 0.30, 0.40, 0.50]:
                    sel = probs > thr
                    nt = sel.sum()
                    if nt < 1: continue
                    w = int(y_te[sel].sum())
                    wr = w / nt * 100
                    ev = (wr/100 * cfg['tp']) - ((100-wr)/100 * cfg['sl'])
                    results.append({
                        'month': month_str, 'dir': direction, 'config': cfg['name'],
                        'tp': cfg['tp'], 'sl': cfg['sl'], 'model': model_name,
                        'thr': thr, 'trades': nt, 'wins': w, 'wr': wr, 'ev': ev,
                        'base_wr': y_te.mean()*100, 'be': be, 'auc': auc, 'btc_ret': btc_ret,
                    })

    # Also try: ML picks direction (LONG or SHORT) each day
    # Train a model to predict "should we go LONG?" (binary)
    for cfg in TP_SL_CONFIGS:
        lw = all_labels[f"L_{cfg['name']}"][tr_idx]
        sw = all_labels[f"S_{cfg['name']}"][tr_idx]
        # Label: 1 if LONG wins, 0 if SHORT wins (or neither)
        y_dir_tr = ((lw == 1) & (sw == 0)).astype(np.int8)
        lw_te = all_labels[f"L_{cfg['name']}"][te_idx]
        sw_te = all_labels[f"S_{cfg['name']}"][te_idx]

        if y_dir_tr.sum() < 2: continue

        m = HistGradientBoostingClassifier(
            max_iter=100, max_depth=3, min_samples_leaf=10,
            learning_rate=0.05, random_state=42, l2_regularization=2.0)
        m.fit(X_tr, y_dir_tr)
        p_long = m.predict_proba(X_te)[:, 1]

        # Each day: if p_long > 0.5 → LONG, else → SHORT
        wins = 0; trades = len(te_idx)
        for i in range(len(te_idx)):
            if p_long[i] > 0.5:
                wins += lw_te[i]
            else:
                wins += sw_te[i]
        wr = wins / trades * 100
        ev = (wr/100 * cfg['tp']) - ((100-wr)/100 * cfg['sl'])
        results.append({
            'month': month_str, 'dir': 'PICK', 'config': cfg['name'],
            'tp': cfg['tp'], 'sl': cfg['sl'], 'model': 'HGB_DIR',
            'thr': 0.50, 'trades': trades, 'wins': wins, 'wr': wr, 'ev': ev,
            'base_wr': max(lw_te.mean(), sw_te.mean())*100, 'be': be,
            'auc': 0, 'btc_ret': btc_ret,
        })

# ══════════════════════════════════════════════════════════════════
# STEP 6: Results
# ══════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("STEP 6: AGGREGATE RESULTS")
print("=" * 70)

res = pd.DataFrame(results)
res.to_csv('models/daily_dir/walkforward_results.csv', index=False)

grp = res.groupby(['dir', 'config', 'model', 'thr']).agg(
    tt=('trades', 'sum'), tw=('wins', 'sum'), mo=('month', 'nunique'),
    mp=('ev', lambda x: (x > 0).sum()), auc=('auc', 'mean')).reset_index()
grp['wr'] = grp['tw'] / grp['tt'] * 100
for i, r in grp.iterrows():
    cfg = next(c for c in TP_SL_CONFIGS if c['name'] == r['config'])
    grp.at[i, 'ev'] = (r['wr']/100*cfg['tp']) - ((100-r['wr'])/100*cfg['sl'])
    grp.at[i, 'be'] = cfg['sl']/(cfg['tp']+cfg['sl'])*100

prof = grp[(grp['ev'] > 0) & (grp['tt'] >= 10)].sort_values('ev', ascending=False)
print(f"\nPROFITABLE (min 10T, positive EV):")
print(f"  {'Dir':>5} {'Config':>12} {'Model':>8} {'Thr':>4} | {'#T':>4} {'WR':>6} {'BE':>5} "
      f"{'EV':>8} | {'AUC':>5} {'Mo':>2} {'P':>2}")
print(f"  {'-'*72}")
for _, r in prof.head(30).iterrows():
    print(f"  {r['dir']:>5} {r['config']:>12} {r['model']:>8} {r['thr']:4.0%} | "
          f"{r['tt']:4.0f} {r['wr']:5.1f}% {r['be']:4.1f}% {r['ev']:+7.3f}% | "
          f"{r['auc']:.3f} {r['mo']:2.0f} {r['mp']:2.0f}")

# Monthly detail for top 5
for rank, (_, br) in enumerate(prof.head(5).iterrows()):
    sub = res[(res['dir']==br['dir']) & (res['config']==br['config']) &
              (res['model']==br['model']) & (res['thr']==br['thr'])].sort_values('month')
    cfg = next(c for c in TP_SL_CONFIGS if c['name'] == br['config'])
    tt = sub['trades'].sum(); tw = sub['wins'].sum()
    wr = tw/tt*100; ev = (wr/100*cfg['tp'])-((100-wr)/100*cfg['sl'])
    print(f"\n  #{rank+1}: {br['dir']} {br['config']} {br['model']} >{br['thr']:.0%} — "
          f"{tt}T WR={wr:.1f}% EV={ev:+.3f}%")
    for _, r in sub.iterrows():
        s = "+" if r['ev'] > 0 else "-"
        print(f"    {r['month']} BTC={r['btc_ret']:+.1f}% {r['trades']:.0f}T {r['wins']:.0f}W "
              f"{r['wr']:.1f}%WR EV={r['ev']:+.3f}% {s}")

if len(prof) == 0:
    print("\n  No profitable strategies found.")

print("\nDone!")
