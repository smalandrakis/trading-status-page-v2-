#!/usr/bin/env python3
"""
Explore entry models using validated 2-sec tick data.
- Compute indicators from raw ticks (no parquet dependency)
- Test multiple feature sets and model types
- Proper temporal train/test split (first 7 days train, last 3 days test)
- Backtest with validated simulation (SL, trailing stop)
- Compare everything against current system baseline

NO implementation — exploration and reporting only.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================
SL_PCT = 0.30       # Use 0.30% SL (what was active for most trades)
TS_ACTIVATION = 0.30
TS_TRAIL = 0.10
TARGET_PCT = 0.50   # Looking for 0.5% moves
MIN_GAP_BARS = 30   # Min 8 min gap between signals (in 16-sec bars)

# =============================================================================
# STEP 1: Load tick data, resample to 16-sec bars
# =============================================================================
print("=" * 70)
print("STEP 1: Load & resample tick data")
print("=" * 70)

raw = pd.read_csv('logs/btc_price_ticks.csv', parse_dates=['timestamp'])
raw = raw.set_index('timestamp').sort_index()
raw = raw[raw['price'] > 0]
print(f"Raw ticks: {len(raw)}, {raw.index.min()} to {raw.index.max()}")

# 16-sec OHLC
bars = raw['price'].resample('16s').agg(
    open='first', high='max', low='min', close='last', count='count'
).dropna()
print(f"16-sec bars: {len(bars)}")

# =============================================================================
# STEP 2: Compute indicators from ticks
# =============================================================================
print(f"\n{'=' * 70}")
print("STEP 2: Compute indicators from tick data")
print("=" * 70)

close = bars['close']
high = bars['high']
low = bars['low']

def compute_rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def compute_bb(series, period, num_std=2):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    bb_pct = (series - lower) / (upper - lower).replace(0, np.nan)
    return bb_pct, upper, lower, sma

def compute_macd(series, fast=12, slow=26, signal=9):
    # Scale periods for 16-sec data: 
    # Standard MACD uses daily close. For 16-sec bars:
    # 5-min equivalent: 1 bar = 16s, so 5min = ~19 bars
    # Use same ratio: fast=12*19=228, slow=26*19=494, signal=9*19=171
    # But that's too long. Let's use "5-min equivalent" periods
    # 12 five-min bars = 60 min = 225 16-sec bars
    # 26 five-min bars = 130 min = ~488 bars
    # 9 five-min bars = 45 min = ~169 bars
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return histogram, macd_line, signal_line

def compute_atr(high, low, close, period):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# Compute at multiple timescales
# "fast" ~ 5 min equivalent (19 bars), "medium" ~ 15 min (56 bars), "slow" ~ 1h (225 bars)
ind = pd.DataFrame(index=bars.index)

# RSI at multiple scales
for p, label in [(19, '5m'), (56, '15m'), (225, '1h')]:
    ind[f'rsi_{label}'] = compute_rsi(close, p)

# Bollinger Bands
for p, label in [(56, '15m'), (225, '1h')]:
    bb_pct, _, _, _ = compute_bb(close, p)
    ind[f'bb_{label}'] = bb_pct

# MACD at different scales
for (f, s, sig, label) in [(19, 56, 14, '5m'), (56, 225, 38, '15m')]:
    hist, _, _ = compute_macd(close, f, s, sig)
    ind[f'macd_{label}'] = hist

# ATR
for p, label in [(56, '15m'), (225, '1h')]:
    atr = compute_atr(high, low, close, p)
    ind[f'atr_pct_{label}'] = atr / close * 100

# EMAs
for p, label in [(19, '5m'), (56, '15m'), (225, '1h'), (450, '2h')]:
    ind[f'ema_{label}'] = close.ewm(span=p, adjust=False).mean()
    ind[f'ema_dist_{label}'] = (close - ind[f'ema_{label}']) / close * 100

# Price-derived features (from our validated exploration)
# Returns
for lb, label in [(1, '16s'), (4, '1m'), (19, '5m'), (56, '15m'), (225, '1h'), (450, '2h')]:
    ind[f'ret_{label}'] = close.pct_change(lb) * 100

# Volatility
ret1 = close.pct_change() * 100
for w, label in [(19, '5m'), (56, '15m'), (225, '1h')]:
    ind[f'vol_{label}'] = ret1.rolling(w).std()

# Channel position
for w, label in [(56, '15m'), (225, '1h'), (450, '2h')]:
    roll_min = close.rolling(w).min()
    roll_max = close.rolling(w).max()
    rng = (roll_max - roll_min).replace(0, np.nan)
    ind[f'chan_{label}'] = (close - roll_min) / rng

# Speed & momentum
ind['speed_5m'] = close.diff(19).abs() / 19
ind['speed_15m'] = close.diff(56).abs() / 56
ind['speed_ratio'] = ind['speed_5m'] / ind['speed_15m'].replace(0, np.nan)

# Consecutive direction
ret_sign = np.sign(close.diff())
ind['consec'] = ret_sign.groupby((ret_sign != ret_sign.shift()).cumsum()).cumcount() + 1
ind['consec'] *= ret_sign

# Time
ind['hour'] = bars.index.hour

# Volume proxy (tick count per bar — more ticks = more activity)
ind['tick_count'] = bars['count']
ind['tick_count_ma'] = bars['count'].rolling(56).mean()
ind['tick_ratio'] = bars['count'] / bars['count'].rolling(56).mean().replace(0, np.nan)

print(f"Indicators computed: {len(ind.columns)} columns")
print(f"Sample columns: {list(ind.columns)[:10]}...")

# =============================================================================
# STEP 3: Create targets via validated simulation
# =============================================================================
print(f"\n{'=' * 70}")
print("STEP 3: Create targets (validated simulation)")
print("=" * 70)

close_arr = close.values
n = len(close_arr)

def simulate_targets(prices, target_pct, sl_pct, ts_act_pct, ts_trail_pct, max_bars=675):
    """Simulate LONG and SHORT trades with trailing stop, return win/loss labels."""
    n = len(prices)
    long_win = np.zeros(n, dtype=int)
    short_win = np.zeros(n, dtype=int)
    long_pnl = np.full(n, np.nan)
    short_pnl = np.full(n, np.nan)
    
    for i in range(n - max_bars):
        entry = prices[i]
        
        # --- LONG ---
        sl = entry * (1 - sl_pct / 100)
        ts_act = entry * (1 + ts_act_pct / 100)
        peak = entry
        ts_on = False
        
        for j in range(i + 1, min(i + max_bars + 1, n)):
            p = prices[j]
            if p <= sl:
                long_pnl[i] = (sl / entry - 1) * 100
                break
            if p > peak:
                peak = p
            if not ts_on and p >= ts_act:
                ts_on = True
            if ts_on:
                trail = peak * (1 - ts_trail_pct / 100)
                if p <= trail:
                    long_pnl[i] = (trail / entry - 1) * 100
                    long_win[i] = 1
                    break
        
        # --- SHORT ---
        sl = entry * (1 + sl_pct / 100)
        ts_act = entry * (1 - ts_act_pct / 100)
        trough = entry
        ts_on = False
        
        for j in range(i + 1, min(i + max_bars + 1, n)):
            p = prices[j]
            if p >= sl:
                short_pnl[i] = (entry / sl - 1) * 100
                break
            if p < trough:
                trough = p
            if not ts_on and p <= ts_act:
                ts_on = True
            if ts_on:
                trail = trough * (1 + ts_trail_pct / 100)
                if p >= trail:
                    short_pnl[i] = (entry / trail - 1) * 100
                    short_win[i] = 1
                    break
    
    return long_win, short_win, long_pnl, short_pnl

long_win, short_win, long_pnl, short_pnl = simulate_targets(
    close_arr, TARGET_PCT, SL_PCT, TS_ACTIVATION, TS_TRAIL, max_bars=675
)

print(f"LONG wins:  {long_win.sum()} / {(~np.isnan(long_pnl)).sum()} resolved "
      f"({long_win.sum()/(~np.isnan(long_pnl)).sum()*100:.1f}%)")
print(f"SHORT wins: {short_win.sum()} / {(~np.isnan(short_pnl)).sum()} resolved "
      f"({short_win.sum()/(~np.isnan(short_pnl)).sum()*100:.1f}%)")

# =============================================================================
# STEP 4: Define feature sets to test
# =============================================================================
print(f"\n{'=' * 70}")
print("STEP 4: Define feature sets")
print("=" * 70)

FEATURE_SETS = {
    'A_price_only': [
        'ret_1m', 'ret_5m', 'ret_15m', 'ret_1h', 'ret_2h',
        'vol_5m', 'vol_15m', 'vol_1h',
        'chan_15m', 'chan_1h', 'chan_2h',
        'speed_ratio', 'consec', 'hour',
    ],
    'B_indicators': [
        'rsi_5m', 'rsi_15m', 'rsi_1h',
        'bb_15m', 'bb_1h',
        'macd_5m', 'macd_15m',
        'atr_pct_15m', 'atr_pct_1h',
        'hour',
    ],
    'C_combined': [
        'ret_5m', 'ret_15m', 'ret_1h',
        'vol_15m', 'vol_1h',
        'chan_1h', 'chan_2h',
        'rsi_15m', 'rsi_1h',
        'bb_1h',
        'macd_15m',
        'atr_pct_1h',
        'speed_ratio', 'hour',
    ],
    'D_momentum_focus': [
        'ret_5m', 'ret_15m', 'ret_1h', 'ret_2h',
        'ema_dist_5m', 'ema_dist_15m', 'ema_dist_1h', 'ema_dist_2h',
        'speed_ratio', 'vol_1h',
        'chan_1h', 'hour',
    ],
    'E_minimal': [
        'ret_1h', 'ret_2h',
        'vol_1h',
        'chan_1h',
        'rsi_1h',
        'hour',
    ],
}

for name, cols in FEATURE_SETS.items():
    print(f"  {name}: {len(cols)} features")

# =============================================================================
# STEP 5: Train/test split — first 7 days / last 3 days
# =============================================================================
print(f"\n{'=' * 70}")
print("STEP 5: Temporal train/test split")
print("=" * 70)

valid = ind.dropna()
# Align targets
long_s = pd.Series(long_win[:len(ind)], index=ind.index).reindex(valid.index).fillna(0).astype(int)
short_s = pd.Series(short_win[:len(ind)], index=ind.index).reindex(valid.index).fillna(0).astype(int)

# Split at ~70% mark
split_date = valid.index[0] + pd.Timedelta(days=7)
train_mask = valid.index < split_date
test_mask = valid.index >= split_date

print(f"Train: {train_mask.sum()} bars ({valid[train_mask].index.min()} to {valid[train_mask].index.max()})")
print(f"Test:  {test_mask.sum()} bars ({valid[test_mask].index.min()} to {valid[test_mask].index.max()})")
print(f"Train LONG WR: {long_s[train_mask].mean()*100:.1f}%, SHORT WR: {short_s[train_mask].mean()*100:.1f}%")
print(f"Test  LONG WR: {long_s[test_mask].mean()*100:.1f}%, SHORT WR: {short_s[test_mask].mean()*100:.1f}%")

# =============================================================================
# STEP 6: Test all model × feature set combinations
# =============================================================================
print(f"\n{'=' * 70}")
print("STEP 6: Model × Feature Set Grid Search")
print("=" * 70)

MODEL_CONFIGS = {
    'GBM_d3': lambda: GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        min_samples_leaf=50, subsample=0.8, random_state=42),
    'GBM_d5': lambda: GradientBoostingClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.03,
        min_samples_leaf=30, subsample=0.8, random_state=42),
    'RF_200': lambda: RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=30, random_state=42),
}

results = []

for direction, y_series in [('LONG', long_s), ('SHORT', short_s)]:
    print(f"\n{'─' * 70}")
    print(f"  {direction}")
    print(f"{'─' * 70}")
    
    y_train = y_series[train_mask]
    y_test = y_series[test_mask]
    
    for feat_name, feat_cols in FEATURE_SETS.items():
        X_train_full = valid[train_mask][feat_cols]
        X_test_full = valid[test_mask][feat_cols]
        
        for model_name, model_fn in MODEL_CONFIGS.items():
            model = model_fn()
            model.fit(X_train_full, y_train)
            probs = model.predict_proba(X_test_full)[:, 1]
            
            # Test at multiple thresholds
            for thresh in [0.50, 0.55, 0.60, 0.65]:
                signal_mask = probs >= thresh
                indices = np.where(signal_mask)[0]
                
                # Enforce gap
                taken = []
                last = -MIN_GAP_BARS
                for idx in indices:
                    if idx - last >= MIN_GAP_BARS:
                        taken.append(idx)
                        last = idx
                
                if len(taken) < 10:
                    continue
                
                wins = sum(y_test.iloc[i] for i in taken)
                n_trades = len(taken)
                wr = wins / n_trades * 100
                
                # Use actual simulated PnL for trades
                pnl_series = long_pnl if direction == 'LONG' else short_pnl
                pnl_aligned = pd.Series(pnl_series[:len(ind)], index=ind.index).reindex(valid.index)
                test_pnl = pnl_aligned[test_mask]
                
                total_pnl_pct = sum(test_pnl.iloc[i] for i in taken if not np.isnan(test_pnl.iloc[i]))
                
                test_days = (X_test_full.index[-1] - X_test_full.index[0]).total_seconds() / 86400
                trades_per_day = n_trades / max(test_days, 1)
                
                results.append({
                    'direction': direction,
                    'features': feat_name,
                    'model': model_name,
                    'threshold': thresh,
                    'n_trades': n_trades,
                    'wins': wins,
                    'wr': wr,
                    'total_pnl_pct': total_pnl_pct,
                    'per_trade_pnl': total_pnl_pct / n_trades,
                    'trades_day': trades_per_day,
                })

# =============================================================================
# STEP 7: Rank and report
# =============================================================================
print(f"\n{'=' * 70}")
print("STEP 7: RESULTS — Sorted by total PnL%")
print("=" * 70)

rdf = pd.DataFrame(results)

for direction in ['LONG', 'SHORT']:
    subset = rdf[rdf['direction'] == direction].sort_values('total_pnl_pct', ascending=False)
    print(f"\n{'─' * 70}")
    print(f"  TOP 15 {direction} CONFIGURATIONS")
    print(f"{'─' * 70}")
    print(f"  {'Features':>20s} {'Model':>8s} {'Thr':>5s} {'Trades':>7s} {'T/day':>6s} "
          f"{'WR':>5s} {'PnL%':>8s} {'$/trade':>8s}")
    
    for _, r in subset.head(15).iterrows():
        print(f"  {r['features']:>20s} {r['model']:>8s} {r['threshold']:>4.0%} "
              f"{r['n_trades']:>7.0f} {r['trades_day']:>5.1f} "
              f"{r['wr']:>4.0f}% {r['total_pnl_pct']:>+7.1f}% "
              f"{r['per_trade_pnl']:>+7.3f}%")

# Also show worst to understand failure modes
for direction in ['LONG', 'SHORT']:
    subset = rdf[rdf['direction'] == direction].sort_values('total_pnl_pct', ascending=True)
    print(f"\n  WORST 5 {direction} (to understand what doesn't work):")
    for _, r in subset.head(5).iterrows():
        print(f"  {r['features']:>20s} {r['model']:>8s} {r['threshold']:>4.0%} "
              f"{r['n_trades']:>7.0f} {r['wr']:>4.0f}% {r['total_pnl_pct']:>+7.1f}%")

# =============================================================================
# STEP 8: Best model deep dive — feature importance + trade-by-trade
# =============================================================================
print(f"\n{'=' * 70}")
print("STEP 8: Best model deep dive")
print("=" * 70)

for direction in ['LONG', 'SHORT']:
    subset = rdf[rdf['direction'] == direction]
    # Filter for reasonable trade frequency
    good = subset[(subset['trades_day'] >= 1) & (subset['trades_day'] <= 20)]
    if len(good) == 0:
        good = subset
    best = good.sort_values('total_pnl_pct', ascending=False).iloc[0]
    
    print(f"\n  BEST {direction}: {best['features']} + {best['model']} @ {best['threshold']:.0%}")
    print(f"  {best['n_trades']:.0f} trades ({best['trades_day']:.1f}/day), "
          f"{best['wr']:.0f}% WR, {best['total_pnl_pct']:+.1f}% total")
    
    # Retrain and show importances
    feat_cols = FEATURE_SETS[best['features']]
    model = MODEL_CONFIGS[best['model']]()
    y_train = long_s[train_mask] if direction == 'LONG' else short_s[train_mask]
    model.fit(valid[train_mask][feat_cols], y_train)
    
    if hasattr(model, 'feature_importances_'):
        imps = sorted(zip(feat_cols, model.feature_importances_), key=lambda x: -x[1])
        print(f"\n  Feature importances:")
        for name, imp in imps:
            print(f"    {name:20s} {imp:.3f} {'█' * int(imp * 40)}")

# =============================================================================
# STEP 9: Compare vs current system baseline
# =============================================================================
print(f"\n{'=' * 70}")
print("STEP 9: Baseline comparison")
print("=" * 70)

# Current system on test period: use the actual trades from trades.db
import sqlite3
conn = sqlite3.connect('trades.db')
actual = pd.read_sql("""
    SELECT entry_time, direction, pnl_dollar, exit_reason 
    FROM trades WHERE model_id LIKE '%pct%'
    ORDER BY entry_time
""", conn)
conn.close()

actual['entry_time'] = pd.to_datetime(actual['entry_time'], format='ISO8601')
test_start = valid[test_mask].index.min()
test_end = valid[test_mask].index.max()
actual_test = actual[(actual['entry_time'] >= test_start) & (actual['entry_time'] <= test_end)]

if len(actual_test) > 0:
    curr_wins = (actual_test['pnl_dollar'] > 0).sum()
    curr_losses = (actual_test['pnl_dollar'] <= 0).sum()
    curr_pnl = actual_test['pnl_dollar'].sum()
    curr_wr = curr_wins / len(actual_test) * 100
    
    print(f"\n  CURRENT SYSTEM (actual trades in test period):")
    print(f"  {len(actual_test)} trades, {curr_wins}W/{curr_losses}L ({curr_wr:.0f}% WR), ${curr_pnl:+.2f}")
    
    for direction in ['LONG', 'SHORT']:
        sub = actual_test[actual_test['direction'] == direction]
        if len(sub) > 0:
            w = (sub['pnl_dollar'] > 0).sum()
            l = (sub['pnl_dollar'] <= 0).sum()
            print(f"    {direction}: {len(sub)} trades, {w}W/{l}L ({w/len(sub)*100:.0f}%), ${sub['pnl_dollar'].sum():+.2f}")
else:
    print("  No actual trades in test period")

print(f"\n  BEST NEW MODELS (on same test period):")
for direction in ['LONG', 'SHORT']:
    subset = rdf[rdf['direction'] == direction]
    good = subset[(subset['trades_day'] >= 1) & (subset['trades_day'] <= 20)]
    if len(good) == 0:
        good = subset
    best = good.sort_values('total_pnl_pct', ascending=False).iloc[0]
    
    # Convert pnl% to approximate dollars (avg entry ~$67,500, contract ~$11)
    avg_price = close[test_mask].mean()
    contract = avg_price * 0.001  # rough contract value
    approx_dollar = best['total_pnl_pct'] / 100 * contract * best['n_trades']
    
    print(f"    {direction}: {best['n_trades']:.0f} trades, {best['wr']:.0f}% WR, "
          f"{best['total_pnl_pct']:+.1f}% (~${approx_dollar:+.0f}), "
          f"{best['features']}+{best['model']}@{best['threshold']:.0%}")

print("\n" + "=" * 70)
print("EXPLORATION COMPLETE — No changes implemented")
print("=" * 70)
