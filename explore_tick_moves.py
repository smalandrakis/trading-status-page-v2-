#!/usr/bin/env python3
"""
Explore 0.5% price movements in BTC tick data.
Step 1: Load 2-sec ticks, resample to 16-sec bars
Step 2: Identify all 0.5% moves (UP and DOWN) using rolling windows
Step 3: Characterize what price looked like BEFORE these moves
Step 4: Train a simple non-linear model using price-only features to predict 0.5% moves
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# STEP 1: Load and resample tick data
# =============================================================================
print("="*60)
print("STEP 1: Loading tick data")
print("="*60)

df = pd.read_csv('logs/btc_price_ticks.csv', parse_dates=['timestamp'])
df = df.set_index('timestamp').sort_index()
df = df[df['price'] > 0]  # Clean
print(f"Raw 2-sec ticks: {len(df)}, {df.index.min()} to {df.index.max()}")

# Resample to 16-sec OHLC bars
bars_16s = df['price'].resample('16s').ohlc().dropna()
bars_16s.columns = ['open', 'high', 'low', 'close']
print(f"16-sec bars: {len(bars_16s)}")

# Also keep 2-sec close prices for fine-grained analysis
prices_2s = df['price'].copy()
print(f"2-sec prices: {len(prices_2s)}")

# =============================================================================
# STEP 2: Identify 0.5% moves
# =============================================================================
print(f"\n{'='*60}")
print("STEP 2: Identifying 0.5% price moves")
print("="*60)

def find_moves(close, min_move_pct=0.5, window_bars=None):
    """Find all points where price subsequently moves >=min_move_pct% 
    before moving >=min_move_pct% in the opposite direction.
    
    Uses variable lookahead windows to find moves of different speeds.
    """
    prices = close.values
    n = len(prices)
    
    # For each bar, check: does price reach +0.5% before -0.5%? (LONG)
    # And: does price reach -0.5% before +0.5%? (SHORT)
    long_wins = np.zeros(n, dtype=int)
    short_wins = np.zeros(n, dtype=int)
    long_bars_to_target = np.full(n, np.nan)  # How many bars to reach target
    short_bars_to_target = np.full(n, np.nan)
    
    max_look = window_bars if window_bars else min(len(prices) - 1, 1000)
    
    for i in range(n - 1):
        entry = prices[i]
        tp_up = entry * (1 + min_move_pct / 100)
        sl_up = entry * (1 - min_move_pct / 100)
        tp_dn = entry * (1 - min_move_pct / 100)
        sl_dn = entry * (1 + min_move_pct / 100)
        
        # Check LONG
        for j in range(i + 1, min(i + max_look + 1, n)):
            if prices[j] >= tp_up:
                long_wins[i] = 1
                long_bars_to_target[i] = j - i
                break
            if prices[j] <= sl_up:
                break
        
        # Check SHORT
        for j in range(i + 1, min(i + max_look + 1, n)):
            if prices[j] <= tp_dn:
                short_wins[i] = 1
                short_bars_to_target[i] = j - i
                break
            if prices[j] >= sl_dn:
                break
    
    return long_wins, short_wins, long_bars_to_target, short_bars_to_target

# Find moves on 16-sec bars (more manageable than 90K 2-sec ticks)
print("\nSearching for 0.5% moves on 16-sec bars...")
long_w, short_w, long_ttg, short_ttg = find_moves(bars_16s['close'], 0.5, window_bars=675)  # 675 bars = 3 hours

print(f"\n  LONG moves (price goes UP 0.5% before DOWN 0.5%):")
print(f"    Count: {long_w.sum()} / {len(long_w)} bars ({long_w.mean()*100:.1f}%)")
print(f"    Avg bars to target: {np.nanmean(long_ttg):.0f} bars ({np.nanmean(long_ttg)*16/60:.0f} min)")
print(f"    Median bars to target: {np.nanmedian(long_ttg[~np.isnan(long_ttg)]):.0f} bars ({np.nanmedian(long_ttg[~np.isnan(long_ttg)])*16/60:.0f} min)")

print(f"\n  SHORT moves (price goes DOWN 0.5% before UP 0.5%):")
print(f"    Count: {short_w.sum()} / {len(short_w)} bars ({short_w.mean()*100:.1f}%)")
print(f"    Avg bars to target: {np.nanmean(short_ttg):.0f} bars ({np.nanmean(short_ttg)*16/60:.0f} min)")
print(f"    Median bars to target: {np.nanmedian(short_ttg[~np.isnan(short_ttg)]):.0f} bars ({np.nanmedian(short_ttg[~np.isnan(short_ttg)])*16/60:.0f} min)")

neither = len(long_w) - long_w.sum() - short_w.sum() + ((long_w == 1) & (short_w == 1)).sum()
print(f"\n  Neither (chop): {(long_w + short_w == 0).sum()} bars ({(long_w + short_w == 0).mean()*100:.1f}%)")

# =============================================================================
# STEP 3: Characterize price action BEFORE these moves
# =============================================================================
print(f"\n{'='*60}")
print("STEP 3: Price features before 0.5% moves")
print("="*60)

close = bars_16s['close']

def build_price_features(close_series):
    """Build features using ONLY price data."""
    f = pd.DataFrame(index=close_series.index)
    
    # Returns at various lookbacks
    for lb in [1, 2, 4, 8, 15, 30, 60, 120]:
        f[f'ret_{lb}'] = close_series.pct_change(lb) * 100
    
    # Volatility (std of returns) at various windows
    ret1 = close_series.pct_change() * 100
    for w in [8, 15, 30, 60]:
        f[f'vol_{w}'] = ret1.rolling(w).std()
    
    # Price relative to rolling min/max (channel position)
    for w in [30, 60, 120]:
        roll_min = close_series.rolling(w).min()
        roll_max = close_series.rolling(w).max()
        rng = roll_max - roll_min
        f[f'chan_pos_{w}'] = ((close_series - roll_min) / rng.replace(0, np.nan))
    
    # Consecutive up/down bars
    ret_sign = np.sign(close_series.diff())
    f['consec_dir'] = ret_sign.groupby((ret_sign != ret_sign.shift()).cumsum()).cumcount() + 1
    f['consec_dir'] *= ret_sign
    
    # Speed: price change per bar recently vs average
    f['speed_8'] = close_series.diff(8).abs() / 8
    f['speed_30'] = close_series.diff(30).abs() / 30
    f['speed_ratio'] = f['speed_8'] / f['speed_30'].replace(0, np.nan)
    
    # Hour of day
    f['hour'] = close_series.index.hour
    
    return f

feat = build_price_features(close)
print(f"Price features: {list(feat.columns)}")
print(f"Shape: {feat.shape}")

# Show feature means for LONG wins vs no-wins
valid = feat.dropna()
long_valid = long_w[feat.index.isin(valid.index)]
short_valid = short_w[feat.index.isin(valid.index)]

# Ensure alignment
valid = valid.iloc[:len(long_valid)]
long_valid = long_valid[:len(valid)]
short_valid = short_valid[:len(valid)]

print(f"\nValid bars for modeling: {len(valid)}")
print(f"  LONG wins: {long_valid.sum()} ({long_valid.mean()*100:.1f}%)")
print(f"  SHORT wins: {short_valid.sum()} ({short_valid.mean()*100:.1f}%)")

print(f"\nFeature means — LONG winners vs losers:")
print(f"  {'Feature':20s} {'Win':>8s} {'Lose':>8s} {'Diff':>8s}")
for col in feat.columns:
    if col in valid.columns:
        w = valid.loc[long_valid == 1, col].mean()
        l = valid.loc[long_valid == 0, col].mean()
        d = w - l
        if abs(d) > 0.01:
            print(f"  {col:20s} {w:8.3f} {l:8.3f} {d:+8.3f}")

print(f"\nFeature means — SHORT winners vs losers:")
print(f"  {'Feature':20s} {'Win':>8s} {'Lose':>8s} {'Diff':>8s}")
for col in feat.columns:
    if col in valid.columns:
        w = valid.loc[short_valid == 1, col].mean()
        l = valid.loc[short_valid == 0, col].mean()
        d = w - l
        if abs(d) > 0.01:
            print(f"  {col:20s} {w:8.3f} {l:8.3f} {d:+8.3f}")

# =============================================================================
# STEP 4: Train non-linear model (price-only features)
# =============================================================================
print(f"\n{'='*60}")
print("STEP 4: Train GBM model (price-only features)")
print("="*60)

# Temporal split: first 70% train, last 30% test
split_idx = int(len(valid) * 0.70)
X_train = valid.iloc[:split_idx]
X_test = valid.iloc[split_idx:]
y_long_train = long_valid[:split_idx]
y_long_test = long_valid[split_idx:]
y_short_train = short_valid[:split_idx]
y_short_test = short_valid[split_idx:]

print(f"Train: {len(X_train)} bars ({X_train.index.min()} to {X_train.index.max()})")
print(f"Test:  {len(X_test)} bars ({X_test.index.min()} to {X_test.index.max()})")

for direction, y_tr, y_te in [('LONG', y_long_train, y_long_test), 
                                ('SHORT', y_short_train, y_short_test)]:
    print(f"\n--- {direction} ---")
    print(f"  Train: {y_tr.sum()}W / {len(y_tr)} ({y_tr.mean()*100:.1f}%)")
    print(f"  Test:  {y_te.sum()}W / {len(y_te)} ({y_te.mean()*100:.1f}%)")
    
    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        min_samples_leaf=30, subsample=0.8, random_state=42
    )
    model.fit(X_train, y_tr)
    
    probs = model.predict_proba(X_test)[:, 1]
    
    # Feature importance
    imps = sorted(zip(valid.columns, model.feature_importances_), key=lambda x: -x[1])[:8]
    print(f"\n  Top features:")
    for name, imp in imps:
        print(f"    {name:20s} {imp:.3f} {'█' * int(imp * 40)}")
    
    # Threshold analysis with signal gap (avoid clustering)
    print(f"\n  Threshold analysis:")
    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70]:
        mask = probs >= thresh
        indices = np.where(mask)[0]
        
        # Enforce minimum gap
        taken = []
        last = -30  # ~8 min gap in 16-sec bars
        for idx in indices:
            if idx - last >= 30:
                taken.append(idx)
                last = idx
        
        if len(taken) == 0:
            continue
        
        wins = sum(y_te.iloc[i] if hasattr(y_te, 'iloc') else y_te[i] for i in taken)
        n_trades = len(taken)
        wr = wins / n_trades * 100
        # PnL: win = +0.5%, loss = -0.5% (symmetric for now)
        net_pct = wins * 0.5 - (n_trades - wins) * 0.5
        per_day = n_trades / max((X_test.index[-1] - X_test.index[0]).days, 1)
        
        print(f"    >{thresh:.0%}: {n_trades:4d} trades ({per_day:.1f}/day), "
              f"{wr:.0f}% WR, {wins}W/{n_trades-wins}L, ~{net_pct:+.1f}% total")

print("\nDone.")
