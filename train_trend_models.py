#!/usr/bin/env python3
"""
Train trend-following models for BTC LONG/SHORT signals.
Uses 5-min bar data with key indicators. Simple features, proper train/test split.

Approach: Instead of mean-reversion (current bot), these models follow momentum.
- LONG: enter when trend is building upward
- SHORT: enter when trend is building downward

Target: price moves ≥0.20% in the desired direction within N bars (2-6 hours).

Data: BTC_2025_full_features.parquet (Jan-Dec 2025, 97K 5-min bars)
      + BTC_features.parquet (Mar 2-9 2026, ~2K bars)
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================
TARGET_PCT = 0.20          # Target move: 0.20%
LOOKAHEAD_BARS = 72        # 6 hours of 5-min bars
SL_PCT = 0.20              # Stop loss: 0.20%
MIN_SIGNAL_GAP = 12        # Min 1 hour between signals (avoid clustering)

# Features — keeping it simple: momentum + volatility + trend
FEATURE_COLS = [
    'momentum_rsi',         # RSI 14
    'volatility_bbp',       # Bollinger Band %B
    'trend_macd_diff',      # MACD histogram
    'volatility_atr',       # ATR (absolute)
    'Close',                # Price (for ATR%)
]

# Derived features we'll compute
# - atr_pct: ATR as % of price
# - rsi_slope: RSI change over last 6 bars (30 min)
# - price_momentum_1h: % change over 12 bars
# - price_momentum_4h: % change over 48 bars
# - bb_trend: BB%B change over 6 bars
# - hour: hour of day
# - macd_crossover: sign change in MACD histogram

def load_data():
    """Load and combine parquet files."""
    print("Loading data...")
    df1 = pd.read_parquet('data/BTC_2025_full_features.parquet')
    
    # Try to load recent data too
    try:
        df2 = pd.read_parquet('data/BTC_features.parquet')
        # Only keep non-overlapping rows
        df2 = df2[df2.index > df1.index.max()]
        if len(df2) > 0:
            df = pd.concat([df1, df2])
            print(f"  Combined: {len(df1)} + {len(df2)} = {len(df)} rows")
        else:
            df = df1
    except:
        df = df1
    
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print(f"  Shape: {df.shape}")
    return df

def engineer_features(df):
    """Create simple trend-following features."""
    print("Engineering features...")
    feat = pd.DataFrame(index=df.index)
    
    # Raw indicators
    feat['rsi'] = df['momentum_rsi']
    feat['bb_pct'] = df['volatility_bbp']
    feat['macd_hist'] = df['trend_macd_diff']
    feat['atr_pct'] = df['volatility_atr'] / df['Close'] * 100
    
    # Momentum features
    feat['price_mom_30m'] = df['Close'].pct_change(6) * 100    # 30-min momentum
    feat['price_mom_1h'] = df['Close'].pct_change(12) * 100    # 1-hour momentum
    feat['price_mom_4h'] = df['Close'].pct_change(48) * 100    # 4-hour momentum
    
    # Trend features
    feat['rsi_slope'] = df['momentum_rsi'].diff(6)              # RSI trend (30 min)
    feat['bb_slope'] = df['volatility_bbp'].diff(6)             # BB%B trend (30 min)
    feat['macd_slope'] = df['trend_macd_diff'].diff(6)          # MACD histogram trend
    
    # Time features
    feat['hour'] = df.index.hour
    
    # Volatility context
    feat['vol_ratio'] = df['volatility_atr'] / df['volatility_atr'].rolling(48).mean()
    
    print(f"  Features: {list(feat.columns)}")
    print(f"  Shape: {feat.shape}")
    return feat

def create_targets(df):
    """Create LONG and SHORT target labels.
    
    LONG target = 1 if price goes UP ≥ TARGET_PCT% before going DOWN ≥ SL_PCT%
                  within LOOKAHEAD_BARS
    SHORT target = 1 if price goes DOWN ≥ TARGET_PCT% before going UP ≥ SL_PCT%
                   within LOOKAHEAD_BARS
    """
    print(f"Creating targets (target={TARGET_PCT}%, SL={SL_PCT}%, lookahead={LOOKAHEAD_BARS} bars)...")
    close = df['Close'].values
    n = len(close)
    long_target = np.zeros(n, dtype=int)
    short_target = np.zeros(n, dtype=int)
    
    for i in range(n - LOOKAHEAD_BARS):
        entry = close[i]
        tp_long = entry * (1 + TARGET_PCT / 100)
        sl_long = entry * (1 - SL_PCT / 100)
        tp_short = entry * (1 - TARGET_PCT / 100)
        sl_short = entry * (1 + SL_PCT / 100)
        
        # Simulate LONG
        for j in range(i + 1, min(i + LOOKAHEAD_BARS + 1, n)):
            if close[j] >= tp_long:
                long_target[i] = 1
                break
            if close[j] <= sl_long:
                break
        
        # Simulate SHORT
        for j in range(i + 1, min(i + LOOKAHEAD_BARS + 1, n)):
            if close[j] <= tp_short:
                short_target[i] = 1
                break
            if close[j] >= sl_short:
                break
    
    print(f"  LONG targets: {long_target.sum()} wins / {n} bars ({long_target.mean()*100:.1f}%)")
    print(f"  SHORT targets: {short_target.sum()} wins / {n} bars ({short_target.mean()*100:.1f}%)")
    return long_target, short_target

def train_and_evaluate(feat, targets, direction, train_end_date):
    """Train model on data before train_end_date, test on data after."""
    print(f"\n{'='*60}")
    print(f"  {direction} MODEL")
    print(f"{'='*60}")
    
    # Clean data
    valid = feat.dropna()
    targets_aligned = targets.reindex(valid.index)
    
    # Temporal split
    train_mask = valid.index < train_end_date
    test_mask = valid.index >= train_end_date
    
    X_train = valid[train_mask]
    y_train = targets_aligned[train_mask]
    X_test = valid[test_mask]
    y_test = targets_aligned[test_mask]
    
    print(f"  Train: {len(X_train)} bars ({X_train.index.min()} to {X_train.index.max()})")
    print(f"  Test:  {len(X_test)} bars ({X_test.index.min()} to {X_test.index.max()})")
    print(f"  Train win rate: {y_train.mean()*100:.1f}%")
    print(f"  Test win rate:  {y_test.mean()*100:.1f}%")
    
    if len(X_train) < 100 or len(X_test) < 100:
        print("  ERROR: Not enough data!")
        return None, None, None
    
    # Train
    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        min_samples_leaf=50, subsample=0.8, random_state=42
    )
    model.fit(X_train, y_train)
    
    # Predict on test
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Feature importance
    importances = sorted(zip(valid.columns, model.feature_importances_), key=lambda x: -x[1])
    print(f"\n  Feature importances:")
    for feat_name, imp in importances:
        bar = '█' * int(imp * 40)
        print(f"    {feat_name:20s} {imp:.3f} {bar}")
    
    # Threshold analysis — simulate actual trading
    print(f"\n  Threshold analysis (test set):")
    print(f"  {'Thresh':>7s} {'Signals':>8s} {'WR':>6s} {'AvgGap':>7s}")
    
    best_thresh = 0.50
    best_pnl = -999
    
    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70]:
        signal_mask = y_prob >= thresh
        if signal_mask.sum() == 0:
            continue
        
        # Simulate with minimum gap between signals
        signal_indices = np.where(signal_mask)[0]
        taken = []
        last_taken = -MIN_SIGNAL_GAP
        for idx in signal_indices:
            if idx - last_taken >= MIN_SIGNAL_GAP:
                taken.append(idx)
                last_taken = idx
        
        if len(taken) == 0:
            continue
        
        wins = sum(y_test.iloc[i] for i in taken)
        losses = len(taken) - wins
        wr = wins / len(taken) * 100
        
        # Approximate PnL: wins get +TARGET_PCT%, losses get -SL_PCT%
        approx_pnl_per_trade = (wins * TARGET_PCT - losses * SL_PCT) / len(taken)
        total_pnl_pct = wins * TARGET_PCT - losses * SL_PCT
        
        avg_gap = len(X_test) / max(len(taken), 1)
        
        print(f"  {thresh:>6.0%} {len(taken):>8d} {wr:>5.0f}% {avg_gap:>6.0f}bars  "
              f"({wins}W/{losses}L, ~{total_pnl_pct:+.1f}% total)")
        
        if total_pnl_pct > best_pnl:
            best_pnl = total_pnl_pct
            best_thresh = thresh
    
    print(f"\n  Best threshold: {best_thresh:.0%}")
    
    return model, X_test, y_test

def simulate_trades(df_close, feat, model, direction, threshold, start_date):
    """Full trade simulation on test data with actual prices."""
    print(f"\n  Simulating {direction} trades (threshold={threshold:.0%})...")
    
    valid = feat.dropna()
    test_feat = valid[valid.index >= start_date]
    close = df_close[df_close.index >= start_date]
    
    # Align
    common_idx = test_feat.index.intersection(close.index)
    test_feat = test_feat.loc[common_idx]
    close = close.loc[common_idx]
    
    probs = model.predict_proba(test_feat)[:, 1]
    
    trades = []
    i = 0
    while i < len(close) - LOOKAHEAD_BARS:
        if probs[i] < threshold:
            i += 1
            continue
        
        entry_price = close.iloc[i]
        entry_time = close.index[i]
        
        if direction == 'LONG':
            tp = entry_price * (1 + TARGET_PCT / 100)
            sl = entry_price * (1 - SL_PCT / 100)
        else:
            tp = entry_price * (1 - TARGET_PCT / 100)
            sl = entry_price * (1 + SL_PCT / 100)
        
        # Simulate trade
        exit_reason = 'TIMEOUT'
        exit_price = close.iloc[min(i + LOOKAHEAD_BARS, len(close) - 1)]
        
        for j in range(i + 1, min(i + LOOKAHEAD_BARS + 1, len(close))):
            p = close.iloc[j]
            if direction == 'LONG':
                if p >= tp:
                    exit_price = tp
                    exit_reason = 'TP'
                    break
                if p <= sl:
                    exit_price = sl
                    exit_reason = 'SL'
                    break
            else:
                if p <= tp:
                    exit_price = tp
                    exit_reason = 'TP'
                    break
                if p >= sl:
                    exit_price = sl
                    exit_reason = 'SL'
                    break
        
        if direction == 'LONG':
            pnl_pct = (exit_price / entry_price - 1) * 100
        else:
            pnl_pct = (entry_price / exit_price - 1) * 100
        
        # Approximate dollar PnL (0.1 BTC contract)
        pnl_dollar = pnl_pct / 100 * entry_price * 0.1
        
        trades.append({
            'entry_time': entry_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'pnl_pct': pnl_pct,
            'pnl_dollar': pnl_dollar,
            'prob': probs[i],
        })
        
        # Skip ahead MIN_SIGNAL_GAP bars after trade
        i += MIN_SIGNAL_GAP
        continue
    
    if not trades:
        print("  No trades generated!")
        return None
    
    tdf = pd.DataFrame(trades)
    wins = tdf[tdf['pnl_dollar'] > 0]
    losses = tdf[tdf['pnl_dollar'] <= 0]
    
    print(f"\n  TRADE SIMULATION RESULTS ({direction}):")
    print(f"  Total trades: {len(tdf)}")
    print(f"  Wins: {len(wins)} ({len(wins)/len(tdf)*100:.0f}%)")
    print(f"  Losses: {len(losses)} ({len(losses)/len(tdf)*100:.0f}%)")
    print(f"  Total PnL: ${tdf['pnl_dollar'].sum():+.2f}")
    print(f"  Avg win:  ${wins['pnl_dollar'].mean():+.2f}" if len(wins) > 0 else "  No wins")
    print(f"  Avg loss: ${losses['pnl_dollar'].mean():+.2f}" if len(losses) > 0 else "  No losses")
    print(f"  Avg trade: ${tdf['pnl_dollar'].mean():+.2f}")
    print(f"  Trades/day: {len(tdf) / max((tdf['entry_time'].max() - tdf['entry_time'].min()).days, 1):.1f}")
    
    # Show exit reason breakdown
    print(f"\n  Exit reasons:")
    for reason in ['TP', 'SL', 'TIMEOUT']:
        subset = tdf[tdf['exit_reason'] == reason]
        if len(subset) > 0:
            print(f"    {reason}: {len(subset)} trades, ${subset['pnl_dollar'].sum():+.2f}")
    
    # Show last 10 trades
    print(f"\n  Last 10 trades:")
    for _, t in tdf.tail(10).iterrows():
        w = 'WIN ' if t['pnl_dollar'] > 0 else 'LOSS'
        print(f"    {t['entry_time']} ${t['pnl_dollar']:+.2f} {w} {t['exit_reason']} (prob={t['prob']:.0%})")
    
    return tdf

def main():
    df = load_data()
    feat = engineer_features(df)
    
    # Use last 3 months for training scope (Oct-Dec 2025)
    # Train: Oct 1 - Nov 15
    # Test: Nov 15 - Dec 5
    TRAIN_END = '2025-11-15'
    
    # Create targets
    long_target, short_target = create_targets(df)
    
    # Align targets with features
    long_s = pd.Series(long_target, index=df.index)
    short_s = pd.Series(short_target, index=df.index)
    
    # Only use last 3 months
    start_date = '2025-10-01'
    mask = feat.index >= start_date
    feat_recent = feat[mask]
    long_recent = long_s[mask]
    short_recent = short_s[mask]
    
    print(f"\nUsing data from {start_date} to {feat_recent.index.max()}")
    print(f"Total bars: {len(feat_recent)}")
    
    # Train LONG model
    long_model, X_test_l, y_test_l = train_and_evaluate(
        feat_recent, long_recent, 'LONG', TRAIN_END)
    
    # Train SHORT model
    short_model, X_test_s, y_test_s = train_and_evaluate(
        feat_recent, short_recent, 'SHORT', TRAIN_END)
    
    # Full trade simulation on test period
    if long_model:
        simulate_trades(df['Close'], feat, long_model, 'LONG', 0.55, TRAIN_END)
    if short_model:
        simulate_trades(df['Close'], feat, short_model, 'SHORT', 0.55, TRAIN_END)
    
    # Save models
    os.makedirs('models', exist_ok=True)
    if long_model:
        joblib.dump(long_model, 'models/trend_long.pkl')
        print(f"\n✓ LONG trend model saved to models/trend_long.pkl")
    if short_model:
        joblib.dump(short_model, 'models/trend_short.pkl')
        print(f"\n✓ SHORT trend model saved to models/trend_short.pkl")

if __name__ == '__main__':
    main()
