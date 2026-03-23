#!/usr/bin/env python3
"""
MNQ ML Direction Prediction Model

Build a sophisticated ML model to predict small directional movements.
Use all 211 features + synthetic features + 15-sec data for validation.

Strategy:
1. Train model to predict direction over next N bars
2. Use trailing stops to capture profits when direction is correct
3. Validate with 15-sec data for realistic simulation
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

# Target configurations
TARGETS = [
    {'name': 'up_0.2pct_12bars', 'threshold': 0.2, 'bars': 12, 'direction': 'LONG'},
    {'name': 'up_0.3pct_24bars', 'threshold': 0.3, 'bars': 24, 'direction': 'LONG'},
    {'name': 'up_0.5pct_48bars', 'threshold': 0.5, 'bars': 48, 'direction': 'LONG'},
    {'name': 'down_0.2pct_12bars', 'threshold': 0.2, 'bars': 12, 'direction': 'SHORT'},
    {'name': 'down_0.3pct_24bars', 'threshold': 0.3, 'bars': 24, 'direction': 'SHORT'},
    {'name': 'down_0.5pct_48bars', 'threshold': 0.5, 'bars': 48, 'direction': 'SHORT'},
]


def load_data():
    """Load feature data."""
    df = pd.read_parquet('data/QQQ_features.parquet')
    print(f"Loaded {len(df)} bars with {len(df.columns)} features")
    return df


def add_synthetic_features(df):
    """Add additional synthetic features for better prediction."""
    df = df.copy()
    
    # Cross-indicator features
    df['rsi_macd_cross'] = df['momentum_rsi'] * np.sign(df['trend_macd_diff'])
    df['bb_rsi_combo'] = df['volatility_bbp'] * df['momentum_rsi'] / 100
    df['adx_trend_strength'] = df['trend_adx'] * np.sign(df['trend_adx_pos'] - df['trend_adx_neg'])
    
    # Momentum divergence
    df['price_rsi_div'] = df['return_5bar'] - (df['momentum_rsi'] - df['momentum_rsi_lag5']) / 100
    df['price_macd_div'] = df['return_5bar'] - df['trend_macd_change_5']
    
    # Volatility regime
    df['vol_regime'] = df['volatility_atr'] / df['volatility_atr'].rolling(50).mean()
    df['vol_expanding'] = (df['volatility_atr'] > df['volatility_atr_lag5']).astype(int)
    
    # Price structure
    df['higher_high'] = (df['High'] > df['High'].shift(1)).astype(int)
    df['higher_low'] = (df['Low'] > df['Low'].shift(1)).astype(int)
    df['lower_high'] = (df['High'] < df['High'].shift(1)).astype(int)
    df['lower_low'] = (df['Low'] < df['Low'].shift(1)).astype(int)
    
    # Trend structure
    df['uptrend_bars'] = (df['higher_high'] & df['higher_low']).rolling(5).sum()
    df['downtrend_bars'] = (df['lower_high'] & df['lower_low']).rolling(5).sum()
    
    # Mean reversion signals
    df['oversold_rsi'] = (df['momentum_rsi'] < 30).astype(int)
    df['overbought_rsi'] = (df['momentum_rsi'] > 70).astype(int)
    df['bb_lower_touch'] = (df['volatility_bbp'] < 0.1).astype(int)
    df['bb_upper_touch'] = (df['volatility_bbp'] > 0.9).astype(int)
    
    # Momentum acceleration
    df['rsi_accel'] = df['momentum_rsi_change_1'] - df['momentum_rsi_change_1'].shift(1)
    df['macd_accel'] = df['trend_macd_change_1'] - df['trend_macd_change_1'].shift(1)
    
    # Volume features
    df['vol_spike'] = df['Volume'] / df['Volume'].rolling(20).mean()
    df['vol_trend'] = df['Volume'].rolling(5).mean() / df['Volume'].rolling(20).mean()
    
    # Time-based features
    df['morning_session'] = ((df['hour'] >= 9) & (df['hour'] < 11)).astype(int)
    df['midday_session'] = ((df['hour'] >= 11) & (df['hour'] < 14)).astype(int)
    df['afternoon_session'] = ((df['hour'] >= 14) & (df['hour'] < 16)).astype(int)
    
    # Candle patterns
    df['body_size'] = abs(df['Close'] - df['Open']) / df['Close'] * 100
    df['upper_wick'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Close'] * 100
    df['lower_wick'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Close'] * 100
    df['bullish_candle'] = (df['Close'] > df['Open']).astype(int)
    
    # Consecutive patterns
    df['consec_up'] = df['bullish_candle'].rolling(3).sum()
    df['consec_down'] = (1 - df['bullish_candle']).rolling(3).sum()
    
    return df


def create_labels(df, threshold_pct, horizon_bars, direction):
    """Create binary labels for direction prediction."""
    
    if direction == 'LONG':
        # Max gain over horizon
        future_max = df['High'].rolling(horizon_bars).max().shift(-horizon_bars)
        df['target'] = ((future_max / df['Close'] - 1) * 100 >= threshold_pct).astype(int)
    else:
        # Max drop over horizon
        future_min = df['Low'].rolling(horizon_bars).min().shift(-horizon_bars)
        df['target'] = ((df['Close'] / future_min - 1) * 100 >= threshold_pct).astype(int)
    
    return df


def get_feature_columns(df):
    """Get all feature columns (exclude OHLCV and target)."""
    exclude = ['Open', 'High', 'Low', 'Close', 'Volume', 'target']
    features = [c for c in df.columns if c not in exclude and not c.startswith('future_')]
    return features


def train_model(df, target_config, model_type='gb'):
    """Train ML model for direction prediction."""
    
    # Create labels
    df = create_labels(df, target_config['threshold'], target_config['bars'], target_config['direction'])
    
    # Get features
    feature_cols = get_feature_columns(df)
    
    # Drop rows with NaN
    df_clean = df.dropna(subset=feature_cols + ['target'])
    
    X = df_clean[feature_cols]
    y = df_clean['target']
    
    print(f"\n{'='*60}")
    print(f"Training: {target_config['name']}")
    print(f"{'='*60}")
    print(f"Samples: {len(X)}, Features: {len(feature_cols)}")
    print(f"Positive class: {y.sum()} ({y.mean()*100:.1f}%)")
    
    # Time-series split (use last 20% for test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    if model_type == 'gb':
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            min_samples_leaf=20,
            subsample=0.8,
            random_state=42
        )
    else:
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1
        )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {accuracy:.1%}")
    
    # Precision at different thresholds
    results = {}
    for thresh in [0.5, 0.55, 0.6, 0.65, 0.7]:
        y_pred_thresh = (y_prob >= thresh).astype(int)
        if y_pred_thresh.sum() > 0:
            prec = precision_score(y_test, y_pred_thresh, zero_division=0)
            signals = y_pred_thresh.sum()
            results[thresh] = {'precision': prec, 'signals': signals}
            print(f"  @{thresh:.0%} threshold: precision={prec:.1%}, signals={signals}")
    
    return model, scaler, feature_cols, results, df_clean.index[split_idx:]


def backtest_with_15sec(model, scaler, feature_cols, df_5min, df_15sec, 
                        target_config, prob_threshold=0.6,
                        sl_pct=0.5, trailing_pct=0.2, trailing_activation=0.15):
    """
    Backtest using 15-sec data for realistic execution.
    
    Strategy:
    1. Get signal from 5-min model
    2. Enter at next bar open
    3. Use trailing stop with 15-sec precision
    """
    
    direction = target_config['direction']
    horizon_bars = target_config['bars']
    max_hold_minutes = horizon_bars * 5  # Convert 5-min bars to minutes
    
    # Get predictions
    df_5min = df_5min.copy()
    X = df_5min[feature_cols].dropna()
    
    if len(X) == 0:
        return pd.DataFrame()
    
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[:, 1]
    
    df_5min.loc[X.index, 'prob'] = probs
    
    # Get signal bars
    signal_mask = df_5min['prob'] >= prob_threshold
    signal_times = df_5min[signal_mask].index
    
    trades = []
    last_trade_time = None
    cooldown = timedelta(minutes=30)
    
    for sig_time in signal_times:
        # Cooldown check
        if last_trade_time and (sig_time - last_trade_time) < cooldown:
            continue
        
        # Get entry price
        entry_price = df_5min.loc[sig_time, 'Close']
        prob = df_5min.loc[sig_time, 'prob']
        
        # Get 15-sec bars for this trade
        end_time = sig_time + timedelta(minutes=max_hold_minutes)
        trade_bars = df_15sec[(df_15sec.index > sig_time) & (df_15sec.index <= end_time)]
        
        if len(trade_bars) < 10:
            continue
        
        # Initialize stops
        if direction == 'LONG':
            sl_price = entry_price * (1 - sl_pct / 100)
            tp_price = entry_price * (1 + target_config['threshold'] / 100)
        else:
            sl_price = entry_price * (1 + sl_pct / 100)
            tp_price = entry_price * (1 - target_config['threshold'] / 100)
        
        exit_price = None
        exit_reason = None
        peak_price = entry_price
        trough_price = entry_price
        trailing_active = False
        bars_held = 0
        
        # Simulate with 15-sec precision
        for bar_time, bar in trade_bars.iterrows():
            bars_held += 1
            
            if direction == 'LONG':
                peak_price = max(peak_price, bar['high'])
                current_profit = (peak_price / entry_price - 1) * 100
                
                # Activate trailing
                if trailing_pct and current_profit >= trailing_activation:
                    trailing_active = True
                    new_sl = peak_price * (1 - trailing_pct / 100)
                    sl_price = max(sl_price, new_sl)
                
                # Check exits
                if bar['low'] <= sl_price:
                    exit_price = sl_price
                    exit_reason = 'TS' if trailing_active else 'SL'
                    break
                if bar['high'] >= tp_price:
                    exit_price = tp_price
                    exit_reason = 'TP'
                    break
            else:
                trough_price = min(trough_price, bar['low'])
                current_profit = (entry_price / trough_price - 1) * 100
                
                if trailing_pct and current_profit >= trailing_activation:
                    trailing_active = True
                    new_sl = trough_price * (1 + trailing_pct / 100)
                    sl_price = min(sl_price, new_sl)
                
                if bar['high'] >= sl_price:
                    exit_price = sl_price
                    exit_reason = 'TS' if trailing_active else 'SL'
                    break
                if bar['low'] <= tp_price:
                    exit_price = tp_price
                    exit_reason = 'TP'
                    break
        
        # Timeout
        if exit_price is None:
            exit_price = trade_bars.iloc[-1]['close']
            exit_reason = 'TO'
        
        # Calculate P&L
        if direction == 'LONG':
            pnl_pct = (exit_price / entry_price - 1) * 100
        else:
            pnl_pct = (entry_price / exit_price - 1) * 100
        
        # MNQ: $2 per point, price ~$525 (QQQ), so ~$10.50 per 1%
        # Or use $5 per 0.1% approximation
        pnl_dollar = pnl_pct * 5 - 1.24
        
        trades.append({
            'entry_time': sig_time,
            'prob': prob,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'pnl_pct': pnl_pct,
            'pnl_dollar': pnl_dollar,
            'peak_profit': current_profit,
            'bars_held': bars_held
        })
        
        last_trade_time = sig_time
    
    return pd.DataFrame(trades)


def analyze_backtest(trades_df, name=""):
    """Analyze backtest results."""
    if trades_df.empty:
        return None
    
    total = len(trades_df)
    wins = (trades_df['pnl_dollar'] > 0).sum()
    wr = wins / total * 100
    total_pnl = trades_df['pnl_dollar'].sum()
    
    days = max(1, (trades_df['entry_time'].max() - trades_df['entry_time'].min()).days)
    pnl_week = total_pnl / days * 7
    trades_day = total / days
    
    exits = trades_df['exit_reason'].value_counts().to_dict()
    
    avg_win = trades_df[trades_df['pnl_dollar'] > 0]['pnl_dollar'].mean() if wins > 0 else 0
    avg_loss = trades_df[trades_df['pnl_dollar'] <= 0]['pnl_dollar'].mean() if total - wins > 0 else 0
    
    return {
        'name': name,
        'trades': total,
        'wins': wins,
        'wr': wr,
        'total_pnl': total_pnl,
        'pnl_week': pnl_week,
        'trades_day': trades_day,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'exits': exits
    }


def main():
    print("="*70)
    print("MNQ ML DIRECTION PREDICTION MODEL")
    print("="*70)
    
    # Load data
    df = load_data()
    
    # Add synthetic features
    print("\nAdding synthetic features...")
    df = add_synthetic_features(df)
    print(f"Total features: {len(df.columns)}")
    
    # Load 15-sec data for validation
    print("\nLoading 15-sec data for validation...")
    df_15sec = pd.read_parquet('data/QQQ_15sec_30days.parquet')
    print(f"15-sec bars: {len(df_15sec)}")
    
    # Get test period (last 30 days - matches 15-sec data)
    # Handle timezone mismatch
    test_start = df_15sec.index[0]
    test_end = df_15sec.index[-1]
    
    # Make 5-min index tz-aware if needed
    if df.index.tz is None and test_start.tz is not None:
        df.index = df.index.tz_localize(test_start.tz)
    elif df.index.tz is not None and test_start.tz is None:
        test_start = test_start.tz_localize(df.index.tz)
        test_end = test_end.tz_localize(df.index.tz)
    
    # Filter 5-min data to test period
    df_test = df[(df.index >= test_start) & (df.index <= test_end)].copy()
    print(f"Test period 5-min bars: {len(df_test)}")
    
    all_results = []
    
    # Train and test each target
    for target in TARGETS:
        # Train on all data before test period
        df_train = df[df.index < test_start].copy()
        
        if len(df_train) < 1000:
            print(f"Skipping {target['name']} - insufficient training data")
            continue
        
        # Train model
        model, scaler, feature_cols, train_results, _ = train_model(
            df_train, target, model_type='gb'
        )
        
        # Test different configurations
        configs = [
            {'prob': 0.55, 'sl': 0.4, 'trail': 0.15, 'act': 0.1},
            {'prob': 0.55, 'sl': 0.5, 'trail': 0.2, 'act': 0.15},
            {'prob': 0.60, 'sl': 0.4, 'trail': 0.15, 'act': 0.1},
            {'prob': 0.60, 'sl': 0.5, 'trail': 0.2, 'act': 0.15},
            {'prob': 0.60, 'sl': 0.6, 'trail': 0.25, 'act': 0.2},
            {'prob': 0.65, 'sl': 0.5, 'trail': 0.2, 'act': 0.15},
            {'prob': 0.65, 'sl': 0.6, 'trail': 0.25, 'act': 0.2},
            {'prob': 0.70, 'sl': 0.5, 'trail': 0.2, 'act': 0.15},
        ]
        
        for cfg in configs:
            trades = backtest_with_15sec(
                model, scaler, feature_cols, df_test, df_15sec,
                target, prob_threshold=cfg['prob'],
                sl_pct=cfg['sl'], trailing_pct=cfg['trail'], 
                trailing_activation=cfg['act']
            )
            
            if len(trades) >= 3:
                result = analyze_backtest(trades, target['name'])
                if result:
                    result['target'] = target['name']
                    result['direction'] = target['direction']
                    result['prob_thresh'] = cfg['prob']
                    result['sl'] = cfg['sl']
                    result['trail'] = cfg['trail']
                    all_results.append(result)
    
    # Sort by P&L
    all_results.sort(key=lambda x: x['pnl_week'], reverse=True)
    
    print("\n" + "="*70)
    print("BACKTEST RESULTS (15-sec validation)")
    print("="*70)
    
    print(f"\n{'Target':<25} {'Dir':<6} {'Prob':<6} {'SL':<5} {'Trail':<6} {'Trades':<7} {'WR%':<7} {'$/Wk'}")
    print("-"*85)
    
    for r in all_results[:20]:
        print(f"{r['target']:<25} {r['direction']:<6} {r['prob_thresh']:<6.0%} {r['sl']:<5.1f} {r['trail']:<6.2f} {r['trades']:<7} {r['wr']:<6.0f}% ${r['pnl_week']:<7.0f}")
    
    # Find profitable
    profitable = [r for r in all_results if r['pnl_week'] > 5]
    
    print(f"\n\nPROFITABLE CONFIGURATIONS (>${'5'}/week): {len(profitable)}")
    print("="*70)
    
    if profitable:
        for r in profitable:
            exits = ', '.join([f"{k}:{v}" for k, v in r['exits'].items()])
            print(f"\n  ✅ {r['target']} {r['direction']}")
            print(f"     Prob threshold: {r['prob_thresh']:.0%}")
            print(f"     SL: {r['sl']}%, Trailing: {r['trail']}%")
            print(f"     WR: {r['wr']:.0f}%, P&L: ${r['pnl_week']:.0f}/week")
            print(f"     Trades: {r['trades']}, Exits: {exits}")
            print(f"     Avg win: ${r['avg_win']:.1f}, Avg loss: ${r['avg_loss']:.1f}")
        
        # Save best model
        best = profitable[0]
        print(f"\n\nSaving best model configuration...")
        
        # Retrain best model on full data
        best_target = [t for t in TARGETS if t['name'] == best['target']][0]
        model, scaler, feature_cols, _, _ = train_model(df, best_target, model_type='gb')
        
        # Save
        joblib.dump(model, f"models_mnq_v3/ml_direction_{best['target']}.joblib")
        joblib.dump(scaler, f"models_mnq_v3/scaler_{best['target']}.joblib")
        
        import json
        with open(f"models_mnq_v3/features_{best['target']}.json", 'w') as f:
            json.dump(feature_cols, f)
        
        print(f"  Saved to models_mnq_v3/")
    else:
        print("\n  No profitable ML configuration found.")
        print("  Even with sophisticated ML + trailing stops, no edge in current market.")
    
    return all_results


if __name__ == "__main__":
    results = main()
