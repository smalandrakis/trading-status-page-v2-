#!/usr/bin/env python3
"""
Retrain ML models for MNQ and SPY using 30-day 15-sec granular signal data.
Uses all available features from the parquet + signal logs.
Targets: Predict if price will move up/down by X% within Y bars.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

QQQ_PARQUET = "data/QQQ_features.parquet"
OUTPUT_DIR = "models_mnq_spy_v3"

def create_labels(df, target_pct=0.5, horizon_bars=24):
    """
    Create labels for ML training.
    LONG label: Price goes up by target_pct% within horizon_bars
    SHORT label: Price goes down by target_pct% within horizon_bars
    """
    df = df.copy()
    
    # Calculate future max/min prices within horizon
    future_max = df['High'].rolling(window=horizon_bars, min_periods=1).max().shift(-horizon_bars)
    future_min = df['Low'].rolling(window=horizon_bars, min_periods=1).min().shift(-horizon_bars)
    
    # Calculate max gain and max loss from current close
    max_gain_pct = (future_max - df['Close']) / df['Close'] * 100
    max_loss_pct = (df['Close'] - future_min) / df['Close'] * 100
    
    # LONG: Max gain >= target AND max loss < target (favorable risk)
    df['label_long'] = ((max_gain_pct >= target_pct) & (max_loss_pct < target_pct)).astype(int)
    
    # SHORT: Max loss >= target AND max gain < target (favorable risk)
    df['label_short'] = ((max_loss_pct >= target_pct) & (max_gain_pct < target_pct)).astype(int)
    
    # Combined label: 1=LONG opportunity, -1=SHORT opportunity, 0=neutral
    df['label_combined'] = df['label_long'] - df['label_short']
    
    return df


def add_extra_features(df):
    """Add extra features for ML training."""
    df = df.copy()
    
    # Price momentum features
    for period in [3, 5, 10, 20]:
        df[f'roc_{period}'] = (df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period) * 100
        df[f'price_vs_high_{period}'] = (df['Close'] - df['High'].rolling(period).max()) / df['Close'] * 100
        df[f'price_vs_low_{period}'] = (df['Close'] - df['Low'].rolling(period).min()) / df['Close'] * 100
    
    # Volatility features
    df['atr_pct'] = df['volatility_atr'] / df['Close'] * 100
    df['bb_width_pct'] = df['volatility_bbw']
    df['range_pct'] = (df['High'] - df['Low']) / df['Close'] * 100
    
    # Trend features
    df['ema_12'] = df['Close'].ewm(span=12).mean()
    df['ema_26'] = df['Close'].ewm(span=26).mean()
    df['ema_50'] = df['Close'].ewm(span=50).mean()
    df['price_vs_ema12'] = (df['Close'] - df['ema_12']) / df['Close'] * 100
    df['price_vs_ema26'] = (df['Close'] - df['ema_26']) / df['Close'] * 100
    df['price_vs_ema50'] = (df['Close'] - df['ema_50']) / df['Close'] * 100
    df['ema_12_26_diff'] = (df['ema_12'] - df['ema_26']) / df['Close'] * 100
    
    # MACD features
    df['macd_hist'] = df['trend_macd'] - df['trend_macd_signal']
    df['macd_hist_change'] = df['macd_hist'] - df['macd_hist'].shift(1)
    
    # RSI features
    df['rsi_change'] = df['momentum_rsi'] - df['momentum_rsi'].shift(1)
    df['rsi_oversold'] = (df['momentum_rsi'] < 30).astype(int)
    df['rsi_overbought'] = (df['momentum_rsi'] > 70).astype(int)
    
    # Stochastic features
    df['stoch_change'] = df['momentum_stoch'] - df['momentum_stoch'].shift(1)
    df['stoch_oversold'] = (df['momentum_stoch'] < 20).astype(int)
    df['stoch_overbought'] = (df['momentum_stoch'] > 80).astype(int)
    
    # BB features
    df['bb_position'] = df['volatility_bbp']
    df['bb_squeeze'] = (df['volatility_bbw'] < df['volatility_bbw'].rolling(20).mean() * 0.8).astype(int)
    
    # Volume features (if available)
    if 'Volume' in df.columns:
        df['volume_sma'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
    
    # Time features
    if df.index.dtype == 'datetime64[ns]' or hasattr(df.index, 'hour'):
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['is_morning'] = ((df['hour'] >= 9) & (df['hour'] < 12)).astype(int)
        df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 16)).astype(int)
    
    return df


def train_model(X_train, y_train, X_test, y_test, model_type='rf'):
    """Train and evaluate a model."""
    if model_type == 'rf':
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    else:
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate metrics at different probability thresholds
    results = {'accuracy': accuracy}
    for thresh in [0.5, 0.55, 0.60, 0.65]:
        y_pred_thresh = (y_prob >= thresh).astype(int)
        if y_pred_thresh.sum() > 0:
            precision = (y_test[y_pred_thresh == 1] == 1).mean()
            signals = y_pred_thresh.sum()
            results[f'precision_{int(thresh*100)}'] = precision
            results[f'signals_{int(thresh*100)}'] = signals
    
    return model, results


def backtest_model(df, model, feature_cols, label_col, symbol, sl_pct=0.5, tp_pct=1.0, 
                   prob_threshold=0.55, cooldown_bars=10):
    """Backtest a trained model."""
    df = df.copy()
    
    # Get predictions
    X = df[feature_cols].fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    
    probs = model.predict_proba(X)[:, 1]
    df['prob'] = probs
    df['signal'] = (probs >= prob_threshold).astype(int)
    
    # Simulate trades
    trades = []
    last_trade_bar = -cooldown_bars
    
    for i in range(50, len(df) - 50):
        if i - last_trade_bar < cooldown_bars:
            continue
        
        if df['signal'].iloc[i] != 1:
            continue
        
        entry_price = df['Close'].iloc[i]
        entry_time = df.index[i]
        
        # Determine direction based on label type
        direction = 'LONG' if 'long' in label_col.lower() else 'SHORT'
        
        if direction == 'LONG':
            target_price = entry_price * (1 + tp_pct / 100)
            stop_price = entry_price * (1 - sl_pct / 100)
        else:
            target_price = entry_price * (1 - tp_pct / 100)
            stop_price = entry_price * (1 + sl_pct / 100)
        
        exit_price = None
        exit_reason = None
        max_hold = 48
        
        for j in range(i + 1, min(i + max_hold + 1, len(df))):
            future_row = df.iloc[j]
            
            if direction == 'LONG':
                if future_row['Low'] <= stop_price:
                    exit_price = stop_price
                    exit_reason = 'SL'
                    break
                if future_row['High'] >= target_price:
                    exit_price = target_price
                    exit_reason = 'TP'
                    break
            else:
                if future_row['High'] >= stop_price:
                    exit_price = stop_price
                    exit_reason = 'SL'
                    break
                if future_row['Low'] <= target_price:
                    exit_price = target_price
                    exit_reason = 'TP'
                    break
        
        if exit_price is None:
            exit_price = df['Close'].iloc[min(i + max_hold, len(df) - 1)]
            exit_reason = 'TO'
        
        if direction == 'LONG':
            pnl_pct = (exit_price / entry_price - 1) * 100
        else:
            pnl_pct = (entry_price / exit_price - 1) * 100
        
        commission = 2.50
        multiplier = 2.0 if symbol == 'MNQ' else 5.0
        pnl_dollar = (pnl_pct / 100) * entry_price * multiplier - commission
        
        trades.append({
            'entry_time': entry_time,
            'direction': direction,
            'prob': df['prob'].iloc[i],
            'pnl_dollar': pnl_dollar,
            'exit_reason': exit_reason
        })
        
        last_trade_bar = i
    
    if not trades:
        return None
    
    trades_df = pd.DataFrame(trades)
    total_trades = len(trades_df)
    wins = len(trades_df[trades_df['pnl_dollar'] > 0])
    win_rate = wins / total_trades * 100
    total_pnl = trades_df['pnl_dollar'].sum()
    
    days = (trades_df['entry_time'].max() - trades_df['entry_time'].min()).days or 1
    pnl_per_week = total_pnl / days * 7
    trades_per_day = total_trades / days
    
    return {
        'trades': total_trades,
        'win_rate': win_rate,
        'pnl_week': pnl_per_week,
        'trades_day': trades_per_day,
        'total_pnl': total_pnl
    }


def main():
    print("="*80)
    print("MNQ/SPY ML MODEL RETRAINING")
    print("="*80)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    print("\nLoading QQQ parquet data...")
    df = pd.read_parquet(QQQ_PARQUET)
    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    
    # Add extra features
    print("Adding extra features...")
    df = add_extra_features(df)
    
    # Define feature columns (exclude OHLCV and labels)
    exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'label_long', 'label_short', 
                    'label_combined', 'ema_12', 'ema_26', 'ema_50', 'volume_sma']
    feature_cols = [col for col in df.columns if col not in exclude_cols and not col.startswith('label')]
    
    print(f"Using {len(feature_cols)} features")
    
    # Test different target configurations
    configs = [
        {'target_pct': 0.3, 'horizon': 12, 'sl': 0.3, 'tp': 0.5},   # Quick scalp
        {'target_pct': 0.5, 'horizon': 24, 'sl': 0.5, 'tp': 1.0},   # Medium
        {'target_pct': 0.7, 'horizon': 36, 'sl': 0.7, 'tp': 1.4},   # Longer
        {'target_pct': 1.0, 'horizon': 48, 'sl': 1.0, 'tp': 2.0},   # Swing
    ]
    
    all_results = []
    
    for config in configs:
        target_pct = config['target_pct']
        horizon = config['horizon']
        sl = config['sl']
        tp = config['tp']
        
        print(f"\n{'='*60}")
        print(f"Training: Target={target_pct}%, Horizon={horizon} bars, SL={sl}%, TP={tp}%")
        print(f"{'='*60}")
        
        # Create labels
        df_labeled = create_labels(df, target_pct=target_pct, horizon_bars=horizon)
        
        # Remove rows with NaN labels (end of dataset)
        df_clean = df_labeled.dropna(subset=['label_long', 'label_short'])
        
        # Use last 60 days for training, last 30 days for testing
        cutoff_train = df_clean.index[-1] - timedelta(days=60)
        cutoff_test = df_clean.index[-1] - timedelta(days=30)
        
        df_train = df_clean[(df_clean.index >= cutoff_train) & (df_clean.index < cutoff_test)]
        df_test = df_clean[df_clean.index >= cutoff_test]
        
        print(f"Training samples: {len(df_train)}, Test samples: {len(df_test)}")
        
        # Prepare features
        X_train = df_train[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        X_test = df_test[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        
        for label_type in ['long', 'short']:
            label_col = f'label_{label_type}'
            y_train = df_train[label_col]
            y_test = df_test[label_col]
            
            # Check class balance
            pos_rate_train = y_train.mean()
            pos_rate_test = y_test.mean()
            print(f"\n{label_type.upper()}: Train pos rate={pos_rate_train:.1%}, Test pos rate={pos_rate_test:.1%}")
            
            if pos_rate_train < 0.05 or pos_rate_train > 0.95:
                print(f"  Skipping - imbalanced data")
                continue
            
            # Train model
            model, metrics = train_model(X_train, y_train, X_test, y_test, model_type='rf')
            
            print(f"  Accuracy: {metrics['accuracy']:.1%}")
            for thresh in [50, 55, 60, 65]:
                if f'precision_{thresh}' in metrics:
                    print(f"  @{thresh}%: Precision={metrics[f'precision_{thresh}']:.1%}, Signals={metrics[f'signals_{thresh}']}")
            
            # Backtest on test set
            for symbol in ['MNQ', 'SPY']:
                for prob_thresh in [0.55, 0.60, 0.65]:
                    bt_result = backtest_model(df_test, model, feature_cols, label_col, symbol,
                                               sl_pct=sl, tp_pct=tp, prob_threshold=prob_thresh)
                    
                    if bt_result and bt_result['trades'] >= 10:
                        result = {
                            'symbol': symbol,
                            'direction': label_type.upper(),
                            'target': target_pct,
                            'horizon': horizon,
                            'sl_tp': f"{sl}/{tp}",
                            'prob_thresh': prob_thresh,
                            **bt_result
                        }
                        all_results.append(result)
                        
                        if bt_result['pnl_week'] > 0:
                            print(f"  ✅ {symbol} {label_type.upper()} @{prob_thresh}: WR={bt_result['win_rate']:.1f}%, $/wk=${bt_result['pnl_week']:.0f}")
            
            # Save best model
            model_name = f"model_{label_type}_{target_pct}pct_{horizon}bars"
            model_path = os.path.join(OUTPUT_DIR, f"{model_name}.joblib")
            joblib.dump(model, model_path)
    
    # Save feature columns
    feature_path = os.path.join(OUTPUT_DIR, "feature_columns.json")
    import json
    with open(feature_path, 'w') as f:
        json.dump(feature_cols, f)
    
    # Summary
    print("\n" + "="*100)
    print("SUMMARY - ALL BACKTEST RESULTS (sorted by $/week)")
    print("="*100)
    
    if all_results:
        results_sorted = sorted(all_results, key=lambda x: x['pnl_week'], reverse=True)
        
        print(f"\n{'Symbol':<6} {'Dir':<6} {'Target':<8} {'SL/TP':<8} {'Prob':<6} {'Trades':<8} {'WR%':<8} {'$/Week':<10}")
        print("-" * 80)
        
        for r in results_sorted[:20]:
            print(f"{r['symbol']:<6} {r['direction']:<6} {r['target']}%     {r['sl_tp']:<8} {r['prob_thresh']:<6} {r['trades']:<8} {r['win_rate']:.1f}%    ${r['pnl_week']:>6.0f}")
        
        # Show profitable only
        profitable = [r for r in results_sorted if r['pnl_week'] > 20]
        print(f"\n\nPROFITABLE MODELS (>${'20'}/week): {len(profitable)}")
        if profitable:
            for r in profitable:
                print(f"  {r['symbol']} {r['direction']} {r['target']}% @{r['prob_thresh']}: WR={r['win_rate']:.1f}%, ${r['pnl_week']:.0f}/wk")
    else:
        print("No valid backtest results generated.")
    
    print(f"\nModels saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
