"""
Diagnose MNQ model signal issues.
Compare live signals vs what backtest would predict on same data.
"""

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime, timedelta
import ta

# Load models
MODELS_DIR = "models"
ENSEMBLE_MODELS = [
    {'horizon': '3h', 'threshold': 0.5, 'horizon_bars': 36, 'direction': 'LONG'},
    {'horizon': '2h', 'threshold': 0.5, 'horizon_bars': 24, 'direction': 'LONG'},
    {'horizon': '2h', 'threshold': 0.75, 'horizon_bars': 24, 'direction': 'LONG'},
    {'horizon': '1h30m', 'threshold': 0.5, 'horizon_bars': 18, 'direction': 'LONG'},
    {'horizon': '2h', 'threshold': 0.75, 'horizon_bars': 24, 'direction': 'SHORT'},
    {'horizon': '3h', 'threshold': 0.5, 'horizon_bars': 36, 'direction': 'SHORT'},
]

def load_models():
    """Load all MNQ models."""
    models = {}
    for m in ENSEMBLE_MODELS:
        direction = m.get('direction', 'LONG')
        suffix = '_SHORT' if direction == 'SHORT' else ''
        model_name = f"{m['horizon']}_{m['threshold']}pct{suffix}"
        model_path = f"{MODELS_DIR}/model_{model_name}.joblib"
        try:
            models[model_name] = {
                'model': joblib.load(model_path),
                'config': m
            }
            print(f"Loaded: {model_name}")
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
    return models

def load_feature_columns():
    """Load feature columns."""
    with open(f"{MODELS_DIR}/feature_columns.json", 'r') as f:
        return json.load(f)

def load_training_data():
    """Load the QQQ training data to check feature distributions."""
    df = pd.read_parquet("data/QQQ_5min_IB_with_indicators.csv.parquet") if \
        pd.io.common.file_exists("data/QQQ_5min_IB_with_indicators.csv.parquet") else \
        pd.read_parquet("data/QQQ_5min_IB_with_indicators.parquet")
    return df

def analyze_signal_logs():
    """Analyze signal logs to understand probability distributions."""
    import glob
    
    files = sorted(glob.glob('signal_logs/mnq_signals_2025-12-1*.csv'))
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except:
            pass
    
    if not dfs:
        print("No signal logs found")
        return None
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"\nSignal log analysis ({len(df)} rows):")
    
    prob_cols = [c for c in df.columns if c.startswith('prob_')]
    
    print("\nProbability Statistics:")
    print("-" * 80)
    for col in prob_cols:
        stats = df[col].describe()
        count_55 = (df[col] >= 0.55).sum()
        count_50 = (df[col] >= 0.50).sum()
        count_45 = (df[col] >= 0.45).sum()
        print(f"{col}:")
        print(f"  Mean: {stats['mean']:.2%}, Std: {stats['std']:.2%}, Max: {stats['max']:.2%}")
        print(f"  >=55%: {count_55}, >=50%: {count_50}, >=45%: {count_45}")
    
    return df

def check_model_on_recent_data():
    """Load recent QQQ data and run models to see expected signals."""
    # Try to load the most recent data file
    try:
        df = pd.read_parquet("data/QQQ_5min_IB_with_indicators.parquet")
        print(f"\nLoaded QQQ data: {len(df)} rows")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
    except Exception as e:
        print(f"Could not load QQQ data: {e}")
        return
    
    # Get last 7 days
    cutoff = df.index[-1] - timedelta(days=7)
    recent = df[df.index >= cutoff]
    print(f"Recent data (last 7 days): {len(recent)} rows")
    
    # Load models and feature columns
    models = load_models()
    feature_cols = load_feature_columns()
    
    # Check which features are available
    available_features = [f for f in feature_cols if f in recent.columns]
    missing_features = [f for f in feature_cols if f not in recent.columns]
    
    print(f"\nFeatures: {len(available_features)} available, {len(missing_features)} missing")
    if missing_features:
        print(f"Missing features (first 10): {missing_features[:10]}")
    
    if len(missing_features) > 0:
        print("\nCannot run backtest - missing features in data")
        return
    
    # Run predictions
    X = recent[feature_cols]
    
    print("\nModel predictions on recent data:")
    print("-" * 80)
    
    for model_name, model_data in models.items():
        model = model_data['model']
        try:
            probs = model.predict_proba(X)[:, 1]
            
            count_55 = (probs >= 0.55).sum()
            count_50 = (probs >= 0.50).sum()
            count_45 = (probs >= 0.45).sum()
            
            print(f"{model_name}:")
            print(f"  Mean: {probs.mean():.2%}, Std: {probs.std():.2%}, Max: {probs.max():.2%}")
            print(f"  >=55%: {count_55}, >=50%: {count_50}, >=45%: {count_45}")
            
            # Show when max probability occurred
            max_idx = probs.argmax()
            max_time = recent.index[max_idx]
            print(f"  Max prob at: {max_time}")
            
        except Exception as e:
            print(f"{model_name}: Error - {e}")

def compare_training_vs_live_features():
    """Compare feature distributions between training data and live data."""
    print("\n" + "=" * 80)
    print("COMPARING TRAINING VS LIVE FEATURE DISTRIBUTIONS")
    print("=" * 80)
    
    # Load training data
    try:
        train_df = pd.read_parquet("data/QQQ_5min_IB_with_indicators.parquet")
        print(f"Training data: {len(train_df)} rows, {train_df.index[0]} to {train_df.index[-1]}")
    except Exception as e:
        print(f"Could not load training data: {e}")
        return
    
    # Load signal logs to get live feature values
    import glob
    files = sorted(glob.glob('signal_logs/mnq_signals_2025-12-1*.csv'))
    if not files:
        print("No signal logs found")
        return
    
    # Signal logs have limited features (rsi, macd, atr)
    # Let's compare those
    live_dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            live_dfs.append(df)
        except:
            pass
    
    live_df = pd.concat(live_dfs, ignore_index=True)
    
    print(f"\nLive data: {len(live_df)} rows")
    
    # Compare RSI
    if 'rsi' in live_df.columns and 'momentum_rsi' in train_df.columns:
        print("\nRSI Comparison:")
        print(f"  Training: mean={train_df['momentum_rsi'].mean():.2f}, std={train_df['momentum_rsi'].std():.2f}")
        print(f"  Live:     mean={live_df['rsi'].mean():.2f}, std={live_df['rsi'].std():.2f}")
    
    # Compare MACD
    if 'macd' in live_df.columns and 'trend_macd' in train_df.columns:
        print("\nMACD Comparison:")
        print(f"  Training: mean={train_df['trend_macd'].mean():.4f}, std={train_df['trend_macd'].std():.4f}")
        print(f"  Live:     mean={live_df['macd'].mean():.4f}, std={live_df['macd'].std():.4f}")
    
    # Compare ATR
    if 'atr' in live_df.columns and 'volatility_atr' in train_df.columns:
        print("\nATR Comparison:")
        print(f"  Training: mean={train_df['volatility_atr'].mean():.4f}, std={train_df['volatility_atr'].std():.4f}")
        print(f"  Live:     mean={live_df['atr'].mean():.4f}, std={live_df['atr'].std():.4f}")

def main():
    print("=" * 80)
    print("MNQ MODEL DIAGNOSTIC")
    print("=" * 80)
    
    # 1. Analyze signal logs
    analyze_signal_logs()
    
    # 2. Check model predictions on recent stored data
    check_model_on_recent_data()
    
    # 3. Compare feature distributions
    compare_training_vs_live_features()
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
