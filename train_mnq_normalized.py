"""
Train MNQ models with normalized cumulative features.

The original models were trained on QQQ at ~$266, now QQQ is ~$619.
Cumulative features have drifted 3-10x from training values.

This script:
1. Normalizes cumulative features (rate of change instead of absolute)
2. Normalizes price-based features (divide by Close)
3. Retrains all MNQ models
"""

import pandas as pd
import numpy as np
import joblib
import ta
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

from feature_engineering import (
    add_time_features, add_price_features, add_daily_context_features,
    add_lagged_indicator_features, add_indicator_changes
)
import config

# Model configurations (same as original)
MODEL_CONFIGS = [
    {'horizon': '3h', 'threshold': 0.5, 'bars': 36, 'direction': 'LONG'},
    {'horizon': '2h', 'threshold': 0.5, 'bars': 24, 'direction': 'LONG'},
    {'horizon': '2h', 'threshold': 0.75, 'bars': 24, 'direction': 'LONG'},
    {'horizon': '1h30m', 'threshold': 0.5, 'bars': 18, 'direction': 'LONG'},
    {'horizon': '2h', 'threshold': 0.75, 'bars': 24, 'direction': 'SHORT'},
    {'horizon': '3h', 'threshold': 0.5, 'bars': 36, 'direction': 'SHORT'},
]

# Features to normalize
CUMULATIVE_FEATURES = ['volume_obv', 'volume_adi', 'volume_nvi', 'volume_vpt', 'others_cr']
PRICE_FEATURES = ['volatility_atr', 'trend_macd', 'trend_macd_signal', 'trend_macd_diff',
                  'volatility_bbh', 'volatility_bbl', 'volatility_bbm', 'volatility_bbw',
                  'volatility_kch', 'volatility_kcl', 'volatility_kcw',
                  'volatility_dch', 'volatility_dcl', 'volatility_dcm', 'volatility_dcw']


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize cumulative and price-based features for scale invariance."""
    df = df.copy()
    
    # 1. Cumulative features: use rate of change (pct_change)
    for feat in CUMULATIVE_FEATURES:
        if feat in df.columns:
            # Replace absolute value with percentage change
            df[f'{feat}_pct'] = df[feat].pct_change().fillna(0)
            # Clip extreme values
            df[f'{feat}_pct'] = df[f'{feat}_pct'].clip(-0.1, 0.1)
            # Keep original for lagged versions but normalize them too
            for lag in [1, 2, 3, 5, 10, 20, 50]:
                lag_feat = f'{feat}_lag{lag}'
                if lag_feat in df.columns:
                    df[f'{lag_feat}_pct'] = df[lag_feat].pct_change().fillna(0).clip(-0.1, 0.1)
    
    # 2. Price-based features: normalize by Close
    for feat in PRICE_FEATURES:
        if feat in df.columns and 'Close' in df.columns:
            df[f'{feat}_norm'] = df[feat] / df['Close']
            for lag in [1, 2, 3, 5, 10, 20, 50]:
                lag_feat = f'{feat}_lag{lag}'
                if lag_feat in df.columns:
                    df[f'{lag_feat}_norm'] = df[lag_feat] / df['Close']
    
    return df


def create_target(df: pd.DataFrame, horizon_bars: int, threshold_pct: float, direction: str) -> pd.Series:
    """Create binary target: 1 if price moves threshold% in direction within horizon."""
    future_high = df['High'].rolling(horizon_bars).max().shift(-horizon_bars)
    future_low = df['Low'].rolling(horizon_bars).min().shift(-horizon_bars)
    current_close = df['Close']
    
    if direction == 'LONG':
        # Target: price goes UP by threshold%
        target = ((future_high - current_close) / current_close * 100) >= threshold_pct
    else:
        # Target: price goes DOWN by threshold%
        target = ((current_close - future_low) / current_close * 100) >= threshold_pct
    
    return target.astype(int)


def get_feature_columns(df: pd.DataFrame) -> list:
    """Get list of feature columns, preferring normalized versions."""
    feature_cols = []
    
    for col in df.columns:
        # Skip target, OHLCV, and original cumulative/price features
        if col in ['Open', 'High', 'Low', 'Close', 'Volume', 'target', 'average', 'barCount']:
            continue
        
        # Skip original cumulative features (use _pct versions)
        skip = False
        for cum_feat in CUMULATIVE_FEATURES:
            if col == cum_feat or (col.startswith(cum_feat) and '_lag' in col and '_pct' not in col):
                skip = True
                break
        
        # Skip original price features (use _norm versions)
        for price_feat in PRICE_FEATURES:
            if col == price_feat or (col.startswith(price_feat) and '_lag' in col and '_norm' not in col):
                skip = True
                break
        
        if not skip:
            feature_cols.append(col)
    
    return feature_cols


def train_model(df: pd.DataFrame, config: dict) -> tuple:
    """Train a single model with normalized features."""
    horizon = config['horizon']
    threshold = config['threshold']
    bars = config['bars']
    direction = config['direction']
    
    print(f"\n{'='*60}")
    print(f"Training: {horizon}_{threshold}pct {'SHORT' if direction == 'SHORT' else ''}")
    print(f"{'='*60}")
    
    # Create target
    df['target'] = create_target(df, bars, threshold, direction)
    
    # Remove rows with NaN target (end of dataset)
    df_clean = df.dropna(subset=['target'])
    
    # Get feature columns
    feature_cols = get_feature_columns(df_clean)
    print(f"Features: {len(feature_cols)}")
    
    # Prepare X, y
    X = df_clean[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y = df_clean['target']
    
    # Train/test split (time-based)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Target distribution - Train: {y_train.mean():.1%} positive, Test: {y_test.mean():.1%} positive")
    
    # Train model
    model = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    train_probs = model.predict_proba(X_train)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]
    
    print(f"Train probs - Max: {train_probs.max():.1%}, Mean: {train_probs.mean():.1%}")
    print(f"Test probs - Max: {test_probs.max():.1%}, Mean: {test_probs.mean():.1%}")
    print(f"Test signals >= 60%: {(test_probs >= 0.6).sum()} ({(test_probs >= 0.6).mean()*100:.1f}%)")
    
    return model, feature_cols


def main():
    print("="*60)
    print("TRAINING MNQ MODELS WITH NORMALIZED FEATURES")
    print("="*60)
    
    # Load data
    print("\nLoading QQQ data...")
    df = pd.read_parquet('data/QQQ_features.parquet')
    print(f"Loaded {len(df)} bars")
    
    # Normalize features
    print("\nNormalizing features...")
    df = normalize_features(df)
    print(f"Total columns after normalization: {len(df.columns)}")
    
    # Create output directory
    os.makedirs('models_normalized', exist_ok=True)
    
    # Train each model
    for cfg in MODEL_CONFIGS:
        model, feature_cols = train_model(df.copy(), cfg)
        
        # Save model
        direction_suffix = '_SHORT' if cfg['direction'] == 'SHORT' else ''
        model_name = f"model_{cfg['horizon']}_{cfg['threshold']}pct{direction_suffix}_normalized.joblib"
        model_path = f"models_normalized/{model_name}"
        
        joblib.dump(model, model_path)
        print(f"Saved: {model_path}")
        
        # Save feature columns
        feature_path = f"models_normalized/features_{cfg['horizon']}_{cfg['threshold']}pct{direction_suffix}_normalized.txt"
        with open(feature_path, 'w') as f:
            f.write('\n'.join(feature_cols))
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print("\nNormalized models saved to models_normalized/")
    print("To use these models, update ensemble_bot.py to:")
    print("  1. Load models from models_normalized/")
    print("  2. Apply normalize_features() before prediction")


if __name__ == '__main__':
    main()
