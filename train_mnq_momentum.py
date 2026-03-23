"""
Train MNQ Momentum/Trend Continuation Models

These models are designed to capture TREND CONTINUATION signals,
as opposed to the existing reversal/bounce models.

Key differences from reversal models:
- Entry conditions: RSI > 50, positive MACD, price above MAs
- Target: Price continues rising X% within horizon
- Filters: Only train on samples where trend is already established
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os
from datetime import datetime

# Configuration
PARQUET_PATH = 'data/QQQ_features.parquet'
OUTPUT_DIR = 'models_momentum'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Momentum model configurations
# Shorter horizons and smaller targets for trend continuation
MOMENTUM_MODELS = [
    {'horizon': '1h', 'horizon_bars': 12, 'threshold': 0.25, 'direction': 'LONG'},
    {'horizon': '1h', 'horizon_bars': 12, 'threshold': 0.5, 'direction': 'LONG'},
    {'horizon': '2h', 'horizon_bars': 24, 'threshold': 0.5, 'direction': 'LONG'},
    {'horizon': '2h', 'horizon_bars': 24, 'threshold': 0.75, 'direction': 'LONG'},
    # SHORT momentum models
    {'horizon': '1h', 'horizon_bars': 12, 'threshold': 0.25, 'direction': 'SHORT'},
    {'horizon': '1h', 'horizon_bars': 12, 'threshold': 0.5, 'direction': 'SHORT'},
    {'horizon': '2h', 'horizon_bars': 24, 'threshold': 0.5, 'direction': 'SHORT'},
    {'horizon': '2h', 'horizon_bars': 24, 'threshold': 0.75, 'direction': 'SHORT'},
]

# Features to normalize (same as existing normalized models)
CUMULATIVE_FEATURES = ['volume_obv', 'volume_adi', 'volume_nvi', 'volume_vpt', 'others_cr']
PRICE_FEATURES = ['volatility_atr', 'trend_macd', 'trend_macd_signal', 'trend_macd_diff',
                  'volatility_bbh', 'volatility_bbl', 'volatility_bbm', 'volatility_bbw',
                  'volatility_kch', 'volatility_kcl', 'volatility_kcw',
                  'volatility_dch', 'volatility_dcl', 'volatility_dcm', 'volatility_dcw']


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize cumulative and price-based features."""
    df = df.copy()
    
    for feat in CUMULATIVE_FEATURES:
        if feat in df.columns:
            df[f'{feat}_pct'] = df[feat].pct_change().fillna(0).clip(-0.1, 0.1)
            for lag in [1, 2, 3, 5, 10, 20, 50]:
                lag_feat = f'{feat}_lag{lag}'
                if lag_feat in df.columns:
                    df[f'{lag_feat}_pct'] = df[lag_feat].pct_change().fillna(0).clip(-0.1, 0.1)
    
    for feat in PRICE_FEATURES:
        if feat in df.columns and 'Close' in df.columns:
            df[f'{feat}_norm'] = df[feat] / df['Close']
            for lag in [1, 2, 3, 5, 10, 20, 50]:
                lag_feat = f'{feat}_lag{lag}'
                if lag_feat in df.columns:
                    df[f'{lag_feat}_norm'] = df[lag_feat] / df['Close']
    
    return df


def create_momentum_labels(df: pd.DataFrame, horizon_bars: int, threshold_pct: float, direction: str) -> pd.Series:
    """
    Create labels for momentum/trend continuation.
    
    For LONG momentum:
    - Entry condition: RSI > 50, MACD > 0, price above MA20
    - Success: Price rises by threshold% within horizon
    
    For SHORT momentum:
    - Entry condition: RSI < 50, MACD < 0, price below MA20
    - Success: Price falls by threshold% within horizon
    """
    # Calculate future returns
    future_return = df['Close'].shift(-horizon_bars) / df['Close'] - 1
    
    # Calculate momentum conditions
    ma20 = df['Close'].rolling(20).mean()
    price_above_ma20 = df['Close'] > ma20
    price_below_ma20 = df['Close'] < ma20
    
    rsi_bullish = df['momentum_rsi'] > 50
    rsi_bearish = df['momentum_rsi'] < 50
    
    macd_bullish = df['trend_macd'] > 0
    macd_bearish = df['trend_macd'] < 0
    
    # Recent momentum (price trending)
    recent_return = df['Close'].pct_change(10)
    trending_up = recent_return > 0
    trending_down = recent_return < 0
    
    if direction == 'LONG':
        # Momentum LONG: already in uptrend, continues rising
        momentum_condition = rsi_bullish & macd_bullish & price_above_ma20 & trending_up
        success = future_return >= (threshold_pct / 100)
    else:
        # Momentum SHORT: already in downtrend, continues falling
        momentum_condition = rsi_bearish & macd_bearish & price_below_ma20 & trending_down
        success = future_return <= -(threshold_pct / 100)
    
    # Label is 1 only if momentum condition is met AND price continues in direction
    labels = (momentum_condition & success).astype(int)
    
    return labels, momentum_condition


def get_feature_columns(df: pd.DataFrame) -> list:
    """Get feature columns for training (excluding OHLCV and labels)."""
    exclude = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 
               'label', 'future_return', 'momentum_condition']
    
    # Also exclude raw cumulative features (use _pct versions)
    for feat in CUMULATIVE_FEATURES:
        exclude.append(feat)
        for lag in [1, 2, 3, 5, 10, 20, 50]:
            exclude.append(f'{feat}_lag{lag}')
    
    # Exclude raw price features (use _norm versions)
    for feat in PRICE_FEATURES:
        exclude.append(feat)
        for lag in [1, 2, 3, 5, 10, 20, 50]:
            exclude.append(f'{feat}_lag{lag}')
    
    features = [c for c in df.columns if c not in exclude and not c.startswith('label')]
    return features


def train_momentum_model(df: pd.DataFrame, config: dict) -> dict:
    """Train a single momentum model."""
    horizon = config['horizon']
    horizon_bars = config['horizon_bars']
    threshold = config['threshold']
    direction = config['direction']
    
    print(f"\n{'='*60}")
    print(f"Training {direction} Momentum Model: {horizon}_{threshold}pct")
    print(f"{'='*60}")
    
    # Create labels
    labels, momentum_condition = create_momentum_labels(df, horizon_bars, threshold, direction)
    
    # Get feature columns
    feature_cols = get_feature_columns(df)
    
    # Prepare data - only use rows where momentum condition could be evaluated
    valid_idx = ~labels.isna() & ~momentum_condition.isna()
    
    # For training, we want a mix of:
    # 1. Momentum condition met + success (positive class)
    # 2. Momentum condition met + failure (negative class - important!)
    # 3. Some non-momentum samples (to learn when NOT to trade)
    
    X = df.loc[valid_idx, feature_cols].copy()
    y = labels[valid_idx].copy()
    
    # Handle infinities and NaNs
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    print(f"Total samples: {len(X)}")
    print(f"Positive samples (momentum + success): {y.sum()}")
    print(f"Positive rate: {y.mean()*100:.2f}%")
    
    if y.sum() < 100:
        print(f"WARNING: Not enough positive samples for {horizon}_{threshold}pct {direction}")
        return None
    
    # Split data - use recent data for testing
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Train positive rate: {y_train.mean()*100:.2f}%")
    print(f"Test positive rate: {y_test.mean()*100:.2f}%")
    
    # Train model
    model = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=6,
        learning_rate=0.05,
        min_samples_leaf=50,
        l2_regularization=1.0,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"\nTest Results:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    try:
        auc = roc_auc_score(y_test, y_proba)
        print(f"AUC-ROC: {auc:.4f}")
    except:
        auc = 0
        print("Could not calculate AUC")
    
    # Check precision at high probability threshold
    high_conf_mask = y_proba >= 0.6
    if high_conf_mask.sum() > 0:
        high_conf_precision = y_test[high_conf_mask].mean()
        print(f"Precision at prob>=60%: {high_conf_precision:.2%} ({high_conf_mask.sum()} samples)")
    
    return {
        'model': model,
        'features': feature_cols,
        'config': config,
        'auc': auc,
        'train_positive_rate': y_train.mean(),
        'test_positive_rate': y_test.mean()
    }


def main():
    print("Loading data...")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"Loaded {len(df)} rows")
    
    # Use data from 2024 onwards for training (more recent patterns)
    df = df[df.index >= '2024-01-01']
    print(f"Using {len(df)} rows from 2024 onwards")
    
    # Normalize features
    print("Normalizing features...")
    df = normalize_features(df)
    
    # Train models
    results = []
    for config in MOMENTUM_MODELS:
        result = train_momentum_model(df, config)
        if result:
            results.append(result)
            
            # Save model
            direction = config['direction']
            suffix = '_SHORT' if direction == 'SHORT' else ''
            model_name = f"model_{config['horizon']}_{config['threshold']}pct{suffix}_momentum.joblib"
            model_path = os.path.join(OUTPUT_DIR, model_name)
            joblib.dump(result['model'], model_path)
            print(f"Saved: {model_path}")
            
            # Save feature list
            feature_name = f"features_{config['horizon']}_{config['threshold']}pct{suffix}_momentum.txt"
            feature_path = os.path.join(OUTPUT_DIR, feature_name)
            with open(feature_path, 'w') as f:
                f.write('\n'.join(result['features']))
            print(f"Saved: {feature_path}")
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    for r in results:
        cfg = r['config']
        direction = cfg['direction']
        print(f"{cfg['horizon']}_{cfg['threshold']}pct {direction}: AUC={r['auc']:.4f}, "
              f"train_pos={r['train_positive_rate']:.2%}, test_pos={r['test_positive_rate']:.2%}")


if __name__ == '__main__':
    main()
