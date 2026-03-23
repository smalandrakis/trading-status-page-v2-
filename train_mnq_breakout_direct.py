"""
Train MNQ Breakout Models DIRECTLY on MNQ data (not QQQ proxy)

Uses 14 months of actual MNQ futures data for more accurate signals.
Features are normalized for scale invariance.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

OUTPUT_DIR = 'models_mnq_breakout'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Breakout model configurations
BREAKOUT_MODELS = [
    {'horizon': '1h', 'horizon_bars': 12, 'threshold': 0.5, 'direction': 'LONG'},
    {'horizon': '2h', 'horizon_bars': 24, 'threshold': 0.75, 'direction': 'LONG'},
    {'horizon': '1h', 'horizon_bars': 12, 'threshold': 0.5, 'direction': 'SHORT'},
    {'horizon': '2h', 'horizon_bars': 24, 'threshold': 0.75, 'direction': 'SHORT'},
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
    
    # 1. Cumulative features: use pct_change (rate of change)
    for feat in CUMULATIVE_FEATURES:
        if feat in df.columns:
            df[f'{feat}_pct'] = df[feat].pct_change().fillna(0).clip(-0.1, 0.1)
            for lag in [1, 2, 3, 5, 10, 20, 50]:
                lag_feat = f'{feat}_lag{lag}'
                if lag_feat in df.columns:
                    df[f'{lag_feat}_pct'] = df[lag_feat].pct_change().fillna(0).clip(-0.1, 0.1)
    
    # 2. Price-based features: normalize by Close price
    for feat in PRICE_FEATURES:
        if feat in df.columns and 'Close' in df.columns:
            df[f'{feat}_norm'] = df[feat] / df['Close']
            for lag in [1, 2, 3, 5, 10, 20, 50]:
                lag_feat = f'{feat}_lag{lag}'
                if lag_feat in df.columns:
                    df[f'{lag_feat}_norm'] = df[lag_feat] / df['Close']
    
    return df


def create_breakout_labels(df: pd.DataFrame, horizon_bars: int, threshold_pct: float, direction: str):
    """
    Create labels for breakout/early momentum signals.
    
    LONG breakout: RSI 45-65, momentum building, not overbought yet
    SHORT breakout: RSI 35-55, momentum building down, not oversold yet
    """
    future_return = df['Close'].shift(-horizon_bars) / df['Close'] - 1
    
    ma10 = df['Close'].rolling(10).mean()
    rsi = df['momentum_rsi']
    rsi_rising = rsi > rsi.shift(3)
    rsi_falling = rsi < rsi.shift(3)
    
    macd_diff = df['trend_macd_diff']
    macd_turning_up = (macd_diff > macd_diff.shift(1)) & (macd_diff.shift(1) > macd_diff.shift(2))
    macd_turning_down = (macd_diff < macd_diff.shift(1)) & (macd_diff.shift(1) < macd_diff.shift(2))
    
    price_above_ma10 = df['Close'] > ma10
    price_below_ma10 = df['Close'] < ma10
    
    if direction == 'LONG':
        # Breakout LONG: catch early momentum, not overbought
        breakout_condition = (
            (rsi < 65) &  # Not overbought yet
            (rsi > 45) &  # But showing some strength
            rsi_rising &  # Momentum building
            (macd_turning_up | (macd_diff > 0)) &
            price_above_ma10
        )
        success = future_return >= (threshold_pct / 100)
    else:
        # Breakout SHORT: catch early downward momentum
        breakout_condition = (
            (rsi > 35) &  # Not oversold yet
            (rsi < 55) &  # But showing weakness
            rsi_falling &
            (macd_turning_down | (macd_diff < 0)) &
            price_below_ma10
        )
        success = future_return <= -(threshold_pct / 100)
    
    labels = (breakout_condition & success).astype(int)
    return labels, breakout_condition


def get_feature_columns(df: pd.DataFrame) -> list:
    """Get feature columns, excluding raw OHLCV and non-normalized cumulative/price features."""
    exclude = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'label', 'date']
    
    # Exclude raw cumulative features (use _pct versions)
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


def train_breakout_model(df: pd.DataFrame, config: dict) -> dict:
    """Train a single breakout model."""
    horizon = config['horizon']
    horizon_bars = config['horizon_bars']
    threshold = config['threshold']
    direction = config['direction']
    
    print(f"\n{'='*60}")
    print(f"Training MNQ {direction} Breakout: {horizon}_{threshold}pct")
    print(f"{'='*60}")
    
    labels, breakout_condition = create_breakout_labels(df, horizon_bars, threshold, direction)
    feature_cols = get_feature_columns(df)
    
    valid_idx = ~labels.isna() & ~breakout_condition.isna()
    X = df.loc[valid_idx, feature_cols].copy()
    y = labels[valid_idx].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    print(f"Total samples: {len(X)}")
    print(f"Positive samples: {y.sum()}")
    print(f"Positive rate: {y.mean()*100:.2f}%")
    
    if y.sum() < 50:
        print(f"WARNING: Not enough positive samples")
        return None
    
    # Time-based split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Train positive rate: {y_train.mean()*100:.2f}%")
    print(f"Test positive rate: {y_test.mean()*100:.2f}%")
    
    model = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=6,
        learning_rate=0.05,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"\nTest Results:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    try:
        auc = roc_auc_score(y_test, y_proba)
        print(f"AUC-ROC: {auc:.4f}")
    except:
        auc = 0
    
    # Check precision at different thresholds
    for thresh in [0.50, 0.55, 0.60]:
        mask = y_proba >= thresh
        if mask.sum() > 0:
            precision = y_test[mask].mean()
            print(f"Precision at prob>={thresh:.0%}: {precision:.2%} ({mask.sum()} samples)")
    
    return {
        'model': model,
        'features': feature_cols,
        'config': config,
        'auc': auc,
        'train_positive_rate': y_train.mean(),
        'test_positive_rate': y_test.mean()
    }


def main():
    print("Loading MNQ data...")
    df = pd.read_parquet('data/MNQ_features.parquet')
    print(f"Loaded {len(df)} rows")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
    
    # Normalize features
    print("\nNormalizing features for scale invariance...")
    df = normalize_features(df)
    print(f"Total columns after normalization: {len(df.columns)}")
    
    # Train models
    results = []
    for config in BREAKOUT_MODELS:
        result = train_breakout_model(df, config)
        if result:
            results.append(result)
            
            direction = config['direction']
            suffix = '_SHORT' if direction == 'SHORT' else ''
            model_name = f"model_{config['horizon']}_{config['threshold']}pct{suffix}_mnq_breakout.joblib"
            model_path = os.path.join(OUTPUT_DIR, model_name)
            joblib.dump(result['model'], model_path)
            print(f"Saved: {model_path}")
            
            feature_name = f"features_{config['horizon']}_{config['threshold']}pct{suffix}_mnq_breakout.txt"
            feature_path = os.path.join(OUTPUT_DIR, feature_name)
            with open(feature_path, 'w') as f:
                f.write('\n'.join(result['features']))
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY - MNQ BREAKOUT MODELS")
    print("="*60)
    for r in results:
        cfg = r['config']
        print(f"{cfg['horizon']}_{cfg['threshold']}pct {cfg['direction']}: "
              f"AUC={r['auc']:.4f}, train_pos={r['train_positive_rate']:.2%}, test_pos={r['test_positive_rate']:.2%}")


if __name__ == '__main__':
    main()
