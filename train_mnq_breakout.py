"""
Train MNQ Breakout/Early Momentum Models

These models fire EARLY in a move - when momentum is just starting.
Key: RSI crossing above 50, MACD turning positive, price breaking above MA.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

OUTPUT_DIR = 'models_breakout'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Breakout model configurations - shorter horizons, catch early moves
BREAKOUT_MODELS = [
    {'horizon': '1h', 'horizon_bars': 12, 'threshold': 0.5, 'direction': 'LONG'},
    {'horizon': '2h', 'horizon_bars': 24, 'threshold': 0.75, 'direction': 'LONG'},
    {'horizon': '1h', 'horizon_bars': 12, 'threshold': 0.5, 'direction': 'SHORT'},
    {'horizon': '2h', 'horizon_bars': 24, 'threshold': 0.75, 'direction': 'SHORT'},
]

CUMULATIVE_FEATURES = ['volume_obv', 'volume_adi', 'volume_nvi', 'volume_vpt', 'others_cr']
PRICE_FEATURES = ['volatility_atr', 'trend_macd', 'trend_macd_signal', 'trend_macd_diff',
                  'volatility_bbh', 'volatility_bbl', 'volatility_bbm', 'volatility_bbw',
                  'volatility_kch', 'volatility_kcl', 'volatility_kcw',
                  'volatility_dch', 'volatility_dcl', 'volatility_dcm', 'volatility_dcw']


def normalize_features(df):
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


def create_breakout_labels(df, horizon_bars, threshold_pct, direction):
    """
    Breakout labels: Catch the START of a move.
    
    LONG breakout conditions:
    - RSI was below 55 recently (not overbought yet)
    - RSI is now rising (momentum building)
    - MACD diff turning positive or just turned positive
    - Price crosses above MA10 or MA20
    
    SHORT breakout conditions:
    - RSI was above 45 recently (not oversold yet)
    - RSI is now falling
    - MACD diff turning negative
    - Price crosses below MA10 or MA20
    """
    future_return = df['Close'].shift(-horizon_bars) / df['Close'] - 1
    
    ma10 = df['Close'].rolling(10).mean()
    ma20 = df['Close'].rolling(20).mean()
    
    # RSI momentum
    rsi = df['momentum_rsi']
    rsi_rising = rsi > rsi.shift(3)  # RSI rising over last 3 bars
    rsi_falling = rsi < rsi.shift(3)
    
    # MACD momentum
    macd_diff = df['trend_macd_diff']
    macd_turning_up = (macd_diff > macd_diff.shift(1)) & (macd_diff.shift(1) > macd_diff.shift(2))
    macd_turning_down = (macd_diff < macd_diff.shift(1)) & (macd_diff.shift(1) < macd_diff.shift(2))
    
    # Price breakout
    price_above_ma10 = df['Close'] > ma10
    price_above_ma20 = df['Close'] > ma20
    price_below_ma10 = df['Close'] < ma10
    price_below_ma20 = df['Close'] < ma20
    
    # Just crossed above/below
    just_crossed_above_ma20 = price_above_ma20 & ~price_above_ma20.shift(1).fillna(False)
    just_crossed_below_ma20 = price_below_ma20 & ~price_below_ma20.shift(1).fillna(False)
    
    if direction == 'LONG':
        # Breakout LONG: RSI not too high yet, momentum building
        breakout_condition = (
            (rsi < 65) &  # Not overbought yet
            (rsi > 45) &  # But not oversold either
            rsi_rising &  # Momentum building
            (macd_turning_up | (macd_diff > 0)) &  # MACD supportive
            price_above_ma10  # Price above short-term MA
        )
        success = future_return >= (threshold_pct / 100)
    else:
        # Breakout SHORT: RSI not too low yet, momentum building down
        breakout_condition = (
            (rsi > 35) &  # Not oversold yet
            (rsi < 55) &  # But not overbought either
            rsi_falling &  # Momentum building down
            (macd_turning_down | (macd_diff < 0)) &  # MACD supportive
            price_below_ma10  # Price below short-term MA
        )
        success = future_return <= -(threshold_pct / 100)
    
    labels = (breakout_condition & success).astype(int)
    return labels, breakout_condition


def get_feature_columns(df):
    exclude = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'label', 'future_return']
    for feat in CUMULATIVE_FEATURES:
        exclude.append(feat)
        for lag in [1, 2, 3, 5, 10, 20, 50]:
            exclude.append(f'{feat}_lag{lag}')
    for feat in PRICE_FEATURES:
        exclude.append(feat)
        for lag in [1, 2, 3, 5, 10, 20, 50]:
            exclude.append(f'{feat}_lag{lag}')
    return [c for c in df.columns if c not in exclude and not c.startswith('label')]


def train_breakout_model(df, config):
    horizon = config['horizon']
    horizon_bars = config['horizon_bars']
    threshold = config['threshold']
    direction = config['direction']
    
    print(f"\n{'='*60}")
    print(f"Training {direction} Breakout Model: {horizon}_{threshold}pct")
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
    
    if y.sum() < 100:
        print(f"WARNING: Not enough positive samples")
        return None
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
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
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"\nTest Results:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    try:
        auc = roc_auc_score(y_test, y_proba)
        print(f"AUC-ROC: {auc:.4f}")
    except:
        auc = 0
    
    high_conf = y_proba >= 0.55
    if high_conf.sum() > 0:
        print(f"Precision at prob>=55%: {y_test[high_conf].mean():.2%} ({high_conf.sum()} samples)")
    
    return {'model': model, 'features': feature_cols, 'config': config, 'auc': auc}


def main():
    print("Loading data...")
    df = pd.read_parquet('data/QQQ_features.parquet')
    df = df[df.index >= '2024-01-01']
    print(f"Using {len(df)} rows from 2024")
    
    print("Normalizing features...")
    df = normalize_features(df)
    
    results = []
    for config in BREAKOUT_MODELS:
        result = train_breakout_model(df, config)
        if result:
            results.append(result)
            
            direction = config['direction']
            suffix = '_SHORT' if direction == 'SHORT' else ''
            model_name = f"model_{config['horizon']}_{config['threshold']}pct{suffix}_breakout.joblib"
            joblib.dump(result['model'], os.path.join(OUTPUT_DIR, model_name))
            
            feature_name = f"features_{config['horizon']}_{config['threshold']}pct{suffix}_breakout.txt"
            with open(os.path.join(OUTPUT_DIR, feature_name), 'w') as f:
                f.write('\n'.join(result['features']))
            
            print(f"Saved: {model_name}")
    
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    for r in results:
        cfg = r['config']
        print(f"{cfg['horizon']}_{cfg['threshold']}pct {cfg['direction']}: AUC={r['auc']:.4f}")


if __name__ == '__main__':
    main()
