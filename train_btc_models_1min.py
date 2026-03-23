"""
Train ML models for BTC futures trading using 1-minute data.
Compare performance with 5-minute data models.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import joblib
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import ta

MODELS_DIR_1MIN = "models_btc_1min"
RESULTS_DIR_1MIN = "results_btc_1min"

HORIZONS_1MIN = {
    "15min": 15,
    "30min": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "8h": 480,
    "12h": 720,
}

THRESHOLDS = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators."""
    print("Adding technical indicators...")
    
    df['sma_10'] = ta.trend.sma_indicator(df['Close'], window=10)
    df['sma_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['sma_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['ema_10'] = ta.trend.ema_indicator(df['Close'], window=10)
    df['ema_20'] = ta.trend.ema_indicator(df['Close'], window=20)
    
    macd = ta.trend.MACD(df['Close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    df['rsi_14'] = ta.momentum.rsi(df['Close'], window=14)
    df['rsi_7'] = ta.momentum.rsi(df['Close'], window=7)
    
    bb = ta.volatility.BollingerBands(df['Close'], window=20)
    df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    df['bb_pct'] = (df['Close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-10)
    
    df['atr_14'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    
    df['volume_sma_20'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / (df['volume_sma_20'] + 1e-10)
    
    df['returns_1'] = df['Close'].pct_change(1)
    df['returns_5'] = df['Close'].pct_change(5)
    df['returns_15'] = df['Close'].pct_change(15)
    df['returns_60'] = df['Close'].pct_change(60)
    
    df['volatility_15'] = df['returns_1'].rolling(15).std()
    df['volatility_60'] = df['returns_1'].rolling(60).std()
    
    df['high_low_range'] = (df['High'] - df['Low']) / df['Close']
    df['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    
    df['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
    df['momentum_30'] = df['Close'] / df['Close'].shift(30) - 1
    df['momentum_60'] = df['Close'] / df['Close'].shift(60) - 1
    
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'])
    df['adx'] = adx.adx()
    
    df['cci'] = ta.trend.cci(df['High'], df['Low'], df['Close'])
    df['williams_r'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'])
    
    return df


def create_target_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Create target variables."""
    print("Creating target variables...")
    
    for horizon_name, horizon_bars in HORIZONS_1MIN.items():
        future_price = df['Close'].shift(-horizon_bars)
        future_return = (future_price / df['Close'] - 1) * 100
        
        for threshold in THRESHOLDS:
            target_col = f"target_up_{horizon_name}_{threshold}pct"
            df[target_col] = (future_return >= threshold).astype(int)
    
    max_horizon = max(HORIZONS_1MIN.values())
    df = df.iloc[:-max_horizon]
    
    print(f"Created {len(HORIZONS_1MIN) * len(THRESHOLDS)} target variables")
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get feature columns."""
    exclude = ['target_', 'future_', 'Open', 'High', 'Low', 'Close', 'Volume']
    return [c for c in df.columns if not any(p in c for p in exclude)]


def train_model(X_train, y_train, X_test, y_test, target_name):
    """Train a model."""
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    
    sample_weight = np.ones(len(y_train))
    if pos_count > 0 and neg_count > 0:
        sample_weight[y_train == 1] = neg_count / pos_count
    
    model = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=6,
        learning_rate=0.05,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
        verbose=0
    )
    
    model.fit(X_train, y_train, sample_weight=sample_weight)
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    metrics = {
        'target': target_name,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'positive_rate_train': float(y_train.mean()),
        'positive_rate_test': float(y_test.mean()),
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_pred_proba)) if y_test.sum() > 0 else 0,
        'n_iterations': model.n_iter_,
    }
    
    return model, metrics


def main():
    print("="*70)
    print("BTC MODEL TRAINING - 1-MINUTE DATA")
    print("="*70)
    print(f"Started: {datetime.now()}")
    
    # Load 1-min data
    print("\nLoading 1-minute data...")
    df = pd.read_parquet('data/historical/BTC_1min.parquet')
    print(f"Loaded {len(df):,} bars")
    print(f"Range: {df.index[0]} to {df.index[-1]}")
    
    # Use FULL 1-min data - no sampling!
    print(f"\nUsing FULL 1-min data: {len(df):,} bars")
    
    # Add features
    df = add_technical_indicators(df)
    
    # Create targets
    df = create_target_variables(df)
    
    # Get features
    feature_cols = get_feature_columns(df)
    print(f"Features: {len(feature_cols)}")
    
    # Clean
    df[feature_cols] = df[feature_cols].fillna(0)
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], 0)
    
    # Train/test split
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    os.makedirs(MODELS_DIR_1MIN, exist_ok=True)
    os.makedirs(RESULTS_DIR_1MIN, exist_ok=True)
    
    # Train models
    target_cols = [c for c in df.columns if c.startswith('target_up_')]
    results = []
    
    print(f"\nTraining {len(target_cols)} models...")
    print("="*70)
    
    for i, target_col in enumerate(target_cols):
        parts = target_col.replace('target_up_', '').replace('pct', '').split('_')
        horizon = parts[0]
        threshold = parts[1]
        
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]
        
        if y_train.sum() < 100 or y_test.sum() < 50:
            print(f"[{i+1}/{len(target_cols)}] {horizon}_{threshold}% - SKIP (low samples)")
            continue
        
        model, metrics = train_model(X_train, y_train, X_test, y_test, target_col)
        metrics['horizon'] = horizon
        metrics['threshold'] = float(threshold)
        
        print(f"[{i+1}/{len(target_cols)}] {horizon}_{threshold}% - AUC:{metrics['roc_auc']:.3f} Prec:{metrics['precision']:.3f}")
        
        model_path = f"{MODELS_DIR_1MIN}/model_{horizon}_{threshold}pct.joblib"
        joblib.dump(model, model_path)
        results.append(metrics)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{RESULTS_DIR_1MIN}/training_results.csv", index=False)
    
    with open(f"{MODELS_DIR_1MIN}/feature_columns.json", 'w') as f:
        json.dump(feature_cols, f)
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    print("\nTop 10 by ROC-AUC:")
    print(results_df.nlargest(10, 'roc_auc')[['horizon', 'threshold', 'precision', 'recall', 'roc_auc']].to_string(index=False))
    
    print("\nTop 10 by Precision:")
    print(results_df.nlargest(10, 'precision')[['horizon', 'threshold', 'precision', 'recall', 'roc_auc']].to_string(index=False))
    
    print(f"\nCompleted: {datetime.now()}")
    print(f"Models: {MODELS_DIR_1MIN}/")
    print(f"Results: {RESULTS_DIR_1MIN}/training_results.csv")


if __name__ == "__main__":
    main()
