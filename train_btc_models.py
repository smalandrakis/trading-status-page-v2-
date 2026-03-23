"""
Train ML models for BTC futures trading.
Uses the same approach as QQQ but with BTC-specific data.
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

import config

# BTC-specific directories
BTC_MODELS_DIR = "models_btc"
BTC_RESULTS_DIR = "results_btc"

# Horizons and thresholds for BTC (can be different from QQQ)
# BTC is more volatile, so we might want different thresholds
BTC_HORIZONS = {
    "5min": 1,
    "10min": 2,
    "15min": 3,
    "30min": 6,
    "45min": 9,
    "1h": 12,
    "1h30m": 18,
    "2h": 24,
    "3h": 36,
    "4h": 48,
    "6h": 72,
    "8h": 96,
    "12h": 144,
}

# Higher thresholds for BTC due to volatility
BTC_THRESHOLDS = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]


def create_target_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Create target variables for all horizon/threshold combinations."""
    print("Creating target variables...")
    
    for horizon_name, horizon_bars in BTC_HORIZONS.items():
        # Calculate future price
        future_price = df['Close'].shift(-horizon_bars)
        future_return = (future_price / df['Close'] - 1) * 100  # Percentage
        
        for threshold in BTC_THRESHOLDS:
            # Binary target: 1 if price goes up by threshold%, 0 otherwise
            target_col = f"target_up_{horizon_name}_{threshold}pct"
            df[target_col] = (future_return >= threshold).astype(int)
            
    # Drop rows with NaN targets (last rows that can't have future price)
    max_horizon = max(BTC_HORIZONS.values())
    df = df.iloc[:-max_horizon]
    
    print(f"Created {len(BTC_HORIZONS) * len(BTC_THRESHOLDS)} target variables")
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get list of feature columns (exclude targets and raw OHLCV)."""
    exclude_patterns = ['target_', 'future_', 'Open', 'High', 'Low', 'Close', 'Volume']
    feature_cols = [col for col in df.columns 
                   if not any(pat in col for pat in exclude_patterns)]
    return feature_cols


def prepare_train_test_split(df: pd.DataFrame, feature_cols: List[str], 
                             target_col: str, train_ratio: float = 0.8) -> Tuple:
    """Time-based train/test split."""
    df = df.sort_index()
    split_idx = int(len(df) * train_ratio)
    
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series, 
                X_test: pd.DataFrame, y_test: pd.Series,
                target_name: str) -> Tuple[HistGradientBoostingClassifier, Dict]:
    """Train a HistGradientBoosting model."""
    
    # Calculate sample weights for imbalanced data
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
    
    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Metrics
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


def train_all_models(df: pd.DataFrame, feature_cols: List[str], batch_size: int = 10) -> pd.DataFrame:
    """Train models for all horizon/threshold combinations in batches."""
    os.makedirs(BTC_MODELS_DIR, exist_ok=True)
    os.makedirs(BTC_RESULTS_DIR, exist_ok=True)
    
    # Get all target columns
    target_cols = [col for col in df.columns if col.startswith('target_up_')]
    all_results = []
    
    print(f"\nTraining {len(target_cols)} models in batches of {batch_size}...")
    print("="*80)
    
    for i, target_col in enumerate(target_cols):
        # Parse horizon and threshold
        parts = target_col.replace('target_up_', '').replace('pct', '').split('_')
        horizon = parts[0]
        threshold = parts[1]
        
        print(f"\n[{i+1}/{len(target_cols)}] Training: {horizon} horizon, {threshold}% threshold")
        
        # Prepare data
        X_train, X_test, y_train, y_test = prepare_train_test_split(
            df, feature_cols, target_col
        )
        
        # Skip if not enough positive samples
        if y_train.sum() < 100 or y_test.sum() < 50:
            print(f"  ⚠️ Skipping - insufficient positive samples (train: {y_train.sum()}, test: {y_test.sum()})")
            continue
        
        # Train model
        model, metrics = train_model(X_train, y_train, X_test, y_test, target_col)
        
        metrics['horizon'] = horizon
        metrics['threshold'] = float(threshold)
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        
        # Save model
        model_path = f"{BTC_MODELS_DIR}/model_{horizon}_{threshold}pct.joblib"
        joblib.dump(model, model_path)
        
        all_results.append(metrics)
        
        # Save intermediate results every batch
        if (i + 1) % batch_size == 0:
            results_df = pd.DataFrame(all_results)
            results_df.to_csv(f"{BTC_RESULTS_DIR}/training_results_partial.csv", index=False)
            print(f"\n  📁 Saved intermediate results ({len(all_results)} models)")
    
    # Final results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f"{BTC_RESULTS_DIR}/training_results.csv", index=False)
    
    # Save feature columns
    with open(f"{BTC_MODELS_DIR}/feature_columns.json", 'w') as f:
        json.dump(feature_cols, f)
    
    return results_df


def analyze_results(results_df: pd.DataFrame) -> None:
    """Analyze and display training results."""
    
    print("\n" + "="*80)
    print("BTC MODEL TRAINING RESULTS")
    print("="*80)
    
    print("\n📊 Top 10 Models by ROC-AUC:")
    top_auc = results_df.nlargest(10, 'roc_auc')[
        ['horizon', 'threshold', 'accuracy', 'precision', 'recall', 'roc_auc', 'positive_rate_test']
    ]
    print(top_auc.to_string(index=False))
    
    print("\n📊 Top 10 Models by Precision:")
    top_precision = results_df.nlargest(10, 'precision')[
        ['horizon', 'threshold', 'accuracy', 'precision', 'recall', 'roc_auc', 'positive_rate_test']
    ]
    print(top_precision.to_string(index=False))
    
    print("\n📊 Average Metrics by Horizon:")
    by_horizon = results_df.groupby('horizon')[['accuracy', 'precision', 'recall', 'roc_auc']].mean()
    print(by_horizon.round(4).to_string())
    
    print("\n📊 Average Metrics by Threshold:")
    by_threshold = results_df.groupby('threshold')[['accuracy', 'precision', 'recall', 'roc_auc']].mean()
    print(by_threshold.round(4).to_string())
    
    # Promising models for trading
    print("\n🎯 Promising Models (Precision > 0.55 AND Recall > 0.3):")
    promising = results_df[
        (results_df['precision'] > 0.55) & 
        (results_df['recall'] > 0.3)
    ].sort_values('precision', ascending=False)
    
    if len(promising) > 0:
        print(promising[['horizon', 'threshold', 'accuracy', 'precision', 'recall', 'roc_auc']].head(15).to_string(index=False))
    else:
        print("No models meet strict criteria. Best available:")
        print(results_df.nlargest(10, 'precision')[
            ['horizon', 'threshold', 'accuracy', 'precision', 'recall', 'roc_auc']
        ].to_string(index=False))


def main():
    print("="*80)
    print("BTC FUTURES MODEL TRAINING")
    print("="*80)
    print(f"Started at: {datetime.now()}")
    
    # Load BTC data with features
    print("\n📂 Loading BTC data...")
    df = pd.read_parquet('data/BTC_features.parquet')
    print(f"Loaded {len(df):,} rows")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Create target variables
    df = create_target_variables(df)
    
    # Get feature columns
    feature_cols = get_feature_columns(df)
    print(f"Features: {len(feature_cols)}")
    
    # Clean data
    df[feature_cols] = df[feature_cols].fillna(0)
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], 0)
    
    # Train all models
    results_df = train_all_models(df, feature_cols, batch_size=10)
    
    # Analyze results
    analyze_results(results_df)
    
    print(f"\n✅ Training complete at: {datetime.now()}")
    print(f"Models saved to: {BTC_MODELS_DIR}/")
    print(f"Results saved to: {BTC_RESULTS_DIR}/training_results.csv")


if __name__ == "__main__":
    main()
