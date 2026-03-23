"""
Train BTC models v2 - with random train/test split and recent data only.
Key changes from v1:
1. Use only last 1-2 years of data (more relevant to current market)
2. Random 75/25 train/test split (not time-based)
3. Train both LONG and SHORT models
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
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

# Directories
MODELS_DIR = "models_btc_v2"
RESULTS_DIR = "results_btc_v2"

# Horizons (in 5-min bars)
HORIZONS = {
    "30min": 6,
    "1h": 12,
    "2h": 24,
    "3h": 36,
    "4h": 48,
    "6h": 72,
    "8h": 96,
    "12h": 144,
}

# Thresholds (percentage move)
THRESHOLDS = [0.5, 0.75, 1.0, 1.5, 2.0]

# Directions
DIRECTIONS = ['LONG', 'SHORT']


def create_target_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Create target variables for all horizon/threshold/direction combinations."""
    print("Creating target variables...")
    
    for horizon_name, horizon_bars in HORIZONS.items():
        future_price = df['Close'].shift(-horizon_bars)
        future_return = (future_price / df['Close'] - 1) * 100  # Percentage
        
        for threshold in THRESHOLDS:
            # LONG target: price goes UP by threshold%
            df[f"target_LONG_{horizon_name}_{threshold}pct"] = (future_return >= threshold).astype(int)
            # SHORT target: price goes DOWN by threshold%
            df[f"target_SHORT_{horizon_name}_{threshold}pct"] = (future_return <= -threshold).astype(int)
    
    # Drop rows with NaN targets
    max_horizon = max(HORIZONS.values())
    df = df.iloc[:-max_horizon]
    
    print(f"Created {len(HORIZONS) * len(THRESHOLDS) * 2} target variables")
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get list of feature columns (exclude targets and raw OHLCV)."""
    exclude_patterns = ['target_', 'future_', 'Open', 'High', 'Low', 'Close', 'Volume']
    feature_cols = [col for col in df.columns 
                   if not any(pat in col for pat in exclude_patterns)]
    return feature_cols


def train_model(X_train: pd.DataFrame, y_train: pd.Series, 
                X_test: pd.DataFrame, y_test: pd.Series,
                target_name: str) -> Tuple[HistGradientBoostingClassifier, Dict]:
    """Train a HistGradientBoosting model with class balancing."""
    
    # Calculate sample weights for imbalanced data
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    
    sample_weight = np.ones(len(y_train))
    if pos_count > 0 and neg_count > 0:
        # Balance classes
        sample_weight[y_train == 1] = neg_count / pos_count
    
    model = HistGradientBoostingClassifier(
        max_iter=300,
        max_depth=5,  # Slightly shallower to reduce overfitting
        learning_rate=0.03,  # Lower learning rate
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        min_samples_leaf=50,  # Require more samples per leaf
        l2_regularization=1.0,  # Add regularization
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
    
    # Calculate win rate at different probability thresholds
    for prob_thresh in [0.55, 0.60, 0.65, 0.70, 0.75]:
        signals = y_pred_proba >= prob_thresh
        if signals.sum() > 0:
            win_rate = y_test[signals].mean()
            signal_pct = signals.mean() * 100
            metrics[f'win_rate_{int(prob_thresh*100)}'] = float(win_rate)
            metrics[f'signal_pct_{int(prob_thresh*100)}'] = float(signal_pct)
        else:
            metrics[f'win_rate_{int(prob_thresh*100)}'] = 0
            metrics[f'signal_pct_{int(prob_thresh*100)}'] = 0
    
    return model, metrics


def train_all_models(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Train models for all combinations with random train/test split."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    all_results = []
    total_models = len(HORIZONS) * len(THRESHOLDS) * len(DIRECTIONS)
    model_count = 0
    
    print(f"\nTraining {total_models} models...")
    print("=" * 80)
    
    for direction in DIRECTIONS:
        for horizon_name, horizon_bars in HORIZONS.items():
            for threshold in THRESHOLDS:
                model_count += 1
                target_col = f"target_{direction}_{horizon_name}_{threshold}pct"
                
                if target_col not in df.columns:
                    print(f"  ⚠️ Target {target_col} not found, skipping")
                    continue
                
                print(f"\n[{model_count}/{total_models}] {direction} {horizon_name} {threshold}%")
                
                # Prepare data
                X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
                y = df[target_col]
                
                # Random 75/25 split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.25, random_state=42, shuffle=True
                )
                
                # Skip if not enough positive samples
                if y_train.sum() < 100 or y_test.sum() < 50:
                    print(f"  ⚠️ Skipping - insufficient positive samples (train: {y_train.sum()}, test: {y_test.sum()})")
                    continue
                
                # Train model
                model, metrics = train_model(X_train, y_train, X_test, y_test, target_col)
                
                metrics['direction'] = direction
                metrics['horizon'] = horizon_name
                metrics['horizon_bars'] = horizon_bars
                metrics['threshold'] = float(threshold)
                
                print(f"  Precision: {metrics['precision']:.3f} | Recall: {metrics['recall']:.3f} | ROC-AUC: {metrics['roc_auc']:.3f}")
                print(f"  Win rates: 55%={metrics.get('win_rate_55', 0):.1%} | 60%={metrics.get('win_rate_60', 0):.1%} | 65%={metrics.get('win_rate_65', 0):.1%}")
                
                # Save model
                suffix = '_SHORT' if direction == 'SHORT' else ''
                model_path = f"{MODELS_DIR}/model_{horizon_name}_{threshold}pct{suffix}.joblib"
                joblib.dump(model, model_path)
                
                all_results.append(metrics)
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f"{RESULTS_DIR}/training_results.csv", index=False)
    
    # Save feature columns
    with open(f"{MODELS_DIR}/feature_columns.json", 'w') as f:
        json.dump(feature_cols, f)
    
    return results_df


def analyze_results(results_df: pd.DataFrame) -> None:
    """Analyze and display training results."""
    
    print("\n" + "=" * 80)
    print("TRAINING RESULTS SUMMARY")
    print("=" * 80)
    
    # Best LONG models
    print("\n📈 TOP 10 LONG MODELS (by win rate at 65% threshold):")
    long_df = results_df[results_df['direction'] == 'LONG'].copy()
    if len(long_df) > 0:
        long_df = long_df.sort_values('win_rate_65', ascending=False)
        cols = ['horizon', 'threshold', 'precision', 'recall', 'win_rate_55', 'win_rate_60', 'win_rate_65', 'signal_pct_65']
        print(long_df[cols].head(10).to_string(index=False))
    
    # Best SHORT models
    print("\n📉 TOP 10 SHORT MODELS (by win rate at 65% threshold):")
    short_df = results_df[results_df['direction'] == 'SHORT'].copy()
    if len(short_df) > 0:
        short_df = short_df.sort_values('win_rate_65', ascending=False)
        cols = ['horizon', 'threshold', 'precision', 'recall', 'win_rate_55', 'win_rate_60', 'win_rate_65', 'signal_pct_65']
        print(short_df[cols].head(10).to_string(index=False))
    
    # Promising models (high win rate AND reasonable signal frequency)
    print("\n🎯 PROMISING MODELS (win_rate_65 > 55% AND signal_pct_65 > 5%):")
    promising = results_df[
        (results_df['win_rate_65'] > 0.55) & 
        (results_df['signal_pct_65'] > 5)
    ].sort_values('win_rate_65', ascending=False)
    
    if len(promising) > 0:
        cols = ['direction', 'horizon', 'threshold', 'win_rate_65', 'signal_pct_65', 'precision']
        print(promising[cols].head(15).to_string(index=False))
    else:
        print("No models meet strict criteria. Relaxing to win_rate_65 > 50%:")
        relaxed = results_df[results_df['win_rate_65'] > 0.50].sort_values('win_rate_65', ascending=False)
        if len(relaxed) > 0:
            cols = ['direction', 'horizon', 'threshold', 'win_rate_65', 'signal_pct_65', 'precision']
            print(relaxed[cols].head(15).to_string(index=False))


def main():
    print("=" * 80)
    print("BTC MODEL TRAINING v2 - RANDOM SPLIT, RECENT DATA")
    print("=" * 80)
    print(f"Started at: {datetime.now()}")
    
    # Load data
    print("\n📂 Loading BTC data...")
    df = pd.read_parquet('data/BTC_features.parquet')
    print(f"Total data: {len(df):,} rows")
    print(f"Full date range: {df.index[0]} to {df.index[-1]}")
    
    # Filter to last 2 years only
    cutoff_date = df.index[-1] - pd.Timedelta(days=730)  # 2 years
    df = df[df.index >= cutoff_date]
    print(f"\nUsing last 2 years: {len(df):,} rows")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Create target variables
    df = create_target_variables(df)
    
    # Get feature columns
    feature_cols = get_feature_columns(df)
    print(f"Features: {len(feature_cols)}")
    
    # Train all models
    results_df = train_all_models(df, feature_cols)
    
    # Analyze results
    analyze_results(results_df)
    
    print(f"\n✅ Training complete at: {datetime.now()}")
    print(f"Models saved to: {MODELS_DIR}/")
    print(f"Results saved to: {RESULTS_DIR}/training_results.csv")


if __name__ == "__main__":
    main()
