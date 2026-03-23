"""
Train ML models for all horizon/threshold combinations.
Uses sklearn's HistGradientBoosting for fast training with good performance.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
import joblib
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import config
from feature_engineering import get_feature_columns


def load_processed_data() -> pd.DataFrame:
    """Load the processed feature data."""
    filepath = f"{config.DATA_DIR}/QQQ_features.parquet"
    df = pd.read_parquet(filepath)
    print(f"Loaded {len(df)} rows from {filepath}")
    return df


def get_target_columns(df: pd.DataFrame) -> List[str]:
    """Get list of target columns for training."""
    # Use the binary 'up' targets for classification
    target_cols = [col for col in df.columns if col.startswith('target_up_')]
    return target_cols


def prepare_train_test_split(df: pd.DataFrame, feature_cols: List[str], 
                             target_col: str) -> Tuple:
    """
    Prepare train/test split.
    Uses time-based split (not random) to avoid look-ahead bias.
    """
    # Sort by index (time)
    df = df.sort_index()
    
    # Calculate split point
    split_idx = int(len(df) * config.TRAIN_RATIO)
    
    # Split data
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    return X_train, X_test, y_train, y_test, train_df.index, test_df.index


def train_model(X_train: pd.DataFrame, y_train: pd.Series, 
                X_test: pd.DataFrame, y_test: pd.Series,
                target_name: str) -> Tuple[HistGradientBoostingClassifier, Dict]:
    """
    Train a HistGradientBoosting model for binary classification.
    """
    # Calculate class weights for imbalanced data
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    
    # Create sample weights
    sample_weight = np.ones(len(y_train))
    if pos_count > 0:
        sample_weight[y_train == 1] = neg_count / pos_count
    
    # HistGradientBoosting model (fast, no OpenMP needed)
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
    
    # Train model
    model.fit(X_train, y_train, sample_weight=sample_weight)
    
    # Make predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Calculate metrics
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
    
    # Note: HistGradientBoosting doesn't have feature_importances_ by default
    # We'll skip top features for now
    metrics['top_features'] = []
    
    return model, metrics


def train_all_models(df: pd.DataFrame, feature_cols: List[str]) -> Dict:
    """
    Train models for all horizon/threshold combinations.
    """
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    target_cols = get_target_columns(df)
    all_results = []
    
    print(f"\nTraining {len(target_cols)} models...")
    print("="*80)
    
    for i, target_col in enumerate(target_cols):
        # Parse horizon and threshold from target name
        # Format: target_up_{horizon}_{threshold}pct
        parts = target_col.replace('target_up_', '').replace('pct', '').split('_')
        horizon = parts[0]
        threshold = parts[1]
        
        print(f"\n[{i+1}/{len(target_cols)}] Training: {horizon} horizon, {threshold}% threshold")
        
        # Prepare data
        X_train, X_test, y_train, y_test, train_idx, test_idx = prepare_train_test_split(
            df, feature_cols, target_col
        )
        
        # Train model
        model, metrics = train_model(X_train, y_train, X_test, y_test, target_col)
        
        # Add horizon and threshold to metrics
        metrics['horizon'] = horizon
        metrics['threshold'] = float(threshold)
        
        # Print summary
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"  Positive rate (train/test): {metrics['positive_rate_train']:.4f} / {metrics['positive_rate_test']:.4f}")
        
        # Save model
        model_path = f"{config.MODELS_DIR}/model_{horizon}_{threshold}pct.joblib"
        joblib.dump(model, model_path)
        
        all_results.append(metrics)
    
    # Create results summary
    results_df = pd.DataFrame(all_results)
    
    # Save results
    results_path = f"{config.RESULTS_DIR}/training_results.csv"
    results_df.to_csv(results_path, index=False)
    
    # Save feature columns for later use
    with open(f"{config.MODELS_DIR}/feature_columns.json", 'w') as f:
        json.dump(feature_cols, f)
    
    return results_df


def analyze_results(results_df: pd.DataFrame) -> None:
    """Analyze and display training results."""
    
    print("\n" + "="*80)
    print("TRAINING RESULTS SUMMARY")
    print("="*80)
    
    # Best models by ROC-AUC
    print("\n📊 Top 10 Models by ROC-AUC:")
    top_auc = results_df.nlargest(10, 'roc_auc')[
        ['horizon', 'threshold', 'accuracy', 'precision', 'recall', 'roc_auc', 'positive_rate_test']
    ]
    print(top_auc.to_string(index=False))
    
    # Best models by Precision (important for trading - fewer false positives)
    print("\n📊 Top 10 Models by Precision:")
    top_precision = results_df.nlargest(10, 'precision')[
        ['horizon', 'threshold', 'accuracy', 'precision', 'recall', 'roc_auc', 'positive_rate_test']
    ]
    print(top_precision.to_string(index=False))
    
    # Results by horizon
    print("\n📊 Average Metrics by Horizon:")
    by_horizon = results_df.groupby('horizon')[['accuracy', 'precision', 'recall', 'roc_auc']].mean()
    print(by_horizon.round(4).to_string())
    
    # Results by threshold
    print("\n📊 Average Metrics by Threshold:")
    by_threshold = results_df.groupby('threshold')[['accuracy', 'precision', 'recall', 'roc_auc']].mean()
    print(by_threshold.round(4).to_string())
    
    # Identify promising models (high precision + reasonable recall)
    print("\n🎯 Promising Models (Precision > 0.55 AND Recall > 0.3):")
    promising = results_df[
        (results_df['precision'] > 0.55) & 
        (results_df['recall'] > 0.3)
    ].sort_values('precision', ascending=False)
    
    if len(promising) > 0:
        print(promising[['horizon', 'threshold', 'accuracy', 'precision', 'recall', 'roc_auc']].to_string(index=False))
    else:
        print("No models meet the criteria. Consider adjusting thresholds.")
        # Show best available
        print("\nBest available by precision:")
        print(results_df.nlargest(5, 'precision')[
            ['horizon', 'threshold', 'accuracy', 'precision', 'recall', 'roc_auc']
        ].to_string(index=False))


def main():
    print("="*80)
    print("MULTI-HORIZON PRICE MOVEMENT PREDICTION - MODEL TRAINING")
    print("="*80)
    print(f"Started at: {datetime.now()}")
    
    # Load data
    print("\n📂 Loading processed data...")
    df = load_processed_data()
    
    # Get feature columns
    feature_cols = get_feature_columns(df)
    print(f"Features: {len(feature_cols)}")
    
    # Check for any remaining NaN
    nan_cols = df[feature_cols].columns[df[feature_cols].isna().any()].tolist()
    if nan_cols:
        print(f"Warning: {len(nan_cols)} columns have NaN values. Filling with 0.")
        df[feature_cols] = df[feature_cols].fillna(0)
    
    # Replace inf values
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], 0)
    
    # Train all models
    results_df = train_all_models(df, feature_cols)
    
    # Analyze results
    analyze_results(results_df)
    
    print(f"\n✅ Training complete at: {datetime.now()}")
    print(f"Models saved to: {config.MODELS_DIR}/")
    print(f"Results saved to: {config.RESULTS_DIR}/training_results.csv")


if __name__ == "__main__":
    main()
