"""
Train Multiple Horizon Models (2h, 4h, 6h)

Creates separate models for each time horizon to capture different patterns.
Models can be used individually or in ensemble for better signal filtering.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

# Constants
TRAIN_SPLIT = 0.8
RANDOM_STATE = 42
HORIZONS = ['2h', '4h', '6h']


def prepare_data_for_horizon(df, horizon):
    """
    Prepare features and labels for specific horizon
    """
    print(f"\nPreparing data for {horizon} horizon...")

    # Get label column
    label_col = f'label_{horizon}'

    # Filter to rows with labels
    df_labeled = df[df[label_col].notna()].copy()

    # Separate features and labels
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'trades',
                    'label_2h', 'label_4h', 'label_6h']
    feature_cols = [col for col in df_labeled.columns if col not in exclude_cols]

    X = df_labeled[feature_cols]
    y = df_labeled[label_col]

    print(f"  Features: {len(feature_cols)} columns")
    print(f"  Total samples: {len(X):,}")
    print(f"    LONG (1):  {(y == 1).sum():,}")
    print(f"    SHORT (0): {(y == 0).sum():,}")

    # Calculate class weights
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    print(f"  Scale_pos_weight: {scale_pos_weight:.2f}")

    # Time-based split
    split_idx = int(len(X) * TRAIN_SPLIT)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    print(f"  Train: {len(X_train):,} samples")
    print(f"  Test:  {len(X_test):,} samples")

    return X_train, X_test, y_train, y_test, feature_cols, scale_pos_weight


def train_model(X_train, y_train, scale_pos_weight, model_type='gbm'):
    """
    Train Gradient Boosting or Random Forest classifier
    """
    if model_type == 'gbm':
        print(f"\n  Training Gradient Boosting...")
        sample_weight = np.where(y_train == 1, scale_pos_weight, 1.0)
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            subsample=0.8,
            max_features='sqrt',
            verbose=0
        )
        model.fit(X_train, y_train, sample_weight=sample_weight)

    else:  # Random Forest
        print(f"\n  Training Random Forest...")
        sample_weight = np.where(y_train == 1, scale_pos_weight, 1.0)
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight='balanced',
            verbose=0
        )
        model.fit(X_train, y_train, sample_weight=sample_weight)

    return model


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Quick evaluation metrics
    """
    # Train metrics
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)

    # Test metrics
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)

    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average='weighted')

    try:
        roc_auc = roc_auc_score(y_test, y_test_proba[:, 1])
    except:
        roc_auc = 0.5

    print(f"  Train Acc: {train_acc:.4f}")
    print(f"  Test Acc:  {test_acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")

    return {
        'train_acc': float(train_acc),
        'test_acc': float(test_acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'roc_auc': float(roc_auc)
    }


def save_model_artifacts(model, scaler, feature_cols, metrics, horizon):
    """
    Save model, scaler, features, and metrics for specific horizon
    """
    models_dir = Path(__file__).parent / 'ml_models'
    models_dir.mkdir(exist_ok=True)

    # Save model
    model_path = models_dir / f'btc_{horizon}_model.pkl'
    joblib.dump(model, model_path)

    # Save scaler
    scaler_path = models_dir / f'btc_{horizon}_scaler.pkl'
    joblib.dump(scaler, scaler_path)

    # Save features
    features_path = models_dir / f'btc_{horizon}_features.json'
    with open(features_path, 'w') as f:
        json.dump(list(feature_cols), f, indent=2)

    # Save metrics
    metrics_path = models_dir / f'btc_{horizon}_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    importance_path = models_dir / f'btc_{horizon}_feature_importance.csv'
    importance_df.to_csv(importance_path, index=False)

    print(f"\n  ✓ Saved all artifacts for {horizon} horizon")

    # Show top 5 features
    print(f"\n  Top 5 features for {horizon}:")
    for i, row in importance_df.head(5).iterrows():
        print(f"    {i+1}. {row['feature'][:35]:<35} {row['importance']:.4f}")


def main():
    # File paths
    data_dir = Path(__file__).parent / 'data'
    input_file = data_dir / 'btc_5m_features_advanced.parquet'

    print("="*60)
    print("Training Multiple Horizon Models")
    print("="*60)

    # Load data
    print(f"\nLoading data from: {input_file}")
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df):,} rows with {len(df.columns)} columns")

    # Train model for each horizon
    all_metrics = {}

    for horizon in HORIZONS:
        print(f"\n{'='*60}")
        print(f"HORIZON: {horizon}")
        print(f"{'='*60}")

        # Prepare data
        X_train, X_test, y_train, y_test, feature_cols, scale_pos_weight = prepare_data_for_horizon(df, horizon)

        # Scale features
        print(f"\n  Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Convert back to DataFrame
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols)

        # Train model
        model = train_model(X_train_scaled, y_train, scale_pos_weight, model_type='gbm')

        # Evaluate
        print(f"\n  Evaluation:")
        metrics = evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test)
        all_metrics[horizon] = metrics

        # Save artifacts
        save_model_artifacts(model, scaler, feature_cols, metrics, horizon)

    # Summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")

    print(f"\n{'Horizon':<10} {'Test Acc':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'ROC-AUC'}")
    print("-" * 72)
    for horizon in HORIZONS:
        m = all_metrics[horizon]
        print(f"{horizon:<10} {m['test_acc']:<12.4f} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1']:<12.4f} {m['roc_auc']:.4f}")

    print("\n✓ All models trained successfully!")
    print("\nNext step: Run ensemble evaluation to test combined signals")


if __name__ == '__main__':
    main()
