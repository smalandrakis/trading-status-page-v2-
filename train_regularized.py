"""
Retrain Models with Regularization and Feature Selection

Improvements to reduce overfitting:
1. Use only 22 selected features (vs 93)
2. Simpler model: 100 trees (vs 200), depth 4 (vs 6)
3. Higher learning rate: 0.15 (vs 0.1) - faster convergence
4. Regularization: min_samples_split=20, min_samples_leaf=10
5. Early stopping based on validation set
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

# Constants
TRAIN_SPLIT = 0.8
RANDOM_STATE = 42
HORIZONS = ['2h', '4h', '6h']


def load_selected_features():
    """Load the selected feature list"""
    models_dir = Path(__file__).parent / 'ml_models'
    with open(models_dir / 'selected_features.json', 'r') as f:
        features = json.load(f)
    return features


def prepare_data_for_horizon(df, horizon, selected_features):
    """
    Prepare features and labels using only selected features
    """
    print(f"\nPreparing data for {horizon} horizon...")

    # Get label column
    label_col = f'label_{horizon}'

    # Filter to rows with labels
    df_labeled = df[df[label_col].notna()].copy()

    # Use only selected features
    X = df_labeled[selected_features]
    y = df_labeled[label_col]

    print(f"  Features: {len(selected_features)} columns (selected)")
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

    return X_train, X_test, y_train, y_test, scale_pos_weight


def train_regularized_model(X_train, y_train, scale_pos_weight):
    """
    Train GBM with strong regularization
    """
    print(f"\n  Training Regularized Gradient Boosting...")

    sample_weight = np.where(y_train == 1, scale_pos_weight, 1.0)

    # REGULARIZED parameters
    model = GradientBoostingClassifier(
        n_estimators=100,              # Fewer trees (was 200)
        max_depth=4,                   # Shallower (was 6)
        learning_rate=0.15,            # Higher LR (was 0.1)
        min_samples_split=20,          # Need 20+ samples to split (regularization)
        min_samples_leaf=10,           # Need 10+ samples in leaf (regularization)
        subsample=0.8,                 # Use 80% of data per tree
        max_features='sqrt',           # Sqrt of features per split
        random_state=RANDOM_STATE,
        verbose=0,
        validation_fraction=0.15,      # Use 15% for early stopping
        n_iter_no_change=10,           # Stop if no improvement for 10 iterations
        tol=1e-4
    )

    print(f"    n_estimators: 100 (was 200)")
    print(f"    max_depth: 4 (was 6)")
    print(f"    learning_rate: 0.15 (was 0.1)")
    print(f"    min_samples_split: 20 (regularization)")
    print(f"    min_samples_leaf: 10 (regularization)")

    model.fit(X_train, y_train, sample_weight=sample_weight)

    print(f"    ✓ Training stopped at iteration: {model.n_estimators_}")

    return model


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Comprehensive evaluation
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

    # Calculate overfitting gap
    overfit_gap = train_acc - test_acc

    print(f"\n  Evaluation:")
    print(f"    Train Acc:     {train_acc:.4f}")
    print(f"    Test Acc:      {test_acc:.4f}")
    print(f"    Overfit Gap:   {overfit_gap:.4f} (lower is better!)")
    print(f"    Precision:     {prec:.4f}")
    print(f"    Recall:        {rec:.4f}")
    print(f"    F1:            {f1:.4f}")
    print(f"    ROC-AUC:       {roc_auc:.4f}")

    if overfit_gap < 0.1:
        print(f"    ✓ Good! Overfitting gap < 10%")
    elif overfit_gap < 0.2:
        print(f"    ⚠ Moderate overfitting (10-20% gap)")
    else:
        print(f"    ⚠⚠ High overfitting (>20% gap)")

    return {
        'train_acc': float(train_acc),
        'test_acc': float(test_acc),
        'overfit_gap': float(overfit_gap),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'roc_auc': float(roc_auc)
    }


def save_model_artifacts(model, scaler, selected_features, metrics, horizon):
    """
    Save regularized model artifacts
    """
    models_dir = Path(__file__).parent / 'ml_models'
    models_dir.mkdir(exist_ok=True)

    # Save model (overwrite previous)
    model_path = models_dir / f'btc_{horizon}_model_v2.pkl'
    joblib.dump(model, model_path)

    # Save scaler
    scaler_path = models_dir / f'btc_{horizon}_scaler_v2.pkl'
    joblib.dump(scaler, scaler_path)

    # Save features (should match selected_features.json but save for consistency)
    features_path = models_dir / f'btc_{horizon}_features_v2.json'
    with open(features_path, 'w') as f:
        json.dump(list(selected_features), f, indent=2)

    # Save metrics
    metrics_path = models_dir / f'btc_{horizon}_metrics_v2.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save feature importance
    importance_df = pd.DataFrame({
        'feature': selected_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    importance_path = models_dir / f'btc_{horizon}_feature_importance_v2.csv'
    importance_df.to_csv(importance_path, index=False)

    print(f"\n  ✓ Saved v2 (regularized) model for {horizon}")

    # Show top 10 features
    print(f"\n  Top 10 features:")
    for i, row in importance_df.head(10).iterrows():
        print(f"    {i+1:2d}. {row['feature'][:35]:<35} {row['importance']:.4f}")


def main():
    data_dir = Path(__file__).parent / 'data'

    print("="*60)
    print("Retraining with Regularization & Feature Selection")
    print("="*60)

    # Load selected features
    print("\nLoading selected features...")
    selected_features = load_selected_features()
    print(f"  Using {len(selected_features)} selected features")

    # Load data
    input_file = data_dir / 'btc_5m_features_advanced.parquet'
    print(f"\nLoading data from: {input_file}")
    df = pd.read_parquet(input_file)
    print(f"  Loaded {len(df):,} rows")

    # Train model for each horizon
    all_metrics = {}

    for horizon in HORIZONS:
        print(f"\n{'='*60}")
        print(f"HORIZON: {horizon}")
        print(f"{'='*60}")

        # Prepare data with selected features
        X_train, X_test, y_train, y_test, scale_pos_weight = prepare_data_for_horizon(
            df, horizon, selected_features
        )

        # Scale features
        print(f"\n  Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Convert back to DataFrame
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=selected_features)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=selected_features)

        # Train regularized model
        model = train_regularized_model(X_train_scaled, y_train, scale_pos_weight)

        # Evaluate
        metrics = evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test)
        all_metrics[horizon] = metrics

        # Save artifacts
        save_model_artifacts(model, scaler, selected_features, metrics, horizon)

    # Summary comparison
    print(f"\n{'='*60}")
    print("COMPARISON: V1 (Original) vs V2 (Regularized)")
    print(f"{'='*60}")

    print(f"\n{'Horizon':<10} {'Model':<12} {'Train Acc':<12} {'Test Acc':<12} {'Gap':<12} {'ROC-AUC'}")
    print("-" * 72)

    for horizon in HORIZONS:
        # V1 metrics (load from file)
        models_dir = Path(__file__).parent / 'ml_models'
        with open(models_dir / f'btc_{horizon}_metrics.json', 'r') as f:
            v1_metrics = json.load(f)

        v2_metrics = all_metrics[horizon]

        # V1
        v1_gap = v1_metrics['train_acc'] - v1_metrics['test_acc']
        print(f"{horizon:<10} {'V1':<12} {v1_metrics['train_acc']:<12.4f} {v1_metrics['test_acc']:<12.4f} {v1_gap:<12.4f} {v1_metrics['roc_auc']:.4f}")

        # V2
        print(f"{horizon:<10} {'V2':<12} {v2_metrics['train_acc']:<12.4f} {v2_metrics['test_acc']:<12.4f} {v2_metrics['overfit_gap']:<12.4f} {v2_metrics['roc_auc']:.4f}")
        print()

    print("✓ All regularized models trained successfully!")
    print("\nKey improvements:")
    print("  - Reduced features: 93 → 22 (76% reduction)")
    print("  - Simpler models: 100 trees, depth 4 (vs 200 trees, depth 6)")
    print("  - Regularization: min_samples_split=20, min_samples_leaf=10")
    print("  - Early stopping enabled")


if __name__ == '__main__':
    main()
