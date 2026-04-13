"""
Train V3 Models with Expanded Dataset

Uses the same regularization strategy as V2 but with 4 years of data.
Expected improvements: lower overfitting, better generalization, more robust metrics.
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


def prepare_data_for_horizon(df, horizon, selected_features):
    """Prepare features and labels for specific horizon"""
    print(f"\nPreparing data for {horizon} horizon...")

    label_col = f'label_{horizon}'
    df_labeled = df[df[label_col].notna()].copy()

    X = df_labeled[selected_features]
    y = df_labeled[label_col]

    print(f"  Features: {len(selected_features)} columns")
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
    """Train GBM with strong regularization (same as V2)"""
    print(f"\n  Training Regularized Gradient Boosting...")

    sample_weight = np.where(y_train == 1, scale_pos_weight, 1.0)

    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.15,
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.8,
        max_features='sqrt',
        random_state=RANDOM_STATE,
        verbose=0,
        validation_fraction=0.15,
        n_iter_no_change=10,
        tol=1e-4
    )

    model.fit(X_train, y_train, sample_weight=sample_weight)

    print(f"    ✓ Training stopped at iteration: {model.n_estimators_}")

    return model


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Comprehensive evaluation"""
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

    overfit_gap = train_acc - test_acc

    print(f"\n  Evaluation:")
    print(f"    Train Acc:     {train_acc:.4f}")
    print(f"    Test Acc:      {test_acc:.4f}")
    print(f"    Overfit Gap:   {overfit_gap:.4f}")
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
    """Save V3 model artifacts"""
    models_dir = Path(__file__).parent / 'ml_models'
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / f'btc_{horizon}_model_v3.pkl'
    joblib.dump(model, model_path)

    scaler_path = models_dir / f'btc_{horizon}_scaler_v3.pkl'
    joblib.dump(scaler, scaler_path)

    features_path = models_dir / f'btc_{horizon}_features_v3.json'
    with open(features_path, 'w') as f:
        json.dump(list(selected_features), f, indent=2)

    metrics_path = models_dir / f'btc_{horizon}_metrics_v3.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    importance_df = pd.DataFrame({
        'feature': selected_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    importance_path = models_dir / f'btc_{horizon}_feature_importance_v3.csv'
    importance_df.to_csv(importance_path, index=False)

    print(f"\n  ✓ Saved V3 model for {horizon}")

    print(f"\n  Top 10 features:")
    for i, row in importance_df.head(10).iterrows():
        print(f"    {i+1:2d}. {row['feature'][:35]:<35} {row['importance']:.4f}")


def main():
    data_dir = Path(__file__).parent / 'data'
    models_dir = Path(__file__).parent / 'ml_models'

    print("="*60)
    print("Training V3 Models with Expanded Dataset")
    print("="*60)

    # Load selected features
    print("\nLoading selected features...")
    with open(models_dir / 'selected_features.json', 'r') as f:
        selected_features = json.load(f)
    print(f"  Target: {len(selected_features)} features")

    # Load V3 data
    input_file = data_dir / 'btc_5m_v3_features.parquet'
    print(f"\nLoading data from: {input_file}")
    df = pd.read_parquet(input_file)
    print(f"  Loaded {len(df):,} rows")

    # Filter to only available features
    available_features = [f for f in selected_features if f in df.columns]
    missing_features = [f for f in selected_features if f not in df.columns]

    if missing_features:
        print(f"  ⚠ Using {len(available_features)}/{len(selected_features)} features")
        print(f"  Missing: {missing_features}")

    selected_features = available_features
    print(f"  Using: {len(selected_features)} features")

    # Train model for each horizon
    all_metrics = {}

    for horizon in HORIZONS:
        print(f"\n{'='*60}")
        print(f"HORIZON: {horizon}")
        print(f"{'='*60}")

        X_train, X_test, y_train, y_test, scale_pos_weight = prepare_data_for_horizon(
            df, horizon, selected_features
        )

        # Scale features
        print(f"\n  Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train_scaled = pd.DataFrame(X_train_scaled, columns=selected_features)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=selected_features)

        # Train
        model = train_regularized_model(X_train_scaled, y_train, scale_pos_weight)

        # Evaluate
        metrics = evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test)
        all_metrics[horizon] = metrics

        # Save
        save_model_artifacts(model, scaler, selected_features, metrics, horizon)

    # Comparison: V2 vs V3
    print(f"\n{'='*60}")
    print("COMPARISON: V2 (12mo) vs V3 (4yr)")
    print(f"{'='*60}")

    print(f"\n{'Horizon':<10} {'Version':<10} {'Train Acc':<12} {'Test Acc':<12} {'Gap':<12} {'ROC-AUC'}")
    print("-" * 72)

    for horizon in HORIZONS:
        # Load V2 metrics
        try:
            with open(models_dir / f'btc_{horizon}_metrics_v2.json', 'r') as f:
                v2_metrics = json.load(f)

            print(f"{horizon:<10} {'V2 (12mo)':<10} {v2_metrics['train_acc']:<12.4f} {v2_metrics['test_acc']:<12.4f} {v2_metrics['overfit_gap']:<12.4f} {v2_metrics['roc_auc']:.4f}")
        except:
            pass

        # V3 metrics
        v3_metrics = all_metrics[horizon]
        print(f"{horizon:<10} {'V3 (4yr)':<10} {v3_metrics['train_acc']:<12.4f} {v3_metrics['test_acc']:<12.4f} {v3_metrics['overfit_gap']:<12.4f} {v3_metrics['roc_auc']:.4f}")
        print()

    print("✓ All V3 models trained successfully!")
    print("\nKey improvements from V3:")
    print("  - 4 years of data (vs 12 months)")
    print("  - More diverse market conditions")
    print("  - Better statistical significance")
    print("  - Expected: lower overfitting, better generalization")


if __name__ == '__main__':
    main()
