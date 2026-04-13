"""
Train XGBoost Model for BTC TP/SL Direction Prediction

Trains a binary classifier to predict whether LONG or SHORT direction
will hit TP/SL first (1% TP vs 0.5% SL).

Uses 80/20 time-based train/test split with StandardScaler.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier

USE_SKLEARN_GBM = True
print("Using scikit-learn GradientBoostingClassifier")

# Constants
TRAIN_SPLIT = 0.8
RANDOM_STATE = 42


def prepare_data(df):
    """Prepare features and labels, split into train/test"""
    print("\nPreparing data...")

    # Separate features and label
    # Exclude original OHLCV columns and label
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'trades', 'label']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols]
    y = df['label']

    print(f"Features: {len(feature_cols)} columns")
    print(f"Total samples: {len(X):,}")
    print(f"  LONG (1):  {(y == 1).sum():,}")
    print(f"  SHORT (0): {(y == 0).sum():,}")

    # Calculate class imbalance
    class_counts = y.value_counts()
    majority_count = class_counts.max()
    minority_count = class_counts.min()
    imbalance_ratio = majority_count / (majority_count + minority_count)

    print(f"\nClass imbalance: {imbalance_ratio*100:.1f}% majority class")

    # Calculate scale_pos_weight for XGBoost
    # scale_pos_weight = (negative class count) / (positive class count)
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

    # Time-based split (no shuffle)
    split_idx = int(len(X) * TRAIN_SPLIT)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    print(f"\nTrain/Test Split (80/20):")
    print(f"  Train: {len(X_train):,} samples")
    print(f"  Test:  {len(X_test):,} samples")

    return X_train, X_test, y_train, y_test, feature_cols, scale_pos_weight


def scale_features(X_train, X_test):
    """Apply StandardScaler to features"""
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrames to preserve column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    print("✓ Scaling complete")
    return X_train_scaled, X_test_scaled, scaler


def train_model(X_train, y_train, X_test, y_test, scale_pos_weight):
    """Train Gradient Boosting classifier"""
    print("\nTraining Gradient Boosting model...")

    # Calculate sample_weight to handle class imbalance
    sample_weight = np.where(y_train == 1, scale_pos_weight, 1.0)

    params = {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': RANDOM_STATE,
        'verbose': 1,
        'subsample': 0.8,
        'max_features': 'sqrt'
    }
    print(f"Parameters: {params}")
    print(f"Using sample_weight to handle class imbalance (scale={scale_pos_weight:.2f})")

    model = GradientBoostingClassifier(**params)
    model.fit(X_train, y_train, sample_weight=sample_weight)

    print("\n✓ Training complete")
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Quick evaluation on train and test sets"""
    print("\nQuick Evaluation:")

    # Train accuracy
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"  Train Accuracy: {train_acc:.4f}")

    # Test accuracy
    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"  Test Accuracy:  {test_acc:.4f}")

    # Check for overfitting
    if train_acc - test_acc > 0.05:
        print(f"  ⚠️  Warning: Possible overfitting (train-test gap: {(train_acc - test_acc)*100:.1f}%)")

    return y_test_pred


def save_model_artifacts(model, scaler, feature_cols):
    """Save model, scaler, features, and feature importance"""
    print("\nSaving model artifacts...")

    # Create ml_models directory if it doesn't exist
    models_dir = Path(__file__).parent / 'ml_models'
    models_dir.mkdir(exist_ok=True)

    # Save model
    model_path = models_dir / 'btc_tpsl_model.pkl'
    joblib.dump(model, model_path)
    print(f"  ✓ Model saved: {model_path}")

    # Save scaler
    scaler_path = models_dir / 'btc_tpsl_scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"  ✓ Scaler saved: {scaler_path}")

    # Save feature names
    features_path = models_dir / 'btc_tpsl_features.json'
    with open(features_path, 'w') as f:
        json.dump(list(feature_cols), f, indent=2)
    print(f"  ✓ Features saved: {features_path}")

    # Save feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    importance_path = models_dir / 'btc_tpsl_feature_importance.csv'
    importance_df.to_csv(importance_path, index=False)
    print(f"  ✓ Feature importance saved: {importance_path}")

    # Show top 10 features
    print("\nTop 10 Most Important Features:")
    for i, row in importance_df.head(10).iterrows():
        print(f"  {i+1:2d}. {row['feature']:30s} {row['importance']:.6f}")


def main():
    # File paths
    data_dir = Path(__file__).parent / 'data'
    input_file = data_dir / 'btc_1m_tpsl_features.parquet'

    print("="*60)
    print("XGBoost Training for BTC TP/SL Direction Prediction")
    print("="*60)

    # Load features
    print(f"\nLoading features from: {input_file}")
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df):,} rows with {len(df.columns)} columns")

    # Prepare data
    X_train, X_test, y_train, y_test, feature_cols, scale_pos_weight = prepare_data(df)

    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Train model
    model = train_model(X_train_scaled, y_train, X_test_scaled, y_test, scale_pos_weight)

    # Quick evaluation
    y_test_pred = evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test)

    # Save artifacts
    save_model_artifacts(model, scaler, feature_cols)

    print("\n" + "="*60)
    print("✓ Training Complete!")
    print("="*60)
    print("\nNext step: Run 'python3 evaluate_tpsl_model.py' for detailed evaluation")


if __name__ == '__main__':
    main()
