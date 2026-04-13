"""
Train Micro-Movement Models (0.3% TP / 0.1% SL)

Trains 3-horizon ensemble optimized for tight targets:
- 30min, 1h, 2h models
- Strong regularization to handle noise
- Time-based train/test split (2022-2024 train, 2024+ test)
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib
import sys

print("="*80)
print("MICRO-MOVEMENT MODEL TRAINING")
print("="*80)

BOT_DIR = Path(__file__).parent

# Load labeled data
data_file = BOT_DIR / 'data' / 'BTC_5min_with_micro_labels.parquet'
print(f"\nLoading labeled data from {data_file}...")
df = pd.read_parquet(data_file)
print(f"Loaded {len(df):,} rows")

# Import feature calculation from existing predictor
sys.path.insert(0, str(BOT_DIR))
from btc_model_package.predictor import BTCPredictor

# Calculate features
print("\nCalculating 22 features...")
predictor = BTCPredictor()

def calculate_features_for_row(df, idx):
    """Calculate features for a single row using 250-bar window"""
    if idx < 250:
        return None
    window = df.iloc[idx-250:idx]
    try:
        features = predictor.calculate_features(window)
        return features
    except Exception as e:
        if idx == 250:  # Print first error only
            print(f"ERROR calculating features: {e}")
            import traceback
            traceback.print_exc()
        return None

# Calculate features for all rows
print("Extracting features from FULL dataset (this will take 15-20 mins)...")
feature_list = []
valid_indices = []

# NO SAMPLING - process all rows for production model
for i in range(250, len(df)):
    features = calculate_features_for_row(df, i)

    if features is not None and isinstance(features, dict):
        feature_list.append(features)
        valid_indices.append(i)

    if (len(feature_list)) % 10000 == 0 and len(feature_list) > 0:
        print(f"  Progress: {len(feature_list):,} rows extracted...")

# Create feature DataFrame
if len(feature_list) == 0:
    print("ERROR: No features calculated!")
    sys.exit(1)

X = pd.DataFrame(feature_list)
feature_names = X.columns.tolist()
print(f"\n✓ Calculated {len(X):,} rows × {len(feature_names)} features")

# Get labels for valid indices
df_valid = df.iloc[valid_indices].copy()

# Train/test split (time-based)
split_date = '2024-01-01'
train_mask = df_valid.index < split_date
test_mask = df_valid.index >= split_date

X_train = X[train_mask]
X_test = X[test_mask]

print(f"\nTrain set: {len(X_train):,} rows (pre-2024)")
print(f"Test set: {len(X_test):,} rows (2024+)")

# Train models for each horizon
horizons = ['30min', '1h', '2h']
models = {}
scalers = {}
results = {}

for horizon in horizons:
    print(f"\n{'='*80}")
    print(f"TRAINING {horizon.upper()} MODEL")
    print('='*80)

    # Get labels
    label_col = f'label_{horizon}'
    y_train = df_valid.loc[train_mask, label_col]
    y_test = df_valid.loc[test_mask, label_col]

    # Label distribution
    train_dist = y_train.value_counts()
    print(f"\nTrain label distribution:")
    for label in ['LONG', 'SHORT', 'NEUTRAL']:
        count = train_dist.get(label, 0)
        pct = count / len(y_train) * 100
        print(f"  {label:8}: {count:,} ({pct:.1f}%)")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model with strong regularization for micro-movements
    print(f"\nTraining HistGradientBoostingClassifier...")
    model = HistGradientBoostingClassifier(
        max_iter=200,          # More iterations for complex patterns
        learning_rate=0.05,    # Lower LR to prevent overfitting to noise
        max_depth=8,           # Deeper for micro-interactions
        min_samples_leaf=50,   # Stronger noise reduction
        l2_regularization=1.0, # Higher regularization
        class_weight='balanced',
        random_state=42,
        verbose=0
    )

    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✓ Test Accuracy: {accuracy:.1%}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Calculate ROC-AUC for LONG vs others and SHORT vs others
    try:
        from sklearn.preprocessing import label_binarize
        y_test_bin = label_binarize(y_test, classes=['LONG', 'NEUTRAL', 'SHORT'])
        if y_proba.shape[1] == 3:
            roc_auc = roc_auc_score(y_test_bin, y_proba, average='macro', multi_class='ovr')
            print(f"ROC-AUC (macro): {roc_auc:.3f}")
    except:
        pass

    # Save
    models[horizon] = model
    scalers[horizon] = scaler
    results[horizon] = {
        'accuracy': accuracy,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_proba': y_proba
    }

# Save models
models_dir = BOT_DIR / 'models'
models_dir.mkdir(exist_ok=True)

print(f"\n{'='*80}")
print("SAVING MODELS")
print('='*80)

for horizon in horizons:
    model_file = models_dir / f'btc_model_{horizon}_micro.pkl'
    scaler_file = models_dir / f'btc_scaler_{horizon}_micro.pkl'

    joblib.dump(models[horizon], model_file)
    joblib.dump(scalers[horizon], scaler_file)

    print(f"✓ Saved {horizon} model: {model_file}")
    print(f"✓ Saved {horizon} scaler: {scaler_file}")

# Save feature names
feature_file = models_dir / 'btc_features_micro.pkl'
joblib.dump(feature_names, feature_file)
print(f"✓ Saved feature names: {feature_file}")

print(f"\n{'='*80}")
print("TRAINING COMPLETE")
print('='*80)

print("\nModel Performance Summary:")
for horizon in horizons:
    acc = results[horizon]['accuracy']
    print(f"  {horizon:6}: {acc:.1%} test accuracy")

print(f"\nModels saved to: {models_dir}/")
print("="*80)
