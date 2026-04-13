"""
Train LightGBM Models with Micro-Structure Features (0.5% TP / 0.15% SL)

Combines:
- 25 existing features from V3 predictor
- 17 new micro-structure features
- LightGBM algorithm (excellent class imbalance handling)
- 0.5% TP / 0.15% SL targets
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib
import sys

print("="*80)
print("LIGHTGBM + MICRO-STRUCTURE TRAINING (0.5% TP / 0.15% SL)")
print("="*80)

BOT_DIR = Path(__file__).parent

# Load labeled data
data_file = BOT_DIR / 'data' / 'BTC_5min_with_wider_labels.parquet'
print(f"\nLoading labeled data from {data_file}...")
df = pd.read_parquet(data_file)
print(f"Loaded {len(df):,} rows")

# Add micro-structure features
print("\nAdding micro-structure features...")
sys.path.insert(0, str(BOT_DIR))
from feature_engineering_microstructure import add_microstructure_features
df = add_microstructure_features(df)

# Calculate V3 features
print("\nCalculating V3 features (25 features)...")
from btc_model_package.predictor import BTCPredictor
predictor = BTCPredictor()

def calculate_features_for_row(df, idx):
    if idx < 250:
        return None
    window = df.iloc[idx-250:idx]
    try:
        features = predictor.calculate_features(window)
        return features
    except:
        return None

# Sample every 10th row for speed
print("Extracting features (sampled every 10th row)...")
feature_list = []
valid_indices = []

for i in range(250, len(df), 10):
    features = calculate_features_for_row(df, i)

    if features is not None and isinstance(features, dict):
        # Add micro-structure features to dict
        row = df.iloc[i]
        micro_features = {
            'vwap_dist_5': row['vwap_dist_5'],
            'vwap_dist_20': row['vwap_dist_20'],
            'tick_velocity': row['tick_velocity'],
            'realized_vol_5': row['realized_vol_5'],
            'dist_to_high': row['dist_to_high'],
            'dist_to_low': row['dist_to_low'],
            'volume_deviation': row['volume_deviation'],
            'session_asia': row['session_asia'],
            'session_europe': row['session_europe'],
            'session_us': row['session_us'],
            'momentum_1bar': row['momentum_1bar'],
            'momentum_3bar': row['momentum_3bar'],
            'momentum_5bar': row['momentum_5bar'],
            'spread_proxy': row['spread_proxy'],
            'price_position': row['price_position'],
            'volume_trend': row['volume_trend']
        }

        # Combine
        features.update(micro_features)
        feature_list.append(features)
        valid_indices.append(i)

    if len(feature_list) % 5000 == 0 and len(feature_list) > 0:
        print(f"  Progress: {len(feature_list):,} rows...")

if len(feature_list) == 0:
    print("ERROR: No features calculated!")
    sys.exit(1)

X = pd.DataFrame(feature_list)

# Drop rows with NaN
X = X.fillna(0)  # Fill NaN with 0 for micro-structure features

feature_names = X.columns.tolist()
print(f"\n✓ Calculated {len(X):,} rows × {len(feature_names)} features")
print(f"   (25 V3 features + 16 micro-structure features = 41 total)")

# Get labels
df_valid = df.iloc[valid_indices].copy()

# Remove NaN labels
valid_mask = pd.Series(True, index=df_valid.index)
for horizon in ['30min', '1h', '2h']:
    valid_mask &= df_valid[f'label_{horizon}'].notna()

df_valid = df_valid[valid_mask]
X = X[valid_mask.values]
print(f"✓ Clean dataset: {len(X):,} rows")

# Train/test split
split_date = '2024-01-01'
train_mask = df_valid.index < split_date
test_mask = df_valid.index >= split_date

X_train = X[train_mask]
X_test = X[test_mask]

print(f"\nTrain set: {len(X_train):,} rows")
print(f"Test set: {len(X_test):,} rows")

# Train models
horizons = ['30min', '1h', '2h']
models = {}
scalers = {}
label_encoders = {}
results = {}

for horizon in horizons:
    print(f"\n{'='*80}")
    print(f"TRAINING {horizon.upper()} MODEL")
    print('='*80)

    label_col = f'label_{horizon}'
    y_train = df_valid.loc[train_mask, label_col]
    y_test = df_valid.loc[test_mask, label_col]

    # Encode labels (XGBoost needs numeric)
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    # Label distribution
    train_dist = y_train.value_counts()
    print(f"\nTrain label distribution:")
    for label in ['LONG', 'SHORT', 'NEUTRAL']:
        count = train_dist.get(label, 0)
        pct = count / len(y_train) * 100
        print(f"  {label:8}: {count:,} ({pct:.1f}%)")

    # Calculate scale_pos_weight for imbalance
    class_counts = train_dist.to_dict()
    neutral_count = class_counts.get('NEUTRAL', 1)
    long_count = class_counts.get('LONG', 1)
    short_count = class_counts.get('SHORT', 1)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train LightGBM with class weights
    print(f"\nTraining LGBMClassifier...")
    model = LGBMClassifier(
        n_estimators=200,          # More trees
        max_depth=8,               # Moderate depth
        learning_rate=0.05,        # Conservative LR
        min_child_samples=50,      # Regularization
        subsample=0.8,             # Row sampling
        colsample_bytree=0.8,      # Column sampling
        reg_alpha=0.5,             # L1 regularization
        reg_lambda=1.0,            # L2 regularization
        class_weight='balanced',   # Handle class imbalance
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    model.fit(X_train_scaled, y_train_encoded)

    # Evaluate
    y_pred_encoded = model.predict(X_test_scaled)
    y_pred = le.inverse_transform(y_pred_encoded)
    y_proba = model.predict_proba(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✓ Test Accuracy: {accuracy:.1%}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Feature importance (top 10)
    importances = model.feature_importances_
    top_indices = np.argsort(importances)[-10:][::-1]
    print("\nTop 10 Features:")
    for idx in top_indices:
        print(f"  {feature_names[idx]:25}: {importances[idx]:.4f}")

    # Save
    models[horizon] = model
    scalers[horizon] = scaler
    label_encoders[horizon] = le
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
    model_file = models_dir / f'btc_model_{horizon}_lightgbm.pkl'
    scaler_file = models_dir / f'btc_scaler_{horizon}_lightgbm.pkl'
    encoder_file = models_dir / f'btc_encoder_{horizon}_lightgbm.pkl'

    joblib.dump(models[horizon], model_file)
    joblib.dump(scalers[horizon], scaler_file)
    joblib.dump(label_encoders[horizon], encoder_file)

    print(f"✓ Saved {horizon}: {model_file}")

feature_file = models_dir / 'btc_features_lightgbm.pkl'
joblib.dump(feature_names, feature_file)
print(f"✓ Saved features: {feature_file}")

print(f"\n{'='*80}")
print("TRAINING COMPLETE")
print('='*80)

print("\nModel Performance Summary:")
for horizon in horizons:
    acc = results[horizon]['accuracy']
    print(f"  {horizon:6}: {acc:.1%}")

print(f"\nModels saved to: {models_dir}/")
print("="*80)
