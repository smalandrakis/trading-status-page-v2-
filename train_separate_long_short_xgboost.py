"""
Train Separate LONG and SHORT Models with XGBoost

Strategy:
- LONG model: Train on bull periods (2024-2025 when BTC 42K→108K)
- SHORT model: Train on bear periods (2022-2023 when BTC 69K→16K)
- Add temporal features: hour, day of week, previous close patterns
- Use XGBoost for better class imbalance handling
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import sys

print("="*80)
print("SEPARATE LONG/SHORT MODELS WITH XGBOOST + TEMPORAL FEATURES")
print("="*80)

BOT_DIR = Path(__file__).parent

# Load labeled data
data_file = BOT_DIR / 'data' / 'BTC_5min_with_wider_labels.parquet'
print(f"\nLoading labeled data from {data_file}...")
df = pd.read_parquet(data_file)
print(f"Loaded {len(df):,} rows")

# Add temporal features
print("\nAdding temporal features...")
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek  # 0=Monday, 6=Sunday
df['prev_close_1'] = df['close'].shift(1)
df['prev_close_5'] = df['close'].shift(5)
df['prev_close_20'] = df['close'].shift(20)
df['price_change_1'] = df['close'].pct_change(1)
df['price_change_5'] = df['close'].pct_change(5)
df['price_change_20'] = df['close'].pct_change(20)

# Calculate V3 features
print("\nCalculating V3 features...")
sys.path.insert(0, str(BOT_DIR))
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

# Sample every 10th row
print("Extracting features (sampled every 10th row)...")
feature_list = []
valid_indices = []

for i in range(250, len(df), 10):
    features = calculate_features_for_row(df, i)

    if features is not None and isinstance(features, dict):
        # Add temporal features
        row = df.iloc[i]
        temporal_features = {
            'hour': row['hour'],
            'day_of_week': row['day_of_week'],
            'prev_close_1': row['prev_close_1'],
            'prev_close_5': row['prev_close_5'],
            'prev_close_20': row['prev_close_20'],
            'price_change_1': row['price_change_1'],
            'price_change_5': row['price_change_5'],
            'price_change_20': row['price_change_20']
        }

        features.update(temporal_features)
        feature_list.append(features)
        valid_indices.append(i)

    if len(feature_list) % 5000 == 0 and len(feature_list) > 0:
        print(f"  Progress: {len(feature_list):,} rows...")

X = pd.DataFrame(feature_list)
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

feature_names = X.columns.tolist()
print(f"\n✓ Calculated {len(X):,} rows × {len(feature_names)} features")
print(f"   (25 V3 + 8 temporal = 33 total)")

# Get labels
df_valid = df.iloc[valid_indices].copy()

# Remove NaN labels
valid_mask = pd.Series(True, index=df_valid.index)
for horizon in ['30min', '1h', '2h']:
    valid_mask &= df_valid[f'label_{horizon}'].notna()

df_valid = df_valid[valid_mask]
X = X[valid_mask.values]
print(f"✓ Clean dataset: {len(X):,} rows")

# Define training periods
# LONG model: Bull market (2024-2025)
# SHORT model: Bear market (2022-2023) + All data
print("\n" + "="*80)
print("TRAINING STRATEGY")
print("="*80)
print("\nLONG Model: Train on bull period (2024-01-01 to 2025-12-31)")
print("  - BTC went from 42K → 108K (+157%)")
print("  - More LONG signals should be present")
print("\nSHORT Model: Train on full dataset (2022-2026)")
print("  - Already has good SHORT signal coverage")
print("="*80)

# Horizon to focus on (2h has best balance)
horizon = '2h'
label_col = f'label_{horizon}'
y_all = df_valid[label_col]

# Train/test split
test_start = '2025-07-01'  # Test on last 6 months of 2025
train_mask = df_valid.index < test_start
test_mask = df_valid.index >= test_start

X_test = X[test_mask]
y_test = y_all[test_mask]

print(f"\nTest set (2025-07+): {len(X_test):,} rows")

# LONG MODEL - Train on bull period only
print("\n" + "="*80)
print("TRAINING LONG MODEL (2h horizon)")
print("="*80)

bull_start = '2024-01-01'
bull_end = '2025-07-01'
long_mask = (df_valid.index >= bull_start) & (df_valid.index < bull_end)

X_train_long = X[long_mask]
y_train_long = y_all[long_mask]

# For LONG model, convert to binary: LONG vs NOT_LONG
y_train_long_binary = (y_train_long == 'LONG').astype(int)
y_test_long_binary = (y_test == 'LONG').astype(int)

print(f"\nTrain period: {bull_start} to {bull_end}")
print(f"Train set: {len(X_train_long):,} rows")
print(f"\nLabel distribution:")
long_count = (y_train_long == 'LONG').sum()
not_long_count = len(y_train_long) - long_count
print(f"  LONG:     {long_count:,} ({long_count/len(y_train_long)*100:.1f}%)")
print(f"  NOT_LONG: {not_long_count:,} ({not_long_count/len(y_train_long)*100:.1f}%)")

# Scale
scaler_long = StandardScaler()
X_train_long_scaled = scaler_long.fit_transform(X_train_long)
X_test_long_scaled = scaler_long.transform(X_test)

# Train XGBoost LONG model
print(f"\nTraining XGBoost LONG model...")
scale_pos_weight = not_long_count / long_count if long_count > 0 else 1.0
print(f"  scale_pos_weight: {scale_pos_weight:.2f}")

model_long = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    min_child_weight=30,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=1.0,
    reg_alpha=0.5,
    reg_lambda=1.0,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    tree_method='hist',
    eval_metric='logloss'
)

model_long.fit(X_train_long_scaled, y_train_long_binary, verbose=False)

# Evaluate LONG model
y_pred_long = model_long.predict(X_test_long_scaled)
y_proba_long = model_long.predict_proba(X_test_long_scaled)

accuracy_long = accuracy_score(y_test_long_binary, y_pred_long)
print(f"\n✓ LONG Model Test Accuracy: {accuracy_long:.1%}")

print("\nClassification Report:")
print(classification_report(y_test_long_binary, y_pred_long,
                          target_names=['NOT_LONG', 'LONG'], zero_division=0))

# Feature importance
importances_long = model_long.feature_importances_
top_indices = np.argsort(importances_long)[-10:][::-1]
print("\nTop 10 Features for LONG:")
for idx in top_indices:
    print(f"  {feature_names[idx]:25}: {importances_long[idx]:.4f}")

# SHORT MODEL - Train on full dataset
print("\n" + "="*80)
print("TRAINING SHORT MODEL (2h horizon)")
print("="*80)

X_train_short = X[train_mask]
y_train_short = y_all[train_mask]

# For SHORT model, convert to binary: SHORT vs NOT_SHORT
y_train_short_binary = (y_train_short == 'SHORT').astype(int)
y_test_short_binary = (y_test == 'SHORT').astype(int)

print(f"\nTrain period: Full dataset up to {test_start}")
print(f"Train set: {len(X_train_short):,} rows")
print(f"\nLabel distribution:")
short_count = (y_train_short == 'SHORT').sum()
not_short_count = len(y_train_short) - short_count
print(f"  SHORT:     {short_count:,} ({short_count/len(y_train_short)*100:.1f}%)")
print(f"  NOT_SHORT: {not_short_count:,} ({not_short_count/len(y_train_short)*100:.1f}%)")

# Scale
scaler_short = StandardScaler()
X_train_short_scaled = scaler_short.fit_transform(X_train_short)
X_test_short_scaled = scaler_short.transform(X_test)

# Train XGBoost SHORT model
print(f"\nTraining XGBoost SHORT model...")
scale_pos_weight = not_short_count / short_count if short_count > 0 else 1.0
print(f"  scale_pos_weight: {scale_pos_weight:.2f}")

model_short = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    min_child_weight=30,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=1.0,
    reg_alpha=0.5,
    reg_lambda=1.0,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    tree_method='hist',
    eval_metric='logloss'
)

model_short.fit(X_train_short_scaled, y_train_short_binary, verbose=False)

# Evaluate SHORT model
y_pred_short = model_short.predict(X_test_short_scaled)
y_proba_short = model_short.predict_proba(X_test_short_scaled)

accuracy_short = accuracy_score(y_test_short_binary, y_pred_short)
print(f"\n✓ SHORT Model Test Accuracy: {accuracy_short:.1%}")

print("\nClassification Report:")
print(classification_report(y_test_short_binary, y_pred_short,
                          target_names=['NOT_SHORT', 'SHORT'], zero_division=0))

# Feature importance
importances_short = model_short.feature_importances_
top_indices = np.argsort(importances_short)[-10:][::-1]
print("\nTop 10 Features for SHORT:")
for idx in top_indices:
    print(f"  {feature_names[idx]:25}: {importances_short[idx]:.4f}")

# Save models
models_dir = BOT_DIR / 'models'
models_dir.mkdir(exist_ok=True)

print(f"\n{'='*80}")
print("SAVING MODELS")
print('='*80)

model_long_file = models_dir / 'btc_model_long_xgboost.pkl'
scaler_long_file = models_dir / 'btc_scaler_long_xgboost.pkl'
model_short_file = models_dir / 'btc_model_short_xgboost.pkl'
scaler_short_file = models_dir / 'btc_scaler_short_xgboost.pkl'
features_file = models_dir / 'btc_features_xgboost.pkl'

joblib.dump(model_long, model_long_file)
joblib.dump(scaler_long, scaler_long_file)
joblib.dump(model_short, model_short_file)
joblib.dump(scaler_short, scaler_short_file)
joblib.dump(feature_names, features_file)

print(f"✓ Saved LONG model: {model_long_file}")
print(f"✓ Saved LONG scaler: {scaler_long_file}")
print(f"✓ Saved SHORT model: {model_short_file}")
print(f"✓ Saved SHORT scaler: {scaler_short_file}")
print(f"✓ Saved features: {features_file}")

print(f"\n{'='*80}")
print("TRAINING COMPLETE")
print('='*80)
print(f"\nLONG Model Accuracy: {accuracy_long:.1%}")
print(f"SHORT Model Accuracy: {accuracy_short:.1%}")
print("\nNext: Validate these models in backtest with dual predictions")
print("="*80)
