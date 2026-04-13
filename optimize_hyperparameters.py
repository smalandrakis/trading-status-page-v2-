"""
Hyperparameter Optimization for LONG/SHORT XGBoost Models

Uses GridSearchCV to find optimal parameters
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score, make_scorer
import joblib
import sys

print("="*80)
print("HYPERPARAMETER OPTIMIZATION - XGBOOST LONG/SHORT MODELS")
print("="*80)

BOT_DIR = Path(__file__).parent

# Load labeled data
data_file = BOT_DIR / 'data' / 'BTC_5min_with_wider_labels.parquet'
print(f"\nLoading labeled data...")
df = pd.read_parquet(data_file)
print(f"Loaded {len(df):,} rows")

# Add temporal features
print("Adding temporal features...")
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['prev_close_1'] = df['close'].shift(1)
df['prev_close_5'] = df['close'].shift(5)
df['prev_close_20'] = df['close'].shift(20)
df['price_change_1'] = df['close'].pct_change(1)
df['price_change_5'] = df['close'].pct_change(5)
df['price_change_20'] = df['close'].pct_change(20)

# Calculate V3 features
print("Calculating features...")
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

# Sample every 20th row for faster optimization
feature_list = []
valid_indices = []

for i in range(250, len(df), 20):
    features = calculate_features_for_row(df, i)

    if features is not None and isinstance(features, dict):
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

    if len(feature_list) % 2000 == 0:
        print(f"  Progress: {len(feature_list):,} rows...")

X = pd.DataFrame(feature_list)
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

feature_names = X.columns.tolist()
print(f"✓ Calculated {len(X):,} rows × {len(feature_names)} features")

# Get labels
df_valid = df.iloc[valid_indices].copy()
valid_mask = pd.Series(True, index=df_valid.index)
for horizon in ['30min', '1h', '2h']:
    valid_mask &= df_valid[f'label_{horizon}'].notna()

df_valid = df_valid[valid_mask]
X = X[valid_mask.values]
print(f"✓ Clean dataset: {len(X):,} rows")

# Focus on 2h horizon
horizon = '2h'
label_col = f'label_{horizon}'
y_all = df_valid[label_col]

# Train/test split
test_start = '2025-07-01'
train_mask = df_valid.index < test_start
test_mask = df_valid.index >= test_start

X_test = X[test_mask]
y_test = y_all[test_mask]

print(f"\nTest set: {len(X_test):,} rows")

# LONG MODEL OPTIMIZATION
print("\n" + "="*80)
print("OPTIMIZING LONG MODEL")
print("="*80)

bull_start = '2024-01-01'
bull_end = '2025-07-01'
long_mask = (df_valid.index >= bull_start) & (df_valid.index < bull_end)

X_train_long = X[long_mask]
y_train_long = y_all[long_mask]
y_train_long_binary = (y_train_long == 'LONG').astype(int)

print(f"Train set: {len(X_train_long):,} rows")
print(f"LONG: {y_train_long_binary.sum():,} ({y_train_long_binary.sum()/len(y_train_long_binary)*100:.1f}%)")

# Scale
scaler_long = StandardScaler()
X_train_long_scaled = scaler_long.fit_transform(X_train_long)

# Parameter grid - focused on most impactful parameters
param_grid_long = {
    'n_estimators': [200, 300, 400],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.03, 0.05, 0.07],
    'min_child_weight': [20, 30, 50],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'gamma': [0.5, 1.0, 1.5]
}

# Calculate scale_pos_weight
not_long = len(y_train_long_binary) - y_train_long_binary.sum()
scale_pos_weight_long = not_long / y_train_long_binary.sum()

print(f"\nHyperparameter grid: {sum([len(v) for v in param_grid_long.values()])} parameters")
print("Starting grid search (this may take 15-30 minutes)...")

# Custom scorer: F1 score for positive class (LONG)
f1_long_scorer = make_scorer(f1_score, pos_label=1)

base_model_long = XGBClassifier(
    scale_pos_weight=scale_pos_weight_long,
    reg_alpha=0.5,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    tree_method='hist',
    eval_metric='logloss'
)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

grid_search_long = GridSearchCV(
    base_model_long,
    param_grid_long,
    cv=cv,
    scoring=f1_long_scorer,
    n_jobs=-1,
    verbose=1
)

grid_search_long.fit(X_train_long_scaled, y_train_long_binary)

print(f"\n✓ Best LONG model score: {grid_search_long.best_score_:.4f}")
print(f"\nBest parameters:")
for param, value in grid_search_long.best_params_.items():
    print(f"  {param}: {value}")

# Save best LONG model
best_model_long = grid_search_long.best_estimator_

# SHORT MODEL OPTIMIZATION
print("\n" + "="*80)
print("OPTIMIZING SHORT MODEL")
print("="*80)

X_train_short = X[train_mask]
y_train_short = y_all[train_mask]
y_train_short_binary = (y_train_short == 'SHORT').astype(int)

print(f"Train set: {len(X_train_short):,} rows")
print(f"SHORT: {y_train_short_binary.sum():,} ({y_train_short_binary.sum()/len(y_train_short_binary)*100:.1f}%)")

# Scale
scaler_short = StandardScaler()
X_train_short_scaled = scaler_short.fit_transform(X_train_short)

# Parameter grid
param_grid_short = {
    'n_estimators': [200, 300, 400],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.03, 0.05, 0.07],
    'min_child_weight': [20, 30, 50],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'gamma': [0.5, 1.0, 1.5]
}

# Calculate scale_pos_weight
not_short = len(y_train_short_binary) - y_train_short_binary.sum()
scale_pos_weight_short = not_short / y_train_short_binary.sum()

print(f"\nHyperparameter grid: {sum([len(v) for v in param_grid_short.values()])} parameters")
print("Starting grid search (this may take 15-30 minutes)...")

# Custom scorer
f1_short_scorer = make_scorer(f1_score, pos_label=1)

base_model_short = XGBClassifier(
    scale_pos_weight=scale_pos_weight_short,
    reg_alpha=0.5,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    tree_method='hist',
    eval_metric='logloss'
)

grid_search_short = GridSearchCV(
    base_model_short,
    param_grid_short,
    cv=cv,
    scoring=f1_short_scorer,
    n_jobs=-1,
    verbose=1
)

grid_search_short.fit(X_train_short_scaled, y_train_short_binary)

print(f"\n✓ Best SHORT model score: {grid_search_short.best_score_:.4f}")
print(f"\nBest parameters:")
for param, value in grid_search_short.best_params_.items():
    print(f"  {param}: {value}")

# Save best SHORT model
best_model_short = grid_search_short.best_estimator_

# Evaluate on test set
print("\n" + "="*80)
print("EVALUATING OPTIMIZED MODELS")
print("="*80)

X_test_long_scaled = scaler_long.transform(X_test)
X_test_short_scaled = scaler_short.transform(X_test)

y_test_long_binary = (y_test == 'LONG').astype(int)
y_test_short_binary = (y_test == 'SHORT').astype(int)

y_pred_long = best_model_long.predict(X_test_long_scaled)
y_pred_short = best_model_short.predict(X_test_short_scaled)

acc_long = accuracy_score(y_test_long_binary, y_pred_long)
acc_short = accuracy_score(y_test_short_binary, y_pred_short)

print(f"\nLONG Model Test Accuracy: {acc_long:.1%}")
print(classification_report(y_test_long_binary, y_pred_long,
                          target_names=['NOT_LONG', 'LONG'], zero_division=0))

print(f"\nSHORT Model Test Accuracy: {acc_short:.1%}")
print(classification_report(y_test_short_binary, y_pred_short,
                          target_names=['NOT_SHORT', 'SHORT'], zero_division=0))

# Save optimized models
models_dir = BOT_DIR / 'models'
models_dir.mkdir(exist_ok=True)

print(f"\n{'='*80}")
print("SAVING OPTIMIZED MODELS")
print('='*80)

joblib.dump(best_model_long, models_dir / 'btc_model_long_xgboost_optimized.pkl')
joblib.dump(scaler_long, models_dir / 'btc_scaler_long_xgboost_optimized.pkl')
joblib.dump(best_model_short, models_dir / 'btc_model_short_xgboost_optimized.pkl')
joblib.dump(scaler_short, models_dir / 'btc_scaler_short_xgboost_optimized.pkl')
joblib.dump(feature_names, models_dir / 'btc_features_xgboost_optimized.pkl')

print("✓ Saved optimized LONG model")
print("✓ Saved optimized SHORT model")
print("✓ Saved optimized scalers and features")

print(f"\n{'='*80}")
print("OPTIMIZATION COMPLETE")
print('='*80)
print("\nNext: Validate optimized models in backtest")
print("="*80)
