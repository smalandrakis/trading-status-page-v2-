"""
Improved LONG Model Training - Multi-Strategy Approach

Improvements:
1. Train on ONLY strong uptrend periods (filter by price momentum)
2. Add LONG-specific features (bullish patterns, breakouts, support bounces)
3. Lower LONG threshold to get more signals
4. Ensemble multiple LONG models from different bull periods
5. SMOTE oversampling for better class balance
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib
import sys

print("="*80)
print("IMPROVED LONG MODEL - MULTI-STRATEGY APPROACH")
print("="*80)

BOT_DIR = Path(__file__).parent

# Load labeled data
data_file = BOT_DIR / 'data' / 'BTC_5min_with_wider_labels.parquet'
print(f"\nLoading labeled data...")
df = pd.read_parquet(data_file)
print(f"Loaded {len(df):,} rows")

# Add temporal features
print("\nAdding temporal features...")
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['prev_close_1'] = df['close'].shift(1)
df['prev_close_5'] = df['close'].shift(5)
df['prev_close_20'] = df['close'].shift(20)
df['price_change_1'] = df['close'].pct_change(1)
df['price_change_5'] = df['close'].pct_change(5)
df['price_change_20'] = df['close'].pct_change(20)

# Add LONG-specific features
print("Adding LONG-specific features...")

# 1. Price momentum (20-bar, 50-bar, 100-bar)
df['momentum_20'] = df['close'].pct_change(20)
df['momentum_50'] = df['close'].pct_change(50)
df['momentum_100'] = df['close'].pct_change(100)

# 2. Is price above moving averages? (bullish signal)
df['ma_20'] = df['close'].rolling(20).mean()
df['ma_50'] = df['close'].rolling(50).mean()
df['above_ma_20'] = (df['close'] > df['ma_20']).astype(int)
df['above_ma_50'] = (df['close'] > df['ma_50']).astype(int)

# 3. Distance from recent low (support bounce indicator)
df['recent_low_20'] = df['low'].rolling(20).min()
df['dist_from_low_20'] = (df['close'] - df['recent_low_20']) / df['recent_low_20']

# 4. Bullish engulfing pattern (simple version)
df['candle_body'] = df['close'] - df['open']
df['prev_candle_body'] = df['candle_body'].shift(1)
df['bullish_engulfing'] = ((df['candle_body'] > 0) &
                           (df['prev_candle_body'] < 0) &
                           (df['candle_body'].abs() > df['prev_candle_body'].abs())).astype(int)

# 5. Higher highs indicator
df['prev_high'] = df['high'].shift(1)
df['higher_high'] = (df['high'] > df['prev_high']).astype(int)

# 6. Volume surge (bullish if high volume + price up)
df['volume_ma_20'] = df['volume'].rolling(20).mean()
df['volume_surge'] = (df['volume'] > df['volume_ma_20'] * 1.5).astype(int)
df['bullish_volume'] = (df['volume_surge'] * (df['close'] > df['open']).astype(int))

print("✓ Added 15 LONG-specific features")

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
feature_list = []
valid_indices = []

for i in range(250, len(df), 10):
    features = calculate_features_for_row(df, i)

    if features is not None and isinstance(features, dict):
        row = df.iloc[i]

        # Temporal features
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

        # LONG-specific features
        long_features = {
            'momentum_20': row['momentum_20'],
            'momentum_50': row['momentum_50'],
            'momentum_100': row['momentum_100'],
            'above_ma_20': row['above_ma_20'],
            'above_ma_50': row['above_ma_50'],
            'dist_from_low_20': row['dist_from_low_20'],
            'bullish_engulfing': row['bullish_engulfing'],
            'higher_high': row['higher_high'],
            'volume_surge': row['volume_surge'],
            'bullish_volume': row['bullish_volume']
        }

        features.update(temporal_features)
        features.update(long_features)
        feature_list.append(features)
        valid_indices.append(i)

    if len(feature_list) % 5000 == 0 and len(feature_list) > 0:
        print(f"  Progress: {len(feature_list):,} rows...")

X = pd.DataFrame(feature_list)
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

feature_names = X.columns.tolist()
print(f"\n✓ Calculated {len(X):,} rows × {len(feature_names)} features")
print(f"   (25 V3 + 8 temporal + 10 LONG-specific = 43 total)")

# Get labels
df_valid = df.iloc[valid_indices].copy()
valid_mask = pd.Series(True, index=df_valid.index)
for horizon in ['30min', '1h', '2h']:
    valid_mask &= df_valid[f'label_{horizon}'].notna()

df_valid = df_valid[valid_mask]
X = X[valid_mask.values]
print(f"✓ Clean dataset: {len(X):,} rows")

# Train/test split
test_start = '2025-07-01'
train_mask = df_valid.index < test_start
test_mask = df_valid.index >= test_start

X_test = X[test_mask]

horizon = '2h'
label_col = f'label_{horizon}'
y_all = df_valid[label_col]
y_test = y_all[test_mask]

# STRATEGY 1: Train on ONLY strong uptrend periods
print("\n" + "="*80)
print("STRATEGY 1: TRAIN ON STRONG UPTREND PERIODS ONLY")
print("="*80)

# Filter for strong uptrends (price up >10% over 30 days)
print("\nIdentifying strong uptrend periods...")
df_valid['price_30d_change'] = df_valid['close'].pct_change(30*288)  # 30 days * 288 5-min bars
strong_uptrend_mask = (df_valid['price_30d_change'] > 0.10) & (df_valid.index < test_start)

X_train_uptrend = X[strong_uptrend_mask]
y_train_uptrend = y_all[strong_uptrend_mask]
y_train_uptrend_binary = (y_train_uptrend == 'LONG').astype(int)

print(f"Strong uptrend periods: {len(X_train_uptrend):,} rows")
print(f"LONG: {y_train_uptrend_binary.sum():,} ({y_train_uptrend_binary.sum()/len(y_train_uptrend_binary)*100:.1f}%)")

# Scale
scaler_uptrend = StandardScaler()
X_train_uptrend_scaled = scaler_uptrend.fit_transform(X_train_uptrend)

# Train
scale_pos_weight = (len(y_train_uptrend_binary) - y_train_uptrend_binary.sum()) / y_train_uptrend_binary.sum()
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

model_uptrend = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    min_child_weight=20,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.5,
    reg_alpha=0.5,
    reg_lambda=1.0,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    tree_method='hist'
)

print("\nTraining uptrend-focused model...")
model_uptrend.fit(X_train_uptrend_scaled, y_train_uptrend_binary, verbose=False)

# Evaluate
X_test_scaled = scaler_uptrend.transform(X_test)
y_test_binary = (y_test == 'LONG').astype(int)
y_pred_uptrend = model_uptrend.predict(X_test_scaled)
acc_uptrend = accuracy_score(y_test_binary, y_pred_uptrend)

print(f"\n✓ Strategy 1 Test Accuracy: {acc_uptrend:.1%}")
print(classification_report(y_test_binary, y_pred_uptrend,
                          target_names=['NOT_LONG', 'LONG'], zero_division=0))

# STRATEGY 2: Use SMOTE for better class balance
print("\n" + "="*80)
print("STRATEGY 2: SMOTE OVERSAMPLING")
print("="*80)

bull_start = '2024-01-01'
bull_mask = (df_valid.index >= bull_start) & (df_valid.index < test_start)

X_train_bull = X[bull_mask]
y_train_bull = y_all[bull_mask]
y_train_bull_binary = (y_train_bull == 'LONG').astype(int)

print(f"\nBefore SMOTE:")
print(f"  Total: {len(y_train_bull_binary):,}")
print(f"  LONG: {y_train_bull_binary.sum():,} ({y_train_bull_binary.sum()/len(y_train_bull_binary)*100:.1f}%)")

# Apply SMOTE
smote = SMOTE(sampling_strategy=0.5, random_state=42)  # Oversample LONG to 50% of NOT_LONG
X_train_smote, y_train_smote = smote.fit_resample(X_train_bull, y_train_bull_binary)

print(f"\nAfter SMOTE:")
print(f"  Total: {len(y_train_smote):,}")
print(f"  LONG: {y_train_smote.sum():,} ({y_train_smote.sum()/len(y_train_smote)*100:.1f}%)")

# Scale
scaler_smote = StandardScaler()
X_train_smote_scaled = scaler_smote.fit_transform(X_train_smote)

# Train
scale_pos_weight_smote = (len(y_train_smote) - y_train_smote.sum()) / y_train_smote.sum()

model_smote = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    min_child_weight=20,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.5,
    reg_alpha=0.5,
    reg_lambda=1.0,
    scale_pos_weight=scale_pos_weight_smote,
    random_state=42,
    n_jobs=-1,
    tree_method='hist'
)

print("\nTraining SMOTE-balanced model...")
model_smote.fit(X_train_smote_scaled, y_train_smote, verbose=False)

# Evaluate
X_test_smote_scaled = scaler_smote.transform(X_test)
y_pred_smote = model_smote.predict(X_test_smote_scaled)
acc_smote = accuracy_score(y_test_binary, y_pred_smote)

print(f"\n✓ Strategy 2 Test Accuracy: {acc_smote:.1%}")
print(classification_report(y_test_binary, y_pred_smote,
                          target_names=['NOT_LONG', 'LONG'], zero_division=0))

# STRATEGY 3: Ensemble of both
print("\n" + "="*80)
print("STRATEGY 3: ENSEMBLE (Uptrend + SMOTE)")
print("="*80)

# Average probabilities
proba_uptrend = model_uptrend.predict_proba(X_test_scaled)[:, 1]
proba_smote = model_smote.predict_proba(X_test_smote_scaled)[:, 1]
proba_ensemble = (proba_uptrend + proba_smote) / 2

y_pred_ensemble = (proba_ensemble >= 0.5).astype(int)
acc_ensemble = accuracy_score(y_test_binary, y_pred_ensemble)

print(f"\n✓ Ensemble Test Accuracy: {acc_ensemble:.1%}")
print(classification_report(y_test_binary, y_pred_ensemble,
                          target_names=['NOT_LONG', 'LONG'], zero_division=0))

# Feature importance comparison
print("\n" + "="*80)
print("TOP FEATURES COMPARISON")
print("="*80)

print("\nUptrend Model Top 10:")
imp_uptrend = model_uptrend.feature_importances_
top_idx = np.argsort(imp_uptrend)[-10:][::-1]
for idx in top_idx:
    print(f"  {feature_names[idx]:30}: {imp_uptrend[idx]:.4f}")

print("\nSMOTE Model Top 10:")
imp_smote = model_smote.feature_importances_
top_idx = np.argsort(imp_smote)[-10:][::-1]
for idx in top_idx:
    print(f"  {feature_names[idx]:30}: {imp_smote[idx]:.4f}")

# Save all models
models_dir = BOT_DIR / 'models'
models_dir.mkdir(exist_ok=True)

print(f"\n{'='*80}")
print("SAVING IMPROVED MODELS")
print('='*80)

joblib.dump(model_uptrend, models_dir / 'btc_model_long_uptrend.pkl')
joblib.dump(scaler_uptrend, models_dir / 'btc_scaler_long_uptrend.pkl')
joblib.dump(model_smote, models_dir / 'btc_model_long_smote.pkl')
joblib.dump(scaler_smote, models_dir / 'btc_scaler_long_smote.pkl')
joblib.dump(feature_names, models_dir / 'btc_features_improved.pkl')

print("✓ Saved uptrend-focused model")
print("✓ Saved SMOTE-balanced model")
print("✓ Saved feature names (43 features)")

print(f"\n{'='*80}")
print("SUMMARY")
print('='*80)
print(f"\nStrategy 1 (Uptrend-only): {acc_uptrend:.1%} accuracy")
print(f"Strategy 2 (SMOTE): {acc_smote:.1%} accuracy")
print(f"Strategy 3 (Ensemble): {acc_ensemble:.1%} accuracy")
print("\nNext: Validate these improved models in backtest")
print("="*80)
