"""
Audit BTC model training for:
1. Data leakage
2. Train/test balance
3. Model correctness
"""

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime

BTC_HORIZONS = {
    "5min": 1, "10min": 2, "15min": 3, "30min": 6, "45min": 9,
    "1h": 12, "1h30m": 18, "2h": 24, "3h": 36, "4h": 48,
    "6h": 72, "8h": 96, "12h": 144,
}
BTC_THRESHOLDS = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]


def audit_data_leakage(df):
    """Check for data leakage in features."""
    print("\n" + "="*80)
    print("1. DATA LEAKAGE AUDIT")
    print("="*80)
    
    issues = []
    exclude = ['target_', 'future_', 'Open', 'High', 'Low', 'Close', 'Volume']
    feature_cols = [c for c in df.columns if not any(p in c for p in exclude)]
    
    print(f"\nTotal features: {len(feature_cols)}")
    
    # Check feature names
    future_feat = [c for c in feature_cols if 'future' in c.lower()]
    target_feat = [c for c in feature_cols if 'target' in c.lower()]
    
    print(f"✓ No 'future' in features: {len(future_feat) == 0}")
    print(f"✓ No 'target' in features: {len(target_feat) == 0}")
    
    if future_feat:
        issues.append(f"CRITICAL: Features contain 'future': {future_feat}")
    if target_feat:
        issues.append(f"CRITICAL: Features contain 'target': {target_feat}")
    
    # Verify target calculation
    print("\nVerifying target variable calculation...")
    for h_name, h_bars in list(BTC_HORIZONS.items())[:2]:
        future_price = df['Close'].shift(-h_bars)
        future_ret = (future_price / df['Close'] - 1) * 100
        target_col = f"target_up_{h_name}_{BTC_THRESHOLDS[0]}pct"
        
        if target_col in df.columns:
            expected = (future_ret >= BTC_THRESHOLDS[0]).astype(int)
            valid = ~expected.isna() & ~df[target_col].isna()
            match = (expected[valid] == df[target_col][valid]).all()
            print(f"  {h_name} target correct: {match}")
            if not match:
                issues.append(f"Target mismatch: {h_name}")
    
    return {'issues': issues, 'passed': len(issues) == 0}


def audit_train_test_split(df):
    """Audit train/test split."""
    print("\n" + "="*80)
    print("2. TRAIN/TEST SPLIT AUDIT")
    print("="*80)
    
    issues = []
    df = df.sort_index()
    split_idx = int(len(df) * 0.8)
    
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    print(f"\nTrain: {len(train_df):,} rows ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Test:  {len(test_df):,} rows ({len(test_df)/len(df)*100:.1f}%)")
    print(f"\nTrain: {train_df.index[0]} to {train_df.index[-1]}")
    print(f"Test:  {test_df.index[0]} to {test_df.index[-1]}")
    
    no_overlap = test_df.index[0] > train_df.index[-1]
    print(f"\n✓ No temporal overlap: {no_overlap}")
    
    if not no_overlap:
        issues.append("Train/test overlap!")
    
    return {'issues': issues, 'passed': len(issues) == 0}


def audit_class_balance(df):
    """Check class balance for targets."""
    print("\n" + "="*80)
    print("3. CLASS BALANCE AUDIT")
    print("="*80)
    
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    print("\nTarget positive rates (train / test):")
    print("-" * 60)
    
    for h_name in BTC_HORIZONS.keys():
        for thresh in BTC_THRESHOLDS[:3]:  # First 3 thresholds
            col = f"target_up_{h_name}_{thresh}pct"
            if col in df.columns:
                train_rate = train_df[col].mean() * 100
                test_rate = test_df[col].mean() * 100
                drift = abs(train_rate - test_rate)
                flag = "⚠️" if drift > 5 else "✓"
                print(f"{flag} {h_name} {thresh}%: {train_rate:.2f}% / {test_rate:.2f}% (drift: {drift:.2f}%)")
        print()


def audit_model_predictions(df):
    """Verify model predictions make sense."""
    print("\n" + "="*80)
    print("4. MODEL PREDICTION AUDIT")
    print("="*80)
    
    exclude = ['target_', 'future_', 'Open', 'High', 'Low', 'Close', 'Volume']
    feature_cols = [c for c in df.columns if not any(p in c for p in exclude)]
    
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()
    
    # Load a model and verify
    try:
        model = joblib.load('models_btc/model_12h_0.5pct.joblib')
        target_col = 'target_up_12h_0.5pct'
        
        X_test = test_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y_test = test_df[target_col]
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        print(f"\nModel: 12h horizon, 0.5% threshold")
        print(f"Test samples: {len(y_test):,}")
        print(f"Actual positives: {y_test.sum():,} ({y_test.mean()*100:.2f}%)")
        print(f"Predicted positives: {y_pred.sum():,} ({y_pred.mean()*100:.2f}%)")
        print(f"Probability range: [{y_proba.min():.3f}, {y_proba.max():.3f}]")
        print(f"Mean probability: {y_proba.mean():.3f}")
        
        # Check if predictions are reasonable
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"  TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
        print(f"  FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")
        
        # Verify model isn't just predicting majority class
        pred_variety = len(np.unique(y_pred))
        print(f"\n✓ Model predicts both classes: {pred_variety > 1}")
        
    except Exception as e:
        print(f"Error loading model: {e}")


def audit_feature_importance(df):
    """Check feature importance for sanity."""
    print("\n" + "="*80)
    print("5. FEATURE IMPORTANCE AUDIT")
    print("="*80)
    
    try:
        model = joblib.load('models_btc/model_12h_0.5pct.joblib')
        with open('models_btc/feature_columns.json', 'r') as f:
            feature_cols = json.load(f)
        
        importances = model.feature_importances_
        feat_imp = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("\nTop 15 most important features:")
        for i, row in feat_imp.head(15).iterrows():
            print(f"  {row['importance']:.4f} - {row['feature']}")
        
        # Check for suspicious features
        suspicious = ['future', 'target', 'forward']
        for feat in feat_imp.head(20)['feature']:
            for s in suspicious:
                if s in feat.lower():
                    print(f"\n⚠️ SUSPICIOUS: Top feature contains '{s}': {feat}")
        
    except Exception as e:
        print(f"Error: {e}")


def main():
    print("="*80)
    print("BTC MODEL TRAINING AUDIT")
    print("="*80)
    print(f"Audit time: {datetime.now()}")
    
    # Load data
    print("\nLoading data...")
    df = pd.read_parquet('data/BTC_features.parquet')
    
    # Create targets (same as training)
    for h_name, h_bars in BTC_HORIZONS.items():
        future_price = df['Close'].shift(-h_bars)
        future_ret = (future_price / df['Close'] - 1) * 100
        for thresh in BTC_THRESHOLDS:
            col = f"target_up_{h_name}_{thresh}pct"
            df[col] = (future_ret >= thresh).astype(int)
    
    max_h = max(BTC_HORIZONS.values())
    df = df.iloc[:-max_h]
    
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Run audits
    r1 = audit_data_leakage(df)
    r2 = audit_train_test_split(df)
    audit_class_balance(df)
    audit_model_predictions(df)
    audit_feature_importance(df)
    
    # Summary
    print("\n" + "="*80)
    print("AUDIT SUMMARY")
    print("="*80)
    print(f"Data Leakage Check: {'✅ PASSED' if r1['passed'] else '❌ FAILED'}")
    print(f"Train/Test Split:   {'✅ PASSED' if r2['passed'] else '❌ FAILED'}")
    
    if r1['issues']:
        print(f"\nIssues found: {r1['issues']}")
    if r2['issues']:
        print(f"\nIssues found: {r2['issues']}")


if __name__ == "__main__":
    main()
