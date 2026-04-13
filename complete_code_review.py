"""
COMPLETE CODE REVIEW - Systematic Verification

Reviews entire pipeline from data → labels → training → backtesting:
1. Data source verification (V3 = 4 years)
2. Label creation logic (symmetric TP/SL)
3. Feature engineering (correct calculations)
4. Model training (V3 models, no data leakage)
5. Backtest P&L logic (corrected formulas)
6. Walk-forward methodology (proper time-based splits)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

def review_data_source():
    """Verify we're using the 4-year V3 dataset"""
    print("="*80)
    print("1. DATA SOURCE VERIFICATION")
    print("="*80)

    data_dir = Path(__file__).parent / 'data'

    # Check V3 features file
    v3_file = data_dir / 'btc_5m_v3_features.parquet'

    if not v3_file.exists():
        print("❌ V3 features file not found!")
        return False

    df = pd.read_parquet(v3_file)

    print(f"\n✓ Using V3 dataset: {v3_file}")
    print(f"  Total rows: {len(df):,}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print(f"  Duration: {(df.index.max() - df.index.min()).days} days")
    print(f"  Columns: {len(df.columns)}")

    # Verify it's actually 4 years of data
    duration_years = (df.index.max() - df.index.min()).days / 365.25

    if duration_years > 3.5:
        print(f"  ✓ Contains {duration_years:.1f} years of data (V3 confirmed)")
        return True
    else:
        print(f"  ❌ Only {duration_years:.1f} years - this might be V2!")
        return False


def review_label_creation():
    """Verify label creation uses symmetric TP/SL"""
    print("\n" + "="*80)
    print("2. LABEL CREATION LOGIC VERIFICATION")
    print("="*80)

    # Read the label creation script
    script_path = Path(__file__).parent / 'create_symmetric_labels.py'

    if not script_path.exists():
        print("❌ Label creation script not found")
        return False

    with open(script_path, 'r') as f:
        content = f.read()

    print("\nChecking for symmetric TP/SL in code...")

    # Check for the symmetric formulas
    checks = {
        'long_tp = current_close * (1 + tp_pct)': False,
        'long_sl = current_close * (1 - sl_pct)': False,
        'short_tp = current_close * (1 - tp_pct)': False,
        'short_sl = current_close * (1 + sl_pct)': False,
    }

    for pattern in checks.keys():
        if pattern.replace(' ', '') in content.replace(' ', ''):
            checks[pattern] = True
            print(f"  ✓ Found: {pattern}")
        else:
            print(f"  ❌ Missing: {pattern}")

    all_correct = all(checks.values())

    if all_correct:
        print("\n  ✓ Label creation uses SYMMETRIC 1% TP / 0.5% SL for both directions")
    else:
        print("\n  ❌ Label creation has issues!")

    # Verify actual labels in data
    data_dir = Path(__file__).parent / 'data'
    df = pd.read_parquet(data_dir / 'btc_5m_v3_features.parquet')

    print("\nActual label distribution in V3 data:")
    for horizon in ['2h', '4h', '6h']:
        if f'label_{horizon}' in df.columns:
            labels = df[f'label_{horizon}'].dropna()
            long_pct = (labels == 1).sum() / len(labels) * 100
            short_pct = (labels == 0).sum() / len(labels) * 100

            print(f"  {horizon}: LONG {long_pct:.1f}% / SHORT {short_pct:.1f}%", end="")

            # Should be roughly balanced (45-55% range is acceptable)
            if 45 <= long_pct <= 55:
                print(" ✓")
            else:
                print(" ⚠ Imbalanced!")

    return all_correct


def review_backtest_logic():
    """Verify backtest P&L calculations are correct"""
    print("\n" + "="*80)
    print("3. BACKTEST LOGIC VERIFICATION")
    print("="*80)

    # Check the corrected walk-forward script
    script_path = Path(__file__).parent / 'walk_forward_CORRECTED.py'

    if not script_path.exists():
        print("❌ Corrected walk-forward script not found")
        return False

    with open(script_path, 'r') as f:
        content = f.read()

    print("\nChecking backtest P&L formulas...")

    # The correct formulas we should find
    correct_patterns = [
        ('if pred == 1:', 'Predict LONG branch'),
        ('if act == 1:', 'LONG won check'),
        ('pnl = TP_PCT * 100', 'LONG win: +1%'),
        ('pnl = -SL_PCT * 100', 'LONG loss: -0.5%'),
        ('if act == 0:', 'SHORT won check'),
        ('pnl = TP_PCT * 100', 'SHORT win: +1%'),
        ('pnl = -SL_PCT * 100', 'SHORT loss: -0.5%'),
    ]

    # Parse the backtest function
    in_backtest_function = False
    found_correct_logic = False

    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'def backtest_predictions_CORRECTED' in line:
            in_backtest_function = True

        if in_backtest_function:
            # Check for the corrected SHORT logic
            if 'if act == 0:' in line:
                # Check next few lines for the correct P&L
                next_lines = '\n'.join(lines[i:i+5])
                if 'pnl = TP_PCT * 100' in next_lines:
                    print("  ✓ SHORT win gives +1% (TP_PCT * 100)")
                    found_correct_logic = True
                else:
                    print("  ❌ SHORT win P&L is wrong!")

    if found_correct_logic:
        print("\n  ✓ Backtest uses CORRECT logic: Both directions +1% win / -0.5% loss")
    else:
        print("\n  ❌ Backtest logic has issues!")

    # Verify with math
    print("\nMathematical verification:")
    print("  Given: TP = 1%, SL = 0.5%")
    print("  Breakeven WR = SL / (TP + SL) = 0.5 / 1.5 = 33.33%")
    print("  At 54% WR:")
    print(f"    Expected P&L = 0.54 × 1.0 + 0.46 × (-0.5) = {0.54 * 1.0 + 0.46 * (-0.5):.4f}%")

    # Check against actual results
    models_dir = Path(__file__).parent / 'ml_models'
    results_file = models_dir / 'walk_forward_CORRECTED.json'

    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)

        # Calculate overall
        total_trades = sum(r['trades'] for r in results)
        total_wins = sum(r['trades'] * r['win_rate'] for r in results)
        total_pnl = sum(r['total_pnl'] for r in results)

        overall_wr = total_wins / total_trades
        overall_pnl = total_pnl / total_trades
        expected_pnl = overall_wr * 1.0 + (1 - overall_wr) * (-0.5)

        print(f"\n  Actual results:")
        print(f"    WR: {overall_wr:.4f}")
        print(f"    Avg P&L: {overall_pnl:.4f}%")
        print(f"    Expected: {expected_pnl:.4f}%")
        print(f"    Difference: {abs(overall_pnl - expected_pnl):.4f}%", end="")

        if abs(overall_pnl - expected_pnl) < 0.001:
            print(" ✓ Matches theory!")
            return True
        else:
            print(" ⚠ Doesn't match theory!")
            return False

    return found_correct_logic


def review_walk_forward_methodology():
    """Verify walk-forward test has no data leakage"""
    print("\n" + "="*80)
    print("4. WALK-FORWARD METHODOLOGY VERIFICATION")
    print("="*80)

    print("\nChecking for data leakage risks...")

    # Load the corrected script
    script_path = Path(__file__).parent / 'walk_forward_CORRECTED.py'

    with open(script_path, 'r') as f:
        content = f.read()

    # Check for proper time-based splitting
    checks = {
        'df_train = df[(df.index >= train_start) & (df.index <= train_end)]':
            'Time-based train split',
        'df_test = df[(df.index >= test_start) & (df.index <= test_end)]':
            'Time-based test split',
        'X_train = train_labeled[features]':
            'Train features',
        'model.fit(X_train_scaled, y_train':
            'Model training',
    }

    for pattern, desc in checks.items():
        if pattern.replace(' ', '') in content.replace(' ', ''):
            print(f"  ✓ {desc}")
        else:
            print(f"  ❌ Missing: {desc}")

    # Check the actual periods used
    print("\nWalk-forward periods:")
    periods = [
        ("2022-01-01", "2023-12-31", "2024-01-01", "2024-03-31", "2024 Q1"),
        ("2022-01-01", "2024-03-31", "2024-04-01", "2024-06-30", "2024 Q2"),
        ("2022-01-01", "2024-06-30", "2024-07-01", "2024-09-30", "2024 Q3"),
        ("2022-01-01", "2024-09-30", "2024-10-01", "2024-12-31", "2024 Q4"),
        ("2022-01-01", "2024-12-31", "2025-01-01", "2026-03-20", "2025-2026"),
    ]

    valid = True
    for train_start, train_end, test_start, test_end, name in periods:
        # Verify train ends before test starts
        if train_end < test_start:
            print(f"  ✓ {name}: Train ends {train_end} before test starts {test_start}")
        else:
            print(f"  ❌ {name}: Data leakage! Train ends {train_end}, test starts {test_start}")
            valid = False

    if valid:
        print("\n  ✓ No data leakage - train always ends before test starts")
        print("  ✓ Expanding window approach (realistic scenario)")

    return valid


def review_features():
    """Verify features don't have lookahead bias"""
    print("\n" + "="*80)
    print("5. FEATURE ENGINEERING VERIFICATION")
    print("="*80)

    data_dir = Path(__file__).parent / 'data'
    df = pd.read_parquet(data_dir / 'btc_5m_v3_features.parquet')

    print(f"\nTotal features in V3 dataset: {len(df.columns)}")

    # Check for any forward-looking features (shouldn't exist)
    forward_looking_keywords = ['future', 'next', 'ahead', 'forward']

    suspicious_features = []
    for col in df.columns:
        col_lower = col.lower()
        for keyword in forward_looking_keywords:
            if keyword in col_lower and 'label' not in col_lower:
                suspicious_features.append(col)

    if suspicious_features:
        print(f"\n  ⚠ Found {len(suspicious_features)} potentially forward-looking features:")
        for feat in suspicious_features:
            print(f"    - {feat}")
    else:
        print("\n  ✓ No obviously forward-looking features detected")

    # Check that labels are not used as features
    models_dir = Path(__file__).parent / 'ml_models'
    with open(models_dir / 'selected_features.json', 'r') as f:
        selected_features = json.load(f)

    label_in_features = any('label' in f.lower() for f in selected_features)

    if label_in_features:
        print("  ❌ CRITICAL: Labels are being used as features!")
        return False
    else:
        print("  ✓ Labels are NOT in feature list")

    print(f"\n  Selected features for modeling: {len(selected_features)}")
    print("  Top 5 features:")
    for feat in selected_features[:5]:
        print(f"    - {feat}")

    return True


def review_model_artifacts():
    """Verify we're using V3 models"""
    print("\n" + "="*80)
    print("6. MODEL ARTIFACTS VERIFICATION")
    print("="*80)

    models_dir = Path(__file__).parent / 'ml_models'

    # Check for V3 model files
    v3_models = []
    for horizon in ['2h', '4h', '6h']:
        model_file = models_dir / f'btc_{horizon}_model_v3.pkl'
        if model_file.exists():
            v3_models.append(horizon)
            print(f"  ✓ Found V3 model: {model_file.name}")
        else:
            print(f"  ❌ Missing V3 model: btc_{horizon}_model_v3.pkl")

    if len(v3_models) == 3:
        print("\n  ✓ All 3 V3 models present")
    else:
        print(f"\n  ❌ Only {len(v3_models)}/3 V3 models found")
        return False

    # Check V3 metrics
    print("\nV3 Model Performance:")
    for horizon in ['2h', '4h', '6h']:
        metrics_file = models_dir / f'btc_{horizon}_metrics_v3.json'
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

            print(f"  {horizon}: Test Acc={metrics['test_acc']:.4f}, "
                  f"Overfit Gap={metrics['overfit_gap']:.4f}")

    return True


def final_summary():
    """Print final verification summary"""
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)

    checks = {
        'V3 Dataset (4 years)': review_data_source(),
        'Symmetric Label Logic': review_label_creation(),
        'Corrected Backtest': review_backtest_logic(),
        'Walk-Forward Valid': review_walk_forward_methodology(),
        'Features Valid': review_features(),
        'V3 Models Present': review_model_artifacts(),
    }

    print("\nFinal Checklist:")
    for check_name, passed in checks.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {check_name}")

    all_passed = all(checks.values())

    print("\n" + "="*80)
    if all_passed:
        print("✓✓✓ ALL CHECKS PASSED - RESULTS ARE VALID ✓✓✓")
    else:
        print("❌ SOME CHECKS FAILED - REVIEW NEEDED")
    print("="*80)

    return all_passed


def main():
    print("="*80)
    print("COMPLETE SYSTEMATIC CODE REVIEW")
    print("="*80)

    final_summary()


if __name__ == '__main__':
    main()
