"""
CORRECTED Walk-Forward Test with Fixed Backtest Logic

CRITICAL FIX: Both LONG and SHORT have 1% TP / 0.5% SL

Label meaning:
- Label = 1: LONG direction hit its TP (+1%) first
- Label = 0: SHORT direction hit its TP (-1% price = +1% for SHORT) first

Backtest P&L:
- Predict LONG, Label=1 (LONG won): +1.0%
- Predict LONG, Label=0 (SHORT won): -0.5%
- Predict SHORT, Label=0 (SHORT won): +1.0%  ← WAS WRONG!
- Predict SHORT, Label=1 (LONG won): -0.5%  ← WAS WRONG!

Both directions have identical 2:1 R:R, need only 33.3% WR to break even!
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

# Constants
TP_PCT = 0.01
SL_PCT = 0.005
RANDOM_STATE = 42


def backtest_predictions_CORRECTED(predictions, actual_labels):
    """
    CORRECTED backtest logic

    Both LONG and SHORT have same P&L structure:
    - Win: +1%
    - Loss: -0.5%
    """
    trades = []

    for pred, act in zip(predictions, actual_labels):
        if pred == -1 or np.isnan(act):
            continue

        if pred == 1:  # Predicted LONG
            if act == 1:  # LONG won
                pnl = TP_PCT * 100  # +1%
            else:  # SHORT won
                pnl = -SL_PCT * 100  # -0.5%
        else:  # Predicted SHORT (pred == 0)
            if act == 0:  # SHORT won
                pnl = TP_PCT * 100  # +1% ← FIXED!
            else:  # LONG won
                pnl = -SL_PCT * 100  # -0.5% ← FIXED!

        trades.append(pnl)

    if not trades:
        return 0, 0, 0

    win_rate = sum(1 for t in trades if t > 0) / len(trades)
    avg_pnl = np.mean(trades)
    total_pnl = np.sum(trades)

    return len(trades), win_rate, avg_pnl


def train_and_predict(X_train, y_train, X_test):
    """Train model and make predictions"""
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    sample_weight = np.where(y_train == 1, scale_pos_weight, 1.0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.15,
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.8,
        max_features='sqrt',
        random_state=RANDOM_STATE,
        verbose=0
    )

    model.fit(X_train_scaled, y_train, sample_weight=sample_weight)

    predictions = model.predict(X_test_scaled)
    probabilities = model.predict_proba(X_test_scaled)

    return predictions, probabilities


def ensemble_confidence_weighted(proba_2h, proba_4h, proba_6h):
    """Apply confidence-weighted ensemble strategy"""
    avg_proba = (proba_2h[:, 1] + proba_4h[:, 1] + proba_6h[:, 1]) / 3

    final_pred = []
    for prob in avg_proba:
        if prob > 0.6:
            final_pred.append(1)  # LONG
        elif prob < 0.4:
            final_pred.append(0)  # SHORT
        else:
            final_pred.append(-1)  # No trade

    return np.array(final_pred)


def main():
    data_dir = Path(__file__).parent / 'data'

    print("="*70)
    print("CORRECTED Walk-Forward Test (Fixed Backtest Logic)")
    print("="*70)

    print("\nCRITICAL FIX:")
    print("  Both LONG and SHORT have 1% TP / 0.5% SL")
    print("  Both need only 33.3% WR to break even")
    print("  Expected P&L at 54% WR: +0.31% per trade\n")

    # Load data
    df = pd.read_parquet(data_dir / 'btc_5m_v3_features.parquet')

    models_dir = Path(__file__).parent / 'ml_models'
    with open(models_dir / 'selected_features.json', 'r') as f:
        all_features = json.load(f)

    features = [f for f in all_features if f in df.columns]
    print(f"Using {len(features)} features")
    print(f"Data: {len(df):,} rows\n")

    # Define walk-forward periods
    periods = [
        ("2022-01-01", "2023-12-31", "2024-01-01", "2024-03-31", "2024 Q1"),
        ("2022-01-01", "2024-03-31", "2024-04-01", "2024-06-30", "2024 Q2"),
        ("2022-01-01", "2024-06-30", "2024-07-01", "2024-09-30", "2024 Q3"),
        ("2022-01-01", "2024-09-30", "2024-10-01", "2024-12-31", "2024 Q4"),
        ("2022-01-01", "2024-12-31", "2025-01-01", "2026-03-20", "2025-2026"),
    ]

    results = []

    for train_start, train_end, test_start, test_end, period_name in periods:
        print(f"{'='*70}")
        print(f"{period_name}: Train to {train_end}, Test {test_start}-{test_end}")
        print(f"{'='*70}")

        df_train = df[(df.index >= train_start) & (df.index <= train_end)]
        df_test = df[(df.index >= test_start) & (df.index <= test_end)]

        if len(df_test) == 0:
            continue

        # Get common labeled data
        has_all_labels = (
            df_test['label_2h'].notna() &
            df_test['label_4h'].notna() &
            df_test['label_6h'].notna()
        )
        df_test_common = df_test[has_all_labels]

        if len(df_test_common) < 100:
            print("Insufficient test data\n")
            continue

        X_test_common = df_test_common[features]
        ensemble_probabilities = {}

        for horizon in ['2h', '4h', '6h']:
            label_col = f'label_{horizon}'
            train_labeled = df_train[df_train[label_col].notna()]

            if len(train_labeled) < 1000:
                continue

            X_train = train_labeled[features]
            y_train = train_labeled[label_col]

            pred, proba = train_and_predict(X_train, y_train, X_test_common)
            ensemble_probabilities[horizon] = proba

        if len(ensemble_probabilities) == 3:
            ensemble_pred = ensemble_confidence_weighted(
                ensemble_probabilities['2h'],
                ensemble_probabilities['4h'],
                ensemble_probabilities['6h']
            )

            trades, wr, avg_pnl = backtest_predictions_CORRECTED(
                ensemble_pred, df_test_common['label_2h'].values
            )

            print(f"  Trades: {trades:,}")
            print(f"  Win Rate: {wr:.2%}")
            print(f"  Avg P&L: {avg_pnl:+.4f}%")
            print()

            results.append({
                'period': period_name,
                'trades': trades,
                'win_rate': float(wr),
                'avg_pnl': float(avg_pnl),
                'total_pnl': trades * avg_pnl
            })

    # Summary
    print("="*70)
    print("CORRECTED WALK-FORWARD SUMMARY")
    print("="*70)

    print(f"\n{'Period':<15} {'Trades':<10} {'Win Rate':<12} {'Avg P&L%':<12} {'Total P&L%'}")
    print("-"*70)

    total_trades = 0
    total_wins = 0
    total_pnl_sum = 0

    for r in results:
        print(f"{r['period']:<15} {r['trades']:<10,} {r['win_rate']:<12.2%} {r['avg_pnl']:<12.4f} {r['total_pnl']:+.2f}")
        total_trades += r['trades']
        total_wins += int(r['trades'] * r['win_rate'])
        total_pnl_sum += r['total_pnl']

    if total_trades > 0:
        overall_wr = total_wins / total_trades
        overall_pnl = total_pnl_sum / total_trades

        print("-"*70)
        print(f"{'OVERALL':<15} {total_trades:<10,} {overall_wr:<12.2%} {overall_pnl:<12.4f} {total_pnl_sum:+.2f}")

        print(f"\n{'='*70}")
        print("VALIDATION RESULTS")
        print(f"{'='*70}")
        print(f"Total Trades: {total_trades:,}")
        print(f"Win Rate: {overall_wr:.2%}")
        print(f"Avg P&L: {overall_pnl:+.4f}%")
        print(f"Total P&L: {total_pnl_sum:+.2f}%")

        # Expected at this WR
        expected_pnl = overall_wr * 1.0 + (1 - overall_wr) * (-0.5)
        print(f"\nExpected P&L at {overall_wr:.2%} WR: {expected_pnl:+.4f}%")
        print(f"Actual vs Expected: {overall_pnl - expected_pnl:+.4f}% difference")

        if overall_pnl > 0.1 and overall_wr > 0.4:
            print(f"\n✓ Strategy IS PROFITABLE!")
        else:
            print(f"\n⚠ Strategy needs improvement")

    # Save
    output_file = models_dir / 'walk_forward_CORRECTED.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Saved to: {output_file}")


if __name__ == '__main__':
    main()
