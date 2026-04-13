"""
Walk-Forward Testing for V3 Models

Tests model performance on truly unseen future data using expanding window:
1. Train on 2022-2023 → Test on 2024 Q1
2. Train on 2022-2024 Q1 → Test on 2024 Q2
3. Train on 2022-2024 Q2 → Test on 2024 Q3
4. Train on 2022-2024 Q3 → Test on 2024 Q4
5. Train on 2022-2024 → Test on 2025-2026

This simulates real trading where you retrain periodically on new data.
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


def backtest_predictions(predictions, actual_labels):
    """Simple backtest of predictions vs actual"""
    trades = []

    for pred, act in zip(predictions, actual_labels):
        if pred == -1 or np.isnan(act):
            continue

        if pred == 1:  # LONG
            pnl = TP_PCT * 100 if act == 1 else -SL_PCT * 100
        else:  # SHORT
            pnl = SL_PCT * 100 if act == 0 else -TP_PCT * 100

        trades.append(pnl)

    if not trades:
        return 0, 0, 0

    win_rate = sum(1 for t in trades if t > 0) / len(trades)
    avg_pnl = np.mean(trades)
    total_pnl = np.sum(trades)

    return len(trades), win_rate, avg_pnl


def train_and_predict(X_train, y_train, X_test, horizon_name):
    """Train model and make predictions"""
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    sample_weight = np.where(y_train == 1, scale_pos_weight, 1.0)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train
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

    # Predict
    predictions = model.predict(X_test_scaled)
    probabilities = model.predict_proba(X_test_scaled)

    return predictions, probabilities


def ensemble_confidence_weighted(proba_2h, proba_4h, proba_6h):
    """Apply confidence-weighted ensemble strategy"""
    avg_proba = (proba_2h[:, 1] + proba_4h[:, 1] + proba_6h[:, 1]) / 3

    final_pred = []
    for prob in avg_proba:
        if prob > 0.6:
            final_pred.append(1)
        elif prob < 0.4:
            final_pred.append(0)
        else:
            final_pred.append(-1)

    return np.array(final_pred)


def main():
    data_dir = Path(__file__).parent / 'data'

    print("="*70)
    print("Walk-Forward Testing: Expanding Window")
    print("="*70)

    # Load data
    df = pd.read_parquet(data_dir / 'btc_5m_v3_features.parquet')

    # Load selected features
    models_dir = Path(__file__).parent / 'ml_models'
    with open(models_dir / 'selected_features.json', 'r') as f:
        all_features = json.load(f)

    features = [f for f in all_features if f in df.columns]
    print(f"\nUsing {len(features)} features")
    print(f"Data: {len(df):,} rows from {df.index.min()} to {df.index.max()}")

    # Define walk-forward periods
    periods = [
        ("2022-01-01", "2023-12-31", "2024-01-01", "2024-03-31", "Train 2022-2023 → Test 2024 Q1"),
        ("2022-01-01", "2024-03-31", "2024-04-01", "2024-06-30", "Train 2022-2024Q1 → Test 2024 Q2"),
        ("2022-01-01", "2024-06-30", "2024-07-01", "2024-09-30", "Train 2022-2024Q2 → Test 2024 Q3"),
        ("2022-01-01", "2024-09-30", "2024-10-01", "2024-12-31", "Train 2022-2024Q3 → Test 2024 Q4"),
        ("2022-01-01", "2024-12-31", "2025-01-01", "2026-03-20", "Train 2022-2024 → Test 2025-2026"),
    ]

    results = []

    for train_start, train_end, test_start, test_end, description in periods:
        print(f"\n{'='*70}")
        print(description)
        print(f"{'='*70}")

        # Split data
        df_train = df[(df.index >= train_start) & (df.index <= train_end)]
        df_test = df[(df.index >= test_start) & (df.index <= test_end)]

        print(f"Train: {len(df_train):,} rows ({df_train.index.min()} to {df_train.index.max()})")
        print(f"Test:  {len(df_test):,} rows ({df_test.index.min()} to {df_test.index.max()})")

        if len(df_test) == 0:
            print("⚠ No test data in this period")
            continue

        # Train and predict for each horizon
        predictions = {}
        probabilities = {}

        for horizon in ['2h', '4h', '6h']:
            label_col = f'label_{horizon}'

            # Filter to labeled data
            train_labeled = df_train[df_train[label_col].notna()]
            test_labeled = df_test[df_test[label_col].notna()]

            if len(train_labeled) < 1000 or len(test_labeled) < 100:
                print(f"  {horizon}: Insufficient data")
                continue

            X_train = train_labeled[features]
            y_train = train_labeled[label_col]
            X_test = test_labeled[features]
            y_test = test_labeled[label_col]

            print(f"\n  {horizon}: Training on {len(X_train):,} samples...")
            pred, proba = train_and_predict(X_train, y_train, X_test, horizon)

            # Individual model performance
            trades, wr, avg_pnl = backtest_predictions(pred, y_test)
            print(f"    Individual: {trades:,} trades, WR={wr:.2%}, Avg P&L={avg_pnl:+.4f}%")

            predictions[horizon] = pred
            probabilities[horizon] = proba

        # Ensemble strategy - need common rows with all 3 labels
        if len(predictions) == 3:
            print(f"\n  Ensemble (Confidence-Weighted):")

            # Get indices where all 3 labels exist
            has_all_labels = (
                df_test['label_2h'].notna() &
                df_test['label_4h'].notna() &
                df_test['label_6h'].notna()
            )
            df_test_common = df_test[has_all_labels]

            if len(df_test_common) < 100:
                print(f"    Insufficient common labeled data ({len(df_test_common)} rows)")
                continue

            # Train on common labeled data
            X_test_common = df_test_common[features]
            ensemble_predictions = {}
            ensemble_probabilities = {}

            for horizon in ['2h', '4h', '6h']:
                label_col = f'label_{horizon}'
                train_labeled = df_train[df_train[label_col].notna()]

                X_train = train_labeled[features]
                y_train = train_labeled[label_col]

                pred, proba = train_and_predict(X_train, y_train, X_test_common, horizon)
                ensemble_predictions[horizon] = pred
                ensemble_probabilities[horizon] = proba

            # Apply ensemble
            ensemble_pred = ensemble_confidence_weighted(
                ensemble_probabilities['2h'],
                ensemble_probabilities['4h'],
                ensemble_probabilities['6h']
            )

            trades, wr, avg_pnl = backtest_predictions(ensemble_pred, df_test_common['label_2h'].values)

            print(f"    Ensemble: {trades:,} trades, WR={wr:.2%}, Avg P&L={avg_pnl:+.4f}%")

            results.append({
                'period': description,
                'test_start': test_start,
                'test_end': test_end,
                'trades': trades,
                'win_rate': float(wr),
                'avg_pnl': float(avg_pnl),
                'total_pnl': trades * avg_pnl if trades > 0 else 0
            })

    # Summary
    print(f"\n{'='*70}")
    print("WALK-FORWARD SUMMARY")
    print(f"{'='*70}")

    print(f"\n{'Period':<40} {'Trades':<10} {'Win Rate':<12} {'Avg P&L%':<12} {'Total P&L%'}")
    print("-" * 90)

    for r in results:
        print(f"{r['period']:<40} {r['trades']:<10,} {r['win_rate']:<12.2%} {r['avg_pnl']:<12.4f} {r['total_pnl']:+.2f}")

    # Overall statistics
    if results:
        total_trades = sum(r['trades'] for r in results)
        weighted_wr = sum(r['win_rate'] * r['trades'] for r in results) / total_trades
        weighted_pnl = sum(r['avg_pnl'] * r['trades'] for r in results) / total_trades
        total_pnl = sum(r['total_pnl'] for r in results)

        print("-" * 90)
        print(f"{'OVERALL':<40} {total_trades:<10,} {weighted_wr:<12.2%} {weighted_pnl:<12.4f} {total_pnl:+.2f}")

        print(f"\n{'='*70}")
        print("VALIDATION RESULTS")
        print(f"{'='*70}")
        print(f"Total Trades: {total_trades:,}")
        print(f"Overall Win Rate: {weighted_wr:.2%}")
        print(f"Overall Avg P&L: {weighted_pnl:+.4f}%")
        print(f"Total P&L: {total_pnl:+.2f}%")

        if weighted_pnl > 0 and weighted_wr > 0.52:
            print(f"\n✓ Strategy is PROFITABLE on out-of-sample walk-forward test!")
            print(f"✓ Win rate > 52% (breakeven at 2:1 R:R)")
        else:
            print(f"\n⚠ Strategy needs improvement")
            print(f"  Target: WR > 52% and Avg P&L > 0")

    # Save results
    output_file = Path(__file__).parent / 'ml_models' / 'walk_forward_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Walk-forward results saved to: {output_file}")


if __name__ == '__main__':
    main()
