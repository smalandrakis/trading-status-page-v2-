"""
Optimized Walk-Forward Test with Asymmetric Thresholds

Key insight: LONG trades have 2:1 R:R (need 33% WR), SHORT have 1:2 R:R (need 67% WR)
Therefore we should:
- Make it EASIER to take LONG trades (lower threshold)
- Make it HARDER to take SHORT trades (higher threshold)

Tests multiple threshold combinations to find optimal settings.
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
    long_trades = []
    short_trades = []

    for pred, act in zip(predictions, actual_labels):
        if pred == -1 or np.isnan(act):
            continue

        if pred == 1:  # LONG
            pnl = TP_PCT * 100 if act == 1 else -SL_PCT * 100
            trades.append(pnl)
            long_trades.append(pnl)
        else:  # SHORT
            pnl = SL_PCT * 100 if act == 0 else -TP_PCT * 100
            trades.append(pnl)
            short_trades.append(pnl)

    if not trades:
        return 0, 0, 0, 0, 0

    win_rate = sum(1 for t in trades if t > 0) / len(trades)
    avg_pnl = np.mean(trades)

    long_pct = len(long_trades) / len(trades) if trades else 0
    short_pct = len(short_trades) / len(trades) if trades else 0

    return len(trades), win_rate, avg_pnl, long_pct, short_pct


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


def ensemble_asymmetric_weighted(proba_2h, proba_4h, proba_6h, long_threshold, short_threshold):
    """
    Apply confidence-weighted ensemble with ASYMMETRIC thresholds

    long_threshold: Probability above which we predict LONG (lower = more LONG trades)
    short_threshold: Probability below which we predict SHORT (lower = fewer SHORT trades)
    """
    avg_proba = (proba_2h[:, 1] + proba_4h[:, 1] + proba_6h[:, 1]) / 3

    final_pred = []
    for prob in avg_proba:
        if prob > long_threshold:
            final_pred.append(1)  # LONG
        elif prob < short_threshold:
            final_pred.append(0)  # SHORT
        else:
            final_pred.append(-1)  # No trade

    return np.array(final_pred)


def test_walk_forward_with_thresholds(long_thresh, short_thresh):
    """Run walk-forward test with specific thresholds"""
    data_dir = Path(__file__).parent / 'data'
    df = pd.read_parquet(data_dir / 'btc_5m_v3_features.parquet')

    models_dir = Path(__file__).parent / 'ml_models'
    with open(models_dir / 'selected_features.json', 'r') as f:
        all_features = json.load(f)

    features = [f for f in all_features if f in df.columns]

    # Define walk-forward periods (using last 3 for speed)
    periods = [
        ("2022-01-01", "2024-06-30", "2024-07-01", "2024-09-30"),
        ("2022-01-01", "2024-09-30", "2024-10-01", "2024-12-31"),
        ("2022-01-01", "2024-12-31", "2025-01-01", "2026-03-20"),
    ]

    all_trades = 0
    all_wins = 0
    all_pnl_sum = 0
    all_long_count = 0
    all_short_count = 0

    for train_start, train_end, test_start, test_end in periods:
        df_train = df[(df.index >= train_start) & (df.index <= train_end)]
        df_test = df[(df.index >= test_start) & (df.index <= test_end)]

        if len(df_test) == 0:
            continue

        # Get common labeled test data
        has_all_labels = (
            df_test['label_2h'].notna() &
            df_test['label_4h'].notna() &
            df_test['label_6h'].notna()
        )
        df_test_common = df_test[has_all_labels]

        if len(df_test_common) < 100:
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

            pred, proba = train_and_predict(X_train, y_train, X_test_common, horizon)
            ensemble_probabilities[horizon] = proba

        if len(ensemble_probabilities) == 3:
            # Apply asymmetric threshold
            ensemble_pred = ensemble_asymmetric_weighted(
                ensemble_probabilities['2h'],
                ensemble_probabilities['4h'],
                ensemble_probabilities['6h'],
                long_thresh,
                short_thresh
            )

            trades, wr, avg_pnl, long_pct, short_pct = backtest_predictions(
                ensemble_pred, df_test_common['label_2h'].values
            )

            all_trades += trades
            all_wins += int(trades * wr)
            all_pnl_sum += trades * avg_pnl
            all_long_count += int(trades * long_pct)
            all_short_count += int(trades * short_pct)

    if all_trades == 0:
        return 0, 0, 0, 0, 0

    overall_wr = all_wins / all_trades
    overall_pnl = all_pnl_sum / all_trades
    overall_long_pct = all_long_count / all_trades
    overall_short_pct = all_short_count / all_trades

    return all_trades, overall_wr, overall_pnl, overall_long_pct, overall_short_pct


def main():
    print("="*80)
    print("Optimizing Ensemble Thresholds (Asymmetric)")
    print("="*80)

    print("\nKey Insight:")
    print("  LONG trades: 2:1 R:R, need only 33% WR to profit")
    print("  SHORT trades: 1:2 R:R, need 67% WR to profit")
    print("  → Should favor LONG trades with asymmetric thresholds!")

    print("\nTesting different threshold combinations...")
    print("(Using last 3 walk-forward periods for speed)\n")

    # Test different combinations
    threshold_combos = [
        (0.60, 0.40, "Symmetric (original)"),
        (0.55, 0.40, "Slightly favor LONG"),
        (0.55, 0.35, "Favor LONG more"),
        (0.50, 0.35, "Favor LONG strongly"),
        (0.50, 0.30, "LONG only (almost)"),
        (0.65, 0.45, "More conservative"),
    ]

    results = []

    print(f"{'Strategy':<25} {'Trades':<8} {'WR':<8} {'Avg P&L':<10} {'LONG%':<8} {'SHORT%'}")
    print("-"*80)

    for long_t, short_t, desc in threshold_combos:
        trades, wr, pnl, long_pct, short_pct = test_walk_forward_with_thresholds(long_t, short_t)

        print(f"{desc:<25} {trades:<8,} {wr:<8.2%} {pnl:<10.4f} {long_pct:<8.1%} {short_pct:.1%}")

        results.append({
            'strategy': desc,
            'long_threshold': long_t,
            'short_threshold': short_t,
            'trades': trades,
            'win_rate': wr,
            'avg_pnl': pnl,
            'long_pct': long_pct,
            'short_pct': short_pct
        })

    # Find best
    best = max(results, key=lambda x: x['avg_pnl'])

    print("\n" + "="*80)
    print("BEST STRATEGY")
    print("="*80)
    print(f"Strategy: {best['strategy']}")
    print(f"Thresholds: LONG >{best['long_threshold']}, SHORT <{best['short_threshold']}")
    print(f"Trades: {best['trades']:,}")
    print(f"Win Rate: {best['win_rate']:.2%}")
    print(f"Avg P&L: {best['avg_pnl']:+.4f}%")
    print(f"LONG: {best['long_pct']:.1%}, SHORT: {best['short_pct']:.1%}")

    if best['avg_pnl'] > 0.05:
        print(f"\n✓ PROFITABLE strategy found!")
    else:
        print(f"\n⚠ Still needs improvement")

    # Save best
    models_dir = Path(__file__).parent / 'ml_models'
    with open(models_dir / 'optimized_thresholds.json', 'w') as f:
        json.dump(best, f, indent=2)

    print(f"\n✓ Saved to: {models_dir / 'optimized_thresholds.json'}")


if __name__ == '__main__':
    main()
