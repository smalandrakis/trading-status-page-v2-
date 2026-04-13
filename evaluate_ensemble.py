"""
Ensemble Evaluation with Multiple Strategies

Tests different ways to combine the 3 horizon models:
1. All 3 models must agree (most conservative)
2. At least 2/3 models agree (balanced)
3. Confidence-weighted voting (weight by probability)
4. Near support/resistance filter (only trade near key levels)
5. Individual models (baseline)

Compares win rate, trade count, and P&L for each strategy.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

# Constants
TP_PCT = 0.01  # 1% Take Profit
SL_PCT = 0.005  # 0.5% Stop Loss
HORIZONS = ['2h', '4h', '6h']
TRAIN_SPLIT = 0.8


def load_models_and_data():
    """Load all V2 models, scalers, and test data"""
    models_dir = Path(__file__).parent / 'ml_models'
    data_dir = Path(__file__).parent / 'data'

    print("Loading models and data...")

    models = {}
    scalers = {}
    features = {}

    for horizon in HORIZONS:
        models[horizon] = joblib.load(models_dir / f'btc_{horizon}_model_v2.pkl')
        scalers[horizon] = joblib.load(models_dir / f'btc_{horizon}_scaler_v2.pkl')
        with open(models_dir / f'btc_{horizon}_features_v2.json', 'r') as f:
            features[horizon] = json.load(f)

    # Load data
    df = pd.read_parquet(data_dir / 'btc_5m_features_advanced.parquet')

    # Get test set (last 20%)
    split_idx = int(len(df) * TRAIN_SPLIT)
    df_test = df.iloc[split_idx:].copy()

    print(f"  Loaded {len(HORIZONS)} models")
    print(f"  Test set: {len(df_test):,} rows")

    return models, scalers, features, df_test


def get_predictions_all_horizons(models, scalers, features, df_test):
    """
    Get predictions and probabilities from all 3 models
    """
    print("\nGenerating predictions from all models...")

    predictions = {}
    probabilities = {}

    for horizon in HORIZONS:
        # Prepare features
        X = df_test[features[horizon]]
        X_scaled = scalers[horizon].transform(X)

        # Predict
        pred = models[horizon].predict(X_scaled)
        proba = models[horizon].predict_proba(X_scaled)

        predictions[horizon] = pred
        probabilities[horizon] = proba

        print(f"  {horizon}: {len(pred):,} predictions")

    return predictions, probabilities


def strategy_all_agree(predictions):
    """Strategy 1: All 3 models must agree"""
    print("\nStrategy 1: All 3 models agree")

    # Check where all 3 predictions are the same
    pred_2h = predictions['2h']
    pred_4h = predictions['4h']
    pred_6h = predictions['6h']

    all_agree = (pred_2h == pred_4h) & (pred_4h == pred_6h)
    final_pred = np.where(all_agree, pred_2h, -1)  # -1 = no trade

    trades = (final_pred != -1).sum()
    print(f"  Trades: {trades:,} ({trades/len(final_pred)*100:.1f}% of samples)")

    return final_pred


def strategy_majority_vote(predictions):
    """Strategy 2: At least 2/3 models agree"""
    print("\nStrategy 2: Majority vote (2/3 agree)")

    pred_2h = predictions['2h']
    pred_4h = predictions['4h']
    pred_6h = predictions['6h']

    # Stack predictions
    stacked = np.column_stack([pred_2h, pred_4h, pred_6h])

    # For each row, check if there's a majority
    final_pred = []
    for row in stacked:
        if np.sum(row == 1) >= 2:
            final_pred.append(1)  # Majority says LONG
        elif np.sum(row == 0) >= 2:
            final_pred.append(0)  # Majority says SHORT
        else:
            final_pred.append(-1)  # No majority

    final_pred = np.array(final_pred)
    trades = (final_pred != -1).sum()
    print(f"  Trades: {trades:,} ({trades/len(final_pred)*100:.1f}% of samples)")

    return final_pred


def strategy_confidence_weighted(probabilities):
    """Strategy 3: Weight by confidence"""
    print("\nStrategy 3: Confidence-weighted voting")

    # Get LONG probabilities for each model
    proba_2h = probabilities['2h'][:, 1]
    proba_4h = probabilities['4h'][:, 1]
    proba_6h = probabilities['6h'][:, 1]

    # Average probabilities
    avg_proba = (proba_2h + proba_4h + proba_6h) / 3

    # Predict LONG if avg prob > 0.5, otherwise SHORT
    # Only trade if confidence is high (> 0.6 or < 0.4)
    final_pred = []
    for prob in avg_proba:
        if prob > 0.6:
            final_pred.append(1)  # LONG
        elif prob < 0.4:
            final_pred.append(0)  # SHORT
        else:
            final_pred.append(-1)  # No trade (uncertain)

    final_pred = np.array(final_pred)
    trades = (final_pred != -1).sum()
    print(f"  Trades: {trades:,} ({trades/len(final_pred)*100:.1f}% of samples)")

    return final_pred


def strategy_near_support_resistance(predictions, df_test, threshold=0.02):
    """Strategy 4: Only trade when near support/resistance (within 2%)"""
    print(f"\nStrategy 4: Near support/resistance (within {threshold*100}%)")

    # Use 2h predictions as base
    base_pred = predictions['2h']

    # Check if price is near support or resistance
    near_support = df_test['dist_to_support_96'].abs() < threshold * 100
    near_resistance = df_test['dist_to_resistance_96'].abs() < threshold * 100
    near_level = near_support | near_resistance

    # Only trade when near a key level
    final_pred = np.where(near_level, base_pred, -1)

    trades = (final_pred != -1).sum()
    print(f"  Trades: {trades:,} ({trades/len(final_pred)*100:.1f}% of samples)")
    print(f"  Near support: {near_support.sum():,}, Near resistance: {near_resistance.sum():,}")

    return final_pred


def backtest_strategy(predictions, df_test, strategy_name):
    """
    Backtest a strategy against actual labels

    For each trade:
    - If predicted LONG (1) and actual is 1 (TP hit): +1% profit
    - If predicted LONG (1) and actual is 0 (SL hit): -0.5% loss
    - If predicted SHORT (0) and actual is 0 (SL hit for us = profit): +0.5% profit
    - If predicted SHORT (0) and actual is 1 (TP hit against us): -1% loss
    """
    # Use 2h labels for evaluation (most conservative)
    actual = df_test['label_2h'].values

    trades = []
    for i, (pred, act) in enumerate(zip(predictions, actual)):
        if pred == -1:  # No trade
            continue

        if np.isnan(act):  # No label (no clear outcome)
            continue

        # Determine P&L
        if pred == 1:  # Predicted LONG
            if act == 1:  # Actually hit TP
                pnl = TP_PCT * 100  # +1%
                outcome = 'WIN'
            else:  # Actually hit SL
                pnl = -SL_PCT * 100  # -0.5%
                outcome = 'LOSS'
        else:  # Predicted SHORT (pred == 0)
            if act == 0:  # Actually hit SL (what we want for SHORT)
                pnl = SL_PCT * 100  # +0.5%
                outcome = 'WIN'
            else:  # Actually hit TP (against us)
                pnl = -TP_PCT * 100  # -1%
                outcome = 'LOSS'

        trades.append({
            'pred': pred,
            'actual': act,
            'pnl': pnl,
            'outcome': outcome
        })

    if not trades:
        return {
            'strategy': strategy_name,
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'avg_pnl': 0.0
        }

    df_trades = pd.DataFrame(trades)
    wins = (df_trades['outcome'] == 'WIN').sum()
    losses = (df_trades['outcome'] == 'LOSS').sum()
    total_pnl = df_trades['pnl'].sum()
    avg_pnl = df_trades['pnl'].mean()
    win_rate = wins / len(df_trades)

    return {
        'strategy': strategy_name,
        'trades': len(df_trades),
        'wins': int(wins),
        'losses': int(losses),
        'win_rate': float(win_rate),
        'total_pnl': float(total_pnl),
        'avg_pnl': float(avg_pnl)
    }


def main():
    print("="*60)
    print("Ensemble Strategy Evaluation")
    print("="*60)

    # Load models and data
    models, scalers, features, df_test = load_models_and_data()

    # Get predictions from all models
    predictions, probabilities = get_predictions_all_horizons(models, scalers, features, df_test)

    # Test individual models (baselines)
    print("\n" + "="*60)
    print("BASELINE: Individual Models")
    print("="*60)

    results = []
    for horizon in HORIZONS:
        result = backtest_strategy(predictions[horizon], df_test, f"Individual {horizon}")
        results.append(result)
        print(f"  {horizon}: {result['trades']:,} trades, {result['win_rate']:.1%} win rate, {result['avg_pnl']:+.4f}% avg P&L")

    # Test ensemble strategies
    print("\n" + "="*60)
    print("ENSEMBLE STRATEGIES")
    print("="*60)

    # Strategy 1: All agree
    pred_all_agree = strategy_all_agree(predictions)
    result = backtest_strategy(pred_all_agree, df_test, "All 3 Agree")
    results.append(result)

    # Strategy 2: Majority vote
    pred_majority = strategy_majority_vote(predictions)
    result = backtest_strategy(pred_majority, df_test, "Majority (2/3)")
    results.append(result)

    # Strategy 3: Confidence weighted
    pred_confidence = strategy_confidence_weighted(probabilities)
    result = backtest_strategy(pred_confidence, df_test, "Confidence Weighted")
    results.append(result)

    # Strategy 4: Near support/resistance
    pred_near_sr = strategy_near_support_resistance(predictions, df_test)
    result = backtest_strategy(pred_near_sr, df_test, "Near Support/Resistance")
    results.append(result)

    # Summary table
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    df_results = pd.DataFrame(results)

    print(f"\n{'Strategy':<25} {'Trades':<10} {'Wins':<8} {'Losses':<8} {'Win Rate':<12} {'Avg P&L%':<12} {'Total P&L%'}")
    print("-" * 95)

    for _, row in df_results.iterrows():
        print(f"{row['strategy']:<25} {row['trades']:<10,} {row['wins']:<8} {row['losses']:<8} "
              f"{row['win_rate']:<12.2%} {row['avg_pnl']:<12.4f} {row['total_pnl']:+.2f}")

    # Find best strategy
    best = df_results.loc[df_results['avg_pnl'].idxmax()]

    print(f"\n{'='*60}")
    print(f"BEST STRATEGY: {best['strategy']}")
    print(f"{'='*60}")
    print(f"  Trades: {best['trades']:,}")
    print(f"  Win Rate: {best['win_rate']:.2%}")
    print(f"  Avg P&L: {best['avg_pnl']:+.4f}%")
    print(f"  Total P&L: {best['total_pnl']:+.2f}%")

    if best['avg_pnl'] > 0:
        print(f"\n  ✓ Profitable strategy found!")
    else:
        print(f"\n  ⚠ Still not profitable, but closer!")

    # Save results
    output_file = Path(__file__).parent / 'ml_models' / 'ensemble_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")


if __name__ == '__main__':
    main()
