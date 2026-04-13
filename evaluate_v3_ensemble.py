"""
Evaluate V3 Ensemble Performance

Tests the same ensemble strategies on V3 models to compare with V2 results.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

# Constants
TP_PCT = 0.01
SL_PCT = 0.005
HORIZONS = ['2h', '4h', '6h']
TRAIN_SPLIT = 0.8


def load_models_and_data():
    """Load V3 models and test data"""
    models_dir = Path(__file__).parent / 'ml_models'
    data_dir = Path(__file__).parent / 'data'

    print("Loading V3 models and data...")

    models = {}
    scalers = {}
    features = {}

    for horizon in HORIZONS:
        models[horizon] = joblib.load(models_dir / f'btc_{horizon}_model_v3.pkl')
        scalers[horizon] = joblib.load(models_dir / f'btc_{horizon}_scaler_v3.pkl')
        with open(models_dir / f'btc_{horizon}_features_v3.json', 'r') as f:
            features[horizon] = json.load(f)

    df = pd.read_parquet(data_dir / 'btc_5m_v3_features.parquet')
    split_idx = int(len(df) * TRAIN_SPLIT)
    df_test = df.iloc[split_idx:].copy()

    print(f"  Loaded {len(HORIZONS)} V3 models")
    print(f"  Test set: {len(df_test):,} rows")

    return models, scalers, features, df_test


def get_predictions_all_horizons(models, scalers, features, df_test):
    """Get predictions from all 3 V3 models"""
    print("\nGenerating predictions from V3 models...")

    predictions = {}
    probabilities = {}

    for horizon in HORIZONS:
        X = df_test[features[horizon]]
        X_scaled = scalers[horizon].transform(X)

        pred = models[horizon].predict(X_scaled)
        proba = models[horizon].predict_proba(X_scaled)

        predictions[horizon] = pred
        probabilities[horizon] = proba

        print(f"  {horizon}: {len(pred):,} predictions")

    return predictions, probabilities


def strategy_confidence_weighted(probabilities):
    """Best strategy from V2: Confidence-weighted voting"""
    print("\nStrategy: Confidence-weighted voting (V2 winner)")

    proba_2h = probabilities['2h'][:, 1]
    proba_4h = probabilities['4h'][:, 1]
    proba_6h = probabilities['6h'][:, 1]

    avg_proba = (proba_2h + proba_4h + proba_6h) / 3

    final_pred = []
    for prob in avg_proba:
        if prob > 0.6:
            final_pred.append(1)
        elif prob < 0.4:
            final_pred.append(0)
        else:
            final_pred.append(-1)

    final_pred = np.array(final_pred)
    trades = (final_pred != -1).sum()
    print(f"  Trades: {trades:,} ({trades/len(final_pred)*100:.1f}% of samples)")

    return final_pred


def backtest_strategy(predictions, df_test, strategy_name):
    """Backtest strategy against 2h labels"""
    actual = df_test['label_2h'].values

    trades = []
    for i, (pred, act) in enumerate(zip(predictions, actual)):
        if pred == -1:
            continue

        if np.isnan(act):
            continue

        if pred == 1:  # LONG
            if act == 1:
                pnl = TP_PCT * 100
                outcome = 'WIN'
            else:
                pnl = -SL_PCT * 100
                outcome = 'LOSS'
        else:  # SHORT
            if act == 0:
                pnl = SL_PCT * 100
                outcome = 'WIN'
            else:
                pnl = -TP_PCT * 100
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
    print("V3 Ensemble Evaluation")
    print("="*60)

    # Load V3 models and data
    models, scalers, features, df_test = load_models_and_data()

    # Get predictions
    predictions, probabilities = get_predictions_all_horizons(models, scalers, features, df_test)

    # Test individual V3 models
    print("\n" + "="*60)
    print("V3 INDIVIDUAL MODELS")
    print("="*60)

    results = []
    for horizon in HORIZONS:
        result = backtest_strategy(predictions[horizon], df_test, f"V3 Individual {horizon}")
        results.append(result)
        print(f"  {horizon}: {result['trades']:,} trades, {result['win_rate']:.1%} win rate, {result['avg_pnl']:+.4f}% avg P&L")

    # Test confidence-weighted ensemble
    print("\n" + "="*60)
    print("V3 ENSEMBLE STRATEGY")
    print("="*60)

    pred_confidence = strategy_confidence_weighted(probabilities)
    result = backtest_strategy(pred_confidence, df_test, "V3 Confidence Weighted")
    results.append(result)

    # Results table
    print("\n" + "="*60)
    print("V3 RESULTS SUMMARY")
    print("="*60)

    df_results = pd.DataFrame(results)

    print(f"\n{'Strategy':<25} {'Trades':<10} {'Wins':<8} {'Losses':<8} {'Win Rate':<12} {'Avg P&L%':<12} {'Total P&L%'}")
    print("-" * 95)

    for _, row in df_results.iterrows():
        print(f"{row['strategy']:<25} {row['trades']:<10,} {row['wins']:<8} {row['losses']:<8} "
              f"{row['win_rate']:<12.2%} {row['avg_pnl']:<12.4f} {row['total_pnl']:+.2f}")

    # Compare V2 vs V3
    print(f"\n{'='*60}")
    print("COMPARISON: V2 (12mo) vs V3 (4yr)")
    print(f"{'='*60}")

    models_dir = Path(__file__).parent / 'ml_models'
    try:
        with open(models_dir / 'ensemble_results.json', 'r') as f:
            v2_results = json.load(f)

        # Find V2 confidence-weighted result
        v2_conf = [r for r in v2_results if 'Confidence' in r['strategy']][0]

        print(f"\n{'Version':<12} {'Trades':<10} {'Win Rate':<12} {'Avg P&L%':<12} {'Total P&L%'}")
        print("-" * 60)
        print(f"{'V2 (12mo)':<12} {v2_conf['trades']:<10,} {v2_conf['win_rate']:<12.2%} {v2_conf['avg_pnl']:<12.4f} {v2_conf['total_pnl']:+.2f}")

        v3_conf = result
        print(f"{'V3 (4yr)':<12} {v3_conf['trades']:<10,} {v3_conf['win_rate']:<12.2%} {v3_conf['avg_pnl']:<12.4f} {v3_conf['total_pnl']:+.2f}")

        # Calculate improvement
        wr_improvement = (v3_conf['win_rate'] - v2_conf['win_rate']) * 100
        pnl_improvement = v3_conf['avg_pnl'] - v2_conf['avg_pnl']

        print(f"\nImprovement:")
        print(f"  Win rate: {wr_improvement:+.2f} percentage points")
        print(f"  Avg P&L: {pnl_improvement:+.4f}%")

        if v3_conf['avg_pnl'] > v2_conf['avg_pnl']:
            print(f"  ✓ V3 is better!")
        else:
            print(f"  → V2 was better (possible overfitting in V2)")

    except:
        print("  Could not load V2 results for comparison")

    # Save V3 results
    output_file = models_dir / 'ensemble_results_v3.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ V3 results saved to: {output_file}")


if __name__ == '__main__':
    main()
