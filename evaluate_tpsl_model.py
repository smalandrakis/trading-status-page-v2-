"""
Evaluate BTC TP/SL Model with Detailed Metrics and Backtesting

Provides comprehensive evaluation including:
- Classification metrics (accuracy, precision, recall, F1, ROC-AUC)
- Confusion matrix
- Confidence-filtered performance
- Backtesting simulation with P&L tracking
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, classification_report
)

# Constants
TP_PCT = 0.01  # 1% Take Profit
SL_PCT = 0.005  # 0.5% Stop Loss
TRAIN_SPLIT = 0.8
CONFIDENCE_THRESHOLDS = [0.6, 0.7, 0.8]


def load_model_artifacts():
    """Load trained model, scaler, and features"""
    models_dir = Path(__file__).parent / 'ml_models'

    print("Loading model artifacts...")
    model = joblib.load(models_dir / 'btc_tpsl_model.pkl')
    scaler = joblib.load(models_dir / 'btc_tpsl_scaler.pkl')

    with open(models_dir / 'btc_tpsl_features.json', 'r') as f:
        feature_cols = json.load(f)

    print(f"  ✓ Model loaded: {type(model).__name__}")
    print(f"  ✓ Scaler loaded")
    print(f"  ✓ Features: {len(feature_cols)} columns")

    return model, scaler, feature_cols


def load_and_prepare_data(feature_cols):
    """Load data and prepare test set"""
    data_dir = Path(__file__).parent / 'data'
    input_file = data_dir / 'btc_1m_tpsl_features.parquet'

    print(f"\nLoading data from: {input_file}")
    df = pd.read_parquet(input_file)

    # Separate features and labels
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'trades', 'label']
    X = df[feature_cols]
    y = df['label']

    # Time-based split (use same split as training)
    split_idx = int(len(X) * TRAIN_SPLIT)
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    print(f"Test set: {len(X_test):,} samples")
    print(f"  LONG (1):  {(y_test == 1).sum():,}")
    print(f"  SHORT (0): {(y_test == 0).sum():,}")

    return X_test, y_test, df.iloc[split_idx:]


def evaluate_classification_metrics(y_true, y_pred, y_pred_proba):
    """Calculate and display classification metrics"""
    print("\n" + "="*60)
    print("CLASSIFICATION METRICS")
    print("="*60)

    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Precision, Recall, F1 per class
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)

    print("\nPer-Class Metrics:")
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support'}")
    print("-" * 60)
    for i, class_name in enumerate(['SHORT (0)', 'LONG (1)']):
        print(f"{class_name:<10} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]}")

    # Macro/Weighted averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    print(f"\n{'Macro Avg':<10} {precision_macro:<12.4f} {recall_macro:<12.4f} {f1_macro:<12.4f}")
    print(f"{'Weighted Avg':<10} {precision_weighted:<12.4f} {recall_weighted:<12.4f} {f1_weighted:<12.4f}")

    # ROC-AUC
    try:
        roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
        print(f"\nROC-AUC Score: {roc_auc:.4f}")
    except:
        print("\nROC-AUC Score: N/A")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(f"{'':>12} {'Pred SHORT':<12} {'Pred LONG'}")
    print(f"{'True SHORT':<12} {cm[0,0]:<12,} {cm[0,1]:,}")
    print(f"{'True LONG':<12} {cm[1,0]:<12,} {cm[1,1]:,}")

    return accuracy, precision, recall, f1


def evaluate_confidence_filtering(y_true, y_pred_proba):
    """Evaluate performance at different confidence thresholds"""
    print("\n" + "="*60)
    print("CONFIDENCE FILTERING ANALYSIS")
    print("="*60)

    print(f"\n{'Threshold':<12} {'Trades':<10} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1'}")
    print("-" * 72)

    for threshold in CONFIDENCE_THRESHOLDS:
        # Filter predictions by confidence
        max_proba = y_pred_proba.max(axis=1)
        confident_mask = max_proba >= threshold
        confident_indices = np.where(confident_mask)[0]

        if len(confident_indices) == 0:
            print(f"{threshold:<12.1%} {'0':<10} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A'}")
            continue

        y_true_conf = y_true.iloc[confident_indices]
        y_pred_conf = y_pred_proba[confident_indices].argmax(axis=1)

        # Metrics for confident predictions
        acc = accuracy_score(y_true_conf, y_pred_conf)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true_conf, y_pred_conf, average='weighted')

        print(f"{threshold:<12.1%} {len(confident_indices):<10,} {acc:<12.4f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f}")


def backtest_simulation(y_true, y_pred, y_pred_proba, df_test):
    """Simulate trading with TP/SL logic"""
    print("\n" + "="*60)
    print("BACKTESTING SIMULATION")
    print("="*60)

    print("\nSimulating trades with TP=1.0%, SL=0.5%...")

    results = []

    # Test both "all signals" and confidence-filtered strategies
    strategies = [
        ("All Signals", 0.0),
    ] + [(f"Confidence >= {t:.0%}", t) for t in CONFIDENCE_THRESHOLDS]

    for strategy_name, threshold in strategies:
        # Filter by confidence
        max_proba = y_pred_proba.max(axis=1)
        if threshold > 0:
            mask = max_proba >= threshold
        else:
            mask = np.ones(len(y_pred), dtype=bool)

        trades = 0
        wins = 0
        losses = 0
        total_pnl = 0.0

        for i, idx in enumerate(df_test.index[mask]):
            pred = y_pred[mask][i]
            actual = y_true.iloc[mask].iloc[i]

            # Determine if trade would win or lose
            if pred == 1:  # Predicted LONG
                if actual == 1:  # Actually hit TP before SL
                    wins += 1
                    total_pnl += TP_PCT * 100  # 1% gain
                else:  # Actually hit SL
                    losses += 1
                    total_pnl -= SL_PCT * 100  # 0.5% loss
                trades += 1

            elif pred == 0:  # Predicted SHORT
                if actual == 0:  # Actually hit SL (which is what we want for SHORT)
                    wins += 1
                    total_pnl += SL_PCT * 100  # 0.5% gain
                else:  # Actually hit TP (against our SHORT)
                    losses += 1
                    total_pnl -= TP_PCT * 100  # 1% loss
                trades += 1

        if trades > 0:
            win_rate = wins / trades
            avg_pnl = total_pnl / trades
            expected_value = win_rate * TP_PCT * 100 - (1 - win_rate) * SL_PCT * 100

            results.append({
                'strategy': strategy_name,
                'trades': trades,
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_pnl': avg_pnl,
                'expected_value': expected_value
            })

    # Display results
    print(f"\n{'Strategy':<20} {'Trades':<10} {'Wins':<8} {'Losses':<8} {'Win Rate':<12} {'Total P&L%':<12} {'Avg P&L%'}")
    print("-" * 90)

    for r in results:
        print(f"{r['strategy']:<20} {r['trades']:<10,} {r['wins']:<8} {r['losses']:<8} "
              f"{r['win_rate']:<12.2%} {r['total_pnl']:<12.2f} {r['avg_pnl']:<12.4f}")

    # Analyze best strategy
    if results:
        best = max(results, key=lambda x: x['avg_pnl'])
        print(f"\n✓ Best Strategy: {best['strategy']}")
        print(f"  Win Rate: {best['win_rate']:.2%}")
        print(f"  Avg P&L per trade: {best['avg_pnl']:.4f}%")
        print(f"  Total P&L: {best['total_pnl']:.2f}%")

        # Check if profitable
        if best['total_pnl'] > 0:
            print(f"\n  💰 Strategy is profitable!")
        else:
            print(f"\n  ⚠️  Strategy is not profitable")

    return results


def save_metrics(metrics):
    """Save evaluation metrics to JSON"""
    models_dir = Path(__file__).parent / 'ml_models'
    metrics_path = models_dir / 'btc_tpsl_metrics.json'

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✓ Metrics saved to: {metrics_path}")


def main():
    print("="*60)
    print("MODEL EVALUATION & BACKTESTING")
    print("="*60)

    # Load artifacts
    model, scaler, feature_cols = load_model_artifacts()

    # Load and prepare data
    X_test, y_test, df_test = load_and_prepare_data(feature_cols)

    # Scale features
    print("\nScaling test features...")
    X_test_scaled = scaler.transform(X_test)

    # Generate predictions
    print("Generating predictions...")
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)

    # Evaluate classification metrics
    accuracy, precision, recall, f1 = evaluate_classification_metrics(y_test, y_pred, y_pred_proba)

    # Confidence filtering analysis
    evaluate_confidence_filtering(y_test, y_pred_proba)

    # Backtesting simulation
    backtest_results = backtest_simulation(y_test, y_pred, y_pred_proba, df_test)

    # Save metrics
    metrics = {
        'accuracy': float(accuracy),
        'precision_macro': float(precision.mean()),
        'recall_macro': float(recall.mean()),
        'f1_macro': float(f1.mean()),
        'backtest_results': backtest_results
    }
    save_metrics(metrics)

    print("\n" + "="*60)
    print("✓ EVALUATION COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
