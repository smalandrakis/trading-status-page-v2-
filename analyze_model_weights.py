"""
Analyze Model Weights and Feature Importance
Determine if models are trend-following or mean-reverting
"""

import joblib
import json
import numpy as np
from pathlib import Path

BOT_DIR = Path(__file__).parent
MODEL_DIR = BOT_DIR / 'btc_model_package'

def analyze_model(horizon):
    """Analyze a single model's weights and feature importance"""

    # Load model and features
    model = joblib.load(MODEL_DIR / f'btc_{horizon}_model_v3.pkl')
    with open(MODEL_DIR / f'btc_{horizon}_features_v3.json') as f:
        features = json.load(f)

    print(f"\n{'='*80}")
    print(f"{horizon.upper()} MODEL ANALYSIS")
    print(f"{'='*80}")

    # Check model type
    print(f"\nModel Type: {type(model).__name__}")

    # Get feature importances (for tree-based) or coefficients (for linear)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        print("\nUsing feature_importances_ (tree-based model)")
    elif hasattr(model, 'coef_'):
        # For logistic regression, coef_ shape is (n_classes, n_features)
        # We want coefficients for predicting class 1 (LONG)
        if len(model.coef_.shape) == 2:
            importances = model.coef_[1] if model.coef_.shape[0] > 1 else model.coef_[0]
        else:
            importances = model.coef_
        print("\nUsing coef_ (linear model - coefficients for LONG class)")
    else:
        print("\nCannot extract weights from this model type")
        return

    # Sort by absolute importance
    feature_importance = sorted(
        zip(features, importances),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    print(f"\n{'='*80}")
    print("TOP 10 MOST IMPORTANT FEATURES")
    print(f"{'='*80}")
    print(f"{'Feature':<30} {'Weight':<12} {'Type'}")
    print("-" * 80)

    # Categorize features
    trend_features = ['trend_', 'adx', 'dist_from_prev']
    mean_rev_features = ['rsi', 'bb_position', 'dist_to_support', 'dist_to_resistance']

    trend_total = 0
    mean_rev_total = 0

    for feat, weight in feature_importance[:10]:
        # Determine type
        is_trend = any(t in feat for t in trend_features)
        is_mean_rev = any(m in feat for m in mean_rev_features)

        if is_trend:
            feat_type = "TREND"
            trend_total += abs(weight)
        elif is_mean_rev:
            feat_type = "MEAN-REV"
            mean_rev_total += abs(weight)
        else:
            feat_type = "OTHER"

        print(f"{feat:<30} {weight:>11.6f}  {feat_type}")

    # Analyze key directional features
    print(f"\n{'='*80}")
    print("KEY DIRECTIONAL FEATURES ANALYSIS")
    print(f"{'='*80}")

    for feat, weight in feature_importance:
        if 'trend_2h_pct' in feat:
            print(f"\n✓ trend_2h_pct: {weight:.6f}")
            if weight > 0:
                print("  → Positive weight = Higher recent trend → Higher LONG probability")
                print("  → TREND-FOLLOWING: Goes WITH momentum")
            else:
                print("  → Negative weight = Higher recent trend → Lower LONG probability")
                print("  → MEAN-REVERTING: Fades momentum")

        if 'rsi' in feat:
            print(f"\n✓ rsi_28: {weight:.6f}")
            if weight > 0:
                print("  → Positive weight = Higher RSI (overbought) → Higher LONG probability")
                print("  → TREND-FOLLOWING: Buys strength")
            else:
                print("  → Negative weight = Higher RSI (overbought) → Lower LONG probability")
                print("  → MEAN-REVERTING: Sells strength, buys weakness")

        if 'bb_position' in feat:
            print(f"\n✓ bb_position_50: {weight:.6f}")
            if weight > 0:
                print("  → Positive weight = Higher in BB (near upper) → Higher LONG probability")
                print("  → TREND-FOLLOWING: Buys at upper BB")
            else:
                print("  → Negative weight = Higher in BB (near upper) → Lower LONG probability")
                print("  → MEAN-REVERTING: Buys at lower BB, sells at upper")

        if 'dist_to_resistance' in feat and '_48' in feat:
            print(f"\n✓ dist_to_resistance_48: {weight:.6f}")
            if weight > 0:
                print("  → Positive weight = Further from resistance → Higher LONG probability")
                print("  → Avoids resistance (mean-rev characteristic)")
            else:
                print("  → Negative weight = Further from resistance → Lower LONG probability")
                print("  → Likes approaching resistance (breakout/trend characteristic)")

        if 'dist_to_support' in feat and '_48' in feat:
            print(f"\n✓ dist_to_support_48: {weight:.6f}")
            if weight > 0:
                print("  → Positive weight = Further from support → Higher LONG probability")
                print("  → Buys breakouts away from support (trend)")
            else:
                print("  → Negative weight = Further from support → Lower LONG probability")
                print("  → Buys at support (mean-rev)")

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Trend features total weight:     {trend_total:.4f}")
    print(f"Mean-rev features total weight:  {mean_rev_total:.4f}")

    if trend_total > mean_rev_total * 1.5:
        print("\n✓ CONCLUSION: TREND-FOLLOWING model")
    elif mean_rev_total > trend_total * 1.5:
        print("\n✓ CONCLUSION: MEAN-REVERTING model")
    else:
        print("\n✓ CONCLUSION: HYBRID model (uses both strategies)")

    return feature_importance


def main():
    print("="*80)
    print("MODEL WEIGHT ANALYSIS - TREND vs MEAN-REVERSION")
    print("="*80)

    all_importances = {}

    for horizon in ['2h', '4h', '6h']:
        importances = analyze_model(horizon)
        all_importances[horizon] = importances

    # Overall summary
    print(f"\n\n{'='*80}")
    print("OVERALL MODEL BEHAVIOR")
    print(f"{'='*80}")

    print("\nBased on the analysis above:")
    print("1. Check if trend_2h_pct has positive weight (trend) or negative (mean-rev)")
    print("2. Check if RSI has negative weight (mean-rev) or positive (trend)")
    print("3. Check if bb_position has negative weight (mean-rev) or positive (trend)")
    print("\nThe signs of these weights will definitively tell us the strategy.")


if __name__ == '__main__':
    main()
