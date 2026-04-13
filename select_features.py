"""
Feature Selection: Top Features + Correlation Removal

Reduces feature set from 93 to ~30 best features:
1. Aggregate feature importance across all 3 horizon models
2. Select top N features by importance
3. Remove highly correlated features (correlation > 0.9)
4. Save reduced feature set for retraining
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path


def load_all_models():
    """Load all 3 horizon models and their feature importance"""
    models_dir = Path(__file__).parent / 'ml_models'
    horizons = ['2h', '4h', '6h']

    importance_data = []

    for horizon in horizons:
        # Load model
        model = joblib.load(models_dir / f'btc_{horizon}_model.pkl')

        # Load feature importance CSV
        importance_df = pd.read_csv(models_dir / f'btc_{horizon}_feature_importance.csv')
        importance_df['horizon'] = horizon
        importance_data.append(importance_df)

    return importance_data


def aggregate_feature_importance(importance_data):
    """
    Aggregate feature importance across all models

    Uses mean importance across all 3 horizons
    """
    print("\nAggregating feature importance across all models...")

    # Combine all importance data
    all_importance = pd.concat(importance_data, ignore_index=True)

    # Group by feature and calculate mean importance
    avg_importance = all_importance.groupby('feature')['importance'].mean().reset_index()
    avg_importance = avg_importance.sort_values('importance', ascending=False)

    print(f"  Found {len(avg_importance)} unique features")

    return avg_importance


def remove_correlated_features(feature_list, df, threshold=0.9):
    """
    Remove highly correlated features

    Keep feature with higher average importance when correlation > threshold
    """
    print(f"\nRemoving correlated features (threshold > {threshold})...")

    # Get correlation matrix for selected features
    corr_matrix = df[feature_list].corr().abs()

    # Find pairs of highly correlated features
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_remove = set()

    for column in upper_tri.columns:
        # Find features correlated with this one
        correlated = upper_tri[column][upper_tri[column] > threshold].index.tolist()

        if correlated:
            # Add the current column to removal list (we keep the correlated one from earlier in the list)
            to_remove.add(column)

    print(f"  Found {len(to_remove)} correlated features to remove")

    # Remove correlated features
    final_features = [f for f in feature_list if f not in to_remove]

    return final_features, list(to_remove)


def main():
    data_dir = Path(__file__).parent / 'data'
    models_dir = Path(__file__).parent / 'ml_models'

    print("="*60)
    print("Feature Selection: Importance + Correlation")
    print("="*60)

    # Load all models' feature importance
    importance_data = load_all_models()

    # Aggregate importance
    avg_importance = aggregate_feature_importance(importance_data)

    # Show top 20 features
    print("\nTop 20 features by average importance:")
    for i, row in avg_importance.head(20).iterrows():
        print(f"  {i+1:2d}. {row['feature'][:40]:<40} {row['importance']:.6f}")

    # Select top N features
    TOP_N = 40  # Start with 40, will reduce further after correlation removal
    print(f"\nSelecting top {TOP_N} features...")

    top_features = avg_importance.head(TOP_N)['feature'].tolist()
    print(f"  Selected {len(top_features)} features")

    # Load data to check correlations
    print(f"\nLoading data for correlation analysis...")
    df = pd.read_parquet(data_dir / 'btc_5m_features_advanced.parquet')

    # Remove correlated features
    final_features, removed = remove_correlated_features(top_features, df, threshold=0.9)

    print(f"\nFinal feature set: {len(final_features)} features")

    if removed:
        print(f"\nRemoved correlated features ({len(removed)}):")
        for feat in removed[:10]:  # Show first 10
            print(f"  - {feat}")
        if len(removed) > 10:
            print(f"  ... and {len(removed) - 10} more")

    # Show final feature list
    print(f"\nFinal selected features:")
    for i, feat in enumerate(final_features[:30], 1):  # Show first 30
        importance = avg_importance[avg_importance['feature'] == feat]['importance'].values[0]
        print(f"  {i:2d}. {feat[:40]:<40} {importance:.6f}")

    if len(final_features) > 30:
        print(f"  ... and {len(final_features) - 30} more")

    # Save selected features
    output_file = models_dir / 'selected_features.json'
    import json
    with open(output_file, 'w') as f:
        json.dump(final_features, f, indent=2)

    print(f"\n✓ Saved selected features to: {output_file}")

    # Also save the full importance ranking
    importance_file = models_dir / 'aggregated_feature_importance.csv'
    avg_importance.to_csv(importance_file, index=False)
    print(f"✓ Saved importance ranking to: {importance_file}")

    # Summary stats
    print(f"\n{'='*60}")
    print("FEATURE SELECTION SUMMARY")
    print(f"{'='*60}")
    print(f"  Original features: 93")
    print(f"  Top N selected: {TOP_N}")
    print(f"  After correlation removal: {len(final_features)}")
    print(f"  Reduction: {(1 - len(final_features)/93)*100:.1f}%")
    print(f"\n  This should reduce overfitting and improve generalization!")


if __name__ == '__main__':
    main()
