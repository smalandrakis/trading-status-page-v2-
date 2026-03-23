"""
Analyze model trade frequency and overlap to design ensemble strategy.
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from typing import Dict, List, Tuple
from itertools import combinations

import config
from feature_engineering import get_feature_columns


def load_data():
    """Load processed data."""
    df = pd.read_parquet(f"{config.DATA_DIR}/QQQ_features.parquet")
    feature_cols = get_feature_columns(df)
    df[feature_cols] = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    return df, feature_cols


def get_model_signals(df: pd.DataFrame, feature_cols: List[str], 
                      horizon: str, threshold: float, 
                      prob_threshold: float = 0.55) -> pd.Series:
    """Get entry signals for a model."""
    model_path = f"{config.MODELS_DIR}/model_{horizon}_{threshold}pct.joblib"
    if not os.path.exists(model_path):
        return pd.Series(False, index=df.index)
    
    model = joblib.load(model_path)
    
    # Use test period only
    split_idx = int(len(df) * config.TRAIN_RATIO)
    test_df = df.iloc[split_idx:].copy()
    
    X = test_df[feature_cols]
    probs = model.predict_proba(X)[:, 1]
    
    signals = pd.Series(probs >= prob_threshold, index=test_df.index)
    return signals


def analyze_trade_frequency():
    """Analyze trade frequency for all models."""
    print("Loading data...")
    df, feature_cols = load_data()
    
    # Test period info
    split_idx = int(len(df) * config.TRAIN_RATIO)
    test_df = df.iloc[split_idx:]
    
    # Calculate trading days in test period
    test_days = len(test_df.index.normalize().unique())
    print(f"\nTest period: {test_df.index[0]} to {test_df.index[-1]}")
    print(f"Trading days in test: {test_days}")
    print(f"Total bars in test: {len(test_df)}")
    
    results = []
    all_signals = {}
    
    print("\nAnalyzing models...")
    
    for horizon in config.HORIZONS.keys():
        horizon_bars = config.HORIZONS[horizon]
        # Skip horizons > 4h (48 bars)
        if horizon_bars > 48:
            continue
            
        for threshold in config.THRESHOLDS:
            model_name = f"{horizon}_{threshold}pct"
            
            signals = get_model_signals(df, feature_cols, horizon, threshold)
            total_signals = signals.sum()
            
            if total_signals > 0:
                signals_per_day = total_signals / test_days
                
                # Store signals for overlap analysis
                all_signals[model_name] = signals
                
                results.append({
                    'model': model_name,
                    'horizon': horizon,
                    'threshold': threshold,
                    'horizon_bars': horizon_bars,
                    'total_signals': int(total_signals),
                    'signals_per_day': signals_per_day,
                    'holding_time_mins': horizon_bars * 5
                })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('signals_per_day', ascending=False)
    
    return results_df, all_signals, test_days


def calculate_overlap(signals1: pd.Series, signals2: pd.Series) -> Dict:
    """Calculate overlap between two signal series."""
    # Align indices
    common_idx = signals1.index.intersection(signals2.index)
    s1 = signals1.loc[common_idx]
    s2 = signals2.loc[common_idx]
    
    both_true = (s1 & s2).sum()
    only_s1 = (s1 & ~s2).sum()
    only_s2 = (~s1 & s2).sum()
    
    total_s1 = s1.sum()
    total_s2 = s2.sum()
    
    # Jaccard similarity (intersection / union)
    union = (s1 | s2).sum()
    jaccard = both_true / union if union > 0 else 0
    
    # Overlap coefficient (intersection / min)
    min_signals = min(total_s1, total_s2)
    overlap_coef = both_true / min_signals if min_signals > 0 else 0
    
    return {
        'both': both_true,
        'only_first': only_s1,
        'only_second': only_s2,
        'jaccard': jaccard,
        'overlap_coef': overlap_coef
    }


def analyze_overlap(all_signals: Dict[str, pd.Series], top_models: List[str]) -> pd.DataFrame:
    """Analyze overlap between top models."""
    overlap_results = []
    
    for m1, m2 in combinations(top_models, 2):
        overlap = calculate_overlap(all_signals[m1], all_signals[m2])
        overlap_results.append({
            'model_1': m1,
            'model_2': m2,
            'signals_1': all_signals[m1].sum(),
            'signals_2': all_signals[m2].sum(),
            'overlap_count': overlap['both'],
            'jaccard': overlap['jaccard'],
            'overlap_coef': overlap['overlap_coef']
        })
    
    return pd.DataFrame(overlap_results)


def find_best_ensemble(all_signals: Dict[str, pd.Series], 
                       results_df: pd.DataFrame,
                       max_signals_per_day: float = 5.0,
                       min_models: int = 3,
                       max_models: int = 5) -> List[Tuple]:
    """
    Find best ensemble with low overlap and reasonable frequency.
    """
    # Filter models by frequency
    filtered = results_df[results_df['signals_per_day'] <= max_signals_per_day]
    candidate_models = filtered['model'].tolist()
    
    print(f"\nCandidate models (≤{max_signals_per_day} signals/day): {len(candidate_models)}")
    
    best_ensembles = []
    
    for n_models in range(min_models, max_models + 1):
        best_score = -1
        best_combo = None
        
        for combo in combinations(candidate_models, n_models):
            # Calculate average pairwise overlap
            total_overlap = 0
            n_pairs = 0
            
            for m1, m2 in combinations(combo, 2):
                if m1 in all_signals and m2 in all_signals:
                    overlap = calculate_overlap(all_signals[m1], all_signals[m2])
                    total_overlap += overlap['jaccard']
                    n_pairs += 1
            
            avg_overlap = total_overlap / n_pairs if n_pairs > 0 else 1
            
            # Score: lower overlap is better (invert for maximization)
            # Also consider diversity of horizons
            horizons = set(m.split('_')[0] for m in combo)
            horizon_diversity = len(horizons) / n_models
            
            score = (1 - avg_overlap) * 0.7 + horizon_diversity * 0.3
            
            if score > best_score:
                best_score = score
                best_combo = combo
                best_avg_overlap = avg_overlap
        
        if best_combo:
            # Calculate combined signals
            combined = pd.Series(False, index=all_signals[best_combo[0]].index)
            for m in best_combo:
                combined = combined | all_signals[m]
            
            best_ensembles.append({
                'n_models': n_models,
                'models': best_combo,
                'avg_overlap': best_avg_overlap,
                'score': best_score,
                'combined_signals': combined.sum()
            })
    
    return best_ensembles


def main():
    print("="*80)
    print("MODEL TRADE FREQUENCY AND OVERLAP ANALYSIS")
    print("="*80)
    
    # Analyze frequency
    results_df, all_signals, test_days = analyze_trade_frequency()
    
    # Display frequency results
    print("\n" + "="*80)
    print("TRADE FREQUENCY BY MODEL (sorted by signals/day)")
    print("="*80)
    print(f"\n{'Model':<20} {'Horizon':<10} {'Thresh':<8} {'Signals':<10} {'Sig/Day':<10} {'Hold(min)':<10}")
    print("-"*78)
    
    for _, row in results_df.head(20).iterrows():
        print(f"{row['model']:<20} {row['horizon']:<10} {row['threshold']:<8} "
              f"{row['total_signals']:<10} {row['signals_per_day']:<10.2f} {row['holding_time_mins']:<10}")
    
    # Load backtest results for context
    backtest_path = f"{config.RESULTS_DIR}/backtest_results.csv"
    if os.path.exists(backtest_path):
        backtest_df = pd.read_csv(backtest_path)
        
        # Merge with frequency data
        merged = results_df.merge(
            backtest_df[['model', 'win_rate', 'total_pnl', 'sharpe']],
            on='model', how='left'
        )
        
        print("\n" + "="*80)
        print("TOP 10 MODELS BY SHARPE (with frequency)")
        print("="*80)
        top_sharpe = merged.nlargest(10, 'sharpe')
        print(f"\n{'Model':<20} {'Sig/Day':<10} {'WinRate':<10} {'PnL':<12} {'Sharpe':<8}")
        print("-"*70)
        for _, row in top_sharpe.iterrows():
            print(f"{row['model']:<20} {row['signals_per_day']:<10.2f} "
                  f"{row['win_rate']:<10.1f} ${row['total_pnl']:<11,.0f} {row['sharpe']:<8.2f}")
    
    # Analyze overlap between top 5 by Sharpe
    print("\n" + "="*80)
    print("OVERLAP ANALYSIS - TOP 5 MODELS BY SHARPE")
    print("="*80)
    
    top5_models = merged.nlargest(5, 'sharpe')['model'].tolist()
    print(f"\nTop 5 models: {top5_models}")
    
    overlap_df = analyze_overlap(all_signals, top5_models)
    print(f"\n{'Model 1':<20} {'Model 2':<20} {'Overlap':<10} {'Jaccard':<10}")
    print("-"*60)
    for _, row in overlap_df.iterrows():
        print(f"{row['model_1']:<20} {row['model_2']:<20} "
              f"{row['overlap_count']:<10} {row['jaccard']:<10.3f}")
    
    # Find low-frequency models (≤2 trades/day)
    print("\n" + "="*80)
    print("LOW FREQUENCY MODELS (≤2 signals/day)")
    print("="*80)
    
    low_freq = merged[merged['signals_per_day'] <= 2].sort_values('sharpe', ascending=False)
    print(f"\n{'Model':<20} {'Sig/Day':<10} {'WinRate':<10} {'PnL':<12} {'Sharpe':<8}")
    print("-"*70)
    for _, row in low_freq.head(15).iterrows():
        print(f"{row['model']:<20} {row['signals_per_day']:<10.2f} "
              f"{row['win_rate']:<10.1f} ${row['total_pnl']:<11,.0f} {row['sharpe']:<8.2f}")
    
    # Find best ensemble with low overlap
    print("\n" + "="*80)
    print("BEST ENSEMBLE COMBINATIONS (low overlap, ≤3 signals/day each)")
    print("="*80)
    
    ensembles = find_best_ensemble(all_signals, results_df, max_signals_per_day=3.0)
    
    for ens in ensembles:
        print(f"\n{ens['n_models']}-Model Ensemble:")
        print(f"  Models: {', '.join(ens['models'])}")
        print(f"  Avg Overlap (Jaccard): {ens['avg_overlap']:.3f}")
        print(f"  Combined Signals: {ens['combined_signals']}")
        print(f"  Combined Signals/Day: {ens['combined_signals']/test_days:.2f}")
    
    # Recommend best ensemble
    print("\n" + "="*80)
    print("RECOMMENDED ENSEMBLE")
    print("="*80)
    
    # Find ensemble with best balance
    best_ens = min(ensembles, key=lambda x: x['avg_overlap'])
    
    print(f"\nRecommended: {best_ens['n_models']}-model ensemble")
    print(f"Models: {', '.join(best_ens['models'])}")
    print(f"Average Overlap: {best_ens['avg_overlap']:.1%}")
    print(f"Combined Signals/Day: {best_ens['combined_signals']/test_days:.2f}")
    
    # Get performance of ensemble models
    print("\nIndividual Model Performance:")
    for model in best_ens['models']:
        model_data = merged[merged['model'] == model].iloc[0]
        print(f"  {model}: WR={model_data['win_rate']:.1f}%, "
              f"PnL=${model_data['total_pnl']:,.0f}, Sharpe={model_data['sharpe']:.2f}")


if __name__ == "__main__":
    main()
