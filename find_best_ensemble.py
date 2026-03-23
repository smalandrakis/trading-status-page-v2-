"""
Find the best ensemble of BTC models with minimal signal overlap.
Backtest with multiple simultaneous positions.
"""

import pandas as pd
import numpy as np
import joblib
import json
from itertools import combinations
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Constants
CAPITAL = 10000  # EUR
BTC_CONTRACT_VALUE = 0.1  # MBT micro bitcoin = 0.1 BTC
COMMISSION = 2.50  # Per trade (round trip estimate)

def load_data_and_models():
    """Load BTC data and all trained models."""
    print("Loading data and models...")
    
    df = pd.read_parquet('data/BTC_features.parquet')
    
    # Load feature columns
    with open('models_btc/feature_columns.json', 'r') as f:
        feature_cols = json.load(f)
    
    # Load results to get model list
    results = pd.read_csv('results_btc/training_results.csv')
    
    # Filter for promising models (precision > 0.5)
    good_models = results[results['precision'] > 0.5].copy()
    print(f"Found {len(good_models)} models with precision > 50%")
    
    # Load each model
    models = {}
    for _, row in good_models.iterrows():
        model_name = f"{row['horizon']}_{row['threshold']}pct"
        model_path = f"models_btc/model_{model_name}.joblib"
        try:
            models[model_name] = {
                'model': joblib.load(model_path),
                'horizon': row['horizon'],
                'threshold': row['threshold'],
                'precision': row['precision'],
                'recall': row['recall'],
                'roc_auc': row['roc_auc']
            }
        except:
            pass
    
    print(f"Loaded {len(models)} models")
    return df, feature_cols, models


def generate_signals(df, feature_cols, models, prob_threshold=0.6):
    """Generate signals for all models on test data."""
    print(f"\nGenerating signals (prob threshold: {prob_threshold})...")
    
    # Prepare features
    X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    
    # Use test period only (last 20%)
    split_idx = int(len(df) * 0.8)
    X_test = X.iloc[split_idx:]
    df_test = df.iloc[split_idx:].copy()
    
    signals = {}
    for name, model_info in models.items():
        proba = model_info['model'].predict_proba(X_test)[:, 1]
        signals[name] = (proba >= prob_threshold).astype(int)
    
    signals_df = pd.DataFrame(signals, index=df_test.index)
    
    print(f"Test period: {df_test.index[0]} to {df_test.index[-1]}")
    print(f"Test samples: {len(df_test):,}")
    
    return signals_df, df_test


def calculate_signal_overlap(signals_df):
    """Calculate pairwise signal overlap between models."""
    print("\nCalculating signal overlap matrix...")
    
    model_names = signals_df.columns.tolist()
    n = len(model_names)
    
    overlap_matrix = pd.DataFrame(index=model_names, columns=model_names, dtype=float)
    
    for i, m1 in enumerate(model_names):
        for j, m2 in enumerate(model_names):
            if i == j:
                overlap_matrix.loc[m1, m2] = 1.0
            else:
                # Jaccard similarity: intersection / union
                s1 = signals_df[m1]
                s2 = signals_df[m2]
                intersection = ((s1 == 1) & (s2 == 1)).sum()
                union = ((s1 == 1) | (s2 == 1)).sum()
                overlap_matrix.loc[m1, m2] = intersection / union if union > 0 else 0
    
    return overlap_matrix.astype(float)


def find_best_ensemble(signals_df, overlap_matrix, models, n_models=5, min_signals=100):
    """Find the best ensemble with minimal overlap."""
    print(f"\nFinding best {n_models}-model ensemble with minimal overlap...")
    
    model_names = [m for m in signals_df.columns if signals_df[m].sum() >= min_signals]
    print(f"Models with >= {min_signals} signals: {len(model_names)}")
    
    # Score each model by precision * signal_count (want high precision AND enough signals)
    model_scores = {}
    for name in model_names:
        if name in models:
            precision = models[name]['precision']
            signal_count = signals_df[name].sum()
            model_scores[name] = precision * np.log1p(signal_count)
    
    # Sort by score
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
    top_candidates = [m[0] for m in sorted_models[:20]]  # Top 20 candidates
    
    print(f"Top candidates: {top_candidates[:10]}")
    
    # Find combination with best score and low overlap
    best_combo = None
    best_score = -1
    
    for combo in combinations(top_candidates, n_models):
        # Calculate average pairwise overlap
        overlaps = []
        for i, m1 in enumerate(combo):
            for j, m2 in enumerate(combo):
                if i < j:
                    overlaps.append(overlap_matrix.loc[m1, m2])
        avg_overlap = np.mean(overlaps)
        
        # Calculate combined precision (weighted by signal count)
        total_precision = sum(models[m]['precision'] for m in combo)
        total_signals = sum(signals_df[m].sum() for m in combo)
        
        # Score: high precision, low overlap, enough signals
        score = total_precision * (1 - avg_overlap) * np.log1p(total_signals)
        
        if score > best_score:
            best_score = score
            best_combo = combo
            best_overlap = avg_overlap
    
    print(f"\nBest ensemble: {best_combo}")
    print(f"Average overlap: {best_overlap:.3f}")
    
    return list(best_combo)


def backtest_ensemble(df_test, signals_df, models, ensemble, max_positions=4):
    """Backtest the ensemble with multiple simultaneous positions."""
    print(f"\n{'='*60}")
    print(f"BACKTESTING ENSEMBLE (max {max_positions} positions)")
    print(f"{'='*60}")
    
    # Get horizon bars for each model
    horizon_map = {
        '5min': 1, '10min': 2, '15min': 3, '30min': 6, '45min': 9,
        '1h': 12, '1h30m': 18, '2h': 24, '3h': 36, '4h': 48,
        '6h': 72, '8h': 96, '12h': 144
    }
    
    positions = []  # Active positions
    trades = []  # Completed trades
    equity_curve = [CAPITAL]
    
    for i, (timestamp, row) in enumerate(df_test.iterrows()):
        current_price = row['Close']
        
        # Check for position exits
        new_positions = []
        for pos in positions:
            bars_held = i - pos['entry_idx']
            horizon_bars = horizon_map.get(pos['horizon'], 12)
            
            if bars_held >= horizon_bars:
                # Exit position
                exit_price = current_price
                pnl_pct = (exit_price / pos['entry_price'] - 1) * 100
                pnl_usd = pnl_pct / 100 * pos['entry_price'] * BTC_CONTRACT_VALUE - COMMISSION
                
                trades.append({
                    'entry_time': pos['entry_time'],
                    'exit_time': timestamp,
                    'model': pos['model'],
                    'horizon': pos['horizon'],
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'pnl_pct': pnl_pct,
                    'pnl_usd': pnl_usd,
                    'bars_held': bars_held
                })
            else:
                new_positions.append(pos)
        
        positions = new_positions
        
        # Check for new entries (if we have capacity)
        if len(positions) < max_positions:
            for model_name in ensemble:
                if len(positions) >= max_positions:
                    break
                
                # Check if this model has a signal
                if signals_df.loc[timestamp, model_name] == 1:
                    # Check if we already have a position from this model
                    model_positions = [p for p in positions if p['model'] == model_name]
                    if len(model_positions) == 0:
                        horizon = models[model_name]['horizon']
                        positions.append({
                            'model': model_name,
                            'horizon': horizon,
                            'entry_time': timestamp,
                            'entry_idx': i,
                            'entry_price': current_price
                        })
        
        # Update equity curve
        unrealized_pnl = sum(
            (current_price / p['entry_price'] - 1) * p['entry_price'] * BTC_CONTRACT_VALUE
            for p in positions
        )
        realized_pnl = sum(t['pnl_usd'] for t in trades)
        equity_curve.append(CAPITAL + realized_pnl + unrealized_pnl)
    
    # Close remaining positions at end
    final_price = df_test['Close'].iloc[-1]
    for pos in positions:
        pnl_pct = (final_price / pos['entry_price'] - 1) * 100
        pnl_usd = pnl_pct / 100 * pos['entry_price'] * BTC_CONTRACT_VALUE - COMMISSION
        trades.append({
            'entry_time': pos['entry_time'],
            'exit_time': df_test.index[-1],
            'model': pos['model'],
            'horizon': pos['horizon'],
            'entry_price': pos['entry_price'],
            'exit_price': final_price,
            'pnl_pct': pnl_pct,
            'pnl_usd': pnl_usd,
            'bars_held': len(df_test) - pos['entry_idx']
        })
    
    return pd.DataFrame(trades), equity_curve


def analyze_backtest(trades_df, equity_curve, test_days):
    """Analyze backtest results."""
    if len(trades_df) == 0:
        print("No trades!")
        return {}
    
    # Basic stats
    total_trades = len(trades_df)
    winners = trades_df[trades_df['pnl_usd'] > 0]
    losers = trades_df[trades_df['pnl_usd'] <= 0]
    
    win_rate = len(winners) / total_trades * 100
    avg_win = winners['pnl_usd'].mean() if len(winners) > 0 else 0
    avg_loss = losers['pnl_usd'].mean() if len(losers) > 0 else 0
    
    total_pnl = trades_df['pnl_usd'].sum()
    
    # Drawdown
    equity = np.array(equity_curve)
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak * 100
    max_drawdown = drawdown.max()
    
    # Trades per day
    trades_per_day = total_trades / test_days
    
    # By model
    print("\nPerformance by Model:")
    print("-" * 60)
    for model in trades_df['model'].unique():
        model_trades = trades_df[trades_df['model'] == model]
        model_wins = len(model_trades[model_trades['pnl_usd'] > 0])
        model_wr = model_wins / len(model_trades) * 100
        model_pnl = model_trades['pnl_usd'].sum()
        print(f"  {model}: {len(model_trades)} trades, {model_wr:.1f}% win rate, ${model_pnl:.2f} PnL")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total Trades:     {total_trades}")
    print(f"Trades/Day:       {trades_per_day:.2f}")
    print(f"Win Rate:         {win_rate:.1f}%")
    print(f"Avg Winner:       ${avg_win:.2f}")
    print(f"Avg Loser:        ${avg_loss:.2f}")
    print(f"Total PnL:        ${total_pnl:.2f}")
    print(f"Return:           {total_pnl/CAPITAL*100:.2f}%")
    print(f"Max Drawdown:     {max_drawdown:.2f}%")
    print(f"Final Equity:     ${equity_curve[-1]:.2f}")
    
    return {
        'total_trades': total_trades,
        'trades_per_day': trades_per_day,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'max_drawdown': max_drawdown,
        'return_pct': total_pnl/CAPITAL*100
    }


def main():
    print("=" * 70)
    print("BTC ENSEMBLE MODEL SELECTION & BACKTEST")
    print("=" * 70)
    print(f"Capital: €{CAPITAL:,}")
    print(f"Started: {datetime.now()}")
    
    # Load data
    df, feature_cols, models = load_data_and_models()
    
    # Generate signals
    signals_df, df_test = generate_signals(df, feature_cols, models, prob_threshold=0.6)
    
    # Calculate overlap
    overlap_matrix = calculate_signal_overlap(signals_df)
    
    # Find best ensemble
    ensemble = find_best_ensemble(signals_df, overlap_matrix, models, n_models=5, min_signals=50)
    
    # Print ensemble details
    print("\n" + "=" * 60)
    print("SELECTED ENSEMBLE")
    print("=" * 60)
    for model_name in ensemble:
        info = models[model_name]
        signals = signals_df[model_name].sum()
        print(f"  {model_name}: precision={info['precision']:.3f}, signals={signals}")
    
    # Calculate test period days
    test_days = (df_test.index[-1] - df_test.index[0]).days
    print(f"\nTest period: {test_days} days")
    
    # Backtest with different position limits
    results = {}
    for max_pos in [1, 2, 3, 4, 5]:
        print(f"\n{'#'*70}")
        trades_df, equity = backtest_ensemble(df_test, signals_df, models, ensemble, max_positions=max_pos)
        results[max_pos] = analyze_backtest(trades_df, equity, test_days)
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("COMPARISON: MAX POSITIONS")
    print("=" * 70)
    print(f"{'Positions':<12} {'Trades':<10} {'Trades/Day':<12} {'Win Rate':<10} {'Return':<10} {'Max DD':<10}")
    print("-" * 70)
    for max_pos, r in results.items():
        if r:
            print(f"{max_pos:<12} {r['total_trades']:<10} {r['trades_per_day']:<12.2f} {r['win_rate']:<10.1f}% {r['return_pct']:<10.2f}% {r['max_drawdown']:<10.2f}%")
    
    # Save ensemble config
    ensemble_config = {
        'models': ensemble,
        'model_details': {m: {
            'horizon': models[m]['horizon'],
            'threshold': models[m]['threshold'],
            'precision': models[m]['precision']
        } for m in ensemble},
        'prob_threshold': 0.6,
        'capital': CAPITAL,
        'backtest_results': results
    }
    
    with open('btc_ensemble_config.json', 'w') as f:
        json.dump(ensemble_config, f, indent=2, default=str)
    
    print(f"\nEnsemble config saved to btc_ensemble_config.json")


if __name__ == "__main__":
    main()
