"""
Train BTC Trend-Following Models v1
====================================
Unlike V2 models that predict "will price cross ±X% threshold",
these models predict "is a smooth, sustained trend underway?"

Target definition:
  LONG trend:  forward return > +0.3% AND max adverse excursion > -0.25%
  SHORT trend: forward return < -0.3% AND max favorable excursion < +0.25%

This filters for QUALITY directional moves — the price goes our way
without significant retracement, which is the ideal entry for a
trend-following system with tight stop losses.

Usage:
  python3 train_btc_trend_models.py
  python3 train_btc_trend_models.py --fetch-fresh   # fetch latest Binance data first
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report
)
import joblib
import os
import json
import sys
import requests
import ta
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# =========================================================================
# CONFIGURATION
# =========================================================================
MODELS_DIR = "models_btc_trend"
RESULTS_DIR = "results_btc_trend"

# Historical data source (338 days of 5-min bars with 168 features)
TRAINING_DATA_PATH = "data/BTC_2025_full_features.parquet"
# If fetching fresh data, we build features from raw OHLCV
RAW_DATA_PATH = "data/BTC_5min_8years.parquet"

# Horizons (in 5-min bars)
HORIZONS = {
    "2h": 24,
    "4h": 48,
    "6h": 72,
}

# Trend quality thresholds
# These define what counts as a "good trend" vs noise
TREND_CONFIGS = [
    # (min_return%, max_drawdown%, label)
    # Conservative: decent return, very limited drawdown
    (0.3, 0.25, "smooth"),
    # Moderate: bigger return, slightly more drawdown allowed
    (0.5, 0.30, "strong"),
]

DIRECTIONS = ['LONG', 'SHORT']

# Model hyperparameters
MODEL_PARAMS = {
    'max_iter': 500,
    'max_depth': 6,
    'learning_rate': 0.05,
    'min_samples_leaf': 50,
    'l2_regularization': 1.0,
    'max_bins': 255,
    'early_stopping': True,
    'n_iter_no_change': 20,
    'validation_fraction': 0.15,
    'random_state': 42,
}

# Train/test split: use last 20% as test (time-based, not random)
TEST_FRACTION = 0.20


def fetch_fresh_data(days: int = 365) -> pd.DataFrame:
    """Fetch fresh BTC 5-min data from Binance."""
    print(f"Fetching {days} days of BTC 5-min data from Binance...")
    
    url = "https://api.binance.com/api/v3/klines"
    all_klines = []
    end_time = None
    target_bars = days * 288  # 288 bars per day
    
    while len(all_klines) < target_bars:
        params = {'symbol': 'BTCUSDT', 'interval': '5m', 'limit': 1000}
        if end_time:
            params['endTime'] = end_time
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        batch = response.json()
        
        if not batch:
            break
        
        all_klines = batch + all_klines
        end_time = batch[0][0] - 1
        
        if len(batch) < 1000:
            break
        
        print(f"  Fetched {len(all_klines):,} bars...")
    
    klines = all_klines[-target_bars:] if len(all_klines) > target_bars else all_klines
    print(f"  Total: {len(klines):,} bars")
    
    df = pd.DataFrame(klines, columns=[
        'open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    df = df.set_index('timestamp')
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = df[col].astype(float)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators and custom features."""
    print("Adding technical indicators...")
    
    # Add all TA features
    df = ta.add_all_ta_features(
        df.copy(), open='Open', high='High', low='Low',
        close='Close', volume='Volume', fillna=True
    )
    
    # Add custom features from feature_engineering if available
    try:
        from feature_engineering import (
            add_time_features, add_price_features,
            add_daily_context_features, add_lagged_indicator_features,
            add_indicator_changes
        )
        df = add_time_features(df)
        df = add_price_features(df)
        df = add_daily_context_features(df)
        df = add_lagged_indicator_features(df, lookback_periods=[1, 2, 3, 5, 10, 20, 50])
        df = add_indicator_changes(df)
        print(f"  Added custom features: {len(df.columns)} total columns")
    except ImportError:
        print("  Warning: feature_engineering not available, using TA features only")
    
    df = df.ffill().bfill().fillna(0)
    df = df.replace([np.inf, -np.inf], 0)
    
    return df


def create_trend_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create trend continuation targets.
    
    For each horizon and quality level:
      LONG: forward_return > min_return AND max_adverse_excursion > -max_drawdown
      SHORT: forward_return < -min_return AND max_favorable_excursion < max_drawdown
    """
    print("\nCreating trend continuation targets...")
    close = df['Close'].values
    n = len(close)
    
    for horizon_name, horizon_bars in HORIZONS.items():
        # Precompute forward stats
        fwd_return = np.full(n, np.nan)
        mae_long = np.full(n, np.nan)  # max adverse excursion for LONG
        mfe_long = np.full(n, np.nan)  # max favorable excursion for LONG
        
        for i in range(n - horizon_bars):
            entry = close[i]
            window = close[i+1:i+horizon_bars+1]
            
            fwd_return[i] = (close[i + horizon_bars] / entry - 1) * 100
            mae_long[i] = (window.min() / entry - 1) * 100   # worst drop (negative for LONG)
            mfe_long[i] = (window.max() / entry - 1) * 100   # best rise (positive for LONG)
        
        for min_ret, max_dd, quality in TREND_CONFIGS:
            # LONG trend: price goes up smoothly
            target_long = np.zeros(n, dtype=int)
            mask = ~np.isnan(fwd_return)
            target_long[mask & (fwd_return > min_ret) & (mae_long > -max_dd)] = 1
            
            col_long = f"target_trend_LONG_{horizon_name}_{quality}"
            df[col_long] = target_long
            pct = target_long[mask].mean() * 100
            print(f"  {col_long}: {target_long.sum():,} positives ({pct:.1f}%)")
            
            # SHORT trend: price goes down smoothly
            target_short = np.zeros(n, dtype=int)
            target_short[mask & (fwd_return < -min_ret) & (mfe_long < max_dd)] = 1
            
            col_short = f"target_trend_SHORT_{horizon_name}_{quality}"
            df[col_short] = target_short
            pct = target_short[mask].mean() * 100
            print(f"  {col_short}: {target_short.sum():,} positives ({pct:.1f}%)")
    
    # Drop rows with NaN targets (last N bars)
    max_horizon = max(HORIZONS.values())
    df = df.iloc[:-max_horizon].copy()
    
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get feature columns (exclude targets, raw OHLCV, timestamps)."""
    exclude_patterns = [
        'target_', 'future_', 'Open', 'High', 'Low', 'Close', 'Volume',
        'open_time', 'close_time', 'quote_volume', 'trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ]
    feature_cols = [col for col in df.columns
                    if not any(pat in col for pat in exclude_patterns)]
    return feature_cols


def train_model(X_train, y_train, X_test, y_test, target_name: str) -> Tuple:
    """Train a HistGradientBoosting model for trend prediction."""
    
    # Class weights via sample_weight (HGB doesn't have class_weight)
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    
    if pos_count == 0 or neg_count == 0:
        print(f"  SKIP {target_name}: no positive/negative samples")
        return None, None
    
    weight_ratio = neg_count / pos_count
    sample_weight = np.where(y_train == 1, weight_ratio, 1.0)
    
    model = HistGradientBoostingClassifier(**MODEL_PARAMS)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'target': target_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0,
        'train_positives': int(pos_count),
        'train_negatives': int(neg_count),
        'test_positives': int(y_test.sum()),
        'test_negatives': int(len(y_test) - y_test.sum()),
        'pos_rate_train': pos_count / len(y_train),
        'pos_rate_test': y_test.sum() / len(y_test),
    }
    
    # Precision at various thresholds (critical for trading)
    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70]:
        pred_at_thresh = (y_proba >= thresh).astype(int)
        tp = ((pred_at_thresh == 1) & (y_test == 1)).sum()
        fp = ((pred_at_thresh == 1) & (y_test == 0)).sum()
        total_pred = (pred_at_thresh == 1).sum()
        prec = tp / total_pred if total_pred > 0 else 0
        metrics[f'precision_at_{int(thresh*100)}'] = prec
        metrics[f'signals_at_{int(thresh*100)}'] = int(total_pred)
    
    return model, metrics


def get_top_features(model, feature_names: List[str], n: int = 20) -> List[Tuple[str, float]]:
    """Get top N most important features."""
    try:
        importances = model.feature_importances_
    except AttributeError:
        # HGB with early stopping may not have feature_importances_
        # Use permutation importance fallback
        from sklearn.inspection import permutation_importance
        result = permutation_importance(model, 
                                         np.zeros((10, len(feature_names))), 
                                         np.zeros(10), n_repeats=1)
        # Just return empty if we can't compute
        return [("N/A", 0.0)]
    indices = np.argsort(importances)[::-1][:n]
    return [(feature_names[i], importances[i]) for i in indices]


def simulate_trading(df: pd.DataFrame, model, feature_cols: List[str],
                     direction: str, horizon_bars: int,
                     threshold: float = 0.55,
                     sl_pct: float = 0.40, 
                     trailing_activation: float = 0.45) -> Dict:
    """Simulate trading with the trend model to estimate real P&L."""
    
    X = df[feature_cols].values
    probas = model.predict_proba(X)[:, 1]
    close = df['Close'].values
    
    trades = []
    i = 0
    cooldown = 0
    
    while i < len(df) - horizon_bars:
        if cooldown > 0:
            cooldown -= 1
            i += 1
            continue
        
        if probas[i] >= threshold:
            entry_price = close[i]
            sl = entry_price * (1 - sl_pct / 100) if direction == 'LONG' else entry_price * (1 + sl_pct / 100)
            peak = entry_price
            trailing = False
            exit_price = None
            exit_reason = 'TIMEOUT'
            bars_held = 0
            
            for j in range(i + 1, min(i + horizon_bars + 1, len(df))):
                bars_held = j - i
                price = close[j]
                
                if direction == 'LONG':
                    if price > peak:
                        peak = price
                    if (peak / entry_price - 1) * 100 >= trailing_activation:
                        trailing = True
                        new_sl = peak * (1 - trailing_activation / 100)
                        sl = max(sl, new_sl)
                    if price <= sl:
                        exit_price = sl
                        exit_reason = 'TRAILING_STOP' if trailing else 'STOP_LOSS'
                        break
                else:  # SHORT
                    if price < peak:
                        peak = price
                    if (entry_price / peak - 1) * 100 >= trailing_activation:
                        trailing = True
                        new_sl = peak * (1 + trailing_activation / 100)
                        sl = min(sl, new_sl)
                    if price >= sl:
                        exit_price = sl
                        exit_reason = 'TRAILING_STOP' if trailing else 'STOP_LOSS'
                        break
            
            if exit_price is None:
                exit_price = close[min(i + horizon_bars, len(df) - 1)]
            
            if direction == 'LONG':
                pnl_pct = (exit_price / entry_price - 1) * 100
            else:
                pnl_pct = (entry_price / exit_price - 1) * 100
            
            pnl_dollar = pnl_pct / 100 * entry_price * 0.1 - 2.02  # BTC contract
            
            trades.append({
                'entry_bar': i,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pct': pnl_pct,
                'pnl_dollar': pnl_dollar,
                'exit_reason': exit_reason,
                'bars_held': bars_held,
                'probability': probas[i],
            })
            
            # Cooldown after trade
            cooldown = 6  # 30 minutes
            i = i + bars_held
        
        i += 1
    
    if not trades:
        return {'total_trades': 0, 'total_pnl': 0, 'win_rate': 0}
    
    trades_df = pd.DataFrame(trades)
    wins = (trades_df['pnl_dollar'] > 0).sum()
    losses = (trades_df['pnl_dollar'] <= 0).sum()
    
    return {
        'total_trades': len(trades),
        'wins': wins,
        'losses': losses,
        'win_rate': wins / len(trades) * 100,
        'total_pnl': trades_df['pnl_dollar'].sum(),
        'avg_win': trades_df[trades_df['pnl_dollar'] > 0]['pnl_dollar'].mean() if wins > 0 else 0,
        'avg_loss': trades_df[trades_df['pnl_dollar'] <= 0]['pnl_dollar'].mean() if losses > 0 else 0,
        'avg_bars_held': trades_df['bars_held'].mean(),
        'by_exit': trades_df['exit_reason'].value_counts().to_dict(),
    }


def main():
    fetch_fresh = '--fetch-fresh' in sys.argv
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # =====================================================================
    # 1. LOAD DATA
    # =====================================================================
    if fetch_fresh:
        print("=" * 70)
        print("FETCHING FRESH DATA FROM BINANCE")
        print("=" * 70)
        raw_df = fetch_fresh_data(days=365)
        df = add_features(raw_df)
    else:
        print("=" * 70)
        print(f"LOADING TRAINING DATA: {TRAINING_DATA_PATH}")
        print("=" * 70)
        df = pd.read_parquet(TRAINING_DATA_PATH)
        df.index = pd.to_datetime(df.index)
    
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # =====================================================================
    # 2. CREATE TARGETS
    # =====================================================================
    df = create_trend_targets(df)
    
    # =====================================================================
    # 3. PREPARE FEATURES
    # =====================================================================
    feature_cols = get_feature_columns(df)
    print(f"\nFeature columns: {len(feature_cols)}")
    
    # Time-based split
    split_idx = int(len(df) * (1 - TEST_FRACTION))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    print(f"Train: {len(train_df):,} bars ({train_df.index[0].date()} to {train_df.index[-1].date()})")
    print(f"Test:  {len(test_df):,} bars ({test_df.index[0].date()} to {test_df.index[-1].date()})")
    
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    
    # =====================================================================
    # 4. TRAIN MODELS
    # =====================================================================
    all_results = []
    
    for horizon_name, horizon_bars in HORIZONS.items():
        for min_ret, max_dd, quality in TREND_CONFIGS:
            for direction in DIRECTIONS:
                target_col = f"target_trend_{direction}_{horizon_name}_{quality}"
                model_name = f"trend_{direction}_{horizon_name}_{quality}"
                
                print(f"\n{'='*70}")
                print(f"TRAINING: {model_name}")
                print(f"  Target: {target_col}")
                print(f"  Trend: {direction} ret>{'+'if direction=='LONG' else '-'}{min_ret}%, MAE<{max_dd}%")
                print(f"{'='*70}")
                
                y_train = train_df[target_col]
                y_test = test_df[target_col]
                
                model, metrics = train_model(X_train, y_train, X_test, y_test, model_name)
                
                if model is None:
                    continue
                
                # Print key metrics
                print(f"\n  Results on TEST set:")
                print(f"    AUC-ROC:   {metrics['auc_roc']:.4f}")
                print(f"    Precision: {metrics['precision']:.4f}")
                print(f"    Recall:    {metrics['recall']:.4f}")
                print(f"    F1:        {metrics['f1']:.4f}")
                print(f"    Pos rate:  {metrics['pos_rate_test']*100:.1f}%")
                
                print(f"\n  Precision at thresholds:")
                for thresh in [50, 55, 60, 65, 70]:
                    prec = metrics[f'precision_at_{thresh}']
                    sigs = metrics[f'signals_at_{thresh}']
                    print(f"    ≥{thresh}%: precision={prec:.3f} ({sigs} signals)")
                
                # Top features
                top_feats = get_top_features(model, feature_cols, n=10)
                print(f"\n  Top features:")
                for fname, fimp in top_feats:
                    print(f"    {fimp:.4f}  {fname}")
                
                # Simulate trading on test set
                for sim_thresh in [0.55, 0.60, 0.65]:
                    sim = simulate_trading(
                        test_df, model, feature_cols,
                        direction, horizon_bars,
                        threshold=sim_thresh,
                        sl_pct=0.40, trailing_activation=0.45
                    )
                    metrics[f'sim_trades_{int(sim_thresh*100)}'] = sim['total_trades']
                    metrics[f'sim_pnl_{int(sim_thresh*100)}'] = sim['total_pnl']
                    metrics[f'sim_wr_{int(sim_thresh*100)}'] = sim['win_rate']
                    
                    if sim['total_trades'] > 0:
                        print(f"\n  Sim @{sim_thresh:.0%}: {sim['total_trades']} trades, "
                              f"WR={sim['win_rate']:.0f}%, P&L=${sim['total_pnl']:.2f}, "
                              f"Avg win=${sim.get('avg_win', 0):.2f}, "
                              f"Avg loss=${sim.get('avg_loss', 0):.2f}")
                        if sim.get('by_exit'):
                            print(f"    Exits: {sim['by_exit']}")
                
                # Add metadata
                metrics['horizon'] = horizon_name
                metrics['horizon_bars'] = horizon_bars
                metrics['direction'] = direction
                metrics['quality'] = quality
                metrics['min_return'] = min_ret
                metrics['max_drawdown'] = max_dd
                metrics['model_name'] = model_name
                
                all_results.append(metrics)
                
                # Save model
                model_path = os.path.join(MODELS_DIR, f"model_{model_name}.joblib")
                joblib.dump(model, model_path)
                print(f"\n  Saved: {model_path}")
    
    # =====================================================================
    # 5. SUMMARY & RANKINGS
    # =====================================================================
    print(f"\n\n{'='*70}")
    print("TRAINING COMPLETE - SUMMARY")
    print(f"{'='*70}")
    
    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(RESULTS_DIR, "training_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    # Rank by simulated P&L at 55% threshold
    print(f"\n{'='*70}")
    print("TOP MODELS BY SIMULATED P&L (threshold=55%)")
    print(f"{'='*70}")
    
    ranked = results_df.sort_values('sim_pnl_55', ascending=False)
    for _, row in ranked.iterrows():
        print(f"  {row['model_name']:<35} "
              f"P&L=${row['sim_pnl_55']:>+8.2f}  "
              f"Trades={int(row['sim_trades_55']):>4}  "
              f"WR={row['sim_wr_55']:>5.1f}%  "
              f"AUC={row['auc_roc']:.3f}  "
              f"Prec@55={row['precision_at_55']:.3f}")
    
    # Rank by simulated P&L at 60% threshold
    print(f"\n{'='*70}")
    print("TOP MODELS BY SIMULATED P&L (threshold=60%)")
    print(f"{'='*70}")
    
    ranked = results_df.sort_values('sim_pnl_60', ascending=False)
    for _, row in ranked.iterrows():
        print(f"  {row['model_name']:<35} "
              f"P&L=${row['sim_pnl_60']:>+8.2f}  "
              f"Trades={int(row['sim_trades_60']):>4}  "
              f"WR={row['sim_wr_60']:>5.1f}%  "
              f"AUC={row['auc_roc']:.3f}  "
              f"Prec@60={row['precision_at_60']:.3f}")
    
    # Save feature lists for each model (needed by bot)
    feature_list_path = os.path.join(MODELS_DIR, "feature_columns.json")
    with open(feature_list_path, 'w') as f:
        json.dump(feature_cols, f, indent=2)
    print(f"\nFeature columns saved to: {feature_list_path}")
    
    print(f"\n{'='*70}")
    print(f"Models saved to: {MODELS_DIR}/")
    print(f"Results saved to: {RESULTS_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
