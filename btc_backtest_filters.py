"""
BTC Backtest with RSI and Previous Day Return Filters.

Compares:
1. Baseline (no filters)
2. RSI filter (skip SHORT if RSI < 30, skip LONG if RSI > 70)
3. Prev day return filter (skip SHORT if prev_day_return < -3%)
4. Combined filters

Uses the same models and logic as btc_ensemble_bot.py
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import warnings
import ta
warnings.filterwarnings('ignore')

# BTC Models directory
BTC_MODELS_DIR = "models_btc"

# Models to test (matching btc_ensemble_bot.py)
BTC_MODELS_SHORT = [
    {'horizon': '4h', 'threshold': 0.75, 'horizon_bars': 48, 'direction': 'SHORT'},
    {'horizon': '8h', 'threshold': 0.75, 'horizon_bars': 96, 'direction': 'SHORT'},
    {'horizon': '12h', 'threshold': 0.75, 'horizon_bars': 144, 'direction': 'SHORT'},
]

BTC_MODELS_LONG = [
    {'horizon': '12h', 'threshold': 0.5, 'horizon_bars': 144, 'direction': 'LONG'},
    {'horizon': '8h', 'threshold': 0.5, 'horizon_bars': 96, 'direction': 'LONG'},
]


@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: datetime
    entry_price: float
    direction: str
    model_id: str
    horizon_bars: int
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl_pct: Optional[float] = None
    pnl_dollar: Optional[float] = None
    exit_reason: Optional[str] = None
    rsi_at_entry: Optional[float] = None
    prev_day_return: Optional[float] = None


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    probability_threshold: float = 0.55
    max_positions: int = 4
    take_profit_pct: float = 0.75  # 0.75% take profit
    stop_loss_pct: float = 1.25   # 1.25% stop loss
    contract_multiplier: float = 0.1  # MBT contract = 0.1 BTC
    commission_per_trade: float = 2.02  # IB commission


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    name: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_pnl_per_trade: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    trades: List[Trade] = field(default_factory=list)


def load_btc_data() -> pd.DataFrame:
    """Load BTC data with pre-computed features."""
    # Use the pre-computed features file that matches model training
    data_path = "data/BTC_features.parquet"
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return None
    
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features are already in the pre-computed file. Just add RSI alias if needed."""
    # The BTC_features.parquet already has all features
    # Just ensure we have the RSI column accessible for filtering
    if 'momentum_rsi' in df.columns and 'rsi_14' not in df.columns:
        df['rsi_14'] = df['momentum_rsi']
    
    return df


def get_feature_columns() -> List[str]:
    """Load feature columns from saved JSON file."""
    import json
    with open('models_btc/feature_columns.json', 'r') as f:
        return json.load(f)


def load_model(horizon: str, threshold: float, direction: str):
    """Load a trained BTC model."""
    if direction == 'SHORT':
        model_path = f"{BTC_MODELS_DIR}/model_{horizon}_{threshold}pct_short.joblib"
    else:
        model_path = f"{BTC_MODELS_DIR}/model_{horizon}_{threshold}pct.joblib"
    
    if os.path.exists(model_path):
        return joblib.load(model_path)
    
    # Try alternative path
    model_path = f"{BTC_MODELS_DIR}/model_{horizon}_{threshold}pct.joblib"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    
    print(f"Model not found: {model_path}")
    return None


def run_backtest(df: pd.DataFrame, feature_cols: List[str], 
                 models: List[Dict], cfg: BacktestConfig,
                 filter_name: str = "baseline",
                 rsi_filter: bool = False,
                 prev_day_filter: bool = False,
                 prev_day_threshold: float = -0.03) -> BacktestResult:
    """
    Run backtest with optional filters.
    
    Filters:
    - rsi_filter: Skip SHORT if RSI < 30, skip LONG if RSI > 70
    - prev_day_filter: Skip SHORT if prev_day_return < prev_day_threshold
    """
    
    # Use last 20% of data for testing (out-of-sample)
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()
    
    print(f"\nRunning backtest: {filter_name}")
    print(f"Test period: {test_df.index[0]} to {test_df.index[-1]} ({len(test_df)} bars)")
    
    # Load models and get predictions
    model_predictions = {}
    for model_cfg in models:
        model_id = f"{model_cfg['horizon']}_{model_cfg['threshold']}pct_{model_cfg['direction']}"
        model = load_model(model_cfg['horizon'], model_cfg['threshold'], model_cfg['direction'])
        if model is not None:
            try:
                X = test_df[feature_cols]
                probs = model.predict_proba(X)[:, 1]
                model_predictions[model_id] = {
                    'probs': probs,
                    'config': model_cfg
                }
                print(f"  Loaded {model_id}")
            except Exception as e:
                print(f"  Error loading {model_id}: {e}")
    
    if not model_predictions:
        print("No models loaded!")
        return None
    
    # Initialize tracking
    trades = []
    open_positions = {}  # model_id -> position info
    equity = 0
    equity_curve = [0]
    
    # Iterate through test data
    for i in range(len(test_df)):
        current_time = test_df.index[i]
        current_price = test_df['Close'].iloc[i]
        current_rsi = test_df['rsi_14'].iloc[i] if 'rsi_14' in test_df.columns else 50
        current_prev_day_return = test_df['prev_day_return'].iloc[i] if 'prev_day_return' in test_df.columns else 0
        
        # Check open positions for exit
        positions_to_close = []
        for model_id, pos in open_positions.items():
            bars_held = i - pos['entry_idx']
            
            if pos['direction'] == 'LONG':
                pnl_pct = (current_price / pos['entry_price'] - 1) * 100
            else:  # SHORT
                pnl_pct = (pos['entry_price'] / current_price - 1) * 100
            
            exit_reason = None
            
            # Check exit conditions
            if pnl_pct >= cfg.take_profit_pct:
                exit_reason = 'TAKE_PROFIT'
            elif pnl_pct <= -cfg.stop_loss_pct:
                exit_reason = 'STOP_LOSS'
            elif bars_held >= pos['horizon_bars']:
                exit_reason = 'TIMEOUT'
            
            if exit_reason:
                pnl_dollar = (pnl_pct / 100) * current_price * cfg.contract_multiplier - cfg.commission_per_trade
                
                trade = Trade(
                    entry_time=pos['entry_time'],
                    entry_price=pos['entry_price'],
                    direction=pos['direction'],
                    model_id=model_id,
                    horizon_bars=pos['horizon_bars'],
                    exit_time=current_time,
                    exit_price=current_price,
                    pnl_pct=pnl_pct,
                    pnl_dollar=pnl_dollar,
                    exit_reason=exit_reason,
                    rsi_at_entry=pos['rsi_at_entry'],
                    prev_day_return=pos['prev_day_return']
                )
                trades.append(trade)
                equity += pnl_dollar
                positions_to_close.append(model_id)
        
        # Remove closed positions
        for model_id in positions_to_close:
            del open_positions[model_id]
        
        # Check for new entry signals
        if len(open_positions) < cfg.max_positions:
            for model_id, model_data in model_predictions.items():
                # Skip if already have position for this model
                if model_id in open_positions:
                    continue
                
                # Skip if max positions reached
                if len(open_positions) >= cfg.max_positions:
                    break
                
                prob = model_data['probs'][i]
                model_cfg = model_data['config']
                direction = model_cfg['direction']
                
                # Check probability threshold
                if prob < cfg.probability_threshold:
                    continue
                
                # Apply filters
                skip_signal = False
                
                if rsi_filter:
                    if direction == 'SHORT' and current_rsi < 30:
                        skip_signal = True  # Don't short when oversold
                    elif direction == 'LONG' and current_rsi > 70:
                        skip_signal = True  # Don't long when overbought
                
                if prev_day_filter:
                    if direction == 'SHORT' and current_prev_day_return < prev_day_threshold:
                        skip_signal = True  # Don't short after big down day
                
                if skip_signal:
                    continue
                
                # Enter position
                open_positions[model_id] = {
                    'entry_time': current_time,
                    'entry_price': current_price,
                    'entry_idx': i,
                    'direction': direction,
                    'horizon_bars': model_cfg['horizon_bars'],
                    'rsi_at_entry': current_rsi,
                    'prev_day_return': current_prev_day_return
                }
        
        equity_curve.append(equity)
    
    # Close remaining positions
    for model_id, pos in open_positions.items():
        current_price = test_df['Close'].iloc[-1]
        if pos['direction'] == 'LONG':
            pnl_pct = (current_price / pos['entry_price'] - 1) * 100
        else:
            pnl_pct = (pos['entry_price'] / current_price - 1) * 100
        
        pnl_dollar = (pnl_pct / 100) * current_price * cfg.contract_multiplier - cfg.commission_per_trade
        
        trade = Trade(
            entry_time=pos['entry_time'],
            entry_price=pos['entry_price'],
            direction=pos['direction'],
            model_id=model_id,
            horizon_bars=pos['horizon_bars'],
            exit_time=test_df.index[-1],
            exit_price=current_price,
            pnl_pct=pnl_pct,
            pnl_dollar=pnl_dollar,
            exit_reason='END_OF_DATA',
            rsi_at_entry=pos['rsi_at_entry'],
            prev_day_return=pos['prev_day_return']
        )
        trades.append(trade)
        equity += pnl_dollar
    
    # Calculate metrics
    if len(trades) == 0:
        return BacktestResult(
            name=filter_name,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0,
            total_pnl=0,
            avg_pnl_per_trade=0,
            max_drawdown=0,
            sharpe_ratio=0,
            profit_factor=0,
            trades=[]
        )
    
    pnls = [t.pnl_dollar for t in trades]
    winning_trades = sum(1 for p in pnls if p > 0)
    losing_trades = sum(1 for p in pnls if p <= 0)
    
    # Max drawdown
    equity_arr = np.array(equity_curve)
    peak = np.maximum.accumulate(equity_arr)
    drawdown = peak - equity_arr
    max_drawdown = np.max(drawdown)
    
    # Sharpe ratio (simplified)
    returns = np.diff(equity_curve)
    if len(returns) > 0 and np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 288)  # 288 5-min bars per day
    else:
        sharpe = 0
    
    # Profit factor
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    return BacktestResult(
        name=filter_name,
        total_trades=len(trades),
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=winning_trades / len(trades) * 100 if len(trades) > 0 else 0,
        total_pnl=sum(pnls),
        avg_pnl_per_trade=np.mean(pnls),
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe,
        profit_factor=profit_factor,
        trades=trades
    )


def analyze_filtered_trades(baseline_result: BacktestResult, filtered_result: BacktestResult):
    """Analyze which trades were filtered out and their impact."""
    print(f"\n{'='*60}")
    print(f"Filter Analysis: {filtered_result.name}")
    print(f"{'='*60}")
    
    trades_removed = baseline_result.total_trades - filtered_result.total_trades
    print(f"Trades removed by filter: {trades_removed}")
    
    if trades_removed > 0:
        # Find which trades were filtered
        baseline_entries = {(t.entry_time, t.model_id) for t in baseline_result.trades}
        filtered_entries = {(t.entry_time, t.model_id) for t in filtered_result.trades}
        
        removed_entries = baseline_entries - filtered_entries
        removed_trades = [t for t in baseline_result.trades 
                        if (t.entry_time, t.model_id) in removed_entries]
        
        if removed_trades:
            removed_pnls = [t.pnl_dollar for t in removed_trades]
            removed_winners = sum(1 for p in removed_pnls if p > 0)
            removed_losers = sum(1 for p in removed_pnls if p <= 0)
            
            print(f"  - Winners removed: {removed_winners}")
            print(f"  - Losers removed: {removed_losers}")
            print(f"  - PnL of removed trades: ${sum(removed_pnls):.2f}")
            print(f"  - Avg RSI at entry (removed): {np.mean([t.rsi_at_entry for t in removed_trades]):.1f}")
            print(f"  - Avg prev_day_return (removed): {np.mean([t.prev_day_return for t in removed_trades])*100:.2f}%")


def main():
    print("="*60)
    print("BTC Backtest with RSI and Previous Day Return Filters")
    print("="*60)
    
    # Load data
    df = load_btc_data()
    if df is None:
        return
    
    # Add features
    print("\nAdding features...")
    df = add_features(df)
    print(f"Features added. Total columns: {len(df.columns)}")
    
    # Get feature columns
    feature_cols = get_feature_columns()
    print(f"Feature columns: {len(feature_cols)}")
    
    # Configuration
    cfg = BacktestConfig()
    
    # Combine all models
    all_models = BTC_MODELS_SHORT + BTC_MODELS_LONG
    
    # Run backtests
    results = []
    
    # 1. Baseline (no filters)
    baseline = run_backtest(df, feature_cols, all_models, cfg, 
                           filter_name="Baseline (No Filters)")
    if baseline:
        results.append(baseline)
    
    # 2. RSI filter only
    rsi_only = run_backtest(df, feature_cols, all_models, cfg,
                           filter_name="RSI Filter (skip SHORT if RSI<30)",
                           rsi_filter=True)
    if rsi_only:
        results.append(rsi_only)
    
    # 3. Previous day return filter only
    prev_day_only = run_backtest(df, feature_cols, all_models, cfg,
                                filter_name="Prev Day Filter (skip SHORT if prev_day<-3%)",
                                prev_day_filter=True,
                                prev_day_threshold=-0.03)
    if prev_day_only:
        results.append(prev_day_only)
    
    # 4. Combined filters
    combined = run_backtest(df, feature_cols, all_models, cfg,
                           filter_name="Combined Filters",
                           rsi_filter=True,
                           prev_day_filter=True,
                           prev_day_threshold=-0.03)
    if combined:
        results.append(combined)
    
    # Print comparison
    print("\n" + "="*80)
    print("BACKTEST RESULTS COMPARISON")
    print("="*80)
    print(f"{'Filter':<40} {'Trades':>8} {'Win%':>8} {'PnL':>12} {'Avg PnL':>10} {'MaxDD':>10} {'Sharpe':>8} {'PF':>8}")
    print("-"*80)
    
    for r in results:
        print(f"{r.name:<40} {r.total_trades:>8} {r.win_rate:>7.1f}% ${r.total_pnl:>10.2f} ${r.avg_pnl_per_trade:>8.2f} ${r.max_drawdown:>8.2f} {r.sharpe_ratio:>8.2f} {r.profit_factor:>7.2f}")
    
    # Analyze filtered trades
    if baseline and rsi_only:
        analyze_filtered_trades(baseline, rsi_only)
    
    if baseline and prev_day_only:
        analyze_filtered_trades(baseline, prev_day_only)
    
    if baseline and combined:
        analyze_filtered_trades(baseline, combined)
    
    # Detailed breakdown by exit reason
    print("\n" + "="*60)
    print("EXIT REASON BREAKDOWN (Baseline)")
    print("="*60)
    if baseline:
        exit_reasons = {}
        for t in baseline.trades:
            reason = t.exit_reason
            if reason not in exit_reasons:
                exit_reasons[reason] = {'count': 0, 'pnl': 0, 'winners': 0}
            exit_reasons[reason]['count'] += 1
            exit_reasons[reason]['pnl'] += t.pnl_dollar
            if t.pnl_dollar > 0:
                exit_reasons[reason]['winners'] += 1
        
        for reason, stats in exit_reasons.items():
            win_rate = stats['winners'] / stats['count'] * 100 if stats['count'] > 0 else 0
            print(f"{reason}: {stats['count']} trades, {win_rate:.1f}% win rate, ${stats['pnl']:.2f} PnL")
    
    return results


if __name__ == "__main__":
    results = main()
