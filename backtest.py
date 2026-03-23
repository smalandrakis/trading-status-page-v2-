"""
Backtest the trained models with realistic trading assumptions.
Includes transaction costs, slippage, and position management.
"""

import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

import config
from feature_engineering import prepare_features, get_feature_columns


@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: datetime
    entry_price: float
    direction: str  # 'long' or 'short'
    size: int
    horizon: str
    threshold: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    exit_reason: Optional[str] = None  # 'target', 'stop', 'timeout'


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 100000.0
    position_size: int = 100  # shares per trade
    commission_per_share: float = 0.005  # $0.005 per share
    slippage_pct: float = 0.01  # 0.01% slippage
    max_positions: int = 1  # Max concurrent positions
    probability_threshold: float = 0.6  # Min probability to enter trade
    stop_loss_pct: float = 2.0  # Stop loss percentage
    use_stop_loss: bool = True


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    model_name: str
    horizon: str
    threshold: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_pnl_per_trade: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    avg_holding_time: float  # in minutes
    trades: List[Trade] = field(default_factory=list)


def load_model(horizon: str, threshold: float):
    """Load a trained model."""
    model_path = f"{config.MODELS_DIR}/model_{horizon}_{threshold}pct.joblib"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


def calculate_transaction_costs(price: float, size: int, cfg: BacktestConfig) -> float:
    """Calculate total transaction costs including commission and slippage."""
    commission = cfg.commission_per_share * size * 2  # Entry and exit
    slippage = price * size * (cfg.slippage_pct / 100) * 2  # Entry and exit
    return commission + slippage


def run_backtest(df: pd.DataFrame, feature_cols: List[str], 
                 horizon: str, threshold: float,
                 cfg: BacktestConfig = None) -> BacktestResult:
    """
    Run backtest for a specific model.
    """
    if cfg is None:
        cfg = BacktestConfig()
    
    # Load model
    model = load_model(horizon, threshold)
    if model is None:
        print(f"Model not found for {horizon} / {threshold}%")
        return None
    
    # Get horizon in bars
    horizon_bars = config.HORIZONS.get(horizon, 12)
    
    # Prepare data - use only test period (last 30%)
    split_idx = int(len(df) * config.TRAIN_RATIO)
    test_df = df.iloc[split_idx:].copy()
    
    # Get predictions
    X_test = test_df[feature_cols]
    probabilities = model.predict_proba(X_test)[:, 1]
    test_df['pred_prob'] = probabilities
    
    # Initialize tracking
    trades = []
    equity_curve = [cfg.initial_capital]
    current_position = None
    capital = cfg.initial_capital
    
    # Iterate through test data
    for i in range(len(test_df) - horizon_bars):
        current_time = test_df.index[i]
        current_price = test_df['Close'].iloc[i]
        pred_prob = test_df['pred_prob'].iloc[i]
        
        # Check if we have an open position
        if current_position is not None:
            bars_held = i - current_position['entry_idx']
            current_pnl_pct = (current_price / current_position['entry_price'] - 1) * 100
            
            if current_position['direction'] == 'short':
                current_pnl_pct = -current_pnl_pct
            
            # Check exit conditions
            exit_reason = None
            
            # 1. Target reached
            if current_pnl_pct >= threshold:
                exit_reason = 'target'
            
            # 2. Stop loss hit
            elif cfg.use_stop_loss and current_pnl_pct <= -cfg.stop_loss_pct:
                exit_reason = 'stop'
            
            # 3. Timeout (max holding period reached)
            elif bars_held >= horizon_bars:
                exit_reason = 'timeout'
            
            if exit_reason:
                # Close position
                exit_price = current_price
                
                # Calculate PnL
                if current_position['direction'] == 'long':
                    gross_pnl = (exit_price - current_position['entry_price']) * cfg.position_size
                else:
                    gross_pnl = (current_position['entry_price'] - exit_price) * cfg.position_size
                
                # Subtract transaction costs
                costs = calculate_transaction_costs(current_position['entry_price'], cfg.position_size, cfg)
                net_pnl = gross_pnl - costs
                
                # Record trade
                trade = Trade(
                    entry_time=current_position['entry_time'],
                    entry_price=current_position['entry_price'],
                    direction=current_position['direction'],
                    size=cfg.position_size,
                    horizon=horizon,
                    threshold=threshold,
                    exit_time=current_time,
                    exit_price=exit_price,
                    pnl=net_pnl,
                    exit_reason=exit_reason
                )
                trades.append(trade)
                
                capital += net_pnl
                current_position = None
        
        # Check for new entry signal (only if no position)
        if current_position is None and pred_prob >= cfg.probability_threshold:
            # Enter long position
            current_position = {
                'entry_time': current_time,
                'entry_price': current_price,
                'entry_idx': i,
                'direction': 'long',
                'size': cfg.position_size
            }
        
        equity_curve.append(capital)
    
    # Close any remaining position at end
    if current_position is not None:
        exit_price = test_df['Close'].iloc[-1]
        if current_position['direction'] == 'long':
            gross_pnl = (exit_price - current_position['entry_price']) * cfg.position_size
        else:
            gross_pnl = (current_position['entry_price'] - exit_price) * cfg.position_size
        
        costs = calculate_transaction_costs(current_position['entry_price'], cfg.position_size, cfg)
        net_pnl = gross_pnl - costs
        
        trade = Trade(
            entry_time=current_position['entry_time'],
            entry_price=current_position['entry_price'],
            direction=current_position['direction'],
            size=cfg.position_size,
            horizon=horizon,
            threshold=threshold,
            exit_time=test_df.index[-1],
            exit_price=exit_price,
            pnl=net_pnl,
            exit_reason='end_of_data'
        )
        trades.append(trade)
        capital += net_pnl
    
    # Calculate metrics
    if len(trades) == 0:
        return BacktestResult(
            model_name=f"{horizon}_{threshold}pct",
            horizon=horizon,
            threshold=threshold,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0,
            total_pnl=0,
            avg_pnl_per_trade=0,
            max_drawdown=0,
            sharpe_ratio=0,
            profit_factor=0,
            avg_holding_time=0,
            trades=[]
        )
    
    pnls = [t.pnl for t in trades]
    winning_trades = sum(1 for p in pnls if p > 0)
    losing_trades = sum(1 for p in pnls if p <= 0)
    
    # Calculate max drawdown
    equity = np.array(equity_curve)
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    max_drawdown = np.max(drawdown) * 100
    
    # Calculate Sharpe ratio (annualized, assuming 5-min bars)
    returns = np.diff(equity_curve) / equity_curve[:-1]
    if len(returns) > 0 and np.std(returns) > 0:
        # ~78 bars per day, ~252 trading days
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(78 * 252)
    else:
        sharpe = 0
    
    # Calculate profit factor
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Average holding time
    holding_times = []
    for t in trades:
        if t.exit_time and t.entry_time:
            delta = (t.exit_time - t.entry_time).total_seconds() / 60
            holding_times.append(delta)
    avg_holding = np.mean(holding_times) if holding_times else 0
    
    return BacktestResult(
        model_name=f"{horizon}_{threshold}pct",
        horizon=horizon,
        threshold=threshold,
        total_trades=len(trades),
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=winning_trades / len(trades) * 100,
        total_pnl=sum(pnls),
        avg_pnl_per_trade=np.mean(pnls),
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe,
        profit_factor=profit_factor,
        avg_holding_time=avg_holding,
        trades=trades
    )


def run_all_backtests(df: pd.DataFrame, feature_cols: List[str],
                      cfg: BacktestConfig = None) -> pd.DataFrame:
    """Run backtests for all trained models."""
    
    if cfg is None:
        cfg = BacktestConfig()
    
    results = []
    
    print("\n" + "="*80)
    print("RUNNING BACKTESTS")
    print("="*80)
    print(f"Probability threshold: {cfg.probability_threshold}")
    print(f"Position size: {cfg.position_size} shares")
    print(f"Stop loss: {cfg.stop_loss_pct}%" if cfg.use_stop_loss else "Stop loss: Disabled")
    print("="*80)
    
    for horizon in config.HORIZONS.keys():
        for threshold in config.THRESHOLDS:
            print(f"\nBacktesting: {horizon} / {threshold}%")
            
            result = run_backtest(df, feature_cols, horizon, threshold, cfg)
            
            if result and result.total_trades > 0:
                print(f"  Trades: {result.total_trades}")
                print(f"  Win Rate: {result.win_rate:.1f}%")
                print(f"  Total PnL: ${result.total_pnl:,.2f}")
                print(f"  Sharpe: {result.sharpe_ratio:.2f}")
                print(f"  Max DD: {result.max_drawdown:.1f}%")
                
                results.append({
                    'model': result.model_name,
                    'horizon': result.horizon,
                    'threshold': result.threshold,
                    'trades': result.total_trades,
                    'win_rate': result.win_rate,
                    'total_pnl': result.total_pnl,
                    'avg_pnl': result.avg_pnl_per_trade,
                    'max_drawdown': result.max_drawdown,
                    'sharpe': result.sharpe_ratio,
                    'profit_factor': result.profit_factor,
                    'avg_hold_mins': result.avg_holding_time
                })
            else:
                print(f"  No trades generated")
    
    results_df = pd.DataFrame(results)
    return results_df


def analyze_backtest_results(results_df: pd.DataFrame) -> None:
    """Analyze and display backtest results."""
    
    if len(results_df) == 0:
        print("\nNo backtest results to analyze.")
        return
    
    print("\n" + "="*80)
    print("BACKTEST RESULTS SUMMARY")
    print("="*80)
    
    # Top performers by total PnL
    print("\n📊 Top 10 Models by Total PnL:")
    top_pnl = results_df.nlargest(10, 'total_pnl')[
        ['horizon', 'threshold', 'trades', 'win_rate', 'total_pnl', 'sharpe', 'max_drawdown']
    ]
    print(top_pnl.to_string(index=False))
    
    # Top performers by Sharpe ratio
    print("\n📊 Top 10 Models by Sharpe Ratio:")
    top_sharpe = results_df.nlargest(10, 'sharpe')[
        ['horizon', 'threshold', 'trades', 'win_rate', 'total_pnl', 'sharpe', 'max_drawdown']
    ]
    print(top_sharpe.to_string(index=False))
    
    # Top performers by win rate (with min trades)
    print("\n📊 Top 10 Models by Win Rate (min 50 trades):")
    filtered = results_df[results_df['trades'] >= 50]
    if len(filtered) > 0:
        top_wr = filtered.nlargest(10, 'win_rate')[
            ['horizon', 'threshold', 'trades', 'win_rate', 'total_pnl', 'sharpe', 'profit_factor']
        ]
        print(top_wr.to_string(index=False))
    else:
        print("No models with >= 50 trades")
    
    # Best risk-adjusted (positive PnL, good Sharpe, low drawdown)
    print("\n🎯 Best Risk-Adjusted Models (PnL > 0, Sharpe > 0.5, DD < 10%):")
    best = results_df[
        (results_df['total_pnl'] > 0) & 
        (results_df['sharpe'] > 0.5) & 
        (results_df['max_drawdown'] < 10)
    ].sort_values('sharpe', ascending=False)
    
    if len(best) > 0:
        print(best[['horizon', 'threshold', 'trades', 'win_rate', 'total_pnl', 'sharpe', 'max_drawdown']].to_string(index=False))
    else:
        print("No models meet all criteria. Relaxing constraints...")
        best = results_df[results_df['total_pnl'] > 0].nlargest(5, 'sharpe')
        if len(best) > 0:
            print(best[['horizon', 'threshold', 'trades', 'win_rate', 'total_pnl', 'sharpe', 'max_drawdown']].to_string(index=False))


def main():
    print("="*80)
    print("BACKTEST - MULTI-HORIZON PRICE MOVEMENT PREDICTION")
    print("="*80)
    print(f"Started at: {datetime.now()}")
    
    # Load processed data
    print("\n📂 Loading processed data...")
    df = pd.read_parquet(f"{config.DATA_DIR}/QQQ_features.parquet")
    feature_cols = get_feature_columns(df)
    
    # Handle NaN and inf
    df[feature_cols] = df[feature_cols].fillna(0)
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], 0)
    
    print(f"Loaded {len(df)} rows")
    print(f"Features: {len(feature_cols)}")
    
    # Configure backtest
    cfg = BacktestConfig(
        initial_capital=100000,
        position_size=100,
        commission_per_share=0.005,
        slippage_pct=0.01,
        probability_threshold=0.55,  # Lower threshold to get more trades
        stop_loss_pct=2.0,
        use_stop_loss=True
    )
    
    # Run all backtests
    results_df = run_all_backtests(df, feature_cols, cfg)
    
    # Analyze results
    analyze_backtest_results(results_df)
    
    # Save results
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    results_path = f"{config.RESULTS_DIR}/backtest_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n✅ Results saved to {results_path}")
    
    print(f"\n✅ Backtest complete at: {datetime.now()}")


if __name__ == "__main__":
    main()
