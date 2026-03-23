"""
Daily Validation Script - Compare Live Trading vs Backtest
Run this at end of each trading day to validate system performance.

Usage:
    python scripts/daily_validation.py --date 2025-12-23
    python scripts/daily_validation.py  # defaults to today
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

# Paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_DIR)


def load_signal_logs(date: str, bot: str = 'mnq') -> pd.DataFrame:
    """Load signal logs for a specific date."""
    log_dir = 'signal_logs'
    pattern = f"{bot}_signals_{date}"
    
    files = [f for f in os.listdir(log_dir) if pattern in f]
    if not files:
        print(f"No signal logs found for {bot} on {date}")
        return pd.DataFrame()
    
    dfs = []
    for f in files:
        df = pd.read_csv(os.path.join(log_dir, f))
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def run_backtest_for_date(date: str, bot: str = 'mnq') -> Dict:
    """Run backtest for a specific date and return metrics."""
    
    # Load data
    if bot == 'mnq':
        df = pd.read_parquet('data/QQQ_features.parquet')
        model_dir = 'models_mnq_v2'
    else:
        df = pd.read_parquet('data/BTC_features.parquet')
        model_dir = 'models_btc_v2'
    
    with open(f'{model_dir}/feature_columns.json', 'r') as f:
        feature_cols = json.load(f)
    
    # Filter to specific date
    target_date = pd.to_datetime(date)
    df = df[df.index.normalize() == target_date]
    
    if len(df) == 0:
        return {'error': f'No data for {date}'}
    
    # Filter to market hours
    df = df[(df.index.hour > 9) | ((df.index.hour == 9) & (df.index.minute >= 30))]
    df = df[df.index.hour < 16]
    
    # Load models
    models = {}
    for suffix in ['2h_0.5pct', '4h_0.5pct', '2h_0.5pct_SHORT', '4h_0.5pct_SHORT']:
        model_path = f'{model_dir}/model_{suffix}.joblib'
        if os.path.exists(model_path):
            models[suffix] = joblib.load(model_path)
    
    # Get predictions
    X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    probs = {}
    for name, model in models.items():
        probs[name] = model.predict_proba(X)[:, 1]
    
    # Count signals at 65% threshold
    PROB_THRESHOLD = 0.65
    signals = {
        'long': sum((probs.get('2h_0.5pct', np.zeros(len(df))) >= PROB_THRESHOLD) | 
                    (probs.get('4h_0.5pct', np.zeros(len(df))) >= PROB_THRESHOLD)),
        'short': sum((probs.get('2h_0.5pct_SHORT', np.zeros(len(df))) >= PROB_THRESHOLD) | 
                     (probs.get('4h_0.5pct_SHORT', np.zeros(len(df))) >= PROB_THRESHOLD)),
    }
    signals['total'] = signals['long'] + signals['short']
    
    # Simulate trades (simplified)
    trades = simulate_trades(df, probs, feature_cols)
    
    return {
        'date': date,
        'bars': len(df),
        'signals': signals,
        'trades': trades,
    }


def simulate_trades(df: pd.DataFrame, probs: Dict, feature_cols: List[str]) -> Dict:
    """Simulate trading for a single day."""
    
    PROB_THRESHOLD = 0.65
    STOP_LOSS = 0.75 / 100
    TARGET = 0.5 / 100
    TRAILING_STOP = 0.15 / 100
    TRAILING_ACTIVATION = 0.15 / 100
    MAX_LONG = 2
    MAX_SHORT = 2
    
    trades = []
    long_positions = {}
    short_positions = {}
    pos_id = 0
    
    for i in range(len(df)):
        current_price = df['Close'].iloc[i]
        
        # Check exits for LONG positions
        to_remove = []
        for pid, pos in long_positions.items():
            if current_price > pos['peak']:
                pos['peak'] = current_price
            
            profit_pct = current_price / pos['entry'] - 1
            
            # Activate trailing
            if not pos['trailing_active'] and profit_pct >= TRAILING_ACTIVATION:
                pos['trailing_active'] = True
                pos['trailing_stop'] = pos['peak'] * (1 - TRAILING_STOP)
            
            if pos['trailing_active']:
                pos['trailing_stop'] = max(pos['trailing_stop'], pos['peak'] * (1 - TRAILING_STOP))
            
            # Check exits
            exit_reason = None
            if current_price >= pos['entry'] * (1 + TARGET):
                exit_reason = 'TARGET'
            elif pos['trailing_active'] and current_price <= pos['trailing_stop']:
                exit_reason = 'TRAILING'
            elif current_price <= pos['entry'] * (1 - STOP_LOSS):
                exit_reason = 'STOP_LOSS'
            elif i - pos['bar'] >= pos['timeout']:
                exit_reason = 'TIMEOUT'
            
            if exit_reason:
                pnl = (current_price / pos['entry'] - 1) * 100
                trades.append({
                    'direction': 'LONG',
                    'entry': pos['entry'],
                    'exit': current_price,
                    'pnl_pct': pnl,
                    'exit_reason': exit_reason,
                })
                to_remove.append(pid)
        
        for pid in to_remove:
            del long_positions[pid]
        
        # Check exits for SHORT positions
        to_remove = []
        for pid, pos in short_positions.items():
            if current_price < pos['trough']:
                pos['trough'] = current_price
            
            profit_pct = pos['entry'] / current_price - 1
            
            if not pos['trailing_active'] and profit_pct >= TRAILING_ACTIVATION:
                pos['trailing_active'] = True
                pos['trailing_stop'] = pos['trough'] * (1 + TRAILING_STOP)
            
            if pos['trailing_active']:
                pos['trailing_stop'] = min(pos['trailing_stop'], pos['trough'] * (1 + TRAILING_STOP))
            
            exit_reason = None
            if current_price <= pos['entry'] * (1 - TARGET):
                exit_reason = 'TARGET'
            elif pos['trailing_active'] and current_price >= pos['trailing_stop']:
                exit_reason = 'TRAILING'
            elif current_price >= pos['entry'] * (1 + STOP_LOSS):
                exit_reason = 'STOP_LOSS'
            elif i - pos['bar'] >= pos['timeout']:
                exit_reason = 'TIMEOUT'
            
            if exit_reason:
                pnl = (pos['entry'] / current_price - 1) * 100
                trades.append({
                    'direction': 'SHORT',
                    'entry': pos['entry'],
                    'exit': current_price,
                    'pnl_pct': pnl,
                    'exit_reason': exit_reason,
                })
                to_remove.append(pid)
        
        for pid in to_remove:
            del short_positions[pid]
        
        # Check entries
        # LONG
        for model in ['2h_0.5pct', '4h_0.5pct']:
            if model in probs and len(long_positions) < MAX_LONG:
                if probs[model][i] >= PROB_THRESHOLD:
                    timeout = 24 * 2 if '2h' in model else 48 * 2
                    long_positions[pos_id] = {
                        'entry': current_price,
                        'bar': i,
                        'timeout': timeout,
                        'peak': current_price,
                        'trailing_active': False,
                        'trailing_stop': 0,
                    }
                    pos_id += 1
                    break  # Only one entry per bar
        
        # SHORT
        for model in ['2h_0.5pct_SHORT', '4h_0.5pct_SHORT']:
            if model in probs and len(short_positions) < MAX_SHORT:
                if probs[model][i] >= PROB_THRESHOLD:
                    timeout = 24 * 2 if '2h' in model else 48 * 2
                    short_positions[pos_id] = {
                        'entry': current_price,
                        'bar': i,
                        'timeout': timeout,
                        'trough': current_price,
                        'trailing_active': False,
                        'trailing_stop': float('inf'),
                    }
                    pos_id += 1
                    break
    
    # Calculate summary
    if not trades:
        return {'count': 0, 'wins': 0, 'losses': 0, 'win_rate': 0, 'total_pnl': 0}
    
    trades_df = pd.DataFrame(trades)
    wins = len(trades_df[trades_df['pnl_pct'] > 0])
    losses = len(trades_df[trades_df['pnl_pct'] <= 0])
    
    return {
        'count': len(trades),
        'wins': wins,
        'losses': losses,
        'win_rate': wins / len(trades) * 100 if trades else 0,
        'total_pnl': trades_df['pnl_pct'].sum(),
        'avg_win': trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].mean() if wins > 0 else 0,
        'avg_loss': trades_df[trades_df['pnl_pct'] <= 0]['pnl_pct'].mean() if losses > 0 else 0,
        'exit_reasons': trades_df['exit_reason'].value_counts().to_dict(),
    }


def parse_bot_logs(date: str, bot: str = 'mnq') -> Dict:
    """Parse bot logs to extract actual trading activity."""
    
    log_file = f'logs/{bot}_bot.log'
    if not os.path.exists(log_file):
        return {'error': f'Log file not found: {log_file}'}
    
    # Read log file
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Filter to target date
    target_date = date
    day_lines = [l for l in lines if target_date in l]
    
    # Count signals and trades
    signals = len([l for l in day_lines if 'SIGNAL:' in l])
    entries = len([l for l in day_lines if 'ENTERED' in l or 'Entered' in l])
    exits = len([l for l in day_lines if 'EXITED' in l or 'Exited' in l or 'closed' in l.lower()])
    
    return {
        'signals': signals,
        'entries': entries,
        'exits': exits,
        'log_lines': len(day_lines),
    }


def generate_report(date: str) -> str:
    """Generate validation report for a specific date."""
    
    report = []
    report.append("=" * 70)
    report.append(f"DAILY VALIDATION REPORT - {date}")
    report.append("=" * 70)
    
    for bot in ['mnq', 'btc']:
        report.append(f"\n{'='*35}")
        report.append(f"{bot.upper()} BOT")
        report.append(f"{'='*35}")
        
        # Backtest results
        backtest = run_backtest_for_date(date, bot)
        if 'error' in backtest:
            report.append(f"Backtest: {backtest['error']}")
        else:
            report.append(f"\nBacktest Results:")
            report.append(f"  Bars analyzed: {backtest['bars']}")
            report.append(f"  Signals: LONG={backtest['signals']['long']}, SHORT={backtest['signals']['short']}")
            report.append(f"  Trades: {backtest['trades']['count']}")
            report.append(f"  Win rate: {backtest['trades']['win_rate']:.1f}%")
            report.append(f"  Total P&L: {backtest['trades']['total_pnl']:.2f}%")
            if backtest['trades'].get('exit_reasons'):
                report.append(f"  Exit reasons: {backtest['trades']['exit_reasons']}")
        
        # Live results from logs
        live = parse_bot_logs(date, bot)
        if 'error' in live:
            report.append(f"\nLive: {live['error']}")
        else:
            report.append(f"\nLive Bot Activity:")
            report.append(f"  Log lines: {live['log_lines']}")
            report.append(f"  Signals: {live['signals']}")
            report.append(f"  Entries: {live['entries']}")
            report.append(f"  Exits: {live['exits']}")
        
        # Comparison
        if 'error' not in backtest and 'error' not in live:
            report.append(f"\nComparison:")
            signal_match = "✅" if abs(backtest['signals']['total'] - live['signals']) < backtest['signals']['total'] * 0.2 else "⚠️"
            report.append(f"  Signals: Backtest={backtest['signals']['total']}, Live={live['signals']} {signal_match}")
    
    report.append("\n" + "=" * 70)
    report.append("END OF REPORT")
    report.append("=" * 70)
    
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description='Daily validation script')
    parser.add_argument('--date', type=str, default=None,
                        help='Date to validate (YYYY-MM-DD), defaults to today')
    parser.add_argument('--bot', type=str, default='both',
                        choices=['mnq', 'btc', 'both'],
                        help='Which bot to validate')
    
    args = parser.parse_args()
    
    if args.date is None:
        args.date = datetime.now().strftime('%Y-%m-%d')
    
    print(generate_report(args.date))
    
    # Save report
    report_dir = 'docs/validation_reports'
    os.makedirs(report_dir, exist_ok=True)
    report_file = f'{report_dir}/validation_{args.date}.txt'
    
    with open(report_file, 'w') as f:
        f.write(generate_report(args.date))
    
    print(f"\nReport saved to: {report_file}")


if __name__ == "__main__":
    main()
