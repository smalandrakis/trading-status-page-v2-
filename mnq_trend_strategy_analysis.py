#!/usr/bin/env python3
"""
MNQ Trend-Following Strategy Analysis

1. Identify trends in past data using multiple methods
2. Calculate noise standard deviation for proper SL sizing
3. Find optimal R:R ratio (1:2 or 1:3) that avoids noise-triggered stops
4. Backtest on 30 days and validate on historical months
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

QQQ_PARQUET = "data/QQQ_features.parquet"

def load_data():
    """Load QQQ parquet data."""
    df = pd.read_parquet(QQQ_PARQUET)
    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    return df


def identify_trends(df, lookback=20):
    """
    Identify trends using multiple methods:
    1. EMA crossover (12/26)
    2. ADX strength
    3. Price vs moving averages
    4. Higher highs/higher lows pattern
    """
    df = df.copy()
    
    # EMA crossover
    df['ema_12'] = df['Close'].ewm(span=12).mean()
    df['ema_26'] = df['Close'].ewm(span=26).mean()
    df['ema_50'] = df['Close'].ewm(span=50).mean()
    df['ema_trend'] = np.where(df['ema_12'] > df['ema_26'], 1, -1)
    
    # Price vs EMAs
    df['above_ema50'] = (df['Close'] > df['ema_50']).astype(int)
    
    # ADX for trend strength (use existing if available)
    if 'trend_adx' in df.columns:
        df['strong_trend'] = df['trend_adx'] > 25
    else:
        df['strong_trend'] = True
    
    # Higher highs / higher lows detection
    df['hh'] = (df['High'] > df['High'].shift(1)) & (df['High'].shift(1) > df['High'].shift(2))
    df['hl'] = (df['Low'] > df['Low'].shift(1)) & (df['Low'].shift(1) > df['Low'].shift(2))
    df['lh'] = (df['High'] < df['High'].shift(1)) & (df['High'].shift(1) < df['High'].shift(2))
    df['ll'] = (df['Low'] < df['Low'].shift(1)) & (df['Low'].shift(1) < df['Low'].shift(2))
    
    # Uptrend: HH + HL pattern
    df['uptrend_pattern'] = df['hh'] & df['hl']
    # Downtrend: LH + LL pattern
    df['downtrend_pattern'] = df['lh'] & df['ll']
    
    # ROC for momentum
    df['roc_10'] = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10) * 100
    df['roc_20'] = (df['Close'] - df['Close'].shift(20)) / df['Close'].shift(20) * 100
    
    # Combined trend signal
    # LONG: EMA bullish + price above EMA50 + positive ROC
    df['trend_long'] = (
        (df['ema_trend'] == 1) & 
        (df['above_ema50'] == 1) & 
        (df['roc_10'] > 0.1)
    )
    
    # SHORT: EMA bearish + price below EMA50 + negative ROC
    df['trend_short'] = (
        (df['ema_trend'] == -1) & 
        (df['above_ema50'] == 0) & 
        (df['roc_10'] < -0.1)
    )
    
    return df


def calculate_noise_stats(df, window=20):
    """
    Calculate noise statistics to size stop losses properly.
    
    Noise = intra-bar volatility that shouldn't trigger stops
    """
    df = df.copy()
    
    # ATR-based noise
    df['atr'] = df['volatility_atr'] if 'volatility_atr' in df.columns else (
        df['High'] - df['Low']
    ).rolling(14).mean()
    
    # ATR as percentage of price
    df['atr_pct'] = df['atr'] / df['Close'] * 100
    
    # Bar range (high-low) as percentage
    df['bar_range_pct'] = (df['High'] - df['Low']) / df['Close'] * 100
    
    # Calculate noise statistics
    noise_stats = {
        'atr_pct_mean': df['atr_pct'].tail(window).mean(),
        'atr_pct_std': df['atr_pct'].tail(window).std(),
        'atr_pct_max': df['atr_pct'].tail(window).max(),
        'bar_range_mean': df['bar_range_pct'].tail(window).mean(),
        'bar_range_std': df['bar_range_pct'].tail(window).std(),
        'bar_range_max': df['bar_range_pct'].tail(window).max(),
    }
    
    # Recommended SL should be > 2 * noise std to avoid noise triggers
    noise_stats['recommended_sl_min'] = noise_stats['atr_pct_mean'] + 2 * noise_stats['atr_pct_std']
    
    return df, noise_stats


def backtest_trend_strategy(df, sl_pct, tp_pct, direction='LONG', cooldown_bars=10, 
                            max_hold_bars=48, use_trailing=False, trailing_pct=0.3):
    """
    Backtest trend-following strategy with given SL/TP.
    
    Entry: When trend signal fires
    Exit: SL, TP, or timeout
    """
    trades = []
    last_trade_bar = -cooldown_bars
    
    signal_col = 'trend_long' if direction == 'LONG' else 'trend_short'
    
    for i in range(50, len(df) - max_hold_bars):
        if i - last_trade_bar < cooldown_bars:
            continue
        
        if not df[signal_col].iloc[i]:
            continue
        
        entry_price = df['Close'].iloc[i]
        entry_time = df.index[i]
        
        if direction == 'LONG':
            target_price = entry_price * (1 + tp_pct / 100)
            stop_price = entry_price * (1 - sl_pct / 100)
        else:
            target_price = entry_price * (1 - tp_pct / 100)
            stop_price = entry_price * (1 + sl_pct / 100)
        
        exit_price = None
        exit_reason = None
        bars_held = 0
        peak_price = entry_price
        trough_price = entry_price
        
        for j in range(i + 1, min(i + max_hold_bars + 1, len(df))):
            bars_held = j - i
            row = df.iloc[j]
            
            # Update peak/trough for trailing stop
            if direction == 'LONG':
                peak_price = max(peak_price, row['High'])
                trailing_stop = peak_price * (1 - trailing_pct / 100) if use_trailing else stop_price
                
                # Check stop loss first
                if row['Low'] <= stop_price:
                    exit_price = stop_price
                    exit_reason = 'SL'
                    break
                # Check trailing stop
                if use_trailing and row['Low'] <= trailing_stop and peak_price > entry_price * 1.002:
                    exit_price = trailing_stop
                    exit_reason = 'TS'
                    break
                # Check take profit
                if row['High'] >= target_price:
                    exit_price = target_price
                    exit_reason = 'TP'
                    break
            else:
                trough_price = min(trough_price, row['Low'])
                trailing_stop = trough_price * (1 + trailing_pct / 100) if use_trailing else stop_price
                
                if row['High'] >= stop_price:
                    exit_price = stop_price
                    exit_reason = 'SL'
                    break
                if use_trailing and row['High'] >= trailing_stop and trough_price < entry_price * 0.998:
                    exit_price = trailing_stop
                    exit_reason = 'TS'
                    break
                if row['Low'] <= target_price:
                    exit_price = target_price
                    exit_reason = 'TP'
                    break
        
        if exit_price is None:
            exit_price = df['Close'].iloc[min(i + max_hold_bars, len(df) - 1)]
            exit_reason = 'TO'
        
        if direction == 'LONG':
            pnl_pct = (exit_price / entry_price - 1) * 100
        else:
            pnl_pct = (entry_price / exit_price - 1) * 100
        
        # MNQ: $2 per point, ~$0.05 per 0.01% move at $25000
        # Simplified: use percentage P&L * multiplier
        multiplier = 2.0  # MNQ multiplier
        commission = 1.24  # Round trip
        pnl_dollar = (pnl_pct / 100) * entry_price * multiplier / 100 - commission
        
        trades.append({
            'entry_time': entry_time,
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'pnl_pct': pnl_pct,
            'pnl_dollar': pnl_dollar,
            'bars_held': bars_held
        })
        
        last_trade_bar = i
    
    return pd.DataFrame(trades)


def analyze_results(trades_df, period_name=""):
    """Analyze backtest results."""
    if trades_df.empty:
        return None
    
    total = len(trades_df)
    wins = len(trades_df[trades_df['pnl_dollar'] > 0])
    losses = len(trades_df[trades_df['pnl_dollar'] <= 0])
    win_rate = wins / total * 100
    
    total_pnl = trades_df['pnl_dollar'].sum()
    avg_win = trades_df[trades_df['pnl_dollar'] > 0]['pnl_dollar'].mean() if wins > 0 else 0
    avg_loss = trades_df[trades_df['pnl_dollar'] <= 0]['pnl_dollar'].mean() if losses > 0 else 0
    
    # Calculate days in period
    if len(trades_df) > 0:
        days = (trades_df['entry_time'].max() - trades_df['entry_time'].min()).days or 1
        pnl_per_week = total_pnl / days * 7
        trades_per_day = total / days
    else:
        pnl_per_week = 0
        trades_per_day = 0
    
    # Exit reason breakdown
    exit_reasons = trades_df['exit_reason'].value_counts().to_dict()
    
    # Max drawdown
    cumulative = trades_df['pnl_dollar'].cumsum()
    peak = cumulative.expanding().max()
    drawdown = cumulative - peak
    max_dd = drawdown.min()
    
    return {
        'period': period_name,
        'trades': total,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'pnl_week': pnl_per_week,
        'trades_day': trades_per_day,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': abs(avg_win * wins / (avg_loss * losses)) if losses > 0 and avg_loss != 0 else 0,
        'max_dd': max_dd,
        'exit_reasons': exit_reasons
    }


def main():
    print("="*80)
    print("MNQ TREND-FOLLOWING STRATEGY ANALYSIS")
    print("="*80)
    
    # Load data
    df = load_data()
    
    # Identify trends
    print("\n1. IDENTIFYING TRENDS...")
    df = identify_trends(df)
    
    # Calculate noise statistics for last 30 days
    print("\n2. CALCULATING NOISE STATISTICS (Last 30 days)...")
    cutoff_30d = df.index[-1] - timedelta(days=30)
    df_30d = df[df.index >= cutoff_30d].copy()
    
    df_30d, noise_stats = calculate_noise_stats(df_30d, window=len(df_30d))
    
    print(f"\n   Noise Analysis (30-day):")
    print(f"   ATR %: mean={noise_stats['atr_pct_mean']:.3f}%, std={noise_stats['atr_pct_std']:.3f}%, max={noise_stats['atr_pct_max']:.3f}%")
    print(f"   Bar Range %: mean={noise_stats['bar_range_mean']:.3f}%, std={noise_stats['bar_range_std']:.3f}%")
    print(f"   Recommended Min SL: {noise_stats['recommended_sl_min']:.3f}% (to avoid noise triggers)")
    
    # Count trend signals
    long_signals = df_30d['trend_long'].sum()
    short_signals = df_30d['trend_short'].sum()
    print(f"\n   Trend Signals (30-day): LONG={long_signals}, SHORT={short_signals}")
    
    # Test different SL/TP combinations
    print("\n3. TESTING SL/TP COMBINATIONS (30-day backtest)...")
    print("-"*80)
    
    # Based on noise analysis, test SL from recommended_min to 2x that
    min_sl = max(0.3, noise_stats['recommended_sl_min'])
    
    configs = [
        # R:R 1:2 configurations
        {'sl': 0.3, 'tp': 0.6, 'rr': '1:2'},
        {'sl': 0.4, 'tp': 0.8, 'rr': '1:2'},
        {'sl': 0.5, 'tp': 1.0, 'rr': '1:2'},
        {'sl': 0.6, 'tp': 1.2, 'rr': '1:2'},
        {'sl': 0.7, 'tp': 1.4, 'rr': '1:2'},
        # R:R 1:3 configurations
        {'sl': 0.3, 'tp': 0.9, 'rr': '1:3'},
        {'sl': 0.4, 'tp': 1.2, 'rr': '1:3'},
        {'sl': 0.5, 'tp': 1.5, 'rr': '1:3'},
        {'sl': 0.6, 'tp': 1.8, 'rr': '1:3'},
        # Wider SL for noise avoidance
        {'sl': 0.8, 'tp': 1.6, 'rr': '1:2'},
        {'sl': 1.0, 'tp': 2.0, 'rr': '1:2'},
        {'sl': 0.8, 'tp': 2.4, 'rr': '1:3'},
    ]
    
    all_results = []
    
    for config in configs:
        sl = config['sl']
        tp = config['tp']
        rr = config['rr']
        
        for direction in ['LONG', 'SHORT']:
            trades = backtest_trend_strategy(df_30d, sl, tp, direction, cooldown_bars=12)
            
            if len(trades) >= 5:
                result = analyze_results(trades, f"30d_{direction}")
                if result:
                    result['sl'] = sl
                    result['tp'] = tp
                    result['rr'] = rr
                    result['direction'] = direction
                    all_results.append(result)
    
    # Sort by P&L per week
    all_results.sort(key=lambda x: x['pnl_week'], reverse=True)
    
    print(f"\n{'Dir':<6} {'SL/TP':<10} {'R:R':<6} {'Trades':<8} {'WR%':<8} {'$/Week':<10} {'MaxDD':<10} {'Exits'}")
    print("-"*90)
    
    for r in all_results[:15]:
        exits = ', '.join([f"{k}:{v}" for k, v in r['exit_reasons'].items()])
        print(f"{r['direction']:<6} {r['sl']:.1f}/{r['tp']:.1f}    {r['rr']:<6} {r['trades']:<8} {r['win_rate']:.1f}%    ${r['pnl_week']:>6.0f}     ${r['max_dd']:>6.0f}     {exits}")
    
    # Find profitable strategies
    profitable = [r for r in all_results if r['pnl_week'] > 10]
    
    print(f"\n\n4. PROFITABLE STRATEGIES (>${'10'}/week): {len(profitable)}")
    print("="*80)
    
    if not profitable:
        print("No profitable strategies found in 30-day backtest.")
        print("\nTrying with trailing stop...")
        
        # Try with trailing stop
        for config in configs[:6]:
            sl = config['sl']
            tp = config['tp']
            rr = config['rr']
            
            for direction in ['LONG', 'SHORT']:
                trades = backtest_trend_strategy(df_30d, sl, tp, direction, cooldown_bars=12,
                                                use_trailing=True, trailing_pct=0.25)
                
                if len(trades) >= 5:
                    result = analyze_results(trades, f"30d_{direction}_TS")
                    if result and result['pnl_week'] > 0:
                        print(f"  {direction} SL={sl}% TP={tp}% (trailing): WR={result['win_rate']:.1f}%, ${result['pnl_week']:.0f}/wk")
    
    # Validate on historical months
    print("\n\n5. HISTORICAL VALIDATION")
    print("="*80)
    
    # Get best config from 30-day test
    if all_results:
        best = all_results[0]
        best_sl = best['sl']
        best_tp = best['tp']
        best_dir = best['direction']
        
        print(f"\nValidating best config: {best_dir} SL={best_sl}% TP={best_tp}%")
        print("-"*60)
        
        # Test on different months
        months_to_test = [
            ('Dec 2025', df.index[-1] - timedelta(days=60), df.index[-1] - timedelta(days=30)),
            ('Nov 2025', df.index[-1] - timedelta(days=90), df.index[-1] - timedelta(days=60)),
            ('Oct 2025', df.index[-1] - timedelta(days=120), df.index[-1] - timedelta(days=90)),
        ]
        
        validation_results = []
        
        for month_name, start, end in months_to_test:
            df_month = df[(df.index >= start) & (df.index < end)].copy()
            if len(df_month) < 100:
                continue
            
            df_month = identify_trends(df_month)
            trades = backtest_trend_strategy(df_month, best_sl, best_tp, best_dir, cooldown_bars=12)
            
            if len(trades) >= 3:
                result = analyze_results(trades, month_name)
                if result:
                    validation_results.append(result)
                    print(f"  {month_name}: {result['trades']} trades, WR={result['win_rate']:.1f}%, ${result['pnl_week']:.0f}/wk")
        
        # Check consistency
        if validation_results:
            profitable_months = sum(1 for r in validation_results if r['pnl_week'] > 0)
            print(f"\n  Profitable months: {profitable_months}/{len(validation_results)}")
            
            if profitable_months >= len(validation_results) * 0.5:
                print("  ✅ Strategy shows consistency across months")
            else:
                print("  ⚠️ Strategy inconsistent - may be overfitted to recent data")
    
    # Final recommendation
    print("\n\n6. FINAL RECOMMENDATION")
    print("="*80)
    
    if profitable:
        best = profitable[0]
        print(f"\n  RECOMMENDED STRATEGY:")
        print(f"  Direction: {best['direction']}")
        print(f"  Stop Loss: {best['sl']}%")
        print(f"  Take Profit: {best['tp']}%")
        print(f"  R:R Ratio: {best['rr']}")
        print(f"  Expected Win Rate: {best['win_rate']:.1f}%")
        print(f"  Expected P&L: ${best['pnl_week']:.0f}/week")
        print(f"  Trades/Day: {best['trades_day']:.1f}")
        print(f"  Max Drawdown: ${best['max_dd']:.0f}")
        
        # Calculate required win rate for breakeven
        rr_ratio = best['tp'] / best['sl']
        breakeven_wr = 1 / (1 + rr_ratio) * 100
        print(f"\n  Breakeven Win Rate: {breakeven_wr:.1f}%")
        print(f"  Win Rate Buffer: {best['win_rate'] - breakeven_wr:.1f}% above breakeven")
    else:
        print("\n  No profitable trend-following strategy found.")
        print("  Consider:")
        print("  1. Mean reversion strategies instead")
        print("  2. Different trend identification methods")
        print("  3. Tighter entry conditions")
    
    return all_results


if __name__ == "__main__":
    results = main()
