#!/usr/bin/env python3
"""
Backtest ROC+MACD Trend Strategy on MNQ and SPY data.
Uses the QQQ parquet file which has 5-min bars with all indicators.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Load QQQ parquet (used for both MNQ and SPY)
QQQ_PARQUET = "data/QQQ_features.parquet"

def calculate_roc(df, period=12):
    """Calculate Rate of Change."""
    return (df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period) * 100

def backtest_roc_macd_strategy(df, symbol, sl_pct=0.70, tp_pct=1.40, cooldown_bars=20, roc_threshold=0.4):
    """
    Backtest ROC+MACD trend strategy.
    
    Entry LONG:  ROC(12) > threshold AND MACD_Hist > 0 AND MACD_Hist increasing
    Entry SHORT: ROC(12) < -threshold AND MACD_Hist < 0 AND MACD_Hist decreasing
    """
    print(f"\n{'='*60}")
    print(f"Backtesting ROC+MACD Trend Strategy on {symbol}")
    print(f"SL: {sl_pct}%, TP: {tp_pct}%, Cooldown: {cooldown_bars} bars, ROC threshold: {roc_threshold}%")
    print(f"{'='*60}")
    
    # Calculate indicators
    df = df.copy()
    df['roc_12'] = calculate_roc(df, 12)
    
    # MACD histogram (already have trend_macd and trend_macd_signal)
    df['macd_hist'] = df['trend_macd'] - df['trend_macd_signal']
    df['macd_hist_prev'] = df['macd_hist'].shift(1)
    df['macd_hist_increasing'] = df['macd_hist'] > df['macd_hist_prev']
    df['macd_hist_decreasing'] = df['macd_hist'] < df['macd_hist_prev']
    
    # Entry conditions
    df['long_signal'] = (
        (df['roc_12'] > roc_threshold) & 
        (df['macd_hist'] > 0) & 
        df['macd_hist_increasing']
    )
    df['short_signal'] = (
        (df['roc_12'] < -roc_threshold) & 
        (df['macd_hist'] < 0) & 
        df['macd_hist_decreasing']
    )
    
    # Simulate trades
    trades = []
    last_trade_bar = -cooldown_bars
    
    for i in range(50, len(df)):
        # Check cooldown
        if i - last_trade_bar < cooldown_bars:
            continue
        
        row = df.iloc[i]
        entry_price = row['Close']
        entry_time = df.index[i]
        
        direction = None
        if row['long_signal']:
            direction = 'LONG'
            target_price = entry_price * (1 + tp_pct / 100)
            stop_price = entry_price * (1 - sl_pct / 100)
        elif row['short_signal']:
            direction = 'SHORT'
            target_price = entry_price * (1 - tp_pct / 100)
            stop_price = entry_price * (1 + sl_pct / 100)
        else:
            continue
        
        # Simulate trade outcome
        exit_price = None
        exit_reason = None
        bars_held = 0
        max_hold = 48  # 4 hours
        
        for j in range(i + 1, min(i + max_hold + 1, len(df))):
            bars_held += 1
            future_row = df.iloc[j]
            high = future_row['High']
            low = future_row['Low']
            close = future_row['Close']
            
            if direction == 'LONG':
                # Check stop loss first (worst case)
                if low <= stop_price:
                    exit_price = stop_price
                    exit_reason = 'STOP_LOSS'
                    break
                # Check take profit
                if high >= target_price:
                    exit_price = target_price
                    exit_reason = 'TAKE_PROFIT'
                    break
            else:  # SHORT
                # Check stop loss first
                if high >= stop_price:
                    exit_price = stop_price
                    exit_reason = 'STOP_LOSS'
                    break
                # Check take profit
                if low <= target_price:
                    exit_price = target_price
                    exit_reason = 'TAKE_PROFIT'
                    break
        
        # Timeout exit
        if exit_price is None:
            exit_price = df.iloc[min(i + max_hold, len(df) - 1)]['Close']
            exit_reason = 'TIMEOUT'
        
        # Calculate P&L
        if direction == 'LONG':
            pnl_pct = (exit_price / entry_price - 1) * 100
        else:
            pnl_pct = (entry_price / exit_price - 1) * 100
        
        # Commission (estimate)
        commission = 2.50  # MNQ/MES commission per round trip
        
        # Dollar P&L (MNQ = $2/point, MES = $5/point)
        if symbol == 'MNQ':
            multiplier = 2.0
            point_value = entry_price * 0.01  # 1% = 1 point for QQQ scale
        else:  # SPY/MES
            multiplier = 5.0
            point_value = entry_price * 0.01
        
        pnl_dollar = (pnl_pct / 100) * entry_price * multiplier - commission
        
        trades.append({
            'entry_time': entry_time,
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'pnl_pct': pnl_pct,
            'pnl_dollar': pnl_dollar,
            'bars_held': bars_held,
            'roc_12': row['roc_12'],
            'macd_hist': row['macd_hist']
        })
        
        last_trade_bar = i
    
    # Analyze results
    if not trades:
        print("No trades generated!")
        return None
    
    trades_df = pd.DataFrame(trades)
    
    # Overall stats
    total_trades = len(trades_df)
    wins = len(trades_df[trades_df['pnl_dollar'] > 0])
    losses = len(trades_df[trades_df['pnl_dollar'] <= 0])
    win_rate = wins / total_trades * 100
    
    total_pnl = trades_df['pnl_dollar'].sum()
    avg_pnl = trades_df['pnl_dollar'].mean()
    
    # By direction
    long_trades = trades_df[trades_df['direction'] == 'LONG']
    short_trades = trades_df[trades_df['direction'] == 'SHORT']
    
    long_wr = len(long_trades[long_trades['pnl_dollar'] > 0]) / len(long_trades) * 100 if len(long_trades) > 0 else 0
    short_wr = len(short_trades[short_trades['pnl_dollar'] > 0]) / len(short_trades) * 100 if len(short_trades) > 0 else 0
    
    # Time period
    days = (trades_df['entry_time'].max() - trades_df['entry_time'].min()).days or 1
    trades_per_day = total_trades / days
    pnl_per_week = total_pnl / days * 7
    
    # By exit reason
    tp_count = len(trades_df[trades_df['exit_reason'] == 'TAKE_PROFIT'])
    sl_count = len(trades_df[trades_df['exit_reason'] == 'STOP_LOSS'])
    to_count = len(trades_df[trades_df['exit_reason'] == 'TIMEOUT'])
    
    # Drawdown
    cumulative_pnl = trades_df['pnl_dollar'].cumsum()
    peak = cumulative_pnl.expanding().max()
    drawdown = cumulative_pnl - peak
    max_drawdown = drawdown.min()
    
    # Consecutive losses
    trades_df['is_loss'] = trades_df['pnl_dollar'] <= 0
    max_consec_losses = 0
    current_streak = 0
    for is_loss in trades_df['is_loss']:
        if is_loss:
            current_streak += 1
            max_consec_losses = max(max_consec_losses, current_streak)
        else:
            current_streak = 0
    
    print(f"\n📊 Results ({days} days):")
    print(f"  Total trades: {total_trades} ({trades_per_day:.1f}/day)")
    print(f"  Win rate: {win_rate:.1f}%")
    print(f"  LONG: {len(long_trades)} trades, {long_wr:.1f}% WR")
    print(f"  SHORT: {len(short_trades)} trades, {short_wr:.1f}% WR")
    print(f"\n💰 P&L:")
    print(f"  Total: ${total_pnl:.2f}")
    print(f"  Per week: ${pnl_per_week:.2f}")
    print(f"  Avg per trade: ${avg_pnl:.2f}")
    print(f"\n📈 Exit reasons:")
    print(f"  Take Profit: {tp_count} ({tp_count/total_trades*100:.1f}%)")
    print(f"  Stop Loss: {sl_count} ({sl_count/total_trades*100:.1f}%)")
    print(f"  Timeout: {to_count} ({to_count/total_trades*100:.1f}%)")
    print(f"\n⚠️ Risk:")
    print(f"  Max drawdown: ${max_drawdown:.2f}")
    print(f"  Max consecutive losses: {max_consec_losses}")
    
    return {
        'symbol': symbol,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'long_wr': long_wr,
        'short_wr': short_wr,
        'total_pnl': total_pnl,
        'pnl_per_week': pnl_per_week,
        'trades_per_day': trades_per_day,
        'max_drawdown': max_drawdown,
        'max_consec_losses': max_consec_losses,
        'trades_df': trades_df
    }


def main():
    print("Loading QQQ parquet data...")
    df = pd.read_parquet(QQQ_PARQUET)
    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    
    # Filter to last 30 days
    cutoff = df.index[-1] - timedelta(days=30)
    df_30d = df[df.index >= cutoff]
    print(f"Using last 30 days: {len(df_30d)} bars")
    
    # Test different parameter combinations
    results = []
    
    # Test ROC+MACD on MNQ
    print("\n" + "="*80)
    print("TESTING MNQ (Nasdaq Micro Futures)")
    print("="*80)
    
    # Standard BTC parameters
    result = backtest_roc_macd_strategy(df_30d, 'MNQ', sl_pct=0.70, tp_pct=1.40, cooldown_bars=20, roc_threshold=0.4)
    if result:
        results.append(result)
    
    # Try different thresholds for MNQ (it may have different volatility)
    for roc_thresh in [0.2, 0.3, 0.5]:
        result = backtest_roc_macd_strategy(df_30d, 'MNQ', sl_pct=0.70, tp_pct=1.40, cooldown_bars=20, roc_threshold=roc_thresh)
        if result:
            results.append(result)
    
    # Try tighter SL/TP for MNQ (lower volatility than BTC)
    result = backtest_roc_macd_strategy(df_30d, 'MNQ', sl_pct=0.35, tp_pct=0.70, cooldown_bars=20, roc_threshold=0.3)
    if result:
        results.append(result)
    
    # Test on SPY/MES
    print("\n" + "="*80)
    print("TESTING SPY/MES (S&P 500 Micro Futures)")
    print("="*80)
    
    # Standard parameters
    result = backtest_roc_macd_strategy(df_30d, 'SPY', sl_pct=0.70, tp_pct=1.40, cooldown_bars=20, roc_threshold=0.4)
    if result:
        results.append(result)
    
    # Try different thresholds
    for roc_thresh in [0.2, 0.3]:
        result = backtest_roc_macd_strategy(df_30d, 'SPY', sl_pct=0.70, tp_pct=1.40, cooldown_bars=20, roc_threshold=roc_thresh)
        if result:
            results.append(result)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY - Best Results")
    print("="*80)
    
    if results:
        # Sort by P&L per week
        results_sorted = sorted(results, key=lambda x: x['pnl_per_week'], reverse=True)
        
        print(f"\n{'Symbol':<8} {'WR%':<8} {'$/Week':<10} {'Trades/Day':<12} {'Max DD':<10} {'Consec L':<10}")
        print("-" * 60)
        for r in results_sorted[:5]:
            print(f"{r['symbol']:<8} {r['win_rate']:.1f}%    ${r['pnl_per_week']:.0f}      {r['trades_per_day']:.1f}          ${r['max_drawdown']:.0f}      {r['max_consec_losses']}")


if __name__ == "__main__":
    main()
