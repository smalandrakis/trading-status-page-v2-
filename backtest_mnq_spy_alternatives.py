#!/usr/bin/env python3
"""
Backtest alternative strategies for MNQ and SPY.
Since ROC+MACD trend doesn't work well, try:
1. Mean reversion (BB %B < 0.5) - already used for MNQ
2. RSI oversold/overbought
3. Tighter SL/TP ratios suited for lower volatility
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

QQQ_PARQUET = "data/QQQ_features.parquet"

def backtest_bb_meanrev(df, symbol, sl_pct=0.30, tp_pct=0.50, cooldown_bars=10, bb_threshold=0.5):
    """
    Backtest BB Mean Reversion strategy (current MNQ strategy).
    Entry LONG: BB %B < threshold (oversold)
    """
    print(f"\n{'='*60}")
    print(f"BB Mean Reversion on {symbol}")
    print(f"SL: {sl_pct}%, TP: {tp_pct}%, Cooldown: {cooldown_bars} bars, BB threshold: {bb_threshold}")
    print(f"{'='*60}")
    
    df = df.copy()
    df['long_signal'] = df['volatility_bbp'] < bb_threshold
    
    trades = []
    last_trade_bar = -cooldown_bars
    
    for i in range(50, len(df)):
        if i - last_trade_bar < cooldown_bars:
            continue
        
        row = df.iloc[i]
        if not row['long_signal']:
            continue
        
        entry_price = row['Close']
        entry_time = df.index[i]
        target_price = entry_price * (1 + tp_pct / 100)
        stop_price = entry_price * (1 - sl_pct / 100)
        
        exit_price = None
        exit_reason = None
        bars_held = 0
        max_hold = 48
        
        for j in range(i + 1, min(i + max_hold + 1, len(df))):
            bars_held += 1
            future_row = df.iloc[j]
            
            if future_row['Low'] <= stop_price:
                exit_price = stop_price
                exit_reason = 'STOP_LOSS'
                break
            if future_row['High'] >= target_price:
                exit_price = target_price
                exit_reason = 'TAKE_PROFIT'
                break
        
        if exit_price is None:
            exit_price = df.iloc[min(i + max_hold, len(df) - 1)]['Close']
            exit_reason = 'TIMEOUT'
        
        pnl_pct = (exit_price / entry_price - 1) * 100
        commission = 2.50
        multiplier = 2.0 if symbol == 'MNQ' else 5.0
        pnl_dollar = (pnl_pct / 100) * entry_price * multiplier - commission
        
        trades.append({
            'entry_time': entry_time,
            'direction': 'LONG',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'pnl_pct': pnl_pct,
            'pnl_dollar': pnl_dollar,
            'bars_held': bars_held,
            'bb_pct_b': row['volatility_bbp']
        })
        
        last_trade_bar = i
    
    return analyze_trades(trades, symbol)


def backtest_rsi_strategy(df, symbol, sl_pct=0.30, tp_pct=0.50, cooldown_bars=10, 
                          rsi_oversold=30, rsi_overbought=70):
    """
    Backtest RSI strategy.
    Entry LONG: RSI < oversold
    Entry SHORT: RSI > overbought
    """
    print(f"\n{'='*60}")
    print(f"RSI Strategy on {symbol}")
    print(f"SL: {sl_pct}%, TP: {tp_pct}%, Cooldown: {cooldown_bars} bars")
    print(f"RSI oversold: {rsi_oversold}, overbought: {rsi_overbought}")
    print(f"{'='*60}")
    
    df = df.copy()
    df['long_signal'] = df['momentum_rsi'] < rsi_oversold
    df['short_signal'] = df['momentum_rsi'] > rsi_overbought
    
    trades = []
    last_trade_bar = -cooldown_bars
    
    for i in range(50, len(df)):
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
        
        exit_price = None
        exit_reason = None
        bars_held = 0
        max_hold = 48
        
        for j in range(i + 1, min(i + max_hold + 1, len(df))):
            bars_held += 1
            future_row = df.iloc[j]
            
            if direction == 'LONG':
                if future_row['Low'] <= stop_price:
                    exit_price = stop_price
                    exit_reason = 'STOP_LOSS'
                    break
                if future_row['High'] >= target_price:
                    exit_price = target_price
                    exit_reason = 'TAKE_PROFIT'
                    break
            else:
                if future_row['High'] >= stop_price:
                    exit_price = stop_price
                    exit_reason = 'STOP_LOSS'
                    break
                if future_row['Low'] <= target_price:
                    exit_price = target_price
                    exit_reason = 'TAKE_PROFIT'
                    break
        
        if exit_price is None:
            exit_price = df.iloc[min(i + max_hold, len(df) - 1)]['Close']
            exit_reason = 'TIMEOUT'
        
        if direction == 'LONG':
            pnl_pct = (exit_price / entry_price - 1) * 100
        else:
            pnl_pct = (entry_price / exit_price - 1) * 100
        
        commission = 2.50
        multiplier = 2.0 if symbol == 'MNQ' else 5.0
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
            'rsi': row['momentum_rsi']
        })
        
        last_trade_bar = i
    
    return analyze_trades(trades, symbol)


def backtest_bb_rsi_combo(df, symbol, sl_pct=0.30, tp_pct=0.50, cooldown_bars=10,
                          bb_threshold=0.3, rsi_threshold=40):
    """
    Backtest BB + RSI combo strategy.
    Entry LONG: BB %B < threshold AND RSI < threshold (double confirmation)
    """
    print(f"\n{'='*60}")
    print(f"BB + RSI Combo on {symbol}")
    print(f"SL: {sl_pct}%, TP: {tp_pct}%, Cooldown: {cooldown_bars} bars")
    print(f"BB threshold: {bb_threshold}, RSI threshold: {rsi_threshold}")
    print(f"{'='*60}")
    
    df = df.copy()
    df['long_signal'] = (df['volatility_bbp'] < bb_threshold) & (df['momentum_rsi'] < rsi_threshold)
    
    trades = []
    last_trade_bar = -cooldown_bars
    
    for i in range(50, len(df)):
        if i - last_trade_bar < cooldown_bars:
            continue
        
        row = df.iloc[i]
        if not row['long_signal']:
            continue
        
        entry_price = row['Close']
        entry_time = df.index[i]
        target_price = entry_price * (1 + tp_pct / 100)
        stop_price = entry_price * (1 - sl_pct / 100)
        
        exit_price = None
        exit_reason = None
        bars_held = 0
        max_hold = 48
        
        for j in range(i + 1, min(i + max_hold + 1, len(df))):
            bars_held += 1
            future_row = df.iloc[j]
            
            if future_row['Low'] <= stop_price:
                exit_price = stop_price
                exit_reason = 'STOP_LOSS'
                break
            if future_row['High'] >= target_price:
                exit_price = target_price
                exit_reason = 'TAKE_PROFIT'
                break
        
        if exit_price is None:
            exit_price = df.iloc[min(i + max_hold, len(df) - 1)]['Close']
            exit_reason = 'TIMEOUT'
        
        pnl_pct = (exit_price / entry_price - 1) * 100
        commission = 2.50
        multiplier = 2.0 if symbol == 'MNQ' else 5.0
        pnl_dollar = (pnl_pct / 100) * entry_price * multiplier - commission
        
        trades.append({
            'entry_time': entry_time,
            'direction': 'LONG',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'pnl_pct': pnl_pct,
            'pnl_dollar': pnl_dollar,
            'bars_held': bars_held
        })
        
        last_trade_bar = i
    
    return analyze_trades(trades, symbol)


def analyze_trades(trades, symbol):
    """Analyze trade results."""
    if not trades:
        print("No trades generated!")
        return None
    
    trades_df = pd.DataFrame(trades)
    
    total_trades = len(trades_df)
    wins = len(trades_df[trades_df['pnl_dollar'] > 0])
    win_rate = wins / total_trades * 100
    
    total_pnl = trades_df['pnl_dollar'].sum()
    
    long_trades = trades_df[trades_df['direction'] == 'LONG']
    short_trades = trades_df[trades_df['direction'] == 'SHORT']
    
    long_wr = len(long_trades[long_trades['pnl_dollar'] > 0]) / len(long_trades) * 100 if len(long_trades) > 0 else 0
    short_wr = len(short_trades[short_trades['pnl_dollar'] > 0]) / len(short_trades) * 100 if len(short_trades) > 0 else 0
    
    days = (trades_df['entry_time'].max() - trades_df['entry_time'].min()).days or 1
    trades_per_day = total_trades / days
    pnl_per_week = total_pnl / days * 7
    
    tp_count = len(trades_df[trades_df['exit_reason'] == 'TAKE_PROFIT'])
    sl_count = len(trades_df[trades_df['exit_reason'] == 'STOP_LOSS'])
    to_count = len(trades_df[trades_df['exit_reason'] == 'TIMEOUT'])
    
    cumulative_pnl = trades_df['pnl_dollar'].cumsum()
    peak = cumulative_pnl.expanding().max()
    drawdown = cumulative_pnl - peak
    max_drawdown = drawdown.min()
    
    print(f"\n📊 Results ({days} days):")
    print(f"  Total trades: {total_trades} ({trades_per_day:.1f}/day)")
    print(f"  Win rate: {win_rate:.1f}%")
    print(f"  LONG: {len(long_trades)} trades, {long_wr:.1f}% WR")
    print(f"  SHORT: {len(short_trades)} trades, {short_wr:.1f}% WR")
    print(f"\n💰 P&L:")
    print(f"  Total: ${total_pnl:.2f}")
    print(f"  Per week: ${pnl_per_week:.2f}")
    print(f"\n📈 Exit reasons:")
    print(f"  Take Profit: {tp_count} ({tp_count/total_trades*100:.1f}%)")
    print(f"  Stop Loss: {sl_count} ({sl_count/total_trades*100:.1f}%)")
    print(f"  Timeout: {to_count} ({to_count/total_trades*100:.1f}%)")
    print(f"\n⚠️ Risk:")
    print(f"  Max drawdown: ${max_drawdown:.2f}")
    
    return {
        'symbol': symbol,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'long_wr': long_wr,
        'short_wr': short_wr,
        'total_pnl': total_pnl,
        'pnl_per_week': pnl_per_week,
        'trades_per_day': trades_per_day,
        'max_drawdown': max_drawdown
    }


def main():
    print("Loading QQQ parquet data...")
    df = pd.read_parquet(QQQ_PARQUET)
    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    
    cutoff = df.index[-1] - timedelta(days=30)
    df_30d = df[df.index >= cutoff]
    print(f"Using last 30 days: {len(df_30d)} bars")
    
    results = []
    
    # Test BB Mean Reversion (current MNQ strategy)
    print("\n" + "="*80)
    print("TESTING BB MEAN REVERSION STRATEGIES")
    print("="*80)
    
    for symbol in ['MNQ', 'SPY']:
        # Current MNQ strategy
        result = backtest_bb_meanrev(df_30d, symbol, sl_pct=0.30, tp_pct=0.50, cooldown_bars=10, bb_threshold=0.5)
        if result:
            result['strategy'] = f'BB<0.5 SL0.3/TP0.5'
            results.append(result)
        
        # Tighter threshold
        result = backtest_bb_meanrev(df_30d, symbol, sl_pct=0.30, tp_pct=0.50, cooldown_bars=10, bb_threshold=0.3)
        if result:
            result['strategy'] = f'BB<0.3 SL0.3/TP0.5'
            results.append(result)
        
        # Wider SL/TP
        result = backtest_bb_meanrev(df_30d, symbol, sl_pct=0.50, tp_pct=1.00, cooldown_bars=15, bb_threshold=0.4)
        if result:
            result['strategy'] = f'BB<0.4 SL0.5/TP1.0'
            results.append(result)
    
    # Test RSI strategies
    print("\n" + "="*80)
    print("TESTING RSI STRATEGIES")
    print("="*80)
    
    for symbol in ['MNQ', 'SPY']:
        result = backtest_rsi_strategy(df_30d, symbol, sl_pct=0.30, tp_pct=0.50, cooldown_bars=10, 
                                       rsi_oversold=30, rsi_overbought=70)
        if result:
            result['strategy'] = f'RSI 30/70'
            results.append(result)
        
        result = backtest_rsi_strategy(df_30d, symbol, sl_pct=0.30, tp_pct=0.50, cooldown_bars=10,
                                       rsi_oversold=35, rsi_overbought=65)
        if result:
            result['strategy'] = f'RSI 35/65'
            results.append(result)
    
    # Test BB + RSI combo
    print("\n" + "="*80)
    print("TESTING BB + RSI COMBO STRATEGIES")
    print("="*80)
    
    for symbol in ['MNQ', 'SPY']:
        result = backtest_bb_rsi_combo(df_30d, symbol, sl_pct=0.30, tp_pct=0.50, cooldown_bars=10,
                                       bb_threshold=0.3, rsi_threshold=40)
        if result:
            result['strategy'] = f'BB<0.3+RSI<40'
            results.append(result)
        
        result = backtest_bb_rsi_combo(df_30d, symbol, sl_pct=0.40, tp_pct=0.80, cooldown_bars=15,
                                       bb_threshold=0.4, rsi_threshold=45)
        if result:
            result['strategy'] = f'BB<0.4+RSI<45'
            results.append(result)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY - All Results Sorted by P&L/Week")
    print("="*80)
    
    if results:
        results_sorted = sorted(results, key=lambda x: x['pnl_per_week'], reverse=True)
        
        print(f"\n{'Symbol':<6} {'Strategy':<20} {'WR%':<8} {'$/Week':<10} {'Trades/Day':<12} {'Max DD':<10}")
        print("-" * 75)
        for r in results_sorted:
            print(f"{r['symbol']:<6} {r['strategy']:<20} {r['win_rate']:.1f}%    ${r['pnl_per_week']:>6.0f}      {r['trades_per_day']:.1f}          ${r['max_drawdown']:.0f}")


if __name__ == "__main__":
    main()
