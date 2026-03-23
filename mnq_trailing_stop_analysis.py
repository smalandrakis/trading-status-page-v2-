#!/usr/bin/env python3
"""
MNQ/QQQ Trailing Stop Analysis using 15-second data.

With 15-sec granularity we can:
1. Measure TRUE intra-trade noise (not just bar-level)
2. Test trailing stop configurations accurately
3. Find optimal SL/TP/trailing combinations
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

def load_15sec_data():
    """Load 15-second QQQ data."""
    df = pd.read_parquet('data/QQQ_15sec_30days.parquet')
    print(f"Loaded {len(df)} 15-sec bars from {df.index[0]} to {df.index[-1]}")
    return df


def calculate_true_noise(df):
    """Calculate true noise statistics with 15-sec granularity."""
    
    # Price change per 15-sec bar
    df['pct_change'] = df['close'].pct_change() * 100
    df['bar_range_pct'] = (df['high'] - df['low']) / df['close'] * 100
    
    print("\n" + "="*70)
    print("TRUE NOISE ANALYSIS (15-second bars)")
    print("="*70)
    
    print(f"\nPer-bar price change:")
    print(f"  Mean: {df['pct_change'].abs().mean():.4f}%")
    print(f"  Std:  {df['pct_change'].std():.4f}%")
    print(f"  Max:  {df['pct_change'].abs().max():.4f}%")
    
    print(f"\nPer-bar range (high-low):")
    print(f"  Mean: {df['bar_range_pct'].mean():.4f}%")
    print(f"  Std:  {df['bar_range_pct'].std():.4f}%")
    print(f"  95th: {df['bar_range_pct'].quantile(0.95):.4f}%")
    
    # Cumulative noise over time windows
    print(f"\nCumulative adverse movement (MAE):")
    
    for minutes in [1, 5, 15, 30, 60]:
        bars = minutes * 4  # 4 bars per minute
        
        # Rolling min/max
        df[f'roll_low_{minutes}m'] = df['low'].rolling(bars).min()
        df[f'roll_high_{minutes}m'] = df['high'].rolling(bars).max()
        
        # MAE from entry (using open as entry proxy)
        df[f'mae_long_{minutes}m'] = (df['open'] - df[f'roll_low_{minutes}m'].shift(-bars+1)) / df['open'] * 100
        df[f'mae_short_{minutes}m'] = (df[f'roll_high_{minutes}m'].shift(-bars+1) - df['open']) / df['open'] * 100
        
        mae_long = df[f'mae_long_{minutes}m'].dropna()
        mae_short = df[f'mae_short_{minutes}m'].dropna()
        
        print(f"  {minutes:2d}min: LONG mae={mae_long.mean():.3f}% (95th={mae_long.quantile(0.95):.3f}%), SHORT mae={mae_short.mean():.3f}% (95th={mae_short.quantile(0.95):.3f}%)")
    
    return df


def calculate_indicators_15sec(df):
    """Calculate indicators on 15-sec data."""
    
    # Resample to 5-min for indicator calculation (more stable)
    df_5min = df.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    # RSI
    delta = df_5min['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df_5min['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    df_5min['ema_12'] = df_5min['close'].ewm(span=12).mean()
    df_5min['ema_26'] = df_5min['close'].ewm(span=26).mean()
    df_5min['macd'] = df_5min['ema_12'] - df_5min['ema_26']
    df_5min['macd_signal'] = df_5min['macd'].ewm(span=9).mean()
    df_5min['macd_hist'] = df_5min['macd'] - df_5min['macd_signal']
    
    # Bollinger Bands
    df_5min['sma_20'] = df_5min['close'].rolling(20).mean()
    df_5min['std_20'] = df_5min['close'].rolling(20).std()
    df_5min['bb_upper'] = df_5min['sma_20'] + 2 * df_5min['std_20']
    df_5min['bb_lower'] = df_5min['sma_20'] - 2 * df_5min['std_20']
    df_5min['bb_pos'] = (df_5min['close'] - df_5min['bb_lower']) / (df_5min['bb_upper'] - df_5min['bb_lower'])
    
    # ROC
    df_5min['roc_5'] = (df_5min['close'] - df_5min['close'].shift(5)) / df_5min['close'].shift(5) * 100
    
    return df_5min


def generate_signals(df_5min):
    """Generate trading signals on 5-min data."""
    
    df_5min['sig_rsi_oversold'] = (df_5min['rsi'] < 30) & (df_5min['rsi'] > df_5min['rsi'].shift(1))
    df_5min['sig_rsi_overbought'] = (df_5min['rsi'] > 70) & (df_5min['rsi'] < df_5min['rsi'].shift(1))
    df_5min['sig_bb_lower'] = (df_5min['bb_pos'] < 0.1) & (df_5min['close'] > df_5min['close'].shift(1))
    df_5min['sig_bb_upper'] = (df_5min['bb_pos'] > 0.9) & (df_5min['close'] < df_5min['close'].shift(1))
    df_5min['sig_momentum_long'] = (df_5min['roc_5'] > 0.3) & (df_5min['macd_hist'] > 0) & (df_5min['rsi'] > 50) & (df_5min['rsi'] < 70)
    df_5min['sig_momentum_short'] = (df_5min['roc_5'] < -0.3) & (df_5min['macd_hist'] < 0) & (df_5min['rsi'] < 50) & (df_5min['rsi'] > 30)
    
    return df_5min


def backtest_with_trailing(df_15sec, df_5min, signal_col, direction, 
                           sl_pct, tp_pct, trailing_pct=None, trailing_activation=0.0,
                           cooldown_5min=10, max_hold_minutes=240):
    """
    Backtest with 15-second price tracking for accurate trailing stops.
    
    trailing_pct: Distance for trailing stop (None = no trailing)
    trailing_activation: Profit % required before trailing activates
    """
    trades = []
    last_signal_idx = -cooldown_5min
    
    signal_times = df_5min[df_5min[signal_col] == True].index
    
    for sig_time in signal_times:
        # Check cooldown
        sig_idx = df_5min.index.get_loc(sig_time)
        if sig_idx - last_signal_idx < cooldown_5min:
            continue
        
        # Get entry price (close of signal bar)
        entry_price = df_5min.loc[sig_time, 'close']
        
        # Find 15-sec bars after signal
        end_time = sig_time + timedelta(minutes=max_hold_minutes)
        trade_bars = df_15sec[(df_15sec.index > sig_time) & (df_15sec.index <= end_time)]
        
        if len(trade_bars) < 10:
            continue
        
        # Initialize
        if direction == 'LONG':
            tp_price = entry_price * (1 + tp_pct / 100)
            sl_price = entry_price * (1 - sl_pct / 100)
        else:
            tp_price = entry_price * (1 - tp_pct / 100)
            sl_price = entry_price * (1 + sl_pct / 100)
        
        exit_price = None
        exit_reason = None
        peak_price = entry_price
        trough_price = entry_price
        trailing_active = False
        
        # Simulate trade bar by bar
        for bar_time, bar in trade_bars.iterrows():
            # Update peak/trough
            if direction == 'LONG':
                peak_price = max(peak_price, bar['high'])
                current_profit_pct = (peak_price / entry_price - 1) * 100
                
                # Check if trailing should activate
                if trailing_pct and current_profit_pct >= trailing_activation:
                    trailing_active = True
                    trailing_stop = peak_price * (1 - trailing_pct / 100)
                    # Trailing stop can only move up
                    if trailing_stop > sl_price:
                        sl_price = trailing_stop
                
                # Check stop loss
                if bar['low'] <= sl_price:
                    exit_price = sl_price
                    exit_reason = 'TS' if trailing_active else 'SL'
                    break
                
                # Check take profit
                if bar['high'] >= tp_price:
                    exit_price = tp_price
                    exit_reason = 'TP'
                    break
            else:
                trough_price = min(trough_price, bar['low'])
                current_profit_pct = (entry_price / trough_price - 1) * 100
                
                if trailing_pct and current_profit_pct >= trailing_activation:
                    trailing_active = True
                    trailing_stop = trough_price * (1 + trailing_pct / 100)
                    if trailing_stop < sl_price:
                        sl_price = trailing_stop
                
                if bar['high'] >= sl_price:
                    exit_price = sl_price
                    exit_reason = 'TS' if trailing_active else 'SL'
                    break
                
                if bar['low'] <= tp_price:
                    exit_price = tp_price
                    exit_reason = 'TP'
                    break
        
        # Timeout
        if exit_price is None:
            exit_price = trade_bars.iloc[-1]['close']
            exit_reason = 'TO'
        
        # Calculate P&L
        if direction == 'LONG':
            pnl_pct = (exit_price / entry_price - 1) * 100
        else:
            pnl_pct = (entry_price / exit_price - 1) * 100
        
        # MNQ: ~$5 per 0.1% at $25000
        pnl_dollar = pnl_pct * 5 - 1.24
        
        trades.append({
            'entry_time': sig_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'pnl_pct': pnl_pct,
            'pnl_dollar': pnl_dollar,
            'peak_profit': (peak_price / entry_price - 1) * 100 if direction == 'LONG' else (entry_price / trough_price - 1) * 100
        })
        
        last_signal_idx = sig_idx
    
    return pd.DataFrame(trades)


def analyze_results(trades_df):
    """Analyze backtest results."""
    if trades_df.empty or len(trades_df) < 3:
        return None
    
    total = len(trades_df)
    wins = (trades_df['pnl_dollar'] > 0).sum()
    wr = wins / total * 100
    total_pnl = trades_df['pnl_dollar'].sum()
    
    days = max(1, (trades_df['entry_time'].max() - trades_df['entry_time'].min()).days)
    pnl_week = total_pnl / days * 7
    
    exits = trades_df['exit_reason'].value_counts().to_dict()
    
    # Avg profit captured vs peak
    avg_peak = trades_df['peak_profit'].mean()
    avg_realized = trades_df['pnl_pct'].mean()
    
    return {
        'trades': total,
        'wr': wr,
        'pnl_week': pnl_week,
        'total_pnl': total_pnl,
        'exits': exits,
        'avg_peak': avg_peak,
        'avg_realized': avg_realized,
        'capture_rate': avg_realized / avg_peak * 100 if avg_peak > 0 else 0
    }


def main():
    print("="*70)
    print("MNQ TRAILING STOP ANALYSIS (15-second data)")
    print("="*70)
    
    # Load data
    df_15sec = load_15sec_data()
    
    # Analyze true noise
    df_15sec = calculate_true_noise(df_15sec)
    
    # Calculate indicators
    print("\nCalculating indicators...")
    df_5min = calculate_indicators_15sec(df_15sec)
    df_5min = generate_signals(df_5min)
    
    # Count signals
    print("\nSignal counts:")
    for col in df_5min.columns:
        if col.startswith('sig_'):
            count = df_5min[col].sum()
            print(f"  {col}: {count}")
    
    # Test configurations
    print("\n" + "="*70)
    print("BACKTESTING WITH TRAILING STOPS")
    print("="*70)
    
    strategies = [
        ('sig_rsi_oversold', 'LONG'),
        ('sig_rsi_overbought', 'SHORT'),
        ('sig_bb_lower', 'LONG'),
        ('sig_bb_upper', 'SHORT'),
        ('sig_momentum_long', 'LONG'),
        ('sig_momentum_short', 'SHORT'),
    ]
    
    # Trailing stop configurations
    configs = [
        # No trailing (baseline)
        {'sl': 0.5, 'tp': 1.0, 'trail': None, 'act': 0},
        {'sl': 0.7, 'tp': 1.4, 'trail': None, 'act': 0},
        {'sl': 1.0, 'tp': 2.0, 'trail': None, 'act': 0},
        
        # Trailing stop (activate at 0.3% profit)
        {'sl': 0.5, 'tp': 1.5, 'trail': 0.3, 'act': 0.3},
        {'sl': 0.7, 'tp': 2.0, 'trail': 0.4, 'act': 0.4},
        {'sl': 1.0, 'tp': 3.0, 'trail': 0.5, 'act': 0.5},
        
        # Tighter trailing
        {'sl': 0.5, 'tp': 2.0, 'trail': 0.2, 'act': 0.2},
        {'sl': 0.7, 'tp': 2.5, 'trail': 0.3, 'act': 0.3},
        
        # Wider trailing (let winners run)
        {'sl': 1.0, 'tp': 4.0, 'trail': 0.6, 'act': 0.6},
        {'sl': 1.5, 'tp': 5.0, 'trail': 0.8, 'act': 0.8},
        
        # Breakeven trailing (move SL to entry after X% profit)
        {'sl': 0.5, 'tp': 2.0, 'trail': 0.5, 'act': 0.5},
        {'sl': 0.7, 'tp': 2.5, 'trail': 0.7, 'act': 0.7},
    ]
    
    all_results = []
    
    for sig_col, direction in strategies:
        for cfg in configs:
            trades = backtest_with_trailing(
                df_15sec, df_5min, sig_col, direction,
                sl_pct=cfg['sl'], tp_pct=cfg['tp'],
                trailing_pct=cfg['trail'], trailing_activation=cfg['act'],
                cooldown_5min=8, max_hold_minutes=240
            )
            
            result = analyze_results(trades)
            if result:
                result['strategy'] = sig_col.replace('sig_', '')
                result['direction'] = direction
                result['sl'] = cfg['sl']
                result['tp'] = cfg['tp']
                result['trail'] = cfg['trail']
                result['act'] = cfg['act']
                all_results.append(result)
    
    # Sort by P&L
    all_results.sort(key=lambda x: x['pnl_week'], reverse=True)
    
    print(f"\n{'Strategy':<18} {'Dir':<6} {'SL':<5} {'TP':<5} {'Trail':<6} {'Trades':<7} {'WR%':<7} {'$/Wk':<8} {'Exits'}")
    print("-"*90)
    
    for r in all_results[:25]:
        trail_str = f"{r['trail']:.1f}" if r['trail'] else "None"
        exits = ', '.join([f"{k}:{v}" for k, v in r['exits'].items()])
        print(f"{r['strategy']:<18} {r['direction']:<6} {r['sl']:<5.1f} {r['tp']:<5.1f} {trail_str:<6} {r['trades']:<7} {r['wr']:<6.0f}% ${r['pnl_week']:<7.0f} {exits}")
    
    # Find profitable
    profitable = [r for r in all_results if r['pnl_week'] > 5]
    
    print(f"\n\nPROFITABLE STRATEGIES (>${'5'}/week): {len(profitable)}")
    print("="*70)
    
    if profitable:
        for r in profitable:
            trail_str = f"trail={r['trail']}% @{r['act']}%" if r['trail'] else "no trail"
            print(f"  ✅ {r['strategy']} {r['direction']}: SL={r['sl']}% TP={r['tp']}% {trail_str}")
            print(f"     WR={r['wr']:.0f}%, ${r['pnl_week']:.0f}/wk, {r['trades']} trades")
            print(f"     Avg peak profit: {r['avg_peak']:.2f}%, Realized: {r['avg_realized']:.2f}%, Capture: {r['capture_rate']:.0f}%")
            print()
    
    # Compare trailing vs no trailing
    print("\n" + "="*70)
    print("TRAILING vs NO TRAILING COMPARISON")
    print("="*70)
    
    for sig_col, direction in strategies:
        strat_name = sig_col.replace('sig_', '')
        
        no_trail = [r for r in all_results if r['strategy'] == strat_name and r['direction'] == direction and r['trail'] is None]
        with_trail = [r for r in all_results if r['strategy'] == strat_name and r['direction'] == direction and r['trail'] is not None]
        
        if no_trail and with_trail:
            best_no_trail = max(no_trail, key=lambda x: x['pnl_week'])
            best_with_trail = max(with_trail, key=lambda x: x['pnl_week'])
            
            print(f"\n{strat_name} {direction}:")
            print(f"  No trailing:   ${best_no_trail['pnl_week']:.0f}/wk (SL={best_no_trail['sl']}% TP={best_no_trail['tp']}%)")
            print(f"  With trailing: ${best_with_trail['pnl_week']:.0f}/wk (SL={best_with_trail['sl']}% TP={best_with_trail['tp']}% trail={best_with_trail['trail']}%)")
            
            diff = best_with_trail['pnl_week'] - best_no_trail['pnl_week']
            if diff > 0:
                print(f"  → Trailing IMPROVES by ${diff:.0f}/wk")
            else:
                print(f"  → Trailing WORSE by ${-diff:.0f}/wk")
    
    # Final recommendation
    print("\n" + "="*70)
    print("FINAL RECOMMENDATION")
    print("="*70)
    
    if profitable:
        best = profitable[0]
        trail_str = f"trail={best['trail']}% @{best['act']}%" if best['trail'] else "no trailing"
        
        print(f"\n  BEST STRATEGY: {best['strategy']} {best['direction']}")
        print(f"  Stop Loss: {best['sl']}%")
        print(f"  Take Profit: {best['tp']}%")
        print(f"  Trailing: {trail_str}")
        print(f"  Win Rate: {best['wr']:.1f}%")
        print(f"  Expected P&L: ${best['pnl_week']:.0f}/week")
        print(f"  Profit Capture: {best['capture_rate']:.0f}% of peak")
    else:
        print("\n  No profitable strategy found even with trailing stops.")
        print("  The 15-sec data confirms: no edge exists in current market conditions.")
    
    return all_results


if __name__ == "__main__":
    results = main()
