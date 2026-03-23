#!/usr/bin/env python3
"""
MNQ Mean Reversion Strategy with Noise Analysis

Since trend-following doesn't work in current market conditions,
try mean reversion: buy oversold, sell overbought.

Key approach:
1. Identify extreme moves (oversold/overbought)
2. Calculate noise to set proper SL that won't get hit by random fluctuations
3. Target reversion to mean with proper R:R
4. Validate across multiple time periods
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


def calculate_indicators(df):
    """Calculate mean reversion indicators."""
    df = df.copy()
    
    # Bollinger Bands
    df['sma_20'] = df['Close'].rolling(20).mean()
    df['std_20'] = df['Close'].rolling(20).std()
    df['bb_upper'] = df['sma_20'] + 2 * df['std_20']
    df['bb_lower'] = df['sma_20'] - 2 * df['std_20']
    df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # RSI
    if 'momentum_rsi' in df.columns:
        df['rsi'] = df['momentum_rsi']
    else:
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
    
    # Stochastic
    if 'momentum_stoch' in df.columns:
        df['stoch'] = df['momentum_stoch']
    else:
        low_14 = df['Low'].rolling(14).min()
        high_14 = df['High'].rolling(14).max()
        df['stoch'] = (df['Close'] - low_14) / (high_14 - low_14) * 100
    
    # Williams %R
    if 'momentum_wr' in df.columns:
        df['williams_r'] = df['momentum_wr']
    else:
        high_14 = df['High'].rolling(14).max()
        low_14 = df['Low'].rolling(14).min()
        df['williams_r'] = (high_14 - df['Close']) / (high_14 - low_14) * -100
    
    # ATR for noise
    if 'volatility_atr' in df.columns:
        df['atr'] = df['volatility_atr']
    else:
        tr = pd.concat([
            df['High'] - df['Low'],
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
    
    df['atr_pct'] = df['atr'] / df['Close'] * 100
    
    # Distance from SMA (z-score)
    df['zscore'] = (df['Close'] - df['sma_20']) / df['std_20']
    
    # Price momentum (for confirmation)
    df['roc_3'] = (df['Close'] - df['Close'].shift(3)) / df['Close'].shift(3) * 100
    df['roc_5'] = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5) * 100
    
    # Candle patterns
    df['body'] = df['Close'] - df['Open']
    df['body_pct'] = abs(df['body']) / df['Close'] * 100
    df['upper_wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['lower_wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    
    # Bullish reversal candle (hammer-like)
    df['bullish_reversal_candle'] = (
        (df['lower_wick'] > df['body'].abs() * 2) &
        (df['Close'] > df['Open'])
    )
    
    # Bearish reversal candle (shooting star-like)
    df['bearish_reversal_candle'] = (
        (df['upper_wick'] > df['body'].abs() * 2) &
        (df['Close'] < df['Open'])
    )
    
    # Consecutive down/up bars
    df['down_bar'] = df['Close'] < df['Open']
    df['up_bar'] = df['Close'] > df['Open']
    df['consec_down'] = df['down_bar'].rolling(3).sum()
    df['consec_up'] = df['up_bar'].rolling(3).sum()
    
    return df


def identify_mean_reversion_signals(df):
    """Identify mean reversion entry signals."""
    df = df.copy()
    
    # STRATEGY 1: RSI Oversold/Overbought
    df['long_rsi_oversold'] = (
        (df['rsi'] < 30) &
        (df['rsi'] > df['rsi'].shift(1))  # RSI turning up
    )
    
    df['short_rsi_overbought'] = (
        (df['rsi'] > 70) &
        (df['rsi'] < df['rsi'].shift(1))  # RSI turning down
    )
    
    # STRATEGY 2: BB Touch with RSI confirmation
    df['long_bb_touch'] = (
        (df['bb_position'] < 0.1) &  # Near lower BB
        (df['rsi'] < 40) &
        (df['Close'] > df['Low'].shift(1))  # Not making new low
    )
    
    df['short_bb_touch'] = (
        (df['bb_position'] > 0.9) &  # Near upper BB
        (df['rsi'] > 60) &
        (df['Close'] < df['High'].shift(1))  # Not making new high
    )
    
    # STRATEGY 3: Z-score extreme
    df['long_zscore_extreme'] = (
        (df['zscore'] < -2) &
        (df['rsi'] < 35)
    )
    
    df['short_zscore_extreme'] = (
        (df['zscore'] > 2) &
        (df['rsi'] > 65)
    )
    
    # STRATEGY 4: Double bottom/top pattern (simplified)
    df['recent_low'] = df['Low'].rolling(10).min()
    df['recent_high'] = df['High'].rolling(10).max()
    
    df['long_double_bottom'] = (
        (df['Low'] <= df['recent_low'] * 1.002) &  # Near recent low
        (df['Close'] > df['Open']) &  # Bullish close
        (df['rsi'] < 40) &
        (df['consec_down'] >= 2)  # After down move
    )
    
    df['short_double_top'] = (
        (df['High'] >= df['recent_high'] * 0.998) &  # Near recent high
        (df['Close'] < df['Open']) &  # Bearish close
        (df['rsi'] > 60) &
        (df['consec_up'] >= 2)  # After up move
    )
    
    # STRATEGY 5: Stochastic oversold/overbought with reversal candle
    df['long_stoch_reversal'] = (
        (df['stoch'] < 20) &
        df['bullish_reversal_candle']
    )
    
    df['short_stoch_reversal'] = (
        (df['stoch'] > 80) &
        df['bearish_reversal_candle']
    )
    
    # STRATEGY 6: Williams %R extreme
    df['long_williams'] = (
        (df['williams_r'] < -80) &
        (df['williams_r'] > df['williams_r'].shift(1))  # Turning up
    )
    
    df['short_williams'] = (
        (df['williams_r'] > -20) &
        (df['williams_r'] < df['williams_r'].shift(1))  # Turning down
    )
    
    # STRATEGY 7: Combined oversold (multiple indicators)
    df['long_multi_oversold'] = (
        (df['rsi'] < 35) &
        (df['bb_position'] < 0.2) &
        (df['stoch'] < 25)
    )
    
    df['short_multi_overbought'] = (
        (df['rsi'] > 65) &
        (df['bb_position'] > 0.8) &
        (df['stoch'] > 75)
    )
    
    return df


def backtest_strategy(df, strategy_name, sl_pct, tp_pct, direction='LONG', 
                      cooldown_bars=10, max_hold_bars=48):
    """Backtest mean reversion strategy."""
    trades = []
    last_trade_bar = -cooldown_bars
    
    signal_col = f"{direction.lower()}_{strategy_name}"
    
    if signal_col not in df.columns:
        return pd.DataFrame()
    
    for i in range(50, len(df) - max_hold_bars):
        if i - last_trade_bar < cooldown_bars:
            continue
        
        if not df[signal_col].iloc[i]:
            continue
        
        entry_price = df['Close'].iloc[i]
        entry_time = df.index[i]
        atr_at_entry = df['atr_pct'].iloc[i]
        
        # Use ATR-based SL (minimum 1.5x ATR to avoid noise)
        effective_sl = max(sl_pct, atr_at_entry * 1.5)
        
        if direction == 'LONG':
            target_price = entry_price * (1 + tp_pct / 100)
            stop_price = entry_price * (1 - effective_sl / 100)
        else:
            target_price = entry_price * (1 - tp_pct / 100)
            stop_price = entry_price * (1 + effective_sl / 100)
        
        exit_price = None
        exit_reason = None
        bars_held = 0
        
        for j in range(i + 1, min(i + max_hold_bars + 1, len(df))):
            bars_held = j - i
            row = df.iloc[j]
            
            if direction == 'LONG':
                if row['Low'] <= stop_price:
                    exit_price = stop_price
                    exit_reason = 'SL'
                    break
                if row['High'] >= target_price:
                    exit_price = target_price
                    exit_reason = 'TP'
                    break
            else:
                if row['High'] >= stop_price:
                    exit_price = stop_price
                    exit_reason = 'SL'
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
        
        # MNQ P&L: $2 per point, price ~$25000, so 1% = $500
        # Simplified: pnl_dollar = pnl_pct * 5 - commission
        multiplier = 5.0  # Approximate $ per 0.1% move
        commission = 1.24
        pnl_dollar = pnl_pct * multiplier - commission
        
        trades.append({
            'entry_time': entry_time,
            'direction': direction,
            'strategy': strategy_name,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'pnl_pct': pnl_pct,
            'pnl_dollar': pnl_dollar,
            'bars_held': bars_held,
            'atr_at_entry': atr_at_entry
        })
        
        last_trade_bar = i
    
    return pd.DataFrame(trades)


def analyze_results(trades_df, period_name=""):
    """Analyze backtest results."""
    if trades_df.empty or len(trades_df) < 3:
        return None
    
    total = len(trades_df)
    wins = len(trades_df[trades_df['pnl_dollar'] > 0])
    losses = len(trades_df[trades_df['pnl_dollar'] <= 0])
    win_rate = wins / total * 100
    
    total_pnl = trades_df['pnl_dollar'].sum()
    avg_win = trades_df[trades_df['pnl_dollar'] > 0]['pnl_dollar'].mean() if wins > 0 else 0
    avg_loss = trades_df[trades_df['pnl_dollar'] <= 0]['pnl_dollar'].mean() if losses > 0 else 0
    
    days = max(1, (trades_df['entry_time'].max() - trades_df['entry_time'].min()).days)
    pnl_per_week = total_pnl / days * 7
    trades_per_day = total / days
    
    exit_reasons = trades_df['exit_reason'].value_counts().to_dict()
    
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
    print("MNQ MEAN REVERSION STRATEGY ANALYSIS")
    print("="*80)
    
    df = load_data()
    
    print("\n1. CALCULATING INDICATORS...")
    df = calculate_indicators(df)
    
    print("\n2. IDENTIFYING MEAN REVERSION SIGNALS...")
    df = identify_mean_reversion_signals(df)
    
    # Get last 30 days
    cutoff_30d = df.index[-1] - timedelta(days=30)
    df_30d = df[df.index >= cutoff_30d].copy()
    
    # Noise analysis
    atr_mean = df_30d['atr_pct'].mean()
    atr_std = df_30d['atr_pct'].std()
    recommended_sl = atr_mean + 2 * atr_std
    
    print(f"\n   Noise Analysis (30-day):")
    print(f"   ATR %: mean={atr_mean:.3f}%, std={atr_std:.3f}%")
    print(f"   Recommended min SL: {recommended_sl:.3f}% (2σ above mean)")
    
    # Count signals
    strategies = ['rsi_oversold', 'rsi_overbought', 'bb_touch', 'zscore_extreme', 
                  'double_bottom', 'double_top', 'stoch_reversal', 'williams', 'multi_oversold', 'multi_overbought']
    
    print(f"\n   Signal Counts (30-day):")
    for strat in strategies:
        long_col = f'long_{strat}'
        short_col = f'short_{strat}'
        if long_col in df_30d.columns:
            count = df_30d[long_col].sum()
            print(f"   {strat}: {count}")
        elif short_col in df_30d.columns:
            count = df_30d[short_col].sum()
            print(f"   {strat}: {count}")
    
    # Test configurations
    print("\n3. BACKTESTING STRATEGIES (30-day)...")
    print("-"*100)
    
    configs = [
        # 1:2 R:R
        {'sl': 0.3, 'tp': 0.6},
        {'sl': 0.4, 'tp': 0.8},
        {'sl': 0.5, 'tp': 1.0},
        # 1:3 R:R
        {'sl': 0.3, 'tp': 0.9},
        {'sl': 0.4, 'tp': 1.2},
        # Wider SL
        {'sl': 0.6, 'tp': 1.2},
        {'sl': 0.7, 'tp': 1.4},
        {'sl': 0.8, 'tp': 1.6},
    ]
    
    all_results = []
    
    # Map strategies to directions
    strategy_directions = {
        'rsi_oversold': 'LONG',
        'rsi_overbought': 'SHORT',
        'bb_touch': 'BOTH',
        'zscore_extreme': 'BOTH',
        'double_bottom': 'LONG',
        'double_top': 'SHORT',
        'stoch_reversal': 'BOTH',
        'williams': 'BOTH',
        'multi_oversold': 'LONG',
        'multi_overbought': 'SHORT',
    }
    
    for strat, dirs in strategy_directions.items():
        directions = ['LONG', 'SHORT'] if dirs == 'BOTH' else [dirs]
        
        for direction in directions:
            for config in configs:
                sl = config['sl']
                tp = config['tp']
                rr = f"1:{tp/sl:.1f}"
                
                trades = backtest_strategy(df_30d, strat, sl, tp, direction, cooldown_bars=8)
                
                if len(trades) >= 5:
                    result = analyze_results(trades, f"30d")
                    if result:
                        result['strategy'] = strat
                        result['sl'] = sl
                        result['tp'] = tp
                        result['rr'] = rr
                        result['direction'] = direction
                        all_results.append(result)
    
    # Sort by P&L
    all_results.sort(key=lambda x: x['pnl_week'], reverse=True)
    
    print(f"\n{'Strategy':<20} {'Dir':<6} {'SL/TP':<10} {'R:R':<8} {'Trades':<8} {'WR%':<8} {'$/Week':<10} {'MaxDD'}")
    print("-"*100)
    
    for r in all_results[:25]:
        print(f"{r['strategy']:<20} {r['direction']:<6} {r['sl']:.1f}/{r['tp']:.1f}    {r['rr']:<8} {r['trades']:<8} {r['win_rate']:.1f}%    ${r['pnl_week']:>6.0f}     ${r['max_dd']:>6.0f}")
    
    # Find profitable
    profitable = [r for r in all_results if r['pnl_week'] > 10]
    
    print(f"\n\n4. PROFITABLE STRATEGIES (>${'10'}/week): {len(profitable)}")
    print("="*80)
    
    if profitable:
        for r in profitable:
            exits = ', '.join([f"{k}:{v}" for k, v in r['exit_reasons'].items()])
            print(f"  ✅ {r['strategy']} {r['direction']}: SL={r['sl']}% TP={r['tp']}% | WR={r['win_rate']:.1f}% | ${r['pnl_week']:.0f}/wk | {r['trades']} trades | {exits}")
    
    # Historical validation
    print("\n\n5. HISTORICAL VALIDATION")
    print("="*80)
    
    if all_results:
        # Test top 5 configs
        top_configs = all_results[:5]
        
        months = [
            ('Dec 2025', df.index[-1] - timedelta(days=60), df.index[-1] - timedelta(days=30)),
            ('Nov 2025', df.index[-1] - timedelta(days=90), df.index[-1] - timedelta(days=60)),
            ('Oct 2025', df.index[-1] - timedelta(days=120), df.index[-1] - timedelta(days=90)),
        ]
        
        for cfg in top_configs:
            strat = cfg['strategy']
            direction = cfg['direction']
            sl = cfg['sl']
            tp = cfg['tp']
            
            print(f"\n  Testing: {strat} {direction} SL={sl}% TP={tp}%")
            print(f"  {'-'*50}")
            
            results_by_month = []
            
            for month_name, start, end in months:
                df_month = df[(df.index >= start) & (df.index < end)].copy()
                if len(df_month) < 100:
                    continue
                
                df_month = calculate_indicators(df_month)
                df_month = identify_mean_reversion_signals(df_month)
                
                trades = backtest_strategy(df_month, strat, sl, tp, direction, cooldown_bars=8)
                
                if len(trades) >= 3:
                    result = analyze_results(trades, month_name)
                    if result:
                        results_by_month.append(result)
                        status = "✅" if result['pnl_week'] > 0 else "❌"
                        print(f"  {status} {month_name}: {result['trades']} trades, WR={result['win_rate']:.1f}%, ${result['pnl_week']:.0f}/wk")
            
            if results_by_month:
                profitable_months = sum(1 for r in results_by_month if r['pnl_week'] > 0)
                total_months = len(results_by_month)
                avg_pnl = sum(r['pnl_week'] for r in results_by_month) / total_months
                print(f"  Consistency: {profitable_months}/{total_months} months profitable, avg ${avg_pnl:.0f}/wk")
    
    # Final recommendation
    print("\n\n6. FINAL RECOMMENDATION")
    print("="*80)
    
    if profitable:
        best = profitable[0]
        
        rr_ratio = best['tp'] / best['sl']
        breakeven_wr = 1 / (1 + rr_ratio) * 100
        
        print(f"\n  RECOMMENDED STRATEGY: {best['strategy']}")
        print(f"  Direction: {best['direction']}")
        print(f"  Stop Loss: {best['sl']}%")
        print(f"  Take Profit: {best['tp']}%")
        print(f"  R:R Ratio: {best['rr']}")
        print(f"  Expected Win Rate: {best['win_rate']:.1f}%")
        print(f"  Breakeven Win Rate: {breakeven_wr:.1f}%")
        print(f"  Win Rate Buffer: {best['win_rate'] - breakeven_wr:.1f}%")
        print(f"  Expected P&L: ${best['pnl_week']:.0f}/week")
        print(f"  Trades/Day: {best['trades_day']:.2f}")
        print(f"  Max Drawdown: ${best['max_dd']:.0f}")
        print(f"  Exit Breakdown: {best['exit_reasons']}")
    else:
        print("\n  No profitable mean reversion strategy found in 30-day data.")
        print("  Market conditions may not favor mean reversion either.")
    
    return all_results


if __name__ == "__main__":
    results = main()
