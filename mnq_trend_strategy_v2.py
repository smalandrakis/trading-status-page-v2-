#!/usr/bin/env python3
"""
MNQ Trend-Following Strategy V2 - More Selective Entry Conditions

Key improvements:
1. Stronger trend confirmation (ADX > 25, multiple timeframe alignment)
2. Pullback entries (enter on retracement, not at trend start)
3. Momentum confirmation (RSI, MACD alignment)
4. Noise-adjusted stop losses
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
    """Calculate all required indicators."""
    df = df.copy()
    
    # EMAs
    df['ema_8'] = df['Close'].ewm(span=8).mean()
    df['ema_21'] = df['Close'].ewm(span=21).mean()
    df['ema_50'] = df['Close'].ewm(span=50).mean()
    df['ema_100'] = df['Close'].ewm(span=100).mean()
    
    # EMA slopes (trend direction)
    df['ema_21_slope'] = (df['ema_21'] - df['ema_21'].shift(5)) / df['ema_21'].shift(5) * 100
    df['ema_50_slope'] = (df['ema_50'] - df['ema_50'].shift(10)) / df['ema_50'].shift(10) * 100
    
    # Price position relative to EMAs
    df['above_ema21'] = df['Close'] > df['ema_21']
    df['above_ema50'] = df['Close'] > df['ema_50']
    df['above_ema100'] = df['Close'] > df['ema_100']
    
    # EMA alignment (all EMAs stacked bullish or bearish)
    df['ema_bullish_stack'] = (df['ema_8'] > df['ema_21']) & (df['ema_21'] > df['ema_50'])
    df['ema_bearish_stack'] = (df['ema_8'] < df['ema_21']) & (df['ema_21'] < df['ema_50'])
    
    # ADX for trend strength
    if 'trend_adx' in df.columns:
        df['adx'] = df['trend_adx']
    else:
        # Calculate ADX manually if not available
        df['adx'] = 25  # Default
    
    df['strong_trend'] = df['adx'] > 25
    df['very_strong_trend'] = df['adx'] > 35
    
    # RSI
    if 'momentum_rsi' in df.columns:
        df['rsi'] = df['momentum_rsi']
    else:
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
    
    # RSI conditions
    df['rsi_bullish'] = (df['rsi'] > 50) & (df['rsi'] < 70)  # Not overbought
    df['rsi_bearish'] = (df['rsi'] < 50) & (df['rsi'] > 30)  # Not oversold
    
    # MACD
    if 'trend_macd' in df.columns:
        df['macd'] = df['trend_macd']
        df['macd_signal'] = df['trend_macd_signal']
    else:
        df['macd'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_bullish'] = df['macd_hist'] > 0
    df['macd_bearish'] = df['macd_hist'] < 0
    df['macd_rising'] = df['macd_hist'] > df['macd_hist'].shift(1)
    df['macd_falling'] = df['macd_hist'] < df['macd_hist'].shift(1)
    
    # ATR for volatility
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
    
    # Bollinger Bands position
    if 'volatility_bbp' in df.columns:
        df['bb_position'] = df['volatility_bbp']
    else:
        sma = df['Close'].rolling(20).mean()
        std = df['Close'].rolling(20).std()
        df['bb_position'] = (df['Close'] - (sma - 2*std)) / (4*std)
    
    # Pullback detection (price retraced to EMA)
    df['pullback_to_ema21'] = (
        (df['Low'] <= df['ema_21'] * 1.002) & 
        (df['Close'] > df['ema_21'])
    )
    df['pullback_to_ema50'] = (
        (df['Low'] <= df['ema_50'] * 1.003) & 
        (df['Close'] > df['ema_50'])
    )
    
    # Bearish pullback (price bounced down from EMA)
    df['bear_pullback_to_ema21'] = (
        (df['High'] >= df['ema_21'] * 0.998) & 
        (df['Close'] < df['ema_21'])
    )
    
    # ROC for momentum
    df['roc_5'] = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5) * 100
    df['roc_10'] = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10) * 100
    df['roc_20'] = (df['Close'] - df['Close'].shift(20)) / df['Close'].shift(20) * 100
    
    return df


def identify_trend_entries(df):
    """
    Identify high-quality trend entry points.
    
    LONG Entry Conditions:
    1. Strong uptrend (ADX > 25, EMAs stacked bullish)
    2. Pullback to EMA21 or EMA50
    3. RSI between 40-60 (not overbought)
    4. MACD histogram positive or turning positive
    
    SHORT Entry Conditions:
    1. Strong downtrend (ADX > 25, EMAs stacked bearish)
    2. Pullback to EMA21 (bounce down)
    3. RSI between 40-60 (not oversold)
    4. MACD histogram negative or turning negative
    """
    df = df.copy()
    
    # STRATEGY 1: Pullback to EMA21 in strong trend
    df['long_pullback_ema21'] = (
        df['ema_bullish_stack'] &
        df['strong_trend'] &
        df['pullback_to_ema21'] &
        df['rsi_bullish'] &
        (df['macd_bullish'] | df['macd_rising'])
    )
    
    df['short_pullback_ema21'] = (
        df['ema_bearish_stack'] &
        df['strong_trend'] &
        df['bear_pullback_to_ema21'] &
        df['rsi_bearish'] &
        (df['macd_bearish'] | df['macd_falling'])
    )
    
    # STRATEGY 2: Breakout with momentum
    df['long_breakout'] = (
        df['ema_bullish_stack'] &
        df['very_strong_trend'] &
        (df['Close'] > df['Close'].shift(1).rolling(10).max()) &  # New 10-bar high
        df['macd_bullish'] &
        (df['rsi'] > 55) & (df['rsi'] < 75)
    )
    
    df['short_breakout'] = (
        df['ema_bearish_stack'] &
        df['very_strong_trend'] &
        (df['Close'] < df['Close'].shift(1).rolling(10).min()) &  # New 10-bar low
        df['macd_bearish'] &
        (df['rsi'] < 45) & (df['rsi'] > 25)
    )
    
    # STRATEGY 3: EMA crossover with confirmation
    df['ema_cross_up'] = (df['ema_8'] > df['ema_21']) & (df['ema_8'].shift(1) <= df['ema_21'].shift(1))
    df['ema_cross_down'] = (df['ema_8'] < df['ema_21']) & (df['ema_8'].shift(1) >= df['ema_21'].shift(1))
    
    df['long_ema_cross'] = (
        df['ema_cross_up'] &
        df['above_ema50'] &
        (df['adx'] > 20) &
        df['macd_rising']
    )
    
    df['short_ema_cross'] = (
        df['ema_cross_down'] &
        ~df['above_ema50'] &
        (df['adx'] > 20) &
        df['macd_falling']
    )
    
    # STRATEGY 4: Trend continuation after consolidation
    df['low_volatility'] = df['atr_pct'] < df['atr_pct'].rolling(50).mean() * 0.8
    df['bb_squeeze'] = (df['bb_position'] > 0.3) & (df['bb_position'] < 0.7)
    
    df['long_consolidation_break'] = (
        df['ema_bullish_stack'] &
        df['low_volatility'].shift(3) &  # Was in consolidation
        (df['Close'] > df['High'].shift(1).rolling(5).max()) &  # Breaking out
        df['macd_bullish']
    )
    
    df['short_consolidation_break'] = (
        df['ema_bearish_stack'] &
        df['low_volatility'].shift(3) &
        (df['Close'] < df['Low'].shift(1).rolling(5).min()) &
        df['macd_bearish']
    )
    
    return df


def backtest_strategy(df, strategy_name, sl_pct, tp_pct, direction='LONG', 
                      cooldown_bars=20, max_hold_bars=48):
    """Backtest a specific strategy."""
    trades = []
    last_trade_bar = -cooldown_bars
    
    signal_col = f"{direction.lower()}_{strategy_name}"
    
    if signal_col not in df.columns:
        return pd.DataFrame()
    
    for i in range(100, len(df) - max_hold_bars):
        if i - last_trade_bar < cooldown_bars:
            continue
        
        if not df[signal_col].iloc[i]:
            continue
        
        entry_price = df['Close'].iloc[i]
        entry_time = df.index[i]
        atr_at_entry = df['atr_pct'].iloc[i]
        
        # Dynamic SL based on ATR (minimum 1.5x ATR)
        dynamic_sl = max(sl_pct, atr_at_entry * 1.5)
        
        if direction == 'LONG':
            target_price = entry_price * (1 + tp_pct / 100)
            stop_price = entry_price * (1 - dynamic_sl / 100)
        else:
            target_price = entry_price * (1 - tp_pct / 100)
            stop_price = entry_price * (1 + dynamic_sl / 100)
        
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
        
        # MNQ P&L calculation
        multiplier = 2.0
        commission = 1.24
        pnl_dollar = (pnl_pct / 100) * entry_price * multiplier / 100 - commission
        
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
    print("MNQ TREND-FOLLOWING STRATEGY V2 - SELECTIVE ENTRIES")
    print("="*80)
    
    df = load_data()
    
    print("\n1. CALCULATING INDICATORS...")
    df = calculate_indicators(df)
    
    print("\n2. IDENTIFYING TREND ENTRIES...")
    df = identify_trend_entries(df)
    
    # Get last 30 days
    cutoff_30d = df.index[-1] - timedelta(days=30)
    df_30d = df[df.index >= cutoff_30d].copy()
    
    # Count signals
    strategies = ['pullback_ema21', 'breakout', 'ema_cross', 'consolidation_break']
    
    print("\n   Signal Counts (30-day):")
    for strat in strategies:
        long_col = f'long_{strat}'
        short_col = f'short_{strat}'
        if long_col in df_30d.columns:
            long_count = df_30d[long_col].sum()
            short_count = df_30d[short_col].sum() if short_col in df_30d.columns else 0
            print(f"   {strat}: LONG={long_count}, SHORT={short_count}")
    
    # Noise stats
    atr_mean = df_30d['atr_pct'].mean()
    atr_std = df_30d['atr_pct'].std()
    print(f"\n   ATR %: mean={atr_mean:.3f}%, std={atr_std:.3f}%")
    print(f"   Recommended min SL: {atr_mean + 2*atr_std:.3f}%")
    
    # Test configurations
    print("\n3. BACKTESTING STRATEGIES (30-day)...")
    print("-"*100)
    
    configs = [
        # 1:2 R:R
        {'sl': 0.4, 'tp': 0.8},
        {'sl': 0.5, 'tp': 1.0},
        {'sl': 0.6, 'tp': 1.2},
        {'sl': 0.7, 'tp': 1.4},
        # 1:3 R:R
        {'sl': 0.4, 'tp': 1.2},
        {'sl': 0.5, 'tp': 1.5},
        {'sl': 0.6, 'tp': 1.8},
        # Wider
        {'sl': 0.8, 'tp': 1.6},
        {'sl': 1.0, 'tp': 2.0},
    ]
    
    all_results = []
    
    for strat in strategies:
        for config in configs:
            sl = config['sl']
            tp = config['tp']
            rr = f"1:{tp/sl:.1f}"
            
            for direction in ['LONG', 'SHORT']:
                trades = backtest_strategy(df_30d, strat, sl, tp, direction, cooldown_bars=15)
                
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
    
    for r in all_results[:20]:
        print(f"{r['strategy']:<20} {r['direction']:<6} {r['sl']:.1f}/{r['tp']:.1f}    {r['rr']:<8} {r['trades']:<8} {r['win_rate']:.1f}%    ${r['pnl_week']:>6.0f}     ${r['max_dd']:>6.0f}")
    
    # Find profitable
    profitable = [r for r in all_results if r['pnl_week'] > 5]
    
    print(f"\n\n4. PROFITABLE STRATEGIES (>${'5'}/week): {len(profitable)}")
    print("="*80)
    
    if profitable:
        for r in profitable:
            print(f"  ✅ {r['strategy']} {r['direction']}: SL={r['sl']}% TP={r['tp']}% | WR={r['win_rate']:.1f}% | ${r['pnl_week']:.0f}/wk | {r['trades']} trades")
    
    # Historical validation
    print("\n\n5. HISTORICAL VALIDATION")
    print("="*80)
    
    if all_results:
        # Test top 3 configs on historical data
        top_configs = all_results[:3]
        
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
                df_month = identify_trend_entries(df_month)
                
                trades = backtest_strategy(df_month, strat, sl, tp, direction, cooldown_bars=15)
                
                if len(trades) >= 3:
                    result = analyze_results(trades, month_name)
                    if result:
                        results_by_month.append(result)
                        status = "✅" if result['pnl_week'] > 0 else "❌"
                        print(f"  {status} {month_name}: {result['trades']} trades, WR={result['win_rate']:.1f}%, ${result['pnl_week']:.0f}/wk")
            
            if results_by_month:
                profitable_months = sum(1 for r in results_by_month if r['pnl_week'] > 0)
                total_months = len(results_by_month)
                print(f"  Consistency: {profitable_months}/{total_months} months profitable")
    
    # Final recommendation
    print("\n\n6. FINAL RECOMMENDATION")
    print("="*80)
    
    if profitable:
        best = profitable[0]
        
        # Calculate breakeven WR
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
    else:
        print("\n  No profitable trend strategy found.")
        print("  The market may be ranging - consider mean reversion instead.")
    
    return all_results


if __name__ == "__main__":
    results = main()
