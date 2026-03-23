#!/usr/bin/env python3
"""
MNQ Comprehensive Strategy Search

Test ALL strategy types across MULTIPLE time periods to find what actually works.
Focus on noise-adjusted stops and proper R:R ratios.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

QQQ_PARQUET = "data/QQQ_features.parquet"

def load_data():
    df = pd.read_parquet(QQQ_PARQUET)
    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    return df


def calculate_all_indicators(df):
    """Calculate comprehensive indicators."""
    df = df.copy()
    
    # EMAs
    for span in [8, 12, 21, 26, 50]:
        df[f'ema_{span}'] = df['Close'].ewm(span=span).mean()
    
    # RSI
    df['rsi'] = df['momentum_rsi'] if 'momentum_rsi' in df.columns else 50
    
    # MACD
    df['macd'] = df['trend_macd'] if 'trend_macd' in df.columns else 0
    df['macd_signal'] = df['trend_macd_signal'] if 'trend_macd_signal' in df.columns else 0
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_position'] = df['volatility_bbp'] if 'volatility_bbp' in df.columns else 0.5
    
    # ATR
    df['atr'] = df['volatility_atr'] if 'volatility_atr' in df.columns else df['High'] - df['Low']
    df['atr_pct'] = df['atr'] / df['Close'] * 100
    
    # ADX
    df['adx'] = df['trend_adx'] if 'trend_adx' in df.columns else 25
    
    # Stochastic
    df['stoch'] = df['momentum_stoch'] if 'momentum_stoch' in df.columns else 50
    
    # ROC
    for period in [3, 5, 10, 20]:
        df[f'roc_{period}'] = (df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period) * 100
    
    # Price vs EMAs
    df['above_ema21'] = df['Close'] > df['ema_21']
    df['above_ema50'] = df['Close'] > df['ema_50']
    
    # Volatility regime
    df['atr_sma'] = df['atr_pct'].rolling(50).mean()
    df['high_vol'] = df['atr_pct'] > df['atr_sma'] * 1.2
    df['low_vol'] = df['atr_pct'] < df['atr_sma'] * 0.8
    
    return df


def generate_all_signals(df):
    """Generate signals for all strategy types."""
    df = df.copy()
    
    # ===== MOMENTUM STRATEGIES =====
    
    # Strong momentum LONG
    df['sig_momentum_long'] = (
        (df['roc_5'] > 0.3) &
        (df['macd_hist'] > 0) &
        (df['rsi'] > 50) & (df['rsi'] < 70)
    )
    
    # Strong momentum SHORT
    df['sig_momentum_short'] = (
        (df['roc_5'] < -0.3) &
        (df['macd_hist'] < 0) &
        (df['rsi'] < 50) & (df['rsi'] > 30)
    )
    
    # ===== MEAN REVERSION =====
    
    # RSI oversold bounce
    df['sig_rsi_oversold'] = (
        (df['rsi'] < 30) &
        (df['rsi'] > df['rsi'].shift(1))
    )
    
    # RSI overbought fade
    df['sig_rsi_overbought'] = (
        (df['rsi'] > 70) &
        (df['rsi'] < df['rsi'].shift(1))
    )
    
    # BB lower touch
    df['sig_bb_lower'] = (
        (df['bb_position'] < 0.1) &
        (df['Close'] > df['Close'].shift(1))
    )
    
    # BB upper touch
    df['sig_bb_upper'] = (
        (df['bb_position'] > 0.9) &
        (df['Close'] < df['Close'].shift(1))
    )
    
    # ===== TREND FOLLOWING =====
    
    # EMA crossover up
    df['sig_ema_cross_up'] = (
        (df['ema_12'] > df['ema_26']) &
        (df['ema_12'].shift(1) <= df['ema_26'].shift(1)) &
        (df['adx'] > 20)
    )
    
    # EMA crossover down
    df['sig_ema_cross_down'] = (
        (df['ema_12'] < df['ema_26']) &
        (df['ema_12'].shift(1) >= df['ema_26'].shift(1)) &
        (df['adx'] > 20)
    )
    
    # ===== BREAKOUT =====
    
    # High breakout
    df['recent_high'] = df['High'].rolling(20).max()
    df['sig_breakout_long'] = (
        (df['Close'] > df['recent_high'].shift(1)) &
        (df['roc_5'] > 0.2)
    )
    
    # Low breakout
    df['recent_low'] = df['Low'].rolling(20).min()
    df['sig_breakout_short'] = (
        (df['Close'] < df['recent_low'].shift(1)) &
        (df['roc_5'] < -0.2)
    )
    
    # ===== VOLATILITY BASED =====
    
    # Low vol expansion LONG
    df['sig_vol_expand_long'] = (
        df['low_vol'].shift(3) &
        (df['atr_pct'] > df['atr_sma']) &
        (df['roc_3'] > 0.2)
    )
    
    # Low vol expansion SHORT
    df['sig_vol_expand_short'] = (
        df['low_vol'].shift(3) &
        (df['atr_pct'] > df['atr_sma']) &
        (df['roc_3'] < -0.2)
    )
    
    # ===== COMBINED SIGNALS =====
    
    # Multi-indicator LONG
    df['sig_multi_long'] = (
        (df['rsi'] < 40) &
        (df['bb_position'] < 0.3) &
        (df['macd_hist'] > df['macd_hist'].shift(1))  # MACD improving
    )
    
    # Multi-indicator SHORT
    df['sig_multi_short'] = (
        (df['rsi'] > 60) &
        (df['bb_position'] > 0.7) &
        (df['macd_hist'] < df['macd_hist'].shift(1))  # MACD weakening
    )
    
    return df


def backtest(df, signal_col, direction, sl_pct, tp_pct, cooldown=10, max_hold=48):
    """Run backtest for a signal."""
    trades = []
    last_bar = -cooldown
    
    for i in range(50, len(df) - max_hold):
        if i - last_bar < cooldown:
            continue
        
        if not df[signal_col].iloc[i]:
            continue
        
        entry = df['Close'].iloc[i]
        entry_time = df.index[i]
        atr = df['atr_pct'].iloc[i]
        
        # Dynamic SL: at least 1.5x ATR
        eff_sl = max(sl_pct, atr * 1.5)
        
        if direction == 'LONG':
            tp_price = entry * (1 + tp_pct / 100)
            sl_price = entry * (1 - eff_sl / 100)
        else:
            tp_price = entry * (1 - tp_pct / 100)
            sl_price = entry * (1 + eff_sl / 100)
        
        exit_price = None
        exit_reason = None
        
        for j in range(i + 1, min(i + max_hold + 1, len(df))):
            row = df.iloc[j]
            
            if direction == 'LONG':
                if row['Low'] <= sl_price:
                    exit_price = sl_price
                    exit_reason = 'SL'
                    break
                if row['High'] >= tp_price:
                    exit_price = tp_price
                    exit_reason = 'TP'
                    break
            else:
                if row['High'] >= sl_price:
                    exit_price = sl_price
                    exit_reason = 'SL'
                    break
                if row['Low'] <= tp_price:
                    exit_price = tp_price
                    exit_reason = 'TP'
                    break
        
        if exit_price is None:
            exit_price = df['Close'].iloc[min(i + max_hold, len(df) - 1)]
            exit_reason = 'TO'
        
        if direction == 'LONG':
            pnl_pct = (exit_price / entry - 1) * 100
        else:
            pnl_pct = (entry / exit_price - 1) * 100
        
        # MNQ: ~$5 per 0.1% at $25000
        pnl_dollar = pnl_pct * 5 - 1.24
        
        trades.append({
            'time': entry_time,
            'pnl': pnl_dollar,
            'exit': exit_reason
        })
        
        last_bar = i
    
    return trades


def analyze(trades, days):
    """Analyze trade results."""
    if len(trades) < 5:
        return None
    
    df = pd.DataFrame(trades)
    total = len(df)
    wins = (df['pnl'] > 0).sum()
    wr = wins / total * 100
    total_pnl = df['pnl'].sum()
    pnl_week = total_pnl / max(1, days) * 7
    
    # Max drawdown
    cum = df['pnl'].cumsum()
    dd = (cum - cum.expanding().max()).min()
    
    exits = df['exit'].value_counts().to_dict()
    
    return {
        'trades': total,
        'wr': wr,
        'pnl_week': pnl_week,
        'max_dd': dd,
        'exits': exits
    }


def main():
    print("="*80)
    print("MNQ COMPREHENSIVE STRATEGY SEARCH")
    print("="*80)
    
    df = load_data()
    df = calculate_all_indicators(df)
    df = generate_all_signals(df)
    
    # Define all strategies
    strategies = [
        ('sig_momentum_long', 'LONG'),
        ('sig_momentum_short', 'SHORT'),
        ('sig_rsi_oversold', 'LONG'),
        ('sig_rsi_overbought', 'SHORT'),
        ('sig_bb_lower', 'LONG'),
        ('sig_bb_upper', 'SHORT'),
        ('sig_ema_cross_up', 'LONG'),
        ('sig_ema_cross_down', 'SHORT'),
        ('sig_breakout_long', 'LONG'),
        ('sig_breakout_short', 'SHORT'),
        ('sig_vol_expand_long', 'LONG'),
        ('sig_vol_expand_short', 'SHORT'),
        ('sig_multi_long', 'LONG'),
        ('sig_multi_short', 'SHORT'),
    ]
    
    # SL/TP configs
    configs = [
        (0.3, 0.6),   # 1:2
        (0.4, 0.8),   # 1:2
        (0.5, 1.0),   # 1:2
        (0.3, 0.9),   # 1:3
        (0.4, 1.2),   # 1:3
        (0.5, 1.5),   # 1:3
        (0.6, 1.2),   # 1:2
        (0.7, 1.4),   # 1:2
        (0.8, 1.6),   # 1:2
        (1.0, 2.0),   # 1:2
    ]
    
    # Test periods
    periods = [
        ('Last 30d', df.index[-1] - timedelta(days=30), df.index[-1]),
        ('Dec 2025', df.index[-1] - timedelta(days=60), df.index[-1] - timedelta(days=30)),
        ('Nov 2025', df.index[-1] - timedelta(days=90), df.index[-1] - timedelta(days=60)),
        ('Oct 2025', df.index[-1] - timedelta(days=120), df.index[-1] - timedelta(days=90)),
    ]
    
    all_results = []
    
    print("\nTesting all combinations...")
    
    for period_name, start, end in periods:
        df_period = df[(df.index >= start) & (df.index < end)].copy()
        days = (end - start).days
        
        if len(df_period) < 100:
            continue
        
        for sig_col, direction in strategies:
            for sl, tp in configs:
                trades = backtest(df_period, sig_col, direction, sl, tp)
                result = analyze(trades, days)
                
                if result:
                    result['period'] = period_name
                    result['strategy'] = sig_col.replace('sig_', '')
                    result['direction'] = direction
                    result['sl'] = sl
                    result['tp'] = tp
                    result['rr'] = f"1:{tp/sl:.1f}"
                    all_results.append(result)
    
    # Find strategies profitable in ALL periods
    print("\n" + "="*80)
    print("STRATEGIES BY PERIOD")
    print("="*80)
    
    for period_name, _, _ in periods:
        period_results = [r for r in all_results if r['period'] == period_name]
        period_results.sort(key=lambda x: x['pnl_week'], reverse=True)
        
        profitable = [r for r in period_results if r['pnl_week'] > 5]
        
        print(f"\n{period_name}: {len(profitable)} profitable strategies (>${'5'}/wk)")
        
        for r in profitable[:5]:
            print(f"  {r['strategy']:<20} {r['direction']:<6} SL={r['sl']:.1f} TP={r['tp']:.1f} | WR={r['wr']:.0f}% | ${r['pnl_week']:.0f}/wk | {r['trades']} trades")
    
    # Find consistent strategies (profitable in 3+ periods)
    print("\n" + "="*80)
    print("CONSISTENT STRATEGIES (Profitable in 3+ periods)")
    print("="*80)
    
    # Group by strategy config
    from collections import defaultdict
    strategy_periods = defaultdict(list)
    
    for r in all_results:
        key = (r['strategy'], r['direction'], r['sl'], r['tp'])
        strategy_periods[key].append(r)
    
    consistent = []
    for key, results in strategy_periods.items():
        profitable_periods = [r for r in results if r['pnl_week'] > 0]
        if len(profitable_periods) >= 3:
            avg_pnl = sum(r['pnl_week'] for r in results) / len(results)
            avg_wr = sum(r['wr'] for r in results) / len(results)
            total_trades = sum(r['trades'] for r in results)
            
            consistent.append({
                'strategy': key[0],
                'direction': key[1],
                'sl': key[2],
                'tp': key[3],
                'profitable_periods': len(profitable_periods),
                'total_periods': len(results),
                'avg_pnl_week': avg_pnl,
                'avg_wr': avg_wr,
                'total_trades': total_trades,
                'results': results
            })
    
    consistent.sort(key=lambda x: x['avg_pnl_week'], reverse=True)
    
    if consistent:
        print(f"\nFound {len(consistent)} consistent strategies:\n")
        
        for c in consistent[:10]:
            print(f"  {c['strategy']:<20} {c['direction']:<6} SL={c['sl']:.1f} TP={c['tp']:.1f}")
            print(f"    Profitable: {c['profitable_periods']}/{c['total_periods']} periods")
            print(f"    Avg P&L: ${c['avg_pnl_week']:.0f}/wk, Avg WR: {c['avg_wr']:.0f}%")
            print(f"    Total trades: {c['total_trades']}")
            
            # Show by period
            for r in c['results']:
                status = "✅" if r['pnl_week'] > 0 else "❌"
                print(f"      {status} {r['period']}: ${r['pnl_week']:.0f}/wk, WR={r['wr']:.0f}%")
            print()
    else:
        print("\nNo strategies found profitable in 3+ periods.")
    
    # Final recommendation
    print("\n" + "="*80)
    print("FINAL RECOMMENDATION")
    print("="*80)
    
    if consistent:
        best = consistent[0]
        rr = best['tp'] / best['sl']
        be_wr = 1 / (1 + rr) * 100
        
        print(f"\n  BEST STRATEGY: {best['strategy']}")
        print(f"  Direction: {best['direction']}")
        print(f"  Stop Loss: {best['sl']}%")
        print(f"  Take Profit: {best['tp']}%")
        print(f"  R:R Ratio: 1:{rr:.1f}")
        print(f"  Avg Win Rate: {best['avg_wr']:.1f}%")
        print(f"  Breakeven WR: {be_wr:.1f}%")
        print(f"  Buffer: {best['avg_wr'] - be_wr:.1f}% above breakeven")
        print(f"  Avg P&L: ${best['avg_pnl_week']:.0f}/week")
        print(f"  Consistency: {best['profitable_periods']}/{best['total_periods']} periods profitable")
    else:
        print("\n  No consistently profitable strategy found for MNQ.")
        print("  The market may be in a difficult regime for systematic trading.")
        print("\n  Recommendations:")
        print("  1. Focus on SPY which shows better results")
        print("  2. Wait for clearer market conditions")
        print("  3. Consider discretionary trading with tighter risk management")
    
    return consistent


if __name__ == "__main__":
    results = main()
