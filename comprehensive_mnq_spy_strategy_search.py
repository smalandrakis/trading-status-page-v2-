He#!/usr/bin/env python3
"""
Comprehensive indicator strategy search for MNQ and SPY.
Tests many different indicator combinations to find profitable strategies.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

QQQ_PARQUET = "data/QQQ_features.parquet"

def simulate_trades(df, long_signals, short_signals, symbol, sl_pct, tp_pct, 
                    cooldown_bars=10, max_hold=48, strategy_name=""):
    """Generic trade simulator."""
    trades = []
    last_trade_bar = -cooldown_bars
    
    for i in range(50, len(df)):
        if i - last_trade_bar < cooldown_bars:
            continue
        
        row = df.iloc[i]
        entry_price = row['Close']
        entry_time = df.index[i]
        
        direction = None
        if long_signals.iloc[i]:
            direction = 'LONG'
            target_price = entry_price * (1 + tp_pct / 100)
            stop_price = entry_price * (1 - sl_pct / 100)
        elif short_signals.iloc[i]:
            direction = 'SHORT'
            target_price = entry_price * (1 - tp_pct / 100)
            stop_price = entry_price * (1 + sl_pct / 100)
        else:
            continue
        
        exit_price = None
        exit_reason = None
        bars_held = 0
        
        for j in range(i + 1, min(i + max_hold + 1, len(df))):
            bars_held += 1
            future_row = df.iloc[j]
            
            if direction == 'LONG':
                if future_row['Low'] <= stop_price:
                    exit_price = stop_price
                    exit_reason = 'SL'
                    break
                if future_row['High'] >= target_price:
                    exit_price = target_price
                    exit_reason = 'TP'
                    break
            else:
                if future_row['High'] >= stop_price:
                    exit_price = stop_price
                    exit_reason = 'SL'
                    break
                if future_row['Low'] <= target_price:
                    exit_price = target_price
                    exit_reason = 'TP'
                    break
        
        if exit_price is None:
            exit_price = df.iloc[min(i + max_hold, len(df) - 1)]['Close']
            exit_reason = 'TO'
        
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
            'pnl_dollar': pnl_dollar,
            'exit_reason': exit_reason
        })
        
        last_trade_bar = i
    
    return trades


def analyze_results(trades, symbol, strategy_name, sl_pct, tp_pct):
    """Analyze trade results and return summary."""
    if not trades or len(trades) < 5:
        return None
    
    trades_df = pd.DataFrame(trades)
    
    total_trades = len(trades_df)
    wins = len(trades_df[trades_df['pnl_dollar'] > 0])
    win_rate = wins / total_trades * 100
    total_pnl = trades_df['pnl_dollar'].sum()
    
    days = (trades_df['entry_time'].max() - trades_df['entry_time'].min()).days or 1
    trades_per_day = total_trades / days
    pnl_per_week = total_pnl / days * 7
    
    # Drawdown
    cumulative_pnl = trades_df['pnl_dollar'].cumsum()
    peak = cumulative_pnl.expanding().max()
    max_drawdown = (cumulative_pnl - peak).min()
    
    return {
        'symbol': symbol,
        'strategy': strategy_name,
        'sl_tp': f"{sl_pct}/{tp_pct}",
        'trades': total_trades,
        'win_rate': win_rate,
        'pnl_week': pnl_per_week,
        'trades_day': trades_per_day,
        'max_dd': max_drawdown
    }


def test_all_strategies(df, symbol):
    """Test all indicator strategies."""
    results = []
    
    # Prepare indicators
    df = df.copy()
    
    # RSI
    rsi = df['momentum_rsi']
    
    # Stochastic
    stoch_k = df['momentum_stoch']
    stoch_d = df['momentum_stoch_signal']
    
    # MACD
    macd = df['trend_macd']
    macd_signal = df['trend_macd_signal']
    macd_hist = macd - macd_signal
    macd_hist_prev = macd_hist.shift(1)
    
    # Bollinger Bands
    bb_pct_b = df['volatility_bbp']
    bb_width = df['volatility_bbw']
    
    # ADX (trend strength)
    adx = df['trend_adx']
    adx_pos = df['trend_adx_pos']
    adx_neg = df['trend_adx_neg']
    
    # CCI
    cci = df['trend_cci']
    
    # Williams %R
    williams_r = df['momentum_wr']
    
    # ATR for volatility
    atr = df['volatility_atr']
    atr_pct = atr / df['Close'] * 100
    
    # EMA crossovers
    ema_12 = df['Close'].ewm(span=12).mean()
    ema_26 = df['Close'].ewm(span=26).mean()
    ema_50 = df['Close'].ewm(span=50).mean()
    
    # Price momentum
    roc_5 = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5) * 100
    roc_10 = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10) * 100
    
    # ============================================================
    # STRATEGY 1: Stochastic Oversold/Overbought
    # ============================================================
    for k_thresh in [20, 25, 30]:
        long_sig = (stoch_k < k_thresh) & (stoch_k > stoch_k.shift(1))  # Oversold + turning up
        short_sig = (stoch_k > (100-k_thresh)) & (stoch_k < stoch_k.shift(1))  # Overbought + turning down
        
        for sl, tp in [(0.3, 0.5), (0.4, 0.8), (0.5, 1.0)]:
            trades = simulate_trades(df, long_sig, short_sig, symbol, sl, tp, cooldown_bars=10)
            result = analyze_results(trades, symbol, f"Stoch<{k_thresh}+rising", sl, tp)
            if result:
                results.append(result)
    
    # ============================================================
    # STRATEGY 2: CCI Extreme
    # ============================================================
    for cci_thresh in [-100, -150, -200]:
        long_sig = cci < cci_thresh
        short_sig = cci > -cci_thresh
        
        for sl, tp in [(0.3, 0.5), (0.4, 0.8)]:
            trades = simulate_trades(df, long_sig, short_sig, symbol, sl, tp, cooldown_bars=10)
            result = analyze_results(trades, symbol, f"CCI<{cci_thresh}", sl, tp)
            if result:
                results.append(result)
    
    # ============================================================
    # STRATEGY 3: Williams %R
    # ============================================================
    for wr_thresh in [-80, -85, -90]:
        long_sig = williams_r < wr_thresh
        short_sig = williams_r > -wr_thresh - 100  # e.g., > -20 for -80 threshold
        
        for sl, tp in [(0.3, 0.5), (0.4, 0.8)]:
            trades = simulate_trades(df, long_sig, short_sig, symbol, sl, tp, cooldown_bars=10)
            result = analyze_results(trades, symbol, f"WR<{wr_thresh}", sl, tp)
            if result:
                results.append(result)
    
    # ============================================================
    # STRATEGY 4: ADX Trend + Direction
    # ============================================================
    for adx_thresh in [20, 25, 30]:
        # Strong trend + bullish
        long_sig = (adx > adx_thresh) & (adx_pos > adx_neg)
        short_sig = (adx > adx_thresh) & (adx_neg > adx_pos)
        
        for sl, tp in [(0.4, 0.8), (0.5, 1.0), (0.7, 1.4)]:
            trades = simulate_trades(df, long_sig, short_sig, symbol, sl, tp, cooldown_bars=15)
            result = analyze_results(trades, symbol, f"ADX>{adx_thresh}+DI", sl, tp)
            if result:
                results.append(result)
    
    # ============================================================
    # STRATEGY 5: EMA Crossover
    # ============================================================
    ema_cross_up = (ema_12 > ema_26) & (ema_12.shift(1) <= ema_26.shift(1))
    ema_cross_down = (ema_12 < ema_26) & (ema_12.shift(1) >= ema_26.shift(1))
    
    for sl, tp in [(0.4, 0.8), (0.5, 1.0), (0.7, 1.4)]:
        trades = simulate_trades(df, ema_cross_up, ema_cross_down, symbol, sl, tp, cooldown_bars=20)
        result = analyze_results(trades, symbol, "EMA12x26", sl, tp)
        if result:
            results.append(result)
    
    # ============================================================
    # STRATEGY 6: MACD Histogram Reversal
    # ============================================================
    macd_hist_reversal_up = (macd_hist < 0) & (macd_hist > macd_hist_prev) & (macd_hist_prev < macd_hist.shift(2))
    macd_hist_reversal_down = (macd_hist > 0) & (macd_hist < macd_hist_prev) & (macd_hist_prev > macd_hist.shift(2))
    
    for sl, tp in [(0.3, 0.5), (0.4, 0.8), (0.5, 1.0)]:
        trades = simulate_trades(df, macd_hist_reversal_up, macd_hist_reversal_down, symbol, sl, tp, cooldown_bars=10)
        result = analyze_results(trades, symbol, "MACD_Hist_Rev", sl, tp)
        if result:
            results.append(result)
    
    # ============================================================
    # STRATEGY 7: BB Squeeze + Breakout
    # ============================================================
    bb_squeeze = bb_width < bb_width.rolling(20).mean() * 0.8  # Squeeze
    breakout_up = (df['Close'] > df['volatility_bbh']) & bb_squeeze.shift(1)
    breakout_down = (df['Close'] < df['volatility_bbl']) & bb_squeeze.shift(1)
    
    for sl, tp in [(0.4, 0.8), (0.5, 1.0), (0.7, 1.4)]:
        trades = simulate_trades(df, breakout_up, breakout_down, symbol, sl, tp, cooldown_bars=15)
        result = analyze_results(trades, symbol, "BB_Squeeze_Break", sl, tp)
        if result:
            results.append(result)
    
    # ============================================================
    # STRATEGY 8: RSI + BB Combo (refined)
    # ============================================================
    for rsi_thresh in [30, 35, 40]:
        for bb_thresh in [0.2, 0.3, 0.4]:
            long_sig = (rsi < rsi_thresh) & (bb_pct_b < bb_thresh)
            short_sig = (rsi > (100-rsi_thresh)) & (bb_pct_b > (1-bb_thresh))
            
            for sl, tp in [(0.3, 0.5), (0.4, 0.8), (0.5, 1.0)]:
                trades = simulate_trades(df, long_sig, short_sig, symbol, sl, tp, cooldown_bars=10)
                result = analyze_results(trades, symbol, f"RSI<{rsi_thresh}+BB<{bb_thresh}", sl, tp)
                if result:
                    results.append(result)
    
    # ============================================================
    # STRATEGY 9: Stochastic + RSI Double Confirmation
    # ============================================================
    for stoch_thresh in [20, 25, 30]:
        for rsi_thresh in [30, 35, 40]:
            long_sig = (stoch_k < stoch_thresh) & (rsi < rsi_thresh)
            short_sig = (stoch_k > (100-stoch_thresh)) & (rsi > (100-rsi_thresh))
            
            for sl, tp in [(0.3, 0.5), (0.4, 0.8)]:
                trades = simulate_trades(df, long_sig, short_sig, symbol, sl, tp, cooldown_bars=10)
                result = analyze_results(trades, symbol, f"Stoch<{stoch_thresh}+RSI<{rsi_thresh}", sl, tp)
                if result:
                    results.append(result)
    
    # ============================================================
    # STRATEGY 10: Price above/below EMA with momentum
    # ============================================================
    price_above_ema50 = df['Close'] > ema_50
    roc_positive = roc_5 > 0.1
    roc_negative = roc_5 < -0.1
    
    long_sig = price_above_ema50 & roc_positive & (rsi < 60)
    short_sig = ~price_above_ema50 & roc_negative & (rsi > 40)
    
    for sl, tp in [(0.4, 0.8), (0.5, 1.0), (0.7, 1.4)]:
        trades = simulate_trades(df, long_sig, short_sig, symbol, sl, tp, cooldown_bars=15)
        result = analyze_results(trades, symbol, "EMA50+ROC+RSI", sl, tp)
        if result:
            results.append(result)
    
    # ============================================================
    # STRATEGY 11: Low volatility mean reversion
    # ============================================================
    low_vol = atr_pct < atr_pct.rolling(20).mean() * 0.8
    
    for bb_thresh in [0.2, 0.3]:
        long_sig = low_vol & (bb_pct_b < bb_thresh)
        short_sig = low_vol & (bb_pct_b > (1-bb_thresh))
        
        for sl, tp in [(0.25, 0.4), (0.3, 0.5)]:
            trades = simulate_trades(df, long_sig, short_sig, symbol, sl, tp, cooldown_bars=10)
            result = analyze_results(trades, symbol, f"LowVol+BB<{bb_thresh}", sl, tp)
            if result:
                results.append(result)
    
    # ============================================================
    # STRATEGY 12: Triple indicator confirmation
    # ============================================================
    long_sig = (rsi < 40) & (stoch_k < 30) & (bb_pct_b < 0.3)
    short_sig = (rsi > 60) & (stoch_k > 70) & (bb_pct_b > 0.7)
    
    for sl, tp in [(0.3, 0.5), (0.4, 0.8), (0.5, 1.0)]:
        trades = simulate_trades(df, long_sig, short_sig, symbol, sl, tp, cooldown_bars=10)
        result = analyze_results(trades, symbol, "RSI+Stoch+BB_Triple", sl, tp)
        if result:
            results.append(result)
    
    # ============================================================
    # STRATEGY 13: MACD Zero Cross
    # ============================================================
    macd_cross_up = (macd > 0) & (macd.shift(1) <= 0)
    macd_cross_down = (macd < 0) & (macd.shift(1) >= 0)
    
    for sl, tp in [(0.4, 0.8), (0.5, 1.0), (0.7, 1.4)]:
        trades = simulate_trades(df, macd_cross_up, macd_cross_down, symbol, sl, tp, cooldown_bars=20)
        result = analyze_results(trades, symbol, "MACD_Zero_Cross", sl, tp)
        if result:
            results.append(result)
    
    # ============================================================
    # STRATEGY 14: Momentum divergence (price down, RSI up)
    # ============================================================
    price_down = df['Close'] < df['Close'].shift(5)
    rsi_up = rsi > rsi.shift(5)
    bullish_div = price_down & rsi_up & (rsi < 40)
    
    price_up = df['Close'] > df['Close'].shift(5)
    rsi_down = rsi < rsi.shift(5)
    bearish_div = price_up & rsi_down & (rsi > 60)
    
    for sl, tp in [(0.3, 0.5), (0.4, 0.8), (0.5, 1.0)]:
        trades = simulate_trades(df, bullish_div, bearish_div, symbol, sl, tp, cooldown_bars=10)
        result = analyze_results(trades, symbol, "RSI_Divergence", sl, tp)
        if result:
            results.append(result)
    
    return results


def main():
    print("Loading QQQ parquet data...")
    df = pd.read_parquet(QQQ_PARQUET)
    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    
    cutoff = df.index[-1] - timedelta(days=30)
    df_30d = df[df.index >= cutoff]
    print(f"Using last 30 days: {len(df_30d)} bars")
    
    all_results = []
    
    for symbol in ['MNQ', 'SPY']:
        print(f"\n{'='*60}")
        print(f"Testing strategies for {symbol}...")
        print(f"{'='*60}")
        
        results = test_all_strategies(df_30d, symbol)
        all_results.extend(results)
        print(f"Found {len(results)} strategy variations")
    
    # Sort by P&L per week
    all_results_sorted = sorted(all_results, key=lambda x: x['pnl_week'], reverse=True)
    
    # Show top 30 results
    print("\n" + "="*100)
    print("TOP 30 STRATEGIES BY P&L/WEEK")
    print("="*100)
    print(f"\n{'Symbol':<6} {'Strategy':<25} {'SL/TP':<8} {'Trades':<8} {'WR%':<8} {'$/Week':<10} {'T/Day':<8} {'MaxDD':<10}")
    print("-" * 100)
    
    for r in all_results_sorted[:30]:
        print(f"{r['symbol']:<6} {r['strategy']:<25} {r['sl_tp']:<8} {r['trades']:<8} {r['win_rate']:.1f}%    ${r['pnl_week']:>6.0f}     {r['trades_day']:.1f}      ${r['max_dd']:.0f}")
    
    # Show only profitable strategies
    profitable = [r for r in all_results_sorted if r['pnl_week'] > 0]
    print(f"\n\n{'='*100}")
    print(f"PROFITABLE STRATEGIES: {len(profitable)} found")
    print("="*100)
    
    if profitable:
        print(f"\n{'Symbol':<6} {'Strategy':<25} {'SL/TP':<8} {'Trades':<8} {'WR%':<8} {'$/Week':<10} {'T/Day':<8} {'MaxDD':<10}")
        print("-" * 100)
        for r in profitable:
            print(f"{r['symbol']:<6} {r['strategy']:<25} {r['sl_tp']:<8} {r['trades']:<8} {r['win_rate']:.1f}%    ${r['pnl_week']:>6.0f}     {r['trades_day']:.1f}      ${r['max_dd']:.0f}")
    else:
        print("\nNo profitable strategies found with current parameters.")
        print("Consider: Different timeframes, longer holding periods, or ML-based approach.")


if __name__ == "__main__":
    main()
