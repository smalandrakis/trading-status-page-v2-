#!/usr/bin/env python3
"""Test the trend-following strategy on historical BTC data."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Load BTC data
df = pd.read_parquet('data/BTC_features.parquet')
df.index = pd.to_datetime(df.index)

# Test multiple configurations
configs = [
    {'sl': 0.50, 'tp': 1.50, 'trail': 0.30, 'roc': 1.5, 'name': 'Original (1.5% ROC)'},
    {'sl': 0.50, 'tp': 1.50, 'trail': 0.30, 'roc': 1.0, 'name': 'Lower ROC (1.0%)'},
    {'sl': 0.40, 'tp': 1.00, 'trail': 0.25, 'roc': 0.8, 'name': 'Tighter (0.8% ROC)'},
    {'sl': 0.60, 'tp': 2.00, 'trail': 0.40, 'roc': 2.0, 'name': 'Wider (2.0% ROC)'},
]
TIMEOUT_BARS = 96

print('='*70)
print('TREND-FOLLOWING STRATEGY BACKTEST')
print('='*70)

# Backtest last 7 days
last_7_days = df[df.index > datetime.now() - timedelta(days=7)]
print(f'\nBacktest period: {last_7_days.index[0]} to {last_7_days.index[-1]}')

def run_backtest(cfg):
    sl_pct = cfg['sl']
    tp_pct = cfg['tp']
    trail_pct = cfg['trail']
    min_roc = cfg['roc']
    
    def detect_signal(df, idx):
        if idx < 50:
            return None
        window = df.iloc[:idx+1]
        price = window['Close'].iloc[-1]
        sma_5 = window['Close'].rolling(5).mean().iloc[-1]
        sma_20 = window['Close'].rolling(20).mean().iloc[-1]
        roc = (price / window['Close'].iloc[-21] - 1) * 100 if len(window) > 21 else 0
        
        if price > sma_5 > sma_20 and roc > min_roc:
            return {'direction': 'LONG', 'roc': roc}
        if price < sma_5 < sma_20 and roc < -min_roc:
            return {'direction': 'SHORT', 'roc': roc}
        return None
    
    def simulate_trade(df, entry_idx, direction):
        entry = df['Close'].iloc[entry_idx]
        if direction == 'LONG':
            sl = entry * (1 - sl_pct / 100)
            tp = entry * (1 + tp_pct / 100)
        else:
            sl = entry * (1 + sl_pct / 100)
            tp = entry * (1 - tp_pct / 100)
        
        peak = entry
        trough = entry
        trailing = False
        
        for i in range(entry_idx + 1, min(entry_idx + TIMEOUT_BARS + 1, len(df))):
            high, low = df['High'].iloc[i], df['Low'].iloc[i]
            
            if direction == 'LONG':
                if high > peak:
                    peak = high
                    if (peak / entry - 1) * 100 > trail_pct:
                        trailing = True
                        new_sl = peak * (1 - trail_pct / 100)
                        if new_sl > sl:
                            sl = new_sl
                if low <= sl:
                    return (sl / entry - 1) * 100, 'TRAIL' if trailing else 'SL'
                if high >= tp:
                    return (tp / entry - 1) * 100, 'TP'
            else:
                if low < trough:
                    trough = low
                    if (entry / trough - 1) * 100 > trail_pct:
                        trailing = True
                        new_sl = trough * (1 + trail_pct / 100)
                        if new_sl < sl:
                            sl = new_sl
                if high >= sl:
                    return (entry / sl - 1) * 100, 'TRAIL' if trailing else 'SL'
                if low <= tp:
                    return (entry / tp - 1) * 100, 'TP'
        
        exit_price = df['Close'].iloc[min(entry_idx + TIMEOUT_BARS, len(df) - 1)]
        if direction == 'LONG':
            return (exit_price / entry - 1) * 100, 'TO'
        return (entry / exit_price - 1) * 100, 'TO'
    
    trades = []
    last_idx = -100
    
    for i in range(50, len(last_7_days)):
        if i - last_idx < 20:
            continue
        global_idx = df.index.get_loc(last_7_days.index[i])
        signal = detect_signal(df, global_idx)
        if signal:
            pnl, reason = simulate_trade(df, global_idx, signal['direction'])
            trades.append({'pnl': pnl, 'reason': reason, 'dir': signal['direction']})
            last_idx = i
    
    return trades

print(f"\n{'Config':<25} {'Trades':<8} {'WR%':<8} {'P&L%':<10} {'LONG':<12} {'SHORT':<12}")
print('-'*75)

for cfg in configs:
    trades = run_backtest(cfg)
    if trades:
        tdf = pd.DataFrame(trades)
        wins = (tdf['pnl'] > 0).sum()
        wr = wins / len(tdf) * 100
        total_pnl = tdf['pnl'].sum()
        
        long_trades = tdf[tdf['dir'] == 'LONG']
        short_trades = tdf[tdf['dir'] == 'SHORT']
        long_pnl = long_trades['pnl'].sum() if len(long_trades) > 0 else 0
        short_pnl = short_trades['pnl'].sum() if len(short_trades) > 0 else 0
        
        marker = '✓' if total_pnl > 0 else ''
        print(f"{cfg['name']:<25} {len(trades):<8} {wr:<7.0f}% {total_pnl:<9.2f}% {len(long_trades)}:{long_pnl:.1f}%    {len(short_trades)}:{short_pnl:.1f}%  {marker}")
    else:
        print(f"{cfg['name']:<25} 0        -        -")

print(f"\nNote: P&L is percentage move. For MBT ($1.25/point), 1% on $92k = $920/contract")
