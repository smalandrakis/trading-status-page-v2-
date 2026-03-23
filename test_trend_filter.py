#!/usr/bin/env python3
"""Test the trend filter with actual BTC trades."""

import pandas as pd
import sqlite3
from datetime import datetime, timedelta

# Load BTC data
df = pd.read_parquet('data/BTC_features.parquet')
df.index = pd.to_datetime(df.index)

# Load actual trades
conn = sqlite3.connect('trades.db')
trades_df = pd.read_sql_query('SELECT * FROM trades ORDER BY entry_time DESC', conn)
trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'], format='mixed')
conn.close()

# Filter SHORT trades from last 7 days
short_trades = trades_df[
    (trades_df['direction'] == 'SHORT') & 
    (trades_df['entry_time'] > datetime.now() - timedelta(days=7)) &
    (trades_df['exit_reason'].notna())
].copy()

print('='*70)
print('TREND FILTER ANALYSIS ON ACTUAL BTC SHORT TRADES')
print('='*70)
print(f'\nActual SHORT trades in last 7 days: {len(short_trades)}')
print(f'Stop losses: {len(short_trades[short_trades["exit_reason"] == "STOP_LOSS"])}')
print(f'Take profits: {len(short_trades[short_trades["exit_reason"] == "TAKE_PROFIT"])}')

def detect_trend_at_time(df, timestamp, sma_short, sma_long, roc_thresh):
    """Detect trend at a specific timestamp."""
    # Get data up to that timestamp
    mask = df.index <= timestamp
    window = df[mask]
    
    if len(window) < sma_long + 1:
        return {'allow_short': True}
    
    current_price = window['Close'].iloc[-1]
    sma_s = window['Close'].rolling(sma_short).mean().iloc[-1]
    sma_l = window['Close'].rolling(sma_long).mean().iloc[-1]
    roc = (current_price / window['Close'].iloc[-21] - 1) * 100 if len(window) > 21 else 0
    
    allow_short = True
    
    # Block SHORT in strong uptrend
    if current_price > sma_s > sma_l:
        allow_short = False
    elif current_price > sma_s and roc > roc_thresh:
        allow_short = False
    
    return {'allow_short': allow_short, 'roc': roc}

# Test different configurations
configs = [
    {'sma_short': 20, 'sma_long': 50, 'roc_thresh': 2.0, 'name': 'Original (20/50, ROC>2%)'},
    {'sma_short': 20, 'sma_long': 50, 'roc_thresh': 1.0, 'name': 'Lower ROC (20/50, ROC>1%)'},
    {'sma_short': 10, 'sma_long': 30, 'roc_thresh': 1.5, 'name': 'Faster (10/30, ROC>1.5%)'},
    {'sma_short': 5, 'sma_long': 20, 'roc_thresh': 1.0, 'name': 'Very Fast (5/20, ROC>1%)'},
]

print(f'\n{"Config":<35} {"Blocked":<10} {"SL Saved":<10} {"TP Missed":<10} {"Net P&L":<10}')
print('-'*75)

for cfg in configs:
    blocked_sl = 0
    blocked_tp = 0
    blocked_other = 0
    
    for _, trade in short_trades.iterrows():
        entry_time = trade['entry_time']
        
        # Make entry_time timezone-naive if needed
        if entry_time.tzinfo is not None:
            entry_time = entry_time.tz_localize(None)
        
        trend = detect_trend_at_time(df, entry_time, cfg['sma_short'], cfg['sma_long'], cfg['roc_thresh'])
        
        if not trend['allow_short']:
            if trade['exit_reason'] == 'STOP_LOSS':
                blocked_sl += 1
            elif trade['exit_reason'] == 'TAKE_PROFIT':
                blocked_tp += 1
            else:
                blocked_other += 1
    
    total_blocked = blocked_sl + blocked_tp + blocked_other
    # Avg SL loss ~$12, Avg TP win ~$40
    net_pnl = blocked_sl * 12 - blocked_tp * 40
    
    print(f'{cfg["name"]:<35} {total_blocked:<10} {blocked_sl:<10} {blocked_tp:<10} ${net_pnl:<10}')

print(f'''
*** INTERPRETATION ***
- "SL Saved" = Stop losses that would NOT have happened (saves money)
- "TP Missed" = Take profits that would NOT have happened (loses money)
- "Net P&L" = (SL Saved * $12) - (TP Missed * $40)
- Positive Net P&L = filter would have helped
''')
