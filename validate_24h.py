#!/usr/bin/env python3
"""
Validate: replay last 24h actual trades against tick data.
Use the exact entry_time, entry_price, direction from trades.db,
then simulate the exit using tick data with the bot's parameters.
Compare simulated exit vs actual exit.
"""
import pandas as pd
import numpy as np
import sqlite3

# Bot parameters that were active for most of last 24h
SL_PCT = 0.30           # Was 0.30% for most of the day
TS_ACTIVATION = 0.30    # Was 0.30% for most of the day
TS_TRAIL = 0.10         # Was 0.10% for most of the day

# Load tick data
print("Loading tick data...")
ticks = pd.read_csv('logs/btc_price_ticks.csv', parse_dates=['timestamp'])
ticks = ticks.set_index('timestamp').sort_index()
print(f"  Ticks: {len(ticks)}, {ticks.index.min()} to {ticks.index.max()}")

# Load trades
print("\nLoading trades from DB...")
conn = sqlite3.connect('trades.db')
trades = pd.read_sql("""
    SELECT entry_time, exit_time, direction, entry_price, exit_price, 
           pnl_dollar, pnl_pct, exit_reason, model_id
    FROM trades 
    WHERE entry_time >= '2026-03-08T22:00'
    ORDER BY entry_time
""", conn)
conn.close()
print(f"  Trades: {len(trades)}")

# Simulate each trade
print(f"\n{'='*90}")
print(f"{'Entry Time':>22s} {'Dir':>5s} {'Entry$':>10s} {'Actual Exit':>12s} {'Sim Exit':>12s} "
      f"{'Act PnL':>8s} {'Sim PnL':>8s} {'Act Reason':>12s} {'Sim Reason':>12s} {'Match':>6s}")
print(f"{'='*90}")

matches = 0
total = 0
act_total_pnl = 0
sim_total_pnl = 0

for _, trade in trades.iterrows():
    entry_time_str = trade['entry_time'].replace('T', ' ')
    entry_price = trade['entry_price']
    direction = trade['direction']
    actual_pnl = trade['pnl_dollar']
    actual_reason = trade['exit_reason']
    actual_exit = trade['exit_price']
    
    # Get tick data starting from entry
    mask = ticks.index >= entry_time_str
    trade_ticks = ticks[mask]['price'].values
    
    if len(trade_ticks) < 10:
        print(f"  {entry_time_str} — not enough tick data, skipping")
        continue
    
    # Simulate with bot parameters
    if direction == 'LONG':
        sl_price = entry_price * (1 - SL_PCT / 100)
        ts_activate = entry_price * (1 + TS_ACTIVATION / 100)
    else:
        sl_price = entry_price * (1 + SL_PCT / 100)
        ts_activate = entry_price * (1 - TS_ACTIVATION / 100)
    
    peak = entry_price
    ts_active = False
    sim_exit_price = None
    sim_reason = 'TIMEOUT'
    
    # Simulate up to 6 hours of ticks (~10800 ticks at 2-sec)
    max_ticks = min(10800, len(trade_ticks))
    
    for i in range(1, max_ticks):
        p = trade_ticks[i]
        
        if direction == 'LONG':
            # Check SL
            if p <= sl_price:
                sim_exit_price = sl_price
                sim_reason = 'STOP_LOSS'
                break
            
            # Track peak and trailing stop
            if p > peak:
                peak = p
            
            if not ts_active and p >= ts_activate:
                ts_active = True
            
            if ts_active:
                trail_price = peak * (1 - TS_TRAIL / 100)
                if p <= trail_price:
                    sim_exit_price = trail_price
                    sim_reason = 'TRAILING_STOP'
                    break
        else:  # SHORT
            # Check SL
            if p >= sl_price:
                sim_exit_price = sl_price
                sim_reason = 'STOP_LOSS'
                break
            
            # Track peak (lowest for short) and trailing stop
            if p < peak:
                peak = p
            
            if not ts_active and p <= ts_activate:
                ts_active = True
            
            if ts_active:
                trail_price = peak * (1 + TS_TRAIL / 100)
                if p >= trail_price:
                    sim_exit_price = trail_price
                    sim_reason = 'TRAILING_STOP'
                    break
    
    if sim_exit_price is None:
        sim_exit_price = trade_ticks[min(max_ticks - 1, len(trade_ticks) - 1)]
    
    # Calculate simulated PnL
    if direction == 'LONG':
        sim_pnl_pct = (sim_exit_price / entry_price - 1) * 100
    else:
        sim_pnl_pct = (entry_price / sim_exit_price - 1) * 100
    sim_pnl_dollar = sim_pnl_pct / 100 * entry_price * 0.001  # scale to match bot sizing
    
    # Use actual pnl to infer contract size, then apply to sim
    if abs(actual_pnl) > 0 and abs(trade['pnl_pct']) > 0:
        contract_value = abs(actual_pnl / (trade['pnl_pct'] / 100))
        sim_pnl_dollar = sim_pnl_pct / 100 * contract_value
    
    reason_match = actual_reason == sim_reason
    pnl_close = abs(actual_pnl - sim_pnl_dollar) < 5.0  # within $5
    match = '✅' if (reason_match and pnl_close) else '❌'
    if reason_match and pnl_close:
        matches += 1
    total += 1
    act_total_pnl += actual_pnl
    sim_total_pnl += sim_pnl_dollar
    
    print(f"{entry_time_str:>22s} {direction:>5s} ${entry_price:>9.2f} "
          f"${actual_exit:>10.2f} ${sim_exit_price:>10.2f} "
          f"${actual_pnl:>+7.2f} ${sim_pnl_dollar:>+7.2f} "
          f"{actual_reason:>12s} {sim_reason:>12s} {match:>6s}")

print(f"{'='*90}")
print(f"\nMatches: {matches}/{total} ({matches/total*100:.0f}%)")
print(f"Actual total PnL:    ${act_total_pnl:+.2f}")
print(f"Simulated total PnL: ${sim_total_pnl:+.2f}")
print(f"Difference:          ${abs(act_total_pnl - sim_total_pnl):.2f}")
