#!/usr/bin/env python3
"""
Analyze BTC bot performance - models, win rates, P&L, direction analysis.
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta

DB_PATH = "trades.db"

def analyze_btc_trades():
    conn = sqlite3.connect(DB_PATH)
    
    # Get all BTC trades from last 48 hours
    query = """
    SELECT model_id, direction, entry_price, exit_price, pnl_dollar, pnl_pct, 
           exit_reason, entry_time, exit_time, bars_held, entry_probability
    FROM trades 
    WHERE bot_type='BTC' 
    AND exit_time > datetime('now', '-48 hours')
    ORDER BY exit_time DESC
    """
    
    df = pd.read_sql_query(query, conn)
    
    if df.empty:
        print("No BTC trades in last 48 hours")
        return
    
    print("="*80)
    print("BTC BOT PERFORMANCE ANALYSIS - Last 48 Hours")
    print("="*80)
    
    # Overall stats
    total_trades = len(df)
    wins = len(df[df['pnl_dollar'] > 0])
    losses = len(df[df['pnl_dollar'] <= 0])
    win_rate = wins / total_trades * 100
    total_pnl = df['pnl_dollar'].sum()
    
    print(f"\n📊 OVERALL STATS (48h):")
    print(f"  Total trades: {total_trades}")
    print(f"  Wins: {wins}, Losses: {losses}")
    print(f"  Win rate: {win_rate:.1f}%")
    print(f"  Total P&L: ${total_pnl:.2f}")
    
    # Last 24 hours
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    cutoff_24h = datetime.now() - timedelta(hours=24)
    df_24h = df[df['exit_time'] > cutoff_24h]
    
    if len(df_24h) > 0:
        wins_24h = len(df_24h[df_24h['pnl_dollar'] > 0])
        total_24h = len(df_24h)
        pnl_24h = df_24h['pnl_dollar'].sum()
        print(f"\n📊 LAST 24 HOURS:")
        print(f"  Trades: {total_24h}")
        print(f"  Win rate: {wins_24h/total_24h*100:.1f}%")
        print(f"  P&L: ${pnl_24h:.2f}")
    
    # By model
    print(f"\n📈 PERFORMANCE BY MODEL:")
    print("-"*70)
    
    model_stats = df.groupby('model_id').agg({
        'pnl_dollar': ['count', 'sum', 'mean'],
        'pnl_pct': 'mean'
    }).round(2)
    model_stats.columns = ['trades', 'total_pnl', 'avg_pnl', 'avg_pnl_pct']
    
    for model_id in df['model_id'].unique():
        model_df = df[df['model_id'] == model_id]
        trades = len(model_df)
        wins = len(model_df[model_df['pnl_dollar'] > 0])
        wr = wins / trades * 100
        total = model_df['pnl_dollar'].sum()
        avg = model_df['pnl_dollar'].mean()
        
        print(f"\n  {model_id}:")
        print(f"    Trades: {trades}, Wins: {wins}, WR: {wr:.1f}%")
        print(f"    Total P&L: ${total:.2f}, Avg: ${avg:.2f}")
        
        # Exit reasons
        exit_reasons = model_df['exit_reason'].value_counts()
        print(f"    Exit reasons: {dict(exit_reasons)}")
    
    # By exit reason
    print(f"\n📋 BY EXIT REASON:")
    print("-"*70)
    for reason in df['exit_reason'].unique():
        reason_df = df[df['exit_reason'] == reason]
        trades = len(reason_df)
        wins = len(reason_df[reason_df['pnl_dollar'] > 0])
        wr = wins / trades * 100 if trades > 0 else 0
        total = reason_df['pnl_dollar'].sum()
        print(f"  {reason}: {trades} trades, {wr:.1f}% WR, ${total:.2f}")
    
    # Best and worst trades
    print(f"\n🏆 BEST TRADES:")
    print("-"*70)
    best = df.nlargest(5, 'pnl_dollar')[['model_id', 'direction', 'pnl_dollar', 'pnl_pct', 'exit_reason', 'bars_held']]
    for _, row in best.iterrows():
        print(f"  {row['model_id']}: ${row['pnl_dollar']:.2f} ({row['pnl_pct']:.2f}%) - {row['exit_reason']} after {row['bars_held']} bars")
    
    print(f"\n💀 WORST TRADES:")
    print("-"*70)
    worst = df.nsmallest(5, 'pnl_dollar')[['model_id', 'direction', 'pnl_dollar', 'pnl_pct', 'exit_reason', 'bars_held']]
    for _, row in worst.iterrows():
        print(f"  {row['model_id']}: ${row['pnl_dollar']:.2f} ({row['pnl_pct']:.2f}%) - {row['exit_reason']} after {row['bars_held']} bars")
    
    # Direction analysis - did price move in right direction initially?
    print(f"\n🎯 DIRECTION ANALYSIS:")
    print("-"*70)
    print("  Checking if trades moved in the right direction initially...")
    
    # For LONG: exit_price > entry_price at some point means it moved right
    # For SHORT: exit_price < entry_price at some point means it moved right
    # We can infer from pnl_pct - if positive, it moved right at exit
    # For stop losses, we know it moved wrong
    
    for model_id in df['model_id'].unique():
        model_df = df[df['model_id'] == model_id]
        
        # Trades that hit TP (definitely moved right)
        tp_trades = len(model_df[model_df['exit_reason'] == 'TAKE_PROFIT'])
        # Trades that hit SL (moved wrong)
        sl_trades = len(model_df[model_df['exit_reason'] == 'STOP_LOSS'])
        # Trailing stop (moved right then reversed)
        ts_trades = len(model_df[model_df['exit_reason'] == 'TRAILING_STOP'])
        # Timeout with profit (moved right)
        timeout_profit = len(model_df[(model_df['exit_reason'] == 'TIMEOUT') & (model_df['pnl_dollar'] > 0)])
        timeout_loss = len(model_df[(model_df['exit_reason'] == 'TIMEOUT') & (model_df['pnl_dollar'] <= 0)])
        
        moved_right = tp_trades + ts_trades + timeout_profit
        moved_wrong = sl_trades + timeout_loss
        total = len(model_df)
        
        print(f"\n  {model_id}:")
        print(f"    Moved RIGHT initially: {moved_right} ({moved_right/total*100:.1f}%)")
        print(f"    Moved WRONG initially: {moved_wrong} ({moved_wrong/total*100:.1f}%)")
        print(f"    (TP:{tp_trades}, TS:{ts_trades}, TO+:{timeout_profit}, SL:{sl_trades}, TO-:{timeout_loss})")
    
    # Recommendation
    print(f"\n💡 RECOMMENDATIONS:")
    print("-"*70)
    
    # Find profitable models
    profitable_models = []
    unprofitable_models = []
    
    for model_id in df['model_id'].unique():
        model_df = df[df['model_id'] == model_id]
        total_pnl = model_df['pnl_dollar'].sum()
        trades = len(model_df)
        wins = len(model_df[model_df['pnl_dollar'] > 0])
        wr = wins / trades * 100
        
        if total_pnl > 0:
            profitable_models.append((model_id, total_pnl, wr, trades))
        else:
            unprofitable_models.append((model_id, total_pnl, wr, trades))
    
    if profitable_models:
        print("\n  ✅ KEEP (Profitable):")
        for m, pnl, wr, t in sorted(profitable_models, key=lambda x: x[1], reverse=True):
            print(f"    {m}: ${pnl:.2f} ({wr:.1f}% WR, {t} trades)")
    
    if unprofitable_models:
        print("\n  ❌ CONSIDER DISABLING (Unprofitable):")
        for m, pnl, wr, t in sorted(unprofitable_models, key=lambda x: x[1]):
            print(f"    {m}: ${pnl:.2f} ({wr:.1f}% WR, {t} trades)")
    
    conn.close()
    return df


if __name__ == "__main__":
    analyze_btc_trades()
