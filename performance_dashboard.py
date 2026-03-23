"""
Model Performance Monitoring Dashboard

Provides real-time and historical performance analysis for trading bots.
Run this script to see current model performance metrics.

Usage:
    python performance_dashboard.py           # Full dashboard
    python performance_dashboard.py --quick   # Quick summary only
    python performance_dashboard.py --export  # Export data for retraining
"""

import argparse
from datetime import datetime, timedelta
from trade_database import TradeDatabase
import pandas as pd
import os


def print_header(title: str, width: int = 70):
    """Print a formatted header."""
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def print_subheader(title: str, width: int = 70):
    """Print a formatted subheader."""
    print("\n" + "-" * width)
    print(f" {title}")
    print("-" * width)


def format_currency(value: float) -> str:
    """Format a value as currency."""
    if value >= 0:
        return f"${value:,.2f}"
    return f"-${abs(value):,.2f}"


def format_pct(value: float) -> str:
    """Format a value as percentage."""
    return f"{value:.1f}%"


def get_performance_summary(db: TradeDatabase, days: int = 30) -> dict:
    """Get overall performance summary."""
    trades = db.get_trades()
    
    if trades.empty:
        return None
    
    # Filter to recent trades
    trades['entry_time'] = pd.to_datetime(trades['entry_time'])
    cutoff = datetime.now() - timedelta(days=days)
    recent = trades[trades['entry_time'] >= cutoff]
    
    if recent.empty:
        return None
    
    total_pnl = recent['pnl_dollar'].sum()
    total_trades = len(recent)
    winners = (recent['pnl_dollar'] > 0).sum()
    losers = (recent['pnl_dollar'] <= 0).sum()
    win_rate = winners / total_trades * 100 if total_trades > 0 else 0
    
    avg_winner = recent[recent['pnl_dollar'] > 0]['pnl_dollar'].mean() if winners > 0 else 0
    avg_loser = recent[recent['pnl_dollar'] < 0]['pnl_dollar'].mean() if losers > 0 else 0
    
    # Profit factor
    gross_profit = recent[recent['pnl_dollar'] > 0]['pnl_dollar'].sum()
    gross_loss = abs(recent[recent['pnl_dollar'] < 0]['pnl_dollar'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    return {
        'total_trades': total_trades,
        'winners': winners,
        'losers': losers,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_pnl': total_pnl / total_trades if total_trades > 0 else 0,
        'avg_winner': avg_winner,
        'avg_loser': avg_loser,
        'profit_factor': profit_factor,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss
    }


def print_quick_summary(db: TradeDatabase):
    """Print a quick performance summary."""
    print_header("TRADING BOT PERFORMANCE SUMMARY")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Overall summary
    summary = get_performance_summary(db, days=30)
    
    if summary is None:
        print("\nNo trades in database yet.")
        return
    
    print(f"\n📊 LAST 30 DAYS:")
    print(f"   Total Trades: {summary['total_trades']}")
    print(f"   Win Rate: {format_pct(summary['win_rate'])} ({summary['winners']}W / {summary['losers']}L)")
    print(f"   Total P&L: {format_currency(summary['total_pnl'])}")
    print(f"   Avg P&L/Trade: {format_currency(summary['avg_pnl'])}")
    print(f"   Profit Factor: {summary['profit_factor']:.2f}")
    print(f"   Avg Winner: {format_currency(summary['avg_winner'])}")
    print(f"   Avg Loser: {format_currency(summary['avg_loser'])}")
    
    # By bot type
    for bot_type in ['BTC', 'MNQ']:
        trades = db.get_trades(bot_type=bot_type)
        if not trades.empty:
            total = len(trades)
            winners = (trades['pnl_dollar'] > 0).sum()
            pnl = trades['pnl_dollar'].sum()
            print(f"\n   {bot_type}: {total} trades, {winners/total*100:.1f}% win rate, {format_currency(pnl)}")


def print_model_performance(db: TradeDatabase, days: int = 30):
    """Print detailed model performance."""
    print_header("MODEL PERFORMANCE BREAKDOWN")
    
    stats = db.get_model_stats(days=days)
    
    if stats.empty:
        print("\nNo trades in database yet.")
        return
    
    # Group by bot type
    for bot_type in stats['bot_type'].unique():
        print_subheader(f"{bot_type} Models")
        
        bot_stats = stats[stats['bot_type'] == bot_type].copy()
        bot_stats = bot_stats.sort_values('total_pnl', ascending=False)
        
        print(f"\n{'Model':<25} {'Trades':>7} {'Win%':>7} {'Total PnL':>12} {'Avg PnL':>10} {'TP':>4} {'SL':>4} {'TO':>4}")
        print("-" * 85)
        
        for _, row in bot_stats.iterrows():
            print(f"{row['model_id']:<25} {row['total_trades']:>7} {row['win_rate']:>6.1f}% "
                  f"{format_currency(row['total_pnl']):>12} {format_currency(row['avg_pnl']):>10} "
                  f"{row['take_profits']:>4} {row['stop_losses']:>4} {row['timeouts']:>4}")


def print_daily_performance(db: TradeDatabase, days: int = 14):
    """Print daily P&L breakdown."""
    print_header("DAILY PERFORMANCE")
    
    daily = db.get_daily_performance(days=days)
    
    if daily.empty:
        print("\nNo trades in database yet.")
        return
    
    # Pivot by bot type
    print(f"\n{'Date':<12} {'BTC Trades':>10} {'BTC P&L':>12} {'MNQ Trades':>10} {'MNQ P&L':>12} {'Total':>12}")
    print("-" * 75)
    
    dates = daily['date'].unique()
    for date in sorted(dates, reverse=True)[:days]:
        day_data = daily[daily['date'] == date]
        
        btc_row = day_data[day_data['bot_type'] == 'BTC']
        mnq_row = day_data[day_data['bot_type'] == 'MNQ']
        
        btc_trades = btc_row['trades'].values[0] if not btc_row.empty else 0
        btc_pnl = btc_row['daily_pnl'].values[0] if not btc_row.empty else 0
        mnq_trades = mnq_row['trades'].values[0] if not mnq_row.empty else 0
        mnq_pnl = mnq_row['daily_pnl'].values[0] if not mnq_row.empty else 0
        
        total_pnl = btc_pnl + mnq_pnl
        
        print(f"{date:<12} {btc_trades:>10} {format_currency(btc_pnl):>12} "
              f"{mnq_trades:>10} {format_currency(mnq_pnl):>12} {format_currency(total_pnl):>12}")


def print_exit_reason_analysis(db: TradeDatabase):
    """Analyze performance by exit reason."""
    print_header("EXIT REASON ANALYSIS")
    
    for bot_type in ['BTC', 'MNQ']:
        stats = db.get_performance_by_exit_reason(bot_type=bot_type)
        
        if stats.empty:
            continue
        
        print_subheader(f"{bot_type}")
        print(f"\n{'Exit Reason':<15} {'Trades':>8} {'Win%':>8} {'Total PnL':>12} {'Avg PnL':>10}")
        print("-" * 60)
        
        for _, row in stats.iterrows():
            print(f"{row['exit_reason']:<15} {row['trades']:>8} {row['win_rate']:>7.1f}% "
                  f"{format_currency(row['total_pnl']):>12} {format_currency(row['avg_pnl']):>10}")


def print_hourly_analysis(db: TradeDatabase):
    """Analyze performance by hour of day."""
    print_header("HOURLY PERFORMANCE ANALYSIS")
    
    for bot_type in ['BTC', 'MNQ']:
        stats = db.get_performance_by_hour(bot_type=bot_type)
        
        if stats.empty:
            continue
        
        print_subheader(f"{bot_type}")
        print(f"\n{'Hour':<6} {'Trades':>8} {'Win%':>8} {'Total PnL':>12} {'Avg PnL':>10}")
        print("-" * 50)
        
        for _, row in stats.iterrows():
            hour_str = f"{int(row['hour']):02d}:00"
            print(f"{hour_str:<6} {row['trades']:>8} {row['win_rate']:>7.1f}% "
                  f"{format_currency(row['total_pnl']):>12} {format_currency(row['avg_pnl']):>10}")


def print_model_alerts(db: TradeDatabase, days: int = 7):
    """Check for models that may need attention."""
    print_header("MODEL ALERTS & RECOMMENDATIONS")
    
    stats = db.get_model_stats(days=days)
    
    if stats.empty:
        print("\nNo trades in database yet.")
        return
    
    alerts = []
    
    for _, row in stats.iterrows():
        model_id = row['model_id']
        bot_type = row['bot_type']
        
        # Alert: Low win rate (below 50%)
        if row['win_rate'] < 50 and row['total_trades'] >= 5:
            alerts.append(f"⚠️  [{bot_type}] {model_id}: Low win rate ({row['win_rate']:.1f}%) over {row['total_trades']} trades")
        
        # Alert: Negative P&L with significant trades
        if row['total_pnl'] < -100 and row['total_trades'] >= 5:
            alerts.append(f"🔴 [{bot_type}] {model_id}: Negative P&L ({format_currency(row['total_pnl'])}) over {row['total_trades']} trades")
        
        # Alert: High stop loss rate
        if row['total_trades'] > 0:
            sl_rate = row['stop_losses'] / row['total_trades'] * 100
            if sl_rate > 30 and row['total_trades'] >= 5:
                alerts.append(f"⚠️  [{bot_type}] {model_id}: High stop loss rate ({sl_rate:.1f}%)")
        
        # Positive: Strong performer
        if row['win_rate'] >= 70 and row['total_pnl'] > 100 and row['total_trades'] >= 5:
            alerts.append(f"✅ [{bot_type}] {model_id}: Strong performer ({row['win_rate']:.1f}% win rate, {format_currency(row['total_pnl'])})")
    
    if alerts:
        print(f"\nLast {days} days analysis:\n")
        for alert in alerts:
            print(f"  {alert}")
    else:
        print("\n  No alerts - all models performing within normal parameters.")


def export_for_retraining(db: TradeDatabase):
    """Export trade data for model retraining."""
    print_header("EXPORTING DATA FOR RETRAINING")
    
    os.makedirs('exports', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for bot_type in ['BTC', 'MNQ']:
        output_path = f"exports/{bot_type.lower()}_trades_{timestamp}.csv"
        df = db.export_for_retraining(bot_type, output_path)
        
        if not df.empty:
            print(f"\n  {bot_type}: Exported {len(df)} trades to {output_path}")
        else:
            print(f"\n  {bot_type}: No trades to export")


def main():
    parser = argparse.ArgumentParser(description='Trading Bot Performance Dashboard')
    parser.add_argument('--quick', action='store_true', help='Show quick summary only')
    parser.add_argument('--export', action='store_true', help='Export data for retraining')
    parser.add_argument('--days', type=int, default=30, help='Number of days to analyze')
    args = parser.parse_args()
    
    db = TradeDatabase()
    
    if args.export:
        export_for_retraining(db)
        return
    
    if args.quick:
        print_quick_summary(db)
        return
    
    # Full dashboard
    print_quick_summary(db)
    print_model_performance(db, days=args.days)
    print_daily_performance(db, days=14)
    print_exit_reason_analysis(db)
    print_model_alerts(db, days=7)
    
    print("\n" + "=" * 70)
    print(" Dashboard generated. Run with --export to export data for retraining.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
