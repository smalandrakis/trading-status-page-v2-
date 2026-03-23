"""
Structured Trade Logging System

Stores all trades in SQLite database for:
1. Future model retraining
2. Performance analysis
3. Model monitoring
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import pandas as pd
import os


DATABASE_PATH = "trades.db"


@dataclass
class TradeRecord:
    """Represents a completed trade for database storage."""
    trade_id: str
    bot_type: str  # 'BTC' or 'MNQ'
    symbol: str
    model_id: str
    direction: str  # 'LONG' or 'SHORT'
    entry_time: str
    entry_price: float
    exit_time: str
    exit_price: float
    pnl_pct: float
    pnl_dollar: float
    exit_reason: str  # 'TAKE_PROFIT', 'STOP_LOSS', 'TIMEOUT'
    bars_held: int
    horizon_bars: int
    model_horizon: str
    model_threshold: float
    # Market context at entry
    entry_rsi: Optional[float] = None
    entry_macd: Optional[float] = None
    entry_bb_position: Optional[float] = None
    entry_atr_pct: Optional[float] = None
    entry_prev_day_return: Optional[float] = None
    entry_volatility: Optional[float] = None
    entry_hour: Optional[int] = None
    entry_day_of_week: Optional[int] = None
    # Model confidence
    entry_probability: Optional[float] = None
    # Enriched metrics (Mar 17, 2026)
    entry_trend_1h: Optional[float] = None
    entry_macro_trend_24h: Optional[float] = None
    entry_prob_2h: Optional[float] = None
    entry_prob_4h: Optional[float] = None
    entry_prob_6h: Optional[float] = None
    entry_prob_2h_short: Optional[float] = None
    entry_prob_4h_short: Optional[float] = None
    max_favorable_excursion: Optional[float] = None  # filled at exit
    max_adverse_excursion: Optional[float] = None    # filled at exit
    # Additional metadata
    metadata: Optional[str] = None  # JSON string for extra data


class TradeDatabase:
    """SQLite database for trade storage and analysis."""
    
    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT UNIQUE,
                bot_type TEXT NOT NULL,
                symbol TEXT NOT NULL,
                model_id TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_time TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_time TEXT NOT NULL,
                exit_price REAL NOT NULL,
                pnl_pct REAL NOT NULL,
                pnl_dollar REAL NOT NULL,
                exit_reason TEXT NOT NULL,
                bars_held INTEGER NOT NULL,
                horizon_bars INTEGER NOT NULL,
                model_horizon TEXT NOT NULL,
                model_threshold REAL NOT NULL,
                entry_rsi REAL,
                entry_macd REAL,
                entry_bb_position REAL,
                entry_atr_pct REAL,
                entry_prev_day_return REAL,
                entry_volatility REAL,
                entry_hour INTEGER,
                entry_day_of_week INTEGER,
                entry_probability REAL,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create index for common queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_bot_type ON trades(bot_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_id ON trades(model_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entry_time ON trades(entry_time)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_exit_reason ON trades(exit_reason)')
        
        # Create model_performance table for tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                bot_type TEXT NOT NULL,
                model_id TEXT NOT NULL,
                trades_count INTEGER NOT NULL,
                win_count INTEGER NOT NULL,
                loss_count INTEGER NOT NULL,
                win_rate REAL NOT NULL,
                total_pnl REAL NOT NULL,
                avg_pnl REAL NOT NULL,
                avg_winner REAL,
                avg_loser REAL,
                profit_factor REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date, bot_type, model_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def insert_trade(self, trade: TradeRecord) -> bool:
        """Insert a trade record into the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO trades (
                    trade_id, bot_type, symbol, model_id, direction,
                    entry_time, entry_price, exit_time, exit_price,
                    pnl_pct, pnl_dollar, exit_reason, bars_held, horizon_bars,
                    model_horizon, model_threshold, entry_rsi, entry_macd,
                    entry_bb_position, entry_atr_pct, entry_prev_day_return,
                    entry_volatility, entry_hour, entry_day_of_week,
                    entry_probability, entry_trend_1h, entry_macro_trend_24h,
                    entry_prob_2h, entry_prob_4h, entry_prob_6h,
                    entry_prob_2h_short, entry_prob_4h_short,
                    max_favorable_excursion, max_adverse_excursion, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade.trade_id, trade.bot_type, trade.symbol, trade.model_id,
                trade.direction, trade.entry_time, trade.entry_price,
                trade.exit_time, trade.exit_price, trade.pnl_pct, trade.pnl_dollar,
                trade.exit_reason, trade.bars_held, trade.horizon_bars,
                trade.model_horizon, trade.model_threshold, trade.entry_rsi,
                trade.entry_macd, trade.entry_bb_position, trade.entry_atr_pct,
                trade.entry_prev_day_return, trade.entry_volatility,
                trade.entry_hour, trade.entry_day_of_week, trade.entry_probability,
                trade.entry_trend_1h, trade.entry_macro_trend_24h,
                trade.entry_prob_2h, trade.entry_prob_4h, trade.entry_prob_6h,
                trade.entry_prob_2h_short, trade.entry_prob_4h_short,
                trade.max_favorable_excursion, trade.max_adverse_excursion,
                trade.metadata
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error inserting trade: {e}")
            return False
    
    def get_trades(self, bot_type: str = None, model_id: str = None,
                   start_date: str = None, end_date: str = None,
                   limit: int = None) -> pd.DataFrame:
        """Query trades with optional filters."""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM trades WHERE 1=1"
        params = []
        
        if bot_type:
            query += " AND bot_type = ?"
            params.append(bot_type)
        
        if model_id:
            query += " AND model_id = ?"
            params.append(model_id)
        
        if start_date:
            query += " AND entry_time >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND entry_time <= ?"
            params.append(end_date)
        
        query += " ORDER BY entry_time DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df
    
    def get_model_stats(self, bot_type: str = None, days: int = 30) -> pd.DataFrame:
        """Get performance statistics by model for the last N days."""
        conn = sqlite3.connect(self.db_path)
        
        query = f'''
            SELECT 
                model_id,
                bot_type,
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl_dollar > 0 THEN 1 ELSE 0 END) as winners,
                SUM(CASE WHEN pnl_dollar <= 0 THEN 1 ELSE 0 END) as losers,
                ROUND(100.0 * SUM(CASE WHEN pnl_dollar > 0 THEN 1 ELSE 0 END) / COUNT(*), 1) as win_rate,
                ROUND(SUM(pnl_dollar), 2) as total_pnl,
                ROUND(AVG(pnl_dollar), 2) as avg_pnl,
                ROUND(AVG(CASE WHEN pnl_dollar > 0 THEN pnl_dollar END), 2) as avg_winner,
                ROUND(AVG(CASE WHEN pnl_dollar < 0 THEN pnl_dollar END), 2) as avg_loser,
                ROUND(AVG(pnl_pct), 2) as avg_pnl_pct,
                COUNT(CASE WHEN exit_reason = 'TAKE_PROFIT' THEN 1 END) as take_profits,
                COUNT(CASE WHEN exit_reason = 'STOP_LOSS' THEN 1 END) as stop_losses,
                COUNT(CASE WHEN exit_reason = 'TIMEOUT' THEN 1 END) as timeouts
            FROM trades
            WHERE entry_time >= datetime('now', '-{days} days')
        '''
        
        if bot_type:
            query += f" AND bot_type = '{bot_type}'"
        
        query += " GROUP BY model_id, bot_type ORDER BY total_pnl DESC"
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def get_daily_performance(self, bot_type: str = None, days: int = 30) -> pd.DataFrame:
        """Get daily P&L summary."""
        conn = sqlite3.connect(self.db_path)
        
        query = f'''
            SELECT 
                DATE(exit_time) as date,
                bot_type,
                COUNT(*) as trades,
                SUM(CASE WHEN pnl_dollar > 0 THEN 1 ELSE 0 END) as winners,
                ROUND(100.0 * SUM(CASE WHEN pnl_dollar > 0 THEN 1 ELSE 0 END) / COUNT(*), 1) as win_rate,
                ROUND(SUM(pnl_dollar), 2) as daily_pnl
            FROM trades
            WHERE exit_time >= datetime('now', '-{days} days')
        '''
        
        if bot_type:
            query += f" AND bot_type = '{bot_type}'"
        
        query += " GROUP BY DATE(exit_time), bot_type ORDER BY date DESC"
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def get_performance_by_hour(self, bot_type: str = None) -> pd.DataFrame:
        """Analyze performance by hour of day."""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                entry_hour as hour,
                COUNT(*) as trades,
                ROUND(100.0 * SUM(CASE WHEN pnl_dollar > 0 THEN 1 ELSE 0 END) / COUNT(*), 1) as win_rate,
                ROUND(SUM(pnl_dollar), 2) as total_pnl,
                ROUND(AVG(pnl_dollar), 2) as avg_pnl
            FROM trades
            WHERE entry_hour IS NOT NULL
        '''
        
        if bot_type:
            query += f" AND bot_type = '{bot_type}'"
        
        query += " GROUP BY entry_hour ORDER BY entry_hour"
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def get_performance_by_exit_reason(self, bot_type: str = None) -> pd.DataFrame:
        """Analyze performance by exit reason."""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                exit_reason,
                COUNT(*) as trades,
                ROUND(100.0 * SUM(CASE WHEN pnl_dollar > 0 THEN 1 ELSE 0 END) / COUNT(*), 1) as win_rate,
                ROUND(SUM(pnl_dollar), 2) as total_pnl,
                ROUND(AVG(pnl_dollar), 2) as avg_pnl
            FROM trades
        '''
        
        if bot_type:
            query += f" WHERE bot_type = '{bot_type}'"
        
        query += " GROUP BY exit_reason ORDER BY total_pnl DESC"
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def export_for_retraining(self, bot_type: str, output_path: str = None) -> pd.DataFrame:
        """Export trades with market context for model retraining."""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                entry_time, exit_time, model_id, direction,
                entry_price, exit_price, pnl_pct, exit_reason,
                entry_rsi, entry_macd, entry_bb_position, entry_atr_pct,
                entry_prev_day_return, entry_volatility, entry_hour,
                entry_day_of_week, entry_probability,
                CASE WHEN pnl_dollar > 0 THEN 1 ELSE 0 END as is_winner
            FROM trades
            WHERE bot_type = ?
            ORDER BY entry_time
        '''
        
        df = pd.read_sql_query(query, conn, params=[bot_type])
        conn.close()
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Exported {len(df)} trades to {output_path}")
        
        return df


def import_trades_from_log(log_path: str, bot_type: str, db: TradeDatabase) -> int:
    """Import historical trades from log file into database."""
    import re
    
    trades_imported = 0
    
    # Pattern to match EXIT log lines
    exit_pattern = re.compile(
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - INFO - EXIT \[([^\]]+)\] (LONG|SHORT): (TAKE PROFIT|STOP LOSS|TIMEOUT).*@ \$([0-9,.]+) \(([+-]?[0-9.]+)%\)'
    )
    
    # Pattern to match ENTRY log lines
    entry_pattern = re.compile(
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - INFO - ENTRY \[([^\]]+)\]: (LONG|SHORT) @ \$([0-9,.]+)'
    )
    
    entries = {}  # Track open entries
    
    with open(log_path, 'r') as f:
        for line in f:
            # Check for entry
            entry_match = entry_pattern.search(line)
            if entry_match:
                time_str, model_id, direction, price = entry_match.groups()
                entries[model_id] = {
                    'entry_time': time_str,
                    'direction': direction,
                    'entry_price': float(price.replace(',', ''))
                }
                continue
            
            # Check for exit
            exit_match = exit_pattern.search(line)
            if exit_match:
                exit_time, model_id, direction, exit_reason, exit_price, pnl_pct = exit_match.groups()
                
                if model_id in entries:
                    entry = entries[model_id]
                    
                    # Parse model_id to get horizon and threshold
                    parts = model_id.split('_')
                    horizon = parts[0]
                    threshold = float(parts[1].replace('pct', ''))
                    
                    # Calculate dollar P&L (approximate)
                    exit_price_float = float(exit_price.replace(',', ''))
                    pnl_pct_float = float(pnl_pct)
                    
                    if bot_type == 'BTC':
                        pnl_dollar = (pnl_pct_float / 100) * exit_price_float * 0.1 - 2.02
                    else:
                        pnl_dollar = (pnl_pct_float / 100) * exit_price_float * 0.2 - 2.02
                    
                    trade = TradeRecord(
                        trade_id=f"{bot_type}_{model_id}_{entry['entry_time'].replace(' ', '_').replace(':', '')}",
                        bot_type=bot_type,
                        symbol='MBT' if bot_type == 'BTC' else 'MNQ',
                        model_id=model_id,
                        direction=direction,
                        entry_time=entry['entry_time'],
                        entry_price=entry['entry_price'],
                        exit_time=exit_time,
                        exit_price=exit_price_float,
                        pnl_pct=pnl_pct_float,
                        pnl_dollar=pnl_dollar,
                        exit_reason=exit_reason.replace(' ', '_').upper(),
                        bars_held=0,  # Not available from log
                        horizon_bars=0,  # Would need to calculate
                        model_horizon=horizon,
                        model_threshold=threshold,
                        entry_hour=int(entry['entry_time'].split(' ')[1].split(':')[0])
                    )
                    
                    if db.insert_trade(trade):
                        trades_imported += 1
                    
                    del entries[model_id]
    
    return trades_imported


# Convenience functions for bot integration
def log_trade_to_db(bot_type: str, model_id: str, direction: str,
                    entry_time: str, entry_price: float,
                    exit_time: str, exit_price: float,
                    pnl_pct: float, pnl_dollar: float,
                    exit_reason: str, bars_held: int, horizon_bars: int,
                    model_horizon: str, model_threshold: float,
                    market_context: Dict[str, Any] = None,
                    entry_probability: float = None) -> bool:
    """
    Convenience function to log a trade from the bot.
    
    Usage in bot:
        from trade_database import log_trade_to_db
        
        log_trade_to_db(
            bot_type='BTC',
            model_id='8h_0.75pct_SHORT',
            direction='SHORT',
            entry_time='2025-12-16 14:30:00',
            entry_price=86886.69,
            exit_time='2025-12-16 22:40:00',
            exit_price=87849.91,
            pnl_pct=-1.10,
            pnl_dollar=-96.0,
            exit_reason='TIMEOUT',
            bars_held=79,
            horizon_bars=96,
            model_horizon='8h',
            model_threshold=0.75,
            market_context={'rsi': 45.2, 'prev_day_return': -0.02},
            entry_probability=0.62
        )
    """
    db = TradeDatabase()
    
    # DEDUPLICATION: Check if a trade with same bot_type, direction, entry_price,
    # and similar exit_time already exists (prevents phantom legacy duplicates)
    try:
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT COUNT(*) FROM trades
            WHERE bot_type = ? AND direction = ? AND entry_time = ?
              AND ABS(entry_price - ?) < 1.0
              AND exit_reason = ?
              AND ABS(julianday(exit_time) - julianday(?)) < 0.001
        ''', (bot_type, direction, entry_time, entry_price, exit_reason, exit_time))
        count = cursor.fetchone()[0]
        conn.close()
        if count > 0:
            print(f"DEDUP: Skipping duplicate trade {bot_type} {direction} entry={entry_time} exit={exit_time}")
            return False
    except Exception as e:
        print(f"Dedup check failed (proceeding): {e}")
    
    trade = TradeRecord(
        trade_id=f"{bot_type}_{model_id}_{entry_time.replace(' ', '_').replace(':', '').replace('-', '')}",
        bot_type=bot_type,
        symbol='MBT' if bot_type == 'BTC' else 'MNQ',
        model_id=model_id,
        direction=direction,
        entry_time=entry_time,
        entry_price=entry_price,
        exit_time=exit_time,
        exit_price=exit_price,
        pnl_pct=pnl_pct,
        pnl_dollar=pnl_dollar,
        exit_reason=exit_reason,
        bars_held=bars_held,
        horizon_bars=horizon_bars,
        model_horizon=model_horizon,
        model_threshold=model_threshold,
        entry_rsi=market_context.get('rsi') if market_context else None,
        entry_macd=market_context.get('macd') if market_context else None,
        entry_bb_position=market_context.get('bb_position') if market_context else None,
        entry_atr_pct=market_context.get('atr_pct') if market_context else None,
        entry_prev_day_return=market_context.get('prev_day_return') if market_context else None,
        entry_volatility=market_context.get('volatility') if market_context else None,
        entry_hour=market_context.get('hour') if market_context else None,
        entry_day_of_week=market_context.get('day_of_week') if market_context else None,
        entry_probability=entry_probability,
        entry_trend_1h=market_context.get('trend_1h') if market_context else None,
        entry_macro_trend_24h=market_context.get('macro_trend_24h') if market_context else None,
        entry_prob_2h=market_context.get('prob_2h') if market_context else None,
        entry_prob_4h=market_context.get('prob_4h') if market_context else None,
        entry_prob_6h=market_context.get('prob_6h') if market_context else None,
        entry_prob_2h_short=market_context.get('prob_2h_short') if market_context else None,
        entry_prob_4h_short=market_context.get('prob_4h_short') if market_context else None,
        max_favorable_excursion=market_context.get('max_favorable_excursion') if market_context else None,
        max_adverse_excursion=market_context.get('max_adverse_excursion') if market_context else None,
        metadata=json.dumps(market_context) if market_context else None
    )
    
    return db.insert_trade(trade)


if __name__ == "__main__":
    # Test the database
    db = TradeDatabase()
    print(f"Database initialized at: {DATABASE_PATH}")
    
    # Import existing trades from logs
    if os.path.exists('logs/btc_bot.log'):
        print("\nImporting BTC trades from log...")
        count = import_trades_from_log('logs/btc_bot.log', 'BTC', db)
        print(f"Imported {count} BTC trades")
    
    if os.path.exists('logs/mnq_bot.log'):
        print("\nImporting MNQ trades from log...")
        count = import_trades_from_log('logs/mnq_bot.log', 'MNQ', db)
        print(f"Imported {count} MNQ trades")
    
    # Show summary
    print("\n" + "="*60)
    print("MODEL PERFORMANCE SUMMARY (Last 30 days)")
    print("="*60)
    stats = db.get_model_stats()
    if not stats.empty:
        print(stats.to_string(index=False))
    else:
        print("No trades in database yet")
    
    print("\n" + "="*60)
    print("DAILY PERFORMANCE")
    print("="*60)
    daily = db.get_daily_performance()
    if not daily.empty:
        print(daily.to_string(index=False))
