#!/usr/bin/env python3
"""
BTC Tick Bot REVERSE — Independent Contrarian Mode (v4)
=======================================================
Reads ENTRY events from the virtual tick bot's trade_events table.
When the tick bot signals LONG, this bot goes SHORT via IB MBT.
Manages its own SL/TP using IB MBT real-time price (no Binance).

TP = 0.50%  (take profit)
SL = 0.25%  (stop loss)

Only reverses LONG signals (goes SHORT). Ignores SHORT signals.
Paper trading on CME MBT futures.
"""

import os
import sys
import time
import sqlite3
import logging
import argparse
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional

from ib_insync import IB, Future, MarketOrder, util
util.logToConsole(level=logging.WARNING)  # suppress ib_insync console spam

# =============================================================================
# CONFIGURATION
# =============================================================================
# Source DB (virtual tick bot)
SOURCE_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tick_trades.db')

# Shared price file written by original bot (for status logging only)
PRICE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.btc_price')

# Reverse bot DB
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tick_trades_reverse.db')

# IB connection
IB_HOST = '127.0.0.1'
IB_PORT = 4002
IB_CLIENT_ID = 402

# Contract
MBT_EXPIRY = '20260424'
BTC_CONTRACT_VALUE = 0.1  # MBT = 0.1 BTC
POSITION_SIZE = 1

# TP / SL (independent management)
TP_PCT = 0.50   # take profit %
SL_PCT = 0.25   # stop loss %

# Which signals to reverse
REVERSE_LONG = True    # tick bot LONG → we go SHORT
REVERSE_SHORT = False  # tick bot SHORT → we go LONG (disabled)

# Max positions
MAX_POSITIONS = 1

# Cooldowns
SL_COOLDOWN_SEC = 300  # 5 min after SL before re-entering

# Polling
POLL_INTERVAL = 2  # seconds between DB polls
PRICE_CHECK_INTERVAL = 1  # seconds between SL/TP checks

# Signal log
SIGNAL_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'signal_logs')

# Logging
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SIGNAL_LOG_DIR, exist_ok=True)

logging.root.handlers.clear()
logger = logging.getLogger('btc_tick_reverse')
logger.setLevel(logging.INFO)
logger.propagate = False
fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh = logging.FileHandler(os.path.join(LOG_DIR, 'btc_tick_bot_reverse.log'))
fh.setFormatter(fmt)
logger.addHandler(fh)


# =============================================================================
# DATA CLASSES
# =============================================================================
@dataclass
class Position:
    model_id: str           # REV_<original_model_id>
    orig_model_id: str      # original model_id
    direction: str          # reversed direction (SHORT for reversed LONGs)
    entry_price: float      # IB fill price
    entry_time: datetime
    size: int
    tp_price: float = 0.0   # take profit trigger
    sl_price: float = 0.0   # stop loss trigger
    order_id: Optional[int] = None
    entry_probability: float = 0.0
    short_agreement: int = 0


# =============================================================================
# TRADE DATABASE
# =============================================================================
def init_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT, direction TEXT,
            entry_time TEXT, exit_time TEXT,
            entry_price REAL, exit_price REAL,
            pnl_pct REAL, pnl_dollar REAL,
            exit_reason TEXT, bars_held INTEGER,
            entry_probability REAL,
            short_agreement INTEGER
        )
    """)
    conn.commit()
    conn.close()


def log_trade(db_path, trade_dict):
    conn = sqlite3.connect(db_path)
    cols = ', '.join(trade_dict.keys())
    placeholders = ', '.join(['?'] * len(trade_dict))
    conn.execute("INSERT INTO trades (%s) VALUES (%s)" % (cols, placeholders),
                 list(trade_dict.values()))
    conn.commit()
    conn.close()


# =============================================================================
# REVERSE BOT
# =============================================================================
class BTCTickBotReverse:
    def __init__(self, paper_trading=True):
        self.paper = paper_trading
        self.ib = IB()
        self.contract = None
        self.ib_mbt_ticker = None  # IB MBT real-time ticker for SL/TP
        self.positions = []        # list of Position

        # Track which trade_events we've already processed
        self.last_seen_event_id = 0

        # Cooldown tracking
        self.last_sl_exit = None

        # Signal logging
        self.signal_csv_date = None
        self.signal_csv_path = None
        self.last_signal_log_time = None

        init_db(DB_PATH)

    def connect_ib(self):
        """Connect to IB Gateway."""
        try:
            self.ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID, timeout=20)
            logging.root.handlers.clear()
            logger.info("Connected to IB Gateway (port=%d, clientId=%d)" % (IB_PORT, IB_CLIENT_ID))

            # Type 4: real-time streaming on paper accounts
            self.ib.reqMarketDataType(4)

            self.contract = Future(
                symbol='MBT',
                exchange='CME',
                currency='USD',
                lastTradeDateOrContractMonth=MBT_EXPIRY
            )
            self.ib.qualifyContracts(self.contract)
            logger.info("Contract: %s" % self.contract)

            # Subscribe to IB MBT real-time ticker for SL/TP monitoring
            try:
                self.ib_mbt_ticker = self.ib.reqMktData(self.contract, '', False, False)
                self.ib.sleep(3)
                mp = self.ib_mbt_ticker.marketPrice()
                if mp and mp > 10000:
                    logger.info("IB MBT ticker active: $%.2f" % mp)
                else:
                    logger.warning("IB MBT ticker no price yet (mp=%s)" % mp)
            except Exception as e:
                logger.warning("IB MBT ticker failed: %s" % e)

            return True
        except Exception as e:
            logger.error("IB connection failed: %s" % e)
            return False

    def reconnect_ib(self):
        """Reconnect to IB."""
        try:
            self.ib.disconnect()
        except:
            pass
        time.sleep(5)
        return self.connect_ib()

    def _get_mbt_price(self):
        """Get IB MBT futures price for SL/TP monitoring."""
        if self.ib_mbt_ticker is not None:
            for attr in ['last', 'close']:
                v = getattr(self.ib_mbt_ticker, attr, None)
                if v and v > 10000:
                    return v
            mp = self.ib_mbt_ticker.marketPrice()
            if mp and mp > 10000:
                return mp
        return None

    def get_current_max_event_id(self):
        """Get the highest trade_events ID from the source DB."""
        try:
            conn = sqlite3.connect(SOURCE_DB)
            c = conn.cursor()
            c.execute("SELECT MAX(id) FROM trade_events")
            row = c.fetchone()
            conn.close()
            return row[0] or 0
        except Exception as e:
            logger.debug("Error reading trade_events: %s" % e)
            return 0

    def poll_entry_events(self):
        """Poll source DB trade_events for new ENTRY events only."""
        try:
            conn = sqlite3.connect(SOURCE_DB)
            c = conn.cursor()
            c.execute("""
                SELECT id, event_type, model_id, direction, price, timestamp,
                       exit_reason, entry_probability, short_agreement
                FROM trade_events
                WHERE id > ? AND event_type = 'ENTRY'
                ORDER BY id ASC
            """, (self.last_seen_event_id,))
            rows = c.fetchall()
            # Also advance past any EXIT events we skip
            c.execute("SELECT MAX(id) FROM trade_events WHERE id > ?", (self.last_seen_event_id,))
            max_row = c.fetchone()
            conn.close()
            new_max = max_row[0] if max_row and max_row[0] else self.last_seen_event_id
            return rows, new_max
        except Exception as e:
            logger.error("Error polling trade_events: %s" % e)
            return [], self.last_seen_event_id

    def handle_entry(self, model_id, orig_direction, price, prob, agreement):
        """Handle an ENTRY event: go SHORT when tick bot goes LONG."""
        # Only reverse LONG signals
        if orig_direction == 'LONG' and not REVERSE_LONG:
            return
        if orig_direction == 'SHORT' and not REVERSE_SHORT:
            logger.debug("Skipping SHORT signal (REVERSE_SHORT=False)")
            return

        rev_direction = 'SHORT' if orig_direction == 'LONG' else 'LONG'
        rev_model_id = 'REV_' + model_id

        # Max positions check
        if len(self.positions) >= MAX_POSITIONS:
            logger.info("Max positions (%d) — skipping %s" % (MAX_POSITIONS, rev_model_id))
            return

        # SL cooldown
        if self.last_sl_exit:
            elapsed = (datetime.now() - self.last_sl_exit).total_seconds()
            if elapsed < SL_COOLDOWN_SEC:
                logger.info("SL cooldown: %.0fs remaining — skipping" % (SL_COOLDOWN_SEC - elapsed))
                return

        # Check if we already have a position for this model
        for p in self.positions:
            if p.orig_model_id == model_id:
                logger.warning("Already have position for %s — skipping" % model_id)
                return

        # Place order via IB
        order_id = None
        fill_price = price  # fallback
        if self.ib.isConnected():
            try:
                action = 'BUY' if rev_direction == 'LONG' else 'SELL'
                order = MarketOrder(action, POSITION_SIZE)
                trade = self.ib.placeOrder(self.contract, order)
                self.ib.sleep(2)
                if trade.orderStatus.status == 'Filled':
                    fill_price = trade.orderStatus.avgFillPrice
                    order_id = trade.order.orderId
                    logger.info("Order filled: %s %d @ $%.2f" % (action, POSITION_SIZE, fill_price))
                else:
                    logger.warning("Order not filled (%s) — aborting entry" % trade.orderStatus.status)
                    return
            except Exception as e:
                logger.error("Order error: %s" % e)
                return
        else:
            logger.error("IB not connected — cannot enter")
            return

        # Mar 31: BINANCE-ONLY — use Binance price for entry/TP/SL (consistent with monitoring)
        binance_price = self._get_binance_price_from_file()
        entry_ref = binance_price if binance_price else fill_price
        if binance_price and abs(fill_price - binance_price) / binance_price > 0.005:
            logger.info("IB fill $%.0f vs Binance $%.0f (basis %+.2f%%) — using Binance for tracking" % (
                fill_price, binance_price, (fill_price / binance_price - 1) * 100))

        if rev_direction == 'SHORT':
            tp_price = entry_ref * (1 - TP_PCT / 100)   # price drops = profit
            sl_price = entry_ref * (1 + SL_PCT / 100)   # price rises = loss
        else:  # LONG
            tp_price = entry_ref * (1 + TP_PCT / 100)
            sl_price = entry_ref * (1 - SL_PCT / 100)

        pos = Position(
            model_id=rev_model_id,
            orig_model_id=model_id,
            direction=rev_direction,
            entry_price=entry_ref,
            entry_time=datetime.now(),
            size=POSITION_SIZE,
            tp_price=tp_price,
            sl_price=sl_price,
            order_id=order_id,
            entry_probability=prob,
            short_agreement=agreement,
        )
        self.positions.append(pos)

        logger.info("ENTRY %s [%s] @ $%.2f (source: %s %s, prob=%.0f%%) | TP=$%.2f SL=$%.2f" % (
            rev_direction, rev_model_id, fill_price,
            orig_direction, model_id, prob * 100,
            tp_price, sl_price))

    def _get_binance_price_from_file(self):
        """Read Binance price from .btc_price file (written by tick bot every ~2s)."""
        try:
            if os.path.exists(PRICE_FILE):
                with open(PRICE_FILE, 'r') as f:
                    price = float(f.read().strip())
                if price > 10000:
                    return price
        except Exception:
            pass
        return None

    def check_exits(self):
        """Check TP/SL using Binance price (Mar 31: consistent Binance-only architecture)."""
        # Mar 31: Use Binance price from .btc_price file (written by tick bot).
        # MBT ticker freezes during CME maintenance breaks → stale price catastrophe.
        price = self._get_binance_price_from_file()
        if price is None:
            price = self._get_mbt_price()  # fallback only
        if price is None:
            return

        closed = []
        for pos in self.positions:
            if pos.direction == 'SHORT':
                if price <= pos.tp_price:
                    self._close_position(pos, 'TAKE_PROFIT')
                    closed.append(pos)
                elif price >= pos.sl_price:
                    self._close_position(pos, 'STOP_LOSS')
                    closed.append(pos)
            else:  # LONG
                if price >= pos.tp_price:
                    self._close_position(pos, 'TAKE_PROFIT')
                    closed.append(pos)
                elif price <= pos.sl_price:
                    self._close_position(pos, 'STOP_LOSS')
                    closed.append(pos)

        for pos in closed:
            self.positions.remove(pos)

    def _close_position(self, pos, reason):
        """Close a position via IB and log the trade."""
        now = datetime.now()

        # Close via IB
        exit_price = pos.entry_price  # fallback
        if self.ib.isConnected():
            try:
                action = 'SELL' if pos.direction == 'LONG' else 'BUY'
                order = MarketOrder(action, pos.size)
                trade = self.ib.placeOrder(self.contract, order)
                self.ib.sleep(2)
                if trade.orderStatus.status == 'Filled':
                    exit_price = trade.orderStatus.avgFillPrice
                    logger.info("Close filled: %s %d @ $%.2f" % (action, pos.size, exit_price))
                else:
                    logger.warning("Close not filled (%s)" % trade.orderStatus.status)
            except Exception as e:
                logger.error("Close order error: %s" % e)

        # Calculate PnL from actual IB fills
        if pos.direction == 'LONG':
            pnl_pct = (exit_price / pos.entry_price - 1) * 100
        else:
            pnl_pct = (pos.entry_price / exit_price - 1) * 100
        pnl_dollar = pnl_pct / 100 * pos.entry_price * BTC_CONTRACT_VALUE * pos.size

        bars_held = int((now - pos.entry_time).total_seconds() / 16)

        # Track SL cooldown
        if reason == 'STOP_LOSS':
            self.last_sl_exit = now

        logger.info("EXIT %s [%s] %s @ $%.2f | PnL: $%.2f (%.3f%%) | held %d bars" % (
            pos.direction, pos.model_id, reason, exit_price,
            pnl_dollar, pnl_pct, bars_held))

        log_trade(DB_PATH, {
            'model_id': pos.model_id,
            'direction': pos.direction,
            'entry_time': pos.entry_time.isoformat(),
            'exit_time': now.isoformat(),
            'entry_price': pos.entry_price,
            'exit_price': exit_price,
            'pnl_pct': pnl_pct,
            'pnl_dollar': pnl_dollar,
            'exit_reason': reason,
            'bars_held': bars_held,
            'entry_probability': pos.entry_probability,
            'short_agreement': pos.short_agreement,
        })

    def read_price(self):
        """Read price from shared file (for status logging only)."""
        try:
            with open(PRICE_FILE, 'r') as f:
                return float(f.read().strip())
        except:
            return 0.0

    def log_status(self):
        """Log current status and write signal CSV."""
        now = datetime.now()

        if self.last_signal_log_time:
            elapsed = (now - self.last_signal_log_time).total_seconds()
            if elapsed < 15:
                return

        n_pos = len(self.positions)
        mbt_price = self._get_mbt_price() or 0.0
        pos_str = ", ".join(["%s %s@$%.0f (TP=$%.0f SL=$%.0f)" % (
            p.direction, p.model_id, p.entry_price, p.tp_price, p.sl_price
        ) for p in self.positions]) or "none"
        logger.info("REV | MBT: $%.2f | Pos: %d | %s" % (mbt_price, n_pos, pos_str))

        # CSV logging
        today = now.strftime('%Y-%m-%d')
        if self.signal_csv_date != today:
            self.signal_csv_date = today
            self.signal_csv_path = os.path.join(SIGNAL_LOG_DIR, 'btc_tick_reverse_signals_%s.csv' % today)
            if not os.path.exists(self.signal_csv_path):
                with open(self.signal_csv_path, 'w') as f:
                    f.write('timestamp,mbt_price,active_positions,positions\n')

        try:
            with open(self.signal_csv_path, 'a') as f:
                f.write('%s,%.2f,%d,%s\n' % (
                    now.strftime('%Y-%m-%d %H:%M:%S'),
                    mbt_price, n_pos,
                    pos_str.replace(',', ';')
                ))
        except Exception as e:
            logger.debug("Signal log error: %s" % e)

        self.last_signal_log_time = now

    def run(self):
        """Main loop."""
        logger.info("=" * 60)
        logger.info("BTC TICK BOT REVERSE [INDEPENDENT CONTRARIAN v4]")
        logger.info("=" * 60)
        logger.info("Source DB: %s" % SOURCE_DB)
        logger.info("Reverse DB: %s" % DB_PATH)
        logger.info("Mode: Contrarian — reverse LONG=%s, reverse SHORT=%s" % (REVERSE_LONG, REVERSE_SHORT))
        logger.info("TP=%.2f%% SL=%.2f%% (using IB MBT price)" % (TP_PCT, SL_PCT))
        logger.info("Max positions: %d" % MAX_POSITIONS)
        logger.info("Paper: %s" % self.paper)

        # Connect to IB
        if not self.connect_ib():
            logger.error("Cannot start without IB connection")
            return

        # Skip any existing events so we don't replay old trades on startup
        self.last_seen_event_id = self.get_current_max_event_id()
        logger.info("Starting from event ID: %d" % self.last_seen_event_id)

        logger.info("Starting main loop.")

        last_ib_check = datetime.now()
        last_reconnect_attempt = None

        while True:
            try:
                now = datetime.now()

                # Check IB connection every 60s
                if (now - last_ib_check).total_seconds() > 60:
                    if not self.ib.isConnected():
                        logger.warning("IB disconnected — reconnecting")
                        if last_reconnect_attempt is None or (now - last_reconnect_attempt).total_seconds() > 30:
                            last_reconnect_attempt = now
                            self.reconnect_ib()
                    last_ib_check = now

                # Check TP/SL exits (high priority — every tick)
                self.check_exits()

                # Poll for new ENTRY events
                entries, new_max = self.poll_entry_events()
                if new_max > self.last_seen_event_id:
                    self.last_seen_event_id = new_max

                for row in entries:
                    evt_id, evt_type, model_id, direction, price, timestamp, exit_reason, prob, agreement = row
                    logger.info("EVENT #%d: ENTRY %s %s @ $%.2f (prob=%.0f%%)" % (
                        evt_id, direction, model_id, price, (prob or 0) * 100))
                    self.handle_entry(model_id, direction, price, prob or 0, agreement or 0)

                # Log status
                self.log_status()

                # IB event processing
                if self.ib.isConnected():
                    self.ib.sleep(0.5)
                else:
                    time.sleep(0.5)

                time.sleep(PRICE_CHECK_INTERVAL)

            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error("Main loop error: %s" % e)
                time.sleep(5)

        # Cleanup: close any open positions
        for pos in list(self.positions):
            logger.info("Closing open position on shutdown: %s" % pos.model_id)
            self._close_position(pos, 'SHUTDOWN')
        self.positions.clear()

        if self.ib.isConnected():
            self.ib.disconnect()
        logger.info("Bot stopped.")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='BTC Tick Bot Reverse (Contrarian v4)')
    parser.add_argument('--paper', action='store_true', default=True)
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()

    paper = not args.live
    bot = BTCTickBotReverse(paper_trading=paper)
    bot.run()


if __name__ == "__main__":
    main()
