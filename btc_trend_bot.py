#!/usr/bin/env python3
"""
BTC Trend Follow Bot — Momentum entry on 5-min bar breakouts.

Strategy:
  LONG  entry: previous 5-min bar return > +ENTRY_THRESHOLD
  SHORT entry: previous 5-min bar return < -ENTRY_THRESHOLD
  SL=0.50%, TP=1.00%, TS activation=0.50%/trail=0.25%, max hold=2h

Execution via IB Gateway on MBT Micro Bitcoin Futures (clientId=50).
"""

import json, os, sys, time, logging, threading, sqlite3
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional
from collections import deque

from ib_insync import IB, Future, MarketOrder, util

try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================
IB_HOST            = '127.0.0.1'
IB_PORT            = 4002
IB_CLIENT_ID       = 402         # stable ID

MBT_EXPIRY         = '20260327'
BTC_CONTRACT_VALUE = 0.1
POSITION_SIZE      = 1
COMMISSION         = 2.02        # per side

ENTRY_THRESHOLD    = 0.15        # % bar return to trigger entry
STOP_LOSS_PCT      = 0.50
TAKE_PROFIT_PCT    = 1.00
TS_ACTIVATION_PCT  = 0.50
TS_TRAIL_PCT       = 0.25
MAX_HOLD_BARS      = 24          # 24 x 5min = 2h

MACRO_FILTER       = False       # disabled for initial test run — re-enable after 24h warmup
MACRO_LONG_MIN     = -2.0        # block LONG if 24h < -2%
MACRO_SHORT_MAX    = 2.0         # block SHORT if 24h > +2%

SL_COOLDOWN_SEC    = 1800        # 30 min

BAR_SECONDS        = 300         # 5-min bars
LOOKBACK_24H       = 288         # bars in 24h
DB_PATH            = 'trend_trades.db'
LOG_DIR            = 'logs'
WEBSOCKET_URL      = "wss://stream.binance.com:9443/ws/btcusdt@trade"

# =============================================================================
# LOGGING
# =============================================================================
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{LOG_DIR}/btc_trend_bot.log'),
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# DATABASE
# =============================================================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            direction     TEXT,
            entry_time    TEXT,
            entry_price   REAL,
            exit_time     TEXT,
            exit_price    REAL,
            exit_reason   TEXT,
            pnl_pct       REAL,
            pnl_dollar    REAL,
            bars_held     INTEGER,
            trigger_ret   REAL,
            macro_24h     REAL,
            created_at    TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def log_trade(direction, entry_time, entry_price, exit_time, exit_price,
              exit_reason, pnl_pct, pnl_dollar, bars_held, trigger_ret, macro_24h):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO trades (direction,entry_time,entry_price,exit_time,exit_price,"
            "exit_reason,pnl_pct,pnl_dollar,bars_held,trigger_ret,macro_24h) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (direction, entry_time, entry_price, exit_time, exit_price,
             exit_reason, round(pnl_pct, 4), round(pnl_dollar, 2),
             bars_held, round(trigger_ret, 4), round(macro_24h, 3) if macro_24h else None)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"DB error: {e}")

# =============================================================================
# MARKET HOURS (CME MBT: Sun 5pm – Fri 4pm CT, daily 4-5pm break)
# =============================================================================
def _ct_now():
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo('America/Chicago')
    except ImportError:
        import pytz
        tz = pytz.timezone('America/Chicago')
    return datetime.now(tz).replace(tzinfo=None)

def is_market_open() -> bool:
    now = _ct_now()
    wd = now.weekday()   # 0=Mon … 6=Sun
    h  = now.hour
    if wd == 5: return False                          # Saturday
    if wd == 6: return h >= 17                        # Sunday opens 5pm
    if h == 16: return False                          # daily maintenance 4-5pm
    if wd == 4 and h >= 16: return False              # Friday after 4pm
    return True

# =============================================================================
# POSITION
# =============================================================================
@dataclass
class Position:
    direction:    str
    entry_price:  float
    entry_time:   str
    stop_price:   float
    target_price: float
    trigger_ret:  float   = 0.0
    macro_24h:    float   = 0.0
    bars_held:    int     = 0
    peak_price:   float   = 0.0
    trough_price: float   = 0.0
    ts_active:    bool    = False
    ts_price:     float   = 0.0

# =============================================================================
# BOT
# =============================================================================
class TrendBot:
    def __init__(self):
        self.ib           = IB()
        self.contract     = None
        self.position: Optional[Position] = None
        self.current_price: Optional[float] = None

        # 5-min bar state
        self._bar_open:   Optional[float]    = None
        self._bar_high:   float              = 0.0
        self._bar_low:    float              = float('inf')
        self._bar_close:  Optional[float]    = None
        self._bar_start:  Optional[datetime] = None
        self._prev_ret:   float              = 0.0   # last completed bar % return

        # Ring buffer of bar closes for macro trend
        self._closes: deque = deque(maxlen=LOOKBACK_24H + 10)

        # SL cooldown
        self._last_sl: dict = {'LONG': datetime.min, 'SHORT': datetime.min}

        self._lock = threading.Lock()
        self._ws   = None
        init_db()

    # ------------------------------------------------------------------
    # IB
    # ------------------------------------------------------------------
    def connect_ib(self):
        max_retries = 999
        for attempt in range(1, max_retries + 1):
            try:
                try:
                    if self.ib.isConnected():
                        self.ib.disconnect()
                        time.sleep(2)
                except Exception:
                    pass
                logger.info(f"Connecting IB {IB_HOST}:{IB_PORT} clientId={IB_CLIENT_ID} (attempt {attempt})")
                self.ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID, timeout=20)
                self.contract = Future(
                    symbol='MBT',
                    lastTradeDateOrContractMonth=MBT_EXPIRY,
                    exchange='CME',
                    currency='USD',
                )
                self.ib.qualifyContracts(self.contract)
                logger.info(f"IB connected — {self.contract.localSymbol}")
                return
            except Exception as e:
                logger.error(f"IB connection attempt {attempt} failed: {e}")
                time.sleep(30)

    def place_order(self, direction: str) -> Optional[tuple]:
        """Place market order. Returns (order_id, fill_price) or None."""
        try:
            action = 'BUY' if direction == 'LONG' else 'SELL'
            trade  = self.ib.placeOrder(self.contract, MarketOrder(action, POSITION_SIZE))
            self.ib.sleep(2)
            fp = trade.orderStatus.avgFillPrice
            logger.info(f"Order filled: {action} {POSITION_SIZE} @ {fp}")
            return trade.order.orderId, fp if fp else self.current_price
        except Exception as e:
            logger.error(f"Order error: {e}")
            return None

    def close_position(self, pos: Position) -> Optional[float]:
        """Close position. Returns fill price or None on failure."""
        try:
            action = 'SELL' if pos.direction == 'LONG' else 'BUY'
            trade  = self.ib.placeOrder(self.contract, MarketOrder(action, POSITION_SIZE))
            self.ib.sleep(2)
            fp = trade.orderStatus.avgFillPrice
            return fp if fp else self.current_price
        except Exception as e:
            logger.error(f"Close error: {e}")
            return None

    # ------------------------------------------------------------------
    # WebSocket
    # ------------------------------------------------------------------
    def _on_message(self, ws, message):
        try:
            price = float(json.loads(message)['p'])
            with self._lock:
                self.current_price = price
                self._tick(price)
        except Exception:
            pass

    def _on_error(self, ws, err):
        logger.warning(f"WS error: {err}")

    def _on_close(self, ws, *args):
        logger.info("WS closed — reconnecting in 5s")
        time.sleep(5)
        self._connect_ws()

    def _connect_ws(self):
        if not WEBSOCKET_AVAILABLE:
            logger.error("websocket-client not installed")
            return
        self._ws = websocket.WebSocketApp(
            WEBSOCKET_URL,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=lambda ws: logger.info("WS connected"),
        )
        threading.Thread(target=self._ws.run_forever, daemon=True).start()

    # ------------------------------------------------------------------
    # 5-min bar builder (called under _lock on every tick)
    # ------------------------------------------------------------------
    def _tick(self, price: float):
        now = datetime.utcnow()

        if self._bar_start is None:
            # Align to 5-min boundary
            aligned = now.replace(second=0, microsecond=0)
            aligned = aligned.replace(minute=(aligned.minute // 5) * 5)
            self._bar_start = aligned
            self._bar_open  = price
            self._bar_high  = price
            self._bar_low   = price
            self._bar_close = price
            return

        self._bar_high  = max(self._bar_high, price)
        self._bar_low   = min(self._bar_low,  price)
        self._bar_close = price

        if (now - self._bar_start).total_seconds() >= BAR_SECONDS:
            self._finish_bar()
            next_start = self._bar_start + timedelta(seconds=BAR_SECONDS)
            self._bar_start = next_start
            self._bar_open  = price
            self._bar_high  = price
            self._bar_low   = price
            self._bar_close = price

    def _finish_bar(self):
        if self._bar_open and self._bar_close:
            self._prev_ret = (self._bar_close / self._bar_open - 1) * 100
            self._closes.append(self._bar_close)
            logger.debug(f"Bar closed: open={self._bar_open:.2f} close={self._bar_close:.2f} "
                         f"ret={self._prev_ret:+.3f}%")
            # Check entry on bar close (no position open)
            if self.position is None and is_market_open():
                self._check_entry(self._bar_close, self._prev_ret)

    # ------------------------------------------------------------------
    # Trailing stop update
    # ------------------------------------------------------------------
    def _update_ts(self, pos: Position, price: float):
        if pos.direction == 'LONG':
            profit_pct = (price / pos.entry_price - 1) * 100
            if not pos.ts_active:
                if profit_pct >= TS_ACTIVATION_PCT:
                    pos.ts_active = True
                    pos.peak_price = price
                    pos.ts_price   = pos.entry_price * (1 + TS_ACTIVATION_PCT / 100)
                    logger.info(f"[TS] LONG activated @ ${price:.2f}, stop=${pos.ts_price:.2f}")
            else:
                if price > pos.peak_price:
                    pos.peak_price = price
                    new_stop = pos.peak_price * (1 - TS_TRAIL_PCT / 100)
                    if new_stop > pos.ts_price:
                        pos.ts_price = new_stop
        else:  # SHORT
            profit_pct = (pos.entry_price / price - 1) * 100
            if not pos.ts_active:
                if profit_pct >= TS_ACTIVATION_PCT:
                    pos.ts_active  = True
                    pos.trough_price = price
                    pos.ts_price     = pos.entry_price * (1 - TS_ACTIVATION_PCT / 100)
                    logger.info(f"[TS] SHORT activated @ ${price:.2f}, stop=${pos.ts_price:.2f}")
            else:
                if price < pos.trough_price:
                    pos.trough_price = price
                    new_stop = pos.trough_price * (1 + TS_TRAIL_PCT / 100)
                    if new_stop < pos.ts_price:
                        pos.ts_price = new_stop

    # ------------------------------------------------------------------
    # Macro 24h trend
    # ------------------------------------------------------------------
    def _macro_24h(self) -> Optional[float]:
        closes = list(self._closes)
        if len(closes) < LOOKBACK_24H:
            return None
        return (closes[-1] / closes[-LOOKBACK_24H] - 1) * 100

    # ------------------------------------------------------------------
    # Entry logic (called at bar close when no position is open)
    # ------------------------------------------------------------------
    def _check_entry(self, price: float, bar_ret: float):
        direction = None
        if bar_ret >= ENTRY_THRESHOLD:
            direction = 'LONG'
        elif bar_ret <= -ENTRY_THRESHOLD:
            direction = 'SHORT'
        else:
            return  # no signal

        # SL cooldown
        secs = (datetime.now() - self._last_sl[direction]).total_seconds()
        if secs < SL_COOLDOWN_SEC:
            logger.info(f"[COOLDOWN] {direction} blocked: {(SL_COOLDOWN_SEC-secs)/60:.0f}min left")
            return

        # Macro trend filter
        macro = self._macro_24h()
        if MACRO_FILTER and macro is not None:
            if direction == 'LONG' and macro < MACRO_LONG_MIN:
                logger.info(f"[MACRO] LONG blocked: 24h={macro:+.2f}% < {MACRO_LONG_MIN}%")
                return
            if direction == 'SHORT' and macro > MACRO_SHORT_MAX:
                logger.info(f"[MACRO] SHORT blocked: 24h={macro:+.2f}% > {MACRO_SHORT_MAX}%")
                return

        logger.info(f"[SIGNAL] {direction} — bar_ret={bar_ret:+.3f}% @ ${price:.2f} | "
                    f"macro_24h={macro:+.2f}%" if macro is not None else
                    f"[SIGNAL] {direction} — bar_ret={bar_ret:+.3f}% @ ${price:.2f}")

        result = self.place_order(direction)
        if result is None:
            logger.error(f"Entry order failed for {direction}")
            return

        _, fill_price = result
        ep = fill_price if fill_price and fill_price > 0 else price

        if direction == 'LONG':
            stop_p   = ep * (1 - STOP_LOSS_PCT   / 100)
            target_p = ep * (1 + TAKE_PROFIT_PCT  / 100)
        else:
            stop_p   = ep * (1 + STOP_LOSS_PCT   / 100)
            target_p = ep * (1 - TAKE_PROFIT_PCT  / 100)

        self.position = Position(
            direction    = direction,
            entry_price  = ep,
            entry_time   = datetime.now().isoformat(),
            stop_price   = stop_p,
            target_price = target_p,
            trigger_ret  = bar_ret,
            macro_24h    = macro if macro is not None else 0.0,
            peak_price   = ep,
            trough_price = ep,
        )
        logger.info(f"ENTERED {direction} @ ${ep:.2f} | SL=${stop_p:.2f} TP=${target_p:.2f}")

    # ------------------------------------------------------------------
    # Exit logic (called every 2 sec from main loop)
    # ------------------------------------------------------------------
    def _check_exits(self):
        pos = self.position
        if pos is None or self.current_price is None:
            return

        price = self.current_price
        self._update_ts(pos, price)

        exit_reason = None

        if pos.direction == 'LONG':
            if price <= pos.stop_price:
                exit_reason = 'STOP_LOSS'
            elif pos.ts_active and price <= pos.ts_price:
                exit_reason = 'TRAILING_STOP'
            elif price >= pos.target_price:
                exit_reason = 'TAKE_PROFIT'
            elif pos.bars_held >= MAX_HOLD_BARS:
                exit_reason = 'TIMEOUT'
        else:  # SHORT
            if price >= pos.stop_price:
                exit_reason = 'STOP_LOSS'
            elif pos.ts_active and price >= pos.ts_price:
                exit_reason = 'TRAILING_STOP'
            elif price <= pos.target_price:
                exit_reason = 'TAKE_PROFIT'
            elif pos.bars_held >= MAX_HOLD_BARS:
                exit_reason = 'TIMEOUT'

        if exit_reason is None:
            return

        fill_price = self.close_position(pos)
        if fill_price is None:
            logger.error("Close order failed — will retry next tick")
            return

        if pos.direction == 'LONG':
            pnl_pct = (fill_price / pos.entry_price - 1) * 100
        else:
            pnl_pct = (pos.entry_price / fill_price - 1) * 100

        pnl_dollar = pnl_pct / 100 * pos.entry_price * BTC_CONTRACT_VALUE - COMMISSION * 2

        logger.info(f"EXIT [{exit_reason}] {pos.direction} @ ${fill_price:.2f} | "
                    f"pnl={pnl_pct:+.3f}% (${pnl_dollar:+.2f}) bars={pos.bars_held}")

        log_trade(
            direction    = pos.direction,
            entry_time   = pos.entry_time,
            entry_price  = pos.entry_price,
            exit_time    = datetime.now().isoformat(),
            exit_price   = fill_price,
            exit_reason  = exit_reason,
            pnl_pct      = pnl_pct,
            pnl_dollar   = pnl_dollar,
            bars_held    = pos.bars_held,
            trigger_ret  = pos.trigger_ret,
            macro_24h    = pos.macro_24h,
        )

        if exit_reason == 'STOP_LOSS':
            self._last_sl[pos.direction] = datetime.now()

        self.position = None

    # ------------------------------------------------------------------
    # Increment bars_held every bar close
    # ------------------------------------------------------------------
    def _on_bar_close(self):
        if self.position is not None:
            self.position.bars_held += 1

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------
    def run(self):
        self.connect_ib()
        self._connect_ws()

        logger.info("BTC Trend Bot running — waiting for price feed...")
        time.sleep(5)

        last_bar_time: Optional[datetime] = None

        try:
            while True:
                try:
                    self.ib.sleep(0.1)  # keep IB event loop alive
                except (ConnectionError, Exception) as e:
                    logger.warning(f"IB socket error: {e} — reconnecting")
                    try:
                        if self.ib.isConnected():
                            self.ib.disconnect()
                            time.sleep(2)
                    except Exception:
                        pass
                    self.connect_ib()
                    continue

                with self._lock:
                    bar_start = self._bar_start

                # Detect bar close transition
                if bar_start is not None and bar_start != last_bar_time:
                    if last_bar_time is not None:
                        self._on_bar_close()
                    last_bar_time = bar_start

                # Check exits every iteration (~2 sec cadence from WS ticks)
                with self._lock:
                    self._check_exits()

                # Status log
                if self.current_price:
                    pos_info = (f"pos={self.position.direction}@${self.position.entry_price:.2f}"
                                f"({self.position.bars_held}bars)"
                                if self.position else "pos=None")
                    logger.debug(f"BTC: ${self.current_price:.2f} | {pos_info} | "
                                 f"last_bar={self._prev_ret:+.3f}%")

                time.sleep(2)

        except KeyboardInterrupt:
            logger.info("Interrupted — shutting down")
        finally:
            if self._ws:
                self._ws.close()
            try:
                self.ib.disconnect()
            except Exception:
                pass
            logger.info("Disconnected")


# =============================================================================
# ENTRY POINT
# =============================================================================
def main():
    util.patchAsyncio()
    bot = TrendBot()
    bot.run()

if __name__ == '__main__':
    main()
