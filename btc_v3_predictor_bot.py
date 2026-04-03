"""
BTC V3 Predictor Trading Bot

Uses btc_model_package.predictor.BTCPredictor (V3 ensemble: 2h/4h/6h models)
with Binance BTCUSDT 5-min data for signals and IB Gateway for MBT execution.

Architecture (Binance-only, consistent with other BTC bots since Mar 31):
- Binance BTCUSDT spot = single source of truth for price, signals, SL/TP/TS
- IB Gateway = execution layer only (place/close orders)
- Position persistence via JSON (survives restarts)
- Trade logging to SQLite via trade_database.py

Model specs (from btc_model_package):
- Strategy: TP=1.0% / SL=0.5% (2:1 R:R, breakeven WR=33.3%)
- Thresholds: LONG >0.60, SHORT <0.30 (optimized for 22-feature model)
- Walk-forward validated: 42.7% WR, +$2.75 avg P&L per trade, +$4,430 total (2yr)
- Features: 22 (support/resistance, ATR, volume MAs, trend, RSI, BB, time encoding)
- Input: 250 bars of 5-min OHLCV from Binance

IB clientId=404, port 4002 (paper)
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import time
import logging
import requests
import threading
import argparse
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, fields
from ib_insync import IB, Future, MarketOrder, util

# Add parent directory so we can import from btc_model_package and shared modules
BOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BOT_DIR)

from btc_model_package.predictor import BTCPredictor
from trade_database import log_trade_to_db
from market_hours import MarketHours, get_cme_market_hours

try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

# =============================================================================
# LOGGING
# =============================================================================
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/btc_v3_predictor.log', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================
# IB Gateway
IB_HOST = '127.0.0.1'
IB_PORT = 4002
IB_CLIENT_ID = 404

# BTC MBT contract
BTC_CONTRACT_VALUE = 0.1  # MBT = 0.1 BTC

# Binance
BINANCE_API = "https://api.binance.com/api/v3"
BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@miniTicker"

# Trading parameters (from predictor defaults)
TAKE_PROFIT_PCT = 1.0     # 1% TP
STOP_LOSS_PCT = 0.5       # 0.5% SL
TRAILING_STOP_PCT = 0.10          # Trail distance after activation
TRAILING_STOP_ACTIVATION_PCT = 0.60  # Activate trailing stop after +0.60%

# Confidence filter (predictor already applies thresholds, but we add a floor)
MIN_CONFIDENCE = 0.60  # Match 22-feature model optimized LONG threshold

# Position limits
MAX_LONG_POSITIONS = 1
MAX_SHORT_POSITIONS = 1
MAX_POSITIONS = MAX_LONG_POSITIONS + MAX_SHORT_POSITIONS

# Timing
SIGNAL_CHECK_INTERVAL = 120  # Check for new signals every 2 minutes (per integration guide)
PRICE_CHECK_INTERVAL = 2     # Check price every 2 seconds for TS/SL
BINANCE_DATA_LIMIT = 250     # 250 bars of 5-min data (~20.8 hours)

# SL cooldown
SL_COOLDOWN_SECONDS = 1800  # 30 min after SL before re-entering same direction

# Timeout: close position if held too long (6h = 72 bars of 5-min)
TIMEOUT_BARS = 72

# Signal logging
SIGNAL_LOG_DIR = "signal_logs"

# Position persistence
POSITION_FILE = "btc_v3_positions.json"

# Trade database
DB_PATH = "v3_predictor_trades.db"


# =============================================================================
# DATA CLASSES
# =============================================================================
@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    direction: str
    size: int
    entry_price: float
    entry_time: str
    model_id: str
    target_price: float = 0.0
    stop_price: float = 0.0
    bars_held: int = 0
    order_id: Optional[int] = None
    pending_close: bool = False
    # Trailing stop
    peak_price: float = 0.0
    trough_price: float = 0.0
    trailing_stop_active: bool = False
    trailing_stop_price: float = 0.0
    # Entry metadata
    entry_confidence: float = 0.0
    entry_avg_probability: float = 0.0
    entry_probabilities: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'Position':
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


class PositionManager:
    """Manages positions with JSON persistence."""

    def __init__(self, filepath: str = POSITION_FILE):
        self.filepath = filepath
        self.positions: Dict[str, Position] = {}
        self.load_positions()

    def load_positions(self) -> None:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                    for pos_id, pos_data in data.items():
                        self.positions[pos_id] = Position.from_dict(pos_data)
                logger.info(f"Loaded {len(self.positions)} positions from {self.filepath}")
                for pos_id, pos in self.positions.items():
                    logger.info(f"  - {pos_id}: {pos.direction} @ ${pos.entry_price:.2f} "
                               f"(stop: ${pos.stop_price:.2f}, TS active: {pos.trailing_stop_active})")
            except Exception as e:
                logger.error(f"Error loading positions: {e}")
                self.positions = {}

    def save_positions(self) -> None:
        try:
            with open(self.filepath, 'w') as f:
                json.dump({k: v.to_dict() for k, v in self.positions.items()}, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving positions: {e}")

    def add_position(self, position: Position) -> None:
        self.positions[position.model_id] = position
        self.save_positions()
        logger.info(f"Added position [{position.model_id}]: {position.direction} "
                   f"{position.size} @ ${position.entry_price:.2f}")

    def remove_position(self, model_id: str) -> Optional[Position]:
        if model_id in self.positions:
            position = self.positions.pop(model_id)
            self.save_positions()
            logger.info(f"Removed position [{model_id}]")
            return position
        return None

    def get_all_positions(self) -> List[Position]:
        return list(self.positions.values())

    def count_positions(self) -> int:
        return len(self.positions)

    def count_by_direction(self, direction: str) -> int:
        return sum(1 for p in self.positions.values() if p.direction == direction)

    def has_position(self, model_id: str) -> bool:
        return model_id in self.positions

    def increment_bars_held(self) -> None:
        for pos in self.positions.values():
            pos.bars_held += 1
        self.save_positions()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_btc_front_month_expiry() -> str:
    """Get MBT front month expiry in YYYYMM format with auto-roll."""
    import calendar
    now = datetime.now()
    year, month = now.year, now.month

    c = calendar.Calendar(firstweekday=calendar.SUNDAY)
    monthcal = c.monthdatescalendar(year, month)
    fridays = [day for week in monthcal for day in week
               if day.weekday() == calendar.FRIDAY and day.month == month]
    last_friday = fridays[-1]

    roll_date = last_friday - timedelta(days=5)
    if now.date() >= roll_date:
        if month == 12:
            month, year = 1, year + 1
        else:
            month += 1

    return f"{year}{month:02d}"


def is_cme_mbt_open() -> bool:
    """Check if CME MBT market is open."""
    market_hours = get_cme_market_hours()
    is_open, reason = market_hours.is_market_open()
    if not is_open:
        logger.debug(f"CME MBT closed: {reason}")
    return is_open


def should_close_before_market_close() -> Tuple[bool, str]:
    """Check if positions should be closed before upcoming market closure."""
    market_hours = get_cme_market_hours()
    return market_hours.should_close_positions()


def should_block_entries() -> Tuple[bool, str]:
    """Check if new entries should be blocked (insufficient time before close)."""
    market_hours = get_cme_market_hours()
    return market_hours.should_block_new_entries(min_holding_time_minutes=60)


# =============================================================================
# MAIN BOT
# =============================================================================
class BTCV3PredictorBot:
    """BTC trading bot using V3 BTCPredictor ensemble + IB execution."""

    def __init__(self, paper_trading: bool = True, position_size: int = 1):
        self.paper_trading = paper_trading
        self.position_size = position_size

        # IB
        self.ib = IB()
        self.connected = False
        expiry = get_btc_front_month_expiry()
        self.contract = Future('MBT', expiry, 'CME')

        # Predictor
        model_dir = os.path.join(BOT_DIR, 'btc_model_package')
        self.predictor = BTCPredictor(model_dir=model_dir)
        self.predictor.LONG_THRESHOLD = 0.60
        self.predictor.SHORT_THRESHOLD = 0.30
        logger.info(f"BTCPredictor loaded from {model_dir} (22-feature, LONG>0.60 SHORT<0.30)")

        # Position manager
        self.position_manager = PositionManager()

        # Price state
        self.current_price: Optional[float] = None
        self.last_price_time: Optional[datetime] = None
        self.ws_price: Optional[float] = None
        self.ws_price_time: Optional[datetime] = None
        self.ws_connected = False
        self.ws = None
        self.ws_thread = None

        # Signal state
        self.last_signal: Optional[str] = None
        self.last_signal_time: Optional[datetime] = None
        self.last_bar_time: Optional[datetime] = None

        # SL cooldown tracking
        self.last_sl_time: Dict[str, datetime] = {
            'LONG': datetime.min,
            'SHORT': datetime.min
        }

        # Pending close orders
        self.pending_close_orders: Dict[str, datetime] = {}

        # Stats
        self.start_time = datetime.now()
        self.total_checks = 0
        self.total_signals = 0
        self.total_trades = 0

        # Signal logger
        os.makedirs(SIGNAL_LOG_DIR, exist_ok=True)

        logger.info("=" * 60)
        logger.info("BTC V3 PREDICTOR BOT INITIALIZED")
        logger.info("=" * 60)
        logger.info(f"Contract: MBT {expiry} (Micro Bitcoin Futures)")
        logger.info(f"TP: {TAKE_PROFIT_PCT}% | SL: {STOP_LOSS_PCT}% | TS: {TRAILING_STOP_ACTIVATION_PCT}%/{TRAILING_STOP_PCT}%")
        logger.info(f"Min confidence: {MIN_CONFIDENCE}")
        logger.info(f"Max positions: {MAX_LONG_POSITIONS} LONG, {MAX_SHORT_POSITIONS} SHORT")
        logger.info(f"Signal interval: {SIGNAL_CHECK_INTERVAL}s | Price check: {PRICE_CHECK_INTERVAL}s")
        logger.info(f"Paper trading: {paper_trading}")
        logger.info(f"IB clientId: {IB_CLIENT_ID}")

    # =========================================================================
    # IB CONNECTION
    # =========================================================================
    def connect(self) -> bool:
        """Connect to IB Gateway."""
        try:
            try:
                if self.ib.isConnected():
                    self.ib.disconnect()
                    time.sleep(2)
            except Exception:
                pass

            port = IB_PORT if self.paper_trading else 7496
            self.ib.connect(IB_HOST, port, clientId=IB_CLIENT_ID, timeout=20)
            self.ib.reqMarketDataType(4)
            self.ib.qualifyContracts(self.contract)
            self.connected = True
            logger.info(f"Connected to IB Gateway (port {port}, clientId {IB_CLIENT_ID})")
            logger.info(f"Trading contract: {self.contract}")
            return True
        except Exception as e:
            logger.error(f"IB connection failed: {e}")
            return False

    def disconnect(self) -> None:
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            logger.info("Disconnected from IB Gateway")

    def is_connected(self) -> bool:
        try:
            return self.ib.isConnected()
        except Exception:
            return False

    def reconnect(self) -> bool:
        """Reconnect to IB Gateway, retrying indefinitely."""
        logger.warning("Attempting IB reconnect...")
        try:
            if self.ib.isConnected():
                self.ib.disconnect()
                time.sleep(2)
            self.connected = False
        except Exception:
            pass

        attempt = 0
        while True:
            attempt += 1
            logger.info(f"Reconnection attempt {attempt}...")
            try:
                if self.connect():
                    logger.info(f"Reconnected on attempt {attempt}")
                    return True
            except Exception as e:
                logger.warning(f"Attempt {attempt} failed: {e}")
            delay = min(30 * (1 + attempt // 10), 300)
            time.sleep(delay)

    # =========================================================================
    # BINANCE DATA
    # =========================================================================
    def get_binance_klines(self, limit: int = BINANCE_DATA_LIMIT) -> pd.DataFrame:
        """Fetch 5-min OHLCV from Binance REST API with retry."""
        max_retries = 5
        for attempt in range(max_retries):
            try:
                url = f"{BINANCE_API}/klines"
                params = {'symbol': 'BTCUSDT', 'interval': '5m', 'limit': limit}
                response = requests.get(url, params=params, timeout=15)
                response.raise_for_status()
                data = response.json()

                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)
                df = df[['open', 'high', 'low', 'close', 'volume']]

                if attempt > 0:
                    logger.info(f"Binance API recovered after {attempt + 1} attempts")
                return df

            except Exception as e:
                delay = (2 ** attempt)
                if attempt < max_retries - 1:
                    logger.warning(f"Binance attempt {attempt+1}/{max_retries} failed: {e}. Retry in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"Binance API failed after {max_retries} attempts: {e}")

        return pd.DataFrame()

    def get_binance_ticker_price(self) -> Optional[float]:
        """Get current BTC price from Binance ticker (lightweight)."""
        try:
            url = f"{BINANCE_API}/ticker/price"
            response = requests.get(url, params={'symbol': 'BTCUSDT'}, timeout=5)
            response.raise_for_status()
            return float(response.json()['price'])
        except Exception as e:
            logger.debug(f"Binance ticker error: {e}")
            return None

    # =========================================================================
    # WEBSOCKET (real-time price for TS/SL)
    # =========================================================================
    def start_websocket(self) -> None:
        """Start Binance WebSocket for real-time price updates."""
        if not WEBSOCKET_AVAILABLE:
            logger.warning("websocket-client not installed — using REST for prices")
            return

        def on_message(ws, message):
            try:
                data = json.loads(message)
                if 'c' in data:
                    self.ws_price = float(data['c'])
                    self.ws_price_time = datetime.now()
            except Exception:
                pass

        def on_error(ws, error):
            logger.warning(f"WebSocket error: {error}")
            self.ws_connected = False

        def on_close(ws, status_code, msg):
            logger.info(f"WebSocket closed: {status_code}")
            self.ws_connected = False

        def on_open(ws):
            logger.info("WebSocket connected — receiving real-time BTC prices")
            self.ws_connected = True

        def run_ws():
            while True:
                try:
                    self.ws = websocket.WebSocketApp(
                        BINANCE_WS_URL,
                        on_message=on_message,
                        on_error=on_error,
                        on_close=on_close,
                        on_open=on_open
                    )
                    self.ws.run_forever()
                except Exception as e:
                    logger.error(f"WebSocket thread error: {e}")
                if not self.ws_connected:
                    logger.info("WebSocket reconnecting in 5s...")
                    time.sleep(5)

        self.ws_thread = threading.Thread(target=run_ws, daemon=True)
        self.ws_thread.start()

    def stop_websocket(self) -> None:
        if self.ws:
            self.ws.close()
            self.ws_connected = False

    def get_realtime_price(self) -> Optional[float]:
        """Get price: WS → REST fallback."""
        # Tier 1: WebSocket (< 5s old)
        if self.ws_price and self.ws_price_time:
            age = (datetime.now() - self.ws_price_time).total_seconds()
            if age < 5:
                return self.ws_price

        # Tier 2: REST ticker
        rest_price = self.get_binance_ticker_price()
        if rest_price:
            return rest_price

        return None

    # =========================================================================
    # IB ORDER EXECUTION
    # =========================================================================
    def place_order(self, direction: str, size: int) -> Optional[Tuple[int, float]]:
        """Place market order. Returns (order_id, fill_price) or None."""
        try:
            action = 'BUY' if direction == 'LONG' else 'SELL'
            order = MarketOrder(action, size)
            trade = self.ib.placeOrder(self.contract, order)
            self.ib.sleep(2)

            if trade.orderStatus.status == 'Filled':
                fill_price = trade.orderStatus.avgFillPrice
                logger.info(f"Order filled: {action} {size} @ ${fill_price:.2f}")
                return trade.order.orderId, fill_price
            else:
                logger.warning(f"Order status: {trade.orderStatus.status}")
                return trade.order.orderId, self.current_price
        except Exception as e:
            logger.error(f"Order error: {e}")
            return None

    def close_position(self, position: Position) -> object:
        """Close a position. Returns fill_price if filled, 'FLAT' if already flat, False otherwise."""
        try:
            # Safety: verify IB position exists
            try:
                ib_positions = self.ib.positions()
                mbt_pos = next((p for p in ib_positions if p.contract.symbol == 'MBT'), None)
                if mbt_pos is None or mbt_pos.position == 0:
                    logger.warning(f"[{position.model_id}] IB already flat — removing tracker")
                    return 'FLAT'
            except Exception as e:
                logger.warning(f"Could not verify IB position: {e} — proceeding")

            action = 'SELL' if position.direction == 'LONG' else 'BUY'
            order = MarketOrder(action, position.size)
            trade = self.ib.placeOrder(self.contract, order)
            self.ib.sleep(2)

            status = trade.orderStatus.status
            if status == 'Filled':
                fill_price = trade.orderStatus.avgFillPrice
                pnl = (fill_price - position.entry_price) * BTC_CONTRACT_VALUE * position.size
                if position.direction == 'SHORT':
                    pnl = -pnl
                logger.info(f"Closed [{position.model_id}]: ${fill_price:.2f}, IB P&L: ${pnl:.2f}")
                if position.model_id in self.pending_close_orders:
                    del self.pending_close_orders[position.model_id]
                return fill_price
            elif status in ['PreSubmitted', 'Submitted', 'Inactive', 'PendingSubmit']:
                self.pending_close_orders[position.model_id] = datetime.now()
                logger.warning(f"Close order PENDING [{position.model_id}]: {status}")
                return False
            else:
                logger.warning(f"Close order status: {status}")
                return False
        except Exception as e:
            logger.error(f"Close error: {e}")
            return False

    # =========================================================================
    # TRAILING STOP
    # =========================================================================
    def _update_trailing_stop(self, pos: Position, price: float) -> None:
        """Update trailing stop tracking for a position."""
        if pos.direction == 'LONG':
            if price > pos.peak_price:
                pos.peak_price = price

            pnl_pct = (price / pos.entry_price - 1) * 100
            if not pos.trailing_stop_active and pnl_pct >= TRAILING_STOP_ACTIVATION_PCT:
                pos.trailing_stop_active = True
                pos.trailing_stop_price = price * (1 - TRAILING_STOP_PCT / 100)
                logger.info(f"[{pos.model_id}] TS ACTIVATED: price ${price:.2f} (+{pnl_pct:.2f}%), "
                           f"trail @ ${pos.trailing_stop_price:.2f}")

            if pos.trailing_stop_active:
                new_trail = price * (1 - TRAILING_STOP_PCT / 100)
                if new_trail > pos.trailing_stop_price:
                    pos.trailing_stop_price = new_trail

        elif pos.direction == 'SHORT':
            if pos.trough_price == 0 or price < pos.trough_price:
                pos.trough_price = price

            pnl_pct = (pos.entry_price / price - 1) * 100
            if not pos.trailing_stop_active and pnl_pct >= TRAILING_STOP_ACTIVATION_PCT:
                pos.trailing_stop_active = True
                pos.trailing_stop_price = price * (1 + TRAILING_STOP_PCT / 100)
                logger.info(f"[{pos.model_id}] TS ACTIVATED: price ${price:.2f} (+{pnl_pct:.2f}%), "
                           f"trail @ ${pos.trailing_stop_price:.2f}")

            if pos.trailing_stop_active:
                new_trail = price * (1 + TRAILING_STOP_PCT / 100)
                if new_trail < pos.trailing_stop_price:
                    pos.trailing_stop_price = new_trail

        self.position_manager.save_positions()

    # =========================================================================
    # EXIT CHECKS
    # =========================================================================
    def check_exits(self) -> None:
        """Check SL/TP/TS/timeout for all positions."""
        positions = self.position_manager.get_all_positions()
        if not positions or self.current_price is None or self.current_price <= 0:
            return

        if not is_cme_mbt_open():
            return

        for pos in positions:
            if pos.model_id in self.pending_close_orders:
                continue

            # Update trailing stop
            self._update_trailing_stop(pos, self.current_price)

            should_exit = False
            reason = ""
            pnl_pct = 0.0

            if pos.direction == 'LONG':
                pnl_pct = (self.current_price / pos.entry_price - 1) * 100

                if pos.target_price > 0 and self.current_price >= pos.target_price:
                    should_exit = True
                    reason = f"TAKE PROFIT @ ${self.current_price:.2f} (+{pnl_pct:.2f}%)"
                elif pos.trailing_stop_active and self.current_price <= pos.trailing_stop_price:
                    should_exit = True
                    reason = f"TRAILING STOP @ ${self.current_price:.2f} ({pnl_pct:+.2f}%)"
                elif pos.stop_price > 0 and self.current_price <= pos.stop_price:
                    should_exit = True
                    reason = f"STOP LOSS @ ${self.current_price:.2f} ({pnl_pct:.2f}%)"

            elif pos.direction == 'SHORT':
                pnl_pct = (pos.entry_price / self.current_price - 1) * 100

                if pos.target_price > 0 and self.current_price <= pos.target_price:
                    should_exit = True
                    reason = f"TAKE PROFIT @ ${self.current_price:.2f} (+{pnl_pct:.2f}%)"
                elif pos.trailing_stop_active and self.current_price >= pos.trailing_stop_price:
                    should_exit = True
                    reason = f"TRAILING STOP @ ${self.current_price:.2f} ({pnl_pct:+.2f}%)"
                elif pos.stop_price > 0 and self.current_price >= pos.stop_price:
                    should_exit = True
                    reason = f"STOP LOSS @ ${self.current_price:.2f} ({pnl_pct:.2f}%)"

            # Timeout
            entry_time = datetime.fromisoformat(pos.entry_time)
            elapsed_mins = (datetime.now() - entry_time).total_seconds() / 60
            bars_elapsed = int(elapsed_mins / 5)

            if not should_exit and bars_elapsed >= TIMEOUT_BARS:
                should_exit = True
                reason = f"TIMEOUT ({bars_elapsed}/{TIMEOUT_BARS} bars) @ ${self.current_price:.2f} ({pnl_pct:+.2f}%)"

            # Pre-close
            if not should_exit:
                pre_close, pre_close_reason = should_close_before_market_close()
                if pre_close:
                    should_exit = True
                    reason = f"PRE-CLOSE ({pre_close_reason}) @ ${self.current_price:.2f} ({pnl_pct:+.2f}%)"

            if should_exit:
                if pos.model_id in self.pending_close_orders:
                    continue
                self.pending_close_orders[pos.model_id] = datetime.now()

                logger.info(f"EXIT [{pos.model_id}] {pos.direction}: {reason}")
                close_result = self.close_position(pos)

                if close_result == 'FLAT':
                    self.position_manager.remove_position(pos.model_id)
                elif close_result and close_result is not False:
                    # Determine exit reason for DB
                    if 'TAKE PROFIT' in reason:
                        exit_reason_db = 'TAKE_PROFIT'
                    elif 'TRAILING STOP' in reason:
                        exit_reason_db = 'TRAILING_STOP'
                    elif 'STOP LOSS' in reason:
                        exit_reason_db = 'STOP_LOSS'
                        self.last_sl_time[pos.direction] = datetime.now()
                        logger.info(f"[SL COOLDOWN] {pos.direction} cooldown {SL_COOLDOWN_SECONDS}s")
                    elif 'PRE-CLOSE' in reason:
                        exit_reason_db = 'PRE_CLOSE'
                    else:
                        exit_reason_db = 'TIMEOUT'

                    pnl_dollar = (pnl_pct / 100) * self.current_price * BTC_CONTRACT_VALUE - 2.02

                    try:
                        log_trade_to_db(
                            bot_type='BTC_V3',
                            model_id=pos.model_id,
                            direction=pos.direction,
                            entry_time=pos.entry_time,
                            entry_price=pos.entry_price,
                            exit_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            exit_price=self.current_price,
                            pnl_pct=pnl_pct,
                            pnl_dollar=pnl_dollar,
                            exit_reason=exit_reason_db,
                            bars_held=bars_elapsed,
                            horizon_bars=TIMEOUT_BARS,
                            model_horizon='v3_ensemble',
                            model_threshold=MIN_CONFIDENCE,
                            entry_probability=pos.entry_confidence
                        )
                    except Exception as e:
                        logger.warning(f"Failed to log trade to DB: {e}")

                    self.position_manager.remove_position(pos.model_id)
                    self.total_trades += 1

    # =========================================================================
    # SIGNAL GENERATION + ENTRY
    # =========================================================================
    def get_signal(self) -> Tuple[Optional[str], float, dict]:
        """Fetch Binance data and run BTCPredictor. Returns (signal, confidence, details)."""
        df = self.get_binance_klines()
        if df.empty or len(df) < 200:
            logger.warning(f"Insufficient Binance data: {len(df)} bars (need 200+)")
            return None, 0.0, {}

        try:
            signal, confidence, details = self.predictor.predict(df)
            return signal, confidence, details
        except Exception as e:
            logger.error(f"Predictor error: {e}")
            return None, 0.0, {}

    def check_entries(self, signal: str, confidence: float, details: dict) -> None:
        """Evaluate signal and execute entry if conditions are met."""
        if signal is None or signal == 'NEUTRAL':
            return

        # Market open check
        if not is_cme_mbt_open():
            logger.debug("Market closed — skipping entry")
            return

        # Block entries near market close
        blocked, block_reason = should_block_entries()
        if blocked:
            logger.info(f"Entry blocked: {block_reason}")
            return

        # Confidence filter
        if confidence < MIN_CONFIDENCE:
            logger.debug(f"Signal {signal} confidence {confidence:.3f} < {MIN_CONFIDENCE} — skip")
            return

        # Position limits
        if signal == 'LONG' and self.position_manager.count_by_direction('LONG') >= MAX_LONG_POSITIONS:
            logger.debug("Max LONG positions reached")
            return
        if signal == 'SHORT' and self.position_manager.count_by_direction('SHORT') >= MAX_SHORT_POSITIONS:
            logger.debug("Max SHORT positions reached")
            return

        # SL cooldown
        sl_elapsed = (datetime.now() - self.last_sl_time[signal]).total_seconds()
        if sl_elapsed < SL_COOLDOWN_SECONDS:
            remaining = SL_COOLDOWN_SECONDS - sl_elapsed
            logger.info(f"SL cooldown: {signal} blocked for {remaining:.0f}s more")
            return

        # Duplicate check: don't enter if already have a position from this signal direction
        model_id = f"v3_{signal.lower()}"
        if self.position_manager.has_position(model_id):
            logger.debug(f"Already have position [{model_id}]")
            return

        # Execute entry
        self._execute_entry(signal, confidence, details, model_id)

    def _execute_entry(self, signal: str, confidence: float, details: dict, model_id: str) -> None:
        """Place order and create position."""
        # Use Binance price as entry price (Binance-only architecture)
        entry_price = self.current_price
        if entry_price is None or entry_price <= 0:
            logger.warning("No current price for entry")
            return

        # Calculate TP/SL from predictor
        tp_price, sl_price = self.predictor.get_tp_sl_prices(entry_price, signal)

        logger.info(f"ENTRY [{model_id}] {signal}: price=${entry_price:.2f}, "
                   f"conf={confidence:.3f}, TP=${tp_price:.2f}, SL=${sl_price:.2f}")
        logger.info(f"  Probabilities: {details.get('probabilities', {})}")

        # Place IB order
        result = self.place_order(signal, self.position_size)
        if result is None:
            logger.error(f"Failed to place {signal} order")
            return

        order_id, ib_fill = result

        # Log IB fill vs Binance for debugging
        if ib_fill and abs(ib_fill - entry_price) / entry_price > 0.002:
            logger.info(f"  IB fill ${ib_fill:.2f} vs Binance ${entry_price:.2f} "
                       f"(basis {(ib_fill/entry_price - 1)*100:+.2f}%)")

        # Create position (using Binance price, not IB fill)
        position = Position(
            symbol='MBT',
            direction=signal,
            size=self.position_size,
            entry_price=entry_price,
            entry_time=datetime.now().isoformat(),
            model_id=model_id,
            target_price=tp_price,
            stop_price=sl_price,
            order_id=order_id,
            peak_price=entry_price if signal == 'LONG' else 0.0,
            trough_price=entry_price if signal == 'SHORT' else 0.0,
            entry_confidence=confidence,
            entry_avg_probability=details.get('avg_probability', 0),
            entry_probabilities=details.get('probabilities', {})
        )

        self.position_manager.add_position(position)
        self.total_signals += 1
        logger.info(f"Position opened [{model_id}]: {signal} {self.position_size} MBT @ ${entry_price:.2f}")

    # =========================================================================
    # SIGNAL LOGGING
    # =========================================================================
    def log_signal(self, signal: str, confidence: float, details: dict) -> None:
        """Log signal to daily CSV."""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            log_path = os.path.join(SIGNAL_LOG_DIR, f"btc_v3_signals_{today}.csv")

            file_exists = os.path.exists(log_path)
            with open(log_path, 'a') as f:
                if not file_exists:
                    f.write("timestamp,price,signal,confidence,avg_prob,prob_2h,prob_4h,prob_6h,positions\n")

                probas = details.get('probabilities', {})
                row = [
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    f"{self.current_price:.2f}" if self.current_price else "0",
                    signal or 'NONE',
                    f"{confidence:.4f}",
                    f"{details.get('avg_probability', 0):.4f}",
                    f"{probas.get('2h', 0):.4f}",
                    f"{probas.get('4h', 0):.4f}",
                    f"{probas.get('6h', 0):.4f}",
                    str(self.position_manager.count_positions())
                ]
                f.write(','.join(row) + '\n')
        except Exception as e:
            logger.debug(f"Signal log error: {e}")

    # =========================================================================
    # IB POSITION SYNC
    # =========================================================================
    def _sync_ib_positions(self) -> None:
        """Validate tracked positions against IB on startup."""
        try:
            self.ib.sleep(2)
            positions = self.ib.positions()
            mbt_pos = next((p for p in positions if p.contract.symbol == 'MBT'), None)
            tracked = self.position_manager.count_positions()
            ib_qty = int(mbt_pos.position) if mbt_pos else 0

            if ib_qty == 0 and tracked > 0:
                logger.warning(f"IB shows 0 MBT but tracking {tracked} — clearing stale positions")
                for pos in self.position_manager.get_all_positions():
                    self.position_manager.remove_position(pos.model_id)
            elif ib_qty != 0 and tracked == 0:
                logger.info(f"IB aggregate: {ib_qty} MBT — other bots own them, not adopting")
            else:
                logger.info(f"IB aggregate: {ib_qty} MBT | This bot: {tracked} tracked positions")
        except Exception as e:
            logger.error(f"IB position sync error: {e}")

    # =========================================================================
    # MAIN LOOP
    # =========================================================================
    def run_iteration(self) -> None:
        """Run one full signal check iteration."""
        self.total_checks += 1

        # Get current price
        price = self.get_realtime_price()
        if price is None:
            logger.warning("No price available")
            return

        self.current_price = price
        self.last_price_time = datetime.now()

        # Determine if we have a new 5-min bar
        now = datetime.now()
        current_bar = now.replace(second=0, microsecond=0)
        current_bar = current_bar.replace(minute=(current_bar.minute // 5) * 5)
        new_bar = self.last_bar_time is None or current_bar > self.last_bar_time

        if new_bar:
            self.last_bar_time = current_bar
            self.position_manager.increment_bars_held()

        # Get signal from predictor
        signal, confidence, details = self.get_signal()

        # Log signal
        if signal is not None:
            self.log_signal(signal, confidence, details)
            action_str = f"{signal} conf={confidence:.3f}" if signal != 'NEUTRAL' else 'NEUTRAL'
            logger.info(f"Signal: {action_str} | Price: ${price:.2f} | "
                       f"Positions: {self.position_manager.count_positions()}/{MAX_POSITIONS}")

            if details.get('probabilities'):
                probas = details['probabilities']
                logger.info(f"  Probabilities: 2h={probas.get('2h', 0):.3f} "
                           f"4h={probas.get('4h', 0):.3f} 6h={probas.get('6h', 0):.3f} "
                           f"avg={details.get('avg_probability', 0):.3f}")

        # Check entries
        if signal:
            self.check_entries(signal, confidence, details)

    def run_price_check(self) -> None:
        """Fast price check for trailing stop / stop loss monitoring."""
        price = self.get_realtime_price()
        if price is None:
            return

        self.current_price = price
        self.last_price_time = datetime.now()

        if self.position_manager.count_positions() > 0:
            self.check_exits()

    def run(self) -> None:
        """Main trading loop with crash recovery."""
        logger.info("Starting BTC V3 Predictor Bot...")
        self.start_websocket()

        while True:
            try:
                if not self.ib.isConnected():
                    if not self.connect():
                        logger.error("IB connect failed. Retrying in 30s...")
                        time.sleep(30)
                        continue

                consecutive_errors = 0
                last_full_iteration = datetime.now()

                time.sleep(3)
                self._sync_ib_positions()

                while True:
                    try:
                        # Connection health
                        if not self.is_connected():
                            logger.warning("IB connection lost — reconnecting...")
                            self.reconnect()
                            consecutive_errors = 0

                        # Fast price check every 2s
                        self.run_price_check()

                        # Full iteration every SIGNAL_CHECK_INTERVAL seconds
                        if (datetime.now() - last_full_iteration).total_seconds() >= SIGNAL_CHECK_INTERVAL:
                            self.run_iteration()
                            last_full_iteration = datetime.now()
                            consecutive_errors = 0

                    except Exception as e:
                        consecutive_errors += 1
                        logger.error(f"Iteration error ({consecutive_errors}/10): {e}")
                        if consecutive_errors >= 10:
                            logger.warning("Too many errors — forcing reconnect...")
                            self.reconnect()
                            consecutive_errors = 0

                    try:
                        time.sleep(PRICE_CHECK_INTERVAL)
                        if self.is_connected():
                            self.ib.sleep(0)
                    except Exception:
                        time.sleep(PRICE_CHECK_INTERVAL)

            except KeyboardInterrupt:
                logger.info("Shutting down...")
                self.stop_websocket()
                self.disconnect()
                self.print_summary()
                return
            except SystemExit as e:
                logger.error(f"SystemExit: {e} — restarting in 30s...")
                try:
                    self.disconnect()
                except Exception:
                    pass
                time.sleep(30)
            except BaseException as e:
                logger.error(f"FATAL: {type(e).__name__}: {e} — restarting in 30s...")
                try:
                    if self.ib.isConnected():
                        self.ib.disconnect()
                    self.connected = False
                except Exception:
                    pass
                time.sleep(30)

    def print_summary(self) -> None:
        """Print session summary."""
        runtime = datetime.now() - self.start_time
        logger.info("=" * 60)
        logger.info("SESSION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Runtime: {runtime}")
        logger.info(f"Total checks: {self.total_checks}")
        logger.info(f"Total signals: {self.total_signals}")
        logger.info(f"Total trades: {self.total_trades}")
        logger.info(f"Open positions: {self.position_manager.count_positions()}")


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='BTC V3 Predictor Trading Bot')
    parser.add_argument('--live', action='store_true', help='Use live trading (default: paper)')
    parser.add_argument('--size', type=int, default=1, help='Position size in MBT contracts')
    parser.add_argument('--test-signal', action='store_true', help='Test signal generation and exit')
    args = parser.parse_args()

    if args.test_signal:
        logger.info("Testing signal generation...")
        bot = BTCV3PredictorBot(paper_trading=True, position_size=1)
        signal, confidence, details = bot.get_signal()
        logger.info(f"Signal: {signal}")
        logger.info(f"Confidence: {confidence:.4f}")
        if details:
            logger.info(f"Avg probability: {details.get('avg_probability', 0):.4f}")
            logger.info(f"Probabilities: {details.get('probabilities', {})}")
            logger.info(f"Price: ${details.get('current_price', 0):,.2f}")
            if signal and signal != 'NEUTRAL':
                tp, sl = bot.predictor.get_tp_sl_prices(details['current_price'], signal)
                logger.info(f"TP: ${tp:,.2f} | SL: ${sl:,.2f}")
        return

    bot = BTCV3PredictorBot(
        paper_trading=not args.live,
        position_size=args.size
    )
    bot.run()


if __name__ == "__main__":
    main()
