"""
BTC Micro-Movement Trading Bot (0.5% TP / 0.15% SL)

Uses separate LONG/SHORT binary classifiers with 33 features (25 V3 + 8 temporal)
Proven performance: 40.3% WR, $18.80/trade on 3,830 trades

Models:
- LONG model: Trained on 2024-2025 bull period
- SHORT model: Trained on full dataset
- Position sizing: 3x-6x based on confidence
- Features: V3 features + hour, day_of_week, prev_close, price_change

Architecture (Binance-only):
- Binance BTCUSDT spot = single source of truth for price, signals, SL/TP
- IB Gateway = execution layer only (place/close orders)
- Position persistence via JSON (survives restarts)
- Trade logging to SQLite via trade_database.py

Model specs:
- Strategy: TP=0.5% / SL=0.15% (3.3:1 R:R, breakeven WR=23.1%)
- Thresholds: LONG ≥0.50, SHORT ≥0.50
- Validated: 40.3% WR, $18.80 avg P&L, $72,015 total (2025-2026, 3,830 trades)
- LONG: 25.4% of trades, 48.7% WR, $34.15 avg
- SHORT: 74.6% of trades, 37.4% WR, $13.57 avg

Position sizing:
- size = (confidence - 0.45) × 15, capped [3, 6]
- Higher confidence → more contracts per trade
- Min 3x (low confidence), Max 6x (high confidence)

IB clientId=420, port 4002 (paper)
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
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, fields
from ib_insync import IB, Future, MarketOrder, util

# Add parent directory
BOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BOT_DIR)

from btc_model_package.predictor import BTCPredictor
from trade_database import log_trade_to_db
from market_hours import is_market_open
from trading_filters import TradingFilters

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
        logging.FileHandler('logs/btc_micro_movement.log', mode='a'),
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
IB_CLIENT_ID = 421  # Unique client ID for micro-movement bot

# BTC MBT contract
BTC_CONTRACT_VALUE = 0.1  # MBT = 0.1 BTC

# Binance
BINANCE_API = "https://api.binance.com/api/v3"
BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@miniTicker"

# Trading parameters (MICRO-MOVEMENT)
TAKE_PROFIT_PCT = 0.5     # 0.5% TP
STOP_LOSS_PCT = 0.15      # 0.15% SL (3.3:1 R:R)
TRAILING_STOP_PCT = 0.05           # Trail distance after activation
TRAILING_STOP_ACTIVATION_PCT = 0.30  # Activate trailing stop after +0.30%

# Model thresholds (calibrated for micro-movements)
LONG_THRESHOLD = 0.50
SHORT_THRESHOLD = 0.50

# Confidence filter floor
MIN_CONFIDENCE = 0.50

# Position sizing parameters (MICRO-MOVEMENT SPECIFIC)
SIZING_BASE = 0.45       # confidence offset
SIZING_FACTOR = 15       # multiplier
MIN_POSITION_SIZE = 3    # minimum contracts (higher for micro-movements)
MAX_POSITION_SIZE = 6    # maximum contracts

# Position limits
MAX_LONG_POSITIONS = 1
MAX_SHORT_POSITIONS = 1
MAX_POSITIONS = MAX_LONG_POSITIONS + MAX_SHORT_POSITIONS

# Timing
SIGNAL_CHECK_INTERVAL = 120  # Check for new signals every 2 minutes
PRICE_CHECK_INTERVAL = 2     # Check price every 2 seconds for TS/SL
BINANCE_DATA_LIMIT = 250     # 250 bars of 5-min data (~20.8 hours)

# SL cooldown
SL_COOLDOWN_SECONDS = 1800  # 30 min after SL before re-entering same direction

# Timeout: close position if held too long (4h = 48 bars of 5-min)
TIMEOUT_BARS = 48  # Shorter timeout for micro-movements

# Signal logging
SIGNAL_LOG_DIR = "signal_logs"

# Position persistence
POSITION_FILE = "btc_micro_positions.json"

# Trade database
DB_PATH = "micro_movement_trades.db"

# JSONL trade log (for monitoring page)
TRADE_LOG_JSONL = "logs/btc_micro_movement_trades.jsonl"


# =============================================================================
# POSITION SIZING (MICRO-MOVEMENT FORMULA)
# =============================================================================
def calculate_position_size(confidence: float) -> int:
    """
    Calculate position size based on model confidence.
    Formula: size = (confidence - 0.45) × 15, capped [3, 6]

    Examples:
    - confidence=0.50: size=3 (min)
    - confidence=0.60: size=5
    - confidence=0.75+: size=6 (max)
    """
    raw_size = (confidence - SIZING_BASE) * SIZING_FACTOR
    capped = max(MIN_POSITION_SIZE, min(MAX_POSITION_SIZE, raw_size))
    return max(3, round(capped))


# =============================================================================
# DATA CLASSES
# =============================================================================
@dataclass
class Position:
    """Represents an open trading position"""
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    entry_time: str
    size: int  # Number of contracts
    confidence: float
    tp_price: float
    sl_price: float
    ts_price: Optional[float] = None
    ts_activated: bool = False
    bars_held: int = 0

    def to_dict(self):
        return asdict(self)


@dataclass
class Trade:
    """Represents a completed trade for logging"""
    direction: str
    entry_price: float
    exit_price: float
    entry_time: str
    exit_time: str
    size: int
    confidence: float
    outcome: str  # 'TP_HIT', 'SL_HIT', 'TS_HIT', 'TIMEOUT', 'MANUAL'
    pnl_dollars: float
    pnl_pct: float
    bars_held: int


# =============================================================================
# MICRO-MOVEMENT PREDICTOR (DUAL BINARY CLASSIFIERS)
# =============================================================================
class MicroMovementPredictor:
    """
    Wrapper for separate LONG/SHORT binary classifiers
    Uses 33 features: 25 V3 + 8 temporal
    """
    def __init__(self):
        models_dir = os.path.join(BOT_DIR, 'models')

        # Load LONG model
        self.model_long = joblib.load(os.path.join(models_dir, 'btc_model_long_xgboost.pkl'))
        self.scaler_long = joblib.load(os.path.join(models_dir, 'btc_scaler_long_xgboost.pkl'))

        # Load SHORT model
        self.model_short = joblib.load(os.path.join(models_dir, 'btc_model_short_xgboost.pkl'))
        self.scaler_short = joblib.load(os.path.join(models_dir, 'btc_scaler_short_xgboost.pkl'))

        # Load feature names
        self.feature_names = joblib.load(os.path.join(models_dir, 'btc_features_xgboost.pkl'))

        # Base V3 predictor for core features
        self.base_predictor = BTCPredictor()

        logger.info("✓ Loaded LONG/SHORT binary classifiers (33 features)")
        logger.info(f"  - LONG model: Trained on 2024-2025 bull period")
        logger.info(f"  - SHORT model: Trained on full dataset")
        logger.info(f"  - Thresholds: LONG≥{LONG_THRESHOLD}, SHORT≥{SHORT_THRESHOLD}")

    def calculate_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate 33 features from 250-bar window:
        - 25 V3 features (from base_predictor)
        - 8 temporal features (hour, day_of_week, prev_close, price_change)
        """
        # Add temporal columns if not present
        if 'hour' not in df.columns:
            df['hour'] = df.index.hour
        if 'day_of_week' not in df.columns:
            df['day_of_week'] = df.index.dayofweek
        if 'prev_close_1' not in df.columns:
            df['prev_close_1'] = df['close'].shift(1)
        if 'prev_close_5' not in df.columns:
            df['prev_close_5'] = df['close'].shift(5)
        if 'prev_close_20' not in df.columns:
            df['prev_close_20'] = df['close'].shift(20)
        if 'price_change_1' not in df.columns:
            df['price_change_1'] = df['close'].pct_change(1)
        if 'price_change_5' not in df.columns:
            df['price_change_5'] = df['close'].pct_change(5)
        if 'price_change_20' not in df.columns:
            df['price_change_20'] = df['close'].pct_change(20)

        # Get V3 features
        v3_features = self.base_predictor.calculate_features(df)

        # Get temporal features from last row
        last_row = df.iloc[-1]
        temporal_features = {
            'hour': last_row['hour'],
            'day_of_week': last_row['day_of_week'],
            'prev_close_1': last_row['prev_close_1'],
            'prev_close_5': last_row['prev_close_5'],
            'prev_close_20': last_row['prev_close_20'],
            'price_change_1': last_row['price_change_1'],
            'price_change_5': last_row['price_change_5'],
            'price_change_20': last_row['price_change_20']
        }

        # Combine
        v3_features.update(temporal_features)

        # Convert to array in correct order
        features_array = np.array([[v3_features.get(f, 0) for f in self.feature_names]])
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)

        return features_array

    def predict(self, df: pd.DataFrame) -> Tuple[str, float, dict]:
        """
        Predict signal using dual binary classifiers.
        Returns: (signal, confidence, details)

        Logic:
        1. Run both LONG and SHORT models
        2. Take highest confidence signal above threshold
        3. If both below threshold, return NEUTRAL
        """
        features = self.calculate_features(df)

        # Scale for each model
        features_long = self.scaler_long.transform(features)
        features_short = self.scaler_short.transform(features)

        # Get probabilities
        proba_long = self.model_long.predict_proba(features_long)[0]
        proba_short = self.model_short.predict_proba(features_short)[0]

        # Binary classifiers: class 1 = LONG/SHORT
        long_confidence = proba_long[1]
        short_confidence = proba_short[1]

        # Decision logic: take highest confidence signal above threshold
        if long_confidence >= LONG_THRESHOLD and long_confidence > short_confidence:
            signal = 'LONG'
            confidence = long_confidence
        elif short_confidence >= SHORT_THRESHOLD:
            signal = 'SHORT'
            confidence = short_confidence
        else:
            signal = 'NEUTRAL'
            confidence = max(long_confidence, short_confidence)

        details = {
            'long_confidence': long_confidence,
            'short_confidence': short_confidence,
            'signal': signal,
            'confidence': confidence
        }

        return signal, confidence, details


# =============================================================================
# BINANCE DATA FETCHER
# =============================================================================
class BinanceDataFetcher:
    """Fetches 5-min OHLCV data from Binance API"""

    def __init__(self, symbol='BTCUSDT', interval='5m'):
        self.symbol = symbol
        self.interval = interval
        self.last_price = None
        self.ws = None
        self.ws_thread = None

        if WEBSOCKET_AVAILABLE:
            self._start_websocket()

    def _start_websocket(self):
        """Start websocket for real-time price updates"""
        def on_message(ws, message):
            data = json.loads(message)
            self.last_price = float(data['c'])

        def on_error(ws, error):
            logger.warning(f"WebSocket error: {error}")

        def on_close(ws, close_status_code, close_msg):
            logger.info("WebSocket closed")

        def run():
            self.ws = websocket.WebSocketApp(
                BINANCE_WS_URL,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            self.ws.run_forever()

        self.ws_thread = threading.Thread(target=run, daemon=True)
        self.ws_thread.start()
        logger.info("✓ Started Binance WebSocket for real-time prices")

    def get_klines(self, limit=250) -> pd.DataFrame:
        """Fetch historical 5-min klines from Binance"""
        url = f"{BINANCE_API}/klines"
        params = {
            'symbol': self.symbol,
            'interval': self.interval,
            'limit': limit
        }

        response = requests.get(url, params=params)
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

        return df[['open', 'high', 'low', 'close', 'volume']]

    def get_current_price(self) -> float:
        """Get current BTC price (WebSocket if available, else API)"""
        if self.last_price is not None:
            return self.last_price

        # Fallback to API
        url = f"{BINANCE_API}/ticker/price"
        params = {'symbol': self.symbol}
        response = requests.get(url, params=params)
        response.raise_for_status()
        return float(response.json()['price'])


# =============================================================================
# TRADING BOT
# =============================================================================
class MicroMovementBot:
    def __init__(self, paper_mode=True):
        self.paper_mode = paper_mode
        self.ib = IB()
        self.binance = BinanceDataFetcher()
        self.predictor = MicroMovementPredictor()

        self.position: Optional[Position] = None
        self.last_sl_time: Dict[str, datetime] = {}

        self.running = False
        self.signal_thread = None
        self.price_thread = None

        # Initialize trading filters
        self.filters = TradingFilters(TAKE_PROFIT_PCT, STOP_LOSS_PCT)

        # Load persisted position
        self._load_position()

        logger.info("="*80)
        logger.info("BTC MICRO-MOVEMENT BOT INITIALIZED")
        logger.info("="*80)
        logger.info(f"Mode: {'PAPER' if paper_mode else 'LIVE'}")
        logger.info(f"Strategy: TP={TAKE_PROFIT_PCT}% / SL={STOP_LOSS_PCT}% (3.3:1 R:R)")
        logger.info(f"Position sizing: {MIN_POSITION_SIZE}x-{MAX_POSITION_SIZE}x based on confidence")
        logger.info(f"Thresholds: LONG≥{LONG_THRESHOLD}, SHORT≥{SHORT_THRESHOLD}")
        logger.info(f"Timeout: {TIMEOUT_BARS} bars (4 hours)")
        logger.info("✓ All filters active")
        logger.info("="*80)

    def _load_position(self):
        """Load persisted position from JSON"""
        if os.path.exists(POSITION_FILE):
            try:
                with open(POSITION_FILE, 'r') as f:
                    data = json.load(f)
                    self.position = Position(**data)
                    logger.info(f"✓ Loaded persisted {self.position.direction} position")
            except Exception as e:
                logger.error(f"Failed to load position: {e}")

    def _save_position(self):
        """Persist position to JSON"""
        try:
            if self.position:
                # Convert numpy types to Python types for JSON serialization
                pos_dict = self.position.to_dict()
                pos_dict['confidence'] = float(pos_dict['confidence'])
                pos_dict['entry_price'] = float(pos_dict['entry_price'])
                pos_dict['tp_price'] = float(pos_dict['tp_price'])
                pos_dict['sl_price'] = float(pos_dict['sl_price'])
                if pos_dict['ts_price']:
                    pos_dict['ts_price'] = float(pos_dict['ts_price'])

                with open(POSITION_FILE, 'w') as f:
                    json.dump(pos_dict, f, indent=2)
            else:
                if os.path.exists(POSITION_FILE):
                    os.remove(POSITION_FILE)
        except Exception as e:
            logger.error(f"Failed to save position: {e}")

    def connect_ib(self):
        """Connect to IB Gateway"""
        try:
            self.ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID)
            logger.info(f"✓ Connected to IB Gateway (clientId={IB_CLIENT_ID})")
        except Exception as e:
            logger.error(f"Failed to connect to IB Gateway: {e}")
            raise

    def disconnect_ib(self):
        """Disconnect from IB Gateway"""
        if self.ib.isConnected():
            self.ib.disconnect()
            logger.info("✓ Disconnected from IB Gateway")

    def get_btc_contract(self):
        """Get BTC Micro futures contract"""
        contract = Future('MBT', '202506', 'CME')
        self.ib.qualifyContracts(contract)
        return contract

    def start(self):
        """Start the trading bot"""
        self.running = True

        # Connect to IB
        self.connect_ib()

        # Start signal checking thread
        self.signal_thread = threading.Thread(target=self._signal_loop, daemon=True)
        self.signal_thread.start()

        # Start price checking thread
        self.price_thread = threading.Thread(target=self._price_loop, daemon=True)
        self.price_thread.start()

        logger.info("✓ Bot started - monitoring for signals...")

        # Keep main thread alive
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("\nShutdown signal received...")
            self.stop()

    def stop(self):
        """Stop the trading bot"""
        self.running = False
        time.sleep(3)  # Allow threads to finish
        self.disconnect_ib()
        logger.info("✓ Bot stopped")

    def _signal_loop(self):
        """Check for new trading signals every 2 minutes"""
        while self.running:
            try:
                # Only check signals if no position open
                if self.position is None:
                    self._check_signal()
                time.sleep(SIGNAL_CHECK_INTERVAL)
            except Exception as e:
                logger.error(f"Error in signal loop: {e}")
                time.sleep(SIGNAL_CHECK_INTERVAL)

    def _price_loop(self):
        """Check price every 2 seconds for SL/TP/TS"""
        while self.running:
            try:
                if self.position:
                    self._check_exit_conditions()
                time.sleep(PRICE_CHECK_INTERVAL)
            except Exception as e:
                logger.error(f"Error in price loop: {e}")
                time.sleep(PRICE_CHECK_INTERVAL)

    def _check_signal(self):
        """Check for new trading signal"""
        try:
            # Fetch 250 bars of 5-min data
            df = self.binance.get_klines(limit=BINANCE_DATA_LIMIT)

            if len(df) < 250:
                logger.warning(f"Insufficient data: {len(df)} bars (need 250)")
                return

            # Get prediction
            signal, confidence, details = self.predictor.predict(df)

            logger.info(f"Signal check: {signal} (conf={confidence:.3f}, "
                       f"L={details['long_confidence']:.3f}, S={details['short_confidence']:.3f})")

            # Check if signal meets criteria
            if signal == 'NEUTRAL':
                return

            if confidence < MIN_CONFIDENCE:
                logger.info(f"  Confidence {confidence:.3f} < {MIN_CONFIDENCE} threshold")
                return

            # Check SL cooldown
            if signal in self.last_sl_time:
                elapsed = (datetime.now() - self.last_sl_time[signal]).total_seconds()
                if elapsed < SL_COOLDOWN_SECONDS:
                    remaining = int(SL_COOLDOWN_SECONDS - elapsed)
                    logger.info(f"  {signal} in SL cooldown ({remaining}s remaining)")
                    return

            # Check market hours
            if not is_market_open():
                logger.info(f"  CME closed - skipping {signal} signal")
                return

            # Run all filters before entry
            current_price = self.binance.get_current_price()
            size = calculate_position_size(confidence)
            filter_pass, filter_reason = self.filters.check_all_filters(
                signal, confidence, current_price, df, size
            )

            if not filter_pass:
                logger.info(f"  Signal filtered: {signal} @ {confidence:.3f} - {filter_reason}")
                return

            # Open position
            self._open_position(signal, confidence, df)

        except Exception as e:
            logger.error(f"Error checking signal: {e}")

    def _open_position(self, direction: str, confidence: float, df: pd.DataFrame):
        """Open a new position"""
        try:
            entry_price = self.binance.get_current_price()
            size = calculate_position_size(confidence)

            # Calculate TP/SL prices
            if direction == 'LONG':
                tp_price = entry_price * (1 + TAKE_PROFIT_PCT / 100)
                sl_price = entry_price * (1 - STOP_LOSS_PCT / 100)
            else:
                tp_price = entry_price * (1 - TAKE_PROFIT_PCT / 100)
                sl_price = entry_price * (1 + STOP_LOSS_PCT / 100)

            # Create position
            self.position = Position(
                direction=direction,
                entry_price=entry_price,
                entry_time=datetime.now().isoformat(),
                size=size,
                confidence=confidence,
                tp_price=tp_price,
                sl_price=sl_price,
                bars_held=0
            )

            # Log entry to JSONL
            self._log_entry_to_jsonl(direction, entry_price, tp_price, sl_price, size, confidence)

            # Execute on IB
            if not self.paper_mode:
                contract = self.get_btc_contract()
                action = 'BUY' if direction == 'LONG' else 'SELL'
                order = MarketOrder(action, size)
                trade = self.ib.placeOrder(contract, order)
                self.ib.sleep(2)
                logger.info(f"  IB order placed: {action} {size}x MBT @ ${entry_price:,.2f}")

            # Persist
            self._save_position()

            # Record trade attempt for filter daily limit
            self.filters.record_trade_attempt()

            logger.info(f"✓ OPENED {direction} @ ${entry_price:,.2f} | "
                       f"Size: {size}x | Conf: {confidence:.3f} | "
                       f"TP: ${tp_price:,.2f} | SL: ${sl_price:,.2f}")

        except Exception as e:
            logger.error(f"Failed to open {direction} position: {e}")
            self.position = None

    def _check_exit_conditions(self):
        """Check if position should be closed (TP/SL/TS/Timeout)"""
        if not self.position:
            return

        try:
            current_price = self.binance.get_current_price()
            self.position.bars_held += 1

            direction = self.position.direction
            entry_price = self.position.entry_price

            # Calculate unrealized P&L
            if direction == 'LONG':
                pnl_pct = (current_price - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - current_price) / entry_price * 100

            # Check TP
            if direction == 'LONG' and current_price >= self.position.tp_price:
                self._close_position('TP_HIT', current_price)
                return
            elif direction == 'SHORT' and current_price <= self.position.tp_price:
                self._close_position('TP_HIT', current_price)
                return

            # Check SL
            if direction == 'LONG' and current_price <= self.position.sl_price:
                self._close_position('SL_HIT', current_price)
                return
            elif direction == 'SHORT' and current_price >= self.position.sl_price:
                self._close_position('SL_HIT', current_price)
                return

            # Check trailing stop activation
            if not self.position.ts_activated:
                if direction == 'LONG':
                    activation_threshold = entry_price * (1 + TRAILING_STOP_ACTIVATION_PCT / 100)
                    if current_price >= activation_threshold:
                        self.position.ts_activated = True
                        self.position.ts_price = current_price * (1 - TRAILING_STOP_PCT / 100)
                        logger.info(f"  TS activated @ ${current_price:,.2f} | TS: ${self.position.ts_price:,.2f}")
                else:  # SHORT
                    activation_threshold = entry_price * (1 - TRAILING_STOP_ACTIVATION_PCT / 100)
                    if current_price <= activation_threshold:
                        self.position.ts_activated = True
                        self.position.ts_price = current_price * (1 + TRAILING_STOP_PCT / 100)
                        logger.info(f"  TS activated @ ${current_price:,.2f} | TS: ${self.position.ts_price:,.2f}")

            # Update trailing stop
            if self.position.ts_activated:
                if direction == 'LONG':
                    new_ts = current_price * (1 - TRAILING_STOP_PCT / 100)
                    if new_ts > self.position.ts_price:
                        self.position.ts_price = new_ts
                    if current_price <= self.position.ts_price:
                        self._close_position('TS_HIT', current_price)
                        return
                else:
                    new_ts = current_price * (1 + TRAILING_STOP_PCT / 100)
                    if new_ts < self.position.ts_price:
                        self.position.ts_price = new_ts
                    if current_price >= self.position.ts_price:
                        self._close_position('TS_HIT', current_price)
                        return

            # Check timeout
            if self.position.bars_held >= TIMEOUT_BARS:
                logger.info(f"  Timeout: {self.position.bars_held} bars")
                self._close_position('TIMEOUT', current_price)
                return

            # Periodic status
            if self.position.bars_held % 10 == 0:
                logger.info(f"  {direction} | Bars: {self.position.bars_held} | "
                           f"P&L: {pnl_pct:+.2f}% | Price: ${current_price:,.2f}")

        except Exception as e:
            logger.error(f"Error checking exit conditions: {e}")

    def _close_position(self, outcome: str, exit_price: float):
        """Close the current position"""
        if not self.position:
            return

        try:
            direction = self.position.direction
            entry_price = self.position.entry_price
            size = self.position.size

            # Calculate P&L
            if direction == 'LONG':
                pnl_pct = (exit_price - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - exit_price) / entry_price * 100

            notional = entry_price * BTC_CONTRACT_VALUE * size
            pnl_dollars = (pnl_pct / 100) * notional

            # Execute close on IB
            if not self.paper_mode:
                contract = self.get_btc_contract()
                action = 'SELL' if direction == 'LONG' else 'BUY'
                order = MarketOrder(action, size)
                trade = self.ib.placeOrder(contract, order)
                self.ib.sleep(2)
                logger.info(f"  IB order placed: {action} {size}x MBT @ ${exit_price:,.2f}")

            # Log trade
            trade = Trade(
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_price,
                entry_time=self.position.entry_time,
                exit_time=datetime.now().isoformat(),
                size=size,
                confidence=self.position.confidence,
                outcome=outcome,
                pnl_dollars=pnl_dollars,
                pnl_pct=pnl_pct,
                bars_held=self.position.bars_held
            )

            log_trade_to_db(
                bot_type='BTC_MICRO',
                model_id='micro_0.5pct',
                direction=trade.direction,
                entry_time=trade.entry_time,
                entry_price=trade.entry_price,
                exit_time=trade.exit_time,
                exit_price=trade.exit_price,
                pnl_pct=trade.pnl_pct,
                pnl_dollar=trade.pnl_dollars,
                exit_reason=trade.outcome,
                bars_held=trade.bars_held,
                horizon_bars=48,  # 4 hours
                model_horizon='4h',
                model_threshold=0.50,
                entry_probability=trade.confidence
            )

            # Log to JSONL for monitoring page
            self._log_exit_to_jsonl(trade)

            logger.info(f"✓ CLOSED {direction} @ ${exit_price:,.2f} | "
                       f"Outcome: {outcome} | P&L: ${pnl_dollars:+.2f} ({pnl_pct:+.2f}%) | "
                       f"Bars: {self.position.bars_held}")

            # Update filter state with trade result
            self.filters.record_trade_result(pnl_dollars)

            # Update SL cooldown
            if outcome == 'SL_HIT':
                self.last_sl_time[direction] = datetime.now()

            # Clear position
            self.position = None
            self._save_position()

        except Exception as e:
            logger.error(f"Failed to close {direction} position: {e}")

    def _log_entry_to_jsonl(self, direction: str, entry_price: float, tp_price: float, sl_price: float, size: int, confidence: float):
        """Log entry signal to JSONL for monitoring page"""
        try:
            os.makedirs('logs', exist_ok=True)
            entry = {
                'timestamp': datetime.now().isoformat(),
                'event': 'ENTRY_EXECUTED',
                'data': {
                    'direction': direction,
                    'entry_price': float(entry_price),
                    'tp_price': float(tp_price),
                    'sl_price': float(sl_price),
                    'position_size': int(size),
                    'confidence': float(confidence)
                }
            }
            with open(TRADE_LOG_JSONL, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to log entry to JSONL: {e}")

    def _log_exit_to_jsonl(self, trade: Trade):
        """Log exit to JSONL for monitoring page"""
        try:
            exit_entry = {
                'timestamp': datetime.now().isoformat(),
                'event': 'EXIT',
                'data': {
                    'direction': trade.direction,
                    'exit_reason': trade.outcome,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'entry_time': trade.entry_time,
                    'exit_time': trade.exit_time,
                    'position_size': float(trade.size),
                    'hold_minutes': (trade.bars_held * 5),  # 5-min bars
                    'pnl_pct': trade.pnl_pct,
                    'pnl_dollar': trade.pnl_dollars
                }
            }
            with open(TRADE_LOG_JSONL, 'a') as f:
                f.write(json.dumps(exit_entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to log exit to JSONL: {e}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='BTC Micro-Movement Trading Bot')
    parser.add_argument('--live', action='store_true', help='Run in LIVE mode (default: PAPER)')
    args = parser.parse_args()

    paper_mode = not args.live

    bot = MicroMovementBot(paper_mode=paper_mode)
    bot.start()


if __name__ == '__main__':
    main()
