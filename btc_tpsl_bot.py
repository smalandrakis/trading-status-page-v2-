#!/usr/bin/env python3
"""
BTC TP/SL Bot v4 — LONG-only, direct TP/SL label model, TP=1.0% / SL=0.5%.

Model predicts "will LONG TP=+1.0% hit before SL=-0.5% within 120 bars?"
Trained on 12 months of 1-min Binance data (525K candles).

Strategies (LONG-only, each trades independently):
  M1_long_hgb70 : HGB P(TP) > 70%  (92T/3mo, 81.5% WR, +0.723% EV)
  M2_long_hgb60 : HGB P(TP) > 60%  (283T/3mo, 62.2% WR, +0.433% EV)

Uses Binance WebSocket for ticks, compute 1-min bar features (75),
and execute via IB Gateway on MBT (Micro Bitcoin Futures).
Exit: pure TP/SL (no trailing stop).
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
import time
import logging
import threading
import sqlite3
import argparse
from datetime import datetime, timedelta
from typing import List, Optional
from dataclasses import dataclass
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
BOT_NAME = "BTC_TPSL"
IB_HOST = '127.0.0.1'
IB_PORT = 4002
IB_CLIENT_ID = 403

# Contract
MBT_EXPIRY = '20260424'
BTC_CONTRACT_VALUE = 0.1  # MBT = 0.1 BTC

# Position management
MAX_POSITIONS = 2          # max 1 per strategy
POSITION_SIZE = 1          # contracts per trade
MIN_ENTRY_GAP_SEC = 300    # 5 min between entries per strategy
SL_COOLDOWN_SEC = 600      # 10 min cooldown after SL per strategy

# TP / SL
TP_PCT = 1.0   # take profit %
SL_PCT = 0.5   # stop loss %

# Signal check interval
SIGNAL_CHECK_INTERVAL = 60  # seconds (one 1-min bar)

# HGB v4 model (LONG-only, direct TP/SL labels, 12mo 1-min data)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'tpsl_v4', 'best_model.pkl')
FEATURE_COLS_PATH = os.path.join(BASE_DIR, 'models', 'tpsl_v4', 'feature_cols.pkl')
CONFIG_PATH = os.path.join(BASE_DIR, 'models', 'tpsl_v4', 'config.pkl')

# Logging
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
SIGNAL_LOG_DIR = os.path.join(BASE_DIR, 'signal_logs')
os.makedirs(SIGNAL_LOG_DIR, exist_ok=True)

DB_PATH = os.path.join(BASE_DIR, 'tpsl_trades.db')

logger = logging.getLogger('btc_tpsl_bot')
logger.setLevel(logging.INFO)
fh = logging.FileHandler(os.path.join(LOG_DIR, 'btc_tpsl_bot.log'))
fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(fh)


# =============================================================================
# DATA STRUCTURES
# =============================================================================
@dataclass
class Position:
    model_id: str       # M1_long_hgb70, M2_long_hgb60
    direction: str      # LONG or SHORT
    entry_price: float
    entry_time: datetime
    size: int
    tp_price: float
    sl_price: float
    order_id: Optional[int] = None
    entry_probability: float = 0.0


# =============================================================================
# FEATURE COMPUTATION — matches train_tpsl_v3.py exactly (75 features on 1-min OHLCV)
# =============================================================================
def compute_features_v3(bars):
    """Compute 75 features from 1-min OHLCV bars. Must match train_tpsl_v3.py."""
    c = bars['close']
    h = bars['high']
    l = bars['low']
    v = bars['volume']
    r1 = c.pct_change()

    feat = pd.DataFrame(index=bars.index)

    # Returns
    for w in [1, 2, 3, 5, 10, 15, 30, 60, 120, 240]:
        feat[f'ret_{w}'] = c.pct_change(w) * 100

    # Volatility (rolling std of 1-min returns)
    for w in [5, 10, 15, 30, 60, 120, 240]:
        feat[f'vol_{w}'] = r1.rolling(w).std() * 100

    # Vol ratios (short/long)
    feat['vol_ratio_5_60'] = feat['vol_5'] / feat['vol_60'].replace(0, np.nan)
    feat['vol_ratio_15_60'] = feat['vol_15'] / feat['vol_60'].replace(0, np.nan)
    feat['vol_ratio_30_120'] = feat['vol_30'] / feat['vol_120'].replace(0, np.nan)
    feat['vol_ratio_60_240'] = feat['vol_60'] / feat['vol_240'].replace(0, np.nan)

    # Volume features
    for w in [5, 15, 30, 60, 120]:
        feat[f'vratio_{w}'] = v / v.rolling(w).mean().replace(0, np.nan)

    # RSI at multiple periods
    for p in [7, 14, 30, 60]:
        delta = c.diff()
        gain = delta.clip(lower=0).ewm(span=p).mean()
        loss = (-delta.clip(upper=0)).ewm(span=p).mean()
        feat[f'rsi_{p}'] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    # Bollinger %B
    for w in [20, 60, 120]:
        ma = c.rolling(w).mean()
        std = c.rolling(w).std()
        feat[f'bb_pct_{w}'] = (c - (ma - 2*std)) / (4*std).replace(0, np.nan)

    # ATR %
    for w in [14, 30, 60, 120]:
        tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        feat[f'atr_pct_{w}'] = tr.rolling(w).mean() / c * 100

    # MACD variants
    for fast, slow in [(12,26), (5,15), (30,60)]:
        ema_f = c.ewm(span=fast).mean()
        ema_s = c.ewm(span=slow).mean()
        diff = ema_f - ema_s
        sig = diff.ewm(span=9).mean()
        feat[f'macd_{fast}_{slow}'] = diff / c * 100
        feat[f'macd_hist_{fast}_{slow}'] = (diff - sig) / c * 100

    # Price position
    for w in [60, 120, 240, 480]:
        rh = h.rolling(w).max()
        rl = l.rolling(w).min()
        feat[f'pos_high_{w}'] = (c - rh) / c * 100
        feat[f'pos_low_{w}'] = (c - rl) / c * 100
        feat[f'pos_range_{w}'] = (c - rl) / (rh - rl).replace(0, np.nan)

    # Bar microstructure
    feat['bar_range'] = (h - l) / c * 100
    feat['bar_body'] = (c - bars['open']).abs() / c * 100
    feat['upper_wick'] = (h - c.clip(lower=bars['open'])) / c * 100
    feat['lower_wick'] = (c.clip(upper=bars['open']) - l) / c * 100
    feat['bar_range_ma30'] = feat['bar_range'].rolling(30).mean()
    feat['bar_range_ratio'] = feat['bar_range'] / feat['bar_range_ma30'].replace(0, np.nan)

    # Trend strength
    for w in [30, 60, 120]:
        ma = c.rolling(w).mean()
        feat[f'dist_ma_{w}'] = (c - ma) / ma * 100

    # Volume-weighted features
    feat['vol_price_corr_30'] = r1.rolling(30).corr(v.pct_change())
    feat['vol_price_corr_60'] = r1.rolling(60).corr(v.pct_change())

    # Momentum
    feat['roc_5'] = c.pct_change(5) * 100
    feat['roc_15'] = c.pct_change(15) * 100
    feat['roc_60'] = c.pct_change(60) * 100

    # Up/down count
    feat['up_count_10'] = sum((r1.shift(i) > 0).astype(int) for i in range(10))
    feat['down_count_10'] = 10 - feat['up_count_10']

    # Cyclical time
    hour = bars.index.hour + bars.index.minute / 60
    feat['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    dow = bars.index.dayofweek
    feat['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    feat['dow_cos'] = np.cos(2 * np.pi * dow / 7)

    return feat.replace([np.inf, -np.inf], np.nan)


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
            entry_probability REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS open_positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT, direction TEXT,
            entry_time TEXT, entry_price REAL,
            tp_price REAL, sl_price REAL,
            entry_probability REAL
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
# BOT CLASS
# =============================================================================
class BTCTPSLBot:
    def __init__(self, paper_trading=True):
        self.paper_trading = paper_trading
        self.ib = IB()
        self.contract = None
        self.positions: List[Position] = []
        self.current_price = None
        self.current_price_time = None

        # IB MBT real-time ticker for SL/TP monitoring (Mar 30, 2026 fix)
        # Entry uses IB fill price — monitoring must use same source
        self.ib_mbt_ticker = None

        # Tick data (price + volume from Binance WS)
        self.tick_prices = deque(maxlen=200000)
        self.tick_volumes = deque(maxlen=200000)
        self.tick_times = deque(maxlen=200000)

        # WebSocket
        self.ws = None
        self.ws_thread = None
        self.ws_connected = False

        # Timing per strategy
        self.last_entry = {}      # model_id -> datetime
        self.last_sl_exit = {}    # model_id -> datetime

        # Signal log
        self.signal_csv_path = None
        self.signal_csv_date = None
        self.last_signal_check = None
        self.last_signal_log_time = None

        # Load HGB v3 model
        self.model = None
        self.feature_cols = None
        self.vol_60_median = None
        self._load_model()

        # Init DB
        init_db(DB_PATH)

        logger.info("=" * 60)
        logger.info("BTC TP/SL MULTI-STRATEGY BOT INITIALIZED")
        logger.info("=" * 60)
        logger.info("Strategies: M1_long_hgb70, M2_long_hgb60 (LONG-only)")
        logger.info("TP=%.1f%% SL=%.1f%% (R:R = %.1f:1)" % (TP_PCT, SL_PCT, TP_PCT / SL_PCT))
        logger.info("HGB v4 model: %s" % ("loaded" if self.model else "MISSING"))
        logger.info("vol_60_median: %.6f" % (self.vol_60_median or 0))
        logger.info("Paper: %s" % paper_trading)

    def _load_model(self):
        try:
            import pickle
            with open(MODEL_PATH, 'rb') as f:
                self.model = pickle.load(f)
            with open(FEATURE_COLS_PATH, 'rb') as f:
                self.feature_cols = pickle.load(f)
            with open(CONFIG_PATH, 'rb') as f:
                config = pickle.load(f)
            self.vol_60_median = config.get('vol_60_median', 0.04)
            logger.info("Loaded HGB v4 model (%d features, vol_60_median=%.6f)" % (
                len(self.feature_cols), self.vol_60_median))
        except Exception as e:
            logger.error("Failed to load HGB v4 model: %s" % e)

    # =================================================================
    # IB CONNECTION
    # =================================================================
    def connect_ib(self):
        for attempt in range(999):
            try:
                try:
                    if self.ib.isConnected():
                        self.ib.disconnect()
                        time.sleep(2)
                except Exception:
                    pass
                self.ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID, timeout=20)
                logger.info("Connected to IB (port=%d, clientId=%d)" % (IB_PORT, IB_CLIENT_ID))
                # Type 4: real-time streaming on paper accounts (type 1 needs paid CME sub)
                self.ib.reqMarketDataType(4)
                self.contract = Future(
                    symbol='MBT',
                    lastTradeDateOrContractMonth=MBT_EXPIRY,
                    exchange='CME', currency='USD', multiplier='0.1'
                )
                self.ib.qualifyContracts(self.contract)
                logger.info("Contract: %s" % self.contract)
                # Subscribe to IB MBT real-time ticker for SL/TP monitoring (Mar 30 fix)
                try:
                    self.ib_mbt_ticker = self.ib.reqMktData(self.contract, '', False, False)
                    self.ib.sleep(3)
                    mp = self.ib_mbt_ticker.marketPrice()
                    if mp and mp > 10000:
                        logger.info("IB MBT ticker active: $%.2f (for SL/TP monitoring)" % mp)
                    else:
                        logger.warning("IB MBT ticker no price yet (mp=%s)" % mp)
                except Exception as e:
                    logger.warning("IB MBT ticker failed: %s" % e)
                return True
            except Exception as e:
                logger.error("IB attempt %d failed: %s" % (attempt + 1, e))
                time.sleep(5)
        return False

    def reconnect_ib(self):
        logger.info("Reconnecting to IB...")
        try:
            if self.ib.isConnected():
                self.ib.disconnect()
            time.sleep(2)
        except:
            pass
        self.ib = IB()
        return self.connect_ib()

    # =================================================================
    # WEBSOCKET PRICE FEED
    # =================================================================
    def start_websocket(self):
        if not WEBSOCKET_AVAILABLE:
            logger.warning("websocket-client not installed")
            return

        def on_message(ws, message):
            try:
                data = json.loads(message)
                if 'c' in data:
                    price = float(data['c'])
                    vol = float(data.get('v', 0))  # 24h volume from miniTicker
                    now = datetime.now()
                    self.current_price = price
                    self.current_price_time = now
                    self.tick_prices.append(price)
                    self.tick_volumes.append(vol)
                    self.tick_times.append(now)
            except Exception as e:
                logger.debug("WS message error: %s" % e)

        def on_error(ws, error):
            logger.warning("WS error: %s" % error)
            self.ws_connected = False

        def on_close(ws, code, msg):
            logger.info("WS closed: %s %s" % (code, msg))
            self.ws_connected = False

        def on_open(ws):
            logger.info("WebSocket connected")
            self.ws_connected = True

        def run_ws():
            while True:
                try:
                    self.ws = websocket.WebSocketApp(
                        "wss://stream.binance.com:9443/ws/btcusdt@miniTicker",
                        on_message=on_message, on_error=on_error,
                        on_close=on_close, on_open=on_open
                    )
                    self.ws.run_forever()
                except Exception as e:
                    logger.error("WS thread error: %s" % e)
                if not self.ws_connected:
                    logger.info("WS reconnecting in 5s...")
                    time.sleep(5)

        self.ws_thread = threading.Thread(target=run_ws, daemon=True)
        self.ws_thread.start()

    def load_historical_ticks(self):
        tick_file = os.path.join(LOG_DIR, 'btc_price_ticks.csv')
        if os.path.exists(tick_file):
            try:
                df = pd.read_csv(tick_file, parse_dates=['timestamp'])
                df = df.sort_values('timestamp')
                cutoff = datetime.now() - timedelta(days=3)
                df = df[df['timestamp'] > cutoff]
                for _, row in df.iterrows():
                    self.tick_prices.append(row['price'])
                    self.tick_volumes.append(row.get('volume', 0) if 'volume' in df.columns else 0)
                    self.tick_times.append(row['timestamp'].to_pydatetime())
                logger.info("Loaded %d historical ticks for warmup" % len(df))
            except Exception as e:
                logger.error("Failed to load historical ticks: %s" % e)

    # =================================================================
    # FEATURE COMPUTATION
    # =================================================================
    def get_features(self):
        if len(self.tick_prices) < 500:
            return None

        tick_df = pd.DataFrame({
            'timestamp': list(self.tick_times),
            'price': list(self.tick_prices),
            'volume': list(self.tick_volumes),
        }).set_index('timestamp').sort_index()

        # Resample to 1-min OHLCV bars
        bars = tick_df['price'].resample('1min').agg(
            **{'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
        )
        bars['volume'] = tick_df['volume'].resample('1min').last().fillna(0)
        bars = bars.dropna(subset=['close'])

        if len(bars) < 500:
            return None

        feat = compute_features_v3(bars)
        return feat

    # =================================================================
    # ENTRY LOGIC
    # =================================================================
    def check_entry(self, feat):
        if feat is None or len(feat) == 0 or self.current_price is None:
            return

        now = datetime.now()
        last_row = feat.iloc[-1]
        price = self.current_price

        # Get vol_60 for logging
        vol_60 = last_row.get('vol_60', 0)
        if pd.isna(vol_60):
            vol_60 = 0

        # Get HGB probability (predicts: will LONG TP hit before SL within 120 bars?)
        hgb_prob = None
        if self.model is not None and self.feature_cols is not None:
            try:
                row_data = last_row[self.feature_cols].fillna(0).values.reshape(1, -1)
                hgb_prob = self.model.predict_proba(row_data)[0][1]
            except Exception as e:
                logger.debug("HGB predict error: %s" % e)

        # --- Strategy M1: LONG when HGB P(TP) > 70% (81.5% WR, +0.723% EV) ---
        if hgb_prob is not None and hgb_prob > 0.70:
            self._try_entry('M1_long_hgb70', now, price,
                            condition=True, direction='LONG', probability=hgb_prob)

        # --- Strategy M2: LONG when HGB P(TP) > 60% (62.2% WR, +0.433% EV) ---
        if hgb_prob is not None and hgb_prob > 0.60:
            self._try_entry('M2_long_hgb60', now, price,
                            condition=True, direction='LONG', probability=hgb_prob)

    def _try_entry(self, model_id, now, price, condition, direction, probability):
        """Attempt entry for a specific strategy."""
        if not condition:
            return

        # Already have a position for this strategy?
        if any(p.model_id == model_id for p in self.positions):
            return

        # Entry gap cooldown
        last = self.last_entry.get(model_id)
        if last and (now - last).total_seconds() < MIN_ENTRY_GAP_SEC:
            return

        # SL cooldown
        last_sl = self.last_sl_exit.get(model_id)
        if last_sl and (now - last_sl).total_seconds() < SL_COOLDOWN_SEC:
            return

        # Max positions
        if len(self.positions) >= MAX_POSITIONS:
            return

        # Calculate TP/SL prices
        if direction == 'LONG':
            tp_price = price * (1 + TP_PCT / 100)
            sl_price = price * (1 - SL_PCT / 100)
        else:
            tp_price = price * (1 - TP_PCT / 100)
            sl_price = price * (1 + SL_PCT / 100)

        # Place IB order
        order_id = None
        if self.ib.isConnected():
            try:
                action = 'BUY' if direction == 'LONG' else 'SELL'
                order = MarketOrder(action, POSITION_SIZE)
                trade = self.ib.placeOrder(self.contract, order)
                self.ib.sleep(2)
                if trade.orderStatus.status == 'Filled':
                    price = trade.orderStatus.avgFillPrice
                    order_id = trade.order.orderId
                    logger.info("Order filled: %s %d @ $%.2f" % (action, POSITION_SIZE, price))
                    # Recalculate TP/SL with fill price
                    if direction == 'LONG':
                        tp_price = price * (1 + TP_PCT / 100)
                        sl_price = price * (1 - SL_PCT / 100)
                    else:
                        tp_price = price * (1 - TP_PCT / 100)
                        sl_price = price * (1 + SL_PCT / 100)
                else:
                    logger.warning("[%s] Order not filled: %s — aborting" % (model_id, trade.orderStatus.status))
                    return
            except Exception as e:
                logger.error("[%s] Order error: %s" % (model_id, e))
                return

        pos = Position(
            model_id=model_id, direction=direction,
            entry_price=price, entry_time=now,
            size=POSITION_SIZE, tp_price=tp_price, sl_price=sl_price,
            order_id=order_id, entry_probability=probability,
        )
        self.positions.append(pos)
        self.last_entry[model_id] = now

        # Write to open_positions for monitor daemon
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.execute(
                "INSERT INTO open_positions (model_id, direction, entry_time, entry_price, tp_price, sl_price, entry_probability) VALUES (?,?,?,?,?,?,?)",
                (model_id, direction, now.isoformat(), price, tp_price, sl_price, probability))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug("open_positions write: %s" % e)

        logger.info("ENTRY %s [%s] @ $%.2f (prob=%.2f) | TP=$%.2f SL=$%.2f" % (
            direction, model_id, price, probability, tp_price, sl_price))

    # =================================================================
    # EXIT LOGIC (pure TP/SL)
    # =================================================================
    def _get_mbt_price(self):
        """Get IB MBT futures price for SL/TP monitoring (Mar 30 fix)."""
        if self.ib_mbt_ticker is not None:
            for attr in ['last', 'close']:
                v = getattr(self.ib_mbt_ticker, attr, None)
                if v and v > 10000:
                    return v
            mp = self.ib_mbt_ticker.marketPrice()
            if mp and mp > 10000:
                return mp
        return None

    def check_exits(self):
        if self.current_price is None:
            return

        # Mar 31: BINANCE-ONLY — use Binance price for all SL/TP monitoring.
        # MBT ticker freezes during CME maintenance breaks → stale price catastrophe.
        price = self.current_price
        closed = []

        for pos in self.positions:
            if pos.direction == 'LONG':
                if price >= pos.tp_price:
                    self._close_position(pos, price, 'TAKE_PROFIT')
                    closed.append(pos)
                elif price <= pos.sl_price:
                    self._close_position(pos, price, 'STOP_LOSS')
                    closed.append(pos)
            else:  # SHORT
                if price <= pos.tp_price:
                    self._close_position(pos, price, 'TAKE_PROFIT')
                    closed.append(pos)
                elif price >= pos.sl_price:
                    self._close_position(pos, price, 'STOP_LOSS')
                    closed.append(pos)

        for pos in closed:
            self.positions.remove(pos)

    def _close_position(self, pos, exit_price, reason):
        now = datetime.now()

        if pos.direction == 'LONG':
            pnl_pct = (exit_price / pos.entry_price - 1) * 100
        else:
            pnl_pct = (pos.entry_price / exit_price - 1) * 100

        pnl_dollar = pnl_pct / 100 * pos.entry_price * BTC_CONTRACT_VALUE * pos.size

        # Close via IB
        if self.ib.isConnected():
            try:
                action = 'SELL' if pos.direction == 'LONG' else 'BUY'
                order = MarketOrder(action, pos.size)
                trade = self.ib.placeOrder(self.contract, order)
                self.ib.sleep(2)
                if trade.orderStatus.status == 'Filled':
                    exit_price = trade.orderStatus.avgFillPrice
                    if pos.direction == 'LONG':
                        pnl_pct = (exit_price / pos.entry_price - 1) * 100
                    else:
                        pnl_pct = (pos.entry_price / exit_price - 1) * 100
                    pnl_dollar = pnl_pct / 100 * pos.entry_price * BTC_CONTRACT_VALUE * pos.size
                else:
                    logger.warning("[%s] Close not filled: %s" % (pos.model_id, trade.orderStatus.status))
            except Exception as e:
                logger.error("[%s] Close error: %s" % (pos.model_id, e))

        bars_held = int((now - pos.entry_time).total_seconds() / 16)

        if reason == 'STOP_LOSS':
            self.last_sl_exit[pos.model_id] = now

        logger.info("EXIT %s [%s] %s @ $%.2f | PnL: $%.2f (%.3f%%) | held %d bars" % (
            pos.direction, pos.model_id, reason, exit_price, pnl_dollar, pnl_pct, bars_held))

        # Remove from open_positions
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.execute("DELETE FROM open_positions WHERE model_id = ? AND entry_time = ?",
                         (pos.model_id, pos.entry_time.isoformat()))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug("open_positions remove: %s" % e)

        # Log to trades table
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
        })

    # =================================================================
    # SIGNAL LOGGING
    # =================================================================
    def log_signals(self, feat, hgb_prob):
        now = datetime.now()
        if self.last_signal_log_time:
            if (now - self.last_signal_log_time).total_seconds() < 15:
                return

        vol_60 = feat.iloc[-1].get('vol_60', 0) if feat is not None and len(feat) > 0 else 0
        if pd.isna(vol_60):
            vol_60 = 0
        ret_5 = feat.iloc[-1].get('ret_5', 0) if feat is not None and len(feat) > 0 else 0
        if pd.isna(ret_5):
            ret_5 = 0

        vol_ratio = vol_60 / self.vol_60_median if self.vol_60_median and self.vol_60_median > 0 else 0
        n_pos = len(self.positions)
        pos_str = ", ".join(["%s(%s)" % (p.model_id, p.direction) for p in self.positions]) if self.positions else "none"

        logger.info("BTC: $%.2f | Pos: %d/%d [%s] | HGB=%.1f%% vol=%.1fx ret5m=%+.2f%%" % (
            self.current_price or 0, n_pos, MAX_POSITIONS, pos_str,
            (hgb_prob or 0) * 100, vol_ratio, ret_5))

        # CSV log
        today = now.strftime('%Y-%m-%d')
        if self.signal_csv_date != today:
            self.signal_csv_date = today
            self.signal_csv_path = os.path.join(SIGNAL_LOG_DIR, 'btc_tpsl_signals_%s.csv' % today)
            if not os.path.exists(self.signal_csv_path):
                with open(self.signal_csv_path, 'w') as f:
                    f.write('timestamp,btc_price,hgb_prob,vol_60,vol_ratio,ret_5,active_positions\n')
        try:
            with open(self.signal_csv_path, 'a') as f:
                f.write('%s,%.2f,%.4f,%.6f,%.2f,%.4f,%d\n' % (
                    now.strftime('%Y-%m-%d %H:%M:%S'),
                    self.current_price or 0,
                    hgb_prob or 0, vol_60, vol_ratio, ret_5, n_pos))
        except Exception:
            pass

        self.last_signal_log_time = now

    # =================================================================
    # MAIN LOOP
    # =================================================================
    def run(self):
        if not self.connect_ib():
            logger.error("Cannot start without IB connection")
            return

        # Clear stale positions
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.execute("DELETE FROM open_positions")
            conn.commit()
            conn.close()
        except Exception:
            pass

        self.load_historical_ticks()
        self.start_websocket()

        logger.info("Waiting for WebSocket price data...")
        for _ in range(30):
            if self.current_price:
                break
            time.sleep(1)

        if not self.current_price:
            logger.error("No price data after 30s")
            return

        logger.info("Starting main loop. Price: $%.2f" % self.current_price)

        last_ib_check = datetime.now()
        last_reconnect_attempt = None

        while True:
            try:
                now = datetime.now()

                # IB connection check every 60s
                if (now - last_ib_check).total_seconds() > 60:
                    if not self.ib.isConnected():
                        logger.warning("IB disconnected — reconnecting")
                        if last_reconnect_attempt is None or (now - last_reconnect_attempt).total_seconds() > 30:
                            last_reconnect_attempt = now
                            self.reconnect_ib()
                    last_ib_check = now

                # Check exits every tick
                self.check_exits()

                # Check signals every 16s
                if self.last_signal_check is None or (now - self.last_signal_check).total_seconds() >= SIGNAL_CHECK_INTERVAL:
                    # Market hours guard — CME MBT
                    try:
                        from zoneinfo import ZoneInfo
                        _tz = ZoneInfo('America/Chicago')
                    except ImportError:
                        import pytz
                        _tz = pytz.timezone('America/Chicago')
                    _ct = datetime.now(_tz).replace(tzinfo=None)
                    _wd, _h = _ct.weekday(), _ct.hour
                    _mkt_open = not (_wd == 5 or (_wd == 6 and _h < 17) or _h == 16 or (_wd == 4 and _h >= 16))
                    if not _mkt_open:
                        self.last_signal_check = now
                        self.ib.sleep(0.1)
                        time.sleep(0.1)
                        continue

                    feat = self.get_features()
                    hgb_prob = None
                    if feat is not None and len(feat) > 0:
                        self.check_entry(feat)
                        # Get HGB prob for logging
                        if self.model is not None and self.feature_cols is not None:
                            try:
                                row = feat.iloc[-1][self.feature_cols].fillna(0).values.reshape(1, -1)
                                hgb_prob = self.model.predict_proba(row)[0][1]
                            except Exception:
                                pass
                        self.log_signals(feat, hgb_prob)

                    self.last_signal_check = now

                if self.ib.isConnected():
                    self.ib.sleep(0.5)
                else:
                    time.sleep(0.5)

            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error("Main loop error: %s" % e)
                time.sleep(5)

        if self.ws:
            self.ws.close()
        if self.ib.isConnected():
            self.ib.disconnect()
        logger.info("Bot stopped.")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='BTC TP/SL Multi-Strategy Bot')
    parser.add_argument('--paper', action='store_true', default=True)
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    bot = BTCTPSLBot(paper_trading=not args.live)
    bot.run()

if __name__ == "__main__":
    main()
