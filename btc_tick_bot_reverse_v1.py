#!/usr/bin/env python3
"""
BTC Tick ML Bot — REVERSE MODE

Fades the original tick bot models:
  - When LONG models fire (L1 >= 55%) → enters SHORT
  - When SHORT models fire (2+ of S1/S3/S4 agree) → enters LONG

Rationale: Original bot went 5W/16L in last 24h (24% WR).
If the model is consistently wrong, the opposite trade should win.

Separate from original: clientId=402, tick_trades_reverse.db, btc_tick_bot_reverse.log

SL/TS kept identical to original (applied to the reversed direction).
Same safety: trend filter, macro block, SL cooldown, consecutive loss pause.
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
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import deque

# IB
from ib_insync import IB, Future, MarketOrder, util

# WebSocket
try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================
BOT_NAME = "BTC_TICK_REVERSE"
IB_HOST = '127.0.0.1'
IB_PORT = 4002
IB_CLIENT_ID = 402  # separate ID for reverse bot

# REVERSE MODE: Flip all model signals — LONG models trigger SHORT entries, vice versa
REVERSE_MODE = True

# Contract
MBT_EXPIRY = '20260327'
BTC_CONTRACT_VALUE = 0.1  # MBT = 0.1 BTC

# Position management
MAX_POSITIONS = 4
MAX_LONG = 1
MAX_SHORT = 1  # Mar 16: reduced from 2 — inverted R:R ($30 SL vs $15 win)
POSITION_SIZE = 1  # contracts per trade
MIN_ENTRY_GAP_SEC = 300  # 5 min between entries of same direction
SL_COOLDOWN_SEC = 600   # 10 min cooldown after any stop loss
MAX_CONSEC_LOSSES = 3   # pause after 3 consecutive losses
CONSEC_LOSS_PAUSE_SEC = 1800  # 30 min pause after consecutive losses

# Signal check interval
SIGNAL_CHECK_INTERVAL = 16  # seconds (one 16-sec bar)
SIGNAL_LOG_INTERVAL = 15  # seconds between signal logs

# Trailing stop params per direction — SWAPPED vs original tick bot for exact P&L mirror
# Reverse SHORT (triggered by original LONG signal) uses original LONG params
# Reverse LONG (triggered by original SHORT signal) uses original SHORT params
LONG_SL_PCT = 0.20      # Original SHORT SL (mirror: SHORT signal → LONG entry)
LONG_TS_ACT_PCT = 0.35  # Original SHORT TS activation
LONG_TS_TRAIL_PCT = 0.15 # Original SHORT TS trail

SHORT_SL_PCT = 0.45     # Original LONG SL (mirror: LONG signal → SHORT entry)
SHORT_TS_ACT_PCT = 0.50 # Original LONG TS activation
SHORT_TS_TRAIL_PCT = 0.15 # Original LONG TS trail

# Model thresholds
LONG_THRESHOLD = 0.55   # Mar 16: enriched L1 model — 55% WR at >55% with volume/momentum features
SHORT_AGREEMENT = 2     # Need 2+ SHORT models >= their threshold (0.70)

# Trend filter — skip LONG if 1h return too negative, skip SHORT if too positive
TREND_FILTER_LONG = -0.30   # block LONG if ret_1h < -0.30%
TREND_FILTER_SHORT = 0.30   # block SHORT if ret_1h > +0.30%

# Macro trend block — hard-block SHORT during sustained BTC uptrend
# Mar 13: backtest shows SHORT model fires 40% of bars during +4%→+10% 24h rally
# at only 51% avg prob, explaining live -$308 net SHORT loss this week
MACRO_SHORT_BLOCK_PCT = 2.0   # block ALL SHORT if 24h change > +2%
MACRO_LONG_BLOCK_PCT = -2.0   # block ALL LONG if 24h change < -2%
MACRO_LOOKBACK_BARS = 5400    # 24h of 16-sec bars (24 * 3600 / 16)

# Trend-following mode for strong trends
TREND_FOLLOW_ENABLED = False  # Mar 16: disabled — 0.30 threshold lets noise through, contributed to losses
TREND_FOLLOW_LONG_THRESHOLD = 1.0   # Activate if ret_1h > +1% (≈4% 24h rally)
TREND_FOLLOW_SHORT_THRESHOLD = -1.0 # Activate if ret_1h < -1%
TREND_FOLLOW_LONG_MODEL_THRESHOLD = 0.30  # Lower threshold for LONG entries
TREND_FOLLOW_SHORT_MODEL_THRESHOLD = 0.30 # Lower threshold for SHORT entries

# Model directory
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'tick_models')

# Enriched features from 5-min parquet (volume, momentum, trend indicators)
PARQUET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'BTC_features.parquet')
ENRICHED_COLS = [
    'Volume', 'volume_obv', 'volume_cmf', 'volume_mfi', 'volume_vwap',
    'volatility_atr', 'volatility_bbp', 'volatility_bbw', 'volatility_kcp',
    'trend_macd_diff', 'trend_adx', 'trend_adx_pos', 'trend_adx_neg', 'trend_cci',
    'trend_aroon_ind', 'trend_stc',
    'momentum_rsi', 'momentum_stoch', 'momentum_wr', 'momentum_ao', 'momentum_roc',
    'momentum_tsi', 'momentum_ppo_hist',
]
PARQUET_RELOAD_SEC = 300  # reload parquet every 5 min

# Logging
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
SIGNAL_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'signal_logs')
os.makedirs(SIGNAL_LOG_DIR, exist_ok=True)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tick_trades_reverse.db')

logger = logging.getLogger('btc_tick_bot_reverse')
logger.setLevel(logging.INFO)

# File handler
fh = logging.FileHandler(os.path.join(LOG_DIR, 'btc_tick_bot_reverse.log'))
fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(fh)

# Console handler removed — nohup redirects stdout to log file causing duplicates


# =============================================================================
# DATA STRUCTURES
# =============================================================================
@dataclass
class Position:
    model_id: str
    direction: str
    entry_price: float
    entry_time: datetime
    size: int
    sl_price: float
    ts_act_price: float
    ts_trail_pct: float
    peak_price: float  # best price seen (highest for LONG, lowest for SHORT)
    ts_active: bool = False
    order_id: Optional[int] = None
    entry_probability: float = 0.0
    short_agreement: int = 0


# =============================================================================
# FEATURE COMPUTATION
# =============================================================================
def compute_rsi(s, p):
    d = s.diff()
    g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - (100 / (1 + g / l.replace(0, np.nan)))

def compute_atr(h, l, c, p):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(p).mean()

def compute_features(bars):
    """Compute all features from 16-sec OHLC bars."""
    close = bars['close']
    high = bars['high']
    low = bars['low']

    feat = pd.DataFrame(index=bars.index)

    for lb, lbl in [(4, '1m'), (8, '2m'), (19, '5m'), (38, '10m'), (56, '15m'), (113, '30m'), (225, '1h'), (5400, '24h')]:
        feat['ret_' + lbl] = close.pct_change(lb) * 100

    ret1 = close.pct_change() * 100
    for w, lbl in [(19, '5m'), (56, '15m'), (113, '30m'), (225, '1h')]:
        feat['vol_' + lbl] = ret1.rolling(w).std()

    feat['vol_ratio_5m_1h'] = feat['vol_5m'] / feat['vol_1h'].replace(0, np.nan)
    feat['vol_ratio_15m_1h'] = feat['vol_15m'] / feat['vol_1h'].replace(0, np.nan)

    for w, lbl in [(56, '15m'), (225, '1h'), (450, '2h')]:
        rmin = close.rolling(w).min()
        rmax = close.rolling(w).max()
        rng = (rmax - rmin).replace(0, np.nan)
        feat['chan_' + lbl] = (close - rmin) / rng

    for p, lbl in [(19, '5m'), (56, '15m'), (225, '1h')]:
        ema = close.ewm(span=p, adjust=False).mean()
        feat['ema_dist_' + lbl] = (close - ema) / close * 100

    for p, lbl in [(56, '15m'), (225, '1h')]:
        feat['rsi_' + lbl] = compute_rsi(close, p)

    for p, lbl in [(56, '15m'), (225, '1h')]:
        sma = close.rolling(p).mean()
        std = close.rolling(p).std()
        lower = sma - 2 * std
        upper = sma + 2 * std
        feat['bb_' + lbl] = (close - lower) / (upper - lower).replace(0, np.nan)

    for (f, s, sig, lbl) in [(19, 56, 14, '5m_15m'), (56, 225, 38, '15m_1h')]:
        ef = close.ewm(span=f, adjust=False).mean()
        es = close.ewm(span=s, adjust=False).mean()
        macd_line = ef - es
        signal_line = macd_line.ewm(span=sig, adjust=False).mean()
        feat['macd_hist_' + lbl] = macd_line - signal_line

    for p, lbl in [(56, '15m'), (225, '1h')]:
        feat['atr_pct_' + lbl] = compute_atr(high, low, close, p) / close * 100

    feat['speed_5m_15m'] = (close.diff(19).abs() / 19) / (close.diff(56).abs() / 56).replace(0, np.nan)

    ret_sign = np.sign(close.diff())
    feat['consec'] = ret_sign.groupby((ret_sign != ret_sign.shift()).cumsum()).cumcount() + 1
    feat['consec'] = feat['consec'] * ret_sign

    feat['bar_range_pct'] = (high - low) / close * 100
    feat['bar_range_ma'] = feat['bar_range_pct'].rolling(56).mean()
    feat['bar_range_ratio'] = feat['bar_range_pct'] / feat['bar_range_ma'].replace(0, np.nan)

    up_ticks = (close > close.shift(1)).astype(int)
    down_ticks = (close < close.shift(1)).astype(int)
    for w, lbl in [(19, '5m'), (56, '15m')]:
        feat['tick_imbal_' + lbl] = (up_ticks.rolling(w).sum() - down_ticks.rolling(w).sum()) / w

    feat['hour'] = bars.index.hour

    return feat


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
# BOT CLASS
# =============================================================================
class BTCTickBot:
    def __init__(self, paper_trading=True):
        self.paper_trading = paper_trading
        self.ib = IB()
        self.contract = None
        self.positions: List[Position] = []
        self.current_price = None
        self.current_price_time = None

        # Tick data storage
        self.tick_prices = deque(maxlen=200000)  # ~4.5 days of 2-sec ticks
        self.tick_times = deque(maxlen=200000)

        # WebSocket
        self.ws = None
        self.ws_thread = None
        self.ws_connected = False

        # Models
        self.models = {}
        self.model_configs = {}

        # Timing
        self.last_signal_check = None
        self.last_long_entry = None
        self.last_short_entry = None
        self.last_signal_log_time = None
        self.last_sl_exit_long = None     # last SL exit time for LONG direction
        self.last_sl_exit_short = None    # last SL exit time for SHORT direction
        self.consec_losses = 0            # consecutive loss counter
        self.consec_loss_pause_until = None  # pause trading until this time

        # Signal log
        self.signal_csv_path = None
        self.signal_csv_date = None

        # Enriched feature cache (from 5-min parquet)
        self.parquet_data = None
        self.parquet_last_load = None
        self._load_parquet()

        # Load models
        self._load_models()

        # Init DB
        init_db(DB_PATH)

        logger.info("=" * 60)
        logger.info("BTC TICK ML BOT [REVERSE MODE] INITIALIZED")
        logger.info("=" * 60)
        logger.info("Models: %s" % ', '.join(self.models.keys()))
        logger.info("LONG: any of [L1, L3] >= %.0f%%" % (LONG_THRESHOLD * 100))
        logger.info("SHORT: %d+ of [S1, S3, S4] >= threshold" % SHORT_AGREEMENT)
        logger.info("LONG SL=%.2f%%, TS=%.2f%%/%.2f%%" % (LONG_SL_PCT, LONG_TS_ACT_PCT, LONG_TS_TRAIL_PCT))
        logger.info("SHORT SL=%.2f%%, TS=%.2f%%/%.2f%%" % (SHORT_SL_PCT, SHORT_TS_ACT_PCT, SHORT_TS_TRAIL_PCT))
        logger.info("Paper: %s" % paper_trading)

    def _load_parquet(self):
        """Load enriched features from 5-min parquet."""
        try:
            pq = pd.read_parquet(PARQUET_PATH)
            available = [c for c in ENRICHED_COLS if c in pq.columns]
            self.parquet_data = pq[available].copy()
            self.parquet_last_load = time.time()
            logger.info("Loaded parquet: %d rows, %d enriched features, latest=%s" % (
                len(self.parquet_data), len(available), self.parquet_data.index[-1]))
        except Exception as e:
            logger.warning("Could not load parquet: %s" % e)
            self.parquet_data = None

    def _load_models(self):
        """Load all trained RF models."""
        for name in ['L1_full_wide', 'L3_mom_wide', 'S1_full_cur', 'S3_mr_cur', 'S4_full_tight']:
            model_path = os.path.join(MODELS_DIR, '%s.pkl' % name)
            config_path = os.path.join(MODELS_DIR, '%s_config.pkl' % name)
            if os.path.exists(model_path) and os.path.exists(config_path):
                self.models[name] = joblib.load(model_path)
                self.model_configs[name] = joblib.load(config_path)
                logger.info("  Loaded %s (features=%d, thresh=%.2f)" % (
                    name, len(self.model_configs[name]['features']),
                    self.model_configs[name]['threshold']))
            else:
                logger.error("  Model not found: %s" % model_path)

    # =================================================================
    # IB CONNECTION
    # =================================================================
    def connect_ib(self):
        """Connect to IB Gateway."""
        max_retries = 999
        for attempt in range(max_retries):
            try:
                try:
                    if self.ib.isConnected():
                        self.ib.disconnect()
                        time.sleep(2)
                except Exception:
                    pass

                port = IB_PORT
                self.ib.connect(IB_HOST, port, clientId=IB_CLIENT_ID, timeout=20)
                logger.info("Connected to IB Gateway (port=%d, clientId=%d)" % (port, IB_CLIENT_ID))

                # Set up MBT contract
                self.contract = Future(
                    symbol='MBT',
                    lastTradeDateOrContractMonth=MBT_EXPIRY,
                    exchange='CME',
                    currency='USD',
                    multiplier='0.1'
                )
                self.ib.qualifyContracts(self.contract)
                logger.info("Contract: %s" % self.contract)
                return True

            except Exception as e:
                logger.error("IB connection attempt %d failed: %s" % (attempt + 1, e))
                time.sleep(5)

        logger.error("Failed to connect to IB after %d attempts" % max_retries)
        return False

    def reconnect_ib(self):
        """Reconnect to IB."""
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
        """Start Binance WebSocket for 2-sec price ticks."""
        if not WEBSOCKET_AVAILABLE:
            logger.warning("websocket-client not installed")
            return

        def on_message(ws, message):
            try:
                data = json.loads(message)
                if 'c' in data:
                    price = float(data['c'])
                    now = datetime.now()
                    self.current_price = price
                    self.current_price_time = now
                    self.tick_prices.append(price)
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
                    ws_url = "wss://stream.binance.com:9443/ws/btcusdt@miniTicker"
                    self.ws = websocket.WebSocketApp(
                        ws_url,
                        on_message=on_message,
                        on_error=on_error,
                        on_close=on_close,
                        on_open=on_open
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
        """Load existing tick data from CSV for feature warmup."""
        tick_file = os.path.join(LOG_DIR, 'btc_price_ticks.csv')
        if os.path.exists(tick_file):
            try:
                df = pd.read_csv(tick_file, parse_dates=['timestamp'])
                df = df.sort_values('timestamp')
                # Only keep last 3 days for feature warmup (need ~2h = 3600 ticks minimum)
                cutoff = datetime.now() - timedelta(days=3)
                df = df[df['timestamp'] > cutoff]
                for _, row in df.iterrows():
                    self.tick_prices.append(row['price'])
                    self.tick_times.append(row['timestamp'].to_pydatetime())
                logger.info("Loaded %d historical ticks for warmup" % len(df))
            except Exception as e:
                logger.error("Failed to load historical ticks: %s" % e)

    # =================================================================
    # FEATURE COMPUTATION (from live ticks)
    # =================================================================
    def get_features(self):
        """Build 16-sec bars from ticks and compute features."""
        if len(self.tick_prices) < 500:
            return None

        # Build DataFrame from ticks
        tick_df = pd.DataFrame({
            'timestamp': list(self.tick_times),
            'price': list(self.tick_prices)
        })
        tick_df = tick_df.set_index('timestamp').sort_index()

        # Resample to 16-sec bars
        bars = tick_df['price'].resample('16s').agg(
            **{'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
        ).dropna()

        if len(bars) < 500:
            return None

        feat = compute_features(bars)

        # Merge enriched features from 5-min parquet (forward-fill onto 16-sec bars)
        if self.parquet_data is not None:
            # Reload parquet periodically
            if self.parquet_last_load and (time.time() - self.parquet_last_load) > PARQUET_RELOAD_SEC:
                self._load_parquet()
            if self.parquet_data is not None:
                enriched = self.parquet_data.reindex(feat.index, method='ffill')
                for col in enriched.columns:
                    feat['ext_' + col] = enriched[col]

        return feat

    def get_probabilities(self, feat):
        """Run all models on latest features, return dict of probabilities."""
        if feat is None or len(feat) == 0:
            return {}

        last_row = feat.iloc[[-1]]
        probs = {}

        for name, model in self.models.items():
            cfg = self.model_configs[name]
            cols = cfg['features']

            # Check all features available
            missing = [c for c in cols if c not in last_row.columns]
            if missing:
                continue

            X = last_row[cols]
            if X.isna().any(axis=1).iloc[0]:
                continue

            prob = model.predict_proba(X)[0][1]
            probs[name] = prob

        return probs

    # =================================================================
    # ENTRY LOGIC
    # =================================================================
    def check_entry(self, probs, feat=None):
        """Check if we should enter a new position."""
        if not probs or self.current_price is None:
            return

        now = datetime.now()
        n_pos = len(self.positions)

        # --- GLOBAL SAFETY CHECKS ---
        # Consecutive loss pause
        if self.consec_loss_pause_until and now < self.consec_loss_pause_until:
            return
        elif self.consec_loss_pause_until and now >= self.consec_loss_pause_until:
            logger.info("Consecutive loss pause expired, resuming trading")
            self.consec_loss_pause_until = None
            self.consec_losses = 0

        # Stop-loss cooldown (per direction)
        last_sl_long = self.last_sl_exit_long
        last_sl_short = self.last_sl_exit_short

        # --- TREND FILTER ---
        ret_1h = None
        if feat is not None and len(feat) > 0 and 'ret_1h' in feat.columns:
            ret_1h = feat['ret_1h'].iloc[-1]
            if pd.isna(ret_1h):
                ret_1h = None

        # --- MACRO TREND (24h) ---
        ret_24h = None
        if feat is not None and len(feat) > 0 and 'ret_24h' in feat.columns:
            v = feat['ret_24h'].iloc[-1]
            if not pd.isna(v):
                ret_24h = v

        # --- TREND-FOLLOWING MODE ---
        trend_follow_mode = False
        trend_follow_reason = ""
        if TREND_FOLLOW_ENABLED and ret_1h is not None:
            if ret_1h > TREND_FOLLOW_LONG_THRESHOLD:
                trend_follow_mode = True
                trend_follow_reason = "TREND FOLLOW: ret_1h=+%.1f%% > +%.1f%%" % (ret_1h, TREND_FOLLOW_LONG_THRESHOLD)
            elif ret_1h < TREND_FOLLOW_SHORT_THRESHOLD:
                trend_follow_mode = True
                trend_follow_reason = "TREND FOLLOW: ret_1h=%.1f%% < %.1f%%" % (ret_1h, TREND_FOLLOW_SHORT_THRESHOLD)

        # === REVERSE MODE ===
        # LONG models fire → enter SHORT (fade the model)
        # SHORT models fire → enter LONG (fade the model)
        # IMPORTANT: Filters match the SIGNAL direction (not traded direction)
        # so the reverse bot fires exactly when the original bot would have fired.

        # --- LONG models fire → enter SHORT ---
        # Use LONG filters (same as original bot applies to LONG signals)
        n_short = sum(1 for p in self.positions if p.direction == 'SHORT')
        if n_short < MAX_SHORT and n_pos < MAX_POSITIONS:
            gap_ok = self.last_short_entry is None or (now - self.last_short_entry).total_seconds() >= MIN_ENTRY_GAP_SEC
            short_sl_ok = last_sl_short is None or (now - last_sl_short).total_seconds() >= SL_COOLDOWN_SEC

            if gap_ok and short_sl_ok:
                # Macro trend block: original blocks LONG if 24h < -2% → same here
                if ret_24h is not None and ret_24h < MACRO_LONG_BLOCK_PCT:
                    logger.info("REV SHORT blocked: 24h change %.2f%% < %.1f%% (original would block LONG)" % (ret_24h, MACRO_LONG_BLOCK_PCT))
                # Trend filter: original blocks LONG if ret_1h < -0.30% → same here
                elif ret_1h is not None and ret_1h < TREND_FILTER_LONG:
                    logger.debug("REV SHORT blocked: ret_1h=%.2f%% < %.2f%% (original would block LONG)" % (ret_1h, TREND_FILTER_LONG))
                else:
                    long_threshold = LONG_THRESHOLD
                    long_models = ['L1_full_wide']
                    for m in long_models:
                        if m in probs and probs[m] >= long_threshold:
                            # Model says LONG → we enter SHORT
                            self._enter_position('SHORT', 'REV_' + m, probs[m], probs)
                            break

        # --- SHORT models fire → enter LONG ---
        # Use SHORT filters (same as original bot applies to SHORT signals)
        n_long = sum(1 for p in self.positions if p.direction == 'LONG')
        if n_long < MAX_LONG and n_pos < MAX_POSITIONS:
            gap_ok = self.last_long_entry is None or (now - self.last_long_entry).total_seconds() >= MIN_ENTRY_GAP_SEC
            long_sl_ok = last_sl_long is None or (now - last_sl_long).total_seconds() >= SL_COOLDOWN_SEC

            if gap_ok and long_sl_ok:
                pass
            elif gap_ok and not long_sl_ok:
                gap_ok = False

            if gap_ok:
                # Macro trend block: original blocks SHORT if 24h > +2% → same here
                if ret_24h is not None and ret_24h > MACRO_SHORT_BLOCK_PCT:
                    logger.info("REV LONG blocked: 24h change +%.2f%% > +%.1f%% (original would block SHORT)" % (ret_24h, MACRO_SHORT_BLOCK_PCT))
                    gap_ok = False
                # Trend filter: original blocks SHORT if ret_1h > +0.30% → same here
                elif ret_1h is not None and ret_1h > TREND_FILTER_SHORT:
                    logger.debug("REV LONG blocked: ret_1h=%.2f%% > %.2f%% (original would block SHORT)" % (ret_1h, TREND_FILTER_SHORT))
                    gap_ok = False

            if gap_ok:
                short_threshold = 0.70
                short_models = ['S1_full_cur', 'S3_mr_cur', 'S4_full_tight']
                agrees = 0
                trigger_model = None
                for m in short_models:
                    if m in probs:
                        thresh = self.model_configs[m]['threshold']
                        if probs[m] >= thresh:
                            agrees += 1
                            if trigger_model is None:
                                trigger_model = m

                if agrees >= SHORT_AGREEMENT and trigger_model:
                    # Model says SHORT → we enter LONG
                    self._enter_position('LONG', 'REV_' + trigger_model, probs[trigger_model], probs, short_agreement=agrees)

    def _enter_position(self, direction, model_id, probability, all_probs, short_agreement=0):
        """Enter a new position."""
        price = self.current_price
        now = datetime.now()

        if direction == 'LONG':
            sl = price * (1 - LONG_SL_PCT / 100)
            ts_act = price * (1 + LONG_TS_ACT_PCT / 100)
            ts_trail = LONG_TS_TRAIL_PCT
        else:
            sl = price * (1 + SHORT_SL_PCT / 100)
            ts_act = price * (1 - SHORT_TS_ACT_PCT / 100)
            ts_trail = SHORT_TS_TRAIL_PCT

        # Place order via IB
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
                else:
                    logger.warning("Order status: %s — not filled, aborting entry" % trade.orderStatus.status)
                    return
            except Exception as e:
                logger.error("Order error: %s" % e)
                return

        # Bug fix: recalculate SL/TS if IB fill price differs
        if direction == 'LONG':
            sl = price * (1 - LONG_SL_PCT / 100)
            ts_act = price * (1 + LONG_TS_ACT_PCT / 100)
        else:
            sl = price * (1 + SHORT_SL_PCT / 100)
            ts_act = price * (1 - SHORT_TS_ACT_PCT / 100)

        pos = Position(
            model_id=model_id,
            direction=direction,
            entry_price=price,
            entry_time=now,
            size=POSITION_SIZE,
            sl_price=sl,
            ts_act_price=ts_act,
            ts_trail_pct=ts_trail,
            peak_price=price,
            order_id=order_id,
            entry_probability=probability,
            short_agreement=short_agreement,
        )
        self.positions.append(pos)

        if direction == 'LONG':
            self.last_long_entry = now
        else:
            self.last_short_entry = now

        prob_str = ', '.join(["%s=%.0f%%" % (k, v * 100) for k, v in all_probs.items()])
        agree_str = " [%d/3 agree]" % short_agreement if direction == 'SHORT' else ""
        logger.info("ENTRY %s [%s] @ $%.2f (prob=%.0f%%%s) | SL=$%.2f | %s" % (
            direction, model_id, price, probability * 100, agree_str,
            sl, prob_str))

    # =================================================================
    # EXIT LOGIC
    # =================================================================
    def check_exits(self):
        """Check SL and trailing stops for all positions."""
        if self.current_price is None:
            return

        price = self.current_price
        closed = []

        for pos in self.positions:
            if pos.direction == 'LONG':
                # Update peak
                if price > pos.peak_price:
                    pos.peak_price = price

                # Stop loss
                if price <= pos.sl_price:
                    self._close_position(pos, price, 'STOP_LOSS')
                    closed.append(pos)
                    continue

                # Trailing stop activation
                if not pos.ts_active and price >= pos.ts_act_price:
                    pos.ts_active = True
                    logger.info("  TS ACTIVATED [%s] at $%.2f (peak=$%.2f)" % (
                        pos.model_id, price, pos.peak_price))

                # Trailing stop exit
                if pos.ts_active:
                    trail_price = pos.peak_price * (1 - pos.ts_trail_pct / 100)
                    if price <= trail_price:
                        self._close_position(pos, price, 'TRAILING_STOP')
                        closed.append(pos)

            else:  # SHORT
                if price < pos.peak_price:
                    pos.peak_price = price

                if price >= pos.sl_price:
                    self._close_position(pos, price, 'STOP_LOSS')
                    closed.append(pos)
                    continue

                if not pos.ts_active and price <= pos.ts_act_price:
                    pos.ts_active = True
                    logger.info("  TS ACTIVATED [%s] at $%.2f (peak=$%.2f)" % (
                        pos.model_id, price, pos.peak_price))

                if pos.ts_active:
                    trail_price = pos.peak_price * (1 + pos.ts_trail_pct / 100)
                    if price >= trail_price:
                        self._close_position(pos, price, 'TRAILING_STOP')
                        closed.append(pos)

        for pos in closed:
            self.positions.remove(pos)

    def _close_position(self, pos, exit_price, reason):
        """Close a position and log the trade."""
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
                    # Recalculate PnL with actual fill
                    if pos.direction == 'LONG':
                        pnl_pct = (exit_price / pos.entry_price - 1) * 100
                    else:
                        pnl_pct = (pos.entry_price / exit_price - 1) * 100
                    pnl_dollar = pnl_pct / 100 * pos.entry_price * BTC_CONTRACT_VALUE * pos.size
                else:
                    logger.warning("Close order status: %s — not filled" % trade.orderStatus.status)
            except Exception as e:
                logger.error("Close order error: %s" % e)

        bars_held = int((now - pos.entry_time).total_seconds() / 16)

        # Track stop-loss cooldown and consecutive losses
        if reason == 'STOP_LOSS':
            if pos.direction == 'LONG':
                self.last_sl_exit_long = now
            else:
                self.last_sl_exit_short = now
            self.consec_losses += 1
            if self.consec_losses >= MAX_CONSEC_LOSSES:
                self.consec_loss_pause_until = now + timedelta(seconds=CONSEC_LOSS_PAUSE_SEC)
                logger.warning("PAUSED: %d consecutive losses, pausing until %s" % (
                    self.consec_losses, self.consec_loss_pause_until.strftime('%H:%M:%S')))
        elif pnl_pct > 0:
            self.consec_losses = 0  # reset only on profitable exits

        logger.info("EXIT %s [%s] %s @ $%.2f | PnL: $%.2f (%.3f%%) | held %d bars | peak=$%.2f" % (
            pos.direction, pos.model_id, reason, exit_price,
            pnl_dollar, pnl_pct, bars_held, pos.peak_price))

        # Log to DB
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

    # =================================================================
    # SIGNAL LOGGING
    # =================================================================
    def log_signals(self, probs):
        """Log probabilities to console and CSV."""
        now = datetime.now()

        if self.last_signal_log_time:
            elapsed = (now - self.last_signal_log_time).total_seconds()
            if elapsed < SIGNAL_LOG_INTERVAL:
                return

        if not probs:
            return

        n_pos = len(self.positions)
        prob_str = " | ".join(["%s:%.0f%%" % (k, v * 100) for k, v in probs.items()])
        logger.info("BTC: $%.2f | Pos: %d/%d | %s" % (
            self.current_price or 0, n_pos, MAX_POSITIONS, prob_str))

        # CSV logging
        today = now.strftime('%Y-%m-%d')
        if self.signal_csv_date != today:
            self.signal_csv_date = today
            self.signal_csv_path = os.path.join(SIGNAL_LOG_DIR, 'btc_tick_reverse_signals_%s.csv' % today)
            if not os.path.exists(self.signal_csv_path):
                with open(self.signal_csv_path, 'w') as f:
                    header = 'timestamp,btc_price,' + ','.join(
                        ['prob_%s' % n for n in sorted(self.models.keys())]) + ',active_positions'
                    f.write(header + '\n')

        try:
            with open(self.signal_csv_path, 'a') as f:
                vals = [now.strftime('%Y-%m-%d %H:%M:%S'), "%.2f" % (self.current_price or 0)]
                for n in sorted(self.models.keys()):
                    vals.append("%.4f" % probs.get(n, 0))
                vals.append(str(n_pos))
                f.write(','.join(vals) + '\n')
        except Exception as e:
            logger.debug("Signal log error: %s" % e)

        self.last_signal_log_time = now

    # =================================================================
    # MAIN LOOP
    # =================================================================
    def run(self):
        """Main bot loop."""
        # Connect to IB
        if not self.connect_ib():
            logger.error("Cannot start without IB connection")
            return

        # Load historical ticks for warmup
        self.load_historical_ticks()

        # Start WebSocket
        self.start_websocket()

        logger.info("Waiting for WebSocket price data...")
        for _ in range(30):
            if self.current_price:
                break
            time.sleep(1)

        if not self.current_price:
            logger.error("No price data after 30s — check WebSocket")
            return

        logger.info("Starting main loop. Price: $%.2f" % self.current_price)

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

                # Check exits (every tick — high priority)
                self.check_exits()

                # Check signals and entries (every 16s)
                if self.last_signal_check is None or (now - self.last_signal_check).total_seconds() >= SIGNAL_CHECK_INTERVAL:
                    # Market hours guard — CME MBT: Sun 5pm–Fri 4pm CT, daily 4-5pm break
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
                    if feat is not None and len(feat) > 0:
                        probs = self.get_probabilities(feat)
                        self.check_entry(probs, feat=feat)
                        self.log_signals(probs)
                    self.last_signal_check = now

                # IB event processing
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

        # Cleanup
        if self.ws:
            self.ws.close()
        if self.ib.isConnected():
            self.ib.disconnect()
        logger.info("Bot stopped.")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='BTC Tick ML Bot')
    parser.add_argument('--paper', action='store_true', default=True, help='Paper trading (default)')
    parser.add_argument('--live', action='store_true', help='Live trading')
    args = parser.parse_args()

    paper = not args.live

    bot = BTCTickBot(paper_trading=paper)
    bot.run()

if __name__ == "__main__":
    main()
