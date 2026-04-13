#!/usr/bin/env python3
"""
MNQ Global Signal Bot
=====================
One trade per day on Micro Nasdaq (MNQ) futures.
Uses Asia/Europe session returns + macro features to predict NQ direction.

Strategy:
  - At ~10:25 ET: Download Asia/Europe/macro data, compute RF prediction
  - At  10:30 ET: Enter LONG (prob>55%) or SHORT (prob<45%), or skip
  - TP: 1.0%, SL: 0.5% (2:1 R:R)
  - Close by 15:55 ET if neither hit
  - Max 1 trade per day

Model: RandomForest trained on 38 features (global indices, VIX, bonds, USD/JPY, etc.)
Backtest: 70.6% WR, PF 3.36, +$5,557/MNQ over 3-month holdout

IB Gateway connection on port 4002.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import json
import os
import time
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, Dict
from dataclasses import dataclass, asdict
from ib_insync import IB, Future, MarketOrder, util
import pytz

# ─── LOGGING ──────────────────────────────────────────────────────────
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/mnq_global_signal.log', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ─── CONFIG ───────────────────────────────────────────────────────────
# IB
IB_HOST = '127.0.0.1'
IB_PORT = 4002
IB_CLIENT_ID = 50  # Unique ID for this bot (BTC=400, tick=40)

# Strategy
RF_PROB_THRESHOLD = 0.55   # LONG if prob > 55%, SHORT if prob < 45%
TP_PCT = 1.0               # Take profit: 1.0%
SL_PCT = 0.5               # Stop loss: 0.5%
POSITION_SIZE = 1           # 1 MNQ contract ($2/point)

# Schedule (Eastern Time)
ENTRY_HOUR = 10
ENTRY_MINUTE = 30
EXIT_HOUR = 15
EXIT_MINUTE = 55
FEATURE_PREP_MINUTES = 5    # Start preparing features 5 min before entry

# Model
MODEL_DIR = "models_mnq_global"
RF_MODEL_PATH = os.path.join(MODEL_DIR, "rf_global_signal.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_global_signal.pkl")
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "feature_cols.pkl")

# Data
MNQ_DATA_FILE = os.path.join("data", "MNQ_5min_IB_with_indicators.csv")
POSITION_FILE = "mnq_global_position.json"
TRADE_DB = "mnq_global_trades.db"

# Tickers
ASIA_TICKERS = ['^N225', '^HSI', '^AXJO']
EUROPE_TICKERS = ['^GDAXI', '^FTSE', '^STOXX50E']
MACRO_TICKERS = ['^VIX', 'JPY=X', 'TLT', 'NQ=F', 'ES=F', 'CL=F']

ET = pytz.timezone('America/New_York')


# ─── POSITION ─────────────────────────────────────────────────────────
@dataclass
class Position:
    direction: str        # LONG or SHORT
    entry_price: float
    entry_time: str
    tp_price: float
    sl_price: float
    rf_prob: float
    size: int = 1
    order_id: Optional[int] = None
    trade_date: str = ""

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ─── TRADE DATABASE ──────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(TRADE_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_date TEXT,
            direction TEXT,
            entry_price REAL,
            entry_time TEXT,
            exit_price REAL,
            exit_time TEXT,
            exit_reason TEXT,
            pnl_pct REAL,
            pnl_points REAL,
            pnl_dollars REAL,
            rf_prob REAL,
            features TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_trade(trade_date, direction, entry_price, entry_time, exit_price,
              exit_time, exit_reason, pnl_pct, pnl_points, pnl_dollars,
              rf_prob, features_json=""):
    conn = sqlite3.connect(TRADE_DB)
    conn.execute("""
        INSERT INTO trades (trade_date, direction, entry_price, entry_time,
            exit_price, exit_time, exit_reason, pnl_pct, pnl_points,
            pnl_dollars, rf_prob, features)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (trade_date, direction, entry_price, entry_time, exit_price,
          exit_time, exit_reason, pnl_pct, pnl_points, pnl_dollars,
          rf_prob, features_json))
    conn.commit()
    conn.close()


# ─── MNQ BOT ─────────────────────────────────────────────────────────
class MNQGlobalSignalBot:

    def __init__(self, paper_trading=True):
        self.paper_trading = paper_trading
        self.ib = IB()
        self.connected = False
        self.position: Optional[Position] = None
        self.today_traded = False
        self.last_trade_date = None
        self.current_price = None

        # Load ML model
        logger.info("Loading RF model...")
        self.rf_model = joblib.load(RF_MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self.feature_cols = joblib.load(FEATURE_COLS_PATH)
        logger.info(f"  RF model loaded: {len(self.feature_cols)} features")

        # MNQ contract (front month)
        self.contract = None

        # Feature cache
        self.daily_features = None
        self.rf_prob = None

        # Load existing position
        self._load_position()

        # Init trade DB
        init_db()

        logger.info("=" * 60)
        logger.info("MNQ GLOBAL SIGNAL BOT INITIALIZED")
        logger.info("=" * 60)
        logger.info(f"  Model:     RF (prob threshold: {RF_PROB_THRESHOLD:.0%})")
        logger.info(f"  TP/SL:     {TP_PCT}% / {SL_PCT}% (R:R = {TP_PCT/SL_PCT:.0f}:1)")
        logger.info(f"  Size:      {POSITION_SIZE} MNQ contract(s)")
        logger.info(f"  Entry:     {ENTRY_HOUR}:{ENTRY_MINUTE:02d} ET")
        logger.info(f"  Exit:      {EXIT_HOUR}:{EXIT_MINUTE:02d} ET")
        logger.info(f"  Paper:     {paper_trading}")

    # ── IB CONNECTION ─────────────────────────────────────────────────

    def connect(self):
        """Connect to IB Gateway."""
        if self.connected:
            return True
        try:
            logger.info(f"Connecting to IB Gateway ({IB_HOST}:{IB_PORT}, clientId={IB_CLIENT_ID})...")
            self.ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID)
            self.connected = True
            logger.info("  Connected to IB Gateway")

            # Qualify MNQ contract
            mnq = Future("MNQ", exchange="CME")
            details = self.ib.reqContractDetails(mnq)
            if details:
                self.contract = details[0].contract
                logger.info(f"  MNQ contract: {self.contract.localSymbol} "
                           f"(expiry: {self.contract.lastTradeDateOrContractMonth})")
            else:
                logger.error("  Could not find MNQ contract!")
                return False
            return True
        except Exception as e:
            logger.error(f"  IB connection failed: {e}")
            self.connected = False
            return False

    def disconnect(self):
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            logger.info("Disconnected from IB")

    def reconnect(self):
        """Reconnect if disconnected."""
        if not self.ib.isConnected():
            self.connected = False
            logger.warning("IB disconnected — reconnecting...")
            time.sleep(5)
            return self.connect()
        return True

    # ── PRICE ─────────────────────────────────────────────────────────

    def get_price(self) -> Optional[float]:
        """Get current MNQ price from IB using historical data (no market data subscription needed)."""
        if not self.connected or self.contract is None:
            return None
        try:
            bars = self.ib.reqHistoricalData(
                self.contract,
                endDateTime='',
                durationStr='60 S',
                barSizeSetting='1 min',
                whatToShow='TRADES',
                useRTH=False,
                formatDate=1,
            )
            if bars:
                price = bars[-1].close
                if price and price > 0:
                    self.current_price = price
                    return price
            return None
        except Exception as e:
            logger.error(f"Error getting price: {e}")
            return None

    # ── ORDERS ────────────────────────────────────────────────────────

    def place_order(self, direction: str, size: int) -> Optional[Tuple[int, float]]:
        """Place a market order via IB. Returns (orderId, fillPrice)."""
        if not self.connected:
            logger.error("Cannot place order — not connected")
            return None

        action = "BUY" if direction == "LONG" else "SELL"
        order = MarketOrder(action, size)

        if self.paper_trading:
            logger.info(f"[PAPER] Would place {action} {size} MNQ @ market")
            # Use current price as simulated fill
            fill_price = self.current_price or 0
            return (0, fill_price)

        try:
            trade = self.ib.placeOrder(self.contract, order)
            logger.info(f"Order placed: {action} {size} MNQ")

            # Wait for fill (up to 30 seconds)
            for _ in range(30):
                self.ib.sleep(1)
                if trade.orderStatus.status == 'Filled':
                    fill_price = trade.orderStatus.avgFillPrice
                    logger.info(f"  Filled @ ${fill_price:.2f}")
                    return (trade.order.orderId, fill_price)

            logger.warning(f"  Order not filled after 30s, status: {trade.orderStatus.status}")
            # Cancel if not filled
            self.ib.cancelOrder(trade.order)
            return None
        except Exception as e:
            logger.error(f"Order error: {e}")
            return None

    def close_position(self, reason: str) -> bool:
        """Close the current position."""
        if self.position is None:
            return False

        close_dir = "SHORT" if self.position.direction == "LONG" else "LONG"
        action = "SELL" if self.position.direction == "LONG" else "BUY"

        price = self.get_price()
        if price is None:
            logger.error("Cannot close — no price")
            return False

        result = self.place_order(close_dir, self.position.size)
        if result is None:
            logger.error("Close order failed")
            return False

        _, fill_price = result
        exit_price = fill_price if fill_price > 0 else price

        # Calculate P&L
        if self.position.direction == "LONG":
            pnl_pct = (exit_price / self.position.entry_price - 1) * 100
            pnl_points = exit_price - self.position.entry_price
        else:
            pnl_pct = (self.position.entry_price / exit_price - 1) * 100
            pnl_points = self.position.entry_price - exit_price

        pnl_dollars = pnl_points * 2.0 * self.position.size  # MNQ = $2/point

        # Log trade
        now_et = datetime.now(ET)
        log_trade(
            trade_date=self.position.trade_date,
            direction=self.position.direction,
            entry_price=self.position.entry_price,
            entry_time=self.position.entry_time,
            exit_price=exit_price,
            exit_time=now_et.strftime('%Y-%m-%d %H:%M:%S'),
            exit_reason=reason,
            pnl_pct=pnl_pct,
            pnl_points=pnl_points,
            pnl_dollars=pnl_dollars,
            rf_prob=self.position.rf_prob,
        )

        logger.info("=" * 60)
        logger.info(f"EXIT [{reason}]: {self.position.direction} @ ${exit_price:.2f}")
        logger.info(f"  Entry: ${self.position.entry_price:.2f}")
        logger.info(f"  PnL:   {pnl_pct:+.3f}% ({pnl_points:+.2f} pts, ${pnl_dollars:+.2f})")
        logger.info("=" * 60)

        self.position = None
        self._save_position()
        return True

    # ── POSITION PERSISTENCE ──────────────────────────────────────────

    def _save_position(self):
        if self.position is None:
            if os.path.exists(POSITION_FILE):
                os.remove(POSITION_FILE)
        else:
            with open(POSITION_FILE, 'w') as f:
                json.dump(self.position.to_dict(), f, indent=2)

    def _load_position(self):
        if os.path.exists(POSITION_FILE):
            try:
                with open(POSITION_FILE, 'r') as f:
                    data = json.load(f)
                self.position = Position.from_dict(data)
                logger.info(f"Loaded existing position: {self.position.direction} "
                           f"@ ${self.position.entry_price:.2f} "
                           f"(TP: ${self.position.tp_price:.2f}, SL: ${self.position.sl_price:.2f})")
            except Exception as e:
                logger.error(f"Error loading position: {e}")
                self.position = None

    # ── FEATURE ENGINEERING ───────────────────────────────────────────

    def prepare_features(self) -> Optional[Dict]:
        """Download Asia/Europe/macro data and compute today's feature vector."""
        logger.info("Preparing features for today's signal...")

        try:
            # Download external data
            ext = pd.DataFrame()
            all_tickers = ASIA_TICKERS + EUROPE_TICKERS + MACRO_TICKERS
            for ticker in all_tickers:
                col = ticker.replace('^', '').replace('=', '_')
                df = yf.download(ticker, period='30d', interval='1d', progress=False)
                if len(df) > 0:
                    if isinstance(df.columns, pd.MultiIndex):
                        close = df[('Close', ticker)]
                    else:
                        close = df['Close']
                    ext[f'{col}_close'] = close
                    ext[f'{col}_ret'] = close.pct_change()

            ext.index = ext.index.tz_localize(None) if ext.index.tz else ext.index

            # Composite signals
            asia_ret = [c for c in ext.columns if c.endswith('_ret')
                        and any(a in c for a in ['N225', 'HSI', 'AXJO'])]
            europe_ret = [c for c in ext.columns if c.endswith('_ret')
                          and any(e in c for e in ['GDAXI', 'FTSE', 'STOXX50E'])]

            if asia_ret:
                ext['asia_ret'] = ext[asia_ret].mean(axis=1)
            if europe_ret:
                ext['europe_ret'] = ext[europe_ret].mean(axis=1)
            if asia_ret and europe_ret:
                ext['combined_ret'] = 0.4 * ext['asia_ret'] + 0.6 * ext['europe_ret']

            # VIX
            if 'VIX_close' in ext:
                ext['vix_level'] = ext['VIX_close']
                ext['vix_change'] = ext['VIX_close'].pct_change()
                ext['vix_5d_avg'] = ext['VIX_close'].rolling(5).mean()
                ext['vix_above_avg'] = (ext['VIX_close'] > ext['vix_5d_avg']).astype(int)
            if 'JPY_X_ret' in ext:
                ext['usdjpy_ret'] = ext['JPY_X_ret']
                ext['usdjpy_5d'] = ext['JPY_X_ret'].rolling(5).sum()
            if 'TLT_ret' in ext:
                ext['bond_ret'] = ext['TLT_ret']
                ext['bond_5d'] = ext['TLT_ret'].rolling(5).sum()
            if 'CL_F_ret' in ext:
                ext['oil_ret'] = ext['CL_F_ret']
            if 'NQ_F_ret' in ext:
                ext['nq_prev_ret'] = ext['NQ_F_ret'].shift(1)
                ext['nq_prev_2d'] = ext['NQ_F_ret'].shift(1).rolling(2).sum()
                ext['nq_prev_5d'] = ext['NQ_F_ret'].shift(1).rolling(5).sum()
            if 'ES_F_ret' in ext:
                ext['es_ret'] = ext['ES_F_ret']
                ext['es_nq_spread'] = ext.get('NQ_F_ret', 0) - ext['ES_F_ret']

            # MNQ daily features from IB data or recent NQ proxy
            # Use NQ=F daily data for rolling features
            nq_daily = ext[['NQ_F_close', 'NQ_F_ret']].dropna().copy() if 'NQ_F_close' in ext else pd.DataFrame()
            if len(nq_daily) > 0:
                nq_daily['daily_range'] = nq_daily['NQ_F_ret'].abs()  # Approximation
                ext['daily_range'] = nq_daily['daily_range']
                ext['ret_1d'] = ext['NQ_F_ret'].shift(1)
                ext['ret_2d'] = ext['NQ_F_ret'].shift(1).rolling(2).sum()
                ext['ret_5d'] = ext['NQ_F_ret'].shift(1).rolling(5).sum()
                ext['vol_5d'] = nq_daily['daily_range'].shift(1).rolling(5).mean()
                ext['vol_10d'] = nq_daily['daily_range'].shift(1).rolling(10).mean()
            else:
                for c in ['daily_range', 'ret_1d', 'ret_2d', 'ret_5d', 'vol_5d', 'vol_10d']:
                    ext[c] = 0.0

            # Placeholder features that come from intraday data (not available pre-market)
            ext['volume_ratio'] = 1.0
            ext['prev_fh_ret'] = 0.0
            ext['prev_fh_range'] = 0.0

            # Get latest row
            latest = ext.iloc[-1]

            # Build feature dict matching training columns
            features = {}
            for col in self.feature_cols:
                if col in latest.index:
                    val = latest[col]
                    features[col] = float(val) if not pd.isna(val) else 0.0
                else:
                    features[col] = 0.0
                    logger.warning(f"  Missing feature: {col} — using 0.0")

            # Compute RF probability
            X = np.array([features[c] for c in self.feature_cols]).reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            prob = self.rf_model.predict_proba(X_scaled)[0][1]

            self.rf_prob = prob
            self.daily_features = features

            logger.info(f"  Features ready. RF probability: {prob:.1%}")
            logger.info(f"  Asia ret:    {features.get('asia_ret', 0):.4f}")
            logger.info(f"  Europe ret:  {features.get('europe_ret', 0):.4f}")
            logger.info(f"  Combined:    {features.get('combined_ret', 0):.4f}")
            logger.info(f"  VIX level:   {features.get('vix_level', 0):.1f}")
            logger.info(f"  VIX change:  {features.get('vix_change', 0):.4f}")
            logger.info(f"  USD/JPY ret: {features.get('usdjpy_ret', 0):.4f}")
            logger.info(f"  Bond ret:    {features.get('bond_ret', 0):.4f}")
            logger.info(f"  NQ prev ret: {features.get('nq_prev_ret', 0):.4f}")

            if prob > RF_PROB_THRESHOLD:
                logger.info(f"  SIGNAL: LONG (prob {prob:.1%} > {RF_PROB_THRESHOLD:.0%})")
            elif prob < (1 - RF_PROB_THRESHOLD):
                logger.info(f"  SIGNAL: SHORT (prob {prob:.1%} < {1-RF_PROB_THRESHOLD:.0%})")
            else:
                logger.info(f"  SIGNAL: SKIP (prob {prob:.1%} in neutral zone)")

            return features

        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            import traceback
            traceback.print_exc()
            return None

    # ── ENTRY LOGIC ───────────────────────────────────────────────────

    def check_entry(self):
        """Check if we should enter a trade at 10:30 ET."""
        if self.position is not None:
            logger.debug("Already in position — skipping entry check")
            return

        now_et = datetime.now(ET)
        today_str = now_et.strftime('%Y-%m-%d')

        if self.last_trade_date == today_str:
            logger.debug("Already traded today — skipping")
            return

        if self.rf_prob is None:
            logger.warning("No RF probability — features not prepared yet")
            return

        # Determine direction
        prob = self.rf_prob
        if prob > RF_PROB_THRESHOLD:
            direction = "LONG"
        elif prob < (1 - RF_PROB_THRESHOLD):
            direction = "SHORT"
        else:
            logger.info(f"No trade today: prob {prob:.1%} in neutral zone "
                       f"({1-RF_PROB_THRESHOLD:.0%}-{RF_PROB_THRESHOLD:.0%})")
            self.last_trade_date = today_str
            return

        # Get price
        price = self.get_price()
        if price is None:
            logger.error("Cannot enter — no price available")
            return

        # Calculate TP/SL
        if direction == "LONG":
            tp_price = price * (1 + TP_PCT / 100)
            sl_price = price * (1 - SL_PCT / 100)
        else:
            tp_price = price * (1 - TP_PCT / 100)
            sl_price = price * (1 + SL_PCT / 100)

        # Place order
        result = self.place_order(direction, POSITION_SIZE)
        if result is None:
            logger.error("Entry order failed")
            return

        order_id, fill_price = result
        entry_price = fill_price if fill_price > 0 else price

        # Recalculate TP/SL from actual fill
        if direction == "LONG":
            tp_price = entry_price * (1 + TP_PCT / 100)
            sl_price = entry_price * (1 - SL_PCT / 100)
        else:
            tp_price = entry_price * (1 - TP_PCT / 100)
            sl_price = entry_price * (1 + SL_PCT / 100)

        self.position = Position(
            direction=direction,
            entry_price=entry_price,
            entry_time=now_et.strftime('%Y-%m-%d %H:%M:%S'),
            tp_price=tp_price,
            sl_price=sl_price,
            rf_prob=prob,
            size=POSITION_SIZE,
            order_id=order_id,
            trade_date=today_str,
        )
        self._save_position()
        self.last_trade_date = today_str

        logger.info("=" * 60)
        logger.info(f"ENTRY: {direction} {POSITION_SIZE} MNQ @ ${entry_price:.2f}")
        logger.info(f"  RF prob:  {prob:.1%}")
        logger.info(f"  TP:       ${tp_price:.2f} ({'+' if direction=='LONG' else '-'}{TP_PCT}%)")
        logger.info(f"  SL:       ${sl_price:.2f} ({'-' if direction=='LONG' else '+'}{SL_PCT}%)")
        logger.info(f"  Close by: {EXIT_HOUR}:{EXIT_MINUTE:02d} ET")
        logger.info("=" * 60)

    # ── EXIT LOGIC ────────────────────────────────────────────────────

    def check_exit(self):
        """Check TP/SL/time exit on current position."""
        if self.position is None:
            return

        price = self.get_price()
        if price is None:
            logger.warning("No price for exit check")
            return

        pos = self.position

        # Check TP
        if pos.direction == "LONG":
            if price >= pos.tp_price:
                self.close_position("TP")
                return
            if price <= pos.sl_price:
                self.close_position("SL")
                return
        else:  # SHORT
            if price <= pos.tp_price:
                self.close_position("TP")
                return
            if price >= pos.sl_price:
                self.close_position("SL")
                return

        # Time exit at 15:55 ET
        now_et = datetime.now(ET)
        if now_et.hour > EXIT_HOUR or (now_et.hour == EXIT_HOUR and now_et.minute >= EXIT_MINUTE):
            self.close_position("TIME")
            return

    # ── MAIN LOOP ─────────────────────────────────────────────────────

    def run(self):
        """Main bot loop."""
        logger.info("Starting MNQ Global Signal Bot...")

        if not self.connect():
            logger.error("Failed to connect to IB. Exiting.")
            return

        features_prepared_today = False

        try:
            while True:
                now_et = datetime.now(ET)
                today_str = now_et.strftime('%Y-%m-%d')
                weekday = now_et.weekday()

                # Skip weekends
                if weekday >= 5:
                    logger.debug("Weekend — sleeping 60s")
                    time.sleep(60)
                    continue

                # Reset daily state
                if self.last_trade_date != today_str:
                    features_prepared_today = False

                # ── PRE-MARKET: Prepare features at 10:25 ET ──
                if (not features_prepared_today
                    and now_et.hour == ENTRY_HOUR
                    and now_et.minute >= (ENTRY_MINUTE - FEATURE_PREP_MINUTES)
                    and now_et.minute < ENTRY_MINUTE):

                    logger.info(f"[{now_et.strftime('%H:%M')} ET] Preparing features...")
                    self.prepare_features()
                    features_prepared_today = True

                # ── ENTRY at 10:30 ET ──
                if (now_et.hour == ENTRY_HOUR
                    and now_et.minute == ENTRY_MINUTE
                    and self.position is None
                    and self.last_trade_date != today_str):

                    if not features_prepared_today:
                        logger.info("Features not ready — preparing now...")
                        self.prepare_features()
                        features_prepared_today = True

                    self.check_entry()

                # ── MONITOR: Check TP/SL every 30s while in position ──
                if self.position is not None:
                    # Only check during RTH
                    if 9 <= now_et.hour < 16:
                        if not self.reconnect():
                            time.sleep(10)
                            continue
                        self.check_exit()

                # ── SLEEP ──
                if self.position is not None:
                    # Active position: check every 30 seconds
                    time.sleep(30)
                elif (now_et.hour == ENTRY_HOUR
                      and now_et.minute >= (ENTRY_MINUTE - FEATURE_PREP_MINUTES)
                      and now_et.minute <= ENTRY_MINUTE):
                    # Near entry time: check every 10 seconds
                    time.sleep(10)
                else:
                    # Idle: check every 60 seconds
                    time.sleep(60)

        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Bot error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Close any open position before shutdown
            if self.position is not None:
                logger.warning("Shutting down with open position — closing...")
                self.close_position("SHUTDOWN")
            self.disconnect()


# ─── MAIN ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='MNQ Global Signal Bot')
    parser.add_argument('--live', action='store_true', help='Enable live trading (default: paper)')
    parser.add_argument('--test-signal', action='store_true', help='Prepare features and print signal, then exit')
    args = parser.parse_args()

    bot = MNQGlobalSignalBot(paper_trading=not args.live)

    if args.test_signal:
        # Just prepare features and print the signal
        bot.connect()
        features = bot.prepare_features()
        if features:
            print(f"\nToday's signal:")
            print(f"  RF probability: {bot.rf_prob:.1%}")
            if bot.rf_prob > RF_PROB_THRESHOLD:
                print(f"  Direction: LONG")
            elif bot.rf_prob < (1 - RF_PROB_THRESHOLD):
                print(f"  Direction: SHORT")
            else:
                print(f"  Direction: SKIP (neutral)")

            price = bot.get_price()
            if price:
                print(f"  MNQ price: ${price:.2f}")
                if bot.rf_prob > RF_PROB_THRESHOLD:
                    print(f"  TP: ${price * (1 + TP_PCT/100):.2f} (+{TP_PCT}%)")
                    print(f"  SL: ${price * (1 - SL_PCT/100):.2f} (-{SL_PCT}%)")
                elif bot.rf_prob < (1 - RF_PROB_THRESHOLD):
                    print(f"  TP: ${price * (1 - TP_PCT/100):.2f} (-{TP_PCT}%)")
                    print(f"  SL: ${price * (1 + SL_PCT/100):.2f} (+{SL_PCT}%)")
        bot.disconnect()
    else:
        bot.run()
