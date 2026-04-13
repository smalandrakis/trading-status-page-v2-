"""
BTC Micro-Movement Bot - 0.3% TP / 0.1% SL with Adaptive Learning

MICRO-MOVEMENT CONFIGURATION:
- TP/SL: 0.3% / 0.1% (3:1 R:R, breakeven WR=38.7%)
- Position sizing: 3x-6x based on confidence
- Thresholds: LONG=0.50, SHORT=0.30 (lower than swing to increase frequency)

Expected: 40-50% WR, $2-15/trade, 15-20 trades/week

Data: Binance (real-time)
Execution: IB Gateway port 4002
Contract: MBT (Micro Bitcoin, 0.1 BTC)

Usage: python3 btc_micro_bot.py
"""

import time
from datetime import datetime
from binance.client import Client
import pandas as pd
import sys
import os
import logging
import json
import sqlite3

BOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BOT_DIR)

from btc_model_package.predictor import BTCPredictor
from ib_insync import IB, Future, MarketOrder, Order
from market_hours import is_market_open, get_next_open_time

# Setup logging
LOG_FILE = os.path.join(BOT_DIR, 'logs', 'btc_micro_bot.log')
TRADE_LOG_FILE = os.path.join(BOT_DIR, 'logs', 'btc_trades_micro.jsonl')
FEEDBACK_DB = os.path.join(BOT_DIR, 'trade_feedback_micro.db')
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIG - MICRO-MOVEMENT
# =============================================================================
BINANCE_API_KEY = ""
BINANCE_API_SECRET = ""

# IB Gateway
IB_HOST = '127.0.0.1'
IB_PORT = 4002
IB_CLIENT_ID = 420  # NEW CLIENT ID for micro bot

# Model settings - LOWER THRESHOLDS for higher frequency
LONG_THRESHOLD = 0.50  # vs 0.65 for swing
SHORT_THRESHOLD = 0.30  # vs 0.25 for swing

# MICRO-MOVEMENT TP/SL
TP_PCT = 0.3  # 0.3% (vs 1.0% for HF)
SL_PCT = 0.1  # 0.1% (vs 0.5% for HF)

# Position sizing - HIGHER MINIMUM for commissions
MIN_SIZE = 3  # 3x minimum (vs 1x for swing)
MAX_SIZE = 6  # 6x maximum (vs 5x for swing)
SIZING_BASE = 0.45  # Lower base for higher min size
SIZING_FACTOR = 15  # size = (confidence - 0.45) × 15

# Trading
CHECK_INTERVAL = 60  # Check every minute (vs 2min for swing)
CONTRACT_SIZE = 0.1  # MBT = 0.1 BTC per contract
MAX_DAILY_TRADES = 20  # Higher frequency expected

# =============================================================================
# FEEDBACK DATABASE
# =============================================================================
def init_feedback_db():
    """Initialize SQLite database for adaptive learning"""
    conn = sqlite3.connect(FEEDBACK_DB)
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS trades_micro (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        direction TEXT,
        entry_price REAL,
        exit_price REAL,
        size INTEGER,
        confidence REAL,
        pnl_dollars REAL,
        pnl_pct REAL,
        outcome TEXT,
        hold_minutes REAL,
        mfe REAL,
        mae REAL,
        feature_snapshot TEXT,
        created_at TEXT
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS feedback_adjustments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        adjustment_type TEXT,
        old_value REAL,
        new_value REAL,
        reason TEXT,
        trades_analyzed INTEGER
    )''')

    conn.commit()
    conn.close()
    logger.info("✓ Feedback database initialized")

def log_trade_to_db(trade_data):
    """Log trade to feedback database"""
    try:
        conn = sqlite3.connect(FEEDBACK_DB)
        c = conn.cursor()
        c.execute('''INSERT INTO trades_micro
                    (timestamp, direction, entry_price, exit_price, size, confidence,
                     pnl_dollars, pnl_pct, outcome, hold_minutes, mfe, mae, feature_snapshot, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                 (trade_data['timestamp'], trade_data['direction'], trade_data['entry_price'],
                  trade_data['exit_price'], trade_data['size'], trade_data['confidence'],
                  trade_data['pnl_dollars'], trade_data['pnl_pct'], trade_data['outcome'],
                  trade_data['hold_minutes'], trade_data.get('mfe', 0), trade_data.get('mae', 0),
                  trade_data.get('feature_snapshot', '{}'), datetime.now().isoformat()))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to log trade to DB: {e}")

def analyze_feedback():
    """Analyze recent trades and suggest adjustments"""
    try:
        conn = sqlite3.connect(FEEDBACK_DB)
        df = pd.read_sql_query("SELECT * FROM trades_micro ORDER BY id DESC LIMIT 100", conn)
        conn.close()

        if len(df) < 50:
            return  # Need at least 50 trades

        # Calculate stats
        win_rate = (df['pnl_dollars'] > 0).mean()
        avg_pnl = df['pnl_dollars'].mean()
        avg_mfe = df['mfe'].mean()
        avg_mae = df['mae'].mean()

        logger.info(f"\n{'='*80}")
        logger.info(f"ADAPTIVE FEEDBACK ANALYSIS (Last {len(df)} trades)")
        logger.info(f"{'='*80}")
        logger.info(f"Win Rate: {win_rate*100:.1f}%")
        logger.info(f"Avg P&L: ${avg_pnl:+.2f}")
        logger.info(f"Avg MFE: {avg_mfe:.3f}%")
        logger.info(f"Avg MAE: {avg_mae:.3f}%")

        # Suggest adjustments
        if win_rate < 0.35:
            logger.warning(f"⚠️ Win rate below 35% - consider raising thresholds")
        if avg_mfe > (TP_PCT * 1.5):
            logger.info(f"💡 MFE {avg_mfe:.3f}% >> TP {TP_PCT:.1f}% - consider wider TP")
        if avg_mae < (SL_PCT * 0.8):
            logger.info(f"💡 MAE {avg_mae:.3f}% << SL {SL_PCT:.1f}% - consider tighter SL")

        logger.info(f"{'='*80}\n")

    except Exception as e:
        logger.error(f"Feedback analysis error: {e}")

# =============================================================================
# POSITION SIZING - MICRO-MOVEMENT FORMULA
# =============================================================================
def calculate_position_size(confidence):
    """
    MICRO-MOVEMENT position sizing (higher minimum for commissions)
    Formula: size = (confidence - 0.45) × 15
    Capped between 3x (minimum) and 6x (maximum)
    """
    size = (confidence - SIZING_BASE) * SIZING_FACTOR
    return int(max(MIN_SIZE, min(MAX_SIZE, size)))

# =============================================================================
# IB CONTRACT SETUP
# =============================================================================
def create_btc_contract():
    """Create BTC futures contract (MBT - Micro Bitcoin)"""
    contract = Future(
        symbol='MBT',
        lastTradeDateOrContractMonth='20260424',
        exchange='CME',
        currency='USD',
        multiplier='0.1'
    )
    return contract

# =============================================================================
# INITIALIZE
# =============================================================================
logger.info("=" * 80)
logger.info("BTC MICRO-MOVEMENT BOT - ADAPTIVE LEARNING")
logger.info("=" * 80)
logger.info(f"TP/SL: {TP_PCT}% / {SL_PCT}% (MICRO-MOVEMENT - 3:1 R:R)")
logger.info(f"LONG Threshold:  {LONG_THRESHOLD}")
logger.info(f"SHORT Threshold: {SHORT_THRESHOLD}")
logger.info(f"Position Scaling: {MIN_SIZE}x - {MAX_SIZE}x (higher minimum for commissions)")
logger.info(f"Formula: size = (confidence - {SIZING_BASE}) × {SIZING_FACTOR}")
logger.info(f"Client ID: {IB_CLIENT_ID}")
logger.info(f"Check Interval: {CHECK_INTERVAL}s (faster for micro-movements)")
logger.info("")
logger.info("ADAPTIVE FEATURES:")
logger.info("  - MFE/MAE tracking for every trade")
logger.info("  - Automatic feedback analysis every 50 trades")
logger.info("  - Threshold and TP/SL suggestions")
logger.info("=" * 80)

# Initialize feedback database
init_feedback_db()

# Load predictor
logger.info("")
logger.info("Loading V3 models...")
predictor = BTCPredictor()
predictor.LONG_THRESHOLD = LONG_THRESHOLD
predictor.SHORT_THRESHOLD = SHORT_THRESHOLD
logger.info(f"✓ Models loaded: {len(predictor.models)} horizons, {len(predictor.features)} features")

# Initialize Binance client
logger.info("")
logger.info("Connecting to Binance (signal generation)...")
binance = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
logger.info("✓ Binance connected")

# Initialize IB Gateway
logger.info(f"")
logger.info(f"Connecting to IB Gateway ({IB_HOST}:{IB_PORT})...")
ib = IB()
btc_contract = None

def connect_ib():
    """Connect to IB Gateway with retry logic"""
    global ib, btc_contract
    max_retries = 10
    retry_delay = 30

    for attempt in range(max_retries):
        try:
            if ib.isConnected():
                return True

            logger.info(f"Connecting to IB Gateway (attempt {attempt + 1}/{max_retries})...")
            ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID, timeout=20)

            # Qualify contract
            btc_contract = create_btc_contract()
            ib.qualifyContracts(btc_contract)

            logger.info("✓ IB Gateway connected")
            return True

        except Exception as e:
            logger.error(f"✗ Connection failed: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error("Max retries reached. Waiting 60s before trying again...")
                time.sleep(60)
                return connect_ib()

    return False

def check_ib_connection():
    """Check IB connection and reconnect if needed"""
    global ib, btc_contract
    try:
        if not ib.isConnected():
            logger.warning("IB Gateway disconnected! Reconnecting...")
            return connect_ib()
        return True
    except:
        logger.warning("IB connection check failed! Reconnecting...")
        return connect_ib()

# Initial connection
connect_ib()

# Track current position
current_position = None
daily_trade_count = 0
last_trade_date = None

# =============================================================================
# TRADING FUNCTIONS
# =============================================================================
def get_btc_data():
    """Fetch latest 250 bars of 5-min BTC data from Binance"""
    max_retries = 5
    retry_delays = [2, 5, 10, 20, 30]

    for attempt in range(max_retries):
        try:
            klines = binance.get_klines(
                symbol='BTCUSDT',
                interval=Client.KLINE_INTERVAL_5MINUTE,
                limit=250
            )

            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])

            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df[['open', 'high', 'low', 'close', 'volume']]

            return df

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = retry_delays[attempt]
                logger.warning(f"Binance API error (attempt {attempt + 1}/{max_retries}): {e}")
                logger.warning(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to fetch Binance data after {max_retries} attempts: {e}")
                raise

def get_ib_current_price():
    """Get current IB price"""
    try:
        bars = ib.reqHistoricalData(
            btc_contract,
            endDateTime='',
            durationStr='60 S',
            barSizeSetting='1 min',
            whatToShow='TRADES',
            useRTH=False
        )
        if bars:
            return bars[-1].close
        return None
    except Exception as e:
        logger.error(f"Error getting IB price: {e}")
        return None

def get_ib_position():
    """Query IB Gateway for actual current position"""
    try:
        positions = ib.positions()
        for pos in positions:
            if pos.contract.symbol == 'MBT':
                return {
                    'quantity': pos.position,
                    'avg_cost': pos.avgCost
                }
        return None
    except Exception as e:
        logger.error(f"Error checking IB position: {e}")
        return None

def place_market_order(direction, quantity):
    """Place market order and return filled trade"""
    try:
        action = 'BUY' if direction == 'LONG' else 'SELL'
        order = MarketOrder(action, quantity)
        trade = ib.placeOrder(btc_contract, order)
        ib.sleep(2)

        if trade.orderStatus.status in ['Filled', 'PreSubmitted', 'Submitted']:
            return trade
        else:
            logger.error(f"Order failed: {trade.orderStatus.status}")
            return None
    except Exception as e:
        logger.error(f"Error placing market order: {e}")
        return None

def place_stop_loss_bracket(direction, quantity, sl_price):
    """Place SL bracket order as safety net"""
    try:
        action = 'SELL' if direction == 'LONG' else 'BUY'
        sl_order = Order()
        sl_order.action = action
        sl_order.orderType = 'STP'
        sl_order.auxPrice = sl_price
        sl_order.totalQuantity = quantity
        sl_trade = ib.placeOrder(btc_contract, sl_order)
        return sl_trade
    except Exception as e:
        logger.error(f"Error placing SL bracket: {e}")
        return None

def log_trade(event_type, data):
    """Log trade events to JSONL file"""
    try:
        with open(TRADE_LOG_FILE, 'a') as f:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'event': event_type,
                'data': data
            }
            f.write(json.dumps(log_entry) + '\n')
    except Exception as e:
        logger.error(f"Failed to log trade: {e}")

def execute_entry(signal, binance_price, position_size, confidence, details):
    """Execute entry via IB Gateway"""
    global daily_trade_count
    quantity = int(position_size)
    total_btc = quantity * CONTRACT_SIZE

    logger.info(f"\n{'='*80}")
    logger.info(f"📈 MICRO-ENTRY SIGNAL: {signal}")
    logger.info(f"Binance Signal Price: ${binance_price:,.2f}")
    logger.info(f"Confidence: {confidence:.1%}")
    logger.info(f"Position Size: {quantity} contracts ({total_btc:.1f} BTC)")
    logger.info(f"{'='*80}\n")

    # Place market order
    trade = place_market_order(signal, quantity)
    if not trade:
        return None

    # Get actual IB fill price
    ib.sleep(1)
    fill_price = trade.orderStatus.avgFillPrice

    # Calculate TP/SL based on IB fill price with MICRO percentages
    if signal == 'LONG':
        tp = fill_price * (1 + TP_PCT / 100)
        sl = fill_price * (1 - SL_PCT / 100)
    else:  # SHORT
        tp = fill_price * (1 - TP_PCT / 100)
        sl = fill_price * (1 + SL_PCT / 100)

    logger.info(f"✓ Entry filled at IB price: ${fill_price:,.2f}")
    logger.info(f"  TP: ${tp:,.2f} (+{TP_PCT}%)")
    logger.info(f"  SL: ${sl:,.2f} (-{SL_PCT}%)")
    logger.info(f"  Price diff (Binance-IB): ${binance_price - fill_price:,.2f}")

    # Place SL bracket as safety net
    sl_trade = place_stop_loss_bracket(signal, quantity, sl)

    # Log entry
    log_trade('ENTRY_SIGNAL', {
        'direction': signal,
        'confidence': confidence,
        'binance_price': binance_price,
        'ib_fill_price': fill_price,
        'price_diff': binance_price - fill_price,
        'tp_price': tp,
        'sl_price': sl,
        'position_size': quantity,
        'tp_pct': TP_PCT,
        'sl_pct': SL_PCT,
        'predictor_details': details
    })

    log_trade('ENTRY_EXECUTED', {
        'direction': signal,
        'ib_fill_price': fill_price,
        'tp_price': tp,
        'sl_price': sl,
        'position_size': quantity
    })

    daily_trade_count += 1

    return {
        'direction': signal,
        'entry_ib': fill_price,
        'tp_ib': tp,
        'sl_ib': sl,
        'size': quantity,
        'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'confidence': confidence,
        'sl_trade': sl_trade,
        'price_history': [],  # Track for MFE/MAE
        'entry_timestamp': datetime.now()
    }

def calculate_mfe_mae(position, exit_price):
    """Calculate Max Favorable/Adverse Excursion"""
    entry_price = position['entry_ib']
    direction = position['direction']
    price_history = position.get('price_history', [])

    if not price_history:
        return 0, 0

    prices = [p for p in price_history if p > 0]
    if not prices:
        return 0, 0

    if direction == 'LONG':
        mfe = (max(prices) / entry_price - 1) * 100  # Max favorable
        mae = (min(prices) / entry_price - 1) * 100  # Max adverse
    else:  # SHORT
        mfe = (entry_price / min(prices) - 1) * 100
        mae = (entry_price / max(prices) - 1) * 100

    return mfe, mae

def execute_exit(position, exit_reason, exit_price):
    """Execute exit via IB Gateway"""
    # Prevent duplicate exit logging (IB partial fills create multiple exit events)
    if position.get('exit_logged', False):
        logger.debug(f"Exit already logged for this position, skipping duplicate")
        return True

    entry_price = position['entry_ib']
    direction = position['direction']
    quantity = position['size']
    entry_time = position['time']
    confidence = position['confidence']

    if direction == 'LONG':
        pnl_pct = (exit_price / entry_price - 1) * 100
    else:
        pnl_pct = (entry_price / exit_price - 1) * 100

    pnl_dollar = pnl_pct / 100 * entry_price * CONTRACT_SIZE * quantity

    hold_duration = (datetime.now() - datetime.strptime(entry_time, "%Y-%m-%d %H:%M:%S")).total_seconds() / 60

    # Calculate MFE/MAE
    mfe, mae = calculate_mfe_mae(position, exit_price)

    logger.info(f"\n{'='*80}")
    logger.info(f"🚪 MICRO-EXIT: {exit_reason}")
    logger.info(f"Entry:  ${entry_price:,.2f} @ {entry_time}")
    logger.info(f"Exit:   ${exit_price:,.2f} (IB price)")
    logger.info(f"Size:   {quantity} contracts")
    logger.info(f"Hold:   {hold_duration:.0f} minutes")
    logger.info(f"P&L:    {pnl_pct:+.2f}% (${pnl_dollar:+,.2f})")
    logger.info(f"MFE:    {mfe:+.3f}% | MAE: {mae:+.3f}%")
    logger.info(f"{'='*80}\n")

    # Log exit (only once per position)
    log_trade('EXIT', {
        'direction': direction,
        'exit_reason': exit_reason,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'entry_time': entry_time,
        'exit_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'position_size': quantity,
        'hold_minutes': hold_duration,
        'pnl_pct': pnl_pct,
        'pnl_dollar': pnl_dollar,
        'mfe': mfe,
        'mae': mae
    })

    # Mark exit as logged to prevent duplicates
    position['exit_logged'] = True

    # Log to feedback database
    log_trade_to_db({
        'timestamp': datetime.now().isoformat(),
        'direction': direction,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'size': quantity,
        'confidence': confidence,
        'pnl_dollars': pnl_dollar,
        'pnl_pct': pnl_pct,
        'outcome': exit_reason,
        'hold_minutes': hold_duration,
        'mfe': mfe,
        'mae': mae
    })

    # Close position
    action = 'SELL' if direction == 'LONG' else 'BUY'
    order = MarketOrder(action, quantity)
    trade = ib.placeOrder(btc_contract, order)
    ib.sleep(1)

    # Cancel SL bracket
    if 'sl_trade' in position and position['sl_trade']:
        try:
            ib.cancelOrder(position['sl_trade'].order)
        except:
            pass

    return True

# =============================================================================
# MAIN LOOP
# =============================================================================
logger.info("")
logger.info("Starting micro-movement loop (Ctrl+C to stop)...")
logger.info(f"Checking every {CHECK_INTERVAL} seconds")
logger.info(f"Log file: {LOG_FILE}")
logger.info(f"Feedback DB: {FEEDBACK_DB}")
logger.info("")
logger.info(f"{'Time':<20} {'Binance $':<12} {'IB $':<12} {'Signal':<8} {'Conf':<7} {'Size':<6} {'Position':<20}")
logger.info("-" * 100)

try:
    while True:
        try:
            # Check IB connection first
            if not check_ib_connection():
                logger.error("Cannot connect to IB Gateway, waiting...")
                time.sleep(60)
                continue

            # Reset daily trade count
            today = datetime.now().date()
            if last_trade_date != today:
                daily_trade_count = 0
                last_trade_date = today

            # Get latest data from Binance for signal
            df = get_btc_data()
            current_price = df['close'].iloc[-1]
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Get IB price for monitoring
            ib_price = get_ib_current_price()
            if ib_price is None:
                ib_price = current_price

            # Track price for MFE/MAE
            if current_position:
                current_position['price_history'].append(ib_price)

            # SYNC WITH IB REALITY
            if current_position:
                ib_pos = get_ib_position()

                # Position closed externally
                if not ib_pos or ib_pos['quantity'] == 0:
                    logger.info("Position closed externally - syncing state")
                    if current_position:
                        execute_exit(current_position, 'EXTERNAL_CLOSE', ib_price)
                    current_position = None

                # Position size mismatch
                elif abs(ib_pos['quantity']) != current_position['size']:
                    old_size = current_position['size']
                    new_size = abs(ib_pos['quantity'])
                    logger.warning(f"Position size mismatch! Bot: {old_size}, IB: {new_size}")
                    current_position['size'] = new_size

                    # Log position adjustment for trade record accuracy
                    log_trade('POSITION_ADJUSTED', {
                        'direction': current_position['direction'],
                        'old_size': old_size,
                        'new_size': new_size,
                        'reason': 'IB position sync (paper account auto-fill)'
                    })
                    logger.info(f"  Position size synced: {old_size} → {new_size} contracts")

            # Check manual exit conditions (bot-managed TP/SL)
            if current_position:
                direction = current_position['direction']
                tp = current_position['tp_ib']
                sl = current_position['sl_ib']

                should_exit = False
                reason = None

                if direction == 'LONG':
                    if ib_price >= tp:
                        should_exit = True
                        reason = 'TAKE_PROFIT'
                    elif ib_price <= sl:
                        should_exit = True
                        reason = 'STOP_LOSS'
                else:  # SHORT
                    if ib_price <= tp:
                        should_exit = True
                        reason = 'TAKE_PROFIT'
                    elif ib_price >= sl:
                        should_exit = True
                        reason = 'STOP_LOSS'

                if should_exit:
                    if execute_exit(current_position, reason, ib_price):
                        current_position = None
                        # Analyze feedback every 50 trades
                        conn = sqlite3.connect(FEEDBACK_DB)
                        count = pd.read_sql_query("SELECT COUNT(*) as cnt FROM trades_micro", conn).iloc[0]['cnt']
                        conn.close()
                        if count > 0 and count % 50 == 0:
                            analyze_feedback()

            # Get prediction from Binance data
            signal, confidence, details = predictor.predict(df)

            # Calculate position size based on confidence
            position_size = calculate_position_size(confidence) if signal != 'NEUTRAL' else 0

            # Format position display
            if current_position:
                pos_display = f"{current_position['direction']} {current_position['size']}x"
            else:
                pos_display = "NONE"

            logger.info(f"{current_time:<20} ${current_price:>10,.2f} ${ib_price:>10,.2f} {signal:<8} "
                       f"{confidence:>6.1%} {position_size:>5.0f}x {pos_display:<20}")

            # Enter new position if conditions met
            if not current_position and signal != 'NEUTRAL' and daily_trade_count < MAX_DAILY_TRADES:
                # Check if market is open
                if not is_market_open():
                    logger.warning(f"⏸  Signal: {signal} @ {confidence:.1%} but market is CLOSED")
                    logger.warning(f"   Market opens: {get_next_open_time()}")
                    time.sleep(CHECK_INTERVAL)
                    continue

                position = execute_entry(signal, current_price, position_size, confidence, details)
                if position:
                    current_position = position

            # Wait
            time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.error(f"ERROR: {e}")
            import traceback
            logger.error(traceback.format_exc())
            time.sleep(CHECK_INTERVAL)

except KeyboardInterrupt:
    logger.info("\n\n" + "="*80)
    logger.info("Bot stopped by user")
    if current_position:
        logger.info(f"WARNING: Position still open: {current_position['direction']} "
              f"{current_position['size']} contracts from ${current_position['entry_ib']:,.2f}")

        response = input("Close position? (y/n): ")
        if response.lower() == 'y':
            ib_price = get_ib_current_price()
            if ib_price:
                execute_exit(current_position, 'MANUAL_CLOSE', ib_price)

    logger.info("Disconnecting from IB Gateway...")
    ib.disconnect()
    logger.info("="*80)
