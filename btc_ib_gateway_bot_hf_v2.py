"""
BTC High Frequency Bot V2 - Bot-Managed TP/SL (No Bracket Orders)

NEW ARCHITECTURE:
- Signals: Binance (ML model trained on spot prices)
- Execution: IB Gateway (CME futures)
- TP/SL: Calculated from IB fill price, monitored via IB prices
- No auto-bracket orders (bot manages exits manually)
- SL bracket as safety net only

OPTIMAL STRATEGY:
- TP/SL: 1.0% / 0.5% (high frequency)
- Position sizing: 1x-5x based on confidence
- Thresholds: LONG=0.65, SHORT=0.25

Usage: python3 btc_ib_gateway_bot_hf_v2.py
"""

import time
from datetime import datetime
from binance.client import Client
import pandas as pd
import sys
import os
import logging
import json

BOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BOT_DIR)

from btc_model_package.predictor import BTCPredictor
from ib_insync import IB, Future, MarketOrder, Order
from market_hours import is_market_open, get_next_open_time

# Setup logging
LOG_FILE = os.path.join(BOT_DIR, 'logs', 'btc_ib_bot_hf_v2.log')
TRADE_LOG_FILE = os.path.join(BOT_DIR, 'logs', 'btc_trades_hf_v2.jsonl')
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
# TRADE LOGGING
# =============================================================================
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

# =============================================================================
# CONFIG
# =============================================================================
BINANCE_API_KEY = ""
BINANCE_API_SECRET = ""

# IB Gateway settings
IB_HOST = '127.0.0.1'
IB_PORT = 4002
IB_CLIENT_ID = 13  # NEW CLIENT ID for HF V2 bot

# Model settings
LONG_THRESHOLD = 0.65
SHORT_THRESHOLD = 0.25

# HIGH FREQUENCY TP/SL
TP_PCT = 1.0
SL_PCT = 0.5

# Position sizing
MIN_SIZE = 1
MAX_SIZE = 5
SCALING_FACTOR = 20

# Trading
CHECK_INTERVAL = 120  # 2 minutes
CONTRACT_SIZE = 0.1  # MBT = 0.1 BTC per contract

# =============================================================================
# POSITION SIZING
# =============================================================================
def calculate_position_size(confidence):
    """
    Calculate position size based on confidence level
    Formula: size = (confidence - 0.60) × 20
    Capped between MIN_SIZE (1) and MAX_SIZE (5)
    """
    size = (confidence - 0.60) * SCALING_FACTOR
    return max(MIN_SIZE, min(MAX_SIZE, size))

# =============================================================================
# IB CONTRACT SETUP
# =============================================================================
def create_btc_contract():
    """Create BTC futures contract (MBT - Micro Bitcoin) - front month"""
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
logger.info("BTC HIGH FREQUENCY BOT V2 - BOT-MANAGED TP/SL")
logger.info("=" * 80)
logger.info(f"TP/SL: {TP_PCT}% / {SL_PCT}% (High Frequency)")
logger.info(f"LONG Threshold:  {LONG_THRESHOLD}")
logger.info(f"SHORT Threshold: {SHORT_THRESHOLD}")
logger.info(f"Position Scaling: {MIN_SIZE}x - {MAX_SIZE}x")
logger.info(f"Client ID: {IB_CLIENT_ID}")
logger.info("")
logger.info("NEW: Bot-managed TP/SL (no bracket orders)")
logger.info("  - Signals from Binance")
logger.info("  - Execution & monitoring from IB")
logger.info("  - Single consistent price source")
logger.info("=" * 80)

# Load predictor
logger.info("")
logger.info("Loading V3 models...")
predictor = BTCPredictor()
predictor.LONG_THRESHOLD = LONG_THRESHOLD
predictor.SHORT_THRESHOLD = SHORT_THRESHOLD
logger.info(f"✓ Models loaded: {len(predictor.models)} horizons, {len(predictor.features)} features")

# Initialize Binance client (for data)
logger.info("")
logger.info("Connecting to Binance (signal generation)...")
binance = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
logger.info("✓ Binance connected")

# Initialize IB Gateway (for trading)
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
                return connect_ib()  # Retry indefinitely

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


def get_btc_data():
    """Fetch latest 250 bars of 5-min BTC data from Binance with retry logic"""
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
    """Get current IB price using historical bar (free, no subscription needed)"""
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
        ib.sleep(2)  # Wait for fill

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


def execute_entry(signal, binance_price, position_size, confidence, details):
    """Execute entry via IB Gateway"""
    quantity = int(position_size)
    total_btc = quantity * CONTRACT_SIZE

    logger.info(f"\n{'='*80}")
    logger.info(f"📈 ENTRY SIGNAL: {signal}")
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

    # Calculate TP/SL based on IB fill price
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

    return {
        'direction': signal,
        'entry_ib': fill_price,
        'tp_ib': tp,
        'sl_ib': sl,
        'size': quantity,
        'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'confidence': confidence,
        'sl_trade': sl_trade
    }


def execute_exit(position, exit_reason, exit_price):
    """Execute exit via IB Gateway"""
    entry_price = position['entry_ib']
    direction = position['direction']
    quantity = position['size']
    entry_time = position['time']

    if direction == 'LONG':
        pnl_pct = (exit_price / entry_price - 1) * 100
    else:
        pnl_pct = (entry_price / exit_price - 1) * 100

    pnl_dollar = pnl_pct / 100 * entry_price * CONTRACT_SIZE * quantity

    hold_duration = (datetime.now() - datetime.strptime(entry_time, "%Y-%m-%d %H:%M:%S")).total_seconds() / 60

    logger.info(f"\n{'='*80}")
    logger.info(f"🚪 EXIT: {exit_reason}")
    logger.info(f"Entry:  ${entry_price:,.2f} @ {entry_time}")
    logger.info(f"Exit:   ${exit_price:,.2f} (IB price)")
    logger.info(f"Size:   {quantity} contracts")
    logger.info(f"Hold:   {hold_duration:.0f} minutes")
    logger.info(f"P&L:    {pnl_pct:+.2f}% (${pnl_dollar:+,.2f})")
    logger.info(f"{'='*80}\n")

    # Log exit
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
        'pnl_dollar': pnl_dollar
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
logger.info("Starting main loop (Ctrl+C to stop)...")
logger.info(f"Checking every {CHECK_INTERVAL} seconds")
logger.info(f"Log file: {LOG_FILE}")
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

            # Get latest data from Binance for signal
            df = get_btc_data()
            current_price = df['close'].iloc[-1]
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Try to get IB price for monitoring (optional)
            ib_price = get_ib_current_price()
            if ib_price is None:
                ib_price = current_price  # Fallback to Binance price

            # SYNC WITH IB REALITY
            if current_position:
                ib_pos = get_ib_position()

                # Position closed by IB's SL bracket or external action
                if not ib_pos or ib_pos['quantity'] == 0:
                    logger.info("Position closed externally - syncing state")
                    # Log if we didn't catch the exit
                    if current_position:
                        execute_exit(current_position, 'EXTERNAL_CLOSE', ib_price)
                    current_position = None

                # Position size mismatch
                elif abs(ib_pos['quantity']) != current_position['size']:
                    logger.warning(f"Position size mismatch! Bot: {current_position['size']}, IB: {ib_pos['quantity']}")
                    current_position['size'] = abs(ib_pos['quantity'])

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
                       f"{confidence:>6.1%} {position_size:>5.1f}x {pos_display:<20}")

            # Enter new position if no current position and signal is strong
            if not current_position and signal != 'NEUTRAL':
                # Check if market is open before attempting to place orders
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
