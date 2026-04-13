"""
BTC Trading Bot - HIGH FREQUENCY - IB Gateway Integration

HIGH FREQUENCY CONFIGURATION (verified 2024-2026 backtest):
- TP/SL: 1.0% / 0.5% (baseline configuration)
- Position sizing: 1x-5x based on confidence
- Thresholds: LONG=0.65, SHORT=0.25
- Expected: 734 trades/2yr, 45.9% WR, +$10,120 ($13.79/trade)
- Frequency: ~7.1 trades/week, ~$97/week expected P&L

Data: Binance (real-time, no delay)
Execution: IB Gateway port 4002 (paper trading)

Usage: python3 btc_ib_gateway_bot_hf.py
"""

import time
import asyncio
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
from ib_insync import IB, Future, MarketOrder, Order, LimitOrder
from market_hours import is_market_open, get_next_open_time
from trading_filters import TradingFilters

# Setup logging
LOG_FILE = os.path.join(BOT_DIR, 'logs', 'btc_ib_bot_hf.log')
TRADE_LOG_FILE = os.path.join(BOT_DIR, 'logs', 'btc_trades_hf.jsonl')
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
# CONFIG - HIGH FREQUENCY CONFIGURATION
# =============================================================================
BINANCE_API_KEY = ""  # Empty for public data
BINANCE_API_SECRET = ""

# IB Gateway settings
IB_HOST = '127.0.0.1'
IB_PORT = 4002  # Paper trading
IB_CLIENT_ID = 11  # Different from swing bot (10)

# Model settings - OPTIMIZED THRESHOLDS
LONG_THRESHOLD = 0.65
SHORT_THRESHOLD = 0.25

# HIGH FREQUENCY TP/SL (verified optimal baseline)
TP_PCT = 1.0  # 1.0% - Baseline configuration
SL_PCT = 0.5  # 0.5% - Verified optimal

# Position sizing - LINEAR SCALING
MIN_SIZE = 1.0  # Minimum contracts
MAX_SIZE = 5.0  # Maximum contracts
SCALING_FACTOR = 20  # size = (confidence - 0.60) × 20

# Trading settings
CHECK_INTERVAL = 120  # Check every 2 minutes
CONTRACT_SIZE = 0.1  # BTC per contract (MBT = 0.1 BTC)

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
# POSITION SIZE CALCULATOR
# =============================================================================
def calculate_position_size(confidence):
    """
    Calculate position size based on confidence level

    Formula: size = (confidence - 0.60) × 20
    Capped between MIN_SIZE (1) and MAX_SIZE (5)

    Examples:
    - confidence 0.65 → 1.0x
    - confidence 0.70 → 2.0x
    - confidence 0.75 → 3.0x
    - confidence 0.80 → 4.0x
    - confidence 0.85+ → 5.0x (max)
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
        lastTradeDateOrContractMonth='20260424',  # Front month (April 2026)
        exchange='CME',
        currency='USD',
        multiplier='0.1'
    )
    return contract


# =============================================================================
# INITIALIZE
# =============================================================================
logger.info("=" * 80)
logger.info("BTC IB GATEWAY BOT - HIGH FREQUENCY CONFIGURATION")
logger.info("=" * 80)
logger.info(f"TP/SL: {TP_PCT}% / {SL_PCT}% (HIGH FREQUENCY - tighter stops)")
logger.info(f"LONG Threshold:  {LONG_THRESHOLD}")
logger.info(f"SHORT Threshold: {SHORT_THRESHOLD}")
logger.info(f"Position Scaling: {MIN_SIZE}x - {MAX_SIZE}x")
logger.info(f"Formula: size = (confidence - 0.60) × {SCALING_FACTOR}")
logger.info(f"")
logger.info(f"Expected Performance (backtest):")
logger.info(f"  790 trades/2yr, 33.7% WR, +$11,689 (+$14.80/trade)")
logger.info(f"  Frequency: ~1.08 trades/day (7-8 per week)")
logger.info(f"  +28% more trades vs swing bot for better validation")
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
logger.info("Connecting to Binance (data source)...")
binance = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
logger.info("✓ Binance connected")

# Initialize IB Gateway (for trading)
logger.info(f"")
logger.info(f"Connecting to IB Gateway ({IB_HOST}:{IB_PORT})...")
ib = IB()
try:
    ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID)
    logger.info("✓ IB Gateway connected")
except Exception as e:
    logger.error(f"✗ Failed to connect to IB Gateway: {e}")
    logger.error(f"  Make sure IB Gateway is running on port {IB_PORT}")
    sys.exit(1)

# Get BTC contract
btc_contract = create_btc_contract()
ib.qualifyContracts(btc_contract)
logger.info(f"✓ BTC contract qualified: {btc_contract}")

# Initialize trading filters
logger.info("")
logger.info("Initializing trading filters...")
filters = TradingFilters(TP_PCT, SL_PCT)
logger.info("✓ All filters active")

# Track current position
current_position = None


def get_btc_data():
    """Fetch latest 250 bars of 5-min BTC data from Binance with retry logic"""
    max_retries = 5
    retry_delays = [2, 5, 10, 20, 30]  # Exponential backoff

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
                df[col] = pd.to_numeric(df[col])

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df[['open', 'high', 'low', 'close', 'volume']]

            return df

        except Exception as e:
            wait_time = retry_delays[attempt]
            logger.warning(f"Binance API error (attempt {attempt + 1}/{max_retries}): {e}")

            if attempt < max_retries - 1:
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to fetch data after {max_retries} attempts")
                raise


def get_current_ib_position():
    """Get current BTC position from IB"""
    positions = ib.positions()
    for pos in positions:
        if pos.contract.symbol == 'MBT':
            return {
                'size': pos.position,
                'avg_price': pos.avgCost,
                'contract': pos.contract
            }
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


def check_exit(position, current_price):
    """Check if TP or SL hit for tracked position"""
    if not position:
        return False, None

    direction = position['direction']
    tp = position['tp']
    sl = position['sl']

    if direction == 'LONG':
        if current_price >= tp:
            return True, 'TAKE_PROFIT'
        if current_price <= sl:
            return True, 'STOP_LOSS'

    elif direction == 'SHORT':
        if current_price <= tp:
            return True, 'TAKE_PROFIT'
        if current_price >= sl:
            return True, 'STOP_LOSS'

    return False, None


def place_bracket_order(direction, entry_price, tp_price, sl_price, quantity):
    """
    Place bracket order: entry + TP + SL

    For paper trading, we'll use market order for entry and limit orders for TP/SL
    """
    try:
        # Determine action
        action = 'BUY' if direction == 'LONG' else 'SELL'

        # Entry order (market)
        entry_order = MarketOrder(action, quantity)

        # Place entry order
        trade = ib.placeOrder(btc_contract, entry_order)
        ib.sleep(2)  # Wait for fill

        # Check if filled
        if trade.orderStatus.status not in ['Filled', 'PreSubmitted', 'Submitted']:
            logger.warning(f"  ✗ Entry order failed: {trade.orderStatus.status}")
            return None

        logger.info(f"  ✓ Entry order placed: {action} {quantity} @ market")

        # Now place TP and SL orders
        # TP: opposite action, limit order at TP price
        tp_action = 'SELL' if direction == 'LONG' else 'BUY'
        tp_order = LimitOrder(tp_action, quantity, tp_price)
        tp_trade = ib.placeOrder(btc_contract, tp_order)
        logger.info(f"  ✓ TP order placed: {tp_action} {quantity} @ ${tp_price:,.2f}")

        # SL: opposite action, stop order at SL price
        sl_order = Order()
        sl_order.action = tp_action
        sl_order.orderType = 'STP'
        sl_order.auxPrice = sl_price
        sl_order.totalQuantity = quantity
        sl_trade = ib.placeOrder(btc_contract, sl_order)
        logger.info(f"  ✓ SL order placed: {tp_action} {quantity} @ ${sl_price:,.2f}")

        return {
            'entry_trade': trade,
            'tp_trade': tp_trade,
            'sl_trade': sl_trade
        }

    except Exception as e:
        logger.error(f"  ✗ Error placing bracket order: {e}")
        return None


def close_position(direction, quantity):
    """Close position with market order"""
    try:
        action = 'SELL' if direction == 'LONG' else 'BUY'
        order = MarketOrder(action, quantity)
        trade = ib.placeOrder(btc_contract, order)
        ib.sleep(1)
        logger.info(f"  ✓ Position closed: {action} {quantity} @ market")
        return True
    except Exception as e:
        logger.error(f"  ✗ Error closing position: {e}")
        return False


def execute_entry(signal, entry_price, tp, sl, position_size, confidence, details):
    """Execute entry via IB Gateway"""
    quantity = int(position_size)  # Round to integer contracts
    total_btc = quantity * CONTRACT_SIZE

    logger.info(f"\n{'='*80}")
    logger.info(f"📈 ENTRY SIGNAL: {signal}")
    logger.info(f"Confidence: {confidence:.1%}")
    logger.info(f"Position Size: {quantity} contracts ({total_btc:.1f} BTC)")
    logger.info(f"Entry: ${entry_price:,.2f}")
    logger.info(f"TP:    ${tp:,.2f} ({'+'if signal=='LONG' else '-'}{TP_PCT}%)")
    logger.info(f"SL:    ${sl:,.2f} ({'-'if signal=='LONG' else '+'}{SL_PCT}%)")
    logger.info(f"Details: {details}")
    logger.info(f"{'='*80}\n")

    # Log entry signal
    log_trade('ENTRY_SIGNAL', {
        'direction': signal,
        'confidence': confidence,
        'entry_price': entry_price,
        'tp_price': tp,
        'sl_price': sl,
        'position_size': quantity,
        'tp_pct': TP_PCT,
        'sl_pct': SL_PCT,
        'predictor_details': details
    })

    # Place bracket order
    trades = place_bracket_order(signal, entry_price, tp, sl, quantity)

    if trades:
        log_trade('ENTRY_EXECUTED', {
            'direction': signal,
            'entry_price': entry_price,
            'tp_price': tp,
            'sl_price': sl,
            'position_size': quantity
        })

        return {
            'direction': signal,
            'entry': entry_price,
            'tp': tp,
            'sl': sl,
            'size': quantity,
            'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'confidence': confidence,
            'trades': trades,
            'details': details
        }

    return None


def execute_exit(position, exit_reason, exit_price):
    """Execute exit via IB Gateway"""
    # Prevent duplicate exit logging (IB partial fills create multiple exit events)
    if position.get('exit_logged', False):
        logger.debug(f"Exit already logged for this position, skipping duplicate")
        return True

    entry_price = position['entry']
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
    logger.info(f"Exit:   ${exit_price:,.2f}")
    logger.info(f"Size:   {quantity} contracts")
    logger.info(f"Hold:   {hold_duration:.0f} minutes")
    logger.info(f"P&L:    {pnl_pct:+.2f}% (${pnl_dollar:+,.2f})")
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
        'pnl_dollar': pnl_dollar
    })

    # Mark exit as logged to prevent duplicates
    position['exit_logged'] = True

    # Update filter state with trade result
    filters.record_trade_result(pnl_dollar)

    # Cancel pending orders and close position
    if 'trades' in position:
        try:
            # Cancel TP/SL orders
            if 'tp_trade' in position['trades']:
                ib.cancelOrder(position['trades']['tp_trade'].order)
            if 'sl_trade' in position['trades']:
                ib.cancelOrder(position['trades']['sl_trade'].order)
        except:
            pass

    # Close position
    return close_position(direction, quantity)


# =============================================================================
# MAIN LOOP
# =============================================================================
logger.info("")
logger.info("Starting main loop (Ctrl+C to stop)...")
logger.info(f"Checking every {CHECK_INTERVAL} seconds")
logger.info(f"Main log: {LOG_FILE}")
logger.info(f"Trade log: {TRADE_LOG_FILE}")
logger.info("")
logger.info(f"{'Time':<20} {'BTC Price':<12} {'Signal':<8} {'Conf':<7} {'Size':<6} {'Position':<20}")
logger.info("-" * 80)

try:
    while True:
        try:
            # Get latest data from Binance
            df = get_btc_data()
            current_price = df['close'].iloc[-1]
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # SYNC WITH IB REALITY - Check if position still exists
            if current_position:
                ib_pos = get_ib_position()

                # Position closed by bracket or externally
                if not ib_pos or ib_pos['quantity'] == 0:
                    logger.info("Position closed by bracket/external - clearing state")
                    current_position = None

                # Position size mismatch
                elif abs(ib_pos['quantity']) != current_position['size']:
                    logger.warning(f"Size mismatch! Bot: {current_position['size']}, IB: {ib_pos['quantity']}")
                    current_position['size'] = abs(ib_pos['quantity'])

            # Check if we need to exit current position
            if current_position:
                should_exit, reason = check_exit(current_position, current_price)
                if should_exit:
                    if execute_exit(current_position, reason, current_price):
                        current_position = None

            # Get prediction
            signal, confidence, details = predictor.predict(df)

            # Calculate position size based on confidence
            position_size = calculate_position_size(confidence) if signal != 'NEUTRAL' else 0

            # Format position display
            if current_position:
                pos_display = f"{current_position['direction']} {current_position['size']}x"
            else:
                pos_display = "NONE"

            logger.info(f"{current_time:<20} ${current_price:>10,.2f} {signal:<8} "
                       f"{confidence:>6.1%} {position_size:>5.1f}x {pos_display:<20}")

            # Enter new position if no current position and signal is strong
            if not current_position and signal != 'NEUTRAL':
                # Check if market is open before attempting to place orders
                if not is_market_open():
                    logger.warning(f"⏸  Signal: {signal} @ {confidence:.1%} but market is CLOSED")
                    logger.warning(f"   Market opens: {get_next_open_time()}")
                    time.sleep(CHECK_INTERVAL)
                    continue

                # Run all filters before entry
                filter_pass, filter_reason = filters.check_all_filters(
                    signal, confidence, current_price, df, position_size
                )

                if not filter_pass:
                    logger.info(f"⏸  Signal filtered: {signal} @ {confidence:.1%} - {filter_reason}")
                    time.sleep(CHECK_INTERVAL)
                    continue

                # Calculate TP/SL with HIGH FREQUENCY 1.5%/0.3% ratios
                if signal == 'LONG':
                    tp = current_price * (1 + TP_PCT / 100)
                    sl = current_price * (1 - SL_PCT / 100)
                else:  # SHORT
                    tp = current_price * (1 - TP_PCT / 100)
                    sl = current_price * (1 + SL_PCT / 100)

                # Execute entry
                position = execute_entry(signal, current_price, tp, sl, position_size, confidence, details)
                if position:
                    current_position = position
                    filters.record_trade_attempt()  # Record trade for daily limit
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
              f"{current_position['size']} contracts from ${current_position['entry']:,.2f}")

        # Ask if user wants to close position
        response = input("Close position? (y/n): ")
        if response.lower() == 'y':
            current_price = get_btc_data()['close'].iloc[-1]
            execute_exit(current_position, 'MANUAL_CLOSE', current_price)

    logger.info("Disconnecting from IB Gateway...")
    ib.disconnect()
    logger.info("="*80)
