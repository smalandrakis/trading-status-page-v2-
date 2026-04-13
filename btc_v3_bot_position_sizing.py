"""
BTC V3 Trading Bot with Confidence-Based Position Sizing

OPTIMAL STRATEGY (from backtest):
- Base thresholds: LONG=0.65, SHORT=0.25
- Position scaling: size = (confidence - 0.60) × 20 (capped at 1-5x)
- Expected: 732 trades/2yr, 45.5% WR, +$7,047 ($9.63/trade)
- Improvement: +59% vs fixed sizing

Usage: python3 btc_v3_bot_position_sizing.py
"""

import time
from datetime import datetime
from binance.client import Client
import pandas as pd
from btc_model_package.predictor import BTCPredictor

# =============================================================================
# CONFIG - OPTIMAL POSITION SIZING
# =============================================================================
BINANCE_API_KEY = ""  # Empty for public data
BINANCE_API_SECRET = ""

# Model settings - OPTIMIZED THRESHOLDS
LONG_THRESHOLD = 0.65   # Higher base (filters weak signals)
SHORT_THRESHOLD = 0.25  # More selective

# Position sizing - LINEAR SCALING
MIN_SIZE = 1.0  # Minimum contracts
MAX_SIZE = 5.0  # Maximum contracts
SCALING_FACTOR = 20  # size = (confidence - 0.60) × 20

# Trading settings
CHECK_INTERVAL = 120  # Check every 2 minutes
BASE_POSITION_SIZE = 0.1  # 0.1 BTC per contract (MBT)

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
# INITIALIZE
# =============================================================================
print("=" * 70)
print("BTC V3 BOT - CONFIDENCE-BASED POSITION SIZING")
print("=" * 70)
print(f"LONG Threshold:  {LONG_THRESHOLD}")
print(f"SHORT Threshold: {SHORT_THRESHOLD}")
print(f"Position Scaling: {MIN_SIZE}x - {MAX_SIZE}x")
print(f"Formula: size = (confidence - 0.60) × {SCALING_FACTOR}")
print("=" * 70)

# Load predictor
print("\nLoading 22-feature V3 models...")
predictor = BTCPredictor()
predictor.LONG_THRESHOLD = LONG_THRESHOLD
predictor.SHORT_THRESHOLD = SHORT_THRESHOLD
print(f"✓ Models loaded: {len(predictor.models)} horizons, {len(predictor.features)} features")

# Initialize Binance client
print("\nConnecting to Binance...")
binance = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
print("✓ Binance connected")

# Track current position
current_position = None


def get_btc_data():
    """Fetch latest 250 bars of 5-min BTC data from Binance"""
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


def check_exit(position, current_price):
    """Check if TP or SL hit"""
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


def execute_entry(signal, entry_price, tp, sl, position_size, confidence):
    """Execute entry via IB Gateway (implement IB connection)"""
    total_btc = position_size * BASE_POSITION_SIZE

    print(f"\n{'='*70}")
    print(f"📈 ENTRY SIGNAL: {signal}")
    print(f"Confidence: {confidence:.1%}")
    print(f"Position Size: {position_size:.2f}x contracts ({total_btc:.2f} BTC)")
    print(f"Entry: ${entry_price:,.2f}")
    print(f"TP:    ${tp:,.2f} ({'+'if signal=='LONG' else '-'}1.0%)")
    print(f"SL:    ${sl:,.2f} ({'-'if signal=='LONG' else '+'}0.5%)")
    print(f"{'='*70}\n")

    # TODO: Place bracket order via IB Gateway
    # For each contract (loop position_size times):
    #   contract = Contract()
    #   contract.symbol = "BTC"
    #   contract.secType = "FUT"
    #   contract.exchange = "CME"
    #   ... submit bracket order

    return True


def execute_exit(position, exit_reason, exit_price):
    """Execute exit via IB Gateway"""
    entry_price = position['entry']
    direction = position['direction']
    position_size = position['size']

    if direction == 'LONG':
        pnl_pct = (exit_price / entry_price - 1) * 100
    else:
        pnl_pct = (entry_price / exit_price - 1) * 100

    pnl_dollar = pnl_pct / 100 * entry_price * BASE_POSITION_SIZE * position_size

    print(f"\n{'='*70}")
    print(f"🚪 EXIT: {exit_reason}")
    print(f"Entry:  ${entry_price:,.2f}")
    print(f"Exit:   ${exit_price:,.2f}")
    print(f"Size:   {position_size:.2f}x contracts")
    print(f"P&L:    {pnl_pct:+.2f}% (${pnl_dollar:+,.2f})")
    print(f"{'='*70}\n")

    # TODO: Close position via IB Gateway

    return True


# =============================================================================
# MAIN LOOP
# =============================================================================
print("\nStarting main loop (Ctrl+C to stop)...")
print(f"Checking every {CHECK_INTERVAL} seconds\n")

try:
    while True:
        try:
            # Get latest data
            df = get_btc_data()
            current_price = df['close'].iloc[-1]
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Check if we need to exit current position
            if current_position:
                should_exit, reason = check_exit(current_position, current_price)
                if should_exit:
                    execute_exit(current_position, reason, current_price)
                    current_position = None

            # Get prediction
            signal, confidence, details = predictor.predict(df)

            # Calculate position size based on confidence
            position_size = calculate_position_size(confidence) if signal != 'NEUTRAL' else 0

            print(f"[{current_time}] BTC: ${current_price:,.2f} | "
                  f"Signal: {signal:7s} | Conf: {confidence:.1%} | "
                  f"Size: {position_size:.2f}x | "
                  f"Pos: {current_position['direction'] + ' ' + str(current_position['size']) + 'x' if current_position else 'NONE'}")

            # Enter new position if no current position and signal is strong
            if not current_position and signal != 'NEUTRAL':
                tp, sl = predictor.get_tp_sl_prices(current_price, signal)

                if execute_entry(signal, current_price, tp, sl, position_size, confidence):
                    current_position = {
                        'direction': signal,
                        'entry': current_price,
                        'tp': tp,
                        'sl': sl,
                        'size': position_size,
                        'time': current_time,
                        'confidence': confidence
                    }

            # Wait
            time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"ERROR: {e}")
            time.sleep(CHECK_INTERVAL)

except KeyboardInterrupt:
    print("\n\n" + "="*70)
    print("Bot stopped by user")
    if current_position:
        print(f"WARNING: Position still open: {current_position['direction']} "
              f"{current_position['size']:.2f}x from ${current_position['entry']:,.2f}")
    print("="*70)
