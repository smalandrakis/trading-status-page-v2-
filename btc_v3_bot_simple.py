"""
Simple BTC Trading Bot - V3 22-Feature Model
Optimized Thresholds: LONG=0.60, SHORT=0.30

Data: Binance (free, real-time)
Execution: IB Gateway Port 4002 (paper trading)

Usage: python3 btc_v3_bot_simple.py
"""

import time
from datetime import datetime
from binance.client import Client
import pandas as pd
from btc_model_package.predictor import BTCPredictor

# =============================================================================
# CONFIG
# =============================================================================
BINANCE_API_KEY = ""  # Empty for public data
BINANCE_API_SECRET = ""

# Model settings - OPTIMIZED THRESHOLDS
LONG_THRESHOLD = 0.60   # Higher threshold for better WR
SHORT_THRESHOLD = 0.30  # More selective

# Trading settings
CHECK_INTERVAL = 120  # Check every 2 minutes
POSITION_SIZE = 0.1   # 0.1 BTC (1 MBT contract)

# =============================================================================
# INITIALIZE
# =============================================================================
print("=" * 70)
print("BTC V3 TRADING BOT - 22 FEATURES, OPTIMIZED THRESHOLDS")
print("=" * 70)
print(f"LONG Threshold:  {LONG_THRESHOLD}")
print(f"SHORT Threshold: {SHORT_THRESHOLD}")
print(f"Position Size:   {POSITION_SIZE} BTC")
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
current_position = None  # {'direction': 'LONG', 'entry': 67000, 'tp': 67670, 'sl': 66665, 'time': ...}


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


def execute_entry(signal, entry_price, tp, sl):
    """Execute entry via IB Gateway (implement IB connection)"""
    # TODO: Implement IB Gateway order submission
    # For now, just print
    print(f"\n{'='*70}")
    print(f"📈 ENTRY SIGNAL: {signal}")
    print(f"Entry: ${entry_price:,.2f}")
    print(f"TP:    ${tp:,.2f} ({'+'if signal=='LONG' else '-'}1.0%)")
    print(f"SL:    ${sl:,.2f} ({'-'if signal=='LONG' else '+'}0.5%)")
    print(f"Size:  {POSITION_SIZE} BTC")
    print(f"{'='*70}\n")

    # TODO: Place bracket order via IB Gateway
    # contract = Contract()
    # contract.symbol = "BTC"
    # contract.secType = "FUT"
    # contract.exchange = "CME"
    # ...

    return True


def execute_exit(position, exit_reason, exit_price):
    """Execute exit via IB Gateway"""
    entry_price = position['entry']
    direction = position['direction']

    if direction == 'LONG':
        pnl_pct = (exit_price / entry_price - 1) * 100
    else:
        pnl_pct = (entry_price / exit_price - 1) * 100

    pnl_dollar = pnl_pct / 100 * entry_price * POSITION_SIZE

    print(f"\n{'='*70}")
    print(f"🚪 EXIT: {exit_reason}")
    print(f"Entry:  ${entry_price:,.2f}")
    print(f"Exit:   ${exit_price:,.2f}")
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

            print(f"[{current_time}] BTC: ${current_price:,.2f} | "
                  f"Signal: {signal:7s} | Conf: {confidence:.1%} | "
                  f"Pos: {current_position['direction'] if current_position else 'NONE'}")

            # Enter new position if no current position and signal is strong
            if not current_position and signal != 'NEUTRAL':
                tp, sl = predictor.get_tp_sl_prices(current_price, signal)

                if execute_entry(signal, current_price, tp, sl):
                    current_position = {
                        'direction': signal,
                        'entry': current_price,
                        'tp': tp,
                        'sl': sl,
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
        print(f"WARNING: Position still open: {current_position['direction']} from ${current_position['entry']:,.2f}")
    print("="*70)
