"""
Validate Live Signals - Compare predictor signals vs actual bot trades

Fetches historical Binance data for the period bots have been trading,
runs the predictor on it, and compares signals with actual trades logged.

This validates:
1. Predictor is generating correct signals
2. Bots are executing on those signals properly
3. Signal quality matches expectations
"""

import pandas as pd
import json
from datetime import datetime, timedelta
from binance.client import Client
import sys
import os

BOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BOT_DIR)

from btc_model_package.predictor import BTCPredictor

# Binance credentials
BINANCE_API_KEY = ""
BINANCE_API_SECRET = ""

def get_binance_historical_data(start_date, end_date):
    """Fetch historical 5-min data from Binance using public API (no auth)"""
    import requests

    # Convert to timestamps
    start_ts = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)

    print(f"Fetching Binance data: {start_date} to {end_date}")

    all_klines = []
    current_ts = start_ts

    while current_ts < end_ts:
        # Use public API endpoint (no auth required)
        url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': 'BTCUSDT',
            'interval': '5m',
            'startTime': current_ts,
            'limit': 1000
        }

        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"API Error: {response.status_code} - {response.text}")
            break

        klines = response.json()
        if not klines:
            break

        all_klines.extend(klines)
        current_ts = klines[-1][0] + 1  # Next millisecond after last candle
        print(f"  Fetched {len(klines)} candles (total: {len(all_klines)})")

    # Convert to DataFrame
    df = pd.DataFrame(all_klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])

    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')

    return df


def load_actual_trades(log_file):
    """Load actual trades from bot log"""
    trades = []
    if not os.path.exists(log_file):
        return trades

    with open(log_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if entry['event'] in ['ENTRY_SIGNAL', 'ENTRY_EXECUTED']:
                    trades.append({
                        'time': entry['timestamp'],
                        'event': entry['event'],
                        'direction': entry['data']['direction'],
                        'confidence': entry['data'].get('confidence', 0),
                        'entry_price': entry['data'].get('ib_fill_price') or entry['data'].get('binance_price', 0)
                    })
            except:
                continue

    return trades


def simulate_signals(df, predictor, check_interval_minutes=2):
    """Simulate signal generation on historical data"""
    signals = []

    # Need at least 250 bars for feature calculation
    start_idx = 250

    # Simulate checking every N minutes
    step = check_interval_minutes  # 5-min bars, check every 2 min = skip 0 bars (check every bar for precision)

    print(f"\nSimulating signals from {df.index[start_idx]} to {df.index[-1]}")
    print(f"Checking every {check_interval_minutes} minutes (every bar)")

    for i in range(start_idx, len(df)):
        # Get window of data up to this point
        window = df.iloc[:i+1]

        # Get prediction
        try:
            signal, confidence, details = predictor.predict(window)

            if signal != 'NEUTRAL':
                signals.append({
                    'time': window.index[-1],
                    'signal': signal,
                    'confidence': confidence,
                    'price': window['close'].iloc[-1],
                    'details': details
                })
                print(f"{window.index[-1]}: {signal} @ ${window['close'].iloc[-1]:,.2f} (conf: {confidence:.1%})")
        except Exception as e:
            print(f"Error at {window.index[-1]}: {e}")
            continue

    return signals


def compare_signals_to_trades(simulated_signals, actual_trades):
    """Compare simulated signals with actual trades"""
    print("\n" + "="*80)
    print("SIGNAL COMPARISON")
    print("="*80)

    print(f"\nSimulated Signals: {len(simulated_signals)}")
    print(f"Actual Trades: {len(actual_trades)}")

    # Convert times for matching
    actual_times = [datetime.fromisoformat(t['time']) for t in actual_trades]

    matches = 0
    mismatches = 0

    print("\n--- Actual Trades vs Simulated Signals ---")
    for trade in actual_trades:
        trade_time = datetime.fromisoformat(trade['time'])
        trade_dir = trade['direction']
        trade_conf = trade['confidence']

        # Find closest simulated signal (within 10 minutes)
        closest_signal = None
        min_diff = timedelta(minutes=10)

        for sig in simulated_signals:
            diff = abs(sig['time'] - trade_time)
            if diff < min_diff and sig['signal'] == trade_dir:
                min_diff = diff
                closest_signal = sig

        if closest_signal:
            matches += 1
            print(f"✓ MATCH   {trade_time} - {trade_dir} (conf: {trade_conf:.1%})")
            print(f"          Simulated: {closest_signal['time']} (diff: {min_diff.total_seconds()/60:.1f}min)")
        else:
            mismatches += 1
            print(f"✗ MISSING {trade_time} - {trade_dir} (conf: {trade_conf:.1%})")
            print(f"          No matching signal found in simulation")

    print(f"\n--- Simulated Signals Not in Actual Trades ---")
    for sig in simulated_signals:
        sig_time = sig['time']
        sig_dir = sig['signal']

        # Check if this signal matches any actual trade
        found = False
        for trade in actual_trades:
            trade_time = datetime.fromisoformat(trade['time'])
            if abs(sig_time - trade_time) < timedelta(minutes=10) and trade['direction'] == sig_dir:
                found = True
                break

        if not found:
            print(f"• {sig_time} - {sig_dir} @ ${sig['price']:,.2f} (conf: {sig['confidence']:.1%})")

    print("\n" + "="*80)
    print(f"Match Rate: {matches}/{len(actual_trades)} ({matches/len(actual_trades)*100:.1f}%)")
    print("="*80)


def main():
    # Initialize predictor
    print("Loading predictor models...")
    predictor = BTCPredictor()
    predictor.LONG_THRESHOLD = 0.65
    predictor.SHORT_THRESHOLD = 0.25
    print(f"✓ Models loaded")

    # Determine date range from actual trades
    print("\nAnalyzing actual trade logs...")

    # Check V1 Swing bot (has most trades)
    trade_log = os.path.join(BOT_DIR, 'logs', 'btc_trades.jsonl')
    actual_trades = load_actual_trades(trade_log)

    if not actual_trades:
        print("No trades found in log!")
        return

    # Get date range
    trade_times = [datetime.fromisoformat(t['time']) for t in actual_trades]
    start_date = min(trade_times) - timedelta(days=1)  # Buffer for feature calculation
    end_date = max(trade_times) + timedelta(hours=1)

    print(f"Found {len(actual_trades)} trades from {min(trade_times)} to {max(trade_times)}")

    # Fetch historical data
    df = get_binance_historical_data(start_date, end_date)
    print(f"✓ Fetched {len(df)} bars of historical data")

    # Simulate signals
    simulated_signals = simulate_signals(df, predictor, check_interval_minutes=2)

    # Compare
    compare_signals_to_trades(simulated_signals, actual_trades)

    # Signal quality analysis
    print("\n" + "="*80)
    print("SIGNAL QUALITY ANALYSIS")
    print("="*80)

    long_signals = [s for s in simulated_signals if s['signal'] == 'LONG']
    short_signals = [s for s in simulated_signals if s['signal'] == 'SHORT']

    print(f"\nTotal Signals: {len(simulated_signals)}")
    print(f"  LONG:  {len(long_signals)} ({len(long_signals)/len(simulated_signals)*100:.1f}%)")
    print(f"  SHORT: {len(short_signals)} ({len(short_signals)/len(simulated_signals)*100:.1f}%)")

    if long_signals:
        avg_long_conf = sum(s['confidence'] for s in long_signals) / len(long_signals)
        print(f"\nAverage LONG confidence: {avg_long_conf:.1%}")

    if short_signals:
        avg_short_conf = sum(s['confidence'] for s in short_signals) / len(short_signals)
        print(f"Average SHORT confidence: {avg_short_conf:.1%}")


if __name__ == '__main__':
    main()
