"""
Replay Validation - Exact 1:1 Signal Matching

Downloads last 2 days of historical data and replays it through the predictor
to validate that live bot signals match exactly with what the model produces.

This tests:
1. Predictor determinism (same data → same signal)
2. Feature calculation consistency
3. No timing or data issues
4. Perfect reproducibility
"""

import pandas as pd
import json
from datetime import datetime, timedelta
import sys
import os
import requests

BOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BOT_DIR)

from btc_model_package.predictor import BTCPredictor

def download_historical_data(days=2):
    """Download last N days of 5-min BTCUSDT data from Binance"""
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)

    start_ts = int(start_time.timestamp() * 1000)
    end_ts = int(end_time.timestamp() * 1000)

    print(f"Downloading data: {start_time} to {end_time}")

    all_klines = []
    current_ts = start_ts

    while current_ts < end_ts:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': 'BTCUSDT',
            'interval': '5m',
            'startTime': current_ts,
            'limit': 1000
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                print(f"API Error: {response.status_code}")
                break

            klines = response.json()
            if not klines:
                break

            all_klines.extend(klines)
            current_ts = klines[-1][0] + 1
            print(f"  Downloaded {len(klines)} candles (total: {len(all_klines)})")

        except Exception as e:
            print(f"Error: {e}")
            break

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

    print(f"✓ Downloaded {len(df)} bars")
    return df


def load_actual_signals(log_file):
    """Load ENTRY_SIGNAL events from bot log"""
    signals = []

    if not os.path.exists(log_file):
        return signals

    with open(log_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if entry['event'] == 'ENTRY_SIGNAL':
                    signals.append({
                        'timestamp': datetime.fromisoformat(entry['timestamp']),
                        'direction': entry['data']['direction'],
                        'confidence': entry['data']['confidence'],
                        'price': entry['data']['entry_price'],
                        'details': entry['data'].get('predictor_details', {})
                    })
            except Exception as e:
                continue

    return signals


def replay_signals(df, predictor, check_interval_minutes=2):
    """Replay predictor on historical data, checking every N minutes"""
    signals = []

    # Need 250+ bars for features
    start_idx = 250

    # Check every N minutes (N * bars_per_minute)
    # 5-min bars, check every 2 min = every 0.4 bars ≈ every bar for safety

    print(f"\nReplaying predictor from {df.index[start_idx]} to {df.index[-1]}")
    print(f"Checking every {check_interval_minutes} minutes")

    for i in range(start_idx, len(df)):
        window = df.iloc[:i+1]

        try:
            signal, confidence, details = predictor.predict(window)

            if signal != 'NEUTRAL':
                signals.append({
                    'timestamp': window.index[-1],
                    'direction': signal,
                    'confidence': confidence,
                    'price': window['close'].iloc[-1],
                    'details': details
                })
                print(f"  {window.index[-1]}: {signal} @ ${window['close'].iloc[-1]:,.2f} (conf: {confidence:.1%})")

        except Exception as e:
            print(f"Error at {window.index[-1]}: {e}")
            continue

    return signals


def compare_signals(replayed, actual, bot_name):
    """Compare replayed signals with actual bot signals"""
    print(f"\n{'='*80}")
    print(f"{bot_name} - SIGNAL VALIDATION")
    print(f"{'='*80}")

    print(f"\nReplayed Signals: {len(replayed)}")
    print(f"Actual Signals:   {len(actual)}")

    if not actual:
        print("No actual signals to compare")
        return

    # Match signals (within 10 minutes)
    matches = []
    mismatches = []
    extra_replayed = []

    print(f"\n--- Matching Actual vs Replayed ---")

    for actual_sig in actual:
        # Find matching replayed signal
        found = None
        min_diff = timedelta(minutes=10)

        for replay_sig in replayed:
            time_diff = abs(replay_sig['timestamp'] - actual_sig['timestamp'])
            if (time_diff < min_diff and
                replay_sig['direction'] == actual_sig['direction']):
                min_diff = time_diff
                found = replay_sig

        if found:
            # Check if it's an exact match
            time_match = min_diff.total_seconds() < 60  # Within 1 minute
            conf_match = abs(found['confidence'] - actual_sig['confidence']) < 0.001

            # Get probability details for comparison
            actual_probs = actual_sig['details'].get('probabilities', {})
            replay_probs = found['details'].get('probabilities', {})

            probs_match = True
            if actual_probs and replay_probs:
                for horizon in ['2h', '4h', '6h']:
                    if abs(actual_probs.get(horizon, 0) - replay_probs.get(horizon, 0)) > 0.001:
                        probs_match = False
                        break

            exact_match = time_match and conf_match and probs_match

            if exact_match:
                print(f"✓ EXACT MATCH")
            else:
                print(f"≈ CLOSE MATCH")

            print(f"  Actual:   {actual_sig['timestamp']} | {actual_sig['direction']} | Conf: {actual_sig['confidence']:.4f}")
            print(f"  Replayed: {found['timestamp']} | {found['direction']} | Conf: {found['confidence']:.4f}")

            if not time_match:
                print(f"  Time diff: {min_diff.total_seconds():.1f}s")

            if not conf_match:
                print(f"  Confidence diff: {abs(found['confidence'] - actual_sig['confidence']):.4f}")

            if actual_probs and replay_probs:
                print(f"  Actual probs:   2h={actual_probs.get('2h', 0):.4f} 4h={actual_probs.get('4h', 0):.4f} 6h={actual_probs.get('6h', 0):.4f}")
                print(f"  Replayed probs: 2h={replay_probs.get('2h', 0):.4f} 4h={replay_probs.get('4h', 0):.4f} 6h={replay_probs.get('6h', 0):.4f}")

            if exact_match:
                matches.append((actual_sig, found))
            else:
                mismatches.append((actual_sig, found))
        else:
            print(f"✗ MISSING IN REPLAY")
            print(f"  Actual: {actual_sig['timestamp']} | {actual_sig['direction']} | Conf: {actual_sig['confidence']:.4f}")
            mismatches.append((actual_sig, None))

    # Check for extra signals in replay
    print(f"\n--- Extra Signals in Replay (not in actual) ---")
    for replay_sig in replayed:
        found = False
        for actual_sig in actual:
            time_diff = abs(replay_sig['timestamp'] - actual_sig['timestamp'])
            if (time_diff < timedelta(minutes=10) and
                replay_sig['direction'] == actual_sig['direction']):
                found = True
                break

        if not found:
            print(f"• {replay_sig['timestamp']} | {replay_sig['direction']} @ ${replay_sig['price']:,.2f} (conf: {replay_sig['confidence']:.1%})")
            extra_replayed.append(replay_sig)

    # Summary
    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print(f"Exact Matches:     {len(matches)}/{len(actual)} ({len(matches)/len(actual)*100:.1f}%)")
    print(f"Close Matches:     {len(mismatches)}")
    print(f"Missing in Replay: {len([m for m in mismatches if m[1] is None])}")
    print(f"Extra in Replay:   {len(extra_replayed)}")

    if len(matches) == len(actual) and len(extra_replayed) == 0:
        print(f"\n✅ PERFECT MATCH - Bot is executing predictor exactly as expected!")
    elif len(matches) > len(actual) * 0.8:
        print(f"\n⚠️  GOOD MATCH - Minor timing or data differences")
    else:
        print(f"\n❌ MISMATCH - Significant differences detected")

    return {
        'exact_matches': len(matches),
        'close_matches': len(mismatches),
        'extra': len(extra_replayed),
        'total_actual': len(actual)
    }


def main():
    print("="*80)
    print("REPLAY VALIDATION - 1:1 Signal Matching")
    print("="*80)

    # Download historical data
    print("\nStep 1: Downloading historical data...")
    df = download_historical_data(days=2)

    if len(df) < 250:
        print("Not enough data downloaded")
        return

    # Initialize predictor
    print("\nStep 2: Loading predictor models...")
    predictor = BTCPredictor()
    predictor.LONG_THRESHOLD = 0.65
    predictor.SHORT_THRESHOLD = 0.25
    print("✓ Models loaded")

    # Replay signals
    print("\nStep 3: Replaying predictor on historical data...")
    replayed_signals = replay_signals(df, predictor, check_interval_minutes=2)

    # Compare with actual bot logs
    print("\nStep 4: Comparing with actual bot signals...")

    bots = [
        ('V1 Swing Bot (2.5/1.0)', os.path.join(BOT_DIR, 'logs', 'btc_trades.jsonl')),
        ('V1 High Frequency (1.0/0.5)', os.path.join(BOT_DIR, 'logs', 'btc_trades_hf.jsonl')),
    ]

    results = {}
    for bot_name, log_file in bots:
        actual_signals = load_actual_signals(log_file)

        if actual_signals:
            # Filter replayed signals to only those in the actual signal timeframe
            first_actual = min(s['timestamp'] for s in actual_signals)
            last_actual = max(s['timestamp'] for s in actual_signals)

            filtered_replayed = [
                s for s in replayed_signals
                if first_actual - timedelta(hours=1) <= s['timestamp'] <= last_actual + timedelta(hours=1)
            ]

            result = compare_signals(filtered_replayed, actual_signals, bot_name)
            results[bot_name] = result

    # Final summary
    print(f"\n\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")

    for bot_name, result in results.items():
        match_rate = result['exact_matches'] / result['total_actual'] * 100 if result['total_actual'] > 0 else 0
        print(f"\n{bot_name}")
        print(f"  Match Rate: {result['exact_matches']}/{result['total_actual']} ({match_rate:.1f}%)")
        print(f"  Status: {'✅ PASS' if match_rate >= 90 else '⚠️  PARTIAL' if match_rate >= 70 else '❌ FAIL'}")


if __name__ == '__main__':
    main()
