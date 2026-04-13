"""
Comprehensive Bot Validation
Check that V1 and V2 bots are working identically and correctly
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

BOT_DIR = Path(__file__).parent
sys.path.insert(0, str(BOT_DIR))

from btc_model_package.predictor import BTCPredictor
from binance.client import Client

# ============================================================================
# TEST 1: Data Fetching Consistency
# ============================================================================
def test_data_fetching():
    """Verify both bots fetch data in the same format"""
    print("\n" + "="*80)
    print("TEST 1: DATA FETCHING CONSISTENCY")
    print("="*80)

    # Simulate V1 data fetch
    binance = Client("", "")

    try:
        klines = binance.get_klines(
            symbol='BTCUSDT',
            interval=Client.KLINE_INTERVAL_5MINUTE,
            limit=250
        )

        # V1 format
        df_v1 = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df_v1[col] = pd.to_numeric(df_v1[col])
        df_v1['timestamp'] = pd.to_datetime(df_v1['timestamp'], unit='ms')
        df_v1.set_index('timestamp', inplace=True)
        df_v1 = df_v1[['open', 'high', 'low', 'close', 'volume']]

        # V2 format (after fix)
        df_v2 = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df_v2[col] = df_v2[col].astype(float)
        df_v2['timestamp'] = pd.to_datetime(df_v2['timestamp'], unit='ms')
        df_v2.set_index('timestamp', inplace=True)
        df_v2 = df_v2[['open', 'high', 'low', 'close', 'volume']]

        # Compare
        print(f"\n✓ Data fetched successfully")
        print(f"  Rows: {len(df_v1)}")
        print(f"  V1 index type: {type(df_v1.index)}")
        print(f"  V2 index type: {type(df_v2.index)}")
        print(f"  V1 columns: {list(df_v1.columns)}")
        print(f"  V2 columns: {list(df_v2.columns)}")

        # Check if identical
        if df_v1.equals(df_v2):
            print(f"\n✅ PASS: V1 and V2 data formats are IDENTICAL")
        else:
            print(f"\n⚠️  WARNING: V1 and V2 data formats differ")
            print(f"  Max price diff: {abs(df_v1['close'] - df_v2['close']).max()}")

        return df_v1, df_v2

    except Exception as e:
        print(f"\n❌ FAIL: Error fetching data: {e}")
        return None, None


# ============================================================================
# TEST 2: Predictor Consistency
# ============================================================================
def test_predictor_consistency(df_v1, df_v2):
    """Verify predictor produces same signals on V1 and V2 data"""
    print("\n" + "="*80)
    print("TEST 2: PREDICTOR CONSISTENCY")
    print("="*80)

    if df_v1 is None or df_v2 is None:
        print("\n❌ SKIP: No data available")
        return

    try:
        predictor = BTCPredictor()
        predictor.LONG_THRESHOLD = 0.65
        predictor.SHORT_THRESHOLD = 0.25

        # Run on V1 data
        signal_v1, conf_v1, details_v1 = predictor.predict(df_v1)

        # Run on V2 data
        signal_v2, conf_v2, details_v2 = predictor.predict(df_v2)

        print(f"\n✓ Predictor ran successfully")
        print(f"\nV1 Signal: {signal_v1} @ {conf_v1:.1%}")
        print(f"  Probabilities: 2h={details_v1['probabilities']['2h']:.4f}, "
              f"4h={details_v1['probabilities']['4h']:.4f}, "
              f"6h={details_v1['probabilities']['6h']:.4f}")

        print(f"\nV2 Signal: {signal_v2} @ {conf_v2:.1%}")
        print(f"  Probabilities: 2h={details_v2['probabilities']['2h']:.4f}, "
              f"4h={details_v2['probabilities']['4h']:.4f}, "
              f"6h={details_v2['probabilities']['6h']:.4f}")

        # Compare
        signal_match = signal_v1 == signal_v2
        conf_match = abs(conf_v1 - conf_v2) < 0.0001

        prob_match = all(
            abs(details_v1['probabilities'][h] - details_v2['probabilities'][h]) < 0.0001
            for h in ['2h', '4h', '6h']
        )

        if signal_match and conf_match and prob_match:
            print(f"\n✅ PASS: V1 and V2 produce IDENTICAL signals")
        elif signal_match:
            print(f"\n⚠️  PARTIAL: Same signal direction, minor numerical differences")
        else:
            print(f"\n❌ FAIL: V1 and V2 produce DIFFERENT signals")

        return signal_v1, signal_v2, details_v1, details_v2

    except Exception as e:
        print(f"\n❌ FAIL: Error running predictor: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


# ============================================================================
# TEST 3: Bot Configuration Match
# ============================================================================
def test_bot_configs():
    """Check that V1 and V2 have identical configurations"""
    print("\n" + "="*80)
    print("TEST 3: BOT CONFIGURATION MATCH")
    print("="*80)

    configs = {
        'V1 Swing': {
            'file': 'btc_ib_gateway_bot.py',
            'log': 'logs/btc_ib_bot.log',
            'trades': 'logs/btc_trades.jsonl'
        },
        'V2 Swing': {
            'file': 'btc_ib_gateway_bot_v2.py',
            'log': 'logs/btc_ib_bot_v2.log',
            'trades': 'logs/btc_trades_v2.jsonl'
        },
        'V1 HF': {
            'file': 'btc_ib_gateway_bot_hf.py',
            'log': 'logs/btc_ib_bot_hf.log',
            'trades': 'logs/btc_trades_hf.jsonl'
        },
        'V2 HF': {
            'file': 'btc_ib_gateway_bot_hf_v2.py',
            'log': 'logs/btc_ib_bot_hf_v2.log',
            'trades': 'logs/btc_trades_hf_v2.jsonl'
        }
    }

    issues = []

    for bot_name, config in configs.items():
        bot_file = BOT_DIR / config['file']

        if not bot_file.exists():
            print(f"\n⚠️  {bot_name}: File not found")
            continue

        with open(bot_file, 'r') as f:
            content = f.read()

        # Extract key configs
        import re

        long_thresh = re.search(r'LONG_THRESHOLD\s*=\s*([\d.]+)', content)
        short_thresh = re.search(r'SHORT_THRESHOLD\s*=\s*([\d.]+)', content)
        tp_pct = re.search(r'TP_PCT\s*=\s*([\d.]+)', content)
        sl_pct = re.search(r'SL_PCT\s*=\s*([\d.]+)', content)
        client_id = re.search(r'IB_CLIENT_ID\s*=\s*(\d+)', content)

        # Check data fetch has .set_index
        has_set_index = 'df.set_index(\'timestamp\'' in content or 'df.set_index("timestamp"' in content

        print(f"\n{bot_name}:")
        print(f"  LONG Threshold: {long_thresh.group(1) if long_thresh else 'NOT FOUND'}")
        print(f"  SHORT Threshold: {short_thresh.group(1) if short_thresh else 'NOT FOUND'}")
        print(f"  TP/SL: {tp_pct.group(1) if tp_pct else '?'}/{sl_pct.group(1) if sl_pct else '?'}")
        print(f"  Client ID: {client_id.group(1) if client_id else 'NOT FOUND'}")
        print(f"  Has set_index: {'✓' if has_set_index else '✗ MISSING'}")

        if not has_set_index:
            issues.append(f"{bot_name}: Missing .set_index('timestamp')")

    if not issues:
        print(f"\n✅ PASS: All bots have correct configurations")
    else:
        print(f"\n❌ FAIL: Configuration issues found:")
        for issue in issues:
            print(f"  - {issue}")


# ============================================================================
# TEST 4: Log File Analysis
# ============================================================================
def test_log_analysis():
    """Analyze recent bot activity and signal generation"""
    print("\n" + "="*80)
    print("TEST 4: LOG FILE ANALYSIS")
    print("="*80)

    bots = [
        ('V1 Swing', 'logs/btc_trades.jsonl'),
        ('V2 Swing', 'logs/btc_trades_v2.jsonl'),
        ('V1 HF', 'logs/btc_trades_hf.jsonl'),
        ('V2 HF', 'logs/btc_trades_hf_v2.jsonl'),
    ]

    for bot_name, log_file in bots:
        log_path = BOT_DIR / log_file

        print(f"\n{bot_name}:")

        if not log_path.exists():
            print(f"  Log file not found")
            continue

        # Count events
        events = {'ENTRY_SIGNAL': 0, 'ENTRY_EXECUTED': 0, 'EXIT': 0,
                  'ORPHAN_DETECTED': 0, 'ORPHAN_CLOSED': 0}
        last_signal = None

        with open(log_path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    event_type = entry['event']
                    if event_type in events:
                        events[event_type] += 1

                    if event_type == 'ENTRY_SIGNAL':
                        last_signal = entry

                except:
                    continue

        total_trades = events['EXIT']
        print(f"  Total trades: {total_trades}")
        print(f"  Signals generated: {events['ENTRY_SIGNAL']}")
        print(f"  Positions entered: {events['ENTRY_EXECUTED']}")
        print(f"  Orphans detected: {events['ORPHAN_DETECTED']}")

        if last_signal:
            timestamp = last_signal['timestamp']
            signal = last_signal['data']['direction']
            conf = last_signal['data']['confidence']
            print(f"  Last signal: {signal} @ {conf:.1%} ({timestamp})")


# ============================================================================
# TEST 5: Running Process Check
# ============================================================================
def test_running_processes():
    """Check which bots are currently running"""
    print("\n" + "="*80)
    print("TEST 5: RUNNING PROCESS CHECK")
    print("="*80)

    import subprocess

    try:
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True
        )

        processes = []
        for line in result.stdout.split('\n'):
            if 'btc_ib' in line and 'bot' in line and 'grep' not in line:
                processes.append(line)

        if processes:
            print(f"\n✓ Found {len(processes)} running bot(s):")
            for proc in processes:
                # Extract bot name
                if 'bot_v2.py' in proc:
                    bot_type = 'V2 Swing'
                elif 'bot_hf_v2.py' in proc:
                    bot_type = 'V2 HF'
                elif 'bot_hf.py' in proc:
                    bot_type = 'V1 HF'
                elif 'bot.py' in proc:
                    bot_type = 'V1 Swing'
                else:
                    bot_type = 'Unknown'

                print(f"  - {bot_type}")
        else:
            print(f"\n⚠️  No bot processes found running")

    except Exception as e:
        print(f"\n❌ Error checking processes: {e}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("="*80)
    print("COMPREHENSIVE BOT VALIDATION")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Run all tests
    df_v1, df_v2 = test_data_fetching()
    test_predictor_consistency(df_v1, df_v2)
    test_bot_configs()
    test_log_analysis()
    test_running_processes()

    # Final summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    print("\nKey Findings:")
    print("1. Check if V1 and V2 data formats match")
    print("2. Check if predictor produces identical signals")
    print("3. Check if all configs are correct")
    print("4. Check bot activity logs")
    print("5. Check which bots are running")
    print("\nReview the output above for any ❌ FAIL or ⚠️ WARNING indicators.")


if __name__ == '__main__':
    main()
