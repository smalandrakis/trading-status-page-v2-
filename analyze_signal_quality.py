"""
Analyze Signal Quality from Actual Trade Logs

Reads ENTRY_SIGNAL events from trade logs and analyzes:
1. Signal distribution (LONG vs SHORT)
2. Confidence levels
3. Actual outcomes (TP vs SL)
4. Win rates by signal type
5. Model feature analysis
"""

import json
import os
from datetime import datetime
from collections import defaultdict

BOT_DIR = os.path.dirname(os.path.abspath(__file__))

def analyze_trade_log(log_file, bot_name):
    """Analyze signals and outcomes from trade log"""
    if not os.path.exists(log_file):
        print(f"✗ {bot_name}: Log file not found")
        return None

    signals = []
    exits = []

    with open(log_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if entry['event'] == 'ENTRY_SIGNAL':
                    signals.append(entry)
                elif entry['event'] == 'EXIT':
                    exits.append(entry)
            except:
                continue

    if not signals:
        print(f"✗ {bot_name}: No signals found")
        return None

    print(f"\n{'='*80}")
    print(f"{bot_name}")
    print(f"{'='*80}")

    # Signal breakdown
    long_signals = [s for s in signals if s['data']['direction'] == 'LONG']
    short_signals = [s for s in signals if s['data']['direction'] == 'SHORT']

    print(f"\n📊 SIGNAL DISTRIBUTION")
    print(f"Total Signals: {len(signals)}")
    print(f"  LONG:  {len(long_signals)} ({len(long_signals)/len(signals)*100:.1f}%)")
    print(f"  SHORT: {len(short_signals)} ({len(short_signals)/len(signals)*100:.1f}%)")

    # Confidence analysis
    if signals[0]['data'].get('confidence'):
        long_conf = [s['data']['confidence'] for s in long_signals]
        short_conf = [s['data']['confidence'] for s in short_signals]

        print(f"\n📈 CONFIDENCE LEVELS")
        if long_conf:
            print(f"  LONG:  Avg {sum(long_conf)/len(long_conf):.1%} | Range {min(long_conf):.1%}-{max(long_conf):.1%}")
        if short_conf:
            print(f"  SHORT: Avg {sum(short_conf)/len(short_conf):.1%} | Range {min(short_conf)/len(short_conf):.1%}-{max(short_conf):.1%}")

    # Match signals with outcomes
    print(f"\n🎯 OUTCOMES")
    print(f"Total Exits: {len(exits)}")

    long_exits = [e for e in exits if e['data']['direction'] == 'LONG']
    short_exits = [e for e in exits if e['data']['direction'] == 'SHORT']

    # LONG outcomes
    if long_exits:
        long_wins = [e for e in long_exits if e['data']['pnl_dollar'] > 0]
        long_losses = [e for e in long_exits if e['data']['pnl_dollar'] <= 0]
        long_wr = len(long_wins) / len(long_exits) * 100

        long_tp = [e for e in long_exits if 'TAKE_PROFIT' in e['data']['exit_reason']]
        long_sl = [e for e in long_exits if 'STOP_LOSS' in e['data']['exit_reason']]

        print(f"  LONG:  {len(long_exits)} trades | {len(long_wins)}W / {len(long_losses)}L ({long_wr:.1f}% WR)")
        print(f"         TP: {len(long_tp)} | SL: {len(long_sl)}")

        if long_wins:
            avg_win = sum(e['data']['pnl_dollar'] for e in long_wins) / len(long_wins)
            print(f"         Avg Win: ${avg_win:.2f}")
        if long_losses:
            avg_loss = sum(e['data']['pnl_dollar'] for e in long_losses) / len(long_losses)
            print(f"         Avg Loss: ${avg_loss:.2f}")

    # SHORT outcomes
    if short_exits:
        short_wins = [e for e in short_exits if e['data']['pnl_dollar'] > 0]
        short_losses = [e for e in short_exits if e['data']['pnl_dollar'] <= 0]
        short_wr = len(short_wins) / len(short_exits) * 100

        short_tp = [e for e in short_exits if 'TAKE_PROFIT' in e['data']['exit_reason']]
        short_sl = [e for e in short_exits if 'STOP_LOSS' in e['data']['exit_reason']]

        print(f"  SHORT: {len(short_exits)} trades | {len(short_wins)}W / {len(short_losses)}L ({short_wr:.1f}% WR)")
        print(f"         TP: {len(short_tp)} | SL: {len(short_sl)}")

        if short_wins:
            avg_win = sum(e['data']['pnl_dollar'] for e in short_wins) / len(short_wins)
            print(f"         Avg Win: ${avg_win:.2f}")
        if short_losses:
            avg_loss = sum(e['data']['pnl_dollar'] for e in short_losses) / len(short_losses)
            print(f"         Avg Loss: ${avg_loss:.2f}")

    # Overall
    if exits:
        total_pnl = sum(e['data']['pnl_dollar'] for e in exits)
        avg_pnl = total_pnl / len(exits)
        wins = [e for e in exits if e['data']['pnl_dollar'] > 0]
        wr = len(wins) / len(exits) * 100

        print(f"\n  OVERALL: {len(exits)} trades | {wr:.1f}% WR")
        print(f"           Total P&L: ${total_pnl:.2f}")
        print(f"           Avg P&L: ${avg_pnl:.2f}")

    # Time analysis
    print(f"\n🕒 TIMING")
    if signals:
        first_signal = datetime.fromisoformat(signals[0]['timestamp'])
        last_signal = datetime.fromisoformat(signals[-1]['timestamp'])
        duration = (last_signal - first_signal).total_seconds() / 3600  # hours

        print(f"  First Signal: {first_signal}")
        print(f"  Last Signal:  {last_signal}")
        print(f"  Duration: {duration:.1f} hours ({duration/24:.1f} days)")
        print(f"  Signal Frequency: {len(signals)/duration:.2f} signals/hour ({len(signals)/(duration/24):.1f} signals/day)")

    # Model probability analysis (if available)
    signals_with_details = [s for s in signals if 'predictor_details' in s['data']]
    if signals_with_details:
        print(f"\n🤖 MODEL PROBABILITIES")
        print(f"  (From {len(signals_with_details)} signals with details)")

        # Extract probabilities per horizon
        probs_2h = []
        probs_4h = []
        probs_6h = []

        for s in signals_with_details:
            probs = s['data']['predictor_details']['probabilities']
            probs_2h.append(probs['2h'])
            probs_4h.append(probs['4h'])
            probs_6h.append(probs['6h'])

        print(f"  2h model: Avg {sum(probs_2h)/len(probs_2h):.1%} | Range {min(probs_2h):.1%}-{max(probs_2h):.1%}")
        print(f"  4h model: Avg {sum(probs_4h)/len(probs_4h):.1%} | Range {min(probs_4h):.1%}-{max(probs_4h):.1%}")
        print(f"  6h model: Avg {sum(probs_6h)/len(probs_6h):.1%} | Range {min(probs_6h):.1%}-{max(probs_6h):.1%}")

    return {
        'signals': len(signals),
        'long_signals': len(long_signals),
        'short_signals': len(short_signals),
        'exits': len(exits),
        'win_rate': wr if exits else 0,
        'total_pnl': total_pnl if exits else 0
    }


def main():
    print("="*80)
    print("TRADING SIGNAL QUALITY ANALYSIS")
    print("="*80)

    bots = [
        ('V1 Swing Bot (2.5/1.0)', os.path.join(BOT_DIR, 'logs', 'btc_trades.jsonl')),
        ('V1 High Frequency (1.0/0.5)', os.path.join(BOT_DIR, 'logs', 'btc_trades_hf.jsonl')),
        ('V2 Swing Bot (2.5/1.0)', os.path.join(BOT_DIR, 'logs', 'btc_trades_v2.jsonl')),
        ('V2 High Frequency (1.0/0.5)', os.path.join(BOT_DIR, 'logs', 'btc_trades_hf_v2.jsonl')),
    ]

    results = []
    for bot_name, log_file in bots:
        result = analyze_trade_log(log_file, bot_name)
        if result:
            results.append((bot_name, result))

    # Summary comparison
    if results:
        print(f"\n\n{'='*80}")
        print("SUMMARY COMPARISON")
        print(f"{'='*80}")
        print(f"{'Bot':<40} {'Signals':<10} {'Trades':<10} {'WR':<10} {'P&L':<15}")
        print("-"*80)

        for bot_name, result in results:
            print(f"{bot_name:<40} {result['signals']:<10} {result['exits']:<10} "
                  f"{result['win_rate']:<9.1f}% ${result['total_pnl']:<14.2f}")


if __name__ == '__main__':
    main()
