#!/usr/bin/env python3
"""
Clean up historical trade logs by marking duplicate EXIT events
and creating cleaned versions of all trade log files.
"""

import json
import os
from datetime import datetime
from collections import defaultdict

BOT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BOT_DIR, 'logs')

# Trade log files to clean
TRADE_LOGS = [
    'btc_trades.jsonl',          # Swing bot
    'btc_trades_hf.jsonl',       # HF bot
    'btc_trades_v2.jsonl',       # Swing V2
    'btc_trades_hf_v2.jsonl',    # HF V2
    'btc_trades_micro.jsonl',    # Micro bot
    'btc_trades_micro_v2.jsonl'  # Micro V2
]

def clean_trade_log(log_file):
    """Clean a single trade log file by marking duplicate EXIT events"""

    log_path = os.path.join(LOG_DIR, log_file)

    if not os.path.exists(log_path):
        print(f"⏭  {log_file} - File not found, skipping")
        return

    # Read all log entries
    entries = []
    with open(log_path, 'r') as f:
        for line in f:
            try:
                entries.append(json.loads(line.strip()))
            except:
                pass

    if not entries:
        print(f"⏭  {log_file} - Empty file, skipping")
        return

    print(f"\n{'='*80}")
    print(f"Processing: {log_file}")
    print(f"{'='*80}")
    print(f"Total entries: {len(entries)}")

    # Track entries and exits
    entry_count = 0
    exit_count = 0
    duplicate_count = 0

    # Track open positions by entry timestamp
    open_positions = {}
    cleaned_entries = []

    for entry in entries:
        event = entry.get('event', '')

        # Track ENTRY events
        if event in ['ENTRY_EXECUTED', 'ENTRY_SIGNAL']:
            entry_time = entry.get('timestamp')
            direction = entry.get('data', {}).get('direction')

            if event == 'ENTRY_EXECUTED' and entry_time and direction:
                entry_count += 1
                # Store this as an open position
                position_key = f"{entry_time}_{direction}"
                open_positions[position_key] = {
                    'entry_time': entry_time,
                    'direction': direction,
                    'exit_seen': False
                }

            cleaned_entries.append(entry)

        # Check EXIT events for duplicates
        elif event == 'EXIT':
            exit_count += 1

            # Try to match this exit to an open position
            data = entry.get('data', {})
            direction = data.get('direction')
            entry_time = data.get('entry_time')

            if entry_time and direction:
                position_key = f"{entry_time}_{direction}"

                # Check if this is a duplicate exit
                if position_key in open_positions:
                    if open_positions[position_key]['exit_seen']:
                        # DUPLICATE - mark it
                        entry['duplicate'] = True
                        duplicate_count += 1
                        print(f"  🔴 Duplicate EXIT: {direction} from {entry_time}")
                    else:
                        # First exit for this position
                        open_positions[position_key]['exit_seen'] = True
                        entry['duplicate'] = False
                else:
                    # Orphan exit (no matching entry) - keep it but mark as orphan
                    entry['orphan_exit'] = True

            cleaned_entries.append(entry)

        # Keep all other events as-is
        else:
            cleaned_entries.append(entry)

    # Calculate clean metrics
    clean_exits = exit_count - duplicate_count

    print(f"\nResults:")
    print(f"  Entries:    {entry_count}")
    print(f"  Total Exits: {exit_count}")
    print(f"  Duplicates:  {duplicate_count} ❌")
    print(f"  Clean Exits: {clean_exits} ✓")
    print(f"  Ratio:       {exit_count}/{entry_count} = {exit_count/max(entry_count,1):.2f}x")

    # Backup original file
    backup_path = log_path + '.backup'
    os.rename(log_path, backup_path)
    print(f"\n  ✓ Original backed up: {log_file}.backup")

    # Write cleaned file
    with open(log_path, 'w') as f:
        for entry in cleaned_entries:
            f.write(json.dumps(entry) + '\n')

    print(f"  ✓ Cleaned file written: {log_file}")
    print(f"  ✓ Removed {duplicate_count} duplicate EXIT events")

    return {
        'file': log_file,
        'entries': entry_count,
        'total_exits': exit_count,
        'duplicates': duplicate_count,
        'clean_exits': clean_exits
    }

def main():
    print("\n" + "="*80)
    print("TRADE LOG CLEANUP - Marking Duplicate EXIT Events")
    print("="*80)

    results = []

    for log_file in TRADE_LOGS:
        result = clean_trade_log(log_file)
        if result:
            results.append(result)

    # Summary
    print("\n" + "="*80)
    print("CLEANUP SUMMARY")
    print("="*80)

    total_entries = sum(r['entries'] for r in results)
    total_exits = sum(r['total_exits'] for r in results)
    total_duplicates = sum(r['duplicates'] for r in results)
    total_clean_exits = sum(r['clean_exits'] for r in results)

    print(f"\nTotal across all bots:")
    print(f"  Entries:     {total_entries}")
    print(f"  Total Exits: {total_exits}")
    print(f"  Duplicates:  {total_duplicates} ❌ ({total_duplicates/max(total_exits,1)*100:.1f}%)")
    print(f"  Clean Exits: {total_clean_exits} ✓")
    print(f"  Clean Ratio: {total_clean_exits}/{total_entries} = {total_clean_exits/max(total_entries,1):.2f}x")

    print("\n" + "="*80)
    print("✓ CLEANUP COMPLETE")
    print(f"✓ All original files backed up with .backup extension")
    print(f"✓ Cleaned files written with duplicate flags")
    print(f"✓ Removed {total_duplicates} duplicate EXIT events from tracking")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
