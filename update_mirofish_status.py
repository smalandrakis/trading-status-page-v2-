#!/usr/bin/env python3
"""
Updates bot_status.json with MiroFish prediction data
Run this periodically to keep the dashboard updated with latest MiroFish signals
"""

import json
import os
from datetime import datetime
from pathlib import Path

# Paths
BOT_STATUS_FILE = '/Users/smalandrakis/trading-status-page-v2/bot_status.json'
MIROFISH_SIGNAL_DIR = os.path.expanduser('~/CascadeProjects/mirofish-signal/output')

def load_mirofish_signals():
    """Load all available MiroFish signals"""
    signals = {}

    signal_files = {
        'cerebras': 'latest_signal_cerebras.json',
        'groq': 'latest_signal_groq.json',
        'mirofish_full': 'latest_signal.json'
    }

    for source, filename in signal_files.items():
        filepath = os.path.join(MIROFISH_SIGNAL_DIR, filename)
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    signal_data = json.load(f)

                    # Calculate age in minutes
                    signal_time = datetime.fromisoformat(signal_data['timestamp'].replace('Z', '+00:00'))
                    age_minutes = (datetime.now(signal_time.tzinfo) - signal_time).total_seconds() / 60

                    signals[source] = {
                        **signal_data,
                        'age_minutes': round(age_minutes, 1),
                        'is_fresh': age_minutes < 1440  # Less than 24 hours
                    }
        except Exception as e:
            print(f"Warning: Could not load {source} signal: {e}")

    return signals

def update_bot_status():
    """Add MiroFish section to bot_status.json"""

    # Load existing bot status
    try:
        with open(BOT_STATUS_FILE, 'r') as f:
            bot_status = json.load(f)
    except FileNotFoundError:
        print(f"Error: {BOT_STATUS_FILE} not found")
        return False

    # Load MiroFish signals
    mirofish_signals = load_mirofish_signals()

    # Add MiroFish section
    bot_status['mirofish_predictions'] = {
        'updated_at': datetime.now().isoformat(),
        'signals': mirofish_signals,
        'status': 'active' if mirofish_signals else 'no_signals'
    }

    # Write updated status
    with open(BOT_STATUS_FILE, 'w') as f:
        json.dump(bot_status, f, indent=2)

    print(f"✓ Updated bot_status.json with {len(mirofish_signals)} MiroFish signals")
    for source, signal in mirofish_signals.items():
        fresh = "✓ FRESH" if signal['is_fresh'] else "⚠ STALE"
        print(f"  {source}: {signal['direction']} ({signal['confidence']:.1%}) - {signal['age_minutes']:.0f}min old {fresh}")

    return True

if __name__ == '__main__':
    import sys

    # Check if signal directory exists
    if not os.path.exists(MIROFISH_SIGNAL_DIR):
        print(f"Warning: Signal directory not found: {MIROFISH_SIGNAL_DIR}")
        print("Creating directory...")
        os.makedirs(MIROFISH_SIGNAL_DIR, exist_ok=True)

    success = update_bot_status()
    sys.exit(0 if success else 1)
