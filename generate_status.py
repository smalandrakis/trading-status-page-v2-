"""
Status Page Generator for Trading Bots

Generates bot_status.json for GitHub Pages integration
Updates every cycle with current bot state, performance, and recent trades

Output: bot_status.json (for https://smalandrakis.github.io/trading-status-page/)
"""

import json
import os
from datetime import datetime
from pathlib import Path

BOT_DIR = Path(__file__).parent

def parse_trade_log(log_file):
    """Parse JSONL trade log and extract statistics"""
    if not os.path.exists(log_file):
        return {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'avg_pnl': 0,
            'recent_trades': []
        }

    trades = []
    with open(log_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if entry['event'] == 'EXIT':
                    trades.append(entry['data'])
            except:
                continue

    if not trades:
        return {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'avg_pnl': 0,
            'recent_trades': []
        }

    wins = sum(1 for t in trades if t['pnl_dollar'] > 0)
    losses = len(trades) - wins
    total_pnl = sum(t['pnl_dollar'] for t in trades)
    avg_pnl = total_pnl / len(trades) if trades else 0
    win_rate = wins / len(trades) * 100 if trades else 0

    # Recent trades (last 10)
    recent = trades[-10:]

    return {
        'total_trades': len(trades),
        'wins': wins,
        'losses': losses,
        'win_rate': round(win_rate, 1),
        'total_pnl': round(total_pnl, 2),
        'avg_pnl': round(avg_pnl, 2),
        'recent_trades': [
            {
                'time': t['exit_time'],
                'direction': t['direction'],
                'pnl': round(t['pnl_dollar'], 2),
                'pnl_pct': round(t['pnl_pct'], 2),
                'hold_minutes': round(t['hold_minutes'], 0)
            }
            for t in recent
        ]
    }


def get_current_signal(log_file):
    """Extract latest signal and confidence from main log"""
    if not os.path.exists(log_file):
        return None, None, None

    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            # Look for last line with signal format
            for line in reversed(lines[-50:]):  # Check last 50 lines
                if 'LONG' in line or 'SHORT' in line or 'NEUTRAL' in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part in ['LONG', 'SHORT', 'NEUTRAL']:
                            # Next part should be confidence like "65.2%"
                            if i + 1 < len(parts):
                                conf_str = parts[i + 1].replace('%', '')
                                try:
                                    confidence = float(conf_str)
                                    signal = part
                                    return signal, confidence, line.split('|')[0].strip() if '|' in line else None
                                except:
                                    pass
    except:
        pass
    return None, None, None


def get_open_position(log_file, trade_log_file):
    """Get current open position from logs"""
    if not os.path.exists(trade_log_file):
        return None

    try:
        # Read trade log to find last entry/exit
        entries = []
        exits = []
        orphan_adopted = None

        with open(trade_log_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if entry['event'] == 'ENTRY_EXECUTED':
                        entries.append(entry)
                    elif entry['event'] == 'EXIT':
                        exits.append(entry)
                    elif entry['event'] == 'ORPHAN_ADOPTED':
                        orphan_adopted = entry
                except:
                    continue

        # Check if there's an open position (entry without matching exit)
        if entries:
            last_entry = entries[-1]
            last_exit = exits[-1] if exits else None

            # Position is open if last entry is after last exit
            entry_time = datetime.fromisoformat(last_entry['timestamp'])
            exit_time = datetime.fromisoformat(last_exit['timestamp']) if last_exit else None

            if not exit_time or entry_time > exit_time:
                data = last_entry['data']
                return {
                    'direction': data['direction'],
                    'size': data['position_size'],
                    'entry_price': data.get('entry_price', 0),
                    'tp_price': data.get('tp_price', 0),
                    'sl_price': data.get('sl_price', 0),
                    'entry_time': data.get('entry_time', 'unknown'),
                    'adopted': False
                }

        # Check if orphan was adopted after last exit
        if orphan_adopted:
            orphan_time = datetime.fromisoformat(orphan_adopted['timestamp'])
            last_exit_time = datetime.fromisoformat(exits[-1]['timestamp']) if exits else None

            if not last_exit_time or orphan_time > last_exit_time:
                data = orphan_adopted['data']
                return {
                    'direction': data['direction'],
                    'size': int(data['position_size']),
                    'entry_price': data.get('entry_price', 0),
                    'tp_price': data.get('tp_price', 0),
                    'sl_price': data.get('sl_price', 0),
                    'entry_time': 'adopted',
                    'adopted': True
                }
    except:
        pass

    return None


def get_bot_status(bot_name, log_file, trade_log_file, config):
    """Get current status for a bot"""

    # Parse trade log
    stats = parse_trade_log(trade_log_file)

    # Check if bot is running
    running = os.path.exists(log_file)

    # Get last update from main log
    last_update = None
    if running and os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1]
                    # Extract timestamp from log line
                    if '|' in last_line:
                        last_update = last_line.split('|')[0].strip()
        except:
            pass

    # Get current signal and confidence
    signal, confidence, signal_time = get_current_signal(log_file)

    # Get open position
    open_position = get_open_position(log_file, trade_log_file)

    return {
        'name': bot_name,
        'status': 'running' if running else 'stopped',
        'last_update': last_update or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'config': config,
        'current_signal': {
            'signal': signal or 'UNKNOWN',
            'confidence': round(confidence, 1) if confidence else 0,
            'time': signal_time
        },
        'open_position': open_position,
        'performance': {
            'total_trades': stats['total_trades'],
            'wins': stats['wins'],
            'losses': stats['losses'],
            'win_rate': stats['win_rate'],
            'total_pnl': stats['total_pnl'],
            'avg_pnl': stats['avg_pnl']
        },
        'recent_trades': stats['recent_trades']
    }


def check_binance_connection():
    """Check if Binance API is accessible"""
    try:
        import requests
        response = requests.get('https://api.binance.com/api/v3/ping', timeout=5)
        if response.status_code == 200:
            return {'status': 'connected', 'message': 'Binance API accessible'}
        else:
            return {'status': 'error', 'message': f'HTTP {response.status_code}'}
    except Exception as e:
        return {'status': 'disconnected', 'message': str(e)}


def sanitize_for_json(obj):
    """Replace NaN and Infinity with None for valid JSON"""
    import math

    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0  # Replace NaN/Infinity with 0
        return obj
    return obj


def generate_status_json():
    """Generate bot_status.json for all bots"""

    # Check Binance connection
    binance_status = check_binance_connection()

    bots = [
        {
            'name': 'BTC Swing Bot (2.5/1.0)',
            'log_file': BOT_DIR / 'logs' / 'btc_ib_bot.log',
            'trade_log_file': BOT_DIR / 'logs' / 'btc_trades.jsonl',
            'config': {
                'tp_pct': 2.5,
                'sl_pct': 1.0,
                'long_threshold': 0.65,
                'short_threshold': 0.25,
                'expected_trades_per_week': '5-6',
                'expected_win_rate': 52.6,
                'expected_pnl_2yr': 16577
            }
        },
        {
            'name': 'BTC High Frequency (1.0/0.5)',
            'log_file': BOT_DIR / 'logs' / 'btc_ib_bot_hf.log',
            'trade_log_file': BOT_DIR / 'logs' / 'btc_trades_hf.jsonl',
            'config': {
                'tp_pct': 1.0,
                'sl_pct': 0.5,
                'long_threshold': 0.65,
                'short_threshold': 0.25,
                'expected_trades_per_week': '7-8',
                'expected_win_rate': 33.7,
                'expected_pnl_2yr': 11689
            }
        },
        {
            'name': 'BTC Swing Bot V2 (2.5/1.0) [Bot-Managed]',
            'log_file': BOT_DIR / 'logs' / 'btc_ib_bot_v2.log',
            'trade_log_file': BOT_DIR / 'logs' / 'btc_trades_v2.jsonl',
            'config': {
                'tp_pct': 2.5,
                'sl_pct': 1.0,
                'long_threshold': 0.65,
                'short_threshold': 0.25,
                'expected_trades_per_week': '5-6',
                'expected_win_rate': 52.6,
                'expected_pnl_2yr': 16577
            }
        },
        {
            'name': 'BTC High Frequency V2 (1.0/0.5) [Bot-Managed]',
            'log_file': BOT_DIR / 'logs' / 'btc_ib_bot_hf_v2.log',
            'trade_log_file': BOT_DIR / 'logs' / 'btc_trades_hf_v2.jsonl',
            'config': {
                'tp_pct': 1.0,
                'sl_pct': 0.5,
                'long_threshold': 0.65,
                'short_threshold': 0.25,
                'expected_trades_per_week': '7-8',
                'expected_win_rate': 33.7,
                'expected_pnl_2yr': 11689
            }
        },
        {
            'name': 'BTC Micro-Movement (0.3/0.1) [Adaptive Learning]',
            'log_file': BOT_DIR / 'logs' / 'btc_micro_bot.log',
            'trade_log_file': BOT_DIR / 'logs' / 'btc_trades_micro.jsonl',
            'config': {
                'tp_pct': 0.3,
                'sl_pct': 0.1,
                'long_threshold': 0.50,
                'short_threshold': 0.30,
                'expected_trades_per_week': '10-15',
                'expected_win_rate': 'TBD (backtest running)',
                'expected_pnl_2yr': 'TBD'
            }
        },
        {
            'name': 'BTC Micro-Movement V2 (0.5/0.15) [LONG/SHORT Models]',
            'log_file': BOT_DIR / 'logs' / 'btc_micro_movement.log',
            'trade_log_file': BOT_DIR / 'logs' / 'btc_micro_movement_trades.jsonl',
            'config': {
                'tp_pct': 0.5,
                'sl_pct': 0.15,
                'long_threshold': 0.50,
                'short_threshold': 0.50,
                'position_sizing': '3x-6x',
                'expected_trades_per_week': '15',
                'expected_win_rate': 40.3,
                'expected_pnl_2yr': 29328,  # $14,664 × 2 years
                'features': '33 (25 V3 + 8 temporal)',
                'model': 'Separate LONG/SHORT XGBoost'
            }
        }
    ]

    status_data = {
        'generated_at': datetime.now().isoformat(),
        'binance_connection': binance_status,
        'bots': [
            get_bot_status(
                bot['name'],
                bot['log_file'],
                bot['trade_log_file'],
                bot['config']
            )
            for bot in bots
        ]
    }

    # Write to JSON file (handle NaN/Infinity values)
    output_file = BOT_DIR / 'bot_status.json'
    with open(output_file, 'w') as f:
        json.dump(sanitize_for_json(status_data), f, indent=2)

    print(f"Status updated: {output_file}")
    return status_data


if __name__ == '__main__':
    generate_status_json()
