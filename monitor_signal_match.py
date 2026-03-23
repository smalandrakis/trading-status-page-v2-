#!/usr/bin/env python3
"""
Signal Match Monitor - Compares live bot signals with backtest predictions.

Runs continuously to ensure live probabilities match what backtest would produce.
Alerts if discrepancy exceeds threshold (default 5%).

Usage:
    python monitor_signal_match.py [--interval 60] [--threshold 0.05] [--bot mnq|btc]
"""

import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import ta
import os
import sys
import time
import argparse
from datetime import datetime, timedelta
import pytz

sys.path.insert(0, '.')
from feature_engineering import (
    add_time_features, add_price_features, add_daily_context_features,
    add_lagged_indicator_features, add_indicator_changes
)
import config

# Alert thresholds
DEFAULT_THRESHOLD = 0.05  # 5% difference triggers alert
CRITICAL_THRESHOLD = 0.10  # 10% difference is critical

def load_models(bot_type='mnq'):
    """Load models for the specified bot."""
    if bot_type == 'mnq':
        model_dir = 'models_mnq_v2'
        models = {
            '2h_0.5pct': joblib.load(f'{model_dir}/model_2h_0.5pct.joblib'),
            '4h_0.5pct': joblib.load(f'{model_dir}/model_4h_0.5pct.joblib'),
            '2h_0.5pct_SHORT': joblib.load(f'{model_dir}/model_2h_0.5pct_SHORT.joblib'),
            '4h_0.5pct_SHORT': joblib.load(f'{model_dir}/model_4h_0.5pct_SHORT.joblib'),
        }
    elif bot_type == 'btc':
        model_dir = 'models_btc_v2'
        models = {
            '2h_0.5pct': joblib.load(f'{model_dir}/model_2h_0.5pct.joblib'),
            '4h_0.5pct': joblib.load(f'{model_dir}/model_4h_0.5pct.joblib'),
            '2h_0.5pct_SHORT': joblib.load(f'{model_dir}/model_2h_0.5pct_SHORT.joblib'),
            '4h_0.5pct_SHORT': joblib.load(f'{model_dir}/model_4h_0.5pct_SHORT.joblib'),
        }
    elif bot_type == 'spy':
        # SPY uses MNQ v2 models (they transfer well to SPY)
        model_dir = 'models_mnq_v2'
        models = {
            '4h_0.5pct': joblib.load(f'{model_dir}/model_4h_0.5pct.joblib'),
            '2h_0.5pct_SHORT': joblib.load(f'{model_dir}/model_2h_0.5pct_SHORT.joblib'),
            '4h_0.5pct_SHORT': joblib.load(f'{model_dir}/model_4h_0.5pct_SHORT.joblib'),
        }
    else:
        raise ValueError(f"Unknown bot type: {bot_type}")
    
    # Get feature columns from first available model
    first_model = list(models.values())[0]
    feature_cols = list(first_model.feature_names_in_)
    return models, feature_cols

def download_fresh_data(bot_type='mnq'):
    """Download fresh data from the same source as the bot.
    
    Note: Cumulative indicators (OBV, ADI) will have different absolute values
    than the bot's parquet, but testing shows predictions are identical because
    the HistGradientBoostingClassifier is robust to absolute value differences.
    """
    if bot_type == 'mnq':
        paris = pytz.timezone('Europe/Paris')
        et = pytz.timezone('US/Eastern')
        now_et = datetime.now(et)
        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        is_market_hours = market_open <= now_et <= market_close and now_et.weekday() < 5
        
        MNQ_QQQ_RATIO = 41.2
        
        if is_market_hours:
            # Use QQQ during market hours (same as bot)
            data = yf.download('QQQ', period='60d', interval='5m', progress=False)
            data_source = 'QQQ'
        else:
            # Use NQ=F outside market hours (same as bot)
            data = yf.download('NQ=F', period='60d', interval='5m', progress=False)
            data_source = 'NQ=F'
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
        
        # Convert NQ to QQQ scale if needed (same as bot)
        if data_source == 'NQ=F' and len(data) > 0:
            data['Open'] = data['Open'] / MNQ_QQQ_RATIO
            data['High'] = data['High'] / MNQ_QQQ_RATIO
            data['Low'] = data['Low'] / MNQ_QQQ_RATIO
            data['Close'] = data['Close'] / MNQ_QQQ_RATIO
            data['Volume'] = data['Volume'] * 133
        
        # Convert to Paris time (same as bot)
        if data.index.tz is not None:
            data.index = data.index.tz_convert(paris).tz_localize(None)
        
        return data
    
    elif bot_type == 'btc':
        # BTC uses Binance BTCUSDT (same as bot)
        # Fetch 2000 bars for stable cumulative indicators (requires 2 API calls)
        import requests
        url = 'https://api.binance.com/api/v3/klines'
        all_klines = []
        end_time = None
        target_bars = 2000
        
        while len(all_klines) < target_bars:
            params = {'symbol': 'BTCUSDT', 'interval': '5m', 'limit': 1000}
            if end_time:
                params['endTime'] = end_time
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            klines = response.json()
            if not klines:
                break
            all_klines = klines + all_klines  # prepend older data
            end_time = klines[0][0] - 1
            if len(klines) < 1000:
                break
        
        # Take most recent target_bars
        klines = all_klines[-target_bars:] if len(all_klines) > target_bars else all_klines
        
        df = pd.DataFrame(klines, columns=[
            'open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        df = df.set_index('timestamp')
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = df[col].astype(float)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        return df
    
    elif bot_type == 'spy':
        # SPY bot uses NQ Futures scaled to QQQ (same as MNQ bot)
        # because SPY uses MNQ models trained on QQQ data
        paris = pytz.timezone('Europe/Paris')
        et = pytz.timezone('US/Eastern')
        now_et = datetime.now(et)
        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        is_market_hours = market_open <= now_et <= market_close and now_et.weekday() < 5
        
        MNQ_QQQ_RATIO = 41.2
        
        if is_market_hours:
            # Use QQQ during market hours (same as MNQ bot)
            data = yf.download('QQQ', period='60d', interval='5m', progress=False)
            data_source = 'QQQ'
        else:
            # Use NQ=F outside market hours (same as MNQ/SPY bots)
            data = yf.download('NQ=F', period='60d', interval='5m', progress=False)
            data_source = 'NQ=F'
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
        
        # Convert NQ to QQQ scale if needed (same as bot)
        if data_source == 'NQ=F' and len(data) > 0:
            data['Open'] = data['Open'] / MNQ_QQQ_RATIO
            data['High'] = data['High'] / MNQ_QQQ_RATIO
            data['Low'] = data['Low'] / MNQ_QQQ_RATIO
            data['Close'] = data['Close'] / MNQ_QQQ_RATIO
            data['Volume'] = data['Volume'] * 133
        
        # Convert to Paris time (same as bot)
        if data.index.tz is not None:
            data.index = data.index.tz_convert(paris).tz_localize(None)
        
        return data
    
    else:
        raise ValueError(f"Unknown bot type: {bot_type}")

def calculate_features(data):
    """Calculate all features for the data."""
    df = data.copy()
    df = ta.add_all_ta_features(
        df, open='Open', high='High', low='Low',
        close='Close', volume='Volume', fillna=True
    )
    df = add_time_features(df)
    df = add_price_features(df)
    df = add_daily_context_features(df)
    df = add_lagged_indicator_features(df, config.LOOKBACK_PERIODS)
    df = add_indicator_changes(df)
    df = df.fillna(0).replace([np.inf, -np.inf], 0)
    return df

def get_backtest_predictions(models, feature_cols, df):
    """Generate predictions from fresh data."""
    for f in feature_cols:
        if f not in df.columns:
            df[f] = 0
    
    X = df[feature_cols].iloc[[-1]].fillna(0).replace([np.inf, -np.inf], 0)
    
    predictions = {}
    for model_name, model in models.items():
        predictions[model_name] = model.predict_proba(X)[0, 1]
    
    return predictions

def get_live_signals(bot_type='mnq'):
    """Get latest live signals from signal log."""
    today = datetime.now().strftime('%Y-%m-%d')
    
    if bot_type == 'mnq':
        log_file = f'signal_logs/mnq_signals_{today}.csv'
        prob_cols = {
            'prob_2h_0.5pct': '2h_0.5pct',
            'prob_4h_0.5pct': '4h_0.5pct',
            'prob_2h_0.5pct_SHORT': '2h_0.5pct_SHORT',
            'prob_4h_0.5pct_SHORT': '4h_0.5pct_SHORT',
        }
    elif bot_type == 'btc':
        log_file = f'signal_logs/btc_signals_{today}.csv'
        prob_cols = {
            'prob_2h_0.5pct': '2h_0.5pct',
            'prob_4h_0.5pct': '4h_0.5pct',
            'prob_2h_0.5pct_SHORT': '2h_0.5pct_SHORT',
            'prob_4h_0.5pct_SHORT': '4h_0.5pct_SHORT',
        }
    elif bot_type == 'spy':
        log_file = f'signal_logs/spy_signals_{today}.csv'
        prob_cols = {
            'prob_4h_0.5pct': '4h_0.5pct',
            'prob_2h_0.5pct_SHORT': '2h_0.5pct_SHORT',
            'prob_4h_0.5pct_SHORT': '4h_0.5pct_SHORT',
        }
    else:
        raise ValueError(f"Unknown bot type: {bot_type}")
    
    if not os.path.exists(log_file):
        return None, None
    
    df = pd.read_csv(log_file)
    if len(df) == 0:
        return None, None
    
    latest = df.iloc[-1]
    timestamp = latest.get('timestamp', 'unknown')
    
    live_probs = {}
    for csv_col, model_name in prob_cols.items():
        if csv_col in latest:
            live_probs[model_name] = float(latest[csv_col])
    
    return live_probs, timestamp

def compare_signals(backtest_probs, live_probs, threshold=DEFAULT_THRESHOLD):
    """Compare backtest and live probabilities."""
    results = []
    has_alert = False
    has_critical = False
    
    for model_name in backtest_probs:
        if model_name not in live_probs:
            continue
        
        bt = backtest_probs[model_name]
        live = live_probs[model_name]
        diff = abs(bt - live)
        
        status = 'OK'
        if diff >= CRITICAL_THRESHOLD:
            status = 'CRITICAL'
            has_critical = True
        elif diff >= threshold:
            status = 'ALERT'
            has_alert = True
        
        results.append({
            'model': model_name,
            'backtest': bt,
            'live': live,
            'diff': diff,
            'status': status
        })
    
    return results, has_alert, has_critical

# Features that MUST match exactly (price-based, don't depend on history)
# These are the only features we can reliably compare between fresh and parquet
# because cumulative indicators (RSI, MACD, etc.) depend on the full history window
PRICE_BASED_FEATURES = [
    'Close', 'Open', 'High', 'Low', 'Volume',
    'hour', 'day_of_week', 'minute',  # Time features
]

# Thresholds for feature mismatch
FEATURE_THRESHOLD = 0.0001  # 0.01% difference triggers alert (prices should match exactly)
FEATURE_CRITICAL = 0.001   # 0.1% difference is critical

def compare_features(fresh_df, parquet_df, feature_cols):
    """
    Compare OHLCV values between fresh data and parquet for the SAME completed bar.
    
    IMPORTANT: We can only compare completed bars. The parquet's last bar must be
    OLDER than fresh data's last bar (meaning parquet's bar is complete).
    
    We only compare OHLCV (price-based features) because cumulative indicators
    (RSI, MACD, etc.) depend on the full history window which may differ.
    
    Returns:
        Tuple of (results, has_alert, has_critical, comparison_info)
    """
    results = []
    has_alert = False
    has_critical = False
    
    # Normalize timestamps to tz-naive for comparison
    parquet_last_ts = parquet_df.index[-1]
    fresh_last_ts = fresh_df.index[-1]
    
    # Convert to tz-naive if needed
    if hasattr(parquet_last_ts, 'tz') and parquet_last_ts.tz is not None:
        parquet_last_ts = parquet_last_ts.tz_localize(None)
    if hasattr(fresh_last_ts, 'tz') and fresh_last_ts.tz is not None:
        fresh_last_ts = fresh_last_ts.tz_localize(None)
    
    comparison_info = {
        'parquet_ts': str(parquet_last_ts),
        'fresh_last_ts': str(fresh_last_ts),
    }
    
    # Only compare if parquet's bar is OLDER than fresh's last bar
    # This ensures parquet's bar is complete (not the current incomplete bar)
    if parquet_last_ts >= fresh_last_ts:
        comparison_info['skip_reason'] = 'Parquet bar may be incomplete (same as or newer than fresh)'
        return [], False, False, comparison_info
    
    # Find parquet's last bar in fresh data
    if parquet_last_ts not in fresh_df.index:
        comparison_info['skip_reason'] = f'Parquet bar {parquet_last_ts} not in fresh data'
        return [], False, False, comparison_info
    
    fresh_row = fresh_df.loc[parquet_last_ts]
    parquet_row = parquet_df.iloc[-1]
    comparison_info['comparing_ts'] = str(parquet_last_ts)
    
    # Compare only price-based features (cumulative indicators depend on history window)
    for feature in PRICE_BASED_FEATURES:
        if feature not in fresh_df.columns or feature not in parquet_df.columns:
            continue
        
        fresh_val = float(fresh_row[feature]) if pd.notna(fresh_row[feature]) else 0.0
        parquet_val = float(parquet_row[feature]) if pd.notna(parquet_row[feature]) else 0.0
        
        # Calculate percentage difference
        abs_diff = abs(fresh_val - parquet_val)
        
        # For small values (< 1), use absolute difference
        if abs(parquet_val) < 1.0 and abs(fresh_val) < 1.0:
            pct_diff = abs_diff
        elif abs(parquet_val) > 1e-10:
            pct_diff = abs_diff / abs(parquet_val)
        elif abs(fresh_val) > 1e-10:
            pct_diff = abs_diff / abs(fresh_val)
        else:
            pct_diff = 0.0 if abs_diff < 1e-10 else 1.0
        
        # Determine status
        status = 'OK'
        if pct_diff >= FEATURE_CRITICAL:
            status = 'CRITICAL'
            has_critical = True
        elif pct_diff >= FEATURE_THRESHOLD:
            status = 'ALERT'
            has_alert = True
        
        results.append({
            'feature': feature,
            'fresh': fresh_val,
            'parquet': parquet_val,
            'pct_diff': pct_diff,
            'status': status
        })
    
    return results, has_alert, has_critical, comparison_info

def log_comparison(results, bot_type, timestamp, fresh_ohlcv=None, parquet_ohlcv=None, feature_results=None):
    """Log comparison results to file with OHLCV and feature verification."""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = f'{log_dir}/signal_match_{bot_type}.log'
    
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Signal Match Check: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Live signal timestamp: {timestamp}\n")
        
        # Log OHLCV comparison to verify data sources
        if fresh_ohlcv and parquet_ohlcv:
            f.write(f"Fresh bar: {fresh_ohlcv['timestamp']} Close=${fresh_ohlcv['close']:.2f}\n")
            f.write(f"Parquet bar: {parquet_ohlcv['timestamp']} Close=${parquet_ohlcv['close']:.2f}\n")
            price_diff = abs(fresh_ohlcv['close'] - parquet_ohlcv['close'])
            f.write(f"Price diff: ${price_diff:.2f}\n")
        
        f.write(f"{'='*60}\n")
        f.write(f"SIGNAL PROBABILITIES:\n")
        
        for r in results:
            status_icon = '✓' if r['status'] == 'OK' else ('⚠️' if r['status'] == 'ALERT' else '🚨')
            f.write(f"{status_icon} {r['model']}: BT={r['backtest']:.4f} Live={r['live']:.4f} Diff={r['diff']:.4f} [{r['status']}]\n")
        
        # Log feature comparison results
        if feature_results:
            f.write(f"\n{'-'*60}\n")
            f.write(f"FEATURE VALIDATION ({len(feature_results)} features):\n")
            
            # Count statuses
            ok_count = sum(1 for r in feature_results if r['status'] == 'OK')
            alert_count = sum(1 for r in feature_results if r['status'] == 'ALERT')
            critical_count = sum(1 for r in feature_results if r['status'] == 'CRITICAL')
            
            f.write(f"Summary: {ok_count} OK, {alert_count} ALERT, {critical_count} CRITICAL\n")
            
            # Only log non-OK features to keep log manageable
            non_ok = [r for r in feature_results if r['status'] != 'OK']
            if non_ok:
                f.write(f"\nMismatched features:\n")
                for r in non_ok:
                    status_icon = '⚠️' if r['status'] == 'ALERT' else '🚨'
                    f.write(f"{status_icon} {r['feature']}: Fresh={r['fresh']:.6f} Parquet={r['parquet']:.6f} Diff={r['pct_diff']:.2%} [{r['status']}]\n")
            else:
                f.write(f"All features match within tolerance.\n")

def run_monitor(bot_type='mnq', interval=60, threshold=DEFAULT_THRESHOLD, continuous=True):
    """Run the signal match monitor."""
    print(f"{'='*60}")
    print(f"SIGNAL MATCH MONITOR - {bot_type.upper()}")
    print(f"{'='*60}")
    print(f"Check interval: {interval}s")
    print(f"Alert threshold: {threshold*100:.1f}%")
    print(f"Critical threshold: {CRITICAL_THRESHOLD*100:.1f}%")
    print(f"{'='*60}\n")
    
    # Load models once
    print("Loading models...")
    models, feature_cols = load_models(bot_type)
    print(f"Loaded {len(models)} models with {len(feature_cols)} features\n")
    
    alert_count = 0
    check_count = 0
    
    while True:
        try:
            check_count += 1
            now = datetime.now().strftime('%H:%M:%S')
            
            # Download fresh data (matching parquet history range)
            data = download_fresh_data(bot_type)
            if len(data) == 0:
                print(f"[{now}] No data available")
                if not continuous:
                    break
                time.sleep(interval)
                continue
            
            # Store fresh OHLCV info for logging
            fresh_ohlcv = {
                'timestamp': str(data.index[-1]),
                'close': float(data['Close'].iloc[-1])
            }
            
            # Calculate features from fresh data
            df = calculate_features(data)
            
            # Get backtest predictions
            backtest_probs = get_backtest_predictions(models, feature_cols, df)
            
            # Get live signals
            live_probs, live_timestamp = get_live_signals(bot_type)
            if live_probs is None:
                print(f"[{now}] No live signals available")
                if not continuous:
                    break
                time.sleep(interval)
                continue
            
            # Load parquet to compare OHLCV and features
            parquet_ohlcv = None
            parquet_df = None
            feature_results = None
            feature_alert = False
            feature_critical = False
            try:
                if bot_type == 'btc':
                    parquet_df = pd.read_parquet('data/BTC_features.parquet')
                else:
                    # Both MNQ and SPY use QQQ_features.parquet (SPY uses MNQ models)
                    parquet_df = pd.read_parquet('data/QQQ_features.parquet')
                parquet_ohlcv = {
                    'timestamp': str(parquet_df.index[-1]),
                    'close': float(parquet_df['Close'].iloc[-1])
                }
                
                # Compare features between fresh calculation and parquet
                feature_results, feature_alert, feature_critical, comparison_info = compare_features(df, parquet_df, feature_cols)
            except Exception as e:
                print(f"[{now}] Error loading parquet for feature comparison: {e}")
                comparison_info = None
            
            # Compare signal probabilities
            results, has_alert, has_critical = compare_signals(backtest_probs, live_probs, threshold)
            
            # Combine alerts from both signal and feature comparison
            has_alert = has_alert or feature_alert
            has_critical = has_critical or feature_critical
            
            # Log results with OHLCV and feature verification
            log_comparison(results, bot_type, live_timestamp, fresh_ohlcv, parquet_ohlcv, feature_results)
            
            # Print results
            if has_critical:
                print(f"\n🚨 [{now}] CRITICAL MISMATCH DETECTED!")
                alert_count += 1
            elif has_alert:
                print(f"\n⚠️  [{now}] Alert: Signal or feature mismatch detected")
                alert_count += 1
            else:
                # Show feature validation summary
                if feature_results and len(feature_results) > 0:
                    ok_count = sum(1 for r in feature_results if r['status'] == 'OK')
                    print(f"✓  [{now}] Signals OK, Features {ok_count}/{len(feature_results)} OK (check #{check_count})")
                elif comparison_info and not comparison_info.get('windows_match', True):
                    print(f"✓  [{now}] Signals OK, Features skipped (windows differ) (check #{check_count})")
                else:
                    print(f"✓  [{now}] All signals match (check #{check_count})")
            
            # Print details if alert
            if has_alert or has_critical:
                print(f"   Live timestamp: {live_timestamp}")
                # Signal mismatches
                for r in results:
                    if r['status'] != 'OK':
                        print(f"   SIGNAL {r['model']}: BT={r['backtest']:.2%} Live={r['live']:.2%} Diff={r['diff']:.2%}")
                # Feature mismatches
                if feature_results:
                    for r in feature_results:
                        if r['status'] != 'OK':
                            print(f"   FEATURE {r['feature']}: Fresh={r['fresh']:.4f} Parquet={r['parquet']:.4f} Diff={r['pct_diff']:.2%}")
            
            if not continuous:
                break
            
            time.sleep(interval)
            
        except KeyboardInterrupt:
            print(f"\n\nMonitor stopped. Checks: {check_count}, Alerts: {alert_count}")
            break
        except Exception as e:
            print(f"[{now}] Error: {e}")
            if not continuous:
                break
            time.sleep(interval)

def main():
    parser = argparse.ArgumentParser(description='Monitor live vs backtest signal match')
    parser.add_argument('--bot', type=str, default='mnq', choices=['mnq', 'btc', 'spy'],
                        help='Bot to monitor (default: mnq)')
    parser.add_argument('--interval', type=int, default=60,
                        help='Check interval in seconds (default: 60)')
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD,
                        help=f'Alert threshold (default: {DEFAULT_THRESHOLD})')
    parser.add_argument('--once', action='store_true',
                        help='Run once and exit')
    
    args = parser.parse_args()
    
    run_monitor(
        bot_type=args.bot,
        interval=args.interval,
        threshold=args.threshold,
        continuous=not args.once
    )

if __name__ == '__main__':
    main()
