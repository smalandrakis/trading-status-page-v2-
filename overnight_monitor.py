#!/usr/bin/env python3
"""
Overnight Monitoring Script
Runs hourly checks on:
1. Parquet refresh status for both MNQ and BTC bots
2. Signal match between live and backtest (<5% threshold)
3. Auto-audit and force refresh if discrepancies found

Logs all results to logs/overnight_monitor.log
"""

import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import ta
import requests
import os
import sys
import time
from datetime import datetime, timedelta
import pytz

sys.path.insert(0, '.')
from feature_engineering import (
    add_time_features, add_price_features, add_daily_context_features,
    add_lagged_indicator_features, add_indicator_changes
)

LOG_FILE = 'logs/overnight_monitor.log'
ALERT_THRESHOLD = 0.05  # 5%
MNQ_QQQ_RATIO = 41.2

def log(msg, level='INFO'):
    """Log message to file and console."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_line = f"[{timestamp}] [{level}] {msg}"
    print(log_line)
    with open(LOG_FILE, 'a') as f:
        f.write(log_line + '\n')

def check_parquet_freshness():
    """Check if parquets are being refreshed properly."""
    results = {'mnq': {}, 'btc': {}}
    
    # MNQ Parquet
    try:
        mnq = pd.read_parquet('data/QQQ_features.parquet')
        mnq_age = (datetime.now() - mnq.index[-1].to_pydatetime().replace(tzinfo=None)).total_seconds() / 60
        results['mnq'] = {
            'rows': len(mnq),
            'latest': str(mnq.index[-1]),
            'close': mnq['Close'].iloc[-1],
            'age_min': mnq_age,
            'features': len(mnq.columns),
            'status': 'OK' if mnq_age < 120 else 'STALE'  # Allow 2 hours for market halt
        }
    except Exception as e:
        results['mnq'] = {'status': 'ERROR', 'error': str(e)}
    
    # BTC Parquet
    try:
        btc = pd.read_parquet('data/BTC_features.parquet')
        btc_age = (datetime.now() - btc.index[-1].to_pydatetime().replace(tzinfo=None)).total_seconds() / 60
        results['btc'] = {
            'rows': len(btc),
            'latest': str(btc.index[-1]),
            'close': btc['Close'].iloc[-1],
            'age_min': btc_age,
            'features': len(btc.columns),
            'status': 'OK' if btc_age < 15 else 'STALE'  # BTC should refresh every 5 min
        }
    except Exception as e:
        results['btc'] = {'status': 'ERROR', 'error': str(e)}
    
    return results

def download_fresh_mnq_data():
    """Download fresh MNQ data from Yahoo."""
    paris = pytz.timezone('Europe/Paris')
    et = pytz.timezone('US/Eastern')
    now_et = datetime.now(et)
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    is_market_hours = market_open <= now_et <= market_close and now_et.weekday() < 5
    
    if is_market_hours:
        data = yf.download('QQQ', period='60d', interval='5m', progress=False)
        source = 'QQQ'
    else:
        data = yf.download('NQ=F', period='60d', interval='5m', progress=False)
        source = 'NQ=F'
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    
    if source == 'NQ=F' and len(data) > 0:
        data['Open'] = data['Open'] / MNQ_QQQ_RATIO
        data['High'] = data['High'] / MNQ_QQQ_RATIO
        data['Low'] = data['Low'] / MNQ_QQQ_RATIO
        data['Close'] = data['Close'] / MNQ_QQQ_RATIO
        data['Volume'] = data['Volume'] * 133
    
    if data.index.tz is not None:
        data.index = data.index.tz_convert(paris).tz_localize(None)
    
    return data, source

def download_fresh_btc_data():
    """Download fresh BTC data from Binance."""
    url = 'https://api.binance.com/api/v3/klines'
    params = {'symbol': 'BTCUSDT', 'interval': '5m', 'limit': 1000}
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    klines = response.json()
    
    df = pd.DataFrame(klines, columns=[
        'open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    df = df.set_index('timestamp')
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = df[col].astype(float)
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]

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
    df = add_lagged_indicator_features(df, lookback_periods=[1, 2, 3, 5, 10, 20, 50])
    df = add_indicator_changes(df)
    df = df.ffill().bfill().fillna(0).replace([np.inf, -np.inf], 0)
    return df

def get_live_signals(bot_type):
    """Get latest live signals from signal log."""
    today = datetime.now().strftime('%Y-%m-%d')
    if bot_type == 'mnq':
        log_file = f'signal_logs/mnq_signals_{today}.csv'
    else:
        log_file = f'signal_logs/btc_signals_{today}.csv'
    
    if not os.path.exists(log_file):
        return None, None
    
    df = pd.read_csv(log_file)
    if len(df) == 0:
        return None, None
    
    latest = df.iloc[-1]
    timestamp = latest.get('timestamp', 'unknown')
    
    probs = {
        '2h_0.5pct': float(latest['prob_2h_0.5pct']),
        '4h_0.5pct': float(latest['prob_4h_0.5pct']),
        '2h_0.5pct_SHORT': float(latest['prob_2h_0.5pct_SHORT']),
        '4h_0.5pct_SHORT': float(latest['prob_4h_0.5pct_SHORT']),
    }
    return probs, timestamp

def check_signal_match(bot_type):
    """Check if backtest predictions match live signals."""
    results = {'bot': bot_type, 'models': {}, 'status': 'OK'}
    
    try:
        # Load models
        if bot_type == 'mnq':
            model_dir = 'models_mnq_v2'
        else:
            model_dir = 'models_btc_v2'
        
        models = {
            '2h_0.5pct': joblib.load(f'{model_dir}/model_2h_0.5pct.joblib'),
            '4h_0.5pct': joblib.load(f'{model_dir}/model_4h_0.5pct.joblib'),
            '2h_0.5pct_SHORT': joblib.load(f'{model_dir}/model_2h_0.5pct_SHORT.joblib'),
            '4h_0.5pct_SHORT': joblib.load(f'{model_dir}/model_4h_0.5pct_SHORT.joblib'),
        }
        feature_cols = list(models['2h_0.5pct'].feature_names_in_)
        
        # Download fresh data
        if bot_type == 'mnq':
            data, source = download_fresh_mnq_data()
        else:
            data = download_fresh_btc_data()
            source = 'Binance'
        
        if len(data) == 0:
            results['status'] = 'NO_DATA'
            return results
        
        # Calculate features
        df = calculate_features(data)
        
        # Get backtest predictions
        X = df[feature_cols].iloc[[-1]].fillna(0)
        for col in feature_cols:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_cols]
        
        backtest_probs = {name: model.predict_proba(X)[0, 1] for name, model in models.items()}
        
        # Get live signals
        live_probs, live_timestamp = get_live_signals(bot_type)
        if live_probs is None:
            results['status'] = 'NO_LIVE_SIGNALS'
            return results
        
        results['live_timestamp'] = live_timestamp
        results['data_source'] = source
        
        # Compare
        max_diff = 0
        for model_name in models.keys():
            bt = backtest_probs[model_name]
            live = live_probs[model_name]
            diff = abs(bt - live)
            max_diff = max(max_diff, diff)
            
            status = 'OK' if diff < ALERT_THRESHOLD else 'ALERT'
            results['models'][model_name] = {
                'backtest': bt,
                'live': live,
                'diff': diff,
                'status': status
            }
            
            if status == 'ALERT':
                results['status'] = 'ALERT'
        
        results['max_diff'] = max_diff
        
    except Exception as e:
        results['status'] = 'ERROR'
        results['error'] = str(e)
    
    return results

def force_refresh_parquet(bot_type):
    """Force refresh parquet with fresh data."""
    log(f"Force refreshing {bot_type.upper()} parquet...", 'WARNING')
    
    try:
        if bot_type == 'mnq':
            data, source = download_fresh_mnq_data()
            parquet_path = 'data/QQQ_features.parquet'
        else:
            data = download_fresh_btc_data()
            source = 'Binance'
            parquet_path = 'data/BTC_features.parquet'
        
        if len(data) == 0:
            log(f"No data available for {bot_type.upper()}", 'ERROR')
            return False
        
        # Calculate features
        df = calculate_features(data)
        
        # Save
        df.to_parquet(parquet_path)
        log(f"Force refreshed {bot_type.upper()} parquet: {len(df)} rows, ends at {df.index[-1]}", 'INFO')
        return True
        
    except Exception as e:
        log(f"Force refresh failed for {bot_type.upper()}: {e}", 'ERROR')
        return False

def run_full_audit(bot_type):
    """Run full audit when discrepancies are found."""
    log(f"=" * 60, 'INFO')
    log(f"FULL AUDIT: {bot_type.upper()}", 'WARNING')
    log(f"=" * 60, 'INFO')
    
    # Check parquet
    parquet_path = 'data/QQQ_features.parquet' if bot_type == 'mnq' else 'data/BTC_features.parquet'
    parquet = pd.read_parquet(parquet_path)
    
    log(f"Parquet rows: {len(parquet)}", 'INFO')
    log(f"Parquet latest: {parquet.index[-1]}", 'INFO')
    log(f"Parquet Close: {parquet['Close'].iloc[-1]:.2f}", 'INFO')
    
    # Download fresh data
    if bot_type == 'mnq':
        fresh_data, source = download_fresh_mnq_data()
    else:
        fresh_data = download_fresh_btc_data()
        source = 'Binance'
    
    log(f"Fresh data source: {source}", 'INFO')
    log(f"Fresh data rows: {len(fresh_data)}", 'INFO')
    log(f"Fresh data latest: {fresh_data.index[-1]}", 'INFO')
    log(f"Fresh data Close: {fresh_data['Close'].iloc[-1]:.2f}", 'INFO')
    
    # Check OHLCV match at common timestamp
    common_time = parquet.index[-1]
    if common_time in fresh_data.index:
        fresh_close = fresh_data.loc[common_time, 'Close']
        parquet_close = parquet.loc[common_time, 'Close']
        close_diff = abs(fresh_close - parquet_close)
        log(f"OHLCV at {common_time}: Parquet={parquet_close:.2f}, Fresh={fresh_close:.2f}, Diff={close_diff:.2f}", 'INFO')
        
        if close_diff > 1:
            log(f"OHLCV MISMATCH DETECTED!", 'ERROR')
            return True  # Needs refresh
    
    # Force refresh if audit finds issues
    return True

def run_hourly_check():
    """Run all hourly checks."""
    log("=" * 70, 'INFO')
    log("HOURLY CHECK STARTED", 'INFO')
    log("=" * 70, 'INFO')
    
    issues_found = False
    
    # 1. Check parquet freshness
    log("\n### PARQUET FRESHNESS CHECK ###", 'INFO')
    parquet_status = check_parquet_freshness()
    
    for bot, status in parquet_status.items():
        if status.get('status') == 'OK':
            log(f"{bot.upper()}: OK - {status['rows']} rows, age={status['age_min']:.1f}min, close={status['close']:.2f}", 'INFO')
        elif status.get('status') == 'STALE':
            log(f"{bot.upper()}: STALE - age={status['age_min']:.1f}min", 'WARNING')
            issues_found = True
        else:
            log(f"{bot.upper()}: ERROR - {status.get('error', 'unknown')}", 'ERROR')
            issues_found = True
    
    # 2. Check signal match
    log("\n### SIGNAL MATCH CHECK ###", 'INFO')
    
    for bot_type in ['mnq', 'btc']:
        match_result = check_signal_match(bot_type)
        
        if match_result['status'] == 'OK':
            log(f"{bot_type.upper()}: OK - max_diff={match_result.get('max_diff', 0):.2%}", 'INFO')
            for model, data in match_result['models'].items():
                log(f"  {model}: BT={data['backtest']:.2%} Live={data['live']:.2%} Diff={data['diff']:.2%}", 'INFO')
        elif match_result['status'] == 'ALERT':
            log(f"{bot_type.upper()}: ALERT - max_diff={match_result.get('max_diff', 0):.2%}", 'WARNING')
            for model, data in match_result['models'].items():
                status_icon = '⚠️' if data['status'] == 'ALERT' else '✓'
                log(f"  {status_icon} {model}: BT={data['backtest']:.2%} Live={data['live']:.2%} Diff={data['diff']:.2%}", 'WARNING' if data['status'] == 'ALERT' else 'INFO')
            issues_found = True
        elif match_result['status'] in ['NO_DATA', 'NO_LIVE_SIGNALS']:
            log(f"{bot_type.upper()}: {match_result['status']}", 'WARNING')
        else:
            log(f"{bot_type.upper()}: ERROR - {match_result.get('error', 'unknown')}", 'ERROR')
            issues_found = True
    
    # 3. If issues found, run audit and force refresh
    if issues_found:
        log("\n### ISSUES DETECTED - RUNNING AUDIT ###", 'WARNING')
        
        for bot_type in ['mnq', 'btc']:
            match_result = check_signal_match(bot_type)
            if match_result['status'] in ['ALERT', 'ERROR']:
                needs_refresh = run_full_audit(bot_type)
                if needs_refresh:
                    force_refresh_parquet(bot_type)
                    # Re-check after refresh
                    time.sleep(5)
                    new_result = check_signal_match(bot_type)
                    if new_result['status'] == 'OK':
                        log(f"{bot_type.upper()}: Fixed after refresh", 'INFO')
                    else:
                        log(f"{bot_type.upper()}: Still has issues after refresh", 'ERROR')
    
    log("\n### HOURLY CHECK COMPLETED ###", 'INFO')
    log("=" * 70, 'INFO')
    
    return not issues_found

def main():
    """Main loop - run checks every hour."""
    log("=" * 70, 'INFO')
    log("OVERNIGHT MONITOR STARTED", 'INFO')
    log(f"Check interval: 1 hour", 'INFO')
    log(f"Alert threshold: {ALERT_THRESHOLD:.0%}", 'INFO')
    log("=" * 70, 'INFO')
    
    check_count = 0
    
    while True:
        try:
            check_count += 1
            log(f"\n{'='*70}", 'INFO')
            log(f"CHECK #{check_count} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 'INFO')
            
            success = run_hourly_check()
            
            if success:
                log("All checks passed ✓", 'INFO')
            else:
                log("Issues detected and addressed", 'WARNING')
            
            # Wait 1 hour
            log(f"Next check in 1 hour...", 'INFO')
            time.sleep(3600)
            
        except KeyboardInterrupt:
            log(f"\nMonitor stopped after {check_count} checks", 'INFO')
            break
        except Exception as e:
            log(f"Error in main loop: {e}", 'ERROR')
            time.sleep(300)  # Wait 5 min on error

if __name__ == '__main__':
    main()
