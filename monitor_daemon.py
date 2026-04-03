#!/usr/bin/env python3
"""
Monitor Daemon - Automated monitoring for trading bots.

Runs hourly checks on:
- Parquet health and completeness (triggers refresh if needed)
- Signal match (parquet vs fresh data)
- Current probabilities for all bots
- Open positions
- P&L last 24 hours

Outputs status.json for Netlify status page.
"""

import os
import sys
import json
import time
import logging
import subprocess
import signal
import threading
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import ta

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/monitor_daemon.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
CHECK_INTERVAL_SECONDS = 1800  # 30 minutes
CHECK_TIMEOUT_SECONDS = 300    # 5 minutes max per check cycle (watchdog)
BINANCE_API = "https://api.binance.com/api/v3"
STATUS_FILE = "status_page/status.json"
BTC_PARQUET = "data/BTC_features.parquet"
QQQ_PARQUET = "data/QQQ_features.parquet"
BTC_MIN_BARS = 1500
QQQ_MIN_BARS = 4000  # Yahoo Finance 5-min data limit is ~60 days (~4500 bars)

# Model paths - Updated Jan 7, 2026
# BTC: ALL models enabled (2h, 4h, 6h LONG + 2h, 4h SHORT)
# Overlap analysis showed only 25-50% overlap - each model catches unique opportunities
BTC_MODELS = {
    '2h_0.5pct': 'models_btc_v2/model_2h_0.5pct.joblib',
    '4h_0.5pct': 'models_btc_v2/model_4h_0.5pct.joblib',
    '6h_0.5pct': 'models_btc_v2/model_6h_0.5pct.joblib',
    '2h_0.5pct_SHORT': 'models_btc_v2/model_2h_0.5pct_SHORT.joblib',
    '4h_0.5pct_SHORT': 'models_btc_v2/model_4h_0.5pct_SHORT.joblib',
}
# MNQ/SPY: 2h + 4h LONG/SHORT
MNQ_MODELS = {
    '2h_0.5pct': 'models_mnq_v2/model_2h_0.5pct.joblib',
    '4h_0.5pct': 'models_mnq_v2/model_4h_0.5pct.joblib',
    '2h_0.5pct_SHORT': 'models_mnq_v2/model_2h_0.5pct_SHORT.joblib',
    '4h_0.5pct_SHORT': 'models_mnq_v2/model_4h_0.5pct_SHORT.joblib',
}

# SL/TP Configuration - Updated Jan 7, 2026
BOT_CONFIG = {
    'BTC': {
        'SL_PCT': 0.70,
        'TP_PCT': 1.40,
        'models': ['2h_0.5pct', '4h_0.5pct', '6h_0.5pct', '2h_0.5pct_SHORT', '4h_0.5pct_SHORT'],
        'thresholds': {
            '2h_0.5pct': 0.40,        # LONG
            '4h_0.5pct': 0.40,        # LONG
            '6h_0.5pct': 0.40,        # LONG
            '2h_0.5pct_SHORT': 0.55,  # SHORT
            '4h_0.5pct_SHORT': 0.55,  # SHORT
        },
        'enabled_long': True,
        'enabled_short': True,
    },
    'BTC_TICK': {
        'SL_PCT_LONG': 0.30,
        'TS_ACT_LONG': 0.40,
        'TS_TRAIL_LONG': 0.10,
        'SL_PCT_SHORT': 0.20,
        'TS_ACT_SHORT': 0.25,
        'TS_TRAIL_SHORT': 0.05,
        'models': ['L1_full_wide', 'L3_mom_wide', 'S1_full_cur', 'S3_mr_cur', 'S4_full_tight'],
        'thresholds': {
            'L1_full_wide': 0.60,
            'L3_mom_wide': 0.60,
            'S1_full_cur': 0.70,
            'S3_mr_cur': 0.70,
            'S4_full_tight': 0.70,
        },
        'entry_long': 'any of L1/L3 >= 60%',
        'entry_short': '2+ of S1/S3/S4 >= 70% (majority vote)',
        'enabled_long': True,
        'enabled_short': True,
        'db_path': 'tick_trades.db',
    },
    'BTC_TICK_REVERSE': {
        'SL_PCT_LONG': 0.20,
        'TS_ACT_LONG': 0.35,
        'TS_TRAIL_LONG': 0.15,
        'SL_PCT_SHORT': 0.45,
        'TS_ACT_SHORT': 0.50,
        'TS_TRAIL_SHORT': 0.15,
        'models': ['L1_full_wide', 'L3_mom_wide', 'S1_full_cur', 'S3_mr_cur', 'S4_full_tight'],
        'thresholds': {
            'L1_full_wide': 0.55,
            'S1_full_cur': 0.70,
            'S3_mr_cur': 0.70,
            'S4_full_tight': 0.70,
        },
        'entry_long': 'REVERSE: SHORT models 2+ agree → enter LONG',
        'entry_short': 'REVERSE: LONG model L1 >= 55% → enter SHORT',
        'enabled_long': True,
        'enabled_short': True,
        'db_path': 'tick_trades_reverse.db',
    },
    'MNQ_GLOBAL': {
        'SL_PCT': 0.50,
        'TP_PCT': 1.00,
        'model': 'RF (rf_global_signal.pkl)',
        'threshold': 0.55,
        'entry_long': 'RF prob > 55% at 10:30 ET',
        'entry_short': 'RF prob < 45% at 10:30 ET',
        'entry_time': '10:30 ET',
        'exit_time': '15:55 ET',
        'max_trades_per_day': 1,
        'position_size': '1 MNQ ($2/point)',
        'enabled_long': True,
        'enabled_short': True,
        'db_path': 'mnq_global_trades.db',
        'position_file': 'mnq_global_position.json',
    },
    'BTC_TPSL': {
        'SL_PCT': 0.50,
        'TP_PCT': 1.00,
        'strategies': ['M1_long_hgb70', 'M2_long_hgb60'],
        'entry_M1': 'LONG HGB P(TP)>70% (81.5% WR, +0.72% EV)',
        'entry_M2': 'LONG HGB P(TP)>60% (62.2% WR, +0.43% EV)',
        'max_positions': 2,
        'enabled_long': True,
        'enabled_short': True,
        'db_path': 'tpsl_trades.db',
    },
    'BTC_V3': {
        'SL_PCT': 0.50,
        'TP_PCT': 1.00,
        'TS_ACT': 0.60,
        'TS_TRAIL': 0.10,
        'model': 'V3 Ensemble (22-feature RF, 2h/4h/6h)',
        'threshold_long': 0.60,
        'threshold_short': 0.30,
        'features': 22,
        'entry_long': 'Avg prob > 60% (3-model ensemble)',
        'entry_short': 'Avg prob < 30% (3-model ensemble)',
        'max_positions': 2,
        'enabled_long': True,
        'enabled_short': True,
        'db_path': 'v3_predictor_trades.db',
        'position_file': 'btc_v3_positions.json',
        'backtest': '1,612 trades, 42.7% WR, +$4,430 (2yr), PF 1.11',
    },
}


class MonitorDaemon:
    def __init__(self):
        self.status = {
            'last_update': None,
            'last_update_utc': None,
            'bots': {},
            'parquet_health': {},
            'signal_match': {},
            'backtest': {},
            'alerts': [],
            'check_count': 0,
            'config': BOT_CONFIG  # Include bot configuration in status
        }
        self.last_backtest_hour = None
        os.makedirs('status_page', exist_ok=True)
        
    def _watchdog_handler(self, signum, frame):
        """Signal handler for watchdog timeout."""
        raise TimeoutError(f"Check cycle exceeded {CHECK_TIMEOUT_SECONDS}s watchdog")
    
    def run(self):
        """Main loop - runs checks every 30 min with watchdog timeout."""
        logger.info("=" * 60)
        logger.info("MONITOR DAEMON STARTED (with watchdog)")
        logger.info(f"Check interval: {CHECK_INTERVAL_SECONDS}s ({CHECK_INTERVAL_SECONDS/60:.0f}min)")
        logger.info(f"Watchdog timeout: {CHECK_TIMEOUT_SECONDS}s ({CHECK_TIMEOUT_SECONDS/60:.0f}min)")
        logger.info("=" * 60)
        
        # Register watchdog signal handler
        signal.signal(signal.SIGALRM, self._watchdog_handler)
        
        while True:
            try:
                self.status['check_count'] += 1
                # Arm the watchdog - kills hung checks after timeout
                signal.alarm(CHECK_TIMEOUT_SECONDS)
                self.run_checks()
                self.save_status()
                # Disarm the watchdog
                signal.alarm(0)
                logger.info(f"Check #{self.status['check_count']} complete. Next in {CHECK_INTERVAL_SECONDS/60:.0f} min")
            except TimeoutError as e:
                signal.alarm(0)  # Disarm
                logger.error(f"WATCHDOG TIMEOUT: {e}")
                self.status['alerts'].append({
                    'time': datetime.utcnow().isoformat(),
                    'type': 'watchdog_timeout',
                    'message': str(e)
                })
                # Still try to save status so page shows something
                try:
                    self.save_status()
                except Exception:
                    pass
            except Exception as e:
                signal.alarm(0)  # Disarm
                logger.error(f"Error in check cycle: {e}")
                self.status['alerts'].append({
                    'time': datetime.utcnow().isoformat(),
                    'type': 'error',
                    'message': f"Check cycle error: {str(e)}"
                })
            
            # Sleep-resistant wait: check wall clock every 30s
            # This handles macOS sleep/wake where time.sleep(1800) can stall
            next_check = time.time() + CHECK_INTERVAL_SECONDS
            while time.time() < next_check:
                time.sleep(30)
                # If wall clock jumped past next_check (e.g. after wake), break
                if time.time() >= next_check:
                    break
    
    def run_checks(self):
        """Run all monitoring checks."""
        logger.info("-" * 40)
        logger.info("Running monitoring checks...")
        
        self.status['alerts'] = []  # Clear old alerts
        
        # 1. Check parquet health
        self.check_parquet_health()
        
        # 2. Signal match validation
        self.check_signal_match()
        
        # 3. Bot status and probabilities
        self.check_bot_status()
        
        # 4. P&L tracking
        self.calculate_pnl()
        
        # 5. Backtest signal matching (4x daily at 00:00, 06:00, 12:00, 18:00 UTC)
        self.run_backtest_if_scheduled()
        
        # Set last_update AFTER all checks complete so status page shows accurate time
        self.status['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')
        self.status['last_update_utc'] = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        
    def check_parquet_health(self):
        """Check parquet files for completeness and freshness."""
        logger.info("Checking parquet health...")
        
        # BTC Parquet
        btc_health = self._check_single_parquet(BTC_PARQUET, BTC_MIN_BARS, 'BTC')
        self.status['parquet_health']['BTC'] = btc_health
        
        if not btc_health['healthy']:
            logger.warning(f"BTC parquet unhealthy: {btc_health['issues']}")
            self._trigger_btc_refresh()
        
        # QQQ Parquet (MNQ/SPY)
        qqq_health = self._check_single_parquet(QQQ_PARQUET, QQQ_MIN_BARS, 'QQQ')
        self.status['parquet_health']['QQQ'] = qqq_health
        
        if not qqq_health['healthy']:
            logger.warning(f"QQQ parquet unhealthy: {qqq_health['issues']}")
            self._trigger_mnq_refresh()
            
    def _check_single_parquet(self, path: str, min_bars: int, name: str) -> Dict:
        """Check a single parquet file."""
        result = {
            'healthy': True,
            'bars': 0,
            'last_timestamp': None,
            'freshness_minutes': None,
            'issues': []
        }
        
        try:
            if not os.path.exists(path):
                result['healthy'] = False
                result['issues'].append('File not found')
                return result
            
            df = pd.read_parquet(path)
            result['bars'] = len(df)
            result['last_timestamp'] = str(df.index[-1])
            
            # Check bar count
            if len(df) < min_bars:
                result['healthy'] = False
                result['issues'].append(f'Insufficient bars: {len(df)} < {min_bars}')
            
            # Check freshness (should be within 15 minutes for BTC, 1 hour for QQQ on weekends)
            last_ts = pd.Timestamp(df.index[-1])
            now_utc = pd.Timestamp(datetime.utcnow())
            freshness = (now_utc - last_ts).total_seconds() / 60
            result['freshness_minutes'] = round(freshness, 1)
            
            # BTC trades 24/7, should be fresh
            if name == 'BTC' and freshness > 15:
                result['issues'].append(f'Stale data: {freshness:.0f} min old')
                # Don't mark unhealthy for slight staleness, bot will refresh
            
            # Check feature count
            if len(df.columns) < 200:
                result['healthy'] = False
                result['issues'].append(f'Missing features: {len(df.columns)} columns')
                
        except Exception as e:
            result['healthy'] = False
            result['issues'].append(f'Error reading: {str(e)}')
        
        return result
    
    def _trigger_btc_refresh(self):
        """Trigger BTC parquet refresh by fetching fresh data."""
        logger.info("Triggering BTC parquet refresh...")
        try:
            from feature_engineering import (
                add_time_features, add_price_features,
                add_daily_context_features, add_lagged_indicator_features,
                add_indicator_changes
            )
            
            # Fetch 2000 bars from Binance
            url = f"{BINANCE_API}/klines"
            all_klines = []
            end_time = None
            target_bars = 2000
            
            while len(all_klines) < target_bars:
                params = {'symbol': 'BTCUSDT', 'interval': '5m', 'limit': 1000}
                if end_time:
                    params['endTime'] = end_time
                
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                batch = response.json()
                
                if not batch:
                    break
                
                all_klines = batch + all_klines
                end_time = batch[0][0] - 1
                
                if len(batch) < 1000:
                    break
            
            klines = all_klines[-target_bars:] if len(all_klines) > target_bars else all_klines
            
            # Convert to DataFrame
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
            
            # Add features
            df = ta.add_all_ta_features(
                df, open='Open', high='High', low='Low', close='Close', volume='Volume',
                fillna=True
            )
            df = add_time_features(df)
            df = add_price_features(df)
            df = add_daily_context_features(df)
            df = add_lagged_indicator_features(df, lookback_periods=[1, 2, 3, 5, 10, 20, 50])
            df = add_indicator_changes(df)
            
            df = df.ffill().bfill().fillna(0)
            df = df.replace([np.inf, -np.inf], 0)
            
            # Save
            df.to_parquet(BTC_PARQUET)
            logger.info(f"BTC parquet refreshed: {len(df)} bars")
            
            self.status['alerts'].append({
                'time': datetime.utcnow().isoformat(),
                'type': 'info',
                'message': f'BTC parquet refreshed with {len(df)} bars'
            })
            
        except Exception as e:
            logger.error(f"Failed to refresh BTC parquet: {e}")
            self.status['alerts'].append({
                'time': datetime.utcnow().isoformat(),
                'type': 'error',
                'message': f'BTC parquet refresh failed: {str(e)}'
            })
    
    def _trigger_mnq_refresh(self):
        """Trigger MNQ/QQQ parquet refresh by fetching fresh data from Yahoo.
        
        Uses QQQ during US market hours, NQ=F (Nasdaq futures) outside market hours.
        """
        logger.info("Triggering MNQ parquet refresh...")
        try:
            from feature_engineering import (
                add_time_features, add_price_features,
                add_daily_context_features, add_lagged_indicator_features,
                add_indicator_changes
            )
            import pytz
            
            # Check if US market is open to determine data source
            et = pytz.timezone('US/Eastern')
            now_et = datetime.now(et)
            weekday = now_et.weekday()
            hour = now_et.hour
            minute = now_et.minute
            
            is_weekend = weekday >= 5
            is_before_open = hour < 9 or (hour == 9 and minute < 30)
            is_after_close = hour >= 16
            is_us_market_hours = not (is_weekend or is_before_open or is_after_close)
            
            # NQ to QQQ conversion ratio
            MNQ_QQQ_RATIO = 41.2
            
            if is_us_market_hours:
                data = yf.download('QQQ', period='60d', interval='5m', progress=False)
                data_source = 'QQQ'
            else:
                data = yf.download('NQ=F', period='60d', interval='5m', progress=False)
                data_source = 'NQ=F'
            
            if data.empty:
                raise ValueError(f"Could not download {data_source} data from Yahoo")
            
            logger.info(f"Using {data_source} for MNQ parquet refresh")
            
            # Prepare DataFrame
            df = data.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index)
            # Convert to Paris time (same as MNQ bot) then remove tz info
            paris = pytz.timezone('Europe/Paris')
            if df.index.tz is not None:
                df.index = df.index.tz_convert(paris).tz_localize(None)
            else:
                # Assume UTC if no timezone
                df.index = df.index.tz_localize('UTC').tz_convert(paris).tz_localize(None)
            
            # Convert NQ to QQQ scale if using futures
            if data_source == 'NQ=F':
                df['Open'] = df['Open'] / MNQ_QQQ_RATIO
                df['High'] = df['High'] / MNQ_QQQ_RATIO
                df['Low'] = df['Low'] / MNQ_QQQ_RATIO
                df['Close'] = df['Close'] / MNQ_QQQ_RATIO
                df['Volume'] = df['Volume'] * 133
            
            # Add features
            df = ta.add_all_ta_features(
                df, open='Open', high='High', low='Low', close='Close', volume='Volume',
                fillna=True
            )
            df = add_time_features(df)
            df = add_price_features(df)
            df = add_daily_context_features(df)
            df = add_lagged_indicator_features(df, lookback_periods=[1, 2, 3, 5, 10, 20, 50])
            df = add_indicator_changes(df)
            
            df = df.ffill().bfill().fillna(0)
            df = df.replace([np.inf, -np.inf], 0)
            
            # Save
            df.to_parquet(QQQ_PARQUET)
            logger.info(f"MNQ parquet refreshed: {len(df)} bars")
            
            self.status['alerts'].append({
                'time': datetime.utcnow().isoformat(),
                'type': 'info',
                'message': f'MNQ parquet refreshed with {len(df)} bars'
            })
            
        except Exception as e:
            logger.error(f"Failed to refresh MNQ parquet: {e}")
            self.status['alerts'].append({
                'time': datetime.utcnow().isoformat(),
                'type': 'error',
                'message': f'MNQ parquet refresh failed: {str(e)}'
            })
    
    def check_signal_match(self):
        """Compare parquet probabilities with fresh data calculations."""
        logger.info("Checking signal match...")
        
        # BTC signal match
        btc_match = self._check_btc_signal_match()
        self.status['signal_match']['BTC'] = btc_match
        
        # MNQ signal match (uses QQQ data)
        mnq_match = self._check_mnq_signal_match()
        self.status['signal_match']['MNQ'] = mnq_match
        
        # SPY signal match (uses same QQQ parquet as MNQ)
        spy_match = self._check_spy_signal_match()
        self.status['signal_match']['SPY'] = spy_match
        
    def _check_btc_signal_match(self) -> Dict:
        """Check BTC signal match between parquet and fresh data."""
        result = {
            'match': True,
            'differences': {},
            'parquet_probs': {},
            'fresh_probs': {},
            'timestamp': None,
            'indicators': {},
            'indicator_signal': False
        }
        
        try:
            from feature_engineering import (
                add_time_features, add_price_features,
                add_daily_context_features, add_lagged_indicator_features,
                add_indicator_changes
            )
            
            # Load parquet
            parquet_df = pd.read_parquet(BTC_PARQUET)
            result['timestamp'] = str(parquet_df.index[-1])
            
            # Load models
            models = {name: joblib.load(path) for name, path in BTC_MODELS.items()}
            feature_cols = list(models['2h_0.5pct'].feature_names_in_)
            
            # Get parquet probabilities
            X_parquet = parquet_df[feature_cols].tail(1)
            for name, model in models.items():
                prob = model.predict_proba(X_parquet)[0, 1]
                result['parquet_probs'][name] = round(prob * 100, 1)
            
            # Download fresh data - match parquet's bar count for fair comparison
            # Cumulative indicators need same history length to match
            parquet_bar_count = len(parquet_df)
            target_bars = max(parquet_bar_count, 2000)
            
            url = f"{BINANCE_API}/klines"
            all_klines = []
            end_time = None
            
            while len(all_klines) < target_bars:
                params = {'symbol': 'BTCUSDT', 'interval': '5m', 'limit': 1000}
                if end_time:
                    params['endTime'] = end_time
                response = requests.get(url, params=params, timeout=10)
                batch = response.json()
                if not batch:
                    break
                all_klines = batch + all_klines
                end_time = batch[0][0] - 1
                if len(batch) < 1000:
                    break
            
            # Use same number of bars as parquet for fair comparison
            klines = all_klines[-parquet_bar_count:] if len(all_klines) >= parquet_bar_count else all_klines
            
            # Process fresh data
            fresh_df = pd.DataFrame(klines, columns=[
                'open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            fresh_df['timestamp'] = pd.to_datetime(fresh_df['open_time'], unit='ms')
            fresh_df = fresh_df.set_index('timestamp')
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                fresh_df[col] = fresh_df[col].astype(float)
            fresh_df = fresh_df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            fresh_df = ta.add_all_ta_features(
                fresh_df, open='Open', high='High', low='Low', close='Close', volume='Volume',
                fillna=True
            )
            fresh_df = add_time_features(fresh_df)
            fresh_df = add_price_features(fresh_df)
            fresh_df = add_daily_context_features(fresh_df)
            fresh_df = add_lagged_indicator_features(fresh_df, lookback_periods=[1, 2, 3, 5, 10, 20, 50])
            fresh_df = add_indicator_changes(fresh_df)
            fresh_df = fresh_df.ffill().bfill().fillna(0)
            fresh_df = fresh_df.replace([np.inf, -np.inf], 0)
            
            # IMPORTANT: Compare at the SAME timestamp to avoid false mismatches
            parquet_last_ts = parquet_df.index[-1]
            
            if parquet_last_ts in fresh_df.index:
                # Compare at same timestamp (ideal case)
                X_fresh = fresh_df.loc[[parquet_last_ts]][feature_cols]
                result['timestamp'] = str(parquet_last_ts)
            else:
                # Parquet timestamp not in fresh data - use fresh latest
                X_fresh = fresh_df[feature_cols].tail(1)
                result['timestamp'] = str(fresh_df.index[-1])
            
            for name, model in models.items():
                prob = model.predict_proba(X_fresh)[0, 1]
                result['fresh_probs'][name] = round(prob * 100, 1)
                
                # Check difference
                diff = abs(result['parquet_probs'][name] - result['fresh_probs'][name])
                result['differences'][name] = round(diff, 1)
                
                if diff > 5:  # 5% threshold
                    result['match'] = False
                    self.status['alerts'].append({
                        'time': datetime.utcnow().isoformat(),
                        'type': 'warning',
                        'message': f'BTC {name} mismatch: parquet={result["parquet_probs"][name]}% vs fresh={result["fresh_probs"][name]}%'
                    })
            
            # AUTO-FIX: If mismatch detected, refresh the parquet and re-check
            if not result['match']:
                logger.warning("BTC signal mismatch detected - triggering parquet refresh")
                self._trigger_btc_refresh()
                result['auto_fixed'] = True
                self.status['alerts'].append({
                    'time': datetime.utcnow().isoformat(),
                    'type': 'info',
                    'message': 'BTC parquet auto-refreshed to fix mismatch'
                })
                
                # Re-check after refresh to verify fix worked
                logger.info("Re-checking BTC signal match after refresh...")
                parquet_df = pd.read_parquet(BTC_PARQUET)
                X_parquet_new = parquet_df[feature_cols].tail(1)
                
                result['parquet_probs'] = {}
                result['differences'] = {}
                result['match'] = True
                
                for name, model in models.items():
                    parquet_prob = model.predict_proba(X_parquet_new)[0, 1] * 100
                    fresh_prob = result['fresh_probs'][name]
                    diff = abs(parquet_prob - fresh_prob)
                    
                    result['parquet_probs'][name] = round(parquet_prob, 1)
                    result['differences'][name] = round(diff, 1)
                    
                    if diff > 5:
                        result['match'] = False
                
                if result['match']:
                    logger.info("BTC signal match verified after refresh")
                    # Remove the old mismatch alerts since fix was successful
                    self.status['alerts'] = [
                        a for a in self.status['alerts'] 
                        if not (a.get('message', '').startswith('BTC') and 'mismatch' in a.get('message', ''))
                    ]
                else:
                    logger.warning("BTC signal still mismatched after refresh")
                    self.status['alerts'].append({
                        'time': datetime.utcnow().isoformat(),
                        'type': 'warning',
                        'message': 'BTC signal still mismatched after refresh - may need investigation'
                    })
            
            # Extract indicator values for BTC ROC+MACD Trend Strategy (Jan 6, 2026)
            # LONG:  ROC(12) > 0.4% AND MACD_Hist > 0 AND MACD_Hist increasing
            # SHORT: ROC(12) < -0.4% AND MACD_Hist < 0 AND MACD_Hist decreasing
            last_row = parquet_df.iloc[-1]
            prev_row = parquet_df.iloc[-2] if len(parquet_df) >= 2 else last_row
            
            # Calculate ROC(12)
            if len(parquet_df) >= 13:
                close_now = float(last_row.get('Close', 0))
                close_12_ago = float(parquet_df['Close'].iloc[-13])
                roc_12 = (close_now - close_12_ago) / close_12_ago * 100 if close_12_ago > 0 else 0
            else:
                roc_12 = 0
            
            # Get MACD values
            macd = float(last_row.get('trend_macd', 0))
            macd_signal = float(last_row.get('trend_macd_signal', 0))
            macd_hist = macd - macd_signal
            
            macd_prev = float(prev_row.get('trend_macd', 0))
            macd_signal_prev = float(prev_row.get('trend_macd_signal', 0))
            macd_hist_prev = macd_prev - macd_signal_prev
            
            # BB %B for reference
            bb_pct_b = float(last_row.get('volatility_bbp', 0.5))
            
            # Check trend conditions
            roc_threshold = 0.4
            long_condition = (roc_12 > roc_threshold and macd_hist > 0 and macd_hist > macd_hist_prev)
            short_condition = (roc_12 < -roc_threshold and macd_hist < 0 and macd_hist < macd_hist_prev)
            
            result['indicators'] = {
                'bb_pct_b': round(bb_pct_b, 4),
                'macd': round(macd, 2),
                'macd_signal': round(macd_signal, 2),
                'macd_hist': round(macd_hist, 4),
                'macd_hist_prev': round(macd_hist_prev, 4),
                'roc_12': round(roc_12, 2),
                'roc_threshold': roc_threshold,
                'long_condition': bool(long_condition),
                'short_condition': bool(short_condition),
                'macd_hist_increasing': bool(macd_hist > macd_hist_prev),
                'macd_hist_decreasing': bool(macd_hist < macd_hist_prev)
            }
            
            if long_condition:
                result['indicator_signal'] = 'LONG'
            elif short_condition:
                result['indicator_signal'] = 'SHORT'
            else:
                result['indicator_signal'] = False
            
            logger.info(f"BTC ROC+MACD: ROC={roc_12:.2f}%, MACD_Hist={macd_hist:.4f} (prev={macd_hist_prev:.4f}) -> {result['indicator_signal'] or 'No signal'}")
                    
        except Exception as e:
            result['match'] = False
            result['error'] = str(e)
            logger.error(f"BTC signal match check failed: {e}")
        
        return result
    
    def _check_mnq_signal_match(self) -> Dict:
        """Check MNQ signal match."""
        result = {
            'match': True,
            'differences': {},
            'parquet_probs': {},
            'fresh_probs': {},
            'timestamp': None,
            'note': None,
            'indicators': {},
            'indicator_signal': False
        }
        
        try:
            # Load parquet
            if not os.path.exists(QQQ_PARQUET):
                result['match'] = False
                result['error'] = 'QQQ parquet not found'
                return result
            
            parquet_df = pd.read_parquet(QQQ_PARQUET)
            result['timestamp'] = str(parquet_df.index[-1])
            
            # Check if market is open (weekday)
            now = datetime.utcnow()
            if now.weekday() >= 5:  # Weekend
                result['note'] = 'Market closed (weekend)'
                # Still calculate parquet probs
                models = {name: joblib.load(path) for name, path in MNQ_MODELS.items()}
                feature_cols = list(models['2h_0.5pct'].feature_names_in_)
                X_parquet = parquet_df[feature_cols].tail(1)
                for name, model in models.items():
                    prob = model.predict_proba(X_parquet)[0, 1]
                    result['parquet_probs'][name] = round(prob * 100, 1)
                return result
            
            # Load models
            models = {name: joblib.load(path) for name, path in MNQ_MODELS.items()}
            feature_cols = list(models['2h_0.5pct'].feature_names_in_)
            
            # Get parquet probabilities
            X_parquet = parquet_df[feature_cols].tail(1)
            for name, model in models.items():
                prob = model.predict_proba(X_parquet)[0, 1]
                result['parquet_probs'][name] = round(prob * 100, 1)
            
            # Check if US stock market is open to determine data source
            # QQQ only trades during US market hours (Mon-Fri 9:30 AM - 4:00 PM ET)
            # Outside market hours, use NQ=F (Nasdaq futures) as proxy (same as MNQ bot)
            import pytz
            et = pytz.timezone('US/Eastern')
            now_et = datetime.now(et)
            weekday = now_et.weekday()  # 0=Monday, 6=Sunday
            hour = now_et.hour
            minute = now_et.minute
            
            # Check if within US market hours
            is_weekend = weekday >= 5  # Saturday or Sunday
            is_before_open = hour < 9 or (hour == 9 and minute < 30)
            is_after_close = hour >= 16
            is_us_market_hours = not (is_weekend or is_before_open or is_after_close)
            
            # NQ to QQQ conversion ratio (same as MNQ bot)
            MNQ_QQQ_RATIO = 41.2
            
            if is_us_market_hours:
                # Use QQQ during market hours
                data = yf.download('QQQ', period='60d', interval='5m', progress=False)
                data_source = 'QQQ'
            else:
                # Use NQ=F (Nasdaq futures) outside market hours - same as MNQ bot
                data = yf.download('NQ=F', period='60d', interval='5m', progress=False)
                data_source = 'NQ=F'
            
            if data.empty:
                result['note'] = f'Could not download fresh {data_source} data'
                return result
            
            # Prepare data
            qqq = data.copy()
            if isinstance(qqq.columns, pd.MultiIndex):
                qqq.columns = qqq.columns.get_level_values(0)
            
            # Convert NQ to QQQ scale if using futures (same as MNQ bot)
            if data_source == 'NQ=F':
                qqq['Open'] = qqq['Open'] / MNQ_QQQ_RATIO
                qqq['High'] = qqq['High'] / MNQ_QQQ_RATIO
                qqq['Low'] = qqq['Low'] / MNQ_QQQ_RATIO
                qqq['Close'] = qqq['Close'] / MNQ_QQQ_RATIO
                qqq['Volume'] = qqq['Volume'] * 133
                result['note'] = f'Using {data_source} as overnight proxy'
            
            # Process fresh data with full TA features (same as MNQ bot)
            import ta
            from feature_engineering import (
                add_time_features, add_price_features,
                add_daily_context_features, add_lagged_indicator_features,
                add_indicator_changes
            )
            
            # Prepare fresh data - convert to Paris time (same as parquet/MNQ bot)
            fresh_df = qqq.copy()
            if isinstance(fresh_df.columns, pd.MultiIndex):
                fresh_df.columns = fresh_df.columns.get_level_values(0)
            fresh_df.index = pd.to_datetime(fresh_df.index)
            paris = pytz.timezone('Europe/Paris')
            if fresh_df.index.tz is not None:
                fresh_df.index = fresh_df.index.tz_convert(paris).tz_localize(None)
            else:
                fresh_df.index = fresh_df.index.tz_localize('UTC').tz_convert(paris).tz_localize(None)
            
            # Add all TA features (same as MNQ bot)
            fresh_df = ta.add_all_ta_features(
                fresh_df, open='Open', high='High', low='Low',
                close='Close', volume='Volume', fillna=True
            )
            
            # Add custom features
            fresh_df = add_time_features(fresh_df)
            fresh_df = add_price_features(fresh_df)
            fresh_df = add_daily_context_features(fresh_df)
            fresh_df = add_lagged_indicator_features(fresh_df, lookback_periods=[1, 2, 3, 5, 10, 20, 50])
            fresh_df = add_indicator_changes(fresh_df)
            fresh_df = fresh_df.dropna()
            
            # IMPORTANT: Compare at the SAME timestamp to avoid false mismatches
            # Find a common timestamp between parquet and fresh data
            parquet_last_ts = parquet_df.index[-1]
            fresh_last_ts = fresh_df.index[-1]
            
            # Find common timestamps
            common_timestamps = parquet_df.index.intersection(fresh_df.index)
            
            if len(common_timestamps) > 0:
                # Use the most recent common timestamp for comparison
                compare_ts = common_timestamps[-1]
                X_parquet_compare = parquet_df.loc[[compare_ts]][feature_cols]
                X_fresh_compare = fresh_df.loc[[compare_ts]][feature_cols]
                result['timestamp'] = str(compare_ts)
                result['note'] = f'Compared at common timestamp: {compare_ts} (parquet latest: {parquet_last_ts}, fresh latest: {fresh_last_ts})'
                
                # Re-calculate parquet probs at the common timestamp
                result['parquet_probs'] = {}
                for name, model in models.items():
                    prob = model.predict_proba(X_parquet_compare)[0, 1]
                    result['parquet_probs'][name] = round(prob * 100, 1)
                
                for name, model in models.items():
                    prob = model.predict_proba(X_fresh_compare)[0, 1]
                    result['fresh_probs'][name] = round(prob * 100, 1)
                    
                    # Check difference at same timestamp
                    diff = abs(result['parquet_probs'][name] - result['fresh_probs'][name])
                    result['differences'][name] = round(diff, 1)
                    
                    if diff > 5:
                        result['match'] = False
                        self.status['alerts'].append({
                            'time': datetime.utcnow().isoformat(),
                            'type': 'warning',
                            'message': f'MNQ {name} mismatch at {compare_ts}: parquet={result["parquet_probs"][name]}% vs fresh={result["fresh_probs"][name]}%'
                        })
            else:
                # No common timestamps - data sources are completely out of sync
                result['note'] = f'No common timestamps: parquet ends at {parquet_last_ts}, fresh ends at {fresh_last_ts}'
                logger.warning(f"MNQ: No common timestamps between parquet and fresh data")
                
                # Still calculate fresh probs for display, but don't flag mismatch
                X_fresh = fresh_df[feature_cols].tail(1)
                for name, model in models.items():
                    prob = model.predict_proba(X_fresh)[0, 1]
                    result['fresh_probs'][name] = round(prob * 100, 1)
                    result['differences'][name] = 'N/A'
            
            # AUTO-FIX: If mismatch detected, trigger MNQ parquet refresh and re-check
            if not result['match']:
                logger.warning("MNQ signal mismatch detected - triggering parquet refresh")
                self._trigger_mnq_refresh()
                result['auto_fixed'] = True
                self.status['alerts'].append({
                    'time': datetime.utcnow().isoformat(),
                    'type': 'info',
                    'message': 'MNQ parquet auto-refreshed to fix mismatch'
                })
                
                # Re-check after refresh to verify fix worked
                logger.info("Re-checking MNQ signal match after refresh...")
                parquet_df = pd.read_parquet(QQQ_PARQUET)
                
                # IMPORTANT: Compare at the SAME timestamp after refresh
                # Find new common timestamps between refreshed parquet and fresh data
                new_common_timestamps = parquet_df.index.intersection(fresh_df.index)
                
                result['parquet_probs'] = {}
                result['differences'] = {}
                result['match'] = True
                
                if len(new_common_timestamps) > 0:
                    new_compare_ts = new_common_timestamps[-1]
                    X_parquet_new = parquet_df.loc[[new_compare_ts]][feature_cols]
                    X_fresh_new = fresh_df.loc[[new_compare_ts]][feature_cols]
                    
                    for name, model in models.items():
                        parquet_prob = model.predict_proba(X_parquet_new)[0, 1] * 100
                        fresh_prob = model.predict_proba(X_fresh_new)[0, 1] * 100
                        diff = abs(parquet_prob - fresh_prob)
                        
                        result['parquet_probs'][name] = round(parquet_prob, 1)
                        result['fresh_probs'][name] = round(fresh_prob, 1)
                        result['differences'][name] = round(diff, 1)
                        
                        if diff > 5:
                            result['match'] = False
                    
                    result['note'] = f'Re-compared at {new_compare_ts} after refresh'
                else:
                    # Still no common timestamps after refresh
                    result['match'] = True  # Don't flag as mismatch if no common timestamps
                    result['note'] = 'No common timestamps after refresh - cannot compare'
                    logger.warning("No common timestamps between parquet and fresh data after refresh")
                
                if result['match']:
                    logger.info("MNQ signal match verified after refresh")
                    # Remove the old mismatch alerts since fix was successful
                    self.status['alerts'] = [
                        a for a in self.status['alerts'] 
                        if not (a.get('message', '').startswith('MNQ') and 'mismatch' in a.get('message', ''))
                    ]
                else:
                    logger.warning("MNQ signal still mismatched after refresh")
                    self.status['alerts'].append({
                        'time': datetime.utcnow().isoformat(),
                        'type': 'warning',
                        'message': 'MNQ signal still mismatched after refresh - may need investigation'
                    })
            
            # Extract indicator values for MNQ - check ROC+MACD trend potential
            last_row = parquet_df.iloc[-1]
            prev_row = parquet_df.iloc[-2] if len(parquet_df) >= 2 else last_row
            
            bb_pct_b = float(last_row.get('volatility_bbp', 0.5))
            
            # Calculate ROC(12) for MNQ
            if len(parquet_df) >= 13:
                close_now = float(last_row.get('Close', 0))
                close_12_ago = float(parquet_df['Close'].iloc[-13])
                roc_12 = (close_now - close_12_ago) / close_12_ago * 100 if close_12_ago > 0 else 0
            else:
                roc_12 = 0
            
            # Get MACD values
            macd = float(last_row.get('trend_macd', 0))
            macd_signal = float(last_row.get('trend_macd_signal', 0))
            macd_hist = macd - macd_signal
            
            macd_prev = float(prev_row.get('trend_macd', 0))
            macd_signal_prev = float(prev_row.get('trend_macd_signal', 0))
            macd_hist_prev = macd_prev - macd_signal_prev
            
            result['indicators'] = {
                'bb_pct_b': round(bb_pct_b, 4),
                'bb_condition': bool(bb_pct_b < 0.5),
                'roc_12': round(roc_12, 2),
                'macd_hist': round(macd_hist, 4),
                'macd_hist_prev': round(macd_hist_prev, 4),
                'macd_hist_increasing': bool(macd_hist > macd_hist_prev),
                'macd_hist_decreasing': bool(macd_hist < macd_hist_prev)
            }
            result['indicator_signal'] = bool(bb_pct_b < 0.5)
            logger.info(f"MNQ indicators: BB%B={bb_pct_b:.4f}, ROC={roc_12:.2f}%, MACD_Hist={macd_hist:.4f} -> {'LONG' if result['indicator_signal'] else 'No signal'}")
            logger.info(f"MNQ ROC+MACD potential: ROC={roc_12:.2f}%, MACD_Hist={macd_hist:.4f} (increasing={macd_hist > macd_hist_prev})")
            
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"MNQ signal match check failed: {e}")
        
        return result
    
    def _check_spy_signal_match(self) -> Dict:
        """Check SPY indicator signal - RSI<40 + BB<0.4 strategy (Jan 6, 2026).
        
        Uses same QQQ parquet as MNQ since SPY bot uses QQQ features.
        Strategy: RSI < 40 AND BB %B < 0.4 (double confirmation mean reversion)
        Backtest: 97 trades, 51.5% WR, $66/week, 3.5 trades/day
        """
        result = {
            'match': True,
            'indicators': {},
            'indicator_signal': False
        }
        
        try:
            # Load QQQ parquet (same as MNQ)
            parquet_df = pd.read_parquet(QQQ_PARQUET)
            result['timestamp'] = str(parquet_df.index[-1])
            
            # Extract indicator values for SPY RSI+BB strategy
            last_row = parquet_df.iloc[-1]
            
            bb_pct_b = float(last_row.get('volatility_bbp', 0.5))
            rsi = float(last_row.get('momentum_rsi', 50))
            
            # Strategy thresholds
            rsi_threshold = 40
            bb_threshold = 0.4
            
            # Check conditions
            rsi_condition = rsi < rsi_threshold
            bb_condition = bb_pct_b < bb_threshold
            signal_active = rsi_condition and bb_condition
            
            result['indicators'] = {
                'rsi': round(rsi, 2),
                'rsi_threshold': rsi_threshold,
                'rsi_condition': bool(rsi_condition),
                'bb_pct_b': round(bb_pct_b, 4),
                'bb_threshold': bb_threshold,
                'bb_condition': bool(bb_condition)
            }
            result['indicator_signal'] = bool(signal_active)
            
            signal_str = 'LONG' if signal_active else 'No signal'
            logger.info(f"SPY RSI+BB: RSI={rsi:.1f} (<{rsi_threshold}={rsi_condition}), BB%B={bb_pct_b:.4f} (<{bb_threshold}={bb_condition}) -> {signal_str}")
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"SPY signal match check failed: {e}")
        
        return result
    
    def check_bot_status(self):
        """Check bot processes and get current probabilities."""
        logger.info("Checking bot status...")
        
        # Check BTC bot
        self.status['bots']['BTC'] = self._check_single_bot('btc_ensemble_bot.py', 'BTC')
        
        # Check BTC Tick ML bot
        self.status['bots']['BTC_TICK'] = self._check_single_bot('btc_tick_bot.py', 'BTC_TICK')
        
        # Check BTC Tick Reverse bot
        self.status['bots']['BTC_TICK_REVERSE'] = self._check_single_bot('btc_tick_bot_reverse.py', 'BTC_TICK_REVERSE')
        
        # Check MNQ Global Signal bot
        self.status['bots']['MNQ_GLOBAL'] = self._check_mnq_global_bot()
        
        # Check BTC TP/SL Multi-Strategy bot
        self.status['bots']['BTC_TPSL'] = self._check_tpsl_bot()
        
        # Check BTC V3 Predictor bot
        self.status['bots']['BTC_V3'] = self._check_v3_bot()
        
    def _check_single_bot(self, script_name: str, bot_name: str) -> Dict:
        """Check a single bot's status."""
        result = {
            'running': False,
            'pid': None,
            'probabilities': {},
            'positions': 0,
            'max_positions': 1 if bot_name == 'BTC_TREND' else (3 if bot_name == 'SPY' else 4),
            'last_price': None,
            'last_log_time': None
        }
        
        try:
            # Check if process is running
            # Use space prefix to avoid substring matches
            # (e.g. ' ensemble_bot.py' must not match 'btc_ensemble_bot.py')
            pgrep_pattern = f' {script_name}' if script_name == 'ensemble_bot.py' else script_name
            ps_result = subprocess.run(
                ['pgrep', '-f', pgrep_pattern],
                capture_output=True, text=True
            )
            if ps_result.returncode == 0:
                result['running'] = True
                result['pid'] = ps_result.stdout.strip().split('\n')[0]
            
            # Get latest from signal logs
            if bot_name == 'BTC_TICK':
                log_pattern = "signal_logs/btc_tick_signals_*.csv"
            elif bot_name == 'BTC_TICK_REVERSE':
                log_pattern = "signal_logs/btc_tick_reverse_signals_*.csv"
            else:
                log_pattern = f"signal_logs/{bot_name.lower()}_signals_*.csv"
            import glob
            log_files = sorted(glob.glob(log_pattern))
            
            if log_files:
                latest_log = log_files[-1]
                df = pd.read_csv(latest_log)
                if len(df) > 0:
                    last_row = df.iloc[-1]
                    result['last_log_time'] = last_row.get('timestamp', None)
                    
                    # Get probabilities
                    for col in df.columns:
                        if col.startswith('prob_'):
                            model_name = col.replace('prob_', '')
                            result['probabilities'][model_name] = round(last_row[col] * 100, 1)
                    
                    # Get positions
                    if 'active_positions' in df.columns:
                        result['positions'] = int(last_row['active_positions'])
                    
                    # Get price
                    price_col = f'{bot_name.lower()}_price' if bot_name != 'MNQ' else 'qqq_price'
                    if price_col in df.columns:
                        result['last_price'] = round(last_row[price_col], 2)
                    elif 'btc_price' in df.columns:
                        result['last_price'] = round(last_row['btc_price'], 2)
                        
        except Exception as e:
            logger.error(f"Error checking {bot_name} bot: {e}")
            result['error'] = str(e)
        
        return result
    
    def _check_mnq_global_bot(self) -> Dict:
        """Check MNQ Global Signal bot status, position, and today's signal."""
        result = {
            'running': False,
            'pid': None,
            'probabilities': {},
            'positions': 0,
            'max_positions': 1,
            'last_price': None,
            'last_log_time': None,
            'today_signal': None,
            'rf_prob': None,
            'position_detail': None,
        }
        
        try:
            # Check if process is running
            ps_result = subprocess.run(
                ['pgrep', '-f', 'mnq_global_signal_bot.py'],
                capture_output=True, text=True
            )
            if ps_result.returncode == 0:
                result['running'] = True
                result['pid'] = ps_result.stdout.strip().split('\n')[0]
            
            # Check for open position
            position_file = 'mnq_global_position.json'
            if os.path.exists(position_file):
                with open(position_file, 'r') as f:
                    pos = json.load(f)
                result['positions'] = 1
                result['position_detail'] = {
                    'direction': pos.get('direction', ''),
                    'entry_price': pos.get('entry_price', 0),
                    'tp_price': pos.get('tp_price', 0),
                    'sl_price': pos.get('sl_price', 0),
                    'rf_prob': pos.get('rf_prob', 0),
                    'entry_time': pos.get('entry_time', ''),
                    'trade_date': pos.get('trade_date', ''),
                }
                result['rf_prob'] = pos.get('rf_prob', 0)
                result['last_price'] = pos.get('entry_price', 0)
            
            # Get last log line for latest status
            log_file = 'logs/mnq_global_signal.log'
            if os.path.exists(log_file):
                tail_result = subprocess.run(
                    ['tail', '-20', log_file],
                    capture_output=True, text=True
                )
                if tail_result.returncode == 0:
                    lines = tail_result.stdout.strip().split('\n')
                    if lines:
                        result['last_log_time'] = lines[-1][:19] if len(lines[-1]) >= 19 else None
                    
                    # Extract RF probability from log
                    for line in reversed(lines):
                        if 'RF probability:' in line:
                            try:
                                prob_str = line.split('RF probability:')[1].strip().replace('%', '')
                                result['rf_prob'] = float(prob_str) / 100
                            except:
                                pass
                            break
                        if 'SIGNAL: LONG' in line:
                            result['today_signal'] = 'LONG'
                        elif 'SIGNAL: SHORT' in line:
                            result['today_signal'] = 'SHORT'
                        elif 'SIGNAL: SKIP' in line:
                            result['today_signal'] = 'SKIP'
            
            # Set probabilities dict for dashboard display
            if result['rf_prob'] is not None:
                result['probabilities']['RF'] = round(result['rf_prob'] * 100, 1)
            
            # Get recent trades from SQLite DB
            import sqlite3
            db_path = 'mnq_global_trades.db'
            if os.path.exists(db_path):
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT COUNT(*) FROM trades 
                    WHERE datetime(exit_time) > datetime('now', '-7 days')
                """)
                week_trades = cursor.fetchone()[0]
                
                cursor.execute("""
                    SELECT COUNT(*) FROM trades 
                    WHERE datetime(exit_time) > datetime('now', '-7 days')
                    AND pnl_dollars > 0
                """)
                week_wins = cursor.fetchone()[0]
                
                cursor.execute("""
                    SELECT COALESCE(SUM(pnl_dollars), 0) FROM trades 
                    WHERE datetime(exit_time) > datetime('now', '-7 days')
                """)
                week_pnl = cursor.fetchone()[0]
                
                result['week_stats'] = {
                    'trades': week_trades,
                    'wins': week_wins,
                    'losses': week_trades - week_wins,
                    'win_rate': round(week_wins / week_trades * 100, 1) if week_trades > 0 else 0,
                    'pnl': round(week_pnl, 2),
                }
                conn.close()
                
        except Exception as e:
            logger.error(f"Error checking MNQ_GLOBAL bot: {e}")
            result['error'] = str(e)
        
        return result
    
    def _check_tpsl_bot(self) -> Dict:
        """Check BTC TP/SL Multi-Strategy bot status, positions, and weekly stats."""
        result = {
            'running': False,
            'pid': None,
            'probabilities': {},
            'positions': 0,
            'max_positions': 3,
            'last_price': None,
            'last_log_time': None,
            'strategy_positions': [],
        }
        
        try:
            # Check if process is running
            ps_result = subprocess.run(
                ['pgrep', '-f', 'btc_tpsl_bot.py'],
                capture_output=True, text=True
            )
            if ps_result.returncode == 0:
                result['running'] = True
                result['pid'] = ps_result.stdout.strip().split('\n')[0]
            
            # Get latest signal log data
            import glob
            log_files = sorted(glob.glob('signal_logs/btc_tpsl_signals_*.csv'))
            if log_files:
                latest_log = log_files[-1]
                df = pd.read_csv(latest_log)
                if len(df) > 0:
                    last_row = df.iloc[-1]
                    result['last_log_time'] = last_row.get('timestamp', None)
                    if 'btc_price' in df.columns:
                        result['last_price'] = round(last_row['btc_price'], 2)
                    if 'active_positions' in df.columns:
                        result['positions'] = int(last_row['active_positions'])
                    # Show RF prob and vol ratio
                    if 'rf_prob' in df.columns:
                        result['probabilities']['RF'] = round(last_row['rf_prob'] * 100, 1)
                    if 'vol_ratio' in df.columns:
                        result['probabilities']['Vol'] = round(last_row['vol_ratio'], 2)
            
            # Get open positions from DB
            import sqlite3
            db_path = 'tpsl_trades.db'
            if os.path.exists(db_path):
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Open positions
                cursor.execute("SELECT model_id, direction, entry_price, tp_price, sl_price, entry_time FROM open_positions")
                for row in cursor.fetchall():
                    result['strategy_positions'].append({
                        'strategy': row[0],
                        'direction': row[1],
                        'entry_price': row[2],
                        'tp_price': row[3],
                        'sl_price': row[4],
                        'entry_time': row[5],
                    })
                result['positions'] = len(result['strategy_positions'])
                
                # Week stats by strategy
                cursor.execute("""
                    SELECT model_id, COUNT(*) as trades,
                           SUM(CASE WHEN pnl_dollar > 0 THEN 1 ELSE 0 END) as wins,
                           COALESCE(SUM(pnl_dollar), 0) as pnl
                    FROM trades 
                    WHERE datetime(exit_time) > datetime('now', '-7 days')
                    GROUP BY model_id
                """)
                strategy_stats = {}
                total_trades = 0
                total_wins = 0
                total_pnl = 0
                for row in cursor.fetchall():
                    model_id, trades, wins, pnl = row
                    losses = trades - wins
                    wr = round(wins / trades * 100, 1) if trades > 0 else 0
                    strategy_stats[model_id] = {
                        'trades': trades, 'wins': wins, 'losses': losses,
                        'win_rate': wr, 'pnl': round(pnl, 2)
                    }
                    total_trades += trades
                    total_wins += wins
                    total_pnl += pnl
                
                result['week_stats'] = {
                    'trades': total_trades,
                    'wins': total_wins,
                    'losses': total_trades - total_wins,
                    'win_rate': round(total_wins / total_trades * 100, 1) if total_trades > 0 else 0,
                    'pnl': round(total_pnl, 2),
                    'by_strategy': strategy_stats,
                }
                conn.close()
                
        except Exception as e:
            logger.error(f"Error checking BTC_TPSL bot: {e}")
            result['error'] = str(e)
        
        return result
    
    def _check_v3_bot(self) -> Dict:
        """Check BTC V3 Predictor bot status, positions, and weekly stats."""
        result = {
            'running': False,
            'pid': None,
            'probabilities': {},
            'positions': 0,
            'max_positions': 2,
            'last_price': None,
            'last_log_time': None,
            'position_detail': [],
        }
        
        try:
            # Check if process is running
            ps_result = subprocess.run(
                ['pgrep', '-f', 'btc_v3_predictor_bot.py'],
                capture_output=True, text=True
            )
            if ps_result.returncode == 0:
                result['running'] = True
                result['pid'] = ps_result.stdout.strip().split('\n')[0]
            
            # Get latest signal log data
            import glob
            log_files = sorted(glob.glob('signal_logs/btc_v3_signals_*.csv'))
            if log_files:
                latest_log = log_files[-1]
                df = pd.read_csv(latest_log)
                if len(df) > 0:
                    last_row = df.iloc[-1]
                    result['last_log_time'] = last_row.get('timestamp', None)
                    if 'price' in df.columns:
                        result['last_price'] = round(last_row['price'], 2)
                    if 'confidence' in df.columns:
                        result['probabilities']['confidence'] = round(float(last_row['confidence']) * 100, 1)
                    if 'avg_prob' in df.columns:
                        result['probabilities']['avg_prob'] = round(float(last_row['avg_prob']) * 100, 1)
                    for col in ['prob_2h', 'prob_4h', 'prob_6h']:
                        if col in df.columns:
                            result['probabilities'][col] = round(float(last_row[col]) * 100, 1)
                    if 'positions' in df.columns:
                        result['positions'] = int(last_row['positions'])
            
            # Check open positions from JSON
            position_file = 'btc_v3_positions.json'
            if os.path.exists(position_file):
                with open(position_file, 'r') as f:
                    positions = json.load(f)
                if positions:
                    result['positions'] = len(positions)
                    for pos_id, pos in positions.items():
                        result['position_detail'].append({
                            'model_id': pos_id,
                            'direction': pos.get('direction', ''),
                            'entry_price': pos.get('entry_price', 0),
                            'target_price': pos.get('target_price', 0),
                            'stop_price': pos.get('stop_price', 0),
                            'entry_time': pos.get('entry_time', ''),
                            'confidence': pos.get('entry_confidence', 0),
                        })
            
            # Get last log line for latest status
            log_file = 'logs/btc_v3_predictor.log'
            if os.path.exists(log_file):
                tail_result = subprocess.run(
                    ['tail', '-5', log_file],
                    capture_output=True, text=True
                )
                if tail_result.returncode == 0:
                    lines = tail_result.stdout.strip().split('\n')
                    if lines:
                        result['last_log_time'] = lines[-1][:19] if len(lines[-1]) >= 19 else result['last_log_time']
            
            # Get recent trades from shared trades.db (bot_type='BTC_V3')
            import sqlite3
            db_path = 'trades.db'
            if os.path.exists(db_path):
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT COUNT(*) FROM trades 
                    WHERE bot_type = 'BTC_V3'
                    AND datetime(exit_time) > datetime('now', '-7 days')
                """)
                week_trades = cursor.fetchone()[0]
                
                cursor.execute("""
                    SELECT COUNT(*) FROM trades 
                    WHERE bot_type = 'BTC_V3'
                    AND datetime(exit_time) > datetime('now', '-7 days')
                    AND pnl_dollar > 0
                """)
                week_wins = cursor.fetchone()[0]
                
                cursor.execute("""
                    SELECT COALESCE(SUM(pnl_dollar), 0) FROM trades 
                    WHERE bot_type = 'BTC_V3'
                    AND datetime(exit_time) > datetime('now', '-7 days')
                """)
                week_pnl = cursor.fetchone()[0]
                
                result['week_stats'] = {
                    'trades': week_trades,
                    'wins': week_wins,
                    'losses': week_trades - week_wins,
                    'win_rate': round(week_wins / week_trades * 100, 1) if week_trades > 0 else 0,
                    'pnl': round(week_pnl, 2),
                }
                conn.close()
                
        except Exception as e:
            logger.error(f"Error checking BTC_V3 bot: {e}")
            result['error'] = str(e)
        
        return result
    
    def calculate_pnl(self):
        """Calculate P&L for last 24 hours from position logs."""
        logger.info("Calculating P&L...")
        
        for bot_name in ['BTC', 'BTC_TICK', 'BTC_TICK_REVERSE', 'MNQ_GLOBAL', 'BTC_TPSL', 'BTC_V3']:
            if bot_name in self.status['bots']:
                self.status['bots'][bot_name]['pnl_24h'] = self._calculate_bot_pnl(bot_name)
    
    def _calculate_bot_pnl(self, bot_name: str) -> Dict:
        """Calculate P&L for a single bot including unrealized from positions and realized from trade DB."""
        result = {
            'realized': 0,
            'unrealized': 0,
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'open_positions': [],
            'closed_trades': []
        }
        
        try:
            # Get realized P&L from trade database (last 24 hours)
            import sqlite3
            
            # BTC_TICK and BTC_TREND use their own DBs with no bot_type column
            if bot_name == 'BTC_TICK':
                db_path = 'tick_trades.db'
            elif bot_name == 'BTC_TICK_REVERSE':
                db_path = 'tick_trades_reverse.db'
            elif bot_name == 'BTC_TREND':
                db_path = 'trend_trades.db'
            elif bot_name == 'MNQ_GLOBAL':
                db_path = 'mnq_global_trades.db'
            elif bot_name == 'BTC_TPSL':
                db_path = 'tpsl_trades.db'
            else:
                db_path = 'trades.db'
            
            if os.path.exists(db_path):
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                if bot_name == 'MNQ_GLOBAL':
                    # mnq_global_trades.db uses pnl_dollars column
                    cursor.execute("""
                        SELECT direction, entry_price, exit_price, pnl_dollars, exit_reason, 
                               entry_time, exit_time, 'RF' as model_id
                        FROM trades 
                        WHERE datetime(exit_time) > datetime('now', '-24 hours')
                        ORDER BY exit_time DESC
                    """)
                elif bot_name in ('BTC_TICK', 'BTC_TICK_REVERSE', 'BTC_TREND', 'BTC_TPSL'):
                    # tick_trades.db / trend_trades.db / tick_trades_reverse.db / tpsl_trades.db have no bot_type column
                    has_model_id = bot_name in ('BTC_TICK', 'BTC_TICK_REVERSE', 'BTC_TPSL')
                    model_col = 'model_id' if has_model_id else "'trend'"
                    cursor.execute(f"""
                        SELECT direction, entry_price, exit_price, pnl_dollar, exit_reason, 
                               entry_time, exit_time, {model_col} as model_id
                        FROM trades 
                        WHERE datetime(exit_time) > datetime('now', '-24 hours')
                        ORDER BY exit_time DESC
                    """)
                else:
                    # trades.db has bot_type column
                    cursor.execute("""
                        SELECT direction, entry_price, exit_price, pnl_dollar, exit_reason, 
                               entry_time, exit_time, model_id
                        FROM trades 
                        WHERE bot_type = ? 
                        AND datetime(exit_time) > datetime('now', '-24 hours')
                        ORDER BY exit_time DESC
                    """, (bot_name,))
                
                trades = cursor.fetchall()
                conn.close()
                
                for direction, entry_price, exit_price, pnl_dollar, exit_reason, entry_time, exit_time, model_id in trades:
                    pnl_dollar = pnl_dollar or 0
                    result['realized'] += pnl_dollar
                    result['trades'] += 1
                    if pnl_dollar > 0:
                        result['wins'] += 1
                    else:
                        result['losses'] += 1
                    
                    # Add to closed trades list
                    result['closed_trades'].append({
                        'direction': direction,
                        'entry': round(entry_price, 2) if entry_price else 0,
                        'exit': round(exit_price, 2) if exit_price else 0,
                        'pnl': round(pnl_dollar, 2),
                        'reason': exit_reason,
                        'exit_time': str(exit_time).split('.')[0] if exit_time else '',
                        'model': model_id
                    })
            
            # Get unrealized P&L from open positions
            position_files = {
                'BTC': 'btc_positions.json',
                'MNQ': 'ensemble_positions.json',  # MNQ bot uses ensemble_positions.json
                'SPY': 'spy_positions.json',
                'MNQ_GLOBAL': 'mnq_global_position.json',
                'BTC_V3': 'btc_v3_positions.json',
            }
            position_file = position_files.get(bot_name, f"{bot_name.lower()}_positions.json")
            
            if os.path.exists(position_file):
                with open(position_file, 'r') as f:
                    positions_raw = json.load(f)
                
                # Get current prices for unrealized P&L
                current_price = self._get_current_price(bot_name)
                
                # MNQ_GLOBAL uses a flat dict (single position), others use dict-of-dicts
                if bot_name == 'MNQ_GLOBAL':
                    # Single position: {direction, entry_price, ...}
                    positions = {'mnq_global': positions_raw}
                else:
                    positions = positions_raw
                
                # Process positions (dict format)
                for pos_id, pos in positions.items():
                    entry_price = pos.get('entry_price', 0)
                    direction = pos.get('direction', 'LONG')
                    size = pos.get('size', 1)
                    
                    # Calculate unrealized P&L
                    if current_price and entry_price:
                        if direction == 'LONG':
                            unrealized = (current_price - entry_price) * size
                        else:
                            unrealized = (entry_price - current_price) * size
                        
                        # For BTC futures, multiply by 0.1 (contract multiplier)
                        if bot_name in ('BTC', 'BTC_V3'):
                            unrealized *= 0.1
                        # For MNQ, multiply by 2 ($2/point)
                        elif bot_name == 'MNQ_GLOBAL':
                            unrealized *= 2.0
                        
                        result['unrealized'] += unrealized
                        result['open_positions'].append({
                            'id': pos_id,
                            'direction': direction,
                            'entry': round(entry_price, 2),
                            'current': round(current_price, 2) if current_price else None,
                            'pnl': round(unrealized, 2)
                        })
                                
        except Exception as e:
            logger.error(f"Error calculating {bot_name} P&L: {e}")
        
        result['realized'] = round(result['realized'], 2)
        result['unrealized'] = round(result['unrealized'], 2)
        return result
    
    def _get_current_price(self, bot_name: str) -> Optional[float]:
        """Get current price for a bot's instrument."""
        try:
            if bot_name in ('BTC', 'BTC_TICK', 'BTC_TICK_REVERSE', 'BTC_TREND', 'BTC_V3', 'BTC_TPSL'):
                response = requests.get(f"{BINANCE_API}/ticker/price", 
                                       params={'symbol': 'BTCUSDT'}, timeout=5)
                return float(response.json()['price'])
            elif bot_name == 'MNQ_GLOBAL':
                # Get NQ=F price directly (MNQ tracks NQ 1:1)
                try:
                    nq = yf.download('NQ=F', period='1d', interval='5m', progress=False)
                    if not nq.empty:
                        if isinstance(nq.columns, pd.MultiIndex):
                            return float(nq[('Close', 'NQ=F')].iloc[-1])
                        return float(nq['Close'].iloc[-1])
                except:
                    pass
                return None
            elif bot_name == 'MNQ':
                # MNQ futures price is ~41.2x QQQ price
                # Get QQQ price and convert to MNQ scale
                if os.path.exists(QQQ_PARQUET):
                    df = pd.read_parquet(QQQ_PARQUET)
                    qqq_price = float(df['Close'].iloc[-1])
                    return qqq_price * 41.2  # MNQ/QQQ ratio
            elif bot_name == 'SPY':
                # MES futures price is ~10x SPY price
                # Get QQQ price, convert to SPY (~0.9x), then to MES (10x)
                if os.path.exists(QQQ_PARQUET):
                    df = pd.read_parquet(QQQ_PARQUET)
                    qqq_price = float(df['Close'].iloc[-1])
                    return qqq_price * 11.2  # MES/QQQ ratio (~10 * 1.12)
        except Exception as e:
            logger.warning(f"Could not get price for {bot_name}: {e}")
        return None
    
    def run_backtest_if_scheduled(self):
        """Run backtest signal matching 4x daily (00:00, 06:00, 12:00, 18:00 UTC)."""
        current_hour = datetime.utcnow().hour
        backtest_hours = [0, 6, 12, 18]
        
        # Check if we should run backtest this hour
        if current_hour not in backtest_hours:
            return
        
        # Don't run twice in the same hour
        if self.last_backtest_hour == current_hour:
            return
        
        self.last_backtest_hour = current_hour
        logger.info(f"Running scheduled backtest (hour {current_hour} UTC)...")
        
        try:
            from feature_engineering import (
                add_time_features, add_price_features,
                add_daily_context_features, add_lagged_indicator_features,
                add_indicator_changes
            )
            
            # Fetch fresh BTC data
            url = f"{BINANCE_API}/klines"
            all_klines = []
            end_time = None
            target_bars = 3000
            
            while len(all_klines) < target_bars:
                params = {'symbol': 'BTCUSDT', 'interval': '5m', 'limit': 1000}
                if end_time:
                    params['endTime'] = end_time
                response = requests.get(url, params=params, timeout=10)
                batch = response.json()
                if not batch:
                    break
                all_klines = batch + all_klines
                end_time = batch[0][0] - 1
                if len(batch) < 1000:
                    break
            
            klines = all_klines[-target_bars:] if len(all_klines) > target_bars else all_klines
            
            btc_df = pd.DataFrame(klines, columns=[
                'open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            btc_df['timestamp'] = pd.to_datetime(btc_df['open_time'], unit='ms')
            btc_df = btc_df.set_index('timestamp')
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                btc_df[col] = btc_df[col].astype(float)
            btc_df = btc_df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Add features
            btc_df = ta.add_all_ta_features(
                btc_df, open='Open', high='High', low='Low', close='Close', volume='Volume',
                fillna=True
            )
            btc_df = add_time_features(btc_df)
            btc_df = add_price_features(btc_df)
            btc_df = add_daily_context_features(btc_df)
            btc_df = add_lagged_indicator_features(btc_df, lookback_periods=[1, 2, 3, 5, 10, 20, 50])
            btc_df = add_indicator_changes(btc_df)
            btc_df = btc_df.ffill().bfill().fillna(0)
            btc_df = btc_df.replace([np.inf, -np.inf], 0)
            
            # Load models
            btc_models = {}
            for name, path in BTC_MODELS.items():
                if os.path.exists(path):
                    btc_models[name] = joblib.load(path)
            
            if not btc_models:
                logger.warning("No BTC models found for backtest")
                return
            
            feature_cols = list(btc_models[list(btc_models.keys())[0]].feature_names_in_)
            
            # Analyze last 24 hours
            cutoff = datetime.utcnow() - timedelta(hours=24)
            today_df = btc_df[btc_df.index >= cutoff]
            
            signals_triggered = []
            hourly_summary = []
            
            # Get signals that crossed threshold
            for ts in today_df.index:
                row = btc_df.loc[[ts]][feature_cols]
                price = btc_df.loc[ts, 'Close']
                
                for name, model in btc_models.items():
                    prob = model.predict_proba(row)[0, 1] * 100
                    if prob >= 65:
                        signals_triggered.append({
                            'time': ts.strftime('%H:%M UTC'),
                            'price': round(price, 0),
                            'model': name,
                            'prob': round(prob, 1)
                        })
            
            # Hourly max probabilities - group by actual hour timestamps (chronological)
            today_copy = today_df.copy()
            today_copy['hour_ts'] = today_copy.index.floor('h')
            
            # Get unique hours in chronological order, take last 12
            unique_hours = sorted(today_copy['hour_ts'].unique())[-12:]
            
            for hour_ts in unique_hours:
                hour_data = today_copy[today_copy['hour_ts'] == hour_ts]
                
                max_long = 0
                max_short = 0
                
                for ts in hour_data.index:
                    row = btc_df.loc[[ts]][feature_cols]
                    for name, model in btc_models.items():
                        prob = model.predict_proba(row)[0, 1] * 100
                        if 'SHORT' in name:
                            max_short = max(max_short, prob)
                        else:
                            max_long = max(max_long, prob)
                
                price = hour_data['Close'].iloc[-1]
                signal = None
                if max_long >= 65:
                    signal = 'LONG'
                elif max_short >= 65:
                    signal = 'SHORT'
                
                # Format as HH:00 UTC with the actual hour
                hour_label = pd.Timestamp(hour_ts).strftime('%H:%M')
                hourly_summary.append({
                    'hour': hour_label,
                    'price': round(price, 0),
                    'long': round(max_long, 0),
                    'short': round(max_short, 0),
                    'signal': signal
                })
            
            # Count signals
            long_signals = len([s for s in signals_triggered if 'SHORT' not in s['model']])
            short_signals = len([s for s in signals_triggered if 'SHORT' in s['model']])
            
            # Compare with actual bot trades from database
            import sqlite3
            actual_trades = []
            
            db_path = 'trades.db'
            if os.path.exists(db_path):
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT direction, entry_price, entry_time, model_id, pnl_dollar, exit_reason
                    FROM trades 
                    WHERE bot_type = 'BTC' 
                    AND datetime(entry_time) > datetime('now', '-24 hours')
                    ORDER BY entry_time DESC
                """)
                trades = cursor.fetchall()
                conn.close()
                
                for direction, entry_price, entry_time, model_id, pnl, exit_reason in trades:
                    # Handle both datetime formats
                    if 'T' in str(entry_time):
                        entry_hour = str(entry_time).split('T')[1][:5]
                    else:
                        entry_hour = str(entry_time).split(' ')[1][:5] if ' ' in str(entry_time) else ''
                    pnl_val = pnl if pnl else 0
                    pnl_str = f"+${pnl_val:.0f}" if pnl_val > 0 else f"-${abs(pnl_val):.0f}" if pnl_val < 0 else ''
                    result = '✅' if pnl_val > 0 else '❌' if pnl_val < 0 else '⏳'
                    actual_trades.append({
                        'time': entry_hour + ' UTC',
                        'direction': direction,
                        'price': round(entry_price, 0) if entry_price else 0,
                        'model': model_id,
                        'pnl': pnl_str,
                        'result': result
                    })
            
            # Count unique hours with signals (not individual bars)
            signal_hours_long = set()
            signal_hours_short = set()
            for sig in signals_triggered:
                sig_hour = sig['time'][:2]
                if 'SHORT' in sig['model']:
                    signal_hours_short.add(sig_hour)
                else:
                    signal_hours_long.add(sig_hour)
            
            # Count how many signal hours had actual trades
            trade_hours_long = set()
            trade_hours_short = set()
            for t in actual_trades:
                t_hour = t['time'][:2]
                if t['direction'] == 'LONG':
                    trade_hours_long.add(t_hour)
                else:
                    trade_hours_short.add(t_hour)
            
            # Matched = hours where both backtest signal AND actual trade occurred
            matched_long = len(signal_hours_long & trade_hours_long)
            matched_short = len(signal_hours_short & trade_hours_short)
            matched_hours = matched_long + matched_short
            
            # Signal hours = unique hours with >=65% signal
            total_signal_hours = len(signal_hours_long) + len(signal_hours_short)
            
            # Missed = signal hours without trades
            missed_hours = total_signal_hours - matched_hours
            
            self.status['backtest'] = {
                'last_run': datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC'),
                'period': '24h',
                'total_signals': len(signals_triggered),
                'long_signals': long_signals,
                'short_signals': short_signals,
                'signal_hours': total_signal_hours,
                'recent_signals': signals_triggered[-10:] if signals_triggered else [],
                'hourly_summary': hourly_summary[-12:],
                'actual_trades': actual_trades[:5],
                'matched_hours': matched_hours,
                'missed_hours': missed_hours,
                'trade_count': len(actual_trades)
            }
            
            logger.info(f"Backtest complete: {len(signals_triggered)} signals in {total_signal_hours} hours, {len(actual_trades)} trades, {matched_hours} matched hours")
            
        except Exception as e:
            logger.error(f"Backtest error: {e}")
            self.status['backtest']['error'] = str(e)
    
    def save_status(self):
        """Save status to JSON file and deploy to GitHub."""
        try:
            with open(STATUS_FILE, 'w') as f:
                json.dump(self.status, f, indent=2, default=str)
            logger.info(f"Status saved to {STATUS_FILE}")
            
            # Deploy to GitHub Pages
            self.deploy_to_github()
        except Exception as e:
            logger.error(f"Error saving status: {e}")
    
    def deploy_to_github(self):
        """Deploy status page to GitHub Pages."""
        github_repo_path = '/Users/smalandrakis/Documents/WindSurf/trading-status-page'
        try:
            # Copy status.json and index.html to GitHub repo
            import shutil
            shutil.copy('status_page/status.json', f'{github_repo_path}/status.json')
            shutil.copy('status_page/index.html', f'{github_repo_path}/index.html')
            
            # Git add, commit, push
            result = subprocess.run(
                ['git', 'add', 'status.json', 'index.html'],
                capture_output=True, text=True, timeout=30,
                cwd=github_repo_path
            )
            
            result = subprocess.run(
                ['git', 'commit', '-m', f'Update status {datetime.now().strftime("%Y-%m-%d %H:%M")}'],
                capture_output=True, text=True, timeout=30,
                cwd=github_repo_path
            )
            
            if 'nothing to commit' in result.stdout or 'nothing to commit' in result.stderr:
                logger.info("GitHub: No changes to deploy")
                return
            
            result = subprocess.run(
                ['git', 'push'],
                capture_output=True, text=True, timeout=60,
                cwd=github_repo_path
            )
            
            if result.returncode == 0:
                logger.info("Deployed to GitHub Pages successfully")
            else:
                logger.warning(f"GitHub push failed: {result.stderr[:200]}")
        except Exception as e:
            logger.warning(f"GitHub deploy error: {e}")


def main():
    daemon = MonitorDaemon()
    
    # Check for --once flag
    if len(sys.argv) > 1 and sys.argv[1] == '--once':
        logger.info("Running single check...")
        daemon.run_checks()
        daemon.save_status()
        print(json.dumps(daemon.status, indent=2, default=str))
    else:
        daemon.run()


if __name__ == '__main__':
    main()
