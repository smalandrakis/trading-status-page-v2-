#!/usr/bin/env python3
"""
Parquet Validator - Comprehensive validation for parquet feature files.

This module ensures parquet data integrity for accurate model predictions.
It validates OHLCV data, all 206+ features, and data consistency.

Usage:
    from parquet_validator import ParquetValidator
    
    validator = ParquetValidator(bot_type='btc')  # or 'mnq'
    is_valid, issues = validator.validate_before_save(df)
    if not is_valid:
        df = validator.repair(df, issues)
"""

import pandas as pd
import numpy as np
import logging
import requests
import yfinance as yf
import pytz
import os
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Optional, Any
import joblib

logger = logging.getLogger(__name__)

# Dedicated log file for validation issues
VALIDATION_LOG_FILE = 'logs/parquet_validation.log'
os.makedirs('logs', exist_ok=True)


def log_validation_event(bot_type: str, event_type: str, message: str, details: Dict = None):
    """Log validation events to dedicated file with timestamp."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_line = f"[{timestamp}] [{bot_type.upper()}] [{event_type}] {message}"
    if details:
        log_line += f" | {details}"
    
    # Log to file
    with open(VALIDATION_LOG_FILE, 'a') as f:
        f.write(log_line + '\n')
    
    # Also log to standard logger
    if event_type == 'ERROR':
        logger.error(log_line)
    elif event_type == 'WARNING':
        logger.warning(log_line)
    else:
        logger.info(log_line)


class ParquetValidator:
    """Comprehensive parquet validation for trading bots."""
    
    # Expected value ranges for BTC
    BTC_RANGES = {
        'Open': (10000, 500000),
        'High': (10000, 500000),
        'Low': (10000, 500000),
        'Close': (10000, 500000),
        'Volume': (0, 100000),
        'momentum_rsi': (0, 100),
        'momentum_stoch': (0, 100),
        'momentum_stoch_signal': (0, 100),
        'volatility_bbp': (-5, 5),
        'trend_adx': (0, 100),
    }
    
    # Expected value ranges for MNQ (QQQ scale)
    MNQ_RANGES = {
        'Open': (100, 1000),
        'High': (100, 1000),
        'Low': (100, 1000),
        'Close': (100, 1000),
        'Volume': (0, 500000000),
        'momentum_rsi': (0, 100),
        'momentum_stoch': (0, 100),
        'momentum_stoch_signal': (0, 100),
        'volatility_bbp': (-5, 5),
        'trend_adx': (0, 100),
    }
    
    # Required minimum rows for accurate feature calculation
    MIN_ROWS = 500
    RECOMMENDED_ROWS = 1000
    
    # Maximum allowed staleness (in minutes)
    # During active trading: should be within 1-2 bars (5-10 min)
    # The validator will check market hours and adjust accordingly
    MAX_STALENESS_BTC = 10  # BTC trades 24/7, should always be fresh
    MAX_STALENESS_MNQ_ACTIVE = 10  # During market hours (9:30-16:00 ET)
    MAX_STALENESS_MNQ_HALT = 70  # During daily halt (17:00-18:00 ET) - 1 hour + buffer
    MAX_STALENESS_MNQ_WEEKEND = 4000  # Weekend (Fri 17:00 - Sun 18:00 ET)
    
    def __init__(self, bot_type: str = 'btc', model_dir: str = None):
        """
        Initialize validator.
        
        Args:
            bot_type: 'btc' or 'mnq'
            model_dir: Path to model directory (to get required features)
        """
        self.bot_type = bot_type.lower()
        self.ranges = self.BTC_RANGES if self.bot_type == 'btc' else self.MNQ_RANGES
        self.max_staleness = self._get_max_staleness()
        
        # Load model to get required features
        if model_dir is None:
            model_dir = f'models_{self.bot_type}_v2'
        
        try:
            model = joblib.load(f'{model_dir}/model_2h_0.5pct.joblib')
            self.required_features = list(model.feature_names_in_)
            logger.info(f"Loaded {len(self.required_features)} required features from model")
        except Exception as e:
            logger.warning(f"Could not load model features: {e}")
            self.required_features = []
    
    def _get_max_staleness(self) -> int:
        """Get max staleness based on bot type and current market hours."""
        if self.bot_type == 'btc':
            return self.MAX_STALENESS_BTC  # 10 min, BTC trades 24/7
        
        # MNQ - check market hours and holidays
        et = pytz.timezone('US/Eastern')
        now_et = datetime.now(et)
        hour = now_et.hour
        weekday = now_et.weekday()  # 0=Monday, 6=Sunday
        
        # Check for CME holidays (Christmas, Thanksgiving, etc.)
        # CME closes for major US holidays - allow extended staleness
        month_day = (now_et.month, now_et.day)
        cme_holidays = [
            (12, 24), (12, 25),  # Christmas Eve/Day
            (1, 1),              # New Year's Day
            (7, 4),              # Independence Day
            (11, 28), (11, 29),  # Thanksgiving (approximate - 4th Thursday)
        ]
        if month_day in cme_holidays:
            log_validation_event(self.bot_type, 'INFO', f'CME holiday detected ({month_day[0]}/{month_day[1]}) - extended staleness allowed')
            return 4000  # ~3 days for holiday closures
        
        # Weekend: Friday 17:00 to Sunday 18:00
        if weekday == 5 or weekday == 6 or (weekday == 4 and hour >= 17):
            return self.MAX_STALENESS_MNQ_WEEKEND
        
        # Daily halt: 17:00-18:00 ET
        if hour == 17:
            return self.MAX_STALENESS_MNQ_HALT
        
        # Active trading hours (including overnight futures)
        return self.MAX_STALENESS_MNQ_ACTIVE  # 10 min
    
    def validate_before_save(self, df: pd.DataFrame) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Comprehensive validation before saving parquet.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # 1. Basic structure validation
        issues.extend(self._validate_structure(df))
        
        # 2. OHLCV validation
        issues.extend(self._validate_ohlcv(df))
        
        # 3. Feature completeness validation
        issues.extend(self._validate_features(df))
        
        # 4. Data quality validation (NaN, Inf, outliers)
        issues.extend(self._validate_data_quality(df))
        
        # 5. Timestamp validation
        issues.extend(self._validate_timestamps(df))
        
        # 6. Cross-validation with fresh data source
        issues.extend(self._validate_against_source(df))
        
        # 7. Feature value range validation
        issues.extend(self._validate_feature_ranges(df))
        
        # Categorize issues
        critical = [i for i in issues if i['severity'] == 'critical']
        warnings = [i for i in issues if i['severity'] == 'warning']
        
        is_valid = len(critical) == 0
        
        # Log all issues to dedicated validation log file
        if issues:
            log_validation_event(
                self.bot_type, 'SUMMARY',
                f"Validation found {len(critical)} critical, {len(warnings)} warnings",
                {'rows': len(df), 'columns': len(df.columns), 'latest': str(df.index[-1]) if len(df) > 0 else 'N/A'}
            )
            for issue in critical:
                log_validation_event(
                    self.bot_type, 'ERROR',
                    issue['message'],
                    issue.get('details')
                )
            for issue in warnings:
                log_validation_event(
                    self.bot_type, 'WARNING',
                    issue['message'],
                    issue.get('details')
                )
        else:
            log_validation_event(
                self.bot_type, 'OK',
                f"Validation passed - {len(df)} rows, {len(df.columns)} features",
                {'latest': str(df.index[-1]) if len(df) > 0 else 'N/A'}
            )
        
        return is_valid, issues
    
    def _validate_structure(self, df: pd.DataFrame) -> List[Dict]:
        """Validate basic DataFrame structure."""
        issues = []
        
        # Check if DataFrame is empty
        if df.empty:
            issues.append({
                'type': 'structure',
                'severity': 'critical',
                'message': 'DataFrame is empty'
            })
            return issues
        
        # Check minimum rows
        if len(df) < self.MIN_ROWS:
            issues.append({
                'type': 'structure',
                'severity': 'critical',
                'message': f'Insufficient rows: {len(df)} < {self.MIN_ROWS} minimum'
            })
        elif len(df) < self.RECOMMENDED_ROWS:
            issues.append({
                'type': 'structure',
                'severity': 'warning',
                'message': f'Below recommended rows: {len(df)} < {self.RECOMMENDED_ROWS}'
            })
        
        # Check index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            issues.append({
                'type': 'structure',
                'severity': 'critical',
                'message': f'Index is not DatetimeIndex: {type(df.index)}'
            })
        
        # Check index is sorted
        if not df.index.is_monotonic_increasing:
            issues.append({
                'type': 'structure',
                'severity': 'warning',
                'message': 'Index is not sorted in ascending order'
            })
        
        return issues
    
    def _validate_ohlcv(self, df: pd.DataFrame) -> List[Dict]:
        """Validate OHLCV columns."""
        issues = []
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Check columns exist
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            issues.append({
                'type': 'ohlcv',
                'severity': 'critical',
                'message': f'Missing OHLCV columns: {missing}'
            })
            return issues
        
        # Check OHLCV logic (High >= Low, High >= Open/Close, Low <= Open/Close)
        invalid_bars = df[(df['High'] < df['Low']) | 
                         (df['High'] < df['Open']) | 
                         (df['High'] < df['Close']) |
                         (df['Low'] > df['Open']) | 
                         (df['Low'] > df['Close'])]
        
        if len(invalid_bars) > 0:
            issues.append({
                'type': 'ohlcv',
                'severity': 'critical',
                'message': f'{len(invalid_bars)} bars have invalid OHLCV logic (High < Low, etc.)',
                'details': invalid_bars.index.tolist()[:5]
            })
        
        # Check for zero or negative prices
        for col in ['Open', 'High', 'Low', 'Close']:
            invalid = df[df[col] <= 0]
            if len(invalid) > 0:
                issues.append({
                    'type': 'ohlcv',
                    'severity': 'critical',
                    'message': f'{len(invalid)} bars have {col} <= 0'
                })
        
        # Check for zero volume (warning only - can happen)
        zero_vol = df[df['Volume'] == 0]
        if len(zero_vol) > len(df) * 0.1:  # More than 10%
            issues.append({
                'type': 'ohlcv',
                'severity': 'warning',
                'message': f'{len(zero_vol)} bars ({len(zero_vol)/len(df)*100:.1f}%) have zero volume'
            })
        
        # Check price is in expected range
        for col in ['Open', 'High', 'Low', 'Close']:
            min_val, max_val = self.ranges[col]
            out_of_range = df[(df[col] < min_val) | (df[col] > max_val)]
            if len(out_of_range) > 0:
                issues.append({
                    'type': 'ohlcv',
                    'severity': 'critical',
                    'message': f'{len(out_of_range)} bars have {col} outside expected range [{min_val}, {max_val}]',
                    'details': {
                        'min_found': df[col].min(),
                        'max_found': df[col].max()
                    }
                })
        
        return issues
    
    def _validate_features(self, df: pd.DataFrame) -> List[Dict]:
        """Validate all required features are present."""
        issues = []
        
        if not self.required_features:
            issues.append({
                'type': 'features',
                'severity': 'warning',
                'message': 'Could not validate features - model not loaded'
            })
            return issues
        
        # Check all required features exist
        missing = [f for f in self.required_features if f not in df.columns]
        if missing:
            issues.append({
                'type': 'features',
                'severity': 'critical',
                'message': f'{len(missing)} required features missing',
                'details': missing[:10]  # First 10
            })
        
        # Check feature count
        if len(df.columns) < 200:
            issues.append({
                'type': 'features',
                'severity': 'warning',
                'message': f'Low feature count: {len(df.columns)} (expected 200+)'
            })
        
        return issues
    
    def _validate_data_quality(self, df: pd.DataFrame) -> List[Dict]:
        """Validate data quality - NaN, Inf, outliers."""
        issues = []
        
        # Check for NaN values
        nan_counts = df.isna().sum()
        nan_cols = nan_counts[nan_counts > 0]
        if len(nan_cols) > 0:
            total_nan = nan_cols.sum()
            issues.append({
                'type': 'quality',
                'severity': 'warning' if total_nan < len(df) * 0.01 else 'critical',
                'message': f'{total_nan} NaN values across {len(nan_cols)} columns',
                'details': nan_cols.head(5).to_dict()
            })
        
        # Check for Inf values
        inf_mask = np.isinf(df.select_dtypes(include=[np.number]))
        inf_counts = inf_mask.sum()
        inf_cols = inf_counts[inf_counts > 0]
        if len(inf_cols) > 0:
            total_inf = inf_cols.sum()
            issues.append({
                'type': 'quality',
                'severity': 'critical',
                'message': f'{total_inf} Inf values across {len(inf_cols)} columns',
                'details': inf_cols.head(5).to_dict()
            })
        
        # Check for extreme outliers (> 10 std from mean)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:50]:  # Check first 50 numeric columns
            if col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                continue  # Skip OHLCV - checked separately
            
            std = df[col].std()
            mean = df[col].mean()
            if std > 0:
                outliers = df[np.abs(df[col] - mean) > 10 * std]
                if len(outliers) > len(df) * 0.01:  # More than 1%
                    issues.append({
                        'type': 'quality',
                        'severity': 'warning',
                        'message': f'{col}: {len(outliers)} extreme outliers (>10 std)',
                        'details': {'mean': mean, 'std': std}
                    })
        
        return issues
    
    def _validate_timestamps(self, df: pd.DataFrame) -> List[Dict]:
        """Validate timestamp consistency and freshness."""
        issues = []
        
        if not isinstance(df.index, pd.DatetimeIndex):
            return issues  # Already reported in structure validation
        
        # Check for duplicate timestamps
        duplicates = df.index.duplicated()
        if duplicates.any():
            issues.append({
                'type': 'timestamp',
                'severity': 'critical',
                'message': f'{duplicates.sum()} duplicate timestamps found'
            })
        
        # Check for gaps (missing 5-min bars)
        expected_freq = pd.Timedelta(minutes=5)
        time_diffs = df.index.to_series().diff()
        gaps = time_diffs[time_diffs > expected_freq * 2]  # Allow some tolerance
        if len(gaps) > len(df) * 0.05:  # More than 5% gaps
            issues.append({
                'type': 'timestamp',
                'severity': 'warning',
                'message': f'{len(gaps)} timestamp gaps detected (>{expected_freq*2})'
            })
        
        # Check freshness
        if self.bot_type == 'btc':
            now_utc = datetime.utcnow()
            latest = df.index[-1].to_pydatetime()
            if latest.tzinfo:
                latest = latest.replace(tzinfo=None)
            staleness = (now_utc - latest).total_seconds() / 60
        else:
            # MNQ uses Paris time
            paris = pytz.timezone('Europe/Paris')
            now_paris = datetime.now(paris).replace(tzinfo=None)
            latest = df.index[-1].to_pydatetime()
            if latest.tzinfo:
                latest = latest.replace(tzinfo=None)
            staleness = (now_paris - latest).total_seconds() / 60
        
        if staleness > self.max_staleness:
            issues.append({
                'type': 'timestamp',
                'severity': 'warning',
                'message': f'Data is {staleness:.1f} min stale (max: {self.max_staleness})'
            })
        
        return issues
    
    def _validate_against_source(self, df: pd.DataFrame) -> List[Dict]:
        """Cross-validate OHLCV against fresh data source."""
        issues = []
        
        try:
            # Get fresh data from source
            if self.bot_type == 'btc':
                fresh_df = self._get_fresh_btc_data()
            else:
                fresh_df = self._get_fresh_mnq_data()
            
            if fresh_df is None or fresh_df.empty:
                issues.append({
                    'type': 'source',
                    'severity': 'warning',
                    'message': 'Could not fetch fresh data for cross-validation'
                })
                return issues
            
            # Find common timestamps (completed bars only - at least 10 min old)
            now = datetime.utcnow() if self.bot_type == 'btc' else datetime.now()
            cutoff = now - timedelta(minutes=10)
            
            common_times = df.index.intersection(fresh_df.index)
            common_times = [t for t in common_times if t.to_pydatetime().replace(tzinfo=None) < cutoff]
            
            if len(common_times) < 5:
                issues.append({
                    'type': 'source',
                    'severity': 'warning',
                    'message': f'Only {len(common_times)} common timestamps for cross-validation'
                })
                return issues
            
            # Check OHLCV match for recent completed bars
            mismatches = []
            for ts in common_times[-10:]:  # Check last 10 common bars
                for col in ['Open', 'High', 'Low', 'Close']:
                    df_val = df.loc[ts, col]
                    fresh_val = fresh_df.loc[ts, col]
                    
                    if df_val != 0:
                        pct_diff = abs(df_val - fresh_val) / df_val * 100
                        if pct_diff > 0.1:  # More than 0.1% difference
                            mismatches.append({
                                'timestamp': str(ts),
                                'column': col,
                                'parquet': df_val,
                                'source': fresh_val,
                                'diff_pct': pct_diff
                            })
            
            if mismatches:
                issues.append({
                    'type': 'source',
                    'severity': 'critical' if len(mismatches) > 5 else 'warning',
                    'message': f'{len(mismatches)} OHLCV mismatches with source data',
                    'details': mismatches[:5]
                })
        
        except Exception as e:
            issues.append({
                'type': 'source',
                'severity': 'warning',
                'message': f'Cross-validation failed: {e}'
            })
        
        return issues
    
    def _validate_feature_ranges(self, df: pd.DataFrame) -> List[Dict]:
        """Validate feature values are in expected ranges."""
        issues = []
        
        # Check bounded features
        bounded_features = {
            'momentum_rsi': (0, 100),
            'momentum_stoch': (0, 100),
            'momentum_stoch_signal': (0, 100),
            'momentum_wr': (-100, 0),
            'trend_adx': (0, 100),
            'volatility_bbp': (-10, 10),
        }
        
        for feat, (min_val, max_val) in bounded_features.items():
            if feat in df.columns:
                out_of_range = df[(df[feat] < min_val) | (df[feat] > max_val)]
                if len(out_of_range) > 0:
                    issues.append({
                        'type': 'feature_range',
                        'severity': 'warning',
                        'message': f'{feat}: {len(out_of_range)} values outside [{min_val}, {max_val}]',
                        'details': {
                            'min_found': df[feat].min(),
                            'max_found': df[feat].max()
                        }
                    })
        
        return issues
    
    def _get_fresh_btc_data(self) -> Optional[pd.DataFrame]:
        """Fetch fresh BTC data from Binance."""
        try:
            url = 'https://api.binance.com/api/v3/klines'
            params = {'symbol': 'BTCUSDT', 'interval': '5m', 'limit': 100}
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
        except Exception as e:
            logger.warning(f"Failed to fetch fresh BTC data: {e}")
            return None
    
    def _get_fresh_mnq_data(self) -> Optional[pd.DataFrame]:
        """Fetch fresh MNQ/QQQ data from Yahoo."""
        try:
            paris = pytz.timezone('Europe/Paris')
            et = pytz.timezone('US/Eastern')
            now_et = datetime.now(et)
            market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
            is_market_hours = market_open <= now_et <= market_close and now_et.weekday() < 5
            
            MNQ_QQQ_RATIO = 41.2
            
            if is_market_hours:
                data = yf.download('QQQ', period='1d', interval='5m', progress=False)
            else:
                data = yf.download('NQ=F', period='1d', interval='5m', progress=False)
            
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] for col in data.columns]
            
            if not is_market_hours and len(data) > 0:
                data['Open'] = data['Open'] / MNQ_QQQ_RATIO
                data['High'] = data['High'] / MNQ_QQQ_RATIO
                data['Low'] = data['Low'] / MNQ_QQQ_RATIO
                data['Close'] = data['Close'] / MNQ_QQQ_RATIO
                data['Volume'] = data['Volume'] * 133
            
            if data.index.tz is not None:
                data.index = data.index.tz_convert(paris).tz_localize(None)
            
            return data[['Open', 'High', 'Low', 'Close', 'Volume']]
        except Exception as e:
            logger.warning(f"Failed to fetch fresh MNQ data: {e}")
            return None
    
    def repair(self, df: pd.DataFrame, issues: List[Dict]) -> pd.DataFrame:
        """
        Attempt to repair issues found during validation.
        
        Args:
            df: DataFrame to repair
            issues: List of issues from validate_before_save
            
        Returns:
            Repaired DataFrame
        """
        df = df.copy()
        repairs_made = []
        
        for issue in issues:
            issue_type = issue['type']
            
            if issue_type == 'quality':
                # Fix NaN values
                if 'NaN' in issue['message']:
                    nan_before = df.isna().sum().sum()
                    df = df.ffill().bfill().fillna(0)
                    repairs_made.append(f"Filled {nan_before} NaN values")
                
                # Fix Inf values
                if 'Inf' in issue['message']:
                    inf_before = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
                    df = df.replace([np.inf, -np.inf], 0)
                    repairs_made.append(f"Replaced {inf_before} Inf values")
            
            elif issue_type == 'structure':
                # Sort index if not sorted
                if 'not sorted' in issue['message']:
                    df = df.sort_index()
                    repairs_made.append("Sorted index")
            
            elif issue_type == 'timestamp':
                # Remove duplicates
                if 'duplicate' in issue['message']:
                    dups_before = df.index.duplicated().sum()
                    df = df[~df.index.duplicated(keep='last')]
                    repairs_made.append(f"Removed {dups_before} duplicate timestamps")
        
        # Log all repairs made
        if repairs_made:
            log_validation_event(
                self.bot_type, 'REPAIR',
                f"Repairs applied: {len(repairs_made)}",
                {'repairs': repairs_made}
            )
        
        return df
    
    def validate_and_save(self, df: pd.DataFrame, path: str) -> bool:
        """
        Validate DataFrame and save to parquet if valid.
        
        Args:
            df: DataFrame to validate and save
            path: Path to save parquet file
            
        Returns:
            True if saved successfully, False otherwise
        """
        # Validate
        is_valid, issues = self.validate_before_save(df)
        
        # Attempt repair if needed
        if not is_valid:
            logger.info("Attempting to repair issues...")
            df = self.repair(df, issues)
            
            # Re-validate after repair
            is_valid, issues = self.validate_before_save(df)
        
        if is_valid:
            df.to_parquet(path)
            logger.info(f"Saved validated parquet to {path}")
            return True
        else:
            critical = [i for i in issues if i['severity'] == 'critical']
            logger.error(f"Cannot save parquet - {len(critical)} critical issues remain")
            return False


def validate_parquet_file(path: str, bot_type: str = 'btc') -> Tuple[bool, List[Dict]]:
    """
    Convenience function to validate an existing parquet file.
    
    Args:
        path: Path to parquet file
        bot_type: 'btc' or 'mnq'
        
    Returns:
        Tuple of (is_valid, issues)
    """
    df = pd.read_parquet(path)
    validator = ParquetValidator(bot_type=bot_type)
    return validator.validate_before_save(df)


if __name__ == '__main__':
    import sys
    
    # Test validation on existing parquets
    print("="*70)
    print("PARQUET VALIDATION TEST")
    print("="*70)
    
    for bot_type, path in [('btc', 'data/BTC_features.parquet'), ('mnq', 'data/QQQ_features.parquet')]:
        print(f"\n### {bot_type.upper()} Parquet ###")
        try:
            is_valid, issues = validate_parquet_file(path, bot_type)
            print(f"Valid: {is_valid}")
            print(f"Issues: {len(issues)}")
            for issue in issues:
                severity = issue['severity'].upper()
                print(f"  [{severity}] {issue['type']}: {issue['message']}")
        except Exception as e:
            print(f"Error: {e}")
