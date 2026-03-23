"""
Data Validation Module for Trading Bots
Validates data quality, feature completeness, and signal validity.

Usage:
    from data_validator import DataValidator
    validator = DataValidator(model_dir='models_mnq_v2')
    is_valid, issues = validator.validate_features(df)
"""

import pandas as pd
import numpy as np
import json
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    check_name: str
    message: str
    severity: str  # 'critical', 'warning', 'info'
    details: Optional[Dict] = None


class DataValidator:
    """Validates data quality and feature completeness for trading signals."""
    
    # Thresholds for validation
    MAX_NAN_RATIO = 0.05  # Max 5% NaN values allowed
    MAX_INF_RATIO = 0.01  # Max 1% Inf values allowed
    MIN_BARS_REQUIRED = 200  # Minimum bars for indicator calculation
    MAX_DATA_AGE_MINUTES = 10  # Max age of data before warning
    MIN_PRICE = 0.01  # Minimum valid price
    MAX_PRICE_CHANGE_PCT = 10  # Max single-bar price change (%)
    
    def __init__(self, model_dir: str, feature_columns_file: str = 'feature_columns.json'):
        """Initialize validator with model feature requirements."""
        self.model_dir = model_dir
        self.feature_columns = self._load_feature_columns(feature_columns_file)
        self.validation_history: List[ValidationResult] = []
        self.alert_callbacks: List[callable] = []
        # Determine bot type from model_dir
        self.bot_type = 'mnq' if 'mnq' in model_dir.lower() else 'btc'
        
    def _load_feature_columns(self, filename: str) -> List[str]:
        """Load required feature columns from model directory."""
        path = os.path.join(self.model_dir, filename)
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load feature columns from {path}: {e}")
            return []
    
    def add_alert_callback(self, callback: callable) -> None:
        """Add a callback function for alerts (e.g., email, Slack, log)."""
        self.alert_callbacks.append(callback)
    
    def _trigger_alert(self, result: ValidationResult) -> None:
        """Trigger alert callbacks for validation failures."""
        for callback in self.alert_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def validate_all(self, df: pd.DataFrame, current_price: Optional[float] = None) -> Tuple[bool, List[ValidationResult]]:
        """Run all validation checks and return overall result."""
        results = []
        
        # 1. Data freshness
        results.append(self.check_data_freshness(df))
        
        # 2. Data completeness (enough bars)
        results.append(self.check_data_completeness(df))
        
        # 3. Feature availability
        results.append(self.check_feature_availability(df))
        
        # 4. NaN values
        results.append(self.check_nan_values(df))
        
        # 5. Inf values
        results.append(self.check_inf_values(df))
        
        # 6. Price validity
        results.append(self.check_price_validity(df))
        
        # 7. Price continuity (no gaps)
        results.append(self.check_price_continuity(df))
        
        # 8. Volume validity
        results.append(self.check_volume_validity(df))
        
        # 9. Feature ranges (detect anomalies)
        results.append(self.check_feature_ranges(df))
        
        # 10. Current price validity (if provided)
        if current_price is not None:
            results.append(self.check_current_price(df, current_price))
        
        # Store history
        self.validation_history.extend(results)
        
        # Trigger alerts for all failures (critical and warnings)
        for result in results:
            if not result.is_valid:
                self._trigger_alert(result)
        
        # Overall validity: no critical failures
        is_valid = all(r.is_valid or r.severity != 'critical' for r in results)
        
        return is_valid, results
    
    def check_data_freshness(self, df: pd.DataFrame) -> ValidationResult:
        """Check if data is recent enough."""
        if df.empty:
            return ValidationResult(
                is_valid=False,
                check_name='data_freshness',
                message='DataFrame is empty',
                severity='critical'
            )
        
        last_time = df.index[-1]
        if isinstance(last_time, pd.Timestamp):
            last_time = last_time.to_pydatetime()
        
        # Handle timezone comparison properly
        # If last_time is timezone-aware (UTC), convert now to UTC
        # If last_time is timezone-naive, assume it's UTC and use UTC now
        from datetime import timezone
        if hasattr(last_time, 'tzinfo') and last_time.tzinfo is not None:
            # last_time is timezone-aware, use UTC now
            now = datetime.now(timezone.utc)
        else:
            # last_time is naive, assume UTC - use UTC now without tzinfo
            now = datetime.utcnow()
        
        # Make both naive for comparison
        if hasattr(last_time, 'tzinfo') and last_time.tzinfo is not None:
            last_time = last_time.replace(tzinfo=None)
            now = now.replace(tzinfo=None)
        
        age_minutes = (now - last_time).total_seconds() / 60
        
        # Check for CME holidays (MNQ) - allow extended staleness
        # BTC trades 24/7 so no holiday exception needed
        import pytz
        et = pytz.timezone('US/Eastern')
        now_et = datetime.now(et)
        month_day = (now_et.month, now_et.day)
        cme_holidays = [(12, 24), (12, 25), (1, 1), (7, 4), (11, 28), (11, 29)]
        
        max_age = self.MAX_DATA_AGE_MINUTES
        if month_day in cme_holidays and self.bot_type == 'mnq':
            max_age = 4000  # ~3 days for CME holiday closures
            if age_minutes > self.MAX_DATA_AGE_MINUTES:
                return ValidationResult(
                    is_valid=True,  # Valid during holiday
                    check_name='data_freshness',
                    message=f'CME holiday - data is {age_minutes:.1f} minutes old (holiday closure expected)',
                    severity='info',
                    details={'age_minutes': age_minutes, 'last_time': str(last_time), 'holiday': True}
                )
        
        if age_minutes > max_age:
            return ValidationResult(
                is_valid=False,
                check_name='data_freshness',
                message=f'Data is {age_minutes:.1f} minutes old (max: {max_age})',
                severity='warning',
                details={'age_minutes': age_minutes, 'last_time': str(last_time)}
            )
        
        return ValidationResult(
            is_valid=True,
            check_name='data_freshness',
            message=f'Data is {age_minutes:.1f} minutes old',
            severity='info',
            details={'age_minutes': age_minutes}
        )
    
    def check_data_completeness(self, df: pd.DataFrame) -> ValidationResult:
        """Check if we have enough bars for indicator calculation."""
        bar_count = len(df)
        
        if bar_count < self.MIN_BARS_REQUIRED:
            return ValidationResult(
                is_valid=False,
                check_name='data_completeness',
                message=f'Only {bar_count} bars available (need {self.MIN_BARS_REQUIRED})',
                severity='critical',
                details={'bar_count': bar_count, 'required': self.MIN_BARS_REQUIRED}
            )
        
        return ValidationResult(
            is_valid=True,
            check_name='data_completeness',
            message=f'{bar_count} bars available',
            severity='info',
            details={'bar_count': bar_count}
        )
    
    def check_feature_availability(self, df: pd.DataFrame) -> ValidationResult:
        """Check if all required features are present."""
        if not self.feature_columns:
            return ValidationResult(
                is_valid=False,
                check_name='feature_availability',
                message='No feature columns loaded',
                severity='critical'
            )
        
        available = set(df.columns)
        required = set(self.feature_columns)
        missing = required - available
        
        if missing:
            return ValidationResult(
                is_valid=False,
                check_name='feature_availability',
                message=f'{len(missing)} features missing',
                severity='critical',
                details={'missing_count': len(missing), 'missing_features': list(missing)[:10]}
            )
        
        return ValidationResult(
            is_valid=True,
            check_name='feature_availability',
            message=f'All {len(required)} features available',
            severity='info',
            details={'feature_count': len(required)}
        )
    
    def check_nan_values(self, df: pd.DataFrame) -> ValidationResult:
        """Check for NaN values in feature columns."""
        if not self.feature_columns:
            return ValidationResult(is_valid=True, check_name='nan_values', message='No features to check', severity='info')
        
        feature_df = df[[c for c in self.feature_columns if c in df.columns]]
        last_row = feature_df.iloc[-1] if len(feature_df) > 0 else pd.Series()
        
        nan_count = last_row.isna().sum()
        nan_ratio = nan_count / len(last_row) if len(last_row) > 0 else 0
        
        if nan_ratio > self.MAX_NAN_RATIO:
            nan_features = last_row[last_row.isna()].index.tolist()
            return ValidationResult(
                is_valid=False,
                check_name='nan_values',
                message=f'{nan_count} NaN values ({nan_ratio:.1%}) in last row',
                severity='critical',
                details={'nan_count': nan_count, 'nan_features': nan_features[:10]}
            )
        
        return ValidationResult(
            is_valid=True,
            check_name='nan_values',
            message=f'{nan_count} NaN values ({nan_ratio:.1%})',
            severity='info',
            details={'nan_count': nan_count}
        )
    
    def check_inf_values(self, df: pd.DataFrame) -> ValidationResult:
        """Check for Inf values in feature columns."""
        if not self.feature_columns:
            return ValidationResult(is_valid=True, check_name='inf_values', message='No features to check', severity='info')
        
        feature_df = df[[c for c in self.feature_columns if c in df.columns]]
        last_row = feature_df.iloc[-1] if len(feature_df) > 0 else pd.Series()
        
        # Replace NaN with 0 for inf check
        last_row_clean = last_row.fillna(0)
        inf_mask = np.isinf(last_row_clean)
        inf_count = inf_mask.sum()
        inf_ratio = inf_count / len(last_row) if len(last_row) > 0 else 0
        
        if inf_ratio > self.MAX_INF_RATIO:
            inf_features = last_row_clean[inf_mask].index.tolist()
            return ValidationResult(
                is_valid=False,
                check_name='inf_values',
                message=f'{inf_count} Inf values ({inf_ratio:.1%}) in last row',
                severity='critical',
                details={'inf_count': inf_count, 'inf_features': inf_features[:10]}
            )
        
        return ValidationResult(
            is_valid=True,
            check_name='inf_values',
            message=f'{inf_count} Inf values',
            severity='info',
            details={'inf_count': inf_count}
        )
    
    def check_price_validity(self, df: pd.DataFrame) -> ValidationResult:
        """Check if prices are valid (positive, reasonable)."""
        if 'Close' not in df.columns:
            return ValidationResult(
                is_valid=False,
                check_name='price_validity',
                message='Close column not found',
                severity='critical'
            )
        
        last_price = df['Close'].iloc[-1]
        
        if pd.isna(last_price) or last_price <= self.MIN_PRICE:
            return ValidationResult(
                is_valid=False,
                check_name='price_validity',
                message=f'Invalid price: {last_price}',
                severity='critical',
                details={'last_price': last_price}
            )
        
        return ValidationResult(
            is_valid=True,
            check_name='price_validity',
            message=f'Price: ${last_price:.2f}',
            severity='info',
            details={'last_price': last_price}
        )
    
    def check_price_continuity(self, df: pd.DataFrame) -> ValidationResult:
        """Check for abnormal price jumps (potential data errors)."""
        if 'Close' not in df.columns or len(df) < 2:
            return ValidationResult(is_valid=True, check_name='price_continuity', message='Not enough data', severity='info')
        
        returns = df['Close'].pct_change().abs() * 100
        max_return = returns.iloc[-10:].max() if len(returns) >= 10 else returns.max()
        
        if max_return > self.MAX_PRICE_CHANGE_PCT:
            return ValidationResult(
                is_valid=False,
                check_name='price_continuity',
                message=f'Large price jump detected: {max_return:.1f}%',
                severity='warning',
                details={'max_return_pct': max_return}
            )
        
        return ValidationResult(
            is_valid=True,
            check_name='price_continuity',
            message=f'Max recent return: {max_return:.2f}%',
            severity='info',
            details={'max_return_pct': max_return}
        )
    
    def check_volume_validity(self, df: pd.DataFrame) -> ValidationResult:
        """Check if volume data is valid."""
        if 'Volume' not in df.columns:
            return ValidationResult(
                is_valid=False,
                check_name='volume_validity',
                message='Volume column not found',
                severity='warning'
            )
        
        last_volume = df['Volume'].iloc[-1]
        avg_volume = df['Volume'].tail(50).mean()
        
        if last_volume < 0:
            return ValidationResult(
                is_valid=False,
                check_name='volume_validity',
                message=f'Negative volume: {last_volume}',
                severity='critical',
                details={'last_volume': last_volume}
            )
        
        # Zero volume is okay for synthetic bars
        return ValidationResult(
            is_valid=True,
            check_name='volume_validity',
            message=f'Volume: {last_volume:,.0f} (avg: {avg_volume:,.0f})',
            severity='info',
            details={'last_volume': last_volume, 'avg_volume': avg_volume}
        )
    
    def check_feature_ranges(self, df: pd.DataFrame) -> ValidationResult:
        """Check if key features are within expected ranges."""
        issues = []
        
        # RSI should be 0-100
        if 'momentum_rsi' in df.columns:
            rsi = df['momentum_rsi'].iloc[-1]
            if not (0 <= rsi <= 100):
                issues.append(f'RSI out of range: {rsi:.1f}')
        
        # ADX should be 0-100
        if 'trend_adx' in df.columns:
            adx = df['trend_adx'].iloc[-1]
            if not (0 <= adx <= 100):
                issues.append(f'ADX out of range: {adx:.1f}')
        
        # Bollinger %B typically -0.5 to 1.5
        if 'volatility_bbp' in df.columns:
            bbp = df['volatility_bbp'].iloc[-1]
            if not (-1 <= bbp <= 2):
                issues.append(f'BB%B unusual: {bbp:.2f}')
        
        if issues:
            return ValidationResult(
                is_valid=False,
                check_name='feature_ranges',
                message=f'{len(issues)} features out of range',
                severity='warning',
                details={'issues': issues}
            )
        
        return ValidationResult(
            is_valid=True,
            check_name='feature_ranges',
            message='All features within expected ranges',
            severity='info'
        )
    
    def check_current_price(self, df: pd.DataFrame, current_price: float) -> ValidationResult:
        """Check if current price is consistent with historical data."""
        if 'Close' not in df.columns:
            return ValidationResult(is_valid=True, check_name='current_price', message='No Close column', severity='info')
        
        last_close = df['Close'].iloc[-1]
        price_diff_pct = abs(current_price - last_close) / last_close * 100
        
        if price_diff_pct > 5:  # More than 5% difference
            return ValidationResult(
                is_valid=False,
                check_name='current_price',
                message=f'Current price ${current_price:.2f} differs from last close ${last_close:.2f} by {price_diff_pct:.1f}%',
                severity='warning',
                details={'current_price': current_price, 'last_close': last_close, 'diff_pct': price_diff_pct}
            )
        
        return ValidationResult(
            is_valid=True,
            check_name='current_price',
            message=f'Price consistent: ${current_price:.2f} vs ${last_close:.2f}',
            severity='info',
            details={'diff_pct': price_diff_pct}
        )
    
    def validate_signal(self, probability: float, direction: str, model_name: str) -> ValidationResult:
        """Validate a trading signal before execution."""
        issues = []
        
        # Check probability range
        if not (0 <= probability <= 1):
            issues.append(f'Invalid probability: {probability}')
        
        # Check direction
        if direction not in ['LONG', 'SHORT', 'long', 'short']:
            issues.append(f'Invalid direction: {direction}')
        
        # Check for extreme probabilities (might indicate data issue)
        if probability > 0.99:
            issues.append(f'Suspiciously high probability: {probability:.2%}')
        
        if issues:
            return ValidationResult(
                is_valid=False,
                check_name='signal_validation',
                message=f'Signal validation failed: {", ".join(issues)}',
                severity='critical',
                details={'probability': probability, 'direction': direction, 'model': model_name, 'issues': issues}
            )
        
        return ValidationResult(
            is_valid=True,
            check_name='signal_validation',
            message=f'Signal valid: {model_name} {direction} {probability:.1%}',
            severity='info',
            details={'probability': probability, 'direction': direction, 'model': model_name}
        )
    
    def get_validation_summary(self) -> str:
        """Get a summary of recent validation results."""
        if not self.validation_history:
            return "No validation history"
        
        recent = self.validation_history[-20:]  # Last 20 checks
        critical = sum(1 for r in recent if not r.is_valid and r.severity == 'critical')
        warnings = sum(1 for r in recent if not r.is_valid and r.severity == 'warning')
        passed = sum(1 for r in recent if r.is_valid)
        
        return f"Last 20 checks: {passed} passed, {critical} critical, {warnings} warnings"


def log_alert(result: ValidationResult) -> None:
    """Default alert callback - logs to file."""
    alert_log = 'logs/validation_alerts.log'
    os.makedirs(os.path.dirname(alert_log), exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(alert_log, 'a') as f:
        f.write(f"{timestamp} | {result.severity.upper()} | {result.check_name} | {result.message}\n")
        if result.details:
            f.write(f"  Details: {result.details}\n")


# Convenience function for quick validation
def validate_features_quick(df: pd.DataFrame, model_dir: str) -> Tuple[bool, str]:
    """Quick validation check - returns (is_valid, message)."""
    validator = DataValidator(model_dir)
    is_valid, results = validator.validate_all(df)
    
    if is_valid:
        return True, "All validation checks passed"
    
    failures = [r for r in results if not r.is_valid]
    messages = [f"{r.check_name}: {r.message}" for r in failures]
    return False, "; ".join(messages)


if __name__ == "__main__":
    # Test the validator
    print("Testing DataValidator...")
    
    # Load sample data
    df = pd.read_parquet('data/QQQ_features.parquet').tail(500)
    
    # Create validator
    validator = DataValidator('models_mnq_v2')
    validator.add_alert_callback(log_alert)
    
    # Run validation
    is_valid, results = validator.validate_all(df)
    
    print(f"\nOverall valid: {is_valid}")
    print("\nResults:")
    for r in results:
        status = "✅" if r.is_valid else ("⚠️" if r.severity == 'warning' else "❌")
        print(f"  {status} {r.check_name}: {r.message}")
    
    print(f"\n{validator.get_validation_summary()}")
