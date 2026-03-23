"""
Market Hours and Holiday Management for Trading Bots

This module provides utilities to:
1. Check if markets are open
2. Get time until market close
3. Determine if positions should be closed before market closure
4. Handle CME holidays and early closes
5. Handle weekend closures

CME Futures (MBT, MNQ, MES) Trading Hours:
- Regular: Sunday 5:00 PM CT to Friday 4:00 PM CT
- Daily maintenance halt: 4:00 PM - 5:00 PM CT (Mon-Thu)
- Holidays: Various closures and early closes

Binance (BTC spot) Trading Hours:
- 24/7/365 (no closures)
"""

from datetime import datetime, time, timedelta
from typing import Tuple, Optional, Dict
import pytz

# Timezones
CT = pytz.timezone('US/Central')  # CME timezone
UTC = pytz.UTC
PARIS = pytz.timezone('Europe/Paris')

# Pre-close buffer (minutes before market close to exit positions)
PRE_CLOSE_BUFFER_MINUTES = 15

# CME Holidays 2025-2026 (date -> (name, close_time_ct or None for full close))
CME_HOLIDAYS = {
    # 2025
    '2025-01-01': ('New Year Day', None),  # Closed
    '2025-01-20': ('MLK Day', time(12, 0)),  # Early close 12:00 CT
    '2025-02-17': ('Presidents Day', time(12, 0)),
    '2025-04-18': ('Good Friday', None),  # Closed
    '2025-05-26': ('Memorial Day', time(12, 0)),
    '2025-06-19': ('Juneteenth', time(12, 0)),
    '2025-07-04': ('Independence Day', None),  # Closed
    '2025-09-01': ('Labor Day', time(12, 0)),
    '2025-11-27': ('Thanksgiving', time(12, 15)),
    '2025-11-28': ('Day after Thanksgiving', time(12, 15)),
    '2025-12-24': ('Christmas Eve', time(12, 45)),
    '2025-12-25': ('Christmas Day', None),  # Closed
    '2025-12-31': ('New Year Eve', time(12, 0)),
    # 2026
    '2026-01-01': ('New Year Day', None),
    '2026-01-19': ('MLK Day', time(12, 0)),
    '2026-02-16': ('Presidents Day', time(12, 0)),
    '2026-04-03': ('Good Friday', None),
    '2026-05-25': ('Memorial Day', time(12, 0)),
    '2026-06-19': ('Juneteenth', time(12, 0)),
    '2026-07-03': ('Independence Day (observed)', None),
    '2026-09-07': ('Labor Day', time(12, 0)),
    '2026-11-26': ('Thanksgiving', time(12, 15)),
    '2026-11-27': ('Day after Thanksgiving', time(12, 15)),
    '2026-12-24': ('Christmas Eve', time(12, 45)),
    '2026-12-25': ('Christmas Day', None),
    '2026-12-31': ('New Year Eve', time(12, 0)),
}

# Regular CME trading hours (CT)
CME_REGULAR_OPEN = time(17, 0)   # 5:00 PM CT (Sunday open)
CME_REGULAR_CLOSE = time(16, 0)  # 4:00 PM CT (Friday close)
CME_DAILY_HALT_START = time(16, 0)  # 4:00 PM CT
CME_DAILY_HALT_END = time(17, 0)    # 5:00 PM CT


class MarketHours:
    """Market hours manager for trading bots."""
    
    def __init__(self, market_type: str = 'cme'):
        """
        Initialize market hours manager.
        
        Args:
            market_type: 'cme' for CME futures (MBT, MNQ, MES), 'crypto' for 24/7 markets
        """
        self.market_type = market_type.lower()
        self.pre_close_buffer = timedelta(minutes=PRE_CLOSE_BUFFER_MINUTES)
    
    def get_current_time_ct(self) -> datetime:
        """Get current time in CT (Chicago) timezone."""
        return datetime.now(CT)
    
    def is_holiday(self, dt: Optional[datetime] = None) -> Tuple[bool, Optional[str], Optional[time]]:
        """
        Check if given date is a CME holiday.
        
        Returns:
            Tuple of (is_holiday, holiday_name, early_close_time or None if fully closed)
        """
        if self.market_type == 'crypto':
            return False, None, None
        
        if dt is None:
            dt = self.get_current_time_ct()
        
        date_str = dt.strftime('%Y-%m-%d')
        if date_str in CME_HOLIDAYS:
            name, close_time = CME_HOLIDAYS[date_str]
            return True, name, close_time
        return False, None, None
    
    def is_weekend(self, dt: Optional[datetime] = None) -> bool:
        """Check if current time is during weekend closure."""
        if self.market_type == 'crypto':
            return False
        
        if dt is None:
            dt = self.get_current_time_ct()
        
        weekday = dt.weekday()  # 0=Monday, 6=Sunday
        current_time = dt.time()
        
        # Saturday: fully closed
        if weekday == 5:
            return True
        
        # Sunday: closed until 5:00 PM CT
        if weekday == 6 and current_time < CME_REGULAR_OPEN:
            return True
        
        # Friday: closed after 4:00 PM CT
        if weekday == 4 and current_time >= CME_REGULAR_CLOSE:
            return True
        
        return False
    
    def is_daily_halt(self, dt: Optional[datetime] = None) -> bool:
        """Check if current time is during daily maintenance halt (4-5 PM CT Mon-Thu)."""
        if self.market_type == 'crypto':
            return False
        
        if dt is None:
            dt = self.get_current_time_ct()
        
        weekday = dt.weekday()
        current_time = dt.time()
        
        # Daily halt only Mon-Thu (0-3)
        if weekday <= 3:
            if CME_DAILY_HALT_START <= current_time < CME_DAILY_HALT_END:
                return True
        
        return False
    
    def is_market_open(self, dt: Optional[datetime] = None) -> Tuple[bool, str]:
        """
        Check if market is currently open.
        
        Returns:
            Tuple of (is_open, reason_if_closed)
        """
        if self.market_type == 'crypto':
            return True, ''
        
        if dt is None:
            dt = self.get_current_time_ct()
        
        # Check holiday
        is_hol, hol_name, early_close = self.is_holiday(dt)
        if is_hol:
            if early_close is None:
                return False, f'Holiday: {hol_name} (closed)'
            elif dt.time() >= early_close:
                return False, f'Holiday: {hol_name} (early close at {early_close})'
        
        # Check weekend
        if self.is_weekend(dt):
            return False, 'Weekend closure'
        
        # Check daily halt
        if self.is_daily_halt(dt):
            return False, 'Daily maintenance halt (4-5 PM CT)'
        
        return True, ''
    
    def get_next_close_time(self, dt: Optional[datetime] = None) -> Tuple[datetime, str]:
        """
        Get the next market close time.
        
        Returns:
            Tuple of (close_datetime_ct, close_reason)
        """
        if self.market_type == 'crypto':
            # Crypto never closes, return far future
            return datetime(2099, 12, 31, tzinfo=CT), 'Never (24/7 market)'
        
        if dt is None:
            dt = self.get_current_time_ct()
        
        weekday = dt.weekday()
        current_time = dt.time()
        today_date = dt.date()
        
        # Check if today is a holiday with early close
        is_hol, hol_name, early_close = self.is_holiday(dt)
        if is_hol and early_close is not None:
            close_dt = CT.localize(datetime.combine(today_date, early_close))
            if dt < close_dt:
                return close_dt, f'Holiday early close: {hol_name}'
        
        # NOTE: Daily halt (4-5 PM CT Mon-Thu) is intentionally NOT included here
        # It's only 1 hour and not worth closing positions for such a short break
        # The market reopens immediately after, so we let positions ride through
        
        # Friday close at 4 PM CT
        if weekday == 4:  # Friday
            friday_close = CT.localize(datetime.combine(today_date, CME_REGULAR_CLOSE))
            if dt < friday_close:
                return friday_close, 'Weekend closure (Friday 4 PM CT)'
        
        # If we're past today's close, find next close
        # This handles edge cases like being in the halt period
        
        # Default: next daily halt or Friday close
        days_until_friday = (4 - weekday) % 7
        if days_until_friday == 0 and current_time >= CME_REGULAR_CLOSE:
            days_until_friday = 7
        
        next_friday = today_date + timedelta(days=days_until_friday)
        next_close = CT.localize(datetime.combine(next_friday, CME_REGULAR_CLOSE))
        
        # Check for earlier closes (holidays only - daily halt excluded)
        for days_ahead in range(days_until_friday + 1):
            check_date = today_date + timedelta(days=days_ahead)
            check_weekday = check_date.weekday()
            
            # Skip weekends
            if check_weekday >= 5:
                continue
            
            # Check holiday
            check_date_str = check_date.strftime('%Y-%m-%d')
            if check_date_str in CME_HOLIDAYS:
                _, early_close = CME_HOLIDAYS[check_date_str]
                if early_close is not None:
                    holiday_close = CT.localize(datetime.combine(check_date, early_close))
                    if holiday_close > dt and holiday_close < next_close:
                        next_close = holiday_close
                        return next_close, f'Holiday early close'
            
            # NOTE: Daily halt (Mon-Thu 4-5 PM CT) intentionally NOT checked here
            # It's only 1 hour - not worth closing positions for such a short break
        
        return next_close, 'Weekend closure'
    
    def get_minutes_until_close(self, dt: Optional[datetime] = None) -> Tuple[float, str]:
        """
        Get minutes until next market close.
        
        Returns:
            Tuple of (minutes_until_close, close_reason)
        """
        if dt is None:
            dt = self.get_current_time_ct()
        
        close_time, reason = self.get_next_close_time(dt)
        
        # Handle timezone-aware comparison
        if dt.tzinfo is None:
            dt = CT.localize(dt)
        
        delta = close_time - dt
        minutes = delta.total_seconds() / 60
        
        return max(0, minutes), reason
    
    def should_close_positions(self, dt: Optional[datetime] = None) -> Tuple[bool, str]:
        """
        Check if positions should be closed due to upcoming market closure.
        
        Returns:
            Tuple of (should_close, reason)
        """
        if self.market_type == 'crypto':
            return False, ''
        
        if dt is None:
            dt = self.get_current_time_ct()
        
        # Check if market is already closed
        is_open, closed_reason = self.is_market_open(dt)
        if not is_open:
            return True, f'Market closed: {closed_reason}'
        
        # Check time until close
        minutes_until_close, close_reason = self.get_minutes_until_close(dt)
        
        if minutes_until_close <= PRE_CLOSE_BUFFER_MINUTES:
            return True, f'Pre-close buffer ({int(minutes_until_close)} min until {close_reason})'
        
        return False, ''
    
    def should_block_new_entries(self, dt: Optional[datetime] = None, 
                                  min_holding_time_minutes: int = 60) -> Tuple[bool, str]:
        """
        Check if new position entries should be blocked due to insufficient time before close.
        
        Args:
            min_holding_time_minutes: Minimum time needed to hold a position
        
        Returns:
            Tuple of (should_block, reason)
        """
        if self.market_type == 'crypto':
            return False, ''
        
        if dt is None:
            dt = self.get_current_time_ct()
        
        # Check if market is open
        is_open, closed_reason = self.is_market_open(dt)
        if not is_open:
            return True, f'Market closed: {closed_reason}'
        
        # Check time until close
        minutes_until_close, close_reason = self.get_minutes_until_close(dt)
        
        # Block if not enough time to hold position
        required_time = min_holding_time_minutes + PRE_CLOSE_BUFFER_MINUTES
        if minutes_until_close <= required_time:
            return True, f'Insufficient time before {close_reason} ({int(minutes_until_close)} min available, need {required_time})'
        
        return False, ''
    
    def get_market_status(self, dt: Optional[datetime] = None) -> Dict:
        """
        Get comprehensive market status.
        
        Returns:
            Dict with market status information
        """
        if dt is None:
            dt = self.get_current_time_ct()
        
        is_open, closed_reason = self.is_market_open(dt)
        minutes_until_close, close_reason = self.get_minutes_until_close(dt)
        should_close, close_positions_reason = self.should_close_positions(dt)
        should_block, block_reason = self.should_block_new_entries(dt)
        is_hol, hol_name, _ = self.is_holiday(dt)
        
        return {
            'current_time_ct': dt.strftime('%Y-%m-%d %H:%M:%S %Z'),
            'market_type': self.market_type,
            'is_open': is_open,
            'closed_reason': closed_reason,
            'is_holiday': is_hol,
            'holiday_name': hol_name,
            'is_weekend': self.is_weekend(dt),
            'is_daily_halt': self.is_daily_halt(dt),
            'minutes_until_close': minutes_until_close,
            'close_reason': close_reason,
            'should_close_positions': should_close,
            'close_positions_reason': close_positions_reason,
            'should_block_entries': should_block,
            'block_entries_reason': block_reason,
        }


# Convenience functions for direct use
def get_cme_market_hours() -> MarketHours:
    """Get CME market hours manager."""
    return MarketHours('cme')

def get_crypto_market_hours() -> MarketHours:
    """Get crypto market hours manager (24/7)."""
    return MarketHours('crypto')

def should_close_cme_positions() -> Tuple[bool, str]:
    """Quick check if CME positions should be closed."""
    return MarketHours('cme').should_close_positions()

def should_block_cme_entries(min_holding_minutes: int = 60) -> Tuple[bool, str]:
    """Quick check if CME entries should be blocked."""
    return MarketHours('cme').should_block_new_entries(min_holding_time_minutes=min_holding_minutes)


if __name__ == '__main__':
    # Test the module
    print("=== CME Market Hours Test ===\n")
    
    cme = MarketHours('cme')
    status = cme.get_market_status()
    
    for key, value in status.items():
        print(f"{key}: {value}")
    
    print("\n=== Crypto Market Hours Test ===\n")
    
    crypto = MarketHours('crypto')
    status = crypto.get_market_status()
    
    for key, value in status.items():
        print(f"{key}: {value}")
