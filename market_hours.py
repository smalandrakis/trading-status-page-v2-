"""
CME Bitcoin Futures (MBT) Market Hours Checker
Trading Hours: Sunday 6:00 PM CT - Friday 5:00 PM CT (closed Saturday)
Daily break: 5:00 PM - 6:00 PM CT
"""

from datetime import datetime
import pytz

def is_market_open():
    """
    Check if CME Bitcoin futures market is currently open
    
    Trading hours:
    - Sunday 6:00 PM CT - Friday 5:00 PM CT
    - Closed all day Saturday
    - Daily 1-hour break from 5:00 PM - 6:00 PM CT
    
    Returns:
        bool: True if market is open, False if closed
    """
    # Get current time in Chicago (CME is in CT)
    chicago_tz = pytz.timezone('America/Chicago')
    now_ct = datetime.now(chicago_tz)
    
    weekday = now_ct.weekday()  # 0=Monday, 6=Sunday
    hour = now_ct.hour
    minute = now_ct.minute
    
    # Saturday (5) - market closed all day
    if weekday == 5:
        return False
    
    # Friday (4) - closes at 5:00 PM CT
    if weekday == 4 and hour >= 17:
        return False
    
    # Sunday (6) - opens at 6:00 PM CT
    if weekday == 6 and hour < 18:
        return False
    
    # Daily break from 5:00 PM - 6:00 PM CT (Monday-Thursday)
    if weekday in [0, 1, 2, 3, 4] and hour == 17:
        return False
    
    return True


def get_next_open_time():
    """
    Get the next market open time
    
    Returns:
        str: Human-readable description of when market opens next
    """
    chicago_tz = pytz.timezone('America/Chicago')
    now_ct = datetime.now(chicago_tz)
    
    weekday = now_ct.weekday()
    hour = now_ct.hour
    
    # Saturday - opens Sunday 6:00 PM CT
    if weekday == 5:
        return "Sunday 6:00 PM CT"
    
    # Friday after 5:00 PM - opens Sunday 6:00 PM CT
    if weekday == 4 and hour >= 17:
        return "Sunday 6:00 PM CT"
    
    # Sunday before 6:00 PM - opens today at 6:00 PM CT
    if weekday == 6 and hour < 18:
        return "Today 6:00 PM CT"
    
    # Daily break 5:00-6:00 PM CT
    if hour == 17:
        return "Today 6:00 PM CT"
    
    return "Market is open"


class MarketHours:
    """CME MBT futures market hours with holiday support.
    
    Regular hours: Sunday 5 PM CT - Friday 4 PM CT
    Daily maintenance: 4-5 PM CT (Mon-Thu) / 5-6 PM CT actual break
    """
    
    def __init__(self):
        self.tz = pytz.timezone('America/Chicago')
    
    def is_market_open(self):
        """Check if market is open. Returns (bool, reason_string)."""
        now_ct = datetime.now(self.tz)
        weekday = now_ct.weekday()
        hour = now_ct.hour
        
        if weekday == 5:
            return False, "Saturday - market closed"
        
        if weekday == 4 and hour >= 17:
            return False, "Friday after 5 PM CT - weekend"
        
        if weekday == 6 and hour < 18:
            return False, "Sunday before 6 PM CT - not yet open"
        
        if weekday in [0, 1, 2, 3, 4] and hour == 17:
            return False, "Daily maintenance break (5-6 PM CT)"
        
        return True, "Market open"
    
    def should_close_positions(self):
        """Check if positions should be closed before market close.
        
        Returns True if within 30 minutes of close (Friday 4:30 PM CT)
        or within 15 minutes of daily break.
        """
        now_ct = datetime.now(self.tz)
        weekday = now_ct.weekday()
        hour = now_ct.hour
        minute = now_ct.minute
        
        # Friday: close positions by 4:30 PM CT (30 min before weekend close)
        if weekday == 4 and hour == 16 and minute >= 30:
            return True
        
        # Daily break: close by 4:45 PM CT (15 min before break)
        if weekday in [0, 1, 2, 3] and hour == 16 and minute >= 45:
            return True
        
        return False
    
    def should_block_new_entries(self, min_holding_time_minutes=60):
        """Check if new entries should be blocked due to insufficient time.
        
        Returns (should_block: bool, reason: str).
        """
        now_ct = datetime.now(self.tz)
        weekday = now_ct.weekday()
        hour = now_ct.hour
        minute = now_ct.minute
        time_in_minutes = hour * 60 + minute
        
        # Friday: block if less than min_holding_time before 5 PM CT (17:00 = 1020 min)
        if weekday == 4:
            minutes_to_close = 1020 - time_in_minutes
            if minutes_to_close < min_holding_time_minutes:
                return True, f"Only {minutes_to_close}min to Friday close"
        
        # Mon-Thu: block if less than min_holding_time before daily break (5 PM CT)
        if weekday in [0, 1, 2, 3]:
            minutes_to_break = 1020 - time_in_minutes
            if 0 < minutes_to_break < min_holding_time_minutes:
                return True, f"Only {minutes_to_break}min to daily break"
        
        return False, ""


def get_cme_market_hours():
    """Factory function returning a MarketHours instance."""
    return MarketHours()


if __name__ == '__main__':
    mh = MarketHours()
    is_open, reason = mh.is_market_open()
    if is_open:
        print("✓ CME Bitcoin futures market is OPEN")
    else:
        print(f"✗ CME Bitcoin futures market is CLOSED")
        print(f"  Reason: {reason}")
        print(f"  Next open: {get_next_open_time()}")
