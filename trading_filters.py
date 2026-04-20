"""
Trading Filters Module
Comprehensive filter system for all trading bots to prevent bad trades.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import pytz
import logging

logger = logging.getLogger(__name__)


class TradingFilters:
    """Comprehensive filter system for trade validation"""

    def __init__(self, tp_pct, sl_pct):
        """
        Initialize filters with bot's TP/SL configuration

        Args:
            tp_pct: Take profit percentage (e.g., 1.0 for 1%)
            sl_pct: Stop loss percentage (e.g., 0.5 for 0.5%)
        """
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct

        # Filter state
        self.consecutive_losses = 0
        self.daily_trades = []
        self.recent_slippage = []

        # Filter thresholds
        self.MAX_CONSECUTIVE_LOSSES = 10
        self.MAX_DAILY_TRADES = 15

        # Adaptive slippage safety ratio based on TP/SL size
        # For tight targets (<1.5% TP), slippage is inherent cost - use lower threshold
        if tp_pct < 1.5:
            self.SLIPPAGE_SAFETY_RATIO = 1.0  # HF bots (1.0%/0.5%)
        elif tp_pct < 2.0:
            self.SLIPPAGE_SAFETY_RATIO = 1.5  # Mid-frequency bots
        else:
            self.SLIPPAGE_SAFETY_RATIO = 2.0  # Swing bots (2.5%+)

        self.SR_PROXIMITY_PCT = 0.002  # 0.2%
        self.POSITION_SIZE_CAP = 8  # Max 8x contracts
        self.MAX_VOLATILITY_MULTIPLIER = 3.0
        self.LOW_LIQUIDITY_HOURS = []  # DISABLED - was [(2, 6)] for 2am-6am ET

        logger.info("TradingFilters initialized:")
        logger.info(f"  TP/SL: {tp_pct}% / {sl_pct}%")
        logger.info(f"  Max Consecutive Losses: {self.MAX_CONSECUTIVE_LOSSES}")
        logger.info(f"  Max Daily Trades: {self.MAX_DAILY_TRADES}")

    def check_all_filters(self, signal, confidence, current_price, df, position_size):
        """
        Run all filters on a potential trade

        Returns:
            (pass, reason) - (True, None) if all pass, (False, reason) if any fail
        """

        # ALL FILTERS DISABLED - Pure backtest replication (no filters, trust the model)
        # pass_filter, reason = self.check_slippage_safety(signal, current_price, position_size)
        # if not pass_filter:
        #     return False, f"[SLIPPAGE] {reason}"

        # pass_filter, reason = self.check_consecutive_losses()
        # if not pass_filter:
        #     return False, f"[KILL-SWITCH] {reason}"

        # pass_filter, reason = self.check_support_resistance(signal, current_price, df)
        # if not pass_filter:
        #     return False, f"[S/R] {reason}"

        # pass_filter, reason = self.check_time_of_day()
        # if not pass_filter:
        #     return False, f"[TIME] {reason}"

        # pass_filter, reason = self.check_volatility_regime(df)
        # if not pass_filter:
        #     return False, f"[VOLATILITY] {reason}"

        # pass_filter, reason = self.check_daily_trade_limit()
        # if not pass_filter:
        #     return False, f"[DAILY-LIMIT] {reason}"

        # pass_filter, reason = self.check_position_accumulation(position_size)
        # if not pass_filter:
        #     return False, f"[POSITION-CAP] {reason}"

        return True, None

    # =========================================================================
    # TIER 1 FILTERS (Essential)
    # =========================================================================

    def check_slippage_safety(self, signal, current_price, position_size):
        """
        Ensure slippage won't eat more than 50% of SL buffer

        Returns: (pass, reason)
        """
        # Calculate average recent slippage
        if len(self.recent_slippage) >= 5:
            avg_slippage = np.mean(self.recent_slippage[-20:])  # Last 20 trades
        else:
            # Bootstrap with typical slippage estimate
            avg_slippage = current_price * 0.0005  # 0.05%

        # Calculate TP/SL distances
        if signal == 'LONG':
            tp_distance = current_price * (self.tp_pct / 100)
            sl_distance = current_price * (self.sl_pct / 100)
        else:  # SHORT
            tp_distance = current_price * (self.tp_pct / 100)
            sl_distance = current_price * (self.sl_pct / 100)

        # Safety ratio: (TP - slippage) / SL should be > 2.0
        safety_ratio = (tp_distance - avg_slippage) / sl_distance

        if safety_ratio < self.SLIPPAGE_SAFETY_RATIO:
            return False, f"Safety ratio {safety_ratio:.2f} < {self.SLIPPAGE_SAFETY_RATIO} (slippage ${avg_slippage:.2f})"

        return True, None

    def check_consecutive_losses(self):
        """
        Kill-switch: Stop after N consecutive losses

        Returns: (pass, reason)
        """
        if self.consecutive_losses >= self.MAX_CONSECUTIVE_LOSSES:
            return False, f"{self.consecutive_losses} consecutive losses - AUTO-STOP"

        return True, None

    def check_support_resistance(self, signal, current_price, df):
        """
        Don't LONG near resistance, don't SHORT near support

        Returns: (pass, reason)
        """
        try:
            # Calculate S/R levels from recent swing highs/lows (4-hour window = 48 bars of 5min)
            resistance = df['high'].rolling(48).max().iloc[-1]
            support = df['low'].rolling(48).min().iloc[-1]

            dist_to_resistance = (resistance - current_price) / current_price
            dist_to_support = (current_price - support) / current_price

            if signal == 'LONG' and dist_to_resistance < self.SR_PROXIMITY_PCT:
                return False, f"Too close to resistance ${resistance:,.0f} ({dist_to_resistance*100:.2f}%)"

            if signal == 'SHORT' and dist_to_support < self.SR_PROXIMITY_PCT:
                return False, f"Too close to support ${support:,.0f} ({dist_to_support*100:.2f}%)"

        except Exception as e:
            logger.warning(f"S/R filter error: {e}")
            # Don't block trade on filter error
            pass

        return True, None

    # =========================================================================
    # TIER 2 FILTERS (Valuable)
    # =========================================================================

    def check_time_of_day(self):
        """
        Avoid low-liquidity hours (worse slippage)

        Returns: (pass, reason)
        """
        try:
            et_tz = pytz.timezone('US/Eastern')
            current_time_et = datetime.now(et_tz)
            current_hour = current_time_et.hour

            for start_hour, end_hour in self.LOW_LIQUIDITY_HOURS:
                if start_hour <= current_hour < end_hour:
                    return False, f"Low liquidity hours ({start_hour}am-{end_hour}am ET)"

        except Exception as e:
            logger.warning(f"Time filter error: {e}")
            pass

        return True, None

    def check_volatility_regime(self, df):
        """
        Skip trades during extreme volatility (>3x normal)

        Returns: (pass, reason)
        """
        try:
            # Calculate realized volatility (5-period rolling std of returns)
            returns = df['close'].pct_change()
            current_vol = returns.rolling(48).std().iloc[-1]  # 4-hour window
            avg_vol = returns.rolling(288).std().mean()  # 24-hour average

            if current_vol > avg_vol * self.MAX_VOLATILITY_MULTIPLIER:
                return False, f"High volatility {current_vol*100:.3f}% vs avg {avg_vol*100:.3f}% ({current_vol/avg_vol:.1f}x)"

        except Exception as e:
            logger.warning(f"Volatility filter error: {e}")
            pass

        return True, None

    def check_daily_trade_limit(self):
        """
        Limit trades per day to force selectivity

        Returns: (pass, reason)
        """
        today = datetime.now().date()

        # Remove old trades
        self.daily_trades = [t for t in self.daily_trades if t == today]

        if len(self.daily_trades) >= self.MAX_DAILY_TRADES:
            return False, f"Daily limit reached ({self.MAX_DAILY_TRADES} trades)"

        return True, None

    # =========================================================================
    # TIER 3 FILTERS (Nice-to-Have)
    # =========================================================================

    def check_position_accumulation(self, position_size):
        """
        Cap position size to prevent runaway accumulation

        Returns: (pass, reason)
        """
        if position_size > self.POSITION_SIZE_CAP:
            return False, f"Position size {position_size}x exceeds cap {self.POSITION_SIZE_CAP}x"

        return True, None

    # =========================================================================
    # FILTER STATE UPDATES
    # =========================================================================

    def record_trade_result(self, pnl, slippage=None):
        """
        Update filter state after trade completes

        Args:
            pnl: P&L in dollars
            slippage: Actual slippage in dollars (optional)
        """
        # Update consecutive losses
        if pnl < 0:
            self.consecutive_losses += 1
            logger.info(f"  Consecutive losses: {self.consecutive_losses}")
        else:
            if self.consecutive_losses > 0:
                logger.info(f"  Consecutive loss streak broken at {self.consecutive_losses}")
            self.consecutive_losses = 0

        # Update slippage history
        if slippage is not None:
            self.recent_slippage.append(abs(slippage))
            if len(self.recent_slippage) > 100:
                self.recent_slippage = self.recent_slippage[-100:]  # Keep last 100

    def record_trade_attempt(self):
        """Record that a trade was attempted today"""
        self.daily_trades.append(datetime.now().date())

    def reset_kill_switch(self):
        """Manually reset the consecutive loss kill-switch"""
        logger.warning("KILL-SWITCH MANUALLY RESET")
        self.consecutive_losses = 0

    def get_filter_stats(self):
        """Get current filter state for monitoring"""
        return {
            'consecutive_losses': self.consecutive_losses,
            'daily_trades': len([t for t in self.daily_trades if t == datetime.now().date()]),
            'avg_slippage': np.mean(self.recent_slippage[-20:]) if len(self.recent_slippage) >= 5 else None
        }
