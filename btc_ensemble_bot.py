"""
BTC Ensemble Trading Bot for Micro Bitcoin Futures (MBT).

V2 Models (Dec 20, 2025) - Retrained with random 75/25 split on 2 years data:
- 2h_1.0pct LONG/SHORT: 77-78% WR
- 4h_1.0pct LONG/SHORT: 80-81% WR

Optimized Parameters:
- SL: 0.75%
- Trailing: 0.15% (activation 0.15%)
- Timeout: 2x horizon
- Probability threshold: 65%

Backtest Results (Dec 1-18, 2025):
- 399 trades, 80% win rate
- +105% P&L in 17 days
- Avg hold time: 0.7-1.5 hours

Features:
- Position persistence (survives restarts)
- Orphan position detection
- Multiple model parallel positions
- Real-time data from Binance (BTC trades 24/7)
- IB Gateway for order execution
- WebSocket for real-time trailing stop checks
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import time
import logging
import requests
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, fields
from ib_insync import IB, Future, MarketOrder, util
import ta
from trade_database import log_trade_to_db
from data_validator import DataValidator, log_alert
from parquet_validator import ParquetValidator
from market_hours import MarketHours, get_cme_market_hours

try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    logging.warning("websocket-client not installed. Using REST API for price updates.")

# Setup logging with append mode to preserve logs across restarts
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/btc_bot.log', mode='a'),  # Append mode
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# BTC Ensemble configuration - LONG models (V2: 2h, 4h, 6h horizons)
# Updated Jan 7, 2026: Enabled ALL LONG models - overlap analysis showed only 25-50% overlap
# Each model catches unique opportunities. Combined P&L: +$513 vs $319 with 6h only
# Using SL=0.70%, TP=1.40% from backtest table (best results)
BTC_ENSEMBLE_MODELS_LONG = [
    {'horizon': '2h', 'threshold': 0.5, 'horizon_bars': 24, 'priority': 3, 'direction': 'LONG'},
    {'horizon': '4h', 'threshold': 0.5, 'horizon_bars': 48, 'priority': 2, 'direction': 'LONG'},
    {'horizon': '6h', 'threshold': 0.5, 'horizon_bars': 72, 'priority': 1, 'direction': 'LONG'},
]

# BTC Ensemble configuration - SHORT models (V2: 2h, 4h horizons)
# Updated Jan 7, 2026: Enabled ALL SHORT models - 2h adds unique signals (43% overlap with 4h)
# Using SL=0.70%, TP=1.40% from backtest table
BTC_ENSEMBLE_MODELS_SHORT = [
    {'horizon': '2h', 'threshold': 0.5, 'horizon_bars': 24, 'priority': 2, 'direction': 'SHORT'},
    {'horizon': '4h', 'threshold': 0.5, 'horizon_bars': 48, 'priority': 1, 'direction': 'SHORT'},
]

# Combined models
BTC_ENSEMBLE_MODELS = BTC_ENSEMBLE_MODELS_LONG + BTC_ENSEMBLE_MODELS_SHORT

# Timeout multiplier: 2x the model horizon for better win rates
# Analysis showed 2x timeout improves win rates from ~50% to ~85%
TIMEOUT_MULTIPLIER = 2.0

# Trailing stop configuration (RE-ENABLED Jan 13, 2026)
# Jan 5: Disabled - tight 0.05% trailing was cutting winners short
# Jan 13: Re-enabled with wider settings after analysis showed 3/6 SL trades
#         went profitable (+0.42% to +0.45%) then reversed to -0.70% SL
#         New config would have saved ~$271 today alone
# Activation: +0.30% (only activate when trade is solidly profitable)
# Trail: 0.10% (Feb 27: tightened from 0.20% - tick sim shows +$16.73 on 49 trades, 0 whipsaws)
# Most TS wins barely cross 0.30% activation, so 0.10% trail locks in ~0.20% vs ~0.10% minimum
TRAILING_STOP_PCT = 0.25  # Mar 13 v2: widened from 0.15% — wider trail gives room to run toward 1.0% TP without premature exit on minor pullbacks
TRAILING_STOP_ACTIVATION_PCT = 0.50  # Mar 13 v2: raised from 0.35% — only lock in once clearly profitable; pairs with new TP=1.0% target

# Stop loss percentage (updated Jan 22, 2026)
# Using SL=0.70%, TP=1.00% - adjusted Jan 12, 2026 based on live trading analysis:
# Original TP=1.40% was hit only 17% of time, many trades went profitable but reversed
# New TP=3.00% gives 4.3:1 R:R - trailing stop handles most exits anyway
# Jan 15, 2026: Raised TP from 1% to 3% to let trailing stop do the work
# Jan 20, 2026: Reduced SL from 0.70% to 0.60% based on MAE analysis
# Jan 22, 2026: Reduced SL from 0.60% to 0.40% - analysis shows 78.6% of SL trades never went positive
# Feb 17, 2026: Reverted to 0.40% - 0.20% too tight, hit on 75% of trades
STOP_LOSS_PCT = 0.50  # Mar 13 v2: widened from 0.30% — analysis: SL=0.50%/TP=1.0% at 4h EV=+0.138% vs SL=0.30%/TP=0.50% EV=+0.065%. R:R = 2:1
TAKE_PROFIT_PCT = 1.00  # Mar 13 v2: reduced from 3.0% (unreachable) to 1.0% — data shows 57.5% of bars hit +1% within 4h; hard TP now acts as real exit target

# =============================================================================
# SL COOLDOWN (Feb 19, 2026)
# =============================================================================
# After a SL exit, block new entries in the same direction for N bars.
# Prevents overtrading in choppy ranges where the bot enters, gets stopped,
# and immediately re-enters at nearly the same price.
# Feb 19 analysis: 8 trades in 3 hours, 5 SL losses, instant re-entries.
SL_COOLDOWN_BARS = 6  # 6 bars = 30 minutes cooldown after SL
SL_COOLDOWN_SECONDS = SL_COOLDOWN_BARS * 5 * 60  # Convert to seconds (1800s = 30 min)

# =============================================================================
# PULLBACK ENTRY with DIP-THEN-RECOVER (Feb 23, 2026; Feb 27: added bounce)
# =============================================================================
# Instead of entering immediately on signal, wait for a pullback + micro-bounce.
# 1. Wait for price to pull back PULLBACK_PCT (0.20%) from signal price
# 2. Once dip level hit, track the local minimum (dip low)
# 3. Only enter when price bounces PULLBACK_BOUNCE_PCT (0.05%) from the dip low
# This avoids entering into falling knives where price dips and keeps falling.
# Feb 27 simulation: +$342 improvement over blind pullback on 44 post-filter trades.
# If 30 min expires without a bounce, skip the trade (no market fallback).
PULLBACK_ENABLED = False  # Mar 2: disabled - 3 trades w/ bounce logic had 33% WR, missed 3 winners via expired signals
PULLBACK_PCT = 0.20        # 0.20% pullback from signal price
PULLBACK_BOUNCE_PCT = 0.05 # 0.05% bounce from dip low required before entry
PULLBACK_WAIT_BARS = 6     # Wait up to 6 bars (30 min) for pullback + bounce
PULLBACK_MARKET_FALLBACK = False  # Feb 27: disabled - if no bounce, skip (falling knife)

# =============================================================================
# INDICATOR-BASED LONG STRATEGY #1: MOMENTUM (Jan 5, 2026)
# =============================================================================
# BB %B > 0.5 + MACD > Signal (momentum - buy when strong)
# 30-day 15-sec backtest: 317 trades, 37.5% WR, -$172/week (NOT PROFITABLE)
# DISABLED Jan 6, 2026: Live trading confirmed losses (-$92.88 in 20 trades, 10% WR)
INDICATOR_LONG_ENABLED = False
INDICATOR_LONG_SL_PCT = 0.30
INDICATOR_LONG_TP_PCT = 0.50
INDICATOR_LONG_TIMEOUT_BARS = 48
MAX_INDICATOR_LONG_POSITIONS = 1  # Reduced to 1 since adding second strategy

# =============================================================================
# INDICATOR-BASED LONG STRATEGY #2: MEAN REVERSION (Jan 5, 2026)
# =============================================================================
# BB %B < 0.5 (mean reversion - buy when oversold, same logic as MNQ)
# 30-day 15-sec backtest: 342 trades, 44.2% WR, +$184/week (PROFITABLE)
# DISABLED Jan 6, 2026: Replaced by ROC+MACD trend strategy
INDICATOR_MEANREV_ENABLED = False
INDICATOR_MEANREV_SL_PCT = 0.30
INDICATOR_MEANREV_TP_PCT = 0.50
INDICATOR_MEANREV_TIMEOUT_BARS = 48
MAX_INDICATOR_MEANREV_POSITIONS = 1

# =============================================================================
# INDICATOR-BASED STRATEGY #3: ROC + MACD TREND (Jan 6, 2026)
# =============================================================================
# Entry: ROC(12) > 0.4% AND MACD_Hist > 0 AND MACD_Hist increasing (LONG)
#        ROC(12) < -0.4% AND MACD_Hist < 0 AND MACD_Hist decreasing (SHORT)
# 30-day 15-sec backtest: 145 trades, 49% WR, +$437/week
# SL: 0.70%, TP: 1.00% (R:R = 1.43:1) - adjusted Jan 12, 2026
# Max hold: 48 bars (4 hours), Cooldown: 20 bars (~1.5 hours)
INDICATOR_TREND_ENABLED = False
INDICATOR_TREND_SL_PCT = 0.70
INDICATOR_TREND_TP_PCT = 1.00
INDICATOR_TREND_TIMEOUT_BARS = 48
INDICATOR_TREND_COOLDOWN_BARS = 20  # ~1.5 hours between trades
INDICATOR_TREND_ROC_THRESHOLD = 0.4  # ROC must exceed this %
MAX_INDICATOR_TREND_POSITIONS = 1

# Position limits per direction (prevents one direction from blocking the other)
# Analysis showed 90% of LONG signals were blocked because SHORT positions filled all slots
MAX_LONG_POSITIONS = 2  # ML V2 LONG positions (currently disabled via high threshold)
MAX_SHORT_POSITIONS = 2  # ML V2 SHORT positions

# =============================================================================
# TREND ALIGNMENT FILTER (Jan 13, 2026)
# =============================================================================
# Analysis of 74 trades showed SHORT trades against rising trend had 0% win rate
# Filter: Skip SHORT entries if price rose >0.4% in last hour (12 bars)
# This would have saved $573-683 with 0 winning trades filtered out
# Jan 21, 2026: Added LONG filter after 14 SL losses in downtrend (-$885)
TREND_FILTER_ENABLED = True
TREND_FILTER_SHORT_THRESHOLD = 0.4  # Skip SHORT if 1-hour trend > +0.4%
TREND_FILTER_LONG_THRESHOLD = -0.4  # Skip LONG if 1-hour trend < -0.4%
TREND_FILTER_LOOKBACK_BARS = 12  # 1 hour of 5-min bars

# =============================================================================
# RSI FILTER (Feb 19, 2026) - Re-enabled with corrected thresholds
# =============================================================================
# Feb analysis (18 LONG trades): RSI < 40 = 0W/8L (0% WR), RSI >= 40 = 8W/2L (80% WR)
# Feb analysis (24 SHORT trades): RSI > 70 = 0W/2L (0% WR)
RSI_FILTER_ENABLED = True
RSI_FILTER_LONG_MIN = 50   # Skip LONG if RSI < 50 (Feb 27: tightened from 42 - blocks 2W/6L, net +$126, RSI<50 = buying into weakness)
RSI_FILTER_SHORT_MAX = 70  # Skip SHORT if RSI > 70 (uptrend, not a reversal)
RSI_FILTER_SHORT_MIN = 40  # Mar 17: Skip SHORT if RSI < 40 (oversold, mean-reversion risk — sim: blocks 1W/2L, +$14)

# =============================================================================
# BB% FILTER (Feb 19, 2026) - Re-enabled with corrected thresholds
# =============================================================================
# Feb analysis (18 LONG): BB% < 0.20 = 0W/7L (0% WR), BB% >= 0.60 = 6W/0L (100% WR)
# Feb analysis (24 SHORT): BB% > 0.80 = 0W/2L (0% WR), BB% <= 0.10 = 3W/0L (100% WR)
BB_FILTER_ENABLED = True
BB_FILTER_LONG_MIN = 0.40   # Mar 2: tightened from 0.25 - BB%<0.40 LONGs: 8% WR (1W/11L), -$411 net
BB_FILTER_SHORT_MAX = 0.80  # Skip SHORT if BB% > 0.80 (price at upper band = uptrend)

# =============================================================================
# MACD FILTER (Feb 19, 2026) - New
# =============================================================================
# Feb analysis (18 LONG): MACD < -10 = 0W/6L (0% WR), MACD >= 0 = 5W/1L (83% WR)
# Feb analysis (24 SHORT): MACD > 10 = 3W/5L (38% WR, negative avg P&L)
MACD_FILTER_ENABLED = True
MACD_FILTER_LONG_MIN = 0     # Feb 26: Tightened from -10 to 0. MACD<0 LONGs: 15% WR, -$389 net vs MACD>=0: 59% WR, +$84 net
MACD_FILTER_SHORT_MAX = 10   # Skip SHORT if MACD > 10 (bullish momentum)

# =============================================================================
# MACRO TREND FILTER (Feb 24, 2026)
# =============================================================================
# Block entries when BTC has a persistent directional move over 24 hours.
# Feb 22-24 selloff ($68,100 → $63,200, -3%): all 12 losing LONG entries
# had 24h change < -2%. Only 2 wins blocked ($51 total). Net +$422.
# Precision: 80% (12L/3W blocked). Stays active through bounces unlike 8h.
MACRO_TREND_FILTER_ENABLED = True
MACRO_TREND_LONG_MIN = -2.0   # Skip LONG if 24h price change < -2.0%
MACRO_TREND_SHORT_MAX = 1.0   # Mar 13: tightened from 2.0% — during Mar 8-12 +6.4% rally, SHORT signals kept firing when 24h dipped <2% early AM. 1% blocks more conservatively
MACRO_TREND_LOOKBACK_BARS = 288  # 24h of 5-min bars (24 * 60 / 5)

# =============================================================================
# TREND-FOLLOWING MODE (Mar 12, 2026)
# =============================================================================
# In strong trends (+4%+), switch from mean-reversion to momentum strategy
# =============================================================================
# CROSS-TIMEFRAME FILTER (Mar 17, 2026)
# =============================================================================
# 6h LONG must have at least one shorter model (2h or 4h) above this threshold.
# Simulation: blocks 0W/1L, saves $39.93 (prevents isolated 6h entries with no support)
CROSS_TF_ENABLED = True
CROSS_TF_6H_LONG_MIN = 0.40  # 2h or 4h LONG prob must be >= 40% for 6h LONG to enter

TREND_FOLLOW_ENABLED = True
TREND_FOLLOW_LONG_THRESHOLD = 4.0   # Activate if 24h change > +4%
TREND_FOLLOW_SHORT_THRESHOLD = -4.0 # Activate if 24h change < -4%
TREND_FOLLOW_LONG_MODEL_THRESHOLD = 0.30  # Lower threshold for LONG entries
TREND_FOLLOW_SHORT_MODEL_THRESHOLD = 0.30 # Lower threshold for SHORT entries
TREND_FOLLOW_REQUIRE_MACD = True  # Require MACD > signal for trend mode

# =============================================================================
# ENTRY GATE MODEL (Mar 9, 2026)
# =============================================================================
# GradientBoosting classifier trained on 352 trades (Jan-Mar 2026).
# Uses RSI, BB%, MACD, ATR%, volatility, hour, day_of_week, probability.
# Walk-forward backtest (70/30 split):
#   LONG: 53% WR → 60% WR at >50% threshold, -$188 → +$131
#   SHORT: 54% WR → 64% WR at >50% threshold, +$227 → +$387
# Adds non-linear interaction effects on top of manual filters.
ENTRY_GATE_ENABLED = True
ENTRY_GATE_THRESHOLD = 0.50  # Only enter if gate model probability >= this
ENTRY_GATE_LONG_MODEL = 'models/entry_filter_long.pkl'
ENTRY_GATE_SHORT_MODEL = 'models/entry_filter_short.pkl'

# IB Configuration
IB_HOST = '127.0.0.1'
IB_PORT = 4002  # IB Gateway
IB_CLIENT_ID = 400  # stable ID

# BTC Configuration
BTC_CONTRACT_VALUE = 0.1  # MBT = 0.1 BTC
BINANCE_API = "https://api.binance.com/api/v3"

# Signal logging configuration
SIGNAL_LOG_DIR = "signal_logs"
SIGNAL_LOG_INTERVAL = 15  # seconds

# Parquet configuration (like MNQ/SPY bots)
PARQUET_PATH = "data/BTC_features.parquet"
PARQUET_REFRESH_INTERVAL = 300  # Refresh parquet every 5 minutes for better signal accuracy


class SignalLogger:
    """Logs all model signals to CSV for audit purposes."""
    
    def __init__(self, log_dir: str = SIGNAL_LOG_DIR):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.current_date = None
        self.csv_path = None
        self._init_csv()
    
    def _init_csv(self) -> None:
        """Initialize CSV file for current date."""
        today = datetime.now().strftime('%Y-%m-%d')
        if today != self.current_date:
            self.current_date = today
            self.csv_path = os.path.join(self.log_dir, f"btc_signals_{today}.csv")
            
            # Write header if file doesn't exist
            if not os.path.exists(self.csv_path):
                header = [
                    'timestamp', 'btc_price',
                    # LONG models (V2: 2h, 4h, 6h)
                    'prob_2h_0.5pct', 'prob_4h_0.5pct', 'prob_6h_0.5pct',
                    'signal_2h_0.5pct', 'signal_4h_0.5pct', 'signal_6h_0.5pct',
                    # SHORT models (V2: 2h + 4h)
                    'prob_2h_0.5pct_SHORT', 'prob_4h_0.5pct_SHORT',
                    'signal_2h_0.5pct_SHORT', 'signal_4h_0.5pct_SHORT',
                    'active_positions', 'rsi', 'macd', 'atr', 'bb_pct_b'
                ]
                with open(self.csv_path, 'w') as f:
                    f.write(','.join(header) + '\n')
                logger.info(f"Created signal log: {self.csv_path}")
    
    def log_signals(self, timestamp: datetime, price: float, 
                    probabilities: Dict[str, float], threshold: float,
                    active_positions: int, indicators: Dict[str, float] = None) -> None:
        """Log signals to CSV."""
        self._init_csv()  # Check if we need a new file for new day
        
        # Model order must match header - V2 LONG models (2h, 4h, 6h)
        long_models = ['2h_0.5pct', '4h_0.5pct', '6h_0.5pct']
        # V2 SHORT models (2h + 4h)
        short_models = ['2h_0.5pct_SHORT', '4h_0.5pct_SHORT']
        
        long_probs = [f"{probabilities.get(m, 0):.4f}" for m in long_models]
        long_signals = ['1' if probabilities.get(m, 0) >= threshold else '0' for m in long_models]
        
        short_probs = [f"{probabilities.get(m, 0):.4f}" for m in short_models]
        short_signals = ['1' if probabilities.get(m, 0) >= threshold else '0' for m in short_models]
        
        indicators = indicators or {}
        rsi = f"{indicators.get('rsi', 0):.2f}"
        macd = f"{indicators.get('macd', 0):.4f}"
        atr = f"{indicators.get('atr', 0):.2f}"
        bb_pct_b = f"{indicators.get('bb_pct_b', 0.5):.4f}"
        
        row = [
            timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            f"{price:.2f}",
            *long_probs,
            *long_signals,
            *short_probs,
            *short_signals,
            str(active_positions),
            rsi, macd, atr, bb_pct_b
        ]
        
        with open(self.csv_path, 'a') as f:
            f.write(','.join(row) + '\n')


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    direction: str
    size: int
    entry_price: float
    entry_time: str
    model_horizon: str
    model_threshold: float
    target_bars: int
    target_price: float = 0.0  # Take profit price
    stop_price: float = 0.0    # Stop loss price
    bars_held: int = 0
    model_id: str = ""
    order_id: Optional[int] = None
    pending_close: bool = False  # Track if close order is pending
    pending_close_time: str = ""  # When close order was placed
    # Trailing stop fields
    peak_price: float = 0.0  # Highest price since entry (for LONG)
    trough_price: float = 0.0  # Lowest price since entry (for SHORT)
    trailing_stop_active: bool = False  # Whether trailing stop is activated
    trailing_stop_price: float = 0.0  # Current trailing stop level
    # Entry features for learning loop (Jan 14, 2026)
    entry_features: Optional[dict] = None  # RSI, MACD, BB, etc. at entry time
    entry_probability: float = 0.0  # Model probability at entry
    entry_trend_1h: float = 0.0  # 1-hour trend at entry
    entry_macro_trend_24h: float = 0.0  # 24h macro trend at entry
    entry_all_probs: Optional[dict] = None  # All model probabilities at entry

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'Position':
        # Filter to only known fields (handles old JSON without new fields)
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


class PositionManager:
    """Manages multiple parallel positions (one per model)."""

    def __init__(self, filepath: str = "btc_positions.json"):
        self.filepath = filepath
        self.positions: Dict[str, Position] = {}
        self.load_positions()

    def load_positions(self) -> None:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                    for pos_id, pos_data in data.items():
                        self.positions[pos_id] = Position.from_dict(pos_data)
                logger.info(f"Loaded {len(self.positions)} positions")
                # Log position details for debugging
                for pos_id, pos in self.positions.items():
                    logger.info(f"  - {pos_id}: {pos.direction} @ ${pos.entry_price:.2f} "
                               f"(stop: ${pos.stop_price:.2f}, trailing: {pos.trailing_stop_active})")
            except Exception as e:
                logger.error(f"Error loading positions: {e}")
                self.positions = {}

    def save_positions(self) -> None:
        try:
            data = {pos_id: pos.to_dict() for pos_id, pos in self.positions.items()}
            with open(self.filepath, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving positions: {e}")
    
    def backup_positions(self) -> str:
        """Create a timestamped backup of positions file.
        
        Returns the backup filepath. Use this before manual edits or restarts.
        """
        if not os.path.exists(self.filepath):
            return ""
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = f"{self.filepath}.backup_{timestamp}"
            import shutil
            shutil.copy2(self.filepath, backup_path)
            logger.info(f"Created positions backup: {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"Error creating positions backup: {e}")
            return ""

    def add_position(self, position: Position) -> None:
        self.positions[position.model_id] = position
        self.save_positions()
        logger.info(f"Added position [{position.model_id}]: {position.direction} "
                   f"{position.size} @ ${position.entry_price:.2f}")

    def remove_position(self, model_id: str) -> Optional[Position]:
        if model_id in self.positions:
            position = self.positions.pop(model_id)
            self.save_positions()
            logger.info(f"Removed position [{model_id}]")
            return position
        return None

    def get_position(self, model_id: str) -> Optional[Position]:
        return self.positions.get(model_id)

    def has_position(self, model_id: str) -> bool:
        return model_id in self.positions
    
    def get_all_positions(self) -> List[Position]:
        return list(self.positions.values())
    
    def count_positions(self) -> int:
        return len(self.positions)
    
    def count_positions_by_direction(self, direction: str) -> int:
        """Count positions for a specific direction (LONG or SHORT)."""
        return sum(1 for pos in self.positions.values() if pos.direction == direction)
    
    def increment_bars_held(self) -> None:
        """Increment bars_held for all positions."""
        for pos in self.positions.values():
            pos.bars_held += 1
        self.save_positions()


def get_btc_front_month_expiry() -> str:
    """Get the front month BTC futures expiry in YYYYMM format.
    
    MBT (Micro Bitcoin) expiries are on last Friday of each month.
    Roll to next month 5 days before expiry to avoid expiry week issues.
    """
    now = datetime.now()
    year = now.year
    month = now.month
    
    # Check if we're within 5 days of last Friday of current month
    import calendar
    from datetime import timedelta
    c = calendar.Calendar(firstweekday=calendar.SUNDAY)
    monthcal = c.monthdatescalendar(year, month)
    
    # Find last Friday
    fridays = [day for week in monthcal for day in week 
               if day.weekday() == calendar.FRIDAY and day.month == month]
    last_friday = fridays[-1]
    
    # Roll 5 days before expiry (or if past expiry)
    roll_date = last_friday - timedelta(days=5)
    if now.date() >= roll_date:
        # Roll to next month
        if month == 12:
            month = 1
            year += 1
        else:
            month += 1
    
    return f"{year}{month:02d}"


def is_cme_mbt_open() -> bool:
    """Check if CME MBT futures market is open.
    
    Uses MarketHours module which handles:
    - Regular trading hours (Sunday 5 PM - Friday 4 PM CT)
    - Daily maintenance break (4-5 PM CT Mon-Thu)
    - CME holidays (full closures and early closes)
    - Weekend closures
    
    Returns True if market is open, False if closed.
    """
    market_hours = get_cme_market_hours()
    is_open, reason = market_hours.is_market_open()
    if not is_open:
        logger.debug(f"CME MBT closed: {reason}")
    return is_open


def should_close_positions_before_market_close() -> Tuple[bool, str]:
    """Check if positions should be closed due to upcoming market closure.
    
    Returns True if within PRE_CLOSE_BUFFER_MINUTES of market close.
    This prevents positions from being stuck over weekends/holidays.
    """
    market_hours = get_cme_market_hours()
    return market_hours.should_close_positions()


def should_block_new_entries() -> Tuple[bool, str]:
    """Check if new entries should be blocked due to insufficient time before close.
    
    Blocks entries if there isn't enough time to hold a position before market closes.
    Uses minimum 60 minutes holding time requirement.
    """
    market_hours = get_cme_market_hours()
    return market_hours.should_block_new_entries(min_holding_time_minutes=60)


class BTCEnsembleBot:
    """BTC Ensemble trading bot using multiple ML models."""

    def __init__(self,
                 probability_threshold_long: float = 0.55,
                 probability_threshold_short: float = 0.55,
                 position_size: int = 1,
                 paper_trading: bool = True,
                 signal_check_interval: int = 60,  # seconds
                 bar_interval: int = 300,  # 5 min bars
                 max_long_positions: int = 2,
                 max_short_positions: int = 2,
                 stop_loss_pct: float = 1.0):  # Stop loss percentage
        
        self.probability_threshold_long = probability_threshold_long
        self.probability_threshold_short = probability_threshold_short
        self.position_size = position_size
        self.paper_trading = paper_trading
        self.signal_check_interval = signal_check_interval
        self.bar_interval = bar_interval
        self.max_long_positions = max_long_positions
        self.max_short_positions = max_short_positions
        self.max_positions = max_long_positions + max_short_positions  # Total for logging
        self.stop_loss_pct = stop_loss_pct

        # IB connection
        self.ib = IB()
        self.connected = False

        # Position manager
        self.position_manager = PositionManager()

        # Load models
        self.models = self._load_models()
        self.feature_cols = self._load_feature_columns()

        # Contract setup
        expiry = get_btc_front_month_expiry()
        self.contract = Future('MBT', expiry, 'CME')
        self.symbol = 'MBT'
        
        # Data cache
        self.data_cache = pd.DataFrame()
        self.last_bar_time = None
        self.current_price = None
        
        # Stats
        self.start_time = datetime.now()
        self.total_checks = 0
        self.total_signals = 0
        
        # Signal logging
        self.signal_logger = SignalLogger()
        self.last_signal_log_time = None
        self.cached_features_df = None  # Cache features for signal logging
        
        # Parquet features (like MNQ/SPY bots - no cumulative offset drift)
        self.parquet_features = None
        self.parquet_last_time = None
        self.parquet_last_refresh = None
        self.use_parquet_features = True
        self._load_parquet_features()
        
        # Synthetic bar for real-time updates
        self.synthetic_bar = None
        
        # Track pending close orders to avoid duplicates
        self.pending_close_orders: Dict[str, datetime] = {}  # model_id -> order_time
        
        # Track last SL exit time per direction for cooldown (Feb 19, 2026)
        # Fixed Feb 23: was bar-based but len(cached_features_df) doesn't increment per bar
        self.last_sl_time: Dict[str, datetime] = {'LONG': datetime.min, 'SHORT': datetime.min}
        
        # Pending pullback entries (Feb 23, 2026)
        # When a signal passes all filters, instead of entering immediately,
        # we wait for a 0.20% pullback. If not filled in 6 bars, enter at market.
        # Dict: model_id -> {signal, signal_price, limit_price, signal_time, bars_waited}
        self.pending_pullbacks: Dict[str, Dict] = {}
        
        # Data validator for quality checks
        self.data_validator = DataValidator(model_dir="models_btc_v2")
        self.data_validator.add_alert_callback(log_alert)
        self.validation_failures = 0
        self.last_validation_time = None
        
        # WebSocket for real-time price updates
        self.ws = None
        self.ws_thread = None
        self.ws_price = None  # Latest price from WebSocket
        self.ws_price_time = None  # Time of last WebSocket price update
        self.ws_connected = False
        
        # IB portfolio price fallback (Feb 26: Binance WS+REST can both fail,
        # but IB updatePortfolio callbacks still provide MBT marketPrice)
        self.ib_mbt_price = None
        self.ib_mbt_price_time = None
        
        # Last known price for emergency stop-loss only mode
        # When ALL price sources fail, use last known price for SL checks only
        self.last_price_time = None  # When current_price was last updated
        self.last_loop_time = None  # Watchdog: detect frozen main loop
        
        # Entry gate models (Mar 9, 2026)
        self.entry_gate_long = None
        self.entry_gate_short = None
        if ENTRY_GATE_ENABLED:
            try:
                self.entry_gate_long = joblib.load(os.path.join(os.path.dirname(__file__), ENTRY_GATE_LONG_MODEL))
                self.entry_gate_short = joblib.load(os.path.join(os.path.dirname(__file__), ENTRY_GATE_SHORT_MODEL))
                logger.info(f"Entry gate models loaded (threshold={ENTRY_GATE_THRESHOLD})")
            except Exception as e:
                logger.warning(f"Entry gate models failed to load: {e} — gate disabled")
        
        # Tick logger: write every 2-sec price check to CSV for replay/simulation
        tick_log_path = os.path.join(os.path.dirname(__file__), 'logs', 'btc_price_ticks.csv')
        tick_file_exists = os.path.exists(tick_log_path)
        self.tick_log_file = open(tick_log_path, 'a', buffering=1)  # line-buffered
        if not tick_file_exists:
            self.tick_log_file.write('timestamp,price,source\n')

        logger.info("="*60)
        logger.info("BTC ENSEMBLE TRADING BOT INITIALIZED")
        logger.info("="*60)
        logger.info(f"Contract: MBT {expiry} (Micro Bitcoin Futures)")
        logger.info(f"Models: {len(self.models)} ({len(BTC_ENSEMBLE_MODELS_LONG)} LONG, {len(BTC_ENSEMBLE_MODELS_SHORT)} SHORT)")
        for m in BTC_ENSEMBLE_MODELS:
            direction = m.get('direction', 'LONG')
            logger.info(f"  - {m['horizon']}_{m['threshold']}% {direction} (priority {m['priority']})")
        logger.info(f"Probability threshold: LONG={probability_threshold_long}, SHORT={probability_threshold_short}")
        logger.info(f"Position size: {position_size} contracts")
        logger.info(f"Stop loss: Dynamic (max of 1% or target%)")
        logger.info(f"Max positions: {max_long_positions} LONG, {max_short_positions} SHORT")
        logger.info(f"Paper trading: {paper_trading}")

    def _load_models(self) -> Dict[str, any]:
        """Load all ensemble models (LONG and SHORT)."""
        models = {}
        for m in BTC_ENSEMBLE_MODELS:
            direction = m.get('direction', 'LONG')
            suffix = '_SHORT' if direction == 'SHORT' else ''
            model_name = f"{m['horizon']}_{m['threshold']}pct{suffix}"
            model_path = f"models_btc_v2/model_{model_name}.joblib"
            if os.path.exists(model_path):
                models[model_name] = {
                    'model': joblib.load(model_path),
                    'config': m
                }
                logger.info(f"Loaded model: {model_name} ({direction})")
            else:
                logger.warning(f"Model not found: {model_path}")
        return models

    def _load_feature_columns(self) -> List[str]:
        """Load feature column names."""
        feature_path = "models_btc_v2/feature_columns.json"
        if os.path.exists(feature_path):
            with open(feature_path, 'r') as f:
                return json.load(f)
        return []

    def _load_parquet_features(self) -> bool:
        """Load parquet features (like MNQ/SPY bots).
        
        This approach uses pre-calculated features from parquet which have
        correct cumulative feature values accumulated over years of data.
        No offset drift issues like the previous Binance-only approach.
        """
        try:
            if not os.path.exists(PARQUET_PATH):
                logger.warning(f"Parquet file not found: {PARQUET_PATH}")
                self.use_parquet_features = False
                return False
            
            # Load parquet
            df = pd.read_parquet(PARQUET_PATH)
            
            # Store last 1000 bars for context (enough for all rolling calculations)
            self.parquet_features = df.tail(1000).copy()
            self.parquet_last_time = df.index[-1]
            self.parquet_last_refresh = datetime.now()
            
            logger.info(f"Loaded parquet features: {len(self.parquet_features)} bars, "
                       f"ending at {self.parquet_last_time}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading parquet features: {e}")
            self.use_parquet_features = False
            return False

    def _refresh_parquet_if_needed(self) -> None:
        """Refresh parquet features periodically by fetching new data from Binance."""
        if not self.use_parquet_features:
            return
        
        if self.parquet_last_refresh is None:
            self._refresh_parquet_from_binance()
            return
        
        elapsed = (datetime.now() - self.parquet_last_refresh).total_seconds()
        if elapsed >= PARQUET_REFRESH_INTERVAL:
            logger.info("Refreshing parquet features from Binance...")
            self._refresh_parquet_from_binance()
    
    def _refresh_parquet_from_binance(self) -> bool:
        """Fetch latest BTC data from Binance and APPEND to existing parquet.
        
        Like MNQ bot - preserves historical data for backtesting past signals.
        Only fetches new bars since last parquet update.
        On first run or if parquet is too small, fetches 2000+ bars for stable indicators.
        """
        try:
            import ta
            from feature_engineering import (
                add_time_features, add_price_features,
                add_daily_context_features, add_lagged_indicator_features,
                add_indicator_changes
            )
            
            logger.info("=" * 60)
            logger.info("PARQUET REFRESH: Fetching BTC data from Binance")
            logger.info("=" * 60)
            
            # Load existing parquet if it exists
            existing_df = None
            last_time = None
            if os.path.exists(PARQUET_PATH):
                try:
                    existing_df = pd.read_parquet(PARQUET_PATH)
                    last_time = existing_df.index[-1]
                    logger.info(f"Existing parquet: {len(existing_df)} bars, ends at {last_time}")
                except Exception as e:
                    logger.warning(f"Could not load existing parquet: {e}")
            
            # If live parquet is missing/small, try loading from archive
            archive_path = os.path.join(os.path.dirname(PARQUET_PATH), 'archive', 'BTC_features_archive.parquet')
            if (existing_df is None or len(existing_df) < 2000) and os.path.exists(archive_path):
                try:
                    archive_df = pd.read_parquet(archive_path)
                    if existing_df is not None:
                        combined = pd.concat([archive_df, existing_df])
                        combined = combined[~combined.index.duplicated(keep='last')]
                        existing_df = combined.sort_index()
                    else:
                        existing_df = archive_df
                    last_time = existing_df.index[-1]
                    logger.info(f"Loaded archive: {len(existing_df)} bars, ends at {last_time}")
                except Exception as e:
                    logger.warning(f"Could not load archive: {e}")
            
            # Fetch recent data from Binance
            # If we have existing data, just get last 1000 bars to find new ones
            # If no existing data or too small, get 2000 bars for stable indicators
            url = f"{BINANCE_API}/klines"
            min_bars_needed = 2000  # Minimum for stable cumulative indicators
            
            if existing_df is not None and len(existing_df) >= min_bars_needed:
                # Just fetch recent bars to append
                params = {'symbol': 'BTCUSDT', 'interval': '5m', 'limit': 1000}
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                klines = response.json()
                logger.info(f"Fetched {len(klines)} recent bars from Binance")
            else:
                # Need full history - fetch 2000+ bars
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
                    
                    all_klines = batch + all_klines  # prepend older data
                    end_time = batch[0][0] - 1
                    
                    if len(batch) < 1000:
                        break
                
                klines = all_klines[-target_bars:] if len(all_klines) > target_bars else all_klines
                logger.info(f"Fetched {len(klines)} bars from Binance (full history for stable indicators)")
            
            if not klines:
                logger.warning("No data from Binance")
                return False
            
            # Convert to DataFrame
            new_df = pd.DataFrame(klines, columns=[
                'open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            new_df['timestamp'] = pd.to_datetime(new_df['open_time'], unit='ms')
            new_df = new_df.set_index('timestamp')
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                new_df[col] = new_df[col].astype(float)
            new_df = new_df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Find new bars to append OR replace if existing is too small
            if existing_df is not None and last_time is not None and len(existing_df) >= min_bars_needed:
                # Existing parquet is large enough - append new bars AND update recent bars
                # Get bars from last_time onwards (including last_time to update its final values)
                new_bars = new_df[new_df.index >= last_time]
                
                if len(new_bars) == 0:
                    logger.info(f"PARQUET REFRESH: Already up to date (ends at {last_time})")
                    return True
                
                # Check if we only have the same bar (no new bars)
                if len(new_bars) == 1 and new_bars.index[0] == last_time:
                    # Just update the last bar's values
                    logger.info(f"Updating last bar values at {last_time}")
                else:
                    logger.info(f"Found {len(new_bars) - 1} new bars to append (plus updating last bar)")
                
                # Combine: keep old bars except recent ones, then add fresh recent bars
                # This ensures the last bar gets updated with final OHLCV values
                existing_ohlcv = existing_df[['Open', 'High', 'Low', 'Close', 'Volume']]
                existing_ohlcv = existing_ohlcv[existing_ohlcv.index < last_time]  # Exclude last bar
                combined_ohlcv = pd.concat([existing_ohlcv, new_bars])
                combined_ohlcv = combined_ohlcv[~combined_ohlcv.index.duplicated(keep='last')]
                combined_ohlcv = combined_ohlcv.sort_index()
            else:
                # No existing data OR existing is too small - use all fetched bars
                combined_ohlcv = new_df
                logger.info(f"Creating/replacing parquet with {len(combined_ohlcv)} bars (need {min_bars_needed} for stable indicators)")
            
            # Recalculate ALL features on combined data
            df = ta.add_all_ta_features(
                combined_ohlcv.copy(), open='Open', high='High', low='Low', 
                close='Close', volume='Volume', fillna=True
            )
            
            # Add custom features
            df = add_time_features(df)
            df = add_price_features(df)
            df = add_daily_context_features(df)
            df = add_lagged_indicator_features(df, lookback_periods=[1, 2, 3, 5, 10, 20, 50])
            df = add_indicator_changes(df)
            
            # Clean up
            df = df.ffill().bfill().fillna(0)
            df = df.replace([np.inf, -np.inf], 0)
            
            # Validate before saving
            validator = ParquetValidator(bot_type='btc')
            is_valid, issues = validator.validate_before_save(df)
            
            if not is_valid:
                df = validator.repair(df, issues)
                is_valid, issues = validator.validate_before_save(df)
                
                if not is_valid:
                    critical = [i for i in issues if i['severity'] == 'critical']
                    logger.error(f"Parquet validation failed: {len(critical)} critical issues")
                    for issue in critical[:3]:
                        logger.error(f"  - {issue['message']}")
                    return self._load_parquet_features()
            
            # Save validated parquet (preserves all historical data)
            df.to_parquet(PARQUET_PATH)
            
            # Store recent bars in memory for trading
            self.parquet_features = df.tail(1000).copy()
            self.parquet_last_time = df.index[-1]
            self.parquet_last_refresh = datetime.now()
            
            logger.info("=" * 60)
            logger.info(f"PARQUET REFRESH: SUCCESS")
            logger.info(f"  Total bars: {len(df)}")
            logger.info(f"  Now ends at: {self.parquet_last_time}")
            logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            logger.error(f"Error refreshing parquet from Binance: {e}")
            return self._load_parquet_features()

    def _update_synthetic_bar(self, price: float) -> None:
        """Update synthetic bar with current price (like MNQ/SPY bots).
        
        The synthetic bar represents the current incomplete 5-min bar.
        It's used to get real-time signal updates between bar closes.
        """
        now = datetime.now()
        # Round down to current 5-min bar start
        bar_time = now.replace(second=0, microsecond=0)
        bar_time = bar_time - timedelta(minutes=bar_time.minute % 5)
        
        if self.synthetic_bar is None or self.synthetic_bar['time'] != bar_time:
            # New bar - initialize
            self.synthetic_bar = {
                'time': bar_time,
                'open': price,
                'high': price,
                'low': price,
                'close': price
            }
        else:
            # Update existing bar
            self.synthetic_bar['high'] = max(self.synthetic_bar['high'], price)
            self.synthetic_bar['low'] = min(self.synthetic_bar['low'], price)
            self.synthetic_bar['close'] = price

    def get_features_with_synthetic_bar(self) -> pd.DataFrame:
        """Get features DataFrame with synthetic bar appended.
        
        HYBRID APPROACH (like MNQ/SPY bots):
        - Uses parquet features (correct cumulative values from years of data)
        - Appends synthetic bar for real-time updates
        - Updates price-dependent features for the new bar
        
        This eliminates the cumulative offset drift problem.
        """
        if self.synthetic_bar is None:
            return pd.DataFrame()
        
        if not self.use_parquet_features or self.parquet_features is None:
            logger.warning("Parquet features not available, falling back to Binance data")
            return pd.DataFrame()
        
        # Get last N bars from parquet for context
        df = self.parquet_features.tail(500).copy()
        
        # Check if synthetic bar time is after parquet end
        synthetic_time = self.synthetic_bar['time']
        
        # If synthetic bar is newer than parquet, append it
        if synthetic_time > self.parquet_last_time:
            # Copy the last parquet row as base for synthetic bar
            # This preserves all feature values (cumulative, MACD, ATR, etc.)
            last_row = df.iloc[-1:].copy()
            last_row.index = [synthetic_time]
            
            # Update OHLCV with synthetic bar values
            last_row['Open'] = self.synthetic_bar['open']
            last_row['High'] = self.synthetic_bar['high']
            last_row['Low'] = self.synthetic_bar['low']
            last_row['Close'] = self.synthetic_bar['close']
            
            # Drop any rows at or after synthetic time and append
            df = df[df.index < synthetic_time]
            df = pd.concat([df, last_row])
            
            # Update price-dependent features for the new bar
            df = self._update_features_for_new_bar(df)
        
        return df

    def _update_features_for_new_bar(self, df: pd.DataFrame) -> pd.DataFrame:
        """Update features for the last bar using existing parquet features.
        
        IMPORTANT: We do NOT recalculate RSI/MACD/ATR/price_to_ma here because:
        1. The parquet has correct values computed from full Binance history
        2. Recalculating from mixed data sources causes discrepancies
        3. The synthetic bar inherits feature values from the last parquet row
        
        Only update simple features that directly depend on the current bar's price.
        """
        if len(df) < 2:
            return df
        
        try:
            last_idx = df.index[-1]
            current_close = df['Close'].iloc[-1]
            prev_close = df['Close'].iloc[-2]
            
            # Update only simple return features (these are straightforward calculations)
            if 'return_1bar' in df.columns and prev_close != 0:
                df.loc[last_idx, 'return_1bar'] = (current_close / prev_close) - 1
            if 'log_return_1bar' in df.columns and prev_close > 0:
                df.loc[last_idx, 'log_return_1bar'] = np.log(current_close / prev_close)
            
            # DO NOT recalculate price_to_ma, momentum_rsi, trend_macd, volatility_atr
            # These should come from the parquet's last row (already copied)
            # Recalculating them causes discrepancies with backtest
            
        except Exception as e:
            logger.debug(f"Error updating features for new bar: {e}")
        
        return df

    def connect(self) -> bool:
        """Connect to IB Gateway."""
        try:
            # Ensure any stale connection is fully closed first
            try:
                if self.ib.isConnected():
                    self.ib.disconnect()
                    time.sleep(2)
            except Exception:
                pass
            
            port = IB_PORT if self.paper_trading else 7496
            self.ib.connect(IB_HOST, port, clientId=IB_CLIENT_ID, timeout=20)
            
            # Request delayed data
            self.ib.reqMarketDataType(3)
            
            # Qualify contract
            self.ib.qualifyContracts(self.contract)
            logger.info(f"Trading contract: {self.contract}")
            
            self.connected = True
            logger.info("Connected to IB Gateway")
            
            # Subscribe to portfolio updates for MBT price fallback (Feb 26 fix)
            # IB sends updatePortfolio with marketPrice even when Binance is down
            self.ib.updatePortfolioEvent += self._on_portfolio_update
            
            return True
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def _on_portfolio_update(self, item) -> None:
        """Capture MBT market price from IB portfolio updates (Feb 26 fix).
        
        IB sends updatePortfolio callbacks with marketPrice even when Binance
        WS+REST are both down. Use as 3rd price fallback for trailing stop checks.
        Note: marketPrice is the actual BTC price (not scaled by multiplier).
        avgCost is the one that's scaled by 0.1 multiplier.
        """
        try:
            if item.contract.symbol == 'MBT' and item.marketPrice > 0:
                # Sanity check: must look like a valid BTC price
                if 10000 < item.marketPrice < 500000:
                    was_none = self.ib_mbt_price is None
                    self.ib_mbt_price = item.marketPrice
                    self.ib_mbt_price_time = datetime.now()
                    if was_none:
                        logger.info(f"IB portfolio price initialized: ${item.marketPrice:.2f}")
        except Exception:
            pass

    def _sync_ib_positions(self) -> None:
        """Validate tracked positions against IB on startup.
        
        IMPORTANT: IB reports AGGREGATE positions across ALL connected clients
        (ensemble, tick, trend bots). We CANNOT determine which contracts belong
        to this bot. Therefore:
        - Trust btc_positions.json as the sole source of truth
        - NEVER adopt IB positions as "legacy" — they likely belong to other bots
        - Only warn if tracked positions exist but IB shows 0 (stale tracker)
        """
        try:
            # Wait for IB to send position data
            self.ib.sleep(2)
            
            # Get all positions from IB
            positions = self.ib.positions()
            
            mbt_position = None
            for pos in positions:
                if pos.contract.symbol == 'MBT':
                    mbt_position = pos
                    break
            
            tracked = self.position_manager.count_positions()
            ib_qty = int(mbt_position.position) if mbt_position else 0
            
            if tracked > 0:
                logger.info(f"Tracking {tracked} positions from btc_positions.json:")
                for pos in self.position_manager.get_all_positions():
                    logger.info(f"  - {pos.model_id}: {pos.direction} @ ${pos.entry_price:.2f} "
                               f"(stop: ${pos.stop_price:.2f}, target: ${pos.target_price:.2f})")
            
            if ib_qty == 0 and tracked > 0:
                logger.warning(f"WARNING: Tracking {tracked} positions but IB shows 0 MBT. "
                              f"Positions may be stale — clearing tracker.")
                for pos in self.position_manager.get_all_positions():
                    self.position_manager.remove_position(pos.model_id)
                logger.info("Cleared stale positions from tracker")
            elif ib_qty != 0 and tracked == 0:
                logger.info(f"IB shows {ib_qty} MBT contracts (aggregate across all clients). "
                           f"This bot has 0 tracked positions — not adopting (other bots own them).")
            elif ib_qty != 0 and tracked > 0:
                logger.info(f"IB aggregate: {ib_qty} MBT | This bot tracking: {tracked} positions")
            else:
                logger.info("No MBT positions in IB, no tracked positions — clean state")
            
        except Exception as e:
            logger.error(f"Error syncing IB positions: {e}")

    def disconnect(self) -> None:
        """Disconnect from IB Gateway."""
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            logger.info("Disconnected from IB Gateway")
    
    def is_connected(self) -> bool:
        """Check if IB connection is active."""
        try:
            return self.ib.isConnected()
        except Exception:
            return False
    
    def reconnect(self, retry_delay: int = 30) -> bool:
        """Attempt to reconnect to IB Gateway indefinitely until successful.
        
        Args:
            retry_delay: Seconds to wait between retries
            
        Returns:
            True when reconnection successful (never returns False, keeps trying)
        """
        logger.warning("Attempting to reconnect to IB Gateway...")
        
        # First disconnect cleanly if needed
        try:
            if self.ib.isConnected():
                self.ib.disconnect()
                time.sleep(2)  # Let socket fully close
            self.connected = False
            self.ticker = None
        except:
            pass
        
        attempt = 0
        while True:
            attempt += 1
            logger.info(f"Reconnection attempt {attempt}...")
            try:
                if self.connect():
                    logger.info(f"Reconnected successfully on attempt {attempt}")
                    return True
            except Exception as e:
                logger.warning(f"Reconnection attempt {attempt} failed: {e}")
            
            # Increase delay after many attempts (max 5 min)
            current_delay = min(retry_delay * (1 + attempt // 10), 300)
            logger.info(f"Waiting {current_delay}s before next attempt...")
            time.sleep(current_delay)

    def get_binance_data(self, limit: int = 500) -> pd.DataFrame:
        """Fetch 5-minute OHLCV data from Binance with retry logic."""
        max_retries = 5
        base_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                url = f"{BINANCE_API}/klines"
                params = {
                    'symbol': 'BTCUSDT',
                    'interval': '5m',
                    'limit': limit
                }
                response = requests.get(url, params=params, timeout=15)
                response.raise_for_status()  # Raise exception for bad status codes
                data = response.json()
                
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])
                
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    df[col] = df[col].astype(float)
                
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                
                # Success - reset any failure tracking
                if attempt > 0:
                    logger.info(f"Binance API recovered after {attempt + 1} attempts")
                
                return df
                
            except Exception as e:
                delay = base_delay * (2 ** attempt)  # Exponential backoff: 1, 2, 4, 8, 16 seconds
                if attempt < max_retries - 1:
                    logger.warning(f"Binance API attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"Binance API failed after {max_retries} attempts: {e}")
        
        return pd.DataFrame()

    def get_binance_ticker_price(self) -> Optional[float]:
        """Get current BTC price from Binance ticker API (lightweight, fast)."""
        try:
            url = f"{BINANCE_API}/ticker/price"
            params = {'symbol': 'BTCUSDT'}
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            return float(data['price'])
        except Exception as e:
            logger.debug(f"Binance ticker API error: {e}")
            return None

    def start_websocket(self) -> None:
        """Start WebSocket connection for real-time price updates."""
        if not WEBSOCKET_AVAILABLE:
            logger.warning("WebSocket not available - using REST API for price updates")
            return
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                if 'c' in data:  # 'c' is the close/current price in mini ticker
                    self.ws_price = float(data['c'])
                    self.ws_price_time = datetime.now()
            except Exception as e:
                logger.debug(f"WebSocket message error: {e}")
        
        def on_error(ws, error):
            logger.warning(f"WebSocket error: {error}")
            self.ws_connected = False
        
        def on_close(ws, close_status_code, close_msg):
            logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
            self.ws_connected = False
        
        def on_open(ws):
            logger.info("WebSocket connected - receiving real-time BTC prices")
            self.ws_connected = True
        
        def run_websocket():
            while True:
                try:
                    # Binance WebSocket for mini ticker (updates every 1 second)
                    ws_url = "wss://stream.binance.com:9443/ws/btcusdt@miniTicker"
                    self.ws = websocket.WebSocketApp(
                        ws_url,
                        on_message=on_message,
                        on_error=on_error,
                        on_close=on_close,
                        on_open=on_open
                    )
                    self.ws.run_forever()
                except Exception as e:
                    logger.error(f"WebSocket thread error: {e}")
                
                # Reconnect after 5 seconds if disconnected
                if not self.ws_connected:
                    logger.info("WebSocket reconnecting in 5 seconds...")
                    time.sleep(5)
        
        self.ws_thread = threading.Thread(target=run_websocket, daemon=True)
        self.ws_thread.start()
        logger.info("WebSocket thread started for real-time price updates")

    def stop_websocket(self) -> None:
        """Stop WebSocket connection."""
        if self.ws:
            self.ws.close()
            self.ws_connected = False
            logger.info("WebSocket stopped")

    def get_realtime_price(self) -> Optional[float]:
        """Get the most recent price with 3-tier fallback (Feb 26 fix).
        
        Priority: 1) Binance WebSocket, 2) Binance REST, 3) IB portfolio price.
        Feb 20 bug: Binance WS+REST both failed for 7 min, bot went blind,
        missed trailing stop activation. IB still had prices via updatePortfolio.
        """
        # Tier 1: Binance WebSocket (freshest, < 5 seconds old)
        if self.ws_price and self.ws_price_time:
            age = (datetime.now() - self.ws_price_time).total_seconds()
            if age < 5:
                return self.ws_price
        
        # Tier 2: Binance REST API ticker
        rest_price = self.get_binance_ticker_price()
        if rest_price:
            return rest_price
        
        # Tier 3: IB portfolio market price (Feb 26 fix)
        # IB sends updatePortfolio with marketPrice even during Binance outages
        if self.ib_mbt_price and self.ib_mbt_price_time:
            age = (datetime.now() - self.ib_mbt_price_time).total_seconds()
            if age < 30:  # IB updates every ~3 min, so 30s is generous
                logger.warning(f"Using IB fallback price: ${self.ib_mbt_price:.2f} ({age:.0f}s old)")
                return self.ib_mbt_price
        
        return None

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators and features matching training data."""
        if len(df) < 50:
            return df
        
        try:
            # Add all TA features (same as training)
            df = ta.add_all_ta_features(
                df, open="Open", high="High", low="Low", 
                close="Close", volume="Volume", fillna=True
            )
            
            # Time features (matching feature_engineering.py)
            df['hour'] = df.index.hour
            df['minute'] = df.index.minute
            df['time_of_day'] = df['hour'] + df['minute'] / 60
            
            # BTC trades 24/7, so market session features are different
            # We'll set them to reasonable defaults
            df['is_premarket'] = 0
            df['is_regular_hours'] = 1  # BTC always "regular"
            df['is_afterhours'] = 0
            df['is_first_hour'] = (df['hour'] == 0).astype(int)
            df['is_last_hour'] = (df['hour'] == 23).astype(int)
            
            df['day_of_week'] = df.index.dayofweek
            df['is_monday'] = (df['day_of_week'] == 0).astype(int)
            df['is_friday'] = (df['day_of_week'] == 4).astype(int)
            
            df['month'] = df.index.month
            df['is_month_end'] = df.index.is_month_end.astype(int)
            df['is_month_start'] = df.index.is_month_start.astype(int)
            df['week_of_year'] = df.index.isocalendar().week.astype(int)
            
            # Price features (matching feature_engineering.py)
            df['return_1bar'] = df['Close'].pct_change()
            df['return_5bar'] = df['Close'].pct_change(5)
            df['return_10bar'] = df['Close'].pct_change(10)
            df['return_20bar'] = df['Close'].pct_change(20)
            df['log_return_1bar'] = np.log(df['Close'] / df['Close'].shift(1))
            
            for window in [5, 10, 20, 50]:
                ma = df['Close'].rolling(window).mean()
                df[f'price_to_ma{window}'] = df['Close'] / ma - 1
            
            for window in [5, 10, 20]:
                df[f'volatility_{window}bar'] = df['return_1bar'].rolling(window).std()
            
            df['bar_range'] = (df['High'] - df['Low']) / df['Close']
            df['bar_range_ma5'] = df['bar_range'].rolling(5).mean()
            df['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
            df['gap'] = df['Open'] / df['Close'].shift(1) - 1
            
            # Daily context features
            df['trade_date'] = df.index.date
            daily = df.groupby('trade_date').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 
                'Close': 'last', 'Volume': 'sum'
            })
            daily['prev_close'] = daily['Close'].shift(1)
            daily['prev_high'] = daily['High'].shift(1)
            daily['prev_low'] = daily['Low'].shift(1)
            daily['prev_volume'] = daily['Volume'].shift(1)
            daily['prev_day_return'] = daily['Close'].pct_change()
            daily['prev_2day_return'] = daily['Close'].pct_change(2)
            daily['prev_5day_return'] = daily['Close'].pct_change(5)
            
            df = df.merge(
                daily[['prev_close', 'prev_high', 'prev_low', 'prev_volume',
                       'prev_day_return', 'prev_2day_return', 'prev_5day_return']], 
                left_on='trade_date', right_index=True, how='left'
            )
            
            df['price_vs_prev_close'] = df['Close'] / df['prev_close'] - 1
            df['price_vs_prev_high'] = df['Close'] / df['prev_high'] - 1
            df['price_vs_prev_low'] = df['Close'] / df['prev_low'] - 1
            df['volume_vs_prev'] = df['Volume'] / (df['prev_volume'] / 288 + 1)  # ~288 bars per day for BTC
            
            df = df.drop(columns=['trade_date'], errors='ignore')
            
            # Lagged indicator features - MUST match training: [1, 2, 3, 5, 10, 20, 50]
            key_indicators = ['momentum_rsi', 'trend_macd', 'trend_macd_signal', 
                            'trend_macd_diff', 'volatility_bbp', 'volatility_atr',
                            'trend_adx', 'momentum_stoch', 'volume_obv', 'volume_mfi']
            LOOKBACK_PERIODS = [1, 2, 3, 5, 10, 20, 50]
            for indicator in key_indicators:
                if indicator in df.columns:
                    for lag in LOOKBACK_PERIODS:
                        df[f'{indicator}_lag{lag}'] = df[indicator].shift(lag)
            
            # Indicator changes
            change_indicators = ['momentum_rsi', 'trend_macd', 'trend_adx', 'volatility_atr']
            for indicator in change_indicators:
                if indicator in df.columns:
                    df[f'{indicator}_change_1'] = df[indicator].diff(1)
                    df[f'{indicator}_change_5'] = df[indicator].diff(5)
            
            # Fill NaN and inf
            df = df.fillna(0)
            df = df.replace([np.inf, -np.inf], 0)
            
            # NOTE: Cumulative feature offsets REMOVED - now using parquet + hybrid bar approach
            # which preserves correct cumulative values from training data
            
        except Exception as e:
            logger.error(f"Error adding features: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        return df

    def get_signals(self, df: pd.DataFrame) -> List[Dict]:
        """Get signals from all models (LONG and SHORT)."""
        signals = []
        
        if len(df) < 50 or len(self.feature_cols) == 0:
            return signals
        
        # Prepare features
        available_cols = [c for c in self.feature_cols if c in df.columns]
        if len(available_cols) < len(self.feature_cols) * 0.8:
            logger.warning(f"Only {len(available_cols)}/{len(self.feature_cols)} features available")
            return signals
        
        X = df[available_cols].iloc[[-1]]  # Last row only
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        
        all_probs = {}  # Store all model probs for cross-timeframe checks
        for model_name, model_info in self.models.items():
            try:
                proba = model_info['model'].predict_proba(X)[0, 1]
                all_probs[model_name] = proba
                config = model_info['config']
                direction = config.get('direction', 'LONG')
                
                # Use different thresholds for LONG vs SHORT
                threshold = self.probability_threshold_long if direction == 'LONG' else self.probability_threshold_short
                if proba >= threshold:
                    signals.append({
                        'model_id': model_name,
                        'horizon': config['horizon'],
                        'threshold': config['threshold'],
                        'probability': proba,
                        'priority': config['priority'],
                        'target_bars': config['horizon_bars'],
                        'direction': direction
                    })
                    
            except Exception as e:
                logger.error(f"Error getting signal from {model_name}: {e}")
        
        # Save for cross-timeframe checks in check_entries
        self._last_model_probs = all_probs
        
        # Sort by priority (LONG and SHORT separately, then interleave)
        long_signals = [s for s in signals if s['direction'] == 'LONG']
        short_signals = [s for s in signals if s['direction'] == 'SHORT']
        
        # =================================================================
        # DIRECTION CONFLICT FILTER (Feb 17, 2026)
        # When both LONG and SHORT signals fire, only keep the stronger
        # direction based on average probability. Prevents opposing trades
        # that cancel each other out (71% of signals were conflicting).
        # =================================================================
        if long_signals and short_signals:
            long_avg_prob = sum(s['probability'] for s in long_signals) / len(long_signals)
            short_avg_prob = sum(s['probability'] for s in short_signals) / len(short_signals)
            
            if long_avg_prob >= short_avg_prob:
                logger.info(f"[CONFLICT FILTER] LONG ({long_avg_prob:.1%}) vs SHORT ({short_avg_prob:.1%}) → keeping LONG")
                short_signals = []
            else:
                logger.info(f"[CONFLICT FILTER] LONG ({long_avg_prob:.1%}) vs SHORT ({short_avg_prob:.1%}) → keeping SHORT")
                long_signals = []
        
        # =================================================================
        # SINGLE ML MODEL PER DIRECTION (Feb 19, 2026)
        # Only keep the highest-probability ML signal per direction.
        # Prevents duplicate entries (e.g. 4h + 6h both LONG at same price)
        # that double the risk and SL losses. Indicator signals are exempt.
        # =================================================================
        ml_long = [s for s in long_signals if s.get('signal_type') != 'indicator']
        ind_long = [s for s in long_signals if s.get('signal_type') == 'indicator']
        ml_short = [s for s in short_signals if s.get('signal_type') != 'indicator']
        ind_short = [s for s in short_signals if s.get('signal_type') == 'indicator']
        
        if len(ml_long) > 1:
            best = max(ml_long, key=lambda s: s['probability'])
            dropped = [s['model_id'] for s in ml_long if s != best]
            logger.info(f"[SINGLE ML] Keeping best LONG: {best['model_id']} ({best['probability']:.1%}), dropping: {dropped}")
            ml_long = [best]
        
        if len(ml_short) > 1:
            best = max(ml_short, key=lambda s: s['probability'])
            dropped = [s['model_id'] for s in ml_short if s != best]
            logger.info(f"[SINGLE ML] Keeping best SHORT: {best['model_id']} ({best['probability']:.1%}), dropping: {dropped}")
            ml_short = [best]
        
        long_signals = ml_long + ind_long
        short_signals = ml_short + ind_short
        
        long_signals.sort(key=lambda x: x['priority'])
        short_signals.sort(key=lambda x: x['priority'])
        
        # Interleave: take best LONG, best SHORT, etc.
        result = []
        while long_signals or short_signals:
            if long_signals:
                result.append(long_signals.pop(0))
            if short_signals:
                result.append(short_signals.pop(0))
        
        return result

    def get_all_probabilities(self, df: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Get probabilities from all models and key indicators."""
        probabilities = {}
        indicators = {}
        
        if len(df) < 50 or len(self.feature_cols) == 0:
            return probabilities, indicators
        
        # Prepare features
        available_cols = [c for c in self.feature_cols if c in df.columns]
        if len(available_cols) < len(self.feature_cols) * 0.8:
            missing_pct = (1 - len(available_cols) / len(self.feature_cols)) * 100
            logger.warning(f"[FEATURE COVERAGE] Only {len(available_cols)}/{len(self.feature_cols)} "
                           f"features available ({missing_pct:.0f}% missing) — suppressing all signals")
            return probabilities, indicators
        
        X = df[available_cols].iloc[[-1]]
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        
        # Get probabilities from all models
        for model_name, model_info in self.models.items():
            try:
                proba = model_info['model'].predict_proba(X)[0, 1]
                probabilities[model_name] = proba
            except Exception as e:
                probabilities[model_name] = 0.0
        
        # Get key indicators from last row
        try:
            last_row = df.iloc[-1]
            indicators['rsi'] = last_row.get('momentum_rsi', 0)
            indicators['macd'] = last_row.get('trend_macd', 0)
            indicators['atr'] = last_row.get('volatility_atr', 0)
            indicators['bb_pct_b'] = last_row.get('volatility_bbp', 0.5)
        except:
            pass
        
        return probabilities, indicators

    def get_indicator_long_signals(self, df: pd.DataFrame) -> List[Dict]:
        """Get LONG signals based on simple indicator rules.
        
        Strategy: BB %B > 0.5 + MACD > Signal
        - Backtest: 95 trades/week, 35% WR, +$1,193/week
        - Uses 0.08% SL, 0.70% TP (8.8:1 R:R ratio)
        """
        signals = []
        
        if not INDICATOR_LONG_ENABLED:
            return signals
        
        if len(df) < 50:
            return signals
        
        try:
            last_row = df.iloc[-1]
            
            # Get indicator values
            bb_pct_b = last_row.get('volatility_bbp', 0.5)
            macd = last_row.get('trend_macd', 0)
            macd_signal = last_row.get('trend_macd_signal', 0)
            
            # Check conditions: BB %B > 0.5 AND MACD > MACD Signal
            bb_condition = bb_pct_b > 0.5
            macd_condition = macd > macd_signal
            
            if bb_condition and macd_condition:
                signals.append({
                    'model_id': 'indicator_long',
                    'horizon': 'indicator',
                    'threshold': INDICATOR_LONG_TP_PCT,  # Use TP as threshold for target price calc
                    'probability': 1.0,  # Indicator signal, not probability
                    'priority': 0,  # Highest priority
                    'target_bars': INDICATOR_LONG_TIMEOUT_BARS,
                    'direction': 'LONG',
                    'signal_type': 'indicator',  # Mark as indicator signal
                    'bb_pct_b': bb_pct_b,
                    'macd': macd,
                    'macd_signal': macd_signal
                })
                logger.info(f"INDICATOR LONG signal: BB%B={bb_pct_b:.3f} (>0.5), MACD={macd:.2f} > Signal={macd_signal:.2f}")
        
        except Exception as e:
            logger.error(f"Error getting indicator signals: {e}")
        
        return signals

    def get_indicator_meanrev_signals(self, df: pd.DataFrame) -> List[Dict]:
        """Get LONG signals based on mean reversion (BB %B < 0.5).
        
        Strategy: BB %B < 0.5 (buy when oversold, same logic as MNQ)
        - 30-day 15-sec backtest: 342 trades, 44.2% WR, +$184/week
        - Uses 0.30% SL, 0.50% TP
        """
        signals = []
        
        if not INDICATOR_MEANREV_ENABLED:
            return signals
        
        if len(df) < 50:
            return signals
        
        try:
            last_row = df.iloc[-1]
            
            # Get indicator values
            bb_pct_b = last_row.get('volatility_bbp', 0.5)
            
            # Check condition: BB %B < 0.5 (price below BB midline = oversold)
            bb_condition = bb_pct_b < 0.5
            
            if bb_condition:
                signals.append({
                    'model_id': 'indicator_meanrev',
                    'horizon': 'indicator',
                    'threshold': INDICATOR_MEANREV_TP_PCT,
                    'probability': 1.0,
                    'priority': 0,
                    'target_bars': INDICATOR_MEANREV_TIMEOUT_BARS,
                    'direction': 'LONG',
                    'signal_type': 'indicator',
                    'bb_pct_b': bb_pct_b
                })
                logger.info(f"INDICATOR MEANREV signal: BB%B={bb_pct_b:.3f} (<0.5)")
        
        except Exception as e:
            logger.error(f"Error getting meanrev signals: {e}")
        
        return signals

    def get_indicator_trend_signals(self, df: pd.DataFrame) -> List[Dict]:
        """Get LONG/SHORT signals based on ROC + MACD Histogram trend strategy.
        
        Strategy (Jan 6, 2026):
        - LONG:  ROC(12) > 0.4% AND MACD_Hist > 0 AND MACD_Hist increasing
        - SHORT: ROC(12) < -0.4% AND MACD_Hist < 0 AND MACD_Hist decreasing
        
        30-day 15-sec backtest: 145 trades, 49% WR, +$437/week
        SL: 0.70%, TP: 1.40% (R:R = 2:1)
        """
        signals = []
        
        if not INDICATOR_TREND_ENABLED:
            return signals
        
        if len(df) < 50:
            return signals
        
        try:
            # Need at least 2 rows for MACD histogram comparison
            if len(df) < 2:
                return signals
            
            last_row = df.iloc[-1]
            prev_row = df.iloc[-2]
            
            # Calculate ROC(12) - Rate of Change over 12 bars (1 hour)
            if len(df) >= 13:
                close_now = last_row.get('Close', 0)
                close_12_ago = df['Close'].iloc[-13] if 'Close' in df.columns else 0
                if close_12_ago > 0:
                    roc_12 = (close_now - close_12_ago) / close_12_ago * 100
                else:
                    roc_12 = 0
            else:
                roc_12 = 0
            
            # Get MACD values
            macd = last_row.get('trend_macd', 0)
            macd_signal = last_row.get('trend_macd_signal', 0)
            macd_hist = macd - macd_signal
            
            macd_prev = prev_row.get('trend_macd', 0)
            macd_signal_prev = prev_row.get('trend_macd_signal', 0)
            macd_hist_prev = macd_prev - macd_signal_prev
            
            # Check cooldown - use class variable to track last trade time
            if not hasattr(self, '_last_trend_trade_bar'):
                self._last_trend_trade_bar = -INDICATOR_TREND_COOLDOWN_BARS
            
            current_bar = len(df)
            bars_since_last_trade = current_bar - self._last_trend_trade_bar
            
            if bars_since_last_trade < INDICATOR_TREND_COOLDOWN_BARS:
                return signals
            
            # LONG condition: ROC > threshold AND MACD_Hist > 0 AND MACD_Hist increasing
            if (roc_12 > INDICATOR_TREND_ROC_THRESHOLD and 
                macd_hist > 0 and 
                macd_hist > macd_hist_prev):
                
                signals.append({
                    'model_id': 'indicator_trend',
                    'horizon': 'indicator',
                    'threshold': INDICATOR_TREND_TP_PCT,
                    'probability': 1.0,
                    'priority': 0,
                    'target_bars': INDICATOR_TREND_TIMEOUT_BARS,
                    'direction': 'LONG',
                    'signal_type': 'indicator',
                    'roc_12': roc_12,
                    'macd_hist': macd_hist
                })
                logger.info(f"INDICATOR TREND LONG signal: ROC={roc_12:.2f}% (>{INDICATOR_TREND_ROC_THRESHOLD}%), MACD_Hist={macd_hist:.4f} (>0, increasing)")
            
            # SHORT condition: ROC < -threshold AND MACD_Hist < 0 AND MACD_Hist decreasing
            elif (roc_12 < -INDICATOR_TREND_ROC_THRESHOLD and 
                  macd_hist < 0 and 
                  macd_hist < macd_hist_prev):
                
                signals.append({
                    'model_id': 'indicator_trend',
                    'horizon': 'indicator',
                    'threshold': INDICATOR_TREND_TP_PCT,
                    'probability': 1.0,
                    'priority': 0,
                    'target_bars': INDICATOR_TREND_TIMEOUT_BARS,
                    'direction': 'SHORT',
                    'signal_type': 'indicator',
                    'roc_12': roc_12,
                    'macd_hist': macd_hist
                })
                logger.info(f"INDICATOR TREND SHORT signal: ROC={roc_12:.2f}% (<-{INDICATOR_TREND_ROC_THRESHOLD}%), MACD_Hist={macd_hist:.4f} (<0, decreasing)")
        
        except Exception as e:
            logger.error(f"Error getting trend signals: {e}")
        
        return signals

    def log_current_signals(self) -> None:
        """Log current signals to console and CSV every 15 seconds."""
        if self.cached_features_df is None or self.current_price is None:
            return
        
        now = datetime.now()
        
        # Check if enough time has passed since last log
        if self.last_signal_log_time is not None:
            elapsed = (now - self.last_signal_log_time).total_seconds()
            if elapsed < SIGNAL_LOG_INTERVAL:
                return
        
        # Get all probabilities
        probabilities, indicators = self.get_all_probabilities(self.cached_features_df)
        
        if probabilities:
            # Log to console (like MNQ/SPY bots do)
            prob_str = " | ".join([f"{k}:{v*100:.0f}%" for k, v in probabilities.items()])
            n_pos = self.position_manager.count_positions()
            logger.info(f"BTC: ${self.current_price:.2f} | Pos: {n_pos}/{self.max_positions} | {prob_str}")
            
            # Log to CSV
            self.signal_logger.log_signals(
                timestamp=now,
                price=self.current_price,
                probabilities=probabilities,
                threshold=self.probability_threshold_long,  # Use LONG threshold for logging
                active_positions=self.position_manager.count_positions(),
                indicators=indicators
            )
            self.last_signal_log_time = now

    def place_order(self, direction: str, size: int) -> Optional[tuple]:
        """Place a market order. Returns (order_id, fill_price) or None."""
        try:
            action = 'BUY' if direction == 'LONG' else 'SELL'
            order = MarketOrder(action, size)
            
            trade = self.ib.placeOrder(self.contract, order)
            self.ib.sleep(2)  # Wait for fill
            
            if trade.orderStatus.status == 'Filled':
                fill_price = trade.orderStatus.avgFillPrice
                logger.info(f"Order filled: {action} {size} @ {fill_price}")
                return trade.order.orderId, fill_price
            else:
                logger.warning(f"Order status: {trade.orderStatus.status}")
                # Return current_price as best-effort fill estimate for pending orders
                return trade.order.orderId, self.current_price
                
        except Exception as e:
            logger.error(f"Order error: {e}")
            return None

    def close_position(self, position: Position) -> bool:
        """Close a position. Returns True if filled, False if pending/failed."""
        try:
            # SAFETY CHECK: Verify IB position before sending close order
            # This prevents sending orders when position is already flat
            try:
                ib_positions = self.ib.positions()
                mbt_pos = next((p for p in ib_positions if p.contract.symbol == 'MBT'), None)
                if mbt_pos is None or mbt_pos.position == 0:
                    logger.warning(f"[{position.model_id}] IB position is already flat - removing from tracker (NO trade logged)")
                    return 'FLAT'  # Position gone but no real trade occurred
            except Exception as e:
                logger.warning(f"Could not verify IB position: {e} - proceeding with close")
            
            action = 'SELL' if position.direction == 'LONG' else 'BUY'
            order = MarketOrder(action, position.size)
            
            trade = self.ib.placeOrder(self.contract, order)
            self.ib.sleep(2)
            
            status = trade.orderStatus.status
            
            if status == 'Filled':
                exit_price = trade.orderStatus.avgFillPrice
                pnl = (exit_price - position.entry_price) * BTC_CONTRACT_VALUE * position.size
                if position.direction == 'SHORT':
                    pnl = -pnl
                
                logger.info(f"Closed [{position.model_id}]: {exit_price:.2f}, PnL: ${pnl:.2f}")
                # Clear pending close tracking
                if position.model_id in self.pending_close_orders:
                    del self.pending_close_orders[position.model_id]
                return True
            
            elif status in ['PreSubmitted', 'Submitted', 'Inactive', 'PendingSubmit']:
                # Order is pending (market closed or queued)
                # Mark position as having a pending close order
                self.pending_close_orders[position.model_id] = datetime.now()
                logger.warning(f"Close order PENDING for [{position.model_id}] - status: {status}")
                logger.warning(f"  Market may be closed. Order will execute when market opens.")
                return False  # Don't remove position yet
            
            else:
                logger.warning(f"Close order status: {status} for [{position.model_id}]")
                return False
            
        except Exception as e:
            logger.error(f"Close error: {e}")
            return False

    def _check_timeout_positions_while_closed(self, positions: List[Position]) -> None:
        """Check for timed-out positions while market is closed.
        
        Marks positions that have exceeded timeout for immediate close when market reopens.
        Also logs warnings about positions that should have been closed.
        """
        for pos in positions:
            entry_time = datetime.fromisoformat(pos.entry_time)
            time_elapsed_mins = (datetime.now() - entry_time).total_seconds() / 60
            timeout_bars = int(pos.target_bars * TIMEOUT_MULTIPLIER)
            bars_elapsed_by_time = int(time_elapsed_mins / 5)
            
            if bars_elapsed_by_time >= timeout_bars:
                # Position has timed out - mark for immediate close when market opens
                if pos.model_id not in self.pending_close_orders:
                    logger.warning(f"[{pos.model_id}] TIMEOUT while market closed: "
                                 f"{bars_elapsed_by_time}/{timeout_bars} bars elapsed. "
                                 f"Will close immediately when market reopens.")
                    # Mark as pending close so it gets closed first thing when market opens
                    self.pending_close_orders[pos.model_id] = datetime.now()
                    pos.pending_close = True
                    self.position_manager.save_positions()

    def check_exits(self) -> None:
        """Check if any positions should be closed (LONG and SHORT)."""
        positions = self.position_manager.get_all_positions()
        
        if self.current_price is None or self.current_price <= 0:
            return
        
        # Check if market is open
        market_open = is_cme_mbt_open()
        
        # Even when market is closed, check for timed-out positions
        # and close them immediately when market reopens
        if not market_open:
            self._check_timeout_positions_while_closed(positions)
            return
        
        # First, check status of any pending close orders
        self._check_pending_close_orders()
        
        for pos in positions:
            # Check if position was marked for close while market was closed
            if pos.pending_close and pos.model_id in self.pending_close_orders:
                # Market just reopened - close this timed-out position immediately
                entry_time = datetime.fromisoformat(pos.entry_time)
                time_elapsed_mins = (datetime.now() - entry_time).total_seconds() / 60
                bars_elapsed = int(time_elapsed_mins / 5)
                timeout_bars = int(pos.target_bars * TIMEOUT_MULTIPLIER)
                
                if pos.direction == 'LONG':
                    pnl_pct = (self.current_price / pos.entry_price - 1) * 100
                else:
                    pnl_pct = (pos.entry_price / self.current_price - 1) * 100
                
                reason = f"TIMEOUT (market reopened, {bars_elapsed}/{timeout_bars} bars) @ ${self.current_price:.2f} ({pnl_pct:+.2f}%)"
                logger.info(f"EXIT [{pos.model_id}] {pos.direction}: {reason}")
                
                # Remove from pending_close_orders since we're about to close
                del self.pending_close_orders[pos.model_id]
                
                close_result = self.close_position(pos)
                if close_result == 'FLAT':
                    self.position_manager.remove_position(pos.model_id)
                    logger.info(f"Removed phantom position [{pos.model_id}] — IB was already flat")
                elif close_result:
                    pnl_dollar = (pnl_pct / 100) * self.current_price * BTC_CONTRACT_VALUE - 2.02
                    try:
                        market_context = self._get_market_context_for_logging(pos)
                        log_trade_to_db(
                            bot_type='BTC',
                            model_id=pos.model_id,
                            direction=pos.direction,
                            entry_time=pos.entry_time,
                            entry_price=pos.entry_price,
                            exit_time=datetime.now().isoformat(),
                            exit_price=self.current_price,
                            exit_reason='TIMEOUT',
                            pnl_pct=pnl_pct,
                            pnl_dollar=pnl_dollar,
                            bars_held=bars_elapsed,
                            horizon_bars=pos.target_bars,
                            model_horizon=pos.model_horizon if hasattr(pos, 'model_horizon') else None,
                            model_threshold=pos.model_threshold if hasattr(pos, 'model_threshold') else None,
                            market_context=market_context,
                            entry_probability=pos.entry_probability if hasattr(pos, 'entry_probability') else None
                        )
                        
                        # Log exit features for learning loop
                        exit_features = self._get_current_features()
                        exit_features['price'] = self.current_price
                        self._log_exit_features(
                            pos.model_id, pos.direction,
                            pos.entry_features if hasattr(pos, 'entry_features') and pos.entry_features else {},
                            exit_features, pnl_pct, pnl_dollar, 'TIMEOUT',
                            pos.entry_probability if hasattr(pos, 'entry_probability') else 0,
                            pos.entry_trend_1h if hasattr(pos, 'entry_trend_1h') else 0
                        )
                    except Exception as e:
                        logger.error(f"Error logging trade to DB: {e}")
                continue
            
            # Skip if there's already a pending close order for this position
            if pos.model_id in self.pending_close_orders:
                pending_time = self.pending_close_orders[pos.model_id]
                elapsed_mins = (datetime.now() - pending_time).total_seconds() / 60
                # Only log every 30 minutes to avoid spam
                if elapsed_mins < 30 or int(elapsed_mins) % 30 != 0:
                    continue
                logger.info(f"[{pos.model_id}] Close order still pending ({elapsed_mins:.0f} mins)")
                continue
            
            # Update trailing stop tracking
            self._update_trailing_stop(pos, self.current_price)
            
            should_exit = False
            reason = ""
            
            if pos.direction == 'LONG':
                # LONG: Take Profit when price rises to target
                if pos.target_price > 0 and self.current_price >= pos.target_price:
                    should_exit = True
                    pnl_pct = (self.current_price / pos.entry_price - 1) * 100
                    reason = f"TAKE PROFIT @ ${self.current_price:.2f} (+{pnl_pct:.2f}%)"
                
                # LONG: Trailing stop hit
                elif pos.trailing_stop_active and self.current_price <= pos.trailing_stop_price:
                    should_exit = True
                    pnl_pct = (self.current_price / pos.entry_price - 1) * 100
                    reason = f"TRAILING STOP @ ${self.current_price:.2f} ({pnl_pct:+.2f}%)"
                
                # LONG: Stop Loss when price falls to stop
                elif pos.stop_price > 0 and self.current_price <= pos.stop_price:
                    should_exit = True
                    pnl_pct = (self.current_price / pos.entry_price - 1) * 100
                    reason = f"STOP LOSS @ ${self.current_price:.2f} ({pnl_pct:.2f}%)"
            
            elif pos.direction == 'SHORT':
                # SHORT: Take Profit when price falls to target (target < entry)
                if pos.target_price > 0 and self.current_price <= pos.target_price:
                    should_exit = True
                    pnl_pct = (pos.entry_price / self.current_price - 1) * 100
                    reason = f"TAKE PROFIT @ ${self.current_price:.2f} (+{pnl_pct:.2f}%)"
                
                # SHORT: Trailing stop hit
                elif pos.trailing_stop_active and self.current_price >= pos.trailing_stop_price:
                    should_exit = True
                    pnl_pct = (pos.entry_price / self.current_price - 1) * 100
                    reason = f"TRAILING STOP @ ${self.current_price:.2f} ({pnl_pct:+.2f}%)"
                
                # SHORT: Stop Loss when price rises to stop (stop > entry)
                elif pos.stop_price > 0 and self.current_price >= pos.stop_price:
                    should_exit = True
                    pnl_pct = (pos.entry_price / self.current_price - 1) * 100
                    reason = f"STOP LOSS @ ${self.current_price:.2f} ({pnl_pct:.2f}%)"
            
            # Time-based exit (timeout) - same for both directions (with 2x multiplier)
            # Check both bars_held counter AND actual time elapsed (in case bot was down)
            entry_time = datetime.fromisoformat(pos.entry_time)
            time_elapsed_mins = (datetime.now() - entry_time).total_seconds() / 60
            timeout_bars = int(pos.target_bars * TIMEOUT_MULTIPLIER)  # 2x timeout
            bars_elapsed_by_time = int(time_elapsed_mins / 5)  # 5-min bars
            
            if not should_exit and (pos.bars_held >= timeout_bars or bars_elapsed_by_time >= timeout_bars):
                should_exit = True
                if pos.direction == 'LONG':
                    pnl_pct = (self.current_price / pos.entry_price - 1) * 100
                else:
                    pnl_pct = (pos.entry_price / self.current_price - 1) * 100
                reason = f"TIMEOUT ({pos.bars_held}/{timeout_bars} bars) @ ${self.current_price:.2f} ({pnl_pct:+.2f}%)"
            
            # PRE-CLOSE EXIT: Close positions before market closure (weekends/holidays)
            # This prevents positions from being stuck when market closes
            if not should_exit:
                pre_close, pre_close_reason = should_close_positions_before_market_close()
                if pre_close:
                    should_exit = True
                    if pos.direction == 'LONG':
                        pnl_pct = (self.current_price / pos.entry_price - 1) * 100
                    else:
                        pnl_pct = (pos.entry_price / self.current_price - 1) * 100
                    reason = f"PRE-CLOSE ({pre_close_reason}) @ ${self.current_price:.2f} ({pnl_pct:+.2f}%)"
                    logger.warning(f"[{pos.model_id}] Closing position before market closure: {pre_close_reason}")
            
            if should_exit:
                # DUPLICATE ORDER PREVENTION: Mark as pending BEFORE sending order
                # This prevents the same position from being processed again in the loop
                if pos.model_id in self.pending_close_orders:
                    logger.debug(f"[{pos.model_id}] Already has pending close order, skipping")
                    continue
                self.pending_close_orders[pos.model_id] = datetime.now()
                
                logger.info(f"EXIT [{pos.model_id}] {pos.direction}: {reason}")
                close_result = self.close_position(pos)
                if close_result == 'FLAT':
                    self.position_manager.remove_position(pos.model_id)
                    logger.info(f"Removed phantom position [{pos.model_id}] — IB was already flat (NO trade logged)")
                elif close_result:
                    # Log trade to database for performance tracking
                    if 'TAKE PROFIT' in reason:
                        exit_reason_db = 'TAKE_PROFIT'
                    elif 'TRAILING STOP' in reason:
                        exit_reason_db = 'TRAILING_STOP'
                    elif 'STOP LOSS' in reason:
                        exit_reason_db = 'STOP_LOSS'
                        # Record SL time for cooldown (Feb 19, 2026)
                        self.last_sl_time[pos.direction] = datetime.now()
                        logger.info(f"[SL COOLDOWN] {pos.direction} SL at {datetime.now().strftime('%H:%M:%S')}, cooldown {SL_COOLDOWN_SECONDS}s")
                    elif 'PRE-CLOSE' in reason:
                        exit_reason_db = 'PRE_CLOSE'
                    else:
                        exit_reason_db = 'TIMEOUT'
                    pnl_dollar = (pnl_pct / 100) * self.current_price * BTC_CONTRACT_VALUE - 2.02
                    try:
                        # Get current market context for feedback loop learning
                        market_context = self._get_market_context_for_logging(pos)
                        
                        log_trade_to_db(
                            bot_type='BTC',
                            model_id=pos.model_id,
                            direction=pos.direction,
                            entry_time=pos.entry_time,
                            entry_price=pos.entry_price,
                            exit_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            exit_price=self.current_price,
                            pnl_pct=pnl_pct,
                            pnl_dollar=pnl_dollar,
                            exit_reason=exit_reason_db,
                            bars_held=pos.bars_held,
                            horizon_bars=pos.target_bars,
                            model_horizon=pos.model_horizon,
                            model_threshold=pos.model_threshold,
                            market_context=market_context,
                            entry_probability=pos.entry_probability if hasattr(pos, 'entry_probability') else None
                        )
                        
                        # Log exit features for learning loop
                        exit_features = self._get_current_features()
                        exit_features['price'] = self.current_price
                        self._log_exit_features(
                            pos.model_id, pos.direction,
                            pos.entry_features if hasattr(pos, 'entry_features') and pos.entry_features else {},
                            exit_features, pnl_pct, pnl_dollar, exit_reason_db,
                            pos.entry_probability if hasattr(pos, 'entry_probability') else 0,
                            pos.entry_trend_1h if hasattr(pos, 'entry_trend_1h') else 0
                        )
                    except Exception as e:
                        logger.warning(f"Failed to log trade to database: {e}")
                    self.position_manager.remove_position(pos.model_id)
    
    def _get_market_context_for_logging(self, pos: Position) -> dict:
        """Get market context at ENTRY time for feedback loop learning.
        
        Feb 25 fix: Uses pos.entry_features (captured at entry) instead of
        reading cached_features_df at exit time, which gave misleading values.
        """
        try:
            # Parse hour from entry_time (handles both ISO format with T and space separator)
            entry_hour = datetime.now().hour
            if isinstance(pos.entry_time, str):
                try:
                    if 'T' in pos.entry_time:
                        entry_hour = int(pos.entry_time.split('T')[1].split(':')[0])
                    elif ' ' in pos.entry_time:
                        entry_hour = int(pos.entry_time.split(' ')[1].split(':')[0])
                except (IndexError, ValueError):
                    pass
            
            context = {
                'hour': entry_hour,
                'day_of_week': datetime.now().weekday(),
            }
            
            # Use entry-time features stored on Position (Feb 25 fix)
            ef = pos.entry_features if hasattr(pos, 'entry_features') and pos.entry_features else None
            if ef:
                if 'rsi' in ef:
                    context['rsi'] = ef['rsi']
                if 'macd' in ef:
                    context['macd'] = ef['macd']
                if 'bb_position' in ef:
                    context['bb_position'] = ef['bb_position']
                if 'roc' in ef:
                    context['roc'] = ef['roc']
                # ATR percentage from entry price
                if 'price' in ef and ef['price'] and ef['price'] > 0:
                    df = None
                    if hasattr(self, 'cached_features_df') and self.cached_features_df is not None and len(self.cached_features_df) > 0:
                        df = self.cached_features_df
                    if df is not None:
                        for col in ['volatility_atr', 'atr', 'ATRr_14']:
                            if col in df.columns:
                                context['atr_pct'] = float(df[col].iloc[-1]) / ef['price'] * 100
                                break
                if 'bb_width' in ef:
                    context['volatility'] = ef['bb_width']
                if 'prev_day_return' in ef:
                    context['prev_day_return'] = ef['prev_day_return']
                logger.info(f"Market context (entry-time): RSI={context.get('rsi', 'N/A'):.1f}, MACD={context.get('macd', 'N/A'):.1f}, BB={context.get('bb_position', 'N/A'):.3f}")
            else:
                # Fallback: read current cached_features_df (legacy behavior for old positions)
                logger.debug("No entry_features on position, falling back to current indicators")
                df = None
                if hasattr(self, 'cached_features_df') and self.cached_features_df is not None and len(self.cached_features_df) > 0:
                    df = self.cached_features_df
                if df is not None:
                    last_row = df.iloc[-1]
                    for col in ['momentum_rsi', 'rsi', 'RSI_14']:
                        if col in df.columns:
                            context['rsi'] = float(last_row[col]); break
                    for col in ['trend_macd_diff', 'trend_macd', 'macd', 'MACD_12_26_9']:
                        if col in df.columns:
                            context['macd'] = float(last_row[col]); break
                    for col in ['volatility_bbp', 'bb_pct_b', 'BBP_20_2.0']:
                        if col in df.columns:
                            context['bb_position'] = float(last_row[col]); break
                    for col in ['volatility_atr', 'atr', 'ATRr_14']:
                        if col in df.columns:
                            context['atr_pct'] = float(last_row[col]) / self.current_price * 100 if self.current_price else 0; break
                    for col in ['volatility_bbw', 'volatility']:
                        if col in df.columns:
                            context['volatility'] = float(last_row[col]); break
                logger.info(f"Market context (exit-time fallback): RSI={context.get('rsi', 'N/A')}, MACD={context.get('macd', 'N/A')}, BB={context.get('bb_position', 'N/A')}")
            
            # ── Enriched metrics (Mar 17, 2026) ─────────────────────────
            # 1h trend and 24h macro trend from entry features
            if hasattr(pos, 'entry_trend_1h') and pos.entry_trend_1h is not None:
                context['trend_1h'] = pos.entry_trend_1h
            if hasattr(pos, 'entry_macro_trend_24h') and pos.entry_macro_trend_24h is not None:
                context['macro_trend_24h'] = pos.entry_macro_trend_24h
            
            # All model probabilities at entry time
            if hasattr(pos, 'entry_all_probs') and pos.entry_all_probs:
                probs = pos.entry_all_probs
                context['prob_2h'] = probs.get('2h_0.5pct')
                context['prob_4h'] = probs.get('4h_0.5pct')
                context['prob_6h'] = probs.get('6h_0.5pct')
                context['prob_2h_short'] = probs.get('2h_0.5pct_SHORT')
                context['prob_4h_short'] = probs.get('4h_0.5pct_SHORT')
            
            # MFE/MAE: max favorable/adverse excursion during trade
            # Uses peak_price (highest) and trough_price (lowest) already tracked by trailing stop
            if pos.entry_price > 0:
                peak = pos.peak_price if pos.peak_price > 0 else pos.entry_price
                trough = pos.trough_price if pos.trough_price > 0 else pos.entry_price
                if pos.direction == 'LONG':
                    context['max_favorable_excursion'] = round((peak / pos.entry_price - 1) * 100, 4)
                    context['max_adverse_excursion'] = round((trough / pos.entry_price - 1) * 100, 4)
                else:  # SHORT
                    context['max_favorable_excursion'] = round((1 - trough / pos.entry_price) * 100, 4)
                    context['max_adverse_excursion'] = round((1 - peak / pos.entry_price) * 100, 4)
            
            return context
        except Exception as e:
            logger.warning(f"Error getting market context: {e}")
            return {'hour': datetime.now().hour}

    def _get_current_features(self) -> dict:
        """Get current market features for entry logging."""
        try:
            features = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'price': self.current_price,
                'hour': datetime.now().hour,
                'day_of_week': datetime.now().weekday(),
            }
            
            df = None
            if hasattr(self, 'cached_features_df') and self.cached_features_df is not None and len(self.cached_features_df) > 0:
                df = self.cached_features_df
            
            if df is not None:
                last_row = df.iloc[-1]
                
                # RSI
                for col in ['momentum_rsi', 'rsi', 'RSI_14']:
                    if col in df.columns:
                        features['rsi'] = float(last_row[col])
                        break
                
                # MACD histogram
                for col in ['trend_macd_diff', 'trend_macd', 'macd']:
                    if col in df.columns:
                        features['macd'] = float(last_row[col])
                        break
                
                # BB position
                for col in ['volatility_bbp', 'bb_pct_b']:
                    if col in df.columns:
                        features['bb_position'] = float(last_row[col])
                        break
                
                # BB width (volatility)
                for col in ['volatility_bbw']:
                    if col in df.columns:
                        features['bb_width'] = float(last_row[col])
                        break
                
                # ROC
                for col in ['momentum_roc']:
                    if col in df.columns:
                        features['roc'] = float(last_row[col])
                        break
                
                # Volume
                if 'Volume' in df.columns:
                    features['volume'] = float(last_row['Volume'])
                
            return features
        except Exception as e:
            logger.debug(f"Error getting current features: {e}")
            return {'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    def _snapshot_entry_features(self, model_id: str, direction: str, 
                                entry_price: float, probability: float):
        """Save full feature vector at trade entry time to parquet.
        
        Stores all 211 model features + trade metadata so every trade has
        the complete feature context for post-hoc analysis and filter retraining.
        File: data/trade_features.parquet
        """
        try:
            snapshot_path = os.path.join(os.path.dirname(__file__), 'data', 'trade_features.parquet')
            
            df = self.cached_features_df
            if df is None or len(df) == 0:
                logger.warning("No cached features for entry snapshot")
                return
            
            # Get the last row (current bar) with all 211 features
            last_row = df.iloc[-1].copy()
            
            # Add trade metadata columns
            row_dict = last_row.to_dict()
            row_dict['_trade_time'] = datetime.now().isoformat()
            row_dict['_model_id'] = model_id
            row_dict['_direction'] = direction
            row_dict['_entry_price'] = entry_price
            row_dict['_probability'] = probability
            
            # Add all model probabilities
            all_probs = getattr(self, '_last_model_probs', {})
            for k, v in all_probs.items():
                row_dict[f'_prob_{k}'] = v
            
            snapshot_df = pd.DataFrame([row_dict])
            snapshot_df.index = [pd.Timestamp(row_dict['_trade_time'])]
            snapshot_df.index.name = 'entry_timestamp'
            
            # Append to existing or create new
            if os.path.exists(snapshot_path):
                existing = pd.read_parquet(snapshot_path)
                snapshot_df = pd.concat([existing, snapshot_df])
            
            os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
            snapshot_df.to_parquet(snapshot_path)
            logger.info(f"[FEATURE SNAPSHOT] Saved {len(snapshot_df.columns)} features for {model_id} {direction} entry")
        except Exception as e:
            logger.warning(f"Failed to save feature snapshot: {e}")

    def _log_entry_features(self, model_id: str, direction: str, features: dict, 
                           probability: float, trend_1h: float):
        """Log entry features to CSV for learning loop analysis."""
        try:
            import os
            import csv
            
            log_dir = "signal_logs"
            os.makedirs(log_dir, exist_ok=True)
            
            log_file = os.path.join(log_dir, "entry_exit_features.csv")
            file_exists = os.path.exists(log_file)
            
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                
                if not file_exists:
                    writer.writerow([
                        'timestamp', 'event_type', 'model_id', 'direction', 'price',
                        'probability', 'trend_1h', 'rsi', 'macd', 'bb_position',
                        'bb_width', 'roc', 'volume', 'hour', 'day_of_week',
                        'pnl_pct', 'pnl_dollar', 'exit_reason'
                    ])
                
                writer.writerow([
                    features.get('timestamp', ''),
                    'ENTRY',
                    model_id,
                    direction,
                    features.get('price', ''),
                    f"{probability:.4f}" if probability else '',
                    f"{trend_1h:.4f}",
                    f"{features.get('rsi', ''):.2f}" if features.get('rsi') else '',
                    f"{features.get('macd', ''):.4f}" if features.get('macd') else '',
                    f"{features.get('bb_position', ''):.4f}" if features.get('bb_position') else '',
                    f"{features.get('bb_width', ''):.4f}" if features.get('bb_width') else '',
                    f"{features.get('roc', ''):.4f}" if features.get('roc') else '',
                    f"{features.get('volume', ''):.0f}" if features.get('volume') else '',
                    features.get('hour', ''),
                    features.get('day_of_week', ''),
                    '', '', ''  # pnl_pct, pnl_dollar, exit_reason (empty for entry)
                ])
                
            logger.debug(f"Logged entry features for {model_id}")
        except Exception as e:
            logger.debug(f"Error logging entry features: {e}")

    def _log_exit_features(self, model_id: str, direction: str, entry_features: dict,
                          exit_features: dict, pnl_pct: float, pnl_dollar: float, 
                          exit_reason: str, entry_prob: float, entry_trend: float):
        """Log exit features to CSV for learning loop analysis."""
        try:
            import os
            import csv
            
            log_dir = "signal_logs"
            log_file = os.path.join(log_dir, "entry_exit_features.csv")
            
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                
                writer.writerow([
                    exit_features.get('timestamp', ''),
                    'EXIT',
                    model_id,
                    direction,
                    exit_features.get('price', ''),
                    f"{entry_prob:.4f}" if entry_prob else '',
                    f"{entry_trend:.4f}" if entry_trend else '',
                    f"{exit_features.get('rsi', ''):.2f}" if exit_features.get('rsi') else '',
                    f"{exit_features.get('macd', ''):.4f}" if exit_features.get('macd') else '',
                    f"{exit_features.get('bb_position', ''):.4f}" if exit_features.get('bb_position') else '',
                    f"{exit_features.get('bb_width', ''):.4f}" if exit_features.get('bb_width') else '',
                    f"{exit_features.get('roc', ''):.4f}" if exit_features.get('roc') else '',
                    f"{exit_features.get('volume', ''):.0f}" if exit_features.get('volume') else '',
                    exit_features.get('hour', ''),
                    exit_features.get('day_of_week', ''),
                    f"{pnl_pct:.4f}",
                    f"{pnl_dollar:.2f}",
                    exit_reason
                ])
                
            logger.debug(f"Logged exit features for {model_id}: {exit_reason}, ${pnl_dollar:.2f}")
        except Exception as e:
            logger.debug(f"Error logging exit features: {e}")

    def _get_current_rsi(self) -> float:
        """Get current RSI value from cached features."""
        try:
            df = None
            if hasattr(self, 'cached_features_df') and self.cached_features_df is not None and len(self.cached_features_df) > 0:
                df = self.cached_features_df
            
            if df is not None:
                last_row = df.iloc[-1]
                for col in ['momentum_rsi', 'rsi', 'RSI_14']:
                    if col in df.columns:
                        return float(last_row[col])
            return None
        except Exception as e:
            logger.debug(f"Error getting RSI: {e}")
            return None

    def _get_current_bb_pct(self) -> float:
        """Get current Bollinger Band %B value from cached features."""
        try:
            df = None
            if hasattr(self, 'cached_features_df') and self.cached_features_df is not None and len(self.cached_features_df) > 0:
                df = self.cached_features_df
            
            if df is not None:
                last_row = df.iloc[-1]
                for col in ['volatility_bbp', 'bb_pct_b', 'BBP_20']:
                    if col in df.columns:
                        return float(last_row[col])
            return None
        except Exception as e:
            logger.debug(f"Error getting BB%: {e}")
            return None

    def _get_current_macd(self) -> float:
        """Get current MACD value from cached features."""
        try:
            df = None
            if hasattr(self, 'cached_features_df') and self.cached_features_df is not None and len(self.cached_features_df) > 0:
                df = self.cached_features_df
            
            if df is not None:
                last_row = df.iloc[-1]
                for col in ['trend_macd_diff', 'macd', 'MACD_12_26_9']:
                    if col in df.columns:
                        return float(last_row[col])
            return None
        except Exception as e:
            logger.debug(f"Error getting MACD: {e}")
            return None

    def _get_current_atr_pct(self) -> float:
        """Get current ATR as percentage of price from cached features."""
        try:
            df = None
            if hasattr(self, 'cached_features_df') and self.cached_features_df is not None and len(self.cached_features_df) > 0:
                df = self.cached_features_df
            
            if df is not None:
                last_row = df.iloc[-1]
                for col in ['volatility_atr', 'atr', 'ATR_14']:
                    if col in df.columns and 'Close' in df.columns:
                        return float(last_row[col]) / float(last_row['Close']) * 100
            return None
        except Exception as e:
            logger.debug(f"Error getting ATR%: {e}")
            return None

    def _get_current_volatility(self) -> float:
        """Get current volatility (std of returns) from cached features."""
        try:
            df = None
            if hasattr(self, 'cached_features_df') and self.cached_features_df is not None and len(self.cached_features_df) > 0:
                df = self.cached_features_df
            
            if df is not None and 'Close' in df.columns and len(df) >= 20:
                returns = df['Close'].pct_change().tail(20)
                return float(returns.std() * 100)
            return None
        except Exception as e:
            logger.debug(f"Error getting volatility: {e}")
            return None

    def _get_24h_change(self) -> float:
        """Get 24-hour price change percentage from cached features.
        
        Uses the cached parquet data (5-min bars). 288 bars = 24 hours.
        Returns percentage change (e.g. -2.5 means BTC dropped 2.5% in 24h).
        """
        try:
            df = None
            if hasattr(self, 'cached_features_df') and self.cached_features_df is not None and len(self.cached_features_df) > 0:
                df = self.cached_features_df
            
            if df is not None and len(df) >= MACRO_TREND_LOOKBACK_BARS:
                current_close = float(df['Close'].iloc[-1])
                past_close = float(df['Close'].iloc[-MACRO_TREND_LOOKBACK_BARS])
                if past_close > 0:
                    return (current_close / past_close - 1) * 100
            return None
        except Exception as e:
            logger.debug(f"Error getting 24h change: {e}")
            return None

    def _log_rsi_filter(self, model_id: str, direction: str, rsi: float, blocked: bool):
        """Log RSI filter decisions for learning."""
        try:
            import os
            import csv
            
            log_dir = "signal_logs"
            os.makedirs(log_dir, exist_ok=True)
            
            log_file = os.path.join(log_dir, "rsi_filter_log.csv")
            file_exists = os.path.exists(log_file)
            
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                
                if not file_exists:
                    writer.writerow([
                        'timestamp', 'model_id', 'direction', 'btc_price',
                        'rsi', 'rsi_threshold', 'blocked', 'filter_enabled'
                    ])
                
                writer.writerow([
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    model_id,
                    direction,
                    f"{self.current_price:.2f}" if self.current_price else "0",
                    f"{rsi:.2f}" if rsi else "",
                    RSI_FILTER_LONG_MAX,
                    "1" if blocked else "0",
                    "1" if RSI_FILTER_ENABLED else "0"
                ])
        except Exception as e:
            logger.debug(f"Error logging RSI filter: {e}")

    def _calculate_recent_trend(self) -> float:
        """Calculate the price trend over the last hour (12 x 5-min bars).
        
        Returns:
            Percentage change over lookback period. Positive = price rising.
        """
        try:
            lookback = TREND_FILTER_LOOKBACK_BARS
            
            # Try cached features first
            if hasattr(self, 'cached_features_df') and self.cached_features_df is not None:
                df = self.cached_features_df
                if len(df) >= lookback:
                    close_col = 'Close' if 'Close' in df.columns else 'close'
                    prices = df[close_col].tail(lookback)
                    trend = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0] * 100
                    return trend
            
            # Fallback to data_cache
            if hasattr(self, 'data_cache') and self.data_cache is not None:
                df = self.data_cache
                if len(df) >= lookback:
                    close_col = 'Close' if 'Close' in df.columns else 'close'
                    prices = df[close_col].tail(lookback)
                    trend = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0] * 100
                    return trend
            
            return 0.0
        except Exception as e:
            logger.debug(f"Error calculating trend: {e}")
            return 0.0
    
    def _calculate_macro_trend(self) -> float:
        """Calculate the price trend over 24 hours for trend-following mode.
        
        Returns:
            Percentage change over 24 hours. Positive = price rising.
        """
        try:
            lookback = MACRO_TREND_LOOKBACK_BARS
            
            # Try cached features first
            if hasattr(self, 'cached_features_df') and self.cached_features_df is not None:
                df = self.cached_features_df
                if len(df) >= lookback:
                    close_col = 'Close' if 'Close' in df.columns else 'close'
                    prices = df[close_col].tail(lookback)
                    trend = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0] * 100
                    return trend
            
            # Fallback to data_cache
            if hasattr(self, 'data_cache') and self.data_cache is not None:
                df = self.data_cache
                if len(df) >= lookback:
                    close_col = 'Close' if 'Close' in df.columns else 'close'
                    prices = df[close_col].tail(lookback)
                    trend = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0] * 100
                    return trend
            
            return 0.0
        except Exception as e:
            logger.debug(f"Error calculating macro trend: {e}")
            return 0.0

    def _log_trend_data(self, model_id: str, direction: str, trend_pct: float, 
                        trend_aligned: bool, probability: float):
        """Log trend alignment data for every signal for learning purposes.
        
        This creates a comprehensive log that can be analyzed to improve the filter.
        """
        try:
            import os
            import csv
            
            log_dir = "signal_logs"
            os.makedirs(log_dir, exist_ok=True)
            
            log_file = os.path.join(log_dir, "trend_filter_log.csv")
            file_exists = os.path.exists(log_file)
            
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                
                # Write header if new file
                if not file_exists:
                    writer.writerow([
                        'timestamp', 'model_id', 'direction', 'btc_price',
                        'trend_1h_pct', 'trend_aligned', 'filter_threshold',
                        'probability', 'signal_taken', 'filter_enabled'
                    ])
                
                writer.writerow([
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    model_id,
                    direction,
                    f"{self.current_price:.2f}" if self.current_price else "0",
                    f"{trend_pct:.4f}",
                    "1" if trend_aligned else "0",
                    f"{TREND_FILTER_SHORT_THRESHOLD}" if direction == 'SHORT' else "",
                    f"{probability:.4f}" if probability else "0",
                    "1" if trend_aligned else "0",  # signal_taken = trend_aligned for now
                    "1" if TREND_FILTER_ENABLED else "0"
                ])
                
        except Exception as e:
            logger.debug(f"Error logging trend data: {e}")

    def _update_trailing_stop(self, pos: Position, current_price: float) -> None:
        """Update trailing stop tracking for a position.
        
        Trailing stop logic (Jan 2, 2026):
        - Activates when profit reaches 0.15%
        - Sets stop to lock in 0.15% profit (at entry + 0.15%)
        - If price keeps moving favorable, trail the stop to lock in more
        - If price reverses to the stop level, exit with locked profit
        """
        if TRAILING_STOP_PCT is None:
            return  # Trailing stop disabled
        
        if pos.direction == 'LONG':
            profit_pct = (current_price / pos.entry_price - 1) * 100
            
            if not pos.trailing_stop_active:
                # Activate when profit >= activation threshold
                if TRAILING_STOP_ACTIVATION_PCT is not None and profit_pct >= TRAILING_STOP_ACTIVATION_PCT:
                    pos.trailing_stop_active = True
                    pos.peak_price = current_price
                    # Set stop to lock in 0.15% profit (entry + 0.15%)
                    pos.trailing_stop_price = pos.entry_price * (1 + TRAILING_STOP_ACTIVATION_PCT / 100)
                    logger.info(f"[{pos.model_id}] Trailing stop ACTIVATED - locking in {TRAILING_STOP_ACTIVATION_PCT}% profit at ${pos.trailing_stop_price:.2f}")
            else:
                # Already active - if price made new high, trail the stop up
                if current_price > pos.peak_price:
                    pos.peak_price = current_price
                    # Trail stop at 0.15% below peak (locks in more profit as price rises)
                    new_stop = pos.peak_price * (1 - TRAILING_STOP_PCT / 100)
                    if new_stop > pos.trailing_stop_price:
                        pos.trailing_stop_price = new_stop
                        logger.debug(f"[{pos.model_id}] New peak ${pos.peak_price:.2f}, stop raised to ${pos.trailing_stop_price:.2f}")
        
        elif pos.direction == 'SHORT':
            profit_pct = (pos.entry_price / current_price - 1) * 100
            
            if not pos.trailing_stop_active:
                # Activate when profit >= activation threshold
                if TRAILING_STOP_ACTIVATION_PCT is not None and profit_pct >= TRAILING_STOP_ACTIVATION_PCT:
                    pos.trailing_stop_active = True
                    pos.trough_price = current_price
                    # Set stop to lock in 0.15% profit (entry - 0.15%)
                    pos.trailing_stop_price = pos.entry_price * (1 - TRAILING_STOP_ACTIVATION_PCT / 100)
                    logger.info(f"[{pos.model_id}] Trailing stop ACTIVATED - locking in {TRAILING_STOP_ACTIVATION_PCT}% profit at ${pos.trailing_stop_price:.2f}")
            else:
                # Already active - if price made new low, trail the stop down
                if current_price < pos.trough_price:
                    pos.trough_price = current_price
                    # Trail stop at 0.15% above trough (locks in more profit as price falls)
                    new_stop = pos.trough_price * (1 + TRAILING_STOP_PCT / 100)
                    if new_stop < pos.trailing_stop_price:
                        pos.trailing_stop_price = new_stop
                        logger.debug(f"[{pos.model_id}] New trough ${pos.trough_price:.2f}, stop lowered to ${pos.trailing_stop_price:.2f}")
        
        # Save position state after trailing stop updates
        self.position_manager.save_positions()

    def _check_pending_close_orders(self) -> None:
        """Check if any pending close orders have been filled."""
        if not self.pending_close_orders:
            return
        
        try:
            # Get all open orders from IB
            open_orders = self.ib.openOrders()
            open_order_ids = {o.orderId for o in open_orders}
            
            # Get current IB positions to check if close orders filled
            ib_positions = self.ib.positions()
            mbt_position = 0
            for p in ib_positions:
                if p.contract.symbol == 'MBT':
                    mbt_position = p.position
            
            # Check each pending close order
            models_to_remove = []
            for model_id, order_time in list(self.pending_close_orders.items()):
                pos = self.position_manager.get_position(model_id)
                if pos is None:
                    # Position already removed
                    models_to_remove.append(model_id)
                    continue
                
                # If IB position is now flat (0) or opposite direction, close orders filled
                if mbt_position == 0:
                    logger.info(f"[{model_id}] Close order FILLED - IB position is flat")
                    self.position_manager.remove_position(model_id)
                    models_to_remove.append(model_id)
                    continue
                
                # Feb 18, 2026: Timeout stale pending close orders after 5 minutes
                # Stale pending orders block ALL exit logic (trailing stop, SL, TP)
                # for the position, causing unmanaged positions during connectivity issues
                elapsed_mins = (datetime.now() - order_time).total_seconds() / 60
                if elapsed_mins > 5:
                    logger.warning(f"[{model_id}] Pending close order TIMED OUT after {elapsed_mins:.0f} mins - clearing to re-evaluate")
                    models_to_remove.append(model_id)
            
            for model_id in models_to_remove:
                if model_id in self.pending_close_orders:
                    del self.pending_close_orders[model_id]
                    
        except Exception as e:
            logger.error(f"Error checking pending close orders: {e}")

    def check_entries(self, signals: List[Dict]) -> None:
        """Check for new entry signals (LONG and SHORT, including indicator signals)."""
        # Don't attempt entries when CME is closed - orders will be rejected
        if not is_cme_mbt_open():
            return
        
        # Block new entries if not enough time before market close
        block_entries, block_reason = should_block_new_entries()
        if block_entries:
            logger.debug(f"Blocking new entries: {block_reason}")
            return
        
        # NOTE: position counts are refreshed inside the signal loop (after each entry)
        # to prevent stale counts from allowing duplicate entries in the same direction.
        
        # =================================================================
        # OPPOSITE DIRECTION BLOCK (Feb 17, 2026)
        # Don't open ML model positions opposite to existing ML positions.
        # This prevents hedging that cancels out P&L.
        # Indicator signals are exempt (they have independent logic).
        # =================================================================
        ml_long_open = sum(1 for pos in self.position_manager.get_all_positions()
                          if pos.direction == 'LONG' and not pos.model_id.startswith('indicator_') and not pos.model_id.startswith('legacy_'))
        ml_short_open = sum(1 for pos in self.position_manager.get_all_positions()
                           if pos.direction == 'SHORT' and not pos.model_id.startswith('indicator_') and not pos.model_id.startswith('legacy_'))
        
        for signal in signals:
            model_id = signal['model_id']
            direction = signal.get('direction', 'LONG')
            is_indicator_signal = signal.get('signal_type') == 'indicator'
            
            # Block opposite-direction ML entries when ML positions are open
            if not is_indicator_signal:
                if direction == 'LONG' and ml_short_open > 0:
                    logger.info(f"[DIRECTION BLOCK] {model_id} LONG blocked: {ml_short_open} SHORT positions open")
                    continue
                if direction == 'SHORT' and ml_long_open > 0:
                    logger.info(f"[DIRECTION BLOCK] {model_id} SHORT blocked: {ml_long_open} LONG positions open")
                    continue
            
            # =================================================================
            # SL COOLDOWN (Feb 19, 2026)
            # After a SL exit, wait N bars before re-entering same direction.
            # Prevents overtrading in choppy ranges (5 SL losses in 3 hours).
            # =================================================================
            if SL_COOLDOWN_SECONDS > 0:
                last_sl = self.last_sl_time.get(direction, datetime.min)
                seconds_since_sl = (datetime.now() - last_sl).total_seconds()
                if seconds_since_sl < SL_COOLDOWN_SECONDS:
                    mins_left = (SL_COOLDOWN_SECONDS - seconds_since_sl) / 60
                    logger.info(f"[SL COOLDOWN] {model_id} {direction} blocked: {mins_left:.1f} min remaining")
                    continue
            
            # Refresh position counts each signal (prevents stale count bypass when
            # multiple signals arrive in the same bar and signal N-1 already entered)
            long_positions  = self.position_manager.count_positions_by_direction('LONG')
            short_positions = self.position_manager.count_positions_by_direction('SHORT')
            indicator_long_positions  = sum(1 for p in self.position_manager.get_all_positions() if p.model_id == 'indicator_long')
            indicator_meanrev_positions = sum(1 for p in self.position_manager.get_all_positions() if p.model_id == 'indicator_meanrev')
            indicator_trend_positions = sum(1 for p in self.position_manager.get_all_positions() if p.model_id == 'indicator_trend')

            # Check position limits based on signal type
            if model_id == 'indicator_long':
                if indicator_long_positions >= MAX_INDICATOR_LONG_POSITIONS:
                    continue
            elif model_id == 'indicator_meanrev':
                if indicator_meanrev_positions >= MAX_INDICATOR_MEANREV_POSITIONS:
                    continue
            elif model_id == 'indicator_trend':
                if indicator_trend_positions >= MAX_INDICATOR_TREND_POSITIONS:
                    continue
            elif direction == 'LONG' and long_positions >= self.max_long_positions:
                continue
            elif direction == 'SHORT' and short_positions >= self.max_short_positions:
                continue
            
            # Skip if already have position from this model
            if self.position_manager.has_position(model_id):
                continue
            
            # Get current price
            if self.current_price is None or self.current_price <= 0:
                logger.warning("No valid price for entry")
                continue
            
            # =================================================================
            # TREND ALIGNMENT FILTER (Jan 13, 2026)
            # Skip SHORT trades if price has been rising strongly
            # =================================================================
            trend_pct = self._calculate_recent_trend()
            trend_aligned = True
            filter_reason = ""
            
            if TREND_FILTER_ENABLED and direction == 'SHORT':
                if TREND_FILTER_SHORT_THRESHOLD is not None and trend_pct > TREND_FILTER_SHORT_THRESHOLD:
                    trend_aligned = False
                    filter_reason = f"SHORT blocked: 1h trend +{trend_pct:.2f}% > +{TREND_FILTER_SHORT_THRESHOLD}%"
            
            if TREND_FILTER_ENABLED and direction == 'LONG':
                if TREND_FILTER_LONG_THRESHOLD is not None and trend_pct < TREND_FILTER_LONG_THRESHOLD:
                    trend_aligned = False
                    filter_reason = f"LONG blocked: 1h trend {trend_pct:.2f}% < {TREND_FILTER_LONG_THRESHOLD}%"
            
            # Log trend data for every signal (for learning)
            self._log_trend_data(model_id, direction, trend_pct, trend_aligned, signal.get('probability', 0))
            
            if not trend_aligned:
                logger.info(f"[TREND FILTER] {filter_reason} - skipping {model_id}")
                continue
            
            # =================================================================
            # TREND-FOLLOWING MODE (Mar 12, 2026)
            # In strong trends, flip to momentum strategy
            # =================================================================
            if TREND_FOLLOW_ENABLED:
                macro_trend = self._calculate_macro_trend()
                trend_follow_mode = False
                trend_follow_reason = ""
                
                # Check if we should activate trend-following mode
                if direction == 'LONG' and macro_trend > TREND_FOLLOW_LONG_THRESHOLD:
                    trend_follow_mode = True
                    trend_follow_reason = f"TREND FOLLOW: 24h change +{macro_trend:.1f}% > +{TREND_FOLLOW_LONG_THRESHOLD}%"
                    
                    # Lower threshold for trend mode
                    if signal.get('probability', 0) < TREND_FOLLOW_LONG_MODEL_THRESHOLD:
                        logger.info(f"[TREND FOLLOW] {model_id} LONG blocked: prob {signal.get('probability', 0)*100:.0f}% < {TREND_FOLLOW_LONG_MODEL_THRESHOLD*100:.0f}% (trend mode)")
                        continue
                    
                    # Require MACD confirmation if enabled
                    if TREND_FOLLOW_REQUIRE_MACD:
                        current_macd = self._get_current_macd()
                        if current_macd is not None and current_macd < 0:
                            logger.info(f"[TREND FOLLOW] {model_id} LONG blocked: MACD {current_macd:.1f} < 0 (need bullish momentum)")
                            continue
                    
                elif direction == 'SHORT' and macro_trend < TREND_FOLLOW_SHORT_THRESHOLD:
                    trend_follow_mode = True
                    trend_follow_reason = f"TREND FOLLOW: 24h change {macro_trend:.1f}% < {TREND_FOLLOW_SHORT_THRESHOLD}%"
                    
                    # Lower threshold for trend mode
                    if signal.get('probability', 0) < TREND_FOLLOW_SHORT_MODEL_THRESHOLD:
                        logger.info(f"[TREND FOLLOW] {model_id} SHORT blocked: prob {signal.get('probability', 0)*100:.0f}% < {TREND_FOLLOW_SHORT_MODEL_THRESHOLD*100:.0f}% (trend mode)")
                        continue
                    
                    # Require MACD confirmation if enabled  
                    if TREND_FOLLOW_REQUIRE_MACD:
                        current_macd = self._get_current_macd()
                        if current_macd is not None and current_macd > 0:
                            logger.info(f"[TREND FOLLOW] {model_id} SHORT blocked: MACD {current_macd:.1f} > 0 (need bearish momentum)")
                            continue
                
                # Log trend-following activation
                if trend_follow_mode:
                    logger.info(f"[{trend_follow_reason}] - entering {model_id} {direction} with momentum")
                else:
                    # In trend-following mode, block opposite direction trades
                    if macro_trend > TREND_FOLLOW_LONG_THRESHOLD and direction == 'SHORT':
                        logger.info(f"[TREND FOLLOW] {model_id} SHORT blocked: 24h change +{macro_trend:.1f}% (uptrend - ride it)")
                        continue
                    elif macro_trend < TREND_FOLLOW_SHORT_THRESHOLD and direction == 'LONG':
                        logger.info(f"[TREND FOLLOW] {model_id} LONG blocked: 24h change {macro_trend:.1f}% (downtrend - ride it)")
                        continue
            
            # =================================================================
            # RSI FILTER (Feb 19, 2026) - Bidirectional
            # LONG: RSI < 40 = 0W/8L → block (downtrend, not a bounce)
            # SHORT: RSI > 70 = 0W/2L → block (uptrend, not a reversal)
            # =================================================================
            if RSI_FILTER_ENABLED:
                current_rsi = self._get_current_rsi()
                if current_rsi is not None:
                    if direction == 'LONG' and current_rsi < RSI_FILTER_LONG_MIN:
                        logger.info(f"[RSI FILTER] LONG blocked: RSI {current_rsi:.1f} < {RSI_FILTER_LONG_MIN} - skipping {model_id}")
                        self._log_rsi_filter(model_id, direction, current_rsi, blocked=True)
                        continue
                    elif direction == 'SHORT' and current_rsi > RSI_FILTER_SHORT_MAX:
                        logger.info(f"[RSI FILTER] SHORT blocked: RSI {current_rsi:.1f} > {RSI_FILTER_SHORT_MAX} - skipping {model_id}")
                        self._log_rsi_filter(model_id, direction, current_rsi, blocked=True)
                        continue
                    elif direction == 'SHORT' and current_rsi < RSI_FILTER_SHORT_MIN:
                        logger.info(f"[RSI FILTER] SHORT blocked: RSI {current_rsi:.1f} < {RSI_FILTER_SHORT_MIN} (oversold) - skipping {model_id}")
                        self._log_rsi_filter(model_id, direction, current_rsi, blocked=True)
                        continue
                    self._log_rsi_filter(model_id, direction, current_rsi, blocked=False)
            
            # =================================================================
            # BB% FILTER (Feb 19, 2026) - Bidirectional
            # LONG: BB% < 0.20 = 0W/7L → block (price at lower band)
            # SHORT: BB% > 0.80 = 0W/2L → block (price at upper band)
            # =================================================================
            if BB_FILTER_ENABLED:
                current_bb = self._get_current_bb_pct()
                if current_bb is not None:
                    if direction == 'LONG' and current_bb < BB_FILTER_LONG_MIN:
                        logger.info(f"[BB FILTER] LONG blocked: BB% {current_bb:.3f} < {BB_FILTER_LONG_MIN} - skipping {model_id}")
                        continue
                    elif direction == 'SHORT' and current_bb > BB_FILTER_SHORT_MAX:
                        logger.info(f"[BB FILTER] SHORT blocked: BB% {current_bb:.3f} > {BB_FILTER_SHORT_MAX} - skipping {model_id}")
                        continue
            
            # =================================================================
            # MACD FILTER (Feb 19, 2026) - New
            # LONG: MACD < -10 = 0W/6L → block (bearish momentum)
            # SHORT: MACD > 10 = 3W/5L negative avg → block (bullish momentum)
            # =================================================================
            if MACD_FILTER_ENABLED:
                current_macd = self._get_current_macd()
                if current_macd is not None:
                    if direction == 'LONG' and current_macd < MACD_FILTER_LONG_MIN:
                        logger.info(f"[MACD FILTER] LONG blocked: MACD {current_macd:.1f} < {MACD_FILTER_LONG_MIN} - skipping {model_id}")
                        continue
                    elif direction == 'SHORT' and current_macd > MACD_FILTER_SHORT_MAX:
                        logger.info(f"[MACD FILTER] SHORT blocked: MACD {current_macd:.1f} > {MACD_FILTER_SHORT_MAX} - skipping {model_id}")
                        continue
            
            # =================================================================
            # CROSS-TIMEFRAME FILTER (Mar 17, 2026)
            # 6h LONG requires at least one shorter model (2h or 4h) >= 40%
            # Prevents isolated long-horizon entries with no shorter-term support.
            # Simulation: blocks 0W/1L, saves $39.93
            # =================================================================
            if CROSS_TF_ENABLED and model_id == '6h_0.5pct' and direction == 'LONG':
                probs = getattr(self, '_last_model_probs', {})
                p2h = probs.get('2h_0.5pct', None)
                p4h = probs.get('4h_0.5pct', None)
                if p2h is not None and p4h is not None:
                    if p2h < CROSS_TF_6H_LONG_MIN and p4h < CROSS_TF_6H_LONG_MIN:
                        logger.info(f"[CROSS-TF FILTER] 6h LONG blocked: 2h={p2h*100:.0f}% 4h={p4h*100:.0f}% (both < {CROSS_TF_6H_LONG_MIN*100:.0f}%) - skipping {model_id}")
                        continue
            
            # =================================================================
            # MACRO TREND FILTER (Feb 24, 2026)
            # Block LONG in persistent bearish trend, SHORT in persistent bullish
            # 24h change < -2% blocks LONG (80% precision, +$422 net in backtest)
            # =================================================================
            if MACRO_TREND_FILTER_ENABLED:
                chg_24h = self._get_24h_change()
                if chg_24h is not None:
                    if direction == 'LONG' and chg_24h < MACRO_TREND_LONG_MIN:
                        logger.info(f"[MACRO TREND] LONG blocked: 24h change {chg_24h:+.2f}% < {MACRO_TREND_LONG_MIN}% - skipping {model_id}")
                        continue
                    elif direction == 'SHORT' and chg_24h > MACRO_TREND_SHORT_MAX:
                        logger.info(f"[MACRO TREND] SHORT blocked: 24h change {chg_24h:+.2f}% > +{MACRO_TREND_SHORT_MAX}% - skipping {model_id}")
                        continue
            
            # =================================================================
            # PULLBACK ENTRY (Feb 23, 2026)
            # Instead of entering immediately, register a pending pullback.
            # Wait for price to pull back 0.20% before entering.
            # If not filled in 6 bars, enter at market (fallback).
            # Simulation: +$1,148 P&L vs +$758 without pullback.
            # =================================================================
            is_indicator = model_id.startswith('indicator_')
            
            if PULLBACK_ENABLED and not is_indicator and model_id not in self.pending_pullbacks:
                signal_price = self.current_price
                if direction == 'LONG':
                    limit_price = signal_price * (1 - PULLBACK_PCT / 100)
                else:
                    limit_price = signal_price * (1 + PULLBACK_PCT / 100)
                
                self.pending_pullbacks[model_id] = {
                    'signal': signal,
                    'direction': direction,
                    'signal_price': signal_price,
                    'limit_price': limit_price,
                    'signal_time': datetime.now(),
                    'bars_waited': 0,
                    'trend_pct': trend_pct,
                    'dip_hit': False,
                    'dip_low': None,
                }
                logger.info(f"[PULLBACK] {model_id} {direction} registered: signal ${signal_price:.2f}, limit ${limit_price:.2f} ({PULLBACK_PCT}% pullback + {PULLBACK_BOUNCE_PCT}% bounce), wait {PULLBACK_WAIT_BARS} bars")
                continue
            elif PULLBACK_ENABLED and not is_indicator and model_id in self.pending_pullbacks:
                # Already pending — skip, check_pending_pullbacks handles it
                continue
            
            # =================================================================
            # ENTRY GATE MODEL (Mar 9, 2026)
            # GradientBoosting classifier — blocks trades with low win probability
            # =================================================================
            gate_passed, gate_prob = self._check_entry_gate(direction, signal.get('probability', 0))
            if not gate_passed:
                logger.info(f"[ENTRY GATE] {direction} blocked: gate prob {gate_prob:.0%} < {ENTRY_GATE_THRESHOLD:.0%} - skipping {model_id}")
                continue
            elif gate_prob is not None:
                logger.info(f"[ENTRY GATE] {direction} passed: gate prob {gate_prob:.0%}")
            
            # Indicator signals or pullback disabled: enter immediately
            self._execute_entry(signal, direction, trend_pct)

    def _execute_entry(self, signal: Dict, direction: str, trend_pct: float, entry_price_override: float = None) -> bool:
        """Execute an entry order. Extracted from check_entries for reuse by pullback system.
        
        Args:
            signal: The signal dict with model_id, probability, horizon, etc.
            direction: 'LONG' or 'SHORT'
            trend_pct: Current trend percentage for logging
            entry_price_override: If set, use this as the entry price (for pullback limit fills)
        
        Returns:
            True if order was placed successfully
        """
        model_id = signal['model_id']
        entry_price = entry_price_override if entry_price_override else self.current_price
        
        # Calculate TP and SL prices based on direction and signal type
        if model_id == 'indicator_long':
            target_price = entry_price * (1 + INDICATOR_LONG_TP_PCT / 100)
            stop_price = entry_price * (1 - INDICATOR_LONG_SL_PCT / 100)
            sl_pct = INDICATOR_LONG_SL_PCT
            tp_pct = INDICATOR_LONG_TP_PCT
        elif model_id == 'indicator_meanrev':
            target_price = entry_price * (1 + INDICATOR_MEANREV_TP_PCT / 100)
            stop_price = entry_price * (1 - INDICATOR_MEANREV_SL_PCT / 100)
            sl_pct = INDICATOR_MEANREV_SL_PCT
            tp_pct = INDICATOR_MEANREV_TP_PCT
        elif model_id == 'indicator_trend':
            if direction == 'LONG':
                target_price = entry_price * (1 + INDICATOR_TREND_TP_PCT / 100)
                stop_price = entry_price * (1 - INDICATOR_TREND_SL_PCT / 100)
            else:
                target_price = entry_price * (1 - INDICATOR_TREND_TP_PCT / 100)
                stop_price = entry_price * (1 + INDICATOR_TREND_SL_PCT / 100)
            sl_pct = INDICATOR_TREND_SL_PCT
            tp_pct = INDICATOR_TREND_TP_PCT
            self._last_trend_trade_bar = len(self.cached_features_df) if self.cached_features_df is not None else 0
        elif direction == 'LONG':
            target_price = entry_price * (1 + TAKE_PROFIT_PCT / 100)
            stop_price = entry_price * (1 - STOP_LOSS_PCT / 100)
            sl_pct = STOP_LOSS_PCT
            tp_pct = TAKE_PROFIT_PCT
        else:
            target_price = entry_price * (1 - TAKE_PROFIT_PCT / 100)
            stop_price = entry_price * (1 + STOP_LOSS_PCT / 100)
            sl_pct = STOP_LOSS_PCT
            tp_pct = TAKE_PROFIT_PCT
        
        # Place order
        order_result = self.place_order(direction, self.position_size)
        
        if order_result:
            order_id, fill_price = order_result
            # Use actual IB fill price for SL/TP; fall back to current_price if unavailable
            actual_entry = fill_price if fill_price and fill_price > 0 else self.current_price
            # Recompute target/stop from actual fill price (corrects for slippage)
            if model_id not in ('indicator_long', 'indicator_meanrev', 'indicator_trend'):
                if direction == 'LONG':
                    target_price = actual_entry * (1 + TAKE_PROFIT_PCT / 100)
                    stop_price   = actual_entry * (1 - STOP_LOSS_PCT / 100)
                else:
                    target_price = actual_entry * (1 - TAKE_PROFIT_PCT / 100)
                    stop_price   = actual_entry * (1 + STOP_LOSS_PCT / 100)
            entry_features = self._get_current_features()
            entry_prob = signal.get('probability', 0)
            
            macro_trend_val = self._calculate_macro_trend()
            position = Position(
                symbol=self.symbol,
                direction=direction,
                size=self.position_size,
                entry_price=actual_entry,  # Actual IB fill price (not stale WebSocket price)
                entry_time=datetime.now().isoformat(),
                model_horizon=signal['horizon'],
                model_threshold=signal['threshold'],
                target_bars=signal['target_bars'],
                target_price=target_price,
                stop_price=stop_price,
                bars_held=0,
                model_id=model_id,
                order_id=order_id,
                entry_features=entry_features,
                entry_probability=entry_prob,
                entry_trend_1h=trend_pct,
                entry_macro_trend_24h=macro_trend_val,
                entry_all_probs=getattr(self, '_last_model_probs', None),
            )
            self.position_manager.add_position(position)
            
            self._snapshot_entry_features(model_id, direction, actual_entry, entry_prob)
            self._log_entry_features(model_id, direction, entry_features, entry_prob, trend_pct)
            
            self.total_signals += 1
            
            logger.info("="*50)
            if model_id.startswith('indicator_'):
                logger.info(f"ENTRY [{model_id}]: {direction} @ ${self.current_price:.2f}")
            else:
                logger.info(f"ENTRY [{model_id}]: {direction} @ ${self.current_price:.2f}")
                logger.info(f"Probability: {signal['probability']:.1%}")
            
            if direction == 'LONG':
                logger.info(f"Target: ${target_price:.2f} (+{tp_pct}%)")
                logger.info(f"Stop: ${stop_price:.2f} (-{sl_pct}%)")
            else:
                logger.info(f"Target: ${target_price:.2f} (-{tp_pct}%)")
                logger.info(f"Stop: ${stop_price:.2f} (+{sl_pct}%)")
            logger.info(f"Max hold: {signal['target_bars']} bars ({signal['horizon']})")
            logger.info("="*50)
            return True
        return False

    def _check_entry_gate(self, direction: str, probability: float) -> tuple:
        """Check entry gate model (Mar 9, 2026).
        
        Uses a GradientBoosting classifier trained on historical trades to predict
        win probability based on current market indicators + signal probability.
        
        Returns:
            (passed: bool, gate_prob: float or None) — True if gate passes or is disabled.
        """
        gate_model = self.entry_gate_long if direction == 'LONG' else self.entry_gate_short
        if not ENTRY_GATE_ENABLED or gate_model is None:
            return True, None
        
        try:
            current_rsi = self._get_current_rsi()
            current_bb = self._get_current_bb_pct()
            current_macd = self._get_current_macd()
            
            # Build feature vector: same order as training
            features = pd.DataFrame([{
                'entry_rsi': current_rsi if current_rsi is not None else 50.0,
                'entry_bb_position': current_bb if current_bb is not None else 0.5,
                'entry_macd': current_macd if current_macd is not None else 0.0,
                'entry_atr_pct': self._get_current_atr_pct() or 0.2,
                'entry_volatility': self._get_current_volatility() or 1.0,
                'entry_hour': datetime.now().hour,
                'entry_day_of_week': datetime.now().weekday(),
                'entry_probability': probability,
            }])
            
            gate_prob = gate_model.predict_proba(features)[0][1]
            passed = gate_prob >= ENTRY_GATE_THRESHOLD
            return passed, gate_prob
        except Exception as e:
            logger.warning(f"Entry gate error: {e} — allowing trade")
            return True, None

    def _check_entry_filters(self, model_id: str, direction: str) -> tuple:
        """Re-check RSI/BB%/MACD filters using current market data.
        
        Returns:
            (passed: bool, reasons: list of str) — True if filters pass, False if blocked.
        """
        reasons = []
        
        if RSI_FILTER_ENABLED:
            current_rsi = self._get_current_rsi()
            if current_rsi is not None:
                if direction == 'LONG' and current_rsi < RSI_FILTER_LONG_MIN:
                    reasons.append(f"RSI {current_rsi:.1f}<{RSI_FILTER_LONG_MIN}")
                elif direction == 'SHORT' and current_rsi > RSI_FILTER_SHORT_MAX:
                    reasons.append(f"RSI {current_rsi:.1f}>{RSI_FILTER_SHORT_MAX}")
        
        if BB_FILTER_ENABLED:
            current_bb = self._get_current_bb_pct()
            if current_bb is not None:
                if direction == 'LONG' and current_bb < BB_FILTER_LONG_MIN:
                    reasons.append(f"BB {current_bb:.2f}<{BB_FILTER_LONG_MIN}")
                elif direction == 'SHORT' and current_bb > BB_FILTER_SHORT_MAX:
                    reasons.append(f"BB {current_bb:.2f}>{BB_FILTER_SHORT_MAX}")
        
        if MACD_FILTER_ENABLED:
            current_macd = self._get_current_macd()
            if current_macd is not None:
                if direction == 'LONG' and current_macd < MACD_FILTER_LONG_MIN:
                    reasons.append(f"MACD {current_macd:.1f}<{MACD_FILTER_LONG_MIN}")
                elif direction == 'SHORT' and current_macd > MACD_FILTER_SHORT_MAX:
                    reasons.append(f"MACD {current_macd:.1f}>{MACD_FILTER_SHORT_MAX}")
        
        if TREND_FILTER_ENABLED:
            trend_pct = self._calculate_recent_trend()
            if direction == 'SHORT' and TREND_FILTER_SHORT_THRESHOLD is not None and trend_pct > TREND_FILTER_SHORT_THRESHOLD:
                reasons.append(f"Trend +{trend_pct:.2f}%>{TREND_FILTER_SHORT_THRESHOLD}%")
            if direction == 'LONG' and TREND_FILTER_LONG_THRESHOLD is not None and trend_pct < TREND_FILTER_LONG_THRESHOLD:
                reasons.append(f"Trend {trend_pct:.2f}%<{TREND_FILTER_LONG_THRESHOLD}%")
        
        if MACRO_TREND_FILTER_ENABLED:
            chg_24h = self._get_24h_change()
            if chg_24h is not None:
                if direction == 'LONG' and chg_24h < MACRO_TREND_LONG_MIN:
                    reasons.append(f"24h chg {chg_24h:+.1f}%<{MACRO_TREND_LONG_MIN}%")
                elif direction == 'SHORT' and chg_24h > MACRO_TREND_SHORT_MAX:
                    reasons.append(f"24h chg {chg_24h:+.1f}%>+{MACRO_TREND_SHORT_MAX}%")
        
        return len(reasons) == 0, reasons

    def check_pending_pullbacks(self) -> None:
        """Check pending pullback entries — dip-then-recover logic (Feb 27).
        
        Called every 2 seconds via run_price_check. Two-phase entry:
        Phase 1: Wait for price to dip to limit_price (PULLBACK_PCT below signal)
        Phase 2: Track dip low, wait for bounce of PULLBACK_BOUNCE_PCT from dip low
        
        Only enter after the bounce confirms selling pressure has eased.
        If 30 min expires without bounce → skip (no falling knife entries).
        
        CRITICAL (Feb 24 fix): Always re-check filters before entering.
        """
        if not self.pending_pullbacks:
            return
        
        if not is_cme_mbt_open():
            if self.pending_pullbacks:
                logger.info(f"[PULLBACK] Cancelling {len(self.pending_pullbacks)} pending pullbacks - market closed")
                self.pending_pullbacks.clear()
            return
        
        to_remove = []
        
        for model_id, pb in self.pending_pullbacks.items():
            direction = pb['direction']
            limit_price = pb['limit_price']
            signal_price = pb['signal_price']
            signal = pb['signal']
            
            # Cancel if position already exists
            if self.position_manager.has_position(model_id):
                logger.info(f"[PULLBACK] {model_id} cancelled - position already exists")
                to_remove.append(model_id)
                continue
            
            # Check position limits
            long_positions = self.position_manager.count_positions_by_direction('LONG')
            short_positions = self.position_manager.count_positions_by_direction('SHORT')
            if direction == 'LONG' and long_positions >= self.max_long_positions:
                logger.info(f"[PULLBACK] {model_id} cancelled - LONG position limit reached")
                to_remove.append(model_id)
                continue
            if direction == 'SHORT' and short_positions >= self.max_short_positions:
                logger.info(f"[PULLBACK] {model_id} cancelled - SHORT position limit reached")
                to_remove.append(model_id)
                continue
            
            elapsed_seconds = (datetime.now() - pb['signal_time']).total_seconds()
            wait_seconds = PULLBACK_WAIT_BARS * 5 * 60  # 6 bars * 5 min = 1800s
            
            # ---- PHASE 1: Wait for dip to limit_price ----
            if not pb['dip_hit']:
                dip_reached = False
                if self.current_price is not None:
                    if direction == 'LONG' and self.current_price <= limit_price:
                        dip_reached = True
                    elif direction == 'SHORT' and self.current_price >= limit_price:
                        dip_reached = True
                
                if dip_reached:
                    pb['dip_hit'] = True
                    pb['dip_low'] = self.current_price
                    logger.info(f"[PULLBACK] {model_id} {direction} DIP HIT @ ${self.current_price:.2f} (limit ${limit_price:.2f}) — now tracking dip low, waiting for +{PULLBACK_BOUNCE_PCT}% bounce")
                elif elapsed_seconds >= wait_seconds:
                    # Dip never reached within time window
                    if PULLBACK_MARKET_FALLBACK:
                        filters_ok, filter_reasons = self._check_entry_filters(model_id, direction)
                        if not filters_ok:
                            logger.info(f"[PULLBACK] {model_id} {direction} fallback BLOCKED by filters: {', '.join(filter_reasons)} - cancelling")
                            to_remove.append(model_id)
                            continue
                        logger.info(f"[PULLBACK] {model_id} {direction} MARKET FALLBACK after {elapsed_seconds/60:.0f}min (limit ${limit_price:.2f} not hit, entering @ ${self.current_price:.2f})")
                        if self._execute_entry(signal, direction, pb['trend_pct']):
                            to_remove.append(model_id)
                        else:
                            logger.warning(f"[PULLBACK] {model_id} market fallback entry failed - will retry")
                    else:
                        logger.info(f"[PULLBACK] {model_id} {direction} EXPIRED after {elapsed_seconds/60:.0f}min - no dip reached (limit ${limit_price:.2f}), skipping")
                        to_remove.append(model_id)
                    continue
                else:
                    # Still waiting for dip — log every ~60 seconds
                    if int(elapsed_seconds) % 60 < 3:
                        mins_left = (wait_seconds - elapsed_seconds) / 60
                        logger.info(f"[PULLBACK] {model_id} {direction} waiting for dip: limit ${limit_price:.2f}, current ${self.current_price:.2f}, {mins_left:.0f}min left")
                continue
            
            # ---- PHASE 2: Dip hit, track low, wait for bounce ----
            if direction == 'LONG':
                if self.current_price < pb['dip_low']:
                    pb['dip_low'] = self.current_price
                bounce_target = pb['dip_low'] * (1 + PULLBACK_BOUNCE_PCT / 100)
                bounce_met = self.current_price >= bounce_target
            else:  # SHORT
                if self.current_price > pb['dip_low']:
                    pb['dip_low'] = self.current_price
                bounce_target = pb['dip_low'] * (1 - PULLBACK_BOUNCE_PCT / 100)
                bounce_met = self.current_price <= bounce_target
            
            if bounce_met:
                # RE-CHECK FILTERS before entering (Feb 24 fix)
                filters_ok, filter_reasons = self._check_entry_filters(model_id, direction)
                if not filters_ok:
                    logger.info(f"[PULLBACK] {model_id} {direction} bounce confirmed BUT BLOCKED by filters: {', '.join(filter_reasons)} - cancelling")
                    to_remove.append(model_id)
                    continue
                
                elapsed_min = elapsed_seconds / 60
                logger.info(f"[PULLBACK] {model_id} {direction} BOUNCE ENTRY @ ${self.current_price:.2f} "
                           f"(dip low ${pb['dip_low']:.2f}, bounce +{PULLBACK_BOUNCE_PCT}%, "
                           f"signal was ${signal_price:.2f}, saved ${abs(self.current_price - signal_price):.2f}, "
                           f"after {elapsed_min:.1f}min)")
                if self._execute_entry(signal, direction, pb['trend_pct']):
                    to_remove.append(model_id)
                else:
                    logger.warning(f"[PULLBACK] {model_id} bounce entry failed - will retry")
                continue
            
            # Check if wait expired without bounce → skip (falling knife)
            if elapsed_seconds >= wait_seconds:
                logger.info(f"[PULLBACK] {model_id} {direction} EXPIRED after {elapsed_seconds/60:.0f}min - "
                           f"dip hit but NO bounce (dip low ${pb['dip_low']:.2f}, needed +{PULLBACK_BOUNCE_PCT}%), skipping")
                to_remove.append(model_id)
                continue
            
            # Still waiting for bounce — log every ~60 seconds
            if int(elapsed_seconds) % 60 < 3:
                mins_left = (wait_seconds - elapsed_seconds) / 60
                logger.info(f"[PULLBACK] {model_id} {direction} waiting for bounce: dip low ${pb['dip_low']:.2f}, "
                           f"target ${bounce_target:.2f}, current ${self.current_price:.2f}, {mins_left:.0f}min left")
        
        for model_id in to_remove:
            del self.pending_pullbacks[model_id]

    def run_price_check(self) -> None:
        """Fast price check for trailing stops and pullback fills (runs every 1-2 seconds).
        
        Feb 26 fix: 3-tier price fallback + emergency SL-only mode.
        If all price sources fail but last price is <60s old, still check stop losses.
        """
        price = self.get_realtime_price()
        if price:
            self.current_price = price
            self.last_price_time = datetime.now()
            # Log tick to CSV for replay/simulation
            try:
                src = 'WS' if (self.ws_price and self.ws_price_time and (datetime.now() - self.ws_price_time).total_seconds() < 5) else 'REST'
                self.tick_log_file.write(f'{self.last_price_time.strftime("%Y-%m-%d %H:%M:%S.%f")},{price:.2f},{src}\n')
            except Exception:
                pass
            # Check trailing stops on open positions
            if self.position_manager.count_positions() > 0:
                self.check_exits()
            # Check pending pullback limit fills (fast detection)
            if self.pending_pullbacks:
                self.check_pending_pullbacks()
        elif self.current_price and self.last_price_time:
            # Emergency SL-only mode: all price sources failed, use last known price
            # Only for stop-loss protection — don't activate trailing stops on stale data
            stale_secs = (datetime.now() - self.last_price_time).total_seconds()
            if stale_secs < 60 and self.position_manager.count_positions() > 0:
                if stale_secs > 10:  # Only log after 10s to avoid spam
                    logger.warning(f"EMERGENCY SL MODE: No fresh price for {stale_secs:.0f}s, "
                                 f"using last known ${self.current_price:.2f} for SL checks only")
                self.check_exits()

    def validate_data_for_signals(self, df: pd.DataFrame, current_price: Optional[float] = None) -> bool:
        """Validate data quality before generating signals."""
        is_valid, results = self.data_validator.validate_all(df, current_price)
        self.last_validation_time = datetime.now()
        
        critical_failures = [r for r in results if not r.is_valid and r.severity == 'critical']
        warnings = [r for r in results if not r.is_valid and r.severity == 'warning']
        
        if critical_failures:
            self.validation_failures += 1
            for r in critical_failures:
                logger.error(f"DATA VALIDATION FAILED: {r.check_name} - {r.message}")
            
            if self.validation_failures >= 3:
                logger.critical(f"⚠️ {self.validation_failures} CONSECUTIVE VALIDATION FAILURES - Check data pipeline!")
            return False
        
        if warnings:
            for r in warnings:
                logger.warning(f"Data validation warning: {r.check_name} - {r.message}")
        else:
            # Log that validation passed (only on new bars to avoid spam)
            logger.info("Data validation: All checks passed")
        
        self.validation_failures = 0
        return True

    def run_iteration(self) -> None:
        """Run one iteration of the trading loop using parquet + hybrid bar approach.
        
        HYBRID APPROACH (like MNQ/SPY bots):
        - Uses parquet features (correct cumulative values from years of data)
        - Updates synthetic bar with real-time Binance price
        - No cumulative offset drift issues
        """
        self.total_checks += 1
        
        # NOTE: Parquet refresh moved to new_bar block to ensure features are current
        # before generating signals (was causing 1hr stale features)
        
        # Get current price (3-tier: Binance WS → Binance REST → IB portfolio)
        price = self.get_realtime_price()
        
        if price is None:
            logger.warning("No price from any source (Binance WS + REST + IB all failed)")
            return
        
        self.current_price = price
        self.last_price_time = datetime.now()
        
        # Update synthetic bar with current price
        self._update_synthetic_bar(price)
        
        # Check if new 5-min bar
        current_bar_time = self.synthetic_bar['time']
        new_bar = self.last_bar_time is None or current_bar_time > self.last_bar_time
        
        if new_bar:
            self.last_bar_time = current_bar_time
            logger.info(f"New bar: {current_bar_time}, Price: ${self.current_price:.2f}")
            
            # CRITICAL: Refresh parquet BEFORE generating signals on new bar
            # This ensures features are up-to-date, not 1 hour stale
            self._refresh_parquet_from_binance()
            
            # Increment bars held
            self.position_manager.increment_bars_held()
            
            # Use parquet features DIRECTLY for signal generation
            # No synthetic bar needed - parquet has all 206 features correctly calculated
            # Synthetic bar only updated 2 features (return_1bar, log_return_1bar) which
            # are already in the fresh parquet. Also had timezone bug (UTC vs Paris).
            if self.parquet_features is not None and not self.parquet_features.empty:
                df = self.parquet_features.tail(1000).copy()
            else:
                # Fallback to Binance data if parquet not available
                logger.warning("Parquet features not available, using Binance fallback")
                df = self.get_binance_data(limit=300)
                if not df.empty:
                    df = self.add_features(df)
            
            if df.empty:
                logger.warning("No features available")
                return
            
            self.cached_features_df = df
            
            # VALIDATE DATA - LOG ONLY, don't block signals
            self.validate_data_for_signals(df, self.current_price)
            
            # Get ML signals
            signals = self.get_signals(df)
            
            # Get indicator-based LONG signals (runs alongside ML)
            indicator_signals = self.get_indicator_long_signals(df)
            
            # Get mean reversion signals (BB < 0.5) - DISABLED Jan 6
            meanrev_signals = self.get_indicator_meanrev_signals(df)
            
            # Get ROC + MACD trend signals (LONG and SHORT)
            trend_signals = self.get_indicator_trend_signals(df)
            
            # Combine all signals (indicator signals first as they have priority 0)
            all_signals = indicator_signals + meanrev_signals + trend_signals + signals
            
            if all_signals:
                logger.info(f"Signals: {[s['model_id'] for s in all_signals]}")
                self.check_entries(all_signals)
            
            # Log status
            positions = self.position_manager.count_positions()
            logger.info(f"Positions: {positions}/{self.max_positions}")
        
        # Log signals every 15 seconds (regardless of new bar)
        self.log_current_signals()

    def run(self) -> None:
        """Main trading loop with crash recovery and real-time price updates."""
        logger.info("Starting BTC Ensemble Bot...")
        
        # Start WebSocket for real-time price updates
        self.start_websocket()
        
        # Outer loop for crash recovery - bot will restart itself on any fatal error
        while True:
            try:
                if not self.ib.isConnected():
                    if not self.connect():
                        logger.error("Failed to connect to IB Gateway. Retrying in 30s...")
                        time.sleep(30)
                        continue
                
                consecutive_errors = 0
                max_consecutive_errors = 10  # Trigger reconnect after this many errors
                last_status_log = datetime.now()
                last_full_iteration = datetime.now()
                price_check_interval = 2  # Check price every 2 seconds for trailing stops
                full_iteration_interval = 15  # Full iteration (signals, features) every 15 seconds
                
                # Sync with existing IB positions after connection is established
                # Wait a bit for IB callbacks to populate position data
                time.sleep(3)
                self._sync_ib_positions()
                
                while True:
                    try:
                        loop_start = datetime.now()
                        
                        # WATCHDOG: Detect frozen main loop (Feb 26 fix)
                        # Feb 26: ib.sleep() hung for 54 min after IB disconnect/reconnect,
                        # causing bot to miss trailing stop activation for entire window
                        if self.last_loop_time:
                            loop_gap = (loop_start - self.last_loop_time).total_seconds()
                            if loop_gap > 30:  # Should be ~2s, >30s = frozen
                                logger.error(f"WATCHDOG: Main loop was frozen for {loop_gap:.0f}s! "
                                           f"Forcing IB reconnect to recover.")
                                self.reconnect()
                                consecutive_errors = 0
                        self.last_loop_time = loop_start
                        
                        # Check connection health and reconnect if needed
                        if not self.is_connected():
                            logger.warning("IB connection lost - attempting reconnect...")
                            self.reconnect()  # This now loops forever until success
                            consecutive_errors = 0
                        
                        # Fast price check for trailing stops (every 2 seconds)
                        self.run_price_check()
                        
                        # Full iteration with signals/features every 15 seconds
                        if (datetime.now() - last_full_iteration).total_seconds() >= full_iteration_interval:
                            self.run_iteration()
                            last_full_iteration = datetime.now()
                            consecutive_errors = 0  # Reset on successful iteration
                        
                        # Log status and sync positions every hour
                        if datetime.now() - last_status_log > timedelta(hours=1):
                            runtime = datetime.now() - self.start_time
                            hours = runtime.total_seconds() / 3600
                            logger.info("="*50)
                            logger.info("HOURLY STATUS REPORT")
                            logger.info(f"  Uptime: {hours:.1f} hours")
                            logger.info(f"  Total checks: {self.total_checks}")
                            logger.info(f"  Total signals: {self.total_signals}")
                            logger.info(f"  Open positions: {self.position_manager.count_positions()}/{self.max_positions}")
                            logger.info(f"  WebSocket connected: {self.ws_connected}")
                            logger.info("="*50)
                            
                            # Periodic position sync - detect and adopt any orphan IB positions
                            logger.info("Running periodic IB position sync...")
                            self._sync_ib_positions()
                            
                            last_status_log = datetime.now()
                        
                    except Exception as e:
                        consecutive_errors += 1
                        logger.error(f"Iteration error ({consecutive_errors}/{max_consecutive_errors}): {e}")
                        
                        # If too many consecutive errors, try reconnecting
                        if consecutive_errors >= max_consecutive_errors:
                            logger.warning("Too many consecutive errors - forcing reconnect...")
                            self.reconnect()  # This now loops forever until success
                            consecutive_errors = 0
                    
                    # Sleep for price check interval (2 seconds)
                    # Use time.sleep as primary to avoid ib.sleep() hangs (Feb 26 fix)
                    # ib.sleep() can hang indefinitely after IB disconnect/reconnect cycles
                    try:
                        time.sleep(price_check_interval)
                        # Process any pending IB events without blocking
                        if self.is_connected():
                            self.ib.sleep(0)  # Non-blocking: just process event queue
                    except Exception:
                        time.sleep(price_check_interval)
                    
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                self.disconnect()
                self.print_summary()
                return  # Only exit on explicit keyboard interrupt
            except SystemExit as e:
                logger.error(f"SystemExit caught: {e} - Restarting bot in 30 seconds...")
                try:
                    self.disconnect()
                except:
                    pass
                time.sleep(30)
            except BaseException as e:
                logger.error(f"FATAL ERROR - Bot crashed: {type(e).__name__}: {e}")
                logger.error("Restarting bot in 30 seconds...")
                try:
                    if self.ib.isConnected():
                        self.ib.disconnect()
                    self.connected = False
                except:
                    pass
                time.sleep(30)

    def print_summary(self) -> None:
        """Print session summary."""
        runtime = datetime.now() - self.start_time
        
        logger.info("="*60)
        logger.info("SESSION SUMMARY")
        logger.info("="*60)
        logger.info(f"Runtime: {runtime}")
        logger.info(f"Total checks: {self.total_checks}")
        logger.info(f"Total signals: {self.total_signals}")
        logger.info(f"Open positions: {self.position_manager.count_positions()}")


def main():
    """Run the BTC ensemble bot with V2 optimized parameters."""
    bot = BTCEnsembleBot(
        probability_threshold_long=0.55,   # Jan 22: Raised from 40% to filter low-confidence entries
        probability_threshold_short=0.55,  # Keep SHORT at 55%
        position_size=1,
        paper_trading=True,
        signal_check_interval=15,  # Check every 15 seconds for signal logging
        max_long_positions=MAX_LONG_POSITIONS,
        max_short_positions=MAX_SHORT_POSITIONS
    )
    bot.run()


if __name__ == "__main__":
    main()
