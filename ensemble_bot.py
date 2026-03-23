"""
MNQ Ensemble Trading Bot - V2 Models (Dec 20, 2025).

V2 Models trained on QQQ data with random 75/25 split:
- 2h_0.5% LONG/SHORT: 55-52% WR at 65% prob
- 4h_0.5% LONG/SHORT: 70-69% WR at 65% prob

Optimized Parameters:
- SL: 0.75%
- Trailing: 0.15% (activation 0.15%)
- Timeout: 2x horizon
- Probability threshold: 65%

Backtest Results (Nov 20 - Dec 18, 2025):
- 155 trades, 83.9% win rate with 0.75% SL
- +46% P&L in 20 trading days
- Break-even WR: 39.9%, Buffer: 43.9%

Features:
- Position persistence (survives restarts)
- Orphan position detection
- Multiple model parallel positions (2 LONG, 2 SHORT max)
- Real-time data from IB Gateway
- Contract auto-rollover for MNQ futures
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import time
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from ib_insync import IB, Stock, Future, MarketOrder, util
import ta
from trade_database import log_trade_to_db
from market_hours import MarketHours, get_cme_market_hours

import config

# Finnhub API for real-time QQQ quotes (free tier: 60 calls/min)
FINNHUB_API_KEY = os.environ.get('FINNHUB_API_KEY', '')  # Set in environment or here
MNQ_QQQ_RATIO = 41.2  # MNQ/QQQ price ratio for conversion
from feature_engineering import (
    add_time_features, add_price_features,
    add_daily_context_features, add_lagged_indicator_features,
    add_indicator_changes, get_feature_columns
)
from data_validator import DataValidator, log_alert, ValidationResult
from parquet_validator import ParquetValidator

# Setup logging with append mode to preserve logs across restarts
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/mnq_bot.log', mode='a'),  # Append mode
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# MNQ Ensemble configuration - V2 LONG models (2h + 4h horizons, 0.5% threshold)
# Optimized Jan 6, 2026: Using QQQ-trained models with lower probability threshold
# Retrained models showed 65% threshold too high - lowering to 50% for more signals
ENSEMBLE_MODELS_LONG = [
    {'horizon': '2h', 'threshold': 0.5, 'horizon_bars': 24, 'priority': 1, 'direction': 'LONG'},
    {'horizon': '4h', 'threshold': 0.5, 'horizon_bars': 48, 'priority': 2, 'direction': 'LONG'},
]

# MNQ Ensemble configuration - V2 SHORT models (2h + 4h horizons, 0.5% threshold)
ENSEMBLE_MODELS_SHORT = [
    {'horizon': '2h', 'threshold': 0.5, 'horizon_bars': 24, 'priority': 1, 'direction': 'SHORT'},
    {'horizon': '4h', 'threshold': 0.5, 'horizon_bars': 48, 'priority': 2, 'direction': 'SHORT'},
]

# Combined models (no breakout models for v2 - simpler config)
ENSEMBLE_MODELS = ENSEMBLE_MODELS_LONG + ENSEMBLE_MODELS_SHORT

# Default probability threshold - LOWERED from 65% to 50% (Jan 6, 2026)
# Analysis showed models never reaching 65% in recent market conditions
# At 50% threshold: More signals, similar win rate
DEFAULT_PROBABILITY_THRESHOLD = 0.50

# Timeout multiplier: 2x the model horizon for better win rates
# Analysis showed 2x timeout improves win rates from ~60% to ~85%
TIMEOUT_MULTIPLIER = 2.0

# Trailing stop configuration - DISABLED (Dec 24, 2025)
# Trailing stop removed to simplify exit logic - only TP/SL/timeout used
# Can be re-enabled in future with price-based trailing (no feature recalc needed)
TRAILING_STOP_PCT = 0.15  # 0.15% trailing stop distance (not used when disabled)
TRAILING_STOP_ACTIVATION_PCT = 999.0  # Set to 999% to effectively DISABLE trailing stop

# Stop loss percentage (updated Jan 7, 2026)
# Extended backtest (3 months) showed SL=0.5%, TP=1.0% gives 78% WR, +$120/month
# Previous 0.75% SL was too wide, 0.15% was too tight
STOP_LOSS_PCT = 0.50
TAKE_PROFIT_PCT = 1.0  # Added explicit TP for ML models

# Position limits per direction
MAX_LONG_POSITIONS = 2
MAX_SHORT_POSITIONS = 2

# Trend filter configuration (Jan 2026)
# Prevents SHORT trades during uptrends to avoid getting stopped out
# Tested on actual trades: "Very Fast" config saves $124 by blocking 17 SL, missing only 2 TP
TREND_FILTER_ENABLED = True
TREND_FILTER_SMA_SHORT = 5   # Short-term SMA period (bars) - fast response
TREND_FILTER_SMA_LONG = 20   # Long-term SMA period (bars)
TREND_FILTER_ROC_PERIOD = 20  # ROC lookback period (bars)
TREND_FILTER_ROC_THRESHOLD = 1.0  # Disable SHORT if ROC > this %

# Trend-following strategy configuration (Jan 2026)
# When strong trend detected, ride it with wider TP and trailing stop
TREND_FOLLOW_ENABLED = True
TREND_FOLLOW_SL_PCT = 0.50   # Wider SL to avoid noise
TREND_FOLLOW_TP_PCT = 1.50   # Larger TP to ride the trend
TREND_FOLLOW_TRAILING_PCT = 0.30  # Trailing stop to lock in profits
TREND_FOLLOW_MIN_ROC = 1.5   # Minimum ROC to trigger trend-following
TREND_FOLLOW_TIMEOUT_BARS = 96  # 8 hour timeout (longer hold)
MAX_TREND_FOLLOW_POSITIONS = 1  # Only 1 trend-following position at a time

# Indicator-based LONG strategy configuration (Jan 2026)
# Strategy: BB %B < 0.5 (mean reversion - buy when price below BB midline)
# Backtest (7-day fresh data): 534 trades/week, 50.4% WR, ~$5.6k/week P&L
# DISABLED: Bug found - stop price being set ~$200 tighter than configured (0.08% vs 0.30%)
# causing 10% WR instead of expected 52% WR. Need to fix entry price vs current_price mismatch.
INDICATOR_LONG_ENABLED = False
INDICATOR_LONG_SL_PCT = 0.30  # Wider SL for higher win rate
INDICATOR_LONG_TP_PCT = 0.75  # 2.5:1 R:R ratio
INDICATOR_LONG_TIMEOUT_BARS = 48  # 4 hour timeout
MAX_INDICATOR_LONG_POSITIONS = 2  # Separate limit for indicator LONG positions

# Signal logging configuration
SIGNAL_LOG_DIR = "signal_logs"
SIGNAL_LOG_INTERVAL = 15  # seconds

# Historical end values for cumulative features (from QQQ training data end: 2025-12-18)
# These are used to compute dynamic offsets so live data matches training scale
# Note: volume_em and volume_sma_em are NOT cumulative (they oscillate around 0)
CUMULATIVE_HIST_END_MNQ = {
    "volume_adi": 1416835090.86,
    "volume_obv": 1517480348.00,
    "volume_nvi": 25839.50,
    "volume_vpt": -221825.54,
    "others_cr": 231.38,  # Cumulative Return
}

# Use normalized models (trained with scale-invariant features)
# Last 5 days backtest: Normalized 11 trades, 1.55% P&L vs Original 9 trades, 1.03% P&L
USE_NORMALIZED_MODELS = True

# Features to normalize for scale invariance
CUMULATIVE_FEATURES = ['volume_obv', 'volume_adi', 'volume_nvi', 'volume_vpt', 'others_cr']
PRICE_FEATURES = ['volatility_atr', 'trend_macd', 'trend_macd_signal', 'trend_macd_diff',
                  'volatility_bbh', 'volatility_bbl', 'volatility_bbm', 'volatility_bbw',
                  'volatility_kch', 'volatility_kcl', 'volatility_kcw',
                  'volatility_dch', 'volatility_dcl', 'volatility_dcm', 'volatility_dcw']


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize cumulative and price-based features for scale invariance.
    
    This matches the normalization used during training of normalized models.
    """
    df = df.copy()
    
    # 1. Cumulative features: use rate of change (pct_change)
    for feat in CUMULATIVE_FEATURES:
        if feat in df.columns:
            df[f'{feat}_pct'] = df[feat].pct_change().fillna(0).clip(-0.1, 0.1)
            for lag in [1, 2, 3, 5, 10, 20, 50]:
                lag_feat = f'{feat}_lag{lag}'
                if lag_feat in df.columns:
                    df[f'{lag_feat}_pct'] = df[lag_feat].pct_change().fillna(0).clip(-0.1, 0.1)
    
    # 2. Price-based features: normalize by Close
    for feat in PRICE_FEATURES:
        if feat in df.columns and 'Close' in df.columns:
            df[f'{feat}_norm'] = df[feat] / df['Close']
            for lag in [1, 2, 3, 5, 10, 20, 50]:
                lag_feat = f'{feat}_lag{lag}'
                if lag_feat in df.columns:
                    df[f'{lag_feat}_norm'] = df[lag_feat] / df['Close']
    
    return df


class SignalLogger:
    """Logs all model signals to CSV for audit purposes."""
    
    def __init__(self, log_dir: str = SIGNAL_LOG_DIR, prefix: str = "mnq"):
        self.log_dir = log_dir
        self.prefix = prefix
        os.makedirs(log_dir, exist_ok=True)
        self.current_date = None
        self.csv_path = None
        self._init_csv()
    
    def _init_csv(self) -> None:
        """Initialize CSV file for current date."""
        today = datetime.now().strftime('%Y-%m-%d')
        if today != self.current_date:
            self.current_date = today
            self.csv_path = os.path.join(self.log_dir, f"{self.prefix}_signals_{today}.csv")
            
            # Write header if file doesn't exist
            if not os.path.exists(self.csv_path):
                header = [
                    'timestamp', 'qqq_price', 'mnq_price', 'price_source',
                    # V2 LONG models
                    'prob_2h_0.5pct', 'prob_4h_0.5pct',
                    'signal_2h_0.5pct', 'signal_4h_0.5pct',
                    # V2 SHORT models
                    'prob_2h_0.5pct_SHORT', 'prob_4h_0.5pct_SHORT',
                    'signal_2h_0.5pct_SHORT', 'signal_4h_0.5pct_SHORT',
                    'active_positions', 'rsi', 'macd', 'atr', 'bb_pct_b', 'is_market_hours'
                ]
                with open(self.csv_path, 'w') as f:
                    f.write(','.join(header) + '\n')
                logger.info(f"Created signal log: {self.csv_path}")
    
    def log_signals(self, timestamp: datetime, qqq_price: float, mnq_price: float,
                    price_source: str, probabilities: Dict[str, float], threshold: float,
                    active_positions: int, indicators: Dict[str, float] = None,
                    is_market_hours: bool = True) -> None:
        """Log signals to CSV with price source tracking."""
        self._init_csv()  # Check if we need a new file for new day
        
        # V2 Model order must match header - LONG models
        long_models = ['2h_0.5pct', '4h_0.5pct']
        # V2 SHORT models
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
            f"{qqq_price:.2f}",
            f"{mnq_price:.2f}",
            price_source,
            *long_probs,
            *long_signals,
            *short_probs,
            *short_signals,
            str(active_positions),
            rsi, macd, atr, bb_pct_b,
            '1' if is_market_hours else '0'
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
    target_price: float
    stop_price: float
    max_hold_bars: int
    model_id: str = ""  # Unique identifier for the model (e.g., "3h_0.5pct")
    order_id: Optional[int] = None
    pending_close: bool = False  # Track if close order is pending
    pending_close_time: str = ""  # When close order was placed
    # Trailing stop fields
    peak_price: float = 0.0  # Highest price since entry (for LONG)
    trough_price: float = 0.0  # Lowest price since entry (for SHORT)
    trailing_stop_active: bool = False  # Whether trailing stop is activated
    trailing_stop_price: float = 0.0  # Current trailing stop level

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'Position':
        return cls(**data)


@dataclass
class ModelSignal:
    """Signal from a single model."""
    horizon: str
    threshold: float
    probability: float
    priority: int
    target_pct: float
    max_hold_mins: int
    direction: str = "LONG"  # "LONG" or "SHORT"


class PositionManager:
    """Manages multiple parallel positions (one per model)."""

    def __init__(self, filepath: str = "ensemble_positions.json"):
        self.filepath = filepath
        self.positions: Dict[str, Position] = {}  # Key is model_id, not symbol
        self.load_positions()

    def load_positions(self) -> None:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                    for pos_id, pos_data in data.items():
                        self.positions[pos_id] = Position.from_dict(pos_data)
                logger.info(f"Loaded {len(self.positions)} positions")
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

    def add_position(self, position: Position) -> None:
        """Add position keyed by model_id."""
        self.positions[position.model_id] = position
        self.save_positions()
        logger.info(f"Added position [{position.model_id}]: {position.direction} "
                   f"{position.size} @ ${position.entry_price:.2f}")

    def remove_position(self, model_id: str) -> Optional[Position]:
        """Remove position by model_id."""
        if model_id in self.positions:
            position = self.positions.pop(model_id)
            self.save_positions()
            logger.info(f"Removed position [{model_id}]")
            return position
        return None

    def get_position(self, model_id: str) -> Optional[Position]:
        """Get position by model_id."""
        return self.positions.get(model_id)

    def has_position(self, model_id: str) -> bool:
        """Check if model has an open position."""
        return model_id in self.positions
    
    def get_all_positions(self) -> List[Position]:
        """Get all open positions."""
        return list(self.positions.values())
    
    def count_positions(self) -> int:
        """Count open positions."""
        return len(self.positions)
    
    def get_total_size(self) -> int:
        """Get total shares across all positions."""
        return sum(p.size for p in self.positions.values())


def get_front_month_expiry() -> str:
    """Get the front month futures expiry in YYYYMM format.
    
    MNQ expiries are on 3rd Friday of Mar, Jun, Sep, Dec (H, M, U, Z months).
    """
    now = datetime.now()
    year = now.year
    month = now.month
    
    # Quarterly months: Mar(3), Jun(6), Sep(9), Dec(12)
    quarterly_months = [3, 6, 9, 12]
    
    # Find next quarterly month
    for qm in quarterly_months:
        if month <= qm:
            expiry_month = qm
            expiry_year = year
            break
    else:
        # Roll to next year's March
        expiry_month = 3
        expiry_year = year + 1
    
    # If we're past the 3rd Friday of current quarter month, roll to next
    if month == expiry_month:
        # Check if we're past expiry (3rd Friday)
        import calendar
        c = calendar.Calendar(firstweekday=calendar.SUNDAY)
        monthcal = c.monthdatescalendar(expiry_year, expiry_month)
        # Find 3rd Friday
        fridays = [day for week in monthcal for day in week 
                   if day.weekday() == calendar.FRIDAY and day.month == expiry_month]
        third_friday = fridays[2]
        
        if now.date() > third_friday:
            # Roll to next quarter
            idx = quarterly_months.index(expiry_month)
            if idx == 3:  # December
                expiry_month = 3
                expiry_year += 1
            else:
                expiry_month = quarterly_months[idx + 1]
    
    return f"{expiry_year}{expiry_month:02d}"


class EnsembleTradingBot:
    """Ensemble trading bot using multiple ML models."""

    def __init__(self,
                 probability_threshold: float = 0.50,  # Lowered from 0.60 on Jan 16, 2026
                 position_size: int = 1,  # Number of contracts (MNQ) or shares (QQQ)
                 stop_loss_pct: float = 1.5,
                 paper_trading: bool = True,
                 voting_mode: str = 'parallel',  # 'parallel', 'best', 'majority', 'any'
                 signal_check_interval: int = 15,  # seconds between price checks
                 bar_interval: int = 300,  # seconds (5 min bars)
                 max_positions: int = 4,  # max concurrent positions
                 instrument: str = 'MNQ'):  # 'MNQ' for futures, 'QQQ' for stock
        
        self.probability_threshold = probability_threshold
        self.position_size = position_size
        self.stop_loss_pct = stop_loss_pct
        self.paper_trading = paper_trading
        self.voting_mode = voting_mode
        self.signal_check_interval = signal_check_interval
        self.bar_interval = bar_interval
        self.max_positions = max_positions
        self.instrument = instrument

        # IB connection
        self.ib = IB()
        self.connected = False

        # Position manager
        self.position_manager = PositionManager()

        # Load all ensemble models
        self.models = self._load_models()
        self.feature_cols = self._load_feature_columns()

        # Contract setup based on instrument
        if instrument == 'MNQ':
            expiry = get_front_month_expiry()
            self.contract = Future('MNQ', expiry, 'CME')
            self.point_value = 2  # $2 per point for MNQ
            self.symbol = 'MNQ'
            # For data, we still use QQQ (model trained on QQQ)
            self.data_contract = Stock("QQQ", "SMART", "USD")
        else:
            self.contract = Stock("QQQ", "SMART", "USD")
            self.point_value = 1  # $1 per share for stock
            self.symbol = 'QQQ'
            self.data_contract = self.contract
        
        # Data cache
        self.data_cache = pd.DataFrame()
        self.last_bar_time = None
        self.features_cache = None
        self.last_feature_update = None
        
        # Parquet-based feature cache (for correct cumulative features)
        self.parquet_features = None  # Will be loaded on startup
        self.parquet_last_time = None  # Last timestamp in parquet
        
        # Real-time price - separate tickers for trading (MNQ) and data (QQQ)
        self.current_price = None
        self.ticker = None  # For trading contract (MNQ)
        self.data_ticker = None  # For data contract (QQQ) - used for synthetic bar
        
        # Uptime tracking
        self.start_time = datetime.now()
        self.total_checks = 0
        self.total_signals = 0
        self.connection_drops = 0
        self.last_connection_drop = None
        
        # Synthetic bar tracking (for intra-bar signal generation)
        self.synthetic_bar = None  # Current incomplete bar: {open, high, low, close, volume, time}
        self.synthetic_bar_start = None
        self.features_with_synthetic = None  # Cached features with synthetic bar appended
        
        # Signal logging
        self.signal_logger = SignalLogger(prefix="mnq")
        self.last_signal_log_time = None
        
        # Dynamic cumulative feature offsets (computed on first data fetch)
        # NOTE: Offset approach is fallback - prefer parquet-based features
        self.cumulative_offsets = None  # Will be set on first data fetch
        self.use_parquet_features = True  # Use parquet for correct cumulative features
        
        # Track pending close orders to avoid duplicates
        self.pending_close_orders: Dict[str, datetime] = {}  # model_id -> order_time
        
        # Data validator for quality checks
        self.data_validator = DataValidator(model_dir="models_mnq_v2")
        self.data_validator.add_alert_callback(log_alert)
        self.validation_failures = 0  # Track consecutive validation failures
        self.last_validation_time = None

        logger.info("="*60)
        logger.info("ENSEMBLE TRADING BOT INITIALIZED")
        logger.info("="*60)
        logger.info(f"Instrument: {instrument}")
        if instrument == 'MNQ':
            logger.info(f"  Contract: MNQ {get_front_month_expiry()} (Micro Nasdaq Futures)")
            logger.info(f"  Point value: ${self.point_value}/point")
        logger.info(f"Models: {len(self.models)}")
        for m in ENSEMBLE_MODELS:
            logger.info(f"  - {m['horizon']}_{m['threshold']}% (priority {m['priority']})")
        logger.info(f"Probability threshold: {probability_threshold}")
        logger.info(f"Position size: {position_size} {'contracts' if instrument == 'MNQ' else 'shares'}")
        logger.info(f"Stop loss: {stop_loss_pct}%")
        logger.info(f"Voting mode: {voting_mode}")
        logger.info(f"Max positions: {max_positions}")
        logger.info(f"Paper trading: {paper_trading}")
        logger.info(f"Signal check interval: {signal_check_interval}s")
        logger.info(f"Bar interval: {bar_interval}s (features update)")

    def _load_models(self) -> Dict[str, any]:
        """Load all ensemble models (LONG, SHORT, and breakout).
        
        If USE_NORMALIZED_MODELS is True, loads from models_normalized/ directory.
        Breakout models are always loaded from models_breakout/ directory.
        """
        models = {}
        
        # V2 models directory (trained Dec 20, 2025 with QQQ data)
        model_dir = "models_mnq_v2"
        logger.info(f"Using V2 models from {model_dir}/")
        
        for m in ENSEMBLE_MODELS:
            direction = m.get('direction', 'LONG')
            suffix = '_SHORT' if direction == 'SHORT' else ''
            model_name = f"{m['horizon']}_{m['threshold']}pct{suffix}"
            model_path = f"{model_dir}/model_{model_name}.joblib"
            
            if os.path.exists(model_path):
                models[model_name] = {
                    'model': joblib.load(model_path),
                    'config': m
                }
                logger.info(f"Loaded model: {model_name} ({direction}) from {os.path.dirname(model_path)}")
            else:
                logger.warning(f"Model not found: {model_path}")
        return models

    def _load_feature_columns(self) -> List[str]:
        """Load feature column names from the first loaded model."""
        # Get feature columns from the first model
        if self.models:
            first_model = list(self.models.values())[0]['model']
            if hasattr(first_model, 'feature_names_in_'):
                return list(first_model.feature_names_in_)
        
        # Fallback: load from file or data
        if USE_NORMALIZED_MODELS:
            feature_path = "models_normalized/features_3h_0.5pct_normalized.txt"
            if os.path.exists(feature_path):
                with open(feature_path, 'r') as f:
                    return [line.strip() for line in f.readlines()]
        
        feature_path = f"{config.MODELS_DIR}/feature_columns.json"
        if os.path.exists(feature_path):
            with open(feature_path, 'r') as f:
                return json.load(f)
        
        # Last fallback: load from data
        df = pd.read_parquet(f"{config.DATA_DIR}/QQQ_features.parquet")
        return get_feature_columns(df)

    def _on_error(self, reqId, errorCode, errorString, contract):
        """Handle IB error events for connection tracking."""
        if errorCode == 1100:  # Connectivity lost
            self.connection_drops += 1
            self.last_connection_drop = datetime.now()
            logger.warning(f"CONNECTION LOST (drop #{self.connection_drops})")
        elif errorCode == 1102:  # Connectivity restored
            if self.last_connection_drop:
                downtime = datetime.now() - self.last_connection_drop
                logger.info(f"CONNECTION RESTORED (downtime: {downtime.total_seconds():.1f}s)")

    def connect(self) -> bool:
        """Connect to IB Gateway."""
        try:
            port = config.IB_PORT if self.paper_trading else 7496
            self.ib.connect(config.IB_HOST, port, clientId=config.IB_CLIENT_ID + 3)
            
            # Register error handler for connection tracking
            self.ib.errorEvent += self._on_error
            
            # Request delayed data if real-time not available
            self.ib.reqMarketDataType(3)  # 3 = delayed data, 4 = delayed-frozen
            logger.info("Requested delayed market data (type 3)")
            
            # Qualify trading contract
            self.ib.qualifyContracts(self.contract)
            logger.info(f"Trading contract: {self.contract}")
            
            # Qualify data contract (QQQ for model predictions)
            if self.data_contract != self.contract:
                self.ib.qualifyContracts(self.data_contract)
                logger.info(f"Data contract: {self.data_contract}")
            
            self.connected = True
            logger.info(f"Connected to IB Gateway at {config.IB_HOST}:{port}")
            
            # Subscribe to market data for trading contract (will use delayed if real-time unavailable)
            self.ticker = self.ib.reqMktData(self.contract, '', False, False)
            logger.info(f"Subscribed to market data for {self.symbol}")
            
            # Subscribe to market data for data contract (QQQ) for synthetic bar
            if self.data_contract != self.contract:
                self.data_ticker = self.ib.reqMktData(self.data_contract, '', False, False)
                logger.info(f"Subscribed to market data for QQQ (for features)")
            else:
                self.data_ticker = self.ticker
            
            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False

    def disconnect(self) -> None:
        if self.connected:
            if self.ticker:
                self.ib.cancelMktData(self.contract)
            self.ib.disconnect()
            self.connected = False
            logger.info("Disconnected from IB Gateway")
    
    def is_connected(self) -> bool:
        """Check if IB connection is active."""
        return self.ib.isConnected()
    
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
            self.connected = False
            self.ticker = None
            self.data_ticker = None
        except:
            pass
        
        # Create new IB instance (sometimes needed after disconnect)
        self.ib = IB()
        
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
    
    def get_realtime_price(self) -> Optional[float]:
        """Get current MNQ price from real-time feed (for trading)."""
        if self.ticker and self.ticker.last:
            return self.ticker.last
        elif self.ticker and self.ticker.close:
            return self.ticker.close
        return None
    
    def get_realtime_data_price(self) -> Optional[float]:
        """Get current QQQ price from real-time feed (for features/synthetic bar)."""
        if self.data_ticker and self.data_ticker.last:
            return self.data_ticker.last
        elif self.data_ticker and self.data_ticker.close:
            return self.data_ticker.close
        # Fallback to last historical bar
        if not self.data_cache.empty:
            return self.data_cache['close'].iloc[-1]
        return None
    
    def get_finnhub_qqq_price(self) -> Optional[float]:
        """Get real-time QQQ price from Finnhub API (free, 60 calls/min).
        
        Returns QQQ price or None if unavailable.
        """
        if not FINNHUB_API_KEY:
            return None
        
        try:
            url = f"https://finnhub.io/api/v1/quote?symbol=QQQ&token={FINNHUB_API_KEY}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                # 'c' is current price, 'pc' is previous close
                price = data.get('c')
                if price and price > 0:
                    return float(price)
        except Exception as e:
            logger.debug(f"Finnhub API error: {e}")
        return None
    
    def get_yahoo_mnq_price(self) -> Optional[float]:
        """Get real-time MNQ price from Yahoo Finance.
        
        MNQ futures trade nearly 24/7, so this works overnight when QQQ is closed.
        Returns MNQ price or None if unavailable.
        """
        try:
            import yfinance as yf
            mnq = yf.Ticker("MNQ=F")
            # Get latest 5-min bar
            hist = mnq.history(period="1d", interval="5m")
            if len(hist) > 0:
                price = float(hist['Close'].iloc[-1])
                if price > 0:
                    return price
        except Exception as e:
            logger.debug(f"Yahoo MNQ price error: {e}")
        return None
    
    def is_us_market_hours(self) -> bool:
        """Check if US stock market is currently open (9:30-16:00 ET)."""
        from datetime import datetime
        import pytz
        
        eastern = pytz.timezone('US/Eastern')
        now_et = datetime.now(eastern)
        
        # Weekend check
        if now_et.weekday() >= 5:
            return False
        
        # Market hours: 9:30 AM - 4:00 PM ET
        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now_et <= market_close
    
    def get_live_mnq_price(self) -> Optional[float]:
        """Get best available MNQ price estimate.
        
        Priority during US market hours (9:30-16:00 ET):
        1. Finnhub real-time QQQ * ratio (fast, reliable)
        2. IB historical QQQ last bar * ratio (0-5 min old)
        3. IB delayed MNQ ticker (15 min old)
        
        Priority outside market hours (overnight/pre-market):
        1. Yahoo Finance MNQ=F (actual futures price, trades 24/7)
        2. IB delayed MNQ ticker
        """
        # Check if we're in US market hours
        if self.is_us_market_hours():
            # During market hours: use Finnhub QQQ (faster, more reliable)
            qqq_price = self.get_finnhub_qqq_price()
            if qqq_price:
                mnq_price = qqq_price * MNQ_QQQ_RATIO
                logger.debug(f"Finnhub QQQ: ${qqq_price:.2f} -> MNQ: ${mnq_price:.2f}")
                return mnq_price
            
            # Fallback to IB historical (0-5 min old)
            if not self.data_cache.empty:
                qqq_price = self.data_cache['close'].iloc[-1]
                mnq_price = qqq_price * MNQ_QQQ_RATIO
                logger.debug(f"IB historical QQQ: ${qqq_price:.2f} -> MNQ: ${mnq_price:.2f}")
                return mnq_price
        else:
            # Outside market hours: use Yahoo MNQ=F (actual futures price)
            mnq_price = self.get_yahoo_mnq_price()
            if mnq_price:
                logger.debug(f"Yahoo MNQ=F (overnight): ${mnq_price:.2f}")
                return mnq_price
        
        # Last resort: IB delayed ticker (15 min old)
        return self.get_realtime_price()

    def check_orphan_positions(self) -> None:
        """Check for orphan positions (including external ones) and stale tracked positions."""
        logger.info(f"Checking for {self.symbol} positions...")
        
        ib_positions = self.ib.positions()
        ib_position_map = {}
        
        for pos in ib_positions:
            if pos.contract.symbol == self.symbol and pos.position != 0:
                ib_position_map[self.symbol] = {
                    'size': pos.position,
                    'avg_cost': pos.avgCost
                }
                logger.info(f"Found {self.symbol} in IB: {pos.position} contracts @ ${pos.avgCost:.2f}")

        # Check for positions in IB not tracked by us - adopt as orphans
        for symbol, ib_pos in ib_position_map.items():
            has_tracked = any(p.symbol == symbol for p in self.position_manager.get_all_positions())
            
            if not has_tracked and ib_pos['size'] != 0:
                # For futures, avgCost is total cost (price * multiplier), need to divide
                entry_price = ib_pos['avg_cost']
                if self.instrument == 'MNQ':
                    entry_price = ib_pos['avg_cost'] / self.point_value  # Divide by multiplier (2)
                
                logger.warning(f"ORPHAN: {symbol} {ib_pos['size']} @ ${entry_price:.2f}")
                
                direction = 'long' if ib_pos['size'] > 0 else 'short'
                orphan = Position(
                    symbol=symbol,
                    direction=direction,
                    size=abs(int(ib_pos['size'])),
                    entry_price=entry_price,
                    entry_time=datetime.now().isoformat(),
                    model_horizon="orphan",
                    model_threshold=0,
                    target_price=entry_price * 1.005,
                    stop_price=entry_price * (1 - self.stop_loss_pct / 100),
                    max_hold_bars=48,
                    model_id="orphan"
                )
                self.position_manager.add_position(orphan)

        # Check for stale positions (tracked but no longer in IB)
        for model_id in list(self.position_manager.positions.keys()):
            pos = self.position_manager.get_position(model_id)
            if pos and pos.symbol == self.symbol and self.symbol not in ib_position_map:
                logger.warning(f"STALE: {model_id} no longer in IB, removing")
                self.position_manager.remove_position(model_id)
        
        logger.info(f"Bot tracking {self.position_manager.count_positions()} positions")

    def load_parquet_features(self) -> bool:
        """Load pre-computed features from parquet file.
        
        This provides correct cumulative feature values computed from full history.
        Returns True if successful, False otherwise.
        """
        try:
            parquet_path = 'data/QQQ_features.parquet'
            self.parquet_features = pd.read_parquet(parquet_path)
            self.parquet_last_time = self.parquet_features.index[-1]
            self.parquet_last_refresh = datetime.now()
            logger.info(f"Loaded parquet features: {len(self.parquet_features)} bars, ends at {self.parquet_last_time}")
            return True
        except Exception as e:
            logger.error(f"Failed to load parquet features: {e}")
            self.use_parquet_features = False
            return False
    
    def _log_parquet_refresh(self, status: str, details: dict) -> None:
        """Log parquet refresh to dedicated log file for tracking reliability."""
        try:
            log_path = 'logs/parquet_refresh.log'
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            log_entry = f"{timestamp} | {status} | "
            log_entry += " | ".join([f"{k}={v}" for k, v in details.items()])
            log_entry += "\n"
            
            with open(log_path, 'a') as f:
                f.write(log_entry)
        except Exception as e:
            logger.debug(f"Failed to write parquet refresh log: {e}")
    
    def refresh_parquet_from_yahoo(self) -> bool:
        """Refresh parquet with latest data from Yahoo Finance.
        
        IMPORTANT: Uses QQQ data during market hours to match training data.
        Models were trained on QQQ with proper volume (~200K/bar).
        NQ=F has much lower volume (~1.5K/bar) which breaks volume-based features.
        
        During market hours (9:30-16:00 ET): Use QQQ (matches training)
        Outside market hours: Use NQ=F converted to QQQ scale (for 24/7 coverage)
        
        Called every 5 minutes to keep features accurate.
        Returns True if successful.
        """
        start_time = datetime.now()
        
        try:
            import yfinance as yf
            import pytz
            
            # Check if market is open (9:30-16:00 ET)
            et = pytz.timezone('US/Eastern')
            now_et = datetime.now(et)
            market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
            is_market_hours = market_open <= now_et <= market_close and now_et.weekday() < 5
            
            logger.info("=" * 60)
            if is_market_hours:
                logger.info("PARQUET REFRESH: Fetching QQQ from Yahoo Finance (market hours)")
            else:
                logger.info("PARQUET REFRESH: Fetching NQ Futures from Yahoo Finance (overnight)")
            logger.info("=" * 60)
            
            # Load existing parquet OHLCV
            parquet_path = 'data/QQQ_features.parquet'
            existing = pd.read_parquet(parquet_path)
            last_time = existing.index[-1]
            existing_bars = len(existing)
            
            logger.info(f"Existing parquet: {existing_bars} bars, ends at {last_time}")
            
            # During market hours: Use QQQ (matches training data with proper volume)
            # Outside market hours: Use NQ=F (trades 24/7) converted to QQQ scale
            if is_market_hours:
                # PRIMARY: QQQ during market hours - matches training data exactly
                qqq = yf.Ticker('QQQ')
                new_data = qqq.history(period='5d', interval='5m')
                data_source = 'QQQ'
                
                if len(new_data) == 0:
                    logger.warning("PARQUET REFRESH: No QQQ data, trying NQ=F fallback")
                    nq = yf.Ticker('NQ=F')
                    new_data = nq.history(period='5d', interval='5m')
                    if len(new_data) > 0:
                        # Convert NQ prices to QQQ scale, scale volume too
                        for col in ['Open', 'High', 'Low', 'Close']:
                            new_data[col] = new_data[col] / MNQ_QQQ_RATIO
                        # Scale volume: NQ avg ~1500, QQQ avg ~200000, ratio ~133
                        new_data['Volume'] = new_data['Volume'] * 133
                        data_source = 'NQ=F (converted)'
                        logger.info(f"Using NQ=F fallback, converted to QQQ scale")
            else:
                # OVERNIGHT: NQ=F (trades 24/7) converted to QQQ scale
                nq = yf.Ticker('NQ=F')
                new_data = nq.history(period='5d', interval='5m')
                data_source = 'NQ=F'
                
                if len(new_data) > 0:
                    # Convert NQ prices to QQQ scale
                    for col in ['Open', 'High', 'Low', 'Close']:
                        new_data[col] = new_data[col] / MNQ_QQQ_RATIO
                    # Scale volume: NQ avg ~1500, QQQ avg ~200000, ratio ~133
                    new_data['Volume'] = new_data['Volume'] * 133
                    logger.info(f"Converted NQ Futures to QQQ scale (prices ÷{MNQ_QQQ_RATIO}, volume ×133)")
            
            if len(new_data) == 0:
                self._log_parquet_refresh("FAILED", {
                    "reason": "no_yahoo_data",
                    "existing_bars": existing_bars,
                    "last_time": str(last_time)
                })
                return False
            
            logger.info(f"Data source: {data_source}")
            
            # Convert Yahoo data to Paris timezone, then make naive for storage
            # Yahoo returns UTC timestamps, parquet stores Paris time as naive
            if new_data.index.tz is not None:
                paris_tz = pytz.timezone('Europe/Paris')
                new_data.index = new_data.index.tz_convert(paris_tz).tz_localize(None)
            new_data = new_data.rename(columns={'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'})
            new_data = new_data[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Ensure parquet index is also timezone-naive for comparison
            if hasattr(last_time, 'tzinfo') and last_time.tzinfo is not None:
                last_time = last_time.replace(tzinfo=None)
            
            yahoo_latest = new_data.index[-1]
            logger.info(f"Yahoo {data_source} data: {len(new_data)} bars, latest at {yahoo_latest}")
            
            # Filter to only new bars
            new_bars = new_data[new_data.index > last_time]
            
            if len(new_bars) == 0:
                logger.info(f"PARQUET REFRESH: Already up to date (ends at {last_time})")
                self._log_parquet_refresh("UP_TO_DATE", {
                    "existing_bars": existing_bars,
                    "last_time": str(last_time),
                    "yahoo_latest": str(yahoo_latest)
                })
                return True
            
            logger.info(f"Found {len(new_bars)} new bars to add (from {new_bars.index[0]} to {new_bars.index[-1]})")
            
            # Get OHLCV from existing parquet
            existing_ohlcv = existing[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Combine
            combined_ohlcv = pd.concat([existing_ohlcv, new_bars])
            combined_ohlcv = combined_ohlcv[~combined_ohlcv.index.duplicated(keep='first')]
            combined_ohlcv = combined_ohlcv.sort_index()
            
            logger.info(f"Combined OHLCV: {len(combined_ohlcv)} bars")
            
            # Recompute all features
            logger.info("Recomputing features...")
            df = ta.add_all_ta_features(
                combined_ohlcv, open='Open', high='High', low='Low',
                close='Close', volume='Volume', fillna=True
            )
            
            df = add_time_features(df)
            df = add_price_features(df)
            df = add_daily_context_features(df)
            df = add_lagged_indicator_features(df, config.LOOKBACK_PERIODS)
            df = add_indicator_changes(df)
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            df = df.replace([np.inf, -np.inf], 0)
            
            # VALIDATE before saving - comprehensive check of all data
            validator = ParquetValidator(bot_type='mnq')
            is_valid, issues = validator.validate_before_save(df)
            
            if not is_valid:
                # Attempt repair
                df = validator.repair(df, issues)
                is_valid, issues = validator.validate_before_save(df)
                
                if not is_valid:
                    critical = [i for i in issues if i['severity'] == 'critical']
                    logger.error(f"Parquet validation failed: {len(critical)} critical issues")
                    for issue in critical[:3]:
                        logger.error(f"  - {issue['message']}")
                    return False
            
            # Save validated parquet
            df.to_parquet(parquet_path)
            
            # Reload into memory
            self.parquet_features = df
            self.parquet_last_time = df.index[-1]
            self.parquet_last_refresh = datetime.now()
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            logger.info("=" * 60)
            logger.info(f"PARQUET REFRESH: SUCCESS")
            logger.info(f"  New bars added: {len(new_bars)}")
            logger.info(f"  Total bars: {len(df)}")
            logger.info(f"  Now ends at: {self.parquet_last_time}")
            logger.info(f"  Elapsed time: {elapsed:.1f}s")
            logger.info("=" * 60)
            
            # Log to dedicated file
            self._log_parquet_refresh("SUCCESS", {
                "new_bars": len(new_bars),
                "total_bars": len(df),
                "prev_end": str(last_time),
                "new_end": str(self.parquet_last_time),
                "elapsed_sec": f"{elapsed:.1f}"
            })
            
            # Update cumulative offset values for fallback
            cumulative_feats = ['volume_adi', 'volume_obv', 'volume_nvi', 'volume_vpt', 'others_cr']
            for feat in cumulative_feats:
                if feat in df.columns:
                    CUMULATIVE_HIST_END_MNQ[feat] = df[feat].iloc[-1]
            
            return True
            
        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.error(f"PARQUET REFRESH: FAILED - {e}")
            self._log_parquet_refresh("ERROR", {
                "error": str(e),
                "elapsed_sec": f"{elapsed:.1f}"
            })
            return False
    
    def maybe_refresh_parquet(self) -> None:
        """Check if parquet needs refresh (hourly) and refresh if needed."""
        if not self.use_parquet_features:
            return
        
        now = datetime.now()
        
        # Check if we have a last refresh time
        if not hasattr(self, 'parquet_last_refresh') or self.parquet_last_refresh is None:
            self.parquet_last_refresh = now
            return
        
        # Refresh every 5 minutes (was hourly - more frequent = better signal accuracy)
        mins_since_refresh = (now - self.parquet_last_refresh).total_seconds() / 60
        if mins_since_refresh >= 5.0:
            logger.info(f"Parquet refresh due ({mins_since_refresh:.1f} mins since last refresh)")
            self.refresh_parquet_from_yahoo()

    def get_historical_data(self, duration: str = "7 D", force_refresh: bool = False) -> pd.DataFrame:
        """Fetch historical data from IB with caching.
        
        Note: 7 days needed for prev_5day_return feature calculation.
        """
        now = datetime.now()
        
        # Check if we need to refresh (new 5-min bar)
        current_bar_time = now.replace(second=0, microsecond=0)
        current_bar_time = current_bar_time.replace(minute=(current_bar_time.minute // 5) * 5)
        
        if not force_refresh and self.last_bar_time == current_bar_time and not self.data_cache.empty:
            logger.debug("Using cached data (same 5-min bar)")
            return self.data_cache
        
        try:
            # Use data_contract (QQQ) for historical data - model trained on QQQ
            bars = self.ib.reqHistoricalData(
                self.data_contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting="5 mins",
                whatToShow="TRADES",
                useRTH=False,
                formatDate=1
            )

            if not bars:
                return self.data_cache if not self.data_cache.empty else pd.DataFrame()

            df = util.df(bars)
            df.set_index('date', inplace=True)
            df.index = pd.to_datetime(df.index, utc=True).tz_convert('US/Eastern').tz_localize(None)
            
            # Update cache
            self.data_cache = df
            self.last_bar_time = current_bar_time
            logger.info(f"Refreshed data cache: {len(df)} bars")
            return df

        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return self.data_cache if not self.data_cache.empty else pd.DataFrame()

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction."""
        try:
            df = df.rename(columns={
                'open': 'Open', 'high': 'High', 'low': 'Low',
                'close': 'Close', 'volume': 'Volume'
            })

            df = ta.add_all_ta_features(
                df, open="Open", high="High", low="Low",
                close="Close", volume="Volume", fillna=True
            )

            df = add_time_features(df)
            df = add_price_features(df)
            df = add_daily_context_features(df)
            df = add_lagged_indicator_features(df, config.LOOKBACK_PERIODS)
            df = add_indicator_changes(df)

            df = df.fillna(0).replace([np.inf, -np.inf], 0)
            
            # Apply dynamic cumulative feature offsets to match historical training data scale
            # On first call, compute offsets based on first bar's values
            # offset = historical_end - live_first_bar (so live continues from historical)
            if self.cumulative_offsets is None:
                self.cumulative_offsets = {}
                for feat, hist_end in CUMULATIVE_HIST_END_MNQ.items():
                    if feat in df.columns:
                        live_first = df[feat].iloc[0]  # First bar's value
                        self.cumulative_offsets[feat] = hist_end - live_first
                        logger.info(f"Cumulative offset for {feat}: {self.cumulative_offsets[feat]:.2f} (hist_end={hist_end:.2f}, live_first={live_first:.2f})")
            
            # Apply offsets to cumulative features and their lagged versions
            if self.cumulative_offsets:
                for base_feat, offset in self.cumulative_offsets.items():
                    if base_feat in df.columns:
                        df[base_feat] = df[base_feat] + offset
                    for lag in [1, 2, 3, 5, 10, 20, 50]:
                        lag_feat = f'{base_feat}_lag{lag}'
                        if lag_feat in df.columns:
                            df[lag_feat] = df[lag_feat] + offset
            
            return df

        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return df

    def update_synthetic_bar(self, price: float) -> None:
        """Update the synthetic (incomplete) bar with new price tick.
        
        This creates a temporary bar that updates every 15 seconds,
        allowing signal generation before the 5-min bar completes.
        """
        now = datetime.now()
        bar_start = now.replace(second=0, microsecond=0)
        bar_start = bar_start.replace(minute=(bar_start.minute // 5) * 5)
        
        # Check if we're in a new bar period
        if self.synthetic_bar_start != bar_start:
            # New bar started - reset synthetic bar
            self.synthetic_bar = {
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': 0,  # We don't have real-time volume
                'time': bar_start
            }
            self.synthetic_bar_start = bar_start
            self.features_with_synthetic = None  # Invalidate cache
            logger.debug(f"New synthetic bar started at {bar_start}")
        else:
            # Update existing synthetic bar
            if self.synthetic_bar:
                self.synthetic_bar['high'] = max(self.synthetic_bar['high'], price)
                self.synthetic_bar['low'] = min(self.synthetic_bar['low'], price)
                self.synthetic_bar['close'] = price
                self.features_with_synthetic = None  # Invalidate cache

    def get_features_with_synthetic_bar(self) -> pd.DataFrame:
        """Get features DataFrame with synthetic bar appended.
        
        HYBRID APPROACH:
        - If parquet features available: use parquet (correct cumulative features)
          and append synthetic bar for real-time updates
        - Fallback: use IB data cache with offset correction
        """
        if self.synthetic_bar is None:
            return pd.DataFrame()
        
        # PREFERRED: Use parquet features (correct cumulative values)
        if self.use_parquet_features and self.parquet_features is not None:
            logger.debug(f"Using PARQUET path (parquet_last_time={self.parquet_last_time})")
            # Get last N bars from parquet for context
            df = self.parquet_features.tail(500).copy()
            
            # Check if synthetic bar time is after parquet end
            synthetic_time = self.synthetic_bar['time']
            
            # If synthetic bar is newer than parquet, append it
            if synthetic_time > self.parquet_last_time:
                # Copy the last parquet row as base for synthetic bar
                # This preserves all feature values (MACD, ATR, etc.)
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
        
        # FALLBACK: Use IB data cache with offset correction
        logger.debug("Using FALLBACK IB data cache path (parquet not available)")
        if self.data_cache.empty:
            return pd.DataFrame()
        
        # Create synthetic bar row
        synthetic_row = pd.DataFrame([{
            'open': self.synthetic_bar['open'],
            'high': self.synthetic_bar['high'],
            'low': self.synthetic_bar['low'],
            'close': self.synthetic_bar['close'],
            'volume': self.synthetic_bar.get('volume', self.data_cache['volume'].iloc[-1]),
            'average': self.synthetic_bar['close'],
            'barCount': 1
        }], index=[self.synthetic_bar['time']])
        
        # Append to cached data (drop last bar if it's the same time)
        df = self.data_cache.copy()
        if df.index[-1] == self.synthetic_bar['time']:
            df = df.iloc[:-1]
        df = pd.concat([df, synthetic_row])
        
        # Prepare features (with offset correction)
        df = self.prepare_features(df)
        return df
    
    def _update_features_for_new_bar(self, df: pd.DataFrame) -> pd.DataFrame:
        """Update features for the last bar using existing parquet features.
        
        IMPORTANT: We do NOT recalculate RSI/MACD/ATR here because:
        1. The parquet has correct values computed from full NQ Futures history
        2. Recalculating from mixed data sources causes discrepancies
        3. The synthetic bar inherits feature values from the last parquet row
        
        Only update features that directly depend on the current bar's price.
        """
        try:
            last_idx = df.index[-1]
            
            # Update only simple price-based features that don't need history
            if 'Close' in df.columns:
                close = df.loc[last_idx, 'Close']
                
                # Update returns (these are simple calculations)
                if len(df) > 1 and 'returns' in df.columns:
                    prev_close = df['Close'].iloc[-2]
                    if prev_close != 0:
                        df.loc[last_idx, 'returns'] = (close - prev_close) / prev_close
                
                # DO NOT recalculate momentum_rsi, trend_macd, volatility_atr
                # These should come from the parquet's last row (already copied)
                # Recalculating them from different data sources causes discrepancies
            
            return df
            
        except Exception as e:
            logger.error(f"Error updating features for new bar: {e}")
            return df

    def validate_data_for_signals(self, df: pd.DataFrame, current_price: Optional[float] = None) -> bool:
        """Validate data quality before generating signals.
        
        Returns True if data is valid for signal generation, False otherwise.
        Logs warnings/errors and tracks consecutive failures.
        """
        is_valid, results = self.data_validator.validate_all(df, current_price)
        self.last_validation_time = datetime.now()
        
        # Log validation results
        critical_failures = [r for r in results if not r.is_valid and r.severity == 'critical']
        warnings = [r for r in results if not r.is_valid and r.severity == 'warning']
        
        if critical_failures:
            self.validation_failures += 1
            for r in critical_failures:
                logger.error(f"DATA VALIDATION FAILED: {r.check_name} - {r.message}")
            
            # Alert if too many consecutive failures
            if self.validation_failures >= 3:
                logger.critical(f"⚠️ {self.validation_failures} CONSECUTIVE VALIDATION FAILURES - Check data pipeline!")
            
            return False
        
        if warnings:
            for r in warnings:
                logger.warning(f"Data validation warning: {r.check_name} - {r.message}")
        else:
            # Log that validation passed (only on new bars to avoid spam)
            logger.info("Data validation: All checks passed")
        
        # Reset failure counter on success
        self.validation_failures = 0
        return True

    def detect_trend(self, df: pd.DataFrame) -> dict:
        """Detect market trend using SMA crossover and ROC.
        
        Returns dict with:
        - trend: 'UPTREND', 'DOWNTREND', or 'SIDEWAYS'
        - allow_short: True if SHORT trades are allowed
        - allow_long: True if LONG trades are allowed
        - reason: Explanation for the trend detection
        """
        result = {
            'trend': 'SIDEWAYS',
            'allow_short': True,
            'allow_long': True,
            'reason': ''
        }
        
        if not TREND_FILTER_ENABLED:
            result['reason'] = 'Trend filter disabled'
            return result
        
        if len(df) < TREND_FILTER_SMA_LONG + 1:
            result['reason'] = f'Not enough data for trend detection (need {TREND_FILTER_SMA_LONG} bars)'
            return result
        
        try:
            current_price = df['Close'].iloc[-1]
            sma_short = df['Close'].rolling(TREND_FILTER_SMA_SHORT).mean().iloc[-1]
            sma_long = df['Close'].rolling(TREND_FILTER_SMA_LONG).mean().iloc[-1]
            
            # Calculate ROC
            roc_period = min(TREND_FILTER_ROC_PERIOD, len(df) - 1)
            roc = (current_price / df['Close'].iloc[-roc_period-1] - 1) * 100
            
            # Determine trend
            if current_price > sma_short > sma_long:
                result['trend'] = 'STRONG_UPTREND'
                result['allow_short'] = False
                result['reason'] = f'Price > SMA({TREND_FILTER_SMA_SHORT}) > SMA({TREND_FILTER_SMA_LONG})'
            elif current_price > sma_short:
                result['trend'] = 'UPTREND'
                if roc > TREND_FILTER_ROC_THRESHOLD:
                    result['allow_short'] = False
                    result['reason'] = f'Price > SMA({TREND_FILTER_SMA_SHORT}), ROC={roc:.1f}% > {TREND_FILTER_ROC_THRESHOLD}%'
                else:
                    result['reason'] = f'Price > SMA({TREND_FILTER_SMA_SHORT}), ROC={roc:.1f}%'
            elif current_price < sma_short < sma_long:
                result['trend'] = 'STRONG_DOWNTREND'
                result['allow_long'] = False
                result['reason'] = f'Price < SMA({TREND_FILTER_SMA_SHORT}) < SMA({TREND_FILTER_SMA_LONG})'
            elif current_price < sma_short:
                result['trend'] = 'DOWNTREND'
                if roc < -TREND_FILTER_ROC_THRESHOLD:
                    result['allow_long'] = False
                    result['reason'] = f'Price < SMA({TREND_FILTER_SMA_SHORT}), ROC={roc:.1f}% < -{TREND_FILTER_ROC_THRESHOLD}%'
                else:
                    result['reason'] = f'Price < SMA({TREND_FILTER_SMA_SHORT}), ROC={roc:.1f}%'
            else:
                result['trend'] = 'SIDEWAYS'
                result['reason'] = 'No clear trend'
            
            return result
            
        except Exception as e:
            logger.error(f"Error detecting trend: {e}")
            result['reason'] = f'Error: {e}'
            return result

    def get_ensemble_signals(self, df: pd.DataFrame) -> List[ModelSignal]:
        """Get signals from all models in the ensemble."""
        signals = []
        latest = df.iloc[-1:]

        # Ensure all features exist
        for f in self.feature_cols:
            if f not in latest.columns:
                latest[f] = 0

        X = latest[self.feature_cols]
        
        # Detect trend and apply filter
        trend_info = self.detect_trend(df)
        if trend_info['trend'] != 'SIDEWAYS':
            logger.info(f"  Trend: {trend_info['trend']} - {trend_info['reason']}")

        for model_name, model_data in self.models.items():
            try:
                model = model_data['model']
                cfg = model_data['config']
                direction = cfg.get('direction', 'LONG')

                # Apply trend filter
                if direction == 'SHORT' and not trend_info['allow_short']:
                    logger.info(f"  Skipping {model_name} SHORT - blocked by trend filter ({trend_info['trend']})")
                    continue
                if direction == 'LONG' and not trend_info['allow_long']:
                    logger.info(f"  Skipping {model_name} LONG - blocked by trend filter ({trend_info['trend']})")
                    continue

                prob = model.predict_proba(X)[0, 1]

                if prob >= self.probability_threshold:
                    # Jan 7, 2026: Use TAKE_PROFIT_PCT (1.0%) instead of model threshold (0.5%)
                    signal = ModelSignal(
                        horizon=cfg['horizon'],
                        threshold=cfg['threshold'],
                        probability=prob,
                        priority=cfg['priority'],
                        target_pct=TAKE_PROFIT_PCT,  # Updated from cfg['threshold']
                        max_hold_mins=cfg['horizon_bars'] * 5,
                        direction=direction
                    )
                    signals.append(signal)
                    logger.info(f"  Signal: {model_name} {direction} prob={prob:.2%}")

            except Exception as e:
                logger.error(f"Error getting prediction from {model_name}: {e}")

        return signals

    def select_best_signal(self, signals: List[ModelSignal]) -> Optional[ModelSignal]:
        """Select the best signal based on voting mode."""
        if not signals:
            return None

        if self.voting_mode == 'best':
            # Select highest probability signal
            return max(signals, key=lambda s: s.probability)

        elif self.voting_mode == 'majority':
            # Need at least half of models to agree
            if len(signals) >= len(self.models) / 2:
                return max(signals, key=lambda s: s.probability)
            return None

        elif self.voting_mode == 'any':
            # Take first signal by priority
            return min(signals, key=lambda s: s.priority)

        return None

    def get_trend_follow_signals(self, df: pd.DataFrame) -> List[Dict]:
        """Get trend-following signals to RIDE the trend.
        
        Strategy: When strong trend detected (ROC > threshold), enter in trend direction
        with wider TP and trailing stop to capture larger moves.
        
        - LONG when: Price > SMA(5) > SMA(20) AND ROC(20) > 1.5%
        - SHORT when: Price < SMA(5) < SMA(20) AND ROC(20) < -1.5%
        - Uses wider SL (0.5%), larger TP (1.5%), and trailing stop (0.3%)
        """
        signals = []
        
        if not TREND_FOLLOW_ENABLED:
            return signals
        
        if len(df) < 50:
            return signals
        
        try:
            current_price = df['Close'].iloc[-1]
            sma_5 = df['Close'].rolling(5).mean().iloc[-1]
            sma_20 = df['Close'].rolling(20).mean().iloc[-1]
            roc_20 = (current_price / df['Close'].iloc[-21] - 1) * 100 if len(df) > 21 else 0
            
            # MACD for trend confirmation
            macd = df.get('trend_macd_diff', pd.Series([0])).iloc[-1] if 'trend_macd_diff' in df.columns else 0
            
            # Strong uptrend: Price > SMA(5) > SMA(20) AND ROC > threshold
            if current_price > sma_5 > sma_20 and roc_20 > TREND_FOLLOW_MIN_ROC:
                signals.append({
                    'model_id': 'trend_follow_long',
                    'horizon': 'trend',
                    'threshold': TREND_FOLLOW_TP_PCT,
                    'probability': min(roc_20 / 3, 1.0),  # Higher ROC = higher confidence
                    'priority': -1,  # Highest priority (trend-following)
                    'target_bars': TREND_FOLLOW_TIMEOUT_BARS,
                    'direction': 'LONG',
                    'signal_type': 'trend_follow',
                    'sl_pct': TREND_FOLLOW_SL_PCT,
                    'tp_pct': TREND_FOLLOW_TP_PCT,
                    'trailing_pct': TREND_FOLLOW_TRAILING_PCT,
                    'roc': roc_20,
                    'macd': macd
                })
                logger.info(f"🚀 TREND FOLLOW LONG: ROC={roc_20:.2f}%, Price>{sma_5:.0f}>{sma_20:.0f}")
            
            # Strong downtrend: Price < SMA(5) < SMA(20) AND ROC < -threshold
            elif current_price < sma_5 < sma_20 and roc_20 < -TREND_FOLLOW_MIN_ROC:
                signals.append({
                    'model_id': 'trend_follow_short',
                    'horizon': 'trend',
                    'threshold': TREND_FOLLOW_TP_PCT,
                    'probability': min(abs(roc_20) / 3, 1.0),
                    'priority': -1,
                    'target_bars': TREND_FOLLOW_TIMEOUT_BARS,
                    'direction': 'SHORT',
                    'signal_type': 'trend_follow',
                    'sl_pct': TREND_FOLLOW_SL_PCT,
                    'tp_pct': TREND_FOLLOW_TP_PCT,
                    'trailing_pct': TREND_FOLLOW_TRAILING_PCT,
                    'roc': roc_20,
                    'macd': macd
                })
                logger.info(f"🔻 TREND FOLLOW SHORT: ROC={roc_20:.2f}%, Price<{sma_5:.0f}<{sma_20:.0f}")
        
        except Exception as e:
            logger.error(f"Error getting trend-follow signals: {e}")
        
        return signals

    def get_indicator_long_signals(self, df: pd.DataFrame) -> List[Dict]:
        """Get LONG signals based on simple indicator rules.
        
        Strategy: BB %B < 0.5 (mean reversion - buy when price below BB midline)
        - Backtest (7-day fresh data): 534 trades/week, 50.4% WR, ~$5.6k/week P&L
        - Uses 0.30% SL, 0.75% TP (2.5:1 R:R ratio)
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
            
            # Check condition: BB %B < 0.5 (price below BB midline = mean reversion buy)
            bb_condition = bb_pct_b < 0.5
            
            if bb_condition:
                signals.append({
                    'model_id': 'indicator_long',
                    'horizon': 'indicator',
                    'threshold': INDICATOR_LONG_TP_PCT,
                    'probability': 1.0,  # Indicator signal, not probability
                    'priority': 0,  # Highest priority
                    'target_bars': INDICATOR_LONG_TIMEOUT_BARS,
                    'direction': 'LONG',
                    'signal_type': 'indicator',
                    'sl_pct': INDICATOR_LONG_SL_PCT,
                    'tp_pct': INDICATOR_LONG_TP_PCT,
                    'bb_pct_b': bb_pct_b
                })
                logger.info(f"INDICATOR LONG signal: BB %B={bb_pct_b:.4f} < 0.5")
        
        except Exception as e:
            logger.error(f"Error getting indicator signals: {e}")
        
        return signals

    def enter_trend_follow_position(self, signal: Dict, current_price: float) -> bool:
        """Enter a position for a trend-following signal with trailing stop."""
        model_id = signal['model_id']
        direction = signal['direction']
        
        # Check if this model already has a position
        if self.position_manager.has_position(model_id):
            return False
        
        # Check trend-follow position limit
        trend_positions = sum(1 for pos in self.position_manager.positions.values() 
                              if pos.model_id.startswith('trend_follow_'))
        if trend_positions >= MAX_TREND_FOLLOW_POSITIONS:
            return False
        
        # Block new entries if not enough time before market close
        market_hours = get_cme_market_hours()
        block_entries, block_reason = market_hours.should_block_new_entries(min_holding_time_minutes=120)
        if block_entries:
            logger.debug(f"Blocking trend-follow entry: {block_reason}")
            return False
        
        sl_pct = signal['sl_pct']
        tp_pct = signal['tp_pct']
        trailing_pct = signal.get('trailing_pct', 0.3)
        max_hold_bars = signal['target_bars']
        
        if direction == 'LONG':
            target_price = current_price * (1 + tp_pct / 100)
            stop_price = current_price * (1 - sl_pct / 100)
            action = "BUY"
            dir_str = "long"
        else:
            target_price = current_price * (1 - tp_pct / 100)
            stop_price = current_price * (1 + sl_pct / 100)
            action = "SELL"
            dir_str = "short"
        
        order_id = self.place_order(action, self.position_size)
        
        if order_id:
            position = Position(
                symbol=self.symbol,
                direction=dir_str,
                size=self.position_size,
                entry_price=current_price,
                entry_time=datetime.now().isoformat(),
                model_horizon=signal['horizon'],
                model_threshold=tp_pct,
                target_price=target_price,
                stop_price=stop_price,
                max_hold_bars=max_hold_bars,
                model_id=model_id,
                order_id=order_id,
                peak_price=current_price,
                trough_price=current_price
            )
            self.position_manager.add_position(position)
            
            unit = "contracts" if self.instrument == 'MNQ' else "shares"
            risk_per_contract = abs(current_price - stop_price) * self.point_value
            total_risk = risk_per_contract * self.position_size
            
            logger.info("="*50)
            logger.info(f"🚀 TREND FOLLOW {direction} [{model_id}]: {self.position_size} {unit} @ {current_price:.2f}")
            logger.info(f"Conditions: ROC={signal.get('roc', 0):.2f}%, MACD={signal.get('macd', 0):.2f}")
            logger.info(f"Target: {target_price:.2f} ({'+' if direction=='LONG' else '-'}{tp_pct}%)")
            logger.info(f"Stop: {stop_price:.2f} ({'-' if direction=='LONG' else '+'}{sl_pct}%)")
            logger.info(f"Trailing: {trailing_pct}% (will activate after profit)")
            logger.info(f"Risk: ${total_risk:.2f} per contract")
            logger.info("="*50)
            return True
        
        return False

    def enter_indicator_position(self, signal: Dict, current_price: float) -> bool:
        """Enter a position for an indicator-based signal with custom SL/TP."""
        model_id = signal['model_id']
        
        # Check if this model already has a position
        if self.position_manager.has_position(model_id):
            return False
        
        # Check indicator position limit
        indicator_positions = sum(1 for pos in self.position_manager.positions.values() 
                                  if pos.model_id.startswith('indicator_'))
        if indicator_positions >= MAX_INDICATOR_LONG_POSITIONS:
            return False
        
        # Block new entries if not enough time before market close
        market_hours = get_cme_market_hours()
        block_entries, block_reason = market_hours.should_block_new_entries(min_holding_time_minutes=60)
        if block_entries:
            logger.debug(f"Blocking indicator entry: {block_reason}")
            return False
        
        # Use indicator-specific SL/TP
        sl_pct = signal['sl_pct']
        tp_pct = signal['tp_pct']
        max_hold_bars = signal['target_bars']
        
        target_price = current_price * (1 + tp_pct / 100)
        stop_price = current_price * (1 - sl_pct / 100)
        
        order_id = self.place_order("BUY", self.position_size)
        
        if order_id:
            position = Position(
                symbol=self.symbol,
                direction="long",
                size=self.position_size,
                entry_price=current_price,
                entry_time=datetime.now().isoformat(),
                model_horizon=signal['horizon'],
                model_threshold=tp_pct,
                target_price=target_price,
                stop_price=stop_price,
                max_hold_bars=max_hold_bars,
                model_id=model_id,
                order_id=order_id
            )
            self.position_manager.add_position(position)
            
            unit = "contracts" if self.instrument == 'MNQ' else "shares"
            risk_per_contract = abs(current_price - stop_price) * self.point_value
            total_risk = risk_per_contract * self.position_size
            
            logger.info("="*50)
            logger.info(f"ENTERED LONG [{model_id}]: {self.position_size} {unit} @ {current_price:.2f} (INDICATOR)")
            logger.info(f"Conditions: BB %B={signal.get('bb_pct_b', 0):.4f} < 0.5")
            logger.info(f"Target: {target_price:.2f} (+{tp_pct}%)")
            logger.info(f"Stop: {stop_price:.2f} (-{sl_pct}%)")
            logger.info(f"Risk: ${total_risk:.2f} per contract")
            logger.info(f"R:R ratio: 1:{tp_pct/sl_pct:.1f}")
            logger.info("="*50)
            return True
        
        return False

    def place_order(self, action: str, quantity: int) -> Optional[int]:
        """Place a market order."""
        try:
            order = MarketOrder(action, quantity)
            trade = self.ib.placeOrder(self.contract, order)
            self.ib.sleep(1)
            logger.info(f"Placed {action} order for {quantity} shares")
            return trade.order.orderId
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None

    def enter_position_for_model(self, model_id: str, signal: ModelSignal, current_price: float) -> bool:
        """Enter a new position for a specific model."""
        # Check if this model already has a position
        if self.position_manager.has_position(model_id):
            logger.debug(f"Model {model_id} already has a position")
            return False
        
        # Check max positions limit
        if self.position_manager.count_positions() >= self.max_positions:
            logger.warning(f"Max positions ({self.max_positions}) reached")
            return False
        
        # Block new entries if not enough time before market close
        market_hours = get_cme_market_hours()
        block_entries, block_reason = market_hours.should_block_new_entries(min_holding_time_minutes=60)
        if block_entries:
            logger.debug(f"Blocking new entry for {model_id}: {block_reason}")
            return False

        max_hold_bars = signal.max_hold_mins // 5
        direction = signal.direction
        
        # Calculate target and stop based on direction
        if direction == "LONG":
            target_price = current_price * (1 + signal.target_pct / 100)
            stop_price = current_price * (1 - self.stop_loss_pct / 100)
            order_action = "BUY"
        else:  # SHORT
            target_price = current_price * (1 - signal.target_pct / 100)
            stop_price = current_price * (1 + self.stop_loss_pct / 100)
            order_action = "SELL"

        order_id = self.place_order(order_action, self.position_size)

        if order_id:
            position = Position(
                symbol=self.symbol,
                direction=direction.lower(),  # "long" or "short"
                size=self.position_size,
                entry_price=current_price,
                entry_time=datetime.now().isoformat(),
                model_horizon=signal.horizon,
                model_threshold=signal.threshold,
                target_price=target_price,
                stop_price=stop_price,
                max_hold_bars=max_hold_bars,
                model_id=model_id,
                order_id=order_id
            )
            self.position_manager.add_position(position)

            # Calculate risk for MNQ
            unit = "contracts" if self.instrument == 'MNQ' else "shares"
            if self.instrument == 'MNQ':
                risk_per_contract = abs(current_price - stop_price) * self.point_value
                total_risk = risk_per_contract * self.position_size
                logger.info("="*50)
                logger.info(f"ENTERED {direction} [{model_id}]: {self.position_size} {unit} @ {current_price:.2f}")
                logger.info(f"Probability: {signal.probability:.2%}")
                if direction == "LONG":
                    logger.info(f"Target: {target_price:.2f} (+{signal.target_pct}%)")
                    logger.info(f"Stop: {stop_price:.2f} (-{self.stop_loss_pct}%)")
                else:
                    logger.info(f"Target: {target_price:.2f} (-{signal.target_pct}%)")
                    logger.info(f"Stop: {stop_price:.2f} (+{self.stop_loss_pct}%)")
                logger.info(f"Risk: ${total_risk:.2f} per contract")
            else:
                logger.info("="*50)
                logger.info(f"ENTERED {direction} [{model_id}]: {self.position_size} {unit} @ ${current_price:.2f}")
                logger.info(f"Probability: {signal.probability:.2%}")
                if direction == "LONG":
                    logger.info(f"Target: ${target_price:.2f} (+{signal.target_pct}%)")
                    logger.info(f"Stop: ${stop_price:.2f} (-{self.stop_loss_pct}%)")
                else:
                    logger.info(f"Target: ${target_price:.2f} (-{signal.target_pct}%)")
                    logger.info(f"Stop: ${stop_price:.2f} (+{self.stop_loss_pct}%)")
            logger.info(f"Max hold: {signal.max_hold_mins} mins")
            logger.info(f"Total positions: {self.position_manager.count_positions()}")
            logger.info("="*50)
            return True

        return False

    def exit_position_for_model(self, model_id: str, current_price: float, reason: str) -> bool:
        """Exit position for a specific model. Returns True if filled, False if pending/failed."""
        position = self.position_manager.get_position(model_id)
        if not position:
            return False

        action = "SELL" if position.direction == "long" else "BUY"
        order = MarketOrder(action, position.size)
        
        try:
            trade = self.ib.placeOrder(self.contract, order)
            self.ib.sleep(2)
            
            status = trade.orderStatus.status
            
            if status == 'Filled':
                # Calculate PnL - for futures multiply by point value
                if position.direction == "long":
                    price_diff = current_price - position.entry_price
                    pnl_pct = (current_price / position.entry_price - 1) * 100
                else:  # short
                    price_diff = position.entry_price - current_price
                    pnl_pct = (position.entry_price / current_price - 1) * 100
                pnl = price_diff * position.size * self.point_value

                # Log trade to database for performance tracking
                if reason == 'target_reached':
                    exit_reason_db = 'TAKE_PROFIT'
                elif reason == 'stop_loss':
                    exit_reason_db = 'STOP_LOSS'
                elif reason == 'trailing_stop':
                    exit_reason_db = 'TRAILING_STOP'
                elif reason == 'pre_close':
                    exit_reason_db = 'PRE_CLOSE'
                else:
                    exit_reason_db = 'TIMEOUT'
                try:
                    log_trade_to_db(
                        bot_type='MNQ',
                        model_id=model_id,
                        direction=position.direction.upper(),
                        entry_time=position.entry_time,
                        entry_price=position.entry_price,
                        exit_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        exit_price=current_price,
                        pnl_pct=pnl_pct,
                        pnl_dollar=pnl,
                        exit_reason=exit_reason_db,
                        bars_held=position.bars_held,
                        horizon_bars=position.target_bars,
                        model_horizon=position.model_horizon,
                        model_threshold=position.model_threshold
                    )
                except Exception as e:
                    logger.warning(f"Failed to log trade to database: {e}")

                self.position_manager.remove_position(model_id)
                
                # Clear pending close tracking
                if model_id in self.pending_close_orders:
                    del self.pending_close_orders[model_id]

                unit = "contracts" if self.instrument == 'MNQ' else "shares"
                logger.info("="*50)
                logger.info(f"EXITED {position.direction.upper()} [{model_id}]: {position.size} {unit} @ {current_price:.2f}")
                logger.info(f"Entry: {position.entry_price:.2f}")
                logger.info(f"PnL: ${pnl:.2f} ({pnl_pct:+.2f}%)")
                logger.info(f"Reason: {reason}")
                logger.info(f"Remaining positions: {self.position_manager.count_positions()}")
                logger.info("="*50)
                return True
            
            elif status in ['PreSubmitted', 'Submitted', 'Inactive', 'PendingSubmit']:
                # Order is pending (market closed or queued)
                self.pending_close_orders[model_id] = datetime.now()
                logger.warning(f"Close order PENDING for [{model_id}] - status: {status}")
                logger.warning(f"  Market may be closed. Order will execute when market opens.")
                return False  # Don't remove position yet
            
            else:
                logger.warning(f"Close order status: {status} for [{model_id}]")
                return False
                
        except Exception as e:
            logger.error(f"Exit order error for [{model_id}]: {e}")
            return False

    def check_exit_conditions_for_position(self, position: Position, current_price: float) -> Optional[str]:
        """Check if we should exit a specific position (LONG or SHORT)."""
        
        # Update trailing stop tracking
        self._update_trailing_stop(position, current_price)
        
        if position.direction == "long":
            # LONG: Target reached when price rises
            if current_price >= position.target_price:
                return "target_reached"
            # LONG: Trailing stop hit
            if position.trailing_stop_active and current_price <= position.trailing_stop_price:
                return "trailing_stop"
            # LONG: Stop loss when price falls
            if current_price <= position.stop_price:
                return "stop_loss"
        elif position.direction == "short":
            # SHORT: Target reached when price falls
            if current_price <= position.target_price:
                return "target_reached"
            # SHORT: Trailing stop hit
            if position.trailing_stop_active and current_price >= position.trailing_stop_price:
                return "trailing_stop"
            # SHORT: Stop loss when price rises
            if current_price >= position.stop_price:
                return "stop_loss"

        # Timeout - same for both directions (with 2x multiplier)
        entry_time = datetime.fromisoformat(position.entry_time)
        max_hold_mins = position.max_hold_bars * 5 * TIMEOUT_MULTIPLIER
        if datetime.now() - entry_time > timedelta(minutes=max_hold_mins):
            return "timeout"

        # PRE-CLOSE EXIT: Close positions before market closure (weekends/holidays)
        market_hours = get_cme_market_hours()
        pre_close, pre_close_reason = market_hours.should_close_positions()
        if pre_close:
            logger.warning(f"[{position.model_id}] Pre-close exit triggered: {pre_close_reason}")
            return "pre_close"

        return None
    
    def _update_trailing_stop(self, position: Position, current_price: float) -> None:
        """Update trailing stop tracking for a position."""
        if position.direction == "long":
            # Update peak price
            if position.peak_price == 0 or current_price > position.peak_price:
                position.peak_price = current_price
            
            # Check if we should activate trailing stop
            profit_pct = (current_price / position.entry_price - 1) * 100
            if not position.trailing_stop_active and profit_pct >= TRAILING_STOP_ACTIVATION_PCT:
                position.trailing_stop_active = True
                position.trailing_stop_price = position.peak_price * (1 - TRAILING_STOP_PCT / 100)
                logger.info(f"[{position.model_id}] Trailing stop ACTIVATED at ${position.trailing_stop_price:.2f} (profit: {profit_pct:.2f}%)")
            
            # Update trailing stop price if active and price made new high
            if position.trailing_stop_active:
                new_stop = position.peak_price * (1 - TRAILING_STOP_PCT / 100)
                if new_stop > position.trailing_stop_price:
                    position.trailing_stop_price = new_stop
                    logger.debug(f"[{position.model_id}] Trailing stop raised to ${position.trailing_stop_price:.2f}")
        
        elif position.direction == "short":
            # Update trough price
            if position.trough_price == 0 or current_price < position.trough_price:
                position.trough_price = current_price
            
            # Check if we should activate trailing stop
            profit_pct = (position.entry_price / current_price - 1) * 100
            if not position.trailing_stop_active and profit_pct >= TRAILING_STOP_ACTIVATION_PCT:
                position.trailing_stop_active = True
                position.trailing_stop_price = position.trough_price * (1 + TRAILING_STOP_PCT / 100)
                logger.info(f"[{position.model_id}] Trailing stop ACTIVATED at ${position.trailing_stop_price:.2f} (profit: {profit_pct:.2f}%)")
            
            # Update trailing stop price if active and price made new low
            if position.trailing_stop_active:
                new_stop = position.trough_price * (1 + TRAILING_STOP_PCT / 100)
                if new_stop < position.trailing_stop_price:
                    position.trailing_stop_price = new_stop
                    logger.debug(f"[{position.model_id}] Trailing stop lowered to ${position.trailing_stop_price:.2f}")
    
    def check_all_positions(self, current_price: float) -> None:
        """Check exit conditions for all open positions."""
        # First, check status of any pending close orders
        self._check_pending_close_orders()
        
        for position in self.position_manager.get_all_positions():
            # Skip if there's already a pending close order for this position
            if position.model_id in self.pending_close_orders:
                pending_time = self.pending_close_orders[position.model_id]
                elapsed_mins = (datetime.now() - pending_time).total_seconds() / 60
                # Only log every 30 minutes to avoid spam
                if elapsed_mins >= 30 and int(elapsed_mins) % 30 == 0:
                    logger.info(f"[{position.model_id}] Close order still pending ({elapsed_mins:.0f} mins)")
                continue
            
            exit_reason = self.check_exit_conditions_for_position(position, current_price)
            if exit_reason:
                self.exit_position_for_model(position.model_id, current_price, exit_reason)
            else:
                if position.direction == "long":
                    pnl_pct = (current_price / position.entry_price - 1) * 100
                else:  # short
                    pnl_pct = (position.entry_price / current_price - 1) * 100
                logger.debug(f"[{position.model_id}] {position.direction.upper()}: {pnl_pct:+.2f}% @ ${current_price:.2f}")
    
    def _check_pending_close_orders(self) -> None:
        """Check if any pending close orders have been filled or timed out."""
        if not self.pending_close_orders:
            return
        
        try:
            # Get current IB positions to check if close orders filled
            ib_positions = self.ib.positions()
            mnq_position = 0
            for p in ib_positions:
                if p.contract.symbol == 'MNQ':
                    mnq_position = p.position
            
            # Check each pending close order
            models_to_remove = []
            for model_id, order_time in list(self.pending_close_orders.items()):
                pos = self.position_manager.get_position(model_id)
                if pos is None:
                    # Position already removed
                    models_to_remove.append(model_id)
                    continue
                
                elapsed_mins = (datetime.now() - order_time).total_seconds() / 60
                
                # If IB position is now flat (0), close orders filled
                if mnq_position == 0:
                    logger.info(f"[{model_id}] Close order FILLED - IB position is flat")
                    # Log trade to database before removing
                    self._log_pending_trade(pos, 'FILLED')
                    self.position_manager.remove_position(model_id)
                    models_to_remove.append(model_id)
                
                # Force clear stuck pending orders after 60 minutes (market likely closed)
                elif elapsed_mins > 60:
                    logger.warning(f"[{model_id}] Pending order TIMEOUT after {elapsed_mins:.0f} mins - force clearing")
                    # Log trade to database with estimated exit price
                    self._log_pending_trade(pos, 'TIMEOUT_FORCED')
                    self.position_manager.remove_position(model_id)
                    models_to_remove.append(model_id)
            
            for model_id in models_to_remove:
                if model_id in self.pending_close_orders:
                    del self.pending_close_orders[model_id]
                    
        except Exception as e:
            logger.error(f"Error checking pending close orders: {e}")
    
    def _log_pending_trade(self, position, exit_reason: str) -> None:
        """Log a trade that was pending/stuck to the database."""
        try:
            # Use current price as exit price estimate
            current_price = self.current_price or position.entry_price
            
            if position.direction == "long":
                pnl_pct = (current_price / position.entry_price - 1) * 100
            else:
                pnl_pct = (position.entry_price / current_price - 1) * 100
            
            price_diff = abs(current_price - position.entry_price)
            pnl = price_diff * position.size * self.point_value
            if pnl_pct < 0:
                pnl = -pnl
            
            log_trade_to_db(
                bot_type='MNQ',
                model_id=position.model_id,
                direction=position.direction.upper(),
                entry_time=position.entry_time,
                entry_price=position.entry_price,
                exit_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                exit_price=current_price,
                pnl_pct=pnl_pct,
                pnl_dollar=pnl,
                exit_reason=exit_reason,
                bars_held=position.bars_held,
                horizon_bars=position.target_bars,
                model_horizon=position.model_horizon,
                model_threshold=position.model_threshold
            )
            logger.info(f"[{position.model_id}] Trade logged to database: {exit_reason}, P&L: ${pnl:.2f}")
        except Exception as e:
            logger.warning(f"Failed to log pending trade to database: {e}")

    def should_update_features(self) -> bool:
        """Check if we need to recalculate features (new 5-min bar)."""
        now = datetime.now()
        current_bar_time = now.replace(second=0, microsecond=0)
        current_bar_time = current_bar_time.replace(minute=(current_bar_time.minute // 5) * 5)
        
        if self.last_feature_update is None or self.last_feature_update != current_bar_time:
            return True
        return False
    
    def run_iteration(self, full_check: bool = False) -> None:
        """Run one iteration of the trading loop.
        
        Uses Finnhub real-time QQQ for synthetic bar updates (every 15 sec).
        Uses IB 5-min historical bars as base for features.
        Runs model predictions every 15 seconds with synthetic bar.
        
        Args:
            full_check: If True, force refresh historical data from IB.
        """
        # Check if we need to refresh historical data (new completed 5-min bar)
        new_bar = self.should_update_features()
        if new_bar or full_check:
            logger.info("-"*50)
            logger.info(f"New bar - refreshing historical data: {datetime.now()}")
            
            # CRITICAL: Refresh parquet on EVERY new bar to ensure features are current
            # (was only refreshing every 5 mins which caused stale features)
            self.refresh_parquet_from_yahoo()
            
            df = self.get_historical_data(force_refresh=True)
            if df.empty:
                logger.warning("No data from IB, using cache")
            else:
                # Update feature timestamp
                now = datetime.now()
                self.last_feature_update = now.replace(second=0, microsecond=0)
                self.last_feature_update = self.last_feature_update.replace(
                    minute=(self.last_feature_update.minute // 5) * 5
                )
            
            # Check for orphan positions every 5-min bar
            self.check_orphan_positions()
        
        if self.data_cache.empty:
            logger.debug("No historical data available yet")
            return
        
        # Get real-time price for synthetic bar
        # During market hours: use Finnhub QQQ (features use QQQ scale)
        # Outside market hours: use Yahoo MNQ=F converted to QQQ scale
        is_market_hours = self.is_us_market_hours()
        price_source = "unknown"
        
        if is_market_hours:
            qqq_price = self.get_finnhub_qqq_price()
            if qqq_price:
                price_source = "finnhub_qqq"
            else:
                qqq_price = self.data_cache['close'].iloc[-1]
                price_source = "ib_cache"
        else:
            # Overnight: get MNQ price and convert to QQQ scale for features
            mnq_price_yahoo = self.get_yahoo_mnq_price()
            if mnq_price_yahoo:
                qqq_price = mnq_price_yahoo / MNQ_QQQ_RATIO
                price_source = "yahoo_mnq"
                logger.debug(f"Overnight: Yahoo MNQ ${mnq_price_yahoo:.2f} -> QQQ ${qqq_price:.2f}")
            else:
                qqq_price = self.data_cache['close'].iloc[-1]
                price_source = "ib_cache_stale"
        
        # Update synthetic bar with real-time QQQ price (kept for price tracking only)
        self.update_synthetic_bar(qqq_price)
        
        # Use parquet features DIRECTLY for signal generation
        # No synthetic bar needed for features - parquet has all 206 features correctly calculated
        # Synthetic bar only updated 2 features (return_1bar, log_return_1bar) which
        # are already in the fresh parquet. Keeps signal generation consistent with backtest.
        if self.use_parquet_features and self.parquet_features is not None and not self.parquet_features.empty:
            df = self.parquet_features.tail(1000).copy()
        else:
            # Fallback to IB data cache if parquet not available
            df = self.get_features_with_synthetic_bar()
        
        if df.empty:
            logger.debug("No features available yet")
            return
        
        # VALIDATE DATA (every 5-min bar or on first check) - LOG ONLY, don't block signals
        if new_bar or self.last_validation_time is None:
            self.validate_data_for_signals(df, qqq_price)  # Logs warnings/errors but continues
        
        # Apply feature normalization if using normalized models
        if USE_NORMALIZED_MODELS:
            df = normalize_features(df)
        
        # Get live MNQ price for trading (Yahoo MNQ=F preferred, fallback to QQQ ratio)
        trading_price = self.get_live_mnq_price()
        if trading_price is None:
            trading_price = qqq_price * MNQ_QQQ_RATIO
        
        # Check exit conditions every iteration (using real-time price)
        if self.position_manager.count_positions() > 0:
            self.check_all_positions(trading_price)
        
        # PARALLEL MODE: Each model can have its own position
        if self.voting_mode == 'parallel':
            # First pass: collect all probabilities (every 15 sec with real-time data)
            probs = {}
            latest = df.iloc[-1:]
            for f in self.feature_cols:
                if f not in latest.columns:
                    latest[f] = 0
            X = latest[self.feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
            
            for model_name, model_data in self.models.items():
                try:
                    prob = model_data['model'].predict_proba(X)[0, 1]
                    probs[model_name] = prob
                except Exception as e:
                    logger.error(f"Error getting prob for {model_name}: {e}")
                    probs[model_name] = 0
            
            # Log probabilities every 15 seconds
            prob_str = " | ".join([f"{k}:{v:.0%}" for k, v in probs.items()])
            n_pos = self.position_manager.count_positions()
            max_prob = max(probs.values()) if probs else 0
            
            # Log at INFO level if interesting (high prob or new bar), else DEBUG
            # Log EVERY signal for uptime monitoring
            logger.info(f"MNQ: ${trading_price:.2f} | QQQ: ${qqq_price:.2f} | Pos: {n_pos}/{self.max_positions} | {prob_str}")
            
            self.total_checks += 1
            
            # Log signals to CSV every 15 seconds
            now = datetime.now()
            indicators = {}
            try:
                last_row = df.iloc[-1]
                indicators['rsi'] = last_row.get('momentum_rsi', 0)
                indicators['macd'] = last_row.get('trend_macd', 0)
                indicators['atr'] = last_row.get('volatility_atr', 0)
                indicators['bb_pct_b'] = last_row.get('volatility_bbp', 0.5)
            except:
                pass
            
            self.signal_logger.log_signals(
                timestamp=now,
                qqq_price=qqq_price,
                mnq_price=trading_price,
                price_source=price_source,
                probabilities=probs,
                threshold=self.probability_threshold,
                active_positions=n_pos,
                indicators=indicators,
                is_market_hours=is_market_hours
            )
            
            # Check for trend-following signals FIRST (highest priority - ride the trend)
            trend_signals = self.get_trend_follow_signals(df)
            for trend_signal in trend_signals:
                if self.enter_trend_follow_position(trend_signal, trading_price):
                    self.total_signals += 1
            
            # Check for indicator-based LONG signals (runs alongside ML models)
            indicator_signals = self.get_indicator_long_signals(df)
            for ind_signal in indicator_signals:
                if self.enter_indicator_position(ind_signal, trading_price):
                    self.total_signals += 1
            
            # Second pass: enter positions for ML signals above threshold
            for model_name, model_data in self.models.items():
                # Skip if this model already has a position
                if self.position_manager.has_position(model_name):
                    continue
                
                # Skip if max positions reached
                if self.position_manager.count_positions() >= self.max_positions:
                    break
                
                prob = probs.get(model_name, 0)
                if prob >= self.probability_threshold:
                    self.total_signals += 1
                    cfg = model_data['config']
                    signal = ModelSignal(
                        horizon=cfg['horizon'],
                        threshold=cfg['threshold'],
                        probability=prob,
                        priority=cfg['priority'],
                        target_pct=cfg['threshold'],
                        max_hold_mins=cfg['horizon_bars'] * 5,
                        direction=cfg.get('direction', 'LONG')
                    )
                    logger.info(f"  SIGNAL: {model_name} prob={prob:.2%} >= {self.probability_threshold:.0%}")
                    self.enter_position_for_model(model_name, signal, trading_price)
        else:
            # Original voting modes (best, majority, any) - single position
            signals = self.get_ensemble_signals(df)

            if signals:
                logger.info(f"Got {len(signals)} signals")
                best_signal = self.select_best_signal(signals)
                if best_signal:
                    model_id = f"{best_signal.horizon}_{best_signal.threshold}pct"
                    self.enter_position_for_model(model_id, best_signal, trading_price)

    def run(self, interval_seconds: int = None) -> None:
        """Main trading loop.
        
        Checks price every signal_check_interval (default 15s).
        Only recalculates features on new 5-min bars.
        """
        if interval_seconds is None:
            interval_seconds = self.signal_check_interval
            
        logger.info("="*60)
        logger.info("STARTING ENSEMBLE TRADING BOT")
        logger.info("="*60)
        logger.info(f"Price check interval: {interval_seconds}s")
        logger.info(f"Feature update: Every 5-min bar")

        # Outer loop for crash recovery - bot will restart itself on any fatal error
        while True:
            try:
                if not self.connect():
                    logger.error("Failed to connect. Retrying in 30s...")
                    time.sleep(30)
                    continue

                self.check_orphan_positions()
                
                # Load parquet features for hybrid approach (correct cumulative features)
                if self.use_parquet_features:
                    logger.info("Loading parquet features for hybrid approach...")
                    if self.load_parquet_features():
                        logger.info(f"Parquet features loaded - will use for correct cumulative values")
                    else:
                        logger.warning("Failed to load parquet - falling back to IB-only with offsets")
                
                # Initial data load (IB data for real-time updates)
                logger.info("Loading initial data from IB...")
                df = self.get_historical_data(force_refresh=True)
                if not df.empty:
                    self.prepare_features(df)
                    logger.info(f"Initial IB data loaded: {len(df)} bars")

                iteration = 0
                last_status_log = datetime.now()
                
                consecutive_errors = 0
                max_consecutive_errors = 10  # Trigger reconnect after this many errors
                
                while True:
                    try:
                        # Check connection health and reconnect if needed
                        if not self.is_connected():
                            logger.warning("IB connection lost - attempting reconnect...")
                            self.reconnect()  # This now loops forever until success
                            consecutive_errors = 0
                        
                        iteration += 1
                        # First iteration forces data refresh
                        self.run_iteration(full_check=(iteration == 1))
                        consecutive_errors = 0  # Reset on successful iteration
                        
                        # Log status every hour
                        if datetime.now() - last_status_log > timedelta(hours=1):
                            uptime = datetime.now() - self.start_time
                            hours = uptime.total_seconds() / 3600
                            checks_per_hour = self.total_checks / max(hours, 0.1)
                            logger.info("="*50)
                            logger.info(f"HOURLY STATUS REPORT")
                            logger.info(f"  Uptime: {hours:.1f} hours")
                            logger.info(f"  Signal checks: {self.total_checks} ({checks_per_hour:.0f}/hr)")
                            logger.info(f"  Signals generated: {self.total_signals}")
                            logger.info(f"  Connection drops: {self.connection_drops}")
                            logger.info(f"  Reconnections: {getattr(self, 'reconnection_count', 0)}")
                            logger.info(f"  Open positions: {self.position_manager.count_positions()}/{self.max_positions}")
                            logger.info("="*50)
                            last_status_log = datetime.now()
                            
                    except Exception as e:
                        consecutive_errors += 1
                        logger.error(f"Error in iteration ({consecutive_errors}/{max_consecutive_errors}): {e}")
                        
                        # If too many consecutive errors, try reconnecting
                        if consecutive_errors >= max_consecutive_errors:
                            logger.warning(f"Too many consecutive errors - forcing reconnect...")
                            self.reconnect()  # This now loops forever until success
                            consecutive_errors = 0

                    # Use ib.sleep() to keep event loop alive, fallback to time.sleep if disconnected
                    try:
                        self.ib.sleep(interval_seconds)
                    except Exception:
                        time.sleep(interval_seconds)

            except KeyboardInterrupt:
                logger.info("Shutdown signal received")
                self.disconnect()
                logger.info("Bot stopped")
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
                    self.disconnect()
                except:
                    pass
                time.sleep(30)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Ensemble Trading Bot for QQQ/MNQ')
    parser.add_argument('--instrument', type=str, default='MNQ',
                        choices=['MNQ', 'QQQ'],
                        help='Trading instrument (default: MNQ futures)')
    parser.add_argument('--prob-threshold', type=float, default=0.50,
                        help='Min probability to enter (default: 0.50 - lowered Jan 6, 2026)')
    parser.add_argument('--position-size', type=int, default=1,
                        help='Contracts (MNQ) or shares (QQQ) per trade (default: 1)')
    parser.add_argument('--stop-loss', type=float, default=0.75,
                        help='Stop loss %% (default: 0.75)')
    parser.add_argument('--interval', type=int, default=15,
                        help='Seconds between price checks (default: 15)')
    parser.add_argument('--voting', type=str, default='parallel',
                        choices=['parallel', 'best', 'majority', 'any'],
                        help='Signal selection mode (default: parallel)')
    parser.add_argument('--max-positions', type=int, default=4,
                        help='Max concurrent positions in parallel mode (default: 4)')
    parser.add_argument('--live', action='store_true',
                        help='Use live trading (default: paper)')

    args = parser.parse_args()

    bot = EnsembleTradingBot(
        probability_threshold=args.prob_threshold,
        position_size=args.position_size,
        stop_loss_pct=args.stop_loss,
        paper_trading=not args.live,
        voting_mode=args.voting,
        signal_check_interval=args.interval,
        max_positions=args.max_positions,
        instrument=args.instrument
    )

    bot.run()


if __name__ == "__main__":
    main()
