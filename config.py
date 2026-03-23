"""
Configuration for the trading model pipeline.
"""

# Data paths
DATA_DIR = "data"
MODELS_DIR = "models"
RESULTS_DIR = "results"

# Data file
DATA_FILE = "QQQ_5min_IB_with_indicators.csv"

# Prediction horizons (in number of 5-min bars)
# 5min=1, 10min=2, 15min=3, 20min=4, 25min=5, 30min=6, 45min=9, 1h=12, 1.5h=18, 2h=24, 3h=36, 4h=48
HORIZONS = {
    "5min": 1,
    "10min": 2,
    "15min": 3,
    "20min": 4,
    "25min": 5,
    "30min": 6,
    "45min": 9,
    "1h": 12,
    "1h30m": 18,
    "2h": 24,
    "3h": 36,
    "4h": 48,
}

# Movement thresholds (percentage)
THRESHOLDS = [0.5, 0.75, 1.0, 1.25]

# Train/test split
TRAIN_RATIO = 0.70
TEST_RATIO = 0.30

# Maximum holding time (4 hours = 48 bars of 5 min)
MAX_HOLDING_BARS = 48

# Feature engineering parameters
LOOKBACK_PERIODS = [1, 2, 3, 5, 10, 20, 50]  # For lagged features
ROLLING_WINDOWS = [5, 10, 20, 50]  # For rolling statistics

# IB Gateway settings
IB_HOST = "127.0.0.1"
IB_PORT = 4002  # Paper trading port (7497 for TWS paper)
IB_CLIENT_ID = 10  # Changed to 10 to avoid stale IB Gateway connections (Jan 7)

# Trading parameters
SYMBOL = "QQQ"
POSITION_SIZE = 100  # Number of shares per trade

# Position persistence
POSITION_FILE = "positions.json"
