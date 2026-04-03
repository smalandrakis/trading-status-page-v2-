# For Windsurf: 22-Feature Model Testing Guide

## Location
**All files are in:** `/Users/smalandrakis/Documents/WindSurf/DS:ML trade model playground/CascadeProjects/windsurf-project/`

## What's Available

### 1. **22-Feature V3 Models** (Ready to Use)
```
btc_model_package/
├── predictor.py                   # Main prediction class (22 features)
├── btc_2h_model_v3.pkl           # 2-hour horizon model
├── btc_4h_model_v3.pkl           # 4-hour horizon model  
├── btc_6h_model_v3.pkl           # 6-hour horizon model
├── btc_*_scaler_v3.pkl           # Feature scalers (3 files)
├── btc_2h_features_v3.json       # Feature list (22 names)
└── test_predictor.py             # Quick test with live Binance data
```

**Model Performance (Realistic Simulation with $2.02 Commission):**
- Default thresholds (0.55/0.35): 2,885 trades, 40.2% WR, +$3,614
- **Optimal thresholds (0.60/0.30): 1,612 trades, 42.7% WR, +$4,430** ✓
- High thresholds (0.70/0.20): 296 trades, 48.0% WR, +$994

### 2. **Simple Trading Bot** (New)
```python
# File: btc_v3_bot_simple.py
python3 btc_v3_bot_simple.py
```

**Features:**
- Uses 22-feature V3 models
- Optimized thresholds: LONG=0.60, SHORT=0.30
- Fetches data from Binance (free, real-time)
- Placeholder for IB Gateway integration (port 4002)
- Checks every 2 minutes
- Automatic TP/SL management

**To Run:**
1. Make sure `python-binance` is installed: `pip install python-binance`
2. Run: `python3 btc_v3_bot_simple.py`
3. Implement IB Gateway order submission in `execute_entry()` and `execute_exit()` functions

### 3. **Backtest Scripts**
```bash
# Original Windsurf backtest (your script)
python3 walkforward_v3_predictor.py

# Threshold sensitivity analysis
python3 test_threshold_sensitivity.py

# High threshold analysis (0.75-0.90)
python3 test_high_thresholds.py
```

## Quick Start for Windsurf

### Test the 22-Feature Model with Live Data
```bash
cd "/Users/smalandrakis/Documents/WindSurf/DS:ML trade model playground/CascadeProjects/windsurf-project"
python3 btc_model_package/test_predictor.py
```

### Run Your Backtest Script
```bash
# Your existing backtest script should work with 22-feature models
python3 walkforward_v3_predictor.py
```

### Integrate into Your Bot Collection
```python
# In your existing bot framework:
from btc_model_package.predictor import BTCPredictor

# Initialize with optimized thresholds
predictor = BTCPredictor()
predictor.LONG_THRESHOLD = 0.60  # More selective
predictor.SHORT_THRESHOLD = 0.30  # More selective

# Get prediction
signal, confidence, details = predictor.predict(df)

# Trade only on strong signals
if signal == 'LONG':
    # Entry at current price
    # TP = entry * 1.01 (+1%)
    # SL = entry * 0.995 (-0.5%)
elif signal == 'SHORT':
    # Entry at current price
    # TP = entry * 0.99 (-1%)
    # SL = entry * 1.005 (+0.5%)
```

## Key Differences from 17-Feature Model

| Metric | 17-Feature | 22-Feature | Improvement |
|--------|-----------|-----------|-------------|
| **Win Rate** | 39.8% | 42.7% | +2.9% |
| **Total P&L** | **-$2,035** | **+$4,430** | +$6,465 |
| **Profit Factor** | 0.97 | 1.11 | +14% |
| **Avg P&L/trade** | -$0.80 | +$2.75 | +$3.55 |
| **Status** | ❌ Losing | ✅ Profitable | |

## What the 5 New Features Added

1. **rsi_28**: Longer-term RSI (captures broader momentum)
2. **day_of_week**: Weekly patterns (some days are more volatile)
3. **hour_sin/hour_cos**: Time of day encoding
4. **volume_hour_median**: Hourly volume patterns
5. **bb_position_50**: Bollinger Band position (50-period)
6. **dist_from_prev_4h**: Distance from 4h-ago close

**Feature Importance:**
- day_of_week: 5.4% (7th most important)
- volume_hour_median: 4.9% (8th most important)
- These new features ARE being used by the models

## Threshold Recommendations

| Use Case | LONG_TH | SHORT_TH | Trades/2yr | WR% | P&L | When to Use |
|----------|---------|----------|-----------|-----|-----|-------------|
| **Balanced** | 0.60 | 0.30 | 1,612 | 42.7% | +$4,430 | **Recommended** - Good balance |
| Conservative | 0.65 | 0.25 | 732 | 45.5% | +$2,761 | Fewer trades, higher WR |
| Aggressive | 0.55 | 0.35 | 2,885 | 40.2% | +$3,614 | More trades, lower WR |
| Ultra-selective | 0.70 | 0.20 | 296 | 48.0% | +$994 | Very few trades |

## Testing with Your Backtest Framework

Your `walkforward_v3_predictor.py` script should automatically use the 22-feature models since they're in `btc_model_package/`. 

To test different thresholds in your script, modify lines 57-58 in `walkforward_v3_predictor.py`:
```python
MIN_CONFIDENCE = 0.60  # Change this (was 0.58)
```

Or test the threshold by changing it in the predictor after loading:
```python
predictor = BTCPredictor(model_dir=MODEL_DIR)
predictor.LONG_THRESHOLD = 0.60  # Your desired threshold
predictor.SHORT_THRESHOLD = 0.30  # Your desired threshold
```

## Files Ready for You

1. ✅ `btc_model_package/` - 22-feature models, ready to use
2. ✅ `btc_v3_bot_simple.py` - Simple bot template
3. ✅ `test_threshold_sensitivity.py` - Threshold testing results
4. ✅ `test_high_thresholds.py` - High threshold analysis (running now)

All in the windsurf-project directory!
