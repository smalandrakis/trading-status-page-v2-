# BTC Trading Model Package - Integration Guide

## Package Location

```
/Users/smalandrakis/Documents/WindSurf/DS:ML trade model playground/CascadeProjects/windsurf-project/btc_model_package/
```

## What You Need for Your Windsurf Bot

### 1. Model Files (Will be regenerated for compatibility)
- `btc_2h_model_v3.pkl` - 2-hour horizon model
- `btc_4h_model_v3.pkl` - 4-hour horizon model  
- `btc_6h_model_v3.pkl` - 6-hour horizon model
- `btc_*_scaler_v3.pkl` - Feature scalers (3 files)
- `selected_features.json` - List of 22 features

### 2. Prediction Interface
- `predictor.py` - Ready-to-use class

## Integration with IB Gateway + Binance Data

Since Binance API is restricted in Netherlands, here's the hybrid approach:

### Data Flow
```
Binance Public API (free, real-time)
    ↓ (fetch 5-min candles)
predictor.py (calculate features + predict)
    ↓ (LONG/SHORT signal)
Your Windsurf Bot
    ↓ (execute trades)
IB Gateway Port 4002 (paper trading)
```

### Minimal Code for Your Bot

```python
from binance.client import Client
from predictor import BTCPredictor
import pandas as pd

# Initialize once
binance = Client("", "")  # Public API, no keys needed
predictor = BTCPredictor(model_dir="/path/to/btc_model_package")

# In your trading loop (every 2 minutes)
def get_signal():
    # 1. Fetch data from Binance
    klines = binance.get_klines(
        symbol='BTCUSDT',
        interval=Client.KLINE_INTERVAL_5MINUTE,
        limit=250
    )
    
    # 2. Convert to DataFrame
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])
    
    df = df[['open', 'high', 'low', 'close', 'volume']]
    
    # 3. Get prediction
    signal, confidence, details = predictor.predict(df)
    
    return signal, confidence, details

# In your trading logic
signal, confidence, details = get_signal()

if signal == 'LONG' and confidence > 0.6:
    # Execute via IB Gateway
    entry_price = details['current_price']
    tp, sl = predictor.get_tp_sl_prices(entry_price, 'LONG')
    # Place bracket order on IB Gateway
    
elif signal == 'SHORT' and confidence > 0.6:
    # Execute via IB Gateway
    entry_price = details['current_price']
    tp, sl = predictor.get_tp_sl_prices(entry_price, 'SHORT')
    # Place bracket order on IB Gateway
```

## Key Parameters

### Model Settings (Don't Change)
- **TP**: 1% (both directions)
- **SL**: 0.5% (both directions)
- **Breakeven WR**: 33.3%
- **Thresholds**: LONG >0.55, SHORT <0.35

### Your Bot Settings (Adjustable)
- **Confidence Filter**: Recommend >0.6 (stricter than model default)
- **Position Size**: Your choice
- **Check Interval**: 2-5 minutes
- **Max Positions**: 1-2 concurrent

## Data Requirements

### From Binance API
- **Symbol**: BTCUSDT
- **Interval**: 5-minute candles
- **Limit**: 250 bars (latest)
- **Update**: Every 2 minutes

### No Subscription Needed
- Binance public API is free
- Real-time data (no delay)
- No geographic restrictions for data
- 24/7 availability

## Expected Performance

Based on walk-forward validation (10,104 trades):
- **Win Rate**: 54.24%
- **Avg P&L**: +0.31% per trade
- **Total P&L**: +3,169% over test period
- **Sharpe Ratio**: ~1.8

**Note**: Past performance doesn't guarantee future results. Start with small position sizes.

## Troubleshooting

### If predictions seem wrong:
1. Check data quality (no NaN, correct columns)
2. Verify 250 bars minimum
3. Ensure 5-minute timeframe (not 1-minute)
4. Check feature calculation (no errors in logs)

### If models don't load:
- Models are being regenerated for compatibility
- Wait for `train_v3_models.py` to complete
- Check sklearn version matches (1.6.1)

## Files Summary

**Essential:**
- `predictor.py` (350 lines, well-documented)
- `btc_*_model_v3.pkl` (3 files, ~240KB each)
- `btc_*_scaler_v3.pkl` (3 files, ~1.5KB each)
- `selected_features.json` (22 feature names)

**Optional:**
- `README.md` (full documentation)
- `test_predictor.py` (testing script)
- `*_metrics_v3.json` (performance stats)

**Total Size**: ~750KB

## Next Steps

1. **Wait for model regeneration** to complete (~2 minutes)
2. **Copy entire `btc_model_package/` folder** to your Windsurf workspace
3. **Test with**: `python3 test_predictor.py`
4. **Integrate** predictor into your existing bot collection
5. **Paper trade** via IB Gateway port 4002 to validate

## Contact

All code is in the windsurf-project directory. Review these files for full implementation:
- `walk_forward_CORRECTED.py` - Validated backtesting logic
- `complete_code_review.py` - Full verification results
- `create_symmetric_labels.py` - Label creation logic
