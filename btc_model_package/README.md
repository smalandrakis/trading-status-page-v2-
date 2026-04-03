# BTC Trading Model Package - V3

Complete package of trained models for BTC trade direction prediction.

## Model Performance

- **Dataset**: 4 years of BTC 5-minute data (2022-2026, 188K samples)
- **Strategy**: 1% Take Profit / 0.5% Stop Loss (symmetric for both directions)
- **Risk/Reward**: 2:1 ratio → only need 33.3% WR to break even
- **Validated Results**: 54.24% WR, +0.31% avg P&L per trade
- **Walk-forward tested**: 5 periods, no data leakage

## Files Included

```
btc_model_package/
├── predictor.py                  # Simple prediction interface
├── btc_2h_model_v3.pkl          # 2-hour horizon model (231KB)
├── btc_2h_scaler_v3.pkl         # Feature scaler for 2h model
├── btc_4h_model_v3.pkl          # 4-hour horizon model (242KB)
├── btc_4h_scaler_v3.pkl         # Feature scaler for 4h model
├── btc_6h_model_v3.pkl          # 6-hour horizon model (244KB)
├── btc_6h_scaler_v3.pkl         # Feature scaler for 6h model
├── selected_features.json        # List of 22 required features
├── btc_2h_metrics_v3.json       # Model performance metrics
├── btc_4h_metrics_v3.json
├── btc_6h_metrics_v3.json
└── README.md                     # This file
```

## Quick Start

### 1. Import the predictor

```python
from predictor import BTCPredictor

predictor = BTCPredictor()
```

### 2. Get data from your source (Binance, IB, etc.)

You need at least 200 x 5-minute candles (~16-17 hours of data)

```python
# Example with pandas DataFrame
df = pd.DataFrame({
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...]
})
```

### 3. Get prediction

```python
signal, confidence, details = predictor.predict(df)

if signal == 'LONG' and confidence > 0.6:
    # Enter long position
    entry_price = details['current_price']
    tp, sl = predictor.get_tp_sl_prices(entry_price, 'LONG')
    # Place order with TP and SL

elif signal == 'SHORT' and confidence > 0.6:
    # Enter short position
    entry_price = details['current_price']
    tp, sl = predictor.get_tp_sl_prices(entry_price, 'SHORT')
    # Place order with TP and SL
```

## Data Requirements

### Input Format

DataFrame with these columns (exact names):
- `open` - Opening price
- `high` - High price  
- `low` - Low price
- `close` - Closing price
- `volume` - Trading volume

### Minimum Data

- **At least 200 rows** (5-minute candles)
- More is better (up to 250 rows used)
- Latest row = most recent candle

### Time Resolution

- Models are trained on **5-minute candles**
- If you have 1-minute data, resample to 5-minute first

## Features Used (22 total)

The predictor automatically calculates these from OHLCV data:

**Moving Averages:**
- close_20, close_50, volume_ma_48

**Support/Resistance:**
- resistance_48, support_48

**Volatility:**
- atr_48_pct, range_4h, vol_20, vol_48

**Momentum:**
- rsi_14, adx_proxy, return_1, return_5, return_10

**Price Position:**
- price_to_resistance, price_to_support
- dist_to_ma20, dist_to_ma50
- bb_position, price_position

**Trend:**
- ma20_slope, volume_ratio

## Signal Logic

### Ensemble Strategy

1. Three models predict (2h, 4h, 6h horizons)
2. Average the probabilities
3. Apply asymmetric thresholds:
   - `avg_prob > 0.55` → **LONG** signal
   - `avg_prob < 0.35` → **SHORT** signal  
   - Between → **NEUTRAL** (no trade)

### Why Asymmetric?

- LONG trades have 2:1 R:R (need only 33% WR)
- SHORT trades also have 2:1 R:R (need only 33% WR)
- But we favor LONG with lower threshold since both are symmetric

## Trading Rules

### Entry
- Only trade when signal = LONG or SHORT
- Recommended: Add confidence filter (>0.6)
- Wait for NEUTRAL to close before new entry

### Exit (1% TP / 0.5% SL)

**LONG Position:**
- Take Profit: Entry × 1.01 (+1%)
- Stop Loss: Entry × 0.995 (-0.5%)

**SHORT Position:**
- Take Profit: Entry × 0.99 (-1%)  
- Stop Loss: Entry × 1.005 (+0.5%)

## Example Integration

### With Binance Data

```python
from binance.client import Client
from predictor import BTCPredictor
import pandas as pd

# Get data from Binance
client = Client("", "")  # Public API
klines = client.get_klines(
    symbol='BTCUSDT',
    interval=Client.KLINE_INTERVAL_5MINUTE,
    limit=250
)

# Convert to DataFrame
df = pd.DataFrame(klines, columns=[
    'timestamp', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_volume', 'trades', 
    'taker_buy_base', 'taker_buy_quote', 'ignore'
])

for col in ['open', 'high', 'low', 'close', 'volume']:
    df[col] = pd.to_numeric(df[col])

# Get prediction
predictor = BTCPredictor()
signal, confidence, details = predictor.predict(df)

print(f"Signal: {signal} ({confidence:.1%})")
```

### With IB Gateway Data

```python
from ib_insync import IB, Future
from predictor import BTCPredictor

ib = IB()
ib.connect('127.0.0.1', 4002, clientId=1)

# Get BTC futures data
contract = Future(symbol='MBT', exchange='CME')
bars = ib.reqHistoricalData(
    contract,
    endDateTime='',
    durationStr='2 D',
    barSizeSetting='5 mins',
    whatToShow='TRADES',
    useRTH=False
)

# Convert to DataFrame
df = pd.DataFrame(bars)[['open', 'high', 'low', 'close', 'volume']]

# Get prediction
predictor = BTCPredictor()
signal, confidence, details = predictor.predict(df)
```

## Model Details

### Training Data
- 4 years: 2022-2026 (188,249 samples after processing)
- 5-minute BTC/USDT candles
- Labels: First-touch TP/SL logic (symmetric 1%/0.5%)

### Algorithm
- Gradient Boosting Classifier (sklearn)
- 100 estimators, max_depth=4
- Regularization: min_samples_split=20, min_samples_leaf=10
- Class balancing with sample weights

### Validation
- Walk-forward testing (5 periods)
- Train/test: time-based split (no shuffle)
- No data leakage (train always before test)
- Test accuracy: ~53%
- Overfitting gap: ~13%

## Backtest Results

```
Period              Trades    Win Rate    Avg P&L%    Total P&L%
─────────────────────────────────────────────────────────────────
2024 Q1            1,837     49.10%      +0.2200     +404.14
2024 Q2            2,084     56.05%      +0.4405     +918.00
2024 Q3            2,098     57.58%      +0.4258     +893.34
2024 Q4            2,145     53.33%      +0.2767     +593.62
2025-2026          1,940     52.47%      +0.2335     +452.99
─────────────────────────────────────────────────────────────────
OVERALL            10,104    54.24%      +0.3136     +3,169.09
```

**Validation:** Expected P&L at 54.24% WR = 0.3136% ✓ Matches exactly!

## Support

For issues or questions:
- Check the `predictor.py` source code
- Review original training scripts in parent directory
- See `complete_code_review.py` for full validation

## License

For personal/educational use.
