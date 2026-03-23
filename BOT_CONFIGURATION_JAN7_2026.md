# Project Freedom - Bot Configuration Reference
## Date: January 7, 2026

---

# Table of Contents
1. [Overview](#overview)
2. [BTC Bot Configuration](#btc-bot-configuration)
3. [MNQ Bot Configuration](#mnq-bot-configuration)
4. [SPY Bot Configuration](#spy-bot-configuration)
5. [Expected Signals Per Day](#expected-signals-per-day)
6. [Backtest Results Summary](#backtest-results-summary)
7. [Monitor Daemon Configuration](#monitor-daemon-configuration)
8. [Data Sources](#data-sources)
9. [File Locations](#file-locations)

---

# Overview

Three trading bots running on IB Gateway (paper trading):
- **BTC Bot**: Trades MBT (Micro Bitcoin Futures) 24/7
- **MNQ Bot**: Trades MNQ (Micro Nasdaq Futures) during market hours
- **SPY Bot**: Trades MES (Micro S&P Futures) during market hours

All bots use ML V2 models trained on historical data with feature engineering.

---

# BTC Bot Configuration

## File: `btc_ensemble_bot.py`

### Models Enabled (5 total)
| Model | Direction | Threshold | Cooldown | Priority |
|-------|-----------|-----------|----------|----------|
| 2h_0.5pct | LONG | 40% | 24 bars (2h) | 3 |
| 4h_0.5pct | LONG | 40% | 48 bars (4h) | 2 |
| 6h_0.5pct | LONG | 40% | 72 bars (6h) | 1 |
| 2h_0.5pct_SHORT | SHORT | 55% | 24 bars (2h) | 2 |
| 4h_0.5pct_SHORT | SHORT | 55% | 48 bars (4h) | 1 |

### Risk Parameters
| Parameter | Value |
|-----------|-------|
| Stop Loss | 0.70% |
| Take Profit | 1.40% |
| Max LONG Positions | 2 |
| Max SHORT Positions | 2 |
| Trailing Stop | DISABLED |
| Timeout Multiplier | 2x horizon |

### Indicator Strategies
| Strategy | Status | Conditions |
|----------|--------|------------|
| indicator_trend LONG | ENABLED | ROC(12) > 0.4% AND MACD > Signal AND MACD increasing |
| indicator_trend SHORT | ENABLED | ROC(12) < -0.4% AND MACD < Signal AND MACD decreasing |
| indicator_long (momentum) | DISABLED | BB%B > 0.5 AND MACD > Signal (not profitable) |

### IB Configuration
- Client ID: 22
- Contract: MBT (Micro Bitcoin Futures)
- Contract Value: 0.1 BTC
- Commission: $2.02/trade

### Expected Performance (from backtest)
| Model | WR% | $/month |
|-------|-----|---------|
| 6h_0.5pct LONG | 46% | +$1,724 |
| 4h_0.5pct LONG | 48% | +$953 |
| 2h_0.5pct LONG | 50% | +$942 |
| 4h_0.5pct SHORT | 45% | +$1,229 |
| 2h_0.5pct SHORT | 46% | +$850 |
| **TOTAL** | | **+$5,698/month** |

---

# MNQ Bot Configuration

## File: `ensemble_bot.py`

### Models Enabled (4 total)
| Model | Direction | Threshold | Cooldown | Priority |
|-------|-----------|-----------|----------|----------|
| 2h_0.5pct | LONG | 50% | 24 bars (2h) | 2 |
| 4h_0.5pct | LONG | 50% | 48 bars (4h) | 1 |
| 2h_0.5pct_SHORT | SHORT | 50% | 24 bars (2h) | 2 |
| 4h_0.5pct_SHORT | SHORT | 50% | 48 bars (4h) | 1 |

### Risk Parameters
| Parameter | Value |
|-----------|-------|
| Stop Loss | 0.50% |
| Take Profit | 1.00% |
| Max LONG Positions | 2 |
| Max SHORT Positions | 2 |
| Trailing Stop | DISABLED (activation at 999%) |
| Timeout Multiplier | 2x horizon |

### Trend Filter
- **Enabled**: Yes
- **SMA Short**: 20 bars
- **SMA Long**: 50 bars
- **ROC Threshold**: 0.5%
- **Effect**: Blocks SHORT trades during strong uptrends

### IB Configuration
- Client ID: 13 (config.IB_CLIENT_ID + 3)
- Contract: MNQ (Micro Nasdaq Futures)
- Point Value: $2/point
- Commission: $1.24/trade

### Expected Performance (from backtest)
| Model | WR% | $/month |
|-------|-----|---------|
| 4h_0.5pct LONG | 78% | +$122 |
| 2h_0.5pct LONG | 69% | +$72 |
| 4h_0.5pct SHORT | 72% | +$94 |
| 2h_0.5pct SHORT | 84% | +$60 |
| **TOTAL** | | **+$348/month** |

---

# SPY Bot Configuration

## File: `spy_ensemble_bot.py`

### Models Enabled (3 ML V2 + 1 Indicator + 1 ML V3)
| Model | Direction | Threshold | Cooldown | Priority |
|-------|-----------|-----------|----------|----------|
| 4h_0.5pct | LONG | 65% | 48 bars (4h) | 1 |
| 2h_0.5pct_SHORT | SHORT | 65% | 24 bars (2h) | 1 |
| 4h_0.5pct_SHORT | SHORT | 65% | 48 bars (4h) | 2 |
| indicator_long | LONG | N/A | 48 bars | 0 |
| ML V3 (1.0pct_48bars) | LONG | 55% | 96 bars | - |

### Risk Parameters
| Parameter | Value |
|-----------|-------|
| Stop Loss | 0.50% |
| Take Profit | 1.00% |
| Max LONG Positions | 1 |
| Max SHORT Positions | 2 |
| Max Indicator LONG | 1 |
| Max ML V3 Positions | 1 |
| Trailing Stop | DISABLED |

### Indicator Strategy (indicator_long)
- **Conditions**: RSI < 40 AND BB %B < 0.4
- **SL**: 0.50%
- **TP**: 1.00%
- **Timeout**: 48 bars (4h)

### ML V3 Model
- **Path**: `models_mnq_spy_v3/model_long_1.0pct_48bars.joblib`
- **Threshold**: 55%
- **SL**: 1.00%
- **TP**: 2.00%
- **Timeout**: 96 bars (8h)

### IB Configuration
- Client ID: 12 (config.IB_CLIENT_ID + 2)
- Contract: MES (Micro S&P Futures)
- Point Value: $5/point
- Commission: $1.24/trade

### Expected Performance (from backtest)
| Model | WR% | $/month |
|-------|-----|---------|
| 4h_0.5pct LONG | 79% | +$348 |
| 4h_0.5pct SHORT | 74% | +$271 |
| 2h_0.5pct SHORT | 92% | +$166 |
| **TOTAL** | | **+$785/month** |

---

# Expected Signals Per Day

## BTC (24/7 trading, 288 bars/day)
| Model | Raw Signals | With Cooldown |
|-------|-------------|---------------|
| 2h_0.5pct LONG | 54/day | 12/day |
| 4h_0.5pct LONG | 48/day | 6/day |
| 6h_0.5pct LONG | 116/day | 4/day |
| 2h_0.5pct SHORT | 24/day | 12/day |
| 4h_0.5pct SHORT | 33/day | 6/day |
| **TOTAL** | 274/day | **~40/day** |

**Expected trades/week: ~280**

## MNQ (Market hours, ~78 bars/day)
| Model | Raw Signals | With Cooldown |
|-------|-------------|---------------|
| 2h_0.5pct LONG | 8/day | 3/day |
| 4h_0.5pct LONG | 7/day | 2/day |
| 2h_0.5pct SHORT | 3/day | 3/day |
| 4h_0.5pct SHORT | 7/day | 2/day |
| **TOTAL** | 24/day | **~9/day** |

**Expected trades/week: ~46**

## SPY (Market hours, ~78 bars/day, 65% threshold)
| Model | Raw Signals | With Cooldown |
|-------|-------------|---------------|
| 4h_0.5pct LONG | 3/day | 2/day |
| 2h_0.5pct SHORT | 1/day | 1/day |
| 4h_0.5pct SHORT | 4/day | 2/day |
| indicator_long | 15/day | 2/day |
| **TOTAL** | 23/day | **~6/day** |

**Expected trades/week: ~31**

---

# Backtest Results Summary

## BTC Models (SL=0.70%, TP=1.40%)
| Model | Trades | WR% | Total P&L | $/month |
|-------|--------|-----|-----------|---------|
| 6h_0.5pct | 8,411 | 46% | +$41,373 | +$1,724 |
| 4h_0.5pct | 6,120 | 48% | +$22,868 | +$953 |
| 2h_0.5pct | 5,042 | 50% | +$22,608 | +$942 |
| 4h_0.5pct_SHORT | 6,380 | 45% | +$29,506 | +$1,229 |
| 2h_0.5pct_SHORT | 6,145 | 46% | +$20,411 | +$850 |

## MNQ Models (SL=0.50%, TP=1.00%)
| Model | Trades | WR% | Total P&L | $/month |
|-------|--------|-----|-----------|---------|
| 4h_0.5pct | 67 | 78% | +$349 | +$122 |
| 2h_0.5pct | 74 | 69% | +$206 | +$72 |
| 4h_0.5pct_SHORT | 54 | 72% | +$270 | +$94 |
| 2h_0.5pct_SHORT | 25 | 84% | +$171 | +$60 |

## SPY Models (SL=0.50%, TP=1.00%)
| Model | Trades | WR% | Total P&L | $/month |
|-------|--------|-----|-----------|---------|
| 4h_0.5pct | 67 | 79% | +$998 | +$348 |
| 4h_0.5pct_SHORT | 54 | 74% | +$775 | +$271 |
| 2h_0.5pct_SHORT | 25 | 92% | +$475 | +$166 |

---

# Monitor Daemon Configuration

## File: `monitor_daemon.py`

### Check Interval
- **Frequency**: Every 30 minutes
- **Backtest**: 4x daily (00:00, 06:00, 12:00, 18:00 UTC)

### Auto-Refresh Triggers
1. **Parquet unhealthy** (missing bars, stale data) → Refresh parquet
2. **Signal mismatch > 5%** → Refresh parquet + re-verify

### Status Output
- **File**: `status_page/status.json`
- **Dashboard**: `status_page/index.html`

### BOT_CONFIG in Daemon
```python
BOT_CONFIG = {
    'BTC': {
        'SL_PCT': 0.70,
        'TP_PCT': 1.40,
        'models': ['2h_0.5pct', '4h_0.5pct', '6h_0.5pct', '2h_0.5pct_SHORT', '4h_0.5pct_SHORT'],
        'thresholds': {
            '2h_0.5pct': 0.40,
            '4h_0.5pct': 0.40,
            '6h_0.5pct': 0.40,
            '2h_0.5pct_SHORT': 0.55,
            '4h_0.5pct_SHORT': 0.55,
        },
    },
    'MNQ': {
        'SL_PCT': 0.5,
        'TP_PCT': 1.0,
        'models': ['2h_0.5pct', '4h_0.5pct', '2h_0.5pct_SHORT', '4h_0.5pct_SHORT'],
        'thresholds': {
            '2h_0.5pct': 0.50,
            '4h_0.5pct': 0.50,
            '2h_0.5pct_SHORT': 0.50,
            '4h_0.5pct_SHORT': 0.50,
        },
    },
    'SPY': {
        'SL_PCT': 0.5,
        'TP_PCT': 1.0,
        'models': ['4h_0.5pct', '2h_0.5pct_SHORT', '4h_0.5pct_SHORT'],
        'thresholds': {
            '4h_0.5pct': 0.65,
            '2h_0.5pct_SHORT': 0.65,
            '4h_0.5pct_SHORT': 0.65,
        },
        'indicator_long': {'RSI': 40, 'BB_PCT_B': 0.4},
        'ml_v3_threshold': 0.55,
    },
}
```

---

# Data Sources

## BTC
- **Real-time price**: Binance WebSocket (BTCUSDT)
- **Features**: `data/BTC_features.parquet`
- **Refresh**: Every 5 minutes from Binance API

## MNQ
- **Market hours**: Finnhub QQQ (`finnhub_qqq`)
- **Overnight**: Yahoo MNQ=F → converted to QQQ scale (`yahoo_mnq`)
- **Features**: `data/QQQ_features.parquet`
- **Refresh**: Every 5 minutes from Yahoo Finance

## SPY
- **Market hours**: Finnhub SPY (`finnhub_spy`)
- **Overnight**: Yahoo ES=F → converted to SPY scale (`yahoo_es`)
- **Features**: `data/QQQ_features.parquet` (same as MNQ)
- **Models**: `models_mnq_v2/` (same as MNQ)

---

# File Locations

## Bot Files
- `btc_ensemble_bot.py` - BTC trading bot
- `ensemble_bot.py` - MNQ trading bot
- `spy_ensemble_bot.py` - SPY trading bot
- `monitor_daemon.py` - Monitoring and auto-refresh

## Model Files
- `models_btc_v2/` - BTC V2 models (2h, 4h, 6h, 8h, 12h horizons)
- `models_mnq_v2/` - MNQ/SPY V2 models (2h, 4h horizons)
- `models_mnq_spy_v3/` - V3 models for SPY

## Data Files
- `data/BTC_features.parquet` - BTC feature data
- `data/QQQ_features.parquet` - QQQ/MNQ/SPY feature data

## Log Files
- `logs/btc_bot.log` - BTC bot logs
- `logs/mnq_bot.log` - MNQ bot logs
- `logs/spy_bot.log` - SPY bot logs
- `logs/monitor_daemon.log` - Monitor daemon logs

## Signal Logs
- `signal_logs/btc_signals_YYYY-MM-DD.csv` - BTC signals
- `signal_logs/mnq_signals_YYYY-MM-DD.csv` - MNQ signals
- `signal_logs/spy_signals_YYYY-MM-DD.csv` - SPY signals

## Position Files
- `btc_positions.json` - BTC open positions
- `ensemble_positions.json` - MNQ open positions
- `spy_positions.json` - SPY open positions

## Configuration
- `config.py` - Shared configuration (IB settings, etc.)
- `status_page/index.html` - Dashboard HTML
- `status_page/status.json` - Status data for dashboard

---

# Change Log

## January 7, 2026
- **BTC**: Enabled ALL models (2h, 4h, 6h LONG + 2h, 4h SHORT)
- **BTC**: Changed SL from 1.0% to 0.70%, TP from 2.0% to 1.40%
- **MNQ**: Changed SL from 0.75% to 0.50%, TP from 0.50% to 1.00%
- **SPY**: Changed SL from 0.75% to 0.50%, TP from 0.50% to 1.00%
- **Monitor Daemon**: Updated BOT_CONFIG with all thresholds
- **Dashboard**: Updated configuration display

---

*Document generated: January 7, 2026 23:09 CET*
