# Trading Bot Configuration & Strategy Reference
**Last Updated: January 7, 2026**

## Bot Overview

| Bot | File | Log File | Client ID | Symbol | Parquet |
|-----|------|----------|-----------|--------|---------|
| **MNQ** | `ensemble_bot.py` | `logs/mnq_bot.log` | 4 | MNQ futures | `data/QQQ_features.parquet` |
| **SPY** | `spy_ensemble_bot.py` | `logs/spy_bot.log` | 3 | MES futures | `data/QQQ_features.parquet` (uses MNQ parquet!) |
| **BTC** | `btc_ensemble_bot.py` | `logs/btc_bot.log` | 1 | MBT futures | `data/BTC_features.parquet` |

### Important: SPY Bot Uses QQQ Parquet
The SPY bot (`spy_ensemble_bot.py`) uses `data/QQQ_features.parquet` (same as MNQ) because:
- SPY bot uses MNQ models (trained on QQQ data)
- Line 890: `parquet_path = 'data/QQQ_features.parquet'`
- The old `data/SPY_features.parquet` is NOT used

---

## MNQ Bot Configuration (`ensemble_bot.py`)

### Active Features
| Feature | Status | Config |
|---------|--------|--------|
| **TREND_FILTER** | ✓ Enabled | Blocks SHORT trades during uptrends (SMA 5 > SMA 20) |
| **TREND_FOLLOW** | ✓ Enabled | Rides strong trends when ROC > 1.5% |
| **INDICATOR_LONG** | ✗ Disabled | BB %B < 0.5 mean reversion (disabled due to poor performance) |
| **ML Models** | ✓ Active | 2h_0.5pct, 4h_0.5pct, 2h_0.5pct_SHORT, 4h_0.5pct_SHORT |

### Key Parameters
```python
TREND_FILTER_ENABLED = True
TREND_FILTER_SMA_SHORT = 5
TREND_FILTER_SMA_LONG = 20

TREND_FOLLOW_ENABLED = True
TREND_FOLLOW_SL_PCT = 0.50
TREND_FOLLOW_TP_PCT = 1.50
TREND_FOLLOW_TRAILING_PCT = 0.30
TREND_FOLLOW_MIN_ROC = 1.5

INDICATOR_LONG_ENABLED = False  # Disabled Jan 5, 2026
```

---

## BTC Bot Configuration (`btc_ensemble_bot.py`)

### Active Features
| Feature | Status | Config |
|---------|--------|--------|
| **INDICATOR_TREND** | ✓ Enabled | ROC + MACD trend following (LONG & SHORT) |
| **INDICATOR_LONG** | ✗ Disabled | BB > 0.5 + MACD momentum |
| **INDICATOR_MEANREV** | ✗ Disabled | BB < 0.5 mean reversion |
| **ML LONG Models** | ✗ Disabled | Underperforming |
| **ML SHORT Models** | ✓ Active | 2h_0.5pct_SHORT, 4h_0.5pct_SHORT |

### Key Parameters
```python
STOP_LOSS_PCT = 0.10  # General ML model SL
INDICATOR_LONG_ENABLED = False
INDICATOR_MEANREV_ENABLED = False
INDICATOR_TREND_ENABLED = True

INDICATOR_LONG_SL_PCT = 0.30
INDICATOR_LONG_TP_PCT = 0.50
```

### BTC indicator_long SL Bug (Fixed Jan 6, 2026)
- **Bug**: 01-05 trades used wrong 0.10% SL instead of configured 0.30%
- **Fix**: 01-06 trades correctly use 0.30% SL
- **Root Cause**: Code path was falling through to general STOP_LOSS_PCT
- **Status**: Bug fixed, but strategy still unprofitable - kept disabled

---

## SPY Bot Configuration (`spy_ensemble_bot.py`)

### Active Features
| Feature | Status | Config |
|---------|--------|--------|
| **ML Models** | ✓ Active | Uses MNQ models (4h_0.5pct, 2h_0.5pct) |

### Key Note
- Uses QQQ parquet (`data/QQQ_features.parquet`), NOT SPY parquet
- Uses MNQ models from `models_mnq_v2/`

---

## Monitor Daemon (`monitor_daemon.py`)

### Features
- Checks parquet health (staleness)
- Validates signal match between parquet and fresh data
- Auto-refreshes parquet when mismatch detected
- Connected to GitHub for deployment

### Signal Mismatch Fix (Jan 7, 2026)
- **Bug**: After parquet refresh, compared different timestamps
- **Fix**: Now finds common timestamps and compares at same timestamp after refresh

---

## Strategy Investigation History

### What Works
1. **indicator_trend** (BTC): +$148.47 (1 trade, 100% WR) - ROC + MACD trend following
2. **2h_0.5pct LONG** (BTC): +$147.96 (7 trades, 43% WR)
3. **4h_0.5pct LONG** (BTC): +$19.40 (20 trades, 25% WR)
4. **TREND_FILTER**: Blocks SHORT trades during uptrends - prevents losses

### What Doesn't Work
1. **2h_0.5pct_SHORT**: -$272.98 (61 trades, 31% WR) - shorting into uptrend
2. **4h_0.5pct_SHORT**: -$151.62 (40 trades, 20% WR) - shorting into uptrend
3. **indicator_long**: -$92.88 (20 trades, 10% WR) - SL bug + poor strategy
4. **indicator_meanrev**: -$51.07 (4 trades, 25% WR)

### Key Findings
- **SHORT models lose money during uptrends** - need trend filter
- **Commissions eat profits** - $1.24/trade for MNQ, $2.02/trade for BTC
- **ML models need higher thresholds** to reduce trade frequency
- **Trend-following strategies** outperform mean reversion in trending markets

---

## File Locations

### Bot Files
- `ensemble_bot.py` - MNQ bot
- `spy_ensemble_bot.py` - SPY bot (uses MNQ models)
- `btc_ensemble_bot.py` - BTC bot

### Data Files
- `data/QQQ_features.parquet` - MNQ/SPY features (refreshed regularly)
- `data/BTC_features.parquet` - BTC features
- `data/SPY_features.parquet` - OLD, NOT USED

### Model Directories
- `models_mnq_v2/` - MNQ models (used by MNQ and SPY bots)
- `models_mnq_spy_v3/` - Alternative models

### Log Files
- `logs/mnq_bot.log` - MNQ bot logs
- `logs/spy_bot.log` - SPY bot logs
- `logs/btc_bot.log` - BTC bot logs
- `logs/monitor_daemon.log` - Monitor logs

### Database
- `trades.db` - SQLite database with `trades` table for performance tracking

---

## Starting/Stopping Bots

### Start Commands
```bash
# MNQ Bot (paper trading by default)
nohup python3 ensemble_bot.py > /dev/null 2>&1 &

# SPY Bot
nohup python3 spy_ensemble_bot.py > /dev/null 2>&1 &

# BTC Bot (paper trading)
nohup python3 btc_ensemble_bot.py --paper > /dev/null 2>&1 &
```

### Check Running Bots
```bash
ps aux | grep -E "ensemble_bot|spy_ensemble|btc_ensemble" | grep -v grep
```

---

## GitHub Deployment
The monitor daemon is connected to GitHub for deployment. Parquet refresh and monitoring can be deployed as needed.
