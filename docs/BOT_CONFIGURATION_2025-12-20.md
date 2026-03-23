# Trading Bot Configuration Documentation
## Date: December 20, 2025

---

# OVERVIEW

Two trading bots are deployed and running:
1. **BTC Bot** (`btc_ensemble_bot.py`) - Trades MBT (Micro Bitcoin Futures)
2. **MNQ Bot** (`ensemble_bot.py`) - Trades MNQ (Micro Nasdaq Futures)

Both use the **V2 training approach**:
- Random 75/25 train/test split (not time-based)
- Last 2 years of data
- 206 features (TA indicators, time, lagged, price ratios)
- HistGradientBoostingClassifier models

---

# BTC BOT CONFIGURATION

## File: `btc_ensemble_bot.py`

### Models (from `models_btc_v2/`)
| Model | Direction | Horizon | Target | Features |
|-------|-----------|---------|--------|----------|
| 2h_0.5pct | LONG | 2 hours | 0.5% | 206 |
| 4h_0.5pct | LONG | 4 hours | 0.5% | 206 |
| 2h_0.5pct_SHORT | SHORT | 2 hours | 0.5% | 206 |
| 4h_0.5pct_SHORT | SHORT | 4 hours | 0.5% | 206 |

### Entry Parameters
| Parameter | Value |
|-----------|-------|
| Probability threshold | **65%** |
| Max LONG positions | 2 |
| Max SHORT positions | 2 |
| Position size | 1 contract |

### Exit Parameters
| Parameter | Value |
|-----------|-------|
| Stop Loss | **0.75%** |
| Target | 0.5% (model threshold) |
| Trailing Stop | 0.15% distance |
| Trailing Activation | 0.15% profit |
| Timeout | 2x horizon |

### Data Configuration
| Parameter | Value |
|-----------|-------|
| Data source | IB Gateway (MBT 5-min bars) |
| Trading contract | MBT (Micro Bitcoin Futures) |
| Feature count | 206 |
| Signal check interval | 15 seconds |
| Feature update | Every 5-min bar |
| IB Client ID | 11 |
| IB Port | 4002 |

### Expected Performance
| Metric | Value |
|--------|-------|
| Trades/day | ~4-6 |
| Win Rate | ~70-75% |
| Break-even WR | ~40% |
| Buffer | ~30-35% |

---

# MNQ BOT CONFIGURATION

## File: `ensemble_bot.py`

### Models (from `models_mnq_v2/`)
| Model | Direction | Horizon | Target | Features |
|-------|-----------|---------|--------|----------|
| 2h_0.5pct | LONG | 2 hours | 0.5% | 206 |
| 4h_0.5pct | LONG | 4 hours | 0.5% | 206 |
| 2h_0.5pct_SHORT | SHORT | 2 hours | 0.5% | 206 |
| 4h_0.5pct_SHORT | SHORT | 4 hours | 0.5% | 206 |

### Entry Parameters
| Parameter | Value |
|-----------|-------|
| Probability threshold | **65%** |
| Max LONG positions | 2 |
| Max SHORT positions | 2 |
| Position size | 1 contract |

### Exit Parameters
| Parameter | Value |
|-----------|-------|
| Stop Loss | **0.75%** |
| Target | 0.5% (model threshold) |
| Trailing Stop | 0.15% distance |
| Trailing Activation | 0.15% profit |
| Timeout | 2x horizon |

### Data Configuration
| Parameter | Value |
|-----------|-------|
| Feature source | QQQ (5-min bars from IB) |
| Trading contract | MNQ (MNQH6 - March 2026) |
| Feature count | 206 |
| Signal check interval | 15 seconds |
| Feature update | Every 5-min bar |
| IB Client ID | 2 |
| IB Port | 4002 |

### Expected Performance (Backtest Nov 20 - Dec 18, 2025)
| Metric | Value |
|--------|-------|
| Trades/day | ~6-8 |
| Win Rate | **83.9%** |
| Break-even WR | 39.9% |
| Buffer | **43.9%** |
| Total P&L (20 days) | +46% |

---

# SIDE-BY-SIDE COMPARISON

| Parameter | BTC Bot | MNQ Bot |
|-----------|---------|---------|
| **Bot file** | btc_ensemble_bot.py | ensemble_bot.py |
| **Model directory** | models_btc_v2/ | models_mnq_v2/ |
| **Trading instrument** | MBT (Micro Bitcoin) | MNQ (Micro Nasdaq) |
| **Data source** | MBT (direct) | QQQ (proxy) |
| **Models** | 2h+4h LONG/SHORT | 2h+4h LONG/SHORT |
| **Threshold** | 0.5% | 0.5% |
| **Probability** | 65% | 65% |
| **Stop Loss** | 0.75% | 0.75% |
| **Trailing Stop** | 0.15% @ 0.15% | 0.15% @ 0.15% |
| **Timeout** | 2x horizon | 2x horizon |
| **Max positions** | 2 LONG + 2 SHORT | 2 LONG + 2 SHORT |
| **Features** | 206 | 206 |
| **IB Client ID** | 11 | 2 |

---

# MODEL TRAINING DETAILS

## V2 Training Approach (Both Bots)
- **Data period**: Last 2 years
- **Split method**: Random 75/25 (not time-based)
- **Algorithm**: HistGradientBoostingClassifier
- **Hyperparameters**:
  - max_iter: 300
  - max_depth: 5
  - learning_rate: 0.03
  - early_stopping: True
  - validation_fraction: 0.15
  - n_iter_no_change: 30
  - min_samples_leaf: 50
  - l2_regularization: 1.0

## Feature Categories (206 total)
- **Technical indicators** (ta library): ~85 features
- **Time features**: 13 features (hour, day, session, etc.)
- **Price features**: 15 features (returns, ratios, gaps)
- **Daily context**: 10 features (prev day OHLC, returns)
- **Lagged indicators**: ~70 features (RSI, MACD, etc. at lags 1-50)
- **Indicator changes**: 8 features (rate of change)

---

# FILE STRUCTURE

```
windsurf-project/
├── btc_ensemble_bot.py          # BTC trading bot
├── ensemble_bot.py              # MNQ trading bot
├── data_validator.py            # Data validation module (added Dec 20)
├── train_btc_models_v2.py       # BTC model training script
├── train_mnq_models_v2.py       # MNQ model training script
├── feature_engineering.py       # Feature generation functions
├── models_btc_v2/               # BTC v2 models
│   ├── model_2h_0.5pct.joblib
│   ├── model_4h_0.5pct.joblib
│   ├── model_2h_0.5pct_SHORT.joblib
│   ├── model_4h_0.5pct_SHORT.joblib
│   └── feature_columns.json
├── models_mnq_v2/               # MNQ v2 models
│   ├── model_2h_0.5pct.joblib
│   ├── model_4h_0.5pct.joblib
│   ├── model_2h_0.5pct_SHORT.joblib
│   ├── model_4h_0.5pct_SHORT.joblib
│   └── feature_columns.json
├── data/
│   ├── BTC_features.parquet     # BTC historical features
│   └── QQQ_features.parquet     # QQQ historical features (for MNQ)
├── logs/
│   ├── btc_bot.log              # BTC bot logs
│   ├── mnq_bot.log              # MNQ bot logs
│   └── validation_alerts.log    # Data validation alerts
├── signal_logs/                 # Signal CSV logs
├── scripts/
│   └── daily_validation.py      # Daily validation script
├── results_btc_v2/              # BTC training results
└── results_mnq_v2/              # MNQ training results
```

---

# DATA VALIDATION (Added Dec 20, 2025)

## Data Validator Module (`data_validator.py`)

Integrated into both bots - **logs only, does not block signals**.

### Validation Checks (10 total)
| Check | Description | Severity |
|-------|-------------|----------|
| `data_freshness` | Data age < 10 minutes | warning |
| `data_completeness` | ≥200 bars available | critical |
| `feature_availability` | All 206 features present | critical |
| `nan_values` | <5% NaN in features | critical |
| `inf_values` | <1% Inf in features | critical |
| `price_validity` | Price > 0 and valid | critical |
| `price_continuity` | No >10% single-bar jumps | warning |
| `volume_validity` | Volume ≥ 0 | warning |
| `feature_ranges` | RSI 0-100, ADX 0-100, etc. | warning |
| `current_price` | Live price matches data | warning |

### Validation Frequency
- **MNQ Bot**: Every 5-min bar (when features refresh)
- **BTC Bot**: Every new bar

### Alert Logging
- Alerts logged to: `logs/validation_alerts.log`
- Console: Critical → ERROR, Warnings → WARNING
- Consecutive failures tracked (3+ triggers CRITICAL alert)

### Signal Generation Frequency
| Frequency | What Happens |
|-----------|--------------|
| **Every 15 sec** | Full 206 features → predict → signal generation |
| **Every 5 min** | Fresh data fetch + full feature recalc + validation |

---

# VALIDATION PLAN (Daily - End of Day)

## Daily Validation Checklist
For each trading day, compare:

1. **Signal Generation**
   - [ ] Count signals generated by live bot
   - [ ] Count signals from daily backtest on same data
   - [ ] Compare: Should match closely

2. **Trade Execution**
   - [ ] Count actual trades executed
   - [ ] Verify entry prices match signal prices
   - [ ] Verify exit reasons (SL, TP, trailing, timeout)

3. **Performance Metrics**
   - [ ] Live P&L vs backtest P&L
   - [ ] Live win rate vs backtest win rate
   - [ ] Slippage analysis (entry/exit price differences)

4. **System Health**
   - [ ] Connection drops/reconnections
   - [ ] Data gaps or missing bars
   - [ ] Feature generation errors

## Expected Tolerances
| Metric | Acceptable Variance |
|--------|---------------------|
| Signal count | ±10% |
| Win rate | ±5% |
| P&L per trade | ±0.1% (slippage) |
| Trade count | ±20% (due to timing) |

---

# CONTACT & ACCOUNTABILITY

- **Configuration date**: December 20, 2025
- **Last model training**: December 20, 2025
- **Validation start**: December 23, 2025 (Monday)
- **Review period**: 1 week (Dec 23-27, 2025)

---

*Document generated automatically. Keep updated with any configuration changes.*
