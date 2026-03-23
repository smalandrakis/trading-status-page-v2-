# Strategy Research & Investigation History
**Last Updated: January 7, 2026**

## Overview
This document tracks all strategy investigations, backtests, and findings for the trading bots.

---

## Strategy Performance Summary (7-Day Live Trading)

### Best Performers
| Strategy | Trades | WR% | P&L | Notes |
|----------|--------|-----|-----|-------|
| indicator_trend | 1 | 100% | +$148.47 | ROC + MACD trend following |
| 2h_0.5pct LONG | 7 | 43% | +$147.96 | ML model |
| legacy_mbt_1 | 1 | 100% | +$92.96 | Legacy model |
| 4h_0.5pct LONG | 20 | 25% | +$19.40 | ML model |

### Worst Performers
| Strategy | Trades | WR% | P&L | Notes |
|----------|--------|-----|-----|-------|
| 2h_0.5pct_SHORT | 61 | 31% | -$272.98 | Shorting into uptrend |
| 4h_0.5pct_SHORT | 40 | 20% | -$151.62 | Shorting into uptrend |
| indicator_long | 20 | 10% | -$92.88 | SL bug + poor strategy |
| indicator_meanrev | 4 | 25% | -$51.07 | Mean reversion failing |

---

## Strategy Investigations

### 1. Trend Filter (January 2026)
**Goal**: Block SHORT trades during uptrends to prevent losses

**Implementation**:
- SMA 5 > SMA 20 = Uptrend → Block SHORT
- SMA 5 < SMA 20 = Downtrend → Block LONG

**Results**:
- Successfully blocks losing SHORT trades during BTC/MNQ uptrends
- Added to MNQ bot (`TREND_FILTER_ENABLED = True`)
- NOT yet added to BTC bot

**Config**:
```python
TREND_FILTER_ENABLED = True
TREND_FILTER_SMA_SHORT = 5
TREND_FILTER_SMA_LONG = 20
```

---

### 2. Trend Following Strategy (January 2026)
**Goal**: Ride strong trends instead of fighting them

**Implementation**:
- Entry: ROC > 1.5% (strong momentum)
- SL: 0.50%, TP: 1.50%
- Trailing stop: 0.30%

**Backtest Results**:
- Captures large moves during strong trends
- Better R:R ratio than mean reversion

**Config**:
```python
TREND_FOLLOW_ENABLED = True
TREND_FOLLOW_SL_PCT = 0.50
TREND_FOLLOW_TP_PCT = 1.50
TREND_FOLLOW_TRAILING_PCT = 0.30
TREND_FOLLOW_MIN_ROC = 1.5
```

---

### 3. Indicator LONG Strategy (BB %B < 0.5)
**Goal**: Mean reversion - buy when price below BB midline

**Backtest (7-day fresh data)**:
- 534 trades/week, 50.4% WR, ~$5.6k/week P&L

**Live Trading Results**:
- 20 trades, 10% WR, -$92.88
- **BUG FOUND**: Stop loss was 0.10% instead of configured 0.30%

**Bug Analysis**:
- 01-05 trades: Used wrong 0.10% SL (STOP_LOSS_PCT)
- 01-06 trades: Used correct 0.30% SL (INDICATOR_LONG_SL_PCT)
- Root cause: Code path falling through to general SL

**Status**: DISABLED - Bug fixed but strategy still unprofitable

---

### 4. Indicator Mean Reversion (BB %B < 0.5)
**Goal**: Buy when oversold (same logic as MNQ)

**Backtest**:
- 342 trades, 44.2% WR, +$184/week

**Live Trading**:
- 4 trades, 25% WR, -$51.07

**Status**: DISABLED - Replaced by ROC+MACD trend strategy

---

### 5. Indicator Trend (ROC + MACD)
**Goal**: Trend following using ROC and MACD histogram

**Live Trading**:
- 1 trade, 100% WR, +$148.47

**Status**: ENABLED - Best performing indicator strategy

---

### 6. ML Models Analysis

#### LONG Models
- **2h_0.5pct**: 7 trades, 43% WR, +$147.96 ✓
- **4h_0.5pct**: 20 trades, 25% WR, +$19.40 ✓

#### SHORT Models
- **2h_0.5pct_SHORT**: 61 trades, 31% WR, -$272.98 ✗
- **4h_0.5pct_SHORT**: 40 trades, 20% WR, -$151.62 ✗

**Finding**: SHORT models lose money during uptrends. Need trend filter.

---

## Key Learnings

### 1. Commission Impact
- MNQ: $1.24/trade (round trip)
- BTC: $2.02/trade (round trip)
- High trade frequency = commissions eat profits

### 2. Trend Matters
- SHORT strategies fail during uptrends
- LONG strategies fail during downtrends
- Trend filter is essential

### 3. Stop Loss Configuration
- Tighter SL (0.10%) = More SL hits, lower WR
- Wider SL (0.30%) = Fewer SL hits, but larger losses when hit
- Optimal depends on strategy and market conditions

### 4. Backtest vs Live Discrepancy
- Backtests often show better results than live trading
- Slippage, timing, and execution affect real results
- Always validate with paper trading first

---

## Recommended Next Steps

1. **Add trend filter to BTC bot** - Currently only MNQ has it
2. **Reduce SHORT model exposure** - They're losing money in uptrends
3. **Focus on trend-following** - indicator_trend is the best performer
4. **Monitor commission impact** - High frequency = high costs

---

## Configuration Files Reference

### MNQ Bot Configs (ensemble_bot.py)
- Lines 112-133: Trend filter and indicator configs
- Lines 1385-1520: Trend detection and filtering logic

### BTC Bot Configs (btc_ensemble_bot.py)
- Lines 95-112: Stop loss and indicator configs
- Lines 1335-1384: Indicator signal generation

### Monitor Daemon (monitor_daemon.py)
- Lines 720-820: Signal mismatch detection and auto-fix
