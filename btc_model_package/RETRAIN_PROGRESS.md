# Full 22-Feature Model Retrain - Progress Report

## Status: IN PROGRESS

### Task 1: Regenerate V3 Dataset ⏳ RUNNING
**Status:** Processing (~2-3 minutes)
**What's happening:**
- Resampling 2.1M 1-min candles → 420K 5-min candles
- Creating symmetric labels for 2h/4h/6h horizons
- Calculating ALL 22 features (added 5 missing ones)

**Missing features now included:**
1. `rsi_28` - 28-period RSI indicator
2. `day_of_week` - Day of week (0=Monday, 6=Sunday)
3. `volume_hour_median` - Median volume for this hour historically
4. `dist_from_prev_4h` - % distance from 4h ago price
5. `bb_position_50` - Bollinger Band position (50-period)

**Output:** `btc_5m_v3_features.parquet` with all 22 features

---

### Task 2: Retrain Models ⏸️ WAITING
**Will start after Task 1 completes**
- Train 2h model on all 22 features
- Train 4h model on all 22 features
- Train 6h model on all 22 features
- Save updated models and scalers

**Expected time:** ~2 minutes

---

### Task 3: Walk-Forward Validation ⏸️ WAITING
**Will start after Task 2 completes**
- Run 5-period walk-forward test
- Calculate P&L for each period
- Verify math: actual P&L vs theoretical
- Compare to 17-feature models

**Expected time:** ~3 minutes

---

### Task 4: Live Prediction Test ⏸️ WAITING
**Will start after Task 3 completes**
- Fetch live Binance data
- Calculate all 22 features
- Generate prediction with new models
- Show confidence and TP/SL levels

**Expected time:** ~10 seconds

---

## Expected Results

### If 22 Features Perform Better:
- Higher accuracy (~54-56%)
- Better P&L per trade (>0.31%)
- More stable predictions

### If Similar to 17 Features:
- Performance ~54% WR, +0.31% P&L
- Additional features don't hurt, might help in different market regimes
- Still worth having complete feature set

---

## Timeline

```
[████████--] Task 1: Dataset (2 min) - RUNNING
[----------] Task 2: Training (2 min)
[----------] Task 3: Validation (3 min)
[----------] Task 4: Testing (10 sec)
═══════════════════════════════════
Total: ~7-8 minutes from start
```

---

## Comparison: 17 vs 22 Features

### Current (17 features):
- Working with real Binance data ✓
- 54.24% WR, +0.31% avg P&L ✓
- Missing: RSI, day patterns, BB position

### After retrain (22 features):
- Complete feature set ✓
- Better capture market patterns? (TBD)
- More robust across different conditions? (TBD)

---

## For Integration

Once complete, you'll have:
1. **Updated predictor.py** - Calculates all 22 features
2. **New models** - Trained on complete feature set
3. **Validation report** - Walk-forward results with P&L
4. **Live test** - Real prediction on current BTC price

The package structure stays the same - just drop-in replacement of model files.

---

Last updated: Good Friday, 2026-04-03 20:02 PM
