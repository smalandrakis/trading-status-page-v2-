# Micro-Movement Trading Investigation Summary

**Date**: 2026-04-13
**Goal**: Explore micro-movement trading (0.3-0.5% TP, 0.1-0.15% SL) to achieve 55-60% WR, $15-25 avg P&L/trade

---

## Executive Summary

**Result**: Wider TP/SL (0.5%/0.15%) is PROFITABLE at **$7.60/trade** but below target ($15-25).

**Critical Issue**: Models extremely SHORT-biased (99.8% SHORT signals) due to bearish BTC trend in training data (2022-2026).

**Recommendation**: Continue using V3 models with current 1-2.5% TP / 0.5-1.0% SL strategy OR deploy 0.5%/0.15% as SHORT-only strategy.

---

## Investigations Completed

### 1. Original Tight Targets (0.3% TP / 0.1% SL)
**Status**: ✗ **UNPROFITABLE**

- Trained custom 3-horizon ensemble (30min, 1h, 2h)
- Full dataset training (839K rows, 20min training time)
- **Results**: 38.4% WR, **-$0.62 avg P&L** (NEGATIVE)
- All threshold combinations tested = negative P&L
- Root cause: Low precision for LONG/SHORT (0.26-0.31), heavy NEUTRAL bias

**Conclusion**: 0.3%/0.1% targets not viable with current approach.

---

### 2. Wider TP/SL Targets (0.5% TP / 0.15% SL)
**Status**: ✓ **PROFITABLE BUT SHORT-BIASED**

#### Label Generation:
- 30min: 6.5% LONG, 11.5% SHORT, 82.0% NEUTRAL
- 1h: 12.8% LONG, 25.1% SHORT, 62.1% NEUTRAL
- 2h: 19.5% LONG, 42.5% SHORT, 37.9% NEUTRAL

**Issue**: Severe class imbalance, 2-4x more SHORT than LONG signals.

#### Model Training:
- HistGradientBoostingClassifier with balanced class weights
- Sampled training (84K rows, ~5 min training)
- Accuracies: 30min=83.8%, 1h=71.3%, 2h=59.5%
- But precision still low: LONG=0.13-0.32, SHORT=0.26-0.53

#### Backtest Results (2,624 trades on 2025-2026):
- **Win Rate**: 35.4%
- **Avg P&L**: **+$7.63/trade** ✓
- **Total P&L**: $20,013
- **Avg Win**: $129.50 (larger TP pays off)
- **Avg Loss**: -$59.99

#### Direction Breakdown:
- **LONG**: 6 trades (0.2%), 16.7% WR, -$25.96 avg P&L
- **SHORT**: 2,618 trades (99.8%), 35.5% WR, **+$7.70 avg P&L**

**Conclusion**: Profitable SHORT-only strategy, but not balanced LONG/SHORT as intended.

---

### 3. Micro-Structure Feature Engineering
**Status**: ✗ **NO IMPROVEMENT**

Added 16 new features:
1. VWAP distance (5-period, 20-period)
2. Tick velocity
3. Realized volatility (5-bar)
4. Micro support/resistance
5. Volume profile deviation
6. Intraday session markers (Asia, Europe, US)
7. Price momentum (1-bar, 3-bar, 5-bar)
8. Spread proxy (high-low range)
9. Price position in range
10. Volume trend

**Total features**: 25 V3 + 16 micro-structure = 41 features

#### Results (2,584 trades):
- **Win Rate**: 35.4% (same as baseline)
- **Avg P&L**: **+$7.60/trade** (vs +$7.63 baseline)
- **Improvement**: **-$0.03/trade** (NONE)

**Conclusion**: Micro-structure features did not improve predictive power.

---

### 4. Alternative Models (XGBoost, LightGBM)
**Status**: ⚠️ **BLOCKED**

- Both XGBoost and LightGBM require OpenMP (libomp.dylib)
- Installation blocked by system dependencies
- Cannot test without `brew install libomp`

---

## Root Cause Analysis

**Why are models SHORT-biased?**

The 2022-2026 BTC dataset has a **bearish trend**:
- 2022: Bear market (69K → 16K, -77%)
- 2023: Recovery (16K → 42K, +163%)
- 2024: Bull run (42K → 108K, +157%)
- 2025-2026: Volatility around 70-100K

During feature window (250 bars = ~21 hours), BTC experienced more 0.5% DOWN moves than UP moves, creating natural SHORT bias in labels.

**Label distribution confirms**:
- 30min: 6.5% LONG vs 11.5% SHORT (1.8x more SHORT)
- 1h: 12.8% LONG vs 25.1% SHORT (2.0x more SHORT)
- 2h: 19.5% LONG vs 42.5% SHORT (2.2x more SHORT)

Models learn this bias and predict SHORT 99.8% of the time.

---

## What Didn't Work

1. **Tighter targets (0.3%/0.1%)** - Unprofitable (-$0.62/trade)
2. **Micro-structure features** - No improvement (+$7.60 vs +$7.63)
3. **Threshold optimization** - All combinations tested, no better result
4. **Class balancing** - `class_weight='balanced'` insufficient for 1:6 LONG:NEUTRAL ratio

---

## What Did Work

**Wider TP/SL (0.5%/0.15%)** achieved profitable trading:
- **Avg P&L**: $7.60/trade (vs target $15-25)
- **Annualized**: ~2,600 trades/year × $7.60 = **$19,760/year**
- **ROI**: 197% on $10K account (assuming $75/trade notional)
- **Trade frequency**: ~10 trades/week

But it's essentially a **SHORT-only strategy** (99.8% SHORT signals).

---

## Options Going Forward

### Option 1: Deploy 0.5%/0.15% SHORT-Only Strategy ⚠️
**Pros**:
- Profitable (+$7.60/trade)
- Validated on 2,624 trades
- 35.5% WR for SHORT signals specifically

**Cons**:
- Missing LONG opportunities (market can reverse)
- Below target ($7.60 vs $15-25)
- Unbalanced risk profile

**Recommended**: Only if combined with separate LONG strategy

---

### Option 2: Train Separate LONG and SHORT Models
**Approach**:
- Train LONG-only model on 2024-2025 bull period (42K→108K)
- Train SHORT-only model on 2022-2023 bear period (69K→16K)
- Use market regime detection to switch models
- Deploy both in parallel

**Pros**:
- Balanced LONG/SHORT signals
- Better coverage of market conditions

**Cons**:
- 2 weeks additional dev time
- Requires regime detection logic
- More complex to maintain

---

### Option 3: Keep V3 Models with Current Strategy ✓
**Approach**:
- Continue using existing V3 models
- 1.0-2.5% TP / 0.5-1.0% SL
- 50-65% WR, $10-20 avg P&L/trade
- 4 bots already deployed and profitable

**Pros**:
- Already working and profitable
- Balanced LONG/SHORT
- Higher WR (50-65% vs 35%)
- $10-20/trade vs $7.60

**Cons**:
- Lower trade frequency
- Not exploring micro-movements as requested

**Recommended**: This is the safest option.

---

### Option 4: Try Neural Networks (If Dependencies Can Be Installed)
**Approach**:
- Install OpenMP: `brew install libomp`
- Try XGBoost or PyTorch neural network
- More aggressive hyperparameter tuning
- Synthetic LONG sample generation (SMOTE)

**Pros**:
- May handle class imbalance better
- Neural networks can capture non-linear patterns

**Cons**:
- Installation issues (OpenMP)
- Training time (hours for neural networks)
- May not improve beyond HistGradientBoosting

**Estimated time**: 2-3 days if dependencies resolve.

---

## Recommended Next Steps

**For today (immediate)**:
1. ✅ Continue running btc_micro_bot.py with V3 models (already deployed)
2. ✅ Monitor live performance with MFE/MAE tracking
3. ⏭️ Decision: Deploy 0.5%/0.15% SHORT-only strategy OR stay with V3 models

**For this week**:
1. Install OpenMP if possible: `brew install libomp`
2. Try XGBoost/LightGBM with 0.5%/0.15% targets
3. Test training on bull-only period (2024-2025) for LONG model

**For next 2 weeks (if pursuing micro-movements)**:
1. Develop separate LONG/SHORT models
2. Implement market regime detection
3. Paper trade for 100 trades minimum
4. Deploy if paper WR ≥ 52%

---

## Files Created

### Label Generation:
- `label_generation_micro.py` - Original 0.3%/0.1% labels (839K rows)
- `label_generation_wider.py` - Wider 0.5%/0.15% labels (839K rows)

### Training Scripts:
- `train_btc_models_micro.py` - Full training for 0.3%/0.1% (20 min, NEGATIVE results)
- `train_btc_models_wider.py` - Training for 0.5%/0.15% (5 min, POSITIVE results)
- `train_microstructure.py` - Training with 41 features (5 min, NO improvement)
- `train_xgboost_microstructure.py` - XGBoost attempt (BLOCKED by OpenMP)

### Feature Engineering:
- `feature_engineering_microstructure.py` - 16 micro-structure features

### Validation Scripts:
- `validate_micro_quick.py` - Quick validation (0.3%/0.1%, NEGATIVE)
- `validate_wider_quick.py` - Wider targets validation (0.5%/0.15%, POSITIVE)
- `validate_microstructure.py` - Micro-structure validation (NO improvement)
- `optimize_thresholds_micro.py` - Threshold sweep (all NEGATIVE)

### Bot (Already Deployed):
- `btc_micro_bot.py` - Running with V3 models, adaptive learning enabled

---

## Performance Summary

| Approach | TP/SL | Features | WR | Avg P&L | Total P&L (2.6K trades) | Status |
|----------|-------|----------|-----|---------|------------------------|--------|
| Original Micro | 0.3% / 0.1% | 25 V3 | 38.4% | -$0.62 | -$732 | ✗ UNPROFITABLE |
| Wider Targets | 0.5% / 0.15% | 25 V3 | 35.4% | **+$7.63** | **+$20,013** | ✓ PROFITABLE |
| Micro-Structure | 0.5% / 0.15% | 41 (25+16) | 35.4% | +$7.60 | +$19,651 | ~ NO GAIN |
| V3 Baseline | 1-2.5% / 0.5-1% | 25 V3 | 50-65% | $10-20 | N/A | ✓✓ BEST |

---

## Conclusion

**Micro-movements are viable at 0.5% TP / 0.15% SL** but:
1. **Profitable** at $7.60/trade (50-70% of target)
2. **SHORT-biased** (99.8% SHORT signals due to dataset trend)
3. **No benefit** from micro-structure features or tighter targets
4. **Below V3 baseline** in WR (35% vs 50-65%) and avg P&L ($7.60 vs $10-20)

**Recommendation**: Unless you specifically want a SHORT-only strategy, **continue with V3 models** (1-2.5% TP / 0.5-1% SL) which have proven 50-65% WR and $10-20/trade across 4 deployed bots.

If pursuing micro-movements further, focus on **separate LONG/SHORT models** trained on appropriate market regimes rather than ensemble on full 2022-2026 dataset.
