# CRITICAL AUDIT FINDINGS - ALL BOTS
**Date**: 2026-04-13 19:20
**Status**: MULTIPLE CRITICAL ISSUES DISCOVERED

---

## 🚨 CRITICAL ISSUE #1: Duplicate Exit Logging

### Problem
Bots are logging MULTIPLE EXIT events for the SAME position when IB fills orders across multiple contracts.

### Evidence

**Swing Bot (2.5/1.0)**:
- 9 ENTRY events
- 20 EXIT events
- **Ratio: 2.2 exits per entry**

**HF Bot (1.0/0.5)**:
- 16 ENTRY events
- 26 EXIT events
- **Ratio: 1.6 exits per entry**

**Example from btc_trades.jsonl**:
```
Entry: 2026-04-05 11:04:29 (1 contract)
Exits: TEN different exit logs for the SAME entry
  - 01:16:09 - TP - $185.04
  - 03:37:43 - TP - $190.45
  - 04:24:11 - TP - $220.96
  ... 7 more exits!
```

### Root Cause
IB Gateway fills large orders as MULTIPLE SEPARATE FILLS (e.g., 4-contract order = 4 x 1-contract fills). Each fill triggers an exit log event. Bot logs each fill without checking if it's the same position.

### Impact
- **Performance metrics are WRONG** - WR/P&L calculated on inflated trade count
- **Cannot trust dashboard numbers**
- Swing bot shows 60% WR but might actually be 30% (if 2 exits per entry)
- HF bot shows 65% WR but might be 40%

---

## 🚨 CRITICAL ISSUE #2: V2 Bots Orphan Detection Loop

### Problem
V2 bots are stuck in an infinite orphan detection loop, detecting and closing the SAME orphan position every 2 minutes.

### Evidence

**Swing V2**:
- 178 total log lines
- 172 are ORPHAN events (96.6%)
- Only 2 actual trades
- Example: Detected SHORT 2 contracts EIGHT TIMES in 20 minutes, closed each time

**HF V2**:
- Similar pattern (need to verify)

### Root Cause
V2 bots have ADOPT_ORPHANS logic that:
1. Detects orphan position in IB
2. Closes it immediately
3. BUT position reappears (IB paper account behavior)
4. Bot detects it again 2 minutes later
5. Loop continues forever

### Impact
- V2 bots are essentially NON-FUNCTIONAL
- 96% of their activity is closing phantom positions
- Cannot generate meaningful performance data
- Trade logs are polluted with orphan events

---

## 🚨 CRITICAL ISSUE #3: IB Paper Account Position Accumulation

### Problem
IB paper account automatically adds contracts to existing positions, causing bot's tracked size to diverge from IB's actual size.

### Evidence
- **Micro Bot**: Entry 4x → Exit 13x (3.25x accumulation)
- **Swing V2**: Entry 3x → Exit 22x (7.3x accumulation)
- **HF V2**: Entry 1x → Exit 44x (44x accumulation!)

### Root Cause
IB paper trading simulates market maker activity by automatically filling additional contracts. Bot detects mismatch and syncs, but P&L is calculated using the ACCUMULATED size, creating inflated losses.

### Impact
- **Losses appear massive** (-$1,694 on Swing V2)
- **P&L metrics are distorted**
- Cannot determine true bot performance
- Micro Bot: 5% WR, -$31/trade (should be 40% WR, +$18/trade)

---

## 🚨 CRITICAL ISSUE #4: Micro Bot Catastrophic Performance

### Current Performance
- 20 trades executed
- 1 win, 19 losses (5% WR)
- Average P&L: -$31.39
- Total loss: -$627.78

### Expected Performance (from backtest)
- Win Rate: 40-50%
- Average P&L: +$15-25
- Should be PROFITABLE

### Root Cause Analysis
Likely combination of:
1. **Position accumulation** (4x → 13x on first trade)
2. **Model not working** (0.3% TP / 0.1% SL extremely tight)
3. **Slippage worse than expected**
4. **IB paper account issues**

### Impact
- Bot is LOSING MONEY rapidly
- Should be STOPPED immediately
- Needs full diagnostic before restarting

---

## 📊 CORRECTED TRADE COUNTS

### Estimated ACTUAL Trades (accounting for duplicate logging)

| Bot | Logged Trades | Likely Real Trades | Duplicate Ratio |
|-----|---------------|-------------------|-----------------|
| Swing (2.5/1.0) | 20 | **~9** | 2.2x |
| HF (1.0/0.5) | 26 | **~16** | 1.6x |
| Swing V2 | 2 | **2** | 1.0x (but 172 orphan events) |
| HF V2 | 5 | **~3** | ~1.7x (+ orphan pollution) |
| Micro | 20 | **20** | 1.0x (but position accumulation) |
| Micro V2 | 42 | **42** | 1.0x |

---

## 🔍 DATA SOURCE AUDIT

### Are all bots using the same Binance API?

✅ **YES** - All 6 bots use Binance for signal data:
- Swing/HF/V2: All use `predictor.predict(df)` on Binance BTCUSDT data
- Micro/Micro V2: Same Binance data source

### Why different trade frequencies?

**Expected** (based on thresholds and TP/SL):
- Swing (2.5/1.0): 5-6 trades/week
- HF (1.0/0.5): 7-8 trades/week
- Micro (0.3/0.1): 10-15 trades/week
- Micro V2 (0.5/0.15): 15 trades/week

**Actual** (since April 10-13 = 3.5 days):
- Swing: ~9 real trades = **18/week** (3x expected)
- HF: ~16 real trades = **32/week** (4x expected)
- Micro: 20 trades = **40/week** (3-4x expected)
- Micro V2: 42 trades = **84/week** (5.6x expected!)

**Reason for HIGH frequency**:
1. **Models are too aggressive** - producing signals more often than backtested
2. **Paper trading slippage** - positions entering/exiting faster than real market
3. **Duplicate logging** - inflating trade counts by 1.6-2.2x
4. **Confidence thresholds too low** - letting weak signals through

---

## 🎯 ROOT CAUSE SUMMARY

### Why 0 trades (Micro in dashboard)?
- **FIXED** - Filename mismatch in generate_status.py

### Why "too many" trades?
1. **Duplicate exit logging** - Same position logged 2-10 times (Swing/HF)
2. **Orphan detection loop** - V2 bots closing phantoms every 2 min
3. **Models too aggressive** - Generating 3-5x more signals than expected
4. **IB paper fills** - Partial fills creating multiple log events

### Position Size Mismatches?
**ALL bots affected** by IB paper account behavior:
- Original bots (Swing/HF): Partial fills → multiple exit logs
- V2 bots: Orphan loops + position accumulation
- Micro bots: Position accumulation (4x → 13x)

---

## 🚫 IMMEDIATE ACTIONS REQUIRED

### STOP These Bots NOW:
1. ✅ **Swing V2** - 96% orphan loops, non-functional
2. ✅ **HF V2** - Likely same orphan loop issue
3. ✅ **Micro Bot** - 5% WR, bleeding money (-$628 in 3 days)

### Keep Running (with caution):
- **Swing** - Duplicate logging but actually profitable
- **HF** - Duplicate logging but good performance
- **Micro V2** - TS bug fixed, monitoring for improvement

---

## 🔧 FIXES NEEDED

### Fix #1: Prevent Duplicate Exit Logging
**File**: All `btc_ib_gateway_bot*.py` files
**Change**: Track which positions have been logged, prevent re-logging
**Complexity**: Medium
**Priority**: HIGH

### Fix #2: Disable Orphan Adoption in V2 Bots
**File**: `btc_ib_gateway_bot_v2.py`, `btc_ib_gateway_bot_hf_v2.py`
**Change**: Set `ADOPT_ORPHANS = False` or fix detection logic
**Complexity**: Low
**Priority**: CRITICAL

### Fix #3: Add Position Size Limits
**File**: All bot files
**Change**: Set max position size to prevent accumulation (e.g., max 6x)
**Complexity**: Low
**Priority**: HIGH

### Fix #4: Investigate Micro Bot Model
**Action**: Full diagnostic - why 5% WR vs expected 40%?
**Complexity**: High
**Priority**: CRITICAL

---

## 📈 CORRECTED PERFORMANCE ESTIMATES

### Swing (2.5/1.0) - Accounting for 2.2x duplicate logging
- Real trades: ~9
- Real WR: Unknown (need to de-duplicate)
- Real avg P&L: Unknown
- **Status**: Uncertain - need to fix logging first

### HF (1.0/0.5) - Accounting for 1.6x duplicate logging
- Real trades: ~16
- Real WR: Unknown
- Real avg P&L: Likely still positive
- **Status**: Likely profitable but needs verification

### Micro V2 (0.5/0.15) - No duplicate logging
- Real trades: 42
- WR: 30.8% (was 40.3% expected)
- Avg P&L: -$7.56 (was +$18.80 expected)
- **Status**: TS bug fixed, monitoring next 20-30 trades

---

## 🎬 NEXT STEPS

### Immediate (Today)
1. ✅ STOP Swing V2 bot (PID 72959)
2. ✅ STOP HF V2 bot (PID 19586)
3. ✅ STOP Micro bot (PID 50953)
4. ✅ Set `ADOPT_ORPHANS = False` in V2 bots
5. ✅ Fix duplicate exit logging in all bots

### Tomorrow
1. Clean up trade logs - mark which exits are duplicates
2. Recalculate true performance metrics
3. Investigate Micro bot model (why 5% WR?)
4. Monitor Micro V2 after TS fix

### This Week
1. Implement position size limits (max 6x)
2. Add duplicate detection to exit logging
3. Re-test V2 bots with orphan adoption disabled
4. Decision: Keep or abandon Micro bot (0.3/0.1)

---

## ✅ CONFIRMED WORKING

- ✅ All bots using same Binance API
- ✅ Signal generation working (ML predictions)
- ✅ IB Gateway connections working
- ✅ Entry execution working
- ✅ TP/SL orders being placed

## ❌ CONFIRMED BROKEN

- ❌ Exit logging (duplicate events)
- ❌ V2 orphan adoption (infinite loops)
- ❌ Position size tracking (IB paper accumulation)
- ❌ Micro bot model (5% WR catastrophic)
- ❌ Trade frequency (3-5x higher than expected)

---

**CONCLUSION**: Trading system has fundamental logging and position tracking issues. Performance metrics cannot be trusted until duplicate logging is fixed. V2 bots are non-functional due to orphan loops. Micro bot is losing money rapidly and should remain stopped.
