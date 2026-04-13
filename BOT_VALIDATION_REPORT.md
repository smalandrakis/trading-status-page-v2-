# Bot Validation Report - April 9, 2026

## ✅ TIMESTAMP BUG FIXED

**Issue:** V2 bots were not setting timestamp as DataFrame index, causing incorrect feature calculations.

**Location:**
- `btc_ib_gateway_bot_v2.py` line 235
- `btc_ib_gateway_bot_hf_v2.py` line 232

**Fix Applied:**
```python
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)
df = df[['open', 'high', 'low', 'close', 'volume']]
return df
```

**Status:** ✅ FIXED and V2 bots RESTARTED at 19:52

---

## VALIDATION RESULTS

### TEST 1: Data Fetching Consistency ✅ PASS
- V1 and V2 now fetch data in IDENTICAL format
- Both use DatetimeIndex
- Both return 250 bars with same columns

### TEST 2: Predictor Consistency ✅ PASS
- V1 Signal: NEUTRAL @ 37.2%
- V2 Signal: NEUTRAL @ 37.2%
- Probabilities match exactly: 2h=0.1899, 4h=0.4035, 6h=0.5212
- **Conclusion: V1 and V2 produce IDENTICAL signals**

### TEST 3: Bot Configuration Match ✅ PASS
All 4 bots have correct configurations:
- **V1 Swing:** TP=2.5%, SL=1.0%, Client ID=10, set_index ✓
- **V2 Swing:** TP=2.5%, SL=1.0%, Client ID=12, set_index ✓
- **V1 HF:** TP=1.0%, SL=0.5%, Client ID=11, set_index ✓
- **V2 HF:** TP=1.0%, SL=0.5%, Client ID=13, set_index ✓

All use same thresholds: LONG=0.65, SHORT=0.25

### TEST 4: Log File Analysis
**V1 Swing:**
- 19 total trades, 63.2% WR
- 7 signals generated, 7 positions entered
- Last signal: SHORT @ 75.4% (April 9, 01:02)
- Active since April 4

**V2 Swing:**
- 0 trades (clean slate after orphan adoption fix)
- Restarted at 19:52 with bug fix

**V1 HF:**
- 21 total trades, 66.7% WR
- 10 signals generated, 10 positions entered
- Last signal: SHORT @ 75.4% (April 9, 01:02)

**V2 HF:**
- Restarted at 19:52 with bug fix

### TEST 5: Running Processes ✅
All 4 bots running:
- V1 Swing (PID: running since 10:41 PM Apr 8)
- V1 HF (PID: running since 10:41 PM Apr 8)
- V2 Swing (PID: 49728, restarted 19:52)
- V2 HF (PID: 49761, restarted 19:52)

---

## KEY FINDINGS

### 1. V1 vs V2 Difference Explained
**Why V2 didn't generate signals:**
- V2 was NOT broken at the predictor level
- V2's orphan adoption triggered at 01:02:49, immediately after V1 entered SHORT
- V2 detected V1's position as "orphan" and closed it
- V2 never ran signal generation because it was busy closing orphan

**Root cause:**
- V1 and V2 share same IB account (DUO071685)
- Different client IDs (10, 11, 12, 13) but same account
- V2 sees V1's positions and treats them as orphans

### 2. Timestamp Bug Impact
**Before fix:** V2 would have generated DIFFERENT signals than V1 due to:
- DataFrame without timestamp index
- Feature calculations relying on datetime index
- Time-based features (hour, day_of_week, etc.) would fail or be wrong

**After fix:** V2 now produces IDENTICAL signals to V1 (verified in TEST 2)

### 3. Model Behavior (from earlier analysis)
**Model Type:** MEAN-REVERTING with trend filters
- Primary: Buy at support, sell at resistance
- Secondary: Only takes trades WITH positive momentum
- **Problem:** SHORTs failing in strong bull market (0% WR recently)

**Key features:**
- Mean-rev importance: 0.188 (dist_to_resistance, dist_to_support, etc.)
- Trend importance: 0.071 (trend_2h_pct, adx_proxy)
- But trend features have POSITIVE weights = buys strength, not weakness

### 4. Recent SHORT Performance
Last SHORT signal (April 9, 01:02):
- Entry: $70,773.70
- Stop Loss: $71,481.44 (+1.0%)
- Result: Hit stop loss at $71,506 after 4h 53min
- Loss: -$434.96 (-1.02%)
- **Outcome:** Model was WRONG, price went UP not DOWN

---

## RECOMMENDATIONS

### Immediate Actions ✅ COMPLETED
1. ✅ Fixed V2 timestamp bug
2. ✅ Restarted V2 bots with fix
3. ✅ Verified V1 and V2 produce identical signals

### Monitoring Plan (Next 1-2 Weeks)
1. **Monitor V2 signal generation:**
   - Check if V2 generates signals now that bug is fixed
   - Verify V2 signals match V1 signals
   - V2 currently at 0 trades, need to accumulate data

2. **Watch for orphan adoption conflicts:**
   - V2 will still detect V1's positions as orphans
   - This is expected with shared IB account
   - If V2 keeps closing V1's positions, disable `ADOPT_ORPHANS` in V2

3. **Track SHORT performance:**
   - All recent SHORTs losing (6/6 losses in V1 Swing)
   - Consider adding market regime filter
   - Or reduce SHORT threshold to 0.20 (more selective)

### Long-term Decision
After 1-2 weeks, choose ONE of:
- **Option A:** Keep V1 only (proven track record)
- **Option B:** Keep V2 only (bot-managed TP/SL, now fixed)
- **Option C:** Use different IB accounts for V1 and V2
- **Option D:** Disable one pair (either Swing or HF)

---

## CURRENT STATUS

**All Systems Operational ✅**
- 4 bots running
- V1 bots trading normally
- V2 bots restarted with timestamp fix
- Binance connection: OK
- IB Gateway connection: OK

**Next Checkpoint:**
- Check V2 logs tomorrow to see if signals generated
- Compare V2 vs V1 signal timing and values
- Monitor for any orphan adoption issues

---

**Generated:** April 9, 2026 19:52
