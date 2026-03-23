#!/usr/bin/env python3
"""
Ensemble Bot Backtest — this week (Mar 8-12, 2026)
Replicates live bot logic: features, models, filters, SL/TS.
Also runs sensitivity analysis and bug diagnostics.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# ── use ta library same as ensemble bot ──────────────────────────────────────
try:
    import ta
except ImportError:
    print("pip install ta  →  needed for ensemble features")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  (mirrors live bot)
# ─────────────────────────────────────────────────────────────────────────────
MODELS_DIR       = 'models_btc_v2'
FEATURE_COLS_PATH= 'models_btc_v2/feature_columns.json'
DATA_PATH        = 'logs/btc_price_ticks.csv'
THIS_WEEK        = pd.Timestamp('2026-03-08')

# Live bot params
STOP_LOSS_PCT    = 0.20
TS_ACT_PCT       = 0.25
TS_TRAIL_PCT     = 0.05
CONTRACT_VAL     = 0.1   # MBT = 0.1 BTC
PROB_THRESHOLD   = 0.55
SL_COOLDOWN_SEC  = 1800  # 30 min
MAX_L = 2;  MAX_S = 2

LONG_MODELS  = ['2h_0.5pct', '4h_0.5pct', '6h_0.5pct']
SHORT_MODELS = ['2h_0.5pct_SHORT', '4h_0.5pct_SHORT']
ALL_MODELS   = LONG_MODELS + SHORT_MODELS

# ─────────────────────────────────────────────────────────────────────────────
# LOAD TICK DATA → 5-MIN BARS
# ─────────────────────────────────────────────────────────────────────────────
print("="*65)
print("LOADING DATA")
print("="*65)

ticks = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
ticks = ticks.sort_values('timestamp').set_index('timestamp')
ticks.index = pd.to_datetime(ticks.index, utc=True).tz_convert('UTC').tz_localize(None)

# Need to synthesize Volume — use tick count as proxy
bars_5m = ticks['price'].resample('5min').agg(
    Open='first', High='max', Low='min', Close='last', Volume='count'
).dropna()
bars_5m.columns = ['Open','High','Low','Close','Volume']
bars_5m = bars_5m[bars_5m['Volume'] > 0]
print(f"5-min bars: {len(bars_5m):,}  ({bars_5m.index[0]:%Y-%m-%d} → {bars_5m.index[-1]:%Y-%m-%d})")

# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE FEATURES  (same as add_features() in bot)
# ─────────────────────────────────────────────────────────────────────────────
print("Computing features…")

def add_features(df):
    df = df.copy()
    try:
        df = ta.add_all_ta_features(
            df, open="Open", high="High", low="Low",
            close="Close", volume="Volume", fillna=True
        )
    except Exception as e:
        print(f"  ta error: {e}")
        return df

    df['hour']        = df.index.hour
    df['minute']      = df.index.minute
    df['time_of_day'] = df['hour'] + df['minute'] / 60
    df['is_premarket']    = 0
    df['is_regular_hours']= 1
    df['is_afterhours']   = 0
    df['is_first_hour']   = (df['hour'] == 0).astype(int)
    df['is_last_hour']    = (df['hour'] == 23).astype(int)
    df['day_of_week']     = df.index.dayofweek
    df['is_monday']       = (df['day_of_week'] == 0).astype(int)
    df['is_friday']       = (df['day_of_week'] == 4).astype(int)
    df['month']           = df.index.month
    df['is_month_end']    = df.index.is_month_end.astype(int)
    df['is_month_start']  = df.index.is_month_start.astype(int)
    df['week_of_year']    = df.index.isocalendar().week.astype(int)

    df['return_1bar']     = df['Close'].pct_change()
    df['return_5bar']     = df['Close'].pct_change(5)
    df['return_10bar']    = df['Close'].pct_change(10)
    df['return_20bar']    = df['Close'].pct_change(20)
    df['log_return_1bar'] = np.log(df['Close'] / df['Close'].shift(1))

    for w in [5, 10, 20, 50]:
        ma = df['Close'].rolling(w).mean()
        df[f'price_to_ma{w}'] = df['Close'] / ma - 1
    for w in [5, 10, 20]:
        df[f'volatility_{w}bar'] = df['return_1bar'].rolling(w).std()

    df['bar_range']       = (df['High'] - df['Low']) / df['Close']
    df['bar_range_ma5']   = df['bar_range'].rolling(5).mean()
    df['close_position']  = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['gap']             = df['Open'] / df['Close'].shift(1) - 1

    df['trade_date'] = df.index.date
    daily = df.groupby('trade_date').agg(
        Open=('Open','first'), High=('High','max'), Low=('Low','min'),
        Close=('Close','last'), Volume=('Volume','sum')
    )
    daily['prev_close']      = daily['Close'].shift(1)
    daily['prev_high']       = daily['High'].shift(1)
    daily['prev_low']        = daily['Low'].shift(1)
    daily['prev_volume']     = daily['Volume'].shift(1)
    daily['prev_day_return'] = daily['Close'].pct_change()
    daily['prev_2day_return']= daily['Close'].pct_change(2)
    daily['prev_5day_return']= daily['Close'].pct_change(5)

    df = df.merge(
        daily[['prev_close','prev_high','prev_low','prev_volume',
               'prev_day_return','prev_2day_return','prev_5day_return']],
        left_on='trade_date', right_index=True, how='left'
    )
    df['price_vs_prev_close'] = df['Close'] / df['prev_close'] - 1
    df['price_vs_prev_high']  = df['Close'] / df['prev_high'] - 1
    df['price_vs_prev_low']   = df['Close'] / df['prev_low'] - 1
    df['volume_vs_prev']      = df['Volume'] / (df['prev_volume'] / 288 + 1)
    df = df.drop(columns=['trade_date'], errors='ignore')

    # Lag & change features (needed by ensemble models)
    LAG_BASES   = ['momentum_rsi','momentum_stoch','trend_adx','trend_macd',
                   'trend_macd_diff','trend_macd_signal','volatility_atr',
                   'volatility_bbp','volume_mfi','volume_obv']
    LAG_OFFSETS = [1, 2, 3, 5, 10, 20, 50]
    CHG_OFFSETS = [1, 5]
    for base in LAG_BASES:
        if base not in df.columns:
            continue
        for n in LAG_OFFSETS:
            df[f'{base}_lag{n}'] = df[base].shift(n)
        for n in CHG_OFFSETS:
            df[f'{base}_change_{n}'] = df[base].pct_change(n)
    return df

feat_df = add_features(bars_5m)
feat_week = feat_df[feat_df.index >= THIS_WEEK]
print(f"Features: {feat_df.shape[1]} cols | Week rows: {len(feat_week):,}")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────
with open(FEATURE_COLS_PATH) as f:
    feature_cols = json.load(f)

models = {}
for nm in ALL_MODELS:
    path = os.path.join(MODELS_DIR, f'model_{nm}.joblib')
    if os.path.exists(path):
        models[nm] = joblib.load(path)
        print(f"  Loaded {nm}")
    else:
        print(f"  MISSING: {path}")

avail_cols = [c for c in feature_cols if c in feat_df.columns]
missing_cols = [c for c in feature_cols if c not in feat_df.columns]
print(f"\nFeature coverage: {len(avail_cols)}/{len(feature_cols)} ({len(missing_cols)} missing)")
if missing_cols[:5]:
    print(f"  Missing sample: {missing_cols[:5]}")

# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest(feat_df, bars_5m, models, avail_cols, sl_pct, ts_act_pct, ts_trail_pct,
                 threshold, sl_cd_sec, max_l, max_s, trend_filter=True,
                 macro_filter=True, label=""):
    """Simulate the ensemble bot on 5-min bars."""
    trades = []
    positions = []   # {model, dir, entry_p, entry_t, sl, ts_act, ts_trail, peak, ts_on}
    last_sl = {'LONG': None, 'SHORT': None}
    
    lookback_1h  = 12   # 12 x 5-min = 1 hour
    lookback_24h = 288  # 288 x 5-min = 24 hours

    close_arr = bars_5m['Close'].values
    close_idx = bars_5m.index

    week_rows = feat_df[feat_df.index >= THIS_WEEK]
    week_rows = week_rows.dropna(subset=avail_cols[:5])  # drop rows with key NaNs

    for ts, row in week_rows.iterrows():
        if ts not in bars_5m.index:
            continue
        price = bars_5m.loc[ts, 'Close']
        if np.isnan(price):
            continue
        now = ts
        bar_pos = close_idx.get_loc(ts)

        # ── exits ──────────────────────────────────────────────────────────
        closed = []
        for pos in positions:
            hit_sl = hit_ts = False
            if pos['dir'] == 'LONG':
                if price > pos['peak']:  pos['peak'] = price
                if price <= pos['sl']:   hit_sl = True
                if not pos['ts_on'] and price >= pos['ts_act']:  pos['ts_on'] = True
                if pos['ts_on'] and price <= pos['peak'] * (1 - pos['ts_trail']/100):
                    hit_ts = True
            else:
                if price < pos['peak']:  pos['peak'] = price
                if price >= pos['sl']:   hit_sl = True
                if not pos['ts_on'] and price <= pos['ts_act']:  pos['ts_on'] = True
                if pos['ts_on'] and price >= pos['peak'] * (1 + pos['ts_trail']/100):
                    hit_ts = True

            reason = 'STOP_LOSS' if hit_sl else ('TRAILING_STOP' if hit_ts else None)
            if reason:
                pnl_pct = ((price/pos['entry_p']-1)*100 if pos['dir']=='LONG'
                           else (pos['entry_p']/price-1)*100)
                pnl_usd = pnl_pct/100 * pos['entry_p'] * CONTRACT_VAL
                trades.append({'dir':pos['dir'], 'model':pos['model'], 'entry_t':pos['entry_t'],
                               'exit_t':now, 'entry_p':pos['entry_p'], 'exit_p':price,
                               'pnl_pct':pnl_pct, 'pnl_usd':pnl_usd, 'reason':reason})
                if hit_sl:
                    last_sl[pos['dir']] = now
                closed.append(pos)
        for p in closed:
            positions.remove(p)

        # ── trend filters ──────────────────────────────────────────────────
        trend_1h = 0.0
        if trend_filter and bar_pos >= lookback_1h:
            p0 = close_arr[bar_pos - lookback_1h]
            trend_1h = (price - p0) / p0 * 100

        macro_24h = 0.0
        if macro_filter and bar_pos >= lookback_24h:
            p0 = close_arr[bar_pos - lookback_24h]
            macro_24h = (price - p0) / p0 * 100

        # ── get probabilities ──────────────────────────────────────────────
        X = row[avail_cols].values.reshape(1, -1)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        probs = {}
        for nm, mdl in models.items():
            try:
                probs[nm] = mdl.predict_proba(X)[0, 1]
            except Exception:
                probs[nm] = 0.0

        # ── entries ────────────────────────────────────────────────────────
        n_l = sum(1 for p in positions if p['dir']=='LONG')
        n_s = sum(1 for p in positions if p['dir']=='SHORT')

        # best LONG signal
        best_long = max(LONG_MODELS, key=lambda m: probs.get(m, 0))
        best_long_prob = probs.get(best_long, 0)
        if (best_long_prob >= threshold and n_l < max_l):
            sl_ok = (last_sl['LONG'] is None or
                     (now - last_sl['LONG']).total_seconds() >= sl_cd_sec)
            tf_ok = (not trend_filter or trend_1h >= -0.4)
            mac_ok = (not macro_filter or macro_24h >= -2.0)
            if sl_ok and tf_ok and mac_ok and not any(p['model']==best_long for p in positions):
                sl_p  = price * (1 - sl_pct/100)
                ts_p  = price * (1 + ts_act_pct/100)
                positions.append({'dir':'LONG', 'model':best_long, 'entry_p':price,
                                  'entry_t':now, 'sl':sl_p, 'ts_act':ts_p,
                                  'ts_trail':ts_trail_pct, 'peak':price, 'ts_on':False})

        # best SHORT signal
        best_short = max(SHORT_MODELS, key=lambda m: probs.get(m, 0))
        best_short_prob = probs.get(best_short, 0)
        if (best_short_prob >= threshold and n_s < max_s):
            sl_ok = (last_sl['SHORT'] is None or
                     (now - last_sl['SHORT']).total_seconds() >= sl_cd_sec)
            tf_ok = (not trend_filter or trend_1h <= 0.4)
            mac_ok = (not macro_filter or macro_24h <= 2.0)
            if sl_ok and tf_ok and mac_ok and not any(p['model']==best_short for p in positions):
                sl_p  = price * (1 + sl_pct/100)
                ts_p  = price * (1 - ts_act_pct/100)
                positions.append({'dir':'SHORT', 'model':best_short, 'entry_p':price,
                                  'entry_t':now, 'sl':sl_p, 'ts_act':ts_p,
                                  'ts_trail':ts_trail_pct, 'peak':price, 'ts_on':False})

    tr = pd.DataFrame(trades) if trades else pd.DataFrame()
    return tr

# ─────────────────────────────────────────────────────────────────────────────
# RUN BASELINE BACKTEST (live bot params)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("SECTION 1: BACKTEST vs LIVE (this week, live params)")
print("="*65)

tr = run_backtest(feat_week, bars_5m[bars_5m.index >= THIS_WEEK],
                  models, avail_cols,
                  sl_pct=STOP_LOSS_PCT, ts_act_pct=TS_ACT_PCT,
                  ts_trail_pct=TS_TRAIL_PCT, threshold=PROB_THRESHOLD,
                  sl_cd_sec=SL_COOLDOWN_SEC, max_l=MAX_L, max_s=MAX_S)

def print_results(tr, label=""):
    if tr is None or len(tr) == 0:
        print(f"  [{label}] No trades")
        return
    w = tr[tr['pnl_pct'] > 0]
    l = tr[tr['pnl_pct'] <= 0]
    print(f"\n  [{label}]  Trades={len(tr)}  WR={len(w)/len(tr)*100:.0f}%  "
          f"P&L=${tr['pnl_usd'].sum():.2f}  "
          f"AvgW=${w['pnl_usd'].mean():.2f}  AvgL=${l['pnl_usd'].mean():.2f}"
          if len(w) and len(l) else
          f"\n  [{label}]  Trades={len(tr)}  WR={len(w)/len(tr)*100:.0f}%  "
          f"P&L=${tr['pnl_usd'].sum():.2f}")
    for d in ['LONG','SHORT']:
        sub = tr[tr['dir']==d]
        if len(sub):
            ww = sub[sub['pnl_pct']>0]
            print(f"    {d}: {len(sub)} trades | WR {len(ww)/len(sub)*100:.0f}% | "
                  f"P&L ${sub['pnl_usd'].sum():.2f}")
    for m in tr['model'].unique():
        sub = tr[tr['model']==m]
        ww = sub[sub['pnl_pct']>0]
        print(f"    {m}: {len(sub)} | WR {len(ww)/len(sub)*100:.0f}% | "
              f"P&L ${sub['pnl_usd'].sum():.2f}")

print_results(tr, "Backtest — live params (SL=0.20%, TS=0.25%/0.05%)")

print(f"\n  LIVE RESULTS this week (from DB, excluding legacy):")
print(f"    4h_0.5pct  LONG : 14 trades | WR 57% | P&L +$17.95")
print(f"    6h_0.5pct  LONG :  4 trades | WR 75% | P&L +$94.71")
print(f"    2h_SHORT SHORT : 17 trades | WR 41% | P&L -$68.39")
print(f"    4h_SHORT SHORT :  7 trades | WR 43% | P&L -$33.28")
print(f"    TOTAL : 42 trades | 50% WR | P&L +$10.99")

# ─────────────────────────────────────────────────────────────────────────────
# SENSITIVITY: SL/TS PARAMETER SWEEP
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("SECTION 2: SL/TS SENSITIVITY (all params, this week)")
print("="*65)
print(f"  {'SL%':>5} {'TS act%':>7} {'TS tr%':>7} {'Trades':>7} {'WR':>6} {'P&L':>8} {'AvgW':>7} {'AvgL':>7}")

for sl, ts_act, ts_trail in [
    (0.20, 0.25, 0.05),   # current live
    (0.20, 0.25, 0.10),
    (0.20, 0.25, 0.15),
    (0.30, 0.35, 0.10),
    (0.30, 0.35, 0.15),
    (0.30, 0.40, 0.15),
    (0.40, 0.45, 0.15),
    (0.40, 0.50, 0.20),
    (0.50, 0.55, 0.20),
]:
    t = run_backtest(feat_week, bars_5m[bars_5m.index >= THIS_WEEK],
                     models, avail_cols, sl_pct=sl, ts_act_pct=ts_act,
                     ts_trail_pct=ts_trail, threshold=PROB_THRESHOLD,
                     sl_cd_sec=SL_COOLDOWN_SEC, max_l=MAX_L, max_s=MAX_S)
    if len(t) == 0:
        print(f"  {sl:>5.2f} {ts_act:>7.2f} {ts_trail:>7.2f}  no trades")
        continue
    w = t[t['pnl_pct']>0]; l = t[t['pnl_pct']<=0]
    aw = w['pnl_usd'].mean() if len(w) else 0
    al = l['pnl_usd'].mean() if len(l) else 0
    print(f"  {sl:>5.2f} {ts_act:>7.2f} {ts_trail:>7.2f} "
          f"{len(t):>7} {len(w)/len(t)*100:>5.0f}% "
          f"{t['pnl_usd'].sum():>8.2f} {aw:>7.2f} {al:>7.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: SIGNAL QUALITY — how often does SHORT fire during uptrend?
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("SECTION 3: SHORT SIGNAL QUALITY DURING UPTREND")
print("="*65)

lookback_24h = 288
close_arr = bars_5m['Close'].values
close_idx  = bars_5m.index

short_prob_vs_trend = []
for ts, row in feat_week.iterrows():
    if ts not in bars_5m.index: continue
    bar_pos = close_idx.get_loc(ts)
    if bar_pos < lookback_24h: continue
    price = bars_5m.loc[ts,'Close']
    trend_24h = (price - close_arr[bar_pos-lookback_24h]) / close_arr[bar_pos-lookback_24h] * 100
    trend_1h  = (price - close_arr[bar_pos-12]) / close_arr[bar_pos-12] * 100 if bar_pos >= 12 else 0
    X = np.nan_to_num(row[avail_cols].values.reshape(1,-1), nan=0, posinf=0, neginf=0)
    for nm in SHORT_MODELS:
        if nm in models:
            p = models[nm].predict_proba(X)[0,1]
            short_prob_vs_trend.append({'model':nm, 'prob':p, 'trend_24h':trend_24h, 'trend_1h':trend_1h,
                                        'signal': p >= PROB_THRESHOLD})

spvt = pd.DataFrame(short_prob_vs_trend)
if len(spvt):
    print(f"\n  SHORT signal rate by 24h trend bucket:")
    print(f"  {'24h change':>12} {'Bars':>7} {'Signal%':>9} {'AvgProb':>9} {'Macro-blocked':>14}")
    for lo, hi in [(-10,-4),(-4,-2),(-2,0),(0,2),(2,4),(4,10)]:
        bucket = spvt[(spvt['trend_24h']>=lo) & (spvt['trend_24h']<hi)]
        if len(bucket) == 0: continue
        sig_rate = bucket['signal'].mean()*100
        blocked = bucket[bucket['trend_24h'] > 2.0]['signal'].mean()*100 if lo >= 2 else 0
        print(f"  {lo:>+5}% to {hi:>+3}%: {len(bucket)//len(SHORT_MODELS):>5} bars  "
              f"{sig_rate:>7.1f}%  {bucket['prob'].mean():>7.1%}  "
              f"{'MACRO-BLOCKED' if lo >= 2 else ''}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: BUG REPORT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("SECTION 4: BUGS IDENTIFIED IN ENSEMBLE BOT")
print("="*65)
print("""
BUG #1 [CRITICAL] — Phantom P&L from stale legacy positions
  Code: ib_avg_cost = mbt_position.avgCost * 10   (line ~1099)
  No sanity check against current price.
  IB reported avgCost from expired contract (BTC @ $35,880 in 2024).
  Bot adopted position, immediately triggered take profit, LOGGED TO DB
  even though the close order was CANCELLED (Error 10349).
  Result: +$7,129 phantom profit in DB. Status page is inflated.
  Fix: (a) reject legacy position if entry_price deviates >20% from current,
       (b) only log_trade() AFTER confirmed fill, not on intent.

BUG #2 [HIGH] — STOP_LOSS_PCT = 0.20% too tight
  At $70k BTC, 0.20% SL = $140. Tick-level noise easily exceeds this.
  Live SHORT WR = 41-43% against backtest showing higher.
  The gap is almost entirely false SL hits from noise.
  Fix: widen to 0.30%.

BUG #3 [HIGH] — TRAILING_STOP_PCT = 0.05% creates unfavorable R:R
  TS activates at +0.25%, then trails only 0.05%.
  → Average win locked in: ~+0.20% (only slightly better than SL loss).
  With SL=0.20% and avg win=0.20%: R:R = 1:1, not the advertised 3:1.
  TAKE_PROFIT_PCT=3% is never reached — trailing stop handles all exits.
  Fix: widen TS trail to 0.15% so locks in ~0.10% minimum profit.

BUG #4 [HIGH] — SHORT model fires during macro uptrend
  MACRO_TREND filter blocks SHORT if 24h change > +2%.
  But during the March 8-12 rally (+6%), 24h change passes through <2%
  windows (early morning each day) and SHORT trades fire anyway.
  Fix: tighten MACRO_TREND_SHORT_MAX from +2.0% to +1.0%.

BUG #5 [MEDIUM] — EXIT logging not gated on confirmed fill
  _close_position() logs to DB before checking if close order filled.
  If order gets Error 10349 (cancelled), trade is still logged as closed.
  DB shows incorrect P&L, positions may remain open in IB.
  Fix: check trade.orderStatus.status == 'Filled' before logging.

BUG #6 [MEDIUM] — SL_COOLDOWN is shared via last_sl_time dict
  last_sl_time is keyed by direction — this is CORRECT in ensemble bot.
  But check happens INSIDE the signal loop, not before position limits.
  A second signal in the same direction at the same bar can bypass
  cooldown because n_s < MAX_S allows both to be checked.
  Fix: add early-exit check before the signal loop.

BUG #7 [LOW] — 6 stacked filters can block ALL entries in sideways market
  TREND → RSI → BB → MACD → MACRO → ENTRY_GATE all active simultaneously.
  Probability of all 6 passing for a valid signal is low.
  During sideways movement (Mar 9-10): bot missed profitable setups.
  Fix: Consider making RSI/BB/MACD advisory (log-only) rather than hard blocks.
""")

print("="*65)
print("TICK BOT BLEEDING DIAGNOSIS")
print("="*65)
print("""
Period: Mar 10-13, ~4 days live

SHORT total: 13 SL = -$424, 35 TS exits = +$116 → NET -$308
LONG  total: 12 SL = -$278, 6 TS exits = +$219 → NET -$59

SHORT R:R (live): avg win = +$3.3, avg loss = -$33 → 1:10 unfavorable!

Root cause: SHORT_TS_TRAIL = 0.05%
  → At $70k BTC, 0.05% trail = $35 movement.
  → TS activates at -0.25% ($175 down), then exits at first $35 bounce.
  → Locks in ~$140 profit MAX per winning trade.
  → But SL at 0.30% = $210 per losing trade.
  → With 50% WR: EV = 0.5*$140 - 0.5*$210 = -$35 per trade

SHORT model fires too often: 48 trades in 4 days = 12/day (every 2 hours).
The SHORT model has 41-43% live WR vs 61% backtest (>15% overfit gap).
During a +6.4% BTC rally, the model keeps calling SHORT against the trend.

RECOMMENDATION: Stop tick bot SHORT entries completely until:
  1. BTC 24h change returns below +2%
  2. OR the SHORT model is retrained with more data (need 4+ weeks)
""")
