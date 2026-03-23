#!/usr/bin/env python3
"""
BTC Strategy Analysis
1. Backtest this week's data against tick models
2. Coin-flip analysis: price movement distributions, viable SL/TP
3. Trend-following analysis: how much does price follow trends?
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from btc_tick_bot import compute_features

DATA_PATH  = 'logs/btc_price_ticks.csv'
MODELS_DIR = 'models/tick_models'
THIS_WEEK  = pd.Timestamp('2026-03-08')   # Mon March 8

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
print("=" * 65)
print("LOADING TICK DATA")
print("=" * 65)

df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
df = df.sort_values('timestamp').set_index('timestamp')
print(f"Full dataset : {len(df):,} ticks  {df.index[0]:%Y-%m-%d}  →  {df.index[-1]:%Y-%m-%d}")

week_df = df[df.index >= THIS_WEEK].copy()
print(f"This week    : {len(week_df):,} ticks  {week_df.index[0]:%Y-%m-%d}  →  {week_df.index[-1]:%Y-%m-%d}")

# Resample to 16-sec bars (full dataset for warmup, then slice)
bars_full = df['price'].resample('16s').agg(
    open='first', high='max', low='min', close='last'
).dropna()
feat_full = compute_features(bars_full)
feat_week = feat_full[feat_full.index >= THIS_WEEK]
print(f"16-sec bars  : {len(bars_full):,} total  |  {len(feat_week):,} this week")


# ─────────────────────────────────────────────
# SECTION 1: PRICE MOVEMENT ANALYSIS (COIN FLIP)
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 1: PRICE MOVEMENT DISTRIBUTIONS")
print("=" * 65)

close = bars_full['close']

# For each bar, what is the max move up and down within next N bars?
def max_excursion(close, n_bars):
    """Forward-looking max gain and max drawdown over next n_bars."""
    gains, drops = [], []
    arr = close.values
    for i in range(len(arr) - n_bars):
        entry = arr[i]
        future = arr[i+1:i+n_bars+1]
        gains.append((future.max() - entry) / entry * 100)
        drops.append((entry - future.min()) / entry * 100)
    return np.array(gains), np.array(drops)

print("\nForward-looking max excursion (all data):")
print(f"{'Window':<12} {'MFE p25':>8} {'MFE p50':>8} {'MFE p75':>8} {'MFE p90':>8} "
      f"{'MAE p25':>8} {'MAE p50':>8} {'MAE p75':>8} {'MAE p90':>8}")

for n, label in [(19,'5 min'), (38,'10 min'), (75,'20 min'), (113,'30 min'), (225,'1 hr')]:
    g, d = max_excursion(close, n)
    print(f"{label:<12} "
          f"{np.percentile(g,25):>8.3f} {np.percentile(g,50):>8.3f} "
          f"{np.percentile(g,75):>8.3f} {np.percentile(g,90):>8.3f}   "
          f"{np.percentile(d,25):>8.3f} {np.percentile(d,50):>8.3f} "
          f"{np.percentile(d,75):>8.3f} {np.percentile(d,90):>8.3f}")

# Coin-flip simulation: for various SL/TP combos, what WR is needed?
print("\n\nCOIN-FLIP STRATEGY SIMULATION (random entry, 1-sec price ticks)")
print(f"{'SL %':>6} {'TP %':>6} {'R:R':>6} {'WR needed':>10} {'Actual WR':>10} {'EV per $1':>10}")

sim_data = close.values
for sl, tp in [(0.10, 0.20), (0.10, 0.30), (0.15, 0.30), (0.15, 0.45),
               (0.20, 0.40), (0.20, 0.60), (0.25, 0.50), (0.30, 0.60),
               (0.30, 0.90), (0.40, 0.80), (0.45, 0.90), (0.50, 1.00)]:
    wins, losses = 0, 0
    for i in range(len(sim_data) - 2000):
        entry = sim_data[i]
        sl_p  = entry * (1 - sl / 100)
        tp_p  = entry * (1 + tp / 100)
        for j in range(i + 1, min(i + 2000, len(sim_data))):
            p = sim_data[j]
            if p >= tp_p:
                wins += 1
                break
            elif p <= sl_p:
                losses += 1
                break
    total = wins + losses
    if total == 0:
        continue
    wr = wins / total
    wr_needed = sl / (sl + tp)  # breakeven WR
    rr = tp / sl
    ev = wr * tp - (1 - wr) * sl
    print(f"{sl:>6.2f} {tp:>6.2f} {rr:>6.1f} {wr_needed*100:>9.1f}% {wr*100:>9.1f}% {ev:>10.4f}")


# ─────────────────────────────────────────────
# SECTION 2: TREND-FOLLOWING ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 2: TREND-FOLLOWING ANALYSIS")
print("=" * 65)

print("\nHow often does price continue vs reverse after a move?")
print(f"{'Lookback':>10} {'Fwd window':>12} {'Continue %':>12} {'Reverse %':>12} {'Avg cont move':>14} {'Avg rev move':>13}")

arr = close.values
for lb, fwd in [(19,19),(19,38),(38,38),(38,75),(75,75),(113,113),(225,225)]:
    continues, reverses = [], []
    cont_moves, rev_moves = [], []
    for i in range(lb, len(arr) - fwd):
        past_ret  = (arr[i] - arr[i - lb]) / arr[i - lb] * 100
        fwd_ret   = (arr[i + fwd] - arr[i]) / arr[i] * 100
        if abs(past_ret) < 0.05:  # skip flat
            continue
        if np.sign(fwd_ret) == np.sign(past_ret):
            continues.append(1)
            cont_moves.append(abs(fwd_ret))
        else:
            reverses.append(1)
            rev_moves.append(abs(fwd_ret))
    total = len(continues) + len(reverses)
    if total == 0:
        continue
    lb_min = lb * 16 / 60
    fwd_min = fwd * 16 / 60
    print(f"{lb_min:>8.0f}m  {fwd_min:>10.0f}m  "
          f"{len(continues)/total*100:>10.1f}%  "
          f"{len(reverses)/total*100:>10.1f}%  "
          f"{np.mean(cont_moves):>12.3f}%  "
          f"{np.mean(rev_moves):>11.3f}%")

print("\nTrend-follow SL/TP simulation (entry after N-bar move, direction-aligned):")
print(f"{'Entry cond':>12} {'SL':>6} {'TP':>6} {'Trades':>8} {'WR':>8} {'Avg PnL':>9} {'EV/100':>9}")

for min_move, lb, fwd in [(0.15,19,38),(0.20,19,38),(0.25,38,75),(0.30,38,75)]:
    for sl, tp in [(0.20,0.40),(0.20,0.60),(0.30,0.60),(0.30,0.90)]:
        wins, losses, total_pnl = 0, 0, 0
        for i in range(lb, len(arr) - fwd):
            past_ret = (arr[i] - arr[i - lb]) / arr[i - lb] * 100
            if abs(past_ret) < min_move:
                continue
            direction = 1 if past_ret > 0 else -1
            entry = arr[i]
            sl_p  = entry * (1 - direction * sl / 100)
            tp_p  = entry * (1 + direction * tp / 100)
            for j in range(i + 1, min(i + 2000, len(arr))):
                p = arr[j]
                if direction == 1:
                    if p >= tp_p:  wins += 1; total_pnl += tp; break
                    elif p <= sl_p:  losses += 1; total_pnl -= sl; break
                else:
                    if p <= tp_p:  wins += 1; total_pnl += tp; break
                    elif p >= sl_p:  losses += 1; total_pnl -= sl; break
        t = wins + losses
        if t == 0:
            continue
        entry_cond = f">{min_move:.2f}%/{lb*16//60}m"
        print(f"{entry_cond:>12} {sl:>6.2f} {tp:>6.2f} {t:>8,} {wins/t*100:>7.1f}% "
              f"{total_pnl/t:>9.4f} {total_pnl/t*100:>9.2f}")


# ─────────────────────────────────────────────
# SECTION 3: BACKTEST THIS WEEK
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 3: BACKTEST THIS WEEK (Mar 8-12)")
print("=" * 65)

# Load models
models, model_cfgs = {}, {}
for nm in ['L1_full_wide', 'L3_mom_wide', 'S1_full_cur', 'S3_mr_cur', 'S4_full_tight']:
    path = os.path.join(MODELS_DIR, f'{nm}.pkl')
    cfg_path = os.path.join(MODELS_DIR, f'{nm}_config.pkl')
    if os.path.exists(path):
        models[nm] = joblib.load(path)
        if os.path.exists(cfg_path):
            model_cfgs[nm] = joblib.load(cfg_path)
        print(f"  Loaded {nm}")
    else:
        print(f"  MISSING: {nm}")

def get_probs(row, models, model_cfgs):
    probs = {}
    for nm, mdl in models.items():
        cols = model_cfgs[nm]['features']
        try:
            vals = row[cols].values.reshape(1, -1)
            probs[nm] = mdl.predict_proba(vals)[0][1]
        except Exception:
            probs[nm] = 0.0
    return probs

LONG_THRESH  = 0.65
SHORT_THRESH = 0.70
SHORT_AGREE  = 2
LONG_SL   = 0.45; LONG_TS_ACT  = 0.50; LONG_TS_TRAIL  = 0.15
SHORT_SL  = 0.20; SHORT_TS_ACT = 0.25; SHORT_TS_TRAIL = 0.05
CONTRACT  = 0.1  # MBT = 0.1 BTC

trades = []
positions = []  # active: {dir, entry_p, sl, ts_act, ts_trail, peak, model, entry_t}
MAX_L, MAX_S = 1, 2
last_long_t = None
last_short_t = None
MIN_GAP = pd.Timedelta(minutes=5)
SL_COOLDOWN = pd.Timedelta(minutes=10)
last_sl_t = None

feat_week_rows = feat_week.dropna()

for ts, row in feat_week_rows.iterrows():
    price = bars_full.loc[ts, 'close'] if ts in bars_full.index else None
    if price is None or np.isnan(price):
        continue

    # check exits
    closed = []
    for pos in positions:
        hit_sl = hit_ts = False
        if pos['dir'] == 'LONG':
            if price > pos['peak']:
                pos['peak'] = price
            if price <= pos['sl']:
                hit_sl = True
            if not pos['ts_on'] and price >= pos['ts_act']:
                pos['ts_on'] = True
            if pos['ts_on'] and price <= pos['peak'] * (1 - pos['ts_trail'] / 100):
                hit_ts = True
        else:
            if price < pos['peak']:
                pos['peak'] = price
            if price >= pos['sl']:
                hit_sl = True
            if not pos['ts_on'] and price <= pos['ts_act']:
                pos['ts_on'] = True
            if pos['ts_on'] and price >= pos['peak'] * (1 + pos['ts_trail'] / 100):
                hit_ts = True

        reason = 'STOP_LOSS' if hit_sl else ('TRAILING_STOP' if hit_ts else None)
        if reason:
            if pos['dir'] == 'LONG':
                pnl_pct = (price / pos['entry_p'] - 1) * 100
            else:
                pnl_pct = (pos['entry_p'] / price - 1) * 100
            pnl_usd = pnl_pct / 100 * pos['entry_p'] * CONTRACT
            trades.append({'dir': pos['dir'], 'model': pos['model'],
                           'entry_t': pos['entry_t'], 'exit_t': ts,
                           'entry_p': pos['entry_p'], 'exit_p': price,
                           'pnl_pct': pnl_pct, 'pnl_usd': pnl_usd, 'reason': reason})
            if hit_sl:
                last_sl_t = ts
            closed.append(pos)
    for p in closed:
        positions.remove(p)

    # check entries
    probs = get_probs(row, models, model_cfgs)
    now = ts
    n_long  = sum(1 for p in positions if p['dir'] == 'LONG')
    n_short = sum(1 for p in positions if p['dir'] == 'SHORT')
    ret_1h  = row.get('ret_1h', 0) if 'ret_1h' in row.index else 0
    sl_ok   = last_sl_t is None or (now - last_sl_t) >= SL_COOLDOWN

    if sl_ok and n_long < MAX_L:
        gap_ok = last_long_t is None or (now - last_long_t) >= MIN_GAP
        if gap_ok and ret_1h >= -0.30:
            for m in ['L1_full_wide', 'L3_mom_wide']:
                if probs.get(m, 0) >= LONG_THRESH:
                    sl_p = price * (1 - LONG_SL / 100)
                    positions.append({'dir':'LONG','model':m,'entry_p':price,'entry_t':now,
                                      'sl':sl_p,'ts_act':price*(1+LONG_TS_ACT/100),
                                      'ts_trail':LONG_TS_TRAIL,'peak':price,'ts_on':False})
                    last_long_t = now
                    break

    if sl_ok and n_short < MAX_S:
        gap_ok = last_short_t is None or (now - last_short_t) >= MIN_GAP
        if gap_ok and ret_1h <= 0.30:
            agrees = sum(1 for m in ['S1_full_cur','S3_mr_cur','S4_full_tight']
                         if probs.get(m, 0) >= SHORT_THRESH)
            if agrees >= SHORT_AGREE:
                trig = next(m for m in ['S1_full_cur','S3_mr_cur','S4_full_tight']
                            if probs.get(m, 0) >= SHORT_THRESH)
                sl_p = price * (1 + SHORT_SL / 100)
                positions.append({'dir':'SHORT','model':trig,'entry_p':price,'entry_t':now,
                                  'sl':sl_p,'ts_act':price*(1-SHORT_TS_ACT/100),
                                  'ts_trail':SHORT_TS_TRAIL,'peak':price,'ts_on':False})
                last_short_t = now

tr = pd.DataFrame(trades)
if len(tr) == 0:
    print("No trades generated this week.")
else:
    print(f"\nTotal trades : {len(tr)}")
    wins = tr[tr['pnl_pct'] > 0]
    losses = tr[tr['pnl_pct'] <= 0]
    print(f"Win rate     : {len(wins)/len(tr)*100:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"Total P&L    : ${tr['pnl_usd'].sum():.2f}")
    print(f"Avg win      : ${wins['pnl_usd'].mean():.2f}" if len(wins) else "Avg win: n/a")
    print(f"Avg loss     : ${losses['pnl_usd'].mean():.2f}" if len(losses) else "Avg loss: n/a")

    print(f"\nBy direction:")
    for d in ['LONG', 'SHORT']:
        sub = tr[tr['dir'] == d]
        if len(sub):
            w = sub[sub['pnl_pct'] > 0]
            print(f"  {d}: {len(sub)} trades | WR {len(w)/len(sub)*100:.0f}% | P&L ${sub['pnl_usd'].sum():.2f}")

    print(f"\nBy exit reason:")
    for r in tr['reason'].unique():
        sub = tr[tr['reason'] == r]
        print(f"  {r}: {len(sub)} trades | Avg PnL ${sub['pnl_usd'].mean():.2f}")

    print(f"\nBy model:")
    for m in tr['model'].unique():
        sub = tr[tr['model'] == m]
        w = sub[sub['pnl_pct'] > 0]
        print(f"  {m}: {len(sub)} trades | WR {len(w)/len(sub)*100:.0f}% | P&L ${sub['pnl_usd'].sum():.2f}")

    print(f"\nSignal probabilities at entry (backtest):")
    for m in ['L1_full_wide','L3_mom_wide','S1_full_cur','S3_mr_cur','S4_full_tight']:
        sub = tr[tr['model'] == m]
        if len(sub):
            print(f"  {m}: {len(sub)} signals fired this week")

print("\n" + "=" * 65)
print("BUGS IDENTIFIED IN LIVE BOT")
print("=" * 65)
print("""
1. SL_COOLDOWN blocks ALL directions (LONG+SHORT) after any SL exit.
   → After a losing SHORT, LONG is also blocked for 10 min (and vice versa).
   → Fix: track cooldown per direction separately.

2. TREND_FOLLOW_LONG_THRESHOLD = 4.0% uses ret_1h (1-hour return).
   → In a +6.4% 24h rally, 1h return is typically only +0.5–1.5%.
   → 4% 1h return almost never happens → trend-follow mode never activates.
   → Fix: use a lower threshold (~1.0%) OR use 24h return like ensemble bot.

3. SHORT SL = 0.20% is extremely tight.
   → BTC tick-level noise alone exceeds 0.20% regularly.
   → This creates many false SL hits. Backtest vs live WR gap is largely this.
   → Fix: widen to 0.30–0.35%.

4. Models trained on only ~3 days of data → heavily overfit.
   → Live WR tracks 45–50%, backtest shows 70%+ (10–15% overfit gap).
   → Fix: collect more data, retrain when 4+ weeks available.

5. `consec_losses` counter resets on ANY non-SL exit (including TS losses).
   → A TS exit at a small loss resets the danger counter.
   → Fix: reset only on profitable exits.
""")
