#!/usr/bin/env python3
"""
Backtest ensemble bot over last 7 active days using archived parquet data.
Replays bar-by-bar with the same models and filters, comparing to actual trades.
"""

import os, json, sqlite3, joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

BASE = os.path.dirname(os.path.abspath(__file__))

# ── Load models ──────────────────────────────────────────────────────────
BTC_ENSEMBLE_MODELS = [
    {'horizon': '2h', 'threshold': '0.5', 'priority': 1, 'horizon_bars': 24, 'direction': 'LONG'},
    {'horizon': '4h', 'threshold': '0.5', 'priority': 2, 'horizon_bars': 48, 'direction': 'LONG'},
    {'horizon': '6h', 'threshold': '0.5', 'priority': 3, 'horizon_bars': 72, 'direction': 'LONG'},
    {'horizon': '2h', 'threshold': '0.5', 'priority': 1, 'horizon_bars': 24, 'direction': 'SHORT'},
    {'horizon': '4h', 'threshold': '0.5', 'priority': 2, 'horizon_bars': 48, 'direction': 'SHORT'},
]

models = {}
for m in BTC_ENSEMBLE_MODELS:
    d = m.get('direction', 'LONG')
    suffix = '_SHORT' if d == 'SHORT' else ''
    name = f"{m['horizon']}_{m['threshold']}pct{suffix}"
    path = os.path.join(BASE, f"models_btc_v2/model_{name}.joblib")
    if os.path.exists(path):
        models[name] = {'model': joblib.load(path), 'config': m}

feature_path = os.path.join(BASE, "models_btc_v2/feature_columns.json")
with open(feature_path) as f:
    feature_cols = json.load(f)

# Entry gate models
entry_gate_long = None
entry_gate_short = None
try:
    entry_gate_long = joblib.load(os.path.join(BASE, 'models/entry_filter_long.pkl'))
    entry_gate_short = joblib.load(os.path.join(BASE, 'models/entry_filter_short.pkl'))
except:
    pass

print(f"Loaded {len(models)} models, {len(feature_cols)} features")

# ── Load parquet ─────────────────────────────────────────────────────────
archive_path = os.path.join(BASE, 'data/archive/BTC_features_archive.parquet')
live_path = os.path.join(BASE, 'data/BTC_features.parquet')

df = pd.read_parquet(archive_path) if os.path.exists(archive_path) else pd.read_parquet(live_path)
print(f"Parquet: {len(df)} bars, {df.index[0]} to {df.index[-1]}")

# ── Load actual trades ───────────────────────────────────────────────────
conn = sqlite3.connect(os.path.join(BASE, 'trades.db'))
actual = pd.read_sql("""
    SELECT entry_time, model_id, direction, entry_price, pnl_dollar, exit_reason
    FROM trades WHERE model_id NOT LIKE 'legacy%' AND entry_time >= '2026-03-10'
    ORDER BY entry_time
""", conn)
conn.close()
print(f"Actual trades: {len(actual)}")

# ── Filter constants (match current bot config) ─────────────────────────
PROB_LONG = 0.55
PROB_SHORT = 0.55
TREND_FILTER_ENABLED = True
TREND_SHORT_THRESH = 0.4
TREND_LONG_THRESH = -0.4
TREND_LOOKBACK = 12
RSI_LONG_MIN = 50
RSI_SHORT_MAX = 70
RSI_SHORT_MIN = 40
BB_LONG_MIN = 0.40
BB_SHORT_MAX = 0.80
MACD_LONG_MIN = 0
MACD_SHORT_MAX = 10
MACRO_LONG_MIN = -2.0
MACRO_SHORT_MAX = 1.0
MACRO_LOOKBACK = 288
ENTRY_GATE_THRESH = 0.50
CROSS_TF_6H_LONG_MIN = 0.40
SL_PCT = 0.50
TS_ACTIVATE = 0.35
TS_TRAIL = 0.15
SL_COOLDOWN_SEC = 300


def get_indicator(row, col, default=None):
    v = row.get(col, default)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return default
    return v


# ── Replay bar-by-bar ────────────────────────────────────────────────────
start_date = '2026-03-10'
replay_df = df[df.index >= start_date].copy()
print(f"\nReplaying {len(replay_df)} bars from {replay_df.index[0]}")

bt_signals = []  # All signals that would trigger entry
bt_entries = []  # Signals that pass all filters
last_sl_time = {'LONG': datetime.min, 'SHORT': datetime.min}
positions = {}  # model_id -> {direction, entry_price, entry_time, high/low}

for i in range(50, len(replay_df)):
    bar_time = replay_df.index[i]
    row = replay_df.iloc[i]
    price = row.get('Close', row.get('close', None))
    if price is None or np.isnan(price):
        continue

    # ── Check exits on existing positions ────────────────────────────
    closed = []
    for mid, pos in positions.items():
        d = pos['direction']
        if d == 'LONG':
            # SL check
            sl_price = pos['entry_price'] * (1 - SL_PCT / 100)
            if price <= sl_price:
                closed.append((mid, 'STOP_LOSS', price))
                last_sl_time[d] = bar_time.to_pydatetime() if hasattr(bar_time, 'to_pydatetime') else bar_time
                continue
            # TS check
            if price > pos.get('high', pos['entry_price']):
                pos['high'] = price
            ts_activate_price = pos['entry_price'] * (1 + TS_ACTIVATE / 100)
            if pos.get('high', pos['entry_price']) >= ts_activate_price:
                trail_price = pos['high'] * (1 - TS_TRAIL / 100)
                if price <= trail_price:
                    closed.append((mid, 'TRAILING_STOP', price))
                    continue
        else:  # SHORT
            sl_price = pos['entry_price'] * (1 + SL_PCT / 100)
            if price >= sl_price:
                closed.append((mid, 'STOP_LOSS', price))
                last_sl_time[d] = bar_time.to_pydatetime() if hasattr(bar_time, 'to_pydatetime') else bar_time
                continue
            if price < pos.get('low', pos['entry_price']):
                pos['low'] = price
            ts_activate_price = pos['entry_price'] * (1 - TS_ACTIVATE / 100)
            if pos.get('low', pos['entry_price']) <= ts_activate_price:
                trail_price = pos['low'] * (1 + TS_TRAIL / 100)
                if price >= trail_price:
                    closed.append((mid, 'TRAILING_STOP', price))
                    continue

    for mid, reason, exit_price in closed:
        pos = positions[mid]
        d = pos['direction']
        if d == 'LONG':
            pnl = (exit_price - pos['entry_price']) * 0.1
        else:
            pnl = (pos['entry_price'] - exit_price) * 0.1
        bt_entries.append({
            'entry_time': pos['entry_time'],
            'exit_time': bar_time,
            'model_id': mid,
            'direction': d,
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'pnl': pnl,
            'exit_reason': reason,
        })
        del positions[mid]

    # ── Generate signals (every 5-min bar) ───────────────────────────
    window = replay_df.iloc[max(0, i-300):i+1]
    available = [c for c in feature_cols if c in window.columns]
    if len(available) < len(feature_cols) * 0.8:
        continue

    X = window[available].iloc[[-1]].fillna(0).replace([np.inf, -np.inf], 0)

    all_probs = {}
    raw_signals = []
    for mname, minfo in models.items():
        try:
            proba = minfo['model'].predict_proba(X)[0, 1]
            all_probs[mname] = proba
            cfg = minfo['config']
            d = cfg.get('direction', 'LONG')
            thresh = PROB_LONG if d == 'LONG' else PROB_SHORT
            if proba >= thresh:
                raw_signals.append({
                    'model_id': mname, 'direction': d, 'probability': proba,
                    'priority': cfg['priority']
                })
        except:
            pass

    if not raw_signals:
        continue

    # Direction conflict: keep stronger side
    longs = [s for s in raw_signals if s['direction'] == 'LONG']
    shorts = [s for s in raw_signals if s['direction'] == 'SHORT']
    if longs and shorts:
        l_avg = np.mean([s['probability'] for s in longs])
        s_avg = np.mean([s['probability'] for s in shorts])
        if l_avg >= s_avg:
            shorts = []
        else:
            longs = []

    # Single ML per direction
    if len(longs) > 1:
        longs = [max(longs, key=lambda s: s['probability'])]
    if len(shorts) > 1:
        shorts = [max(shorts, key=lambda s: s['probability'])]

    signals = longs + shorts

    # ── Apply filters ────────────────────────────────────────────────
    for sig in signals:
        mid = sig['model_id']
        d = sig['direction']
        prob = sig['probability']
        blocked = False
        block_reason = ""

        # Already have position?
        if mid in positions:
            continue
        # Max positions
        long_pos = sum(1 for p in positions.values() if p['direction'] == 'LONG')
        short_pos = sum(1 for p in positions.values() if p['direction'] == 'SHORT')
        if d == 'LONG' and long_pos >= 2:
            continue
        if d == 'SHORT' and short_pos >= 2:
            continue

        # Opposite direction block
        ml_long_open = sum(1 for p in positions.values() if p['direction'] == 'LONG')
        ml_short_open = sum(1 for p in positions.values() if p['direction'] == 'SHORT')
        if d == 'LONG' and ml_short_open > 0:
            continue
        if d == 'SHORT' and ml_long_open > 0:
            continue

        # SL cooldown
        bar_dt = bar_time.to_pydatetime() if hasattr(bar_time, 'to_pydatetime') else bar_time
        sl_elapsed = (bar_dt - last_sl_time[d]).total_seconds()
        if sl_elapsed < SL_COOLDOWN_SEC:
            block_reason = f"SL_COOLDOWN ({SL_COOLDOWN_SEC - sl_elapsed:.0f}s left)"
            blocked = True

        # 1h trend filter
        if not blocked and TREND_FILTER_ENABLED and i >= TREND_LOOKBACK:
            trend_slice = replay_df.iloc[i-TREND_LOOKBACK:i+1]
            if len(trend_slice) >= 2:
                t_pct = (trend_slice['Close'].iloc[-1] / trend_slice['Close'].iloc[0] - 1) * 100
                if d == 'SHORT' and t_pct > TREND_SHORT_THRESH:
                    block_reason = f"TREND {t_pct:+.2f}% > +{TREND_SHORT_THRESH}%"
                    blocked = True
                if d == 'LONG' and t_pct < TREND_LONG_THRESH:
                    block_reason = f"TREND {t_pct:+.2f}% < {TREND_LONG_THRESH}%"
                    blocked = True

        # RSI filter
        if not blocked:
            rsi = get_indicator(row, 'momentum_rsi')
            if rsi is not None:
                if d == 'LONG' and rsi < RSI_LONG_MIN:
                    block_reason = f"RSI {rsi:.1f} < {RSI_LONG_MIN}"
                    blocked = True
                elif d == 'SHORT' and rsi > RSI_SHORT_MAX:
                    block_reason = f"RSI {rsi:.1f} > {RSI_SHORT_MAX}"
                    blocked = True
                elif d == 'SHORT' and rsi < RSI_SHORT_MIN:
                    block_reason = f"RSI {rsi:.1f} < {RSI_SHORT_MIN} (oversold)"
                    blocked = True

        # BB filter
        if not blocked:
            bb = get_indicator(row, 'volatility_bbp')
            if bb is not None:
                if d == 'LONG' and bb < BB_LONG_MIN:
                    block_reason = f"BB {bb:.3f} < {BB_LONG_MIN}"
                    blocked = True
                elif d == 'SHORT' and bb > BB_SHORT_MAX:
                    block_reason = f"BB {bb:.3f} > {BB_SHORT_MAX}"
                    blocked = True

        # MACD filter
        if not blocked:
            macd = get_indicator(row, 'trend_macd')
            if macd is not None:
                if d == 'LONG' and macd < MACD_LONG_MIN:
                    block_reason = f"MACD {macd:.1f} < {MACD_LONG_MIN}"
                    blocked = True
                elif d == 'SHORT' and macd > MACD_SHORT_MAX:
                    block_reason = f"MACD {macd:.1f} > {MACD_SHORT_MAX}"
                    blocked = True

        # Cross-TF filter (6h LONG needs 2h or 4h >= 40%)
        if not blocked and mid == '6h_0.5pct' and d == 'LONG':
            p2h = all_probs.get('2h_0.5pct')
            p4h = all_probs.get('4h_0.5pct')
            if p2h is not None and p4h is not None:
                if p2h < CROSS_TF_6H_LONG_MIN and p4h < CROSS_TF_6H_LONG_MIN:
                    block_reason = f"CROSS-TF 2h={p2h*100:.0f}% 4h={p4h*100:.0f}% both < {CROSS_TF_6H_LONG_MIN*100:.0f}%"
                    blocked = True

        # Macro trend (24h)
        if not blocked and i >= MACRO_LOOKBACK:
            macro_slice = replay_df.iloc[i-MACRO_LOOKBACK:i+1]
            if len(macro_slice) >= 2:
                chg = (macro_slice['Close'].iloc[-1] / macro_slice['Close'].iloc[0] - 1) * 100
                if d == 'LONG' and chg < MACRO_LONG_MIN:
                    block_reason = f"MACRO 24h {chg:+.1f}% < {MACRO_LONG_MIN}%"
                    blocked = True
                elif d == 'SHORT' and chg > MACRO_SHORT_MAX:
                    block_reason = f"MACRO 24h {chg:+.1f}% > +{MACRO_SHORT_MAX}%"
                    blocked = True

        # Entry gate
        if not blocked and entry_gate_long is not None:
            try:
                gate_model = entry_gate_long if d == 'LONG' else entry_gate_short
                if gate_model is not None:
                    gate_features = {
                        'probability': prob,
                        'rsi': get_indicator(row, 'momentum_rsi', 50),
                        'bb_pct': get_indicator(row, 'volatility_bbp', 0.5),
                        'macd': get_indicator(row, 'trend_macd', 0),
                        'atr_pct': get_indicator(row, 'volatility_atr', 0) / price * 100 if price > 0 else 0,
                        'volatility': get_indicator(row, 'volatility_bbw', 0),
                        'hour': bar_time.hour,
                        'day_of_week': bar_time.dayofweek,
                    }
                    gate_X = pd.DataFrame([gate_features])
                    gate_prob = gate_model.predict_proba(gate_X)[0, 1]
                    if gate_prob < ENTRY_GATE_THRESH:
                        block_reason = f"GATE {gate_prob*100:.0f}% < {ENTRY_GATE_THRESH*100:.0f}%"
                        blocked = True
            except:
                pass

        bt_signals.append({
            'time': bar_time,
            'model_id': mid,
            'direction': d,
            'probability': prob,
            'blocked': blocked,
            'block_reason': block_reason,
            'price': price,
        })

        if not blocked:
            positions[mid] = {
                'direction': d,
                'entry_price': price,
                'entry_time': bar_time,
                'high': price,
                'low': price,
            }

# Close remaining positions at last price
last_price = replay_df['Close'].iloc[-1]
for mid, pos in list(positions.items()):
    d = pos['direction']
    if d == 'LONG':
        pnl = (last_price - pos['entry_price']) * 0.1
    else:
        pnl = (pos['entry_price'] - last_price) * 0.1
    bt_entries.append({
        'entry_time': pos['entry_time'],
        'exit_time': replay_df.index[-1],
        'model_id': mid,
        'direction': d,
        'entry_price': pos['entry_price'],
        'exit_price': last_price,
        'pnl': pnl,
        'exit_reason': 'OPEN',
    })

bt_df = pd.DataFrame(bt_entries)
sig_df = pd.DataFrame(bt_signals)

print(f"\n{'='*80}")
print(f"BACKTEST RESULTS (Mar 10-17)")
print(f"{'='*80}")

if len(bt_df) > 0:
    wins = (bt_df['pnl'] > 0).sum()
    total = len(bt_df)
    print(f"Backtest trades: {total}")
    print(f"Wins: {wins} ({100*wins/total:.0f}% WR)")
    print(f"PnL: ${bt_df['pnl'].sum():.2f}")
    print(f"Avg win: ${bt_df[bt_df['pnl']>0]['pnl'].mean():.2f}" if wins > 0 else "")
    print(f"Avg loss: ${bt_df[bt_df['pnl']<=0]['pnl'].mean():.2f}" if total-wins > 0 else "")

    print(f"\nBacktest trades detail:")
    for _, t in bt_df.iterrows():
        ts = str(t['entry_time'])[:16]
        print(f"  {ts} | {t['model_id']:20s} | {t['direction']:5s} | ${t['entry_price']:>10.2f} → ${t['exit_price']:>10.2f} | ${t['pnl']:+7.2f} | {t['exit_reason']}")
else:
    print("No backtest trades!")

# ── Compare to actual trades ─────────────────────────────────────────────
print(f"\n{'='*80}")
print(f"ACTUAL TRADES ({len(actual)} trades)")
print(f"{'='*80}")
for _, t in actual.iterrows():
    ts = str(t['entry_time'])[:16]
    print(f"  {ts} | {t['model_id']:20s} | {t['direction']:5s} | ${t['entry_price']:>10.2f} | ${t['pnl_dollar']:+7.2f} | {t['exit_reason']}")

actual_pnl = actual['pnl_dollar'].sum()
actual_wins = (actual['pnl_dollar'] > 0).sum()
print(f"\nActual: {len(actual)}T, {actual_wins}W ({100*actual_wins/len(actual):.0f}% WR), ${actual_pnl:.2f}")

# ── Match entries ────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print(f"ENTRY MATCHING: Backtest vs Actual")
print(f"{'='*80}")

# For each actual trade, find matching backtest trade within +-5 min window
matched = 0
unmatched_actual = []
for _, at in actual.iterrows():
    at_time = pd.Timestamp(at['entry_time'])
    at_model = at['model_id']
    found = False
    if len(bt_df) > 0:
        for _, bt in bt_df.iterrows():
            bt_time = bt['entry_time']
            if abs((bt_time - at_time).total_seconds()) < 600 and bt['model_id'] == at_model:
                price_diff = abs(bt['entry_price'] - at['entry_price'])
                print(f"  ✓ MATCH: {str(at_time)[:16]} {at_model:20s} | Actual=${at['entry_price']:.2f} BT=${bt['entry_price']:.2f} (Δ${price_diff:.2f})")
                matched += 1
                found = True
                break
    if not found:
        unmatched_actual.append(at)

# Check backtest entries not in actual
unmatched_bt = []
if len(bt_df) > 0:
    for _, bt in bt_df.iterrows():
        bt_time = bt['entry_time']
        bt_model = bt['model_id']
        found = False
        for _, at in actual.iterrows():
            at_time = pd.Timestamp(at['entry_time'])
            if abs((bt_time - at_time).total_seconds()) < 600 and bt_model == at['model_id']:
                found = True
                break
        if not found:
            unmatched_bt.append(bt)

print(f"\nMatched: {matched}/{len(actual)} actual trades")
if unmatched_actual:
    print(f"\nActual trades NOT in backtest ({len(unmatched_actual)}):")
    for at in unmatched_actual:
        ts = str(at['entry_time'])[:16]
        print(f"  ✗ {ts} | {at['model_id']:20s} | {at['direction']:5s} | ${at['entry_price']:.2f}")
if unmatched_bt:
    print(f"\nBacktest trades NOT in actual ({len(unmatched_bt)}):")
    for bt in unmatched_bt:
        ts = str(bt['entry_time'])[:16]
        print(f"  ✗ {ts} | {bt['model_id']:20s} | {bt['direction']:5s} | ${bt['entry_price']:.2f}")

# ── Filter impact summary ────────────────────────────────────────────────
if len(sig_df) > 0:
    blocked = sig_df[sig_df['blocked']]
    print(f"\n{'='*80}")
    print(f"FILTER BLOCKING SUMMARY")
    print(f"{'='*80}")
    print(f"Total signals generated: {len(sig_df)}")
    print(f"Blocked: {len(blocked)} ({100*len(blocked)/len(sig_df):.0f}%)")
    print(f"Passed: {len(sig_df)-len(blocked)}")
    if len(blocked) > 0:
        reasons = blocked['block_reason'].value_counts()
        print(f"\nBlock reasons:")
        for reason, count in reasons.items():
            print(f"  {reason}: {count}")
