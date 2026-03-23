#!/usr/bin/env python3
"""
Overnight monitor: paper-trade the best SHORT model.
Config: C_combined RF, hours 00-08 UTC+1, threshold >55%.

Runs alongside the real bot. Reads the same tick CSV,
computes features, trains model on existing data, then
monitors live ticks and logs signals + simulated trades.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import time, os, datetime, json

LOG_FILE = 'logs/short_model_monitor.log'
SIGNAL_THRESHOLD = 0.55
HOUR_START = 0   # UTC+1
HOUR_END = 8     # UTC+1
TICK_FILE = 'logs/btc_price_ticks.csv'
CHECK_INTERVAL = 16  # seconds between checks

# SL/TS params (current live)
SL_PCT = 0.20
TS_ACT_PCT = 0.25
TS_TRAIL_PCT = 0.05
MIN_GAP_SEC = 480  # 8 min between signals

C_COMBINED = ['ret_5m','ret_15m','ret_1h','vol_15m','vol_1h','chan_1h','chan_2h',
              'rsi_15m','rsi_1h','bb_1h','macd_15m','atr_pct_1h','speed_ratio','hour']

def log(msg):
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"{ts} - {msg}"
    print(line)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')

def compute_rsi(s, p):
    d = s.diff(); g = d.clip(lower=0).rolling(p).mean(); l = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - (100 / (1 + g / l.replace(0, np.nan)))

def compute_atr(h, l, c, p):
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(p).mean()

def load_and_prepare():
    """Load all tick data, resample to 16-sec bars, compute features."""
    raw = pd.read_csv(TICK_FILE, parse_dates=['timestamp'])
    raw = raw.set_index('timestamp').sort_index()
    raw = raw[raw['price'] > 0]
    
    # Also load bot log data for older prices
    log_file = 'logs/btc_16sec_from_log.csv'
    if os.path.exists(log_file):
        log_data = pd.read_csv(log_file, parse_dates=['timestamp'])
        log_data = log_data.set_index('timestamp').sort_index()
        log_data = log_data[log_data['price'] > 0]
        # Combine: use log data for older, tick data for recent
        combined = pd.concat([log_data, raw])
        combined = combined[~combined.index.duplicated(keep='last')]
        combined = combined.sort_index()
    else:
        combined = raw
    
    bars = combined['price'].resample('16s').agg(
        open='first', high='max', low='min', close='last'
    ).dropna()
    
    close = bars['close']
    high = bars['high']
    low = bars['low']
    
    ind = pd.DataFrame(index=bars.index)
    
    for lb, label in [(4, '1m'), (19, '5m'), (56, '15m'), (225, '1h'), (450, '2h')]:
        ind[f'ret_{label}'] = close.pct_change(lb) * 100
    
    ret1 = close.pct_change() * 100
    for w, label in [(19, '5m'), (56, '15m'), (225, '1h')]:
        ind[f'vol_{label}'] = ret1.rolling(w).std()
    
    for w, label in [(56, '15m'), (225, '1h'), (450, '2h')]:
        roll_min = close.rolling(w).min(); roll_max = close.rolling(w).max()
        ind[f'chan_{label}'] = (close - roll_min) / (roll_max - roll_min).replace(0, np.nan)
    
    for p, label in [(56, '15m'), (225, '1h')]:
        ind[f'rsi_{label}'] = compute_rsi(close, p)
    
    for p, label in [(225, '1h')]:
        sma = close.rolling(p).mean(); std = close.rolling(p).std()
        ind[f'bb_{label}'] = (close - (sma - 2*std)) / ((sma + 2*std) - (sma - 2*std)).replace(0, np.nan)
    
    ind['atr_pct_1h'] = compute_atr(high, low, close, 225) / close * 100
    
    for (f, s, sig, label) in [(56, 225, 38, '15m')]:
        ef = close.ewm(span=f, adjust=False).mean()
        es = close.ewm(span=s, adjust=False).mean()
        ind[f'macd_{label}'] = (ef - es) - (ef - es).ewm(span=sig, adjust=False).mean()
    
    ind['speed_ratio'] = (close.diff(19).abs() / 19) / (close.diff(56).abs() / 56).replace(0, np.nan)
    ind['hour'] = bars.index.hour
    
    return bars, ind, close

def simulate_short_targets(close_arr, n, max_bars=675):
    """Create SHORT win/loss labels for training."""
    sw = np.zeros(n, dtype=int)
    for i in range(n - max_bars):
        entry = close_arr[i]
        sl = entry * (1 + SL_PCT / 100)
        ta = entry * (1 - TS_ACT_PCT / 100)
        pk = entry; ts_on = False
        for j in range(i+1, min(i+max_bars+1, n)):
            p = close_arr[j]
            if p >= sl: break
            if p < pk: pk = p
            if not ts_on and p <= ta: ts_on = True
            if ts_on:
                tr = pk * (1 + TS_TRAIL_PCT / 100)
                if p >= tr: sw[i] = 1; break
    return sw

# ===================================================================
# TRAIN MODEL
# ===================================================================
log("=" * 60)
log("SHORT MODEL MONITOR — Starting")
log("=" * 60)

log("Loading data and training model...")
bars, ind, close = load_and_prepare()
valid = ind.dropna()

close_arr = close.reindex(valid.index).values
sw = simulate_short_targets(close_arr, len(close_arr))
short_s = pd.Series(sw, index=valid.index)

# Train on all data except last day (use last day as warm-up)
split = valid.index[-1] - pd.Timedelta(days=1)
train_mask = np.array(valid.index < split)

log(f"Training on {train_mask.sum()} bars")
model = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=30, random_state=42)
model.fit(valid[train_mask][C_COMBINED], short_s[train_mask])
log(f"Model trained. SHORT WR in training: {short_s[train_mask].mean()*100:.1f}%")

# ===================================================================
# MONITOR LOOP
# ===================================================================
log(f"\nMonitoring started. Config: SHORT, 00-08h, threshold>{SIGNAL_THRESHOLD:.0%}")
log(f"SL={SL_PCT}%, TS activation={TS_ACT_PCT}%, TS trail={TS_TRAIL_PCT}%")
log(f"Min gap between signals: {MIN_GAP_SEC}s")

active_trade = None  # {entry_price, entry_time, peak, ts_active}
last_signal_time = None
signals = []
completed_trades = []

while True:
    try:
        now = datetime.datetime.now()
        hour = now.hour
        
        # Reload latest ticks
        raw = pd.read_csv(TICK_FILE, parse_dates=['timestamp'])
        raw = raw.set_index('timestamp').sort_index()
        raw = raw[raw['price'] > 0]
        
        if len(raw) < 500:
            time.sleep(CHECK_INTERVAL)
            continue
        
        current_price = raw['price'].iloc[-1]
        
        # Manage active trade
        if active_trade is not None:
            entry = active_trade['entry_price']
            sl = entry * (1 + SL_PCT / 100)
            
            # Update peak (lowest for short)
            if current_price < active_trade['peak']:
                active_trade['peak'] = current_price
            
            # Check SL
            if current_price >= sl:
                pnl_pct = (entry / current_price - 1) * 100
                log(f"  ❌ SL HIT: entry=${entry:.2f}, exit=${current_price:.2f}, "
                    f"pnl={pnl_pct:+.3f}%")
                completed_trades.append({
                    'entry_time': active_trade['entry_time'],
                    'exit_time': now.isoformat(),
                    'entry_price': entry,
                    'exit_price': current_price,
                    'pnl_pct': pnl_pct,
                    'result': 'LOSS'
                })
                active_trade = None
            else:
                # Check trailing stop
                ta = entry * (1 - TS_ACT_PCT / 100)
                if not active_trade['ts_active'] and current_price <= ta:
                    active_trade['ts_active'] = True
                    log(f"  ⚡ TS ACTIVATED at ${current_price:.2f}")
                
                if active_trade['ts_active']:
                    trail = active_trade['peak'] * (1 + TS_TRAIL_PCT / 100)
                    if current_price >= trail:
                        pnl_pct = (entry / current_price - 1) * 100
                        log(f"  ✅ TS EXIT: entry=${entry:.2f}, peak=${active_trade['peak']:.2f}, "
                            f"exit=${current_price:.2f}, pnl={pnl_pct:+.3f}%")
                        completed_trades.append({
                            'entry_time': active_trade['entry_time'],
                            'exit_time': now.isoformat(),
                            'entry_price': entry,
                            'exit_price': current_price,
                            'pnl_pct': pnl_pct,
                            'result': 'WIN'
                        })
                        active_trade = None
        
        # Check for new signal (only during target hours, no active trade)
        if active_trade is None and HOUR_START <= hour < HOUR_END:
            # Resample recent ticks to 16-sec bars
            recent_bars = raw['price'].resample('16s').agg(
                open='first', high='max', low='min', close='last'
            ).dropna()
            
            if len(recent_bars) < 500:
                time.sleep(CHECK_INTERVAL)
                continue
            
            rc = recent_bars['close']
            rh = recent_bars['high']
            rl = recent_bars['low']
            
            # Compute features for latest bar
            feat = {}
            for lb, label in [(4, '1m'), (19, '5m'), (56, '15m'), (225, '1h'), (450, '2h')]:
                if len(rc) > lb:
                    feat[f'ret_{label}'] = (rc.iloc[-1] / rc.iloc[-1-lb] - 1) * 100
            
            ret1 = rc.pct_change() * 100
            for w, label in [(19, '5m'), (56, '15m'), (225, '1h')]:
                if len(ret1) > w:
                    feat[f'vol_{label}'] = ret1.iloc[-w:].std()
            
            for w, label in [(56, '15m'), (225, '1h'), (450, '2h')]:
                if len(rc) > w:
                    rmin = rc.iloc[-w:].min()
                    rmax = rc.iloc[-w:].max()
                    rng = rmax - rmin
                    feat[f'chan_{label}'] = (rc.iloc[-1] - rmin) / rng if rng > 0 else 0.5
            
            if len(rc) > 225:
                rsi_s = compute_rsi(rc, 225)
                feat['rsi_1h'] = rsi_s.iloc[-1]
            if len(rc) > 56:
                rsi_s = compute_rsi(rc, 56)
                feat['rsi_15m'] = rsi_s.iloc[-1]
            
            if len(rc) > 225:
                sma = rc.rolling(225).mean()
                std = rc.rolling(225).std()
                bb = (rc - (sma - 2*std)) / ((sma + 2*std) - (sma - 2*std))
                feat['bb_1h'] = bb.iloc[-1]
            
            if len(rc) > 225:
                atr = compute_atr(rh, rl, rc, 225)
                feat['atr_pct_1h'] = (atr.iloc[-1] / rc.iloc[-1]) * 100
            
            if len(rc) > 225:
                ef = rc.ewm(span=56, adjust=False).mean()
                es = rc.ewm(span=225, adjust=False).mean()
                macd = (ef - es) - (ef - es).ewm(span=38, adjust=False).mean()
                feat['macd_15m'] = macd.iloc[-1]
            
            if len(rc) > 56:
                s1 = abs(rc.iloc[-1] - rc.iloc[-20]) / 19 if len(rc) > 20 else 0
                s2 = abs(rc.iloc[-1] - rc.iloc[-57]) / 56 if len(rc) > 57 else 1
                feat['speed_ratio'] = s1 / s2 if s2 > 0 else 0
            
            feat['hour'] = hour
            
            # Check all features present
            missing = [f for f in C_COMBINED if f not in feat]
            if missing:
                time.sleep(CHECK_INTERVAL)
                continue
            
            # Predict
            X = pd.DataFrame([{f: feat[f] for f in C_COMBINED}])
            prob = model.predict_proba(X)[0][1]
            
            # Check gap
            gap_ok = (last_signal_time is None or 
                     (now - last_signal_time).total_seconds() >= MIN_GAP_SEC)
            
            if prob >= SIGNAL_THRESHOLD and gap_ok:
                log(f"🔴 SHORT SIGNAL: prob={prob:.1%}, price=${current_price:.2f}, "
                    f"rsi_1h={feat.get('rsi_1h',0):.1f}, ret_1h={feat.get('ret_1h',0):+.3f}%")
                active_trade = {
                    'entry_price': current_price,
                    'entry_time': now.isoformat(),
                    'peak': current_price,
                    'ts_active': False,
                }
                last_signal_time = now
                signals.append({'time': now.isoformat(), 'price': current_price, 'prob': prob})
        
        # Periodic summary
        if now.minute == 0 and now.second < CHECK_INTERVAL + 5:
            w = sum(1 for t in completed_trades if t['result'] == 'WIN')
            l = sum(1 for t in completed_trades if t['result'] == 'LOSS')
            total_pnl = sum(t['pnl_pct'] for t in completed_trades)
            status = f"active={active_trade is not None}" if active_trade else "no position"
            log(f"HOURLY: BTC=${current_price:.2f}, signals={len(signals)}, "
                f"trades={w}W/{l}L, pnl={total_pnl:+.3f}%, {status}")
        
        time.sleep(CHECK_INTERVAL)
        
    except KeyboardInterrupt:
        log("Monitor stopped by user")
        break
    except Exception as e:
        log(f"ERROR: {e}")
        time.sleep(30)

# Final summary
log("\n" + "=" * 60)
log("FINAL SUMMARY")
log("=" * 60)
w = sum(1 for t in completed_trades if t['result'] == 'WIN')
l = sum(1 for t in completed_trades if t['result'] == 'LOSS')
total_pnl = sum(t['pnl_pct'] for t in completed_trades)
log(f"Signals: {len(signals)}")
log(f"Trades: {w}W/{l}L")
log(f"Total PnL: {total_pnl:+.3f}%")
for t in completed_trades:
    log(f"  {t['entry_time']} -> {t['exit_time']}: ${t['entry_price']:.2f} -> ${t['exit_price']:.2f}, {t['pnl_pct']:+.3f}% ({t['result']})")
