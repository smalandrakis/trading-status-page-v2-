#!/usr/bin/env python3
"""
Validate ensemble bot: for each actual trade, check that signal logs show
the model probability was above threshold at entry time.
This is the proper validation — same data source the bot used for decisions.
"""

import os, sqlite3
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))

# Load actual trades
conn = sqlite3.connect(os.path.join(BASE, 'trades.db'))
trades = pd.read_sql("""
    SELECT entry_time, exit_time, model_id, direction, entry_price, exit_price,
           pnl_dollar, exit_reason
    FROM trades WHERE model_id NOT LIKE 'legacy%' AND entry_time >= '2026-03-10'
    ORDER BY entry_time
""", conn)
conn.close()

# Load all signal logs for the period
sig_frames = []
for f in sorted(os.listdir(os.path.join(BASE, 'signal_logs'))):
    if f.startswith('btc_signals_2026-03-1') and f.endswith('.csv'):
        path = os.path.join(BASE, 'signal_logs', f)
        try:
            df = pd.read_csv(path, parse_dates=['timestamp'])
            sig_frames.append(df)
        except:
            pass

sigs = pd.concat(sig_frames, ignore_index=True).sort_values('timestamp')
sigs = sigs.set_index('timestamp')
print(f"Signal logs: {len(sigs)} rows, {sigs.index[0]} to {sigs.index[-1]}")
print(f"Actual trades: {len(trades)}")

# Model → signal log column mapping
model_prob_col = {
    '2h_0.5pct': 'prob_2h_0.5pct',
    '4h_0.5pct': 'prob_4h_0.5pct',
    '6h_0.5pct': 'prob_6h_0.5pct',
    '2h_0.5pct_SHORT': 'prob_2h_0.5pct_SHORT',
    '4h_0.5pct_SHORT': 'prob_4h_0.5pct_SHORT',
}

PROB_THRESH = 0.55  # Both LONG and SHORT

print(f"\n{'='*100}")
print(f"TRADE-BY-TRADE SIGNAL VALIDATION")
print(f"{'='*100}")
print(f"{'Time':>16s} | {'Model':>20s} | {'Dir':>5s} | {'Entry$':>10s} | {'Sig$':>10s} | {'Prob':>6s} | {'RSI':>5s} | {'MACD':>6s} | {'BB':>5s} | {'PnL':>8s} | {'Valid':>5s}")
print("-"*120)

matched = 0
valid = 0
mismatched_price = []
missing_signal = []

for _, t in trades.iterrows():
    entry_ts = pd.Timestamp(t['entry_time'])
    model = t['model_id']
    direction = t['direction']
    
    # Find closest signal log entry within 60 seconds of trade entry
    time_mask = (sigs.index >= entry_ts - pd.Timedelta(seconds=60)) & \
                (sigs.index <= entry_ts + pd.Timedelta(seconds=60))
    nearby = sigs[time_mask]
    
    if len(nearby) == 0:
        # Try wider window
        time_mask = (sigs.index >= entry_ts - pd.Timedelta(minutes=5)) & \
                    (sigs.index <= entry_ts + pd.Timedelta(minutes=5))
        nearby = sigs[time_mask]
    
    if len(nearby) == 0:
        missing_signal.append(t)
        ts_str = str(entry_ts)[:16]
        print(f"{ts_str:>16s} | {model:>20s} | {direction:>5s} | ${t['entry_price']:>9.2f} | {'N/A':>10s} | {'N/A':>6s} | {'N/A':>5s} | {'N/A':>6s} | {'N/A':>5s} | ${t['pnl_dollar']:>+7.2f} | {'MISS':>5s}")
        continue
    
    # Find the row closest in time
    idx = (nearby.index - entry_ts).map(lambda x: abs(x.total_seconds())).argmin()
    row = nearby.iloc[idx]
    sig_time = nearby.index[idx]
    
    prob_col = model_prob_col.get(model)
    prob = row.get(prob_col, np.nan) if prob_col else np.nan
    sig_price = row.get('btc_price', np.nan)
    rsi = row.get('rsi', np.nan)
    macd = row.get('macd', np.nan)
    bb = row.get('bb_pct_b', np.nan)
    
    # Validate: prob should be >= threshold
    prob_valid = not np.isnan(prob) and prob >= PROB_THRESH
    price_diff = abs(sig_price - t['entry_price']) if not np.isnan(sig_price) else float('inf')
    
    ts_str = str(entry_ts)[:16]
    status = "✓" if prob_valid else "✗"
    
    print(f"{ts_str:>16s} | {model:>20s} | {direction:>5s} | ${t['entry_price']:>9.2f} | ${sig_price:>9.2f} | {prob*100:>5.1f}% | {rsi:>5.1f} | {macd:>6.1f} | {bb:>5.3f} | ${t['pnl_dollar']:>+7.2f} | {status:>5s}")
    
    matched += 1
    if prob_valid:
        valid += 1
    if price_diff > 200:
        mismatched_price.append((t, price_diff))

print(f"\n{'='*100}")
print(f"SUMMARY")
print(f"{'='*100}")
print(f"Trades checked: {len(trades)}")
print(f"Signal log match: {matched}/{len(trades)} ({100*matched/len(trades):.0f}%)")
print(f"Probability valid (>= {PROB_THRESH*100:.0f}%): {valid}/{matched} ({100*valid/matched:.0f}%)" if matched > 0 else "")
print(f"Missing from signal logs: {len(missing_signal)}")

if mismatched_price:
    print(f"\nPrice mismatches > $200 ({len(mismatched_price)}):")
    for t, diff in mismatched_price:
        print(f"  {str(t['entry_time'])[:16]} | {t['model_id']} | Δ${diff:.2f}")

# Check filter context for losing trades
print(f"\n{'='*100}")
print(f"LOSING TRADE FILTER CONTEXT (would new filters have helped?)")
print(f"{'='*100}")
losers = trades[trades['pnl_dollar'] < 0]
for _, t in losers.iterrows():
    entry_ts = pd.Timestamp(t['entry_time'])
    time_mask = (sigs.index >= entry_ts - pd.Timedelta(minutes=5)) & \
                (sigs.index <= entry_ts + pd.Timedelta(minutes=5))
    nearby = sigs[time_mask]
    if len(nearby) == 0:
        continue
    idx = (nearby.index - entry_ts).map(lambda x: abs(x.total_seconds())).argmin()
    row = nearby.iloc[idx]
    
    rsi = row.get('rsi', np.nan)
    macd = row.get('macd', np.nan)
    bb = row.get('bb_pct_b', np.nan)
    
    blocks = []
    if t['direction'] == 'SHORT' and not np.isnan(rsi) and rsi < 40:
        blocks.append(f"RSI<40 ({rsi:.1f})")
    if t['direction'] == 'LONG' and not np.isnan(rsi) and rsi > 70:
        blocks.append(f"RSI>70 ({rsi:.1f})")
    if t['model_id'] == '6h_0.5pct' and t['direction'] == 'LONG':
        p2h = row.get('prob_2h_0.5pct', np.nan)
        p4h = row.get('prob_4h_0.5pct', np.nan)
        if not np.isnan(p2h) and not np.isnan(p4h) and p2h < 0.40 and p4h < 0.40:
            blocks.append(f"CROSS-TF (2h={p2h*100:.0f}%,4h={p4h*100:.0f}%)")
    
    block_str = " → WOULD BLOCK" if blocks else ""
    block_detail = f" [{', '.join(blocks)}]" if blocks else ""
    ts_str = str(entry_ts)[:16]
    print(f"  {ts_str} | {t['model_id']:20s} | ${t['pnl_dollar']:+.2f} | RSI={rsi:.1f} MACD={macd:.1f} BB={bb:.3f}{block_str}{block_detail}")
