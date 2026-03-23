#!/usr/bin/env python3
"""
Backfill missing trade metrics from signal logs and parquet data.
1. Fills NULL entry_rsi, entry_macd, entry_bb_position, entry_probability from signal logs
2. Adds new columns: entry_trend_1h, entry_macro_trend_24h, entry_prob_2h, entry_prob_4h,
   entry_prob_6h, entry_prob_2h_short, entry_prob_4h_short, max_favorable_excursion,
   max_adverse_excursion
3. Backfills new columns from signal logs and parquet price data
"""

import os, sys, json, sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, 'trades.db')

# ── Step 1: Add new columns if they don't exist ─────────────────────────
NEW_COLUMNS = [
    ("entry_trend_1h", "REAL"),
    ("entry_macro_trend_24h", "REAL"),
    ("entry_prob_2h", "REAL"),
    ("entry_prob_4h", "REAL"),
    ("entry_prob_6h", "REAL"),
    ("entry_prob_2h_short", "REAL"),
    ("entry_prob_4h_short", "REAL"),
    ("max_favorable_excursion", "REAL"),   # max % in profit direction
    ("max_adverse_excursion", "REAL"),     # max % against trade
]

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Get existing columns
cursor.execute("PRAGMA table_info(trades)")
existing_cols = {row[1] for row in cursor.fetchall()}

added = 0
for col_name, col_type in NEW_COLUMNS:
    if col_name not in existing_cols:
        cursor.execute(f"ALTER TABLE trades ADD COLUMN {col_name} {col_type}")
        added += 1
        print(f"  Added column: {col_name} ({col_type})")

if added:
    conn.commit()
    print(f"Added {added} new columns to trades table")
else:
    print("All columns already exist")

# ── Step 2: Load all signal logs ─────────────────────────────────────────
sig_dir = os.path.join(BASE, 'signal_logs')
sig_frames = []
for f in sorted(os.listdir(sig_dir)):
    if f.startswith('btc_signals_') and f.endswith('.csv'):
        path = os.path.join(sig_dir, f)
        try:
            df = pd.read_csv(path, parse_dates=['timestamp'])
            sig_frames.append(df)
        except Exception as e:
            print(f"  Warning: could not load {f}: {e}")

if sig_frames:
    sigs = pd.concat(sig_frames, ignore_index=True).sort_values('timestamp')
    sigs = sigs.set_index('timestamp')
    print(f"\nSignal logs: {len(sigs)} rows, {sigs.index[0]} to {sigs.index[-1]}")
else:
    print("No signal logs found!")
    sys.exit(1)

# ── Step 3: Load parquet for trend/MFE/MAE calculations ─────────────────
archive_path = os.path.join(BASE, 'data/archive/BTC_features_archive.parquet')
live_path = os.path.join(BASE, 'data/BTC_features.parquet')

pq = None
if os.path.exists(archive_path):
    pq = pd.read_parquet(archive_path)
    # Also load live and merge
    if os.path.exists(live_path):
        pq_live = pd.read_parquet(live_path)
        pq = pd.concat([pq, pq_live[~pq_live.index.isin(pq.index)]]).sort_index()
elif os.path.exists(live_path):
    pq = pd.read_parquet(live_path)

if pq is not None:
    print(f"Parquet: {len(pq)} bars, {pq.index[0]} to {pq.index[-1]}")
    # Ensure we have Close column
    if 'Close' not in pq.columns and 'close' in pq.columns:
        pq['Close'] = pq['close']
else:
    print("Warning: No parquet data for trend/MFE/MAE calculations")

# ── Step 4: Load trades ─────────────────────────────────────────────────
trades = pd.read_sql("SELECT * FROM trades WHERE bot_type='BTC' ORDER BY entry_time", conn)
print(f"\nTotal trades: {len(trades)}")
print(f"  Missing RSI: {trades['entry_rsi'].isna().sum()}")
print(f"  Missing probability: {(trades['entry_probability'].isna() | (trades['entry_probability'] == 0)).sum()}")
print(f"  Missing trend_1h: {trades['entry_trend_1h'].isna().sum()}")
print(f"  Missing MFE: {trades['max_favorable_excursion'].isna().sum()}")

# ── Step 5: Backfill from signal logs ────────────────────────────────────
def find_nearest_signal(entry_time_str, max_seconds=300):
    """Find nearest signal log entry within max_seconds of trade entry."""
    try:
        entry_ts = pd.Timestamp(entry_time_str)
    except:
        return None
    
    mask = (sigs.index >= entry_ts - pd.Timedelta(seconds=max_seconds)) & \
           (sigs.index <= entry_ts + pd.Timedelta(seconds=max_seconds))
    nearby = sigs[mask]
    
    if len(nearby) == 0:
        return None
    
    idx = (nearby.index - entry_ts).map(lambda x: abs(x.total_seconds())).argmin()
    return nearby.iloc[idx]


def compute_trend_1h(entry_time_str):
    """Compute 1h trend % from parquet at entry time."""
    if pq is None:
        return None
    try:
        entry_ts = pd.Timestamp(entry_time_str)
        # Find closest bar
        idx = pq.index.searchsorted(entry_ts)
        if idx < 12 or idx >= len(pq):
            return None
        # 1h = 12 bars of 5-min
        close_now = pq['Close'].iloc[min(idx, len(pq)-1)]
        close_1h_ago = pq['Close'].iloc[idx - 12]
        if close_1h_ago > 0:
            return (close_now / close_1h_ago - 1) * 100
    except:
        pass
    return None


def compute_macro_trend_24h(entry_time_str):
    """Compute 24h trend % from parquet at entry time."""
    if pq is None:
        return None
    try:
        entry_ts = pd.Timestamp(entry_time_str)
        idx = pq.index.searchsorted(entry_ts)
        if idx < 288 or idx >= len(pq):
            return None
        close_now = pq['Close'].iloc[min(idx, len(pq)-1)]
        close_24h_ago = pq['Close'].iloc[idx - 288]
        if close_24h_ago > 0:
            return (close_now / close_24h_ago - 1) * 100
    except:
        pass
    return None


def compute_mfe_mae(entry_time_str, exit_time_str, entry_price, direction):
    """Compute max favorable and adverse excursion from parquet."""
    if pq is None:
        return None, None
    try:
        entry_ts = pd.Timestamp(entry_time_str)
        exit_ts = pd.Timestamp(exit_time_str)
        
        mask = (pq.index >= entry_ts) & (pq.index <= exit_ts)
        bars = pq[mask]
        
        if len(bars) < 1 or entry_price <= 0:
            return None, None
        
        if 'High' in bars.columns and 'Low' in bars.columns:
            highs = bars['High']
            lows = bars['Low']
        else:
            highs = bars['Close']
            lows = bars['Close']
        
        if direction == 'LONG':
            mfe = (highs.max() / entry_price - 1) * 100  # max up
            mae = (lows.min() / entry_price - 1) * 100   # max down (negative)
        else:  # SHORT
            mfe = (1 - lows.min() / entry_price) * 100    # max down = profit
            mae = (1 - highs.max() / entry_price) * 100   # max up = adverse (negative)
        
        return round(mfe, 4), round(mae, 4)
    except:
        return None, None


# Process each trade
updated = 0
backfilled_rsi = 0
backfilled_prob = 0
backfilled_trend = 0
backfilled_mfe = 0

for _, trade in trades.iterrows():
    trade_id = trade['id']
    updates = {}
    
    # Find nearest signal log
    sig_row = find_nearest_signal(trade['entry_time'])
    
    # Backfill basic indicators if missing
    if sig_row is not None:
        if pd.isna(trade['entry_rsi']) and 'rsi' in sig_row.index:
            rsi_val = sig_row['rsi']
            if not np.isnan(rsi_val):
                updates['entry_rsi'] = round(float(rsi_val), 2)
                backfilled_rsi += 1
        
        if pd.isna(trade['entry_macd']) and 'macd' in sig_row.index:
            macd_val = sig_row['macd']
            if not np.isnan(macd_val):
                updates['entry_macd'] = round(float(macd_val), 4)
        
        if pd.isna(trade['entry_bb_position']) and 'bb_pct_b' in sig_row.index:
            bb_val = sig_row['bb_pct_b']
            if not np.isnan(bb_val):
                updates['entry_bb_position'] = round(float(bb_val), 4)
        
        if pd.isna(trade['entry_atr_pct']) and 'atr' in sig_row.index:
            atr_val = sig_row['atr']
            price = sig_row.get('btc_price', trade['entry_price'])
            if not np.isnan(atr_val) and price > 0:
                updates['entry_atr_pct'] = round(float(atr_val) / price * 100, 4)
        
        if (pd.isna(trade['entry_probability']) or trade['entry_probability'] == 0):
            # Try to get model-specific probability
            model = trade['model_id']
            prob_col_map = {
                '2h_0.5pct': 'prob_2h_0.5pct',
                '4h_0.5pct': 'prob_4h_0.5pct',
                '6h_0.5pct': 'prob_6h_0.5pct',
                '2h_0.5pct_SHORT': 'prob_2h_0.5pct_SHORT',
                '4h_0.5pct_SHORT': 'prob_4h_0.5pct_SHORT',
            }
            col = prob_col_map.get(model)
            if col and col in sig_row.index:
                p = sig_row[col]
                if not np.isnan(p) and p > 0:
                    updates['entry_probability'] = round(float(p), 6)
                    backfilled_prob += 1
        
        # All model probabilities (new columns)
        prob_map = {
            'entry_prob_2h': 'prob_2h_0.5pct',
            'entry_prob_4h': 'prob_4h_0.5pct',
            'entry_prob_6h': 'prob_6h_0.5pct',
            'entry_prob_2h_short': 'prob_2h_0.5pct_SHORT',
            'entry_prob_4h_short': 'prob_4h_0.5pct_SHORT',
        }
        for db_col, sig_col in prob_map.items():
            if pd.isna(trade.get(db_col)) and sig_col in sig_row.index:
                v = sig_row[sig_col]
                if not np.isnan(v):
                    updates[db_col] = round(float(v), 6)
    
    # Backfill trend metrics from parquet
    if pd.isna(trade.get('entry_trend_1h')):
        t1h = compute_trend_1h(trade['entry_time'])
        if t1h is not None:
            updates['entry_trend_1h'] = round(t1h, 4)
            backfilled_trend += 1
    
    if pd.isna(trade.get('entry_macro_trend_24h')):
        t24h = compute_macro_trend_24h(trade['entry_time'])
        if t24h is not None:
            updates['entry_macro_trend_24h'] = round(t24h, 4)
    
    # MFE / MAE from parquet
    if pd.isna(trade.get('max_favorable_excursion')):
        mfe, mae = compute_mfe_mae(
            trade['entry_time'], trade['exit_time'],
            trade['entry_price'], trade['direction']
        )
        if mfe is not None:
            updates['max_favorable_excursion'] = mfe
            updates['max_adverse_excursion'] = mae
            backfilled_mfe += 1
    
    # Hour/day from entry_time if missing
    if pd.isna(trade.get('entry_hour')):
        try:
            et = pd.Timestamp(trade['entry_time'])
            updates['entry_hour'] = et.hour
            updates['entry_day_of_week'] = et.dayofweek
        except:
            pass
    
    # Apply updates
    if updates:
        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [trade_id]
        cursor.execute(f"UPDATE trades SET {set_clause} WHERE id = ?", values)
        updated += 1

conn.commit()

# ── Step 6: Final stats ─────────────────────────────────────────────────
trades2 = pd.read_sql("SELECT * FROM trades WHERE bot_type='BTC'", conn)
conn.close()

print(f"\n{'='*60}")
print(f"BACKFILL COMPLETE")
print(f"{'='*60}")
print(f"Trades updated: {updated}/{len(trades)}")
print(f"  RSI backfilled: {backfilled_rsi}")
print(f"  Probability backfilled: {backfilled_prob}")
print(f"  Trend 1h backfilled: {backfilled_trend}")
print(f"  MFE/MAE backfilled: {backfilled_mfe}")

print(f"\nCoverage after backfill:")
for col in ['entry_rsi', 'entry_macd', 'entry_bb_position', 'entry_probability',
            'entry_trend_1h', 'entry_macro_trend_24h', 'entry_prob_2h',
            'entry_prob_4h', 'entry_prob_6h', 'entry_prob_2h_short',
            'entry_prob_4h_short', 'max_favorable_excursion', 'max_adverse_excursion',
            'entry_hour', 'entry_day_of_week']:
    if col in trades2.columns:
        filled = trades2[col].notna().sum()
        if col == 'entry_probability':
            filled = ((trades2[col].notna()) & (trades2[col] != 0)).sum()
        total = len(trades2)
        pct = 100 * filled / total
        print(f"  {col:30s}: {filled:>4d}/{total} ({pct:5.1f}%)")
