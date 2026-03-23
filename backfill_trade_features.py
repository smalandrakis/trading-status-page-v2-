#!/usr/bin/env python3
"""
Backfill full 211-feature vectors for all historical trades.

Strategy:
1. Fetch historical 5-min OHLCV from Binance for each unique trade date range
2. Compute ta.add_all_ta_features + custom features (same pipeline as bot)
3. Match each trade entry time to nearest 5-min bar
4. Save to data/trade_features.parquet

This gives us the exact same feature vector the model saw at trade time,
enabling post-hoc analysis, filter retraining, and feature importance studies.
"""

import os, sys, json, sqlite3, time
import pandas as pd
import numpy as np
import requests
import ta

# Add project dir to path for imports
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from feature_engineering import (
    add_time_features, add_price_features, 
    add_daily_context_features, add_lagged_indicator_features,
    add_indicator_changes
)

SNAPSHOT_PATH = os.path.join(BASE, 'data', 'trade_features.parquet')
DB_PATH = os.path.join(BASE, 'trades.db')
BINANCE_URL = "https://api.binance.com/api/v3/klines"

# ── Step 1: Load trades ─────────────────────────────────────────────────
conn = sqlite3.connect(DB_PATH)
trades = pd.read_sql("""
    SELECT id, entry_time, model_id, direction, entry_price, entry_probability,
           pnl_dollar, exit_reason
    FROM trades WHERE bot_type='BTC' AND model_id NOT LIKE 'legacy%'
    ORDER BY entry_time
""", conn)
conn.close()
print(f"Total non-legacy trades: {len(trades)}")

# Load existing snapshots to skip already-done trades
existing_times = set()
if os.path.exists(SNAPSHOT_PATH):
    existing = pd.read_parquet(SNAPSHOT_PATH)
    existing_times = set(existing['_trade_time'].values) if '_trade_time' in existing.columns else set()
    print(f"Existing snapshots: {len(existing_times)}")

# ── Step 2: Group trades by date range (fetch data in batches) ──────────
trades['entry_ts'] = pd.to_datetime(trades['entry_time'], format='mixed')
trades = trades.sort_values('entry_ts')

# We need ~2000 bars of history before each trade for stable indicators.
# Fetch in weekly batches, each with 2000-bar lookback.
unique_dates = trades['entry_ts'].dt.date.unique()
print(f"Trade dates span: {unique_dates[0]} to {unique_dates[-1]}")


def fetch_binance_klines(start_ms, end_ms, interval='5m', limit=1000):
    """Fetch klines from Binance API."""
    all_klines = []
    current_start = start_ms
    
    while current_start < end_ms:
        params = {
            'symbol': 'BTCUSDT',
            'interval': interval,
            'startTime': int(current_start),
            'endTime': int(end_ms),
            'limit': limit,
        }
        resp = requests.get(BINANCE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        
        if not data:
            break
        
        all_klines.extend(data)
        # Move start to after last candle
        current_start = data[-1][0] + 1
        
        if len(data) < limit:
            break
        
        time.sleep(0.2)  # Rate limit
    
    return all_klines


def klines_to_df(klines):
    """Convert Binance klines to OHLCV DataFrame."""
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = df[col].astype(float)
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]


def compute_all_features(ohlcv_df):
    """Compute full feature set matching bot pipeline."""
    df = ta.add_all_ta_features(
        ohlcv_df.copy(), open='Open', high='High', low='Low',
        close='Close', volume='Volume', fillna=True
    )
    df = add_time_features(df)
    df = add_price_features(df)
    df = add_daily_context_features(df)
    df = add_lagged_indicator_features(df, lookback_periods=[1, 2, 3, 5, 10, 20, 50])
    df = add_indicator_changes(df)
    df = df.ffill().bfill().fillna(0)
    df = df.replace([np.inf, -np.inf], 0)
    return df


# ── Step 3: Process in weekly batches ────────────────────────────────────
# Group trades into weekly chunks and fetch data for each
snapshots = []
processed = 0
skipped = 0

# Create weekly groups
trades['week'] = trades['entry_ts'].dt.isocalendar().week.astype(int)
trades['year'] = trades['entry_ts'].dt.year

for (year, week), group in trades.groupby(['year', 'week']):
    # Check if all trades in this week are already snapshotted
    group_times = set(group['entry_time'].values)
    if group_times.issubset(existing_times):
        skipped += len(group)
        continue
    
    # Need 2000 bars (7 days) before earliest trade in group for indicator warmup
    earliest = group['entry_ts'].min()
    latest = group['entry_ts'].max()
    
    # Fetch from 8 days before earliest to end of latest trade
    fetch_start = earliest - pd.Timedelta(days=8)
    fetch_end = latest + pd.Timedelta(hours=1)
    
    start_ms = int(fetch_start.timestamp() * 1000)
    end_ms = int(fetch_end.timestamp() * 1000)
    
    print(f"\nWeek {year}-W{week:02d}: {len(group)} trades ({earliest.date()} to {latest.date()})")
    print(f"  Fetching {fetch_start.date()} to {fetch_end.date()} from Binance...")
    
    try:
        klines = fetch_binance_klines(start_ms, end_ms)
        if not klines:
            print(f"  No data from Binance — skipping")
            continue
        
        ohlcv = klines_to_df(klines)
        print(f"  Got {len(ohlcv)} bars ({ohlcv.index[0]} to {ohlcv.index[-1]})")
        
        if len(ohlcv) < 300:
            print(f"  Too few bars for stable indicators — skipping")
            continue
        
        # Compute features
        print(f"  Computing features...")
        features_df = compute_all_features(ohlcv)
        print(f"  {len(features_df)} bars × {len(features_df.columns)} features")
        
        # Match each trade to nearest bar
        for _, trade in group.iterrows():
            if trade['entry_time'] in existing_times:
                skipped += 1
                continue
            
            entry_ts = trade['entry_ts']
            
            # Find nearest bar (within 5 minutes)
            time_diffs = abs(features_df.index - entry_ts)
            nearest_idx = time_diffs.argmin()
            nearest_diff = time_diffs[nearest_idx].total_seconds()
            
            if nearest_diff > 300:  # > 5 minutes away
                print(f"  ⚠ No bar within 5 min for {trade['entry_time']} (nearest: {nearest_diff:.0f}s)")
                continue
            
            row = features_df.iloc[nearest_idx]
            row_dict = row.to_dict()
            
            # Add trade metadata
            row_dict['_trade_time'] = trade['entry_time']
            row_dict['_trade_id'] = int(trade['id'])
            row_dict['_model_id'] = trade['model_id']
            row_dict['_direction'] = trade['direction']
            row_dict['_entry_price'] = trade['entry_price']
            row_dict['_probability'] = trade['entry_probability'] if pd.notna(trade['entry_probability']) else 0
            row_dict['_pnl_dollar'] = trade['pnl_dollar']
            row_dict['_exit_reason'] = trade['exit_reason']
            row_dict['_bar_time'] = str(features_df.index[nearest_idx])
            row_dict['_bar_offset_sec'] = nearest_diff
            
            snapshots.append(row_dict)
            processed += 1
        
    except Exception as e:
        print(f"  ERROR: {e}")
        continue
    
    # Brief pause between weeks
    time.sleep(1)

# ── Step 4: Save results ────────────────────────────────────────────────
if snapshots:
    new_df = pd.DataFrame(snapshots)
    
    # Merge with existing
    if os.path.exists(SNAPSHOT_PATH):
        existing = pd.read_parquet(SNAPSHOT_PATH)
        # Avoid duplicates
        if '_trade_time' in existing.columns:
            existing_set = set(existing['_trade_time'].values)
            new_df = new_df[~new_df['_trade_time'].isin(existing_set)]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    
    os.makedirs(os.path.dirname(SNAPSHOT_PATH), exist_ok=True)
    combined.to_parquet(SNAPSHOT_PATH, index=False)
    print(f"\n{'='*60}")
    print(f"SAVED: {len(combined)} total trade feature snapshots")
    print(f"  New: {len(new_df)}")
    print(f"  Columns: {len(combined.columns)}")
    print(f"  File: {SNAPSHOT_PATH}")
else:
    print("\nNo new snapshots to save")

print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"Total trades: {len(trades)}")
print(f"Processed: {processed}")
print(f"Skipped (already done): {skipped}")
print(f"Missing: {len(trades) - processed - skipped}")
