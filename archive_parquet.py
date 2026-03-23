#!/usr/bin/env python3
"""
Daily feature archiver — builds and preserves 16-sec bar features for training.

Reads raw 2-sec tick data from logs/btc_price_ticks.csv, resamples to 16-sec
OHLC bars, computes the same features the tick bot uses, merges enriched 5-min
parquet features (volume, momentum, trend indicators), and appends to a growing
archive file.

Archives:
  data/archive/tick_features_archive.parquet  — 16-sec features (growing)
  data/archive/BTC_features_archive.parquet   — 5-min features (growing)
  data/archive/BTC_features_YYYY-MM-DD.parquet — 5-min daily snapshots (7-day)

Runs daily via cron at midnight.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TICK_CSV = os.path.join(BASE_DIR, 'logs', 'btc_price_ticks.csv')
PARQUET_5MIN = os.path.join(BASE_DIR, 'data', 'BTC_features.parquet')
ARCHIVE_DIR = os.path.join(BASE_DIR, 'data', 'archive')
TICK_ARCHIVE = os.path.join(ARCHIVE_DIR, 'tick_features_archive.parquet')
BARS_ARCHIVE = os.path.join(ARCHIVE_DIR, 'tick_bars_16sec.parquet')
RAW_TICKS_ARCHIVE = os.path.join(ARCHIVE_DIR, 'raw_ticks_2sec.parquet')
PARQUET_ARCHIVE = os.path.join(ARCHIVE_DIR, 'BTC_features_archive.parquet')

# Import feature computation from tick bot
sys.path.insert(0, BASE_DIR)
from btc_tick_bot import compute_features, ENRICHED_COLS


def load_ticks():
    """Load raw 2-sec ticks from CSV."""
    if not os.path.exists(TICK_CSV):
        print("No tick data at %s" % TICK_CSV)
        return None
    ticks = pd.read_csv(TICK_CSV, parse_dates=['timestamp'])
    ticks = ticks.set_index('timestamp').sort_index()
    print("Raw ticks: %d rows, %s to %s" % (len(ticks), ticks.index[0], ticks.index[-1]))
    return ticks


def archive_raw_ticks(ticks):
    """Archive raw 2-sec ticks in compressed parquet (much smaller than CSV)."""
    if ticks is None:
        return

    if os.path.exists(RAW_TICKS_ARCHIVE):
        existing = pd.read_parquet(RAW_TICKS_ARCHIVE)
        print("Existing raw ticks archive: %d rows" % len(existing))
        combined = pd.concat([existing, ticks])
        combined = combined[~combined.index.duplicated(keep='last')]
        combined = combined.sort_index()
    else:
        combined = ticks
        print("No existing raw ticks archive — creating new one")

    combined.to_parquet(RAW_TICKS_ARCHIVE, compression='snappy')
    print("Raw ticks archive: %d rows (%.1f MB)" % (
        len(combined), os.path.getsize(RAW_TICKS_ARCHIVE) / 1024 / 1024))


def archive_16sec_bars(ticks):
    """Archive raw 16-sec OHLC bars — can be resampled to 32s/48s/64s for training."""
    if ticks is None:
        return None

    bars = ticks['price'].resample('16s').agg(
        **{'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
    ).dropna()
    print("16-sec bars: %d" % len(bars))

    if os.path.exists(BARS_ARCHIVE):
        existing = pd.read_parquet(BARS_ARCHIVE)
        print("Existing bars archive: %d rows" % len(existing))
        combined = pd.concat([existing, bars])
        combined = combined[~combined.index.duplicated(keep='last')]
        combined = combined.sort_index()
    else:
        combined = bars
        print("No existing bars archive — creating new one")

    combined.to_parquet(BARS_ARCHIVE)
    print("16-sec bars archive: %d rows, %s to %s (%.1f MB)" % (
        len(combined), combined.index[0], combined.index[-1],
        os.path.getsize(BARS_ARCHIVE) / 1024 / 1024))
    return bars


def build_16sec_features(bars):
    """Compute features from 16-sec bars + merge enriched 5-min parquet."""
    if bars is None or len(bars) < 500:
        print("Not enough bars for features: %d" % (len(bars) if bars is not None else 0))
        return None

    feat = compute_features(bars)
    print("Tick features: %d cols" % len(feat.columns))

    # Merge enriched 5-min parquet features
    if os.path.exists(PARQUET_5MIN):
        pq = pd.read_parquet(PARQUET_5MIN)
        available = [c for c in ENRICHED_COLS if c in pq.columns]
        if available:
            enriched = pq[available].reindex(feat.index, method='ffill')
            for col in enriched.columns:
                feat['ext_' + col] = enriched[col]
            print("Enriched features merged: %d cols from 5-min parquet" % len(available))

    feat = feat.replace([np.inf, -np.inf], np.nan)
    print("Final features: %d rows x %d cols" % feat.shape)
    return feat


def archive_tick_features(feat):
    """Append new 16-sec features to the growing archive."""
    if feat is None or len(feat) == 0:
        return

    if os.path.exists(TICK_ARCHIVE):
        existing = pd.read_parquet(TICK_ARCHIVE)
        print("Existing tick features archive: %d rows, %s to %s" % (
            len(existing), existing.index[0], existing.index[-1]))
        combined = pd.concat([existing, feat])
        combined = combined[~combined.index.duplicated(keep='last')]
        combined = combined.sort_index()
    else:
        combined = feat
        print("No existing tick features archive — creating new one")

    combined.to_parquet(TICK_ARCHIVE)
    print("Tick features archive: %d rows, %s to %s (%.1f MB)" % (
        len(combined), combined.index[0], combined.index[-1],
        os.path.getsize(TICK_ARCHIVE) / 1024 / 1024))


def archive_5min_parquet():
    """Also archive the 5-min parquet for ensemble bot training."""
    if not os.path.exists(PARQUET_5MIN):
        print("No 5-min parquet at %s" % PARQUET_5MIN)
        return

    live = pd.read_parquet(PARQUET_5MIN)
    print("\n5-min parquet: %d rows, %s to %s" % (len(live), live.index[0], live.index[-1]))

    if os.path.exists(PARQUET_ARCHIVE):
        archive_df = pd.read_parquet(PARQUET_ARCHIVE)
        combined = pd.concat([archive_df, live])
        combined = combined[~combined.index.duplicated(keep='last')]
        combined = combined.sort_index()
    else:
        combined = live

    combined.to_parquet(PARQUET_ARCHIVE)
    print("5-min archive saved: %d rows, %s to %s (%.1f MB)" % (
        len(combined), combined.index[0], combined.index[-1],
        os.path.getsize(PARQUET_ARCHIVE) / 1024 / 1024))

    # Daily snapshot (keep last 7)
    today = datetime.now().strftime('%Y-%m-%d')
    snapshot = os.path.join(ARCHIVE_DIR, 'BTC_features_%s.parquet' % today)
    live.to_parquet(snapshot)

    snapshots = sorted([f for f in os.listdir(ARCHIVE_DIR)
                       if f.startswith('BTC_features_2') and f.endswith('.parquet')])
    for old in snapshots[:-7]:
        os.remove(os.path.join(ARCHIVE_DIR, old))
        print("Removed old snapshot: %s" % old)


def archive():
    os.makedirs(ARCHIVE_DIR, exist_ok=True)

    # 1. Load raw 2-sec ticks
    print("=" * 60)
    print("LOADING RAW TICKS")
    print("=" * 60)
    ticks = load_ticks()

    # 2. Archive raw 2-sec ticks in compressed parquet
    print("\n" + "=" * 60)
    print("ARCHIVING RAW 2-SEC TICKS")
    print("=" * 60)
    archive_raw_ticks(ticks)

    # 3. Archive 16-sec OHLC bars (can resample to 32s/48s/64s for training)
    print("\n" + "=" * 60)
    print("ARCHIVING 16-SEC OHLC BARS")
    print("=" * 60)
    bars = archive_16sec_bars(ticks)

    # 4. Build and archive 16-sec features
    print("\n" + "=" * 60)
    print("ARCHIVING 16-SEC TICK FEATURES")
    print("=" * 60)
    feat = build_16sec_features(bars)
    archive_tick_features(feat)

    # 5. Archive 5-min parquet (for ensemble bot training)
    print("\n" + "=" * 60)
    print("ARCHIVING 5-MIN PARQUET FEATURES")
    print("=" * 60)
    archive_5min_parquet()


if __name__ == '__main__':
    archive()
