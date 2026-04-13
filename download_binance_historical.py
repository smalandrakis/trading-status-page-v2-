"""
Download Historical BTC/USDT Data from Binance

Downloads 3 years (2022-2024) of 1-minute candles from Binance public API.
Combines with existing 2025-2026 data to create a 4-year dataset for V3 models.
"""

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from pathlib import Path

# Configuration
SYMBOL = 'BTCUSDT'
INTERVAL = '1m'
START_DATE = '2022-01-01'
END_DATE = '2024-12-31'
BINANCE_API = 'https://api.binance.com/api/v3/klines'


def get_binance_data(symbol, interval, start_time, end_time, limit=1000):
    """
    Fetch klines from Binance API

    Parameters:
    - start_time: Unix timestamp in milliseconds
    - end_time: Unix timestamp in milliseconds
    - limit: Max 1000 candles per request
    """
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': start_time,
        'endTime': end_time,
        'limit': limit
    }

    try:
        response = requests.get(BINANCE_API, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"  Error fetching data: {e}")
        return None


def download_full_history(symbol, interval, start_date, end_date):
    """
    Download full historical data in chunks

    Binance limits to 1000 candles per request, so we need to paginate.
    """
    print(f"Downloading {symbol} {interval} data from {start_date} to {end_date}")

    # Convert dates to timestamps
    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)

    all_data = []
    current_start = start_ts

    # Calculate expected number of candles (rough estimate)
    days = (end_ts - start_ts) / (1000 * 60 * 60 * 24)
    expected_candles = int(days * 24 * 60)  # 1-min candles
    print(f"  Expected ~{expected_candles:,} candles")

    request_count = 0

    while current_start < end_ts:
        # Fetch batch
        data = get_binance_data(symbol, interval, current_start, end_ts, limit=1000)

        if not data or len(data) == 0:
            print("  No more data available")
            break

        all_data.extend(data)
        request_count += 1

        # Update progress
        if request_count % 100 == 0:
            current_date = datetime.fromtimestamp(current_start / 1000).strftime('%Y-%m-%d')
            print(f"  Progress: {len(all_data):,} candles downloaded (at {current_date})")

        # Update start time for next batch
        # Last candle's close time + 1ms
        current_start = data[-1][6] + 1

        # Rate limiting (be nice to Binance API)
        time.sleep(0.1)  # 100ms delay between requests

    print(f"  ✓ Downloaded {len(all_data):,} candles in {request_count} requests")

    return all_data


def parse_binance_klines(data):
    """
    Convert Binance klines format to DataFrame

    Binance returns:
    [
      [
        1499040000000,      // Open time
        "0.01634790",       // Open
        "0.80000000",       // High
        "0.01575800",       // Low
        "0.01577100",       // Close
        "148976.11427815",  // Volume
        1499644799999,      // Close time
        "2434.19055334",    // Quote asset volume
        308,                // Number of trades
        "1756.87402397",    // Taker buy base asset volume
        "28.46694368",      // Taker buy quote asset volume
        "17928899.62484339" // Ignore
      ]
    ]
    """
    df = pd.DataFrame(data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])

    # Convert types
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)

    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    df['trades'] = df['trades'].astype(int)

    # Keep only columns matching our existing data format
    df = df[['open_time', 'open', 'high', 'low', 'close', 'volume', 'trades']]
    df = df.rename(columns={'open_time': 'timestamp'})
    df = df.set_index('timestamp')

    return df


def combine_with_existing(new_df, existing_file):
    """
    Combine new historical data with existing 2025-2026 data
    """
    if not existing_file.exists():
        print(f"  No existing file found at {existing_file}")
        return new_df

    print(f"\nCombining with existing data from {existing_file}")
    existing_df = pd.read_parquet(existing_file)

    print(f"  Existing: {len(existing_df):,} rows ({existing_df.index.min()} to {existing_df.index.max()})")
    print(f"  New:      {len(new_df):,} rows ({new_df.index.min()} to {new_df.index.max()})")

    # Combine and sort
    combined = pd.concat([new_df, existing_df])
    combined = combined.sort_index()

    # Remove duplicates (keep first occurrence)
    combined = combined[~combined.index.duplicated(keep='first')]

    print(f"  Combined: {len(combined):,} rows ({combined.index.min()} to {combined.index.max()})")

    return combined


def main():
    data_dir = Path(__file__).parent / 'data'
    data_dir.mkdir(exist_ok=True)

    print("="*60)
    print("Binance Historical Data Download")
    print("="*60)

    # Download data from Binance
    raw_data = download_full_history(SYMBOL, INTERVAL, START_DATE, END_DATE)

    if not raw_data:
        print("❌ Failed to download data")
        return

    # Parse to DataFrame
    print("\nParsing data...")
    df = parse_binance_klines(raw_data)
    print(f"  ✓ Parsed {len(df):,} rows")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print(f"  Columns: {list(df.columns)}")

    # Check for missing data
    time_diff = df.index.to_series().diff()
    expected_diff = pd.Timedelta(minutes=1)
    gaps = time_diff[time_diff > expected_diff * 2]

    if len(gaps) > 0:
        print(f"\n  ⚠ Warning: Found {len(gaps)} gaps in data")
        print(f"    Largest gap: {gaps.max()}")
    else:
        print(f"  ✓ No significant gaps in data")

    # Combine with existing data
    existing_file = data_dir / 'btc_1m_12mo.parquet'
    combined_df = combine_with_existing(df, existing_file)

    # Save combined dataset
    output_file = data_dir / 'btc_1m_4yr.parquet'
    print(f"\nSaving to {output_file}")
    combined_df.to_parquet(output_file)
    print(f"  ✓ Saved {len(combined_df):,} rows")

    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"  Total rows: {len(combined_df):,}")
    print(f"  Date range: {combined_df.index.min()} to {combined_df.index.max()}")
    print(f"  Duration: {(combined_df.index.max() - combined_df.index.min()).days} days")
    print(f"  File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")

    # Basic stats
    print(f"\n  BTC Price Stats:")
    print(f"    Min:  ${combined_df['close'].min():,.2f}")
    print(f"    Max:  ${combined_df['close'].max():,.2f}")
    print(f"    Mean: ${combined_df['close'].mean():,.2f}")

    print("\n✓ Ready for V3 training with 4 years of data!")


if __name__ == '__main__':
    main()
