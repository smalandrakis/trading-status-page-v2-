"""
Download 1-minute BTC historical data from Binance.
Full history from Dec 2017 to present.
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import os

# Configuration
SYMBOL = "BTCUSDT"
INTERVAL = "1m"
OUTPUT_FILE = "data/historical/BTC_1min.parquet"
START_DATE = "2017-12-07"
END_DATE = "2025-12-08"

# Binance API
BASE_URL = "https://api.binance.com/api/v3/klines"
MAX_BARS_PER_REQUEST = 1000
RATE_LIMIT_DELAY = 0.2  # seconds between requests


def fetch_klines(symbol: str, interval: str, start_time: int, end_time: int) -> list:
    """Fetch klines from Binance API."""
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "endTime": end_time,
        "limit": MAX_BARS_PER_REQUEST
    }
    
    response = requests.get(BASE_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def download_historical_data():
    """Download full historical 1-minute data."""
    
    print("="*70)
    print("DOWNLOADING 1-MINUTE BTC DATA FROM BINANCE")
    print("="*70)
    
    start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_dt = datetime.strptime(END_DATE, "%Y-%m-%d")
    
    print(f"Symbol: {SYMBOL}")
    print(f"Interval: {INTERVAL}")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Output: {OUTPUT_FILE}")
    
    # Calculate expected bars
    total_minutes = int((end_dt - start_dt).total_seconds() / 60)
    expected_requests = total_minutes // MAX_BARS_PER_REQUEST + 1
    print(f"\nExpected bars: ~{total_minutes:,}")
    print(f"Expected requests: ~{expected_requests:,}")
    print(f"Estimated time: ~{expected_requests * RATE_LIMIT_DELAY / 60:.0f} minutes")
    print()
    
    all_data = []
    current_start = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    
    request_count = 0
    last_progress = 0
    
    while current_start < end_ms:
        try:
            # Fetch batch
            klines = fetch_klines(SYMBOL, INTERVAL, current_start, end_ms)
            
            if not klines:
                break
            
            all_data.extend(klines)
            request_count += 1
            
            # Update start time for next batch
            last_timestamp = klines[-1][0]
            current_start = last_timestamp + 60000  # +1 minute in ms
            
            # Progress update every 5%
            progress = int((current_start - int(start_dt.timestamp() * 1000)) / 
                          (end_ms - int(start_dt.timestamp() * 1000)) * 100)
            if progress >= last_progress + 5:
                current_date = datetime.fromtimestamp(current_start / 1000)
                print(f"  {progress}% complete - {current_date.strftime('%Y-%m-%d')} - {len(all_data):,} bars")
                last_progress = progress
            
            # Rate limiting
            time.sleep(RATE_LIMIT_DELAY)
            
        except Exception as e:
            print(f"Error at request {request_count}: {e}")
            print("Retrying in 5 seconds...")
            time.sleep(5)
            continue
    
    print(f"\nDownload complete! Total bars: {len(all_data):,}")
    
    # Convert to DataFrame
    print("Converting to DataFrame...")
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    # Process data
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Convert to numeric
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Keep only OHLCV
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    # Remove duplicates
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()
    
    print(f"\nFinal DataFrame:")
    print(f"  Rows: {len(df):,}")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    print(f"  Columns: {list(df.columns)}")
    
    # Save to parquet
    os.makedirs('data', exist_ok=True)
    df.to_parquet(OUTPUT_FILE)
    
    file_size = os.path.getsize(OUTPUT_FILE) / 1024 / 1024
    print(f"\nSaved to {OUTPUT_FILE} ({file_size:.1f} MB)")
    
    return df


if __name__ == "__main__":
    download_historical_data()
