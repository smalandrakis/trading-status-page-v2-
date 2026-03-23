"""Fetch 12 months of 1-min BTC/USDT klines from Binance public API.
No API key needed. Downloads in chunks of 1000 candles, saves to parquet.
Prints progress after each chunk.
"""
import requests, time, os
import pandas as pd
from datetime import datetime, timedelta

OUT_DIR = 'data'
OUT_FILE = os.path.join(OUT_DIR, 'btc_1m_12mo.parquet')
SYMBOL = 'BTCUSDT'
INTERVAL = '1m'
LIMIT = 1000  # max per request

# 12 months back from now
end_ts = int(time.time() * 1000)
start_ts = int((datetime.utcnow() - timedelta(days=365)).timestamp() * 1000)

url = 'https://api.binance.com/api/v3/klines'
all_rows = []
current_ts = start_ts
chunk = 0

print(f"Fetching {SYMBOL} 1-min data: {datetime.utcfromtimestamp(start_ts/1000):%Y-%m-%d} to {datetime.utcfromtimestamp(end_ts/1000):%Y-%m-%d}")
print(f"Expected: ~525,600 candles ({365*24*60:,})")

while current_ts < end_ts:
    params = {
        'symbol': SYMBOL,
        'interval': INTERVAL,
        'startTime': current_ts,
        'limit': LIMIT,
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  Error at chunk {chunk}: {e}, retrying in 2s...")
        time.sleep(2)
        continue

    if not data:
        break

    for k in data:
        all_rows.append({
            'timestamp': pd.Timestamp(k[0], unit='ms', tz='UTC'),
            'open': float(k[1]),
            'high': float(k[2]),
            'low': float(k[3]),
            'close': float(k[4]),
            'volume': float(k[5]),
            'trades': int(k[8]),
        })

    current_ts = data[-1][0] + 60000  # next minute
    chunk += 1

    if chunk % 50 == 0:
        pct = (current_ts - start_ts) / (end_ts - start_ts) * 100
        dt = datetime.utcfromtimestamp(current_ts / 1000)
        print(f"  Chunk {chunk}: {len(all_rows):,} rows ({pct:.0f}%) — up to {dt:%Y-%m-%d %H:%M}")

    # Rate limit: Binance allows 1200 req/min, be conservative
    time.sleep(0.08)

df = pd.DataFrame(all_rows)
df = df.set_index('timestamp').sort_index()
df = df[~df.index.duplicated(keep='last')]
df.to_parquet(OUT_FILE)

print(f"\nDONE: {len(df):,} candles saved to {OUT_FILE}")
print(f"  Date range: {df.index[0]} to {df.index[-1]}")
print(f"  File size: {os.path.getsize(OUT_FILE)/1e6:.1f} MB")
