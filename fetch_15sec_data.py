#!/usr/bin/env python3
"""
Fetch 15-second MNQ/QQQ data from IB for trailing stop analysis.
IB limits: 15-sec data available for last 30 days only.
"""

from ib_insync import IB, Future, Stock
import pandas as pd
from datetime import datetime, timedelta
import time

def fetch_15sec_data(symbol='QQQ', days=30):
    """Fetch 15-second bars from IB."""
    
    ib = IB()
    try:
        ib.connect('127.0.0.1', 4002, clientId=99)
        print(f"Connected to IB Gateway")
    except Exception as e:
        print(f"Failed to connect: {e}")
        return None
    
    # Use QQQ stock (more liquid, same underlying as MNQ)
    if symbol == 'QQQ':
        contract = Stock('QQQ', 'SMART', 'USD')
    else:
        contract = Future('MNQ', '202503', 'CME')
    
    ib.qualifyContracts(contract)
    
    all_bars = []
    
    # IB limits: can only request ~1 day of 15-sec data at a time
    # Request in chunks
    end_date = datetime.now()
    
    for day_offset in range(days):
        end_dt = end_date - timedelta(days=day_offset)
        
        try:
            bars = ib.reqHistoricalData(
                contract,
                endDateTime=end_dt,
                durationStr='1 D',
                barSizeSetting='15 secs',
                whatToShow='TRADES',
                useRTH=True,  # Regular trading hours only
                formatDate=1
            )
            
            if bars:
                df_day = pd.DataFrame([{
                    'datetime': b.date,
                    'open': b.open,
                    'high': b.high,
                    'low': b.low,
                    'close': b.close,
                    'volume': b.volume
                } for b in bars])
                
                all_bars.append(df_day)
                print(f"  Day {day_offset+1}/{days}: {len(bars)} bars")
            
            time.sleep(0.5)  # Rate limiting
            
        except Exception as e:
            print(f"  Day {day_offset+1}: Error - {e}")
            continue
    
    ib.disconnect()
    
    if all_bars:
        df = pd.concat(all_bars, ignore_index=True)
        df = df.drop_duplicates(subset=['datetime'])
        df = df.sort_values('datetime')
        df.set_index('datetime', inplace=True)
        
        # Save
        output_path = f'data/{symbol}_15sec_30days.parquet'
        df.to_parquet(output_path)
        print(f"\nSaved {len(df)} bars to {output_path}")
        
        return df
    
    return None


if __name__ == "__main__":
    df = fetch_15sec_data('QQQ', days=30)
    if df is not None:
        print(f"\nData range: {df.index[0]} to {df.index[-1]}")
        print(f"Total bars: {len(df)}")
