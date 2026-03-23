"""
Download historical data from Interactive Brokers Gateway.
Connects to IB Gateway on port 4002 and downloads 5-min bars.
"""

import pandas as pd
import ta
from datetime import datetime, timedelta
from ib_insync import IB, Stock, Future, util
import os
import time

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# IB Gateway connection settings
IB_HOST = "127.0.0.1"
IB_PORT = 4002
CLIENT_ID = 1


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all available technical indicators from the 'ta' library."""
    df = df.copy()
    
    # Rename columns to match ta library expectations
    col_map = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
    df = df.rename(columns=col_map)
    
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required:
        if col not in df.columns:
            print(f"Warning: Missing column {col}")
            return df
    
    # Add ALL indicators
    df = ta.add_all_ta_features(
        df,
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
        fillna=True
    )
    
    return df


def download_historical_data(ib: IB, contract, duration: str, bar_size: str, what_to_show: str = "TRADES"):
    """Download historical data for a contract."""
    
    print(f"Requesting {duration} of {bar_size} bars...")
    
    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",  # Empty = now
        durationStr=duration,
        barSizeSetting=bar_size,
        whatToShow=what_to_show,
        useRTH=False,  # Include extended hours
        formatDate=1
    )
    
    if not bars:
        print("No data returned")
        return None
    
    # Convert to DataFrame
    df = util.df(bars)
    df.set_index('date', inplace=True)
    
    return df


def download_in_chunks(ib: IB, contract, bar_size: str, years: int = 5, what_to_show: str = "TRADES"):
    """
    Download data in chunks to work around IB's limitations.
    For 5-min bars, IB allows max ~60 days per request.
    """
    
    all_data = []
    end_date = datetime.now()
    
    # For 5-min bars, download in 30-day chunks
    if "min" in bar_size or "sec" in bar_size:
        chunk_days = 30
        chunk_duration = "30 D"
    else:
        # For daily bars, can get 1 year at a time
        chunk_days = 365
        chunk_duration = "1 Y"
    
    total_days = years * 365
    chunks_needed = (total_days // chunk_days) + 1
    
    print(f"Downloading {years} years of {bar_size} data in {chunks_needed} chunks...")
    
    for i in range(chunks_needed):
        print(f"\nChunk {i+1}/{chunks_needed} - End date: {end_date.strftime('%Y%m%d %H:%M:%S')}")
        
        try:
            bars = ib.reqHistoricalData(
                contract,
                endDateTime=end_date.strftime("%Y%m%d %H:%M:%S"),
                durationStr=chunk_duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=False,
                formatDate=1
            )
            
            if bars:
                df = util.df(bars)
                all_data.append(df)
                print(f"  Got {len(df)} bars: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
                
                # Move end date back
                end_date = df['date'].iloc[0] - timedelta(minutes=1)
            else:
                print("  No data returned for this chunk")
                break
                
        except Exception as e:
            print(f"  Error: {e}")
            break
        
        # Rate limiting - IB has pacing restrictions
        time.sleep(2)
    
    if not all_data:
        return None
    
    # Combine all chunks
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.drop_duplicates(subset=['date'])
    combined = combined.sort_values('date')
    combined.set_index('date', inplace=True)
    
    return combined


def main():
    # Connect to IB Gateway
    ib = IB()
    
    print("="*60)
    print("Connecting to IB Gateway...")
    print(f"Host: {IB_HOST}, Port: {IB_PORT}, Client ID: {CLIENT_ID}")
    print("="*60)
    
    try:
        ib.connect(IB_HOST, IB_PORT, clientId=CLIENT_ID)
        print("Connected successfully!")
        print(f"Server version: {ib.client.serverVersion()}")
    except Exception as e:
        print(f"Connection failed: {e}")
        print("\nMake sure:")
        print("  1. IB Gateway is running")
        print("  2. API connections are enabled in Gateway settings")
        print("  3. Port 4002 is correct (or 7497 for TWS)")
        return
    
    try:
        # Define contracts
        # QQQ ETF
        qqq = Stock("QQQ", "SMART", "USD")
        ib.qualifyContracts(qqq)
        print(f"\nQQQ Contract: {qqq}")
        
        # MNQ Micro Nasdaq Futures (front month)
        # Note: You may need to adjust the expiry
        mnq = Future("MNQ", exchange="CME")
        mnq_contracts = ib.reqContractDetails(mnq)
        
        if mnq_contracts:
            # Get the front month contract
            mnq = mnq_contracts[0].contract
            print(f"MNQ Contract: {mnq}")
        else:
            print("Could not find MNQ contract")
            mnq = None
        
        # Download QQQ 5-min data
        print("\n" + "="*60)
        print("Downloading QQQ 5-minute data (5 years)")
        print("="*60)
        
        df_qqq_5min = download_in_chunks(
            ib, qqq, 
            bar_size="5 mins", 
            years=5,
            what_to_show="TRADES"
        )
        
        if df_qqq_5min is not None:
            print(f"\nTotal QQQ 5-min bars: {len(df_qqq_5min)}")
            print(f"Date range: {df_qqq_5min.index[0]} to {df_qqq_5min.index[-1]}")
            
            # Add indicators
            print("Computing technical indicators...")
            df_qqq_5min = add_all_indicators(df_qqq_5min)
            
            # Save
            filepath = os.path.join(DATA_DIR, "QQQ_5min_IB_with_indicators.csv")
            df_qqq_5min.to_csv(filepath)
            print(f"Saved to {filepath}")
            print(f"Total columns: {len(df_qqq_5min.columns)}")
        
        # Download MNQ if available
        if mnq:
            print("\n" + "="*60)
            print("Downloading MNQ 5-minute data")
            print("="*60)
            
            df_mnq_5min = download_in_chunks(
                ib, mnq,
                bar_size="5 mins",
                years=5,
                what_to_show="TRADES"
            )
            
            if df_mnq_5min is not None:
                print(f"\nTotal MNQ 5-min bars: {len(df_mnq_5min)}")
                print(f"Date range: {df_mnq_5min.index[0]} to {df_mnq_5min.index[-1]}")
                
                # Add indicators
                print("Computing technical indicators...")
                df_mnq_5min = add_all_indicators(df_mnq_5min)
                
                # Save
                filepath = os.path.join(DATA_DIR, "MNQ_5min_IB_with_indicators.csv")
                df_mnq_5min.to_csv(filepath)
                print(f"Saved to {filepath}")
                print(f"Total columns: {len(df_mnq_5min.columns)}")
        
        print("\n" + "="*60)
        print("DOWNLOAD COMPLETE")
        print("="*60)
        
    finally:
        ib.disconnect()
        print("\nDisconnected from IB Gateway")


if __name__ == "__main__":
    main()
