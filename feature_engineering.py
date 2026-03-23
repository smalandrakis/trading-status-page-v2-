"""
Feature engineering for price movement prediction.
Creates time-based features, lagged features, and target variables.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import config


def load_data(filepath: str) -> pd.DataFrame:
    """Load and prepare the data."""
    df = pd.read_csv(filepath, index_col=0)
    
    # Ensure index is datetime (handle timezone-aware strings)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    
    # Convert to US Eastern time and remove timezone info for simplicity
    df.index = df.index.tz_convert('US/Eastern').tz_localize(None)
    
    # Sort by time
    df = df.sort_index()
    
    print(f"Loaded {len(df)} rows")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Columns: {len(df.columns)}")
    
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features."""
    df = df.copy()
    
    # Time of day features
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['time_of_day'] = df['hour'] + df['minute'] / 60  # Continuous time
    
    # Market session indicators (US Eastern Time assumed)
    # Pre-market: 4:00-9:30, Regular: 9:30-16:00, After-hours: 16:00-20:00
    df['is_premarket'] = ((df['hour'] >= 4) & (df['hour'] < 9)) | \
                         ((df['hour'] == 9) & (df['minute'] < 30))
    df['is_regular_hours'] = ((df['hour'] == 9) & (df['minute'] >= 30)) | \
                             ((df['hour'] >= 10) & (df['hour'] < 16))
    df['is_afterhours'] = (df['hour'] >= 16) & (df['hour'] < 20)
    
    # First/last hour of regular trading (often more volatile)
    df['is_first_hour'] = ((df['hour'] == 9) & (df['minute'] >= 30)) | (df['hour'] == 10)
    df['is_last_hour'] = (df['hour'] == 15)
    
    # Day of week (0=Monday, 4=Friday)
    df['day_of_week'] = df.index.dayofweek
    df['is_monday'] = df['day_of_week'] == 0
    df['is_friday'] = df['day_of_week'] == 4
    
    # Month features
    df['month'] = df.index.month
    df['is_month_end'] = df.index.is_month_end
    df['is_month_start'] = df.index.is_month_start
    
    # Week of year
    df['week_of_year'] = df.index.isocalendar().week.astype(int)
    
    # Convert boolean to int
    bool_cols = ['is_premarket', 'is_regular_hours', 'is_afterhours', 
                 'is_first_hour', 'is_last_hour', 'is_monday', 'is_friday',
                 'is_month_end', 'is_month_start']
    for col in bool_cols:
        df[col] = df[col].astype(int)
    
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add price-based features."""
    df = df.copy()
    
    # Returns
    df['return_1bar'] = df['Close'].pct_change()
    df['return_5bar'] = df['Close'].pct_change(5)
    df['return_10bar'] = df['Close'].pct_change(10)
    df['return_20bar'] = df['Close'].pct_change(20)
    
    # Log returns
    df['log_return_1bar'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Price relative to moving averages
    for window in [5, 10, 20, 50]:
        ma = df['Close'].rolling(window).mean()
        df[f'price_to_ma{window}'] = df['Close'] / ma - 1
    
    # Volatility (rolling std of returns)
    for window in [5, 10, 20]:
        df[f'volatility_{window}bar'] = df['return_1bar'].rolling(window).std()
    
    # High-Low range
    df['bar_range'] = (df['High'] - df['Low']) / df['Close']
    df['bar_range_ma5'] = df['bar_range'].rolling(5).mean()
    
    # Close position within bar
    df['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    
    # Gap from previous close
    df['gap'] = df['Open'] / df['Close'].shift(1) - 1
    
    return df


def add_daily_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features based on daily context (previous day's data)."""
    df = df.copy()
    
    # Create a date column for grouping
    df['trade_date'] = df.index.date
    
    # Get daily OHLC
    daily = df.groupby('trade_date').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).rename(columns={
        'Open': 'daily_open',
        'High': 'daily_high',
        'Low': 'daily_low',
        'Close': 'daily_close',
        'Volume': 'daily_volume'
    })
    
    # Previous day's values
    daily['prev_close'] = daily['daily_close'].shift(1)
    daily['prev_high'] = daily['daily_high'].shift(1)
    daily['prev_low'] = daily['daily_low'].shift(1)
    daily['prev_volume'] = daily['daily_volume'].shift(1)
    
    # Previous day's return
    daily['prev_day_return'] = daily['daily_close'].pct_change()
    
    # 2-day and 5-day returns
    daily['prev_2day_return'] = daily['daily_close'].pct_change(2)
    daily['prev_5day_return'] = daily['daily_close'].pct_change(5)
    
    # Merge back to intraday data
    df = df.merge(daily[['prev_close', 'prev_high', 'prev_low', 'prev_volume',
                         'prev_day_return', 'prev_2day_return', 'prev_5day_return']], 
                  left_on='trade_date', right_index=True, how='left')
    
    # Current price relative to previous day
    df['price_vs_prev_close'] = df['Close'] / df['prev_close'] - 1
    df['price_vs_prev_high'] = df['Close'] / df['prev_high'] - 1
    df['price_vs_prev_low'] = df['Close'] / df['prev_low'] - 1
    
    # Volume relative to previous day
    df['volume_vs_prev'] = df['Volume'] / (df['prev_volume'] / 78 + 1)  # ~78 bars per day
    
    # Drop temporary columns
    df = df.drop(columns=['trade_date'])
    
    return df


def add_lagged_indicator_features(df: pd.DataFrame, lookback_periods: List[int]) -> pd.DataFrame:
    """Add lagged versions of key indicators."""
    df = df.copy()
    
    # Key indicators to lag
    key_indicators = [
        'momentum_rsi', 'trend_macd', 'trend_macd_signal', 'trend_macd_diff',
        'volatility_bbp', 'volatility_atr', 'trend_adx', 'momentum_stoch',
        'volume_obv', 'volume_mfi'
    ]
    
    # Only use indicators that exist in the dataframe
    available_indicators = [col for col in key_indicators if col in df.columns]
    
    for indicator in available_indicators:
        for lag in lookback_periods:
            df[f'{indicator}_lag{lag}'] = df[indicator].shift(lag)
    
    return df


def add_indicator_changes(df: pd.DataFrame) -> pd.DataFrame:
    """Add rate of change for key indicators."""
    df = df.copy()
    
    key_indicators = [
        'momentum_rsi', 'trend_macd', 'trend_adx', 'volatility_atr'
    ]
    
    available_indicators = [col for col in key_indicators if col in df.columns]
    
    for indicator in available_indicators:
        df[f'{indicator}_change_1'] = df[indicator].diff(1)
        df[f'{indicator}_change_5'] = df[indicator].diff(5)
    
    return df


def create_target_variables(df: pd.DataFrame, 
                           horizons: Dict[str, int], 
                           thresholds: List[float]) -> pd.DataFrame:
    """
    Create target variables for all horizon/threshold combinations.
    
    Target = 1 if price moves UP by >= threshold within horizon
    Target = -1 if price moves DOWN by >= threshold within horizon
    Target = 0 otherwise (no significant move)
    """
    df = df.copy()
    
    for horizon_name, horizon_bars in horizons.items():
        # Future price at horizon
        future_price = df['Close'].shift(-horizon_bars)
        
        # Future return at horizon
        future_return = (future_price / df['Close'] - 1) * 100  # In percentage
        
        # Store the raw future return
        df[f'future_return_{horizon_name}'] = future_return
        
        # Create binary targets for each threshold
        for threshold in thresholds:
            col_name = f'target_{horizon_name}_{threshold}pct'
            
            # 1 = up move >= threshold, -1 = down move >= threshold, 0 = no significant move
            df[col_name] = 0
            df.loc[future_return >= threshold, col_name] = 1
            df.loc[future_return <= -threshold, col_name] = -1
            
            # Also create a simpler binary version (just up vs not up)
            df[f'target_up_{horizon_name}_{threshold}pct'] = (future_return >= threshold).astype(int)
            df[f'target_down_{horizon_name}_{threshold}pct'] = (future_return <= -threshold).astype(int)
    
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get list of feature columns (excluding targets and metadata)."""
    
    # Columns to exclude
    exclude_patterns = ['target_', 'future_return_', 'date', 'Unnamed']
    exclude_exact = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
    
    feature_cols = []
    for col in df.columns:
        # Skip if matches exclude patterns
        if any(pattern in col for pattern in exclude_patterns):
            continue
        # Skip if exact match
        if col in exclude_exact:
            continue
        feature_cols.append(col)
    
    return feature_cols


def prepare_features(filepath: str = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Main function to prepare all features.
    Returns the processed dataframe and list of feature columns.
    """
    if filepath is None:
        filepath = f"{config.DATA_DIR}/{config.DATA_FILE}"
    
    print("Loading data...")
    df = load_data(filepath)
    
    print("Adding time features...")
    df = add_time_features(df)
    
    print("Adding price features...")
    df = add_price_features(df)
    
    print("Adding daily context features...")
    df = add_daily_context_features(df)
    
    print("Adding lagged indicator features...")
    df = add_lagged_indicator_features(df, config.LOOKBACK_PERIODS)
    
    print("Adding indicator changes...")
    df = add_indicator_changes(df)
    
    print("Creating target variables...")
    df = create_target_variables(df, config.HORIZONS, config.THRESHOLDS)
    
    # Get feature columns
    feature_cols = get_feature_columns(df)
    
    # Drop rows with NaN in features (due to lagging/rolling)
    initial_len = len(df)
    df = df.dropna(subset=feature_cols)
    print(f"Dropped {initial_len - len(df)} rows with NaN values")
    
    # Also drop rows where targets are NaN (end of dataset)
    target_cols = [col for col in df.columns if col.startswith('target_')]
    df = df.dropna(subset=target_cols)
    
    print(f"\nFinal dataset: {len(df)} rows")
    print(f"Features: {len(feature_cols)}")
    print(f"Target combinations: {len(config.HORIZONS) * len(config.THRESHOLDS)}")
    
    return df, feature_cols


if __name__ == "__main__":
    # Test the feature engineering
    df, feature_cols = prepare_features()
    
    print("\n" + "="*60)
    print("Feature Engineering Complete")
    print("="*60)
    
    print(f"\nSample features ({len(feature_cols)} total):")
    for col in feature_cols[:20]:
        print(f"  - {col}")
    print("  ...")
    
    print(f"\nTarget columns:")
    target_cols = [col for col in df.columns if col.startswith('target_') and 'up_' not in col and 'down_' not in col]
    for col in target_cols[:10]:
        print(f"  - {col}")
    print("  ...")
    
    # Save processed data
    output_path = f"{config.DATA_DIR}/QQQ_features.parquet"
    df.to_parquet(output_path)
    print(f"\nSaved processed data to {output_path}")
