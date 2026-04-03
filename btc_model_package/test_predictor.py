"""
Test Predictor with Real Binance Data

Fetches live BTC data from Binance and tests the prediction flow
"""

from binance.client import Client
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path to import predictor
sys.path.append(str(Path(__file__).parent))

from predictor import BTCPredictor


def test_with_binance_data():
    print("="*60)
    print("Testing BTC Predictor with Real Binance Data")
    print("="*60)

    # Connect to Binance (public API, no credentials needed)
    print("\n1. Connecting to Binance...")
    client = Client("", "")
    print("   ✓ Connected")

    # Fetch latest 5-minute candles
    print("\n2. Fetching BTC 5-minute data...")
    klines = client.get_klines(
        symbol='BTCUSDT',
        interval=Client.KLINE_INTERVAL_5MINUTE,
        limit=250
    )
    print(f"   ✓ Fetched {len(klines)} candles")

    # Convert to DataFrame
    print("\n3. Processing data...")
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])

    # Convert to numeric
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    # Keep only OHLCV
    df = df[['open', 'high', 'low', 'close', 'volume']]

    print(f"   ✓ Processed {len(df)} rows")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")
    print(f"   Current BTC price: ${df['close'].iloc[-1]:,.2f}")

    # Load predictor
    print("\n4. Loading V3 models...")
    predictor = BTCPredictor()

    # Get prediction
    print("\n5. Generating prediction...")
    signal, confidence, details = predictor.predict(df)

    # Display results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"\nSignal:     {signal}")
    print(f"Confidence: {confidence:.2%}")
    print(f"\nModel Probabilities (LONG):")
    for horizon, prob in details['probabilities'].items():
        print(f"  {horizon}: {prob:.3f}")
    print(f"  Average: {details['avg_probability']:.3f}")

    print(f"\nThresholds:")
    print(f"  LONG if avg_prob > {details['thresholds']['long']:.2f}")
    print(f"  SHORT if avg_prob < {details['thresholds']['short']:.2f}")

    print(f"\nCurrent Price: ${details['current_price']:,.2f}")

    if signal != 'NEUTRAL':
        tp, sl = predictor.get_tp_sl_prices(details['current_price'], signal)
        print(f"\nRecommended Levels:")
        print(f"  Entry: ${details['current_price']:,.2f}")
        print(f"  TP:    ${tp:,.2f} ({'+' if signal == 'LONG' else '-'}1.0%)")
        print(f"  SL:    ${sl:,.2f} ({'-' if signal == 'LONG' else '+'}0.5%)")

        # Calculate potential P&L
        if signal == 'LONG':
            tp_pnl = (tp - details['current_price']) / details['current_price'] * 100
            sl_pnl = (sl - details['current_price']) / details['current_price'] * 100
        else:
            tp_pnl = (details['current_price'] - tp) / details['current_price'] * 100
            sl_pnl = (details['current_price'] - sl) / details['current_price'] * 100

        print(f"\nPotential P&L:")
        print(f"  If TP hit: +{tp_pnl:.2f}%")
        print(f"  If SL hit: {sl_pnl:.2f}%")
        print(f"  R:R Ratio: {abs(tp_pnl/sl_pnl):.1f}:1")
    else:
        print("\n⚠ No strong signal - wait for better opportunity")

    print("\n" + "="*60)
    print("✓ Test completed successfully!")
    print("="*60)


if __name__ == '__main__':
    test_with_binance_data()
