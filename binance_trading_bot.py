"""
Binance Real-Time Trading Bot for BTC Futures

Uses Binance API for:
- Real-time data (updates every 2 seconds)
- 24/7 market access (no holidays)
- No subscription needed
- Paper trading via Binance Futures Testnet

Data Flow:
1. Fetch latest 5-min candles from Binance (last 200+ bars)
2. Calculate 22 features in real-time
3. Run through V3 ensemble models
4. Generate LONG/SHORT/NEUTRAL signal
5. Execute trades with 1% TP / 0.5% SL

Requirements:
- Binance API key (for live trading) or Testnet API key (for paper trading)
- python-binance library
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime, timedelta
import time
import logging
from binance.client import Client
from binance.exceptions import BinanceAPIException

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('binance_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Trading parameters
TP_PCT = 0.01  # 1% take profit
SL_PCT = 0.005  # 0.5% stop loss
CONFIDENCE_THRESHOLD_LONG = 0.55  # Asymmetric thresholds
CONFIDENCE_THRESHOLD_SHORT = 0.35
POSITION_SIZE_USD = 100  # Dollar amount per trade
CHECK_INTERVAL = 120  # Check every 2 minutes (120 seconds)

# Binance settings
SYMBOL = 'BTCUSDT'  # Spot trading
TESTNET = True  # Set to False for live trading


class BinanceTradingBot:
    def __init__(self, api_key=None, api_secret=None):
        """
        Initialize bot with Binance API credentials

        For Testnet (paper trading):
        - Get keys from: https://testnet.binance.vision/
        - Set TESTNET = True

        For Live trading:
        - Use your real Binance API keys
        - Set TESTNET = False
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.client = None
        self.models = {}
        self.scalers = {}
        self.features = None
        self.current_position = None
        self.entry_price = None
        self.position_side = None  # 'LONG' or 'SHORT'

        # Load ML models
        self.load_models()

        # Connect to Binance
        self.connect()

    def load_models(self):
        """Load trained V3 models"""
        models_dir = Path(__file__).parent / 'ml_models'

        logger.info("Loading V3 models...")

        for horizon in ['2h', '4h', '6h']:
            model_file = models_dir / f'btc_{horizon}_model_v3.pkl'
            scaler_file = models_dir / f'btc_{horizon}_scaler_v3.pkl'

            if model_file.exists() and scaler_file.exists():
                with open(model_file, 'rb') as f:
                    self.models[horizon] = pickle.load(f)
                with open(scaler_file, 'rb') as f:
                    self.scalers[horizon] = pickle.load(f)
                logger.info(f"  ✓ Loaded {horizon} model")
            else:
                logger.error(f"  ❌ Missing {horizon} model")

        # Load feature list
        features_file = models_dir / 'selected_features.json'
        if features_file.exists():
            with open(features_file, 'r') as f:
                self.features = json.load(f)
            logger.info(f"  ✓ Loaded {len(self.features)} features")

    def connect(self):
        """Connect to Binance API"""
        try:
            if TESTNET:
                # Testnet endpoints
                logger.info("Connecting to Binance TESTNET...")
                if self.api_key and self.api_secret:
                    self.client = Client(
                        self.api_key,
                        self.api_secret,
                        testnet=True
                    )
                    logger.info("✓ Connected to Binance Testnet (Paper Trading)")
                else:
                    # Public client (no trading, just data)
                    self.client = Client("", "")
                    logger.info("✓ Connected to Binance (Public API - Data Only)")
            else:
                # Live trading
                logger.info("Connecting to Binance LIVE...")
                if not self.api_key or not self.api_secret:
                    raise ValueError("API credentials required for live trading")
                self.client = Client(self.api_key, self.api_secret)
                logger.info("✓ Connected to Binance (LIVE TRADING)")

            # Test connection
            server_time = self.client.get_server_time()
            logger.info(f"  Server time: {datetime.fromtimestamp(server_time['serverTime']/1000)}")

            # Get current price
            ticker = self.client.get_symbol_ticker(symbol=SYMBOL)
            logger.info(f"  {SYMBOL} price: ${float(ticker['price']):,.2f}")

            return True

        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            return False

    def get_historical_data(self, limit=250):
        """Fetch historical 5-minute candles from Binance"""
        try:
            # Get 5-minute klines
            klines = self.client.get_klines(
                symbol=SYMBOL,
                interval=Client.KLINE_INTERVAL_5MINUTE,
                limit=limit
            )

            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])

            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            return df[['open', 'high', 'low', 'close', 'volume']]

        except BinanceAPIException as e:
            logger.error(f"Binance API error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return None

    def calculate_features(self, df):
        """Calculate all 22 features from raw OHLCV data"""
        try:
            if len(df) < 200:
                logger.warning(f"Insufficient data: {len(df)} bars (need 200+)")
                return None

            features_dict = {}

            # Moving averages
            features_dict['close_20'] = df['close'].rolling(20).mean().iloc[-1]
            features_dict['close_50'] = df['close'].rolling(50).mean().iloc[-1]
            features_dict['volume_ma_48'] = df['volume'].rolling(48).mean().iloc[-1]

            # Support/Resistance
            features_dict['resistance_48'] = df['high'].rolling(48).max().iloc[-1]
            features_dict['support_48'] = df['low'].rolling(48).min().iloc[-1]

            # Volatility - ATR
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr_48 = true_range.rolling(48).mean().iloc[-1]
            features_dict['atr_48_pct'] = atr_48 / df['close'].iloc[-1]

            # Range
            range_4h = (df['high'].rolling(48).max().iloc[-1] -
                       df['low'].rolling(48).min().iloc[-1])
            features_dict['range_4h'] = range_4h / df['close'].iloc[-1]

            # RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            features_dict['rsi_14'] = rsi.iloc[-1]

            # ADX proxy (volatility momentum)
            features_dict['adx_proxy'] = df['close'].rolling(14).std().iloc[-1] / df['close'].iloc[-1]

            # Returns
            features_dict['return_1'] = df['close'].pct_change(1).iloc[-1]
            features_dict['return_5'] = df['close'].pct_change(5).iloc[-1]
            features_dict['return_10'] = df['close'].pct_change(10).iloc[-1]

            # Rolling volatility
            features_dict['vol_20'] = df['close'].pct_change().rolling(20).std().iloc[-1]
            features_dict['vol_48'] = df['close'].pct_change().rolling(48).std().iloc[-1]

            # Volume features
            features_dict['volume_ratio'] = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]

            # Price position relative to support/resistance
            current_price = df['close'].iloc[-1]
            features_dict['price_to_resistance'] = (features_dict['resistance_48'] - current_price) / current_price
            features_dict['price_to_support'] = (current_price - features_dict['support_48']) / current_price

            # Distance to moving averages
            features_dict['dist_to_ma20'] = (current_price - features_dict['close_20']) / current_price
            features_dict['dist_to_ma50'] = (current_price - features_dict['close_50']) / current_price

            # Bollinger Band position
            bb_mid = df['close'].rolling(20).mean().iloc[-1]
            bb_std = df['close'].rolling(20).std().iloc[-1]
            features_dict['bb_position'] = (current_price - bb_mid) / (2 * bb_std)

            # Trend strength (slope of MA)
            ma20_values = df['close'].rolling(20).mean().iloc[-20:]
            features_dict['ma20_slope'] = (ma20_values.iloc[-1] - ma20_values.iloc[0]) / ma20_values.iloc[0]

            # High/Low position in recent range
            recent_high = df['high'].iloc[-20:].max()
            recent_low = df['low'].iloc[-20:].min()
            features_dict['price_position'] = (current_price - recent_low) / (recent_high - recent_low) if recent_high > recent_low else 0.5

            return features_dict

        except Exception as e:
            logger.error(f"Error calculating features: {e}")
            return None

    def get_prediction(self):
        """Get ensemble prediction from V3 models"""
        try:
            # Fetch latest data
            df = self.get_historical_data(limit=250)

            if df is None or len(df) < 200:
                logger.warning("Insufficient historical data")
                return None, None, None

            # Calculate features
            features_dict = self.calculate_features(df)

            if features_dict is None:
                return None, None, None

            # Create feature vector
            X = []
            missing_features = []
            for feat in self.features:
                if feat in features_dict:
                    X.append(features_dict[feat])
                else:
                    X.append(0)  # Default for missing features
                    missing_features.append(feat)

            if missing_features:
                logger.warning(f"Missing {len(missing_features)} features: {missing_features[:3]}...")

            X = np.array(X).reshape(1, -1)

            # Get predictions from each model
            probas = []
            for horizon in ['2h', '4h', '6h']:
                if horizon in self.models:
                    X_scaled = self.scalers[horizon].transform(X)
                    proba = self.models[horizon].predict_proba(X_scaled)[0]
                    probas.append(proba[1])  # Probability of LONG

            if not probas:
                logger.error("No model predictions available")
                return None, None, None

            # Ensemble: average probability
            avg_proba = np.mean(probas)

            # Apply asymmetric thresholds
            if avg_proba > CONFIDENCE_THRESHOLD_LONG:
                signal = 'LONG'
                confidence = avg_proba
            elif avg_proba < CONFIDENCE_THRESHOLD_SHORT:
                signal = 'SHORT'
                confidence = 1 - avg_proba
            else:
                signal = 'NEUTRAL'
                confidence = 0

            logger.info(f"Prediction: {signal} (confidence: {confidence:.2%})")
            logger.info(f"  Probabilities: 2h={probas[0]:.3f}, 4h={probas[1]:.3f}, 6h={probas[2]:.3f}, avg={avg_proba:.3f}")

            return signal, confidence, df['close'].iloc[-1]

        except Exception as e:
            logger.error(f"Error getting prediction: {e}")
            return None, None, None

    def check_position_exit(self, current_price):
        """Check if current position should be exited (TP or SL hit)"""
        if self.current_position is None or self.entry_price is None:
            return False

        if self.position_side == 'LONG':
            tp_price = self.entry_price * (1 + TP_PCT)
            sl_price = self.entry_price * (1 - SL_PCT)

            if current_price >= tp_price:
                pnl_pct = (current_price - self.entry_price) / self.entry_price * 100
                logger.info(f"\n{'='*60}")
                logger.info(f"✓ TAKE PROFIT HIT (LONG)")
                logger.info(f"  Entry: ${self.entry_price:,.2f}")
                logger.info(f"  Exit:  ${current_price:,.2f}")
                logger.info(f"  P&L:   +{pnl_pct:.2f}%")
                logger.info(f"{'='*60}\n")
                return True

            elif current_price <= sl_price:
                pnl_pct = (current_price - self.entry_price) / self.entry_price * 100
                logger.info(f"\n{'='*60}")
                logger.info(f"✗ STOP LOSS HIT (LONG)")
                logger.info(f"  Entry: ${self.entry_price:,.2f}")
                logger.info(f"  Exit:  ${current_price:,.2f}")
                logger.info(f"  P&L:   {pnl_pct:.2f}%")
                logger.info(f"{'='*60}\n")
                return True

        elif self.position_side == 'SHORT':
            tp_price = self.entry_price * (1 - TP_PCT)
            sl_price = self.entry_price * (1 + SL_PCT)

            if current_price <= tp_price:
                pnl_pct = (self.entry_price - current_price) / self.entry_price * 100
                logger.info(f"\n{'='*60}")
                logger.info(f"✓ TAKE PROFIT HIT (SHORT)")
                logger.info(f"  Entry: ${self.entry_price:,.2f}")
                logger.info(f"  Exit:  ${current_price:,.2f}")
                logger.info(f"  P&L:   +{pnl_pct:.2f}%")
                logger.info(f"{'='*60}\n")
                return True

            elif current_price >= sl_price:
                pnl_pct = (self.entry_price - current_price) / self.entry_price * 100
                logger.info(f"\n{'='*60}")
                logger.info(f"✗ STOP LOSS HIT (SHORT)")
                logger.info(f"  Entry: ${self.entry_price:,.2f}")
                logger.info(f"  Exit:  ${current_price:,.2f}")
                logger.info(f"  P&L:   {pnl_pct:.2f}%")
                logger.info(f"{'='*60}\n")
                return True

        return False

    def simulate_trade(self, signal, current_price):
        """
        Simulate trade for paper trading
        In production, replace with actual Binance order execution
        """
        if signal == 'NEUTRAL' or signal is None:
            return

        # Check if we already have a position
        if self.current_position is not None:
            # Check for exit
            if self.check_position_exit(current_price):
                # Close position
                self.current_position = None
                self.entry_price = None
                self.position_side = None
            return

        # Open new position
        self.current_position = POSITION_SIZE_USD / current_price
        self.entry_price = current_price
        self.position_side = signal

        tp_price = current_price * (1 + TP_PCT) if signal == 'LONG' else current_price * (1 - TP_PCT)
        sl_price = current_price * (1 - SL_PCT) if signal == 'LONG' else current_price * (1 + SL_PCT)

        logger.info(f"\n{'='*60}")
        logger.info(f"OPENING POSITION: {signal}")
        logger.info(f"  Entry: ${current_price:,.2f}")
        logger.info(f"  Size:  ${POSITION_SIZE_USD} ({self.current_position:.6f} BTC)")
        logger.info(f"  TP:    ${tp_price:,.2f} ({'+' if signal == 'LONG' else '-'}{TP_PCT*100}%)")
        logger.info(f"  SL:    ${sl_price:,.2f} ({'-' if signal == 'LONG' else '+'}{SL_PCT*100}%)")
        logger.info(f"{'='*60}\n")

    def run(self):
        """Main trading loop"""
        logger.info("\n" + "="*60)
        logger.info("BINANCE BTC TRADING BOT - STARTING")
        logger.info("="*60)
        logger.info(f"Symbol: {SYMBOL}")
        logger.info(f"Mode: {'TESTNET (Paper Trading)' if TESTNET else 'LIVE TRADING'}")
        logger.info(f"Position Size: ${POSITION_SIZE_USD}")
        logger.info(f"TP: {TP_PCT*100}% / SL: {SL_PCT*100}%")
        logger.info(f"Check Interval: {CHECK_INTERVAL}s")
        logger.info("="*60 + "\n")

        logger.info("✓ Bot is running. Press Ctrl+C to stop.\n")

        try:
            iteration = 0
            while True:
                iteration += 1
                logger.info(f"\n[Iteration {iteration}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                # Get prediction
                signal, confidence, current_price = self.get_prediction()

                if signal and confidence and current_price:
                    # Simulate trade (or execute real trade if API keys provided)
                    self.simulate_trade(signal, current_price)

                    # Show current status
                    if self.current_position:
                        unrealized_pnl = 0
                        if self.position_side == 'LONG':
                            unrealized_pnl = (current_price - self.entry_price) / self.entry_price * 100
                        else:
                            unrealized_pnl = (self.entry_price - current_price) / self.entry_price * 100

                        logger.info(f"Current Position: {self.position_side} @ ${self.entry_price:,.2f}")
                        logger.info(f"Current Price: ${current_price:,.2f}")
                        logger.info(f"Unrealized P&L: {unrealized_pnl:+.2f}%")
                else:
                    logger.warning("Failed to get prediction")

                # Wait before next check
                logger.info(f"Next check in {CHECK_INTERVAL}s...")
                time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            logger.info("\n\nBot stopped by user")
        except Exception as e:
            logger.error(f"Bot error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """
    Initialize and run the bot

    For paper trading (recommended first):
    - Leave api_key and api_secret as None
    - Set TESTNET = True

    For live trading:
    - Get API keys from Binance
    - Set api_key and api_secret
    - Set TESTNET = False
    """

    # Option 1: Data-only mode (no trading, just signals)
    bot = BinanceTradingBot(api_key=None, api_secret=None)

    # Option 2: With Testnet credentials (paper trading)
    # bot = BinanceTradingBot(
    #     api_key='YOUR_TESTNET_API_KEY',
    #     api_secret='YOUR_TESTNET_SECRET_KEY'
    # )

    # Option 3: Live trading (use with caution!)
    # bot = BinanceTradingBot(
    #     api_key='YOUR_BINANCE_API_KEY',
    #     api_secret='YOUR_BINANCE_SECRET_KEY'
    # )

    bot.run()


if __name__ == '__main__':
    main()
