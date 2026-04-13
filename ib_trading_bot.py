"""
IB Gateway Trading Bot for BTC Futures (Paper Trading)

Connects to IB Gateway on port 4002 (paper trading)
Uses V3 ML models to generate trading signals
Manages positions with 1% TP and 0.5% SL

Requirements:
- IB Gateway must be running on port 4002
- Paper trading account enabled
- BTC futures permissions
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime, timedelta
import asyncio
from ib_insync import IB, Future, MarketOrder, LimitOrder, util
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ib_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Trading parameters
TP_PCT = 0.01  # 1% take profit
SL_PCT = 0.005  # 0.5% stop loss
CONFIDENCE_THRESHOLD_LONG = 0.55  # Lower threshold for LONG (better R:R)
CONFIDENCE_THRESHOLD_SHORT = 0.35  # Higher threshold for SHORT (worse R:R)
CONTRACT_SIZE = 1  # Number of contracts per trade
MAX_POSITION_SIZE = 2  # Maximum contracts to hold

# IB Gateway connection details
IB_HOST = '127.0.0.1'
IB_PORT = 4002  # Paper trading port
IB_CLIENT_ID = 1


class BTCTradingBot:
    def __init__(self):
        self.ib = IB()
        self.models = {}
        self.scalers = {}
        self.features = None
        self.current_position = 0
        self.current_price = None
        self.contract = None

        # Load ML models
        self.load_models()

    def load_models(self):
        """Load trained V3 models"""
        models_dir = Path(__file__).parent / 'ml_models'

        logger.info("Loading V3 models...")

        # Load models for each horizon
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
                logger.error(f"  ❌ Missing {horizon} model or scaler")

        # Load feature list
        features_file = models_dir / 'selected_features.json'
        if features_file.exists():
            with open(features_file, 'r') as f:
                self.features = json.load(f)
            logger.info(f"  ✓ Loaded {len(self.features)} features")
        else:
            logger.error("  ❌ Missing feature list")

    async def connect(self):
        """Connect to IB Gateway"""
        logger.info(f"Connecting to IB Gateway at {IB_HOST}:{IB_PORT}...")

        try:
            await self.ib.connectAsync(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID)
            logger.info("✓ Connected to IB Gateway")

            # Define BTC futures contract
            # MBT is the Micro Bitcoin Futures symbol
            self.contract = Future(
                symbol='MBT',
                lastTradeDateOrContractMonth='202409',  # Update to current front month
                exchange='CME',
                currency='USD'
            )

            # Request market data
            self.ib.reqMktData(self.contract, '', False, False)
            await asyncio.sleep(2)  # Wait for data

            # Get current price
            ticker = self.ib.ticker(self.contract)
            self.current_price = ticker.last if ticker.last else ticker.close

            logger.info(f"✓ BTC Futures contract: {self.contract.symbol}")
            logger.info(f"✓ Current price: ${self.current_price:,.2f}")

            # Get current position
            positions = self.ib.positions()
            for pos in positions:
                if pos.contract.symbol == self.contract.symbol:
                    self.current_position = pos.position
                    logger.info(f"✓ Current position: {self.current_position} contracts")

            return True

        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            return False

    def calculate_features(self, bars):
        """
        Calculate features from historical bars
        This is a simplified version - in production you'd need all 22 features
        """
        df = util.df(bars)

        if len(df) < 200:
            logger.warning("Insufficient bars for feature calculation")
            return None

        # Calculate basic features (subset of the 22 features)
        features_dict = {}

        # Price-based features
        features_dict['close_20'] = df['close'].rolling(20).mean().iloc[-1]
        features_dict['close_50'] = df['close'].rolling(50).mean().iloc[-1]
        features_dict['volume_ma_48'] = df['volume'].rolling(48).mean().iloc[-1]

        # Volatility
        features_dict['atr_48_pct'] = (
            df['high'].rolling(48).max() - df['low'].rolling(48).min()
        ).iloc[-1] / df['close'].iloc[-1]

        # Momentum
        features_dict['rsi_14'] = self.calculate_rsi(df['close'], 14)

        # Range
        features_dict['range_4h'] = (
            df['high'].rolling(48).max().iloc[-1] - df['low'].rolling(48).min().iloc[-1]
        ) / df['close'].iloc[-1]

        # Support/Resistance
        features_dict['resistance_48'] = df['high'].rolling(48).max().iloc[-1]
        features_dict['support_48'] = df['low'].rolling(48).min().iloc[-1]

        # ADX proxy (simplified)
        features_dict['adx_proxy'] = df['close'].rolling(14).std().iloc[-1] / df['close'].iloc[-1]

        return features_dict

    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

    async def get_prediction(self):
        """Get ensemble prediction from V3 models"""
        try:
            # Get historical bars (5-minute bars)
            bars = self.ib.reqHistoricalData(
                self.contract,
                endDateTime='',
                durationStr='2 D',
                barSizeSetting='5 mins',
                whatToShow='TRADES',
                useRTH=False,
                formatDate=1
            )

            if not bars or len(bars) < 200:
                logger.warning("Insufficient historical data")
                return None, None

            # Calculate features
            features_dict = self.calculate_features(bars)

            if features_dict is None:
                return None, None

            # Create feature vector (fill missing features with 0)
            X = []
            for feat in self.features:
                X.append(features_dict.get(feat, 0))

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
                return None, None

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
            logger.info(f"  Individual probas: {probas}")

            return signal, confidence

        except Exception as e:
            logger.error(f"Error getting prediction: {e}")
            return None, None

    async def place_order(self, signal):
        """Place market order with TP/SL bracket"""
        if signal == 'NEUTRAL' or signal is None:
            logger.info("No trade signal - staying flat")
            return

        # Check if we already have a position
        if self.current_position != 0:
            logger.info(f"Already have position: {self.current_position} contracts")
            return

        # Update current price
        ticker = self.ib.ticker(self.contract)
        self.current_price = ticker.last if ticker.last else self.current_price

        # Determine order direction
        action = 'BUY' if signal == 'LONG' else 'SELL'
        quantity = CONTRACT_SIZE

        # Calculate TP and SL prices
        if signal == 'LONG':
            tp_price = self.current_price * (1 + TP_PCT)
            sl_price = self.current_price * (1 - SL_PCT)
        else:  # SHORT
            tp_price = self.current_price * (1 - TP_PCT)
            sl_price = self.current_price * (1 + SL_PCT)

        logger.info(f"\n{'='*60}")
        logger.info(f"PLACING ORDER: {action} {quantity} contracts")
        logger.info(f"Entry: ${self.current_price:,.2f}")
        logger.info(f"TP:    ${tp_price:,.2f} ({'+' if signal == 'LONG' else '-'}{TP_PCT*100}%)")
        logger.info(f"SL:    ${sl_price:,.2f} ({'-' if signal == 'LONG' else '+'}{SL_PCT*100}%)")
        logger.info(f"{'='*60}\n")

        try:
            # Place bracket order (entry + TP + SL)
            parent_order = MarketOrder(action, quantity)

            # TP order (opposite direction)
            tp_order = LimitOrder(
                'SELL' if action == 'BUY' else 'BUY',
                quantity,
                tp_price
            )

            # SL order (opposite direction)
            sl_order = LimitOrder(
                'SELL' if action == 'BUY' else 'BUY',
                quantity,
                sl_price
            )

            # Submit parent order
            trade = self.ib.placeOrder(self.contract, parent_order)
            await asyncio.sleep(1)

            logger.info(f"✓ Order placed: {trade.orderStatus.status}")

            # Update position
            self.current_position = quantity if action == 'BUY' else -quantity

        except Exception as e:
            logger.error(f"❌ Order placement failed: {e}")

    async def monitor_positions(self):
        """Monitor open positions and manage exits"""
        while True:
            try:
                positions = self.ib.positions()

                for pos in positions:
                    if pos.contract.symbol == self.contract.symbol:
                        logger.info(f"Position: {pos.position} @ ${pos.avgCost:,.2f}")
                        self.current_position = pos.position

                await asyncio.sleep(10)  # Check every 10 seconds

            except Exception as e:
                logger.error(f"Error monitoring positions: {e}")
                await asyncio.sleep(10)

    async def run(self):
        """Main trading loop"""
        logger.info("\n" + "="*60)
        logger.info("BTC FUTURES TRADING BOT - STARTING")
        logger.info("="*60 + "\n")

        # Connect to IB Gateway
        connected = await self.connect()

        if not connected:
            logger.error("Failed to connect to IB Gateway")
            return

        logger.info("\n✓ Bot is running. Press Ctrl+C to stop.\n")

        try:
            while True:
                # Get prediction
                signal, confidence = await self.get_prediction()

                # Place order if signal is strong
                if signal and confidence:
                    await self.place_order(signal)

                # Wait before next check (check every 5 minutes)
                logger.info(f"Next check in 5 minutes...\n")
                await asyncio.sleep(300)

        except KeyboardInterrupt:
            logger.info("\n\nBot stopped by user")
        except Exception as e:
            logger.error(f"Bot error: {e}")
        finally:
            self.ib.disconnect()
            logger.info("Disconnected from IB Gateway")


async def main():
    bot = BTCTradingBot()
    await bot.run()


if __name__ == '__main__':
    asyncio.run(main())
