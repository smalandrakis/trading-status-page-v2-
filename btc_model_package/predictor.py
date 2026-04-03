"""
BTC Trading Signal Predictor - V3 Models

Simple interface to use the V3 ensemble models in your trading bots.

Usage:
    from predictor import BTCPredictor

    predictor = BTCPredictor()
    signal, confidence = predictor.predict(df)

    if signal == 'LONG':
        # Enter long position
    elif signal == 'SHORT':
        # Enter short position
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path


class BTCPredictor:
    """
    BTC Trade Direction Predictor using V3 Ensemble Models

    Models trained on 4 years of BTC 5-minute data (2022-2026)
    Strategy: 1% TP / 0.5% SL (2:1 R:R, need 33.3% WR to break even)
    Validated: 54.24% WR, +0.31% avg P&L per trade
    """

    def __init__(self, model_dir=None):
        """
        Initialize predictor and load models

        Args:
            model_dir: Path to model directory (default: same directory as this file)
        """
        if model_dir is None:
            model_dir = Path(__file__).parent
        else:
            model_dir = Path(model_dir)

        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        self.features = None

        # Trading parameters (informational)
        self.TP_PCT = 0.01  # 1%
        self.SL_PCT = 0.005  # 0.5%
        self.BREAKEVEN_WR = 0.333  # 33.3%

        # Asymmetric thresholds (favor LONG with better R:R)
        self.LONG_THRESHOLD = 0.55
        self.SHORT_THRESHOLD = 0.35

        self._load_models()

    def _load_models(self):
        """Load all V3 models and scalers"""
        print("Loading V3 models...")

        for horizon in ['2h', '4h', '6h']:
            model_file = self.model_dir / f'btc_{horizon}_model_v3.pkl'
            scaler_file = self.model_dir / f'btc_{horizon}_scaler_v3.pkl'

            if not model_file.exists() or not scaler_file.exists():
                raise FileNotFoundError(f"Missing {horizon} model or scaler")

            self.models[horizon] = joblib.load(model_file)
            self.scalers[horizon] = joblib.load(scaler_file)

            print(f"  ✓ Loaded {horizon} model")

        # Load feature list - use actual features from trained models
        # Try horizon-specific feature list first (more accurate)
        features_file = self.model_dir / 'btc_2h_features_v3.json'
        if features_file.exists():
            with open(features_file, 'r') as f:
                self.features = json.load(f)
        else:
            # Fallback to selected_features.json
            features_file = self.model_dir / 'selected_features.json'
            if not features_file.exists():
                raise FileNotFoundError("Missing feature list")
            with open(features_file, 'r') as f:
                self.features = json.load(f)

        print(f"  ✓ Loaded {len(self.features)} features")
        print("="*60)

    def calculate_features(self, df):
        """
        Calculate all 22 features from OHLCV DataFrame

        Args:
            df: DataFrame with columns [open, high, low, close, volume]
                Must have at least 200 rows (5-min candles = ~16-17 hours)

        Returns:
            dict: Feature values keyed by feature name
        """
        if len(df) < 200:
            raise ValueError(f"Insufficient data: {len(df)} rows (need 200+)")

        features_dict = {}
        current_price = df['close'].iloc[-1]

        # Support/Resistance (48 bars = 4 hours)
        features_dict['resistance_48'] = df['high'].rolling(48).max().iloc[-1]
        features_dict['support_48'] = df['low'].rolling(48).min().iloc[-1]

        # Support/Resistance (96 bars = 8 hours)
        features_dict['resistance_96'] = df['high'].rolling(96).max().iloc[-1]
        features_dict['support_96'] = df['low'].rolling(96).min().iloc[-1]

        # Distance to support/resistance
        features_dict['dist_to_resistance_48'] = ((features_dict['resistance_48'] - current_price) /
                                                  current_price)
        features_dict['dist_to_support_48'] = ((current_price - features_dict['support_48']) /
                                              current_price)
        features_dict['dist_to_resistance_96'] = ((features_dict['resistance_96'] - current_price) /
                                                  current_price)
        features_dict['dist_to_support_96'] = ((current_price - features_dict['support_96']) /
                                              current_price)

        # Volume moving averages
        features_dict['volume_ma_12'] = df['volume'].rolling(12).mean().iloc[-1]
        features_dict['volume_ma_24'] = df['volume'].rolling(24).mean().iloc[-1]
        features_dict['volume_ma_48'] = df['volume'].rolling(48).mean().iloc[-1]

        # Volatility - ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        atr_12 = true_range.rolling(12).mean().iloc[-1]
        atr_24 = true_range.rolling(24).mean().iloc[-1]
        atr_48 = true_range.rolling(48).mean().iloc[-1]

        features_dict['atr_12'] = atr_12
        features_dict['atr_24'] = atr_24
        features_dict['atr_48_pct'] = atr_48 / current_price

        # Range (4h = 48 bars)
        range_4h = (df['high'].rolling(48).max().iloc[-1] -
                   df['low'].rolling(48).min().iloc[-1])
        features_dict['range_4h'] = range_4h / current_price

        # ADX proxy (volatility momentum)
        features_dict['adx_proxy'] = df['close'].rolling(14).std().iloc[-1] / current_price

        # Trend (2h = 24 bars)
        close_24_ago = df['close'].iloc[-24] if len(df) >= 24 else df['close'].iloc[0]
        features_dict['trend_2h_pct'] = (current_price - close_24_ago) / close_24_ago * 100

        # Distance from previous 4h close (48 bars)
        close_48_ago = df['close'].iloc[-48] if len(df) >= 48 else df['close'].iloc[0]
        features_dict['dist_from_prev_4h'] = (current_price - close_48_ago) / close_48_ago * 100

        # RSI (28 period)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(28).mean()
        loss = -delta.where(delta < 0, 0).rolling(28).mean()
        rs = gain / (loss + 1e-8)
        features_dict['rsi_28'] = (100 - (100 / (1 + rs))).iloc[-1]

        # Bollinger Band position (50 period)
        bb_mid_50 = df['close'].rolling(50).mean().iloc[-1]
        bb_std_50 = df['close'].rolling(50).std().iloc[-1]
        features_dict['bb_position_50'] = (current_price - bb_mid_50) / (2 * bb_std_50 + 1e-8)

        # Time features
        if hasattr(df.index, 'hour'):
            hour = df.index[-1].hour
            day_of_week = df.index[-1].dayofweek
        else:
            hour = 12  # default
            day_of_week = 0  # Monday default

        features_dict['hour'] = hour
        features_dict['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        features_dict['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        features_dict['day_of_week'] = day_of_week

        # Volume hour median (simplified - use recent median)
        features_dict['volume_hour_median'] = df['volume'].rolling(20).median().iloc[-1]

        return features_dict

    def predict(self, df):
        """
        Generate trading signal from OHLCV data

        Args:
            df: DataFrame with columns [open, high, low, close, volume]
                Index should be datetime (optional)
                Must have at least 200 rows

        Returns:
            tuple: (signal, confidence, details)
                signal: 'LONG', 'SHORT', or 'NEUTRAL'
                confidence: float (0-1)
                details: dict with probabilities and metadata
        """
        # Calculate features
        features_dict = self.calculate_features(df)

        # Create feature vector
        X = []
        missing_features = []
        for feat in self.features:
            if feat in features_dict:
                X.append(features_dict[feat])
            else:
                X.append(0)  # Default for missing features
                missing_features.append(feat)

        X = np.array(X).reshape(1, -1)

        # Get predictions from each model
        probas = {}
        for horizon in ['2h', '4h', '6h']:
            X_scaled = self.scalers[horizon].transform(X)
            proba = self.models[horizon].predict_proba(X_scaled)[0]
            probas[horizon] = proba[1]  # Probability of LONG

        # Ensemble: average probability
        avg_proba = np.mean(list(probas.values()))

        # Apply asymmetric thresholds
        if avg_proba > self.LONG_THRESHOLD:
            signal = 'LONG'
            confidence = avg_proba
        elif avg_proba < self.SHORT_THRESHOLD:
            signal = 'SHORT'
            confidence = 1 - avg_proba
        else:
            signal = 'NEUTRAL'
            confidence = 0

        # Details
        details = {
            'probabilities': probas,
            'avg_probability': avg_proba,
            'thresholds': {
                'long': self.LONG_THRESHOLD,
                'short': self.SHORT_THRESHOLD
            },
            'current_price': df['close'].iloc[-1],
            'missing_features': missing_features
        }

        return signal, confidence, details

    def get_tp_sl_prices(self, entry_price, signal):
        """
        Calculate TP and SL prices for a given entry and direction

        Args:
            entry_price: Entry price
            signal: 'LONG' or 'SHORT'

        Returns:
            tuple: (tp_price, sl_price)
        """
        if signal == 'LONG':
            tp = entry_price * (1 + self.TP_PCT)
            sl = entry_price * (1 - self.SL_PCT)
        elif signal == 'SHORT':
            tp = entry_price * (1 - self.TP_PCT)
            sl = entry_price * (1 + self.SL_PCT)
        else:
            return None, None

        return tp, sl


# Example usage
if __name__ == '__main__':
    # Example: Create sample data
    print("Example Usage\n" + "="*60)

    # You would replace this with real data from Binance/IB/etc
    dates = pd.date_range(end=pd.Timestamp.now(), periods=250, freq='5min')
    sample_df = pd.DataFrame({
        'open': np.random.randn(250).cumsum() + 50000,
        'high': np.random.randn(250).cumsum() + 50100,
        'low': np.random.randn(250).cumsum() + 49900,
        'close': np.random.randn(250).cumsum() + 50000,
        'volume': np.random.rand(250) * 1000
    }, index=dates)

    # Initialize predictor
    predictor = BTCPredictor()

    # Get prediction
    signal, confidence, details = predictor.predict(sample_df)

    print(f"\nSignal: {signal}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Probabilities: {details['probabilities']}")
    print(f"Current Price: ${details['current_price']:,.2f}")

    if signal != 'NEUTRAL':
        tp, sl = predictor.get_tp_sl_prices(details['current_price'], signal)
        print(f"\nTP: ${tp:,.2f} ({'+' if signal == 'LONG' else '-'}1%)")
        print(f"SL: ${sl:,.2f} ({'-' if signal == 'LONG' else '+'}0.5%)")
