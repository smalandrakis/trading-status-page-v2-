"""
Micro-Movement Predictor

Uses 3-horizon ensemble (30min, 1h, 2h) trained specifically for 0.3% TP / 0.1% SL
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

class MicroPredictor:
    """Predictor for micro-movement trading (0.3% TP / 0.1% SL)"""

    def __init__(self):
        """Load micro-movement models"""
        print("Loading Micro-Movement models...")

        models_dir = Path(__file__).parent.parent / 'models'

        # Load models
        self.model_30min = joblib.load(models_dir / 'btc_model_30min_micro.pkl')
        self.model_1h = joblib.load(models_dir / 'btc_model_1h_micro.pkl')
        self.model_2h = joblib.load(models_dir / 'btc_model_2h_micro.pkl')
        print("  ✓ Loaded 30min model")
        print("  ✓ Loaded 1h model")
        print("  ✓ Loaded 2h model")

        # Load scalers
        self.scaler_30min = joblib.load(models_dir / 'btc_scaler_30min_micro.pkl')
        self.scaler_1h = joblib.load(models_dir / 'btc_scaler_1h_micro.pkl')
        self.scaler_2h = joblib.load(models_dir / 'btc_scaler_2h_micro.pkl')

        # Load feature names
        self.feature_names = joblib.load(models_dir / 'btc_features_micro.pkl')
        print(f"  ✓ Loaded {len(self.feature_names)} features")

        # Initialize V3 predictor ONCE for feature calculation
        from .predictor import BTCPredictor
        self._base_predictor = BTCPredictor()
        print("  ✓ Initialized feature calculator")

        # Ensemble weights (favor shorter horizons for micro-movements)
        self.weights = [0.5, 0.3, 0.2]  # 30min, 1h, 2h

        # Thresholds (can be adjusted by bot)
        self.LONG_THRESHOLD = 0.50
        self.SHORT_THRESHOLD = 0.30

        print("="*60)

    def _calculate_features(self, df_window):
        """Calculate features from OHLCV window (reuse V3 feature calculator)"""
        return self._base_predictor.calculate_features(df_window)

    def predict(self, df_window):
        """
        Predict signal and confidence from OHLCV window

        Args:
            df_window: DataFrame with OHLCV data (at least 250 bars)

        Returns:
            signal: 'LONG', 'SHORT', or 'NEUTRAL'
            confidence: float 0-1
            details: dict with model predictions
        """
        # Calculate features (returns dict)
        features_dict = self._calculate_features(df_window)

        # Convert dict to array in correct order
        features_array = np.array([[features_dict[f] for f in self.feature_names]])

        # Get predictions from each model
        X_30min = self.scaler_30min.transform(features_array)
        X_1h = self.scaler_1h.transform(features_array)
        X_2h = self.scaler_2h.transform(features_array)

        proba_30min = self.model_30min.predict_proba(X_30min)[0]
        proba_1h = self.model_1h.predict_proba(X_1h)[0]
        proba_2h = self.model_2h.predict_proba(X_2h)[0]

        # Ensemble (weighted average of probabilities)
        proba_ensemble = (
            self.weights[0] * proba_30min +
            self.weights[1] * proba_1h +
            self.weights[2] * proba_2h
        )

        # Get class labels
        classes = self.model_30min.classes_

        # Find indices
        long_idx = np.where(classes == 'LONG')[0][0]
        short_idx = np.where(classes == 'SHORT')[0][0]

        long_prob = proba_ensemble[long_idx]
        short_prob = proba_ensemble[short_idx]

        # Determine signal
        if long_prob >= self.LONG_THRESHOLD:
            signal = 'LONG'
            confidence = long_prob
        elif short_prob >= (1 - self.SHORT_THRESHOLD):  # HIGH short probability
            signal = 'SHORT'
            confidence = short_prob
        else:
            signal = 'NEUTRAL'
            confidence = max(long_prob, short_prob)

        details = {
            '30min_probs': {classes[i]: float(proba_30min[i]) for i in range(len(classes))},
            '1h_probs': {classes[i]: float(proba_1h[i]) for i in range(len(classes))},
            '2h_probs': {classes[i]: float(proba_2h[i]) for i in range(len(classes))},
            'ensemble_probs': {classes[i]: float(proba_ensemble[i]) for i in range(len(classes))},
            'long_prob': float(long_prob),
            'short_prob': float(short_prob)
        }

        return signal, confidence, details
