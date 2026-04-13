"""
Quick Test: Check Micro Predictor Confidence Levels
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
from btc_model_package.micro_predictor import MicroPredictor

BOT_DIR = Path(__file__).parent

# Load data
data_file = BOT_DIR / 'data' / 'BTC_5min_8years.parquet'
df = pd.read_parquet(data_file)
df.columns = [c.lower() for c in df.columns]
df = df[df.index >= '2025-01-01'][:1000]  # Test on first 1000 bars of 2025

# Load predictor
print("Loading predictor...")
predictor = MicroPredictor()

# Test predictions
print("\nTesting predictions on 10 random windows...")
confidences = []
signals = []

for i in range(250, min(len(df), 500), 25):
    window = df.iloc[i-250:i]
    try:
        signal, confidence, details = predictor.predict(window)
        confidences.append(confidence)
        signals.append(signal)

        if len(signals) <= 5:
            print(f"\nBar {i}:")
            print(f"  Signal: {signal}")
            print(f"  Confidence: {confidence:.3f}")
            print(f"  Long prob: {details['long_prob']:.3f}")
            print(f"  Short prob: {details['short_prob']:.3f}")
    except Exception as e:
        print(f"ERROR at bar {i}: {e}")
        continue

# Summary
print(f"\n{'='*60}")
print("SUMMARY OF {len(signals)} PREDICTIONS:")
print('='*60)

from collections import Counter
signal_counts = Counter(signals)
for sig, count in signal_counts.items():
    pct = count / len(signals) * 100
    print(f"{sig}: {count} ({pct:.1f}%)")

print(f"\nConfidence range: {min(confidences):.3f} - {max(confidences):.3f}")
print(f"Mean confidence: {np.mean(confidences):.3f}")
print(f"Median confidence: {np.median(confidences):.3f}")

# Check how many would pass current thresholds
long_signals = [c for s,c in zip(signals, confidences) if s == 'LONG']
short_signals = [c for s,c in zip(signals, confidences) if s == 'SHORT']

print(f"\nLONG signals: {len(long_signals)}")
if long_signals:
    print(f"  Confidence range: {min(long_signals):.3f} - {max(long_signals):.3f}")

print(f"\nSHORT signals: {len(short_signals)}")
if short_signals:
    print(f"  Confidence range: {min(short_signals):.3f} - {max(short_signals):.3f}")

print(f"\n{'='*60}")
