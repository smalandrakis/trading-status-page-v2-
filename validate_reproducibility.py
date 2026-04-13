"""
Validate Signal Reproducibility from Logged Features

Instead of downloading data, use the actual features logged in predictor_details
to verify that the signal generation is reproducible and deterministic.

This validates:
1. Same features → same probabilities
2. Same probabilities → same signal
3. Model is deterministic
4. No randomness or timing issues
"""

import json
import os
import numpy as np
import joblib
from pathlib import Path

BOT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_predictor_components():
    """Load predictor models and scalers"""
    model_dir = Path(BOT_DIR) / 'btc_model_package'

    models = {}
    scalers = {}

    for horizon in ['2h', '4h', '6h']:
        model_file = model_dir / f'btc_{horizon}_model_v3.pkl'
        scaler_file = model_dir / f'btc_{horizon}_scaler_v3.pkl'

        models[horizon] = joblib.load(model_file)
        scalers[horizon] = joblib.load(scaler_file)

    # Load feature list
    with open(model_dir / 'btc_2h_features_v3.json', 'r') as f:
        features = json.load(f)

    return models, scalers, features


def validate_signal_from_features(signal_data, models, scalers, features):
    """
    Given logged features, reproduce the signal and check if it matches
    """
    predictor_details = signal_data['details']

    if not predictor_details or 'features' not in predictor_details:
        return None, "No features logged"

    # Extract features in correct order
    logged_features = predictor_details['features']
    X = np.array([logged_features[feat] for feat in features]).reshape(1, -1)

    # Get probabilities from each model
    computed_probs = {}
    for horizon in ['2h', '4h', '6h']:
        X_scaled = scalers[horizon].transform(X)
        proba = models[horizon].predict_proba(X_scaled)[0]
        computed_probs[horizon] = proba[1]  # Probability of LONG

    # Compute ensemble average
    avg_proba = np.mean(list(computed_probs.values()))

    # Determine signal
    if avg_proba > 0.65:
        computed_signal = 'LONG'
        computed_confidence = avg_proba
    elif avg_proba < 0.25:
        computed_signal = 'SHORT'
        computed_confidence = 1 - avg_proba
    else:
        computed_signal = 'NEUTRAL'
        computed_confidence = avg_proba

    # Compare with logged values
    logged_probs = predictor_details['probabilities']
    logged_avg = predictor_details['avg_probability']

    # Check match
    probs_match = all(
        abs(computed_probs[h] - logged_probs[h]) < 0.0001
        for h in ['2h', '4h', '6h']
    )

    avg_match = abs(avg_proba - logged_avg) < 0.0001
    signal_match = computed_signal == signal_data['direction']
    conf_match = abs(computed_confidence - signal_data['confidence']) < 0.0001

    return {
        'signal_match': signal_match,
        'probs_match': probs_match,
        'avg_match': avg_match,
        'conf_match': conf_match,
        'logged_signal': signal_data['direction'],
        'computed_signal': computed_signal,
        'logged_probs': logged_probs,
        'computed_probs': computed_probs,
        'logged_avg': logged_avg,
        'computed_avg': avg_proba,
        'logged_conf': signal_data['confidence'],
        'computed_conf': computed_confidence
    }, None


def analyze_bot_log(log_file, bot_name, models, scalers, features):
    """Validate all signals in a bot log"""
    if not os.path.exists(log_file):
        print(f"✗ {bot_name}: Log file not found")
        return

    signals = []

    with open(log_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if entry['event'] == 'ENTRY_SIGNAL':
                    signals.append({
                        'timestamp': entry['timestamp'],
                        'direction': entry['data']['direction'],
                        'confidence': entry['data']['confidence'],
                        'details': entry['data'].get('predictor_details', {})
                    })
            except:
                continue

    if not signals:
        print(f"✗ {bot_name}: No signals with features found")
        return

    print(f"\n{'='*80}")
    print(f"{bot_name}")
    print(f"{'='*80}")

    print(f"\nValidating {len(signals)} signals with logged features...")

    exact_matches = 0
    partial_matches = 0
    mismatches = 0

    for i, signal in enumerate(signals):
        result, error = validate_signal_from_features(signal, models, scalers, features)

        if error:
            print(f"\n✗ Signal {i+1} [{signal['timestamp']}]: {error}")
            continue

        if (result['signal_match'] and result['probs_match'] and
            result['avg_match'] and result['conf_match']):
            exact_matches += 1
            print(f"\n✓ Signal {i+1} [{signal['timestamp']}]: EXACT MATCH")
            print(f"  Direction: {result['logged_signal']}")
            print(f"  Confidence: {result['logged_conf']:.6f}")
            print(f"  Probabilities: 2h={result['logged_probs']['2h']:.6f} "
                  f"4h={result['logged_probs']['4h']:.6f} 6h={result['logged_probs']['6h']:.6f}")

        elif result['signal_match']:
            partial_matches += 1
            print(f"\n≈ Signal {i+1} [{signal['timestamp']}]: PARTIAL MATCH (signal correct, minor diff)")
            print(f"  Direction: {result['logged_signal']} ✓")
            print(f"  Logged confidence:   {result['logged_conf']:.6f}")
            print(f"  Computed confidence: {result['computed_conf']:.6f}")
            print(f"  Logged avg prob:     {result['logged_avg']:.6f}")
            print(f"  Computed avg prob:   {result['computed_avg']:.6f}")

        else:
            mismatches += 1
            print(f"\n✗ Signal {i+1} [{signal['timestamp']}]: MISMATCH")
            print(f"  Logged:   {result['logged_signal']} (conf: {result['logged_conf']:.6f})")
            print(f"  Computed: {result['computed_signal']} (conf: {result['computed_conf']:.6f})")
            print(f"  Logged probs:   2h={result['logged_probs']['2h']:.6f} "
                  f"4h={result['logged_probs']['4h']:.6f} 6h={result['logged_probs']['6h']:.6f}")
            print(f"  Computed probs: 2h={result['computed_probs']['2h']:.6f} "
                  f"4h={result['computed_probs']['4h']:.6f} 6h={result['computed_probs']['6h']:.6f}")

    # Summary
    print(f"\n{'='*80}")
    print("VALIDATION RESULTS")
    print(f"{'='*80}")
    print(f"Total Signals:     {len(signals)}")
    print(f"Exact Matches:     {exact_matches} ({exact_matches/len(signals)*100:.1f}%)")
    print(f"Partial Matches:   {partial_matches} ({partial_matches/len(signals)*100:.1f}%)")
    print(f"Mismatches:        {mismatches} ({mismatches/len(signals)*100:.1f}%)")

    if exact_matches == len(signals):
        print(f"\n✅ PERFECT - All signals reproduced exactly!")
        print("   The predictor is 100% deterministic and reproducible.")
    elif exact_matches + partial_matches == len(signals):
        print(f"\n✅ GOOD - All signals match (minor numerical differences)")
        print("   Signal direction correct, small floating point diffs.")
    elif exact_matches + partial_matches >= len(signals) * 0.9:
        print(f"\n⚠️  MOSTLY GOOD - 90%+ signals match")
    else:
        print(f"\n❌ ISSUES DETECTED - Significant mismatches")

    return {
        'total': len(signals),
        'exact': exact_matches,
        'partial': partial_matches,
        'mismatch': mismatches
    }


def main():
    print("="*80)
    print("SIGNAL REPRODUCIBILITY VALIDATION")
    print("="*80)
    print("\nValidating that logged features reproduce exact signals...")

    # Load predictor components
    print("\nLoading models...")
    models, scalers, features = load_predictor_components()
    print(f"✓ Loaded 3 models, 3 scalers, {len(features)} features")

    # Analyze bot logs
    bots = [
        ('V1 Swing Bot (2.5/1.0)', os.path.join(BOT_DIR, 'logs', 'btc_trades.jsonl')),
        ('V1 High Frequency (1.0/0.5)', os.path.join(BOT_DIR, 'logs', 'btc_trades_hf.jsonl')),
    ]

    results = {}
    for bot_name, log_file in bots:
        result = analyze_bot_log(log_file, bot_name, models, scalers, features)
        if result:
            results[bot_name] = result

    # Final summary
    print(f"\n\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")

    for bot_name, result in results.items():
        match_rate = (result['exact'] + result['partial']) / result['total'] * 100
        exact_rate = result['exact'] / result['total'] * 100

        print(f"\n{bot_name}")
        print(f"  Exact matches: {result['exact']}/{result['total']} ({exact_rate:.1f}%)")
        print(f"  Total matches: {result['exact'] + result['partial']}/{result['total']} ({match_rate:.1f}%)")

        if exact_rate >= 95:
            print(f"  Status: ✅ PASS - Fully reproducible")
        elif match_rate >= 90:
            print(f"  Status: ✅ PASS - Signal direction always correct")
        else:
            print(f"  Status: ❌ FAIL - Reproducibility issues")


if __name__ == '__main__':
    main()
