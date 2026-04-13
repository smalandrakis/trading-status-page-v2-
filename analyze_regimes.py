"""
Market Regime Analysis

Analyzes model performance across different market conditions:
1. Volatility regimes (low/medium/high ATR)
2. Trading sessions (Asian/European/US/Overlap)
3. Trend regimes (uptrend/downtrend/sideways)
4. Volume conditions (low/medium/high)

Identifies which market conditions produce the best signals.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Constants
TRAIN_SPLIT = 0.8
TP_PCT = 0.01
SL_PCT = 0.005


def load_best_model_and_data():
    """Load the best performing model (2h) and test data"""
    models_dir = Path(__file__).parent / 'ml_models'
    data_dir = Path(__file__).parent / 'data'

    print("Loading best model (2h horizon) and data...")

    model = joblib.load(models_dir / 'btc_2h_model_v2.pkl')
    scaler = joblib.load(models_dir / 'btc_2h_scaler_v2.pkl')

    import json
    with open(models_dir / 'btc_2h_features_v2.json', 'r') as f:
        features = json.load(f)

    # Load full data
    df = pd.read_parquet(data_dir / 'btc_5m_features_advanced.parquet')

    # Get test set
    split_idx = int(len(df) * TRAIN_SPLIT)
    df_test = df.iloc[split_idx:].copy()

    # Generate predictions
    X = df_test[features]
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)

    df_test['prediction'] = predictions
    df_test['prob_long'] = probabilities[:, 1]

    print(f"  Test set: {len(df_test):,} rows")

    return df_test


def analyze_by_volatility(df_test):
    """Analyze performance by volatility regime"""
    print("\n" + "="*60)
    print("VOLATILITY REGIME ANALYSIS")
    print("="*60)

    # Define volatility regimes based on ATR
    atr = df_test['atr_48_pct']
    low_vol_threshold = atr.quantile(0.33)
    high_vol_threshold = atr.quantile(0.67)

    df_test['vol_regime'] = 'Medium'
    df_test.loc[atr < low_vol_threshold, 'vol_regime'] = 'Low'
    df_test.loc[atr > high_vol_threshold, 'vol_regime'] = 'High'

    print(f"\nVolatility thresholds:")
    print(f"  Low: < {low_vol_threshold:.4f}%")
    print(f"  Medium: {low_vol_threshold:.4f}% - {high_vol_threshold:.4f}%")
    print(f"  High: > {high_vol_threshold:.4f}%")

    results = []
    for regime in ['Low', 'Medium', 'High']:
        df_regime = df_test[df_test['vol_regime'] == regime]
        result = backtest_regime(df_regime, regime)
        results.append(result)

    return pd.DataFrame(results)


def analyze_by_session(df_test):
    """Analyze performance by trading session"""
    print("\n" + "="*60)
    print("TRADING SESSION ANALYSIS")
    print("="*60)

    # Define sessions based on hour
    hour = df_test.index.hour

    df_test['session'] = 'Other'
    df_test.loc[(hour >= 0) & (hour < 8), 'session'] = 'Asian'
    df_test.loc[(hour >= 8) & (hour < 13), 'session'] = 'European'
    df_test.loc[(hour >= 13) & (hour < 22), 'session'] = 'US'
    df_test.loc[(hour >= 13) & (hour < 16), 'session'] = 'EU-US Overlap'

    results = []
    for session in ['Asian', 'European', 'US', 'EU-US Overlap']:
        df_session = df_test[df_test['session'] == session]
        if len(df_session) > 0:
            result = backtest_regime(df_session, session)
            results.append(result)

    return pd.DataFrame(results)


def analyze_by_trend(df_test):
    """Analyze performance by trend regime"""
    print("\n" + "="*60)
    print("TREND REGIME ANALYSIS")
    print("="*60)

    # Define trend based on 4h price change
    trend = df_test['trend_4h_pct']

    df_test['trend_regime'] = 'Sideways'
    df_test.loc[trend > 0.5, 'trend_regime'] = 'Uptrend'
    df_test.loc[trend < -0.5, 'trend_regime'] = 'Downtrend'

    print(f"\nTrend definition:")
    print(f"  Uptrend: 4h change > +0.5%")
    print(f"  Downtrend: 4h change < -0.5%")
    print(f"  Sideways: -0.5% to +0.5%")

    results = []
    for regime in ['Uptrend', 'Sideways', 'Downtrend']:
        df_regime = df_test[df_test['trend_regime'] == regime]
        result = backtest_regime(df_regime, regime)
        results.append(result)

    return pd.DataFrame(results)


def analyze_by_volume(df_test):
    """Analyze performance by volume regime"""
    print("\n" + "="*60)
    print("VOLUME REGIME ANALYSIS")
    print("="*60)

    # Define volume regimes
    vol_ratio = df_test['volume_ratio_48']
    low_vol_threshold = vol_ratio.quantile(0.33)
    high_vol_threshold = vol_ratio.quantile(0.67)

    df_test['volume_regime'] = 'Medium'
    df_test.loc[vol_ratio < low_vol_threshold, 'volume_regime'] = 'Low'
    df_test.loc[vol_ratio > high_vol_threshold, 'volume_regime'] = 'High'

    results = []
    for regime in ['Low', 'Medium', 'High']:
        df_regime = df_test[df_test['volume_regime'] == regime]
        result = backtest_regime(df_regime, regime)
        results.append(result)

    return pd.DataFrame(results)


def backtest_regime(df_regime, regime_name):
    """Backtest within a specific regime"""
    # Get labels and predictions
    actual = df_regime['label_2h']
    predictions = df_regime['prediction']

    # Filter to labeled data
    mask = actual.notna()
    actual = actual[mask]
    predictions = predictions[mask]

    if len(actual) == 0:
        return {
            'regime': regime_name,
            'samples': 0,
            'trades': 0,
            'wins': 0,
            'win_rate': 0.0,
            'avg_pnl': 0.0
        }

    # Calculate P&L
    trades = 0
    wins = 0
    total_pnl = 0

    for pred, act in zip(predictions, actual):
        if pred == 1:  # Predicted LONG
            if act == 1:  # Hit TP
                total_pnl += TP_PCT * 100
                wins += 1
            else:  # Hit SL
                total_pnl -= SL_PCT * 100
        else:  # Predicted SHORT
            if act == 0:  # Hit SL (what we want)
                total_pnl += SL_PCT * 100
                wins += 1
            else:  # Hit TP (against us)
                total_pnl -= TP_PCT * 100
        trades += 1

    win_rate = wins / trades if trades > 0 else 0
    avg_pnl = total_pnl / trades if trades > 0 else 0

    print(f"\n  {regime_name}:")
    print(f"    Samples: {len(df_regime):,}")
    print(f"    Labeled: {len(actual):,}")
    print(f"    Trades: {trades}")
    print(f"    Win Rate: {win_rate:.2%}")
    print(f"    Avg P&L: {avg_pnl:+.4f}%")

    return {
        'regime': regime_name,
        'samples': len(df_regime),
        'trades': trades,
        'wins': wins,
        'win_rate': float(win_rate),
        'avg_pnl': float(avg_pnl)
    }


def main():
    print("="*60)
    print("Market Regime Analysis")
    print("="*60)

    # Load data with predictions
    df_test = load_best_model_and_data()

    # Analyze by different regimes
    vol_results = analyze_by_volatility(df_test)
    session_results = analyze_by_session(df_test)
    trend_results = analyze_by_trend(df_test)
    volume_results = analyze_by_volume(df_test)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: BEST PERFORMING REGIMES")
    print("="*60)

    print("\nBy Volatility:")
    best_vol = vol_results.loc[vol_results['avg_pnl'].idxmax()]
    print(f"  Best: {best_vol['regime']} ({best_vol['win_rate']:.2%} WR, {best_vol['avg_pnl']:+.4f}% avg)")

    print("\nBy Session:")
    best_session = session_results.loc[session_results['avg_pnl'].idxmax()]
    print(f"  Best: {best_session['regime']} ({best_session['win_rate']:.2%} WR, {best_session['avg_pnl']:+.4f}% avg)")

    print("\nBy Trend:")
    best_trend = trend_results.loc[trend_results['avg_pnl'].idxmax()]
    print(f"  Best: {best_trend['regime']} ({best_trend['win_rate']:.2%} WR, {best_trend['avg_pnl']:+.4f}% avg)")

    print("\nBy Volume:")
    best_volume = volume_results.loc[volume_results['avg_pnl'].idxmax()]
    print(f"  Best: {best_volume['regime']} ({best_volume['win_rate']:.2%} WR, {best_volume['avg_pnl']:+.4f}% avg)")

    # Recommendations
    print("\n" + "="*60)
    print("TRADING RECOMMENDATIONS")
    print("="*60)
    print("\nFocus trading on:")
    print(f"  ✓ {best_vol['regime']} volatility periods")
    print(f"  ✓ {best_session['regime']} session")
    print(f"  ✓ {best_trend['regime']} market conditions")
    print(f"  ✓ {best_volume['regime']} volume periods")

    print("\nCombining these filters may further improve performance!")

    # Save results
    all_results = {
        'volatility': vol_results.to_dict('records'),
        'session': session_results.to_dict('records'),
        'trend': trend_results.to_dict('records'),
        'volume': volume_results.to_dict('records')
    }

    output_file = Path(__file__).parent / 'ml_models' / 'regime_analysis.json'
    import json
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Regime analysis saved to: {output_file}")


if __name__ == '__main__':
    main()
