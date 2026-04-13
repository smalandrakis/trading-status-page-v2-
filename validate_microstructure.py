"""
Validate Micro-Structure Models (0.5% TP / 0.15% SL)

Test if adding 16 micro-structure features improves performance
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import sys

BOT_DIR = Path(__file__).parent

# Load data
data_file = BOT_DIR / 'data' / 'BTC_5min_8years.parquet'
print("Loading data...")
df = pd.read_parquet(data_file)
df.columns = [c.lower() for c in df.columns]
df = df[df.index >= '2025-01-01']
print(f"Testing on {len(df):,} bars\n")

# Add micro-structure features
print("Adding micro-structure features...")
sys.path.insert(0, str(BOT_DIR))
from feature_engineering_microstructure import add_microstructure_features
df = add_microstructure_features(df)

# Load models
print("\nLoading micro-structure models...")
models_dir = BOT_DIR / 'models'
model_30min = joblib.load(models_dir / 'btc_model_30min_microstructure.pkl')
model_1h = joblib.load(models_dir / 'btc_model_1h_microstructure.pkl')
model_2h = joblib.load(models_dir / 'btc_model_2h_microstructure.pkl')
scaler_30min = joblib.load(models_dir / 'btc_scaler_30min_microstructure.pkl')
scaler_1h = joblib.load(models_dir / 'btc_scaler_1h_microstructure.pkl')
scaler_2h = joblib.load(models_dir / 'btc_scaler_2h_microstructure.pkl')
feature_names = joblib.load(models_dir / 'btc_features_microstructure.pkl')

# Load base predictor
from btc_model_package.predictor import BTCPredictor
base_predictor = BTCPredictor()

# Parameters
TP_PCT = 0.005
SL_PCT = 0.0015
COMMISSION = 2.02
SLIPPAGE = 0.0005
BTC_CONTRACT_SIZE = 0.1

WEIGHTS = [0.5, 0.3, 0.2]
LONG_THRESHOLD = 0.50
SHORT_THRESHOLD = 0.30

def calculate_position_size(confidence):
    size = (confidence - 0.45) * 15
    return int(max(3, min(6, size)))

def calculate_features(window, idx):
    # V3 features
    features_dict = base_predictor.calculate_features(window)

    # Micro-structure features
    row = df.iloc[idx]
    micro_features = {
        'vwap_dist_5': row['vwap_dist_5'],
        'vwap_dist_20': row['vwap_dist_20'],
        'tick_velocity': row['tick_velocity'],
        'realized_vol_5': row['realized_vol_5'],
        'dist_to_high': row['dist_to_high'],
        'dist_to_low': row['dist_to_low'],
        'volume_deviation': row['volume_deviation'],
        'session_asia': row['session_asia'],
        'session_europe': row['session_europe'],
        'session_us': row['session_us'],
        'momentum_1bar': row['momentum_1bar'],
        'momentum_3bar': row['momentum_3bar'],
        'momentum_5bar': row['momentum_5bar'],
        'spread_proxy': row['spread_proxy'],
        'price_position': row['price_position'],
        'volume_trend': row['volume_trend']
    }

    features_dict.update(micro_features)

    # Convert to array in correct order
    features_array = np.array([[features_dict.get(f, 0) for f in feature_names]])

    # Replace inf with 0
    features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)

    return features_array

def predict(window, idx):
    features = calculate_features(window, idx)

    # Scale and predict
    features_30min = scaler_30min.transform(features)
    features_1h = scaler_1h.transform(features)
    features_2h = scaler_2h.transform(features)

    proba_30min = model_30min.predict_proba(features_30min)[0]
    proba_1h = model_1h.predict_proba(features_1h)[0]
    proba_2h = model_2h.predict_proba(features_2h)[0]

    # Ensemble
    proba = WEIGHTS[0] * proba_30min + WEIGHTS[1] * proba_1h + WEIGHTS[2] * proba_2h

    classes = model_30min.classes_
    long_idx = list(classes).index('LONG')
    short_idx = list(classes).index('SHORT')

    long_prob = proba[long_idx]
    short_prob = proba[short_idx]

    if long_prob >= LONG_THRESHOLD:
        return 'LONG', long_prob
    elif short_prob >= SHORT_THRESHOLD:
        return 'SHORT', short_prob
    else:
        return 'NEUTRAL', max(long_prob, short_prob)

def simulate_trade(entry_price, direction, size, forward_prices):
    if direction == 'LONG':
        tp = entry_price * (1 + TP_PCT - SLIPPAGE)
        sl = entry_price * (1 - SL_PCT - SLIPPAGE)
    else:
        tp = entry_price * (1 - TP_PCT + SLIPPAGE)
        sl = entry_price * (1 + SL_PCT + SLIPPAGE)

    for price in forward_prices:
        if direction == 'LONG':
            if price >= tp:
                notional = entry_price * BTC_CONTRACT_SIZE * size
                pnl = notional * (TP_PCT - SLIPPAGE) - COMMISSION
                return 'WIN', pnl
            elif price <= sl:
                notional = entry_price * BTC_CONTRACT_SIZE * size
                pnl = -(notional * (SL_PCT + SLIPPAGE) + COMMISSION)
                return 'LOSS', pnl
        else:
            if price <= tp:
                notional = entry_price * BTC_CONTRACT_SIZE * size
                pnl = notional * (TP_PCT - SLIPPAGE) - COMMISSION
                return 'WIN', pnl
            elif price >= sl:
                notional = entry_price * BTC_CONTRACT_SIZE * size
                pnl = -(notional * (SL_PCT + SLIPPAGE) + COMMISSION)
                return 'LOSS', pnl

    return 'TIMEOUT', 0

# Run backtest
print("Running backtest (every 10 bars)...")
trades = []
for i in range(250, len(df) - 48, 10):
    window = df.iloc[i-250:i]
    try:
        signal, confidence = predict(window, i)
    except:
        continue

    if signal == 'NEUTRAL':
        continue

    size = calculate_position_size(confidence)
    entry_price = df.iloc[i]['close']
    forward_prices = df.iloc[i+1:i+49]['close'].values

    outcome, pnl = simulate_trade(entry_price, signal, size, forward_prices)

    trades.append({
        'direction': signal,
        'confidence': confidence,
        'size': size,
        'outcome': outcome,
        'pnl': pnl
    })

    if len(trades) % 100 == 0:
        print(f"  Progress: {len(trades)} trades...")

# Results
if len(trades) == 0:
    print("\nERROR: No trades generated!")
    exit(1)

trades_df = pd.DataFrame(trades)
print(f"\n{'='*80}")
print("MICRO-STRUCTURE MODELS (0.5% TP / 0.15% SL)")
print('='*80)

total = len(trades_df)
wins = (trades_df['pnl'] > 0).sum()
losses = (trades_df['pnl'] < 0).sum()
timeouts = (trades_df['outcome'] == 'TIMEOUT').sum()
win_rate = wins / total * 100

total_pnl = trades_df['pnl'].sum()
avg_pnl = trades_df['pnl'].mean()
avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if wins > 0 else 0
avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losses > 0 else 0

print(f"\nTotal Trades: {total:,}")
print(f"Wins: {wins} ({win_rate:.1f}%)")
print(f"Losses: {losses} ({losses/total*100:.1f}%)")
print(f"Timeouts: {timeouts} ({timeouts/total*100:.1f}%)")
print(f"\nTotal P&L: ${total_pnl:,.2f}")
print(f"Avg P&L/Trade: ${avg_pnl:+.2f}")
print(f"Avg Win: ${avg_win:+.2f}")
print(f"Avg Loss: ${avg_loss:+.2f}")

# By direction
print(f"\n{'='*80}")
print("PERFORMANCE BY DIRECTION:")
print('='*80)
for direction in ['LONG', 'SHORT']:
    dir_trades = trades_df[trades_df['direction'] == direction]
    if len(dir_trades) == 0:
        continue
    dir_total = len(dir_trades)
    dir_wins = (dir_trades['pnl'] > 0).sum()
    dir_wr = dir_wins / dir_total * 100
    dir_avg_pnl = dir_trades['pnl'].mean()
    print(f"\n{direction:5}: {dir_total:,} trades, {dir_wr:.1f}% WR, ${dir_avg_pnl:+.2f} avg P&L")

print(f"\n{'='*80}")
print("COMPARISON:")
print('='*80)
print("\nOriginal 0.3%/0.1% (25 features only):")
print("  Win Rate: 38.4%")
print("  Avg P&L: -$0.62/trade")

print(f"\nWider 0.5%/0.15% (25 features only):")
print(f"  Win Rate: 35.4%")
print(f"  Avg P&L: +$7.63/trade")

print(f"\nMicro-Structure (41 features, 0.5%/0.15%):")
print(f"  Win Rate: {win_rate:.1f}%")
print(f"  Avg P&L: ${avg_pnl:+.2f}/trade")

improvement = avg_pnl - 7.63
print(f"\nImprovement vs wider targets: ${improvement:+.2f}/trade")

print(f"\n{'='*80}")
print("DECISION:")
print('='*80)

if avg_pnl >= 15:
    print(f"✓ EXCELLENT: {win_rate:.1f}% WR, ${avg_pnl:.2f}/trade")
    print("  → DEPLOY micro-structure models")
elif avg_pnl >= 10:
    print(f"✓ GOOD: {win_rate:.1f}% WR, ${avg_pnl:.2f}/trade")
    print("  → Consider deployment")
elif avg_pnl > 7.63:
    print(f"⚠ IMPROVEMENT: {win_rate:.1f}% WR, ${avg_pnl:.2f}/trade")
    print(f"  Better than baseline but below target")
else:
    print(f"⚠ NO IMPROVEMENT: {win_rate:.1f}% WR, ${avg_pnl:.2f}/trade")
    print("  → No significant gain from micro-structure features")

print('='*80)
