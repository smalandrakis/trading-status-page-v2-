"""Quick verification of walk-forward backtest correctness."""
import pandas as pd
import numpy as np
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from btc_model_package.predictor import BTCPredictor

df_raw = pd.read_parquet('data/BTC_5min_8years.parquet')
df_raw.columns = [c.lower() for c in df_raw.columns]

# 1. Data quality
print("=== DATA QUALITY ===")
print(f"Shape: {df_raw.shape}")
print(f"Range: {df_raw.index[0]} to {df_raw.index[-1]}")
print(f"Index dtype: {df_raw.index.dtype}")
print(f"Has .hour: {hasattr(df_raw.index, 'hour')}")
if hasattr(df_raw.index, 'hour'):
    print(f"Sample hour: {df_raw.index[100].hour}")
else:
    print("WARNING: No hour attr -> hour feature defaults to 12!")
print(f"NaN count: {df_raw.isnull().sum().sum()}")
print(f"Sample:\n{df_raw.tail(3)}")
print()

# 2. Signal distribution on 20 dates
predictor = BTCPredictor(model_dir='btc_model_package')
print("=== SIGNAL CHECK (20 dates) ===")
test_dates = pd.date_range('2024-06-01', '2025-06-01', freq='14D')
sigs = {'LONG': 0, 'SHORT': 0, 'NEUTRAL': 0}
for d in test_dates:
    w = df_raw.loc[:str(d)].tail(250)
    if len(w) < 200:
        continue
    s, c, det = predictor.predict(w)
    sigs[s] += 1
    print(f"  {d.date()} ${w.close.iloc[-1]:.0f} -> {s} conf={c:.3f} avg_p={det['avg_probability']:.3f}")
print(f"Distribution: {sigs}")
print()

# 3. Manual trade verification
print("=== MANUAL TRADE EXAMPLE ===")
w = df_raw.loc[:'2025-03-15'].tail(250)
s, c, det = predictor.predict(w)
entry_price = w.close.iloc[-1]
entry_time = w.index[-1]
print(f"Entry: {entry_time}, ${entry_price:.2f}, signal={s}, conf={c:.3f}")

forward = df_raw.loc[entry_time:].iloc[1:74]  # skip entry bar
if s in ('LONG', 'SHORT'):
    if s == 'LONG':
        tp = entry_price * 1.01
        sl = entry_price * 0.995
    else:
        tp = entry_price * 0.99
        sl = entry_price * 1.005
    print(f"TP=${tp:.2f}, SL=${sl:.2f}")
    
    for i in range(len(forward)):
        bar = forward.iloc[i]
        if s == 'LONG':
            hit_sl = bar['low'] <= sl
            hit_tp = bar['high'] >= tp
        else:
            hit_sl = bar['high'] >= sl
            hit_tp = bar['low'] <= tp
        
        marker = ''
        if hit_sl: marker = ' <-- SL'
        if hit_tp: marker = ' <-- TP'
        if marker:
            pnl = (1.0 if 'TP' in marker else -0.5)
            print(f"  Bar {i+1}: {forward.index[i]} H={bar.high:.2f} L={bar.low:.2f} C={bar.close:.2f}{marker}")
            print(f"  Result: {marker.strip()} after {i+1} bars ({(i+1)*5} min)")
            break
    else:
        last = forward.iloc[-1]
        if s == 'LONG':
            pnl = (last.close / entry_price - 1) * 100
        else:
            pnl = (entry_price / last.close - 1) * 100
        print(f"  TIMEOUT after {len(forward)} bars, close=${last.close:.2f}, pnl={pnl:+.2f}%")
print()

# 4. Check the saved trade results for consistency
print("=== VERIFY SAVED RESULTS ===")
trades = pd.read_csv('results/v3_predictor_walkforward.csv')
print(f"Total trades: {len(trades)}")
print(f"Direction counts:\n{trades['direction'].value_counts()}")
print(f"Exit reason counts:\n{trades['exit_reason'].value_counts()}")

# Verify P&L calc: for TP trades, pnl_pct should be ~+1.0
tp_trades = trades[trades['exit_reason'] == 'TAKE_PROFIT']
sl_trades = trades[trades['exit_reason'] == 'STOP_LOSS']
print(f"\nTP trades avg pnl_pct: {tp_trades['pnl_pct'].mean():.4f}% (expected ~1.0)")
print(f"SL trades avg pnl_pct: {sl_trades['pnl_pct'].mean():.4f}% (expected ~-0.5)")

# Check for any anomalous pnl values
print(f"\nPnl_pct range: {trades['pnl_pct'].min():.3f}% to {trades['pnl_pct'].max():.3f}%")
print(f"Pnl_dollar range: ${trades['pnl_dollar'].min():.2f} to ${trades['pnl_dollar'].max():.2f}")

# Verify: WR above breakeven?
wr = (trades['pnl_pct'] > 0).mean() * 100
breakeven = 0.5 / (1.0 + 0.5) * 100
print(f"\nWR: {wr:.1f}% vs breakeven: {breakeven:.1f}%")
print(f"Edge (WR - BE): {wr - breakeven:+.1f}pp")

# Gross P&L without commissions
gross = trades['pnl_pct'].sum() / 100 * trades['entry_price'].mean() * 0.1
commissions = len(trades) * 2.02
print(f"\nGross P&L (no commissions): ${gross:+,.2f}")
print(f"Total commissions: ${commissions:,.2f}")
print(f"Net P&L: ${gross - commissions:+,.2f}")
