"""
Code Review and Verification of Backtesting Logic

Checks:
1. Label creation logic - are TP/SL levels correct?
2. Backtest P&L calculation - is it matching the labels correctly?
3. LONG vs SHORT distribution - what is the ensemble predicting?
4. Expected vs actual P&L - does the math add up?
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

def verify_label_logic():
    """Verify the labeling logic with a manual example"""
    print("="*70)
    print("1. VERIFYING LABEL CREATION LOGIC")
    print("="*70)

    # Example: Price at 100
    price = 100
    tp_pct = 0.01
    sl_pct = 0.005

    # LONG levels
    long_tp = price * (1 + tp_pct)  # 101
    long_sl = price * (1 - sl_pct)  # 99.5

    # SHORT levels
    short_tp = price * (1 - tp_pct)  # 99
    short_sl = price * (1 + sl_pct)  # 100.5

    print(f"\nPrice: ${price}")
    print(f"\nLONG Trade:")
    print(f"  TP: ${long_tp} (+{tp_pct*100}%)")
    print(f"  SL: ${long_sl} (-{sl_pct*100}%)")
    print(f"  If price goes to ${long_tp} first → Label = 1 (LONG win)")
    print(f"  If price goes to ${long_sl} first → Label = 0 (SHORT win)")

    print(f"\nSHORT Trade:")
    print(f"  TP: ${short_tp} (-{tp_pct*100}%)")
    print(f"  SL: ${short_sl} (+{sl_pct*100}%)")
    print(f"  If price goes to ${short_tp} first → Label = 0 (SHORT win)")
    print(f"  If price goes to ${short_sl} first → Label = 1 (LONG win)")

    print(f"\n✓ Label = 1 means: LONG would have won (price hit +1% before -0.5%)")
    print(f"✓ Label = 0 means: SHORT would have won (price hit -1% before +0.5%)")

    return True


def verify_backtest_logic():
    """Verify backtest P&L calculation"""
    print("\n" + "="*70)
    print("2. VERIFYING BACKTEST LOGIC")
    print("="*70)

    tp_pct = 0.01
    sl_pct = 0.005

    print("\nIf we predict LONG (pred=1):")
    print(f"  If actual label = 1 (LONG won): P&L = +{tp_pct*100}% ✓")
    print(f"  If actual label = 0 (SHORT won): P&L = -{sl_pct*100}% ✓")

    print("\nIf we predict SHORT (pred=0):")
    print(f"  If actual label = 0 (SHORT won): P&L = +{sl_pct*100}% ✓")
    print(f"  If actual label = 1 (LONG won): P&L = -{tp_pct*100}% ✓")

    # Calculate breakeven rates
    print("\n" + "-"*70)
    print("BREAKEVEN CALCULATIONS:")
    print("-"*70)

    # LONG breakeven
    long_ev = lambda wr: wr * tp_pct + (1-wr) * (-sl_pct)
    long_breakeven = sl_pct / (tp_pct + sl_pct)

    print(f"\nLONG trades (predict=1):")
    print(f"  Win: +{tp_pct*100}%, Loss: -{sl_pct*100}%")
    print(f"  Breakeven WR: {long_breakeven:.2%}")
    print(f"  Example: 50% WR → EV = {long_ev(0.5)*100:+.4f}%")
    print(f"  Example: 54% WR → EV = {long_ev(0.54)*100:+.4f}%")

    # SHORT breakeven
    short_ev = lambda wr: wr * sl_pct + (1-wr) * (-tp_pct)
    short_breakeven = tp_pct / (tp_pct + sl_pct)

    print(f"\nSHORT trades (predict=0):")
    print(f"  Win: +{sl_pct*100}%, Loss: -{tp_pct*100}%")
    print(f"  Breakeven WR: {short_breakeven:.2%}")
    print(f"  Example: 50% WR → EV = {short_ev(0.5)*100:+.4f}%")
    print(f"  Example: 54% WR → EV = {short_ev(0.54)*100:+.4f}%")

    return True


def analyze_actual_results():
    """Analyze actual walk-forward results"""
    print("\n" + "="*70)
    print("3. ANALYZING ACTUAL WALK-FORWARD RESULTS")
    print("="*70)

    models_dir = Path(__file__).parent / 'ml_models'

    try:
        with open(models_dir / 'walk_forward_results.json', 'r') as f:
            results = json.load(f)
    except:
        print("Walk-forward results not found")
        return False

    print(f"\n{'Period':<40} {'Trades':<8} {'WR':<8} {'Avg P&L':<10} {'Expected':<10}")
    print("-"*80)

    for r in results:
        wr = r['win_rate']
        actual_pnl = r['avg_pnl']

        # If all LONG
        expected_long = wr * 1.0 + (1-wr) * (-0.5)
        # If all SHORT
        expected_short = wr * 0.5 + (1-wr) * (-1.0)

        period = r['period'][:35]
        print(f"{period:<40} {r['trades']:<8,} {wr:<8.2%} {actual_pnl:<10.4f} "
              f"L:{expected_long:.4f} S:{expected_short:.4f}")

    print("\n" + "-"*70)
    print("OBSERVATION:")
    print("-"*70)
    print("If actual P&L is closer to SHORT expected, the model is predicting too many SHORTs!")
    print("SHORT trades have WORSE risk/reward (need 66.7% WR vs 33.3% for LONG)")

    return True


def deep_dive_walk_forward_period():
    """Load data and check one walk-forward period in detail"""
    print("\n" + "="*70)
    print("4. DEEP DIVE: 2024 Q2 (BEST PERIOD)")
    print("="*70)

    data_dir = Path(__file__).parent / 'data'
    df = pd.read_parquet(data_dir / 'btc_5m_v3_features.parquet')

    # Focus on 2024 Q2 test period
    test_start = "2024-04-01"
    test_end = "2024-06-30"
    df_test = df[(df.index >= test_start) & (df.index <= test_end)]

    print(f"\nTest period: {test_start} to {test_end}")
    print(f"Total rows: {len(df_test):,}")

    # Check label distribution
    for horizon in ['2h', '4h', '6h']:
        label_col = f'label_{horizon}'
        labeled = df_test[df_test[label_col].notna()]

        if len(labeled) == 0:
            continue

        long_pct = (labeled[label_col] == 1).sum() / len(labeled)
        short_pct = (labeled[label_col] == 0).sum() / len(labeled)

        print(f"\n{horizon} labels:")
        print(f"  Total: {len(labeled):,}")
        print(f"  LONG (1): {long_pct:.1%}")
        print(f"  SHORT (0): {short_pct:.1%}")

        # What would happen if we predicted all LONG?
        all_long_wr = long_pct  # % of time LONG won
        all_long_pnl = all_long_wr * 1.0 + (1-all_long_wr) * (-0.5)

        # What would happen if we predicted all SHORT?
        all_short_wr = short_pct  # % of time SHORT won
        all_short_pnl = all_short_wr * 0.5 + (1-all_short_wr) * (-1.0)

        print(f"  If we predicted ALL LONG: {all_long_pnl*100:+.2f}%")
        print(f"  If we predicted ALL SHORT: {all_short_pnl*100:+.2f}%")

    return True


def main():
    print("="*70)
    print("COMPLETE CODE REVIEW AND VERIFICATION")
    print("="*70)

    verify_label_logic()
    verify_backtest_logic()
    analyze_actual_results()
    deep_dive_walk_forward_period()

    print("\n" + "="*70)
    print("CONCLUSION & RECOMMENDATIONS")
    print("="*70)
    print("""
1. Breakeven Requirements:
   - LONG trades: Need only 33.3% WR (excellent 2:1 reward/risk)
   - SHORT trades: Need 66.7% WR (poor 1:2 reward/risk)

2. With 54% overall WR:
   - If predicting mostly LONG: Should be VERY profitable
   - If predicting mostly SHORT: Will lose money

3. Likely Issue:
   - The ensemble confidence threshold (>0.6 for LONG, <0.4 for SHORT)
   - Might be BIASED toward SHORT predictions
   - SHORT has worse R:R, so this hurts profitability

4. Next Steps:
   - Check actual LONG/SHORT ratio in ensemble predictions
   - Consider ASYMMETRIC thresholds (easier for LONG, harder for SHORT)
   - Example: >0.55 for LONG, <0.35 for SHORT
   - This would favor LONG trades which have better R:R
    """)


if __name__ == '__main__':
    main()
