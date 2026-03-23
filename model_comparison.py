#!/usr/bin/env python3
"""
Model Comparison: RF vs GBM vs Ensemble at each threshold level.
Uses the same test holdout, bar-by-bar backtest.
"""
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# Reuse prepare_test_data from threshold_analysis
from threshold_analysis import prepare_test_data


def backtest(mnq, test, thresh, tp_pct, sl_pct, prob_col='ens_prob'):
    """Bar-by-bar backtest using a specific probability column."""
    trades = []
    for date in sorted(set(mnq.index.date)):
        dt = pd.Timestamp(date)
        if dt not in test.index:
            continue
        prob = test.loc[dt, prob_col]
        if prob > thresh:
            direction = 1
        elif prob < (1 - thresh):
            direction = -1
        else:
            continue

        day = mnq[mnq.index.date == date]
        if len(day) < 20:
            continue
        post = day.between_time('10:30', '15:55')
        if len(post) < 2:
            continue
        entry = post['Open'].iloc[0]
        ex_price = None
        for _, bar in post.iterrows():
            if direction == 1:
                if bar['High'] / entry - 1 >= tp_pct:
                    ex_price = entry * (1 + tp_pct); break
                elif bar['Low'] / entry - 1 <= -sl_pct:
                    ex_price = entry * (1 - sl_pct); break
            else:
                if bar['Low'] / entry - 1 <= -tp_pct:
                    ex_price = entry * (1 - tp_pct); break
                elif bar['High'] / entry - 1 >= sl_pct:
                    ex_price = entry * (1 + sl_pct); break
        if ex_price is None:
            ex_price = post['Close'].iloc[-1]
        pnl = ((ex_price / entry) - 1) * direction
        trades.append({'pnl': pnl, 'dir': 'L' if direction == 1 else 'S', 'date': date})

    if not trades:
        return None
    df = pd.DataFrame(trades)
    n = len(df)
    wins = (df['pnl'] > 0).sum()
    wr = wins / n
    total = df['pnl'].sum()
    w_sum = df.loc[df['pnl'] > 0, 'pnl'].sum()
    l_sum = abs(df.loc[df['pnl'] < 0, 'pnl'].sum())
    pf = w_sum / l_sum if l_sum > 0 else float('inf')
    sharpe = df['pnl'].mean() / df['pnl'].std() * np.sqrt(252) if df['pnl'].std() > 0 else 0
    mnq_d = total * 40000
    exp = mnq_d / n
    n_l = (df['dir'] == 'L').sum()
    n_s = (df['dir'] == 'S').sum()

    # Max drawdown
    cum = df['pnl'].cumsum()
    max_dd = (cum - cum.cummax()).min()

    # Monthly consistency: how many months profitable?
    df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
    monthly_pnl = df.groupby('month')['pnl'].sum()
    months_pos = (monthly_pnl > 0).sum()
    months_tot = len(monthly_pnl)

    return {'n': n, 'wr': wr, 'pf': pf, 'total': total, 'mnq': mnq_d,
            'sharpe': sharpe, 'exp': exp, 'n_l': n_l, 'n_s': n_s,
            'max_dd': max_dd, 'months_pos': months_pos, 'months_tot': months_tot,
            'trades_df': df}


if __name__ == '__main__':
    mnq, test = prepare_test_data()

    # Probability correlation between models
    print(f"\n{'='*90}")
    print("MODEL PROBABILITY CORRELATION")
    print(f"{'='*90}")
    print(f"\n  GBM vs RF correlation: {test['gbm_prob'].corr(test['rf_prob']):.3f}")
    print(f"  GBM range: {test['gbm_prob'].min():.2%} to {test['gbm_prob'].max():.2%}")
    print(f"  RF  range: {test['rf_prob'].min():.2%} to {test['rf_prob'].max():.2%}")
    print(f"  ENS range: {test['ens_prob'].min():.2%} to {test['ens_prob'].max():.2%}")

    # How often do they disagree?
    for thresh in [0.55, 0.60, 0.65, 0.70]:
        gbm_long = test['gbm_prob'] > thresh
        gbm_short = test['gbm_prob'] < (1 - thresh)
        rf_long = test['rf_prob'] > thresh
        rf_short = test['rf_prob'] < (1 - thresh)
        agree = ((gbm_long & rf_long) | (gbm_short & rf_short) |
                 (~gbm_long & ~gbm_short & ~rf_long & ~rf_short)).sum()
        disagree = len(test) - agree
        print(f"\n  At {thresh:.0%} threshold:")
        print(f"    GBM signals: {(gbm_long|gbm_short).sum()} trades, RF signals: {(rf_long|rf_short).sum()} trades")
        print(f"    Agreement: {agree}/{len(test)} ({agree/len(test):.0%}), Disagreement: {disagree} days")

        # When they disagree, who's right?
        # GBM says trade, RF says no (or vice versa)
        gbm_only = (gbm_long | gbm_short) & ~(rf_long | rf_short)
        rf_only = (rf_long | rf_short) & ~(gbm_long | gbm_short)
        both = (gbm_long & rf_long) | (gbm_short & rf_short)
        print(f"    Both agree to trade: {both.sum()} days")
        print(f"    Only GBM trades: {gbm_only.sum()} days")
        print(f"    Only RF trades: {rf_only.sum()} days")

    # Full sweep: RF vs GBM vs Ensemble
    for tp, sl, tpsl_label in [(0.010, 0.005, "TP 1.0%/SL 0.5%"), (0.015, 0.0075, "TP 1.5%/SL 0.75%")]:
        print(f"\n{'='*90}")
        print(f"RF vs GBM vs ENSEMBLE — {tpsl_label}")
        print(f"{'='*90}")

        for thresh in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
            print(f"\n  Threshold: {thresh:.0%}")
            print(f"  {'Model':12s} {'Trd':>4s} {'L':>3s} {'S':>3s} {'WR':>6s} {'PF':>5s} "
                  f"{'PnL%':>8s} {'$/MNQ':>8s} {'Sharpe':>7s} {'$/trd':>7s} {'MaxDD':>7s} {'Mo+':>4s}")
            print(f"  {'-'*12} {'-'*4} {'-'*3} {'-'*3} {'-'*6} {'-'*5} "
                  f"{'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*4}")

            for prob_col, model_label in [('rf_prob', 'RF'), ('gbm_prob', 'GBM'), ('ens_prob', 'Ensemble')]:
                r = backtest(mnq, test, thresh, tp, sl, prob_col=prob_col)
                if r is None:
                    print(f"  {model_label:12s}   no trades")
                    continue
                print(f"  {model_label:12s} {r['n']:4d} {r['n_l']:3d} {r['n_s']:3d} {r['wr']:6.1%} "
                      f"{r['pf']:5.2f} {r['total']:+8.2%} {r['mnq']:+8,.0f} {r['sharpe']:7.2f} "
                      f"${r['exp']:+6,.0f} {r['max_dd']:+7.2%} {r['months_pos']}/{r['months_tot']}")

    # Final detailed comparison at key thresholds
    print(f"\n{'='*90}")
    print("DETAILED: RF-only at 65% vs Ensemble at 65% — TP 1.0%/SL 0.5%")
    print(f"{'='*90}")

    for prob_col, label in [('rf_prob', 'RF-only'), ('ens_prob', 'Ensemble')]:
        r = backtest(mnq, test, 0.65, 0.010, 0.005, prob_col=prob_col)
        if r is None:
            continue
        df = r['trades_df']
        print(f"\n  {label}:")
        print(f"    Trades: {r['n']}, WR: {r['wr']:.1%}, PF: {r['pf']:.2f}")
        print(f"    Total PnL: {r['total']:+.2%} (${r['mnq']:+,.0f})")
        print(f"    Sharpe: {r['sharpe']:.2f}, MaxDD: {r['max_dd']:+.2%}")
        print(f"    $/trade: ${r['exp']:+,.0f}")
        print(f"    Profitable months: {r['months_pos']}/{r['months_tot']}")

        for d in ['L', 'S']:
            sub = df[df['dir'] == d]
            if len(sub) > 0:
                print(f"    {'LONG' if d=='L' else 'SHORT'}: {len(sub)}T, WR={(sub['pnl']>0).mean():.1%}, PnL={sub['pnl'].sum():+.3%}")

        # Monthly
        df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
        for m, g in df.groupby('month'):
            wr_m = (g['pnl'] > 0).mean()
            print(f"      {m}: {len(g)}T, WR={wr_m:.0%}, PnL={g['pnl'].sum():+.3%} (${g['pnl'].sum()*40000:+,.0f})")

    print(f"\n{'='*90}")
    print("VERDICT")
    print(f"{'='*90}")
