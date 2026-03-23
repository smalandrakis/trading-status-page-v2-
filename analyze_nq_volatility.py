#!/usr/bin/env python3
"""
NQ Volatility Analysis + Trade Frequency + TP/SL Optimization
=============================================================
1. How volatile is NQ on a 5-min basis? (explains SL clipping)
2. How many trades per day does the strategy generate?
3. Simulate TP 1% / SL 0.5% vs other configs on real intraday data
"""

import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "data"
MNQ_FILE = f"{DATA_DIR}/MNQ_5min_IB_with_indicators.csv"

# ═══════════════════════════════════════════════════════════════════════
# PART 1: NQ 5-MIN VOLATILITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
def analyze_volatility(mnq):
    print("=" * 70)
    print("PART 1: NQ 5-MIN BAR VOLATILITY")
    print("=" * 70)
    
    # Per-bar stats
    mnq['bar_range_pct'] = (mnq['High'] - mnq['Low']) / mnq['Open'] * 100
    mnq['bar_ret_pct'] = (mnq['Close'] / mnq['Open'] - 1) * 100
    mnq['bar_abs_ret'] = mnq['bar_ret_pct'].abs()
    
    print(f"\n  5-min bar range (High-Low as % of Open):")
    print(f"    Mean:   {mnq['bar_range_pct'].mean():.3f}%")
    print(f"    Median: {mnq['bar_range_pct'].median():.3f}%")
    print(f"    P75:    {mnq['bar_range_pct'].quantile(0.75):.3f}%")
    print(f"    P90:    {mnq['bar_range_pct'].quantile(0.90):.3f}%")
    print(f"    P95:    {mnq['bar_range_pct'].quantile(0.95):.3f}%")
    print(f"    P99:    {mnq['bar_range_pct'].quantile(0.99):.3f}%")
    print(f"    Max:    {mnq['bar_range_pct'].max():.3f}%")
    
    print(f"\n  5-min bar |return| (|Close-Open|/Open):")
    print(f"    Mean:   {mnq['bar_abs_ret'].mean():.3f}%")
    print(f"    Median: {mnq['bar_abs_ret'].median():.3f}%")
    print(f"    P75:    {mnq['bar_abs_ret'].quantile(0.75):.3f}%")
    print(f"    P90:    {mnq['bar_abs_ret'].quantile(0.90):.3f}%")
    print(f"    P95:    {mnq['bar_abs_ret'].quantile(0.95):.3f}%")
    
    # How often does a SINGLE 5-min bar move > X%?
    print(f"\n  How often does a single 5-min bar move (range) exceed:")
    for thresh in [0.10, 0.15, 0.20, 0.25, 0.30, 0.50]:
        pct = (mnq['bar_range_pct'] > thresh).mean()
        print(f"    > {thresh:.2f}%: {pct:.1%} of bars ({(mnq['bar_range_pct'] > thresh).sum()} bars)")
    
    # Max adverse excursion from entry over N bars
    print(f"\n  Max Adverse Excursion (MAE) from entry over next N bars:")
    print(f"  (i.e., if you enter LONG at bar open, what's the worst drawdown?)\n")
    
    closes = mnq['Close'].values
    lows = mnq['Low'].values
    highs = mnq['High'].values
    opens = mnq['Open'].values
    
    for n_bars, minutes in [(1, 5), (3, 15), (6, 30), (12, 60), (24, 120)]:
        # For LONG: MAE = min(Low[i:i+n]) / Open[i] - 1
        maes = []
        for i in range(len(mnq) - n_bars):
            entry = opens[i]
            if entry == 0:
                continue
            worst_low = lows[i:i+n_bars].min()
            mae = (worst_low / entry - 1) * 100
            maes.append(mae)
        
        maes = np.array(maes)
        print(f"    {minutes:3d} min ({n_bars:2d} bars): "
              f"mean MAE={maes.mean():.3f}%, "
              f"median={np.median(maes):.3f}%, "
              f"P10={np.percentile(maes, 10):.3f}%, "
              f"P5={np.percentile(maes, 5):.3f}%")
    
    # Same but focusing on post-10:30 entries only
    print(f"\n  MAE from 10:30 ET entries (post first-hour):")
    mnq_post = mnq.between_time('10:30', '15:00')
    closes_p = mnq_post['Close'].values
    lows_p = mnq_post['Low'].values
    opens_p = mnq_post['Open'].values
    
    for n_bars, minutes in [(6, 30), (12, 60), (24, 120), (66, 330)]:
        maes = []
        for i in range(len(mnq_post) - n_bars):
            entry = opens_p[i]
            if entry == 0:
                continue
            worst_low = lows_p[i:i+n_bars].min()
            mae = (worst_low / entry - 1) * 100
            maes.append(mae)
        
        maes = np.array(maes)
        pct_hit_025 = (maes < -0.25).mean()
        pct_hit_050 = (maes < -0.50).mean()
        print(f"    {minutes:3d} min: mean MAE={maes.mean():.3f}%, "
              f"hit -0.25%={pct_hit_025:.1%}, hit -0.50%={pct_hit_050:.1%}")
    
    print(f"\n  CONCLUSION: A 0.25% SL gets hit {pct_hit_025:.0%} of the time within the "
          f"trading session just from random noise.")
    print(f"  A 0.50% SL gets hit {pct_hit_050:.0%} of the time — much more selective.")


# ═══════════════════════════════════════════════════════════════════════
# PART 2: TRADE FREQUENCY
# ═══════════════════════════════════════════════════════════════════════
def analyze_trade_frequency(mnq):
    print(f"\n{'='*70}")
    print("PART 2: EXPECTED TRADES PER DAY")
    print("=" * 70)
    
    print(f"\n  This strategy trades MAX 1 trade per day.")
    print(f"  The question is: how many days produce a signal?\n")
    
    # Download signals
    tickers = ['^N225', '^HSI', '^AXJO', '^GDAXI', '^FTSE', '^STOXX50E']
    print("  Downloading Asia/Europe data for signal frequency...")
    
    returns = pd.DataFrame()
    for ticker in tickers:
        df = yf.download(ticker, period='1y', interval='1d', progress=False)
        if len(df) > 0:
            col = ticker.replace('^', '')
            if isinstance(df.columns, pd.MultiIndex):
                close = df[('Close', ticker)]
            else:
                close = df['Close']
            returns[col] = close.pct_change()
    
    returns = returns.dropna()
    
    asia_cols = [c for c in returns.columns if c in ['N225', 'HSI', 'AXJO']]
    europe_cols = [c for c in returns.columns if c in ['GDAXI', 'FTSE', 'STOXX50E']]
    
    combined = 0.4 * returns[asia_cols].mean(axis=1) + 0.6 * returns[europe_cols].mean(axis=1)
    
    total_days = len(combined)
    
    print(f"\n  Total trading days in sample: {total_days}")
    print(f"\n  Signal threshold  |  Days with signal  |  % of days  |  Trades/week")
    print(f"  {'-'*16} | {'-'*18} | {'-'*11} | {'-'*12}")
    
    for thresh in [0.000, 0.001, 0.002, 0.003, 0.004, 0.005]:
        signal_days = (combined.abs() > thresh).sum()
        pct = signal_days / total_days
        per_week = pct * 5
        print(f"  {thresh:.3f} ({thresh*100:.1f}%)      | {signal_days:18d} | {pct:11.1%} | {per_week:8.1f}")
    
    print(f"\n  At 0.2% threshold: ~{(combined.abs() > 0.002).mean()*5:.1f} trades/week")
    print(f"  At 0.3% threshold: ~{(combined.abs() > 0.003).mean()*5:.1f} trades/week")
    print(f"  This is a ONCE-A-DAY strategy (max). Most days trigger.")


# ═══════════════════════════════════════════════════════════════════════
# PART 3: TP 1% / SL 0.5% AND OTHER CONFIGS — INTRADAY BACKTEST
# ═══════════════════════════════════════════════════════════════════════
def simulate_intraday_config(mnq, signals, tp_pct, sl_pct, threshold=0.002):
    """Bar-by-bar backtest on MNQ 5-min data."""
    trades = []
    
    for date in sorted(set(mnq.index.date)):
        date_ts = pd.Timestamp(date)
        if date_ts not in signals.index:
            continue
        
        sig = signals.loc[date_ts, 'combined_ret']
        if abs(sig) < threshold:
            continue
        
        direction = 1 if sig > 0 else -1
        
        day_bars = mnq[mnq.index.date == date]
        rth = day_bars.between_time('09:30', '16:00')
        if len(rth) < 17:
            continue
        
        # Entry at 10:30
        post_fh = rth.iloc[12:]
        if len(post_fh) < 2:
            continue
        
        entry_price = post_fh['Open'].iloc[0]
        fh_ret = (rth['Close'].iloc[11] / rth['Open'].iloc[0]) - 1
        
        exit_price = None
        exit_reason = None
        bars_held = 0
        
        for ts, bar in post_fh.iterrows():
            bars_held += 1
            if direction == 1:
                if (bar['High'] / entry_price - 1) >= tp_pct:
                    exit_price = entry_price * (1 + tp_pct)
                    exit_reason = 'TP'
                    break
                elif (bar['Low'] / entry_price - 1) <= -sl_pct:
                    exit_price = entry_price * (1 - sl_pct)
                    exit_reason = 'SL'
                    break
            else:
                if (bar['Low'] / entry_price - 1) <= -tp_pct:
                    exit_price = entry_price * (1 - tp_pct)
                    exit_reason = 'TP'
                    break
                elif (bar['High'] / entry_price - 1) >= sl_pct:
                    exit_price = entry_price * (1 + sl_pct)
                    exit_reason = 'SL'
                    break
        
        if exit_price is None:
            exit_price = post_fh['Close'].iloc[-1]
            exit_reason = 'CLOSE'
        
        pnl = ((exit_price / entry_price) - 1) * direction
        hold_min = bars_held * 5
        
        trades.append({
            'date': date,
            'direction': 'LONG' if direction == 1 else 'SHORT',
            'signal': sig,
            'pnl': pnl,
            'exit_reason': exit_reason,
            'bars_held': bars_held,
            'hold_min': hold_min,
        })
    
    return pd.DataFrame(trades) if trades else None


def run_config_comparison(mnq, signals):
    print(f"\n{'='*70}")
    print("PART 3: TP/SL CONFIGURATION COMPARISON (INTRADAY 5-MIN MNQ)")
    print("=" * 70)
    
    configs = [
        # (tp, sl, label)
        (0.005, 0.005, "TP 0.5% / SL 0.5% (1:1)"),
        (0.010, 0.005, "TP 1.0% / SL 0.5% (1:2)"),
        (0.0075, 0.005, "TP 0.75%/ SL 0.5% (1:1.5)"),
        (0.010, 0.0075,"TP 1.0% / SL 0.75%(1:1.3)"),
        (0.010, 0.010, "TP 1.0% / SL 1.0% (1:1)"),
        (0.005, 0.0025,"TP 0.5% / SL 0.25%(1:2)*"),
        (0.015, 0.005, "TP 1.5% / SL 0.5% (1:3)"),
        (0.010, 0.003, "TP 1.0% / SL 0.3% (1:3.3)*"),
    ]
    
    print(f"\n  {'Config':28s} {'Trades':>6s} {'WR':>6s} {'TP':>4s} {'SL':>4s} {'CL':>4s} "
          f"{'PnL%':>8s} {'$/MNQ':>8s} {'PF':>5s} {'AvgHold':>8s} {'Exp$/tr':>8s}")
    print(f"  {'-'*28} {'-'*6} {'-'*6} {'-'*4} {'-'*4} {'-'*4} "
          f"{'-'*8} {'-'*8} {'-'*5} {'-'*8} {'-'*8}")
    
    for tp, sl, label in configs:
        df = simulate_intraday_config(mnq, signals, tp, sl)
        if df is None or len(df) == 0:
            print(f"  {label:28s}   no trades")
            continue
        
        n = len(df)
        wins = (df['pnl'] > 0).sum()
        losses = (df['pnl'] < 0).sum()
        wr = wins / n
        total = df['pnl'].sum()
        mnq_pnl = total * 20000 * 2
        
        tp_hits = (df['exit_reason'] == 'TP').sum()
        sl_hits = (df['exit_reason'] == 'SL').sum()
        cl_hits = (df['exit_reason'] == 'CLOSE').sum()
        
        win_sum = df.loc[df['pnl'] > 0, 'pnl'].sum()
        loss_sum = abs(df.loc[df['pnl'] < 0, 'pnl'].sum())
        pf = win_sum / loss_sum if loss_sum > 0 else float('inf')
        
        avg_hold = df['hold_min'].mean()
        exp_per_trade = total * 20000 * 2 / n
        
        print(f"  {label:28s} {n:6d} {wr:6.1%} {tp_hits:4d} {sl_hits:4d} {cl_hits:4d} "
              f"{total:+8.2%} {mnq_pnl:+8,.0f} {pf:5.2f} {avg_hold:6.0f}min ${exp_per_trade:+7,.0f}")
    
    # Deep dive on TP 1% / SL 0.5%
    print(f"\n{'─'*70}")
    print(f"  DEEP DIVE: TP 1.0% / SL 0.5% (1:2 R:R)")
    print(f"{'─'*70}")
    
    df = simulate_intraday_config(mnq, signals, 0.010, 0.005)
    if df is not None and len(df) > 0:
        n = len(df)
        wins = (df['pnl'] > 0).sum()
        wr = wins / n
        
        print(f"\n  Trades: {n}, Wins: {wins}, WR: {wr:.1%}")
        print(f"  Total PnL: {df['pnl'].sum():+.2%} (${df['pnl'].sum() * 40000:+,.0f}/MNQ)")
        
        # By direction
        for d in ['LONG', 'SHORT']:
            sub = df[df['direction'] == d]
            if len(sub) > 0:
                sub_wr = (sub['pnl'] > 0).mean()
                print(f"\n  {d}: {len(sub)} trades, WR={sub_wr:.1%}, PnL={sub['pnl'].sum():+.3%}")
                for reason in ['TP', 'SL', 'CLOSE']:
                    r = sub[sub['exit_reason'] == reason]
                    if len(r) > 0:
                        print(f"    {reason}: {len(r)} ({len(r)/len(sub):.0%}), avg PnL={r['pnl'].mean():+.4%}, avg hold={r['hold_min'].mean():.0f}min")
        
        # How long to hit TP 1%?
        tp_trades = df[df['exit_reason'] == 'TP']
        if len(tp_trades) > 0:
            print(f"\n  When TP 1% hits:")
            print(f"    Count: {len(tp_trades)} ({len(tp_trades)/n:.0%} of trades)")
            print(f"    Avg time to hit: {tp_trades['hold_min'].mean():.0f} min")
            print(f"    Median time: {tp_trades['hold_min'].median():.0f} min")
        
        sl_trades = df[df['exit_reason'] == 'SL']
        if len(sl_trades) > 0:
            print(f"\n  When SL 0.5% hits:")
            print(f"    Count: {len(sl_trades)} ({len(sl_trades)/n:.0%} of trades)")
            print(f"    Avg time to hit: {sl_trades['hold_min'].mean():.0f} min")
            print(f"    Median time: {sl_trades['hold_min'].median():.0f} min")
        
        cl_trades = df[df['exit_reason'] == 'CLOSE']
        if len(cl_trades) > 0:
            print(f"\n  Held to close:")
            print(f"    Count: {len(cl_trades)} ({len(cl_trades)/n:.0%} of trades)")
            print(f"    Avg PnL: {cl_trades['pnl'].mean():+.4%}")
            print(f"    WR: {(cl_trades['pnl'] > 0).mean():.1%}")
        
        # Monthly
        df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
        monthly = df.groupby('month').agg(
            trades=('pnl', 'count'),
            wins=('pnl', lambda x: (x > 0).sum()),
            pnl=('pnl', 'sum')
        )
        print(f"\n  Monthly (TP 1% / SL 0.5%):")
        for m, row in monthly.iterrows():
            wr_m = row['wins'] / row['trades'] if row['trades'] > 0 else 0
            print(f"    {m}: {row['trades']:2.0f}T, WR={wr_m:.0%}, PnL={row['pnl']:+.3%} (${row['pnl']*40000:+,.0f})")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    # Load MNQ
    print("Loading MNQ 5-min data...")
    mnq = pd.read_csv(MNQ_FILE, usecols=['date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    mnq['date'] = pd.to_datetime(mnq['date'], utc=True)
    mnq['date'] = mnq['date'].dt.tz_convert('America/New_York')
    mnq = mnq.set_index('date').sort_index()
    mnq_rth = mnq.between_time('09:30', '16:00')
    
    print(f"  {len(mnq_rth)} RTH bars, {len(set(mnq_rth.index.date))} trading days")
    print(f"  {mnq_rth.index[0]} to {mnq_rth.index[-1]}")
    
    # Part 1: Volatility
    analyze_volatility(mnq_rth.copy())
    
    # Part 2: Trade frequency
    analyze_trade_frequency(mnq_rth)
    
    # Part 3: Build signals and run config comparison
    print(f"\n  Building signals for config comparison...")
    tickers = ['^N225', '^HSI', '^AXJO', '^GDAXI', '^FTSE', '^STOXX50E']
    returns = pd.DataFrame()
    for ticker in tickers:
        df = yf.download(ticker, period='2y', interval='1d', progress=False)
        if len(df) > 0:
            col = ticker.replace('^', '')
            if isinstance(df.columns, pd.MultiIndex):
                close = df[('Close', ticker)]
            else:
                close = df['Close']
            returns[col] = close.pct_change()
    returns = returns.dropna()
    
    asia_cols = [c for c in returns.columns if c in ['N225', 'HSI', 'AXJO']]
    europe_cols = [c for c in returns.columns if c in ['GDAXI', 'FTSE', 'STOXX50E']]
    
    signals = pd.DataFrame(index=returns.index)
    signals['combined_ret'] = 0.4 * returns[asia_cols].mean(axis=1) + 0.6 * returns[europe_cols].mean(axis=1)
    signals.index = signals.index.tz_localize(None)
    
    run_config_comparison(mnq_rth, signals)
