#!/usr/bin/env python3
"""
Quick simulation: 1:2 R:R ratio on the global session signal.
Compares 1:1 (0.5% SL / 0.5% TP) vs 1:2 (0.25% SL / 0.5% TP) vs other ratios.
Uses both daily data and 5-min intraday NQ data.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

PERIOD = "2y"
ASIA_TICKERS = ['^N225', '^HSI', '^AXJO']
EUROPE_TICKERS = ['^GDAXI', '^FTSE', '^STOXX50E']

def download_and_build_signals():
    """Download data and build composite signal (same as main analysis)."""
    print("Downloading daily data...")
    data = {}
    for ticker in ASIA_TICKERS + EUROPE_TICKERS + ['NQ=F']:
        df = yf.download(ticker, period=PERIOD, interval='1d', progress=False)
        if len(df) > 0:
            data[ticker] = df
    
    # Build returns
    returns = pd.DataFrame()
    for ticker, df in data.items():
        col = ticker.replace('^', '').replace('=', '_')
        if isinstance(df.columns, pd.MultiIndex):
            close = df[('Close', ticker)]
        else:
            close = df['Close']
        returns[col] = close.pct_change()
    returns = returns.dropna()
    
    asia_cols = [c for c in returns.columns if c in ['N225', 'HSI', 'AXJO']]
    europe_cols = [c for c in returns.columns if c in ['GDAXI', 'FTSE', 'STOXX50E']]
    
    signals = pd.DataFrame(index=returns.index)
    signals['asia_ret'] = returns[asia_cols].mean(axis=1)
    signals['europe_ret'] = returns[europe_cols].mean(axis=1)
    signals['combined_ret'] = 0.4 * signals['asia_ret'] + 0.6 * signals['europe_ret']
    signals['nq_ret'] = returns['NQ_F']
    
    return signals


def simulate_daily_rr(signals, tp_pct, sl_pct, threshold=0.002, label=""):
    """
    Simulate with specific TP/SL using daily data.
    Approximation: assume intraday can hit either TP or SL.
    If daily |move| > TP in our direction → book TP
    If daily |move| > SL against us → book -SL
    Otherwise → book actual return (held to close)
    """
    trades = []
    for idx, row in signals.iterrows():
        sig = row['combined_ret']
        nq = row['nq_ret']
        if abs(sig) < threshold:
            continue
        
        direction = 1 if sig > 0 else -1
        raw_pnl = nq * direction
        
        if raw_pnl >= tp_pct:
            booked = tp_pct
            outcome = 'TP'
        elif raw_pnl <= -sl_pct:
            booked = -sl_pct
            outcome = 'SL'
        else:
            booked = raw_pnl
            outcome = 'CLOSE'
        
        trades.append({
            'date': idx,
            'direction': 'LONG' if direction == 1 else 'SHORT',
            'signal': sig,
            'raw_pnl': raw_pnl,
            'booked_pnl': booked,
            'outcome': outcome
        })
    
    if not trades:
        return None
    
    df = pd.DataFrame(trades)
    n = len(df)
    wins = (df['booked_pnl'] > 0).sum()
    losses = (df['booked_pnl'] < 0).sum()
    wr = wins / n
    total = df['booked_pnl'].sum()
    avg_win = df.loc[df['booked_pnl'] > 0, 'booked_pnl'].mean() if wins > 0 else 0
    avg_loss = df.loc[df['booked_pnl'] < 0, 'booked_pnl'].mean() if losses > 0 else 0
    pf = abs(df.loc[df['booked_pnl'] > 0, 'booked_pnl'].sum() / df.loc[df['booked_pnl'] < 0, 'booked_pnl'].sum()) if losses > 0 else float('inf')
    
    # Max drawdown
    df['cum'] = df['booked_pnl'].cumsum()
    max_dd = (df['cum'] - df['cum'].cummax()).min()
    sharpe = df['booked_pnl'].mean() / df['booked_pnl'].std() * np.sqrt(252) if df['booked_pnl'].std() > 0 else 0
    
    # Expectancy per trade
    expectancy = df['booked_pnl'].mean()
    
    # MNQ $$
    nq_price = 20000
    mnq_pt = 2.0
    pnl_dollars = total * nq_price * mnq_pt
    
    return {
        'label': label,
        'tp': tp_pct, 'sl': sl_pct, 'rr': tp_pct / sl_pct,
        'trades': n, 'wins': wins, 'losses': losses,
        'wr': wr, 'avg_win': avg_win, 'avg_loss': avg_loss,
        'pf': pf, 'total_pnl': total, 'expectancy': expectancy,
        'max_dd': max_dd, 'sharpe': sharpe, 'pnl_mnq': pnl_dollars,
        'tp_hits': (df['outcome'] == 'TP').sum(),
        'sl_hits': (df['outcome'] == 'SL').sum(),
        'close_hits': (df['outcome'] == 'CLOSE').sum(),
        'df': df
    }


def simulate_intraday_rr(nq_5m, signals, tp_pct, sl_pct, threshold=0.002, label=""):
    """Proper bar-by-bar intraday simulation with specific TP/SL."""
    if nq_5m.index.tz is None:
        nq_5m.index = nq_5m.index.tz_localize('America/New_York')
    else:
        nq_5m.index = nq_5m.index.tz_convert('America/New_York')
    
    if isinstance(nq_5m.columns, pd.MultiIndex):
        nq_5m.columns = nq_5m.columns.get_level_values(0)
    
    nq_5m['date'] = nq_5m.index.date
    trades = []
    
    for date, day_data in nq_5m.groupby('date'):
        sig_matches = signals.index[signals.index.date == date]
        if len(sig_matches) == 0:
            sig_matches = signals.index[signals.index.date < date]
            if len(sig_matches) == 0:
                continue
            date_ts = sig_matches[-1]
        else:
            date_ts = sig_matches[0]
        
        sig_val = signals.loc[date_ts, 'combined_ret']
        if abs(sig_val) < threshold:
            continue
        
        direction = 1 if sig_val > 0 else -1
        
        rth = day_data.between_time('09:30', '16:00')
        if len(rth) < 17:
            continue
        
        # Entry after first hour (bar 12 = 10:30 ET)
        post_fh = rth.iloc[12:]
        if len(post_fh) < 2:
            continue
        
        entry_price = post_fh['Open'].iloc[0]
        fh_ret = (rth['Close'].iloc[11] / rth['Open'].iloc[0]) - 1
        fh_confirms = (fh_ret > 0 and direction == 1) or (fh_ret < 0 and direction == -1)
        
        exit_price = None
        exit_reason = None
        
        for ts, bar in post_fh.iterrows():
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
        
        trades.append({
            'date': date, 'direction': 'LONG' if direction == 1 else 'SHORT',
            'signal': sig_val, 'fh_confirms': fh_confirms,
            'entry': entry_price, 'exit': exit_price,
            'exit_reason': exit_reason, 'pnl': pnl
        })
    
    if not trades:
        return None
    
    df = pd.DataFrame(trades)
    n = len(df)
    wins = (df['pnl'] > 0).sum()
    wr = wins / n
    total = df['pnl'].sum()
    pnl_dollars = total * 20000 * 2.0
    
    return {
        'label': label, 'tp': tp_pct, 'sl': sl_pct, 'rr': tp_pct / sl_pct,
        'trades': n, 'wins': wins, 'wr': wr,
        'total_pnl': total, 'pnl_mnq': pnl_dollars,
        'tp_hits': (df['exit_reason'] == 'TP').sum(),
        'sl_hits': (df['exit_reason'] == 'SL').sum(),
        'close_hits': (df['exit_reason'] == 'CLOSE').sum(),
        'confirmed_wr': df.loc[df['fh_confirms'], 'pnl'].apply(lambda x: x > 0).mean() if df['fh_confirms'].sum() > 0 else 0,
        'confirmed_n': df['fh_confirms'].sum(),
        'df': df
    }


def print_comparison(results, title):
    """Print comparison table."""
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")
    print(f"  {'Config':22s} {'R:R':>4s} {'Trades':>6s} {'WR':>6s} {'TP':>4s} {'SL':>4s} {'CL':>4s} {'PF':>5s} {'PnL%':>8s} {'$/MNQ':>9s} {'MaxDD%':>7s} {'Sharpe':>7s}")
    print(f"  {'-'*22} {'-'*4} {'-'*6} {'-'*6} {'-'*4} {'-'*4} {'-'*4} {'-'*5} {'-'*8} {'-'*9} {'-'*7} {'-'*7}")
    
    for r in results:
        if r is None:
            continue
        print(f"  {r['label']:22s} {r['rr']:4.1f} {r['trades']:6d} {r['wr']:6.1%} "
              f"{r['tp_hits']:4d} {r['sl_hits']:4d} {r['close_hits']:4d} "
              f"{r.get('pf', 0):5.2f} {r['total_pnl']:+8.2%} {r['pnl_mnq']:+9,.0f} "
              f"{r.get('max_dd', 0):+7.2%} {r.get('sharpe', 0):7.2f}")


def print_intraday_comparison(results, title):
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")
    print(f"  {'Config':22s} {'R:R':>4s} {'Trades':>6s} {'WR':>6s} {'TP':>4s} {'SL':>4s} {'CL':>4s} {'PnL%':>8s} {'$/MNQ':>9s} {'Conf.WR':>8s} {'Conf.N':>6s}")
    print(f"  {'-'*22} {'-'*4} {'-'*6} {'-'*6} {'-'*4} {'-'*4} {'-'*4} {'-'*8} {'-'*9} {'-'*8} {'-'*6}")
    
    for r in results:
        if r is None:
            continue
        print(f"  {r['label']:22s} {r['rr']:4.1f} {r['trades']:6d} {r['wr']:6.1%} "
              f"{r['tp_hits']:4d} {r['sl_hits']:4d} {r['close_hits']:4d} "
              f"{r['total_pnl']:+8.2%} {r['pnl_mnq']:+9,.0f} "
              f"{r['confirmed_wr']:8.1%} {r['confirmed_n']:6d}")


if __name__ == '__main__':
    print("=" * 90)
    print("  R:R RATIO COMPARISON — GLOBAL SESSION SIGNAL → NQ")
    print("=" * 90)
    
    signals = download_and_build_signals()
    print(f"Signal data: {len(signals)} days, {signals.index[0].date()} to {signals.index[-1].date()}")
    
    # ─── DAILY SIMULATION ────────────────────────────────────────────
    configs = [
        # (tp, sl, label)
        (0.005, 0.005, "1:1  (0.50/0.50)"),
        (0.005, 0.0025, "1:2  (0.25/0.50)"),
        (0.005, 0.0033, "1:1.5(0.33/0.50)"),
        (0.004, 0.002, "1:2  (0.20/0.40)"),
        (0.006, 0.003, "1:2  (0.30/0.60)"),
        (0.005, 0.002, "1:2.5(0.20/0.50)"),
        (0.0075, 0.0025,"1:3  (0.25/0.75)"),
        (0.003, 0.003, "1:1  (0.30/0.30)"),
    ]
    
    for thresh in [0.002, 0.003]:
        results = []
        for tp, sl, label in configs:
            r = simulate_daily_rr(signals, tp, sl, threshold=thresh, label=label)
            results.append(r)
        print_comparison(results, f"DAILY DATA — Signal threshold: {thresh:.2%}")
    
    # ─── INTRADAY SIMULATION ─────────────────────────────────────────
    print("\nDownloading 5-min NQ data for intraday sim...")
    nq_5m = yf.download('NQ=F', period='60d', interval='5m', progress=False)
    
    if len(nq_5m) > 0:
        intra_results = []
        for tp, sl, label in configs:
            nq_copy = nq_5m.copy()
            r = simulate_intraday_rr(nq_copy, signals, tp, sl, threshold=0.002, label=label)
            intra_results.append(r)
        print_intraday_comparison(intra_results, "INTRADAY 5-MIN DATA — Signal threshold: 0.20%")
    
    # ─── EXPECTANCY ANALYSIS ─────────────────────────────────────────
    print(f"\n{'='*90}")
    print("  EXPECTANCY ANALYSIS (per trade, daily data, threshold=0.20%)")
    print(f"{'='*90}")
    print(f"  {'Config':22s} {'WR':>6s} {'AvgWin':>8s} {'AvgLoss':>8s} {'Expect':>8s} {'$/MNQ/trade':>12s}")
    print(f"  {'-'*22} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*12}")
    
    for tp, sl, label in configs:
        r = simulate_daily_rr(signals, tp, sl, threshold=0.002, label=label)
        if r:
            exp = r['expectancy']
            exp_dollars = exp * 20000 * 2.0
            print(f"  {label:22s} {r['wr']:6.1%} {r['avg_win']:+8.4%} {r['avg_loss']:+8.4%} {exp:+8.4%} ${exp_dollars:+11,.0f}")
    
    print(f"\n  Note: 1:2 R:R needs only ~34% WR to break even (vs 50% for 1:1)")
    print(f"  Key: Does the tighter stop (0.25%) get hit too often on NQ's intraday noise?")
