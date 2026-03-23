#!/usr/bin/env python3
"""
Global Session Signal Analysis for NQ/MNQ
==========================================
Hypothesis: Asia + Europe session returns predict US NQ direction.
Strategy: After first hour (10:30 ET), go LONG or SHORT NQ, hold for 0.5% move.

Indices used:
- Asia:   Nikkei 225 (^N225), Hang Seng (^HSI), ASX 200 (^AXJO)
- Europe: DAX (^GDAXI), FTSE 100 (^FTSE), Euro Stoxx 50 (^STOXX50E)
- US:     Nasdaq 100 (NQ=F futures), or QQQ as proxy
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIG ───────────────────────────────────────────────────────────
PERIOD = "2y"            # 2 years of daily data
TARGET_MOVE = 0.005      # 0.5% target
MAX_HOLD_BARS = 78       # max hold = rest of day (~6.5h in 5-min bars)
FIRST_HOUR_BARS = 12     # first hour = 12 x 5-min bars (9:30-10:30)

ASIA_TICKERS = ['^N225', '^HSI', '^AXJO']
EUROPE_TICKERS = ['^GDAXI', '^FTSE', '^STOXX50E']
US_TICKERS = ['NQ=F', 'QQQ']

ALL_TICKERS = ASIA_TICKERS + EUROPE_TICKERS + US_TICKERS

# ─── DOWNLOAD DATA ───────────────────────────────────────────────────
def download_daily_data():
    """Download daily OHLC for all indices."""
    print("=" * 70)
    print("DOWNLOADING DAILY DATA FOR GLOBAL INDICES")
    print("=" * 70)
    
    data = {}
    for ticker in ALL_TICKERS:
        print(f"  Downloading {ticker}...", end=" ")
        try:
            df = yf.download(ticker, period=PERIOD, interval='1d', progress=False)
            if len(df) > 0:
                data[ticker] = df
                print(f"OK - {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}")
            else:
                print("EMPTY")
        except Exception as e:
            print(f"ERROR: {e}")
    
    return data


def download_nq_intraday():
    """Download 5-min NQ futures data for intraday strategy simulation."""
    print("\nDownloading NQ=F 5-min intraday data (60 days max from yfinance)...")
    try:
        df = yf.download('NQ=F', period='60d', interval='5m', progress=False)
        if len(df) > 0:
            print(f"  OK - {len(df)} bars")
            return df
        else:
            print("  Empty, trying QQQ...")
            df = yf.download('QQQ', period='60d', interval='5m', progress=False)
            print(f"  QQQ: {len(df)} bars")
            return df
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


# ─── COMPUTE SESSION RETURNS ─────────────────────────────────────────
def compute_session_returns(data):
    """Compute daily % returns for each index."""
    returns = pd.DataFrame()
    
    for ticker, df in data.items():
        # Use Close-to-Close daily return
        col_name = ticker.replace('^', '').replace('=', '_')
        if 'Close' in df.columns:
            close = df['Close']
        else:
            # Multi-level columns from yfinance
            close = df[('Close', ticker)] if ('Close', ticker) in df.columns else df.iloc[:, 3]
        
        returns[col_name] = close.pct_change()
    
    returns = returns.dropna()
    return returns


def build_signals(data):
    """
    Build daily signals from Asia/Europe sessions.
    
    For each US trading day:
    - Asia signal = avg return of N225, HSI, AXJO (they close before US opens)
    - Europe signal = avg return of GDAXI, FTSE, STOXX50E (morning session before US)
    - Combined signal = weighted avg
    
    Target: NQ direction from 10:30 ET to close (or 0.5% move)
    """
    returns = compute_session_returns(data)
    
    # Build composite signals
    asia_cols = [c for c in returns.columns if c in ['N225', 'HSI', 'AXJO']]
    europe_cols = [c for c in returns.columns if c in ['GDAXI', 'FTSE', 'STOXX50E']]
    us_cols = [c for c in returns.columns if c in ['NQ_F', 'QQQ']]
    
    print(f"\nAvailable columns: {list(returns.columns)}")
    print(f"  Asia:   {asia_cols}")
    print(f"  Europe: {europe_cols}")
    print(f"  US:     {us_cols}")
    
    if not asia_cols or not europe_cols or not us_cols:
        print("ERROR: Missing required index data")
        return None
    
    signals = pd.DataFrame(index=returns.index)
    signals['asia_ret'] = returns[asia_cols].mean(axis=1)
    signals['europe_ret'] = returns[europe_cols].mean(axis=1)
    signals['combined_ret'] = 0.4 * signals['asia_ret'] + 0.6 * signals['europe_ret']
    
    # Individual index returns for analysis
    for col in asia_cols + europe_cols:
        signals[f'{col}_ret'] = returns[col]
    
    # US target (NQ preferred, QQQ fallback)
    us_col = 'NQ_F' if 'NQ_F' in us_cols else us_cols[0]
    signals['nq_ret'] = returns[us_col]
    signals['nq_direction'] = (signals['nq_ret'] > 0).astype(int)  # 1=up, 0=down
    signals['nq_hit_target'] = (signals['nq_ret'].abs() >= TARGET_MOVE).astype(int)
    
    return signals


# ─── ANALYSIS ─────────────────────────────────────────────────────────
def analyze_predictive_power(signals):
    """Analyze how well Asia/Europe sessions predict NQ direction."""
    print("\n" + "=" * 70)
    print("PREDICTIVE POWER ANALYSIS")
    print("=" * 70)
    
    n = len(signals)
    print(f"\nSample size: {n} trading days")
    print(f"Date range: {signals.index[0].date()} to {signals.index[-1].date()}")
    
    # ─── 1. RAW CORRELATIONS ─────────────────────────────────────────
    print("\n--- 1. CORRELATIONS WITH NQ DAILY RETURN ---")
    for col in ['asia_ret', 'europe_ret', 'combined_ret']:
        corr = signals[col].corr(signals['nq_ret'])
        print(f"  {col:20s} vs NQ return: r = {corr:+.4f}")
    
    # Individual indices
    idx_cols = [c for c in signals.columns if c.endswith('_ret') and c not in ['asia_ret', 'europe_ret', 'combined_ret', 'nq_ret']]
    print("\n  Individual indices:")
    for col in idx_cols:
        corr = signals[col].corr(signals['nq_ret'])
        print(f"    {col:20s} vs NQ return: r = {corr:+.4f}")
    
    # ─── 2. DIRECTIONAL ACCURACY ─────────────────────────────────────
    print("\n--- 2. DIRECTIONAL ACCURACY (same-sign prediction) ---")
    for col, label in [('asia_ret', 'Asia'), ('europe_ret', 'Europe'), ('combined_ret', 'Combined')]:
        # Predict: if signal > 0, go LONG; if signal < 0, go SHORT
        pred_dir = (signals[col] > 0).astype(int)
        accuracy = (pred_dir == signals['nq_direction']).mean()
        
        # Split by signal direction
        long_mask = signals[col] > 0
        short_mask = signals[col] < 0
        
        long_wr = signals.loc[long_mask, 'nq_direction'].mean() if long_mask.sum() > 0 else 0
        short_wr = (1 - signals.loc[short_mask, 'nq_direction']).mean() if short_mask.sum() > 0 else 0
        
        print(f"\n  {label} Signal:")
        print(f"    Overall accuracy:  {accuracy:.1%} ({int(accuracy*n)}/{n})")
        print(f"    LONG  (signal>0):  {long_wr:.1%} WR on {long_mask.sum()} days")
        print(f"    SHORT (signal<0):  {short_wr:.1%} WR on {short_mask.sum()} days")
    
    # ─── 3. SIGNAL STRENGTH BUCKETS ──────────────────────────────────
    print("\n--- 3. SIGNAL STRENGTH ANALYSIS (combined signal) ---")
    print("  Bucket the combined signal by magnitude and check NQ direction:\n")
    
    sig = signals['combined_ret']
    buckets = [
        ('Strong DOWN (<-0.5%)', sig < -0.005),
        ('Mild DOWN (-0.5% to -0.1%)', (sig >= -0.005) & (sig < -0.001)),
        ('Flat (-0.1% to +0.1%)', (sig >= -0.001) & (sig <= 0.001)),
        ('Mild UP (+0.1% to +0.5%)', (sig > 0.001) & (sig <= 0.005)),
        ('Strong UP (>+0.5%)', sig > 0.005),
    ]
    
    print(f"  {'Bucket':35s} {'Days':>5s} {'NQ Up%':>8s} {'Avg NQ Ret':>12s} {'Median NQ':>12s}")
    print(f"  {'-'*35} {'-'*5} {'-'*8} {'-'*12} {'-'*12}")
    
    for label, mask in buckets:
        count = mask.sum()
        if count > 0:
            pct_up = signals.loc[mask, 'nq_direction'].mean()
            avg_ret = signals.loc[mask, 'nq_ret'].mean()
            med_ret = signals.loc[mask, 'nq_ret'].median()
            print(f"  {label:35s} {count:5d} {pct_up:8.1%} {avg_ret:+12.4%} {med_ret:+12.4%}")
        else:
            print(f"  {label:35s} {count:5d}      -            -            -")
    
    # ─── 4. STRONG SIGNAL ONLY ────────────────────────────────────────
    print("\n--- 4. STRONG SIGNAL FILTER (|combined| > 0.3%) ---")
    strong_mask = signals['combined_ret'].abs() > 0.003
    strong = signals[strong_mask].copy()
    
    if len(strong) > 0:
        pred_dir = (strong['combined_ret'] > 0).astype(int)
        accuracy = (pred_dir == strong['nq_direction']).mean()
        avg_nq = strong['nq_ret'].mean()
        
        # Strategy: follow the signal
        strong['strat_ret'] = strong['nq_ret'] * np.sign(strong['combined_ret'])
        strat_wr = (strong['strat_ret'] > 0).mean()
        
        print(f"  Days with strong signal: {len(strong)} ({len(strong)/n:.0%} of all days)")
        print(f"  Direction accuracy: {accuracy:.1%}")
        print(f"  Strategy WR (follow signal): {strat_wr:.1%}")
        print(f"  Avg strategy return/day: {strong['strat_ret'].mean():+.4%}")
        print(f"  Total strategy return: {strong['strat_ret'].sum():+.2%}")
    
    # ─── 5. 0.5% TARGET HIT RATE ─────────────────────────────────────
    print("\n--- 5. 0.5% TARGET ANALYSIS ---")
    print(f"  Days NQ moved >= 0.5% (either direction): {signals['nq_hit_target'].sum()}/{n} ({signals['nq_hit_target'].mean():.1%})")
    
    # When signal is strong, does NQ reach 0.5% in predicted direction?
    if len(strong) > 0:
        strong['correct_big_move'] = (
            (strong['strat_ret'] >= TARGET_MOVE)
        ).astype(int)
        strong['wrong_big_move'] = (
            (strong['strat_ret'] <= -TARGET_MOVE)
        ).astype(int)
        
        print(f"\n  When combined signal is strong (|>0.3%):")
        print(f"    NQ hits +0.5% in predicted dir: {strong['correct_big_move'].sum()}/{len(strong)} ({strong['correct_big_move'].mean():.1%})")
        print(f"    NQ hits -0.5% AGAINST predicted: {strong['wrong_big_move'].sum()}/{len(strong)} ({strong['wrong_big_move'].mean():.1%})")
    
    return signals


# ─── SIMPLE STRATEGY SIMULATION ──────────────────────────────────────
def simulate_strategy_daily(signals, threshold=0.002):
    """
    Simulate daily strategy using close-to-close returns.
    
    Rules:
    - If combined signal > threshold: go LONG NQ
    - If combined signal < -threshold: go SHORT NQ
    - Otherwise: no trade
    - Target: 0.5% in predicted direction
    - Since we only have daily data, approximate:
      - If |NQ daily return| >= 0.5% and in our direction: book +0.5%
      - If NQ daily return against us by >= 0.5%: book -0.5% (stopped out)
      - If NQ moves < 0.5% either way: book actual NQ return
    """
    print("\n" + "=" * 70)
    print(f"STRATEGY SIMULATION (signal threshold: {threshold:.2%})")
    print("=" * 70)
    
    trades = []
    for idx, row in signals.iterrows():
        sig = row['combined_ret']
        nq = row['nq_ret']
        
        if abs(sig) < threshold:
            continue
        
        direction = 1 if sig > 0 else -1
        dir_label = 'LONG' if direction == 1 else 'SHORT'
        raw_pnl = nq * direction
        
        # Apply 0.5% target and stop
        if raw_pnl >= TARGET_MOVE:
            booked_pnl = TARGET_MOVE  # hit target
            outcome = 'TARGET'
        elif raw_pnl <= -TARGET_MOVE:
            booked_pnl = -TARGET_MOVE  # stopped out
            outcome = 'STOPPED'
        else:
            booked_pnl = raw_pnl  # held to close
            outcome = 'CLOSE'
        
        trades.append({
            'date': idx,
            'direction': dir_label,
            'signal': sig,
            'nq_ret': nq,
            'raw_pnl': raw_pnl,
            'booked_pnl': booked_pnl,
            'outcome': outcome
        })
    
    if not trades:
        print("  No trades generated!")
        return
    
    df = pd.DataFrame(trades)
    
    n_trades = len(df)
    wins = (df['booked_pnl'] > 0).sum()
    losses = (df['booked_pnl'] < 0).sum()
    flat = (df['booked_pnl'] == 0).sum()
    wr = wins / n_trades
    
    total_pnl = df['booked_pnl'].sum()
    avg_win = df.loc[df['booked_pnl'] > 0, 'booked_pnl'].mean() if wins > 0 else 0
    avg_loss = df.loc[df['booked_pnl'] < 0, 'booked_pnl'].mean() if losses > 0 else 0
    
    # MNQ point value = $2/point, 1 contract NQ = $20/point
    # 0.5% of NQ ~20000 = 100 points = $2000 per NQ contract or $200 per MNQ
    nq_price = 20000  # approximate
    mnq_per_point = 2.0  # MNQ $2/point
    
    print(f"\n  Total trades: {n_trades}")
    print(f"  Wins: {wins}, Losses: {losses}, Flat: {flat}")
    print(f"  Win Rate: {wr:.1%}")
    print(f"  Avg Win:  {avg_win:+.4%}")
    print(f"  Avg Loss: {avg_loss:+.4%}")
    print(f"  Total PnL (% terms): {total_pnl:+.2%}")
    print(f"  Profit Factor: {abs(df.loc[df['booked_pnl']>0,'booked_pnl'].sum() / df.loc[df['booked_pnl']<0,'booked_pnl'].sum()):.2f}" if losses > 0 else "  Profit Factor: inf")
    
    # $$ estimates for 1 MNQ contract
    pnl_points = total_pnl * nq_price
    pnl_dollars = pnl_points * mnq_per_point
    print(f"\n  Estimated P&L (1 MNQ, ~$2/pt):")
    print(f"    Total points: {pnl_points:+.0f}")
    print(f"    Total $: ${pnl_dollars:+,.0f}")
    print(f"    Per trade avg: ${pnl_dollars/n_trades:+,.0f}")
    
    # Breakdown by direction
    for d in ['LONG', 'SHORT']:
        sub = df[df['direction'] == d]
        if len(sub) > 0:
            sub_wr = (sub['booked_pnl'] > 0).mean()
            sub_pnl = sub['booked_pnl'].sum()
            print(f"\n  {d}: {len(sub)} trades, WR={sub_wr:.1%}, PnL={sub_pnl:+.2%}")
    
    # Breakdown by outcome
    print("\n  Outcome breakdown:")
    for outcome in ['TARGET', 'STOPPED', 'CLOSE']:
        sub = df[df['outcome'] == outcome]
        if len(sub) > 0:
            print(f"    {outcome:8s}: {len(sub):3d} trades ({len(sub)/n_trades:.0%}), avg PnL={sub['booked_pnl'].mean():+.4%}")
    
    # Monthly performance
    print("\n  Monthly performance:")
    df['month'] = df['date'].dt.to_period('M')
    monthly = df.groupby('month').agg(
        trades=('booked_pnl', 'count'),
        wins=('booked_pnl', lambda x: (x > 0).sum()),
        pnl=('booked_pnl', 'sum')
    )
    for m, row in monthly.iterrows():
        wr_m = row['wins'] / row['trades'] if row['trades'] > 0 else 0
        print(f"    {m}: {row['trades']:2.0f} trades, WR={wr_m:.0%}, PnL={row['pnl']:+.3%}")
    
    # Equity curve
    df['cum_pnl'] = df['booked_pnl'].cumsum()
    max_dd = (df['cum_pnl'] - df['cum_pnl'].cummax()).min()
    print(f"\n  Max Drawdown: {max_dd:+.2%} ({max_dd * nq_price * mnq_per_point:+,.0f}$ per MNQ)")
    print(f"  Sharpe (daily, annualized): {df['booked_pnl'].mean() / df['booked_pnl'].std() * np.sqrt(252):.2f}" if df['booked_pnl'].std() > 0 else "  Sharpe: N/A")
    
    return df


# ─── INTRADAY SIMULATION ─────────────────────────────────────────────
def simulate_intraday(nq_5m, signals):
    """
    More accurate simulation using 5-min intraday NQ data.
    
    For each day:
    1. Check Asia/Europe signal (from daily data, same day)
    2. Wait for first hour (9:30-10:30 ET) = 12 bars of 5-min
    3. Check if first hour confirms signal direction
    4. Enter at 10:30 ET, hold until +0.5% or -0.5% or close
    """
    if nq_5m is None or len(nq_5m) == 0:
        print("\n  No intraday data available for simulation")
        return
    
    print("\n" + "=" * 70)
    print("INTRADAY SIMULATION (5-min NQ data)")
    print("=" * 70)
    
    # Ensure timezone-aware
    if nq_5m.index.tz is None:
        nq_5m.index = nq_5m.index.tz_localize('America/New_York')
    else:
        nq_5m.index = nq_5m.index.tz_convert('America/New_York')
    
    # Handle multi-level columns
    if isinstance(nq_5m.columns, pd.MultiIndex):
        nq_5m.columns = nq_5m.columns.get_level_values(0)
    
    # Group by date
    nq_5m['date'] = nq_5m.index.date
    
    trades = []
    
    for date, day_data in nq_5m.groupby('date'):
        date_ts = pd.Timestamp(date)
        
        # Find matching signal (same date or previous trading day)
        sig_matches = signals.index[signals.index.date == date]
        if len(sig_matches) == 0:
            # Try previous day
            sig_matches = signals.index[signals.index.date < date]
            if len(sig_matches) == 0:
                continue
            date_ts = sig_matches[-1]
        else:
            date_ts = sig_matches[0]
        
        sig_val = signals.loc[date_ts, 'combined_ret']
        
        # Skip weak signals
        if abs(sig_val) < 0.002:
            continue
        
        direction = 1 if sig_val > 0 else -1
        dir_label = 'LONG' if direction == 1 else 'SHORT'
        
        # Filter to RTH (9:30-16:00 ET)
        rth = day_data.between_time('09:30', '16:00')
        if len(rth) < FIRST_HOUR_BARS + 5:
            continue
        
        # First hour return (9:30-10:30)
        first_hour = rth.iloc[:FIRST_HOUR_BARS]
        fh_ret = (first_hour['Close'].iloc[-1] / first_hour['Open'].iloc[0]) - 1
        
        # Optional: check if first hour confirms signal
        fh_confirms = (fh_ret > 0 and direction == 1) or (fh_ret < 0 and direction == -1)
        
        # Entry at 10:30 (after first hour)
        post_fh = rth.iloc[FIRST_HOUR_BARS:]
        if len(post_fh) < 2:
            continue
        
        entry_price = post_fh['Open'].iloc[0]
        
        # Simulate bar-by-bar
        exit_price = None
        exit_reason = None
        
        for i, (ts, bar) in enumerate(post_fh.iterrows()):
            if direction == 1:  # LONG
                # Check high for target
                pct_high = (bar['High'] / entry_price) - 1
                pct_low = (bar['Low'] / entry_price) - 1
                
                if pct_high >= TARGET_MOVE:
                    exit_price = entry_price * (1 + TARGET_MOVE)
                    exit_reason = 'TARGET'
                    break
                elif pct_low <= -TARGET_MOVE:
                    exit_price = entry_price * (1 - TARGET_MOVE)
                    exit_reason = 'STOPPED'
                    break
            else:  # SHORT
                pct_high = (bar['High'] / entry_price) - 1
                pct_low = (bar['Low'] / entry_price) - 1
                
                if pct_low <= -TARGET_MOVE:
                    exit_price = entry_price * (1 - TARGET_MOVE)
                    exit_reason = 'TARGET'
                    break
                elif pct_high >= TARGET_MOVE:
                    exit_price = entry_price * (1 + TARGET_MOVE)
                    exit_reason = 'STOPPED'
                    break
        
        if exit_price is None:
            # Held to close
            exit_price = post_fh['Close'].iloc[-1]
            exit_reason = 'CLOSE'
        
        pnl_pct = ((exit_price / entry_price) - 1) * direction
        
        trades.append({
            'date': date,
            'direction': dir_label,
            'signal': sig_val,
            'fh_ret': fh_ret,
            'fh_confirms': fh_confirms,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'pnl_pct': pnl_pct,
        })
    
    if not trades:
        print("  No intraday trades generated!")
        return
    
    df = pd.DataFrame(trades)
    
    n_trades = len(df)
    wins = (df['pnl_pct'] > 0).sum()
    wr = wins / n_trades
    total_pnl = df['pnl_pct'].sum()
    
    print(f"\n  All trades (signal > 0.2% threshold):")
    print(f"    Trades: {n_trades}, Wins: {wins}, WR: {wr:.1%}")
    print(f"    Avg PnL: {df['pnl_pct'].mean():+.4%}")
    print(f"    Total PnL: {total_pnl:+.2%}")
    
    # With first-hour confirmation
    confirmed = df[df['fh_confirms']]
    if len(confirmed) > 0:
        c_wins = (confirmed['pnl_pct'] > 0).sum()
        c_wr = c_wins / len(confirmed)
        print(f"\n  With first-hour confirmation:")
        print(f"    Trades: {len(confirmed)}, Wins: {c_wins}, WR: {c_wr:.1%}")
        print(f"    Avg PnL: {confirmed['pnl_pct'].mean():+.4%}")
        print(f"    Total PnL: {confirmed['pnl_pct'].sum():+.2%}")
    
    # Without confirmation (contrarian first hour)
    unconfirmed = df[~df['fh_confirms']]
    if len(unconfirmed) > 0:
        u_wins = (unconfirmed['pnl_pct'] > 0).sum()
        u_wr = u_wins / len(unconfirmed)
        print(f"\n  WITHOUT first-hour confirmation (contrarian 1st hour):")
        print(f"    Trades: {len(unconfirmed)}, Wins: {u_wins}, WR: {u_wr:.1%}")
        print(f"    Avg PnL: {unconfirmed['pnl_pct'].mean():+.4%}")
        print(f"    Total PnL: {unconfirmed['pnl_pct'].sum():+.2%}")
    
    # By outcome
    print("\n  Outcome breakdown:")
    for reason in ['TARGET', 'STOPPED', 'CLOSE']:
        sub = df[df['exit_reason'] == reason]
        if len(sub) > 0:
            print(f"    {reason:8s}: {len(sub):3d} trades, avg PnL={sub['pnl_pct'].mean():+.4%}")
    
    # By direction
    for d in ['LONG', 'SHORT']:
        sub = df[df['direction'] == d]
        if len(sub) > 0:
            sub_wr = (sub['pnl_pct'] > 0).mean()
            print(f"\n  {d}: {len(sub)} trades, WR={sub_wr:.1%}, avg PnL={sub['pnl_pct'].mean():+.4%}")
    
    # MNQ $$ estimate
    nq_price = 20000
    mnq_per_point = 2.0
    pnl_dollars = total_pnl * nq_price * mnq_per_point
    print(f"\n  Estimated P&L (1 MNQ): ${pnl_dollars:+,.0f} over {n_trades} trades (~{(df['date'].iloc[-1] - df['date'].iloc[0]).days} days)")
    
    return df


# ─── MAIN ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("🌏🌍🌎 GLOBAL SESSION → NQ DIRECTION PREDICTOR ANALYSIS")
    print(f"    Target move: {TARGET_MOVE:.1%}")
    print(f"    Analysis period: {PERIOD}")
    print()
    
    # Step 1: Download daily data
    data = download_daily_data()
    
    if len(data) < 4:
        print("\nERROR: Not enough data downloaded. Check internet connection.")
        exit(1)
    
    # Step 2: Build signals
    signals = build_signals(data)
    if signals is None:
        exit(1)
    
    # Step 3: Analyze predictive power
    signals = analyze_predictive_power(signals)
    
    # Step 4: Daily strategy simulation
    print("\n" + "=" * 70)
    print("STRATEGY SIMULATIONS AT DIFFERENT THRESHOLDS")
    print("=" * 70)
    
    for thresh in [0.001, 0.002, 0.003, 0.005]:
        simulate_strategy_daily(signals, threshold=thresh)
    
    # Step 5: Intraday simulation
    nq_5m = download_nq_intraday()
    if nq_5m is not None:
        simulate_intraday(nq_5m, signals)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY & NEXT STEPS")
    print("=" * 70)
    print("""
    Key questions answered:
    1. Correlation of Asia/Europe returns with NQ daily return
    2. Directional accuracy (can we predict UP/DOWN?)
    3. Does signal strength matter?
    4. Hit rate for 0.5% target
    5. First-hour confirmation effect
    
    If results are promising (>55% WR with reasonable sample):
    - Build proper backtest with tick-level MNQ data from IB
    - Add more features: VIX, USD/JPY, bond yields, oil
    - Train ML model on the signal
    - Deploy as live bot alongside BTC bots
    """)
