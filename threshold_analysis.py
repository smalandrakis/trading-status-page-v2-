#!/usr/bin/env python3
"""
Threshold Analysis: What probability cutoff should we use for the ML ensemble?
"""
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "data"
MODEL_DIR = "models_mnq_global"
MNQ_FILE = f"{DATA_DIR}/MNQ_5min_IB_with_indicators.csv"

ASIA = ['^N225', '^HSI', '^AXJO']
EUROPE = ['^GDAXI', '^FTSE', '^STOXX50E']
MACRO = ['^VIX', 'JPY=X', 'TLT', 'NQ=F', 'ES=F', 'CL=F']


def prepare_test_data():
    """Load data, features, ML predictions — return test set + MNQ 5min."""
    mnq = pd.read_csv(MNQ_FILE, usecols=['date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    mnq['date'] = pd.to_datetime(mnq['date'], utc=True)
    mnq['date'] = mnq['date'].dt.tz_convert('America/New_York')
    mnq = mnq.set_index('date').sort_index().between_time('09:30', '16:00')

    daily = []
    for date, dd in mnq.groupby(mnq.index.date):
        if len(dd) < 20:
            continue
        o, h, l, c = dd['Open'].iloc[0], dd['High'].max(), dd['Low'].min(), dd['Close'].iloc[-1]
        fh = dd.iloc[:12]
        fh_ret = (fh['Close'].iloc[-1] / fh['Open'].iloc[0] - 1) if len(fh) >= 12 else 0
        fh_range = (fh['High'].max() - fh['Low'].min()) / fh['Open'].iloc[0] if len(fh) >= 12 else 0
        post_fh = dd.iloc[12:]
        entry = post_fh['Open'].iloc[0] if len(post_fh) > 0 else None
        daily.append({
            'date': pd.Timestamp(date), 'open': o, 'high': h, 'low': l, 'close': c,
            'volume': dd['Volume'].sum(), 'daily_ret': c / o - 1, 'daily_range': (h - l) / o,
            'fh_ret': fh_ret, 'fh_range': fh_range, 'fh_vol': fh['Volume'].sum(),
            'entry_price': entry,
        })

    md = pd.DataFrame(daily).set_index('date')
    md['ret_1d'] = md['daily_ret'].shift(1)
    md['ret_2d'] = md['daily_ret'].shift(1).rolling(2).sum()
    md['ret_5d'] = md['daily_ret'].shift(1).rolling(5).sum()
    md['vol_5d'] = md['daily_range'].shift(1).rolling(5).mean()
    md['vol_10d'] = md['daily_range'].shift(1).rolling(10).mean()
    md['volume_ratio'] = md['volume'].shift(1) / md['volume'].shift(1).rolling(10).mean()
    md['prev_fh_ret'] = md['fh_ret'].shift(1)
    md['prev_fh_range'] = md['fh_range'].shift(1)

    start = (mnq.index[0] - timedelta(days=30)).strftime('%Y-%m-%d')
    end = (mnq.index[-1] + timedelta(days=5)).strftime('%Y-%m-%d')
    ext = pd.DataFrame()
    print("Downloading external data...", flush=True)
    for t in ASIA + EUROPE + MACRO:
        col = t.replace('^', '').replace('=', '_')
        df = yf.download(t, start=start, end=end, interval='1d', progress=False)
        if len(df) > 0:
            close = df[('Close', t)] if isinstance(df.columns, pd.MultiIndex) else df['Close']
            ext[f'{col}_close'] = close
            ext[f'{col}_ret'] = close.pct_change()

    ext.index = ext.index.tz_localize(None) if ext.index.tz else ext.index
    a_ret = [c for c in ext.columns if c.endswith('_ret') and any(x in c for x in ['N225', 'HSI', 'AXJO'])]
    e_ret = [c for c in ext.columns if c.endswith('_ret') and any(x in c for x in ['GDAXI', 'FTSE', 'STOXX50E'])]
    if a_ret: ext['asia_ret'] = ext[a_ret].mean(axis=1)
    if e_ret: ext['europe_ret'] = ext[e_ret].mean(axis=1)
    if a_ret and e_ret: ext['combined_ret'] = 0.4 * ext['asia_ret'] + 0.6 * ext['europe_ret']
    if 'VIX_close' in ext:
        ext['vix_level'] = ext['VIX_close']; ext['vix_change'] = ext['VIX_close'].pct_change()
        ext['vix_5d_avg'] = ext['VIX_close'].rolling(5).mean()
        ext['vix_above_avg'] = (ext['VIX_close'] > ext['vix_5d_avg']).astype(int)
    if 'JPY_X_ret' in ext: ext['usdjpy_ret'] = ext['JPY_X_ret']; ext['usdjpy_5d'] = ext['JPY_X_ret'].rolling(5).sum()
    if 'TLT_ret' in ext: ext['bond_ret'] = ext['TLT_ret']; ext['bond_5d'] = ext['TLT_ret'].rolling(5).sum()
    if 'CL_F_ret' in ext: ext['oil_ret'] = ext['CL_F_ret']
    if 'NQ_F_ret' in ext:
        ext['nq_prev_ret'] = ext['NQ_F_ret'].shift(1)
        ext['nq_prev_2d'] = ext['NQ_F_ret'].shift(1).rolling(2).sum()
        ext['nq_prev_5d'] = ext['NQ_F_ret'].shift(1).rolling(5).sum()
    if 'ES_F_ret' in ext:
        ext['es_ret'] = ext['ES_F_ret']
        ext['es_nq_spread'] = ext.get('NQ_F_ret', 0) - ext['ES_F_ret']

    md.index = pd.to_datetime(md.index).normalize()
    ext.index = pd.to_datetime(ext.index).normalize()
    feat = md.join(ext, how='left')
    ec = [c for c in ext.columns if c in feat.columns]
    feat[ec] = feat[ec].ffill()
    feat['target_ret'] = (feat['close'] - feat['entry_price']) / feat['entry_price']
    feat['target_dir'] = (feat['target_ret'] > 0).astype(int)
    exclude = ['open', 'high', 'low', 'close', 'volume', 'entry_price',
               'target_ret', 'target_dir', 'daily_ret', 'fh_ret', 'fh_range', 'fh_vol']
    fcols = [c for c in feat.columns if c not in exclude and '_close' not in c]
    feat = feat.dropna(subset=fcols + ['target_dir', 'entry_price'])

    gbm = joblib.load(f'{MODEL_DIR}/gbm_global_signal.pkl')
    rf = joblib.load(f'{MODEL_DIR}/rf_global_signal.pkl')
    scaler = joblib.load(f'{MODEL_DIR}/scaler_global_signal.pkl')
    saved_cols = joblib.load(f'{MODEL_DIR}/feature_cols.pkl')

    cutoff = feat.index.max() - timedelta(days=90)
    test = feat[feat.index > cutoff].copy()
    X = scaler.transform(test[saved_cols].values)
    test['gbm_prob'] = gbm.predict_proba(X)[:, 1]
    test['rf_prob'] = rf.predict_proba(X)[:, 1]
    test['ens_prob'] = 0.5 * test['gbm_prob'] + 0.5 * test['rf_prob']

    return mnq, test


def backtest_threshold(mnq, test, thresh, tp_pct, sl_pct):
    """Run bar-by-bar backtest at a given probability threshold."""
    trades = []
    for date in sorted(set(mnq.index.date)):
        dt = pd.Timestamp(date)
        if dt not in test.index:
            continue
        prob = test.loc[dt, 'ens_prob']
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
        trades.append({'pnl': pnl, 'dir': 'L' if direction == 1 else 'S'})

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
    return {'n': n, 'wr': wr, 'pf': pf, 'total': total, 'mnq': mnq_d,
            'sharpe': sharpe, 'exp': exp, 'n_l': n_l, 'n_s': n_s}


if __name__ == '__main__':
    mnq, test = prepare_test_data()

    print(f"\n{'='*90}")
    print("PROBABILITY DISTRIBUTION on test set")
    print(f"{'='*90}")
    print(f"\n  {len(test)} test days, prob range: {test['ens_prob'].min():.2%} to {test['ens_prob'].max():.2%}")
    for lo, hi in [(0, .30), (.30, .40), (.40, .45), (.45, .50), (.50, .55), (.55, .60),
                   (.60, .65), (.65, .70), (.70, .80), (.80, 1.01)]:
        n = ((test['ens_prob'] >= lo) & (test['ens_prob'] < hi)).sum()
        print(f"  {lo:.0%}-{hi:.0%}: {n:3d} days ({n/len(test):.0%})")

    for tp, sl, label in [(0.010, 0.005, "TP 1.0%/SL 0.5%"), (0.015, 0.0075, "TP 1.5%/SL 0.75%")]:
        print(f"\n{'='*90}")
        print(f"THRESHOLD SWEEP — {label}")
        print(f"{'='*90}")
        hdr = f"  {'Thresh':>7s} {'Trades':>6s} {'LONG':>5s} {'SHORT':>5s} {'WR':>6s} {'PF':>5s} {'PnL%':>8s} {'$/MNQ':>8s} {'Sharpe':>7s} {'$/trade':>8s}"
        print(hdr)
        print(f"  {'-'*7} {'-'*6} {'-'*5} {'-'*5} {'-'*6} {'-'*5} {'-'*8} {'-'*8} {'-'*7} {'-'*8}")

        for thresh in [0.50, 0.525, 0.55, 0.575, 0.60, 0.625, 0.65, 0.70, 0.75, 0.80]:
            r = backtest_threshold(mnq, test, thresh, tp, sl)
            if r is None:
                print(f"  {thresh:>7.1%}   no trades")
                continue
            print(f"  {thresh:>7.1%} {r['n']:6d} {r['n_l']:5d} {r['n_s']:5d} {r['wr']:6.1%} "
                  f"{r['pf']:5.2f} {r['total']:+8.2%} {r['mnq']:+8,.0f} {r['sharpe']:7.2f} ${r['exp']:+7,.0f}")

    print(f"\n{'='*90}")
    print("CONCLUSION")
    print(f"{'='*90}")
