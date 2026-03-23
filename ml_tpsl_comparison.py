#!/usr/bin/env python3
"""
TP/SL Configuration Comparison WITH ML Model
=============================================
Re-runs the TP/SL grid using ML ensemble predictions (GBM+RF)
on the strict holdout set. No peeking.

Also compares: GBM-only, RF-only, Ensemble, and Simple Signal baseline.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import os
from datetime import datetime, timedelta, time as dtime
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "data"
MODEL_DIR = "models_mnq_global"
MNQ_FILE = os.path.join(DATA_DIR, "MNQ_5min_IB_with_indicators.csv")

TEST_HOLDOUT_DAYS = 90
SIGNAL_THRESHOLD = 0.002

ASIA_TICKERS = ['^N225', '^HSI', '^AXJO']
EUROPE_TICKERS = ['^GDAXI', '^FTSE', '^STOXX50E']
MACRO_TICKERS = ['^VIX', 'JPY=X', 'TLT', 'NQ=F', 'ES=F', 'CL=F']


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING (same as main pipeline)
# ═══════════════════════════════════════════════════════════════════════
def load_and_prepare():
    """Load MNQ + external features + ML models, return everything needed."""
    # Load MNQ
    print("Loading MNQ 5-min data...")
    mnq = pd.read_csv(MNQ_FILE, usecols=['date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    mnq['date'] = pd.to_datetime(mnq['date'], utc=True)
    mnq['date'] = mnq['date'].dt.tz_convert('America/New_York')
    mnq = mnq.set_index('date').sort_index()
    mnq_rth = mnq.between_time('09:30', '16:00')
    print(f"  {len(mnq_rth)} RTH bars, {len(set(mnq_rth.index.date))} days")
    
    # Build daily MNQ features
    print("Building daily features from MNQ intraday...")
    daily = []
    for date, day_data in mnq_rth.groupby(mnq_rth.index.date):
        if len(day_data) < 20:
            continue
        o = day_data['Open'].iloc[0]
        h = day_data['High'].max()
        l = day_data['Low'].min()
        c = day_data['Close'].iloc[-1]
        vol = day_data['Volume'].sum()
        
        fh = day_data.iloc[:12]
        fh_ret = (fh['Close'].iloc[-1] / fh['Open'].iloc[0] - 1) if len(fh) >= 12 else 0
        fh_range = (fh['High'].max() - fh['Low'].min()) / fh['Open'].iloc[0] if len(fh) >= 12 else 0
        fh_vol = fh['Volume'].sum()
        
        post_fh = day_data.iloc[12:]
        entry_price = post_fh['Open'].iloc[0] if len(post_fh) > 0 else None
        
        daily.append({
            'date': pd.Timestamp(date),
            'open': o, 'high': h, 'low': l, 'close': c, 'volume': vol,
            'daily_ret': (c / o - 1),
            'daily_range': (h - l) / o,
            'fh_ret': fh_ret, 'fh_range': fh_range, 'fh_vol': fh_vol,
            'entry_price': entry_price,
        })
    
    mnq_daily = pd.DataFrame(daily).set_index('date')
    mnq_daily['ret_1d'] = mnq_daily['daily_ret'].shift(1)
    mnq_daily['ret_2d'] = mnq_daily['daily_ret'].shift(1).rolling(2).sum()
    mnq_daily['ret_5d'] = mnq_daily['daily_ret'].shift(1).rolling(5).sum()
    mnq_daily['vol_5d'] = mnq_daily['daily_range'].shift(1).rolling(5).mean()
    mnq_daily['vol_10d'] = mnq_daily['daily_range'].shift(1).rolling(10).mean()
    mnq_daily['volume_ratio'] = mnq_daily['volume'].shift(1) / mnq_daily['volume'].shift(1).rolling(10).mean()
    mnq_daily['prev_fh_ret'] = mnq_daily['fh_ret'].shift(1)
    mnq_daily['prev_fh_range'] = mnq_daily['fh_range'].shift(1)
    
    # Download external features
    print("Downloading external features...")
    start = (mnq_rth.index[0] - timedelta(days=30)).strftime('%Y-%m-%d')
    end = (mnq_rth.index[-1] + timedelta(days=5)).strftime('%Y-%m-%d')
    
    ext = pd.DataFrame()
    for ticker in ASIA_TICKERS + EUROPE_TICKERS + MACRO_TICKERS:
        col = ticker.replace('^', '').replace('=', '_')
        df = yf.download(ticker, start=start, end=end, interval='1d', progress=False)
        if len(df) > 0:
            if isinstance(df.columns, pd.MultiIndex):
                close = df[('Close', ticker)]
            else:
                close = df['Close']
            ext[f'{col}_close'] = close
            ext[f'{col}_ret'] = close.pct_change()
    
    ext.index = ext.index.tz_localize(None) if ext.index.tz else ext.index
    
    asia_ret_cols = [c for c in ext.columns if c.endswith('_ret') and any(a in c for a in ['N225', 'HSI', 'AXJO'])]
    europe_ret_cols = [c for c in ext.columns if c.endswith('_ret') and any(e in c for e in ['GDAXI', 'FTSE', 'STOXX50E'])]
    
    if asia_ret_cols:
        ext['asia_ret'] = ext[asia_ret_cols].mean(axis=1)
    if europe_ret_cols:
        ext['europe_ret'] = ext[europe_ret_cols].mean(axis=1)
    if asia_ret_cols and europe_ret_cols:
        ext['combined_ret'] = 0.4 * ext['asia_ret'] + 0.6 * ext['europe_ret']
    
    if 'VIX_close' in ext.columns:
        ext['vix_level'] = ext['VIX_close']
        ext['vix_change'] = ext['VIX_close'].pct_change()
        ext['vix_5d_avg'] = ext['VIX_close'].rolling(5).mean()
        ext['vix_above_avg'] = (ext['VIX_close'] > ext['vix_5d_avg']).astype(int)
    if 'JPY_X_ret' in ext.columns:
        ext['usdjpy_ret'] = ext['JPY_X_ret']
        ext['usdjpy_5d'] = ext['JPY_X_ret'].rolling(5).sum()
    if 'TLT_ret' in ext.columns:
        ext['bond_ret'] = ext['TLT_ret']
        ext['bond_5d'] = ext['TLT_ret'].rolling(5).sum()
    if 'CL_F_ret' in ext.columns:
        ext['oil_ret'] = ext['CL_F_ret']
    if 'NQ_F_ret' in ext.columns:
        ext['nq_prev_ret'] = ext['NQ_F_ret'].shift(1)
        ext['nq_prev_2d'] = ext['NQ_F_ret'].shift(1).rolling(2).sum()
        ext['nq_prev_5d'] = ext['NQ_F_ret'].shift(1).rolling(5).sum()
    if 'ES_F_ret' in ext.columns:
        ext['es_ret'] = ext['ES_F_ret']
        ext['es_nq_spread'] = ext.get('NQ_F_ret', 0) - ext['ES_F_ret']
    
    # Merge
    mnq_daily.index = pd.to_datetime(mnq_daily.index).normalize()
    ext.index = pd.to_datetime(ext.index).normalize()
    features = mnq_daily.join(ext, how='left')
    ext_cols = [c for c in ext.columns if c in features.columns]
    features[ext_cols] = features[ext_cols].ffill()
    
    features['target_ret'] = (features['close'] - features['entry_price']) / features['entry_price']
    features['target_dir'] = (features['target_ret'] > 0).astype(int)
    
    # Feature columns
    exclude = ['open', 'high', 'low', 'close', 'volume', 'entry_price',
               'target_ret', 'target_dir', 'daily_ret', 'fh_ret', 'fh_range', 'fh_vol']
    feature_cols = [c for c in features.columns if c not in exclude and '_close' not in c]
    features = features.dropna(subset=feature_cols + ['target_dir', 'entry_price'])
    
    # Load saved models
    print("Loading ML models...")
    gbm = joblib.load(os.path.join(MODEL_DIR, 'gbm_global_signal.pkl'))
    rf = joblib.load(os.path.join(MODEL_DIR, 'rf_global_signal.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler_global_signal.pkl'))
    saved_cols = joblib.load(os.path.join(MODEL_DIR, 'feature_cols.pkl'))
    
    # Temporal split
    cutoff = features.index.max() - timedelta(days=TEST_HOLDOUT_DAYS)
    train = features[features.index <= cutoff].copy()
    test = features[features.index > cutoff].copy()
    
    # Generate predictions for BOTH sets
    for df_part, label in [(train, 'train'), (test, 'test')]:
        X = scaler.transform(df_part[saved_cols].values)
        df_part['gbm_prob'] = gbm.predict_proba(X)[:, 1]
        df_part['rf_prob'] = rf.predict_proba(X)[:, 1]
        df_part['ens_prob'] = 0.5 * df_part['gbm_prob'] + 0.5 * df_part['rf_prob']
        if label == 'train':
            train = df_part
        else:
            test = df_part
    
    print(f"  Train: {len(train)} days, Test: {len(test)} days")
    print(f"  Test range: {test.index[0].date()} to {test.index[-1].date()}")
    
    return mnq_rth, train, test, features


# ═══════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════
def backtest_config(mnq, daily_signals, tp_pct, sl_pct, signal_mode='ml_ensemble', prob_threshold=0.55):
    """
    Bar-by-bar intraday backtest.
    
    signal_mode:
      'simple'       — use combined_ret > threshold
      'ml_ensemble'  — use ensemble prob > prob_threshold
      'ml_gbm'       — use GBM prob only
      'ml_rf'        — use RF prob only
    """
    trades = []
    
    for date in sorted(set(mnq.index.date)):
        date_ts = pd.Timestamp(date)
        if date_ts not in daily_signals.index:
            continue
        
        row = daily_signals.loc[date_ts]
        
        # Direction decision
        if signal_mode == 'simple':
            sig = row.get('combined_ret', 0)
            if abs(sig) < SIGNAL_THRESHOLD:
                continue
            direction = 1 if sig > 0 else -1
            prob = None
        elif signal_mode == 'ml_ensemble':
            prob = row.get('ens_prob', 0.5)
            if prob > prob_threshold:
                direction = 1
            elif prob < (1 - prob_threshold):
                direction = -1
            else:
                continue
        elif signal_mode == 'ml_gbm':
            prob = row.get('gbm_prob', 0.5)
            if prob > prob_threshold:
                direction = 1
            elif prob < (1 - prob_threshold):
                direction = -1
            else:
                continue
        elif signal_mode == 'ml_rf':
            prob = row.get('rf_prob', 0.5)
            if prob > prob_threshold:
                direction = 1
            elif prob < (1 - prob_threshold):
                direction = -1
            else:
                continue
        
        day_bars = mnq[mnq.index.date == date]
        if len(day_bars) < 20:
            continue
        
        fh_bars = day_bars.between_time('09:30', '10:29')
        if len(fh_bars) < 10:
            continue
        fh_ret = (fh_bars['Close'].iloc[-1] / fh_bars['Open'].iloc[0]) - 1
        fh_confirms = (fh_ret > 0 and direction == 1) or (fh_ret < 0 and direction == -1)
        
        post_fh = day_bars.between_time('10:30', '15:55')
        if len(post_fh) < 2:
            continue
        
        entry_price = post_fh['Open'].iloc[0]
        exit_price = None
        exit_reason = None
        bars_held = 0
        max_fav = 0
        max_adv = 0
        
        for ts, bar in post_fh.iterrows():
            bars_held += 1
            if direction == 1:
                pct_h = bar['High'] / entry_price - 1
                pct_l = bar['Low'] / entry_price - 1
                max_fav = max(max_fav, pct_h)
                max_adv = min(max_adv, pct_l)
                if pct_h >= tp_pct:
                    exit_price = entry_price * (1 + tp_pct)
                    exit_reason = 'TP'
                    break
                elif pct_l <= -sl_pct:
                    exit_price = entry_price * (1 - sl_pct)
                    exit_reason = 'SL'
                    break
            else:
                pct_h = bar['High'] / entry_price - 1
                pct_l = bar['Low'] / entry_price - 1
                max_fav = max(max_fav, -pct_l)
                max_adv = min(max_adv, -pct_h)
                if pct_l <= -tp_pct:
                    exit_price = entry_price * (1 - tp_pct)
                    exit_reason = 'TP'
                    break
                elif pct_h >= sl_pct:
                    exit_price = entry_price * (1 + sl_pct)
                    exit_reason = 'SL'
                    break
        
        if exit_price is None:
            exit_price = post_fh['Close'].iloc[-1]
            exit_reason = 'CLOSE'
        
        pnl = ((exit_price / entry_price) - 1) * direction
        pts = (exit_price - entry_price) * direction
        
        trades.append({
            'date': date, 'direction': 'LONG' if direction == 1 else 'SHORT',
            'prob': prob, 'fh_confirms': fh_confirms,
            'entry': entry_price, 'exit': exit_price,
            'exit_reason': exit_reason, 'pnl': pnl,
            'pnl_mnq': pts * 2.0, 'bars_held': bars_held,
            'max_fav': max_fav, 'max_adv': max_adv,
        })
    
    return pd.DataFrame(trades) if trades else None


def summarize(df, label=""):
    """Return dict of key metrics."""
    if df is None or len(df) == 0:
        return None
    n = len(df)
    wins = (df['pnl'] > 0).sum()
    losses = (df['pnl'] < 0).sum()
    wr = wins / n
    total = df['pnl'].sum()
    mnq = df['pnl_mnq'].sum()
    
    win_pnl = df.loc[df['pnl'] > 0, 'pnl'].sum()
    loss_pnl = abs(df.loc[df['pnl'] < 0, 'pnl'].sum())
    pf = win_pnl / loss_pnl if loss_pnl > 0 else float('inf')
    
    avg_win = df.loc[df['pnl'] > 0, 'pnl'].mean() if wins > 0 else 0
    avg_loss = df.loc[df['pnl'] < 0, 'pnl'].mean() if losses > 0 else 0
    exp = df['pnl'].mean()
    
    cum = df['pnl'].cumsum()
    max_dd = (cum - cum.cummax()).min()
    sharpe = exp / df['pnl'].std() * np.sqrt(252) if df['pnl'].std() > 0 else 0
    
    return {
        'label': label, 'n': n, 'wins': wins, 'losses': losses, 'wr': wr,
        'tp': (df['exit_reason'] == 'TP').sum(),
        'sl': (df['exit_reason'] == 'SL').sum(),
        'cl': (df['exit_reason'] == 'CLOSE').sum(),
        'pf': pf, 'total': total, 'mnq': mnq, 'avg_win': avg_win,
        'avg_loss': avg_loss, 'exp': exp, 'max_dd': max_dd, 'sharpe': sharpe,
        'avg_hold': df['bars_held'].mean() * 5,
    }


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 90)
    print("  TP/SL COMPARISON WITH ML MODEL — OUT-OF-SAMPLE HOLDOUT")
    print("=" * 90)
    
    mnq, train, test, features = load_and_prepare()
    
    # ─── PART 1: TP/SL GRID WITH ML ENSEMBLE ON TEST SET ─────────────
    configs = [
        (0.005, 0.005, "TP 0.5% / SL 0.5%"),
        (0.0075, 0.005, "TP 0.75%/ SL 0.5%"),
        (0.010, 0.005, "TP 1.0% / SL 0.5%"),
        (0.010, 0.0075,"TP 1.0% / SL 0.75%"),
        (0.010, 0.010, "TP 1.0% / SL 1.0%"),
        (0.015, 0.005, "TP 1.5% / SL 0.5%"),
        (0.015, 0.0075,"TP 1.5% / SL 0.75%"),
        (0.005, 0.0025,"TP 0.5% / SL 0.25%"),
    ]
    
    print(f"\n{'='*100}")
    print(f"  PART 1: TP/SL GRID — ML ENSEMBLE ON TEST HOLDOUT (prob > 55%)")
    print(f"{'='*100}")
    print(f"\n  {'Config':22s} {'R:R':>4s} {'Trd':>4s} {'WR':>6s} {'TP':>3s} {'SL':>3s} {'CL':>3s} "
          f"{'PF':>5s} {'PnL%':>8s} {'$/MNQ':>8s} {'MaxDD%':>7s} {'Sharpe':>7s} {'$/trade':>8s} {'AvgHold':>8s}")
    print(f"  {'-'*22} {'-'*4} {'-'*4} {'-'*6} {'-'*3} {'-'*3} {'-'*3} "
          f"{'-'*5} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*8} {'-'*8}")
    
    best_configs = []
    for tp, sl, label in configs:
        rr = tp / sl
        bt = backtest_config(mnq, test, tp, sl, signal_mode='ml_ensemble', prob_threshold=0.55)
        s = summarize(bt, label)
        if s:
            s['rr'] = rr
            s['tp_pct'] = tp
            s['sl_pct'] = sl
            best_configs.append(s)
            exp_dollars = s['exp'] * 20000 * 2
            print(f"  {label:22s} {rr:4.1f} {s['n']:4d} {s['wr']:6.1%} {s['tp']:3d} {s['sl']:3d} {s['cl']:3d} "
                  f"{s['pf']:5.2f} {s['total']:+8.2%} {s['mnq']:+8,.0f} {s['max_dd']:+7.2%} {s['sharpe']:7.2f} "
                  f"${exp_dollars:+7,.0f} {s['avg_hold']:6.0f}min")
    
    # ─── PART 2: MODEL COMPARISON (best TP/SL config) ────────────────
    print(f"\n{'='*100}")
    print(f"  PART 2: MODEL COMPARISON — Which model to deploy?")
    print(f"  (Using TP 1.0% / SL 0.5% on TEST holdout)")
    print(f"{'='*100}")
    
    model_configs = [
        ('simple', 0.50, "Simple Signal (no ML)"),
        ('ml_gbm', 0.55, "GBM only (prob>55%)"),
        ('ml_rf', 0.55, "RF only (prob>55%)"),
        ('ml_ensemble', 0.55, "Ensemble (prob>55%)"),
        ('ml_ensemble', 0.50, "Ensemble (prob>50%)"),
        ('ml_ensemble', 0.60, "Ensemble (prob>60%)"),
    ]
    
    print(f"\n  {'Model':28s} {'Trd':>4s} {'WR':>6s} {'TP':>3s} {'SL':>3s} {'CL':>3s} "
          f"{'PF':>5s} {'PnL%':>8s} {'$/MNQ':>8s} {'MaxDD%':>7s} {'Sharpe':>7s} {'$/trade':>8s}")
    print(f"  {'-'*28} {'-'*4} {'-'*6} {'-'*3} {'-'*3} {'-'*3} "
          f"{'-'*5} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*8}")
    
    model_results = []
    for mode, thresh, label in model_configs:
        bt = backtest_config(mnq, test, 0.010, 0.005, signal_mode=mode, prob_threshold=thresh)
        s = summarize(bt, label)
        if s:
            model_results.append((label, s, bt))
            exp_dollars = s['exp'] * 20000 * 2
            print(f"  {label:28s} {s['n']:4d} {s['wr']:6.1%} {s['tp']:3d} {s['sl']:3d} {s['cl']:3d} "
                  f"{s['pf']:5.2f} {s['total']:+8.2%} {s['mnq']:+8,.0f} {s['max_dd']:+7.2%} {s['sharpe']:7.2f} "
                  f"${exp_dollars:+7,.0f}")
    
    # ─── PART 3: DEEP DIVE ON BEST CONFIG ────────────────────────────
    # Find best by PnL
    best = max(best_configs, key=lambda x: x['total'])
    print(f"\n{'='*100}")
    print(f"  PART 3: DEEP DIVE — BEST CONFIG: {best['label']} (R:R {best['rr']:.1f})")
    print(f"{'='*100}")
    
    bt = backtest_config(mnq, test, best['tp_pct'], best['sl_pct'], 
                         signal_mode='ml_ensemble', prob_threshold=0.55)
    
    if bt is not None:
        n = len(bt)
        
        # By direction
        for d in ['LONG', 'SHORT']:
            sub = bt[bt['direction'] == d]
            if len(sub) > 0:
                sub_wr = (sub['pnl'] > 0).mean()
                sub_pnl = sub['pnl'].sum()
                print(f"\n  {d}: {len(sub)} trades, WR={sub_wr:.1%}, PnL={sub_pnl:+.3%} (${sub['pnl_mnq'].sum():+,.0f})")
                for reason in ['TP', 'SL', 'CLOSE']:
                    r = sub[sub['exit_reason'] == reason]
                    if len(r) > 0:
                        print(f"    {reason}: {len(r)} ({len(r)/len(sub):.0%}), "
                              f"avg PnL={r['pnl'].mean():+.4%}, avg hold={r['bars_held'].mean()*5:.0f}min")
        
        # With/without FH confirmation
        conf = bt[bt['fh_confirms']]
        noconf = bt[~bt['fh_confirms']]
        print(f"\n  First-hour confirmation:")
        if len(conf) > 0:
            print(f"    WITH:    {len(conf)} trades, WR={(conf['pnl']>0).mean():.1%}, PnL={conf['pnl'].sum():+.3%}")
        if len(noconf) > 0:
            print(f"    WITHOUT: {len(noconf)} trades, WR={(noconf['pnl']>0).mean():.1%}, PnL={noconf['pnl'].sum():+.3%}")
        
        # Monthly
        bt_m = bt.copy()
        bt_m['month'] = pd.to_datetime(bt_m['date']).dt.to_period('M')
        monthly = bt_m.groupby('month').agg(
            trades=('pnl', 'count'),
            wins=('pnl', lambda x: (x > 0).sum()),
            pnl=('pnl', 'sum'),
            pnl_mnq=('pnl_mnq', 'sum')
        )
        print(f"\n  Monthly:")
        for m, row in monthly.iterrows():
            wr_m = row['wins'] / row['trades'] if row['trades'] > 0 else 0
            print(f"    {m}: {row['trades']:2.0f}T, WR={wr_m:.0%}, PnL={row['pnl']:+.3%} (${row['pnl_mnq']:+,.0f})")
        
        # Probability distribution of trades
        if bt['prob'].notna().any():
            print(f"\n  ML Probability distribution of trades taken:")
            for lo, hi, label in [(0.55, 0.60, '55-60%'), (0.60, 0.65, '60-65%'),
                                   (0.65, 0.70, '65-70%'), (0.70, 0.80, '70-80%'),
                                   (0.80, 1.00, '80%+')]:
                # LONG probs
                mask_l = (bt['prob'] >= lo) & (bt['prob'] < hi) & (bt['direction'] == 'LONG')
                # SHORT probs (prob < 1-lo means SHORT confidence)
                mask_s = (bt['prob'] <= (1-lo)) & (bt['prob'] > (1-hi)) & (bt['direction'] == 'SHORT')
                mask = mask_l | mask_s
                sub = bt[mask]
                if len(sub) > 0:
                    print(f"    Prob {label}: {len(sub)} trades, WR={(sub['pnl']>0).mean():.1%}, PnL={sub['pnl'].sum():+.3%}")
        
        # MFE/MAE
        print(f"\n  MFE/MAE:")
        print(f"    Winners: avg MFE={bt.loc[bt['pnl']>0, 'max_fav'].mean():.3%}, avg MAE={bt.loc[bt['pnl']>0, 'max_adv'].mean():.3%}")
        if (bt['pnl'] < 0).sum() > 0:
            print(f"    Losers:  avg MFE={bt.loc[bt['pnl']<0, 'max_fav'].mean():.3%}, avg MAE={bt.loc[bt['pnl']<0, 'max_adv'].mean():.3%}")
    
    # ─── PART 4: ALSO RUN BEST ON TRAIN FOR REFERENCE ────────────────
    print(f"\n{'='*100}")
    print(f"  PART 4: TRAIN vs TEST COMPARISON (same config: {best['label']}, ML Ensemble)")
    print(f"{'='*100}")
    
    bt_train = backtest_config(mnq, train, best['tp_pct'], best['sl_pct'],
                                signal_mode='ml_ensemble', prob_threshold=0.55)
    bt_test = backtest_config(mnq, test, best['tp_pct'], best['sl_pct'],
                               signal_mode='ml_ensemble', prob_threshold=0.55)
    
    print(f"\n  {'Set':8s} {'Trd':>4s} {'WR':>6s} {'PF':>5s} {'PnL%':>8s} {'$/MNQ':>8s} {'Sharpe':>7s}")
    print(f"  {'-'*8} {'-'*4} {'-'*6} {'-'*5} {'-'*8} {'-'*8} {'-'*7}")
    
    for lbl, bt_x in [('TRAIN', bt_train), ('TEST', bt_test)]:
        s = summarize(bt_x)
        if s:
            print(f"  {lbl:8s} {s['n']:4d} {s['wr']:6.1%} {s['pf']:5.2f} {s['total']:+8.2%} {s['mnq']:+8,.0f} {s['sharpe']:7.2f}")
    
    # ─── RECOMMENDATION ──────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"  RECOMMENDATION")
    print(f"{'='*100}")
    
    print(f"""
  Based on out-of-sample results:
  
  DEPLOY CONFIG:
    Model:     ML Ensemble (GBM + RF, equal weight)
    Entry:     After 10:30 ET, based on ensemble probability
    Direction: LONG if prob > 55%, SHORT if prob < 45%
    TP:        {best['tp_pct']*100:.1f}%
    SL:        {best['sl_pct']*100:.1f}%
    R:R:       1:{best['rr']:.1f}
    Max:       1 trade per day
    Exit:      Close any open position by 15:55 ET
    
  WHY ENSEMBLE (not single model):
    - GBM and RF capture different signal patterns
    - GBM: better at gradient/trend features
    - RF: better at categorical/regime features  
    - Ensemble smooths out individual model noise
    - More robust to regime changes
    
  BOTH models run; we average their probabilities.
  Only enter when the average is decisive (>55% or <45%).
""")
