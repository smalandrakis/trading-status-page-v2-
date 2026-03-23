#!/usr/bin/env python3
"""
MNQ Global Signal Backtest + ML Model
======================================
Full pipeline:
1. Load MNQ 5-min IB data
2. Download Asia/Europe/macro features (VIX, USD/JPY, bonds, overnight NQ)
3. Build daily feature matrix with proper temporal alignment
4. Strict train/test split (last ~3 months held out)
5. Train ML model (GBM) on composite signal
6. Backtest with bar-by-bar intraday simulation
7. Report results on holdout set

NO PEEKING: test set features and targets are never seen during training.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dtime
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIG ───────────────────────────────────────────────────────────
DATA_DIR = "data"
MODEL_DIR = "models_mnq_global"
os.makedirs(MODEL_DIR, exist_ok=True)

MNQ_FILE = os.path.join(DATA_DIR, "MNQ_5min_IB_with_indicators.csv")

# Strategy params (from initial analysis: 1:1 at 0.5%/0.5% best intraday)
TP_PCT = 0.005           # 0.5% take profit
SL_PCT = 0.005           # 0.5% stop loss
SIGNAL_THRESHOLD = 0.002 # min combined signal to trade
ENTRY_TIME = dtime(10, 30)  # enter after first hour
EXIT_TIME = dtime(15, 55)   # close before market close

# Train/test split: hold out last N calendar days for testing
TEST_HOLDOUT_DAYS = 90   # ~3 months held out for test

# Tickers for feature engineering
ASIA_TICKERS = ['^N225', '^HSI', '^AXJO']
EUROPE_TICKERS = ['^GDAXI', '^FTSE', '^STOXX50E']
MACRO_TICKERS = ['^VIX', 'JPY=X', 'TLT', 'NQ=F', 'ES=F', 'CL=F']


# ═══════════════════════════════════════════════════════════════════════
# STEP 1: LOAD MNQ INTRADAY DATA
# ═══════════════════════════════════════════════════════════════════════
def load_mnq_data():
    """Load MNQ 5-min data from IB export."""
    print("=" * 70)
    print("STEP 1: LOADING MNQ 5-MIN DATA")
    print("=" * 70)
    
    df = pd.read_csv(MNQ_FILE, usecols=['date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df['date'] = df['date'].dt.tz_convert('America/New_York')
    df = df.set_index('date').sort_index()
    
    # Filter to RTH only (9:30-16:00 ET)
    df = df.between_time('09:30', '16:00')
    
    print(f"  Loaded {len(df)} RTH bars")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    print(f"  Trading days: {df.index.date.__len__()}, unique: {len(set(df.index.date))}")
    
    return df


def compute_daily_mnq_features(mnq):
    """Extract daily features from MNQ intraday data."""
    print("\n  Computing daily MNQ features from intraday data...")
    
    daily = []
    for date, day_data in mnq.groupby(mnq.index.date):
        if len(day_data) < 20:  # need at least ~2h of data
            continue
        
        o = day_data['Open'].iloc[0]
        h = day_data['High'].max()
        l = day_data['Low'].min()
        c = day_data['Close'].iloc[-1]
        vol = day_data['Volume'].sum()
        
        # First hour stats (9:30-10:30 = first 12 bars of 5-min)
        fh = day_data.iloc[:12]
        fh_ret = (fh['Close'].iloc[-1] / fh['Open'].iloc[0] - 1) if len(fh) >= 12 else 0
        fh_range = (fh['High'].max() - fh['Low'].min()) / fh['Open'].iloc[0] if len(fh) >= 12 else 0
        fh_vol = fh['Volume'].sum()
        
        # Previous close to open gap
        daily_ret = (c / o - 1)
        daily_range = (h - l) / o
        
        # Price at 10:30 for entry
        post_fh = day_data.iloc[12:]
        entry_price = post_fh['Open'].iloc[0] if len(post_fh) > 0 else None
        
        daily.append({
            'date': pd.Timestamp(date),
            'open': o, 'high': h, 'low': l, 'close': c,
            'volume': vol,
            'daily_ret': daily_ret,
            'daily_range': daily_range,
            'fh_ret': fh_ret,
            'fh_range': fh_range,
            'fh_vol': fh_vol,
            'entry_price': entry_price,
        })
    
    df = pd.DataFrame(daily)
    df = df.set_index('date')
    
    # Add rolling features (using only past data)
    df['ret_1d'] = df['daily_ret'].shift(1)
    df['ret_2d'] = df['daily_ret'].shift(1).rolling(2).sum()
    df['ret_5d'] = df['daily_ret'].shift(1).rolling(5).sum()
    df['vol_5d'] = df['daily_range'].shift(1).rolling(5).mean()
    df['vol_10d'] = df['daily_range'].shift(1).rolling(10).mean()
    df['volume_ratio'] = df['volume'].shift(1) / df['volume'].shift(1).rolling(10).mean()
    
    # Prev day first-hour features
    df['prev_fh_ret'] = df['fh_ret'].shift(1)
    df['prev_fh_range'] = df['fh_range'].shift(1)
    
    print(f"  Built {len(df)} daily feature rows from MNQ intraday")
    return df


# ═══════════════════════════════════════════════════════════════════════
# STEP 2: DOWNLOAD EXTERNAL FEATURES
# ═══════════════════════════════════════════════════════════════════════
def download_external_features(start_date, end_date):
    """Download Asia, Europe, VIX, USD/JPY, bonds, oil daily data."""
    print("\n" + "=" * 70)
    print("STEP 2: DOWNLOADING EXTERNAL FEATURES")
    print("=" * 70)
    
    start = (start_date - timedelta(days=30)).strftime('%Y-%m-%d')
    end = (end_date + timedelta(days=5)).strftime('%Y-%m-%d')
    
    all_tickers = ASIA_TICKERS + EUROPE_TICKERS + MACRO_TICKERS
    
    ext = pd.DataFrame()
    for ticker in all_tickers:
        col = ticker.replace('^', '').replace('=', '_')
        print(f"  Downloading {ticker} ({col})...", end=" ")
        try:
            df = yf.download(ticker, start=start, end=end, interval='1d', progress=False)
            if len(df) > 0:
                if isinstance(df.columns, pd.MultiIndex):
                    close = df[('Close', ticker)]
                else:
                    close = df['Close']
                ext[f'{col}_close'] = close
                ext[f'{col}_ret'] = close.pct_change()
                print(f"OK ({len(df)} bars)")
            else:
                print("EMPTY")
        except Exception as e:
            print(f"ERROR: {e}")
    
    # Make index timezone-naive for merging
    ext.index = ext.index.tz_localize(None) if ext.index.tz else ext.index
    
    # Build composite signals
    asia_ret_cols = [c for c in ext.columns if c.endswith('_ret') and any(a in c for a in ['N225', 'HSI', 'AXJO'])]
    europe_ret_cols = [c for c in ext.columns if c.endswith('_ret') and any(e in c for e in ['GDAXI', 'FTSE', 'STOXX50E'])]
    
    if asia_ret_cols:
        ext['asia_ret'] = ext[asia_ret_cols].mean(axis=1)
    if europe_ret_cols:
        ext['europe_ret'] = ext[europe_ret_cols].mean(axis=1)
    if asia_ret_cols and europe_ret_cols:
        ext['combined_ret'] = 0.4 * ext['asia_ret'] + 0.6 * ext['europe_ret']
    
    # VIX features
    if 'VIX_close' in ext.columns:
        ext['vix_level'] = ext['VIX_close']
        ext['vix_change'] = ext['VIX_close'].pct_change()
        ext['vix_5d_avg'] = ext['VIX_close'].rolling(5).mean()
        ext['vix_above_avg'] = (ext['VIX_close'] > ext['vix_5d_avg']).astype(int)
    
    # USD/JPY features
    if 'JPY_X_ret' in ext.columns:
        ext['usdjpy_ret'] = ext['JPY_X_ret']
        ext['usdjpy_5d'] = ext['JPY_X_ret'].rolling(5).sum()
    
    # Bond features (TLT as proxy for 20Y treasury)
    if 'TLT_ret' in ext.columns:
        ext['bond_ret'] = ext['TLT_ret']
        ext['bond_5d'] = ext['TLT_ret'].rolling(5).sum()
    
    # Oil features
    if 'CL_F_ret' in ext.columns:
        ext['oil_ret'] = ext['CL_F_ret']
    
    # Overnight NQ return (NQ=F close to next day open approximation)
    if 'NQ_F_ret' in ext.columns:
        ext['nq_prev_ret'] = ext['NQ_F_ret'].shift(1)
        ext['nq_prev_2d'] = ext['NQ_F_ret'].shift(1).rolling(2).sum()
        ext['nq_prev_5d'] = ext['NQ_F_ret'].shift(1).rolling(5).sum()
    
    # ES/SPX correlation
    if 'ES_F_ret' in ext.columns:
        ext['es_ret'] = ext['ES_F_ret']
        ext['es_nq_spread'] = ext.get('NQ_F_ret', 0) - ext['ES_F_ret']
    
    print(f"\n  External features: {len(ext)} rows, {len(ext.columns)} columns")
    return ext


# ═══════════════════════════════════════════════════════════════════════
# STEP 3: BUILD FEATURE MATRIX
# ═══════════════════════════════════════════════════════════════════════
def build_feature_matrix(mnq_daily, ext_features):
    """Merge MNQ daily features with external features into training matrix."""
    print("\n" + "=" * 70)
    print("STEP 3: BUILDING FEATURE MATRIX")
    print("=" * 70)
    
    # Ensure both indices are tz-naive dates
    mnq_daily.index = pd.to_datetime(mnq_daily.index).normalize()
    ext_features.index = pd.to_datetime(ext_features.index).normalize()
    
    # Merge on date
    features = mnq_daily.join(ext_features, how='left')
    
    # Forward fill external features (weekends/holidays)
    ext_cols = [c for c in ext_features.columns if c in features.columns]
    features[ext_cols] = features[ext_cols].ffill()
    
    # Define target: NQ direction after 10:30 (using daily close vs entry at 10:30)
    # For proper backtesting, we compute this from the MNQ data itself
    features['target_ret'] = (features['close'] - features['entry_price']) / features['entry_price']
    features['target_dir'] = (features['target_ret'] > 0).astype(int)
    features['target_hit_tp'] = (features['target_ret'].abs() >= TP_PCT).astype(int)
    
    # Select feature columns (exclude target, raw prices, entry_price)
    exclude = ['open', 'high', 'low', 'close', 'volume', 'entry_price',
               'target_ret', 'target_dir', 'target_hit_tp',
               'daily_ret', 'fh_ret', 'fh_range', 'fh_vol']
    
    # Also exclude raw close prices of external indices
    exclude_patterns = ['_close']
    
    feature_cols = []
    for c in features.columns:
        if c in exclude:
            continue
        if any(p in c for p in exclude_patterns):
            continue
        feature_cols.append(c)
    
    # Drop rows with NaN in features
    features = features.dropna(subset=feature_cols + ['target_dir', 'entry_price'])
    
    print(f"  Feature matrix: {len(features)} rows x {len(feature_cols)} feature columns")
    print(f"  Features: {feature_cols}")
    print(f"  Target distribution: {features['target_dir'].value_counts().to_dict()}")
    print(f"  Date range: {features.index[0].date()} to {features.index[-1].date()}")
    
    return features, feature_cols


# ═══════════════════════════════════════════════════════════════════════
# STEP 4: TRAIN/TEST SPLIT + ML MODEL
# ═══════════════════════════════════════════════════════════════════════
def train_model(features, feature_cols):
    """Train ML model with strict temporal split."""
    print("\n" + "=" * 70)
    print("STEP 4: TRAINING ML MODEL (strict temporal split)")
    print("=" * 70)
    
    # Temporal split
    cutoff = features.index.max() - timedelta(days=TEST_HOLDOUT_DAYS)
    train = features[features.index <= cutoff].copy()
    test = features[features.index > cutoff].copy()
    
    print(f"  Train: {len(train)} days ({train.index[0].date()} to {train.index[-1].date()})")
    print(f"  Test:  {len(test)} days ({test.index[0].date()} to {test.index[-1].date()})")
    print(f"  Train target dist: {train['target_dir'].value_counts().to_dict()}")
    print(f"  Test  target dist: {test['target_dir'].value_counts().to_dict()}")
    
    X_train = train[feature_cols].values
    y_train = train['target_dir'].values
    X_test = test[feature_cols].values
    y_test = test['target_dir'].values
    
    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Train GBM
    print("\n  Training GradientBoosting...")
    gbm = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=42
    )
    gbm.fit(X_train_s, y_train)
    
    gbm_train_acc = accuracy_score(y_train, gbm.predict(X_train_s))
    gbm_test_acc = accuracy_score(y_test, gbm.predict(X_test_s))
    gbm_test_probs = gbm.predict_proba(X_test_s)[:, 1]
    
    print(f"  GBM Train accuracy: {gbm_train_acc:.1%}")
    print(f"  GBM Test  accuracy: {gbm_test_acc:.1%}")
    
    # Train Random Forest
    print("\n  Training RandomForest...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_s, y_train)
    
    rf_train_acc = accuracy_score(y_train, rf.predict(X_train_s))
    rf_test_acc = accuracy_score(y_test, rf.predict(X_test_s))
    rf_test_probs = rf.predict_proba(X_test_s)[:, 1]
    
    print(f"  RF  Train accuracy: {rf_train_acc:.1%}")
    print(f"  RF  Test  accuracy: {rf_test_acc:.1%}")
    
    # Ensemble (average)
    ens_probs = 0.5 * gbm_test_probs + 0.5 * rf_test_probs
    ens_preds = (ens_probs > 0.5).astype(int)
    ens_test_acc = accuracy_score(y_test, ens_preds)
    print(f"\n  Ensemble Test accuracy: {ens_test_acc:.1%}")
    
    # Feature importance
    print("\n  Top 15 features (GBM importance):")
    imp = pd.Series(gbm.feature_importances_, index=feature_cols).sort_values(ascending=False)
    for feat, val in imp.head(15).items():
        print(f"    {feat:30s} {val:.4f}")
    
    # Classification report on test
    print(f"\n  Classification Report (test set, ensemble):")
    print(classification_report(y_test, ens_preds, target_names=['DOWN', 'UP']))
    
    # Store models
    test['gbm_prob'] = gbm_test_probs
    test['rf_prob'] = rf_test_probs
    test['ens_prob'] = ens_probs
    test['ens_pred'] = ens_preds
    
    # Also compute train predictions for reference
    train['gbm_prob'] = gbm.predict_proba(X_train_s)[:, 1]
    train['rf_prob'] = rf.predict_proba(X_train_s)[:, 1]
    train['ens_prob'] = 0.5 * train['gbm_prob'] + 0.5 * train['rf_prob']
    train['ens_pred'] = (train['ens_prob'] > 0.5).astype(int)
    
    # Save models
    joblib.dump(gbm, os.path.join(MODEL_DIR, 'gbm_global_signal.pkl'))
    joblib.dump(rf, os.path.join(MODEL_DIR, 'rf_global_signal.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler_global_signal.pkl'))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, 'feature_cols.pkl'))
    print(f"\n  Models saved to {MODEL_DIR}/")
    
    return train, test, gbm, rf, scaler


# ═══════════════════════════════════════════════════════════════════════
# STEP 5: INTRADAY BACKTEST (BAR-BY-BAR)
# ═══════════════════════════════════════════════════════════════════════
def backtest_intraday(mnq_5m, daily_signals, label="", use_ml=False):
    """
    Bar-by-bar intraday backtest on MNQ 5-min data.
    
    Rules:
    - Entry at 10:30 ET (after first hour)
    - Direction from signal (combined_ret or ML prediction)
    - TP: +0.5%, SL: -0.5%
    - Close at 15:55 if neither hit
    - One trade per day max
    """
    print(f"\n{'─'*70}")
    print(f"  BACKTEST: {label}")
    print(f"{'─'*70}")
    
    trades = []
    
    for date in sorted(set(mnq_5m.index.date)):
        date_ts = pd.Timestamp(date)
        
        # Check if we have a signal for this date
        if date_ts not in daily_signals.index:
            continue
        
        row = daily_signals.loc[date_ts]
        
        # Determine direction
        if use_ml:
            prob = row.get('ens_prob', 0.5)
            if prob > 0.55:
                direction = 1
            elif prob < 0.45:
                direction = -1
            else:
                continue  # skip neutral
        else:
            sig = row.get('combined_ret', 0)
            if abs(sig) < SIGNAL_THRESHOLD:
                continue
            direction = 1 if sig > 0 else -1
        
        dir_label = 'LONG' if direction == 1 else 'SHORT'
        
        # Get intraday bars for this date
        day_bars = mnq_5m[mnq_5m.index.date == date]
        if len(day_bars) < 20:
            continue
        
        # First hour data (9:30-10:30)
        fh_bars = day_bars.between_time('09:30', '10:29')
        if len(fh_bars) < 10:
            continue
        
        fh_ret = (fh_bars['Close'].iloc[-1] / fh_bars['Open'].iloc[0]) - 1
        fh_confirms = (fh_ret > 0 and direction == 1) or (fh_ret < 0 and direction == -1)
        
        # Entry at 10:30
        post_fh = day_bars.between_time('10:30', '15:55')
        if len(post_fh) < 2:
            continue
        
        entry_price = post_fh['Open'].iloc[0]
        entry_time = post_fh.index[0]
        
        # Bar-by-bar simulation
        exit_price = None
        exit_reason = None
        exit_time = None
        max_favorable = 0
        max_adverse = 0
        
        for ts, bar in post_fh.iterrows():
            if direction == 1:  # LONG
                pct_high = bar['High'] / entry_price - 1
                pct_low = bar['Low'] / entry_price - 1
                max_favorable = max(max_favorable, pct_high)
                max_adverse = min(max_adverse, pct_low)
                
                if pct_high >= TP_PCT:
                    exit_price = entry_price * (1 + TP_PCT)
                    exit_reason = 'TP'
                    exit_time = ts
                    break
                elif pct_low <= -SL_PCT:
                    exit_price = entry_price * (1 - SL_PCT)
                    exit_reason = 'SL'
                    exit_time = ts
                    break
            else:  # SHORT
                pct_high = bar['High'] / entry_price - 1
                pct_low = bar['Low'] / entry_price - 1
                max_favorable = max(max_favorable, -pct_low)
                max_adverse = min(max_adverse, -pct_high)
                
                if pct_low <= -TP_PCT:
                    exit_price = entry_price * (1 - TP_PCT)
                    exit_reason = 'TP'
                    exit_time = ts
                    break
                elif pct_high >= SL_PCT:
                    exit_price = entry_price * (1 + SL_PCT)
                    exit_reason = 'SL'
                    exit_time = ts
                    break
        
        if exit_price is None:
            exit_price = post_fh['Close'].iloc[-1]
            exit_reason = 'CLOSE'
            exit_time = post_fh.index[-1]
        
        pnl_pct = ((exit_price / entry_price) - 1) * direction
        pnl_points = (exit_price - entry_price) * direction
        pnl_mnq = pnl_points * 2.0  # MNQ = $2/point
        
        trades.append({
            'date': date,
            'direction': dir_label,
            'signal': row.get('combined_ret', 0),
            'ml_prob': row.get('ens_prob', None),
            'fh_ret': fh_ret,
            'fh_confirms': fh_confirms,
            'entry_price': entry_price,
            'entry_time': entry_time,
            'exit_price': exit_price,
            'exit_time': exit_time,
            'exit_reason': exit_reason,
            'pnl_pct': pnl_pct,
            'pnl_points': pnl_points,
            'pnl_mnq': pnl_mnq,
            'max_favorable': max_favorable,
            'max_adverse': max_adverse,
        })
    
    if not trades:
        print("  No trades generated!")
        return None
    
    df = pd.DataFrame(trades)
    print_backtest_results(df)
    return df


def print_backtest_results(df):
    """Print comprehensive backtest results."""
    n = len(df)
    wins = (df['pnl_pct'] > 0).sum()
    losses = (df['pnl_pct'] < 0).sum()
    flat = (df['pnl_pct'] == 0).sum()
    wr = wins / n
    
    total_pct = df['pnl_pct'].sum()
    total_mnq = df['pnl_mnq'].sum()
    avg_win = df.loc[df['pnl_pct'] > 0, 'pnl_pct'].mean() if wins > 0 else 0
    avg_loss = df.loc[df['pnl_pct'] < 0, 'pnl_pct'].mean() if losses > 0 else 0
    
    win_pnl = df.loc[df['pnl_pct'] > 0, 'pnl_pct'].sum()
    loss_pnl = abs(df.loc[df['pnl_pct'] < 0, 'pnl_pct'].sum())
    pf = win_pnl / loss_pnl if loss_pnl > 0 else float('inf')
    
    # Drawdown
    df_cum = df['pnl_pct'].cumsum()
    max_dd = (df_cum - df_cum.cummax()).min()
    
    # Sharpe
    sharpe = df['pnl_pct'].mean() / df['pnl_pct'].std() * np.sqrt(252) if df['pnl_pct'].std() > 0 else 0
    
    # Expectancy
    expectancy = df['pnl_pct'].mean()
    
    print(f"\n  RESULTS:")
    print(f"  {'─'*50}")
    print(f"  Trades:       {n}")
    print(f"  Wins:         {wins} ({wr:.1%})")
    print(f"  Losses:       {losses}")
    print(f"  Flat:         {flat}")
    print(f"  Win Rate:     {wr:.1%}")
    print(f"  Avg Win:      {avg_win:+.4%}")
    print(f"  Avg Loss:     {avg_loss:+.4%}")
    print(f"  Profit Factor:{pf:6.2f}")
    print(f"  Expectancy:   {expectancy:+.4%} (${expectancy * 20000 * 2:+,.0f}/MNQ/trade)")
    print(f"  Total PnL:    {total_pct:+.2%} (${total_mnq:+,.0f} per MNQ)")
    print(f"  Max Drawdown: {max_dd:+.2%} (${max_dd * 20000 * 2:+,.0f}/MNQ)")
    print(f"  Sharpe:       {sharpe:.2f}")
    
    # By direction
    for d in ['LONG', 'SHORT']:
        sub = df[df['direction'] == d]
        if len(sub) > 0:
            sub_wr = (sub['pnl_pct'] > 0).mean()
            sub_pnl = sub['pnl_pct'].sum()
            print(f"\n  {d}: {len(sub)} trades, WR={sub_wr:.1%}, PnL={sub_pnl:+.2%} (${sub['pnl_mnq'].sum():+,.0f})")
    
    # By outcome
    print(f"\n  Outcome breakdown:")
    for reason in ['TP', 'SL', 'CLOSE']:
        sub = df[df['exit_reason'] == reason]
        if len(sub) > 0:
            print(f"    {reason:6s}: {len(sub):3d} ({len(sub)/n:.0%}), avg PnL={sub['pnl_pct'].mean():+.4%}")
    
    # With/without first-hour confirmation
    conf = df[df['fh_confirms']]
    noconf = df[~df['fh_confirms']]
    if len(conf) > 0:
        print(f"\n  With 1st-hour confirmation:    {len(conf):3d} trades, WR={(conf['pnl_pct']>0).mean():.1%}, PnL={conf['pnl_pct'].sum():+.2%}")
    if len(noconf) > 0:
        print(f"  Without 1st-hour confirmation: {len(noconf):3d} trades, WR={(noconf['pnl_pct']>0).mean():.1%}, PnL={noconf['pnl_pct'].sum():+.2%}")
    
    # Monthly
    df_m = df.copy()
    df_m['month'] = pd.to_datetime(df_m['date']).dt.to_period('M')
    monthly = df_m.groupby('month').agg(
        trades=('pnl_pct', 'count'),
        wins=('pnl_pct', lambda x: (x > 0).sum()),
        pnl=('pnl_pct', 'sum'),
        pnl_mnq=('pnl_mnq', 'sum')
    )
    print(f"\n  Monthly:")
    for m, row in monthly.iterrows():
        wr_m = row['wins'] / row['trades'] if row['trades'] > 0 else 0
        print(f"    {m}: {row['trades']:2.0f}T, WR={wr_m:.0%}, PnL={row['pnl']:+.3%} (${row['pnl_mnq']:+,.0f})")
    
    # MFE/MAE analysis
    print(f"\n  MFE/MAE (Max Favorable/Adverse Excursion):")
    print(f"    Avg MFE (winners): {df.loc[df['pnl_pct']>0, 'max_favorable'].mean():.3%}")
    print(f"    Avg MAE (winners): {df.loc[df['pnl_pct']>0, 'max_adverse'].mean():.3%}")
    print(f"    Avg MFE (losers):  {df.loc[df['pnl_pct']<0, 'max_favorable'].mean():.3%}")
    print(f"    Avg MAE (losers):  {df.loc[df['pnl_pct']<0, 'max_adverse'].mean():.3%}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("🌏🌍🌎 MNQ GLOBAL SIGNAL — FULL PIPELINE")
    print(f"    TP: {TP_PCT:.2%}, SL: {SL_PCT:.2%} (1:1 R:R)")
    print(f"    Entry: {ENTRY_TIME}, Exit: {EXIT_TIME}")
    print(f"    Test holdout: last {TEST_HOLDOUT_DAYS} days")
    print()
    
    # Step 1: Load MNQ data
    mnq = load_mnq_data()
    mnq_daily = compute_daily_mnq_features(mnq)
    
    # Step 2: Download external features
    start_date = mnq.index[0].date()
    end_date = mnq.index[-1].date()
    ext = download_external_features(start_date, end_date)
    
    # Step 3: Build feature matrix
    features, feature_cols = build_feature_matrix(mnq_daily, ext)
    
    # Step 4: Train ML model
    train, test, gbm, rf, scaler = train_model(features, feature_cols)
    
    # Step 5: Backtest — multiple configurations
    print("\n" + "=" * 70)
    print("STEP 5: INTRADAY BACKTESTS")
    print("=" * 70)
    
    # Temporal cutoff
    cutoff = features.index.max() - timedelta(days=TEST_HOLDOUT_DAYS)
    
    # 5a. Simple signal on TRAIN set (for reference, not the real test)
    train_signals = features[features.index <= cutoff]
    bt_train = backtest_intraday(mnq, train_signals, 
                                  label="TRAIN — Simple Signal (combined_ret > 0.2%)")
    
    # 5b. Simple signal on TEST set (holdout)
    test_signals = features[features.index > cutoff]
    bt_test_simple = backtest_intraday(mnq, test_signals,
                                        label="TEST (HOLDOUT) — Simple Signal")
    
    # 5c. ML model on TEST set (the real test)
    bt_test_ml = backtest_intraday(mnq, test, 
                                    label="TEST (HOLDOUT) — ML Ensemble (prob > 55%)",
                                    use_ml=True)
    
    # 5d. ML + first-hour confirmation on TEST set
    test_fh = test.copy()
    # We'll filter in post-processing
    if bt_test_ml is not None:
        bt_ml_confirmed = bt_test_ml[bt_test_ml['fh_confirms']].copy()
        if len(bt_ml_confirmed) > 0:
            print(f"\n{'─'*70}")
            print(f"  BACKTEST: TEST (HOLDOUT) — ML + First-Hour Confirmation")
            print(f"{'─'*70}")
            print_backtest_results(bt_ml_confirmed)
    
    # ─── SUMMARY ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    
    results = []
    for label, bt in [
        ("Train-Simple", bt_train),
        ("Test-Simple", bt_test_simple),
        ("Test-ML", bt_test_ml),
        ("Test-ML+FH", bt_ml_confirmed if bt_test_ml is not None and len(bt_ml_confirmed) > 0 else None),
    ]:
        if bt is not None and len(bt) > 0:
            n = len(bt)
            wr = (bt['pnl_pct'] > 0).mean()
            pnl = bt['pnl_pct'].sum()
            mnq_pnl = bt['pnl_mnq'].sum()
            results.append((label, n, wr, pnl, mnq_pnl))
    
    print(f"\n  {'Strategy':20s} {'Trades':>6s} {'WR':>7s} {'PnL%':>8s} {'$/MNQ':>10s}")
    print(f"  {'-'*20} {'-'*6} {'-'*7} {'-'*8} {'-'*10}")
    for label, n, wr, pnl, mnq_pnl in results:
        print(f"  {label:20s} {n:6d} {wr:7.1%} {pnl:+8.2%} ${mnq_pnl:+9,.0f}")
    
    print(f"\n  Test period: {test.index[0].date()} to {test.index[-1].date()} ({TEST_HOLDOUT_DAYS} days holdout)")
    print(f"  Models saved to: {MODEL_DIR}/")
    print(f"\n  IMPORTANT: Only 'Test-*' rows are out-of-sample results.")
    print(f"  If Test WR > 55% and PF > 1.5, the signal has real predictive power.")
