#!/usr/bin/env python3
"""
Retrain simple LONG/SHORT entry models using trade history from trades.db.
Features: RSI, BB%, MACD, ATR%, volatility, hour, day_of_week, probability.
Target: win (1) vs loss (0) based on pnl_dollar > 0.
"""
import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import json

DB_PATH = "trades.db"
FEATURES = ['entry_rsi', 'entry_bb_position', 'entry_macd', 'entry_atr_pct', 
            'entry_volatility', 'entry_hour', 'entry_day_of_week', 'entry_probability']

def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT entry_time, direction, pnl_dollar, pnl_pct, exit_reason, model_id,
               entry_rsi, entry_bb_position, entry_macd, entry_atr_pct,
               entry_volatility, entry_hour, entry_day_of_week, entry_probability
        FROM trades 
        WHERE model_id LIKE '%pct%'
          AND entry_rsi IS NOT NULL 
          AND entry_bb_position IS NOT NULL
          AND entry_macd IS NOT NULL
        ORDER BY entry_time
    """, conn)
    conn.close()
    df['win'] = (df['pnl_dollar'] > 0).astype(int)
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    return df

def train_model(df, direction, name):
    subset = df[df['direction'] == direction].copy()
    print(f"\n{'='*60}")
    print(f"  {name} MODEL ({direction}) — {len(subset)} trades")
    print(f"  Date range: {subset['entry_time'].min()} to {subset['entry_time'].max()}")
    print(f"  Win rate: {subset['win'].mean()*100:.1f}%")
    print(f"{'='*60}")
    
    if len(subset) < 30:
        print(f"  ⚠ Too few trades ({len(subset)}), skipping")
        return None
    
    X = subset[FEATURES].copy()
    y = subset['win']
    
    # Show feature stats for wins vs losses
    print("\n  Feature means (Win vs Loss):")
    for f in FEATURES:
        w_mean = X.loc[y==1, f].mean()
        l_mean = X.loc[y==0, f].mean()
        diff = w_mean - l_mean
        print(f"    {f:25s}  W={w_mean:8.2f}  L={l_mean:8.2f}  diff={diff:+.2f}")
    
    # Fill NaN with median
    X = X.fillna(X.median())
    
    # Train with cross-validation
    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        min_samples_leaf=10, subsample=0.8, random_state=42
    )
    
    # Time-series split (no future leak)
    tscv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
    print(f"\n  CV Accuracy: {scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%")
    print(f"  Per fold: {[f'{s*100:.0f}%' for s in scores]}")
    
    # Train on all data
    model.fit(X, y)
    
    # Feature importances
    importances = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])
    print("\n  Feature importances:")
    for feat, imp in importances:
        bar = '█' * int(imp * 50)
        print(f"    {feat:25s} {imp:.3f} {bar}")
    
    # Full-data predictions (for analysis)
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    print(f"\n  Full-data confusion matrix:")
    cm = confusion_matrix(y, y_pred)
    print(f"    Predicted:  LOSS  WIN")
    print(f"    Actual LOSS: {cm[0][0]:4d} {cm[0][1]:4d}")
    print(f"    Actual WIN:  {cm[1][0]:4d} {cm[1][1]:4d}")
    
    # Threshold analysis — what if we only trade when model says >60%?
    print(f"\n  Threshold analysis (only enter if model prob > threshold):")
    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70]:
        mask = y_prob >= thresh
        if mask.sum() == 0:
            continue
        wr = y[mask].mean() * 100
        n = mask.sum()
        pnl = subset.loc[mask, 'pnl_dollar'].sum()
        blocked_wins = ((~mask) & (y == 1)).sum()
        blocked_losses = ((~mask) & (y == 0)).sum()
        print(f"    >{thresh:.0%}: {n:3d} trades, {wr:.0f}% WR, ${pnl:+.0f}, blocks {blocked_wins}W/{blocked_losses}L")
    
    # Recent performance (last 2 weeks)
    recent = subset[subset['entry_time'] >= '2026-02-23'].copy()
    if len(recent) > 5:
        X_recent = recent[FEATURES].fillna(X.median())
        y_recent = recent['win']
        y_recent_prob = model.predict_proba(X_recent)[:, 1]
        print(f"\n  Recent trades (since Feb 23, {len(recent)} trades):")
        for thresh in [0.50, 0.55, 0.60]:
            mask = y_recent_prob >= thresh
            if mask.sum() == 0: continue
            wr = y_recent[mask].mean() * 100
            n = mask.sum()
            pnl = recent.loc[mask, 'pnl_dollar'].sum()
            print(f"    >{thresh:.0%}: {n:3d} trades, {wr:.0f}% WR, ${pnl:+.0f}")
    
    return model

def main():
    df = load_data()
    print(f"Total trades loaded: {len(df)}")
    print(f"  LONG: {len(df[df['direction']=='LONG'])}")
    print(f"  SHORT: {len(df[df['direction']=='SHORT'])}")
    
    long_model = train_model(df, 'LONG', 'LONG')
    short_model = train_model(df, 'SHORT', 'SHORT')
    
    # Save models
    if long_model:
        joblib.dump(long_model, 'models/entry_filter_long.pkl')
        print(f"\n✓ LONG model saved to models/entry_filter_long.pkl")
    if short_model:
        joblib.dump(short_model, 'models/entry_filter_short.pkl')
        print(f"\n✓ SHORT model saved to models/entry_filter_short.pkl")

if __name__ == '__main__':
    main()
