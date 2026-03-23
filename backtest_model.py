#!/usr/bin/env python3
"""Walk-forward backtest: train on first 70%, test on last 30%."""
import sqlite3, pandas as pd, numpy as np
from sklearn.ensemble import GradientBoostingClassifier

FEATURES = ['entry_rsi', 'entry_bb_position', 'entry_macd', 'entry_atr_pct', 
            'entry_volatility', 'entry_hour', 'entry_day_of_week', 'entry_probability']

conn = sqlite3.connect("trades.db")
df = pd.read_sql_query("""
    SELECT entry_time, direction, pnl_dollar, entry_rsi, entry_bb_position, 
           entry_macd, entry_atr_pct, entry_volatility, entry_hour, 
           entry_day_of_week, entry_probability
    FROM trades WHERE model_id LIKE '%pct%'
      AND entry_rsi IS NOT NULL AND entry_bb_position IS NOT NULL AND entry_macd IS NOT NULL
    ORDER BY entry_time
""", conn)
conn.close()
df['win'] = (df['pnl_dollar'] > 0).astype(int)

for direction in ['LONG', 'SHORT']:
    sub = df[df['direction'] == direction].reset_index(drop=True)
    split = int(len(sub) * 0.70)
    train, test = sub.iloc[:split], sub.iloc[split:]
    
    X_tr = train[FEATURES].fillna(train[FEATURES].median())
    X_te = test[FEATURES].fillna(train[FEATURES].median())
    
    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        min_samples_leaf=10, subsample=0.8, random_state=42)
    model.fit(X_tr, train['win'])
    probs = model.predict_proba(X_te)[:, 1]
    
    print(f"\n{'='*60}")
    print(f"{direction}: Train {split} ({train['entry_time'].iloc[0][:10]} → {train['entry_time'].iloc[-1][:10]})")
    print(f"{direction}: Test  {len(test)} ({test['entry_time'].iloc[0][:10]} → {test['entry_time'].iloc[-1][:10]})")
    
    base_pnl = test['pnl_dollar'].sum()
    base_wr = test['win'].mean() * 100
    print(f"\nBaseline (all): {len(test)} trades, {base_wr:.0f}% WR, ${base_pnl:+.1f}")
    
    for t in [0.45, 0.50, 0.55, 0.60, 0.65]:
        m = probs >= t
        if m.sum() == 0: continue
        wr = test.loc[m, 'win'].mean() * 100
        pnl = test.loc[m, 'pnl_dollar'].sum()
        bw = ((~m) & (test['win'] == 1)).sum()
        bl = ((~m) & (test['win'] == 0)).sum()
        print(f"  >{t:.0%}: {m.sum():3d} trades, {wr:.0f}% WR, ${pnl:+.1f} (blocks {bw}W/{bl}L)")
    
    # Last 15 trades detail
    print(f"\nLast 15 test trades:")
    for i in range(max(0, len(test)-15), len(test)):
        row = test.iloc[i]
        p = probs[i]
        act = 'WIN' if row['pnl_dollar'] > 0 else 'LOSS'
        do = 'TAKE' if p >= 0.50 else 'SKIP'
        ok = 'ok' if (do=='TAKE' and act=='WIN') or (do=='SKIP' and act=='LOSS') else 'XX'
        print(f"  {row['entry_time'][:16]} ${row['pnl_dollar']:+7.1f} {act:4s} prob={p:.0%} → {do} {ok}")
