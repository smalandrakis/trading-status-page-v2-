#!/usr/bin/env python3
"""
Validate BTC signal data integrity by backtesting against parquet data.
Compare actual trades with what backtest would have produced.
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import joblib

BTC_PARQUET = "data/BTC_features.parquet"
DB_PATH = "trades.db"

# Model configurations (from btc_ensemble_bot.py)
MODELS = {
    '2h_0.5pct': {'path': 'models_btc_v2/model_2h_0.5pct.joblib', 'direction': 'LONG', 'threshold': 0.5, 'horizon': 24},
    '4h_0.5pct': {'path': 'models_btc_v2/model_4h_0.5pct.joblib', 'direction': 'LONG', 'threshold': 0.5, 'horizon': 48},
    '2h_0.5pct_SHORT': {'path': 'models_btc_v2/model_2h_0.5pct_SHORT.joblib', 'direction': 'SHORT', 'threshold': 0.5, 'horizon': 24},
    '4h_0.5pct_SHORT': {'path': 'models_btc_v2/model_4h_0.5pct_SHORT.joblib', 'direction': 'SHORT', 'threshold': 0.5, 'horizon': 48},
}

# Indicator strategy configs
INDICATOR_LONG_SL = 0.10
INDICATOR_LONG_TP = 0.70
INDICATOR_MEANREV_SL = 0.30
INDICATOR_MEANREV_TP = 0.50
INDICATOR_TREND_SL = 0.70
INDICATOR_TREND_TP = 1.40
INDICATOR_TREND_ROC_THRESHOLD = 0.4

def load_models():
    """Load ML models."""
    models = {}
    for name, cfg in MODELS.items():
        try:
            models[name] = {
                'model': joblib.load(cfg['path']),
                'config': cfg
            }
        except Exception as e:
            print(f"Could not load {name}: {e}")
    return models

def backtest_ml_models(df, models, prob_threshold=0.40):
    """Backtest ML models on parquet data."""
    results = []
    
    feature_cols = list(models[list(models.keys())[0]]['model'].feature_names_in_)
    
    # Get last 48 hours of data
    cutoff = df.index[-1] - timedelta(hours=48)
    df_48h = df[df.index >= cutoff].copy()
    
    print(f"\nBacktesting ML models on {len(df_48h)} bars (last 48h)")
    
    for model_name, model_data in models.items():
        model = model_data['model']
        cfg = model_data['config']
        direction = cfg['direction']
        sl_pct = 0.30  # Default SL
        tp_pct = cfg['threshold']
        max_hold = cfg['horizon'] * 2
        
        trades = []
        last_trade_bar = -20  # Cooldown
        
        for i in range(50, len(df_48h) - max_hold):
            if i - last_trade_bar < 20:
                continue
            
            row = df_48h.iloc[i:i+1]
            X = row[feature_cols]
            
            try:
                prob = model.predict_proba(X)[0, 1]
            except:
                continue
            
            if prob < prob_threshold:
                continue
            
            entry_price = df_48h['Close'].iloc[i]
            entry_time = df_48h.index[i]
            
            if direction == 'LONG':
                target_price = entry_price * (1 + tp_pct / 100)
                stop_price = entry_price * (1 - sl_pct / 100)
            else:
                target_price = entry_price * (1 - tp_pct / 100)
                stop_price = entry_price * (1 + sl_pct / 100)
            
            exit_price = None
            exit_reason = None
            
            for j in range(i + 1, min(i + max_hold + 1, len(df_48h))):
                future = df_48h.iloc[j]
                
                if direction == 'LONG':
                    if future['Low'] <= stop_price:
                        exit_price = stop_price
                        exit_reason = 'SL'
                        break
                    if future['High'] >= target_price:
                        exit_price = target_price
                        exit_reason = 'TP'
                        break
                else:
                    if future['High'] >= stop_price:
                        exit_price = stop_price
                        exit_reason = 'SL'
                        break
                    if future['Low'] <= target_price:
                        exit_price = target_price
                        exit_reason = 'TP'
                        break
            
            if exit_price is None:
                exit_price = df_48h['Close'].iloc[min(i + max_hold, len(df_48h) - 1)]
                exit_reason = 'TO'
            
            if direction == 'LONG':
                pnl_pct = (exit_price / entry_price - 1) * 100
            else:
                pnl_pct = (entry_price / exit_price - 1) * 100
            
            pnl_dollar = pnl_pct / 100 * entry_price * 0.001 - 1.24  # Commission
            
            trades.append({
                'model': model_name,
                'direction': direction,
                'entry_time': entry_time,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'exit_reason': exit_reason,
                'pnl_pct': pnl_pct,
                'pnl_dollar': pnl_dollar,
                'prob': prob
            })
            
            last_trade_bar = i
        
        results.extend(trades)
    
    return pd.DataFrame(results)

def backtest_indicator_trend(df):
    """Backtest ROC+MACD trend strategy."""
    results = []
    
    cutoff = df.index[-1] - timedelta(hours=48)
    df_48h = df[df.index >= cutoff].copy()
    
    # Calculate ROC(12)
    df_48h['roc_12'] = (df_48h['Close'] - df_48h['Close'].shift(12)) / df_48h['Close'].shift(12) * 100
    
    # MACD histogram
    df_48h['macd_hist'] = df_48h['trend_macd'] - df_48h['trend_macd_signal']
    df_48h['macd_hist_prev'] = df_48h['macd_hist'].shift(1)
    
    # Entry conditions
    df_48h['long_signal'] = (
        (df_48h['roc_12'] > INDICATOR_TREND_ROC_THRESHOLD) & 
        (df_48h['macd_hist'] > 0) & 
        (df_48h['macd_hist'] > df_48h['macd_hist_prev'])
    )
    df_48h['short_signal'] = (
        (df_48h['roc_12'] < -INDICATOR_TREND_ROC_THRESHOLD) & 
        (df_48h['macd_hist'] < 0) & 
        (df_48h['macd_hist'] < df_48h['macd_hist_prev'])
    )
    
    print(f"\nBacktesting indicator_trend on {len(df_48h)} bars")
    print(f"  LONG signals: {df_48h['long_signal'].sum()}")
    print(f"  SHORT signals: {df_48h['short_signal'].sum()}")
    
    last_trade_bar = -20
    max_hold = 48
    
    for i in range(50, len(df_48h) - max_hold):
        if i - last_trade_bar < 20:
            continue
        
        row = df_48h.iloc[i]
        entry_price = row['Close']
        entry_time = df_48h.index[i]
        
        direction = None
        if row['long_signal']:
            direction = 'LONG'
            target_price = entry_price * (1 + INDICATOR_TREND_TP / 100)
            stop_price = entry_price * (1 - INDICATOR_TREND_SL / 100)
        elif row['short_signal']:
            direction = 'SHORT'
            target_price = entry_price * (1 - INDICATOR_TREND_TP / 100)
            stop_price = entry_price * (1 + INDICATOR_TREND_SL / 100)
        else:
            continue
        
        exit_price = None
        exit_reason = None
        
        for j in range(i + 1, min(i + max_hold + 1, len(df_48h))):
            future = df_48h.iloc[j]
            
            if direction == 'LONG':
                if future['Low'] <= stop_price:
                    exit_price = stop_price
                    exit_reason = 'SL'
                    break
                if future['High'] >= target_price:
                    exit_price = target_price
                    exit_reason = 'TP'
                    break
            else:
                if future['High'] >= stop_price:
                    exit_price = stop_price
                    exit_reason = 'SL'
                    break
                if future['Low'] <= target_price:
                    exit_price = target_price
                    exit_reason = 'TP'
                    break
        
        if exit_price is None:
            exit_price = df_48h['Close'].iloc[min(i + max_hold, len(df_48h) - 1)]
            exit_reason = 'TO'
        
        if direction == 'LONG':
            pnl_pct = (exit_price / entry_price - 1) * 100
        else:
            pnl_pct = (entry_price / exit_price - 1) * 100
        
        pnl_dollar = pnl_pct / 100 * entry_price * 0.001 - 1.24
        
        results.append({
            'model': 'indicator_trend',
            'direction': direction,
            'entry_time': entry_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'pnl_pct': pnl_pct,
            'pnl_dollar': pnl_dollar,
            'roc_12': row['roc_12'],
            'macd_hist': row['macd_hist']
        })
        
        last_trade_bar = i
    
    return pd.DataFrame(results)

def main():
    print("="*80)
    print("BTC SIGNAL DATA INTEGRITY VALIDATION")
    print("="*80)
    
    # Load parquet
    print("\nLoading BTC parquet...")
    df = pd.read_parquet(BTC_PARQUET)
    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    
    # Load models
    print("\nLoading ML models...")
    models = load_models()
    print(f"Loaded {len(models)} models")
    
    # Backtest ML models
    ml_results = backtest_ml_models(df, models, prob_threshold=0.40)
    
    # Backtest indicator_trend
    trend_results = backtest_indicator_trend(df)
    
    # Combine results
    all_results = pd.concat([ml_results, trend_results], ignore_index=True)
    
    if all_results.empty:
        print("\nNo backtest trades generated!")
        return
    
    print("\n" + "="*80)
    print("BACKTEST RESULTS (Last 48 Hours)")
    print("="*80)
    
    # Overall
    total = len(all_results)
    wins = len(all_results[all_results['pnl_dollar'] > 0])
    total_pnl = all_results['pnl_dollar'].sum()
    
    print(f"\n📊 OVERALL:")
    print(f"  Total trades: {total}")
    print(f"  Win rate: {wins/total*100:.1f}%")
    print(f"  Total P&L: ${total_pnl:.2f}")
    
    # By model
    print(f"\n📈 BY MODEL:")
    for model in all_results['model'].unique():
        model_df = all_results[all_results['model'] == model]
        trades = len(model_df)
        wins = len(model_df[model_df['pnl_dollar'] > 0])
        wr = wins / trades * 100 if trades > 0 else 0
        pnl = model_df['pnl_dollar'].sum()
        print(f"  {model}: {trades} trades, {wr:.1f}% WR, ${pnl:.2f}")
    
    # Compare with actual trades
    print("\n" + "="*80)
    print("COMPARISON: BACKTEST vs ACTUAL TRADES")
    print("="*80)
    
    conn = sqlite3.connect(DB_PATH)
    actual_df = pd.read_sql_query("""
        SELECT model_id, direction, pnl_dollar, exit_reason
        FROM trades 
        WHERE bot_type='BTC' 
        AND exit_time > datetime('now', '-48 hours')
    """, conn)
    conn.close()
    
    print(f"\n  Backtest trades: {len(all_results)}")
    print(f"  Actual trades: {len(actual_df)}")
    
    print(f"\n  Backtest P&L: ${all_results['pnl_dollar'].sum():.2f}")
    print(f"  Actual P&L: ${actual_df['pnl_dollar'].sum():.2f}")
    
    # By model comparison
    print(f"\n  BY MODEL COMPARISON:")
    print(f"  {'Model':<20} {'BT Trades':<12} {'Actual':<12} {'BT P&L':<12} {'Actual P&L':<12}")
    print("-"*70)
    
    all_models = set(all_results['model'].unique()) | set(actual_df['model_id'].unique())
    for model in sorted(all_models):
        bt_df = all_results[all_results['model'] == model]
        act_df = actual_df[actual_df['model_id'] == model]
        
        bt_trades = len(bt_df)
        act_trades = len(act_df)
        bt_pnl = bt_df['pnl_dollar'].sum() if len(bt_df) > 0 else 0
        act_pnl = act_df['pnl_dollar'].sum() if len(act_df) > 0 else 0
        
        print(f"  {model:<20} {bt_trades:<12} {act_trades:<12} ${bt_pnl:<11.2f} ${act_pnl:<11.2f}")
    
    # Data integrity check
    print("\n" + "="*80)
    print("DATA INTEGRITY CHECK")
    print("="*80)
    
    # Check if backtest and actual are roughly aligned
    bt_total = all_results['pnl_dollar'].sum()
    act_total = actual_df['pnl_dollar'].sum()
    
    diff = abs(bt_total - act_total)
    if diff < 50:
        print("\n  ✅ Data integrity looks GOOD - backtest and actual P&L are close")
    elif diff < 200:
        print("\n  ⚠️ Data integrity WARNING - some discrepancy between backtest and actual")
        print(f"     Difference: ${diff:.2f}")
    else:
        print("\n  ❌ Data integrity ISSUE - significant discrepancy")
        print(f"     Difference: ${diff:.2f}")
        print("     Possible causes: timing differences, price slippage, or data issues")


if __name__ == "__main__":
    main()
