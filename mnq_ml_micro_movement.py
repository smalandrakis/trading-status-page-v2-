#!/usr/bin/env python3
"""
MNQ ML Micro-Movement Model

Different approach: Predict ANY positive movement and use aggressive trailing
to capture small profits before reversals.

Key insight from previous analysis:
- ML model has 88-93% precision for predicting 0.2% moves
- But trades aren't profitable because we're not capturing the moves

New strategy:
1. Predict direction with high confidence
2. Enter immediately
3. Use very tight trailing stop (0.1%) that activates early
4. Exit with small profit rather than waiting for full target
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def load_data():
    df = pd.read_parquet('data/QQQ_features.parquet')
    df_15sec = pd.read_parquet('data/QQQ_15sec_30days.parquet')
    return df, df_15sec


def add_features(df):
    """Add synthetic features."""
    df = df.copy()
    
    # Cross-indicator
    df['rsi_macd_cross'] = df['momentum_rsi'] * np.sign(df['trend_macd_diff'])
    df['bb_rsi_combo'] = df['volatility_bbp'] * df['momentum_rsi'] / 100
    df['adx_trend'] = df['trend_adx'] * np.sign(df['trend_adx_pos'] - df['trend_adx_neg'])
    
    # Momentum
    df['rsi_accel'] = df['momentum_rsi'] - df['momentum_rsi_lag1']
    df['macd_accel'] = df['trend_macd_diff'] - df['trend_macd_diff_lag1']
    
    # Volatility
    df['vol_regime'] = df['volatility_atr'] / df['volatility_atr'].rolling(50).mean()
    
    # Price structure
    df['higher_high'] = (df['High'] > df['High'].shift(1)).astype(int)
    df['higher_low'] = (df['Low'] > df['Low'].shift(1)).astype(int)
    df['bullish_candle'] = (df['Close'] > df['Open']).astype(int)
    df['consec_up'] = df['bullish_candle'].rolling(3).sum()
    df['consec_down'] = (1 - df['bullish_candle']).rolling(3).sum()
    
    # Volume
    df['vol_spike'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    return df


def create_micro_labels(df, min_move_pct=0.1, horizon_bars=6):
    """
    Create labels for micro-movements.
    
    Label = 1 if price moves UP by at least min_move_pct within horizon_bars
    Label = -1 if price moves DOWN by at least min_move_pct within horizon_bars
    Label = 0 if neither (choppy/sideways)
    """
    df = df.copy()
    
    # Max up move
    future_high = df['High'].rolling(horizon_bars).max().shift(-horizon_bars)
    up_move = (future_high / df['Close'] - 1) * 100
    
    # Max down move
    future_low = df['Low'].rolling(horizon_bars).min().shift(-horizon_bars)
    down_move = (df['Close'] / future_low - 1) * 100
    
    # Which happens first? (simplified: just use magnitude)
    df['label_up'] = (up_move >= min_move_pct).astype(int)
    df['label_down'] = (down_move >= min_move_pct).astype(int)
    
    # Net direction (which move is larger)
    df['net_move'] = up_move - down_move
    df['label'] = np.where(df['net_move'] > 0.05, 1, np.where(df['net_move'] < -0.05, -1, 0))
    
    return df


def get_features(df):
    exclude = ['Open', 'High', 'Low', 'Close', 'Volume', 'label', 'label_up', 'label_down', 'net_move']
    return [c for c in df.columns if c not in exclude and not c.startswith('future_')]


def train_direction_model(df):
    """Train model to predict direction."""
    
    df = create_micro_labels(df, min_move_pct=0.15, horizon_bars=6)
    features = get_features(df)
    
    df_clean = df.dropna(subset=features + ['label'])
    
    # Only train on clear signals (not sideways)
    df_train = df_clean[df_clean['label'] != 0]
    
    X = df_train[features]
    y = (df_train['label'] == 1).astype(int)  # 1 = up, 0 = down
    
    print(f"Training samples: {len(X)}")
    print(f"Up signals: {y.sum()} ({y.mean()*100:.1f}%)")
    
    # Time split
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    model = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.08,
        min_samples_leaf=30,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train_s, y_train)
    
    # Test
    probs = model.predict_proba(X_test_s)[:, 1]
    
    print("\nTest precision by threshold:")
    for thresh in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
        pred = (probs >= thresh).astype(int)
        if pred.sum() > 0:
            correct = (pred == y_test).sum()
            prec = (pred & y_test).sum() / pred.sum()
            print(f"  @{thresh:.0%}: precision={prec:.1%}, signals={pred.sum()}")
    
    return model, scaler, features


def simulate_micro_trades(model, scaler, features, df_5min, df_15sec,
                          prob_long=0.65, prob_short=0.35,
                          initial_sl=0.3, trail_pct=0.1, trail_activation=0.08,
                          max_hold_min=30):
    """
    Simulate trades with micro trailing stops using 15-sec data.
    
    Strategy:
    - LONG when prob > prob_long
    - SHORT when prob < prob_short
    - Very tight trailing stop (0.1%) activates after 0.08% profit
    - Quick exits to capture small moves
    """
    
    # Align timezones
    if df_5min.index.tz is None and df_15sec.index.tz is not None:
        df_5min.index = df_5min.index.tz_localize(df_15sec.index.tz)
    
    # Get predictions
    df_5min = df_5min.copy()
    X = df_5min[features].dropna()
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[:, 1]
    df_5min.loc[X.index, 'prob'] = probs
    
    trades = []
    last_trade = None
    cooldown = timedelta(minutes=15)
    
    for idx in X.index:
        prob = df_5min.loc[idx, 'prob']
        
        # Determine direction
        if prob >= prob_long:
            direction = 'LONG'
        elif prob <= prob_short:
            direction = 'SHORT'
        else:
            continue
        
        # Cooldown
        if last_trade and (idx - last_trade) < cooldown:
            continue
        
        entry_price = df_5min.loc[idx, 'Close']
        
        # Get 15-sec bars
        end_time = idx + timedelta(minutes=max_hold_min)
        trade_bars = df_15sec[(df_15sec.index > idx) & (df_15sec.index <= end_time)]
        
        if len(trade_bars) < 4:
            continue
        
        # Initialize
        if direction == 'LONG':
            sl_price = entry_price * (1 - initial_sl / 100)
        else:
            sl_price = entry_price * (1 + initial_sl / 100)
        
        exit_price = None
        exit_reason = None
        peak = entry_price
        trough = entry_price
        trailing_active = False
        max_profit = 0
        
        for bar_time, bar in trade_bars.iterrows():
            if direction == 'LONG':
                peak = max(peak, bar['high'])
                current_profit = (peak / entry_price - 1) * 100
                max_profit = max(max_profit, current_profit)
                
                # Activate trailing
                if current_profit >= trail_activation:
                    trailing_active = True
                    new_sl = peak * (1 - trail_pct / 100)
                    sl_price = max(sl_price, new_sl)
                
                # Check stop
                if bar['low'] <= sl_price:
                    exit_price = sl_price
                    exit_reason = 'TS' if trailing_active else 'SL'
                    break
            else:
                trough = min(trough, bar['low'])
                current_profit = (entry_price / trough - 1) * 100
                max_profit = max(max_profit, current_profit)
                
                if current_profit >= trail_activation:
                    trailing_active = True
                    new_sl = trough * (1 + trail_pct / 100)
                    sl_price = min(sl_price, new_sl)
                
                if bar['high'] >= sl_price:
                    exit_price = sl_price
                    exit_reason = 'TS' if trailing_active else 'SL'
                    break
        
        # Timeout
        if exit_price is None:
            exit_price = trade_bars.iloc[-1]['close']
            exit_reason = 'TO'
        
        # P&L
        if direction == 'LONG':
            pnl_pct = (exit_price / entry_price - 1) * 100
        else:
            pnl_pct = (entry_price / exit_price - 1) * 100
        
        pnl_dollar = pnl_pct * 5 - 1.24  # MNQ approx
        
        trades.append({
            'time': idx,
            'direction': direction,
            'prob': prob,
            'entry': entry_price,
            'exit': exit_price,
            'reason': exit_reason,
            'pnl_pct': pnl_pct,
            'pnl_dollar': pnl_dollar,
            'max_profit': max_profit,
            'trailing_active': trailing_active
        })
        
        last_trade = idx
    
    return pd.DataFrame(trades)


def analyze(trades_df):
    if trades_df.empty:
        return None
    
    total = len(trades_df)
    wins = (trades_df['pnl_dollar'] > 0).sum()
    wr = wins / total * 100
    total_pnl = trades_df['pnl_dollar'].sum()
    
    days = max(1, (trades_df['time'].max() - trades_df['time'].min()).days)
    pnl_week = total_pnl / days * 7
    
    exits = trades_df['reason'].value_counts().to_dict()
    
    # Profit capture rate
    avg_max = trades_df['max_profit'].mean()
    avg_realized = trades_df['pnl_pct'].mean()
    
    return {
        'trades': total,
        'wr': wr,
        'pnl_week': pnl_week,
        'total_pnl': total_pnl,
        'exits': exits,
        'avg_max_profit': avg_max,
        'avg_realized': avg_realized,
        'capture_rate': avg_realized / avg_max * 100 if avg_max > 0 else 0
    }


def main():
    print("="*70)
    print("MNQ ML MICRO-MOVEMENT MODEL")
    print("="*70)
    
    df, df_15sec = load_data()
    df = add_features(df)
    
    # Align timezones
    test_start = df_15sec.index[0]
    if df.index.tz is None:
        df.index = df.index.tz_localize(test_start.tz)
    
    # Train on data before test period
    df_train = df[df.index < test_start]
    df_test = df[(df.index >= test_start) & (df.index <= df_15sec.index[-1])]
    
    print(f"\nTraining data: {len(df_train)} bars")
    print(f"Test data: {len(df_test)} bars")
    
    # Train model
    print("\n" + "="*70)
    print("TRAINING DIRECTION MODEL")
    print("="*70)
    
    model, scaler, features = train_direction_model(df_train)
    
    # Test configurations
    print("\n" + "="*70)
    print("BACKTESTING MICRO-MOVEMENT STRATEGY")
    print("="*70)
    
    configs = [
        # Aggressive trailing
        {'prob_l': 0.60, 'prob_s': 0.40, 'sl': 0.25, 'trail': 0.08, 'act': 0.05, 'hold': 20},
        {'prob_l': 0.60, 'prob_s': 0.40, 'sl': 0.30, 'trail': 0.10, 'act': 0.08, 'hold': 30},
        {'prob_l': 0.65, 'prob_s': 0.35, 'sl': 0.25, 'trail': 0.08, 'act': 0.05, 'hold': 20},
        {'prob_l': 0.65, 'prob_s': 0.35, 'sl': 0.30, 'trail': 0.10, 'act': 0.08, 'hold': 30},
        {'prob_l': 0.65, 'prob_s': 0.35, 'sl': 0.35, 'trail': 0.12, 'act': 0.10, 'hold': 30},
        {'prob_l': 0.70, 'prob_s': 0.30, 'sl': 0.25, 'trail': 0.08, 'act': 0.05, 'hold': 20},
        {'prob_l': 0.70, 'prob_s': 0.30, 'sl': 0.30, 'trail': 0.10, 'act': 0.08, 'hold': 30},
        {'prob_l': 0.70, 'prob_s': 0.30, 'sl': 0.35, 'trail': 0.12, 'act': 0.10, 'hold': 30},
        # Wider trailing
        {'prob_l': 0.65, 'prob_s': 0.35, 'sl': 0.40, 'trail': 0.15, 'act': 0.12, 'hold': 45},
        {'prob_l': 0.70, 'prob_s': 0.30, 'sl': 0.40, 'trail': 0.15, 'act': 0.12, 'hold': 45},
        # Very tight
        {'prob_l': 0.65, 'prob_s': 0.35, 'sl': 0.20, 'trail': 0.06, 'act': 0.04, 'hold': 15},
        {'prob_l': 0.70, 'prob_s': 0.30, 'sl': 0.20, 'trail': 0.06, 'act': 0.04, 'hold': 15},
        # LONG only (based on model showing better LONG precision)
        {'prob_l': 0.65, 'prob_s': 0.00, 'sl': 0.30, 'trail': 0.10, 'act': 0.08, 'hold': 30},
        {'prob_l': 0.70, 'prob_s': 0.00, 'sl': 0.30, 'trail': 0.10, 'act': 0.08, 'hold': 30},
        {'prob_l': 0.75, 'prob_s': 0.00, 'sl': 0.30, 'trail': 0.10, 'act': 0.08, 'hold': 30},
    ]
    
    all_results = []
    
    for cfg in configs:
        trades = simulate_micro_trades(
            model, scaler, features, df_test, df_15sec,
            prob_long=cfg['prob_l'], prob_short=cfg['prob_s'],
            initial_sl=cfg['sl'], trail_pct=cfg['trail'], 
            trail_activation=cfg['act'], max_hold_min=cfg['hold']
        )
        
        result = analyze(trades)
        if result and result['trades'] >= 5:
            result['config'] = cfg
            all_results.append(result)
    
    # Sort by P&L
    all_results.sort(key=lambda x: x['pnl_week'], reverse=True)
    
    print(f"\n{'ProbL':<7} {'ProbS':<7} {'SL':<6} {'Trail':<7} {'Hold':<6} {'Trades':<8} {'WR%':<7} {'$/Wk':<8} {'Capture'}")
    print("-"*80)
    
    for r in all_results:
        c = r['config']
        print(f"{c['prob_l']:<7.0%} {c['prob_s']:<7.0%} {c['sl']:<6.2f} {c['trail']:<7.2f} {c['hold']:<6} {r['trades']:<8} {r['wr']:<6.0f}% ${r['pnl_week']:<7.0f} {r['capture_rate']:.0f}%")
    
    # Find profitable
    profitable = [r for r in all_results if r['pnl_week'] > 0]
    
    print(f"\n\nPROFITABLE CONFIGURATIONS: {len(profitable)}")
    print("="*70)
    
    if profitable:
        for r in profitable:
            c = r['config']
            exits = ', '.join([f"{k}:{v}" for k, v in r['exits'].items()])
            print(f"\n  ✅ ProbL={c['prob_l']:.0%} ProbS={c['prob_s']:.0%} SL={c['sl']}% Trail={c['trail']}%")
            print(f"     WR: {r['wr']:.0f}%, P&L: ${r['pnl_week']:.0f}/week, Trades: {r['trades']}")
            print(f"     Avg max profit: {r['avg_max_profit']:.2f}%, Realized: {r['avg_realized']:.2f}%")
            print(f"     Capture rate: {r['capture_rate']:.0f}%, Exits: {exits}")
    else:
        print("\n  No profitable configuration found.")
        
        # Analyze why
        if all_results:
            best = all_results[0]
            print(f"\n  Best result: ${best['pnl_week']:.0f}/week")
            print(f"  Avg max profit seen: {best['avg_max_profit']:.2f}%")
            print(f"  Avg realized: {best['avg_realized']:.2f}%")
            print(f"  Capture rate: {best['capture_rate']:.0f}%")
            print(f"\n  Issue: Even when direction is correct, trailing stops are")
            print(f"  getting hit before capturing enough profit.")
    
    # Show trade details for best config
    if all_results:
        print("\n" + "="*70)
        print("SAMPLE TRADES FROM BEST CONFIG")
        print("="*70)
        
        best_cfg = all_results[0]['config']
        trades = simulate_micro_trades(
            model, scaler, features, df_test, df_15sec,
            prob_long=best_cfg['prob_l'], prob_short=best_cfg['prob_s'],
            initial_sl=best_cfg['sl'], trail_pct=best_cfg['trail'], 
            trail_activation=best_cfg['act'], max_hold_min=best_cfg['hold']
        )
        
        print(f"\n{'Time':<20} {'Dir':<6} {'Prob':<6} {'MaxProf':<8} {'Realized':<9} {'P&L':<8} {'Exit'}")
        print("-"*75)
        for _, t in trades.head(15).iterrows():
            print(f"{str(t['time'])[:19]:<20} {t['direction']:<6} {t['prob']:.0%}   {t['max_profit']:>6.2f}%  {t['pnl_pct']:>7.2f}%  ${t['pnl_dollar']:>6.1f}  {t['reason']}")
    
    return all_results


if __name__ == "__main__":
    results = main()
