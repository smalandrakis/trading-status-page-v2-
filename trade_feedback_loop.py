#!/usr/bin/env python3
"""
Trade Feedback Loop System

This module implements a self-learning feedback loop that:
1. Analyzes past trades to identify winning/losing patterns
2. Tracks which features/indicators correlate with trade success
3. Provides a signal quality score for new trades
4. Continuously updates feature importance based on outcomes

The system learns from:
- Initial price direction after entry (first 5-10 bars)
- Max Favorable Excursion (MFE) - how far price went in our favor
- Max Adverse Excursion (MAE) - how far price went against us
- Feature values at entry time vs trade outcome
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import os
import logging

logger = logging.getLogger(__name__)

# Feature importance database
FEEDBACK_DB = 'trade_feedback.db'

class TradeFeedbackLoop:
    """
    Self-learning system that analyzes trades and identifies patterns
    that predict winning vs losing trades.
    """
    
    def __init__(self, trades_db: str = 'trades.db', min_trades: int = 20):
        self.trades_db = trades_db
        self.feedback_db = FEEDBACK_DB
        self.min_trades = min_trades  # Minimum trades before making predictions
        
        # Feature importance scores (updated after each trade)
        self.feature_scores = {}
        
        # Initialize feedback database
        self._init_feedback_db()
        
        # Load existing feature scores
        self._load_feature_scores()
    
    def _init_feedback_db(self):
        """Initialize the feedback database tables."""
        conn = sqlite3.connect(self.feedback_db)
        cursor = conn.cursor()
        
        # Table for feature importance scores
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feature_importance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feature_name TEXT NOT NULL,
                bot_type TEXT NOT NULL,
                direction TEXT NOT NULL,
                
                -- Correlation with trade success
                win_correlation REAL DEFAULT 0,
                
                -- Optimal ranges for this feature
                optimal_min REAL,
                optimal_max REAL,
                
                -- Statistics
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                
                -- When feature is in optimal range
                optimal_range_trades INTEGER DEFAULT 0,
                optimal_range_wins INTEGER DEFAULT 0,
                
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                UNIQUE(feature_name, bot_type, direction)
            )
        ''')
        
        # Table for trade analysis results
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trade_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT NOT NULL UNIQUE,
                bot_type TEXT NOT NULL,
                model_id TEXT NOT NULL,
                direction TEXT NOT NULL,
                
                -- Entry conditions
                entry_time TIMESTAMP,
                entry_price REAL,
                entry_rsi REAL,
                entry_macd REAL,
                entry_bb_position REAL,
                entry_atr_pct REAL,
                entry_probability REAL,
                entry_hour INTEGER,
                entry_volatility REAL,
                
                -- Price movement analysis
                initial_direction_correct INTEGER,  -- 1 if price moved right way in first 5 bars
                bars_to_first_profit INTEGER,       -- How many bars until first profitable
                max_favorable_excursion REAL,       -- Best unrealized P&L
                max_adverse_excursion REAL,         -- Worst unrealized P&L
                mfe_bar INTEGER,                    -- Bar number of MFE
                mae_bar INTEGER,                    -- Bar number of MAE
                
                -- Outcome
                exit_reason TEXT,
                pnl_pct REAL,
                pnl_dollar REAL,
                is_winner INTEGER,
                
                -- Quality score at entry (if available)
                entry_quality_score REAL,
                
                analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Table for pattern recognition
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS winning_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_name TEXT NOT NULL,
                bot_type TEXT NOT NULL,
                direction TEXT NOT NULL,
                
                -- Pattern conditions (JSON)
                conditions JSON,
                
                -- Performance
                total_matches INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0,
                avg_pnl REAL DEFAULT 0,
                
                -- Is this pattern active for filtering?
                is_active INTEGER DEFAULT 1,
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_feature_scores(self):
        """Load feature importance scores from database."""
        try:
            conn = sqlite3.connect(self.feedback_db)
            query = "SELECT * FROM feature_importance"
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            for _, row in df.iterrows():
                key = f"{row['bot_type']}_{row['direction']}_{row['feature_name']}"
                self.feature_scores[key] = {
                    'win_correlation': row['win_correlation'],
                    'optimal_min': row['optimal_min'],
                    'optimal_max': row['optimal_max'],
                    'total_trades': row['total_trades'],
                    'winning_trades': row['winning_trades'],
                    'optimal_range_trades': row['optimal_range_trades'],
                    'optimal_range_wins': row['optimal_range_wins'],
                }
        except Exception as e:
            logger.warning(f"Could not load feature scores: {e}")
    
    def analyze_trade(self, trade_id: str, price_data: pd.DataFrame = None) -> Dict:
        """
        Analyze a single trade to extract learning signals.
        
        Args:
            trade_id: The trade ID to analyze
            price_data: Optional price data for MFE/MAE calculation
            
        Returns:
            Dictionary with analysis results
        """
        # Get trade from database
        conn = sqlite3.connect(self.trades_db)
        query = f"SELECT * FROM trades WHERE trade_id = '{trade_id}' OR id = '{trade_id}'"
        trade_df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(trade_df) == 0:
            return {'error': 'Trade not found'}
        
        trade = trade_df.iloc[0]
        
        analysis = {
            'trade_id': trade_id,
            'bot_type': trade['bot_type'],
            'model_id': trade['model_id'],
            'direction': trade['direction'],
            'entry_time': trade['entry_time'],
            'entry_price': trade['entry_price'],
            'exit_reason': trade['exit_reason'],
            'pnl_pct': trade['pnl_pct'],
            'pnl_dollar': trade['pnl_dollar'],
            'is_winner': 1 if trade['pnl_dollar'] > 0 else 0,
            
            # Entry conditions
            'entry_rsi': trade.get('entry_rsi'),
            'entry_macd': trade.get('entry_macd'),
            'entry_bb_position': trade.get('entry_bb_position'),
            'entry_atr_pct': trade.get('entry_atr_pct'),
            'entry_probability': trade.get('entry_probability'),
            'entry_hour': trade.get('entry_hour'),
            'entry_volatility': trade.get('entry_volatility'),
        }
        
        # If we have price data, calculate MFE/MAE
        if price_data is not None and len(price_data) > 0:
            mfe_mae = self._calculate_mfe_mae(
                price_data, 
                trade['entry_price'], 
                trade['direction'],
                trade['entry_time'],
                trade['exit_time']
            )
            analysis.update(mfe_mae)
        
        # Save analysis to database
        self._save_trade_analysis(analysis)
        
        return analysis
    
    def _calculate_mfe_mae(self, price_data: pd.DataFrame, entry_price: float,
                          direction: str, entry_time: str, exit_time: str) -> Dict:
        """Calculate Max Favorable and Adverse Excursion."""
        try:
            # Filter price data to trade duration
            mask = (price_data.index >= entry_time) & (price_data.index <= exit_time)
            trade_prices = price_data.loc[mask, 'close'] if 'close' in price_data.columns else price_data.loc[mask, 'Close']
            
            if len(trade_prices) == 0:
                return {}
            
            # Calculate excursions
            if direction.upper() == 'LONG':
                excursions = (trade_prices - entry_price) / entry_price * 100
            else:  # SHORT
                excursions = (entry_price - trade_prices) / entry_price * 100
            
            mfe = excursions.max()
            mae = excursions.min()
            mfe_bar = excursions.idxmax() if mfe > 0 else None
            mae_bar = excursions.idxmin() if mae < 0 else None
            
            # Check initial direction (first 5 bars)
            initial_excursion = excursions.iloc[:5].mean() if len(excursions) >= 5 else excursions.mean()
            initial_direction_correct = 1 if initial_excursion > 0 else 0
            
            # Bars to first profit
            profitable_bars = excursions[excursions > 0]
            bars_to_first_profit = profitable_bars.index[0] if len(profitable_bars) > 0 else -1
            
            return {
                'max_favorable_excursion': mfe,
                'max_adverse_excursion': mae,
                'mfe_bar': str(mfe_bar) if mfe_bar else None,
                'mae_bar': str(mae_bar) if mae_bar else None,
                'initial_direction_correct': initial_direction_correct,
                'bars_to_first_profit': bars_to_first_profit,
            }
        except Exception as e:
            logger.warning(f"Error calculating MFE/MAE: {e}")
            return {}
    
    def _save_trade_analysis(self, analysis: Dict):
        """Save trade analysis to database."""
        conn = sqlite3.connect(self.feedback_db)
        cursor = conn.cursor()
        
        columns = ', '.join(analysis.keys())
        placeholders = ', '.join(['?' for _ in analysis])
        values = list(analysis.values())
        
        try:
            cursor.execute(f'''
                INSERT OR REPLACE INTO trade_analysis ({columns})
                VALUES ({placeholders})
            ''', values)
            conn.commit()
        except Exception as e:
            logger.error(f"Error saving trade analysis: {e}")
        finally:
            conn.close()
    
    def update_feature_importance(self):
        """
        Analyze all trades and update feature importance scores.
        This identifies which features correlate with winning trades.
        """
        conn = sqlite3.connect(self.trades_db)
        query = """
            SELECT * FROM trades 
            WHERE pnl_dollar IS NOT NULL
            ORDER BY exit_time
        """
        trades_df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(trades_df) < self.min_trades:
            logger.info(f"Not enough trades ({len(trades_df)}) for feature analysis. Need {self.min_trades}.")
            return {}
        
        # Features to analyze
        features = ['entry_rsi', 'entry_macd', 'entry_bb_position', 'entry_atr_pct', 
                   'entry_probability', 'entry_hour', 'entry_volatility']
        
        results = {}
        
        for bot_type in trades_df['bot_type'].unique():
            for direction in trades_df['direction'].unique():
                subset = trades_df[(trades_df['bot_type'] == bot_type) & 
                                  (trades_df['direction'] == direction)]
                
                if len(subset) < 10:
                    continue
                
                subset['is_winner'] = (subset['pnl_dollar'] > 0).astype(int)
                
                for feature in features:
                    if feature not in subset.columns or subset[feature].isna().all():
                        continue
                    
                    # Calculate correlation with winning
                    valid_data = subset[[feature, 'is_winner']].dropna()
                    if len(valid_data) < 10:
                        continue
                    
                    correlation = valid_data[feature].corr(valid_data['is_winner'])
                    
                    # Find optimal range (where win rate is highest)
                    winners = valid_data[valid_data['is_winner'] == 1][feature]
                    losers = valid_data[valid_data['is_winner'] == 0][feature]
                    
                    if len(winners) > 0:
                        optimal_min = winners.quantile(0.25)
                        optimal_max = winners.quantile(0.75)
                    else:
                        optimal_min = valid_data[feature].quantile(0.25)
                        optimal_max = valid_data[feature].quantile(0.75)
                    
                    # Calculate win rate in optimal range
                    in_range = valid_data[(valid_data[feature] >= optimal_min) & 
                                         (valid_data[feature] <= optimal_max)]
                    optimal_range_trades = len(in_range)
                    optimal_range_wins = in_range['is_winner'].sum()
                    
                    # Save to database
                    self._save_feature_importance(
                        feature_name=feature,
                        bot_type=bot_type,
                        direction=direction,
                        win_correlation=correlation,
                        optimal_min=optimal_min,
                        optimal_max=optimal_max,
                        total_trades=len(valid_data),
                        winning_trades=valid_data['is_winner'].sum(),
                        optimal_range_trades=optimal_range_trades,
                        optimal_range_wins=optimal_range_wins
                    )
                    
                    key = f"{bot_type}_{direction}_{feature}"
                    results[key] = {
                        'correlation': correlation,
                        'optimal_range': (optimal_min, optimal_max),
                        'win_rate_in_range': optimal_range_wins / optimal_range_trades if optimal_range_trades > 0 else 0,
                        'total_trades': len(valid_data)
                    }
        
        # Reload scores
        self._load_feature_scores()
        
        return results
    
    def _save_feature_importance(self, feature_name: str, bot_type: str, direction: str,
                                 win_correlation: float, optimal_min: float, optimal_max: float,
                                 total_trades: int, winning_trades: int,
                                 optimal_range_trades: int, optimal_range_wins: int):
        """Save feature importance to database."""
        conn = sqlite3.connect(self.feedback_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO feature_importance 
            (feature_name, bot_type, direction, win_correlation, optimal_min, optimal_max,
             total_trades, winning_trades, optimal_range_trades, optimal_range_wins, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (feature_name, bot_type, direction, win_correlation, optimal_min, optimal_max,
              total_trades, winning_trades, optimal_range_trades, optimal_range_wins,
              datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def calculate_signal_quality(self, bot_type: str, direction: str, 
                                 features: Dict[str, float]) -> Dict:
        """
        Calculate a quality score for a potential trade signal.
        
        Args:
            bot_type: 'BTC', 'MNQ', or 'SPY'
            direction: 'LONG' or 'SHORT'
            features: Dictionary of feature values at entry
            
        Returns:
            Dictionary with quality score and breakdown
        """
        if len(self.feature_scores) == 0:
            return {'score': 0.5, 'confidence': 'low', 'reason': 'No historical data'}
        
        score_components = []
        breakdown = {}
        
        feature_mapping = {
            'rsi': 'entry_rsi',
            'macd': 'entry_macd',
            'bb_pct_b': 'entry_bb_position',
            'atr': 'entry_atr_pct',
            'probability': 'entry_probability',
            'hour': 'entry_hour',
            'volatility': 'entry_volatility',
        }
        
        for input_name, db_name in feature_mapping.items():
            if input_name not in features:
                continue
            
            key = f"{bot_type}_{direction}_{db_name}"
            if key not in self.feature_scores:
                continue
            
            score_data = self.feature_scores[key]
            value = features[input_name]
            
            # Check if value is in optimal range
            if score_data['optimal_min'] is not None and score_data['optimal_max'] is not None:
                in_range = score_data['optimal_min'] <= value <= score_data['optimal_max']
                
                # Calculate component score
                if in_range:
                    # Win rate in optimal range
                    if score_data['optimal_range_trades'] > 0:
                        component_score = score_data['optimal_range_wins'] / score_data['optimal_range_trades']
                    else:
                        component_score = 0.5
                else:
                    # Outside optimal range - use overall win rate minus penalty
                    if score_data['total_trades'] > 0:
                        overall_wr = score_data['winning_trades'] / score_data['total_trades']
                        component_score = overall_wr * 0.8  # 20% penalty for being outside range
                    else:
                        component_score = 0.4
                
                # Weight by correlation strength
                weight = abs(score_data['win_correlation']) if score_data['win_correlation'] else 0.1
                
                score_components.append((component_score, weight))
                breakdown[input_name] = {
                    'value': value,
                    'optimal_range': (score_data['optimal_min'], score_data['optimal_max']),
                    'in_range': in_range,
                    'component_score': component_score,
                    'correlation': score_data['win_correlation']
                }
        
        # Calculate weighted average score
        if len(score_components) > 0:
            total_weight = sum(w for _, w in score_components)
            if total_weight > 0:
                final_score = sum(s * w for s, w in score_components) / total_weight
            else:
                final_score = 0.5
        else:
            final_score = 0.5
        
        # Determine confidence level
        if len(score_components) >= 4:
            confidence = 'high'
        elif len(score_components) >= 2:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        # Generate recommendation
        if final_score >= 0.6:
            recommendation = 'TAKE'
        elif final_score >= 0.4:
            recommendation = 'NEUTRAL'
        else:
            recommendation = 'SKIP'
        
        return {
            'score': round(final_score, 3),
            'confidence': confidence,
            'recommendation': recommendation,
            'breakdown': breakdown,
            'features_analyzed': len(score_components)
        }
    
    def identify_patterns(self) -> List[Dict]:
        """
        Identify winning and losing patterns from historical trades.
        
        Returns:
            List of identified patterns with their win rates
        """
        conn = sqlite3.connect(self.trades_db)
        query = """
            SELECT * FROM trades 
            WHERE pnl_dollar IS NOT NULL
            AND entry_rsi IS NOT NULL
        """
        trades_df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(trades_df) < self.min_trades:
            return []
        
        trades_df['is_winner'] = (trades_df['pnl_dollar'] > 0).astype(int)
        
        patterns = []
        
        # Pattern 1: RSI extremes
        for direction in ['LONG', 'SHORT']:
            subset = trades_df[trades_df['direction'] == direction]
            if len(subset) < 10:
                continue
            
            # Low RSI for LONG
            if direction == 'LONG':
                low_rsi = subset[subset['entry_rsi'] < 40]
                if len(low_rsi) >= 5:
                    wr = low_rsi['is_winner'].mean()
                    patterns.append({
                        'name': 'Low RSI LONG',
                        'condition': 'RSI < 40',
                        'direction': direction,
                        'trades': len(low_rsi),
                        'win_rate': wr,
                        'avg_pnl': low_rsi['pnl_pct'].mean()
                    })
            
            # High RSI for SHORT
            if direction == 'SHORT':
                high_rsi = subset[subset['entry_rsi'] > 60]
                if len(high_rsi) >= 5:
                    wr = high_rsi['is_winner'].mean()
                    patterns.append({
                        'name': 'High RSI SHORT',
                        'condition': 'RSI > 60',
                        'direction': direction,
                        'trades': len(high_rsi),
                        'win_rate': wr,
                        'avg_pnl': high_rsi['pnl_pct'].mean()
                    })
        
        # Pattern 2: High probability signals
        high_prob = trades_df[trades_df['entry_probability'] > 0.6]
        if len(high_prob) >= 5:
            wr = high_prob['is_winner'].mean()
            patterns.append({
                'name': 'High Probability',
                'condition': 'Probability > 60%',
                'direction': 'ANY',
                'trades': len(high_prob),
                'win_rate': wr,
                'avg_pnl': high_prob['pnl_pct'].mean()
            })
        
        # Pattern 3: Time of day
        for hour_range, name in [((9, 12), 'Morning'), ((12, 15), 'Midday'), ((15, 17), 'Afternoon')]:
            hour_trades = trades_df[(trades_df['entry_hour'] >= hour_range[0]) & 
                                   (trades_df['entry_hour'] < hour_range[1])]
            if len(hour_trades) >= 5:
                wr = hour_trades['is_winner'].mean()
                patterns.append({
                    'name': f'{name} Trades',
                    'condition': f'Hour {hour_range[0]}-{hour_range[1]}',
                    'direction': 'ANY',
                    'trades': len(hour_trades),
                    'win_rate': wr,
                    'avg_pnl': hour_trades['pnl_pct'].mean()
                })
        
        # Sort by win rate
        patterns.sort(key=lambda x: x['win_rate'], reverse=True)
        
        return patterns
    
    def get_learning_summary(self) -> Dict:
        """
        Get a summary of what the system has learned.
        
        Returns:
            Dictionary with learning insights
        """
        conn = sqlite3.connect(self.trades_db)
        query = "SELECT * FROM trades WHERE pnl_dollar IS NOT NULL"
        trades_df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(trades_df) == 0:
            return {'status': 'No trades to analyze'}
        
        trades_df['is_winner'] = (trades_df['pnl_dollar'] > 0).astype(int)
        
        summary = {
            'total_trades_analyzed': len(trades_df),
            'overall_win_rate': trades_df['is_winner'].mean(),
            'total_pnl': trades_df['pnl_dollar'].sum(),
            
            'by_bot': {},
            'by_direction': {},
            'feature_insights': [],
            'patterns': self.identify_patterns()
        }
        
        # By bot
        for bot in trades_df['bot_type'].unique():
            bot_df = trades_df[trades_df['bot_type'] == bot]
            summary['by_bot'][bot] = {
                'trades': len(bot_df),
                'win_rate': bot_df['is_winner'].mean(),
                'pnl': bot_df['pnl_dollar'].sum()
            }
        
        # By direction
        for direction in trades_df['direction'].unique():
            dir_df = trades_df[trades_df['direction'] == direction]
            summary['by_direction'][direction] = {
                'trades': len(dir_df),
                'win_rate': dir_df['is_winner'].mean(),
                'pnl': dir_df['pnl_dollar'].sum()
            }
        
        # Feature insights
        for key, scores in self.feature_scores.items():
            if scores['total_trades'] >= 10:
                parts = key.split('_')
                feature = '_'.join(parts[2:])
                
                opt_trades = scores.get('optimal_range_trades', 0)
                opt_wins = scores.get('optimal_range_wins', 0)
                if isinstance(opt_trades, bytes):
                    opt_trades = int.from_bytes(opt_trades, 'little') if opt_trades else 0
                if isinstance(opt_wins, bytes):
                    opt_wins = int.from_bytes(opt_wins, 'little') if opt_wins else 0
                
                insight = {
                    'feature': feature,
                    'bot': parts[0],
                    'direction': parts[1],
                    'correlation': scores['win_correlation'],
                    'optimal_range': (scores['optimal_min'], scores['optimal_max']),
                    'win_rate_in_range': opt_wins / opt_trades if opt_trades > 0 else 0
                }
                summary['feature_insights'].append(insight)
        
        # Sort insights by correlation strength
        summary['feature_insights'].sort(key=lambda x: abs(x['correlation'] or 0), reverse=True)
        
        return summary


def run_analysis():
    """Run a full analysis and print results."""
    print('='*70)
    print('TRADE FEEDBACK LOOP - ANALYSIS')
    print('='*70)
    
    feedback = TradeFeedbackLoop()
    
    # Update feature importance
    print("\n## Updating Feature Importance ##")
    importance = feedback.update_feature_importance()
    
    if importance:
        print("\nFeature correlations with winning trades:")
        for key, data in sorted(importance.items(), key=lambda x: abs(x[1]['correlation']), reverse=True):
            print(f"  {key}: correlation={data['correlation']:.3f}, "
                  f"win_rate_in_range={data['win_rate_in_range']:.1%}")
    
    # Get learning summary
    print("\n## Learning Summary ##")
    summary = feedback.get_learning_summary()
    
    print(f"\nTotal trades analyzed: {summary['total_trades_analyzed']}")
    print(f"Overall win rate: {summary['overall_win_rate']:.1%}")
    print(f"Total P&L: ${summary['total_pnl']:+.2f}")
    
    print("\nBy Bot:")
    for bot, data in summary['by_bot'].items():
        print(f"  {bot}: {data['trades']} trades, {data['win_rate']:.1%} WR, ${data['pnl']:+.2f}")
    
    print("\nBy Direction:")
    for direction, data in summary['by_direction'].items():
        print(f"  {direction}: {data['trades']} trades, {data['win_rate']:.1%} WR, ${data['pnl']:+.2f}")
    
    print("\nTop Patterns Identified:")
    for pattern in summary['patterns'][:5]:
        print(f"  {pattern['name']}: {pattern['win_rate']:.1%} WR ({pattern['trades']} trades)")
    
    print("\nTop Feature Insights:")
    for insight in summary['feature_insights'][:5]:
        print(f"  {insight['feature']} ({insight['bot']} {insight['direction']}): "
              f"corr={insight['correlation']:.3f}, range_WR={insight['win_rate_in_range']:.1%}")
    
    print('\n' + '='*70)
    
    return feedback


if __name__ == '__main__':
    run_analysis()
