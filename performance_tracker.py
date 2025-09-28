import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import sqlite3
import datetime
from dataclasses import dataclass
import json
import time
from collections import defaultdict, deque

@dataclass
class PerformanceMetrics:
    """کلاس نگهداری معیارهای عملکرد"""
    timestamp: datetime.datetime
    symbol: str
    total_return: float
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    avg_trade_duration: float
    avg_profit_per_trade: float

class PerformanceTracker:
    """
    ردیابی و محاسبه معیارهای عملکرد real-time
    """
    
    def __init__(self, db_path: str = "monitoring.db"):
        self.db_path = db_path
        self._init_database()
        
        # Cache برای محاسبات سریع
        self.metrics_cache = {}
        self.recent_trades = defaultdict(lambda: deque(maxlen=100))
        self.performance_history = defaultdict(list)
        
        # تنظیمات محاسبات
        self.calculation_windows = {
            'short_term': 24,    # 24 ساعت
            'medium_term': 168,  # 1 هفته  
            'long_term': 720     # 30 روز
        }
        
    def _init_database(self):
        """راه‌اندازی پایگاه داده"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # جدول معیارهای عملکرد
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                total_return REAL,
                win_rate REAL,
                profit_factor REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                total_trades INTEGER,
                avg_trade_duration REAL,
                avg_profit_per_trade REAL,
                volatility REAL,
                calmar_ratio REAL,
                sortino_ratio REAL
            )
        ''')
        
        # جدول معاملات برای محاسبات
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trade_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT NOT NULL,
                entry_time DATETIME,
                exit_time DATETIME,
                entry_price REAL,
                exit_price REAL,
                quantity REAL,
                side TEXT,
                pnl REAL,
                pnl_percentage REAL,
                duration_hours REAL,
                fees REAL,
                slippage REAL,
                signal_quality TEXT,
                market_conditions TEXT
            )
        ''')
        
        # ایندکس‌ها برای سرعت
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_perf_symbol_time ON performance_metrics(symbol, timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trade_records(symbol, timestamp)')
        
        conn.commit()
        conn.close()
        
    def record_trade(self, trade_data: Dict) -> bool:
        """ثبت معامله جدید"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # محاسبه مدت معامله
            if trade_data.get('entry_time') and trade_data.get('exit_time'):
                entry_time = pd.to_datetime(trade_data['entry_time'])
                exit_time = pd.to_datetime(trade_data['exit_time'])
                duration_hours = (exit_time - entry_time).total_seconds() / 3600
            else:
                duration_hours = 0
            
            # محاسبه PnL درصدی
            entry_price = trade_data.get('entry_price', 0)
            exit_price = trade_data.get('exit_price', 0)
            
            if entry_price > 0:
                if trade_data.get('side') == 'LONG':
                    pnl_percentage = (exit_price - entry_price) / entry_price * 100
                else:
                    pnl_percentage = (entry_price - exit_price) / entry_price * 100
            else:
                pnl_percentage = 0
            
            cursor.execute('''
                INSERT INTO trade_records (
                    symbol, entry_time, exit_time, entry_price, exit_price,
                    quantity, side, pnl, pnl_percentage, duration_hours,
                    fees, slippage, signal_quality, market_conditions
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_data.get('symbol'),
                trade_data.get('entry_time'),
                trade_data.get('exit_time'),
                entry_price,
                exit_price,
                trade_data.get('quantity', 0),
                trade_data.get('side'),
                trade_data.get('pnl', 0),
                pnl_percentage,
                duration_hours,
                trade_data.get('fees', 0),
                trade_data.get('slippage', 0),
                trade_data.get('signal_quality'),
                trade_data.get('market_conditions')
            ))
            
            conn.commit()
            conn.close()
            
            # به‌روزرسانی cache
            symbol = trade_data.get('symbol')
            self.recent_trades[symbol].append(trade_data)
            
            # محاسبه مجدد معیارها
            self._update_performance_metrics(symbol)
            
            return True
            
        except Exception as e:
            print(f"خطا در ثبت معامله: {e}")
            return False
    
    def calculate_current_metrics(self, symbol: str, timeframe: str = 'all') -> Optional[PerformanceMetrics]:
        """محاسبه معیارهای فعلی"""
        try:
            # دریافت معاملات
            trades_df = self._get_trades_dataframe(symbol, timeframe)
            
            if trades_df.empty:
                return None
            
            # محاسبه معیارهای اصلی
            metrics = self._calculate_metrics_from_trades(trades_df, symbol)
            
            return metrics
            
        except Exception as e:
            print(f"خطا در محاسبه معیارها برای {symbol}: {e}")
            return None
    
    def _get_trades_dataframe(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """دریافت معاملات از پایگاه داده"""
        conn = sqlite3.connect(self.db_path)
        
        # تعیین بازه زمانی
        if timeframe == 'short_term':
            hours_back = self.calculation_windows['short_term']
        elif timeframe == 'medium_term':
            hours_back = self.calculation_windows['medium_term']
        elif timeframe == 'long_term':
            hours_back = self.calculation_windows['long_term']
        else:
            hours_back = None
        
        if hours_back:
            cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=hours_back)
            query = '''
                SELECT * FROM trade_records 
                WHERE symbol = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            '''
            df = pd.read_sql_query(query, conn, params=(symbol, cutoff_time))
        else:
            query = '''
                SELECT * FROM trade_records 
                WHERE symbol = ?
                ORDER BY timestamp DESC
            '''
            df = pd.read_sql_query(query, conn, params=(symbol,))
        
        conn.close()
        return df
    
    def _calculate_metrics_from_trades(self, trades_df: pd.DataFrame, symbol: str) -> PerformanceMetrics:
        """محاسبه معیارها از DataFrame معاملات"""
        
        if trades_df.empty:
            return PerformanceMetrics(
                timestamp=datetime.datetime.now(),
                symbol=symbol,
                total_return=0,
                win_rate=0,
                profit_factor=0,
                sharpe_ratio=0,
                max_drawdown=0,
                total_trades=0,
                avg_trade_duration=0,
                avg_profit_per_trade=0
            )
        
        # محاسبات اصلی
        total_trades = len(trades_df)
        
        # PnL calculations
        total_pnl = trades_df['pnl'].sum()
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        # Win rate
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        
        # Profit factor
        gross_profit = winning_trades['pnl'].sum() if not winning_trades.empty else 0
        gross_loss = abs(losing_trades['pnl'].sum()) if not losing_trades.empty else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Returns برای Sharpe ratio
        returns = trades_df['pnl_percentage'].values
        
        # Sharpe ratio (فرض: risk-free rate = 0)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns)
        else:
            sharpe_ratio = 0
        
        # Max drawdown
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        # میانگین مدت معامله
        avg_trade_duration = trades_df['duration_hours'].mean() if not trades_df.empty else 0
        
        # میانگین سود هر معامله
        avg_profit_per_trade = total_pnl / total_trades if total_trades > 0 else 0
        
        # Total return (cumulative)
        total_return = np.sum(returns)
        
        return PerformanceMetrics(
            timestamp=datetime.datetime.now(),
            symbol=symbol,
            total_return=total_return,
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            total_trades=total_trades,
            avg_trade_duration=avg_trade_duration,
            avg_profit_per_trade=avg_profit_per_trade
        )
    
    def _update_performance_metrics(self, symbol: str):
        """به‌روزرسانی معیارهای عملکرد در پایگاه داده"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # محاسبه برای هر timeframe
            for timeframe in ['short_term', 'medium_term', 'long_term', 'all']:
                metrics = self.calculate_current_metrics(symbol, timeframe)
                
                if metrics:
                    # محاسبه معیارهای اضافی
                    trades_df = self._get_trades_dataframe(symbol, timeframe)
                    
                    volatility = trades_df['pnl_percentage'].std() if not trades_df.empty else 0
                    
                    # Calmar ratio
                    calmar_ratio = abs(metrics.total_return / metrics.max_drawdown) if metrics.max_drawdown != 0 else 0
                    
                    # Sortino ratio
                    negative_returns = trades_df[trades_df['pnl_percentage'] < 0]['pnl_percentage']
                    downside_deviation = negative_returns.std() if not negative_returns.empty else 0
                    sortino_ratio = metrics.total_return / downside_deviation if downside_deviation > 0 else 0
                    
                    cursor.execute('''
                        INSERT INTO performance_metrics (
                            symbol, timeframe, total_return, win_rate, profit_factor,
                            sharpe_ratio, max_drawdown, total_trades, avg_trade_duration,
                            avg_profit_per_trade, volatility, calmar_ratio, sortino_ratio
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol, timeframe, metrics.total_return, metrics.win_rate,
                        metrics.profit_factor, metrics.sharpe_ratio, metrics.max_drawdown,
                        metrics.total_trades, metrics.avg_trade_duration,
                        metrics.avg_profit_per_trade, volatility, calmar_ratio, sortino_ratio
                    ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"خطا در به‌روزرسانی معیارها: {e}")
    
    def get_performance_summary(self, symbols: List[str] = None, timeframe: str = 'all') -> Dict:
        """دریافت خلاصه عملکرد"""
        try:
            if symbols is None:
                symbols = self.get_tracked_symbols()
            
            summary = {
                'timestamp': datetime.datetime.now(),
                'timeframe': timeframe,
                'symbols': {},
                'portfolio_metrics': {}
            }
            
            all_metrics = []
            
            for symbol in symbols:
                metrics = self.calculate_current_metrics(symbol, timeframe)
                if metrics:
                    summary['symbols'][symbol] = {
                        'total_return': metrics.total_return,
                        'win_rate': metrics.win_rate,
                        'profit_factor': metrics.profit_factor,
                        'sharpe_ratio': metrics.sharpe_ratio,
                        'max_drawdown': metrics.max_drawdown,
                        'total_trades': metrics.total_trades,
                        'avg_profit_per_trade': metrics.avg_profit_per_trade
                    }
                    all_metrics.append(metrics)
            
            # محاسبه معیارهای پورتفولیو
            if all_metrics:
                portfolio_return = sum(m.total_return for m in all_metrics)
                portfolio_trades = sum(m.total_trades for m in all_metrics)
                avg_win_rate = np.mean([m.win_rate for m in all_metrics])
                avg_sharpe = np.mean([m.sharpe_ratio for m in all_metrics])
                worst_drawdown = min(m.max_drawdown for m in all_metrics)
                
                summary['portfolio_metrics'] = {
                    'total_return': portfolio_return,
                    'total_trades': portfolio_trades,
                    'avg_win_rate': avg_win_rate,
                    'avg_sharpe_ratio': avg_sharpe,
                    'worst_drawdown': worst_drawdown,
                    'active_symbols': len([m for m in all_metrics if m.total_trades > 0])
                }
            
            return summary
            
        except Exception as e:
            print(f"خطا در تهیه خلاصه عملکرد: {e}")
            return {}
    
    def get_tracked_symbols(self) -> List[str]:
        """دریافت لیست نمادهای ردیابی شده"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT DISTINCT symbol FROM trade_records')
            symbols = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            return symbols
            
        except Exception as e:
            print(f"خطا در دریافت نمادها: {e}")
            return []
    
    def get_performance_trend(self, symbol: str, days: int = 30) -> Dict:
        """دریافت روند عملکرد"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
            
            query = '''
                SELECT DATE(timestamp) as date, 
                       AVG(total_return) as avg_return,
                       AVG(win_rate) as avg_win_rate,
                       AVG(sharpe_ratio) as avg_sharpe,
                       COUNT(*) as records
                FROM performance_metrics 
                WHERE symbol = ? AND timestamp >= ? AND timeframe = 'all'
                GROUP BY DATE(timestamp)
                ORDER BY date
            '''
            
            df = pd.read_sql_query(query, conn, params=(symbol, cutoff_date))
            conn.close()
            
            return {
                'symbol': symbol,
                'period_days': days,
                'trend_data': df.to_dict('records'),
                'trend_analysis': self._analyze_trend(df)
            }
            
        except Exception as e:
            print(f"خطا در دریافت روند عملکرد: {e}")
            return {}
    
    def _analyze_trend(self, trend_df: pd.DataFrame) -> Dict:
        """تحلیل روند عملکرد"""
        if trend_df.empty:
            return {}
        
        # محاسبه slope برای هر معیار
        returns_trend = np.polyfit(range(len(trend_df)), trend_df['avg_return'], 1)[0]
        winrate_trend = np.polyfit(range(len(trend_df)), trend_df['avg_win_rate'], 1)[0]
        sharpe_trend = np.polyfit(range(len(trend_df)), trend_df['avg_sharpe'], 1)[0]
        
        return {
            'returns_trend': 'IMPROVING' if returns_trend > 0 else 'DECLINING',
            'winrate_trend': 'IMPROVING' if winrate_trend > 0 else 'DECLINING',
            'sharpe_trend': 'IMPROVING' if sharpe_trend > 0 else 'DECLINING',
            'overall_trend': 'POSITIVE' if (returns_trend + winrate_trend + sharpe_trend) > 0 else 'NEGATIVE',
            'trend_strength': abs(returns_trend) + abs(winrate_trend) + abs(sharpe_trend)
        }
    
    def export_performance_data(self, symbol: str = None, format: str = 'json') -> str:
        """صادرات داده‌های عملکرد"""
        try:
            if symbol:
                data = self.get_performance_summary([symbol])
            else:
                data = self.get_performance_summary()
            
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if format == 'json':
                filename = f"performance_export_{timestamp}.json"
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            
            elif format == 'csv':
                filename = f"performance_export_{timestamp}.csv"
                # تبدیل به DataFrame و ذخیره
                flat_data = []
                for sym, metrics in data.get('symbols', {}).items():
                    metrics['symbol'] = sym
                    flat_data.append(metrics)
                
                df = pd.DataFrame(flat_data)
                df.to_csv(filename, index=False)
            
            print(f"داده‌های عملکرد در {filename} صادر شد")
            return filename
            
        except Exception as e:
            print(f"خطا در صادرات داده‌ها: {e}")
            return ""
    
    def clear_old_data(self, days_to_keep: int = 90):
        """پاک‌سازی داده‌های قدیمی"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_to_keep)
            
            # پاک‌سازی معیارهای قدیمی
            cursor.execute('DELETE FROM performance_metrics WHERE timestamp < ?', (cutoff_date,))
            
            # پاک‌سازی معاملات قدیمی
            cursor.execute('DELETE FROM trade_records WHERE timestamp < ?', (cutoff_date,))
            
            deleted_metrics = cursor.rowcount
            conn.commit()
            conn.close()
            
            print(f"داده‌های قدیمی‌تر از {days_to_keep} روز پاک شد ({deleted_metrics} رکورد)")
            
        except Exception as e:
            print(f"خطا در پاک‌سازی داده‌ها: {e}")
