"""
connectors/database_connector.py
کانکتور پایگاه داده با قابلیت‌های پیشرفته
"""

import sqlite3
import json
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import threading
import os

class DatabaseConnector:
    """کانکتور پایگاه داده SQLite با thread safety"""
    
    def __init__(self, db_path: str = "multi_crypto_trading.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._create_tables()
        
        # آمار
        self.stats = {
            'queries_executed': 0,
            'errors_count': 0,
            'last_backup': None
        }
    
    def _create_tables(self):
        """ایجاد جدول‌های مورد نیاز"""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # جدول سیگنال‌ها
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        symbol TEXT NOT NULL,
                        signal_type TEXT,
                        position TEXT,
                        confidence REAL,
                        entry_price REAL,
                        stop_loss REAL,
                        take_profit REAL,
                        leverage INTEGER,
                        reasoning TEXT,
                        ai_evaluation TEXT,
                        status TEXT DEFAULT 'ACTIVE',
                        result TEXT,
                        pnl REAL,
                        closed_at DATETIME
                    )
                ''')
                
                # جدول عملکرد
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date DATE NOT NULL,
                        symbol TEXT NOT NULL,
                        total_signals INTEGER DEFAULT 0,
                        successful_signals INTEGER DEFAULT 0,
                        failed_signals INTEGER DEFAULT 0,
                        win_rate REAL DEFAULT 0.0,
                        total_pnl REAL DEFAULT 0.0,
                        best_trade REAL DEFAULT 0.0,
                        worst_trade REAL DEFAULT 0.0,
                        UNIQUE(date, symbol)
                    )
                ''')
                
                # جدول تحلیل بازار
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS market_analysis (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        symbol TEXT NOT NULL,
                        price REAL,
                        volume REAL,
                        market_structure TEXT,
                        trend_strength TEXT,
                        volatility REAL,
                        support_level REAL,
                        resistance_level REAL,
                        analysis_data TEXT
                    )
                ''')
                
                # جدول آمار AI
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS ai_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        ai_model TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        prediction_quality REAL,
                        response_time REAL,
                        accuracy REAL,
                        confidence REAL
                    )
                ''')
                
                # ایندکس‌ها برای بهبود عملکرد
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_symbol_timestamp ON signals(symbol, timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_performance_date_symbol ON performance(date, symbol)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_market_analysis_symbol_timestamp ON market_analysis(symbol, timestamp)')
                
                conn.commit()
                conn.close()
                
            except Exception as e:
                print(f"❌ خطا در ایجاد جدول‌ها: {e}")
    
    def test_connection(self) -> bool:
        """تست اتصال پایگاه داده"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                conn.close()
                return result is not None
        except Exception as e:
            print(f"❌ Database test failed: {e}")
            return False
    
    def save_signal(self, signal_data: Dict) -> Optional[int]:
        """ذخیره سیگنال جدید"""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO signals 
                    (symbol, signal_type, position, confidence, entry_price, 
                     stop_loss, take_profit, leverage, reasoning, ai_evaluation)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    signal_data.get('symbol'),
                    signal_data.get('signal_type'),
                    signal_data.get('position'),
                    signal_data.get('confidence'),
                    signal_data.get('entry_price'),
                    signal_data.get('stop_loss'),
                    signal_data.get('take_profits', [0])[0] if signal_data.get('take_profits') else 0,
                    int(signal_data.get('leverage', '1X').replace('X', '')),
                    json.dumps(signal_data.get('reasoning', [])),
                    json.dumps(signal_data.get('ai_evaluation', {}))
                ))
                
                signal_id = cursor.lastrowid
                conn.commit()
                conn.close()
                
                self.stats['queries_executed'] += 1
                return signal_id
                
            except Exception as e:
                print(f"❌ خطا در ذخیره سیگنال: {e}")
                self.stats['errors_count'] += 1
                return None
    
    def update_signal_result(self, signal_id: int, result: str, pnl: float) -> bool:
        """به‌روزرسانی نتیجه سیگنال"""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE signals 
                    SET status = 'CLOSED', result = ?, pnl = ?, closed_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (result, pnl, signal_id))
                
                conn.commit()
                conn.close()
                
                self.stats['queries_executed'] += 1
                return True
                
            except Exception as e:
                print(f"❌ خطا در به‌روزرسانی سیگنال: {e}")
                self.stats['errors_count'] += 1
                return False
    
    def get_performance_summary(self, days: int = 30) -> Dict[str, Any]:
        """خلاصه عملکرد در بازه زمانی"""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # محاسبه تاریخ شروع
                start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
                
                # آمار کلی
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total_signals,
                        SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END) as losses,
                        AVG(pnl) as avg_pnl,
                        SUM(pnl) as total_pnl,
                        MAX(pnl) as best_trade,
                        MIN(pnl) as worst_trade
                    FROM signals 
                    WHERE timestamp >= ? AND status = 'CLOSED'
                ''', (start_date,))
                
                summary = cursor.fetchone()
                
                # آمار به تفکیک نماد
                cursor.execute('''
                    SELECT 
                        symbol,
                        COUNT(*) as signals,
                        SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                        AVG(pnl) as avg_pnl
                    FROM signals 
                    WHERE timestamp >= ? AND status = 'CLOSED'
                    GROUP BY symbol
                ''', (start_date,))
                
                symbol_stats = cursor.fetchall()
                
                conn.close()
                
                return {
                    'period_days': days,
                    'total_signals': summary[0] or 0,
                    'wins': summary[1] or 0,
                    'losses': summary[2] or 0,
                    'win_rate': (summary[1] / summary[0] * 100) if summary[0] > 0 else 0,
                    'avg_pnl': summary[3] or 0,
                    'total_pnl': summary[4] or 0,
                    'best_trade': summary[5] or 0,
                    'worst_trade': summary[6] or 0,
                    'symbol_breakdown': [
                        {
                            'symbol': row[0],
                            'signals': row[1],
                            'wins': row[2],
                            'win_rate': (row[2] / row[1] * 100) if row[1] > 0 else 0,
                            'avg_pnl': row[3] or 0
                        }
                        for row in symbol_stats
                    ]
                }
                
            except Exception as e:
                print(f"❌ خطا در دریافت خلاصه عملکرد: {e}")
                return {}
    
    def save_market_analysis(self, analysis_data: Dict) -> bool:
        """ذخیره تحلیل بازار"""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO market_analysis 
                    (symbol, price, volume, market_structure, trend_strength, 
                     volatility, support_level, resistance_level, analysis_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    analysis_data.get('symbol'),
                    analysis_data.get('current_price'),
                    analysis_data.get('volume'),
                    analysis_data.get('market_context', {}).get('structure'),
                    analysis_data.get('market_context', {}).get('trend_strength'),
                    analysis_data.get('volatility'),
                    analysis_data.get('support_level'),
                    analysis_data.get('resistance_level'),
                    json.dumps(analysis_data)
                ))
                
                conn.commit()
                conn.close()
                
                self.stats['queries_executed'] += 1
                return True
                
            except Exception as e:
                print(f"❌ خطا در ذخیره تحلیل بازار: {e}")
                self.stats['errors_count'] += 1
                return False
    
    def backup_database(self, backup_path: Optional[str] = None) -> bool:
        """پشتیبان‌گیری از پایگاه داده"""
        if not backup_path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = f"backup_db_{timestamp}.db"
        
        try:
            import shutil
            shutil.copy2(self.db_path, backup_path)
            self.stats['last_backup'] = datetime.now()
            print(f"✅ پشتیبان‌گیری موفق: {backup_path}")
            return True
        except Exception as e:
            print(f"❌ خطا در پشتیبان‌گیری: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """آمار کانکتور"""
        return {
            'database_path': self.db_path,
            'database_size_mb': os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0,
            'queries_executed': self.stats['queries_executed'],
            'errors_count': self.stats['errors_count'],
            'success_rate': (
                (self.stats['queries_executed'] - self.stats['errors_count']) / 
                self.stats['queries_executed'] * 100
                if self.stats['queries_executed'] > 0 else 100
            ),
            'last_backup': self.stats['last_backup']
        }
