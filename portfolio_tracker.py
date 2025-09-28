import sqlite3
import pandas as pd
import numpy as np
import datetime
import json
from typing import Dict, List, Optional, Tuple
from enum import Enum
import config

class PositionStatus(Enum):
    """وضعیت پوزیشن‌ها"""
    ACTIVE = "ACTIVE"
    CLOSED = "CLOSED"
    PENDING = "PENDING"
    CANCELLED = "CANCELLED"

class PortfolioMetrics(Enum):
    """معیارهای عملکرد پورتفولیو"""
    TOTAL_PNL = "TOTAL_PNL"
    WIN_RATE = "WIN_RATE"
    SHARPE_RATIO = "SHARPE_RATIO"
    MAX_DRAWDOWN = "MAX_DRAWDOWN"
    PROFIT_FACTOR = "PROFIT_FACTOR"
    AVERAGE_WIN = "AVERAGE_WIN"
    AVERAGE_LOSS = "AVERAGE_LOSS"

class PortfolioTracker:
    """
    کلاس ردیابی و مدیریت پورتفولیو معاملاتی
    """

    def __init__(self, db_path: str = "portfolio_tracking.db"):
        self.db_path = db_path
        self.name = "Portfolio Tracker"
        
        # وضعیت فعلی پورتفولیو
        self.current_portfolio = {
            'total_balance': 10000.0,  # مبلغ اولیه
            'available_balance': 10000.0,
            'used_margin': 0.0,
            'unrealized_pnl': 0.0,
            'realized_pnl': 0.0,
            'total_equity': 10000.0,
            'margin_level': 0.0,
            'active_positions_count': 0,
            'daily_pnl': 0.0
        }
        
        # پوزیشن‌های فعال
        self.active_positions = {}
        
        # آمار عملکرد
        self.performance_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0,
            'best_trade': 0.0,
            'worst_trade': 0.0,
            'average_win': 0.0,
            'average_loss': 0.0,
            'consecutive_wins': 0,
            'consecutive_losses': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0
        }
        
        # آمار روزانه
        self.daily_stats = {}
        
        self._init_database()
        self._load_portfolio_state()

    def _init_database(self):
        """ایجاد جداول پایگاه داده"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # جدول پوزیشن‌ها
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    position_id TEXT UNIQUE NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_time DATETIME NOT NULL,
                    entry_price REAL NOT NULL,
                    position_size REAL NOT NULL,
                    leverage REAL NOT NULL,
                    stop_loss REAL,
                    take_profit REAL,
                    exit_time DATETIME,
                    exit_price REAL,
                    pnl_amount REAL DEFAULT 0,
                    pnl_percentage REAL DEFAULT 0,
                    fees REAL DEFAULT 0,
                    status TEXT DEFAULT 'ACTIVE',
                    signal_quality TEXT,
                    signal_type TEXT,
                    confidence REAL,
                    market_context TEXT,
                    notes TEXT
                )
            ''')

            # جدول تاریخچه پورتفولیو
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    total_balance REAL,
                    available_balance REAL,
                    used_margin REAL,
                    unrealized_pnl REAL,
                    realized_pnl REAL,
                    total_equity REAL,
                    margin_level REAL,
                    active_positions INTEGER,
                    daily_pnl REAL,
                    drawdown REAL
                )
            ''')

            # جدول معاملات تکمیل شده
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS completed_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT UNIQUE NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_time DATETIME NOT NULL,
                    exit_time DATETIME NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    position_size REAL NOT NULL,
                    leverage REAL NOT NULL,
                    pnl_amount REAL NOT NULL,
                    pnl_percentage REAL NOT NULL,
                    fees REAL DEFAULT 0,
                    duration_minutes INTEGER,
                    signal_quality TEXT,
                    signal_type TEXT,
                    market_conditions TEXT,
                    success BOOLEAN
                )
            ''')

            # جدول آمار روزانه
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE UNIQUE NOT NULL,
                    starting_balance REAL,
                    ending_balance REAL,
                    daily_pnl REAL,
                    daily_return_pct REAL,
                    trades_count INTEGER,
                    winning_trades INTEGER,
                    losing_trades INTEGER,
                    largest_win REAL,
                    largest_loss REAL,
                    drawdown REAL,
                    volume_traded REAL
                )
            ''')

            # جدول حدود ریسک
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS risk_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    event_type TEXT NOT NULL,
                    description TEXT,
                    portfolio_impact REAL,
                    action_taken TEXT,
                    severity TEXT
                )
            ''')

            conn.commit()
            conn.close()
            print("✅ جداول ردیابی پورتفولیو ایجاد شدند")

        except Exception as e:
            print(f"خطا در ایجاد جداول: {e}")

    def _load_portfolio_state(self):
        """بارگیری وضعیت پورتفولیو از پایگاه داده"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # بارگیری پوزیشن‌های فعال
            positions_df = pd.read_sql_query(
                "SELECT * FROM positions WHERE status = 'ACTIVE'",
                conn
            )
            
            for _, position in positions_df.iterrows():
                self.active_positions[position['position_id']] = {
                    'symbol': position['symbol'],
                    'side': position['side'],
                    'entry_time': datetime.datetime.fromisoformat(position['entry_time']),
                    'entry_price': position['entry_price'],
                    'position_size': position['position_size'],
                    'leverage': position['leverage'],
                    'stop_loss': position['stop_loss'],
                    'take_profit': position['take_profit'],
                    'signal_quality': position['signal_quality'],
                    'signal_type': position['signal_type'],
                    'confidence': position['confidence'],
                    'current_pnl': 0.0,
                    'current_pnl_pct': 0.0
                }
            
            # بارگیری آخرین وضعیت پورتفولیو
            latest_portfolio = pd.read_sql_query(
                "SELECT * FROM portfolio_history ORDER BY timestamp DESC LIMIT 1",
                conn
            )
            
            if not latest_portfolio.empty:
                row = latest_portfolio.iloc[0]
                self.current_portfolio.update({
                    'total_balance': row['total_balance'],
                    'available_balance': row['available_balance'],
                    'used_margin': row['used_margin'],
                    'unrealized_pnl': row['unrealized_pnl'],
                    'realized_pnl': row['realized_pnl'],
                    'total_equity': row['total_equity'],
                    'margin_level': row['margin_level'],
                    'active_positions_count': row['active_positions'],
                    'daily_pnl': row['daily_pnl']
                })
            
            conn.close()
            self._calculate_performance_stats()
            print(f"✅ وضعیت پورتفولیو بارگیری شد - {len(self.active_positions)} پوزیشن فعال")

        except Exception as e:
            print(f"خطا در بارگیری وضعیت پورتفولیو: {e}")

    def add_position(self, signal_data: Dict, position_data: Dict) -> str:
        """اضافه کردن پوزیشن جدید"""
        try:
            position_id = f"{signal_data['symbol']}_{datetime.datetime.now().timestamp()}"
            
            # ذخیره در پایگاه داده
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO positions (
                    position_id, symbol, side, entry_time, entry_price,
                    position_size, leverage, stop_loss, take_profit,
                    signal_quality, signal_type, confidence, market_context
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                position_id,
                signal_data['symbol'],
                signal_data['position'],
                datetime.datetime.now(),
                signal_data['entry_price'],
                position_data['position_size_crypto'],
                position_data['leverage'],
                signal_data.get('stop_loss'),
                signal_data.get('take_profits', [None])[0] if signal_data.get('take_profits') else None,
                signal_data.get('signal_quality'),
                signal_data.get('signal_type'),
                signal_data.get('confidence'),
                json.dumps(signal_data.get('market_context', {}))
            ))
            
            conn.commit()
            conn.close()
            
            # اضافه به پوزیشن‌های فعال
            self.active_positions[position_id] = {
                'symbol': signal_data['symbol'],
                'side': signal_data['position'],
                'entry_time': datetime.datetime.now(),
                'entry_price': signal_data['entry_price'],
                'position_size': position_data['position_size_crypto'],
                'leverage': position_data['leverage'],
                'stop_loss': signal_data.get('stop_loss'),
                'take_profit': signal_data.get('take_profits', [None])[0] if signal_data.get('take_profits') else None,
                'signal_quality': signal_data.get('signal_quality'),
                'signal_type': signal_data.get('signal_type'),
                'confidence': signal_data.get('confidence'),
                'current_pnl': 0.0,
                'current_pnl_pct': 0.0
            }
            
            # به‌روزرسانی وضعیت پورتفولیو
            margin_used = position_data['position_size_usd']
            self.current_portfolio['used_margin'] += margin_used
            self.current_portfolio['available_balance'] -= margin_used
            self.current_portfolio['active_positions_count'] += 1
            
            self._save_portfolio_snapshot()
            
            print(f"✅ پوزیشن {position_id} اضافه شد")
            return position_id
            
        except Exception as e:
            print(f"خطا در اضافه کردن پوزیشن: {e}")
            return ""

    def update_position(self, position_id: str, current_price: float) -> Dict:
        """به‌روزرسانی پوزیشن با قیمت فعلی"""
        if position_id not in self.active_positions:
            return {'error': 'Position not found'}
        
        try:
            position = self.active_positions[position_id]
            entry_price = position['entry_price']
            position_size = position['position_size']
            leverage = position['leverage']
            side = position['side']
            
            # محاسبه PnL
            if side == 'LONG':
                price_change_pct = (current_price - entry_price) / entry_price
            else:  # SHORT
                price_change_pct = (entry_price - current_price) / entry_price
            
            pnl_pct = price_change_pct * leverage
            pnl_amount = (position_size * entry_price) * pnl_pct
            
            # به‌روزرسانی پوزیشن
            position['current_pnl'] = pnl_amount
            position['current_pnl_pct'] = pnl_pct * 100
            position['current_price'] = current_price
            position['last_update'] = datetime.datetime.now()
            
            # بررسی stop loss و take profit
            alerts = []
            if position['stop_loss']:
                if (side == 'LONG' and current_price <= position['stop_loss']) or \
                   (side == 'SHORT' and current_price >= position['stop_loss']):
                    alerts.append('STOP_LOSS_HIT')
            
            if position['take_profit']:
                if (side == 'LONG' and current_price >= position['take_profit']) or \
                   (side == 'SHORT' and current_price <= position['take_profit']):
                    alerts.append('TAKE_PROFIT_HIT')
            
            return {
                'position_id': position_id,
                'symbol': position['symbol'],
                'current_pnl': pnl_amount,
                'current_pnl_pct': pnl_pct * 100,
                'current_price': current_price,
                'alerts': alerts,
                'unrealized_pnl': pnl_amount
            }
            
        except Exception as e:
            print(f"خطا در به‌روزرسانی پوزیشن {position_id}: {e}")
            return {'error': str(e)}

    def close_position(self, position_id: str, exit_price: float, reason: str = "MANUAL") -> Dict:
        """بستن پوزیشن"""
        if position_id not in self.active_positions:
            return {'error': 'Position not found'}
        
        try:
            position = self.active_positions[position_id]
            
            # محاسبه PnL نهایی
            entry_price = position['entry_price']
            position_size = position['position_size']
            leverage = position['leverage']
            side = position['side']
            
            if side == 'LONG':
                price_change_pct = (exit_price - entry_price) / entry_price
            else:  # SHORT
                price_change_pct = (entry_price - exit_price) / entry_price
            
            pnl_pct = price_change_pct * leverage
            pnl_amount = (position_size * entry_price) * pnl_pct
            
            # محاسبه مدت زمان
            duration = datetime.datetime.now() - position['entry_time']
            duration_minutes = int(duration.total_seconds() / 60)
            
            # به‌روزرسانی پایگاه داده
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # به‌روزرسانی جدول positions
            cursor.execute('''
                UPDATE positions SET
                    exit_time = ?, exit_price = ?, pnl_amount = ?,
                    pnl_percentage = ?, status = 'CLOSED', notes = ?
                WHERE position_id = ?
            ''', (
                datetime.datetime.now(), exit_price, pnl_amount,
                pnl_pct * 100, reason, position_id
            ))
            
            # اضافه به جدول completed_trades
            cursor.execute('''
                INSERT INTO completed_trades (
                    trade_id, symbol, side, entry_time, exit_time,
                    entry_price, exit_price, position_size, leverage,
                    pnl_amount, pnl_percentage, duration_minutes,
                    signal_quality, signal_type, success
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                position_id, position['symbol'], position['side'],
                position['entry_time'], datetime.datetime.now(),
                entry_price, exit_price, position_size, leverage,
                pnl_amount, pnl_pct * 100, duration_minutes,
                position.get('signal_quality'), position.get('signal_type'),
                pnl_amount > 0
            ))
            
            conn.commit()
            conn.close()
            
            # به‌روزرسانی پورتفولیو
            margin_released = position_size * entry_price
            self.current_portfolio['used_margin'] -= margin_released
            self.current_portfolio['available_balance'] += margin_released + pnl_amount
            self.current_portfolio['realized_pnl'] += pnl_amount
            self.current_portfolio['total_balance'] += pnl_amount
            self.current_portfolio['total_equity'] += pnl_amount
            self.current_portfolio['active_positions_count'] -= 1
            self.current_portfolio['daily_pnl'] += pnl_amount
            
            # حذف از پوزیشن‌های فعال
            del self.active_positions[position_id]
            
            # به‌روزرسانی آمار عملکرد
            self._update_performance_stats(pnl_amount, pnl_pct * 100)
            self._save_portfolio_snapshot()
            
            result = {
                'position_id': position_id,
                'symbol': position['symbol'],
                'side': position['side'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_amount': pnl_amount,
                'pnl_percentage': pnl_pct * 100,
                'duration_minutes': duration_minutes,
                'reason': reason,
                'success': pnl_amount > 0
            }
            
            print(f"✅ پوزیشن {position_id} بسته شد - PnL: {pnl_amount:.2f} ({pnl_pct*100:.2f}%)")
            return result
            
        except Exception as e:
            print(f"خطا در بستن پوزیشن {position_id}: {e}")
            return {'error': str(e)}

    def update_all_positions(self, price_data: Dict):
        """به‌روزرسانی همه پوزیشن‌های فعال"""
        total_unrealized_pnl = 0
        
        for position_id, position in self.active_positions.items():
            symbol = position['symbol']
            if symbol in price_data:
                current_price = price_data[symbol]
                update_result = self.update_position(position_id, current_price)
                
                if 'unrealized_pnl' in update_result:
                    total_unrealized_pnl += update_result['unrealized_pnl']
                
                # بررسی هشدارها
                if 'alerts' in update_result and update_result['alerts']:
                    for alert in update_result['alerts']:
                        self._log_risk_event(
                            'POSITION_ALERT',
                            f"{alert} for {position_id}",
                            update_result.get('current_pnl', 0),
                            'MEDIUM'
                        )
        
        # به‌روزرسانی PnL غیرمحقق کل
        self.current_portfolio['unrealized_pnl'] = total_unrealized_pnl
        self.current_portfolio['total_equity'] = (
            self.current_portfolio['total_balance'] + total_unrealized_pnl
        )

    def _update_performance_stats(self, pnl_amount: float, pnl_percentage: float):
        """به‌روزرسانی آمار عملکرد"""
        self.performance_stats['total_trades'] += 1
        
        if pnl_amount > 0:
            self.performance_stats['winning_trades'] += 1
            self.performance_stats['consecutive_wins'] += 1
            self.performance_stats['consecutive_losses'] = 0
            
            if pnl_amount > self.performance_stats['best_trade']:
                self.performance_stats['best_trade'] = pnl_amount
            
            self.performance_stats['max_consecutive_wins'] = max(
                self.performance_stats['max_consecutive_wins'],
                self.performance_stats['consecutive_wins']
            )
            
        else:
            self.performance_stats['losing_trades'] += 1
            self.performance_stats['consecutive_losses'] += 1
            self.performance_stats['consecutive_wins'] = 0
            
            if pnl_amount < self.performance_stats['worst_trade']:
                self.performance_stats['worst_trade'] = pnl_amount
            
            self.performance_stats['max_consecutive_losses'] = max(
                self.performance_stats['max_consecutive_losses'],
                self.performance_stats['consecutive_losses']
            )
        
        # محاسبه نسبت برد
        if self.performance_stats['total_trades'] > 0:
            self.performance_stats['win_rate'] = (
                self.performance_stats['winning_trades'] / 
                self.performance_stats['total_trades']
            )
        
        # محاسبه میانگین سود/ضرر
        self._calculate_averages()

    def _calculate_averages(self):
        """محاسبه میانگین سود و ضرر"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # میانگین معاملات برنده
            winning_trades = pd.read_sql_query(
                "SELECT pnl_amount FROM completed_trades WHERE pnl_amount > 0",
                conn
            )
            
            if not winning_trades.empty:
                self.performance_stats['average_win'] = winning_trades['pnl_amount'].mean()
            
            # میانگین معاملات بازنده
            losing_trades = pd.read_sql_query(
                "SELECT pnl_amount FROM completed_trades WHERE pnl_amount < 0",
                conn
            )
            
            if not losing_trades.empty:
                self.performance_stats['average_loss'] = abs(losing_trades['pnl_amount'].mean())
            
            # محاسبه Profit Factor
            total_wins = winning_trades['pnl_amount'].sum() if not winning_trades.empty else 0
            total_losses = abs(losing_trades['pnl_amount'].sum()) if not losing_trades.empty else 1
            
            if total_losses > 0:
                self.performance_stats['profit_factor'] = total_wins / total_losses
            
            conn.close()
            
        except Exception as e:
            print(f"خطا در محاسبه میانگین‌ها: {e}")

    def _calculate_performance_stats(self):
        """محاسبه کامل آمار عملکرد"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # تعداد کل معاملات
            total_trades = pd.read_sql_query(
                "SELECT COUNT(*) as count FROM completed_trades",
                conn
            ).iloc[0]['count']
            
            self.performance_stats['total_trades'] = total_trades
            
            if total_trades > 0:
                # معاملات برنده و بازنده
                winning_trades = pd.read_sql_query(
                    "SELECT COUNT(*) as count FROM completed_trades WHERE pnl_amount > 0",
                    conn
                ).iloc[0]['count']
                
                self.performance_stats['winning_trades'] = winning_trades
                self.performance_stats['losing_trades'] = total_trades - winning_trades
                self.performance_stats['win_rate'] = winning_trades / total_trades
                
                # بهترین و بدترین معامله
                best_trade = pd.read_sql_query(
                    "SELECT MAX(pnl_amount) as max_pnl FROM completed_trades",
                    conn
                ).iloc[0]['max_pnl']
                
                worst_trade = pd.read_sql_query(
                    "SELECT MIN(pnl_amount) as min_pnl FROM completed_trades",
                    conn
                ).iloc[0]['min_pnl']
                
                self.performance_stats['best_trade'] = best_trade or 0
                self.performance_stats['worst_trade'] = worst_trade or 0
                
                self._calculate_averages()
                self._calculate_drawdown()
            
            conn.close()
            
        except Exception as e:
            print(f"خطا در محاسبه آمار عملکرد: {e}")

    def _calculate_drawdown(self):
        """محاسبه Maximum Drawdown"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # دریافت تاریخچه equity
            portfolio_history = pd.read_sql_query(
                "SELECT timestamp, total_equity FROM portfolio_history ORDER BY timestamp",
                conn
            )
            
            if len(portfolio_history) > 1:
                equity_values = portfolio_history['total_equity'].values
                
                # محاسبه running maximum
                running_max = np.maximum.accumulate(equity_values)
                
                # محاسبه drawdown
                drawdown = (equity_values - running_max) / running_max
                
                # Maximum drawdown
                max_drawdown = abs(drawdown.min())
                self.performance_stats['max_drawdown'] = max_drawdown
                
                # Current drawdown
                current_drawdown = abs(drawdown[-1])
                self.performance_stats['current_drawdown'] = current_drawdown
            
            conn.close()
            
        except Exception as e:
            print(f"خطا در محاسبه drawdown: {e}")

    def _save_portfolio_snapshot(self):
        """ذخیره snapshot از وضعیت فعلی پورتفولیو"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO portfolio_history (
                    total_balance, available_balance, used_margin,
                    unrealized_pnl, realized_pnl, total_equity,
                    margin_level, active_positions, daily_pnl, drawdown
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                self.current_portfolio['total_balance'],
                self.current_portfolio['available_balance'],
                self.current_portfolio['used_margin'],
                self.current_portfolio['unrealized_pnl'],
                self.current_portfolio['realized_pnl'],
                self.current_portfolio['total_equity'],
                self.current_portfolio['margin_level'],
                self.current_portfolio['active_positions_count'],
                self.current_portfolio['daily_pnl'],
                self.performance_stats['current_drawdown']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"خطا در ذخیره snapshot پورتفولیو: {e}")

    def _log_risk_event(self, event_type: str, description: str, impact: float, severity: str):
        """ثبت رویداد ریسک"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO risk_events (event_type, description, portfolio_impact, severity)
                VALUES (?, ?, ?, ?)
            ''', (event_type, description, impact, severity))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"خطا در ثبت رویداد ریسک: {e}")

    def get_portfolio_summary(self) -> Dict:
        """خلاصه وضعیت پورتفولیو"""
        return {
            'current_portfolio': self.current_portfolio.copy(),
            'active_positions': len(self.active_positions),
            'performance_stats': self.performance_stats.copy(),
            'risk_metrics': {
                'margin_usage_pct': (
                    self.current_portfolio['used_margin'] / 
                    self.current_portfolio['total_balance'] * 100
                    if self.current_portfolio['total_balance'] > 0 else 0
                ),
                'equity_ratio': (
                    self.current_portfolio['total_equity'] / 
                    self.current_portfolio['total_balance']
                    if self.current_portfolio['total_balance'] > 0 else 1
                ),
                'daily_risk_pct': (
                    abs(self.current_portfolio['daily_pnl']) / 
                    self.current_portfolio['total_balance'] * 100
                    if self.current_portfolio['total_balance'] > 0 else 0
                )
            }
        }

    def get_active_positions_summary(self) -> Dict:
        """خلاصه پوزیشن‌های فعال"""
        summary = {
            'total_positions': len(self.active_positions),
            'positions_by_symbol': {},
            'positions_by_side': {'LONG': 0, 'SHORT': 0},
            'total_unrealized_pnl': 0,
            'positions': []
        }
        
        for position_id, position in self.active_positions.items():
            symbol = position['symbol']
            side = position['side']
            
            # شمارش بر اساس نماد
            if symbol not in summary['positions_by_symbol']:
                summary['positions_by_symbol'][symbol] = 0
            summary['positions_by_symbol'][symbol] += 1
            
            # شمارش بر اساس جهت
            summary['positions_by_side'][side] += 1
            
            # مجموع PnL
            summary['total_unrealized_pnl'] += position.get('current_pnl', 0)
            
            # جزئیات پوزیشن
            summary['positions'].append({
                'position_id': position_id,
                'symbol': symbol,
                'side': side,
                'entry_price': position['entry_price'],
                'current_pnl': position.get('current_pnl', 0),
                'current_pnl_pct': position.get('current_pnl_pct', 0),
                'leverage': position['leverage'],
                'signal_quality': position.get('signal_quality'),
                'entry_time': position['entry_time'].isoformat()
            })
        
        return summary

    def get_performance_report(self, days: int = 30) -> Dict:
        """گزارش عملکرد برای مدت زمان مشخص"""
        try:
            start_date = datetime.datetime.now() - datetime.timedelta(days=days)
            
            conn = sqlite3.connect(self.db_path)
            
            # معاملات در بازه زمانی
            trades_df = pd.read_sql_query('''
                SELECT * FROM completed_trades 
                WHERE exit_time >= ? 
                ORDER BY exit_time DESC
            ''', conn, params=[start_date])
            
            if trades_df.empty:
                return {'error': 'No trades in specified period'}
            
            # آمار کلی
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl_amount'] > 0])
            total_pnl = trades_df['pnl_amount'].sum()
            
            # آمار بر اساس نماد
            symbol_stats = trades_df.groupby('symbol').agg({
                'pnl_amount': ['count', 'sum', 'mean'],
                'pnl_percentage': 'mean',
                'success': 'mean'
            }).round(4)
            
            # آمار بر اساس نوع سیگنال
            signal_stats = trades_df.groupby('signal_type').agg({
                'pnl_amount': ['count', 'sum', 'mean'],
                'success': 'mean'
            }).round(4)
            
            conn.close()
            
            return {
                'period_days': days,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
                'total_pnl': total_pnl,
                'average_pnl': trades_df['pnl_amount'].mean(),
                'best_trade': trades_df['pnl_amount'].max(),
                'worst_trade': trades_df['pnl_amount'].min(),
                'symbol_performance': symbol_stats.to_dict(),
                'signal_type_performance': signal_stats.to_dict(),
                'recent_trades': trades_df.head(10).to_dict('records')
            }
            
        except Exception as e:
            return {'error': f"Error generating report: {e}"}

    def check_risk_limits(self) -> Dict:
        """بررسی محدودیت‌های ریسک"""
        warnings = []
        risk_level = "LOW"
        
        portfolio_config = config.PORTFOLIO_MANAGEMENT
        
        # بررسی ریسک روزانه
        daily_risk_pct = abs(self.current_portfolio['daily_pnl']) / self.current_portfolio['total_balance']
        if daily_risk_pct > portfolio_config['max_daily_risk']:
            warnings.append(f"Daily risk exceeded: {daily_risk_pct*100:.2f}%")
            risk_level = "HIGH"
        
        # بررسی Drawdown
        if self.performance_stats['current_drawdown'] > portfolio_config['drawdown_limit']:
            warnings.append(f"Drawdown limit exceeded: {self.performance_stats['current_drawdown']*100:.2f}%")
            risk_level = "HIGH"
        
        # بررسی تعداد پوزیشن‌ها
        if len(self.active_positions) > portfolio_config['max_concurrent_positions']:
            warnings.append(f"Too many active positions: {len(self.active_positions)}")
            risk_level = "MEDIUM"
        
        # بررسی exposure کل
        total_exposure = self.current_portfolio['used_margin'] / self.current_portfolio['total_balance']
        if total_exposure > portfolio_config['max_total_exposure']:
            warnings.append(f"Total exposure too high: {total_exposure*100:.2f}%")
            risk_level = "HIGH"
        
        return {
            'risk_level': risk_level,
            'warnings': warnings,
            'daily_risk_pct': daily_risk_pct * 100,
            'current_drawdown_pct': self.performance_stats['current_drawdown'] * 100,
            'total_exposure_pct': total_exposure * 100,
            'active_positions': len(self.active_positions),
            'margin_usage_pct': (self.current_portfolio['used_margin'] / 
                                self.current_portfolio['total_balance'] * 100)
        }

    def reset_daily_stats(self):
        """ریست آمار روزانه"""
        self.current_portfolio['daily_pnl'] = 0.0
        self._save_portfolio_snapshot()
        print("✅ آمار روزانه ریست شد")
