import pandas as pd
import numpy as np
import sqlite3
import datetime
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum
import config

class RiskLevel(Enum):
    """سطوح ریسک"""
    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"
    EXTREME = "EXTREME"

class PositionStatus(Enum):
    """وضعیت پوزیشن"""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PARTIAL_CLOSED = "PARTIAL_CLOSED"
    STOPPED_OUT = "STOPPED_OUT"
    TAKE_PROFIT_HIT = "TAKE_PROFIT_HIT"

class RiskManager:
    """
    مدیر ریسک جامع برای سیستم معاملاتی چند ارزی
    """

    def __init__(self):
        self.name = "Risk Manager"
        
        # پایگاه داده برای ذخیره اطلاعات ریسک
        self.db_path = "risk_management.db"
        self._init_database()
        
        # ردیابی پوزیشن‌های فعال
        self.active_positions = {}
        
        # آمار روزانه و هفتگی
        self.daily_stats = self._load_daily_stats()
        self.weekly_stats = self._load_weekly_stats()
        
        # وضعیت circuit breaker
        self.circuit_breaker_active = False
        self.circuit_breaker_reason = None
        self.circuit_breaker_timestamp = None
        
        # آمار عملکرد
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0
        }

    def _init_database(self):
        """راه‌اندازی پایگاه داده"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # جدول پوزیشن‌ها
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    position_id TEXT UNIQUE,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    position_size REAL NOT NULL,
                    leverage REAL NOT NULL,
                    stop_loss REAL,
                    take_profit REAL,
                    entry_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    exit_time DATETIME,
                    exit_price REAL,
                    pnl REAL DEFAULT 0,
                    status TEXT DEFAULT 'OPEN',
                    risk_amount REAL,
                    max_loss_allowed REAL,
                    reason_for_exit TEXT
                )
            ''')
            
            # جدول ریسک روزانه
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_risk (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE UNIQUE,
                    total_risk_taken REAL DEFAULT 0,
                    max_risk_allowed REAL,
                    daily_pnl REAL DEFAULT 0,
                    trades_count INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    losing_trades INTEGER DEFAULT 0,
                    max_drawdown REAL DEFAULT 0,
                    circuit_breaker_triggered INTEGER DEFAULT 0
                )
            ''')
            
            # جدول هشدارهای ریسک
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS risk_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    symbol TEXT,
                    resolved INTEGER DEFAULT 0
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"خطا در راه‌اندازی پایگاه داده ریسک: {e}")

    def calculate_position_size(self, symbol: str, signal_data: Dict, 
                              account_balance: float = 10000) -> Dict:
        """
        محاسبه اندازه پوزیشن بر اساس ریسک
        """
        try:
            # دریافت تنظیمات نماد
            symbol_config = config.POSITION_SIZING.get(symbol, {
                'base_size': 0.008, 
                'max_size': 0.012
            })
            
            max_leverage = config.SYMBOL_MAX_LEVERAGE.get(symbol, config.MAX_LEVERAGE)
            
            # محاسبه ریسک مجاز برای این معامله
            max_risk_per_trade = config.PORTFOLIO_MANAGEMENT['max_single_position_risk']
            max_risk_amount = account_balance * max_risk_per_trade
            
            # محاسبه فاصله تا stop loss
            entry_price = signal_data.get('current_price', 0)
            stop_loss = signal_data.get('stop_loss', 0)
            
            if entry_price <= 0 or stop_loss <= 0:
                return self._get_default_position_data(symbol, "Invalid price data")
            
            # محاسبه درصد ریسک
            if signal_data.get('signal', 0) > 0:  # Long position
                risk_percentage = abs(entry_price - stop_loss) / entry_price
            else:  # Short position
                risk_percentage = abs(stop_loss - entry_price) / entry_price
            
            if risk_percentage <= 0:
                return self._get_default_position_data(symbol, "Invalid risk calculation")
            
            # محاسبه اندازه پوزیشن بر اساس ریسک
            position_value = max_risk_amount / risk_percentage
            
            # محدود کردن به حداکثر اندازه مجاز
            max_position_value = account_balance * symbol_config['max_size']
            position_value = min(position_value, max_position_value)
            
            # محاسبه اهرم مناسب
            base_position_value = account_balance * symbol_config['base_size']
            calculated_leverage = min(position_value / base_position_value, max_leverage)
            calculated_leverage = max(calculated_leverage, config.MIN_LEVERAGE)
            
            # تنظیم اهرم بر اساس کیفیت سیگنال
            signal_quality = signal_data.get('signal_quality', 'MEDIUM')
            quality_multiplier = {
                'HIGH': 1.0,
                'MEDIUM': 0.8,
                'LOW': 0.6
            }.get(signal_quality, 0.8)
            
            final_leverage = calculated_leverage * quality_multiplier
            final_leverage = max(min(final_leverage, max_leverage), config.MIN_LEVERAGE)
            
            # محاسبه مقادیر نهایی
            position_size_usd = base_position_value * final_leverage
            position_size_crypto = position_size_usd / entry_price
            
            # محاسبه ریسک واقعی
            actual_risk_amount = position_size_usd * risk_percentage
            risk_percentage_of_account = actual_risk_amount / account_balance
            
            return {
                'symbol': symbol,
                'leverage': round(final_leverage, 1),
                'position_size_usd': round(position_size_usd, 2),
                'position_size_crypto': round(position_size_crypto, 6),
                'risk_amount': round(actual_risk_amount, 2),
                'risk_percentage': round(risk_percentage * 100, 2),
                'risk_percentage_of_account': round(risk_percentage_of_account * 100, 2),
                'quality_adjustment': quality_multiplier,
                'max_loss_allowed': round(max_risk_amount, 2),
                'status': 'APPROVED',
                'warnings': []
            }
            
        except Exception as e:
            print(f"خطا در محاسبه اندازه پوزیشن: {e}")
            return self._get_default_position_data(symbol, f"Calculation error: {e}")

    def _get_default_position_data(self, symbol: str, reason: str) -> Dict:
        """داده‌های پیش‌فرض در صورت خطا"""
        return {
            'symbol': symbol,
            'leverage': config.MIN_LEVERAGE,
            'position_size_usd': 0,
            'position_size_crypto': 0,
            'risk_amount': 0,
            'risk_percentage': 0,
            'risk_percentage_of_account': 0,
            'quality_adjustment': 0,
            'max_loss_allowed': 0,
            'status': 'REJECTED',
            'warnings': [reason]
        }

    def assess_portfolio_risk(self) -> Dict:
        """ارزیابی ریسک کل پورتفولیو"""
        try:
            # محاسبه exposure کل
            total_exposure = 0
            active_positions_count = len(self.active_positions)
            
            for position in self.active_positions.values():
                total_exposure += position.get('position_size_usd', 0)
            
            # محاسبه ریسک روزانه
            daily_risk = self.daily_stats.get('total_risk_taken', 0)
            max_daily_risk = config.PORTFOLIO_MANAGEMENT['max_daily_risk']
            
            # بررسی drawdown
            current_drawdown = self.performance_metrics['current_drawdown']
            max_allowed_drawdown = config.PORTFOLIO_MANAGEMENT['drawdown_limit']
            
            # تعیین سطح ریسک
            risk_level = self._calculate_risk_level(
                total_exposure, daily_risk, current_drawdown, active_positions_count
            )
            
            # بررسی circuit breaker
            should_stop, stop_reason = self._check_circuit_breaker_conditions()
            
            return {
                'risk_level': risk_level.value,
                'total_exposure': total_exposure,
                'exposure_percentage': total_exposure / 10000 * 100,  # فرض بر 10k حساب
                'daily_risk_used': daily_risk,
                'daily_risk_remaining': max(0, max_daily_risk - daily_risk),
                'current_drawdown': current_drawdown * 100,
                'max_drawdown_allowed': max_allowed_drawdown * 100,
                'active_positions': active_positions_count,
                'max_positions_allowed': config.PORTFOLIO_MANAGEMENT['max_concurrent_positions'],
                'circuit_breaker_active': should_stop,
                'circuit_breaker_reason': stop_reason,
                'safe_to_trade': not should_stop and risk_level != RiskLevel.EXTREME,
                'recommendations': self._get_risk_recommendations(risk_level, should_stop)
            }
            
        except Exception as e:
            print(f"خطا در ارزیابی ریسک پورتفولیو: {e}")
            return {
                'risk_level': RiskLevel.EXTREME.value,
                'safe_to_trade': False,
                'error': str(e)
            }

    def _calculate_risk_level(self, exposure: float, daily_risk: float, 
                            drawdown: float, positions_count: int) -> RiskLevel:
        """محاسبه سطح ریسک"""
        
        # تعریف آستانه‌ها
        risk_score = 0
        
        # امتیاز بر اساس exposure
        if exposure > 5000:  # بیش از 50% حساب
            risk_score += 3
        elif exposure > 3000:  # 30-50%
            risk_score += 2
        elif exposure > 1500:  # 15-30%
            risk_score += 1
        
        # امتیاز بر اساس ریسک روزانه
        max_daily = config.PORTFOLIO_MANAGEMENT['max_daily_risk']
        if daily_risk > max_daily * 0.8:
            risk_score += 3
        elif daily_risk > max_daily * 0.6:
            risk_score += 2
        elif daily_risk > max_daily * 0.4:
            risk_score += 1
        
        # امتیاز بر اساس drawdown
        max_dd = config.PORTFOLIO_MANAGEMENT['drawdown_limit']
        if drawdown > max_dd * 0.8:
            risk_score += 3
        elif drawdown > max_dd * 0.6:
            risk_score += 2
        elif drawdown > max_dd * 0.4:
            risk_score += 1
        
        # امتیاز بر اساس تعداد پوزیشن
        max_positions = config.PORTFOLIO_MANAGEMENT['max_concurrent_positions']
        if positions_count >= max_positions:
            risk_score += 2
        elif positions_count >= max_positions * 0.8:
            risk_score += 1
        
        # تعیین سطح نهایی
        if risk_score >= 8:
            return RiskLevel.EXTREME
        elif risk_score >= 6:
            return RiskLevel.VERY_HIGH
        elif risk_score >= 4:
            return RiskLevel.HIGH
        elif risk_score >= 2:
            return RiskLevel.MODERATE
        elif risk_score >= 1:
            return RiskLevel.LOW
        else:
            return RiskLevel.VERY_LOW

    def _check_circuit_breaker_conditions(self) -> Tuple[bool, Optional[str]]:
        """بررسی شرایط فعال‌سازی circuit breaker"""
        
        # بررسی ضرر روزانه
        daily_loss_limit = config.PORTFOLIO_MANAGEMENT['daily_loss_limit']
        if self.daily_stats.get('daily_pnl', 0) < -daily_loss_limit:
            return True, f"Daily loss limit exceeded: {daily_loss_limit*100}%"
        
        # بررسی ضررهای متوالی
        consecutive_limit = config.CIRCUIT_BREAKERS['consecutive_loss_limit']
        if self.daily_stats.get('consecutive_losses', 0) >= consecutive_limit:
            return True, f"Consecutive losses limit: {consecutive_limit}"
        
        # بررسی حداکثر معاملات روزانه
        max_daily_trades = config.CIRCUIT_BREAKERS['max_total_daily_trades']
        if self.daily_stats.get('trades_count', 0) >= max_daily_trades:
            return True, f"Daily trades limit: {max_daily_trades}"
        
        # بررسی drawdown
        max_drawdown = config.PORTFOLIO_MANAGEMENT['drawdown_limit']
        if self.performance_metrics['current_drawdown'] > max_drawdown:
            return True, f"Maximum drawdown exceeded: {max_drawdown*100}%"
        
        return False, None

    def _get_risk_recommendations(self, risk_level: RiskLevel, circuit_breaker: bool) -> List[str]:
        """توصیه‌های مدیریت ریسک"""
        recommendations = []
        
        if circuit_breaker:
            recommendations.append("🛑 توقف کامل معاملات تا بررسی شرایط")
            return recommendations
        
        if risk_level == RiskLevel.EXTREME:
            recommendations.extend([
                "🚨 کاهش فوری اندازه پوزیشن‌ها",
                "⛔ عدم ورود به پوزیشن جدید",
                "📉 بستن پوزیشن‌های پرریسک"
            ])
        elif risk_level == RiskLevel.VERY_HIGH:
            recommendations.extend([
                "🔴 کاهش اندازه پوزیشن‌ها",
                "⚠️ محدود کردن پوزیشن‌های جدید",
                "🎯 تمرکز بر سیگنال‌های با کیفیت بالا"
            ])
        elif risk_level == RiskLevel.HIGH:
            recommendations.extend([
                "🟡 مراقبت بیشتر در انتخاب سیگنال",
                "📊 کاهش اهرم",
                "🔍 بررسی مداوم پوزیشن‌ها"
            ])
        elif risk_level == RiskLevel.MODERATE:
            recommendations.extend([
                "✅ ادامه با احتیاط معقول",
                "📈 حفظ انضباط در مدیریت ریسک"
            ])
        else:
            recommendations.append("🟢 شرایط مناسب برای معاملات محافظه‌کارانه")
        
        return recommendations

    def register_position(self, position_data: Dict) -> bool:
        """ثبت پوزیشن جدید"""
        try:
            position_id = f"{position_data['symbol']}_{datetime.datetime.now().timestamp()}"
            
            # ذخیره در حافظه
            self.active_positions[position_id] = {
                'position_id': position_id,
                'symbol': position_data['symbol'],
                'side': position_data.get('position', 'LONG'),
                'entry_price': position_data.get('entry_price', 0),
                'position_size_usd': position_data.get('position_size_usd', 0),
                'leverage': position_data.get('leverage', config.MIN_LEVERAGE),
                'stop_loss': position_data.get('stop_loss'),
                'take_profit': position_data.get('take_profits', [None])[0],
                'risk_amount': position_data.get('risk_amount', 0),
                'entry_time': datetime.datetime.now(),
                'status': PositionStatus.OPEN.value
            }
            
            # ذخیره در پایگاه داده
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO positions 
                (position_id, symbol, side, entry_price, position_size, leverage, 
                 stop_loss, take_profit, risk_amount, max_loss_allowed, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                position_id,
                position_data['symbol'],
                position_data.get('position', 'LONG'),
                position_data.get('entry_price', 0),
                position_data.get('position_size_usd', 0),
                position_data.get('leverage', config.MIN_LEVERAGE),
                position_data.get('stop_loss'),
                position_data.get('take_profits', [None])[0],
                position_data.get('risk_amount', 0),
                position_data.get('max_loss_allowed', 0),
                PositionStatus.OPEN.value
            ))
            
            conn.commit()
            conn.close()
            
            # به‌روزرسانی آمار روزانه
            self._update_daily_stats('position_opened', position_data.get('risk_amount', 0))
            
            print(f"✅ پوزیشن {position_id} ثبت شد")
            return True
            
        except Exception as e:
            print(f"خطا در ثبت پوزیشن: {e}")
            return False

    def close_position(self, position_id: str, exit_price: float, reason: str = "Manual") -> bool:
        """بستن پوزیشن"""
        try:
            if position_id not in self.active_positions:
                print(f"پوزیشن {position_id} یافت نشد")
                return False
            
            position = self.active_positions[position_id]
            
            # محاسبه PnL
            entry_price = position['entry_price']
            position_size = position['position_size_usd']
            leverage = position['leverage']
            side = position['side']
            
            if side == 'LONG':
                pnl_percentage = (exit_price - entry_price) / entry_price
            else:  # SHORT
                pnl_percentage = (entry_price - exit_price) / entry_price
            
            pnl_amount = position_size * leverage * pnl_percentage
            
            # به‌روزرسانی پایگاه داده
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE positions 
                SET exit_time = ?, exit_price = ?, pnl = ?, status = ?, reason_for_exit = ?
                WHERE position_id = ?
            ''', (
                datetime.datetime.now(),
                exit_price,
                pnl_amount,
                PositionStatus.CLOSED.value,
                reason,
                position_id
            ))
            
            conn.commit()
            conn.close()
            
            # حذف از پوزیشن‌های فعال
            del self.active_positions[position_id]
            
            # به‌روزرسانی آمار عملکرد
            self._update_performance_metrics(pnl_amount, pnl_amount > 0)
            
            print(f"✅ پوزیشن {position_id} بسته شد - PnL: ${pnl_amount:.2f}")
            return True
            
        except Exception as e:
            print(f"خطا در بستن پوزیشن: {e}")
            return False

    def _update_daily_stats(self, action: str, amount: float = 0):
        """به‌روزرسانی آمار روزانه"""
        today = datetime.date.today()
        
        if action == 'position_opened':
            self.daily_stats['total_risk_taken'] = self.daily_stats.get('total_risk_taken', 0) + amount
            self.daily_stats['trades_count'] = self.daily_stats.get('trades_count', 0) + 1
        
        # ذخیره در پایگاه داده
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO daily_risk 
                (date, total_risk_taken, trades_count, daily_pnl)
                VALUES (?, ?, ?, ?)
            ''', (
                today,
                self.daily_stats.get('total_risk_taken', 0),
                self.daily_stats.get('trades_count', 0),
                self.daily_stats.get('daily_pnl', 0)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"خطا در به‌روزرسانی آمار روزانه: {e}")

    def _update_performance_metrics(self, pnl: float, is_winner: bool):
        """به‌روزرسانی معیارهای عملکرد"""
        self.performance_metrics['total_trades'] += 1
        self.performance_metrics['total_pnl'] += pnl
        
        if is_winner:
            self.performance_metrics['winning_trades'] += 1
        else:
            self.performance_metrics['losing_trades'] += 1
        
        # محاسبه win rate
        total = self.performance_metrics['total_trades']
        wins = self.performance_metrics['winning_trades']
        self.performance_metrics['win_rate'] = wins / total if total > 0 else 0
        
        # به‌روزرسانی drawdown
        if pnl < 0:
            self.performance_metrics['current_drawdown'] += abs(pnl) / 10000  # فرض 10k حساب
            self.performance_metrics['max_drawdown'] = max(
                self.performance_metrics['max_drawdown'],
                self.performance_metrics['current_drawdown']
            )
        else:
            self.performance_metrics['current_drawdown'] = max(0, 
                self.performance_metrics['current_drawdown'] - pnl / 10000)

    def _load_daily_stats(self) -> Dict:
        """بارگذاری آمار روزانه"""
        try:
            today = datetime.date.today()
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM daily_risk WHERE date = ?
            ''', (today,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    'total_risk_taken': result[2],
                    'daily_pnl': result[4],
                    'trades_count': result[5],
                    'winning_trades': result[6],
                    'losing_trades': result[7]
                }
            else:
                return {
                    'total_risk_taken': 0,
                    'daily_pnl': 0,
                    'trades_count': 0,
                    'winning_trades': 0,
                    'losing_trades': 0
                }
                
        except Exception as e:
            print(f"خطا در بارگذاری آمار روزانه: {e}")
            return {}

    def _load_weekly_stats(self) -> Dict:
        """بارگذاری آمار هفتگی"""
        # پیاده‌سازی مشابه daily_stats
        return {
            'weekly_pnl': 0,
            'weekly_trades': 0,
            'weekly_win_rate': 0
        }

    def get_risk_summary(self) -> Dict:
        """خلاصه وضعیت ریسک"""
        portfolio_risk = self.assess_portfolio_risk()
        
        return {
            'timestamp': datetime.datetime.now(),
            'portfolio_risk': portfolio_risk,
            'active_positions_count': len(self.active_positions),
            'daily_stats': self.daily_stats,
            'performance_metrics': self.performance_metrics,
            'circuit_breaker_active': self.circuit_breaker_active,
            'recommendations': portfolio_risk.get('recommendations', [])
        }

    def emergency_stop(self, reason: str) -> bool:
        """توقف اضطراری"""
        try:
            self.circuit_breaker_active = True
            self.circuit_breaker_reason = reason
            self.circuit_breaker_timestamp = datetime.datetime.now()
            
            # ثبت هشدار
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO risk_alerts (alert_type, severity, message)
                VALUES (?, ?, ?)
            ''', ('EMERGENCY_STOP', 'CRITICAL', reason))
            
            conn.commit()
            conn.close()
            
            print(f"🚨 EMERGENCY STOP ACTIVATED: {reason}")
            return True
            
        except Exception as e:
            print(f"خطا در توقف اضطراری: {e}")
            return False

    def reset_circuit_breaker(self) -> bool:
        """بازنشانی circuit breaker"""
        try:
            self.circuit_breaker_active = False
            self.circuit_breaker_reason = None
            self.circuit_breaker_timestamp = None
            
            print("✅ Circuit breaker reset شد")
            return True
            
        except Exception as e:
            print(f"خطا در reset circuit breaker: {e}")
            return False
