import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import datetime
import sqlite3
import json
from dataclasses import dataclass
from enum import Enum
import warnings

class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

@dataclass
class RiskAlert:
    timestamp: datetime.datetime
    risk_type: str
    level: RiskLevel
    symbol: str
    message: str
    current_value: float
    threshold: float
    recommendation: str

class RiskMonitor:
    """
    سیستم نظارت بر ریسک real-time
    """
    
    def __init__(self, db_path: str = "risk_monitoring.db"):
        self.db_path = db_path
        self._init_database()
        
        # تنظیمات آستانه‌های ریسک
        self.risk_thresholds = {
            'max_drawdown': {
                'medium': 15.0,    # 15%
                'high': 25.0,      # 25%
                'critical': 40.0   # 40%
            },
            'daily_loss': {
                'medium': 5.0,     # 5%
                'high': 10.0,      # 10%
                'critical': 15.0   # 15%
            },
            'portfolio_exposure': {
                'medium': 70.0,    # 70%
                'high': 85.0,      # 85%
                'critical': 95.0   # 95%
            },
            'correlation': {
                'medium': 0.7,     # 70%
                'high': 0.85,      # 85%
                'critical': 0.95   # 95%
            },
            'volatility': {
                'medium': 20.0,    # 20%
                'high': 35.0,      # 35%
                'critical': 50.0   # 50%
            },
            'var_95': {
                'medium': 8.0,     # 8%
                'high': 12.0,      # 12%
                'critical': 18.0   # 18%
            }
        }
        
        # Cache برای محاسبات
        self.portfolio_cache = {}
        self.correlation_cache = {}
        self.active_alerts = {}
        
    def _init_database(self):
        """راه‌اندازی پایگاه داده"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # جدول alerts
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS risk_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                risk_type TEXT NOT NULL,
                level TEXT NOT NULL,
                symbol TEXT,
                message TEXT,
                current_value REAL,
                threshold REAL,
                recommendation TEXT,
                status TEXT DEFAULT 'ACTIVE',
                resolved_time DATETIME
            )
        ''')
        
        # جدول risk metrics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS risk_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT,
                metric_type TEXT NOT NULL,
                value REAL,
                risk_level TEXT,
                notes TEXT
            )
        ''')
        
        # جدول portfolio exposure
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_exposure (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT NOT NULL,
                position_size REAL,
                market_value REAL,
                exposure_percentage REAL,
                leverage REAL,
                unrealized_pnl REAL,
                risk_contribution REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def monitor_portfolio_risk(self, portfolio_data: Dict) -> List[RiskAlert]:
        """نظارت بر ریسک کل پورتفولیو"""
        alerts = []
        
        try:
            # محاسبه معیارهای ریسک
            risk_metrics = self._calculate_portfolio_risk_metrics(portfolio_data)
            
            # بررسی هر معیار
            for metric_name, value in risk_metrics.items():
                alert = self._check_risk_threshold(metric_name, value)
                if alert:
                    alerts.append(alert)
            
            # ذخیره معیارها
            self._save_risk_metrics(risk_metrics)
            
            # بررسی ریسک‌های ترکیبی
            combined_alerts = self._check_combined_risks(portfolio_data, risk_metrics)
            alerts.extend(combined_alerts)
            
            # ذخیره alerts
            for alert in alerts:
                self._save_alert(alert)
            
            return alerts
            
        except Exception as e:
            print(f"خطا در نظارت ریسک پورتفولیو: {e}")
            return []
    
    def monitor_position_risk(self, symbol: str, position_data: Dict) -> List[RiskAlert]:
        """نظارت بر ریسک پوزیشن مشخص"""
        alerts = []
        
        try:
            # محاسبه ریسک پوزیشن
            position_risk = self._calculate_position_risk(symbol, position_data)
            
            # بررسی آستانه‌ها
            if position_risk['loss_percentage'] > self.risk_thresholds['daily_loss']['critical']:
                alert = RiskAlert(
                    timestamp=datetime.datetime.now(),
                    risk_type='POSITION_LOSS',
                    level=RiskLevel.CRITICAL,
                    symbol=symbol,
                    message=f"ضرر {position_risk['loss_percentage']:.1f}% در پوزیشن {symbol}",
                    current_value=position_risk['loss_percentage'],
                    threshold=self.risk_thresholds['daily_loss']['critical'],
                    recommendation="بررسی فوری پوزیشن و احتمال بسته شدن"
                )
                alerts.append(alert)
            
            # بررسی اهرم بالا
            if position_data.get('leverage', 1) > 10:
                alert = RiskAlert(
                    timestamp=datetime.datetime.now(),
                    risk_type='HIGH_LEVERAGE',
                    level=RiskLevel.HIGH,
                    symbol=symbol,
                    message=f"اهرم بالا {position_data['leverage']}X در {symbol}",
                    current_value=position_data['leverage'],
                    threshold=10,
                    recommendation="کاهش اهرم برای مدیریت ریسک"
                )
                alerts.append(alert)
            
            # ذخیره exposure
            self._save_portfolio_exposure(symbol, position_data)
            
            return alerts
            
        except Exception as e:
            print(f"خطا در نظارت ریسک پوزیشن {symbol}: {e}")
            return []
    
    def monitor_market_risk(self, market_data: Dict) -> List[RiskAlert]:
        """نظارت بر ریسک‌های بازار"""
        alerts = []
        
        try:
            # بررسی نوسانات غیرعادی
            for symbol, data in market_data.items():
                volatility = self._calculate_volatility(data)
                
                if volatility > self.risk_thresholds['volatility']['critical']:
                    alert = RiskAlert(
                        timestamp=datetime.datetime.now(),
                        risk_type='HIGH_VOLATILITY',
                        level=RiskLevel.CRITICAL,
                        symbol=symbol,
                        message=f"نوسانات بالا {volatility:.1f}% در {symbol}",
                        current_value=volatility,
                        threshold=self.risk_thresholds['volatility']['critical'],
                        recommendation="احتیاط در ورود پوزیشن جدید"
                    )
                    alerts.append(alert)
            
            # بررسی همبستگی بین نمادها
            correlation_matrix = self._calculate_correlations(market_data)
            high_correlations = self._find_high_correlations(correlation_matrix)
            
            for symbol_pair, correlation in high_correlations:
                if abs(correlation) > self.risk_thresholds['correlation']['high']:
                    alert = RiskAlert(
                        timestamp=datetime.datetime.now(),
                        risk_type='HIGH_CORRELATION',
                        level=RiskLevel.HIGH,
                        symbol=f"{symbol_pair[0]}-{symbol_pair[1]}",
                        message=f"همبستگی بالا {correlation:.2f} بین {symbol_pair[0]} و {symbol_pair[1]}",
                        current_value=abs(correlation),
                        threshold=self.risk_thresholds['correlation']['high'],
                        recommendation="کاهش exposure در نمادهای همبسته"
                    )
                    alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            print(f"خطا در نظارت ریسک بازار: {e}")
            return []
    
    def _calculate_portfolio_risk_metrics(self, portfolio_data: Dict) -> Dict:
        """محاسبه معیارهای ریسک پورتفولیو"""
        metrics = {}
        
        try:
            # Total exposure
            total_exposure = sum(pos.get('market_value', 0) for pos in portfolio_data.values())
            portfolio_value = portfolio_data.get('total_value', total_exposure)
            exposure_ratio = (total_exposure / portfolio_value * 100) if portfolio_value > 0 else 0
            
            metrics['portfolio_exposure'] = exposure_ratio
            
            # محاسبه drawdown
            pnl_values = [pos.get('unrealized_pnl', 0) for pos in portfolio_data.values()]
            cumulative_pnl = np.cumsum(pnl_values)
            
            if len(cumulative_pnl) > 0:
                running_max = np.maximum.accumulate(cumulative_pnl)
                drawdown = (cumulative_pnl - running_max) / running_max * 100
                max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
                metrics['max_drawdown'] = abs(max_drawdown)
            
            # محاسبه VaR
            if len(pnl_values) > 1:
                var_95 = np.percentile(pnl_values, 5)  # 95% VaR
                metrics['var_95'] = abs(var_95)
            
            # محاسبه نوسانات پورتفولیو
            if len(pnl_values) > 1:
                portfolio_volatility = np.std(pnl_values)
                metrics['volatility'] = portfolio_volatility
            
            # ضرر روزانه
            daily_pnl = sum(pos.get('daily_pnl', 0) for pos in portfolio_data.values())
            daily_loss_pct = abs(daily_pnl / portfolio_value * 100) if portfolio_value > 0 and daily_pnl < 0 else 0
            metrics['daily_loss'] = daily_loss_pct
            
            return metrics
            
        except Exception as e:
            print(f"خطا در محاسبه معیارهای ریسک: {e}")
            return {}
    
    def _calculate_position_risk(self, symbol: str, position_data: Dict) -> Dict:
        """محاسبه ریسک پوزیشن"""
        risk_metrics = {}
        
        try:
            # ضرر/سود درصدی
            entry_price = position_data.get('entry_price', 0)
            current_price = position_data.get('current_price', 0)
            
            if entry_price > 0:
                price_change = (current_price - entry_price) / entry_price * 100
                
                if position_data.get('side') == 'SHORT':
                    price_change = -price_change
                
                risk_metrics['loss_percentage'] = abs(price_change) if price_change < 0 else 0
                risk_metrics['profit_percentage'] = price_change if price_change > 0 else 0
            
            # ریسک اهرم
            leverage = position_data.get('leverage', 1)
            risk_metrics['leverage_risk'] = leverage
            
            # مقدار در معرض خطر
            position_size = position_data.get('position_size', 0)
            risk_metrics['amount_at_risk'] = position_size * leverage
            
            return risk_metrics
            
        except Exception as e:
            print(f"خطا در محاسبه ریسک پوزیشن: {e}")
            return {}
    
    def _calculate_volatility(self, price_data: List[float], periods: int = 24) -> float:
        """محاسبه نوسانات"""
        if len(price_data) < 2:
            return 0
        
        returns = np.diff(price_data) / price_data[:-1]
        volatility = np.std(returns) * np.sqrt(periods) * 100
        
        return volatility
    
    def _calculate_correlations(self, market_data: Dict) -> pd.DataFrame:
        """محاسبه ماتریس همبستگی"""
        try:
            # تبدیل به DataFrame
            df_data = {}
            min_length = min(len(data) for data in market_data.values())
            
            for symbol, prices in market_data.items():
                df_data[symbol] = prices[-min_length:]  # استفاده از آخرین داده‌ها
            
            df = pd.DataFrame(df_data)
            correlation_matrix = df.corr()
            
            return correlation_matrix
            
        except Exception as e:
            print(f"خطا در محاسبه همبستگی: {e}")
            return pd.DataFrame()
    
    def _find_high_correlations(self, correlation_matrix: pd.DataFrame) -> List[Tuple]:
        """یافتن همبستگی‌های بالا"""
        high_correlations = []
        
        try:
            symbols = correlation_matrix.columns
            
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols):
                    if i < j:  # جلوگیری از تکرار
                        correlation = correlation_matrix.loc[symbol1, symbol2]
                        if abs(correlation) > self.risk_thresholds['correlation']['medium']:
                            high_correlations.append(((symbol1, symbol2), correlation))
            
            return high_correlations
            
        except Exception as e:
            print(f"خطا در یافتن همبستگی‌های بالا: {e}")
            return []
    
    def _check_risk_threshold(self, metric_name: str, value: float, symbol: str = None) -> Optional[RiskAlert]:
        """بررسی آستانه‌های ریسک"""
        
        if metric_name not in self.risk_thresholds:
            return None
        
        thresholds = self.risk_thresholds[metric_name]
        
        # تعیین سطح ریسک
        if value >= thresholds['critical']:
            level = RiskLevel.CRITICAL
            threshold = thresholds['critical']
        elif value >= thresholds['high']:
            level = RiskLevel.HIGH
            threshold = thresholds['high']
        elif value >= thresholds['medium']:
            level = RiskLevel.MEDIUM
            threshold = thresholds['medium']
        else:
            return None  # ریسک پایین
        
        # تولید پیام و توصیه
        message, recommendation = self._generate_risk_message(metric_name, value, level)
        
        return RiskAlert(
            timestamp=datetime.datetime.now(),
            risk_type=metric_name.upper(),
            level=level,
            symbol=symbol or "PORTFOLIO",
            message=message,
            current_value=value,
            threshold=threshold,
            recommendation=recommendation
        )
    
    def _generate_risk_message(self, metric_name: str, value: float, level: RiskLevel) -> Tuple[str, str]:
        """تولید پیام و توصیه ریسک"""
        
        messages = {
            'max_drawdown': (
                f"حداکثر افت {value:.1f}% - سطح {level.value}",
                "بازبینی استراتژی و کاهش اندازه پوزیشن‌ها"
            ),
            'daily_loss': (
                f"ضرر روزانه {value:.1f}% - سطح {level.value}",
                "توقف معاملات جدید و بررسی پوزیشن‌های فعال"
            ),
            'portfolio_exposure': (
                f"درصد exposure {value:.1f}% - سطح {level.value}",
                "کاهش اهرم و محدود کردن پوزیشن‌های جدید"
            ),
            'volatility': (
                f"نوسانات {value:.1f}% - سطح {level.value}",
                "اجتناب از پوزیشن‌های جدید تا کاهش نوسانات"
            ),
            'var_95': (
                f"VaR 95% برابر {value:.1f}% - سطح {level.value}",
                "کاهش ریسک پورتفولیو و diversification بیشتر"
            )
        }
        
        return messages.get(metric_name, (f"ریسک {metric_name}: {value:.2f}", "بررسی شرایط"))
    
    def _check_combined_risks(self, portfolio_data: Dict, risk_metrics: Dict) -> List[RiskAlert]:
        """بررسی ریسک‌های ترکیبی"""
        alerts = []
        
        try:
            # ترکیب drawdown بالا + نوسانات بالا
            if (risk_metrics.get('max_drawdown', 0) > self.risk_thresholds['max_drawdown']['medium'] and
                risk_metrics.get('volatility', 0) > self.risk_thresholds['volatility']['medium']):
                
                alert = RiskAlert(
                    timestamp=datetime.datetime.now(),
                    risk_type='COMBINED_HIGH_RISK',
                    level=RiskLevel.HIGH,
                    symbol="PORTFOLIO",
                    message="ترکیب drawdown و نوسانات بالا",
                    current_value=risk_metrics.get('max_drawdown', 0),
                    threshold=self.risk_thresholds['max_drawdown']['medium'],
                    recommendation="کاهش فوری اندازه پوزیشن‌ها و توقف معاملات جدید"
                )
                alerts.append(alert)
            
            # ضرر روزانه + exposure بالا
            if (risk_metrics.get('daily_loss', 0) > self.risk_thresholds['daily_loss']['medium'] and
                risk_metrics.get('portfolio_exposure', 0) > self.risk_thresholds['portfolio_exposure']['medium']):
                
                alert = RiskAlert(
                    timestamp=datetime.datetime.now(),
                    risk_type='OVEREXPOSURE_LOSS',
                    level=RiskLevel.CRITICAL,
                    symbol="PORTFOLIO",
                    message="ترکیب exposure بالا و ضرر روزانه",
                    current_value=risk_metrics.get('daily_loss', 0),
                    threshold=self.risk_thresholds['daily_loss']['medium'],
                    recommendation="بسته شدن فوری بخشی از پوزیشن‌ها"
                )
                alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            print(f"خطا در بررسی ریسک‌های ترکیبی: {e}")
            return []
    
    def _save_alert(self, alert: RiskAlert):
        """ذخیره alert در پایگاه داده"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO risk_alerts (
                    risk_type, level, symbol, message, current_value,
                    threshold, recommendation
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.risk_type, alert.level.value, alert.symbol,
                alert.message, alert.current_value, alert.threshold,
                alert.recommendation
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"خطا در ذخیره alert: {e}")
    
    def _save_risk_metrics(self, metrics: Dict, symbol: str = None):
        """ذخیره معیارهای ریسک"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for metric_name, value in metrics.items():
                # تعیین سطح ریسک
                alert = self._check_risk_threshold(metric_name, value, symbol)
                risk_level = alert.level.value if alert else 'LOW'
                
                cursor.execute('''
                    INSERT INTO risk_metrics (
                        symbol, metric_type, value, risk_level
                    ) VALUES (?, ?, ?, ?)
                ''', (symbol, metric_name, value, risk_level))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"خطا در ذخیره معیارهای ریسک: {e}")
    
    def _save_portfolio_exposure(self, symbol: str, position_data: Dict):
        """ذخیره exposure پورتفولیو"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO portfolio_exposure (
                    symbol, position_size, market_value, exposure_percentage,
                    leverage, unrealized_pnl, risk_contribution
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                position_data.get('position_size', 0),
                position_data.get('market_value', 0),
                position_data.get('exposure_percentage', 0),
                position_data.get('leverage', 1),
                position_data.get('unrealized_pnl', 0),
                position_data.get('risk_contribution', 0)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"خطا در ذخیره exposure: {e}")
    
    def get_active_alerts(self, level: RiskLevel = None) -> List[Dict]:
        """دریافت alertهای فعال"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            if level:
                query = '''
                    SELECT * FROM risk_alerts 
                    WHERE status = 'ACTIVE' AND level = ?
                    ORDER BY timestamp DESC
                '''
                df = pd.read_sql_query(query, conn, params=(level.value,))
            else:
                query = '''
                    SELECT * FROM risk_alerts 
                    WHERE status = 'ACTIVE'
                    ORDER BY timestamp DESC
                '''
                df = pd.read_sql_query(query, conn)
            
            conn.close()
            return df.to_dict('records')
            
        except Exception as e:
            print(f"خطا در دریافت alerts: {e}")
            return []
    
    def resolve_alert(self, alert_id: int):
        """حل کردن alert"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE risk_alerts 
                SET status = 'RESOLVED', resolved_time = ?
                WHERE id = ?
            ''', (datetime.datetime.now(), alert_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"خطا در حل alert: {e}")
    
    def get_risk_summary(self) -> Dict:
        """دریافت خلاصه ریسک"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # تعداد alerts فعال
            active_alerts_query = '''
                SELECT level, COUNT(*) as count
                FROM risk_alerts 
                WHERE status = 'ACTIVE'
                GROUP BY level
            '''
            alerts_df = pd.read_sql_query(active_alerts_query, conn)
            
            # آخرین معیارهای ریسک
            latest_metrics_query = '''
                SELECT metric_type, value, risk_level
                FROM risk_metrics 
                WHERE timestamp >= datetime('now', '-1 hour')
                ORDER BY timestamp DESC
                LIMIT 10
            '''
            metrics_df = pd.read_sql_query(latest_metrics_query, conn)
            
            conn.close()
            
            return {
                'timestamp': datetime.datetime.now(),
                'active_alerts': alerts_df.to_dict('records'),
                'latest_metrics': metrics_df.to_dict('records'),
                'overall_risk_level': self._determine_overall_risk_level(alerts_df)
            }
            
        except Exception as e:
            print(f"خطا در تهیه خلاصه ریسک: {e}")
            return {}
    
    def _determine_overall_risk_level(self, alerts_df: pd.DataFrame) -> str:
        """تعیین سطح کلی ریسک"""
        if alerts_df.empty:
            return 'LOW'
        
        if any(alerts_df['level'] == 'CRITICAL'):
            return 'CRITICAL'
        elif any(alerts_df['level'] == 'HIGH'):
            return 'HIGH'
        elif any(alerts_df['level'] == 'MEDIUM'):
            return 'MEDIUM'
        else:
            return 'LOW'
