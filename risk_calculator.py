import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from enum import Enum
import config
import datetime
import sqlite3
import json

class RiskLevel(Enum):
    """سطوح ریسک"""
    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"
    EXTREME = "EXTREME"

class RiskMetric(Enum):
    """متریک‌های ریسک"""
    POSITION_SIZE = "POSITION_SIZE"
    LEVERAGE = "LEVERAGE"
    CORRELATION = "CORRELATION"
    VOLATILITY = "VOLATILITY"
    LIQUIDITY = "LIQUIDITY"
    DRAWDOWN = "DRAWDOWN"
    VAR = "VAR"
    CONCENTRATION = "CONCENTRATION"

class RiskCalculator:
    """
    ماشین حساب ریسک پیشرفته برای سیستم معاملاتی چند ارزی
    """

    def __init__(self, db_path: str = "risk_calculations.db"):
        self.db_path = db_path
        self.name = "Advanced Risk Calculator"
        
        # پارامترهای ریسک از config
        self.risk_params = {
            'max_daily_risk': config.PORTFOLIO_MANAGEMENT['max_daily_risk'],
            'max_single_position_risk': config.PORTFOLIO_MANAGEMENT['max_single_position_risk'],
            'max_total_exposure': config.PORTFOLIO_MANAGEMENT['max_total_exposure'],
            'drawdown_limit': config.PORTFOLIO_MANAGEMENT['drawdown_limit'],
            'daily_loss_limit': config.PORTFOLIO_MANAGEMENT['daily_loss_limit'],
            'max_concurrent_positions': config.PORTFOLIO_MANAGEMENT['max_concurrent_positions']
        }
        
        # تنظیمات نماد-محور
        self.symbol_settings = config.SYMBOL_SPECIFIC_SETTINGS
        self.position_sizing = config.POSITION_SIZING
        self.symbol_max_leverage = config.SYMBOL_MAX_LEVERAGE
        
        # کش محاسبات
        self.calculation_cache = {}
        self.correlation_cache = {}
        self.volatility_cache = {}
        
        # تاریخچه محاسبات
        self.risk_history = []
        
        self._init_database()

    def _init_database(self):
        """ایجاد جداول ریسک"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # جدول محاسبات ریسک
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS risk_calculations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT NOT NULL,
                    calculation_type TEXT NOT NULL,
                    input_data TEXT,
                    risk_score REAL,
                    risk_level TEXT,
                    recommended_position_size REAL,
                    recommended_leverage REAL,
                    max_loss_amount REAL,
                    var_95 REAL,
                    var_99 REAL,
                    expected_shortfall REAL,
                    correlation_risk REAL,
                    liquidity_risk REAL,
                    volatility_risk REAL,
                    concentration_risk REAL,
                    total_portfolio_risk REAL
                )
            ''')

            # جدول ماتریس همبستگی
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS correlation_matrix (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol1 TEXT NOT NULL,
                    symbol2 TEXT NOT NULL,
                    correlation_coefficient REAL,
                    timeframe TEXT,
                    period_days INTEGER,
                    significance_level REAL
                )
            ''')

            # جدول محاسبات نوسانات
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS volatility_calculations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT NOT NULL,
                    timeframe TEXT,
                    historical_volatility REAL,
                    realized_volatility REAL,
                    implied_volatility REAL,
                    volatility_percentile REAL,
                    volatility_trend TEXT,
                    risk_adjusted_volatility REAL
                )
            ''')

            # جدول ارزیابی ریسک پورتفولیو
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_risk_assessment (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    total_exposure REAL,
                    diversification_ratio REAL,
                    correlation_risk_score REAL,
                    concentration_risk_score REAL,
                    liquidity_risk_score REAL,
                    leverage_risk_score REAL,
                    var_portfolio_95 REAL,
                    var_portfolio_99 REAL,
                    max_drawdown_estimate REAL,
                    risk_budget_utilization REAL,
                    overall_risk_score REAL,
                    overall_risk_level TEXT,
                    recommendations TEXT
                )
            ''')

            conn.commit()
            conn.close()
            print("✅ جداول محاسبات ریسک ایجاد شدند")

        except Exception as e:
            print(f"خطا در ایجاد جداول ریسک: {e}")

    def calculate_position_risk(self, symbol: str, entry_price: float, stop_loss: float,
                              position_size_pct: float, leverage: int, market_data: Dict,
                              portfolio_data: Dict) -> Dict:
        """محاسبه جامع ریسک پوزیشن"""
        try:
            # ابتدا محاسبات پایه
            basic_risk = self._calculate_basic_position_risk(
                symbol, entry_price, stop_loss, position_size_pct, leverage
            )
            
            # محاسبه ریسک نوسانات
            volatility_risk = self._calculate_volatility_risk(symbol, market_data)
            
            # محاسبه ریسک همبستگی
            correlation_risk = self._calculate_correlation_risk(symbol, market_data, portfolio_data)
            
            # محاسبه ریسک نقدینگی
            liquidity_risk = self._calculate_liquidity_risk(symbol, market_data)
            
            # محاسبه VaR و Expected Shortfall
            var_metrics = self._calculate_var_metrics(
                symbol, entry_price, position_size_pct, leverage, market_data
            )
            
            # محاسبه ریسک تمرکز
            concentration_risk = self._calculate_concentration_risk(symbol, portfolio_data)
            
            # ترکیب همه متریک‌ها
            total_risk_score = self._combine_risk_metrics({
                'basic_risk': basic_risk['risk_score'],
                'volatility_risk': volatility_risk['risk_score'],
                'correlation_risk': correlation_risk['risk_score'],
                'liquidity_risk': liquidity_risk['risk_score'],
                'concentration_risk': concentration_risk['risk_score']
            })
            
            # تعیین سطح ریسک
            risk_level = self._determine_risk_level(total_risk_score)
            
            # محاسبه توصیه‌های بهینه
            recommendations = self._calculate_optimal_recommendations(
                symbol, total_risk_score, basic_risk, market_data, portfolio_data
            )
            
            # نتیجه نهایی
            result = {
                'symbol': symbol,
                'timestamp': datetime.datetime.now(),
                'total_risk_score': total_risk_score,
                'risk_level': risk_level,
                'basic_risk': basic_risk,
                'volatility_risk': volatility_risk,
                'correlation_risk': correlation_risk,
                'liquidity_risk': liquidity_risk,
                'concentration_risk': concentration_risk,
                'var_metrics': var_metrics,
                'recommendations': recommendations,
                'risk_breakdown': {
                    'position_risk': basic_risk['risk_score'] * 0.25,
                    'market_risk': volatility_risk['risk_score'] * 0.20,
                    'correlation_risk': correlation_risk['risk_score'] * 0.20,
                    'liquidity_risk': liquidity_risk['risk_score'] * 0.15,
                    'concentration_risk': concentration_risk['risk_score'] * 0.20
                }
            }
            
            # ذخیره در پایگاه داده
            self._save_risk_calculation(result)
            
            return result

        except Exception as e:
            print(f"خطا در محاسبه ریسک پوزیشن {symbol}: {e}")
            return None

    def _calculate_basic_position_risk(self, symbol: str, entry_price: float, 
                                     stop_loss: float, position_size_pct: float, 
                                     leverage: int) -> Dict:
        """محاسبه ریسک پایه پوزیشن"""
        try:
            # محاسبه ریسک بر اساس stop loss
            price_risk = abs(entry_price - stop_loss) / entry_price
            
            # ریسک با در نظر گیری اهرم
            leveraged_risk = price_risk * leverage
            
            # ریسک نهایی با اندازه پوزیشن
            position_risk = leveraged_risk * position_size_pct
            
            # محاسبه حداکثر ضرر
            max_loss_pct = position_risk
            max_loss_amount = max_loss_pct  # به صورت درصد
            
            # امتیازدهی ریسک (0 تا 100)
            risk_score = min(position_risk * 100, 100)
            
            # تطبیق بر اساس تنظیمات نماد
            symbol_multiplier = self.symbol_settings.get(symbol, {}).get('volatility_multiplier', 1.0)
            adjusted_risk_score = risk_score * symbol_multiplier
            
            return {
                'price_risk': price_risk,
                'leveraged_risk': leveraged_risk,
                'position_risk': position_risk,
                'max_loss_pct': max_loss_pct,
                'max_loss_amount': max_loss_amount,
                'risk_score': min(adjusted_risk_score, 100),
                'risk_factors': {
                    'stop_loss_distance': price_risk,
                    'leverage_multiplier': leverage,
                    'position_size': position_size_pct,
                    'symbol_volatility': symbol_multiplier
                }
            }

        except Exception as e:
            print(f"خطا در محاسبه ریسک پایه: {e}")
            return {'risk_score': 50, 'max_loss_pct': 0.02}

    def _calculate_volatility_risk(self, symbol: str, market_data: Dict) -> Dict:
        """محاسبه ریسک نوسانات"""
        try:
            if symbol not in market_data:
                return {'risk_score': 50, 'volatility_level': 'MEDIUM'}
            
            data = market_data[symbol]
            
            # استخراج اطلاعات نوسانات
            atr = data.get('atr', 0)
            current_price = data.get('close', 1)
            price_changes = data.get('price_changes', [])
            
            # محاسبه نوسانات تاریخی
            if price_changes and len(price_changes) > 1:
                returns = np.array(price_changes) / 100  # تبدیل درصد به عدد اعشاری
                historical_volatility = np.std(returns) * np.sqrt(252)  # سالانه
            else:
                historical_volatility = atr / current_price if current_price > 0 else 0.02
            
            # محاسبه ATR نرمال شده
            normalized_atr = atr / current_price if current_price > 0 else 0.02
            
            # ترکیب دو روش
            volatility_estimate = (historical_volatility + normalized_atr) / 2
            
            # تعیین سطح نوسانات
            if volatility_estimate < 0.15:
                volatility_level = 'LOW'
                base_score = 20
            elif volatility_estimate < 0.25:
                volatility_level = 'MEDIUM'
                base_score = 40
            elif volatility_estimate < 0.40:
                volatility_level = 'HIGH'
                base_score = 70
            else:
                volatility_level = 'EXTREME'
                base_score = 90
            
            # تطبیق نهایی
            risk_score = min(base_score + (volatility_estimate * 100), 100)
            
            return {
                'risk_score': risk_score,
                'volatility_level': volatility_level,
                'historical_volatility': historical_volatility,
                'normalized_atr': normalized_atr,
                'volatility_estimate': volatility_estimate,
                'volatility_percentile': self._get_volatility_percentile(symbol, volatility_estimate)
            }

        except Exception as e:
            print(f"خطا در محاسبه ریسک نوسانات: {e}")
            return {'risk_score': 50, 'volatility_level': 'MEDIUM'}

    def _calculate_correlation_risk(self, symbol: str, market_data: Dict, 
                                  portfolio_data: Dict) -> Dict:
        """محاسبه ریسک همبستگی"""
        try:
            active_positions = portfolio_data.get('active_positions', {})
            
            if not active_positions:
                return {'risk_score': 0, 'correlation_level': 'NONE'}
            
            # بررسی همبستگی با پوزیشن‌های فعال
            correlations = []
            position_symbols = [pos.get('symbol', '') for pos in active_positions.values()]
            
            for pos_symbol in position_symbols:
                if pos_symbol != symbol and pos_symbol in market_data and symbol in market_data:
                    correlation = self._calculate_pairwise_correlation(
                        symbol, pos_symbol, market_data
                    )
                    correlations.append(abs(correlation))
            
            if not correlations:
                return {'risk_score': 0, 'correlation_level': 'NONE'}
            
            # محاسبه متوسط همبستگی
            avg_correlation = np.mean(correlations)
            max_correlation = np.max(correlations)
            
            # امتیازدهی بر اساس همبستگی
            if max_correlation < 0.3:
                correlation_level = 'LOW'
                base_score = 10
            elif max_correlation < 0.5:
                correlation_level = 'MEDIUM'
                base_score = 30
            elif max_correlation < 0.7:
                correlation_level = 'HIGH'
                base_score = 60
            else:
                correlation_level = 'EXTREME'
                base_score = 90
            
            # تعدیل بر اساس تعداد پوزیشن‌های همبسته
            correlation_adjustment = len(correlations) * 10
            final_score = min(base_score + correlation_adjustment, 100)
            
            return {
                'risk_score': final_score,
                'correlation_level': correlation_level,
                'avg_correlation': avg_correlation,
                'max_correlation': max_correlation,
                'correlated_positions': len(correlations),
                'individual_correlations': dict(zip(position_symbols, correlations))
            }

        except Exception as e:
            print(f"خطا در محاسبه ریسک همبستگی: {e}")
            return {'risk_score': 30, 'correlation_level': 'MEDIUM'}

    def _calculate_liquidity_risk(self, symbol: str, market_data: Dict) -> Dict:
        """محاسبه ریسک نقدینگی"""
        try:
            if symbol not in market_data:
                return {'risk_score': 50, 'liquidity_level': 'MEDIUM'}
            
            data = market_data[symbol]
            
            # اطلاعات حجم و نقدینگی
            volume_24h = data.get('volume', 0)
            volume_ma = data.get('volume_ma', volume_24h)
            bid_ask_spread = data.get('bid_ask_spread', 0.001)  # فرض: 0.1%
            market_depth = data.get('market_depth', 1000000)  # فرض: $1M
            
            # محاسبه نسبت حجم
            volume_ratio = volume_24h / volume_ma if volume_ma > 0 else 1
            
            # آستانه‌های نقدینگی برای هر نماد
            min_volumes = config.MARKET_DATA_SETTINGS
            min_volume = min_volumes.get(f'min_volume_threshold_{symbol.split("/")[0].lower()}', 10000000)
            
            # امتیازدهی نقدینگی
            if volume_24h >= min_volume * 2:
                liquidity_level = 'HIGH'
                volume_score = 10
            elif volume_24h >= min_volume:
                liquidity_level = 'MEDIUM'
                volume_score = 30
            elif volume_24h >= min_volume * 0.5:
                liquidity_level = 'LOW'
                volume_score = 60
            else:
                liquidity_level = 'VERY_LOW'
                volume_score = 90
            
            # امتیاز spread
            if bid_ask_spread < 0.001:  # کمتر از 0.1%
                spread_score = 5
            elif bid_ask_spread < 0.005:  # کمتر از 0.5%
                spread_score = 15
            elif bid_ask_spread < 0.01:  # کمتر از 1%
                spread_score = 30
            else:
                spread_score = 50
            
            # امتیاز نهایی
            final_score = min(volume_score + spread_score, 100)
            
            return {
                'risk_score': final_score,
                'liquidity_level': liquidity_level,
                'volume_24h': volume_24h,
                'volume_ratio': volume_ratio,
                'bid_ask_spread': bid_ask_spread,
                'market_depth': market_depth,
                'min_volume_threshold': min_volume,
                'volume_adequacy': volume_24h / min_volume if min_volume > 0 else 1
            }

        except Exception as e:
            print(f"خطا در محاسبه ریسک نقدینگی: {e}")
            return {'risk_score': 40, 'liquidity_level': 'MEDIUM'}

    def _calculate_concentration_risk(self, symbol: str, portfolio_data: Dict) -> Dict:
        """محاسبه ریسک تمرکز"""
        try:
            active_positions = portfolio_data.get('active_positions', {})
            total_exposure = portfolio_data.get('total_exposure', 0)
            
            if not active_positions:
                return {'risk_score': 0, 'concentration_level': 'NONE'}
            
            # محاسبه وزن هر پوزیشن
            position_weights = {}
            total_position_value = 0
            
            for pos_id, position in active_positions.items():
                pos_symbol = position.get('symbol', '')
                pos_size = position.get('position_size_pct', 0)
                pos_leverage = position.get('leverage', 1)
                exposure = pos_size * pos_leverage
                
                position_weights[pos_symbol] = position_weights.get(pos_symbol, 0) + exposure
                total_position_value += exposure
            
            # اگر نماد جدید اضافه شود
            new_position_size = 0.01  # فرض: 1% جدید
            new_leverage = self.symbol_max_leverage.get(symbol, 5)
            new_exposure = new_position_size * new_leverage
            
            # وزن جدید نماد
            total_with_new = total_position_value + new_exposure
            new_symbol_weight = (position_weights.get(symbol, 0) + new_exposure) / total_with_new
            
            # محاسبه شاخص هرفیندال (Herfindahl Index)
            weights_with_new = position_weights.copy()
            weights_with_new[symbol] = weights_with_new.get(symbol, 0) + new_exposure
            
            normalized_weights = [w / total_with_new for w in weights_with_new.values()]
            herfindahl_index = sum(w**2 for w in normalized_weights)
            
            # امتیازدهی تمرکز
            if herfindahl_index < 0.2:  # تنوع بالا
                concentration_level = 'LOW'
                base_score = 10
            elif herfindahl_index < 0.4:  # تنوع متوسط
                concentration_level = 'MEDIUM'
                base_score = 30
            elif herfindahl_index < 0.6:  # تمرکز بالا
                concentration_level = 'HIGH'
                base_score = 60
            else:  # تمرکز خیلی بالا
                concentration_level = 'EXTREME'
                base_score = 90
            
            # جریمه اگر یک نماد بیش از حد غالب باشد
            if new_symbol_weight > 0.5:  # بیش از 50%
                concentration_penalty = 40
            elif new_symbol_weight > 0.3:  # بیش از 30%
                concentration_penalty = 20
            else:
                concentration_penalty = 0
            
            final_score = min(base_score + concentration_penalty, 100)
            
            return {
                'risk_score': final_score,
                'concentration_level': concentration_level,
                'herfindahl_index': herfindahl_index,
                'symbol_weight': new_symbol_weight,
                'total_positions': len(active_positions),
                'position_weights': position_weights,
                'diversification_ratio': 1 / herfindahl_index if herfindahl_index > 0 else 1
            }

        except Exception as e:
            print(f"خطا در محاسبه ریسک تمرکز: {e}")
            return {'risk_score': 30, 'concentration_level': 'MEDIUM'}

    def _calculate_var_metrics(self, symbol: str, entry_price: float, 
                             position_size_pct: float, leverage: int, 
                             market_data: Dict) -> Dict:
        """محاسبه Value at Risk و Expected Shortfall"""
        try:
            if symbol not in market_data:
                return {'var_95': 0.02, 'var_99': 0.03, 'expected_shortfall': 0.04}
            
            data = market_data[symbol]
            
            # دریافت تغییرات قیمت تاریخی
            price_changes = data.get('price_changes', [])
            if not price_changes or len(price_changes) < 30:
                # استفاده از روش پارامتری
                volatility = data.get('atr', 0) / entry_price if entry_price > 0 else 0.02
                return self._parametric_var(volatility, position_size_pct, leverage)
            
            # تبدیل به بازده
            returns = np.array(price_changes) / 100  # از درصد به عدد
            
            # تطبیق برای اهرم و اندازه پوزیشن
            portfolio_returns = returns * leverage * position_size_pct
            
            # محاسبه VaR
            var_95 = np.percentile(portfolio_returns, 5) * -1  # 5th percentile منفی
            var_99 = np.percentile(portfolio_returns, 1) * -1  # 1st percentile منفی
            
            # محاسبه Expected Shortfall (Conditional VaR)
            tail_losses_95 = portfolio_returns[portfolio_returns <= -var_95]
            tail_losses_99 = portfolio_returns[portfolio_returns <= -var_99]
            
            expected_shortfall_95 = np.mean(tail_losses_95) * -1 if len(tail_losses_95) > 0 else var_95 * 1.3
            expected_shortfall_99 = np.mean(tail_losses_99) * -1 if len(tail_losses_99) > 0 else var_99 * 1.3
            
            return {
                'var_95': float(var_95),
                'var_99': float(var_99),
                'expected_shortfall_95': float(expected_shortfall_95),
                'expected_shortfall_99': float(expected_shortfall_99),
                'calculation_method': 'historical',
                'sample_size': len(price_changes),
                'worst_case_scenario': float(np.min(portfolio_returns)) * -1
            }

        except Exception as e:
            print(f"خطا در محاسبه VaR: {e}")
            return self._parametric_var(0.02, position_size_pct, leverage)

    def _parametric_var(self, volatility: float, position_size_pct: float, leverage: int) -> Dict:
        """محاسبه VaR پارامتری"""
        try:
            # فرض توزیع نرمال
            z_95 = 1.645  # 95% confidence
            z_99 = 2.326  # 99% confidence
            
            # تطبیق برای اهرم و اندازه پوزیشن
            portfolio_volatility = volatility * leverage * position_size_pct
            
            var_95 = z_95 * portfolio_volatility
            var_99 = z_99 * portfolio_volatility
            
            # Expected Shortfall برای توزیع نرمال
            expected_shortfall_95 = portfolio_volatility * 2.063  # E[X|X > 95th percentile]
            expected_shortfall_99 = portfolio_volatility * 2.665  # E[X|X > 99th percentile]
            
            return {
                'var_95': float(var_95),
                'var_99': float(var_99),
                'expected_shortfall_95': float(expected_shortfall_95),
                'expected_shortfall_99': float(expected_shortfall_99),
                'calculation_method': 'parametric',
                'portfolio_volatility': float(portfolio_volatility)
            }

        except Exception as e:
            print(f"خطا در محاسبه VaR پارامتری: {e}")
            return {'var_95': 0.02, 'var_99': 0.03, 'expected_shortfall_95': 0.025, 'expected_shortfall_99': 0.04}

    def _calculate_pairwise_correlation(self, symbol1: str, symbol2: str, market_data: Dict) -> float:
        """محاسبه همبستگی بین دو نماد"""
        try:
            # بررسی کش
            cache_key = f"{symbol1}_{symbol2}"
            if cache_key in self.correlation_cache:
                cached_data = self.correlation_cache[cache_key]
                if (datetime.datetime.now() - cached_data['timestamp']).seconds < 3600:  # یک ساعت
                    return cached_data['correlation']
            
            data1 = market_data.get(symbol1, {})
            data2 = market_data.get(symbol2, {})
            
            changes1 = data1.get('price_changes', [])
            changes2 = data2.get('price_changes', [])
            
            if not changes1 or not changes2 or len(changes1) < 10 or len(changes2) < 10:
                # همبستگی پیش‌فرض بر اساس نوع ارز
                default_correlations = {
                    ('BTC/USDT', 'ETH/USDT'): 0.7,
                    ('BTC/USDT', 'SOL/USDT'): 0.6,
                    ('ETH/USDT', 'SOL/USDT'): 0.65
                }
                pair = tuple(sorted([symbol1, symbol2]))
                correlation = default_correlations.get(pair, 0.5)
            else:
                # همبستگی واقعی
                min_length = min(len(changes1), len(changes2))
                array1 = np.array(changes1[-min_length:])
                array2 = np.array(changes2[-min_length:])
                
                correlation_matrix = np.corrcoef(array1, array2)
                correlation = float(correlation_matrix[0, 1]) if not np.isnan(correlation_matrix[0, 1]) else 0.5
            
            # ذخیره در کش
            self.correlation_cache[cache_key] = {
                'correlation': correlation,
                'timestamp': datetime.datetime.now()
            }
            
            return correlation

        except Exception as e:
            print(f"خطا در محاسبه همبستگی: {e}")
            return 0.5

    def _combine_risk_metrics(self, risk_components: Dict) -> float:
        """ترکیب متریک‌های مختلف ریسک"""
        try:
            # وزن‌های هر متریک
            weights = {
                'basic_risk': 0.30,      # ریسک پایه مهم‌ترین است
                'volatility_risk': 0.25,  # نوسانات مهم
                'correlation_risk': 0.20, # همبستگی
                'liquidity_risk': 0.15,   # نقدینگی
                'concentration_risk': 0.10 # تمرکز
            }
            
            # محاسبه میانگین وزنی
            total_score = 0
            total_weight = 0
            
            for component, score in risk_components.items():
                if component in weights:
                    weight = weights[component]
                    total_score += score * weight
                    total_weight += weight
            
            # نرمال‌سازی
            if total_weight > 0:
                final_score = total_score / total_weight
            else:
                final_score = 50  # متوسط
            
            return min(max(final_score, 0), 100)

        except Exception as e:
            print(f"خطا در ترکیب متریک‌های ریسک: {e}")
            return 50

    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """تعیین سطح ریسک بر اساس امتیاز"""
        if risk_score < 20:
            return RiskLevel.VERY_LOW
        elif risk_score < 40:
            return RiskLevel.LOW
        elif risk_score < 60:
            return RiskLevel.MEDIUM
        elif risk_score < 80:
            return RiskLevel.HIGH
        elif risk_score < 95:
            return RiskLevel.VERY_HIGH
        else:
            return RiskLevel.EXTREME

    def _calculate_optimal_recommendations(self, symbol: str, risk_score: float,
                                         basic_risk: Dict, market_data: Dict,
                                         portfolio_data: Dict) -> Dict:
        """محاسبه توصیه‌های بهینه"""
        try:
            # اندازه پوزیشن پیشنهادی
            base_position_size = self.position_sizing.get(symbol, {}).get('base_size', 0.01)
            max_position_size = self.position_sizing.get(symbol, {}).get('max_size', 0.015)
            
            # تطبیق بر اساس ریسک
            if risk_score < 30:
                recommended_position_size = max_position_size
                risk_multiplier = 1.0
            elif risk_score < 50:
                recommended_position_size = (base_position_size + max_position_size) / 2
                risk_multiplier = 0.8
            elif risk_score < 70:
                recommended_position_size = base_position_size
                risk_multiplier = 0.6
            elif risk_score < 85:
                recommended_position_size = base_position_size * 0.7
                risk_multiplier = 0.4
            else:
                recommended_position_size = base_position_size * 0.5
                risk_multiplier = 0.2
            
            # اهرم پیشنهادی
            max_leverage = self.symbol_max_leverage.get(symbol, 5)
            min_leverage = config.MIN_LEVERAGE
            
            if risk_score < 30:
                recommended_leverage = max_leverage
            elif risk_score < 50:
                recommended_leverage = int(max_leverage * 0.8)
            elif risk_score < 70:
                recommended_leverage = int(max_leverage * 0.6)
            else:
                recommended_leverage = min_leverage
            
            recommended_leverage = max(min_leverage, recommended_leverage)
            
            # محاسبه stop loss بهینه
            atr = market_data.get(symbol, {}).get('atr', 0)
            current_price = market_data.get(symbol, {}).get('close', 1)
            atr_multiplier = 2.0 if risk_score < 50 else 1.5
            
            optimal_stop_distance = (atr / current_price) * atr_multiplier if current_price > 0 else 0.02
            
            return {
                'recommended_position_size': recommended_position_size,
                'recommended_leverage': recommended_leverage,
                'risk_multiplier': risk_multiplier,
                'optimal_stop_distance': optimal_stop_distance,
                'max_loss_recommendation': recommended_position_size * recommended_leverage * optimal_stop_distance,
                'confidence_level': 'HIGH' if risk_score < 40 else 'MEDIUM' if risk_score < 70 else 'LOW',
                'trade_recommendation': 'PROCEED' if risk_score < 70 else 'CAUTION' if risk_score < 85 else 'AVOID',
                'risk_management_notes': self._generate_risk_notes(risk_score, symbol)
            }

        except Exception as e:
            print(f"خطا در محاسبه توصیه‌ها: {e}")
            return {
                'recommended_position_size': 0.005,
                'recommended_leverage': 2,
                'trade_recommendation': 'CAUTION'
            }

    def _generate_risk_notes(self, risk_score: float, symbol: str) -> List[str]:
        """تولید یادداشت‌های ریسک"""
        notes = []
        
        if risk_score < 30:
            notes.append("Risk level acceptable for standard position sizing")
        elif risk_score < 50:
            notes.append("Moderate risk - consider reducing position size")
        elif risk_score < 70:
            notes.append("High risk - use conservative position sizing")
        elif risk_score < 85:
            notes.append("Very high risk - minimum position size recommended")
        else:
            notes.append("Extreme risk - consider avoiding this trade")
        
        # یادداشت‌های اختصاصی نماد
        if symbol == 'SOL/USDT':
            notes.append("SOL typically shows higher volatility")
        elif symbol == 'BTC/USDT':
            notes.append("BTC movements affect entire crypto market")
        elif symbol == 'ETH/USDT':
            notes.append("ETH correlation with BTC should be monitored")
        
        return notes

    def calculate_portfolio_risk(self, portfolio_data: Dict, market_data: Dict) -> Dict:
        """محاسبه ریسک کل پورتفولیو"""
        try:
            active_positions = portfolio_data.get('active_positions', {})
            
            if not active_positions:
                return {
                    'overall_risk_score': 0,
                    'overall_risk_level': RiskLevel.VERY_LOW,
                    'recommendations': ['No active positions - risk is minimal']
                }
            
            # محاسبه ریسک هر پوزیشن
            position_risks = []
            total_exposure = 0
            
            for pos_id, position in active_positions.items():
                symbol = position.get('symbol', '')
                entry_price = position.get('entry_price', 0)
                current_price = position.get('current_price', entry_price)
                position_size = position.get('position_size_pct', 0)
                leverage = position.get('leverage', 1)
                
                # محاسبه ریسک این پوزیشن
                if symbol in market_data:
                    pos_risk = self.calculate_position_risk(
                        symbol, current_price, entry_price * 0.95,  # فرض 5% stop loss
                        position_size, leverage, market_data, portfolio_data
                    )
                    
                    if pos_risk:
                        position_risks.append({
                            'symbol': symbol,
                            'risk_score': pos_risk['total_risk_score'],
                            'exposure': position_size * leverage
                        })
                        total_exposure += position_size * leverage
            
            # محاسبه ریسک پورتفولیو
            if position_risks:
                # میانگین وزنی ریسک
                weighted_risk = sum(
                    pos['risk_score'] * pos['exposure'] for pos in position_risks
                ) / total_exposure if total_exposure > 0 else 0
                
                # تطبیق برای تنوع (یا عدم تنوع)
                diversification_adjustment = self._calculate_diversification_adjustment(position_risks)
                
                overall_risk = weighted_risk * diversification_adjustment
            else:
                overall_risk = 0
            
            # محاسبه متریک‌های اضافی
            correlation_matrix = self._build_correlation_matrix(market_data)
            portfolio_var = self._calculate_portfolio_var(portfolio_data, market_data, correlation_matrix)
            
            # تعیین سطح ریسک کلی
            overall_risk_level = self._determine_risk_level(overall_risk)
            
            # تولید توصیه‌ها
            recommendations = self._generate_portfolio_recommendations(
                overall_risk, total_exposure, position_risks
            )
            
            result = {
                'overall_risk_score': overall_risk,
                'overall_risk_level': overall_risk_level,
                'total_exposure': total_exposure,
                'position_count': len(active_positions),
                'individual_position_risks': position_risks,
                'portfolio_var_95': portfolio_var['var_95'],
                'portfolio_var_99': portfolio_var['var_99'],
                'correlation_risk': self._assess_correlation_risk(correlation_matrix, position_risks),
                'diversification_score': 1 / diversification_adjustment if diversification_adjustment > 0 else 1,
                'recommendations': recommendations,
                'risk_budget_utilization': total_exposure / self.risk_params['max_total_exposure'],
                'timestamp': datetime.datetime.now()
            }
            
            # ذخیره در پایگاه داده
            self._save_portfolio_risk_assessment(result)
            
            return result

        except Exception as e:
            print(f"خطا در محاسبه ریسک پورتفولیو: {e}")
            return {
                'overall_risk_score': 50,
                'overall_risk_level': RiskLevel.MEDIUM,
                'recommendations': ['Unable to calculate portfolio risk']
            }

    def _calculate_diversification_adjustment(self, position_risks: List[Dict]) -> float:
        """محاسبه تطبیق تنوع"""
        try:
            if len(position_risks) <= 1:
                return 1.2  # جریمه عدم تنوع
            
            # محاسبه ضریب تنوع
            unique_symbols = len(set(pos['symbol'] for pos in position_risks))
            
            if unique_symbols == 1:
                return 1.3  # همه در یک نماد
            elif unique_symbols == 2:
                return 1.1  # دو نماد
            else:
                return 0.9  # تنوع خوب
            
        except Exception as e:
            print(f"خطا در محاسبه تطبیق تنوع: {e}")
            return 1.0

    def _build_correlation_matrix(self, market_data: Dict) -> np.ndarray:
        """ساخت ماتریس همبستگی"""
        try:
            symbols = list(market_data.keys())
            n = len(symbols)
            
            if n < 2:
                return np.array([[1.0]])
            
            correlation_matrix = np.eye(n)
            
            for i in range(n):
                for j in range(i+1, n):
                    correlation = self._calculate_pairwise_correlation(
                        symbols[i], symbols[j], market_data
                    )
                    correlation_matrix[i, j] = correlation
                    correlation_matrix[j, i] = correlation
            
            return correlation_matrix
            
        except Exception as e:
            print(f"خطا در ساخت ماتریس همبستگی: {e}")
            return np.eye(len(market_data))

    def _calculate_portfolio_var(self, portfolio_data: Dict, market_data: Dict,
                               correlation_matrix: np.ndarray) -> Dict:
        """محاسبه VaR پورتفولیو"""
        try:
            active_positions = portfolio_data.get('active_positions', {})
            
            if not active_positions:
                return {'var_95': 0, 'var_99': 0}
            
            # وزن‌ها و نوسانات
            weights = []
            volatilities = []
            
            for position in active_positions.values():
                symbol = position.get('symbol', '')
                position_size = position.get('position_size_pct', 0)
                leverage = position.get('leverage', 1)
                weight = position_size * leverage
                
                if symbol in market_data:
                    atr = market_data[symbol].get('atr', 0)
                    price = market_data[symbol].get('close', 1)
                    volatility = atr / price if price > 0 else 0.02
                else:
                    volatility = 0.02
                
                weights.append(weight)
                volatilities.append(volatility)
            
            if not weights:
                return {'var_95': 0, 'var_99': 0}
            
            # تبدیل به numpy arrays
            w = np.array(weights)
            vol = np.array(volatilities)
            
            # محاسبه واریانس پورتفولیو
            # σ²_p = w^T * Σ * w که Σ = D * C * D (D: diagonal volatilities, C: correlation matrix)
            if len(vol) == len(correlation_matrix):
                covariance_matrix = np.outer(vol, vol) * correlation_matrix
                portfolio_variance = np.dot(w, np.dot(covariance_matrix, w))
                portfolio_volatility = np.sqrt(portfolio_variance)
            else:
                # fallback
                portfolio_volatility = np.sqrt(np.dot(w**2, vol**2))
            
            # VaR با فرض توزیع نرمال
            var_95 = 1.645 * portfolio_volatility
            var_99 = 2.326 * portfolio_volatility
            
            return {
                'var_95': float(var_95),
                'var_99': float(var_99),
                'portfolio_volatility': float(portfolio_volatility)
            }
            
        except Exception as e:
            print(f"خطا در محاسبه VaR پورتفولیو: {e}")
            return {'var_95': 0.02, 'var_99': 0.03}

    def _assess_correlation_risk(self, correlation_matrix: np.ndarray, position_risks: List[Dict]) -> float:
        """ارزیابی ریسک همبستگی پورتفولیو"""
        try:
            if correlation_matrix.shape[0] < 2:
                return 0
            
            # میانگین همبستگی‌های غیرقطری
            upper_triangle = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
            avg_correlation = np.mean(np.abs(upper_triangle))
            
            # امتیاز ریسک بر اساس میانگین همبستگی
            if avg_correlation < 0.3:
                return 20  # همبستگی پایین - ریسک پایین
            elif avg_correlation < 0.5:
                return 40  # همبستگی متوسط
            elif avg_correlation < 0.7:
                return 70  # همبستگی بالا
            else:
                return 90  # همبستگی خیلی بالا - ریسک بالا
            
        except Exception as e:
            print(f"خطا در ارزیابی ریسک همبستگی: {e}")
            return 50

    def _generate_portfolio_recommendations(self, overall_risk: float, total_exposure: float,
                                          position_risks: List[Dict]) -> List[str]:
        """تولید توصیه‌های پورتفولیو"""
        recommendations = []
        
        # توصیه‌های ریسک کلی
        if overall_risk < 30:
            recommendations.append("Portfolio risk is within acceptable limits")
        elif overall_risk < 50:
            recommendations.append("Consider monitoring portfolio risk more closely")
        elif overall_risk < 70:
            recommendations.append("Portfolio risk is elevated - consider reducing positions")
        else:
            recommendations.append("Portfolio risk is too high - immediate action recommended")
        
        # توصیه‌های exposure
        max_exposure = self.risk_params['max_total_exposure']
        if total_exposure > max_exposure:
            recommendations.append(f"Total exposure ({total_exposure:.1%}) exceeds limit ({max_exposure:.1%})")
        
        # توصیه‌های تنوع
        unique_symbols = len(set(pos['symbol'] for pos in position_risks))
        if unique_symbols < 2:
            recommendations.append("Consider diversifying across multiple symbols")
        
        # توصیه‌های پوزیشن‌های پرریسک
        high_risk_positions = [pos for pos in position_risks if pos['risk_score'] > 70]
        if high_risk_positions:
            symbols = [pos['symbol'] for pos in high_risk_positions]
            recommendations.append(f"High risk positions detected: {', '.join(symbols)}")
        
        return recommendations

    def _save_risk_calculation(self, risk_result: Dict):
        """ذخیره محاسبات ریسک"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO risk_calculations (
                    symbol, calculation_type, input_data, risk_score, risk_level,
                    recommended_position_size, recommended_leverage, max_loss_amount,
                    var_95, var_99, expected_shortfall, correlation_risk,
                    liquidity_risk, volatility_risk, concentration_risk, total_portfolio_risk
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                risk_result['symbol'],
                'POSITION_RISK',
                json.dumps({k: v for k, v in risk_result.items() if k not in ['timestamp']}),
                risk_result['total_risk_score'],
                risk_result['risk_level'].value,
                risk_result['recommendations']['recommended_position_size'],
                risk_result['recommendations']['recommended_leverage'],
                risk_result['basic_risk']['max_loss_amount'],
                risk_result['var_metrics']['var_95'],
                risk_result['var_metrics']['var_99'],
                risk_result['var_metrics'].get('expected_shortfall_95', 0),
                risk_result['correlation_risk']['risk_score'],
                risk_result['liquidity_risk']['risk_score'],
                risk_result['volatility_risk']['risk_score'],
                risk_result['concentration_risk']['risk_score'],
                0  # will be updated by portfolio calculation
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"خطا در ذخیره محاسبات ریسک: {e}")

    def _save_portfolio_risk_assessment(self, portfolio_result: Dict):
        """ذخیره ارزیابی ریسک پورتفولیو"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO portfolio_risk_assessment (
                    total_exposure, diversification_ratio, correlation_risk_score,
                    concentration_risk_score, var_portfolio_95, var_portfolio_99,
                    risk_budget_utilization, overall_risk_score, overall_risk_level,
                    recommendations
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                portfolio_result['total_exposure'],
                portfolio_result['diversification_score'],
                portfolio_result['correlation_risk'],
                0,  # concentration will be calculated separately
                portfolio_result['portfolio_var_95'],
                portfolio_result['portfolio_var_99'],
                portfolio_result['risk_budget_utilization'],
                portfolio_result['overall_risk_score'],
                portfolio_result['overall_risk_level'].value,
                json.dumps(portfolio_result['recommendations'])
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"خطا در ذخیره ارزیابی پورتفولیو: {e}")

    def _get_volatility_percentile(self, symbol: str, current_volatility: float) -> float:
        """محاسبه پرسنتایل نوسانات فعلی"""
        try:
            # این تابع باید از داده‌های تاریخی استفاده کند
            # برای سادگی، فرض می‌کنیم:
            typical_volatilities = {
                'BTC/USDT': 0.04,
                'ETH/USDT': 0.05,
                'SOL/USDT': 0.07
            }
            
            typical_vol = typical_volatilities.get(symbol, 0.05)
            
            if current_volatility < typical_vol * 0.5:
                return 10  # خیلی پایین
            elif current_volatility < typical_vol * 0.8:
                return 30  # پایین
            elif current_volatility < typical_vol * 1.2:
                return 50  # متوسط
            elif current_volatility < typical_vol * 1.5:
                return 70  # بالا
            else:
                return 90  # خیلی بالا
                
        except Exception as e:
            print(f"خطا در محاسبه پرسنتایل نوسانات: {e}")
            return 50

    def get_risk_summary(self, portfolio_data: Dict, market_data: Dict) -> Dict:
        """خلاصه جامع ریسک"""
        try:
            portfolio_risk = self.calculate_portfolio_risk(portfolio_data, market_data)
            
            return {
                'timestamp': datetime.datetime.now(),
                'overall_assessment': {
                    'risk_score': portfolio_risk['overall_risk_score'],
                    'risk_level': portfolio_risk['overall_risk_level'].value,
                    'total_exposure': portfolio_risk['total_exposure'],
                    'positions_count': portfolio_risk['position_count']
                },
                'risk_metrics': {
                    'var_95': portfolio_risk['portfolio_var_95'],
                    'var_99': portfolio_risk['portfolio_var_99'],
                    'correlation_risk': portfolio_risk['correlation_risk'],
                    'diversification_score': portfolio_risk['diversification_score']
                },
                'budget_utilization': {
                    'exposure_used': portfolio_risk['risk_budget_utilization'],
                    'exposure_remaining': 1 - portfolio_risk['risk_budget_utilization'],
                    'positions_used': portfolio_risk['position_count'],
                    'positions_remaining': self.risk_params['max_concurrent_positions'] - portfolio_risk['position_count']
                },
                'recommendations': portfolio_risk['recommendations'],
                'action_required': portfolio_risk['overall_risk_score'] > 70
            }
            
        except Exception as e:
            print(f"خطا در تولید خلاصه ریسک: {e}")
            return {
                'overall_assessment': {'risk_level': 'UNKNOWN'},
                'action_required': True
            }
