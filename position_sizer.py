import pandas as pd
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from enum import Enum
import config

class SizingMethod(Enum):
    """روش‌های محاسبه اندازه پوزیشن"""
    FIXED_FRACTIONAL = "FIXED_FRACTIONAL"
    KELLY_CRITERION = "KELLY_CRITERION"
    VOLATILITY_BASED = "VOLATILITY_BASED"
    RISK_PARITY = "RISK_PARITY"
    ATR_BASED = "ATR_BASED"
    ADAPTIVE = "ADAPTIVE"

class PositionSizer:
    """
    کلاس محاسبه اندازه پوزیشن با روش‌های مختلف
    """

    def __init__(self):
        self.name = "Position Sizer"
        
        # پارامترهای پیش‌فرض
        self.default_params = {
            'account_balance': 10000,  # مبلغ پیش‌فرض حساب
            'max_risk_per_trade': 0.01,  # 1% ریسک هر معامله
            'kelly_lookback_period': 30,  # دوره بررسی برای Kelly
            'volatility_window': 20,  # پنجره محاسبه نوسانات
            'atr_multiplier': 2.0,  # ضریب ATR برای stop loss
            'min_position_size': 10,  # حداقل اندازه پوزیشن (دلار)
            'max_position_size_ratio': 0.2  # حداکثر 20% حساب در یک معامله
        }
        
        # نگهداری تاریخچه عملکرد برای Kelly Criterion
        self.performance_history = {
            'BTC/USDT': [],
            'ETH/USDT': [],
            'SOL/USDT': []
        }

    def calculate_position_size(self, 
                              signal_data: Dict,
                              method: SizingMethod = SizingMethod.ADAPTIVE,
                              account_balance: float = None,
                              custom_params: Dict = None) -> Dict:
        """
        محاسبه اندازه پوزیشن با روش مشخص شده
        """
        try:
            # تنظیم پارامترها
            balance = account_balance or self.default_params['account_balance']
            params = self.default_params.copy()
            if custom_params:
                params.update(custom_params)
            
            symbol = signal_data.get('symbol', '')
            entry_price = signal_data.get('current_price', 0)
            stop_loss = signal_data.get('stop_loss', 0)
            signal_quality = signal_data.get('signal_quality', 'MEDIUM')
            confidence = signal_data.get('confidence', 0.7)
            
            if entry_price <= 0 or stop_loss <= 0:
                return self._get_error_result(symbol, "Invalid price data")
            
            # انتخاب روش محاسبه
            if method == SizingMethod.FIXED_FRACTIONAL:
                result = self._fixed_fractional_sizing(signal_data, balance, params)
            elif method == SizingMethod.KELLY_CRITERION:
                result = self._kelly_criterion_sizing(signal_data, balance, params)
            elif method == SizingMethod.VOLATILITY_BASED:
                result = self._volatility_based_sizing(signal_data, balance, params)
            elif method == SizingMethod.RISK_PARITY:
                result = self._risk_parity_sizing(signal_data, balance, params)
            elif method == SizingMethod.ATR_BASED:
                result = self._atr_based_sizing(signal_data, balance, params)
            elif method == SizingMethod.ADAPTIVE:
                result = self._adaptive_sizing(signal_data, balance, params)
            else:
                result = self._fixed_fractional_sizing(signal_data, balance, params)
            
            # اعمال محدودیت‌های نهایی
            result = self._apply_final_constraints(result, balance, signal_data)
            
            return result
            
        except Exception as e:
            return self._get_error_result(signal_data.get('symbol', ''), f"Calculation error: {e}")

    def _fixed_fractional_sizing(self, signal_data: Dict, balance: float, params: Dict) -> Dict:
        """روش Fixed Fractional - درصد ثابت از حساب"""
        symbol = signal_data.get('symbol', '')
        entry_price = signal_data.get('current_price', 0)
        stop_loss = signal_data.get('stop_loss', 0)
        
        # محاسبه ریسک
        if signal_data.get('signal', 0) > 0:  # Long
            risk_per_share = abs(entry_price - stop_loss)
        else:  # Short
            risk_per_share = abs(stop_loss - entry_price)
        
        risk_percentage = risk_per_share / entry_price
        
        # مبلغ ریسک مجاز
        risk_amount = balance * params['max_risk_per_trade']
        
        # محاسبه اندازه پوزیشن
        position_value = risk_amount / risk_percentage if risk_percentage > 0 else 0
        position_size_crypto = position_value / entry_price
        
        # محاسبه اهرم مورد نیاز
        symbol_config = config.POSITION_SIZING.get(symbol, {'base_size': 0.01})
        base_position = balance * symbol_config['base_size']
        required_leverage = min(position_value / base_position, config.SYMBOL_MAX_LEVERAGE.get(symbol, config.MAX_LEVERAGE))
        required_leverage = max(required_leverage, config.MIN_LEVERAGE)
        
        return {
            'method': 'FIXED_FRACTIONAL',
            'symbol': symbol,
            'position_size_usd': position_value,
            'position_size_crypto': position_size_crypto,
            'recommended_leverage': round(required_leverage, 1),
            'risk_amount': risk_amount,
            'risk_percentage': risk_percentage * 100,
            'reasoning': f"Fixed {params['max_risk_per_trade']*100}% risk per trade",
            'confidence_multiplier': 1.0,
            'status': 'CALCULATED'
        }

    def _kelly_criterion_sizing(self, signal_data: Dict, balance: float, params: Dict) -> Dict:
        """Kelly Criterion - بر اساس احتمال برد و میانگین سود/ضرر"""
        symbol = signal_data.get('symbol', '')
        
        # دریافت تاریخچه عملکرد
        history = self.performance_history.get(symbol, [])
        
        if len(history) < 10:
            # اگر تاریخچه کافی نداریم، از fixed fractional استفاده کن
            base_result = self._fixed_fractional_sizing(signal_data, balance, params)
            base_result['method'] = 'KELLY_CRITERION (FALLBACK)'
            base_result['reasoning'] = "Insufficient history, using fixed fractional"
            return base_result
        
        # محاسبه پارامترهای Kelly
        wins = [p for p in history if p > 0]
        losses = [p for p in history if p < 0]
        
        win_rate = len(wins) / len(history)
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 0
        
        # Kelly formula: f = (bp - q) / b
        # b = average win / average loss
        # p = win probability
        # q = loss probability (1-p)
        
        if avg_loss > 0:
            b = avg_win / avg_loss
            kelly_fraction = (b * win_rate - (1 - win_rate)) / b
        else:
            kelly_fraction = 0
        
        # محدود کردن Kelly fraction (معمولاً نباید بیش از 25% باشد)
        kelly_fraction = max(0, min(kelly_fraction, 0.25))
        
        # تعدیل بر اساس کیفیت سیگنال
        confidence = signal_data.get('confidence', 0.7)
        signal_quality = signal_data.get('signal_quality', 'MEDIUM')
        
        quality_multiplier = {
            'HIGH': 1.0,
            'MEDIUM': 0.7,
            'LOW': 0.5
        }.get(signal_quality, 0.7)
        
        adjusted_kelly = kelly_fraction * quality_multiplier * confidence
        
        # محاسبه اندازه پوزیشن
        position_value = balance * adjusted_kelly
        entry_price = signal_data.get('current_price', 0)
        position_size_crypto = position_value / entry_price if entry_price > 0 else 0
        
        # محاسبه اهرم
        symbol_config = config.POSITION_SIZING.get(symbol, {'base_size': 0.01})
        base_position = balance * symbol_config['base_size']
        required_leverage = min(position_value / base_position, config.SYMBOL_MAX_LEVERAGE.get(symbol, config.MAX_LEVERAGE))
        required_leverage = max(required_leverage, config.MIN_LEVERAGE)
        
        return {
            'method': 'KELLY_CRITERION',
            'symbol': symbol,
            'position_size_usd': position_value,
            'position_size_crypto': position_size_crypto,
            'recommended_leverage': round(required_leverage, 1),
            'kelly_fraction': kelly_fraction,
            'adjusted_kelly': adjusted_kelly,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'samples_used': len(history),
            'reasoning': f"Kelly: {kelly_fraction:.3f}, Adjusted: {adjusted_kelly:.3f}",
            'confidence_multiplier': quality_multiplier,
            'status': 'CALCULATED'
        }

    def _volatility_based_sizing(self, signal_data: Dict, balance: float, params: Dict) -> Dict:
        """محاسبه بر اساس نوسانات بازار"""
        symbol = signal_data.get('symbol', '')
        
        # دریافت داده‌های قیمت برای محاسبه نوسانات
        market_context = signal_data.get('market_context', {})
        atr = market_context.get('atr', 0)
        current_price = signal_data.get('current_price', 0)
        
        if atr <= 0 or current_price <= 0:
            # fallback به fixed fractional
            base_result = self._fixed_fractional_sizing(signal_data, balance, params)
            base_result['method'] = 'VOLATILITY_BASED (FALLBACK)'
            return base_result
        
        # محاسبه نوسانات روزانه
        daily_volatility = atr / current_price
        
        # تعیین اندازه پوزیشن معکوس با نوسانات
        # نوسانات بالا = پوزیشن کوچک‌تر
        base_risk = params['max_risk_per_trade']
        volatility_adjusted_risk = base_risk / (1 + daily_volatility * 10)  # تنظیم بر اساس نوسانات
        
        # محدود کردن در بازه مناسب
        volatility_adjusted_risk = max(volatility_adjusted_risk, base_risk * 0.3)  # حداقل 30% ریسک پایه
        volatility_adjusted_risk = min(volatility_adjusted_risk, base_risk * 1.5)  # حداکثر 150% ریسک پایه
        
        # محاسبه اندازه پوزیشن
        risk_amount = balance * volatility_adjusted_risk
        
        # فاصله تا stop loss
        stop_loss = signal_data.get('stop_loss', 0)
        if signal_data.get('signal', 0) > 0:  # Long
            risk_per_share = abs(current_price - stop_loss)
        else:  # Short
            risk_per_share = abs(stop_loss - current_price)
        
        risk_percentage = risk_per_share / current_price if current_price > 0 else 0
        
        if risk_percentage > 0:
            position_value = risk_amount / risk_percentage
            position_size_crypto = position_value / current_price
        else:
            position_value = 0
            position_size_crypto = 0
        
        # محاسبه اهرم
        symbol_config = config.POSITION_SIZING.get(symbol, {'base_size': 0.01})
        base_position = balance * symbol_config['base_size']
        required_leverage = min(position_value / base_position, config.SYMBOL_MAX_LEVERAGE.get(symbol, config.MAX_LEVERAGE))
        required_leverage = max(required_leverage, config.MIN_LEVERAGE)
        
        return {
            'method': 'VOLATILITY_BASED',
            'symbol': symbol,
            'position_size_usd': position_value,
            'position_size_crypto': position_size_crypto,
            'recommended_leverage': round(required_leverage, 1),
            'risk_amount': risk_amount,
            'daily_volatility': daily_volatility * 100,
            'volatility_adjustment': volatility_adjusted_risk / base_risk,
            'atr_used': atr,
            'reasoning': f"Volatility-adjusted risk: {volatility_adjusted_risk*100:.2f}%",
            'status': 'CALCULATED'
        }

    def _atr_based_sizing(self, signal_data: Dict, balance: float, params: Dict) -> Dict:
        """محاسبه بر اساس ATR برای تعیین stop loss"""
        symbol = signal_data.get('symbol', '')
        current_price = signal_data.get('current_price', 0)
        market_context = signal_data.get('market_context', {})
        atr = market_context.get('atr', 0)
        
        if atr <= 0 or current_price <= 0:
            base_result = self._fixed_fractional_sizing(signal_data, balance, params)
            base_result['method'] = 'ATR_BASED (FALLBACK)'
            return base_result
        
        # محاسبه stop loss بر اساس ATR
        atr_multiplier = params.get('atr_multiplier', 2.0)
        signal_direction = signal_data.get('signal', 0)
        
        if signal_direction > 0:  # Long
            calculated_stop_loss = current_price - (atr * atr_multiplier)
        else:  # Short
            calculated_stop_loss = current_price + (atr * atr_multiplier)
        
        # محاسبه ریسک بر اساس ATR stop loss
        risk_per_share = atr * atr_multiplier
        risk_percentage = risk_per_share / current_price
        
        # مبلغ ریسک مجاز
        risk_amount = balance * params['max_risk_per_trade']
        
        # محاسبه اندازه پوزیشن
        position_value = risk_amount / risk_percentage if risk_percentage > 0 else 0
        position_size_crypto = position_value / current_price
        
        # محاسبه اهرم
        symbol_config = config.POSITION_SIZING.get(symbol, {'base_size': 0.01})
        base_position = balance * symbol_config['base_size']
        required_leverage = min(position_value / base_position, config.SYMBOL_MAX_LEVERAGE.get(symbol, config.MAX_LEVERAGE))
        required_leverage = max(required_leverage, config.MIN_LEVERAGE)
        
        return {
            'method': 'ATR_BASED',
            'symbol': symbol,
            'position_size_usd': position_value,
            'position_size_crypto': position_size_crypto,
            'recommended_leverage': round(required_leverage, 1),
            'risk_amount': risk_amount,
            'atr_stop_loss': calculated_stop_loss,
            'atr_value': atr,
            'atr_multiplier': atr_multiplier,
            'risk_percentage': risk_percentage * 100,
            'reasoning': f"ATR-based stop at {calculated_stop_loss:.4f}",
            'status': 'CALCULATED'
        }

    def _risk_parity_sizing(self, signal_data: Dict, balance: float, params: Dict) -> Dict:
        """محاسبه بر اساس تساوی ریسک بین دارایی‌ها"""
        symbol = signal_data.get('symbol', '')
        
        # فرض: سه نماد اصلی BTC, ETH, SOL
        total_symbols = len(config.SYMBOLS_TO_MONITOR)
        risk_per_symbol = params['max_risk_per_trade'] / total_symbols
        
        # تعدیل بر اساس نوسانات نسبی
        symbol_volatility_weights = {
            'BTC/USDT': 0.8,   # کم‌نوسان
            'ETH/USDT': 1.0,   # متوسط
            'SOL/USDT': 1.3    # پرنوسان
        }
        
        weight = symbol_volatility_weights.get(symbol, 1.0)
        adjusted_risk = risk_per_symbol / weight  # نوسانات بالا = ریسک کمتر
        
        # محاسبه اندازه پوزیشن
        current_price = signal_data.get('current_price', 0)
        stop_loss = signal_data.get('stop_loss', 0)
        
        if signal_data.get('signal', 0) > 0:  # Long
            risk_per_share = abs(current_price - stop_loss)
        else:  # Short
            risk_per_share = abs(stop_loss - current_price)
        
        risk_percentage = risk_per_share / current_price if current_price > 0 else 0
        risk_amount = balance * adjusted_risk
        
        if risk_percentage > 0:
            position_value = risk_amount / risk_percentage
            position_size_crypto = position_value / current_price
        else:
            position_value = 0
            position_size_crypto = 0
        
        # محاسبه اهرم
        symbol_config = config.POSITION_SIZING.get(symbol, {'base_size': 0.01})
        base_position = balance * symbol_config['base_size']
        required_leverage = min(position_value / base_position, config.SYMBOL_MAX_LEVERAGE.get(symbol, config.MAX_LEVERAGE))
        required_leverage = max(required_leverage, config.MIN_LEVERAGE)
        
        return {
            'method': 'RISK_PARITY',
            'symbol': symbol,
            'position_size_usd': position_value,
            'position_size_crypto': position_size_crypto,
            'recommended_leverage': round(required_leverage, 1),
            'risk_amount': risk_amount,
            'risk_per_symbol': risk_per_symbol * 100,
            'volatility_weight': weight,
            'adjusted_risk': adjusted_risk * 100,
            'reasoning': f"Equal risk across {total_symbols} symbols, volatility-adjusted",
            'status': 'CALCULATED'
        }

    def _adaptive_sizing(self, signal_data: Dict, balance: float, params: Dict) -> Dict:
        """روش تطبیقی - ترکیب چند روش"""
        symbol = signal_data.get('symbol', '')
        signal_quality = signal_data.get('signal_quality', 'MEDIUM')
        confidence = signal_data.get('confidence', 0.7)
        
        # محاسبه با چند روش
        fixed_result = self._fixed_fractional_sizing(signal_data, balance, params)
        volatility_result = self._volatility_based_sizing(signal_data, balance, params)
        
        # اگر تاریخچه کافی داریم، Kelly هم اضافه کن
        history = self.performance_history.get(symbol, [])
        if len(history) >= 10:
            kelly_result = self._kelly_criterion_sizing(signal_data, balance, params)
            
            # میانگین وزنی از سه روش
            weights = {
                'fixed': 0.3,
                'volatility': 0.3,
                'kelly': 0.4
            }
            
            weighted_position_value = (
                fixed_result['position_size_usd'] * weights['fixed'] +
                volatility_result['position_size_usd'] * weights['volatility'] +
                kelly_result['position_size_usd'] * weights['kelly']
            )
        else:
            # فقط دو روش اول
            weights = {'fixed': 0.5, 'volatility': 0.5}
            
            weighted_position_value = (
                fixed_result['position_size_usd'] * weights['fixed'] +
                volatility_result['position_size_usd'] * weights['volatility']
            )
        
        # تعدیل بر اساس کیفیت سیگنال
        quality_multipliers = {
            'HIGH': 1.2,
            'MEDIUM': 1.0,
            'LOW': 0.7
        }
        
        quality_multiplier = quality_multipliers.get(signal_quality, 1.0)
        confidence_multiplier = 0.5 + (confidence * 0.5)  # بین 0.5 تا 1.0
        
        final_position_value = weighted_position_value * quality_multiplier * confidence_multiplier
        
        # محاسبه مقادیر نهایی
        current_price = signal_data.get('current_price', 0)
        position_size_crypto = final_position_value / current_price if current_price > 0 else 0
        
        # محاسبه اهرم
        symbol_config = config.POSITION_SIZING.get(symbol, {'base_size': 0.01})
        base_position = balance * symbol_config['base_size']
        required_leverage = min(final_position_value / base_position, config.SYMBOL_MAX_LEVERAGE.get(symbol, config.MAX_LEVERAGE))
        required_leverage = max(required_leverage, config.MIN_LEVERAGE)
        
        # محاسبه ریسک
        stop_loss = signal_data.get('stop_loss', 0)
        if signal_data.get('signal', 0) > 0:  # Long
            risk_per_share = abs(current_price - stop_loss)
        else:  # Short
            risk_per_share = abs(stop_loss - current_price)
        
        risk_percentage = risk_per_share / current_price if current_price > 0 else 0
        risk_amount = final_position_value * risk_percentage
        
        return {
            'method': 'ADAPTIVE',
            'symbol': symbol,
            'position_size_usd': final_position_value,
            'position_size_crypto': position_size_crypto,
            'recommended_leverage': round(required_leverage, 1),
            'risk_amount': risk_amount,
            'quality_multiplier': quality_multiplier,
            'confidence_multiplier': confidence_multiplier,
            'methods_used': list(weights.keys()),
            'weights_applied': weights,
            'reasoning': f"Adaptive sizing: Quality={signal_quality}, Confidence={confidence:.2f}",
            'status': 'CALCULATED'
        }

    def _apply_final_constraints(self, result: Dict, balance: float, signal_data: Dict) -> Dict:
        """اعمال محدودیت‌های نهایی"""
        symbol = result.get('symbol', '')
        
        # حداقل و حداکثر اندازه
        min_size = self.default_params['min_position_size']
        max_size = balance * self.default_params['max_position_size_ratio']
        
        original_size = result['position_size_usd']
        constrained_size = max(min_size, min(original_size, max_size))
        
        if constrained_size != original_size:
            ratio = constrained_size / original_size if original_size > 0 else 0
            result['position_size_usd'] = constrained_size
            result['position_size_crypto'] = result['position_size_crypto'] * ratio
            result['risk_amount'] = result.get('risk_amount', 0) * ratio
            
            if constrained_size == min_size:
                result['constraint_applied'] = f"Increased to minimum size: ${min_size}"
            else:
                result['constraint_applied'] = f"Reduced to maximum size: ${max_size:.2f}"
        
        # بررسی محدودیت‌های config
        symbol_config = config.POSITION_SIZING.get(symbol, {})
        symbol_max_size = symbol_config.get('max_size', 0.02) * balance
        
        if result['position_size_usd'] > symbol_max_size:
            ratio = symbol_max_size / result['position_size_usd']
            result['position_size_usd'] = symbol_max_size
            result['position_size_crypto'] *= ratio
            result['risk_amount'] = result.get('risk_amount', 0) * ratio
            result['constraint_applied'] = f"Symbol-specific limit: ${symbol_max_size:.2f}"
        
        # بررسی اهرم
        max_leverage = config.SYMBOL_MAX_LEVERAGE.get(symbol, config.MAX_LEVERAGE)
        if result['recommended_leverage'] > max_leverage:
            result['recommended_leverage'] = max_leverage
            result['leverage_constraint'] = f"Limited to max leverage: {max_leverage}x"
        
        return result

    def _get_error_result(self, symbol: str, error_message: str) -> Dict:
        """نتیجه خطا"""
        return {
            'method': 'ERROR',
            'symbol': symbol,
            'position_size_usd': 0,
            'position_size_crypto': 0,
            'recommended_leverage': config.MIN_LEVERAGE,
            'risk_amount': 0,
            'status': 'ERROR',
            'error': error_message,
            'reasoning': f"Cannot calculate position size: {error_message}"
        }

    def update_performance_history(self, symbol: str, pnl_percentage: float):
        """به‌روزرسانی تاریخچه عملکرد برای Kelly Criterion"""
        if symbol not in self.performance_history:
            self.performance_history[symbol] = []
        
        self.performance_history[symbol].append(pnl_percentage)
        
        # نگهداری حداکثر 100 مورد اخیر
        if len(self.performance_history[symbol]) > 100:
            self.performance_history[symbol] = self.performance_history[symbol][-100:]

    def get_sizing_recommendation(self, signal_data: Dict, account_balance: float = None) -> Dict:
        """توصیه بهترین روش sizing برای سیگنال"""
        symbol = signal_data.get('symbol', '')
        signal_quality = signal_data.get('signal_quality', 'MEDIUM')
        market_context = signal_data.get('market_context', {})
        
        # بررسی در دسترس بودن داده‌ها
        has_history = len(self.performance_history.get(symbol, [])) >= 10
        has_atr = market_context.get('atr', 0) > 0
        
        recommendations = []
        
        if signal_quality == 'HIGH' and has_history:
            recommendations.append({
                'method': SizingMethod.KELLY_CRITERION,
                'reason': 'High quality signal with sufficient history',
                'priority': 1
            })
        
        if has_atr and market_context.get('trend_strength', '') in ['STRONG', 'MEDIUM']:
            recommendations.append({
                'method': SizingMethod.ATR_BASED,
                'reason': 'Strong trend with reliable ATR data',
                'priority': 2
            })
        
        if market_context.get('volume_context', '') == 'HIGH':
            recommendations.append({
                'method': SizingMethod.VOLATILITY_BASED,
                'reason': 'High volume environment',
                'priority': 3
            })
        
        recommendations.append({
            'method': SizingMethod.ADAPTIVE,
            'reason': 'Safe default combining multiple methods',
            'priority': 4
        })
        
        # محاسبه با روش‌های مختلف
        results = {}
        balance = account_balance or self.default_params['account_balance']
        
        for rec in recommendations[:2]:  # فقط دو روش اول
            try:
                result = self.calculate_position_size(signal_data, rec['method'], balance)
                results[rec['method'].value] = result
            except Exception as e:
                print(f"خطا در محاسبه {rec['method'].value}: {e}")
        
        return {
            'symbol': symbol,
            'recommendations': recommendations,
            'calculations': results,
            'best_method': recommendations[0]['method'].value if recommendations else 'ADAPTIVE'
        }

    def get_sizing_summary(self) -> Dict:
        """خلاصه وضعیت سیستم sizing"""
        return {
            'performance_history_symbols': list(self.performance_history.keys()),
            'history_lengths': {k: len(v) for k, v in self.performance_history.items()},
            'available_methods': [method.value for method in SizingMethod],
            'default_params': self.default_params,
            'kelly_ready_symbols': [k for k, v in self.performance_history.items() if len(v) >= 10]
        }
