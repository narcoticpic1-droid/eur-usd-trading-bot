"""
Risk Analyzer - تحلیلگر ریسک پیشرفته
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import math

class RiskAnalyzer:
    """تحلیلگر ریسک برای معاملات ارزهای دیجیتال"""
    
    def __init__(self, risk_config: Dict):
        self.config = risk_config
        self.risk_metrics = {}
        
    def assess_symbol_risk(self, symbol: str, data: pd.DataFrame, 
                          signal_data: Dict) -> Dict:
        """ارزیابی جامع ریسک برای یک نماد"""
        
        # محاسبه ریسک‌های مختلف
        volatility_risk = self._calculate_volatility_risk(data)
        liquidity_risk = self._calculate_liquidity_risk(data)
        drawdown_risk = self._calculate_drawdown_risk(data)
        correlation_risk = self._calculate_correlation_risk(symbol)
        
        # ریسک مخصوص سیگنال
        signal_risk = self._calculate_signal_risk(signal_data)
        
        # ریسک کلی
        overall_risk = self._calculate_overall_risk_score([
            volatility_risk, liquidity_risk, drawdown_risk, 
            correlation_risk, signal_risk
        ])
        
        # توصیه اندازه پوزیشن
        position_size = self._calculate_optimal_position_size(
            symbol, overall_risk, signal_data
        )
        
        return {
            'symbol': symbol,
            'overall_risk_score': overall_risk,
            'risk_level': self._categorize_risk_level(overall_risk),
            'volatility_risk': volatility_risk,
            'liquidity_risk': liquidity_risk,
            'drawdown_risk': drawdown_risk,
            'correlation_risk': correlation_risk,
            'signal_risk': signal_risk,
            'recommended_position_size': position_size,
            'max_leverage_recommended': self._calculate_max_leverage(overall_risk),
            'stop_loss_distance': self._calculate_optimal_stop_loss(data, overall_risk),
            'risk_assessment_timestamp': datetime.now()
        }
    
    def _calculate_volatility_risk(self, data: pd.DataFrame) -> Dict:
        """محاسبه ریسک نوسانات"""
        # محاسبه انحراف معیار بازدهی
        returns = data['close'].pct_change().dropna()
        volatility = returns.std() * math.sqrt(24)  # روزانه برای timeframe ساعتی
        
        # محاسبه VaR (Value at Risk)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # تحلیل نوسانات اخیر
        recent_volatility = returns.iloc[-24:].std() * math.sqrt(24)
        volatility_trend = "INCREASING" if recent_volatility > volatility else "DECREASING"
        
        # امتیاز ریسک (0-100)
        risk_score = min(volatility * 1000, 100)
        
        return {
            'volatility': volatility,
            'recent_volatility': recent_volatility,
            'volatility_trend': volatility_trend,
            'var_95': var_95,
            'var_99': var_99,
            'risk_score': risk_score
        }
    
    def _calculate_liquidity_risk(self, data: pd.DataFrame) -> Dict:
        """محاسبه ریسک نقدینگی"""
        # تحلیل حجم معاملات
        volume_ma = data['volume'].rolling(24).mean()
        current_volume = data['volume'].iloc[-1]
        
        # ریسک نقدینگی بر اساس حجم
        liquidity_ratio = current_volume / volume_ma.iloc[-1] if volume_ma.iloc[-1] > 0 else 0
        
        # امتیاز ریسک
        if liquidity_ratio > 1.5:
            risk_score = 20
        elif liquidity_ratio > 1.0:
            risk_score = 40
        elif liquidity_ratio > 0.7:
            risk_score = 60
        else:
            risk_score = 80
        
        return {
            'current_volume': current_volume,
            'average_volume': volume_ma.iloc[-1],
            'liquidity_ratio': liquidity_ratio,
            'risk_score': risk_score
        }
    
    def _calculate_drawdown_risk(self, data: pd.DataFrame) -> Dict:
        """محاسبه ریسک افت قیمت"""
        # محاسبه maximum drawdown
        cumulative = (1 + data['close'].pct_change()).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        
        max_drawdown = drawdown.min()
        current_drawdown = drawdown.iloc[-1]
        
        # تحلیل مدت زمان recovery
        drawdown_periods = []
        in_drawdown = False
        start_dd = None
        
        for i, dd in enumerate(drawdown):
            if dd < -0.05 and not in_drawdown:  # شروع drawdown بیش از 5%
                in_drawdown = True
                start_dd = i
            elif dd >= -0.01 and in_drawdown:  # پایان drawdown
                in_drawdown = False
                if start_dd is not None:
                    drawdown_periods.append(i - start_dd)
        
        avg_recovery_time = np.mean(drawdown_periods) if drawdown_periods else 0
        
        # امتیاز ریسک
        risk_score = min(abs(max_drawdown) * 200, 100)
        
        return {
            'max_drawdown': max_drawdown,
            'current_drawdown': current_drawdown,
            'average_recovery_time': avg_recovery_time,
            'risk_score': risk_score
        }
    
    def _calculate_correlation_risk(self, symbol: str) -> Dict:
        """محاسبه ریسک همبستگی"""
        # این قسمت در correlation_analyzer پیاده‌سازی خواهد شد
        # فعلاً مقدار پیش‌فرض
        return {
            'btc_correlation': 0.7,  # همبستگی با بیت‌کوین
            'market_correlation': 0.6,  # همبستگی با کل بازار
            'risk_score': 50
        }
    
    def _calculate_signal_risk(self, signal_data: Dict) -> Dict:
        """محاسبه ریسک مخصوص سیگنال"""
        if not signal_data.get('has_signal', False):
            return {'risk_score': 0}
        
        confidence = signal_data.get('confidence', 0)
        quality = signal_data.get('signal_quality', 'LOW')
        risk_reward = signal_data.get('risk_reward_ratio', 1)
        
        # امتیاز ریسک بر اساس کیفیت سیگنال
        quality_scores = {'HIGH': 20, 'MEDIUM': 50, 'LOW': 80}
        quality_risk = quality_scores.get(quality, 80)
        
        # ریسک بر اساس confidence
        confidence_risk = (1 - confidence) * 100
        
        # ریسک بر اساس risk/reward
        rr_risk = max(100 - (risk_reward * 25), 0)
        
        overall_signal_risk = (quality_risk + confidence_risk + rr_risk) / 3
        
        return {
            'quality_risk': quality_risk,
            'confidence_risk': confidence_risk,
            'risk_reward_risk': rr_risk,
            'risk_score': overall_signal_risk
        }
    
    def _calculate_overall_risk_score(self, risk_components: List[Dict]) -> float:
        """محاسبه امتیاز کلی ریسک"""
        weights = [0.3, 0.2, 0.2, 0.15, 0.15]  # وزن هر کامپوننت ریسک
        
        total_score = 0
        for i, component in enumerate(risk_components):
            if i < len(weights):
                total_score += component.get('risk_score', 0) * weights[i]
        
        return min(total_score, 100)
    
    def _categorize_risk_level(self, risk_score: float) -> str:
        """دسته‌بندی سطح ریسک"""
        if risk_score <= 25:
            return "LOW"
        elif risk_score <= 50:
            return "MODERATE"
        elif risk_score <= 75:
            return "HIGH"
        else:
            return "VERY_HIGH"
    
    def _calculate_optimal_position_size(self, symbol: str, risk_score: float, 
                                       signal_data: Dict) -> float:
        """محاسبه اندازه بهینه پوزیشن"""
        base_size = self.config.get('POSITION_SIZING', {}).get(symbol, {}).get('base_size', 0.01)
        
        # تنظیم بر اساس ریسک
        risk_multiplier = max(0.3, 1 - (risk_score / 100))
        
        # تنظیم بر اساس کیفیت سیگنال
        quality = signal_data.get('signal_quality', 'LOW')
        quality_multipliers = {'HIGH': 1.2, 'MEDIUM': 1.0, 'LOW': 0.7}
        quality_multiplier = quality_multipliers.get(quality, 0.7)
        
        optimal_size = base_size * risk_multiplier * quality_multiplier
        max_size = self.config.get('POSITION_SIZING', {}).get(symbol, {}).get('max_size', 0.02)
        
        return min(optimal_size, max_size)
    
    def _calculate_max_leverage(self, risk_score: float) -> int:
        """محاسبه حداکثر اهرم توصیه شده"""
        if risk_score <= 25:
            return 10
        elif risk_score <= 50:
            return 6
        elif risk_score <= 75:
            return 3
        else:
            return 2
    
    def _calculate_optimal_stop_loss(self, data: pd.DataFrame, risk_score: float) -> float:
        """محاسبه فاصله بهینه stop loss"""
        atr = self._calculate_atr(data)
        
        # تنظیم بر اساس ریسک
        if risk_score <= 25:
            multiplier = 1.5
        elif risk_score <= 50:
            multiplier = 2.0
        elif risk_score <= 75:
            multiplier = 2.5
        else:
            multiplier = 3.0
        
        return atr * multiplier
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """محاسبه Average True Range"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr = np.maximum(high - low,
                       np.maximum(abs(high - close.shift(1)),
                                 abs(low - close.shift(1))))
        
        return tr.rolling(period).mean().iloc[-1]
