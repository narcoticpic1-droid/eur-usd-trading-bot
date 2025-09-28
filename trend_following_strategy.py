import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from .base_strategy import BaseStrategy, StrategyType, StrategySignal
import talib

class TrendFollowingStrategy(BaseStrategy):
    """
    استراتژی دنبال کردن روند - مناسب برای ترندهای قوی
    """
    
    def __init__(self):
        super().__init__("Trend Following", StrategyType.TREND_FOLLOWING)
        
        # پارامترهای تخصصی
        self.ema_fast = 8
        self.ema_medium = 21
        self.ema_slow = 50
        self.atr_period = 14
        self.adx_period = 14
        self.min_adx = 25  # حداقل قدرت ترند
        
        # تنظیمات ریسک
        self.trail_stop_atr_multiplier = 2.0
        self.min_trend_strength = 0.6

    def analyze(self, df: pd.DataFrame, symbol: str) -> Optional[Dict]:
        """تحلیل اصلی استراتژی دنبال کردن روند"""
        if not self.validate_data(df):
            return None
        
        df = df.copy()
        
        # محاسبه اندیکاتورهای ترند
        df = self._calculate_trend_indicators(df)
        
        # تشخیص جهت ترند
        trend_direction = self._identify_trend_direction(df)
        
        if trend_direction == 0:  # بدون ترند مشخص
            return None
        
        # تحلیل نقاط ورود
        entry_analysis = self._analyze_entry_points(df, trend_direction)
        
        if not entry_analysis['valid']:
            return None
        
        return {
            'strategy': self.name,
            'symbol': symbol,
            'signal': trend_direction,
            'signal_type': 'TREND_CONTINUATION',
            'confidence': entry_analysis['confidence'],
            'current_price': df['close'].iloc[-1],
            'entry_price': entry_analysis['entry_price'],
            'stop_loss': entry_analysis['stop_loss'],
            'take_profits': entry_analysis['take_profits'],
            'risk_reward_ratio': entry_analysis['risk_reward'],
            'reasoning': entry_analysis['reasoning'],
            'market_context': {
                'trend_direction': 'BULLISH' if trend_direction > 0 else 'BEARISH',
                'trend_strength': entry_analysis['trend_strength'],
                'adx_value': df['adx'].iloc[-1],
                'ema_alignment': entry_analysis['ema_alignment']
            }
        }

    def _calculate_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """محاسبه اندیکاتورهای ترند"""
        # EMAs
        df['ema_fast'] = talib.EMA(df['close'], timeperiod=self.ema_fast)
        df['ema_medium'] = talib.EMA(df['close'], timeperiod=self.ema_medium)
        df['ema_slow'] = talib.EMA(df['close'], timeperiod=self.ema_slow)
        
        # ATR برای محاسبه stop loss
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=self.atr_period)
        
        # ADX برای قدرت ترند
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=self.adx_period)
        
        # MACD برای تأیید
        df['macd'], df['macdsignal'], df['macdhist'] = talib.MACD(df['close'])
        
        return df

    def _identify_trend_direction(self, df: pd.DataFrame) -> int:
        """تشخیص جهت ترند"""
        latest_data = df.iloc[-1]
        
        # بررسی ADX - باید بالای حد آستانه باشد
        if latest_data['adx'] < self.min_adx:
            return 0
        
        # بررسی ترتیب EMAs
        ema_fast = latest_data['ema_fast']
        ema_medium = latest_data['ema_medium']
        ema_slow = latest_data['ema_slow']
        current_price = latest_data['close']
        
        # آپ ترند: قیمت > EMA_fast > EMA_medium > EMA_slow
        if (current_price > ema_fast > ema_medium > ema_slow):
            return 1
        
        # دان ترند: قیمت < EMA_fast < EMA_medium < EMA_slow
        elif (current_price < ema_fast < ema_medium < ema_slow):
            return -1
        
        return 0

    def _analyze_entry_points(self, df: pd.DataFrame, trend_direction: int) -> Dict:
        """تحلیل نقاط ورود"""
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        current_price = latest['close']
        atr = latest['atr']
        
        confidence = 0.5
        reasoning = []
        
        # تحلیل کراس‌اور EMAs
        ema_alignment = self._check_ema_alignment(df)
        if ema_alignment:
            confidence += 0.2
            reasoning.append("EMAs در ترتیب صحیح")
        
        # تحلیل MACD
        if trend_direction > 0 and latest['macd'] > latest['macdsignal']:
            confidence += 0.15
            reasoning.append("MACD صعودی")
        elif trend_direction < 0 and latest['macd'] < latest['macdsignal']:
            confidence += 0.15
            reasoning.append("MACD نزولی")
        
        # تحلیل ADX
        adx_strength = min(1.0, latest['adx'] / 50.0)
        confidence += adx_strength * 0.15
        reasoning.append(f"قدرت ترند ADX: {latest['adx']:.1f}")
        
        # محاسبه نقاط ورود و خروج
        if trend_direction > 0:
            entry_price = current_price
            stop_loss = current_price - (atr * self.trail_stop_atr_multiplier)
            take_profit_1 = current_price + (atr * 2)
            take_profit_2 = current_price + (atr * 4)
        else:
            entry_price = current_price
            stop_loss = current_price + (atr * self.trail_stop_atr_multiplier)
            take_profit_1 = current_price - (atr * 2)
            take_profit_2 = current_price - (atr * 4)
        
        risk_reward = abs(take_profit_1 - entry_price) / abs(entry_price - stop_loss)
        
        return {
            'valid': confidence >= self.adaptive_params['confidence_threshold'],
            'confidence': confidence,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profits': [take_profit_1, take_profit_2],
            'risk_reward': risk_reward,
            'trend_strength': adx_strength,
            'ema_alignment': ema_alignment,
            'reasoning': reasoning
        }

    def _check_ema_alignment(self, df: pd.DataFrame) -> bool:
        """بررسی ترتیب صحیح EMAs"""
        latest = df.iloc[-1]
        
        # برای آپ ترند
        if (latest['ema_fast'] > latest['ema_medium'] > latest['ema_slow']):
            return True
        
        # برای دان ترند
        if (latest['ema_fast'] < latest['ema_medium'] < latest['ema_slow']):
            return True
        
        return False

    def generate_signal(self, analysis_data: Dict) -> Dict:
        """تولید سیگنال نهایی"""
        signal_strength = analysis_data['confidence']
        
        if signal_strength >= 0.8:
            signal_type = StrategySignal.STRONG_BUY if analysis_data['signal'] > 0 else StrategySignal.STRONG_SELL
        elif signal_strength >= 0.7:
            signal_type = StrategySignal.BUY if analysis_data['signal'] > 0 else StrategySignal.SELL
        else:
            signal_type = StrategySignal.HOLD
        
        return {
            'signal': signal_type.value,
            'confidence': signal_strength,
            'strategy': self.name,
            'analysis': analysis_data
        }

    def is_market_suitable(self, market_context: Dict) -> bool:
        """بررسی مناسب بودن بازار برای دنبال کردن روند"""
        trend_strength = market_context.get('trend_strength', 0)
        volatility = market_context.get('volatility', 1.0)
        
        # مناسب برای ترندهای قوی و نوسانات متوسط
        return (trend_strength >= self.min_trend_strength and 
                0.5 <= volatility <= 2.0)
