"""
Market Analyzer - تحلیلگر شرایط کلی بازار
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import asyncio

class MarketAnalyzer:
    """تحلیل شرایط کلی بازار ارزهای دیجیتال"""
    
    def __init__(self):
        self.market_phases = {
            'BULL_MARKET': 'بازار صعودی',
            'BEAR_MARKET': 'بازار نزولی', 
            'SIDEWAYS': 'بازار خنثی',
            'ACCUMULATION': 'فاز انباشت',
            'DISTRIBUTION': 'فاز توزیع'
        }
        
    async def analyze_market_context(self, symbol: str, data: pd.DataFrame) -> Dict:
        """تحلیل شرایط بازار برای یک نماد"""
        
        # تحلیل ترند کلی
        trend_analysis = self._analyze_trend(data)
        
        # تحلیل حجم
        volume_analysis = self._analyze_volume(data)
        
        # تحلیل نوسانات
        volatility_analysis = self._analyze_volatility(data)
        
        # تحلیل سطوح کلیدی
        key_levels = self._identify_key_levels(data)
        
        # تحلیل فاز بازار
        market_phase = self._determine_market_phase(data)
        
        # قدرت ترند
        trend_strength = self._calculate_trend_strength(data)
        
        return {
            'symbol': symbol,
            'trend_analysis': trend_analysis,
            'volume_analysis': volume_analysis,
            'volatility_analysis': volatility_analysis,
            'key_levels': key_levels,
            'market_phase': market_phase,
            'trend_strength': trend_strength,
            'analysis_timestamp': datetime.now()
        }
    
    def _analyze_trend(self, data: pd.DataFrame) -> Dict:
        """تحلیل ترند"""
        # محاسبه میانگین‌های متحرک
        data['ema_20'] = data['close'].ewm(span=20).mean()
        data['ema_50'] = data['close'].ewm(span=50).mean()
        data['ema_200'] = data['close'].ewm(span=200).mean()
        
        current_price = data['close'].iloc[-1]
        ema_20 = data['ema_20'].iloc[-1]
        ema_50 = data['ema_50'].iloc[-1]
        ema_200 = data['ema_200'].iloc[-1]
        
        # تعیین جهت ترند
        if current_price > ema_20 > ema_50 > ema_200:
            trend_direction = "STRONG_UPTREND"
        elif current_price > ema_20 > ema_50:
            trend_direction = "UPTREND"
        elif current_price < ema_20 < ema_50 < ema_200:
            trend_direction = "STRONG_DOWNTREND"
        elif current_price < ema_20 < ema_50:
            trend_direction = "DOWNTREND"
        else:
            trend_direction = "SIDEWAYS"
        
        return {
            'direction': trend_direction,
            'ema_alignment': {
                'current_price': current_price,
                'ema_20': ema_20,
                'ema_50': ema_50,
                'ema_200': ema_200
            },
            'price_above_ema20': current_price > ema_20,
            'price_above_ema50': current_price > ema_50,
            'price_above_ema200': current_price > ema_200
        }
    
    def _analyze_volume(self, data: pd.DataFrame) -> Dict:
        """تحلیل حجم معاملات"""
        # میانگین حجم
        volume_ma_20 = data['volume'].rolling(20).mean()
        current_volume = data['volume'].iloc[-1]
        avg_volume = volume_ma_20.iloc[-1]
        
        # تحلیل حجم نسبت به میانگین
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        if volume_ratio > 1.5:
            volume_context = "HIGH"
        elif volume_ratio > 1.2:
            volume_context = "ABOVE_AVERAGE"
        elif volume_ratio < 0.7:
            volume_context = "LOW"
        else:
            volume_context = "NORMAL"
        
        # تحلیل ترند حجم
        volume_trend = "INCREASING" if data['volume'].iloc[-5:].mean() > data['volume'].iloc[-10:-5].mean() else "DECREASING"
        
        return {
            'current_volume': current_volume,
            'average_volume': avg_volume,
            'volume_ratio': volume_ratio,
            'volume_context': volume_context,
            'volume_trend': volume_trend
        }
    
    def _analyze_volatility(self, data: pd.DataFrame) -> Dict:
        """تحلیل نوسانات"""
        # محاسبه ATR
        data['tr'] = np.maximum(
            data['high'] - data['low'],
            np.maximum(
                abs(data['high'] - data['close'].shift(1)),
                abs(data['low'] - data['close'].shift(1))
            )
        )
        data['atr'] = data['tr'].rolling(14).mean()
        
        current_atr = data['atr'].iloc[-1]
        current_price = data['close'].iloc[-1]
        atr_percentage = (current_atr / current_price) * 100
        
        # تحلیل سطح نوسانات
        if atr_percentage > 5:
            volatility_level = "VERY_HIGH"
        elif atr_percentage > 3:
            volatility_level = "HIGH"
        elif atr_percentage > 2:
            volatility_level = "MODERATE"
        else:
            volatility_level = "LOW"
        
        return {
            'atr': current_atr,
            'atr_percentage': atr_percentage,
            'volatility_level': volatility_level
        }
    
    def _identify_key_levels(self, data: pd.DataFrame) -> Dict:
        """شناسایی سطوح کلیدی"""
        # محاسبه سطوح حمایت و مقاومت
        highs = data['high'].rolling(20).max()
        lows = data['low'].rolling(20).min()
        
        # سطوح فیبوناچی
        max_price = data['high'].iloc[-100:].max()
        min_price = data['low'].iloc[-100:].min()
        fib_levels = self._calculate_fibonacci_levels(max_price, min_price)
        
        return {
            'resistance_levels': [highs.iloc[-1], highs.iloc[-20], highs.iloc[-50]],
            'support_levels': [lows.iloc[-1], lows.iloc[-20], lows.iloc[-50]],
            'fibonacci_levels': fib_levels,
            'swing_high': max_price,
            'swing_low': min_price
        }
    
    def _calculate_fibonacci_levels(self, high: float, low: float) -> Dict:
        """محاسبه سطوح فیبوناچی"""
        diff = high - low
        return {
            '0.0': high,
            '23.6': high - (diff * 0.236),
            '38.2': high - (diff * 0.382),
            '50.0': high - (diff * 0.5),
            '61.8': high - (diff * 0.618),
            '78.6': high - (diff * 0.786),
            '100.0': low
        }
    
    def _determine_market_phase(self, data: pd.DataFrame) -> str:
        """تعیین فاز بازار"""
        # تحلیل بر اساس حجم و قیمت
        price_change = (data['close'].iloc[-1] / data['close'].iloc[-30] - 1) * 100
        volume_trend = data['volume'].iloc[-10:].mean() / data['volume'].iloc[-30:-10].mean()
        
        if price_change > 10 and volume_trend > 1.2:
            return "BULL_MARKET"
        elif price_change < -10 and volume_trend > 1.2:
            return "BEAR_MARKET"
        elif abs(price_change) < 5 and volume_trend < 0.8:
            return "ACCUMULATION"
        elif abs(price_change) < 5 and volume_trend > 1.3:
            return "DISTRIBUTION"
        else:
            return "SIDEWAYS"
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """محاسبه قدرت ترند (0-100)"""
        # محاسبه ADX
        high = data['high']
        low = data['low'] 
        close = data['close']
        
        # محاسبه True Range
        tr = np.maximum(high - low, 
                       np.maximum(abs(high - close.shift(1)),
                                 abs(low - close.shift(1))))
        
        # محاسبه Directional Movement
        dm_plus = np.where((high - high.shift(1)) > (low.shift(1) - low), 
                          np.maximum(high - high.shift(1), 0), 0)
        dm_minus = np.where((low.shift(1) - low) > (high - high.shift(1)), 
                           np.maximum(low.shift(1) - low, 0), 0)
        
        # Smoothed versions
        tr_smooth = pd.Series(tr).rolling(14).mean()
        dm_plus_smooth = pd.Series(dm_plus).rolling(14).mean()
        dm_minus_smooth = pd.Series(dm_minus).rolling(14).mean()
        
        # Directional Indicators
        di_plus = 100 * (dm_plus_smooth / tr_smooth)
        di_minus = 100 * (dm_minus_smooth / tr_smooth)
        
        # ADX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(14).mean().iloc[-1]
        
        return min(adx, 100) if not pd.isna(adx) else 0
