import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from .base_strategy import BaseStrategy, StrategyType, StrategySignal
import talib

class BreakoutStrategy(BaseStrategy):
    """
    استراتژی شکست سطوح - شناسایی شکست سطوح مقاومت و حمایت
    """
    
    def __init__(self):
        super().__init__("Breakout Strategy", StrategyType.BREAKOUT)
        
        # پارامترهای شناسایی سطوح
        self.level_lookback = 20
        self.level_tolerance = 0.002  # 0.2% تلرانس
        self.min_touches = 3  # حداقل تعداد تماس با سطح
        
        # پارامترهای تأیید شکست
        self.breakout_threshold = 0.005  # 0.5% شکست
        self.volume_confirmation_ratio = 1.8
        self.candle_close_confirmation = True
        
        # پارامترهای Bollinger Bands
        self.bb_period = 20
        self.bb_std = 2
        
        # پارامترهای range detection
        self.min_range_periods = 10
        self.max_range_periods = 50

    def analyze(self, df: pd.DataFrame, symbol: str) -> Optional[Dict]:
        """تحلیل اصلی استراتژی شکست"""
        if not self.validate_data(df):
            return None
        
        df = df.copy()
        
        # محاسبه اندیکاتورها
        df = self._calculate_breakout_indicators(df)
        
        # شناسایی سطوح کلیدی
        key_levels = self._identify_key_levels(df)
        
        if not key_levels:
            return None
        
        # تشخیص شکست
        breakout_analysis = self._detect_breakout(df, key_levels)
        
        if not breakout_analysis['found']:
            return None
        
        # تحلیل کیفیت شکست
        quality_analysis = self._analyze_breakout_quality(df, breakout_analysis)
        
        if not quality_analysis['valid']:
            return None
        
        return {
            'strategy': self.name,
            'symbol': symbol,
            'signal': breakout_analysis['direction'],
            'signal_type': 'BREAKOUT',
            'confidence': quality_analysis['confidence'],
            'current_price': df['close'].iloc[-1],
            'entry_price': quality_analysis['entry_price'],
            'stop_loss': quality_analysis['stop_loss'],
            'take_profits': quality_analysis['take_profits'],
            'risk_reward_ratio': quality_analysis['risk_reward'],
            'reasoning': quality_analysis['reasoning'],
            'market_context': {
                'breakout_type': breakout_analysis['type'],
                'broken_level': breakout_analysis['level'],
                'volume_confirmation': quality_analysis['volume_confirmation'],
                'level_strength': breakout_analysis['level_strength']
            }
        }

    def _calculate_breakout_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """محاسبه اندیکاتورهای شکست"""
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
            df['close'], timeperiod=self.bb_period, nbdevup=self.bb_std, nbdevdn=self.bb_std
        )
        
        # ATR برای volatility
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Volume indicators
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # RSI برای momentum
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        
        # ADX برای trend strength
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        
        return df

    def _identify_key_levels(self, df: pd.DataFrame) -> List[Dict]:
        """شناسایی سطوح کلیدی Support/Resistance"""
        levels = []
        
        # شناسایی swing highs و lows
        highs = self._find_swing_highs(df)
        lows = self._find_swing_lows(df)
        
        # تبدیل به سطوح
        resistance_levels = self._cluster_levels(highs, 'resistance')
        support_levels = self._cluster_levels(lows, 'support')
        
        levels.extend(resistance_levels)
        levels.extend(support_levels)
        
        # اضافه کردن Bollinger Bands
        latest = df.iloc[-1]
        levels.append({
            'type': 'resistance',
            'level': latest['bb_upper'],
            'strength': 0.6,
            'touches': 1,
            'source': 'bollinger_upper'
        })
        
        levels.append({
            'type': 'support',
            'level': latest['bb_lower'],
            'strength': 0.6,
            'touches': 1,
            'source': 'bollinger_lower'
        })
        
        return levels

    def _find_swing_highs(self, df: pd.DataFrame) -> List[Tuple]:
        """پیدا کردن swing highs"""
        highs = []
        lookback = 5
        
        for i in range(lookback, len(df) - lookback):
            current_high = df['high'].iloc[i]
            
            # بررسی آیا این نقطه بالاترین است
            is_highest = True
            for j in range(i - lookback, i + lookback + 1):
                if j != i and df['high'].iloc[j] >= current_high:
                    is_highest = False
                    break
            
            if is_highest:
                highs.append((i, current_high))
        
        return highs

    def _find_swing_lows(self, df: pd.DataFrame) -> List[Tuple]:
        """پیدا کردن swing lows"""
        lows = []
        lookback = 5
        
        for i in range(lookback, len(df) - lookback):
            current_low = df['low'].iloc[i]
            
            # بررسی آیا این نقطه پایین‌ترین است
            is_lowest = True
            for j in range(i - lookback, i + lookback + 1):
                if j != i and df['low'].iloc[j] <= current_low:
                    is_lowest = False
                    break
            
            if is_lowest:
                lows.append((i, current_low))
        
        return lows

    def _cluster_levels(self, points: List[Tuple], level_type: str) -> List[Dict]:
        """خوشه‌بندی نقاط به سطوح"""
        if not points:
            return []
        
        levels = []
        
        # مرتب‌سازی بر اساس قیمت
        points.sort(key=lambda x: x[1])
        
        current_cluster = [points[0]]
        
        for i in range(1, len(points)):
            price_diff = abs(points[i][1] - current_cluster[-1][1])
            tolerance = current_cluster[-1][1] * self.level_tolerance
            
            if price_diff <= tolerance:
                current_cluster.append(points[i])
            else:
                # ایجاد سطح از cluster فعلی
                if len(current_cluster) >= self.min_touches:
                    level_price = np.mean([p[1] for p in current_cluster])
                    strength = min(1.0, len(current_cluster) / 10.0)
                    
                    levels.append({
                        'type': level_type,
                        'level': level_price,
                        'strength': strength,
                        'touches': len(current_cluster),
                        'source': 'swing_points'
                    })
                
                current_cluster = [points[i]]
        
        # بررسی cluster آخر
        if len(current_cluster) >= self.min_touches:
            level_price = np.mean([p[1] for p in current_cluster])
            strength = min(1.0, len(current_cluster) / 10.0)
            
            levels.append({
                'type': level_type,
                'level': level_price,
                'strength': strength,
                'touches': len(current_cluster),
                'source': 'swing_points'
            })
        
        return levels

    def _detect_breakout(self, df: pd.DataFrame, key_levels: List[Dict]) -> Dict:
        """تشخیص شکست سطوح"""
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        current_price = latest['close']
        previous_price = prev['close']
        
        breakout = {
            'found': False,
            'direction': 0,
            'type': None,
            'level': 0,
            'level_strength': 0
        }
        
        for level in key_levels:
            level_price = level['level']
            level_type = level['type']
            level_strength = level['strength']
            
            # شکست resistance (صعودی)
            if (level_type == 'resistance' and 
                previous_price <= level_price and 
                current_price > level_price * (1 + self.breakout_threshold)):
                
                breakout['found'] = True
                breakout['direction'] = 1
                breakout['type'] = 'RESISTANCE_BREAKOUT'
                breakout['level'] = level_price
                breakout['level_strength'] = level_strength
                break
            
            # شکست support (نزولی)
            elif (level_type == 'support' and 
                  previous_price >= level_price and 
                  current_price < level_price * (1 - self.breakout_threshold)):
                
                breakout['found'] = True
                breakout['direction'] = -1
                breakout['type'] = 'SUPPORT_BREAKOUT'
                breakout['level'] = level_price
                breakout['level_strength'] = level_strength
                break
        
        return breakout

    def _analyze_breakout_quality(self, df: pd.DataFrame, breakout_analysis: Dict) -> Dict:
        """تحلیل کیفیت شکست"""
        latest = df.iloc[-1]
        
        confidence = 0.5  # شروع از 50%
        reasoning = []
        
        # بررسی تأیید حجم
        volume_confirmation = latest['volume_ratio'] >= self.volume_confirmation_ratio
        if volume_confirmation:
            confidence += 0.25
            reasoning.append("تأیید حجم بالا")
        
        # بررسی قدرت سطح شکسته شده
        level_strength = breakout_analysis['level_strength']
        confidence += level_strength * 0.2
        reasoning.append(f"قدرت سطح: {level_strength:.2f}")
        
        # بررسی momentum با RSI
        rsi = latest['rsi']
        if breakout_analysis['direction'] > 0 and rsi > 50:
            confidence += 0.1
            reasoning.append("RSI حمایتی")
        elif breakout_analysis['direction'] < 0 and rsi < 50:
            confidence += 0.1
            reasoning.append("RSI حمایتی")
        
        # بررسی volatility
        atr = latest['atr']
        current_price = latest['close']
        volatility_ratio = atr / current_price
        
        if 0.01 <= volatility_ratio <= 0.05:  # نوسانات متعادل
            confidence += 0.1
            reasoning.append("نوسانات متعادل")
        
        # محاسبه نقاط ورود و خروج
        entry_price = current_price
        broken_level = breakout_analysis['level']
        
        if breakout_analysis['direction'] > 0:  # شکست صعودی
            stop_loss = broken_level * 0.995  # 0.5% زیر سطح شکسته شده
            distance = entry_price - broken_level
            take_profit_1 = entry_price + distance
            take_profit_2 = entry_price + (distance * 2)
        else:  # شکست نزولی
            stop_loss = broken_level * 1.005  # 0.5% بالای سطح شکسته شده
            distance = broken_level - entry_price
            take_profit_1 = entry_price - distance
            take_profit_2 = entry_price - (distance * 2)
        
        risk_reward = abs(take_profit_1 - entry_price) / abs(entry_price - stop_loss)
        
        return {
            'valid': confidence >= self.adaptive_params['confidence_threshold'],
            'confidence': min(1.0, confidence),
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profits': [take_profit_1, take_profit_2],
            'risk_reward': risk_reward,
            'volume_confirmation': volume_confirmation,
            'reasoning': reasoning
        }

    def generate_signal(self, analysis_data: Dict) -> Dict:
        """تولید سیگنال نهایی"""
        confidence = analysis_data['confidence']
        
        if confidence >= 0.8:
            signal = StrategySignal.STRONG_BUY if analysis_data['signal'] > 0 else StrategySignal.STRONG_SELL
        elif confidence >= 0.7:
            signal = StrategySignal.BUY if analysis_data['signal'] > 0 else StrategySignal.SELL
        else:
            signal = StrategySignal.HOLD
        
        return {
            'signal': signal.value,
            'confidence': confidence,
            'strategy': self.name,
            'analysis': analysis_data
        }

    def is_market_suitable(self, market_context: Dict) -> bool:
        """بررسی مناسب بودن بازار برای استراتژی شکست"""
        volatility = market_context.get('volatility', 1.0)
        volume_trend = market_context.get('volume_trend', 'normal')
        
        # مناسب برای نوسانات متوسط تا بالا و حجم فعال
        return (volatility >= 0.8 and volume_trend in ['increasing', 'high'])
