# analyzers/pattern_analyzer.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum

class PatternType(Enum):
    # Candlestick Patterns
    HAMMER = "HAMMER"
    SHOOTING_STAR = "SHOOTING_STAR"
    DOJI = "DOJI"
    ENGULFING_BULLISH = "ENGULFING_BULLISH"
    ENGULFING_BEARISH = "ENGULFING_BEARISH"
    MORNING_STAR = "MORNING_STAR"
    EVENING_STAR = "EVENING_STAR"
    
    # Chart Patterns
    HEAD_AND_SHOULDERS = "HEAD_AND_SHOULDERS"
    DOUBLE_TOP = "DOUBLE_TOP"
    DOUBLE_BOTTOM = "DOUBLE_BOTTOM"
    TRIANGLE_ASCENDING = "TRIANGLE_ASCENDING"
    TRIANGLE_DESCENDING = "TRIANGLE_DESCENDING"
    WEDGE_RISING = "WEDGE_RISING"
    WEDGE_FALLING = "WEDGE_FALLING"
    FLAG_BULLISH = "FLAG_BULLISH"
    FLAG_BEARISH = "FLAG_BEARISH"
    
    # Support/Resistance Patterns
    BREAKOUT_RESISTANCE = "BREAKOUT_RESISTANCE"
    BREAKOUT_SUPPORT = "BREAKOUT_SUPPORT"
    FALSE_BREAKOUT = "FALSE_BREAKOUT"
    RETEST_SUPPORT = "RETEST_SUPPORT"
    RETEST_RESISTANCE = "RETEST_RESISTANCE"

class PatternSignal(Enum):
    STRONG_BULLISH = 2
    BULLISH = 1
    NEUTRAL = 0
    BEARISH = -1
    STRONG_BEARISH = -2

class PatternAnalyzer:
    """
    تحلیلگر الگوهای تکنیکال برای شناسایی pattern های مختلف
    """

    def __init__(self):
        self.name = "Pattern Analyzer"
        self.min_candles = 50
        
        # تنظیمات حساسیت
        self.sensitivity = {
            'candlestick': 0.7,
            'chart_pattern': 0.6,
            'support_resistance': 0.8
        }

    def analyze_patterns(self, df: pd.DataFrame) -> Dict:
        """تحلیل جامع الگوها"""
        if df is None or len(df) < self.min_candles:
            return {
                'patterns_found': [],
                'strongest_pattern': None,
                'overall_signal': PatternSignal.NEUTRAL.value,
                'confidence': 0.0
            }

        df = df.copy()
        
        # تحلیل انواع مختلف الگو
        candlestick_patterns = self._analyze_candlestick_patterns(df)
        chart_patterns = self._analyze_chart_patterns(df)
        sr_patterns = self._analyze_support_resistance_patterns(df)
        
        # ترکیب نتایج
        all_patterns = candlestick_patterns + chart_patterns + sr_patterns
        
        # یافتن قوی‌ترین الگو
        strongest_pattern = self._find_strongest_pattern(all_patterns)
        
        # محاسبه سیگنال کلی
        overall_signal, confidence = self._calculate_overall_signal(all_patterns)
        
        return {
            'patterns_found': all_patterns,
            'candlestick_patterns': candlestick_patterns,
            'chart_patterns': chart_patterns,
            'support_resistance_patterns': sr_patterns,
            'strongest_pattern': strongest_pattern,
            'overall_signal': overall_signal,
            'confidence': confidence,
            'pattern_count': len(all_patterns),
            'bullish_patterns': len([p for p in all_patterns if p['signal'] > 0]),
            'bearish_patterns': len([p for p in all_patterns if p['signal'] < 0])
        }

    def _analyze_candlestick_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """تحلیل الگوهای کندل"""
        patterns = []
        
        # محاسبه خصوصیات کندل
        df = self._calculate_candle_properties(df)
        
        # بررسی الگوهای مختلف
        patterns.extend(self._detect_hammer(df))
        patterns.extend(self._detect_shooting_star(df))
        patterns.extend(self._detect_doji(df))
        patterns.extend(self._detect_engulfing(df))
        patterns.extend(self._detect_morning_evening_star(df))
        
        return patterns

    def _calculate_candle_properties(self, df: pd.DataFrame) -> pd.DataFrame:
        """محاسبه خصوصیات کندل"""
        df['body'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
        df['total_range'] = df['high'] - df['low']
        df['body_ratio'] = df['body'] / df['total_range']
        df['upper_shadow_ratio'] = df['upper_shadow'] / df['total_range']
        df['lower_shadow_ratio'] = df['lower_shadow'] / df['total_range']
        df['is_bullish'] = df['close'] > df['open']
        df['is_bearish'] = df['close'] < df['open']
        
        return df

    def _detect_hammer(self, df: pd.DataFrame) -> List[Dict]:
        """تشخیص الگوی Hammer"""
        patterns = []
        
        for i in range(1, len(df)):
            if (df['lower_shadow_ratio'].iloc[i] > 0.6 and  # سایه پایین بلند
                df['upper_shadow_ratio'].iloc[i] < 0.1 and   # سایه بالا کوتاه
                df['body_ratio'].iloc[i] < 0.3):             # بدنه کوچک
                
                # بررسی موقعیت در ترند نزولی
                if self._is_in_downtrend(df, i):
                    confidence = self._calculate_hammer_confidence(df, i)
                    
                    if confidence >= self.sensitivity['candlestick']:
                        patterns.append({
                            'type': PatternType.HAMMER.value,
                            'signal': PatternSignal.BULLISH.value,
                            'confidence': confidence,
                            'position': i,
                            'description': 'Hammer در انتهای ترند نزولی - احتمال برگشت صعودی',
                            'target_levels': self._calculate_hammer_targets(df, i)
                        })
        
        return patterns

    def _detect_shooting_star(self, df: pd.DataFrame) -> List[Dict]:
        """تشخیص الگوی Shooting Star"""
        patterns = []
        
        for i in range(1, len(df)):
            if (df['upper_shadow_ratio'].iloc[i] > 0.6 and  # سایه بالا بلند
                df['lower_shadow_ratio'].iloc[i] < 0.1 and   # سایه پایین کوتاه
                df['body_ratio'].iloc[i] < 0.3):             # بدنه کوچک
                
                # بررسی موقعیت در ترند صعودی
                if self._is_in_uptrend(df, i):
                    confidence = self._calculate_shooting_star_confidence(df, i)
                    
                    if confidence >= self.sensitivity['candlestick']:
                        patterns.append({
                            'type': PatternType.SHOOTING_STAR.value,
                            'signal': PatternSignal.BEARISH.value,
                            'confidence': confidence,
                            'position': i,
                            'description': 'Shooting Star در انتهای ترند صعودی - احتمال برگشت نزولی',
                            'target_levels': self._calculate_shooting_star_targets(df, i)
                        })
        
        return patterns

    def _detect_engulfing(self, df: pd.DataFrame) -> List[Dict]:
        """تشخیص الگوی Engulfing"""
        patterns = []
        
        for i in range(1, len(df)):
            current_open = df['open'].iloc[i]
            current_close = df['close'].iloc[i]
            prev_open = df['open'].iloc[i-1]
            prev_close = df['close'].iloc[i-1]
            
            # Bullish Engulfing
            if (df['is_bearish'].iloc[i-1] and df['is_bullish'].iloc[i] and
                current_open < prev_close and current_close > prev_open):
                
                confidence = self._calculate_engulfing_confidence(df, i, True)
                if confidence >= self.sensitivity['candlestick']:
                    patterns.append({
                        'type': PatternType.ENGULFING_BULLISH.value,
                        'signal': PatternSignal.BULLISH.value,
                        'confidence': confidence,
                        'position': i,
                        'description': 'Bullish Engulfing - سیگنال خرید قوی',
                        'target_levels': self._calculate_engulfing_targets(df, i, True)
                    })
            
            # Bearish Engulfing
            elif (df['is_bullish'].iloc[i-1] and df['is_bearish'].iloc[i] and
                  current_open > prev_close and current_close < prev_open):
                
                confidence = self._calculate_engulfing_confidence(df, i, False)
                if confidence >= self.sensitivity['candlestick']:
                    patterns.append({
                        'type': PatternType.ENGULFING_BEARISH.value,
                        'signal': PatternSignal.BEARISH.value,
                        'confidence': confidence,
                        'position': i,
                        'description': 'Bearish Engulfing - سیگنال فروش قوی',
                        'target_levels': self._calculate_engulfing_targets(df, i, False)
                    })
        
        return patterns

    def _analyze_chart_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """تحلیل الگوهای چارت"""
        patterns = []
        
        # محاسبه swing points
        df = self._calculate_swing_points(df)
        
        # تشخیص الگوهای مختلف
        patterns.extend(self._detect_double_top_bottom(df))
        patterns.extend(self._detect_head_and_shoulders(df))
        patterns.extend(self._detect_triangles(df))
        patterns.extend(self._detect_flags_pennants(df))
        
        return patterns

    def _detect_double_top_bottom(self, df: pd.DataFrame) -> List[Dict]:
        """تشخیص Double Top/Bottom"""
        patterns = []
        
        swing_highs = df[df['swing_high']]['high'].values
        swing_lows = df[df['swing_low']]['low'].values
        
        # Double Top
        if len(swing_highs) >= 2:
            for i in range(len(swing_highs) - 1):
                for j in range(i + 1, len(swing_highs)):
                    if abs(swing_highs[i] - swing_highs[j]) / swing_highs[i] < 0.02:  # 2% tolerance
                        confidence = self._calculate_double_top_confidence(df, i, j)
                        if confidence >= self.sensitivity['chart_pattern']:
                            patterns.append({
                                'type': PatternType.DOUBLE_TOP.value,
                                'signal': PatternSignal.BEARISH.value,
                                'confidence': confidence,
                                'description': 'Double Top - الگوی برگشتی نزولی',
                                'target_levels': self._calculate_double_top_targets(df, swing_highs[i])
                            })
        
        # Double Bottom
        if len(swing_lows) >= 2:
            for i in range(len(swing_lows) - 1):
                for j in range(i + 1, len(swing_lows)):
                    if abs(swing_lows[i] - swing_lows[j]) / swing_lows[i] < 0.02:  # 2% tolerance
                        confidence = self._calculate_double_bottom_confidence(df, i, j)
                        if confidence >= self.sensitivity['chart_pattern']:
                            patterns.append({
                                'type': PatternType.DOUBLE_BOTTOM.value,
                                'signal': PatternSignal.BULLISH.value,
                                'confidence': confidence,
                                'description': 'Double Bottom - الگوی برگشتی صعودی',
                                'target_levels': self._calculate_double_bottom_targets(df, swing_lows[i])
                            })
        
        return patterns

    def _analyze_support_resistance_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """تحلیل الگوهای Support/Resistance"""
        patterns = []
        
        # یافتن سطوح کلیدی
        support_levels = self._find_support_levels(df)
        resistance_levels = self._find_resistance_levels(df)
        
        current_price = df['close'].iloc[-1]
        
        # بررسی breakout ها
        for level in resistance_levels:
            if current_price > level * 1.005:  # 0.5% breakout threshold
                confidence = self._calculate_breakout_confidence(df, level, 'resistance')
                if confidence >= self.sensitivity['support_resistance']:
                    patterns.append({
                        'type': PatternType.BREAKOUT_RESISTANCE.value,
                        'signal': PatternSignal.BULLISH.value,
                        'confidence': confidence,
                        'description': f'شکست مقاومت در سطح {level:.4f}',
                        'level': level,
                        'target_levels': self._calculate_breakout_targets(df, level, 'up')
                    })
        
        for level in support_levels:
            if current_price < level * 0.995:  # 0.5% breakdown threshold
                confidence = self._calculate_breakout_confidence(df, level, 'support')
                if confidence >= self.sensitivity['support_resistance']:
                    patterns.append({
                        'type': PatternType.BREAKOUT_SUPPORT.value,
                        'signal': PatternSignal.BEARISH.value,
                        'confidence': confidence,
                        'description': f'شکست حمایت در سطح {level:.4f}',
                        'level': level,
                        'target_levels': self._calculate_breakout_targets(df, level, 'down')
                    })
        
        return patterns

    def _find_strongest_pattern(self, patterns: List[Dict]) -> Optional[Dict]:
        """یافتن قوی‌ترین الگو"""
        if not patterns:
            return None
        
        # مرتب‌سازی بر اساس confidence و قدرت سیگنال
        scored_patterns = []
        for pattern in patterns:
            score = pattern['confidence'] * abs(pattern['signal'])
            scored_patterns.append((score, pattern))
        
        scored_patterns.sort(key=lambda x: x[0], reverse=True)
        return scored_patterns[0][1] if scored_patterns else None

    def _calculate_overall_signal(self, patterns: List[Dict]) -> Tuple[int, float]:
        """محاسبه سیگنال کلی"""
        if not patterns:
            return PatternSignal.NEUTRAL.value, 0.0
        
        weighted_signals = []
        total_weight = 0
        
        for pattern in patterns:
            weight = pattern['confidence']
            signal = pattern['signal']
            weighted_signals.append(signal * weight)
            total_weight += weight
        
        if total_weight == 0:
            return PatternSignal.NEUTRAL.value, 0.0
        
        overall_signal = sum(weighted_signals) / total_weight
        confidence = min(total_weight / len(patterns), 1.0)
        
        # تبدیل به enum values
        if overall_signal >= 1.5:
            return PatternSignal.STRONG_BULLISH.value, confidence
        elif overall_signal >= 0.5:
            return PatternSignal.BULLISH.value, confidence
        elif overall_signal <= -1.5:
            return PatternSignal.STRONG_BEARISH.value, confidence
        elif overall_signal <= -0.5:
            return PatternSignal.BEARISH.value, confidence
        else:
            return PatternSignal.NEUTRAL.value, confidence

    # Helper methods for pattern detection
    def _is_in_downtrend(self, df: pd.DataFrame, position: int) -> bool:
        """بررسی ترند نزولی"""
        if position < 10:
            return False
        
        recent_prices = df['close'].iloc[position-10:position+1]
        return recent_prices.iloc[-1] < recent_prices.iloc[0] * 0.95

    def _is_in_uptrend(self, df: pd.DataFrame, position: int) -> bool:
        """بررسی ترند صعودی"""
        if position < 10:
            return False
        
        recent_prices = df['close'].iloc[position-10:position+1]
        return recent_prices.iloc[-1] > recent_prices.iloc[0] * 1.05

    def _calculate_swing_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """محاسبه swing points"""
        period = 5
        df['swing_high'] = False
        df['swing_low'] = False
        
        for i in range(period, len(df) - period):
            # Swing High
            if (df['high'].iloc[i] == df['high'].iloc[i-period:i+period+1].max()):
                df.iloc[i, df.columns.get_loc('swing_high')] = True
            
            # Swing Low
            if (df['low'].iloc[i] == df['low'].iloc[i-period:i+period+1].min()):
                df.iloc[i, df.columns.get_loc('swing_low')] = True
        
        return df

    # Confidence calculation methods (simplified for brevity)
    def _calculate_hammer_confidence(self, df: pd.DataFrame, position: int) -> float:
        """محاسبه confidence برای Hammer"""
        base_confidence = 0.7
        
        # بررسی حجم
        if df['volume'].iloc[position] > df['volume'].iloc[position-5:position].mean() * 1.2:
            base_confidence += 0.1
        
        # بررسی موقعیت در ترند
        if self._is_in_downtrend(df, position):
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)

    def _calculate_shooting_star_confidence(self, df: pd.DataFrame, position: int) -> float:
        """محاسبه confidence برای Shooting Star"""
        return self._calculate_hammer_confidence(df, position)  # منطق مشابه

    def _calculate_engulfing_confidence(self, df: pd.DataFrame, position: int, is_bullish: bool) -> float:
        """محاسبه confidence برای Engulfing"""
        base_confidence = 0.8
        
        # بررسی اندازه کندل
        current_body = df['body'].iloc[position]
        prev_body = df['body'].iloc[position-1]
        
        if current_body > prev_body * 1.5:
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)

    # Target calculation methods (simplified)
    def _calculate_hammer_targets(self, df: pd.DataFrame, position: int) -> Dict:
        """محاسبه اهداف برای Hammer"""
        current_price = df['close'].iloc[position]
        atr = df['high'].iloc[position-14:position+1].subtract(df['low'].iloc[position-14:position+1]).mean()
        
        return {
            'target_1': current_price + atr,
            'target_2': current_price + atr * 2,
            'stop_loss': df['low'].iloc[position] - atr * 0.5
        }

    def _calculate_shooting_star_targets(self, df: pd.DataFrame, position: int) -> Dict:
        """محاسبه اهداف برای Shooting Star"""
        current_price = df['close'].iloc[position]
        atr = df['high'].iloc[position-14:position+1].subtract(df['low'].iloc[position-14:position+1]).mean()
        
        return {
            'target_1': current_price - atr,
            'target_2': current_price - atr * 2,
            'stop_loss': df['high'].iloc[position] + atr * 0.5
        }

    def _calculate_engulfing_targets(self, df: pd.DataFrame, position: int, is_bullish: bool) -> Dict:
        """محاسبه اهداف برای Engulfing"""
        current_price = df['close'].iloc[position]
        pattern_range = df['body'].iloc[position]
        
        if is_bullish:
            return {
                'target_1': current_price + pattern_range,
                'target_2': current_price + pattern_range * 1.618,
                'stop_loss': df['low'].iloc[position-1]
            }
        else:
            return {
                'target_1': current_price - pattern_range,
                'target_2': current_price - pattern_range * 1.618,
                'stop_loss': df['high'].iloc[position-1]
            }

    # Additional helper methods would be implemented here...
    def _find_support_levels(self, df: pd.DataFrame) -> List[float]:
        """یافتن سطوح حمایت"""
        swing_lows = df[df['swing_low']]['low'].values
        return list(swing_lows[-5:])  # آخرین 5 swing low

    def _find_resistance_levels(self, df: pd.DataFrame) -> List[float]:
        """یافتن سطوح مقاومت"""
        swing_highs = df[df['swing_high']]['high'].values
        return list(swing_highs[-5:])  # آخرین 5 swing high

    def _calculate_breakout_confidence(self, df: pd.DataFrame, level: float, level_type: str) -> float:
        """محاسبه confidence برای breakout"""
        return 0.75  # مقدار ساده برای مثال

    def _calculate_breakout_targets(self, df: pd.DataFrame, level: float, direction: str) -> Dict:
        """محاسبه اهداف برای breakout"""
        current_price = df['close'].iloc[-1]
        distance = abs(current_price - level)
        
        if direction == 'up':
            return {
                'target_1': current_price + distance,
                'target_2': current_price + distance * 1.618,
                'stop_loss': level * 0.98
            }
        else:
            return {
                'target_1': current_price - distance,
                'target_2': current_price - distance * 1.618,
                'stop_loss': level * 1.02
            }
