"""
analyzers/price_action_analyzer.py
تحلیلگر Pure Price Action - نسخه بهبود یافته برای سیستم چندگانه
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum
import datetime

class MarketStructure(Enum):
    STRONG_UPTREND = "STRONG_UPTREND"
    WEAK_UPTREND = "WEAK_UPTREND"
    CONSOLIDATION = "CONSOLIDATION"
    WEAK_DOWNTREND = "WEAK_DOWNTREND"
    STRONG_DOWNTREND = "STRONG_DOWNTREND"

class SignalQuality(Enum):
    EXCELLENT = "EXCELLENT"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INVALID = "INVALID"

class SignalType(Enum):
    BREAKOUT = "BREAKOUT"
    PULLBACK = "PULLBACK"
    REVERSAL = "REVERSAL"
    CONTINUATION = "CONTINUATION"
    RANGE_TRADE = "RANGE_TRADE"

class PriceActionAnalyzer:
    """
    تحلیلگر Pure Price Action با قابلیت‌های پیشرفته
    """

    def __init__(self):
        self.name = "Pure Price Action Analyzer"
        self.min_candles = 150
        self.swing_period = 12
        self.trend_strength_period = 14
        self.volume_analysis_period = 20
        
        # آستانه‌های کیفیت سیگنال
        self.quality_thresholds = {
            'excellent': {'confidence': 0.85, 'risk_reward': 3.0, 'volume_confirmation': True},
            'high': {'confidence': 0.75, 'risk_reward': 2.5, 'volume_confirmation': True},
            'medium': {'confidence': 0.65, 'risk_reward': 2.0, 'volume_confirmation': False},
            'low': {'confidence': 0.5, 'risk_reward': 1.5, 'volume_confirmation': False}
        }

    def analyze(self, df: pd.DataFrame, symbol: str = None) -> Optional[Dict]:
        """تحلیل کامل Price Action"""
        if df is None or len(df) < self.min_candles:
            return None

        try:
            # کپی و پاکسازی داده‌ها
            df = df.copy()
            df = self._clean_data(df)

            # مراحل تحلیل
            df = self._identify_market_structure(df)
            df = self._find_key_levels(df)
            df = self._analyze_candle_patterns(df)
            df = self._detect_trend_changes(df)
            df = self._analyze_volume_context(df)
            df = self._calculate_momentum_indicators(df)

            # تولید سیگنال نهایی
            signal_analysis = self._generate_comprehensive_signal(df, symbol)

            return signal_analysis

        except Exception as e:
            print(f"خطا در تحلیل Price Action: {e}")
            return None

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """پاکسازی و آماده‌سازی داده‌ها"""
        # حذف داده‌های ناقص
        df = df.dropna()
        
        # بررسی صحت داده‌ها
        df = df[(df['high'] >= df['low']) & 
                (df['high'] >= df['open']) & 
                (df['high'] >= df['close']) &
                (df['low'] <= df['open']) & 
                (df['low'] <= df['close']) &
                (df['volume'] > 0)]
        
        return df

    def _identify_market_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """شناسایی ساختار بازار با دقت بالا"""
        # محاسبه swing points
        df = self._find_swing_points(df)
        
        # تعیین ترند با multiple timeframe concept
        df = self._determine_multi_timeframe_trend(df)
        
        # محاسبه قدرت ترند
        df = self._calculate_advanced_trend_strength(df)
        
        return df

    def _find_swing_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """پیدا کردن نقاط swing با دقت بالا"""
        period = self.swing_period
        
        df['swing_high'] = False
        df['swing_low'] = False
        df['swing_high_value'] = 0.0
        df['swing_low_value'] = 0.0

        for i in range(period, len(df) - period):
            # Swing High
            current_high = df['high'].iloc[i]
            left_highs = df['high'].iloc[i-period:i]
            right_highs = df['high'].iloc[i+1:i+period+1]
            
            if (current_high > left_highs.max() and 
                current_high > right_highs.max()):
                df.iloc[i, df.columns.get_loc('swing_high')] = True
                df.iloc[i, df.columns.get_loc('swing_high_value')] = current_high

            # Swing Low
            current_low = df['low'].iloc[i]
            left_lows = df['low'].iloc[i-period:i]
            right_lows = df['low'].iloc[i+1:i+period+1]
            
            if (current_low < left_lows.min() and 
                current_low < right_lows.min()):
                df.iloc[i, df.columns.get_loc('swing_low')] = True
                df.iloc[i, df.columns.get_loc('swing_low_value')] = current_low

        return df

    def _determine_multi_timeframe_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """تعیین ترند با نگاه چند زمانه"""
        # EMAs برای timeframe های مختلف
        df['ema_8'] = df['close'].ewm(span=8).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        df['ema_100'] = df['close'].ewm(span=100).mean()
        df['ema_200'] = df['close'].ewm(span=200).mean()

        # تحلیل Higher Highs و Lower Lows
        df = self._analyze_swing_structure(df)
        
        # تعیین ساختار بازار
        df['market_structure'] = df.apply(self._classify_market_structure, axis=1)
        
        return df

    def _analyze_swing_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """تحلیل ساختار swing points"""
        swing_highs = df[df['swing_high']]['high']
        swing_lows = df[df['swing_low']]['low']
        
        df['higher_highs'] = False
        df['lower_lows'] = False
        df['higher_lows'] = False
        df['lower_highs'] = False

        # تحلیل الگوی swing points
        for i in range(len(df)):
            current_idx = df.index[i]
            
            # Higher Highs
            recent_highs = swing_highs[swing_highs.index <= current_idx].tail(3)
            if len(recent_highs) >= 2 and recent_highs.iloc[-1] > recent_highs.iloc[-2]:
                df.iloc[i, df.columns.get_loc('higher_highs')] = True
                
            # Lower Lows
            recent_lows = swing_lows[swing_lows.index <= current_idx].tail(3)
            if len(recent_lows) >= 2 and recent_lows.iloc[-1] < recent_lows.iloc[-2]:
                df.iloc[i, df.columns.get_loc('lower_lows')] = True
                
            # Higher Lows
            if len(recent_lows) >= 2 and recent_lows.iloc[-1] > recent_lows.iloc[-2]:
                df.iloc[i, df.columns.get_loc('higher_lows')] = True
                
            # Lower Highs
            if len(recent_highs) >= 2 and recent_highs.iloc[-1] < recent_highs.iloc[-2]:
                df.iloc[i, df.columns.get_loc('lower_highs')] = True

        return df

    def _classify_market_structure(self, row) -> str:
        """تصنیف ساختار بازار براساس EMA و swing analysis"""
        price = row['close']
        ema8, ema21, ema50, ema100, ema200 = row['ema_8'], row['ema_21'], row['ema_50'], row['ema_100'], row['ema_200']
        
        # Strong Uptrend
        if (price > ema8 > ema21 > ema50 > ema100 and 
            row['higher_highs'] and row['higher_lows']):
            return MarketStructure.STRONG_UPTREND.value
            
        # Weak Uptrend
        elif (price > ema21 > ema50 and 
              not row['lower_lows']):
            return MarketStructure.WEAK_UPTREND.value
            
        # Strong Downtrend
        elif (price < ema8 < ema21 < ema50 < ema100 and 
              row['lower_lows'] and row['lower_highs']):
            return MarketStructure.STRONG_DOWNTREND.value
            
        # Weak Downtrend
        elif (price < ema21 < ema50 and 
              not row['higher_highs']):
            return MarketStructure.WEAK_DOWNTREND.value
            
        # Consolidation
        else:
            return MarketStructure.CONSOLIDATION.value

    def _calculate_advanced_trend_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """محاسبه قدرت ترند پیشرفته"""
        period = self.trend_strength_period
        
        # ADX محاسبه
        df['tr'] = np.maximum(df['high'] - df['low'],
                             np.maximum(abs(df['high'] - df['close'].shift(1)),
                                       abs(df['low'] - df['close'].shift(1))))
        
        df['atr'] = df['tr'].rolling(window=period).mean()
        
        # DM+ و DM-
        df['dm_plus'] = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
                                np.maximum(df['high'] - df['high'].shift(1), 0), 0)
        df['dm_minus'] = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
                                 np.maximum(df['low'].shift(1) - df['low'], 0), 0)
        
        df['di_plus'] = (df['dm_plus'].rolling(window=period).mean() / df['atr']) * 100
        df['di_minus'] = (df['dm_minus'].rolling(window=period).mean() / df['atr']) * 100
        
        df['adx'] = abs(df['di_plus'] - df['di_minus']) / (df['di_plus'] + df['di_minus']) * 100
        df['adx'] = df['adx'].rolling(window=period).mean()
        
        # Trend Strength Classification
        df['trend_strength'] = 'WEAK'
        df.loc[df['adx'] > 25, 'trend_strength'] = 'MEDIUM'
        df.loc[df['adx'] > 40, 'trend_strength'] = 'STRONG'
        df.loc[df['adx'] > 60, 'trend_strength'] = 'VERY_STRONG'
        
        return df

    def _find_key_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """شناسایی سطوح کلیدی Support/Resistance پیشرفته"""
        # تعریف ستون‌ها
        df['resistance_level'] = 0.0
        df['support_level'] = 0.0
        df['distance_to_resistance'] = 0.0
        df['distance_to_support'] = 0.0
        df['level_strength'] = 0
        
        # جمع‌آوری تمام swing points
        swing_highs = df[df['swing_high']]['high'].values
        swing_lows = df[df['swing_low']]['low'].values
        
        # محاسبه سطوح با cluster analysis
        resistance_levels = self._find_resistance_clusters(swing_highs)
        support_levels = self._find_support_clusters(swing_lows)
        
        for i in range(len(df)):
            current_price = df['close'].iloc[i]
            
            # پیدا کردن نزدیکترین resistance
            resistance_above = [level for level in resistance_levels if level['price'] > current_price]
            if resistance_above:
                nearest_resistance = min(resistance_above, key=lambda x: x['price'])
                df.iloc[i, df.columns.get_loc('resistance_level')] = nearest_resistance['price']
                df.iloc[i, df.columns.get_loc('level_strength')] = nearest_resistance['strength']
            
            # پیدا کردن نزدیکترین support
            support_below = [level for level in support_levels if level['price'] < current_price]
            if support_below:
                nearest_support = max(support_below, key=lambda x: x['price'])
                df.iloc[i, df.columns.get_loc('support_level')] = nearest_support['price']
        
        # محاسبه فاصله تا سطوح
        df['distance_to_resistance'] = (df['resistance_level'] - df['close']) / df['close']
        df['distance_to_support'] = (df['close'] - df['support_level']) / df['close']
        
        # پر کردن NaN ها
        df[['distance_to_resistance', 'distance_to_support']] = df[['distance_to_resistance', 'distance_to_support']].fillna(0)
        
        return df

    def _find_resistance_clusters(self, swing_highs: np.ndarray) -> List[Dict]:
        """پیدا کردن cluster های resistance"""
        if len(swing_highs) == 0:
            return []
            
        clusters = []
        tolerance = 0.02  # 2% tolerance
        
        for high in swing_highs:
            # پیدا کردن swing های نزدیک
            similar_highs = [h for h in swing_highs if abs(h - high) / high < tolerance]
            
            if len(similar_highs) >= 2:  # حداقل 2 تست
                clusters.append({
                    'price': np.mean(similar_highs),
                    'strength': len(similar_highs),
                    'touches': len(similar_highs)
                })
        
        # حذف تکراری‌ها
        unique_clusters = []
        for cluster in clusters:
            is_duplicate = False
            for existing in unique_clusters:
                if abs(cluster['price'] - existing['price']) / existing['price'] < tolerance:
                    if cluster['strength'] > existing['strength']:
                        unique_clusters.remove(existing)
                        unique_clusters.append(cluster)
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_clusters.append(cluster)
        
        return sorted(unique_clusters, key=lambda x: x['strength'], reverse=True)

    def _find_support_clusters(self, swing_lows: np.ndarray) -> List[Dict]:
        """پیدا کردن cluster های support"""
        if len(swing_lows) == 0:
            return []
            
        clusters = []
        tolerance = 0.02
        
        for low in swing_lows:
            similar_lows = [l for l in swing_lows if abs(l - low) / low < tolerance]
            
            if len(similar_lows) >= 2:
                clusters.append({
                    'price': np.mean(similar_lows),
                    'strength': len(similar_lows),
                    'touches': len(similar_lows)
                })
        
        # حذف تکراری‌ها
        unique_clusters = []
        for cluster in clusters:
            is_duplicate = False
            for existing in unique_clusters:
                if abs(cluster['price'] - existing['price']) / existing['price'] < tolerance:
                    if cluster['strength'] > existing['strength']:
                        unique_clusters.remove(existing)
                        unique_clusters.append(cluster)
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_clusters.append(cluster)
        
        return sorted(unique_clusters, key=lambda x: x['strength'], reverse=True)

    def _analyze_candle_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """تحلیل پیشرفته الگوهای کندل"""
        # خصوصیات کندل
        df['body'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
        df['total_range'] = df['high'] - df['low']
        df['body_ratio'] = df['body'] / df['total_range']
        df['body_ratio'] = df['body_ratio'].fillna(0)
        
        # الگوهای پایه
        df['is_bullish'] = df['close'] > df['open']
        df['is_bearish'] = df['close'] < df['open']
        df['is_doji'] = df['body_ratio'] < 0.1
        
        # الگوهای پیشرفته
        df = self._identify_advanced_patterns(df)
        
        return df

    def _identify_advanced_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """شناسایی الگوهای پیشرفته کندل"""
        # Hammer و Shooting Star
        df['is_hammer'] = ((df['lower_shadow'] > df['body'] * 2) & 
                          (df['upper_shadow'] < df['body'] * 0.5) &
                          (df['body_ratio'] > 0.1))
        
        df['is_shooting_star'] = ((df['upper_shadow'] > df['body'] * 2) & 
                                 (df['lower_shadow'] < df['body'] * 0.5) &
                                 (df['body_ratio'] > 0.1))
        
        # Pin Bar
        df['is_pin_bar'] = df['is_hammer'] | df['is_shooting_star']
        
        # Engulfing Pattern
        df['is_bullish_engulfing'] = False
        df['is_bearish_engulfing'] = False
        
        for i in range(1, len(df)):
            # Bullish Engulfing
            if (df['is_bullish'].iloc[i] and df['is_bearish'].iloc[i-1] and
                df['open'].iloc[i] < df['close'].iloc[i-1] and
                df['close'].iloc[i] > df['open'].iloc[i-1]):
                df.iloc[i, df.columns.get_loc('is_bullish_engulfing')] = True
            
            # Bearish Engulfing
            if (df['is_bearish'].iloc[i] and df['is_bullish'].iloc[i-1] and
                df['open'].iloc[i] > df['close'].iloc[i-1] and
                df['close'].iloc[i] < df['open'].iloc[i-1]):
                df.iloc[i, df.columns.get_loc('is_bearish_engulfing')] = True
        
        # Inside Bar و Outside Bar
        df['is_inside_bar'] = False
        df['is_outside_bar'] = False
        
        for i in range(1, len(df)):
            # Inside Bar
            if (df['high'].iloc[i] < df['high'].iloc[i-1] and
                df['low'].iloc[i] > df['low'].iloc[i-1]):
                df.iloc[i, df.columns.get_loc('is_inside_bar')] = True
            
            # Outside Bar
            if (df['high'].iloc[i] > df['high'].iloc[i-1] and
                df['low'].iloc[i] < df['low'].iloc[i-1]):
                df.iloc[i, df.columns.get_loc('is_outside_bar')] = True
        
        return df

    def _detect_trend_changes(self, df: pd.DataFrame) -> pd.DataFrame:
        """تشخیص تغییرات ترند"""
        df['trend_change_signal'] = 0
        df['trend_change_strength'] = 0.0
        
        for i in range(2, len(df)):
            current_structure = df['market_structure'].iloc[i]
            prev_structure = df['market_structure'].iloc[i-1]
            
            # تغییر از نزولی به صعودی
            if (prev_structure in [MarketStructure.STRONG_DOWNTREND.value, MarketStructure.WEAK_DOWNTREND.value] and
                current_structure in [MarketStructure.STRONG_UPTREND.value, MarketStructure.WEAK_UPTREND.value]):
                df.iloc[i, df.columns.get_loc('trend_change_signal')] = 1
                df.iloc[i, df.columns.get_loc('trend_change_strength')] = self._calculate_trend_change_strength(df, i)
            
            # تغییر از صعودی به نزولی
            elif (prev_structure in [MarketStructure.STRONG_UPTREND.value, MarketStructure.WEAK_UPTREND.value] and
                  current_structure in [MarketStructure.STRONG_DOWNTREND.value, MarketStructure.WEAK_DOWNTREND.value]):
                df.iloc[i, df.columns.get_loc('trend_change_signal')] = -1
                df.iloc[i, df.columns.get_loc('trend_change_strength')] = self._calculate_trend_change_strength(df, i)
        
        return df

    def _calculate_trend_change_strength(self, df: pd.DataFrame, index: int) -> float:
        """محاسبه قدرت تغییر ترند"""
        try:
            # فاکتورهای مختلف برای محاسبه قدرت
            volume_factor = df['volume'].iloc[index] / df['volume'].rolling(20).mean().iloc[index]
            adx_factor = df['adx'].iloc[index] / 100
            pattern_factor = 1.0
            
            # اگر الگوی کندل مهم وجود دارد
            if (df['is_bullish_engulfing'].iloc[index] or 
                df['is_bearish_engulfing'].iloc[index] or
                df['is_pin_bar'].iloc[index]):
                pattern_factor = 1.5
            
            strength = min(1.0, (volume_factor * 0.4 + adx_factor * 0.4 + pattern_factor * 0.2))
            return strength
            
        except:
            return 0.5

    def _analyze_volume_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """تحلیل بافت حجم پیشرفته"""
        period = self.volume_analysis_period
        
        # میانگین‌های حجم
        df['volume_sma'] = df['volume'].rolling(window=period).mean()
        df['volume_ema'] = df['volume'].ewm(span=period).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Volume Profile
        df['high_volume'] = df['volume_ratio'] > 1.5
        df['low_volume'] = df['volume_ratio'] < 0.7
        df['average_volume'] = (df['volume_ratio'] >= 0.7) & (df['volume_ratio'] <= 1.5)
        
        # On Balance Volume (OBV)
        df['obv'] = 0.0
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                df.iloc[i, df.columns.get_loc('obv')] = df['obv'].iloc[i-1] + df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                df.iloc[i, df.columns.get_loc('obv')] = df['obv'].iloc[i-1] - df['volume'].iloc[i]
            else:
                df.iloc[i, df.columns.get_loc('obv')] = df['obv'].iloc[i-1]
        
        # Volume Trend
        df['obv_trend'] = df['obv'].rolling(window=10).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1)
        
        # Buying/Selling Pressure
        df['buying_pressure'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        df['selling_pressure'] = (df['high'] - df['close']) / (df['high'] - df['low'])
        
        # Volume Price Trend (VPT)
        df['vpt'] = 0.0
        for i in range(1, len(df)):
            price_change = (df['close'].iloc[i] - df['close'].iloc[i-1]) / df['close'].iloc[i-1]
            df.iloc[i, df.columns.get_loc('vpt')] = df['vpt'].iloc[i-1] + (df['volume'].iloc[i] * price_change)
        
        return df

    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """محاسبه اندیکاتورهای مومنتوم"""
        # RSI
        df = self._calculate_rsi(df)
        
        # MACD
        df = self._calculate_macd(df)
        
        # Stochastic
        df = self._calculate_stochastic(df)
        
        return df

    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """محاسبه RSI"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df

    def _calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """محاسبه MACD"""
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        return df

    def _calculate_stochastic(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """محاسبه Stochastic"""
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()
        
        df['stoch_k'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        return df

    def _generate_comprehensive_signal(self, df: pd.DataFrame, symbol: str = None) -> Dict:
        """تولید سیگنال جامع"""
        try:
            latest = df.iloc[-1]
            recent_data = df.tail(20)
            
            # تحلیل شرایط فعلی
            market_analysis = self._analyze_current_market_conditions(latest, recent_data)
            
            # شناسایی فرصت‌های معاملاتی
            trading_opportunities = self._identify_trading_opportunities(latest, recent_data)
            
            # محاسبه سطوح ورود و خروج
            entry_exit_levels = self._calculate_entry_exit_levels(latest, recent_data)
            
            # ارزیابی کیفیت سیگنال
            signal_quality = self._assess_signal_quality(latest, recent_data, trading_opportunities)
            
            # تعیین نوع سیگنال
            signal_type = self._determine_signal_type(latest, recent_data, trading_opportunities)
            
            return {
                'symbol': symbol,
                'timestamp': datetime.datetime.now(),
                'current_price': float(latest['close']),
                'market_structure': latest['market_structure'],
                'trend_strength': latest['trend_strength'],
                'signal': trading_opportunities.get('signal', 0),
                'signal_type': signal_type,
                'signal_quality': signal_quality['quality'],
                'confidence': signal_quality['confidence'],
                'reasoning': self._generate_reasoning(latest, recent_data, trading_opportunities),
                'entry_price': entry_exit_levels.get('entry_price', float(latest['close'])),
                'stop_loss': entry_exit_levels.get('stop_loss', 0),
                'take_profits': entry_exit_levels.get('take_profits', []),
                'risk_reward_ratio': entry_exit_levels.get('risk_reward_ratio', 0),
                'volume_confirmation': signal_quality.get('volume_confirmation', False),
                'market_context': market_analysis,
                'technical_analysis': {
                    'rsi': float(latest.get('rsi', 50)),
                    'macd': float(latest.get('macd', 0)),
                    'adx': float(latest.get('adx', 25)),
                    'support_level': float(latest.get('support_level', 0)),
                    'resistance_level': float(latest.get('resistance_level', 0)),
                    'distance_to_support': float(latest.get('distance_to_support', 0)),
                    'distance_to_resistance': float(latest.get('distance_to_resistance', 0))
                }
            }
            
        except Exception as e:
            print(f"خطا در تولید سیگنال جامع: {e}")
            return None

    def _analyze_current_market_conditions(self, latest: pd.Series, recent_data: pd.DataFrame) -> Dict:
        """تحلیل شرایط فعلی بازار"""
        return {
            'market_phase': latest['market_structure'],
            'trend_direction': 'BULLISH' if latest['ema_21'] > latest['ema_50'] else 'BEARISH',
            'volatility_level': 'HIGH' if latest.get('atr', 0) > recent_data['atr'].mean() * 1.5 else 'NORMAL',
            'volume_condition': 'HIGH' if latest.get('volume_ratio', 1) > 1.5 else 'NORMAL',
            'momentum_status': self._assess_momentum(latest),
            'key_level_proximity': self._assess_key_level_proximity(latest)
        }

    def _identify_trading_opportunities(self, latest: pd.Series, recent_data: pd.DataFrame) -> Dict:
        """شناسایی فرصت‌های معاملاتی"""
        signal = 0
        signal_strength = 0.0
        opportunities = []
        
        # بررسی breakout
        if self._is_breakout_opportunity(latest, recent_data):
            if latest['close'] > latest['resistance_level']:
                signal = 1
                signal_strength = 0.8
                opportunities.append('RESISTANCE_BREAKOUT')
            elif latest['close'] < latest['support_level']:
                signal = -1
                signal_strength = 0.8
                opportunities.append('SUPPORT_BREAKDOWN')
        
        # بررسی pullback
        elif self._is_pullback_opportunity(latest, recent_data):
            if latest['market_structure'] in ['STRONG_UPTREND', 'WEAK_UPTREND']:
                signal = 1
                signal_strength = 0.7
                opportunities.append('BULLISH_PULLBACK')
            elif latest['market_structure'] in ['STRONG_DOWNTREND', 'WEAK_DOWNTREND']:
                signal = -1
                signal_strength = 0.7
                opportunities.append('BEARISH_PULLBACK')
        
        # بررسی reversal
        elif self._is_reversal_opportunity(latest, recent_data):
            if latest.get('trend_change_signal', 0) == 1:
                signal = 1
                signal_strength = 0.6
                opportunities.append('BULLISH_REVERSAL')
            elif latest.get('trend_change_signal', 0) == -1:
                signal = -1
                signal_strength = 0.6
                opportunities.append('BEARISH_REVERSAL')
        
        return {
            'signal': signal,
            'signal_strength': signal_strength,
            'opportunities': opportunities
        }

    def _is_breakout_opportunity(self, latest: pd.Series, recent_data: pd.DataFrame) -> bool:
        """بررسی فرصت breakout"""
        # بررسی نزدیکی به سطوح کلیدی
        near_resistance = latest.get('distance_to_resistance', 1) < 0.01
        near_support = latest.get('distance_to_support', 1) < 0.01
        
        # بررسی حجم
        high_volume = latest.get('volume_ratio', 1) > 1.5
        
        # بررسی قدرت ترند
        strong_trend = latest.get('adx', 0) > 25
        
        return (near_resistance or near_support) and high_volume and strong_trend

    def _is_pullback_opportunity(self, latest: pd.Series, recent_data: pd.DataFrame) -> bool:
        """بررسی فرصت pullback"""
        # بازگشت به EMA در ترند قوی
        uptrend = latest['market_structure'] in ['STRONG_UPTREND', 'WEAK_UPTREND']
        downtrend = latest['market_structure'] in ['STRONG_DOWNTREND', 'WEAK_DOWNTREND']
        
        near_ema21 = abs(latest['close'] - latest['ema_21']) / latest['close'] < 0.02
        
        return (uptrend or downtrend) and near_ema21

    def _is_reversal_opportunity(self, latest: pd.Series, recent_data: pd.DataFrame) -> bool:
        """بررسی فرصت reversal"""
        # الگوهای کندل reversal
        reversal_pattern = (latest.get('is_bullish_engulfing', False) or 
                           latest.get('is_bearish_engulfing', False) or
                           latest.get('is_pin_bar', False))
        
        # divergence در RSI
        rsi_extreme = latest.get('rsi', 50) > 70 or latest.get('rsi', 50) < 30
        
        return reversal_pattern and rsi_extreme

    def _calculate_entry_exit_levels(self, latest: pd.Series, recent_data: pd.DataFrame) -> Dict:
        """محاسبه سطوح ورود و خروج"""
        current_price = latest['close']
        atr = latest.get('atr', current_price * 0.02)
        
        # تعیین سطوح بر اساس نوع سیگنال
        if latest.get('distance_to_resistance', 1) < 0.01:  # نزدیک resistance
            entry_price = current_price
            stop_loss = current_price - (atr * 2)
            take_profit_1 = latest.get('resistance_level', current_price) * 1.01
            take_profit_2 = latest.get('resistance_level', current_price) * 1.03
            
        elif latest.get('distance_to_support', 1) < 0.01:  # نزدیک support
            entry_price = current_price
            stop_loss = current_price + (atr * 2)
            take_profit_1 = latest.get('support_level', current_price) * 0.99
            take_profit_2 = latest.get('support_level', current_price) * 0.97
            
        else:  # سایر شرایط
            entry_price = current_price
            stop_loss = current_price - (atr * 1.5) if latest.get('signal', 0) > 0 else current_price + (atr * 1.5)
            take_profit_1 = current_price + (atr * 2) if latest.get('signal', 0) > 0 else current_price - (atr * 2)
            take_profit_2 = current_price + (atr * 3) if latest.get('signal', 0) > 0 else current_price - (atr * 3)
        
        # محاسبه risk/reward
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit_1 - entry_price)
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
        return {
            'entry_price': float(entry_price),
            'stop_loss': float(stop_loss),
            'take_profits': [float(take_profit_1), float(take_profit_2)],
            'risk_reward_ratio': risk_reward_ratio
        }

    def _assess_signal_quality(self, latest: pd.Series, recent_data: pd.DataFrame, opportunities: Dict) -> Dict:
        """ارزیابی کیفیت سیگنال"""
        confidence_factors = []
        
        # قدرت سیگنال
        signal_strength = opportunities.get('signal_strength', 0)
        confidence_factors.append(signal_strength)
        
        # تأیید حجم
        volume_confirmation = latest.get('volume_ratio', 1) > 1.2
        if volume_confirmation:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.4)
        
        # قدرت ترند
        trend_strength = latest.get('adx', 0) / 100
        confidence_factors.append(trend_strength)
        
        # نزدیکی به سطوح کلیدی
        level_proximity = min(latest.get('distance_to_resistance', 1), 
                             latest.get('distance_to_support', 1))
        proximity_score = 1 - min(level_proximity * 10, 1)  # هرچه نزدیک‌تر، امتیاز بهتر
        confidence_factors.append(proximity_score)
        
        # محاسبه confidence کلی
        overall_confidence = np.mean(confidence_factors)
        
        # تعیین کیفیت
        if overall_confidence >= 0.8:
            quality = SignalQuality.EXCELLENT.value
        elif overall_confidence >= 0.7:
            quality = SignalQuality.HIGH.value
        elif overall_confidence >= 0.6:
            quality = SignalQuality.MEDIUM.value
        elif overall_confidence >= 0.4:
            quality = SignalQuality.LOW.value
        else:
            quality = SignalQuality.INVALID.value
        
        return {
            'quality': quality,
            'confidence': overall_confidence,
            'volume_confirmation': volume_confirmation,
            'confidence_factors': {
                'signal_strength': signal_strength,
                'volume_confirmation': volume_confirmation,
                'trend_strength': trend_strength,
                'level_proximity': proximity_score
            }
        }

    def _determine_signal_type(self, latest: pd.Series, recent_data: pd.DataFrame, opportunities: Dict) -> str:
        """تعیین نوع سیگنال"""
        opportunity_types = opportunities.get('opportunities', [])
        
        if any('BREAKOUT' in opp for opp in opportunity_types):
            return SignalType.BREAKOUT.value
        elif any('PULLBACK' in opp for opp in opportunity_types):
            return SignalType.PULLBACK.value
        elif any('REVERSAL' in opp for opp in opportunity_types):
            return SignalType.REVERSAL.value
        elif latest['market_structure'] != 'CONSOLIDATION':
            return SignalType.CONTINUATION.value
        else:
            return SignalType.RANGE_TRADE.value

    def _generate_reasoning(self, latest: pd.Series, recent_data: pd.DataFrame, opportunities: Dict) -> str:
        """تولید توضیح سیگنال"""
        reasoning_parts = []
        
        # ساختار بازار
        reasoning_parts.append(f"ساختار بازار: {latest['market_structure']}")
        
        # قدرت ترند
        reasoning_parts.append(f"قدرت ترند: {latest['trend_strength']} (ADX: {latest.get('adx', 0):.1f})")
        
        # موقعیت نسبت به سطوح کلیدی
        if latest.get('distance_to_resistance', 1) < 0.02:
            reasoning_parts.append(f"نزدیک resistance: {latest.get('resistance_level', 0):.4f}")
        if latest.get('distance_to_support', 1) < 0.02:
            reasoning_parts.append(f"نزدیک support: {latest.get('support_level', 0):.4f}")
        
        # شرایط حجم
        volume_ratio = latest.get('volume_ratio', 1)
        if volume_ratio > 1.5:
            reasoning_parts.append("حجم بالا")
        elif volume_ratio < 0.7:
            reasoning_parts.append("حجم پایین")
        
        # فرصت‌های شناسایی شده
        if opportunities.get('opportunities'):
            reasoning_parts.append(f"فرصت: {', '.join(opportunities['opportunities'])}")
        
        return " | ".join(reasoning_parts)

    def _assess_momentum(self, latest: pd.Series) -> str:
        """ارزیابی مومنتوم"""
        rsi = latest.get('rsi', 50)
        macd = latest.get('macd', 0)
        
        if rsi > 60 and macd > 0:
            return 'BULLISH'
        elif rsi < 40 and macd < 0:
            return 'BEARISH'
        else:
            return 'NEUTRAL'

    def _assess_key_level_proximity(self, latest: pd.Series) -> str:
        """ارزیابی نزدیکی به سطوح کلیدی"""
        dist_resistance = latest.get('distance_to_resistance', 1)
        dist_support = latest.get('distance_to_support', 1)
        
        if min(dist_resistance, dist_support) < 0.01:
            return 'VERY_CLOSE'
        elif min(dist_resistance, dist_support) < 0.02:
            return 'CLOSE'
        else:
            return 'AWAY'

    def get_analyzer_info(self) -> Dict:
        """اطلاعات تحلیلگر"""
        return {
            'name': self.name,
            'version': '2.0',
            'min_candles_required': self.min_candles,
            'features': [
                'Advanced Market Structure Analysis',
                'Multi-timeframe Trend Detection',
                'Key Level Clustering',
                'Pattern Recognition',
                'Volume Analysis',
                'Momentum Indicators',
                'Signal Quality Assessment'
            ],
            'signal_types': [e.value for e in SignalType],
            'quality_levels': [e.value for e in SignalQuality]
        }
