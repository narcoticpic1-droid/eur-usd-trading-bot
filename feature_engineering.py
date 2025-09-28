# models/feature_engineering.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import talib
from utils.logger import setup_logger
import config

class ForexFeatureEngineer:
    """
    مهندسی ویژگی برای داده‌های Forex
    """
    
    def __init__(self):
        self.logger = setup_logger('feature_engineer')
        self.feature_names = []
    
    def create_features(self, df: pd.DataFrame, pair: str) -> pd.DataFrame:
        """
        ایجاد ویژگی‌های کامل برای تحلیل Forex
        """
        try:
            self.logger.info(f"شروع مهندسی ویژگی برای {pair}")
            
            # کپی داده‌ها
            features = df.copy()
            
            # ویژگی‌های قیمت پایه
            features = self._add_price_features(features)
            
            # ویژگی‌های تکنیکال
            features = self._add_technical_features(features)
            
            # ویژگی‌های زمانی
            features = self._add_time_features(features)
            
            # ویژگی‌های نوسانات
            features = self._add_volatility_features(features)
            
            # ویژگی‌های مومنتوم
            features = self._add_momentum_features(features)
            
            # ویژگی‌های ساختار بازار
            features = self._add_market_structure_features(features)
            
            # ویژگی‌های مخصوص Forex
            features = self._add_forex_specific_features(features, pair)
            
            # پاکسازی NaN ها
            features = features.dropna()
            
            self.logger.info(f"مهندسی ویژگی کامل شد: {len(features.columns)} ویژگی")
            return features
            
        except Exception as e:
            self.logger.error(f"خطا در مهندسی ویژگی: {e}")
            return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ویژگی‌های قیمت پایه"""
        # قیمت میانگین
        df['hl2'] = (df['high'] + df['low']) / 2
        df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
        df['ohlc4'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        
        # محدوده قیمت
        df['range'] = df['high'] - df['low']
        df['range_pct'] = df['range'] / df['close']
        
        # بدنه کندل
        df['body'] = abs(df['close'] - df['open'])
        df['body_pct'] = df['body'] / df['range']
        
        # سایه‌ها
        df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
        
        # نسبت‌ها
        df['upper_shadow_pct'] = df['upper_shadow'] / df['range']
        df['lower_shadow_pct'] = df['lower_shadow'] / df['range']
        
        return df
    
    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ویژگی‌های تکنیکال کلاسیک"""
        # میانگین‌های متحرک
        for period in [8, 21, 50, 200]:
            df[f'sma_{period}'] = talib.SMA(df['close'], timeperiod=period)
            df[f'ema_{period}'] = talib.EMA(df['close'], timeperiod=period)
            
            # فاصله از میانگین
            df[f'close_sma_{period}_dist'] = (df['close'] - df[f'sma_{period}']) / df['close']
            df[f'close_ema_{period}_dist'] = (df['close'] - df[f'ema_{period}']) / df['close']
        
        # RSI
        df['rsi_14'] = talib.RSI(df['close'], timeperiod=14)
        df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
        df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(df['close'])
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_histogram'] = macd_hist
        df['macd_bullish'] = (df['macd'] > df['macd_signal']).astype(int)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'])
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # ATR
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['atr_pct'] = df['atr'] / df['close']
        
        # ADX
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        df['adx_strong_trend'] = (df['adx'] > 25).astype(int)
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ویژگی‌های زمانی"""
        # ساعت روز (UTC)
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        
        # سشن‌های تجاری
        df['asian_session'] = ((df['hour'] >= 23) | (df['hour'] < 8)).astype(int)
        df['london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['new_york_session'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)
        df['overlap_london_ny'] = ((df['hour'] >= 13) & (df['hour'] < 16)).astype(int)
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ویژگی‌های نوسانات"""
        # نوسانات تاریخی
        for period in [5, 10, 20]:
            returns = df['close'].pct_change()
            df[f'volatility_{period}'] = returns.rolling(period).std() * np.sqrt(period)
            
        # نوسانات نرمال شده
        df['normalized_range'] = df['range'] / df['atr']
        
        # پیک‌های نوسانات
        df['high_volatility'] = (df['volatility_20'] > df['volatility_20'].rolling(50).quantile(0.8)).astype(int)
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ویژگی‌های مومنتوم"""
        # بازده‌ها
        for period in [1, 5, 10, 20]:
            df[f'return_{period}'] = df['close'].pct_change(period)
            
        # Stochastic
        stoch_k, stoch_d = talib.STOCH(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d
        df['stoch_overbought'] = (stoch_k > 80).astype(int)
        df['stoch_oversold'] = (stoch_k < 20).astype(int)
        
        # Williams %R
        df['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'])
        
        return df
    
    def _add_market_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ویژگی‌های ساختار بازار"""
        # Swing Points
        df['swing_high'] = self._identify_swing_highs(df['high'])
        df['swing_low'] = self._identify_swing_lows(df['low'])
        
        # ترند
        df['uptrend'] = (df['ema_21'] > df['ema_50']).astype(int)
        df['strong_uptrend'] = ((df['ema_8'] > df['ema_21']) & 
                               (df['ema_21'] > df['ema_50']) & 
                               (df['ema_50'] > df['ema_200'])).astype(int)
        
        # سطوح حمایت و مقاومت
        df['resistance_level'] = self._calculate_resistance_levels(df)
        df['support_level'] = self._calculate_support_levels(df)
        
        return df
    
    def _add_forex_specific_features(self, df: pd.DataFrame, pair: str) -> pd.DataFrame:
        """ویژگی‌های مخصوص Forex"""
        # تنظیمات مخصوص جفت ارز
        pair_config = config.PAIR_SPECIFIC_SETTINGS.get(pair, {})
        pip_value = pair_config.get('pip_value', 0.0001)
        
        # محاسبه pip
        df['range_pips'] = df['range'] / pip_value
        df['body_pips'] = df['body'] / pip_value
        
        # الگوهای کندل Forex
        df['doji'] = (df['body_pct'] < 0.1).astype(int)
        df['hammer'] = ((df['lower_shadow'] > 2 * df['body']) & 
                       (df['upper_shadow'] < df['body'])).astype(int)
        df['shooting_star'] = ((df['upper_shadow'] > 2 * df['body']) & 
                              (df['lower_shadow'] < df['body'])).astype(int)
        
        return df
    
    def _identify_swing_highs(self, high_series: pd.Series, window: int = 5) -> pd.Series:
        """شناسایی نقاط swing high"""
        swing_highs = pd.Series(0, index=high_series.index)
        
        for i in range(window, len(high_series) - window):
            if high_series.iloc[i] == high_series.iloc[i-window:i+window+1].max():
                swing_highs.iloc[i] = 1
                
        return swing_highs
    
    def _identify_swing_lows(self, low_series: pd.Series, window: int = 5) -> pd.Series:
        """شناسایی نقاط swing low"""
        swing_lows = pd.Series(0, index=low_series.index)
        
        for i in range(window, len(low_series) - window):
            if low_series.iloc[i] == low_series.iloc[i-window:i+window+1].min():
                swing_lows.iloc[i] = 1
                
        return swing_lows
    
    def _calculate_resistance_levels(self, df: pd.DataFrame) -> pd.Series:
        """محاسبه سطوح مقاومت"""
        # پیاده‌سازی ساده - می‌تواند پیچیده‌تر شود
        return df['high'].rolling(20).max()
    
    def _calculate_support_levels(self, df: pd.DataFrame) -> pd.Series:
        """محاسبه سطوح حمایت"""
        # پیاده‌سازی ساده - می‌تواند پیچیده‌تر شود
        return df['low'].rolling(20).min()
    
    def get_feature_importance(self, features: pd.DataFrame, target: pd.Series) -> Dict:
        """محاسبه اهمیت ویژگی‌ها"""
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.feature_selection import mutual_info_regression
            
            # حذف ویژگی‌های غیرعددی
            numeric_features = features.select_dtypes(include=[np.number])
            
            if len(numeric_features.columns) == 0:
                return {}
            
            # Random Forest
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(numeric_features.fillna(0), target.fillna(0))
            
            rf_importance = dict(zip(numeric_features.columns, rf.feature_importances_))
            
            # Mutual Information
            mi_scores = mutual_info_regression(numeric_features.fillna(0), target.fillna(0))
            mi_importance = dict(zip(numeric_features.columns, mi_scores))
            
            return {
                'random_forest': rf_importance,
                'mutual_information': mi_importance
            }
            
        except Exception as e:
            self.logger.error(f"خطا در محاسبه اهمیت ویژگی: {e}")
            return {}
