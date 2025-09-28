import pandas as pd
import numpy as np
from typing import Dict, Optional
from .base_strategy import BaseStrategy, StrategyType, StrategySignal
import talib

class ReversalStrategy(BaseStrategy):
    """
    استراتژی برگشت روند - شناسایی نقاط برگشت و انتهای ترند
    """
    
    def __init__(self):
        super().__init__("Reversal Strategy", StrategyType.REVERSAL)
        
        # پارامترهای RSI
        self.rsi_period = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        
        # پارامترهای Stochastic
        self.stoch_k = 14
        self.stoch_d = 3
        
        # پارامترهای حجم
        self.volume_ma_period = 20
        self.min_volume_ratio = 1.5
        
        # پارامترهای الگوی کندل
        self.min_candle_body_ratio = 0.6
        
        # تنظیمات divergence
        self.lookback_periods = 20

    def analyze(self, df: pd.DataFrame, symbol: str) -> Optional[Dict]:
        """تحلیل اصلی استراتژی برگشت"""
        if not self.validate_data(df):
            return None
        
        df = df.copy()
        
        # محاسبه اندیکاتورها
        df = self._calculate_reversal_indicators(df)
        
        # تشخیص الگوهای برگشت
        reversal_signals = self._detect_reversal_patterns(df)
        
        if not reversal_signals['found']:
            return None
        
        # تحلیل divergence
        divergence_analysis = self._analyze_divergence(df)
        
        # ترکیب سیگنال‌ها
        final_analysis = self._combine_reversal_signals(
            df, reversal_signals, divergence_analysis
        )
        
        if not final_analysis['valid']:
            return None
        
        return {
            'strategy': self.name,
            'symbol': symbol,
            'signal': final_analysis['direction'],
            'signal_type': 'REVERSAL',
            'confidence': final_analysis['confidence'],
            'current_price': df['close'].iloc[-1],
            'entry_price': final_analysis['entry_price'],
            'stop_loss': final_analysis['stop_loss'],
            'take_profits': final_analysis['take_profits'],
            'risk_reward_ratio': final_analysis['risk_reward'],
            'reasoning': final_analysis['reasoning'],
            'market_context': {
                'reversal_type': final_analysis['reversal_type'],
                'rsi_level': df['rsi'].iloc[-1],
                'volume_confirmation': final_analysis['volume_confirmation'],
                'divergence_present': divergence_analysis['found']
            }
        }

    def _calculate_reversal_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """محاسبه اندیکاتورهای برگشت"""
        # RSI
        df['rsi'] = talib.RSI(df['close'], timeperiod=self.rsi_period)
        
        # Stochastic
        df['stoch_k'], df['stoch_d'] = talib.STOCH(
            df['high'], df['low'], df['close'],
            fastk_period=self.stoch_k, slowk_period=self.stoch_d, slowd_period=self.stoch_d
        )
        
        # Williams %R
        df['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # CCI (Commodity Channel Index)
        df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=20)
        
        # حجم
        df['volume_ma'] = df['volume'].rolling(window=self.volume_ma_period).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # ATR برای stop loss
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        return df

    def _detect_reversal_patterns(self, df: pd.DataFrame) -> Dict:
        """تشخیص الگوهای برگشت"""
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        signals = {
            'found': False,
            'direction': 0,
            'patterns': [],
            'strength': 0.0
        }
        
        # بررسی RSI oversold/overbought
        if latest['rsi'] <= self.rsi_oversold:
            signals['patterns'].append('RSI_OVERSOLD')
            signals['direction'] = 1  # احتمال صعود
            signals['strength'] += 0.3
        elif latest['rsi'] >= self.rsi_overbought:
            signals['patterns'].append('RSI_OVERBOUGHT')
            signals['direction'] = -1  # احتمال نزول
            signals['strength'] += 0.3
        
        # بررسی Stochastic
        if latest['stoch_k'] <= 20 and latest['stoch_d'] <= 20:
            if latest['stoch_k'] > prev['stoch_k']:  # شروع برگشت
                signals['patterns'].append('STOCH_OVERSOLD_REVERSAL')
                if signals['direction'] == 0:
                    signals['direction'] = 1
                signals['strength'] += 0.2
        elif latest['stoch_k'] >= 80 and latest['stoch_d'] >= 80:
            if latest['stoch_k'] < prev['stoch_k']:  # شروع برگشت
                signals['patterns'].append('STOCH_OVERBOUGHT_REVERSAL')
                if signals['direction'] == 0:
                    signals['direction'] = -1
                signals['strength'] += 0.2
        
        # بررسی الگوهای کندل برگشتی
        candle_patterns = self._identify_reversal_candlestick_patterns(df)
        if candle_patterns['found']:
            signals['patterns'].extend(candle_patterns['patterns'])
            signals['strength'] += candle_patterns['strength']
            if signals['direction'] == 0:
                signals['direction'] = candle_patterns['direction']
        
        signals['found'] = signals['strength'] >= 0.4
        return signals

    def _identify_reversal_candlestick_patterns(self, df: pd.DataFrame) -> Dict:
        """شناسایی الگوهای کندل برگشتی"""
        patterns = {
            'found': False,
            'patterns': [],
            'direction': 0,
            'strength': 0.0
        }
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # محاسبه خصوصیات کندل
        body = abs(latest['close'] - latest['open'])
        total_range = latest['high'] - latest['low']
        upper_shadow = latest['high'] - max(latest['open'], latest['close'])
        lower_shadow = min(latest['open'], latest['close']) - latest['low']
        
        if total_range == 0:
            return patterns
        
        body_ratio = body / total_range
        
        # Hammer (چکش) - نشانه برگشت صعودی
        if (lower_shadow > body * 2 and 
            upper_shadow < body * 0.5 and 
            latest['close'] < prev['close']):
            patterns['patterns'].append('HAMMER')
            patterns['direction'] = 1
            patterns['strength'] += 0.25
            patterns['found'] = True
        
        # Shooting Star - نشانه برگشت نزولی
        elif (upper_shadow > body * 2 and 
              lower_shadow < body * 0.5 and 
              latest['close'] > prev['close']):
            patterns['patterns'].append('SHOOTING_STAR')
            patterns['direction'] = -1
            patterns['strength'] += 0.25
            patterns['found'] = True
        
        # Doji - عدم قطعیت و احتمال برگشت
        elif body_ratio < 0.1:
            patterns['patterns'].append('DOJI')
            patterns['strength'] += 0.15
            patterns['found'] = True
        
        # Engulfing Pattern
        if len(df) >= 2:
            prev_body = abs(prev['close'] - prev['open'])
            
            # Bullish Engulfing
            if (latest['close'] > latest['open'] and  # کندل سبز
                prev['close'] < prev['open'] and      # کندل قبلی قرمز
                latest['close'] > prev['open'] and    # بسته کردن بالاتر از باز شدن قبلی
                latest['open'] < prev['close'] and    # باز شدن پایین‌تر از بسته شدن قبلی
                body > prev_body):                    # بدنه بزرگ‌تر
                patterns['patterns'].append('BULLISH_ENGULFING')
                patterns['direction'] = 1
                patterns['strength'] += 0.3
                patterns['found'] = True
            
            # Bearish Engulfing
            elif (latest['close'] < latest['open'] and  # کندل قرمز
                  prev['close'] > prev['open'] and      # کندل قبلی سبز
                  latest['close'] < prev['open'] and    # بسته شدن پایین‌تر از باز شدن قبلی
                  latest['open'] > prev['close'] and    # باز شدن بالاتر از بسته شدن قبلی
                  body > prev_body):                    # بدنه بزرگ‌تر
                patterns['patterns'].append('BEARISH_ENGULFING')
                patterns['direction'] = -1
                patterns['strength'] += 0.3
                patterns['found'] = True
        
        return patterns

    def _analyze_divergence(self, df: pd.DataFrame) -> Dict:
        """تحلیل واگرایی (Divergence) بین قیمت و اندیکاتورها"""
        divergence = {
            'found': False,
            'type': None,
            'strength': 0.0,
            'indicators': []
        }
        
        if len(df) < self.lookback_periods:
            return divergence
        
        # بررسی واگرایی RSI
        rsi_div = self._check_rsi_divergence(df)
        if rsi_div['found']:
            divergence['found'] = True
            divergence['type'] = rsi_div['type']
            divergence['strength'] += 0.3
            divergence['indicators'].append('RSI')
        
        # بررسی واگرایی MACD
        df['macd'], df['macdsignal'], df['macdhist'] = talib.MACD(df['close'])
        macd_div = self._check_macd_divergence(df)
        if macd_div['found']:
            divergence['found'] = True
            if divergence['type'] is None:
                divergence['type'] = macd_div['type']
            divergence['strength'] += 0.25
            divergence['indicators'].append('MACD')
        
        return divergence

    def _check_rsi_divergence(self, df: pd.DataFrame) -> Dict:
        """بررسی واگرایی RSI"""
        recent_data = df.tail(self.lookback_periods)
        
        # پیدا کردن نقاط بالا و پایین قیمت
        price_highs = recent_data[recent_data['high'] == recent_data['high'].rolling(5).max()]
        price_lows = recent_data[recent_data['low'] == recent_data['low'].rolling(5).min()]
        
        divergence = {'found': False, 'type': None}
        
        # بررسی Bullish Divergence (قیمت پایین‌تر، RSI بالاتر)
        if len(price_lows) >= 2:
            last_two_lows = price_lows.tail(2)
            if (last_two_lows['low'].iloc[-1] < last_two_lows['low'].iloc[-2] and
                last_two_lows['rsi'].iloc[-1] > last_two_lows['rsi'].iloc[-2]):
                divergence['found'] = True
                divergence['type'] = 'BULLISH'
        
        # بررسی Bearish Divergence (قیمت بالاتر، RSI پایین‌تر)
        if len(price_highs) >= 2:
            last_two_highs = price_highs.tail(2)
            if (last_two_highs['high'].iloc[-1] > last_two_highs['high'].iloc[-2] and
                last_two_highs['rsi'].iloc[-1] < last_two_highs['rsi'].iloc[-2]):
                divergence['found'] = True
                divergence['type'] = 'BEARISH'
        
        return divergence

    def _check_macd_divergence(self, df: pd.DataFrame) -> Dict:
        """بررسی واگرایی MACD"""
        # پیاده‌سازی مشابه RSI divergence
        return {'found': False, 'type': None}

    def _combine_reversal_signals(self, df: pd.DataFrame, reversal_signals: Dict, 
                                 divergence_analysis: Dict) -> Dict:
        """ترکیب سیگنال‌های برگشت"""
        latest = df.iloc[-1]
        
        confidence = reversal_signals['strength']
        reasoning = []
        
        # اضافه کردن قدرت از divergence
        if divergence_analysis['found']:
            confidence += divergence_analysis['strength']
            reasoning.append(f"واگرایی {' و '.join(divergence_analysis['indicators'])}")
        
        # بررسی تأیید حجم
        volume_confirmation = latest['volume_ratio'] >= self.min_volume_ratio
        if volume_confirmation:
            confidence += 0.15
            reasoning.append("تأیید حجم معاملات")
        
        # اضافه کردن الگوهای شناسایی شده
        reasoning.extend(reversal_signals['patterns'])
        
        # محاسبه نقاط ورود و خروج
        current_price = latest['close']
        atr = latest['atr']
        direction = reversal_signals['direction']
        
        if direction > 0:  # صعودی
            entry_price = current_price
            stop_loss = current_price - (atr * 1.5)
            take_profit_1 = current_price + (atr * 2.5)
            take_profit_2 = current_price + (atr * 4.0)
            reversal_type = "BULLISH_REVERSAL"
        else:  # نزولی
            entry_price = current_price
            stop_loss = current_price + (atr * 1.5)
            take_profit_1 = current_price - (atr * 2.5)
            take_profit_2 = current_price - (atr * 4.0)
            reversal_type = "BEARISH_REVERSAL"
        
        risk_reward = abs(take_profit_1 - entry_price) / abs(entry_price - stop_loss)
        
        return {
            'valid': confidence >= self.adaptive_params['confidence_threshold'],
            'direction': direction,
            'confidence': min(1.0, confidence),
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profits': [take_profit_1, take_profit_2],
            'risk_reward': risk_reward,
            'reversal_type': reversal_type,
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
        """بررسی مناسب بودن بازار برای استراتژی برگشت"""
        trend_strength = market_context.get('trend_strength', 0.5)
        volatility = market_context.get('volatility', 1.0)
        
        # مناسب برای ترندهای ضعیف یا در حال تغییر و نوسانات بالا
        return (trend_strength <= 0.6 and volatility >= 1.2)
