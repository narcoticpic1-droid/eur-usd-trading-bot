import pandas as pd
import numpy as np
from typing import Dict, Optional
from .base_strategy import BaseStrategy, StrategyType, StrategySignal
import talib

class ScalpingStrategy(BaseStrategy):
    """
    استراتژی اسکالپینگ - معاملات کوتاه‌مدت با سود کم و سرعت بالا
    """
    
    def __init__(self):
        super().__init__("Scalping Strategy", StrategyType.SCALPING)
        
        # تنظیمات خاص اسکالپینگ
        self.min_timeframe = '5m'
        self.max_timeframe = '15m'
        self.min_candles_required = 50
        
        # پارامترهای EMA سریع
        self.ema_fast = 5
        self.ema_slow = 13
        
        # پارامترهای Stochastic سریع
        self.stoch_k = 5
        self.stoch_d = 3
        self.stoch_oversold = 20
        self.stoch_overbought = 80
        
        # پارامترهای MACD سریع
        self.macd_fast = 5
        self.macd_slow = 13
        self.macd_signal = 5
        
        # تنظیمات ریسک اسکالپینگ
        self.max_risk_per_trade = 0.005  # 0.5% ریسک
        self.min_risk_reward_ratio = 1.0  # کمتر از سایر استراتژی‌ها
        self.target_profit_pips = 10
        self.stop_loss_pips = 5
        
        # تنظیمات حجم
        self.min_volume_ratio = 1.2
        self.volume_spike_threshold = 2.0

    def analyze(self, df: pd.DataFrame, symbol: str) -> Optional[Dict]:
        """تحلیل اصلی استراتژی اسکالپینگ"""
        if not self.validate_data(df):
            return None
        
        df = df.copy()
        
        # محاسبه اندیکاتورهای سریع
        df = self._calculate_scalping_indicators(df)
        
        # تشخیص سیگنال‌های سریع
        scalp_signals = self._detect_scalping_signals(df)
        
        if not scalp_signals['found']:
            return None
        
        # تحلیل micro-trend
        micro_trend = self._analyze_micro_trend(df)
        
        # تحلیل نهایی
        final_analysis = self._finalize_scalping_signal(df, scalp_signals, micro_trend)
        
        if not final_analysis['valid']:
            return None
        
        return {
            'strategy': self.name,
            'symbol': symbol,
            'signal': final_analysis['direction'],
            'signal_type': 'SCALPING',
            'confidence': final_analysis['confidence'],
            'current_price': df['close'].iloc[-1],
            'entry_price': final_analysis['entry_price'],
            'stop_loss': final_analysis['stop_loss'],
            'take_profits': final_analysis['take_profits'],
            'risk_reward_ratio': final_analysis['risk_reward'],
            'reasoning': final_analysis['reasoning'],
            'market_context': {
                'micro_trend': micro_trend['direction'],
                'momentum': final_analysis['momentum'],
                'volume_spike': final_analysis['volume_spike'],
                'entry_timing': 'IMMEDIATE'
            }
        }

    def _calculate_scalping_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """محاسبه اندیکاتورهای اسکالپینگ"""
        # EMAs سریع
        df['ema_fast'] = talib.EMA(df['close'], timeperiod=self.ema_fast)
        df['ema_slow'] = talib.EMA(df['close'], timeperiod=self.ema_slow)
        
        # MACD سریع
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
            df['close'], 
            fastperiod=self.macd_fast, 
            slowperiod=self.macd_slow, 
            signalperiod=self.macd_signal
        )
        
        # Stochastic سریع
        df['stoch_k'], df['stoch_d'] = talib.STOCH(
            df['high'], df['low'], df['close'],
            fastk_period=self.stoch_k, 
            slowk_period=self.stoch_d, 
            slowd_period=self.stoch_d
        )
        
        # RSI سریع
        df['rsi'] = talib.RSI(df['close'], timeperiod=7)
        
        # حجم
        df['volume_ma'] = df['volume'].rolling(window=10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # ATR برای volatility
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=10)
        
        # Bollinger Bands کوچک
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
            df['close'], timeperiod=10, nbdevup=1.5, nbdevdn=1.5
        )
        
        return df

    def _detect_scalping_signals(self, df: pd.DataFrame) -> Dict:
        """تشخیص سیگنال‌های اسکالپینگ"""
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        signals = {
            'found': False,
            'direction': 0,
            'strength': 0.0,
            'signals': []
        }
        
        # سیگنال 1: EMA Crossover
        if (latest['ema_fast'] > latest['ema_slow'] and 
            prev['ema_fast'] <= prev['ema_slow']):
            signals['signals'].append('EMA_BULLISH_CROSS')
            signals['direction'] = 1
            signals['strength'] += 0.3
        elif (latest['ema_fast'] < latest['ema_slow'] and 
              prev['ema_fast'] >= prev['ema_slow']):
            signals['signals'].append('EMA_BEARISH_CROSS')
            signals['direction'] = -1
            signals['strength'] += 0.3
        
        # سیگنال 2: MACD Histogram
        if (latest['macd_hist'] > 0 and prev['macd_hist'] <= 0):
            signals['signals'].append('MACD_BULLISH')
            if signals['direction'] == 0:
                signals['direction'] = 1
            elif signals['direction'] == 1:
                signals['strength'] += 0.2
        elif (latest['macd_hist'] < 0 and prev['macd_hist'] >= 0):
            signals['signals'].append('MACD_BEARISH')
            if signals['direction'] == 0:
                signals['direction'] = -1
            elif signals['direction'] == -1:
                signals['strength'] += 0.2
        
        # سیگنال 3: Stochastic Extremes
        if (latest['stoch_k'] <= self.stoch_oversold and 
            latest['stoch_k'] > prev['stoch_k']):
            signals['signals'].append('STOCH_OVERSOLD_BOUNCE')
            if signals['direction'] >= 0:
                signals['strength'] += 0.25
        elif (latest['stoch_k'] >= self.stoch_overbought and 
              latest['stoch_k'] < prev['stoch_k']):
            signals['signals'].append('STOCH_OVERBOUGHT_DECLINE')
            if signals['direction'] <= 0:
                signals['strength'] += 0.25
        
        # سیگنال 4: Bollinger Bounce
        if latest['close'] <= latest['bb_lower'] and latest['close'] > prev['close']:
            signals['signals'].append('BB_LOWER_BOUNCE')
            signals['strength'] += 0.2
        elif latest['close'] >= latest['bb_upper'] and latest['close'] < prev['close']:
            signals['signals'].append('BB_UPPER_BOUNCE')
            signals['strength'] += 0.2
        
        signals['found'] = signals['strength'] >= 0.4
        return signals

    def _analyze_micro_trend(self, df: pd.DataFrame) -> Dict:
        """تحلیل micro-trend کوتاه‌مدت"""
        # بررسی آخرین 5 کندل
        recent_data = df.tail(5)
        
        price_changes = recent_data['close'].diff().dropna()
        positive_moves = (price_changes > 0).sum()
        negative_moves = (price_changes < 0).sum()
        
        if positive_moves > negative_moves:
            direction = 'BULLISH'
            strength = positive_moves / len(price_changes)
        elif negative_moves > positive_moves:
            direction = 'BEARISH'
            strength = negative_moves / len(price_changes)
        else:
            direction = 'SIDEWAYS'
            strength = 0.5
        
        return {
            'direction': direction,
            'strength': strength
        }

    def _finalize_scalping_signal(self, df: pd.DataFrame, scalp_signals: Dict, 
                                 micro_trend: Dict) -> Dict:
        """نهایی‌سازی سیگنال اسکالپینگ"""
        latest = df.iloc[-1]
        
        confidence = scalp_signals['strength']
        reasoning = scalp_signals['signals'].copy()
        
        # تأیید micro-trend
        if ((scalp_signals['direction'] > 0 and micro_trend['direction'] == 'BULLISH') or
            (scalp_signals['direction'] < 0 and micro_trend['direction'] == 'BEARISH')):
            confidence += 0.15
            reasoning.append("تأیید micro-trend")
        
        # تأیید حجم
        volume_spike = latest['volume_ratio'] >= self.volume_spike_threshold
        if volume_spike:
            confidence += 0.2
            reasoning.append("spike حجم")
        elif latest['volume_ratio'] >= self.min_volume_ratio:
            confidence += 0.1
            reasoning.append("حجم کافی")
        
        # تحلیل momentum
        momentum = 'STRONG' if latest['rsi'] > 60 or latest['rsi'] < 40 else 'WEAK'
        if momentum == 'STRONG':
            confidence += 0.1
            reasoning.append("momentum قوی")
        
        # محاسبه نقاط ورود و خروج (اسکالپینگ دقیق)
        current_price = latest['close']
        atr = latest['atr']
        
        # اندازه pip برای محاسبه (فرض: 4 رقم اعشار)
        pip_size = 0.0001 if 'JPY' not in df.index.name else 0.01
        
        if scalp_signals['direction'] > 0:  # خرید
            entry_price = current_price
            stop_loss = current_price - (self.stop_loss_pips * pip_size)
            take_profit_1 = current_price + (self.target_profit_pips * pip_size)
            take_profit_2 = current_price + (self.target_profit_pips * 2 * pip_size)
        else:  # فروش
            entry_price = current_price
            stop_loss = current_price + (self.stop_loss_pips * pip_size)
            take_profit_1 = current_price - (self.target_profit_pips * pip_size)
            take_profit_2 = current_price - (self.target_profit_pips * 2 * pip_size)
        
        risk_reward = abs(take_profit_1 - entry_price) / abs(entry_price - stop_loss)
        
        return {
            'valid': confidence >= 0.6,  # آستانه پایین‌تر برای اسکالپینگ
            'direction': scalp_signals['direction'],
            'confidence': min(1.0, confidence),
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profits': [take_profit_1, take_profit_2],
            'risk_reward': risk_reward,
            'momentum': momentum,
            'volume_spike': volume_spike,
            'reasoning': reasoning
        }

    def generate_signal(self, analysis_data: Dict) -> Dict:
        """تولید سیگنال نهایی اسکالپینگ"""
        confidence = analysis_data['confidence']
        
        # آستانه‌های متفاوت برای اسکالپینگ
        if confidence >= 0.75:
            signal = StrategySignal.STRONG_BUY if analysis_data['signal'] > 0 else StrategySignal.STRONG_