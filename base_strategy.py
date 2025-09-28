from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import pandas as pd
from enum import Enum
import datetime

class StrategyType(Enum):
    TREND_FOLLOWING = "TREND_FOLLOWING"
    REVERSAL = "REVERSAL"
    BREAKOUT = "BREAKOUT"
    SCALPING = "SCALPING"
    MOMENTUM = "MOMENTUM"
    MEAN_REVERSION = "MEAN_REVERSION"

class StrategySignal(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

class BaseStrategy(ABC):
    """
    کلاس پایه برای تمام استراتژی‌های معاملاتی
    """
    
    def __init__(self, name: str, strategy_type: StrategyType):
        self.name = name
        self.strategy_type = strategy_type
        self.is_enabled = True
        self.min_timeframe = '1h'
        self.max_timeframe = '1d'
        self.min_candles_required = 100
        
        # پارامترهای ریسک
        self.max_risk_per_trade = 0.02
        self.max_leverage = 10
        self.min_risk_reward_ratio = 1.5
        
        # آمار عملکرد
        self.performance_stats = {
            'total_signals': 0,
            'successful_signals': 0,
            'failed_signals': 0,
            'win_rate': 0.0,
            'avg_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
        
        # پارامترهای تطبیقی
        self.adaptive_params = {
            'confidence_threshold': 0.7,
            'volume_threshold': 1.5,
            'volatility_adjustment': 1.0
        }

    @abstractmethod
    def analyze(self, df: pd.DataFrame, symbol: str) -> Optional[Dict]:
        """تحلیل اصلی استراتژی - باید در هر زیرکلاس پیاده‌سازی شود"""
        pass

    @abstractmethod
    def generate_signal(self, analysis_data: Dict) -> Dict:
        """تولید سیگنال معاملاتی"""
        pass

    def validate_data(self, df: pd.DataFrame) -> bool:
        """اعتبارسنجی داده‌ها"""
        if df is None or df.empty:
            return False
        
        if len(df) < self.min_candles_required:
            return False
            
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            return False
            
        return True

    def calculate_position_size(self, price: float, stop_loss: float, 
                              account_balance: float) -> Dict:
        """محاسبه اندازه پوزیشن"""
        risk_amount = account_balance * self.max_risk_per_trade
        price_diff = abs(price - stop_loss)
        
        if price_diff == 0:
            return {'size': 0, 'leverage': 1}
        
        position_size = risk_amount / price_diff
        leverage = min(position_size / account_balance, self.max_leverage)
        
        return {
            'size': position_size,
            'leverage': max(1, int(leverage)),
            'risk_amount': risk_amount,
            'risk_percentage': self.max_risk_per_trade * 100
        }

    def update_performance(self, signal_result: Dict):
        """بروزرسانی آمار عملکرد"""
        self.performance_stats['total_signals'] += 1
        
        if signal_result.get('success', False):
            self.performance_stats['successful_signals'] += 1
        else:
            self.performance_stats['failed_signals'] += 1
        
        # محاسبه win rate
        total = self.performance_stats['total_signals']
        if total > 0:
            self.performance_stats['win_rate'] = (
                self.performance_stats['successful_signals'] / total
            )

    def get_strategy_info(self) -> Dict:
        """اطلاعات کلی استراتژی"""
        return {
            'name': self.name,
            'type': self.strategy_type.value,
            'enabled': self.is_enabled,
            'timeframe_range': f"{self.min_timeframe} - {self.max_timeframe}",
            'min_candles': self.min_candles_required,
            'performance': self.performance_stats,
            'risk_params': {
                'max_risk_per_trade': self.max_risk_per_trade,
                'max_leverage': self.max_leverage,
                'min_rr_ratio': self.min_risk_reward_ratio
            }
        }

    def adapt_parameters(self, market_conditions: Dict):
        """تطبیق پارامترها با شرایط بازار"""
        volatility = market_conditions.get('volatility', 1.0)
        trend_strength = market_conditions.get('trend_strength', 0.5)
        
        # تنظیم آستانه اطمینان بر اساس شرایط
        if volatility > 1.5:
            self.adaptive_params['confidence_threshold'] = 0.8
        elif volatility < 0.7:
            self.adaptive_params['confidence_threshold'] = 0.6
        else:
            self.adaptive_params['confidence_threshold'] = 0.7
        
        # تنظیم تعدیل نوسانات
        self.adaptive_params['volatility_adjustment'] = min(2.0, max(0.5, volatility))

    def is_market_suitable(self, market_context: Dict) -> bool:
        """بررسی مناسب بودن بازار برای این استراتژی"""
        return True  # پیاده‌سازی پایه - در زیرکلاس‌ها تخصصی‌تر می‌شود
