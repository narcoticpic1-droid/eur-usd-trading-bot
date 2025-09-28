from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime
import json
import sqlite3
import pandas as pd

class SignalType(Enum):
    """انواع سیگنال"""
    BREAKOUT = "BREAKOUT"
    PULLBACK = "PULLBACK" 
    REVERSAL = "REVERSAL"
    TREND_CONTINUATION = "TREND_CONTINUATION"
    SUPPORT_BOUNCE = "SUPPORT_BOUNCE"
    RESISTANCE_BREAK = "RESISTANCE_BREAK"
    MOMENTUM_SHIFT = "MOMENTUM_SHIFT"
    VOLUME_SPIKE = "VOLUME_SPIKE"

class SignalQuality(Enum):
    """کیفیت سیگنال"""
    VERY_HIGH = "VERY_HIGH"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    VERY_LOW = "VERY_LOW"

class SignalDirection(Enum):
    """جهت سیگنال"""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"

class SignalStatus(Enum):
    """وضعیت سیگنال"""
    ACTIVE = "ACTIVE"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    STOPPED_OUT = "STOPPED_OUT"
    TARGET_HIT = "TARGET_HIT"

@dataclass
class TechnicalIndicators:
    """اندیکاتورهای فنی"""
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_middle: Optional[float] = None
    ema_20: Optional[float] = None
    ema_50: Optional[float] = None
    ema_200: Optional[float] = None
    volume_sma: Optional[float] = None
    adx: Optional[float] = None
    stochastic_k: Optional[float] = None
    stochastic_d: Optional[float] = None
    williams_r: Optional[float] = None
    cci: Optional[float] = None
    obv: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """تبدیل به دیکشنری"""
        return {k: v for k, v in self.__dict__.items() if v is not None}

@dataclass
class PriceAction:
    """تحلیل Price Action"""
    current_price: float
    open_price: float
    high_price: float
    low_price: float
    volume: float
    
    # Swing Points
    last_swing_high: Optional[float] = None
    last_swing_low: Optional[float] = None
    swing_high_distance: Optional[float] = None
    swing_low_distance: Optional[float] = None
    
    # Support/Resistance
    nearest_support: Optional[float] = None
    nearest_resistance: Optional[float] = None
    support_strength: Optional[int] = None
    resistance_strength: Optional[int] = None
    
    # Candlestick Patterns
    is_doji: bool = False
    is_hammer: bool = False
    is_shooting_star: bool = False
    is_engulfing: bool = False
    is_pin_bar: bool = False
    
    # Body and Shadow Analysis
    body_size: Optional[float] = None
    upper_shadow_size: Optional[float] = None
    lower_shadow_size: Optional[float] = None
    body_to_range_ratio: Optional[float] = None
    
    def calculate_derived_metrics(self):
        """محاسبه معیارهای مشتق شده"""
        total_range = self.high_price - self.low_price
        if total_range > 0:
            self.body_size = abs(self.current_price - self.open_price)
            self.upper_shadow_size = self.high_price - max(self.current_price, self.open_price)
            self.lower_shadow_size = min(self.current_price, self.open_price) - self.low_price
            self.body_to_range_ratio = self.body_size / total_range
            
            # تشخیص الگوهای کندل
            self.is_doji = self.body_to_range_ratio < 0.1
            self.is_hammer = (self.lower_shadow_size > self.body_size * 2 and 
                            self.upper_shadow_size < self.body_size * 0.5)
            self.is_shooting_star = (self.upper_shadow_size > self.body_size * 2 and 
                                   self.lower_shadow_size < self.body_size * 0.5)
            self.is_pin_bar = self.is_hammer or self.is_shooting_star

@dataclass
class MarketContext:
    """بافت بازار"""
    trend_direction: str
    trend_strength: str
    market_structure: str
    volatility_level: str
    volume_profile: str
    session_time: str
    
    # Market Regime
    is_trending: bool = False
    is_ranging: bool = False
    is_breakout_environment: bool = False
    
    # Volatility Metrics
    atr: Optional[float] = None
    volatility_percentile: Optional[float] = None
    
    # Volume Analysis
    volume_trend: Optional[str] = None
    volume_breakout: bool = False
    
    # Time-based factors
    is_major_session: bool = False
    is_overlap_session: bool = False

@dataclass
class RiskMetrics:
    """معیارهای ریسک"""
    stop_loss: float
    take_profit_1: float
    take_profit_2: Optional[float] = None
    take_profit_3: Optional[float] = None
    
    risk_reward_ratio: float = 0.0
    position_size_percent: float = 1.0
    max_loss_percent: float = 2.0
    
    # Risk Assessment
    risk_level: str = "MEDIUM"
    confidence_level: float = 0.5
    
    # Portfolio Impact
    correlation_risk: float = 0.0
    portfolio_heat: float = 0.0
    
    def calculate_risk_reward(self, entry_price: float):
        """محاسبه نسبت ریسک به ریوارد"""
        risk = abs(entry_price - self.stop_loss)
        reward = abs(self.take_profit_1 - entry_price)
        
        if risk > 0:
            self.risk_reward_ratio = reward / risk
        
        # تنظیم سطح ریسک
        if self.risk_reward_ratio >= 3.0:
            self.risk_level = "LOW"
        elif self.risk_reward_ratio >= 2.0:
            self.risk_level = "MEDIUM" 
        else:
            self.risk_level = "HIGH"

@dataclass
class AIEvaluation:
    """ارزیابی AI"""
    model_predictions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    consensus_score: float = 0.0
    consensus_direction: str = "NEUTRAL"
    confidence_distribution: Dict[str, float] = field(default_factory=dict)
    
    # Individual Model Scores
    gemini_score: Optional[float] = None
    openai_score: Optional[float] = None
    claude_score: Optional[float] = None
    
    # Aggregate Metrics
    avg_confidence: float = 0.0
    prediction_variance: float = 0.0
    unanimous_agreement: bool = False
    
    def calculate_consensus(self):
        """محاسبه اجماع AI ها"""
        if not self.model_predictions:
            return
        
        predictions = []
        confidences = []
        
        for model, data in self.model_predictions.items():
            if 'prediction' in data and 'confidence' in data:
                predictions.append(data['prediction'])
                confidences.append(data['confidence'])
        
        if predictions:
            # محاسبه میانگین اطمینان
            self.avg_confidence = sum(confidences) / len(confidences)
            
            # تشخیص اجماع
            unique_predictions = set(predictions)
            if len(unique_predictions) == 1:
                self.unanimous_agreement = True
                self.consensus_direction = list(unique_predictions)[0]
                self.consensus_score = self.avg_confidence
            else:
                # اکثریت
                from collections import Counter
                prediction_counts = Counter(predictions)
                majority_prediction, count = prediction_counts.most_common(1)[0]
                
                self.consensus_direction = majority_prediction
                self.consensus_score = (count / len(predictions)) * self.avg_confidence
                self.unanimous_agreement = False

class TradingSignal:
    """مدل اصلی سیگنال معاملاتی"""
    
    def __init__(self):
        # Identification
        self.signal_id: str = ""
        self.timestamp: datetime = datetime.now()
        self.symbol: str = ""
        self.timeframe: str = ""
        
        # Signal Core
        self.signal_type: SignalType = SignalType.BREAKOUT
        self.direction: SignalDirection = SignalDirection.NEUTRAL
        self.quality: SignalQuality = SignalQuality.MEDIUM
        self.status: SignalStatus = SignalStatus.ACTIVE
        
        # Analysis Components
        self.technical_indicators: TechnicalIndicators = TechnicalIndicators()
        self.price_action: PriceAction = PriceAction(0, 0, 0, 0, 0)
        self.market_context: MarketContext = MarketContext("", "", "", "", "", "")
        self.risk_metrics: RiskMetrics = RiskMetrics(0, 0)
        self.ai_evaluation: AIEvaluation = AIEvaluation()
        
        # Additional Metadata
        self.reasoning: List[str] = []
        self.tags: List[str] = []
        self.notes: str = ""
        
        # Performance Tracking
        self.entry_price: Optional[float] = None
        self.exit_price: Optional[float] = None
        self.pnl: Optional[float] = None
        self.pnl_percent: Optional[float] = None
        self.execution_time: Optional[datetime] = None
        self.exit_time: Optional[datetime] = None
    
    def generate_signal_id(self) -> str:
        """تولید شناسه یکتا برای سیگنال"""
        timestamp_str = self.timestamp.strftime("%Y%m%d_%H%M%S")
        symbol_clean = self.symbol.replace("/", "").replace("-", "")
        direction_short = self.direction.value[0]  # L or S
        
        self.signal_id = f"{symbol_clean}_{timestamp_str}_{direction_short}"
        return self.signal_id
    
    def validate_signal(self) -> tuple[bool, List[str]]:
        """اعتبارسنجی سیگنال"""
        errors = []
        
        # بررسی فیلدهای ضروری
        if not self.symbol:
            errors.append("Symbol is required")
        
        if self.price_action.current_price <= 0:
            errors.append("Current price must be positive")
        
        if self.risk_metrics.stop_loss == 0:
            errors.append("Stop loss is required")
        
        if self.risk_metrics.take_profit_1 == 0:
            errors.append("Take profit is required")
        
        # بررسی منطق ریسک/ریوارد
        if self.direction == SignalDirection.LONG:
            if self.risk_metrics.stop_loss >= self.price_action.current_price:
                errors.append("Stop loss must be below entry for LONG position")
            if self.risk_metrics.take_profit_1 <= self.price_action.current_price:
                errors.append("Take profit must be above entry for LONG position")
        
        elif self.direction == SignalDirection.SHORT:
            if self.risk_metrics.stop_loss <= self.price_action.current_price:
                errors.append("Stop loss must be above entry for SHORT position")
            if self.risk_metrics.take_profit_1 >= self.price_action.current_price:
                errors.append("Take profit must be below entry for SHORT position")
        
        # بررسی کیفیت
        if self.quality == SignalQuality.VERY_LOW:
            errors.append("Signal quality too low for trading")
        
        return len(errors) == 0, errors
    
    def calculate_signal_strength(self) -> float:
        """محاسبه قدرت کلی سیگنال"""
        strength_factors = []
        
        # Technical Indicators Weight (30%)
        tech_score = 0.5  # پیش‌فرض
        if self.technical_indicators.rsi:
            if self.direction == SignalDirection.LONG and self.technical_indicators.rsi < 70:
                tech_score += 0.2
            elif self.direction == SignalDirection.SHORT and self.technical_indicators.rsi > 30:
                tech_score += 0.2
        
        strength_factors.append(('technical', tech_score, 0.30))
        
        # Price Action Weight (25%)
        pa_score = 0.5
        if self.price_action.body_to_range_ratio:
            if self.price_action.body_to_range_ratio > 0.6:  # Strong body
                pa_score += 0.3
        
        strength_factors.append(('price_action', pa_score, 0.25))
        
        # Market Context Weight (20%)
        market_score = 0.5
        if self.market_context.is_trending and self.signal_type == SignalType.TREND_CONTINUATION:
            market_score += 0.3
        elif self.market_context.is_breakout_environment and self.signal_type == SignalType.BREAKOUT:
            market_score += 0.4
        
        strength_factors.append(('market', market_score, 0.20))
        
        # AI Consensus Weight (15%)
        ai_score = self.ai_evaluation.consensus_score
        strength_factors.append(('ai', ai_score, 0.15))
        
        # Risk/Reward Weight (10%)
        rr_score = min(self.risk_metrics.risk_reward_ratio / 4.0, 1.0)  # Cap at 4:1
        strength_factors.append(('risk_reward', rr_score, 0.10))
        
        # Calculate weighted average
        total_score = sum(score * weight for _, score, weight in strength_factors)
        return min(max(total_score, 0.0), 1.0)  # Clamp between 0 and 1
    
    def to_dict(self) -> Dict[str, Any]:
        """تبدیل به دیکشنری برای ذخیره یا انتقال"""
        return {
            'signal_id': self.signal_id,
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'signal_type': self.signal_type.value,
            'direction': self.direction.value,
            'quality': self.quality.value,
            'status': self.status.value,
            'technical_indicators': self.technical_indicators.to_dict(),
            'price_action': self.price_action.__dict__,
            'market_context': self.market_context.__dict__,
            'risk_metrics': self.risk_metrics.__dict__,
            'ai_evaluation': {
                'model_predictions': self.ai_evaluation.model_predictions,
                'consensus_score': self.ai_evaluation.consensus_score,
                'consensus_direction': self.ai_evaluation.consensus_direction,
                'avg_confidence': self.ai_evaluation.avg_confidence,
                'unanimous_agreement': self.ai_evaluation.unanimous_agreement
            },
            'reasoning': self.reasoning,
            'tags': self.tags,
            'notes': self.notes,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'pnl': self.pnl,
            'pnl_percent': self.pnl_percent,
            'signal_strength': self.calculate_signal_strength()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingSignal':
        """ایجاد سیگنال از دیکشنری"""
        signal = cls()
        
        # Basic fields
        signal.signal_id = data.get('signal_id', '')
        if 'timestamp' in data:
            signal.timestamp = datetime.fromisoformat(data['timestamp'])
        signal.symbol = data.get('symbol', '')
        signal.timeframe = data.get('timeframe', '')
        
        # Enums
        if 'signal_type' in data:
            signal.signal_type = SignalType(data['signal_type'])
        if 'direction' in data:
            signal.direction = SignalDirection(data['direction'])
        if 'quality' in data:
            signal.quality = SignalQuality(data['quality'])
        if 'status' in data:
            signal.status = SignalStatus(data['status'])
        
        # Complex objects
        if 'technical_indicators' in data:
            ti_data = data['technical_indicators']
            signal.technical_indicators = TechnicalIndicators(**ti_data)
        
        if 'price_action' in data:
            pa_data = data['price_action']
            signal.price_action = PriceAction(**pa_data)
        
        # Other fields
        signal.reasoning = data.get('reasoning', [])
        signal.tags = data.get('tags', [])
        signal.notes = data.get('notes', '')
        signal.entry_price = data.get('entry_price')
        signal.exit_price = data.get('exit_price')
        signal.pnl = data.get('pnl')
        signal.pnl_percent = data.get('pnl_percent')
        
        return signal

class SignalDatabase:
    """مدیریت پایگاه داده سیگنال‌ها"""
    
    def __init__(self, db_path: str = "trading_signals.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """راه‌اندازی جداول پایگاه داده"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trading_signals (
                signal_id TEXT PRIMARY KEY,
                timestamp DATETIME,
                symbol TEXT,
                timeframe TEXT,
                signal_type TEXT,
                direction TEXT,
                quality TEXT,
                status TEXT,
                signal_data TEXT,
                signal_strength REAL,
                entry_price REAL,
                exit_price REAL,
                pnl REAL,
                pnl_percent REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Index for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_symbol_timestamp 
            ON trading_signals(symbol, timestamp)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_status_quality 
            ON trading_signals(status, quality)
        ''')
        
        conn.commit()
        conn.close()
    
    def save_signal(self, signal: TradingSignal) -> bool:
        """ذخیره سیگنال در پایگاه داده"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            signal_dict = signal.to_dict()
            signal_data_json = json.dumps(signal_dict)
            
            cursor.execute('''
                INSERT OR REPLACE INTO trading_signals
                (signal_id, timestamp, symbol, timeframe, signal_type, direction,
                 quality, status, signal_data, signal_strength, entry_price,
                 exit_price, pnl, pnl_percent, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                signal.signal_id,
                signal.timestamp,
                signal.symbol,
                signal.timeframe,
                signal.signal_type.value,
                signal.direction.value,
                signal.quality.value,
                signal.status.value,
                signal_data_json,
                signal.calculate_signal_strength(),
                signal.entry_price,
                signal.exit_price,
                signal.pnl,
                signal.pnl_percent
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"خطا در ذخیره سیگنال: {e}")
            return False
    
    def get_signal(self, signal_id: str) -> Optional[TradingSignal]:
        """دریافت سیگنال با شناسه"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT signal_data FROM trading_signals WHERE signal_id = ?
            ''', (signal_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                signal_data = json.loads(result[0])
                return TradingSignal.from_dict(signal_data)
            
            return None
            
        except Exception as e:
            print(f"خطا در دریافت سیگنال: {e}")
            return None
    
    def get_signals_by_symbol(self, symbol: str, limit: int = 100) -> List[TradingSignal]:
        """دریافت سیگنال‌های یک نماد"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT signal_data FROM trading_signals 
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (symbol, limit))
            
            results = cursor.fetchall()
            conn.close()
            
            signals = []
            for result in results:
                signal_data = json.loads(result[0])
                signals.append(TradingSignal.from_dict(signal_data))
            
            return signals
            
        except Exception as e:
            print(f"خطا در دریافت سیگنال‌ها: {e}")
            return []
    
    def update_signal_outcome(self, signal_id: str, exit_price: float, 
                            pnl: float, pnl_percent: float) -> bool:
        """به‌روزرسانی نتیجه سیگنال"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE trading_signals
                SET exit_price = ?, pnl = ?, pnl_percent = ?, 
                    status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE signal_id = ?
            ''', (
                exit_price, pnl, pnl_percent,
                SignalStatus.FILLED.value if pnl > 0 else SignalStatus.STOPPED_OUT.value,
                signal_id
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"خطا در به‌روزرسانی نتیجه سیگنال: {e}")
            return False
    
    def get_performance_summary(self, symbol: Optional[str] = None, 
                              days: int = 30) -> Dict[str, Any]:
        """خلاصه عملکرد سیگنال‌ها"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            where_clause = "WHERE timestamp >= datetime('now', '-{} days')".format(days)
            if symbol:
                where_clause += f" AND symbol = '{symbol}'"
            
            # آمار کلی
            cursor.execute(f'''
                SELECT 
                    COUNT(*) as total_signals,
                    COUNT(CASE WHEN pnl > 0 THEN 1 END) as winning_signals,
                    COUNT(CASE WHEN pnl < 0 THEN 1 END) as losing_signals,
                    AVG(pnl_percent) as avg_pnl_percent,
                    AVG(signal_strength) as avg_signal_strength,
                    AVG(CASE WHEN quality = 'HIGH' THEN 1.0 ELSE 0.0 END) as high_quality_rate
                FROM trading_signals
                {where_clause}
            ''')
            
            stats = cursor.fetchone()
            conn.close()
            
            if stats and stats[0] > 0:
                return {
                    'total_signals': stats[0],
                    'winning_signals': stats[1] or 0,
                    'losing_signals': stats[2] or 0,
                    'win_rate': (stats[1] or 0) / stats[0],
                    'avg_pnl_percent': stats[3] or 0,
                    'avg_signal_strength': stats[4] or 0,
                    'high_quality_rate': stats[5] or 0
                }
            
            return {}
            
        except Exception as e:
            print(f"خطا در محاسبه خلاصه عملکرد: {e}")
            return {}
