from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime, timedelta
import json
import sqlite3
import uuid

class TradeStatus(Enum):
    """وضعیت معامله"""
    PENDING = "PENDING"
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    STOPPED_OUT = "STOPPED_OUT"
    TARGET_HIT = "TARGET_HIT"

class TradeType(Enum):
    """نوع معامله"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"

class PositionSide(Enum):
    """سمت پوزیشن"""
    LONG = "LONG"
    SHORT = "SHORT"

class OrderType(Enum):
    """نوع سفارش"""
    ENTRY = "ENTRY"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    PARTIAL_CLOSE = "PARTIAL_CLOSE"
    FULL_CLOSE = "FULL_CLOSE"

@dataclass
class OrderExecution:
    """اجرای سفارش"""
    order_id: str
    execution_time: datetime
    executed_price: float
    executed_quantity: float
    commission: float = 0.0
    slippage: float = 0.0
    
    def calculate_slippage(self, expected_price: float):
        """محاسبه slippage"""
        self.slippage = abs(self.executed_price - expected_price) / expected_price

@dataclass
class TradeOrder:
    """سفارش معامله"""
    order_id: str = ""
    order_type: OrderType = OrderType.ENTRY
    trade_type: TradeType = TradeType.MARKET
    price: float = 0.0
    quantity: float = 0.0
    status: str = "PENDING"
    
    # Execution details
    execution: Optional[OrderExecution] = None
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    
    # Order management
    parent_trade_id: str = ""
    is_active: bool = True
    
    def __post_init__(self):
        if not self.order_id:
            self.order_id = str(uuid.uuid4())

@dataclass
class RiskManagement:
    """مدیریت ریسک معامله"""
    initial_stop_loss: float = 0.0
    current_stop_loss: float = 0.0
    take_profit_levels: List[float] = field(default_factory=list)
    
    # Position sizing
    position_size_usd: float = 0.0
    position_size_percent: float = 1.0
    max_risk_percent: float = 2.0
    leverage: float = 1.0
    
    # Risk metrics
    initial_risk_usd: float = 0.0
    current_risk_usd: float = 0.0
    risk_reward_ratio: float = 0.0
    
    # Trailing stop
    trailing_stop_enabled: bool = False
    trailing_stop_distance: float = 0.0
    highest_price: float = 0.0  # For LONG positions
    lowest_price: float = 0.0   # For SHORT positions
    
    def update_trailing_stop(self, current_price: float, position_side: PositionSide):
        """به‌روزرسانی trailing stop"""
        if not self.trailing_stop_enabled:
            return
        
        if position_side == PositionSide.LONG:
            if current_price > self.highest_price:
                self.highest_price = current_price
                new_stop = current_price - self.trailing_stop_distance
                if new_stop > self.current_stop_loss:
                    self.current_stop_loss = new_stop
        
        elif position_side == PositionSide.SHORT:
            if current_price < self.lowest_price or self.lowest_price == 0:
                self.lowest_price = current_price
                new_stop = current_price + self.trailing_stop_distance
                if new_stop < self.current_stop_loss or self.current_stop_loss == 0:
                    self.current_stop_loss = new_stop

@dataclass
class TradePerformance:
    """عملکرد معامله"""
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl_percent: float = 0.0
    realized_pnl_percent: float = 0.0
    
    # Execution metrics
    total_commission: float = 0.0
    total_slippage: float = 0.0
    
    # Timing metrics
    duration_minutes: float = 0.0
    holding_period_return: float = 0.0
    
    # Risk metrics
    max_adverse_excursion: float = 0.0  # MAE
    max_favorable_excursion: float = 0.0  # MFE
    
    # Trade efficiency
    efficiency_ratio: float = 0.0  # Final PnL / MFE
    
    def calculate_performance_metrics(self, entry_price: float, current_price: float, 
                                    position_size: float, position_side: PositionSide):
        """محاسبه معیارهای عملکرد"""
        if position_side == PositionSide.LONG:
            price_change = current_price - entry_price
        else:
            price_change = entry_price - current_price
        
        self.unrealized_pnl = price_change * position_size
        self.unrealized_pnl_percent = price_change / entry_price
        
        # Update MAE and MFE
        if self.unrealized_pnl < self.max_adverse_excursion:
            self.max_adverse_excursion = self.unrealized_pnl
        
        if self.unrealized_pnl > self.max_favorable_excursion:
            self.max_favorable_excursion = self.unrealized_pnl

class Trade:
    """مدل اصلی معامله"""
    
    def __init__(self):
        # Identification
        self.trade_id: str = str(uuid.uuid4())
        self.signal_id: str = ""
        self.symbol: str = ""
        self.timeframe: str = ""
        
        # Trade basic info
        self.position_side: PositionSide = PositionSide.LONG
        self.status: TradeStatus = TradeStatus.PENDING
        self.strategy_name: str = ""
        
        # Pricing and execution
        self.entry_price: float = 0.0
        self.current_price: float = 0.0
        self.exit_price: float = 0.0
        self.quantity: float = 0.0
        
        # Timestamps
        self.created_at: datetime = datetime.now()
        self.opened_at: Optional[datetime] = None
        self.closed_at: Optional[datetime] = None
        
        # Components
        self.risk_management: RiskManagement = RiskManagement()
        self.performance: TradePerformance = TradePerformance()
        self.orders: List[TradeOrder] = []
        
        # Metadata
        self.tags: List[str] = []
        self.notes: str = ""
        self.confidence_score: float = 0.5
        
        # AI and analysis
        self.ai_prediction: Dict[str, Any] = {}
        self.market_conditions: Dict[str, Any] = {}
    
    def add_order(self, order: TradeOrder) -> str:
        """اضافه کردن سفارش به معامله"""
        order.parent_trade_id = self.trade_id
        self.orders.append(order)
        return order.order_id
    
    def execute_order(self, order_id: str, execution: OrderExecution) -> bool:
        """اجرای سفارش"""
        try:
            order = next(o for o in self.orders if o.order_id == order_id)
            order.execution = execution
            order.status = "FILLED"
            order.filled_at = execution.execution_time
            
            # به‌روزرسانی وضعیت معامله بر اساس نوع سفارش
            if order.order_type == OrderType.ENTRY:
                self.entry_price = execution.executed_price
                self.opened_at = execution.execution_time
                self.status = TradeStatus.OPEN
                
                # تنظیم trailing stop
                if self.risk_management.trailing_stop_enabled:
                    if self.position_side == PositionSide.LONG:
                        self.risk_management.highest_price = execution.executed_price
                    else:
                        self.risk_management.lowest_price = execution.executed_price
            
            elif order.order_type in [OrderType.STOP_LOSS, OrderType.TAKE_PROFIT, OrderType.FULL_CLOSE]:
                self.exit_price = execution.executed_price
                self.closed_at = execution.execution_time
                self.status = TradeStatus.CLOSED
                
                # محاسبه PnL نهایی
                self._calculate_final_pnl()
            
            # به‌روزرسانی کمیسیون کل
            self.performance.total_commission += execution.commission
            self.performance.total_slippage += execution.slippage
            
            return True
            
        except StopIteration:
            return False
    
    def update_current_price(self, price: float):
        """به‌روزرسانی قیمت فعلی"""
        self.current_price = price
        
        if self.status == TradeStatus.OPEN:
            # به‌روزرسانی عملکرد
            self.performance.calculate_performance_metrics(
                self.entry_price, price, self.quantity, self.position_side
            )
            
            # به‌روزرسانی trailing stop
            self.risk_management.update_trailing_stop(price, self.position_side)
            
            # بررسی stop loss یا take profit
            self._check_exit_conditions()
    
    def _check_exit_conditions(self):
        """بررسی شرایط خروج"""
        if self.status != TradeStatus.OPEN:
            return
        
        current_price = self.current_price
        
        # بررسی stop loss
        if self.position_side == PositionSide.LONG:
            if current_price <= self.risk_management.current_stop_loss:
                self._trigger_stop_loss()
        else:
            if current_price >= self.risk_management.current_stop_loss:
                self._trigger_stop_loss()
        
        # بررسی take profit levels
        for tp_level in self.risk_management.take_profit_levels:
            if self.position_side == PositionSide.LONG:
                if current_price >= tp_level:
                    self._trigger_take_profit(tp_level)
                    break
            else:
                if current_price <= tp_level:
                    self._trigger_take_profit(tp_level)
                    break
    
    def _trigger_stop_loss(self):
        """فعال‌سازی stop loss"""
        stop_order = TradeOrder(
            order_type=OrderType.STOP_LOSS,
            trade_type=TradeType.MARKET,
            price=self.risk_management.current_stop_loss,
            quantity=self.quantity
        )
        self.add_order(stop_order)
        self.status = TradeStatus.STOPPED_OUT
    
    def _trigger_take_profit(self, tp_level: float):
        """فعال‌سازی take profit"""
        tp_order = TradeOrder(
            order_type=OrderType.TAKE_PROFIT,
            trade_type=TradeType.MARKET,
            price=tp_level,
            quantity=self.quantity
        )
        self.add_order(tp_order)
        self.status = TradeStatus.TARGET_HIT
    
    def _calculate_final_pnl(self):
        """محاسبه PnL نهایی"""
        if not self.exit_price or not self.entry_price:
            return
        
        if self.position_side == PositionSide.LONG:
            price_change = self.exit_price - self.entry_price
        else:
            price_change = self.entry_price - self.exit_price
        
        self.performance.realized_pnl = price_change * self.quantity
        self.performance.realized_pnl_percent = price_change / self.entry_price
        
        # Duration
        if self.opened_at and self.closed_at:
            duration = self.closed_at - self.opened_at
            self.performance.duration_minutes = duration.total_seconds() / 60
        
        # Efficiency ratio
        if self.performance.max_favorable_excursion > 0:
            self.performance.efficiency_ratio = (
                self.performance.realized_pnl / self.performance.max_favorable_excursion
            )
    
    def calculate_position_size(self, account_balance: float, risk_percent: float = None) -> float:
        """محاسبه اندازه پوزیشن"""
        if risk_percent is None:
            risk_percent = self.risk_management.max_risk_percent
        
        risk_amount = account_balance * (risk_percent / 100)
        
        if self.risk_management.initial_stop_loss:
            risk_per_unit = abs(self.entry_price - self.risk_management.initial_stop_loss)
            if risk_per_unit > 0:
                self.quantity = risk_amount / risk_per_unit
                self.risk_management.position_size_usd = self.quantity * self.entry_price
                self.risk_management.initial_risk_usd = risk_amount
        
        return self.quantity
    
    def close_trade(self, close_price: float, reason: str = "Manual close"):
        """بستن دستی معامله"""
        if self.status != TradeStatus.OPEN:
            return False
        
        close_order = TradeOrder(
            order_type=OrderType.FULL_CLOSE,
            trade_type=TradeType.MARKET,
            price=close_price,
            quantity=self.quantity
        )
        
        execution = OrderExecution(
            order_id=close_order.order_id,
            execution_time=datetime.now(),
            executed_price=close_price,
            executed_quantity=self.quantity
        )
        
        self.add_order(close_order)
        self.execute_order(close_order.order_id, execution)
        self.notes += f" | Closed: {reason}"
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """تبدیل به دیکشنری"""
        return {
            'trade_id': self.trade_id,
            'signal_id': self.signal_id,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'position_side': self.position_side.value,
            'status': self.status.value,
            'strategy_name': self.strategy_name,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'exit_price': self.exit_price,
            'quantity': self.quantity,
            'created_at': self.created_at.isoformat(),
            'opened_at': self.opened_at.isoformat() if self.opened_at else None,
            'closed_at': self.closed_at.isoformat() if self.closed_at else None,
            'risk_management': self.risk_management.__dict__,
            'performance': self.performance.__dict__,
            'orders': [order.__dict__ for order in self.orders],
            'tags': self.tags,
            'notes': self.notes,
            'confidence_score': self.confidence_score,
            'ai_prediction': self.ai_prediction,
            'market_conditions': self.market_conditions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Trade':
        """ایجاد از دیکشنری"""
        trade = cls()
        
        # Basic fields
        trade.trade_id = data.get('trade_id', trade.trade_id)
        trade.signal_id = data.get('signal_id', '')
        trade.symbol = data.get('symbol', '')
        trade.timeframe = data.get('timeframe', '')
        trade.strategy_name = data.get('strategy_name', '')
        
        # Enums
        if 'position_side' in data:
            trade.position_side = PositionSide(data['position_side'])
        if 'status' in data:
            trade.status = TradeStatus(data['status'])
        
        # Prices and quantities
        trade.entry_price = data.get('entry_price', 0.0)
        trade.current_price = data.get('current_price', 0.0)
        trade.exit_price = data.get('exit_price', 0.0)
        trade.quantity = data.get('quantity', 0.0)
        
        # Timestamps
        if 'created_at' in data:
            trade.created_at = datetime.fromisoformat(data['created_at'])
        if data.get('opened_at'):
            trade.opened_at = datetime.fromisoformat(data['opened_at'])
        if data.get('closed_at'):
            trade.closed_at = datetime.fromisoformat(data['closed_at'])
        
        # Complex objects would need similar reconstruction
        trade.tags = data.get('tags', [])
        trade.notes = data.get('notes', '')
        trade.confidence_score = data.get('confidence_score', 0.5)
        trade.ai_prediction = data.get('ai_prediction', {})
        trade.market_conditions = data.get('market_conditions', {})
        
        return trade

class TradeManager:
    """مدیر معاملات"""
    
    def __init__(self, db_path: str = "trades.db"):
        self.db_path = db_path
        self.active_trades: Dict[str, Trade] = {}
        self.init_database()
    
    def init_database(self):
        """راه‌اندازی پایگاه داده معاملات"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                signal_id TEXT,
                symbol TEXT,
                timeframe TEXT,
                position_side TEXT,
                status TEXT,
                strategy_name TEXT,
                entry_price REAL,
                exit_price REAL,
                quantity REAL,
                created_at DATETIME,
                opened_at DATETIME,
                closed_at DATETIME,
                realized_pnl REAL,
                realized_pnl_percent REAL,
                trade_data TEXT,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trade_orders (
                order_id TEXT PRIMARY KEY,
                trade_id TEXT,
                order_type TEXT,
                trade_type TEXT,
                price REAL,
                quantity REAL,
                status TEXT,
                created_at DATETIME,
                filled_at DATETIME,
                execution_data TEXT,
                FOREIGN KEY (trade_id) REFERENCES trades (trade_id)
            )
        ''')
        
        # Indexes
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_trade_symbol_status 
            ON trades(symbol, status)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_trade_created_at 
            ON trades(created_at)
        ''')
        
        conn.commit()
        conn.close()
    
    def create_trade(self, signal_id: str, symbol: str, position_side: PositionSide,
                    entry_price: float, stop_loss: float, take_profits: List[float],
                    risk_percent: float = 2.0, leverage: float = 1.0) -> Trade:
        """ایجاد معامله جدید"""
        trade = Trade()
        trade.signal_id = signal_id
        trade.symbol = symbol
        trade.position_side = position_side
        trade.entry_price = entry_price
        
        # تنظیم ریسک منجمنت
        trade.risk_management.initial_stop_loss = stop_loss
        trade.risk_management.current_stop_loss = stop_loss
        trade.risk_management.take_profit_levels = take_profits
        trade.risk_management.max_risk_percent = risk_percent
        trade.risk_management.leverage = leverage
        
        # محاسبه ریسک/ریوارد
        risk = abs(entry_price - stop_loss)
        if take_profits:
            reward = abs(take_profits[0] - entry_price)
            trade.risk_management.risk_reward_ratio = reward / risk if risk > 0 else 0
        
        # ایجاد سفارش ورود
        entry_order = TradeOrder(
            order_type=OrderType.ENTRY,
            trade_type=TradeType.MARKET,
            price=entry_price,
            quantity=0  # محاسبه می‌شود
        )
        trade.add_order(entry_order)
        
        self.active_trades[trade.trade_id] = trade
        return trade
    
    def save_trade(self, trade: Trade) -> bool:
        """ذخیره معامله"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            trade_data = json.dumps(trade.to_dict())
            
            cursor.execute('''
                INSERT OR REPLACE INTO trades
                (trade_id, signal_id, symbol, timeframe, position_side, status,
                 strategy_name, entry_price, exit_price, quantity, created_at,
                 opened_at, closed_at, realized_pnl, realized_pnl_percent, trade_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade.trade_id, trade.signal_id, trade.symbol, trade.timeframe,
                trade.position_side.value, trade.status.value, trade.strategy_name,
                trade.entry_price, trade.exit_price, trade.quantity,
                trade.created_at, trade.opened_at, trade.closed_at,
                trade.performance.realized_pnl, trade.performance.realized_pnl_percent,
                trade_data
            ))
            
            # ذخیره سفارشات
            for order in trade.orders:
                execution_data = json.dumps(order.execution.__dict__ if order.execution else {})
                
                cursor.execute('''
                    INSERT OR REPLACE INTO trade_orders
                    (order_id, trade_id, order_type, trade_type, price, quantity,
                     status, created_at, filled_at, execution_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    order.order_id, trade.trade_id, order.order_type.value,
                    order.trade_type.value, order.price, order.quantity,
                    order.status, order.created_at, order.filled_at, execution_data
                ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"خطا در ذخیره معامله: {e}")
            return False
    
    def get_trade(self, trade_id: str) -> Optional[Trade]:
        """دریافت معامله"""
        # ابتدا از active trades
        if trade_id in self.active_trades:
            return self.active_trades[trade_id]
        
        # سپس از پایگاه داده
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT trade_data FROM trades WHERE trade_id = ?
            ''', (trade_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                trade_data = json.loads(result[0])
                return Trade.from_dict(trade_data)
            
            return None
            
        except Exception as e:
            print(f"خطا در دریافت معامله: {e}")
            return None
    
    def get_active_trades(self, symbol: Optional[str] = None) -> List[Trade]:
        """دریافت معاملات فعال"""
        trades = list(self.active_trades.values())
        
        if symbol:
            trades = [t for t in trades if t.symbol == symbol]
        
        return [t for t in trades if t.status == TradeStatus.OPEN]
    
    def update_trades_prices(self, symbol: str, current_price: float):
        """به‌روزرسانی قیمت معاملات فعال"""
        for trade in self.get_active_trades(symbol):
            trade.update_current_price(current_price)
            
            # ذخیره اگر وضعیت تغییر کرده
            if trade.status != TradeStatus.OPEN:
                self.save_trade(trade)
                if trade.trade_id in self.active_trades:
                    del self.active_trades[trade.trade_id]
    
    def close_all_trades(self, symbol: str, reason: str = "System close"):
        """بستن همه معاملات یک نماد"""
        for trade in self.get_active_trades(symbol):
            trade.close_trade(trade.current_price, reason)
            self.save_trade(trade)
            if trade.trade_id in self.active_trades:
                del self.active_trades[trade.trade_id]
    
    def get_performance_summary(self, days: int = 30) -> Dict[str, Any]:
        """خلاصه عملکرد معاملات"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_trades,
                    COUNT(CASE WHEN realized_pnl > 0 THEN 1 END) as winning_trades,
                    SUM(realized_pnl) as total_pnl,
                    AVG(realized_pnl_percent) as avg_pnl_percent,
                    MAX(realized_pnl) as best_trade,
                    MIN(realized_pnl) as worst_trade
                FROM trades
                WHERE closed_at >= datetime('now', '-{} days')
                AND status = 'CLOSED'
            '''.format(days))
            
            result = cursor.fetchone()
            conn.close()
            
            if result and result[0] > 0:
                total_trades, winning_trades, total_pnl, avg_pnl_percent, best_trade, worst_trade = result
                
                return {
                    'total_trades': total_trades,
                    'winning_trades': winning_trades or 0,
                    'losing_trades': total_trades - (winning_trades or 0),
                    'win_rate': (winning_trades or 0) / total_trades,
                    'total_pnl': total_pnl or 0,
                    'avg_pnl_percent': avg_pnl_percent or 0,
                    'best_trade': best_trade or 0,
                    'worst_trade': worst_trade or 0,
                    'profit_factor': self._calculate_profit_factor(days)
                }
            
            return {}
            
        except Exception as e:
            print(f"خطا در محاسبه خلاصه عملکرد: {e}")
            return {}
    
    def _calculate_profit_factor(self, days: int) -> float:
        """محاسبه Profit Factor"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT 
                    SUM(CASE WHEN realized_pnl > 0 THEN realized_pnl ELSE 0 END) as gross_profit,
                    SUM(CASE WHEN realized_pnl < 0 THEN ABS(realized_pnl) ELSE 0 END) as gross_loss
                FROM trades
                WHERE closed_at >= datetime('now', '-{} days')
                AND status = 'CLOSED'
            '''.format(days))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                gross_profit, gross_loss = result
                if gross_loss and gross_loss > 0:
                    return gross_profit / gross_loss
            
            return 0.0
            
        except Exception as e:
            print(f"خطا در محاسبه Profit Factor: {e}")
            return 0.0
