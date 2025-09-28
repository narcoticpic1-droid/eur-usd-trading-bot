from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import sqlite3

from .trade_model import Trade, TradeStatus, PositionSide
from .signal_model import TradingSignal

class PortfolioStatus(Enum):
    """وضعیت پورتفولیو"""
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY_STOP = "EMERGENCY_STOP"

class RiskLevel(Enum):
    """سطوح ریسک"""
    VERY_LOW = "VERY_LOW"
    LOW = "LOW" 
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"

@dataclass
class Asset:
    """دارایی در پورتفولیو"""
    symbol: str
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_percent: float = 0.0
    weight: float = 0.0  # درصد از کل پورتفولیو
    
    # Trade tracking
    active_trades: List[str] = field(default_factory=list)  # Trade IDs
    total_trades: int = 0
    winning_trades: int = 0
    
    def update_price(self, new_price: float):
        """به‌روزرسانی قیمت و محاسبه PnL"""
        self.current_price = new_price
        self.market_value = self.quantity * new_price
        
        if self.avg_entry_price > 0 and self.quantity != 0:
            self.unrealized_pnl = (new_price - self.avg_entry_price) * self.quantity
            self.unrealized_pnl_percent = (new_price - self.avg_entry_price) / self.avg_entry_price
    
    def add_position(self, quantity: float, price: float):
        """اضافه کردن پوزیشن (محاسبه میانگین)"""
        if self.quantity == 0:
            self.avg_entry_price = price
            self.quantity = quantity
        else:
            total_cost = (self.avg_entry_price * self.quantity) + (price * quantity)
            self.quantity += quantity
            self.avg_entry_price = total_cost / self.quantity
        
        self.update_price(self.current_price or price)
    
    def remove_position(self, quantity: float) -> float:
        """حذف پوزیشن و محاسبه PnL"""
        if quantity > self.quantity:
            quantity = self.quantity
        
        realized_pnl = (self.current_price - self.avg_entry_price) * quantity
        self.quantity -= quantity
        
        if self.quantity <= 0:
            self.quantity = 0
            self.avg_entry_price = 0
            self.unrealized_pnl = 0
            self.unrealized_pnl_percent = 0
        
        self.update_price(self.current_price)
        return realized_pnl

@dataclass
class PortfolioMetrics:
    """معیارهای عملکرد پورتفولیو"""
    total_value: float = 0.0
    total_equity: float = 0.0
    available_balance: float = 0.0
    
    # P&L
    total_unrealized_pnl: float = 0.0
    total_realized_pnl: float = 0.0
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    monthly_pnl: float = 0.0
    
    # Risk metrics
    total_exposure: float = 0.0
    leverage_ratio: float = 1.0
    portfolio_heat: float = 0.0  # درصد سرمایه در معرض ریسک
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    
    # Performance metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    
    # Risk assessment
    var_95: float = 0.0  # Value at Risk (95%)
    correlation_risk: float = 0.0
    concentration_risk: float = 0.0

class PortfolioManager:
    """مدیر پورتفولیو"""
    
    def __init__(self, initial_balance: float = 10000.0, db_path: str = "portfolio.db"):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.db_path = db_path
        
        # Assets
        self.assets: Dict[str, Asset] = {}
        self.active_trades: Dict[str, Trade] = {}
        
        # Metrics
        self.metrics = PortfolioMetrics()
        self.status = PortfolioStatus.NORMAL
        self.risk_level = RiskLevel.LOW
        
        # Risk settings
        self.max_portfolio_risk = 0.10  # 10% max risk
        self.max_position_size = 0.20   # 20% max per position
        self.max_correlation_exposure = 0.60  # 60% max in correlated assets
        self.max_drawdown_limit = 0.15  # 15% max drawdown
        
        # Performance tracking
        self.equity_curve = []
        self.trade_history = []
        self.daily_returns = []
        
        # Initialize database
        self.init_database()
        
        print(f"✅ Portfolio Manager initialized - Balance: ${initial_balance:,.2f}")
    
    def init_database(self):
        """راه‌اندازی پایگاه داده پورتفولیو"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Portfolio snapshots
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                total_value REAL,
                available_balance REAL,
                total_unrealized_pnl REAL,
                total_realized_pnl REAL,
                max_drawdown REAL,
                sharpe_ratio REAL,
                portfolio_data TEXT
            )
        ''')
        
        # Asset positions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS asset_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT,
                quantity REAL,
                avg_entry_price REAL,
                current_price REAL,
                market_value REAL,
                unrealized_pnl REAL,
                weight REAL
            )
        ''')
        
        # Performance metrics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE,
                total_value REAL,
                daily_return REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                win_rate REAL,
                profit_factor REAL,
                var_95 REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_trade(self, trade: Trade):
        """اضافه کردن معامله به پورتفولیو"""
        self.active_trades[trade.trade_id] = trade
        
        # اضافه کردن به asset
        symbol = trade.symbol
        if symbol not in self.assets:
            self.assets[symbol] = Asset(symbol=symbol)
        
        self.assets[symbol].active_trades.append(trade.trade_id)
        self.assets[symbol].total_trades += 1
        
        # به‌روزرسانی پوزیشن اگر معامله باز شده
        if trade.status == TradeStatus.OPEN:
            if trade.position_side == PositionSide.LONG:
                self.assets[symbol].add_position(trade.quantity, trade.entry_price)
            else:
                # برای short position، quantity منفی ذخیره می‌شود
                self.assets[symbol].add_position(-trade.quantity, trade.entry_price)
    
    def close_trade(self, trade_id: str, exit_price: float):
        """بستن معامله و به‌روزرسانی پورتفولیو"""
        if trade_id not in self.active_trades:
            return False
        
        trade = self.active_trades[trade_id]
        symbol = trade.symbol
        
        if symbol in self.assets:
            # محاسبه PnL
            if trade.position_side == PositionSide.LONG:
                realized_pnl = self.assets[symbol].remove_position(trade.quantity)
            else:
                realized_pnl = self.assets[symbol].remove_position(-trade.quantity)
            
            # به‌روزرسانی موجودی
            self.current_balance += realized_pnl
            self.metrics.total_realized_pnl += realized_pnl
            
            # به‌روزرسانی آمار asset
            if realized_pnl > 0:
                self.assets[symbol].winning_trades += 1
            
            # حذف از لیست فعال
            self.assets[symbol].active_trades.remove(trade_id)
        
        # حذف از معاملات فعال
        del self.active_trades[trade_id]
        
        # اضافه به تاریخچه
        self.trade_history.append({
            'trade_id': trade_id,
            'symbol': symbol,
            'exit_time': datetime.now(),
            'realized_pnl': realized_pnl,
            'exit_price': exit_price
        })
        
        return True
    
    def update_prices(self, price_data: Dict[str, float]):
        """به‌روزرسانی قیمت‌ها و محاسبه معیارها"""
        for symbol, price in price_data.items():
            if symbol in self.assets:
                self.assets[symbol].update_price(price)
        
        # محاسبه معیارهای کلی
        self._calculate_portfolio_metrics()
        
        # به‌روزرسانی وضعیت ریسک
        self._assess_risk_level()
        
        # ذخیره snapshot