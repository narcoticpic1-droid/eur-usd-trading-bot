import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import datetime
from dataclasses import dataclass
import json
import sqlite3
import config

@dataclass
class Trade:
    """کلاس نمایندگی یک معامله"""
    id: str
    symbol: str
    strategy: str
    entry_time: datetime.datetime
    entry_price: float
    direction: int  # 1 برای خرید، -1 برای فروش
    size: float
    leverage: int
    stop_loss: float
    take_profit: List[float]
    exit_time: Optional[datetime.datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    pnl_percentage: Optional[float] = None
    fees: float = 0.0
    slippage: float = 0.0
    max_favorable: float = 0.0
    max_adverse: float = 0.0
    duration_minutes: Optional[int] = None

@dataclass
class BacktestConfig:
    """تنظیمات بک‌تست"""
    start_date: datetime.datetime
    end_date: datetime.datetime
    initial_balance: float = 10000.0
    commission_rate: float = 0.001  # 0.1%
    slippage_bps: float = 1.0  # 1 basis point
    max_leverage: int = 10
    risk_per_trade: float = 0.02  # 2%
    max_concurrent_trades: int = 3
    enable_stop_loss: bool = True
    enable_take_profit: bool = True
    risk_free_rate: float = 0.02  # 2% سالانه

class BacktestEngine:
    """
    موتور بک‌تست حرفه‌ای برای تست استراتژی‌ها
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.current_balance = config.initial_balance
        self.initial_balance = config.initial_balance
        self.trades: List[Trade] = []
        self.open_trades: List[Trade] = []
        self.equity_curve = []
        self.daily_returns = []
        
        # آمار بک‌تست
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_balance = config.initial_balance
        
        # تنظیمات پیشرفته
        self.enable_compound_returns = True
        self.transaction_costs = True
        self.realistic_execution = True

    def run_backtest(self, data: Dict[str, pd.DataFrame], 
                    strategy_func: callable, 
                    strategy_params: Dict = None) -> Dict:
        """اجرای بک‌تست اصلی"""
        
        print(f"🔄 شروع بک‌تست از {self.config.start_date} تا {self.config.end_date}")
        print(f"💰 موجودی اولیه: ${self.initial_balance:,.2f}")
        
        # ترکیب داده‌ها از همه نمادها
        combined_data = self._combine_data_sources(data)
        
        # فیلتر بر اساس تاریخ
        combined_data = combined_data[
            (combined_data.index >= self.config.start_date) & 
            (combined_data.index <= self.config.end_date)
        ]
        
        if combined_data.empty:
            raise ValueError("داده‌ای در بازه زمانی مشخص شده موجود نیست")
        
        print(f"📊 تعداد کندل‌ها: {len(combined_data)}")
        
        # حلقه اصلی بک‌تست
        for i, (timestamp, row) in enumerate(combined_data.iterrows()):
            
            # بروزرسانی معاملات باز
            self._update_open_trades(timestamp, row)
            
            # بررسی سیگنال‌های جدید
            if strategy_params is None:
                strategy_params = {}
            
            # دریافت داده‌های تاریخی تا این نقطه
            historical_data = self._get_historical_data(combined_data, i)
            
            # اجرای استراتژی
            signals = strategy_func(historical_data, **strategy_params)
            
            if signals:
                for signal in signals if isinstance(signals, list) else [signals]:
                    self._process_signal(signal, timestamp, row)
            
            # ثبت equity curve
            self._record_equity_point(timestamp)
            
            # نمایش پیشرفت
            if i % 1000 == 0:
                progress = (i / len(combined_data)) * 100
                print(f"پیشرفت: {progress:.1f}% - موجودی: ${self.current_balance:,.2f}")
        
        # بستن همه معاملات باز
        self._close_all_open_trades(combined_data.iloc[-1])
        
        # محاسبه نتایج نهایی
        results = self._calculate_final_results()
        
        print(f"✅ بک‌تست تکمیل شد")
        print(f"📈 تعداد کل معاملات: {self.total_trades}")
        print(f"💹 بازدهی کل: {results['total_return']:.2f}%")
        print(f"📉 حداکثر افت: {results['max_drawdown']:.2f}%")
        
        return results

    def _combine_data_sources(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """ترکیب داده‌های چندین نماد"""
        if not data:
            raise ValueError("داده‌ای برای بک‌تست ارائه نشده")
        
        # اگر فقط یک نماد است
        if len(data) == 1:
            symbol = list(data.keys())[0]
            df = data[symbol].copy()
            df['symbol'] = symbol
            return df
        
        # ترکیب چندین نماد
        combined_frames = []
        for symbol, df in data.items():
            df_copy = df.copy()
            df_copy['symbol'] = symbol
            combined_frames.append(df_copy)
        
        # ترکیب بر اساس timestamp
        combined = pd.concat(combined_frames, sort=True)
        combined = combined.sort_index()
        
        return combined

    def _get_historical_data(self, full_data: pd.DataFrame, current_index: int) -> Dict:
        """دریافت داده‌های تاریخی تا نقطه فعلی"""
        historical = full_data.iloc[:current_index+1]
        
        # تفکیک بر اساس نماد
        result = {}
        if 'symbol' in historical.columns:
            for symbol in historical['symbol'].unique():
                if pd.notna(symbol):
                    symbol_data = historical[historical['symbol'] == symbol].copy()
                    symbol_data = symbol_data.drop('symbol', axis=1)
                    result[symbol] = symbol_data
        else:
            # یک نماد
            result['main'] = historical
        
        return result

    def _process_signal(self, signal: Dict, timestamp: datetime.datetime, 
                       current_data: pd.Series):
        """پردازش سیگنال معاملاتی"""
        
        # بررسی محدودیت‌ها
        if len(self.open_trades) >= self.config.max_concurrent_trades:
            return
        
        if not self._validate_signal(signal):
            return
        
        # محاسبه اندازه پوزیشن
        position_size = self._calculate_position_size(signal, current_data)
        
        if position_size <= 0:
            return
        
        # ایجاد معامله جدید
        trade = Trade(
            id=f"trade_{len(self.trades)}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
            symbol=signal.get('symbol', 'UNKNOWN'),
            strategy=signal.get('strategy', 'UNKNOWN'),
            entry_time=timestamp,
            entry_price=self._calculate_entry_price(signal, current_data),
            direction=1 if signal.get('signal', 0) > 0 else -1,
            size=position_size,
            leverage=signal.get('leverage', 1),
            stop_loss=signal.get('stop_loss', 0),
            take_profit=signal.get('take_profits', []),
            fees=self._calculate_fees(position_size, current_data)
        )
        
        # اضافه کردن slippage
        if self.realistic_execution:
            trade.slippage = self._calculate_slippage(trade, current_data)
            trade.entry_price += trade.slippage * trade.direction
        
        # به‌روزرسانی موجودی
        total_cost = trade.size * trade.entry_price / trade.leverage + trade.fees
        if total_cost <= self.current_balance:
            self.current_balance -= total_cost
            self.open_trades.append(trade)
            self.total_trades += 1
            
            print(f"🔵 معامله جدید: {trade.symbol} {trade.direction} @ ${trade.entry_price:.4f}")

    def _update_open_trades(self, timestamp: datetime.datetime, current_data: pd.Series):
        """بروزرسانی معاملات باز"""
        trades_to_close = []
        
        for trade in self.open_trades:
            # بروزرسانی قیمت فعلی برای این نماد
            if 'symbol' in current_data.index and current_data['symbol'] != trade.symbol:
                continue
            
            current_price = current_data['close']
            
            # محاسبه P&L فعلی
            pnl = self._calculate_unrealized_pnl(trade, current_price)
            
            # بروزرسانی حداکثر سود/ضرر
            if pnl > trade.max_favorable:
                trade.max_favorable = pnl
            if pnl < trade.max_adverse:
                trade.max_adverse = pnl
            
            # بررسی شرایط خروج
            exit_reason = self._check_exit_conditions(trade, current_price, timestamp)
            
            if exit_reason:
                trade.exit_time = timestamp
                trade.exit_price = current_price
                trade.exit_reason = exit_reason
                trade.pnl = pnl
                trade.pnl_percentage = (pnl / (trade.size * trade.entry_price)) * 100
                trade.duration_minutes = int((timestamp - trade.entry_time).total_seconds() / 60)
                
                # بازگشت سرمایه به موجودی
                returned_capital = trade.size * trade.entry_price / trade.leverage
                self.current_balance += returned_capital + pnl - trade.fees
                
                # آمار
                if pnl > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
                
                self.total_pnl += pnl
                
                trades_to_close.append(trade)
                
                print(f"🔴 بسته شدن معامله: {trade.symbol} P&L: ${pnl:.2f} ({exit_reason})")
        
        # حذف معاملات بسته شده
        for trade in trades_to_close:
            self.open_trades.remove(trade)
            self.trades.append(trade)

    def _check_exit_conditions(self, trade: Trade, current_price: float, 
                              timestamp: datetime.datetime) -> Optional[str]:
        """بررسی شرایط خروج از معامله"""
        
        # Stop Loss
        if self.config.enable_stop_loss and trade.stop_loss > 0:
            if ((trade.direction > 0 and current_price <= trade.stop_loss) or
                (trade.direction < 0 and current_price >= trade.stop_loss)):
                return "STOP_LOSS"
        
        # Take Profit
        if self.config.enable_take_profit and trade.take_profit:
            for tp in trade.take_profit:
                if ((trade.direction > 0 and current_price >= tp) or
                    (trade.direction < 0 and current_price <= tp)):
                    return "TAKE_PROFIT"
        
        # Time-based exit (مثلاً برای اسکالپینگ)
        duration = timestamp - trade.entry_time
        if trade.strategy == "Scalping Strategy" and duration.total_seconds() > 3600:  # 1 ساعت
            return "TIME_EXIT"
        
        return None

    def _calculate_position_size(self, signal: Dict, current_data: pd.Series) -> float:
        """محاسبه اندازه پوزیشن"""
        current_price = current_data['close']
        stop_loss = signal.get('stop_loss', 0)
        
        if stop_loss <= 0:
            return 0
        
        # محاسبه ریسک
        risk_amount = self.current_balance * self.config.risk_per_trade
        price_diff = abs(current_price - stop_loss)
        
        if price_diff <= 0:
            return 0
        
        # اندازه پوزیشن بر اساس ریسک
        position_size = risk_amount / price_diff
        
        # محدودیت leverage
        max_position = (self.current_balance * self.config.max_leverage) / current_price
        position_size = min(position_size, max_position)
        
        return max(0, position_size)

    def _calculate_entry_price(self, signal: Dict, current_data: pd.Series) -> float:
        """محاسبه قیمت ورود (شامل slippage احتمالی)"""
        return current_data['close']

    def _calculate_fees(self, position_size: float, current_data: pd.Series) -> float:
        """محاسبه کارمزد معاملات"""
        if not self.transaction_costs:
            return 0.0
        
        trade_value = position_size * current_data['close']
        return trade_value * self.config.commission_rate

    def _calculate_slippage(self, trade: Trade, current_data: pd.Series) -> float:
        """محاسبه slippage"""
        if not self.realistic_execution:
            return 0.0
        
        # slippage بر اساس volatility و حجم
        volatility_factor = 1.0
        if 'atr' in current_data.index:
            volatility_factor = current_data['atr'] / current_data['close']
        
        base_slippage = current_data['close'] * (self.config.slippage_bps / 10000)
        return base_slippage * (1 + volatility_factor)

    def _calculate_unrealized_pnl(self, trade: Trade, current_price: float) -> float:
        """محاسبه سود/ضرر تحقق نیافته"""
        price_diff = (current_price - trade.entry_price) * trade.direction
        return price_diff * trade.size

    def _validate_signal(self, signal: Dict) -> bool:
        """اعتبارسنجی سیگنال"""
        required_fields = ['signal', 'symbol', 'confidence']
        
        for field in required_fields:
            if field not in signal:
                return False
        
        # بررسی حداقل اطمینان
        if signal.get('confidence', 0) < 0.6:
            return False
        
        return True

    def _record_equity_point(self, timestamp: datetime.datetime):
        """ثبت نقطه equity curve"""
        # محاسبه ارزش کل پورتفولیو
        total_equity = self.current_balance
        
        # اضافه کردن ارزش معاملات باز
        for trade in self.open_trades:
            # فرض می‌کنیم قیمت فعلی همان آخرین قیمت بسته شدن است
            # (در عمل باید قیمت real-time داشته باشیم)
            pass
        
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': total_equity,
            'balance': self.current_balance,
            'open_trades': len(self.open_trades)
        })
        
        # بروزرسانی peak و drawdown
        if total_equity > self.peak_balance:
            self.peak_balance = total_equity
        
        current_drawdown = (self.peak_balance - total_equity) / self.peak_balance
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown

    def _close_all_open_trades(self, final_data: pd.Series):
        """بستن همه معاملات باز در پایان بک‌تست"""
        final_price = final_data['close']
        final_timestamp = final_data.name
        
        for trade in self.open_trades.copy():
            trade.exit_time = final_timestamp
            trade.exit_price = final_price
            trade.exit_reason = "BACKTEST_END"
            
            pnl = self._calculate_unrealized_pnl(trade, final_price)
            trade.pnl = pnl
            trade.pnl_percentage = (pnl / (trade.size * trade.entry_price)) * 100
            
            self.current_balance += pnl
            self.total_pnl += pnl
            
            if pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            
            self.trades.append(trade)
        
        self.open_trades.clear()

    def _calculate_final_results(self) -> Dict:
        """محاسبه نتایج نهایی بک‌تست"""
        
        # محاسبه بازدهی کل
        total_return = ((self.current_balance - self.initial_balance) / self.initial_balance) * 100
        
        # Win Rate
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        # میانگین سود/ضرر
        avg_win = np.mean([t.pnl for t in self.trades if t.pnl > 0]) if self.winning_trades > 0 else 0
        avg_loss = np.mean([t.pnl for t in self.trades if t.pnl < 0]) if self.losing_trades > 0 else 0
        
        # Profit Factor
        total_wins = sum([t.pnl for t in self.trades if t.pnl > 0])
        total_losses = abs(sum([t.pnl for t in self.trades if t.pnl < 0]))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Sharpe Ratio (ساده شده)
        if len(self.equity_curve) > 1:
            returns = []
            for i in range(1, len(self.equity_curve)):
                ret = (self.equity_curve[i]['equity'] - self.equity_curve[i-1]['equity']) / self.equity_curve[i-1]['equity']
                returns.append(ret)
            
            if returns:
                avg_return = np.mean(returns)
                return_std = np.std(returns)
                sharpe_ratio = (avg_return - self.config.risk_free_rate/252) / return_std if return_std > 0 else 0
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        return {
            'total_return': total_return,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': self.max_drawdown * 100,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'sharpe_ratio': sharpe_ratio,
            'final_balance': self.current_balance,
            'total_pnl': self.total_pnl,
            'start_date': self.config.start_date,
            'end_date': self.config.end_date,
            'trades': [self._trade_to_dict(t) for t in self.trades],
            'equity_curve': self.equity_curve
        }

    def _trade_to_dict(self, trade: Trade) -> Dict:
        """تبدیل Trade به dictionary"""
        return {
            'id': trade.id,
            'symbol': trade.symbol,
            'strategy': trade.strategy,
            'entry_time': trade.entry_time.isoformat() if trade.entry_time else None,
            'entry_price': trade.entry_price,
            'direction': trade.direction,
            'size': trade.size,
            'leverage': trade.leverage,
            'exit_time': trade.exit_time.isoformat() if trade.exit_time else None,
            'exit_price': trade.exit_price,
            'exit_reason': trade.exit_reason,
            'pnl': trade.pnl,
            'pnl_percentage': trade.pnl_percentage,
            'duration_minutes': trade.duration_minutes,
            'fees': trade.fees,
            'slippage': trade.slippage
        }

    def save_results(self, filepath: str):
        """ذخیره نتایج بک‌تست"""
        results = self._calculate_final_results()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"💾 نتایج بک‌تست در {filepath} ذخیره شد")

    def get_performance_summary(self) -> str:
        """خلاصه عملکرد به صورت متنی"""
        results = self._calculate_final_results()
        
        summary = f"""
📊 خلاصه نتایج بک‌تست
{'='*50}
📅 بازه زمانی: {results['start_date'].strftime('%Y-%m-%d')} تا {results['end_date'].strftime('%Y-%m-%d')}
💰 موجودی ابتدایی: ${self.initial_balance:,.2f}
💰 موجودی نهایی: ${results['final_balance']:,.2f}
📈 بازدهی کل: {results['total_return']:.2f}%
📉 حداکثر افت: {results['max_drawdown']:.2f}%

📊 آمار معاملات:
   🔢 تعداد کل: {results['total_trades']}
   ✅ موفق: {results['winning_trades']} ({results['win_rate']:.1f}%)
   ❌ ناموفق: {results['losing_trades']}
   💹 ضریب سود: {results['profit_factor']:.2f}
   
📊 میانگین سود/ضرر:
   💚 میانگین سود: ${results['avg_win']:.2f}
   🔴 میانگین ضرر: ${results['avg_loss']:.2f}
   
📊 شاخص‌های ریسک:
   📏 Sharpe Ratio: {results['sharpe_ratio']:.3f}
   📉 حداکثر افت: {results['max_drawdown']:.2f}%
"""
        return summary
