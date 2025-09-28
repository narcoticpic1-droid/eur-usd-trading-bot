import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

class PerformanceMetrics:
    """
    محاسبه‌گر معیارهای عملکرد حرفه‌ای
    """
    
    def __init__(self, trades: List[Dict], equity_curve: List[Dict], 
                 initial_balance: float, risk_free_rate: float = 0.02):
        self.trades = trades
        self.equity_curve = equity_curve
        self.initial_balance = initial_balance
        self.risk_free_rate = risk_free_rate
        
        # تبدیل به DataFrame برای تحلیل آسان‌تر
        self.trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        self.equity_df = pd.DataFrame(equity_curve) if equity_curve else pd.DataFrame()
        
        if not self.equity_df.empty:
            self.equity_df['timestamp'] = pd.to_datetime(self.equity_df['timestamp'])
            self.equity_df.set_index('timestamp', inplace=True)

    def calculate_all_metrics(self) -> Dict:
        """محاسبه تمام معیارهای عملکرد"""
        metrics = {}
        
        # معیارهای پایه
        metrics.update(self._basic_metrics())
        
        # معیارهای بازدهی
        metrics.update(self._return_metrics())
        
        # معیارهای ریسک
        metrics.update(self._risk_metrics())
        
        # معیارهای Drawdown
        metrics.update(self._drawdown_metrics())
        
        # معیارهای توزیعی
        metrics.update(self._distribution_metrics())
        
        # معیارهای زمانی
        metrics.update(self._time_based_metrics())
        
        # معیارهای پیشرفته
        metrics.update(self._advanced_metrics())
        
        return metrics

    def _basic_metrics(self) -> Dict:
        """معیارهای پایه"""
        if self.trades_df.empty:
            return {}
        
        total_trades = len(self.trades_df)
        winning_trades = len(self.trades_df[self.trades_df['pnl'] > 0])
        losing_trades = len(self.trades_df[self.trades_df['pnl'] < 0])
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
            'loss_rate': (losing_trades / total_trades * 100) if total_trades > 0 else 0
        }

    def _return_metrics(self) -> Dict:
        """معیارهای بازدهی"""
        if self.trades_df.empty or self.equity_df.empty:
            return {}
        
        # بازدهی کل
        final_equity = self.equity_df['equity'].iloc[-1]
        total_return = ((final_equity - self.initial_balance) / self.initial_balance) * 100
        
        # بازدهی سالانه
        start_date = self.equity_df.index[0]
        end_date = self.equity_df.index[-1]
        days = (end_date - start_date).days
        years = days / 365.25
        
        if years > 0:
            annual_return = ((final_equity / self.initial_balance) ** (1/years) - 1) * 100
        else:
            annual_return = 0
        
        # محاسبه بازدهی روزانه
        daily_returns = self.equity_df['equity'].pct_change().dropna()
        
        return {
            'total_return_pct': total_return,
            'annual_return_pct': annual_return,
            'avg_daily_return_pct': daily_returns.mean() * 100,
            'total_pnl': self.trades_df['pnl'].sum(),
            'avg_trade_pnl': self.trades_df['pnl'].mean(),
            'median_trade_pnl': self.trades_df['pnl'].median()
        }

    def _risk_metrics(self) -> Dict:
        """معیارهای ریسک"""
        if self.equity_df.empty:
            return {}
        
        daily_returns = self.equity_df['equity'].pct_change().dropna()
        
        if len(daily_returns) < 2:
            return {}
        
        # نوسانات (Volatility)
        daily_volatility = daily_returns.std()
        annual_volatility = daily_volatility * np.sqrt(252) * 100
        
        # Sharpe Ratio
        excess_return = daily_returns.mean() - (self.risk_free_rate / 252)
        sharpe_ratio = excess_return / daily_volatility if daily_volatility > 0 else 0
        annual_sharpe = sharpe_ratio * np.sqrt(252)
        
        # Sortino Ratio (فقط نوسانات منفی)
        negative_returns = daily_returns[daily_returns < 0]
        downside_deviation = negative_returns.std()
        sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else 0
        annual_sortino = sortino_ratio * np.sqrt(252)
        
        # Calmar Ratio
        max_dd = self._calculate_max_drawdown()
        calmar_ratio = (annual_volatility / 100) / (max_dd / 100) if max_dd > 0 else 0
        
        return {
            'daily_volatility_pct': daily_volatility * 100,
            'annual_volatility_pct': annual_volatility,
            'sharpe_ratio': annual_sharpe,
            'sortino_ratio': annual_sortino,
            'calmar_ratio': calmar_ratio,
            'var_95_pct': np.percentile(daily_returns, 5) * 100,  # Value at Risk
            'cvar_95_pct': daily_returns[daily_returns <= np.percentile(daily_returns, 5)].mean() * 100  # Conditional VaR
        }

    def _drawdown_metrics(self) -> Dict:
        """معیارهای Drawdown"""
        if self.equity_df.empty:
            return {}
        
        equity_series = self.equity_df['equity']
        
        # محاسبه running maximum
        running_max = equity_series.expanding().max()
        
        # محاسبه drawdown
        drawdown = (equity_series - running_max) / running_max * 100
        
        # حداکثر drawdown
        max_drawdown = drawdown.min()
        
        # مدت زمان حداکثر drawdown
        max_dd_start = None
        max_dd_end = None
        max_dd_duration = 0
        
        current_dd_start = None
        for i, dd in enumerate(drawdown):
            if dd < -0.01 and current_dd_start is None:  # شروع drawdown
                current_dd_start = i
            elif dd >= -0.01 and current_dd_start is not None:  # پایان drawdown
                duration = i - current_dd_start
                if duration > max_dd_duration:
                    max_dd_duration = duration
                    max_dd_start = current_dd_start
                    max_dd_end = i
                current_dd_start = None
        
        # تعداد دوره‌های drawdown
        drawdown_periods = 0
        in_drawdown = False
        for dd in drawdown:
            if dd < -0.01 and not in_drawdown:
                drawdown_periods += 1
                in_drawdown = True
            elif dd >= -0.01:
                in_drawdown = False
        
        return {
            'max_drawdown_pct': abs(max_drawdown),
            'max_drawdown_duration_days': max_dd_duration,
            'avg_drawdown_pct': abs(drawdown[drawdown < 0].mean()) if len(drawdown[drawdown < 0]) > 0 else 0,
            'drawdown_periods': drawdown_periods,
            'current_drawdown_pct': abs(drawdown.iloc[-1]) if drawdown.iloc[-1] < 0 else 0
        }

    def _distribution_metrics(self) -> Dict:
        """معیارهای توزیعی"""
        if self.trades_df.empty:
            return {}
        
        pnl_series = self.trades_df['pnl']
        
        # آمار توصیفی
        skewness = pnl_series.skew()  # چولگی
        kurtosis = pnl_series.kurtosis()  # کشیدگی
        
        # میانه و انحراف میانه
        percentiles = np.percentile(pnl_series, [10, 25, 50, 75, 90])
        
        # Best/Worst trades
        best_trade = pnl_series.max()
        worst_trade = pnl_series.min()
        
        # Consecutive wins/losses
        consecutive_wins, consecutive_losses = self._calculate_consecutive_trades()
        
        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'pnl_10th_percentile': percentiles[0],
            'pnl_25th_percentile': percentiles[1],
            'pnl_75th_percentile': percentiles[3],
            'pnl_90th_percentile': percentiles[4],
            'max_consecutive_wins': consecutive_wins,
            'max_consecutive_losses': consecutive_losses
        }

    def _time_based_metrics(self) -> Dict:
        """معیارهای زمانی"""
        if self.trades_df.empty:
            return {}
        
        # مدت زمان معاملات
        durations = self.trades_df['duration_minutes'].dropna()
        
        if durations.empty:
            return {}
        
        # تحلیل بر اساس روز هفته (اگر داده کافی داشته باشیم)
        if 'entry_time' in self.trades_df.columns:
            entry_times = pd.to_datetime(self.trades_df['entry_time'])
            day_performance = {}
            for day in range(7):  # 0=Monday, 6=Sunday
                day_trades = self.trades_df[entry_times.dt.dayofweek == day]
                if not day_trades.empty:
                    day_performance[day] = {
                        'trades': len(day_trades),
                        'avg_pnl': day_trades['pnl'].mean(),
                        'win_rate': len(day_trades[day_trades['pnl'] > 0]) / len(day_trades) * 100
                    }
        
        return {
            'avg_trade_duration_minutes': durations.mean(),
            'median_trade_duration_minutes': durations.median(),
            'min_trade_duration_minutes': durations.min(),
            'max_trade_duration_minutes': durations.max(),
            'trades_per_day': len(self.trades_df) / len(self.equity_df) if len(self.equity_df) > 0 else 0
        }

    def _advanced_metrics(self) -> Dict:
        """معیارهای پیشرفته"""
        if self.trades_df.empty or self.equity_df.empty:
            return {}
        
        # Profit Factor
        winning_pnl = self.trades_df[self.trades_df['pnl'] > 0]['pnl'].sum()
        losing_pnl = abs(self.trades_df[self.trades_df['pnl'] < 0]['pnl'].sum())
        profit_factor = winning_pnl / losing_pnl if losing_pnl > 0 else float('inf')
        
        # Recovery Factor
        max_drawdown = abs(self._calculate_max_drawdown())
        total_return = ((self.equity_df['equity'].iloc[-1] - self.initial_balance) / self.initial_balance) * 100
        recovery_factor = total_return / max_drawdown if max_drawdown > 0 else float('inf')
        
        # Expectancy
        win_rate = len(self.trades_df[self.trades_df['pnl'] > 0]) / len(self.trades_df)
        avg_win = self.trades_df[self.trades_df['pnl'] > 0]['pnl'].mean() if len(self.trades_df[self.trades_df['pnl'] > 0]) > 0 else 0
        avg_loss = self.trades_df[self.trades_df['pnl'] < 0]['pnl'].mean() if len(self.trades_df[self.trades_df['pnl'] < 0]) > 0 else 0
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        # Kelly Criterion
        if avg_loss != 0:
            kelly_percentage = (win_rate - ((1 - win_rate) / (avg_win / abs(avg_loss)))) * 100
        else:
            kelly_percentage = 0
        
        # Information Ratio (اگر benchmark داشته باشیم)
        daily_returns = self.equity_df['equity'].pct_change().dropna()
        tracking_error = daily_returns.std() * np.sqrt(252)
        information_ratio = (daily_returns.mean() * 252) / tracking_error if tracking_error > 0 else 0
        
        return {
            'profit_factor': profit_factor,
            'recovery_factor': recovery_factor,
            'expectancy': expectancy,
            'kelly_percentage': kelly_percentage,
            'information_ratio': information_ratio,
            'ulcer_index': self._calculate_ulcer_index(),
            'pain_index': self._calculate_pain_index()
        }

    def _calculate_max_drawdown(self) -> float:
        """محاسبه حداکثر drawdown"""
        if self.equity_df.empty:
            return 0
        
        equity_series = self.equity_df['equity']
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max * 100
        return abs(drawdown.min())

    def _calculate_consecutive_trades(self) -> Tuple[int, int]:
        """محاسبه حداکثر برد/باخت پشت سر هم"""
        if self.trades_df.empty:
            return 0, 0
        
        results = (self.trades_df['pnl'] > 0).astype(int)
        
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0
        
        for result in results:
            if result == 1:  # برد
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:  # باخت
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
        
        return max_wins, max_losses

    def _calculate_ulcer_index(self) -> float:
        """محاسبه Ulcer Index"""
        if self.equity_df.empty:
            return 0
        
        equity_series = self.equity_df['equity']
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max * 100
        
        # Ulcer Index = sqrt(mean(drawdown^2))
        ulcer_index = np.sqrt((drawdown ** 2).mean())
        return ulcer_index

    def _calculate_pain_index(self) -> float:
        """محاسبه Pain Index"""
        if self.equity_df.empty:
            return 0
        
        equity_series = self.equity_df['equity']
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max * 100
        
        # Pain Index = mean(abs(drawdown))
        pain_index = abs(drawdown).mean()
        return pain_index

    def generate_performance_report(self) -> str:
        """تولید گزارش کامل عملکرد"""
        metrics = self.calculate_all_metrics()
        
        report = f"""
📊 گزارش کامل عملکرد
{'='*60}

📈 معیارهای بازدهی:
   • بازدهی کل: {metrics.get('total_return_pct', 0):.2f}%
   • بازدهی سالانه: {metrics.get('annual_return_pct', 0):.2f}%
   • میانگین بازدهی روزانه: {metrics.get('avg_daily_return_pct', 0):.3f}%
   • سود کل: ${metrics.get('total_pnl', 0):,.2f}
   • میانگین سود هر معامله: ${metrics.get('avg_trade_pnl', 0):.2f}

📊 معیارهای معاملاتی:
   • تعداد کل معاملات: {metrics.get('total_trades', 0)}
   • معاملات موفق: {metrics.get('winning_trades', 0)}
   • معاملات ناموفق: {metrics.get('losing_trades', 0)}
   • نرخ موفقیت: {metrics.get('win_rate', 0):.1f}%
   • ضریب سود: {metrics.get('profit_factor', 0):.2f}

📉 معیارهای ریسک:
   • حداکثر افت: {metrics.get('max_drawdown_pct', 0):.2f}%
   • نوسانات سالانه: {metrics.get('annual_volatility_pct', 0):.2f}%
   • نسبت شارپ: {metrics.get('sharpe_ratio', 0):.3f}
   • نسبت سورتینو: {metrics.get('sortino_ratio', 0):.3f}
   • نسبت کالمار: {metrics.get('calmar_ratio', 0):.3f}

🎯 معیارهای پیشرفته:
   • ضریب بازیابی: {metrics.get('recovery_factor', 0):.2f}
   • انتظار ریاضی: ${metrics.get('expectancy', 0):.2f}
   • درصد کلی: {metrics.get('kelly_percentage', 0):.1f}%
   • شاخص اولسر: {metrics.get('ulcer_index', 0):.2f}

🏆 رکوردها:
   • بهترین معامله: ${metrics.get('best_trade', 0):.2f}
   • بدترین معامله: ${metrics.get('worst_trade', 0):.2f}
   • حداکثر برد پشت سر هم: {metrics.get('max_consecutive_wins', 0)}
   • حداکثر باخت پشت سر هم: {metrics.get('max_consecutive_losses', 0)}

⏱️ معیارهای زمانی:
   • میانگین مدت معامله: {metrics.get('avg_trade_duration_minutes', 0):.0f} دقیقه
   • معاملات در روز: {metrics.get('trades_per_day', 0):.1f}
"""
        return report

    def plot_performance_charts(self, save_path: Optional[str] = None):
        """رسم نمودارهای عملکرد"""
        if self.equity_df.empty or self.trades_df.empty:
            print("داده کافی برای رسم نمودار وجود ندارد")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('تحلیل عملکرد معاملات', fontsize=16, fontweight='bold')
        
        # نمودار 1: Equity Curve
        axes[0, 0].plot(self.equity_df.index, self.equity_df['equity'], 
                       linewidth=2, color='blue', label='Equity')
        axes[0, 0].axhline(y=self.initial_balance, color='red', linestyle='--', 
                          alpha=0.7, label='سرمایه اولیه')
        axes[0, 0].set_title('منحنی رشد سرمایه')
        axes[0, 0].set_ylabel('ارزش پورتفولیو ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # نمودار 2: Drawdown
        equity_series = self.equity_df['equity']
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max * 100
        
        axes[0, 1].fill_between(self.equity_df.index, drawdown, 0, 
                               color='red', alpha=0.3)
        axes[0, 1].plot(self.equity_df.index, drawdown, color='red', linewidth=1)
        axes[0, 1].set_title('منحنی افت سرمایه (Drawdown)')
        axes[0, 1].set_ylabel('افت (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # نمودار 3: توزیع P&L
        axes[1, 0].hist(self.trades_df['pnl'], bins=30, alpha=0.7, 
                       color='green', edgecolor='black')
        axes[1, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        axes[1, 0].set_title('توزیع سود و ضرر معاملات')
        axes[1, 0].set_xlabel('P&L ($)')
        axes[1, 0].set_ylabel('تعداد')
        axes[1, 0].grid(True, alpha=0.3)
        
        # نمودار 4: P&L تجمعی
        cumulative_pnl = self.trades_df['pnl'].cumsum()
        trade_numbers = range(1, len(cumulative_pnl) + 1)
        
        colors = ['green' if pnl >= 0 else 'red' for pnl in cumulative_pnl]
        axes[1, 1].plot(trade_numbers, cumulative_pnl, linewidth=2, color='blue')
        axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[1, 1].set_title('سود و ضرر تجمعی')
        axes[1, 1].set_xlabel('شماره معامله')
        axes[1, 1].set_ylabel('P&L تجمعی ($)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"نمودارها در {save_path} ذخیره شدند")
        
        plt.show()
