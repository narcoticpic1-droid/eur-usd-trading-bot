import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json

class PortfolioAnalyzer:
    """
    تحلیلگر پورتفولیو چند نمادی
    """
    
    def __init__(self, trades_data: Dict[str, List[Dict]], 
                 initial_balance: float = 10000.0):
        self.trades_data = trades_data  # {symbol: [trades]}
        self.initial_balance = initial_balance
        
        # تبدیل به DataFrame برای تحلیل
        self.portfolio_df = self._create_portfolio_dataframe()
        self.symbols = list(trades_data.keys())
        
    def _create_portfolio_dataframe(self) -> pd.DataFrame:
        """ایجاد DataFrame کلی پورتفولیو"""
        all_trades = []
        
        for symbol, trades in self.trades_data.items():
            for trade in trades:
                trade_copy = trade.copy()
                trade_copy['symbol'] = symbol
                all_trades.append(trade_copy)
        
        if not all_trades:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_trades)
        
        # تبدیل تاریخ‌ها
        if 'entry_time' in df.columns:
            df['entry_time'] = pd.to_datetime(df['entry_time'])
        if 'exit_time' in df.columns:
            df['exit_time'] = pd.to_datetime(df['exit_time'])
            
        return df.sort_values('entry_time') if 'entry_time' in df.columns else df

    def analyze_portfolio_performance(self) -> Dict:
        """تحلیل کلی عملکرد پورتفولیو"""
        if self.portfolio_df.empty:
            return {}
        
        analysis = {}
        
        # عملکرد کلی
        analysis.update(self._overall_performance())
        
        # تحلیل per symbol
        analysis['symbol_analysis'] = self._per_symbol_analysis()
        
        # تحلیل همبستگی
        analysis['correlation_analysis'] = self._correlation_analysis()
        
        # تحلیل diversification
        analysis['diversification_metrics'] = self._diversification_analysis()
        
        # تحلیل risk contribution
        analysis['risk_contribution'] = self._risk_contribution_analysis()
        
        # تحلیل زمانی
        analysis['time_analysis'] = self._time_based_analysis()
        
        return analysis

    def _overall_performance(self) -> Dict:
        """عملکرد کلی پورتفولیو"""
        total_pnl = self.portfolio_df['pnl'].sum()
        total_trades = len(self.portfolio_df)
        winning_trades = len(self.portfolio_df[self.portfolio_df['pnl'] > 0])
        
        final_balance = self.initial_balance + total_pnl
        total_return = (total_pnl / self.initial_balance) * 100
        
        return {
            'total_pnl': total_pnl,
            'total_return_pct': total_return,
            'final_balance': final_balance,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
            'avg_trade_pnl': total_pnl / total_trades if total_trades > 0 else 0
        }

    def _per_symbol_analysis(self) -> Dict:
        """تحلیل عملکرد هر نماد"""
        symbol_stats = {}
        
        for symbol in self.symbols:
            symbol_trades = self.portfolio_df[self.portfolio_df['symbol'] == symbol]
            
            if symbol_trades.empty:
                continue
            
            symbol_pnl = symbol_trades['pnl'].sum()
            symbol_trades_count = len(symbol_trades)
            symbol_wins = len(symbol_trades[symbol_trades['pnl'] > 0])
            
            symbol_stats[symbol] = {
                'total_pnl': symbol_pnl,
                'total_trades': symbol_trades_count,
                'winning_trades': symbol_wins,
                'win_rate': (symbol_wins / symbol_trades_count * 100) if symbol_trades_count > 0 else 0,
                'avg_trade_pnl': symbol_pnl / symbol_trades_count if symbol_trades_count > 0 else 0,
                'best_trade': symbol_trades['pnl'].max(),
                'worst_trade': symbol_trades['pnl'].min(),
                'pnl_std': symbol_trades['pnl'].std(),
                'sharpe_ratio': self._calculate_symbol_sharpe(symbol_trades)
            }
        
        return symbol_stats

    def _correlation_analysis(self) -> Dict:
        """تحلیل همبستگی بین نمادها"""
        # ایجاد ماتریس بازدهی روزانه
        daily_returns = self._calculate_daily_returns_by_symbol()
        
        if daily_returns.empty or len(daily_returns.columns) < 2:
            return {'correlation_matrix': {}, 'avg_correlation': 0}
        
        # ماتریس همبستگی
        correlation_matrix = daily_returns.corr()
        
        # میانگین همبستگی (بدون diagonal)
        correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                correlations.append(correlation_matrix.iloc[i, j])
        
        avg_correlation = np.mean(correlations) if correlations else 0
        
        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'avg_correlation': avg_correlation,
            'max_correlation': max(correlations) if correlations else 0,
            'min_correlation': min(correlations) if correlations else 0
        }

    def _diversification_analysis(self) -> Dict:
        """تحلیل تنوع‌سازی"""
        # محاسبه سهم هر نماد از کل PnL
        symbol_contributions = {}
        total_pnl = self.portfolio_df['pnl'].sum()
        
        for symbol in self.symbols:
            symbol_pnl = self.portfolio_df[self.portfolio_df['symbol'] == symbol]['pnl'].sum()
            symbol_contributions[symbol] = symbol_pnl / total_pnl if total_pnl != 0 else 0
        
        # محاسبه Herfindahl Index (measure of concentration)
        contributions_squared = [contribution**2 for contribution in symbol_contributions.values()]
        herfindahl_index = sum(contributions_squared)
        
        # Diversification ratio
        diversification_ratio = 1 - herfindahl_index
        
        # تعداد نمادهای مؤثر
        effective_symbols = 1 / herfindahl_index if herfindahl_index > 0 else 0
        
        return {
            'symbol_contributions': symbol_contributions,
            'herfindahl_index': herfindahl_index,
            'diversification_ratio': diversification_ratio,
            'effective_number_of_symbols': effective_symbols,
            'concentration_risk': 'HIGH' if herfindahl_index > 0.5 else 'MEDIUM' if herfindahl_index > 0.33 else 'LOW'
        }

    def _risk_contribution_analysis(self) -> Dict:
        """تحلیل سهم ریسک هر نماد"""
        risk_contributions = {}
        
        # محاسبه volatility هر نماد
        for symbol in self.symbols:
            symbol_trades = self.portfolio_df[self.portfolio_df['symbol'] == symbol]
            
            if not symbol_trades.empty:
                symbol_volatility = symbol_trades['pnl'].std()
                risk_contributions[symbol] = {
                    'volatility': symbol_volatility,
                    'var_95': np.percentile(symbol_trades['pnl'], 5),
                    'max_loss': symbol_trades['pnl'].min()
                }
        
        # Portfolio level risk
        portfolio_volatility = self.portfolio_df['pnl'].std()
        portfolio_var_95 = np.percentile(self.portfolio_df['pnl'], 5)
        
        return {
            'symbol_risks': risk_contributions,
            'portfolio_volatility': portfolio_volatility,
            'portfolio_var_95': portfolio_var_95,
            'diversification_benefit': self._calculate_diversification_benefit()
        }

    def _time_based_analysis(self) -> Dict:
        """تحلیل بر اساس زمان"""
        if 'entry_time' not in self.portfolio_df.columns:
            return {}
        
        # عملکرد ماهانه
        self.portfolio_df['year_month'] = self.portfolio_df['entry_time'].dt.to_period('M')
        monthly_performance = self.portfolio_df.groupby('year_month')['pnl'].sum()
        
        # عملکرد روز هفته
        self.portfolio_df['day_of_week'] = self.portfolio_df['entry_time'].dt.dayofweek
        daily_performance = self.portfolio_df.groupby('day_of_week')['pnl'].mean()
        
        # عملکرد ساعتی
        self.portfolio_df['hour'] = self.portfolio_df['entry_time'].dt.hour
        hourly_performance = self.portfolio_df.groupby('hour')['pnl'].mean()
        
        return {
            'monthly_performance': monthly_performance.to_dict(),
            'daily_performance': daily_performance.to_dict(),
            'hourly_performance': hourly_performance.to_dict(),
            'best_month': monthly_performance.idxmax() if not monthly_performance.empty else None,
            'worst_month': monthly_performance.idxmin() if not monthly_performance.empty else None
        }

    def _calculate_daily_returns_by_symbol(self) -> pd.DataFrame:
        """محاسبه بازدهی روزانه برای هر نماد"""
        if 'entry_time' not in self.portfolio_df.columns:
            return pd.DataFrame()
        
        # گروه‌بندی بر اساس روز و نماد
        daily_pnl = self.portfolio_df.groupby([
            self.portfolio_df['entry_time'].dt.date, 'symbol'
        ])['pnl'].sum().unstack(fill_value=0)
        
        # محاسبه بازدهی (فرض: مبلغ یکسان برای همه نمادها)
        initial_per_symbol = self.initial_balance / len(self.symbols)
        daily_returns = daily_pnl / initial_per_symbol
        
        return daily_returns

    def _calculate_symbol_sharpe(self, symbol_trades: pd.DataFrame) -> float:
        """محاسبه Sharpe ratio برای یک نماد"""
        if symbol_trades.empty:
            return 0
        
        returns = symbol_trades['pnl']
        avg_return = returns.mean()
        return_std = returns.std()
        
        return avg_return / return_std if return_std > 0 else 0

    def _calculate_diversification_benefit(self) -> float:
        """محاسبه مزیت تنوع‌سازی"""
        # ریسک individual symbols vs portfolio risk
        individual_risks = []
        
        for symbol in self.symbols:
            symbol_trades = self.portfolio_df[self.portfolio_df['symbol'] == symbol]
            if not symbol_trades.empty:
                individual_risks.append(symbol_trades['pnl'].std())
        
        if not individual_risks:
            return 0
        
        avg_individual_risk = np.mean(individual_risks)
        portfolio_risk = self.portfolio_df['pnl'].std()
        
        # diversification benefit = کاهش ریسک نسبت به میانگین individual
        return (avg_individual_risk - portfolio_risk) / avg_individual_risk if avg_individual_risk > 0 else 0

    def generate_portfolio_report(self) -> str:
        """تولید گزارش کامل پورتفولیو"""
        analysis = self.analyze_portfolio_performance()
        
        report = f"""
📊 گزارش تحلیل پورتفولیو چند نمادی
{'='*70}

💰 عملکرد کلی:
   • سود کل: ${analysis.get('total_pnl', 0):,.2f}
   • بازدهی کل: {analysis.get('total_return_pct', 0):.2f}%
   • موجودی نهایی: ${analysis.get('final_balance', 0):,.2f}
   • تعداد کل معاملات: {analysis.get('total_trades', 0)}
   • نرخ موفقیت: {analysis.get('win_rate', 0):.1f}%

📈 عملکرد هر نماد:"""
        
        symbol_analysis = analysis.get('symbol_analysis', {})
        for symbol, stats in symbol_analysis.items():
            report += f"""
   
   {symbol}:
   • سود: ${stats.get('total_pnl', 0):,.2f}
   • معاملات: {stats.get('total_trades', 0)}
   • نرخ موفقیت: {stats.get('win_rate', 0):.1f}%
   • بهترین معامله: ${stats.get('best_trade', 0):,.2f}
   • بدترین معامله: ${stats.get('worst_trade', 0):,.2f}"""
        
        correlation_analysis = analysis.get('correlation_analysis', {})
        report += f"""

🔗 تحلیل همبستگی:
   • میانگین همبستگی: {correlation_analysis.get('avg_correlation', 0):.3f}
   • حداکثر همبستگی: {correlation_analysis.get('max_correlation', 0):.3f}
   • حداقل همبستگی: {correlation_analysis.get('min_correlation', 0):.3f}"""
        
        diversification = analysis.get('diversification_metrics', {})
        report += f"""

🎯 تنوع‌سازی:
   • ضریب تنوع: {diversification.get('diversification_ratio', 0):.3f}
   • تعداد مؤثر نمادها: {diversification.get('effective_number_of_symbols', 0):.1f}
   • ریسک تمرکز: {diversification.get('concentration_risk', 'N/A')}
   • شاخص هرفیندال: {diversification.get('herfindahl_index', 0):.3f}"""
        
        risk_analysis = analysis.get('risk_contribution', {})
        report += f"""

⚠️ تحلیل ریسک:
   • نوسانات پورتفولیو: {risk_analysis.get('portfolio_volatility', 0):.2f}
   • VaR 95%: ${risk_analysis.get('portfolio_var_95', 0):.2f}
   • مزیت تنوع‌سازی: {risk_analysis.get('diversification_benefit', 0):.1%}"""
        
        return report

    def plot_portfolio_analysis(self, save_path: Optional[str] = None):
        """رسم نمودارهای تحلیل پورتفولیو"""
        if self.portfolio_df.empty:
            print("داده کافی برای رسم نمودار وجود ندارد")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('تحلیل جامع پورتفولیو', fontsize=16, fontweight='bold')
        
        # نمودار 1: سهم PnL هر نماد
        symbol_pnl = self.portfolio_df.groupby('symbol')['pnl'].sum()
        axes[0, 0].pie(symbol_pnl.values, labels=symbol_pnl.index, autopct='%1.1f%%')
        axes[0, 0].set_title('سهم سود هر نماد')
        
        # نمودار 2: مقایسه تعداد معاملات
        symbol_counts = self.portfolio_df['symbol'].value_counts()
        axes[0, 1].bar(symbol_counts.index, symbol_counts.values)
        axes[0, 1].set_title('تعداد معاملات هر نماد')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # نمودار 3: Box plot توزیع PnL
        symbol_groups = [group['pnl'].values for name, group in self.portfolio_df.groupby('symbol')]
        axes[0, 2].boxplot(symbol_groups, labels=self.symbols)
        axes[0, 2].set_title('توزیع سود/ضرر هر نماد')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # نمودار 4: همبستگی
        daily_returns = self._calculate_daily_returns_by_symbol()
        if not daily_returns.empty and len(daily_returns.columns) > 1:
            correlation_matrix = daily_returns.corr()
            im = axes[1, 0].imshow(correlation_matrix.values, cmap='coolwarm', vmin=-1, vmax=1)
            axes[1, 0].set_xticks(range(len(correlation_matrix.columns)))
            axes[1, 0].set_yticks(range(len(correlation_matrix.columns)))
            axes[1, 0].set_xticklabels(correlation_matrix.columns, rotation=45)
            axes[1, 0].set_yticklabels(correlation_matrix.columns)
            axes[1, 0].set_title('ماتریس همبستگی')
            plt.colorbar(im, ax=axes[1, 0])
        
        # نمودار 5: عملکرد زمانی
        if 'entry_time' in self.portfolio_df.columns:
            monthly_pnl = self.portfolio_df.groupby(
                self.portfolio_df['entry_time'].dt.to_period('M')
            )['pnl'].sum()
            axes[1, 1].plot(range(len(monthly_pnl)), monthly_pnl.values, marker='o')
            axes[1, 1].set_title('عملکرد ماهانه')
            axes[1, 1].set_ylabel('PnL ($)')
            axes[1, 1].grid(True, alpha=0.3)
        
        # نمودار 6: Risk-Return scatter
        symbol_stats = {}
        for symbol in self.symbols:
            symbol_trades = self.portfolio_df[self.portfolio_df['symbol'] == symbol]
            if not symbol_trades.empty:
                symbol_stats[symbol] = {
                    'return': symbol_trades['pnl'].mean(),
                    'risk': symbol_trades['pnl'].std()
                }
        
        if symbol_stats:
            returns = [stats['return'] for stats in symbol_stats.values()]
            risks = [stats['risk'] for stats in symbol_stats.values()]
            labels = list(symbol_stats.keys())
            
            axes[1, 2].scatter(risks, returns, s=100, alpha=0.7)
            for i, label in enumerate(labels):
                axes[1, 2].annotate(label, (risks[i], returns[i]), 
                                  xytext=(5, 5), textcoords='offset points')
            axes[1, 2].set_xlabel('ریسک (انحراف معیار)')
            axes[1, 2].set_ylabel('بازدهی میانگین')
            axes[1, 2].set_title('نمودار ریسک-بازدهی')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"نمودارها در {save_path} ذخیره شدند")
        
        plt.show()

    def export_analysis_to_json(self, filepath: str):
        """صادرات تحلیل به فایل JSON"""
        analysis = self.analyze_portfolio_performance()
        
        # تبدیل datetime objects به string
        def convert_datetime(obj):
            if hasattr(obj, 'isoformat'):
                return obj.isoformat()
            elif isinstance(obj, pd.Period):
                return str(obj)
            return obj
        
        # پاک‌سازی داده‌ها برای JSON
        clean_analysis = json.loads(json.dumps(analysis, default=convert_datetime))
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(clean_analysis, f, ensure_ascii=False, indent=2)
        
        print(f"تحلیل پورتفولیو در {filepath} ذخیره شد")

    def get_rebalancing_suggestions(self) -> Dict:
        """پیشنهادات تعادل مجدد پورتفولیو"""
        analysis = self.analyze_portfolio_performance()
        suggestions = {}
        
        # بررسی تمرکز بیش از حد
        diversification = analysis.get('diversification_metrics', {})
        if diversification.get('concentration_risk') == 'HIGH':
            suggestions['reduce_concentration'] = True
            suggestions['concentration_details'] = "پورتفولیو روی تعداد کمی نماد متمرکز است"
        
        # بررسی نمادهای ضررده
        symbol_analysis = analysis.get('symbol_analysis', {})
        losing_symbols = []
        winning_symbols = []
        
        for symbol, stats in symbol_analysis.items():
            if stats.get('total_pnl', 0) < 0:
                losing_symbols.append(symbol)
            else:
                winning_symbols.append(symbol)
        
        if losing_symbols:
            suggestions['review_losing_symbols'] = losing_symbols
            suggestions['losing_symbols_advice'] = "بررسی استراتژی برای این نمادها"
        
        # بررسی همبستگی بالا
        correlation_analysis = analysis.get('correlation_analysis', {})
        if correlation_analysis.get('avg_correlation', 0) > 0.7:
            suggestions['high_correlation_warning'] = True
            suggestions['correlation_advice'] = "همبستگی بالا بین نمادها - تنوع کاذب"
        
        return suggestions
