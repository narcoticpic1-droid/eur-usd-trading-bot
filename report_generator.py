import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import io
import base64
from pathlib import Path
import json
from jinja2 import Template

class ReportGenerator:
    """
    تولیدکننده گزارش‌های جامع برای نتایج backtesting
    """
    
    def __init__(self):
        self.report_templates = {
            'executive_summary': self._executive_summary_template(),
            'detailed_analysis': self._detailed_analysis_template(),
            'risk_analysis': self._risk_analysis_template(),
            'trade_analysis': self._trade_analysis_template()
        }
        
    def generate_comprehensive_report(self, 
                                    backtest_results: Dict,
                                    performance_metrics: Dict,
                                    portfolio_analysis: Dict,
                                    trade_simulations: List[Dict],
                                    output_path: str = "reports/") -> Dict:
        """تولید گزارش جامع"""
        
        # ایجاد پوشه گزارش
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        # تهیه داده‌های گزارش
        report_data = self._prepare_report_data(
            backtest_results, performance_metrics, 
            portfolio_analysis, trade_simulations
        )
        
        # تولید بخش‌های مختلف گزارش
        reports = {}
        
        # خلاصه اجرایی
        reports['executive_summary'] = self._generate_executive_summary(report_data)
        
        # تحلیل جامع
        reports['detailed_analysis'] = self._generate_detailed_analysis(report_data)
        
        # تحلیل ریسک
        reports['risk_analysis'] = self._generate_risk_analysis(report_data)
        
        # تحلیل معاملات
        reports['trade_analysis'] = self._generate_trade_analysis(report_data)
        
        # تولید نمودارها
        charts = self._generate_all_charts(report_data, f"{output_path}/charts/")
        reports['charts'] = charts
        
        # ترکیب در گزارش نهایی
        final_report = self._combine_reports(reports, report_data)
        
        # ذخیره فایل‌ها
        output_files = self._save_reports(final_report, output_path)
        
        return {
            'report_data': report_data,
            'generated_reports': reports,
            'output_files': output_files,
            'summary': self._generate_report_summary(report_data)
        }

    def _prepare_report_data(self, backtest_results: Dict, performance_metrics: Dict,
                           portfolio_analysis: Dict, trade_simulations: List[Dict]) -> Dict:
        """آماده‌سازی داده‌های گزارش"""
        
        return {
            'metadata': {
                'generation_date': datetime.now(),
                'report_period': self._extract_period(backtest_results),
                'symbols_analyzed': self._extract_symbols(backtest_results),
                'total_trades': len(trade_simulations),
                'backtest_duration_days': self._calculate_duration(backtest_results)
            },
            'performance': {
                'total_return': performance_metrics.get('total_return', 0),
                'annualized_return': performance_metrics.get('annualized_return', 0),
                'sharpe_ratio': performance_metrics.get('sharpe_ratio', 0),
                'max_drawdown': performance_metrics.get('max_drawdown', 0),
                'win_rate': performance_metrics.get('win_rate', 0),
                'profit_factor': performance_metrics.get('profit_factor', 0),
                'calmar_ratio': performance_metrics.get('calmar_ratio', 0)
            },
            'portfolio': portfolio_analysis,
            'trades': trade_simulations,
            'risk_metrics': {
                'var_95': performance_metrics.get('var_95', 0),
                'cvar_95': performance_metrics.get('cvar_95', 0),
                'volatility': performance_metrics.get('volatility', 0),
                'downside_deviation': performance_metrics.get('downside_deviation', 0)
            },
            'backtest_config': backtest_results.get('config', {}),
            'execution_quality': self._analyze_execution_quality(trade_simulations)
        }

    def _generate_executive_summary(self, report_data: Dict) -> str:
        """تولید خلاصه اجرایی"""
        
        metadata = report_data['metadata']
        performance = report_data['performance']
        risk = report_data['risk_metrics']
        
        # تعیین وضعیت کلی
        overall_score = self._calculate_overall_score(performance, risk)
        status = self._determine_status(overall_score)
        
        # Key highlights
        highlights = self._extract_key_highlights(report_data)
        
        template = f"""
# خلاصه اجرایی - گزارش Backtesting

## اطلاعات کلی
- **تاریخ تولید گزارش:** {metadata['generation_date'].strftime('%Y-%m-%d %H:%M')}
- **دوره تحلیل:** {metadata['report_period']}
- **نمادهای تحلیل شده:** {', '.join(metadata['symbols_analyzed'])}
- **مدت backtest:** {metadata['backtest_duration_days']} روز
- **تعداد کل معاملات:** {metadata['total_trades']}

## عملکرد کلی ({status})
- **بازدهی کل:** {performance['total_return']:.2f}%
- **بازدهی سالانه:** {performance['annualized_return']:.2f}%
- **نسبت شارپ:** {performance['sharpe_ratio']:.3f}
- **حداکثر افت:** {performance['max_drawdown']:.2f}%
- **نرخ موفقیت:** {performance['win_rate']:.1f}%

## نکات کلیدی
{chr(10).join(f"• {highlight}" for highlight in highlights)}

## توصیه کلی
{self._generate_recommendation(overall_score, performance, risk)}
"""
        return template

    def _generate_detailed_analysis(self, report_data: Dict) -> str:
        """تولید تحلیل جامع"""
        
        performance = report_data['performance']
        portfolio = report_data['portfolio']
        trades_data = report_data['trades']
        
        # تحلیل معاملات
        trade_analysis = self._detailed_trade_analysis(trades_data)
        
        # تحلیل نمادها
        symbol_analysis = self._detailed_symbol_analysis(portfolio)
        
        # تحلیل زمانی
        time_analysis = self._detailed_time_analysis(trades_data)
        
        template = f"""
# تحلیل جامع عملکرد

## 1. معیارهای عملکرد تفصیلی

### بازدهی و سودآوری
- **بازدهی کل:** {performance['total_return']:.2f}%
- **بازدهی سالانه:** {performance['annualized_return']:.2f}%
- **ضریب سود:** {performance['profit_factor']:.2f}
- **نسبت کالمار:** {performance['calmar_ratio']:.3f}

### کیفیت معاملات
{trade_analysis}

## 2. تحلیل هر نماد
{symbol_analysis}

## 3. تحلیل زمانی
{time_analysis}

## 4. مقایسه با benchmark
{self._benchmark_comparison(report_data)}
"""
        return template

    def _generate_risk_analysis(self, report_data: Dict) -> str:
        """تولید تحلیل ریسک"""
        
        risk = report_data['risk_metrics']
        performance = report_data['performance']
        
        # تحلیل drawdown
        drawdown_analysis = self._analyze_drawdowns(report_data['trades'])
        
        # تحلیل volatility
        volatility_analysis = self._analyze_volatility_patterns(report_data['trades'])
        
        # تحلیل tail risk
        tail_risk = self._analyze_tail_risk(report_data['trades'])
        
        template = f"""
# تحلیل جامع ریسک

## 1. معیارهای ریسک اصلی

### ریسک نوسانات
- **نوسانات کل:** {risk['volatility']:.2f}%
- **انحراف منفی:** {risk['downside_deviation']:.2f}%
- **حداکثر افت:** {performance['max_drawdown']:.2f}%

### ریسک دنباله (Tail Risk)
- **VaR 95%:** {risk['var_95']:.2f}%
- **CVaR 95%:** {risk['cvar_95']:.2f}%

## 2. تحلیل Drawdown
{drawdown_analysis}

## 3. الگوهای نوسانات
{volatility_analysis}

## 4. تحلیل ریسک دنباله
{tail_risk}

## 5. سنجش ریسک تطبیقی
{self._adaptive_risk_assessment(report_data)}
"""
        return template

    def _generate_trade_analysis(self, report_data: Dict) -> str:
        """تولید تحلیل معاملات"""
        
        trades = report_data['trades']
        execution_quality = report_data['execution_quality']
        
        # تحلیل الگوهای ورود و خروج
        entry_exit_analysis = self._analyze_entry_exit_patterns(trades)
        
        # تحلیل کیفیت اجرا
        execution_analysis = self._analyze_execution_metrics(execution_quality)
        
        # تحلیل holding periods
        holding_analysis = self._analyze_holding_periods(trades)
        
        template = f"""
# تحلیل جامع معاملات

## 1. آمار کلی معاملات
- **تعداد کل معاملات:** {len(trades)}
- **معاملات سودآور:** {len([t for t in trades if t.get('pnl', 0) > 0])}
- **معاملات ضرردهنده:** {len([t for t in trades if t.get('pnl', 0) < 0])}

## 2. تحلیل الگوهای ورود/خروج
{entry_exit_analysis}

## 3. کیفیت اجرای معاملات
{execution_analysis}

## 4. تحلیل مدت نگهداری
{holding_analysis}

## 5. تحلیل serial correlation
{self._analyze_serial_correlation(trades)}
"""
        return template

    def _generate_all_charts(self, report_data: Dict, output_path: str) -> Dict:
        """تولید تمام نمودارها"""
        
        Path(output_path).mkdir(parents=True, exist_ok=True)
        charts = {}
        
        # نمودار equity curve
        charts['equity_curve'] = self._plot_equity_curve(report_data, f"{output_path}/equity_curve.png")
        
        # نمودار drawdown
        charts['drawdown'] = self._plot_drawdown_chart(report_data, f"{output_path}/drawdown.png")
        
        # نمودار توزیع returns
        charts['returns_distribution'] = self._plot_returns_distribution(report_data, f"{output_path}/returns_dist.png")
        
        # نمودار monthly performance
        charts['monthly_performance'] = self._plot_monthly_performance(report_data, f"{output_path}/monthly_perf.png")
        
        # نمودار risk metrics
        charts['risk_metrics'] = self._plot_risk_metrics(report_data, f"{output_path}/risk_metrics.png")
        
        # نمودار trade analysis
        charts['trade_analysis'] = self._plot_trade_analysis(report_data, f"{output_path}/trade_analysis.png")
        
        return charts

    def _plot_equity_curve(self, report_data: Dict, save_path: str) -> str:
        """رسم منحنی equity"""
        
        trades = report_data['trades']
        if not trades:
            return ""
        
        # محاسبه cumulative returns
        cumulative_pnl = np.cumsum([t.get('pnl', 0) for t in trades])
        dates = [t.get('exit_time', datetime.now()) for t in trades]
        
        plt.figure(figsize=(12, 6))
        plt.plot(dates, cumulative_pnl, linewidth=2, label='Strategy')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.title('منحنی رشد سرمایه', fontsize=14, fontweight='bold')
        plt.xlabel('تاریخ')
        plt.ylabel('سود تجمعی ($)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path

    def _plot_drawdown_chart(self, report_data: Dict, save_path: str) -> str:
        """رسم نمودار drawdown"""
        
        trades = report_data['trades']
        if not trades:
            return ""
        
        # محاسبه drawdown
        cumulative_pnl = np.cumsum([t.get('pnl', 0) for t in trades])
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = (cumulative_pnl - running_max) / running_max * 100
        
        dates = [t.get('exit_time', datetime.now()) for t in trades]
        
        plt.figure(figsize=(12, 6))
        plt.fill_between(dates, drawdown, 0, alpha=0.3, color='red', label='Drawdown')
        plt.plot(dates, drawdown, color='red', linewidth=1)
        plt.title('نمودار Drawdown', fontsize=14, fontweight='bold')
        plt.xlabel('تاریخ')
        plt.ylabel('Drawdown (%)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path

    def _calculate_overall_score(self, performance: Dict, risk: Dict) -> float:
        """محاسبه امتیاز کلی"""
        
        # وزن‌دهی معیارها
        weights = {
            'return': 0.3,
            'sharpe': 0.25,
            'max_dd': 0.2,
            'win_rate': 0.15,
            'profit_factor': 0.1
        }
        
        # نرمال‌سازی معیارها (0-100)
        normalized_scores = {
            'return': min(100, max(0, performance.get('annualized_return', 0) * 2)),  # 50% = score 100
            'sharpe': min(100, max(0, performance.get('sharpe_ratio', 0) * 40)),      # 2.5 = score 100
            'max_dd': min(100, max(0, 100 - abs(performance.get('max_drawdown', 0)) * 2)),  # 50% dd = score 0
            'win_rate': performance.get('win_rate', 0),
            'profit_factor': min(100, max(0, performance.get('profit_factor', 0) * 25))     # 4.0 = score 100
        }
        
        # محاسبه امتیاز وزنی
        overall_score = sum(weights[key] * normalized_scores[key] for key in weights.keys())
        
        return overall_score

    def _determine_status(self, score: float) -> str:
        """تعیین وضعیت بر اساس امتیاز"""
        if score >= 80:
            return "عالی 🟢"
        elif score >= 60:
            return "خوب 🟡"
        elif score >= 40:
            return "متوسط 🟠"
        else:
            return "ضعیف 🔴"

    def _extract_key_highlights(self, report_data: Dict) -> List[str]:
        """استخراج نکات کلیدی"""
        
        highlights = []
        performance = report_data['performance']
        risk = report_data['risk_metrics']
        
        # بررسی نکات مثبت
        if performance.get('sharpe_ratio', 0) > 1.5:
            highlights.append(f"نسبت شارپ عالی: {performance['sharpe_ratio']:.2f}")
        
        if performance.get('win_rate', 0) > 70:
            highlights.append(f"نرخ موفقیت بالا: {performance['win_rate']:.1f}%")
        
        if abs(performance.get('max_drawdown', 0)) < 10:
            highlights.append(f"Drawdown پایین: {performance['max_drawdown']:.1f}%")
        
        # بررسی نکات منفی
        if performance.get('sharpe_ratio', 0) < 0.5:
            highlights.append(f"⚠️ نسبت شارپ پایین: {performance['sharpe_ratio']:.2f}")
        
        if abs(performance.get('max_drawdown', 0)) > 25:
            highlights.append(f"⚠️ Drawdown بالا: {performance['max_drawdown']:.1f}%")
        
        if performance.get('win_rate', 0) < 40:
            highlights.append(f"⚠️ نرخ موفقیت پایین: {performance['win_rate']:.1f}%")
        
        return highlights[:5]  # حداکثر 5 نکته

    def _generate_recommendation(self, score: float, performance: Dict, risk: Dict) -> str:
        """تولید توصیه کلی"""
        
        if score >= 80:
            return "استراتژی عملکرد بسیار خوبی دارد و آماده اجرای live است. توصیه می‌شود با capital کم شروع کنید."
        
        elif score >= 60:
            return "استراتژی نتایج قابل قبولی دارد اما نیاز به بهینه‌سازی بیشتر است. risk management را تقویت کنید."
        
        elif score >= 40:
            return "استراتژی نیاز به بهبود جدی دارد. پارامترها را بازبینی کنید و risk را کاهش دهید."
        
        else:
            return "استراتژی در وضعیت فعلی قابل اجرا نیست. بازطراحی کامل توصیه می‌شود."

    def export_to_excel(self, report_data: Dict, output_path: str):
        """صادرات گزارش به Excel"""
        
        filename = f"{output_path}/backtest_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            
            # صفحه خلاصه
            summary_df = pd.DataFrame([{
                'metric': key,
                'value': value
            } for key, value in report_data['performance'].items()])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # صفحه معاملات
            if report_data['trades']:
                trades_df = pd.DataFrame(report_data['trades'])
                trades_df.to_excel(writer, sheet_name='Trades', index=False)
            
            # صفحه آمار ریسک
            risk_df = pd.DataFrame([{
                'metric': key,
                'value': value
            } for key, value in report_data['risk_metrics'].items()])
            risk_df.to_excel(writer, sheet_name='Risk_Metrics', index=False)
        
        print(f"گزارش Excel در {filename} ذخیره شد")
        return filename

    def _save_reports(self, final_report: str, output_path: str) -> Dict:
        """ذخیره گزارش‌ها"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        
        # ذخیره HTML
        html_file = f"{output_path}/backtest_report_{timestamp}.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(final_report)
        
        # ذخیره Markdown
        md_file = f"{output_path}/backtest_report_{timestamp}.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(final_report)
        
        return {
            'html_report': html_file,
            'markdown_report': md_file
        }

    def _combine_reports(self, reports: Dict, report_data: Dict) -> str:
        """ترکیب تمام بخش‌های گزارش"""
        
        combined = f"""
<!DOCTYPE html>
<html>
<head>
    <title>گزارش جامع Backtesting</title>
    <meta charset="utf-8">
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; }}
        h1 {{ color: #2E86AB; border-bottom: 3px solid #2E86AB; }}
        h2 {{ color: #A23B72; }}
        h3 {{ color: #F18F01; }}
        .metric {{ background: #f8f9fa; padding: 10px; border-left: 4px solid #28a745; margin: 10px 0; }}
        .warning {{ background: #fff3cd; border-left: 4px solid #ffc107; }}
        .danger {{ background: #f8d7da; border-left: 4px solid #dc3545; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>

{reports['executive_summary']}

<hr style="margin: 40px 0;">

{reports['detailed_analysis']}

<hr style="margin: 40px 0;">

{reports['risk_analysis']}

<hr style="margin: 40px 0;">

{reports['trade_analysis']}

<hr style="margin: 40px 0;">

<h1>نمودارها و تصاویر</h1>
<p>نمودارها در پوشه charts/ ذخیره شده‌اند:</p>
<ul>
{chr(10).join(f"<li>{chart_name}: {chart_path}</li>" for chart_name, chart_path in reports.get('charts', {}).items())}
</ul>

<hr style="margin: 40px 0;">

<div style="text-align: center; color: #666; margin-top: 50px;">
    <p>گزارش تولید شده در {report_data['metadata']['generation_date'].strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p>سیستم Multi-Crypto Pure Price Action Trading Bot</p>
</div>

</body>
</html>
"""
        return combined
