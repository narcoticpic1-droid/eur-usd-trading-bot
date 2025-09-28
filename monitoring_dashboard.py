import pandas as pd
import json
import datetime
from typing import Dict, List, Optional
import sqlite3
from .performance_tracker import PerformanceTracker
from .risk_monitor import RiskMonitor
from .system_health import SystemHealthMonitor

class MonitoringDashboard:
    """
    داشبورد ساده برای نمایش داده‌های نظارت
    """
    
    def __init__(self, 
                 performance_tracker: PerformanceTracker,
                 risk_monitor: RiskMonitor,
                 health_monitor: SystemHealthMonitor):
        
        self.performance_tracker = performance_tracker
        self.risk_monitor = risk_monitor  
        self.health_monitor = health_monitor
    
    def generate_html_report(self, output_file: str = None) -> str:
        """تولید گزارش HTML"""
        
        if output_file is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"monitoring_report_{timestamp}.html"
        
        # دریافت داده‌ها
        performance_summary = self.performance_tracker.get_performance_summary()
        risk_summary = self.risk_monitor.get_risk_summary()
        health_summary = self.health_monitor.get_health_summary()
        
        # تولید HTML
        html_content = self._create_html_template(
            performance_summary, 
            risk_summary, 
            health_summary
        )
        
        # ذخیره فایل
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"گزارش HTML در {output_file} ایجاد شد")
        return output_file
    
    def _create_html_template(self, performance_data: dict, risk_data: dict, health_data: dict) -> str:
        """ایجاد قالب HTML"""
        
        html = f"""
        <!DOCTYPE html>
        <html dir="rtl" lang="fa">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>گزارش نظارت سیستم معاملاتی</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }}
                .header {{ text-align: center; border-bottom: 3px solid #007bff; padding-bottom: 20px; margin-bottom: 30px; }}
                .section {{ margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }}
                .status-good {{ color: #28a745; font-weight: bold; }}
                .status-warning {{ color: #ffc107; font-weight: bold; }}
                .status-critical {{ color: #dc3545; font-weight: bold; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #f8f9fa; border-radius: 5px; min-width: 200px; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
                th, td {{ padding: 8px; text-align: right; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #007bff; color: white; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🤖 گزارش نظارت سیستم معاملاتی</h1>
                    <p>تاریخ تولید: {datetime.datetime.now().strftime('%Y/%m/%d - %H:%M:%S')}</p>
                </div>
                
                {self._create_performance_section(performance_data)}
                {self._create_risk_section(risk_data)}
                {self._create_health_section(health_data)}
                
                <div class="section">
                    <h2>📋 خلاصه و توصیه‌ها</h2>
                    {self._create_recommendations_section(performance_data, risk_data, health_data)}
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _create_performance_section(self, data: dict) -> str:
        """بخش عملکرد"""
        
        portfolio_metrics = data.get('portfolio_metrics', {})
        
        return f"""
        <div class="section">
            <h2>📊 عملکرد پورتفولیو</h2>
            
            <div class="metric">
                <strong>بازده کل:</strong><br>
                {portfolio_metrics.get('total_return', 0):.2f}%
            </div>
            
            <div class="metric">
                <strong>نرخ برد متوسط:</strong><br>
                {portfolio_metrics.get('avg_win_rate', 0):.1f}%
            </div>
            
            <div class="metric">
                <strong>نسبت شارپ:</strong><br>
                {portfolio_metrics.get('avg_sharpe_ratio', 0):.2f}
            </div>
            
            <div class="metric">
                <strong>کل معاملات:</strong><br>
                {portfolio_metrics.get('total_trades', 0)}
            </div>
            
            <div class="metric">
                <strong>بدترین افت:</strong><br>
                {portfolio_metrics.get('worst_drawdown', 0):.2f}%
            </div>
            
            <h3>عملکرد هر نماد:</h3>
            <table>
                <tr>
                    <th>نماد</th>
                    <th>بازده (%)</th>
                    <th>نرخ برد (%)</th>
                    <th>تعداد معاملات</th>
                    <th>حداکثر افت (%)</th>
                </tr>
        """
        
        # اضافه کردن داده‌های هر نماد
        for symbol, metrics in data.get('symbols', {}).items():
            html += f"""
                <tr>
                    <td>{symbol}</td>
                    <td>{metrics.get('total_return', 0):.2f}</td>
                    <td>{metrics.get('win_rate', 0):.1f}</td>
                    <td>{metrics.get('total_trades', 0)}</td>
                    <td>{metrics.get('max_drawdown', 0):.2f}</td>
                </tr>
            """
        
        html += """
            </table>
        </div>
        """
        
        return html
    
    def _create_risk_section(self, data: dict) -> str:
        """بخش ریسک"""
        
        overall_risk = data.get('overall_risk_level', 'UNKNOWN')
        status_class = self._get_status_class(overall_risk)
        
        html = f"""
        <div class="section">
            <h2>⚠️ وضعیت ریسک</h2>
            
            <div class="metric">
                <strong>سطح کلی ریسک:</strong><br>
                <span class="{status_class}">{overall_risk}</span>
            </div>
        """
        
        # نمایش alerts فعال
        active_alerts = data.get('active_alerts', [])
        if active_alerts:
            html += """
            <h3>هشدارهای فعال:</h3>
            <table>
                <tr>
                    <th>سطح</th>
                    <th>نوع</th>
                    <th>تعداد</th>
                </tr>
            """
            
            for alert in active_alerts:
                level = alert.get('level', 'UNKNOWN')
                status_class = self._get_status_class(level)
                html += f"""
                    <tr>
                        <td><span class="{status_class}">{level}</span></td>
                        <td>{alert.get('risk_type', 'نامشخص')}</td>
                        <td>{alert.get('count', 0)}</td>
                    </tr>
                """
            
            html += "</table>"
        
        html += "</div>"
        return html
    
    def _create_health_section(self, data: dict) -> str:
        """بخش سلامت سیستم"""
        
        overall_status = data.get('overall_status', 'UNKNOWN')
        status_class = self._get_status_class(overall_status)
        
        html = f"""
        <div class="section">
            <h2>🔍 سلامت سیستم</h2>
            
            <div class="metric">
                <strong>وضعیت کلی:</strong><br>
                <span class="{status_class}">{overall_status}</span>
            </div>
            
            <h3>وضعیت اجزا:</h3>
            <table>
                <tr>
                    <th>جزء</th>
                    <th>وضعیت</th>
                </tr>
        """
        
        # نمایش وضعیت components
        for component, status in data.get('component_status', {}).items():
            status_class = self._get_status_class(status)
            html += f"""
                <tr>
                    <td>{component}</td>
                    <td><span class="{status_class}">{status}</span></td>
                </tr>
            """
        
        html += """
            </table>
            
            <h3>معیارهای سیستم:</h3>
            <table>
                <tr>
                    <th>معیار</th>
                    <th>مقدار</th>
                    <th>وضعیت</th>
                </tr>
        """
        
        # نمایش system metrics
        for metric_name, metric_data in data.get('system_metrics', {}).items():
            value = metric_data.get('value', 0)
            status = metric_data.get('status', 'UNKNOWN')
            status_class = self._get_status_class(status)
            
            html += f"""
                <tr>
                    <td>{metric_name}</td>
                    <td>{value:.1f}</td>
                    <td><span class="{status_class}">{status}</span></td>
                </tr>
            """
        
        html += """
            </table>
        </div>
        """
        
        return html
    
    def _create_recommendations_section(self, performance_data: dict, risk_data: dict, health_data: dict) -> str:
        """بخش توصیه‌ها"""
        
        recommendations = []
        
        # توصیه‌های عملکرد
        portfolio_metrics = performance_data.get('portfolio_metrics', {})
        if portfolio_metrics.get('avg_win_rate', 0) < 50:
            recommendations.append("🔄 بررسی و بهبود استراتژی معاملاتی")
        
        if portfolio_metrics.get('avg_sharpe_ratio', 0) < 1:
            recommendations.append("⚖️ بهینه‌سازی نسبت ریسک به بازده")
        
        # توصیه‌های ریسک
        if risk_data.get('overall_risk_level') in ['HIGH', 'CRITICAL']:
            recommendations.append("🚨 کاهش فوری ریسک پورتفولیو")
        
        # توصیه‌های سلامت
        health_recommendations = health_data.get('recommendations', [])
        recommendations.extend([f"🔧 {rec}" for rec in health_recommendations])
        
        if not recommendations:
            recommendations.append("✅ سیستم در وضعیت مطلوب قرار دارد")
        
        html = "<ul>"
        for rec in recommendations:
            html += f"<li>{rec}</li>"
        html += "</ul>"
        
        return html
    
    def _get_status_class(self, status: str) -> str:
        """تعیین کلاس CSS برای وضعیت"""
        status = status.upper()
        
        if status in ['HEALTHY', 'LOW', 'GOOD', 'EXCELLENT']:
            return 'status-good'
        elif status in ['WARNING', 'MEDIUM', 'FAIR']:
            return 'status-warning'
        elif status in ['CRITICAL', 'HIGH', 'POOR_PERFORMANCE', 'DOWN']:
            return 'status-critical'
        else:
            return ''
    
    def export_json_data(self, output_file: str = None) -> str:
        """صادرات داده‌ها به فرمت JSON"""
        
        if output_file is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"monitoring_data_{timestamp}.json"
        
        # جمع‌آوری داده‌ها
        data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'performance': self.performance_tracker.get_performance_summary(),
            'risk': self.risk_monitor.get_risk_summary(),
            'health': self.health_monitor.get_health_summary()
        }
        
        # ذخیره فایل
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"داده‌های نظارت در {output_file} صادر شد")
        return output_file
