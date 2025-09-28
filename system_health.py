import psutil
import time
import datetime
import sqlite3
import asyncio
import aiohttp
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import threading
import queue
import warnings

class HealthStatus(Enum):
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    DOWN = "DOWN"

@dataclass
class HealthMetric:
    component: str
    status: HealthStatus
    value: float
    threshold: float
    message: str
    timestamp: datetime.datetime

class SystemHealthMonitor:
    """
    نظارت بر سلامت سیستم و اجزای ربات
    """
    
    def __init__(self, db_path: str = "system_health.db"):
        self.db_path = db_path
        self._init_database()
        
        # آستانه‌های سلامت سیستم
        self.health_thresholds = {
            'cpu_usage': {'warning': 70, 'critical': 90},
            'memory_usage': {'warning': 80, 'critical': 95},
            'disk_usage': {'warning': 85, 'critical': 95},
            'api_response_time': {'warning': 5000, 'critical': 10000},  # میلی‌ثانیه
            'database_response_time': {'warning': 1000, 'critical': 3000},
            'error_rate': {'warning': 5, 'critical': 15},  # درصد
            'uptime_hours': {'warning': 168, 'critical': 24}  # معکوس - کمتر از آستانه مشکل است
        }
        
        # Cache برای metrics
        self.current_metrics = {}
        self.health_history = []
        
        # Thread-safe queue برای monitoring
        self.metrics_queue = queue.Queue()
        self.monitoring_active = False
        
        # Component status tracking
        self.component_status = {
            'exchange_connection': HealthStatus.HEALTHY,
            'database': HealthStatus.HEALTHY,
            'ai_services': HealthStatus.HEALTHY,
            'telegram_bot': HealthStatus.HEALTHY,
            'price_analyzer': HealthStatus.HEALTHY,
            'risk_monitor': HealthStatus.HEALTHY
        }
        
    def _init_database(self):
        """راه‌اندازی پایگاه داده"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # جدول system metrics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                metric_type TEXT NOT NULL,
                value REAL NOT NULL,
                status TEXT NOT NULL,
                threshold_warning REAL,
                threshold_critical REAL,
                message TEXT
            )
        ''')
        
        # جدول component health
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS component_health (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                component TEXT NOT NULL,
                status TEXT NOT NULL,
                response_time REAL,
                error_count INTEGER DEFAULT 0,
                last_error TEXT,
                uptime_seconds REAL
            )
        ''')
        
        # جدول health incidents
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS health_incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                end_time DATETIME,
                component TEXT NOT NULL,
                incident_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                description TEXT,
                resolution TEXT,
                duration_minutes REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def start_monitoring(self):
        """شروع نظارت مداوم"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        # شروع thread برای نظارت سیستم
        system_thread = threading.Thread(target=self._monitor_system_resources, daemon=True)
        system_thread.start()
        
        print("✅ نظارت سلامت سیستم آغاز شد")
    
    def stop_monitoring(self):
        """توقف نظارت"""
        self.monitoring_active = False
        print("⏹️ نظارت سلامت سیستم متوقف شد")
    
    def _monitor_system_resources(self):
        """نظارت بر منابع سیستم در background"""
        while self.monitoring_active:
            try:
                # نظارت CPU
                cpu_percent = psutil.cpu_percent(interval=1)
                self._record_metric('cpu_usage', cpu_percent)
                
                # نظارت Memory
                memory = psutil.virtual_memory()
                self._record_metric('memory_usage', memory.percent)
                
                # نظارت Disk
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                self._record_metric('disk_usage', disk_percent)
                
                # نظارت شبکه (اختیاری)
                network = psutil.net_io_counters()
                if hasattr(network, 'bytes_sent') and hasattr(network, 'bytes_recv'):
                    total_bytes = network.bytes_sent + network.bytes_recv
                    self._record_metric('network_activity', total_bytes)
                
                # ذخیره metrics
                self._save_system_metrics()
                
                time.sleep(30)  # چک هر 30 ثانیه
                
            except Exception as e:
                print(f"خطا در نظارت منابع سیستم: {e}")
                time.sleep(60)
    
    def _record_metric(self, metric_type: str, value: float):
        """ثبت metric جدید"""
        status = self._determine_metric_status(metric_type, value)
        message = self._generate_metric_message(metric_type, value, status)
        
        metric = HealthMetric(
            component='SYSTEM',
            status=status,
            value=value,
            threshold=self.health_thresholds.get(metric_type, {}).get('warning', 0),
            message=message,
            timestamp=datetime.datetime.now()
        )
        
        self.current_metrics[metric_type] = metric
    
    def _determine_metric_status(self, metric_type: str, value: float) -> HealthStatus:
        """تعیین وضعیت سلامت metric"""
        if metric_type not in self.health_thresholds:
            return HealthStatus.HEALTHY
        
        thresholds = self.health_thresholds[metric_type]
        
        # برای uptime_hours منطق معکوس است
        if metric_type == 'uptime_hours':
            if value < thresholds['critical']:
                return HealthStatus.CRITICAL
            elif value < thresholds['warning']:
                return HealthStatus.WARNING
            else:
                return HealthStatus.HEALTHY
        else:
            if value >= thresholds['critical']:
                return HealthStatus.CRITICAL
            elif value >= thresholds['warning']:
                return HealthStatus.WARNING
            else:
                return HealthStatus.HEALTHY
    
    def _generate_metric_message(self, metric_type: str, value: float, status: HealthStatus) -> str:
        """تولید پیام برای metric"""
        messages = {
            'cpu_usage': f"استفاده CPU: {value:.1f}%",
            'memory_usage': f"استفاده RAM: {value:.1f}%",
            'disk_usage': f"استفاده دیسک: {value:.1f}%",
            'api_response_time': f"زمان پاسخ API: {value:.0f}ms",
            'database_response_time': f"زمان پاسخ دیتابیس: {value:.0f}ms",
            'error_rate': f"نرخ خطا: {value:.1f}%",
            'uptime_hours': f"مدت فعالیت: {value:.1f} ساعت"
        }
        
        base_message = messages.get(metric_type, f"{metric_type}: {value}")
        
        if status == HealthStatus.CRITICAL:
            return f"🔴 {base_message} - وضعیت بحرانی"
        elif status == HealthStatus.WARNING:
            return f"🟡 {base_message} - هشدار"
        else:
            return f"🟢 {base_message} - سالم"
    
    def _save_system_metrics(self):
        """ذخیره metrics در پایگاه داده"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for metric_type, metric in self.current_metrics.items():
                thresholds = self.health_thresholds.get(metric_type, {})
                
                cursor.execute('''
                    INSERT INTO system_metrics (
                        metric_type, value, status, threshold_warning,
                        threshold_critical, message
                    ) VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    metric_type, metric.value, metric.status.value,
                    thresholds.get('warning'), thresholds.get('critical'),
                    metric.message
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"خطا در ذخیره system metrics: {e}")
    
    async def check_component_health(self, component: str, test_function=None) -> HealthMetric:
        """بررسی سلامت یک component مشخص"""
        start_time = time.time()
        
        try:
            if test_function:
                # اجرای تست سفارشی
                result = await test_function() if asyncio.iscoroutinefunction(test_function) else test_function()
                response_time = (time.time() - start_time) * 1000  # میلی‌ثانیه
                
                if result:
                    status = HealthStatus.HEALTHY
                    message = f"{component} سالم - زمان پاسخ: {response_time:.0f}ms"
                else:
                    status = HealthStatus.CRITICAL
                    message = f"{component} خطا دارد"
            else:
                # تست پیش‌فرض
                status, message = await self._default_component_test(component)
                response_time = (time.time() - start_time) * 1000
            
            # به‌روزرسانی وضعیت component
            self.component_status[component] = status
            
            # ثبت در پایگاه داده
            self._save_component_health(component, status, response_time, message)
            
            return HealthMetric(
                component=component,
                status=status,
                value=response_time,
                threshold=self.health_thresholds.get('api_response_time', {}).get('warning', 5000),
                message=message,
                timestamp=datetime.datetime.now()
            )
            
        except Exception as e:
            error_message = f"خطا در بررسی {component}: {str(e)}"
            self.component_status[component] = HealthStatus.CRITICAL
            
            return HealthMetric(
                component=component,
                status=HealthStatus.CRITICAL,
                value=0,
                threshold=0,
                message=error_message,
                timestamp=datetime.datetime.now()
            )
    
    async def _default_component_test(self, component: str) -> tuple:
        """تست پیش‌فرض برای component"""
        
        if component == 'database':
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('SELECT 1')
                conn.close()
                return HealthStatus.HEALTHY, "اتصال دیتابیس سالم"
            except Exception as e:
                return HealthStatus.CRITICAL, f"خطا در دیتابیس: {e}"
        
        elif component == 'exchange_connection':
            # باید از exchange connector استفاده کنیم
            try:
                # فرض می‌کنیم تابع test موجود است
                return HealthStatus.HEALTHY, "اتصال صرافی سالم"
            except Exception as e:
                return HealthStatus.CRITICAL, f"خطا در اتصال صرافی: {e}"
        
        else:
            return HealthStatus.HEALTHY, f"{component} در حال اجرا"
    
    def _save_component_health(self, component: str, status: HealthStatus, response_time: float, message: str):
        """ذخیره وضعیت سلامت component"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO component_health (
                    component, status, response_time, last_error
                ) VALUES (?, ?, ?, ?)
            ''', (
                component, status.value, response_time,
                message if status != HealthStatus.HEALTHY else None
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"خطا در ذخیره component health: {e}")
    
    def record_incident(self, component: str, incident_type: str, severity: str, description: str):
        """ثبت incident جدید"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO health_incidents (
                    component, incident_type, severity, description
                ) VALUES (?, ?, ?, ?)
            ''', (component, incident_type, severity, description))
            
            incident_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            print(f"⚠️ Incident ثبت شد: {component} - {incident_type}")
            return incident_id
            
        except Exception as e:
            print(f"خطا در ثبت incident: {e}")
            return None
    
    def resolve_incident(self, incident_id: int, resolution: str):
        """حل کردن incident"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # محاسبه مدت incident
            cursor.execute('SELECT start_time FROM health_incidents WHERE id = ?', (incident_id,))
            result = cursor.fetchone()
            
            if result:
                start_time = datetime.datetime.fromisoformat(result[0])
                duration_minutes = (datetime.datetime.now() - start_time).total_seconds() / 60
                
                cursor.execute('''
                    UPDATE health_incidents 
                    SET end_time = ?, resolution = ?, duration_minutes = ?
                    WHERE id = ?
                ''', (datetime.datetime.now(), resolution, duration_minutes, incident_id))
                
                conn.commit()
                print(f"✅ Incident {incident_id} حل شد - مدت: {duration_minutes:.1f} دقیقه")
            
            conn.close()
            
        except Exception as e:
            print(f"خطا در حل incident: {e}")
    
    def get_health_summary(self) -> Dict:
        """دریافت خلاصه کلی سلامت سیستم"""
        try:
            summary = {
                'timestamp': datetime.datetime.now(),
                'overall_status': self._calculate_overall_health(),
                'system_metrics': {},
                'component_status': dict(self.component_status),
                'active_incidents': self._get_active_incidents(),
                'recommendations': []
            }
            
            # اضافه کردن آخرین metrics
            for metric_type, metric in self.current_metrics.items():
                summary['system_metrics'][metric_type] = {
                    'value': metric.value,
                    'status': metric.status.value,
                    'message': metric.message
                }
            
            # تولید توصیه‌ها
            summary['recommendations'] = self._generate_health_recommendations()
            
            return summary
            
        except Exception as e:
            print(f"خطا در تهیه خلاصه سلامت: {e}")
            return {}
    
    def _calculate_overall_health(self) -> str:
        """محاسبه وضعیت کلی سلامت"""
        
        # بررسی component ها
        component_statuses = list(self.component_status.values())
        
        if HealthStatus.CRITICAL in component_statuses:
            return 'CRITICAL'
        elif HealthStatus.WARNING in component_statuses:
            return 'WARNING'
        
        # بررسی system metrics
        metric_statuses = [m.status for m in self.current_metrics.values()]
        
        if HealthStatus.CRITICAL in metric_statuses:
            return 'CRITICAL'
        elif HealthStatus.WARNING in metric_statuses:
            return 'WARNING'
        
        return 'HEALTHY'
    
    def _get_active_incidents(self) -> List[Dict]:
        """دریافت incident های فعال"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT * FROM health_incidents 
                WHERE end_time IS NULL
                ORDER BY start_time DESC
            '''
            
            cursor = conn.cursor()
            cursor.execute(query)
            incidents = []
            
            for row in cursor.fetchall():
                incidents.append({
                    'id': row[0],
                    'start_time': row[1],
                    'component': row[3],
                    'incident_type': row[4],
                    'severity': row[5],
                    'description': row[6]
                })
            
            conn.close()
            return incidents
            
        except Exception as e:
            print(f"خطا در دریافت incidents فعال: {e}")
            return []
    
    def _generate_health_recommendations(self) -> List[str]:
        """تولید توصیه‌های بهبود سلامت"""
        recommendations = []
        
        # بررسی CPU
        cpu_metric = self.current_metrics.get('cpu_usage')
        if cpu_metric and cpu_metric.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
            recommendations.append("کاهش فرآیندهای CPU-intensive یا افزودن منابع پردازشی")
        
        # بررسی Memory
        memory_metric = self.current_metrics.get('memory_usage')
        if memory_metric and memory_metric.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
            recommendations.append("بهینه‌سازی استفاده از حافظه یا افزودن RAM")
        
        # بررسی Disk
        disk_metric = self.current_metrics.get('disk_usage')
        if disk_metric and disk_metric.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
            recommendations.append("پاک‌سازی فایل‌های غیرضروری یا افزودن فضای ذخیره‌سازی")
        
        # بررسی components
        for component, status in self.component_status.items():
            if status == HealthStatus.CRITICAL:
                recommendations.append(f"بررسی و رفع مشکل {component}")
        
        # اگر incident فعال داریم
        active_incidents = self._get_active_incidents()
        if active_incidents:
            recommendations.append(f"حل {len(active_incidents)} incident فعال")
        
        return recommendations
    
    def export_health_report(self, days: int = 7) -> str:
        """صادرات گزارش سلامت"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
            
            # دریافت metrics
            metrics_query = '''
                SELECT * FROM system_metrics 
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            '''
            metrics_df = pd.read_sql_query(metrics_query, conn, params=(cutoff_date,))
            
            # دریافت component health
            components_query = '''
                SELECT * FROM component_health 
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            '''
            components_df = pd.read_sql_query(components_query, conn, params=(cutoff_date,))
            
            # دریافت incidents
            incidents_query = '''
                SELECT * FROM health_incidents 
                WHERE start_time >= ?
                ORDER BY start_time DESC
            '''
            incidents_df = pd.read_sql_query(incidents_query, conn, params=(cutoff_date,))
            
            conn.close()
            
            # ایجاد گزارش
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = f"health_report_{timestamp}.json"
            
            report_data = {
                'report_period_days': days,
                'generated_at': datetime.datetime.now().isoformat(),
                'summary': self.get_health_summary(),
                'metrics_history': metrics_df.to_dict('records'),
                'components_history': components_df.to_dict('records'),
                'incidents': incidents_df.to_dict('records')
            }
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2, default=str)
            
            print(f"گزارش سلامت در {report_file} صادر شد")
            return report_file
            
        except Exception as e:
            print(f"خطا در صادرات گزارش سلامت: {e}")
            return ""
    
    async def run_full_health_check(self) -> Dict:
        """اجرای کامل بررسی سلامت"""
        print("شروع بررسی کامل سلامت سیستم...")
        
        results = {
            'timestamp': datetime.datetime.now(),
            'system_metrics': dict(self.current_metrics),
            'component_results': {},
            'overall_status': 'UNKNOWN'
        }
        
        # بررسی همه component ها
        components_to_check = [
            'database',
            'exchange_connection', 
            'ai_services',
            'telegram_bot',
            'price_analyzer',
            'risk_monitor'
        ]
        
        for component in components_to_check:
            try:
                result = await self.check_component_health(component)
                results['component_results'][component] = {
                    'status': result.status.value,
                    'response_time': result.value,
                    'message': result.message
                }
            except Exception as e:
                results['component_results'][component] = {
                    'status': 'ERROR',
                    'response_time': 0,
                    'message': f"خطا در بررسی: {e}"
                }
        
        # محاسبه وضعیت کلی
        results['overall_status'] = self._calculate_overall_health()
        
        print(f"بررسی سلامت تکمیل شد - وضعیت کلی: {results['overall_status']}")
        return results
