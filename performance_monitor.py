import time
import psutil
import threading
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
import statistics
from dataclasses import dataclass
from collections import deque
import gc

@dataclass
class PerformanceMetric:
    """کلاس نگهداری معیارهای عملکرد"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    category: str = "general"

class PerformanceMonitor:
    """مانیتور عملکرد سیستم"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # تنظیمات
        self.monitoring_interval = self.config.get('monitoring_interval', 5.0)  # ثانیه
        self.max_history_size = self.config.get('max_history_size', 1000)
        self.alert_thresholds = self.config.get('alert_thresholds', {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'response_time_ms': 5000.0
        })
        
        # ذخیره‌سازی داده‌ها
        self.metrics_history = deque(maxlen=self.max_history_size)
        self.function_timings = {}
        self.api_response_times = deque(maxlen=100)
        self.error_counts = {}
        
        # وضعیت مانیتورینگ
        self.is_monitoring = False
        self.monitor_thread = None
        self.start_time = datetime.now()
        
        # کالبک‌ها
        self.alert_callbacks = []
        
        # شمارنده‌های عملکرد
        self.counters = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'signals_generated': 0,
            'ai_calls': 0
        }
    
    def start_monitoring(self):
        """شروع مانیتورینگ"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("Performance monitoring شروع شد")
    
    def stop_monitoring(self):
        """توقف مانیتورینگ"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        print("Performance monitoring متوقف شد")
    
    def _monitor_loop(self):
        """حلقه اصلی مانیتورینگ"""
        while self.is_monitoring:
            try:
                # جمع‌آوری معیارهای سیستم
                self._collect_system_metrics()
                
                # بررسی آلارم‌ها
                self._check_alerts()
                
                # پاک‌سازی حافظه
                if len(self.metrics_history) % 100 == 0:
                    gc.collect()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                print(f"خطا در مانیتورینگ: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self):
        """جمع‌آوری معیارهای سیستم"""
        now = datetime.now()
        
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=None)
            self._add_metric("cpu_percent", cpu_percent, "%", "system", now)
            
            # Memory
            memory = psutil.virtual_memory()
            self._add_metric("memory_percent", memory.percent, "%", "system", now)
            self._add_metric("memory_available_gb", memory.available / 1024**3, "GB", "system", now)
            
            # Disk
            disk = psutil.disk_usage('/')
            self._add_metric("disk_percent", disk.percent, "%", "system", now)
            
            # Network (اگر در دسترس باشد)
            try:
                network = psutil.net_io_counters()
                self._add_metric("network_bytes_sent", network.bytes_sent, "bytes", "network", now)
                self._add_metric("network_bytes_recv", network.bytes_recv, "bytes", "network", now)
            except:
                pass
            
            # Process specific
            process = psutil.Process()
            self._add_metric("process_memory_mb", process.memory_info().rss / 1024**2, "MB", "process", now)
            self._add_metric("process_cpu_percent", process.cpu_percent(), "%", "process", now)
            
        except Exception as e:
            print(f"خطا در جمع‌آوری معیارهای سیستم: {e}")
    
    def _add_metric(self, name: str, value: float, unit: str, category: str, timestamp: datetime):
        """اضافه کردن معیار جدید"""
        metric = PerformanceMetric(name, value, unit, timestamp, category)
        self.metrics_history.append(metric)
    
    def _check_alerts(self):
        """بررسی شرایط آلارم"""
        if not self.metrics_history:
            return
        
        latest_metrics = {}
        for metric in list(self.metrics_history)[-10:]:  # آخرین 10 معیار
            latest_metrics[metric.name] = metric.value
        
        # بررسی آستانه‌ها
        for threshold_name, threshold_value in self.alert_thresholds.items():
            if threshold_name in latest_metrics:
                current_value = latest_metrics[threshold_name]
                
                if current_value > threshold_value:
                    alert_data = {
                        'metric': threshold_name,
                        'current_value': current_value,
                        'threshold': threshold_value,
                        'severity': 'HIGH' if current_value > threshold_value * 1.2 else 'MEDIUM',
                        'timestamp': datetime.now()
                    }
                    
                    self._trigger_alert(alert_data)
    
    def _trigger_alert(self, alert_data: Dict[str, Any]):
        """فعال‌سازی آلارم"""
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                print(f"خطا در callback آلارم: {e}")
    
    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """اضافه کردن callback برای آلارم‌ها"""
        self.alert_callbacks.append(callback)
    
    def timer(self, name: str):
        """دکوراتور برای اندازه‌گیری زمان اجرای تابع"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    execution_time = (time.time() - start_time) * 1000  # میلی‌ثانیه
                    
                    self.record_function_timing(name, execution_time, True)
                    return result
                    
                except Exception as e:
                    execution_time = (time.time() - start_time) * 1000
                    self.record_function_timing(name, execution_time, False)
                    raise
            
            return wrapper
        return decorator
    
    def record_function_timing(self, function_name: str, execution_time_ms: float, success: bool):
        """ثبت زمان اجرای تابع"""
        if function_name not in self.function_timings:
            self.function_timings[function_name] = {
                'times': deque(maxlen=100),
                'success_count': 0,
                'failure_count': 0
            }
        
        timing_data = self.function_timings[function_name]
        timing_data['times'].append(execution_time_ms)
        
        if success:
            timing_data['success_count'] += 1
        else:
            timing_data['failure_count'] += 1
        
        # اضافه کردن به معیارهای کلی
        self._add_metric(
            f"function_{function_name}_timing",
            execution_time_ms,
            "ms",
            "function_timing",
            datetime.now()
        )
    
    def record_api_response_time(self, api_name: str, response_time_ms: float):
        """ثبت زمان پاسخ API"""
        self.api_response_times.append({
            'api_name': api_name,
            'response_time': response_time_ms,
            'timestamp': datetime.now()
        })
        
        self._add_metric(
            f"api_{api_name}_response_time",
            response_time_ms,
            "ms",
            "api_timing",
            datetime.now()
        )
    
    def increment_counter(self, counter_name: str, amount: int = 1):
        """افزایش شمارنده"""
        if counter_name in self.counters:
            self.counters[counter_name] += amount
        else:
            self.counters[counter_name] = amount
    
    def record_error(self, error_type: str, error_message: str = None):
        """ثبت خطا"""
        if error_type not in self.error_counts:
            self.error_counts[error_type] = {
                'count': 0,
                'last_occurrence': None,
                'sample_messages': deque(maxlen=5)
            }
        
        error_data = self.error_counts[error_type]
        error_data['count'] += 1
        error_data['last_occurrence'] = datetime.now()
        
        if error_message:
            error_data['sample_messages'].append(error_message)
        
        self.increment_counter('failed_requests')
    
    def get_system_status(self) -> Dict[str, Any]:
        """دریافت وضعیت کلی سیستم"""
        if not self.metrics_history:
            return {'status': 'NO_DATA'}
        
        # آخرین معیارها
        latest_metrics = {}
        for metric in list(self.metrics_history)[-20:]:
            latest_metrics[metric.name] = metric.value
        
        # محاسبه وضعیت کلی
        status = 'HEALTHY'
        warnings = []
        
        # بررسی CPU
        cpu_percent = latest_metrics.get('cpu_percent', 0)
        if cpu_percent > 80:
            status = 'WARNING'
            warnings.append(f"CPU usage بالا: {cpu_percent:.1f}%")
        
        # بررسی Memory
        memory_percent = latest_metrics.get('memory_percent', 0)
        if memory_percent > 85:
            status = 'CRITICAL'
            warnings.append(f"Memory usage بالا: {memory_percent:.1f}%")
        
        # بررسی Response Time
        recent_api_times = [r['response_time'] for r in list(self.api_response_times)[-10:]]
        if recent_api_times:
            avg_response_time = statistics.mean(recent_api_times)
            if avg_response_time > 5000:  # 5 ثانیه
                status = 'WARNING'
                warnings.append(f"Response time بالا: {avg_response_time:.0f}ms")
        
        uptime = datetime.now() - self.start_time
        
        return {
            'status': status,
            'uptime_seconds': uptime.total_seconds(),
            'uptime_formatted': str(uptime).split('.')[0],
            'warnings': warnings,
            'latest_metrics': latest_metrics,
            'counters': self.counters.copy(),
            'total_errors': sum(e['count'] for e in self.error_counts.values())
        }
    
    def get_function_stats(self) -> Dict[str, Any]:
        """آمار توابع"""
        stats = {}
        
        for func_name, timing_data in self.function_timings.items():
            times = list(timing_data['times'])
            
            if times:
                stats[func_name] = {
                    'call_count': len(times),
                    'success_rate': timing_data['success_count'] / (timing_data['success_count'] + timing_data['failure_count']),
                    'avg_time_ms': statistics.mean(times),
                    'min_time_ms': min(times),
                    'max_time_ms': max(times),
                    'median_time_ms': statistics.median(times),
                    'recent_times': times[-5:]  # آخرین 5 اجرا
                }
        
        return stats
    
    def get_api_stats(self) -> Dict[str, Any]:
        """آمار API ها"""
        api_stats = {}
        
        for api_data in self.api_response_times:
            api_name = api_data['api_name']
            
            if api_name not in api_stats:
                api_stats[api_name] = []
            
            api_stats[api_name].append(api_data['response_time'])
        
        # محاسبه آمار
        final_stats = {}
        for api_name, times in api_stats.items():
            if times:
                final_stats[api_name] = {
                    'call_count': len(times),
                    'avg_response_time': statistics.mean(times),
                    'min_response_time': min(times),
                    'max_response_time': max(times),
                    'median_response_time': statistics.median(times)
                }
        
        return final_stats
    
    def get_error_summary(self) -> Dict[str, Any]:
        """خلاصه خطاها"""
        total_errors = sum(e['count'] for e in self.error_counts.values())
        
        error_summary = {
            'total_errors': total_errors,
            'error_types': len(self.error_counts),
            'top_errors': []
        }
        
        # مرتب‌سازی خطاها بر اساس تعداد
        sorted_errors = sorted(
            self.error_counts.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )
        
        for error_type, error_data in sorted_errors[:5]:  # 5 خطای برتر
            error_summary['top_errors'].append({
                'type': error_type,
                'count': error_data['count'],
                'last_occurrence': error_data['last_occurrence'],
                'sample_messages': list(error_data['sample_messages'])
            })
        
        return error_summary
    
    def get_performance_report(self) -> Dict[str, Any]:
        """گزارش کامل عملکرد"""
        return {
            'system_status': self.get_system_status(),
            'function_stats': self.get_function_stats(),
            'api_stats': self.get_api_stats(),
            'error_summary': self.get_error_summary(),
            'monitoring_config': {
                'interval': self.monitoring_interval,
                'history_size': len(self.metrics_history),
                'max_history_size': self.max_history_size,
                'is_monitoring': self.is_monitoring
            }
        }
    
    def export_metrics(self, format_type: str = 'json') -> str:
        """صادرات معیارها"""
        metrics_data = []
        
        for metric in self.metrics_history:
            metrics_data.append({
                'name': metric.name,
                'value': metric.value,
                'unit': metric.unit,
                'category': metric.category,
                'timestamp': metric.timestamp.isoformat()
            })
        
        if format_type == 'json':
            import json
            return json.dumps(metrics_data, ensure_ascii=False, indent=2)
        elif format_type == 'csv':
            import csv
            import io
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=['name', 'value', 'unit', 'category', 'timestamp'])
            writer.writeheader()
            writer.writerows(metrics_data)
            return output.getvalue()
        else:
            raise ValueError(f"فرمت پشتیبانی نشده: {format_type}")

# تابع راحت برای استفاده
def create_performance_monitor(config: Dict[str, Any] = None) -> PerformanceMonitor:
    """ایجاد مانیتور عملکرد"""
    monitor = PerformanceMonitor(config)
    monitor.start_monitoring()
    return monitor
