import logging
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any
import json
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import traceback

class CustomFormatter(logging.Formatter):
    """فرمت‌کننده سفارشی برای لاگ‌ها"""
    
    def __init__(self):
        super().__init__()
        
        # رنگ‌ها برای خروجی کنسول
        self.COLORS = {
            'DEBUG': '\033[36m',      # آبی روشن
            'INFO': '\033[32m',       # سبز
            'WARNING': '\033[33m',    # زرد
            'ERROR': '\033[31m',      # قرمز
            'CRITICAL': '\033[35m',   # بنفش
            'RESET': '\033[0m'        # بازگشت به حالت عادی
        }
        
        # فرمت‌های مختلف
        self.console_format = "{color}[{time}] {level:<8} | {name:<15} | {message}{reset}"
        self.file_format = "[{time}] {level:<8} | {name:<15} | {message}"
        self.detailed_format = "[{time}] {level:<8} | {name:<15} | {filename}:{lineno} | {funcname}() | {message}"
    
    def format(self, record):
        # آماده‌سازی اطلاعات رکورد
        log_data = {
            'time': datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S'),
            'level': record.levelname,
            'name': record.name,
            'message': record.getMessage(),
            'filename': record.filename,
            'lineno': record.lineno,
            'funcname': record.funcName,
            'color': self.COLORS.get(record.levelname, ''),
            'reset': self.COLORS['RESET']
        }
        
        # انتخاب فرمت
        if hasattr(record, 'console_output') and record.console_output:
            format_str = self.console_format
        elif hasattr(record, 'detailed') and record.detailed:
            format_str = self.detailed_format
        else:
            format_str = self.file_format
        
        return format_str.format(**log_data)

class LoggerManager:
    """مدیریت سیستم لاگ‌گیری"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.loggers = {}
        
        # تنظیمات پیش‌فرض
        self.log_dir = self.config.get('log_dir', 'logs')
        self.log_level = self.config.get('log_level', 'INFO')
        self.max_file_size = self.config.get('max_file_size_mb', 10) * 1024 * 1024
        self.backup_count = self.config.get('backup_count', 5)
        self.console_output = self.config.get('console_output', True)
        
        # ایجاد پوشه لاگ
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        # تنظیم formatter
        self.formatter = CustomFormatter()
    
    def get_logger(self, name: str, 
                   level: Optional[str] = None,
                   file_name: Optional[str] = None,
                   console: Optional[bool] = None) -> logging.Logger:
        """دریافت logger با تنظیمات مشخص"""
        
        if name in self.loggers:
            return self.loggers[name]
        
        # ایجاد logger جدید
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level or self.log_level))
        
        # جلوگیری از تکرار handler ها
        logger.handlers.clear()
        
        # File Handler
        if file_name is None:
            file_name = f"{name.lower().replace(' ', '_')}.log"
        
        file_path = os.path.join(self.log_dir, file_name)
        
        # استفاده از RotatingFileHandler
        file_handler = RotatingFileHandler(
            file_path,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(self.formatter)
        logger.addHandler(file_handler)
        
        # Console Handler
        if console if console is not None else self.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(self.formatter)
            logger.addHandler(console_handler)
        
        # ذخیره logger
        self.loggers[name] = logger
        
        return logger
    
    def create_trading_logger(self, symbol: str) -> logging.Logger:
        """ایجاد logger مخصوص معاملات"""
        logger_name = f"Trading_{symbol.replace('/', '_')}"
        file_name = f"trading_{symbol.replace('/', '_').lower()}.log"
        
        logger = self.get_logger(logger_name, file_name=file_name)
        
        # اضافه کردن متدهای سفارشی
        def log_signal(signal_data: Dict[str, Any]):
            logger.info(f"SIGNAL: {json.dumps(signal_data, ensure_ascii=False, indent=2)}")
        
        def log_trade_result(trade_data: Dict[str, Any]):
            logger.info(f"TRADE_RESULT: {json.dumps(trade_data, ensure_ascii=False, indent=2)}")
        
        def log_error_with_context(error: Exception, context: Dict[str, Any] = None):
            error_data = {
                'error_type': type(error).__name__,
                'error_message': str(error),
                'traceback': traceback.format_exc(),
                'context': context or {}
            }
            logger.error(f"ERROR_WITH_CONTEXT: {json.dumps(error_data, ensure_ascii=False, indent=2)}")
        
        # اتصال متدها به logger
        logger.log_signal = log_signal
        logger.log_trade_result = log_trade_result
        logger.log_error_with_context = log_error_with_context
        
        return logger
    
    def create_ai_logger(self) -> logging.Logger:
        """ایجاد logger مخصوص AI"""
        logger = self.get_logger("AI_Evaluator", file_name="ai_evaluator.log")
        
        def log_ai_response(ai_name: str, prompt: str, response: str, response_time: float):
            ai_data = {
                'ai_name': ai_name,
                'prompt_length': len(prompt),
                'response_length': len(response),
                'response_time': response_time,
                'timestamp': datetime.now().isoformat()
            }
            logger.info(f"AI_RESPONSE: {json.dumps(ai_data, ensure_ascii=False)}")
        
        def log_consensus(evaluations: Dict[str, Any], final_decision: str):
            consensus_data = {
                'participating_ais': list(evaluations.keys()),
                'final_decision': final_decision,
                'timestamp': datetime.now().isoformat()
            }
            logger.info(f"AI_CONSENSUS: {json.dumps(consensus_data, ensure_ascii=False)}")
        
        logger.log_ai_response = log_ai_response
        logger.log_consensus = log_consensus
        
        return logger
    
    def create_system_logger(self) -> logging.Logger:
        """ایجاد logger مخصوص سیستم"""
        logger = self.get_logger("System", file_name="system.log")
        
        def log_startup(config_summary: Dict[str, Any]):
            logger.info(f"SYSTEM_STARTUP: {json.dumps(config_summary, ensure_ascii=False, indent=2)}")
        
        def log_shutdown(stats: Dict[str, Any]):
            logger.info(f"SYSTEM_SHUTDOWN: {json.dumps(stats, ensure_ascii=False, indent=2)}")
        
        def log_performance(metrics: Dict[str, Any]):
            logger.info(f"PERFORMANCE: {json.dumps(metrics, ensure_ascii=False)}")
        
        logger.log_startup = log_startup
        logger.log_shutdown = log_shutdown
        logger.log_performance = log_performance
        
        return logger
    
    def create_risk_logger(self) -> logging.Logger:
        """ایجاد logger مخصوص مدیریت ریسک"""
        logger = self.get_logger("Risk_Management", file_name="risk_management.log")
        
        def log_risk_alert(alert_type: str, details: Dict[str, Any]):
            risk_data = {
                'alert_type': alert_type,
                'severity': details.get('severity', 'MEDIUM'),
                'details': details,
                'timestamp': datetime.now().isoformat()
            }
            logger.warning(f"RISK_ALERT: {json.dumps(risk_data, ensure_ascii=False, indent=2)}")
        
        def log_position_size(symbol: str, calculation: Dict[str, Any]):
            position_data = {
                'symbol': symbol,
                'calculation': calculation,
                'timestamp': datetime.now().isoformat()
            }
            logger.info(f"POSITION_SIZE: {json.dumps(position_data, ensure_ascii=False)}")
        
        logger.log_risk_alert = log_risk_alert
        logger.log_position_size = log_position_size
        
        return logger
    
    def setup_daily_rotation(self, logger_name: str) -> logging.Logger:
        """تنظیم چرخش روزانه لاگ‌ها"""
        if logger_name in self.loggers:
            logger = self.loggers[logger_name]
        else:
            logger = logging.getLogger(logger_name)
        
        # حذف handler های قبلی
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Daily rotating file handler
        file_path = os.path.join(self.log_dir, f"{logger_name.lower()}_daily.log")
        daily_handler = TimedRotatingFileHandler(
            file_path,
            when='midnight',
            interval=1,
            backupCount=30,
            encoding='utf-8'
        )
        daily_handler.setFormatter(self.formatter)
        logger.addHandler(daily_handler)
        
        # Console handler
        if self.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(self.formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def log_structured_data(self, logger_name: str, data_type: str, data: Dict[str, Any]):
        """لاگ داده‌های ساختاریافته"""
        if logger_name not in self.loggers:
            raise ValueError(f"Logger {logger_name} یافت نشد")
        
        logger = self.loggers[logger_name]
        
        structured_data = {
            'data_type': data_type,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        logger.info(f"STRUCTURED_DATA: {json.dumps(structured_data, ensure_ascii=False)}")
    
    def get_log_stats(self) -> Dict[str, Any]:
        """آمار لاگ‌ها"""
        stats = {
            'active_loggers': len(self.loggers),
            'log_files': [],
            'total_size_mb': 0
        }
        
        try:
            for file in os.listdir(self.log_dir):
                if file.endswith('.log'):
                    file_path = os.path.join(self.log_dir, file)
                    file_size = os.path.getsize(file_path)
                    
                    stats['log_files'].append({
                        'name': file,
                        'size_mb': round(file_size / 1024 / 1024, 2),
                        'modified': datetime.fromtimestamp(os.path.getmtime(file_path))
                    })
                    
                    stats['total_size_mb'] += file_size / 1024 / 1024
            
            stats['total_size_mb'] = round(stats['total_size_mb'], 2)
            
        except Exception as e:
            print(f"خطا در محاسبه آمار لاگ: {e}")
        
        return stats
    
    def cleanup_old_logs(self, days_to_keep: int = 30) -> int:
        """پاک‌سازی لاگ‌های قدیمی"""
        deleted_count = 0
        cutoff_date = datetime.now().timestamp() - (days_to_keep * 24 * 3600)
        
        try:
            for file in os.listdir(self.log_dir):
                file_path = os.path.join(self.log_dir, file)
                
                if os.path.isfile(file_path) and file.endswith('.log'):
                    if os.path.getmtime(file_path) < cutoff_date:
                        os.remove(file_path)
                        deleted_count += 1
            
        except Exception as e:
            print(f"خطا در پاک‌سازی لاگ‌ها: {e}")
        
        return deleted_count

# تابع راحت برای دسترسی سریع
def setup_logger(name: str, config: Dict[str, Any] = None) -> logging.Logger:
    """تابع راحت برای ایجاد logger"""
    manager = LoggerManager(config)
    return manager.get_logger(name)
