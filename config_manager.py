import json
import os
import yaml
from typing import Dict, Any, Optional, List
from datetime import datetime
import copy

class ConfigError(Exception):
    """خطای مدیریت تنظیمات"""
    pass

class ConfigManager:
    """
    مدیریت تنظیمات و پیکربندی
    """
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or "config"
        self.configs = {}
        self.config_history = []
        self.default_configs = {}
        
        # ایجاد پوشه تنظیمات اگر وجود ندارد
        if not os.path.exists(self.config_path):
            os.makedirs(self.config_path)
    
    def load_config(self, config_name: str, file_path: str = None) -> Dict[str, Any]:
        """بارگذاری تنظیمات از فایل"""
        try:
            if file_path is None:
                file_path = os.path.join(self.config_path, f"{config_name}.json")
            
            if not os.path.exists(file_path):
                raise ConfigError(f"فایل تنظیمات یافت نشد: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as file:
                if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                    config_data = yaml.safe_load(file)
                else:
                    config_data = json.load(file)
            
            self.configs[config_name] = config_data
            self._add_to_history('load', config_name, file_path)
            
            return config_data
            
        except Exception as e:
            raise ConfigError(f"خطا در بارگذاری تنظیمات {config_name}: {e}")
    
    def save_config(self, config_name: str, config_data: Dict[str, Any], 
                   file_path: str = None, backup: bool = True) -> bool:
        """ذخیره تنظیمات در فایل"""
        try:
            if file_path is None:
                file_path = os.path.join(self.config_path, f"{config_name}.json")
            
            # پشتیبان‌گیری قبل از ذخیره
            if backup and os.path.exists(file_path):
                self._create_backup(file_path)
            
            # ایجاد پوشه اگر وجود ندارد
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as file:
                if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                    yaml.dump(config_data, file, default_flow_style=False, 
                             allow_unicode=True, indent=2)
                else:
                    json.dump(config_data, file, ensure_ascii=False, indent=2)
            
            self.configs[config_name] = copy.deepcopy(config_data)
            self._add_to_history('save', config_name, file_path)
            
            return True
            
        except Exception as e:
            raise ConfigError(f"خطا در ذخیره تنظیمات {config_name}: {e}")
    
    def get_config(self, config_name: str, key: str = None, default: Any = None) -> Any:
        """دریافت تنظیمات"""
        try:
            if config_name not in self.configs:
                raise ConfigError(f"تنظیمات {config_name} بارگذاری نشده")
            
            config = self.configs[config_name]
            
            if key is None:
                return config
            
            # پشتیبانی از کلیدهای تودرتو (nested keys)
            keys = key.split('.')
            value = config
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            return value
            
        except Exception as e:
            if default is not None:
                return default
            raise ConfigError(f"خطا در دریافت تنظیمات: {e}")
    
    def set_config(self, config_name: str, key: str, value: Any) -> bool:
        """تنظیم مقدار"""
        try:
            if config_name not in self.configs:
                self.configs[config_name] = {}
            
            config = self.configs[config_name]
            
            # پشتیبانی از کلیدهای تودرتو
            keys = key.split('.')
            current = config
            
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            current[keys[-1]] = value
            self._add_to_history('set', config_name, f"{key}={value}")
            
            return True
            
        except Exception as e:
            raise ConfigError(f"خطا در تنظیم مقدار: {e}")
    
    def merge_configs(self, base_config: str, override_config: str, 
                     result_config: str = None) -> Dict[str, Any]:
        """ترکیب دو تنظیمات"""
        try:
            if base_config not in self.configs:
                raise ConfigError(f"تنظیمات پایه {base_config} یافت نشد")
            
            if override_config not in self.configs:
                raise ConfigError(f"تنظیمات بازنویسی {override_config} یافت نشد")
            
            base = copy.deepcopy(self.configs[base_config])
            override = self.configs[override_config]
            
            merged = self._deep_merge(base, override)
            
            if result_config:
                self.configs[result_config] = merged
            
            return merged
            
        except Exception as e:
            raise ConfigError(f"خطا در ترکیب تنظیمات: {e}")
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """ترکیب عمیق dictionary ها"""
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        
        return result
    
    def validate_config(self, config_name: str, schema: Dict) -> Dict[str, List[str]]:
        """اعتبارسنجی تنظیمات"""
        try:
            if config_name not in self.configs:
                raise ConfigError(f"تنظیمات {config_name} یافت نشد")
            
            config = self.configs[config_name]
            errors = []
            warnings = []
            
            # بررسی فیلدهای مورد نیاز
            required_fields = schema.get('required', [])
            for field in required_fields:
                if field not in config:
                    errors.append(f"فیلد مورد نیاز یافت نشد: {field}")
            
            # بررسی نوع داده‌ها
            field_types = schema.get('types', {})
            for field, expected_type in field_types.items():
                if field in config and not isinstance(config[field], expected_type):
                    errors.append(f"نوع داده اشتباه برای {field}: انتظار {expected_type}")
            
            # بررسی محدوده‌ها
            ranges = schema.get('ranges', {})
            for field, (min_val, max_val) in ranges.items():
                if field in config:
                    value = config[field]
                    if isinstance(value, (int, float)):
                        if value < min_val or value > max_val:
                            errors.append(f"مقدار {field} خارج از محدوده: {min_val}-{max_val}")
            
            # بررسی فیلدهای منسوخ شده
            deprecated_fields = schema.get('deprecated', [])
            for field in deprecated_fields:
                if field in config:
                    warnings.append(f"فیلد منسوخ شده: {field}")
            
            return {'errors': errors, 'warnings': warnings}
            
        except Exception as e:
            raise ConfigError(f"خطا در اعتبارسنجی: {e}")
    
    def create_default_config(self, config_name: str, template: str = 'basic') -> Dict[str, Any]:
        """ایجاد تنظیمات پیش‌فرض"""
        try:
            if template == 'basic':
                default_config = {
                    'version': '1.0.0',
                    'created_at': datetime.now().isoformat(),
                    'settings': {
                        'debug_mode': False,
                        'log_level': 'INFO'
                    }
                }
            
            elif template == 'trading':
                default_config = {
                    'version': '1.0.0',
                    'created_at': datetime.now().isoformat(),
                    'trading': {
                        'max_leverage': 10,
                        'risk_per_trade': 0.02,
                        'max_daily_trades': 5
                    },
                    'symbols': ['BTC/USDT', 'ETH/USDT'],
                    'timeframe': '1h',
                    'analysis': {
                        'min_confidence': 0.7,
                        'use_ai_validation': True
                    }
                }
            
            elif template == 'notifications':
                default_config = {
                    'version': '1.0.0',
                    'created_at': datetime.now().isoformat(),
                    'telegram': {
                        'enabled': True,
                        'send_signals': True,
                        'send_alerts': True
                    },
                    'email': {
                        'enabled': False
                    },
                    'discord': {
                        'enabled': False
                    }
                }
            
            else:
                raise ConfigError(f"الگوی ناشناخته: {template}")
            
            self.configs[config_name] = default_config
            self.default_configs[config_name] = copy.deepcopy(default_config)
            
            return default_config
            
        except Exception as e:
            raise ConfigError(f"خطا در ایجاد تنظیمات پیش‌فرض: {e}")
    
    def reset_to_default(self, config_name: str) -> bool:
        """بازگشت به تنظیمات پیش‌فرض"""
        try:
            if config_name not in self.default_configs:
                raise ConfigError(f"تنظیمات پیش‌فرض برای {config_name} یافت نشد")
            
            self.configs[config_name] = copy.deepcopy(self.default_configs[config_name])
            self._add_to_history('reset', config_name, 'to_default')
            
            return True
            
        except Exception as e:
            raise ConfigError(f"خطا در بازگشت به پیش‌فرض: {e}")
    
    def _create_backup(self, file_path: str) -> str:
        """ایجاد پشتیبان"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = f"{file_path}.backup_{timestamp}"
            
            with open(file_path, 'r', encoding='utf-8') as src:
                with open(backup_path, 'w', encoding='utf-8') as dst:
                    dst.write(src.read())
            
            return backup_path
            
        except Exception as e:
            raise ConfigError(f"خطا در ایجاد پشتیبان: {e}")
    
    def _add_to_history(self, action: str, config_name: str, details: str):
        """اضافه کردن به تاریخچه"""
        self.config_history.append({
            'timestamp': datetime.now(),
            'action': action,
            'config_name': config_name,
            'details': details
        })
        
        # حفظ حداکثر 100 رکورد
        if len(self.config_history) > 100:
            self.config_history = self.config_history[-100:]
    
    def get_history(self, config_name: str = None) -> List[Dict]:
        """دریافت تاریخچه تغییرات"""
        if config_name:
            return [h for h in self.config_history if h['config_name'] == config_name]
        return self.config_history
    
    def export_config(self, config_name: str, format_type: str = 'json') -> str:
        """صادرات تنظیمات"""
        try:
            if config_name not in self.configs:
                raise ConfigError(f"تنظیمات {config_name} یافت نشد")
            
            config = self.configs[config_name]
            
            if format_type == 'json':
                return json.dumps(config, ensure_ascii=False, indent=2)
            elif format_type == 'yaml':
                return yaml.dump(config, default_flow_style=False, allow_unicode=True)
            else:
                raise ConfigError(f"فرمت ناشناخته: {format_type}")
                
        except Exception as e:
            raise ConfigError(f"خطا در صادرات: {e}")
    
    def import_config(self, config_name: str, config_string: str, 
                     format_type: str = 'json') -> bool:
        """وارد کردن تنظیمات از رشته"""
        try:
            if format_type == 'json':
                config_data = json.loads(config_string)
            elif format_type == 'yaml':
                config_data = yaml.safe_load(config_string)
            else:
                raise ConfigError(f"فرمت ناشناخته: {format_type}")
            
            self.configs[config_name] = config_data
            self._add_to_history('import', config_name, format_type)
            
            return True
            
        except Exception as e:
            raise ConfigError(f"خطا در وارد کردن: {e}")
