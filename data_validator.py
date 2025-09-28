import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import re
import json
from enum import Enum

class ValidationError(Exception):
    """خطای اعتبارسنجی"""
    pass

class DataType(Enum):
    NUMERIC = "NUMERIC"
    STRING = "STRING"
    DATETIME = "DATETIME"
    BOOLEAN = "BOOLEAN"
    JSON = "JSON"
    PRICE = "PRICE"
    VOLUME = "VOLUME"
    PERCENTAGE = "PERCENTAGE"

class DataValidator:
    """
    کلاس اعتبارسنجی داده‌ها
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.strict_mode = self.config.get('strict_mode', True)
        self.cache_enabled = self.config.get('cache_validations', True)
        self.validation_cache = {}
        
        # الگوهای regex
        self.patterns = {
            'symbol': r'^[A-Z]{3,10}\/[A-Z]{3,10}$',  # BTC/USDT
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'api_key': r'^[a-zA-Z0-9]{20,64}$',
            'timeframe': r'^[1-9]\d*[smhd]$',  # 1m, 5m, 1h, 1d
            'price': r'^\d+(\.\d{1,8})?$'
        }
    
    def validate_dataframe(self, df: pd.DataFrame, required_columns: List[str] = None, 
                          min_rows: int = 1) -> bool:
        """اعتبارسنجی DataFrame"""
        try:
            if df is None:
                raise ValidationError("DataFrame نمی‌تواند None باشد")
            
            if df.empty:
                raise ValidationError("DataFrame خالی است")
            
            if len(df) < min_rows:
                raise ValidationError(f"DataFrame باید حداقل {min_rows} ردیف داشته باشد")
            
            # بررسی ستون‌های مورد نیاز
            if required_columns:
                missing_columns = set(required_columns) - set(df.columns)
                if missing_columns:
                    raise ValidationError(f"ستون‌های مورد نیاز یافت نشد: {missing_columns}")
            
            # بررسی وجود NaN
            if self.strict_mode and df.isnull().any().any():
                null_columns = df.columns[df.isnull().any()].tolist()
                raise ValidationError(f"مقادیر null در ستون‌ها: {null_columns}")
            
            return True
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"خطا در اعتبارسنجی DataFrame: {e}")
    
    def validate_kline_data(self, df: pd.DataFrame) -> bool:
        """اعتبارسنجی داده‌های کندل"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # اعتبارسنجی پایه
        self.validate_dataframe(df, required_columns, min_rows=10)
        
        try:
            # بررسی منطق OHLC
            for idx, row in df.iterrows():
                high = row['high']
                low = row['low']
                open_price = row['open']
                close = row['close']
                
                # High باید بالاترین باشد
                if high < max(open_price, close):
                    raise ValidationError(f"High < max(Open, Close) در ردیف {idx}")
                
                # Low باید پایین‌ترین باشد
                if low > min(open_price, close):
                    raise ValidationError(f"Low > min(Open, Close) در ردیف {idx}")
                
                # قیمت‌ها نباید منفی باشند
                if any(val <= 0 for val in [high, low, open_price, close]):
                    raise ValidationError(f"قیمت منفی یا صفر در ردیف {idx}")
                
                # حجم نباید منفی باشد
                if row['volume'] < 0:
                    raise ValidationError(f"حجم منفی در ردیف {idx}")
            
            # بررسی ترتیب زمانی (اگر timestamp موجود است)
            if 'timestamp' in df.columns:
                timestamps = pd.to_datetime(df['timestamp'])
                if not timestamps.is_monotonic_increasing:
                    raise ValidationError("ترتیب زمانی داده‌ها صحیح نیست")
            
            return True
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"خطا در اعتبارسنجی کندل: {e}")
    
    def validate_price(self, price: float, min_price: float = 0.0001, 
                      max_price: float = 1000000) -> bool:
        """اعتبارسنجی قیمت"""
        try:
            if not isinstance(price, (int, float)):
                raise ValidationError("قیمت باید عددی باشد")
            
            if np.isnan(price) or np.isinf(price):
                raise ValidationError("قیمت نامعتبر (NaN یا Inf)")
            
            if price <= min_price:
                raise ValidationError(f"قیمت باید بیشتر از {min_price} باشد")
            
            if price >= max_price:
                raise ValidationError(f"قیمت باید کمتر از {max_price} باشد")
            
            return True
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"خطا در اعتبارسنجی قیمت: {e}")
    
    def validate_symbol(self, symbol: str) -> bool:
        """اعتبارسنجی نماد معاملاتی"""
        try:
            if not isinstance(symbol, str):
                raise ValidationError("نماد باید رشته باشد")
            
            if not re.match(self.patterns['symbol'], symbol):
                raise ValidationError(f"فرمت نماد صحیح نیست: {symbol}")
            
            return True
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"خطا در اعتبارسنجی نماد: {e}")
    
    def validate_timeframe(self, timeframe: str) -> bool:
        """اعتبارسنجی تایم‌فریم"""
        try:
            if not isinstance(timeframe, str):
                raise ValidationError("تایم‌فریم باید رشته باشد")
            
            valid_timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
            
            if timeframe not in valid_timeframes:
                raise ValidationError(f"تایم‌فریم نامعتبر: {timeframe}")
            
            return True
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"خطا در اعتبارسنجی تایم‌فریم: {e}")
    
    def validate_percentage(self, value: float, min_val: float = -100, 
                           max_val: float = 1000) -> bool:
        """اعتبارسنجی درصد"""
        try:
            if not isinstance(value, (int, float)):
                raise ValidationError("درصد باید عددی باشد")
            
            if np.isnan(value) or np.isinf(value):
                raise ValidationError("درصد نامعتبر (NaN یا Inf)")
            
            if value < min_val or value > max_val:
                raise ValidationError(f"درصد باید بین {min_val} تا {max_val} باشد")
            
            return True
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"خطا در اعتبارسنجی درصد: {e}")
    
    def validate_signal_data(self, signal_data: dict) -> bool:
        """اعتبارسنجی داده‌های سیگنال"""
        try:
            required_fields = ['symbol', 'signal', 'confidence', 'current_price']
            
            # بررسی فیلدهای مورد نیاز
            for field in required_fields:
                if field not in signal_data:
                    raise ValidationError(f"فیلد مورد نیاز یافت نشد: {field}")
            
            # اعتبارسنجی نماد
            self.validate_symbol(signal_data['symbol'])
            
            # اعتبارسنجی سیگنال
            signal = signal_data['signal']
            if not isinstance(signal, (int, float)) or signal not in [-1, 0, 1]:
                raise ValidationError("سیگنال باید -1، 0 یا 1 باشد")
            
            # اعتبارسنجی اطمینان
            confidence = signal_data['confidence']
            if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
                raise ValidationError("اطمینان باید بین 0 تا 1 باشد")
            
            # اعتبارسنجی قیمت
            self.validate_price(signal_data['current_price'])
            
            # اعتبارسنجی اختیاری
            if 'stop_loss' in signal_data:
                self.validate_price(signal_data['stop_loss'])
            
            if 'take_profits' in signal_data:
                for tp in signal_data['take_profits']:
                    self.validate_price(tp)
            
            return True
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"خطا در اعتبارسنجی سیگنال: {e}")
    
    def validate_config_data(self, config_data: dict, schema: dict) -> bool:
        """اعتبارسنجی داده‌های تنظیمات"""
        try:
            return self._validate_schema(config_data, schema)
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"خطا در اعتبارسنجی تنظیمات: {e}")
    
    def _validate_schema(self, data: dict, schema: dict) -> bool:
        """اعتبارسنجی بر اساس schema"""
        try:
            for key, rules in schema.items():
                if key not in data:
                    if rules.get('required', False):
                        raise ValidationError(f"فیلد مورد نیاز یافت نشد: {key}")
                    continue
                
                value = data[key]
                expected_type = rules.get('type')
                
                # بررسی نوع داده
                if expected_type and not isinstance(value, expected_type):
                    raise ValidationError(f"نوع داده اشتباه برای {key}: انتظار {expected_type}")
                
                # بررسی محدوده
                if 'min' in rules and value < rules['min']:
                    raise ValidationError(f"مقدار {key} کمتر از حداقل: {rules['min']}")
                
                if 'max' in rules and value > rules['max']:
                    raise ValidationError(f"مقدار {key} بیشتر از حداکثر: {rules['max']}")
                
                # بررسی مقادیر مجاز
                if 'allowed' in rules and value not in rules['allowed']:
                    raise ValidationError(f"مقدار غیرمجاز برای {key}: {value}")
                
                # بررسی regex
                if 'pattern' in rules and isinstance(value, str):
                    if not re.match(rules['pattern'], value):
                        raise ValidationError(f"فرمت اشتباه برای {key}: {value}")
            
            return True
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"خطا در اعتبارسنجی schema: {e}")
    
    def sanitize_data(self, data: Any, data_type: DataType) -> Any:
        """پاک‌سازی و استاندارد کردن داده"""
        try:
            if data_type == DataType.NUMERIC:
                return float(data) if data is not None else None
            
            elif data_type == DataType.STRING:
                return str(data).strip() if data is not None else None
            
            elif data_type == DataType.PRICE:
                price = float(data)
                return round(price, 8)  # 8 رقم اعشار برای ارزهای دیجیتال
            
            elif data_type == DataType.PERCENTAGE:
                return round(float(data), 4)
            
            elif data_type == DataType.DATETIME:
                if isinstance(data, str):
                    return pd.to_datetime(data)
                return data
            
            elif data_type == DataType.JSON:
                if isinstance(data, str):
                    return json.loads(data)
                return data
            
            else:
                return data
                
        except Exception as e:
            raise ValidationError(f"خطا در پاک‌سازی داده: {e}")
    
    def validate_api_response(self, response: dict, expected_fields: List[str]) -> bool:
        """اعتبارسنجی پاسخ API"""
        try:
            if not isinstance(response, dict):
                raise ValidationError("پاسخ API باید dictionary باشد")
            
            # بررسی فیلدهای مورد انتظار
            for field in expected_fields:
                if field not in response:
                    raise ValidationError(f"فیلد مورد انتظار یافت نشد: {field}")
            
            # بررسی وجود خطا
            if 'error' in response:
                raise ValidationError(f"خطا در پاسخ API: {response['error']}")
            
            return True
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"خطا در اعتبارسنجی API: {e}")
    
    def validate_portfolio_data(self, portfolio: dict) -> bool:
        """اعتبارسنجی داده‌های پورتفولیو"""
        try:
            required_fields = ['total_balance', 'available_balance', 'positions']
            
            for field in required_fields:
                if field not in portfolio:
                    raise ValidationError(f"فیلد مورد نیاز در پورتفولیو یافت نشد: {field}")
            
            # اعتبارسنجی مقادیر
            if portfolio['total_balance'] < 0:
                raise ValidationError("موجودی کل نمی‌تواند منفی باشد")
            
            if portfolio['available_balance'] < 0:
                raise ValidationError("موجودی در دسترس نمی‌تواند منفی باشد")
            
            if portfolio['available_balance'] > portfolio['total_balance']:
                raise ValidationError("موجودی در دسترس نمی‌تواند از کل بیشتر باشد")
            
            # اعتبارسنجی پوزیشن‌ها
            positions = portfolio['positions']
            if not isinstance(positions, list):
                raise ValidationError("پوزیشن‌ها باید لیست باشد")
            
            for pos in positions:
                if 'symbol' not in pos or 'size' not in pos:
                    raise ValidationError("پوزیشن باید symbol و size داشته باشد")
                
                self.validate_symbol(pos['symbol'])
            
            return True
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"خطا در اعتبارسنجی پورتفولیو: {e}")
    
    def get_validation_report(self, data: Any, data_type: str) -> dict:
        """تولید گزارش اعتبارسنجی"""
        report = {
            'timestamp': datetime.now(),
            'data_type': data_type,
            'is_valid': False,
            'errors': [],
            'warnings': [],
            'suggestions': []
        }
        
        try:
            if data_type == 'kline':
                self.validate_kline_data(data)
            elif data_type == 'signal':
                self.validate_signal_data(data)
            elif data_type == 'portfolio':
                self.validate_portfolio_data(data)
            else:
                raise ValidationError(f"نوع داده ناشناخته: {data_type}")
            
            report['is_valid'] = True
            
        except ValidationError as e:
            report['errors'].append(str(e))
        except Exception as e:
            report['errors'].append(f"خطای غیرمنتظره: {e}")
        
        return report
