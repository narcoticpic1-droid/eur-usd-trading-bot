import json
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import re
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum

class FormatType(Enum):
    """انواع فرمت‌بندی"""
    PERSIAN = "persian"
    ENGLISH = "english"
    MIXED = "mixed"

class FormattingHelper:
    """کلاس کمکی برای فرمت‌بندی داده‌ها"""
    
    def __init__(self, default_format: FormatType = FormatType.MIXED):
        self.default_format = default_format
        
        # اعداد فارسی
        self.persian_digits = {
            '0': '۰', '1': '۱', '2': '۲', '3': '۳', '4': '۴',
            '5': '۵', '6': '۶', '7': '۷', '8': '۸', '9': '۹'
        }
        
        # اعداد انگلیسی
        self.english_digits = {v: k for k, v in self.persian_digits.items()}
        
        # واحدهای زمانی فارسی
        self.time_units_persian = {
            'second': 'ثانیه', 'seconds': 'ثانیه',
            'minute': 'دقیقه', 'minutes': 'دقیقه',
            'hour': 'ساعت', 'hours': 'ساعت',
            'day': 'روز', 'days': 'روز',
            'week': 'هفته', 'weeks': 'هفته',
            'month': 'ماه', 'months': 'ماه',
            'year': 'سال', 'years': 'سال'
        }
        
        # واحدهای مالی فارسی
        self.currency_units_persian = {
            'USD': 'دلار', 'USDT': 'تتر', 'BTC': 'بیت‌کوین',
            'ETH': 'اتریوم', 'SOL': 'سولانا'
        }
        
        # الگوهای تحلیل فارسی
        self.analysis_terms_persian = {
            'STRONG_UPTREND': 'صعودی قوی',
            'WEAK_UPTREND': 'صعودی ضعیف',
            'CONSOLIDATION': 'تثبیت',
            'WEAK_DOWNTREND': 'نزولی ضعیف',
            'STRONG_DOWNTREND': 'نزولی قوی',
            'BREAKOUT': 'شکست سطح',
            'PULLBACK': 'اصلاح',
            'REVERSAL': 'برگشت',
            'CONTINUATION': 'ادامه روند',
            'RANGE_TRADE': 'معامله محدوده‌ای',
            'HIGH': 'بالا',
            'MEDIUM': 'متوسط',
            'LOW': 'پایین',
            'EXCELLENT': 'عالی',
            'INVALID': 'نامعتبر'
        }
    
    def format_number(self, number: Union[int, float, Decimal], 
                     decimal_places: int = 2,
                     use_persian: bool = None,
                     add_commas: bool = True) -> str:
        """فرمت‌بندی اعداد"""
        if use_persian is None:
            use_persian = self.default_format in [FormatType.PERSIAN, FormatType.MIXED]
        
        try:
            # تبدیل به Decimal برای دقت بالا
            if isinstance(number, (int, float)):
                decimal_number = Decimal(str(number))
            else:
                decimal_number = number
            
            # گرد کردن
            if decimal_places >= 0:
                rounded = decimal_number.quantize(
                    Decimal('0.' + '0' * decimal_places),
                    rounding=ROUND_HALF_UP
                )
            else:
                rounded = decimal_number
            
            # تبدیل به رشته
            formatted = f"{rounded:.{max(0, decimal_places)}f}"
            
            # اضافه کردن کامای جداکننده
            if add_commas:
                parts = formatted.split('.')
                parts[0] = self._add_thousand_separators(parts[0])
                formatted = '.'.join(parts)
            
            # تبدیل به فارسی
            if use_persian:
                formatted = self._to_persian_digits(formatted)
            
            return formatted
            
        except Exception:
            return str(number)
    
    def format_price(self, price: Union[int, float], 
                    symbol: str = "USD",
                    use_persian: bool = None) -> str:
        """فرمت‌بندی قیمت"""
        if use_persian is None:
            use_persian = self.default_format in [FormatType.PERSIAN, FormatType.MIXED]
        
        # تعیین تعداد اعشار بر اساس قیمت
        if price >= 1000:
            decimal_places = 2
        elif price >= 1:
            decimal_places = 4
        else:
            decimal_places = 6
        
        formatted_price = self.format_number(price, decimal_places, use_persian)
        
        # اضافه کردن واحد
        if use_persian and symbol in self.currency_units_persian:
            currency_unit = self.currency_units_persian[symbol]
            return f"{formatted_price} {currency_unit}"
        else:
            return f"${formatted_price}" if symbol == "USD" else f"{formatted_price} {symbol}"
    
    def format_percentage(self, percentage: Union[int, float],
                         include_sign: bool = True,
                         use_persian: bool = None) -> str:
        """فرمت‌بندی درصد"""
        if use_persian is None:
            use_persian = self.default_format in [FormatType.PERSIAN, FormatType.MIXED]
        
        formatted = self.format_number(percentage, 2, use_persian)
        
        if include_sign and percentage > 0:
            formatted = f"+{formatted}"
        
        return f"{formatted}%" if not use_persian else f"{formatted}٪"
    
    def format_volume(self, volume: Union[int, float],
                     use_persian: bool = None) -> str:
        """فرمت‌بندی حجم"""
        if use_persian is None:
            use_persian = self.default_format in [FormatType.PERSIAN, FormatType.MIXED]
        
        if volume >= 1_000_000_000:  # میلیارد
            formatted = self.format_number(volume / 1_000_000_000, 2, use_persian)
            unit = "میلیارد" if use_persian else "B"
        elif volume >= 1_000_000:  # میلیون
            formatted = self.format_number(volume / 1_000_000, 2, use_persian)
            unit = "میلیون" if use_persian else "M"
        elif volume >= 1_000:  # هزار
            formatted = self.format_number(volume / 1_000, 2, use_persian)
            unit = "هزار" if use_persian else "K"
        else:
            formatted = self.format_number(volume, 0, use_persian)
            unit = ""
        
        return f"{formatted} {unit}".strip()
    
    def format_duration(self, seconds: Union[int, float],
                       use_persian: bool = None,
                       precision: str = "auto") -> str:
        """فرمت‌بندی مدت زمان"""
        if use_persian is None:
            use_persian = self.default_format in [FormatType.PERSIAN, FormatType.MIXED]
        
        if seconds < 60:  # کمتر از 1 دقیقه
            formatted = self.format_number(seconds, 1, use_persian)
            unit = "ثانیه" if use_persian else "s"
            return f"{formatted} {unit}"
        
        elif seconds < 3600:  # کمتر از 1 ساعت
            minutes = seconds / 60
            formatted = self.format_number(minutes, 1, use_persian)
            unit = "دقیقه" if use_persian else "min"
            return f"{formatted} {unit}"
        
        elif seconds < 86400:  # کمتر از 1 روز
            hours = seconds / 3600
            formatted = self.format_number(hours, 1, use_persian)
            unit = "ساعت" if use_persian else "h"
            return f"{formatted} {unit}"
        
        else:  # بیش از 1 روز
            days = seconds / 86400
            formatted = self.format_number(days, 1, use_persian)
            unit = "روز" if use_persian else "d"
            return f"{formatted} {unit}"
    
    def format_datetime(self, dt: datetime,
                       format_type: str = "full",
                       use_persian: bool = None) -> str:
        """فرمت‌بندی تاریخ و زمان"""
        if use_persian is None:
            use_persian = self.default_format in [FormatType.PERSIAN, FormatType.MIXED]
        
        if format_type == "time_only":
            formatted = dt.strftime("%H:%M:%S")
        elif format_type == "date_only":
            formatted = dt.strftime("%Y-%m-%d")
        elif format_type == "short":
            formatted = dt.strftime("%m/%d %H:%M")
        else:  # full
            formatted = dt.strftime("%Y-%m-%d %H:%M:%S")
        
        if use_persian:
            formatted = self._to_persian_digits(formatted)
        
        return formatted
    
    def format_signal_data(self, signal_data: Dict[str, Any],
                          use_persian: bool = None) -> Dict[str, str]:
        """فرمت‌بندی داده‌های سیگنال"""
        if use_persian is None:
            use_persian = self.default_format in [FormatType.PERSIAN, FormatType.MIXED]
        
        formatted = {}
        
        # قیمت‌ها
        if 'current_price' in signal_data:
            formatted['current_price'] = self.format_price(signal_data['current_price'])
        
        if 'entry_price' in signal_data:
            formatted['entry_price'] = self.format_price(signal_data['entry_price'])
        
        if 'stop_loss' in signal_data:
            formatted['stop_loss'] = self.format_price(signal_data['stop_loss'])
        
        if 'take_profits' in signal_data and isinstance(signal_data['take_profits'], list):
            formatted['take_profits'] = [
                self.format_price(tp) for tp in signal_data['take_profits']
            ]
        
        # درصدها
        if 'confidence' in signal_data:
            formatted['confidence'] = self.format_percentage(signal_data['confidence'] * 100)
        
        if 'risk_reward_ratio' in signal_data:
            formatted['risk_reward_ratio'] = self.format_number(signal_data['risk_reward_ratio'], 2, use_persian)
        
        # ترجمه terms
        if 'signal_type' in signal_data and use_persian:
            signal_type = signal_data['signal_type']
            formatted['signal_type_persian'] = self.analysis_terms_persian.get(signal_type, signal_type)
        
        if 'signal_quality' in signal_data and use_persian:
            quality = signal_data['signal_quality']
            formatted['signal_quality_persian'] = self.analysis_terms_persian.get(quality, quality)
        
        # ساختار بازار
        if 'market_context' in signal_data and isinstance(signal_data['market_context'], dict):
            market_context = signal_data['market_context']
            
            if 'structure' in market_context and use_persian:
                structure = market_context['structure']
                formatted['market_structure_persian'] = self.analysis_terms_persian.get(structure, structure)
        
        return formatted
    
    def format_performance_stats(self, stats: Dict[str, Any],
                                use_persian: bool = None) -> Dict[str, str]:
        """فرمت‌بندی آمار عملکرد"""
        if use_persian is None:
            use_persian = self.default_format in [FormatType.PERSIAN, FormatType.MIXED]
        
        formatted = {}
        
        # درصدها
        percentage_fields = ['win_rate', 'success_rate', 'accuracy', 'cpu_percent', 'memory_percent']
        for field in percentage_fields:
            if field in stats:
                formatted[field] = self.format_percentage(stats[field])
        
        # اعداد معمولی
        number_fields = ['total_signals', 'successful_trades', 'failed_trades']
        for field in number_fields:
            if field in stats:
                formatted[field] = self.format_number(stats[field], 0, use_persian)
        
        # زمان‌ها
        time_fields = ['avg_response_time', 'execution_time']
        for field in time_fields:
            if field in stats:
                # اگر بر حسب میلی‌ثانیه باشد
                if stats[field] > 1000:
                    formatted[field] = self.format_duration(stats[field] / 1000)
                else:
                    formatted[field] = f"{self.format_number(stats[field], 0, use_persian)} ms"
        
        return formatted
    
    def _to_persian_digits(self, text: str) -> str:
        """تبدیل اعداد انگلیسی به فارسی"""
        for english, persian in self.persian_digits.items():
            text = text.replace(english, persian)
        return text
    
    def _to_english_digits(self, text: str) -> str:
        """تبدیل اعداد فارسی به انگلیسی"""
        for persian, english in self.english_digits.items():
            text = text.replace(persian, english)
        return text
    
    def _add_thousand_separators(self, number_str: str) -> str:
        """اضافه کردن کامای جداکننده هزارگان"""
        # حذف علامت منفی موقت
        negative = number_str.startswith('-')
        if negative:
            number_str = number_str[1:]
        
        # اضافه کردن کاما
        if len(number_str) > 3:
            reversed_digits = number_str[::-1]
            groups = [reversed_digits[i:i+3] for i in range(0, len(reversed_digits), 3)]
            formatted = ','.join(groups)[::-1]
        else:
            formatted = number_str
        
        # بازگرداندن علامت منفی
        if negative:
            formatted = '-' + formatted
        
        return formatted
    
    def create_table(self, data: List[Dict[str, Any]], 
                    headers: List[str],
                    use_persian: bool = None) -> str:
        """ایجاد جدول فرمت شده"""
        if use_persian is None:
            use_persian = self.default_format in [FormatType.PERSIAN, FormatType.MIXED]
        
        if not data:
            return "داده‌ای موجود نیست" if use_persian else "No data available"
        
        # محاسبه عرض ستون‌ها
        col_widths = {}
        for header in headers:
            col_widths[header] = len(header)
            
            for row in data:
                if header in row:
                    value_str = str(row[header])
                    col_widths[header] = max(col_widths[header], len(value_str))
        
        # ایجاد خط جداکننده
        separator = '+' + '+'.join(['-' * (col_widths[h] + 2) for h in headers]) + '+'
        
        # ایجاد header
        header_row = '|' + '|'.join([f" {h:<{col_widths[h]}} " for h in headers]) + '|'
        
        # ایجاد سطرهای داده
        data_rows = []
        for row in data:
            row_str = '|'
            for header in headers:
                value = row.get(header, '')
                row_str += f" {str(value):<{col_widths[header]}} |"
            data_rows.append(row_str)
        
        # ترکیب همه
        table = '\n'.join([separator, header_row, separator] + data_rows + [separator])
        
        return table
    
    def format_json_pretty(self, data: Dict[str, Any],
                          use_persian: bool = None) -> str:
        """فرمت‌بندی JSON زیبا"""
        if use_persian is None:
            use_persian = self.default_format in [FormatType.PERSIAN, FormatType.MIXED]
        
        formatted = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)
        
        if use_persian:
            formatted = self._to_persian_digits(formatted)
        
        return formatted
    
    def clean_text(self, text: str) -> str:
        """پاک‌سازی متن"""
        # حذف فضاهای اضافی
        text = re.sub(r'\s+', ' ', text)
        
        # حذف کاراکترهای کنترلی
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        
        # تمیز کردن ابتدا و انتها
        text = text.strip()
        
        return text
    
    def truncate_text(self, text: str, max_length: int, 
                     suffix: str = "...") -> str:
        """کوتاه کردن متن"""
        if len(text) <= max_length:
            return text
        
        truncated = text[:max_length - len(suffix)]
        return truncated + suffix
    
    def format_emoji_status(self, status: str) -> str:
        """اضافه کردن ایموجی بر اساس وضعیت"""
        emoji_map = {
            'success': '✅',
            'error': '❌',
            'warning': '⚠️',
            'info': 'ℹ️',
            'loading': '🔄',
            'buy': '📈',
            'sell': '📉',
            'hold': '⏸️',
            'high': '🟢',
            'medium': '🟡',
            'low': '🟠',
            'critical': '🔴'
        }
        
        return emoji_map.get(status.lower(), '📊')

# تابع راحت برای استفاده
def format_price_simple(price: Union[int, float], decimals: int = 4) -> str:
    """تابع ساده برای فرمت قیمت"""
    formatter = FormattingHelper()
    return formatter.format_price(price)

def format_percentage_simple(percentage: Union[int, float]) -> str:
    """تابع ساده برای فرمت درصد"""
    formatter = FormattingHelper()
    return formatter.format_percentage(percentage)
