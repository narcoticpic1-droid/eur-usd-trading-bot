import datetime
import pytz
from typing import Optional, List, Union
import pandas as pd
from enum import Enum
import time

class TimeFrame(Enum):
    ONE_MINUTE = "1m"
    THREE_MINUTES = "3m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    THIRTY_MINUTES = "30m"
    ONE_HOUR = "1h"
    TWO_HOURS = "2h"
    FOUR_HOURS = "4h"
    SIX_HOURS = "6h"
    EIGHT_HOURS = "8h"
    TWELVE_HOURS = "12h"
    ONE_DAY = "1d"
    THREE_DAYS = "3d"
    ONE_WEEK = "1w"
    ONE_MONTH = "1M"

class TimeUtils:
    """
    ابزارهای کار با زمان
    """
    
    def __init__(self, default_timezone: str = 'UTC'):
        self.default_timezone = pytz.timezone(default_timezone)
        self.iranian_timezone = pytz.timezone('Asia/Tehran')
        
        # تعریف مدت زمان هر تایم‌فریم به ثانیه
        self.timeframe_seconds = {
            TimeFrame.ONE_MINUTE: 60,
            TimeFrame.THREE_MINUTES: 180,
            TimeFrame.FIVE_MINUTES: 300,
            TimeFrame.FIFTEEN_MINUTES: 900,
            TimeFrame.THIRTY_MINUTES: 1800,
            TimeFrame.ONE_HOUR: 3600,
            TimeFrame.TWO_HOURS: 7200,
            TimeFrame.FOUR_HOURS: 14400,
            TimeFrame.SIX_HOURS: 21600,
            TimeFrame.EIGHT_HOURS: 28800,
            TimeFrame.TWELVE_HOURS: 43200,
            TimeFrame.ONE_DAY: 86400,
            TimeFrame.THREE_DAYS: 259200,
            TimeFrame.ONE_WEEK: 604800,
            TimeFrame.ONE_MONTH: 2592000  # تقریبی
        }
    
    def now(self, timezone: str = None) -> datetime.datetime:
        """زمان فعلی"""
        tz = pytz.timezone(timezone) if timezone else self.default_timezone
        return datetime.datetime.now(tz)
    
    def now_utc(self) -> datetime.datetime:
        """زمان فعلی UTC"""
        return datetime.datetime.now(pytz.UTC)
    
    def now_iranian(self) -> datetime.datetime:
        """زمان فعلی ایران"""
        return datetime.datetime.now(self.iranian_timezone)
    
    def timestamp_to_datetime(self, timestamp: Union[int, float], 
                             timezone: str = None) -> datetime.datetime:
        """تبدیل timestamp به datetime"""
        try:
            # تشخیص نوع timestamp (ثانیه یا میلی‌ثانیه)
            if timestamp > 1e10:  # میلی‌ثانیه
                timestamp = timestamp / 1000
            
            dt = datetime.datetime.fromtimestamp(timestamp)
            
            if timezone:
                tz = pytz.timezone(timezone)
                dt = dt.replace(tzinfo=pytz.UTC).astimezone(tz)
            else:
                dt = dt.replace(tzinfo=self.default_timezone)
            
            return dt
            
        except Exception as e:
            raise ValueError(f"خطا در تبدیل timestamp: {e}")
    
    def datetime_to_timestamp(self, dt: datetime.datetime, 
                             milliseconds: bool = False) -> Union[int, float]:
        """تبدیل datetime به timestamp"""
        try:
            timestamp = dt.timestamp()
            
            if milliseconds:
                return int(timestamp * 1000)
            else:
                return int(timestamp)
                
        except Exception as e:
            raise ValueError(f"خطا در تبدیل datetime: {e}")
    
    def format_datetime(self, dt: datetime.datetime, 
                       format_type: str = 'persian') -> str:
        """فرمت کردن datetime"""
        try:
            if format_type == 'persian':
                # تبدیل به زمان ایران
                iranian_dt = dt.astimezone(self.iranian_timezone)
                return iranian_dt.strftime('%Y/%m/%d %H:%M:%S')
            
            elif format_type == 'iso':
                return dt.isoformat()
            
            elif format_type == 'readable':
                return dt.strftime('%Y-%m-%d %H:%M:%S')
            
            elif format_type == 'date_only':
                return dt.strftime('%Y-%m-%d')
            
            elif format_type == 'time_only':
                return dt.strftime('%H:%M:%S')
            
            else:
                return dt.strftime(format_type)
                
        except Exception as e:
            raise ValueError(f"خطا در فرمت کردن زمان: {e}")
    
    def parse_datetime(self, date_string: str, 
                      format_string: str = None) -> datetime.datetime:
        """تجزیه رشته زمان"""
        try:
            if format_string:
                return datetime.datetime.strptime(date_string, format_string)
            else:
                # تلاش برای تشخیص خودکار فرمت
                formats = [
                    '%Y-%m-%d %H:%M:%S',
                    '%Y/%m/%d %H:%M:%S',
                    '%Y-%m-%d',
                    '%Y/%m/%d',
                    '%d/%m/%Y %H:%M:%S',
                    '%d-%m-%Y %H:%M:%S'
                ]
                
                for fmt in formats:
                    try:
                        return datetime.datetime.strptime(date_string, fmt)
                    except ValueError:
                        continue
                
                # اگر هیچ فرمت مناسب نبود، از pandas استفاده کن
                return pd.to_datetime(date_string)
                
        except Exception as e:
            raise ValueError(f"خطا در تجزیه زمان: {e}")
    
    def timeframe_to_seconds(self, timeframe: Union[TimeFrame, str]) -> int:
        """تبدیل تایم‌فریم به ثانیه"""
        try:
            if isinstance(timeframe, str):
                # تبدیل رشته به enum
                for tf in TimeFrame:
                    if tf.value == timeframe:
                        timeframe = tf
                        break
                else:
                    raise ValueError(f"تایم‌فریم نامعتبر: {timeframe}")
            
            return self.timeframe_seconds[timeframe]
            
        except Exception as e:
            raise ValueError(f"خطا در تبدیل تایم‌فریم: {e}")
    
    def timeframe_to_timedelta(self, timeframe: Union[TimeFrame, str]) -> datetime.timedelta:
        """تبدیل تایم‌فریم به timedelta"""
        seconds = self.timeframe_to_seconds(timeframe)
        return datetime.timedelta(seconds=seconds)
    
    def get_timeframe_start(self, dt: datetime.datetime, 
                           timeframe: Union[TimeFrame, str]) -> datetime.datetime:
        """محاسبه شروع تایم‌فریم"""
        try:
            if isinstance(timeframe, str):
                timeframe = TimeFrame(timeframe)
            
            # تنظیم بر اساس نوع تایم‌فریم
            if timeframe == TimeFrame.ONE_MINUTE:
                return dt.replace(second=0, microsecond=0)
            
            elif timeframe == TimeFrame.FIVE_MINUTES:
                minute = (dt.minute // 5) * 5
                return dt.replace(minute=minute, second=0, microsecond=0)
            
            elif timeframe == TimeFrame.FIFTEEN_MINUTES:
                minute = (dt.minute // 15) * 15
                return dt.replace(minute=minute, second=0, microsecond=0)
            
            elif timeframe == TimeFrame.THIRTY_MINUTES:
                minute = (dt.minute // 30) * 30
                return dt.replace(minute=minute, second=0, microsecond=0)
            
            elif timeframe == TimeFrame.ONE_HOUR:
                return dt.replace(minute=0, second=0, microsecond=0)
            
            elif timeframe == TimeFrame.FOUR_HOURS:
                hour = (dt.hour // 4) * 4
                return dt.replace(hour=hour, minute=0, second=0, microsecond=0)
            
            elif timeframe == TimeFrame.ONE_DAY:
                return dt.replace(hour=0, minute=0, second=0, microsecond=0)
            
            elif timeframe == TimeFrame.ONE_WEEK:
                days_since_monday = dt.weekday()
                start_of_week = dt - datetime.timedelta(days=days_since_monday)
                return start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
            
            else:
                # برای سایر تایم‌فریم‌ها
                seconds = self.timeframe_to_seconds(timeframe)
                timestamp = self.datetime_to_timestamp(dt)
                aligned_timestamp = (timestamp // seconds) * seconds
                return self.timestamp_to_datetime(aligned_timestamp)
                
        except Exception as e:
            raise ValueError(f"خطا در محاسبه شروع تایم‌فریم: {e}")
    
    def calculate_duration(self, start_time: datetime.datetime, 
                          end_time: datetime.datetime = None) -> dict:
        """محاسبه مدت زمان"""
        if end_time is None:
            end_time = self.now_utc()
        
        duration = end_time - start_time
        
        total_seconds = int(duration.total_seconds())
        days = duration.days
        hours = total_seconds // 3600 % 24
        minutes = total_seconds // 60 % 60
        seconds = total_seconds % 60
        
        return {
            'total_seconds': total_seconds,
            'days': days,
            'hours': hours,
            'minutes': minutes,
            'seconds': seconds,
            'formatted': f"{days}d {hours}h {minutes}m {seconds}s"
        }
    
    def is_market_open(self, current_time: datetime.datetime = None) -> bool:
        """بررسی باز بودن بازار (ارزهای دیجیتال همیشه باز است)"""
        # بازار ارزهای دیجیتال 24/7 است
        return True
    
    def get_next_timeframe(self, current_time: datetime.datetime, 
                          timeframe: Union[TimeFrame, str]) -> datetime.datetime:
        """محاسبه زمان تایم‌فریم بعدی"""
        try:
            current_start = self.get_timeframe_start(current_time, timeframe)
            delta = self.timeframe_to_timedelta(timeframe)
            return current_start + delta
            
        except Exception as e:
            raise ValueError(f"خطا در محاسبه تایم‌فریم بعدی: {e}")
    
    def time_until_next_timeframe(self, timeframe: Union[TimeFrame, str], 
                                 current_time: datetime.datetime = None) -> dict:
        """زمان باقی‌مانده تا تایم‌فریم بعدی"""
        if current_time is None:
            current_time = self.now_utc()
        
        next_timeframe = self.get_next_timeframe(current_time, timeframe)
        return self.calculate_duration(current_time, next_timeframe)
    
    def generate_time_range(self, start_time: datetime.datetime, 
                           end_time: datetime.datetime, 
                           timeframe: Union[TimeFrame, str]) -> List[datetime.datetime]:
        """تولید محدوده زمانی"""
        try:
            delta = self.timeframe_to_timedelta(timeframe)
            time_range = []
            
            current = self.get_timeframe_start(start_time, timeframe)
            
            while current <= end_time:
                time_range.append(current)
                current += delta
            
            return time_range
            
        except Exception as e:
            raise ValueError(f"خطا در تولید محدوده زمانی: {e}")
    
    def format_uptime(self, start_time: datetime.datetime) -> str:
        """فرمت کردن مدت فعالیت"""
        duration = self.calculate_duration(start_time)
        
        if duration['days'] > 0:
            return f"{duration['days']} روز، {duration['hours']} ساعت"
        elif duration['hours'] > 0:
            return f"{duration['hours']} ساعت، {duration['minutes']} دقیقه"
        else:
            return f"{duration['minutes']} دقیقه، {duration['seconds']} ثانیه"
    
    def convert_timezone(self, dt: datetime.datetime, 
                        from_tz: str, to_tz: str) -> datetime.datetime:
        """تبدیل منطقه زمانی"""
        try:
            from_timezone = pytz.timezone(from_tz)
            to_timezone = pytz.timezone(to_tz)
            
            # اگر timezone ندارد، اضافه کن
            if dt.tzinfo is None:
                dt = from_timezone.localize(dt)
            
            return dt.astimezone(to_timezone)
            
        except Exception as e:
            raise ValueError(f"خطا در تبدیل منطقه زمانی: {e}")
    
    def sleep_until_next_timeframe(self, timeframe: Union[TimeFrame, str]):
        """انتظار تا تایم‌فریم بعدی"""
        remaining = self.time_until_next_timeframe(timeframe)
        sleep_seconds = remaining['total_seconds']
        
        if sleep_seconds > 0:
            print(f"انتظار {sleep_seconds} ثانیه تا تایم‌فریم بعدی...")
            time.sleep(sleep_seconds)
    
    def get_trading_session_info(self, current_time: datetime.datetime = None) -> dict:
        """اطلاعات جلسه معاملاتی"""
        if current_time is None:
            current_time = self.now_utc()
        
        # در بازار ارزهای دیجیتال همیشه جلسه معاملاتی فعال است
        return {
            'is_active': True,
            'session_type': 'CRYPTO_24_7',
            'current_time_utc': current_time,
            'current_time_iranian': self.convert_timezone(current_time, 'UTC', 'Asia/Tehran'),
            'next_daily_close': current_time.replace(hour=0, minute=0, second=0, microsecond=0) + datetime.timedelta(days=1)
        }
