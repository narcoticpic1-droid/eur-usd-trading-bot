# connectors/forex_connector.py
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.accounts as accounts
import pandas as pd
import time
from typing import Optional, Dict, List
import datetime
from utils.logger import setup_logger
import config

class ForexConnector:
    """
    اتصال به OANDA API برای دریافت داده‌های Forex
    """
    
    def __init__(self):
        self.logger = setup_logger('forex_connector')
        self.client = None
        self.account_id = config.OANDA_ACCOUNT_ID
        self._initialize_connection()
        
    def _initialize_connection(self):
        """راه‌اندازی اتصال به OANDA"""
        try:
            self.client = oandapyV20.API(
                access_token=config.OANDA_API_KEY,
                environment=config.OANDA_ENVIRONMENT
            )
            self.logger.info(f"اتصال به OANDA ({config.OANDA_ENVIRONMENT}) برقرار شد")
            
            # تست اتصال
            if self.test_connection():
                self.logger.info("تست اتصال OANDA موفق")
            else:
                self.logger.error("تست اتصال OANDA ناموفق")
                
        except Exception as e:
            self.logger.error(f"خطا در اتصال به OANDA: {e}")
            raise
    
    def get_forex_data(self, pair: str, timeframe: str = 'H1', count: int = 200) -> Optional[pd.DataFrame]:
        """
        دریافت داده‌های قیمت Forex
        
        Args:
            pair: جفت ارز (مثل EUR_USD)
            timeframe: بازه زمانی (H1, H4, D)
            count: تعداد کندل‌ها
        """
        max_retries = config.BROKER_RETRIES
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"دریافت {count} کندل {pair} - {timeframe} (تلاش {attempt + 1})")
                
                params = {
                    "granularity": timeframe,
                    "count": count,
                    "price": "M"  # Mid prices
                }
                
                r = instruments.InstrumentsCandles(
                    instrument=pair,
                    params=params
                )
                
                response = self.client.request(r)
                
                if not response.get('candles'):
                    self.logger.warning(f"داده‌ای برای {pair} دریافت نشد")
                    if attempt < max_retries - 1:
                        time.sleep(config.BROKER_DELAY)
                        continue
                    return None
                
                # تبدیل به DataFrame
                data = []
                for candle in response['candles']:
                    if candle.get('complete', False):
                        mid = candle.get('mid', {})
                        data.append({
                            'timestamp': candle['time'],
                            'open': float(mid.get('o', 0)),
                            'high': float(mid.get('h', 0)),
                            'low': float(mid.get('l', 0)),
                            'close': float(mid.get('c', 0)),
                            'volume': int(candle.get('volume', 0))
                        })
                
                if not data:
                    self.logger.warning(f"هیچ کندل کاملی برای {pair} یافت نشد")
                    if attempt < max_retries - 1:
                        time.sleep(config.BROKER_DELAY)
                        continue
                    return None
                
                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                
                # اعتبارسنجی داده‌ها
                if self._validate_forex_data(df, pair):
                    self.logger.info(f"✅ {len(df)} کندل معتبر برای {pair} دریافت شد")
                    return df
                else:
                    if attempt < max_retries - 1:
                        time.sleep(config.BROKER_DELAY)
                        continue
                    return None
                    
            except Exception as e:
                self.logger.error(f"خطا در دریافت داده {pair}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(config.BROKER_DELAY * (attempt + 1))
                    continue
                    
        self.logger.error(f"تمام تلاش‌ها برای دریافت {pair} ناموفق")
        return None
    
    def _validate_forex_data(self, df: pd.DataFrame, pair: str) -> bool:
        """اعتبارسنجی داده‌های Forex"""
        try:
            # بررسی حداقل تعداد کندل
            if len(df) < config.MIN_CANDLES_REQUIRED:
                self.logger.warning(f"داده ناکافی برای {pair}: {len(df)} < {config.MIN_CANDLES_REQUIRED}")
                return False
            
            # بررسی null values
            if df.isnull().any().any():
                self.logger.warning(f"داده‌های ناقص برای {pair}")
                return False
            
            # بررسی منطقی بودن قیمت‌ها
            if (df['high'] < df['low']).any():
                self.logger.error(f"داده نامعتبر برای {pair}: high < low")
                return False
            
            if (df['high'] < df['open']).any() or (df['high'] < df['close']).any():
                self.logger.error(f"داده نامعتبر برای {pair}: high < open/close")
                return False
            
            if (df['low'] > df['open']).any() or (df['low'] > df['close']).any():
                self.logger.error(f"داده نامعتبر برای {pair}: low > open/close")
                return False
            
            # بررسی تغییرات غیرمعقول (بیش از 5% در یک کندل)
            price_changes = abs(df['close'].pct_change())
            if (price_changes > 0.05).any():
                self.logger.warning(f"تغییرات قیمت غیرمعقول برای {pair}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"خطا در اعتبارسنجی {pair}: {e}")
            return False
    
    def get_account_info(self) -> Dict:
        """دریافت اطلاعات حساب"""
        try:
            r = accounts.AccountDetails(accountID=self.account_id)
            response = self.client.request(r)
            
            account = response.get('account', {})
            return {
                'balance': float(account.get('balance', 0)),
                'nav': float(account.get('NAV', 0)),
                'unrealized_pl': float(account.get('unrealizedPL', 0)),
                'margin_used': float(account.get('marginUsed', 0)),
                'margin_available': float(account.get('marginAvailable', 0)),
                'currency': account.get('currency', 'USD')
            }
            
        except Exception as e:
            self.logger.error(f"خطا در دریافت اطلاعات حساب: {e}")
            return {}
    
    def get_current_price(self, pair: str) -> Optional[Dict]:
        """دریافت قیمت فعلی"""
        try:
            # دریافت آخرین کندل
            recent_data = self.get_forex_data(pair, 'M1', 1)
            if recent_data is not None and len(recent_data) > 0:
                latest = recent_data.iloc[-1]
                return {
                    'pair': pair,
                    'price': latest['close'],
                    'timestamp': recent_data.index[-1]
                }
        except Exception as e:
            self.logger.error(f"خطا در دریافت قیمت فعلی {pair}: {e}")
        
        return None
    
    def is_market_open(self) -> bool:
        """بررسی باز بودن بازار Forex"""
        try:
            now = datetime.datetime.utcnow()
            weekday = now.weekday()
            
            # بازار Forex از یکشنبه ۲۲:۰۰ تا جمعه ۲۲:۰۰ UTC باز است
            if weekday == 6:  # یکشنبه
                return now.hour >= 22
            elif weekday < 5:  # دوشنبه تا جمعه
                return True
            elif weekday == 4:  # جمعه
                return now.hour < 22
            else:  # شنبه
                return False
                
        except Exception as e:
            self.logger.error(f"خطا در بررسی وضعیت بازار: {e}")
            return False
    
    def test_connection(self) -> bool:
        """تست اتصال"""
        try:
            # تست دریافت اطلاعات حساب
            account_info = self.get_account_info()
            if account_info and 'balance' in account_info:
                self.logger.info("تست اتصال OANDA موفق")
                return True
            else:
                self.logger.error("تست اتصال OANDA ناموفق")
                return False
                
        except Exception as e:
            self.logger.error(f"خطا در تست اتصال: {e}")
            return False
    
    def get_supported_pairs(self) -> List[str]:
        """دریافت لیست جفت ارزهای پشتیبانی شده"""
        try:
            r = accounts.AccountInstruments(accountID=self.account_id)
            response = self.client.request(r)
            
            instruments = response.get('instruments', [])
            pairs = []
            
            for instrument in instruments:
                if instrument.get('type') == 'CURRENCY':
                    pairs.append(instrument.get('name'))
            
            return pairs
            
        except Exception as e:
            self.logger.error(f"خطا در دریافت جفت ارزها: {e}")
            return config.FOREX_PAIRS
