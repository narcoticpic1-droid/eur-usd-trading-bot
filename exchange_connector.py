"""
connectors/exchange_connector.py
کانکتور صرافی بهینه‌شده با قابلیت‌های پیشرفته
"""

import ccxt
import pandas as pd
import time
import asyncio
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime, timedelta
import logging
import threading

class ExchangeConnector:
    """
    کانکتور صرافی با قابلیت‌های حرفه‌ای:
    - پشتیبانی از چندین صرافی
    - Auto-failover
    - Data validation
    - Performance monitoring
    - Rate limiting
    """

    def __init__(self, exchange_id: str = 'bybit'):
        self.exchange_id = exchange_id
        self.exchange = None
        self.lock = threading.Lock()
        
        # صرافی‌های پشتیبانی شده (به ترتیب اولویت)
        self.supported_exchanges = [
            'bybit', 'kucoin', 'binance', 'okx',
            'gateio', 'mexc', 'huobi', 'bitget',
            'coinex', 'phemex', 'bingx'
        ]
        
        # آمار عملکرد
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'rate_limits_hit': 0,
            'data_validation_failures': 0,
            'exchange_switches': 0,
            'avg_response_time': 0.0,
            'last_successful_request': None,
            'last_error': None
        }
        
        # تنظیمات retry
        self.max_retries = 3
        self.retry_delays = [2, 5, 10]  # Exponential backoff
        self.rate_limit_delay = 60
        
        # Cache برای markets
        self.markets_cache = {}
        self.cache_expiry = None
        self.cache_duration = timedelta(hours=1)
        
        self._initialize_exchange()

    def _initialize_exchange(self) -> bool:
        """راه‌اندازی اتصال صرافی با Auto-failover"""
        
        # ترتیب تلاش: صرافی انتخابی + backup ها
        exchanges_to_try = [self.exchange_id] + [
            ex for ex in self.supported_exchanges 
            if ex != self.exchange_id
        ]

        for exchange_name in exchanges_to_try:
            try:
                print(f"🔄 تلاش اتصال به {exchange_name}...")
                
                # تنظیمات پایه
                config = {
                    'timeout': 30000,  # 30 seconds
                    'enableRateLimit': True,
                    'sandbox': False,
                    'options': {'adjustForTimeDifference': True}
                }
                
                # تنظیمات خاص هر صرافی
                config.update(self._get_exchange_config(exchange_name))
                
                # ایجاد instance
                exchange_class = getattr(ccxt, exchange_name)
                self.exchange = exchange_class(config)
                
                # تست اتصال و بارگذاری markets
                start_time = time.time()
                markets = self.exchange.load_markets()
                response_time = time.time() - start_time
                
                if self._validate_exchange_connection(markets, response_time):
                    print(f"✅ اتصال به {exchange_name} موفق ({len(markets)} بازار، {response_time:.2f}s)")
                    
                    # به‌روزرسانی cache
                    self.markets_cache = markets
                    self.cache_expiry = datetime.now() + self.cache_duration
                    
                    # تغییر exchange_id اگر failover صورت گرفت
                    if exchange_name != self.exchange_id:
                        print(f"🔄 Failover: {self.exchange_id} → {exchange_name}")
                        self.exchange_id = exchange_name
                        self.stats['exchange_switches'] += 1
                    
                    return True

            except Exception as e:
                print(f"❌ خطا در اتصال به {exchange_name}: {str(e)[:100]}...")
                continue

        raise Exception("❌ اتصال به هیچ کدام از صرافی‌ها موفق نبود")

    def _get_exchange_config(self, exchange_name: str) -> Dict:
        """دریافت تنظیمات خاص هر صرافی"""
        configs = {
            'bybit': {
                'options': {'defaultType': 'spot'}
            },
            'kucoin': {
                'password': '',
                'options': {'partner': 'ccxt'}
            },
            'binance': {
                'options': {
                    'defaultType': 'spot',
                    'adjustForTimeDifference': True
                }
            },
            'okx': {
                'password': '',
                'options': {'defaultType': 'spot'}
            },
            'gateio': {
                'options': {'defaultType': 'spot'}
            },
            'mexc': {
                'options': {'defaultType': 'spot'}
            },
            'huobi': {
                'options': {'defaultType': 'spot'}
            },
            'bitget': {
                'options': {'defaultType': 'spot'}
            },
            'coinex': {
                'options': {'defaultType': 'spot'}
            },
            'phemex': {
                'options': {'defaultType': 'spot'}
            },
            'bingx': {
                'options': {'defaultType': 'spot'}
            }
        }
        
        return configs.get(exchange_name, {})

    def _validate_exchange_connection(self, markets: Dict, response_time: float) -> bool:
        """اعتبارسنجی اتصال صرافی"""
        
        # بررسی تعداد markets
        if not markets or len(markets) < 50:
            return False
        
        # بررسی response time
        if response_time > 30:  # بیش از 30 ثانیه
            return False
        
        # بررسی وجود نمادهای اصلی
        required_symbols = ['BTC/USDT', 'ETH/USDT']
        for symbol in required_symbols:
            if symbol not in markets:
                return False
            
            market = markets[symbol]
            if not market.get('active', True):
                return False
        
        return True

    async def get_kline_async(self, symbol: str, timeframe: str = '1h', limit: int = 200) -> Optional[pd.DataFrame]:
        """نسخه async دریافت کندل"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.get_kline, symbol, timeframe, limit
        )

    def get_kline(self, symbol: str, timeframe: str = '1h', limit: int = 200) -> Optional[pd.DataFrame]:
        """
        دریافت داده‌های کندل با validation پیشرفته
        """
        with self.lock:
            return self._get_kline_internal(symbol, timeframe, limit)

    def _get_kline_internal(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        """Implementation اصلی دریافت کندل"""
        
        # به‌روزرسانی آمار
        self.stats['total_requests'] += 1
        start_time = time.time()
        
        try:
            # بررسی cache markets
            if not self._is_markets_cache_valid():
                self.exchange.load_markets()
                self.markets_cache = self.exchange.markets
                self.cache_expiry = datetime.now() + self.cache_duration
            
            # بررسی وجود نماد
            if symbol not in self.markets_cache:
                print(f"⚠️ نماد {symbol} در {self.exchange_id} موجود نیست")
                available_symbols = [s for s in self.markets_cache.keys() if 'USDT' in s][:5]
                print(f"نمادهای موجود (نمونه): {available_symbols}")
                self.stats['failed_requests'] += 1
                return None

            # دریافت داده‌ها با retry mechanism
            ohlcv_data = self._fetch_ohlcv_with_retry(symbol, timeframe, limit)
            
            if not ohlcv_data:
                self.stats['failed_requests'] += 1
                return None

            # تبدیل به DataFrame
            df = self._create_dataframe(ohlcv_data)
            
            # اعتبارسنجی داده‌ها
            if not self._validate_kline_data(df, symbol, limit):
                self.stats['data_validation_failures'] += 1
                self.stats['failed_requests'] += 1
                return None

            # به‌روزرسانی آمار موفقیت
            response_time = time.time() - start_time
            self._update_success_stats(response_time)
            
            print(f"✅ {len(df)} کندل معتبر {symbol} از {self.exchange_id} ({response_time:.2f}s)")
            return df

        except ccxt.RateLimitExceeded as e:
            self.stats['rate_limits_hit'] += 1
            print(f"⚡ Rate limit hit: {e}")
            return self._handle_rate_limit(symbol, timeframe, limit)
            
        except ccxt.NetworkError as e:
            print(f"🌐 Network error: {e}")
            return self._handle_network_error(symbol, timeframe, limit)
            
        except Exception as e:
            self.stats['failed_requests'] += 1
            self.stats['last_error'] = {'error': str(e), 'time': datetime.now()}
            print(f"❌ خطای غیرمنتظره: {e}")
            return None

    def _fetch_ohlcv_with_retry(self, symbol: str, timeframe: str, limit: int) -> Optional[List]:
        """دریافت OHLCV با retry mechanism"""
        
        for attempt in range(self.max_retries):
            try:
                print(f"📡 دریافت {limit} کندل {symbol} (تلاش {attempt + 1}/{self.max_retries})")
                
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=limit
                )
                
                if ohlcv and len(ohlcv) > 0:
                    return ohlcv
                    
            except ccxt.RateLimitExceeded:
                raise  # بازگردانی به caller برای handle کردن
                
            except ccxt.NetworkError as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delays[min(attempt, len(self.retry_delays) - 1)]
                    print(f"🌐 Network error, انتظار {delay}s: {str(e)[:50]}...")
                    time.sleep(delay)
                    continue
                raise
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delays[min(attempt, len(self.retry_delays) - 1)]
                    print(f"❌ خطا، انتظار {delay}s: {str(e)[:50]}...")
                    time.sleep(delay)
                    continue
                raise
        
        return None

    def _create_dataframe(self, ohlcv_data: List) -> pd.DataFrame:
        """ایجاد DataFrame از داده‌های OHLCV"""
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # تبدیل timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # تبدیل به float
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_columns] = df[numeric_columns].astype(float)
        
        # مرتب‌سازی بر اساس زمان
        df.sort_index(inplace=True)
        
        return df

    def _validate_kline_data(self, df: pd.DataFrame, symbol: str, expected_limit: int) -> bool:
        """اعتبارسنجی پیشرفته داده‌های کندل"""
        
        # بررسی حداقل داده
        min_required = min(150, expected_limit * 0.8)  # حداقل 80% از درخواست یا 150
        if len(df) < min_required:
            print(f"⚠️ داده ناکافی برای {symbol}: {len(df)} < {min_required}")
            return False
        
        # بررسی null values
        if df.isnull().any().any():
            print(f"⚠️ داده‌های ناقص در {symbol}")
            return False
        
        # بررسی منطق قیمت‌ها
        invalid_highs = (df['high'] < df['low']).sum()
        invalid_prices = (
            (df['high'] < df['open']) | 
            (df['high'] < df['close']) | 
            (df['low'] > df['open']) | 
            (df['low'] > df['close'])
        ).sum()
        
        if invalid_highs > 0 or invalid_prices > 0:
            print(f"⚠️ داده‌های نامعتبر در {symbol}: highs={invalid_highs}, prices={invalid_prices}")
            return False
        
        # بررسی حجم
        zero_volume_count = (df['volume'] <= 0).sum()
        if zero_volume_count > len(df) * 0.1:  # بیش از 10% حجم صفر
            print(f"⚠️ حجم نامعتبر در {symbol}: {zero_volume_count} کندل با حجم صفر")
            return False
        
        # بررسی تغییرات غیرمعقول (بیش از 50% در یک کندل)
        price_changes = abs(df['close'].pct_change())
        extreme_changes = (price_changes > 0.5).sum()
        if extreme_changes > 0:
            print(f"⚠️ تغییرات قیمت غیرمعقول در {symbol}: {extreme_changes} مورد")
            return False
        
        # بررسی تسلسل زمانی
        time_diffs = df.index.to_series().diff()
        expected_interval = pd.Timedelta(hours=1)  # فرض 1h timeframe
        irregular_intervals = (time_diffs > expected_interval * 2).sum()
        if irregular_intervals > len(df) * 0.05:  # بیش از 5% فواصل نامنظم
            print(f"⚠️ فواصل زمانی نامنظم در {symbol}: {irregular_intervals} مورد")
        
        return True

    def _is_markets_cache_valid(self) -> bool:
        """بررسی اعتبار cache markets"""
        return (
            self.markets_cache and 
            self.cache_expiry and 
            datetime.now() < self.cache_expiry
        )

    def _update_success_stats(self, response_time: float):
        """به‌روزرسانی آمار موفقیت"""
        self.stats['successful_requests'] += 1
        self.stats['last_successful_request'] = datetime.now()
        
        # محاسبه میانگین response time
        total_successful = self.stats['successful_requests']
        current_avg = self.stats['avg_response_time']
        self.stats['avg_response_time'] = (
            (current_avg * (total_successful - 1) + response_time) / total_successful
        )

    def _handle_rate_limit(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        """مدیریت rate limit"""
        print(f"⚡ انتظار {self.rate_limit_delay}s برای rate limit...")
        time.sleep(self.rate_limit_delay)
        
        # تلاش مجدد یک بار
        try:
            return self._get_kline_internal(symbol, timeframe, limit)
        except:
            return None

    def _handle_network_error(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        """مدیریت خطای شبکه با failover"""
        print("🔄 تلاش failover به صرافی دیگر...")
        
        current_exchange = self.exchange_id
        try:
            # تلاش failover
            self._initialize_exchange()
            
            if self.exchange_id != current_exchange:
                # اگر صرافی تغییر کرد، تلاش مجدد
                return self._get_kline_internal(symbol, timeframe, limit)
        except:
            pass
        
        return None

    def get_multiple_symbols_data(self, symbols: List[str], timeframe: str = '1h', limit: int = 200) -> Dict[str, Optional[pd.DataFrame]]:
        """دریافت داده‌های چندین نماد به صورت همزمان"""
        results = {}
        
        for symbol in symbols:
            print(f"📊 پردازش {symbol}...")
            results[symbol] = self.get_kline(symbol, timeframe, limit)
            
            # کمی انتظار برای جلوگیری از rate limit
            time.sleep(0.5)
        
        return results

    def test_connection(self) -> bool:
        """تست جامع اتصال"""
        try:
            # تست با BTC/USDT
            test_data = self.get_kline('BTC/USDT', '1h', 50)
            
            if test_data is not None and len(test_data) >= 40:
                print(f"✅ تست اتصال {self.exchange_id} موفق")
                return True
            else:
                print(f"❌ تست اتصال {self.exchange_id} ناموفق")
                return False
                
        except Exception as e:
            print(f"❌ خطا در تست اتصال: {e}")
            return False

    def get_exchange_info(self) -> Dict[str, Any]:
        """اطلاعات کامل صرافی و آمار عملکرد"""
        
        try:
            success_rate = (
                self.stats['successful_requests'] / self.stats['total_requests'] * 100
                if self.stats['total_requests'] > 0 else 0
            )
            
            info = {
                'exchange': self.exchange_id,
                'status': 'connected' if self.exchange else 'disconnected',
                'markets_count': len(self.markets_cache),
                'cache_valid': self._is_markets_cache_valid(),
                
                # آمار عملکرد
                'performance': {
                    'total_requests': self.stats['total_requests'],
                    'success_rate': f"{success_rate:.1f}%",
                    'avg_response_time': f"{self.stats['avg_response_time']:.2f}s",
                    'rate_limits_hit': self.stats['rate_limits_hit'],
                    'exchange_switches': self.stats['exchange_switches'],
                    'data_validation_failures': self.stats['data_validation_failures']
                },
                
                # آخرین فعالیت
                'last_activity': {
                    'last_successful_request': self.stats['last_successful_request'],
                    'last_error': self.stats['last_error']
                },
                
                # قابلیت‌ها
                'capabilities': {
                    'fetchOHLCV': self.exchange.has.get('fetchOHLCV', False) if self.exchange else False,
                    'timeframes': list(self.exchange.timeframes.keys()) if self.exchange and hasattr(self.exchange, 'timeframes') else []
                } if self.exchange else {},
                
                # نمادهای موجود
                'symbols_available': {
                    'btc_usdt': 'BTC/USDT' in self.markets_cache,
                    'eth_usdt': 'ETH/USDT' in self.markets_cache,
                    'sol_usdt': 'SOL/USDT' in self.markets_cache
                }
            }
            
            return info
            
        except Exception as e:
            return {
                'exchange': self.exchange_id,
                'status': 'error',
                'error': str(e)
            }

    def get_available_symbols(self, min_volume: float = 1000000) -> List[str]:
        """دریافت نمادهای پرحجم"""
        try:
            if not self._is_markets_cache_valid():
                self.exchange.load_markets()
                self.markets_cache = self.exchange.markets
            
            # فیلتر USDT pairs
            usdt_pairs = [
                symbol for symbol in self.markets_cache.keys() 
                if symbol.endswith('/USDT') and self.markets_cache[symbol].get('active', True)
            ]
            
            # اگر امکان دریافت volume وجود دارد، فیلتر کن
            popular_symbols = []
            for symbol in usdt_pairs[:30]:  # بررسی 30 نماد اول
                try:
                    ticker = self.exchange.fetch_ticker(symbol)
                    volume_24h = ticker.get('quoteVolume', 0)
                    
                    if volume_24h >= min_volume:
                        popular_symbols.append(symbol)
                        print(f"✅ {symbol}: ${volume_24h:,.0f} حجم 24h")
                        
                        if len(popular_symbols) >= 20:
                            break
                            
                except:
                    continue
                    
                time.sleep(0.1)  # جلوگیری از rate limit
            
            return popular_symbols if popular_symbols else usdt_pairs[:10]
            
        except Exception as e:
            print(f"❌ خطا در دریافت نمادها: {e}")
            return ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']  # fallback

    def switch_exchange(self, new_exchange_id: str) -> bool:
        """تغییر صرافی"""
        if new_exchange_id not in self.supported_exchanges:
            print(f"❌ صرافی {new_exchange_id} پشتیبانی نمی‌شود")
            return False
        
        try:
            old_exchange = self.exchange_id
            self.exchange_id = new_exchange_id
            self._initialize_exchange()
            
            if self.test_connection():
                print(f"✅ تغییر از {old_exchange} به {new_exchange_id} موفق")
                self.stats['exchange_switches'] += 1
                return True
            else:
                # بازگشت به صرافی قبلی
                self.exchange_id = old_exchange
                self._initialize_exchange()
                return False
                
        except Exception as e:
            print(f"❌ خطا در تغییر صرافی: {e}")
            return False

    def __del__(self):
        """Cleanup هنگام حذف object"""
        try:
            if hasattr(self, 'exchange') and self.exchange:
                self.exchange.close()
        except:
            pass
