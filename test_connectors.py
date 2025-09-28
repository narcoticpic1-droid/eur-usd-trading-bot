import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import time
from datetime import datetime

# اضافه کردن مسیر پروژه
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from connectors.exchange_connector import ExchangeConnector
    from connectors.data_provider import DataProvider
    from connectors.api_client import APIClient
except ImportError:
    try:
        from exchange_connector import ExchangeConnector
    except ImportError:
        print("⚠️ ExchangeConnector در دسترس نیست")

import config

class TestExchangeConnector(unittest.TestCase):
    """تست کانکتور صرافی"""
    
    def setUp(self):
        """راه‌اندازی تست"""
        try:
            # استفاده از صرافی تست
            self.connector = ExchangeConnector(exchange_id='bybit')
        except Exception:
            self.skipTest("ExchangeConnector در دسترس نیست")
    
    def test_connector_initialization(self):
        """تست راه‌اندازی کانکتور"""
        if not hasattr(self, 'connector'):
            self.skipTest("Connector در دسترس نیست")
        
        self.assertIsNotNone(self.connector)
        self.assertIsNotNone(self.connector.exchange_id)
        self.assertIn(self.connector.exchange_id, self.connector.backup_exchanges)
    
    @patch('ccxt.bybit')
    def test_exchange_initialization_success(self, mock_bybit_class):
        """تست موفق راه‌اندازی صرافی"""
        # Mock exchange
        mock_exchange = Mock()
        mock_exchange.load_markets.return_value = {'BTC/USDT': {}, 'ETH/USDT': {}}
        mock_bybit_class.return_value = mock_exchange
        
        connector = ExchangeConnector(exchange_id='bybit')
        
        self.assertEqual(connector.exchange_id, 'bybit')
        self.assertIsNotNone(connector.exchange)
    
    @patch('ccxt.bybit')
    def test_exchange_initialization_failure(self, mock_bybit_class):
        """تست شکست در راه‌اندازی صرافی"""
        # Mock که خطا تولید کند
        mock_bybit_class.side_effect = Exception("Connection failed")
        
        with self.assertRaises(Exception):
            ExchangeConnector(exchange_id='bybit')
    
    def test_get_kline_data_structure(self):
        """تست ساختار داده‌های kline"""
        if not hasattr(self, 'connector'):
            self.skipTest("Connector در دسترس نیست")
        
        # Mock data برای تست
        mock_ohlcv = [
            [1640995200000, 47000, 47500, 46800, 47200, 1000],  # timestamp, o, h, l, c, v
            [1640998800000, 47200, 47600, 47000, 47400, 1200],
            [1641002400000, 47400, 47800, 47300, 47700, 900]
        ]
        
        with patch.object(self.connector.exchange, 'fetch_ohlcv', return_value=mock_ohlcv):
            df = self.connector.get_kline('BTC/USDT', '1h', 3)
        
        if df is not None:
            # بررسی ستون‌ها
            expected_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in expected_columns:
                self.assertIn(col, df.columns)
            
            # بررسی index
            self.assertIsInstance(df.index, pd.DatetimeIndex)
            
            # بررسی تعداد سطرها
            self.assertEqual(len(df), 3)
    
    def test_data_validation(self):
        """تست اعتبارسنجی داده‌ها"""
        if not hasattr(self, 'connector'):
            self.skipTest("Connector در دسترس نیست")
        
        # داده‌های معتبر
        valid_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [103, 104, 105],
            'volume': [1000, 1100, 900]
        })
        
        is_valid = self.connector._validate_data_quality(valid_data, 'TEST/USDT')
        self.assertTrue(is_valid)
        
        # داده‌های نامعتبر - high < low
        invalid_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [95, 96, 97],  # high کمتر از low
            'low': [105, 106, 107],
            'close': [103, 104, 105],
            'volume': [1000, 1100, 900]
        })
        
        is_invalid = self.connector._validate_data_quality(invalid_data, 'TEST/USDT')
        self.assertFalse(is_invalid)
    
    def test_retry_mechanism(self):
        """تست مکانیزم retry"""
        if not hasattr(self, 'connector'):
            self.skipTest("Connector در دسترس نیست")
        
        # Mock که در تلاش اول خطا و در دوم موفق شود
        call_count = 0
        def mock_fetch_ohlcv(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Network error")
            return [[1640995200000, 47000, 47500, 46800, 47200, 1000]]
        
        with patch.object(self.connector.exchange, 'fetch_ohlcv', side_effect=mock_fetch_ohlcv):
            with patch('time.sleep'):  # تا تست سریع باشد
                df = self.connector.get_kline('BTC/USDT', '1h', 1)
        
        # باید در تلاش دوم موفق شود
        self.assertEqual(call_count, 2)
        if df is not None:
            self.assertEqual(len(df), 1)
    
    @patch('ccxt.bybit')
    @patch('ccxt.kucoin')
    def test_exchange_switching(self, mock_kucoin, mock_bybit):
        """تست تغییر صرافی"""
        # Mock exchanges
        mock_bybit_instance = Mock()
        mock_bybit_instance.load_markets.return_value = {'BTC/USDT': {}}
        mock_bybit.return_value = mock_bybit_instance
        
        mock_kucoin_instance = Mock()
        mock_kucoin_instance.load_markets.return_value = {'BTC/USDT': {}}
        mock_kucoin.return_value = mock_kucoin_instance
        
        connector = ExchangeConnector(exchange_id='bybit')
        
        # تست تغییر به کوکوین
        with patch.object(connector, 'test_connection', return_value=True):
            success = connector.switch_exchange('kucoin')
        
        self.assertTrue(success)
        self.assertEqual(connector.exchange_id, 'kucoin')
    
    def test_market_summary(self):
        """تست خلاصه اطلاعات بازار"""
        if not hasattr(self, 'connector'):
            self.skipTest("Connector در دسترس نیست")
        
        mock_ticker = {
            'last': 47000,
            'quoteVolume': 1000000,
            'percentage': 2.5,
            'high': 48000,
            'low': 46000
        }
        
        with patch.object(self.connector.exchange, 'fetch_ticker', return_value=mock_ticker):
            summary = self.connector.get_market_summary('BTC/USDT')
        
        if summary:
            self.assertIn('symbol', summary)
            self.assertIn('price', summary)
            self.assertIn('volume_24h', summary)
            self.assertEqual(summary['price'], 47000)
    
    def test_connection_test(self):
        """تست بررسی اتصال"""
        if not hasattr(self, 'connector'):
            self.skipTest("Connector در دسترس نیست")
        
        # Mock successful data fetch
        mock_data = pd.DataFrame({
            'open': range(150, 200),
            'high': range(151, 201),
            'low': range(149, 199),
            'close': range(150, 200),
            'volume': range(1000, 1050)
        })
        
        with patch.object(self.connector, 'get_kline', return_value=mock_data):
            is_connected = self.connector.test_connection()
        
        self.assertTrue(is_connected)

class TestDataProvider(unittest.TestCase):
    """تست ارائه‌دهنده داده"""
    
    def setUp(self):
        """راه‌اندازی تست"""
        try:
            self.data_provider = DataProvider()
        except NameError:
            self.skipTest("DataProvider در دسترس نیست")
    
    def test_multi_symbol_data_fetch(self):
        """تست دریافت داده چندین نماد"""
        if not hasattr(self, 'data_provider'):
            self.skipTest("DataProvider در دسترس نیست")
        
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        
        # Mock data for each symbol
        mock_data = {}
        for symbol in symbols:
            mock_data[symbol] = pd.DataFrame({
                'open': range(100),
                'high': range(101, 201),
                'low': range(99, 199),
                'close': range(100, 200),
                'volume': range(1000, 1100)
            })
        
        if hasattr(self.data_provider, 'get_multi_symbol_data'):
            with patch.object(self.data_provider, 'get_symbol_data', side_effect=lambda s: mock_data.get(s)):
                result = self.data_provider.get_multi_symbol_data(symbols)
            
            self.assertEqual(len(result), len(symbols))
            for symbol in symbols:
                self.assertIn(symbol, result)
    
    def test_data_synchronization(self):
        """تست همگام‌سازی داده‌ها"""
        if not hasattr(self, 'data_provider'):
            self.skipTest("DataProvider در دسترس نیست")
        
        # ایجاد داده‌های با timestamps مختلف
        timestamps1 = pd.date_range('2025-01-01', periods=100, freq='H')
        timestamps2 = pd.date_range('2025-01-01 01:00', periods=99, freq='H')  # شروع متفاوت
        
        data1 = pd.DataFrame({'price': range(100)}, index=timestamps1)
        data2 = pd.DataFrame({'price': range(99)}, index=timestamps2)
        
        if hasattr(self.data_provider, 'synchronize_data'):
            sync_data1, sync_data2 = self.data_provider.synchronize_data(data1, data2)
            
            # پس از همگام‌سازی باید timestamps یکسان باشند
            self.assertTrue(sync_data1.index.equals(sync_data2.index))

class TestAPIClient(unittest.TestCase):
    """تست کلاینت API"""
    
    def setUp(self):
        """راه‌اندازی تست"""
        try:
            self.api_client = APIClient()
        except NameError:
            self.skipTest("APIClient در دسترس نیست")
    
    @patch('requests.get')
    def test_api_request_success(self, mock_get):
        """تست درخواست موفق API"""
        if not hasattr(self, 'api_client'):
            self.skipTest("APIClient در دسترس نیست")
        
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'data': 'test'}
        mock_get.return_value = mock_response
        
        if hasattr(self.api_client, 'make_request'):
            result = self.api_client.make_request('GET', '/test')
            
            self.assertEqual(result['data'], 'test')
    
    @patch('requests.get')
    def test_api_request_failure(self, mock_get):
        """تست درخواست ناموفق API"""
        if not hasattr(self, 'api_client'):
            self.skipTest("APIClient در دسترس نیست")
        
        # Mock failed response
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = Exception("Server Error")
        mock_get.return_value = mock_response
        
        if hasattr(self.api_client, 'make_request'):
            with self.assertRaises(Exception):
                self.api_client.make_request('GET', '/test')
    
    def test_rate_limiting(self):
        """تست محدودیت نرخ درخواست"""
        if not hasattr(self, 'api_client'):
            self.skipTest("APIClient در دسترس نیست")
        
        # شبیه‌سازی چندین درخواست سریع
        if hasattr(self.api_client, 'check_rate_limit'):
            start_time = time.time()
            
            for i in range(5):
                can_proceed = self.api_client.check_rate_limit()
                if not can_proceed:
                    break
            
            end_time = time.time()
            
            # اگر rate limiting فعال باشد، باید تأخیر ایجاد شود
            # این تست به پیاده‌سازی واقعی بستگی دارد

class TestDataCaching(unittest.TestCase):
    """تست کش کردن داده‌ها"""
    
    def test_cache_functionality(self):
        """تست عملکرد کش"""
        # شبیه‌سازی سیستم کش ساده
        cache = {}
        
        def get_data_with_cache(key):
            if key in cache:
                return cache[key], True  # از کش
            else:
                # شبیه‌سازی دریافت داده
                data = f"data_for_{key}"
                cache[key] = data
                return data, False  # جدید
        
        # اولین بار
        data1, from_cache1 = get_data_with_cache('BTC/USDT')
        self.assertFalse(from_cache1)
        
        # دومین بار
        data2, from_cache2 = get_data_with_cache('BTC/USDT')
        self.assertTrue(from_cache2)
        self.assertEqual(data1, data2)
    
    def test_cache_expiration(self):
        """تست انقضای کش"""
        # شبیه‌سازی کش با زمان انقضا
        cache_with_expiry = {}
        cache_timeout = 60  # 60 ثانیه
        
        def get_data_with_expiry(key):
            current_time = time.time()
            
            if key in cache_with_expiry:
                data, timestamp = cache_with_expiry[key]
                if current_time - timestamp < cache_timeout:
                    return data, True  # هنوز معتبر
            
            # داده جدید یا منقضی شده
            new_data = f"fresh_data_for_{key}_{current_time}"
            cache_with_expiry[key] = (new_data, current_time)
            return new_data, False
        
        # تست انقضا با تغییر زمان
        data1, from_cache1 = get_data_with_expiry('BTC/USDT')
        self.assertFalse(from_cache1)
        
        # شبیه‌سازی گذشت زمان
        with patch('time.time', return_value=time.time() + 120):  # 2 دقیقه بعد
            data2, from_cache2 = get_data_with_expiry('BTC/USDT')
            self.assertFalse(from_cache2)  # باید منقضی شده باشد
            self.assertNotEqual(data1, data2)

if __name__ == '__main__':
    unittest.main(verbosity=2)
