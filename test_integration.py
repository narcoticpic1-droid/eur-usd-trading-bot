import unittest
import asyncio
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import sys
import os
from datetime import datetime, timedelta
import json
import sqlite3

# اضافه کردن مسیر پروژه
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
try:
    from main import MultiCryptoPriceActionBot
    from pure_price_action_analyzer import PurePriceActionAnalyzer
    from multi_ai_evaluator import MultiAIEvaluator
    from telegram_notifier import TelegramNotifier
    from exchange_connector import ExchangeConnector
    from learning_system import AdaptiveLearningSystem
except ImportError as e:
    print(f"⚠️ خطا در import: {e}")

import config

class TestFullSystemIntegration(unittest.TestCase):
    """تست یکپارچگی کامل سیستم"""
    
    def setUp(self):
        """راه‌اندازی تست یکپارچگی"""
        try:
            self.bot = MultiCryptoPriceActionBot()
        except Exception:
            self.skipTest("MultiCryptoPriceActionBot در دسترس نیست")
        
        # Mock کردن اجزای خارجی
        self.setup_mocks()
    
    def setup_mocks(self):
        """راه‌اندازی mock ها"""
        # Mock Exchange Connector
        self.mock_exchange = Mock(spec=ExchangeConnector)
        self.mock_exchange.test_connection.return_value = True
        self.mock_exchange.get_kline.return_value = self._create_sample_kline_data()
        
        # Mock Price Action Analyzer
        self.mock_analyzer = Mock(spec=PurePriceActionAnalyzer)
        self.mock_analyzer.analyze.return_value = self._create_sample_analysis()
        
        # Mock Multi AI Evaluator
        self.mock_ai_evaluator = Mock(spec=MultiAIEvaluator)
        self.mock_ai_evaluator.evaluate_signal = AsyncMock(return_value=self._create_sample_ai_evaluation())
        
        # Mock Telegram Notifier
        self.mock_notifier = Mock(spec=TelegramNotifier)
        self.mock_notifier.test_connection = AsyncMock(return_value=True)
        self.mock_notifier.send_multi_crypto_signal = AsyncMock(return_value=True)
    
    def _create_sample_kline_data(self):
        """ایجاد داده‌های کندل نمونه"""
        dates = pd.date_range(start='2025-01-01', periods=200, freq='H')
        
        # قیمت‌های شبیه‌سازی شده
        np.random.seed(42)
        prices = np.cumsum(np.random.normal(0.001, 0.02, 200)) + 50000
        
        data = []
        for i, price in enumerate(prices):
            high = price * np.random.uniform(1.001, 1.02)
            low = price * np.random.uniform(0.98, 0.999)
            open_price = prices[i-1] if i > 0 else price
            volume = np.random.uniform(100, 1000)
            
            data.append({
                'timestamp': dates[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def _create_sample_analysis(self):
        """ایجاد نتیجه تحلیل نمونه"""
        return {
            'symbol': 'BTC/USDT',
            'timestamp': datetime.now(),
            'current_price': 50000,
            'signal': 1,  # سیگنال خرید
            'signal_type': 'BREAKOUT',
            'signal_quality': 'HIGH',
            'confidence': 0.85,
            'reasoning': ['شکست سطح مقاومت', 'حجم بالا', 'کندل قوی'],
            'market_context': {
                'structure': 'STRONG_UPTREND',
                'trend_strength': 'STRONG',
                'volume_context': 'HIGH'
            },
            'stop_loss': 49000,
            'take_profits': [51000, 52000, 53000],
            'risk_reward_ratio': 3.0
        }
    
    def _create_sample_ai_evaluation(self):
        """ایجاد ارزیابی AI نمونه"""
        return {
            'symbol': 'BTC/USDT',
            'ai_evaluations': {
                'gemini': {
                    'recommendation': 'BUY',
                    'confidence': 0.8,
                    'quality_score': 9
                },
                'openai': {
                    'recommendation': 'BUY',
                    'confidence': 0.85,
                    'quality_score': 8.5
                }
            },
            'consensus': {
                'type': 'UNANIMOUS',
                'strength': 1.0,
                'avg_confidence': 0.825
            },
            'final_decision': {
                'action': 'BUY',
                'confidence': 0.9,
                'trade_worthy': True
            },
            'recommendation': 'BUY'
        }
    
    @patch('main.ExchangeConnector')
    @patch('main.PurePriceActionAnalyzer')
    @patch('main.MultiAIEvaluator')
    @patch('main.TelegramNotifier')
    async def test_full_bot_initialization(self, mock_notifier_class, mock_ai_class, 
                                         mock_analyzer_class, mock_exchange_class):
        """تست راه‌اندازی کامل ربات"""
        # Setup mocks
        mock_exchange_class.return_value = self.mock_exchange
        mock_analyzer_class.return_value = self.mock_analyzer
        mock_ai_class.return_value = self.mock_ai_evaluator
        mock_notifier_class.return_value = self.mock_notifier
        
        # اجرای initialization
        success = await self.bot.initialize()
        
        self.assertTrue(success)
        self.assertIsNotNone(self.bot.exchange)
        self.assertIsNotNone(self.bot.price_action_analyzer)
        self.assertIsNotNone(self.bot.multi_ai_evaluator)
        self.assertIsNotNone(self.bot.notifier)
    
    async def test_single_symbol_analysis_flow(self):
        """تست جریان تحلیل یک نماد"""
        # Setup
        self.bot.exchange = self.mock_exchange
        self.bot.price_action_analyzer = self.mock_analyzer
        
        # اجرای تحلیل
        result = await self.bot.analyze_symbol('BTC/USDT')
        
        # بررسی نتیجه
        self.assertIsNotNone(result)
        self.assertEqual(result['symbol'], 'BTC/USDT')
        self.assertIn('signal', result)
        self.assertIn('confidence', result)
        
        # بررسی فراخوانی متدها
        self.mock_exchange.get_kline.assert_called_once()
        self.mock_analyzer.analyze.assert_called_once()
    
    async def test_multi_symbol_analysis_flow(self):
        """تست جریان تحلیل چندین نماد"""
        # Setup
        self.bot.exchange = self.mock_exchange
        self.bot.price_action_analyzer = self.mock_analyzer
        
        # Mock برای هر نماد
        symbols = config.SYMBOLS_TO_MONITOR
        
        # اجرای تحلیل همزمان
        results = await self.bot.analyze_all_symbols()
        
        # بررسی نتایج
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), len(symbols))
        
        for symbol in symbols:
            self.assertIn(symbol, results)
    
    async def test_ai_evaluation_integration(self):
        """تست یکپارچگی ارزیابی AI"""
        # Setup
        self.bot.multi_ai_evaluator = self.mock_ai_evaluator
        
        analysis_results = {
            'BTC/USDT': self._create_sample_analysis(),
            'ETH/USDT': None,  # فرض: سیگنال ضعیف
            'SOL/USDT': self._create_sample_analysis()
        }
        
        # اجرای ارزیابی AI
        evaluated_results = await self.bot.evaluate_with_multi_ai(analysis_results)
        
        # بررسی نتایج
        self.assertIsInstance(evaluated_results, dict)
        
        # BTC و SOL باید ارزیابی شده باشند
        self.assertIsNotNone(evaluated_results['BTC/USDT'])
        self.assertIsNone(evaluated_results['ETH/USDT'])
        self.assertIsNotNone(evaluated_results['SOL/USDT'])
    
    async def test_signal_processing_workflow(self):
        """تست جریان کار پردازش سیگنال"""
        # Setup
        self.bot.notifier = self.mock_notifier
        
        evaluated_results = {
            'BTC/USDT': {
                'original_analysis': self._create_sample_analysis(),
                'ai_evaluation': self._create_sample_ai_evaluation(),
                'final_recommendation': 'BUY',
                'trade_worthy': True
            }
        }
        
        # اجرای پردازش سیگنال
        signals_sent = await self.bot.process_trading_signals(evaluated_results)
        
        # بررسی نتایج
        self.assertIsInstance(signals_sent, list)
        
        if signals_sent:
            signal = signals_sent[0]
            self.assertEqual(signal['symbol'], 'BTC/USDT')
            self.assertEqual(signal['position'], 'LONG')
            self.assertIn('leverage', signal)
            self.assertIn('entry_price', signal)
    
    def test_portfolio_risk_management(self):
        """تست مدیریت ریسک پورتفولیو"""
        # شبیه‌سازی چندین سیگنال همزمان
        potential_signals = {
            'BTC/USDT': {'trade_worthy': True},
            'ETH/USDT': {'trade_worthy': True},
            'SOL/USDT': {'trade_worthy': True}
        }
        
        # بررسی ریسک
        risk_assessment = self.bot.check_portfolio_risk(potential_signals)
        
        # بررسی نتیجه
        self.assertIsInstance(risk_assessment, dict)
        self.assertIn('safe_to_trade', risk_assessment)
        self.assertIn('warnings', risk_assessment)
        self.assertIn('correlation_risk', risk_assessment)
        
        # اگر هر 3 نماد سیگنال دهند، ریسک همبستگی باید بالا باشد
        if len(potential_signals) >= 3:
            self.assertEqual(risk_assessment['correlation_risk'], 'HIGH')
    
    async def test_learning_system_integration(self):
        """تست یکپارچگی سیستم یادگیری"""
        if not self.bot.learning_enabled:
            self.skipTest("سیستم یادگیری فعال نیست")
        
        # Mock learning system
        mock_learning = Mock(spec=AdaptiveLearningSystem)
        mock_learning.predict_signal_success.return_value = 0.8
        mock_learning.get_adaptive_parameters.return_value = {
            'confidence_threshold': 0.7,
            'position_size_multiplier': 1.0
        }
        
        self.bot.learning_system = mock_learning
        
        analysis = self._create_sample_analysis()
        
        # شبیه‌سازی محاسبه پوزیشن با یادگیری
        position_data = self.bot.calculate_adaptive_position_size('BTC/USDT', analysis, {
            'confidence_threshold': 0.7,
            'position_size_multiplier': 1.0
        })
        
        self.assertIsInstance(position_data, dict)
        self.assertIn('leverage', position_data)

class TestDatabaseIntegration(unittest.TestCase):
    """تست یکپارچگی پایگاه داده"""
    
    def setUp(self):
        """راه‌اندازی تست دیتابیس"""
        self.test_db = ":memory:"  # SQLite در حافظه
        self.setup_test_database()
    
    def setup_test_database(self):
        """راه‌اندازی دیتابیس تست"""
        conn = sqlite3.connect(self.test_db)
        cursor = conn.cursor()
        
        # ایجاد جداول تست
        cursor.execute('''
            CREATE TABLE test_signals (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                signal_type TEXT,
                confidence REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE test_performance (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                entry_price REAL,
                exit_price REAL,
                pnl REAL,
                entry_time DATETIME,
                exit_time DATETIME
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def test_signal_storage(self):
        """تست ذخیره سیگنال در دیتابیس"""
        conn = sqlite3.connect(self.test_db)
        cursor = conn.cursor()
        
        # درج سیگنال
        cursor.execute('''
            INSERT INTO test_signals (symbol, signal_type, confidence)
            VALUES (?, ?, ?)
        ''', ('BTC/USDT', 'BREAKOUT', 0.85))
        
        conn.commit()
        
        # بازیابی و بررسی
        cursor.execute('SELECT * FROM test_signals WHERE symbol = ?', ('BTC/USDT',))
        result = cursor.fetchone()
        
        self.assertIsNotNone(result)
        self.assertEqual(result[1], 'BTC/USDT')  # symbol
        self.assertEqual(result[2], 'BREAKOUT')  # signal_type
        self.assertEqual(result[3], 0.85)        # confidence
        
        conn.close()
    
    def test_performance_tracking(self):
        """تست ردیابی عملکرد"""
        conn = sqlite3.connect(self.test_db)
        cursor = conn.cursor()
        
        # درج چندین معامله
        trades = [
            ('BTC/USDT', 50000, 51000, 1000, '2025-01-01 10:00:00', '2025-01-01 14:00:00'),
            ('BTC/USDT', 51000, 50500, -500, '2025-01-01 15:00:00', '2025-01-01 18:00:00'),
            ('ETH/USDT', 3000, 3150, 150, '2025-01-01 11:00:00', '2025-01-01 16:00:00')
        ]
        
        for trade in trades:
            cursor.execute('''
                INSERT INTO test_performance 
                (symbol, entry_price, exit_price, pnl, entry_time, exit_time)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', trade)
        
        conn.commit()
        
        # محاسبه آمار عملکرد
        cursor.execute('''
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                SUM(pnl) as total_pnl,
                AVG(pnl) as avg_pnl
            FROM test_performance
        ''')
        
        stats = cursor.fetchone()
        
        self.assertEqual(stats[0], 3)     # total_trades
        self.assertEqual(stats[1], 2)     # winning_trades
        self.assertEqual(stats[2], 650)   # total_pnl
        self.assertAlmostEqual(stats[3], 216.67, places=1)  # avg_pnl
        
        conn.close()

class TestErrorHandlingIntegration(unittest.TestCase):
    """تست مدیریت خطا در سطح سیستم"""
    
    def setUp(self):
        """راه‌اندازی تست مدیریت خطا"""
        try:
            self.bot = MultiCryptoPriceActionBot()
        except Exception:
            self.skipTest("Bot در دسترس نیست")
    
    async def test_exchange_connection_failure(self):
        """تست شکست اتصال صرافی"""
        # Mock exchange که خطا تولید کند
        mock_exchange = Mock()
        mock_exchange.test_connection.return_value = False
        mock_exchange.get_kline.side_effect = Exception("Network error")
        
        self.bot.exchange = mock_exchange
        
        # تست تحلیل با خطای شبکه
        result = await self.bot.analyze_symbol('BTC/USDT')
        
        # باید None برگردد و خطا handle شود
        self.assertIsNone(result)
    
    async def test_ai_evaluation_failure(self):
        """تست شکست ارزیابی AI"""
        # Mock AI evaluator که خطا تولید کند
        mock_ai = Mock()
        mock_ai.evaluate_signal = AsyncMock(side_effect=Exception("API error"))
        
        self.bot.multi_ai_evaluator = mock_ai
        
        analysis_results = {
            'BTC/USDT': {'signal': 1, 'confidence': 0.8}
        }
        
        # تست ارزیابی با خطای API
        evaluated_results = await self.bot.evaluate_with_multi_ai(analysis_results)
        
        # باید gracefully handle شود
        self.assertIsInstance(evaluated_results, dict)
        self.assertIsNone(evaluated_results['BTC/USDT'])
    
    async def test_notification_failure(self):
        """تست شکست ارسال اطلاع‌رسانی"""
        # Mock notifier که خطا تولید کند
        mock_notifier = Mock()
        mock_notifier.send_multi_crypto_signal = AsyncMock(return_value=False)
        
        self.bot.notifier = mock_notifier
        
        evaluated_results = {
            'BTC/USDT': {
                'original_analysis': {'signal': 1, 'confidence': 0.8},
                'trade_worthy': True
            }
        }
        
        # تست پردازش با شکست ارسال
        signals_sent = await self.bot.process_trading_signals(evaluated_results)
        
        # نباید سیگنال ارسال شود
        self.assertEqual(len(signals_sent), 0)

class TestCircuitBreakerIntegration(unittest.TestCase):
    """تست یکپارچگی Circuit Breaker"""
    
    def setUp(self):
        """راه‌اندازی تست Circuit Breaker"""
        try:
            self.bot = MultiCryptoPriceActionBot()
        except Exception:
            self.skipTest("Bot در دسترس نیست")
    
    def test_daily_loss_circuit_breaker(self):
        """تست Circuit Breaker برای ضرر روزانه"""
        # شبیه‌سازی ضرر بالا
        self.bot.circuit_breaker['daily_losses'] = 10  # بیش از حد مجاز
        
        # بررسی فعال بودن circuit breaker
        should_stop = self.bot.circuit_breaker['daily_losses'] >= config.CIRCUIT_BREAKERS.get('consecutive_loss_limit', 3)
        
        self.assertTrue(should_stop)
    
    def test_correlation_circuit_breaker(self):
        """تست Circuit Breaker برای همبستگی بالا"""
        # شبیه‌سازی همبستگی بالا بین نمادها
        potential_signals = {
            'BTC/USDT': {'trade_worthy': True},
            'ETH/USDT': {'trade_worthy': True},
            'SOL/USDT': {'trade_worthy': True}
        }
        
        # بررسی ریسک همبستگی
        risk_assessment = self.bot.check_portfolio_risk(potential_signals)
        
        # باید ریسک همبستگی بالا تشخیص دهد
        self.assertEqual(risk_assessment['correlation_risk'], 'HIGH')

class TestPerformanceMetrics(unittest.TestCase):
    """تست معیارهای عملکرد"""
    
    def test_win_rate_calculation(self):
        """تست محاسبه نرخ برد"""
        trades = [
            {'result': 'win', 'pnl': 100},
            {'result': 'loss', 'pnl': -50},
            {'result': 'win', 'pnl': 75},
            {'result': 'loss', 'pnl': -25},
            {'result': 'win', 'pnl': 150}
        ]
        
        wins = sum(1 for t in trades if t['result'] == 'win')
        total = len(trades)
        win_rate = wins / total
        
        self.assertEqual(win_rate, 0.6)  # 60%
    
    def test_profit_factor_calculation(self):
        """تست محاسبه Profit Factor"""
        pnls = [100, -50, 75, -25, 150, -30]
        
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        expected_pf = 325 / 105  # ~3.095
        self.assertAlmostEqual(profit_factor, expected_pf, places=2)
    
    def test_sharpe_ratio_calculation(self):
        """تست محاسبه نسبت شارپ"""
        # فرض: بازده‌های روزانه
        daily_returns = np.array([0.01, -0.005, 0.02, 0.015, -0.01, 0.008, 0.012])
        
        mean_return = np.mean(daily_returns)
        std_return = np.std(daily_returns, ddof=1)
        risk_free_rate = 0.0001  # نرخ بدون ریسک روزانه
        
        sharpe_ratio = (mean_return - risk_free_rate) / std_return
        
        self.assertIsInstance(sharpe_ratio, float)
        self.assertGreater(sharpe_ratio, -2)  # حد منطقی
        self.assertLess(sharpe_ratio, 5)       # حد منطقی

class TestEndToEndWorkflow(unittest.TestCase):
    """تست جریان کاری End-to-End"""
    
    def setUp(self):
        """راه‌اندازی تست E2E"""
        try:
            self.bot = MultiCryptoPriceActionBot()
        except Exception:
            self.skipTest("Bot در دسترس نیست")
        
        self.setup_comprehensive_mocks()
    
    def setup_comprehensive_mocks(self):
        """راه‌اندازی جامع mock ها"""
        # Mock all components
        self.bot.exchange = Mock()
        self.bot.exchange.test_connection.return_value = True
        self.bot.exchange.get_kline.return_value = self._create_realistic_data()
        
        self.bot.price_action_analyzer = Mock()
        self.bot.price_action_analyzer.analyze.return_value = self._create_strong_signal()
        
        self.bot.multi_ai_evaluator = Mock()
        self.bot.multi_ai_evaluator.evaluate_signal = AsyncMock(return_value=self._create_positive_ai_eval())
        
        self.bot.notifier = Mock()
        self.bot.notifier.test_connection = AsyncMock(return_value=True)
        self.bot.notifier.send_multi_crypto_signal = AsyncMock(return_value=True)
    
    def _create_realistic_data(self):
        """ایجاد داده‌های واقع‌گرایانه"""
        dates = pd.date_range(start='2025-01-01', periods=200, freq='H')
        base_price = 50000
        
        # شبیه‌سازی الگوی breakout
        prices = []
        for i in range(200):
            if i < 150:
                # consolidation
                noise = np.random.normal(0, 0.01)
                price = base_price * (1 + noise)
            else:
                # breakout
                growth = 0.002 * (i - 150)
                noise = np.random.normal(0, 0.01)
                price = base_price * (1 + 0.03 + growth + noise)  # 3% initial breakout
            
            prices.append(price)
        
        data = []
        for i, close in enumerate(prices):
            high = close * np.random.uniform(1.001, 1.015)
            low = close * np.random.uniform(0.985, 0.999)
            open_price = prices[i-1] if i > 0 else close
            
            # حجم بالا در breakout
            if i >= 150:
                volume = np.random.uniform(800, 1500)
            else:
                volume = np.random.uniform(200, 600)
            
            data.append({
                'timestamp': dates[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def _create_strong_signal(self):
        """ایجاد سیگنال قوی"""
        return {
            'symbol': 'BTC/USDT',
            'signal': 1,
            'signal_type': 'BREAKOUT',
            'signal_quality': 'HIGH',
            'confidence': 0.92,
            'current_price': 51500,
            'stop_loss': 50000,
            'take_profits': [52500, 54000, 55500],
            'risk_reward_ratio': 3.5,
            'reasoning': [
                'شکست قوی سطح مقاومت 51000',
                'حجم معاملات 2.5 برابر میانگین',
                'کندل بازگشایی قوی',
                'تأیید اندیکاتورهای momentum'
            ],
            'market_context': {
                'structure': 'STRONG_UPTREND',
                'trend_strength': 'STRONG',
                'volume_context': 'VERY_HIGH'
            }
        }
    
    def _create_positive_ai_eval(self):
        """ایجاد ارزیابی مثبت AI"""
        return {
            'ai_evaluations': {
                'gemini': {
                    'recommendation': 'STRONG_BUY',
                    'confidence': 0.9,
                    'quality_score': 9.2
                },
                'openai': {
                    'recommendation': 'BUY',
                    'confidence': 0.85,
                    'quality_score': 8.8
                }
            },
            'consensus': {
                'type': 'UNANIMOUS',
                'strength': 1.0,
                'avg_confidence': 0.875
            },
            'final_decision': {
                'action': 'STRONG_BUY',
                'confidence': 0.95,
                'trade_worthy': True,
                'quality_score': 9.0
            }
        }
    
    async def test_complete_trading_cycle(self):
        """تست چرخه کامل معاملاتی"""
        # مرحله 1: راه‌اندازی
        init_success = await self.bot.initialize()
        self.assertTrue(init_success)
        
        # مرحله 2: تحلیل همه نمادها
        analysis_results = await self.bot.analyze_all_symbols()
        self.assertIsInstance(analysis_results, dict)
        
        # مرحله 3: ارزیابی AI
        evaluated_results = await self.bot.evaluate_with_multi_ai(analysis_results)
        self.assertIsInstance(evaluated_results, dict)
        
        # مرحله 4: پردازش سیگنال‌ها
        signals_sent = await self.bot.process_trading_signals(evaluated_results)
        
        # بررسی نتیجه نهایی
        self.assertIsInstance(signals_sent, list)
        
        # با سیگنال قوی و ارزیابی مثبت AI، باید سیگنال ارسال شود
        if signals_sent:
            signal = signals_sent[0]
            self.assertEqual(signal['symbol'], 'BTC/USDT')
            self.assertEqual(signal['position'], 'LONG')
            self.assertGreater(signal['confidence'], 0.8)
            self.assertTrue(signal['trade_worthy'])

if __name__ == '__main__':
    # اجرای تست‌های یکپارچگی
    unittest.main(verbosity=2)
