import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from datetime import datetime, timedelta

# اضافه کردن مسیر پروژه
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from analyzers.pure_price_action_analyzer import PurePriceActionAnalyzer, MarketStructure, SignalQuality
    from analyzers.multi_ai_evaluator import MultiAIEvaluator
    from analyzers.correlation_analyzer import CorrelationAnalyzer
except ImportError:
    from pure_price_action_analyzer import PurePriceActionAnalyzer, MarketStructure, SignalQuality
    from multi_ai_evaluator import MultiAIEvaluator

class TestPurePriceActionAnalyzer(unittest.TestCase):
    """تست تحلیلگر Pure Price Action"""
    
    def setUp(self):
        """راه‌اندازی تست"""
        self.analyzer = PurePriceActionAnalyzer()
        self.sample_data = self._create_sample_data()
    
    def _create_sample_data(self):
        """ایجاد داده‌های نمونه برای تست"""
        np.random.seed(42)
        
        # ایجاد 200 کندل
        dates = pd.date_range(start='2025-01-01', periods=200, freq='H')
        
        # قیمت پایه
        base_price = 100
        prices = []
        
        # شبیه‌سازی حرکت قیمت با روند صعودی
        for i in range(200):
            if i == 0:
                prices.append(base_price)
            else:
                # اضافه کردن نویز تصادفی + روند
                change = np.random.normal(0.2, 1.5)  # روند صعودی آرام
                new_price = prices[-1] * (1 + change/100)
                prices.append(max(new_price, 1))  # جلوگیری از قیمت منفی
        
        # ایجاد OHLCV
        data = []
        for i, price in enumerate(prices):
            high = price * np.random.uniform(1.001, 1.02)
            low = price * np.random.uniform(0.98, 0.999)
            open_price = prices[i-1] if i > 0 else price
            close = price
            volume = np.random.uniform(1000, 10000)
            
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
    
    def test_analyzer_initialization(self):
        """تست راه‌اندازی تحلیلگر"""
        self.assertEqual(self.analyzer.name, "Pure Price Action")
        self.assertEqual(self.analyzer.min_candles, 150)
    
    def test_insufficient_data(self):
        """تست با داده‌های ناکافی"""
        small_data = self.sample_data.head(100)  # کمتر از 150 کندل
        result = self.analyzer.analyze(small_data)
        self.assertIsNone(result)
    
    def test_none_data(self):
        """تست با داده None"""
        result = self.analyzer.analyze(None)
        self.assertIsNone(result)
    
    def test_empty_data(self):
        """تست با DataFrame خالی"""
        empty_df = pd.DataFrame()
        result = self.analyzer.analyze(empty_df)
        self.assertIsNone(result)
    
    def test_swing_points_detection(self):
        """تست شناسایی نقاط swing"""
        df = self.analyzer._find_swing_points(self.sample_data.copy())
        
        # بررسی وجود ستون‌های swing
        self.assertIn('swing_high', df.columns)
        self.assertIn('swing_low', df.columns)
        
        # بررسی اینکه حداقل چند swing point یافت شده
        swing_highs = df[df['swing_high']].shape[0]
        swing_lows = df[df['swing_low']].shape[0]
        
        self.assertGreater(swing_highs, 0, "هیچ swing high یافت نشد")
        self.assertGreater(swing_lows, 0, "هیچ swing low یافت نشد")
    
    def test_market_structure_identification(self):
        """تست شناسایی ساختار بازار"""
        df = self.analyzer._identify_market_structure(self.sample_data.copy())
        
        # بررسی وجود ستون‌های مورد نیاز
        required_columns = ['ema_8', 'ema_21', 'ema_50', 'market_structure', 
                          'adx', 'trend_strength']
        
        for col in required_columns:
            self.assertIn(col, df.columns, f"ستون {col} موجود نیست")
        
        # بررسی مقادیر معتبر market structure
        valid_structures = [e.value for e in MarketStructure]
        unique_structures = df['market_structure'].unique()
        
        for structure in unique_structures:
            self.assertIn(structure, valid_structures, 
                         f"ساختار نامعتبر: {structure}")
    
    def test_key_levels_detection(self):
        """تست شناسایی سطوح کلیدی"""
        df = self.sample_data.copy()
        df = self.analyzer._find_swing_points(df)
        df = self.analyzer._find_key_levels(df)
        
        # بررسی وجود ستون‌ها
        required_columns = ['support_level', 'resistance_level', 
                          'distance_to_support', 'distance_to_resistance']
        
        for col in required_columns:
            self.assertIn(col, df.columns, f"ستون {col} موجود نیست")
        
        # بررسی عدم وجود NaN (پس از fill)
        self.assertFalse(df['distance_to_support'].isna().any(), 
                        "مقادیر NaN در distance_to_support")
        self.assertFalse(df['distance_to_resistance'].isna().any(), 
                        "مقادیر NaN در distance_to_resistance")
    
    def test_candle_patterns_analysis(self):
        """تست تحلیل الگوهای کندل"""
        df = self.analyzer._analyze_candle_patterns(self.sample_data.copy())
        
        # بررسی وجود ستون‌های الگوهای کندل
        pattern_columns = ['body', 'upper_shadow', 'lower_shadow', 'total_range',
                         'body_ratio', 'is_doji', 'is_hammer', 'is_shooting_star',
                         'is_pin_bar', 'is_engulfing']
        
        for col in pattern_columns:
            self.assertIn(col, df.columns, f"ستون {col} موجود نیست")
        
        # بررسی منطقی بودن محاسبات
        # body_ratio باید بین 0 و 1 باشد
        self.assertTrue((df['body_ratio'] >= 0).all(), "body_ratio منفی")
        self.assertTrue((df['body_ratio'] <= 1).all(), "body_ratio بیش از 1")
        
        # total_range باید مثبت باشد
        self.assertTrue((df['total_range'] > 0).all(), "total_range غیرمثبت")
    
    def test_trend_change_detection(self):
        """تست تشخیص تغییرات ترند"""
        df = self.sample_data.copy()
        df = self.analyzer._identify_market_structure(df)
        df = self.analyzer._detect_trend_changes(df)
        
        self.assertIn('trend_change_signal', df.columns)
        
        # بررسی مقادیر معتبر (-1, 0, 1)
        valid_signals = [-1, 0, 1]
        unique_signals = df['trend_change_signal'].unique()
        
        for signal in unique_signals:
            self.assertIn(signal, valid_signals, f"سیگنال نامعتبر: {signal}")
    
    def test_volume_analysis(self):
        """تست تحلیل حجم"""
        df = self.analyzer._analyze_volume_context(self.sample_data.copy())
        
        volume_columns = ['volume_ma', 'volume_ratio', 'high_volume', 'low_volume',
                         'buying_pressure', 'selling_pressure']
        
        for col in volume_columns:
            self.assertIn(col, df.columns, f"ستون {col} موجود نیست")
        
        # بررسی محدوده‌های منطقی
        self.assertTrue((df['buying_pressure'] >= 0).all() and 
                       (df['buying_pressure'] <= 1).all(), 
                       "buying_pressure خارج از محدوده 0-1")
        
        self.assertTrue((df['selling_pressure'] >= 0).all() and 
                       (df['selling_pressure'] <= 1).all(), 
                       "selling_pressure خارج از محدوده 0-1")
    
    def test_full_analysis(self):
        """تست تحلیل کامل"""
        result = self.analyzer.analyze(self.sample_data)
        
        self.assertIsNotNone(result, "نتیجه تحلیل None است")
        self.assertIsInstance(result, dict, "نتیجه باید dictionary باشد")
        
        # بررسی کلیدهای ضروری
        required_keys = ['symbol', 'timestamp', 'current_price', 'signal',
                        'signal_type', 'signal_quality', 'confidence',
                        'reasoning', 'market_context']
        
        for key in required_keys:
            self.assertIn(key, result, f"کلید {key} در نتیجه موجود نیست")
        
        # بررسی نوع داده‌ها
        self.assertIsInstance(result['signal'], int)
        self.assertIn(result['signal'], [-1, 0, 1])
        self.assertIsInstance(result['confidence'], (int, float))
        self.assertTrue(0 <= result['confidence'] <= 1)
    
    def test_market_structure_transitions(self):
        """تست انتقالات ساختار بازار"""
        # ایجاد داده با تغییر روند مشخص
        trend_change_data = self._create_trend_change_data()
        result = self.analyzer.analyze(trend_change_data)
        
        if result:
            # باید تغییر روند را تشخیص دهد
            self.assertNotEqual(result['signal'], 0, 
                              "تغییر روند تشخیص داده نشد")
    
    def _create_trend_change_data(self):
        """ایجاد داده با تغییر روند مشخص"""
        dates = pd.date_range(start='2025-01-01', periods=200, freq='H')
        
        prices = []
        base_price = 100
        
        for i in range(200):
            if i < 100:
                # 100 کندل اول: روند نزولی
                change = np.random.normal(-0.3, 1.0)
            else:
                # 100 کندل دوم: روند صعودی
                change = np.random.normal(0.5, 1.0)
            
            if i == 0:
                prices.append(base_price)
            else:
                new_price = prices[-1] * (1 + change/100)
                prices.append(max(new_price, 1))
        
        data = []
        for i, price in enumerate(prices):
            high = price * np.random.uniform(1.001, 1.015)
            low = price * np.random.uniform(0.985, 0.999)
            open_price = prices[i-1] if i > 0 else price
            volume = np.random.uniform(1000, 10000)
            
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

class TestMultiAIEvaluator(unittest.TestCase):
    """تست ارزیابی‌کننده چندگانه AI"""
    
    def setUp(self):
        """راه‌اندازی تست"""
        self.evaluator = MultiAIEvaluator()
        self.sample_analysis = {
            'symbol': 'BTC/USDT',
            'signal': 1,
            'signal_type': 'BREAKOUT',
            'confidence': 0.8,
            'current_price': 50000,
            'reasoning': 'Strong breakout above resistance'
        }
    
    @patch('multi_ai_evaluator.genai')
    def test_gemini_evaluation(self, mock_genai):
        """تست ارزیابی Gemini"""
        # Mock response
        mock_response = Mock()
        mock_response.text = """{
            "recommendation": "BUY",
            "confidence": 0.85,
            "quality_score": 0.9,
            "reasoning": "Strong technical setup"
        }"""
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        mock_genai.configure = Mock()
        
        result = self.evaluator._evaluate_with_gemini(self.sample_analysis, 'BTC/USDT')
        
        self.assertIsNotNone(result)
        self.assertIn('recommendation', result)
        self.assertIn('confidence', result)
    
    @patch('multi_ai_evaluator.openai.OpenAI')
    def test_openai_evaluation(self, mock_openai_class):
        """تست ارزیابی OpenAI"""
        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """{
            "recommendation": "BUY",
            "confidence": 0.8,
            "quality_score": 0.85,
            "reasoning": "Good signal quality"
        }"""
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        result = self.evaluator._evaluate_with_openai(self.sample_analysis, 'BTC/USDT')
        
        self.assertIsNotNone(result)
        self.assertIn('recommendation', result)
    
    def test_consensus_analysis_unanimous(self):
        """تست تحلیل consensus - اتفاق نظر کامل"""
        evaluations = {
            'gemini': {
                'recommendation': 'BUY',
                'confidence': 0.8,
                'quality_score': 0.9
            },
            'openai': {
                'recommendation': 'BUY', 
                'confidence': 0.85,
                'quality_score': 0.88
            }
        }
        
        consensus = self.evaluator._analyze_consensus(evaluations)
        
        self.assertEqual(consensus['type'], 'UNANIMOUS')
        self.assertEqual(consensus['strength'], 1.0)
    
    def test_consensus_analysis_split(self):
        """تست تحلیل consensus - تقسیم نظرات"""
        evaluations = {
            'gemini': {
                'recommendation': 'BUY',
                'confidence': 0.8,
                'quality_score': 0.9
            },
            'openai': {
                'recommendation': 'SELL',
                'confidence': 0.7,
                'quality_score': 0.8
            }
        }
        
        consensus = self.evaluator._analyze_consensus(evaluations)
        
        self.assertIn(consensus['type'], ['MAJORITY', 'SPLIT'])
        self.assertLess(consensus['strength'], 1.0)
    
    def test_final_decision_strong_consensus(self):
        """تست تصمیم نهایی با consensus قوی"""
        consensus = {
            'strength': 0.9,
            'recommendations': ['BUY', 'BUY'],
            'avg_confidence': 0.85,
            'avg_quality': 0.9
        }
        
        decision = self.evaluator._make_final_decision(
            self.sample_analysis, {}, consensus
        )
        
        self.assertEqual(decision['action'], 'BUY')
        self.assertGreater(decision['confidence'], 0.8)
        self.assertTrue(decision['trade_worthy'])
    
    def test_final_decision_weak_consensus(self):
        """تست تصمیم نهایی با consensus ضعیف"""
        consensus = {
            'strength': 0.2,
            'recommendations': ['BUY', 'SELL', 'HOLD'],
            'avg_confidence': 0.5,
            'avg_quality': 0.6
        }
        
        decision = self.evaluator._make_final_decision(
            self.sample_analysis, {}, consensus
        )
        
        self.assertEqual(decision['action'], 'HOLD')
        self.assertLess(decision['confidence'], 0.7)
        self.assertFalse(decision['trade_worthy'])

class TestCorrelationAnalyzer(unittest.TestCase):
    """تست تحلیلگر همبستگی"""
    
    def setUp(self):
        """راه‌اندازی تست"""
        try:
            self.analyzer = CorrelationAnalyzer()
        except NameError:
            self.skipTest("CorrelationAnalyzer در دسترس نیست")
        
        # ایجاد داده‌های نمونه برای چندین نماد
        self.sample_data = self._create_multi_symbol_data()
    
    def _create_multi_symbol_data(self):
        """ایجاد داده‌های چند نمادی"""
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        dates = pd.date_range(start='2025-01-01', periods=100, freq='H')
        
        data = {}
        np.random.seed(42)
        
        for i, symbol in enumerate(symbols):
            # قیمت پایه متفاوت برای هر نماد
            base_price = [50000, 3000, 100][i]
            prices = [base_price]
            
            for j in range(1, 100):
                # همبستگی بین نمادها
                if symbol == 'BTC/USDT':
                    change = np.random.normal(0, 2)
                elif symbol == 'ETH/USDT':
                    # ETH کمی همبسته با BTC
                    btc_influence = 0.7 if i > 0 else 0
                    change = np.random.normal(0, 2.5) + btc_influence * 0.5
                else:  # SOL
                    # SOL متغیرتر
                    change = np.random.normal(0, 3)
                
                new_price = prices[-1] * (1 + change/100)
                prices.append(max(new_price, 1))
            
            data[symbol] = prices
        
        return data
    
    def test_correlation_matrix_calculation(self):
        """تست محاسبه ماتریس همبستگی"""
        if not hasattr(self, 'analyzer'):
            self.skipTest("CorrelationAnalyzer در دسترس نیست")
        
        correlation_matrix = self.analyzer._calculate_correlation_matrix(self.sample_data)
        
        self.assertIsInstance(correlation_matrix, pd.DataFrame)
        self.assertEqual(correlation_matrix.shape[0], correlation_matrix.shape[1])
        
        # بررسی مقادیر قطر اصلی (باید 1 باشند)
        for symbol in correlation_matrix.index:
            self.assertAlmostEqual(correlation_matrix.loc[symbol, symbol], 1.0, places=2)

if __name__ == '__main__':
    # اجرای تست‌ها
    unittest.main(verbosity=2)
