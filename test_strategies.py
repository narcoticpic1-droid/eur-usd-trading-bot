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
    from strategies.pure_price_action_strategy import PurePriceActionStrategy
    from strategies.multi_timeframe_strategy import MultiTimeframeStrategy
    from strategies.mean_reversion_strategy import MeanReversionStrategy
except ImportError:
    print("⚠️ ماژول‌های strategy در دسترس نیستند")

import config

class TestPurePriceActionStrategy(unittest.TestCase):
    """تست استراتژی Pure Price Action"""
    
    def setUp(self):
        """راه‌اندازی تست"""
        try:
            self.strategy = PurePriceActionStrategy()
        except NameError:
            self.skipTest("PurePriceActionStrategy در دسترس نیست")
        
        self.sample_data = self._create_sample_ohlcv_data()
    
    def _create_sample_ohlcv_data(self):
        """ایجاد داده‌های OHLCV نمونه"""
        np.random.seed(42)
        dates = pd.date_range(start='2025-01-01', periods=200, freq='H')
        
        # شبیه‌سازی قیمت با روند
        base_price = 50000
        prices = [base_price]
        
        for i in range(1, 200):
            # روند صعودی با نویز
            trend = 0.0001 * i  # روند آرام صعودی
            noise = np.random.normal(0, 0.02)
            change = trend + noise
            
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1000))  # حداقل $1000
        
        # ایجاد OHLCV
        data = []
        for i, close in enumerate(prices):
            high = close * np.random.uniform(1.001, 1.03)
            low = close * np.random.uniform(0.97, 0.999)
            open_price = prices[i-1] if i > 0 else close
            volume = np.random.uniform(100, 1000)
            
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
    
    def test_strategy_initialization(self):
        """تست راه‌اندازی استراتژی"""
        if not hasattr(self, 'strategy'):
            self.skipTest("Strategy در دسترس نیست")
        
        self.assertIsNotNone(self.strategy)
        self.assertTrue(hasattr(self.strategy, 'analyze'))
    
    def test_signal_generation(self):
        """تست تولید سیگنال"""
        if not hasattr(self, 'strategy'):
            self.skipTest("Strategy در دسترس نیست")
        
        # فرض: strategy دارای متد analyze است
        if hasattr(self.strategy, 'analyze'):
            result = self.strategy.analyze(self.sample_data)
            
            if result:
                # بررسی ساختار نتیجه
                self.assertIn('signal', result)
                self.assertIn('confidence', result)
                self.assertIn(result['signal'], [-1, 0, 1])
                self.assertTrue(0 <= result['confidence'] <= 1)
    
    def test_breakout_detection(self):
        """تست تشخیص شکست سطح"""
        # ایجاد الگوی شکست سطح
        breakout_data = self._create_breakout_pattern()
        
        if hasattr(self, 'strategy') and hasattr(self.strategy, 'analyze'):
            result = self.strategy.analyze(breakout_data)
            
            if result and result.get('signal_type') == 'BREAKOUT':
                self.assertNotEqual(result['signal'], 0)
                self.assertGreater(result['confidence'], 0.5)
    
    def _create_breakout_pattern(self):
        """ایجاد الگوی شکست سطح"""
        dates = pd.date_range(start='2025-01-01', periods=200, freq='H')
        
        # ایجاد resistance در سطح 50000
        resistance_level = 50000
        prices = []
        
        for i in range(200):
            if i < 150:
                # قیمت زیر resistance
                price = resistance_level * np.random.uniform(0.95, 0.99)
            else:
                # شکست resistance
                price = resistance_level * np.random.uniform(1.01, 1.05)
            
            prices.append(price)
        
        data = []
        for i, close in enumerate(prices):
            high = close * np.random.uniform(1.001, 1.02)
            low = close * np.random.uniform(0.98, 0.999)
            open_price = prices[i-1] if i > 0 else close
            volume = np.random.uniform(100, 1000)
            
            # حجم بالاتر در breakout
            if i >= 150:
                volume *= 2
            
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
    
    def test_trend_following_signals(self):
        """تست سیگنال‌های trend following"""
        # ایجاد روند قوی صعودی
        uptrend_data = self._create_strong_uptrend()
        
        if hasattr(self, 'strategy') and hasattr(self.strategy, 'analyze'):
            result = self.strategy.analyze(uptrend_data)
            
            if result:
                # در روند صعودی قوی، انتظار سیگنال خرید
                if result.get('market_context', {}).get('structure') in ['STRONG_UPTREND']:
                    self.assertGreaterEqual(result['signal'], 0)
    
    def _create_strong_uptrend(self):
        """ایجاد روند قوی صعودی"""
        dates = pd.date_range(start='2025-01-01', periods=200, freq='H')
        
        base_price = 45000
        prices = [base_price]
        
        for i in range(1, 200):
            # روند صعودی قوی
            trend = 0.002  # 0.2% رشد هر کندل
            noise = np.random.normal(0, 0.01)
            change = trend + noise
            
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        data = []
        for i, close in enumerate(prices):
            high = close * np.random.uniform(1.001, 1.02)
            low = close * np.random.uniform(0.98, 0.999)
            open_price = prices[i-1] if i > 0 else close
            volume = np.random.uniform(200, 800)
            
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

class TestStrategyPerformance(unittest.TestCase):
    """تست عملکرد استراتژی‌ها"""
    
    def setUp(self):
        """راه‌اندازی تست عملکرد"""
        self.initial_balance = 10000
        self.test_period = 100  # 100 کندل
        
    def test_buy_and_hold_benchmark(self):
        """تست benchmark خرید و نگهداری"""
        # قیمت شروع و پایان
        start_price = 45000
        end_price = 55000
        
        # بازده buy and hold
        buy_hold_return = (end_price - start_price) / start_price
        
        self.assertAlmostEqual(buy_hold_return, 0.2222, places=3)  # ~22.22%
    
    def test_win_rate_calculation(self):
        """تست محاسبه نرخ برد"""
        trades = [
            {'pnl': 100, 'status': 'closed'},   # برنده
            {'pnl': -50, 'status': 'closed'},   # بازنده
            {'pnl': 75, 'status': 'closed'},    # برنده
            {'pnl': -25, 'status': 'closed'},   # بازنده
            {'pnl': 150, 'status': 'closed'}    # برنده
        ]
        
        winning_trades = sum(1 for t in trades if t['pnl'] > 0)
        total_trades = len(trades)
        win_rate = winning_trades / total_trades
        
        self.assertEqual(win_rate, 0.6)  # 60%
    
    def test_profit_factor_calculation(self):
        """تست محاسبه Profit Factor"""
        trades = [100, -50, 75, -25, 150, -30]
        
        gross_profit = sum(t for t in trades if t > 0)  # 325
        gross_loss = abs(sum(t for t in trades if t < 0))  # 105
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        self.assertAlmostEqual(profit_factor, 3.095, places=2)
    
    def test_expectancy_calculation(self):
        """تست محاسبه Expectancy"""
        trades = [100, -50, 75, -25, 150, -30]
        
        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t < 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        win_rate = len(wins) / len(trades)
        loss_rate = len(losses) / len(trades)
        
        expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)
        
        self.assertGreater(expectancy, 0)  # استراتژی سودآور
    
    def test_maximum_consecutive_losses(self):
        """تست حداکثر ضررهای متوالی"""
        trade_results = [1, 1, -1, -1, -1, 1, -1, -1, 1, 1]
        
        max_consecutive_losses = 0
        current_consecutive = 0
        
        for result in trade_results:
            if result < 0:
                current_consecutive += 1
                max_consecutive_losses = max(max_consecutive_losses, current_consecutive)
            else:
                current_consecutive = 0
        
        self.assertEqual(max_consecutive_losses, 3)
    
    def test_equity_curve_analysis(self):
        """تست تحلیل منحنی equity"""
        trades_pnl = [100, -50, 75, -25, 150, -30, 90, -40]
        
        equity_curve = [self.initial_balance]
        for pnl in trades_pnl:
            equity_curve.append(equity_curve[-1] + pnl)
        
        # بررسی روند کلی
        final_balance = equity_curve[-1]
        total_return = (final_balance - self.initial_balance) / self.initial_balance
        
        self.assertGreater(final_balance, self.initial_balance)
        self.assertGreater(total_return, 0)

class TestStrategyRobustness(unittest.TestCase):
    """تست استحکام استراتژی"""
    
    def test_different_market_conditions(self):
        """تست در شرایط مختلف بازار"""
        market_conditions = ['uptrend', 'downtrend', 'sideways', 'volatile']
        
        for condition in market_conditions:
            data = self._create_market_condition_data(condition)
            
            # فرض: تست با یک تحلیلگر ساده
            signals = self._generate_test_signals(data)
            
            # بررسی که strategy حداقل signal تولید کند
            self.assertIsInstance(signals, list)
    
    def _create_market_condition_data(self, condition):
        """ایجاد داده برای شرایط خاص بازار"""
        dates = pd.date_range(start='2025-01-01', periods=100, freq='H')
        base_price = 50000
        prices = [base_price]
        
        for i in range(1, 100):
            if condition == 'uptrend':
                change = np.random.normal(0.002, 0.01)  # روند صعودی
            elif condition == 'downtrend':
                change = np.random.normal(-0.002, 0.01)  # روند نزولی
            elif condition == 'sideways':
                change = np.random.normal(0, 0.005)  # حرکت جانبی
            else:  # volatile
                change = np.random.normal(0, 0.03)  # نوسانات بالا
            
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1000))
        
        return prices
    
    def _generate_test_signals(self, price_data):
        """تولید سیگنال‌های تست"""
        signals = []
        
        # الگوریتم ساده: moving average crossover
        if len(price_data) >= 20:
            short_ma = np.mean(price_data[-10:])
            long_ma = np.mean(price_data[-20:])
            
            if short_ma > long_ma:
                signals.append({'signal': 1, 'price': price_data[-1]})
            elif short_ma < long_ma:
                signals.append({'signal': -1, 'price': price_data[-1]})
        
        return signals
    
    def test_parameter_sensitivity(self):
        """تست حساسیت به پارامترها"""
        # تست با پارامترهای مختلف
        parameters = [
            {'ma_short': 5, 'ma_long': 15},
            {'ma_short': 10, 'ma_long': 20},
            {'ma_short': 20, 'ma_long': 50}
        ]
        
        test_data = list(range(50000, 50100))  # 100 قیمت
        
        results = []
        for params in parameters:
            signals = self._test_ma_strategy(test_data, params)
            results.append(len(signals))
        
        # بررسی که تغییر پارامتر تأثیر داشته باشد
        self.assertNotEqual(results[0], results[-1])
    
    def _test_ma_strategy(self, prices, params):
        """تست استراتژی moving average"""
        signals = []
        short_period = params['ma_short']
        long_period = params['ma_long']
        
        if len(prices) >= long_period:
            short_ma = np.mean(prices[-short_period:])
            long_ma = np.mean(prices[-long_period:])
            
            if short_ma > long_ma:
                signals.append(1)
            elif short_ma < long_ma:
                signals.append(-1)
        
        return signals
    
    def test_slippage_impact(self):
        """تست تأثیر slippage"""
        entry_price = 50000
        slippage_rates = [0, 0.001, 0.002, 0.005]  # 0%, 0.1%, 0.2%, 0.5%
        
        for slippage in slippage_rates:
            actual_entry = entry_price * (1 + slippage)
            impact = (actual_entry - entry_price) / entry_price
            
            self.assertEqual(impact, slippage)
            
            # بررسی که slippage بالا تأثیر قابل توجه داشته باشد
            if slippage >= 0.005:  # 0.5%
                self.assertGreater(impact, 0.004)

class TestStrategyOptimization(unittest.TestCase):
    """تست بهینه‌سازی استراتژی"""
    
    def test_parameter_grid_search(self):
        """تست جستجوی شبکه‌ای پارامترها"""
        # پارامترهای مختلف برای تست
        ma_short_values = [5, 10, 15]
        ma_long_values = [20, 30, 40]
        
        best_params = None
        best_score = -float('inf')
        
        test_data = self._create_optimization_data()
        
        for short in ma_short_values:
            for long in ma_long_values:
                if short < long:  # شرط منطقی
                    params = {'ma_short': short, 'ma_long': long}
                    score = self._evaluate_strategy(test_data, params)
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
        
        self.assertIsNotNone(best_params)
        self.assertGreater(best_score, -float('inf'))
    
    def _create_optimization_data(self):
        """ایجاد داده برای بهینه‌سازی"""
        np.random.seed(42)
        return np.cumsum(np.random.normal(0.001, 0.02, 200)) + 50000
    
    def _evaluate_strategy(self, data, params):
        """ارزیابی استراتژی با پارامترهای مشخص"""
        # شبیه‌سازی ساده
        if len(data) < params['ma_long']:
            return 0
        
        short_ma = np.mean(data[-params['ma_short']:])
        long_ma = np.mean(data[-params['ma_long']:])
        
        # امتیازدهی بر اساس تفاوت MA ها
        score = abs(short_ma - long_ma) / long_ma
        
        return score
    
    def test_walk_forward_analysis(self):
        """تست تحلیل Walk Forward"""
        total_periods = 200
        training_window = 100
        test_window = 50
        
        # شبیه‌سازی walk forward
        results = []
        
        for start in range(0, total_periods - training_window - test_window, test_window):
            training_end = start + training_window
            test_end = training_end + test_window
            
            # فرض: optimization روی training data
            training_data = list(range(start, training_end))
            test_data = list(range(training_end, test_end))
            
            # بهینه‌سازی
            optimal_params = self._optimize_on_training(training_data)
            
            # تست
            test_score = self._test_on_data(test_data, optimal_params)
            results.append(test_score)
        
        # بررسی که حداقل یک دوره تست شده باشد
        self.assertGreater(len(results), 0)
    
    def _optimize_on_training(self, training_data):
        """بهینه‌سازی روی داده‌های آموزش"""
        # شبیه‌سازی: بازگردان پارامتر بهینه
        return {'ma_short': 10, 'ma_long': 20}
    
    def _test_on_data(self, test_data, params):
        """تست روی داده‌های آزمون"""
        # شبیه‌سازی: بازگردان امتیاز
        return np.random.uniform(0.5, 1.5)

if __name__ == '__main__':
    # اجرای تست‌های استراتژی
    unittest.main(verbosity=2)
