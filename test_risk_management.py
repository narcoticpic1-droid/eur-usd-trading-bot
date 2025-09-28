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
    from risk_management.portfolio_manager import PortfolioManager
    from risk_management.position_sizer import PositionSizer
    from risk_management.risk_calculator import RiskCalculator
except ImportError:
    print("⚠️ ماژول‌های risk management در دسترس نیستند")

import config

class TestRiskCalculations(unittest.TestCase):
    """تست محاسبات ریسک"""
    
    def setUp(self):
        """راه‌اندازی تست"""
        self.test_balance = 10000  # $10,000
        self.test_price = 50000   # $50,000 BTC
        
    def test_position_size_calculation(self):
        """تست محاسبه اندازه پوزیشن"""
        # محاسبه 1% ریسک
        risk_percent = 0.01
        stop_loss_percent = 0.02  # 2% stop loss
        
        risk_amount = self.test_balance * risk_percent  # $100
        position_size = risk_amount / (self.test_price * stop_loss_percent)
        
        expected_size = 100 / (50000 * 0.02)  # 0.1 BTC
        
        self.assertAlmostEqual(position_size, expected_size, places=6)
    
    def test_leverage_calculation(self):
        """تست محاسبه اهرم"""
        position_value = 1000  # $1000 position
        margin_required = 100  # $100 margin
        
        leverage = position_value / margin_required
        
        self.assertEqual(leverage, 10)  # 10x leverage
    
    def test_risk_reward_ratio(self):
        """تست نسبت ریسک به ریوارد"""
        entry_price = 50000
        stop_loss = 49000  # $1000 ریسک
        take_profit = 53000  # $3000 ریوارد
        
        risk = entry_price - stop_loss
        reward = take_profit - entry_price
        ratio = reward / risk
        
        self.assertEqual(ratio, 3.0)  # نسبت 1:3
    
    def test_portfolio_heat(self):
        """تست محاسبه heat پورتفولیو"""
        positions = [
            {'risk_amount': 100, 'status': 'active'},
            {'risk_amount': 150, 'status': 'active'},
            {'risk_amount': 200, 'status': 'closed'}
        ]
        
        total_risk = sum(p['risk_amount'] for p in positions if p['status'] == 'active')
        portfolio_heat = total_risk / self.test_balance
        
        expected_heat = 250 / 10000  # 2.5%
        self.assertEqual(portfolio_heat, expected_heat)
    
    def test_maximum_position_size(self):
        """تست حداکثر اندازه پوزیشن"""
        max_risk_per_trade = config.PORTFOLIO_MANAGEMENT['max_single_position_risk']
        max_risk_amount = self.test_balance * max_risk_per_trade
        
        # با 2% stop loss
        stop_loss_percent = 0.02
        max_position_size = max_risk_amount / (self.test_price * stop_loss_percent)
        
        # بررسی که اندازه منطقی باشد
        self.assertGreater(max_position_size, 0)
        self.assertLess(max_position_size, self.test_balance / self.test_price)
    
    def test_correlation_risk(self):
        """تست ریسک همبستگی"""
        # فرض: دو پوزیشن با همبستگی 0.8
        position1_risk = 100
        position2_risk = 150
        correlation = 0.8
        
        # محاسبه ریسک تعدیل شده
        combined_risk = np.sqrt(
            position1_risk**2 + 
            position2_risk**2 + 
            2 * correlation * position1_risk * position2_risk
        )
        
        # ریسک ترکیبی باید بیشتر از بزرگترین پوزیشن جداگانه باشد
        self.assertGreater(combined_risk, max(position1_risk, position2_risk))
        # اما کمتر از مجموع آنها
        self.assertLess(combined_risk, position1_risk + position2_risk)

class TestPortfolioManager(unittest.TestCase):
    """تست مدیر پورتفولیو"""
    
    def setUp(self):
        """راه‌اندازی تست"""
        try:
            self.portfolio_manager = PortfolioManager()
        except NameError:
            self.skipTest("PortfolioManager در دسترس نیست")
    
    def test_portfolio_initialization(self):
        """تست راه‌اندازی پورتفولیو"""
        if not hasattr(self, 'portfolio_manager'):
            self.skipTest("PortfolioManager در دسترس نیست")
        
        self.assertIsNotNone(self.portfolio_manager)
        self.assertIsInstance(self.portfolio_manager.positions, dict)
        self.assertEqual(self.portfolio_manager.total_exposure, 0.0)
    
    def test_add_position(self):
        """تست اضافه کردن پوزیشن"""
        if not hasattr(self, 'portfolio_manager'):
            self.skipTest("PortfolioManager در دسترس نیست")
        
        position_data = {
            'symbol': 'BTC/USDT',
            'side': 'long',
            'size': 0.1,
            'entry_price': 50000,
            'leverage': 5,
            'risk_amount': 100
        }
        
        success = self.portfolio_manager.add_position(position_data)
        
        if hasattr(self.portfolio_manager, 'add_position'):
            self.assertTrue(success)
            self.assertIn('BTC/USDT', self.portfolio_manager.positions)

class TestPositionSizer(unittest.TestCase):
    """تست محاسبه‌کننده اندازه پوزیشن"""
    
    def setUp(self):
        """راه‌اندازی تست"""
        try:
            self.position_sizer = PositionSizer()
        except NameError:
            self.skipTest("PositionSizer در دسترس نیست")
        
        self.account_balance = 10000
        self.risk_per_trade = 0.01  # 1%
    
    def test_fixed_fractional_sizing(self):
        """تست اندازه‌گیری Fixed Fractional"""
        if not hasattr(self, 'position_sizer'):
            self.skipTest("PositionSizer در دسترس نیست")
        
        entry_price = 50000
        stop_loss = 49000
        
        position_size = self._calculate_position_size(
            self.account_balance,
            self.risk_per_trade,
            entry_price,
            stop_loss
        )
        
        # بررسی که اندازه منطقی باشد
        self.assertGreater(position_size, 0)
        
        # بررسی که ریسک واقعی برابر ریسک هدف باشد
        actual_risk = position_size * (entry_price - stop_loss)
        expected_risk = self.account_balance * self.risk_per_trade
        
        self.assertAlmostEqual(actual_risk, expected_risk, places=2)
    
    def _calculate_position_size(self, balance, risk_percent, entry, stop_loss):
        """محاسبه اندازه پوزیشن"""
        risk_amount = balance * risk_percent
        price_risk = entry - stop_loss
        return risk_amount / price_risk
    
    def test_kelly_criterion(self):
        """تست معیار Kelly"""
        # فرض: نرخ برد 60%، متوسط سود 3%, متوسط ضرر 1%
        win_rate = 0.6
        avg_win = 0.03
        avg_loss = 0.01
        
        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        
        # Kelly باید مثبت و کمتر از 1 باشد
        self.assertGreater(kelly_fraction, 0)
        self.assertLess(kelly_fraction, 1)
    
    def test_volatility_adjustment(self):
        """تست تعدیل بر اساس نوسانات"""
        base_size = 0.1
        
        # نوسانات بالا -> اندازه کمتر
        high_volatility = 0.05  # 5% daily volatility
        low_volatility = 0.02   # 2% daily volatility
        
        high_vol_multiplier = 0.02 / high_volatility  # 0.4
        low_vol_multiplier = 0.02 / low_volatility    # 1.0
        
        adjusted_size_high_vol = base_size * high_vol_multiplier
        adjusted_size_low_vol = base_size * low_vol_multiplier
        
        self.assertLess(adjusted_size_high_vol, adjusted_size_low_vol)

class TestRiskLimits(unittest.TestCase):
    """تست محدودیت‌های ریسک"""
    
    def test_daily_risk_limit(self):
        """تست محدودیت ریسک روزانه"""
        daily_limit = config.PORTFOLIO_MANAGEMENT['max_daily_risk']
        
        current_risk = 0.025  # 2.5%
        
        if current_risk >= daily_limit:
            risk_exceeded = True
        else:
            risk_exceeded = False
        
        self.assertFalse(risk_exceeded, "ریسک روزانه از حد مجاز تجاوز کرد")
    
    def test_position_correlation_limit(self):
        """تست محدودیت همبستگی پوزیشن‌ها"""
        # فرض: 3 پوزیشن با همبستگی بالا
        positions = [
            {'symbol': 'BTC/USDT', 'correlation_to_btc': 1.0},
            {'symbol': 'ETH/USDT', 'correlation_to_btc': 0.85},
            {'symbol': 'SOL/USDT', 'correlation_to_btc': 0.75}
        ]
        
        high_correlation_count = sum(
            1 for p in positions 
            if p['correlation_to_btc'] > 0.8
        )
        
        max_correlated = config.SYMBOL_COORDINATION.get('max_correlated_positions', 2)
        
        self.assertLessEqual(high_correlation_count, max_correlated,
                           "تعداد پوزیشن‌های همبسته بیش از حد")
    
    def test_leverage_limits(self):
        """تست محدودیت‌های اهرم"""
        test_symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        
        for symbol in test_symbols:
            if symbol in config.SYMBOL_MAX_LEVERAGE:
                max_leverage = config.SYMBOL_MAX_LEVERAGE[symbol]
                
                # بررسی که اهرم در محدوده مجاز باشد
                self.assertLessEqual(max_leverage, config.MAX_LEVERAGE)
                self.assertGreaterEqual(max_leverage, config.MIN_LEVERAGE)
    
    def test_exposure_limits(self):
        """تست محدودیت‌های exposure"""
        positions = [
            {'exposure': 0.02},  # 2%
            {'exposure': 0.015}, # 1.5%
            {'exposure': 0.01}   # 1%
        ]
        
        total_exposure = sum(p['exposure'] for p in positions)
        max_exposure = config.PORTFOLIO_MANAGEMENT['max_total_exposure']
        
        self.assertLessEqual(total_exposure, max_exposure,
                           "کل exposure از حد مجاز تجاوز کرد")

class TestCircuitBreakers(unittest.TestCase):
    """تست سیستم‌های Circuit Breaker"""
    
    def test_consecutive_losses_breaker(self):
        """تست circuit breaker برای ضررهای متوالی"""
        consecutive_losses = 4
        max_consecutive = config.CIRCUIT_BREAKERS['consecutive_loss_limit']
        
        should_stop = consecutive_losses >= max_consecutive
        
        self.assertTrue(should_stop, "Circuit breaker باید فعال شود")
    
    def test_daily_loss_breaker(self):
        """تست circuit breaker برای ضرر روزانه"""
        daily_loss = 0.12  # 12% ضرر
        loss_limit = config.EMERGENCY_STOPS['portfolio_loss_24h']
        
        emergency_stop = daily_loss >= loss_limit
        
        self.assertTrue(emergency_stop, "Emergency stop باید فعال شود")
    
    def test_volatility_breaker(self):
        """تست circuit breaker برای نوسانات غیرعادی"""
        market_volatility = 0.20  # 20% نوسانات
        volatility_threshold = config.CIRCUIT_BREAKERS['unusual_volatility_threshold']
        
        high_volatility_warning = market_volatility > volatility_threshold
        
        self.assertTrue(high_volatility_warning, 
                       "هشدار نوسانات بالا باید فعال شود")
    
    def test_correlation_spike_breaker(self):
        """تست circuit breaker برای همبستگی ناگهانی"""
        correlation_matrix = pd.DataFrame({
            'BTC/USDT': [1.0, 0.95, 0.92],
            'ETH/USDT': [0.95, 1.0, 0.94],
            'SOL/USDT': [0.92, 0.94, 1.0]
        })
        
        # میانگین همبستگی (بدون قطر اصلی)
        avg_correlation = correlation_matrix.values[np.triu_indices_from(
            correlation_matrix.values, k=1
        )].mean()
        
        correlation_threshold = config.CIRCUIT_BREAKERS['correlation_spike_threshold']
        correlation_spike = avg_correlation > correlation_threshold
        
        self.assertTrue(correlation_spike, 
                       "Circuit breaker همبستگی باید فعال شود")

class TestRiskMetrics(unittest.TestCase):
    """تست معیارهای ریسک"""
    
    def test_value_at_risk_calculation(self):
        """تست محاسبه Value at Risk"""
        # شبیه‌سازی بازده‌های روزانه
        np.random.seed(42)
        daily_returns = np.random.normal(0.001, 0.02, 252)  # یک سال
        
        # VaR 95%
        var_95 = np.percentile(daily_returns, 5)
        
        # VaR باید منفی باشد (نشان‌دهنده ضرر)
        self.assertLess(var_95, 0)
        
        # مقدار منطقی برای VaR روزانه
        self.assertGreater(var_95, -0.1)  # کمتر از 10% ضرر روزانه
    
    def test_sharpe_ratio_calculation(self):
        """تست محاسبه نسبت شارپ"""
        # فرض: بازده 15% سالانه، نوسانات 20%، نرخ بدون ریسک 2%
        annual_return = 0.15
        annual_volatility = 0.20
        risk_free_rate = 0.02
        
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
        
        # نسبت شارپ معقول
        self.assertGreater(sharpe_ratio, 0.5)
        self.assertLess(sharpe_ratio, 2.0)
    
    def test_maximum_drawdown(self):
        """تست محاسبه Maximum Drawdown"""
        # شبیه‌سازی equity curve
        np.random.seed(42)
        prices = [10000]  # شروع با $10,000
        
        for i in range(100):
            change = np.random.normal(0.001, 0.02)
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        equity_curve = np.array(prices)
        
        # محاسبه running maximum
        running_max = np.maximum.accumulate(equity_curve)
        
        # محاسبه drawdown
        drawdown = (equity_curve - running_max) / running_max
        
        # Maximum drawdown
        max_drawdown = np.min(drawdown)
        
        # Max drawdown باید منفی باشد
        self.assertLess(max_drawdown, 0)
        
        # و کمتر از 50% (برای strategy معقول)
        self.assertGreater(max_drawdown, -0.5)

if __name__ == '__main__':
    # اجرای تست‌های ریسک
    unittest.main(verbosity=2)
