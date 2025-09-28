import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import random
from datetime import datetime, timedelta
from dataclasses import dataclass
import config

@dataclass
class MarketCondition:
    """شرایط بازار برای شبیه‌سازی"""
    volatility: float = 1.0
    liquidity: float = 1.0
    spread: float = 0.001
    slippage_factor: float = 1.0
    market_impact: float = 0.001

@dataclass
class SimulationSettings:
    """تنظیمات شبیه‌سازی"""
    enable_slippage: bool = True
    enable_spread: bool = True
    enable_commission: bool = True
    enable_market_impact: bool = True
    enable_liquidity_gaps: bool = True
    enable_weekend_gaps: bool = True
    realistic_execution: bool = True
    latency_ms: int = 50  # تأخیر شبکه
    

class TradeSimulator:
    """
    شبیه‌ساز واقعی معاملات - شامل تمام عوامل دنیای واقعی
    """
    
    def __init__(self, settings: SimulationSettings = None):
        self.settings = settings or SimulationSettings()
        
        # تنظیمات کمیسیون برای صرافی‌های مختلف
        self.commission_rates = {
            'bybit': {'maker': 0.0001, 'taker': 0.0006},
            'binance': {'maker': 0.001, 'taker': 0.001},
            'kucoin': {'maker': 0.001, 'taker': 0.001},
            'okx': {'maker': 0.0008, 'taker': 0.001}
        }
        
        # spread معمول برای نمادها
        self.typical_spreads = {
            'BTC/USDT': 0.01,  # $0.01
            'ETH/USDT': 0.01,  # $0.01
            'SOL/USDT': 0.001, # $0.001
        }
        
        # تنظیمات slippage
        self.slippage_models = {
            'linear': self._linear_slippage,
            'square_root': self._sqrt_slippage,
            'market_impact': self._market_impact_slippage
        }

    def simulate_order_execution(self, order: Dict, market_data: pd.Series, 
                                market_condition: MarketCondition = None) -> Dict:
        """شبیه‌سازی اجرای سفارش"""
        
        if market_condition is None:
            market_condition = MarketCondition()
        
        result = {
            'order_id': order.get('id', 'sim_order'),
            'symbol': order.get('symbol', ''),
            'side': order.get('side', 'buy'),  # buy/sell
            'size': order.get('size', 0),
            'requested_price': order.get('price', market_data['close']),
            'order_type': order.get('type', 'market'),  # market/limit
            'executed': False,
            'executed_price': 0,
            'executed_size': 0,
            'commission': 0,
            'slippage': 0,
            'total_cost': 0,
            'execution_time_ms': 0,
            'reject_reason': None
        }
        
        # بررسی اعتبار سفارش
        validation_result = self._validate_order(order, market_data)
        if not validation_result['valid']:
            result['reject_reason'] = validation_result['reason']
            return result
        
        # شبیه‌سازی تأخیر شبکه
        if self.settings.realistic_execution:
            result['execution_time_ms'] = self._simulate_latency()
        
        # محاسبه قیمت اجرا
        execution_price = self._calculate_execution_price(
            order, market_data, market_condition
        )
        
        # محاسبه slippage
        slippage = 0
        if self.settings.enable_slippage:
            slippage = self._calculate_slippage(
                order, market_data, market_condition
            )
            execution_price += slippage
        
        # محاسبه spread
        if self.settings.enable_spread:
            spread_cost = self._calculate_spread_cost(
                order, market_data, market_condition
            )
            execution_price += spread_cost
        
        # محاسبه کمیسیون
        commission = 0
        if self.settings.enable_commission:
            commission = self._calculate_commission(order, execution_price)
        
        # تصمیم نهایی اجرا
        if self._should_execute_order(order, execution_price, market_condition):
            result.update({
                'executed': True,
                'executed_price': execution_price,
                'executed_size': order['size'],
                'commission': commission,
                'slippage': slippage,
                'total_cost': (execution_price * order['size']) + commission
            })
        else:
            result['reject_reason'] = 'INSUFFICIENT_LIQUIDITY'
        
        return result

    def simulate_market_hours(self, timestamp: datetime) -> bool:
        """شبیه‌سازی ساعات بازار (کریپتو 24/7 اما با نوسانات نقدینگی)"""
        
        # کریپتو 24/7 است اما نقدینگی متغیر
        hour = timestamp.hour
        weekday = timestamp.weekday()
        
        # ساعات کم نقدینگی (شب آمریکا + آخر هفته)
        if (0 <= hour <= 6) or (weekday >= 5):  # شنبه = 5, یکشنبه = 6
            return False  # نقدینگی کم
        
        return True  # نقدینگی عادی

    def simulate_gap_events(self, current_data: pd.Series, 
                           previous_data: pd.Series) -> Optional[Dict]:
        """شبیه‌سازی رویدادهای gap"""
        
        if not self.settings.enable_weekend_gaps:
            return None
        
        # محاسبه gap
        gap_percentage = (current_data['open'] - previous_data['close']) / previous_data['close']
        
        # threshold برای gap قابل توجه
        significant_gap_threshold = 0.02  # 2%
        
        if abs(gap_percentage) > significant_gap_threshold:
            return {
                'type': 'weekend_gap' if abs(gap_percentage) > 0.05 else 'news_gap',
                'direction': 'up' if gap_percentage > 0 else 'down',
                'magnitude': abs(gap_percentage),
                'impact_on_stops': True
            }
        
        return None

    def _validate_order(self, order: Dict, market_data: pd.Series) -> Dict:
        """اعتبارسنجی سفارش"""
        
        # بررسی حداقل اندازه سفارش
        min_order_size = 0.001  # حداقل 0.001 واحد
        if order.get('size', 0) < min_order_size:
            return {'valid': False, 'reason': 'BELOW_MIN_SIZE'}
        
        # بررسی حداکثر اندازه سفارش (بر اساس نقدینگی)
        if 'volume' in market_data.index:
            max_order_ratio = 0.05  # حداکثر 5% از volume
            max_order_size = market_data['volume'] * max_order_ratio
            if order.get('size', 0) > max_order_size:
                return {'valid': False, 'reason': 'EXCEEDS_VOLUME_LIMIT'}
        
        # بررسی قیمت limit order
        if order.get('type') == 'limit':
            price = order.get('price', 0)
            current_price = market_data['close']
            
            # بررسی انحراف زیاد از قیمت بازار
            max_deviation = 0.1  # 10%
            if abs(price - current_price) / current_price > max_deviation:
                return {'valid': False, 'reason': 'PRICE_TOO_FAR_FROM_MARKET'}
        
        return {'valid': True, 'reason': None}

    def _simulate_latency(self) -> int:
        """شبیه‌سازی تأخیر شبکه"""
        # توزیع normal برای latency
        base_latency = self.settings.latency_ms
        variation = base_latency * 0.3  # 30% variation
        
        latency = max(1, int(np.random.normal(base_latency, variation)))
        
        # گاهی spike های latency
        if random.random() < 0.05:  # 5% احتمال
            latency *= random.randint(2, 5)
        
        return latency

    def _calculate_execution_price(self, order: Dict, market_data: pd.Series,
                                  market_condition: MarketCondition) -> float:
        """محاسبه قیمت اجرا"""
        
        if order.get('type') == 'market':
            # Market order: قیمت ask/bid
            if order.get('side') == 'buy':
                # خرید: ask price (بالاتر از mid)
                base_price = market_data['close']
                spread = self.typical_spreads.get(order.get('symbol', ''), 0.01)
                return base_price + (spread / 2) * market_condition.spread
            else:
                # فروش: bid price (پایین‌تر از mid)
                base_price = market_data['close']
                spread = self.typical_spreads.get(order.get('symbol', ''), 0.01)
                return base_price - (spread / 2) * market_condition.spread
        
        elif order.get('type') == 'limit':
            # Limit order: اگر قیمت مناسب باشد
            requested_price = order.get('price', market_data['close'])
            current_price = market_data['close']
            
            if order.get('side') == 'buy' and requested_price >= current_price:
                return requested_price
            elif order.get('side') == 'sell' and requested_price <= current_price:
                return requested_price
            else:
                # Limit order اجرا نمی‌شود
                return 0
        
        return market_data['close']

    def _calculate_slippage(self, order: Dict, market_data: pd.Series,
                           market_condition: MarketCondition) -> float:
        """محاسبه slippage"""
        
        size = order.get('size', 0)
        price = market_data['close']
        
        # انتخاب مدل slippage
        slippage_model = self.slippage_models.get('market_impact', self._linear_slippage)
        
        base_slippage = slippage_model(size, price, market_data)
        
        # تأثیر شرایط بازار
        adjusted_slippage = base_slippage * market_condition.slippage_factor
        
        # جهت slippage بر اساس side
        direction = 1 if order.get('side') == 'buy' else -1
        
        return adjusted_slippage * direction

    def _linear_slippage(self, size: float, price: float, market_data: pd.Series) -> float:
        """مدل خطی slippage"""
        # slippage متناسب با اندازه سفارش
        size_factor = size / 1000  # فرض: 1000 واحد = baseline
        base_slippage_bps = 1.0  # 1 basis point
        
        return price * (base_slippage_bps / 10000) * size_factor

    def _sqrt_slippage(self, size: float, price: float, market_data: pd.Series) -> float:
        """مدل square root slippage"""
        size_factor = np.sqrt(size / 1000)
        base_slippage_bps = 1.0
        
        return price * (base_slippage_bps / 10000) * size_factor

    def _market_impact_slippage(self, size: float, price: float, market_data: pd.Series) -> float:
        """مدل market impact slippage"""
        
        # تخمین daily volume
        if 'volume' in market_data.index:
            daily_volume = market_data['volume'] * 24  # فرض: داده ساعتی
        # تخمین daily volume
        if 'volume' in market_data.index:
            daily_volume = market_data['volume'] * 24  # فرض: داده ساعتی
            
            # نسبت اندازه سفارش به حجم روزانه
            volume_ratio = size / daily_volume if daily_volume > 0 else 0
            
            # market impact بر اساس مدل empirical
            impact_bps = 0.5 * (volume_ratio ** 0.5) * 10000  # basis points
            
            return price * (impact_bps / 10000)
        
        # fallback به مدل ساده
        return self._linear_slippage(size, price, market_data)

    def _calculate_spread_cost(self, order: Dict, market_data: pd.Series,
                              market_condition: MarketCondition) -> float:
        """محاسبه هزینه spread"""
        
        symbol = order.get('symbol', '')
        base_spread = self.typical_spreads.get(symbol, 0.01)
        
        # تنظیم spread بر اساس شرایط بازار
        adjusted_spread = base_spread * market_condition.spread
        
        # فقط نیمی از spread (چون قیمت اجرا قبلاً شامل نیم spread است)
        return adjusted_spread / 2

    def _calculate_commission(self, order: Dict, execution_price: float) -> float:
        """محاسبه کمیسیون"""
        
        exchange = order.get('exchange', 'bybit')
        order_type = order.get('type', 'market')
        
        # انتخاب نرخ کمیسیون
        rates = self.commission_rates.get(exchange, self.commission_rates['bybit'])
        
        if order_type == 'limit':
            commission_rate = rates['maker']
        else:
            commission_rate = rates['taker']
        
        trade_value = execution_price * order.get('size', 0)
        return trade_value * commission_rate

    def _should_execute_order(self, order: Dict, execution_price: float,
                             market_condition: MarketCondition) -> bool:
        """تصمیم‌گیری برای اجرای سفارش"""
        
        # بررسی نقدینگی
        if market_condition.liquidity < 0.3:  # نقدینگی بسیار کم
            return random.random() < 0.7  # 70% احتمال اجرا
        
        # بررسی volatility
        if market_condition.volatility > 3.0:  # نوسانات بالا
            return random.random() < 0.9  # 90% احتمال اجرا
        
        # اجرای عادی
        return True

    def simulate_partial_fills(self, order: Dict, market_condition: MarketCondition) -> List[Dict]:
        """شبیه‌سازی اجرای جزئی سفارش‌ها"""
        
        fills = []
        remaining_size = order.get('size', 0)
        
        # تعیین تعداد اجراهای جزئی
        if market_condition.liquidity < 0.5:
            num_fills = random.randint(2, 5)
        else:
            num_fills = 1  # اجرای یکجا
        
        for i in range(num_fills):
            if remaining_size <= 0:
                break
            
            # اندازه هر اجرای جزئی
            if i == num_fills - 1:  # آخرین اجرا
                fill_size = remaining_size
            else:
                max_fill = remaining_size * 0.8
                fill_size = random.uniform(remaining_size * 0.2, max_fill)
            
            # قیمت متغیر برای هر اجرا
            price_variation = random.uniform(-0.001, 0.001)  # تا 0.1% تغییر
            base_price = order.get('price', 0)
            fill_price = base_price * (1 + price_variation)
            
            fills.append({
                'fill_id': f"{order.get('id', 'order')}_{i+1}",
                'size': fill_size,
                'price': fill_price,
                'timestamp': datetime.now(),
                'commission': self._calculate_commission({'size': fill_size}, fill_price)
            })
            
            remaining_size -= fill_size
        
        return fills

    def simulate_order_book_impact(self, order: Dict, market_data: pd.Series) -> Dict:
        """شبیه‌سازی تأثیر سفارش روی order book"""
        
        # شبیه‌سازی ساده order book
        mid_price = market_data['close']
        spread = self.typical_spreads.get(order.get('symbol', ''), 0.01)
        
        # تخمین depth در هر سطح قیمتی
        base_depth = order.get('size', 0) * 10  # فرض: 10x depth
        
        levels = []
        for i in range(5):  # 5 سطح
            price_offset = spread * (i + 1)
            
            if order.get('side') == 'buy':
                level_price = mid_price - price_offset
                available_size = base_depth * (1 - i * 0.2)  # کاهش depth در سطوح دورتر
            else:
                level_price = mid_price + price_offset
                available_size = base_depth * (1 - i * 0.2)
            
            levels.append({
                'price': level_price,
                'size': max(0, available_size),
                'level': i + 1
            })
        
        return {
            'symbol': order.get('symbol', ''),
            'side': 'bids' if order.get('side') == 'buy' else 'asks',
            'levels': levels,
            'spread': spread,
            'mid_price': mid_price
        }

    def simulate_network_issues(self) -> Optional[Dict]:
        """شبیه‌سازی مشکلات شبکه"""
        
        # احتمال مشکلات مختلف
        issues = [
            {'type': 'high_latency', 'probability': 0.1, 'impact': 'delay'},
            {'type': 'connection_drop', 'probability': 0.02, 'impact': 'reconnect'},
            {'type': 'api_rate_limit', 'probability': 0.05, 'impact': 'throttle'},
            {'type': 'exchange_maintenance', 'probability': 0.001, 'impact': 'halt'}
        ]
        
        for issue in issues:
            if random.random() < issue['probability']:
                return {
                    'issue_type': issue['type'],
                    'impact': issue['impact'],
                    'duration_seconds': self._get_issue_duration(issue['type']),
                    'suggested_action': self._get_suggested_action(issue['type'])
                }
        
        return None

    def _get_issue_duration(self, issue_type: str) -> int:
        """مدت زمان مشکل بر اساس نوع"""
        durations = {
            'high_latency': random.randint(10, 60),
            'connection_drop': random.randint(5, 30),
            'api_rate_limit': random.randint(60, 300),
            'exchange_maintenance': random.randint(300, 1800)
        }
        return durations.get(issue_type, 30)

    def _get_suggested_action(self, issue_type: str) -> str:
        """پیشنهاد اقدام برای هر نوع مشکل"""
        actions = {
            'high_latency': 'wait_and_retry',
            'connection_drop': 'reconnect',
            'api_rate_limit': 'reduce_frequency',
            'exchange_maintenance': 'use_backup_exchange'
        }
        return actions.get(issue_type, 'wait')

    def generate_realistic_market_conditions(self, timestamp: datetime, 
                                           base_conditions: MarketCondition = None) -> MarketCondition:
        """تولید شرایط واقعی بازار"""
        
        if base_conditions is None:
            base_conditions = MarketCondition()
        
        # تأثیر ساعت روز
        hour = timestamp.hour
        weekday = timestamp.weekday()
        
        # ضرایب بر اساس زمان
        time_factors = self._get_time_factors(hour, weekday)
        
        # شرایط تصادفی
        volatility_noise = random.uniform(0.8, 1.2)
        liquidity_noise = random.uniform(0.9, 1.1)
        
        return MarketCondition(
            volatility=base_conditions.volatility * time_factors['volatility'] * volatility_noise,
            liquidity=base_conditions.liquidity * time_factors['liquidity'] * liquidity_noise,
            spread=base_conditions.spread * time_factors['spread'],
            slippage_factor=base_conditions.slippage_factor * time_factors['slippage'],
            market_impact=base_conditions.market_impact * time_factors['impact']
        )

    def _get_time_factors(self, hour: int, weekday: int) -> Dict[str, float]:
        """ضرایب زمانی برای شرایط بازار"""
        
        # ساعات پرترافیک (اروپا + آمریکا)
        if 12 <= hour <= 22:  # UTC
            return {
                'volatility': 1.2,
                'liquidity': 1.3,
                'spread': 0.8,
                'slippage': 0.9,
                'impact': 0.8
            }
        
        # ساعات آرام (شب آمریکا)
        elif 2 <= hour <= 6:
            return {
                'volatility': 0.7,
                'liquidity': 0.6,
                'spread': 1.5,
                'slippage': 1.4,
                'impact': 1.3
            }
        
        # آخر هفته
        elif weekday >= 5:  # شنبه یکشنبه
            return {
                'volatility': 0.8,
                'liquidity': 0.7,
                'spread': 1.2,
                'slippage': 1.1,
                'impact': 1.1
            }
        
        # شرایط عادی
        else:
            return {
                'volatility': 1.0,
                'liquidity': 1.0,
                'spread': 1.0,
                'slippage': 1.0,
                'impact': 1.0
            }

    def create_execution_report(self, executions: List[Dict]) -> Dict:
        """تولید گزارش اجرا"""
        
        if not executions:
            return {'total_executions': 0}
        
        total_size = sum(ex.get('executed_size', 0) for ex in executions)
        total_commission = sum(ex.get('commission', 0) for ex in executions)
        total_slippage = sum(ex.get('slippage', 0) for ex in executions)
        
        successful_executions = [ex for ex in executions if ex.get('executed', False)]
        failed_executions = [ex for ex in executions if not ex.get('executed', False)]
        
        avg_latency = np.mean([ex.get('execution_time_ms', 0) for ex in executions])
        
        return {
            'total_executions': len(executions),
            'successful_executions': len(successful_executions),
            'failed_executions': len(failed_executions),
            'success_rate': len(successful_executions) / len(executions) * 100,
            'total_size': total_size,
            'total_commission': total_commission,
            'total_slippage': total_slippage,
            'avg_execution_latency_ms': avg_latency,
            'failure_reasons': [ex.get('reject_reason') for ex in failed_executions if ex.get('reject_reason')]
        }
