# main.py - EUR/USD Forex Trading Bot

import asyncio
import sys
import os
import traceback
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

# اضافه کردن مسیر پروژه
project_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_path)

# Import modules
import config
from connectors.forex_connector import ForexConnector
from analyzers import AnalysisManager
from models.feature_engineering import ForexFeatureExtractor
from forex_telegram_notifier import ForexTelegramNotifier
from learning_system import AdaptiveLearningSystem

# تنظیم logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('forex_bot.log'),
        logging.StreamHandler()
    ]
)

class EURUSDForexBot:
    """
    ربات معاملات Forex EUR/USD با سیستم یادگیری تطبیقی
    """
    
    def __init__(self):
        # تنظیمات اولیه
        self.running = False
        self.pair = config.PRIMARY_PAIR
        self.display_pair = config.DISPLAY_PAIR
        
        # کامپوننت‌های اصلی
        self.forex_connector = None
        self.analysis_manager = None
        self.feature_extractor = None
        self.notifier = None
        self.learning_system = None
        
        # ردیابی سیگنال‌های فعال
        self.active_signals = []
        
        # معیارهای عملکرد
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'total_pips': 0.0,
            'largest_win_pips': 0.0,
            'largest_loss_pips': 0.0,
            'consecutive_wins': 0,
            'consecutive_losses': 0,
            'average_trade_duration_minutes': 0.0
        }
        
        # آمار روزانه
        self.daily_stats = {
            'date': datetime.now().date(),
            'signals_sent': 0,
            'profit_loss_pips': 0.0,
            'trades_by_session': {
                'ASIAN': {'count': 0, 'profit': 0.0},
                'LONDON': {'count': 0, 'profit': 0.0},
                'NEW_YORK': {'count': 0, 'profit': 0.0},
                'OVERLAP': {'count': 0, 'profit': 0.0}
            },
            'session_performance': {
                'ASIAN': {'trades': 0, 'profit': 0.0, 'win_rate': 0.0},
                'LONDON': {'trades': 0, 'profit': 0.0, 'win_rate': 0.0},
                'NEW_YORK': {'trades': 0, 'profit': 0.0, 'win_rate': 0.0},
                'OVERLAP': {'trades': 0, 'profit': 0.0, 'win_rate': 0.0}
            }
        }
        
        # مدیریت ریسک
        self.risk_manager = {
            'daily_loss_pips': 0.0,
            'consecutive_losses': 0,
            'max_daily_loss_reached': False,
            'emergency_stop_active': False,
            'last_trade_time': None,
            'current_exposure': 0.0
        }

    async def initialize(self) -> bool:
        """راه‌اندازی اولیه ربات"""
        try:
            logging.info("🚀 شروع راه‌اندازی EUR/USD Forex Bot...")
            
            # راه‌اندازی اتصال Forex
            self.forex_connector = ForexConnector()
            if not await self.forex_connector.initialize():
                logging.error("❌ خطا در اتصال به OANDA")
                return False
            logging.info("✅ اتصال به OANDA برقرار شد")
            
            # راه‌اندازی تحلیلگر
            self.analysis_manager = AnalysisManager()
            logging.info("✅ سیستم تحلیل آماده شد")
            
            # راه‌اندازی استخراج ویژگی
            self.feature_extractor = ForexFeatureExtractor()
            logging.info("✅ استخراج ویژگی آماده شد")
            
            # راه‌اندازی اطلاع‌رسان
            self.notifier = ForexTelegramNotifier(
                token=config.TELEGRAM_BOT_TOKEN,
                chat_id=config.TELEGRAM_CHAT_ID
            )
            
            if not await self.notifier.test_connection():
                logging.error("❌ خطا در اتصال تلگرام")
                return False
            logging.info("✅ اتصال تلگرام برقرار شد")
            
            # راه‌اندازی سیستم یادگیری
            if config.LEARNING_ENABLED:
                self.learning_system = AdaptiveLearningSystem()
                logging.info("✅ سیستم یادگیری فعال شد")
            else:
                logging.info("ℹ️ سیستم یادگیری غیرفعال")
            
            # ارسال پیام راه‌اندازی
            await self._send_startup_message()
            
            return True
            
        except Exception as e:
            logging.error(f"❌ خطا در راه‌اندازی: {e}")
            logging.error(traceback.format_exc())
            return False

    async def run(self):
        """حلقه اصلی اجرای ربات"""
        self.running = True
        
        try:
            logging.info("🔄 شروع حلقه اصلی ربات...")
            
            while self.running:
                try:
                    # بررسی تاریخ جدید
                    await self._check_new_day()
                    
                    # بررسی emergency stops
                    if await self._check_emergency_stops():
                        logging.warning("⚠️ Emergency stop فعال - انتظار...")
                        await asyncio.sleep(300)  # 5 دقیقه انتظار
                        continue
                    
                    # اجرای تحلیل اصلی
                    await self._run_main_analysis()
                    
                    # ردیابی عملکرد سیگنال‌های فعال
                    await self._track_active_signals()
                    
                    # انتظار تا تحلیل بعدی
                    await asyncio.sleep(config.ANALYSIS_INTERVAL_MINUTES * 60)
                    
                except KeyboardInterrupt:
                    logging.info("⏹️ توقف ربات توسط کاربر")
                    break
                except Exception as e:
                    logging.error(f"❌ خطا در حلقه اصلی: {e}")
                    await self._send_error_alert(str(e))
                    await asyncio.sleep(60)
            
        finally:
            await self._shutdown()

    async def _run_main_analysis(self):
        """اجرای تحلیل اصلی"""
        try:
            logging.info(f"📊 شروع تحلیل {self.display_pair}...")
            
            # دریافت داده‌های قیمتی
            df = await self.forex_connector.get_historical_data(
                instrument=self.pair,
                timeframe=config.TIMEFRAME,
                count=config.ANALYSIS_CANDLES_COUNT
            )
            
            if df is None or len(df) < 100:
                logging.warning("⚠️ داده کافی برای تحلیل موجود نیست")
                return
            
            # استخراج ویژگی‌های فنی
            enhanced_df = self.feature_extractor.extract_features(df)
            
            # اجرای تحلیل جامع
            analysis_result = await self.analysis_manager.run_comprehensive_analysis(
                enhanced_df, self.pair
            )
            
            if not analysis_result:
                logging.info("ℹ️ هیچ سیگنال معتبری یافت نشد")
                return
            
            # بررسی کیفیت سیگنال
            if not self._validate_signal_quality(analysis_result):
                logging.info("⚠️ کیفیت سیگنال پایین است")
                return
            
            # اعمال یادگیری (اگر فعال باشد)
            if self.learning_system:
                analysis_result = await self._apply_learning_adjustments(analysis_result)
            
            # بررسی مدیریت ریسک
            if not self._check_risk_management(analysis_result):
                logging.warning("⚠️ سیگنال به دلیل مدیریت ریسک رد شد")
                return
            
            # تعیین جلسه معاملاتی
            market_session = self._get_current_market_session()
            analysis_result['market_session'] = market_session
            
            # محاسبه اندازه پوزیشن
            position_data = self._calculate_position_size(
                analysis_result['confidence'],
                analysis_result['risk_reward_ratio'],
                market_session
            )
            
            # آماده‌سازی سیگنال
            signal_data = self._prepare_signal_data(analysis_result, position_data)
            
            # ارسال سیگنال
            success = await self.notifier.send_forex_signal(signal_data)
            
            if success:
                # ثبت سیگنال در لیست فعال
                self.active_signals.append(signal_data)
                
                # آپدیت آمار
                self._update_daily_stats(signal_data)
                
                logging.info(f"✅ سیگنال {signal_data['direction']} ارسال شد")
            
        except Exception as e:
            logging.error(f"❌ خطا در تحلیل اصلی: {e}")
            logging.error(traceback.format_exc())

    def _validate_signal_quality(self, analysis_result: Dict) -> bool:
        """اعتبارسنجی کیفیت سیگنال"""
        try:
            # بررسی حداقل اطمینان
            if analysis_result.get('confidence', 0) < config.MIN_SIGNAL_CONFIDENCE:
                return False
            
            # بررسی قدرت سیگنال
            if analysis_result.get('signal_strength', 0) < config.MIN_SIGNAL_STRENGTH:
                return False
            
            # بررسی نسبت ریسک/ریوارد
            if analysis_result.get('risk_reward_ratio', 0) < config.MIN_RISK_REWARD_RATIO:
                return False
            
            # بررسی اجماع AI
            ai_consensus = analysis_result.get('ai_evaluation', {}).get('consensus_strength', 0)
            if ai_consensus < config.MIN_AI_CONSENSUS:
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"خطا در اعتبارسنجی سیگنال: {e}")
            return False

    async def _apply_learning_adjustments(self, analysis_result: Dict) -> Dict:
        """اعمال تنظیمات یادگیری"""
        try:
            if not self.learning_system:
                return analysis_result
            
            # پیش‌بینی احتمال موفقیت
            success_probability = self.learning_system.predict_signal_success(analysis_result)
            
            # دریافت پارامترهای تطبیقی
            adaptive_params = self.learning_system.get_adaptive_parameters(self.pair)
            
            # تنظیم آستانه اطمینان
            adjusted_confidence = analysis_result['confidence'] * adaptive_params.get('confidence_multiplier', 1.0)
            
            # اعمال تنظیمات
            analysis_result['adjusted_confidence'] = adjusted_confidence
            analysis_result['success_probability'] = success_probability
            analysis_result['learning_applied'] = True
            
            logging.info(f"🧠 یادگیری اعمال شد - احتمال موفقیت: {success_probability:.2f}")
            
            return analysis_result
            
        except Exception as e:
            logging.error(f"خطا در اعمال یادگیری: {e}")
            return analysis_result

    def _check_risk_management(self, analysis_result: Dict) -> bool:
        """بررسی مدیریت ریسک"""
        try:
            # بررسی حداکثر ضرر روزانه
            if self.risk_manager['daily_loss_pips'] <= -config.MAX_DAILY_LOSS_PERCENT * 1000:
                logging.warning("⚠️ حداکثر ضرر روزانه رسیده")
                return False
            
            # بررسی ضررهای متوالی
            if self.risk_manager['consecutive_losses'] >= config.MAX_CONSECUTIVE_LOSSES:
                logging.warning("⚠️ حداکثر ضرر متوالی رسیده")
                return False
            
            # بررسی فاصله زمانی از آخرین معامله
            if self.risk_manager['last_trade_time']:
                time_since_last = datetime.now() - self.risk_manager['last_trade_time']
                if time_since_last < timedelta(minutes=30):
                    logging.info("ℹ️ فاصله زمانی کافی از آخرین معامله نیست")
                    return False
            
            # بررسی exposure فعلی
            if len(self.active_signals) >= config.EMERGENCY_STOPS['max_concurrent_signals']:
                logging.warning("⚠️ حداکثر سیگنال همزمان رسیده")
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"خطا در بررسی ریسک: {e}")
            return True  # در صورت خطا، اجازه ادامه

    def _get_current_market_session(self) -> str:
        """تعیین جلسه معاملاتی فعلی"""
        try:
            now_utc = datetime.utcnow()
            hour = now_utc.hour
            
            # جلسه آسیا
            if 22 <= hour or hour <= 7:
                return 'ASIAN'
            # جلسه لندن
            elif 7 <= hour < 16:
                return 'LONDON'
            # جلسه نیویورک
            elif 13 <= hour < 22:
                # بررسی همپوشانی
                if 13 <= hour < 16:
                    return 'OVERLAP'
                else:
                    return 'NEW_YORK'
            
            return 'LONDON'  # پیش‌فرض
            
        except Exception as e:
            logging.error(f"خطا در تعیین جلسه: {e}")
            return 'LONDON'

    def _calculate_position_size(self, confidence: float, risk_reward: float, session: str) -> Dict:
        """محاسبه اندازه پوزیشن"""
        try:
            # اندازه پایه بر اساس ریسک
            base_risk = config.RISK_PER_TRADE_PERCENT
            
            # تنظیم بر اساس اطمینان
            confidence_multiplier = min(confidence * 1.5, 1.0)
            
            # تنظیم بر اساس risk/reward
            rr_multiplier = min(risk_reward / 2.0, 1.2)
            
            # تنظیم بر اساس جلسه
            session_multiplier = config.SESSION_STRENGTH.get(session, 1.0)
            
            # محاسبه نهایی
            final_risk = base_risk * confidence_multiplier * rr_multiplier * session_multiplier
            final_risk = min(final_risk, config.MAX_POSITION_SIZE_PERCENT)
            
            # تعیین leverage
            leverage = min(int(final_risk * 500), config.MAX_LEVERAGE)
            leverage = max(leverage, 5)  # حداقل 5x
            
            return {
                'risk_percent': final_risk,
                'leverage': leverage,
                'confidence_multiplier': confidence_multiplier,
                'rr_multiplier': rr_multiplier,
                'session_multiplier': session_multiplier
            }
            
        except Exception as e:
            logging.error(f"خطا در محاسبه اندازه پوزیشن: {e}")
            return {
                'risk_percent': config.RISK_PER_TRADE_PERCENT,
                'leverage': config.DEFAULT_LEVERAGE
            }

    def _prepare_signal_data(self, analysis_result: Dict, position_data: Dict) -> Dict:
        """آماده‌سازی داده‌های سیگنال"""
        try:
            signal_id = f"EUR_USD_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # تعیین جهت
            direction = 'BUY' if analysis_result['signal'] > 0 else 'SELL'
            
            # محاسبه قیمت‌های ورود، SL و TP
            current_price = analysis_result['current_price']
            stop_loss = analysis_result['stop_loss']
            take_profits = analysis_result['take_profits']
            
            # محاسبه PIP
            pip_value = config.EUR_USD_PIP_VALUE
            if direction == 'BUY':
                sl_pips = (current_price - stop_loss) / pip_value
                tp_pips = [(tp - current_price) / pip_value for tp in take_profits]
            else:
                sl_pips = (stop_loss - current_price) / pip_value
                tp_pips = [(current_price - tp) / pip_value for tp in take_profits]
            
            signal_data = {
                'signal_id': signal_id,
                'timestamp': datetime.now(),
                'pair': self.display_pair,
                'direction': direction,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profits': take_profits,
                'stop_loss_pips': abs(sl_pips),
                'take_profit_pips': [abs(tp) for tp in tp_pips],
                'leverage': position_data['leverage'],
                'risk_percent': position_data['risk_percent'],
                'confidence': analysis_result['confidence'],
                'signal_strength': analysis_result.get('signal_strength', 0),
                'signal_type': analysis_result.get('signal_type', 'UNKNOWN'),
                'risk_reward_ratio': analysis_result['risk_reward_ratio'],
                'market_session': analysis_result.get('market_session', 'UNKNOWN'),
                'reasoning': analysis_result.get('reasoning', []),
                'market_context': analysis_result.get('market_context', {}),
                'ai_evaluation': analysis_result.get('ai_evaluation', {}),
                'learning_applied': analysis_result.get('learning_applied', False),
                'success_probability': analysis_result.get('success_probability', 0.7),
                'pip_value': pip_value
            }
            
            return signal_data
            
        except Exception as e:
            logging.error(f"خطا در آماده‌سازی سیگنال: {e}")
            return {}

    async def _track_active_signals(self):
        """ردیابی سیگنال‌های فعال"""
        try:
            if not self.active_signals:
                return
            
            # بررسی هر سیگنال فعال
            for signal in self.active_signals[:]:  # کپی لیست
                try:
                    # بررسی زمان سیگنال (بستن پس از 4 ساعت)
                    signal_age = datetime.now() - signal['timestamp']
                    if signal_age > timedelta(hours=4):
                        await self._close_signal(signal, 'TIMEOUT')
                        continue
                    
                    # دریافت قیمت فعلی
                    current_price = await self.forex_connector.get_current_price(self.pair)
                    if current_price is None:
                        continue
                    
                    # بررسی SL/TP
                    if signal['direction'] == 'BUY':
                        # بررسی Stop Loss
                        if current_price <= signal['stop_loss']:
                            await self._close_signal(signal, 'STOP_LOSS', current_price)
                            continue
                        
                        # بررسی Take Profit
                        for i, tp in enumerate(signal['take_profits']):
                            if current_price >= tp:
                                await self._close_signal(signal, f'TAKE_PROFIT_{i+1}', current_price)
                                break
                    
                    else:  # SELL
                        # بررسی Stop Loss
                        if current_price >= signal['stop_loss']:
                            await self._close_signal(signal, 'STOP_LOSS', current_price)
                            continue
                        
                        # بررسی Take Profit
                        for i, tp in enumerate(signal['take_profits']):
                            if current_price <= tp:
                                await self._close_signal(signal, f'TAKE_PROFIT_{i+1}', current_price)
                                break
                
                except Exception as e:
                    logging.error(f"خطا در ردیابی سیگنال {signal.get('signal_id', 'UNKNOWN')}: {e}")
                    continue
            
        except Exception as e:
            logging.error(f"خطا در ردیابی سیگنال‌های فعال: {e}")

    async def _close_signal(self, signal: Dict, close_reason: str, exit_price: float = None):
        """بستن سیگنال و ثبت نتیجه"""
        try:
            if exit_price is None:
                exit_price = await self.forex_connector.get_current_price(self.pair)
            
            if exit_price is None:
                logging.error("نمی‌توان قیمت خروج را دریافت کرد")
                return
            
            # محاسبه PnL
            entry_price = signal['entry_price']
            pip_value = signal['pip_value']
            
            if signal['direction'] == 'BUY':
                pips_change = (exit_price - entry_price) / pip_value
            else:  # SELL
                pips_change = (entry_price - exit_price) / pip_value
            
            # تعیین موفقیت
            success = pips_change > 0
            
            # آپدیت معیارهای عملکرد
            await self._update_performance_metrics(signal, pips_change, success, close_reason)
            
            # ارسال نتیجه
            result_data = {
                'signal_id': signal['signal_id'],
                'pair': signal['pair'],
                'direction': signal['direction'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pips_result': pips_change,
                'success': success,
                'close_reason': close_reason,
                'duration': datetime.now() - signal['timestamp']
            }
            
            await self.notifier.send_signal_result(result_data)
            
            # حذف از لیست فعال
            if signal in self.active_signals:
                self.active_signals.remove(signal)
            
            # ثبت در سیستم یادگیری
            if self.learning_system:
                self.learning_system.record_signal_result(signal, result_data)
            
            logging.info(f"🎯 سیگنال بسته شد: {pips_change:+.1f} pips ({close_reason})")
            
        except Exception as e:
            logging.error(f"خطا در بستن سیگنال: {e}")

    async def _update_performance_metrics(self, signal: Dict, pips_change: float, success: bool, close_reason: str):
        """آپدیت معیارهای عملکرد"""
        try:
            self.performance_metrics['total_trades'] += 1
            self.performance_metrics['total_pips'] += pips_change
            
            if success:
                self.performance_metrics['winning_trades'] += 1
                self.performance_metrics['consecutive_wins'] += 1
                self.performance_metrics['consecutive_losses'] = 0
                
                if pips_change > self.performance_metrics['largest_win_pips']:
                    self.performance_metrics['largest_win_pips'] = pips_change
            else:
                self.performance_metrics['losing_trades'] += 1
                self.performance_metrics['consecutive_losses'] += 1
                self.performance_metrics['consecutive_wins'] = 0
                
                if pips_change < self.performance_metrics['largest_loss_pips']:
                    self.performance_metrics['largest_loss_pips'] = pips_change
            
            # محاسبه win rate
            total = self.performance_metrics['total_trades']
            wins = self.performance_metrics['winning_trades']
            self.performance_metrics['win_rate'] = wins / total if total > 0 else 0
            
            # آپدیت آمار جلسه
            session = signal.get('market_session', 'UNKNOWN')
            if session in self.daily_stats['session_performance']:
                self.daily_stats['session_performance'][session]['trades'] += 1
                self.daily_stats['session_performance'][session]['profit'] += pips_change
            
            # آپدیت ریسک منیجر
            self.risk_manager['daily_loss_pips'] += pips_change
            if not success:
                self.risk_manager['consecutive_losses'] += 1
            else:
                self.risk_manager['consecutive_losses'] = 0
            
            self.risk_manager['last_trade_time'] = datetime.now()
            
        except Exception as e:
            logging.error(f"خطا در آپدیت عملکرد: {e}")

    async def _check_emergency_stops(self) -> bool:
        """بررسی شرایط توقف اضطراری"""
        try:
            # بررسی حداکثر ضرر روزانه
            daily_loss_limit = config.MAX_DAILY_LOSS_PERCENT * 1000  # تبدیل به pips
            if self.risk_manager['daily_loss_pips'] <= -daily_loss_limit:
                if not self.risk_manager['emergency_stop_active']:
                    await self.notifier.send_emergency_alert({
                        'type': 'DAILY_LOSS_LIMIT',
                        'current_loss': self.risk_manager['daily_loss_pips'],
                        'limit': daily_loss_limit
                    })
                    self.risk_manager['emergency_stop_active'] = True
                return True
            
            # بررسی ضررهای متوالی
            if self.risk_manager['consecutive_losses'] >= config.MAX_CONSECUTIVE_LOSSES:
                if not self.risk_manager['emergency_stop_active']:
                    await self.notifier.send_emergency_alert({
                        'type': 'CONSECUTIVE_LOSSES',
                        'count': self.risk_manager['consecutive_losses']
                    })
                    self.risk_manager['emergency_stop_active'] = True
                return True
            
            # ریست emergency stop اگر شرایط بهبود یافت
            if self.risk_manager['emergency_stop_active']:
                if (self.risk_manager['daily_loss_pips'] > -daily_loss_limit/2 and 
                    self.risk_manager['consecutive_losses'] < config.MAX_CONSECUTIVE_LOSSES):
                    self.risk_manager['emergency_stop_active'] = False
                    logging.info("✅ Emergency stop غیرفعال شد")
            
            return False
            
        except Exception as e:
            logging.error(f"خطا در بررسی emergency stops: {e}")
            return False

    async def _check_new_day(self):
        """بررسی شروع روز جدید"""
        try:
            today = datetime.now().date()
            if self.daily_stats['date'] != today:
                # ارسال گزارش روز قبل
                await self._send_daily_report()
                
                # ریست آمار روزانه
                self.daily_stats = {
                    'date': today,
                    'signals_sent': 0,
                    'profit_loss_pips': 0.0,
                    'trades_by_session': {
                        'ASIAN': {'count': 0, 'profit': 0.0},
                        'LONDON': {'count': 0, 'profit': 0.0},
                        'NEW_YORK': {'count': 0, 'profit': 0.0},
                        'OVERLAP': {'count': 0, 'profit': 0.0}
                    },
                    'session_performance': {
                        'ASIAN': {'trades': 0, 'profit': 0.0, 'win_rate': 0.0},
                        'LONDON': {'trades': 0, 'profit': 0.0, 'win_rate': 0.0},
                        'NEW_YORK': {'trades': 0, 'profit': 0.0, 'win_rate': 0.0},
                        'OVERLAP': {'trades': 0, 'profit': 0.0, 'win_rate': 0.0}
                    }
                }
                
                # ریست ریسک منیجر روزانه
                self.risk_manager['daily_loss_pips'] = 0.0
                self.risk_manager['emergency_stop_active'] = False
                
                logging.info(f"🆕 شروع روز جدید: {today}")
                
        except Exception as e:
            logging.error(f"خطا در بررسی روز جدید: {e}")

    def _update_daily_stats(self, signal: Dict):
        """آپدیت آمار روزانه"""
        try:
            self.daily_stats['signals_sent'] += 1
            
            session = signal.get('market_session', 'UNKNOWN')
            if session in self.daily_stats['trades_by_session']:
                self.daily_stats['trades_by_session'][session]['count'] += 1
            
        except Exception as e:
            logging.error(f"خطا در آپدیت آمار روزانه: {e}")

    async def _send_startup_message(self):
        """ارسال پیام راه‌اندازی"""
        try:
            startup_data = {
                'timestamp': datetime.now(),
                'pair': self.display_pair,
                'timeframe': config.TIMEFRAME,
                'analysis_interval': config.ANALYSIS_INTERVAL_MINUTES,
                'learning_enabled': config.LEARNING_ENABLED,
                'environment': config.OANDA_ENVIRONMENT
            }
            
            await self.notifier.send_startup_notification(startup_data)
            
        except Exception as e:
            logging.error(f"خطا در ارسال پیام راه‌اندازی: {e}")

    async def _send_daily_report(self):
        """ارسال گزارش روزانه"""
        try:
            report_data = {
                'date': self.daily_stats['date'],
                'daily_stats': self.daily_stats,
                'performance_metrics': self.performance_metrics,
                'risk_metrics': self.risk_manager,
                'learning_stats': self.learning_system.get_daily_stats() if self.learning_system else {}
            }
            
            await self.notifier.send_daily_report(report_data)
            
        except Exception as e:
            logging.error(f"خطا در ارسال گزارش روزانه: {e}")

    async def _send_error_alert(self, error_message: str):
        """ارسال هشدار خطا"""
        try:
            alert_data = {
                'type': 'ERROR',
                'message': error_message,
                'timestamp': datetime.now(),
                'severity': 'HIGH'
            }
            
            await self.notifier.send_error_alert(alert_data)
            
        except Exception as e:
            logging.error(f"خطا در ارسال هشدار خطا: {e}")

    async def _shutdown(self):
        """خاموش کردن ربات"""
        try:
            logging.info("🔄 شروع خاموش کردن ربات...")
            
            self.running = False
            
            # بستن سیگنال‌های فعال
            for signal in self.active_signals[:]:
                await self._close_signal(signal, 'SHUTDOWN')
            
            # ارسال گزارش نهایی
            shutdown_data = {
                'timestamp': datetime.now(),
                'final_performance': self.performance_metrics,
                'active_signals_closed': len(self.active_signals)
            }
            
            await self.notifier.send_shutdown_notification(shutdown_data)
            
            # بستن اتصالات
            if self.forex_connector:
                await self.forex_connector.close()
            
            logging.info("✅ ربات با موفقیت خاموش شد")
            
        except Exception as e:
            logging.error(f"خطا در خاموش کردن ربات: {e}")


async def main():
    """تابع اصلی"""
    bot = None
    
    try:
        # نمایش اطلاعات راه‌اندازی
        print("=" * 60)
        print("🚀 EUR/USD Forex Trading Bot v2.0")
        print("💱 تخصص در جفت ارز EUR/USD")
        print("📊 تحلیل Pure Price Action + AI")
        print("🧠 سیستم یادگیری تطبیقی")
        print("🛡️ مدیریت ریسک پیشرفته")
        print(f"⚡ محیط: {config.OANDA_ENVIRONMENT.upper()}")
        print("=" * 60)
        
        # هشدارهای ایمنی
        print("⚠️ هشدارهای مهم:")
        print("• معاملات فارکس دارای ریسک بالایی است")
        print("• هرگز بیش از توانایی خود سرمایه‌گذاری نکنید")
        print("• این نرم‌افزار صرفاً آموزشی است، نه مشاوره مالی")
        print("• عملکرد گذشته تضمینی برای آینده نیست")
        print("=" * 60)
        
        # بررسی تنظیمات
        config_errors = config.validate_config()
        if config_errors:
            print("❌ خطاهای تنظیمات:")
            for error in config_errors:
                print(f"   • {error}")
            return
        
        # راه‌اندازی ربات
        bot = EURUSDForexBot()
        
        if not await bot.initialize():
            logging.error("❌ خطا در راه‌اندازی ربات")
            return
        
        print("✅ ربات آماده اجرا - Ctrl+C برای توقف")
        print("=" * 60)
        
        # اجرای ربات
        await bot.run()
        
    except KeyboardInterrupt:
        print("\n⏹️ توقف ربات توسط کاربر...")
    except Exception as e:
        logging.error(f"❌ خطای جدی: {e}")
        logging.error(traceback.format_exc())
    finally:
        if bot:
            await bot._shutdown()


if __name__ == "__main__":
    # تنظیم event loop برای Windows
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # اجرای ربات
    asyncio.run(main())
