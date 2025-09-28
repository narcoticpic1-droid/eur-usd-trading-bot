# forex_telegram_notifier.py
import asyncio
from typing import Dict, Optional, List
from telegram import Bot
from telegram.error import TelegramError
import datetime
import json
from utils.logger import setup_logger
import config

class ForexTelegramNotifier:
    """
    سیستم اطلاع‌رسانی تلگرام مخصوص Forex
    """

    def __init__(self, token: str, chat_id: str):
        self.bot = Bot(token=token)
        self.chat_id = chat_id
        self.logger = setup_logger('telegram_notifier')
        self.sent_signals = {}
        self.message_history = []

    async def send_forex_signal(self, signal_data: Dict) -> bool:
        """ارسال سیگنال Forex"""
        try:
            pair = signal_data['pair']
            direction = signal_data['direction']
            entry_price = signal_data['entry_price']
            signal_type = signal_data['signal_type']
            signal_quality = signal_data['signal_quality']
            confidence = signal_data['confidence']
            reasoning = signal_data['reasoning']
            stop_loss = signal_data['stop_loss']
            take_profits = signal_data['take_profits']
            risk_reward_ratio = signal_data['risk_reward_ratio']
            pips_risk = signal_data.get('pips_risk', 0)
            pips_profit = signal_data.get('pips_profit', 0)
            
            # اموجی جفت ارز
            pair_emojis = {
                'EUR_USD': '🇪🇺🇺🇸',
                'GBP_USD': '🇬🇧🇺🇸', 
                'USD_JPY': '🇺🇸🇯🇵',
                'USD_CHF': '🇺🇸🇨🇭',
                'AUD_USD': '🇦🇺🇺🇸'
            }
            pair_emoji = pair_emojis.get(pair, '💱')

            # اموجی کیفیت
            quality_emojis = {
                'HIGH': '🟢',
                'MEDIUM': '🟡',
                'LOW': '🟠'
            }
            quality_emoji = quality_emojis.get(signal_quality, '⚪')

            # اموجی جهت
            direction_emoji = "📈" if direction == 'BUY' else "📉"

            # اموجی نوع سیگنال  
            signal_type_emojis = {
                'TREND_CONTINUATION': '🔥',
                'REVERSAL': '🔄',
                'BREAKOUT': '⚡',
                'PULLBACK': '↩️'
            }
            signal_emoji = signal_type_emojis.get(signal_type, '📊')

            message = f"""{signal_emoji} <b>FOREX SIGNAL</b>

{pair_emoji} <b>جفت ارز:</b> {pair.replace('_', '/')}
{quality_emoji} <b>کیفیت:</b> {signal_quality}
{direction_emoji} <b>جهت:</b> {direction}
💵 <b>قیمت ورود:</b> {entry_price:.5f}

📊 <b>تحلیل:</b>
• نوع: {self._translate_signal_type(signal_type)}
• اطمینان: {confidence:.2f}
• نسبت R/R: {risk_reward_ratio:.2f}

🎯 <b>اهداف سود:</b>
• Target 1: {take_profits[0]:.5f} (+{pips_profit[0]:.0f} pips)
• Target 2: {take_profits[1]:.5f} (+{pips_profit[1]:.0f} pips)
• Target 3: {take_profits[2]:.5f} (+{pips_profit[2]:.0f} pips)

🛑 <b>حد ضرر:</b> {stop_loss:.5f} (-{pips_risk:.0f} pips)

💡 <b>دلایل تحلیل:</b>"""

            # اضافه کردن دلایل
            for reason in reasoning:
                message += f"\n• {reason}"

            # اطلاعات بازار
            market_context = signal_data.get('market_context', {})
            if market_context:
                message += f"\n\n📊 <b>وضعیت بازار:</b>"
                message += f"\n• ترند: {market_context.get('trend', 'N/A')}"
                message += f"\n• قدرت ADX: {market_context.get('adx', 0):.1f}"
                message += f"\n• سشن: {market_context.get('session', 'N/A')}"

            # اطلاعات ریسک
            message += f"\n\n🛡️ <b>مدیریت ریسک:</b>"
            message += f"\n• ریسک: {pips_risk:.0f} pips"
            message += f"\n• سود محتمل: {pips_profit[0]:.0f} pips"
            message += f"\n• اندازه پوزیشن: {signal_data.get('position_size', 0)*100:.1f}%"

            # اطلاعات AI
            ai_evaluation = signal_data.get('ai_evaluation', {})
            if ai_evaluation:
                message += f"\n\n🧠 <b>ارزیابی AI:</b>"
                consensus = ai_evaluation.get('consensus', {})
                if consensus:
                    message += f"\n• توافق: {consensus.get('strength', 0):.1f}/1.0"
                    participating_ais = consensus.get('participating_ais', [])
                    if participating_ais:
                        message += f"\n• AI ها: {', '.join(participating_ais)}"

            message += f"\n\n⏰ <b>زمان:</b> {datetime.datetime.now().strftime('%H:%M:%S UTC')}"
            message += f"\n🏦 <b>بازار:</b> Forex"
            message += f"\n⚠️ <b>توجه:</b> مدیریت ریسک الزامی است"

            # ارسال پیام
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='HTML'
            )

            self.logger.info(f"سیگنال {pair} ارسال شد")
            
            # ذخیره در تاریخچه
            self.message_history.append({
                'pair': pair,
                'timestamp': datetime.datetime.now(),
                'type': 'SIGNAL',
                'data': signal_data
            })

            return True

        except Exception as e:
            self.logger.error(f"خطا در ارسال سیگنال {signal_data.get('pair', 'Unknown')}: {e}")
            return False

    async def send_daily_forex_summary(self, summary_data: Dict) -> bool:
        """ارسال خلاصه روزانه Forex"""
        try:
            date = summary_data['date']
            daily_stats = summary_data['daily_stats']
            pair_performance = summary_data['pair_performance']
            market_overview = summary_data.get('market_overview', {})

            message = f"""📊 <b>گزارش روزانه Forex Bot</b>

📅 <b>تاریخ:</b> {date.strftime('%Y-%m-%d')}
🎯 <b>کل سیگنال‌های امروز:</b> {daily_stats['signals_today']}

💰 <b>عملکرد هر جفت ارز:</b>"""

            # آمار هر جفت ارز
            pair_emojis = {
                'EUR_USD': '🇪🇺🇺🇸',
                'GBP_USD': '🇬🇧🇺🇸',
                'USD_JPY': '🇺🇸🇯🇵',
                'USD_CHF': '🇺🇸🇨🇭',
                'AUD_USD': '🇦🇺🇺🇸'
            }

            for pair, count in daily_stats['signals_per_pair'].items():
                emoji = pair_emojis.get(pair, '💱')
                performance = pair_performance.get(pair, {})
                win_rate = performance.get('win_rate', 0)
                
                message += f"\n{emoji} <b>{pair.replace('_', '/')}:</b> {count} سیگنال"
                if win_rate > 0:
                    message += f" (موفقیت: {win_rate:.1f}%)"

            # توزیع کیفیت
            message += f"\n\n📈 <b>توزیع کیفیت:</b>"
            message += f"\n🟢 عالی: {daily_stats['quality_distribution']['HIGH']}"
            message += f"\n🟡 متوسط: {daily_stats['quality_distribution']['MEDIUM']}"
            message += f"\n🟠 ضعیف: {daily_stats['quality_distribution']['LOW']}"

            # بررسی بازار
            if market_overview:
                message += f"\n\n🌍 <b>بررسی بازار:</b>"
                message += f"\n• قوی‌ترین ارز: {market_overview.get('strongest_currency', 'N/A')}"
                message += f"\n• ضعیف‌ترین ارز: {market_overview.get('weakest_currency', 'N/A')}"
                message += f"\n• نوسانات: {market_overview.get('volatility_level', 'N/A')}"

            message += f"\n\n⏰ <b>زمان گزارش:</b> {datetime.datetime.now().strftime('%H:%M UTC')}"
            message += f"\n🏦 <b>نوع بازار:</b> Forex"

            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='HTML'
            )

            self.logger.info("گزارش روزانه ارسال شد")
            return True

        except Exception as e:
            self.logger.error(f"خطا در ارسال گزارش روزانه: {e}")
            return False

    def _translate_signal_type(self, signal_type: str) -> str:
        """ترجمه نوع سیگنال"""
        translations = {
            'TREND_CONTINUATION': 'ادامه روند',
            'REVERSAL': 'برگشت روند',
            'BREAKOUT': 'شکست سطح',
            'PULLBACK': 'بازگشت قیمت'
        }
        return translations.get(signal_type, signal_type)

    async def send_risk_alert(self, alert_data: Dict) -> bool:
        """ارسال هشدار ریسک"""
        try:
            alert_type = alert_data['type']
            message = alert_data['message']
            severity = alert_data.get('severity', 'MEDIUM')
            
            severity_emojis = {
                'HIGH': '🚨',
                'MEDIUM': '⚠️',
                'LOW': 'ℹ️'
            }
            
            emoji = severity_emojis.get(severity, '⚠️')
            
            alert_message = f"""{emoji} <b>هشدار ریسک</b>

<b>نوع:</b> {alert_type}
<b>پیام:</b> {message}
<b>زمان:</b> {datetime.datetime.now().strftime('%H:%M:%S UTC')}

⚠️ لطفاً احتیاط کرده و ریسک را بررسی کنید."""

            await self.bot.send_message(
                chat_id=self.chat_id,
                text=alert_message,
                parse_mode='HTML'
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"خطا در ارسال هشدار ریسک: {e}")
            return False

    async def test_connection(self) -> bool:
        """تست اتصال تلگرام"""
        try:
            test_message = """🧪 <b>تست اتصال Forex Bot</b>

✅ اتصال تلگرام برقرار است
🏦 سیستم تحلیل Forex آماده
📊 جفت ارزهای تحت نظارت: EUR/USD, GBP/USD, USD/JPY
🧠 سیستم AI چندگانه فعال
⏰ تحلیل هر ساعت

<b>ربات آماده دریافت و ارسال سیگنال‌های Forex است</b>"""

            await self.bot.send_message(
                chat_id=self.chat_id,
                text=test_message,
                parse_mode='HTML'
            )
            
            self.logger.info("تست اتصال تلگرام موفق")
            return True
            
        except Exception as e:
            self.logger.error(f"خطا در تست اتصال تلگرام: {e}")
            return False
