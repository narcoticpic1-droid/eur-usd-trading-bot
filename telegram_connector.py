"""
connectors/telegram_connector.py
کانکتور بهینه‌شده Telegram با قابلیت‌های پیشرفته
"""

import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

class TelegramConnector:
    """کانکتور Telegram با قابلیت‌های حرفه‌ای"""
    
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.session = None
        
        # تنظیمات پیام
        self.max_message_length = 4000
        self.retry_attempts = 3
        self.retry_delay = 5
        
        # آمار
        self.stats = {
            'messages_sent': 0,
            'messages_failed': 0,
            'last_success': None,
            'last_error': None
        }
    
    async def __aenter__(self):
        """Context manager entry"""
        await self._create_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        await self._close_session()
    
    async def _create_session(self):
        """ایجاد session HTTP"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def _close_session(self):
        """بستن session HTTP"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def test_connection(self) -> bool:
        """تست اتصال Telegram"""
        try:
            await self._create_session()
            
            async with self.session.get(f"{self.base_url}/getMe") as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('ok'):
                        print(f"✅ Telegram Bot: {data['result']['first_name']}")
                        return True
            
            return False
            
        except Exception as e:
            print(f"❌ Telegram test failed: {e}")
            return False
        finally:
            await self._close_session()
    
    async def send_message(self, text: str, parse_mode: str = 'Markdown') -> bool:
        """ارسال پیام با retry mechanism"""
        if not text.strip():
            return False
        
        # تقسیم پیام‌های طولانی
        messages = self._split_long_message(text)
        
        success_count = 0
        for message in messages:
            if await self._send_single_message(message, parse_mode):
                success_count += 1
            else:
                break  # توقف در صورت خطا
        
        return success_count == len(messages)
    
    async def _send_single_message(self, text: str, parse_mode: str) -> bool:
        """ارسال یک پیام"""
        await self._create_session()
        
        for attempt in range(self.retry_attempts):
            try:
                payload = {
                    'chat_id': self.chat_id,
                    'text': text,
                    'parse_mode': parse_mode,
                    'disable_web_page_preview': True
                }
                
                async with self.session.post(
                    f"{self.base_url}/sendMessage",
                    json=payload
                ) as response:
                    
                    if response.status == 200:
                        self.stats['messages_sent'] += 1
                        self.stats['last_success'] = datetime.now()
                        return True
                    
                    elif response.status == 429:  # Rate limit
                        retry_after = int(response.headers.get('Retry-After', 60))
                        print(f"⚡ Rate limit - انتظار {retry_after} ثانیه")
                        await asyncio.sleep(retry_after)
                        continue
                    
                    else:
                        error_data = await response.text()
                        print(f"❌ Telegram error {response.status}: {error_data}")
                        
            except Exception as e:
                print(f"❌ Telegram send error (attempt {attempt + 1}): {e}")
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        self.stats['messages_failed'] += 1
        self.stats['last_error'] = datetime.now()
        return False
    
    def _split_long_message(self, text: str) -> List[str]:
        """تقسیم پیام‌های طولانی"""
        if len(text) <= self.max_message_length:
            return [text]
        
        messages = []
        current_message = ""
        lines = text.split('\n')
        
        for line in lines:
            if len(current_message + line + '\n') <= self.max_message_length:
                current_message += line + '\n'
            else:
                if current_message:
                    messages.append(current_message.rstrip())
                    current_message = line + '\n'
                else:
                    # خط خیلی طولانی - تقسیم کن
                    while len(line) > self.max_message_length:
                        messages.append(line[:self.max_message_length])
                        line = line[self.max_message_length:]
                    current_message = line + '\n'
        
        if current_message:
            messages.append(current_message.rstrip())
        
        return messages
    
    async def send_alert(self, title: str, message: str, severity: str = 'INFO') -> bool:
        """ارسال هشدار با فرمت خاص"""
        severity_icons = {
            'EMERGENCY': '🚨',
            'HIGH': '🔴',
            'MEDIUM': '🟡',
            'LOW': '🟢',
            'INFO': 'ℹ️'
        }
        
        icon = severity_icons.get(severity, 'ℹ️')
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        formatted_message = f"""
{icon} **{title}**
⏰ {timestamp}

{message}
"""
        
        return await self.send_message(formatted_message)
    
    async def send_trading_signal(self, signal_data: Dict) -> bool:
        """ارسال سیگنال معاملاتی فرمت‌شده"""
        try:
            symbol = signal_data.get('symbol', 'نامشخص')
            position = signal_data.get('position', 'نامشخص')
            confidence = signal_data.get('confidence', 0)
            leverage = signal_data.get('leverage', '1X')
            
            # ایموجی بر اساس نوع پوزیشن
            position_emoji = "🟢" if position == 'LONG' else "🔴"
            
            message = f"""
{position_emoji} **سیگنال معاملاتی**

📊 **نماد:** {symbol}
📈 **پوزیشن:** {position}
⚡ **اهرم:** {leverage}
🎯 **اطمینان:** {confidence:.1%}

💰 **قیمت ورود:** ${signal_data.get('entry_price', 0):.4f}
🛑 **Stop Loss:** ${signal_data.get('stop_loss', 0):.4f}
🎯 **Take Profit:** ${signal_data.get('take_profits', [0])[0]:.4f}

**دلایل:**"""
            
            for reason in signal_data.get('reasoning', [])[:3]:
                message += f"\n• {reason}"
            
            # اطلاعات AI اگر موجود باشد
            if signal_data.get('ai_evaluation'):
                ai_rec = signal_data['ai_evaluation'].get('final_decision', {}).get('action', 'HOLD')
                message += f"\n\n🧠 **تأیید AI:** {ai_rec}"
            
            return await self.send_message(message)
            
        except Exception as e:
            print(f"❌ خطا در ارسال سیگنال: {e}")
            return False
    
    async def send_portfolio_update(self, portfolio_data: Dict) -> bool:
        """ارسال به‌روزرسانی پورتفولیو"""
        try:
            message = f"""
📊 **به‌روزرسانی پورتفولیو**
⏰ {datetime.now().strftime('%H:%M:%S')}

💼 **پوزیشن‌های فعال:** {portfolio_data.get('active_positions', 0)}
💰 **سود/زیان روزانه:** {portfolio_data.get('daily_pnl', 0):.2f}%
📈 **نرخ موفقیت:** {portfolio_data.get('win_rate', 0):.1f}%
🛡️ **سطح ریسک:** {portfolio_data.get('risk_level', 'نامشخص')}

**آمار امروز:**
• سیگنال‌ها: {portfolio_data.get('signals_today', 0)}
• موفق: {portfolio_data.get('successful_today', 0)}
• ناموفق: {portfolio_data.get('failed_today', 0)}
"""
            
            return await self.send_message(message)
            
        except Exception as e:
            print(f"❌ خطا در ارسال portfolio update: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """دریافت آمار کانکتور"""
        return {
            'messages_sent': self.stats['messages_sent'],
            'messages_failed': self.stats['messages_failed'],
            'success_rate': (
                self.stats['messages_sent'] / 
                (self.stats['messages_sent'] + self.stats['messages_failed'])
                if (self.stats['messages_sent'] + self.stats['messages_failed']) > 0 else 0
            ),
            'last_success': self.stats['last_success'],
            'last_error': self.stats['last_error']
        }
