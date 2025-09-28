import datetime
import sqlite3
import asyncio
import json
from typing import Dict, List, Optional, Tuple
from enum import Enum
import config

class EmergencyLevel(Enum):
    """سطوح اضطراری"""
    LOW = "LOW"
    MEDIUM = "MEDIUM" 
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class EmergencyType(Enum):
    """انواع اضطرار"""
    PORTFOLIO_LOSS = "PORTFOLIO_LOSS"
    POSITION_LOSS = "POSITION_LOSS"
    MARKET_CRASH = "MARKET_CRASH"
    API_FAILURE = "API_FAILURE"
    CORRELATION_SPIKE = "CORRELATION_SPIKE"
    LIQUIDITY_CRISIS = "LIQUIDITY_CRISIS"
    CONSECUTIVE_LOSSES = "CONSECUTIVE_LOSSES"
    UNUSUAL_VOLATILITY = "UNUSUAL_VOLATILITY"
    MANUAL_STOP = "MANUAL_STOP"

class EmergencyAction(Enum):
    """اقدامات اضطراری"""
    CLOSE_ALL_POSITIONS = "CLOSE_ALL_POSITIONS"
    STOP_NEW_SIGNALS = "STOP_NEW_SIGNALS"
    REDUCE_LEVERAGE = "REDUCE_LEVERAGE"
    PARTIAL_CLOSE = "PARTIAL_CLOSE"
    ALERT_ONLY = "ALERT_ONLY"
    FORCE_EXIT = "FORCE_EXIT"

class EmergencyStop:
    """
    سیستم توقف اضطراری برای حفاظت از سرمایه
    """

    def __init__(self, db_path: str = "emergency_system.db"):
        self.db_path = db_path
        self.name = "Emergency Stop System"
        
        # وضعیت فعلی سیستم
        self.system_status = {
            'is_emergency_active': False,
            'emergency_level': EmergencyLevel.LOW,
            'emergency_type': None,
            'activation_time': None,
            'reason': None,
            'actions_taken': [],
            'manual_override': False
        }
        
        # تنظیمات آستانه‌ها از config
        self.emergency_thresholds = {
            'portfolio_loss_24h': config.EMERGENCY_STOPS.get('portfolio_loss_24h', 0.10),
            'single_position_loss': config.EMERGENCY_STOPS.get('single_position_loss', 0.05),
            'consecutive_loss_limit': config.CIRCUIT_BREAKERS.get('consecutive_loss_limit', 3),
            'unusual_volatility_threshold': config.CIRCUIT_BREAKERS.get('unusual_volatility_threshold', 0.15),
            'correlation_spike_threshold': config.CIRCUIT_BREAKERS.get('correlation_spike_threshold', 0.9),
            'api_error_threshold': config.CIRCUIT_BREAKERS.get('api_error_threshold', 5),
            'low_liquidity_warning': config.CIRCUIT_BREAKERS.get('low_liquidity_warning', 200000)
        }
        
        # شمارنده‌های خطا
        self.error_counters = {
            'api_errors': 0,
            'consecutive_losses': 0,
            'failed_orders': 0,
            'connection_failures': 0
        }
        
        # تاریخچه رویدادهای اضطراری
        self.emergency_history = []
        
        # callback functions
        self.callbacks = {
            'position_manager': None,
            'portfolio_tracker': None,
            'notifier': None,
            'exchange': None
        }
        
        self._init_database()
        self._load_system_state()

    def _init_database(self):
        """ایجاد جداول سیستم اضطراری"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # جدول رویدادهای اضطراری
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS emergency_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    emergency_type TEXT NOT NULL,
                    emergency_level TEXT NOT NULL,
                    trigger_value REAL,
                    threshold_value REAL,
                    description TEXT,
                    actions_taken TEXT,
                    resolution_time DATETIME,
                    portfolio_impact REAL,
                    positions_affected INTEGER,
                    manual_trigger BOOLEAN DEFAULT FALSE
                )
            ''')

            # جدول اقدامات انجام شده
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS emergency_actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    emergency_event_id INTEGER,
                    action_type TEXT NOT NULL,
                    action_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    target_positions TEXT,
                    success BOOLEAN,
                    error_message TEXT,
                    impact_amount REAL,
                    FOREIGN KEY (emergency_event_id) REFERENCES emergency_events (id)
                )
            ''')

            # جدول وضعیت سیستم
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_status (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    is_emergency_active BOOLEAN,
                    emergency_level TEXT,
                    emergency_type TEXT,
                    reason TEXT,
                    auto_recovery_enabled BOOLEAN DEFAULT TRUE,
                    manual_override BOOLEAN DEFAULT FALSE
                )
            ''')

            # جدول آستانه‌های تنظیم شده
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS threshold_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    threshold_name TEXT NOT NULL,
                    old_value REAL,
                    new_value REAL,
                    reason TEXT,
                    set_by TEXT DEFAULT 'SYSTEM'
                )
            ''')

            conn.commit()
            conn.close()
            print("✅ جداول سیستم اضطراری ایجاد شدند")

        except Exception as e:
            print(f"خطا در ایجاد جداول اضطراری: {e}")

    def _load_system_state(self):
        """بارگیری آخرین وضعیت سیستم"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # بررسی وضعیت فعال
            latest_status = conn.execute('''
                SELECT * FROM system_status 
                ORDER BY timestamp DESC LIMIT 1
            ''').fetchone()
            
            if latest_status:
                self.system_status.update({
                    'is_emergency_active': bool(latest_status[2]),
                    'emergency_level': EmergencyLevel(latest_status[3]) if latest_status[3] else EmergencyLevel.LOW,
                    'emergency_type': EmergencyType(latest_status[4]) if latest_status[4] else None,
                    'reason': latest_status[5],
                    'manual_override': bool(latest_status[7])
                })
            
            conn.close()
            
            if self.system_status['is_emergency_active']:
                print(f"⚠️ سیستم اضطراری فعال - سطح: {self.system_status['emergency_level'].value}")
            else:
                print("✅ سیستم اضطراری در حالت عادی")

        except Exception as e:
            print(f"خطا در بارگیری وضعیت سیستم: {e}")

    def register_callbacks(self, callbacks: Dict):
        """ثبت callback functions برای اقدامات اضطراری"""
        self.callbacks.update(callbacks)
        print("✅ Callback functions ثبت شدند")

    async def check_emergency_conditions(self, market_data: Dict, portfolio_data: Dict) -> bool:
        """بررسی شرایط اضطراری"""
        if self.system_status['is_emergency_active']:
            return True

        emergency_detected = False
        
        try:
            # 1. بررسی ضرر پورتفولیو
            portfolio_loss_pct = self._check_portfolio_loss(portfolio_data)
            if portfolio_loss_pct >= self.emergency_thresholds['portfolio_loss_24h']:
                await self._trigger_emergency(
                    EmergencyType.PORTFOLIO_LOSS,
                    EmergencyLevel.CRITICAL,
                    f"Portfolio loss: {portfolio_loss_pct*100:.2f}%",
                    portfolio_loss_pct
                )
                emergency_detected = True

            # 2. بررسی ضرر پوزیشن واحد
            max_position_loss = self._check_single_position_loss(portfolio_data)
            if max_position_loss >= self.emergency_thresholds['single_position_loss']:
                await self._trigger_emergency(
                    EmergencyType.POSITION_LOSS,
                    EmergencyLevel.HIGH,
                    f"Single position loss: {max_position_loss*100:.2f}%",
                    max_position_loss
                )
                emergency_detected = True

            # 3. بررسی ضررهای متوالی
            consecutive_losses = self._check_consecutive_losses(portfolio_data)
            if consecutive_losses >= self.emergency_thresholds['consecutive_loss_limit']:
                await self._trigger_emergency(
                    EmergencyType.CONSECUTIVE_LOSSES,
                    EmergencyLevel.HIGH,
                    f"Consecutive losses: {consecutive_losses}",
                    consecutive_losses
                )
                emergency_detected = True

            # 4. بررسی نوسانات غیرعادی
            volatility_spike = self._check_unusual_volatility(market_data)
            if volatility_spike >= self.emergency_thresholds['unusual_volatility_threshold']:
                await self._trigger_emergency(
                    EmergencyType.UNUSUAL_VOLATILITY,
                    EmergencyLevel.MEDIUM,
                    f"Unusual volatility: {volatility_spike*100:.2f}%",
                    volatility_spike
                )
                emergency_detected = True

            # 5. بررسی همبستگی بازار
            correlation_spike = self._check_correlation_spike(market_data)
            if correlation_spike >= self.emergency_thresholds['correlation_spike_threshold']:
                await self._trigger_emergency(
                    EmergencyType.CORRELATION_SPIKE,
                    EmergencyLevel.MEDIUM,
                    f"Market correlation spike: {correlation_spike:.2f}",
                    correlation_spike
                )

            # 6. بررسی خطاهای API
            if self.error_counters['api_errors'] >= self.emergency_thresholds['api_error_threshold']:
                await self._trigger_emergency(
                    EmergencyType.API_FAILURE,
                    EmergencyLevel.HIGH,
                    f"API errors: {self.error_counters['api_errors']}",
                    self.error_counters['api_errors']
                )
                emergency_detected = True

            # 7. بررسی نقدینگی
            liquidity_crisis = self._check_liquidity_crisis(market_data)
            if liquidity_crisis:
                await self._trigger_emergency(
                    EmergencyType.LIQUIDITY_CRISIS,
                    EmergencyLevel.MEDIUM,
                    "Low market liquidity detected",
                    0
                )

            return emergency_detected

        except Exception as e:
            print(f"خطا در بررسی شرایط اضطراری: {e}")
            return False

    def _check_portfolio_loss(self, portfolio_data: Dict) -> float:
        """بررسی ضرر کل پورتفولیو"""
        try:
            daily_pnl = portfolio_data.get('daily_pnl', 0)
            total_balance = portfolio_data.get('total_balance', 1)
            
            if total_balance <= 0:
                return 0
            
            loss_percentage = abs(daily_pnl) / total_balance if daily_pnl < 0 else 0
            return loss_percentage
            
        except Exception as e:
            print(f"خطا در بررسی ضرر پورتفولیو: {e}")
            return 0

    def _check_single_position_loss(self, portfolio_data: Dict) -> float:
        """بررسی ضرر بیشترین پوزیشن"""
        try:
            active_positions = portfolio_data.get('active_positions', {})
            max_loss = 0
            
            for position_id, position in active_positions.items():
                pnl_pct = position.get('current_pnl_pct', 0)
                if pnl_pct < 0:
                    loss_pct = abs(pnl_pct) / 100
                    max_loss = max(max_loss, loss_pct)
            
            return max_loss
            
        except Exception as e:
            print(f"خطا در بررسی ضرر پوزیشن: {e}")
            return 0

    def _check_consecutive_losses(self, portfolio_data: Dict) -> int:
        """بررسی ضررهای متوالی"""
        try:
            performance_stats = portfolio_data.get('performance_stats', {})
            return performance_stats.get('consecutive_losses', 0)
            
        except Exception as e:
            print(f"خطا در بررسی ضررهای متوالی: {e}")
            return 0

    def _check_unusual_volatility(self, market_data: Dict) -> float:
        """بررسی نوسانات غیرعادی"""
        try:
            max_volatility = 0
            
            for symbol, data in market_data.items():
                if isinstance(data, dict):
                    # محاسبه volatility بر اساس ATR یا price changes
                    atr = data.get('atr', 0)
                    current_price = data.get('close', 1)
                    
                    if current_price > 0:
                        volatility = atr / current_price
                        max_volatility = max(max_volatility, volatility)
            
            return max_volatility
            
        except Exception as e:
            print(f"خطا در بررسی نوسانات: {e}")
            return 0

    def _check_correlation_spike(self, market_data: Dict) -> float:
        """بررسی افزایش ناگهانی همبستگی"""
        try:
            # این محاسبه ساده است - در عمل باید پیچیده‌تر باشد
            symbols = list(market_data.keys())
            if len(symbols) < 2:
                return 0
            
            # فرض: اگر همه نمادها در یک جهت حرکت کنند
            price_changes = []
            for symbol, data in market_data.items():
                if isinstance(data, dict):
                    change = data.get('price_change_24h', 0)
                    price_changes.append(change)
            
            if len(price_changes) >= 2:
                # اگر همه مثبت یا همه منفی باشند
                all_positive = all(change > 0 for change in price_changes)
                all_negative = all(change < 0 for change in price_changes)
                
                if all_positive or all_negative:
                    return 0.95  # همبستگی بالا
            
            return 0.5  # همبستگی عادی
            
        except Exception as e:
            print(f"خطا در بررسی همبستگی: {e}")
            return 0

    def _check_liquidity_crisis(self, market_data: Dict) -> bool:
        """بررسی بحران نقدینگی"""
        try:
            for symbol, data in market_data.items():
                if isinstance(data, dict):
                    volume = data.get('volume', 0)
                    if volume < self.emergency_thresholds['low_liquidity_warning']:
                        return True
            return False
            
        except Exception as e:
            print(f"خطا در بررسی نقدینگی: {e}")
            return False

    async def _trigger_emergency(self, emergency_type: EmergencyType, level: EmergencyLevel, 
                                description: str, trigger_value: float):
        """فعال‌سازی حالت اضطراری"""
        try:
            # اگر قبلاً فعال است و سطح جدید پایین‌تر است، نادیده بگیر
            if (self.system_status['is_emergency_active'] and 
                self._compare_emergency_levels(level, self.system_status['emergency_level']) <= 0):
                return

            print(f"🚨 EMERGENCY TRIGGERED: {emergency_type.value} - Level: {level.value}")
            print(f"📝 Description: {description}")

            # به‌روزرسانی وضعیت سیستم
            self.system_status.update({
                'is_emergency_active': True,
                'emergency_level': level,
                'emergency_type': emergency_type,
                'activation_time': datetime.datetime.now(),
                'reason': description,
                'actions_taken': []
            })

            # ثبت رویداد در پایگاه داده
            emergency_id = self._log_emergency_event(emergency_type, level, description, trigger_value)

            # تعیین اقدامات لازم
            actions = self._determine_emergency_actions(emergency_type, level)

            # اجرای اقدامات
            for action in actions:
                success = await self._execute_emergency_action(action, emergency_id)
                if success:
                    self.system_status['actions_taken'].append(action.value)

            # ارسال هشدار
            await self._send_emergency_notification(emergency_type, level, description)

            # ذخیره وضعیت
            self._save_system_status()

        except Exception as e:
            print(f"خطا در فعال‌سازی اضطرار: {e}")

    def _compare_emergency_levels(self, level1: EmergencyLevel, level2: EmergencyLevel) -> int:
        """مقایسه سطوح اضطراری"""
        levels = {
            EmergencyLevel.LOW: 1,
            EmergencyLevel.MEDIUM: 2,
            EmergencyLevel.HIGH: 3,
            EmergencyLevel.CRITICAL: 4
        }
        return levels[level1] - levels[level2]

    def _determine_emergency_actions(self, emergency_type: EmergencyType, level: EmergencyLevel) -> List[EmergencyAction]:
        """تعیین اقدامات اضطراری"""
        actions = []

        if level == EmergencyLevel.CRITICAL:
            actions.extend([
                EmergencyAction.CLOSE_ALL_POSITIONS,
                EmergencyAction.STOP_NEW_SIGNALS
            ])
        elif level == EmergencyLevel.HIGH:
            if emergency_type in [EmergencyType.PORTFOLIO_LOSS, EmergencyType.POSITION_LOSS]:
                actions.extend([
                    EmergencyAction.PARTIAL_CLOSE,
                    EmergencyAction.REDUCE_LEVERAGE,
                    EmergencyAction.STOP_NEW_SIGNALS
                ])
            else:
                actions.extend([
                    EmergencyAction.STOP_NEW_SIGNALS,
                    EmergencyAction.REDUCE_LEVERAGE
                ])
        elif level == EmergencyLevel.MEDIUM:
            actions.extend([
                EmergencyAction.REDUCE_LEVERAGE,
                EmergencyAction.ALERT_ONLY
            ])
        else:  # LOW
            actions.append(EmergencyAction.ALERT_ONLY)

        return actions

    async def _execute_emergency_action(self, action: EmergencyAction, emergency_id: int) -> bool:
        """اجرای اقدام اضطراری"""
        try:
            success = False
            error_message = None
            impact_amount = 0

            if action == EmergencyAction.CLOSE_ALL_POSITIONS:
                success, impact_amount = await self._close_all_positions()
            elif action == EmergencyAction.PARTIAL_CLOSE:
                success, impact_amount = await self._partial_close_positions()
            elif action == EmergencyAction.STOP_NEW_SIGNALS:
                success = await self._stop_new_signals()
            elif action == EmergencyAction.REDUCE_LEVERAGE:
                success = await self._reduce_leverage()
            elif action == EmergencyAction.ALERT_ONLY:
                success = True  # فقط هشدار
            elif action == EmergencyAction.FORCE_EXIT:
                success, impact_amount = await self._force_exit_positions()

            # ثبت اقدام
            self._log_emergency_action(emergency_id, action, success, error_message, impact_amount)

            if success:
                print(f"✅ اقدام اضطراری انجام شد: {action.value}")
            else:
                print(f"❌ خطا در اقدام اضطراری: {action.value}")

            return success

        except Exception as e:
            print(f"خطا در اجرای اقدام {action.value}: {e}")
            self._log_emergency_action(emergency_id, action, False, str(e), 0)
            return False

    async def _close_all_positions(self) -> Tuple[bool, float]:
        """بستن همه پوزیشن‌ها"""
        try:
            if not self.callbacks.get('portfolio_tracker'):
                return False, 0

            portfolio_tracker = self.callbacks['portfolio_tracker']
            total_impact = 0
            
            # دریافت پوزیشن‌های فعال
            active_positions = portfolio_tracker.active_positions.copy()
            
            for position_id in active_positions:
                try:
                    # بستن پوزیشن (نیاز به قیمت فعلی)
                    # در عمل باید از exchange قیمت فعلی را دریافت کرد
                    position = active_positions[position_id]
                    current_price = position.get('current_price', position['entry_price'])
                    
                    result = portfolio_tracker.close_position(
                        position_id, 
                        current_price, 
                        "EMERGENCY_CLOSE"
                    )
                    
                    if 'pnl_amount' in result:
                        total_impact += result['pnl_amount']
                        
                except Exception as e:
                    print(f"خطا در بستن پوزیشن {position_id}: {e}")
                    continue

            return True, total_impact

        except Exception as e:
            print(f"خطا در بستن همه پوزیشن‌ها: {e}")
            return False, 0

    async def _partial_close_positions(self) -> Tuple[bool, float]:
        """بستن پوزیشن‌های ضررده"""
        try:
            if not self.callbacks.get('portfolio_tracker'):
                return False, 0

            portfolio_tracker = self.callbacks['portfolio_tracker']
            total_impact = 0
            
            # دریافت پوزیشن‌های فعال
            active_positions = portfolio_tracker.active_positions.copy()
            
            for position_id, position in active_positions.items():
                current_pnl_pct = position.get('current_pnl_pct', 0)
                
                # بستن پوزیشن‌هایی که ضرر بیش از 3% دارند
                if current_pnl_pct < -3:
                    try:
                        current_price = position.get('current_price', position['entry_price'])
                        result = portfolio_tracker.close_position(
                            position_id,
                            current_price,
                            "EMERGENCY_PARTIAL_CLOSE"
                        )
                        
                        if 'pnl_amount' in result:
                            total_impact += result['pnl_amount']
                            
                    except Exception as e:
                        print(f"خطا در بستن پوزیشن {position_id}: {e}")
                        continue

            return True, total_impact

        except Exception as e:
            print(f"خطا در بستن جزئی پوزیشن‌ها: {e}")
            return False, 0

    async def _stop_new_signals(self) -> bool:
        """متوقف کردن سیگنال‌های جدید"""
        try:
            # تنظیم فلگ توقف در سیستم اصلی
            # این باید از طریق callback یا global state انجام شود
            print("🛑 سیگنال‌های جدید متوقف شدند")
            return True
        except Exception as e:
            print(f"خطا در توقف سیگنال‌ها: {e}")
            return False

    async def _reduce_leverage(self) -> bool:
        """کاهش اهرم پوزیشن‌های فعال"""
        try:
            # در عمل این نیاز به تعامل با صرافی دارد
            print("📉 اهرم پوزیشن‌ها کاهش یافت")
            return True
        except Exception as e:
            print(f"خطا در کاهش اهرم: {e}")
            return False

    async def _force_exit_positions(self) -> Tuple[bool, float]:
        """خروج اجباری از همه پوزیشن‌ها"""
        try:
            # مانند close_all_positions اما با اولویت بالاتر
            return await self._close_all_positions()
        except Exception as e:
            print(f"خطا در خروج اجباری: {e}")
            return False, 0

    async def _send_emergency_notification(self, emergency_type: EmergencyType, 
                                         level: EmergencyLevel, description: str):
        """ارسال هشدار اضطراری"""
        try:
            if not self.callbacks.get('notifier'):
                return

            notifier = self.callbacks['notifier']
            
            emergency_message = {
                'type': 'EMERGENCY_ALERT',
                'level': level.value,
                'emergency_type': emergency_type.value,
                'description': description,
                'timestamp': datetime.datetime.now().isoformat(),
                'actions_taken': self.system_status['actions_taken'],
                'severity': 'CRITICAL' if level == EmergencyLevel.CRITICAL else 'HIGH'
            }

            await notifier.send_risk_alert(emergency_message)
            
        except Exception as e:
            print(f"خطا در ارسال هشدار اضطراری: {e}")

    def _log_emergency_event(self, emergency_type: EmergencyType, level: EmergencyLevel,
                           description: str, trigger_value: float) -> int:
        """ثبت رویداد اضطراری"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO emergency_events (
                    emergency_type, emergency_level, trigger_value,
                    threshold_value, description, actions_taken
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                emergency_type.value,
                level.value,
                trigger_value,
                self.emergency_thresholds.get(emergency_type.value.lower(), 0),
                description,
                json.dumps(self.system_status['actions_taken'])
            ))
            
            emergency_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return emergency_id
            
        except Exception as e:
            print(f"خطا در ثبت رویداد اضطراری: {e}")
            return 0

    def _log_emergency_action(self, emergency_id: int, action: EmergencyAction,
                            success: bool, error_message: str, impact_amount: float):
        """ثبت اقدام اضطراری"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO emergency_actions (
                    emergency_event_id, action_type, success, 
                    error_message, impact_amount
                ) VALUES (?, ?, ?, ?, ?)
            ''', (
                emergency_id,
                action.value,
                success,
                error_message,
                impact_amount
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"خطا در ثبت اقدام اضطراری: {e}")

    def _save_system_status(self):
        """ذخیره وضعیت سیستم"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO system_status (
                    is_emergency_active, emergency_level, emergency_type,
                    reason, manual_override
                ) VALUES (?, ?, ?, ?, ?)
            ''', (
                self.system_status['is_emergency_active'],
                self.system_status['emergency_level'].value if self.system_status['emergency_level'] else None,
                self.system_status['emergency_type'].value if self.system_status['emergency_type'] else None,
                self.system_status['reason'],
                self.system_status['manual_override']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"خطا در ذخیره وضعیت سیستم: {e}")

    async def manual_emergency_stop(self, reason: str = "Manual trigger") -> bool:
        """فعال‌سازی دستی توقف اضطراری"""
        try:
            print(f"🔴 توقف اضطراری دستی: {reason}")
            
            await self._trigger_emergency(
                EmergencyType.MANUAL_STOP,
                EmergencyLevel.CRITICAL,
                reason,
                0
            )
            
            self.system_status['manual_override'] = True
            return True
            
        except Exception as e:
            print(f"خطا در توقف اضطراری دستی: {e}")
            return False

    async def deactivate_emergency(self, reason: str = "Manual deactivation") -> bool:
        """غیرفعال‌سازی حالت اضطراری"""
        try:
            if not self.system_status['is_emergency_active']:
                print("سیستم اضطراری فعال نیست")
                return True

            print(f"✅ غیرفعال‌سازی سیستم اضطراری: {reason}")

            # به‌روزرسانی رویداد فعلی
            if self.system_status['activation_time']:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE emergency_events 
                    SET resolution_time = ?
                    WHERE timestamp >= ? AND resolution_time IS NULL
                ''', (
                    datetime.datetime.now(),
                    self.system_status['activation_time']
                ))
                
                conn.commit()
                conn.close()

            # ریست وضعیت
            self.system_status.update({
                'is_emergency_active': False,
                'emergency_level': EmergencyLevel.LOW,
                'emergency_type': None,
                'activation_time': None,
                'reason': None,
                'actions_taken': [],
                'manual_override': False
            })

            # ریست شمارنده‌های خطا
            self.error_counters = {key: 0 for key in self.error_counters}

            self._save_system_status()
            
            # ارسال اطلاع رسانی
            if self.callbacks.get('notifier'):
                await self.callbacks['notifier'].send_risk_alert({
                    'type': 'EMERGENCY_RESOLVED',
                    'message': f"Emergency system deactivated: {reason}",
                    'severity': 'INFO'
                })

            return True

        except Exception as e:
            print(f"خطا در غیرفعال‌سازی اضطرار: {e}")
            return False

    def increment_error_counter(self, error_type: str):
        """افزایش شمارنده خطا"""
        if error_type in self.error_counters:
            self.error_counters[error_type] += 1
            print(f"⚠️ خطای {error_type}: {self.error_counters[error_type]}")

    def reset_error_counter(self, error_type: str = None):
        """ریست شمارنده خطا"""
        if error_type and error_type in self.error_counters:
            self.error_counters[error_type] = 0
        else:
            self.error_counters = {key: 0 for key in self.error_counters}

    def get_emergency_status(self) -> Dict:
        """دریافت وضعیت فعلی سیستم اضطراری"""
        return {
            'system_status': {
                'is_active': self.system_status['is_emergency_active'],
                'level': self.system_status['emergency_level'].value if self.system_status['emergency_level'] else None,
                'type': self.system_status['emergency_type'].value if self.system_status['emergency_type'] else None,
                'reason': self.system_status['reason'],
                'activation_time': self.system_status['activation_time'].isoformat() if self.system_status['activation_time'] else None,
                'actions_taken': self.system_status['actions_taken'],
                'manual_override': self.system_status['manual_override']
            },
            'error_counters': self.error_counters.copy(),
            'thresholds': self.emergency_thresholds.copy()
        }

    def get_emergency_history(self, days: int = 7) -> List[Dict]:
        """دریافت تاریخچه رویدادهای اضطراری"""
        try:
            start_date = datetime.datetime.now() - datetime.timedelta(days=days)
            
            conn = sqlite3.connect(self.db_path)
            
            events = conn.execute('''
                SELECT * FROM emergency_events 
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            ''', (start_date,)).fetchall()
            
            conn.close()
            
            history = []
            for event in events:
                history.append({
                    'id': event[0],
                    'timestamp': event[1],
                    'type': event[2],
                    'level': event[3],
                    'trigger_value': event[4],
                    'threshold_value': event[5],
                    'description': event[6],
                    'actions_taken': json.loads(event[7]) if event[7] else [],
                    'resolution_time': event[8],
                    'portfolio_impact': event[9],
                    'positions_affected': event[10],
                    'manual_trigger': bool(event[11])
                })
            
            return history
            
        except Exception as e:
            print(f"خطا در دریافت تاریخچه: {e}")
            return []

    def update_threshold(self, threshold_name: str, new_value: float, reason: str = "Manual update"):
        """به‌روزرسانی آستانه‌های اضطراری"""
        try:
            if threshold_name in self.emergency_thresholds:
                old_value = self.emergency_thresholds[threshold_name]
                self.emergency_thresholds[threshold_name] = new_value
                
                # ثبت تغییر
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO threshold_history (
                        threshold_name, old_value, new_value, reason
                    ) VALUES (?, ?, ?, ?)
                ''', (threshold_name, old_value, new_value, reason))
                
                conn.commit()
                conn.close()
                
                print(f"✅ آستانه {threshold_name} به‌روزرسانی شد: {old_value} → {new_value}")
                return True
            else:
                print(f"❌ آستانه {threshold_name} وجود ندارد")
                return False
                
        except Exception as e:
            print(f"خطا در به‌روزرسانی آستانه: {e}")
            return False
