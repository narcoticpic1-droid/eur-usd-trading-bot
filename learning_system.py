import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import logging
import os

class AdaptiveLearningSystem:
    """
    سیستم یادگیری تطبیقی برای بهبود عملکرد سیگنال‌های Forex
    """
    
    def __init__(self, db_path: str = "forex_learning.db"):
        self.db_path = db_path
        self.models = {}
        self.scalers = {}
        self.performance_history = {}
        
        # تنظیمات یادگیری
        self.min_samples_for_training = 50
        self.retrain_frequency_days = 7
        self.confidence_threshold = 0.6
        
        # پایگاه داده
        self._init_database()
        
        # مدل‌های ML
        self._init_models()
        
        logging.info("🧠 سیستم یادگیری تطبیقی Forex راه‌اندازی شد")

    def _init_database(self):
        """راه‌اندازی پایگاه داده یادگیری"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # جدول سیگنال‌ها و نتایج
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signal_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    pair TEXT NOT NULL,
                    signal_type TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    profit_loss_pips REAL,
                    success BOOLEAN,
                    confidence REAL,
                    market_conditions TEXT,
                    technical_features TEXT,
                    session_type TEXT,
                    volatility REAL,
                    spread REAL,
                    news_impact TEXT
                )
            ''')
            
            # جدول عملکرد مدل‌ها
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    pair TEXT NOT NULL,
                    model_type TEXT,
                    accuracy REAL,
                    precision_score REAL,
                    recall_score REAL,
                    total_predictions INTEGER,
                    successful_predictions INTEGER
                )
            ''')
            
            # جدول پارامترهای تطبیقی
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS adaptive_parameters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    pair TEXT NOT NULL,
                    parameter_name TEXT,
                    parameter_value REAL,
                    performance_impact REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.error(f"خطا در راه‌اندازی پایگاه داده یادگیری: {e}")

    def _init_models(self):
        """راه‌اندازی مدل‌های یادگیری ماشین"""
        try:
            # مدل‌های مختلف برای انواع سیگنال
            self.model_configs = {
                'trend_following': {
                    'model': RandomForestClassifier(n_estimators=100, random_state=42),
                    'features': ['ema_trend', 'macd_signal', 'rsi_value', 'atr_ratio', 'session_strength']
                },
                'reversal': {
                    'model': GradientBoostingClassifier(n_estimators=100, random_state=42),
                    'features': ['support_resistance_distance', 'rsi_extreme', 'divergence', 'volume_spike']
                },
                'breakout': {
                    'model': RandomForestClassifier(n_estimators=150, random_state=42),
                    'features': ['consolidation_time', 'volume_increase', 'volatility_expansion', 'news_impact']
                }
            }
            
            # تلاش برای بارگذاری مدل‌های ذخیره شده
            self._load_saved_models()
            
        except Exception as e:
            logging.error(f"خطا در راه‌اندازی مدل‌ها: {e}")

    def _load_saved_models(self):
        """بارگذاری مدل‌های ذخیره شده"""
        models_dir = "saved_models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            return
            
        for model_type in self.model_configs.keys():
            model_path = f"{models_dir}/{model_type}_model.joblib"
            scaler_path = f"{models_dir}/{model_type}_scaler.joblib"
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                try:
                    self.models[model_type] = joblib.load(model_path)
                    self.scalers[model_type] = joblib.load(scaler_path)
                    logging.info(f"مدل {model_type} بارگذاری شد")
                except Exception as e:
                    logging.warning(f"خطا در بارگذاری مدل {model_type}: {e}")

    def record_signal_result(self, signal_data: Dict, result_data: Dict):
        """ثبت نتیجه سیگنال برای یادگیری"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO signal_results (
                    pair, signal_type, entry_price, exit_price, 
                    profit_loss_pips, success, confidence, 
                    market_conditions, technical_features, session_type,
                    volatility, spread, news_impact
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal_data.get('pair', 'EUR/USD'),
                signal_data.get('signal_type', ''),
                signal_data.get('entry_price', 0),
                result_data.get('exit_price', 0),
                result_data.get('profit_loss_pips', 0),
                result_data.get('success', False),
                signal_data.get('confidence', 0),
                str(signal_data.get('market_conditions', {})),
                str(signal_data.get('technical_features', {})),
                signal_data.get('session_type', ''),
                signal_data.get('volatility', 0),
                signal_data.get('spread', 0),
                signal_data.get('news_impact', '')
            ))
            
            conn.commit()
            conn.close()
            
            # بررسی نیاز به بازآموزی
            self._check_retrain_trigger(signal_data.get('pair', 'EUR/USD'))
            
        except Exception as e:
            logging.error(f"خطا در ثبت نتیجه سیگنال: {e}")

    def predict_signal_success(self, signal_data: Dict) -> float:
        """پیش‌بینی احتمال موفقیت سیگنال"""
        try:
            signal_type = signal_data.get('signal_type', 'trend_following')
            pair = signal_data.get('pair', 'EUR/USD')
            
            # اگر مدل وجود ندارد، مقدار پیش‌فرض برگردان
            if signal_type not in self.models:
                return 0.7  # احتمال پیش‌فرض
            
            # استخراج ویژگی‌ها
            features = self._extract_features(signal_data, signal_type)
            if features is None:
                return 0.7
            
            # پیش‌بینی
            model = self.models[signal_type]
            scaler = self.scalers.get(signal_type)
            
            if scaler:
                features_scaled = scaler.transform([features])
            else:
                features_scaled = [features]
            
            # احتمال کلاس مثبت (موفقیت)
            probabilities = model.predict_proba(features_scaled)
            success_probability = probabilities[0][1] if len(probabilities[0]) > 1 else 0.7
            
            return float(success_probability)
            
        except Exception as e:
            logging.error(f"خطا در پیش‌بینی موفقیت سیگنال: {e}")
            return 0.7

    def get_adaptive_parameters(self, pair: str = 'EUR/USD') -> Dict:
        """دریافت پارامترهای تطبیقی برای جفت ارز"""
        try:
            # پارامترهای پیش‌فرض
            default_params = {
                'confidence_threshold': 0.6,
                'risk_multiplier': 1.0,
                'leverage_adjustment': 1.0,
                'stop_loss_multiplier': 1.0,
                'take_profit_multiplier': 1.0
            }
            
            # دریافت عملکرد اخیر
            recent_performance = self._get_recent_performance(pair)
            
            if not recent_performance:
                return default_params
            
            # تنظیم پارامترها بر اساس عملکرد
            win_rate = recent_performance['win_rate']
            avg_profit = recent_performance['avg_profit_pips']
            
            adaptive_params = default_params.copy()
            
            # تنظیم آستانه اطمینان
            if win_rate > 0.8:
                adaptive_params['confidence_threshold'] = 0.5  # کاهش آستانه
            elif win_rate < 0.4:
                adaptive_params['confidence_threshold'] = 0.8  # افزایش آستانه
            
            # تنظیم ریسک
            if win_rate > 0.7 and avg_profit > 10:
                adaptive_params['risk_multiplier'] = 1.2
            elif win_rate < 0.5:
                adaptive_params['risk_multiplier'] = 0.8
            
            # تنظیم اهرم
            if recent_performance['max_drawdown'] > 5:  # 5% drawdown
                adaptive_params['leverage_adjustment'] = 0.8
            elif recent_performance['sharpe_ratio'] > 1.5:
                adaptive_params['leverage_adjustment'] = 1.2
            
            return adaptive_params
            
        except Exception as e:
            logging.error(f"خطا در دریافت پارامترهای تطبیقی: {e}")
            return {
                'confidence_threshold': 0.6,
                'risk_multiplier': 1.0,
                'leverage_adjustment': 1.0,
                'stop_loss_multiplier': 1.0,
                'take_profit_multiplier': 1.0
            }

    def _extract_features(self, signal_data: Dict, signal_type: str) -> Optional[List[float]]:
        """استخراج ویژگی‌ها برای مدل ML"""
        try:
            feature_names = self.model_configs[signal_type]['features']
            features = []
            
            technical_data = signal_data.get('technical_features', {})
            market_data = signal_data.get('market_conditions', {})
            
            for feature_name in feature_names:
                if feature_name == 'ema_trend':
                    features.append(technical_data.get('ema_trend_strength', 0))
                elif feature_name == 'macd_signal':
                    features.append(technical_data.get('macd_histogram', 0))
                elif feature_name == 'rsi_value':
                    features.append(technical_data.get('rsi', 50))
                elif feature_name == 'atr_ratio':
                    features.append(technical_data.get('atr_ratio', 1))
                elif feature_name == 'session_strength':
                    features.append(self._calculate_session_strength(signal_data))
                elif feature_name == 'support_resistance_distance':
                    features.append(technical_data.get('sr_distance', 0))
                elif feature_name == 'rsi_extreme':
                    rsi = technical_data.get('rsi', 50)
                    features.append(1 if rsi > 70 or rsi < 30 else 0)
                elif feature_name == 'divergence':
                    features.append(technical_data.get('divergence_score', 0))
                elif feature_name == 'volume_spike':
                    features.append(technical_data.get('volume_ratio', 1))
                elif feature_name == 'consolidation_time':
                    features.append(market_data.get('consolidation_periods', 0))
                elif feature_name == 'volume_increase':
                    features.append(technical_data.get('volume_increase', 0))
                elif feature_name == 'volatility_expansion':
                    features.append(market_data.get('volatility_ratio', 1))
                elif feature_name == 'news_impact':
                    features.append(self._calculate_news_impact(signal_data))
                else:
                    features.append(0)  # مقدار پیش‌فرض
            
            return features if len(features) == len(feature_names) else None
            
        except Exception as e:
            logging.error(f"خطا در استخراج ویژگی‌ها: {e}")
            return None

    def _calculate_session_strength(self, signal_data: Dict) -> float:
        """محاسبه قدرت سشن معاملاتی"""
        session = signal_data.get('session_type', '')
        hour = datetime.now().hour
        
        # قدرت سشن‌های مختلف
        if session == 'LONDON' or (7 <= hour <= 16):
            return 0.9
        elif session == 'NEW_YORK' or (13 <= hour <= 22):
            return 0.8
        elif session == 'ASIAN' or (22 <= hour or hour <= 7):
            return 0.6
        else:
            return 0.5

    def _calculate_news_impact(self, signal_data: Dict) -> float:
        """محاسبه تأثیر اخبار"""
        news_impact = signal_data.get('news_impact', '')
        
        if news_impact == 'HIGH':
            return 0.9
        elif news_impact == 'MEDIUM':
            return 0.6
        elif news_impact == 'LOW':
            return 0.3
        else:
            return 0.1

    def _get_recent_performance(self, pair: str, days: int = 30) -> Optional[Dict]:
        """دریافت عملکرد اخیر"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            cutoff_date = datetime.now() - timedelta(days=days)
            
            query = '''
                SELECT success, profit_loss_pips, confidence
                FROM signal_results
                WHERE pair = ? AND timestamp > ?
                ORDER BY timestamp DESC
            '''
            
            df = pd.read_sql_query(query, conn, params=(pair, cutoff_date))
            conn.close()
            
            if df.empty:
                return None
            
            win_rate = df['success'].mean()
            avg_profit = df['profit_loss_pips'].mean()
            max_drawdown = abs(df['profit_loss_pips'].cumsum().min())
            
            # محاسبه Sharpe ratio ساده
            if df['profit_loss_pips'].std() > 0:
                sharpe_ratio = avg_profit / df['profit_loss_pips'].std()
            else:
                sharpe_ratio = 0
            
            return {
                'win_rate': win_rate,
                'avg_profit_pips': avg_profit,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'total_trades': len(df)
            }
            
        except Exception as e:
            logging.error(f"خطا در دریافت عملکرد اخیر: {e}")
            return None

    def _check_retrain_trigger(self, pair: str):
        """بررسی نیاز به بازآموزی مدل"""
        try:
            # بررسی تعداد نمونه‌ها
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT COUNT(*) FROM signal_results 
                WHERE pair = ? AND timestamp > datetime('now', '-7 days')
            ''', (pair,))
            
            recent_count = cursor.fetchone()[0]
            conn.close()
            
            if recent_count >= self.min_samples_for_training:
                self._retrain_models(pair)
                
        except Exception as e:
            logging.error(f"خطا در بررسی نیاز به بازآموزی: {e}")

    def _retrain_models(self, pair: str):
        """بازآموزی مدل‌ها"""
        try:
            logging.info(f"شروع بازآموزی مدل‌ها برای {pair}")
            
            # دریافت داده‌های آموزشی
            training_data = self._prepare_training_data(pair)
            
            if training_data is None or len(training_data) < self.min_samples_for_training:
                logging.warning(f"داده کافی برای بازآموزی {pair} موجود نیست")
                return
            
            # آموزش هر نوع مدل
            for signal_type, config in self.model_configs.items():
                type_data = training_data[training_data['signal_type'] == signal_type]
                
                if len(type_data) < 20:  # حداقل 20 نمونه
                    continue
                
                self._train_model(signal_type, type_data, config)
            
            logging.info(f"بازآموزی مدل‌ها برای {pair} تکمیل شد")
            
        except Exception as e:
            logging.error(f"خطا در بازآموزی مدل‌ها: {e}")

    def _prepare_training_data(self, pair: str) -> Optional[pd.DataFrame]:
        """آماده‌سازی داده‌های آموزشی"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT * FROM signal_results
                WHERE pair = ? AND timestamp > datetime('now', '-90 days')
                ORDER BY timestamp DESC
            '''
            
            df = pd.read_sql_query(query, conn, params=(pair,))
            conn.close()
            
            return df if not df.empty else None
            
        except Exception as e:
            logging.error(f"خطا در آماده‌سازی داده‌های آموزشی: {e}")
            return None

    def _train_model(self, signal_type: str, data: pd.DataFrame, config: Dict):
        """آموزش یک مدل خاص"""
        try:
            # آماده‌سازی ویژگی‌ها و برچسب‌ها
            features = []
            labels = []
            
            for _, row in data.iterrows():
                try:
                    # تبدیل technical_features از string به dict
                    tech_features = eval(row['technical_features']) if row['technical_features'] else {}
                    market_conditions = eval(row['market_conditions']) if row['market_conditions'] else {}
                    
                    signal_data = {
                        'technical_features': tech_features,
                        'market_conditions': market_conditions,
                        'session_type': row['session_type'],
                        'news_impact': row['news_impact']
                    }
                    
                    feature_vector = self._extract_features(signal_data, signal_type)
                    if feature_vector:
                        features.append(feature_vector)
                        labels.append(row['success'])
                        
                except:
                    continue
            
            if len(features) < 10:
                return
            
            # تقسیم داده‌ها
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, random_state=42
            )
            
            # نرمال‌سازی
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # آموزش مدل
            model = config['model']
            model.fit(X_train_scaled, y_train)
            
            # ارزیابی
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            
            # ذخیره مدل
            self.models[signal_type] = model
            self.scalers[signal_type] = scaler
            
            # ذخیره عملکرد
            self._save_model_performance(signal_type, accuracy, precision, recall, len(y_test))
            
            # ذخیره مدل در فایل
            self._save_model_to_file(signal_type, model, scaler)
            
            logging.info(f"مدل {signal_type} آموزش داده شد - دقت: {accuracy:.3f}")
            
        except Exception as e:
            logging.error(f"خطا در آموزش مدل {signal_type}: {e}")

    def _save_model_performance(self, model_type: str, accuracy: float, 
                               precision: float, recall: float, total_predictions: int):
        """ذخیره عملکرد مدل"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO model_performance (
                    pair, model_type, accuracy, precision_score, 
                    recall_score, total_predictions, successful_predictions
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                'EUR/USD', model_type, accuracy, precision, recall,
                total_predictions, int(total_predictions * accuracy)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.error(f"خطا در ذخیره عملکرد مدل: {e}")

    def _save_model_to_file(self, model_type: str, model, scaler):
        """ذخیره مدل در فایل"""
        try:
            models_dir = "saved_models"
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)
            
            model_path = f"{models_dir}/{model_type}_model.joblib"
            scaler_path = f"{models_dir}/{model_type}_scaler.joblib"
            
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            
        except Exception as e:
            logging.error(f"خطا در ذخیره مدل به فایل: {e}")

    def get_learning_stats(self) -> Dict:
        """دریافت آمار یادگیری"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # آمار کلی
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM signal_results')
            total_signals = cursor.fetchone()[0]
            
            cursor.execute('SELECT AVG(success) FROM signal_results WHERE timestamp > datetime("now", "-30 days")')
            recent_win_rate = cursor.fetchone()[0] or 0
            
            # آمار مدل‌ها
            model_stats = {}
            for model_type in self.models.keys():
                cursor.execute('''
                    SELECT accuracy, precision_score, recall_score
                    FROM model_performance
                    WHERE model_type = ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                ''', (model_type,))
                
                result = cursor.fetchone()
                if result:
                    model_stats[model_type] = {
                        'accuracy': result[0],
                        'precision': result[1],
                        'recall': result[2]
                    }
            
            conn.close()
            
            return {
                'total_signals_recorded': total_signals,
                'recent_win_rate': recent_win_rate,
                'active_models': list(self.models.keys()),
                'model_performance': model_stats,
                'learning_enabled': True
            }
            
        except Exception as e:
            logging.error(f"خطا در دریافت آمار یادگیری: {e}")
            return {'learning_enabled': False, 'error': str(e)}
