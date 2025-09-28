import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
import pickle
import os
from pathlib import Path

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    from sklearn.preprocessing import StandardScaler, RobustScaler
    import joblib
    ML_AVAILABLE = True
except ImportError:
    print("⚠️ sklearn کتابخانه‌های ML در دسترس نیستند")
    ML_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from .feature_engineering import FeatureEngineer
from .signal_model import TradingSignal, SignalDirection, SignalQuality

class MLPredictor:
    """
    پیش‌بین یادگیری ماشین برای سیگنال‌های معاملاتی
    """
    
    def __init__(self, model_path: str = "models/ml_models"):
        self.model_path = Path(model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # تنظیم مدل‌ها
        self.models = {}
        self.scalers = {}
        self.feature_engineer = FeatureEngineer() if ML_AVAILABLE else None
        self.is_available = ML_AVAILABLE
        
        # تنظیمات مدل
        self.model_configs = {
            'random_forest': {
                'model': RandomForestClassifier,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42
                }
            },
            'logistic_regression': {
                'model': LogisticRegression,
                'params': {
                    'random_state': 42,
                    'max_iter': 1000
                }
            }
        }
        
        if XGBOOST_AVAILABLE:
            self.model_configs['xgboost'] = {
                'model': xgb.XGBClassifier,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42
                }
            }
        
        # آمار عملکرد
        self.performance_metrics = {}
        self.training_history = []
        
        print(f"✅ MLPredictor راه‌اندازی شد - ML Available: {self.is_available}")
    
    def prepare_training_data(self, signals: List[TradingSignal], 
                            market_data: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """آماده‌سازی داده‌های آموزش"""
        if not self.is_available:
            return np.array([]), np.array([])
        
        features_list = []
        targets = []
        
        for signal in signals:
            try:
                # استخراج ویژگی‌ها
                symbol_data = market_data.get(signal.symbol)
                if symbol_data is None:
                    continue
                
                # ویژگی‌های فنی
                features = self.feature_engineer.create_features(symbol_data)
                
                # ویژگی‌های سیگنال
                signal_features = self._extract_signal_features(signal)
                
                # ترکیب ویژگی‌ها
                combined_features = np.concatenate([features.iloc[-1].values, signal_features])
                features_list.append(combined_features)
                
                # هدف: موفقیت سیگنال (1 برای موفق، 0 برای ناموفق)
                target = 1 if signal.pnl and signal.pnl > 0 else 0
                targets.append(target)
                
            except Exception as e:
                print(f"خطا در آماده‌سازی داده سیگنال {signal.signal_id}: {e}")
                continue
        
        if not features_list:
            return np.array([]), np.array([])
        
        return np.array(features_list), np.array(targets)
    
    def _extract_signal_features(self, signal: TradingSignal) -> np.ndarray:
        """استخراج ویژگی‌های مربوط به سیگنال"""
        features = []
        
        # ویژگی‌های اساسی
        features.append(1 if signal.direction == SignalDirection.LONG else 0)
        features.append(1 if signal.quality == SignalQuality.HIGH else 0)
        features.append(signal.risk_metrics.risk_reward_ratio if signal.risk_metrics.risk_reward_ratio else 0)
        
        # ویژگی‌های Price Action
        if signal.price_action:
            features.extend([
                signal.price_action.body_to_range_ratio or 0,
                1 if signal.price_action.is_hammer else 0,
                1 if signal.price_action.is_engulfing else 0,
                1 if signal.price_action.is_pin_bar else 0,
                signal.price_action.support_strength or 0,
                signal.price_action.resistance_strength or 0
            ])
        else:
            features.extend([0] * 6)
        
        # ویژگی‌های بازار
        if signal.market_context:
            features.extend([
                1 if signal.market_context.is_trending else 0,
                1 if signal.market_context.is_breakout_environment else 0,
                signal.market_context.volatility_percentile or 0,
                1 if signal.market_context.volume_breakout else 0
            ])
        else:
            features.extend([0] * 4)
        
        # ویژگی‌های AI
        if signal.ai_evaluation:
            features.extend([
                signal.ai_evaluation.consensus_score,
                signal.ai_evaluation.avg_confidence,
                1 if signal.ai_evaluation.unanimous_agreement else 0
            ])
        else:
            features.extend([0] * 3)
        
        return np.array(features)
    
    def train_models(self, X: np.ndarray, y: np.ndarray, 
                    symbol: str = "ALL", test_size: float = 0.2) -> Dict[str, Any]:
        """آموزش مدل‌ها"""
        if not self.is_available or len(X) == 0:
            return {'success': False, 'message': 'ML not available or no data'}
        
        try:
            # تقسیم داده‌ها
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # نرمال‌سازی
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # ذخیره scaler
            self.scalers[symbol] = scaler
            
            results = {}
            best_model = None
            best_score = 0
            
            # آموزش هر مدل
            for model_name, config in self.model_configs.items():
                try:
                    print(f"آموزش مدل {model_name} برای {symbol}...")
                    
                    # ایجاد مدل
                    model = config['model'](**config['params'])
                    
                    # آموزش
                    model.fit(X_train_scaled, y_train)
                    
                    # ارزیابی
                    train_score = model.score(X_train_scaled, y_train)
                    test_score = model.score(X_test_scaled, y_test)
                    
                    # Cross validation
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
                    
                    # پیش‌بینی برای محاسبه معیارهای بیشتر
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
                    
                    model_results = {
                        'train_accuracy': train_score,
                        'test_accuracy': test_score,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'classification_report': classification_report(y_test, y_pred, output_dict=True)
                    }
                    
                    if y_pred_proba is not None:
                        model_results['auc_score'] = roc_auc_score(y_test, y_pred_proba)
                    
                    results[model_name] = model_results
                    
                    # انتخاب بهترین مدل بر اساس CV score
                    if cv_scores.mean() > best_score:
                        best_score = cv_scores.mean()
                        best_model = (model_name, model)
                    
                    print(f"✅ {model_name}: CV Score = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
                    
                except Exception as e:
                    print(f"خطا در آموزش {model_name}: {e}")
                    continue
            
            # ذخیره بهترین مدل
            if best_model:
                model_name, model = best_model
                self.models[symbol] = {
                    'model': model,
                    'model_type': model_name,
                    'trained_at': datetime.now(),
                    'performance': results[model_name]
                }
                
                # ذخیره در فایل
                self._save_model(symbol, model, scaler, model_name)
                
                print(f"🏆 بهترین مدل برای {symbol}: {model_name} (Score: {best_score:.3f})")
            
            # ذخیره آمار آموزش
            training_record = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'samples_count': len(X),
                'test_size': test_size,
                'results': results,
                'best_model': best_model[0] if best_model else None
            }
            self.training_history.append(training_record)
            
            return {
                'success': True,
                'results': results,
                'best_model': best_model[0] if best_model else None,
                'samples_count': len(X)
            }
            
        except Exception as e:
            print(f"خطا در آموزش مدل‌ها: {e}")
            return {'success': False, 'message': str(e)}
    
    def predict_signal_success(self, signal: TradingSignal, 
                             market_data: pd.DataFrame) -> Dict[str, Any]:
        """پیش‌بینی احتمال موفقیت سیگنال"""
        if not self.is_available:
            return {
                'prediction_available': False,
                'success_probability': 0.7,  # مقدار پیش‌فرض
                'confidence': 0.0,
                'model_used': 'none'
            }
        
        symbol = signal.symbol
        
        # بارگذاری مدل اگر در حافظه نیست
        if symbol not in self.models:
            if not self._load_model(symbol):
                # استفاده از مدل عمومی
                if 'ALL' not in self.models and not self._load_model('ALL'):
                    return {
                        'prediction_available': False,
                        'success_probability': 0.7,
                        'confidence': 0.0,
                        'model_used': 'none'
                    }
                symbol = 'ALL'
        
        try:
            # آماده‌سازی ویژگی‌ها
            features = self.feature_engineer.create_features(market_data)
            signal_features = self._extract_signal_features(signal)
            combined_features = np.concatenate([features.iloc[-1].values, signal_features])
            
            # نرمال‌سازی
            scaler = self.scalers.get(symbol)
            if scaler:
                combined_features = scaler.transform(combined_features.reshape(1, -1))
            else:
                combined_features = combined_features.reshape(1, -1)
            
            # پیش‌بینی
            model_info = self.models[symbol]
            model = model_info['model']
            
            # احتمال موفقیت
            if hasattr(model, 'predict_proba'):
                success_probability = model.predict_proba(combined_features)[0, 1]
                confidence = max(success_probability, 1 - success_probability)
            else:
                prediction = model.predict(combined_features)[0]
                success_probability = 0.8 if prediction == 1 else 0.2
                confidence = 0.6
            
            return {
                'prediction_available': True,
                'success_probability': float(success_probability),
                'confidence': float(confidence),
                'model_used': model_info['model_type'],
                'trained_at': model_info['trained_at'].isoformat()
            }
            
        except Exception as e:
            print(f"خطا در پیش‌بینی: {e}")
            return {
                'prediction_available': False,
                'success_probability': 0.7,
                'confidence': 0.0,
                'model_used': 'error',
                'error': str(e)
            }
    
    def retrain_with_new_data(self, new_signals: List[TradingSignal], 
                            market_data: Dict[str, pd.DataFrame], 
                            symbol: str = "ALL") -> bool:
        """بازآموزی با داده‌های جدید"""
        if not self.is_available:
            return False
        
        try:
            # آماده‌سازی داده‌های جدید
            X_new, y_new = self.prepare_training_data(new_signals, market_data)
            
            if len(X_new) < 10:  # حداقل 10 نمونه
                print(f"داده‌های کافی برای بازآموزی وجود ندارد: {len(X_new)} نمونه")
                return False
            
            # بازآموزی
            result = self.train_models(X_new, y_new, symbol)
            
            if result['success']:
                print(f"✅ بازآموزی موفق برای {symbol} با {len(X_new)} نمونه جدید")
                return True
            else:
                print(f"❌ خطا در بازآموزی: {result.get('message', 'Unknown error')}")
                return False
                
        except Exception as e:
            print(f"خطا در بازآموزی: {e}")
            return False
    
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray, 
                                model_name: str = 'random_forest') -> Dict[str, Any]:
        """بهینه‌سازی هایپرپارامترها"""
        if not self.is_available or model_name not in self.model_configs:
            return {}
        
        try:
            # تعریف فضای جستجو
            param_grids = {
                'random_forest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'gradient_boosting': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2]
                },
                'logistic_regression': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            }
            
            if XGBOOST_AVAILABLE and model_name == 'xgboost':
                param_grids['xgboost'] = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0]
                }
            
            # ایجاد مدل
            model_class = self.model_configs[model_name]['model']
            base_params = self.model_configs[model_name]['params'].copy()
            base_params.pop('random_state', None)  # حذف برای Grid Search
            
            model = model_class(random_state=42)
            
            # Grid Search
            grid_search = GridSearchCV(
                model, 
                param_grids[model_name],
                cv=5, 
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            
            # نرمال‌سازی داده‌ها
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            
            print(f"شروع بهینه‌سازی هایپرپارامترها برای {model_name}...")
            grid_search.fit(X_scaled, y)
            
            # نتایج
            results = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            
            print(f"✅ بهترین پارامترها: {grid_search.best_params_}")
            print(f"✅ بهترین امتیاز: {grid_search.best_score_:.4f}")
            
            return results
            
        except Exception as e:
            print(f"خطا در بهینه‌سازی: {e}")
            return {}
    
    def _save_model(self, symbol: str, model: Any, scaler: Any, model_type: str):
        """ذخیره مدل در فایل"""
        try:
            symbol_clean = symbol.replace('/', '_').replace('-', '_')
            
            # ذخیره مدل
            model_file = self.model_path / f"{symbol_clean}_{model_type}_model.joblib"
            joblib.dump(model, model_file)
            
            # ذخیره scaler
            scaler_file = self.model_path / f"{symbol_clean}_{model_type}_scaler.joblib"
            joblib.dump(scaler, scaler_file)
            
            # ذخیره metadata
            metadata = {
                'symbol': symbol,
                'model_type': model_type,
                'saved_at': datetime.now().isoformat(),
                'feature_count': getattr(model, 'n_features_in_', 0)
            }
            
            metadata_file = self.model_path / f"{symbol_clean}_{model_type}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ مدل {model_type} برای {symbol} ذخیره شد")
            
        except Exception as e:
            print(f"خطا در ذخیره مدل: {e}")
    
    def _load_model(self, symbol: str) -> bool:
        """بارگذاری مدل از فایل"""
        try:
            symbol_clean = symbol.replace('/', '_').replace('-', '_')
            
            # یافتن فایل‌های مدل
            model_files = list(self.model_path.glob(f"{symbol_clean}_*_model.joblib"))
            
            if not model_files:
                return False
            
            # بارگذاری آخرین مدل (بر اساس نام فایل)
            latest_model_file = sorted(model_files)[-1]
            model_type = latest_model_file.stem.split('_')[-2]  # استخراج نوع مدل از نام فایل
            
            # بارگذاری مدل
            model = joblib.load(latest_model_file)
            
            # بارگذاری scaler
            scaler_file = self.model_path / f"{symbol_clean}_{model_type}_scaler.joblib"
            if scaler_file.exists():
                scaler = joblib.load(scaler_file)
                self.scalers[symbol] = scaler
            
            # بارگذاری metadata
            metadata_file = self.model_path / f"{symbol_clean}_{model_type}_metadata.json"
            metadata = {}
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            
            # ذخیره در حافظه
            self.models[symbol] = {
                'model': model,
                'model_type': model_type,
                'trained_at': datetime.fromisoformat(metadata.get('saved_at', datetime.now().isoformat())),
                'performance': {}
            }
            
            print(f"✅ مدل {model_type} برای {symbol} بارگذاری شد")
            return True
            
        except Exception as e:
            print(f"خطا در بارگذاری مدل برای {symbol}: {e}")
            return False
    
    def get_model_performance(self, symbol: str = None) -> Dict[str, Any]:
        """دریافت عملکرد مدل‌ها"""
        if symbol and symbol in self.models:
            return self.models[symbol].get('performance', {})
        
        # عملکرد همه مدل‌ها
        performance = {}
        for sym, model_info in self.models.items():
            performance[sym] = model_info.get('performance', {})
        
        return performance
    
    def get_feature_importance(self, symbol: str) -> Optional[Dict[str, float]]:
        """دریافت اهمیت ویژگی‌ها"""
        if symbol not in self.models:
            return None
        
        model = self.models[symbol]['model']
        
        if hasattr(model, 'feature_importances_'):
            # برای Random Forest, Gradient Boosting, XGBoost
            importances = model.feature_importances_
            
            # نام ویژگی‌ها (باید از FeatureEngineer بگیریم)
            feature_names = self.feature_engineer.get_feature_names() if self.feature_engineer else []
            
            # اضافه کردن نام‌های ویژگی‌های سیگنال
            signal_feature_names = [
                'direction_long', 'quality_high', 'risk_reward_ratio',
                'body_to_range_ratio', 'is_hammer', 'is_engulfing', 'is_pin_bar',
                'support_strength', 'resistance_strength',
                'is_trending', 'is_breakout_env', 'volatility_percentile', 'volume_breakout',
                'ai_consensus_score', 'ai_avg_confidence', 'ai_unanimous_agreement'
            ]
            
            all_feature_names = feature_names + signal_feature_names
            
            if len(all_feature_names) == len(importances):
                return dict(zip(all_feature_names, importances))
            else:
                # اگر نام‌ها نمی‌خوانند، شماره‌گذاری کن
                return {f'feature_{i}': imp for i, imp in enumerate(importances)}
        
        return None
    
    def export_model_summary(self, output_file: str = "model_summary.json"):
        """خروجی خلاصه مدل‌ها"""
        try:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'ml_available': self.is_available,
                'models_count': len(self.models),
                'training_history_count': len(self.training_history),
                'models': {},
                'training_history': self.training_history[-10:]  # آخرین 10 آموزش
            }
            
            for symbol, model_info in self.models.items():
                summary['models'][symbol] = {
                    'model_type': model_info['model_type'],
                    'trained_at': model_info['trained_at'].isoformat(),
                    'performance': model_info.get('performance', {}),
                    'feature_importance': self.get_feature_importance(symbol)
                }
            
            # ذخیره در فایل
            output_path = self.model_path / output_file
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            print(f"✅ خلاصه مدل‌ها در {output_path} ذخیره شد")
            return summary
            
        except Exception as e:
            print(f"خطا در خروجی خلاصه: {e}")
            return {}
