import numpy as np
import pandas as pd
from typing import List, Union, Tuple, Optional
import statistics
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings

class MathUtils:
    """
    ابزارهای ریاضی و محاسباتی
    """
    
    @staticmethod
    def safe_divide(a: float, b: float, default: float = 0.0) -> float:
        """تقسیم ایمن"""
        try:
            if b == 0 or np.isnan(b) or np.isinf(b):
                return default
            
            result = a / b
            
            if np.isnan(result) or np.isinf(result):
                return default
            
            return result
            
        except Exception:
            return default
    
    @staticmethod
    def safe_log(x: float, base: float = np.e) -> float:
        """لگاریتم ایمن"""
        try:
            if x <= 0:
                return 0.0
            
            if base == np.e:
                result = np.log(x)
            else:
                result = np.log(x) / np.log(base)
            
            if np.isnan(result) or np.isinf(result):
                return 0.0
            
            return result
            
        except Exception:
            return 0.0
    
    @staticmethod
    def calculate_percentage_change(old_value: float, new_value: float) -> float:
        """محاسبه درصد تغییر"""
        return MathUtils.safe_divide((new_value - old_value) * 100, old_value)
    
    @staticmethod
    def calculate_compound_return(returns: List[float]) -> float:
        """محاسبه بازده مرکب"""
        try:
            if not returns:
                return 0.0
            
            compound = 1.0
            for ret in returns:
                compound *= (1 + ret / 100)
            
            return (compound - 1) * 100
            
        except Exception:
            return 0.0
    
    @staticmethod
    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
        """محاسبه نسبت شارپ"""
        try:
            if not returns or len(returns) < 2:
                return 0.0
            
            mean_return = np.mean(returns)
            std_return = np.std(returns, ddof=1)
            
            if std_return == 0:
                return 0.0
            
            return (mean_return - risk_free_rate) / std_return
            
        except Exception:
            return 0.0
    
    @staticmethod
    def calculate_sortino_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
        """محاسبه نسبت سورتینو"""
        try:
            if not returns:
                return 0.0
            
            mean_return = np.mean(returns)
            negative_returns = [r for r in returns if r < 0]
            
            if not negative_returns:
                return float('inf') if mean_return > risk_free_rate else 0.0
            
            downside_deviation = np.std(negative_returns, ddof=1)
            
            if downside_deviation == 0:
                return 0.0
            
            return (mean_return - risk_free_rate) / downside_deviation
            
        except Exception:
            return 0.0
    
    @staticmethod
    def calculate_max_drawdown(prices: List[float]) -> dict:
        """محاسبه حداکثر افت"""
        try:
            if not prices or len(prices) < 2:
                return {'max_drawdown': 0.0, 'start_idx': 0, 'end_idx': 0}
            
            prices = np.array(prices)
            cumulative_max = np.maximum.accumulate(prices)
            drawdowns = (prices - cumulative_max) / cumulative_max * 100
            
            max_drawdown_idx = np.argmin(drawdowns)
            max_drawdown = drawdowns[max_drawdown_idx]
            
            # پیدا کردن شروع drawdown
            start_idx = 0
            for i in range(max_drawdown_idx, -1, -1):
                if drawdowns[i] == 0:
                    start_idx = i
                    break
            
            return {
                'max_drawdown': float(max_drawdown),
                'start_idx': int(start_idx),
                'end_idx': int(max_drawdown_idx),
                'duration': int(max_drawdown_idx - start_idx)
            }
            
        except Exception:
            return {'max_drawdown': 0.0, 'start_idx': 0, 'end_idx': 0}
    
    @staticmethod
    def calculate_volatility(returns: List[float], annualize: bool = True) -> float:
        """محاسبه نوسانات"""
        try:
            if not returns or len(returns) < 2:
                return 0.0
            
            volatility = np.std(returns, ddof=1)
            
            if annualize:
                # فرض: بازده‌های روزانه
                volatility *= np.sqrt(365)
            
            return float(volatility)
            
        except Exception:
            return 0.0
    
    @staticmethod
    def calculate_correlation(x: List[float], y: List[float]) -> float:
        """محاسبه همبستگی"""
        try:
            if not x or not y or len(x) != len(y) or len(x) < 2:
                return 0.0
            
            correlation, _ = stats.pearsonr(x, y)
            
            if np.isnan(correlation):
                return 0.0
            
            return float(correlation)
            
        except Exception:
            return 0.0
    
    @staticmethod
    def calculate_beta(asset_returns: List[float], market_returns: List[float]) -> float:
        """محاسبه بتا"""
        try:
            if not asset_returns or not market_returns or len(asset_returns) != len(market_returns):
                return 1.0
            
            covariance = np.cov(asset_returns, market_returns)[0][1]
            market_variance = np.var(market_returns, ddof=1)
            
            if market_variance == 0:
                return 1.0
            
            beta = covariance / market_variance
            
            if np.isnan(beta) or np.isinf(beta):
                return 1.0
            
            return float(beta)
            
        except Exception:
            return 1.0
    
    @staticmethod
    def calculate_var(returns: List[float], confidence_level: float = 0.05) -> float:
        """محاسبه ارزش در معرض ریسک (VaR)"""
        try:
            if not returns:
                return 0.0
            
            var = np.percentile(returns, confidence_level * 100)
            return float(var)
            
        except Exception:
            return 0.0
    
    @staticmethod
    def calculate_cvar(returns: List[float], confidence_level: float = 0.05) -> float:
        """محاسبه ارزش شرطی در معرض ریسک (CVaR)"""
        try:
            if not returns:
                return 0.0
            
            var = MathUtils.calculate_var(returns, confidence_level)
            cvar_returns = [r for r in returns if r <= var]
            
            if not cvar_returns:
                return var
            
            return float(np.mean(cvar_returns))
            
        except Exception:
            return 0.0
    
    @staticmethod
    def normalize_data(data: List[float], method: str = 'z_score') -> List[float]:
        """نرمال‌سازی داده‌ها"""
        try:
            if not data:
                return []
            
            data_array = np.array(data).reshape(-1, 1)
            
            if method == 'z_score':
                scaler = StandardScaler()
                normalized = scaler.fit_transform(data_array)
            
            elif method == 'min_max':
                scaler = MinMaxScaler()
                normalized = scaler.fit_transform(data_array)
            
            elif method == 'robust':
                median = np.median(data)
                mad = np.median(np.abs(data - median))
                normalized = (data_array - median) / (mad * 1.4826)
            
            else:
                return data
            
            return normalized.flatten().tolist()
            
        except Exception:
            return data
    
    @staticmethod
    def calculate_moving_average(data: List[float], window: int, 
                               type_ma: str = 'simple') -> List[float]:
        """محاسبه میانگین متحرک"""
        try:
            if not data or window <= 0 or window > len(data):
                return data
            
            if type_ma == 'simple':
                # میانگین ساده
                ma = []
                for i in range(len(data)):
                    if i < window - 1:
                        ma.append(np.nan)
                    else:
                        ma.append(np.mean(data[i-window+1:i+1]))
                return ma
            
            elif type_ma == 'exponential':
                # میانگین نمایی
                alpha = 2 / (window + 1)
                ema = [data[0]]
                
                for i in range(1, len(data)):
                    ema.append(alpha * data[i] + (1 - alpha) * ema[-1])
                
                return ema
            
            else:
                return data
                
        except Exception:
            return data
    
    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> List[float]:
        """محاسبه RSI"""
        try:
            if not prices or len(prices) < period + 1:
                return [50.0] * len(prices)
            
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            # میانگین اولیه
            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])
            
            rsi = [50.0] * (period + 1)  # مقادیر اولیه
            
            for i in range(period, len(deltas)):
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period
                
                if avg_loss == 0:
                    rsi.append(100.0)
                else:
                    rs = avg_gain / avg_loss
                    rsi_value = 100 - (100 / (1 + rs))
                    rsi.append(rsi_value)
            
            return rsi
            
        except Exception:
            return [50.0] * len(prices)

class StatisticalAnalyzer:
    """
    تحلیلگر آماری پیشرفته
    """
    
    @staticmethod
    def detect_outliers(data: List[float], method: str = 'iqr') -> dict:
        """تشخیص داده‌های پرت"""
        try:
            if not data or len(data) < 4:
                return {'outliers': [], 'indices': [], 'method': method}
            
            data_array = np.array(data)
            
            if method == 'iqr':
                Q1 = np.percentile(data_array, 25)
                Q3 = np.percentile(data_array, 75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (data_array < lower_bound) | (data_array > upper_bound)
            
            elif method == 'z_score':
                z_scores = np.abs(stats.zscore(data_array))
                outlier_mask = z_scores > 3
            
            elif method == 'modified_z_score':
                median = np.median(data_array)
                mad = np.median(np.abs(data_array - median))
                modified_z_scores = 0.6745 * (data_array - median) / mad
                outlier_mask = np.abs(modified_z_scores) > 3.5
            
            else:
                outlier_mask = np.zeros(len(data), dtype=bool)
            
            outliers = data_array[outlier_mask].tolist()
            indices = np.where(outlier_mask)[0].tolist()
            
            return {
                'outliers': outliers,
                'indices': indices,
                'count': len(outliers),
                'percentage': (len(outliers) / len(data)) * 100,
                'method': method
            }
            
        except Exception:
            return {'outliers': [], 'indices': [], 'method': method}
    
    @staticmethod
    def test_normality(data: List[float]) -> dict:
        """تست نرمال بودن داده‌ها"""
        try:
            if not data or len(data) < 8:
                return {'is_normal': False, 'p_value': 0.0, 'test': 'insufficient_data'}
            
            # تست Shapiro-Wilk
            statistic, p_value = stats.shapiro(data)
            
            is_normal = p_value > 0.05
            
            return {
                'is_normal': is_normal,
                'p_value': float(p_value),
                'statistic': float(statistic),
                'test': 'shapiro_wilk',
                'alpha': 0.05
            }
            
        except Exception:
            return {'is_normal': False, 'p_value': 0.0, 'test': 'error'}
    
    @staticmethod
    def calculate_descriptive_stats(data: List[float]) -> dict:
        """محاسبه آمار توصیفی"""
        try:
            if not data:
                return {}
            
            data_array = np.array(data)
            
            stats_dict = {
                'count': len(data),
                'mean': float(np.mean(data_array)),
                'median': float(np.median(data_array)),
                'mode': float(statistics.mode(data)) if len(set(data)) < len(data) else None,
                'std': float(np.std(data_array, ddof=1)) if len(data) > 1 else 0.0,
                'var': float(np.var(data_array, ddof=1)) if len(data) > 1 else 0.0,
                'min': float(np.min(data_array)),
                'max': float(np.max(data_array)),
                'range': float(np.max(data_array) - np.min(data_array)),
                'q25': float(np.percentile(data_array, 25)),
                'q75': float(np.percentile(data_array, 75)),
                'iqr': float(np.percentile(data_array, 75) - np.percentile(data_array, 25)),
                'skewness': float(stats.skew(data_array)),
                'kurtosis': float(stats.kurtosis(data_array))
            }
            
            return stats_dict
            
        except Exception:
            return {}
    
    @staticmethod
    def perform_stationarity_test(data: List[float]) -> dict:
        """تست ایستایی (Stationarity)"""
        try:
            from statsmodels.tsa.stattools import adfuller
            
            if not data or len(data) < 10:
                return {'is_stationary': False, 'test': 'insufficient_data'}
            
            # Augmented Dickey-Fuller test
            result = adfuller(data, autolag='AIC')
            
            is_stationary = result[1] <= 0.05
            
            return {
                'is_stationary': is_stationary,
                'adf_statistic': float(result[0]),
                'p_value': float(result[1]),
                'critical_values': {k: float(v) for k, v in result[4].items()},
                'test': 'augmented_dickey_fuller'
            }
            
        except ImportError:
            # اگر statsmodels موجود نیست
            return {'is_stationary': False, 'test': 'statsmodels_not_available'}
        except Exception:
            return {'is_stationary': False, 'test': 'error'}
