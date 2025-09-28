"""
Correlation Analyzer - تحلیلگر همبستگی بین ارزهای دیجیتال
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import math

class CorrelationAnalyzer:
    """تحلیلگر همبستگی برای شناسایی روابط بین نمادهای مختلف"""
    
    def __init__(self):
        self.correlation_cache = {}
        self.correlation_history = []
        self.risk_thresholds = {
            'high_correlation': 0.8,
            'medium_correlation': 0.6,
            'low_correlation': 0.3
        }
        
        # ماتریس وزن برای محاسبه ریسک همبستگی
        self.weight_matrix = {
            'price_correlation': 0.4,
            'volume_correlation': 0.2,
            'volatility_correlation': 0.2,
            'trend_correlation': 0.2
        }
        
    async def analyze_correlations(self, symbols_data: Dict[str, pd.DataFrame], 
                                 analysis_results: Dict) -> Dict:
        """تحلیل جامع همبستگی بین تمام نمادها"""
        
        # محاسبه ماتریس همبستگی
        correlation_matrix = self._calculate_correlation_matrix(symbols_data)
        
        # تحلیل همبستگی با بیت‌کوین
        btc_correlations = self._analyze_btc_correlations(symbols_data)
        
        # تحلیل همبستگی حجم
        volume_correlations = self._analyze_volume_correlations(symbols_data)
        
        # تحلیل همبستگی نوسانات
        volatility_correlations = self._analyze_volatility_correlations(symbols_data)
        
        # تحلیل همبستگی سیگنال‌ها
        signal_correlations = self._analyze_signal_correlations(analysis_results)
        
        # تشخیص cluster های همبستگی
        correlation_clusters = self._detect_correlation_clusters(correlation_matrix)
        
        # ارزیابی ریسک همبستگی
        correlation_risk = self._assess_correlation_risk(
            correlation_matrix, signal_correlations
        )
        
        # هشدارهای همبستگی
        correlation_warnings = self._generate_correlation_warnings(
            correlation_matrix, signal_correlations
        )
        
        return {
            'correlation_matrix': correlation_matrix,
            'btc_correlations': btc_correlations,
            'volume_correlations': volume_correlations,
            'volatility_correlations': volatility_correlations,
            'signal_correlations': signal_correlations,
            'correlation_clusters': correlation_clusters,
            'correlation_risk': correlation_risk,
            'warnings': correlation_warnings,
            'market_regime': self._determine_market_regime(correlation_matrix),
            'diversification_score': self._calculate_diversification_score(correlation_matrix),
            'analysis_timestamp': datetime.now()
        }
    
    def _calculate_correlation_matrix(self, symbols_data: Dict[str, pd.DataFrame]) -> Dict:
        """محاسبه ماتریس همبستگی قیمت"""
        symbols = list(symbols_data.keys())
        correlation_matrix = {}
        
        for symbol1 in symbols:
            correlation_matrix[symbol1] = {}
            for symbol2 in symbols:
                if symbol1 == symbol2:
                    correlation_matrix[symbol1][symbol2] = 1.0
                else:
                    corr = self._calculate_price_correlation(
                        symbols_data[symbol1], symbols_data[symbol2]
                    )
                    correlation_matrix[symbol1][symbol2] = corr
        
        return correlation_matrix
    
    def _calculate_price_correlation(self, data1: pd.DataFrame, data2: pd.DataFrame) -> float:
        """محاسبه همبستگی قیمت بین دو نماد"""
        try:
            # محاسبه بازدهی
            returns1 = data1['close'].pct_change().dropna()
            returns2 = data2['close'].pct_change().dropna()
            
            # هم‌زمان‌سازی داده‌ها
            min_length = min(len(returns1), len(returns2))
            returns1 = returns1.tail(min_length)
            returns2 = returns2.tail(min_length)
            
            # محاسبه همبستگی
            correlation = np.corrcoef(returns1, returns2)[0, 1]
            
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            print(f"خطا در محاسبه همبستگی: {e}")
            return 0.0
    
    def _analyze_btc_correlations(self, symbols_data: Dict[str, pd.DataFrame]) -> Dict:
        """تحلیل همبستگی تمام ارزها با بیت‌کوین"""
        btc_correlations = {}
        btc_symbol = 'BTC/USDT'
        
        if btc_symbol not in symbols_data:
            # اگر BTC در داده‌ها نیست، از ETH به عنوان benchmark استفاده می‌کنیم
            btc_symbol = 'ETH/USDT' if 'ETH/USDT' in symbols_data else list(symbols_data.keys())[0]
        
        btc_data = symbols_data[btc_symbol]
        
        for symbol, data in symbols_data.items():
            if symbol != btc_symbol:
                # همبستگی قیمت
                price_corr = self._calculate_price_correlation(btc_data, data)
                
                # همبستگی حجم
                volume_corr = self._calculate_volume_correlation(btc_data, data)
                
                # همبستگی نوسانات
                volatility_corr = self._calculate_volatility_correlation(btc_data, data)
                
                # محاسبه امتیاز کلی همبستگی
                overall_correlation = (
                    price_corr * 0.6 + 
                    volume_corr * 0.2 + 
                    volatility_corr * 0.2
                )
                
                btc_correlations[symbol] = {
                    'price_correlation': price_corr,
                    'volume_correlation': volume_corr,
                    'volatility_correlation': volatility_corr,
                    'overall_correlation': overall_correlation,
                    'correlation_strength': self._categorize_correlation_strength(overall_correlation)
                }
        
        return btc_correlations
    
    def _calculate_volume_correlation(self, data1: pd.DataFrame, data2: pd.DataFrame) -> float:
        """محاسبه همبستگی حجم"""
        try:
            volume1 = data1['volume'].pct_change().dropna()
            volume2 = data2['volume'].pct_change().dropna()
            
            min_length = min(len(volume1), len(volume2))
            volume1 = volume1.tail(min_length)
            volume2 = volume2.tail(min_length)
            
            correlation = np.corrcoef(volume1, volume2)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_volatility_correlation(self, data1: pd.DataFrame, data2: pd.DataFrame) -> float:
        """محاسبه همبستگی نوسانات"""
        try:
            # محاسبه نوسانات روزانه
            volatility1 = data1['close'].pct_change().rolling(24).std()
            volatility2 = data2['close'].pct_change().rolling(24).std()
            
            # حذف NaN
            volatility1 = volatility1.dropna()
            volatility2 = volatility2.dropna()
            
            min_length = min(len(volatility1), len(volatility2))
            volatility1 = volatility1.tail(min_length)
            volatility2 = volatility2.tail(min_length)
            
            correlation = np.corrcoef(volatility1, volatility2)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception:
            return 0.0
    
    def _analyze_volume_correlations(self, symbols_data: Dict[str, pd.DataFrame]) -> Dict:
        """تحلیل همبستگی حجم بین تمام نمادها"""
        symbols = list(symbols_data.keys())
        volume_matrix = {}
        
        for symbol1 in symbols:
            volume_matrix[symbol1] = {}
            for symbol2 in symbols:
                if symbol1 == symbol2:
                    volume_matrix[symbol1][symbol2] = 1.0
                else:
                    corr = self._calculate_volume_correlation(
                        symbols_data[symbol1], symbols_data[symbol2]
                    )
                    volume_matrix[symbol1][symbol2] = corr
        
        # تحلیل الگوهای حجم
        volume_patterns = self._analyze_volume_patterns(volume_matrix)
        
        return {
            'volume_correlation_matrix': volume_matrix,
            'volume_patterns': volume_patterns,
            'high_volume_correlation_pairs': self._find_high_correlation_pairs(volume_matrix)
        }
    
    def _analyze_volatility_correlations(self, symbols_data: Dict[str, pd.DataFrame]) -> Dict:
        """تحلیل همبستگی نوسانات"""
        symbols = list(symbols_data.keys())
        volatility_matrix = {}
        
        for symbol1 in symbols:
            volatility_matrix[symbol1] = {}
            for symbol2 in symbols:
                if symbol1 == symbol2:
                    volatility_matrix[symbol1][symbol2] = 1.0
                else:
                    corr = self._calculate_volatility_correlation(
                        symbols_data[symbol1], symbols_data[symbol2]
                    )
                    volatility_matrix[symbol1][symbol2] = corr
        
        return {
            'volatility_correlation_matrix': volatility_matrix,
            'volatility_clusters': self._detect_volatility_clusters(volatility_matrix)
        }
    
    def _analyze_signal_correlations(self, analysis_results: Dict) -> Dict:
        """تحلیل همبستگی سیگنال‌ها"""
        signal_correlations = {}
        symbols = list(analysis_results.keys())
        
        # بررسی همزمانی سیگنال‌ها
        simultaneous_signals = []
        opposite_signals = []
        
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols[i+1:], i+1):
                result1 = analysis_results.get(symbol1, {}).get('price_action', {})
                result2 = analysis_results.get(symbol2, {}).get('price_action', {})
                
                signal1 = result1.get('signal', 0)
                signal2 = result2.get('signal', 0)
                
                # بررسی سیگنال‌های همزمان
                if signal1 != 0 and signal2 != 0:
                    if signal1 == signal2:
                        simultaneous_signals.append({
                            'symbol1': symbol1,
                            'symbol2': symbol2,
                            'signal_type': 'BUY' if signal1 > 0 else 'SELL',
                            'confidence1': result1.get('confidence', 0),
                            'confidence2': result2.get('confidence', 0)
                        })
                    else:
                        opposite_signals.append({
                            'symbol1': symbol1,
                            'symbol2': symbol2,
                            'signal1': 'BUY' if signal1 > 0 else 'SELL',
                            'signal2': 'BUY' if signal2 > 0 else 'SELL'
                        })
        
        return {
            'simultaneous_signals': simultaneous_signals,
            'opposite_signals': opposite_signals,
            'signal_synchronization_rate': len(simultaneous_signals) / max(1, len(simultaneous_signals) + len(opposite_signals))
        }
    
    def _detect_correlation_clusters(self, correlation_matrix: Dict) -> List[List[str]]:
        """تشخیص خوشه‌های همبستگی"""
        symbols = list(correlation_matrix.keys())
        clusters = []
        used_symbols = set()
        
        for symbol in symbols:
            if symbol in used_symbols:
                continue
                
            cluster = [symbol]
            used_symbols.add(symbol)
            
            for other_symbol in symbols:
                if other_symbol != symbol and other_symbol not in used_symbols:
                    correlation = correlation_matrix[symbol].get(other_symbol, 0)
                    if abs(correlation) > self.risk_thresholds['high_correlation']:
                        cluster.append(other_symbol)
                        used_symbols.add(other_symbol)
            
            if len(cluster) > 1:
                clusters.append(cluster)
        
        return clusters
    
    def _detect_volatility_clusters(self, volatility_matrix: Dict) -> List[Dict]:
        """تشخیص خوشه‌های نوسانات"""
        clusters = []
        symbols = list(volatility_matrix.keys())
        
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols[i+1:], i+1):
                correlation = volatility_matrix[symbol1].get(symbol2, 0)
                
                if abs(correlation) > self.risk_thresholds['high_correlation']:
                    clusters.append({
                        'symbols': [symbol1, symbol2],
                        'volatility_correlation': correlation,
                        'risk_level': 'HIGH' if abs(correlation) > 0.9 else 'MEDIUM'
                    })
        
        return clusters
    
    def _assess_correlation_risk(self, correlation_matrix: Dict, signal_correlations: Dict) -> Dict:
        """ارزیابی ریسک همبستگی"""
        symbols = list(correlation_matrix.keys())
        
        # محاسبه میانگین همبستگی
        correlations = []
        for symbol1 in symbols:
            for symbol2 in symbols:
                if symbol1 != symbol2:
                    correlations.append(abs(correlation_matrix[symbol1][symbol2]))
        
        avg_correlation = np.mean(correlations) if correlations else 0
        max_correlation = np.max(correlations) if correlations else 0
        
        # ریسک بر اساس سیگنال‌های همزمان
        simultaneous_signals = len(signal_correlations.get('simultaneous_signals', []))
        
        # امتیاز ریسک کلی (0-100)
        risk_score = (
            avg_correlation * 40 +
            max_correlation * 30 +
            (simultaneous_signals / len(symbols)) * 30
        )
        
        return {
            'overall_risk_score': min(risk_score, 100),
            'risk_level': self._categorize_risk_level(risk_score),
            'average_correlation': avg_correlation,
            'maximum_correlation': max_correlation,
            'simultaneous_signals_count': simultaneous_signals,
            'diversification_benefit': 1 - avg_correlation
        }
    
    def _generate_correlation_warnings(self, correlation_matrix: Dict, signal_correlations: Dict) -> List[Dict]:
        """تولید هشدارهای همبستگی"""
        warnings = []
        
        # هشدار همبستگی بالا
        for symbol1, correlations in correlation_matrix.items():
            for symbol2, correlation in correlations.items():
                if symbol1 != symbol2 and abs(correlation) > self.risk_thresholds['high_correlation']:
                    warnings.append({
                        'type': 'HIGH_CORRELATION',
                        'severity': 'HIGH',
                        'message': f"همبستگی بالا بین {symbol1} و {symbol2}: {correlation:.3f}",
                        'symbols': [symbol1, symbol2],
                        'correlation': correlation,
                        'recommendation': "کاهش اندازه پوزیشن یا تنویع بیشتر"
                    })
        
        # هشدار سیگنال‌های همزمان
        simultaneous_signals = signal_correlations.get('simultaneous_signals', [])
        if len(simultaneous_signals) >= 2:
            warnings.append({
                'type': 'SIMULTANEOUS_SIGNALS',
                'severity': 'MEDIUM',
                'message': f"{len(simultaneous_signals)} سیگنال همزمان شناسایی شد",
                'signals': simultaneous_signals,
                'recommendation': "بررسی دقیق‌تر و مدیریت ریسک اضافی"
            })
        
        # هشدار ریسک سیستماتیک
        avg_correlation = np.mean([
            abs(correlation_matrix[s1][s2]) 
            for s1 in correlation_matrix 
            for s2 in correlation_matrix[s1] 
            if s1 != s2
        ])
        
        if avg_correlation > 0.7:
            warnings.append({
                'type': 'SYSTEMATIC_RISK',
                'severity': 'HIGH',
                'message': f"ریسک سیستماتیک بالا - میانگین همبستگی: {avg_correlation:.3f}",
                'average_correlation': avg_correlation,
                'recommendation': "کاهش اهرم و افزایش cash position"
            })
        
        return warnings
    
    def _determine_market_regime(self, correlation_matrix: Dict) -> str:
        """تعیین رژیم بازار بر اساس همبستگی"""
        symbols = list(correlation_matrix.keys())
        correlations = [
            abs(correlation_matrix[s1][s2]) 
            for s1 in symbols 
            for s2 in symbols 
            if s1 != s2
        ]
        
        avg_correlation = np.mean(correlations) if correlations else 0
        
        if avg_correlation > 0.8:
            return "CRISIS_MODE"  # بحران - همه چیز با هم حرکت می‌کند
        elif avg_correlation > 0.6:
            return "RISK_OFF"     # اجتناب از ریسک
        elif avg_correlation < 0.3:
            return "DIVERSIFIED"  # تنوع خوب
        else:
            return "NORMAL"       # عادی
    
    def _calculate_diversification_score(self, correlation_matrix: Dict) -> float:
        """محاسبه امتیاز تنوع (0-100)"""
        symbols = list(correlation_matrix.keys())
        correlations = [
            abs(correlation_matrix[s1][s2]) 
            for s1 in symbols 
            for s2 in symbols 
            if s1 != s2
        ]
        
        avg_correlation = np.mean(correlations) if correlations else 0
        diversification_score = (1 - avg_correlation) * 100
        
        return max(0, min(100, diversification_score))
    
    def _categorize_correlation_strength(self, correlation: float) -> str:
        """دسته‌بندی قدرت همبستگی"""
        abs_corr = abs(correlation)
        
        if abs_corr > 0.8:
            return "VERY_HIGH"
        elif abs_corr > 0.6:
            return "HIGH"
        elif abs_corr > 0.4:
            return "MODERATE"
        elif abs_corr > 0.2:
            return "LOW"
        else:
            return "VERY_LOW"
    
    def _categorize_risk_level(self, risk_score: float) -> str:
        """دسته‌بندی سطح ریسک"""
        if risk_score > 75:
            return "VERY_HIGH"
        elif risk_score > 50:
            return "HIGH"
        elif risk_score > 25:
            return "MODERATE"
        else:
            return "LOW"
    
    def _find_high_correlation_pairs(self, correlation_matrix: Dict) -> List[Dict]:
        """یافتن جفت‌های با همبستگی بالا"""
        high_corr_pairs = []
        symbols = list(correlation_matrix.keys())
        
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols[i+1:], i+1):
                correlation = correlation_matrix[symbol1].get(symbol2, 0)
                
                if abs(correlation) > self.risk_thresholds['high_correlation']:
                    high_corr_pairs.append({
                        'symbol1': symbol1,
                        'symbol2': symbol2,
                        'correlation': correlation,
                        'strength': self._categorize_correlation_strength(correlation)
                    })
        
        return sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)
    
    def _analyze_volume_patterns(self, volume_matrix: Dict) -> Dict:
        """تحلیل الگوهای حجم"""
        # الگوهای حجم مشترک
        return {
            'synchronized_volume_events': 0,  # تعداد رویدادهای حجم همزمان
            'volume_leadership': None,        # نمادی که رهبری حجم را دارد
            'volume_correlation_trend': 'STABLE'  # روند همبستگی حجم
        }
    
    def get_correlation_summary(self) -> Dict:
        """خلاصه وضعیت همبستگی"""
        if not self.correlation_history:
            return {'status': 'No data available'}
        
        latest = self.correlation_history[-1] if self.correlation_history else {}
        
        return {
            'current_market_regime': latest.get('market_regime', 'UNKNOWN'),
            'diversification_score': latest.get('diversification_score', 0),
            'correlation_risk_level': latest.get('correlation_risk', {}).get('risk_level', 'UNKNOWN'),
            'active_warnings_count': len(latest.get('warnings', [])),
            'last_analysis': latest.get('analysis_timestamp')
        }
    
    def update_correlation_cache(self, symbol_pair: Tuple[str, str], correlation: float):
        """به‌روزرسانی کش همبستگی"""
        cache_key = f"{symbol_pair[0]}_{symbol_pair[1]}"
        self.correlation_cache[cache_key] = {
            'correlation': correlation,
            'timestamp': datetime.now()
        }
        
        # پاک کردن کش قدیمی (بیش از یک ساعت)
        cutoff_time = datetime.now() - timedelta(hours=1)
        self.correlation_cache = {
            k: v for k, v in self.correlation_cache.items() 
            if v['timestamp'] > cutoff_time
        }
