"""
Analyzers Package - مجموعه تحلیلگرهای هوشمند
"""

from .price_action_analyzer import PriceActionAnalyzer
from .multi_ai_evaluator import MultiAIEvaluator
from .market_analyzer import MarketAnalyzer
from .risk_analyzer import RiskAnalyzer
from .correlation_analyzer import CorrelationAnalyzer
from .pattern_analyzer import PatternAnalyzer

class AnalysisManager:
    """مدیر مرکزی تحلیل‌ها"""
    
    def __init__(self, config_manager):
        self.config = config_manager
        self.price_action = PriceActionAnalyzer()
        self.ai_evaluator = MultiAIEvaluator(
            self.config.get_ai_config()
        )
        self.market_analyzer = MarketAnalyzer()
        self.risk_analyzer = RiskAnalyzer(
            self.config.get_risk_config()
        )
        self.correlation_analyzer = CorrelationAnalyzer()
        self.pattern_analyzer = PatternAnalyzer()
        
    async def run_comprehensive_analysis(self, symbols_data: dict):
        """تحلیل جامع تمام نمادها"""
        results = {}
        
        for symbol, data in symbols_data.items():
            print(f"🔍 تحلیل جامع {symbol}...")
            
            # تحلیل Price Action
            price_action_result = self.price_action.analyze(data)
            
            # تحلیل بازار کلی
            market_context = await self.market_analyzer.analyze_market_context(
                symbol, data
            )
            
            # تحلیل ریسک
            risk_assessment = self.risk_analyzer.assess_symbol_risk(
                symbol, data, price_action_result
            )
            
            # تحلیل الگوها
            patterns = await self.pattern_analyzer.detect_patterns(data)
            
            # ارزیابی AI
            if price_action_result.get('has_signal'):
                ai_evaluation = await self.ai_evaluator.evaluate_signal(
                    symbol, price_action_result, market_context
                )
            else:
                ai_evaluation = None
            
            results[symbol] = {
                'price_action': price_action_result,
                'market_context': market_context,
                'risk_assessment': risk_assessment,
                'patterns': patterns,
                'ai_evaluation': ai_evaluation,
                'analysis_timestamp': datetime.now()
            }
        
        # تحلیل همبستگی بین نمادها
        correlation_analysis = await self.correlation_analyzer.analyze_correlations(
            symbols_data, results
        )
        
        return {
            'individual_analysis': results,
            'correlation_analysis': correlation_analysis,
            'overall_market_sentiment': self._calculate_overall_sentiment(results)
        }
    
    def _calculate_overall_sentiment(self, results: dict) -> str:
        """محاسبه احساسات کلی بازار"""
        bullish_signals = 0
        bearish_signals = 0
        
        for symbol_result in results.values():
            price_action = symbol_result.get('price_action', {})
            if price_action.get('signal') == 1:
                bullish_signals += 1
            elif price_action.get('signal') == -1:
                bearish_signals += 1
        
        if bullish_signals > bearish_signals:
            return "BULLISH"
        elif bearish_signals > bullish_signals:
            return "BEARISH"
        else:
            return "NEUTRAL"

# Export کلاس‌ها
__all__ = [
    'PriceActionAnalyzer',
    'MultiAIEvaluator', 
    'MarketAnalyzer',
    'RiskAnalyzer',
    'CorrelationAnalyzer',
    'PatternAnalyzer',
    'AnalysisManager'
]
