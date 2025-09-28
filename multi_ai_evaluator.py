# analyzers/multi_ai_evaluator.py
import asyncio
import datetime
import json
import sqlite3
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ OpenAI library not installed")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️ Google Generative AI library not installed")

try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False
    print("⚠️ Anthropic library not installed")

import config

class MultiAIEvaluator:
    """
    سیستم ارزیابی چندگانه با AI های مختلف برای تحلیل سیگنال‌های معاملاتی
    """

    def __init__(self):
        self.db_path = "multi_crypto_analysis.db"
        self.ai_clients = {}
        self.ai_performance = {
            'gemini': {'correct': 0, 'total': 0, 'response_times': [], 'accuracy': 0.0},
            'openai': {'correct': 0, 'total': 0, 'response_times': [], 'accuracy': 0.0},
            'claude': {'correct': 0, 'total': 0, 'response_times': [], 'accuracy': 0.0}
        }
        
        # وزن‌های پویا برای هر AI
        self.ai_weights = {
            'gemini': 0.4,
            'openai': 0.35,
            'claude': 0.25
        }
        
        # آستانه‌های کیفیت
        self.quality_thresholds = {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        }
        
        self._setup_ai_clients()
        self._init_database()

    def _setup_ai_clients(self):
        """راه‌اندازی کلاینت‌های AI"""
        print("راه‌اندازی AI Clients...")
        
        # Gemini Setup
        if GEMINI_AVAILABLE and hasattr(config, 'GEMINI_API_KEY') and config.GEMINI_API_KEY:
            try:
                genai.configure(api_key=config.GEMINI_API_KEY)
                self.ai_clients['gemini'] = genai.GenerativeModel(
                    model_name=getattr(config, 'GEMINI_MODEL', 'gemini-pro')
                )
                print("✅ Gemini AI connected")
            except Exception as e:
                print(f"❌ Gemini setup failed: {e}")
        
        # OpenAI Setup
        if OPENAI_AVAILABLE and hasattr(config, 'OPENAI_API_KEY') and config.OPENAI_API_KEY:
            try:
                self.ai_clients['openai'] = openai.AsyncOpenAI(api_key=config.OPENAI_API_KEY)
                print("✅ OpenAI connected")
            except Exception as e:
                print("❌ OpenAI setup failed: {e}")
        
        # Claude Setup
        if CLAUDE_AVAILABLE and hasattr(config, 'CLAUDE_API_KEY') and config.CLAUDE_API_KEY:
            try:
                self.ai_clients['claude'] = anthropic.AsyncAnthropic(api_key=config.CLAUDE_API_KEY)
                print("✅ Claude AI connected")
            except Exception as e:
                print(f"❌ Claude setup failed: {e}")

    def _init_database(self):
        """راه‌اندازی پایگاه داده"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # جدول تحلیل‌های چندگانه
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS multi_ai_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT NOT NULL,
                    signal_type TEXT,
                    original_confidence REAL,
                    gemini_score REAL,
                    gemini_recommendation TEXT,
                    openai_score REAL,
                    openai_recommendation TEXT,
                    claude_score REAL,
                    claude_recommendation TEXT,
                    consensus_score REAL,
                    final_recommendation TEXT,
                    execution_time REAL,
                    market_conditions TEXT
                )
            ''')
            
            # جدول عملکرد AI ها
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ai_performance_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    ai_name TEXT,
                    signal_id TEXT,
                    prediction_score REAL,
                    actual_outcome REAL,
                    accuracy REAL,
                    response_time REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            print("✅ Multi-AI database initialized")
            
        except Exception as e:
            print(f"❌ Database initialization failed: {e}")

    async def evaluate_signal(self, analysis_data: Dict, symbol: str) -> Optional[Dict]:
        """ارزیابی جامع سیگنال با چندین AI"""
        start_time = time.time()
        
        try:
            if not analysis_data or analysis_data.get('signal', 0) == 0:
                return None
            
            print(f"🧠 Starting multi-AI evaluation for {symbol}...")
            
            # دریافت نظرات همه AI ها به صورت همزمان
            ai_evaluations = await self._get_all_ai_evaluations(analysis_data, symbol)
            
            if not ai_evaluations:
                print(f"❌ No AI evaluations received for {symbol}")
                return None
            
            # تحلیل consensus
            consensus_analysis = self._analyze_consensus(ai_evaluations)
            
            # تصمیم‌گیری نهایی
            final_decision = self._make_final_decision(analysis_data, ai_evaluations, consensus_analysis)
            
            # محاسبه execution time
            execution_time = time.time() - start_time
            
            # ذخیره در پایگاه داده
            self._save_evaluation_result(symbol, analysis_data, ai_evaluations, consensus_analysis, final_decision, execution_time)
            
            result = {
                'symbol': symbol,
                'original_analysis': analysis_data,
                'ai_evaluations': ai_evaluations,
                'consensus': consensus_analysis,
                'final_decision': final_decision,
                'execution_time': execution_time,
                'recommendation': final_decision.get('action', 'HOLD'),
                'confidence': final_decision.get('confidence', 0.0),
                'trade_worthy': final_decision.get('trade_worthy', False)
            }
            
            print(f"✅ Multi-AI evaluation completed for {symbol} in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            print(f"❌ Multi-AI evaluation failed for {symbol}: {e}")
            return None

    async def _get_all_ai_evaluations(self, analysis_data: Dict, symbol: str) -> Dict:
        """دریافت ارزیابی از همه AI های موجود"""
        tasks = []
        ai_names = []
        
        # ایجاد task برای هر AI موجود
        for ai_name, client in self.ai_clients.items():
            if client:
                if ai_name == 'gemini':
                    tasks.append(self._evaluate_with_gemini(analysis_data, symbol))
                elif ai_name == 'openai':
                    tasks.append(self._evaluate_with_openai(analysis_data, symbol))
                elif ai_name == 'claude':
                    tasks.append(self._evaluate_with_claude(analysis_data, symbol))
                ai_names.append(ai_name)
        
        if not tasks:
            return {}
        
        # اجرای همزمان با timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=30.0
            )
            
            evaluations = {}
            for i, result in enumerate(results):
                if not isinstance(result, Exception) and result:
                    evaluations[ai_names[i]] = result
                elif isinstance(result, Exception):
                    print(f"⚠️ {ai_names[i]} evaluation failed: {result}")
            
            return evaluations
            
        except asyncio.TimeoutError:
            print("⚠️ AI evaluation timeout")
            return {}

    async def _evaluate_with_gemini(self, analysis_data: Dict, symbol: str) -> Optional[Dict]:
        """ارزیابی با Gemini AI"""
        if 'gemini' not in self.ai_clients:
            return None
        
        try:
            prompt = self._create_evaluation_prompt(analysis_data, symbol, 'gemini')
            
            start_time = time.time()
            response = await self.ai_clients['gemini'].generate_content_async(prompt)
            response_time = time.time() - start_time
            
            self.ai_performance['gemini']['response_times'].append(response_time)
            
            evaluation = self._parse_ai_response(response.text, 'gemini')
            if evaluation:
                evaluation['response_time'] = response_time
            
            return evaluation
            
        except Exception as e:
            print(f"Gemini evaluation error: {e}")
            return None

    async def _evaluate_with_openai(self, analysis_data: Dict, symbol: str) -> Optional[Dict]:
        """ارزیابی با OpenAI"""
        if 'openai' not in self.ai_clients:
            return None
        
        try:
            prompt = self._create_evaluation_prompt(analysis_data, symbol, 'openai')
            
            start_time = time.time()
            response = await self.ai_clients['openai'].chat.completions.create(
                model=getattr(config, 'OPENAI_MODEL', 'gpt-4'),
                messages=[
                    {"role": "system", "content": self._get_system_prompt('openai')},
                    {"role": "user", "content": prompt}
                ],
                temperature=getattr(config, 'OPENAI_TEMPERATURE', 0.1),
                max_tokens=getattr(config, 'OPENAI_MAX_TOKENS', 1500)
            )
            response_time = time.time() - start_time
            
            self.ai_performance['openai']['response_times'].append(response_time)
            
            evaluation = self._parse_ai_response(response.choices[0].message.content, 'openai')
            if evaluation:
                evaluation['response_time'] = response_time
            
            return evaluation
            
        except Exception as e:
            print(f"OpenAI evaluation error: {e}")
            return None

    async def _evaluate_with_claude(self, analysis_data: Dict, symbol: str) -> Optional[Dict]:
        """ارزیابی با Claude AI"""
        if 'claude' not in self.ai_clients:
            return None
        
        try:
            prompt = self._create_evaluation_prompt(analysis_data, symbol, 'claude')
            
            start_time = time.time()
            response = await self.ai_clients['claude'].messages.create(
                model=getattr(config, 'CLAUDE_MODEL', 'claude-3-5-sonnet-20240620'),
                max_tokens=getattr(config, 'CLAUDE_MAX_TOKENS', 1500),
                messages=[{"role": "user", "content": prompt}]
            )
            response_time = time.time() - start_time
            
            self.ai_performance['claude']['response_times'].append(response_time)
            
            evaluation = self._parse_ai_response(response.content[0].text, 'claude')
            if evaluation:
                evaluation['response_time'] = response_time
            
            return evaluation
            
        except Exception as e:
            print(f"Claude evaluation error: {e}")
            return None

    def _get_system_prompt(self, ai_type: str) -> str:
        """سیستم prompt برای هر نوع AI"""
        base_prompt = """شما یک متخصص تحلیل فنی و معاملات ارزهای دیجیتال هستید.

تخصص‌های شما:
- تحلیل Price Action و الگوهای تکنیکال
- مدیریت ریسک و سایزینگ
- ارزیابی کیفیت سیگنال‌های معاملاتی
- تحلیل شرایط بازار

رویکرد شما:
- محافظه‌کارانه و مبتنی بر داده
- دقیق در محاسبات ریسک/ریوارد
- صادق در ارزیابی نقاط ضعف سیگنال‌ها
- متمرکز بر حفظ سرمایه"""

        ai_specializations = {
            'gemini': "\nتخصص خاص شما: تحلیل دقیق عددی و محاسبات احتمالاتی",
            'openai': "\nتخصص خاص شما: تحلیل جامع و ترکیب عوامل مختلف",
            'claude': "\nتخصص خاص شما: مدیریت ریسک و تصمیم‌گیری محتاطانه"
        }
        
        return base_prompt + ai_specializations.get(ai_type, "")

    def _create_evaluation_prompt(self, analysis_data: Dict, symbol: str, ai_type: str) -> str:
        """ایجاد prompt ارزیابی برای AI"""
        signal_direction = "خرید" if analysis_data['signal'] > 0 else "فروش"
        signal_type = analysis_data.get('signal_type', 'نامشخص')
        confidence = analysis_data.get('confidence', 0)
        current_price = analysis_data.get('current_price', 0)
        risk_reward = analysis_data.get('risk_reward_ratio', 0)
        
        prompt = f"""لطفاً سیگنال معاملاتی زیر را ارزیابی کنید:

=== اطلاعات سیگنال ===
نماد: {symbol}
قیمت فعلی: ${current_price:.4f}
جهت سیگنال: {signal_direction}
نوع سیگنال: {signal_type}
اطمینان سیستم: {confidence:.2f}
نسبت ریسک/ریوارد: {risk_reward:.2f}

=== دلایل سیستم ==="""

        # اضافه کردن دلایل
        for reason in analysis_data.get('reasoning', []):
            prompt += f"\n• {reason}"

        # اضافه کردن شرایط بازار
        market_context = analysis_data.get('market_context', {})
        prompt += f"""

=== شرایط بازار ===
ساختار: {market_context.get('structure', 'نامشخص')}
قدرت ترند: {market_context.get('trend_strength', 'نامشخص')}
وضعیت حجم: {market_context.get('volume_context', 'نامشخص')}
ADX: {market_context.get('adx', 'نامشخص')}

=== سطوح کلیدی ===
ورود پیشنهادی: ${analysis_data.get('entry_price', 0):.4f}
حد ضرر: ${analysis_data.get('stop_loss', 0):.4f}
اهداف سود: {analysis_data.get('take_profits', [])}

=== سوالات ارزیابی ===
1. کیفیت setup از نظر فنی چگونه است؟
2. آیا timing مناسب است؟
3. ریسک/ریوارد منطقی است؟
4. چه نگرانی‌هایی وجود دارد؟
5. توصیه نهایی شما چیست؟

پاسخ را در قالب JSON ارائه دهید:
{{
    "technical_quality": 1-10,
    "timing_score": 1-10,
    "risk_assessment": "LOW/MEDIUM/HIGH/VERY_HIGH",
    "recommendation": "STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL",
    "confidence": 0.0-1.0,
    "key_strengths": ["قوت1", "قوت2"],
    "main_concerns": ["نگرانی1", "نگرانی2"],
    "suggested_improvements": ["پیشنهاد1", "پیشنهاد2"],
    "overall_score": 1-10
}}"""

        return prompt

    def _parse_ai_response(self, response_text: str, ai_name: str) -> Optional[Dict]:
        """تجزیه پاسخ AI"""
        try:
            # یافتن JSON در پاسخ
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            
            if start == -1 or end == 0:
                print(f"⚠️ No JSON found in {ai_name} response")
                return None
            
            json_str = response_text[start:end]
            parsed = json.loads(json_str)
            
            # استانداردسازی فیلدها
            standardized = {
                'ai_name': ai_name,
                'technical_quality': parsed.get('technical_quality', 5),
                'timing_score': parsed.get('timing_score', 5),
                'risk_assessment': parsed.get('risk_assessment', 'MEDIUM'),
                'recommendation': parsed.get('recommendation', 'HOLD'),
                'confidence': float(parsed.get('confidence', 0.5)),
                'key_strengths': parsed.get('key_strengths', []),
                'main_concerns': parsed.get('main_concerns', []),
                'suggested_improvements': parsed.get('suggested_improvements', []),
                'overall_score': parsed.get('overall_score', 5),
                'raw_response': response_text[:500]  # محدود کردن طول
            }
            
            return standardized
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parse error for {ai_name}: {e}")
            return None
        except Exception as e:
            print(f"❌ Response parsing error for {ai_name}: {e}")
            return None

    def _analyze_consensus(self, ai_evaluations: Dict) -> Dict:
        """تحلیل اجماع بین AI ها"""
        if not ai_evaluations:
            return {
                'type': 'NO_DATA',
                'strength': 0.0,
                'agreement_level': 'NONE',
                'conflicting_views': []
            }
        
        recommendations = []
        scores = []
        confidences = []
        
        # جمع‌آوری داده‌ها
        for ai_name, evaluation in ai_evaluations.items():
            recommendations.append(evaluation['recommendation'])
            scores.append(evaluation['overall_score'])
            confidences.append(evaluation['confidence'])
        
        # تحلیل توافق
        unique_recommendations = list(set(recommendations))
        agreement_level = self._calculate_agreement_level(recommendations)
        
        # محاسبه قدرت consensus
        consensus_strength = self._calculate_consensus_strength(recommendations, scores, confidences)
        
        # یافتن نظرات متضاد
        conflicting_views = self._identify_conflicts(ai_evaluations)
        
        # تعیین consensus type
        consensus_type = self._determine_consensus_type(unique_recommendations, agreement_level)
        
        return {
            'type': consensus_type,
            'strength': consensus_strength,
            'agreement_level': agreement_level,
            'unique_recommendations': unique_recommendations,
            'average_score': np.mean(scores),
            'average_confidence': np.mean(confidences),
            'conflicting_views': conflicting_views,
            'ai_count': len(ai_evaluations)
        }

    def _calculate_agreement_level(self, recommendations: List[str]) -> str:
        """محاسبه سطح توافق"""
        unique_count = len(set(recommendations))
        total_count = len(recommendations)
        
        if unique_count == 1:
            return 'FULL_AGREEMENT'
        elif unique_count == 2 and total_count >= 3:
            return 'MAJORITY_AGREEMENT'
        elif unique_count <= total_count * 0.6:
            return 'PARTIAL_AGREEMENT'
        else:
            return 'HIGH_DISAGREEMENT'

    def _calculate_consensus_strength(self, recommendations: List[str], scores: List[float], confidences: List[float]) -> float:
        """محاسبه قدرت consensus"""
        if not recommendations:
            return 0.0
        
        # وزن‌دهی بر اساس اجماع
        most_common = max(set(recommendations), key=recommendations.count)
        agreement_ratio = recommendations.count(most_common) / len(recommendations)
        
        # وزن‌دهی بر اساس کیفیت
        avg_score = np.mean(scores) / 10.0  # normalize to 0-1
        avg_confidence = np.mean(confidences)
        
        # محاسبه نهایی
        consensus_strength = (agreement_ratio * 0.5) + (avg_score * 0.3) + (avg_confidence * 0.2)
        
        return min(consensus_strength, 1.0)

    def _identify_conflicts(self, ai_evaluations: Dict) -> List[Dict]:
        """شناسایی تضادها"""
        conflicts = []
        
        ai_names = list(ai_evaluations.keys())
        for i in range(len(ai_names)):
            for j in range(i + 1, len(ai_names)):
                ai1, ai2 = ai_names[i], ai_names[j]
                eval1, eval2 = ai_evaluations[ai1], ai_evaluations[ai2]
                
                # بررسی تضاد در recommendation
                rec1, rec2 = eval1['recommendation'], eval2['recommendation']
                
                if self._are_conflicting_recommendations(rec1, rec2):
                    conflicts.append({
                        'ai1': ai1,
                        'ai2': ai2,
                        'ai1_recommendation': rec1,
                        'ai2_recommendation': rec2,
                        'ai1_confidence': eval1['confidence'],
                        'ai2_confidence': eval2['confidence'],
                        'conflict_type': 'RECOMMENDATION_CONFLICT'
                    })
        
        return conflicts

    def _are_conflicting_recommendations(self, rec1: str, rec2: str) -> bool:
        """بررسی تضاد بین توصیه‌ها"""
        bullish = {'STRONG_BUY', 'BUY'}
        bearish = {'STRONG_SELL', 'SELL'}
        
        return (rec1 in bullish and rec2 in bearish) or (rec1 in bearish and rec2 in bullish)

    def _determine_consensus_type(self, unique_recommendations: List[str], agreement_level: str) -> str:
        """تعیین نوع consensus"""
        if agreement_level == 'FULL_AGREEMENT':
            return 'STRONG_CONSENSUS'
        elif agreement_level == 'MAJORITY_AGREEMENT':
            return 'MAJORITY_CONSENSUS'
        elif agreement_level == 'PARTIAL_AGREEMENT':
            return 'WEAK_CONSENSUS'
        else:
            return 'NO_CONSENSUS'

    def _make_final_decision(self, original_analysis: Dict, ai_evaluations: Dict, consensus: Dict) -> Dict:
        """تصمیم‌گیری نهایی"""
        try:
            # اگر هیچ ارزیابی وجود ندارد
            if not ai_evaluations:
                return {
                    'action': 'HOLD',
                    'confidence': 0.0,
                    'trade_worthy': False,
                    'reason': 'No AI evaluations available'
                }
            
            # محاسبه امتیاز وزن‌دار
            weighted_score = 0.0
            total_weight = 0.0
            
            recommendations_mapping = {
                'STRONG_BUY': 2,
                'BUY': 1,
                'HOLD': 0,
                'SELL': -1,
                'STRONG_SELL': -2
            }
            
            for ai_name, evaluation in ai_evaluations.items():
                weight = self.ai_weights.get(ai_name, 0.33)
                score = recommendations_mapping.get(evaluation['recommendation'], 0)
                confidence = evaluation['confidence']
                
                weighted_score += score * confidence * weight
                total_weight += weight
            
            if total_weight > 0:
                final_score = weighted_score / total_weight
            else:
                final_score = 0
            
            # تعیین action نهایی
            action = self._score_to_action(final_score)
            
            # محاسبه confidence نهایی
            final_confidence = self._calculate_final_confidence(ai_evaluations, consensus, original_analysis)
            
            # بررسی trade worthy
            trade_worthy = self._is_trade_worthy(action, final_confidence, consensus, original_analysis)
            
            return {
                'action': action,
                'confidence': final_confidence,
                'trade_worthy': trade_worthy,
                'weighted_score': final_score,
                'consensus_strength': consensus.get('strength', 0),
                'agreement_level': consensus.get('agreement_level', 'NONE'),
                'reason': self._generate_decision_reason(action, consensus, ai_evaluations),
                'risk_factors': self._identify_risk_factors(ai_evaluations),
                'supporting_factors': self._identify_supporting_factors(ai_evaluations)
            }
            
        except Exception as e:
            print(f"❌ Final decision error: {e}")
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'trade_worthy': False,
                'reason': f'Decision error: {str(e)}'
            }

    def _score_to_action(self, score: float) -> str:
        """تبدیل امتیاز به action"""
        if score >= 1.5:
            return 'STRONG_BUY'
        elif score >= 0.5:
            return 'BUY'
        elif score <= -1.5:
            return 'STRONG_SELL'
        elif score <= -0.5:
            return 'SELL'
        else:
            return 'HOLD'

    def _calculate_final_confidence(self, ai_evaluations: Dict, consensus: Dict, original_analysis: Dict) -> float:
        """محاسبه confidence نهایی"""
        try:
            # confidence میانگین از AI ها
            ai_confidences = [eval_data['confidence'] for eval_data in ai_evaluations.values()]
            avg_ai_confidence = np.mean(ai_confidences)
            
            # consensus strength
            consensus_strength = consensus.get('strength', 0)
            
            # confidence اصلی سیستم
            original_confidence = original_analysis.get('confidence', 0)
            
            # ترکیب با وزن‌های مختلف
            final_confidence = (
                avg_ai_confidence * 0.4 +
                consensus_strength * 0.35 +
                original_confidence * 0.25
            )
            
            return min(final_confidence, 1.0)
            
        except Exception:
            return 0.5

    def _is_trade_worthy(self, action: str, confidence: float, consensus: Dict, original_analysis: Dict) -> bool:
        """بررسی ارزش معامله"""
        # شرایط اصلی
        min_confidence = getattr(config, 'SIGNAL_FILTERS', {}).get('min_confidence', 0.7)
        
        if action == 'HOLD':
            return False
        
        if confidence < min_confidence:
            return False
        
        # بررسی consensus
        if consensus.get('agreement_level') in ['HIGH_DISAGREEMENT']:
            return False
        
        # بررسی ریسک/ریوارد
        min_risk_reward = getattr(config, 'SIGNAL_FILTERS', {}).get('min_risk_reward', 2.0)
        if original_analysis.get('risk_reward_ratio', 0) < min_risk_reward:
            return False
        
        return True

    def _generate_decision_reason(self, action: str, consensus: Dict, ai_evaluations: Dict) -> str:
        """تولید دلیل تصمیم"""
        agreement = consensus.get('agreement_level', 'NONE')
        ai_count = len(ai_evaluations)
        
        reason_parts = [
            f"Action: {action}",
            f"Based on {ai_count} AI evaluations",
            f"Agreement level: {agreement}",
            f"Consensus strength: {consensus.get('strength', 0):.2f}"
        ]
        
        return " | ".join(reason_parts)

    def _identify_risk_factors(self, ai_evaluations: Dict) -> List[str]:
        """شناسایی عوامل ریسک"""
        risk_factors = []
        
        for ai_name, evaluation in ai_evaluations.items():
            for concern in evaluation.get('main_concerns', []):
                if concern not in risk_factors:
                    risk_factors.append(f"{ai_name}: {concern}")
        
        return risk_factors

    def _identify_supporting_factors(self, ai_evaluations: Dict) -> List[str]:
        """شناسایی عوامل حمایت‌کننده"""
        supporting_factors = []
        
        for ai_name, evaluation in ai_evaluations.items():
            for strength in evaluation.get('key_strengths', []):
                if strength not in supporting_factors:
                    supporting_factors.append(f"{ai_name}: {strength}")
        
        return supporting_factors

    def _save_evaluation_result(self, symbol: str, analysis_data: Dict, ai_evaluations: Dict, 
                              consensus: Dict, final_decision: Dict, execution_time: float):
        """ذخیره نتیجه ارزیابی"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # تهیه داده‌ها برای ذخیره
            gemini_data = ai_evaluations.get('gemini', {})
            openai_data = ai_evaluations.get('openai', {})
            claude_data = ai_evaluations.get('claude', {})
            
            cursor.execute('''
                INSERT INTO multi_ai_evaluations (
                    symbol, signal_type, original_confidence,
                    gemini_score, gemini_recommendation,
                    openai_score, openai_recommendation,
                    claude_score, claude_recommendation,
                    consensus_score, final_recommendation,
                    execution_time, market_conditions
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                analysis_data.get('signal_type', ''),
                analysis_data.get('confidence', 0),
                gemini_data.get('overall_score', 0),
                gemini_data.get('recommendation', ''),
                openai_data.get('overall_score', 0),
                openai_data.get('recommendation', ''),
                claude_data.get('overall_score', 0),
                claude_data.get('recommendation', ''),
                consensus.get('strength', 0),
                final_decision.get('action', ''),
                execution_time,
                json.dumps(analysis_data.get('market_context', {}))
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"❌ Save evaluation error: {e}")

    def update_ai_performance(self, ai_name: str, prediction_score: float, actual_outcome: float):
        """به‌روزرسانی عملکرد AI"""
        try:
            if ai_name in self.ai_performance:
                # محاسبه accuracy
                accuracy = 1.0 - abs(prediction_score - actual_outcome)
                
                # به‌روزرسانی آمار
                self.ai_performance[ai_name]['total'] += 1
                if accuracy > 0.6:  # آستانه موفقیت
                    self.ai_performance[ai_name]['correct'] += 1
                
                # محاسبه accuracy کلی
                total = self.ai_performance[ai_name]['total']
                correct = self.ai_performance[ai_name]['correct']
                self.ai_performance[ai_name]['accuracy'] = correct / total if total > 0 else 0
                
                # تنظیم وزن‌ها بر اساس عملکرد
                self._adjust_ai_weights()
                
                print(f"📊 {ai_name} performance updated: {self.ai_performance[ai_name]['accuracy']:.2%}")
        
        except Exception as e:
            print(f"❌ Performance update error: {e}")

    def _adjust_ai_weights(self):
        """تنظیم وزن‌های AI بر اساس عملکرد"""
        try:
            total_accuracy = 0
            active_ais = 0
            
            # محاسبه accuracy کل
            for ai_name, stats in self.ai_performance.items():
                if stats['total'] > 5:  # حداقل 5 نمونه
                    total_accuracy += stats['accuracy']
                    active_ais += 1
            
            if active_ais == 0:
                return
            
            # تنظیم وزن‌ها
            for ai_name, stats in self.ai_performance.items():
                if stats['total'] > 5:
                    # وزن بر اساس نسبت accuracy به میانگین
                    avg_accuracy = total_accuracy / active_ais
                    if avg_accuracy > 0:
                        weight_ratio = stats['accuracy'] / avg_accuracy
                        self.ai_weights[ai_name] = min(max(weight_ratio * 0.33, 0.1), 0.6)
            
            # نرمال‌سازی وزن‌ها
            total_weight = sum(self.ai_weights.values())
            if total_weight > 0:
                for ai_name in self.ai_weights:
                    self.ai_weights[ai_name] /= total_weight
                    
        except Exception as e:
            print(f"❌ Weight adjustment error: {e}")

    def get_performance_report(self) -> Dict:
        """گزارش عملکرد AI ها"""
        return {
            'ai_performance': self.ai_performance.copy(),
            'current_weights': self.ai_weights.copy(),
            'available_ais': list(self.ai_clients.keys()),
            'last_updated': datetime.datetime.now().isoformat()
        }
