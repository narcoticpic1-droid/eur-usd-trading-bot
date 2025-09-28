import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any
import itertools
from scipy.optimize import minimize, differential_evolution
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import time
import json
from dataclasses import dataclass
from enum import Enum

class OptimizationMethod(Enum):
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    GENETIC_ALGORITHM = "genetic_algorithm"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    WALK_FORWARD = "walk_forward"

@dataclass
class ParameterRange:
    """تعریف محدوده پارامتر"""
    name: str
    min_value: float
    max_value: float
    step: Optional[float] = None
    values: Optional[List] = None  # برای discrete values
    
class OptimizationEngine:
    """
    موتور بهینه‌سازی پارامترهای استراتژی
    """
    
    def __init__(self, strategy_function: Callable, data: Dict[str, pd.DataFrame]):
        self.strategy_function = strategy_function
        self.data = data
        self.optimization_history = []
        self.best_results = {}
        
    def optimize_parameters(self,
                          parameter_ranges: List[ParameterRange],
                          method: OptimizationMethod = OptimizationMethod.GRID_SEARCH,
                          objective_function: str = 'sharpe_ratio',
                          max_iterations: int = 1000,
                          n_jobs: int = -1,
                          validation_split: float = 0.3,
                          **kwargs) -> Dict:
        """بهینه‌سازی پارامترهای استراتژی"""
        
        print(f"شروع بهینه‌سازی با روش {method.value}...")
        print(f"تعداد پارامترها: {len(parameter_ranges)}")
        print(f"حداکثر iterations: {max_iterations}")
        
        start_time = time.time()
        
        # تقسیم داده به train/validation
        train_data, validation_data = self._split_data(validation_split)
        
        # انتخاب روش بهینه‌سازی
        if method == OptimizationMethod.GRID_SEARCH:
            results = self._grid_search(parameter_ranges, train_data, objective_function, n_jobs)
        elif method == OptimizationMethod.RANDOM_SEARCH:
            results = self._random_search(parameter_ranges, max_iterations, train_data, objective_function, n_jobs)
        elif method == OptimizationMethod.GENETIC_ALGORITHM:
            results = self._genetic_algorithm(parameter_ranges, max_iterations, train_data, objective_function)
        elif method == OptimizationMethod.WALK_FORWARD:
            results = self._walk_forward_optimization(parameter_ranges, objective_function, **kwargs)
        else:
            raise ValueError(f"روش {method.value} پشتیبانی نمی‌شود")
        
        # اعتبارسنجی نتایج روی validation data
        validation_results = self._validate_results(results, validation_data, objective_function)
        
        optimization_time = time.time() - start_time
        
        # ذخیره نتایج
        final_results = {
            'method': method.value,
            'optimization_time': optimization_time,
            'parameter_ranges': [self._parameter_range_to_dict(pr) for pr in parameter_ranges],
            'objective_function': objective_function,
            'best_parameters': results['best_parameters'],
            'best_score': results['best_score'],
            'optimization_history': results['optimization_history'],
            'validation_results': validation_results,
            'overfitting_analysis': self._analyze_overfitting(results, validation_results),
            'robustness_analysis': self._analyze_robustness(results),
            'parameter_sensitivity': self._analyze_parameter_sensitivity(results)
        }
        
        self.best_results = final_results
        
        print(f"بهینه‌سازی کامل شد در {optimization_time:.2f} ثانیه")
        print(f"بهترین {objective_function}: {results['best_score']:.4f}")
        
        return final_results

    def _grid_search(self, parameter_ranges: List[ParameterRange], 
                    data: Dict, objective: str, n_jobs: int) -> Dict:
        """Grid Search بهینه‌سازی"""
        
        # تولید grid
        parameter_combinations = self._generate_parameter_grid(parameter_ranges)
        total_combinations = len(parameter_combinations)
        
        print(f"تعداد کل ترکیبات: {total_combinations}")
        
        if total_combinations > 10000:
            print("⚠️ تعداد ترکیبات زیاد است - در نظر گیرید از Random Search استفاده کنید")
        
        # تنظیم multiprocessing
        if n_jobs == -1:
            n_jobs = mp.cpu_count()
        
        # اجرای موازی
        results = []
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = []
            
            for params in parameter_combinations:
                future = executor.submit(self._evaluate_parameters, params, data, objective)
                futures.append((future, params))
            
            # جمع‌آوری نتایج
            for i, (future, params) in enumerate(futures):
                try:
                    score = future.result(timeout=300)  # 5 minute timeout
                    results.append({
                        'parameters': params,
                        'score': score,
                        'iteration': i
                    })
                    
                    if (i + 1) % 100 == 0:
                        print(f"پیشرفت: {i+1}/{total_combinations} ({(i+1)/total_combinations*100:.1f}%)")
                        
                except Exception as e:
                    print(f"خطا در ارزیابی پارامترها {params}: {e}")
                    results.append({
                        'parameters': params,
                        'score': float('-inf'),
                        'error': str(e),
                        'iteration': i
                    })
        
        # یافتن بهترین نتیجه
        valid_results = [r for r in results if r['score'] != float('-inf')]
        if not valid_results:
            raise ValueError("هیچ ترکیب پارامتری موفق نبود")
        
        best_result = max(valid_results, key=lambda x: x['score'])
        
        return {
            'best_parameters': best_result['parameters'],
            'best_score': best_result['score'],
            'optimization_history': results,
            'total_evaluations': len(results)
        }

    def _random_search(self, parameter_ranges: List[ParameterRange], 
                      max_iterations: int, data: Dict, objective: str, n_jobs: int) -> Dict:
        """Random Search بهینه‌سازی"""
        
        print(f"شروع Random Search با {max_iterations} iteration")
        
        # تولید نمونه‌های تصادفی
        random_combinations = self._generate_random_samples(parameter_ranges, max_iterations)
        
        # تنظیم multiprocessing
        if n_jobs == -1:
            n_jobs = mp.cpu_count()
        
        results = []
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = []
            
            for params in random_combinations:
                future = executor.submit(self._evaluate_parameters, params, data, objective)
                futures.append((future, params))
            
            # جمع‌آوری نتایج
            for i, (future, params) in enumerate(futures):
                try:
                    score = future.result(timeout=300)
                    results.append({
                        'parameters': params,
                        'score': score,
                        'iteration': i
                    })
                    
                    if (i + 1) % 50 == 0:
                        current_best = max(results, key=lambda x: x['score'])
                        print(f"Iteration {i+1}: بهترین {objective} = {current_best['score']:.4f}")
                        
                except Exception as e:
                    print(f"خطا در iteration {i}: {e}")
                    results.append({
                        'parameters': params,
                        'score': float('-inf'),
                        'error': str(e),
                        'iteration': i
                    })
        
        # یافتن بهترین نتیجه
        valid_results = [r for r in results if r['score'] != float('-inf')]
        best_result = max(valid_results, key=lambda x: x['score'])
        
        return {
            'best_parameters': best_result['parameters'],
            'best_score': best_result['score'],
            'optimization_history': results,
            'total_evaluations': len(results)
        }

    def _genetic_algorithm(self, parameter_ranges: List[ParameterRange],
                          max_generations: int, data: Dict, objective: str) -> Dict:
        """Genetic Algorithm بهینه‌سازی"""
        
        print(f"شروع Genetic Algorithm با {max_generations} نسل")
        
        # تعریف bounds برای scipy
        bounds = [(pr.min_value, pr.max_value) for pr in parameter_ranges]
        
        # تابع objective برای scipy (باید minimization باشد)
        def scipy_objective(params_array):
            params_dict = {}
            for i, pr in enumerate(parameter_ranges):
                params_dict[pr.name] = params_array[i]
            
            score = self._evaluate_parameters(params_dict, data, objective)
            return -score  # منفی چون scipy minimize می‌کند
        
        # اجرای Differential Evolution
        result = differential_evolution(
            scipy_objective,
            bounds,
            maxiter=max_generations,
            popsize=15,  # اندازه جمعیت
            mutation=(0.5, 1),
            recombination=0.7,
            seed=42,
            disp=True,
            callback=self._ga_callback
        )
        
        # تبدیل نتیجه به فرمت مناسب
        best_params = {}
        for i, pr in enumerate(parameter_ranges):
            best_params[pr.name] = result.x[i]
        
        return {
            'best_parameters': best_params,
            'best_score': -result.fun,  # برگرداندن علامت
            'optimization_history': self.optimization_history,
            'total_evaluations': result.nfev,
            'success': result.success,
            'message': result.message
        }

    def _walk_forward_optimization(self, parameter_ranges: List[ParameterRange],
                                 objective: str, window_size: int = 252,
                                 step_size: int = 63) -> Dict:
        """Walk-Forward بهینه‌سازی"""
        
        print("شروع Walk-Forward Optimization...")
        
        # تقسیم داده به windows
        windows = self._create_walk_forward_windows(window_size, step_size)
        
        all_results = []
        out_of_sample_results = []
        
        for i, (train_window, test_window) in enumerate(windows):
            print(f"پردازش window {i+1}/{len(windows)}")
            
            # بهینه‌سازی روی train window
            train_data = {symbol: df.iloc[train_window] for symbol, df in self.data.items()}
            
            # استفاده از Grid Search برای هر window
            window_result = self._grid_search(parameter_ranges, train_data, objective, n_jobs=4)
            
            # تست روی out-of-sample data
            test_data = {symbol: df.iloc[test_window] for symbol, df in self.data.items()}
            oos_score = self._evaluate_parameters(window_result['best_parameters'], test_data, objective)
            
            all_results.append({
                'window': i,
                'train_period': train_window,
                'test_period': test_window,
                'best_parameters': window_result['best_parameters'],
                'in_sample_score': window_result['best_score'],
                'out_of_sample_score': oos_score
            })
            
            out_of_sample_results.append(oos_score)
        
        # تحلیل نتایج
        avg_oos_score = np.mean(out_of_sample_results)
        std_oos_score = np.std(out_of_sample_results)
        
        # انتخاب بهترین پارامترها (میانگین از همه windows)
        best_parameters = self._average_parameters(all_results, parameter_ranges)
        
        return {
            'best_parameters': best_parameters,
            'best_score': avg_oos_score,
            'optimization_history': all_results,
            'out_of_sample_scores': out_of_sample_results,
            'oos_score_mean': avg_oos_score,
            'oos_score_std': std_oos_score,
            'total_windows': len(windows)
        }

    def _evaluate_parameters(self, parameters: Dict, data: Dict, objective: str) -> float:
        """ارزیابی یک ست پارامتر"""
        try:
            # اجرای استراتژی با پارامترهای داده شده
            results = self.strategy_function(data, parameters)
            
            # محاسبه objective function
            if objective == 'sharpe_ratio':
                returns = results.get('returns', [])
                if len(returns) < 2:
                    return 0
                return np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            
            elif objective == 'total_return':
                return results.get('total_return', 0)
            
            elif objective == 'profit_factor':
                trades = results.get('trades', [])
                if not trades:
                    return 0
                
                wins = [t['pnl'] for t in trades if t['pnl'] > 0]
                losses = [abs(t['pnl']) for t in trades if t['pnl'] < 0]
                
                if not wins or not losses:
                    return 0
                
                return sum(wins) / sum(losses)
            
            elif objective == 'calmar_ratio':
                total_return = results.get('total_return', 0)
                max_drawdown = abs(results.get('max_drawdown', 1))
                return total_return / max_drawdown if max_drawdown > 0 else 0
            
            elif objective == 'custom':
                # امکان تعریف objective function سفارشی
                return results.get('custom_score', 0)
            
            else:
                return results.get(objective, 0)
                
        except Exception as e:
            print(f"خطا در ارزیابی پارامترها {parameters}: {e}")
            return float('-inf')

    def _generate_parameter_grid(self, parameter_ranges: List[ParameterRange]) -> List[Dict]:
        """تولید grid کامل پارامترها"""
        
        param_lists = []
        param_names = []
        
        for pr in parameter_ranges:
            param_names.append(pr.name)
            
            if pr.values is not None:
                # Discrete values
                param_lists.append(pr.values)
            else:
                # Continuous range
                if pr.step:
                    values = np.arange(pr.min_value, pr.max_value + pr.step, pr.step)
                else:
                    # Default: 10 steps
                    values = np.linspace(pr.min_value, pr.max_value, 10)
                param_lists.append(values.tolist())
        
        # تولید تمام ترکیبات
        combinations = list(itertools.product(*param_lists))
        
        # تبدیل به list of dictionaries
        parameter_combinations = []
        for combo in combinations:
            param_dict = {}
            for i, param_name in enumerate(param_names):
                param_dict[param_name] = combo[i]
            parameter_combinations.append(param_dict)
        
        return parameter_combinations

    def _generate_random_samples(self, parameter_ranges: List[ParameterRange], 
                                n_samples: int) -> List[Dict]:
        """تولید نمونه‌های تصادفی"""
        
        samples = []
        
        for _ in range(n_samples):
            sample = {}
            
            for pr in parameter_ranges:
                if pr.values is not None:
                    # Discrete values
                    sample[pr.name] = np.random.choice(pr.values)
                else:
                    # Continuous range
                    sample[pr.name] = np.random.uniform(pr.min_value, pr.max_value)
            
            samples.append(sample)
        
        return samples

    def _split_data(self, validation_split: float) -> Tuple[Dict, Dict]:
        """تقسیم داده به train/validation"""
        
        train_data = {}
        validation_data = {}
        
        for symbol, df in self.data.items():
            split_index = int(len(df) * (1 - validation_split))
            train_data[symbol] = df.iloc[:split_index].copy()
            validation_data[symbol] = df.iloc[split_index:].copy()
        
        return train_data, validation_data

    def _validate_results(self, optimization_results: Dict, 
                         validation_data: Dict, objective: str) -> Dict:
        """اعتبارسنجی نتایج روی validation data"""
        
        best_params = optimization_results['best_parameters']
        validation_score = self._evaluate_parameters(best_params, validation_data, objective)
        
        in_sample_score = optimization_results['best_score']
        overfitting_ratio = validation_score / in_sample_score if in_sample_score != 0 else 0
        
        return {
            'validation_score': validation_score,
            'in_sample_score': in_sample_score,
            'overfitting_ratio': overfitting_ratio,
            'is_overfitted': overfitting_ratio < 0.8  # اگر کمتر از 80% باشد
        }

    def _analyze_overfitting(self, optimization_results: Dict, validation_results: Dict) -> Dict:
        """تحلیل overfitting"""
        
        overfitting_ratio = validation_results['overfitting_ratio']
        
        analysis = {
            'overfitting_detected': overfitting_ratio < 0.8,
            'overfitting_severity': 'HIGH' if overfitting_ratio < 0.6 else 'MEDIUM' if overfitting_ratio < 0.8 else 'LOW',
            'recommendations': []
        }
        
        if analysis['overfitting_detected']:
            analysis['recommendations'].extend([
                "پارامترهای ساده‌تری انتخاب کنید",
                "regularization اضافه کنید",
                "داده‌های بیشتری جمع‌آوری کنید",
                "cross-validation استفاده کنید"
            ])
        
        return analysis

    def _analyze_robustness(self, optimization_results: Dict) -> Dict:
        """تحلیل robustness نتایج"""
        
        history = optimization_results['optimization_history']
        if not history:
            return {}
        
        # تحلیل پراکندگی scores
        scores = [r['score'] for r in history if r['score'] != float('-inf')]
        
        if not scores:
            return {}
        
        best_score = max(scores)
        top_10_percent_threshold = np.percentile(scores, 90)
        
        # تعداد parameter sets که نتیجه خوب دادند
        good_results = [r for r in history if r['score'] >= top_10_percent_threshold]
        
        robustness = {
            'score_std': np.std(scores),
            'score_range': max(scores) - min(scores),
            'good_results_count': len(good_results),
            'robustness_ratio': len(good_results) / len(scores),
            'is_robust': len(good_results) / len(scores) > 0.1  # اگر بیش از 10% نتایج خوب باشند
        }
        
        return robustness

    def _analyze_parameter_sensitivity(self, optimization_results: Dict) -> Dict:
        """تحلیل حساسیت پارامترها"""
        
        history = optimization_results['optimization_history']
        if not history:
            return {}
        
        # استخراج تمام نام‌های پارامتر
        param_names = list(history[0]['parameters'].keys()) if history else []
        
        sensitivity_analysis = {}
        
        for param_name in param_names:
            param_values = [r['parameters'][param_name] for r in history if r['score'] != float('-inf')]
            scores = [r['score'] for r in history if r['score'] != float('-inf')]
            
            if len(param_values) > 5:  # حداقل 5 نمونه
                correlation = np.corrcoef(param_values, scores)[0, 1]
                
                sensitivity_analysis[param_name] = {
                    'correlation_with_score': correlation,
                    'sensitivity': 'HIGH' if abs(correlation) > 0.5 else 'MEDIUM' if abs(correlation) > 0.2 else 'LOW',
                    'optimal_range': self._find_optimal_parameter_range(param_name, history)
                }
        
        return sensitivity_analysis

    def generate_optimization_report(self) -> str:
        """تولید گزارش بهینه‌سازی"""
        
        if not self.best_results:
            return "هیچ نتیجه بهینه‌سازی موجود نیست"
        
        results = self.best_results
        
        report = f"""
# گزارش بهینه‌سازی پارامترها

## اطلاعات کلی
- **روش بهینه‌سازی:** {results['method']}
- **زمان بهینه‌سازی:** {results['optimization_time']:.2f} ثانیه
- **تابع هدف:** {results['objective_function']}
- **بهترین نتیجه:** {results['best_score']:.4f}

## بهترین پارامترها
"""
        
        for param_name, value in results['best_parameters'].items():
            report += f"- **{param_name}:** {value:.4f}\n"
        
        # اضافه کردن تحلیل validation
        if 'validation_results' in results:
            val_results = results['validation_results']
            report += f"""
## نتایج اعتبارسنجی
- **نتیجه validation:** {val_results['validation_score']:.4f}
- **نسبت overfitting:** {val_results['overfitting_ratio']:.3f}
- **overfitting تشخیص داده شده:** {'بله' if val_results['is_overfitted'] else 'خیر'}
"""
        
        # اضافه کردن تحلیل robustness
        if 'robustness_analysis' in results:
            rob_analysis = results['robustness_analysis']
            report += f"""
## تحلیل robustness
- **انحراف معیار scores:** {rob_analysis.get('score_std', 0):.4f}
- **نسبت robustness:** {rob_analysis.get('robustness_ratio', 0):.3f}
- **robust است:** {'بله' if rob_analysis.get('is_robust', False) else 'خیر'}
"""
        
        return report

    def export_results(self, filepath: str):
        """صادرات نتایج به فایل JSON"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.best_results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"نتایج بهینه‌سازی در {filepath} ذخیره شد")
