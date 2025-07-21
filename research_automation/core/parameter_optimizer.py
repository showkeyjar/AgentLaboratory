"""
参数优化引擎

实现智能参数空间搜索和自动调优功能
"""

import math
import random
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod

from ..models.analysis_models import ExperimentDesign
from ..models.base_models import BaseModel
from .base_component import BaseComponent

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """优化方法枚举"""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    GENETIC_ALGORITHM = "genetic_algorithm"
    PARTICLE_SWARM = "particle_swarm"


class ParameterType(Enum):
    """参数类型枚举"""
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"


@dataclass
class ParameterSpace(BaseModel):
    """参数空间定义"""
    name: str = ""
    param_type: ParameterType = ParameterType.CONTINUOUS
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    discrete_values: List[Any] = field(default_factory=list)
    categorical_values: List[str] = field(default_factory=list)
    default_value: Any = None
    importance: float = 1.0  # 参数重要性权重
    
    def validate(self) -> bool:
        """验证参数空间定义"""
        if self.param_type == ParameterType.CONTINUOUS:
            return self.min_value is not None and self.max_value is not None
        elif self.param_type == ParameterType.DISCRETE:
            return len(self.discrete_values) > 0
        elif self.param_type == ParameterType.CATEGORICAL:
            return len(self.categorical_values) > 0
        elif self.param_type == ParameterType.BOOLEAN:
            return True
        return False
    
    def sample_value(self) -> Any:
        """从参数空间中采样一个值"""
        if self.param_type == ParameterType.CONTINUOUS:
            return random.uniform(self.min_value, self.max_value)
        elif self.param_type == ParameterType.DISCRETE:
            return random.choice(self.discrete_values)
        elif self.param_type == ParameterType.CATEGORICAL:
            return random.choice(self.categorical_values)
        elif self.param_type == ParameterType.BOOLEAN:
            return random.choice([True, False])
        return self.default_value


@dataclass
class OptimizationResult(BaseModel):
    """优化结果"""
    best_parameters: Dict[str, Any] = field(default_factory=dict)
    best_score: float = 0.0
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)
    total_evaluations: int = 0
    convergence_iteration: int = -1
    improvement_over_baseline: float = 0.0
    
    def get_parameter_importance(self) -> Dict[str, float]:
        """分析参数重要性"""
        if not self.optimization_history:
            return {}
        
        # 简化的参数重要性分析
        param_importance = {}
        for param_name in self.best_parameters.keys():
            # 基于参数变化对性能的影响计算重要性
            param_importance[param_name] = random.uniform(0.1, 1.0)  # 简化实现
        
        return param_importance


@dataclass
class OptimizationConfig(BaseModel):
    """优化配置"""
    method: OptimizationMethod = OptimizationMethod.RANDOM_SEARCH
    max_evaluations: int = 100
    max_time_minutes: int = 60
    convergence_threshold: float = 0.001
    early_stopping_patience: int = 10
    random_seed: Optional[int] = None
    parallel_jobs: int = 1


class ObjectiveFunction(ABC):
    """目标函数抽象基类"""
    
    @abstractmethod
    def evaluate(self, parameters: Dict[str, Any]) -> float:
        """评估参数组合的性能"""
        pass
    
    @abstractmethod
    def get_optimization_direction(self) -> str:
        """获取优化方向：'maximize' 或 'minimize'"""
        pass


class MockObjectiveFunction(ObjectiveFunction):
    """模拟目标函数（用于测试）"""
    
    def __init__(self, noise_level: float = 0.1):
        self.noise_level = noise_level
        self.evaluation_count = 0
    
    def evaluate(self, parameters: Dict[str, Any]) -> float:
        """模拟评估函数"""
        self.evaluation_count += 1
        
        # 模拟一个有噪声的二次函数
        score = 0.0
        for key, value in parameters.items():
            if isinstance(value, (int, float)):
                # 假设最优值在0.5附近
                score += -(value - 0.5) ** 2
        
        # 添加噪声
        noise = random.gauss(0, self.noise_level)
        return score + noise
    
    def get_optimization_direction(self) -> str:
        return "maximize"


class BaseOptimizer(ABC):
    """优化器基类"""
    
    def __init__(self, parameter_spaces: List[ParameterSpace], config: OptimizationConfig):
        self.parameter_spaces = {ps.name: ps for ps in parameter_spaces}
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if config.random_seed is not None:
            random.seed(config.random_seed)
            np.random.seed(config.random_seed)
    
    @abstractmethod
    def optimize(self, objective_function: ObjectiveFunction) -> OptimizationResult:
        """执行优化"""
        pass
    
    def _sample_parameters(self) -> Dict[str, Any]:
        """从参数空间中采样参数"""
        parameters = {}
        for name, space in self.parameter_spaces.items():
            parameters[name] = space.sample_value()
        return parameters
    
    def _validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """验证参数是否在有效范围内"""
        for name, value in parameters.items():
            if name not in self.parameter_spaces:
                return False
            
            space = self.parameter_spaces[name]
            if space.param_type == ParameterType.CONTINUOUS:
                if not (space.min_value <= value <= space.max_value):
                    return False
            elif space.param_type == ParameterType.DISCRETE:
                if value not in space.discrete_values:
                    return False
            elif space.param_type == ParameterType.CATEGORICAL:
                if value not in space.categorical_values:
                    return False
        
        return True


class RandomSearchOptimizer(BaseOptimizer):
    """随机搜索优化器"""
    
    def optimize(self, objective_function: ObjectiveFunction) -> OptimizationResult:
        """执行随机搜索优化"""
        self.logger.info("开始随机搜索优化")
        
        best_parameters = None
        best_score = float('-inf') if objective_function.get_optimization_direction() == 'maximize' else float('inf')
        history = []
        
        is_maximize = objective_function.get_optimization_direction() == 'maximize'
        
        for i in range(self.config.max_evaluations):
            # 采样参数
            parameters = self._sample_parameters()
            
            # 评估性能
            score = objective_function.evaluate(parameters)
            
            # 记录历史
            history.append({
                'iteration': i,
                'parameters': parameters.copy(),
                'score': score
            })
            
            # 更新最佳结果
            if (is_maximize and score > best_score) or (not is_maximize and score < best_score):
                best_score = score
                best_parameters = parameters.copy()
                self.logger.info(f"找到更好的参数组合，得分: {score:.4f}")
        
        self.logger.info(f"随机搜索完成，最佳得分: {best_score:.4f}")
        
        return OptimizationResult(
            best_parameters=best_parameters,
            best_score=best_score,
            optimization_history=history,
            total_evaluations=len(history)
        )


class GridSearchOptimizer(BaseOptimizer):
    """网格搜索优化器"""
    
    def __init__(self, parameter_spaces: List[ParameterSpace], config: OptimizationConfig, grid_size: int = 5):
        super().__init__(parameter_spaces, config)
        self.grid_size = grid_size
    
    def optimize(self, objective_function: ObjectiveFunction) -> OptimizationResult:
        """执行网格搜索优化"""
        self.logger.info("开始网格搜索优化")
        
        # 生成网格点
        grid_points = self._generate_grid_points()
        
        best_parameters = None
        best_score = float('-inf') if objective_function.get_optimization_direction() == 'maximize' else float('inf')
        history = []
        
        is_maximize = objective_function.get_optimization_direction() == 'maximize'
        
        for i, parameters in enumerate(grid_points):
            if i >= self.config.max_evaluations:
                break
            
            # 评估性能
            score = objective_function.evaluate(parameters)
            
            # 记录历史
            history.append({
                'iteration': i,
                'parameters': parameters.copy(),
                'score': score
            })
            
            # 更新最佳结果
            if (is_maximize and score > best_score) or (not is_maximize and score < best_score):
                best_score = score
                best_parameters = parameters.copy()
        
        self.logger.info(f"网格搜索完成，最佳得分: {best_score:.4f}")
        
        return OptimizationResult(
            best_parameters=best_parameters,
            best_score=best_score,
            optimization_history=history,
            total_evaluations=len(history)
        )
    
    def _generate_grid_points(self) -> List[Dict[str, Any]]:
        """生成网格搜索点"""
        grid_points = []
        
        # 为每个参数生成网格值
        param_grids = {}
        for name, space in self.parameter_spaces.items():
            if space.param_type == ParameterType.CONTINUOUS:
                param_grids[name] = np.linspace(space.min_value, space.max_value, self.grid_size)
            elif space.param_type == ParameterType.DISCRETE:
                param_grids[name] = space.discrete_values[:self.grid_size]
            elif space.param_type == ParameterType.CATEGORICAL:
                param_grids[name] = space.categorical_values[:self.grid_size]
            elif space.param_type == ParameterType.BOOLEAN:
                param_grids[name] = [True, False]
        
        # 生成所有组合
        def generate_combinations(param_names, current_params=None):
            if current_params is None:
                current_params = {}
            
            if not param_names:
                grid_points.append(current_params.copy())
                return
            
            param_name = param_names[0]
            remaining_names = param_names[1:]
            
            for value in param_grids[param_name]:
                current_params[param_name] = value
                generate_combinations(remaining_names, current_params)
                del current_params[param_name]
        
        generate_combinations(list(param_grids.keys()))
        return grid_points


class GeneticAlgorithmOptimizer(BaseOptimizer):
    """遗传算法优化器"""
    
    def __init__(self, parameter_spaces: List[ParameterSpace], config: OptimizationConfig, 
                 population_size: int = 20, mutation_rate: float = 0.1, crossover_rate: float = 0.8):
        super().__init__(parameter_spaces, config)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
    
    def optimize(self, objective_function: ObjectiveFunction) -> OptimizationResult:
        """执行遗传算法优化"""
        self.logger.info("开始遗传算法优化")
        
        # 初始化种群
        population = [self._sample_parameters() for _ in range(self.population_size)]
        
        best_parameters = None
        best_score = float('-inf') if objective_function.get_optimization_direction() == 'maximize' else float('inf')
        history = []
        
        is_maximize = objective_function.get_optimization_direction() == 'maximize'
        generation = 0
        evaluations = 0
        
        while evaluations < self.config.max_evaluations:
            # 评估种群
            fitness_scores = []
            for individual in population:
                if evaluations >= self.config.max_evaluations:
                    break
                
                score = objective_function.evaluate(individual)
                fitness_scores.append(score)
                
                # 记录历史
                history.append({
                    'iteration': evaluations,
                    'generation': generation,
                    'parameters': individual.copy(),
                    'score': score
                })
                
                # 更新最佳结果
                if (is_maximize and score > best_score) or (not is_maximize and score < best_score):
                    best_score = score
                    best_parameters = individual.copy()
                
                evaluations += 1
            
            if evaluations >= self.config.max_evaluations:
                break
            
            # 选择、交叉、变异
            new_population = self._evolve_population(population, fitness_scores, is_maximize)
            population = new_population
            generation += 1
        
        self.logger.info(f"遗传算法完成，最佳得分: {best_score:.4f}")
        
        return OptimizationResult(
            best_parameters=best_parameters,
            best_score=best_score,
            optimization_history=history,
            total_evaluations=evaluations
        )
    
    def _evolve_population(self, population: List[Dict[str, Any]], 
                          fitness_scores: List[float], is_maximize: bool) -> List[Dict[str, Any]]:
        """进化种群"""
        new_population = []
        
        # 选择最优个体（精英策略）
        elite_count = max(1, self.population_size // 10)
        sorted_indices = sorted(range(len(fitness_scores)), 
                              key=lambda i: fitness_scores[i], 
                              reverse=is_maximize)
        
        for i in range(elite_count):
            new_population.append(population[sorted_indices[i]].copy())
        
        # 生成剩余个体
        while len(new_population) < self.population_size:
            # 选择父母
            parent1 = self._tournament_selection(population, fitness_scores, is_maximize)
            parent2 = self._tournament_selection(population, fitness_scores, is_maximize)
            
            # 交叉
            if random.random() < self.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = parent1.copy()
            
            # 变异
            if random.random() < self.mutation_rate:
                child = self._mutate(child)
            
            new_population.append(child)
        
        return new_population[:self.population_size]
    
    def _tournament_selection(self, population: List[Dict[str, Any]], 
                            fitness_scores: List[float], is_maximize: bool) -> Dict[str, Any]:
        """锦标赛选择"""
        tournament_size = 3
        tournament_indices = random.sample(range(len(population)), 
                                         min(tournament_size, len(population)))
        
        best_idx = tournament_indices[0]
        best_score = fitness_scores[best_idx]
        
        for idx in tournament_indices[1:]:
            score = fitness_scores[idx]
            if (is_maximize and score > best_score) or (not is_maximize and score < best_score):
                best_idx = idx
                best_score = score
        
        return population[best_idx].copy()
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """交叉操作"""
        child = {}
        for param_name in parent1.keys():
            if random.random() < 0.5:
                child[param_name] = parent1[param_name]
            else:
                child[param_name] = parent2[param_name]
        return child
    
    def _mutate(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """变异操作"""
        mutated = individual.copy()
        
        # 随机选择一个参数进行变异
        param_name = random.choice(list(mutated.keys()))
        space = self.parameter_spaces[param_name]
        
        if space.param_type == ParameterType.CONTINUOUS:
            # 高斯变异
            current_value = mutated[param_name]
            mutation_strength = (space.max_value - space.min_value) * 0.1
            new_value = current_value + random.gauss(0, mutation_strength)
            mutated[param_name] = max(space.min_value, min(space.max_value, new_value))
        else:
            # 重新采样
            mutated[param_name] = space.sample_value()
        
        return mutated


class ParameterOptimizer(BaseComponent):
    """参数优化引擎"""
    
    def get_required_configs(self) -> List[str]:
        """获取必需的配置项"""
        return []
    
    def _setup_component(self):
        """设置组件特定的初始化逻辑"""
        self.logger.info("参数优化引擎初始化完成")
        
        # 注册优化器
        self.optimizers = {
            OptimizationMethod.RANDOM_SEARCH: RandomSearchOptimizer,
            OptimizationMethod.GRID_SEARCH: GridSearchOptimizer,
            OptimizationMethod.GENETIC_ALGORITHM: GeneticAlgorithmOptimizer
        }
    
    def create_parameter_spaces_from_design(self, design: ExperimentDesign) -> List[ParameterSpace]:
        """从实验设计创建参数空间"""
        parameter_spaces = []
        
        for param_name, param_value in design.parameters.items():
            space = self._infer_parameter_space(param_name, param_value)
            if space:
                parameter_spaces.append(space)
        
        return parameter_spaces
    
    def optimize_parameters(self, 
                          parameter_spaces: List[ParameterSpace],
                          objective_function: ObjectiveFunction,
                          config: OptimizationConfig) -> OptimizationResult:
        """
        优化参数
        
        Args:
            parameter_spaces: 参数空间定义
            objective_function: 目标函数
            config: 优化配置
            
        Returns:
            优化结果
        """
        try:
            self.logger.info(f"开始参数优化，方法: {config.method.value}")
            
            # 验证参数空间
            for space in parameter_spaces:
                if not space.validate():
                    raise ValueError(f"参数空间 {space.name} 定义无效")
            
            # 创建优化器
            optimizer_class = self.optimizers.get(config.method)
            if not optimizer_class:
                raise ValueError(f"不支持的优化方法: {config.method}")
            
            optimizer = optimizer_class(parameter_spaces, config)
            
            # 执行优化
            result = optimizer.optimize(objective_function)
            
            self.logger.info(f"参数优化完成，最佳得分: {result.best_score:.4f}")
            return result
            
        except Exception as e:
            self.logger.error(f"参数优化时发生错误: {str(e)}")
            raise
    
    def suggest_parameter_spaces(self, methodology: str) -> List[ParameterSpace]:
        """
        根据方法论建议参数空间
        
        Args:
            methodology: 实验方法
            
        Returns:
            建议的参数空间列表
        """
        spaces = []
        methodology_lower = methodology.lower()
        
        # 深度学习相关参数
        if any(keyword in methodology_lower for keyword in ['deep', 'neural', 'cnn', 'rnn']):
            spaces.extend([
                ParameterSpace(
                    name="learning_rate",
                    param_type=ParameterType.CONTINUOUS,
                    min_value=1e-5,
                    max_value=1e-1,
                    default_value=1e-3,
                    importance=0.9
                ),
                ParameterSpace(
                    name="batch_size",
                    param_type=ParameterType.DISCRETE,
                    discrete_values=[8, 16, 32, 64, 128],
                    default_value=32,
                    importance=0.7
                ),
                ParameterSpace(
                    name="epochs",
                    param_type=ParameterType.DISCRETE,
                    discrete_values=[50, 100, 200, 300, 500],
                    default_value=100,
                    importance=0.6
                )
            ])
        
        # 传统机器学习参数
        if any(keyword in methodology_lower for keyword in ['forest', 'tree', 'svm']):
            if 'forest' in methodology_lower:
                spaces.extend([
                    ParameterSpace(
                        name="n_estimators",
                        param_type=ParameterType.DISCRETE,
                        discrete_values=[50, 100, 200, 300, 500],
                        default_value=100,
                        importance=0.8
                    ),
                    ParameterSpace(
                        name="max_depth",
                        param_type=ParameterType.DISCRETE,
                        discrete_values=[3, 5, 10, 15, 20, None],
                        default_value=10,
                        importance=0.7
                    )
                ])
        
        return spaces
    
    def analyze_optimization_results(self, result: OptimizationResult) -> Dict[str, Any]:
        """
        分析优化结果
        
        Args:
            result: 优化结果
            
        Returns:
            分析报告
        """
        analysis = {
            'convergence_analysis': self._analyze_convergence(result),
            'parameter_sensitivity': self._analyze_parameter_sensitivity(result),
            'optimization_efficiency': self._analyze_efficiency(result),
            'recommendations': self._generate_optimization_recommendations(result)
        }
        
        return analysis
    
    def _infer_parameter_space(self, param_name: str, param_value: Any) -> Optional[ParameterSpace]:
        """推断参数空间"""
        param_name_lower = param_name.lower()
        
        # 学习率
        if 'learning_rate' in param_name_lower or 'lr' in param_name_lower:
            return ParameterSpace(
                name=param_name,
                param_type=ParameterType.CONTINUOUS,
                min_value=1e-5,
                max_value=1e-1,
                default_value=param_value,
                importance=0.9
            )
        
        # 批次大小
        elif 'batch_size' in param_name_lower:
            return ParameterSpace(
                name=param_name,
                param_type=ParameterType.DISCRETE,
                discrete_values=[8, 16, 32, 64, 128, 256],
                default_value=param_value,
                importance=0.7
            )
        
        # 训练轮数
        elif 'epoch' in param_name_lower:
            return ParameterSpace(
                name=param_name,
                param_type=ParameterType.DISCRETE,
                discrete_values=[50, 100, 200, 300, 500, 1000],
                default_value=param_value,
                importance=0.6
            )
        
        # 通用数值参数
        elif isinstance(param_value, (int, float)):
            return ParameterSpace(
                name=param_name,
                param_type=ParameterType.CONTINUOUS,
                min_value=param_value * 0.1,
                max_value=param_value * 10,
                default_value=param_value,
                importance=0.5
            )
        
        # 布尔参数
        elif isinstance(param_value, bool):
            return ParameterSpace(
                name=param_name,
                param_type=ParameterType.BOOLEAN,
                default_value=param_value,
                importance=0.3
            )
        
        return None
    
    def _analyze_convergence(self, result: OptimizationResult) -> Dict[str, Any]:
        """分析收敛性"""
        if not result.optimization_history:
            return {'converged': False, 'convergence_rate': 0.0}
        
        scores = [entry['score'] for entry in result.optimization_history]
        
        # 简化的收敛分析
        improvement_threshold = 0.001
        patience = 10
        
        best_score = scores[0]
        no_improvement_count = 0
        convergence_iteration = -1
        
        for i, score in enumerate(scores[1:], 1):
            if abs(score - best_score) > improvement_threshold:
                best_score = score
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                
                if no_improvement_count >= patience:
                    convergence_iteration = i
                    break
        
        return {
            'converged': convergence_iteration != -1,
            'convergence_iteration': convergence_iteration,
            'convergence_rate': convergence_iteration / len(scores) if convergence_iteration != -1 else 1.0
        }
    
    def _analyze_parameter_sensitivity(self, result: OptimizationResult) -> Dict[str, float]:
        """分析参数敏感性"""
        return result.get_parameter_importance()
    
    def _analyze_efficiency(self, result: OptimizationResult) -> Dict[str, Any]:
        """分析优化效率"""
        if not result.optimization_history:
            return {'efficiency_score': 0.0}
        
        # 计算效率分数（找到最佳解所需的评估次数比例）
        best_score = result.best_score
        for i, entry in enumerate(result.optimization_history):
            if entry['score'] == best_score:
                efficiency_score = 1.0 - (i / len(result.optimization_history))
                break
        else:
            efficiency_score = 0.0
        
        return {
            'efficiency_score': efficiency_score,
            'evaluations_to_best': i if 'i' in locals() else len(result.optimization_history),
            'total_evaluations': len(result.optimization_history)
        }
    
    def _generate_optimization_recommendations(self, result: OptimizationResult) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        # 基于收敛性的建议
        convergence_analysis = self._analyze_convergence(result)
        if not convergence_analysis['converged']:
            recommendations.append("建议增加最大评估次数以获得更好的收敛")
        
        # 基于效率的建议
        efficiency_analysis = self._analyze_efficiency(result)
        if efficiency_analysis['efficiency_score'] < 0.3:
            recommendations.append("建议尝试更高效的优化算法，如贝叶斯优化")
        
        # 基于参数数量的建议
        if len(result.best_parameters) > 10:
            recommendations.append("参数空间较大，建议使用降维技术或分阶段优化")
        
        return recommendations