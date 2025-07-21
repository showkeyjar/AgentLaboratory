"""
参数优化引擎测试

测试参数优化器的各项功能
"""

import unittest
import sys
import os
import math
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_automation.core.parameter_optimizer import (
    ParameterOptimizer, ParameterSpace, OptimizationResult, OptimizationConfig,
    OptimizationMethod, ParameterType, MockObjectiveFunction,
    RandomSearchOptimizer, GridSearchOptimizer, GeneticAlgorithmOptimizer
)
from research_automation.models.analysis_models import ExperimentDesign


class TestParameterOptimizer(unittest.TestCase):
    """参数优化器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.optimizer = ParameterOptimizer()
        
        # 创建测试用的参数空间
        self.test_spaces = [
            ParameterSpace(
                name="learning_rate",
                param_type=ParameterType.CONTINUOUS,
                min_value=0.001,
                max_value=0.1,
                default_value=0.01,
                importance=0.9
            ),
            ParameterSpace(
                name="batch_size",
                param_type=ParameterType.DISCRETE,
                discrete_values=[16, 32, 64, 128],
                default_value=32,
                importance=0.7
            ),
            ParameterSpace(
                name="optimizer_type",
                param_type=ParameterType.CATEGORICAL,
                categorical_values=["adam", "sgd", "rmsprop"],
                default_value="adam",
                importance=0.6
            ),
            ParameterSpace(
                name="use_dropout",
                param_type=ParameterType.BOOLEAN,
                default_value=True,
                importance=0.4
            )
        ]
        
        # 创建测试用的优化配置
        self.test_config = OptimizationConfig(
            method=OptimizationMethod.RANDOM_SEARCH,
            max_evaluations=50,
            max_time_minutes=5,
            random_seed=42
        )
        
        # 创建测试用的目标函数
        self.test_objective = MockObjectiveFunction(noise_level=0.05)
    
    def test_parameter_space_validation(self):
        """测试参数空间验证"""
        # 有效的连续参数空间
        valid_continuous = ParameterSpace(
            name="test_param",
            param_type=ParameterType.CONTINUOUS,
            min_value=0.0,
            max_value=1.0
        )
        self.assertTrue(valid_continuous.validate())
        
        # 无效的连续参数空间（缺少边界）
        invalid_continuous = ParameterSpace(
            name="test_param",
            param_type=ParameterType.CONTINUOUS
        )
        self.assertFalse(invalid_continuous.validate())
        
        # 有效的离散参数空间
        valid_discrete = ParameterSpace(
            name="test_param",
            param_type=ParameterType.DISCRETE,
            discrete_values=[1, 2, 3, 4, 5]
        )
        self.assertTrue(valid_discrete.validate())
        
        print("参数空间验证测试通过")
    
    def test_parameter_space_sampling(self):
        """测试参数空间采样"""
        # 测试连续参数采样
        continuous_space = self.test_spaces[0]  # learning_rate
        for _ in range(10):
            value = continuous_space.sample_value()
            self.assertGreaterEqual(value, continuous_space.min_value)
            self.assertLessEqual(value, continuous_space.max_value)
        
        # 测试离散参数采样
        discrete_space = self.test_spaces[1]  # batch_size
        for _ in range(10):
            value = discrete_space.sample_value()
            self.assertIn(value, discrete_space.discrete_values)
        
        # 测试分类参数采样
        categorical_space = self.test_spaces[2]  # optimizer_type
        for _ in range(10):
            value = categorical_space.sample_value()
            self.assertIn(value, categorical_space.categorical_values)
        
        # 测试布尔参数采样
        boolean_space = self.test_spaces[3]  # use_dropout
        values = set()
        for _ in range(20):
            value = boolean_space.sample_value()
            self.assertIn(value, [True, False])
            values.add(value)
        
        # 应该能采样到两种值
        self.assertEqual(len(values), 2)
        
        print("参数空间采样测试通过")
    
    def test_random_search_optimizer(self):
        """测试随机搜索优化器"""
        optimizer = RandomSearchOptimizer(self.test_spaces, self.test_config)
        result = optimizer.optimize(self.test_objective)
        
        # 验证结果结构
        self.assertIsInstance(result, OptimizationResult)
        self.assertIsInstance(result.best_parameters, dict)
        self.assertIsInstance(result.best_score, float)
        self.assertEqual(len(result.optimization_history), self.test_config.max_evaluations)
        
        # 验证最佳参数在有效范围内
        self.assertIn("learning_rate", result.best_parameters)
        self.assertGreaterEqual(result.best_parameters["learning_rate"], 0.001)
        self.assertLessEqual(result.best_parameters["learning_rate"], 0.1)
        
        print(f"随机搜索优化结果:")
        print(f"- 最佳得分: {result.best_score:.4f}")
        print(f"- 最佳参数: {result.best_parameters}")
        print(f"- 总评估次数: {result.total_evaluations}")
    
    def test_grid_search_optimizer(self):
        """测试网格搜索优化器"""
        # 使用较小的参数空间进行网格搜索
        small_spaces = [
            ParameterSpace(
                name="param1",
                param_type=ParameterType.CONTINUOUS,
                min_value=0.0,
                max_value=1.0
            ),
            ParameterSpace(
                name="param2",
                param_type=ParameterType.DISCRETE,
                discrete_values=[1, 2, 3]
            )
        ]
        
        config = OptimizationConfig(
            method=OptimizationMethod.GRID_SEARCH,
            max_evaluations=20,
            random_seed=42
        )
        
        optimizer = GridSearchOptimizer(small_spaces, config, grid_size=3)
        result = optimizer.optimize(self.test_objective)
        
        # 验证结果
        self.assertIsInstance(result, OptimizationResult)
        self.assertGreater(result.total_evaluations, 0)
        self.assertLessEqual(result.total_evaluations, config.max_evaluations)
        
        print(f"网格搜索优化结果:")
        print(f"- 最佳得分: {result.best_score:.4f}")
        print(f"- 最佳参数: {result.best_parameters}")
        print(f"- 总评估次数: {result.total_evaluations}")
    
    def test_genetic_algorithm_optimizer(self):
        """测试遗传算法优化器"""
        config = OptimizationConfig(
            method=OptimizationMethod.GENETIC_ALGORITHM,
            max_evaluations=100,
            random_seed=42
        )
        
        optimizer = GeneticAlgorithmOptimizer(
            self.test_spaces, config, 
            population_size=10, mutation_rate=0.2, crossover_rate=0.8
        )
        result = optimizer.optimize(self.test_objective)
        
        # 验证结果
        self.assertIsInstance(result, OptimizationResult)
        self.assertEqual(result.total_evaluations, config.max_evaluations)
        
        # 验证历史记录包含代数信息
        self.assertTrue(any('generation' in entry for entry in result.optimization_history))
        
        print(f"遗传算法优化结果:")
        print(f"- 最佳得分: {result.best_score:.4f}")
        print(f"- 最佳参数: {result.best_parameters}")
        print(f"- 总评估次数: {result.total_evaluations}")
    
    def test_parameter_optimizer_integration(self):
        """测试参数优化器集成功能"""
        result = self.optimizer.optimize_parameters(
            self.test_spaces, self.test_objective, self.test_config
        )
        
        # 验证结果
        self.assertIsInstance(result, OptimizationResult)
        self.assertGreater(result.best_score, float('-inf'))
        
        print(f"集成优化结果:")
        print(f"- 最佳得分: {result.best_score:.4f}")
        print(f"- 参数数量: {len(result.best_parameters)}")
    
    def test_create_parameter_spaces_from_design(self):
        """测试从实验设计创建参数空间"""
        design = ExperimentDesign(
            methodology="deep learning",
            parameters={
                'learning_rate': 0.01,
                'batch_size': 32,
                'epochs': 100,
                'use_dropout': True,
                'custom_param': 'some_value'
            }
        )
        
        spaces = self.optimizer.create_parameter_spaces_from_design(design)
        
        # 验证创建的参数空间
        self.assertGreater(len(spaces), 0)
        
        space_names = [space.name for space in spaces]
        self.assertIn('learning_rate', space_names)
        self.assertIn('batch_size', space_names)
        self.assertIn('epochs', space_names)
        self.assertIn('use_dropout', space_names)
        
        print(f"从实验设计创建的参数空间:")
        for space in spaces:
            print(f"- {space.name}: {space.param_type.value}")
    
    def test_suggest_parameter_spaces(self):
        """测试参数空间建议"""
        # 测试深度学习方法
        dl_spaces = self.optimizer.suggest_parameter_spaces("deep learning with CNN")
        self.assertGreater(len(dl_spaces), 0)
        
        space_names = [space.name for space in dl_spaces]
        self.assertIn('learning_rate', space_names)
        self.assertIn('batch_size', space_names)
        
        # 测试随机森林方法
        rf_spaces = self.optimizer.suggest_parameter_spaces("random forest classification")
        self.assertGreater(len(rf_spaces), 0)
        
        rf_space_names = [space.name for space in rf_spaces]
        self.assertIn('n_estimators', rf_space_names)
        
        print(f"深度学习建议参数: {[s.name for s in dl_spaces]}")
        print(f"随机森林建议参数: {[s.name for s in rf_spaces]}")
    
    def test_optimization_result_analysis(self):
        """测试优化结果分析"""
        # 运行优化
        result = self.optimizer.optimize_parameters(
            self.test_spaces, self.test_objective, self.test_config
        )
        
        # 分析结果
        analysis = self.optimizer.analyze_optimization_results(result)
        
        # 验证分析结果结构
        self.assertIn('convergence_analysis', analysis)
        self.assertIn('parameter_sensitivity', analysis)
        self.assertIn('optimization_efficiency', analysis)
        self.assertIn('recommendations', analysis)
        
        # 验证收敛分析
        convergence = analysis['convergence_analysis']
        self.assertIn('converged', convergence)
        self.assertIsInstance(convergence['converged'], bool)
        
        # 验证效率分析
        efficiency = analysis['optimization_efficiency']
        self.assertIn('efficiency_score', efficiency)
        self.assertGreaterEqual(efficiency['efficiency_score'], 0.0)
        self.assertLessEqual(efficiency['efficiency_score'], 1.0)
        
        print(f"优化结果分析:")
        print(f"- 是否收敛: {convergence['converged']}")
        print(f"- 效率分数: {efficiency['efficiency_score']:.3f}")
        print(f"- 建议数量: {len(analysis['recommendations'])}")
    
    def test_mock_objective_function(self):
        """测试模拟目标函数"""
        objective = MockObjectiveFunction(noise_level=0.1)
        
        # 测试评估
        params1 = {'param1': 0.5, 'param2': 0.5}
        params2 = {'param1': 0.0, 'param2': 1.0}
        
        score1 = objective.evaluate(params1)
        score2 = objective.evaluate(params2)
        
        # 验证评估结果
        self.assertIsInstance(score1, float)
        self.assertIsInstance(score2, float)
        
        # 验证优化方向
        self.assertEqual(objective.get_optimization_direction(), "maximize")
        
        # 验证评估计数
        self.assertEqual(objective.evaluation_count, 2)
        
        print(f"目标函数测试:")
        print(f"- 参数 {params1} 得分: {score1:.4f}")
        print(f"- 参数 {params2} 得分: {score2:.4f}")
    
    def test_optimization_methods_comparison(self):
        """测试不同优化方法的比较"""
        methods = [
            OptimizationMethod.RANDOM_SEARCH,
            OptimizationMethod.GRID_SEARCH,
            OptimizationMethod.GENETIC_ALGORITHM
        ]
        
        results = {}
        
        # 使用简化的参数空间
        simple_spaces = [
            ParameterSpace(
                name="x",
                param_type=ParameterType.CONTINUOUS,
                min_value=0.0,
                max_value=1.0
            ),
            ParameterSpace(
                name="y",
                param_type=ParameterType.CONTINUOUS,
                min_value=0.0,
                max_value=1.0
            )
        ]
        
        for method in methods:
            config = OptimizationConfig(
                method=method,
                max_evaluations=50,
                random_seed=42
            )
            
            try:
                result = self.optimizer.optimize_parameters(
                    simple_spaces, self.test_objective, config
                )
                results[method.value] = result.best_score
            except Exception as e:
                print(f"方法 {method.value} 执行失败: {e}")
                results[method.value] = None
        
        print(f"\\n不同优化方法比较:")
        for method, score in results.items():
            if score is not None:
                print(f"- {method}: {score:.4f}")
            else:
                print(f"- {method}: 执行失败")
    
    def test_parameter_importance_analysis(self):
        """测试参数重要性分析"""
        result = self.optimizer.optimize_parameters(
            self.test_spaces, self.test_objective, self.test_config
        )
        
        importance = result.get_parameter_importance()
        
        # 验证重要性分析结果
        self.assertIsInstance(importance, dict)
        self.assertEqual(len(importance), len(result.best_parameters))
        
        for param_name, imp_score in importance.items():
            self.assertIn(param_name, result.best_parameters)
            self.assertGreaterEqual(imp_score, 0.0)
            self.assertLessEqual(imp_score, 1.0)
        
        print(f"\\n参数重要性分析:")
        for param, imp in importance.items():
            print(f"- {param}: {imp:.3f}")


if __name__ == '__main__':
    unittest.main(verbosity=2)