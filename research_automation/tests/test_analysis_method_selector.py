"""
分析方法选择器测试
"""

import unittest
import sys
import os
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.analysis_method_selector import (
    AnalysisMethodSelector,
    AnalysisContext,
    ResearchObjective,
    DataCharacteristic,
    AnalysisMethod,
    DataProfile
)

class TestAnalysisMethodSelector(unittest.TestCase):
    """分析方法选择器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.selector = AnalysisMethodSelector()
        
        # 准备测试数据
        self.numerical_data = [
            {"feature1": 1.0, "feature2": 2.0, "target": 0},
            {"feature1": 2.0, "feature2": 3.0, "target": 1},
            {"feature1": 3.0, "feature2": 4.0, "target": 0},
            {"feature1": 4.0, "feature2": 5.0, "target": 1},
            {"feature1": 5.0, "feature2": 6.0, "target": 0}
        ]
        
        self.categorical_data = [
            {"category": "A", "value": "high", "target": "positive"},
            {"category": "B", "value": "low", "target": "negative"},
            {"category": "A", "value": "medium", "target": "neutral"},
            {"category": "C", "value": "high", "target": "positive"},
            {"category": "B", "value": "low", "target": "negative"}
        ]
        
        self.text_data = [
            {"text": "This is a positive review", "sentiment": "positive"},
            {"text": "This is a negative review", "sentiment": "negative"},
            {"text": "This is a neutral review", "sentiment": "neutral"},
            {"text": "Great product, highly recommended", "sentiment": "positive"},
            {"text": "Poor quality, not recommended", "sentiment": "negative"}
        ]
    
    def test_method_library_initialization(self):
        """测试方法库初始化"""
        # 验证方法库不为空
        self.assertGreater(len(self.selector.method_library), 0)
        
        # 验证包含基本的分类方法
        self.assertIn("logistic_regression", self.selector.method_library)
        self.assertIn("random_forest", self.selector.method_library)
        self.assertIn("svm", self.selector.method_library)
        
        # 验证包含回归方法
        self.assertIn("linear_regression", self.selector.method_library)
        self.assertIn("ridge_regression", self.selector.method_library)
        
        # 验证包含聚类方法
        self.assertIn("kmeans", self.selector.method_library)
        self.assertIn("hierarchical_clustering", self.selector.method_library)
        
        # 验证包含统计方法
        self.assertIn("t_test", self.selector.method_library)
        self.assertIn("correlation_analysis", self.selector.method_library)
    
    def test_get_method_by_name(self):
        """测试根据名称获取方法"""
        method = self.selector.get_method_by_name("logistic_regression")
        self.assertIsNotNone(method)
        self.assertEqual(method.name, "logistic_regression")
        self.assertEqual(method.category, "classification")
        
        # 测试不存在的方法
        method = self.selector.get_method_by_name("nonexistent_method")
        self.assertIsNone(method)
    
    def test_get_methods_by_category(self):
        """测试根据类别获取方法"""
        classification_methods = self.selector.get_methods_by_category("classification")
        self.assertGreater(len(classification_methods), 0)
        
        for method in classification_methods:
            self.assertEqual(method.category, "classification")
        
        # 测试不存在的类别
        nonexistent_methods = self.selector.get_methods_by_category("nonexistent_category")
        self.assertEqual(len(nonexistent_methods), 0)
    
    def test_get_methods_by_objective(self):
        """测试根据研究目标获取方法"""
        classification_methods = self.selector.get_methods_by_objective(ResearchObjective.CLASSIFICATION)
        self.assertGreater(len(classification_methods), 0)
        
        for method in classification_methods:
            self.assertIn(ResearchObjective.CLASSIFICATION, method.suitable_objectives)
        
        regression_methods = self.selector.get_methods_by_objective(ResearchObjective.REGRESSION)
        self.assertGreater(len(regression_methods), 0)
        
        for method in regression_methods:
            self.assertIn(ResearchObjective.REGRESSION, method.suitable_objectives)
    
    def test_select_methods_for_classification(self):
        """测试分类任务的方法选择"""
        context = AnalysisContext(
            primary_objective=ResearchObjective.CLASSIFICATION,
            interpretability_requirement="high",
            available_libraries=["sklearn", "pandas", "numpy"]
        )
        
        result = self.selector.select_methods(self.numerical_data, context)
        
        # 验证选择结果
        self.assertIsNotNone(result.selected_method)
        self.assertIn(ResearchObjective.CLASSIFICATION, result.selected_method.suitable_objectives)
        self.assertGreater(len(result.single_method_recommendations), 0)
        self.assertIsNotNone(result.selection_rationale)
        
        # 验证推荐方法按评分排序
        scores = [rec.suitability_score for rec in result.single_method_recommendations]
        self.assertEqual(scores, sorted(scores, reverse=True))
    
    def test_select_methods_for_regression(self):
        """测试回归任务的方法选择"""
        context = AnalysisContext(
            primary_objective=ResearchObjective.REGRESSION,
            computational_constraint="low",
            available_libraries=["sklearn", "pandas", "numpy"]
        )
        
        result = self.selector.select_methods(self.numerical_data, context)
        
        # 验证选择结果
        self.assertIsNotNone(result.selected_method)
        self.assertIn(ResearchObjective.REGRESSION, result.selected_method.suitable_objectives)
        
        # 验证选择了计算成本较低的方法
        self.assertIn(result.selected_method.computational_cost, ["low", "medium"])
    
    def test_select_methods_for_clustering(self):
        """测试聚类任务的方法选择"""
        context = AnalysisContext(
            primary_objective=ResearchObjective.CLUSTERING,
            interpretability_requirement="medium",
            available_libraries=["sklearn", "pandas", "numpy"]
        )
        
        result = self.selector.select_methods(self.numerical_data, context)
        
        # 验证选择结果
        self.assertIsNotNone(result.selected_method)
        self.assertIn(ResearchObjective.CLUSTERING, result.selected_method.suitable_objectives)
    
    def test_select_methods_for_text_analysis(self):
        """测试文本分析任务的方法选择"""
        context = AnalysisContext(
            primary_objective=ResearchObjective.CLASSIFICATION,
            available_libraries=["sklearn", "nltk", "textblob"]
        )
        
        result = self.selector.select_methods(self.text_data, context)
        
        # 验证选择结果
        self.assertIsNotNone(result.selected_method)
        
        # 验证数据概况正确识别了文本特征
        self.assertIn(DataCharacteristic.TEXT, result.data_profile.data_characteristics)
    
    def test_select_methods_with_constraints(self):
        """测试带约束条件的方法选择"""
        context = AnalysisContext(
            primary_objective=ResearchObjective.CLASSIFICATION,
            time_constraint=0.5,  # 30分钟限制
            computational_constraint="low",
            interpretability_requirement="high",
            excluded_methods=["svm"],  # 排除SVM
            available_libraries=["sklearn"]
        )
        
        result = self.selector.select_methods(self.numerical_data, context)
        
        # 验证排除的方法没有被选择
        self.assertNotEqual(result.selected_method.name, "svm")
        
        # 验证选择了高可解释性的方法
        self.assertEqual(result.selected_method.interpretability, "high")
        
        # 验证选择了低计算成本的方法
        self.assertEqual(result.selected_method.computational_cost, "low")
    
    def test_method_combination_generation(self):
        """测试方法组合生成"""
        context = AnalysisContext(
            primary_objective=ResearchObjective.CLASSIFICATION,
            secondary_objectives=[ResearchObjective.CORRELATION_ANALYSIS],
            available_libraries=["sklearn", "scipy", "pandas"]
        )
        
        result = self.selector.select_methods(self.numerical_data, context)
        
        # 验证生成了方法组合
        if len(result.single_method_recommendations) >= 2:
            self.assertGreater(len(result.combination_recommendations), 0)
            
            # 验证组合的合理性
            for combination in result.combination_recommendations:
                self.assertGreater(len(combination.methods), 1)
                self.assertGreater(combination.synergy_score, 0)
                self.assertIn(combination.overall_complexity, ["low", "medium", "high"])
    
    def test_update_method_performance(self):
        """测试更新方法性能统计"""
        method_name = "logistic_regression"
        original_method = self.selector.get_method_by_name(method_name)
        original_usage_count = original_method.usage_count
        original_success_rate = original_method.success_rate
        
        # 更新成功的性能
        self.selector.update_method_performance(method_name, 0.85, 2.0, True)
        
        updated_method = self.selector.get_method_by_name(method_name)
        self.assertEqual(updated_method.usage_count, original_usage_count + 1)
        self.assertGreaterEqual(updated_method.success_rate, original_success_rate)
        
        # 更新失败的性能
        self.selector.update_method_performance(method_name, 0.0, 0.0, False)
        
        updated_method = self.selector.get_method_by_name(method_name)
        self.assertEqual(updated_method.usage_count, original_usage_count + 2)
    
    def test_add_custom_method(self):
        """测试添加自定义方法"""
        custom_method = AnalysisMethod(
            name="custom_classifier",
            category="classification",
            description="自定义分类器",
            suitable_data_types=[DataCharacteristic.NUMERICAL],
            suitable_objectives=[ResearchObjective.CLASSIFICATION],
            suitable_sample_sizes=(50, 5000),
            complexity="medium",
            interpretability="medium",
            computational_cost="medium",
            required_libraries=["custom_lib"],
            default_parameters={"param1": 1.0}
        )
        
        # 添加自定义方法
        self.selector.add_custom_method(custom_method)
        
        # 验证方法已添加
        retrieved_method = self.selector.get_method_by_name("custom_classifier")
        self.assertIsNotNone(retrieved_method)
        self.assertEqual(retrieved_method.name, "custom_classifier")
        self.assertEqual(retrieved_method.category, "classification")
    
    def test_remove_method(self):
        """测试移除方法"""
        method_name = "naive_bayes"
        
        # 确认方法存在
        self.assertIsNotNone(self.selector.get_method_by_name(method_name))
        
        # 移除方法
        success = self.selector.remove_method(method_name)
        self.assertTrue(success)
        
        # 确认方法已移除
        self.assertIsNone(self.selector.get_method_by_name(method_name))
        
        # 尝试移除不存在的方法
        success = self.selector.remove_method("nonexistent_method")
        self.assertFalse(success)
    
    def test_data_profiler(self):
        """测试数据概况分析器"""
        from core.analysis_method_selector import DataProfiler
        
        profiler = DataProfiler()
        
        # 测试数值数据分析
        profile = profiler.profile_data(self.numerical_data)
        self.assertEqual(profile.sample_size, 5)
        self.assertEqual(profile.feature_count, 3)
        self.assertGreater(profile.numerical_features, 0)
        self.assertIn(DataCharacteristic.NUMERICAL, profile.data_characteristics)
        
        # 测试文本数据分析
        profile = profiler.profile_data(self.text_data)
        self.assertEqual(profile.sample_size, 5)
        self.assertGreater(profile.categorical_features, 0)
    
    def test_compatibility_evaluator(self):
        """测试兼容性评估器"""
        from core.analysis_method_selector import CompatibilityEvaluator, DataProfiler
        
        evaluator = CompatibilityEvaluator()
        profiler = DataProfiler()
        
        # 创建测试数据和上下文
        data_profile = profiler.profile_data(self.numerical_data)
        context = AnalysisContext(
            primary_objective=ResearchObjective.CLASSIFICATION,
            available_libraries=["sklearn"]
        )
        
        # 评估逻辑回归方法
        method = self.selector.get_method_by_name("logistic_regression")
        recommendation = evaluator.evaluate_method(method, data_profile, context)
        
        # 验证评估结果
        self.assertGreater(recommendation.suitability_score, 0)
        self.assertGreater(recommendation.data_compatibility, 0)
        self.assertGreater(recommendation.objective_alignment, 0)
        self.assertGreater(recommendation.resource_feasibility, 0)
        self.assertGreaterEqual(recommendation.confidence, 0)
        self.assertIsInstance(recommendation.reasons, list)
        self.assertIsInstance(recommendation.warnings, list)
    
    def test_performance_predictor(self):
        """测试性能预测器"""
        from core.analysis_method_selector import PerformancePredictor, DataProfiler
        
        predictor = PerformancePredictor()
        profiler = DataProfiler()
        
        # 创建测试数据
        data_profile = profiler.profile_data(self.numerical_data)
        method = self.selector.get_method_by_name("logistic_regression")
        
        # 预测性能
        accuracy = predictor.predict_accuracy(method, data_profile)
        runtime = predictor.predict_runtime(method, data_profile)
        
        # 验证预测结果
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)
        self.assertGreater(runtime, 0)

if __name__ == '__main__':
    unittest.main()