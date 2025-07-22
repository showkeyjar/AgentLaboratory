"""
分析方法选择器测试

测试分析方法选择器的核心功能
"""

import unittest
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_automation.core.analysis_method_selector import (
    AnalysisMethodSelector,
    AnalysisMethod,
    DataContext,
    ResearchContext,
    MethodRecommendation,
    MethodSelectionResult,
    AnalysisType,
    DataCharacteristic,
    ResearchObjective
)


class TestAnalysisMethodSelector(unittest.TestCase):
    """分析方法选择器测试类"""
    
    def setUp(self):
        """设置测试环境"""
        self.config = {}
        self.selector = AnalysisMethodSelector(self.config)
        
        # 创建测试数据上下文
        self.data_context = DataContext(
            sample_size=1000,
            feature_count=10,
            data_characteristics={DataCharacteristic.NUMERICAL, DataCharacteristic.LARGE_SAMPLE},
            missing_rate=0.05,
            noise_level=0.1,
            data_quality_score=0.8,
            numerical_features=8,
            categorical_features=2,
            is_balanced=True,
            has_outliers=False
        )
        
        # 创建测试研究上下文
        self.research_context = ResearchContext(
            primary_objective=ResearchObjective.CLASSIFICATION,
            secondary_objectives={ResearchObjective.CORRELATION},
            analysis_type=AnalysisType.PREDICTIVE,
            interpretability_requirement=0.7,
            accuracy_requirement=0.8,
            preferred_methods=[],
            excluded_methods=[]
        )
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsInstance(self.selector, AnalysisMethodSelector)
        self.assertGreater(len(self.selector.method_library), 0)
        self.assertEqual(len(self.selector.selection_history), 0)
    
    def test_method_library_initialization(self):
        """测试方法库初始化"""
        method_library = self.selector.get_method_library()
        
        # 检查是否包含基本方法
        expected_methods = ["描述性统计", "t检验", "线性回归", "逻辑回归", "随机森林", "K-means聚类", "主成分分析", "相关性分析"]
        
        for method_name in expected_methods:
            self.assertIn(method_name, method_library)
            method = method_library[method_name]
            self.assertIsInstance(method, AnalysisMethod)
            self.assertEqual(method.name, method_name)
            self.assertGreater(len(method.description), 0)
    
    def test_select_methods_basic(self):
        """测试基本方法选择"""
        result = self.selector.select_methods(self.data_context, self.research_context)
        
        # 检查结果结构
        self.assertIsInstance(result, MethodSelectionResult)
        self.assertTrue(result.success)
        self.assertEqual(result.error_message, "")
        
        # 检查推荐结果
        self.assertGreater(len(result.primary_recommendations), 0)
        self.assertLessEqual(len(result.primary_recommendations), 3)
        
        # 检查推荐质量
        for recommendation in result.primary_recommendations:
            self.assertIsInstance(recommendation, MethodRecommendation)
            self.assertGreater(recommendation.suitability_score, 0.1)
            self.assertGreaterEqual(recommendation.confidence, 0.0)
            self.assertLessEqual(recommendation.confidence, 1.0)
    
    def test_select_methods_for_regression(self):
        """测试回归任务的方法选择"""
        # 修改研究上下文为回归任务
        regression_context = ResearchContext(
            primary_objective=ResearchObjective.REGRESSION,
            analysis_type=AnalysisType.PREDICTIVE,
            interpretability_requirement=0.8,
            accuracy_requirement=0.7
        )
        
        result = self.selector.select_methods(self.data_context, regression_context)
        
        self.assertTrue(result.success)
        self.assertGreater(len(result.primary_recommendations), 0)
        
        # 检查是否推荐了适合回归的方法
        recommended_methods = [rec.method.name for rec in result.primary_recommendations]
        self.assertTrue(any("回归" in method for method in recommended_methods))
    
    def test_select_methods_for_clustering(self):
        """测试聚类任务的方法选择"""
        clustering_context = ResearchContext(
            primary_objective=ResearchObjective.CLUSTERING,
            analysis_type=AnalysisType.EXPLORATORY,
            interpretability_requirement=0.6,
            accuracy_requirement=0.6
        )
        
        result = self.selector.select_methods(self.data_context, clustering_context)
        
        self.assertTrue(result.success)
        self.assertGreater(len(result.primary_recommendations), 0)
        
        # 检查是否推荐了聚类方法
        recommended_methods = [rec.method.name for rec in result.primary_recommendations]
        self.assertTrue(any("聚类" in method for method in recommended_methods))
    
    def test_small_sample_selection(self):
        """测试小样本数据的方法选择"""
        small_data_context = DataContext(
            sample_size=20,
            feature_count=5,
            data_characteristics={DataCharacteristic.NUMERICAL, DataCharacteristic.SMALL_SAMPLE},
            data_quality_score=0.9
        )
        
        result = self.selector.select_methods(small_data_context, self.research_context)
        
        self.assertTrue(result.success)
        
        # 检查推荐的方法是否适合小样本
        for recommendation in result.primary_recommendations:
            method = recommendation.method
            self.assertLessEqual(method.min_sample_size, 20)
    
    def test_high_interpretability_requirement(self):
        """测试高可解释性要求的方法选择"""
        interpretable_context = ResearchContext(
            primary_objective=ResearchObjective.CLASSIFICATION,
            analysis_type=AnalysisType.PREDICTIVE,
            interpretability_requirement=0.9,
            accuracy_requirement=0.6
        )
        
        result = self.selector.select_methods(self.data_context, interpretable_context)
        
        self.assertTrue(result.success)
        
        # 检查推荐的方法是否具有高可解释性
        for recommendation in result.primary_recommendations:
            method = recommendation.method
            self.assertGreaterEqual(method.interpretability, 0.5)  # 至少中等可解释性
    
    def test_method_exclusion(self):
        """测试方法排除功能"""
        exclusion_context = ResearchContext(
            primary_objective=ResearchObjective.CLASSIFICATION,
            analysis_type=AnalysisType.PREDICTIVE,
            excluded_methods=["随机森林"]
        )
        
        result = self.selector.select_methods(self.data_context, exclusion_context)
        
        self.assertTrue(result.success)
        
        # 检查被排除的方法是否不在推荐中
        recommended_methods = [rec.method.name for rec in result.primary_recommendations]
        self.assertNotIn("随机森林", recommended_methods)
    
    def test_method_preference(self):
        """测试方法偏好功能"""
        preference_context = ResearchContext(
            primary_objective=ResearchObjective.CLASSIFICATION,
            analysis_type=AnalysisType.PREDICTIVE,
            preferred_methods=["逻辑回归"]
        )
        
        result = self.selector.select_methods(self.data_context, preference_context)
        
        self.assertTrue(result.success)
        
        # 检查偏好的方法是否获得了更高的评分
        for recommendation in result.primary_recommendations:
            if recommendation.method.name == "逻辑回归":
                self.assertGreater(recommendation.suitability_score, 0.5)
                break
    
    def test_add_custom_method(self):
        """测试添加自定义方法"""
        custom_method = AnalysisMethod(
            name="自定义方法",
            description="测试用的自定义分析方法",
            analysis_type=AnalysisType.PREDICTIVE,
            required_data_types={DataCharacteristic.NUMERICAL},
            suitable_objectives={ResearchObjective.CLASSIFICATION},
            min_sample_size=50,
            complexity=0.6,
            interpretability=0.7,
            accuracy_potential=0.8,
            computational_cost=0.5
        )
        
        success = self.selector.add_custom_method(custom_method)
        self.assertTrue(success)
        
        # 检查方法是否被添加到库中
        method_library = self.selector.get_method_library()
        self.assertIn("自定义方法", method_library)
        self.assertEqual(method_library["自定义方法"].description, "测试用的自定义分析方法")
    
    def test_update_method_performance(self):
        """测试更新方法性能"""
        method_name = "逻辑回归"
        
        # 获取初始性能
        initial_method = self.selector.method_library[method_name]
        initial_usage = initial_method.usage_count
        
        # 更新性能
        self.selector.update_method_performance(method_name, True, 0.85)
        
        # 检查更新后的性能
        updated_method = self.selector.method_library[method_name]
        self.assertEqual(updated_method.usage_count, initial_usage + 1)
        self.assertGreater(updated_method.success_rate, 0.0)
        self.assertGreater(updated_method.average_performance, 0.0)
    
    def test_method_combinations(self):
        """测试方法组合生成"""
        result = self.selector.select_methods(self.data_context, self.research_context)
        
        self.assertTrue(result.success)
        
        # 检查是否生成了方法组合
        if len(result.primary_recommendations) >= 2:
            self.assertGreaterEqual(len(result.method_combinations), 0)
            
            # 检查组合的结构
            for combination in result.method_combinations:
                self.assertIsInstance(combination, list)
                self.assertGreaterEqual(len(combination), 2)
                for method_rec in combination:
                    self.assertIsInstance(method_rec, MethodRecommendation)
    
    def test_reasoning_generation(self):
        """测试推理说明生成"""
        result = self.selector.select_methods(self.data_context, self.research_context)
        
        self.assertTrue(result.success)
        
        # 检查推理说明
        for recommendation in result.primary_recommendations:
            self.assertIsInstance(recommendation.reasoning, list)
            self.assertGreater(len(recommendation.reasoning), 0)
            
            # 检查推理说明的内容
            for reason in recommendation.reasoning:
                self.assertIsInstance(reason, str)
                self.assertGreater(len(reason), 0)
    
    def test_parameter_suggestions(self):
        """测试参数建议"""
        result = self.selector.select_methods(self.data_context, self.research_context)
        
        self.assertTrue(result.success)
        
        # 检查参数建议
        for recommendation in result.primary_recommendations:
            self.assertIsInstance(recommendation.recommended_parameters, dict)
            
            # 特定方法的参数检查
            if recommendation.method.name == "随机森林":
                params = recommendation.recommended_parameters
                if "n_estimators" in params:
                    self.assertIsInstance(params["n_estimators"], int)
                    self.assertGreater(params["n_estimators"], 0)
    
    def test_preprocessing_suggestions(self):
        """测试预处理建议"""
        # 创建有缺失值的数据上下文
        noisy_data_context = DataContext(
            sample_size=500,
            feature_count=8,
            data_characteristics={DataCharacteristic.NUMERICAL},
            missing_rate=0.15,  # 高缺失率
            has_outliers=True,   # 有异常值
            data_quality_score=0.6
        )
        
        result = self.selector.select_methods(noisy_data_context, self.research_context)
        
        self.assertTrue(result.success)
        
        # 检查预处理建议
        for recommendation in result.primary_recommendations:
            suggestions = recommendation.preprocessing_suggestions
            self.assertIsInstance(suggestions, list)
            
            # 应该建议处理缺失值
            self.assertTrue(any("缺失值" in suggestion for suggestion in suggestions))
    
    def test_potential_issues_identification(self):
        """测试潜在问题识别"""
        # 创建有问题的数据上下文
        problematic_context = DataContext(
            sample_size=10,  # 样本量很小
            feature_count=5,
            data_characteristics={DataCharacteristic.SMALL_SAMPLE},
            missing_rate=0.3,  # 高缺失率
            data_quality_score=0.4
        )
        
        result = self.selector.select_methods(problematic_context, self.research_context)
        
        self.assertTrue(result.success)
        
        # 检查潜在问题识别
        for recommendation in result.primary_recommendations:
            issues = recommendation.potential_issues
            self.assertIsInstance(issues, list)
            
            # 应该识别出样本量不足的问题
            if recommendation.method.min_sample_size > 10:
                self.assertTrue(any("样本量" in issue for issue in issues))
    
    def test_confidence_calculation(self):
        """测试置信度计算"""
        result = self.selector.select_methods(self.data_context, self.research_context)
        
        self.assertTrue(result.success)
        self.assertGreaterEqual(result.confidence_level, 0.0)
        self.assertLessEqual(result.confidence_level, 1.0)
        
        # 如果有多个推荐，置信度应该考虑分数差距
        if len(result.primary_recommendations) > 1:
            first_score = result.primary_recommendations[0].suitability_score
            second_score = result.primary_recommendations[1].suitability_score
            
            # 分数差距越大，置信度应该越高
            if first_score > second_score:
                self.assertGreater(result.confidence_level, 0.3)
    
    def test_selection_history(self):
        """测试选择历史记录"""
        initial_history_length = len(self.selector.selection_history)
        
        # 执行几次选择
        self.selector.select_methods(self.data_context, self.research_context)
        self.selector.select_methods(self.data_context, self.research_context)
        
        # 检查历史记录是否增加
        self.assertEqual(len(self.selector.selection_history), initial_history_length + 2)
        
        # 检查历史记录的内容
        for history_item in self.selector.selection_history:
            self.assertIsInstance(history_item, MethodSelectionResult)
            self.assertTrue(history_item.success)


if __name__ == '__main__':
    unittest.main()
