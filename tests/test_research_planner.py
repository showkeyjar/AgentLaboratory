"""
研究规划组件单元测试

测试主题分析引擎的各项功能
"""

import unittest
from datetime import datetime

from research_automation.core.research_planner import ResearchPlannerComponent
from research_automation.models.research_models import ResearchComplexity, ResearchType
from research_automation.core.exceptions import ValidationError


class TestResearchPlannerComponent(unittest.TestCase):
    """研究规划组件测试类"""
    
    def setUp(self):
        """测试前设置"""
        self.config = {
            'llm_model': 'test_model',
            'max_analysis_time': 300
        }
        self.planner = ResearchPlannerComponent(config=self.config)
    
    def test_initialization(self):
        """测试组件初始化"""
        self.assertTrue(self.planner.is_initialized)
        self.assertEqual(self.planner.get_config('llm_model'), 'test_model')
    
    def test_analyze_simple_topic(self):
        """测试简单主题分析"""
        topic = "A survey of machine learning algorithms"
        analysis = self.planner.analyze_topic(topic)
        
        # 验证基本属性
        self.assertIsNotNone(analysis)
        self.assertEqual(analysis.topic, topic.lower())
        self.assertTrue(analysis.validate())
        
        # 验证复杂度评估
        self.assertGreaterEqual(analysis.complexity_score, 0.0)
        self.assertLessEqual(analysis.complexity_score, 1.0)
        
        # 验证研究类型识别
        self.assertEqual(analysis.research_type, ResearchType.SURVEY)
        
        # 验证关键词提取
        self.assertIn('machine learning', analysis.keywords)
        self.assertIn('survey', analysis.keywords)
    
    def test_analyze_complex_topic(self):
        """测试复杂主题分析"""
        topic = "Deep reinforcement learning for autonomous vehicle navigation using multi-modal sensor fusion and quantum computing optimization"
        analysis = self.planner.analyze_topic(topic)
        
        # 验证复杂度等级
        self.assertIn(analysis.complexity_level, [ResearchComplexity.HIGH, ResearchComplexity.VERY_HIGH])
        
        # 验证跨学科特征
        self.assertGreater(len(analysis.related_fields), 1)
        
        # 验证挑战识别
        self.assertGreater(len(analysis.potential_challenges), 3)
        
        # 验证资源需求
        self.assertIn("高性能计算资源", analysis.required_resources)
    
    def test_analyze_theoretical_topic(self):
        """测试理论研究主题分析"""
        topic = "Mathematical proof of convergence properties in optimization algorithms"
        analysis = self.planner.analyze_topic(topic)
        
        # 验证研究类型
        self.assertEqual(analysis.research_type, ResearchType.THEORETICAL)
        
        # 验证研究方向
        self.assertTrue(any("理论" in direction for direction in analysis.suggested_directions))
    
    def test_keyword_extraction(self):
        """测试关键词提取功能"""
        topic = "Natural language processing using transformer models for sentiment analysis"
        keywords = self.planner._extract_keywords(topic)
        
        # 验证关键词提取
        self.assertIn('natural language processing', keywords)
        self.assertIn('transformer', keywords)
        self.assertIn('sentiment analysis', keywords)
        
        # 验证停用词过滤
        self.assertNotIn('using', keywords)
        self.assertNotIn('for', keywords)
    
    def test_complexity_calculation(self):
        """测试复杂度计算"""
        # 简单主题
        simple_topic = "literature review of basic algorithms"
        simple_keywords = self.planner._extract_keywords(simple_topic)
        simple_score = self.planner._calculate_complexity_score(simple_topic, simple_keywords)
        
        # 复杂主题
        complex_topic = "quantum machine learning with deep neural networks for multi-modal data fusion"
        complex_keywords = self.planner._extract_keywords(complex_topic)
        complex_score = self.planner._calculate_complexity_score(complex_topic, complex_keywords)
        
        # 验证复杂主题得分更高
        self.assertGreater(complex_score, simple_score)
    
    def test_research_type_identification(self):
        """测试研究类型识别"""
        test_cases = [
            ("systematic review of deep learning methods", ResearchType.SURVEY),
            ("theoretical analysis of algorithm complexity", ResearchType.THEORETICAL),
            ("experimental evaluation of machine learning models", ResearchType.EXPERIMENTAL),
            ("case study of real-world application", ResearchType.CASE_STUDY)
        ]
        
        for topic, expected_type in test_cases:
            keywords = self.planner._extract_keywords(topic)
            identified_type = self.planner._identify_research_type(topic, keywords)
            self.assertEqual(identified_type, expected_type, 
                           f"Failed for topic: {topic}")
    
    def test_duration_estimation(self):
        """测试研究时长估算"""
        # 不同复杂度的时长估算
        low_duration = self.planner._estimate_duration(
            ResearchComplexity.LOW, ResearchType.SURVEY, 1
        )
        high_duration = self.planner._estimate_duration(
            ResearchComplexity.VERY_HIGH, ResearchType.EXPERIMENTAL, 3
        )
        
        # 验证高复杂度需要更长时间
        self.assertGreater(high_duration, low_duration)
        
        # 验证时长合理性
        self.assertGreater(low_duration, 0)
        self.assertLess(high_duration, 1000)  # 不超过3年
    
    def test_success_probability_estimation(self):
        """测试成功概率估算"""
        # 简单项目
        simple_prob = self.planner._estimate_success_probability(0.2, 2)
        
        # 复杂项目
        complex_prob = self.planner._estimate_success_probability(0.9, 8)
        
        # 验证简单项目成功概率更高
        self.assertGreater(simple_prob, complex_prob)
        
        # 验证概率范围
        self.assertGreaterEqual(simple_prob, 0.1)
        self.assertLessEqual(simple_prob, 0.95)
        self.assertGreaterEqual(complex_prob, 0.1)
        self.assertLessEqual(complex_prob, 0.95)
    
    def test_empty_topic_validation(self):
        """测试空主题验证"""
        with self.assertRaises(ValidationError):
            self.planner.analyze_topic("")
        
        with self.assertRaises(ValidationError):
            self.planner.analyze_topic("   ")
    
    def test_related_fields_identification(self):
        """测试相关领域识别"""
        # 计算机科学主题
        cs_keywords = ['algorithm', 'programming', 'machine learning']
        cs_fields = self.planner._identify_related_fields(cs_keywords)
        self.assertIn('Computer Science', cs_fields)
        
        # 生物学主题
        bio_keywords = ['genomics', 'protein', 'bioinformatics']
        bio_fields = self.planner._identify_related_fields(bio_keywords)
        self.assertIn('Biology', bio_fields)
        
        # 跨学科主题
        multi_keywords = ['machine learning', 'genomics', 'quantum']
        multi_fields = self.planner._identify_related_fields(multi_keywords)
        self.assertGreater(len(multi_fields), 1)
    
    def test_challenge_identification(self):
        """测试挑战识别"""
        # 高复杂度挑战
        high_challenges = self.planner._identify_challenges(
            ResearchComplexity.VERY_HIGH, ResearchType.EXPERIMENTAL, ['CS', 'Biology', 'Physics']
        )
        
        # 验证包含复杂性相关挑战
        self.assertIn('项目管理复杂性', high_challenges)
        self.assertIn('知识整合', high_challenges)  # 跨学科挑战
        
        # 低复杂度挑战
        low_challenges = self.planner._identify_challenges(
            ResearchComplexity.LOW, ResearchType.SURVEY, ['CS']
        )
        
        # 验证高复杂度挑战更多
        self.assertGreater(len(high_challenges), len(low_challenges))
    
    def test_resource_identification(self):
        """测试资源识别"""
        resources = self.planner._identify_required_resources(
            ResearchComplexity.HIGH, ResearchType.EXPERIMENTAL, ['Computer Science', 'Biology']
        )
        
        # 验证基础资源
        self.assertIn('文献数据库访问', resources)
        self.assertIn('计算设备', resources)
        
        # 验证高复杂度资源
        self.assertIn('高性能计算资源', resources)
        
        # 验证实验类型资源
        self.assertIn('实验环境', resources)
        
        # 验证领域特定资源
        self.assertIn('编程环境', resources)  # CS相关
    
    def test_analysis_validation(self):
        """测试分析结果验证"""
        topic = "Machine learning for healthcare applications"
        analysis = self.planner.analyze_topic(topic)
        
        # 验证所有必需字段
        self.assertTrue(analysis.validate())
        self.assertIsNotNone(analysis.id)
        self.assertIsInstance(analysis.created_at, datetime)
        self.assertGreater(len(analysis.keywords), 0)
        self.assertGreater(len(analysis.research_scope), 0)
        self.assertGreater(analysis.estimated_duration, 0)


if __name__ == '__main__':
    unittest.main()