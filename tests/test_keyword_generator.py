"""
智能关键词生成器测试

测试关键词提取、扩展和搜索策略生成功能
"""

import unittest
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_automation.core.keyword_generator import KeywordGeneratorComponent, KeywordAnalysis
from research_automation.models.research_models import TopicAnalysis, ResearchType, ResearchComplexity


class TestKeywordGenerator(unittest.TestCase):
    """关键词生成器测试类"""
    
    def setUp(self):
        """设置测试环境"""
        self.config = {
            'max_keywords': 10,
            'similarity_threshold': 0.5
        }
        self.generator = KeywordGeneratorComponent(config=self.config)
    
    def test_component_initialization(self):
        """测试组件初始化"""
        self.assertTrue(self.generator.is_initialized)
        self.assertIsNotNone(self.generator.stop_words)
        self.assertIsNotNone(self.generator.academic_indicators)
        self.assertIsNotNone(self.generator.domain_vocabularies)
    
    def test_generate_keywords_basic(self):
        """测试基础关键词生成"""
        topic = "Machine learning for natural language processing"
        
        result = self.generator.generate_keywords(topic)
        
        # 验证结果类型
        self.assertIsInstance(result, KeywordAnalysis)
        self.assertTrue(result.validate())
        
        # 验证关键词数量
        self.assertGreater(len(result.primary_keywords), 0)
        self.assertGreater(len(result.search_strategies), 0)
        
        # 验证关键词质量
        self.assertIn('machine learning', ' '.join(result.primary_keywords).lower())
        self.assertIn('natural language processing', ' '.join(result.primary_keywords).lower())
    
    def test_generate_keywords_complex_topic(self):
        """测试复杂主题的关键词生成"""
        topic = "Deep learning approaches for quantum computing optimization in healthcare applications"
        
        result = self.generator.generate_keywords(topic)
        
        # 验证结果
        self.assertTrue(result.validate())
        self.assertGreater(len(result.primary_keywords), 3)
        self.assertGreater(len(result.domain_keywords), 0)
        self.assertGreater(len(result.method_keywords), 0)
        
        # 验证领域关键词识别
        all_keywords = ' '.join(result.primary_keywords + result.domain_keywords).lower()
        self.assertTrue(any(term in all_keywords for term in ['deep learning', 'quantum', 'healthcare']))
    
    def test_primary_keyword_extraction(self):
        """测试主要关键词提取"""
        topic = "Artificial intelligence applications in medical diagnosis"
        
        primary_keywords = self.generator._extract_primary_keywords(topic)
        
        self.assertIsInstance(primary_keywords, list)
        self.assertGreater(len(primary_keywords), 0)
        
        # 验证关键词质量
        keywords_str = ' '.join(primary_keywords).lower()
        self.assertIn('artificial intelligence', keywords_str)
        self.assertIn('medical', keywords_str)
    
    def test_secondary_keyword_generation(self):
        """测试次要关键词生成"""
        primary_keywords = ['machine learning', 'neural network', 'classification']
        topic = "Machine learning neural network classification algorithms"
        
        secondary_keywords = self.generator._generate_secondary_keywords(topic, primary_keywords)
        
        self.assertIsInstance(secondary_keywords, list)
        # 次要关键词不应包含主要关键词
        for secondary in secondary_keywords:
            self.assertNotIn(secondary, primary_keywords)
    
    def test_domain_keyword_identification(self):
        """测试领域关键词识别"""
        topic = "quantum computing algorithms for optimization problems"
        primary_keywords = ['quantum computing', 'algorithms', 'optimization']
        
        domain_keywords = self.generator._identify_domain_keywords(topic, primary_keywords)
        
        self.assertIsInstance(domain_keywords, list)
        # 应该识别出物理学和计算机科学相关的关键词
        domain_str = ' '.join(domain_keywords).lower()
        self.assertTrue(any(term in domain_str for term in ['quantum', 'algorithm', 'optimization']))
    
    def test_method_keyword_extraction(self):
        """测试方法关键词提取"""
        topic = "Statistical analysis and machine learning methods for data mining"
        primary_keywords = ['statistical analysis', 'machine learning', 'data mining']
        
        method_keywords = self.generator._extract_method_keywords(topic, primary_keywords)
        
        self.assertIsInstance(method_keywords, list)
        # 应该包含方法相关的关键词
        method_str = ' '.join(method_keywords).lower()
        self.assertTrue(any(term in method_str for term in ['analysis', 'learning', 'method']))
    
    def test_keyword_expansion(self):
        """测试关键词扩展"""
        keywords = ['machine learning', 'artificial intelligence', 'neural network']
        
        expanded_keywords = self.generator._expand_keywords(keywords)
        
        self.assertIsInstance(expanded_keywords, list)
        # 扩展后的关键词应该包含同义词
        expanded_str = ' '.join(expanded_keywords).lower()
        self.assertTrue(any(term in expanded_str for term in ['ml', 'ai', 'deep learning']))
    
    def test_keyword_combinations(self):
        """测试关键词组合生成"""
        primary = ['machine learning', 'classification']
        secondary = ['algorithm', 'model']
        domain = ['computer science', 'artificial intelligence']
        
        combinations = self.generator._generate_keyword_combinations(primary, secondary, domain)
        
        self.assertIsInstance(combinations, list)
        self.assertGreater(len(combinations), 0)
        
        # 验证组合格式
        combination_str = ' '.join(combinations)
        self.assertTrue('AND' in combination_str or 'OR' in combination_str)
    
    def test_relevance_score_calculation(self):
        """测试相关性分数计算"""
        topic = "Deep learning for image recognition"
        keywords = ['deep learning', 'image recognition', 'neural network', 'computer vision', 'unrelated term']
        
        scores = self.generator._calculate_relevance_scores(topic, keywords)
        
        self.assertIsInstance(scores, dict)
        self.assertEqual(len(scores), len(keywords))
        
        # 验证分数范围
        for score in scores.values():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
        
        # 相关关键词应该有更高的分数
        self.assertGreater(scores['deep learning'], scores['unrelated term'])
        self.assertGreater(scores['image recognition'], scores['unrelated term'])
    
    def test_search_strategy_generation(self):
        """测试搜索策略生成"""
        primary = ['machine learning', 'classification']
        secondary = ['algorithm', 'model']
        domain = ['computer science']
        method = ['statistical analysis']
        context = {'search_type': 'balanced'}
        
        strategies = self.generator._generate_search_strategies(
            primary, secondary, domain, method, context
        )
        
        self.assertIsInstance(strategies, list)
        self.assertGreater(len(strategies), 0)
        
        # 验证策略结构
        for strategy in strategies:
            self.assertIn('name', strategy)
            self.assertIn('description', strategy)
            self.assertIn('keywords', strategy)
            self.assertIn('search_query', strategy)
    
    def test_search_query_optimization(self):
        """测试搜索查询优化"""
        keywords = ['machine learning', 'neural network', 'classification']
        
        # 测试不同搜索类型
        contexts = [
            {'search_type': 'broad'},
            {'search_type': 'precise'},
            {'search_type': 'balanced'},
            {'search_type': 'balanced', 'time_filter': '2020-2023'}
        ]
        
        for context in contexts:
            query = self.generator.optimize_search_query(keywords, context)
            self.assertIsInstance(query, str)
            self.assertGreater(len(query), 0)
            
            # 验证查询包含关键词
            for keyword in keywords[:3]:  # 前3个关键词应该在查询中
                if context['search_type'] != 'precise':
                    self.assertIn(keyword, query.lower())
    
    def test_keyword_expansion_with_context(self):
        """测试基于上下文的关键词扩展"""
        keywords = ['machine learning', 'classification']
        
        # 创建模拟的主题分析结果
        topic_analysis = type('TopicAnalysis', (), {
            'research_type': ResearchType.EXPERIMENTAL,
            'complexity_level': ResearchComplexity.HIGH,
            'related_fields': ['Computer Science', 'Mathematics']
        })()
        
        expanded_keywords = self.generator.expand_keywords_with_context(keywords, topic_analysis)
        
        self.assertIsInstance(expanded_keywords, list)
        self.assertGreaterEqual(len(expanded_keywords), len(keywords))
        
        # 验证包含原始关键词
        for keyword in keywords:
            self.assertIn(keyword, expanded_keywords)
        
        # 验证包含实验相关的关键词
        expanded_str = ' '.join(expanded_keywords).lower()
        self.assertTrue(any(term in expanded_str for term in ['experiment', 'statistical', 'hypothesis']))
    
    def test_empty_topic_handling(self):
        """测试空主题处理"""
        with self.assertRaises(Exception):
            self.generator.generate_keywords("")
        
        with self.assertRaises(Exception):
            self.generator.generate_keywords("   ")
    
    def test_chinese_topic_handling(self):
        """测试中文主题处理"""
        topic = "机器学习在自然语言处理中的应用"
        
        # 当前实现主要针对英文，中文主题可能返回较少的关键词
        result = self.generator.generate_keywords(topic)
        
        self.assertIsInstance(result, KeywordAnalysis)
        self.assertTrue(result.validate())
        # 至少应该有一些基础的搜索策略
        self.assertGreater(len(result.search_strategies), 0)
    
    def test_word_variants_generation(self):
        """测试单词变体生成"""
        # 测试单复数
        variants = self.generator._generate_word_variants("algorithm")
        self.assertIn("algorithms", variants)
        
        variants = self.generator._generate_word_variants("algorithms")
        self.assertIn("algorithm", variants)
        
        # 测试动词变体
        variants = self.generator._generate_word_variants("learning")
        self.assertTrue(any(v in variants for v in ["learn", "learned"]))
    
    def test_semantic_similarity_calculation(self):
        """测试语义相似度计算"""
        # 测试相似的词汇
        similarity1 = self.generator._calculate_semantic_similarity(
            "machine learning", "artificial intelligence machine learning"
        )
        
        # 测试不相关的词汇
        similarity2 = self.generator._calculate_semantic_similarity(
            "machine learning", "cooking recipes"
        )
        
        self.assertGreater(similarity1, similarity2)
        self.assertGreaterEqual(similarity1, 0.0)
        self.assertLessEqual(similarity1, 1.0)
    
    def test_academic_importance_calculation(self):
        """测试学术重要性计算"""
        # 测试学术关键词
        importance1 = self.generator._calculate_academic_importance("machine learning algorithm")
        importance2 = self.generator._calculate_academic_importance("random word")
        
        self.assertGreater(importance1, importance2)
        self.assertGreaterEqual(importance1, 0.0)
        self.assertLessEqual(importance1, 1.0)


class TestKeywordAnalysis(unittest.TestCase):
    """关键词分析结果测试类"""
    
    def test_keyword_analysis_validation(self):
        """测试关键词分析结果验证"""
        # 有效的分析结果
        valid_analysis = KeywordAnalysis(
            primary_keywords=['machine learning', 'classification'],
            secondary_keywords=['algorithm', 'model'],
            domain_keywords=['computer science'],
            method_keywords=['statistical analysis'],
            expanded_keywords=['ML', 'AI'],
            keyword_combinations=['machine learning AND classification'],
            search_strategies=[{
                'name': 'test strategy',
                'keywords': ['test'],
                'search_query': 'test'
            }],
            relevance_scores={'machine learning': 0.9, 'classification': 0.8}
        )
        
        self.assertTrue(valid_analysis.validate())
        
        # 无效的分析结果（缺少主要关键词）
        invalid_analysis = KeywordAnalysis(
            primary_keywords=[],
            secondary_keywords=['algorithm'],
            domain_keywords=['computer science'],
            method_keywords=['statistical analysis'],
            expanded_keywords=['ML'],
            keyword_combinations=['test'],
            search_strategies=[],
            relevance_scores={'test': 0.5}
        )
        
        self.assertFalse(invalid_analysis.validate())


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)