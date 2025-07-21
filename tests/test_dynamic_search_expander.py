"""
动态检索范围扩展器测试
"""

import unittest
import sys
import os
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_automation.core.dynamic_search_expander import (
    DynamicSearchExpander, SearchStrategy, SearchResult, ExpansionSuggestion
)
from research_automation.models.analysis_models import Paper, PaperType


class TestDynamicSearchExpander(unittest.TestCase):
    """动态检索范围扩展器测试类"""
    
    def setUp(self):
        """设置测试环境"""
        self.expander = DynamicSearchExpander()
        
        # 创建测试论文数据
        self.test_papers = [
            Paper(
                title="Deep Learning for Computer Vision Applications",
                authors=["Alice Johnson", "Bob Smith"],
                abstract="This paper explores deep learning techniques for computer vision tasks including image classification, object detection, and semantic segmentation.",
                keywords=["deep learning", "computer vision", "image classification", "object detection", "CNN"],
                publication_year=2022,
                journal_or_venue="IEEE CVPR",
                citation_count=120,
                paper_type=PaperType.CONFERENCE_PAPER,
                research_fields=["computer vision", "machine learning"]
            ),
            Paper(
                title="Transformer Networks for Natural Language Processing",
                authors=["Carol Davis", "David Wilson"],
                abstract="We present a comprehensive study of transformer architectures for various NLP tasks including machine translation and text summarization.",
                keywords=["transformers", "natural language processing", "machine translation", "attention mechanism"],
                publication_year=2021,
                journal_or_venue="ACL",
                citation_count=200,
                paper_type=PaperType.CONFERENCE_PAPER,
                research_fields=["natural language processing", "machine learning"]
            ),
            Paper(
                title="Reinforcement Learning in Robotics: A Survey",
                authors=["Eve Brown", "Frank Miller"],
                abstract="This survey covers recent advances in applying reinforcement learning techniques to robotic control and navigation problems.",
                keywords=["reinforcement learning", "robotics", "control systems", "navigation"],
                publication_year=2020,
                journal_or_venue="Robotics and Autonomous Systems",
                citation_count=80,
                paper_type=PaperType.JOURNAL_ARTICLE,
                research_fields=["robotics", "reinforcement learning"]
            ),
            Paper(
                title="Graph Neural Networks for Social Network Analysis",
                authors=["Grace Lee", "Henry Zhang"],
                abstract="We propose novel graph neural network architectures for analyzing social networks and predicting user behavior.",
                keywords=["graph neural networks", "social networks", "user behavior", "network analysis"],
                publication_year=2023,
                journal_or_venue="KDD",
                citation_count=45,
                paper_type=PaperType.CONFERENCE_PAPER,
                research_fields=["machine learning", "social computing"]
            ),
            Paper(
                title="Federated Learning for Privacy-Preserving Machine Learning",
                authors=["Ivy Chen", "Jack Wang"],
                abstract="This work addresses privacy concerns in machine learning through federated learning approaches that keep data decentralized.",
                keywords=["federated learning", "privacy", "machine learning", "distributed systems"],
                publication_year=2023,
                journal_or_venue="ICML",
                citation_count=30,
                paper_type=PaperType.CONFERENCE_PAPER,
                research_fields=["machine learning", "privacy"]
            )
        ]
        
        # 创建测试检索策略
        self.test_strategy = SearchStrategy(
            keywords=["machine learning", "deep learning"],
            search_fields=["title", "abstract", "keywords"],
            time_range=(2020, 2023),
            venue_filters=["ICML", "NIPS"],
            citation_threshold=50,
            max_results=100
        )
        
        # 创建测试检索结果
        self.test_search_result = SearchResult(
            papers=self.test_papers,
            total_found=len(self.test_papers),
            search_time=1.5,
            strategy_used=self.test_strategy,
            relevance_scores={paper.id: 0.8 for paper in self.test_papers},
            coverage_metrics={'coverage': 0.7}
        )
    
    def test_expand_search_strategy(self):
        """测试检索策略扩展"""
        search_results = [self.test_search_result]
        
        expanded_strategy = self.expander.expand_search_strategy(
            self.test_strategy, search_results, target_coverage=0.8
        )
        
        # 验证扩展结果
        self.assertIsInstance(expanded_strategy, SearchStrategy)
        self.assertGreaterEqual(len(expanded_strategy.keywords), len(self.test_strategy.keywords))
        
        print(f"原始关键词数量: {len(self.test_strategy.keywords)}")
        print(f"扩展后关键词数量: {len(expanded_strategy.keywords)}")
        print(f"新增关键词: {set(expanded_strategy.keywords) - set(self.test_strategy.keywords)}")
    
    def test_discover_new_directions(self):
        """测试新研究方向发现"""
        new_directions = self.expander.discover_new_directions(self.test_papers)
        
        # 验证发现结果
        self.assertIsInstance(new_directions, list)
        self.assertGreater(len(new_directions), 0)
        
        print(f"发现的新研究方向数量: {len(new_directions)}")
        print(f"新研究方向: {new_directions}")
    
    def test_evaluate_search_effectiveness(self):
        """测试检索效果评估"""
        metrics = self.expander.evaluate_search_effectiveness(self.test_search_result)
        
        # 验证评估指标
        expected_metrics = ['coverage', 'relevance', 'diversity', 'novelty', 'quality', 'overall']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertGreaterEqual(metrics[metric], 0.0)
            self.assertLessEqual(metrics[metric], 1.0)
        
        print(f"检索效果评估结果:")
        for metric, value in metrics.items():
            print(f"- {metric}: {value:.3f}")
    
    def test_extract_emerging_concepts(self):
        """测试新兴概念提取"""
        emerging_concepts = self.expander._extract_emerging_concepts(self.test_papers)
        
        # 验证提取结果
        self.assertIsInstance(emerging_concepts, list)
        
        print(f"提取的新兴概念: {emerging_concepts}")
    
    def test_identify_cross_domain_connections(self):
        """测试跨领域连接识别"""
        cross_domain = self.expander._identify_cross_domain_connections(self.test_papers)
        
        # 验证识别结果
        self.assertIsInstance(cross_domain, list)
        
        print(f"识别的跨领域连接: {cross_domain}")
    
    def test_analyze_methodological_innovations(self):
        """测试方法论创新分析"""
        innovations = self.expander._analyze_methodological_innovations(self.test_papers)
        
        # 验证分析结果
        self.assertIsInstance(innovations, list)
        
        print(f"分析的方法论创新: {innovations}")
    
    def test_search_strategy_serialization(self):
        """测试检索策略序列化"""
        strategy_dict = self.test_strategy.to_dict()
        
        # 验证序列化结果
        self.assertIsInstance(strategy_dict, dict)
        self.assertIn('keywords', strategy_dict)
        self.assertIn('time_range', strategy_dict)
        self.assertIn('max_results', strategy_dict)
        
        print(f"策略序列化结果: {strategy_dict}")
    
    def test_search_result_metrics(self):
        """测试检索结果指标"""
        avg_relevance = self.test_search_result.get_average_relevance()
        high_quality_papers = self.test_search_result.get_high_quality_papers(threshold=0.7)
        
        # 验证指标计算
        self.assertGreaterEqual(avg_relevance, 0.0)
        self.assertLessEqual(avg_relevance, 1.0)
        self.assertIsInstance(high_quality_papers, list)
        self.assertLessEqual(len(high_quality_papers), len(self.test_papers))
        
        print(f"平均相关性: {avg_relevance:.3f}")
        print(f"高质量论文数量: {len(high_quality_papers)}")
    
    def test_expansion_suggestions(self):
        """测试扩展建议生成"""
        suggestions = self.expander._generate_expansion_suggestions(
            self.test_strategy, [self.test_search_result], target_coverage=0.8
        )
        
        # 验证建议生成
        self.assertIsInstance(suggestions, list)
        
        for suggestion in suggestions:
            self.assertIsInstance(suggestion, ExpansionSuggestion)
            self.assertGreaterEqual(suggestion.confidence_score, 0.0)
            self.assertLessEqual(suggestion.confidence_score, 1.0)
        
        print(f"生成的扩展建议数量: {len(suggestions)}")
        for i, suggestion in enumerate(suggestions):
            print(f"建议 {i+1}:")
            print(f"  置信度: {suggestion.confidence_score:.3f}")
            print(f"  新关键词: {suggestion.new_keywords[:3]}")
            print(f"  推理: {suggestion.reasoning}")
    
    def test_performance_summary(self):
        """测试性能摘要"""
        # 添加一些历史数据
        self.expander.search_history.append(self.test_search_result)
        
        summary = self.expander.get_performance_summary()
        
        # 验证摘要内容
        self.assertIn('total_searches', summary)
        self.assertIn('total_expansions', summary)
        self.assertIn('average_effectiveness', summary)
        self.assertIn('performance_metrics', summary)
        
        print(f"性能摘要: {summary}")


if __name__ == '__main__':
    unittest.main()