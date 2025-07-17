"""
文献质量评估系统测试

测试论文质量评估、筛选和报告生成功能
"""

import unittest
import sys
import os
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_automation.core.literature_quality_assessor import (
    LiteratureQualityAssessor, Paper, QualityScore, QualityDimension
)
from research_automation.models.research_models import TopicAnalysis, ResearchType, ResearchComplexity


class TestLiteratureQualityAssessor(unittest.TestCase):
    """文献质量评估系统测试类"""
    
    def setUp(self):
        """设置测试环境"""
        self.config = {
            'quality_threshold': 0.6,
            'citation_weight': 0.3
        }
        self.assessor = LiteratureQualityAssessor(config=self.config)
        
        # 创建测试论文
        self.high_quality_paper = Paper(
            title="A Novel Deep Learning Approach for Medical Image Analysis",
            authors=["John Smith", "Jane Doe", "Bob Johnson"],
            abstract="This paper presents a comprehensive and innovative deep learning framework for medical image analysis. We propose a novel convolutional neural network architecture that significantly outperforms existing methods. Extensive experiments on multiple datasets demonstrate the effectiveness and robustness of our approach. Statistical analysis shows significant improvements with p-value < 0.001.",
            keywords=["deep learning", "medical imaging", "CNN", "image analysis"],
            publication_year=2023,
            journal="Nature Medicine",
            citation_count=45,
            doi="10.1038/s41591-023-12345",
            venue_type="journal"
        )
        
        self.medium_quality_paper = Paper(
            title="Machine Learning Applications in Data Processing",
            authors=["Alice Brown"],
            abstract="This study explores machine learning applications in data processing. We implement several algorithms and compare their performance. Results show moderate improvements over baseline methods.",
            keywords=["machine learning", "data processing"],
            publication_year=2020,
            journal="IEEE Transactions on Data Engineering",
            citation_count=15,
            venue_type="journal"
        )
        
        self.low_quality_paper = Paper(
            title="Some Research",
            authors=["Unknown Author"],
            abstract="This is a preliminary study with limited scope. The results are incomplete and require further investigation.",
            keywords=[],
            publication_year=2019,
            journal="Unknown Journal",
            citation_count=2,
            venue_type="journal"
        )
    
    def test_component_initialization(self):
        """测试组件初始化"""
        self.assertTrue(self.assessor.is_initialized)
        self.assertIsNotNone(self.assessor.dimension_weights)
        self.assertIsNotNone(self.assessor.quality_indicators)
        self.assertIsNotNone(self.assessor.top_venues)
    
    def test_paper_validation(self):
        """测试论文数据验证"""
        # 有效论文
        self.assertTrue(self.high_quality_paper.validate())
        self.assertTrue(self.medium_quality_paper.validate())
        
        # 无效论文
        invalid_paper = Paper(
            title="",  # 空标题
            authors=[],  # 无作者
            abstract="",  # 空摘要
            keywords=[],
            publication_year=1800,  # 无效年份
            journal="Test Journal",
            citation_count=-1  # 负引用数
        )
        self.assertFalse(invalid_paper.validate())
    
    def test_evaluate_paper_quality_high_quality(self):
        """测试高质量论文评估"""
        quality_score = self.assessor.evaluate_paper_quality(self.high_quality_paper)
        
        # 验证结果类型和有效性
        self.assertIsInstance(quality_score, QualityScore)
        self.assertTrue(quality_score.validate())
        
        # 验证高质量论文的分数
        self.assertGreater(quality_score.overall_score, 0.7)
        self.assertGreater(quality_score.confidence, 0.6)
        
        # 验证各维度分数
        self.assertIn(QualityDimension.NOVELTY, quality_score.dimension_scores)
        self.assertIn(QualityDimension.METHODOLOGY, quality_score.dimension_scores)
        self.assertIn(QualityDimension.IMPACT, quality_score.dimension_scores)
        
        # 验证推理和建议
        self.assertGreater(len(quality_score.reasoning), 0)
        self.assertGreater(len(quality_score.recommendations), 0)
    
    def test_evaluate_paper_quality_low_quality(self):
        """测试低质量论文评估"""
        quality_score = self.assessor.evaluate_paper_quality(self.low_quality_paper)
        
        # 验证结果
        self.assertTrue(quality_score.validate())
        self.assertLess(quality_score.overall_score, 0.6)
        # 低质量论文的置信度可能仍然较高，因为数据完整性好
        self.assertGreater(quality_score.confidence, 0.0)
        
        # 应该包含改进建议
        self.assertGreater(len(quality_score.recommendations), 0)
    
    def test_evaluate_paper_quality_with_context(self):
        """测试带上下文的论文质量评估"""
        # 创建模拟主题分析
        context = type('TopicAnalysis', (), {
            'keywords': ['deep learning', 'medical imaging', 'neural network'],
            'research_type': ResearchType.EXPERIMENTAL,
            'complexity_level': ResearchComplexity.HIGH
        })()
        
        quality_score = self.assessor.evaluate_paper_quality(self.high_quality_paper, context)
        
        # 有上下文的评估应该有更高的相关性分数
        relevance_score = quality_score.dimension_scores[QualityDimension.RELEVANCE]
        self.assertGreater(relevance_score, 0.6)
    
    def test_assess_relevance(self):
        """测试相关性评估"""
        context = type('TopicAnalysis', (), {
            'keywords': ['deep learning', 'medical imaging']
        })()
        
        score, reasons = self.assessor._assess_relevance(self.high_quality_paper, context)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertIsInstance(reasons, list)
        self.assertGreater(len(reasons), 0)
    
    def test_assess_novelty(self):
        """测试新颖性评估"""
        score, reasons = self.assessor._assess_novelty(self.high_quality_paper)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertIsInstance(reasons, list)
        
        # 高质量论文应该有较高的新颖性分数
        self.assertGreater(score, 0.5)
    
    def test_assess_methodology(self):
        """测试方法论评估"""
        score, reasons = self.assessor._assess_methodology(self.high_quality_paper)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertIsInstance(reasons, list)
    
    def test_assess_impact(self):
        """测试影响力评估"""
        score, reasons = self.assessor._assess_impact(self.high_quality_paper)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertIsInstance(reasons, list)
        
        # 高引用论文应该有较高的影响力分数
        self.assertGreater(score, 0.4)
    
    def test_assess_credibility(self):
        """测试可信度评估"""
        score, reasons = self.assessor._assess_credibility(self.high_quality_paper)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertIsInstance(reasons, list)
    
    def test_assess_clarity(self):
        """测试清晰度评估"""
        score, reasons = self.assessor._assess_clarity(self.high_quality_paper)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertIsInstance(reasons, list)
    
    def test_assess_venue_quality(self):
        """测试期刊/会议质量评估"""
        # 测试顶级期刊
        nature_score = self.assessor._assess_venue_quality("Nature Medicine", "journal")
        self.assertGreater(nature_score, 0.8)
        
        # 测试普通期刊
        unknown_score = self.assessor._assess_venue_quality("Unknown Journal", "journal")
        self.assertLess(unknown_score, 0.7)
        
        # 测试会议
        conf_score = self.assessor._assess_venue_quality("ICML", "conference")
        self.assertGreater(conf_score, 0.8)
    
    def test_calculate_confidence(self):
        """测试置信度计算"""
        dimension_scores = {
            QualityDimension.RELEVANCE: 0.8,
            QualityDimension.NOVELTY: 0.7,
            QualityDimension.METHODOLOGY: 0.9,
            QualityDimension.IMPACT: 0.6,
            QualityDimension.CREDIBILITY: 0.8,
            QualityDimension.CLARITY: 0.7
        }
        
        confidence = self.assessor._calculate_confidence(self.high_quality_paper, dimension_scores)
        
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.1)
        self.assertLessEqual(confidence, 1.0)
        
        # 完整数据的论文应该有较高置信度
        self.assertGreater(confidence, 0.5)
    
    def test_generate_recommendations(self):
        """测试建议生成"""
        dimension_scores = {
            QualityDimension.RELEVANCE: 0.3,  # 低分
            QualityDimension.NOVELTY: 0.8,
            QualityDimension.METHODOLOGY: 0.4,  # 低分
            QualityDimension.IMPACT: 0.7,
            QualityDimension.CREDIBILITY: 0.6,
            QualityDimension.CLARITY: 0.5
        }
        
        recommendations = self.assessor._generate_recommendations(self.medium_quality_paper, dimension_scores)
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        # 应该包含针对低分维度的建议
        recommendations_text = ' '.join(recommendations)
        self.assertTrue(any(word in recommendations_text for word in ['相关性', '方法论']))
    
    def test_filter_high_quality_papers(self):
        """测试高质量论文筛选"""
        papers = [self.high_quality_paper, self.medium_quality_paper, self.low_quality_paper]
        
        high_quality_papers = self.assessor.filter_high_quality_papers(papers, min_score=0.6)
        
        self.assertIsInstance(high_quality_papers, list)
        # 应该筛选出至少一篇高质量论文
        self.assertGreater(len(high_quality_papers), 0)
        
        # 验证结果格式
        for paper, score in high_quality_papers:
            self.assertIsInstance(paper, Paper)
            self.assertIsInstance(score, QualityScore)
            self.assertGreaterEqual(score.overall_score, 0.6)
        
        # 验证排序（按分数降序）
        if len(high_quality_papers) > 1:
            for i in range(len(high_quality_papers) - 1):
                self.assertGreaterEqual(
                    high_quality_papers[i][1].overall_score,
                    high_quality_papers[i+1][1].overall_score
                )
    
    def test_batch_evaluate_papers(self):
        """测试批量论文评估"""
        papers = [self.high_quality_paper, self.medium_quality_paper, self.low_quality_paper]
        
        quality_scores = self.assessor.batch_evaluate_papers(papers)
        
        self.assertIsInstance(quality_scores, list)
        self.assertEqual(len(quality_scores), len(papers))
        
        # 验证每个评分结果
        for score in quality_scores:
            self.assertIsInstance(score, QualityScore)
            self.assertTrue(score.validate())
    
    def test_generate_quality_report(self):
        """测试质量评估报告生成"""
        papers = [self.high_quality_paper, self.medium_quality_paper, self.low_quality_paper]
        quality_scores = self.assessor.batch_evaluate_papers(papers)
        
        report = self.assessor.generate_quality_report(papers, quality_scores)
        
        # 验证报告结构
        self.assertIn('summary', report)
        self.assertIn('dimension_analysis', report)
        self.assertIn('top_papers', report)
        self.assertIn('recommendations', report)
        
        # 验证摘要信息
        summary = report['summary']
        self.assertEqual(summary['total_papers'], len(papers))
        self.assertIsInstance(summary['average_quality'], float)
        self.assertGreaterEqual(summary['high_quality_count'], 0)
        self.assertGreaterEqual(summary['medium_quality_count'], 0)
        self.assertGreaterEqual(summary['low_quality_count'], 0)
        
        # 验证维度分析
        dimension_analysis = report['dimension_analysis']
        for dimension in QualityDimension:
            self.assertIn(dimension.value, dimension_analysis)
            dim_data = dimension_analysis[dimension.value]
            self.assertIn('average', dim_data)
            self.assertIn('max', dim_data)
            self.assertIn('min', dim_data)
        
        # 验证顶级论文
        top_papers = report['top_papers']
        self.assertIsInstance(top_papers, list)
        self.assertLessEqual(len(top_papers), 5)
        
        # 验证建议
        recommendations = report['recommendations']
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
    
    def test_quality_score_validation(self):
        """测试质量评分验证"""
        # 有效的质量评分
        valid_score = QualityScore(
            overall_score=0.75,
            dimension_scores={
                QualityDimension.RELEVANCE: 0.8,
                QualityDimension.NOVELTY: 0.7
            },
            confidence=0.85,
            reasoning=["高质量论文"],
            recommendations=["建议优先阅读"]
        )
        self.assertTrue(valid_score.validate())
        
        # 无效的质量评分（分数超出范围）
        invalid_score = QualityScore(
            overall_score=1.5,  # 超出范围
            dimension_scores={QualityDimension.RELEVANCE: -0.1},  # 负分
            confidence=0.5,
            reasoning=[],
            recommendations=[]
        )
        self.assertFalse(invalid_score.validate())
    
    def test_empty_paper_list_handling(self):
        """测试空论文列表处理"""
        # 空列表筛选
        result = self.assessor.filter_high_quality_papers([])
        self.assertEqual(result, [])
        
        # 空列表批量评估
        scores = self.assessor.batch_evaluate_papers([])
        self.assertEqual(scores, [])
    
    def test_invalid_paper_handling(self):
        """测试无效论文处理"""
        invalid_paper = Paper(
            title="",
            authors=[],
            abstract="",
            keywords=[],
            publication_year=1800,
            journal="",
            citation_count=-1
        )
        
        # 应该抛出验证错误
        with self.assertRaises(Exception):
            self.assessor.evaluate_paper_quality(invalid_paper)
    
    def test_dimension_weights_sum(self):
        """测试维度权重总和"""
        total_weight = sum(self.assessor.dimension_weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=2)
    
    def test_quality_indicators_completeness(self):
        """测试质量指示词完整性"""
        indicators = self.assessor.quality_indicators
        
        # 验证必要的指示词类别存在
        self.assertIn('high_quality', indicators)
        self.assertIn('methodology', indicators)
        self.assertIn('negative', indicators)
        
        # 验证每个类别都有指示词
        for category, words in indicators.items():
            self.assertIsInstance(words, list)
            self.assertGreater(len(words), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)