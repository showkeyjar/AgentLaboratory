"""
知识图谱构建器测试
"""

import unittest
import sys
import os
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_automation.core.knowledge_graph_builder import KnowledgeGraphBuilder
from research_automation.models.analysis_models import Paper, PaperType


class TestKnowledgeGraphBuilder(unittest.TestCase):
    """知识图谱构建器测试类"""
    
    def setUp(self):
        """设置测试环境"""
        self.builder = KnowledgeGraphBuilder()
        
        # 创建测试论文数据
        self.test_papers = [
            Paper(
                title="Deep Learning for Computer Vision",
                authors=["Alice Smith", "Bob Johnson"],
                abstract="This paper presents a comprehensive study of deep learning methods for computer vision tasks including image classification and object detection.",
                keywords=["deep learning", "computer vision", "neural networks", "image classification"],
                publication_year=2023,
                journal_or_venue="IEEE CVPR",
                citation_count=150,
                paper_type=PaperType.CONFERENCE_PAPER,
                methodology="convolutional neural networks",
                key_findings=["Improved accuracy on ImageNet", "Better object detection performance"],
                research_fields=["computer vision", "machine learning"]
            ),
            Paper(
                title="Natural Language Processing with Transformers",
                authors=["Carol Davis", "David Wilson"],
                abstract="We explore the application of transformer models in natural language processing tasks such as machine translation and sentiment analysis.",
                keywords=["transformers", "natural language processing", "machine translation", "attention mechanism"],
                publication_year=2022,
                journal_or_venue="ACL",
                citation_count=200,
                paper_type=PaperType.CONFERENCE_PAPER,
                methodology="transformer architecture",
                key_findings=["State-of-the-art translation quality", "Effective attention mechanisms"],
                research_fields=["natural language processing", "machine learning"]
            ),
            Paper(
                title="Reinforcement Learning for Robotics",
                authors=["Eve Brown", "Frank Miller"],
                abstract="This work investigates the use of reinforcement learning algorithms for robotic control and navigation tasks.",
                keywords=["reinforcement learning", "robotics", "control systems", "navigation"],
                publication_year=2021,
                journal_or_venue="ICRA",
                citation_count=80,
                paper_type=PaperType.CONFERENCE_PAPER,
                methodology="deep reinforcement learning",
                key_findings=["Improved robot navigation", "Better control policies"],
                research_fields=["robotics", "machine learning"]
            )
        ]
    
    def test_build_knowledge_graph(self):
        """测试知识图谱构建"""
        knowledge_graph = self.builder.build_knowledge_graph(self.test_papers)
        
        # 验证图谱结构
        self.assertIsNotNone(knowledge_graph)
        self.assertGreater(len(knowledge_graph.nodes), 0)
        self.assertGreater(len(knowledge_graph.edges), 0)
        
        # 验证节点类型
        node_types = set(node.node_type for node in knowledge_graph.nodes)
        expected_types = {"concept", "method", "author", "venue"}
        self.assertTrue(node_types.intersection(expected_types))
        
        print(f"构建的知识图谱包含 {len(knowledge_graph.nodes)} 个节点和 {len(knowledge_graph.edges)} 条边")
    
    def test_entity_extraction(self):
        """测试实体提取"""
        entities = self.builder._extract_entities(self.test_papers)
        
        # 验证提取的实体
        self.assertGreater(len(entities.concepts), 0)
        self.assertGreater(len(entities.methods), 0)
        self.assertGreater(len(entities.authors), 0)
        self.assertGreater(len(entities.venues), 0)
        
        # 验证特定实体
        self.assertIn("deep learning", [c.lower() for c in entities.concepts + entities.methods])
        self.assertIn("Alice Smith", entities.authors)
        self.assertIn("IEEE CVPR", entities.venues)
        
        print(f"提取的实体: 概念{len(entities.concepts)}个, 方法{len(entities.methods)}个, 作者{len(entities.authors)}个, 场所{len(entities.venues)}个")
    
    def test_research_gap_identification(self):
        """测试研究空白识别"""
        knowledge_graph = self.builder.build_knowledge_graph(self.test_papers)
        
        # 验证研究空白识别
        self.assertIsInstance(knowledge_graph.research_gaps, list)
        
        if knowledge_graph.research_gaps:
            gap = knowledge_graph.research_gaps[0]
            self.assertIsNotNone(gap.description)
            self.assertIn(gap.gap_type, ["methodological", "empirical", "theoretical"])
            self.assertGreaterEqual(gap.importance_level, 0.0)
            self.assertLessEqual(gap.importance_level, 1.0)
        
        print(f"识别的研究空白数量: {len(knowledge_graph.research_gaps)}")
    
    def test_hot_topics_identification(self):
        """测试热点主题识别"""
        knowledge_graph = self.builder.build_knowledge_graph(self.test_papers)
        
        # 验证热点主题识别
        self.assertIsInstance(knowledge_graph.hot_topics, list)
        
        print(f"识别的热点主题: {knowledge_graph.hot_topics}")
    
    def test_emerging_trends_identification(self):
        """测试新兴趋势识别"""
        knowledge_graph = self.builder.build_knowledge_graph(self.test_papers)
        
        # 验证新兴趋势识别
        self.assertIsInstance(knowledge_graph.emerging_trends, list)
        
        print(f"识别的新兴趋势: {knowledge_graph.emerging_trends}")
    
    def test_graph_statistics(self):
        """测试图谱统计信息"""
        knowledge_graph = self.builder.build_knowledge_graph(self.test_papers)
        stats = self.builder.get_graph_statistics(knowledge_graph)
        
        # 验证统计信息
        self.assertIn('node_count', stats)
        self.assertIn('edge_count', stats)
        self.assertIn('node_type_distribution', stats)
        self.assertIn('edge_type_distribution', stats)
        
        self.assertEqual(stats['node_count'], len(knowledge_graph.nodes))
        self.assertEqual(stats['edge_count'], len(knowledge_graph.edges))
        
        print(f"图谱统计信息: {stats}")
    
    def test_visualization(self):
        """测试可视化功能"""
        knowledge_graph = self.builder.build_knowledge_graph(self.test_papers)
        
        # 测试HTML可视化生成
        output_path = "test_knowledge_graph.html"
        result_path = self.builder.visualize_knowledge_graph(knowledge_graph, output_path)
        
        self.assertEqual(result_path, output_path)
        
        # 验证文件是否创建
        import os
        self.assertTrue(os.path.exists(output_path))
        
        # 清理测试文件
        if os.path.exists(output_path):
            os.remove(output_path)
        
        print("可视化功能测试通过")


if __name__ == '__main__':
    unittest.main()