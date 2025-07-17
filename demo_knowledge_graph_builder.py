"""
知识图谱构建器演示脚本

展示如何使用知识图谱构建器从学术论文中构建知识图谱
"""

import sys
import os
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from research_automation.core.knowledge_graph_builder import KnowledgeGraphBuilder
from research_automation.models.analysis_models import Paper, PaperType


def create_sample_papers():
    """创建示例论文数据"""
    papers = [
        Paper(
            title="Attention Is All You Need: Transformer Architecture for Natural Language Processing",
            authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
            abstract="We propose the Transformer, a novel neural network architecture based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. The model achieves superior performance on machine translation tasks and establishes new state-of-the-art results.",
            keywords=["transformer", "attention mechanism", "neural networks", "machine translation", "natural language processing"],
            publication_year=2017,
            journal_or_venue="NIPS",
            citation_count=45000,
            paper_type=PaperType.CONFERENCE_PAPER,
            methodology="transformer architecture with multi-head attention",
            key_findings=["Attention mechanisms can replace recurrence", "Parallel computation advantages", "State-of-the-art translation quality"],
            research_fields=["natural language processing", "machine learning", "deep learning"]
        ),
        Paper(
            title="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            authors=["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee"],
            abstract="We introduce BERT, which stands for Bidirectional Encoder Representations from Transformers. BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context.",
            keywords=["BERT", "bidirectional", "transformers", "pre-training", "language understanding"],
            publication_year=2018,
            journal_or_venue="NAACL",
            citation_count=35000,
            paper_type=PaperType.CONFERENCE_PAPER,
            methodology="bidirectional transformer pre-training",
            key_findings=["Bidirectional context improves understanding", "Pre-training on large corpora", "Transfer learning effectiveness"],
            research_fields=["natural language processing", "machine learning"]
        ),
        Paper(
            title="ResNet: Deep Residual Learning for Image Recognition",
            authors=["Kaiming He", "Xiangyu Zhang", "Shaoqing Ren"],
            abstract="We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs.",
            keywords=["ResNet", "residual learning", "deep learning", "image recognition", "computer vision"],
            publication_year=2016,
            journal_or_venue="CVPR",
            citation_count=50000,
            paper_type=PaperType.CONFERENCE_PAPER,
            methodology="residual neural networks",
            key_findings=["Residual connections enable deeper networks", "Improved image classification accuracy", "Skip connections prevent vanishing gradients"],
            research_fields=["computer vision", "deep learning", "machine learning"]
        ),
        Paper(
            title="Generative Adversarial Networks",
            authors=["Ian Goodfellow", "Jean Pouget-Abadie", "Mehdi Mirza"],
            abstract="We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G.",
            keywords=["GAN", "generative models", "adversarial training", "deep learning", "neural networks"],
            publication_year=2014,
            journal_or_venue="NIPS",
            citation_count=40000,
            paper_type=PaperType.CONFERENCE_PAPER,
            methodology="adversarial training framework",
            key_findings=["Adversarial training generates realistic data", "Game-theoretic approach to learning", "Unsupervised learning breakthrough"],
            research_fields=["machine learning", "deep learning", "computer vision"]
        ),
        Paper(
            title="Deep Reinforcement Learning with Double Q-Learning",
            authors=["Hado van Hasselt", "Arthur Guez", "David Silver"],
            abstract="The popular Q-learning algorithm is known to overestimate action values under certain conditions. We show that the idea behind the Double Q-learning algorithm can be generalized to work with large-scale function approximation.",
            keywords=["reinforcement learning", "Q-learning", "deep learning", "function approximation"],
            publication_year=2015,
            journal_or_venue="AAAI",
            citation_count=8000,
            paper_type=PaperType.CONFERENCE_PAPER,
            methodology="double Q-learning with neural networks",
            key_findings=["Double Q-learning reduces overestimation", "Improved stability in deep RL", "Better performance on Atari games"],
            research_fields=["reinforcement learning", "machine learning", "deep learning"]
        ),
        Paper(
            title="Graph Neural Networks: A Review of Methods and Applications",
            authors=["Jie Zhou", "Ganqu Cui", "Shengding Hu"],
            abstract="Graph neural networks (GNNs) have emerged as a powerful tool for analyzing graph-structured data. This paper provides a comprehensive review of GNN methods and their applications across various domains.",
            keywords=["graph neural networks", "graph learning", "neural networks", "graph analysis"],
            publication_year=2020,
            journal_or_venue="AI Open",
            citation_count=3000,
            paper_type=PaperType.JOURNAL_ARTICLE,
            methodology="comprehensive survey and analysis",
            key_findings=["GNNs effective for graph data", "Various architectures available", "Wide range of applications"],
            research_fields=["machine learning", "graph analysis", "deep learning"]
        )
    ]
    return papers


def main():
    """主函数"""
    print("=" * 60)
    print("知识图谱构建器演示")
    print("=" * 60)
    
    # 创建知识图谱构建器
    builder = KnowledgeGraphBuilder()
    
    # 创建示例论文数据
    papers = create_sample_papers()
    print(f"\n创建了 {len(papers)} 篇示例论文")
    
    # 构建知识图谱
    print("\n正在构建知识图谱...")
    knowledge_graph = builder.build_knowledge_graph(papers)
    
    # 显示图谱统计信息
    stats = builder.get_graph_statistics(knowledge_graph)
    print(f"\n知识图谱统计信息:")
    print(f"- 节点数量: {stats['node_count']}")
    print(f"- 边数量: {stats['edge_count']}")
    print(f"- 研究空白数量: {stats['research_gaps_count']}")
    print(f"- 热点主题数量: {stats['hot_topics_count']}")
    print(f"- 新兴趋势数量: {stats['emerging_trends_count']}")
    
    # 显示节点类型分布
    print(f"\n节点类型分布:")
    for node_type, count in stats['node_type_distribution'].items():
        print(f"- {node_type}: {count}")
    
    # 显示关系类型分布
    print(f"\n关系类型分布:")
    for edge_type, count in stats['edge_type_distribution'].items():
        print(f"- {edge_type}: {count}")
    
    # 显示热点主题
    if knowledge_graph.hot_topics:
        print(f"\n热点主题 (前5个):")
        for i, topic in enumerate(knowledge_graph.hot_topics[:5], 1):
            print(f"{i}. {topic}")
    
    # 显示新兴趋势
    if knowledge_graph.emerging_trends:
        print(f"\n新兴趋势 (前5个):")
        for i, trend in enumerate(knowledge_graph.emerging_trends[:5], 1):
            print(f"{i}. {trend}")
    
    # 显示研究空白
    if knowledge_graph.research_gaps:
        print(f"\n主要研究空白 (前3个):")
        for i, gap in enumerate(knowledge_graph.research_gaps[:3], 1):
            print(f"{i}. {gap.description}")
            print(f"   类型: {gap.gap_type}")
            print(f"   重要性: {gap.importance_level:.2f}")
            print(f"   难度: {gap.difficulty_level:.2f}")
            if gap.suggested_approaches:
                print(f"   建议方法: {gap.suggested_approaches[0]}")
            print()
    
    # 生成可视化
    print("正在生成知识图谱可视化...")
    output_path = builder.visualize_knowledge_graph(knowledge_graph, "demo_knowledge_graph.html")
    print(f"可视化文件已保存到: {output_path}")
    
    # 显示一些具体的节点信息
    print(f"\n重要节点示例:")
    important_nodes = sorted(knowledge_graph.nodes, key=lambda x: x.importance_score, reverse=True)[:5]
    for i, node in enumerate(important_nodes, 1):
        print(f"{i}. {node.label} ({node.node_type})")
        print(f"   重要性分数: {node.importance_score:.3f}")
        print(f"   相关论文数: {len(node.related_papers)}")
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()