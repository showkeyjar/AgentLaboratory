"""
动态检索范围扩展器演示脚本

展示如何使用动态检索范围扩展器来优化文献检索策略
"""

import sys
import os
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from research_automation.core.dynamic_search_expander import (
    DynamicSearchExpander, SearchStrategy, SearchResult
)
from research_automation.models.analysis_models import Paper, PaperType


def create_sample_papers():
    """创建示例论文数据"""
    papers = [
        Paper(
            title="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            authors=["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee", "Kristina Toutanova"],
            abstract="We introduce BERT, which stands for Bidirectional Encoder Representations from Transformers. BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.",
            keywords=["BERT", "transformers", "bidirectional", "pre-training", "language understanding", "natural language processing"],
            publication_year=2019,
            journal_or_venue="NAACL",
            citation_count=45000,
            paper_type=PaperType.CONFERENCE_PAPER,
            research_fields=["natural language processing", "machine learning"]
        ),
        Paper(
            title="GPT-3: Language Models are Few-Shot Learners",
            authors=["Tom Brown", "Benjamin Mann", "Nick Ryder"],
            abstract="We show that scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art fine-tuning approaches.",
            keywords=["GPT-3", "language models", "few-shot learning", "scaling", "transformers"],
            publication_year=2020,
            journal_or_venue="NeurIPS",
            citation_count=25000,
            paper_type=PaperType.CONFERENCE_PAPER,
            research_fields=["natural language processing", "machine learning"]
        ),
        Paper(
            title="Vision Transformer (ViT) for Image Recognition at Scale",
            authors=["Alexey Dosovitskiy", "Lucas Beyer", "Alexander Kolesnikov"],
            abstract="While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited.",
            keywords=["vision transformer", "ViT", "image recognition", "computer vision", "transformers"],
            publication_year=2021,
            journal_or_venue="ICLR",
            citation_count=15000,
            paper_type=PaperType.CONFERENCE_PAPER,
            research_fields=["computer vision", "machine learning"]
        ),
        Paper(
            title="Diffusion Models Beat GANs on Image Synthesis",
            authors=["Prafulla Dhariwal", "Alexander Nichol"],
            abstract="We show that diffusion models can achieve image sample quality superior to the current state-of-the-art generative models.",
            keywords=["diffusion models", "image synthesis", "generative models", "GANs"],
            publication_year=2021,
            journal_or_venue="NeurIPS",
            citation_count=8000,
            paper_type=PaperType.CONFERENCE_PAPER,
            research_fields=["computer vision", "machine learning", "generative models"]
        ),
        Paper(
            title="ChatGPT: Optimizing Language Models for Dialogue",
            authors=["OpenAI Team"],
            abstract="We've trained a model called ChatGPT which interacts in a conversational way. The dialogue format makes it possible for ChatGPT to answer followup questions, admit its mistakes, challenge incorrect premises, and reject inappropriate requests.",
            keywords=["ChatGPT", "dialogue", "conversational AI", "language models", "reinforcement learning"],
            publication_year=2022,
            journal_or_venue="OpenAI Blog",
            citation_count=5000,
            paper_type=PaperType.TECHNICAL_REPORT,
            research_fields=["natural language processing", "conversational AI"]
        ),
        Paper(
            title="LLaMA: Open and Efficient Foundation Language Models",
            authors=["Hugo Touvron", "Thibaut Lavril", "Gautier Izacard"],
            abstract="We introduce LLaMA, a collection of foundation language models ranging from 7B to 65B parameters. Our models outperform GPT-3 on most benchmarks.",
            keywords=["LLaMA", "foundation models", "language models", "efficiency", "open source"],
            publication_year=2023,
            journal_or_venue="arXiv",
            citation_count=2000,
            paper_type=PaperType.PREPRINT,
            research_fields=["natural language processing", "machine learning"]
        ),
        Paper(
            title="Segment Anything Model (SAM)",
            authors=["Alexander Kirillov", "Eric Mintun", "Nikhila Ravi"],
            abstract="We introduce the Segment Anything (SA) project: a new task, model, and dataset for image segmentation.",
            keywords=["SAM", "segmentation", "computer vision", "foundation models", "zero-shot"],
            publication_year=2023,
            journal_or_venue="ICCV",
            citation_count=1500,
            paper_type=PaperType.CONFERENCE_PAPER,
            research_fields=["computer vision", "machine learning"]
        ),
        Paper(
            title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
            authors=["Patrick Lewis", "Ethan Perez", "Aleksandara Piktus"],
            abstract="Large pre-trained language models have been shown to store factual knowledge in their parameters, and achieve state-of-the-art results when fine-tuned on downstream NLP tasks.",
            keywords=["RAG", "retrieval", "generation", "knowledge", "NLP"],
            publication_year=2020,
            journal_or_venue="NeurIPS",
            citation_count=3000,
            paper_type=PaperType.CONFERENCE_PAPER,
            research_fields=["natural language processing", "information retrieval"]
        )
    ]
    return papers


def main():
    """主函数"""
    print("=" * 70)
    print("动态检索范围扩展器演示")
    print("=" * 70)
    
    # 创建动态检索范围扩展器
    expander = DynamicSearchExpander()
    
    # 创建示例论文数据
    papers = create_sample_papers()
    print(f"\n创建了 {len(papers)} 篇示例论文")
    
    # 创建初始检索策略
    initial_strategy = SearchStrategy(
        keywords=["transformers", "language models"],
        search_fields=["title", "abstract", "keywords"],
        time_range=(2019, 2022),
        venue_filters=["NeurIPS", "ICLR"],
        citation_threshold=1000,
        max_results=50
    )
    
    print(f"\n初始检索策略:")
    print(f"- 关键词: {initial_strategy.keywords}")
    print(f"- 时间范围: {initial_strategy.time_range}")
    print(f"- 场所过滤: {initial_strategy.venue_filters}")
    print(f"- 引用阈值: {initial_strategy.citation_threshold}")
    
    # 模拟检索结果
    search_result = SearchResult(
        papers=papers,
        total_found=len(papers),
        search_time=2.5,
        strategy_used=initial_strategy,
        relevance_scores={paper.id: 0.85 for paper in papers},
        coverage_metrics={'coverage': 0.75}
    )
    
    print(f"\n模拟检索结果:")
    print(f"- 找到论文数: {search_result.total_found}")
    print(f"- 平均相关性: {search_result.get_average_relevance():.3f}")
    print(f"- 高质量论文数: {len(search_result.get_high_quality_papers())}")
    
    # 评估检索效果
    print(f"\n正在评估检索效果...")
    effectiveness_metrics = expander.evaluate_search_effectiveness(search_result)
    
    print(f"检索效果评估:")
    for metric, value in effectiveness_metrics.items():
        print(f"- {metric}: {value:.3f}")
    
    # 扩展检索策略
    print(f"\n正在扩展检索策略...")
    expanded_strategy = expander.expand_search_strategy(
        initial_strategy, [search_result], target_coverage=0.9
    )
    
    print(f"\n扩展后的检索策略:")
    print(f"- 关键词: {expanded_strategy.keywords}")
    print(f"- 时间范围: {expanded_strategy.time_range}")
    print(f"- 场所过滤: {expanded_strategy.venue_filters}")
    print(f"- 作者过滤: {expanded_strategy.author_filters}")
    
    # 发现新研究方向
    print(f"\n正在发现新研究方向...")
    new_directions = expander.discover_new_directions(papers)
    
    print(f"\n发现的新研究方向:")
    for i, direction in enumerate(new_directions[:10], 1):
        print(f"{i}. {direction}")
    
    # 分析论文趋势
    print(f"\n论文发表趋势分析:")
    year_counts = {}
    for paper in papers:
        year = paper.publication_year
        year_counts[year] = year_counts.get(year, 0) + 1
    
    for year in sorted(year_counts.keys()):
        print(f"- {year}: {year_counts[year]} 篇论文")
    
    # 分析研究领域分布
    print(f"\n研究领域分布:")
    field_counts = {}
    for paper in papers:
        for field in paper.research_fields:
            field_counts[field] = field_counts.get(field, 0) + 1
    
    for field, count in sorted(field_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"- {field}: {count} 篇论文")
    
    # 分析高引用论文
    print(f"\n高引用论文 (引用数 > 10000):")
    high_citation_papers = [p for p in papers if p.citation_count > 10000]
    for paper in sorted(high_citation_papers, key=lambda x: x.citation_count, reverse=True):
        print(f"- {paper.title[:50]}... ({paper.citation_count} 引用)")
    
    # 展示策略优化建议
    print(f"\n策略优化建议:")
    if len(expanded_strategy.keywords) > len(initial_strategy.keywords):
        new_keywords = set(expanded_strategy.keywords) - set(initial_strategy.keywords)
        print(f"- 建议添加关键词: {list(new_keywords)}")
    
    if expanded_strategy.time_range != initial_strategy.time_range:
        print(f"- 建议调整时间范围: {initial_strategy.time_range} → {expanded_strategy.time_range}")
    
    if len(expanded_strategy.venue_filters) > len(initial_strategy.venue_filters):
        new_venues = set(expanded_strategy.venue_filters) - set(initial_strategy.venue_filters)
        print(f"- 建议添加发表场所: {list(new_venues)}")
    
    # 获取性能摘要
    expander.search_history.append(search_result)
    performance_summary = expander.get_performance_summary()
    
    print(f"\n性能摘要:")
    print(f"- 总检索次数: {performance_summary['total_searches']}")
    print(f"- 总扩展次数: {performance_summary['total_expansions']}")
    print(f"- 平均效果: {performance_summary['average_effectiveness']:.3f}")
    
    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()