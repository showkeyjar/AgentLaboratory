"""
文献质量评估系统演示

展示论文质量评估、筛选和报告生成功能
"""

import sys
sys.path.append('.')

from research_automation.core.literature_quality_assessor import (
    LiteratureQualityAssessor, Paper, QualityDimension
)
from research_automation.models.research_models import TopicAnalysis, ResearchType, ResearchComplexity


def demo_literature_quality_assessment():
    """演示文献质量评估功能"""
    print("📚 文献质量评估系统演示")
    print("=" * 50)
    
    # 初始化组件
    config = {
        'quality_threshold': 0.6,
        'citation_weight': 0.3
    }
    
    assessor = LiteratureQualityAssessor(config=config)
    
    # 创建测试论文
    papers = create_test_papers()
    
    print(f"\n📋 测试论文数量: {len(papers)}")
    print("-" * 50)
    
    # 演示单篇论文评估
    demo_single_paper_evaluation(assessor, papers)
    
    # 演示批量论文评估
    demo_batch_evaluation(assessor, papers)
    
    # 演示高质量论文筛选
    demo_quality_filtering(assessor, papers)
    
    # 演示质量评估报告
    demo_quality_report(assessor, papers)
    
    # 演示上下文相关评估
    demo_context_aware_evaluation(assessor, papers)
    
    print(f"\n🎉 演示完成!")
    print("=" * 50)


def create_test_papers():
    """创建测试论文"""
    papers = [
        Paper(
            title="A Novel Deep Learning Framework for Medical Image Analysis with Explainable AI",
            authors=["Dr. Sarah Johnson", "Prof. Michael Chen", "Dr. Emily Rodriguez"],
            abstract="This paper presents a comprehensive and innovative deep learning framework for medical image analysis that incorporates explainable AI techniques. We propose a novel convolutional neural network architecture that significantly outperforms existing methods on multiple medical imaging datasets. Extensive experiments demonstrate the effectiveness and robustness of our approach. Statistical analysis shows significant improvements with p-value < 0.001. The proposed method achieves state-of-the-art performance while providing interpretable results for clinical decision-making.",
            keywords=["deep learning", "medical imaging", "explainable AI", "CNN", "image analysis"],
            publication_year=2023,
            journal="Nature Medicine",
            citation_count=89,
            doi="10.1038/s41591-023-12345",
            venue_type="journal"
        ),
        Paper(
            title="Machine Learning Applications in Data Processing: A Comparative Study",
            authors=["Alice Brown", "Bob Wilson"],
            abstract="This study explores various machine learning applications in data processing tasks. We implement several popular algorithms including decision trees, random forests, and support vector machines. Results show moderate improvements over baseline methods. The evaluation is conducted on standard datasets with cross-validation. Further research is needed to optimize hyperparameters.",
            keywords=["machine learning", "data processing", "algorithms"],
            publication_year=2021,
            journal="IEEE Transactions on Data Engineering",
            citation_count=23,
            venue_type="journal"
        ),
        Paper(
            title="Preliminary Investigation of Some Methods",
            authors=["Unknown Author"],
            abstract="This is a preliminary study with limited scope and incomplete methodology. The results are inconclusive and require further investigation. Sample size is small and statistical significance is not achieved.",
            keywords=[],
            publication_year=2019,
            journal="Unknown Conference Proceedings",
            citation_count=3,
            venue_type="conference"
        ),
        Paper(
            title="Quantum Computing Algorithms for Optimization: A Comprehensive Survey",
            authors=["Prof. David Kim", "Dr. Lisa Zhang", "Dr. James Miller", "Dr. Anna Petrov"],
            abstract="This comprehensive survey reviews the current state of quantum computing algorithms for optimization problems. We systematically analyze various quantum optimization approaches including QAOA, VQE, and quantum annealing. The paper provides a thorough comparison of theoretical foundations and practical implementations. We identify key challenges and future research directions in quantum optimization. This work serves as a valuable reference for researchers entering the field.",
            keywords=["quantum computing", "optimization", "QAOA", "VQE", "quantum algorithms"],
            publication_year=2022,
            journal="Reviews of Modern Physics",
            citation_count=156,
            doi="10.1103/RevModPhys.94.015004",
            venue_type="journal"
        ),
        Paper(
            title="An Approach to Something",
            authors=["Student A"],
            abstract="We tried some things and got some results. The method is not clearly described. Results are presented without proper statistical analysis.",
            keywords=["approach", "method"],
            publication_year=2020,
            journal="Workshop Proceedings",
            citation_count=1,
            venue_type="workshop"
        )
    ]
    
    return papers


def demo_single_paper_evaluation(assessor, papers):
    """演示单篇论文评估"""
    print(f"\n🔍 单篇论文质量评估演示")
    
    # 选择第一篇论文进行详细评估
    paper = papers[0]
    print(f"\n📄 评估论文: {paper.title[:60]}...")
    print(f"   作者: {', '.join(paper.authors[:2])}{'等' if len(paper.authors) > 2 else ''}")
    print(f"   期刊: {paper.journal}")
    print(f"   年份: {paper.publication_year}")
    print(f"   引用: {paper.citation_count}")
    
    # 进行质量评估
    quality_score = assessor.evaluate_paper_quality(paper)
    
    print(f"\n✅ 评估结果:")
    print(f"   综合分数: {quality_score.overall_score:.2f}")
    print(f"   置信度: {quality_score.confidence:.2f}")
    
    print(f"\n📊 各维度分数:")
    for dimension, score in quality_score.dimension_scores.items():
        print(f"   {dimension.value}: {score:.2f}")
    
    print(f"\n💡 评估理由:")
    for reason in quality_score.reasoning[:5]:
        print(f"   • {reason}")
    
    print(f"\n🎯 建议:")
    for recommendation in quality_score.recommendations[:3]:
        print(f"   • {recommendation}")


def demo_batch_evaluation(assessor, papers):
    """演示批量论文评估"""
    print(f"\n📊 批量论文评估演示")
    
    # 批量评估所有论文
    quality_scores = assessor.batch_evaluate_papers(papers)
    
    print(f"\n📈 评估结果汇总:")
    for i, (paper, score) in enumerate(zip(papers, quality_scores)):
        print(f"   {i+1}. {paper.title[:40]}...")
        print(f"      综合分数: {score.overall_score:.2f}")
        print(f"      置信度: {score.confidence:.2f}")
        
        # 显示质量等级
        if score.overall_score >= 0.8:
            quality_level = "优秀"
        elif score.overall_score >= 0.6:
            quality_level = "良好"
        elif score.overall_score >= 0.4:
            quality_level = "一般"
        else:
            quality_level = "较差"
        
        print(f"      质量等级: {quality_level}")
        print()


def demo_quality_filtering(assessor, papers):
    """演示高质量论文筛选"""
    print(f"\n🔎 高质量论文筛选演示")
    
    # 设置不同的质量阈值进行筛选
    thresholds = [0.8, 0.6, 0.4]
    
    for threshold in thresholds:
        print(f"\n   质量阈值: {threshold}")
        
        high_quality_papers = assessor.filter_high_quality_papers(
            papers, min_score=threshold
        )
        
        print(f"   筛选结果: {len(papers)}篇论文中筛选出{len(high_quality_papers)}篇")
        
        if high_quality_papers:
            print(f"   高质量论文:")
            for i, (paper, score) in enumerate(high_quality_papers[:3], 1):
                print(f"      {i}. {paper.title[:50]}...")
                print(f"         分数: {score.overall_score:.2f}")
        else:
            print(f"   无论文达到该质量阈值")


def demo_quality_report(assessor, papers):
    """演示质量评估报告"""
    print(f"\n📋 质量评估报告演示")
    
    # 批量评估
    quality_scores = assessor.batch_evaluate_papers(papers)
    
    # 生成质量报告
    report = assessor.generate_quality_report(papers, quality_scores)
    
    print(f"\n📊 报告摘要:")
    summary = report['summary']
    print(f"   总论文数: {summary['total_papers']}")
    print(f"   平均质量: {summary['average_quality']:.2f}")
    print(f"   高质量论文: {summary['high_quality_count']}篇")
    print(f"   中等质量论文: {summary['medium_quality_count']}篇")
    print(f"   低质量论文: {summary['low_quality_count']}篇")
    
    print(f"\n📈 维度分析:")
    dimension_analysis = report['dimension_analysis']
    for dimension, stats in dimension_analysis.items():
        print(f"   {dimension}:")
        print(f"      平均分: {stats['average']:.2f}")
        print(f"      最高分: {stats['max']:.2f}")
        print(f"      最低分: {stats['min']:.2f}")
    
    print(f"\n🏆 顶级论文:")
    for i, top_paper in enumerate(report['top_papers'][:3], 1):
        print(f"   {i}. {top_paper['title'][:50]}...")
        print(f"      分数: {top_paper['overall_score']:.2f}")
        print(f"      置信度: {top_paper['confidence']:.2f}")
    
    print(f"\n💡 总体建议:")
    for recommendation in report['recommendations']:
        print(f"   • {recommendation}")


def demo_context_aware_evaluation(assessor, papers):
    """演示上下文相关评估"""
    print(f"\n🎯 上下文相关评估演示")
    
    # 创建模拟主题分析上下文
    context = type('TopicAnalysis', (), {
        'keywords': ['deep learning', 'medical imaging', 'neural network', 'AI'],
        'research_type': ResearchType.EXPERIMENTAL,
        'complexity_level': ResearchComplexity.HIGH
    })()
    
    print(f"\n🔬 研究上下文:")
    print(f"   关键词: {', '.join(context.keywords)}")
    print(f"   研究类型: {context.research_type.value}")
    print(f"   复杂度: {context.complexity_level.value}")
    
    # 对比有无上下文的评估结果
    paper = papers[0]  # 选择第一篇论文
    
    print(f"\n📄 测试论文: {paper.title[:50]}...")
    
    # 无上下文评估
    score_without_context = assessor.evaluate_paper_quality(paper)
    
    # 有上下文评估
    score_with_context = assessor.evaluate_paper_quality(paper, context)
    
    print(f"\n📊 评估结果对比:")
    print(f"   无上下文:")
    print(f"      综合分数: {score_without_context.overall_score:.2f}")
    print(f"      相关性分数: {score_without_context.dimension_scores[QualityDimension.RELEVANCE]:.2f}")
    
    print(f"   有上下文:")
    print(f"      综合分数: {score_with_context.overall_score:.2f}")
    print(f"      相关性分数: {score_with_context.dimension_scores[QualityDimension.RELEVANCE]:.2f}")
    
    # 分析差异
    score_diff = score_with_context.overall_score - score_without_context.overall_score
    relevance_diff = (score_with_context.dimension_scores[QualityDimension.RELEVANCE] - 
                     score_without_context.dimension_scores[QualityDimension.RELEVANCE])
    
    print(f"\n📈 差异分析:")
    print(f"   综合分数提升: {score_diff:+.2f}")
    print(f"   相关性分数提升: {relevance_diff:+.2f}")
    
    if score_diff > 0:
        print(f"   结论: 上下文信息提高了论文的质量评分")
    else:
        print(f"   结论: 上下文信息对该论文评分影响较小")


def demo_quality_dimensions():
    """演示质量维度分析"""
    print(f"\n🔍 质量维度详细分析演示")
    
    config = {'quality_threshold': 0.6, 'citation_weight': 0.3}
    assessor = LiteratureQualityAssessor(config=config)
    
    # 创建一篇高质量论文进行详细分析
    paper = Paper(
        title="Breakthrough Advances in Quantum Machine Learning: A Comprehensive Framework",
        authors=["Prof. Alice Quantum", "Dr. Bob Neural", "Dr. Carol Computing"],
        abstract="This groundbreaking research presents a novel and comprehensive framework for quantum machine learning that revolutionizes the field. Our innovative approach combines quantum computing principles with advanced machine learning techniques, achieving unprecedented performance improvements. Extensive experimental validation on multiple datasets demonstrates significant statistical improvements (p < 0.001). The methodology is rigorously designed with proper controls and comprehensive evaluation metrics.",
        keywords=["quantum computing", "machine learning", "quantum algorithms", "optimization"],
        publication_year=2023,
        journal="Nature",
        citation_count=234,
        doi="10.1038/nature12345",
        venue_type="journal"
    )
    
    print(f"\n📄 分析论文: {paper.title}")
    
    # 详细评估
    quality_score = assessor.evaluate_paper_quality(paper)
    
    print(f"\n📊 详细维度分析:")
    
    dimension_names = {
        QualityDimension.RELEVANCE: "相关性",
        QualityDimension.NOVELTY: "新颖性", 
        QualityDimension.METHODOLOGY: "方法论",
        QualityDimension.IMPACT: "影响力",
        QualityDimension.CREDIBILITY: "可信度",
        QualityDimension.CLARITY: "清晰度"
    }
    
    for dimension, score in quality_score.dimension_scores.items():
        name = dimension_names.get(dimension, dimension.value)
        weight = assessor.dimension_weights[dimension]
        
        # 生成评分条形图
        bar_length = int(score * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        
        print(f"   {name:8} [{bar}] {score:.2f} (权重: {weight:.1%})")
    
    print(f"\n🎯 综合评估:")
    print(f"   综合分数: {quality_score.overall_score:.2f}")
    print(f"   置信度: {quality_score.confidence:.2f}")
    
    # 质量等级判定
    if quality_score.overall_score >= 0.9:
        level = "卓越"
        emoji = "🌟"
    elif quality_score.overall_score >= 0.8:
        level = "优秀"
        emoji = "⭐"
    elif quality_score.overall_score >= 0.7:
        level = "良好"
        emoji = "👍"
    elif quality_score.overall_score >= 0.6:
        level = "合格"
        emoji = "✅"
    else:
        level = "需改进"
        emoji = "⚠️"
    
    print(f"   质量等级: {emoji} {level}")


if __name__ == "__main__":
    # 运行主演示
    demo_literature_quality_assessment()
    
    # 运行维度分析演示
    demo_quality_dimensions()