"""
智能关键词生成器演示

展示关键词提取、扩展和搜索策略生成功能
"""

import sys
sys.path.append('.')

from research_automation.core.keyword_generator import KeywordGeneratorComponent
from research_automation.models.research_models import ResearchType, ResearchComplexity


def demo_keyword_generation():
    """演示关键词生成功能"""
    print("🔍 智能关键词生成器演示")
    print("=" * 50)
    
    # 初始化组件
    config = {
        'max_keywords': 15,
        'similarity_threshold': 0.6
    }
    
    generator = KeywordGeneratorComponent(config=config)
    
    # 演示案例
    test_cases = [
        {
            "name": "机器学习应用",
            "topic": "Machine learning applications in natural language processing",
            "description": "经典的机器学习主题"
        },
        {
            "name": "量子计算研究",
            "topic": "Quantum computing algorithms for optimization problems",
            "description": "前沿的量子计算主题"
        },
        {
            "name": "医疗AI应用",
            "topic": "Deep learning approaches for medical image analysis and diagnosis",
            "description": "跨学科的医疗AI主题"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📋 案例 {i}: {case['name']}")
        print(f"主题: {case['topic']}")
        print(f"描述: {case['description']}")
        print("-" * 50)
        
        try:
            # 生成关键词分析
            result = generator.generate_keywords(case['topic'])
            
            # 显示结果
            display_keyword_analysis(result)
            
            # 演示搜索查询优化
            demo_search_optimization(generator, result.primary_keywords[:3])
            
        except Exception as e:
            print(f"❌ 生成失败: {str(e)}")
    
    # 演示上下文扩展
    demo_context_expansion(generator)
    
    print(f"\n🎉 演示完成!")
    print("=" * 50)


def display_keyword_analysis(result):
    """显示关键词分析结果"""
    print(f"\n✅ 关键词分析完成")
    
    # 主要关键词
    if result.primary_keywords:
        print(f"\n🎯 主要关键词 ({len(result.primary_keywords)}个):")
        for i, keyword in enumerate(result.primary_keywords[:8], 1):
            relevance = result.relevance_scores.get(keyword, 0)
            print(f"   {i}. {keyword} (相关性: {relevance:.2f})")
    
    # 次要关键词
    if result.secondary_keywords:
        print(f"\n🔄 次要关键词 ({len(result.secondary_keywords)}个):")
        for keyword in result.secondary_keywords[:5]:
            print(f"   • {keyword}")
    
    # 领域关键词
    if result.domain_keywords:
        print(f"\n🏷️  领域关键词 ({len(result.domain_keywords)}个):")
        for keyword in result.domain_keywords[:5]:
            print(f"   • {keyword}")
    
    # 方法关键词
    if result.method_keywords:
        print(f"\n⚙️  方法关键词 ({len(result.method_keywords)}个):")
        for keyword in result.method_keywords[:5]:
            print(f"   • {keyword}")
    
    # 扩展关键词
    if result.expanded_keywords:
        print(f"\n📈 扩展关键词 ({len(result.expanded_keywords)}个):")
        for keyword in result.expanded_keywords[:5]:
            print(f"   • {keyword}")
    
    # 关键词组合
    if result.keyword_combinations:
        print(f"\n🔗 关键词组合 ({len(result.keyword_combinations)}个):")
        for combo in result.keyword_combinations[:3]:
            print(f"   • {combo}")
    
    # 搜索策略
    if result.search_strategies:
        print(f"\n🎯 搜索策略 ({len(result.search_strategies)}个):")
        for i, strategy in enumerate(result.search_strategies[:3], 1):
            print(f"   {i}. {strategy['name']}")
            print(f"      描述: {strategy['description']}")
            print(f"      查询: {strategy['search_query']}")
            print(f"      预期结果: {strategy.get('expected_results', 'N/A')}")
            print()


def demo_search_optimization(generator, keywords):
    """演示搜索查询优化"""
    print(f"\n🔧 搜索查询优化演示")
    
    # 不同的搜索上下文
    contexts = [
        {
            'name': '广泛搜索',
            'context': {'search_type': 'broad'},
        },
        {
            'name': '精确搜索',
            'context': {'search_type': 'precise'},
        },
        {
            'name': '平衡搜索',
            'context': {'search_type': 'balanced', 'time_filter': '2020-2023'},
        }
    ]
    
    for ctx in contexts:
        try:
            query = generator.optimize_search_query(keywords, ctx['context'])
            print(f"   {ctx['name']}: {query}")
        except Exception as e:
            print(f"   {ctx['name']}: 优化失败 - {str(e)}")


def demo_context_expansion(generator):
    """演示上下文关键词扩展"""
    print(f"\n🌟 上下文关键词扩展演示")
    
    # 模拟主题分析结果
    class MockTopicAnalysis:
        def __init__(self, research_type, complexity_level, related_fields):
            self.research_type = research_type
            self.complexity_level = complexity_level
            self.related_fields = related_fields
    
    # 测试案例
    base_keywords = ['machine learning', 'classification']
    
    contexts = [
        {
            'name': '实验研究',
            'analysis': MockTopicAnalysis(
                ResearchType.EXPERIMENTAL,
                ResearchComplexity.MEDIUM,
                ['Computer Science']
            )
        },
        {
            'name': '理论研究',
            'analysis': MockTopicAnalysis(
                ResearchType.THEORETICAL,
                ResearchComplexity.HIGH,
                ['Mathematics', 'Computer Science']
            )
        },
        {
            'name': '调研研究',
            'analysis': MockTopicAnalysis(
                ResearchType.SURVEY,
                ResearchComplexity.LOW,
                ['Computer Science']
            )
        }
    ]
    
    for ctx in contexts:
        try:
            expanded = generator.expand_keywords_with_context(
                base_keywords, ctx['analysis']
            )
            
            new_keywords = [kw for kw in expanded if kw not in base_keywords]
            
            print(f"\n   {ctx['name']}:")
            print(f"      原始关键词: {', '.join(base_keywords)}")
            print(f"      新增关键词: {', '.join(new_keywords[:5])}")
            print(f"      总计: {len(expanded)}个关键词")
            
        except Exception as e:
            print(f"   {ctx['name']}: 扩展失败 - {str(e)}")


def demo_keyword_quality_analysis():
    """演示关键词质量分析"""
    print(f"\n📊 关键词质量分析演示")
    
    config = {
        'max_keywords': 10,
        'similarity_threshold': 0.5
    }
    
    generator = KeywordGeneratorComponent(config=config)
    
    # 测试不同质量的主题
    topics = [
        {
            'name': '高质量主题',
            'topic': 'Deep learning neural networks for computer vision image recognition',
            'expected_quality': 'high'
        },
        {
            'name': '中等质量主题',
            'topic': 'Machine learning applications',
            'expected_quality': 'medium'
        },
        {
            'name': '低质量主题',
            'topic': 'Some research about things',
            'expected_quality': 'low'
        }
    ]
    
    for topic_info in topics:
        print(f"\n   {topic_info['name']}: {topic_info['topic']}")
        
        try:
            result = generator.generate_keywords(topic_info['topic'])
            
            # 计算质量指标
            avg_relevance = sum(result.relevance_scores.values()) / len(result.relevance_scores) if result.relevance_scores else 0
            keyword_diversity = len(set(result.primary_keywords + result.secondary_keywords + result.domain_keywords))
            strategy_count = len(result.search_strategies)
            
            print(f"      平均相关性: {avg_relevance:.2f}")
            print(f"      关键词多样性: {keyword_diversity}")
            print(f"      搜索策略数: {strategy_count}")
            
            # 质量评估
            if avg_relevance > 0.7 and keyword_diversity > 15:
                quality = "高"
            elif avg_relevance > 0.5 and keyword_diversity > 10:
                quality = "中"
            else:
                quality = "低"
            
            print(f"      质量评估: {quality}")
            
        except Exception as e:
            print(f"      分析失败: {str(e)}")


if __name__ == "__main__":
    # 运行主演示
    demo_keyword_generation()
    
    # 运行质量分析演示
    demo_keyword_quality_analysis()