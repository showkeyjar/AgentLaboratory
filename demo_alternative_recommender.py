"""
替代方案推荐系统演示脚本

展示如何使用替代方案推荐器在资源约束下生成和评估替代实验方案
"""

import sys
import os
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from research_automation.core.alternative_recommender import (
    AlternativeRecommender, ResourceConstraint, AlternativeType, SimplificationStrategy
)
from research_automation.models.analysis_models import ExperimentDesign


def create_complex_experiment_design():
    """创建一个复杂的实验设计"""
    return ExperimentDesign(
        methodology="多模态深度学习图像-文本分类",
        parameters={
            'learning_rate': 0.0001,
            'batch_size': 128,
            'epochs': 500,
            'model_depth': 18,
            'attention_heads': 16,
            'hidden_size': 1024,
            'dropout_rate': 0.3,
            'optimizer': 'adamw',
            'weight_decay': 0.01,
            'warmup_steps': 1000,
            'gradient_clip': 1.0,
            'label_smoothing': 0.1
        },
        expected_outcomes=[
            "实现98%以上的多模态分类准确率",
            "模型在多个数据集上表现稳定",
            "推理速度满足实时要求",
            "模型具有良好的跨域泛化能力",
            "支持零样本学习",
            "可解释性分析完整"
        ],
        timeline_weeks=24,
        success_criteria=[
            "训练收敛且稳定",
            "验证准确率>98%",
            "推理时间<200ms",
            "内存使用<8GB",
            "支持批量处理"
        ],
        risk_factors=[
            "多模态数据对齐困难",
            "模型复杂度过高",
            "计算资源需求巨大",
            "训练不稳定",
            "过拟合风险高"
        ]
    )


def create_resource_constraints_scenarios():
    """创建不同的资源约束场景"""
    scenarios = []
    
    # 场景1: 严重资源受限
    scenario1 = {
        'name': '严重资源受限环境',
        'description': '预算紧张、时间紧迫、计算资源有限的典型学术环境',
        'constraints': [
            ResourceConstraint(
                constraint_type="budget",
                max_value=3000.0,
                current_usage=15000.0,  # 严重超出预算
                unit="元",
                priority=1.0
            ),
            ResourceConstraint(
                constraint_type="time",
                max_value=8.0,
                current_usage=24.0,  # 时间严重不足
                unit="周",
                priority=0.9
            ),
            ResourceConstraint(
                constraint_type="computational",
                max_value=50.0,
                current_usage=200.0,  # GPU时间严重不足
                unit="GPU小时",
                priority=0.8
            ),
            ResourceConstraint(
                constraint_type="human",
                max_value=1.0,
                current_usage=3.0,  # 人力资源不足
                unit="人",
                priority=0.7
            )
        ]
    }
    scenarios.append(scenario1)
    
    # 场景2: 中等资源约束
    scenario2 = {
        'name': '中等资源约束环境',
        'description': '有一定资源但仍需优化的工业研发环境',
        'constraints': [
            ResourceConstraint(
                constraint_type="budget",
                max_value=10000.0,
                current_usage=15000.0,  # 超出预算50%
                unit="元",
                priority=1.0
            ),
            ResourceConstraint(
                constraint_type="time",
                max_value=16.0,
                current_usage=24.0,  # 时间超出50%
                unit="周",
                priority=0.8
            ),
            ResourceConstraint(
                constraint_type="computational",
                max_value=150.0,
                current_usage=200.0,  # GPU时间略微不足
                unit="GPU小时",
                priority=0.9
            )
        ]
    }
    scenarios.append(scenario2)
    
    # 场景3: 时间优先约束
    scenario3 = {
        'name': '时间优先约束环境',
        'description': '需要快速出结果的竞赛或紧急项目环境',
        'constraints': [
            ResourceConstraint(
                constraint_type="time",
                max_value=4.0,
                current_usage=24.0,  # 时间极度紧张
                unit="周",
                priority=1.0
            ),
            ResourceConstraint(
                constraint_type="budget",
                max_value=20000.0,
                current_usage=15000.0,  # 预算充足
                unit="元",
                priority=0.3
            ),
            ResourceConstraint(
                constraint_type="computational",
                max_value=300.0,
                current_usage=200.0,  # 计算资源充足
                unit="GPU小时",
                priority=0.5
            )
        ]
    }
    scenarios.append(scenario3)
    
    return scenarios


def demonstrate_constraint_analysis():
    """演示约束分析功能"""
    print("=" * 80)
    print("资源约束分析演示")
    print("=" * 80)
    
    recommender = AlternativeRecommender()
    design = create_complex_experiment_design()
    scenarios = create_resource_constraints_scenarios()
    
    for scenario in scenarios:
        print(f"\\n📊 {scenario['name']}")
        print("-" * 60)
        print(f"场景描述: {scenario['description']}")
        
        constraints = scenario['constraints']
        analysis = recommender.analyze_constraints(design, constraints)
        
        print(f"\\n约束分析结果:")
        print(f"• 违反的约束: {len(analysis['violated_constraints'])} 个")
        print(f"• 关键约束: {len(analysis['critical_constraints'])} 个")
        print(f"• 瓶颈资源: {', '.join(analysis['bottleneck_resources'])}")
        
        print(f"\\n约束详情:")
        for constraint in constraints:
            utilization = constraint.get_utilization_rate()
            status = "❌ 违反" if constraint.is_violated() else "⚠️ 紧张" if utilization > 0.8 else "✅ 正常"
            print(f"  {status} {constraint.constraint_type}: {utilization:.1%} 利用率 (优先级: {constraint.priority})")
        
        print(f"\\n优化潜力:")
        for resource, potential in analysis['optimization_potential'].items():
            if potential > 0:
                print(f"  🎯 {resource}: {potential:.1%} 优化空间")


def demonstrate_alternative_generation():
    """演示替代方案生成"""
    print("\\n" + "=" * 80)
    print("替代方案生成演示")
    print("=" * 80)
    
    recommender = AlternativeRecommender()
    design = create_complex_experiment_design()
    
    # 使用严重资源受限场景
    constraints = create_resource_constraints_scenarios()[0]['constraints']
    
    print(f"\\n🔬 原始实验设计:")
    print(f"方法: {design.methodology}")
    print(f"参数数量: {len(design.parameters)}")
    print(f"预期结果: {len(design.expected_outcomes)} 个")
    print(f"时间线: {design.timeline_weeks} 周")
    print(f"风险因素: {len(design.risk_factors)} 个")
    
    # 生成替代方案
    alternatives = recommender.generate_alternatives(design, constraints)
    
    print(f"\\n🔄 生成的替代方案 ({len(alternatives)} 个):")
    
    # 按类型分组显示
    type_groups = {}
    for alt in alternatives:
        alt_type = alt.alternative_type
        if alt_type not in type_groups:
            type_groups[alt_type] = []
        type_groups[alt_type].append(alt)
    
    for alt_type, alts in type_groups.items():
        print(f"\\n📋 {alt_type.value.upper()} 方案 ({len(alts)} 个):")
        for i, alt in enumerate(alts, 1):
            print(f"  {i}. {alt.description}")
            print(f"     简化策略: {[s.value for s in alt.simplification_strategies]}")
            print(f"     实现难度: {alt.implementation_difficulty:.1f}")
            print(f"     时间线: {alt.alternative_design.timeline_weeks} 周")
            print()


def demonstrate_alternative_evaluation():
    """演示替代方案评估"""
    print("\\n" + "=" * 80)
    print("替代方案评估演示")
    print("=" * 80)
    
    recommender = AlternativeRecommender()
    design = create_complex_experiment_design()
    constraints = create_resource_constraints_scenarios()[0]['constraints']
    
    # 生成并评估替代方案
    alternatives = recommender.generate_alternatives(design, constraints)
    evaluated_alternatives = recommender.evaluate_alternatives(alternatives, constraints)
    
    print(f"\\n📈 评估结果 (按综合评分排序):")
    
    # 按评分排序
    sorted_alternatives = sorted(evaluated_alternatives, 
                               key=lambda alt: alt.get_overall_score(), 
                               reverse=True)
    
    for i, alt in enumerate(sorted_alternatives, 1):
        print(f"\\n{i}. {alt.alternative_type.value.upper()} 方案")
        print(f"   📊 综合评分: {alt.get_overall_score():.3f}")
        print(f"   ✅ 可行性: {alt.feasibility_score:.3f}")
        print(f"   🎯 质量: {alt.quality_score:.3f}")
        print(f"   ⚡ 效率: {alt.resource_efficiency:.3f}")
        print(f"   ⚖️ 性能权衡: {alt.performance_trade_off:.3f}")
        
        if alt.resource_savings:
            print(f"   💰 资源节省:")
            for resource, savings in alt.resource_savings.items():
                if savings > 0:
                    print(f"      • {resource}: {savings:.1%}")
        
        if alt.advantages:
            print(f"   ✨ 优势: {', '.join(alt.advantages)}")
        
        if alt.disadvantages:
            print(f"   ⚠️ 劣势: {', '.join(alt.disadvantages)}")
        
        if alt.recommendations:
            print(f"   💡 建议: {', '.join(alt.recommendations)}")


def demonstrate_comprehensive_recommendation():
    """演示综合推荐功能"""
    print("\\n" + "=" * 80)
    print("综合推荐演示")
    print("=" * 80)
    
    recommender = AlternativeRecommender()
    design = create_complex_experiment_design()
    scenarios = create_resource_constraints_scenarios()
    
    for scenario in scenarios:
        print(f"\\n🎯 {scenario['name']} 推荐方案")
        print("-" * 60)
        
        constraints = scenario['constraints']
        report = recommender.recommend_alternatives(design, constraints)
        
        print(f"推荐摘要: {report.recommendation_summary}")
        
        if report.best_alternative:
            best = report.best_alternative
            print(f"\\n🏆 最佳方案: {best.alternative_type.value}")
            print(f"   描述: {best.description}")
            print(f"   综合评分: {best.get_overall_score():.3f}")
            print(f"   时间线: {best.alternative_design.timeline_weeks} 周")
            print(f"   参数数量: {len(best.alternative_design.parameters)}")
            
            if best.resource_savings:
                total_savings = sum(best.resource_savings.values()) / len(best.resource_savings)
                print(f"   平均资源节省: {total_savings:.1%}")
        
        # 显示前3个推荐方案
        top_alternatives = report.get_top_alternatives(3)
        print(f"\\n📋 前3个推荐方案:")
        for i, alt in enumerate(top_alternatives, 1):
            print(f"   {i}. {alt.alternative_type.value} (评分: {alt.get_overall_score():.3f})")


def demonstrate_alternative_comparison():
    """演示替代方案比较"""
    print("\\n" + "=" * 80)
    print("替代方案比较分析")
    print("=" * 80)
    
    recommender = AlternativeRecommender()
    design = create_complex_experiment_design()
    constraints = create_resource_constraints_scenarios()[1]['constraints']  # 中等约束
    
    # 生成、评估和比较替代方案
    alternatives = recommender.generate_alternatives(design, constraints)
    evaluated_alternatives = recommender.evaluate_alternatives(alternatives, constraints)
    comparison = recommender.compare_alternatives(evaluated_alternatives)
    
    print(f"\\n🏆 方案排名:")
    for rank_info in comparison['ranking']:
        print(f"   {rank_info['rank']}. {rank_info['type']} - 评分: {rank_info['overall_score']:.3f}")
    
    print(f"\\n📊 性能比较:")
    perf_comp = comparison['performance_comparison']
    print(f"   最佳可行性: {perf_comp['best_feasibility']:.3f}")
    print(f"   最佳质量: {perf_comp['best_quality']:.3f}")
    print(f"   最佳效率: {perf_comp['best_efficiency']:.3f}")
    print(f"   平均性能权衡: {perf_comp['average_trade_off']:.3f}")
    
    print(f"\\n💰 资源比较:")
    resource_comp = comparison['resource_comparison']
    for resource_type, stats in resource_comp.items():
        print(f"   {resource_type}:")
        print(f"      最大节省: {stats['max_savings']:.1%}")
        print(f"      平均节省: {stats['average_savings']:.1%}")
    
    print(f"\\n💡 比较建议:")
    for rec in comparison['recommendations']:
        print(f"   • {rec}")


def demonstrate_strategy_analysis():
    """演示简化策略分析"""
    print("\\n" + "=" * 80)
    print("简化策略分析")
    print("=" * 80)
    
    recommender = AlternativeRecommender()
    design = create_complex_experiment_design()
    constraints = create_resource_constraints_scenarios()[0]['constraints']
    
    alternatives = recommender.generate_alternatives(design, constraints)
    evaluated_alternatives = recommender.evaluate_alternatives(alternatives, constraints)
    
    # 按策略分组分析
    strategy_analysis = {}
    for alt in evaluated_alternatives:
        for strategy in alt.simplification_strategies:
            if strategy not in strategy_analysis:
                strategy_analysis[strategy] = {
                    'count': 0,
                    'avg_score': 0.0,
                    'avg_efficiency': 0.0,
                    'avg_trade_off': 0.0,
                    'alternatives': []
                }
            
            strategy_analysis[strategy]['count'] += 1
            strategy_analysis[strategy]['alternatives'].append(alt)
    
    # 计算平均值
    for strategy, data in strategy_analysis.items():
        alts = data['alternatives']
        data['avg_score'] = sum(alt.get_overall_score() for alt in alts) / len(alts)
        data['avg_efficiency'] = sum(alt.resource_efficiency for alt in alts) / len(alts)
        data['avg_trade_off'] = sum(alt.performance_trade_off for alt in alts) / len(alts)
    
    print(f"\\n🔧 简化策略效果分析:")
    sorted_strategies = sorted(strategy_analysis.items(), 
                             key=lambda x: x[1]['avg_score'], 
                             reverse=True)
    
    for strategy, data in sorted_strategies:
        print(f"\\n📋 {strategy.value}:")
        print(f"   使用次数: {data['count']}")
        print(f"   平均评分: {data['avg_score']:.3f}")
        print(f"   平均效率: {data['avg_efficiency']:.3f}")
        print(f"   平均权衡: {data['avg_trade_off']:.3f}")
        
        # 显示使用该策略的最佳方案
        best_alt = max(data['alternatives'], key=lambda alt: alt.get_overall_score())
        print(f"   最佳应用: {best_alt.alternative_type.value} (评分: {best_alt.get_overall_score():.3f})")


def main():
    """主函数"""
    print("🚀 替代方案推荐系统演示")
    print("=" * 80)
    print("本演示将展示在资源约束下如何生成和评估替代实验方案:")
    print("• 资源约束分析")
    print("• 替代方案生成")
    print("• 方案评估和比较")
    print("• 综合推荐")
    print("• 简化策略分析")
    print("=" * 80)
    
    try:
        # 1. 约束分析演示
        demonstrate_constraint_analysis()
        
        # 2. 替代方案生成演示
        demonstrate_alternative_generation()
        
        # 3. 替代方案评估演示
        demonstrate_alternative_evaluation()
        
        # 4. 综合推荐演示
        demonstrate_comprehensive_recommendation()
        
        # 5. 替代方案比较演示
        demonstrate_alternative_comparison()
        
        # 6. 简化策略分析演示
        demonstrate_strategy_analysis()
        
        print("\\n" + "=" * 80)
        print("🎉 演示完成!")
        print("=" * 80)
        
        print("\\n📋 总结:")
        print("• 替代方案推荐系统能够智能分析资源约束")
        print("• 自动生成多种类型的替代实验方案")
        print("• 全面评估方案的可行性、质量和效率")
        print("• 提供详细的比较分析和推荐建议")
        print("• 帮助研究者在资源受限情况下做出最佳选择")
        print("• 支持多种简化策略的灵活组合")
        
    except Exception as e:
        print(f"\\n❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()