"""
可行性评估系统演示脚本

展示如何使用可行性评估器评估实验设计的可行性
"""

import sys
import os
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from research_automation.core.feasibility_assessor import FeasibilityAssessor
from research_automation.models.analysis_models import ExperimentDesign, Paper


def create_sample_designs():
    """创建示例实验设计"""
    designs = []
    
    # 设计1: 深度学习方案
    design1 = ExperimentDesign(
        methodology="深度学习图像分类",
        parameters={
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 200,
            'model_architecture': 'ResNet50',
            'optimizer': 'Adam'
        },
        expected_outcomes=[
            "实现95%以上的图像分类准确率",
            "模型在验证集上表现稳定",
            "推理速度满足实时要求"
        ],
        timeline_weeks=12,
        success_criteria=[
            "训练损失收敛",
            "验证准确率>95%",
            "推理时间<100ms"
        ],
        risk_factors=[
            "数据质量问题",
            "过拟合风险",
            "计算资源不足"
        ]
    )
    designs.append(("深度学习方案", design1))
    
    # 设计2: 传统机器学习方案
    design2 = ExperimentDesign(
        methodology="传统机器学习分类",
        parameters={
            'algorithm': 'Random Forest',
            'n_estimators': 100,
            'max_depth': 10,
            'feature_selection': 'PCA'
        },
        expected_outcomes=[
            "建立基准分类性能",
            "快速原型验证",
            "可解释性分析"
        ],
        timeline_weeks=6,
        success_criteria=[
            "模型训练完成",
            "准确率>85%",
            "特征重要性分析"
        ],
        risk_factors=[
            "特征工程复杂",
            "性能上限较低"
        ]
    )
    designs.append(("传统机器学习方案", design2))
    
    # 设计3: 高风险实验方案
    design3 = ExperimentDesign(
        methodology="创新神经网络架构",
        parameters={
            'learning_rate': 0.1,  # 较高学习率
            'batch_size': 8,       # 小批次
            'epochs': 1000,        # 大量训练轮数
            'custom_architecture': 'Novel-CNN-Transformer',
            'experimental_loss': 'Custom-Loss'
        },
        expected_outcomes=[
            "突破现有性能上限",
            "验证新架构有效性",
            "发表高影响力论文"
        ],
        timeline_weeks=24,
        success_criteria=[
            "新架构收敛",
            "性能超越基准10%",
            "理论分析完成"
        ],
        risk_factors=[
            "架构设计缺陷",
            "训练不稳定",
            "理论基础不足",
            "实现复杂度高"
        ]
    )
    designs.append(("创新架构方案", design3))
    
    # 设计4: 资源受限方案
    design4 = ExperimentDesign(
        methodology="轻量级移动端模型",
        parameters={
            'learning_rate': 0.01,
            'batch_size': 64,
            'epochs': 50,
            'model_size': 'MobileNet-v2',
            'quantization': True
        },
        expected_outcomes=[
            "模型大小<10MB",
            "移动端推理<50ms",
            "准确率>90%"
        ],
        timeline_weeks=8,
        success_criteria=[
            "模型压缩成功",
            "移动端部署成功",
            "性能满足要求"
        ],
        risk_factors=[
            "压缩后性能下降",
            "移动端兼容性问题"
        ]
    )
    designs.append(("轻量级方案", design4))
    
    return designs


def create_constraint_scenarios():
    """创建不同的约束场景"""
    scenarios = []
    
    # 场景1: 充足资源
    scenario1 = {
        'name': '充足资源环境',
        'constraints': {
            'max_timeline_weeks': 20,
            'max_budget': 50000.0,
            'max_gpu_hours': 500,
            'team_size': 3
        }
    }
    scenarios.append(scenario1)
    
    # 场景2: 资源受限
    scenario2 = {
        'name': '资源受限环境',
        'constraints': {
            'max_timeline_weeks': 8,
            'max_budget': 5000.0,
            'max_gpu_hours': 50,
            'team_size': 1
        }
    }
    scenarios.append(scenario2)
    
    # 场景3: 时间紧迫
    scenario3 = {
        'name': '时间紧迫环境',
        'constraints': {
            'max_timeline_weeks': 4,
            'max_budget': 20000.0,
            'max_gpu_hours': 200,
            'team_size': 2
        }
    }
    scenarios.append(scenario3)
    
    return scenarios


def create_literature_context():
    """创建文献背景"""
    papers = [
        Paper(
            title="Deep Residual Learning for Image Recognition",
            methodology="deep learning",
            publication_year=2016,
            citation_count=50000
        ),
        Paper(
            title="MobileNets: Efficient Convolutional Neural Networks",
            methodology="lightweight neural networks",
            publication_year=2017,
            citation_count=8000
        ),
        Paper(
            title="Random Forests for Image Classification",
            methodology="random forest",
            publication_year=2018,
            citation_count=1200
        )
    ]
    return papers


def main():
    """主函数"""
    print("=" * 80)
    print("可行性评估系统演示")
    print("=" * 80)
    
    # 创建可行性评估器
    assessor = FeasibilityAssessor()
    
    # 创建示例数据
    designs = create_sample_designs()
    scenarios = create_constraint_scenarios()
    literature = create_literature_context()
    
    print(f"\\n准备评估 {len(designs)} 个实验设计方案")
    print(f"在 {len(scenarios)} 种不同约束场景下进行评估")
    
    # 1. 基础可行性评估
    print("\\n" + "=" * 60)
    print("1. 基础可行性评估（无约束条件）")
    print("=" * 60)
    
    for name, design in designs:
        print(f"\\n评估方案: {name}")
        print("-" * 40)
        
        report = assessor.assess_feasibility(design, literature_context=literature)
        
        print(f"综合可行性: {report.overall_feasibility:.3f} ({report.get_feasibility_level()})")
        print(f"技术可行性: {report.technical_feasibility:.3f}")
        print(f"资源可行性: {report.resource_feasibility:.3f}")
        print(f"时间可行性: {report.time_feasibility:.3f}")
        print(f"风险评估: {report.risk_assessment:.3f}")
        
        if report.recommendations:
            print("\\n主要建议:")
            for rec in report.recommendations[:2]:
                print(f"  • {rec}")
    
    # 2. 约束条件下的可行性评估
    print("\\n" + "=" * 60)
    print("2. 不同约束条件下的可行性评估")
    print("=" * 60)
    
    # 选择一个代表性设计进行详细分析
    representative_design = designs[0][1]  # 深度学习方案
    
    for scenario in scenarios:
        print(f"\\n约束场景: {scenario['name']}")
        print("-" * 40)
        
        report = assessor.assess_feasibility(
            representative_design, 
            scenario['constraints'],
            literature
        )
        
        print(f"可行性等级: {report.get_feasibility_level()}")
        print(f"综合得分: {report.overall_feasibility:.3f}")
        
        # 显示资源需求
        print("\\n资源需求:")
        for req in report.resource_requirements:
            availability = "✓" if req.availability_score > 0.8 else "⚠" if req.availability_score > 0.5 else "✗"
            print(f"  {availability} {req.description} (可用性: {req.availability_score:.2f})")
        
        # 显示主要风险
        if report.risk_factors:
            print("\\n主要风险:")
            high_risks = [r for r in report.risk_factors if r.get_risk_score() > 0.4]
            for risk in high_risks[:3]:
                print(f"  • {risk.name}: {risk.description}")
    
    # 3. 方案比较分析
    print("\\n" + "=" * 60)
    print("3. 方案比较分析")
    print("=" * 60)
    
    # 在资源受限场景下比较所有方案
    constrained_scenario = scenarios[1]['constraints']  # 资源受限环境
    
    design_list = [design for _, design in designs]
    comparison_results = assessor.compare_feasibility(design_list, constrained_scenario)
    
    print(f"\\n在{scenarios[1]['name']}下的方案排名:")
    print("-" * 40)
    
    for i, (design, report) in enumerate(comparison_results, 1):
        design_name = next(name for name, d in designs if d == design)
        print(f"{i}. {design_name}")
        print(f"   可行性: {report.overall_feasibility:.3f} ({report.get_feasibility_level()})")
        print(f"   优势: ", end="")
        
        # 分析优势
        strengths = []
        if report.technical_feasibility > 0.8:
            strengths.append("技术成熟")
        if report.resource_feasibility > 0.8:
            strengths.append("资源友好")
        if report.time_feasibility > 0.8:
            strengths.append("时间合理")
        if report.risk_assessment < 0.3:
            strengths.append("风险较低")
        
        print(", ".join(strengths) if strengths else "需要改进")
        print()
    
    # 4. 详细风险分析
    print("\\n" + "=" * 60)
    print("4. 详细风险分析")
    print("=" * 60)
    
    # 分析高风险方案
    high_risk_design = designs[2][1]  # 创新架构方案
    risk_report = assessor.assess_feasibility(high_risk_design, constrained_scenario)
    
    print(f"\\n高风险方案分析: {designs[2][0]}")
    print("-" * 40)
    print(f"风险评估得分: {risk_report.risk_assessment:.3f}")
    
    print("\\n识别的风险因素:")
    for risk in risk_report.risk_factors:
        risk_score = risk.get_risk_score()
        risk_level = "🔴" if risk_score > 0.6 else "🟡" if risk_score > 0.3 else "🟢"
        print(f"  {risk_level} {risk.name}")
        print(f"     概率: {risk.probability:.2f}, 影响: {risk.impact:.2f}, 风险分数: {risk_score:.3f}")
        if risk.mitigation_strategies:
            print(f"     缓解策略: {', '.join(risk.mitigation_strategies[:2])}")
        print()
    
    # 5. 改进建议
    print("\\n" + "=" * 60)
    print("5. 改进建议总结")
    print("=" * 60)
    
    print("\\n针对不同方案的改进建议:")
    for name, design in designs:
        report = assessor.assess_feasibility(design, constrained_scenario)
        if report.recommendations:
            print(f"\\n{name}:")
            for rec in report.recommendations:
                print(f"  • {rec}")
        
        if report.alternative_approaches:
            print(f"  替代方案: {', '.join(report.alternative_approaches[:2])}")
    
    print("\\n" + "=" * 80)
    print("演示完成！")
    print("=" * 80)
    
    # 总结
    print("\\n总结:")
    print("• 可行性评估系统能够全面评估实验设计的技术、资源、时间和风险维度")
    print("• 支持在不同约束条件下进行比较分析")
    print("• 提供具体的改进建议和替代方案")
    print("• 帮助研究者做出更明智的实验设计决策")


if __name__ == "__main__":
    main()