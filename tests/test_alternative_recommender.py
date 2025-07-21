"""
替代方案推荐系统测试

测试替代方案推荐器的各项功能
"""

import unittest
import sys
import os
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_automation.core.alternative_recommender import (
    AlternativeRecommender, ResourceConstraint, AlternativeDesign, RecommendationReport,
    SimplificationStrategy, AlternativeType
)
from research_automation.models.analysis_models import ExperimentDesign


class TestAlternativeRecommender(unittest.TestCase):
    """替代方案推荐器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.recommender = AlternativeRecommender()
        
        # 创建测试用的实验设计
        self.test_design = ExperimentDesign(
            methodology="深度学习图像分类",
            parameters={
                'learning_rate': 0.001,
                'batch_size': 64,
                'epochs': 200,
                'model_depth': 10,
                'optimizer': 'adam',
                'regularization': 'dropout'
            },
            expected_outcomes=[
                "实现95%以上的分类准确率",
                "模型在验证集上表现稳定",
                "推理速度满足实时要求",
                "模型具有良好的泛化能力"
            ],
            timeline_weeks=16,
            success_criteria=[
                "训练收敛",
                "验证准确率>95%",
                "推理时间<100ms"
            ],
            risk_factors=[
                "数据质量问题",
                "过拟合风险",
                "计算资源不足"
            ]
        )
        
        # 创建测试用的资源约束
        self.test_constraints = [
            ResourceConstraint(
                constraint_type="budget",
                max_value=5000.0,
                current_usage=6000.0,  # 超出预算
                unit="元",
                priority=1.0
            ),
            ResourceConstraint(
                constraint_type="time",
                max_value=10.0,
                current_usage=16.0,  # 超出时间
                unit="周",
                priority=0.9
            ),
            ResourceConstraint(
                constraint_type="computational",
                max_value=100.0,
                current_usage=80.0,  # 接近上限
                unit="GPU小时",
                priority=0.8
            )
        ]
    
    def test_resource_constraint_methods(self):
        """测试资源约束的方法"""
        constraint = self.test_constraints[0]  # 预算约束
        
        # 测试利用率计算
        utilization = constraint.get_utilization_rate()
        self.assertEqual(utilization, 1.2)  # 6000/5000 = 1.2
        
        # 测试约束违反检查
        self.assertTrue(constraint.is_violated())
        
        # 测试可用容量
        available = constraint.get_available_capacity()
        self.assertEqual(available, 0.0)  # max(0, 5000-6000) = 0
        
        print(f"资源约束测试:")
        print(f"- 利用率: {utilization:.2f}")
        print(f"- 是否违反: {constraint.is_violated()}")
        print(f"- 可用容量: {available}")
    
    def test_analyze_constraints(self):
        """测试约束分析"""
        analysis = self.recommender.analyze_constraints(self.test_design, self.test_constraints)
        
        # 验证分析结果结构
        self.assertIn('violated_constraints', analysis)
        self.assertIn('critical_constraints', analysis)
        self.assertIn('constraint_severity', analysis)
        self.assertIn('bottleneck_resources', analysis)
        self.assertIn('optimization_potential', analysis)
        
        # 验证违反的约束
        violated = analysis['violated_constraints']
        self.assertEqual(len(violated), 2)  # 预算和时间都违反了
        
        # 验证关键约束
        critical = analysis['critical_constraints']
        self.assertGreaterEqual(len(critical), 1)  # 至少有一个关键约束
        
        # 验证瓶颈资源
        bottlenecks = analysis['bottleneck_resources']
        self.assertGreater(len(bottlenecks), 0)
        
        print(f"\\n约束分析结果:")
        print(f"- 违反的约束: {len(violated)}")
        print(f"- 关键约束: {len(critical)}")
        print(f"- 瓶颈资源: {bottlenecks}")
        print(f"- 约束严重程度: {analysis['constraint_severity']}")
    
    def test_generate_alternatives(self):
        """测试替代方案生成"""
        alternatives = self.recommender.generate_alternatives(self.test_design, self.test_constraints)
        
        # 验证生成了替代方案
        self.assertGreater(len(alternatives), 0)
        
        # 验证不同类型的方案都有生成
        alternative_types = set(alt.alternative_type for alt in alternatives)
        expected_types = {AlternativeType.SIMPLIFIED, AlternativeType.PHASED, 
                         AlternativeType.LIGHTWEIGHT, AlternativeType.BASELINE}
        
        # 至少应该有一些预期的类型
        self.assertTrue(len(alternative_types.intersection(expected_types)) > 0)
        
        print(f"\\n生成的替代方案:")
        for i, alt in enumerate(alternatives, 1):
            print(f"{i}. {alt.alternative_type.value}: {alt.description}")
            print(f"   简化策略: {[s.value for s in alt.simplification_strategies]}")
            print(f"   实现难度: {alt.implementation_difficulty}")
    
    def test_evaluate_alternatives(self):
        """测试替代方案评估"""
        # 先生成替代方案
        alternatives = self.recommender.generate_alternatives(self.test_design, self.test_constraints)
        
        # 评估替代方案
        evaluated_alternatives = self.recommender.evaluate_alternatives(alternatives, self.test_constraints)
        
        # 验证评估结果
        self.assertEqual(len(evaluated_alternatives), len(alternatives))
        
        for alt in evaluated_alternatives:
            # 验证评分在有效范围内
            self.assertGreaterEqual(alt.feasibility_score, 0.0)
            self.assertLessEqual(alt.feasibility_score, 1.0)
            self.assertGreaterEqual(alt.quality_score, 0.0)
            self.assertLessEqual(alt.quality_score, 1.0)
            self.assertGreaterEqual(alt.resource_efficiency, 0.0)
            self.assertLessEqual(alt.resource_efficiency, 1.0)
            
            # 验证有描述信息
            self.assertIsInstance(alt.advantages, list)
            self.assertIsInstance(alt.disadvantages, list)
            self.assertIsInstance(alt.recommendations, list)
        
        print(f"\\n评估后的替代方案:")
        for alt in evaluated_alternatives:
            print(f"- {alt.alternative_type.value}:")
            print(f"  可行性: {alt.feasibility_score:.3f}")
            print(f"  质量: {alt.quality_score:.3f}")
            print(f"  效率: {alt.resource_efficiency:.3f}")
            print(f"  综合评分: {alt.get_overall_score():.3f}")
    
    def test_recommend_alternatives(self):
        """测试替代方案推荐"""
        report = self.recommender.recommend_alternatives(self.test_design, self.test_constraints)
        
        # 验证报告结构
        self.assertIsInstance(report, RecommendationReport)
        self.assertEqual(report.original_design, self.test_design)
        self.assertEqual(report.constraints, self.test_constraints)
        self.assertGreater(len(report.alternatives), 0)
        self.assertIsInstance(report.constraint_analysis, dict)
        self.assertIsInstance(report.recommendation_summary, str)
        
        # 验证最佳方案
        if report.best_alternative:
            self.assertIsInstance(report.best_alternative, AlternativeDesign)
            # 最佳方案应该在替代方案列表中
            self.assertIn(report.best_alternative, report.alternatives)
        
        # 验证前N个最佳方案
        top_alternatives = report.get_top_alternatives(3)
        self.assertLessEqual(len(top_alternatives), 3)
        self.assertLessEqual(len(top_alternatives), len(report.alternatives))
        
        # 验证排序正确性
        if len(top_alternatives) > 1:
            for i in range(len(top_alternatives) - 1):
                self.assertGreaterEqual(
                    top_alternatives[i].get_overall_score(),
                    top_alternatives[i + 1].get_overall_score()
                )
        
        print(f"\\n推荐报告:")
        print(f"- 原始设计: {report.original_design.methodology}")
        print(f"- 约束数量: {len(report.constraints)}")
        print(f"- 替代方案数量: {len(report.alternatives)}")
        print(f"- 推荐摘要: {report.recommendation_summary}")
        
        if report.best_alternative:
            best = report.best_alternative
            print(f"\\n最佳替代方案:")
            print(f"- 类型: {best.alternative_type.value}")
            print(f"- 描述: {best.description}")
            print(f"- 综合评分: {best.get_overall_score():.3f}")
            print(f"- 优势: {best.advantages}")
            print(f"- 建议: {best.recommendations}")
    
    def test_compare_alternatives(self):
        """测试替代方案比较"""
        # 生成和评估替代方案
        alternatives = self.recommender.generate_alternatives(self.test_design, self.test_constraints)
        evaluated_alternatives = self.recommender.evaluate_alternatives(alternatives, self.test_constraints)
        
        # 比较替代方案
        comparison = self.recommender.compare_alternatives(evaluated_alternatives)
        
        # 验证比较结果结构
        self.assertIn('ranking', comparison)
        self.assertIn('performance_comparison', comparison)
        self.assertIn('resource_comparison', comparison)
        self.assertIn('trade_off_analysis', comparison)
        self.assertIn('recommendations', comparison)
        
        # 验证排名
        ranking = comparison['ranking']
        self.assertEqual(len(ranking), len(evaluated_alternatives))
        
        # 验证排名顺序
        for i in range(len(ranking) - 1):
            self.assertGreaterEqual(ranking[i]['overall_score'], ranking[i + 1]['overall_score'])
        
        # 验证性能比较
        perf_comp = comparison['performance_comparison']
        self.assertIn('best_feasibility', perf_comp)
        self.assertIn('best_quality', perf_comp)
        self.assertIn('best_efficiency', perf_comp)
        
        print(f"\\n方案比较结果:")
        print(f"排名前3的方案:")
        for i, rank_info in enumerate(ranking[:3], 1):
            print(f"  {i}. {rank_info['type']} (评分: {rank_info['overall_score']:.3f})")
        
        print(f"\\n性能比较:")
        print(f"- 最佳可行性: {perf_comp['best_feasibility']:.3f}")
        print(f"- 最佳质量: {perf_comp['best_quality']:.3f}")
        print(f"- 最佳效率: {perf_comp['best_efficiency']:.3f}")
        
        print(f"\\n比较建议:")
        for rec in comparison['recommendations']:
            print(f"- {rec}")
    
    def test_alternative_design_methods(self):
        """测试替代设计的方法"""
        # 创建一个测试用的替代设计
        alternative = AlternativeDesign(
            original_design=self.test_design,
            alternative_design=ExperimentDesign(
                methodology="简化的深度学习",
                timeline_weeks=8
            ),
            alternative_type=AlternativeType.SIMPLIFIED,
            feasibility_score=0.8,
            quality_score=0.7,
            resource_efficiency=0.9,
            implementation_difficulty=0.3
        )
        
        # 测试综合评分计算
        overall_score = alternative.get_overall_score()
        self.assertGreaterEqual(overall_score, 0.0)
        self.assertLessEqual(overall_score, 1.0)
        
        print(f"\\n替代设计测试:")
        print(f"- 可行性: {alternative.feasibility_score}")
        print(f"- 质量: {alternative.quality_score}")
        print(f"- 效率: {alternative.resource_efficiency}")
        print(f"- 实现难度: {alternative.implementation_difficulty}")
        print(f"- 综合评分: {overall_score:.3f}")
    
    def test_simplification_strategies(self):
        """测试简化策略"""
        # 测试所有简化策略都被考虑
        all_strategies = list(SimplificationStrategy)
        self.assertGreater(len(all_strategies), 0)
        
        # 生成替代方案并检查策略使用
        alternatives = self.recommender.generate_alternatives(self.test_design, self.test_constraints)
        
        used_strategies = set()
        for alt in alternatives:
            used_strategies.update(alt.simplification_strategies)
        
        # 应该使用了多种策略
        self.assertGreater(len(used_strategies), 0)
        
        print(f"\\n使用的简化策略:")
        for strategy in used_strategies:
            print(f"- {strategy.value}")
    
    def test_resource_constraint_edge_cases(self):
        """测试资源约束的边界情况"""
        # 测试零约束
        zero_constraint = ResourceConstraint(
            constraint_type="test",
            max_value=0.0,
            current_usage=0.0
        )
        self.assertEqual(zero_constraint.get_utilization_rate(), 0.0)
        self.assertFalse(zero_constraint.is_violated())
        
        # 测试负值约束
        negative_constraint = ResourceConstraint(
            constraint_type="test",
            max_value=-10.0,
            current_usage=5.0
        )
        self.assertEqual(negative_constraint.get_utilization_rate(), 0.0)
        self.assertTrue(negative_constraint.is_violated())
        
        print("边界情况测试通过")
    
    def test_empty_alternatives_handling(self):
        """测试空替代方案列表的处理"""
        # 测试空列表比较
        comparison = self.recommender.compare_alternatives([])
        self.assertEqual(comparison, {})
        
        # 测试空设计的约束分析
        empty_design = ExperimentDesign()
        analysis = self.recommender.analyze_constraints(empty_design, [])
        self.assertIsInstance(analysis, dict)
        
        print("空列表处理测试通过")
    
    def test_constraint_priority_handling(self):
        """测试约束优先级处理"""
        # 创建不同优先级的约束
        high_priority = ResourceConstraint(
            constraint_type="critical",
            max_value=100.0,
            current_usage=120.0,
            priority=1.0
        )
        
        low_priority = ResourceConstraint(
            constraint_type="optional",
            max_value=100.0,
            current_usage=120.0,
            priority=0.3
        )
        
        constraints = [high_priority, low_priority]
        analysis = self.recommender.analyze_constraints(self.test_design, constraints)
        
        # 验证分析考虑了优先级
        violated = analysis['violated_constraints']
        self.assertEqual(len(violated), 2)
        
        # 高优先级约束应该被识别
        priorities = [v['priority'] for v in violated]
        self.assertIn(1.0, priorities)
        self.assertIn(0.3, priorities)
        
        print(f"\\n优先级处理测试:")
        print(f"- 违反约束数量: {len(violated)}")
        print(f"- 约束优先级: {priorities}")


if __name__ == '__main__':
    unittest.main(verbosity=2)