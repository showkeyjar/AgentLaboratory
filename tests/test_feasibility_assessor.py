"""
可行性评估系统测试

测试可行性评估器的各项功能
"""

import unittest
import sys
import os
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_automation.core.feasibility_assessor import (
    FeasibilityAssessor, ResourceRequirement, RiskFactor, TechnicalConstraint,
    FeasibilityReport, ResourceType, RiskLevel
)
from research_automation.models.analysis_models import ExperimentDesign, Paper


class TestFeasibilityAssessor(unittest.TestCase):
    """可行性评估器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.assessor = FeasibilityAssessor()
        
        # 创建测试用的实验设计
        self.test_design = ExperimentDesign(
            methodology="deep learning with CNN",
            parameters={
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100,
                'model_type': 'CNN'
            },
            expected_outcomes=[
                "实现图像分类功能",
                "达到90%以上准确率"
            ],
            timeline_weeks=8,
            success_criteria=[
                "模型收敛",
                "验证集准确率>90%"
            ],
            risk_factors=[
                "数据质量问题",
                "过拟合风险"
            ]
        )
        
        # 创建约束条件
        self.test_constraints = {
            'max_timeline_weeks': 12,
            'max_budget': 5000.0,
            'max_gpu_hours': 100
        }
        
        # 创建文献背景
        self.test_literature = [
            Paper(
                title="CNN for Image Classification",
                methodology="convolutional neural networks",
                publication_year=2020
            ),
            Paper(
                title="Deep Learning Applications",
                methodology="deep learning",
                publication_year=2021
            )
        ]
    
    def test_assess_feasibility_basic(self):
        """测试基础可行性评估"""
        report = self.assessor.assess_feasibility(self.test_design)
        
        # 验证报告结构
        self.assertIsInstance(report, FeasibilityReport)
        self.assertGreaterEqual(report.overall_feasibility, 0.0)
        self.assertLessEqual(report.overall_feasibility, 1.0)
        
        # 验证各项评分
        self.assertGreaterEqual(report.technical_feasibility, 0.0)
        self.assertGreaterEqual(report.resource_feasibility, 0.0)
        self.assertGreaterEqual(report.time_feasibility, 0.0)
        self.assertGreaterEqual(report.risk_assessment, 0.0)
        
        print(f"基础可行性评估结果:")
        print(f"- 综合可行性: {report.overall_feasibility:.3f}")
        print(f"- 技术可行性: {report.technical_feasibility:.3f}")
        print(f"- 资源可行性: {report.resource_feasibility:.3f}")
        print(f"- 时间可行性: {report.time_feasibility:.3f}")
        print(f"- 风险评估: {report.risk_assessment:.3f}")
    
    def test_assess_feasibility_with_constraints(self):
        """测试带约束条件的可行性评估"""
        report = self.assessor.assess_feasibility(
            self.test_design, 
            self.test_constraints
        )
        
        # 验证约束条件的影响
        self.assertIsInstance(report, FeasibilityReport)
        self.assertTrue(len(report.resource_requirements) > 0)
        
        # 检查资源需求
        gpu_req = next((req for req in report.resource_requirements 
                       if req.resource_type == ResourceType.COMPUTATIONAL), None)
        self.assertIsNotNone(gpu_req)
        self.assertGreater(gpu_req.amount, 0)
        
        print(f"\\n带约束条件的可行性评估:")
        print(f"- 可行性等级: {report.get_feasibility_level()}")
        print(f"- 资源需求数量: {len(report.resource_requirements)}")
        print(f"- 风险因素数量: {len(report.risk_factors)}")
    
    def test_assess_feasibility_with_literature(self):
        """测试带文献背景的可行性评估"""
        report = self.assessor.assess_feasibility(
            self.test_design,
            self.test_constraints,
            self.test_literature
        )
        
        # 文献背景应该提高技术可行性
        self.assertIsInstance(report, FeasibilityReport)
        self.assertGreater(report.technical_feasibility, 0.5)
        
        print(f"\\n带文献背景的技术可行性: {report.technical_feasibility:.3f}")
    
    def test_compare_feasibility(self):
        """测试多个设计的可行性比较"""
        # 创建另一个设计
        simple_design = ExperimentDesign(
            methodology="traditional machine learning",
            parameters={
                'algorithm': 'random_forest',
                'n_estimators': 100
            },
            expected_outcomes=["基准性能"],
            timeline_weeks=4
        )
        
        designs = [self.test_design, simple_design]
        results = self.assessor.compare_feasibility(designs, self.test_constraints)
        
        # 验证比较结果
        self.assertEqual(len(results), 2)
        self.assertTrue(all(isinstance(result[1], FeasibilityReport) for result in results))
        
        # 结果应该按可行性排序
        feasibilities = [result[1].overall_feasibility for result in results]
        self.assertEqual(feasibilities, sorted(feasibilities, reverse=True))
        
        print(f"\\n可行性比较结果:")
        for i, (design, report) in enumerate(results, 1):
            print(f"排名 {i}: {design.methodology} - 可行性: {report.overall_feasibility:.3f}")
    
    def test_technical_feasibility_assessment(self):
        """测试技术可行性评估"""
        # 测试参数无效的情况
        invalid_design = ExperimentDesign(
            methodology="deep learning",
            parameters={
                'learning_rate': 2.0,  # 无效学习率
                'batch_size': -10,     # 无效批次大小
                'epochs': 0            # 无效训练轮数
            },
            timeline_weeks=8
        )
        
        report = self.assessor.assess_feasibility(invalid_design)
        
        # 技术可行性应该较低
        self.assertLess(report.technical_feasibility, 0.8)
        self.assertTrue(len(report.technical_constraints) > 0)
        
        print(f"\\n无效参数设计的技术可行性: {report.technical_feasibility:.3f}")
        print(f"技术约束数量: {len(report.technical_constraints)}")
    
    def test_resource_feasibility_assessment(self):
        """测试资源可行性评估"""
        # 测试资源需求超出约束的情况
        high_resource_design = ExperimentDesign(
            methodology="deep learning ensemble",
            parameters={
                'learning_rate': 0.001,
                'batch_size': 8,  # 小批次需要更多时间
                'epochs': 1000    # 大量训练轮数
            },
            timeline_weeks=20
        )
        
        tight_constraints = {
            'max_gpu_hours': 10,  # 很少的GPU时间
            'max_budget': 500     # 很少的预算
        }
        
        report = self.assessor.assess_feasibility(high_resource_design, tight_constraints)
        
        # 资源可行性应该较低
        self.assertLess(report.resource_feasibility, 0.8)
        
        print(f"\\n高资源需求设计的资源可行性: {report.resource_feasibility:.3f}")
    
    def test_time_feasibility_assessment(self):
        """测试时间可行性评估"""
        # 测试时间过短的情况
        short_time_design = ExperimentDesign(
            methodology="complex deep learning",
            parameters={'epochs': 1000},
            timeline_weeks=1  # 时间过短
        )
        
        report = self.assessor.assess_feasibility(short_time_design)
        
        # 时间可行性应该较低
        self.assertLess(report.time_feasibility, 0.5)
        
        print(f"\\n时间过短设计的时间可行性: {report.time_feasibility:.3f}")
    
    def test_risk_assessment(self):
        """测试风险评估"""
        # 测试高风险设计
        risky_design = ExperimentDesign(
            methodology="experimental new method",
            parameters={
                'learning_rate': 0.5,  # 高学习率
                'epochs': 2000         # 大量训练轮数
            },
            timeline_weeks=2  # 时间紧张
        )
        
        report = self.assessor.assess_feasibility(risky_design)
        
        # 风险评估应该较高
        self.assertGreater(report.risk_assessment, 0.3)
        self.assertTrue(len(report.risk_factors) > 0)
        
        print(f"\\n高风险设计的风险评估: {report.risk_assessment:.3f}")
        print(f"识别的风险因素:")
        for risk in report.risk_factors:
            print(f"- {risk.name}: 概率={risk.probability:.2f}, 影响={risk.impact:.2f}")
    
    def test_recommendations_generation(self):
        """测试建议生成"""
        # 创建一个有问题的设计
        problematic_design = ExperimentDesign(
            methodology="untested new approach",
            parameters={
                'learning_rate': 1.5,
                'batch_size': 2048,
                'epochs': 5000
            },
            timeline_weeks=1
        )
        
        report = self.assessor.assess_feasibility(problematic_design)
        
        # 应该生成建议
        self.assertTrue(len(report.recommendations) > 0)
        self.assertTrue(len(report.alternative_approaches) > 0)
        
        print(f"\\n问题设计的改进建议:")
        for rec in report.recommendations:
            print(f"- {rec}")
        
        print(f"\\n替代方案:")
        for alt in report.alternative_approaches:
            print(f"- {alt}")
    
    def test_feasibility_report_methods(self):
        """测试可行性报告的方法"""
        report = self.assessor.assess_feasibility(self.test_design)
        
        # 测试可行性等级
        level = report.get_feasibility_level()
        self.assertIn(level, ["高度可行", "基本可行", "可行性较低", "不可行"])
        
        print(f"\\n可行性等级: {level}")
    
    def test_resource_requirement_creation(self):
        """测试资源需求创建"""
        req = ResourceRequirement(
            resource_type=ResourceType.COMPUTATIONAL,
            amount=50.0,
            unit="GPU小时",
            description="测试资源需求",
            criticality=0.8
        )
        
        self.assertEqual(req.resource_type, ResourceType.COMPUTATIONAL)
        self.assertEqual(req.amount, 50.0)
        self.assertEqual(req.criticality, 0.8)
    
    def test_risk_factor_creation(self):
        """测试风险因素创建"""
        risk = RiskFactor(
            name="测试风险",
            description="这是一个测试风险",
            probability=0.6,
            impact=0.8,
            risk_level=RiskLevel.HIGH
        )
        
        risk_score = risk.get_risk_score()
        self.assertEqual(risk_score, 0.6 * 0.8)
        self.assertEqual(risk.risk_level, RiskLevel.HIGH)


if __name__ == '__main__':
    unittest.main(verbosity=2)