"""
实验设计组件测试
"""

import unittest
import sys
import os
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_automation.core.experiment_designer import ExperimentDesignComponent
from research_automation.models.analysis_models import Paper, ExperimentDesign, FeasibilityScore


class TestExperimentDesignComponent(unittest.TestCase):
    """实验设计组件测试类"""
    
    def setUp(self):
        """设置测试环境"""
        self.designer = ExperimentDesignComponent()
        
        # 创建测试文献分析数据
        self.test_literature_analysis = {
            'papers': [
                {
                    'methodology': 'deep learning with CNN',
                    'parameters': {'learning_rate': 0.001, 'batch_size': 32}
                },
                {
                    'methodology': 'transformer architecture',
                    'parameters': {'learning_rate': 0.0001, 'batch_size': 64}
                }
            ],
            'knowledge_graph': None
        }
        
        # 创建测试约束条件
        self.test_constraints = {
            'max_timeline_weeks': 10,
            'max_budget': 5000.0,
            'max_gpu_hours': 100
        }
    
    def test_generate_experiment_designs(self):
        """测试实验设计生成"""
        designs = self.designer.generate_experiment_designs(
            self.test_literature_analysis,
            "图像分类",
            self.test_constraints
        )
        
        # 验证生成结果
        self.assertIsInstance(designs, list)
        self.assertGreater(len(designs), 0)
        
        for design in designs:
            self.assertIsInstance(design, ExperimentDesign)
            self.assertIsNotNone(design.methodology)
            self.assertIsInstance(design.parameters, dict)
            self.assertIsInstance(design.expected_outcomes, list)
            self.assertGreater(design.timeline_weeks, 0)
        
        print(f"生成了 {len(designs)} 个实验设计方案")
        for i, design in enumerate(designs, 1):
            print(f"方案 {i}: {design.methodology}")
            print(f"  时间线: {design.timeline_weeks} 周")
            print(f"  参数数量: {len(design.parameters)}")
    
    def test_evaluate_feasibility(self):
        """测试可行性评估"""
        # 创建测试实验设计
        test_design = ExperimentDesign(
            methodology="深度学习分类",
            parameters={
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100
            },
            expected_outcomes=["提高分类准确率"],
            timeline_weeks=8,
            success_criteria=["准确率>90%"],
            risk_factors=["数据不足"]
        )
        
        feasibility = self.designer.evaluate_feasibility(test_design, self.test_constraints)
        
        # 验证可行性评估
        self.assertIsInstance(feasibility, FeasibilityScore)
        self.assertGreaterEqual(feasibility.technical_feasibility, 0.0)
        self.assertLessEqual(feasibility.technical_feasibility, 1.0)
        self.assertGreaterEqual(feasibility.resource_feasibility, 0.0)
        self.assertLessEqual(feasibility.resource_feasibility, 1.0)
        
        print(f"可行性评估结果:")
        print(f"- 技术可行性: {feasibility.technical_feasibility:.3f}")
        print(f"- 资源可行性: {feasibility.resource_feasibility:.3f}")
        print(f"- 时间可行性: {feasibility.time_feasibility:.3f}")
        print(f"- 风险评估: {feasibility.risk_assessment:.3f}")
        print(f"- 综合可行性: {feasibility.value:.3f}")
    
    def test_optimize_parameters(self):
        """测试参数优化"""
        # 创建测试实验设计
        original_design = ExperimentDesign(
            methodology="基础CNN",
            parameters={
                'learning_rate': 0.1,  # 过高的学习率
                'batch_size': 8,       # 过小的批次大小
                'epochs': 300          # 过多的训练轮数
            },
            expected_outcomes=["基础分类"],
            timeline_weeks=6
        )
        
        optimized_design = self.designer.optimize_parameters(original_design)
        
        # 验证优化结果
        self.assertIsInstance(optimized_design, ExperimentDesign)
        self.assertNotEqual(original_design.parameters, optimized_design.parameters)
        
        # 检查具体优化
        if 'learning_rate' in optimized_design.parameters:
            self.assertLessEqual(optimized_design.parameters['learning_rate'], 0.01)
        
        print(f"参数优化结果:")
        print(f"原始参数: {original_design.parameters}")
        print(f"优化参数: {optimized_design.parameters}")
        print(f"新增预期结果: {set(optimized_design.expected_outcomes) - set(original_design.expected_outcomes)}")
    
    def test_extract_methodologies(self):
        """测试方法提取"""
        methodologies = self.designer._extract_methodologies(self.test_literature_analysis)
        
        # 验证提取结果
        self.assertIsInstance(methodologies, list)
        self.assertGreater(len(methodologies), 0)
        
        print(f"提取的方法: {methodologies}")
    
    def test_identify_key_parameters(self):
        """测试关键参数识别"""
        parameters = self.designer._identify_key_parameters(self.test_literature_analysis)
        
        # 验证识别结果
        self.assertIsInstance(parameters, dict)
        self.assertIn('learning_rate', parameters)
        self.assertIn('batch_size', parameters)
        
        print(f"识别的关键参数: {list(parameters.keys())}")
    
    def test_create_baseline_design(self):
        """测试基础设计创建"""
        methodologies = ["deep learning", "machine learning"]
        parameters = {
            'learning_rate': [0.001, 0.01],
            'batch_size': [32, 64],
            'epochs': [100, 200]
        }
        
        baseline = self.designer._create_baseline_design("文本分类", methodologies, parameters)
        
        # 验证基础设计
        self.assertIsInstance(baseline, ExperimentDesign)
        self.assertEqual(baseline.methodology, "deep learning")
        self.assertIn('learning_rate', baseline.parameters)
        self.assertGreater(len(baseline.expected_outcomes), 0)
        
        print(f"基础设计:")
        print(f"- 方法: {baseline.methodology}")
        print(f"- 参数: {baseline.parameters}")
        print(f"- 预期结果: {baseline.expected_outcomes}")
    
    def test_create_improved_designs(self):
        """测试改进设计创建"""
        baseline = ExperimentDesign(
            methodology="基础方法",
            parameters={'learning_rate': 0.001, 'batch_size': 32},
            timeline_weeks=6
        )
        
        improved = self.designer._create_improved_designs(baseline, self.test_literature_analysis)
        
        # 验证改进设计
        self.assertIsInstance(improved, list)
        self.assertGreater(len(improved), 0)
        
        for design in improved:
            self.assertIsInstance(design, ExperimentDesign)
            self.assertNotEqual(design.methodology, baseline.methodology)
        
        print(f"改进设计数量: {len(improved)}")
        for i, design in enumerate(improved, 1):
            print(f"改进方案 {i}: {design.methodology}")
    
    def test_create_innovative_designs(self):
        """测试创新设计创建"""
        methodologies = ["deep learning"]
        parameters = {'learning_rate': [0.001]}
        
        innovative = self.designer._create_innovative_designs("AI研究", methodologies, parameters)
        
        # 验证创新设计
        self.assertIsInstance(innovative, list)
        self.assertGreater(len(innovative), 0)
        
        for design in innovative:
            self.assertIsInstance(design, ExperimentDesign)
            self.assertIn("学习", design.methodology)  # 应该包含学习相关的创新方法
        
        print(f"创新设计数量: {len(innovative)}")
        for i, design in enumerate(innovative, 1):
            print(f"创新方案 {i}: {design.methodology}")
    
    def test_apply_constraints(self):
        """测试约束应用"""
        designs = [
            ExperimentDesign(
                methodology="长期实验",
                timeline_weeks=15,  # 超过约束
                parameters={'epochs': 500}
            ),
            ExperimentDesign(
                methodology="正常实验",
                timeline_weeks=8,
                parameters={'epochs': 100}
            )
        ]
        
        constrained = self.designer._apply_constraints(designs, self.test_constraints)
        
        # 验证约束应用
        self.assertEqual(len(constrained), len(designs))
        
        for design in constrained:
            self.assertLessEqual(design.timeline_weeks, self.test_constraints['max_timeline_weeks'])
        
        print(f"约束应用结果:")
        for i, design in enumerate(constrained, 1):
            print(f"设计 {i}: {design.timeline_weeks} 周")


if __name__ == '__main__':
    unittest.main()