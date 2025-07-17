"""
研究计划生成器集成测试

测试研究计划生成的完整功能
"""

import unittest
from datetime import datetime, timedelta

from research_automation.core.research_planner import ResearchPlannerComponent
from research_automation.models.research_models import ResearchComplexity, ResearchType
from research_automation.core.exceptions import ValidationError


class TestResearchPlanGenerator(unittest.TestCase):
    """研究计划生成器测试类"""
    
    def setUp(self):
        """测试前设置"""
        self.config = {
            'llm_model': 'test_model',
            'max_analysis_time': 300
        }
        self.planner = ResearchPlannerComponent(config=self.config)
    
    def test_generate_simple_research_plan(self):
        """测试简单研究计划生成"""
        topic = "A comparative study of machine learning algorithms"
        analysis = self.planner.analyze_topic(topic)
        plan = self.planner.generate_research_plan(analysis)
        
        # 验证计划基本属性
        self.assertTrue(plan.validate())
        self.assertEqual(plan.topic_analysis, analysis)
        self.assertGreater(len(plan.timeline), 0)
        self.assertGreater(len(plan.research_paths), 0)
        self.assertGreater(len(plan.success_metrics), 0)
        
        # 验证时间线
        self.assertGreaterEqual(len(plan.timeline), 4)  # 至少4个阶段
        
        # 验证里程碑顺序
        for i in range(1, len(plan.timeline)):
            self.assertGreater(
                plan.timeline[i].due_date, 
                plan.timeline[i-1].due_date,
                "里程碑时间顺序错误"
            )
        
        # 验证依赖关系
        for i, milestone in enumerate(plan.timeline[1:], 1):
            self.assertIn(plan.timeline[i-1].id, milestone.dependencies)
    
    def test_generate_complex_research_plan(self):
        """测试复杂研究计划生成"""
        topic = "Quantum-enhanced deep reinforcement learning for multi-agent systems optimization"
        analysis = self.planner.analyze_topic(topic)
        plan = self.planner.generate_research_plan(analysis)
        
        # 验证复杂项目特征
        self.assertEqual(analysis.complexity_level, ResearchComplexity.HIGH)
        self.assertGreaterEqual(len(plan.research_paths), 3)  # 复杂项目应有更多路径选择
        self.assertGreater(len(plan.risk_mitigation_strategies), 5)  # 更多风险缓解策略
        
        # 验证预算合理性
        budget = plan.resource_allocation['budget']
        self.assertGreater(budget['total'], 50000)  # 复杂项目预算应该较高
        
        # 验证团队规模
        hr = plan.resource_allocation['human_resources']
        self.assertGreaterEqual(hr['team_size'], 3)  # 复杂项目需要更大团队
    
    def test_research_path_generation(self):
        """测试研究路径生成"""
        topic = "Natural language processing for sentiment analysis"
        analysis = self.planner.analyze_topic(topic)
        plan = self.planner.generate_research_plan(analysis)
        
        # 验证路径数量和类型
        self.assertGreaterEqual(len(plan.research_paths), 2)
        
        path_names = [path.name for path in plan.research_paths]
        self.assertIn("标准研究路径", path_names)
        self.assertIn("保守研究路径", path_names)
        
        # 验证路径属性
        for path in plan.research_paths:
            self.assertTrue(0 <= path.risk_level <= 1)
            self.assertTrue(0 <= path.innovation_potential <= 1)
            self.assertTrue(0 <= path.resource_intensity <= 1)
            self.assertGreater(path.timeline_months, 0)
            self.assertGreater(len(path.expected_outcomes), 0)
    
    def test_resource_allocation(self):
        """测试资源分配"""
        topic = "Computer vision for autonomous driving"
        analysis = self.planner.analyze_topic(topic)
        plan = self.planner.generate_research_plan(analysis)
        
        resource_allocation = plan.resource_allocation
        
        # 验证预算结构
        budget = resource_allocation['budget']
        required_budget_keys = ['personnel', 'equipment_software', 'data_literature', 'overhead', 'total']
        for key in required_budget_keys:
            self.assertIn(key, budget)
            self.assertGreater(budget[key], 0)
        
        # 验证预算合理性
        calculated_total = (budget['personnel'] + budget['equipment_software'] + 
                          budget['data_literature'] + budget['overhead'])
        self.assertAlmostEqual(budget['total'], calculated_total, places=2)
        
        # 验证人力资源分配
        hr = resource_allocation['human_resources']
        self.assertIn('team_size', hr)
        self.assertIn('role_allocation', hr)
        self.assertIn('workload_distribution', hr)
        self.assertGreater(hr['team_size'], 0)
        
        # 验证时间分配
        time_allocation = resource_allocation['time_allocation']
        total_time = sum(time_allocation.values())
        self.assertAlmostEqual(total_time, analysis.estimated_duration, delta=5)
    
    def test_milestone_generation(self):
        """测试里程碑生成"""
        topic = "Blockchain technology for supply chain management"
        analysis = self.planner.analyze_topic(topic)
        plan = self.planner.generate_research_plan(analysis)
        
        # 验证里程碑基本属性
        for milestone in plan.timeline:
            self.assertIsNotNone(milestone.title)
            self.assertIsNotNone(milestone.description)
            self.assertIsInstance(milestone.due_date, datetime)
            self.assertGreater(len(milestone.deliverables), 0)
            self.assertGreater(len(milestone.assigned_roles), 0)
        
        # 验证里程碑时间合理性
        start_date = min(m.due_date for m in plan.timeline)
        end_date = max(m.due_date for m in plan.timeline)
        total_duration = (end_date - start_date).days
        
        # 总时长应该接近估算时长
        self.assertLessEqual(abs(total_duration - analysis.estimated_duration), 30)
    
    def test_success_metrics_definition(self):
        """测试成功指标定义"""
        topic = "Reinforcement learning for game AI"
        analysis = self.planner.analyze_topic(topic)
        plan = self.planner.generate_research_plan(analysis)
        
        # 验证成功指标
        self.assertGreater(len(plan.success_metrics), 3)
        
        # 验证包含基础指标
        metrics_text = ' '.join(plan.success_metrics).lower()
        self.assertIn('完成', metrics_text)
        self.assertIn('质量', metrics_text)
        
        # 验证实验类型特定指标
        if analysis.research_type == ResearchType.EXPERIMENTAL:
            self.assertTrue(any('实验' in metric for metric in plan.success_metrics))
    
    def test_risk_mitigation_strategies(self):
        """测试风险缓解策略"""
        topic = "Federated learning with privacy preservation"
        analysis = self.planner.analyze_topic(topic)
        plan = self.planner.generate_research_plan(analysis)
        
        # 验证风险缓解策略
        self.assertGreater(len(plan.risk_mitigation_strategies), 3)
        
        # 验证策略针对性
        strategies_text = ' '.join(plan.risk_mitigation_strategies).lower()
        
        # 应该包含通用策略
        self.assertIn('评估', strategies_text)
        self.assertIn('检查', strategies_text)
        
        # 复杂项目应该有更多策略
        if analysis.complexity_level in [ResearchComplexity.HIGH, ResearchComplexity.VERY_HIGH]:
            self.assertGreater(len(plan.risk_mitigation_strategies), 6)
    
    def test_path_suggestion_and_ranking(self):
        """测试路径建议和排序"""
        topic = "Edge computing for IoT applications"
        analysis = self.planner.analyze_topic(topic)
        plan = self.planner.generate_research_plan(analysis)
        
        # 获取路径建议
        suggested_paths = self.planner.suggest_research_paths(plan)
        
        # 验证排序正确性
        for i in range(1, len(suggested_paths)):
            current_priority = suggested_paths[i].calculate_priority_score()
            previous_priority = suggested_paths[i-1].calculate_priority_score()
            self.assertGreaterEqual(previous_priority, current_priority)
        
        # 验证路径选择理由
        for path in suggested_paths:
            self.assertTrue(hasattr(path, 'selection_rationale'))
            self.assertIsNotNone(path.selection_rationale)
            self.assertGreater(len(path.selection_rationale), 10)
    
    def test_scope_refinement(self):
        """测试范围细化"""
        topic = "Artificial intelligence for climate change prediction"
        
        # 获取范围细化建议
        suggestions = self.planner.refine_scope(topic)
        
        # 验证建议质量
        self.assertGreater(len(suggestions), 3)
        
        # 验证建议内容
        suggestions_text = ' '.join(suggestions).lower()
        self.assertTrue(any(keyword in suggestions_text 
                          for keyword in ['聚焦', '明确', '定义', '选择', '确定']))
    
    def test_plan_validation(self):
        """测试计划验证"""
        topic = "Cybersecurity for cloud computing"
        analysis = self.planner.analyze_topic(topic)
        plan = self.planner.generate_research_plan(analysis)
        
        # 验证完整计划
        self.assertTrue(plan.validate())
        
        # 验证计划完整性
        self.assertIsNotNone(plan.topic_analysis)
        self.assertGreater(len(plan.timeline), 0)
        self.assertGreater(len(plan.research_paths), 0)
        self.assertGreater(len(plan.success_metrics), 0)
        self.assertIsNotNone(plan.resource_allocation)
    
    def test_different_research_types(self):
        """测试不同研究类型的计划生成"""
        test_cases = [
            ("A systematic review of deep learning methods", ResearchType.SURVEY),
            ("Theoretical analysis of algorithm complexity", ResearchType.THEORETICAL),
            ("Experimental evaluation of machine learning models", ResearchType.EXPERIMENTAL)
        ]
        
        for topic, expected_type in test_cases:
            with self.subTest(topic=topic):
                analysis = self.planner.analyze_topic(topic)
                plan = self.planner.generate_research_plan(analysis)
                
                # 验证研究类型识别
                self.assertEqual(analysis.research_type, expected_type)
                
                # 验证计划适应性
                self.assertTrue(plan.validate())
                
                # 验证类型特定的调整
                if expected_type == ResearchType.SURVEY:
                    # 调研类应该有更长的文献综述阶段
                    lit_review_milestone = next(
                        (m for m in plan.timeline if '文献综述' in m.title), None
                    )
                    self.assertIsNotNone(lit_review_milestone)
                elif expected_type == ResearchType.THEORETICAL:
                    # 理论研究应该有理论建模阶段
                    theory_milestone = next(
                        (m for m in plan.timeline if '理论' in m.title or '建模' in m.title), None
                    )
                    self.assertIsNotNone(theory_milestone)


if __name__ == '__main__':
    unittest.main()