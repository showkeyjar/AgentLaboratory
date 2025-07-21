"""
实验方案生成器

基于文献分析结果自动生成多种实验设计方案
"""

import re
import random
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging

from ..models.analysis_models import Paper, ExperimentDesign, FeasibilityScore
from ..models.research_models import ResearchPlan, TopicAnalysis
from ..models.base_models import BaseModel
from .base_component import BaseComponent

logger = logging.getLogger(__name__)


class ExperimentDesignComponent(BaseComponent):
    """智能实验设计组件"""
    
    def get_required_configs(self) -> List[str]:
        """获取必需的配置项"""
        return []  # 实验设计组件不需要特殊配置
    
    def _setup_component(self):
        """设置组件特定的初始化逻辑"""
        self.logger.info("实验设计组件初始化完成")
    
    def generate_experiment_designs(self, 
                                   literature_analysis: Dict[str, Any], 
                                   research_topic: str,
                                   constraints: Optional[Dict[str, Any]] = None) -> List[ExperimentDesign]:
        """
        基于文献分析生成多个实验设计方案
        
        Args:
            literature_analysis: 文献分析结果
            research_topic: 研究主题
            constraints: 资源约束条件
            
        Returns:
            实验设计方案列表
        """
        try:
            self.logger.info(f"开始为主题 '{research_topic}' 生成实验设计方案")
            
            # 1. 分析文献中的实验方法
            methodologies = self._extract_methodologies(literature_analysis)
            
            # 2. 识别关键参数和变量
            parameters = self._identify_key_parameters(literature_analysis)
            
            # 3. 生成多种实验设计
            designs = []
            
            # 基础实验设计
            baseline_design = self._create_baseline_design(research_topic, methodologies, parameters)
            designs.append(baseline_design)
            
            # 改进型实验设计
            improved_designs = self._create_improved_designs(baseline_design, literature_analysis)
            designs.extend(improved_designs)
            
            # 4. 应用约束条件
            if constraints:
                designs = self._apply_constraints(designs, constraints)
            
            self.logger.info(f"生成了 {len(designs)} 个实验设计方案")
            return designs
            
        except Exception as e:
            self.logger.error(f"生成实验设计时发生错误: {str(e)}")
            raise
    
    def evaluate_feasibility(self, design: ExperimentDesign, 
                           constraints: Optional[Dict[str, Any]] = None) -> FeasibilityScore:
        """
        评估实验设计的可行性
        
        Args:
            design: 实验设计
            constraints: 约束条件
            
        Returns:
            可行性评分
        """
        try:
            # 技术可行性评估
            technical_feasibility = self._assess_technical_feasibility(design)
            
            # 资源可行性评估
            resource_feasibility = self._assess_resource_feasibility(design, constraints)
            
            # 时间可行性评估
            time_feasibility = self._assess_time_feasibility(design)
            
            # 风险评估
            risk_assessment = self._assess_risks(design)
            
            # 创建可行性评分
            feasibility_score = FeasibilityScore(
                technical_feasibility=technical_feasibility,
                resource_feasibility=resource_feasibility,
                time_feasibility=time_feasibility,
                risk_assessment=risk_assessment
            )
            
            # 计算综合可行性
            feasibility_score.calculate_overall_feasibility()
            
            return feasibility_score
            
        except Exception as e:
            self.logger.error(f"评估可行性时发生错误: {str(e)}")
            raise
    
    def optimize_parameters(self, design: ExperimentDesign) -> ExperimentDesign:
        """
        优化实验参数
        
        Args:
            design: 原始实验设计
            
        Returns:
            优化后的实验设计
        """
        try:
            self.logger.info("开始优化实验参数")
            
            # 创建优化后的设计副本
            optimized_design = ExperimentDesign(
                methodology=design.methodology,
                parameters=design.parameters.copy(),
                expected_outcomes=design.expected_outcomes.copy(),
                feasibility_score=design.feasibility_score,
                resource_requirements=design.resource_requirements.copy(),
                timeline_weeks=design.timeline_weeks,
                success_criteria=design.success_criteria.copy(),
                risk_factors=design.risk_factors.copy(),
                alternative_designs=design.alternative_designs.copy()
            )
            
            # 1. 优化模型参数
            optimized_design.parameters = self._optimize_model_parameters(optimized_design.parameters)
            
            # 2. 更新预期结果
            optimized_design.expected_outcomes = self._update_expected_outcomes(optimized_design)
            
            self.logger.info("参数优化完成")
            return optimized_design
            
        except Exception as e:
            self.logger.error(f"优化参数时发生错误: {str(e)}")
            raise    

    def _extract_methodologies(self, literature_analysis: Dict[str, Any]) -> List[str]:
        """从文献分析中提取实验方法"""
        methodologies = []
        
        # 从论文中提取方法
        if 'papers' in literature_analysis:
            for paper_data in literature_analysis['papers']:
                if isinstance(paper_data, dict):
                    methodology = paper_data.get('methodology', '')
                elif hasattr(paper_data, 'methodology'):
                    methodology = paper_data.methodology
                else:
                    continue
                
                if methodology and methodology not in methodologies:
                    methodologies.append(methodology)
        
        # 添加常见的机器学习方法
        common_methods = [
            "supervised learning",
            "deep learning",
            "machine learning",
            "neural networks"
        ]
        
        for method in common_methods:
            if method not in methodologies:
                methodologies.append(method)
        
        return methodologies[:5]  # 限制数量
    
    def _identify_key_parameters(self, literature_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """识别关键参数"""
        parameters = {
            'learning_rate': [0.001, 0.01, 0.1],
            'batch_size': [16, 32, 64],
            'epochs': [50, 100, 200],
            'model_type': ['CNN', 'RNN', 'MLP']
        }
        
        return parameters
    
    def _create_baseline_design(self, topic: str, methodologies: List[str], parameters: Dict[str, Any]) -> ExperimentDesign:
        """创建基础实验设计"""
        # 选择最常见的方法
        methodology = methodologies[0] if methodologies else "supervised learning"
        
        # 基础参数配置
        base_params = {
            'learning_rate': parameters.get('learning_rate', [0.001])[0],
            'batch_size': parameters.get('batch_size', [32])[0],
            'epochs': parameters.get('epochs', [100])[0]
        }
        
        design = ExperimentDesign(
            methodology=methodology,
            parameters=base_params,
            expected_outcomes=[
                f"建立{topic}的基准性能",
                "验证方法的有效性"
            ],
            timeline_weeks=8,
            success_criteria=[
                "模型收敛",
                "达到预期精度"
            ],
            risk_factors=[
                "数据质量问题",
                "模型过拟合"
            ]
        )
        
        return design
    
    def _create_improved_designs(self, baseline: ExperimentDesign, literature_analysis: Dict[str, Any]) -> List[ExperimentDesign]:
        """创建改进型实验设计"""
        improved_designs = []
        
        # 改进1: 参数调优版本
        tuned_design = ExperimentDesign(
            methodology=f"优化的{baseline.methodology}",
            parameters={
                **baseline.parameters,
                'learning_rate': baseline.parameters.get('learning_rate', 0.001) * 0.1,
                'regularization': 'dropout'
            },
            expected_outcomes=[
                "提高模型性能",
                "减少过拟合"
            ],
            timeline_weeks=baseline.timeline_weeks + 2,
            success_criteria=baseline.success_criteria + ["性能提升10%以上"],
            risk_factors=baseline.risk_factors + ["调参时间过长"]
        )
        improved_designs.append(tuned_design)
        
        return improved_designs
    
    def _apply_constraints(self, designs: List[ExperimentDesign], constraints: Dict[str, Any]) -> List[ExperimentDesign]:
        """应用约束条件"""
        filtered_designs = []
        
        for design in designs:
            # 检查时间约束
            if 'max_timeline_weeks' in constraints:
                if design.timeline_weeks > constraints['max_timeline_weeks']:
                    design.timeline_weeks = constraints['max_timeline_weeks']
                    design.risk_factors.append("时间压缩可能影响质量")
            
            filtered_designs.append(design)
        
        return filtered_designs
    
    def _assess_technical_feasibility(self, design: ExperimentDesign) -> float:
        """评估技术可行性"""
        score = 1.0
        
        # 检查参数合理性
        if 'learning_rate' in design.parameters:
            lr = design.parameters['learning_rate']
            if lr > 0.1 or lr < 0.0001:
                score -= 0.1
        
        return max(0.0, score)
    
    def _assess_resource_feasibility(self, design: ExperimentDesign, constraints: Optional[Dict[str, Any]]) -> float:
        """评估资源可行性"""
        score = 1.0
        
        if constraints:
            # 检查预算约束
            if 'max_budget' in constraints:
                estimated_cost = design.estimate_cost()
                if estimated_cost > constraints['max_budget']:
                    score -= 0.3
        
        return max(0.0, score)
    
    def _assess_time_feasibility(self, design: ExperimentDesign) -> float:
        """评估时间可行性"""
        score = 1.0
        
        # 基于时间线合理性
        if design.timeline_weeks > 20:
            score -= 0.2
        elif design.timeline_weeks < 4:
            score -= 0.3
        
        return max(0.0, score)
    
    def _assess_risks(self, design: ExperimentDesign) -> float:
        """评估风险"""
        risk_score = 0.0
        
        # 基于风险因素数量
        risk_score += len(design.risk_factors) * 0.1
        
        return min(1.0, risk_score)
    
    def _optimize_model_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """优化模型参数"""
        optimized = parameters.copy()
        
        # 优化学习率
        if 'learning_rate' in optimized:
            lr = optimized['learning_rate']
            if lr > 0.01:
                optimized['learning_rate'] = 0.01
            elif lr < 0.0001:
                optimized['learning_rate'] = 0.0001
        
        # 添加正则化
        if 'regularization' not in optimized:
            optimized['regularization'] = 'dropout'
        
        return optimized
    
    def _update_expected_outcomes(self, design: ExperimentDesign) -> List[str]:
        """更新预期结果"""
        outcomes = design.expected_outcomes.copy()
        
        # 基于优化添加新的预期结果
        if 'regularization' in design.parameters:
            outcomes.append("避免过拟合")
        
        return outcomes