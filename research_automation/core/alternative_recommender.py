"""
替代方案推荐系统

在资源约束条件下生成和推荐替代的实验设计方案
"""

import copy
import math
import random
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

from ..models.analysis_models import ExperimentDesign, FeasibilityScore
from ..models.base_models import BaseModel
from .base_component import BaseComponent
from .feasibility_assessor import FeasibilityAssessor

logger = logging.getLogger(__name__)


class SimplificationStrategy(Enum):
    """简化策略枚举"""
    REDUCE_COMPLEXITY = "reduce_complexity"
    REDUCE_SCOPE = "reduce_scope"
    REDUCE_PRECISION = "reduce_precision"
    REDUCE_TIMELINE = "reduce_timeline"
    CHANGE_METHOD = "change_method"
    SPLIT_PHASES = "split_phases"


class AlternativeType(Enum):
    """替代方案类型枚举"""
    SIMPLIFIED = "simplified"
    PHASED = "phased"
    LIGHTWEIGHT = "lightweight"
    BASELINE = "baseline"
    HYBRID = "hybrid"


@dataclass
class ResourceConstraint(BaseModel):
    """资源约束"""
    constraint_type: str = ""  # budget, time, computational, human
    max_value: float = 0.0
    current_usage: float = 0.0
    unit: str = ""
    priority: float = 1.0  # 约束优先级
    
    def get_utilization_rate(self) -> float:
        """获取资源利用率"""
        if self.max_value <= 0:
            return 0.0
        return self.current_usage / self.max_value
    
    def is_violated(self) -> bool:
        """检查约束是否被违反"""
        return self.current_usage > self.max_value
    
    def get_available_capacity(self) -> float:
        """获取可用容量"""
        return max(0.0, self.max_value - self.current_usage)


@dataclass
class AlternativeDesign(BaseModel):
    """替代设计方案"""
    original_design: ExperimentDesign = field(default_factory=ExperimentDesign)
    alternative_design: ExperimentDesign = field(default_factory=ExperimentDesign)
    alternative_type: AlternativeType = AlternativeType.SIMPLIFIED
    simplification_strategies: List[SimplificationStrategy] = field(default_factory=list)
    
    # 质量评估
    feasibility_score: float = 0.0
    quality_score: float = 0.0
    resource_efficiency: float = 0.0
    
    # 对比分析
    performance_trade_off: float = 0.0  # 性能损失
    resource_savings: Dict[str, float] = field(default_factory=dict)
    implementation_difficulty: float = 0.5
    
    # 详细说明
    description: str = ""
    advantages: List[str] = field(default_factory=list)
    disadvantages: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def get_overall_score(self) -> float:
        """计算综合评分"""
        weights = {
            'feasibility': 0.3,
            'quality': 0.25,
            'efficiency': 0.25,
            'difficulty': 0.2
        }
        
        difficulty_score = 1.0 - self.implementation_difficulty
        
        return (
            weights['feasibility'] * self.feasibility_score +
            weights['quality'] * self.quality_score +
            weights['efficiency'] * self.resource_efficiency +
            weights['difficulty'] * difficulty_score
        )


@dataclass
class RecommendationReport(BaseModel):
    """推荐报告"""
    original_design: ExperimentDesign = field(default_factory=ExperimentDesign)
    constraints: List[ResourceConstraint] = field(default_factory=list)
    alternatives: List[AlternativeDesign] = field(default_factory=list)
    
    # 分析结果
    constraint_analysis: Dict[str, Any] = field(default_factory=dict)
    recommendation_summary: str = ""
    best_alternative: Optional[AlternativeDesign] = None
    
    def get_top_alternatives(self, n: int = 3) -> List[AlternativeDesign]:
        """获取前N个最佳替代方案"""
        sorted_alternatives = sorted(
            self.alternatives,
            key=lambda alt: alt.get_overall_score(),
            reverse=True
        )
        return sorted_alternatives[:n]


class AlternativeRecommender(BaseComponent):
    """替代方案推荐器"""
    
    def get_required_configs(self) -> List[str]:
        """获取必需的配置项"""
        return []
    
    def _setup_component(self):
        """设置组件特定的初始化逻辑"""
        self.logger.info("替代方案推荐器初始化完成")
        
        # 初始化可行性评估器
        self.feasibility_assessor = FeasibilityAssessor()
        
        # 初始化简化策略权重
        self.strategy_weights = {
            SimplificationStrategy.REDUCE_COMPLEXITY: 0.8,
            SimplificationStrategy.REDUCE_SCOPE: 0.7,
            SimplificationStrategy.REDUCE_PRECISION: 0.6,
            SimplificationStrategy.REDUCE_TIMELINE: 0.5,
            SimplificationStrategy.CHANGE_METHOD: 0.9,
            SimplificationStrategy.SPLIT_PHASES: 0.7
        }
    
    def analyze_constraints(self, design: ExperimentDesign, 
                          constraints: List[ResourceConstraint]) -> Dict[str, Any]:
        """
        分析资源约束
        
        Args:
            design: 原始实验设计
            constraints: 资源约束列表
            
        Returns:
            约束分析结果
        """
        try:
            self.logger.info("开始分析资源约束")
            
            analysis = {
                'violated_constraints': [],
                'critical_constraints': [],
                'constraint_severity': {},
                'bottleneck_resources': [],
                'optimization_potential': {}
            }
            
            # 分析每个约束
            for constraint in constraints:
                utilization = constraint.get_utilization_rate()
                
                # 检查违反的约束
                if constraint.is_violated():
                    analysis['violated_constraints'].append({
                        'type': constraint.constraint_type,
                        'violation_rate': utilization - 1.0,
                        'priority': constraint.priority
                    })
                
                # 识别关键约束（利用率>80%）
                if utilization > 0.8:
                    analysis['critical_constraints'].append({
                        'type': constraint.constraint_type,
                        'utilization': utilization,
                        'available_capacity': constraint.get_available_capacity()
                    })
                
                # 计算约束严重程度
                if utilization > 1.0:
                    severity = min(1.0, (utilization - 1.0) * 2)  # 超出部分的2倍作为严重程度
                elif utilization > 0.8:
                    severity = (utilization - 0.8) * 2.5  # 80%-100%区间的严重程度
                else:
                    severity = 0.0
                
                analysis['constraint_severity'][constraint.constraint_type] = severity
                
                # 识别瓶颈资源
                if utilization > 0.9:
                    analysis['bottleneck_resources'].append(constraint.constraint_type)
                
                # 评估优化潜力
                if utilization > 0.5:
                    potential = min(1.0, utilization - 0.5)
                    analysis['optimization_potential'][constraint.constraint_type] = potential
            
            self.logger.info(f"约束分析完成，发现 {len(analysis['violated_constraints'])} 个违反的约束")
            return analysis
            
        except Exception as e:
            self.logger.error(f"分析约束时发生错误: {str(e)}")
            raise
    
    def generate_alternatives(self, design: ExperimentDesign,
                            constraints: List[ResourceConstraint]) -> List[AlternativeDesign]:
        """
        生成替代方案
        
        Args:
            design: 原始实验设计
            constraints: 资源约束
            
        Returns:
            替代方案列表
        """
        try:
            self.logger.info("开始生成替代方案")
            
            alternatives = []
            
            # 1. 简化方案
            simplified_alternatives = self._generate_simplified_alternatives(design, constraints)
            alternatives.extend(simplified_alternatives)
            
            # 2. 分阶段方案
            phased_alternatives = self._generate_phased_alternatives(design, constraints)
            alternatives.extend(phased_alternatives)
            
            # 3. 轻量级方案
            lightweight_alternatives = self._generate_lightweight_alternatives(design, constraints)
            alternatives.extend(lightweight_alternatives)
            
            # 4. 基准方案
            baseline_alternatives = self._generate_baseline_alternatives(design, constraints)
            alternatives.extend(baseline_alternatives)
            
            # 5. 混合方案
            hybrid_alternatives = self._generate_hybrid_alternatives(design, constraints)
            alternatives.extend(hybrid_alternatives)
            
            self.logger.info(f"生成了 {len(alternatives)} 个替代方案")
            return alternatives
            
        except Exception as e:
            self.logger.error(f"生成替代方案时发生错误: {str(e)}")
            raise
    
    def evaluate_alternatives(self, alternatives: List[AlternativeDesign],
                            constraints: List[ResourceConstraint]) -> List[AlternativeDesign]:
        """
        评估替代方案
        
        Args:
            alternatives: 替代方案列表
            constraints: 资源约束
            
        Returns:
            评估后的替代方案列表
        """
        try:
            self.logger.info(f"开始评估 {len(alternatives)} 个替代方案")
            
            evaluated_alternatives = []
            
            for alternative in alternatives:
                # 评估可行性
                feasibility_report = self.feasibility_assessor.assess_feasibility(
                    alternative.alternative_design,
                    self._constraints_to_dict(constraints)
                )
                alternative.feasibility_score = feasibility_report.overall_feasibility
                
                # 评估质量
                alternative.quality_score = self._evaluate_quality(alternative)
                
                # 评估资源效率
                alternative.resource_efficiency = self._evaluate_resource_efficiency(
                    alternative, constraints
                )
                
                # 计算性能权衡
                alternative.performance_trade_off = self._calculate_performance_trade_off(alternative)
                
                # 计算资源节省
                alternative.resource_savings = self._calculate_resource_savings(
                    alternative, constraints
                )
                
                # 生成详细说明
                self._generate_alternative_description(alternative)
                
                evaluated_alternatives.append(alternative)
            
            self.logger.info("替代方案评估完成")
            return evaluated_alternatives
            
        except Exception as e:
            self.logger.error(f"评估替代方案时发生错误: {str(e)}")
            raise
    
    def recommend_alternatives(self, design: ExperimentDesign,
                             constraints: List[ResourceConstraint]) -> RecommendationReport:
        """
        推荐替代方案
        
        Args:
            design: 原始实验设计
            constraints: 资源约束
            
        Returns:
            推荐报告
        """
        try:
            self.logger.info("开始生成替代方案推荐")
            
            # 1. 分析约束
            constraint_analysis = self.analyze_constraints(design, constraints)
            
            # 2. 生成替代方案
            alternatives = self.generate_alternatives(design, constraints)
            
            # 3. 评估替代方案
            evaluated_alternatives = self.evaluate_alternatives(alternatives, constraints)
            
            # 4. 选择最佳方案
            best_alternative = max(evaluated_alternatives, 
                                 key=lambda alt: alt.get_overall_score()) if evaluated_alternatives else None
            
            # 5. 生成推荐摘要
            summary = self._generate_recommendation_summary(
                design, constraint_analysis, evaluated_alternatives, best_alternative
            )
            
            # 6. 创建推荐报告
            report = RecommendationReport(
                original_design=design,
                constraints=constraints,
                alternatives=evaluated_alternatives,
                constraint_analysis=constraint_analysis,
                recommendation_summary=summary,
                best_alternative=best_alternative
            )
            
            self.logger.info("替代方案推荐完成")
            return report
            
        except Exception as e:
            self.logger.error(f"推荐替代方案时发生错误: {str(e)}")
            raise
    
    def compare_alternatives(self, alternatives: List[AlternativeDesign]) -> Dict[str, Any]:
        """
        比较替代方案
        
        Args:
            alternatives: 替代方案列表
            
        Returns:
            比较分析结果
        """
        try:
            if not alternatives:
                return {}
            
            comparison = {
                'ranking': [],
                'performance_comparison': {},
                'resource_comparison': {},
                'trade_off_analysis': {},
                'recommendations': []
            }
            
            # 排序
            sorted_alternatives = sorted(alternatives, 
                                       key=lambda alt: alt.get_overall_score(), 
                                       reverse=True)
            
            for i, alt in enumerate(sorted_alternatives, 1):
                comparison['ranking'].append({
                    'rank': i,
                    'type': alt.alternative_type.value,
                    'overall_score': alt.get_overall_score(),
                    'description': alt.description
                })
            
            # 性能比较
            comparison['performance_comparison'] = {
                'best_feasibility': max(alt.feasibility_score for alt in alternatives),
                'best_quality': max(alt.quality_score for alt in alternatives),
                'best_efficiency': max(alt.resource_efficiency for alt in alternatives),
                'average_trade_off': sum(alt.performance_trade_off for alt in alternatives) / len(alternatives)
            }
            
            # 资源比较
            all_resource_types = set()
            for alt in alternatives:
                all_resource_types.update(alt.resource_savings.keys())
            
            for resource_type in all_resource_types:
                savings = [alt.resource_savings.get(resource_type, 0.0) for alt in alternatives]
                comparison['resource_comparison'][resource_type] = {
                    'max_savings': max(savings),
                    'min_savings': min(savings),
                    'average_savings': sum(savings) / len(savings)
                }
            
            # 权衡分析
            comparison['trade_off_analysis'] = self._analyze_trade_offs(alternatives)
            
            # 生成比较建议
            comparison['recommendations'] = self._generate_comparison_recommendations(alternatives)
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"比较替代方案时发生错误: {str(e)}")
            raise
    
    def _generate_simplified_alternatives(self, design: ExperimentDesign,
                                        constraints: List[ResourceConstraint]) -> List[AlternativeDesign]:
        """生成简化方案"""
        alternatives = []
        
        # 策略1: 减少复杂度
        if len(design.parameters) > 3:
            simplified_design = copy.deepcopy(design)
            # 保留最重要的参数
            important_params = list(design.parameters.keys())[:3]
            simplified_design.parameters = {k: v for k, v in design.parameters.items() 
                                          if k in important_params}
            simplified_design.methodology = f"简化的{design.methodology}"
            
            alternative = AlternativeDesign(
                original_design=design,
                alternative_design=simplified_design,
                alternative_type=AlternativeType.SIMPLIFIED,
                simplification_strategies=[SimplificationStrategy.REDUCE_COMPLEXITY],
                description="通过减少参数复杂度来降低实现难度和资源需求",
                implementation_difficulty=0.3
            )
            alternatives.append(alternative)
        
        # 策略2: 减少范围
        scope_reduced_design = copy.deepcopy(design)
        scope_reduced_design.expected_outcomes = design.expected_outcomes[:2]  # 减少预期结果
        scope_reduced_design.timeline_weeks = max(4, int(design.timeline_weeks * 0.7))
        
        alternative = AlternativeDesign(
            original_design=design,
            alternative_design=scope_reduced_design,
            alternative_type=AlternativeType.SIMPLIFIED,
            simplification_strategies=[SimplificationStrategy.REDUCE_SCOPE],
            description="通过缩小实验范围来减少时间和资源投入",
            implementation_difficulty=0.4
        )
        alternatives.append(alternative)
        
        return alternatives
    
    def _generate_phased_alternatives(self, design: ExperimentDesign,
                                    constraints: List[ResourceConstraint]) -> List[AlternativeDesign]:
        """生成分阶段方案"""
        alternatives = []
        
        # 分阶段实施
        phase1_design = copy.deepcopy(design)
        phase1_design.methodology = f"第一阶段：{design.methodology}基础验证"
        phase1_design.timeline_weeks = max(4, int(design.timeline_weeks * 0.4))
        phase1_design.expected_outcomes = [
            "验证方法可行性",
            "建立基础实验框架"
        ]
        
        alternative = AlternativeDesign(
            original_design=design,
            alternative_design=phase1_design,
            alternative_type=AlternativeType.PHASED,
            simplification_strategies=[SimplificationStrategy.SPLIT_PHASES],
            description="将实验分为多个阶段，先验证核心可行性",
            implementation_difficulty=0.5
        )
        alternatives.append(alternative)
        
        return alternatives
    
    def _generate_lightweight_alternatives(self, design: ExperimentDesign,
                                         constraints: List[ResourceConstraint]) -> List[AlternativeDesign]:
        """生成轻量级方案"""
        alternatives = []
        
        # 轻量级实现
        lightweight_design = copy.deepcopy(design)
        
        # 调整参数为更轻量的配置
        if 'batch_size' in lightweight_design.parameters:
            lightweight_design.parameters['batch_size'] = min(16, 
                                                             lightweight_design.parameters.get('batch_size', 32))
        if 'epochs' in lightweight_design.parameters:
            lightweight_design.parameters['epochs'] = min(50, 
                                                         lightweight_design.parameters.get('epochs', 100))
        
        lightweight_design.methodology = f"轻量级{design.methodology}"
        lightweight_design.timeline_weeks = max(2, int(design.timeline_weeks * 0.5))
        
        alternative = AlternativeDesign(
            original_design=design,
            alternative_design=lightweight_design,
            alternative_type=AlternativeType.LIGHTWEIGHT,
            simplification_strategies=[SimplificationStrategy.REDUCE_PRECISION, SimplificationStrategy.REDUCE_TIMELINE],
            description="使用轻量级配置快速验证方法有效性",
            implementation_difficulty=0.2
        )
        alternatives.append(alternative)
        
        return alternatives
    
    def _generate_baseline_alternatives(self, design: ExperimentDesign,
                                      constraints: List[ResourceConstraint]) -> List[AlternativeDesign]:
        """生成基准方案"""
        alternatives = []
        
        # 传统基准方法
        baseline_design = copy.deepcopy(design)
        
        # 根据原方法选择合适的基准方法
        if 'deep learning' in design.methodology.lower():
            baseline_design.methodology = "传统机器学习基准"
            baseline_design.parameters = {
                'algorithm': 'random_forest',
                'n_estimators': 100,
                'max_depth': 10
            }
        elif 'neural' in design.methodology.lower():
            baseline_design.methodology = "线性模型基准"
            baseline_design.parameters = {
                'algorithm': 'logistic_regression',
                'regularization': 'l2'
            }
        else:
            baseline_design.methodology = f"简化版{design.methodology}"
        
        baseline_design.timeline_weeks = max(3, int(design.timeline_weeks * 0.3))
        baseline_design.expected_outcomes = ["建立性能基准", "验证数据质量"]
        
        alternative = AlternativeDesign(
            original_design=design,
            alternative_design=baseline_design,
            alternative_type=AlternativeType.BASELINE,
            simplification_strategies=[SimplificationStrategy.CHANGE_METHOD],
            description="使用成熟的传统方法建立性能基准",
            implementation_difficulty=0.1
        )
        alternatives.append(alternative)
        
        return alternatives
    
    def _generate_hybrid_alternatives(self, design: ExperimentDesign,
                                    constraints: List[ResourceConstraint]) -> List[AlternativeDesign]:
        """生成混合方案"""
        alternatives = []
        
        # 混合方法
        hybrid_design = copy.deepcopy(design)
        hybrid_design.methodology = f"混合方法：{design.methodology} + 传统方法"
        hybrid_design.timeline_weeks = int(design.timeline_weeks * 0.8)
        
        # 结合原方法和传统方法的参数
        hybrid_params = design.parameters.copy()
        hybrid_params['hybrid_ratio'] = 0.7  # 70%新方法，30%传统方法
        hybrid_design.parameters = hybrid_params
        
        alternative = AlternativeDesign(
            original_design=design,
            alternative_design=hybrid_design,
            alternative_type=AlternativeType.HYBRID,
            simplification_strategies=[SimplificationStrategy.CHANGE_METHOD, SimplificationStrategy.REDUCE_COMPLEXITY],
            description="结合创新方法和传统方法的优势",
            implementation_difficulty=0.6
        )
        alternatives.append(alternative)
        
        return alternatives
    
    def _evaluate_quality(self, alternative: AlternativeDesign) -> float:
        """评估方案质量"""
        quality_score = 0.5  # 基础分数
        
        # 基于方法类型调整
        if alternative.alternative_type == AlternativeType.BASELINE:
            quality_score += 0.2  # 基准方法通常更可靠
        elif alternative.alternative_type == AlternativeType.HYBRID:
            quality_score += 0.1  # 混合方法平衡性好
        
        # 基于简化策略调整
        for strategy in alternative.simplification_strategies:
            weight = self.strategy_weights.get(strategy, 0.5)
            quality_score += (weight - 0.5) * 0.2
        
        # 基于预期结果数量
        outcome_count = len(alternative.alternative_design.expected_outcomes)
        if outcome_count >= 2:
            quality_score += 0.1
        
        return max(0.0, min(1.0, quality_score))
    
    def _evaluate_resource_efficiency(self, alternative: AlternativeDesign,
                                    constraints: List[ResourceConstraint]) -> float:
        """评估资源效率"""
        efficiency_score = 0.5
        
        # 基于时间线减少
        original_weeks = alternative.original_design.timeline_weeks
        alternative_weeks = alternative.alternative_design.timeline_weeks
        
        if original_weeks > 0:
            time_savings = (original_weeks - alternative_weeks) / original_weeks
            efficiency_score += time_savings * 0.3
        
        # 基于参数复杂度减少
        original_params = len(alternative.original_design.parameters)
        alternative_params = len(alternative.alternative_design.parameters)
        
        if original_params > 0:
            complexity_reduction = (original_params - alternative_params) / original_params
            efficiency_score += complexity_reduction * 0.2
        
        # 基于方法类型
        if alternative.alternative_type == AlternativeType.LIGHTWEIGHT:
            efficiency_score += 0.2
        elif alternative.alternative_type == AlternativeType.BASELINE:
            efficiency_score += 0.3
        
        return max(0.0, min(1.0, efficiency_score))
    
    def _calculate_performance_trade_off(self, alternative: AlternativeDesign) -> float:
        """计算性能权衡"""
        trade_off = 0.0
        
        # 基于简化策略的性能损失
        strategy_penalties = {
            SimplificationStrategy.REDUCE_COMPLEXITY: 0.1,
            SimplificationStrategy.REDUCE_SCOPE: 0.15,
            SimplificationStrategy.REDUCE_PRECISION: 0.2,
            SimplificationStrategy.REDUCE_TIMELINE: 0.1,
            SimplificationStrategy.CHANGE_METHOD: 0.05,
            SimplificationStrategy.SPLIT_PHASES: 0.05
        }
        
        for strategy in alternative.simplification_strategies:
            trade_off += strategy_penalties.get(strategy, 0.1)
        
        # 基于方案类型
        type_penalties = {
            AlternativeType.SIMPLIFIED: 0.1,
            AlternativeType.PHASED: 0.05,
            AlternativeType.LIGHTWEIGHT: 0.2,
            AlternativeType.BASELINE: 0.3,
            AlternativeType.HYBRID: 0.05
        }
        
        trade_off += type_penalties.get(alternative.alternative_type, 0.1)
        
        return min(1.0, trade_off)
    
    def _calculate_resource_savings(self, alternative: AlternativeDesign,
                                  constraints: List[ResourceConstraint]) -> Dict[str, float]:
        """计算资源节省"""
        savings = {}
        
        # 时间节省
        original_weeks = alternative.original_design.timeline_weeks
        alternative_weeks = alternative.alternative_design.timeline_weeks
        if original_weeks > 0:
            savings['time'] = (original_weeks - alternative_weeks) / original_weeks
        
        # 计算资源节省（基于方案类型估算）
        type_savings = {
            AlternativeType.SIMPLIFIED: {'computational': 0.3, 'human': 0.2},
            AlternativeType.PHASED: {'computational': 0.6, 'human': 0.1},
            AlternativeType.LIGHTWEIGHT: {'computational': 0.5, 'human': 0.3},
            AlternativeType.BASELINE: {'computational': 0.7, 'human': 0.4},
            AlternativeType.HYBRID: {'computational': 0.2, 'human': 0.1}
        }
        
        type_specific_savings = type_savings.get(alternative.alternative_type, {})
        savings.update(type_specific_savings)
        
        return savings
    
    def _generate_alternative_description(self, alternative: AlternativeDesign):
        """生成替代方案描述"""
        # 优势
        advantages = []
        if alternative.resource_efficiency > 0.7:
            advantages.append("资源效率高")
        if alternative.feasibility_score > 0.8:
            advantages.append("实现可行性强")
        if alternative.implementation_difficulty < 0.3:
            advantages.append("实现难度低")
        if alternative.performance_trade_off < 0.1:
            advantages.append("性能损失小")
        
        alternative.advantages = advantages
        
        # 劣势
        disadvantages = []
        if alternative.performance_trade_off > 0.2:
            disadvantages.append("可能存在性能损失")
        if alternative.quality_score < 0.5:
            disadvantages.append("方案质量有待提升")
        if alternative.alternative_type == AlternativeType.PHASED:
            disadvantages.append("需要分阶段实施")
        
        alternative.disadvantages = disadvantages
        
        # 建议
        recommendations = []
        if alternative.alternative_type == AlternativeType.BASELINE:
            recommendations.append("建议作为初始验证方案")
        if alternative.resource_efficiency > 0.8:
            recommendations.append("适合资源受限环境")
        if alternative.feasibility_score > 0.9:
            recommendations.append("推荐优先考虑")
        
        alternative.recommendations = recommendations
    
    def _generate_recommendation_summary(self, design: ExperimentDesign,
                                       constraint_analysis: Dict[str, Any],
                                       alternatives: List[AlternativeDesign],
                                       best_alternative: Optional[AlternativeDesign]) -> str:
        """生成推荐摘要"""
        summary_parts = []
        
        # 约束分析摘要
        violated_count = len(constraint_analysis.get('violated_constraints', []))
        if violated_count > 0:
            summary_parts.append(f"原方案违反了 {violated_count} 个资源约束")
        
        # 替代方案摘要
        summary_parts.append(f"生成了 {len(alternatives)} 个替代方案")
        
        # 最佳方案推荐
        if best_alternative:
            summary_parts.append(
                f"推荐使用{best_alternative.alternative_type.value}方案，"
                f"综合评分 {best_alternative.get_overall_score():.2f}"
            )
        
        return "；".join(summary_parts) + "。"
    
    def _analyze_trade_offs(self, alternatives: List[AlternativeDesign]) -> Dict[str, Any]:
        """分析权衡关系"""
        analysis = {
            'performance_vs_efficiency': [],
            'feasibility_vs_quality': [],
            'complexity_vs_resources': []
        }
        
        for alt in alternatives:
            # 性能 vs 效率
            analysis['performance_vs_efficiency'].append({
                'type': alt.alternative_type.value,
                'performance_loss': alt.performance_trade_off,
                'efficiency_gain': alt.resource_efficiency
            })
            
            # 可行性 vs 质量
            analysis['feasibility_vs_quality'].append({
                'type': alt.alternative_type.value,
                'feasibility': alt.feasibility_score,
                'quality': alt.quality_score
            })
        
        return analysis
    
    def _generate_comparison_recommendations(self, alternatives: List[AlternativeDesign]) -> List[str]:
        """生成比较建议"""
        recommendations = []
        
        # 找出最佳方案
        best_alt = max(alternatives, key=lambda alt: alt.get_overall_score())
        recommendations.append(f"推荐选择{best_alt.alternative_type.value}方案作为首选")
        
        # 资源效率最高的方案
        most_efficient = max(alternatives, key=lambda alt: alt.resource_efficiency)
        if most_efficient != best_alt:
            recommendations.append(f"如果资源非常紧张，考虑{most_efficient.alternative_type.value}方案")
        
        # 可行性最高的方案
        most_feasible = max(alternatives, key=lambda alt: alt.feasibility_score)
        if most_feasible != best_alt:
            recommendations.append(f"如果追求稳妥实施，选择{most_feasible.alternative_type.value}方案")
        
        return recommendations
    
    def _constraints_to_dict(self, constraints: List[ResourceConstraint]) -> Dict[str, Any]:
        """将约束列表转换为字典格式"""
        constraint_dict = {}
        for constraint in constraints:
            if constraint.constraint_type == 'budget':
                constraint_dict['max_budget'] = constraint.max_value
            elif constraint.constraint_type == 'time':
                constraint_dict['max_timeline_weeks'] = constraint.max_value
            elif constraint.constraint_type == 'computational':
                constraint_dict['max_gpu_hours'] = constraint.max_value
        
        return constraint_dict