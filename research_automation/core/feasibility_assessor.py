"""
可行性评估系统

对实验设计方案进行全面的可行性评估，包括技术、资源、时间和风险评估
"""

import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

from ..models.analysis_models import ExperimentDesign, FeasibilityScore, Paper
from ..models.base_models import BaseModel
from .base_component import BaseComponent

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """资源类型枚举"""
    COMPUTATIONAL = "computational"
    HUMAN = "human"
    FINANCIAL = "financial"
    DATA = "data"
    EQUIPMENT = "equipment"


class RiskLevel(Enum):
    """风险等级枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ResourceRequirement(BaseModel):
    """资源需求"""
    resource_type: ResourceType = ResourceType.COMPUTATIONAL
    amount: float = 0.0
    unit: str = ""
    description: str = ""
    criticality: float = 0.5  # 0-1, 1为最关键
    availability_score: float = 1.0  # 0-1, 1为完全可用


@dataclass
class RiskFactor(BaseModel):
    """风险因素"""
    name: str = ""
    description: str = ""
    probability: float = 0.0  # 0-1
    impact: float = 0.0  # 0-1
    risk_level: RiskLevel = RiskLevel.MEDIUM
    mitigation_strategies: List[str] = field(default_factory=list)
    
    def get_risk_score(self) -> float:
        """计算风险分数"""
        return self.probability * self.impact


@dataclass
class TechnicalConstraint(BaseModel):
    """技术约束"""
    constraint_type: str = ""
    description: str = ""
    severity: float = 0.5  # 0-1, 1为最严重
    workaround_available: bool = False
    workaround_cost: float = 0.0


@dataclass
class FeasibilityReport(BaseModel):
    """可行性评估报告"""
    overall_feasibility: float = 0.0
    technical_feasibility: float = 0.0
    resource_feasibility: float = 0.0
    time_feasibility: float = 0.0
    risk_assessment: float = 0.0
    
    resource_requirements: List[ResourceRequirement] = field(default_factory=list)
    risk_factors: List[RiskFactor] = field(default_factory=list)
    technical_constraints: List[TechnicalConstraint] = field(default_factory=list)
    
    recommendations: List[str] = field(default_factory=list)
    alternative_approaches: List[str] = field(default_factory=list)
    
    def get_feasibility_level(self) -> str:
        """获取可行性等级"""
        if self.overall_feasibility >= 0.8:
            return "高度可行"
        elif self.overall_feasibility >= 0.6:
            return "基本可行"
        elif self.overall_feasibility >= 0.4:
            return "可行性较低"
        else:
            return "不可行"


class FeasibilityAssessor(BaseComponent):
    """可行性评估器"""
    
    def get_required_configs(self) -> List[str]:
        """获取必需的配置项"""
        return []
    
    def _setup_component(self):
        """设置组件特定的初始化逻辑"""
        self.logger.info("可行性评估器初始化完成")
        
        # 初始化评估权重
        self.assessment_weights = {
            'technical': 0.3,
            'resource': 0.3,
            'time': 0.2,
            'risk': 0.2
        }
        
        # 初始化风险阈值
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        }
    
    def assess_feasibility(self, 
                          design: ExperimentDesign,
                          constraints: Optional[Dict[str, Any]] = None,
                          literature_context: Optional[List[Paper]] = None) -> FeasibilityReport:
        """
        全面评估实验设计的可行性
        
        Args:
            design: 实验设计
            constraints: 约束条件
            literature_context: 文献背景
            
        Returns:
            可行性评估报告
        """
        try:
            self.logger.info(f"开始评估实验设计 '{design.methodology}' 的可行性")
            
            # 1. 技术可行性评估
            technical_score, technical_constraints = self._assess_technical_feasibility(
                design, literature_context
            )
            
            # 2. 资源可行性评估
            resource_score, resource_requirements = self._assess_resource_feasibility(
                design, constraints
            )
            
            # 3. 时间可行性评估
            time_score = self._assess_time_feasibility(design, constraints)
            
            # 4. 风险评估
            risk_score, risk_factors = self._assess_risks(design, constraints)
            
            # 5. 计算综合可行性
            overall_score = self._calculate_overall_feasibility(
                technical_score, resource_score, time_score, risk_score
            )
            
            # 6. 生成建议和替代方案
            recommendations = self._generate_recommendations(
                design, technical_score, resource_score, time_score, risk_score
            )
            alternatives = self._generate_alternatives(design, constraints)
            
            # 创建评估报告
            report = FeasibilityReport(
                overall_feasibility=overall_score,
                technical_feasibility=technical_score,
                resource_feasibility=resource_score,
                time_feasibility=time_score,
                risk_assessment=risk_score,
                resource_requirements=resource_requirements,
                risk_factors=risk_factors,
                technical_constraints=technical_constraints,
                recommendations=recommendations,
                alternative_approaches=alternatives
            )
            
            self.logger.info(f"可行性评估完成，综合得分: {overall_score:.2f}")
            return report
            
        except Exception as e:
            self.logger.error(f"可行性评估时发生错误: {str(e)}")
            raise
    
    def compare_feasibility(self, designs: List[ExperimentDesign],
                           constraints: Optional[Dict[str, Any]] = None) -> List[Tuple[ExperimentDesign, FeasibilityReport]]:
        """
        比较多个实验设计的可行性
        
        Args:
            designs: 实验设计列表
            constraints: 约束条件
            
        Returns:
            按可行性排序的设计和报告列表
        """
        try:
            self.logger.info(f"开始比较 {len(designs)} 个实验设计的可行性")
            
            results = []
            for design in designs:
                report = self.assess_feasibility(design, constraints)
                results.append((design, report))
            
            # 按可行性得分排序
            results.sort(key=lambda x: x[1].overall_feasibility, reverse=True)
            
            self.logger.info("可行性比较完成")
            return results
            
        except Exception as e:
            self.logger.error(f"比较可行性时发生错误: {str(e)}")
            raise
    
    def _assess_technical_feasibility(self, 
                                    design: ExperimentDesign,
                                    literature_context: Optional[List[Paper]] = None) -> Tuple[float, List[TechnicalConstraint]]:
        """评估技术可行性"""
        score = 1.0
        constraints = []
        
        # 1. 方法成熟度评估
        methodology_score = self._evaluate_methodology_maturity(design.methodology, literature_context)
        score *= methodology_score
        
        # 2. 参数合理性检查
        param_score, param_constraints = self._check_parameter_validity(design.parameters)
        score *= param_score
        constraints.extend(param_constraints)
        
        # 3. 实现复杂度评估
        complexity_score = self._assess_implementation_complexity(design)
        score *= complexity_score
        
        # 4. 技术依赖性检查
        dependency_score, dependency_constraints = self._check_technical_dependencies(design)
        score *= dependency_score
        constraints.extend(dependency_constraints)
        
        return max(0.0, score), constraints
    
    def _assess_resource_feasibility(self, 
                                   design: ExperimentDesign,
                                   constraints: Optional[Dict[str, Any]] = None) -> Tuple[float, List[ResourceRequirement]]:
        """评估资源可行性"""
        score = 1.0
        requirements = []
        
        # 1. 计算资源需求
        computational_req = self._estimate_computational_resources(design)
        requirements.append(computational_req)
        
        human_req = self._estimate_human_resources(design)
        requirements.append(human_req)
        
        financial_req = self._estimate_financial_resources(design)
        requirements.append(financial_req)
        
        data_req = self._estimate_data_resources(design)
        requirements.append(data_req)
        
        # 2. 检查资源可用性
        if constraints:
            for req in requirements:
                availability = self._check_resource_availability(req, constraints)
                req.availability_score = availability
                score *= availability
        
        return max(0.0, score), requirements
    
    def _assess_time_feasibility(self, 
                               design: ExperimentDesign,
                               constraints: Optional[Dict[str, Any]] = None) -> float:
        """评估时间可行性"""
        score = 1.0
        
        # 1. 基础时间合理性
        if design.timeline_weeks <= 0:
            return 0.0
        elif design.timeline_weeks < 2:
            score *= 0.3  # 时间过短
        elif design.timeline_weeks > 52:
            score *= 0.5  # 时间过长
        
        # 2. 检查时间约束
        if constraints and 'max_timeline_weeks' in constraints:
            max_weeks = constraints['max_timeline_weeks']
            if design.timeline_weeks > max_weeks:
                score *= max_weeks / design.timeline_weeks
        
        # 3. 复杂度与时间匹配度
        complexity_factor = self._estimate_time_complexity_factor(design)
        expected_weeks = complexity_factor * 4  # 基础4周
        
        if design.timeline_weeks < expected_weeks * 0.7:
            score *= 0.6  # 时间不足
        elif design.timeline_weeks > expected_weeks * 2:
            score *= 0.8  # 时间冗余
        
        return max(0.0, score)
    
    def _assess_risks(self, 
                     design: ExperimentDesign,
                     constraints: Optional[Dict[str, Any]] = None) -> Tuple[float, List[RiskFactor]]:
        """评估风险"""
        risk_factors = []
        
        # 1. 技术风险
        tech_risks = self._identify_technical_risks(design)
        risk_factors.extend(tech_risks)
        
        # 2. 资源风险
        resource_risks = self._identify_resource_risks(design, constraints)
        risk_factors.extend(resource_risks)
        
        # 3. 时间风险
        time_risks = self._identify_time_risks(design)
        risk_factors.extend(time_risks)
        
        # 4. 外部风险
        external_risks = self._identify_external_risks(design)
        risk_factors.extend(external_risks)
        
        # 计算综合风险分数
        total_risk = 0.0
        for risk in risk_factors:
            total_risk += risk.get_risk_score()
        
        # 归一化风险分数
        risk_score = min(1.0, total_risk / len(risk_factors) if risk_factors else 0.0)
        
        return risk_score, risk_factors
    
    def _calculate_overall_feasibility(self, technical: float, resource: float, 
                                     time: float, risk: float) -> float:
        """计算综合可行性分数"""
        # 风险分数需要反转（风险越高，可行性越低）
        risk_adjusted = 1.0 - risk
        
        overall = (
            self.assessment_weights['technical'] * technical +
            self.assessment_weights['resource'] * resource +
            self.assessment_weights['time'] * time +
            self.assessment_weights['risk'] * risk_adjusted
        )
        
        return max(0.0, min(1.0, overall))
    
    def _evaluate_methodology_maturity(self, methodology: str, 
                                     literature_context: Optional[List[Paper]] = None) -> float:
        """评估方法成熟度"""
        # 基于方法名称的简单评估
        mature_methods = [
            'supervised learning', 'deep learning', 'machine learning',
            'neural networks', 'convolutional neural networks', 'random forest'
        ]
        
        methodology_lower = methodology.lower()
        for method in mature_methods:
            if method in methodology_lower:
                return 0.9
        
        # 如果有文献背景，基于文献数量评估
        if literature_context:
            relevant_papers = [p for p in literature_context 
                             if methodology_lower in p.methodology.lower()]
            if len(relevant_papers) >= 5:
                return 0.8
            elif len(relevant_papers) >= 2:
                return 0.6
        
        return 0.4  # 新方法或不常见方法
    
    def _check_parameter_validity(self, parameters: Dict[str, Any]) -> Tuple[float, List[TechnicalConstraint]]:
        """检查参数有效性"""
        score = 1.0
        constraints = []
        
        # 检查学习率
        if 'learning_rate' in parameters:
            lr = parameters['learning_rate']
            if not isinstance(lr, (int, float)) or lr <= 0 or lr > 1:
                score *= 0.5
                constraints.append(TechnicalConstraint(
                    constraint_type="parameter_validity",
                    description=f"学习率 {lr} 超出合理范围 (0, 1]",
                    severity=0.7
                ))
        
        # 检查批次大小
        if 'batch_size' in parameters:
            batch_size = parameters['batch_size']
            if not isinstance(batch_size, int) or batch_size <= 0 or batch_size > 1024:
                score *= 0.6
                constraints.append(TechnicalConstraint(
                    constraint_type="parameter_validity",
                    description=f"批次大小 {batch_size} 超出合理范围 [1, 1024]",
                    severity=0.6
                ))
        
        # 检查训练轮数
        if 'epochs' in parameters:
            epochs = parameters['epochs']
            if not isinstance(epochs, int) or epochs <= 0 or epochs > 10000:
                score *= 0.7
                constraints.append(TechnicalConstraint(
                    constraint_type="parameter_validity",
                    description=f"训练轮数 {epochs} 超出合理范围 [1, 10000]",
                    severity=0.5
                ))
        
        return score, constraints
    
    def _assess_implementation_complexity(self, design: ExperimentDesign) -> float:
        """评估实现复杂度"""
        complexity_score = 1.0
        
        # 基于参数数量
        param_count = len(design.parameters)
        if param_count > 10:
            complexity_score *= 0.7
        elif param_count > 5:
            complexity_score *= 0.8
        
        # 基于预期结果数量
        outcome_count = len(design.expected_outcomes)
        if outcome_count > 5:
            complexity_score *= 0.8
        
        # 基于方法复杂度关键词
        complex_keywords = ['ensemble', 'multi-modal', 'adversarial', 'reinforcement']
        methodology_lower = design.methodology.lower()
        for keyword in complex_keywords:
            if keyword in methodology_lower:
                complexity_score *= 0.6
                break
        
        return complexity_score
    
    def _check_technical_dependencies(self, design: ExperimentDesign) -> Tuple[float, List[TechnicalConstraint]]:
        """检查技术依赖性"""
        score = 1.0
        constraints = []
        
        # 检查GPU依赖
        if 'deep learning' in design.methodology.lower() or 'neural' in design.methodology.lower():
            constraints.append(TechnicalConstraint(
                constraint_type="hardware_dependency",
                description="需要GPU加速支持",
                severity=0.6,
                workaround_available=True,
                workaround_cost=0.3
            ))
        
        # 检查大数据依赖
        if 'big data' in design.methodology.lower() or design.parameters.get('batch_size', 0) > 512:
            constraints.append(TechnicalConstraint(
                constraint_type="data_dependency",
                description="需要大规模数据处理能力",
                severity=0.5
            ))
        
        return score, constraints
    
    def _estimate_computational_resources(self, design: ExperimentDesign) -> ResourceRequirement:
        """估算计算资源需求"""
        # 基础计算需求
        base_hours = 10.0
        
        # 基于方法调整
        if 'deep learning' in design.methodology.lower():
            base_hours *= 5
        elif 'machine learning' in design.methodology.lower():
            base_hours *= 2
        
        # 基于参数调整
        epochs = design.parameters.get('epochs', 100)
        batch_size = design.parameters.get('batch_size', 32)
        
        estimated_hours = base_hours * (epochs / 100) * (64 / batch_size)
        
        return ResourceRequirement(
            resource_type=ResourceType.COMPUTATIONAL,
            amount=estimated_hours,
            unit="GPU小时",
            description=f"预估需要 {estimated_hours:.1f} GPU小时",
            criticality=0.8
        )
    
    def _estimate_human_resources(self, design: ExperimentDesign) -> ResourceRequirement:
        """估算人力资源需求"""
        base_days = design.timeline_weeks * 3  # 每周3天工作量
        
        # 基于复杂度调整
        complexity_factor = len(design.parameters) / 5.0
        estimated_days = base_days * max(0.5, complexity_factor)
        
        return ResourceRequirement(
            resource_type=ResourceType.HUMAN,
            amount=estimated_days,
            unit="人天",
            description=f"预估需要 {estimated_days:.1f} 人天",
            criticality=0.7
        )
    
    def _estimate_financial_resources(self, design: ExperimentDesign) -> ResourceRequirement:
        """估算财务资源需求"""
        base_cost = 1000.0  # 基础成本
        
        # 基于时间线调整
        time_factor = design.timeline_weeks / 8.0
        
        # 基于复杂度调整
        complexity_factor = 1.0 + (len(design.parameters) - 3) * 0.1
        
        estimated_cost = base_cost * time_factor * complexity_factor
        
        return ResourceRequirement(
            resource_type=ResourceType.FINANCIAL,
            amount=estimated_cost,
            unit="元",
            description=f"预估成本 {estimated_cost:.0f} 元",
            criticality=0.6
        )
    
    def _estimate_data_resources(self, design: ExperimentDesign) -> ResourceRequirement:
        """估算数据资源需求"""
        base_samples = 10000
        
        # 基于方法调整
        if 'deep learning' in design.methodology.lower():
            base_samples *= 10
        
        return ResourceRequirement(
            resource_type=ResourceType.DATA,
            amount=base_samples,
            unit="样本",
            description=f"预估需要 {base_samples} 个数据样本",
            criticality=0.9
        )
    
    def _check_resource_availability(self, requirement: ResourceRequirement, 
                                   constraints: Dict[str, Any]) -> float:
        """检查资源可用性"""
        if requirement.resource_type == ResourceType.COMPUTATIONAL:
            max_gpu_hours = constraints.get('max_gpu_hours', float('inf'))
            if requirement.amount > max_gpu_hours:
                return max_gpu_hours / requirement.amount
        
        elif requirement.resource_type == ResourceType.FINANCIAL:
            max_budget = constraints.get('max_budget', float('inf'))
            if requirement.amount > max_budget:
                return max_budget / requirement.amount
        
        return 1.0
    
    def _estimate_time_complexity_factor(self, design: ExperimentDesign) -> float:
        """估算时间复杂度因子"""
        factor = 1.0
        
        # 基于方法复杂度
        if 'deep learning' in design.methodology.lower():
            factor *= 3.0
        elif 'ensemble' in design.methodology.lower():
            factor *= 2.0
        
        # 基于参数数量
        param_count = len(design.parameters)
        factor *= (1.0 + param_count * 0.1)
        
        return factor
    
    def _identify_technical_risks(self, design: ExperimentDesign) -> List[RiskFactor]:
        """识别技术风险"""
        risks = []
        
        # 过拟合风险
        if design.parameters.get('epochs', 0) > 500:
            risks.append(RiskFactor(
                name="过拟合风险",
                description="训练轮数过多可能导致过拟合",
                probability=0.6,
                impact=0.7,
                risk_level=RiskLevel.MEDIUM,
                mitigation_strategies=["添加正则化", "早停机制", "交叉验证"]
            ))
        
        # 收敛风险
        if design.parameters.get('learning_rate', 0) > 0.1:
            risks.append(RiskFactor(
                name="收敛困难",
                description="学习率过高可能导致训练不稳定",
                probability=0.5,
                impact=0.8,
                risk_level=RiskLevel.MEDIUM,
                mitigation_strategies=["降低学习率", "学习率调度", "梯度裁剪"]
            ))
        
        return risks
    
    def _identify_resource_risks(self, design: ExperimentDesign, 
                               constraints: Optional[Dict[str, Any]]) -> List[RiskFactor]:
        """识别资源风险"""
        risks = []
        
        # 计算资源不足风险
        if constraints and 'max_gpu_hours' in constraints:
            estimated_hours = self._estimate_computational_resources(design).amount
            if estimated_hours > constraints['max_gpu_hours'] * 0.8:
                risks.append(RiskFactor(
                    name="计算资源不足",
                    description="GPU时间可能不够完成实验",
                    probability=0.7,
                    impact=0.9,
                    risk_level=RiskLevel.HIGH,
                    mitigation_strategies=["优化模型", "减少训练轮数", "申请更多资源"]
                ))
        
        return risks
    
    def _identify_time_risks(self, design: ExperimentDesign) -> List[RiskFactor]:
        """识别时间风险"""
        risks = []
        
        # 时间不足风险
        if design.timeline_weeks < 4:
            risks.append(RiskFactor(
                name="时间不足",
                description="时间线过于紧张",
                probability=0.8,
                impact=0.6,
                risk_level=RiskLevel.HIGH,
                mitigation_strategies=["简化实验", "并行处理", "延长时间线"]
            ))
        
        return risks
    
    def _identify_external_risks(self, design: ExperimentDesign) -> List[RiskFactor]:
        """识别外部风险"""
        risks = []
        
        # 数据质量风险
        risks.append(RiskFactor(
            name="数据质量问题",
            description="数据质量可能影响实验结果",
            probability=0.4,
            impact=0.7,
            risk_level=RiskLevel.MEDIUM,
            mitigation_strategies=["数据预处理", "数据清洗", "数据验证"]
        ))
        
        return risks
    
    def _generate_recommendations(self, design: ExperimentDesign, 
                                technical: float, resource: float, 
                                time: float, risk: float) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        if technical < 0.7:
            recommendations.append("建议简化技术方案或选择更成熟的方法")
        
        if resource < 0.7:
            recommendations.append("建议优化资源使用或申请更多资源")
        
        if time < 0.7:
            recommendations.append("建议调整时间安排或简化实验范围")
        
        if risk > 0.6:
            recommendations.append("建议制定详细的风险缓解策略")
        
        return recommendations
    
    def _generate_alternatives(self, design: ExperimentDesign, 
                             constraints: Optional[Dict[str, Any]]) -> List[str]:
        """生成替代方案"""
        alternatives = []
        
        # 简化版本
        alternatives.append(f"简化版{design.methodology}（减少参数复杂度）")
        
        # 分阶段实施
        alternatives.append("分阶段实施方案（先验证核心功能）")
        
        # 传统方法替代
        if 'deep learning' in design.methodology.lower():
            alternatives.append("使用传统机器学习方法作为基准")
        
        return alternatives