"""
研究相关数据模型

包含研究规划、主题分析等核心研究功能的数据结构
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from .base_models import BaseModel, MetricScore, StatusInfo, ValidationMixin


class ResearchComplexity(Enum):
    """研究复杂度枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ResearchType(Enum):
    """研究类型枚举"""
    THEORETICAL = "theoretical"
    EMPIRICAL = "empirical"
    EXPERIMENTAL = "experimental"
    SURVEY = "survey"
    CASE_STUDY = "case_study"
    MIXED_METHODS = "mixed_methods"


@dataclass
class TopicAnalysis(BaseModel, ValidationMixin):
    """研究主题分析结果"""
    topic: str = ""
    complexity_score: float = 0.0
    research_scope: List[str] = field(default_factory=list)
    suggested_directions: List[str] = field(default_factory=list)
    estimated_duration: int = 0  # 预估天数
    required_resources: List[str] = field(default_factory=list)
    research_type: ResearchType = ResearchType.EXPERIMENTAL
    complexity_level: ResearchComplexity = ResearchComplexity.MEDIUM
    keywords: List[str] = field(default_factory=list)
    related_fields: List[str] = field(default_factory=list)
    potential_challenges: List[str] = field(default_factory=list)
    success_probability: float = 0.5
    
    def validate(self) -> bool:
        """验证数据有效性"""
        if not self.topic or not self.topic.strip():
            return False
        if not (0.0 <= self.complexity_score <= 1.0):
            return False
        if not (0.0 <= self.success_probability <= 1.0):
            return False
        if self.estimated_duration < 0:
            return False
        return True
    
    def get_complexity_description(self) -> str:
        """获取复杂度描述"""
        descriptions = {
            ResearchComplexity.LOW: "简单研究，资源需求较少",
            ResearchComplexity.MEDIUM: "中等复杂度，需要适中资源投入",
            ResearchComplexity.HIGH: "高复杂度研究，需要大量资源和时间",
            ResearchComplexity.VERY_HIGH: "极高复杂度，需要跨学科合作和长期投入"
        }
        return descriptions.get(self.complexity_level, "未知复杂度")
    
    def estimate_team_size(self) -> int:
        """估算所需团队规模"""
        complexity_mapping = {
            ResearchComplexity.LOW: 1,
            ResearchComplexity.MEDIUM: 2,
            ResearchComplexity.HIGH: 4,
            ResearchComplexity.VERY_HIGH: 6
        }
        return complexity_mapping.get(self.complexity_level, 2)


@dataclass
class Milestone(BaseModel):
    """研究里程碑"""
    title: str = ""
    description: str = ""
    due_date: datetime = field(default_factory=datetime.now)
    dependencies: List[str] = field(default_factory=list)  # 依赖的里程碑ID
    deliverables: List[str] = field(default_factory=list)
    status: str = "not_started"
    completion_percentage: float = 0.0
    assigned_roles: List[str] = field(default_factory=list)
    
    def is_overdue(self) -> bool:
        """判断是否逾期"""
        return datetime.now() > self.due_date and self.status != "completed"
    
    def days_until_due(self) -> int:
        """距离截止日期的天数"""
        delta = self.due_date - datetime.now()
        return delta.days
    
    def mark_completed(self):
        """标记为已完成"""
        self.status = "completed"
        self.completion_percentage = 100.0
        self.update_timestamp()


@dataclass
class ResourceConstraints(BaseModel):
    """资源约束"""
    budget_limit: float = 0.0
    time_limit_days: int = 0
    team_size_limit: int = 0
    computational_resources: Dict[str, Any] = field(default_factory=dict)
    data_access_limitations: List[str] = field(default_factory=list)
    equipment_requirements: List[str] = field(default_factory=list)
    software_licenses: List[str] = field(default_factory=list)
    
    def is_feasible_for_analysis(self, analysis: TopicAnalysis) -> bool:
        """判断资源约束是否满足研究需求"""
        if self.time_limit_days > 0 and analysis.estimated_duration > self.time_limit_days:
            return False
        if self.team_size_limit > 0 and analysis.estimate_team_size() > self.team_size_limit:
            return False
        return True
    
    def get_constraint_violations(self, analysis: TopicAnalysis) -> List[str]:
        """获取约束违反列表"""
        violations = []
        if self.time_limit_days > 0 and analysis.estimated_duration > self.time_limit_days:
            violations.append(f"时间超限：需要{analysis.estimated_duration}天，限制{self.time_limit_days}天")
        if self.team_size_limit > 0 and analysis.estimate_team_size() > self.team_size_limit:
            violations.append(f"团队规模超限：需要{analysis.estimate_team_size()}人，限制{self.team_size_limit}人")
        return violations


@dataclass
class ResearchPath(BaseModel):
    """研究路径"""
    name: str = ""
    description: str = ""
    methodology: str = ""
    expected_outcomes: List[str] = field(default_factory=list)
    risk_level: float = 0.5
    innovation_potential: float = 0.5
    resource_intensity: float = 0.5
    timeline_months: int = 6
    prerequisites: List[str] = field(default_factory=list)
    alternative_approaches: List[str] = field(default_factory=list)
    
    def calculate_priority_score(self) -> float:
        """计算优先级分数"""
        # 综合考虑创新潜力、风险和资源强度
        innovation_weight = 0.4
        risk_weight = 0.3  # 风险越低越好
        resource_weight = 0.3  # 资源强度越低越好
        
        score = (
            innovation_weight * self.innovation_potential +
            risk_weight * (1.0 - self.risk_level) +
            resource_weight * (1.0 - self.resource_intensity)
        )
        return min(max(score, 0.0), 1.0)
    
    def get_risk_description(self) -> str:
        """获取风险等级描述"""
        if self.risk_level < 0.3:
            return "低风险"
        elif self.risk_level < 0.7:
            return "中等风险"
        else:
            return "高风险"


@dataclass
class ResearchPlan(BaseModel, ValidationMixin):
    """研究计划"""
    topic_analysis: Optional[TopicAnalysis] = None
    timeline: List[Milestone] = field(default_factory=list)
    research_paths: List[ResearchPath] = field(default_factory=list)
    resource_allocation: Dict[str, Any] = field(default_factory=dict)
    success_metrics: List[str] = field(default_factory=list)
    risk_mitigation_strategies: List[str] = field(default_factory=list)
    quality_checkpoints: List[str] = field(default_factory=list)
    collaboration_requirements: List[str] = field(default_factory=list)
    
    def validate(self) -> bool:
        """验证研究计划的完整性"""
        if not self.topic_analysis or not self.topic_analysis.validate():
            return False
        if not self.timeline:
            return False
        if not self.research_paths:
            return False
        if not self.success_metrics:
            return False
        return True
    
    def get_total_duration(self) -> int:
        """获取总研究时长（天）"""
        if not self.timeline:
            return self.topic_analysis.estimated_duration
        
        start_date = min(milestone.due_date for milestone in self.timeline)
        end_date = max(milestone.due_date for milestone in self.timeline)
        return (end_date - start_date).days
    
    def get_current_progress(self) -> float:
        """获取当前进度百分比"""
        if not self.timeline:
            return 0.0
        
        total_milestones = len(self.timeline)
        completed_milestones = sum(1 for m in self.timeline if m.status == "completed")
        return completed_milestones / total_milestones if total_milestones > 0 else 0.0
    
    def get_overdue_milestones(self) -> List[Milestone]:
        """获取逾期的里程碑"""
        return [m for m in self.timeline if m.is_overdue()]
    
    def get_recommended_path(self) -> Optional[ResearchPath]:
        """获取推荐的研究路径"""
        if not self.research_paths:
            return None
        return max(self.research_paths, key=lambda p: p.calculate_priority_score())
    
    def estimate_budget_requirement(self) -> Dict[str, float]:
        """估算预算需求"""
        # 基于复杂度和时长估算
        base_cost_per_day = {
            ResearchComplexity.LOW: 100.0,
            ResearchComplexity.MEDIUM: 300.0,
            ResearchComplexity.HIGH: 600.0,
            ResearchComplexity.VERY_HIGH: 1000.0
        }
        
        daily_cost = base_cost_per_day.get(
            self.topic_analysis.complexity_level, 
            300.0
        )
        
        total_duration = self.get_total_duration()
        team_size = self.topic_analysis.estimate_team_size()
        
        personnel_cost = daily_cost * total_duration * team_size
        equipment_cost = personnel_cost * 0.2  # 设备成本约为人员成本的20%
        overhead_cost = (personnel_cost + equipment_cost) * 0.15  # 管理费用15%
        
        return {
            "personnel": personnel_cost,
            "equipment": equipment_cost,
            "overhead": overhead_cost,
            "total": personnel_cost + equipment_cost + overhead_cost
        }