"""
协作相关数据模型

包含协作编辑、版本控制、质量评估、个性化等功能的数据结构
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from .base_models import BaseModel, MetricScore, StatusInfo, ValidationMixin


class ChangeType(Enum):
    """变更类型枚举"""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    MOVE = "move"
    RENAME = "rename"


class ConflictType(Enum):
    """冲突类型枚举"""
    CONTENT = "content"
    STRUCTURE = "structure"
    METADATA = "metadata"
    PERMISSION = "permission"


class UserRole(Enum):
    """用户角色枚举"""
    OWNER = "owner"
    COLLABORATOR = "collaborator"
    REVIEWER = "reviewer"
    VIEWER = "viewer"


@dataclass
class Change(BaseModel):
    """变更记录"""
    change_type: ChangeType = ChangeType.UPDATE
    target_path: str = ""
    old_content: str = ""
    new_content: str = ""
    user_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_change_size(self) -> int:
        """获取变更大小（字符数差异）"""
        return abs(len(self.new_content) - len(self.old_content))
    
    def is_major_change(self, threshold: int = 100) -> bool:
        """判断是否为重大变更"""
        return self.get_change_size() >= threshold


@dataclass
class Version(BaseModel):
    """版本信息"""
    version_number: str = ""
    parent_version_id: Optional[str] = None
    changes: List[Change] = field(default_factory=list)
    commit_message: str = ""
    author_id: str = ""
    tags: List[str] = field(default_factory=list)
    is_stable: bool = False
    
    def get_total_changes(self) -> int:
        """获取总变更数量"""
        return len(self.changes)
    
    def get_changes_by_type(self, change_type: ChangeType) -> List[Change]:
        """根据类型获取变更"""
        return [change for change in self.changes if change.change_type == change_type]
    
    def add_tag(self, tag: str):
        """添加标签"""
        if tag not in self.tags:
            self.tags.append(tag)
            self.update_timestamp()


@dataclass
class Conflict(BaseModel):
    """冲突信息"""
    conflict_type: ConflictType = ConflictType.CONTENT
    file_path: str = ""
    conflicting_versions: List[str] = field(default_factory=list)
    description: str = ""
    auto_resolvable: bool = False
    resolution_suggestions: List[str] = field(default_factory=list)
    
    def is_resolvable(self) -> bool:
        """判断冲突是否可解决"""
        return self.auto_resolvable or bool(self.resolution_suggestions)


@dataclass
class Resolution(BaseModel):
    """冲突解决方案"""
    conflict_id: str = ""
    resolution_strategy: str = ""  # merge, overwrite, manual, etc.
    resolved_content: str = ""
    resolver_id: str = ""
    resolution_notes: str = ""
    
    def apply_resolution(self) -> bool:
        """应用解决方案"""
        # 实际应用逻辑需要在服务层实现
        return True


@dataclass
class EditingSession(BaseModel):
    """编辑会话"""
    project_id: str = ""
    user_id: str = ""
    start_time: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    active_files: List[str] = field(default_factory=list)
    cursor_position: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    
    def update_activity(self):
        """更新活动时间"""
        self.last_activity = datetime.now()
        self.update_timestamp()
    
    def get_session_duration(self) -> int:
        """获取会话持续时间（分钟）"""
        duration = self.last_activity - self.start_time
        return int(duration.total_seconds() / 60)
    
    def is_idle(self, idle_threshold_minutes: int = 30) -> bool:
        """判断会话是否空闲"""
        idle_time = datetime.now() - self.last_activity
        return idle_time.total_seconds() / 60 > idle_threshold_minutes


@dataclass
class ActivityLog(BaseModel):
    """活动日志"""
    project_id: str = ""
    user_id: str = ""
    action: str = ""
    target_resource: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: str = ""
    user_agent: str = ""
    
    def get_action_category(self) -> str:
        """获取操作类别"""
        action_categories = {
            'create': 'creation',
            'update': 'modification',
            'delete': 'deletion',
            'view': 'access',
            'download': 'access',
            'share': 'collaboration',
            'comment': 'collaboration'
        }
        
        for keyword, category in action_categories.items():
            if keyword in self.action.lower():
                return category
        return 'other'


@dataclass
class QualityIssue(BaseModel):
    """质量问题"""
    issue_type: str = ""  # methodology, data, analysis, presentation, etc.
    severity: str = ""  # low, medium, high, critical
    description: str = ""
    location: str = ""  # 问题位置
    suggested_fix: str = ""
    auto_fixable: bool = False
    related_requirements: List[str] = field(default_factory=list)
    
    def get_severity_score(self) -> float:
        """获取严重程度分数"""
        severity_mapping = {
            'low': 0.25,
            'medium': 0.5,
            'high': 0.75,
            'critical': 1.0
        }
        return severity_mapping.get(self.severity.lower(), 0.5)


@dataclass
class Improvement(BaseModel):
    """改进建议"""
    title: str = ""
    description: str = ""
    category: str = ""  # methodology, presentation, analysis, etc.
    priority: str = ""  # low, medium, high
    estimated_effort: str = ""  # hours, days, weeks
    expected_impact: str = ""  # low, medium, high
    implementation_steps: List[str] = field(default_factory=list)
    resources_needed: List[str] = field(default_factory=list)
    
    def get_priority_score(self) -> float:
        """计算优先级分数"""
        priority_mapping = {'low': 0.3, 'medium': 0.6, 'high': 1.0}
        impact_mapping = {'low': 0.3, 'medium': 0.6, 'high': 1.0}
        effort_mapping = {'hours': 1.0, 'days': 0.7, 'weeks': 0.4}
        
        priority_score = priority_mapping.get(self.priority.lower(), 0.6)
        impact_score = impact_mapping.get(self.expected_impact.lower(), 0.6)
        effort_score = effort_mapping.get(self.estimated_effort.lower(), 0.7)
        
        return (priority_score + impact_score + effort_score) / 3.0


@dataclass
class QualityAssessment(BaseModel):
    """质量评估结果"""
    overall_score: float = 0.0
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    improvement_suggestions: List[Improvement] = field(default_factory=list)
    quality_issues: List[QualityIssue] = field(default_factory=list)
    assessment_criteria: List[str] = field(default_factory=list)
    
    def get_grade(self) -> str:
        """获取质量等级"""
        if self.overall_score >= 0.9:
            return "优秀"
        elif self.overall_score >= 0.8:
            return "良好"
        elif self.overall_score >= 0.7:
            return "中等"
        elif self.overall_score >= 0.6:
            return "及格"
        else:
            return "不及格"
    
    def get_critical_issues(self) -> List[QualityIssue]:
        """获取严重质量问题"""
        return [issue for issue in self.quality_issues 
                if issue.severity.lower() in ['high', 'critical']]
    
    def get_high_priority_improvements(self) -> List[Improvement]:
        """获取高优先级改进建议"""
        return sorted([imp for imp in self.improvement_suggestions 
                      if imp.priority.lower() == 'high'],
                     key=lambda x: x.get_priority_score(), reverse=True)


@dataclass
class ResearchResults(BaseModel):
    """研究结果"""
    experiment_results: str = ""
    analysis_summary: str = ""
    key_findings: List[str] = field(default_factory=list)
    statistical_significance: Dict[str, float] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    limitations: List[str] = field(default_factory=list)
    
    def is_successful(self, success_threshold: float = 0.7) -> bool:
        """判断研究是否成功"""
        if not self.performance_metrics:
            return False
        avg_performance = sum(self.performance_metrics.values()) / len(self.performance_metrics)
        return avg_performance >= success_threshold


@dataclass
class FailureReason(BaseModel):
    """失败原因"""
    category: str = ""  # methodology, data, resources, external
    description: str = ""
    impact_level: str = ""  # low, medium, high
    preventable: bool = True
    lessons_learned: str = ""


@dataclass
class EffectReport(BaseModel):
    """改进效果报告"""
    improvement_id: str = ""
    before_metrics: Dict[str, float] = field(default_factory=dict)
    after_metrics: Dict[str, float] = field(default_factory=dict)
    implementation_date: datetime = field(default_factory=datetime.now)
    evaluation_date: datetime = field(default_factory=datetime.now)
    success_indicators: List[str] = field(default_factory=list)
    unexpected_effects: List[str] = field(default_factory=list)
    
    def calculate_improvement_percentage(self, metric_name: str) -> Optional[float]:
        """计算特定指标的改进百分比"""
        if metric_name not in self.before_metrics or metric_name not in self.after_metrics:
            return None
        
        before = self.before_metrics[metric_name]
        after = self.after_metrics[metric_name]
        
        if before == 0:
            return None
        
        return ((after - before) / before) * 100.0
    
    def is_successful(self, threshold: float = 5.0) -> bool:
        """判断改进是否成功（基于主要指标的改进幅度）"""
        improvements = []
        for metric in self.before_metrics.keys():
            improvement = self.calculate_improvement_percentage(metric)
            if improvement is not None:
                improvements.append(improvement)
        
        if not improvements:
            return False
        
        avg_improvement = sum(improvements) / len(improvements)
        return avg_improvement >= threshold


@dataclass
class UserBehavior(BaseModel):
    """用户行为记录"""
    user_id: str = ""
    action: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    success: bool = True
    error_message: str = ""
    
    def get_action_type(self) -> str:
        """获取行为类型"""
        if 'search' in self.action.lower():
            return 'search'
        elif 'edit' in self.action.lower():
            return 'edit'
        elif 'view' in self.action.lower():
            return 'view'
        elif 'create' in self.action.lower():
            return 'create'
        else:
            return 'other'


@dataclass
class UserPreferences(BaseModel):
    """用户偏好"""
    user_id: str = ""
    research_interests: List[str] = field(default_factory=list)
    preferred_methodologies: List[str] = field(default_factory=list)
    interface_preferences: Dict[str, Any] = field(default_factory=dict)
    notification_settings: Dict[str, bool] = field(default_factory=dict)
    collaboration_style: str = "balanced"  # independent, collaborative, balanced
    work_patterns: Dict[str, Any] = field(default_factory=dict)
    
    def update_interest(self, interest: str, weight: float = 1.0):
        """更新研究兴趣"""
        if interest not in self.research_interests:
            self.research_interests.append(interest)
        # 可以添加权重逻辑
        self.update_timestamp()


@dataclass
class Context(BaseModel):
    """上下文信息"""
    current_project_id: str = ""
    current_phase: str = ""
    recent_actions: List[str] = field(default_factory=list)
    active_collaborators: List[str] = field(default_factory=list)
    time_of_day: str = ""
    day_of_week: str = ""
    
    def is_collaborative_context(self) -> bool:
        """判断是否为协作环境"""
        return len(self.active_collaborators) > 0


@dataclass
class Recommendations(BaseModel):
    """推荐结果"""
    recommendation_type: str = ""  # papers, methods, collaborators, etc.
    items: List[Dict[str, Any]] = field(default_factory=list)
    confidence_scores: List[float] = field(default_factory=list)
    reasoning: List[str] = field(default_factory=list)
    personalization_factors: List[str] = field(default_factory=list)
    
    def get_top_recommendations(self, limit: int = 5) -> List[Dict[str, Any]]:
        """获取置信度最高的推荐"""
        if not self.confidence_scores:
            return self.items[:limit]
        
        # 按置信度排序
        sorted_items = sorted(zip(self.items, self.confidence_scores), 
                            key=lambda x: x[1], reverse=True)
        return [item for item, _ in sorted_items[:limit]]


@dataclass
class InterfaceConfig(BaseModel):
    """界面配置"""
    user_id: str = ""
    theme: str = "default"
    layout_preferences: Dict[str, Any] = field(default_factory=dict)
    widget_positions: Dict[str, Dict[str, int]] = field(default_factory=dict)
    shortcuts: Dict[str, str] = field(default_factory=dict)
    display_options: Dict[str, bool] = field(default_factory=dict)
    
    def apply_config(self) -> Dict[str, Any]:
        """应用配置并返回界面设置"""
        return {
            'theme': self.theme,
            'layout': self.layout_preferences,
            'widgets': self.widget_positions,
            'shortcuts': self.shortcuts,
            'display': self.display_options
        }


@dataclass
class PreferenceChange(BaseModel):
    """偏好变化记录"""
    user_id: str = ""
    preference_type: str = ""
    old_value: Any = None
    new_value: Any = None
    change_reason: str = ""
    confidence: float = 0.0
    
    def is_significant_change(self) -> bool:
        """判断是否为显著变化"""
        return self.confidence >= 0.7