"""
核心数据模型定义

包含所有研究自动化功能所需的数据结构和模型类
"""

from .base_models import *
from .research_models import *
from .analysis_models import *
from .collaboration_models import *
from .report_models import *

__all__ = [
    # 基础模型
    'BaseModel',
    'TimestampMixin',
    'ValidationMixin',
    
    # 研究相关模型
    'TopicAnalysis',
    'ResearchPlan', 
    'ResearchPath',
    'Milestone',
    'ResourceConstraints',
    
    # 分析相关模型
    'Paper',
    'QualityScore',
    'KnowledgeGraph',
    'KnowledgeNode',
    'KnowledgeEdge',
    'ResearchGap',
    'ExperimentDesign',
    'FeasibilityScore',
    'ExperimentStatus',
    
    # 数据分析模型
    'DataFormat',
    'ProcessedData',
    'AnalysisMethod',
    'AnalysisResults',
    'Visualization',
    'Anomaly',
    'Insight',
    
    # 报告相关模型
    'ReportSection',
    'ReportStructure',
    'Report',
    'Journal',
    'LogicalIssue',
    'ResearchData',
    'ReportFormat',
    'JournalType',
    
    # 协作相关模型
    'EditingSession',
    'Version',
    'Change',
    'Conflict',
    'Resolution',
    'ActivityLog',
    
    # 质量评估模型
    'QualityAssessment',
    'QualityIssue',
    'Improvement',
    'ResearchResults',
    'FailureReason',
    'EffectReport',
    
    # 个性化模型
    'UserBehavior',
    'UserPreferences',
    'Context',
    'Recommendations',
    'InterfaceConfig',
    'PreferenceChange'
]