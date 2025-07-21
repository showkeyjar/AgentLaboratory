"""
核心组件模块

包含所有智能化研究自动化的核心组件类
"""

from .research_planner import ResearchPlannerComponent
from .literature_analyzer import AdaptiveLiteratureComponent
from .experiment_designer import ExperimentDesignComponent
from .feasibility_assessor import FeasibilityAssessor
from .parameter_optimizer import ParameterOptimizer
from .alternative_recommender import AlternativeRecommender
from .experiment_monitor import ExperimentMonitor
from .data_analyzer import DataAnalysisComponent
from .report_generator import ReportGenerationComponent
from .collaboration_manager import CollaborationComponent
from .quality_assessor import QualityAssessmentComponent
from .personalization_engine import PersonalizationComponent
from .base_component import BaseComponent
from .exceptions import *

__all__ = [
    'BaseComponent',
    'ResearchPlannerComponent',
    'AdaptiveLiteratureComponent',
    'ExperimentDesignComponent',
    'FeasibilityAssessor',
    'ParameterOptimizer',
    'AlternativeRecommender',
    'ExperimentMonitor',
    'DataAnalysisComponent',
    'ReportGenerationComponent',
    'CollaborationComponent',
    'QualityAssessmentComponent',
    'PersonalizationComponent',
    
    # 异常类
    'ResearchAutomationError',
    'ValidationError',
    'ProcessingError',
    'ConfigurationError'
]