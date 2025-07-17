"""
学术研究自动化增强功能模块

本模块为Agent Laboratory系统提供智能化研究自动化功能，包括：
- 智能研究规划
- 自适应文献分析  
- 智能实验设计
- 自动化数据分析
- 智能报告生成
- 协作管理
- 质量评估
- 个性化学习
"""

__version__ = "1.0.0"
__author__ = "Agent Laboratory Team"

# 导入核心组件
from .core import *
from .models import *
from .services import *
from .utils import *

__all__ = [
    'ResearchPlannerComponent',
    'AdaptiveLiteratureComponent', 
    'ExperimentDesignComponent',
    'DataAnalysisComponent',
    'ReportGenerationComponent',
    'CollaborationComponent',
    'QualityAssessmentComponent',
    'PersonalizationComponent'
]