"""
报告相关数据模型

包含报告生成、格式化等功能的数据结构
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from .base_models import BaseModel, ValidationMixin


class ReportFormat(Enum):
    """报告格式枚举"""
    PDF = "pdf"
    WORD = "word"
    LATEX = "latex"
    HTML = "html"
    MARKDOWN = "markdown"


class JournalType(Enum):
    """期刊类型枚举"""
    IEEE = "ieee"
    ACM = "acm"
    SPRINGER = "springer"
    ELSEVIER = "elsevier"
    NATURE = "nature"
    SCIENCE = "science"


@dataclass
class ReportSection(BaseModel):
    """报告章节"""
    title: str = ""
    content: str = ""
    section_type: str = ""  # abstract, introduction, methods, results, discussion, etc.
    order: int = 0
    subsections: List['ReportSection'] = field(default_factory=list)
    
    def add_subsection(self, subsection: 'ReportSection'):
        """添加子章节"""
        self.subsections.append(subsection)
        self.update_timestamp()


@dataclass
class ReportStructure(BaseModel):
    """报告结构"""
    title: str = ""
    authors: List[str] = field(default_factory=list)
    abstract: str = ""
    sections: List[ReportSection] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    
    def add_section(self, section: ReportSection):
        """添加章节"""
        self.sections.append(section)
        self.update_timestamp()
    
    def get_section_by_type(self, section_type: str) -> Optional[ReportSection]:
        """根据类型获取章节"""
        for section in self.sections:
            if section.section_type == section_type:
                return section
        return None


@dataclass
class Journal(BaseModel):
    """期刊信息"""
    name: str = ""
    journal_type: JournalType = JournalType.IEEE
    format_requirements: Dict[str, Any] = field(default_factory=dict)
    citation_style: str = ""
    page_limit: int = 0
    template_url: str = ""
    
    def get_format_requirement(self, key: str) -> Any:
        """获取格式要求"""
        return self.format_requirements.get(key)


@dataclass
class LogicalIssue(BaseModel):
    """逻辑问题"""
    issue_type: str = ""  # inconsistency, gap, contradiction, etc.
    description: str = ""
    location: str = ""  # section or paragraph reference
    severity: str = ""  # low, medium, high
    suggested_fix: str = ""
    
    def is_critical(self) -> bool:
        """判断是否为严重问题"""
        return self.severity.lower() == 'high'


@dataclass
class Report(BaseModel, ValidationMixin):
    """完整报告"""
    structure: Optional[ReportStructure] = None
    content: str = ""
    format: ReportFormat = ReportFormat.PDF
    target_journal: Optional[Journal] = None
    word_count: int = 0
    logical_issues: List[LogicalIssue] = field(default_factory=list)
    quality_score: float = 0.0
    
    def validate(self) -> bool:
        """验证报告完整性"""
        if not self.structure:
            return False
        if not self.structure.title:
            return False
        if not self.structure.sections:
            return False
        return True
    
    def get_critical_issues(self) -> List[LogicalIssue]:
        """获取严重逻辑问题"""
        return [issue for issue in self.logical_issues if issue.is_critical()]
    
    def calculate_word_count(self) -> int:
        """计算字数"""
        if self.structure:
            total_words = len(self.structure.abstract.split())
            for section in self.structure.sections:
                total_words += len(section.content.split())
                for subsection in section.subsections:
                    total_words += len(subsection.content.split())
            self.word_count = total_words
        return self.word_count
    
    def meets_journal_requirements(self) -> bool:
        """检查是否满足期刊要求"""
        if not self.target_journal:
            return True
        
        # 检查页数限制
        if self.target_journal.page_limit > 0:
            estimated_pages = self.word_count / 250  # 假设每页250字
            if estimated_pages > self.target_journal.page_limit:
                return False
        
        return True


@dataclass
class ResearchData(BaseModel):
    """研究数据汇总"""
    topic_analysis: Optional[Any] = None  # TopicAnalysis
    literature_review: str = ""
    experiment_results: str = ""
    data_analysis_results: Optional[Any] = None  # AnalysisResults
    quality_assessment: Optional[Any] = None  # QualityAssessment
    
    def is_complete(self) -> bool:
        """检查研究数据是否完整"""
        return all([
            self.topic_analysis is not None,
            self.literature_review,
            self.experiment_results
        ])