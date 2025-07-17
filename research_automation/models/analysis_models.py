"""
分析相关数据模型

包含文献分析、实验设计、数据分析等功能的数据结构
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Tuple
from enum import Enum

from .base_models import BaseModel, MetricScore, StatusInfo, ValidationMixin


class PaperType(Enum):
    """论文类型枚举"""
    JOURNAL_ARTICLE = "journal_article"
    CONFERENCE_PAPER = "conference_paper"
    PREPRINT = "preprint"
    THESIS = "thesis"
    BOOK_CHAPTER = "book_chapter"
    TECHNICAL_REPORT = "technical_report"


class DataFormat(Enum):
    """数据格式枚举"""
    CSV = "csv"
    JSON = "json"
    XML = "xml"
    EXCEL = "excel"
    PARQUET = "parquet"
    HDF5 = "hdf5"
    PICKLE = "pickle"
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


@dataclass
class Paper(BaseModel, ValidationMixin):
    """学术论文模型"""
    title: str = ""
    authors: List[str] = field(default_factory=list)
    abstract: str = ""
    keywords: List[str] = field(default_factory=list)
    publication_year: int = 0
    journal_or_venue: str = ""
    doi: str = ""
    arxiv_id: str = ""
    citation_count: int = 0
    paper_type: PaperType = PaperType.JOURNAL_ARTICLE
    full_text: str = ""
    references: List[str] = field(default_factory=list)
    research_fields: List[str] = field(default_factory=list)
    methodology: str = ""
    key_findings: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    
    def validate(self) -> bool:
        """验证论文数据完整性"""
        if not self.title or not self.title.strip():
            return False
        if not self.authors:
            return False
        if self.publication_year < 1900 or self.publication_year > datetime.now().year + 1:
            return False
        return True
    
    def get_age_years(self) -> int:
        """获取论文发表年限"""
        return datetime.now().year - self.publication_year
    
    def is_recent(self, years_threshold: int = 5) -> bool:
        """判断是否为近期论文"""
        return self.get_age_years() <= years_threshold
    
    def get_impact_score(self) -> float:
        """计算影响力分数（基于引用数和发表年限）"""
        if self.citation_count == 0:
            return 0.0
        
        # 考虑论文年龄的影响
        age_factor = max(0.1, 1.0 - (self.get_age_years() * 0.05))
        return min(1.0, (self.citation_count / 100.0) * age_factor)


@dataclass
class QualityScore(MetricScore):
    """论文质量评分"""
    novelty_score: float = 0.0
    methodology_score: float = 0.0
    impact_score: float = 0.0
    clarity_score: float = 0.0
    reproducibility_score: float = 0.0
    
    def calculate_overall_score(self) -> float:
        """计算综合质量分数"""
        weights = {
            'novelty': 0.25,
            'methodology': 0.25,
            'impact': 0.20,
            'clarity': 0.15,
            'reproducibility': 0.15
        }
        
        score = (
            weights['novelty'] * self.novelty_score +
            weights['methodology'] * self.methodology_score +
            weights['impact'] * self.impact_score +
            weights['clarity'] * self.clarity_score +
            weights['reproducibility'] * self.reproducibility_score
        )
        
        self.value = score
        return score


@dataclass
class KnowledgeNode(BaseModel):
    """知识图谱节点"""
    label: str = ""
    node_type: str = ""  # concept, method, dataset, author, etc.
    properties: Dict[str, Any] = field(default_factory=dict)
    importance_score: float = 0.0
    related_papers: List[str] = field(default_factory=list)  # Paper IDs
    
    def add_property(self, key: str, value: Any):
        """添加属性"""
        self.properties[key] = value
        self.update_timestamp()


@dataclass
class KnowledgeEdge(BaseModel):
    """知识图谱边"""
    source_node_id: str = ""
    target_node_id: str = ""
    relationship_type: str = ""  # uses, extends, contradicts, supports, etc.
    strength: float = 1.0
    evidence_papers: List[str] = field(default_factory=list)  # Paper IDs
    
    def is_strong_relationship(self, threshold: float = 0.7) -> bool:
        """判断是否为强关系"""
        return self.strength >= threshold


@dataclass
class ResearchGap(BaseModel):
    """研究空白"""
    description: str = ""
    gap_type: str = ""  # methodological, empirical, theoretical
    importance_level: float = 0.5
    difficulty_level: float = 0.5
    related_concepts: List[str] = field(default_factory=list)
    potential_impact: str = ""
    suggested_approaches: List[str] = field(default_factory=list)
    
    def get_priority_score(self) -> float:
        """计算研究空白的优先级分数"""
        # 重要性高、难度适中的空白优先级更高
        return self.importance_level * (1.0 - abs(self.difficulty_level - 0.5))


@dataclass
class KnowledgeGraph(BaseModel):
    """知识图谱"""
    nodes: List[KnowledgeNode] = field(default_factory=list)
    edges: List[KnowledgeEdge] = field(default_factory=list)
    research_gaps: List[ResearchGap] = field(default_factory=list)
    hot_topics: List[str] = field(default_factory=list)
    emerging_trends: List[str] = field(default_factory=list)
    
    def add_node(self, node: KnowledgeNode):
        """添加节点"""
        self.nodes.append(node)
        self.update_timestamp()
    
    def add_edge(self, edge: KnowledgeEdge):
        """添加边"""
        self.edges.append(edge)
        self.update_timestamp()
    
    def find_node_by_label(self, label: str) -> Optional[KnowledgeNode]:
        """根据标签查找节点"""
        for node in self.nodes:
            if node.label == label:
                return node
        return None
    
    def get_node_connections(self, node_id: str) -> List[KnowledgeEdge]:
        """获取节点的所有连接"""
        return [edge for edge in self.edges 
                if edge.source_node_id == node_id or edge.target_node_id == node_id]
    
    def get_top_research_gaps(self, limit: int = 5) -> List[ResearchGap]:
        """获取优先级最高的研究空白"""
        sorted_gaps = sorted(self.research_gaps, 
                           key=lambda gap: gap.get_priority_score(), 
                           reverse=True)
        return sorted_gaps[:limit]


@dataclass
class ExperimentDesign(BaseModel, ValidationMixin):
    """实验设计"""
    methodology: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_outcomes: List[str] = field(default_factory=list)
    feasibility_score: float = 0.0
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    timeline_weeks: int = 4
    success_criteria: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    alternative_designs: List[str] = field(default_factory=list)
    
    def validate(self) -> bool:
        """验证实验设计的完整性"""
        if not self.methodology or not self.methodology.strip():
            return False
        if not self.expected_outcomes:
            return False
        if not (0.0 <= self.feasibility_score <= 1.0):
            return False
        if self.timeline_weeks <= 0:
            return False
        return True
    
    def estimate_cost(self) -> float:
        """估算实验成本"""
        base_cost = self.resource_requirements.get('base_cost', 1000.0)
        complexity_multiplier = 1.0 + (1.0 - self.feasibility_score)
        time_multiplier = self.timeline_weeks / 4.0  # 以4周为基准
        return base_cost * complexity_multiplier * time_multiplier


@dataclass
class FeasibilityScore(MetricScore):
    """可行性评分"""
    technical_feasibility: float = 0.0
    resource_feasibility: float = 0.0
    time_feasibility: float = 0.0
    risk_assessment: float = 0.0
    
    def calculate_overall_feasibility(self) -> float:
        """计算综合可行性分数"""
        weights = [0.3, 0.3, 0.2, 0.2]  # 技术、资源、时间、风险权重
        scores = [self.technical_feasibility, self.resource_feasibility, 
                 self.time_feasibility, 1.0 - self.risk_assessment]
        
        self.value = sum(w * s for w, s in zip(weights, scores))
        return self.value


@dataclass
class ExperimentStatus(StatusInfo):
    """实验状态"""
    start_date: Optional[datetime] = None
    expected_end_date: Optional[datetime] = None
    actual_end_date: Optional[datetime] = None
    current_phase: str = ""
    completed_phases: List[str] = field(default_factory=list)
    issues_encountered: List[str] = field(default_factory=list)
    
    def get_time_remaining(self) -> Optional[int]:
        """获取剩余时间（天）"""
        if not self.expected_end_date:
            return None
        delta = self.expected_end_date - datetime.now()
        return max(0, delta.days)
    
    def is_on_schedule(self) -> bool:
        """判断是否按计划进行"""
        if not self.expected_end_date or not self.start_date:
            return True
        
        total_duration = (self.expected_end_date - self.start_date).days
        elapsed_duration = (datetime.now() - self.start_date).days
        expected_progress = elapsed_duration / total_duration if total_duration > 0 else 0
        
        return abs(self.progress - expected_progress) <= 0.1  # 10%的容差


@dataclass
class ProcessedData(BaseModel):
    """处理后的数据"""
    original_format: DataFormat = DataFormat.CSV
    processed_format: DataFormat = DataFormat.CSV
    data_summary: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    preprocessing_steps: List[str] = field(default_factory=list)
    data_size: int = 0
    missing_values_count: int = 0
    outliers_count: int = 0
    
    def get_quality_score(self) -> float:
        """计算数据质量分数"""
        if self.data_size == 0:
            return 0.0
        
        completeness = 1.0 - (self.missing_values_count / self.data_size)
        cleanliness = 1.0 - (self.outliers_count / self.data_size)
        
        return (completeness + cleanliness) / 2.0


@dataclass
class AnalysisMethod(BaseModel):
    """分析方法"""
    name: str = ""
    method_type: str = ""  # statistical, machine_learning, visualization, etc.
    parameters: Dict[str, Any] = field(default_factory=dict)
    applicability_score: float = 0.0
    computational_complexity: str = "medium"  # low, medium, high
    required_assumptions: List[str] = field(default_factory=list)
    
    def is_suitable_for_data(self, data: ProcessedData) -> bool:
        """判断方法是否适用于给定数据"""
        # 简化的适用性判断逻辑
        return self.applicability_score >= 0.5


@dataclass
class Visualization(BaseModel):
    """可视化结果"""
    chart_type: str = ""
    title: str = ""
    data_source: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    file_path: str = ""
    interactive: bool = False
    insights: List[str] = field(default_factory=list)
    
    def get_file_extension(self) -> str:
        """获取文件扩展名"""
        if self.file_path:
            return self.file_path.split('.')[-1].lower()
        return "png"  # 默认PNG格式


@dataclass
class Anomaly(BaseModel):
    """异常数据点"""
    data_point_id: str = ""
    anomaly_type: str = ""  # outlier, missing, inconsistent, etc.
    severity_level: float = 0.5
    description: str = ""
    suggested_action: str = ""
    confidence_score: float = 0.0
    
    def is_critical(self, threshold: float = 0.8) -> bool:
        """判断是否为严重异常"""
        return self.severity_level >= threshold


@dataclass
class Insight(BaseModel):
    """数据洞察"""
    title: str = ""
    description: str = ""
    insight_type: str = ""  # trend, correlation, pattern, anomaly, etc.
    confidence_level: float = 0.0
    supporting_evidence: List[str] = field(default_factory=list)
    actionable_recommendations: List[str] = field(default_factory=list)
    statistical_significance: Optional[float] = None
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        """判断是否具有统计显著性"""
        if self.statistical_significance is None:
            return False
        return self.statistical_significance < alpha


@dataclass
class AnalysisResults(BaseModel):
    """分析结果"""
    analysis_methods_used: List[AnalysisMethod] = field(default_factory=list)
    visualizations: List[Visualization] = field(default_factory=list)
    anomalies: List[Anomaly] = field(default_factory=list)
    insights: List[Insight] = field(default_factory=list)
    statistical_summary: Dict[str, Any] = field(default_factory=dict)
    execution_time_seconds: float = 0.0
    
    def get_significant_insights(self) -> List[Insight]:
        """获取具有统计显著性的洞察"""
        return [insight for insight in self.insights if insight.is_significant()]
    
    def get_critical_anomalies(self) -> List[Anomaly]:
        """获取严重异常"""
        return [anomaly for anomaly in self.anomalies if anomaly.is_critical()]
    
    def generate_summary_report(self) -> str:
        """生成分析结果摘要报告"""
        report_lines = [
            f"分析完成时间: {self.updated_at}",
            f"执行时间: {self.execution_time_seconds:.2f}秒",
            f"使用的分析方法: {len(self.analysis_methods_used)}种",
            f"生成的可视化: {len(self.visualizations)}个",
            f"发现的异常: {len(self.anomalies)}个（其中{len(self.get_critical_anomalies())}个严重）",
            f"提取的洞察: {len(self.insights)}个（其中{len(self.get_significant_insights())}个显著）"
        ]
        
        if self.insights:
            report_lines.append("\n主要洞察:")
            for insight in self.insights[:3]:  # 显示前3个洞察
                report_lines.append(f"- {insight.title}: {insight.description}")
        
        return "\n".join(report_lines)