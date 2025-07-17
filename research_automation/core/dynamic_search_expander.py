"""
动态检索范围扩展器

根据检索结果自动调整和扩展检索策略，发现新的研究方向
"""

import re
import math
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

from ..models.analysis_models import Paper, KnowledgeGraph
from ..models.base_models import BaseModel
from .base_component import BaseComponent

logger = logging.getLogger(__name__)


@dataclass
class SearchStrategy:
    """检索策略"""
    keywords: List[str] = field(default_factory=list)
    search_fields: List[str] = field(default_factory=list)  # title, abstract, keywords, etc.
    time_range: Tuple[int, int] = (2010, 2024)  # (start_year, end_year)
    venue_filters: List[str] = field(default_factory=list)
    author_filters: List[str] = field(default_factory=list)
    citation_threshold: int = 0
    relevance_threshold: float = 0.5
    max_results: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'keywords': self.keywords,
            'search_fields': self.search_fields,
            'time_range': self.time_range,
            'venue_filters': self.venue_filters,
            'author_filters': self.author_filters,
            'citation_threshold': self.citation_threshold,
            'relevance_threshold': self.relevance_threshold,
            'max_results': self.max_results
        }


@dataclass
class SearchResult:
    """检索结果"""
    papers: List[Paper] = field(default_factory=list)
    total_found: int = 0
    search_time: float = 0.0
    strategy_used: Optional[SearchStrategy] = None
    relevance_scores: Dict[str, float] = field(default_factory=dict)  # paper_id -> relevance_score
    coverage_metrics: Dict[str, float] = field(default_factory=dict)
    
    def get_average_relevance(self) -> float:
        """获取平均相关性分数"""
        if not self.relevance_scores:
            return 0.0
        return sum(self.relevance_scores.values()) / len(self.relevance_scores)
    
    def get_high_quality_papers(self, threshold: float = 0.7) -> List[Paper]:
        """获取高质量论文"""
        return [paper for paper in self.papers 
                if self.relevance_scores.get(paper.id, 0) >= threshold]


@dataclass
class ExpansionSuggestion:
    """扩展建议"""
    new_keywords: List[str] = field(default_factory=list)
    new_venues: List[str] = field(default_factory=list)
    new_authors: List[str] = field(default_factory=list)
    time_range_adjustment: Optional[Tuple[int, int]] = None
    confidence_score: float = 0.0
    reasoning: str = ""
    expected_improvement: float = 0.0


class DynamicSearchExpander(BaseComponent):
    """动态检索范围扩展器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.search_history: List[SearchResult] = []
        self.expansion_history: List[ExpansionSuggestion] = []
        self.performance_metrics: Dict[str, float] = {}
    
    def get_required_configs(self) -> List[str]:
        """获取必需的配置项列表"""
        return []  # 动态检索扩展器不需要特殊配置
    
    def _setup_component(self):
        """设置组件特定的初始化逻辑"""
        self.logger.info("动态检索范围扩展器初始化完成")
    
    def expand_search_strategy(self, 
                             current_strategy: SearchStrategy, 
                             search_results: List[SearchResult],
                             target_coverage: float = 0.8) -> SearchStrategy:
        """
        根据检索结果扩展检索策略
        
        Args:
            current_strategy: 当前检索策略
            search_results: 历史检索结果
            target_coverage: 目标覆盖率
            
        Returns:
            扩展后的检索策略
        """
        try:
            self.logger.info("开始分析检索结果并扩展策略")
            
            # 分析当前检索效果
            effectiveness = self._analyze_search_effectiveness(search_results)
            self.logger.info(f"当前检索效果评分: {effectiveness:.3f}")
            
            # 生成扩展建议
            suggestions = self._generate_expansion_suggestions(
                current_strategy, search_results, target_coverage
            )
            
            # 应用最佳建议
            expanded_strategy = self._apply_expansion_suggestions(
                current_strategy, suggestions
            )
            
            self.logger.info(f"策略扩展完成，新增关键词: {len(expanded_strategy.keywords) - len(current_strategy.keywords)}")
            return expanded_strategy
            
        except Exception as e:
            self.logger.error(f"扩展检索策略时发生错误: {str(e)}")
            raise
    
    def discover_new_directions(self, papers: List[Paper]) -> List[str]:
        """
        从论文中发现新的研究方向
        
        Args:
            papers: 论文列表
            
        Returns:
            新发现的研究方向列表
        """
        try:
            self.logger.info(f"开始从 {len(papers)} 篇论文中发现新研究方向")
            
            # 1. 提取新兴概念
            emerging_concepts = self._extract_emerging_concepts(papers)
            
            # 2. 识别跨领域连接
            cross_domain_connections = self._identify_cross_domain_connections(papers)
            
            # 3. 分析方法论创新
            methodological_innovations = self._analyze_methodological_innovations(papers)
            
            # 4. 合并和排序新方向
            new_directions = self._merge_and_rank_directions(
                emerging_concepts, cross_domain_connections, methodological_innovations
            )
            
            self.logger.info(f"发现 {len(new_directions)} 个新研究方向")
            return new_directions
            
        except Exception as e:
            self.logger.error(f"发现新研究方向时发生错误: {str(e)}")
            raise
    
    def evaluate_search_effectiveness(self, search_result: SearchResult) -> Dict[str, float]:
        """
        评估检索效果
        
        Args:
            search_result: 检索结果
            
        Returns:
            效果评估指标
        """
        try:
            metrics = {}
            
            # 1. 覆盖率指标
            metrics['coverage'] = self._calculate_coverage(search_result)
            
            # 2. 相关性指标
            metrics['relevance'] = search_result.get_average_relevance()
            
            # 3. 多样性指标
            metrics['diversity'] = self._calculate_diversity(search_result.papers)
            
            # 4. 新颖性指标
            metrics['novelty'] = self._calculate_novelty(search_result.papers)
            
            # 5. 质量指标
            metrics['quality'] = self._calculate_quality(search_result.papers)
            
            # 6. 综合效果分数
            metrics['overall'] = self._calculate_overall_effectiveness(metrics)
            
            self.logger.info(f"检索效果评估完成: 综合分数 {metrics['overall']:.3f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"评估检索效果时发生错误: {str(e)}")
            raise
    
    def _analyze_search_effectiveness(self, search_results: List[SearchResult]) -> float:
        """分析检索效果"""
        if not search_results:
            return 0.0
        
        # 计算平均效果指标
        total_relevance = 0.0
        total_coverage = 0.0
        total_results = 0
        
        for result in search_results:
            total_relevance += result.get_average_relevance()
            total_coverage += result.coverage_metrics.get('coverage', 0.0)
            total_results += len(result.papers)
        
        avg_relevance = total_relevance / len(search_results)
        avg_coverage = total_coverage / len(search_results)
        avg_results = total_results / len(search_results)
        
        # 综合评分
        effectiveness = (avg_relevance * 0.4 + avg_coverage * 0.3 + 
                        min(1.0, avg_results / 50) * 0.3)
        
        return effectiveness
    
    def _generate_expansion_suggestions(self, 
                                      strategy: SearchStrategy, 
                                      results: List[SearchResult],
                                      target_coverage: float) -> List[ExpansionSuggestion]:
        """生成扩展建议"""
        suggestions = []
        
        # 合并所有论文
        all_papers = []
        for result in results:
            all_papers.extend(result.papers)
        
        if not all_papers:
            return suggestions
        
        # 1. 基于关键词共现的扩展
        keyword_suggestion = self._suggest_keyword_expansion(all_papers, strategy.keywords)
        if keyword_suggestion.confidence_score > 0.5:
            suggestions.append(keyword_suggestion)
        
        # 2. 基于作者网络的扩展
        author_suggestion = self._suggest_author_expansion(all_papers, strategy.author_filters)
        if author_suggestion.confidence_score > 0.5:
            suggestions.append(author_suggestion)
        
        # 3. 基于发表场所的扩展
        venue_suggestion = self._suggest_venue_expansion(all_papers, strategy.venue_filters)
        if venue_suggestion.confidence_score > 0.5:
            suggestions.append(venue_suggestion)
        
        # 4. 基于时间趋势的扩展
        time_suggestion = self._suggest_time_expansion(all_papers, strategy.time_range)
        if time_suggestion.confidence_score > 0.5:
            suggestions.append(time_suggestion)
        
        # 按置信度排序
        suggestions.sort(key=lambda x: x.confidence_score, reverse=True)
        return suggestions
    
    def _suggest_keyword_expansion(self, papers: List[Paper], current_keywords: List[str]) -> ExpansionSuggestion:
        """建议关键词扩展"""
        # 统计所有关键词频率
        keyword_freq = Counter()
        keyword_cooccurrence = defaultdict(int)
        
        for paper in papers:
            paper_keywords = [kw.lower().strip() for kw in paper.keywords]
            
            # 统计频率
            for keyword in paper_keywords:
                if keyword not in [kw.lower() for kw in current_keywords]:
                    keyword_freq[keyword] += 1
            
            # 统计共现
            for i, kw1 in enumerate(paper_keywords):
                for j, kw2 in enumerate(paper_keywords):
                    if i != j and kw1 in [kw.lower() for kw in current_keywords]:
                        keyword_cooccurrence[kw2] += 1
        
        # 选择高频且与现有关键词共现的新关键词
        candidate_keywords = []
        for keyword, freq in keyword_freq.most_common(20):
            if freq >= 2 and keyword_cooccurrence[keyword] > 0:
                candidate_keywords.append(keyword)
        
        # 计算置信度
        confidence = min(1.0, len(candidate_keywords) / 10.0) if candidate_keywords else 0.0
        
        return ExpansionSuggestion(
            new_keywords=candidate_keywords[:10],
            confidence_score=confidence,
            reasoning=f"基于 {len(papers)} 篇论文的关键词共现分析",
            expected_improvement=confidence * 0.3
        )
    
    def _suggest_author_expansion(self, papers: List[Paper], current_authors: List[str]) -> ExpansionSuggestion:
        """建议作者扩展"""
        # 统计作者频率和合作关系
        author_freq = Counter()
        author_collaboration = defaultdict(set)
        
        for paper in papers:
            for author in paper.authors:
                if author not in current_authors:
                    author_freq[author] += 1
                
                # 记录合作关系
                for other_author in paper.authors:
                    if other_author != author:
                        author_collaboration[author].add(other_author)
        
        # 选择高产且有合作关系的作者
        candidate_authors = []
        for author, freq in author_freq.most_common(15):
            if freq >= 2:
                # 检查是否与现有作者有合作
                has_collaboration = any(
                    existing_author in author_collaboration[author] 
                    for existing_author in current_authors
                )
                if has_collaboration or not current_authors:
                    candidate_authors.append(author)
        
        confidence = min(1.0, len(candidate_authors) / 8.0) if candidate_authors else 0.0
        
        return ExpansionSuggestion(
            new_authors=candidate_authors[:8],
            confidence_score=confidence,
            reasoning=f"基于作者合作网络和发表频率分析",
            expected_improvement=confidence * 0.2
        )
    
    def _suggest_venue_expansion(self, papers: List[Paper], current_venues: List[str]) -> ExpansionSuggestion:
        """建议发表场所扩展"""
        venue_freq = Counter()
        venue_quality = defaultdict(list)
        
        for paper in papers:
            if paper.journal_or_venue and paper.journal_or_venue not in current_venues:
                venue_freq[paper.journal_or_venue] += 1
                venue_quality[paper.journal_or_venue].append(paper.citation_count)
        
        # 选择高频且高质量的场所
        candidate_venues = []
        for venue, freq in venue_freq.most_common(10):
            if freq >= 2:
                avg_citations = sum(venue_quality[venue]) / len(venue_quality[venue])
                if avg_citations >= 10:  # 平均引用数阈值
                    candidate_venues.append(venue)
        
        confidence = min(1.0, len(candidate_venues) / 6.0) if candidate_venues else 0.0
        
        return ExpansionSuggestion(
            new_venues=candidate_venues[:6],
            confidence_score=confidence,
            reasoning=f"基于发表场所质量和相关性分析",
            expected_improvement=confidence * 0.25
        )
    
    def _suggest_time_expansion(self, papers: List[Paper], current_range: Tuple[int, int]) -> ExpansionSuggestion:
        """建议时间范围扩展"""
        if not papers:
            return ExpansionSuggestion(confidence_score=0.0)
        
        years = [paper.publication_year for paper in papers if paper.publication_year > 0]
        if not years:
            return ExpansionSuggestion(confidence_score=0.0)
        
        min_year = min(years)
        max_year = max(years)
        current_start, current_end = current_range
        
        # 检查是否需要扩展时间范围
        suggested_start = min(current_start, min_year - 1)
        suggested_end = max(current_end, max_year + 1)
        
        if suggested_start < current_start or suggested_end > current_end:
            confidence = 0.7
            new_range = (suggested_start, suggested_end)
            
            return ExpansionSuggestion(
                time_range_adjustment=new_range,
                confidence_score=confidence,
                reasoning=f"基于论文发表年份分布，建议扩展到 {suggested_start}-{suggested_end}",
                expected_improvement=0.15
            )
        
        return ExpansionSuggestion(confidence_score=0.0)
    
    def _apply_expansion_suggestions(self, 
                                   strategy: SearchStrategy, 
                                   suggestions: List[ExpansionSuggestion]) -> SearchStrategy:
        """应用扩展建议"""
        expanded_strategy = SearchStrategy(
            keywords=strategy.keywords.copy(),
            search_fields=strategy.search_fields.copy(),
            time_range=strategy.time_range,
            venue_filters=strategy.venue_filters.copy(),
            author_filters=strategy.author_filters.copy(),
            citation_threshold=strategy.citation_threshold,
            relevance_threshold=strategy.relevance_threshold,
            max_results=strategy.max_results
        )
        
        for suggestion in suggestions:
            if suggestion.confidence_score > 0.6:  # 只应用高置信度建议
                # 添加新关键词
                for keyword in suggestion.new_keywords:
                    if keyword not in expanded_strategy.keywords:
                        expanded_strategy.keywords.append(keyword)
                
                # 添加新作者
                for author in suggestion.new_authors:
                    if author not in expanded_strategy.author_filters:
                        expanded_strategy.author_filters.append(author)
                
                # 添加新场所
                for venue in suggestion.new_venues:
                    if venue not in expanded_strategy.venue_filters:
                        expanded_strategy.venue_filters.append(venue)
                
                # 调整时间范围
                if suggestion.time_range_adjustment:
                    expanded_strategy.time_range = suggestion.time_range_adjustment
        
        return expanded_strategy