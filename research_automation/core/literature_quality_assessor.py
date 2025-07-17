"""
文献质量评估系统

负责分析和评估学术文献的质量，提供多维度的质量评分和筛选建议
"""

import re
import math
from typing import Any, Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum

from .base_component import BaseComponent
from ..models.research_models import TopicAnalysis
from .exceptions import ValidationError, ProcessingError


class QualityDimension(Enum):
    """质量评估维度"""
    RELEVANCE = "relevance"           # 相关性
    NOVELTY = "novelty"              # 新颖性
    METHODOLOGY = "methodology"       # 方法论
    IMPACT = "impact"                # 影响力
    CREDIBILITY = "credibility"      # 可信度
    CLARITY = "clarity"              # 清晰度


@dataclass
class Paper:
    """论文数据模型"""
    title: str
    authors: List[str]
    abstract: str
    keywords: List[str]
    publication_year: int
    journal: str
    citation_count: int
    doi: Optional[str] = None
    venue_type: str = "journal"  # journal, conference, workshop
    
    def validate(self) -> bool:
        """验证论文数据"""
        return (
            bool(self.title and self.title.strip()) and
            bool(self.authors) and
            bool(self.abstract and self.abstract.strip()) and
            self.publication_year > 1900 and
            self.citation_count >= 0
        )


@dataclass 
class QualityScore:
    """质量评分结果"""
    overall_score: float
    dimension_scores: Dict[QualityDimension, float]
    confidence: float
    reasoning: List[str]
    recommendations: List[str]
    
    def validate(self) -> bool:
        """验证质量评分"""
        return (
            0.0 <= self.overall_score <= 1.0 and
            all(0.0 <= score <= 1.0 for score in self.dimension_scores.values()) and
            0.0 <= self.confidence <= 1.0 and
            len(self.dimension_scores) > 0
        )


class LiteratureQualityAssessor(BaseComponent):
    """文献质量评估组件"""
    
    def get_required_configs(self) -> List[str]:
        """获取必需的配置项"""
        return ['quality_threshold', 'citation_weight']
    
    def _setup_component(self):
        """设置组件特定的初始化逻辑"""
        # 初始化质量评估模型和指标
        self._setup_quality_metrics()
        self._setup_venue_rankings()
        self._setup_evaluation_criteria()
        self.logger.info("文献质量评估系统初始化完成")
    
    def _setup_quality_metrics(self):
        """设置质量评估指标"""
        # 质量维度权重
        self.dimension_weights = {
            QualityDimension.RELEVANCE: 0.25,
            QualityDimension.NOVELTY: 0.20,
            QualityDimension.METHODOLOGY: 0.20,
            QualityDimension.IMPACT: 0.15,
            QualityDimension.CREDIBILITY: 0.15,
            QualityDimension.CLARITY: 0.05
        }
        
        # 质量指示词
        self.quality_indicators = {
            'high_quality': [
                'novel', 'innovative', 'breakthrough', 'significant', 'comprehensive',
                'rigorous', 'systematic', 'extensive', 'thorough', 'state-of-the-art',
                'cutting-edge', 'pioneering', 'groundbreaking', 'seminal'
            ],
            'methodology': [
                'experiment', 'evaluation', 'validation', 'comparison', 'analysis',
                'methodology', 'approach', 'framework', 'algorithm', 'model',
                'statistical', 'empirical', 'quantitative', 'qualitative'
            ],
            'negative': [
                'preliminary', 'limited', 'partial', 'incomplete', 'insufficient',
                'unclear', 'ambiguous', 'weak', 'poor', 'inadequate'
            ]
        }
        
        # 学术写作质量指标
        self.writing_quality_indicators = {
            'clear_structure': ['introduction', 'methodology', 'results', 'conclusion'],
            'academic_language': ['furthermore', 'however', 'therefore', 'moreover'],
            'citation_patterns': ['et al.', 'cited', 'reference', 'according to']
        }
    
    def _setup_venue_rankings(self):
        """设置期刊和会议排名"""
        # 顶级期刊和会议（简化版本）
        self.top_venues = {
            'journals': {
                'nature', 'science', 'cell', 'lancet', 'nejm',
                'ieee transactions', 'acm transactions', 'journal of machine learning research'
            },
            'conferences': {
                'nips', 'icml', 'iclr', 'aaai', 'ijcai', 'cvpr', 'iccv', 'eccv',
                'acl', 'emnlp', 'naacl', 'sigir', 'www', 'kdd', 'icde'
            }
        }
        
        # 期刊影响因子估算（简化版本）
        self.impact_factors = {
            'nature': 42.0, 'science': 41.0, 'cell': 38.0,
            'ieee transactions': 5.0, 'acm transactions': 4.0,
            'default_journal': 2.0, 'default_conference': 1.5
        }
    
    def _setup_evaluation_criteria(self):
        """设置评估标准"""
        # 引用数量评估标准（按年份调整）
        self.citation_thresholds = {
            'excellent': 100,
            'good': 50,
            'average': 10,
            'poor': 0
        }
        
        # 时间衰减因子
        self.time_decay_factor = 0.95  # 每年衰减5%
        
        # 最小质量阈值
        self.min_quality_threshold = self.get_config('quality_threshold', 0.6)
    
    def evaluate_paper_quality(self, paper: Paper, context: Optional[TopicAnalysis] = None) -> QualityScore:
        """
        评估单篇论文质量
        
        Args:
            paper: 论文对象
            context: 主题分析上下文
            
        Returns:
            QualityScore: 质量评分结果
        """
        try:
            self.log_operation("evaluate_paper_quality", {
                "paper_title": paper.title[:50],
                "publication_year": paper.publication_year
            })
            
            # 验证输入
            if not paper.validate():
                raise ValidationError("论文数据验证失败")
            
            # 计算各维度分数
            dimension_scores = {}
            reasoning = []
            
            # 1. 相关性评估
            relevance_score, relevance_reason = self._assess_relevance(paper, context)
            dimension_scores[QualityDimension.RELEVANCE] = relevance_score
            reasoning.extend(relevance_reason)
            
            # 2. 新颖性评估
            novelty_score, novelty_reason = self._assess_novelty(paper)
            dimension_scores[QualityDimension.NOVELTY] = novelty_score
            reasoning.extend(novelty_reason)
            
            # 3. 方法论评估
            methodology_score, methodology_reason = self._assess_methodology(paper)
            dimension_scores[QualityDimension.METHODOLOGY] = methodology_score
            reasoning.extend(methodology_reason)
            
            # 4. 影响力评估
            impact_score, impact_reason = self._assess_impact(paper)
            dimension_scores[QualityDimension.IMPACT] = impact_score
            reasoning.extend(impact_reason)
            
            # 5. 可信度评估
            credibility_score, credibility_reason = self._assess_credibility(paper)
            dimension_scores[QualityDimension.CREDIBILITY] = credibility_score
            reasoning.extend(credibility_reason)
            
            # 6. 清晰度评估
            clarity_score, clarity_reason = self._assess_clarity(paper)
            dimension_scores[QualityDimension.CLARITY] = clarity_score
            reasoning.extend(clarity_reason)
            
            # 计算综合分数
            overall_score = sum(
                score * self.dimension_weights[dimension]
                for dimension, score in dimension_scores.items()
            )
            
            # 计算置信度
            confidence = self._calculate_confidence(paper, dimension_scores)
            
            # 生成建议
            recommendations = self._generate_recommendations(paper, dimension_scores)
            
            # 创建质量评分结果
            quality_score = QualityScore(
                overall_score=overall_score,
                dimension_scores=dimension_scores,
                confidence=confidence,
                reasoning=reasoning,
                recommendations=recommendations
            )
            
            # 验证结果
            if not quality_score.validate():
                raise ProcessingError("质量评分结果验证失败")
            
            self.update_metric("papers_evaluated", 
                             self.get_metric("papers_evaluated") or 0 + 1)
            
            self.logger.info(f"论文质量评估完成: 综合分数={overall_score:.2f}, 置信度={confidence:.2f}")
            
            return quality_score
            
        except Exception as e:
            self.handle_error(e, "evaluate_paper_quality")
    
    def _assess_relevance(self, paper: Paper, context: Optional[TopicAnalysis]) -> Tuple[float, List[str]]:
        """评估相关性"""
        score = 0.5  # 基础分数
        reasons = []
        
        if context:
            # 基于主题分析的相关性评估
            topic_keywords = set(kw.lower() for kw in context.keywords)
            paper_text = f"{paper.title} {paper.abstract}".lower()
            paper_keywords = set(paper.keywords) if paper.keywords else set()
            
            # 关键词匹配度
            title_matches = sum(1 for kw in topic_keywords if kw in paper.title.lower())
            abstract_matches = sum(1 for kw in topic_keywords if kw in paper.abstract.lower())
            keyword_matches = len(topic_keywords.intersection(paper_keywords))
            
            total_matches = title_matches * 3 + abstract_matches + keyword_matches * 2
            max_possible = len(topic_keywords) * 6
            
            if max_possible > 0:
                match_ratio = total_matches / max_possible
                score = 0.3 + match_ratio * 0.7  # 调整到0.3-1.0范围
                
                if match_ratio > 0.7:
                    reasons.append("与研究主题高度相关")
                elif match_ratio > 0.4:
                    reasons.append("与研究主题中度相关")
                else:
                    reasons.append("与研究主题相关性较低")
        else:
            # 无上下文时的基础相关性评估
            reasons.append("缺少主题上下文，使用基础相关性评估")
        
        return min(1.0, max(0.0, score)), reasons
    
    def _assess_novelty(self, paper: Paper) -> Tuple[float, List[str]]:
        """评估新颖性"""
        score = 0.5
        reasons = []
        
        text = f"{paper.title} {paper.abstract}".lower()
        
        # 检查新颖性指示词
        novelty_indicators = self.quality_indicators['high_quality']
        novelty_count = sum(1 for indicator in novelty_indicators if indicator in text)
        
        if novelty_count > 0:
            score += min(0.3, novelty_count * 0.1)
            reasons.append(f"包含{novelty_count}个新颖性指示词")
        
        # 基于发表年份的新颖性（较新的论文可能更新颖）
        current_year = datetime.now().year
        years_old = current_year - paper.publication_year
        
        if years_old <= 2:
            score += 0.2
            reasons.append("发表时间较新")
        elif years_old <= 5:
            score += 0.1
            reasons.append("发表时间适中")
        else:
            reasons.append("发表时间较早")
        
        # 检查是否有负面指示词
        negative_indicators = self.quality_indicators['negative']
        negative_count = sum(1 for indicator in negative_indicators if indicator in text)
        
        if negative_count > 0:
            score -= min(0.2, negative_count * 0.05)
            reasons.append(f"包含{negative_count}个限制性词汇")
        
        return min(1.0, max(0.0, score)), reasons
    
    def _assess_methodology(self, paper: Paper) -> Tuple[float, List[str]]:
        """评估方法论"""
        score = 0.4
        reasons = []
        
        text = f"{paper.title} {paper.abstract}".lower()
        
        # 检查方法论指示词
        methodology_indicators = self.quality_indicators['methodology']
        method_count = sum(1 for indicator in methodology_indicators if indicator in text)
        
        if method_count > 0:
            score += min(0.4, method_count * 0.08)
            reasons.append(f"包含{method_count}个方法论相关词汇")
        
        # 检查实验和评估相关词汇
        evaluation_words = ['experiment', 'evaluation', 'validation', 'comparison', 'benchmark']
        eval_count = sum(1 for word in evaluation_words if word in text)
        
        if eval_count > 0:
            score += min(0.2, eval_count * 0.05)
            reasons.append("包含实验评估内容")
        
        # 检查统计和定量分析
        stats_words = ['statistical', 'significance', 'p-value', 'confidence', 'correlation']
        stats_count = sum(1 for word in stats_words if word in text)
        
        if stats_count > 0:
            score += 0.1
            reasons.append("包含统计分析")
        
        return min(1.0, max(0.0, score)), reasons 
   
    def _assess_impact(self, paper: Paper) -> Tuple[float, List[str]]:
        """评估影响力"""
        score = 0.3
        reasons = []
        
        # 基于引用数量评估
        citation_count = paper.citation_count
        years_since_publication = datetime.now().year - paper.publication_year
        
        # 调整引用数量（考虑时间因素）
        if years_since_publication > 0:
            adjusted_citations = citation_count / max(1, years_since_publication)
        else:
            adjusted_citations = citation_count
        
        # 引用数量评分
        if adjusted_citations >= self.citation_thresholds['excellent']:
            score += 0.5
            reasons.append(f"引用数量优秀 ({citation_count}次)")
        elif adjusted_citations >= self.citation_thresholds['good']:
            score += 0.3
            reasons.append(f"引用数量良好 ({citation_count}次)")
        elif adjusted_citations >= self.citation_thresholds['average']:
            score += 0.1
            reasons.append(f"引用数量一般 ({citation_count}次)")
        else:
            reasons.append(f"引用数量较少 ({citation_count}次)")
        
        # 基于期刊/会议声誉评估
        venue_score = self._assess_venue_quality(paper.journal, paper.venue_type)
        score += venue_score * 0.2
        
        if venue_score > 0.8:
            reasons.append("发表在顶级期刊/会议")
        elif venue_score > 0.6:
            reasons.append("发表在知名期刊/会议")
        
        return min(1.0, max(0.0, score)), reasons
    
    def _assess_credibility(self, paper: Paper) -> Tuple[float, List[str]]:
        """评估可信度"""
        score = 0.6
        reasons = []
        
        # 作者数量评估
        author_count = len(paper.authors)
        if author_count >= 3:
            score += 0.1
            reasons.append("多作者合作研究")
        elif author_count == 1:
            score -= 0.05
            reasons.append("单作者研究")
        
        # DOI存在性
        if paper.doi:
            score += 0.1
            reasons.append("具有DOI标识")
        
        # 期刊类型评估
        if paper.venue_type == 'journal':
            score += 0.1
            reasons.append("期刊论文")
        elif paper.venue_type == 'conference':
            score += 0.05
            reasons.append("会议论文")
        
        # 摘要长度评估（合理的摘要长度表明研究的完整性）
        abstract_length = len(paper.abstract.split())
        if 100 <= abstract_length <= 300:
            score += 0.1
            reasons.append("摘要长度适中")
        elif abstract_length < 50:
            score -= 0.1
            reasons.append("摘要过短")
        elif abstract_length > 500:
            score -= 0.05
            reasons.append("摘要过长")
        
        return min(1.0, max(0.0, score)), reasons
    
    def _assess_clarity(self, paper: Paper) -> Tuple[float, List[str]]:
        """评估清晰度"""
        score = 0.5
        reasons = []
        
        text = f"{paper.title} {paper.abstract}".lower()
        
        # 检查学术写作结构
        structure_indicators = self.writing_quality_indicators['clear_structure']
        structure_count = sum(1 for indicator in structure_indicators if indicator in text)
        
        if structure_count >= 3:
            score += 0.2
            reasons.append("具有清晰的学术结构")
        elif structure_count >= 1:
            score += 0.1
            reasons.append("部分学术结构清晰")
        
        # 检查学术语言使用
        academic_language = self.writing_quality_indicators['academic_language']
        language_count = sum(1 for word in academic_language if word in text)
        
        if language_count > 0:
            score += min(0.15, language_count * 0.05)
            reasons.append("使用规范的学术语言")
        
        # 检查引用模式
        citation_patterns = self.writing_quality_indicators['citation_patterns']
        citation_count = sum(1 for pattern in citation_patterns if pattern in text)
        
        if citation_count > 0:
            score += 0.1
            reasons.append("包含适当的引用")
        
        # 标题长度评估
        title_words = len(paper.title.split())
        if 5 <= title_words <= 15:
            score += 0.05
            reasons.append("标题长度适中")
        elif title_words < 3:
            score -= 0.1
            reasons.append("标题过短")
        elif title_words > 20:
            score -= 0.05
            reasons.append("标题过长")
        
        return min(1.0, max(0.0, score)), reasons
    
    def _assess_venue_quality(self, venue: str, venue_type: str) -> float:
        """评估期刊/会议质量"""
        venue_lower = venue.lower()
        
        # 检查是否为顶级期刊/会议
        if venue_type == 'journal':
            top_venues = self.top_venues['journals']
        else:
            top_venues = self.top_venues['conferences']
        
        for top_venue in top_venues:
            if top_venue in venue_lower:
                return 0.9
        
        # 基于影响因子估算
        for venue_name, impact_factor in self.impact_factors.items():
            if venue_name in venue_lower:
                # 将影响因子转换为0-1分数
                return min(1.0, impact_factor / 10.0)
        
        # 默认分数
        if venue_type == 'journal':
            return 0.5
        else:
            return 0.4
    
    def _calculate_confidence(self, paper: Paper, dimension_scores: Dict[QualityDimension, float]) -> float:
        """计算评估置信度"""
        confidence = 0.5
        
        # 基于数据完整性
        data_completeness = 0
        if paper.title and paper.title.strip():
            data_completeness += 0.2
        if paper.abstract and len(paper.abstract) > 50:
            data_completeness += 0.3
        if paper.authors:
            data_completeness += 0.1
        if paper.keywords:
            data_completeness += 0.1
        if paper.citation_count >= 0:
            data_completeness += 0.1
        if paper.journal and paper.journal.strip():
            data_completeness += 0.2
        
        confidence += data_completeness * 0.4
        
        # 基于评分一致性
        scores = list(dimension_scores.values())
        if scores:
            score_variance = sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)
            consistency = 1.0 - min(1.0, score_variance * 4)  # 方差越小，一致性越高
            confidence += consistency * 0.3
        
        # 基于论文年份（较新的论文信息更可靠）
        years_old = datetime.now().year - paper.publication_year
        if years_old <= 5:
            confidence += 0.1
        elif years_old > 20:
            confidence -= 0.1
        
        return min(1.0, max(0.1, confidence))
    
    def _generate_recommendations(self, paper: Paper, dimension_scores: Dict[QualityDimension, float]) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 基于各维度分数生成建议
        for dimension, score in dimension_scores.items():
            if score < 0.5:
                if dimension == QualityDimension.RELEVANCE:
                    recommendations.append("建议检查论文与研究主题的相关性")
                elif dimension == QualityDimension.NOVELTY:
                    recommendations.append("建议关注论文的创新性和新颖性")
                elif dimension == QualityDimension.METHODOLOGY:
                    recommendations.append("建议评估论文的方法论严谨性")
                elif dimension == QualityDimension.IMPACT:
                    recommendations.append("建议考虑论文的学术影响力")
                elif dimension == QualityDimension.CREDIBILITY:
                    recommendations.append("建议验证论文的可信度和权威性")
                elif dimension == QualityDimension.CLARITY:
                    recommendations.append("建议关注论文的表达清晰度")
        
        # 基于综合分数生成建议
        overall_score = sum(
            score * self.dimension_weights[dimension]
            for dimension, score in dimension_scores.items()
        )
        
        if overall_score >= 0.8:
            recommendations.append("高质量论文，建议优先阅读")
        elif overall_score >= 0.6:
            recommendations.append("中等质量论文，可以参考")
        else:
            recommendations.append("质量较低，建议谨慎使用")
        
        return recommendations
    
    def filter_high_quality_papers(self, papers: List[Paper], 
                                 context: Optional[TopicAnalysis] = None,
                                 min_score: Optional[float] = None) -> List[Tuple[Paper, QualityScore]]:
        """
        筛选高质量论文
        
        Args:
            papers: 论文列表
            context: 主题分析上下文
            min_score: 最小质量分数阈值
            
        Returns:
            List[Tuple[Paper, QualityScore]]: 高质量论文及其评分
        """
        try:
            self.log_operation("filter_high_quality_papers", {
                "paper_count": len(papers),
                "min_score": min_score
            })
            
            if not papers:
                return []
            
            min_threshold = min_score or self.min_quality_threshold
            high_quality_papers = []
            
            for paper in papers:
                try:
                    quality_score = self.evaluate_paper_quality(paper, context)
                    
                    if quality_score.overall_score >= min_threshold:
                        high_quality_papers.append((paper, quality_score))
                        
                except Exception as e:
                    self.logger.warning(f"评估论文失败: {paper.title[:50]} - {str(e)}")
                    continue
            
            # 按质量分数排序
            high_quality_papers.sort(key=lambda x: x[1].overall_score, reverse=True)
            
            self.logger.info(f"筛选完成: {len(papers)}篇论文中筛选出{len(high_quality_papers)}篇高质量论文")
            
            return high_quality_papers
            
        except Exception as e:
            self.handle_error(e, "filter_high_quality_papers")
    
    def batch_evaluate_papers(self, papers: List[Paper], 
                            context: Optional[TopicAnalysis] = None) -> List[QualityScore]:
        """
        批量评估论文质量
        
        Args:
            papers: 论文列表
            context: 主题分析上下文
            
        Returns:
            List[QualityScore]: 质量评分列表
        """
        try:
            self.log_operation("batch_evaluate_papers", {"paper_count": len(papers)})
            
            quality_scores = []
            
            for i, paper in enumerate(papers):
                try:
                    quality_score = self.evaluate_paper_quality(paper, context)
                    quality_scores.append(quality_score)
                    
                    if (i + 1) % 10 == 0:
                        self.logger.info(f"已评估 {i + 1}/{len(papers)} 篇论文")
                        
                except Exception as e:
                    self.logger.warning(f"评估第{i+1}篇论文失败: {str(e)}")
                    # 添加默认的低质量评分
                    default_score = QualityScore(
                        overall_score=0.3,
                        dimension_scores={dim: 0.3 for dim in QualityDimension},
                        confidence=0.1,
                        reasoning=["评估失败，使用默认分数"],
                        recommendations=["建议人工审查"]
                    )
                    quality_scores.append(default_score)
            
            self.logger.info(f"批量评估完成: {len(papers)}篇论文")
            
            return quality_scores
            
        except Exception as e:
            self.handle_error(e, "batch_evaluate_papers")
    
    def generate_quality_report(self, papers: List[Paper], 
                              quality_scores: List[QualityScore]) -> Dict[str, Any]:
        """
        生成质量评估报告
        
        Args:
            papers: 论文列表
            quality_scores: 质量评分列表
            
        Returns:
            Dict: 质量评估报告
        """
        try:
            self.log_operation("generate_quality_report", {
                "paper_count": len(papers),
                "score_count": len(quality_scores)
            })
            
            if len(papers) != len(quality_scores):
                raise ValidationError("论文数量与评分数量不匹配")
            
            # 统计分析
            overall_scores = [score.overall_score for score in quality_scores]
            
            report = {
                "summary": {
                    "total_papers": len(papers),
                    "average_quality": sum(overall_scores) / len(overall_scores) if overall_scores else 0,
                    "high_quality_count": sum(1 for score in overall_scores if score >= 0.8),
                    "medium_quality_count": sum(1 for score in overall_scores if 0.6 <= score < 0.8),
                    "low_quality_count": sum(1 for score in overall_scores if score < 0.6)
                },
                "dimension_analysis": {},
                "top_papers": [],
                "recommendations": []
            }
            
            # 各维度分析
            for dimension in QualityDimension:
                dimension_scores = [
                    score.dimension_scores.get(dimension, 0) 
                    for score in quality_scores
                ]
                report["dimension_analysis"][dimension.value] = {
                    "average": sum(dimension_scores) / len(dimension_scores) if dimension_scores else 0,
                    "max": max(dimension_scores) if dimension_scores else 0,
                    "min": min(dimension_scores) if dimension_scores else 0
                }
            
            # 顶级论文
            paper_score_pairs = list(zip(papers, quality_scores))
            paper_score_pairs.sort(key=lambda x: x[1].overall_score, reverse=True)
            
            for paper, score in paper_score_pairs[:5]:
                report["top_papers"].append({
                    "title": paper.title,
                    "authors": paper.authors,
                    "year": paper.publication_year,
                    "journal": paper.journal,
                    "overall_score": score.overall_score,
                    "confidence": score.confidence
                })
            
            # 总体建议
            avg_quality = report["summary"]["average_quality"]
            if avg_quality >= 0.8:
                report["recommendations"].append("整体文献质量优秀，可以放心使用")
            elif avg_quality >= 0.6:
                report["recommendations"].append("整体文献质量良好，建议重点关注高分论文")
            else:
                report["recommendations"].append("整体文献质量偏低，建议扩大检索范围或调整检索策略")
            
            self.logger.info("质量评估报告生成完成")
            
            return report
            
        except Exception as e:
            self.handle_error(e, "generate_quality_report")