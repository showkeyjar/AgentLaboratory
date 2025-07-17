"""
智能研究规划组件

负责分析研究主题，生成详细研究计划，提供研究路径建议
"""

import re
import math
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import Counter

from .base_component import BaseComponent
from .direction_recommender import DirectionRecommenderMixin
from ..models.research_models import (
    TopicAnalysis, ResearchPlan, ResearchPath, Milestone, 
    ResourceConstraints, ResearchComplexity, ResearchType
)
from .exceptions import ValidationError, ProcessingError


class ResearchPlannerComponent(BaseComponent, DirectionRecommenderMixin):
    """智能研究规划组件"""
    
    def get_required_configs(self) -> List[str]:
        """获取必需的配置项"""
        return ['llm_model', 'max_analysis_time']
    
    def _setup_component(self):
        """设置组件特定的初始化逻辑"""
        # 初始化关键词词典和复杂度评估模型
        self._setup_keyword_dictionaries()
        self._setup_complexity_models()
        self.logger.info("研究规划组件初始化完成")
    
    def _setup_keyword_dictionaries(self):
        """设置关键词词典"""
        # 复杂度指示词
        self.complexity_indicators = {
            'high': [
                'deep learning', 'neural network', 'artificial intelligence',
                'machine learning', 'quantum', 'blockchain', 'genomics',
                'bioinformatics', 'computational biology', 'systems biology',
                'multi-modal', 'cross-domain', 'interdisciplinary'
            ],
            'medium': [
                'algorithm', 'optimization', 'statistical', 'analysis',
                'modeling', 'simulation', 'classification', 'prediction',
                'clustering', 'regression', 'feature selection'
            ],
            'low': [
                'survey', 'review', 'comparison', 'evaluation',
                'case study', 'descriptive', 'exploratory'
            ]
        }
        
        # 研究类型指示词
        self.research_type_indicators = {
            ResearchType.THEORETICAL: [
                'theory', 'theoretical', 'mathematical', 'formal',
                'proof', 'theorem', 'model', 'framework'
            ],
            ResearchType.EMPIRICAL: [
                'empirical', 'experimental', 'data', 'dataset',
                'measurement', 'observation', 'validation'
            ],
            ResearchType.SURVEY: [
                'survey', 'review', 'systematic review', 'meta-analysis',
                'literature review', 'state-of-the-art'
            ],
            ResearchType.CASE_STUDY: [
                'case study', 'case analysis', 'real-world',
                'application', 'implementation'
            ]
        }
        
        # 学科领域关键词
        self.field_keywords = {
            'computer_science': [
                'algorithm', 'programming', 'software', 'computer',
                'artificial intelligence', 'machine learning', 'data mining'
            ],
            'mathematics': [
                'mathematical', 'theorem', 'proof', 'equation',
                'optimization', 'statistics', 'probability'
            ],
            'biology': [
                'biological', 'genomics', 'protein', 'cell',
                'molecular', 'bioinformatics', 'evolution'
            ],
            'physics': [
                'quantum', 'particle', 'energy', 'wave',
                'mechanics', 'thermodynamics', 'relativity'
            ],
            'engineering': [
                'engineering', 'design', 'system', 'control',
                'optimization', 'manufacturing', 'robotics'
            ]
        }
    
    def _setup_complexity_models(self):
        """设置复杂度评估模型"""
        # 复杂度权重配置
        self.complexity_weights = {
            'keyword_complexity': 0.3,
            'scope_breadth': 0.25,
            'methodology_complexity': 0.2,
            'interdisciplinary_factor': 0.15,
            'novelty_factor': 0.1
        }
    
    def analyze_topic(self, topic: str) -> TopicAnalysis:
        """
        分析研究主题
        
        Args:
            topic: 研究主题描述
            
        Returns:
            TopicAnalysis: 主题分析结果
        """
        try:
            self.log_operation("analyze_topic", {"topic_length": len(topic)})
            
            # 验证输入
            if not topic or not topic.strip():
                raise ValidationError("研究主题不能为空")
            
            topic = topic.strip().lower()
            
            # 提取关键词
            keywords = self._extract_keywords(topic)
            
            # 识别研究类型
            research_type = self._identify_research_type(topic, keywords)
            
            # 计算复杂度分数
            complexity_score = self._calculate_complexity_score(topic, keywords)
            
            # 确定复杂度等级
            complexity_level = self._determine_complexity_level(complexity_score)
            
            # 识别研究范围
            research_scope = self._identify_research_scope(topic, keywords)
            
            # 识别相关领域
            related_fields = self._identify_related_fields(keywords)
            
            # 生成研究方向建议
            suggested_directions = self._generate_research_directions(
                topic, keywords, research_type, complexity_level
            )
            
            # 识别潜在挑战
            potential_challenges = self._identify_challenges(
                complexity_level, research_type, related_fields
            )
            
            # 估算研究时长
            estimated_duration = self._estimate_duration(
                complexity_level, research_type, len(related_fields)
            )
            
            # 识别所需资源
            required_resources = self._identify_required_resources(
                complexity_level, research_type, related_fields
            )
            
            # 评估成功概率
            success_probability = self._estimate_success_probability(
                complexity_score, len(potential_challenges)
            )
            
            # 创建主题分析结果
            analysis = TopicAnalysis(
                topic=topic,
                complexity_score=complexity_score,
                research_scope=research_scope,
                suggested_directions=suggested_directions,
                estimated_duration=estimated_duration,
                required_resources=required_resources,
                research_type=research_type,
                complexity_level=complexity_level,
                keywords=keywords,
                related_fields=related_fields,
                potential_challenges=potential_challenges,
                success_probability=success_probability
            )
            
            # 验证结果
            if not analysis.validate():
                raise ProcessingError("主题分析结果验证失败")
            
            self.update_metric("topics_analyzed", self.get_metric("topics_analyzed") or 0 + 1)
            self.logger.info(f"主题分析完成: 复杂度={complexity_score:.2f}, 类型={research_type.value}")
            
            return analysis
            
        except Exception as e:
            self.handle_error(e, "analyze_topic")
    
    def _extract_keywords(self, topic: str) -> List[str]:
        """提取关键词"""
        # 简单的关键词提取逻辑
        # 移除停用词和标点符号
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        }
        
        # 分词并清理
        words = re.findall(r'\b[a-zA-Z]+\b', topic.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # 提取短语（2-3个词的组合）
        phrases = []
        for i in range(len(words) - 1):
            if words[i] not in stop_words and words[i+1] not in stop_words:
                phrase = f"{words[i]} {words[i+1]}"
                phrases.append(phrase)
        
        # 合并关键词和短语
        all_keywords = list(set(keywords + phrases))
        
        # 按相关性排序（简化版本）
        keyword_scores = {}
        for keyword in all_keywords:
            score = 0
            # 检查是否在复杂度指示词中
            for level, indicators in self.complexity_indicators.items():
                if any(indicator in keyword for indicator in indicators):
                    score += 3 if level == 'high' else 2 if level == 'medium' else 1
            
            # 检查是否在学科关键词中
            for field, field_keywords in self.field_keywords.items():
                if any(fk in keyword for fk in field_keywords):
                    score += 2
            
            keyword_scores[keyword] = score
        
        # 返回评分最高的关键词
        sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
        return [kw for kw, score in sorted_keywords[:20]]  # 最多返回20个关键词
    
    def _identify_research_type(self, topic: str, keywords: List[str]) -> ResearchType:
        """识别研究类型"""
        type_scores = {research_type: 0 for research_type in ResearchType}
        
        # 基于关键词计算各类型的分数
        for research_type, indicators in self.research_type_indicators.items():
            for indicator in indicators:
                if indicator in topic:
                    type_scores[research_type] += 2
                for keyword in keywords:
                    if indicator in keyword:
                        type_scores[research_type] += 1
        
        # 返回得分最高的类型
        best_type = max(type_scores.items(), key=lambda x: x[1])
        return best_type[0] if best_type[1] > 0 else ResearchType.EXPERIMENTAL
    
    def _calculate_complexity_score(self, topic: str, keywords: List[str]) -> float:
        """计算复杂度分数"""
        scores = {}
        
        # 1. 关键词复杂度
        keyword_complexity = 0
        for level, indicators in self.complexity_indicators.items():
            weight = 3 if level == 'high' else 2 if level == 'medium' else 1
            for indicator in indicators:
                if indicator in topic:
                    keyword_complexity += weight * 0.1
                for keyword in keywords:
                    if indicator in keyword:
                        keyword_complexity += weight * 0.05
        
        scores['keyword_complexity'] = min(1.0, keyword_complexity)
        
        # 2. 范围广度（基于关键词数量和多样性）
        scope_breadth = min(1.0, len(keywords) / 20.0)
        scores['scope_breadth'] = scope_breadth
        
        # 3. 方法论复杂度（基于特定方法论关键词）
        methodology_keywords = [
            'deep learning', 'neural network', 'reinforcement learning',
            'natural language processing', 'computer vision', 'optimization',
            'statistical modeling', 'machine learning', 'data mining'
        ]
        methodology_complexity = sum(1 for mk in methodology_keywords 
                                   if any(mk in kw for kw in keywords)) / len(methodology_keywords)
        scores['methodology_complexity'] = methodology_complexity
        
        # 4. 跨学科因子
        field_count = sum(1 for field_kws in self.field_keywords.values()
                         if any(fk in topic or any(fk in kw for kw in keywords) 
                               for fk in field_kws))
        interdisciplinary_factor = min(1.0, (field_count - 1) / 3.0) if field_count > 1 else 0
        scores['interdisciplinary_factor'] = interdisciplinary_factor
        
        # 5. 新颖性因子（基于新兴技术关键词）
        novelty_keywords = [
            'quantum', 'blockchain', 'edge computing', 'federated learning',
            'transformer', 'attention mechanism', 'generative ai', 'llm'
        ]
        novelty_factor = sum(1 for nk in novelty_keywords 
                           if any(nk in kw for kw in keywords)) / len(novelty_keywords)
        scores['novelty_factor'] = novelty_factor
        
        # 计算加权总分
        total_score = sum(scores[key] * self.complexity_weights[key] 
                         for key in scores.keys())
        
        return min(1.0, total_score)
    
    def _determine_complexity_level(self, complexity_score: float) -> ResearchComplexity:
        """确定复杂度等级"""
        if complexity_score >= 0.8:
            return ResearchComplexity.VERY_HIGH
        elif complexity_score >= 0.6:
            return ResearchComplexity.HIGH
        elif complexity_score >= 0.4:
            return ResearchComplexity.MEDIUM
        else:
            return ResearchComplexity.LOW
    
    def _identify_research_scope(self, topic: str, keywords: List[str]) -> List[str]:
        """识别研究范围"""
        scope_areas = []
        
        # 基于关键词识别研究范围
        scope_mapping = {
            'data collection': ['data', 'dataset', 'collection', 'gathering'],
            'algorithm development': ['algorithm', 'method', 'approach', 'technique'],
            'model training': ['training', 'learning', 'optimization', 'tuning'],
            'evaluation': ['evaluation', 'testing', 'validation', 'assessment'],
            'analysis': ['analysis', 'statistical', 'comparison', 'study'],
            'implementation': ['implementation', 'system', 'application', 'deployment'],
            'literature review': ['review', 'survey', 'literature', 'state-of-the-art']
        }
        
        for scope, scope_keywords in scope_mapping.items():
            if any(sk in topic for sk in scope_keywords) or \
               any(any(sk in kw for sk in scope_keywords) for kw in keywords):
                scope_areas.append(scope)
        
        # 如果没有识别到特定范围，添加默认范围
        if not scope_areas:
            scope_areas = ['literature review', 'method development', 'evaluation']
        
        return scope_areas
    
    def _identify_related_fields(self, keywords: List[str]) -> List[str]:
        """识别相关领域"""
        related_fields = []
        
        for field, field_keywords in self.field_keywords.items():
            if any(any(fk in kw for fk in field_keywords) for kw in keywords):
                related_fields.append(field.replace('_', ' ').title())
        
        return related_fields if related_fields else ['Computer Science']
    
    def _generate_research_directions(self, topic: str, keywords: List[str], 
                                    research_type: ResearchType, 
                                    complexity_level: ResearchComplexity) -> List[str]:
        """生成研究方向建议"""
        directions = []
        
        # 基于研究类型的方向建议
        type_directions = {
            ResearchType.THEORETICAL: [
                "理论框架构建与验证",
                "数学模型开发与分析",
                "算法理论复杂度分析"
            ],
            ResearchType.EMPIRICAL: [
                "大规模实验设计与执行",
                "数据驱动的实证分析",
                "统计验证与假设检验"
            ],
            ResearchType.EXPERIMENTAL: [
                "对比实验设计",
                "参数敏感性分析",
                "性能基准测试"
            ],
            ResearchType.SURVEY: [
                "系统性文献综述",
                "技术发展趋势分析",
                "研究空白识别"
            ]
        }
        
        directions.extend(type_directions.get(research_type, []))
        
        # 基于复杂度的方向建议
        if complexity_level in [ResearchComplexity.HIGH, ResearchComplexity.VERY_HIGH]:
            directions.extend([
                "跨学科协作研究",
                "创新方法论开发",
                "长期影响评估"
            ])
        
        # 基于关键词的特定方向
        if any('machine learning' in kw for kw in keywords):
            directions.append("机器学习算法优化")
        if any('deep learning' in kw for kw in keywords):
            directions.append("深度学习架构创新")
        if any('data' in kw for kw in keywords):
            directions.append("数据质量与预处理优化")
        
        return list(set(directions))[:5]  # 最多返回5个方向
    
    def generate_research_directions(self, topic: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成综合研究方向建议
        
        Args:
            topic: 研究主题
            context: 用户上下文信息（经验水平、资源、时间约束等）
            
        Returns:
            Dict: 包含主题分析、研究方向、个性化推荐和选择指导的完整结果
        """
        try:
            self.log_operation("generate_research_directions", {
                "topic": topic,
                "context_keys": list(context.keys())
            })
            
            # 1. 首先进行主题分析
            topic_analysis = self.analyze_topic(topic)
            
            # 2. 生成分类研究方向
            research_directions = self._generate_categorized_directions(topic_analysis)
            
            # 3. 生成个性化推荐
            personalized_recommendations = self._generate_personalized_recommendations(
                topic_analysis, research_directions, context
            )
            
            # 4. 生成选择指导
            selection_guidance = self._generate_selection_guidance(
                topic_analysis, research_directions, context
            )
            
            # 5. 生成实施建议
            implementation_suggestions = self._generate_implementation_suggestions(
                topic_analysis, context
            )
            
            result = {
                "topic_analysis": topic_analysis,
                "research_directions": research_directions,
                "personalized_recommendations": personalized_recommendations,
                "selection_guidance": selection_guidance,
                "implementation_suggestions": implementation_suggestions
            }
            
            current_count = self.get_metric("direction_recommendations_generated") or 0
            self.update_metric("direction_recommendations_generated", current_count + 1)
            
            self.logger.info(f"研究方向建议生成完成: {len(research_directions)}个类别, "
                           f"{sum(len(dirs) for dirs in research_directions.values())}个方向")
            
            return result
            
        except Exception as e:
            self.handle_error(e, "generate_research_directions")
    
    def _generate_categorized_directions(self, analysis: TopicAnalysis) -> Dict[str, List[Dict[str, Any]]]:
        """生成分类的研究方向"""
        directions = {
            "核心研究方向": [],
            "应用导向方向": [],
            "理论扩展方向": [],
            "方法论创新方向": []
        }
        
        # 核心研究方向 - 基于主题分析的直接方向
        core_directions = self._generate_core_directions(analysis)
        directions["核心研究方向"] = core_directions
        
        # 应用导向方向 - 实际应用场景
        application_directions = self._generate_application_directions(analysis)
        directions["应用导向方向"] = application_directions
        
        # 理论扩展方向 - 理论深化和扩展
        theoretical_directions = self._generate_theoretical_directions(analysis)
        directions["理论扩展方向"] = theoretical_directions
        
        # 方法论创新方向 - 新方法和工具
        methodological_directions = self._generate_methodological_directions(analysis)
        directions["方法论创新方向"] = methodological_directions
        
        # 为所有方向添加详细信息
        enriched_directions = self._enrich_directions_with_details(directions, analysis)
        
        return enriched_directions
    
    def _generate_core_directions(self, analysis: TopicAnalysis) -> List[Dict[str, Any]]:
        """生成核心研究方向"""
        directions = []
        
        # 基于研究类型的核心方向
        if analysis.research_type == ResearchType.EXPERIMENTAL:
            directions.extend([
                {
                    "title": "性能优化与基准测试",
                    "description": self._generate_direction_description("性能优化与基准测试", analysis),
                    "feasibility": self._assess_direction_feasibility("性能优化与基准测试", analysis),
                    "innovation_potential": self._assess_innovation_potential("性能优化", analysis),
                    "resource_requirements": self._estimate_direction_resources("性能优化", analysis),
                    "expected_timeline": self._estimate_direction_timeline("性能优化", analysis),
                    "success_probability": self._estimate_direction_success_probability("性能优化", analysis)
                },
                {
                    "title": "算法改进与创新",
                    "description": self._generate_direction_description("算法改进与创新", analysis),
                    "feasibility": self._assess_direction_feasibility("算法改进", analysis),
                    "innovation_potential": self._assess_innovation_potential("算法创新", analysis),
                    "resource_requirements": self._estimate_direction_resources("算法改进", analysis),
                    "expected_timeline": self._estimate_direction_timeline("算法改进", analysis),
                    "success_probability": self._estimate_direction_success_probability("算法改进", analysis)
                }
            ])
        
        elif analysis.research_type == ResearchType.THEORETICAL:
            directions.extend([
                {
                    "title": "理论框架构建",
                    "description": self._generate_direction_description("理论框架构建", analysis),
                    "feasibility": self._assess_direction_feasibility("理论框架", analysis),
                    "innovation_potential": self._assess_innovation_potential("理论创新", analysis),
                    "resource_requirements": self._estimate_direction_resources("理论研究", analysis),
                    "expected_timeline": self._estimate_direction_timeline("理论研究", analysis),
                    "success_probability": self._estimate_direction_success_probability("理论研究", analysis)
                },
                {
                    "title": "数学模型开发",
                    "description": self._generate_direction_description("数学模型开发", analysis),
                    "feasibility": self._assess_direction_feasibility("数学模型", analysis),
                    "innovation_potential": self._assess_innovation_potential("数学建模", analysis),
                    "resource_requirements": self._estimate_direction_resources("数学建模", analysis),
                    "expected_timeline": self._estimate_direction_timeline("数学建模", analysis),
                    "success_probability": self._estimate_direction_success_probability("数学建模", analysis)
                }
            ])
        
        elif analysis.research_type == ResearchType.SURVEY:
            directions.extend([
                {
                    "title": "系统性文献综述",
                    "description": self._generate_direction_description("系统性文献综述", analysis),
                    "feasibility": self._assess_direction_feasibility("文献综述", analysis),
                    "innovation_potential": self._assess_innovation_potential("文献分析", analysis),
                    "resource_requirements": self._estimate_direction_resources("文献综述", analysis),
                    "expected_timeline": self._estimate_direction_timeline("文献综述", analysis),
                    "success_probability": self._estimate_direction_success_probability("文献综述", analysis)
                }
            ])
        
        # 基于复杂度添加方向
        if analysis.complexity_level in [ResearchComplexity.HIGH, ResearchComplexity.VERY_HIGH]:
            directions.append({
                "title": "跨学科整合研究",
                "description": f"整合{', '.join(analysis.related_fields)}等多个领域的知识和方法",
                "feasibility": self._assess_direction_feasibility("跨学科", analysis),
                "innovation_potential": self._assess_innovation_potential("跨学科创新", analysis),
                "resource_requirements": self._estimate_direction_resources("跨学科", analysis),
                "expected_timeline": self._estimate_direction_timeline("跨学科", analysis),
                "success_probability": self._estimate_direction_success_probability("跨学科", analysis)
            })
        
        return directions
    
    def _generate_application_directions(self, analysis: TopicAnalysis) -> List[Dict[str, Any]]:
        """生成应用导向研究方向"""
        directions = []
        
        # 基于关键词识别应用场景
        application_mapping = {
            "healthcare": ["医疗诊断", "药物发现", "健康监测"],
            "finance": ["金融风控", "投资分析", "欺诈检测"],
            "autonomous": ["自动驾驶", "智能导航", "路径规划"],
            "manufacturing": ["智能制造", "质量控制", "预测维护"],
            "education": ["个性化学习", "智能辅导", "教育评估"],
            "security": ["网络安全", "异常检测", "威胁分析"]
        }
        
        # 检查主题中的应用关键词
        topic_lower = analysis.topic.lower()
        for domain, applications in application_mapping.items():
            if domain in topic_lower or any(keyword in topic_lower for keyword in analysis.keywords):
                for app in applications:
                    directions.append({
                        "title": f"{app}应用研究",
                        "description": f"将{analysis.topic}技术应用于{app}领域的研究",
                        "feasibility": self._assess_direction_feasibility(app, analysis),
                        "innovation_potential": self._assess_innovation_potential(f"{app}应用", analysis),
                        "resource_requirements": self._estimate_direction_resources(f"{app}应用", analysis),
                        "expected_timeline": self._estimate_direction_timeline(f"{app}应用", analysis),
                        "success_probability": self._estimate_direction_success_probability(f"{app}应用", analysis),
                        "market_potential": self._assess_market_potential(app),
                        "industry_relevance": self._assess_industry_relevance(app)
                    })
                break
        
        # 如果没有识别到特定应用，添加通用应用方向
        if not directions:
            generic_applications = ["实际系统部署", "用户体验优化", "性能监控与分析"]
            for app in generic_applications:
                directions.append({
                    "title": app,
                    "description": f"针对{analysis.topic}的{app}研究",
                    "feasibility": self._assess_direction_feasibility(app, analysis),
                    "innovation_potential": self._assess_innovation_potential(app, analysis),
                    "resource_requirements": self._estimate_direction_resources(app, analysis),
                    "expected_timeline": self._estimate_direction_timeline(app, analysis),
                    "success_probability": self._estimate_direction_success_probability(app, analysis)
                })
        
        return directions[:3]  # 限制应用方向数量
    
    def _generate_theoretical_directions(self, analysis: TopicAnalysis) -> List[Dict[str, Any]]:
        """生成理论扩展研究方向"""
        directions = []
        
        theoretical_extensions = [
            "复杂度理论分析",
            "收敛性与稳定性研究", 
            "泛化能力理论研究",
            "数学模型扩展",
            "算法理论分析"
        ]
        
        # 基于研究类型选择合适的理论方向
        if analysis.research_type in [ResearchType.THEORETICAL, ResearchType.EXPERIMENTAL]:
            selected_extensions = theoretical_extensions[:3]
        else:
            selected_extensions = theoretical_extensions[:2]
        
        for extension in selected_extensions:
            directions.append({
                "title": extension,
                "description": f"从理论角度深入研究{analysis.topic}的{extension.replace('研究', '').replace('分析', '')}",
                "feasibility": self._assess_direction_feasibility(extension, analysis),
                "innovation_potential": self._assess_innovation_potential(f"理论{extension}", analysis),
                "resource_requirements": self._estimate_direction_resources("理论研究", analysis),
                "expected_timeline": self._estimate_direction_timeline("理论研究", analysis),
                "success_probability": self._estimate_direction_success_probability("理论研究", analysis),
                "mathematical_complexity": self._assess_mathematical_complexity(extension)
            })
        
        return directions
    
    def _generate_methodological_directions(self, analysis: TopicAnalysis) -> List[Dict[str, Any]]:
        """生成方法论创新研究方向"""
        directions = []
        
        methodological_innovations = [
            "新型算法设计",
            "混合方法论开发",
            "评估指标创新",
            "自动化工具开发",
            "可解释性方法研究"
        ]
        
        # 基于复杂度选择方法论创新方向
        if analysis.complexity_level in [ResearchComplexity.HIGH, ResearchComplexity.VERY_HIGH]:
            selected_innovations = methodological_innovations
        else:
            selected_innovations = methodological_innovations[:3]
        
        for innovation in selected_innovations:
            directions.append({
                "title": innovation,
                "description": f"针对{analysis.topic}开发{innovation.replace('研究', '').replace('开发', '')}的新方法",
                "feasibility": self._assess_direction_feasibility(innovation, analysis),
                "innovation_potential": self._assess_innovation_potential(f"方法论{innovation}", analysis),
                "resource_requirements": self._estimate_direction_resources(innovation, analysis),
                "expected_timeline": self._estimate_direction_timeline(innovation, analysis),
                "success_probability": self._estimate_direction_success_probability(innovation, analysis),
                "technical_challenge": self._assess_technical_challenge(innovation)
            })
        
        return directions[:3]  # 限制方法论方向数量
    
    def _identify_challenges(self, complexity_level: ResearchComplexity, 
                           research_type: ResearchType, 
                           related_fields: List[str]) -> List[str]:
        """识别潜在挑战"""
        challenges = []
        
        # 基于复杂度的挑战
        complexity_challenges = {
            ResearchComplexity.LOW: ["时间管理", "文献获取"],
            ResearchComplexity.MEDIUM: ["方法选择", "数据质量", "结果解释"],
            ResearchComplexity.HIGH: ["技术复杂性", "资源协调", "跨领域知识整合"],
            ResearchComplexity.VERY_HIGH: ["项目管理复杂性", "团队协作", "长期可持续性", "创新风险"]
        }
        
        challenges.extend(complexity_challenges.get(complexity_level, []))
        
        # 基于研究类型的挑战
        if research_type == ResearchType.EXPERIMENTAL:
            challenges.extend(["实验设计", "变量控制", "结果重现性"])
        elif research_type == ResearchType.THEORETICAL:
            challenges.extend(["理论验证", "数学证明", "模型简化"])
        
        # 基于跨学科的挑战
        if len(related_fields) > 2:
            challenges.extend(["知识整合", "术语统一", "方法论差异"])
        
        return list(set(challenges))
    
    def _estimate_duration(self, complexity_level: ResearchComplexity, 
                          research_type: ResearchType, 
                          field_count: int) -> int:
        """估算研究时长（天）"""
        # 基础时长
        base_duration = {
            ResearchComplexity.LOW: 30,
            ResearchComplexity.MEDIUM: 90,
            ResearchComplexity.HIGH: 180,
            ResearchComplexity.VERY_HIGH: 365
        }
        
        duration = base_duration.get(complexity_level, 90)
        
        # 研究类型调整
        type_multipliers = {
            ResearchType.SURVEY: 0.7,
            ResearchType.THEORETICAL: 1.2,
            ResearchType.EXPERIMENTAL: 1.5,
            ResearchType.EMPIRICAL: 1.3
        }
        
        duration *= type_multipliers.get(research_type, 1.0)
        
        # 跨学科调整
        if field_count > 2:
            duration *= 1.3
        
        return int(duration)
    
    def _identify_required_resources(self, complexity_level: ResearchComplexity, 
                                   research_type: ResearchType, 
                                   related_fields: List[str]) -> List[str]:
        """识别所需资源"""
        resources = []
        
        # 基础资源
        resources.extend(["文献数据库访问", "计算设备", "办公软件"])
        
        # 基于复杂度的资源
        if complexity_level in [ResearchComplexity.HIGH, ResearchComplexity.VERY_HIGH]:
            resources.extend(["高性能计算资源", "专业软件许可", "专家咨询"])
        
        # 基于研究类型的资源
        if research_type == ResearchType.EXPERIMENTAL:
            resources.extend(["实验环境", "数据集", "统计软件"])
        elif research_type == ResearchType.THEORETICAL:
            resources.extend(["数学软件", "建模工具"])
        
        # 基于领域的资源
        if 'Biology' in related_fields:
            resources.extend(["生物数据库", "基因分析工具"])
        if 'Computer Science' in related_fields:
            resources.extend(["编程环境", "开发工具", "云计算平台"])
        
        return list(set(resources))
    
    def _estimate_success_probability(self, complexity_score: float, 
                                    challenge_count: int) -> float:
        """评估成功概率"""
        # 基础成功概率
        base_probability = 0.8
        
        # 复杂度影响
        complexity_penalty = complexity_score * 0.3
        
        # 挑战数量影响
        challenge_penalty = min(0.4, challenge_count * 0.05)
        
        # 计算最终概率
        probability = base_probability - complexity_penalty - challenge_penalty
        
        return max(0.1, min(0.95, probability))  # 限制在0.1-0.95之间
    
    def generate_research_plan(self, analysis: TopicAnalysis) -> ResearchPlan:
        """
        基于主题分析生成详细研究计划
        
        Args:
            analysis: 主题分析结果
            
        Returns:
            ResearchPlan: 详细研究计划
        """
        try:
            self.log_operation("generate_research_plan", {
                "topic": analysis.topic,
                "complexity": analysis.complexity_level.value
            })
            
            # 验证输入
            if not analysis.validate():
                raise ValidationError("主题分析结果无效")
            
            # 生成时间线和里程碑
            timeline = self._generate_timeline(analysis)
            
            # 生成研究路径
            research_paths = self._generate_research_paths(analysis)
            
            # 生成资源分配计划
            resource_allocation = self._generate_resource_allocation(analysis)
            
            # 定义成功指标
            success_metrics = self._define_success_metrics(analysis)
            
            # 制定风险缓解策略
            risk_mitigation_strategies = self._develop_risk_mitigation_strategies(analysis)
            
            # 设置质量检查点
            quality_checkpoints = self._setup_quality_checkpoints(analysis)
            
            # 确定协作需求
            collaboration_requirements = self._identify_collaboration_requirements(analysis)
            
            # 创建研究计划
            plan = ResearchPlan(
                topic_analysis=analysis,
                timeline=timeline,
                research_paths=research_paths,
                resource_allocation=resource_allocation,
                success_metrics=success_metrics,
                risk_mitigation_strategies=risk_mitigation_strategies,
                quality_checkpoints=quality_checkpoints,
                collaboration_requirements=collaboration_requirements
            )
            
            # 验证计划
            if not plan.validate():
                raise ProcessingError("生成的研究计划验证失败")
            
            self.update_metric("plans_generated", self.get_metric("plans_generated") or 0 + 1)
            self.logger.info(f"研究计划生成完成: {len(timeline)}个里程碑, {len(research_paths)}条路径")
            
            return plan
            
        except Exception as e:
            self.handle_error(e, "generate_research_plan")
    
    def _generate_timeline(self, analysis: TopicAnalysis) -> List[Milestone]:
        """生成研究时间线和里程碑"""
        timeline = []
        start_date = datetime.now()
        
        # 基于复杂度和研究类型确定阶段
        phases = self._get_research_phases(analysis)
        
        current_date = start_date
        for i, (phase_name, phase_duration, deliverables) in enumerate(phases):
            milestone = Milestone(
                title=f"阶段 {i+1}: {phase_name}",
                description=f"完成{phase_name}相关工作",
                due_date=current_date + timedelta(days=phase_duration),
                deliverables=deliverables,
                status="not_started",
                assigned_roles=self._get_phase_roles(phase_name, analysis)
            )
            
            # 设置依赖关系
            if i > 0:
                milestone.dependencies = [timeline[i-1].id]
            
            timeline.append(milestone)
            current_date = milestone.due_date
        
        return timeline
    
    def _get_research_phases(self, analysis: TopicAnalysis) -> List[Tuple[str, int, List[str]]]:
        """获取研究阶段定义"""
        base_phases = [
            ("文献综述", 14, ["文献综述报告", "研究空白分析"]),
            ("方法设计", 21, ["方法论文档", "实验设计方案"]),
            ("数据准备", 14, ["数据集", "预处理脚本"]),
            ("实验执行", 28, ["实验结果", "性能评估"]),
            ("结果分析", 14, ["分析报告", "可视化图表"]),
            ("论文撰写", 21, ["论文初稿", "最终论文"])
        ]
        
        # 根据复杂度调整时长
        complexity_multipliers = {
            ResearchComplexity.LOW: 0.7,
            ResearchComplexity.MEDIUM: 1.0,
            ResearchComplexity.HIGH: 1.4,
            ResearchComplexity.VERY_HIGH: 1.8
        }
        
        multiplier = complexity_multipliers.get(analysis.complexity_level, 1.0)
        
        # 根据研究类型调整阶段
        if analysis.research_type == ResearchType.SURVEY:
            # 调研类研究更注重文献综述
            base_phases[0] = ("深度文献综述", int(28 * multiplier), 
                            ["系统性文献综述", "元分析", "研究趋势报告"])
            base_phases[2] = ("数据收集", int(7 * multiplier), ["调研数据", "统计分析"])
        elif analysis.research_type == ResearchType.THEORETICAL:
            # 理论研究更注重方法设计和分析
            base_phases[1] = ("理论建模", int(35 * multiplier), 
                            ["数学模型", "理论框架", "证明过程"])
            base_phases[3] = ("理论验证", int(21 * multiplier), ["验证结果", "理论分析"])
        
        # 应用时长调整
        adjusted_phases = []
        for phase_name, duration, deliverables in base_phases:
            adjusted_duration = max(7, int(duration * multiplier))  # 最少7天
            adjusted_phases.append((phase_name, adjusted_duration, deliverables))
        
        return adjusted_phases
    
    def _get_phase_roles(self, phase_name: str, analysis: TopicAnalysis) -> List[str]:
        """获取阶段所需角色"""
        role_mapping = {
            "文献综述": ["研究员", "信息专员"],
            "深度文献综述": ["高级研究员", "信息专员", "统计分析师"],
            "方法设计": ["方法论专家", "技术专家"],
            "理论建模": ["理论专家", "数学家"],
            "数据准备": ["数据工程师", "领域专家"],
            "数据收集": ["调研专员", "统计师"],
            "实验执行": ["实验员", "技术专家"],
            "理论验证": ["理论专家", "同行评议员"],
            "结果分析": ["数据分析师", "统计专家"],
            "论文撰写": ["科技写作专家", "领域专家"]
        }
        
        base_roles = role_mapping.get(phase_name, ["研究员"])
        
        # 根据复杂度添加额外角色
        if analysis.complexity_level in [ResearchComplexity.HIGH, ResearchComplexity.VERY_HIGH]:
            base_roles.append("项目经理")
        
        # 根据跨学科特征添加角色
        if len(analysis.related_fields) > 2:
            base_roles.append("跨学科协调员")
        
        return base_roles
    
    def _generate_research_paths(self, analysis: TopicAnalysis) -> List[ResearchPath]:
        """生成研究路径选项"""
        paths = []
        
        # 主要路径：基于研究类型的标准路径
        main_path = self._create_main_research_path(analysis)
        paths.append(main_path)
        
        # 替代路径：风险较低的保守路径
        conservative_path = self._create_conservative_path(analysis)
        paths.append(conservative_path)
        
        # 创新路径：高风险高回报路径
        if analysis.complexity_level in [ResearchComplexity.HIGH, ResearchComplexity.VERY_HIGH]:
            innovative_path = self._create_innovative_path(analysis)
            paths.append(innovative_path)
        
        # 快速路径：时间优化路径
        if analysis.estimated_duration > 90:  # 超过3个月的项目
            fast_track_path = self._create_fast_track_path(analysis)
            paths.append(fast_track_path)
        
        return paths
    
    def _create_main_research_path(self, analysis: TopicAnalysis) -> ResearchPath:
        """创建主要研究路径"""
        path_descriptions = {
            ResearchType.EXPERIMENTAL: "基于实验验证的标准研究路径，通过对照实验验证假设",
            ResearchType.THEORETICAL: "理论驱动的研究路径，重点在于理论建模和数学证明",
            ResearchType.SURVEY: "系统性调研路径，通过文献综述和元分析获得洞察",
            ResearchType.EMPIRICAL: "数据驱动的实证研究路径，基于大规模数据分析",
            ResearchType.CASE_STUDY: "案例研究路径，通过深入分析特定案例获得见解"
        }
        
        methodologies = {
            ResearchType.EXPERIMENTAL: "对照实验设计 + 统计分析",
            ResearchType.THEORETICAL: "数学建模 + 理论分析",
            ResearchType.SURVEY: "系统性文献综述 + 元分析",
            ResearchType.EMPIRICAL: "大数据分析 + 机器学习",
            ResearchType.CASE_STUDY: "定性分析 + 案例比较"
        }
        
        return ResearchPath(
            name="标准研究路径",
            description=path_descriptions.get(analysis.research_type, "标准研究方法"),
            methodology=methodologies.get(analysis.research_type, "混合方法"),
            expected_outcomes=analysis.suggested_directions[:3],
            risk_level=0.4,
            innovation_potential=0.6,
            resource_intensity=0.5,
            timeline_months=max(1, analysis.estimated_duration // 30),
            prerequisites=["文献调研完成", "方法论确定"],
            alternative_approaches=["简化版本", "分阶段实施"]
        )
    
    def _create_conservative_path(self, analysis: TopicAnalysis) -> ResearchPath:
        """创建保守研究路径"""
        return ResearchPath(
            name="保守研究路径",
            description="风险较低的研究路径，采用成熟方法和技术",
            methodology="成熟方法论 + 渐进式改进",
            expected_outcomes=["稳定的研究结果", "可重现的发现", "渐进式创新"],
            risk_level=0.2,
            innovation_potential=0.4,
            resource_intensity=0.3,
            timeline_months=max(1, int(analysis.estimated_duration * 0.8) // 30),
            prerequisites=["充分的文献基础", "成熟工具可用"],
            alternative_approaches=["分步验证", "小规模试点"]
        )
    
    def _create_innovative_path(self, analysis: TopicAnalysis) -> ResearchPath:
        """创建创新研究路径"""
        return ResearchPath(
            name="创新研究路径",
            description="高风险高回报的创新路径，探索前沿方法和技术",
            methodology="前沿技术 + 创新方法论",
            expected_outcomes=["突破性发现", "方法论创新", "领域推进"],
            risk_level=0.8,
            innovation_potential=0.9,
            resource_intensity=0.8,
            timeline_months=max(2, int(analysis.estimated_duration * 1.3) // 30),
            prerequisites=["充足资源支持", "专家团队", "风险承受能力"],
            alternative_approaches=["阶段性验证", "并行探索", "快速原型"]
        )
    
    def _create_fast_track_path(self, analysis: TopicAnalysis) -> ResearchPath:
        """创建快速路径"""
        return ResearchPath(
            name="快速研究路径",
            description="时间优化的研究路径，通过并行处理和资源集中加速进度",
            methodology="并行处理 + 敏捷方法",
            expected_outcomes=["快速验证", "初步结果", "概念验证"],
            risk_level=0.6,
            innovation_potential=0.5,
            resource_intensity=0.7,
            timeline_months=max(1, int(analysis.estimated_duration * 0.6) // 30),
            prerequisites=["充足人力资源", "并行处理能力"],
            alternative_approaches=["MVP方法", "迭代开发", "快速反馈"]
        )
    
    def _generate_resource_allocation(self, analysis: TopicAnalysis) -> Dict[str, Any]:
        """生成资源分配计划"""
        # 估算预算需求
        budget_estimate = self._estimate_budget_requirements(analysis)
        
        # 人力资源分配
        human_resources = self._allocate_human_resources(analysis)
        
        # 技术资源分配
        technical_resources = self._allocate_technical_resources(analysis)
        
        # 时间资源分配
        time_allocation = self._allocate_time_resources(analysis)
        
        return {
            "budget": budget_estimate,
            "human_resources": human_resources,
            "technical_resources": technical_resources,
            "time_allocation": time_allocation,
            "contingency_reserve": 0.15  # 15%的应急储备
        }
    
    def _estimate_budget_requirements(self, analysis: TopicAnalysis) -> Dict[str, float]:
        """估算预算需求"""
        # 基础成本（每天）
        base_cost_per_day = {
            ResearchComplexity.LOW: 200.0,
            ResearchComplexity.MEDIUM: 500.0,
            ResearchComplexity.HIGH: 1000.0,
            ResearchComplexity.VERY_HIGH: 1500.0
        }
        
        daily_cost = base_cost_per_day.get(analysis.complexity_level, 500.0)
        total_duration = analysis.estimated_duration
        team_size = analysis.estimate_team_size()
        
        # 人员成本
        personnel_cost = daily_cost * total_duration * team_size
        
        # 设备和软件成本
        equipment_cost = personnel_cost * 0.25
        
        # 数据和文献获取成本
        data_cost = personnel_cost * 0.1
        
        # 管理费用
        overhead_cost = (personnel_cost + equipment_cost + data_cost) * 0.2
        
        total_cost = personnel_cost + equipment_cost + data_cost + overhead_cost
        
        return {
            "personnel": personnel_cost,
            "equipment_software": equipment_cost,
            "data_literature": data_cost,
            "overhead": overhead_cost,
            "total": total_cost
        }
    
    def _allocate_human_resources(self, analysis: TopicAnalysis) -> Dict[str, Any]:
        """分配人力资源"""
        team_size = analysis.estimate_team_size()
        
        # 基础角色分配
        role_allocation = {
            "主研究员": 1,
            "研究助理": max(1, team_size - 2),
            "数据分析师": 1 if team_size > 2 else 0,
            "技术专家": 1 if analysis.complexity_level in [ResearchComplexity.HIGH, ResearchComplexity.VERY_HIGH] else 0,
            "项目经理": 1 if team_size > 4 else 0
        }
        
        # 根据研究类型调整
        if analysis.research_type == ResearchType.THEORETICAL:
            role_allocation["理论专家"] = 1
        elif analysis.research_type == ResearchType.EXPERIMENTAL:
            role_allocation["实验员"] = max(1, team_size // 3)
        
        # 工作量分配（百分比）
        workload_distribution = {
            "文献综述": 15,
            "方法设计": 20,
            "数据处理": 25,
            "实验执行": 25,
            "结果分析": 10,
            "论文撰写": 5
        }
        
        return {
            "team_size": team_size,
            "role_allocation": role_allocation,
            "workload_distribution": workload_distribution,
            "collaboration_model": "混合协作" if team_size > 3 else "紧密协作"
        }
    
    def _allocate_technical_resources(self, analysis: TopicAnalysis) -> Dict[str, Any]:
        """分配技术资源"""
        resources = {
            "computing_power": "标准",
            "software_licenses": [],
            "data_storage": "云存储",
            "development_tools": []
        }
        
        # 根据复杂度调整计算资源
        if analysis.complexity_level in [ResearchComplexity.HIGH, ResearchComplexity.VERY_HIGH]:
            resources["computing_power"] = "高性能"
            resources["data_storage"] = "高性能云存储"
        
        # 根据领域添加专业软件
        if 'Computer Science' in analysis.related_fields:
            resources["software_licenses"].extend(["Python", "R", "MATLAB"])
            resources["development_tools"].extend(["IDE", "版本控制", "CI/CD"])
        
        if 'Mathematics' in analysis.related_fields:
            resources["software_licenses"].extend(["Mathematica", "MATLAB", "R"])
        
        if 'Biology' in analysis.related_fields:
            resources["software_licenses"].extend(["Bioinformatics Suite", "统计软件"])
        
        return resources
    
    def _allocate_time_resources(self, analysis: TopicAnalysis) -> Dict[str, int]:
        """分配时间资源"""
        total_duration = analysis.estimated_duration
        
        # 基础时间分配（百分比）
        time_allocation = {
            "文献综述": int(total_duration * 0.15),
            "方法设计": int(total_duration * 0.20),
            "数据准备": int(total_duration * 0.15),
            "实验执行": int(total_duration * 0.30),
            "结果分析": int(total_duration * 0.15),
            "论文撰写": int(total_duration * 0.05)
        }
        
        # 根据研究类型调整
        if analysis.research_type == ResearchType.SURVEY:
            time_allocation["文献综述"] = int(total_duration * 0.40)
            time_allocation["实验执行"] = int(total_duration * 0.10)
        elif analysis.research_type == ResearchType.THEORETICAL:
            time_allocation["方法设计"] = int(total_duration * 0.35)
            time_allocation["数据准备"] = int(total_duration * 0.05)
        
        return time_allocation
    
    def _define_success_metrics(self, analysis: TopicAnalysis) -> List[str]:
        """定义成功指标"""
        metrics = []
        
        # 基础指标
        metrics.extend([
            "按时完成各阶段里程碑",
            "达到预期的研究质量标准",
            "产出符合要求的研究成果"
        ])
        
        # 根据研究类型添加特定指标
        if analysis.research_type == ResearchType.EXPERIMENTAL:
            metrics.extend([
                "实验结果具有统计显著性",
                "实验可重现性达到95%以上",
                "性能指标达到或超过基准"
            ])
        elif analysis.research_type == ResearchType.THEORETICAL:
            metrics.extend([
                "理论模型通过同行评议",
                "数学证明严格完整",
                "理论贡献得到认可"
            ])
        elif analysis.research_type == ResearchType.SURVEY:
            metrics.extend([
                "文献覆盖率达到90%以上",
                "识别出关键研究空白",
                "提供有价值的研究方向建议"
            ])
        
        # 根据复杂度添加指标
        if analysis.complexity_level in [ResearchComplexity.HIGH, ResearchComplexity.VERY_HIGH]:
            metrics.extend([
                "跨学科协作效果良好",
                "创新性得到专家认可",
                "研究影响力达到预期"
            ])
        
        return metrics
    
    def _develop_risk_mitigation_strategies(self, analysis: TopicAnalysis) -> List[str]:
        """制定风险缓解策略"""
        strategies = []
        
        # 针对潜在挑战的策略
        challenge_strategies = {
            "技术复杂性": "分阶段实施，建立技术原型验证",
            "资源协调": "建立资源共享机制，制定备用方案",
            "跨领域知识整合": "组建跨学科团队，定期知识分享",
            "项目管理复杂性": "采用敏捷项目管理方法，设置检查点",
            "团队协作": "建立有效沟通机制，定期团队建设",
            "时间管理": "制定详细时间计划，设置缓冲时间",
            "数据质量": "建立数据质量检查流程，准备备用数据源",
            "方法选择": "进行方法论预研究，准备多种方案",
            "结果解释": "邀请领域专家参与，进行同行评议"
        }
        
        for challenge in analysis.potential_challenges:
            if challenge in challenge_strategies:
                strategies.append(challenge_strategies[challenge])
        
        # 通用风险缓解策略
        strategies.extend([
            "建立定期进度评估机制",
            "设置质量检查点",
            "准备应急预案",
            "建立外部专家咨询网络"
        ])
        
        return list(set(strategies))
    
    def _setup_quality_checkpoints(self, analysis: TopicAnalysis) -> List[str]:
        """设置质量检查点"""
        checkpoints = []
        
        # 基础检查点
        checkpoints.extend([
            "文献综述质量评估",
            "方法论可行性评估",
            "中期进度和质量评估",
            "结果有效性验证",
            "最终成果质量评估"
        ])
        
        # 根据研究类型添加特定检查点
        if analysis.research_type == ResearchType.EXPERIMENTAL:
            checkpoints.extend([
                "实验设计同行评议",
                "实验结果可重现性检查",
                "统计分析有效性验证"
            ])
        elif analysis.research_type == ResearchType.THEORETICAL:
            checkpoints.extend([
                "理论模型逻辑一致性检查",
                "数学证明严格性评估",
                "理论贡献新颖性评估"
            ])
        
        # 根据复杂度添加检查点
        if analysis.complexity_level in [ResearchComplexity.HIGH, ResearchComplexity.VERY_HIGH]:
            checkpoints.extend([
                "跨学科整合效果评估",
                "创新性和影响力评估",
                "长期可持续性评估"
            ])
        
        return checkpoints
    
    def _identify_collaboration_requirements(self, analysis: TopicAnalysis) -> List[str]:
        """确定协作需求"""
        requirements = []
        
        # 基础协作需求
        requirements.extend([
            "团队内部定期沟通",
            "进度同步和协调",
            "知识和资源共享"
        ])
        
        # 根据团队规模确定协作需求
        team_size = analysis.estimate_team_size()
        if team_size > 3:
            requirements.extend([
                "项目管理工具使用",
                "任务分配和跟踪",
                "冲突解决机制"
            ])
        
        # 根据跨学科特征确定协作需求
        if len(analysis.related_fields) > 2:
            requirements.extend([
                "跨学科知识整合",
                "术语和概念统一",
                "不同方法论的协调"
            ])
        
        # 根据复杂度确定协作需求
        if analysis.complexity_level in [ResearchComplexity.HIGH, ResearchComplexity.VERY_HIGH]:
            requirements.extend([
                "外部专家咨询",
                "同行评议和反馈",
                "国际合作可能性"
            ])
        
        return requirements
    
    def suggest_research_paths(self, plan: ResearchPlan) -> List[ResearchPath]:
        """
        基于研究计划建议研究路径
        
        Args:
            plan: 研究计划
            
        Returns:
            List[ResearchPath]: 推荐的研究路径列表
        """
        try:
            self.log_operation("suggest_research_paths", {
                "plan_id": plan.id,
                "paths_count": len(plan.research_paths)
            })
            
            # 验证输入
            if not plan.validate():
                raise ValidationError("研究计划无效")
            
            # 按优先级排序现有路径
            sorted_paths = sorted(
                plan.research_paths,
                key=lambda p: p.calculate_priority_score(),
                reverse=True
            )
            
            # 添加路径选择建议
            for path in sorted_paths:
                path.selection_rationale = self._generate_path_rationale(path, plan.topic_analysis)
            
            self.logger.info(f"路径建议完成: 推荐{len(sorted_paths)}条路径")
            return sorted_paths
            
        except Exception as e:
            self.handle_error(e, "suggest_research_paths")
    
    def _generate_path_rationale(self, path: ResearchPath, analysis: TopicAnalysis) -> str:
        """生成路径选择理由"""
        rationale_parts = []
        
        # 基于优先级分数
        priority_score = path.calculate_priority_score()
        if priority_score > 0.7:
            rationale_parts.append("高优先级路径，综合评分优秀")
        elif priority_score > 0.5:
            rationale_parts.append("中等优先级路径，平衡风险和收益")
        else:
            rationale_parts.append("低优先级路径，适合特定情况")
        
        # 基于风险水平
        if path.risk_level < 0.3:
            rationale_parts.append("风险较低，适合稳妥推进")
        elif path.risk_level > 0.7:
            rationale_parts.append("高风险高回报，需要充分准备")
        
        # 基于创新潜力
        if path.innovation_potential > 0.7:
            rationale_parts.append("创新潜力高，可能带来突破性成果")
        
        # 基于资源强度
        if path.resource_intensity > 0.7:
            rationale_parts.append("资源需求较高，需要充足支持")
        elif path.resource_intensity < 0.4:
            rationale_parts.append("资源需求适中，实施相对容易")
        
        return "；".join(rationale_parts)
    
    def refine_scope(self, topic: str) -> List[str]:
        """
        细化研究范围建议
        
        Args:
            topic: 研究主题
            
        Returns:
            List[str]: 细化的研究方向建议
        """
        try:
            self.log_operation("refine_scope", {"topic": topic})
            
            # 验证输入
            if not topic or not topic.strip():
                raise ValidationError("研究主题不能为空")
            
            # 分析主题获取基础信息
            analysis = self.analyze_topic(topic)
            
            # 生成细化建议
            refined_suggestions = []
            
            # 基于复杂度的细化建议
            if analysis.complexity_level == ResearchComplexity.VERY_HIGH:
                refined_suggestions.extend([
                    "考虑将研究分解为多个子问题",
                    "优先选择最核心的研究问题",
                    "建立阶段性目标和里程碑"
                ])
            elif analysis.complexity_level == ResearchComplexity.LOW:
                refined_suggestions.extend([
                    "考虑扩展研究范围以增加深度",
                    "探索相关的延伸问题",
                    "增加比较分析的维度"
                ])
            
            # 基于研究范围的建议
            if len(analysis.research_scope) > 5:
                refined_suggestions.append("研究范围较广，建议聚焦于2-3个核心领域")
            elif len(analysis.research_scope) < 3:
                refined_suggestions.append("研究范围可以适当扩展，增加研究的全面性")
            
            # 基于跨学科特征的建议
            if len(analysis.related_fields) > 3:
                refined_suggestions.append("跨学科研究，建议明确各领域的贡献和整合方式")
            
            # 具体的范围细化建议
            specific_suggestions = self._generate_specific_scope_suggestions(analysis)
            refined_suggestions.extend(specific_suggestions)
            
            self.logger.info(f"范围细化完成: 生成{len(refined_suggestions)}条建议")
            return refined_suggestions
            
        except Exception as e:
            self.handle_error(e, "refine_scope")
    
    def _generate_specific_scope_suggestions(self, analysis: TopicAnalysis) -> List[str]:
        """生成具体的范围细化建议"""
        suggestions = []
        
        # 基于关键词生成具体建议
        key_terms = analysis.keywords[:5]  # 取前5个关键词
        
        for term in key_terms:
            if 'machine learning' in term:
                suggestions.extend([
                    f"聚焦于特定的机器学习算法类型（如监督学习、无监督学习）",
                    f"明确应用领域和数据类型",
                    f"定义性能评估指标和基准"
                ])
            elif 'deep learning' in term:
                suggestions.extend([
                    f"选择特定的深度学习架构（如CNN、RNN、Transformer）",
                    f"明确任务类型（分类、回归、生成等）",
                    f"确定数据规模和计算资源需求"
                ])
            elif 'optimization' in term:
                suggestions.extend([
                    f"明确优化问题的类型和约束条件",
                    f"选择合适的优化算法类别",
                    f"定义收敛标准和性能指标"
                ])
        
        # 基于研究类型的建议
        if analysis.research_type == ResearchType.SURVEY:
            suggestions.extend([
                "明确文献检索的时间范围和数据库范围",
                "定义文献筛选和质量评估标准",
                "确定综述的结构和分析框架"
            ])
        elif analysis.research_type == ResearchType.EXPERIMENTAL:
            suggestions.extend([
                "明确实验的自变量和因变量",
                "定义对照组和实验组的设置",
                "确定样本大小和统计功效"
            ])
        
        return list(set(suggestions))  # 去重
    

    
    def _generate_primary_directions(self, analysis: TopicAnalysis, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成主要研究方向"""
        directions = []
        
        # 基于研究类型的主要方向
        type_directions = {
            ResearchType.EXPERIMENTAL: [
                "性能优化与基准测试",
                "算法改进与创新",
                "实验设计与验证"
            ],
            ResearchType.THEORETICAL: [
                "理论框架构建",
                "数学模型开发",
                "理论证明与分析"
            ],
            ResearchType.SURVEY: [
                "系统性文献综述",
                "技术发展趋势分析",
                "研究空白识别"
            ],
            ResearchType.EMPIRICAL: [
                "大规模数据分析",
                "实证研究设计",
                "统计模型构建"
            ],
            ResearchType.CASE_STUDY: [
                "典型案例深度分析",
                "最佳实践总结",
                "案例比较研究"
            ]
        }
        
        base_directions = type_directions.get(analysis.research_type, ["综合研究方法"])
        
        for direction in base_directions:
            directions.append({
                "title": direction,
                "description": self._generate_direction_description(direction, analysis),
                "feasibility": self._assess_direction_feasibility(direction, analysis),
                "innovation_potential": self._assess_innovation_potential(direction, analysis),
                "resource_requirements": self._estimate_direction_resources(direction, analysis),
                "expected_timeline": self._estimate_direction_timeline(direction, analysis),
                "success_probability": self._estimate_direction_success_probability(direction, analysis)
            })
        
        return directions
    
    def _generate_alternative_directions(self, analysis: TopicAnalysis, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成替代研究方向"""
        directions = []
        
        # 基于关键词生成替代方向
        for keyword in analysis.keywords[:5]:
            if 'machine learning' in keyword:
                alt_directions = [
                    "传统机器学习方法比较研究",
                    "机器学习可解释性研究",
                    "机器学习鲁棒性分析"
                ]
            elif 'deep learning' in keyword:
                alt_directions = [
                    "轻量级深度学习模型",
                    "深度学习迁移学习",
                    "深度学习压缩技术"
                ]
            elif 'optimization' in keyword:
                alt_directions = [
                    "多目标优化方法",
                    "约束优化技术",
                    "启发式优化算法"
                ]
            else:
                alt_directions = [f"{keyword}的替代方法研究"]
            
            for direction in alt_directions[:2]:  # 每个关键词最多2个替代方向
                directions.append({
                    "title": direction,
                    "description": self._generate_direction_description(direction, analysis),
                    "feasibility": self._assess_direction_feasibility(direction, analysis),
                    "innovation_potential": self._assess_innovation_potential(direction, analysis),
                    "resource_requirements": self._estimate_direction_resources(direction, analysis),
                    "expected_timeline": self._estimate_direction_timeline(direction, analysis),
                    "success_probability": self._estimate_direction_success_probability(direction, analysis)
                })
        
        return directions[:6]  # 最多6个替代方向
    
    def _generate_emerging_directions(self, analysis: TopicAnalysis, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成新兴研究方向"""
        directions = []
        
        # 新兴技术方向
        emerging_topics = [
            "大语言模型应用",
            "联邦学习技术",
            "边缘计算优化",
            "量子机器学习",
            "神经符号推理",
            "可解释AI技术",
            "多模态学习",
            "自监督学习"
        ]
        
        # 根据研究主题选择相关的新兴方向
        relevant_emerging = []
        topic_lower = analysis.topic.lower()
        
        for emerging in emerging_topics:
            # 简单的相关性判断
            emerging_keywords = emerging.lower().split()
            if any(keyword in topic_lower for keyword in emerging_keywords):
                relevant_emerging.append(emerging)
        
        # 如果没有直接相关的，选择通用新兴方向
        if not relevant_emerging:
            relevant_emerging = emerging_topics[:3]
        
        for direction in relevant_emerging[:4]:  # 最多4个新兴方向
            directions.append({
                "title": f"{direction}在{analysis.topic}中的应用",
                "description": self._generate_direction_description(direction, analysis),
                "feasibility": max(0.3, self._assess_direction_feasibility(direction, analysis) - 0.2),  # 新兴方向可行性稍低
                "innovation_potential": min(0.9, self._assess_innovation_potential(direction, analysis) + 0.2),  # 创新性更高
                "resource_requirements": self._estimate_direction_resources(direction, analysis),
                "expected_timeline": int(self._estimate_direction_timeline(direction, analysis) * 1.3),  # 时间稍长
                "success_probability": max(0.2, self._estimate_direction_success_probability(direction, analysis) - 0.1)
            })
        
        return directions
    
    def _generate_interdisciplinary_directions(self, analysis: TopicAnalysis, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成跨学科研究方向"""
        directions = []
        
        if len(analysis.related_fields) < 2:
            return directions  # 非跨学科研究不生成此类方向
        
        # 跨学科组合
        field_combinations = [
            ("Computer Science", "Biology", "生物信息学应用"),
            ("Computer Science", "Medicine", "医疗AI技术"),
            ("Mathematics", "Physics", "计算物理方法"),
            ("Engineering", "Computer Science", "智能系统设计"),
            ("Computer Science", "Psychology", "认知计算模型"),
            ("Mathematics", "Economics", "计算经济学"),
            ("Computer Science", "Education", "智能教育技术")
        ]
        
        for field1, field2, direction_name in field_combinations:
            if field1 in analysis.related_fields and field2 in analysis.related_fields:
                directions.append({
                    "title": f"{direction_name}研究",
                    "description": f"结合{field1}和{field2}的跨学科研究方向",
                    "feasibility": self._assess_direction_feasibility(direction_name, analysis),
                    "innovation_potential": min(0.9, self._assess_innovation_potential(direction_name, analysis) + 0.15),
                    "resource_requirements": self._estimate_direction_resources(direction_name, analysis),
                    "expected_timeline": int(self._estimate_direction_timeline(direction_name, analysis) * 1.2),
                    "success_probability": self._estimate_direction_success_probability(direction_name, analysis),
                    "collaboration_requirements": ["跨学科专家", "领域知识整合", "方法论协调"]
                })
        
        return directions[:3]  # 最多3个跨学科方向
    
    def _generate_practical_applications(self, analysis: TopicAnalysis, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成实际应用方向"""
        directions = []
        
        # 应用领域映射
        application_domains = {
            'machine learning': ['医疗诊断', '金融风控', '智能制造', '自动驾驶'],
            'deep learning': ['图像识别', '语音处理', '自然语言理解', '推荐系统'],
            'optimization': ['供应链优化', '资源调度', '路径规划', '参数调优'],
            'data mining': ['商业智能', '用户行为分析', '市场预测', '异常检测'],
            'computer vision': ['医学影像', '安防监控', '工业检测', '增强现实'],
            'natural language processing': ['智能客服', '文档分析', '机器翻译', '情感分析']
        }
        
        # 根据关键词匹配应用领域
        relevant_applications = []
        for keyword in analysis.keywords:
            for tech, apps in application_domains.items():
                if tech in keyword.lower():
                    relevant_applications.extend(apps)
        
        # 去重并限制数量
        relevant_applications = list(set(relevant_applications))[:5]
        
        for app in relevant_applications:
            directions.append({
                "title": f"{app}应用研究",
                "description": f"将研究成果应用于{app}领域的实际问题解决",
                "feasibility": self._assess_direction_feasibility(app, analysis),
                "innovation_potential": self._assess_innovation_potential(app, analysis),
                "resource_requirements": self._estimate_direction_resources(app, analysis),
                "expected_timeline": self._estimate_direction_timeline(app, analysis),
                "success_probability": self._estimate_direction_success_probability(app, analysis),
                "market_potential": self._assess_market_potential(app),
                "industry_relevance": self._assess_industry_relevance(app)
            })
        
        return directions
    
    def _generate_theoretical_extensions(self, analysis: TopicAnalysis, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成理论扩展方向"""
        directions = []
        
        # 理论扩展类型
        theoretical_extensions = [
            "理论基础强化",
            "数学模型扩展",
            "算法理论分析",
            "复杂度理论研究",
            "收敛性分析",
            "稳定性理论",
            "泛化理论研究"
        ]
        
        # 根据研究类型选择相关的理论扩展
        if analysis.research_type in [ResearchType.THEORETICAL, ResearchType.EXPERIMENTAL]:
            selected_extensions = theoretical_extensions[:4]
        else:
            selected_extensions = theoretical_extensions[:2]
        
        for extension in selected_extensions:
            directions.append({
                "title": f"{extension}研究",
                "description": self._generate_direction_description(extension, analysis),
                "feasibility": self._assess_direction_feasibility(extension, analysis),
                "innovation_potential": self._assess_innovation_potential(extension, analysis),
                "resource_requirements": self._estimate_direction_resources(extension, analysis),
                "expected_timeline": self._estimate_direction_timeline(extension, analysis),
                "success_probability": self._estimate_direction_success_probability(extension, analysis),
                "theoretical_depth": "高",
                "mathematical_complexity": self._assess_mathematical_complexity(extension)
            })
        
        return directions
    
    def _generate_methodology_innovations(self, analysis: TopicAnalysis, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成方法论创新方向"""
        directions = []
        
        # 方法论创新类型
        methodology_innovations = [
            "新型算法设计",
            "混合方法论开发",
            "评估指标创新",
            "实验设计改进",
            "数据处理方法创新",
            "可视化技术创新",
            "自动化工具开发"
        ]
        
        # 根据复杂度选择创新方向数量
        if analysis.complexity_level in [ResearchComplexity.HIGH, ResearchComplexity.VERY_HIGH]:
            selected_innovations = methodology_innovations[:5]
        else:
            selected_innovations = methodology_innovations[:3]
        
        for innovation in selected_innovations:
            directions.append({
                "title": f"{innovation}",
                "description": self._generate_direction_description(innovation, analysis),
                "feasibility": self._assess_direction_feasibility(innovation, analysis),
                "innovation_potential": min(0.95, self._assess_innovation_potential(innovation, analysis) + 0.1),
                "resource_requirements": self._estimate_direction_resources(innovation, analysis),
                "expected_timeline": self._estimate_direction_timeline(innovation, analysis),
                "success_probability": self._estimate_direction_success_probability(innovation, analysis),
                "novelty_level": "高",
                "technical_challenge": self._assess_technical_challenge(innovation)
            })
        
        return directions    

    def _generate_direction_selection_guidance(self, directions: Dict[str, List[Dict[str, Any]]], 
                                             analysis: TopicAnalysis, 
                                             context: Dict[str, Any]) -> Dict[str, Any]:
        """生成方向选择指导"""
        guidance = {
            "selection_criteria": self._define_selection_criteria(analysis),
            "decision_framework": self._create_decision_framework(analysis),
            "risk_assessment": self._assess_overall_risks(directions, analysis),
            "resource_planning": self._create_resource_planning_guidance(directions, analysis),
            "timeline_considerations": self._create_timeline_guidance(directions, analysis),
            "success_factors": self._identify_success_factors(analysis)
        }
        
        return guidance
    
    def _define_selection_criteria(self, analysis: TopicAnalysis) -> List[Dict[str, Any]]:
        """定义方向选择标准"""
        criteria = [
            {
                "name": "可行性",
                "weight": 0.3,
                "description": "研究方向的技术和资源可行性",
                "evaluation_method": "基于现有资源和技术能力评估"
            },
            {
                "name": "创新性",
                "weight": 0.25,
                "description": "研究方向的创新潜力和学术价值",
                "evaluation_method": "基于现有文献和技术发展趋势评估"
            },
            {
                "name": "影响力",
                "weight": 0.2,
                "description": "研究成果的潜在影响和应用价值",
                "evaluation_method": "基于应用前景和社会需求评估"
            },
            {
                "name": "资源匹配度",
                "weight": 0.15,
                "description": "研究方向与现有资源的匹配程度",
                "evaluation_method": "基于人力、财力、时间资源评估"
            },
            {
                "name": "风险控制",
                "weight": 0.1,
                "description": "研究风险的可控性和缓解措施",
                "evaluation_method": "基于风险识别和缓解策略评估"
            }
        ]
        
        # 根据研究复杂度调整权重
        if analysis.complexity_level in [ResearchComplexity.HIGH, ResearchComplexity.VERY_HIGH]:
            # 复杂研究更注重可行性和风险控制
            for criterion in criteria:
                if criterion["name"] == "可行性":
                    criterion["weight"] = 0.35
                elif criterion["name"] == "风险控制":
                    criterion["weight"] = 0.15
                elif criterion["name"] == "创新性":
                    criterion["weight"] = 0.2
        
        return criteria
    
    def _create_decision_framework(self, analysis: TopicAnalysis) -> Dict[str, Any]:
        """创建决策框架"""
        framework = {
            "decision_process": [
                "1. 评估个人/团队能力和资源",
                "2. 明确研究目标和期望成果",
                "3. 分析各方向的优劣势",
                "4. 考虑时间和资源约束",
                "5. 评估风险承受能力",
                "6. 做出初步选择",
                "7. 制定备选方案",
                "8. 定期评估和调整"
            ],
            "key_questions": [
                "这个方向是否符合我的研究兴趣和专业背景？",
                "我是否具备完成这个方向研究的必要技能？",
                "这个方向的研究成果是否有实际应用价值？",
                "我是否有足够的时间和资源支持？",
                "这个方向的风险是否在可接受范围内？",
                "是否有合适的导师或合作伙伴？"
            ],
            "decision_matrix": self._create_decision_matrix_template()
        }
        
        return framework
    
    def _create_decision_matrix_template(self) -> Dict[str, Any]:
        """创建决策矩阵模板"""
        return {
            "description": "使用1-5分评估每个方向在各标准上的表现",
            "scoring_guide": {
                "1": "很差/很低",
                "2": "较差/较低", 
                "3": "一般/中等",
                "4": "较好/较高",
                "5": "很好/很高"
            },
            "calculation_method": "加权平均分 = Σ(标准分数 × 权重)",
            "interpretation": {
                "4.0-5.0": "强烈推荐",
                "3.0-3.9": "推荐",
                "2.0-2.9": "需要谨慎考虑",
                "1.0-1.9": "不推荐"
            }
        }
    
    def _assess_overall_risks(self, directions: Dict[str, List[Dict[str, Any]]], 
                            analysis: TopicAnalysis) -> Dict[str, Any]:
        """评估整体风险"""
        all_directions = []
        for category_directions in directions.values():
            all_directions.extend(category_directions)
        
        risk_summary = {
            "high_risk_directions": [],
            "medium_risk_directions": [],
            "low_risk_directions": [],
            "common_risks": [],
            "mitigation_strategies": []
        }
        
        # 分类方向风险
        for direction in all_directions:
            risk_factors = direction.get("risk_factors", [])
            if len(risk_factors) > 3 or any("高" in risk for risk in risk_factors):
                risk_summary["high_risk_directions"].append(direction["title"])
            elif len(risk_factors) > 1:
                risk_summary["medium_risk_directions"].append(direction["title"])
            else:
                risk_summary["low_risk_directions"].append(direction["title"])
        
        # 识别共同风险
        all_risks = []
        for direction in all_directions:
            all_risks.extend(direction.get("risk_factors", []))
        
        risk_counter = Counter(all_risks)
        risk_summary["common_risks"] = [risk for risk, count in risk_counter.most_common(5)]
        
        # 通用缓解策略
        risk_summary["mitigation_strategies"] = [
            "建立定期进度评估机制",
            "准备备选方案和应急计划",
            "寻求专家指导和同行评议",
            "分阶段实施，降低单次风险",
            "建立风险预警和响应机制"
        ]
        
        return risk_summary
    
    def _create_resource_planning_guidance(self, directions: Dict[str, List[Dict[str, Any]]], 
                                         analysis: TopicAnalysis) -> Dict[str, Any]:
        """创建资源规划指导"""
        guidance = {
            "resource_assessment": {
                "current_resources": "评估现有的人力、财力、技术资源",
                "resource_gaps": "识别各研究方向的资源缺口",
                "acquisition_plan": "制定资源获取和配置计划"
            },
            "resource_optimization": [
                "优先选择资源需求与现有能力匹配的方向",
                "考虑资源共享和复用的可能性",
                "评估外部资源获取的可行性",
                "制定资源使用的优先级排序"
            ],
            "budget_planning": self._create_budget_planning_template(analysis),
            "team_building": self._create_team_building_guidance(analysis)
        }
        
        return guidance
    
    def _create_budget_planning_template(self, analysis: TopicAnalysis) -> Dict[str, Any]:
        """创建预算规划模板"""
        return {
            "budget_categories": [
                "人员费用（工资、津贴）",
                "设备费用（硬件、软件）",
                "材料费用（数据、文献）",
                "差旅费用（会议、调研）",
                "其他费用（管理、应急）"
            ],
            "budget_allocation_guide": {
                "人员费用": "通常占总预算的60-70%",
                "设备费用": "通常占总预算的15-25%",
                "材料费用": "通常占总预算的5-10%",
                "差旅费用": "通常占总预算的3-8%",
                "其他费用": "通常占总预算的5-10%"
            },
            "cost_estimation_factors": [
                "研究复杂度和时长",
                "团队规模和专业水平",
                "技术要求和设备需求",
                "数据获取和处理成本",
                "不确定性和风险因素"
            ]
        }
    
    def _create_team_building_guidance(self, analysis: TopicAnalysis) -> Dict[str, Any]:
        """创建团队建设指导"""
        return {
            "team_composition": {
                "core_roles": ["项目负责人", "技术专家", "数据分析师"],
                "support_roles": ["研究助理", "文献专员", "项目协调员"],
                "advisory_roles": ["领域专家", "方法论顾问", "同行评议员"]
            },
            "skill_requirements": self._identify_required_skills(analysis),
            "collaboration_model": self._suggest_collaboration_model(analysis),
            "team_development": [
                "明确角色分工和责任",
                "建立有效沟通机制",
                "制定团队协作规范",
                "定期进行团队建设活动"
            ]
        }
    
    def _identify_required_skills(self, analysis: TopicAnalysis) -> List[str]:
        """识别所需技能"""
        skills = ["研究方法论", "数据分析", "学术写作"]
        
        # 基于研究类型添加技能
        if analysis.research_type == ResearchType.EXPERIMENTAL:
            skills.extend(["实验设计", "统计分析", "结果解释"])
        elif analysis.research_type == ResearchType.THEORETICAL:
            skills.extend(["数学建模", "理论分析", "逻辑推理"])
        elif analysis.research_type == ResearchType.SURVEY:
            skills.extend(["文献检索", "系统综述", "元分析"])
        
        # 基于关键词添加技能
        for keyword in analysis.keywords[:5]:
            if 'machine learning' in keyword:
                skills.extend(["机器学习", "编程", "算法设计"])
            elif 'data' in keyword:
                skills.extend(["数据挖掘", "数据可视化", "数据库管理"])
        
        return list(set(skills))
    
    def _suggest_collaboration_model(self, analysis: TopicAnalysis) -> str:
        """建议协作模式"""
        team_size = analysis.estimate_team_size()
        
        if team_size <= 2:
            return "紧密协作模式：小团队密切合作，频繁沟通"
        elif team_size <= 5:
            return "分工协作模式：明确分工，定期同步进展"
        else:
            return "层级协作模式：分层管理，专业化分工"
    
    def _create_timeline_guidance(self, directions: Dict[str, List[Dict[str, Any]]], 
                                analysis: TopicAnalysis) -> Dict[str, Any]:
        """创建时间线指导"""
        return {
            "timeline_planning": [
                "设定明确的阶段性目标",
                "预留充足的缓冲时间",
                "考虑外部依赖和约束",
                "建立里程碑检查机制"
            ],
            "time_management": [
                "使用项目管理工具跟踪进度",
                "定期评估和调整时间安排",
                "识别关键路径和瓶颈",
                "平衡质量和进度要求"
            ],
            "scheduling_tips": [
                "将复杂任务分解为小步骤",
                "为不确定性预留20-30%缓冲时间",
                "考虑团队成员的其他承诺",
                "设置定期检查点和调整机会"
            ]
        }
    
    def _identify_success_factors(self, analysis: TopicAnalysis) -> List[Dict[str, str]]:
        """识别成功因素"""
        factors = [
            {
                "factor": "明确的研究目标",
                "description": "设定具体、可测量、可达成的研究目标"
            },
            {
                "factor": "充分的前期准备",
                "description": "深入的文献调研和方法论准备"
            },
            {
                "factor": "合适的团队配置",
                "description": "具备必要技能和经验的团队成员"
            },
            {
                "factor": "有效的项目管理",
                "description": "科学的计划制定和进度控制"
            },
            {
                "factor": "持续的质量监控",
                "description": "定期的质量检查和改进机制"
            }
        ]
        
        # 基于复杂度添加特定成功因素
        if analysis.complexity_level in [ResearchComplexity.HIGH, ResearchComplexity.VERY_HIGH]:
            factors.extend([
                {
                    "factor": "强有力的领导",
                    "description": "有经验的项目负责人和决策机制"
                },
                {
                    "factor": "外部支持网络",
                    "description": "专家顾问和同行合作网络"
                }
            ])
        
        return factors
    
    def _generate_personalized_recommendations(self, directions: Dict[str, List[Dict[str, Any]]], 
                                             analysis: TopicAnalysis, 
                                             context: Dict[str, Any]) -> Dict[str, Any]:
        """生成个性化推荐"""
        recommendations = {
            "top_recommendations": self._select_top_recommendations(directions, analysis, context),
            "alternative_options": self._select_alternative_options(directions, analysis, context),
            "learning_path": self._create_learning_path(directions, analysis, context),
            "next_steps": self._suggest_next_steps(analysis, context)
        }
        
        return recommendations
    
    def _select_top_recommendations(self, directions: Dict[str, List[Dict[str, Any]]], 
                                  analysis: TopicAnalysis, 
                                  context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """选择顶级推荐"""
        all_directions = []
        for category, direction_list in directions.items():
            for direction in direction_list:
                direction["category"] = category
                all_directions.append(direction)
        
        # 按优先级分数排序
        sorted_directions = sorted(all_directions, 
                                 key=lambda x: x.get("priority_score", 0), 
                                 reverse=True)
        
        # 选择前3个作为顶级推荐
        top_3 = sorted_directions[:3]
        
        for direction in top_3:
            direction["recommendation_reason"] = self._generate_recommendation_reason(direction, analysis)
        
        return top_3
    
    def _generate_recommendation_reason(self, direction: Dict[str, Any], analysis: TopicAnalysis) -> str:
        """生成推荐理由"""
        reasons = []
        
        priority_score = direction.get("priority_score", 0)
        if priority_score > 0.7:
            reasons.append("综合评分优秀")
        
        feasibility = direction.get("feasibility", 0)
        if feasibility > 0.7:
            reasons.append("可行性高")
        
        innovation = direction.get("innovation_potential", 0)
        if innovation > 0.7:
            reasons.append("创新潜力大")
        
        success_prob = direction.get("success_probability", 0)
        if success_prob > 0.7:
            reasons.append("成功概率高")
        
        if not reasons:
            reasons.append("平衡了各项因素")
        
        return "；".join(reasons)
    
    def _select_alternative_options(self, directions: Dict[str, List[Dict[str, Any]]], 
                                  analysis: TopicAnalysis, 
                                  context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """选择替代选项"""
        all_directions = []
        for category, direction_list in directions.items():
            for direction in direction_list:
                direction["category"] = category
                all_directions.append(direction)
        
        # 按优先级分数排序，选择4-6名作为替代选项
        sorted_directions = sorted(all_directions, 
                                 key=lambda x: x.get("priority_score", 0), 
                                 reverse=True)
        
        alternatives = sorted_directions[3:6]
        
        for direction in alternatives:
            direction["alternative_reason"] = self._generate_alternative_reason(direction)
        
        return alternatives
    
    def _generate_alternative_reason(self, direction: Dict[str, Any]) -> str:
        """生成替代选项理由"""
        reasons = []
        
        if direction.get("risk_factors") and len(direction["risk_factors"]) <= 2:
            reasons.append("风险相对较低")
        
        resources = direction.get("resource_requirements", {})
        if isinstance(resources, dict) and any(level == "低" for level in resources.values()):
            reasons.append("资源需求适中")
        
        timeline = direction.get("expected_timeline", 12)
        if timeline <= 6:
            reasons.append("时间周期较短")
        
        if not reasons:
            reasons.append("可作为备选方案")
        
        return "；".join(reasons)
    
    def _create_learning_path(self, directions: Dict[str, List[Dict[str, Any]]], 
                            analysis: TopicAnalysis, 
                            context: Dict[str, Any]) -> Dict[str, Any]:
        """创建学习路径"""
        return {
            "prerequisite_knowledge": self._identify_prerequisite_knowledge(analysis),
            "skill_development": self._create_skill_development_plan(analysis),
            "learning_resources": self._suggest_learning_resources(analysis),
            "practice_opportunities": self._suggest_practice_opportunities(analysis)
        }
    
    def _identify_prerequisite_knowledge(self, analysis: TopicAnalysis) -> List[str]:
        """识别前置知识"""
        knowledge = ["研究方法论基础", "学术写作规范"]
        
        # 基于研究领域添加前置知识
        if 'Computer Science' in analysis.related_fields:
            knowledge.extend(["编程基础", "算法与数据结构", "统计学基础"])
        
        if 'Mathematics' in analysis.related_fields:
            knowledge.extend(["高等数学", "线性代数", "概率论"])
        
        # 基于研究类型添加前置知识
        if analysis.research_type == ResearchType.EXPERIMENTAL:
            knowledge.extend(["实验设计原理", "统计假设检验"])
        elif analysis.research_type == ResearchType.THEORETICAL:
            knowledge.extend(["数学建模", "逻辑推理"])
        
        return list(set(knowledge))
    
    def _create_skill_development_plan(self, analysis: TopicAnalysis) -> List[Dict[str, Any]]:
        """创建技能发展计划"""
        skills = []
        
        # 基础技能
        skills.append({
            "skill": "文献检索与分析",
            "priority": "高",
            "learning_time": "2-4周",
            "resources": ["学术数据库使用指南", "文献管理工具教程"]
        })
        
        # 基于关键词的专业技能
        for keyword in analysis.keywords[:3]:
            if 'machine learning' in keyword:
                skills.append({
                    "skill": "机器学习基础",
                    "priority": "高",
                    "learning_time": "8-12周",
                    "resources": ["在线课程", "实践项目", "开源工具"]
                })
            elif 'data' in keyword:
                skills.append({
                    "skill": "数据分析与可视化",
                    "priority": "中",
                    "learning_time": "4-6周",
                    "resources": ["Python/R教程", "数据分析工具", "案例研究"]
                })
        
        return skills
    
    def _suggest_learning_resources(self, analysis: TopicAnalysis) -> Dict[str, List[str]]:
        """建议学习资源"""
        return {
            "在线课程": [
                "Coursera专业课程",
                "edX大学课程",
                "Udacity纳米学位"
            ],
            "学术资源": [
                "Google Scholar",
                "IEEE Xplore",
                "ACM Digital Library"
            ],
            "实践平台": [
                "Kaggle竞赛",
                "GitHub开源项目",
                "研究实验室"
            ],
            "社区支持": [
                "学术会议",
                "研究小组",
                "在线论坛"
            ]
        }
    
    def _suggest_practice_opportunities(self, analysis: TopicAnalysis) -> List[str]:
        """建议实践机会"""
        opportunities = [
            "参与开源项目贡献",
            "完成小型研究项目",
            "参加学术会议和研讨会",
            "寻找研究实习机会"
        ]
        
        # 基于研究类型添加特定机会
        if analysis.research_type == ResearchType.EXPERIMENTAL:
            opportunities.extend([
                "设计并执行小规模实验",
                "复现已发表的实验结果"
            ])
        elif analysis.research_type == ResearchType.SURVEY:
            opportunities.extend([
                "撰写文献综述报告",
                "参与系统性综述项目"
            ])
        
        return opportunities
    
    def _suggest_next_steps(self, analysis: TopicAnalysis, context: Dict[str, Any]) -> List[Dict[str, str]]:
        """建议下一步行动"""
        steps = [
            {
                "step": "深入文献调研",
                "description": "针对选定方向进行深入的文献调研",
                "timeline": "2-4周"
            },
            {
                "step": "制定详细计划",
                "description": "基于选定方向制定详细的研究计划",
                "timeline": "1-2周"
            },
            {
                "step": "组建研究团队",
                "description": "根据需要组建合适的研究团队",
                "timeline": "2-3周"
            },
            {
                "step": "申请资源支持",
                "description": "申请必要的资金和资源支持",
                "timeline": "4-8周"
            },
            {
                "step": "开始初步研究",
                "description": "启动研究项目的初步工作",
                "timeline": "根据计划"
            }
        ]
        
        return steps
    
    def _generate_personalized_recommendations(self, analysis: TopicAnalysis, 
                                             research_directions: Dict[str, List[Dict[str, Any]]], 
                                             context: Dict[str, Any]) -> Dict[str, Any]:
        """生成个性化推荐"""
        # 获取用户经验水平
        experience_level = context.get('user_experience', 'intermediate')
        available_resources = context.get('available_resources', 'moderate')
        time_constraint = context.get('time_constraint', '6_months')
        
        # 收集所有方向并评分
        all_directions = []
        for category, directions in research_directions.items():
            for direction in directions:
                direction_copy = direction.copy()
                direction_copy['category'] = category
                all_directions.append(direction_copy)
        
        # 基于用户上下文调整评分
        for direction in all_directions:
            direction['personalized_score'] = self._calculate_personalized_score(
                direction, analysis, context
            )
            direction['recommendation_reason'] = self._generate_recommendation_reason(
                direction, analysis, context
            )
        
        # 按个性化评分排序
        all_directions.sort(key=lambda x: x['personalized_score'], reverse=True)
        
        # 生成顶级推荐
        top_recommendations = all_directions[:5]
        
        # 生成学习路径
        learning_path = self._generate_learning_path(analysis, context)
        
        # 生成下一步建议
        next_steps = self._generate_next_steps(analysis, top_recommendations[0] if top_recommendations else None, context)
        
        return {
            "top_recommendations": top_recommendations,
            "learning_path": learning_path,
            "next_steps": next_steps,
            "experience_match": self._assess_experience_match(analysis, context),
            "resource_alignment": self._assess_resource_alignment(analysis, context)
        }
    
    def _calculate_personalized_score(self, direction: Dict[str, Any], 
                                    analysis: TopicAnalysis, 
                                    context: Dict[str, Any]) -> float:
        """计算个性化评分"""
        base_score = direction.get('priority_score', 0.5)
        
        # 经验水平调整
        experience_level = context.get('user_experience', 'intermediate')
        experience_adjustments = {
            'beginner': {
                'feasibility_weight': 0.5,
                'complexity_penalty': 0.3
            },
            'intermediate': {
                'feasibility_weight': 0.3,
                'complexity_penalty': 0.1
            },
            'advanced': {
                'feasibility_weight': 0.1,
                'complexity_penalty': -0.1  # 高级用户喜欢挑战
            }
        }
        
        exp_adj = experience_adjustments.get(experience_level, experience_adjustments['intermediate'])
        
        # 可行性调整
        feasibility = direction.get('feasibility', 0.5)
        base_score += (feasibility - 0.5) * exp_adj['feasibility_weight']
        
        # 复杂度调整
        if analysis.complexity_level in [ResearchComplexity.HIGH, ResearchComplexity.VERY_HIGH]:
            base_score += exp_adj['complexity_penalty']
        
        # 资源匹配调整
        available_resources = context.get('available_resources', 'moderate')
        resource_requirements = direction.get('resource_requirements', {})
        resource_match = self._calculate_resource_match(resource_requirements, available_resources)
        base_score += (resource_match - 0.5) * 0.2
        
        # 时间约束调整
        time_constraint = context.get('time_constraint', '6_months')
        timeline_months = direction.get('expected_timeline', 6)
        time_match = self._calculate_time_match(timeline_months, time_constraint)
        base_score += (time_match - 0.5) * 0.15
        
        return max(0.0, min(1.0, base_score))
    
    def _calculate_resource_match(self, requirements: Dict[str, str], available: str) -> float:
        """计算资源匹配度"""
        if not requirements:
            return 0.5
        
        resource_levels = {'低': 1, '中等': 2, '高': 3}
        available_level = resource_levels.get(available, 2)
        
        total_match = 0
        for req_level in requirements.values():
            req_num = resource_levels.get(req_level, 2)
            if available_level >= req_num:
                total_match += 1
            else:
                total_match += available_level / req_num
        
        return total_match / len(requirements)
    
    def _calculate_time_match(self, required_months: int, available_time: str) -> float:
        """计算时间匹配度"""
        time_mapping = {
            '3_months': 3,
            '6_months': 6,
            '12_months': 12,
            'flexible': 18
        }
        
        available_months = time_mapping.get(available_time, 6)
        
        if required_months <= available_months:
            return 1.0
        else:
            return available_months / required_months
    
    def _generate_recommendation_reason(self, direction: Dict[str, Any], 
                                      analysis: TopicAnalysis, 
                                      context: Dict[str, Any]) -> str:
        """生成推荐理由"""
        reasons = []
        
        # 基于可行性
        feasibility = direction.get('feasibility', 0.5)
        if feasibility > 0.7:
            reasons.append("可行性高")
        elif feasibility < 0.4:
            reasons.append("具有挑战性")
        
        # 基于创新潜力
        innovation = direction.get('innovation_potential', 0.5)
        if innovation > 0.7:
            reasons.append("创新潜力大")
        
        # 基于用户经验
        experience = context.get('user_experience', 'intermediate')
        if experience == 'beginner' and feasibility > 0.6:
            reasons.append("适合初学者")
        elif experience == 'advanced' and innovation > 0.6:
            reasons.append("适合高级研究者")
        
        # 基于资源匹配
        available_resources = context.get('available_resources', 'moderate')
        resource_requirements = direction.get('resource_requirements', {})
        if self._calculate_resource_match(resource_requirements, available_resources) > 0.7:
            reasons.append("资源需求匹配")
        
        # 基于时间匹配
        time_constraint = context.get('time_constraint', '6_months')
        timeline = direction.get('expected_timeline', 6)
        if self._calculate_time_match(timeline, time_constraint) > 0.8:
            reasons.append("时间安排合理")
        
        return "、".join(reasons) if reasons else "综合评估推荐"
    
    def _generate_learning_path(self, analysis: TopicAnalysis, context: Dict[str, Any]) -> Dict[str, Any]:
        """生成学习路径"""
        experience_level = context.get('user_experience', 'intermediate')
        
        # 前置知识要求
        prerequisite_knowledge = []
        if analysis.complexity_level in [ResearchComplexity.HIGH, ResearchComplexity.VERY_HIGH]:
            prerequisite_knowledge.extend([
                "研究方法论基础",
                "统计分析基础",
                "文献检索技能"
            ])
        
        # 基于领域添加特定知识
        for field in analysis.related_fields:
            if field == 'Computer Science':
                prerequisite_knowledge.extend(["编程基础", "算法与数据结构"])
            elif field == 'Mathematics':
                prerequisite_knowledge.extend(["高等数学", "线性代数"])
            elif field == 'Biology':
                prerequisite_knowledge.extend(["生物学基础", "分子生物学"])
        
        # 技能发展建议
        skill_development = []
        if analysis.research_type == ResearchType.EXPERIMENTAL:
            skill_development.extend([
                "实验设计技能",
                "数据分析技能",
                "统计软件使用"
            ])
        elif analysis.research_type == ResearchType.THEORETICAL:
            skill_development.extend([
                "数学建模技能",
                "理论分析能力",
                "证明技巧"
            ])
        elif analysis.research_type == ResearchType.SURVEY:
            skill_development.extend([
                "文献综述技能",
                "批判性思维",
                "信息整合能力"
            ])
        
        # 推荐资源
        recommended_resources = []
        if experience_level == 'beginner':
            recommended_resources.extend([
                "入门教程和在线课程",
                "基础教材和参考书",
                "学术写作指南"
            ])
        else:
            recommended_resources.extend([
                "高级专业课程",
                "最新研究论文",
                "专业会议和研讨会"
            ])
        
        return {
            "prerequisite_knowledge": list(set(prerequisite_knowledge)),
            "skill_development": list(set(skill_development)),
            "recommended_resources": recommended_resources,
            "learning_timeline": self._estimate_learning_timeline(analysis, context)
        }
    
    def _estimate_learning_timeline(self, analysis: TopicAnalysis, context: Dict[str, Any]) -> Dict[str, str]:
        """估算学习时间线"""
        experience_level = context.get('user_experience', 'intermediate')
        
        base_times = {
            'beginner': {
                'prerequisite_learning': '2-3个月',
                'skill_development': '3-4个月',
                'research_preparation': '1-2个月'
            },
            'intermediate': {
                'prerequisite_learning': '1-2个月',
                'skill_development': '2-3个月',
                'research_preparation': '2-3周'
            },
            'advanced': {
                'prerequisite_learning': '2-4周',
                'skill_development': '1-2个月',
                'research_preparation': '1-2周'
            }
        }
        
        return base_times.get(experience_level, base_times['intermediate'])
    
    def _generate_next_steps(self, analysis: TopicAnalysis, 
                           top_recommendation: Optional[Dict[str, Any]], 
                           context: Dict[str, Any]) -> List[Dict[str, str]]:
        """生成下一步建议"""
        steps = []
        
        # 第一步：文献调研
        steps.append({
            "step": "进行初步文献调研",
            "description": f"收集{analysis.topic}相关的核心文献和最新研究",
            "timeline": "1-2周",
            "priority": "高"
        })
        
        # 第二步：确定具体方向
        if top_recommendation:
            steps.append({
                "step": f"深入研究{top_recommendation['title']}",
                "description": top_recommendation['description'],
                "timeline": f"{top_recommendation.get('expected_timeline', 6)}个月",
                "priority": "高"
            })
        
        # 第三步：准备资源
        steps.append({
            "step": "准备研究资源",
            "description": "获取必要的工具、数据和计算资源",
            "timeline": "1-2周",
            "priority": "中"
        })
        
        # 第四步：制定详细计划
        steps.append({
            "step": "制定详细研究计划",
            "description": "确定具体的研究问题、方法和时间安排",
            "timeline": "1周",
            "priority": "高"
        })
        
        # 基于经验水平调整
        experience_level = context.get('user_experience', 'intermediate')
        if experience_level == 'beginner':
            steps.insert(1, {
                "step": "学习基础知识",
                "description": "掌握研究领域的基础概念和方法",
                "timeline": "2-4周",
                "priority": "高"
            })
        
        return steps
    
    def _assess_experience_match(self, analysis: TopicAnalysis, context: Dict[str, Any]) -> Dict[str, Any]:
        """评估经验匹配度"""
        experience_level = context.get('user_experience', 'intermediate')
        
        # 计算匹配分数
        match_score = 0.5  # 基础分数
        
        if experience_level == 'beginner':
            if analysis.complexity_level == ResearchComplexity.LOW:
                match_score = 0.9
            elif analysis.complexity_level == ResearchComplexity.MEDIUM:
                match_score = 0.6
            else:
                match_score = 0.3
        elif experience_level == 'intermediate':
            if analysis.complexity_level in [ResearchComplexity.MEDIUM, ResearchComplexity.HIGH]:
                match_score = 0.8
            else:
                match_score = 0.6
        else:  # advanced
            if analysis.complexity_level in [ResearchComplexity.HIGH, ResearchComplexity.VERY_HIGH]:
                match_score = 0.9
            else:
                match_score = 0.7
        
        # 生成建议
        suggestions = []
        if match_score < 0.5:
            if experience_level == 'beginner':
                suggestions.append("建议先从简单的调研项目开始")
                suggestions.append("考虑寻找导师或合作者")
            else:
                suggestions.append("可以考虑更具挑战性的研究方向")
        
        return {
            "match_score": match_score,
            "level_assessment": "匹配" if match_score > 0.6 else "需要调整",
            "suggestions": suggestions
        }
    
    def _assess_resource_alignment(self, analysis: TopicAnalysis, context: Dict[str, Any]) -> Dict[str, Any]:
        """评估资源对齐度"""
        available_resources = context.get('available_resources', 'moderate')
        required_resources = analysis.required_resources
        
        # 简化的资源评估
        resource_categories = {
            "计算资源": ["计算设备", "高性能计算资源", "云计算平台"],
            "数据资源": ["数据集", "数据库访问", "调研数据"],
            "软件工具": ["专业软件许可", "统计软件", "编程环境"],
            "人力资源": ["专家咨询", "团队协作", "导师指导"]
        }
        
        alignment_score = 0.7  # 默认对齐度
        
        # 基于可用资源调整
        if available_resources == 'limited':
            alignment_score = 0.4
        elif available_resources == 'abundant':
            alignment_score = 0.9
        
        recommendations = []
        if alignment_score < 0.6:
            recommendations.extend([
                "考虑申请研究资助",
                "寻找合作机构",
                "使用开源替代工具"
            ])
        
        return {
            "alignment_score": alignment_score,
            "resource_assessment": "充足" if alignment_score > 0.7 else "需要补充",
            "recommendations": recommendations
        }
    
    def _generate_selection_guidance(self, analysis: TopicAnalysis, 
                                   research_directions: Dict[str, List[Dict[str, Any]]], 
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        """生成选择指导"""
        # 定义选择标准
        selection_criteria = [
            {
                "name": "可行性",
                "description": "研究方向的实施可行性",
                "weight": 0.3,
                "considerations": [
                    "资源需求是否匹配",
                    "技术难度是否适中",
                    "时间安排是否合理"
                ]
            },
            {
                "name": "创新性",
                "description": "研究的创新潜力和学术价值",
                "weight": 0.25,
                "considerations": [
                    "是否填补研究空白",
                    "方法是否新颖",
                    "结果是否有突破性"
                ]
            },
            {
                "name": "影响力",
                "description": "研究成果的潜在影响",
                "weight": 0.2,
                "considerations": [
                    "学术影响力",
                    "实际应用价值",
                    "社会意义"
                ]
            },
            {
                "name": "个人匹配度",
                "description": "与个人能力和兴趣的匹配程度",
                "weight": 0.15,
                "considerations": [
                    "专业背景匹配",
                    "技能要求匹配",
                    "兴趣爱好匹配"
                ]
            },
            {
                "name": "风险控制",
                "description": "研究风险的可控性",
                "weight": 0.1,
                "considerations": [
                    "失败风险评估",
                    "备选方案可用性",
                    "风险缓解策略"
                ]
            }
        ]
        
        # 生成决策矩阵
        decision_matrix = self._create_decision_matrix(research_directions, selection_criteria)
        
        # 生成选择建议
        selection_recommendations = self._generate_selection_recommendations(
            analysis, research_directions, context
        )
        
        return {
            "selection_criteria": selection_criteria,
            "decision_matrix": decision_matrix,
            "selection_recommendations": selection_recommendations,
            "decision_process": self._outline_decision_process()
        }
    
    def _create_decision_matrix(self, research_directions: Dict[str, List[Dict[str, Any]]], 
                              selection_criteria: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """创建决策矩阵"""
        matrix = []
        
        for category, directions in research_directions.items():
            for direction in directions[:2]:  # 每个类别取前2个
                scores = {}
                for criterion in selection_criteria:
                    criterion_name = criterion["name"]
                    if criterion_name == "可行性":
                        scores[criterion_name] = direction.get("feasibility", 0.5)
                    elif criterion_name == "创新性":
                        scores[criterion_name] = direction.get("innovation_potential", 0.5)
                    elif criterion_name == "个人匹配度":
                        scores[criterion_name] = direction.get("personalized_score", 0.5)
                    else:
                        scores[criterion_name] = 0.6  # 默认分数
                
                matrix.append({
                    "direction": direction["title"],
                    "category": category,
                    "scores": scores,
                    "weighted_score": sum(
                        scores[c["name"]] * c["weight"] 
                        for c in selection_criteria
                    )
                })
        
        # 按加权分数排序
        matrix.sort(key=lambda x: x["weighted_score"], reverse=True)
        return matrix
    
    def _generate_selection_recommendations(self, analysis: TopicAnalysis, 
                                          research_directions: Dict[str, List[Dict[str, Any]]], 
                                          context: Dict[str, Any]) -> List[str]:
        """生成选择建议"""
        recommendations = []
        
        experience_level = context.get('user_experience', 'intermediate')
        
        # 基于经验水平的建议
        if experience_level == 'beginner':
            recommendations.extend([
                "优先选择可行性高的研究方向",
                "从文献综述类项目开始积累经验",
                "寻找有经验的导师或合作者"
            ])
        elif experience_level == 'advanced':
            recommendations.extend([
                "可以考虑高风险高回报的创新方向",
                "关注研究的长期影响和学术价值",
                "考虑跨学科合作的可能性"
            ])
        
        # 基于复杂度的建议
        if analysis.complexity_level == ResearchComplexity.VERY_HIGH:
            recommendations.append("建议分阶段实施，设置中期检查点")
        
        # 基于时间约束的建议
        time_constraint = context.get('time_constraint', '6_months')
        if time_constraint in ['3_months', '6_months']:
            recommendations.append("选择时间线较短的研究方向")
        
        return recommendations
    
    def _outline_decision_process(self) -> List[str]:
        """概述决策过程"""
        return [
            "1. 评估个人能力和资源条件",
            "2. 分析各研究方向的优缺点",
            "3. 使用决策矩阵进行量化比较",
            "4. 考虑风险因素和备选方案",
            "5. 征求导师或专家意见",
            "6. 做出最终决策并制定实施计划"
        ]
    
    def _generate_implementation_suggestions(self, analysis: TopicAnalysis, 
                                          context: Dict[str, Any]) -> Dict[str, Any]:
        """生成实施建议"""
        return {
            "project_management": [
                "使用项目管理工具跟踪进度",
                "设置定期检查点和里程碑",
                "建立风险监控机制"
            ],
            "collaboration_tips": [
                "建立有效的沟通机制",
                "明确角色分工和责任",
                "定期举行项目会议"
            ],
            "quality_assurance": [
                "建立代码和数据版本控制",
                "进行同行评议",
                "记录详细的实验日志"
            ],
            "timeline_management": [
                "制定详细的工作计划",
                "预留缓冲时间应对意外",
                "定期评估和调整计划"
            ]
        }