"""
知识图谱构建器

从文献集合中构建知识图谱，识别实体关系和研究空白
"""

import re
import json
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging

from ..models.analysis_models import (
    Paper, KnowledgeNode, KnowledgeEdge, KnowledgeGraph, ResearchGap
)
from ..models.base_models import BaseModel
from .base_component import BaseComponent

logger = logging.getLogger(__name__)


@dataclass
class EntityExtractionResult:
    """实体提取结果"""
    concepts: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)
    datasets: List[str] = field(default_factory=list)
    authors: List[str] = field(default_factory=list)
    venues: List[str] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class RelationshipExtractionResult:
    """关系提取结果"""
    relationships: List[Tuple[str, str, str, float]] = field(default_factory=list)
    co_occurrence_matrix: Dict[Tuple[str, str], int] = field(default_factory=dict)
    semantic_similarities: Dict[Tuple[str, str], float] = field(default_factory=dict)


class KnowledgeGraphBuilder(BaseComponent):
    """知识图谱构建器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._initialize_patterns()
    
    def get_required_configs(self) -> List[str]:
        """获取必需的配置项列表"""
        return []  # 知识图谱构建器不需要特殊配置
    
    def _setup_component(self):
        """设置组件特定的初始化逻辑"""
        self.logger.info("知识图谱构建器初始化完成")
    
    def _initialize_patterns(self):
        """初始化实体和关系识别模式"""
        # 概念和方法识别模式
        self.concept_patterns = {
            'method': [
                r'(?:deep|machine|statistical|neural|reinforcement)\s+learning',
                r'(?:support vector|decision tree|random forest|gradient boosting)',
                r'(?:convolutional|recurrent|transformer|attention)\s+(?:neural\s+)?network',
                r'(?:natural language|computer vision|speech recognition)',
                r'(?:clustering|classification|regression|optimization)',
                r'(?:supervised|unsupervised|semi-supervised)\s+learning'
            ],
            'dataset': [
                r'(?:ImageNet|CIFAR|MNIST|CoNLL|SQuAD|GLUE|SuperGLUE)',
                r'(?:\w+\s+)?(?:dataset|corpus|benchmark|collection)',
                r'(?:training|test|validation)\s+(?:set|data)'
            ],
            'metric': [
                r'(?:accuracy|precision|recall|F1|BLEU|ROUGE|perplexity)',
                r'(?:mean|root mean)\s+squared\s+error',
                r'(?:area under|receiver operating)\s+(?:curve|characteristic)'
            ],
            'domain': [
                r'(?:computer vision|natural language processing|speech recognition)',
                r'(?:machine translation|sentiment analysis|question answering)',
                r'(?:image classification|object detection|semantic segmentation)',
                r'(?:reinforcement learning|transfer learning|few-shot learning)'
            ]
        }
        
        # 关系类型定义
        self.relationship_types = {
            'uses': ['use', 'uses', 'using', 'utilized', 'applied', 'employ'],
            'extends': ['extend', 'extends', 'extension', 'improve', 'enhanced'],
            'compares': ['compare', 'comparison', 'versus', 'vs', 'against'],
            'evaluates': ['evaluate', 'evaluation', 'test', 'assess', 'measure'],
            'based_on': ['based on', 'build on', 'derived from', 'inspired by'],
            'outperforms': ['outperform', 'better than', 'superior', 'exceed']
        }
    
    def build_knowledge_graph(self, papers: List[Paper]) -> KnowledgeGraph:
        """
        从论文集合构建知识图谱
        
        Args:
            papers: 论文列表
            
        Returns:
            构建的知识图谱
        """
        try:
            logger.info(f"开始构建知识图谱，输入论文数量: {len(papers)}")
            
            # 1. 实体提取
            entities = self._extract_entities(papers)
            logger.info(f"提取实体数量: {len(entities.concepts + entities.methods + entities.datasets)}")
            
            # 2. 创建节点
            nodes = self._create_nodes(entities, papers)
            logger.info(f"创建节点数量: {len(nodes)}")
            
            # 3. 关系提取
            relationships = self._extract_relationships(papers, nodes)
            logger.info(f"提取关系数量: {len(relationships.relationships)}")
            
            # 4. 创建边
            edges = self._create_edges(relationships, nodes)
            logger.info(f"创建边数量: {len(edges)}")
            
            # 5. 识别研究空白
            research_gaps = self._identify_research_gaps(nodes, edges, papers)
            logger.info(f"识别研究空白数量: {len(research_gaps)}")
            
            # 6. 识别热点主题
            hot_topics = self._identify_hot_topics(papers, nodes)
            logger.info(f"识别热点主题数量: {len(hot_topics)}")
            
            # 7. 识别新兴趋势
            emerging_trends = self._identify_emerging_trends(papers, nodes)
            logger.info(f"识别新兴趋势数量: {len(emerging_trends)}")
            
            # 构建知识图谱
            knowledge_graph = KnowledgeGraph(
                nodes=nodes,
                edges=edges,
                research_gaps=research_gaps,
                hot_topics=hot_topics,
                emerging_trends=emerging_trends
            )
            
            logger.info("知识图谱构建完成")
            return knowledge_graph
            
        except Exception as e:
            logger.error(f"构建知识图谱时发生错误: {str(e)}")
            raise
    
    def _extract_entities(self, papers: List[Paper]) -> EntityExtractionResult:
        """提取实体"""
        concepts = set()
        methods = set()
        datasets = set()
        authors = set()
        venues = set()
        confidence_scores = {}
        
        for paper in papers:
            # 合并文本内容
            text_content = f"{paper.title} {paper.abstract} {' '.join(paper.keywords)}"
            text_content = text_content.lower()
            
            # 提取概念和方法
            for category, patterns in self.concept_patterns.items():
                for pattern in patterns:
                    matches = re.findall(pattern, text_content, re.IGNORECASE)
                    for match in matches:
                        entity = match.strip()
                        if len(entity) > 2:  # 过滤太短的实体
                            if category in ['method', 'domain']:
                                methods.add(entity)
                            else:
                                concepts.add(entity)
                            confidence_scores[entity] = confidence_scores.get(entity, 0) + 1
            
            # 提取关键词作为概念
            for keyword in paper.keywords:
                if len(keyword) > 2:
                    concepts.add(keyword.lower())
                    confidence_scores[keyword.lower()] = confidence_scores.get(keyword.lower(), 0) + 1
            
            # 提取作者
            for author in paper.authors:
                authors.add(author)
            
            # 提取发表场所
            if paper.journal_or_venue:
                venues.add(paper.journal_or_venue)
        
        # 归一化置信度分数
        max_count = max(confidence_scores.values()) if confidence_scores else 1
        for entity in confidence_scores:
            confidence_scores[entity] = confidence_scores[entity] / max_count
        
        return EntityExtractionResult(
            concepts=list(concepts),
            methods=list(methods),
            datasets=list(datasets),
            authors=list(authors),
            venues=list(venues),
            confidence_scores=confidence_scores
        ) 
   
    def _create_nodes(self, entities: EntityExtractionResult, papers: List[Paper]) -> List[KnowledgeNode]:
        """创建知识图谱节点"""
        nodes = []
        
        # 创建概念节点
        for concept in entities.concepts:
            node = KnowledgeNode(
                label=concept,
                node_type="concept",
                importance_score=entities.confidence_scores.get(concept, 0.1),
                properties={
                    "frequency": entities.confidence_scores.get(concept, 0),
                    "category": "concept"
                }
            )
            # 找到相关论文
            for paper in papers:
                text_content = f"{paper.title} {paper.abstract} {' '.join(paper.keywords)}".lower()
                if concept in text_content:
                    node.related_papers.append(paper.id)
            nodes.append(node)
        
        # 创建方法节点
        for method in entities.methods:
            node = KnowledgeNode(
                label=method,
                node_type="method",
                importance_score=entities.confidence_scores.get(method, 0.1),
                properties={
                    "frequency": entities.confidence_scores.get(method, 0),
                    "category": "method"
                }
            )
            # 找到相关论文
            for paper in papers:
                text_content = f"{paper.title} {paper.abstract} {paper.methodology}".lower()
                if method in text_content:
                    node.related_papers.append(paper.id)
            nodes.append(node)
        
        # 创建作者节点
        author_paper_count = Counter()
        for paper in papers:
            for author in paper.authors:
                author_paper_count[author] += 1
        
        for author in entities.authors:
            node = KnowledgeNode(
                label=author,
                node_type="author",
                importance_score=min(1.0, author_paper_count[author] / 10.0),  # 归一化
                properties={
                    "paper_count": author_paper_count[author],
                    "category": "author"
                }
            )
            # 找到相关论文
            for paper in papers:
                if author in paper.authors:
                    node.related_papers.append(paper.id)
            nodes.append(node)
        
        # 创建发表场所节点
        venue_paper_count = Counter()
        for paper in papers:
            if paper.journal_or_venue:
                venue_paper_count[paper.journal_or_venue] += 1
        
        for venue in entities.venues:
            node = KnowledgeNode(
                label=venue,
                node_type="venue",
                importance_score=min(1.0, venue_paper_count[venue] / 20.0),  # 归一化
                properties={
                    "paper_count": venue_paper_count[venue],
                    "category": "venue"
                }
            )
            # 找到相关论文
            for paper in papers:
                if paper.journal_or_venue == venue:
                    node.related_papers.append(paper.id)
            nodes.append(node)
        
        return nodes
    
    def _extract_relationships(self, papers: List[Paper], nodes: List[KnowledgeNode]) -> RelationshipExtractionResult:
        """提取实体间关系"""
        relationships = []
        co_occurrence_matrix = defaultdict(int)
        semantic_similarities = {}
        
        # 创建节点标签到节点的映射
        label_to_node = {node.label: node for node in nodes}
        
        # 1. 基于共现的关系提取
        for paper in papers:
            text_content = f"{paper.title} {paper.abstract} {paper.methodology}".lower()
            
            # 找到在此论文中出现的节点
            paper_nodes = []
            for node in nodes:
                if node.label.lower() in text_content:
                    paper_nodes.append(node)
            
            # 计算共现关系
            for i, node1 in enumerate(paper_nodes):
                for j, node2 in enumerate(paper_nodes):
                    if i != j:
                        pair = tuple(sorted([node1.label, node2.label]))
                        co_occurrence_matrix[pair] += 1
        
        # 2. 基于文本模式的关系提取
        for paper in papers:
            text_content = f"{paper.title} {paper.abstract}".lower()
            
            # 查找特定的关系模式
            for relation_type, patterns in self.relationship_types.items():
                for pattern in patterns:
                    # 简化的模式匹配
                    if pattern in text_content:
                        # 在模式前后查找实体
                        sentences = re.split(r'[.!?]', text_content)
                        for sentence in sentences:
                            if pattern in sentence:
                                # 找到句子中的实体
                                sentence_entities = []
                                for node in nodes:
                                    if node.label.lower() in sentence:
                                        sentence_entities.append(node.label)
                                
                                # 创建关系
                                if len(sentence_entities) >= 2:
                                    for i in range(len(sentence_entities) - 1):
                                        relationships.append((
                                            sentence_entities[i],
                                            sentence_entities[i + 1],
                                            relation_type,
                                            0.7  # 基础置信度
                                        ))
        
        return RelationshipExtractionResult(
            relationships=relationships,
            co_occurrence_matrix=dict(co_occurrence_matrix),
            semantic_similarities=semantic_similarities
        )
    
    def _create_edges(self, relationships: RelationshipExtractionResult, nodes: List[KnowledgeNode]) -> List[KnowledgeEdge]:
        """创建知识图谱边"""
        edges = []
        label_to_id = {node.label: node.id for node in nodes}
        
        # 从关系提取结果创建边
        for source_label, target_label, relation_type, strength in relationships.relationships:
            if source_label in label_to_id and target_label in label_to_id:
                edge = KnowledgeEdge(
                    source_node_id=label_to_id[source_label],
                    target_node_id=label_to_id[target_label],
                    relationship_type=relation_type,
                    strength=strength
                )
                edges.append(edge)
        
        # 从共现矩阵创建边
        for (label1, label2), count in relationships.co_occurrence_matrix.items():
            if label1 in label_to_id and label2 in label_to_id and count > 1:
                # 归一化共现强度
                strength = min(1.0, count / 10.0)
                edge = KnowledgeEdge(
                    source_node_id=label_to_id[label1],
                    target_node_id=label_to_id[label2],
                    relationship_type="co_occurs",
                    strength=strength
                )
                edges.append(edge)
        
        return edges 
   
    def _identify_research_gaps(self, nodes: List[KnowledgeNode], edges: List[KnowledgeEdge], papers: List[Paper]) -> List[ResearchGap]:
        """识别研究空白"""
        gaps = []
        
        # 1. 基于连接度识别空白 - 找到重要但连接度低的节点
        node_connections = defaultdict(int)
        for edge in edges:
            node_connections[edge.source_node_id] += 1
            node_connections[edge.target_node_id] += 1
        
        for node in nodes:
            if node.importance_score > 0.5 and node_connections[node.id] < 2:
                gap = ResearchGap(
                    description=f"概念 '{node.label}' 缺乏与其他重要概念的充分连接",
                    gap_type="methodological",
                    importance_level=node.importance_score,
                    difficulty_level=0.6,
                    related_concepts=[node.label],
                    potential_impact="可能存在未被充分探索的研究方向",
                    suggested_approaches=[
                        f"探索 {node.label} 与其他相关概念的结合",
                        f"调研 {node.label} 的跨领域应用"
                    ]
                )
                gaps.append(gap)
        
        # 2. 基于时间趋势识别空白
        recent_papers = [p for p in papers if p.is_recent(3)]  # 近3年论文
        older_papers = [p for p in papers if not p.is_recent(3)]
        
        if recent_papers and older_papers:
            # 找到在旧论文中频繁出现但在新论文中较少的概念
            old_concepts = Counter()
            new_concepts = Counter()
            
            for paper in older_papers:
                text = f"{paper.title} {paper.abstract}".lower()
                for node in nodes:
                    if node.node_type == "concept" and node.label.lower() in text:
                        old_concepts[node.label] += 1
            
            for paper in recent_papers:
                text = f"{paper.title} {paper.abstract}".lower()
                for node in nodes:
                    if node.node_type == "concept" and node.label.lower() in text:
                        new_concepts[node.label] += 1
            
            for concept, old_count in old_concepts.items():
                new_count = new_concepts.get(concept, 0)
                if old_count > 5 and new_count < old_count * 0.3:  # 显著下降
                    gap = ResearchGap(
                        description=f"概念 '{concept}' 在近期研究中关注度下降",
                        gap_type="empirical",
                        importance_level=0.7,
                        difficulty_level=0.4,
                        related_concepts=[concept],
                        potential_impact="可能存在被忽视但仍有价值的研究方向",
                        suggested_approaches=[
                            f"重新审视 {concept} 的现代应用",
                            f"结合新技术重新探索 {concept}"
                        ]
                    )
                    gaps.append(gap)
        
        # 3. 基于方法组合识别空白
        method_nodes = [node for node in nodes if node.node_type == "method"]
        method_combinations = set()
        
        for paper in papers:
            paper_methods = []
            text = f"{paper.title} {paper.abstract} {paper.methodology}".lower()
            for method_node in method_nodes:
                if method_node.label.lower() in text:
                    paper_methods.append(method_node.label)
            
            if len(paper_methods) > 1:
                for i in range(len(paper_methods)):
                    for j in range(i + 1, len(paper_methods)):
                        combo = tuple(sorted([paper_methods[i], paper_methods[j]]))
                        method_combinations.add(combo)
        
        # 找到理论上可能但实际上很少结合的方法
        for i, method1 in enumerate(method_nodes):
            for j, method2 in enumerate(method_nodes):
                if i < j:
                    pair = tuple(sorted([method1.label, method2.label]))
                    if (pair not in method_combinations and 
                        method1.importance_score > 0.3 and method2.importance_score > 0.3):
                        gap = ResearchGap(
                            description=f"方法 '{method1.label}' 和 '{method2.label}' 的结合应用研究不足",
                            gap_type="methodological",
                            importance_level=0.6,
                            difficulty_level=0.7,
                            related_concepts=[method1.label, method2.label],
                            potential_impact="可能产生新的方法论突破",
                            suggested_approaches=[
                                f"探索 {method1.label} 和 {method2.label} 的协同效应",
                                f"设计结合两种方法的新框架"
                            ]
                        )
                        gaps.append(gap)
                        if len(gaps) >= 10:  # 限制数量
                            break
            if len(gaps) >= 10:
                break
        
        return gaps[:10]  # 返回前10个最重要的研究空白   
 
    def _identify_hot_topics(self, papers: List[Paper], nodes: List[KnowledgeNode]) -> List[str]:
        """识别热点主题"""
        # 基于近期论文数量和引用数识别热点
        recent_papers = [p for p in papers if p.is_recent(2)]  # 近2年
        
        if not recent_papers:
            return []
        
        topic_scores = {}
        for node in nodes:
            if node.node_type in ["concept", "method"]:
                # 计算在近期论文中的出现频率
                recent_count = 0
                total_citations = 0
                
                for paper in recent_papers:
                    text = f"{paper.title} {paper.abstract}".lower()
                    if node.label.lower() in text:
                        recent_count += 1
                        total_citations += paper.citation_count
                
                if recent_count > 0:
                    # 综合考虑频率和引用数
                    frequency_score = recent_count / len(recent_papers)
                    citation_score = total_citations / recent_count if recent_count > 0 else 0
                    topic_scores[node.label] = frequency_score * 0.7 + min(1.0, citation_score / 100) * 0.3
        
        # 返回得分最高的主题
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, score in sorted_topics[:10]]
    
    def _identify_emerging_trends(self, papers: List[Paper], nodes: List[KnowledgeNode]) -> List[str]:
        """识别新兴趋势"""
        if not papers:
            return []
            
        # 比较不同时期的主题流行度
        current_year = max(paper.publication_year for paper in papers)
        recent_papers = [p for p in papers if p.publication_year >= current_year - 1]
        older_papers = [p for p in papers if p.publication_year < current_year - 1]
        
        if not recent_papers or not older_papers:
            return []
        
        trend_scores = {}
        for node in nodes:
            if node.node_type in ["concept", "method"]:
                # 计算在不同时期的出现频率
                recent_count = sum(1 for paper in recent_papers 
                                 if node.label.lower() in f"{paper.title} {paper.abstract}".lower())
                older_count = sum(1 for paper in older_papers 
                                if node.label.lower() in f"{paper.title} {paper.abstract}".lower())
                
                recent_freq = recent_count / len(recent_papers) if recent_papers else 0
                older_freq = older_count / len(older_papers) if older_papers else 0
                
                # 计算增长率
                if older_freq > 0:
                    growth_rate = (recent_freq - older_freq) / older_freq
                else:
                    growth_rate = recent_freq * 10  # 全新概念给予高分
                
                # 只考虑有一定基础频率的概念
                if recent_freq > 0.05:  # 至少在5%的近期论文中出现
                    trend_scores[node.label] = growth_rate
        
        # 返回增长最快的趋势
        sorted_trends = sorted(trend_scores.items(), key=lambda x: x[1], reverse=True)
        return [trend for trend, score in sorted_trends[:8] if score > 0.5]   
 
    def visualize_knowledge_graph(self, knowledge_graph: KnowledgeGraph, output_path: str = "knowledge_graph.html") -> str:
        """
        可视化知识图谱
        
        Args:
            knowledge_graph: 知识图谱
            output_path: 输出文件路径
            
        Returns:
            可视化文件路径
        """
        try:
            # 生成简单的HTML可视化
            html_content = self._generate_graph_html(knowledge_graph)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"知识图谱可视化已保存到: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"可视化知识图谱时发生错误: {str(e)}")
            raise
    
    def _generate_graph_html(self, knowledge_graph: KnowledgeGraph) -> str:
        """生成图谱的HTML可视化"""
        # 节点颜色映射
        color_map = {
            'concept': '#FF6B6B',
            'method': '#4ECDC4', 
            'author': '#45B7D1',
            'venue': '#96CEB4'
        }
        
        # 生成节点数据
        nodes_data = []
        for i, node in enumerate(knowledge_graph.nodes):
            nodes_data.append({
                'id': node.id,
                'label': node.label,
                'x': (i % 10) * 80 + 50,
                'y': (i // 10) * 80 + 50,
                'size': max(10, node.importance_score * 50),
                'color': color_map.get(node.node_type, '#999999'),
                'type': node.node_type
            })
        
        # 生成边数据
        edges_data = []
        for edge in knowledge_graph.edges:
            edges_data.append({
                'source': edge.source_node_id,
                'target': edge.target_node_id,
                'weight': edge.strength,
                'relation': edge.relationship_type
            })
        
        # HTML模板
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>知识图谱可视化</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .graph-container {{ border: 1px solid #ccc; width: 800px; height: 600px; }}
                .legend {{ margin-top: 20px; }}
                .legend-item {{ display: inline-block; margin-right: 20px; }}
                .legend-color {{ width: 15px; height: 15px; display: inline-block; margin-right: 5px; }}
                .info-panel {{ margin-top: 20px; padding: 15px; background: #f5f5f5; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>学术研究知识图谱</h1>
            
            <div class="legend">
                <div class="legend-item">
                    <span class="legend-color" style="background-color: #FF6B6B;"></span>概念
                </div>
                <div class="legend-item">
                    <span class="legend-color" style="background-color: #4ECDC4;"></span>方法
                </div>
                <div class="legend-item">
                    <span class="legend-color" style="background-color: #45B7D1;"></span>作者
                </div>
                <div class="legend-item">
                    <span class="legend-color" style="background-color: #96CEB4;"></span>发表场所
                </div>
            </div>
            
            <div class="graph-container">
                <p>图谱可视化区域（需要D3.js或其他可视化库支持）</p>
            </div>
            
            <div class="info-panel">
                <h3>图谱统计信息</h3>
                <p>节点数量: {len(knowledge_graph.nodes)}</p>
                <p>边数量: {len(knowledge_graph.edges)}</p>
                <p>研究空白: {len(knowledge_graph.research_gaps)}</p>
                <p>热点主题: {', '.join(knowledge_graph.hot_topics[:5])}</p>
                <p>新兴趋势: {', '.join(knowledge_graph.emerging_trends[:5])}</p>
            </div>
            
            <div class="info-panel">
                <h3>主要研究空白</h3>
                <ul>
                    {''.join(f'<li>{gap.description}</li>' for gap in knowledge_graph.research_gaps[:5])}
                </ul>
            </div>
        </body>
        </html>
        """
        
        return html_template
    
    def get_graph_statistics(self, knowledge_graph: KnowledgeGraph) -> Dict[str, Any]:
        """获取知识图谱统计信息"""
        # 计算统计指标
        stats = {
            'node_count': len(knowledge_graph.nodes),
            'edge_count': len(knowledge_graph.edges),
            'research_gaps_count': len(knowledge_graph.research_gaps),
            'hot_topics_count': len(knowledge_graph.hot_topics),
            'emerging_trends_count': len(knowledge_graph.emerging_trends)
        }
        
        # 节点类型分布
        node_types = {}
        for node in knowledge_graph.nodes:
            node_types[node.node_type] = node_types.get(node.node_type, 0) + 1
        stats['node_type_distribution'] = node_types
        
        # 关系类型分布
        edge_types = {}
        for edge in knowledge_graph.edges:
            edge_types[edge.relationship_type] = edge_types.get(edge.relationship_type, 0) + 1
        stats['edge_type_distribution'] = edge_types
        
        return stats