"""
智能关键词生成器

负责从研究主题中提取和扩展关键词，支持文献检索和分析
"""

import re
import math
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import Counter, defaultdict
from dataclasses import dataclass

from .base_component import BaseComponent
from ..models.research_models import TopicAnalysis, ResearchType, ResearchComplexity
from .exceptions import ValidationError, ProcessingError


@dataclass
class KeywordAnalysis:
    """关键词分析结果"""
    primary_keywords: List[str]
    secondary_keywords: List[str]
    domain_keywords: List[str]
    method_keywords: List[str]
    expanded_keywords: List[str]
    keyword_combinations: List[str]
    search_strategies: List[Dict[str, Any]]
    relevance_scores: Dict[str, float]
    
    def validate(self) -> bool:
        """验证关键词分析结果"""
        return (
            len(self.primary_keywords) > 0 and
            len(self.search_strategies) > 0 and
            all(score >= 0 for score in self.relevance_scores.values())
        )


class KeywordGeneratorComponent(BaseComponent):
    """智能关键词生成器组件"""
    
    def get_required_configs(self) -> List[str]:
        """获取必需的配置项"""
        return ['max_keywords', 'similarity_threshold']
    
    def _setup_component(self):
        """设置组件特定的初始化逻辑"""
        # 初始化关键词词典和语义模型
        self._setup_keyword_dictionaries()
        self._setup_semantic_models()
        self._setup_domain_vocabularies()
        self.logger.info("智能关键词生成器初始化完成")
    
    def _setup_keyword_dictionaries(self):
        """设置关键词词典"""
        # 停用词列表
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'from', 'up', 'out', 'down', 'off', 'over', 'under', 'again', 'further',
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
            'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very'
        }
        
        # 学术关键词指示词
        self.academic_indicators = {
            'method': ['method', 'approach', 'technique', 'algorithm', 'framework', 'model'],
            'analysis': ['analysis', 'evaluation', 'assessment', 'comparison', 'study'],
            'application': ['application', 'implementation', 'deployment', 'system'],
            'theory': ['theory', 'theoretical', 'principle', 'concept', 'foundation'],
            'empirical': ['empirical', 'experimental', 'data', 'evidence', 'observation']
        }
        
        # 研究动词
        self.research_verbs = {
            'investigate', 'analyze', 'evaluate', 'compare', 'develop', 'design',
            'implement', 'optimize', 'improve', 'enhance', 'explore', 'examine',
            'study', 'research', 'propose', 'present', 'demonstrate', 'validate'
        }
    
    def _setup_semantic_models(self):
        """设置语义相似度模型"""
        # 简化的语义相似度计算
        # 在实际应用中，这里可以集成更复杂的NLP模型
        self.semantic_similarity_cache = {}
        
        # 同义词词典
        self.synonyms = {
            'artificial intelligence': ['AI', 'machine intelligence', 'computational intelligence'],
            'machine learning': ['ML', 'statistical learning', 'automated learning'],
            'deep learning': ['neural networks', 'deep neural networks', 'DNN'],
            'natural language processing': ['NLP', 'computational linguistics', 'text processing'],
            'computer vision': ['image processing', 'visual recognition', 'image analysis'],
            'data mining': ['knowledge discovery', 'data analysis', 'pattern recognition'],
            'optimization': ['optimization', 'minimization', 'maximization'],
            'algorithm': ['method', 'procedure', 'technique', 'approach'],
            'model': ['framework', 'system', 'architecture'],
            'analysis': ['evaluation', 'assessment', 'examination', 'study']
        }
        
        # 反向同义词映射
        self.reverse_synonyms = {}
        for key, values in self.synonyms.items():
            for value in values:
                if value not in self.reverse_synonyms:
                    self.reverse_synonyms[value] = []
                self.reverse_synonyms[value].append(key)
    
    def _setup_domain_vocabularies(self):
        """设置领域词汇表"""
        self.domain_vocabularies = {
            'computer_science': {
                'core': ['algorithm', 'data structure', 'programming', 'software', 'hardware'],
                'ai_ml': ['artificial intelligence', 'machine learning', 'neural network', 'deep learning'],
                'systems': ['distributed system', 'database', 'network', 'security', 'cloud computing'],
                'theory': ['complexity theory', 'formal methods', 'computational theory']
            },
            'mathematics': {
                'core': ['theorem', 'proof', 'equation', 'function', 'variable'],
                'statistics': ['probability', 'distribution', 'regression', 'correlation', 'hypothesis'],
                'optimization': ['linear programming', 'convex optimization', 'gradient descent'],
                'analysis': ['calculus', 'differential equation', 'linear algebra', 'topology']
            },
            'biology': {
                'core': ['gene', 'protein', 'cell', 'organism', 'evolution'],
                'molecular': ['DNA', 'RNA', 'enzyme', 'metabolism', 'pathway'],
                'bioinformatics': ['sequence analysis', 'phylogenetics', 'genomics', 'proteomics'],
                'systems': ['systems biology', 'network biology', 'computational biology']
            },
            'physics': {
                'core': ['particle', 'wave', 'energy', 'force', 'field'],
                'quantum': ['quantum mechanics', 'quantum computing', 'entanglement', 'superposition'],
                'classical': ['mechanics', 'thermodynamics', 'electromagnetism', 'optics'],
                'modern': ['relativity', 'cosmology', 'particle physics', 'condensed matter']
            },
            'engineering': {
                'core': ['design', 'system', 'control', 'optimization', 'modeling'],
                'electrical': ['circuit', 'signal processing', 'communication', 'power system'],
                'mechanical': ['dynamics', 'materials', 'manufacturing', 'robotics'],
                'software': ['software engineering', 'requirements', 'testing', 'architecture']
            }
        }
    
    def generate_keywords(self, topic: str, context: Dict[str, Any] = None) -> KeywordAnalysis:
        """
        生成智能关键词分析
        
        Args:
            topic: 研究主题
            context: 上下文信息（主题分析结果等）
            
        Returns:
            KeywordAnalysis: 关键词分析结果
        """
        try:
            self.log_operation("generate_keywords", {"topic_length": len(topic)})
            
            # 验证输入
            if not topic or not topic.strip():
                raise ValidationError("研究主题不能为空")
            
            context = context or {}
            topic = topic.strip()
            
            # 1. 提取基础关键词
            primary_keywords = self._extract_primary_keywords(topic)
            
            # 2. 生成次要关键词
            secondary_keywords = self._generate_secondary_keywords(topic, primary_keywords)
            
            # 3. 识别领域关键词
            domain_keywords = self._identify_domain_keywords(topic, primary_keywords)
            
            # 4. 提取方法关键词
            method_keywords = self._extract_method_keywords(topic, primary_keywords)
            
            # 5. 扩展关键词
            expanded_keywords = self._expand_keywords(primary_keywords + secondary_keywords)
            
            # 6. 生成关键词组合
            keyword_combinations = self._generate_keyword_combinations(
                primary_keywords, secondary_keywords, domain_keywords
            )
            
            # 7. 计算相关性分数
            all_keywords = list(set(
                primary_keywords + secondary_keywords + domain_keywords + 
                method_keywords + expanded_keywords
            ))
            relevance_scores = self._calculate_relevance_scores(topic, all_keywords)
            
            # 8. 生成搜索策略
            search_strategies = self._generate_search_strategies(
                primary_keywords, secondary_keywords, domain_keywords, 
                method_keywords, context
            )
            
            # 创建关键词分析结果
            analysis = KeywordAnalysis(
                primary_keywords=primary_keywords,
                secondary_keywords=secondary_keywords,
                domain_keywords=domain_keywords,
                method_keywords=method_keywords,
                expanded_keywords=expanded_keywords,
                keyword_combinations=keyword_combinations,
                search_strategies=search_strategies,
                relevance_scores=relevance_scores
            )
            
            # 验证结果
            if not analysis.validate():
                self.logger.warning(f"关键词分析结果验证失败: primary={len(analysis.primary_keywords)}, strategies={len(analysis.search_strategies)}")
                # 如果主要关键词为空，至少添加一个基础关键词
                if not analysis.primary_keywords:
                    analysis.primary_keywords = [topic.split()[0]] if topic.split() else ['research']
                # 如果搜索策略为空，添加一个基础策略
                if not analysis.search_strategies:
                    analysis.search_strategies = [{
                        "name": "基础搜索策略",
                        "description": "使用主题进行基础搜索",
                        "keywords": analysis.primary_keywords,
                        "search_query": " OR ".join(analysis.primary_keywords),
                        "expected_results": "medium_volume",
                        "precision": "medium",
                        "recall": "medium"
                    }]
            
            self.update_metric("keywords_generated", 
                             self.get_metric("keywords_generated") or 0 + 1)
            
            self.logger.info(f"关键词生成完成: {len(all_keywords)}个关键词, "
                           f"{len(search_strategies)}个搜索策略")
            
            return analysis
            
        except Exception as e:
            self.handle_error(e, "generate_keywords")
    
    def _extract_primary_keywords(self, topic: str) -> List[str]:
        """提取主要关键词"""
        # 清理和标准化文本
        cleaned_topic = self._clean_text(topic)
        
        # 提取单词和短语
        words = self._tokenize(cleaned_topic)
        phrases = self._extract_phrases(words)
        
        # 过滤停用词
        filtered_words = [w for w in words if w.lower() not in self.stop_words and len(w) > 2]
        
        # 计算词频和重要性
        word_scores = self._calculate_word_importance(filtered_words, cleaned_topic)
        phrase_scores = self._calculate_phrase_importance(phrases, cleaned_topic)
        
        # 合并和排序
        all_candidates = list(word_scores.items()) + list(phrase_scores.items())
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 选择前N个作为主要关键词
        max_keywords = self.get_config('max_keywords', 10)
        primary_keywords = [kw for kw, score in all_candidates[:max_keywords]]
        
        return primary_keywords
    
    def _clean_text(self, text: str) -> str:
        """清理文本"""
        # 转换为小写
        text = text.lower()
        
        # 移除特殊字符，保留字母、数字、空格和连字符
        text = re.sub(r'[^a-zA-Z0-9\s\-]', ' ', text)
        
        # 标准化空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _tokenize(self, text: str) -> List[str]:
        """分词"""
        # 简单的基于空格的分词
        words = text.split()
        
        # 处理连字符
        expanded_words = []
        for word in words:
            if '-' in word:
                # 添加原词和分割后的词
                expanded_words.append(word)
                expanded_words.extend(word.split('-'))
            else:
                expanded_words.append(word)
        
        return [w for w in expanded_words if w]
    
    def _extract_phrases(self, words: List[str]) -> List[str]:
        """提取短语"""
        phrases = []
        
        # 提取2-4个词的短语
        for n in range(2, 5):
            for i in range(len(words) - n + 1):
                phrase = ' '.join(words[i:i+n])
                # 过滤包含停用词的短语
                if not any(word in self.stop_words for word in words[i:i+n]):
                    phrases.append(phrase)
        
        return phrases
    
    def _calculate_word_importance(self, words: List[str], context: str) -> Dict[str, float]:
        """计算单词重要性"""
        word_scores = {}
        word_counts = Counter(words)
        
        for word, count in word_counts.items():
            score = 0.0
            
            # 基础频率分数
            score += math.log(count + 1)
            
            # 长度奖励（较长的词通常更重要）
            score += len(word) * 0.1
            
            # 学术指示词奖励
            for category, indicators in self.academic_indicators.items():
                if any(indicator in word for indicator in indicators):
                    score += 2.0
                    break
            
            # 研究动词奖励
            if word in self.research_verbs:
                score += 1.5
            
            # 领域词汇奖励
            for domain, vocab in self.domain_vocabularies.items():
                for category, terms in vocab.items():
                    if any(term in word for term in terms):
                        score += 1.0
                        break
            
            word_scores[word] = score
        
        return word_scores
    
    def _calculate_phrase_importance(self, phrases: List[str], context: str) -> Dict[str, float]:
        """计算短语重要性"""
        phrase_scores = {}
        phrase_counts = Counter(phrases)
        
        for phrase, count in phrase_counts.items():
            score = 0.0
            
            # 基础频率分数
            score += math.log(count + 1)
            
            # 短语长度奖励
            words_in_phrase = phrase.split()
            score += len(words_in_phrase) * 0.5
            
            # 学术短语奖励
            for category, indicators in self.academic_indicators.items():
                if any(indicator in phrase for indicator in indicators):
                    score += 3.0
                    break
            
            # 领域短语奖励
            for domain, vocab in self.domain_vocabularies.items():
                for category, terms in vocab.items():
                    if any(term in phrase for term in terms):
                        score += 2.0
                        break
            
            # 同义词奖励
            if phrase in self.synonyms or phrase in self.reverse_synonyms:
                score += 1.5
            
            phrase_scores[phrase] = score
        
        return phrase_scores
    
    def _generate_secondary_keywords(self, topic: str, primary_keywords: List[str]) -> List[str]:
        """生成次要关键词"""
        secondary_keywords = []
        
        # 基于主要关键词生成变体
        for keyword in primary_keywords:
            # 添加单复数变体
            variants = self._generate_word_variants(keyword)
            secondary_keywords.extend(variants)
            
            # 添加相关术语
            related_terms = self._find_related_terms(keyword)
            secondary_keywords.extend(related_terms)
        
        # 去重并过滤
        secondary_keywords = list(set(secondary_keywords))
        secondary_keywords = [kw for kw in secondary_keywords if kw not in primary_keywords]
        
        return secondary_keywords[:15]  # 限制数量
    
    def _generate_word_variants(self, word: str) -> List[str]:
        """生成单词变体"""
        variants = []
        
        # 简单的单复数处理
        if word.endswith('s') and len(word) > 3:
            variants.append(word[:-1])  # 去掉s
        elif not word.endswith('s'):
            variants.append(word + 's')  # 加上s
        
        # 动词变体
        if word.endswith('ing'):
            base = word[:-3]
            variants.extend([base, base + 'e', base + 'ed'])
        elif word.endswith('ed'):
            base = word[:-2]
            variants.extend([base, base + 'ing'])
        
        # 形容词变体
        if word.endswith('al'):
            variants.append(word[:-2] + 'ic')
        elif word.endswith('ic'):
            variants.append(word[:-2] + 'al')
        
        return [v for v in variants if len(v) > 2]
    
    def _find_related_terms(self, keyword: str) -> List[str]:
        """查找相关术语"""
        related_terms = []
        
        # 从同义词词典查找
        if keyword in self.synonyms:
            related_terms.extend(self.synonyms[keyword])
        
        if keyword in self.reverse_synonyms:
            related_terms.extend(self.reverse_synonyms[keyword])
        
        # 从领域词汇查找相关术语
        for domain, vocab in self.domain_vocabularies.items():
            for category, terms in vocab.items():
                if keyword in terms:
                    # 添加同类别的其他术语
                    related_terms.extend([t for t in terms if t != keyword])
                    break
        
        return list(set(related_terms))
    
    def _identify_domain_keywords(self, topic: str, primary_keywords: List[str]) -> List[str]:
        """识别领域关键词"""
        domain_keywords = []
        topic_lower = topic.lower()
        all_keywords = ' '.join(primary_keywords).lower()
        
        # 检查每个领域的词汇
        for domain, vocab in self.domain_vocabularies.items():
            domain_score = 0
            matched_terms = []
            
            for category, terms in vocab.items():
                for term in terms:
                    # 更宽松的匹配条件
                    if (term in topic_lower or 
                        term in all_keywords or 
                        any(term in kw.lower() for kw in primary_keywords) or
                        any(word in term for word in topic_lower.split() if len(word) > 3)):
                        domain_score += 1
                        matched_terms.append(term)
            
            # 降低匹配阈值，如果该领域有匹配，添加相关术语
            if domain_score >= 1:
                domain_keywords.extend(matched_terms)
                # 添加该领域的核心术语
                if 'core' in vocab:
                    domain_keywords.extend(vocab['core'][:2])
        
        # 如果仍然没有领域关键词，基于主要关键词推断
        if not domain_keywords:
            for keyword in primary_keywords[:3]:
                keyword_lower = keyword.lower()
                if any(cs_term in keyword_lower for cs_term in ['algorithm', 'computer', 'software', 'data', 'machine', 'artificial']):
                    domain_keywords.extend(['algorithm', 'computer science', 'data structure'])
                    break
                elif any(math_term in keyword_lower for math_term in ['mathematical', 'equation', 'optimization', 'statistical']):
                    domain_keywords.extend(['mathematics', 'statistics', 'optimization'])
                    break
        
        return list(set(domain_keywords))
    
    def _extract_method_keywords(self, topic: str, primary_keywords: List[str]) -> List[str]:
        """提取方法关键词"""
        method_keywords = []
        
        # 检查学术指示词中的方法类关键词
        for keyword in primary_keywords:
            for category, indicators in self.academic_indicators.items():
                if category == 'method' and any(indicator in keyword for indicator in indicators):
                    method_keywords.append(keyword)
        
        # 添加常见的研究方法术语
        common_methods = [
            'statistical analysis', 'machine learning', 'deep learning',
            'optimization', 'simulation', 'modeling', 'algorithm',
            'experimental design', 'case study', 'survey', 'interview'
        ]
        
        topic_lower = topic.lower()
        for method in common_methods:
            if method in topic_lower:
                method_keywords.append(method)
        
        return list(set(method_keywords))
    
    def _expand_keywords(self, keywords: List[str]) -> List[str]:
        """扩展关键词"""
        expanded = []
        
        for keyword in keywords:
            # 添加同义词
            if keyword in self.synonyms:
                expanded.extend(self.synonyms[keyword])
            
            # 添加相关术语
            related = self._find_related_terms(keyword)
            expanded.extend(related[:2])  # 限制每个关键词的扩展数量
        
        return list(set(expanded))
    
    def _generate_keyword_combinations(self, primary: List[str], secondary: List[str], 
                                     domain: List[str]) -> List[str]:
        """生成关键词组合"""
        combinations = []
        
        # 主要关键词之间的组合
        for i, kw1 in enumerate(primary[:5]):  # 限制数量避免组合爆炸
            for kw2 in primary[i+1:6]:
                combinations.append(f"{kw1} AND {kw2}")
                combinations.append(f"{kw1} OR {kw2}")
        
        # 主要关键词与领域关键词的组合
        for primary_kw in primary[:3]:
            for domain_kw in domain[:3]:
                combinations.append(f"{primary_kw} AND {domain_kw}")
        
        # 添加一些常用的搜索模式
        for kw in primary[:3]:
            combinations.append(f'"{kw}"')  # 精确匹配
            combinations.append(f"{kw}*")   # 通配符搜索
        
        return combinations[:20]  # 限制组合数量
    
    def _calculate_relevance_scores(self, topic: str, keywords: List[str]) -> Dict[str, float]:
        """计算关键词相关性分数"""
        scores = {}
        topic_lower = topic.lower()
        
        for keyword in keywords:
            score = 0.0
            keyword_lower = keyword.lower()
            
            # 直接匹配分数
            if keyword_lower in topic_lower:
                score += 1.0
            
            # 部分匹配分数
            keyword_words = keyword_lower.split()
            topic_words = topic_lower.split()
            
            matches = sum(1 for kw_word in keyword_words if kw_word in topic_words)
            if len(keyword_words) > 0:
                score += matches / len(keyword_words) * 0.5
            
            # 语义相似度分数（简化版本）
            semantic_score = self._calculate_semantic_similarity(keyword_lower, topic_lower)
            score += semantic_score * 0.3
            
            # 学术重要性分数
            academic_score = self._calculate_academic_importance(keyword)
            score += academic_score * 0.2
            
            scores[keyword] = min(1.0, score)  # 限制在0-1之间
        
        return scores
    
    def _calculate_semantic_similarity(self, keyword: str, topic: str) -> float:
        """计算语义相似度（简化版本）"""
        # 这里使用简单的词汇重叠计算
        # 在实际应用中可以使用更复杂的语义模型
        
        keyword_words = set(keyword.split())
        topic_words = set(topic.split())
        
        if not keyword_words or not topic_words:
            return 0.0
        
        intersection = keyword_words.intersection(topic_words)
        union = keyword_words.union(topic_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_academic_importance(self, keyword: str) -> float:
        """计算学术重要性"""
        importance = 0.0
        
        # 检查是否为学术指示词
        for category, indicators in self.academic_indicators.items():
            if any(indicator in keyword for indicator in indicators):
                importance += 0.3
                break
        
        # 检查是否为领域核心词汇
        for domain, vocab in self.domain_vocabularies.items():
            if 'core' in vocab and any(term in keyword for term in vocab['core']):
                importance += 0.4
                break
        
        # 检查是否为研究方法词
        if any(method in keyword for method in ['method', 'approach', 'technique', 'algorithm']):
            importance += 0.3
        
        return min(1.0, importance)
    
    def _generate_search_strategies(self, primary: List[str], secondary: List[str],
                                  domain: List[str], method: List[str],
                                  context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成搜索策略"""
        strategies = []
        
        # 策略1: 广泛搜索
        broad_keywords = primary[:3] + domain[:2]
        strategies.append({
            "name": "广泛搜索策略",
            "description": "使用核心关键词进行广泛搜索，获取大量相关文献",
            "keywords": broad_keywords,
            "search_query": " OR ".join(broad_keywords),
            "expected_results": "high_volume",
            "precision": "medium",
            "recall": "high"
        })
        
        # 策略2: 精确搜索
        precise_keywords = primary[:2]
        strategies.append({
            "name": "精确搜索策略", 
            "description": "使用精确匹配获取高度相关的文献",
            "keywords": precise_keywords,
            "search_query": " AND ".join([f'"{kw}"' for kw in precise_keywords]),
            "expected_results": "low_volume",
            "precision": "high",
            "recall": "medium"
        })
        
        # 策略3: 方法导向搜索
        if method:
            strategies.append({
                "name": "方法导向搜索策略",
                "description": "专注于研究方法和技术的文献搜索",
                "keywords": method[:3] + primary[:2],
                "search_query": f"({' OR '.join(method[:3])}) AND ({' OR '.join(primary[:2])})",
                "expected_results": "medium_volume",
                "precision": "high",
                "recall": "medium"
            })
        
        # 策略4: 领域特定搜索
        if domain:
            strategies.append({
                "name": "领域特定搜索策略",
                "description": "在特定学科领域内进行深度搜索",
                "keywords": domain[:3] + primary[:2],
                "search_query": f"({' OR '.join(domain[:3])}) AND ({' OR '.join(primary[:2])})",
                "expected_results": "medium_volume", 
                "precision": "high",
                "recall": "medium"
            })
        
        # 策略5: 时间敏感搜索
        recent_keywords = primary[:3]
        strategies.append({
            "name": "最新研究搜索策略",
            "description": "搜索最近发表的相关研究",
            "keywords": recent_keywords,
            "search_query": " OR ".join(recent_keywords),
            "time_filter": "last_3_years",
            "expected_results": "medium_volume",
            "precision": "medium",
            "recall": "medium"
        })
        
        return strategies
    
    def optimize_search_query(self, keywords: List[str], search_context: Dict[str, Any]) -> str:
        """优化搜索查询"""
        try:
            self.log_operation("optimize_search_query", {
                "keyword_count": len(keywords),
                "context_keys": list(search_context.keys())
            })
            
            if not keywords:
                raise ValidationError("关键词列表不能为空")
            
            # 根据搜索上下文优化查询
            search_type = search_context.get('search_type', 'balanced')
            max_results = search_context.get('max_results', 1000)
            
            if search_type == 'broad':
                # 广泛搜索：使用OR连接
                query = " OR ".join(keywords[:5])
            elif search_type == 'precise':
                # 精确搜索：使用AND连接并加引号
                query = " AND ".join([f'"{kw}"' for kw in keywords[:3]])
            elif search_type == 'balanced':
                # 平衡搜索：混合策略
                primary = keywords[:2]
                secondary = keywords[2:5]
                query = f"({' AND '.join(primary)}) OR ({' OR '.join(secondary)})"
            else:
                # 默认策略
                query = " OR ".join(keywords[:4])
            
            # 添加时间过滤
            if 'time_filter' in search_context:
                time_filter = search_context['time_filter']
                query += f" AND publication_date:{time_filter}"
            
            # 添加领域过滤
            if 'domain_filter' in search_context:
                domain = search_context['domain_filter']
                query += f" AND domain:{domain}"
            
            self.logger.info(f"搜索查询优化完成: {query}")
            return query
            
        except Exception as e:
            self.handle_error(e, "optimize_search_query")
    
    def expand_keywords_with_context(self, keywords: List[str], 
                                   topic_analysis: Optional[TopicAnalysis] = None) -> List[str]:
        """基于主题分析结果扩展关键词"""
        try:
            self.log_operation("expand_keywords_with_context", {
                "keyword_count": len(keywords),
                "has_topic_analysis": topic_analysis is not None
            })
            
            expanded_keywords = keywords.copy()
            
            if topic_analysis:
                # 基于研究类型添加相关关键词
                if topic_analysis.research_type == ResearchType.EXPERIMENTAL:
                    expanded_keywords.extend([
                        'experiment', 'experimental design', 'control group',
                        'statistical analysis', 'hypothesis testing'
                    ])
                elif topic_analysis.research_type == ResearchType.THEORETICAL:
                    expanded_keywords.extend([
                        'theory', 'theoretical framework', 'mathematical model',
                        'formal analysis', 'proof'
                    ])
                elif topic_analysis.research_type == ResearchType.SURVEY:
                    expanded_keywords.extend([
                        'survey', 'literature review', 'systematic review',
                        'meta-analysis', 'state-of-the-art'
                    ])
                
                # 基于复杂度添加相关关键词
                if topic_analysis.complexity_level in [ResearchComplexity.HIGH, ResearchComplexity.VERY_HIGH]:
                    expanded_keywords.extend([
                        'complex system', 'interdisciplinary', 'multi-modal',
                        'advanced method', 'novel approach'
                    ])
                
                # 基于相关领域添加关键词
                for field in topic_analysis.related_fields:
                    field_lower = field.lower().replace(' ', '_')
                    if field_lower in self.domain_vocabularies:
                        vocab = self.domain_vocabularies[field_lower]
                        if 'core' in vocab:
                            expanded_keywords.extend(vocab['core'][:3])
            
            # 去重并返回
            expanded_keywords = list(set(expanded_keywords))
            
            self.logger.info(f"关键词扩展完成: {len(keywords)} -> {len(expanded_keywords)}")
            return expanded_keywords
            
        except Exception as e:
            self.handle_error(e, "expand_keywords_with_context")