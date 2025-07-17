"""
研究方向推荐系统辅助模块

为研究方向建议系统提供辅助功能和评估方法
"""

from typing import Any, Dict, List, Optional
from ..models.research_models import TopicAnalysis, ResearchComplexity, ResearchType


class DirectionRecommenderMixin:
    """研究方向推荐混入类，提供方向评估和生成的辅助方法"""
    
    def _generate_direction_description(self, direction: str, analysis: TopicAnalysis) -> str:
        """生成研究方向描述"""
        descriptions = {
            "性能优化与基准测试": f"针对{analysis.topic}进行系统性的性能优化研究，建立标准化的评估基准",
            "算法改进与创新": f"在{analysis.topic}领域开发新的算法或改进现有算法",
            "实验设计与验证": f"设计严格的实验来验证{analysis.topic}相关的假设和理论",
            "理论框架构建": f"为{analysis.topic}建立完整的理论基础和概念框架",
            "数学模型开发": f"开发描述{analysis.topic}现象的数学模型",
            "系统性文献综述": f"对{analysis.topic}领域进行全面的文献调研和分析",
            "大规模数据分析": f"利用大数据技术分析{analysis.topic}相关的数据模式",
            "典型案例深度分析": f"通过深入分析{analysis.topic}的典型案例获得洞察"
        }
        
        return descriptions.get(direction, f"针对{analysis.topic}的{direction}研究")
    
    def _assess_direction_feasibility(self, direction: str, analysis: TopicAnalysis) -> float:
        """评估研究方向的可行性"""
        base_feasibility = 0.7
        
        # 基于复杂度调整
        complexity_adjustment = {
            ResearchComplexity.LOW: 0.1,
            ResearchComplexity.MEDIUM: 0.0,
            ResearchComplexity.HIGH: -0.1,
            ResearchComplexity.VERY_HIGH: -0.2
        }
        
        feasibility = base_feasibility + complexity_adjustment.get(analysis.complexity_level, 0.0)
        
        # 基于研究类型调整
        if analysis.research_type == ResearchType.SURVEY and "文献" in direction:
            feasibility += 0.1
        elif analysis.research_type == ResearchType.THEORETICAL and "理论" in direction:
            feasibility += 0.1
        elif analysis.research_type == ResearchType.EXPERIMENTAL and "实验" in direction:
            feasibility += 0.1
        
        # 基于资源需求调整
        if "大规模" in direction or "高性能" in direction:
            feasibility -= 0.1
        
        return max(0.1, min(0.95, feasibility))
    
    def _assess_innovation_potential(self, direction: str, analysis: TopicAnalysis) -> float:
        """评估创新潜力"""
        base_innovation = 0.5
        
        # 基于方向类型调整
        innovation_keywords = {
            "创新": 0.3,
            "新型": 0.25,
            "前沿": 0.2,
            "突破": 0.3,
            "改进": 0.15,
            "优化": 0.1,
            "传统": -0.1,
            "标准": -0.05
        }
        
        for keyword, adjustment in innovation_keywords.items():
            if keyword in direction:
                base_innovation += adjustment
        
        # 基于复杂度调整
        if analysis.complexity_level in [ResearchComplexity.HIGH, ResearchComplexity.VERY_HIGH]:
            base_innovation += 0.1
        
        # 基于跨学科特征调整
        if len(analysis.related_fields) > 2:
            base_innovation += 0.1
        
        return max(0.1, min(0.95, base_innovation))
    
    def _estimate_direction_resources(self, direction: str, analysis: TopicAnalysis) -> Dict[str, str]:
        """估算研究方向所需资源"""
        resources = {
            "computational": "中等",
            "human": "中等",
            "financial": "中等",
            "time": "中等"
        }
        
        # 基于方向类型调整
        if "大规模" in direction or "高性能" in direction:
            resources["computational"] = "高"
            resources["financial"] = "高"
        
        if "理论" in direction or "数学" in direction:
            resources["computational"] = "低"
            resources["human"] = "高"  # 需要专业理论人才
        
        if "实验" in direction:
            resources["time"] = "高"
            resources["financial"] = "高"
        
        if "文献" in direction or "综述" in direction:
            resources["computational"] = "低"
            resources["time"] = "中等"
        
        # 基于复杂度调整
        if analysis.complexity_level in [ResearchComplexity.HIGH, ResearchComplexity.VERY_HIGH]:
            for key in resources:
                if resources[key] == "低":
                    resources[key] = "中等"
                elif resources[key] == "中等":
                    resources[key] = "高"
        
        return resources
    
    def _estimate_direction_timeline(self, direction: str, analysis: TopicAnalysis) -> int:
        """估算研究方向时间线（月）"""
        base_timeline = analysis.estimated_duration // 30  # 转换为月
        
        # 基于方向类型调整
        timeline_multipliers = {
            "文献综述": 0.6,
            "理论": 1.2,
            "实验": 1.4,
            "大规模": 1.5,
            "创新": 1.3,
            "优化": 0.9,
            "分析": 0.8
        }
        
        multiplier = 1.0
        for keyword, mult in timeline_multipliers.items():
            if keyword in direction:
                multiplier = mult
                break
        
        return max(1, int(base_timeline * multiplier))
    
    def _estimate_direction_success_probability(self, direction: str, analysis: TopicAnalysis) -> float:
        """估算研究方向成功概率"""
        base_probability = analysis.success_probability
        
        # 基于方向类型调整
        if "传统" in direction or "标准" in direction:
            base_probability += 0.1
        elif "创新" in direction or "前沿" in direction:
            base_probability -= 0.1
        
        if "文献" in direction:
            base_probability += 0.05  # 文献研究相对容易成功
        elif "实验" in direction:
            base_probability -= 0.05  # 实验研究风险较高
        
        return max(0.1, min(0.95, base_probability))
    
    def _assess_market_potential(self, application: str) -> str:
        """评估市场潜力"""
        high_potential = ["医疗诊断", "金融风控", "自动驾驶", "智能制造"]
        medium_potential = ["推荐系统", "图像识别", "语音处理", "商业智能"]
        
        if application in high_potential:
            return "高"
        elif application in medium_potential:
            return "中等"
        else:
            return "低"
    
    def _assess_industry_relevance(self, application: str) -> str:
        """评估行业相关性"""
        high_relevance = ["医疗诊断", "金融风控", "智能制造", "自动驾驶"]
        medium_relevance = ["推荐系统", "商业智能", "安防监控"]
        
        if application in high_relevance:
            return "高"
        elif application in medium_relevance:
            return "中等"
        else:
            return "低"
    
    def _assess_mathematical_complexity(self, extension: str) -> str:
        """评估数学复杂度"""
        high_complexity = ["复杂度理论", "收敛性分析", "稳定性理论"]
        medium_complexity = ["数学模型扩展", "算法理论分析"]
        
        if extension in high_complexity:
            return "高"
        elif extension in medium_complexity:
            return "中等"
        else:
            return "低"
    
    def _assess_technical_challenge(self, innovation: str) -> str:
        """评估技术挑战"""
        high_challenge = ["新型算法设计", "自动化工具开发"]
        medium_challenge = ["混合方法论开发", "评估指标创新"]
        
        if innovation in high_challenge:
            return "高"
        elif innovation in medium_challenge:
            return "中等"
        else:
            return "低"
    
    def _enrich_directions_with_details(self, directions: Dict[str, List[Dict[str, Any]]], 
                                      analysis: TopicAnalysis) -> Dict[str, List[Dict[str, Any]]]:
        """为研究方向添加详细信息"""
        enriched = {}
        
        for category, direction_list in directions.items():
            enriched[category] = []
            for direction in direction_list:
                # 添加优先级分数
                direction["priority_score"] = self._calculate_direction_priority(direction, analysis)
                
                # 添加适用性评估
                direction["suitability"] = self._assess_direction_suitability(direction, analysis)
                
                # 添加风险评估
                direction["risk_factors"] = self._identify_direction_risks(direction, analysis)
                
                # 添加前置条件
                direction["prerequisites"] = self._identify_direction_prerequisites(direction, analysis)
                
                enriched[category].append(direction)
            
            # 按优先级排序
            enriched[category].sort(key=lambda x: x["priority_score"], reverse=True)
        
        return enriched
    
    def _calculate_direction_priority(self, direction: Dict[str, Any], analysis: TopicAnalysis) -> float:
        """计算研究方向优先级分数"""
        weights = {
            "feasibility": 0.3,
            "innovation_potential": 0.25,
            "success_probability": 0.25,
            "resource_efficiency": 0.2
        }
        
        feasibility = direction.get("feasibility", 0.5)
        innovation = direction.get("innovation_potential", 0.5)
        success_prob = direction.get("success_probability", 0.5)
        
        # 资源效率 = 1 / 资源需求强度
        resource_intensity = self._calculate_resource_intensity(direction.get("resource_requirements", {}))
        resource_efficiency = 1.0 - resource_intensity
        
        priority = (
            weights["feasibility"] * feasibility +
            weights["innovation_potential"] * innovation +
            weights["success_probability"] * success_prob +
            weights["resource_efficiency"] * resource_efficiency
        )
        
        return min(1.0, max(0.0, priority))
    
    def _calculate_resource_intensity(self, resources: Dict[str, str]) -> float:
        """计算资源需求强度"""
        intensity_mapping = {"低": 0.2, "中等": 0.5, "高": 0.8}
        
        if not resources:
            return 0.5
        
        total_intensity = sum(intensity_mapping.get(level, 0.5) for level in resources.values())
        return total_intensity / len(resources)
    
    def _assess_direction_suitability(self, direction: Dict[str, Any], analysis: TopicAnalysis) -> str:
        """评估研究方向适用性"""
        priority_score = direction.get("priority_score", 0.5)
        
        if priority_score >= 0.7:
            return "高度适用"
        elif priority_score >= 0.5:
            return "适用"
        else:
            return "需要谨慎考虑"
    
    def _identify_direction_risks(self, direction: Dict[str, Any], analysis: TopicAnalysis) -> List[str]:
        """识别研究方向风险"""
        risks = []
        
        # 基于可行性的风险
        feasibility = direction.get("feasibility", 0.5)
        if feasibility < 0.4:
            risks.append("可行性风险较高")
        
        # 基于创新性的风险
        innovation = direction.get("innovation_potential", 0.5)
        if innovation > 0.8:
            risks.append("创新风险，可能面临技术不确定性")
        
        # 基于资源需求的风险
        resources = direction.get("resource_requirements", {})
        if any(level == "高" for level in resources.values()):
            risks.append("资源需求风险")
        
        # 基于时间线的风险
        timeline = direction.get("expected_timeline", 6)
        if timeline > 12:
            risks.append("时间风险，项目周期较长")
        
        return risks if risks else ["风险较低"]
    
    def _identify_direction_prerequisites(self, direction: Dict[str, Any], analysis: TopicAnalysis) -> List[str]:
        """识别研究方向前置条件"""
        prerequisites = []
        
        direction_title = direction.get("title", "")
        
        # 基于方向类型的前置条件
        if "理论" in direction_title:
            prerequisites.extend(["扎实的数学基础", "理论分析能力"])
        
        if "实验" in direction_title:
            prerequisites.extend(["实验设计经验", "统计分析能力"])
        
        if "大规模" in direction_title:
            prerequisites.extend(["高性能计算资源", "分布式系统经验"])
        
        if "文献" in direction_title:
            prerequisites.extend(["文献检索能力", "批判性思维"])
        
        # 基于复杂度的前置条件
        if analysis.complexity_level in [ResearchComplexity.HIGH, ResearchComplexity.VERY_HIGH]:
            prerequisites.extend(["项目管理经验", "团队协作能力"])
        
        # 基于跨学科特征的前置条件
        if len(analysis.related_fields) > 2:
            prerequisites.append("跨学科知识整合能力")
        
        return prerequisites if prerequisites else ["基础研究能力"]