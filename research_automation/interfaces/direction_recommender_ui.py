"""
研究方向建议系统用户界面

提供交互式的研究方向建议功能
"""

import json
from typing import Dict, Any, List
from ..core.research_planner import ResearchPlannerComponent


class DirectionRecommenderUI:
    """研究方向建议系统用户界面"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化界面"""
        self.planner = ResearchPlannerComponent(config=config)
    
    def interactive_recommendation(self) -> Dict[str, Any]:
        """交互式研究方向推荐"""
        print("=== 智能研究方向建议系统 ===\n")
        
        # 收集用户输入
        topic = self._get_research_topic()
        context = self._get_user_context()
        
        print("\n正在分析研究主题并生成建议...")
        
        try:
            # 生成研究方向建议
            result = self.planner.generate_research_directions(topic, context)
            
            # 显示结果
            self._display_results(result)
            
            return result
            
        except Exception as e:
            print(f"生成建议时出错: {str(e)}")
            return {}
    
    def _get_research_topic(self) -> str:
        """获取研究主题"""
        print("请输入您的研究主题:")
        print("例如: 'Artificial intelligence for healthcare applications'")
        print("     '深度学习在图像识别中的应用'")
        print("     'Quantum computing algorithms optimization'")
        
        while True:
            topic = input("\n研究主题: ").strip()
            if topic:
                return topic
            print("请输入有效的研究主题")
    
    def _get_user_context(self) -> Dict[str, Any]:
        """获取用户上下文信息"""
        context = {}
        
        print("\n=== 个人信息收集 ===")
        
        # 经验水平
        print("\n1. 您的研究经验水平:")
        print("   a) 初学者 (beginner)")
        print("   b) 中级 (intermediate)")  
        print("   c) 高级 (advanced)")
        
        exp_choice = input("请选择 (a/b/c): ").strip().lower()
        exp_mapping = {'a': 'beginner', 'b': 'intermediate', 'c': 'advanced'}
        context['user_experience'] = exp_mapping.get(exp_choice, 'intermediate')
        
        # 可用资源
        print("\n2. 您的可用资源情况:")
        print("   a) 有限 (limited)")
        print("   b) 中等 (moderate)")
        print("   c) 充足 (abundant)")
        
        res_choice = input("请选择 (a/b/c): ").strip().lower()
        res_mapping = {'a': 'limited', 'b': 'moderate', 'c': 'abundant'}
        context['available_resources'] = res_mapping.get(res_choice, 'moderate')
        
        # 时间约束
        print("\n3. 您的时间约束:")
        print("   a) 3个月 (3_months)")
        print("   b) 6个月 (6_months)")
        print("   c) 12个月 (12_months)")
        print("   d) 灵活安排 (flexible)")
        
        time_choice = input("请选择 (a/b/c/d): ").strip().lower()
        time_mapping = {'a': '3_months', 'b': '6_months', 'c': '12_months', 'd': 'flexible'}
        context['time_constraint'] = time_mapping.get(time_choice, '6_months')
        
        # 研究目标
        print("\n4. 您的主要研究目标 (可选):")
        goal = input("研究目标: ").strip()
        if goal:
            context['research_goal'] = goal
        
        return context
    
    def _display_results(self, result: Dict[str, Any]):
        """显示推荐结果"""
        if not result:
            print("未能生成有效的建议结果")
            return
        
        print("\n" + "="*60)
        print("研究方向建议结果")
        print("="*60)
        
        # 显示主题分析
        self._display_topic_analysis(result.get('topic_analysis'))
        
        # 显示研究方向分类
        self._display_research_directions(result.get('research_directions', {}))
        
        # 显示个性化推荐
        self._display_personalized_recommendations(result.get('personalized_recommendations', {}))
        
        # 显示选择指导
        self._display_selection_guidance(result.get('selection_guidance', {}))
        
        # 显示实施建议
        self._display_implementation_suggestions(result.get('implementation_suggestions', {}))
    
    def _display_topic_analysis(self, analysis):
        """显示主题分析"""
        if not analysis:
            return
        
        print(f"\n📊 主题分析")
        print(f"   研究主题: {analysis.topic}")
        print(f"   复杂度等级: {analysis.complexity_level.value}")
        print(f"   复杂度分数: {analysis.complexity_score:.2f}")
        print(f"   研究类型: {analysis.research_type.value}")
        print(f"   预估时长: {analysis.estimated_duration}天")
        print(f"   成功概率: {analysis.success_probability:.1%}")
        print(f"   相关领域: {', '.join(analysis.related_fields)}")
        
        if analysis.potential_challenges:
            print(f"   潜在挑战: {', '.join(analysis.potential_challenges[:3])}")
    
    def _display_research_directions(self, directions: Dict[str, List[Dict[str, Any]]]):
        """显示研究方向分类"""
        if not directions:
            return
        
        print(f"\n🎯 研究方向分类")
        
        for category, direction_list in directions.items():
            if not direction_list:
                continue
                
            print(f"\n   {category}:")
            for i, direction in enumerate(direction_list[:2], 1):  # 每类显示前2个
                print(f"      {i}. {direction['title']}")
                print(f"         可行性: {direction.get('feasibility', 0):.1%}")
                print(f"         创新性: {direction.get('innovation_potential', 0):.1%}")
                print(f"         时间线: {direction.get('expected_timeline', 'N/A')}个月")
                print(f"         适用性: {direction.get('suitability', 'N/A')}")
    
    def _display_personalized_recommendations(self, recommendations: Dict[str, Any]):
        """显示个性化推荐"""
        if not recommendations:
            return
        
        print(f"\n⭐ 个性化推荐")
        
        # 顶级推荐
        top_recs = recommendations.get('top_recommendations', [])
        if top_recs:
            print(f"\n   🏆 顶级推荐:")
            for i, rec in enumerate(top_recs[:3], 1):
                print(f"      {i}. {rec['title']} ({rec['category']})")
                print(f"         个性化评分: {rec.get('personalized_score', 0):.2f}")
                print(f"         推荐理由: {rec.get('recommendation_reason', 'N/A')}")
                print(f"         适用性: {rec.get('suitability', 'N/A')}")
                print()
        
        # 经验匹配度
        exp_match = recommendations.get('experience_match', {})
        if exp_match:
            print(f"   📈 经验匹配度: {exp_match.get('match_score', 0):.1%}")
            print(f"   评估结果: {exp_match.get('level_assessment', 'N/A')}")
        
        # 资源对齐度
        res_align = recommendations.get('resource_alignment', {})
        if res_align:
            print(f"   💰 资源对齐度: {res_align.get('alignment_score', 0):.1%}")
            print(f"   资源评估: {res_align.get('resource_assessment', 'N/A')}")
    
    def _display_selection_guidance(self, guidance: Dict[str, Any]):
        """显示选择指导"""
        if not guidance:
            return
        
        print(f"\n🧭 选择指导")
        
        # 选择标准
        criteria = guidance.get('selection_criteria', [])
        if criteria:
            print(f"\n   选择标准:")
            for criterion in criteria[:3]:  # 显示前3个标准
                print(f"      • {criterion['name']} (权重: {criterion['weight']:.1%})")
                print(f"        {criterion['description']}")
        
        # 选择建议
        recommendations = guidance.get('selection_recommendations', [])
        if recommendations:
            print(f"\n   选择建议:")
            for rec in recommendations[:3]:
                print(f"      • {rec}")
        
        # 决策过程
        process = guidance.get('decision_process', [])
        if process:
            print(f"\n   决策过程:")
            for step in process[:4]:
                print(f"      {step}")
    
    def _display_implementation_suggestions(self, suggestions: Dict[str, Any]):
        """显示实施建议"""
        if not suggestions:
            return
        
        print(f"\n🚀 实施建议")
        
        for category, tips in suggestions.items():
            if not tips:
                continue
            
            category_names = {
                'project_management': '项目管理',
                'collaboration_tips': '协作建议',
                'quality_assurance': '质量保证',
                'timeline_management': '时间管理'
            }
            
            print(f"\n   {category_names.get(category, category)}:")
            for tip in tips[:2]:  # 每类显示前2个建议
                print(f"      • {tip}")
    
    def save_results_to_file(self, result: Dict[str, Any], filename: str = "research_directions.json"):
        """保存结果到文件"""
        try:
            # 转换不可序列化的对象
            serializable_result = self._make_serializable(result)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(serializable_result, f, ensure_ascii=False, indent=2)
            
            print(f"\n💾 结果已保存到 {filename}")
            
        except Exception as e:
            print(f"保存文件时出错: {str(e)}")
    
    def _make_serializable(self, obj):
        """将对象转换为可序列化格式"""
        if hasattr(obj, '__dict__'):
            # 处理自定义对象
            result = {}
            for key, value in obj.__dict__.items():
                if not key.startswith('_'):  # 跳过私有属性
                    result[key] = self._make_serializable(value)
            return result
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, 'value'):  # 处理枚举类型
            return obj.value
        else:
            return obj


def main():
    """主函数"""
    config = {
        'llm_model': 'test_model',
        'max_analysis_time': 300
    }
    
    ui = DirectionRecommenderUI(config)
    result = ui.interactive_recommendation()
    
    if result:
        # 询问是否保存结果
        save_choice = input("\n是否保存结果到文件? (y/n): ").strip().lower()
        if save_choice == 'y':
            filename = input("文件名 (默认: research_directions.json): ").strip()
            if not filename:
                filename = "research_directions.json"
            ui.save_results_to_file(result, filename)


if __name__ == "__main__":
    main()