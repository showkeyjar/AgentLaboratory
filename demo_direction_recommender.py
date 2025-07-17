"""
研究方向建议系统演示

展示完整的研究方向建议功能
"""

import sys
sys.path.append('.')

from research_automation.core.research_planner import ResearchPlannerComponent


def demo_research_direction_recommendation():
    """演示研究方向建议功能"""
    print("🎯 研究方向建议系统演示")
    print("=" * 50)
    
    # 初始化组件
    config = {
        'llm_model': 'test_model',
        'max_analysis_time': 300
    }
    
    planner = ResearchPlannerComponent(config=config)
    
    # 演示案例
    topic = "Artificial intelligence for healthcare applications"
    context = {
        'user_experience': 'intermediate',
        'available_resources': 'moderate',
        'time_constraint': '6_months',
        'research_goal': 'Develop AI solutions for medical diagnosis'
    }
    
    print(f"\n📋 研究主题: {topic}")
    print(f"👤 用户经验: {context['user_experience']}")
    print(f"💰 可用资源: {context['available_resources']}")
    print(f"⏰ 时间约束: {context['time_constraint']}")
    
    print(f"\n🔄 正在生成研究方向建议...")
    
    try:
        # 生成研究方向建议
        result = planner.generate_research_directions(topic, context)
        
        # 显示结果
        display_demo_results(result)
        
    except Exception as e:
        print(f"❌ 生成建议时出错: {str(e)}")


def display_demo_results(result):
    """显示演示结果"""
    print(f"\n✅ 建议生成完成!")
    print("=" * 50)
    
    # 主题分析
    analysis = result.get('topic_analysis')
    if analysis:
        print(f"\n📊 主题分析结果:")
        print(f"   🎯 研究类型: {analysis.research_type.value}")
        print(f"   📈 复杂度等级: {analysis.complexity_level.value}")
        print(f"   🔢 复杂度分数: {analysis.complexity_score:.2f}")
        print(f"   ⏱️  预估时长: {analysis.estimated_duration}天")
        print(f"   🎲 成功概率: {analysis.success_probability:.1%}")
        print(f"   🏷️  关键词: {', '.join(analysis.keywords[:5])}")
        print(f"   🔬 相关领域: {', '.join(analysis.related_fields)}")
    
    # 研究方向分类
    directions = result.get('research_directions', {})
    if directions:
        print(f"\n🎯 研究方向分类:")
        total_directions = sum(len(dirs) for dirs in directions.values())
        print(f"   📊 总计: {len(directions)}个类别, {total_directions}个方向")
        
        for category, direction_list in directions.items():
            if direction_list:
                print(f"\n   📂 {category} ({len(direction_list)}个):")
                for i, direction in enumerate(direction_list[:2], 1):  # 显示前2个
                    print(f"      {i}. {direction['title']}")
                    print(f"         可行性: {direction.get('feasibility', 0):.1%}")
                    print(f"         创新性: {direction.get('innovation_potential', 0):.1%}")
                    print(f"         适用性: {direction.get('suitability', 'N/A')}")
    
    # 个性化推荐
    recommendations = result.get('personalized_recommendations', {})
    if recommendations:
        print(f"\n⭐ 个性化推荐:")
        
        top_recs = recommendations.get('top_recommendations', [])
        if top_recs:
            print(f"   🏆 顶级推荐 (前3个):")
            for i, rec in enumerate(top_recs[:3], 1):
                print(f"      {i}. {rec['title']}")
                print(f"         评分: {rec.get('personalized_score', 0):.2f}")
                print(f"         理由: {rec.get('recommendation_reason', 'N/A')}")
                print(f"         类别: {rec.get('category', 'N/A')}")
                print()
        
        # 学习路径
        learning_path = recommendations.get('learning_path', {})
        if learning_path:
            prereq = learning_path.get('prerequisite_knowledge', [])
            skills = learning_path.get('skill_development', [])
            
            print(f"   📚 学习路径:")
            if prereq:
                print(f"      前置知识: {', '.join(prereq[:3])}")
            if skills:
                print(f"      技能发展: {', '.join(skills[:3])}")
        
        # 下一步建议
        next_steps = recommendations.get('next_steps', [])
        if next_steps:
            print(f"\n   🚀 下一步建议:")
            for i, step in enumerate(next_steps[:3], 1):
                print(f"      {i}. {step['step']} ({step.get('timeline', 'N/A')})")
    
    # 选择指导
    guidance = result.get('selection_guidance', {})
    if guidance:
        print(f"\n🧭 选择指导:")
        
        criteria = guidance.get('selection_criteria', [])
        if criteria:
            print(f"   📋 选择标准:")
            for criterion in criteria[:3]:
                print(f"      • {criterion['name']} (权重: {criterion['weight']:.1%})")
        
        recommendations = guidance.get('selection_recommendations', [])
        if recommendations:
            print(f"   💡 选择建议:")
            for rec in recommendations[:2]:
                print(f"      • {rec}")
    
    print(f"\n🎉 演示完成!")
    print("=" * 50)


if __name__ == "__main__":
    demo_research_direction_recommendation()