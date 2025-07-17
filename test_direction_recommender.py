"""
测试研究方向建议系统

验证研究方向建议功能的完整性和准确性
"""

import sys
import json
from typing import Dict, Any

# 添加项目路径
sys.path.append('.')

from research_automation.core.research_planner import ResearchPlannerComponent


def test_direction_recommendation_system():
    """测试研究方向建议系统"""
    print("=== 测试研究方向建议系统 ===\n")
    
    # 初始化组件
    config = {
        'llm_model': 'test_model',
        'max_analysis_time': 300
    }
    
    planner = ResearchPlannerComponent(config=config)
    
    # 测试案例
    test_cases = [
        {
            "name": "AI医疗应用研究",
            "topic": "Artificial intelligence for healthcare applications",
            "context": {
                'user_experience': 'intermediate',
                'available_resources': 'moderate',
                'time_constraint': '6_months',
                'research_goal': 'Develop AI solutions for medical diagnosis'
            }
        },
        {
            "name": "量子计算算法优化",
            "topic": "Quantum computing algorithms optimization",
            "context": {
                'user_experience': 'advanced',
                'available_resources': 'abundant',
                'time_constraint': '12_months'
            }
        },
        {
            "name": "深度学习图像识别",
            "topic": "深度学习在图像识别中的应用",
            "context": {
                'user_experience': 'beginner',
                'available_resources': 'limited',
                'time_constraint': '3_months'
            }
        }
    ]
    
    # 执行测试
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"测试案例 {i}: {test_case['name']}")
        print(f"{'='*60}")
        
        try:
            # 生成研究方向建议
            result = planner.generate_research_directions(
                test_case['topic'], 
                test_case['context']
            )
            
            # 显示结果摘要
            display_test_results(result, test_case['name'])
            
            # 保存详细结果
            save_test_results(result, f"test_result_{i}_{test_case['name']}.json")
            
        except Exception as e:
            print(f"❌ 测试失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("测试完成")
    print(f"{'='*60}")


def display_test_results(result: Dict[str, Any], test_name: str):
    """显示测试结果摘要"""
    if not result:
        print("❌ 未生成有效结果")
        return
    
    print(f"✅ {test_name} - 测试成功")
    
    # 主题分析摘要
    analysis = result.get('topic_analysis')
    if analysis:
        print(f"\n📊 主题分析:")
        print(f"   复杂度: {analysis.complexity_level.value} ({analysis.complexity_score:.2f})")
        print(f"   类型: {analysis.research_type.value}")
        print(f"   时长: {analysis.estimated_duration}天")
        print(f"   成功率: {analysis.success_probability:.1%}")
    
    # 研究方向摘要
    directions = result.get('research_directions', {})
    total_directions = sum(len(dirs) for dirs in directions.values())
    print(f"\n🎯 研究方向: {len(directions)}个类别, {total_directions}个方向")
    
    for category, direction_list in directions.items():
        if direction_list:
            print(f"   {category}: {len(direction_list)}个")
    
    # 个性化推荐摘要
    recommendations = result.get('personalized_recommendations', {})
    top_recs = recommendations.get('top_recommendations', [])
    if top_recs:
        print(f"\n⭐ 顶级推荐:")
        for i, rec in enumerate(top_recs[:3], 1):
            score = rec.get('personalized_score', 0)
            print(f"   {i}. {rec['title']} (评分: {score:.2f})")
    
    # 经验和资源匹配
    exp_match = recommendations.get('experience_match', {})
    res_align = recommendations.get('resource_alignment', {})
    
    if exp_match:
        print(f"\n📈 经验匹配: {exp_match.get('match_score', 0):.1%}")
    if res_align:
        print(f"💰 资源对齐: {res_align.get('alignment_score', 0):.1%}")
    
    # 选择指导摘要
    guidance = result.get('selection_guidance', {})
    criteria = guidance.get('selection_criteria', [])
    if criteria:
        print(f"\n🧭 选择标准: {len(criteria)}个")
    
    print()


def save_test_results(result: Dict[str, Any], filename: str):
    """保存测试结果到文件"""
    try:
        # 转换为可序列化格式
        serializable_result = make_serializable(result)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, ensure_ascii=False, indent=2)
        
        print(f"💾 详细结果已保存到 {filename}")
        
    except Exception as e:
        print(f"⚠️  保存文件时出错: {str(e)}")


def make_serializable(obj):
    """将对象转换为可序列化格式"""
    if hasattr(obj, '__dict__'):
        # 处理自定义对象
        result = {}
        for key, value in obj.__dict__.items():
            if not key.startswith('_'):  # 跳过私有属性
                result[key] = make_serializable(value)
        return result
    elif isinstance(obj, dict):
        return {key: make_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    elif hasattr(obj, 'value'):  # 处理枚举类型
        return obj.value
    else:
        return obj


def test_individual_components():
    """测试各个组件功能"""
    print("\n=== 测试各个组件功能 ===\n")
    
    config = {
        'llm_model': 'test_model',
        'max_analysis_time': 300
    }
    
    planner = ResearchPlannerComponent(config=config)
    
    # 测试主题分析
    print("1. 测试主题分析功能...")
    try:
        topic = "Machine learning for natural language processing"
        analysis = planner.analyze_topic(topic)
        
        print(f"✅ 主题分析成功")
        print(f"   主题: {analysis.topic}")
        print(f"   复杂度: {analysis.complexity_level.value}")
        print(f"   关键词数量: {len(analysis.keywords)}")
        print(f"   相关领域: {len(analysis.related_fields)}")
        
    except Exception as e:
        print(f"❌ 主题分析失败: {str(e)}")
    
    # 测试研究计划生成
    print("\n2. 测试研究计划生成...")
    try:
        if 'analysis' in locals():
            plan = planner.generate_research_plan(analysis)
            
            print(f"✅ 研究计划生成成功")
            print(f"   时间线: {len(plan.timeline)}个里程碑")
            print(f"   研究路径: {len(plan.research_paths)}条")
            print(f"   成功指标: {len(plan.success_metrics)}个")
        
    except Exception as e:
        print(f"❌ 研究计划生成失败: {str(e)}")
    
    print("\n组件测试完成")


if __name__ == "__main__":
    # 运行主要测试
    test_direction_recommendation_system()
    
    # 运行组件测试
    test_individual_components()
    
    print("\n🎉 所有测试完成!")