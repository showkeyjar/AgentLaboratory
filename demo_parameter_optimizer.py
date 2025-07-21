"""
参数优化引擎演示脚本

展示如何使用参数优化引擎进行智能参数调优
"""

import sys
import os
import math
import time
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from research_automation.core.parameter_optimizer import (
    ParameterOptimizer, ParameterSpace, OptimizationConfig, OptimizationMethod,
    ParameterType, ObjectiveFunction
)
from research_automation.models.analysis_models import ExperimentDesign


class ImageClassificationObjective(ObjectiveFunction):
    """图像分类任务的目标函数"""
    
    def __init__(self):
        self.evaluation_count = 0
        self.best_score_so_far = 0.0
    
    def evaluate(self, parameters: dict) -> float:
        """
        模拟图像分类模型的性能评估
        基于参数组合计算模拟的准确率
        """
        self.evaluation_count += 1
        
        # 获取参数
        learning_rate = parameters.get('learning_rate', 0.01)
        batch_size = parameters.get('batch_size', 32)
        epochs = parameters.get('epochs', 100)
        optimizer_type = parameters.get('optimizer_type', 'adam')
        use_dropout = parameters.get('use_dropout', True)
        dropout_rate = parameters.get('dropout_rate', 0.5)
        
        # 模拟性能计算（基于经验规则）
        base_accuracy = 0.85
        
        # 学习率影响
        if 0.001 <= learning_rate <= 0.01:
            lr_bonus = 0.05  # 最佳范围
        elif 0.0001 <= learning_rate < 0.001:
            lr_bonus = 0.02  # 较慢但稳定
        elif 0.01 < learning_rate <= 0.1:
            lr_bonus = -0.02  # 可能不稳定
        else:
            lr_bonus = -0.05  # 太高或太低
        
        # 批次大小影响
        if batch_size == 32:
            batch_bonus = 0.03
        elif batch_size in [16, 64]:
            batch_bonus = 0.01
        else:
            batch_bonus = -0.01
        
        # 训练轮数影响
        if 100 <= epochs <= 200:
            epoch_bonus = 0.02
        elif epochs < 50:
            epoch_bonus = -0.03  # 训练不足
        elif epochs > 500:
            epoch_bonus = -0.01  # 可能过拟合
        else:
            epoch_bonus = 0.0
        
        # 优化器影响
        optimizer_bonus = {
            'adam': 0.02,
            'sgd': 0.0,
            'rmsprop': 0.01
        }.get(optimizer_type, 0.0)
        
        # Dropout影响
        dropout_bonus = 0.01 if use_dropout and 0.3 <= dropout_rate <= 0.7 else 0.0
        
        # 计算最终准确率
        accuracy = base_accuracy + lr_bonus + batch_bonus + epoch_bonus + optimizer_bonus + dropout_bonus
        
        # 添加一些随机噪声
        import random
        noise = random.gauss(0, 0.005)
        accuracy += noise
        
        # 确保在合理范围内
        accuracy = max(0.0, min(1.0, accuracy))
        
        # 记录最佳性能
        if accuracy > self.best_score_so_far:
            self.best_score_so_far = accuracy
            print(f"  🎯 发现更好的参数组合! 准确率: {accuracy:.4f}")
        
        return accuracy
    
    def get_optimization_direction(self) -> str:
        return "maximize"


class NLPTaskObjective(ObjectiveFunction):
    """自然语言处理任务的目标函数"""
    
    def __init__(self):
        self.evaluation_count = 0
    
    def evaluate(self, parameters: dict) -> float:
        """模拟NLP任务的F1分数评估"""
        self.evaluation_count += 1
        
        # 获取参数
        learning_rate = parameters.get('learning_rate', 0.001)
        sequence_length = parameters.get('sequence_length', 128)
        hidden_size = parameters.get('hidden_size', 256)
        num_layers = parameters.get('num_layers', 2)
        attention_heads = parameters.get('attention_heads', 8)
        
        # 模拟F1分数计算
        base_f1 = 0.75
        
        # 学习率影响（NLP任务通常需要较小的学习率）
        if 0.0001 <= learning_rate <= 0.001:
            lr_bonus = 0.08
        elif 0.001 < learning_rate <= 0.01:
            lr_bonus = 0.02
        else:
            lr_bonus = -0.05
        
        # 序列长度影响
        if 64 <= sequence_length <= 256:
            seq_bonus = 0.03
        else:
            seq_bonus = -0.02
        
        # 隐藏层大小影响
        if 128 <= hidden_size <= 512:
            hidden_bonus = 0.02
        else:
            hidden_bonus = -0.01
        
        # 层数影响
        if 2 <= num_layers <= 6:
            layer_bonus = 0.01
        else:
            layer_bonus = -0.02
        
        # 注意力头数影响
        if attention_heads in [4, 8, 12]:
            attention_bonus = 0.02
        else:
            attention_bonus = 0.0
        
        f1_score = base_f1 + lr_bonus + seq_bonus + hidden_bonus + layer_bonus + attention_bonus
        
        # 添加噪声
        import random
        noise = random.gauss(0, 0.01)
        f1_score += noise
        
        return max(0.0, min(1.0, f1_score))
    
    def get_optimization_direction(self) -> str:
        return "maximize"


def create_image_classification_spaces():
    """创建图像分类任务的参数空间"""
    return [
        ParameterSpace(
            name="learning_rate",
            param_type=ParameterType.CONTINUOUS,
            min_value=0.0001,
            max_value=0.1,
            default_value=0.01,
            importance=0.9
        ),
        ParameterSpace(
            name="batch_size",
            param_type=ParameterType.DISCRETE,
            discrete_values=[8, 16, 32, 64, 128],
            default_value=32,
            importance=0.7
        ),
        ParameterSpace(
            name="epochs",
            param_type=ParameterType.DISCRETE,
            discrete_values=[50, 100, 150, 200, 300, 500],
            default_value=100,
            importance=0.6
        ),
        ParameterSpace(
            name="optimizer_type",
            param_type=ParameterType.CATEGORICAL,
            categorical_values=["adam", "sgd", "rmsprop"],
            default_value="adam",
            importance=0.5
        ),
        ParameterSpace(
            name="use_dropout",
            param_type=ParameterType.BOOLEAN,
            default_value=True,
            importance=0.4
        ),
        ParameterSpace(
            name="dropout_rate",
            param_type=ParameterType.CONTINUOUS,
            min_value=0.1,
            max_value=0.8,
            default_value=0.5,
            importance=0.3
        )
    ]


def create_nlp_task_spaces():
    """创建NLP任务的参数空间"""
    return [
        ParameterSpace(
            name="learning_rate",
            param_type=ParameterType.CONTINUOUS,
            min_value=0.00001,
            max_value=0.01,
            default_value=0.0001,
            importance=0.9
        ),
        ParameterSpace(
            name="sequence_length",
            param_type=ParameterType.DISCRETE,
            discrete_values=[64, 128, 256, 512],
            default_value=128,
            importance=0.8
        ),
        ParameterSpace(
            name="hidden_size",
            param_type=ParameterType.DISCRETE,
            discrete_values=[128, 256, 512, 768, 1024],
            default_value=256,
            importance=0.7
        ),
        ParameterSpace(
            name="num_layers",
            param_type=ParameterType.DISCRETE,
            discrete_values=[1, 2, 3, 4, 6, 8],
            default_value=2,
            importance=0.6
        ),
        ParameterSpace(
            name="attention_heads",
            param_type=ParameterType.DISCRETE,
            discrete_values=[1, 2, 4, 8, 12, 16],
            default_value=8,
            importance=0.5
        )
    ]


def demonstrate_optimization_methods():
    """演示不同的优化方法"""
    print("=" * 80)
    print("参数优化方法比较演示")
    print("=" * 80)
    
    optimizer = ParameterOptimizer()
    spaces = create_image_classification_spaces()
    objective = ImageClassificationObjective()
    
    methods = [
        (OptimizationMethod.RANDOM_SEARCH, "随机搜索"),
        (OptimizationMethod.GRID_SEARCH, "网格搜索"),
        (OptimizationMethod.GENETIC_ALGORITHM, "遗传算法")
    ]
    
    results = {}
    
    for method, method_name in methods:
        print(f"\\n{'-' * 60}")
        print(f"正在运行: {method_name}")
        print(f"{'-' * 60}")
        
        config = OptimizationConfig(
            method=method,
            max_evaluations=50,
            max_time_minutes=2,
            random_seed=42
        )
        
        start_time = time.time()
        
        try:
            result = optimizer.optimize_parameters(spaces, objective, config)
            end_time = time.time()
            
            results[method_name] = {
                'result': result,
                'time': end_time - start_time
            }
            
            print(f"✅ {method_name} 完成!")
            print(f"   最佳准确率: {result.best_score:.4f}")
            print(f"   最佳参数: {result.best_parameters}")
            print(f"   评估次数: {result.total_evaluations}")
            print(f"   用时: {end_time - start_time:.2f}秒")
            
        except Exception as e:
            print(f"❌ {method_name} 执行失败: {e}")
            results[method_name] = None
    
    # 比较结果
    print(f"\\n{'=' * 60}")
    print("优化方法比较结果")
    print(f"{'=' * 60}")
    
    valid_results = [(name, data) for name, data in results.items() if data is not None]
    valid_results.sort(key=lambda x: x[1]['result'].best_score, reverse=True)
    
    for i, (method_name, data) in enumerate(valid_results, 1):
        result = data['result']
        time_taken = data['time']
        print(f"{i}. {method_name}")
        print(f"   准确率: {result.best_score:.4f}")
        print(f"   效率: {result.total_evaluations/time_taken:.1f} 评估/秒")
        print()


def demonstrate_task_specific_optimization():
    """演示特定任务的参数优化"""
    print("\\n" + "=" * 80)
    print("特定任务参数优化演示")
    print("=" * 80)
    
    optimizer = ParameterOptimizer()
    
    # 1. 图像分类任务优化
    print("\\n📸 图像分类任务参数优化")
    print("-" * 50)
    
    img_spaces = create_image_classification_spaces()
    img_objective = ImageClassificationObjective()
    
    img_config = OptimizationConfig(
        method=OptimizationMethod.GENETIC_ALGORITHM,
        max_evaluations=80,
        random_seed=123
    )
    
    img_result = optimizer.optimize_parameters(img_spaces, img_objective, img_config)
    
    print(f"图像分类最佳结果:")
    print(f"  准确率: {img_result.best_score:.4f}")
    print(f"  最佳参数:")
    for param, value in img_result.best_parameters.items():
        print(f"    {param}: {value}")
    
    # 2. NLP任务优化
    print("\\n📝 自然语言处理任务参数优化")
    print("-" * 50)
    
    nlp_spaces = create_nlp_task_spaces()
    nlp_objective = NLPTaskObjective()
    
    nlp_config = OptimizationConfig(
        method=OptimizationMethod.RANDOM_SEARCH,
        max_evaluations=60,
        random_seed=456
    )
    
    nlp_result = optimizer.optimize_parameters(nlp_spaces, nlp_objective, nlp_config)
    
    print(f"NLP任务最佳结果:")
    print(f"  F1分数: {nlp_result.best_score:.4f}")
    print(f"  最佳参数:")
    for param, value in nlp_result.best_parameters.items():
        print(f"    {param}: {value}")


def demonstrate_automatic_space_creation():
    """演示自动参数空间创建"""
    print("\\n" + "=" * 80)
    print("自动参数空间创建演示")
    print("=" * 80)
    
    optimizer = ParameterOptimizer()
    
    # 从实验设计创建参数空间
    design = ExperimentDesign(
        methodology="深度学习图像分类",
        parameters={
            'learning_rate': 0.01,
            'batch_size': 32,
            'epochs': 100,
            'use_dropout': True,
            'model_depth': 5,
            'activation': 'relu'
        }
    )
    
    print("\\n🔧 从实验设计自动创建参数空间")
    print("-" * 50)
    
    auto_spaces = optimizer.create_parameter_spaces_from_design(design)
    
    print(f"自动创建的参数空间 ({len(auto_spaces)}个):")
    for space in auto_spaces:
        print(f"  📊 {space.name}")
        print(f"     类型: {space.param_type.value}")
        if space.param_type == ParameterType.CONTINUOUS:
            print(f"     范围: [{space.min_value}, {space.max_value}]")
        elif space.param_type == ParameterType.DISCRETE:
            print(f"     选项: {space.discrete_values}")
        print(f"     重要性: {space.importance}")
        print()
    
    # 基于方法论建议参数空间
    print("\\n💡 基于方法论的参数空间建议")
    print("-" * 50)
    
    methodologies = [
        "深度学习CNN",
        "随机森林分类",
        "支持向量机",
        "循环神经网络"
    ]
    
    for methodology in methodologies:
        suggested_spaces = optimizer.suggest_parameter_spaces(methodology)
        print(f"\\n{methodology}:")
        if suggested_spaces:
            for space in suggested_spaces:
                print(f"  • {space.name} (重要性: {space.importance})")
        else:
            print("  • 暂无特定建议")


def demonstrate_optimization_analysis():
    """演示优化结果分析"""
    print("\\n" + "=" * 80)
    print("优化结果分析演示")
    print("=" * 80)
    
    optimizer = ParameterOptimizer()
    spaces = create_image_classification_spaces()
    objective = ImageClassificationObjective()
    
    config = OptimizationConfig(
        method=OptimizationMethod.GENETIC_ALGORITHM,
        max_evaluations=100,
        random_seed=789
    )
    
    print("\\n🔍 执行参数优化...")
    result = optimizer.optimize_parameters(spaces, objective, config)
    
    print("\\n📊 分析优化结果...")
    analysis = optimizer.analyze_optimization_results(result)
    
    print(f"\\n优化结果分析报告:")
    print(f"{'=' * 40}")
    
    # 收敛分析
    convergence = analysis['convergence_analysis']
    print(f"\\n🎯 收敛分析:")
    print(f"  是否收敛: {'是' if convergence['converged'] else '否'}")
    if convergence['converged']:
        print(f"  收敛轮次: {convergence['convergence_iteration']}")
        print(f"  收敛率: {convergence['convergence_rate']:.2%}")
    
    # 效率分析
    efficiency = analysis['optimization_efficiency']
    print(f"\\n⚡ 优化效率:")
    print(f"  效率分数: {efficiency['efficiency_score']:.3f}")
    print(f"  找到最佳解用时: {efficiency['evaluations_to_best']} 次评估")
    print(f"  总评估次数: {efficiency['total_evaluations']}")
    
    # 参数敏感性
    sensitivity = analysis['parameter_sensitivity']
    print(f"\\n🎛️ 参数重要性排序:")
    sorted_params = sorted(sensitivity.items(), key=lambda x: x[1], reverse=True)
    for param, importance in sorted_params:
        print(f"  {param}: {importance:.3f}")
    
    # 优化建议
    recommendations = analysis['recommendations']
    if recommendations:
        print(f"\\n💡 优化建议:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")


def main():
    """主函数"""
    print("🚀 参数优化引擎演示系统")
    print("=" * 80)
    print("本演示将展示智能参数优化的各种功能:")
    print("• 多种优化算法比较")
    print("• 特定任务的参数优化")
    print("• 自动参数空间创建")
    print("• 优化结果深度分析")
    print("=" * 80)
    
    try:
        # 1. 优化方法比较
        demonstrate_optimization_methods()
        
        # 2. 特定任务优化
        demonstrate_task_specific_optimization()
        
        # 3. 自动空间创建
        demonstrate_automatic_space_creation()
        
        # 4. 结果分析
        demonstrate_optimization_analysis()
        
        print("\\n" + "=" * 80)
        print("🎉 演示完成!")
        print("=" * 80)
        
        print("\\n📋 总结:")
        print("• 参数优化引擎支持多种优化算法")
        print("• 能够自动适应不同的机器学习任务")
        print("• 提供智能的参数空间建议")
        print("• 包含详细的优化过程分析")
        print("• 帮助研究者找到最佳的实验参数配置")
        
    except Exception as e:
        print(f"\\n❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()