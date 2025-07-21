"""
实验进度监控演示脚本

展示如何使用实验进度监控组件进行实验状态跟踪、预警和优化建议
"""

import sys
import os
import time
import json
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from research_automation.core.experiment_monitor import (
    ExperimentMonitor, ExperimentStatus, AlertLevel
)
from research_automation.models.analysis_models import ExperimentDesign


def create_sample_experiments():
    """创建示例实验设计"""
    experiments = []
    
    # 实验1: 图像分类实验
    exp1 = ExperimentDesign(
        methodology="深度学习图像分类优化实验",
        parameters={
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'optimizer': 'Adam',
            'architecture': 'ResNet50'
        },
        timeline_weeks=6,
        expected_outcomes=["提升图像分类准确率到95%", "减少训练时间20%"],
        resource_requirements={'base_cost': 8000.0, 'gpu_hours': 120},
        feasibility_score=0.85
    )
    experiments.append(("图像分类实验", exp1))
    
    # 实验2: 自然语言处理实验
    exp2 = ExperimentDesign(
        methodology="BERT模型微调实验",
        parameters={
            'learning_rate': 2e-5,
            'batch_size': 16,
            'max_length': 512,
            'epochs': 5,
            'warmup_steps': 500
        },
        timeline_weeks=4,
        expected_outcomes=["文本分类F1分数达到0.9", "推理速度提升30%"],
        resource_requirements={'base_cost': 6000.0, 'gpu_hours': 80},
        feasibility_score=0.9
    )
    experiments.append(("NLP实验", exp2))
    
    # 实验3: 强化学习实验
    exp3 = ExperimentDesign(
        methodology="深度Q网络游戏AI实验",
        parameters={
            'learning_rate': 0.0001,
            'epsilon': 0.1,
            'gamma': 0.99,
            'replay_buffer_size': 10000,
            'target_update_freq': 1000
        },
        timeline_weeks=8,
        expected_outcomes=["游戏得分超过人类平均水平", "训练稳定性提升"],
        resource_requirements={'base_cost': 12000.0, 'gpu_hours': 200},
        feasibility_score=0.75
    )
    experiments.append(("强化学习实验", exp3))
    
    return experiments


def simulate_experiment_progress(monitor, experiment_id, experiment_name):
    """模拟实验进度"""
    print(f"\n🚀 开始模拟 {experiment_name}")
    print("-" * 60)
    
    # 阶段1: 准备阶段
    print("📋 阶段1: 实验准备")
    monitor.update_experiment_status(experiment_id, ExperimentStatus.PREPARING, "数据预处理", 1)
    time.sleep(0.5)
    
    # 更新资源使用
    monitor.update_resource_usage(experiment_id, {
        'cpu': 0.3,
        'memory': 0.4,
        'gpu': 0.2
    })
    
    # 阶段2: 开始训练
    print("🏃 阶段2: 开始训练")
    monitor.update_experiment_status(experiment_id, ExperimentStatus.RUNNING, "模型训练初始化", 2)
    time.sleep(0.5)
    
    # 增加资源使用
    monitor.update_resource_usage(experiment_id, {
        'cpu': 0.7,
        'memory': 0.6,
        'gpu': 0.8
    })
    
    # 添加初始性能指标
    monitor.add_performance_metrics(experiment_id, {
        'accuracy': 0.45,
        'loss': 2.1,
        'epoch': 1
    })
    
    # 阶段3: 训练进行中
    print("⚡ 阶段3: 训练进行中")
    for step in range(3, 7):
        monitor.update_experiment_status(experiment_id, ExperimentStatus.RUNNING, f"训练轮次 {step-2}", step)
        
        # 模拟资源使用波动
        cpu_usage = 0.6 + (step * 0.05)
        gpu_usage = 0.75 + (step * 0.03)
        memory_usage = 0.5 + (step * 0.04)
        
        monitor.update_resource_usage(experiment_id, {
            'cpu': min(cpu_usage, 0.95),
            'memory': min(memory_usage, 0.85),
            'gpu': min(gpu_usage, 0.98)  # 可能触发高使用率预警
        })
        
        # 模拟性能改善
        accuracy = 0.45 + (step - 2) * 0.1
        loss = 2.1 - (step - 2) * 0.3
        
        monitor.add_performance_metrics(experiment_id, {
            'accuracy': min(accuracy, 0.95),
            'loss': max(loss, 0.1),
            'epoch': step - 1
        })
        
        time.sleep(0.3)
    
    # 阶段4: 验证和测试
    print("🔍 阶段4: 模型验证")
    monitor.update_experiment_status(experiment_id, ExperimentStatus.RUNNING, "模型验证", 7)
    
    # 降低资源使用
    monitor.update_resource_usage(experiment_id, {
        'cpu': 0.4,
        'memory': 0.3,
        'gpu': 0.5
    })
    
    # 最终性能指标
    final_accuracy = 0.88 if "图像分类" in experiment_name else 0.92
    monitor.add_performance_metrics(experiment_id, {
        'accuracy': final_accuracy,
        'loss': 0.15,
        'val_accuracy': final_accuracy - 0.02,
        'test_accuracy': final_accuracy - 0.01
    })
    
    time.sleep(0.5)
    
    # 阶段5: 完成
    print("✅ 阶段5: 实验完成")
    monitor.update_experiment_status(experiment_id, ExperimentStatus.COMPLETED, "实验完成", 8)
    
    print(f"🎉 {experiment_name} 模拟完成!")


def demonstrate_monitoring_features(monitor, experiment_id, experiment_name):
    """演示监控功能"""
    print(f"\n📊 {experiment_name} 监控结果分析")
    print("=" * 60)
    
    # 获取实验进度
    progress = monitor.get_experiment_progress(experiment_id)
    if progress:
        print(f"📈 实验进度:")
        print(f"  • 状态: {progress.status.value}")
        print(f"  • 进度: {progress.progress_percentage:.1f}%")
        print(f"  • 当前步骤: {progress.current_step}")
        
        elapsed_time = progress.get_elapsed_time()
        if elapsed_time:
            print(f"  • 执行时间: {elapsed_time.total_seconds():.1f}秒")
        
        print(f"  • 资源使用:")
        for resource, usage in progress.resource_usage.items():
            print(f"    - {resource}: {usage:.1%}")
        
        print(f"  • 性能指标:")
        for metric, value in progress.performance_metrics.items():
            if isinstance(value, float):
                print(f"    - {metric}: {value:.3f}")
            else:
                print(f"    - {metric}: {value}")
    
    # 获取预警信息
    alerts = monitor.get_experiment_alerts(experiment_id)
    if alerts:
        print(f"\n⚠️  预警信息 ({len(alerts)}个):")
        for alert in alerts:
            status = "✅已解决" if alert.resolved else "🔴活跃"
            print(f"  • [{alert.level.value.upper()}] {alert.message} ({status})")
            if alert.details:
                for key, value in alert.details.items():
                    if isinstance(value, float):
                        print(f"    - {key}: {value:.3f}")
                    else:
                        print(f"    - {key}: {value}")
    else:
        print("\n✅ 无预警信息")
    
    # 获取优化建议
    suggestions = monitor.get_optimization_suggestions(experiment_id)
    if suggestions:
        print(f"\n💡 优化建议 ({len(suggestions)}个):")
        for suggestion in suggestions:
            status = "✅已应用" if suggestion.applied else "⏳待处理"
            priority_stars = "⭐" * suggestion.priority
            print(f"  • [{suggestion.category.upper()}] {suggestion.title} ({status})")
            print(f"    优先级: {priority_stars} ({suggestion.priority}/5)")
            print(f"    描述: {suggestion.description}")
            print(f"    预期影响: {suggestion.expected_impact}")
            print(f"    实施难度: {suggestion.implementation_effort}")
            print()
    else:
        print("\n✅ 无优化建议")


def generate_comprehensive_report(monitor, experiment_ids, experiment_names):
    """生成综合监控报告"""
    print("\n" + "=" * 80)
    print("📋 综合监控报告")
    print("=" * 80)
    
    # 实验概览
    print("\n📊 实验概览:")
    all_experiments = monitor.get_all_experiments()
    
    status_counts = {}
    total_alerts = 0
    total_suggestions = 0
    
    for exp_id, progress in all_experiments.items():
        status = progress.status.value
        status_counts[status] = status_counts.get(status, 0) + 1
        
        alerts = monitor.get_experiment_alerts(exp_id)
        suggestions = monitor.get_optimization_suggestions(exp_id)
        total_alerts += len(alerts)
        total_suggestions += len(suggestions)
    
    print(f"  • 总实验数: {len(all_experiments)}")
    print(f"  • 状态分布:")
    for status, count in status_counts.items():
        print(f"    - {status}: {count}")
    print(f"  • 总预警数: {total_alerts}")
    print(f"  • 总建议数: {total_suggestions}")
    
    # 详细实验报告
    print(f"\n📈 详细实验报告:")
    for i, (exp_id, exp_name) in enumerate(zip(experiment_ids, experiment_names)):
        print(f"\n{i+1}. {exp_name}")
        print("-" * 40)
        
        # 生成监控报告
        report = monitor.generate_monitoring_report(exp_id)
        
        print(f"  状态: {report.progress.status.value}")
        print(f"  进度: {report.progress.progress_percentage:.1f}%")
        print(f"  执行时间: {report.statistics.get('experiment_duration', 0):.1f}秒")
        print(f"  活跃预警: {report.statistics.get('active_alerts', 0)}")
        print(f"  待处理建议: {report.statistics.get('pending_suggestions', 0)}")
        
        # 显示最终性能
        if report.progress.performance_metrics:
            final_accuracy = report.progress.performance_metrics.get('accuracy', 0)
            print(f"  最终准确率: {final_accuracy:.1%}")
    
    # 性能对比
    print(f"\n🏆 性能对比:")
    performance_data = []
    for exp_id, exp_name in zip(experiment_ids, experiment_names):
        progress = monitor.get_experiment_progress(exp_id)
        if progress and 'accuracy' in progress.performance_metrics:
            accuracy = progress.performance_metrics['accuracy']
            performance_data.append((exp_name, accuracy))
    
    # 按准确率排序
    performance_data.sort(key=lambda x: x[1], reverse=True)
    for i, (name, accuracy) in enumerate(performance_data):
        medal = ["🥇", "🥈", "🥉"][i] if i < 3 else "  "
        print(f"  {medal} {name}: {accuracy:.1%}")


def save_monitoring_results(monitor, experiment_ids, experiment_names):
    """保存监控结果"""
    print(f"\n💾 保存监控结果...")
    
    # 创建结果目录
    results_dir = "experiment_monitoring_results"
    os.makedirs(results_dir, exist_ok=True)
    
    saved_files = []
    
    # 保存每个实验的监控数据
    for exp_id, exp_name in zip(experiment_ids, experiment_names):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{exp_name.replace(' ', '_')}_monitoring.json"
        filepath = os.path.join(results_dir, filename)
        
        try:
            monitor.save_monitoring_data(exp_id, filepath)
            saved_files.append(filepath)
        except Exception as e:
            print(f"保存 {exp_name} 监控数据失败: {e}")
    
    # 生成汇总报告
    summary_file = os.path.join(results_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_summary.json")
    try:
        summary_data = {
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(experiment_ids),
            'experiments': []
        }
        
        for exp_id, exp_name in zip(experiment_ids, experiment_names):
            report = monitor.generate_monitoring_report(exp_id)
            summary_data['experiments'].append({
                'name': exp_name,
                'id': exp_id,
                'status': report.progress.status.value,
                'progress': report.progress.progress_percentage,
                'alerts': len(report.alerts),
                'suggestions': len(report.suggestions),
                'final_accuracy': report.progress.performance_metrics.get('accuracy', 0)
            })
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
        saved_files.append(summary_file)
        
    except Exception as e:
        print(f"保存汇总报告失败: {e}")
    
    print(f"✅ 监控结果已保存:")
    for filepath in saved_files:
        file_size = os.path.getsize(filepath)
        print(f"  • {filepath} ({file_size} 字节)")


def main():
    """主函数"""
    print("🔬 实验进度监控演示")
    print("=" * 80)
    print("本演示将展示实验进度监控的完整功能:")
    print("• 实验状态跟踪和进度更新")
    print("• 资源使用监控和预警")
    print("• 性能指标跟踪")
    print("• 自动优化建议生成")
    print("• 综合监控报告")
    print("=" * 80)
    
    try:
        # 创建监控器
        monitor = ExperimentMonitor()
        print("\n✅ 实验监控器初始化完成")
        
        # 创建示例实验
        experiments = create_sample_experiments()
        print(f"\n📋 准备监控 {len(experiments)} 个实验")
        
        # 开始监控所有实验
        experiment_ids = []
        experiment_names = []
        
        for exp_name, exp_design in experiments:
            exp_id = monitor.start_experiment_monitoring(exp_design)
            experiment_ids.append(exp_id)
            experiment_names.append(exp_name)
            print(f"  ✅ 开始监控: {exp_name} (ID: {exp_id})")
        
        print(f"\n🚀 开始模拟实验执行...")
        
        # 模拟实验进度（并行进行）
        for exp_id, exp_name in zip(experiment_ids, experiment_names):
            simulate_experiment_progress(monitor, exp_id, exp_name)
        
        print(f"\n🎉 所有实验模拟完成!")
        
        # 展示监控功能
        for exp_id, exp_name in zip(experiment_ids, experiment_names):
            demonstrate_monitoring_features(monitor, exp_id, exp_name)
        
        # 生成综合报告
        generate_comprehensive_report(monitor, experiment_ids, experiment_names)
        
        # 保存监控结果
        save_monitoring_results(monitor, experiment_ids, experiment_names)
        
        print("\n" + "=" * 80)
        print("🎊 实验进度监控演示完成!")
        print("=" * 80)
        
        print("\n📋 功能总结:")
        print("• ✅ 实验状态跟踪 - 完整记录实验从准备到完成的全过程")
        print("• ✅ 资源监控预警 - 自动检测高资源使用并生成预警")
        print("• ✅ 性能指标跟踪 - 实时监控模型性能变化")
        print("• ✅ 智能优化建议 - 基于监控数据自动生成改进建议")
        print("• ✅ 综合报告生成 - 提供详细的实验分析和对比")
        print("• ✅ 数据持久化 - 保存完整的监控历史记录")
        
    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()