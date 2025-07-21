"""
实验进度监控测试

测试实验状态跟踪、进度预警和优化建议功能
"""

import unittest
import sys
import os
import json
import tempfile
import shutil
from datetime import datetime, timedelta
import time

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_automation.core.experiment_monitor import (
    ExperimentMonitor, ExperimentStatus, AlertLevel,
    ExperimentProgress, ExperimentAlert, OptimizationSuggestion
)
from research_automation.models.analysis_models import ExperimentDesign


class TestExperimentMonitor(unittest.TestCase):
    """实验进度监控测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.monitor = ExperimentMonitor()
        
        # 创建测试用的实验设计
        self.experiment_design = ExperimentDesign(
            methodology="convolutional neural networks",
            parameters={
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100
            },
            timeline_weeks=8,
            expected_outcomes=["提升图像分类准确率", "优化模型性能"],
            resource_requirements={'base_cost': 5000.0, 'gpu_hours': 100},
            feasibility_score=0.8
        )
        
        # 创建临时目录用于测试文件保存
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """测试后清理"""
        # 删除临时目录
        shutil.rmtree(self.temp_dir)
    
    def test_monitor_initialization(self):
        """测试监控器初始化"""
        # 验证监控器已正确初始化
        self.assertIsInstance(self.monitor.experiments, dict)
        self.assertIsInstance(self.monitor.alerts, dict)
        self.assertIsInstance(self.monitor.suggestions, dict)
        self.assertIn('check_interval_minutes', self.monitor.monitoring_config)
        
        print("监控器初始化测试通过")
    
    def test_start_experiment_monitoring(self):
        """测试开始实验监控"""
        # 开始监控实验
        experiment_id = self.monitor.start_experiment_monitoring(self.experiment_design)
        
        # 验证实验已被添加到监控列表
        self.assertIn(experiment_id, self.monitor.experiments)
        self.assertIn(experiment_id, self.monitor.alerts)
        self.assertIn(experiment_id, self.monitor.suggestions)
        
        # 验证实验进度信息
        progress = self.monitor.experiments[experiment_id]
        self.assertEqual(progress.experiment_name, self.experiment_design.methodology)
        self.assertEqual(progress.status, ExperimentStatus.PLANNED)
        self.assertIsNotNone(progress.start_time)
        
        print(f"开始实验监控测试通过，实验ID: {experiment_id}")
    
    def test_update_experiment_status(self):
        """测试更新实验状态"""
        # 开始监控实验
        experiment_id = self.monitor.start_experiment_monitoring(self.experiment_design)
        
        # 更新实验状态
        self.monitor.update_experiment_status(
            experiment_id, 
            ExperimentStatus.RUNNING,
            "数据预处理",
            2
        )
        
        # 验证状态更新
        progress = self.monitor.get_experiment_progress(experiment_id)
        self.assertEqual(progress.status, ExperimentStatus.RUNNING)
        self.assertEqual(progress.current_step, "数据预处理")
        self.assertEqual(progress.completed_steps, 2)
        self.assertGreater(progress.progress_percentage, 0)
        
        print(f"实验状态更新测试通过，当前进度: {progress.progress_percentage:.1f}%")
    
    def test_resource_usage_monitoring(self):
        """测试资源使用监控"""
        # 开始监控实验
        experiment_id = self.monitor.start_experiment_monitoring(self.experiment_design)
        
        # 更新资源使用情况
        resource_usage = {
            'cpu': 0.75,
            'memory': 0.60,
            'gpu': 0.85
        }
        
        self.monitor.update_resource_usage(experiment_id, resource_usage)
        
        # 验证资源使用情况
        progress = self.monitor.get_experiment_progress(experiment_id)
        self.assertEqual(progress.resource_usage['cpu'], 0.75)
        self.assertEqual(progress.resource_usage['memory'], 0.60)
        self.assertEqual(progress.resource_usage['gpu'], 0.85)
        
        # 验证是否生成了高资源使用预警
        alerts = self.monitor.get_experiment_alerts(experiment_id)
        gpu_alerts = [alert for alert in alerts if 'gpu' in alert.alert_type]
        self.assertGreater(len(gpu_alerts), 0)
        
        print(f"资源使用监控测试通过，生成了 {len(alerts)} 个预警")
    
    def test_performance_metrics_monitoring(self):
        """测试性能指标监控"""
        # 开始监控实验
        experiment_id = self.monitor.start_experiment_monitoring(self.experiment_design)
        
        # 添加性能指标
        metrics = {
            'accuracy': 0.85,
            'loss': 0.25,
            'f1_score': 0.82
        }
        
        self.monitor.add_performance_metrics(experiment_id, metrics)
        
        # 验证性能指标
        progress = self.monitor.get_experiment_progress(experiment_id)
        self.assertEqual(progress.performance_metrics['accuracy'], 0.85)
        self.assertEqual(progress.performance_metrics['loss'], 0.25)
        self.assertEqual(progress.performance_metrics['f1_score'], 0.82)
        
        print("性能指标监控测试通过")
    
    def test_low_performance_alert(self):
        """测试低性能预警"""
        # 开始监控实验
        experiment_id = self.monitor.start_experiment_monitoring(self.experiment_design)
        
        # 添加低性能指标
        low_metrics = {
            'accuracy': 0.35,  # 低于50%的准确率
            'loss': 1.5
        }
        
        self.monitor.add_performance_metrics(experiment_id, low_metrics)
        
        # 验证是否生成了低性能预警
        alerts = self.monitor.get_experiment_alerts(experiment_id)
        performance_alerts = [alert for alert in alerts if alert.alert_type == 'low_performance']
        self.assertGreater(len(performance_alerts), 0)
        
        # 验证预警级别
        self.assertEqual(performance_alerts[0].level, AlertLevel.WARNING)
        
        print(f"低性能预警测试通过，准确率: {low_metrics['accuracy']:.1%}")
    
    def test_optimization_suggestions(self):
        """测试优化建议生成"""
        # 开始监控实验
        experiment_id = self.monitor.start_experiment_monitoring(self.experiment_design)
        
        # 触发资源使用预警（会自动生成优化建议）
        high_resource_usage = {
            'gpu': 0.95  # 高GPU使用率
        }
        
        self.monitor.update_resource_usage(experiment_id, high_resource_usage)
        
        # 获取优化建议
        suggestions = self.monitor.get_optimization_suggestions(experiment_id)
        
        # 验证建议生成
        self.assertGreater(len(suggestions), 0)
        
        # 验证建议内容
        resource_suggestions = [s for s in suggestions if s.category == 'resource']
        self.assertGreater(len(resource_suggestions), 0)
        
        print(f"优化建议测试通过，生成了 {len(suggestions)} 个建议")
        for suggestion in suggestions:
            print(f"- {suggestion.title}: {suggestion.description}")
    
    def test_monitoring_report_generation(self):
        """测试监控报告生成"""
        # 开始监控实验
        experiment_id = self.monitor.start_experiment_monitoring(self.experiment_design)
        
        # 更新实验状态和数据
        self.monitor.update_experiment_status(experiment_id, ExperimentStatus.RUNNING, "模型训练", 5)
        self.monitor.update_resource_usage(experiment_id, {'cpu': 0.70, 'gpu': 0.90})
        self.monitor.add_performance_metrics(experiment_id, {'accuracy': 0.88, 'loss': 0.15})
        
        # 生成监控报告
        report = self.monitor.generate_monitoring_report(experiment_id)
        
        # 验证报告内容
        self.assertIsNotNone(report.progress)
        self.assertEqual(report.experiment_id, experiment_id)
        self.assertGreater(len(report.statistics), 0)
        
        # 验证统计信息
        self.assertIn('progress_percentage', report.statistics)
        self.assertIn('total_alerts', report.statistics)
        self.assertIn('resource_usage_summary', report.statistics)
        
        print("监控报告生成测试通过")
        print(f"- 进度: {report.statistics['progress_percentage']:.1f}%")
        print(f"- 预警数量: {report.statistics['total_alerts']}")
        print(f"- 建议数量: {report.statistics['total_suggestions']}")
    
    def test_experiment_completion(self):
        """测试实验完成流程"""
        # 开始监控实验
        experiment_id = self.monitor.start_experiment_monitoring(self.experiment_design)
        
        # 模拟实验进行过程
        self.monitor.update_experiment_status(experiment_id, ExperimentStatus.RUNNING, "开始训练", 1)
        time.sleep(0.1)  # 短暂等待以确保时间差异
        
        self.monitor.update_experiment_status(experiment_id, ExperimentStatus.RUNNING, "模型训练", 5)
        time.sleep(0.1)
        
        # 完成实验
        self.monitor.update_experiment_status(experiment_id, ExperimentStatus.COMPLETED, "实验完成", 10)
        
        # 验证实验状态
        progress = self.monitor.get_experiment_progress(experiment_id)
        self.assertEqual(progress.status, ExperimentStatus.COMPLETED)
        self.assertEqual(progress.progress_percentage, 100.0)
        self.assertIsNotNone(progress.end_time)
        
        # 验证执行时间
        elapsed_time = progress.get_elapsed_time()
        self.assertIsNotNone(elapsed_time)
        self.assertGreater(elapsed_time.total_seconds(), 0)
        
        print(f"实验完成测试通过，总耗时: {elapsed_time.total_seconds():.2f}秒")
    
    def test_save_monitoring_data(self):
        """测试监控数据保存"""
        # 开始监控实验
        experiment_id = self.monitor.start_experiment_monitoring(self.experiment_design)
        
        # 添加一些监控数据
        self.monitor.update_experiment_status(experiment_id, ExperimentStatus.RUNNING, "测试保存", 3)
        self.monitor.add_performance_metrics(experiment_id, {'accuracy': 0.75})
        
        # 保存监控数据
        filepath = os.path.join(self.temp_dir, f"{experiment_id}_monitoring.json")
        self.monitor.save_monitoring_data(experiment_id, filepath)
        
        # 验证文件已创建
        self.assertTrue(os.path.exists(filepath))
        
        # 验证文件内容
        with open(filepath, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        self.assertEqual(saved_data['experiment_id'], experiment_id)
        self.assertIn('progress', saved_data)
        self.assertIn('statistics', saved_data)
        
        print(f"监控数据保存测试通过，文件大小: {os.path.getsize(filepath)} 字节")
    
    def test_multiple_experiments_monitoring(self):
        """测试多实验监控"""
        # 创建多个实验
        experiment_ids = []
        for i in range(3):
            design = ExperimentDesign(
                methodology=f"test_method_{i+1}",
                parameters={'param': i},
                timeline_weeks=4,
                expected_outcomes=[f"测试结果 {i+1}"],
                feasibility_score=0.7
            )
            exp_id = self.monitor.start_experiment_monitoring(design, f"test_exp_{i+1}")
            experiment_ids.append(exp_id)
        
        # 更新不同实验的状态
        self.monitor.update_experiment_status(experiment_ids[0], ExperimentStatus.RUNNING, "步骤1", 2)
        self.monitor.update_experiment_status(experiment_ids[1], ExperimentStatus.PAUSED, "暂停", 1)
        self.monitor.update_experiment_status(experiment_ids[2], ExperimentStatus.COMPLETED, "完成", 5)
        
        # 验证所有实验都在监控中
        all_experiments = self.monitor.get_all_experiments()
        self.assertEqual(len(all_experiments), 3)
        
        # 验证不同实验的状态
        self.assertEqual(all_experiments[experiment_ids[0]].status, ExperimentStatus.RUNNING)
        self.assertEqual(all_experiments[experiment_ids[1]].status, ExperimentStatus.PAUSED)
        self.assertEqual(all_experiments[experiment_ids[2]].status, ExperimentStatus.COMPLETED)
        
        print(f"多实验监控测试通过，监控 {len(all_experiments)} 个实验")
        for exp_id, progress in all_experiments.items():
            print(f"- {exp_id}: {progress.status.value} ({progress.progress_percentage:.1f}%)")
    
    def test_alert_resolution(self):
        """测试预警解决"""
        # 开始监控实验
        experiment_id = self.monitor.start_experiment_monitoring(self.experiment_design)
        
        # 触发预警
        self.monitor.update_resource_usage(experiment_id, {'gpu': 0.95})
        
        # 获取预警
        alerts = self.monitor.get_experiment_alerts(experiment_id)
        self.assertGreater(len(alerts), 0)
        
        # 解决预警
        alert = alerts[0]
        self.assertFalse(alert.resolved)
        alert.resolve()
        self.assertTrue(alert.resolved)
        
        # 生成报告验证活跃预警数量
        report = self.monitor.generate_monitoring_report(experiment_id)
        active_alerts = report.get_active_alerts()
        self.assertEqual(len(active_alerts), len(alerts) - 1)  # 减少一个已解决的预警
        
        print(f"预警解决测试通过，剩余活跃预警: {len(active_alerts)}")


if __name__ == '__main__':
    unittest.main(verbosity=2)