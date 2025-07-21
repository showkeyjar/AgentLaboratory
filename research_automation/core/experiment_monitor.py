"""
实验进度监控组件

提供实验状态跟踪、进度预警和优化建议功能
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import os

from ..models.analysis_models import ExperimentDesign
from ..models.base_models import BaseModel
from .base_component import BaseComponent

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """实验状态枚举"""
    PLANNED = "planned"          # 已计划
    PREPARING = "preparing"      # 准备中
    RUNNING = "running"          # 运行中
    PAUSED = "paused"           # 暂停
    COMPLETED = "completed"      # 已完成
    FAILED = "failed"           # 失败
    CANCELLED = "cancelled"      # 已取消


class AlertLevel(Enum):
    """预警级别枚举"""
    INFO = "info"               # 信息
    WARNING = "warning"         # 警告
    CRITICAL = "critical"       # 严重
    URGENT = "urgent"           # 紧急


@dataclass
class ExperimentProgress(BaseModel):
    """实验进度信息"""
    experiment_id: str = ""
    experiment_name: str = ""
    status: ExperimentStatus = ExperimentStatus.PLANNED
    
    # 时间信息
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    last_update: datetime = field(default_factory=datetime.now)
    
    # 进度信息
    total_steps: int = 0
    completed_steps: int = 0
    current_step: str = ""
    progress_percentage: float = 0.0
    
    # 资源使用情况
    resource_usage: Dict[str, float] = field(default_factory=dict)
    estimated_remaining_time: Optional[timedelta] = None
    
    # 结果信息
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    def update_progress(self, completed_steps: int, current_step: str = ""):
        """更新进度信息"""
        self.completed_steps = min(completed_steps, self.total_steps)
        self.current_step = current_step
        self.progress_percentage = (self.completed_steps / self.total_steps * 100) if self.total_steps > 0 else 0.0
        self.last_update = datetime.now()
    
    def add_intermediate_result(self, key: str, value: Any):
        """添加中间结果"""
        self.intermediate_results[key] = value
        self.last_update = datetime.now()
    
    def update_resource_usage(self, resource_type: str, usage: float):
        """更新资源使用情况"""
        self.resource_usage[resource_type] = usage
        self.last_update = datetime.now()
    
    def get_elapsed_time(self) -> Optional[timedelta]:
        """获取已用时间"""
        if self.start_time:
            end_time = self.end_time or datetime.now()
            return end_time - self.start_time
        return None
    
    def is_overdue(self, planned_duration: timedelta) -> bool:
        """检查是否超时"""
        elapsed = self.get_elapsed_time()
        return elapsed is not None and elapsed > planned_duration


@dataclass
class ExperimentAlert(BaseModel):
    """实验预警信息"""
    alert_id: str = ""
    experiment_id: str = ""
    alert_type: str = ""
    level: AlertLevel = AlertLevel.INFO
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    
    def resolve(self):
        """标记预警为已解决"""
        self.resolved = True


@dataclass
class OptimizationSuggestion(BaseModel):
    """优化建议"""
    suggestion_id: str = ""
    experiment_id: str = ""
    category: str = ""  # performance, resource, quality, time
    priority: int = 1   # 1-5, 5为最高优先级
    title: str = ""
    description: str = ""
    expected_impact: str = ""
    implementation_effort: str = ""  # low, medium, high
    timestamp: datetime = field(default_factory=datetime.now)
    applied: bool = False
    
    def apply(self):
        """标记建议为已应用"""
        self.applied = True


@dataclass
class MonitoringReport(BaseModel):
    """监控报告"""
    report_id: str = ""
    experiment_id: str = ""
    report_type: str = "progress"  # progress, alert, optimization
    timestamp: datetime = field(default_factory=datetime.now)
    
    # 进度信息
    progress: Optional[ExperimentProgress] = None
    
    # 预警信息
    alerts: List[ExperimentAlert] = field(default_factory=list)
    
    # 优化建议
    suggestions: List[OptimizationSuggestion] = field(default_factory=list)
    
    # 统计信息
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    def get_active_alerts(self) -> List[ExperimentAlert]:
        """获取未解决的预警"""
        return [alert for alert in self.alerts if not alert.resolved]
    
    def get_high_priority_suggestions(self) -> List[OptimizationSuggestion]:
        """获取高优先级建议"""
        return [suggestion for suggestion in self.suggestions 
                if suggestion.priority >= 4 and not suggestion.applied]


class ExperimentMonitor(BaseComponent):
    """实验进度监控器"""
    
    def get_required_configs(self) -> List[str]:
        """获取必需的配置项"""
        return []
    
    def _setup_component(self):
        """设置组件特定的初始化逻辑"""
        self.logger.info("实验进度监控器初始化")
        
        # 存储实验进度信息
        self.experiments: Dict[str, ExperimentProgress] = {}
        
        # 存储预警信息
        self.alerts: Dict[str, List[ExperimentAlert]] = {}
        
        # 存储优化建议
        self.suggestions: Dict[str, List[OptimizationSuggestion]] = {}
        
        # 监控配置
        self.monitoring_config = {
            'check_interval_minutes': 5,
            'alert_thresholds': {
                'progress_delay_hours': 2,
                'resource_usage_threshold': 0.8,
                'performance_drop_threshold': 0.1
            },
            'auto_suggestions': True
        }
        
        self.logger.info("实验进度监控器初始化完成")
    
    def start_experiment_monitoring(self, 
                                  experiment_design: ExperimentDesign,
                                  experiment_id: Optional[str] = None) -> str:
        """
        开始监控实验
        
        Args:
            experiment_design: 实验设计
            experiment_id: 实验ID（可选）
            
        Returns:
            实验ID
        """
        try:
            # 生成实验ID
            if experiment_id is None:
                experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # 创建进度跟踪对象
            progress = ExperimentProgress(
                experiment_id=experiment_id,
                experiment_name=experiment_design.methodology,
                status=ExperimentStatus.PLANNED,
                total_steps=len(experiment_design.parameters) + 5,  # 参数数量加上固定步骤
                start_time=datetime.now()
            )
            
            # 存储进度信息
            self.experiments[experiment_id] = progress
            self.alerts[experiment_id] = []
            self.suggestions[experiment_id] = []
            
            self.logger.info(f"开始监控实验: {experiment_id}")
            return experiment_id
            
        except Exception as e:
            self.logger.error(f"开始实验监控失败: {str(e)}")
            raise
    
    def update_experiment_status(self, 
                               experiment_id: str, 
                               status: ExperimentStatus,
                               current_step: str = "",
                               completed_steps: Optional[int] = None):
        """
        更新实验状态
        
        Args:
            experiment_id: 实验ID
            status: 新状态
            current_step: 当前步骤
            completed_steps: 已完成步骤数
        """
        try:
            if experiment_id not in self.experiments:
                raise ValueError(f"实验 {experiment_id} 不存在")
            
            progress = self.experiments[experiment_id]
            progress.status = status
            progress.current_step = current_step
            
            if completed_steps is not None:
                progress.update_progress(completed_steps, current_step)
            
            # 根据状态更新时间
            if status == ExperimentStatus.COMPLETED or status == ExperimentStatus.FAILED:
                progress.end_time = datetime.now()
            
            self.logger.info(f"实验 {experiment_id} 状态更新为: {status.value}")
            
            # 检查是否需要生成预警
            self._check_for_alerts(experiment_id)
            
        except Exception as e:
            self.logger.error(f"更新实验状态失败: {str(e)}")
            raise
    
    def update_resource_usage(self, 
                            experiment_id: str, 
                            resource_usage: Dict[str, float]):
        """
        更新资源使用情况
        
        Args:
            experiment_id: 实验ID
            resource_usage: 资源使用情况
        """
        try:
            if experiment_id not in self.experiments:
                raise ValueError(f"实验 {experiment_id} 不存在")
            
            progress = self.experiments[experiment_id]
            for resource_type, usage in resource_usage.items():
                progress.update_resource_usage(resource_type, usage)
            
            self.logger.debug(f"实验 {experiment_id} 资源使用情况已更新")
            
            # 检查资源使用预警
            self._check_resource_alerts(experiment_id)
            
        except Exception as e:
            self.logger.error(f"更新资源使用情况失败: {str(e)}")
            raise
    
    def add_performance_metrics(self, 
                              experiment_id: str, 
                              metrics: Dict[str, float]):
        """
        添加性能指标
        
        Args:
            experiment_id: 实验ID
            metrics: 性能指标
        """
        try:
            if experiment_id not in self.experiments:
                raise ValueError(f"实验 {experiment_id} 不存在")
            
            progress = self.experiments[experiment_id]
            progress.performance_metrics.update(metrics)
            progress.last_update = datetime.now()
            
            self.logger.debug(f"实验 {experiment_id} 性能指标已更新")
            
            # 检查性能预警
            self._check_performance_alerts(experiment_id)
            
        except Exception as e:
            self.logger.error(f"添加性能指标失败: {str(e)}")
            raise
    
    def get_experiment_progress(self, experiment_id: str) -> Optional[ExperimentProgress]:
        """
        获取实验进度
        
        Args:
            experiment_id: 实验ID
            
        Returns:
            实验进度信息
        """
        return self.experiments.get(experiment_id)
    
    def get_experiment_alerts(self, experiment_id: str) -> List[ExperimentAlert]:
        """
        获取实验预警
        
        Args:
            experiment_id: 实验ID
            
        Returns:
            预警列表
        """
        return self.alerts.get(experiment_id, [])
    
    def get_optimization_suggestions(self, experiment_id: str) -> List[OptimizationSuggestion]:
        """
        获取优化建议
        
        Args:
            experiment_id: 实验ID
            
        Returns:
            优化建议列表
        """
        return self.suggestions.get(experiment_id, [])
    
    def generate_monitoring_report(self, experiment_id: str) -> MonitoringReport:
        """
        生成监控报告
        
        Args:
            experiment_id: 实验ID
            
        Returns:
            监控报告
        """
        try:
            if experiment_id not in self.experiments:
                raise ValueError(f"实验 {experiment_id} 不存在")
            
            progress = self.experiments[experiment_id]
            alerts = self.alerts.get(experiment_id, [])
            suggestions = self.suggestions.get(experiment_id, [])
            
            # 生成统计信息
            statistics = self._generate_statistics(experiment_id)
            
            report = MonitoringReport(
                report_id=f"report_{experiment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                experiment_id=experiment_id,
                progress=progress,
                alerts=alerts,
                suggestions=suggestions,
                statistics=statistics
            )
            
            self.logger.info(f"生成实验 {experiment_id} 监控报告")
            return report
            
        except Exception as e:
            self.logger.error(f"生成监控报告失败: {str(e)}")
            raise
    
    def _check_for_alerts(self, experiment_id: str):
        """检查并生成预警"""
        try:
            progress = self.experiments[experiment_id]
            
            # 检查进度延迟
            if progress.status == ExperimentStatus.RUNNING:
                elapsed = progress.get_elapsed_time()
                if elapsed and elapsed.total_seconds() > self.monitoring_config['alert_thresholds']['progress_delay_hours'] * 3600:
                    if progress.progress_percentage < 50:  # 如果进度小于50%但时间已过半
                        self._create_alert(
                            experiment_id,
                            "progress_delay",
                            AlertLevel.WARNING,
                            f"实验进度可能延迟，当前进度: {progress.progress_percentage:.1f}%",
                            {"elapsed_hours": elapsed.total_seconds() / 3600}
                        )
            
            # 检查实验状态异常
            if progress.status == ExperimentStatus.FAILED:
                self._create_alert(
                    experiment_id,
                    "experiment_failed",
                    AlertLevel.CRITICAL,
                    "实验执行失败",
                    {"current_step": progress.current_step}
                )
            
        except Exception as e:
            self.logger.error(f"检查预警失败: {str(e)}")
    
    def _check_resource_alerts(self, experiment_id: str):
        """检查资源使用预警"""
        try:
            progress = self.experiments[experiment_id]
            threshold = self.monitoring_config['alert_thresholds']['resource_usage_threshold']
            
            for resource_type, usage in progress.resource_usage.items():
                if usage > threshold:
                    self._create_alert(
                        experiment_id,
                        f"high_{resource_type}_usage",
                        AlertLevel.WARNING,
                        f"{resource_type}使用率过高: {usage:.1%}",
                        {"resource_type": resource_type, "usage": usage}
                    )
            
        except Exception as e:
            self.logger.error(f"检查资源预警失败: {str(e)}")
    
    def _check_performance_alerts(self, experiment_id: str):
        """检查性能预警"""
        try:
            progress = self.experiments[experiment_id]
            
            # 检查性能下降
            if 'accuracy' in progress.performance_metrics:
                current_accuracy = progress.performance_metrics['accuracy']
                if current_accuracy < 0.5:  # 假设准确率低于50%为异常
                    self._create_alert(
                        experiment_id,
                        "low_performance",
                        AlertLevel.WARNING,
                        f"模型性能较低: 准确率 {current_accuracy:.2%}",
                        {"accuracy": current_accuracy}
                    )
            
        except Exception as e:
            self.logger.error(f"检查性能预警失败: {str(e)}")
    
    def _create_alert(self, 
                     experiment_id: str, 
                     alert_type: str, 
                     level: AlertLevel, 
                     message: str, 
                     details: Dict[str, Any]):
        """创建预警"""
        try:
            alert = ExperimentAlert(
                alert_id=f"alert_{experiment_id}_{len(self.alerts[experiment_id])}",
                experiment_id=experiment_id,
                alert_type=alert_type,
                level=level,
                message=message,
                details=details
            )
            
            self.alerts[experiment_id].append(alert)
            self.logger.warning(f"创建预警: {message}")
            
            # 如果启用自动建议，生成优化建议
            if self.monitoring_config['auto_suggestions']:
                self._generate_optimization_suggestions(experiment_id, alert)
            
        except Exception as e:
            self.logger.error(f"创建预警失败: {str(e)}")
    
    def _generate_optimization_suggestions(self, experiment_id: str, alert: ExperimentAlert):
        """基于预警生成优化建议"""
        try:
            suggestions = []
            
            if alert.alert_type == "progress_delay":
                suggestions.append(OptimizationSuggestion(
                    suggestion_id=f"sugg_{experiment_id}_{len(self.suggestions[experiment_id])}",
                    experiment_id=experiment_id,
                    category="time",
                    priority=4,
                    title="加速实验进度",
                    description="考虑增加计算资源或优化算法参数以加速实验进度",
                    expected_impact="减少20-30%的执行时间",
                    implementation_effort="medium"
                ))
            
            elif alert.alert_type.startswith("high_") and alert.alert_type.endswith("_usage"):
                resource_type = alert.alert_type.replace("high_", "").replace("_usage", "")
                suggestions.append(OptimizationSuggestion(
                    suggestion_id=f"sugg_{experiment_id}_{len(self.suggestions[experiment_id])}",
                    experiment_id=experiment_id,
                    category="resource",
                    priority=3,
                    title=f"优化{resource_type}使用",
                    description=f"考虑优化{resource_type}使用策略或增加{resource_type}配额",
                    expected_impact="降低资源使用压力",
                    implementation_effort="low"
                ))
            
            elif alert.alert_type == "low_performance":
                suggestions.append(OptimizationSuggestion(
                    suggestion_id=f"sugg_{experiment_id}_{len(self.suggestions[experiment_id])}",
                    experiment_id=experiment_id,
                    category="performance",
                    priority=5,
                    title="提升模型性能",
                    description="考虑调整超参数、增加训练数据或改进模型架构",
                    expected_impact="提升10-20%的模型性能",
                    implementation_effort="high"
                ))
            
            # 添加建议到列表
            self.suggestions[experiment_id].extend(suggestions)
            
            for suggestion in suggestions:
                self.logger.info(f"生成优化建议: {suggestion.title}")
            
        except Exception as e:
            self.logger.error(f"生成优化建议失败: {str(e)}")
    
    def _generate_statistics(self, experiment_id: str) -> Dict[str, Any]:
        """生成统计信息"""
        try:
            progress = self.experiments[experiment_id]
            alerts = self.alerts.get(experiment_id, [])
            suggestions = self.suggestions.get(experiment_id, [])
            
            statistics = {
                'experiment_duration': None,
                'progress_percentage': progress.progress_percentage,
                'total_alerts': len(alerts),
                'active_alerts': len([a for a in alerts if not a.resolved]),
                'total_suggestions': len(suggestions),
                'pending_suggestions': len([s for s in suggestions if not s.applied]),
                'resource_usage_summary': dict(progress.resource_usage),
                'performance_summary': dict(progress.performance_metrics)
            }
            
            # 计算实验持续时间
            if progress.start_time:
                end_time = progress.end_time or datetime.now()
                duration = end_time - progress.start_time
                statistics['experiment_duration'] = duration.total_seconds()
            
            return statistics
            
        except Exception as e:
            self.logger.error(f"生成统计信息失败: {str(e)}")
            return {}
    
    def save_monitoring_data(self, experiment_id: str, filepath: str):
        """保存监控数据到文件"""
        try:
            report = self.generate_monitoring_report(experiment_id)
            
            # 创建目录（如果不存在）
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # 保存到JSON文件
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report.to_dict(), f, ensure_ascii=False, indent=2, default=str)
            
            self.logger.info(f"监控数据已保存到: {filepath}")
            
        except Exception as e:
            self.logger.error(f"保存监控数据失败: {str(e)}")
            raise
    
    def get_all_experiments(self) -> Dict[str, ExperimentProgress]:
        """获取所有实验的进度信息"""
        return self.experiments.copy()
    
    def stop_experiment_monitoring(self, experiment_id: str):
        """停止监控实验"""
        try:
            if experiment_id in self.experiments:
                progress = self.experiments[experiment_id]
                if progress.status == ExperimentStatus.RUNNING:
                    progress.status = ExperimentStatus.CANCELLED
                    progress.end_time = datetime.now()
                
                self.logger.info(f"停止监控实验: {experiment_id}")
            
        except Exception as e:
            self.logger.error(f"停止实验监控失败: {str(e)}")
            raise