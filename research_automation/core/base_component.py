"""
基础组件类

为所有研究自动化组件提供通用功能和接口
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import logging
import uuid

from ..models.base_models import BaseModel
from .exceptions import ResearchAutomationError, ConfigurationError, handle_exception


class BaseComponent(ABC):
    """所有研究自动化组件的基类"""
    
    def __init__(self, config: Dict[str, Any] = None, logger: logging.Logger = None):
        """
        初始化基础组件
        
        Args:
            config: 组件配置字典
            logger: 日志记录器
        """
        self.config = config or {}
        self.component_id = str(uuid.uuid4())
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
        self.is_initialized = False
        self.metrics = {}
        self.logger = logger or self._setup_default_logger()
        
        # 验证配置
        self._validate_config()
        
        # 初始化组件
        self._initialize()
    
    def _setup_default_logger(self) -> logging.Logger:
        """设置默认日志记录器"""
        logger = logging.getLogger(f"{self.__class__.__name__}_{self.component_id[:8]}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _validate_config(self):
        """验证组件配置"""
        required_configs = self.get_required_configs()
        for config_key in required_configs:
            if config_key not in self.config:
                raise ConfigurationError(
                    f"缺少必需的配置项: {config_key}",
                    config_key=config_key
                )
    
    @abstractmethod
    def get_required_configs(self) -> List[str]:
        """获取必需的配置项列表"""
        pass
    
    def _initialize(self):
        """初始化组件"""
        try:
            self.logger.info(f"正在初始化组件: {self.__class__.__name__}")
            self._setup_component()
            self.is_initialized = True
            self.logger.info(f"组件初始化完成: {self.__class__.__name__}")
        except Exception as e:
            self.logger.error(f"组件初始化失败: {str(e)}")
            raise
    
    @abstractmethod
    def _setup_component(self):
        """设置组件特定的初始化逻辑"""
        pass
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        return self.config.get(key, default)
    
    def set_config(self, key: str, value: Any):
        """设置配置值"""
        self.config[key] = value
        self.last_updated = datetime.now()
        self.logger.debug(f"配置已更新: {key} = {value}")
    
    def update_config(self, new_config: Dict[str, Any]):
        """批量更新配置"""
        self.config.update(new_config)
        self.last_updated = datetime.now()
        self.logger.info(f"配置已批量更新: {list(new_config.keys())}")
    
    def get_status(self) -> Dict[str, Any]:
        """获取组件状态"""
        return {
            'component_id': self.component_id,
            'component_name': self.__class__.__name__,
            'is_initialized': self.is_initialized,
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'metrics': self.metrics
        }
    
    def update_metric(self, metric_name: str, value: Any):
        """更新性能指标"""
        self.metrics[metric_name] = {
            'value': value,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_metric(self, metric_name: str) -> Optional[Any]:
        """获取性能指标"""
        metric_data = self.metrics.get(metric_name)
        return metric_data['value'] if metric_data else None
    
    @handle_exception
    def validate_input(self, data: Any, schema: Dict[str, Any] = None) -> bool:
        """验证输入数据"""
        if schema is None:
            return True
        
        # 简化的验证逻辑，实际应用中可以使用更复杂的验证框架
        if isinstance(data, dict) and isinstance(schema, dict):
            for key, expected_type in schema.items():
                if key not in data:
                    raise ValidationError(f"缺少必需字段: {key}", field_name=key)
                if not isinstance(data[key], expected_type):
                    raise ValidationError(
                        f"字段类型错误: {key}，期望 {expected_type.__name__}",
                        field_name=key,
                        invalid_value=data[key]
                    )
        
        return True
    
    def log_operation(self, operation: str, details: Dict[str, Any] = None):
        """记录操作日志"""
        log_entry = {
            'component': self.__class__.__name__,
            'operation': operation,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
        self.logger.info(f"操作记录: {operation}", extra=log_entry)
    
    def handle_error(self, error: Exception, context: str = ""):
        """处理错误"""
        error_msg = f"组件错误 [{context}]: {str(error)}"
        self.logger.error(error_msg, exc_info=True)
        
        if isinstance(error, ResearchAutomationError):
            raise error
        else:
            raise ResearchAutomationError(
                error_msg,
                error_code="COMPONENT_ERROR",
                details={'context': context, 'original_error': str(error)}
            )
    
    def cleanup(self):
        """清理资源"""
        self.logger.info(f"正在清理组件资源: {self.__class__.__name__}")
        # 子类可以重写此方法来实现特定的清理逻辑
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.cleanup()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'component_id': self.component_id,
            'component_name': self.__class__.__name__,
            'config': self.config,
            'status': self.get_status(),
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat()
        }
    
    def __repr__(self) -> str:
        """字符串表示"""
        return f"{self.__class__.__name__}(id={self.component_id[:8]}, initialized={self.is_initialized})"


# 导入ValidationError
from .exceptions import ValidationError