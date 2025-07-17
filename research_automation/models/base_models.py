"""
基础数据模型定义

提供所有其他模型的基础类和通用功能
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod
import uuid
import json


@dataclass
class BaseModel:
    """所有数据模型的基类"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, BaseModel):
                result[key] = value.to_dict()
            elif isinstance(value, list) and value and isinstance(value[0], BaseModel):
                result[key] = [item.to_dict() for item in value]
            else:
                result[key] = value
        return result
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """从字典创建实例"""
        # 处理datetime字段
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        
        return cls(**data)


class TimestampMixin:
    """时间戳混入类"""
    
    def update_timestamp(self):
        """更新时间戳"""
        self.updated_at = datetime.now()


class ValidationMixin:
    """验证混入类"""
    
    @abstractmethod
    def validate(self) -> bool:
        """验证数据有效性"""
        pass
    
    def get_validation_errors(self) -> List[str]:
        """获取验证错误列表"""
        errors = []
        try:
            if not self.validate():
                errors.append("数据验证失败")
        except Exception as e:
            errors.append(f"验证过程中出现错误: {str(e)}")
        return errors


@dataclass
class MetricScore:
    """评分指标基类"""
    value: float
    max_value: float = 1.0
    min_value: float = 0.0
    confidence: float = 1.0
    
    def __post_init__(self):
        """初始化后验证"""
        if not (self.min_value <= self.value <= self.max_value):
            raise ValueError(f"分数 {self.value} 超出范围 [{self.min_value}, {self.max_value}]")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"置信度 {self.confidence} 必须在 [0.0, 1.0] 范围内")
    
    def normalize(self) -> float:
        """归一化分数到 [0, 1] 范围"""
        return (self.value - self.min_value) / (self.max_value - self.min_value)
    
    def is_high_quality(self, threshold: float = 0.7) -> bool:
        """判断是否为高质量"""
        return self.normalize() >= threshold


@dataclass 
class StatusInfo:
    """状态信息基类"""
    status: str
    message: str = ""
    progress: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后验证"""
        if not (0.0 <= self.progress <= 1.0):
            raise ValueError(f"进度 {self.progress} 必须在 [0.0, 1.0] 范围内")
    
    def is_completed(self) -> bool:
        """判断是否已完成"""
        return self.status.lower() in ['completed', 'finished', 'done']
    
    def is_failed(self) -> bool:
        """判断是否失败"""
        return self.status.lower() in ['failed', 'error', 'cancelled']
    
    def is_in_progress(self) -> bool:
        """判断是否进行中"""
        return self.status.lower() in ['running', 'processing', 'in_progress']