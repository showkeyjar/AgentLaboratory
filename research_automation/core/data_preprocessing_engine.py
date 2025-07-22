"""
数据预处理引擎
提供数据清洗、转换、标准化和特征工程功能
"""

import logging
import os
import re
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import datetime
import uuid

from ..models.base_models import BaseModel
from .base_component import BaseComponent

# 尝试导入可选依赖
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

try:
    from sklearn import preprocessing
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class PreprocessingStepType(Enum):
    """预处理步骤类型"""
    CLEANING = "cleaning"           # 数据清洗
    TRANSFORMATION = "transformation" # 数据转换
    NORMALIZATION = "normalization"   # 数据标准化
    FEATURE_ENGINEERING = "feature_engineering" # 特征工程
    CUSTOM = "custom"             # 自定义处理


class DataType(Enum):
    """数据类型"""
    NUMERIC = "numeric"           # 数值型
    CATEGORICAL = "categorical"     # 类别型
    TEXT = "text"               # 文本型
    DATETIME = "datetime"          # 日期时间型
    BOOLEAN = "boolean"           # 布尔型
    ARRAY = "array"              # 数组型
    MIXED = "mixed"              # 混合型
    UNKNOWN = "unknown"           # 未知型


@dataclass
class ColumnProfile(BaseModel):
    """列数据概况"""
    name: str = ""
    data_type: DataType = DataType.UNKNOWN
    # 基本统计信息
    count: int = 0
    missing_count: int = 0
    unique_count: int = 0
    # 数值型统计
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    std_dev: Optional[float] = None
    # 类别型统计
    top_categories: Dict[str, int] = field(default_factory=dict)
    # 文本型统计
    avg_length: Optional[float] = None
    max_length: Optional[int] = None
    # 质量指标
    completeness: float = 1.0  # 完整度(非空值比例)
    uniqueness: float = 0.0    # 唯一性(唯一值比例)
    
    def get_missing_rate(self) -> float:
        """获取缺失率"""
        return self.missing_count / self.count if self.count > 0 else 0.0
    
    def get_quality_score(self) -> float:
        """获取质量评分"""
        # 简单加权平均
        return 0.7 * self.completeness + 0.3 * self.uniqueness


@dataclass
class DatasetProfile(BaseModel):
    """数据集概况"""
    # 基本信息
    row_count: int = 0
    column_count: int = 0
    memory_usage: Optional[int] = None
    # 列概况
    columns: Dict[str, ColumnProfile] = field(default_factory=dict)
    # 相关性矩阵
    correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None
    # 质量评分
    quality_score: float = 0.0
    # 数据类型分布
    data_type_counts: Dict[str, int] = field(default_factory=dict)
    
    def get_column_names(self) -> List[str]:
        """获取列名列表"""
        return list(self.columns.keys())
    
    def get_numeric_columns(self) -> List[str]:
        """获取数值型列名列表"""
        return [name for name, profile in self.columns.items() 
                if profile.data_type == DataType.NUMERIC]
    
    def get_categorical_columns(self) -> List[str]:
        """获取类别型列名列表"""
        return [name for name, profile in self.columns.items() 
                if profile.data_type == DataType.CATEGORICAL]
    
    def get_text_columns(self) -> List[str]:
        """获取文本型列名列表"""
        return [name for name, profile in self.columns.items() 
                if profile.data_type == DataType.TEXT]
    
    def get_datetime_columns(self) -> List[str]:
        """获取日期时间型列名列表"""
        return [name for name, profile in self.columns.items() 
                if profile.data_type == DataType.DATETIME]
    
    def get_columns_with_missing_values(self) -> List[Tuple[str, int]]:
        """获取有缺失值的列及缺失数量"""
        return [(name, profile.missing_count) for name, profile in self.columns.items() 
                if profile.missing_count > 0]
    
    def calculate_quality_score(self) -> float:
        """计算整体质量评分"""
        if not self.columns:
            return 0.0
        # 计算所有列的质量评分平均值
        column_scores = [profile.get_quality_score() for profile in self.columns.values()]
        self.quality_score = sum(column_scores) / len(column_scores)
        return self.quality_score


@dataclass
class PreprocessingStep(BaseModel):
    """预处理步骤"""
    step_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    step_type: PreprocessingStepType = PreprocessingStepType.CLEANING
    description: str = ""
    # 目标列
    target_columns: List[str] = field(default_factory=list)
    # 步骤参数
    parameters: Dict[str, Any] = field(default_factory=dict)
    # 执行函数名称
    function_name: str = ""
    # 执行状态
    is_enabled: bool = True
    is_executed: bool = False
    execution_time: Optional[float] = None
    # 执行结果统计
    affected_rows: int = 0
    affected_columns: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            'step_id': self.step_id,
            'name': self.name,
            'step_type': self.step_type.value,
            'description': self.description,
            'target_columns': self.target_columns,
            'parameters': self.parameters,
            'function_name': self.function_name,
            'is_enabled': self.is_enabled,
            'is_executed': self.is_executed,
            'execution_time': self.execution_time,
            'affected_rows': self.affected_rows,
            'affected_columns': self.affected_columns
        }
        return result


@dataclass
class PreprocessingPipeline(BaseModel):
    """预处理流水线"""
    pipeline_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    # 预处理步骤
    steps: List[PreprocessingStep] = field(default_factory=list)
    # 执行状态
    is_executed: bool = False
    total_execution_time: Optional[float] = None
    
    def add_step(self, step: PreprocessingStep) -> None:
        """添加预处理步骤"""
        self.steps.append(step)
        self.updated_at = datetime.datetime.now()
    
    def remove_step(self, step_id: str) -> bool:
        """移除预处理步骤"""
        initial_length = len(self.steps)
        self.steps = [step for step in self.steps if step.step_id != step_id]
        self.updated_at = datetime.datetime.now()
        return len(self.steps) < initial_length
    
    def move_step(self, step_id: str, new_position: int) -> bool:
        """移动预处理步骤"""
        if new_position < 0 or new_position >= len(self.steps):
            return False
        # 找到步骤的当前位置
        current_position = None
        for i, step in enumerate(self.steps):
            if step.step_id == step_id:
                current_position = i
                break
        if current_position is None:
            return False
        # 移动步骤
        step = self.steps.pop(current_position)
        self.steps.insert(new_position, step)
        self.updated_at = datetime.datetime.now()
        return True
    
    def get_step(self, step_id: str) -> Optional[PreprocessingStep]:
        """获取预处理步骤"""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'pipeline_id': self.pipeline_id,
            'name': self.name,
            'description': self.description,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'steps': [step.to_dict() for step in self.steps],
            'is_executed': self.is_executed,
            'total_execution_time': self.total_execution_time
        }
    
    def save_to_json(self, filepath: str) -> None:
        """保存到JSON文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load_from_json(cls, filepath: str) -> 'PreprocessingPipeline':
        """从JSON文件加载"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        pipeline = cls(
            pipeline_id=data['pipeline_id'],
            name=data['name'],
            description=data['description'],
            created_at=datetime.datetime.fromisoformat(data['created_at']),
            updated_at=datetime.datetime.fromisoformat(data['updated_at']),
            is_executed=data['is_executed'],
            total_execution_time=data['total_execution_time']
        )
        # 加载步骤
        for step_data in data['steps']:
            step = PreprocessingStep(
                step_id=step_data['step_id'],
                name=step_data['name'],
                step_type=PreprocessingStepType(step_data['step_type']),
                description=step_data['description'],
                target_columns=step_data['target_columns'],
                parameters=step_data['parameters'],
                function_name=step_data['function_name'],
                is_enabled=step_data['is_enabled'],
                is_executed=step_data['is_executed'],
                execution_time=step_data['execution_time'],
                affected_rows=step_data['affected_rows'],
                affected_columns=step_data['affected_columns']
            )
            pipeline.steps.append(step)
        return pipeline


@dataclass
class PreprocessingResult(BaseModel):
    """预处理结果"""
    # 基本信息
    pipeline_id: str = ""
    execution_time: float = 0.0
    executed_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    success: bool = True
    # 数据统计
    input_rows: int = 0
    output_rows: int = 0
    input_columns: int = 0
    output_columns: int = 0
    # 步骤执行结果
    step_results: List[Dict[str, Any]] = field(default_factory=list)
    # 错误信息
    errors: List[str] = field(default_factory=list)
    # 数据质量变化
    input_quality_score: float = 0.0
    output_quality_score: float = 0.0
    
    def get_quality_improvement(self) -> float:
        """获取质量提升幅度"""
        return self.output_quality_score - self.input_quality_score
    
    def get_row_reduction_rate(self) -> float:
        """获取行数减少率"""
        if self.input_rows == 0:
            return 0.0
        return (self.input_rows - self.output_rows) / self.input_rows
    
    def get_summary(self) -> str:
        """获取结果摘要"""
        status = "成功" if self.success else "失败"
        quality_change = self.get_quality_improvement()
        quality_direction = "提升" if quality_change >= 0 else "下降"
        summary = [
            f"预处理结果: {status}",
            f"执行时间: {self.execution_time:.2f}秒",
            f"数据行数: {self.input_rows} → {self.output_rows}",
            f"数据列数: {self.input_columns} → {self.output_columns}",
            f"数据质量: {self.input_quality_score:.2f} → {self.output_quality_score:.2f} ({abs(quality_change):.2f}点{quality_direction})",
            f"执行步骤: {len(self.step_results)}个"
        ]
        if self.errors:
            summary.append(f"错误数量: {len(self.errors)}个")
        return "\n".join(summary)


class DataPreprocessingEngine(BaseComponent):
    """数据预处理引擎"""
    def get_required_configs(self) -> List[str]:
        """获取必需的配置项"""
        return []
    
    def _setup_component(self):
        """设置组件特定的初始化逻辑"""
        self.logger.info("数据预处理引擎初始化")
        # 注册预处理函数
        self._register_preprocessing_functions()
        self.logger.info("数据预处理引擎初始化完成")
    
    def _register_preprocessing_functions(self):
        """注册预处理函数"""
        # 清洗函数
        self.cleaning_functions = {
            'remove_missing_values': self._remove_missing_values,
            'fill_missing_values': self._fill_missing_values,
            'remove_duplicates': self._remove_duplicates,
            'remove_outliers': self._remove_outliers,
            'filter_by_value': self._filter_by_value,
            'clean_text': self._clean_text
        }
        # 转换函数
        self.transformation_functions = {
            'convert_type': self._convert_type,
            'apply_function': self._apply_function,
            'bin_numeric_data': self._bin_numeric_data,
            'encode_categorical': self._encode_categorical,
            'extract_datetime_features': self._extract_datetime_features
        }
        # 标准化函数
        self.normalization_functions = {
            'standardize': self._standardize,
            'min_max_scale': self._min_max_scale,
            'robust_scale': self._robust_scale,
            'normalize': self._normalize
        }
        # 特征工程函数
        self.feature_engineering_functions = {
            'create_interaction_features': self._create_interaction_features,
            'polynomial_features': self._polynomial_features,
            'aggregate_features': self._aggregate_features,
            'extract_text_features': self._extract_text_features
        }
        # 所有函数的集合
        self.all_functions = {}
        self.all_functions.update(self.cleaning_functions)
        self.all_functions.update(self.transformation_functions)
        self.all_functions.update(self.normalization_functions)
        self.all_functions.update(self.feature_engineering_functions)
    
    def analyze_data(self, data: Any) -> DatasetProfile:
        """
        分析数据集，生成数据概况
        Args:
            data: 输入数据（DataFrame或类似结构）
        Returns:
            数据集概况
        """
        try:
            self.logger.info("开始分析数据集")
            # 检查数据类型
            if HAS_PANDAS and isinstance(data, pd.DataFrame):
                return self._analyze_pandas_dataframe(data)
            elif isinstance(data, list) and data and isinstance(data[0], dict):
                # 转换为DataFrame（如果pandas可用）
                if HAS_PANDAS:
                    df = pd.DataFrame(data)
                    return self._analyze_pandas_dataframe(df)
                else:
                    return self._analyze_list_of_dicts(data)
            else:
                raise ValueError("不支持的数据格式，请提供DataFrame或字典列表")
        except Exception as e:
            self.logger.error(f"数据分析失败: {str(e)}")
            raise
    
    def _analyze_pandas_dataframe(self, df: 'pd.DataFrame') -> DatasetProfile:
        """分析Pandas DataFrame"""
        profile = DatasetProfile(
            row_count=len(df),
            column_count=len(df.columns),
            memory_usage=df.memory_usage(deep=True).sum()
        )
        # 分析每一列
        for column_name in df.columns:
            column_data = df[column_name]
            column_profile = self._analyze_column(column_name, column_data)
            profile.columns[column_name] = column_profile
        
        # 计算数据类型分布
        for col_profile in profile.columns.values():
            data_type = col_profile.data_type.value
            profile.data_type_counts[data_type] = profile.data_type_counts.get(data_type, 0) + 1
        
        # 计算相关性矩阵（仅对数值列）
        numeric_columns = profile.get_numeric_columns()
        if len(numeric_columns) > 1:
            try:
                corr_matrix = df[numeric_columns].corr().to_dict()
                profile.correlation_matrix = corr_matrix
            except Exception as e:
                self.logger.warning(f"计算相关性矩阵失败: {str(e)}")
        
        # 计算整体质量评分
        profile.calculate_quality_score()
        return profile
    
    def _analyze_list_of_dicts(self, data: List[Dict[str, Any]]) -> DatasetProfile:
        """分析字典列表"""
        profile = DatasetProfile(
            row_count=len(data),
            column_count=len(data[0]) if data else 0
        )
        # 获取所有列名
        column_names = set()
        for item in data:
            column_names.update(item.keys())
        
        # 分析每一列
        for column_name in column_names:
            # 提取列数据
            column_data = [item.get(column_name) for item in data]
            column_profile = self._analyze_column_from_list(column_name, column_data)
            profile.columns[column_name] = column_profile
        
        # 计算数据类型分布
        for col_profile in profile.columns.values():
            data_type = col_profile.data_type.value
            profile.data_type_counts[data_type] = profile.data_type_counts.get(data_type, 0) + 1
        
        # 计算整体质量评分
        profile.calculate_quality_score()
        return profile
    
    def _analyze_column(self, column_name: str, column_data: Any) -> ColumnProfile:
        """分析DataFrame的列"""
        profile = ColumnProfile(name=column_name)
        # 基本统计
        profile.count = len(column_data)
        profile.missing_count = column_data.isna().sum()
        profile.unique_count = column_data.nunique()
        
        # 计算完整度和唯一性
        profile.completeness = 1.0 - (profile.missing_count / profile.count) if profile.count > 0 else 0.0
        profile.uniqueness = profile.unique_count / profile.count if profile.count > 0 else 0.0
        
        # 确定数据类型
        if pd.api.types.is_numeric_dtype(column_data):
            profile.data_type = DataType.NUMERIC
            # 数值型统计
            non_null_data = column_data.dropna()
            if len(non_null_data) > 0:
                profile.min_value = float(non_null_data.min())
                profile.max_value = float(non_null_data.max())
                profile.mean = float(non_null_data.mean())
                profile.median = float(non_null_data.median())
                profile.std_dev = float(non_null_data.std())
        elif pd.api.types.is_categorical_dtype(column_data) or (profile.unique_count < 0.2 * profile.count and profile.unique_count < 100):
            profile.data_type = DataType.CATEGORICAL
            # 类别型统计
            value_counts = column_data.value_counts().head(10).to_dict()
            profile.top_categories = value_counts
        elif pd.api.types.is_datetime64_dtype(column_data):
            profile.data_type = DataType.DATETIME
        elif pd.api.types.is_bool_dtype(column_data):
            profile.data_type = DataType.BOOLEAN
        elif pd.api.types.is_string_dtype(column_data):
            # 进一步区分文本和类别
            if profile.unique_count < 0.2 * profile.count and profile.unique_count < 100:
                profile.data_type = DataType.CATEGORICAL
                value_counts = column_data.value_counts().head(10).to_dict()
                profile.top_categories = value_counts
            else:
                profile.data_type = DataType.TEXT
                # 文本型统计
                text_lengths = column_data.dropna().str.len()
                if len(text_lengths) > 0:
                    profile.avg_length = float(text_lengths.mean())
                    profile.max_length = int(text_lengths.max())
        else:
            profile.data_type = DataType.UNKNOWN
        
        return profile
    
    def _analyze_column_from_list(self, column_name: str, column_data: List[Any]) -> ColumnProfile:
        """从列表分析列"""
        profile = ColumnProfile(name=column_name)
        # 基本统计
        profile.count = len(column_data)
        profile.missing_count = sum(1 for x in column_data if x is None)
        unique_values = set(x for x in column_data if x is not None)
        profile.unique_count = len(unique_values)
        
        # 计算完整度和唯一性
        profile.completeness = 1.0 - (profile.missing_count / profile.count) if profile.count > 0 else 0.0
        profile.uniqueness = profile.unique_count / profile.count if profile.count > 0 else 0.0
        
        # 确定数据类型
        non_null_data = [x for x in column_data if x is not None]
        if not non_null_data:
            profile.data_type = DataType.UNKNOWN
            return profile
        
        # 检查数据类型
        sample_value = non_null_data[0]
        if isinstance(sample_value, (int, float)):
            profile.data_type = DataType.NUMERIC
            # 数值型统计
            profile.min_value = float(min(non_null_data))
            profile.max_value = float(max(non_null_data))
            profile.mean = float(sum(non_null_data) / len(non_null_data))
            # 简单计算中位数
            sorted_data = sorted(non_null_data)
            mid = len(sorted_data) // 2
            profile.median = float(sorted_data[mid] if len(sorted_data) % 2 == 1 else 
                               (sorted_data[mid-1] + sorted_data[mid]) / 2)
            # 标准差
            variance = sum((x - profile.mean) ** 2 for x in non_null_data) / len(non_null_data)
            profile.std_dev = float(variance ** 0.5)
        elif isinstance(sample_value, bool):
            profile.data_type = DataType.BOOLEAN
        elif isinstance(sample_value, str):
            # 检查是否为日期时间
            if self._is_datetime_string(sample_value):
                profile.data_type = DataType.DATETIME
            # 检查是否为类别型
            elif profile.unique_count < 0.2 * profile.count and profile.unique_count < 100:
                profile.data_type = DataType.CATEGORICAL
                # 计算前10个最常见值
                value_counts = {}
                for value in non_null_data:
                    value_counts[value] = value_counts.get(value, 0) + 1
                # 排序并获取前10个
                top_items = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                profile.top_categories = dict(top_items)
            else:
                profile.data_type = DataType.TEXT
                # 文本型统计
                text_lengths = [len(str(x)) for x in non_null_data]
                profile.avg_length = float(sum(text_lengths) / len(text_lengths))
                profile.max_length = max(text_lengths)
        elif isinstance(sample_value, (list, tuple)):
            profile.data_type = DataType.ARRAY
        elif isinstance(sample_value, dict):
            profile.data_type = DataType.MIXED
        else:
            profile.data_type = DataType.UNKNOWN
        
        return profile
    
    def _is_datetime_string(self, text: str) -> bool:
        """检查字符串是否为日期时间格式"""
        # 简单的日期时间格式检测
        patterns = [
            r'^\d{4}-\d{2}-\d{2}$',  # YYYY-MM-DD
            r'^\d{4}/\d{2}/\d{2}$',  # YYYY/MM/DD
            r'^\d{2}-\d{2}-\d{4}$',  # DD-MM-YYYY
            r'^\d{2}/\d{2}/\d{4}$',  # DD/MM/YYYY
            r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$',  # YYYY-MM-DD HH:MM:SS
            r'^\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}$'   # YYYY/MM/DD HH:MM:SS
        ]
        return any(re.match(pattern, text) for pattern in patterns)
    
    def create_preprocessing_pipeline(self, name: str, description: str = "") -> PreprocessingPipeline:
        """
        创建预处理流水线
        Args:
            name: 流水线名称
            description: 流水线描述
        Returns:
            预处理流水线
        """
        pipeline = PreprocessingPipeline(
            name=name,
            description=description
        )
        return pipeline
    
    def add_preprocessing_step(self, 
                             pipeline: PreprocessingPipeline,
                             step_type: PreprocessingStepType,
                             function_name: str,
                             target_columns: List[str],
                             parameters: Dict[str, Any] = None,
                             name: str = None,
                             description: str = None) -> PreprocessingStep:
        """
        添加预处理步骤
        Args:
            pipeline: 预处理流水线
            step_type: 步骤类型
            function_name: 函数名称
            target_columns: 目标列
            parameters: 参数
            name: 步骤名称
            description: 步骤描述
        Returns:
            预处理步骤
        """
        # 验证函数名称
        if function_name not in self.all_functions:
            raise ValueError(f"未知的预处理函数: {function_name}")
        
        # 设置默认参数
        if parameters is None:
            parameters = {}
        
        # 设置默认名称和描述
        if name is None:
            name = f"{step_type.value}_{function_name}"
        if description is None:
            description = f"对列 {', '.join(target_columns)} 执行 {function_name} 操作"
        
        # 创建步骤
        step = PreprocessingStep(
            name=name,
            step_type=step_type,
            description=description,
            target_columns=target_columns,
            parameters=parameters,
            function_name=function_name
        )
        
        # 添加到流水线
        pipeline.add_step(step)
        return step
    
    def execute_pipeline(self, pipeline: PreprocessingPipeline, data: Any) -> Tuple[Any, PreprocessingResult]:
        """
        执行预处理流水线
        Args:
            pipeline: 预处理流水线
            data: 输入数据
        Returns:
            处理后的数据和预处理结果
        """
        try:
            self.logger.info(f"开始执行预处理流水线: {pipeline.name}")
            start_time = datetime.datetime.now()
            
            # 初始化结果
            result = PreprocessingResult(
                pipeline_id=pipeline.pipeline_id
            )
            
            # 分析输入数据
            input_profile = self.analyze_data(data)
            result.input_rows = input_profile.row_count
            result.input_columns = input_profile.column_count
            result.input_quality_score = input_profile.quality_score
            
            # 检查数据类型
            if HAS_PANDAS and isinstance(data, pd.DataFrame):
                processed_data = data.copy()
            elif isinstance(data, list) and data and isinstance(data[0], dict):
                if HAS_PANDAS:
                    processed_data = pd.DataFrame(data).copy()
                else:
                    processed_data = [item.copy() for item in data]
            else:
                raise ValueError("不支持的数据格式，请提供DataFrame或字典列表")
            
            # 执行每个步骤
            for step in pipeline.steps:
                if not step.is_enabled:
                    continue
                
                step_start_time = datetime.datetime.now()
                try:
                    # 获取处理函数
                    process_func = self.all_functions.get(step.function_name)
                    if process_func is None:
                        raise ValueError(f"未知的预处理函数: {step.function_name}")
                    
                    # 执行处理
                    processed_data, step_stats = process_func(
                        processed_data, step.target_columns, step.parameters
                    )
                    
                    # 更新步骤状态
                    step.is_executed = True
                    step.execution_time = (datetime.datetime.now() - step_start_time).total_seconds()
                    step.affected_rows = step_stats.get('affected_rows', 0)
                    step.affected_columns = step_stats.get('affected_columns', 0)
                    
                    # 记录步骤结果
                    result.step_results.append({
                        'step_id': step.step_id,
                        'name': step.name,
                        'success': True,
                        'execution_time': step.execution_time,
                        'affected_rows': step.affected_rows,
                        'affected_columns': step.affected_columns,
                        'details': step_stats
                    })
                except Exception as e:
                    error_msg = f"步骤 '{step.name}' 执行失败: {str(e)}"
                    self.logger.error(error_msg)
                    result.errors.append(error_msg)
                    
                    # 记录失败的步骤结果
                    result.step_results.append({
                        'step_id': step.step_id,
                        'name': step.name,
                        'success': False,
                        'error': str(e)
                    })
                    
                    # 如果有错误，标记整个流水线为失败
                    result.success = False
            
            # 分析输出数据
            try:
                output_profile = self.analyze_data(processed_data)
                result.output_rows = output_profile.row_count
                result.output_columns = output_profile.column_count
                result.output_quality_score = output_profile.quality_score
            except Exception as e:
                self.logger.error(f"分析输出数据失败: {str(e)}")
                # 使用输入数据的值作为回退
                result.output_rows = result.input_rows
                result.output_columns = result.input_columns
                result.output_quality_score = result.input_quality_score
            
            # 更新流水线状态
            pipeline.is_executed = True
            pipeline.total_execution_time = (datetime.datetime.now() - start_time).total_seconds()
            result.execution_time = pipeline.total_execution_time
            
            self.logger.info(f"预处理流水线执行完成，耗时: {result.execution_time:.2f}秒")
            return processed_data, result
        
        except Exception as e:
            self.logger.error(f"执行预处理流水线失败: {str(e)}")
            # 创建失败结果
            result = PreprocessingResult(
                pipeline_id=pipeline.pipeline_id,
                success=False,
                errors=[str(e)],
                input_rows=result.input_rows if 'result' in locals() else 0,
                input_columns=result.input_columns if 'result' in locals() else 0,
                input_quality_score=result.input_quality_score if 'result' in locals() else 0,
                output_rows=result.input_rows if 'result' in locals() else 0,
                output_columns=result.input_columns if 'result' in locals() else 0,
                output_quality_score=result.input_quality_score if 'result' in locals() else 0,
                execution_time=(datetime.datetime.now() - start_time).total_seconds() if 'start_time' in locals() else 0
            )
            return data, result
    
    def get_available_preprocessing_functions(self, step_type: Optional[PreprocessingStepType] = None) -> Dict[str, str]:
        """
        获取可用的预处理函数
        Args:
            step_type: 步骤类型（可选）
        Returns:
            函数名称和描述的字典
        """
        function_descriptions = {
            # 清洗函数
            'remove_missing_values': "移除包含缺失值的行",
            'fill_missing_values': "填充缺失值",
            'remove_duplicates': "移除重复行",
            'remove_outliers': "移除异常值",
            'filter_by_value': "按值过滤数据",
            'clean_text': "清洗文本数据",
            # 转换函数
            'convert_type': "转换数据类型",
            'apply_function': "应用自定义函数",
            'bin_numeric_data': "数值数据分箱",
            'encode_categorical': "编码类别数据",
            'extract_datetime_features': "提取日期时间特征",
            # 标准化函数
            'standardize': "标准化（Z-score）",
            'min_max_scale': "最小-最大缩放",
            'robust_scale': "稳健缩放",
            'normalize': "归一化",
            # 特征工程函数
            'create_interaction_features': "创建交互特征",
            'polynomial_features': "多项式特征",
            'aggregate_features': "聚合特征",
            'extract_text_features': "提取文本特征"
        }
        
        if step_type is None:
            return function_descriptions
        
        # 根据步骤类型筛选函数
        if step_type == PreprocessingStepType.CLEANING:
            return {k: v for k, v in function_descriptions.items() if k in self.cleaning_functions}
        elif step_type == PreprocessingStepType.TRANSFORMATION:
            return {k: v for k, v in function_descriptions.items() if k in self.transformation_functions}
        elif step_type == PreprocessingStepType.NORMALIZATION:
            return {k: v for k, v in function_descriptions.items() if k in self.normalization_functions}
        elif step_type == PreprocessingStepType.FEATURE_ENGINEERING:
            return {k: v for k, v in function_descriptions.items() if k in self.feature_engineering_functions}
        else:
            return {}
    
    def get_function_parameters(self, function_name: str) -> Dict[str, Dict[str, Any]]:
        """
        获取函数参数信息
        Args:
            function_name: 函数名称
        Returns:
            参数信息字典
        """
        # 参数信息字典
        parameter_info = {
            # 清洗函数参数
            'remove_missing_values': {
                'threshold': {'type': 'float', 'default': 0.0, 'description': '缺失值比例阈值，超过此阈值的行将被移除'}
            },
            'fill_missing_values': {
                'method': {'type': 'str', 'default': 'mean', 'options': ['mean', 'median', 'mode', 'constant'], 'description': '填充方法'},
                'value': {'type': 'any', 'default': None, 'description': '使用constant方法时的填充值'}
            },
            'remove_duplicates': {
                'keep': {'type': 'str', 'default': 'first', 'options': ['first', 'last', 'False'], 'description': '保留重复项的哪一个'}
            },
            'remove_outliers': {
                'method': {'type': 'str', 'default': 'zscore', 'options': ['zscore', 'iqr'], 'description': '异常值检测方法'},
                'threshold': {'type': 'float', 'default': 3.0, 'description': 'Z-score阈值或IQR倍数'}
            },
            'filter_by_value': {
                'condition': {'type': 'str', 'default': '>', 'options': ['>', '>=', '<', '<=', '==', '!='], 'description': '过滤条件'},
                'value': {'type': 'any', 'required': True, 'description': '过滤值'}
            },
            'clean_text': {
                'lower': {'type': 'bool', 'default': True, 'description': '转换为小写'},
                'remove_punctuation': {'type': 'bool', 'default': True, 'description': '移除标点符号'},
                'remove_digits': {'type': 'bool', 'default': False, 'description': '移除数字'},
                'remove_whitespace': {'type': 'bool', 'default': True, 'description': '移除多余空白'}
            },
            # 转换函数参数
            'convert_type': {
                'target_type': {'type': 'str', 'required': True, 'options': ['int', 'float', 'str', 'bool', 'datetime'], 'description': '目标数据类型'},
                'errors': {'type': 'str', 'default': 'coerce', 'options': ['raise', 'ignore', 'coerce'], 'description': '错误处理方式'}
            },
            'apply_function': {
                'function': {'type': 'str', 'required': True, 'description': '要应用的函数表达式，如 "x + 1" 或 "x.lower()"'}
            },
            'bin_numeric_data': {
                'bins': {'type': 'int', 'default': 5, 'description': '分箱数量'},
                'strategy': {'type': 'str', 'default': 'uniform', 'options': ['uniform', 'quantile', 'kmeans'], 'description': '分箱策略'},
                'labels': {'type': 'list', 'default': None, 'description': '分箱标签，默认为None使用区间'}
            },
            'encode_categorical': {
                'method': {'type': 'str', 'default': 'onehot', 'options': ['onehot', 'label', 'ordinal', 'binary'], 'description': '编码方法'},
                'drop_first': {'type': 'bool', 'default': False, 'description': '是否删除第一个类别（用于One-Hot编码）'}
            },
            'extract_datetime_features': {
                'features': {'type': 'list', 'default': ['year', 'month', 'day'], 'options': ['year', 'month', 'day', 'hour', 'minute', 'second', 'dayofweek', 'quarter'], 'description': '要提取的特征'}
            },
            # 标准化函数参数
            'standardize': {
                'with_mean': {'type': 'bool', 'default': True, 'description': '是否减去均值'},
                'with_std': {'type': 'bool', 'default': True, 'description': '是否除以标准差'}
            },
            'min_max_scale': {
                'feature_range': {'type': 'tuple', 'default': (0, 1), 'description': '缩放范围'}
            },
            'robust_scale': {
                'quantile_range': {'type': 'tuple', 'default': (25.0, 75.0), 'description': '用于计算缩放的分位数范围'}
            },
            'normalize': {
                'norm': {'type': 'str', 'default': 'l2', 'options': ['l1', 'l2', 'max'], 'description': '归一化范数'}
            },
            # 特征工程函数参数
            'create_interaction_features': {
                'interaction_type': {'type': 'str', 'default': 'multiplication', 'options': ['multiplication', 'addition', 'subtraction', 'division'], 'description': '交互类型'}
            },
            'polynomial_features': {
                'degree': {'type': 'int', 'default': 2, 'description': '多项式阶数'},
                'interaction_only': {'type': 'bool', 'default': False, 'description': '是否只包括交互项'}
            },
            'aggregate_features': {
                'groupby_columns': {'type': 'list', 'required': True, 'description': '分组列'},
                'agg_functions': {'type': 'dict', 'required': True, 'description': '聚合函数，如 {"column1": "mean", "column2": "sum"}'}
            },
            'extract_text_features': {
                'features': {'type': 'list', 'default': ['length', 'word_count'], 'options': ['length', 'word_count', 'char_count', 'sentence_count'], 'description': '要提取的特征'}
            }
        }
        
        if function_name in parameter_info:
            return parameter_info[function_name]
        else:
            return {}
    # 数据清洗函数
    def _remove_missing_values(self, data: Any, target_columns: List[str], parameters: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """移除包含缺失值的行"""
        threshold = parameters.get('threshold', 0.0)
        stats = {'affected_rows': 0, 'affected_columns': len(target_columns)}
        
        if HAS_PANDAS and isinstance(data, pd.DataFrame):
            # 计算每行在目标列中的缺失值比例
            if not target_columns:  # 如果未指定列，使用所有列
                target_columns = data.columns.tolist()
            
            # 计算每行在目标列中的缺失值比例
            missing_ratio = data[target_columns].isna().mean(axis=1)
            
            # 筛选出缺失值比例小于等于阈值的行
            original_count = len(data)
            data = data[missing_ratio <= threshold]
            stats['affected_rows'] = original_count - len(data)
        
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            # 处理字典列表
            if not target_columns:  # 如果未指定列，使用第一个字典的所有键
                target_columns = list(data[0].keys())
            
            original_count = len(data)
            filtered_data = []
            
            for item in data:
                # 计算目标列中的缺失值比例
                missing_count = sum(1 for col in target_columns if col in item and item[col] is None)
                missing_ratio = missing_count / len(target_columns) if target_columns else 0
                
                if missing_ratio <= threshold:
                    filtered_data.append(item)
            
            data = filtered_data
            stats['affected_rows'] = original_count - len(data)
        
        return data, stats
    
    def _fill_missing_values(self, data: Any, target_columns: List[str], parameters: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """填充缺失值"""
        method = parameters.get('method', 'mean')
        fill_value = parameters.get('value', None)
        stats = {'affected_rows': 0, 'affected_columns': len(target_columns), 'fill_values': {}}
        
        if HAS_PANDAS and isinstance(data, pd.DataFrame):
            # 如果未指定列，使用所有列
            if not target_columns:
                target_columns = data.columns.tolist()
            
            # 记录原始缺失值数量
            original_missing = data[target_columns].isna().sum().sum()
            
            # 对每列应用填充方法
            for col in target_columns:
                if method == 'mean' and pd.api.types.is_numeric_dtype(data[col]):
                    fill_val = data[col].mean()
                    data[col] = data[col].fillna(fill_val)
                    stats['fill_values'][col] = float(fill_val)
                elif method == 'median' and pd.api.types.is_numeric_dtype(data[col]):
                    fill_val = data[col].median()
                    data[col] = data[col].fillna(fill_val)
                    stats['fill_values'][col] = float(fill_val)
                elif method == 'mode':
                    fill_val = data[col].mode().iloc[0] if not data[col].mode().empty else None
                    data[col] = data[col].fillna(fill_val)
                    stats['fill_values'][col] = fill_val
                elif method == 'constant':
                    data[col] = data[col].fillna(fill_value)
                    stats['fill_values'][col] = fill_value
            
            # 计算受影响的行数
            stats['affected_rows'] = data[target_columns].notna().sum().sum() - (len(data) * len(target_columns) - original_missing)
        
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            # 处理字典列表
            if not target_columns:
                # 收集所有可能的键
                target_columns = set()
                for item in data:
                    target_columns.update(item.keys())
                target_columns = list(target_columns)
            
            # 计算填充值
            fill_values = {}
            for col in target_columns:
                if method == 'mean' or method == 'median':
                    # 收集数值
                    values = [item[col] for item in data if col in item and item[col] is not None and isinstance(item[col], (int, float))]
                    if values:
                        if method == 'mean':
                            fill_values[col] = sum(values) / len(values)
                        else:  # median
                            values.sort()
                            mid = len(values) // 2
                            fill_values[col] = values[mid] if len(values) % 2 == 1 else (values[mid-1] + values[mid]) / 2
                elif method == 'mode':
                    # 计算最常见的值
                    value_counts = {}
                    for item in data:
                        if col in item and item[col] is not None:
                            value = item[col]
                            value_counts[value] = value_counts.get(value, 0) + 1
                    if value_counts:
                        fill_values[col] = max(value_counts.items(), key=lambda x: x[1])[0]
                elif method == 'constant':
                    fill_values[col] = fill_value
            
            # 应用填充
            affected_rows = 0
            for item in data:
                for col in target_columns:
                    if col in item and item[col] is None and col in fill_values:
                        item[col] = fill_values[col]
                        affected_rows += 1
            
            stats['affected_rows'] = affected_rows
            stats['fill_values'] = fill_values
        
        return data, stats    d
ef _remove_duplicates(self, data: Any, target_columns: List[str], parameters: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """移除重复行"""
        keep = parameters.get('keep', 'first')
        stats = {'affected_rows': 0, 'affected_columns': len(target_columns)}
        
        if HAS_PANDAS and isinstance(data, pd.DataFrame):
            # 如果未指定列，使用所有列
            if not target_columns:
                target_columns = data.columns.tolist()
            
            # 记录原始行数
            original_count = len(data)
            
            # 移除重复行
            data = data.drop_duplicates(subset=target_columns, keep=keep)
            
            # 计算受影响的行数
            stats['affected_rows'] = original_count - len(data)
        
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            # 处理字典列表
            if not target_columns:
                # 使用所有键
                target_columns = list(data[0].keys())
            
            original_count = len(data)
            seen = set()
            unique_data = []
            
            # 根据keep参数决定处理方式
            if keep == 'first':
                for item in data:
                    # 创建键值对的元组作为唯一标识
                    key = tuple((col, item.get(col)) for col in target_columns if col in item)
                    if key not in seen:
                        seen.add(key)
                        unique_data.append(item)
                data = unique_data
            elif keep == 'last':
                # 反向处理
                for item in reversed(data):
                    key = tuple((col, item.get(col)) for col in target_columns if col in item)
                    if key not in seen:
                        seen.add(key)
                        unique_data.append(item)
                data = list(reversed(unique_data))
            else:  # keep=False，保留所有不重复的行
                unique_items = {}
                for i, item in enumerate(data):
                    key = tuple((col, item.get(col)) for col in target_columns if col in item)
                    if key in unique_items:
                        unique_items[key].append(i)
                    else:
                        unique_items[key] = [i]
                
                # 只保留没有重复的项
                unique_data = [item for i, item in enumerate(data) 
                              if len(unique_items[tuple((col, item.get(col)) for col in target_columns if col in item)]) == 1]
                data = unique_data
            
            stats['affected_rows'] = original_count - len(data)
        
        return data, stats
    
    def _remove_outliers(self, data: Any, target_columns: List[str], parameters: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """移除异常值"""
        method = parameters.get('method', 'zscore')
        threshold = parameters.get('threshold', 3.0)
        stats = {'affected_rows': 0, 'affected_columns': len(target_columns), 'outliers_per_column': {}}
        
        if HAS_PANDAS and isinstance(data, pd.DataFrame):
            # 如果未指定列，使用所有数值列
            if not target_columns:
                target_columns = data.select_dtypes(include=['number']).columns.tolist()
            
            # 记录原始行数
            original_count = len(data)
            mask = pd.Series(True, index=data.index)
            
            for col in target_columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    continue
                
                if method == 'zscore':
                    # Z-score方法
                    mean = data[col].mean()
                    std = data[col].std()
                    if std == 0:  # 避免除以零
                        continue
                    
                    z_scores = (data[col] - mean) / std
                    col_mask = z_scores.abs() <= threshold
                    outliers_count = (~col_mask).sum()
                    stats['outliers_per_column'][col] = int(outliers_count)
                    mask = mask & col_mask
                
                elif method == 'iqr':
                    # IQR方法
                    q1 = data[col].quantile(0.25)
                    q3 = data[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - threshold * iqr
                    upper_bound = q3 + threshold * iqr
                    
                    col_mask = (data[col] >= lower_bound) & (data[col] <= upper_bound)
                    outliers_count = (~col_mask).sum()
                    stats['outliers_per_column'][col] = int(outliers_count)
                    mask = mask & col_mask
            
            # 应用掩码
            data = data[mask]
            
            # 计算受影响的行数
            stats['affected_rows'] = original_count - len(data)
        
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            # 处理字典列表
            if not target_columns:
                # 尝试找出数值列
                target_columns = []
                for key, value in data[0].items():
                    if isinstance(value, (int, float)):
                        target_columns.append(key)
            
            original_count = len(data)
            
            # 计算每列的统计量
            stats_by_col = {}
            for col in target_columns:
                values = [item[col] for item in data if col in item and item[col] is not None and isinstance(item[col], (int, float))]
                if not values:
                    continue
                
                if method == 'zscore':
                    # 计算均值和标准差
                    mean = sum(values) / len(values)
                    variance = sum((x - mean) ** 2 for x in values) / len(values)
                    std = variance ** 0.5
                    if std == 0:  # 避免除以零
                        continue
                    stats_by_col[col] = {'mean': mean, 'std': std, 'method': 'zscore'}
                
                elif method == 'iqr':
                    # 计算四分位数
                    values.sort()
                    n = len(values)
                    q1_idx = int(n * 0.25)
                    q3_idx = int(n * 0.75)
                    q1 = values[q1_idx]
                    q3 = values[q3_idx]
                    iqr = q3 - q1
                    stats_by_col[col] = {'q1': q1, 'q3': q3, 'iqr': iqr, 'method': 'iqr'}
            
            # 筛选非异常值
            filtered_data = []
            outliers_per_column = {col: 0 for col in target_columns}
            
            for item in data:
                is_outlier = False
                for col in target_columns:
                    if col not in item or item[col] is None or not isinstance(item[col], (int, float)) or col not in stats_by_col:
                        continue
                    
                    value = item[col]
                    col_stats = stats_by_col[col]
                    
                    if col_stats['method'] == 'zscore':
                        z_score = abs((value - col_stats['mean']) / col_stats['std'])
                        if z_score > threshold:
                            is_outlier = True
                            outliers_per_column[col] += 1
                            break
                    
                    elif col_stats['method'] == 'iqr':
                        lower_bound = col_stats['q1'] - threshold * col_stats['iqr']
                        upper_bound = col_stats['q3'] + threshold * col_stats['iqr']
                        if value < lower_bound or value > upper_bound:
                            is_outlier = True
                            outliers_per_column[col] += 1
                            break
                
                if not is_outlier:
                    filtered_data.append(item)
            
            data = filtered_data
            stats['affected_rows'] = original_count - len(data)
            stats['outliers_per_column'] = outliers_per_column
        
        return data, stats 
   def _filter_by_value(self, data: Any, target_columns: List[str], parameters: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """按值过滤数据"""
        condition = parameters.get('condition', '>')
        value = parameters.get('value')
        stats = {'affected_rows': 0, 'affected_columns': len(target_columns)}
        
        if value is None:
            return data, stats
        
        if HAS_PANDAS and isinstance(data, pd.DataFrame):
            # 如果未指定列，使用第一列
            if not target_columns:
                if data.columns.empty:
                    return data, stats
                target_columns = [data.columns[0]]
            
            # 记录原始行数
            original_count = len(data)
            
            # 对每列应用过滤条件
            mask = pd.Series(False, index=data.index)
            for col in target_columns:
                if condition == '>':
                    col_mask = data[col] > value
                elif condition == '>=':
                    col_mask = data[col] >= value
                elif condition == '<':
                    col_mask = data[col] < value
                elif condition == '<=':
                    col_mask = data[col] <= value
                elif condition == '==':
                    col_mask = data[col] == value
                elif condition == '!=':
                    col_mask = data[col] != value
                else:
                    raise ValueError(f"不支持的条件: {condition}")
                
                mask = mask | col_mask
            
            # 应用过滤
            data = data[mask]
            
            # 计算受影响的行数
            stats['affected_rows'] = original_count - len(data)
        
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            # 处理字典列表
            if not target_columns:
                if not data[0]:
                    return data, stats
                target_columns = [next(iter(data[0].keys()))]
            
            original_count = len(data)
            filtered_data = []
            
            for item in data:
                keep_item = False
                for col in target_columns:
                    if col not in item or item[col] is None:
                        continue
                    
                    item_value = item[col]
                    
                    if condition == '>' and item_value > value:
                        keep_item = True
                    elif condition == '>=' and item_value >= value:
                        keep_item = True
                    elif condition == '<' and item_value < value:
                        keep_item = True
                    elif condition == '<=' and item_value <= value:
                        keep_item = True
                    elif condition == '==' and item_value == value:
                        keep_item = True
                    elif condition == '!=' and item_value != value:
                        keep_item = True
                
                if keep_item:
                    filtered_data.append(item)
            
            data = filtered_data
            stats['affected_rows'] = original_count - len(data)
        
        return data, stats
    
    def _clean_text(self, data: Any, target_columns: List[str], parameters: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """清洗文本数据"""
        lower = parameters.get('lower', True)
        remove_punctuation = parameters.get('remove_punctuation', True)
        remove_digits = parameters.get('remove_digits', False)
        remove_whitespace = parameters.get('remove_whitespace', True)
        stats = {'affected_rows': 0, 'affected_columns': len(target_columns)}
        
        # 创建清洗函数
        def clean_text_value(text):
            if not isinstance(text, str):
                return text
            
            if lower:
                text = text.lower()
            
            if remove_punctuation:
                text = re.sub(r'[^\w\s]', '', text)
            
            if remove_digits:
                text = re.sub(r'\d+', '', text)
            
            if remove_whitespace:
                text = re.sub(r'\s+', ' ', text).strip()
            
            return text
        
        if HAS_PANDAS and isinstance(data, pd.DataFrame):
            # 如果未指定列，使用所有文本列
            if not target_columns:
                target_columns = data.select_dtypes(include=['object']).columns.tolist()
            
            # 记录原始数据
            affected_rows = 0
            
            # 对每列应用清洗函数
            for col in target_columns:
                if not pd.api.types.is_string_dtype(data[col]):
                    continue
                
                # 应用清洗函数
                original_values = data[col].copy()
                data[col] = data[col].apply(clean_text_value)
                
                # 计算受影响的行数
                affected_rows += (original_values != data[col]).sum()
            
            stats['affected_rows'] = affected_rows
        
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            # 处理字典列表
            if not target_columns:
                # 尝试找出文本列
                target_columns = []
                for key, value in data[0].items():
                    if isinstance(value, str):
                        target_columns.append(key)
            
            affected_rows = 0
            
            # 对每个项目的每个目标列应用清洗函数
            for item in data:
                for col in target_columns:
                    if col in item and isinstance(item[col], str):
                        original_value = item[col]
                        item[col] = clean_text_value(item[col])
                        if original_value != item[col]:
                            affected_rows += 1
            
            stats['affected_rows'] = affected_rows
        
        return data, stats   
 # 转换函数
    def _convert_type(self, data: Any, target_columns: List[str], parameters: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """转换数据类型"""
        target_type = parameters.get('target_type', 'float')
        errors = parameters.get('errors', 'coerce')
        stats = {'affected_rows': 0, 'affected_columns': len(target_columns), 'conversion_failures': {}}
        
        if HAS_PANDAS and isinstance(data, pd.DataFrame):
            # 如果未指定列，使用所有列
            if not target_columns:
                target_columns = data.columns.tolist()
            
            # 记录原始数据
            affected_rows = 0
            conversion_failures = {}
            
            # 对每列应用类型转换
            for col in target_columns:
                try:
                    if target_type == 'int':
                        original_values = data[col].copy()
                        data[col] = pd.to_numeric(data[col], errors=errors).astype('Int64')  # 使用可空整数类型
                        affected_rows += (original_values != data[col]).sum()
                        conversion_failures[col] = data[col].isna().sum() - original_values.isna().sum()
                    
                    elif target_type == 'float':
                        original_values = data[col].copy()
                        data[col] = pd.to_numeric(data[col], errors=errors)
                        affected_rows += (original_values != data[col]).sum()
                        conversion_failures[col] = data[col].isna().sum() - original_values.isna().sum()
                    
                    elif target_type == 'str':
                        original_values = data[col].copy()
                        data[col] = data[col].astype(str)
                        affected_rows += (original_values != data[col]).sum()
                        conversion_failures[col] = 0
                    
                    elif target_type == 'bool':
                        original_values = data[col].copy()
                        data[col] = data[col].astype(bool)
                        affected_rows += (original_values != data[col]).sum()
                        conversion_failures[col] = 0
                    
                    elif target_type == 'datetime':
                        original_values = data[col].copy()
                        data[col] = pd.to_datetime(data[col], errors=errors)
                        affected_rows += (original_values != data[col]).sum()
                        conversion_failures[col] = data[col].isna().sum() - original_values.isna().sum()
                    
                    else:
                        raise ValueError(f"不支持的目标类型: {target_type}")
                
                except Exception as e:
                    self.logger.warning(f"列 {col} 转换为 {target_type} 类型失败: {str(e)}")
                    conversion_failures[col] = len(data)
            
            stats['affected_rows'] = affected_rows
            stats['conversion_failures'] = conversion_failures
        
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            # 处理字典列表
            if not target_columns:
                target_columns = list(data[0].keys())
            
            affected_rows = 0
            conversion_failures = {col: 0 for col in target_columns}
            
            # 对每个项目的每个目标列应用类型转换
            for item in data:
                for col in target_columns:
                    if col not in item or item[col] is None:
                        continue
                    
                    original_value = item[col]
                    try:
                        if target_type == 'int':
                            item[col] = int(float(item[col]))
                        elif target_type == 'float':
                            item[col] = float(item[col])
                        elif target_type == 'str':
                            item[col] = str(item[col])
                        elif target_type == 'bool':
                            if isinstance(item[col], str):
                                item[col] = item[col].lower() in ('true', 'yes', '1', 't', 'y')
                            else:
                                item[col] = bool(item[col])
                        elif target_type == 'datetime':
                            if isinstance(item[col], str):
                                try:
                                    from dateutil import parser
                                    item[col] = parser.parse(item[col])
                                except:
                                    if errors == 'raise':
                                        raise
                                    elif errors == 'coerce':
                                        item[col] = None
                            else:
                                if errors == 'raise':
                                    raise ValueError(f"无法将 {type(item[col])} 转换为 datetime")
                                elif errors == 'coerce':
                                    item[col] = None
                        else:
                            raise ValueError(f"不支持的目标类型: {target_type}")
                        
                        if original_value != item[col]:
                            affected_rows += 1
                    
                    except Exception as e:
                        if errors == 'raise':
                            raise
                        elif errors == 'coerce':
                            item[col] = None
                            conversion_failures[col] += 1
                        # 如果errors='ignore'，则保持原值不变
            
            stats['affected_rows'] = affected_rows
            stats['conversion_failures'] = conversion_failures
        
        return data, stats
    
    def _apply_function(self, data: Any, target_columns: List[str], parameters: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """应用自定义函数"""
        function_expr = parameters.get('function', 'x')
        stats = {'affected_rows': 0, 'affected_columns': len(target_columns), 'function_failures': {}}
        
        # 创建可执行的函数
        try:
            # 创建一个安全的局部命名空间
            safe_locals = {
                'np': np,
                'math': __import__('math'),
                're': re,
                'datetime': datetime
            }
            
            # 编译函数表达式
            func_code = f"lambda x: {function_expr}"
            func = eval(func_code, {"__builtins__": {}}, safe_locals)
        except Exception as e:
            raise ValueError(f"函数表达式编译失败: {str(e)}")
        
        if HAS_PANDAS and isinstance(data, pd.DataFrame):
            # 如果未指定列，使用所有列
            if not target_columns:
                target_columns = data.columns.tolist()
            
            # 记录原始数据
            affected_rows = 0
            function_failures = {}
            
            # 对每列应用函数
            for col in target_columns:
                try:
                    original_values = data[col].copy()
                    data[col] = data[col].apply(lambda x: func(x) if x is not None else None)
                    affected_rows += (original_values != data[col]).sum()
                    function_failures[col] = data[col].isna().sum() - original_values.isna().sum()
                except Exception as e:
                    self.logger.warning(f"对列 {col} 应用函数失败: {str(e)}")
                    function_failures[col] = len(data)
            
            stats['affected_rows'] = affected_rows
            stats['function_failures'] = function_failures
        
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            # 处理字典列表
            if not target_columns:
                target_columns = list(data[0].keys())
            
            affected_rows = 0
            function_failures = {col: 0 for col in target_columns}
            
            # 对每个项目的每个目标列应用函数
            for item in data:
                for col in target_columns:
                    if col not in item or item[col] is None:
                        continue
                    
                    original_value = item[col]
                    try:
                        item[col] = func(item[col])
                        if original_value != item[col]:
                            affected_rows += 1
                    except Exception as e:
                        function_failures[col] += 1
            
            stats['affected_rows'] = affected_rows
            stats['function_failures'] = function_failures
        
        return data, stats  
  def _bin_numeric_data(self, data: Any, target_columns: List[str], parameters: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """数值数据分箱"""
        bins = parameters.get('bins', 5)
        strategy = parameters.get('strategy', 'uniform')
        labels = parameters.get('labels', None)
        stats = {'affected_rows': 0, 'affected_columns': len(target_columns), 'bin_edges': {}}
        
        if HAS_PANDAS and isinstance(data, pd.DataFrame):
            # 如果未指定列，使用所有数值列
            if not target_columns:
                target_columns = data.select_dtypes(include=['number']).columns.tolist()
            
            # 记录原始数据
            affected_rows = 0
            bin_edges = {}
            
            # 对每列应用分箱
            for col in target_columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    continue
                
                try:
                    original_values = data[col].copy()
                    
                    if strategy == 'uniform':
                        # 等宽分箱
                        bin_result = pd.cut(data[col], bins=bins, labels=labels)
                        data[col] = bin_result
                        bin_edges[col] = list(bin_result.categories) if hasattr(bin_result, 'categories') else []
                    
                    elif strategy == 'quantile':
                        # 等频分箱
                        quantiles = np.linspace(0, 1, bins + 1)
                        edges = data[col].quantile(quantiles).unique()
                        bin_result = pd.cut(data[col], bins=edges, labels=labels)
                        data[col] = bin_result
                        bin_edges[col] = list(bin_result.categories) if hasattr(bin_result, 'categories') else []
                    
                    elif strategy == 'kmeans' and HAS_SKLEARN:
                        # K-means分箱
                        from sklearn.cluster import KMeans
                        
                        # 准备数据
                        X = data[col].values.reshape(-1, 1)
                        X_valid = X[~np.isnan(X)]
                        
                        # 应用K-means
                        kmeans = KMeans(n_clusters=bins, random_state=0).fit(X_valid)
                        centers = kmeans.cluster_centers_.flatten()
                        centers.sort()
                        
                        # 创建分箱边界
                        edges = np.concatenate([[-np.inf], (centers[:-1] + centers[1:]) / 2, [np.inf]])
                        bin_result = pd.cut(data[col], bins=edges, labels=labels)
                        data[col] = bin_result
                        bin_edges[col] = list(bin_result.categories) if hasattr(bin_result, 'categories') else []
                    
                    else:
                        raise ValueError(f"不支持的分箱策略: {strategy}")
                    
                    affected_rows += (original_values != data[col]).sum()
                
                except Exception as e:
                    self.logger.warning(f"对列 {col} 应用分箱失败: {str(e)}")
            
            stats['affected_rows'] = affected_rows
            stats['bin_edges'] = bin_edges
        
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            # 处理字典列表
            if not target_columns:
                # 尝试找出数值列
                target_columns = []
                for key, value in data[0].items():
                    if isinstance(value, (int, float)):
                        target_columns.append(key)
            
            # 如果有pandas，转换为DataFrame处理后再转回来
            if HAS_PANDAS:
                df = pd.DataFrame(data)
                df, df_stats = self._bin_numeric_data(df, target_columns, parameters)
                return df.to_dict('records'), df_stats
            
            # 如果没有pandas，简单实现等宽分箱
            affected_rows = 0
            bin_edges = {}
            
            for col in target_columns:
                # 收集数值
                values = [item[col] for item in data if col in item and item[col] is not None and isinstance(item[col], (int, float))]
                if not values:
                    continue
                
                # 计算分箱边界
                min_val = min(values)
                max_val = max(values)
                bin_width = (max_val - min_val) / bins
                edges = [min_val + i * bin_width for i in range(bins + 1)]
                bin_edges[col] = edges
                
                # 应用分箱
                for item in data:
                    if col in item and item[col] is not None and isinstance(item[col], (int, float)):
                        original_value = item[col]
                        value = item[col]
                        
                        # 找到对应的分箱
                        bin_idx = 0
                        for i, edge in enumerate(edges[1:], 1):
                            if value <= edge:
                                bin_idx = i - 1
                                break
                        
                        # 设置分箱结果
                        if labels is not None and bin_idx < len(labels):
                            item[col] = labels[bin_idx]
                        else:
                            item[col] = bin_idx
                        
                        if original_value != item[col]:
                            affected_rows += 1
            
            stats['affected_rows'] = affected_rows
            stats['bin_edges'] = bin_edges
        
        return data, stats
    
    def _encode_categorical(self, data: Any, target_columns: List[str], parameters: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """编码类别数据"""
        method = parameters.get('method', 'onehot')
        drop_first = parameters.get('drop_first', False)
        stats = {'affected_rows': 0, 'affected_columns': len(target_columns), 'new_columns': [], 'encoding_map': {}}
        
        if HAS_PANDAS and isinstance(data, pd.DataFrame):
            # 如果未指定列，使用所有类别列
            if not target_columns:
                target_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # 记录原始数据
            original_columns = set(data.columns)
            encoding_map = {}
            
            # 对每列应用编码
            for col in target_columns:
                try:
                    if method == 'onehot':
                        # One-hot编码
                        dummies = pd.get_dummies(data[col], prefix=col, drop_first=drop_first)
                        data = pd.concat([data, dummies], axis=1)
                        # 记录编码映射
                        encoding_map[col] = {cat: i for i, cat in enumerate(data[col].unique())}
                    
                    elif method == 'label':
                        # 标签编码
                        if HAS_SKLEARN:
                            from sklearn.preprocessing import LabelEncoder
                            le = LabelEncoder()
                            data[col] = le.fit_transform(data[col].astype(str))
                            # 记录编码映射
                            encoding_map[col] = {cat: i for i, cat in enumerate(le.classes_)}
                        else:
                            # 简单实现
                            categories = data[col].unique()
                            cat_map = {cat: i for i, cat in enumerate(categories)}
                            data[col] = data[col].map(cat_map)
                            # 记录编码映射
                            encoding_map[col] = cat_map
                    
                    elif method == 'ordinal':
                        # 序数编码（假设类别已经有序）
                        categories = sorted(data[col].unique())
                        cat_map = {cat: i for i, cat in enumerate(categories)}
                        data[col] = data[col].map(cat_map)
                        # 记录编码映射
                        encoding_map[col] = cat_map
                    
                    elif method == 'binary':
                        # 二进制编码
                        categories = data[col].unique()
                        n_categories = len(categories)
                        n_bits = int(np.ceil(np.log2(n_categories)))
                        
                        # 创建二进制编码映射
                        cat_map = {}
                        for i, cat in enumerate(categories):
                            binary = format(i, f'0{n_bits}b')
                            cat_map[cat] = binary
                        
                        # 应用编码
                        for bit in range(n_bits):
                            col_name = f"{col}_bit{bit}"
                            data[col_name] = data[col].map(lambda x: int(cat_map[x][bit]))
                        
                        # 记录编码映射
                        encoding_map[col] = cat_map
                    
                    else:
                        raise ValueError(f"不支持的编码方法: {method}")
                
                except Exception as e:
                    self.logger.warning(f"对列 {col} 应用编码失败: {str(e)}")
            
            # 计算新增的列
            new_columns = list(set(data.columns) - original_columns)
            stats['new_columns'] = new_columns
            stats['encoding_map'] = encoding_map
            stats['affected_rows'] = len(data)
        
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            # 如果有pandas，转换为DataFrame处理后再转回来
            if HAS_PANDAS:
                df = pd.DataFrame(data)
                df, df_stats = self._encode_categorical(df, target_columns, parameters)
                return df.to_dict('records'), df_stats
            
            # 如果没有pandas，简单实现标签编码
            if not target_columns:
                # 尝试找出类别列
                target_columns = []
                for key, value in data[0].items():
                    if isinstance(value, str) and not key.startswith('_'):
                        target_columns.append(key)
            
            encoding_map = {}
            
            for col in target_columns:
                # 收集唯一值
                categories = set()
                for item in data:
                    if col in item and item[col] is not None:
                        categories.add(item[col])
                
                categories = sorted(categories)
                
                if method == 'label' or method == 'ordinal':
                    # 标签编码
                    cat_map = {cat: i for i, cat in enumerate(categories)}
                    
                    # 应用编码
                    for item in data:
                        if col in item and item[col] is not None:
                            item[col] = cat_map[item[col]]
                    
                    # 记录编码映射
                    encoding_map[col] = cat_map
                
                elif method == 'onehot':
                    # One-hot编码
                    for item in data:
                        if col in item and item[col] is not None:
                            category = item[col]
                            # 删除原始列
                            del item[col]
                            # 添加one-hot列
                            for cat in categories:
                                if drop_first and cat == categories[0]:
                                    continue
                                item[f"{col}_{cat}"] = 1 if category == cat else 0
                    
                    # 记录新列
                    new_columns = [f"{col}_{cat}" for cat in categories if not (drop_first and cat == categories[0])]
                    stats['new_columns'].extend(new_columns)
                    
                    # 记录编码映射
                    encoding_map[col] = {cat: i for i, cat in enumerate(categories)}
                
                else:
                    self.logger.warning(f"对于字典列表，不支持的编码方法: {method}")
            
            stats['encoding_map'] = encoding_map
            stats['affected_rows'] = len(data)
        
        return data, stats    d
ef _extract_datetime_features(self, data: Any, target_columns: List[str], parameters: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """提取日期时间特征"""
        features = parameters.get('features', ['year', 'month', 'day'])
        stats = {'affected_rows': 0, 'affected_columns': len(target_columns), 'new_columns': []}
        
        if HAS_PANDAS and isinstance(data, pd.DataFrame):
            # 如果未指定列，使用所有日期时间列
            if not target_columns:
                target_columns = data.select_dtypes(include=['datetime']).columns.tolist()
                if not target_columns:
                    # 尝试转换字符串列为日期时间
                    for col in data.select_dtypes(include=['object']).columns:
                        try:
                            data[col] = pd.to_datetime(data[col], errors='coerce')
                            if not data[col].isna().all():
                                target_columns.append(col)
                        except:
                            pass
            
            # 记录原始数据
            original_columns = set(data.columns)
            new_columns = []
            
            # 对每列提取特征
            for col in target_columns:
                try:
                    # 确保列是日期时间类型
                    if not pd.api.types.is_datetime64_dtype(data[col]):
                        data[col] = pd.to_datetime(data[col], errors='coerce')
                    
                    # 提取特征
                    for feature in features:
                        feature_col = f"{col}_{feature}"
                        
                        if feature == 'year':
                            data[feature_col] = data[col].dt.year
                        elif feature == 'month':
                            data[feature_col] = data[col].dt.month
                        elif feature == 'day':
                            data[feature_col] = data[col].dt.day
                        elif feature == 'hour':
                            data[feature_col] = data[col].dt.hour
                        elif feature == 'minute':
                            data[feature_col] = data[col].dt.minute
                        elif feature == 'second':
                            data[feature_col] = data[col].dt.second
                        elif feature == 'dayofweek':
                            data[feature_col] = data[col].dt.dayofweek
                        elif feature == 'quarter':
                            data[feature_col] = data[col].dt.quarter
                        else:
                            self.logger.warning(f"不支持的日期时间特征: {feature}")
                            continue
                        
                        new_columns.append(feature_col)
                
                except Exception as e:
                    self.logger.warning(f"对列 {col} 提取日期时间特征失败: {str(e)}")
            
            # 计算新增的列
            stats['new_columns'] = new_columns
            stats['affected_rows'] = len(data)
        
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            # 如果有pandas，转换为DataFrame处理后再转回来
            if HAS_PANDAS:
                df = pd.DataFrame(data)
                df, df_stats = self._extract_datetime_features(df, target_columns, parameters)
                return df.to_dict('records'), df_stats
            
            # 如果没有pandas，简单实现日期时间特征提取
            if not target_columns:
                # 尝试找出日期时间列
                target_columns = []
                for key, value in data[0].items():
                    if isinstance(value, (datetime.datetime, datetime.date)) or (isinstance(value, str) and self._is_datetime_string(value)):
                        target_columns.append(key)
            
            new_columns = []
            
            for col in target_columns:
                # 提取特征
                for feature in features:
                    feature_col = f"{col}_{feature}"
                    new_columns.append(feature_col)
                    
                    for item in data:
                        if col not in item or item[col] is None:
                            item[feature_col] = None
                            continue
                        
                        # 转换为日期时间对象
                        dt_value = item[col]
                        if isinstance(dt_value, str):
                            try:
                                from dateutil import parser
                                dt_value = parser.parse(dt_value)
                            except:
                                item[feature_col] = None
                                continue
                        
                        # 提取特征
                        if feature == 'year':
                            item[feature_col] = dt_value.year
                        elif feature == 'month':
                            item[feature_col] = dt_value.month
                        elif feature == 'day':
                            item[feature_col] = dt_value.day
                        elif feature == 'hour':
                            item[feature_col] = dt_value.hour if hasattr(dt_value, 'hour') else 0
                        elif feature == 'minute':
                            item[feature_col] = dt_value.minute if hasattr(dt_value, 'minute') else 0
                        elif feature == 'second':
                            item[feature_col] = dt_value.second if hasattr(dt_value, 'second') else 0
                        elif feature == 'dayofweek':
                            item[feature_col] = dt_value.weekday()
                        elif feature == 'quarter':
                            item[feature_col] = (dt_value.month - 1) // 3 + 1
                        else:
                            item[feature_col] = None
            
            stats['new_columns'] = new_columns
            stats['affected_rows'] = len(data)
        
        return data, stats
    
    # 标准化函数
    def _standardize(self, data: Any, target_columns: List[str], parameters: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """标准化（Z-score）"""
        with_mean = parameters.get('with_mean', True)
        with_std = parameters.get('with_std', True)
        stats = {'affected_rows': 0, 'affected_columns': len(target_columns), 'means': {}, 'stds': {}}
        
        if HAS_PANDAS and isinstance(data, pd.DataFrame):
            # 如果未指定列，使用所有数值列
            if not target_columns:
                target_columns = data.select_dtypes(include=['number']).columns.tolist()
            
            # 记录原始数据
            means = {}
            stds = {}
            
            # 对每列应用标准化
            for col in target_columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    continue
                
                try:
                    # 计算均值和标准差
                    mean = data[col].mean() if with_mean else 0
                    std = data[col].std() if with_std else 1
                    
                    if std == 0:  # 避免除以零
                        std = 1
                    
                    # 应用标准化
                    data[col] = (data[col] - mean) / std
                    
                    # 记录统计量
                    means[col] = float(mean)
                    stds[col] = float(std)
                
                except Exception as e:
                    self.logger.warning(f"对列 {col} 应用标准化失败: {str(e)}")
            
            stats['means'] = means
            stats['stds'] = stds
            stats['affected_rows'] = len(data)
        
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            # 处理字典列表
            if not target_columns:
                # 尝试找出数值列
                target_columns = []
                for key, value in data[0].items():
                    if isinstance(value, (int, float)):
                        target_columns.append(key)
            
            # 计算均值和标准差
            means = {}
            stds = {}
            
            for col in target_columns:
                values = [item[col] for item in data if col in item and item[col] is not None and isinstance(item[col], (int, float))]
                if not values:
                    continue
                
                mean = sum(values) / len(values) if with_mean else 0
                
                if with_std:
                    variance = sum((x - mean) ** 2 for x in values) / len(values)
                    std = variance ** 0.5
                else:
                    std = 1
                
                if std == 0:  # 避免除以零
                    std = 1
                
                means[col] = mean
                stds[col] = std
                
                # 应用标准化
                for item in data:
                    if col in item and item[col] is not None and isinstance(item[col], (int, float)):
                        item[col] = (item[col] - mean) / std
            
            stats['means'] = means
            stats['stds'] = stds
            stats['affected_rows'] = len(data)
        
        return data, stats  
  def _min_max_scale(self, data: Any, target_columns: List[str], parameters: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """最小-最大缩放"""
        feature_range = parameters.get('feature_range', (0, 1))
        stats = {'affected_rows': 0, 'affected_columns': len(target_columns), 'mins': {}, 'maxs': {}}
        
        min_val, max_val = feature_range
        
        if HAS_PANDAS and isinstance(data, pd.DataFrame):
            # 如果未指定列，使用所有数值列
            if not target_columns:
                target_columns = data.select_dtypes(include=['number']).columns.tolist()
            
            # 记录原始数据
            mins = {}
            maxs = {}
            
            # 对每列应用缩放
            for col in target_columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    continue
                
                try:
                    # 计算最小值和最大值
                    col_min = data[col].min()
                    col_max = data[col].max()
                    
                    # 避免除以零
                    if col_min == col_max:
                        data[col] = min_val
                    else:
                        # 应用缩放
                        data[col] = min_val + (data[col] - col_min) * (max_val - min_val) / (col_max - col_min)
                    
                    # 记录统计量
                    mins[col] = float(col_min)
                    maxs[col] = float(col_max)
                
                except Exception as e:
                    self.logger.warning(f"对列 {col} 应用最小-最大缩放失败: {str(e)}")
            
            stats['mins'] = mins
            stats['maxs'] = maxs
            stats['affected_rows'] = len(data)
        
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            # 处理字典列表
            if not target_columns:
                # 尝试找出数值列
                target_columns = []
                for key, value in data[0].items():
                    if isinstance(value, (int, float)):
                        target_columns.append(key)
            
            # 计算最小值和最大值
            mins = {}
            maxs = {}
            
            for col in target_columns:
                values = [item[col] for item in data if col in item and item[col] is not None and isinstance(item[col], (int, float))]
                if not values:
                    continue
                
                col_min = min(values)
                col_max = max(values)
                
                mins[col] = col_min
                maxs[col] = col_max
                
                # 应用缩放
                for item in data:
                    if col in item and item[col] is not None and isinstance(item[col], (int, float)):
                        if col_min == col_max:
                            item[col] = min_val
                        else:
                            item[col] = min_val + (item[col] - col_min) * (max_val - min_val) / (col_max - col_min)
            
            stats['mins'] = mins
            stats['maxs'] = maxs
            stats['affected_rows'] = len(data)
        
        return data, stats
    
    def _robust_scale(self, data: Any, target_columns: List[str], parameters: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """稳健缩放"""
        quantile_range = parameters.get('quantile_range', (25.0, 75.0))
        stats = {'affected_rows': 0, 'affected_columns': len(target_columns), 'centers': {}, 'scales': {}}
        
        q_min, q_max = quantile_range
        
        if HAS_PANDAS and isinstance(data, pd.DataFrame):
            # 如果未指定列，使用所有数值列
            if not target_columns:
                target_columns = data.select_dtypes(include=['number']).columns.tolist()
            
            # 记录原始数据
            centers = {}
            scales = {}
            
            # 对每列应用稳健缩放
            for col in target_columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    continue
                
                try:
                    # 计算中位数和四分位距
                    q_min_val = np.percentile(data[col].dropna(), q_min)
                    q_max_val = np.percentile(data[col].dropna(), q_max)
                    center = np.median(data[col].dropna())
                    scale = q_max_val - q_min_val
                    
                    # 避免除以零
                    if scale == 0:
                        scale = 1.0
                    
                    # 应用缩放
                    data[col] = (data[col] - center) / scale
                    
                    # 记录统计量
                    centers[col] = float(center)
                    scales[col] = float(scale)
                
                except Exception as e:
                    self.logger.warning(f"对列 {col} 应用稳健缩放失败: {str(e)}")
            
            stats['centers'] = centers
            stats['scales'] = scales
            stats['affected_rows'] = len(data)
        
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            # 处理字典列表
            if not target_columns:
                # 尝试找出数值列
                target_columns = []
                for key, value in data[0].items():
                    if isinstance(value, (int, float)):
                        target_columns.append(key)
            
            # 计算中位数和四分位距
            centers = {}
            scales = {}
            
            for col in target_columns:
                values = [item[col] for item in data if col in item and item[col] is not None and isinstance(item[col], (int, float))]
                if not values:
                    continue
                
                # 排序值
                sorted_values = sorted(values)
                n = len(sorted_values)
                
                # 计算中位数
                if n % 2 == 0:
                    center = (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
                else:
                    center = sorted_values[n//2]
                
                # 计算四分位数
                q_min_idx = int(n * q_min / 100)
                q_max_idx = int(n * q_max / 100)
                q_min_val = sorted_values[q_min_idx]
                q_max_val = sorted_values[q_max_idx]
                scale = q_max_val - q_min_val
                
                # 避免除以零
                if scale == 0:
                    scale = 1.0
                
                centers[col] = center
                scales[col] = scale
                
                # 应用缩放
                for item in data:
                    if col in item and item[col] is not None and isinstance(item[col], (int, float)):
                        item[col] = (item[col] - center) / scale
            
            stats['centers'] = centers
            stats['scales'] = scales
            stats['affected_rows'] = len(data)
        
        return data, stats    def
 _normalize(self, data: Any, target_columns: List[str], parameters: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """归一化"""
        norm = parameters.get('norm', 'l2')
        stats = {'affected_rows': 0, 'affected_columns': len(target_columns), 'norms': {}}
        
        if HAS_PANDAS and isinstance(data, pd.DataFrame):
            # 如果未指定列，使用所有数值列
            if not target_columns:
                target_columns = data.select_dtypes(include=['number']).columns.tolist()
            
            # 记录原始数据
            norms = {}
            
            # 对每行应用归一化
            for i, row in data.iterrows():
                values = row[target_columns].values
                
                if norm == 'l1':
                    # L1范数（曼哈顿距离）
                    norm_value = np.sum(np.abs(values))
                elif norm == 'l2':
                    # L2范数（欧几里得距离）
                    norm_value = np.sqrt(np.sum(values ** 2))
                elif norm == 'max':
                    # 最大范数
                    norm_value = np.max(np.abs(values))
                else:
                    raise ValueError(f"不支持的范数: {norm}")
                
                # 避免除以零
                if norm_value == 0:
                    norm_value = 1.0
                
                # 应用归一化
                data.loc[i, target_columns] = values / norm_value
                norms[i] = float(norm_value)
            
            stats['norms'] = norms
            stats['affected_rows'] = len(data)
        
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            # 处理字典列表
            if not target_columns:
                # 尝试找出数值列
                target_columns = []
                for key, value in data[0].items():
                    if isinstance(value, (int, float)):
                        target_columns.append(key)
            
            # 对每个项目应用归一化
            norms = {}
            
            for i, item in enumerate(data):
                # 收集数值
                values = np.array([item.get(col, 0) for col in target_columns])
                
                if norm == 'l1':
                    # L1范数（曼哈顿距离）
                    norm_value = np.sum(np.abs(values))
                elif norm == 'l2':
                    # L2范数（欧几里得距离）
                    norm_value = np.sqrt(np.sum(values ** 2))
                elif norm == 'max':
                    # 最大范数
                    norm_value = np.max(np.abs(values))
                else:
                    raise ValueError(f"不支持的范数: {norm}")
                
                # 避免除以零
                if norm_value == 0:
                    norm_value = 1.0
                
                # 应用归一化
                for j, col in enumerate(target_columns):
                    if col in item and item[col] is not None and isinstance(item[col], (int, float)):
                        item[col] = item[col] / norm_value
                
                norms[i] = float(norm_value)
            
            stats['norms'] = norms
            stats['affected_rows'] = len(data)
        
        return data, stats
    
    # 特征工程函数
    def _create_interaction_features(self, data: Any, target_columns: List[str], parameters: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """创建交互特征"""
        interaction_type = parameters.get('interaction_type', 'multiplication')
        stats = {'affected_rows': 0, 'affected_columns': len(target_columns), 'new_columns': []}
        
        if len(target_columns) < 2:
            self.logger.warning("创建交互特征需要至少两列")
            return data, stats
        
        if HAS_PANDAS and isinstance(data, pd.DataFrame):
            # 记录原始数据
            original_columns = set(data.columns)
            new_columns = []
            
            # 创建所有可能的列对组合
            from itertools import combinations
            column_pairs = list(combinations(target_columns, 2))
            
            # 对每对列创建交互特征
            for col1, col2 in column_pairs:
                if not pd.api.types.is_numeric_dtype(data[col1]) or not pd.api.types.is_numeric_dtype(data[col2]):
                    continue
                
                try:
                    interaction_col = f"{col1}_{interaction_type}_{col2}"
                    
                    if interaction_type == 'multiplication':
                        data[interaction_col] = data[col1] * data[col2]
                    elif interaction_type == 'addition':
                        data[interaction_col] = data[col1] + data[col2]
                    elif interaction_type == 'subtraction':
                        data[interaction_col] = data[col1] - data[col2]
                    elif interaction_type == 'division':
                        # 避免除以零
                        data[interaction_col] = data[col1] / data[col2].replace(0, np.nan)
                    else:
                        raise ValueError(f"不支持的交互类型: {interaction_type}")
                    
                    new_columns.append(interaction_col)
                
                except Exception as e:
                    self.logger.warning(f"创建交互特征 {col1} 和 {col2} 失败: {str(e)}")
            
            # 计算新增的列
            stats['new_columns'] = new_columns
            stats['affected_rows'] = len(data)
        
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            # 如果有pandas，转换为DataFrame处理后再转回来
            if HAS_PANDAS:
                df = pd.DataFrame(data)
                df, df_stats = self._create_interaction_features(df, target_columns, parameters)
                return df.to_dict('records'), df_stats
            
            # 创建所有可能的列对组合
            from itertools import combinations
            column_pairs = list(combinations(target_columns, 2))
            new_columns = []
            
            # 对每对列创建交互特征
            for col1, col2 in column_pairs:
                interaction_col = f"{col1}_{interaction_type}_{col2}"
                new_columns.append(interaction_col)
                
                for item in data:
                    if col1 not in item or col2 not in item or item[col1] is None or item[col2] is None:
                        item[interaction_col] = None
                        continue
                    
                    if not isinstance(item[col1], (int, float)) or not isinstance(item[col2], (int, float)):
                        item[interaction_col] = None
                        continue
                    
                    if interaction_type == 'multiplication':
                        item[interaction_col] = item[col1] * item[col2]
                    elif interaction_type == 'addition':
                        item[interaction_col] = item[col1] + item[col2]
                    elif interaction_type == 'subtraction':
                        item[interaction_col] = item[col1] - item[col2]
                    elif interaction_type == 'division':
                        # 避免除以零
                        item[interaction_col] = item[col1] / item[col2] if item[col2] != 0 else None
                    else:
                        item[interaction_col] = None
            
            stats['new_columns'] = new_columns
            stats['affected_rows'] = len(data)
        
        return data, stats   
 def _polynomial_features(self, data: Any, target_columns: List[str], parameters: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """多项式特征"""
        degree = parameters.get('degree', 2)
        interaction_only = parameters.get('interaction_only', False)
        stats = {'affected_rows': 0, 'affected_columns': len(target_columns), 'new_columns': []}
        
        if HAS_PANDAS and isinstance(data, pd.DataFrame):
            # 如果未指定列，使用所有数值列
            if not target_columns:
                target_columns = data.select_dtypes(include=['number']).columns.tolist()
            
            # 使用sklearn的PolynomialFeatures
            if HAS_SKLEARN:
                from sklearn.preprocessing import PolynomialFeatures
                
                try:
                    # 提取特征矩阵
                    X = data[target_columns].values
                    
                    # 创建多项式特征
                    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
                    X_poly = poly.fit_transform(X)
                    
                    # 获取特征名称
                    feature_names = poly.get_feature_names_out(target_columns)
                    
                    # 添加新特征到DataFrame
                    new_columns = []
                    for i, name in enumerate(feature_names):
                        # 跳过原始特征
                        if name in target_columns:
                            continue
                        
                        # 格式化特征名称
                        formatted_name = name.replace(' ', '_').replace('^', '_pow_')
                        data[formatted_name] = X_poly[:, i]
                        new_columns.append(formatted_name)
                    
                    stats['new_columns'] = new_columns
                    stats['affected_rows'] = len(data)
                
                except Exception as e:
                    self.logger.warning(f"创建多项式特征失败: {str(e)}")
            
            else:
                # 简单实现（仅支持二次项和交互项）
                new_columns = []
                
                # 创建二次项
                if not interaction_only and degree >= 2:
                    for col in target_columns:
                        if not pd.api.types.is_numeric_dtype(data[col]):
                            continue
                        
                        squared_col = f"{col}_pow_2"
                        data[squared_col] = data[col] ** 2
                        new_columns.append(squared_col)
                
                # 创建交互项
                from itertools import combinations
                for cols in combinations(target_columns, 2):
                    col1, col2 = cols
                    if not pd.api.types.is_numeric_dtype(data[col1]) or not pd.api.types.is_numeric_dtype(data[col2]):
                        continue
                    
                    interaction_col = f"{col1}_mul_{col2}"
                    data[interaction_col] = data[col1] * data[col2]
                    new_columns.append(interaction_col)
                
                stats['new_columns'] = new_columns
                stats['affected_rows'] = len(data)
        
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            # 如果有pandas，转换为DataFrame处理后再转回来
            if HAS_PANDAS:
                df = pd.DataFrame(data)
                df, df_stats = self._polynomial_features(df, target_columns, parameters)
                return df.to_dict('records'), df_stats
            
            # 简单实现（仅支持二次项和交互项）
            if not target_columns:
                # 尝试找出数值列
                target_columns = []
                for key, value in data[0].items():
                    if isinstance(value, (int, float)):
                        target_columns.append(key)
            
            new_columns = []
            
            # 创建二次项
            if not interaction_only and degree >= 2:
                for col in target_columns:
                    squared_col = f"{col}_pow_2"
                    new_columns.append(squared_col)
                    
                    for item in data:
                        if col in item and item[col] is not None and isinstance(item[col], (int, float)):
                            item[squared_col] = item[col] ** 2
                        else:
                            item[squared_col] = None
            
            # 创建交互项
            from itertools import combinations
            for cols in combinations(target_columns, 2):
                col1, col2 = cols
                interaction_col = f"{col1}_mul_{col2}"
                new_columns.append(interaction_col)
                
                for item in data:
                    if col1 in item and col2 in item and item[col1] is not None and item[col2] is not None and isinstance(item[col1], (int, float)) and isinstance(item[col2], (int, float)):
                        item[interaction_col] = item[col1] * item[col2]
                    else:
                        item[interaction_col] = None
            
            stats['new_columns'] = new_columns
            stats['affected_rows'] = len(data)
        
        return data, stats
    
    def _aggregate_features(self, data: Any, target_columns: List[str], parameters: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """聚合特征"""
        groupby_columns = parameters.get('groupby_columns', [])
        agg_functions = parameters.get('agg_functions', {})
        stats = {'affected_rows': 0, 'affected_columns': len(target_columns), 'new_columns': []}
        
        if not groupby_columns or not agg_functions:
            self.logger.warning("聚合特征需要指定分组列和聚合函数")
            return data, stats
        
        if HAS_PANDAS and isinstance(data, pd.DataFrame):
            # 记录原始数据
            original_columns = set(data.columns)
            
            try:
                # 创建聚合特征
                agg_data = data.groupby(groupby_columns).agg(agg_functions)
                
                # 重置索引，使分组列成为普通列
                agg_data = agg_data.reset_index()
                
                # 创建列名
                agg_data.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in agg_data.columns]
                
                # 合并回原始数据
                data = pd.merge(data, agg_data, on=groupby_columns, how='left')
                
                # 计算新增的列
                new_columns = list(set(data.columns) - original_columns)
                stats['new_columns'] = new_columns
                stats['affected_rows'] = len(data)
            
            except Exception as e:
                self.logger.warning(f"创建聚合特征失败: {str(e)}")
        
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            # 如果有pandas，转换为DataFrame处理后再转回来
            if HAS_PANDAS:
                df = pd.DataFrame(data)
                df, df_stats = self._aggregate_features(df, target_columns, parameters)
                return df.to_dict('records'), df_stats
            
            # 简单实现（仅支持基本聚合函数）
            # 按分组列分组
            groups = {}
            for item in data:
                # 创建分组键
                group_key = tuple(item.get(col) for col in groupby_columns)
                if group_key not in groups:
                    groups[group_key] = []
                groups[group_key].append(item)
            
            # 计算聚合值
            agg_results = {}
            for group_key, group_items in groups.items():
                agg_results[group_key] = {}
                
                for col, func_name in agg_functions.items():
                    values = [item.get(col) for item in group_items if col in item and item[col] is not None and isinstance(item[col], (int, float))]
                    
                    if not values:
                        continue
                    
                    if func_name == 'mean':
                        agg_results[group_key][f"{col}_{func_name}"] = sum(values) / len(values)
                    elif func_name == 'sum':
                        agg_results[group_key][f"{col}_{func_name}"] = sum(values)
                    elif func_name == 'min':
                        agg_results[group_key][f"{col}_{func_name}"] = min(values)
                    elif func_name == 'max':
                        agg_results[group_key][f"{col}_{func_name}"] = max(values)
                    elif func_name == 'count':
                        agg_results[group_key][f"{col}_{func_name}"] = len(values)
            
            # 将聚合结果合并回原始数据
            new_columns = []
            for col, func_name in agg_functions.items():
                new_col = f"{col}_{func_name}"
                new_columns.append(new_col)
            
            for item in data:
                group_key = tuple(item.get(col) for col in groupby_columns)
                if group_key in agg_results:
                    for new_col, value in agg_results[group_key].items():
                        item[new_col] = value
            
            stats['new_columns'] = new_columns
            stats['affected_rows'] = len(data)
        
        return data, stats 
   def _extract_text_features(self, data: Any, target_columns: List[str], parameters: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """提取文本特征"""
        features = parameters.get('features', ['length', 'word_count'])
        stats = {'affected_rows': 0, 'affected_columns': len(target_columns), 'new_columns': []}
        
        if HAS_PANDAS and isinstance(data, pd.DataFrame):
            # 如果未指定列，使用所有文本列
            if not target_columns:
                target_columns = data.select_dtypes(include=['object']).columns.tolist()
            
            # 记录原始数据
            original_columns = set(data.columns)
            new_columns = []
            
            # 对每列提取特征
            for col in target_columns:
                if not pd.api.types.is_string_dtype(data[col]):
                    continue
                
                try:
                    # 提取特征
                    for feature in features:
                        feature_col = f"{col}_{feature}"
                        
                        if feature == 'length':
                            data[feature_col] = data[col].str.len()
                        elif feature == 'word_count':
                            data[feature_col] = data[col].str.split().str.len()
                        elif feature == 'char_count':
                            data[feature_col] = data[col].str.replace(r'\s', '', regex=True).str.len()
                        elif feature == 'sentence_count':
                            data[feature_col] = data[col].str.count(r'[.!?]+')
                        else:
                            self.logger.warning(f"不支持的文本特征: {feature}")
                            continue
                        
                        new_columns.append(feature_col)
                
                except Exception as e:
                    self.logger.warning(f"对列 {col} 提取文本特征失败: {str(e)}")
            
            # 计算新增的列
            stats['new_columns'] = new_columns
            stats['affected_rows'] = len(data)
        
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            # 如果未指定列，尝试找出文本列
            if not target_columns:
                target_columns = []
                for key, value in data[0].items():
                    if isinstance(value, str):
                        target_columns.append(key)
            
            new_columns = []
            
            # 对每列提取特征
            for col in target_columns:
                for feature in features:
                    feature_col = f"{col}_{feature}"
                    new_columns.append(feature_col)
                    
                    for item in data:
                        if col not in item or item[col] is None or not isinstance(item[col], str):
                            item[feature_col] = None
                            continue
                        
                        text = item[col]
                        
                        if feature == 'length':
                            item[feature_col] = len(text)
                        elif feature == 'word_count':
                            item[feature_col] = len(text.split())
                        elif feature == 'char_count':
                            item[feature_col] = len(re.sub(r'\s', '', text))
                        elif feature == 'sentence_count':
                            item[feature_col] = len(re.findall(r'[.!?]+', text))
                        else:
                            item[feature_col] = None
            
            stats['new_columns'] = new_columns
            stats['affected_rows'] = len(data)
        
        return data, stats