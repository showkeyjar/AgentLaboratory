"""
数据预处理引擎
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import datetime
import uuid

from ..models.base_models import BaseModel
from .base_component import BaseComponent


class PreprocessingStepType(Enum):
    CLEANING = "cleaning"
    IMPUTATION = "imputation"


class DataType(Enum):
    NUMERIC = "numeric"
    TEXT = "text"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    UNKNOWN = "unknown"


@dataclass
class PreprocessingStep(BaseModel):
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    step_type: PreprocessingStepType = PreprocessingStepType.CLEANING
    target_columns: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    order: int = 0
    execution_count: int = 0
    last_execution_time: Optional[datetime.datetime] = None
    average_execution_time_ms: float = 0.0
    
    def update_execution_stats(self, execution_time_ms: float):
        self.execution_count += 1
        self.last_execution_time = datetime.datetime.now()
        if self.average_execution_time_ms == 0.0:
            self.average_execution_time_ms = execution_time_ms
        else:
            self.average_execution_time_ms = (
                (self.average_execution_time_ms * (self.execution_count - 1) + execution_time_ms) / 
                self.execution_count
            )


@dataclass
class ColumnProfile(BaseModel):
    name: str = ""
    data_type: DataType = DataType.UNKNOWN
    count: int = 0
    missing_count: int = 0
    unique_count: int = 0
    quality_score: float = 1.0
    
    def get_missing_rate(self) -> float:
        if self.count > 0:
            return self.missing_count / self.count
        return 0.0
    
    def get_unique_rate(self) -> float:
        if self.count > 0:
            return self.unique_count / self.count
        return 0.0
    
    def get_anomaly_rate(self) -> float:
        return 0.0


@dataclass
class DataProfile(BaseModel):
    row_count: int = 0
    column_count: int = 0
    columns: Dict[str, ColumnProfile] = field(default_factory=dict)
    data_type_counts: Dict[str, int] = field(default_factory=dict)
    overall_quality_score: float = 1.0
    missing_cells_count: int = 0
    duplicate_rows_count: int = 0
    
    def get_missing_rate(self) -> float:
        total_cells = self.row_count * self.column_count
        if total_cells > 0:
            return self.missing_cells_count / total_cells
        return 0.0
    
    def get_duplicate_rate(self) -> float:
        if self.row_count > 0:
            return self.duplicate_rows_count / self.row_count
        return 0.0
    
    def get_column_types_summary(self) -> Dict[str, int]:
        return self.data_type_counts
    
    def get_problematic_columns(self, threshold: float = 0.1) -> List[str]:
        return [name for name, profile in self.columns.items() 
                if profile.get_missing_rate() > threshold]


@dataclass 
class PreprocessingResult(BaseModel):
    success: bool = True
    error_message: str = ""
    input_rows: int = 0
    output_rows: int = 0
    dropped_rows: int = 0
    modified_columns: List[str] = field(default_factory=list)
    added_columns: List[str] = field(default_factory=list)
    removed_columns: List[str] = field(default_factory=list)
    steps_executed: List[str] = field(default_factory=list)
    steps_skipped: List[str] = field(default_factory=list)
    steps_failed: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    step_execution_times: Dict[str, float] = field(default_factory=dict)
    input_profile: Optional[DataProfile] = None
    output_profile: Optional[DataProfile] = None
    
    def get_row_change_rate(self) -> float:
        if self.input_rows > 0:
            return (self.input_rows - self.output_rows) / self.input_rows
        return 0.0
    
    def get_column_change_summary(self) -> Dict[str, int]:
        return {
            "added": len(self.added_columns),
            "modified": len(self.modified_columns),
            "removed": len(self.removed_columns)
        }
    
    def get_quality_improvement(self) -> Optional[float]:
        if self.input_profile and self.output_profile:
            return self.output_profile.overall_quality_score - self.input_profile.overall_quality_score
        return None


class DataPreprocessingEngine(BaseComponent):
    def get_required_configs(self) -> List[str]:
        return []
    
    def _setup_component(self):
        self.logger.info("数据预处理引擎初始化")
        self.pipeline: List[PreprocessingStep] = []
        self.logger.info("数据预处理引擎初始化完成")
    
    def add_preprocessing_step(self, step: PreprocessingStep) -> str:
        if step.order == 0:
            step.order = len(self.pipeline) + 1
        self.pipeline.append(step)
        self.pipeline.sort(key=lambda s: s.order)
        self.logger.info(f"添加预处理步骤: {step.name} (ID: {step.step_id})")
        return step.step_id
    
    def remove_preprocessing_step(self, step_id: str) -> bool:
        for i, step in enumerate(self.pipeline):
            if step.step_id == step_id:
                removed_step = self.pipeline.pop(i)
                self.logger.info(f"移除预处理步骤: {removed_step.name} (ID: {step_id})")
                return True
        return False
    
    def update_preprocessing_step(self, step_id: str, updates: Dict[str, Any]) -> bool:
        for step in self.pipeline:
            if step.step_id == step_id:
                for key, value in updates.items():
                    if hasattr(step, key):
                        setattr(step, key, value)
                if 'order' in updates:
                    self.pipeline.sort(key=lambda s: s.order)
                return True
        return False
    
    def get_preprocessing_step(self, step_id: str) -> Optional[PreprocessingStep]:
        for step in self.pipeline:
            if step.step_id == step_id:
                return step
        return None
    
    def get_all_preprocessing_steps(self) -> List[PreprocessingStep]:
        return self.pipeline.copy()
    
    def clear_pipeline(self):
        self.pipeline.clear()
        self.logger.info("清空预处理流水线")
    
    def preprocess_data(self, data: Any) -> Tuple[Any, PreprocessingResult]:
        import time
        start_time = time.time()
        
        result = PreprocessingResult()
        
        if not self.pipeline:
            result.success = True
            result.error_message = "预处理流水线为空，未执行任何处理"
            return data, result
        
        try:
            input_profile = self._profile_data(data)
            result.input_profile = input_profile
            result.input_rows = input_profile.row_count
        except Exception as e:
            self.logger.warning(f"获取输入数据概况失败: {str(e)}")
        
        processed_data = data
        for step in self.pipeline:
            if not step.enabled:
                result.steps_skipped.append(step.step_id)
                continue
            
            try:
                step_start_time = time.time()
                processed_data = self._execute_preprocessing_step(step, processed_data)
                
                step_execution_time = (time.time() - step_start_time) * 1000
                result.step_execution_times[step.step_id] = step_execution_time
                step.update_execution_stats(step_execution_time)
                
                result.steps_executed.append(step.step_id)
                
            except Exception as e:
                self.logger.error(f"执行预处理步骤失败: {step.name} - {str(e)}")
                result.steps_failed.append(step.step_id)
                result.success = False
                result.error_message = f"步骤 '{step.name}' 执行失败: {str(e)}"
                break
        
        try:
            output_profile = self._profile_data(processed_data)
            result.output_profile = output_profile
            result.output_rows = output_profile.row_count
            result.dropped_rows = result.input_rows - result.output_rows
        except Exception as e:
            self.logger.warning(f"获取输出数据概况失败: {str(e)}")
        
        result.execution_time_ms = (time.time() - start_time) * 1000
        return processed_data, result
    
    def _execute_preprocessing_step(self, step: PreprocessingStep, data: Any) -> Any:
        func_name = step.parameters.get('function', '')
        if func_name == 'remove_duplicates':
            return self._remove_duplicates(data, step.target_columns, step.parameters)
        elif func_name == 'remove_missing':
            return self._remove_missing(data, step.target_columns, step.parameters)
        elif func_name == 'fill_missing_mean':
            return self._fill_missing_mean(data, step.target_columns, step.parameters)
        elif func_name == 'fill_missing_constant':
            return self._fill_missing_constant(data, step.target_columns, step.parameters)
        else:
            raise ValueError(f"未知的预处理函数: {func_name}")
    
    def _profile_data(self, data: Any) -> DataProfile:
        if isinstance(data, list):
            row_count = len(data)
            if row_count > 0 and isinstance(data[0], dict):
                column_count = len(data[0])
            else:
                column_count = 1
        else:
            row_count = 1
            column_count = 1
        
        profile = DataProfile(row_count=row_count, column_count=column_count)
        
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            columns = set()
            for item in data:
                columns.update(item.keys())
            
            for column in columns:
                column_data = [item.get(column) for item in data]
                column_profile = self._profile_column(column_data)
                column_profile.name = column
                profile.columns[column] = column_profile
                
                data_type = column_profile.data_type.value
                profile.data_type_counts[data_type] = profile.data_type_counts.get(data_type, 0) + 1
            
            profile.missing_cells_count = sum(
                1 for item in data for col in columns if col not in item or item[col] is None
            )
        
        return profile
    
    def _profile_column(self, column_data: Any) -> ColumnProfile:
        profile = ColumnProfile()
        
        if isinstance(column_data, list):
            profile.count = len(column_data)
            profile.missing_count = sum(1 for x in column_data if x is None)
            profile.unique_count = len(set(x for x in column_data if x is not None))
        
        profile.data_type = self._infer_data_type(column_data)
        
        if profile.count > 0:
            missing_rate = profile.get_missing_rate()
            profile.quality_score = max(0, 1 - missing_rate)
        
        return profile
    
    def _infer_data_type(self, column_data: Any) -> DataType:
        if isinstance(column_data, list):
            if not column_data:
                return DataType.UNKNOWN
            
            sample = [x for x in column_data[:10] if x is not None]
            if not sample:
                return DataType.UNKNOWN
            
            if all(isinstance(x, (int, float)) for x in sample):
                return DataType.NUMERIC
            elif all(isinstance(x, bool) for x in sample):
                return DataType.BOOLEAN
            elif all(isinstance(x, str) for x in sample):
                unique_ratio = len(set(sample)) / len(sample)
                if unique_ratio < 0.5:
                    return DataType.CATEGORICAL
                else:
                    return DataType.TEXT
            else:
                return DataType.UNKNOWN
        else:
            return DataType.UNKNOWN
    
    def _remove_duplicates(self, data: Any, target_columns: List[str], params: Dict[str, Any]) -> Any:
        if isinstance(data, list) and data and isinstance(data[0], dict):
            if not target_columns:
                seen = set()
                result = []
                for item in data:
                    item_tuple = tuple(sorted(item.items()))
                    if item_tuple not in seen:
                        seen.add(item_tuple)
                        result.append(item)
                return result
            else:
                seen = set()
                result = []
                for item in data:
                    key = tuple(item.get(col) for col in target_columns)
                    if key not in seen:
                        seen.add(key)
                        result.append(item)
                return result
        return data
    
    def _remove_missing(self, data: Any, target_columns: List[str], params: Dict[str, Any]) -> Any:
        if isinstance(data, list):
            if data and isinstance(data[0], dict):
                if not target_columns:
                    return [item for item in data if all(v is not None for v in item.values())]
                else:
                    return [item for item in data if all(item.get(col) is not None for col in target_columns)]
            else:
                return [item for item in data if item is not None]
        return data
    
    def _fill_missing_mean(self, data: Any, target_columns: List[str], params: Dict[str, Any]) -> Any:
        if isinstance(data, list) and data and isinstance(data[0], dict):
            columns = target_columns if target_columns else list(data[0].keys())
            
            means = {}
            for col in columns:
                values = [item.get(col) for item in data if item.get(col) is not None and isinstance(item.get(col), (int, float))]
                if values:
                    means[col] = sum(values) / len(values)
            
            result = []
            for item in data:
                new_item = dict(item)
                for col in columns:
                    if col in means and (col not in item or item[col] is None):
                        new_item[col] = means[col]
                result.append(new_item)
            return result
        return data
    
    def _fill_missing_constant(self, data: Any, target_columns: List[str], params: Dict[str, Any]) -> Any:
        fill_value = params.get('fill_value', 0)
        
        if isinstance(data, list) and data and isinstance(data[0], dict):
            columns = target_columns if target_columns else list(data[0].keys())
            
            result = []
            for item in data:
                new_item = dict(item)
                for col in columns:
                    if col not in item or item[col] is None:
                        new_item[col] = fill_value
                result.append(new_item)
            return result
        return data
