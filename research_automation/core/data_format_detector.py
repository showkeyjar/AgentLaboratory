"""
数据格式检测器组件

提供多格式数据识别和自动格式转换功能
"""

import logging
import os
import json
import csv
import xml.etree.ElementTree as ET
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import mimetypes
try:
    import chardet
    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False
    chardet = None
from pathlib import Path

from ..models.analysis_models import DataFormat, ProcessedData
from ..models.base_models import BaseModel
from .base_component import BaseComponent

logger = logging.getLogger(__name__)


class DataFormatConfidence(Enum):
    """数据格式检测置信度"""
    VERY_HIGH = "very_high"     # 95%+
    HIGH = "high"               # 80-95%
    MEDIUM = "medium"           # 60-80%
    LOW = "low"                 # 40-60%
    VERY_LOW = "very_low"       # <40%


@dataclass
class FormatDetectionResult(BaseModel):
    """格式检测结果"""
    detected_format: DataFormat = DataFormat.TEXT
    confidence: DataFormatConfidence = DataFormatConfidence.MEDIUM
    confidence_score: float = 0.5
    
    # 检测详情
    file_extension: str = ""
    mime_type: str = ""
    encoding: str = "utf-8"
    
    # 格式特征
    format_features: Dict[str, Any] = field(default_factory=dict)
    
    # 转换建议
    conversion_suggestions: List[str] = field(default_factory=list)
    
    # 错误信息
    detection_errors: List[str] = field(default_factory=list)
    
    def is_reliable(self) -> bool:
        """判断检测结果是否可靠"""
        return self.confidence_score >= 0.7
    
    def get_confidence_level(self) -> DataFormatConfidence:
        """根据置信度分数获取置信度级别"""
        if self.confidence_score >= 0.95:
            return DataFormatConfidence.VERY_HIGH
        elif self.confidence_score >= 0.8:
            return DataFormatConfidence.HIGH
        elif self.confidence_score >= 0.6:
            return DataFormatConfidence.MEDIUM
        elif self.confidence_score >= 0.4:
            return DataFormatConfidence.LOW
        else:
            return DataFormatConfidence.VERY_LOW


@dataclass
class ConversionResult(BaseModel):
    """格式转换结果"""
    success: bool = False
    source_format: DataFormat = DataFormat.TEXT
    target_format: DataFormat = DataFormat.CSV
    
    # 转换后的数据
    converted_data: Any = None
    output_path: Optional[str] = None
    
    # 转换统计
    rows_processed: int = 0
    columns_processed: int = 0
    conversion_time_seconds: float = 0.0
    
    # 质量信息
    data_quality_score: float = 1.0
    conversion_warnings: List[str] = field(default_factory=list)
    conversion_errors: List[str] = field(default_factory=list)
    
    def has_warnings(self) -> bool:
        """是否有转换警告"""
        return len(self.conversion_warnings) > 0
    
    def has_errors(self) -> bool:
        """是否有转换错误"""
        return len(self.conversion_errors) > 0


class DataFormatDetector(BaseComponent):
    """数据格式检测器"""
    
    def get_required_configs(self) -> List[str]:
        """获取必需的配置项"""
        return []
    
    def _setup_component(self):
        """设置组件特定的初始化逻辑"""
        self.logger.info("数据格式检测器初始化")
        
        # 格式检测规则
        self.format_rules = {
            DataFormat.CSV: {
                'extensions': ['.csv', '.tsv'],
                'mime_types': ['text/csv', 'application/csv'],
                'signatures': [b',', b';', b'\t'],
                'validators': [self._validate_csv]
            },
            DataFormat.JSON: {
                'extensions': ['.json', '.jsonl'],
                'mime_types': ['application/json', 'text/json'],
                'signatures': [b'{', b'['],
                'validators': [self._validate_json]
            },
            DataFormat.XML: {
                'extensions': ['.xml', '.xsd', '.xsl'],
                'mime_types': ['application/xml', 'text/xml'],
                'signatures': [b'<?xml', b'<'],
                'validators': [self._validate_xml]
            },
            DataFormat.EXCEL: {
                'extensions': ['.xlsx', '.xls', '.xlsm'],
                'mime_types': ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                              'application/vnd.ms-excel'],
                'signatures': [b'PK\x03\x04', b'\xd0\xcf\x11\xe0'],
                'validators': [self._validate_excel]
            },
            DataFormat.PARQUET: {
                'extensions': ['.parquet', '.pq'],
                'mime_types': ['application/octet-stream'],
                'signatures': [b'PAR1'],
                'validators': [self._validate_parquet]
            },
            DataFormat.TEXT: {
                'extensions': ['.txt', '.log', '.md'],
                'mime_types': ['text/plain', 'text/markdown'],
                'signatures': [],
                'validators': [self._validate_text]
            }
        }
        
        # 支持的转换路径
        self.conversion_paths = {
            (DataFormat.CSV, DataFormat.JSON): self._convert_csv_to_json,
            (DataFormat.CSV, DataFormat.EXCEL): self._convert_csv_to_excel,
            (DataFormat.JSON, DataFormat.CSV): self._convert_json_to_csv,
            (DataFormat.EXCEL, DataFormat.CSV): self._convert_excel_to_csv,
            (DataFormat.XML, DataFormat.JSON): self._convert_xml_to_json,
            (DataFormat.TEXT, DataFormat.CSV): self._convert_text_to_csv
        }
        
        self.logger.info("数据格式检测器初始化完成")
    
    def detect_format(self, 
                     file_path: str, 
                     sample_size: int = 1024) -> FormatDetectionResult:
        """
        检测文件格式
        
        Args:
            file_path: 文件路径
            sample_size: 采样大小（字节）
            
        Returns:
            格式检测结果
        """
        try:
            self.logger.info(f"开始检测文件格式: {file_path}")
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"文件不存在: {file_path}")
            
            # 初始化检测结果
            result = FormatDetectionResult()
            
            # 获取文件基本信息
            file_path_obj = Path(file_path)
            result.file_extension = file_path_obj.suffix.lower()
            
            # 检测MIME类型
            mime_type, _ = mimetypes.guess_type(file_path)
            result.mime_type = mime_type or ""
            
            # 检测文件编码
            result.encoding = self._detect_encoding(file_path)
            
            # 读取文件样本
            sample_data = self._read_file_sample(file_path, sample_size)
            
            # 执行格式检测
            format_scores = {}
            
            for data_format, rules in self.format_rules.items():
                score = self._calculate_format_score(
                    sample_data, result.file_extension, 
                    result.mime_type, rules
                )
                format_scores[data_format] = score
            
            # 选择最高分的格式
            best_format = max(format_scores, key=format_scores.get)
            best_score = format_scores[best_format]
            
            result.detected_format = best_format
            result.confidence_score = best_score
            result.confidence = result.get_confidence_level()
            
            # 提取格式特征
            result.format_features = self._extract_format_features(
                sample_data, best_format
            )
            
            # 生成转换建议
            result.conversion_suggestions = self._generate_conversion_suggestions(
                best_format, result.confidence_score
            )
            
            self.logger.info(f"格式检测完成: {best_format.value} (置信度: {best_score:.2f})")
            return result
            
        except Exception as e:
            self.logger.error(f"格式检测失败: {str(e)}")
            result = FormatDetectionResult()
            result.detection_errors.append(str(e))
            return result
    
    def detect_format_from_data(self, 
                               data: Union[str, bytes], 
                               filename: str = "") -> FormatDetectionResult:
        """
        从数据内容检测格式
        
        Args:
            data: 数据内容
            filename: 文件名（可选）
            
        Returns:
            格式检测结果
        """
        try:
            self.logger.info("从数据内容检测格式")
            
            result = FormatDetectionResult()
            
            # 从文件名获取扩展名
            if filename:
                result.file_extension = Path(filename).suffix.lower()
            
            # 转换为字节数据
            if isinstance(data, str):
                sample_data = data.encode('utf-8')
                result.encoding = 'utf-8'
            else:
                sample_data = data
                result.encoding = self._detect_encoding_from_bytes(sample_data)
            
            # 执行格式检测
            format_scores = {}
            
            for data_format, rules in self.format_rules.items():
                score = self._calculate_format_score(
                    sample_data, result.file_extension, 
                    result.mime_type, rules
                )
                format_scores[data_format] = score
            
            # 选择最高分的格式
            best_format = max(format_scores, key=format_scores.get)
            best_score = format_scores[best_format]
            
            result.detected_format = best_format
            result.confidence_score = best_score
            result.confidence = result.get_confidence_level()
            
            # 提取格式特征
            result.format_features = self._extract_format_features(
                sample_data, best_format
            )
            
            self.logger.info(f"格式检测完成: {best_format.value} (置信度: {best_score:.2f})")
            return result
            
        except Exception as e:
            self.logger.error(f"格式检测失败: {str(e)}")
            result = FormatDetectionResult()
            result.detection_errors.append(str(e))
            return result
    
    def convert_format(self, 
                      source_path: str,
                      target_format: DataFormat,
                      output_path: Optional[str] = None) -> ConversionResult:
        """
        转换文件格式
        
        Args:
            source_path: 源文件路径
            target_format: 目标格式
            output_path: 输出路径（可选）
            
        Returns:
            转换结果
        """
        try:
            import time
            start_time = time.time()
            
            self.logger.info(f"开始格式转换: {source_path} -> {target_format.value}")
            
            # 检测源格式
            detection_result = self.detect_format(source_path)
            source_format = detection_result.detected_format
            
            # 初始化转换结果
            result = ConversionResult(
                source_format=source_format,
                target_format=target_format
            )
            
            # 检查是否需要转换
            if source_format == target_format:
                result.success = True
                result.converted_data = self._read_file_content(source_path)
                result.conversion_warnings.append("源格式与目标格式相同，无需转换")
                return result
            
            # 检查转换路径是否支持
            conversion_key = (source_format, target_format)
            if conversion_key not in self.conversion_paths:
                result.conversion_errors.append(
                    f"不支持从 {source_format.value} 转换到 {target_format.value}"
                )
                return result
            
            # 执行转换
            converter = self.conversion_paths[conversion_key]
            converted_data = converter(source_path)
            
            result.converted_data = converted_data
            result.success = True
            
            # 保存转换结果
            if output_path:
                self._save_converted_data(converted_data, target_format, output_path)
                result.output_path = output_path
            
            # 计算转换统计
            result.conversion_time_seconds = time.time() - start_time
            result.rows_processed, result.columns_processed = self._get_data_dimensions(converted_data)
            
            self.logger.info(f"格式转换完成: {result.conversion_time_seconds:.2f}秒")
            return result
            
        except Exception as e:
            self.logger.error(f"格式转换失败: {str(e)}")
            result = ConversionResult(source_format=source_format, target_format=target_format)
            result.conversion_errors.append(str(e))
            return result
    
    def _detect_encoding(self, file_path: str) -> str:
        """检测文件编码"""
        if HAS_CHARDET:
            try:
                with open(file_path, 'rb') as f:
                    raw_data = f.read(10000)  # 读取前10KB
                    result = chardet.detect(raw_data)
                    return result.get('encoding', 'utf-8') or 'utf-8'
            except Exception:
                pass
        
        # 简单的编码检测回退
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                f.read(1000)  # 尝试读取一部分
                return 'utf-8'
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    f.read(1000)
                    return 'gbk'
            except Exception:
                return 'utf-8'
        except Exception:
            return 'utf-8'
    
    def _detect_encoding_from_bytes(self, data: bytes) -> str:
        """从字节数据检测编码"""
        if HAS_CHARDET:
            try:
                result = chardet.detect(data[:10000])
                return result.get('encoding', 'utf-8') or 'utf-8'
            except Exception:
                pass
        
        # 简单的编码检测回退
        try:
            data.decode('utf-8')
            return 'utf-8'
        except UnicodeDecodeError:
            try:
                data.decode('gbk')
                return 'gbk'
            except Exception:
                return 'utf-8'
        except Exception:
            return 'utf-8'
    
    def _read_file_sample(self, file_path: str, sample_size: int) -> bytes:
        """读取文件样本"""
        try:
            with open(file_path, 'rb') as f:
                return f.read(sample_size)
        except Exception as e:
            self.logger.error(f"读取文件样本失败: {str(e)}")
            return b""
    
    def _calculate_format_score(self, 
                               sample_data: bytes, 
                               file_extension: str,
                               mime_type: str,
                               rules: Dict[str, Any]) -> float:
        """计算格式匹配分数"""
        score = 0.0
        
        # 扩展名匹配 (权重: 0.3)
        if file_extension in rules['extensions']:
            score += 0.3
        
        # MIME类型匹配 (权重: 0.2)
        if mime_type in rules['mime_types']:
            score += 0.2
        
        # 文件签名匹配 (权重: 0.3)
        for signature in rules['signatures']:
            if sample_data.startswith(signature):
                score += 0.3
                break
        
        # 内容验证 (权重: 0.2)
        for validator in rules['validators']:
            try:
                if validator(sample_data):
                    score += 0.2
                    break
            except Exception:
                continue
        
        return min(score, 1.0)
    
    def _validate_csv(self, data: bytes) -> bool:
        """验证CSV格式"""
        try:
            text_data = data.decode('utf-8', errors='ignore')
            lines = text_data.split('\n')[:5]  # 检查前5行
            
            # 检查是否有分隔符
            separators = [',', ';', '\t', '|']
            for sep in separators:
                if any(sep in line for line in lines):
                    # 尝试解析CSV
                    csv.Sniffer().sniff(text_data[:1000])
                    return True
            return False
        except Exception:
            return False
    
    def _validate_json(self, data: bytes) -> bool:
        """验证JSON格式"""
        try:
            text_data = data.decode('utf-8', errors='ignore')
            json.loads(text_data)
            return True
        except Exception:
            # 尝试JSONL格式
            try:
                lines = text_data.strip().split('\n')
                for line in lines[:3]:  # 检查前3行
                    if line.strip():
                        json.loads(line)
                return True
            except Exception:
                return False
    
    def _validate_xml(self, data: bytes) -> bool:
        """验证XML格式"""
        try:
            text_data = data.decode('utf-8', errors='ignore')
            ET.fromstring(text_data)
            return True
        except Exception:
            return False
    
    def _validate_excel(self, data: bytes) -> bool:
        """验证Excel格式"""
        try:
            # 检查Excel文件签名
            if data.startswith(b'PK\x03\x04') or data.startswith(b'\xd0\xcf\x11\xe0'):
                return True
            return False
        except Exception:
            return False
    
    def _validate_parquet(self, data: bytes) -> bool:
        """验证Parquet格式"""
        try:
            return data.startswith(b'PAR1')
        except Exception:
            return False
    
    def _validate_text(self, data: bytes) -> bool:
        """验证文本格式"""
        try:
            # 尝试解码为文本
            data.decode('utf-8')
            return True
        except Exception:
            try:
                data.decode('gbk')
                return True
            except Exception:
                return False
    
    def _extract_format_features(self, 
                                sample_data: bytes, 
                                data_format: DataFormat) -> Dict[str, Any]:
        """提取格式特征"""
        features = {}
        
        try:
            if data_format == DataFormat.CSV:
                text_data = sample_data.decode('utf-8', errors='ignore')
                # 检测分隔符
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(text_data[:1000])
                features['delimiter'] = dialect.delimiter
                features['quote_char'] = dialect.quotechar
                features['has_header'] = sniffer.has_header(text_data[:1000])
                
            elif data_format == DataFormat.JSON:
                text_data = sample_data.decode('utf-8', errors='ignore')
                try:
                    parsed = json.loads(text_data)
                    features['json_type'] = type(parsed).__name__
                    if isinstance(parsed, dict):
                        features['keys'] = list(parsed.keys())[:10]
                    elif isinstance(parsed, list) and parsed:
                        features['array_length'] = len(parsed)
                        if isinstance(parsed[0], dict):
                            features['keys'] = list(parsed[0].keys())[:10]
                except Exception:
                    features['json_type'] = 'jsonlines'
                    
            elif data_format == DataFormat.XML:
                text_data = sample_data.decode('utf-8', errors='ignore')
                try:
                    root = ET.fromstring(text_data)
                    features['root_tag'] = root.tag
                    features['namespaces'] = list(root.nsmap.keys()) if hasattr(root, 'nsmap') else []
                except Exception:
                    pass
                    
        except Exception as e:
            self.logger.debug(f"提取格式特征失败: {str(e)}")
        
        return features
    
    def _generate_conversion_suggestions(self, 
                                       detected_format: DataFormat,
                                       confidence_score: float) -> List[str]:
        """生成转换建议"""
        suggestions = []
        
        if confidence_score < 0.7:
            suggestions.append("检测置信度较低，建议手动确认格式")
        
        # 基于格式的转换建议
        if detected_format == DataFormat.CSV:
            suggestions.extend([
                "可转换为JSON格式以支持嵌套数据",
                "可转换为Excel格式以便于编辑",
                "可转换为Parquet格式以提高存储效率"
            ])
        elif detected_format == DataFormat.JSON:
            suggestions.extend([
                "可转换为CSV格式以便于表格分析",
                "可转换为Excel格式以便于查看"
            ])
        elif detected_format == DataFormat.EXCEL:
            suggestions.extend([
                "可转换为CSV格式以便于程序处理",
                "可转换为JSON格式以支持API传输"
            ])
        elif detected_format == DataFormat.XML:
            suggestions.extend([
                "可转换为JSON格式以简化数据结构",
                "可转换为CSV格式以便于分析"
            ])
        
        return suggestions
    
    def _read_file_content(self, file_path: str) -> Any:
        """读取文件内容"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            with open(file_path, 'rb') as f:
                return f.read()
    
    def _convert_csv_to_json(self, source_path: str) -> List[Dict[str, Any]]:
        """CSV转JSON"""
        if HAS_PANDAS:
            df = pd.read_csv(source_path)
            return df.to_dict('records')
        else:
            # 使用标准库实现
            result = []
            with open(source_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    result.append(dict(row))
            return result
    
    def _convert_csv_to_excel(self, source_path: str) -> List[Dict[str, Any]]:
        """CSV转Excel（返回数据，需要pandas保存为Excel）"""
        if not HAS_PANDAS:
            raise NotImplementedError("Excel转换需要pandas库")
        df = pd.read_csv(source_path)
        return df
    
    def _convert_json_to_csv(self, source_path: str) -> List[Dict[str, Any]]:
        """JSON转CSV"""
        with open(source_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        else:
            return [data]
    
    def _convert_excel_to_csv(self, source_path: str) -> List[Dict[str, Any]]:
        """Excel转CSV"""
        if not HAS_PANDAS:
            raise NotImplementedError("Excel读取需要pandas库")
        df = pd.read_excel(source_path)
        return df.to_dict('records')
    
    def _convert_xml_to_json(self, source_path: str) -> Dict[str, Any]:
        """XML转JSON"""
        tree = ET.parse(source_path)
        root = tree.getroot()
        
        def xml_to_dict(element):
            result = {}
            for child in element:
                if len(child) == 0:
                    result[child.tag] = child.text
                else:
                    result[child.tag] = xml_to_dict(child)
            return result
        
        return {root.tag: xml_to_dict(root)}
    
    def _convert_text_to_csv(self, source_path: str) -> List[Dict[str, Any]]:
        """文本转CSV（简单的行分割）"""
        with open(source_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 简单处理：每行作为一个记录
        data = [{'line_number': i+1, 'content': line.strip()} 
                for i, line in enumerate(lines) if line.strip()]
        
        return data
    
    def _save_converted_data(self, 
                           data: Any, 
                           target_format: DataFormat, 
                           output_path: str):
        """保存转换后的数据"""
        try:
            if target_format == DataFormat.CSV:
                if HAS_PANDAS and isinstance(data, pd.DataFrame):
                    data.to_csv(output_path, index=False)
                elif isinstance(data, list) and data and isinstance(data[0], dict):
                    # 使用标准库保存CSV
                    with open(output_path, 'w', newline='', encoding='utf-8') as f:
                        if data:
                            fieldnames = data[0].keys()
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(data)
                else:
                    # 尝试使用pandas（如果可用）
                    if HAS_PANDAS:
                        pd.DataFrame(data).to_csv(output_path, index=False)
                    else:
                        raise NotImplementedError("CSV保存需要pandas库或字典列表格式")
                    
            elif target_format == DataFormat.JSON:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                    
            elif target_format == DataFormat.EXCEL:
                if not HAS_PANDAS:
                    raise NotImplementedError("Excel保存需要pandas库")
                if isinstance(data, pd.DataFrame):
                    data.to_excel(output_path, index=False)
                else:
                    pd.DataFrame(data).to_excel(output_path, index=False)
                    
            elif target_format == DataFormat.TEXT:
                with open(output_path, 'w', encoding='utf-8') as f:
                    if isinstance(data, str):
                        f.write(data)
                    else:
                        f.write(str(data))
                        
        except Exception as e:
            self.logger.error(f"保存转换数据失败: {str(e)}")
            raise
    
    def _get_data_dimensions(self, data: Any) -> Tuple[int, int]:
        """获取数据维度"""
        try:
            if HAS_PANDAS and isinstance(data, pd.DataFrame):
                return data.shape
            elif isinstance(data, list):
                rows = len(data)
                cols = len(data[0]) if data and isinstance(data[0], dict) else 1
                return rows, cols
            elif isinstance(data, dict):
                return 1, len(data)
            else:
                return 1, 1
        except Exception:
            return 0, 0
    
    def get_supported_formats(self) -> List[DataFormat]:
        """获取支持的格式列表"""
        return list(self.format_rules.keys())
    
    def get_supported_conversions(self) -> List[Tuple[DataFormat, DataFormat]]:
        """获取支持的转换路径"""
        return list(self.conversion_paths.keys())
    
    def batch_detect_formats(self, 
                           file_paths: List[str]) -> Dict[str, FormatDetectionResult]:
        """批量检测文件格式"""
        results = {}
        
        for file_path in file_paths:
            try:
                result = self.detect_format(file_path)
                results[file_path] = result
            except Exception as e:
                self.logger.error(f"批量检测失败 {file_path}: {str(e)}")
                result = FormatDetectionResult()
                result.detection_errors.append(str(e))
                results[file_path] = result
        
        return results