"""
分析方法选择器
根据数据特征和研究目标自动选择最适合的分析方法
"""

import logging
import json
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import datetime
import uuid

from ..models.base_models import BaseModel, MetricScore
from .base_component import BaseComponent

# 尝试导入可选依赖
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


class AnalysisType(Enum):
    """分析类型"""
    DESCRIPTIVE = "descriptive"           # 描述性分析
    INFERENTIAL = "inferential"           # 推断性分析
    PREDICTIVE = "predictive"             # 预测性分析
    DIAGNOSTIC = "diagnostic"             # 诊断性分析
    PRESCRIPTIVE = "prescriptive"         # 规范性分析
    EXPLORATORY = "exploratory"           # 探索性分析


class DataCharacteristic(Enum):
    """数据特征"""
    NUMERICAL = "numerical"               # 数值型
    CATEGORICAL = "categorical"           # 类别型
    TIME_SERIES = "time_series"          # 时间序列
    TEXT = "text"                        # 文本
    MIXED = "mixed"                      # 混合型
    HIGH_DIMENSIONAL = "high_dimensional" # 高维
    SPARSE = "sparse"                    # 稀疏
    IMBALANCED = "imbalanced"            # 不平衡


class ResearchObjective(Enum):
    """研究目标"""
    CLASSIFICATION = "classification"     # 分类
    REGRESSION = "regression"            # 回归
    CLUSTERING = "clustering"            # 聚类
    ASSOCIATION = "association"          # 关联分析
    ANOMALY_DETECTION = "anomaly_detection" # 异常检测
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction" # 降维
    HYPOTHESIS_TESTING = "hypothesis_testing" # 假设检验
    CORRELATION_ANALYSIS = "correlation_analysis" # 相关性分析
    TREND_ANALYSIS = "trend_analysis"    # 趋势分析
    COMPARISON = "comparison"            # 比较分析


@dataclass
class AnalysisMethod(BaseModel):
    """分析方法"""
    name: str = ""
    category: str = ""
    description: str = ""
    # 适用条件
    suitable_data_types: List[DataCharacteristic] = field(default_factory=list)
    suitable_objectives: List[ResearchObjective] = field(default_factory=list)
    suitable_sample_sizes: Tuple[int, int] = (0, float('inf'))  # (min, max)
    # 方法特性
    complexity: str = "medium"  # low, medium, high
    interpretability: str = "medium"  # low, medium, high
    computational_cost: str = "medium"  # low, medium, high
    # 依赖和参数
    required_libraries: List[str] = field(default_factory=list)
    default_parameters: Dict[str, Any] = field(default_factory=dict)
    parameter_ranges: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)
    # 性能指标
    typical_accuracy: Optional[float] = None
    typical_speed: Optional[float] = None
    # 使用统计
    usage_count: int = 0
    success_rate: float = 1.0
    average_performance: float = 0.0


@dataclass
class DataProfile(BaseModel):
    """数据概况"""
    # 基本信息
    sample_size: int = 0
    feature_count: int = 0
    target_count: int = 0
    # 数据特征
    data_characteristics: List[DataCharacteristic] = field(default_factory=list)
    missing_rate: float = 0.0
    outlier_rate: float = 0.0
    # 数据类型分布
    numerical_features: int = 0
    categorical_features: int = 0
    text_features: int = 0
    datetime_features: int = 0
    # 统计信息
    correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None
    class_distribution: Optional[Dict[str, int]] = None
    feature_importance: Optional[Dict[str, float]] = None


@dataclass
class AnalysisContext(BaseModel):
    """分析上下文"""
    # 研究目标
    primary_objective: ResearchObjective = ResearchObjective.CLASSIFICATION
    secondary_objectives: List[ResearchObjective] = field(default_factory=list)
    # 约束条件
    time_constraint: Optional[float] = None  # 时间限制（小时）
    computational_constraint: Optional[str] = None  # 计算资源限制
    interpretability_requirement: str = "medium"  # 可解释性要求
    accuracy_requirement: Optional[float] = None  # 准确性要求
    # 用户偏好
    preferred_methods: List[str] = field(default_factory=list)
    excluded_methods: List[str] = field(default_factory=list)
    # 环境信息
    available_libraries: List[str] = field(default_factory=list)
    hardware_specs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MethodRecommendation(BaseModel):
    """方法推荐"""
    method: AnalysisMethod = field(default_factory=AnalysisMethod)
    suitability_score: float = 0.0
    confidence: float = 0.0
    # 评估详情
    data_compatibility: float = 0.0
    objective_alignment: float = 0.0
    resource_feasibility: float = 0.0
    performance_expectation: float = 0.0
    # 推荐理由
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    # 参数建议
    suggested_parameters: Dict[str, Any] = field(default_factory=dict)
    parameter_explanations: Dict[str, str] = field(default_factory=dict)


@dataclass
class MethodCombination(BaseModel):
    """方法组合"""
    combination_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    methods: List[AnalysisMethod] = field(default_factory=list)
    execution_order: List[int] = field(default_factory=list)
    # 组合特性
    synergy_score: float = 0.0
    overall_complexity: str = "medium"
    expected_performance: float = 0.0
    # 数据流
    data_dependencies: Dict[int, List[int]] = field(default_factory=dict)
    output_types: Dict[int, str] = field(default_factory=dict)


@dataclass
class SelectionResult(BaseModel):
    """选择结果"""
    # 基本信息
    selection_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    # 输入信息
    data_profile: DataProfile = field(default_factory=DataProfile)
    analysis_context: AnalysisContext = field(default_factory=AnalysisContext)
    # 推荐结果
    single_method_recommendations: List[MethodRecommendation] = field(default_factory=list)
    combination_recommendations: List[MethodCombination] = field(default_factory=list)
    # 最终选择
    selected_method: Optional[AnalysisMethod] = None
    selected_combination: Optional[MethodCombination] = None
    selection_rationale: str = ""
    # 性能预测
    expected_accuracy: Optional[float] = None
    expected_runtime: Optional[float] = None
    confidence_level: float = 0.0


class AnalysisMethodSelector(BaseComponent):
    """分析方法选择器"""
    
    def get_required_configs(self) -> List[str]:
        """获取必需的配置项"""
        return []
    
    def _setup_component(self):
        """设置组件特定的初始化逻辑"""
        self.logger.info("分析方法选择器初始化")
        
        # 初始化方法库
        self.method_library = {}
        self._initialize_method_library()
        
        # 初始化评估器
        self.data_profiler = DataProfiler()
        self.compatibility_evaluator = CompatibilityEvaluator()
        self.performance_predictor = PerformancePredictor()
        self.combination_optimizer = CombinationOptimizer()
        
        self.logger.info("分析方法选择器初始化完成")
    
    def _initialize_method_library(self):
        """初始化方法库"""
        # 分类方法
        self._add_classification_methods()
        # 回归方法
        self._add_regression_methods()
        # 聚类方法
        self._add_clustering_methods()
        # 统计分析方法
        self._add_statistical_methods()
        # 时间序列方法
        self._add_time_series_methods()
        # 文本分析方法
        self._add_text_analysis_methods()
    
    def _add_classification_methods(self):
        """添加分类方法"""
        methods = [
            AnalysisMethod(
                name="logistic_regression",
                category="classification",
                description="逻辑回归分类器",
                suitable_data_types=[DataCharacteristic.NUMERICAL, DataCharacteristic.MIXED],
                suitable_objectives=[ResearchObjective.CLASSIFICATION],
                suitable_sample_sizes=(100, 100000),
                complexity="low",
                interpretability="high",
                computational_cost="low",
                required_libraries=["sklearn"],
                default_parameters={"C": 1.0, "max_iter": 1000},
                parameter_ranges={"C": (0.001, 100), "max_iter": (100, 10000)}
            ),
            AnalysisMethod(
                name="random_forest",
                category="classification",
                description="随机森林分类器",
                suitable_data_types=[DataCharacteristic.NUMERICAL, DataCharacteristic.CATEGORICAL, DataCharacteristic.MIXED],
                suitable_objectives=[ResearchObjective.CLASSIFICATION],
                suitable_sample_sizes=(500, 1000000),
                complexity="medium",
                interpretability="medium",
                computational_cost="medium",
                required_libraries=["sklearn"],
                default_parameters={"n_estimators": 100, "max_depth": None},
                parameter_ranges={"n_estimators": (10, 1000), "max_depth": (1, 50)}
            ),
            AnalysisMethod(
                name="svm",
                category="classification",
                description="支持向量机分类器",
                suitable_data_types=[DataCharacteristic.NUMERICAL, DataCharacteristic.HIGH_DIMENSIONAL],
                suitable_objectives=[ResearchObjective.CLASSIFICATION],
                suitable_sample_sizes=(100, 50000),
                complexity="high",
                interpretability="low",
                computational_cost="high",
                required_libraries=["sklearn"],
                default_parameters={"C": 1.0, "kernel": "rbf"},
                parameter_ranges={"C": (0.001, 100), "gamma": (0.001, 10)}
            ),
            AnalysisMethod(
                name="naive_bayes",
                category="classification",
                description="朴素贝叶斯分类器",
                suitable_data_types=[DataCharacteristic.CATEGORICAL, DataCharacteristic.TEXT],
                suitable_objectives=[ResearchObjective.CLASSIFICATION],
                suitable_sample_sizes=(50, 1000000),
                complexity="low",
                interpretability="high",
                computational_cost="low",
                required_libraries=["sklearn"],
                default_parameters={"alpha": 1.0},
                parameter_ranges={"alpha": (0.01, 10)}
            )
        ]
        
        for method in methods:
            self.method_library[method.name] = method
    
    def _add_regression_methods(self):
        """添加回归方法"""
        methods = [
            AnalysisMethod(
                name="linear_regression",
                category="regression",
                description="线性回归",
                suitable_data_types=[DataCharacteristic.NUMERICAL],
                suitable_objectives=[ResearchObjective.REGRESSION],
                suitable_sample_sizes=(30, 100000),
                complexity="low",
                interpretability="high",
                computational_cost="low",
                required_libraries=["sklearn"],
                default_parameters={"fit_intercept": True},
                parameter_ranges={}
            ),
            AnalysisMethod(
                name="ridge_regression",
                category="regression",
                description="岭回归",
                suitable_data_types=[DataCharacteristic.NUMERICAL, DataCharacteristic.HIGH_DIMENSIONAL],
                suitable_objectives=[ResearchObjective.REGRESSION],
                suitable_sample_sizes=(50, 100000),
                complexity="low",
                interpretability="high",
                computational_cost="low",
                required_libraries=["sklearn"],
                default_parameters={"alpha": 1.0},
                parameter_ranges={"alpha": (0.001, 100)}
            ),
            AnalysisMethod(
                name="random_forest_regressor",
                category="regression",
                description="随机森林回归器",
                suitable_data_types=[DataCharacteristic.NUMERICAL, DataCharacteristic.MIXED],
                suitable_objectives=[ResearchObjective.REGRESSION],
                suitable_sample_sizes=(500, 1000000),
                complexity="medium",
                interpretability="medium",
                computational_cost="medium",
                required_libraries=["sklearn"],
                default_parameters={"n_estimators": 100, "max_depth": None},
                parameter_ranges={"n_estimators": (10, 1000), "max_depth": (1, 50)}
            )
        ]
        
        for method in methods:
            self.method_library[method.name] = method
    
    def _add_clustering_methods(self):
        """添加聚类方法"""
        methods = [
            AnalysisMethod(
                name="kmeans",
                category="clustering",
                description="K均值聚类",
                suitable_data_types=[DataCharacteristic.NUMERICAL],
                suitable_objectives=[ResearchObjective.CLUSTERING],
                suitable_sample_sizes=(100, 1000000),
                complexity="low",
                interpretability="medium",
                computational_cost="low",
                required_libraries=["sklearn"],
                default_parameters={"n_clusters": 3, "random_state": 42},
                parameter_ranges={"n_clusters": (2, 20)}
            ),
            AnalysisMethod(
                name="hierarchical_clustering",
                category="clustering",
                description="层次聚类",
                suitable_data_types=[DataCharacteristic.NUMERICAL, DataCharacteristic.MIXED],
                suitable_objectives=[ResearchObjective.CLUSTERING],
                suitable_sample_sizes=(50, 10000),
                complexity="medium",
                interpretability="high",
                computational_cost="high",
                required_libraries=["sklearn"],
                default_parameters={"linkage": "ward"},
                parameter_ranges={}
            ),
            AnalysisMethod(
                name="dbscan",
                category="clustering",
                description="DBSCAN密度聚类",
                suitable_data_types=[DataCharacteristic.NUMERICAL, DataCharacteristic.SPARSE],
                suitable_objectives=[ResearchObjective.CLUSTERING, ResearchObjective.ANOMALY_DETECTION],
                suitable_sample_sizes=(100, 100000),
                complexity="medium",
                interpretability="medium",
                computational_cost="medium",
                required_libraries=["sklearn"],
                default_parameters={"eps": 0.5, "min_samples": 5},
                parameter_ranges={"eps": (0.1, 2.0), "min_samples": (2, 20)}
            )
        ]
        
        for method in methods:
            self.method_library[method.name] = method  
  
    def _add_statistical_methods(self):
        """添加统计分析方法"""
        methods = [
            AnalysisMethod(
                name="t_test",
                category="statistical",
                description="t检验",
                suitable_data_types=[DataCharacteristic.NUMERICAL],
                suitable_objectives=[ResearchObjective.HYPOTHESIS_TESTING, ResearchObjective.COMPARISON],
                suitable_sample_sizes=(10, 10000),
                complexity="low",
                interpretability="high",
                computational_cost="low",
                required_libraries=["scipy"],
                default_parameters={"alternative": "two-sided"},
                parameter_ranges={}
            ),
            AnalysisMethod(
                name="chi_square_test",
                category="statistical",
                description="卡方检验",
                suitable_data_types=[DataCharacteristic.CATEGORICAL],
                suitable_objectives=[ResearchObjective.HYPOTHESIS_TESTING, ResearchObjective.ASSOCIATION],
                suitable_sample_sizes=(20, 100000),
                complexity="low",
                interpretability="high",
                computational_cost="low",
                required_libraries=["scipy"],
                default_parameters={},
                parameter_ranges={}
            ),
            AnalysisMethod(
                name="anova",
                category="statistical",
                description="方差分析",
                suitable_data_types=[DataCharacteristic.NUMERICAL, DataCharacteristic.MIXED],
                suitable_objectives=[ResearchObjective.HYPOTHESIS_TESTING, ResearchObjective.COMPARISON],
                suitable_sample_sizes=(30, 10000),
                complexity="medium",
                interpretability="high",
                computational_cost="low",
                required_libraries=["scipy"],
                default_parameters={},
                parameter_ranges={}
            ),
            AnalysisMethod(
                name="correlation_analysis",
                category="statistical",
                description="相关性分析",
                suitable_data_types=[DataCharacteristic.NUMERICAL],
                suitable_objectives=[ResearchObjective.CORRELATION_ANALYSIS],
                suitable_sample_sizes=(10, 100000),
                complexity="low",
                interpretability="high",
                computational_cost="low",
                required_libraries=["scipy", "pandas"],
                default_parameters={"method": "pearson"},
                parameter_ranges={}
            )
        ]
        
        for method in methods:
            self.method_library[method.name] = method
    
    def _add_time_series_methods(self):
        """添加时间序列方法"""
        methods = [
            AnalysisMethod(
                name="arima",
                category="time_series",
                description="ARIMA时间序列模型",
                suitable_data_types=[DataCharacteristic.TIME_SERIES],
                suitable_objectives=[ResearchObjective.TREND_ANALYSIS, ResearchObjective.PREDICTIVE],
                suitable_sample_sizes=(50, 10000),
                complexity="high",
                interpretability="medium",
                computational_cost="medium",
                required_libraries=["statsmodels"],
                default_parameters={"order": (1, 1, 1)},
                parameter_ranges={}
            ),
            AnalysisMethod(
                name="seasonal_decompose",
                category="time_series",
                description="季节性分解",
                suitable_data_types=[DataCharacteristic.TIME_SERIES],
                suitable_objectives=[ResearchObjective.TREND_ANALYSIS],
                suitable_sample_sizes=(24, 10000),
                complexity="low",
                interpretability="high",
                computational_cost="low",
                required_libraries=["statsmodels"],
                default_parameters={"model": "additive"},
                parameter_ranges={}
            ),
            AnalysisMethod(
                name="exponential_smoothing",
                category="time_series",
                description="指数平滑",
                suitable_data_types=[DataCharacteristic.TIME_SERIES],
                suitable_objectives=[ResearchObjective.TREND_ANALYSIS, ResearchObjective.PREDICTIVE],
                suitable_sample_sizes=(20, 10000),
                complexity="medium",
                interpretability="medium",
                computational_cost="low",
                required_libraries=["statsmodels"],
                default_parameters={"trend": "add", "seasonal": "add"},
                parameter_ranges={}
            )
        ]
        
        for method in methods:
            self.method_library[method.name] = method
    
    def _add_text_analysis_methods(self):
        """添加文本分析方法"""
        methods = [
            AnalysisMethod(
                name="tfidf_vectorization",
                category="text_analysis",
                description="TF-IDF向量化",
                suitable_data_types=[DataCharacteristic.TEXT],
                suitable_objectives=[ResearchObjective.DIMENSIONALITY_REDUCTION],
                suitable_sample_sizes=(10, 1000000),
                complexity="low",
                interpretability="medium",
                computational_cost="low",
                required_libraries=["sklearn"],
                default_parameters={"max_features": 1000, "stop_words": "english"},
                parameter_ranges={"max_features": (100, 10000)}
            ),
            AnalysisMethod(
                name="sentiment_analysis",
                category="text_analysis",
                description="情感分析",
                suitable_data_types=[DataCharacteristic.TEXT],
                suitable_objectives=[ResearchObjective.CLASSIFICATION],
                suitable_sample_sizes=(100, 1000000),
                complexity="medium",
                interpretability="medium",
                computational_cost="medium",
                required_libraries=["textblob", "nltk"],
                default_parameters={},
                parameter_ranges={}
            ),
            AnalysisMethod(
                name="topic_modeling",
                category="text_analysis",
                description="主题建模",
                suitable_data_types=[DataCharacteristic.TEXT],
                suitable_objectives=[ResearchObjective.CLUSTERING, ResearchObjective.DIMENSIONALITY_REDUCTION],
                suitable_sample_sizes=(100, 1000000),
                complexity="high",
                interpretability="medium",
                computational_cost="high",
                required_libraries=["sklearn", "gensim"],
                default_parameters={"n_topics": 5},
                parameter_ranges={"n_topics": (2, 50)}
            )
        ]
        
        for method in methods:
            self.method_library[method.name] = method
    
    def select_methods(self, data: Any, context: AnalysisContext) -> SelectionResult:
        """
        选择分析方法
        Args:
            data: 输入数据
            context: 分析上下文
        Returns:
            选择结果
        """
        try:
            self.logger.info("开始选择分析方法")
            
            # 分析数据概况
            data_profile = self.data_profiler.profile_data(data)
            
            # 评估方法兼容性
            method_recommendations = []
            for method_name, method in self.method_library.items():
                if method_name in context.excluded_methods:
                    continue
                
                recommendation = self.compatibility_evaluator.evaluate_method(
                    method, data_profile, context
                )
                
                if recommendation.suitability_score > 0.3:  # 过滤掉不太适合的方法
                    method_recommendations.append(recommendation)
            
            # 按适用性评分排序
            method_recommendations.sort(key=lambda x: x.suitability_score, reverse=True)
            
            # 生成方法组合
            combination_recommendations = self.combination_optimizer.generate_combinations(
                method_recommendations[:5], data_profile, context
            )
            
            # 创建选择结果
            result = SelectionResult(
                data_profile=data_profile,
                analysis_context=context,
                single_method_recommendations=method_recommendations,
                combination_recommendations=combination_recommendations
            )
            
            # 选择最佳方法或组合
            if method_recommendations:
                result.selected_method = method_recommendations[0].method
                result.selection_rationale = f"选择 {result.selected_method.name}，适用性评分: {method_recommendations[0].suitability_score:.3f}"
                
                # 预测性能
                result.expected_accuracy = self.performance_predictor.predict_accuracy(
                    result.selected_method, data_profile
                )
                result.expected_runtime = self.performance_predictor.predict_runtime(
                    result.selected_method, data_profile
                )
                result.confidence_level = method_recommendations[0].confidence
            
            self.logger.info(f"方法选择完成，推荐方法: {result.selected_method.name if result.selected_method else 'None'}")
            return result
            
        except Exception as e:
            self.logger.error(f"方法选择失败: {str(e)}")
            raise
    
    def get_method_by_name(self, method_name: str) -> Optional[AnalysisMethod]:
        """根据名称获取方法"""
        return self.method_library.get(method_name)
    
    def get_methods_by_category(self, category: str) -> List[AnalysisMethod]:
        """根据类别获取方法"""
        return [method for method in self.method_library.values() if method.category == category]
    
    def get_methods_by_objective(self, objective: ResearchObjective) -> List[AnalysisMethod]:
        """根据研究目标获取方法"""
        return [method for method in self.method_library.values() if objective in method.suitable_objectives]
    
    def update_method_performance(self, method_name: str, accuracy: float, runtime: float, success: bool):
        """更新方法性能统计"""
        if method_name in self.method_library:
            method = self.method_library[method_name]
            method.usage_count += 1
            
            if success:
                # 更新成功率
                method.success_rate = (method.success_rate * (method.usage_count - 1) + 1) / method.usage_count
                # 更新平均性能
                method.average_performance = (method.average_performance * (method.usage_count - 1) + accuracy) / method.usage_count
            else:
                method.success_rate = (method.success_rate * (method.usage_count - 1)) / method.usage_count
    
    def add_custom_method(self, method: AnalysisMethod):
        """添加自定义方法"""
        self.method_library[method.name] = method
        self.logger.info(f"添加自定义方法: {method.name}")
    
    def remove_method(self, method_name: str) -> bool:
        """移除方法"""
        if method_name in self.method_library:
            del self.method_library[method_name]
            self.logger.info(f"移除方法: {method_name}")
            return True
        return False


class DataProfiler:
    """数据概况分析器"""
    
    def profile_data(self, data: Any) -> DataProfile:
        """分析数据概况"""
        profile = DataProfile()
        
        if HAS_PANDAS and isinstance(data, pd.DataFrame):
            return self._profile_dataframe(data)
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            return self._profile_dict_list(data)
        else:
            # 简单数据类型
            profile.sample_size = len(data) if hasattr(data, '__len__') else 1
            profile.feature_count = 1
            profile.data_characteristics = [DataCharacteristic.UNKNOWN]
            return profile
    
    def _profile_dataframe(self, df: pd.DataFrame) -> DataProfile:
        """分析DataFrame"""
        profile = DataProfile()
        
        # 基本信息
        profile.sample_size = len(df)
        profile.feature_count = len(df.columns)
        
        # 缺失值统计
        profile.missing_rate = df.isnull().sum().sum() / (len(df) * len(df.columns))
        
        # 数据类型统计
        profile.numerical_features = len(df.select_dtypes(include=['number']).columns)
        profile.categorical_features = len(df.select_dtypes(include=['object', 'category']).columns)
        profile.datetime_features = len(df.select_dtypes(include=['datetime']).columns)
        
        # 数据特征识别
        characteristics = []
        if profile.numerical_features > 0:
            characteristics.append(DataCharacteristic.NUMERICAL)
        if profile.categorical_features > 0:
            characteristics.append(DataCharacteristic.CATEGORICAL)
        if profile.datetime_features > 0:
            characteristics.append(DataCharacteristic.TIME_SERIES)
        if profile.feature_count > 100:
            characteristics.append(DataCharacteristic.HIGH_DIMENSIONAL)
        if profile.missing_rate > 0.5:
            characteristics.append(DataCharacteristic.SPARSE)
        
        profile.data_characteristics = characteristics
        
        # 相关性矩阵
        if profile.numerical_features > 1:
            try:
                numeric_df = df.select_dtypes(include=['number'])
                corr_matrix = numeric_df.corr()
                profile.correlation_matrix = corr_matrix.to_dict()
            except:
                pass
        
        return profile
    
    def _profile_dict_list(self, data: List[Dict[str, Any]]) -> DataProfile:
        """分析字典列表"""
        profile = DataProfile()
        
        if not data:
            return profile
        
        # 基本信息
        profile.sample_size = len(data)
        
        # 获取所有键
        all_keys = set()
        for item in data:
            all_keys.update(item.keys())
        
        profile.feature_count = len(all_keys)
        
        # 分析数据类型
        for key in all_keys:
            values = [item.get(key) for item in data if key in item and item[key] is not None]
            if not values:
                continue
            
            sample_value = values[0]
            if isinstance(sample_value, (int, float)):
                profile.numerical_features += 1
            elif isinstance(sample_value, str):
                if self._is_datetime_string(sample_value):
                    profile.datetime_features += 1
                else:
                    profile.categorical_features += 1
        
        # 计算缺失率
        total_cells = profile.sample_size * profile.feature_count
        missing_cells = sum(1 for item in data for key in all_keys if key not in item or item[key] is None)
        profile.missing_rate = missing_cells / total_cells if total_cells > 0 else 0
        
        # 数据特征识别
        characteristics = []
        if profile.numerical_features > 0:
            characteristics.append(DataCharacteristic.NUMERICAL)
        if profile.categorical_features > 0:
            characteristics.append(DataCharacteristic.CATEGORICAL)
        if profile.datetime_features > 0:
            characteristics.append(DataCharacteristic.TIME_SERIES)
        if profile.feature_count > 100:
            characteristics.append(DataCharacteristic.HIGH_DIMENSIONAL)
        if profile.missing_rate > 0.5:
            characteristics.append(DataCharacteristic.SPARSE)
        
        profile.data_characteristics = characteristics
        
        return profile
    
    def _is_datetime_string(self, text: str) -> bool:
        """检查字符串是否为日期时间格式"""
        import re
        patterns = [
            r'^\d{4}-\d{2}-\d{2}$',
            r'^\d{4}/\d{2}/\d{2}$',
            r'^\d{2}-\d{2}-\d{4}$',
            r'^\d{2}/\d{2}/\d{4}$',
            r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$'
        ]
        return any(re.match(pattern, text) for pattern in patterns)


class CompatibilityEvaluator:
    """兼容性评估器"""
    
    def evaluate_method(self, method: AnalysisMethod, data_profile: DataProfile, context: AnalysisContext) -> MethodRecommendation:
        """评估方法兼容性"""
        recommendation = MethodRecommendation(method=method)
        
        # 数据兼容性评估
        recommendation.data_compatibility = self._evaluate_data_compatibility(method, data_profile)
        
        # 目标对齐评估
        recommendation.objective_alignment = self._evaluate_objective_alignment(method, context)
        
        # 资源可行性评估
        recommendation.resource_feasibility = self._evaluate_resource_feasibility(method, context)
        
        # 性能期望评估
        recommendation.performance_expectation = self._evaluate_performance_expectation(method, data_profile)
        
        # 计算总体适用性评分
        weights = {
            'data_compatibility': 0.4,
            'objective_alignment': 0.3,
            'resource_feasibility': 0.2,
            'performance_expectation': 0.1
        }
        
        recommendation.suitability_score = (
            weights['data_compatibility'] * recommendation.data_compatibility +
            weights['objective_alignment'] * recommendation.objective_alignment +
            weights['resource_feasibility'] * recommendation.resource_feasibility +
            weights['performance_expectation'] * recommendation.performance_expectation
        )
        
        # 计算置信度
        recommendation.confidence = min(recommendation.suitability_score, method.success_rate)
        
        # 生成推荐理由
        recommendation.reasons = self._generate_reasons(method, data_profile, context, recommendation)
        
        # 生成警告
        recommendation.warnings = self._generate_warnings(method, data_profile, context)
        
        # 建议参数
        recommendation.suggested_parameters = self._suggest_parameters(method, data_profile)
        
        return recommendation
    
    def _evaluate_data_compatibility(self, method: AnalysisMethod, data_profile: DataProfile) -> float:
        """评估数据兼容性"""
        score = 0.0
        
        # 检查数据类型兼容性
        compatible_types = 0
        for data_char in data_profile.data_characteristics:
            if data_char in method.suitable_data_types:
                compatible_types += 1
        
        if data_profile.data_characteristics:
            score += 0.4 * (compatible_types / len(data_profile.data_characteristics))
        
        # 检查样本大小
        min_size, max_size = method.suitable_sample_sizes
        if min_size <= data_profile.sample_size <= max_size:
            score += 0.3
        elif data_profile.sample_size < min_size:
            score += 0.1 * (data_profile.sample_size / min_size)
        else:  # data_profile.sample_size > max_size
            score += 0.1
        
        # 检查特征数量
        if data_profile.feature_count > 0:
            if data_profile.feature_count <= 10:
                score += 0.2
            elif data_profile.feature_count <= 100:
                score += 0.15
            else:
                score += 0.1
        
        # 检查缺失值容忍度
        if data_profile.missing_rate <= 0.1:
            score += 0.1
        elif data_profile.missing_rate <= 0.3:
            score += 0.05
        
        return min(score, 1.0)
    
    def _evaluate_objective_alignment(self, method: AnalysisMethod, context: AnalysisContext) -> float:
        """评估目标对齐度"""
        score = 0.0
        
        # 主要目标匹配
        if context.primary_objective in method.suitable_objectives:
            score += 0.7
        
        # 次要目标匹配
        if context.secondary_objectives:
            matched_secondary = sum(1 for obj in context.secondary_objectives if obj in method.suitable_objectives)
            score += 0.3 * (matched_secondary / len(context.secondary_objectives))
        
        return min(score, 1.0)
    
    def _evaluate_resource_feasibility(self, method: AnalysisMethod, context: AnalysisContext) -> float:
        """评估资源可行性"""
        score = 1.0
        
        # 检查库依赖
        if context.available_libraries:
            missing_libs = set(method.required_libraries) - set(context.available_libraries)
            if missing_libs:
                score -= 0.3
        
        # 检查计算成本
        if context.computational_constraint:
            if method.computational_cost == "high" and context.computational_constraint == "low":
                score -= 0.4
            elif method.computational_cost == "medium" and context.computational_constraint == "low":
                score -= 0.2
        
        # 检查可解释性要求
        interpretability_map = {"low": 0, "medium": 1, "high": 2}
        method_interp = interpretability_map.get(method.interpretability, 1)
        required_interp = interpretability_map.get(context.interpretability_requirement, 1)
        
        if method_interp < required_interp:
            score -= 0.3
        
        return max(score, 0.0)
    
    def _evaluate_performance_expectation(self, method: AnalysisMethod, data_profile: DataProfile) -> float:
        """评估性能期望"""
        score = 0.5  # 基础分数
        
        # 基于历史性能
        if method.usage_count > 0:
            score = method.average_performance
        
        # 基于典型准确性
        if method.typical_accuracy:
            score = max(score, method.typical_accuracy)
        
        return min(score, 1.0)
    
    def _generate_reasons(self, method: AnalysisMethod, data_profile: DataProfile, context: AnalysisContext, recommendation: MethodRecommendation) -> List[str]:
        """生成推荐理由"""
        reasons = []
        
        if recommendation.data_compatibility > 0.7:
            reasons.append(f"数据类型与方法高度兼容")
        
        if context.primary_objective in method.suitable_objectives:
            reasons.append(f"完全符合主要研究目标")
        
        if method.interpretability == "high" and context.interpretability_requirement == "high":
            reasons.append(f"具有高可解释性")
        
        if method.computational_cost == "low":
            reasons.append(f"计算成本较低")
        
        if method.success_rate > 0.8:
            reasons.append(f"历史成功率较高 ({method.success_rate:.1%})")
        
        return reasons
    
    def _generate_warnings(self, method: AnalysisMethod, data_profile: DataProfile, context: AnalysisContext) -> List[str]:
        """生成警告"""
        warnings = []
        
        min_size, max_size = method.suitable_sample_sizes
        if data_profile.sample_size < min_size:
            warnings.append(f"样本量可能不足，建议至少 {min_size} 个样本")
        elif data_profile.sample_size > max_size:
            warnings.append(f"样本量过大，可能影响性能")
        
        if data_profile.missing_rate > 0.3:
            warnings.append(f"数据缺失率较高 ({data_profile.missing_rate:.1%})，建议先进行数据清洗")
        
        missing_libs = set(method.required_libraries) - set(context.available_libraries)
        if missing_libs:
            warnings.append(f"需要安装依赖库: {', '.join(missing_libs)}")
        
        if method.computational_cost == "high" and context.time_constraint and context.time_constraint < 1:
            warnings.append(f"计算成本较高，可能超出时间限制")
        
        return warnings
    
    def _suggest_parameters(self, method: AnalysisMethod, data_profile: DataProfile) -> Dict[str, Any]:
        """建议参数"""
        suggested = method.default_parameters.copy()
        
        # 根据数据特征调整参数
        if method.name == "kmeans" and data_profile.sample_size > 0:
            # 根据样本大小建议聚类数
            suggested_k = min(max(2, int(data_profile.sample_size ** 0.5 / 10)), 20)
            suggested["n_clusters"] = suggested_k
        
        elif method.name == "random_forest":
            # 根据特征数量调整树的数量
            if data_profile.feature_count > 50:
                suggested["n_estimators"] = 200
            elif data_profile.feature_count < 10:
                suggested["n_estimators"] = 50
        
        return suggested


class PerformancePredictor:
    """性能预测器"""
    
    def predict_accuracy(self, method: AnalysisMethod, data_profile: DataProfile) -> float:
        """预测准确性"""
        base_accuracy = method.typical_accuracy or 0.7
        
        # 根据数据特征调整
        if data_profile.missing_rate > 0.3:
            base_accuracy *= 0.9
        
        if data_profile.sample_size < 100:
            base_accuracy *= 0.8
        elif data_profile.sample_size > 10000:
            base_accuracy *= 1.1
        
        # 根据历史性能调整
        if method.usage_count > 0:
            base_accuracy = 0.7 * base_accuracy + 0.3 * method.average_performance
        
        return min(base_accuracy, 1.0)
    
    def predict_runtime(self, method: AnalysisMethod, data_profile: DataProfile) -> float:
        """预测运行时间（分钟）"""
        base_time = 1.0  # 基础时间1分钟
        
        # 根据计算复杂度调整
        complexity_multiplier = {"low": 0.5, "medium": 1.0, "high": 3.0}
        base_time *= complexity_multiplier.get(method.computational_cost, 1.0)
        
        # 根据数据大小调整
        if data_profile.sample_size > 10000:
            base_time *= (data_profile.sample_size / 10000) ** 0.5
        
        if data_profile.feature_count > 100:
            base_time *= (data_profile.feature_count / 100) ** 0.3
        
        return base_time


class CombinationOptimizer:
    """组合优化器"""
    
    def generate_combinations(self, recommendations: List[MethodRecommendation], data_profile: DataProfile, context: AnalysisContext) -> List[MethodCombination]:
        """生成方法组合"""
        combinations = []
        
        if len(recommendations) < 2:
            return combinations
        
        # 生成简单的两方法组合
        for i in range(len(recommendations) - 1):
            for j in range(i + 1, min(i + 3, len(recommendations))):  # 限制组合数量
                method1 = recommendations[i].method
                method2 = recommendations[j].method
                
                # 检查组合的合理性
                if self._is_compatible_combination(method1, method2):
                    combination = MethodCombination(
                        name=f"{method1.name} + {method2.name}",
                        description=f"结合 {method1.description} 和 {method2.description}",
                        methods=[method1, method2],
                        execution_order=[0, 1]
                    )
                    
                    # 计算协同评分
                    combination.synergy_score = self._calculate_synergy_score(method1, method2, data_profile)
                    
                    # 计算整体复杂度
                    combination.overall_complexity = self._calculate_overall_complexity(method1, method2)
                    
                    # 预测整体性能
                    combination.expected_performance = self._predict_combination_performance(
                        recommendations[i], recommendations[j]
                    )
                    
                    combinations.append(combination)
        
        # 按协同评分排序
        combinations.sort(key=lambda x: x.synergy_score, reverse=True)
        
        return combinations[:3]  # 返回前3个组合
    
    def _is_compatible_combination(self, method1: AnalysisMethod, method2: AnalysisMethod) -> bool:
        """检查方法组合的兼容性"""
        # 避免相同类别的方法组合
        if method1.category == method2.category:
            return False
        
        # 检查是否有互补的目标
        common_objectives = set(method1.suitable_objectives) & set(method2.suitable_objectives)
        if not common_objectives:
            return False
        
        return True
    
    def _calculate_synergy_score(self, method1: AnalysisMethod, method2: AnalysisMethod, data_profile: DataProfile) -> float:
        """计算协同评分"""
        score = 0.0
        
        # 互补性评分
        if method1.interpretability == "high" and method2.interpretability == "low":
            score += 0.3  # 高解释性 + 高性能
        
        if method1.computational_cost == "low" and method2.computational_cost == "high":
            score += 0.2  # 快速筛选 + 精确分析
        
        # 目标覆盖评分
        combined_objectives = set(method1.suitable_objectives) | set(method2.suitable_objectives)
        score += 0.3 * (len(combined_objectives) / 10)  # 假设最多10个目标
        
        # 数据类型覆盖评分
        combined_data_types = set(method1.suitable_data_types) | set(method2.suitable_data_types)
        score += 0.2 * (len(combined_data_types) / 8)  # 假设最多8种数据类型
        
        return min(score, 1.0)
    
    def _calculate_overall_complexity(self, method1: AnalysisMethod, method2: AnalysisMethod) -> str:
        """计算整体复杂度"""
        complexity_map = {"low": 1, "medium": 2, "high": 3}
        
        total_complexity = complexity_map.get(method1.complexity, 2) + complexity_map.get(method2.complexity, 2)
        
        if total_complexity <= 2:
            return "low"
        elif total_complexity <= 4:
            return "medium"
        else:
            return "high"
    
    def _predict_combination_performance(self, rec1: MethodRecommendation, rec2: MethodRecommendation) -> float:
        """预测组合性能"""
        # 简单的加权平均
        return 0.6 * max(rec1.suitability_score, rec2.suitability_score) + 0.4 * min(rec1.suitability_score, rec2.suitability_score)