"""
数据预处理引擎测试

测试数据清洗、转换、标准化和特征工程功能
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_automation.core.data_preprocessing_engine import (
    DataPreprocessingEngine, 
    PreprocessingStep, 
    PreprocessingStepType,
    DataType,
    ColumnProfile,
    DataProfile,
    PreprocessingResult
)


class TestDataPreprocessingEngine(unittest.TestCase):
    """数据预处理引擎测试类"""
    
    def setUp(self):
        """设置测试环境"""
        self.config = {}
        self.engine = DataPreprocessingEngine(self.config)
        
        # 创建测试数据
        self.test_df = pd.DataFrame({
            'numeric_col': [1, 2, 3, 4, 5, None, 7, 8, 9, 100],
            'text_col': ['Hello', 'World', 'Test', 'Data', None, 'Processing', 'Engine', 'AI', 'ML', 'NLP'],
            'category_col': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'A', 'B', 'C'],
            'duplicate_col': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        })
        
        self.test_dict_list = [
            {'id': 1, 'name': 'Alice', 'age': 25, 'score': 85.5},
            {'id': 2, 'name': 'Bob', 'age': 30, 'score': 92.0},
            {'id': 3, 'name': 'Charlie', 'age': None, 'score': 78.5},
            {'id': 4, 'name': 'David', 'age': 28, 'score': None},
            {'id': 1, 'name': 'Alice', 'age': 25, 'score': 85.5}  # 重复行
        ]
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsInstance(self.engine, DataPreprocessingEngine)
        self.assertEqual(len(self.engine.pipeline), 0)
        self.assertIsInstance(self.engine.preprocessing_functions, dict)
        self.assertGreater(len(self.engine.preprocessing_functions), 0)
    
    def test_add_preprocessing_step(self):
        """测试添加预处理步骤"""
        step = PreprocessingStep(
            name="测试步骤",
            step_type=PreprocessingStepType.CLEANING,
            description="测试描述",
            target_columns=['numeric_col'],
            parameters={'function': 'remove_duplicates'}
        )
        
        step_id = self.engine.add_preprocessing_step(step)
        
        self.assertEqual(len(self.engine.pipeline), 1)
        self.assertEqual(self.engine.pipeline[0].step_id, step_id)
        self.assertEqual(self.engine.pipeline[0].name, "测试步骤")
    
    def test_remove_preprocessing_step(self):
        """测试移除预处理步骤"""
        step = PreprocessingStep(
            name="测试步骤",
            parameters={'function': 'remove_duplicates'}
        )
        
        step_id = self.engine.add_preprocessing_step(step)
        self.assertEqual(len(self.engine.pipeline), 1)
        
        success = self.engine.remove_preprocessing_step(step_id)
        self.assertTrue(success)
        self.assertEqual(len(self.engine.pipeline), 0)
        
        # 测试移除不存在的步骤
        success = self.engine.remove_preprocessing_step("non_existent_id")
        self.assertFalse(success)
    
    def test_update_preprocessing_step(self):
        """测试更新预处理步骤"""
        step = PreprocessingStep(
            name="原始名称",
            parameters={'function': 'remove_duplicates'}
        )
        
        step_id = self.engine.add_preprocessing_step(step)
        
        updates = {
            'name': '更新后的名称',
            'description': '新的描述'
        }
        
        success = self.engine.update_preprocessing_step(step_id, updates)
        self.assertTrue(success)
        
        updated_step = self.engine.get_preprocessing_step(step_id)
        self.assertEqual(updated_step.name, '更新后的名称')
        self.assertEqual(updated_step.description, '新的描述')
    
    def test_data_profiling(self):
        """测试数据概况分析"""
        profile = self.engine._profile_data(self.test_df)
        
        self.assertIsInstance(profile, DataProfile)
        self.assertEqual(profile.row_count, 10)
        self.assertEqual(profile.column_count, 4)
        self.assertGreater(len(profile.columns), 0)
        
        # 检查列概况
        numeric_profile = profile.columns.get('numeric_col')
        self.assertIsNotNone(numeric_profile)
        self.assertEqual(numeric_profile.data_type, DataType.NUMERIC)
        self.assertEqual(numeric_profile.missing_count, 1)
    
    def test_column_profiling(self):
        """测试列数据概况分析"""
        numeric_column = self.test_df['numeric_col']
        profile = self.engine._profile_column(numeric_column)
        
        self.assertIsInstance(profile, ColumnProfile)
        self.assertEqual(profile.data_type, DataType.NUMERIC)
        self.assertEqual(profile.count, 9)  # 非空值数量
        self.assertEqual(profile.missing_count, 1)
        self.assertIsNotNone(profile.min_value)
        self.assertIsNotNone(profile.max_value)
        self.assertIsNotNone(profile.mean)
    
    def test_data_type_inference(self):
        """测试数据类型推断"""
        # 测试数值型
        numeric_type = self.engine._infer_data_type(self.test_df['numeric_col'])
        self.assertEqual(numeric_type, DataType.NUMERIC)
        
        # 测试文本型
        text_type = self.engine._infer_data_type(self.test_df['text_col'])
        self.assertEqual(text_type, DataType.TEXT)
        
        # 测试类别型
        category_type = self.engine._infer_data_type(self.test_df['category_col'])
        self.assertEqual(category_type, DataType.CATEGORICAL)
    
    def test_remove_duplicates(self):
        """测试移除重复行"""
        # 测试DataFrame
        result = self.engine._remove_duplicates(self.test_df, [], {})
        self.assertLess(len(result), len(self.test_df))
        
        # 测试字典列表
        result = self.engine._remove_duplicates(self.test_dict_list, [], {})
        self.assertEqual(len(result), 4)  # 应该移除一个重复行
    
    def test_remove_missing(self):
        """测试移除缺失值"""
        # 测试DataFrame
        result = self.engine._remove_missing(self.test_df, [], {})
        self.assertLess(len(result), len(self.test_df))
        
        # 测试指定列
        result = self.engine._remove_missing(self.test_df, ['numeric_col'], {})
        self.assertEqual(len(result), 9)  # 应该移除一行
    
    def test_remove_outliers(self):
        """测试移除异常值"""
        # 测试Z-score方法
        result = self.engine._remove_outliers(
            self.test_df, 
            ['numeric_col'], 
            {'method': 'zscore', 'threshold': 2.0}
        )
        self.assertLessEqual(len(result), len(self.test_df))
        
        # 测试IQR方法
        result = self.engine._remove_outliers(
            self.test_df, 
            ['numeric_col'], 
            {'method': 'iqr', 'threshold': 1.5}
        )
        self.assertLessEqual(len(result), len(self.test_df))
    
    def test_fill_missing_mean(self):
        """测试用均值填充缺失值"""
        result = self.engine._fill_missing_mean(self.test_df.copy(), ['numeric_col'], {})
        
        # 检查是否还有缺失值
        missing_count = result['numeric_col'].isna().sum()
        self.assertEqual(missing_count, 0)
    
    def test_fill_missing_median(self):
        """测试用中位数填充缺失值"""
        result = self.engine._fill_missing_median(self.test_df.copy(), ['numeric_col'], {})
        
        # 检查是否还有缺失值
        missing_count = result['numeric_col'].isna().sum()
        self.assertEqual(missing_count, 0)
    
    def test_fill_missing_constant(self):
        """测试用常数填充缺失值"""
        result = self.engine._fill_missing_constant(
            self.test_df.copy(), 
            ['numeric_col'], 
            {'fill_value': -999}
        )
        
        # 检查是否还有缺失值
        missing_count = result['numeric_col'].isna().sum()
        self.assertEqual(missing_count, 0)
        
        # 检查填充值
        filled_values = result[result['numeric_col'] == -999]
        self.assertGreater(len(filled_values), 0)
    
    def test_min_max_scale(self):
        """测试最小-最大缩放"""
        result = self.engine._min_max_scale(
            self.test_df.copy(), 
            ['numeric_col'], 
            {'feature_range': (0, 1)}
        )
        
        # 检查缩放后的范围（忽略NaN值）
        scaled_values = result['numeric_col'].dropna()
        self.assertGreaterEqual(scaled_values.min(), 0)
        self.assertLessEqual(scaled_values.max(), 1)
    
    def test_z_score_normalize(self):
        """测试Z-score标准化"""
        result = self.engine._z_score_normalize(self.test_df.copy(), ['numeric_col'], {})
        
        # 检查标准化后的均值和标准差（忽略NaN值）
        normalized_values = result['numeric_col'].dropna()
        self.assertAlmostEqual(normalized_values.mean(), 0, places=10)
        self.assertAlmostEqual(normalized_values.std(), 1, places=10)
    
    def test_clean_text(self):
        """测试文本清洗"""
        test_df = pd.DataFrame({
            'text': ['Hello World!', 'Test 123', 'AI & ML', 'Data Science']
        })
        
        result = self.engine._clean_text(
            test_df.copy(), 
            ['text'], 
            {
                'lowercase': True,
                'remove_punctuation': True,
                'remove_numbers': True
            }
        )
        
        # 检查清洗效果
        cleaned_text = result['text'].iloc[0]
        self.assertEqual(cleaned_text, 'hello world')
    
    def test_one_hot_encode(self):
        """测试独热编码"""
        result = self.engine._one_hot_encode(self.test_df.copy(), ['category_col'], {})
        
        # 检查是否创建了新列
        original_columns = set(self.test_df.columns)
        new_columns = set(result.columns)
        added_columns = new_columns - original_columns
        
        self.assertGreater(len(added_columns), 0)
        self.assertNotIn('category_col', result.columns)  # 原列应该被删除
    
    def test_label_encode(self):
        """测试标签编码"""
        result = self.engine._label_encode(self.test_df.copy(), ['category_col'], {})
        
        # 检查编码结果
        encoded_values = result['category_col'].dropna().unique()
        self.assertTrue(all(isinstance(x, (int, np.integer)) for x in encoded_values))
    
    def test_preprocess_data_pipeline(self):
        """测试完整的预处理流水线"""
        # 添加多个预处理步骤
        steps = [
            PreprocessingStep(
                name="移除重复行",
                step_type=PreprocessingStepType.CLEANING,
                parameters={'function': 'remove_duplicates'},
                order=1
            ),
            PreprocessingStep(
                name="填充缺失值",
                step_type=PreprocessingStepType.IMPUTATION,
                target_columns=['numeric_col'],
                parameters={'function': 'fill_missing_mean'},
                order=2
            ),
            PreprocessingStep(
                name="标准化数值列",
                step_type=PreprocessingStepType.NORMALIZATION,
                target_columns=['numeric_col'],
                parameters={'function': 'z_score_normalize'},
                order=3
            )
        ]
        
        for step in steps:
            self.engine.add_preprocessing_step(step)
        
        # 执行预处理
        processed_data, result = self.engine.preprocess_data(self.test_df.copy())
        
        # 检查结果
        self.assertIsInstance(result, PreprocessingResult)
        self.assertTrue(result.success)
        self.assertEqual(len(result.steps_executed), 3)
        self.assertEqual(len(result.steps_failed), 0)
        
        # 检查数据变化
        self.assertIsNotNone(result.input_profile)
        self.assertIsNotNone(result.output_profile)
        
        # 检查缺失值是否被填充
        missing_count = processed_data['numeric_col'].isna().sum()
        self.assertEqual(missing_count, 0)
    
    def test_preprocess_data_empty_pipeline(self):
        """测试空流水线"""
        processed_data, result = self.engine.preprocess_data(self.test_df.copy())
        
        self.assertTrue(result.success)
        self.assertEqual(len(result.steps_executed), 0)
        self.assertIn("预处理流水线为空", result.error_message)
    
    def test_preprocess_data_with_error(self):
        """测试处理错误情况"""
        # 添加一个会失败的步骤
        step = PreprocessingStep(
            name="错误步骤",
            parameters={'function': 'non_existent_function'}
        )
        
        self.engine.add_preprocessing_step(step)
        
        processed_data, result = self.engine.preprocess_data(self.test_df.copy())
        
        self.assertFalse(result.success)
        self.assertEqual(len(result.steps_failed), 1)
        self.assertIn("未知的预处理函数", result.error_message)
    
    def test_step_execution_stats(self):
        """测试步骤执行统计"""
        step = PreprocessingStep(
            name="测试步骤",
            parameters={'function': 'remove_duplicates'}
        )
        
        # 模拟执行时间
        step.update_execution_stats(100.0)
        step.update_execution_stats(200.0)
        
        self.assertEqual(step.execution_count, 2)
        self.assertEqual(step.average_execution_time_ms, 150.0)
        self.assertIsNotNone(step.last_execution_time)
    
    def test_preprocessing_result_methods(self):
        """测试预处理结果的方法"""
        result = PreprocessingResult(
            input_rows=100,
            output_rows=90,
            added_columns=['new_col1', 'new_col2'],
            modified_columns=['col1'],
            removed_columns=['old_col']
        )
        
        # 测试行变化率
        change_rate = result.get_row_change_rate()
        self.assertEqual(change_rate, 0.1)  # (100-90)/100
        
        # 测试列变化汇总
        column_summary = result.get_column_change_summary()
        self.assertEqual(column_summary['added'], 2)
        self.assertEqual(column_summary['modified'], 1)
        self.assertEqual(column_summary['removed'], 1)
    
    def test_data_profile_methods(self):
        """测试数据概况的方法"""
        profile = DataProfile(
            row_count=100,
            column_count=5,
            missing_cells_count=10,
            duplicate_rows_count=5
        )
        
        # 测试缺失率
        missing_rate = profile.get_missing_rate()
        self.assertEqual(missing_rate, 0.02)  # 10/(100*5)
        
        # 测试重复率
        duplicate_rate = profile.get_duplicate_rate()
        self.assertEqual(duplicate_rate, 0.05)  # 5/100
    
    def test_column_profile_methods(self):
        """测试列概况的方法"""
        profile = ColumnProfile(
            count=100,
            missing_count=10,
            unique_count=80,
            anomaly_count=5
        )
        
        # 测试各种率
        self.assertEqual(profile.get_missing_rate(), 0.1)
        self.assertEqual(profile.get_unique_rate(), 0.8)
        self.assertEqual(profile.get_anomaly_rate(), 0.05)


if __name__ == '__main__':
    unittest.main()