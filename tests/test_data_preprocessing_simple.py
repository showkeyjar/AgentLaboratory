"""
数据预处理引擎简化测试

不依赖外部库的基本功能测试
"""

import unittest
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


class TestDataPreprocessingEngineSimple(unittest.TestCase):
    """数据预处理引擎简化测试类"""
    
    def setUp(self):
        """设置测试环境"""
        self.config = {}
        self.engine = DataPreprocessingEngine(self.config)
        
        # 创建简单测试数据
        self.test_dict_list = [
            {'id': 1, 'name': 'Alice', 'age': 25, 'score': 85.5},
            {'id': 2, 'name': 'Bob', 'age': 30, 'score': 92.0},
            {'id': 3, 'name': 'Charlie', 'age': None, 'score': 78.5},
            {'id': 4, 'name': 'David', 'age': 28, 'score': None},
            {'id': 1, 'name': 'Alice', 'age': 25, 'score': 85.5}  # 重复行
        ]
        
        self.test_numeric_list = [1, 2, 3, 4, 5, None, 7, 8, 9, 100]
    
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
            target_columns=['age'],
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
    
    def test_data_type_inference_list(self):
        """测试列表数据类型推断"""
        # 测试数值型列表
        numeric_list = [1, 2, 3, 4, 5]
        numeric_type = self.engine._infer_data_type(numeric_list)
        self.assertEqual(numeric_type, DataType.NUMERIC)
        
        # 测试文本型列表
        text_list = ['hello', 'world', 'test', 'data']
        text_type = self.engine._infer_data_type(text_list)
        self.assertEqual(text_type, DataType.TEXT)
        
        # 测试类别型列表（重复值多）
        category_list = ['A', 'B', 'A', 'C', 'B', 'A']
        category_type = self.engine._infer_data_type(category_list)
        self.assertEqual(category_type, DataType.CATEGORICAL)
        
        # 测试布尔型列表
        bool_list = [True, False, True, False]
        bool_type = self.engine._infer_data_type(bool_list)
        self.assertEqual(bool_type, DataType.BOOLEAN)
    
    def test_remove_duplicates_dict_list(self):
        """测试移除字典列表中的重复行"""
        result = self.engine._remove_duplicates(self.test_dict_list, [], {})
        self.assertEqual(len(result), 4)  # 应该移除一个重复行
        
        # 测试指定列去重
        result = self.engine._remove_duplicates(self.test_dict_list, ['id'], {})
        self.assertEqual(len(result), 4)  # 基于id列去重
    
    def test_remove_missing_dict_list(self):
        """测试移除字典列表中的缺失值"""
        # 移除所有有缺失值的行
        result = self.engine._remove_missing(self.test_dict_list, [], {})
        self.assertEqual(len(result), 2)  # 只有前两行没有缺失值
        
        # 移除指定列有缺失值的行
        result = self.engine._remove_missing(self.test_dict_list, ['age'], {})
        self.assertEqual(len(result), 4)  # 移除age为None的行
    
    def test_fill_missing_mean_dict_list(self):
        """测试用均值填充字典列表中的缺失值"""
        result = self.engine._fill_missing_mean(self.test_dict_list.copy(), ['age'], {})
        
        # 检查是否填充了缺失值
        ages = [item['age'] for item in result]
        none_count = sum(1 for age in ages if age is None)
        self.assertEqual(none_count, 0)
        
        # 检查填充的值是否合理
        filled_age = result[2]['age']  # 原来是None的位置
        self.assertIsNotNone(filled_age)
        self.assertIsInstance(filled_age, (int, float))
    
    def test_fill_missing_constant_dict_list(self):
        """测试用常数填充字典列表中的缺失值"""
        result = self.engine._fill_missing_constant(
            self.test_dict_list.copy(), 
            ['age'], 
            {'fill_value': -1}
        )
        
        # 检查填充值
        filled_item = result[2]  # 原来age为None的项
        self.assertEqual(filled_item['age'], -1)
    
    def test_data_profiling_dict_list(self):
        """测试字典列表的数据概况分析"""
        profile = self.engine._profile_data(self.test_dict_list)
        
        self.assertIsInstance(profile, DataProfile)
        self.assertEqual(profile.row_count, 5)
        self.assertEqual(profile.column_count, 4)
        self.assertGreater(len(profile.columns), 0)
        
        # 检查缺失单元格数量
        self.assertGreater(profile.missing_cells_count, 0)
    
    def test_column_profiling_list(self):
        """测试列表的列数据概况分析"""
        ages = [item.get('age') for item in self.test_dict_list]
        profile = self.engine._profile_column(ages)
        
        self.assertIsInstance(profile, ColumnProfile)
        self.assertEqual(profile.count, 5)
        self.assertEqual(profile.missing_count, 1)  # 一个None值
        self.assertGreater(profile.unique_count, 0)
    
    def test_preprocess_data_pipeline_dict_list(self):
        """测试字典列表的完整预处理流水线"""
        # 添加预处理步骤
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
                target_columns=['age'],
                parameters={'function': 'fill_missing_mean'},
                order=2
            )
        ]
        
        for step in steps:
            self.engine.add_preprocessing_step(step)
        
        # 执行预处理
        processed_data, result = self.engine.preprocess_data(self.test_dict_list.copy())
        
        # 检查结果
        self.assertIsInstance(result, PreprocessingResult)
        self.assertTrue(result.success)
        self.assertEqual(len(result.steps_executed), 2)
        self.assertEqual(len(result.steps_failed), 0)
        
        # 检查数据变化
        self.assertLess(len(processed_data), len(self.test_dict_list))  # 去重后行数减少
        
        # 检查缺失值是否被填充
        ages = [item.get('age') for item in processed_data]
        none_count = sum(1 for age in ages if age is None)
        self.assertEqual(none_count, 0)
    
    def test_preprocess_data_empty_pipeline(self):
        """测试空流水线"""
        processed_data, result = self.engine.preprocess_data(self.test_dict_list.copy())
        
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
        
        processed_data, result = self.engine.preprocess_data(self.test_dict_list.copy())
        
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
        
        # 测试问题列识别
        profile.columns = {
            'good_col': ColumnProfile(count=100, missing_count=5),  # 5% 缺失
            'bad_col': ColumnProfile(count=100, missing_count=15)   # 15% 缺失
        }
        problematic_cols = profile.get_problematic_columns(threshold=0.1)
        self.assertEqual(len(problematic_cols), 1)
        self.assertIn('bad_col', problematic_cols)
    
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
    
    def test_clear_pipeline(self):
        """测试清空流水线"""
        # 添加一些步骤
        step1 = PreprocessingStep(name="步骤1", parameters={'function': 'remove_duplicates'})
        step2 = PreprocessingStep(name="步骤2", parameters={'function': 'remove_missing'})
        
        self.engine.add_preprocessing_step(step1)
        self.engine.add_preprocessing_step(step2)
        
        self.assertEqual(len(self.engine.pipeline), 2)
        
        # 清空流水线
        self.engine.clear_pipeline()
        self.assertEqual(len(self.engine.pipeline), 0)
    
    def test_get_all_preprocessing_steps(self):
        """测试获取所有预处理步骤"""
        step1 = PreprocessingStep(name="步骤1", parameters={'function': 'remove_duplicates'})
        step2 = PreprocessingStep(name="步骤2", parameters={'function': 'remove_missing'})
        
        self.engine.add_preprocessing_step(step1)
        self.engine.add_preprocessing_step(step2)
        
        all_steps = self.engine.get_all_preprocessing_steps()
        self.assertEqual(len(all_steps), 2)
        self.assertIsInstance(all_steps, list)
        
        # 确保返回的是副本，不是原始列表
        all_steps.clear()
        self.assertEqual(len(self.engine.pipeline), 2)


if __name__ == '__main__':
    unittest.main()