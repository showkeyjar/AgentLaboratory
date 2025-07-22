"""
数据预处理引擎测试
"""

import unittest
import sys
import os
import json
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.data_preprocessing_engine import (
    DataPreprocessingEngine, 
    PreprocessingPipeline,
    PreprocessingStep,
    PreprocessingStepType
)

class TestDataPreprocessingEngine(unittest.TestCase):
    """数据预处理引擎测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.engine = DataPreprocessingEngine()
        
        # 准备测试数据
        self.test_data = [
            {"id": 1, "name": "张三", "age": 25, "score": 85.5, "joined_date": "2020-01-15", "is_active": True},
            {"id": 2, "name": "李四", "age": 30, "score": 92.0, "joined_date": "2019-05-20", "is_active": True},
            {"id": 3, "name": "王五", "age": None, "score": 78.5, "joined_date": "2021-03-10", "is_active": False},
            {"id": 4, "name": "赵六", "age": 22, "score": None, "joined_date": "2022-07-05", "is_active": True},
            {"id": 5, "name": "钱七", "age": 35, "score": 88.0, "joined_date": None, "is_active": None}
        ]
    
    def test_analyze_data(self):
        """测试数据分析功能"""
        profile = self.engine.analyze_data(self.test_data)
        
        # 验证基本信息
        self.assertEqual(profile.row_count, 5)
        self.assertEqual(profile.column_count, 6)
        
        # 验证列信息
        self.assertIn('age', profile.columns)
        self.assertIn('score', profile.columns)
        
        # 验证缺失值统计
        self.assertEqual(profile.columns['age'].missing_count, 1)
        self.assertEqual(profile.columns['score'].missing_count, 1)
        
        # 验证数据类型
        self.assertEqual(profile.columns['age'].data_type.value, 'numeric')
        self.assertEqual(profile.columns['name'].data_type.value, 'text')
        self.assertEqual(profile.columns['is_active'].data_type.value, 'boolean')
    
    def test_create_pipeline(self):
        """测试创建预处理流水线"""
        pipeline = self.engine.create_preprocessing_pipeline("测试流水线", "用于测试的预处理流水线")
        
        self.assertEqual(pipeline.name, "测试流水线")
        self.assertEqual(pipeline.description, "用于测试的预处理流水线")
        self.assertEqual(len(pipeline.steps), 0)
    
    def test_add_preprocessing_step(self):
        """测试添加预处理步骤"""
        pipeline = self.engine.create_preprocessing_pipeline("测试流水线")
        
        # 添加清洗步骤
        step1 = self.engine.add_preprocessing_step(
            pipeline,
            PreprocessingStepType.CLEANING,
            'remove_missing_values',
            ['age', 'score']
        )
        
        # 添加转换步骤
        step2 = self.engine.add_preprocessing_step(
            pipeline,
            PreprocessingStepType.TRANSFORMATION,
            'convert_type',
            ['age'],
            {'target_type': 'int'}
        )
        
        self.assertEqual(len(pipeline.steps), 2)
        self.assertEqual(pipeline.steps[0].function_name, 'remove_missing_values')
        self.assertEqual(pipeline.steps[1].function_name, 'convert_type')
    
    def test_execute_pipeline(self):
        """测试执行预处理流水线"""
        pipeline = self.engine.create_preprocessing_pipeline("测试流水线")
        
        # 添加清洗步骤：移除缺失值
        self.engine.add_preprocessing_step(
            pipeline,
            PreprocessingStepType.CLEANING,
            'remove_missing_values',
            ['age', 'score']
        )
        
        # 执行流水线
        processed_data, result = self.engine.execute_pipeline(pipeline, self.test_data)
        
        # 验证结果
        self.assertTrue(result.success)
        self.assertEqual(result.input_rows, 5)
        self.assertEqual(result.output_rows, 3)  # 应该移除了2行（有缺失值的行）
    
    def test_fill_missing_values(self):
        """测试填充缺失值"""
        pipeline = self.engine.create_preprocessing_pipeline("填充缺失值测试")
        
        # 添加填充缺失值步骤
        self.engine.add_preprocessing_step(
            pipeline,
            PreprocessingStepType.CLEANING,
            'fill_missing_values',
            ['age', 'score'],
            {'method': 'mean'}
        )
        
        # 执行流水线
        processed_data, result = self.engine.execute_pipeline(pipeline, self.test_data)
        
        # 验证结果
        self.assertTrue(result.success)
        self.assertEqual(result.input_rows, 5)
        self.assertEqual(result.output_rows, 5)
        
        # 验证填充后没有缺失值
        for item in processed_data:
            self.assertIsNotNone(item['age'])
            self.assertIsNotNone(item['score'])
    
    def test_remove_outliers(self):
        """测试移除异常值"""
        # 创建包含异常值的测试数据
        outlier_data = self.test_data.copy()
        outlier_data.append({"id": 6, "name": "异常值", "age": 100, "score": 200, "joined_date": "2023-01-01", "is_active": True})
        
        pipeline = self.engine.create_preprocessing_pipeline("异常值测试")
        
        # 添加移除异常值步骤
        self.engine.add_preprocessing_step(
            pipeline,
            PreprocessingStepType.CLEANING,
            'remove_outliers',
            ['age', 'score'],
            {'method': 'zscore', 'threshold': 2.0}
        )
        
        # 执行流水线
        processed_data, result = self.engine.execute_pipeline(pipeline, outlier_data)
        
        # 验证结果
        self.assertTrue(result.success)
        self.assertEqual(result.input_rows, 6)
        self.assertEqual(result.output_rows, 5)  # 应该移除了异常值行
    
    def test_standardize(self):
        """测试标准化"""
        pipeline = self.engine.create_preprocessing_pipeline("标准化测试")
        
        # 添加标准化步骤
        self.engine.add_preprocessing_step(
            pipeline,
            PreprocessingStepType.NORMALIZATION,
            'standardize',
            ['age', 'score']
        )
        
        # 执行流水线
        processed_data, result = self.engine.execute_pipeline(pipeline, self.test_data)
        
        # 验证结果
        self.assertTrue(result.success)
        
        # 验证标准化后的值
        for item in processed_data:
            if item['age'] is not None:
                self.assertIsInstance(item['age'], float)
            if item['score'] is not None:
                self.assertIsInstance(item['score'], float)
    
    def test_extract_text_features(self):
        """测试提取文本特征"""
        pipeline = self.engine.create_preprocessing_pipeline("文本特征测试")
        
        # 添加提取文本特征步骤
        self.engine.add_preprocessing_step(
            pipeline,
            PreprocessingStepType.FEATURE_ENGINEERING,
            'extract_text_features',
            ['name'],
            {'features': ['length', 'word_count']}
        )
        
        # 执行流水线
        processed_data, result = self.engine.execute_pipeline(pipeline, self.test_data)
        
        # 验证结果
        self.assertTrue(result.success)
        
        # 验证新增的特征列
        self.assertIn('name_length', processed_data[0])
        self.assertIn('name_word_count', processed_data[0])
        
        # 验证特征值
        self.assertEqual(processed_data[0]['name_length'], 2)
        self.assertEqual(processed_data[0]['name_word_count'], 1)
    
    def test_pipeline_serialization(self):
        """测试流水线序列化"""
        pipeline = self.engine.create_preprocessing_pipeline("序列化测试")
        
        # 添加几个步骤
        self.engine.add_preprocessing_step(
            pipeline,
            PreprocessingStepType.CLEANING,
            'remove_missing_values',
            ['age', 'score']
        )
        
        self.engine.add_preprocessing_step(
            pipeline,
            PreprocessingStepType.NORMALIZATION,
            'standardize',
            ['age', 'score']
        )
        
        # 序列化为字典
        pipeline_dict = pipeline.to_dict()
        
        # 验证序列化结果
        self.assertEqual(pipeline_dict['name'], "序列化测试")
        self.assertEqual(len(pipeline_dict['steps']), 2)
        
        # 保存到临时文件
        temp_file = "temp_pipeline.json"
        pipeline.save_to_json(temp_file)
        
        # 从文件加载
        loaded_pipeline = PreprocessingPipeline.load_from_json(temp_file)
        
        # 验证加载结果
        self.assertEqual(loaded_pipeline.name, pipeline.name)
        self.assertEqual(len(loaded_pipeline.steps), len(pipeline.steps))
        
        # 清理临时文件
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == '__main__':
    unittest.main()