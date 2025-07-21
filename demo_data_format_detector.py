"""
数据格式检测器演示脚本

展示如何使用数据格式检测器进行多格式数据识别和自动格式转换
"""

import sys
import os
import json
import csv
import tempfile
import shutil
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from research_automation.core.data_format_detector import (
    DataFormatDetector, DataFormat, DataFormatConfidence
)


def create_sample_data_files(temp_dir):
    """创建示例数据文件"""
    sample_files = {}
    
    # 1. CSV文件 - 学生成绩数据
    csv_file = os.path.join(temp_dir, "students_scores.csv")
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['学号', '姓名', '数学', '英语', '物理', '总分'])
        writer.writerow(['2021001', '张三', 95, 88, 92, 275])
        writer.writerow(['2021002', '李四', 87, 94, 89, 270])
        writer.writerow(['2021003', '王五', 92, 85, 96, 273])
        writer.writerow(['2021004', '赵六', 89, 91, 88, 268])
        writer.writerow(['2021005', '钱七', 96, 87, 94, 277])
    sample_files['CSV学生成绩'] = csv_file
    
    # 2. JSON文件 - 产品信息
    json_file = os.path.join(temp_dir, "products.json")
    products_data = [
        {
            "id": "P001",
            "name": "智能手机",
            "category": "电子产品",
            "price": 2999.99,
            "stock": 150,
            "specifications": {
                "screen": "6.1英寸",
                "memory": "128GB",
                "camera": "48MP"
            }
        },
        {
            "id": "P002", 
            "name": "笔记本电脑",
            "category": "电子产品",
            "price": 5999.99,
            "stock": 75,
            "specifications": {
                "cpu": "Intel i7",
                "memory": "16GB",
                "storage": "512GB SSD"
            }
        },
        {
            "id": "P003",
            "name": "无线耳机",
            "category": "音频设备", 
            "price": 299.99,
            "stock": 200,
            "specifications": {
                "battery": "24小时",
                "connectivity": "蓝牙5.0",
                "noise_cancelling": True
            }
        }
    ]
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(products_data, f, ensure_ascii=False, indent=2)
    sample_files['JSON产品信息'] = json_file
    
    # 3. XML文件 - 图书目录
    xml_file = os.path.join(temp_dir, "books.xml")
    xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<library>
    <book id="B001">
        <title>Python编程从入门到实践</title>
        <author>Eric Matthes</author>
        <publisher>人民邮电出版社</publisher>
        <year>2020</year>
        <price>89.00</price>
        <category>编程</category>
    </book>
    <book id="B002">
        <title>深度学习</title>
        <author>Ian Goodfellow</author>
        <publisher>人民邮电出版社</publisher>
        <year>2019</year>
        <price>168.00</price>
        <category>人工智能</category>
    </book>
    <book id="B003">
        <title>算法导论</title>
        <author>Thomas H. Cormen</author>
        <publisher>机械工业出版社</publisher>
        <year>2018</year>
        <price>128.00</price>
        <category>算法</category>
    </book>
</library>'''
    with open(xml_file, 'w', encoding='utf-8') as f:
        f.write(xml_content)
    sample_files['XML图书目录'] = xml_file
    
    # 4. TSV文件 - 实验数据
    tsv_file = os.path.join(temp_dir, "experiment_data.tsv")
    with open(tsv_file, 'w', encoding='utf-8') as f:
        f.write("实验编号\t温度\t压力\t湿度\t结果\n")
        f.write("EXP001\t25.5\t101.3\t65\t成功\n")
        f.write("EXP002\t27.2\t102.1\t68\t成功\n")
        f.write("EXP003\t23.8\t100.8\t62\t失败\n")
        f.write("EXP004\t26.1\t101.7\t66\t成功\n")
    sample_files['TSV实验数据'] = tsv_file
    
    # 5. JSONL文件 - 日志数据
    jsonl_file = os.path.join(temp_dir, "logs.jsonl")
    log_entries = [
        {"timestamp": "2024-01-15T10:30:00", "level": "INFO", "message": "系统启动成功", "module": "main"},
        {"timestamp": "2024-01-15T10:31:15", "level": "DEBUG", "message": "加载配置文件", "module": "config"},
        {"timestamp": "2024-01-15T10:32:30", "level": "WARNING", "message": "内存使用率较高", "module": "monitor"},
        {"timestamp": "2024-01-15T10:33:45", "level": "ERROR", "message": "数据库连接失败", "module": "database"},
        {"timestamp": "2024-01-15T10:34:00", "level": "INFO", "message": "重新连接数据库成功", "module": "database"}
    ]
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for entry in log_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    sample_files['JSONL日志数据'] = jsonl_file
    
    # 6. 文本文件 - 研究报告
    text_file = os.path.join(temp_dir, "research_report.txt")
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write("人工智能在教育领域的应用研究报告\n")
        f.write("=" * 40 + "\n\n")
        f.write("摘要：\n")
        f.write("本研究探讨了人工智能技术在现代教育中的应用现状和发展趋势。\n")
        f.write("通过分析大量案例和数据，我们发现AI技术正在革命性地改变教育方式。\n\n")
        f.write("关键词：人工智能、教育技术、个性化学习、智能评估\n\n")
        f.write("1. 引言\n")
        f.write("随着人工智能技术的快速发展，其在教育领域的应用越来越广泛...\n\n")
        f.write("2. 研究方法\n")
        f.write("本研究采用了定量和定性相结合的研究方法...\n\n")
        f.write("3. 结果与分析\n")
        f.write("研究结果表明，AI技术在以下几个方面显示出巨大潜力...\n")
    sample_files['TXT研究报告'] = text_file
    
    return sample_files


def demonstrate_format_detection(detector, sample_files):
    """演示格式检测功能"""
    print("\n" + "=" * 80)
    print("📋 数据格式检测演示")
    print("=" * 80)
    
    detection_results = {}
    
    for file_desc, file_path in sample_files.items():
        print(f"\n🔍 检测文件: {file_desc}")
        print("-" * 60)
        
        # 执行格式检测
        result = detector.detect_format(file_path)
        detection_results[file_desc] = result
        
        # 显示检测结果
        print(f"📁 文件路径: {os.path.basename(file_path)}")
        print(f"🎯 检测格式: {result.detected_format.value}")
        print(f"📊 置信度: {result.confidence_score:.2%} ({result.confidence.value})")
        print(f"🔤 文件编码: {result.encoding}")
        
        if result.format_features:
            print(f"🔧 格式特征:")
            for key, value in result.format_features.items():
                if key == 'delimiter':
                    value = repr(value)  # 显示特殊字符
                elif isinstance(value, list) and len(value) > 5:
                    value = value[:5] + ['...']  # 截断长列表
                print(f"   • {key}: {value}")
        
        if result.conversion_suggestions:
            print(f"💡 转换建议:")
            for i, suggestion in enumerate(result.conversion_suggestions[:3], 1):
                print(f"   {i}. {suggestion}")
        
        if result.detection_errors:
            print(f"❌ 检测错误:")
            for error in result.detection_errors:
                print(f"   • {error}")
        
        # 显示可靠性
        reliability = "✅ 可靠" if result.is_reliable() else "⚠️ 需确认"
        print(f"🎖️ 检测可靠性: {reliability}")
    
    return detection_results


def demonstrate_batch_detection(detector, sample_files):
    """演示批量检测功能"""
    print("\n" + "=" * 80)
    print("📦 批量格式检测演示")
    print("=" * 80)
    
    file_paths = list(sample_files.values())
    
    print(f"🚀 开始批量检测 {len(file_paths)} 个文件...")
    
    # 执行批量检测
    batch_results = detector.batch_detect_formats(file_paths)
    
    # 创建检测结果汇总表
    print(f"\n📊 批量检测结果汇总:")
    print("-" * 80)
    print(f"{'文件名':<25} {'格式':<10} {'置信度':<10} {'可靠性':<10} {'编码':<10}")
    print("-" * 80)
    
    for file_path, result in batch_results.items():
        filename = os.path.basename(file_path)
        format_name = result.detected_format.value
        confidence = f"{result.confidence_score:.1%}"
        reliability = "可靠" if result.is_reliable() else "需确认"
        encoding = result.encoding
        
        print(f"{filename:<25} {format_name:<10} {confidence:<10} {reliability:<10} {encoding:<10}")
    
    # 统计信息
    format_counts = {}
    reliable_count = 0
    
    for result in batch_results.values():
        format_name = result.detected_format.value
        format_counts[format_name] = format_counts.get(format_name, 0) + 1
        if result.is_reliable():
            reliable_count += 1
    
    print("\n📈 检测统计:")
    print(f"• 总文件数: {len(batch_results)}")
    print(f"• 可靠检测: {reliable_count}/{len(batch_results)} ({reliable_count/len(batch_results):.1%})")
    print(f"• 格式分布:")
    for format_name, count in format_counts.items():
        print(f"  - {format_name}: {count} 个文件")


def demonstrate_format_conversion(detector, sample_files, temp_dir):
    """演示格式转换功能"""
    print("\n" + "=" * 80)
    print("🔄 数据格式转换演示")
    print("=" * 80)
    
    conversion_demos = [
        {
            'name': 'CSV → JSON',
            'source': sample_files['CSV学生成绩'],
            'target_format': DataFormat.JSON,
            'output': os.path.join(temp_dir, 'students_scores.json')
        },
        {
            'name': 'JSON → CSV', 
            'source': sample_files['JSON产品信息'],
            'target_format': DataFormat.CSV,
            'output': os.path.join(temp_dir, 'products.csv')
        },
        {
            'name': 'XML → JSON',
            'source': sample_files['XML图书目录'],
            'target_format': DataFormat.JSON,
            'output': os.path.join(temp_dir, 'books.json')
        },
        {
            'name': 'TSV → JSON',
            'source': sample_files['TSV实验数据'],
            'target_format': DataFormat.JSON,
            'output': os.path.join(temp_dir, 'experiment_data.json')
        }
    ]
    
    conversion_results = []
    
    for demo in conversion_demos:
        print(f"\n🔄 转换演示: {demo['name']}")
        print("-" * 60)
        
        # 执行格式转换
        result = detector.convert_format(
            demo['source'],
            demo['target_format'],
            demo['output']
        )
        
        conversion_results.append((demo['name'], result))
        
        # 显示转换结果
        if result.success:
            print(f"✅ 转换成功!")
            print(f"📁 源格式: {result.source_format.value}")
            print(f"🎯 目标格式: {result.target_format.value}")
            print(f"⏱️ 转换时间: {result.conversion_time_seconds:.3f}秒")
            print(f"📊 处理数据: {result.rows_processed} 行 × {result.columns_processed} 列")
            print(f"💾 输出文件: {os.path.basename(result.output_path)}")
            
            if result.has_warnings():
                print(f"⚠️ 转换警告:")
                for warning in result.conversion_warnings:
                    print(f"   • {warning}")
            
            # 显示转换后数据的预览
            if hasattr(result.converted_data, '__len__') and len(result.converted_data) > 0:
                print(f"👀 数据预览:")
                if isinstance(result.converted_data, list):
                    preview_count = min(2, len(result.converted_data))
                    for i in range(preview_count):
                        print(f"   记录{i+1}: {str(result.converted_data[i])[:100]}...")
                elif isinstance(result.converted_data, dict):
                    keys = list(result.converted_data.keys())[:3]
                    print(f"   主要键: {keys}")
        else:
            print(f"❌ 转换失败!")
            if result.has_errors():
                print(f"🚫 错误信息:")
                for error in result.conversion_errors:
                    print(f"   • {error}")
    
    # 转换结果汇总
    print(f"\n📊 转换结果汇总:")
    print("-" * 60)
    
    successful_conversions = sum(1 for _, result in conversion_results if result.success)
    total_conversions = len(conversion_results)
    
    print(f"• 总转换任务: {total_conversions}")
    print(f"• 成功转换: {successful_conversions}")
    print(f"• 成功率: {successful_conversions/total_conversions:.1%}")
    
    if successful_conversions > 0:
        avg_time = sum(result.conversion_time_seconds for _, result in conversion_results if result.success) / successful_conversions
        print(f"• 平均转换时间: {avg_time:.3f}秒")


def demonstrate_data_content_detection(detector):
    """演示从数据内容检测格式"""
    print("\n" + "=" * 80)
    print("📝 数据内容格式检测演示")
    print("=" * 80)
    
    # 准备不同格式的数据内容
    data_samples = [
        {
            'name': 'CSV数据',
            'content': 'name,age,city\n张三,25,北京\n李四,30,上海\n王五,28,广州',
            'filename': 'sample.csv'
        },
        {
            'name': 'JSON对象',
            'content': '{"name": "张三", "age": 25, "city": "北京", "hobbies": ["读书", "游泳"]}',
            'filename': 'sample.json'
        },
        {
            'name': 'JSON数组',
            'content': '[{"id": 1, "name": "产品A"}, {"id": 2, "name": "产品B"}]',
            'filename': 'products.json'
        },
        {
            'name': 'XML文档',
            'content': '<?xml version="1.0"?><root><item>值1</item><item>值2</item></root>',
            'filename': 'data.xml'
        },
        {
            'name': '纯文本',
            'content': '这是一段纯文本内容。\n包含多行文字。\n用于测试文本格式检测。',
            'filename': 'text.txt'
        }
    ]
    
    for sample in data_samples:
        print(f"\n🔍 检测数据: {sample['name']}")
        print("-" * 40)
        
        # 显示数据内容（截断显示）
        content_preview = sample['content'][:100]
        if len(sample['content']) > 100:
            content_preview += "..."
        print(f"📄 数据内容: {content_preview}")
        
        # 执行格式检测
        result = detector.detect_format_from_data(sample['content'], sample['filename'])
        
        # 显示检测结果
        print(f"🎯 检测格式: {result.detected_format.value}")
        print(f"📊 置信度: {result.confidence_score:.2%}")
        print(f"🔤 编码: {result.encoding}")
        
        if result.format_features:
            print(f"🔧 格式特征: {list(result.format_features.keys())}")


def demonstrate_supported_formats(detector):
    """演示支持的格式和转换路径"""
    print("\n" + "=" * 80)
    print("🛠️ 支持的格式和转换路径")
    print("=" * 80)
    
    # 显示支持的格式
    supported_formats = detector.get_supported_formats()
    print(f"\n📋 支持的数据格式 ({len(supported_formats)} 种):")
    for i, data_format in enumerate(supported_formats, 1):
        print(f"  {i}. {data_format.value}")
    
    # 显示支持的转换路径
    supported_conversions = detector.get_supported_conversions()
    print(f"\n🔄 支持的转换路径 ({len(supported_conversions)} 种):")
    
    # 按源格式分组显示
    conversion_groups = {}
    for source_format, target_format in supported_conversions:
        if source_format not in conversion_groups:
            conversion_groups[source_format] = []
        conversion_groups[source_format].append(target_format)
    
    for source_format, target_formats in conversion_groups.items():
        targets = ", ".join([tf.value for tf in target_formats])
        print(f"  • {source_format.value} → {targets}")


def generate_summary_report(detection_results, temp_dir):
    """生成检测和转换汇总报告"""
    print("\n" + "=" * 80)
    print("📊 汇总报告")
    print("=" * 80)
    
    # 检测结果统计
    total_files = len(detection_results)
    reliable_detections = sum(1 for result in detection_results.values() if result.is_reliable())
    
    format_distribution = {}
    confidence_levels = {}
    
    for result in detection_results.values():
        # 格式分布
        format_name = result.detected_format.value
        format_distribution[format_name] = format_distribution.get(format_name, 0) + 1
        
        # 置信度级别分布
        confidence_level = result.confidence.value
        confidence_levels[confidence_level] = confidence_levels.get(confidence_level, 0) + 1
    
    print(f"\n📈 检测性能统计:")
    print(f"• 总检测文件数: {total_files}")
    print(f"• 可靠检测数: {reliable_detections}")
    print(f"• 可靠检测率: {reliable_detections/total_files:.1%}")
    
    print(f"\n📊 格式分布:")
    for format_name, count in sorted(format_distribution.items()):
        percentage = count / total_files * 100
        print(f"• {format_name}: {count} 个文件 ({percentage:.1f}%)")
    
    print(f"\n🎯 置信度分布:")
    for level, count in sorted(confidence_levels.items()):
        percentage = count / total_files * 100
        print(f"• {level}: {count} 个文件 ({percentage:.1f}%)")
    
    # 保存详细报告
    report_file = os.path.join(temp_dir, "detection_report.json")
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_files': total_files,
            'reliable_detections': reliable_detections,
            'reliability_rate': reliable_detections / total_files,
            'format_distribution': format_distribution,
            'confidence_distribution': confidence_levels
        },
        'detailed_results': {
            file_desc: {
                'detected_format': result.detected_format.value,
                'confidence_score': result.confidence_score,
                'confidence_level': result.confidence.value,
                'is_reliable': result.is_reliable(),
                'encoding': result.encoding,
                'format_features': result.format_features,
                'conversion_suggestions': result.conversion_suggestions
            }
            for file_desc, result in detection_results.items()
        }
    }
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 详细报告已保存到: {os.path.basename(report_file)}")


def main():
    """主函数"""
    print("🔍 数据格式检测器演示")
    print("=" * 80)
    print("本演示将展示数据格式检测器的完整功能:")
    print("• 多种数据格式的自动识别")
    print("• 格式检测置信度评估")
    print("• 批量文件格式检测")
    print("• 自动格式转换")
    print("• 从数据内容检测格式")
    print("• 支持的格式和转换路径")
    print("=" * 80)
    
    # 创建临时目录和示例文件
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 初始化检测器
        print("\n🚀 初始化数据格式检测器...")
        detector = DataFormatDetector()
        print("✅ 检测器初始化完成")
        
        # 创建示例数据文件
        print("\n📁 创建示例数据文件...")
        sample_files = create_sample_data_files(temp_dir)
        print(f"✅ 创建了 {len(sample_files)} 个示例文件")
        
        # 演示格式检测
        detection_results = demonstrate_format_detection(detector, sample_files)
        
        # 演示批量检测
        demonstrate_batch_detection(detector, sample_files)
        
        # 演示格式转换
        demonstrate_format_conversion(detector, sample_files, temp_dir)
        
        # 演示数据内容检测
        demonstrate_data_content_detection(detector)
        
        # 演示支持的格式
        demonstrate_supported_formats(detector)
        
        # 生成汇总报告
        generate_summary_report(detection_results, temp_dir)
        
        print("\n" + "=" * 80)
        print("🎉 数据格式检测器演示完成!")
        print("=" * 80)
        
        print("\n📋 功能总结:")
        print("• ✅ 多格式识别 - 支持CSV、JSON、XML、Excel、文本等格式")
        print("• ✅ 智能检测 - 基于文件扩展名、MIME类型、内容特征的综合判断")
        print("• ✅ 置信度评估 - 提供检测结果的可靠性评估")
        print("• ✅ 批量处理 - 支持批量文件格式检测")
        print("• ✅ 格式转换 - 支持多种格式间的自动转换")
        print("• ✅ 内容检测 - 可从数据内容直接检测格式")
        print("• ✅ 错误处理 - 完善的异常处理和错误报告")
        
        print(f"\n📁 临时文件目录: {temp_dir}")
        print("💡 提示: 演示结束后临时文件将被自动清理")
        
    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理临时文件
        try:
            shutil.rmtree(temp_dir)
            print(f"\n🧹 临时文件已清理")
        except Exception as e:
            print(f"\n⚠️ 清理临时文件时出错: {e}")


if __name__ == "__main__":
    main()