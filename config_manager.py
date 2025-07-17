from enum import Enum
from typing import Dict, Any
import yaml
import os
from pathlib import Path

class ConfigType(Enum):
    """配置类型枚举"""
    BASIC = "basic"
    ENVIRONMENT = "environment"
    RESEARCH = "research"
    TASK = "task"
    ADVANCED = "advanced"

class ConfigManager:
    """配置管理器，提供选择题式配置界面"""
    
    def __init__(self):
        self.configs: Dict[str, Dict[str, Any]] = {
            "project-info": {
                "name": "项目基本信息",
                "questions": {
                    "name": "请输入项目名称：",
                    "description": "请简要描述项目：",
                    "goal": "请输入项目目标：",
                }
            },
            "environment": {
                "name": "环境配置",
                "questions": {
                    "language": {
                        "question": "请选择项目语言：",
                        "options": {
                            "1": "中文",
                            "2": "English",
                        },
                        "default": "1"
                    },
                    "model": {
                        "question": "请选择使用的模型：",
                        "options": {
                            "1": "o3-mini",
                            "2": "gpt-4o-mini",
                            "3": "deepseek-chat",
                        },
                        "default": "1"
                    }
                }
            },
            "research-params": {
                "name": "研究参数",
                "questions": {
                    "dataset": {
                        "question": "请选择数据集：",
                        "options": {
                            "1": "MATH-500",
                            "2": "自定义数据集",
                        },
                        "default": "1"
                    },
                    "target-metric": {
                        "question": "请选择目标指标：",
                        "options": {
                            "1": "准确性",
                            "2": "F1分数",
                            "3": "自定义指标",
                        },
                        "default": "1"
                    }
                }
            },
            "tasks": {
                "name": "任务配置",
                "questions": {
                    "literature": {
                        "question": "是否需要进行文献综述？",
                        "options": {
                            "1": "是",
                            "2": "否",
                        },
                        "default": "1"
                    },
                    "experiment": {
                        "question": "是否需要设计实验？",
                        "options": {
                            "1": "是",
                            "2": "否",
                        },
                        "default": "1"
                    },
                    "analysis": {
                        "question": "是否需要进行结果分析？",
                        "options": {
                            "1": "是",
                            "2": "否",
                        },
                        "default": "1"
                    }
                }
            },
            "advanced": {
                "name": "高级设置",
                "questions": {
                    "parallel": {
                        "question": "是否启用并行运行？",
                        "options": {
                            "1": "是",
                            "2": "否",
                        },
                        "default": "2"
                    },
                    "save": {
                        "question": "是否保存进度？",
                        "options": {
                            "1": "是",
                            "2": "否",
                        },
                        "default": "1"
                    },
                    "debug": {
                        "question": "是否开启调试模式？",
                        "options": {
                            "1": "是",
                            "2": "否",
                        },
                        "default": "2"
                    }
                }
            }
        }
        
    def get_config_question(self, section: str, config_name: str) -> str:
        """获取配置项的选择题"""
        config = self.configs[section]["questions"].get(config_name)
        if not config:
            return ""
            
        if isinstance(config, dict):  # 选择题
            question = config["question"]
            for key, option in config["options"].items():
                question += f"\n{key}. {option}"
            return question
        else:  # 文本输入题
            return config
        
    def get_default_value(self, section: str, config_name: str) -> str:
        """获取配置项的默认值"""
        config = self.configs[section]["questions"].get(config_name)
        if isinstance(config, dict):
            return config.get("default", "1")
        return ""
        
    def generate_yaml(self, config_dict: Dict[str, Any]) -> str:
        """生成yaml配置文件"""
        return yaml.dump(config_dict, allow_unicode=True, sort_keys=False)
        
    def save_config(self, config_dict: Dict[str, Any], filename: str = "config.yaml"):
        """保存配置文件"""
        with open(filename, "w", encoding="utf-8") as f:
            f.write(self.generate_yaml(config_dict))
            
def generate_config():
    """生成配置文件"""
    print("\n欢迎使用项目配置向导！\n")
    config_manager = ConfigManager()
    
    # 创建配置字典
    config_dict = {}
    
    # 逐个处理每个配置部分
    for section in config_manager.configs:
        print(f"\n{config_manager.configs[section]['name']}配置：")
        section_dict = {}
        
        for config_name, config_data in config_manager.configs[section]["questions"].items():
            question = config_manager.get_config_question(section, config_name)
            default = config_manager.get_default_value(section, config_name)
            
            if isinstance(config_data, dict):  # 选择题
                print(f"\n{question}")
                while True:
                    choice = input("\n请输入选择: ")
                    if choice in config_data["options"]:
                        section_dict[config_name] = config_data["options"][choice]
                        break
                    print("无效的选择，请重新输入！")
            else:  # 文本输入题
                print(f"\n{question}")
                value = input("请输入内容: ")
                section_dict[config_name] = value
        
        config_dict[section] = section_dict
    
    # 保存配置文件
    config_manager.save_config(config_dict)
    print("\n配置完成！配置文件已保存为 config.yaml")
    print("请根据需要修改 config.yaml 中的具体值")

if __name__ == '__main__':
    generate_config()
