from enum import Enum
from typing import Dict, Any
import os
from pathlib import Path


class ConfigType(Enum):
    """配置类型枚举"""
    SECRET = "secret"
    PATH = "path"
    DATABASE = "database"
    BOOLEAN = "boolean"


class ConfigManager:
    """配置管理器，提供选择题式配置界面"""
    
    def __init__(self):
        self.configs: Dict[str, Dict[str, Any]] = {
            "SECRET_KEY": {
                "type": ConfigType.SECRET,
                "question": "请选择应用的密钥类型：",
                "options": {
                    "1": "使用随机生成的密钥",
                    "2": "使用自定义密钥",
                },
                "default": "1"
            },
            "UPLOAD_FOLDER": {
                "type": ConfigType.PATH,
                "question": "请选择上传文件夹的位置：",
                "options": {
                    "1": "默认位置 (uploads/)",
                    "2": "自定义位置",
                },
                "default": "1"
            },
            "SQLALCHEMY_DATABASE_URI": {
                "type": ConfigType.DATABASE,
                "question": "请选择数据库类型：",
                "options": {
                    "1": "SQLite (默认)",
                    "2": "PostgreSQL",
                    "3": "MySQL",
                },
                "default": "1"
            },
            "SQLALCHEMY_TRACK_MODIFICATIONS": {
                "type": ConfigType.BOOLEAN,
                "question": "是否开启数据库修改跟踪：",
                "options": {
                    "1": "是",
                    "2": "否",
                },
                "default": "2"
            }
        }
        
    def get_config_question(self, config_name: str) -> str:
        """获取配置项的选择题"""
        config = self.configs.get(config_name)
        if not config:
            return ""
            
        question = config["question"]
        for key, option in config["options"].items():
            question += f"\n{key}. {option}"
        
        return question
        
    def get_default_value(self, config_name: str) -> str:
        """获取配置项的默认值"""
        return self.configs[config_name]["default"]


def get_config():
    """获取优化后的配置"""
    config = ConfigManager()
    
    # 生成配置文件
    config_file = Path("config.env")
    if not config_file.exists():
        with open(config_file, "w", encoding="utf-8") as f:
            for config_name, config_data in config.configs.items():
                question = config.get_config_question(config_name)
                default = config.get_default_value(config_name)
                
                f.write(f"# {question}\n")
                f.write(f"{config_name}={default}\n\n")
    
    # 读取配置
    env_config = {}
    with open(config_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                key, value = line.strip().split("=", 1)
                env_config[key] = value
    
    return env_config
