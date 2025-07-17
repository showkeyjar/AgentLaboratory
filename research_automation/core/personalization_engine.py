"""
个性化学习组件

负责学习用户偏好，提供个性化服务
"""

from typing import Any, Dict, List, Optional
from .base_component import BaseComponent


class PersonalizationComponent(BaseComponent):
    """个性化学习组件"""
    
    def get_required_configs(self) -> List[str]:
        """获取必需的配置项"""
        return ['llm_model']
    
    def _setup_component(self):
        """设置组件特定的初始化逻辑"""
        self.logger.info("个性化学习组件初始化完成")