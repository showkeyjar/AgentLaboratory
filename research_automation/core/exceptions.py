"""
研究自动化异常类定义

定义所有研究自动化功能相关的异常类
"""


class ResearchAutomationError(Exception):
    """研究自动化基础异常类"""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.details = details or {}
    
    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'error_code': self.error_code,
            'details': self.details
        }


class ValidationError(ResearchAutomationError):
    """数据验证异常"""
    
    def __init__(self, message: str, field_name: str = None, invalid_value=None):
        super().__init__(message, "VALIDATION_ERROR")
        self.field_name = field_name
        self.invalid_value = invalid_value
        self.details.update({
            'field_name': field_name,
            'invalid_value': str(invalid_value) if invalid_value is not None else None
        })


class ProcessingError(ResearchAutomationError):
    """处理过程异常"""
    
    def __init__(self, message: str, component_name: str = None, operation: str = None):
        super().__init__(message, "PROCESSING_ERROR")
        self.component_name = component_name
        self.operation = operation
        self.details.update({
            'component_name': component_name,
            'operation': operation
        })


class ConfigurationError(ResearchAutomationError):
    """配置异常"""
    
    def __init__(self, message: str, config_key: str = None, expected_type: str = None):
        super().__init__(message, "CONFIGURATION_ERROR")
        self.config_key = config_key
        self.expected_type = expected_type
        self.details.update({
            'config_key': config_key,
            'expected_type': expected_type
        })


class ResourceError(ResearchAutomationError):
    """资源相关异常"""
    
    def __init__(self, message: str, resource_type: str = None, resource_id: str = None):
        super().__init__(message, "RESOURCE_ERROR")
        self.resource_type = resource_type
        self.resource_id = resource_id
        self.details.update({
            'resource_type': resource_type,
            'resource_id': resource_id
        })


class APIError(ResearchAutomationError):
    """API调用异常"""
    
    def __init__(self, message: str, api_name: str = None, status_code: int = None):
        super().__init__(message, "API_ERROR")
        self.api_name = api_name
        self.status_code = status_code
        self.details.update({
            'api_name': api_name,
            'status_code': status_code
        })


class DataError(ResearchAutomationError):
    """数据相关异常"""
    
    def __init__(self, message: str, data_source: str = None, data_format: str = None):
        super().__init__(message, "DATA_ERROR")
        self.data_source = data_source
        self.data_format = data_format
        self.details.update({
            'data_source': data_source,
            'data_format': data_format
        })


class AuthenticationError(ResearchAutomationError):
    """认证异常"""
    
    def __init__(self, message: str, user_id: str = None, required_permission: str = None):
        super().__init__(message, "AUTHENTICATION_ERROR")
        self.user_id = user_id
        self.required_permission = required_permission
        self.details.update({
            'user_id': user_id,
            'required_permission': required_permission
        })


class ConcurrencyError(ResearchAutomationError):
    """并发操作异常"""
    
    def __init__(self, message: str, resource_id: str = None, conflicting_operation: str = None):
        super().__init__(message, "CONCURRENCY_ERROR")
        self.resource_id = resource_id
        self.conflicting_operation = conflicting_operation
        self.details.update({
            'resource_id': resource_id,
            'conflicting_operation': conflicting_operation
        })


class TimeoutError(ResearchAutomationError):
    """超时异常"""
    
    def __init__(self, message: str, operation: str = None, timeout_seconds: float = None):
        super().__init__(message, "TIMEOUT_ERROR")
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        self.details.update({
            'operation': operation,
            'timeout_seconds': timeout_seconds
        })


# 异常处理工具函数
def handle_exception(func):
    """异常处理装饰器"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ResearchAutomationError:
            # 重新抛出已知的研究自动化异常
            raise
        except ValueError as e:
            raise ValidationError(f"参数验证失败: {str(e)}")
        except KeyError as e:
            raise ConfigurationError(f"缺少必需的配置项: {str(e)}")
        except FileNotFoundError as e:
            raise ResourceError(f"文件未找到: {str(e)}")
        except PermissionError as e:
            raise AuthenticationError(f"权限不足: {str(e)}")
        except TimeoutError as e:
            raise TimeoutError(f"操作超时: {str(e)}")
        except Exception as e:
            # 包装未知异常
            raise ProcessingError(f"未知错误: {str(e)}")
    
    return wrapper


def format_error_response(error: ResearchAutomationError) -> dict:
    """格式化错误响应"""
    return {
        'success': False,
        'error': error.to_dict(),
        'timestamp': str(datetime.now())
    }


# 导入datetime用于错误响应格式化
from datetime import datetime