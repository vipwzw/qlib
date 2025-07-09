#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置加载器
支持从.env文件加载环境变量并替换YAML配置文件中的占位符
"""

import os
import re
import yaml
from pathlib import Path
from typing import Dict, Any, Union
from dotenv import load_dotenv
import logging

class ConfigLoader:
    """配置加载器"""
    
    def __init__(self, env_file: str = ".env", config_file: str = "configs/config.yaml"):
        """
        初始化配置加载器
        
        Args:
            env_file: 环境变量文件路径
            config_file: 配置文件路径
        """
        self.env_file = env_file
        self.config_file = config_file
        self.logger = logging.getLogger(__name__)
        
        # 加载环境变量
        self._load_env_file()
    
    def _load_env_file(self):
        """加载.env文件"""
        env_path = Path(self.env_file)
        
        if env_path.exists():
            load_dotenv(env_path)
            self.logger.info(f"已加载环境变量文件: {env_path}")
        else:
            self.logger.warning(f"环境变量文件不存在: {env_path}")
            self.logger.info("将使用系统环境变量或默认值")
    
    def _substitute_env_variables(self, obj: Any) -> Any:
        """
        递归替换对象中的环境变量占位符
        
        支持格式:
        - ${VAR_NAME}: 必需的环境变量
        - ${VAR_NAME:default_value}: 带默认值的环境变量
        
        Args:
            obj: 要处理的对象
            
        Returns:
            替换后的对象
        """
        if isinstance(obj, dict):
            return {key: self._substitute_env_variables(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_variables(item) for item in obj]
        elif isinstance(obj, str):
            return self._substitute_string(obj)
        else:
            return obj
    
    def _substitute_string(self, text: str) -> Union[str, int, float, bool]:
        """
        替换字符串中的环境变量占位符
        
        Args:
            text: 输入文本
            
        Returns:
            替换后的值，自动转换类型
        """
        # 匹配 ${VAR_NAME} 或 ${VAR_NAME:default_value} 格式
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
        
        def replace_env_var(match):
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) is not None else None
            
            # 获取环境变量值
            env_value = os.getenv(var_name, default_value)
            
            if env_value is None:
                raise ValueError(f"环境变量 '{var_name}' 未设置且无默认值")
            
            return env_value
        
        # 执行替换
        result = re.sub(pattern, replace_env_var, text)
        
        # 如果整个字符串都是占位符，尝试转换类型
        if re.match(r'^\$\{[^}]+\}$', text):
            return self._convert_type(result)
        
        return result
    
    def _convert_type(self, value: str) -> Union[str, int, float, bool]:
        """
        自动转换字符串类型
        
        Args:
            value: 字符串值
            
        Returns:
            转换后的值
        """
        if value.lower() in ('true', 'yes', '1', 'on'):
            return True
        elif value.lower() in ('false', 'no', '0', 'off'):
            return False
        
        # 尝试转换为数字
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            return value
    
    def load_config(self) -> Dict[str, Any]:
        """
        加载并处理配置文件
        
        Returns:
            处理后的配置字典
        """
        config_path = Path(self.config_file)
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        # 读取YAML文件
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 替换环境变量
        processed_config = self._substitute_env_variables(config)
        
        self.logger.info(f"已加载配置文件: {config_path}")
        return processed_config
    
    def validate_required_env_vars(self, required_vars: list) -> Dict[str, bool]:
        """
        验证必需的环境变量
        
        Args:
            required_vars: 必需的环境变量列表
            
        Returns:
            验证结果字典
        """
        results = {}
        for var in required_vars:
            value = os.getenv(var)
            results[var] = value is not None and value.strip() != ""
        
        return results
    
    def get_safe_config_preview(self) -> Dict[str, Any]:
        """
        获取安全的配置预览（隐藏敏感信息）
        
        Returns:
            隐藏敏感信息的配置字典
        """
        config = self.load_config()
        return self._mask_sensitive_data(config)
    
    def _mask_sensitive_data(self, obj: Any, sensitive_keys: set = None) -> Any:
        """
        隐藏敏感数据
        
        Args:
            obj: 配置对象
            sensitive_keys: 敏感字段名集合
            
        Returns:
            隐藏敏感信息后的对象
        """
        if sensitive_keys is None:
            sensitive_keys = {
                'api_key', 'secret_key', 'bearer_token', 'password', 
                'token', 'key', 'secret', 'credential'
            }
        
        if isinstance(obj, dict):
            masked = {}
            for key, value in obj.items():
                if any(sensitive_key in key.lower() for sensitive_key in sensitive_keys):
                    if value and str(value).strip():
                        masked[key] = "***隐藏***"
                    else:
                        masked[key] = value
                else:
                    masked[key] = self._mask_sensitive_data(value, sensitive_keys)
            return masked
        elif isinstance(obj, list):
            return [self._mask_sensitive_data(item, sensitive_keys) for item in obj]
        else:
            return obj


def load_project_config(env_file: str = ".env", config_file: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    便捷函数：加载项目配置
    
    Args:
        env_file: 环境变量文件路径
        config_file: 配置文件路径
        
    Returns:
        处理后的配置字典
    """
    loader = ConfigLoader(env_file, config_file)
    return loader.load_config()


def create_env_file_template(output_path: str = ".env.example"):
    """
    创建.env文件模板
    
    Args:
        output_path: 输出文件路径
    """
    template_content = """# 新闻情感量化因子分析项目环境变量配置
# 复制此文件为 .env 并填入真实的API密钥

# ===========================================
# API配置 (必需)
# ===========================================

# Binance API配置 (获取价格数据)
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET_KEY=your_binance_secret_key_here
BINANCE_SANDBOX=true

# News API配置 (可选，增强新闻采集)
NEWS_API_KEY=your_news_api_key_here
NEWS_API_ENABLED=false

# Twitter API配置 (可选)
TWITTER_BEARER_TOKEN=your_twitter_bearer_token_here
TWITTER_ENABLED=false

# ===========================================
# 数据源配置 (可选)
# ===========================================

# 自定义新闻源
CUSTOM_NEWS_RSS_URL=https://example.com/rss
CUSTOM_NEWS_API_URL=https://api.example.com/news

# ===========================================
# 数据库配置 (可选)
# ===========================================

# 如果使用数据库存储
DATABASE_URL=sqlite:///data/news_factor.db
REDIS_URL=redis://localhost:6379/0

# ===========================================
# 模型配置 (可选)
# ===========================================

# Hugging Face API Token (用于高级NLP模型)
HUGGINGFACE_TOKEN=your_huggingface_token_here

# OpenAI API Key (用于GPT情感分析)
OPENAI_API_KEY=your_openai_api_key_here

# ===========================================
# 日志和调试 (可选)
# ===========================================

# 日志级别: DEBUG, INFO, WARNING, ERROR
LOG_LEVEL=INFO

# 调试模式
DEBUG=false

# ===========================================
# 回测配置 (可选)
# ===========================================

# 初始资金
INITIAL_CAPITAL=100000

# 交易费率
TRADING_FEE=0.001

# 最大仓位
MAX_POSITION=0.95"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(template_content)
    
    print(f"✅ 环境变量模板已创建: {output_path}")
    print("📝 请复制为 .env 文件并填入真实的API密钥")


if __name__ == "__main__":
    """测试配置加载器"""
    import sys
    from pathlib import Path
    
    # 测试创建模板
    create_env_file_template()
    
    # 测试配置加载
    try:
        loader = ConfigLoader()
        
        # 验证必需的环境变量
        required_vars = ['BINANCE_API_KEY', 'BINANCE_SECRET_KEY']
        validation_results = loader.validate_required_env_vars(required_vars)
        
        print("\n环境变量验证结果:")
        for var, is_set in validation_results.items():
            status = "✅" if is_set else "❌"
            print(f"{status} {var}: {'已设置' if is_set else '未设置'}")
        
        # 加载配置（安全预览）
        print("\n配置预览:")
        safe_config = loader.get_safe_config_preview()
        print(f"API配置: {safe_config.get('apis', {})}")
        
    except Exception as e:
        print(f"❌ 配置加载测试失败: {e}")
        sys.exit(1)
    
    print("\n✅ 配置加载器测试完成！") 