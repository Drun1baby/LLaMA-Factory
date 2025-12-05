"""
工具函数模块
"""
import logging
import os
from datetime import datetime


def setup_logger(name: str, log_dir: str = None) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        log_dir: 日志目录路径
    
    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # 文件处理器
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(
            log_dir, 
            f'{name}_{datetime.now().strftime("%Y%m%d")}.log'
        )
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


def validate_prompt(prompt: str, max_length: int = 4096) -> tuple:
    """
    验证用户输入的 prompt
    
    Args:
        prompt: 用户输入
        max_length: 最大长度限制
    
    Returns:
        (is_valid, error_message)
    """
    if not prompt:
        return False, "Prompt 不能为空"
    
    if len(prompt) > max_length:
        return False, f"Prompt 长度超过限制（最大 {max_length} 字符）"
    
    return True, None
