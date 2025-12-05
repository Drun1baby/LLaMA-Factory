"""
Flask 应用配置文件
"""
import os

# 基础路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# 模型配置
MODEL_CONFIG = {
    # 基础模型路径（Hugging Face 模型名或本地路径）
    "base_model_path": "tencent/Hunyuan-7B-Instruct",
    # LoRA 适配器路径
    "lora_adapter_path": os.path.join(PROJECT_ROOT, "saves/Hunyuan-7B-Instruct/lora/save_Test"),
    # 设备配置：auto, cuda, cpu, mps
    "device": "auto",
    # 量化配置：None, "4bit", "8bit"
    "quantization": None,
    # 最大生成长度（减少以提升速度，可根据需要调整）
    "max_new_tokens": 256,
    # 采样温度
    "temperature": 0.7,
    # Top-p 采样
    "top_p": 0.9,
    # 是否使用流式生成
    "use_streaming": False,
}

# Flask 配置
class Config:
    """Flask 基础配置类"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    DEBUG = False
    TESTING = False

class DevelopmentConfig(Config):
    """开发环境配置"""
    DEBUG = True

class ProductionConfig(Config):
    """生产环境配置"""
    DEBUG = False

class TestingConfig(Config):
    """测试环境配置"""
    TESTING = True

# 配置映射
config_map = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
