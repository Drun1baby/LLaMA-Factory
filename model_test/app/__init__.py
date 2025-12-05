"""
Flask 应用工厂
"""
from flask import Flask
import os
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config_map


def create_app(config_name='default'):
    """
    Flask 应用工厂函数
    
    Args:
        config_name: 配置名称 (development, production, testing, default)
    
    Returns:
        Flask 应用实例
    """
    app = Flask(__name__, 
                template_folder='templates',
                static_folder='static')
    
    # 加载配置
    app.config.from_object(config_map[config_name])
    
    # 注册蓝图
    from app.routes.main import main_bp
    from app.routes.api import api_bp
    
    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # 初始化模型服务（懒加载）
    with app.app_context():
        from app.services.model_service import ModelService
        app.model_service = None  # 延迟加载，首次请求时初始化
    
    return app
