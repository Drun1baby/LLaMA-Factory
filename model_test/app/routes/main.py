"""
主页路由模块
"""
from flask import Blueprint, render_template

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def index():
    """主页 - 提供模型测试界面"""
    return render_template('index.html')


@main_bp.route('/health')
def health():
    """健康检查接口"""
    return {'status': 'healthy'}
