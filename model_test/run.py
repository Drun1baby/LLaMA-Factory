#!/usr/bin/env python
"""
Flask 应用启动入口
"""
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app


# 创建应用实例
app = create_app(os.environ.get('FLASK_ENV', 'development'))


if __name__ == '__main__':
    # 获取配置
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 5005))
    debug = os.environ.get('FLASK_DEBUG', 'true').lower() == 'true'
    
    print(f"""
    ╔══════════════════════════════════════════════════════════╗
    ║           微调模型测试 Web 服务                          ║
    ║   基于 Hunyuan-7B-Instruct + LoRA 适配器                 ║
    ╠══════════════════════════════════════════════════════════╣
    ║   访问地址: http://{host}:{port}                           ║
    ║   调试模式: {debug}                                        ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # 启动服务
    app.run(host=host, port=port, debug=debug, threaded=True)
