"""
API 路由模块
提供模型推理的 RESTful API
"""
from flask import Blueprint, request, jsonify, current_app, Response, stream_with_context
import json
from app.services.model_service import model_service

api_bp = Blueprint('api', __name__)


@api_bp.route('/chat', methods=['POST'])
def chat():
    """
    聊天接口 - 处理用户输入并返回模型回复
    
    请求体 JSON:
    {
        "prompt": "用户输入的问题",
        "max_new_tokens": 512,  // 可选
        "temperature": 0.7,     // 可选
        "top_p": 0.9           // 可选
    }
    
    响应:
    {
        "success": true,
        "response": "模型生成的回复",
        "error": null
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({
                'success': False,
                'response': None,
                'error': '请提供 prompt 参数'
            }), 400
        
        prompt = data['prompt']
        
        # 获取可选的生成参数
        gen_kwargs = {}
        if 'max_new_tokens' in data:
            gen_kwargs['max_new_tokens'] = int(data['max_new_tokens'])
        if 'temperature' in data:
            gen_kwargs['temperature'] = float(data['temperature'])
        if 'top_p' in data:
            gen_kwargs['top_p'] = float(data['top_p'])
        
        # 调用模型生成
        response = model_service.generate(
            prompt=prompt,
            **gen_kwargs
        )
        
        return jsonify({
            'success': True,
            'response': response,
            'error': None
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'response': None,
            'error': str(e)
        }), 500


@api_bp.route('/load_model', methods=['POST'])
def load_model():
    """
    加载模型接口
    用于手动触发模型加载（首次调用 /chat 会自动加载）
    """
    try:
        if model_service.is_loaded():
            return jsonify({
                'success': True,
                'message': '模型已加载'
            })
        
        model_service.load_model()
        return jsonify({
            'success': True,
            'message': '模型加载成功'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'模型加载失败: {str(e)}'
        }), 500


@api_bp.route('/model_status', methods=['GET'])
def model_status():
    """
    获取模型状态
    """
    return jsonify({
        'loaded': model_service.is_loaded(),
        'config': {
            'base_model': model_service.config['base_model_path'],
            'lora_adapter': model_service.config['lora_adapter_path'],
            'device': model_service.device if model_service.device else model_service.config['device']
        }
    })


@api_bp.route('/chat_stream', methods=['POST'])
def chat_stream():
    """
    流式聊天接口 - 使用 Server-Sent Events (SSE) 实时返回生成的文本
    
    请求体 JSON:
    {
        "prompt": "用户输入的问题",
        "max_new_tokens": 256,  // 可选
        "temperature": 0.7,     // 可选
        "top_p": 0.9           // 可选
    }
    
    响应: text/event-stream (SSE)
    data: {"token": "生成的文本片段"}
    data: {"done": true}
    """
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({
                'success': False,
                'error': '请提供 prompt 参数'
            }), 400
        
        prompt = data['prompt']
        
        # 获取可选的生成参数
        gen_kwargs = {}
        if 'max_new_tokens' in data:
            gen_kwargs['max_new_tokens'] = int(data['max_new_tokens'])
        if 'temperature' in data:
            gen_kwargs['temperature'] = float(data['temperature'])
        if 'top_p' in data:
            gen_kwargs['top_p'] = float(data['top_p'])
        
        def generate():
            """生成器函数，逐个 token 返回"""
            try:
                # 调用模型的流式生成
                for token in model_service.generate_stream(
                    prompt=prompt,
                    **gen_kwargs
                ):
                    # 发送 token
                    yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
                
                # 发送完成信号
                yield f"data: {json.dumps({'done': True})}\n\n"
                
            except Exception as e:
                # 发送错误信息
                yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
        
        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',
                'Connection': 'keep-alive'
            }
        )
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
