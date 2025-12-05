"""
llama.cpp 模型服务模块（可选）
使用 llama-cpp-python 提供更快的 CPU 推理速度

安装方法：
pip install llama-cpp-python

注意：需要将模型转换为 GGUF 格式
"""
import os
import sys

# 添加配置路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import MODEL_CONFIG

try:
    from llama_cpp import Llama
    LLAMACPP_AVAILABLE = True
except ImportError:
    LLAMACPP_AVAILABLE = False
    print("警告: llama-cpp-python 未安装，无法使用 llama.cpp 加速")


class LlamaCppService:
    """
    llama.cpp 模型服务类
    提供基于 llama-cpp-python 的高性能 CPU 推理
    """
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LlamaCppService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if LlamaCppService._initialized:
            return
        
        self.model = None
        self.config = MODEL_CONFIG
        LlamaCppService._initialized = True
    
    def load_model(self, model_path: str, **kwargs):
        """
        加载 GGUF 格式的模型
        
        Args:
            model_path: GGUF 模型文件路径
            **kwargs: llama.cpp 的其他参数
        """
        if not LLAMACPP_AVAILABLE:
            raise ImportError("llama-cpp-python 未安装，请运行: pip install llama-cpp-python")
        
        if self.model is not None:
            print("模型已加载，跳过重复加载")
            return
        
        print(f"使用 llama.cpp 加载模型: {model_path}")
        
        # llama.cpp 参数
        llama_kwargs = {
            "model_path": model_path,
            "n_ctx": kwargs.get("n_ctx", 2048),  # 上下文长度
            "n_threads": kwargs.get("n_threads", None),  # CPU 线程数（None = 自动）
            "n_gpu_layers": kwargs.get("n_gpu_layers", 0),  # GPU 层数（0 = 纯 CPU）
            "verbose": kwargs.get("verbose", False),
        }
        
        self.model = Llama(**llama_kwargs)
        print("llama.cpp 模型加载完成!")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        生成文本回复
        
        Args:
            prompt: 用户输入的问题
            **kwargs: 生成参数
        
        Returns:
            生成的文本回复
        """
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")
        
        # 生成参数
        gen_kwargs = {
            "max_tokens": kwargs.get("max_new_tokens", self.config["max_new_tokens"]),
            "temperature": kwargs.get("temperature", self.config["temperature"]),
            "top_p": kwargs.get("top_p", self.config["top_p"]),
            "echo": False,  # 不回显输入
            "stop": kwargs.get("stop", ["</s>", "<|endoftext|>"]),
        }
        
        # 调用 llama.cpp 生成
        output = self.model(prompt, **gen_kwargs)
        
        # 提取生成的文本
        response = output["choices"][0]["text"]
        return response.strip()
    
    def generate_stream(self, prompt: str, **kwargs):
        """
        流式生成文本回复
        
        Args:
            prompt: 用户输入的问题
            **kwargs: 生成参数
        
        Yields:
            生成的文本 token
        """
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")
        
        # 生成参数
        gen_kwargs = {
            "max_tokens": kwargs.get("max_new_tokens", self.config["max_new_tokens"]),
            "temperature": kwargs.get("temperature", self.config["temperature"]),
            "top_p": kwargs.get("top_p", self.config["top_p"]),
            "stream": True,  # 启用流式输出
            "echo": False,
            "stop": kwargs.get("stop", ["</s>", "<|endoftext|>"]),
        }
        
        # 流式生成
        for output in self.model(prompt, **gen_kwargs):
            token = output["choices"][0]["text"]
            if token:
                yield token
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model is not None


# 全局 llama.cpp 服务实例
llamacpp_service = LlamaCppService()
