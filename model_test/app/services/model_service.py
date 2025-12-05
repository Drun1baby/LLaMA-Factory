"""
模型服务模块
负责加载微调模型并提供推理接口
"""
import os
import sys

# 在导入 torch 之前设置环境变量，避免 MPS 后端的问题
# Apple Silicon 上 MPS 对大模型支持有限，强制使用 CPU
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # 禁用 MPS 内存缓存

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# 添加配置路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import MODEL_CONFIG


class ModelService:
    """
    模型服务类
    单例模式，负责模型的加载、管理和推理
    """
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if ModelService._initialized:
            return
        
        self.model = None
        self.tokenizer = None
        self.device = None
        self.config = MODEL_CONFIG
        ModelService._initialized = True
    
    def load_model(self):
        """
        加载基础模型和 LoRA 适配器
        """
        if self.model is not None:
            print("模型已加载，跳过重复加载")
            return
        
        print("开始加载模型...")
        
        base_model_path = self.config["base_model_path"]
        lora_adapter_path = self.config["lora_adapter_path"]
        
        # 确定设备
        # 注意：Apple Silicon 的 MPS 后端对 7B+ 大模型支持有限
        # 会出现 "total bytes of NDArray > 2**32" 错误
        # 因此在 Mac 上强制使用 CPU
        if self.config["device"] == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                # MPS 对大模型支持有限，7B 模型建议使用 CPU
                print("检测到 MPS 设备，但由于 7B 模型内存限制，将使用 CPU")
                self.device = "cpu"
            else:
                self.device = "cpu"
        else:
            self.device = self.config["device"]
        
        # 如果强制使用 MPS 但遇到问题，可以在配置中设置 device: "cpu"
        print(f"使用设备: {self.device}")
        
        # 配置量化（如果需要）
        quantization_config = None
        if self.config["quantization"] == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif self.config["quantization"] == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        
        # 设置 torch_dtype
        # 使用 bfloat16 可以显著提升 CPU 推理速度（约 2-3 倍）
        # 现代 CPU（Intel 第 3 代 Xeon 及以上，Apple M1/M2/M3）都支持 bfloat16
        if self.device == "cuda":
            torch_dtype = torch.float16
        else:
            # CPU 模式：使用 bfloat16 提升速度，如遇到兼容性问题可改回 float32
            torch_dtype = torch.bfloat16
        
        # 加载分词器
        print(f"加载分词器: {lora_adapter_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            lora_adapter_path,
            trust_remote_code=True
        )
        
        # 加载基础模型
        print(f"加载基础模型: {base_model_path}")
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch_dtype,
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        elif self.device != "cpu":
            model_kwargs["device_map"] = "auto"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            **model_kwargs
        )
        
        # 加载 LoRA 适配器
        print(f"加载 LoRA 适配器: {lora_adapter_path}")
        self.model = PeftModel.from_pretrained(
            self.model,
            lora_adapter_path
        )
        
        # 合并 LoRA 权重以提高推理速度（推荐）
        print("合并 LoRA 权重以优化推理速度...")
        self.model = self.model.merge_and_unload()
        
        # 设置为评估模式
        self.model.eval()
        
        # 如果没有使用 device_map，手动移动到设备
        if self.device == "cpu" or (not quantization_config and self.device != "auto"):
            self.model = self.model.to(self.device)
        
        print("模型加载完成!")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        生成文本回复
        
        Args:
            prompt: 用户输入的问题
            **kwargs: 其他生成参数
        
        Returns:
            生成的文本回复
        """
        if self.model is None:
            self.load_model()
        
        # 构建消息格式（只有用户消息）
        messages = [{"role": "user", "content": prompt}]
        
        # 使用 tokenizer 的 apply_chat_template 方法
        try:
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            # 如果 chat_template 不可用，使用简单格式
            print(f"Chat template 应用失败: {e}，使用简单格式")
            input_text = f"<|startoftext|>{prompt}<|extra_0|>"
        
        # 分词
        inputs = self.tokenizer(input_text, return_tensors="pt")
        
        # 移动到正确的设备
        if hasattr(self.model, 'device'):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 移除 token_type_ids（Hunyuan/LLaMA 架构模型不需要此参数）
        inputs.pop("token_type_ids", None)
        
        # 生成参数
        gen_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", self.config["max_new_tokens"]),
            "temperature": kwargs.get("temperature", self.config["temperature"]),
            "top_p": kwargs.get("top_p", self.config["top_p"]),
            "do_sample": kwargs.get("temperature", self.config["temperature"]) > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            # 启用 KV cache 优化（默认已启用，显式声明）
            "use_cache": True,
        }
        
        # 生成回复
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        # 解码输出
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response.strip()
    
    def generate_stream(self, prompt: str, **kwargs):
        """
        流式生成文本回复（逐 token 返回）
        
        Args:
            prompt: 用户输入的问题
            **kwargs: 其他生成参数
        
        Yields:
            生成的文本 token
        """
        if self.model is None:
            self.load_model()
        
        # 构建消息格式（只有用户消息）
        messages = [{"role": "user", "content": prompt}]
        
        # 使用 tokenizer 的 apply_chat_template 方法
        try:
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            # 如果 chat_template 不可用，使用简单格式
            print(f"Chat template 应用失败: {e}，使用简单格式")
            input_text = f"<|startoftext|>{prompt}<|extra_0|>"
        
        # 分词
        inputs = self.tokenizer(input_text, return_tensors="pt")
        
        # 移动到正确的设备
        if hasattr(self.model, 'device'):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 移除 token_type_ids
        inputs.pop("token_type_ids", None)
        
        # 生成参数
        gen_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", self.config["max_new_tokens"]),
            "temperature": kwargs.get("temperature", self.config["temperature"]),
            "top_p": kwargs.get("top_p", self.config["top_p"]),
            "do_sample": kwargs.get("temperature", self.config["temperature"]) > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        
        # 使用 TextIteratorStreamer 实现流式输出
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        # 在单独的线程中运行生成
        generation_kwargs = {**inputs, **gen_kwargs, "streamer": streamer}
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # 逐个 yield 生成的文本
        for text in streamer:
            if text:
                yield text
        
        thread.join()
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model is not None


# 全局模型服务实例
model_service = ModelService()
