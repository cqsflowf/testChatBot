"""
全双工语音对话系统 v3.0 - 配置加载
支持配置验证、默认值处理、环境变量替换

v3.0 新增配置：
- QWEN_ASR: Qwen ASR 17B 流式配置
- SEMANTIC_VAD: 语义VAD流式判断配置
- EMOTION: 并行情绪识别配置
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# 自动加载 .env 文件
try:
    from dotenv import load_dotenv
    # 查找 .env 文件
    env_paths = [
        Path(__file__).parent.parent.parent.parent / ".env",
        Path.cwd() / ".env",
    ]
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            break
except ImportError:
    pass


@dataclass
class ConfigValidationResult:
    """配置验证结果"""
    valid: bool
    errors: List[str]
    warnings: List[str]


class ConfigError(Exception):
    """配置错误"""
    pass


class Config:
    """配置管理器 v3.0 - 支持验证和默认值"""

    _instance: Optional['Config'] = None
    _config: Dict[str, Any] = {}

    # 默认配置 v3.4
    DEFAULTS = {
        "LLM": {
            "model": "qwen-flash",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "generation": {
                "temperature": 0.7,
                "max_tokens": 1024
            },
            "emotion_context": {
                "enabled": True
            },
            "tools": {
                "enabled": True
            }
        },
        "QWEN_ASR": {
            "model": "paraformer-realtime-v2",
            "frame_duration_ms": 20,
            "sample_rate": 16000
        },
        "QWEN_OMNI": {
            "model": "qwen3-omni-flash"
        },
        "SEMANTIC_VAD": {
            "enabled": True,
            "model": "qwen3-omni-flash",
            "streaming": {
                "min_text_length": 2,
                "max_wait_ms": 5000
            }
        },
        "EMOTION": {
            "mode": "parallel",
            "model": "qwen3-omni-flash",
            "sentence_based": True,
            "use_audio": True  # v3.3: 启用音频情绪识别
        },
        "TTS": {
            "provider": "qwen3",  # v3.3: 默认使用Qwen3 TTS
            "model": "qwen3-tts-flash",
            "voice": "Cherry",
            "sample_rate": 24000
        },
        "ACOUSTIC_VAD": {
            "engine": "silero",
            "aggressiveness": 3,
            "frame_duration_ms": 20,
            "silence_threshold_ms": 500
        },
        "SERVER": {
            "host": "0.0.0.0",
            "port": 8765
        },
        "SYSTEM": {
            "conversation": {
                "max_history": 10
            },
            "interrupt": {
                "enabled": True
            }
        }
    }

    def __new__(cls, config_path: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def __init__(self, config_path: Optional[str] = None):
        if not self._loaded:
            self.config_path = config_path or self._find_config()
            self.load()
            self._loaded = True

    def _find_config(self) -> str:
        """查找配置文件"""
        possible_paths = [
            Path(__file__).parent.parent.parent.parent / "config" / "model_config.yaml",
            Path.cwd() / "config" / "model_config.yaml",
            Path.cwd() / "model_config.yaml",
        ]
        for p in possible_paths:
            if p.exists():
                return str(p)
        # 如果配置文件不存在，使用默认配置
        logger_msg = "未找到配置文件，使用默认配置"
        print(f"警告: {logger_msg}")
        return ""

    def load(self):
        """加载配置"""
        if self.config_path and Path(self.config_path).exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f) or {}
        else:
            self._config = {}

        # 应用默认值
        self._apply_defaults()

        # 环境变量替换
        self._resolve_env_vars(self._config)

        # 验证配置
        validation = self.validate()
        if not validation.valid:
            raise ConfigError(f"配置验证失败: {validation.errors}")
        for warning in validation.warnings:
            print(f"配置警告: {warning}")

    def _apply_defaults(self):
        """应用默认配置值"""
        def deep_merge(base: Dict, defaults: Dict) -> Dict:
            for key, value in defaults.items():
                if key not in base:
                    base[key] = value
                elif isinstance(value, dict) and isinstance(base.get(key), dict):
                    deep_merge(base[key], value)
            return base

        deep_merge(self._config, self.DEFAULTS)

    def _resolve_env_vars(self, config: Dict):
        """递归替换环境变量"""
        for key, value in config.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                config[key] = os.getenv(env_var, "")
            elif isinstance(value, dict):
                self._resolve_env_vars(value)

    def validate(self) -> ConfigValidationResult:
        """验证配置"""
        errors = []
        warnings = []

        # 检查 API Key
        api_key = self._get_nested("QWEN_OMNI", "api_key") or os.getenv("DASHSCOPE_API_KEY", "")
        if not api_key:
            warnings.append("未配置 DASHSCOPE_API_KEY，将使用模拟模式")

        # 检查 LLM 配置
        llm_config = self._config.get("LLM", {})
        if not llm_config.get("model"):
            errors.append("LLM 模型未配置")

        # 检查服务器配置
        server_config = self._config.get("SERVER", {})
        port = server_config.get("port", 8765)
        if not (1 <= port <= 65535):
            errors.append(f"无效的服务器端口: {port}")

        return ConfigValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def _get_nested(self, *keys) -> Any:
        """获取嵌套配置值"""
        result = self._config
        for key in keys:
            if isinstance(result, dict) and key in result:
                result = result[key]
            else:
                return None
        return result

    def get(self, *keys, default=None):
        """获取嵌套配置值"""
        result = self._config
        for key in keys:
            if isinstance(result, dict) and key in result:
                result = result[key]
            else:
                return default
        return result

    @property
    def qwen_asr(self) -> Dict:
        """Qwen ASR 17B 配置"""
        return self._config.get("QWEN_ASR", {})

    @property
    def qwen_omni(self) -> Dict:
        """Qwen Omni Flash 配置"""
        return self._config.get("QWEN_OMNI", {})

    @property
    def semantic_vad(self) -> Dict:
        """语义VAD配置"""
        return self._config.get("SEMANTIC_VAD", {})

    @property
    def emotion(self) -> Dict:
        """情绪识别配置"""
        return self._config.get("EMOTION", {})

    @property
    def llm(self) -> Dict:
        """LLM配置"""
        return self._config.get("LLM", {})

    @property
    def tts(self) -> Dict:
        """TTS配置"""
        return self._config.get("TTS", {})

    @property
    def acoustic_vad(self) -> Dict:
        """声学VAD配置"""
        return self._config.get("ACOUSTIC_VAD", {})

    @property
    def server(self) -> Dict:
        """服务器配置"""
        return self._config.get("SERVER", {})

    @property
    def system(self) -> Dict:
        """系统配置"""
        return self._config.get("SYSTEM", {})

    # 兼容旧属性名
    @property
    def asr(self) -> Dict:
        """ASR配置（兼容）"""
        return self._config.get("ASR", self.qwen_asr)

    def reload(self):
        """重新加载配置"""
        self._loaded = False
        self.load()
        self._loaded = True

    def get_api_key(self) -> str:
        """获取 API Key（优先从环境变量）"""
        # 优先从环境变量获取
        api_key = os.getenv("DASHSCOPE_API_KEY", "")
        if api_key:
            return api_key
        # 其次从配置文件获取
        return self.qwen_omni.get("api_key", "") or self.llm.get("api_key", "")


def get_config(config_path: Optional[str] = None) -> Config:
    """获取配置实例"""
    return Config(config_path)
