"""
全双工语音对话系统 v3.0

架构特点：
- Qwen ASR 17B 流式识别（帧长20ms）
- Qwen Omni Flash 流式语义VAD
- Qwen Omni Flash 并行情绪识别
- 声学VAD阈值<500ms
- 支持打断的全双工交互
"""
from .core import (
    DialogState,
    SemanticState,
    EmotionType,
    AudioSegment,
    ASRResult,
    SemanticVADResult,
    QwenOmniResult,
    EmotionResult,
    LLMInput,
    ToolCall,
    ToolResult,
    LLMResponse,
    TTSResult,
    DialogResult,
    Message,
    Config,
    get_config,
    DialogStateMachine,
    SemanticStateMachine,
)
from .system import VoiceDialogSystem

__version__ = "3.0.0"

# 延迟导入服务器（避免在没有安装fastapi时导入失败）
def get_app():
    """获取FastAPI应用"""
    from .websocket_server import app
    return app

def get_run_server():
    """获取服务器启动函数"""
    from .websocket_server import run_server
    return run_server

__all__ = [
    # 类型
    "DialogState",
    "SemanticState",
    "EmotionType",
    "AudioSegment",
    "ASRResult",
    "SemanticVADResult",
    "QwenOmniResult",
    "EmotionResult",
    "LLMInput",
    "ToolCall",
    "ToolResult",
    "LLMResponse",
    "TTSResult",
    "DialogResult",
    "Message",
    # 配置
    "Config",
    "get_config",
    # 状态机
    "DialogStateMachine",
    "SemanticStateMachine",
    # 系统
    "VoiceDialogSystem",
    # 服务器（延迟导入）
    "get_app",
    "get_run_server",
]