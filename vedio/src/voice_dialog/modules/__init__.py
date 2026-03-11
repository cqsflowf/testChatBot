"""
全双工语音对话系统 v3.0 - 模块初始化

架构设计（v3.0）：
1. 声学VAD检测语音段（acoustic_vad.py）
   - 阈值<500ms
   - 检测到人声开始 → 打开音频流送给ASR
   - 检测到静音 → 不立刻断，交给语义VAD判断
2. Qwen ASR 17B 流式处理（qwen_asr.py）
   - 帧长20ms
   - 流式输出文本
3. Qwen Omni Flash 语义VAD（semantic_vad.py）
   - 流式文本实时检测
   - 边接收ASR文本边判断
4. Qwen Omni Flash 情绪识别（emotion.py）
   - 与语义VAD并行
   - 每次只判断完整一句话
5. LLM任务规划（llm_planner.py）
6. 工具引擎（tools.py）
7. TTS语音合成（tts.py）
   - v3.2.2: StreamingTTSProcessor 支持LLM流式输出实时转语音
"""
from .acoustic_vad import AcousticVAD, StreamingVAD
from .qwen_asr import QwenASRProcessor, QwenASRStreamIterator
from .semantic_vad import SemanticVADProcessor, StreamingSemanticVAD
from .emotion import EmotionRecognizer, ParallelEmotionRecognizer, recognize_emotion_from_text
from .llm_planner import LLMTaskPlanner
from .tools import ToolEngine, MCPClient, SKILLSEngine
from .tts import TTSEngine, StreamingTTS, StreamingTTSProcessor
# 保留旧模块以兼容
from .qwen_omni import QwenOmniProcessor, QwenOmniStreamProcessor

__all__ = [
    # 声学VAD
    "AcousticVAD",
    "StreamingVAD",
    # Qwen ASR 17B（流式）★新增
    "QwenASRProcessor",
    "QwenASRStreamIterator",
    # 语义VAD（流式判断）★新增
    "SemanticVADProcessor",
    "StreamingSemanticVAD",
    # 情绪识别（并行）
    "EmotionRecognizer",
    "ParallelEmotionRecognizer",
    "recognize_emotion_from_text",
    # Qwen Omni（保留兼容）
    "QwenOmniProcessor",
    "QwenOmniStreamProcessor",
    # LLM任务规划
    "LLMTaskPlanner",
    # 工具引擎
    "ToolEngine",
    "MCPClient",
    "SKILLSEngine",
    # TTS
    "TTSEngine",
    "StreamingTTS",
    "StreamingTTSProcessor",
]