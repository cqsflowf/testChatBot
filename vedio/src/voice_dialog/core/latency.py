"""
全双工语音对话系统 - 时延追踪模块

用于追踪用户输入在各关键模块的处理时延，便于定位定界和性能分析。

关键追踪点：
1. 语音开始 - 声学VAD检测到人声
2. ASR首字 - 首个文字识别结果
3. ASR完成 - 语音识别完成
4. 语义VAD判断 - 语义完整性判断
5. 情绪识别完成 - 情绪分析完成
6. LLM开始 - 进入LLM处理
7. LLM响应 - LLM返回结果
8. 工具调用 - 工具执行（如有）
9. TTS开始 - 开始语音合成
10. TTS完成 - 语音合成完成
11. 端到端 - 整体耗时
"""
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import threading


@dataclass
class LatencyRecord:
    """单个时延记录"""
    name: str              # 节点名称
    start_time: float      # 开始时间 (ms, 相对于句子开始)
    end_time: float        # 结束时间 (ms)
    duration: float        # 耗时 (ms)
    metadata: Dict = field(default_factory=dict)  # 附加信息

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "start_time": round(self.start_time, 2),
            "end_time": round(self.end_time, 2),
            "duration": round(self.duration, 2),
            "metadata": self.metadata
        }


@dataclass
class SentenceLatency:
    """一句话的完整时延记录"""
    sentence_id: str                    # 句子ID
    text: str = ""                      # 用户文本
    start_time: float = 0.0             # 句子开始时间戳 (绝对时间, ms)
    records: List[LatencyRecord] = field(default_factory=list)
    is_complete: bool = False           # 是否已完成

    # 关键指标
    first_asr_latency: float = 0.0      # ASR首字延迟 (ms)
    asr_total_latency: float = 0.0      # ASR总耗时 (ms)
    semantic_vad_latency: float = 0.0   # 语义VAD耗时 (ms)
    emotion_latency: float = 0.0        # 情绪识别耗时 (ms)
    llm_latency: float = 0.0            # LLM处理耗时 (ms)
    tool_latency: float = 0.0           # 工具调用耗时 (ms)
    tts_latency: float = 0.0            # TTS合成耗时 (ms)
    total_latency: float = 0.0          # 端到端总耗时 (ms)

    def to_dict(self) -> Dict:
        return {
            "sentence_id": self.sentence_id,
            "text": self.text,
            "start_time": round(self.start_time, 2),
            "is_complete": self.is_complete,
            "records": [r.to_dict() for r in self.records],
            "metrics": {
                "first_asr_latency": round(self.first_asr_latency, 2),
                "asr_total_latency": round(self.asr_total_latency, 2),
                "semantic_vad_latency": round(self.semantic_vad_latency, 2),
                "emotion_latency": round(self.emotion_latency, 2),
                "llm_latency": round(self.llm_latency, 2),
                "tool_latency": round(self.tool_latency, 2),
                "tts_latency": round(self.tts_latency, 2),
                "total_latency": round(self.total_latency, 2)
            }
        }


class LatencyTracker:
    """
    时延追踪器

    追踪一句话在系统中各阶段的时间开销

    使用方式：
    1. start_sentence() - 开始追踪新句子
    2. mark_start(name) - 标记节点开始
    3. mark_end(name) - 标记节点结束
    4. end_sentence() - 结束当前句子追踪
    """

    # 追踪节点名称
    NODES = {
        # 输入阶段
        "vad_detect": "声学VAD检测",
        "asr_streaming": "ASR流式识别",
        "asr_first_text": "ASR首字输出",

        # 理解阶段
        "semantic_vad": "语义VAD判断",
        "emotion": "情绪识别",

        # 处理阶段
        "llm_process": "LLM处理",
        "llm_first_token": "LLM首Token",
        "tool_execute": "工具调用",

        # 输出阶段
        "tts_synthesize": "TTS合成",
        "tts_first_chunk": "TTS首块输出",
    }

    # 节点颜色（用于可视化）
    NODE_COLORS = {
        "vad_detect": "#4CAF50",
        "asr_streaming": "#2196F3",
        "asr_first_text": "#03A9F4",
        "semantic_vad": "#9C27B0",
        "emotion": "#E91E63",
        "llm_process": "#FF9800",
        "llm_first_token": "#FFC107",
        "tool_execute": "#795548",
        "tts_synthesize": "#00BCD4",
        "tts_first_chunk": "#00E5FF",
    }

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True

        self._current_sentence: Optional[SentenceLatency] = None
        self._sentence_counter = 0
        self._pending_starts: Dict[str, float] = {}
        self._history: List[SentenceLatency] = []
        self._max_history = 100

        # 回调
        self._on_update_callbacks: List = []

    def start_sentence(self) -> str:
        """
        开始追踪新句子

        Returns:
            句子ID
        """
        self._sentence_counter += 1
        sentence_id = f"sentence_{self._sentence_counter}_{int(time.time()*1000)}"

        self._current_sentence = SentenceLatency(
            sentence_id=sentence_id,
            start_time=time.time() * 1000
        )
        self._pending_starts.clear()

        # 标记语音开始
        self.mark_start("vad_detect")

        return sentence_id

    def mark_start(self, name: str, metadata: Dict = None):
        """
        标记节点开始

        Args:
            name: 节点名称
            metadata: 附加信息
        """
        if not self._current_sentence:
            return

        relative_time = time.time() * 1000 - self._current_sentence.start_time
        self._pending_starts[name] = relative_time

        if metadata:
            if name not in self._current_sentence.records:
                pass  # 将在mark_end时处理

    def mark_end(self, name: str, metadata: Dict = None):
        """
        标记节点结束

        Args:
            name: 节点名称
            metadata: 附加信息
        """
        if not self._current_sentence:
            return

        end_relative_time = time.time() * 1000 - self._current_sentence.start_time

        if name in self._pending_starts:
            start_relative_time = self._pending_starts[name]
            duration = end_relative_time - start_relative_time
        else:
            # 如果没有开始时间，从句子开始计算
            start_relative_time = 0
            duration = end_relative_time

        record = LatencyRecord(
            name=name,
            start_time=start_relative_time,
            end_time=end_relative_time,
            duration=duration,
            metadata=metadata or {}
        )

        # 避免重复记录
        existing_names = [r.name for r in self._current_sentence.records]
        if name not in existing_names:
            self._current_sentence.records.append(record)

        # 更新关键指标
        self._update_metrics(name, duration, metadata)

        # 触发回调
        self._notify_update()

    def _update_metrics(self, name: str, duration: float, metadata: Dict = None):
        """更新关键指标"""
        if not self._current_sentence:
            return

        if name == "asr_first_text":
            self._current_sentence.first_asr_latency = duration
        elif name == "asr_streaming":
            self._current_sentence.asr_total_latency = duration
        elif name == "semantic_vad":
            self._current_sentence.semantic_vad_latency = duration
        elif name == "emotion":
            self._current_sentence.emotion_latency = duration
        elif name == "llm_process":
            self._current_sentence.llm_latency = duration
        elif name == "tool_execute":
            self._current_sentence.tool_latency = duration
        elif name == "tts_synthesize":
            self._current_sentence.tts_latency = duration

        # 更新文本
        if metadata and "text" in metadata:
            self._current_sentence.text = metadata["text"]

    def end_sentence(self) -> Optional[SentenceLatency]:
        """
        结束当前句子追踪

        Returns:
            完整的时延记录
        """
        if not self._current_sentence:
            return None

        # 计算总耗时
        self._current_sentence.total_latency = time.time() * 1000 - self._current_sentence.start_time
        self._current_sentence.is_complete = True

        # 添加到历史
        self._history.append(self._current_sentence)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        result = self._current_sentence
        self._current_sentence = None
        self._pending_starts.clear()

        # 触发回调
        self._notify_update()

        return result

    def update_text(self, text: str):
        """更新当前句子的文本"""
        if self._current_sentence:
            self._current_sentence.text = text
            self._notify_update()

    def get_current(self) -> Optional[SentenceLatency]:
        """获取当前句子"""
        return self._current_sentence

    def get_history(self, limit: int = 10) -> List[SentenceLatency]:
        """获取历史记录"""
        return self._history[-limit:]

    def get_stats(self) -> Dict:
        """获取统计信息"""
        if not self._history:
            return {
                "total_sentences": 0,
                "avg_total_latency": 0,
                "avg_asr_latency": 0,
                "avg_llm_latency": 0,
                "avg_tts_latency": 0
            }

        total = len(self._history)
        avg_total = sum(s.total_latency for s in self._history) / total
        avg_asr = sum(s.first_asr_latency for s in self._history if s.first_asr_latency > 0) / max(1, sum(1 for s in self._history if s.first_asr_latency > 0))
        avg_llm = sum(s.llm_latency for s in self._history if s.llm_latency > 0) / max(1, sum(1 for s in self._history if s.llm_latency > 0))
        avg_tts = sum(s.tts_latency for s in self._history if s.tts_latency > 0) / max(1, sum(1 for s in self._history if s.tts_latency > 0))

        return {
            "total_sentences": total,
            "avg_total_latency": round(avg_total, 2),
            "avg_asr_latency": round(avg_asr, 2),
            "avg_llm_latency": round(avg_llm, 2),
            "avg_tts_latency": round(avg_tts, 2)
        }

    def on_update(self, callback):
        """注册更新回调"""
        self._on_update_callbacks.append(callback)

    def _notify_update(self):
        """通知更新"""
        data = self.get_current()
        for callback in self._on_update_callbacks:
            try:
                callback(data)
            except Exception as e:
                pass

    def reset(self):
        """重置追踪器"""
        self._current_sentence = None
        self._pending_starts.clear()
        self._history.clear()


# 全局单例
latency_tracker = LatencyTracker()