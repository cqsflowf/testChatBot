"""
全双工语音对话系统 v3.0 - Qwen ASR 17B 流式识别模块

职责：
- 流式语音转文本
- 帧长20ms
- 实时输出识别结果

注意：此模块独立于语义VAD和情绪识别
"""
import asyncio
import json
from typing import Optional, Callable, AsyncIterator
from ..core.logger import logger

try:
    import dashscope
    from dashscope.audio.asr import Recognition, RecognitionCallback
    HAS_DASHSCOPE = True
except ImportError:
    HAS_DASHSCOPE = False

from ..core.types import AudioSegment, ASRResult
from ..core.config import get_config


class StreamingASRCallback(RecognitionCallback):
    """流式ASR回调处理器"""

    def __init__(self, on_result: Optional[Callable] = None):
        self.result_text = ""
        self.partial_text = ""
        self.is_complete = False
        self.error = None
        self._stopped = False
        self._on_result = on_result
        self._loop = None  # 保存事件循环引用
        self._result_queue = []  # 存储结果的队列

    def set_loop(self, loop):
        """设置事件循环"""
        self._loop = loop

    def on_open(self):
        """连接打开"""
        logger.debug("ASR 流式连接已打开")

    def on_close(self):
        """连接关闭"""
        logger.debug("ASR 流式连接已关闭")
        self.is_complete = True
        self._stopped = True

    def on_event(self, result):
        """收到识别结果 - 流式输出"""
        try:
            if result.get_sentence():
                sentence = result.get_sentence()
                text = sentence.get('text', '')
                if text:
                    # 部分结果
                    self.partial_text = text
                    self.result_text = text
                    logger.debug(f"ASR 流式结果: {text}")

                    # 回调通知 - 使用线程安全的方式
                    if self._on_result and self._loop:
                        # 使用 call_soon_threadsafe 在主线程事件循环中调度
                        self._loop.call_soon_threadsafe(
                            lambda: asyncio.create_task(
                                self._on_result(text, is_final=False)
                            )
                        )
                    elif self._on_result:
                        # 没有事件循环，存储结果供后续处理
                        self._result_queue.append((text, False))
        except Exception as e:
            logger.error(f"ASR 回调错误: {e}")

    def on_error(self, error):
        """发生错误"""
        logger.error(f"ASR 错误: {error}")
        self.error = error
        self.is_complete = True


class QwenASRProcessor:
    """
    Qwen ASR 17B 流式处理器

    使用 DashScope Paraformer 实时识别模型
    帧长: 20ms
    流式输出文本结果
    """

    # 使用 Paraformer 实时识别
    MODEL_NAME = "paraformer-realtime-v2"
    FRAME_DURATION_MS = 20  # 帧长20ms

    def __init__(self):
        self.config = get_config().qwen_asr if hasattr(get_config(), 'qwen_asr') else {}
        self.api_key = self.config.get("api_key", "") or get_config().qwen_omni.get("api_key", "")
        self.model = self.config.get("model", self.MODEL_NAME)
        self.frame_duration_ms = self.config.get("frame_duration_ms", self.FRAME_DURATION_MS)

        self._recognition = None
        self._callback = None
        self._is_streaming = False
        self._text_buffer = ""

        self._init_client()

    def _init_client(self):
        """初始化 DashScope 客户端"""
        if not HAS_DASHSCOPE:
            logger.warning("dashscope 库未安装，将使用模拟模式")
            return

        if self.api_key:
            dashscope.api_key = self.api_key
            logger.info(f"Qwen ASR 17B 客户端初始化成功 (模型: {self.model}, 帧长: {self.frame_duration_ms}ms)")
        else:
            logger.warning("未配置 API 密钥，将使用模拟模式")

    async def start_stream(self, on_result: Optional[Callable] = None) -> bool:
        """
        开始流式识别会话

        Args:
            on_result: 流式结果回调函数 async def on_result(text: str, is_final: bool)

        Returns:
            是否成功启动
        """
        # 先停止旧的识别器
        if self._is_streaming:
            try:
                if self._recognition and self._callback and not self._callback._stopped:
                    self._recognition.stop()
            except Exception:
                pass
            self._is_streaming = False

        if not HAS_DASHSCOPE or not self.api_key:
            logger.warning("ASR 未初始化，使用模拟模式")
            self._is_streaming = True
            self._text_buffer = ""
            return True

        try:
            # 获取当前事件循环
            loop = asyncio.get_running_loop()

            self._callback = StreamingASRCallback(on_result)
            self._callback.set_loop(loop)  # 设置事件循环引用

            self._recognition = Recognition(
                model=self.model,
                callback=self._callback,
                format='pcm',
                sample_rate=16000
            )

            self._recognition.start()
            self._is_streaming = True
            self._text_buffer = ""

            logger.info("ASR 流式会话已启动")
            return True

        except Exception as e:
            logger.error(f"启动 ASR 流式会话失败: {e}")
            return False

    async def process_chunk(self, audio_chunk: bytes) -> Optional[str]:
        """
        处理音频块 - 流式输出

        Args:
            audio_chunk: 音频数据 (PCM 16kHz 16-bit mono)

        Returns:
            当前识别的文本（部分结果）
        """
        import time
        receive_time = time.time() * 1000
        logger.info(f"[ASR] 收到声学VAD音频流, 时间: {receive_time:.0f}ms, 数据大小: {len(audio_chunk)} bytes")

        if not self._is_streaming:
            logger.warning("ASR 流式会话未启动")
            return None

        if not HAS_DASHSCOPE or not self.api_key:
            # 模拟模式
            return await self._mock_process_chunk(audio_chunk)

        try:
            # 发送音频帧到识别器
            if self._recognition:
                self._recognition.send_audio_frame(audio_chunk)

            # 返回当前部分结果
            return self._callback.partial_text if self._callback else None

        except Exception as e:
            logger.error(f"ASR 处理音频块失败: {e}")
            return None

    async def stop_stream(self) -> ASRResult:
        """
        停止流式识别会话并获取最终结果

        Returns:
            最终的ASR识别结果
        """
        if not self._is_streaming:
            return ASRResult(text="", confidence=0.5, is_final=True)

        self._is_streaming = False

        if not HAS_DASHSCOPE or not self.api_key:
            # 模拟模式
            return ASRResult(
                text=self._text_buffer or "[模拟模式: ASR识别结果]",
                confidence=0.9,
                is_final=True
            )

        try:
            # 停止识别
            if self._recognition and self._callback and not self._callback._stopped:
                self._recognition.stop()

            final_text = self._callback.result_text if self._callback else ""
            logger.info(f"ASR 流式会话结束，最终结果: '{final_text}'")

            return ASRResult(
                text=final_text,
                confidence=0.95,
                is_final=True
            )

        except Exception as e:
            logger.error(f"停止 ASR 流式会话失败: {e}")
            return ASRResult(text="", confidence=0.5, is_final=True)

    async def process_segment(self, audio: AudioSegment) -> ASRResult:
        """
        处理完整音频段（非流式）

        Args:
            audio: 音频段

        Returns:
            ASR识别结果
        """
        await self.start_stream()

        # 分块发送音频
        chunk_size = int(16000 * 2 * self.frame_duration_ms / 1000)  # 20ms 帧大小
        for i in range(0, len(audio.data), chunk_size):
            chunk = audio.data[i:i + chunk_size]
            await self.process_chunk(chunk)
            await asyncio.sleep(0.01)  # 小延迟模拟实时

        return await self.stop_stream()

    async def _mock_process_chunk(self, audio_chunk: bytes) -> Optional[str]:
        """模拟处理音频块"""
        # 简单的能量检测
        import struct
        try:
            samples = struct.unpack(f'<{len(audio_chunk)//2}h', audio_chunk)
            if not samples:
                return None
            rms = (sum(s*s for s in samples) / len(samples)) ** 0.5
            if rms > 500:
                self._text_buffer = "[模拟ASR结果]"
                return self._text_buffer
        except:
            pass
        return None

    @property
    def is_streaming(self) -> bool:
        """是否正在流式识别"""
        return self._is_streaming

    def reset(self):
        """重置状态"""
        self._text_buffer = ""
        if self._callback:
            self._callback.result_text = ""
            self._callback.partial_text = ""


class QwenASRStreamIterator:
    """
    Qwen ASR 流式迭代器
    用于异步迭代音频流并输出识别结果
    """

    def __init__(self, processor: QwenASRProcessor):
        self.processor = processor
        self._queue = asyncio.Queue()
        self._running = False

    async def start(self):
        """启动迭代器"""
        self._running = True
        await self.processor.start_stream(self._on_result)

    async def _on_result(self, text: str, is_final: bool):
        """结果回调"""
        await self._queue.put((text, is_final))

    async def send_audio(self, audio_chunk: bytes):
        """发送音频数据"""
        if self._running:
            await self.processor.process_chunk(audio_chunk)

    async def __aiter__(self) -> AsyncIterator[tuple]:
        """异步迭代"""
        while self._running:
            try:
                text, is_final = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=0.5
                )
                yield text, is_final

                if is_final:
                    break
            except asyncio.TimeoutError:
                continue

    async def stop(self) -> ASRResult:
        """停止迭代器"""
        self._running = False
        return await self.processor.stop_stream()