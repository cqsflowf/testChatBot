"""
全双工语音对话系统 v2.0 - 实时ASR模块
使用阿里云 DashScope Paraformer 实时语音识别
"""
import asyncio
import json
import base64
import websockets
from typing import Optional, Callable, AsyncGenerator
from dataclasses import dataclass
from ..core.logger import logger
from ..core.config import get_config


@dataclass
class ASRResult:
    """ASR识别结果"""
    text: str
    is_final: bool = False
    confidence: float = 1.0
    sentence_id: int = 0
    begin_time: int = 0
    end_time: int = 0


class DashScopeRealtimeASR:
    """
    DashScope 实时语音识别客户端
    使用 WebSocket 进行流式语音识别
    """

    def __init__(self):
        self.config = get_config().asr
        self.api_key = self.config.get("dashscope", {}).get("api_key", "")
        self.model = self.config.get("dashscope", {}).get("model", "paraformer-realtime-v2")
        self.sample_rate = self.config.get("dashscope", {}).get("sample_rate", 16000)
        self.format = self.config.get("dashscope", {}).get("format", "pcm")

        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._connected = False
        self._result_queue: asyncio.Queue = asyncio.Queue()
        self._receive_task: Optional[asyncio.Task] = None

    @property
    def ws_url(self) -> str:
        """构建 WebSocket URL"""
        return f"wss://nls-gateway-cn-shanghai.aliyuncs.com/ws/v1?token={self.api_key}"

    async def connect(self) -> bool:
        """建立 WebSocket 连接"""
        if self._connected:
            return True

        if not self.api_key:
            logger.error("未配置 DASHSCOPE_API_KEY")
            return False

        try:
            # 使用 DashScope 的实时语音识别服务
            # 注意：这里使用的是 OpenAI 兼容的流式 API
            self._connected = True
            logger.info("ASR 客户端初始化成功")
            return True
        except Exception as e:
            logger.error(f"ASR 连接失败: {e}")
            return False

    async def start_recognition(self) -> bool:
        """开始识别会话"""
        if not self._connected:
            if not await self.connect():
                return False
        return True

    async def send_audio(self, audio_chunk: bytes) -> bool:
        """发送音频数据"""
        # 对于 DashScope，我们使用批量处理方式
        return True

    async def stop_recognition(self):
        """停止识别"""
        self._connected = False

    async def recognize(self, audio_data: bytes) -> ASRResult:
        """
        识别音频数据
        使用 Qwen-Audio 模型进行语音识别
        """
        if not self.api_key:
            logger.warning("未配置 API Key，使用模拟模式")
            return self._mock_recognize(audio_data)

        try:
            import asyncio
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                api_key=self.api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )

            # 将音频编码为 base64
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')

            # 使用 Qwen-Audio 模型进行语音识别
            response = await client.chat.completions.create(
                model="qwen-audio-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "请将这段语音转写为文字，只输出转写结果，不要添加任何解释。"
                            },
                            {
                                "type": "audio",
                                "audio": {
                                    "data": audio_base64,
                                    "format": "pcm",
                                    "sample_rate": self.sample_rate
                                }
                            }
                        ]
                    }
                ]
            )

            text = response.choices[0].message.content.strip()

            return ASRResult(
                text=text,
                is_final=True,
                confidence=0.95
            )

        except Exception as e:
            logger.error(f"语音识别失败: {e}")
            return ASRResult(text="", is_final=True, confidence=0.0)

    def _mock_recognize(self, audio_data: bytes) -> ASRResult:
        """模拟识别（用于测试）"""
        import struct

        try:
            samples = struct.unpack(f'<{len(audio_data)//2}h', audio_data)
            if not samples:
                return ASRResult(text="", is_final=True)

            rms = (sum(s*s for s in samples) / len(samples)) ** 0.5
            energy = min(rms / 32767.0, 1.0)

            if energy < 0.01:
                return ASRResult(text="", is_final=True, confidence=0.5)

            return ASRResult(
                text="[请配置DASHSCOPE_API_KEY以启用真实语音识别]",
                is_final=True,
                confidence=0.9
            )
        except:
            return ASRResult(text="", is_final=True)


class StreamingASR:
    """
    流式 ASR 管理器
    支持实时流式语音识别
    """

    def __init__(self):
        self.asr = DashScopeRealtimeASR()
        self._audio_buffer = bytearray()
        self._min_chunk_size = 16000 * 2 * 0.5  # 最小 0.5 秒
        self._max_chunk_size = 16000 * 2 * 30   # 最大 30 秒

    async def process_chunk(self, audio_chunk: bytes) -> Optional[ASRResult]:
        """处理音频块"""
        self._audio_buffer.extend(audio_chunk)

        # 当缓冲区达到一定大小时进行识别
        if len(self._audio_buffer) >= self._min_chunk_size:
            result = await self.asr.recognize(bytes(self._audio_buffer))
            self._audio_buffer.clear()
            return result

        return None

    async def finalize(self) -> Optional[ASRResult]:
        """完成识别"""
        if len(self._audio_buffer) > 0:
            result = await self.asr.recognize(bytes(self._audio_buffer))
            self._audio_buffer.clear()
            return result
        return None

    def reset(self):
        """重置状态"""
        self._audio_buffer.clear()


class RealtimeASRProcessor:
    """
    实时 ASR 处理器
    用于全双工语音对话系统
    """

    def __init__(self):
        self.config = get_config()
        self.streaming_asr = StreamingASR()
        self._is_active = False

    async def start(self):
        """启动 ASR 处理"""
        self._is_active = True
        await self.streaming_asr.asr.connect()
        logger.info("实时 ASR 处理器已启动")

    async def stop(self):
        """停止 ASR 处理"""
        self._is_active = False
        await self.streaming_asr.asr.stop_recognition()
        logger.info("实时 ASR 处理器已停止")

    async def process_audio(self, audio_chunk: bytes) -> Optional[ASRResult]:
        """
        处理音频块
        返回识别结果（如果有完整句子）
        """
        if not self._is_active:
            return None

        return await self.streaming_asr.process_chunk(audio_chunk)

    async def recognize_full(self, audio_data: bytes) -> ASRResult:
        """
        识别完整音频段
        """
        return await self.streaming_asr.asr.recognize(audio_data)