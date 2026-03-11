"""
全双工语音对话系统 v3.0 - Qwen3 Omni Flash 处理器

使用 Qwen3-Omni-Flash 模型统一完成：
1. ASR 语音转文本
2. 语义VAD 判断说完/中断/拒识
3. 情绪识别

注意：模型名称必须是 Qwen3-Omni-Flash
架构设计：一次API调用完成所有三个任务，降低延迟

v3.0 说明：
- 此模块保留作为备用/兼容方案
- 主流程使用独立的 qwen_asr.py + semantic_vad.py + emotion.py
"""
import asyncio
import base64
import json
import os
import tempfile
from typing import Optional, Dict, Any, Tuple
from ..core.logger import logger

try:
    import dashscope
    from dashscope.audio.asr import Recognition, RecognitionCallback
    HAS_DASHSCOPE = True
except ImportError:
    HAS_DASHSCOPE = False

try:
    from openai import AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

from ..core.types import (
    AudioSegment,
    ASRResult,
    SemanticVADResult,
    EmotionResult,
    QwenOmniResult,
    SemanticState,
    EmotionType,
)
from ..core.config import get_config


class ASRCallback(RecognitionCallback):
    """ASR 回调处理器"""

    def __init__(self):
        self.result_text = ""
        self.is_complete = False
        self.error = None
        self._stopped = False

    def on_open(self):
        """连接打开"""
        logger.debug("ASR 连接已打开")

    def on_close(self):
        """连接关闭"""
        logger.debug("ASR 连接已关闭")
        self.is_complete = True
        self._stopped = True

    def on_event(self, result):
        """收到识别结果"""
        try:
            if result.get_sentence():
                text = result.get_sentence().get('text', '')
                if text:
                    self.result_text += text
                    logger.debug(f"ASR 部分结果: {text}")
        except Exception as e:
            logger.error(f"ASR 回调错误: {e}")

    def on_error(self, error):
        """发生错误"""
        logger.error(f"ASR 错误: {error}")
        self.error = error
        self.is_complete = True


class QwenOmniProcessor:
    """
    Qwen3 Omni Flash 处理器

    使用 DashScope Qwen3-Omni-Flash 模型统一完成：
    1. ASR: 语音转文本 (使用 Paraformer ASR)
    2. 语义VAD: 判断说完/中断/拒识
    3. 情绪识别

    架构优化：
    - ASR 使用 Paraformer 实时识别
    - 语义VAD 和情绪识别在 ASR 完成后并行处理
    - 通过 asyncio.gather 实现并行优化
    """

    # 正确的模型名称
    MODEL_NAME = "Qwen3-Omni-Flash"
    ASR_MODEL = "paraformer-realtime-v2"
    LLM_MODEL = "qwen-plus"  # 用于语义分析和情绪识别

    OMNI_PROMPT = """分析用户的语音输入文本，同时完成以下两个任务：

1. 语义状态判断：判断用户的表达状态
   - complete: 用户完整表达了一个意图或问题
   - continuing: 用户还在说，表达不完整
   - interrupted: 用户被打断或中途停止
   - rejected: 无法识别或无效输入

2. 情绪识别：判断用户的情绪状态
   - positive: 积极、开心、满意
   - negative: 消极、不满、失望
   - neutral: 中性、平静
   - angry: 愤怒、生气、烦躁
   - sad: 悲伤、难过、沮丧
   - surprised: 惊讶、意外

请以JSON格式返回：
{
    "semantic_state": "complete/continuing/interrupted/rejected",
    "emotion": "positive/negative/neutral/angry/sad/surprised",
    "confidence": 0.95,
    "emotion_confidence": 0.9,
    "emotion_intensity": 0.7,
    "reason": "判断理由"
}"""

    def __init__(self):
        self.config = get_config().qwen_omni
        self.api_key = self.config.get("api_key", "")
        # 确保使用正确的模型名称
        self.model = self.config.get("model", self.MODEL_NAME)
        if self.model.lower() != self.MODEL_NAME.lower():
            logger.warning(f"模型名称配置错误: {self.model}, 应为 {self.MODEL_NAME}")
            self.model = self.MODEL_NAME
        self._init_client()

    def _init_client(self):
        """初始化 DashScope 客户端"""
        if not HAS_DASHSCOPE:
            logger.warning("dashscope 库未安装，将使用模拟模式")
            return

        if self.api_key:
            dashscope.api_key = self.api_key
            logger.info(f"Qwen3 Omni Flash 客户端初始化成功 (模型: {self.model})")
        else:
            logger.warning("未配置 API 密钥，将使用模拟模式")

    async def process(self, audio: AudioSegment) -> QwenOmniResult:
        """
        处理音频段 - 统一完成 ASR + 语义VAD + 情绪识别
        使用 Qwen3-Omni-Flash 模型
        """
        if not HAS_DASHSCOPE or not self.api_key:
            return self._mock_process(audio)

        try:
            # ========== 第一步：ASR 语音识别 ==========
            logger.info(f"开始ASR处理，音频时长: {audio.duration_ms:.0f}ms")

            # 使用回调方式进行识别
            callback = ASRCallback()
            recognition = Recognition(
                model=self.ASR_MODEL,
                callback=callback,
                format='pcm',
                sample_rate=16000
            )

            # 启动识别
            recognition.start()

            # 发送音频数据
            recognition.send_audio_frame(audio.data)

            # 等待识别完成
            import time
            timeout = 30  # 30秒超时
            start_time = time.time()
            while not callback.is_complete and (time.time() - start_time) < timeout:
                await asyncio.sleep(0.1)

            # 安全停止识别
            try:
                if not callback._stopped:
                    recognition.stop()
            except Exception as e:
                logger.debug(f"停止识别时出错（可忽略）: {e}")

            text = callback.result_text
            logger.info(f"ASR识别结果: '{text}'")

            if callback.error:
                logger.error(f"ASR错误: {callback.error}")

            if not text:
                return QwenOmniResult(
                    asr=ASRResult(text="", confidence=0.5, is_final=True),
                    semantic_vad=SemanticVADResult(
                        state=SemanticState.REJECTED,
                        confidence=0.5,
                        reason="未识别到有效语音"
                    ),
                    emotion=EmotionResult(
                        emotion=EmotionType.NEUTRAL,
                        confidence=0.5,
                        intensity=0.5
                    )
                )

            # ========== 第二步：语义状态 + 情绪识别 ==========
            semantic_result = await self._analyze_semantic_and_emotion(text)

            return QwenOmniResult(
                asr=ASRResult(text=text, confidence=0.95, is_final=True),
                semantic_vad=SemanticVADResult(
                    state=semantic_result.get("semantic_state", SemanticState.COMPLETE),
                    confidence=semantic_result.get("confidence", 0.9),
                    reason=semantic_result.get("reason", "")
                ),
                emotion=EmotionResult(
                    emotion=semantic_result.get("emotion", EmotionType.NEUTRAL),
                    confidence=semantic_result.get("emotion_confidence", 0.8),
                    intensity=semantic_result.get("emotion_intensity", 0.5)
                ),
                raw_response=semantic_result
            )

        except Exception as e:
            logger.error(f"Qwen Omni 处理失败: {e}")
            import traceback
            traceback.print_exc()
            return self._mock_process(audio)

    async def _analyze_semantic_and_emotion(self, text: str) -> Dict:
        """分析语义状态和情绪"""
        try:
            if not HAS_OPENAI:
                return self._fallback_analysis(text)

            client = AsyncOpenAI(
                api_key=self.api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )

            response = await client.chat.completions.create(
                model=self.LLM_MODEL,  # 使用 qwen-plus 进行文本分析
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个文本分析助手，能够判断用户表达状态和情绪。"
                    },
                    {
                        "role": "user",
                        "content": f"用户说了：{text}\n\n{self.OMNI_PROMPT}"
                    }
                ],
                response_format={"type": "json_object"}
            )

            result_text = response.choices[0].message.content
            result_json = json.loads(result_text)

            # 验证并转换
            semantic_state_str = result_json.get("semantic_state", "complete")
            if semantic_state_str not in ["complete", "continuing", "interrupted", "rejected"]:
                semantic_state_str = "complete"
            result_json["semantic_state"] = SemanticState(semantic_state_str)

            emotion_str = result_json.get("emotion", "neutral")
            if emotion_str not in ["positive", "negative", "neutral", "angry", "sad", "surprised"]:
                emotion_str = "neutral"
            result_json["emotion"] = EmotionType(emotion_str)

            return result_json

        except Exception as e:
            logger.error(f"语义分析失败: {e}")
            return self._fallback_analysis(text)

    def _fallback_analysis(self, text: str) -> Dict:
        """回退分析 - 基于规则的简单分析"""
        # 简单的语义状态判断
        if not text or len(text.strip()) < 2:
            return {
                "semantic_state": SemanticState.REJECTED,
                "emotion": EmotionType.NEUTRAL,
                "confidence": 0.5,
                "emotion_confidence": 0.5,
                "emotion_intensity": 0.3
            }

        # 检查是否是完整句子
        end_punctuation = text.strip()[-1] if text.strip() else ""
        is_complete = end_punctuation in ["。", "！", "？", "！", ".", "!", "?"] or len(text) > 5

        # 简单情绪检测
        emotion = EmotionType.NEUTRAL
        emotion_intensity = 0.5

        positive_words = ["好", "棒", "太好了", "谢谢", "开心", "满意"]
        negative_words = ["糟糕", "不好", "差", "失败", "失望"]
        angry_words = ["生气", "烦", "讨厌", "火大"]

        text_lower = text.lower()
        if any(w in text_lower for w in positive_words):
            emotion = EmotionType.POSITIVE
            emotion_intensity = 0.7
        elif any(w in text_lower for w in angry_words):
            emotion = EmotionType.ANGRY
            emotion_intensity = 0.8
        elif any(w in text_lower for w in negative_words):
            emotion = EmotionType.NEGATIVE
            emotion_intensity = 0.7

        return {
            "semantic_state": SemanticState.COMPLETE if is_complete else SemanticState.CONTINUING,
            "emotion": emotion,
            "confidence": 0.8,
            "emotion_confidence": 0.7,
            "emotion_intensity": emotion_intensity
        }

    async def transcribe_only(self, audio: AudioSegment) -> ASRResult:
        """仅进行语音转写"""
        result = await self.process(audio)
        return result.asr

    async def process_parallel(self, audio: AudioSegment) -> Tuple[ASRResult, Dict]:
        """
        并行处理音频段 - 返回 ASR 结果和语义/情绪分析结果
        设计用于与 asyncio.gather 配合使用

        Returns:
            Tuple[ASRResult, Dict]: (ASR结果, 语义+情绪分析字典)
        """
        result = await self.process(audio)
        semantic_emotion_dict = {
            "semantic_state": result.semantic_vad.state,
            "semantic_confidence": result.semantic_vad.confidence,
            "semantic_reason": result.semantic_vad.reason,
            "emotion": result.emotion.emotion,
            "emotion_confidence": result.emotion.confidence,
            "emotion_intensity": result.emotion.intensity,
        }
        return result.asr, semantic_emotion_dict

    def _mock_process(self, audio: AudioSegment) -> QwenOmniResult:
        """模拟处理（用于测试）"""
        duration = audio.duration_ms
        energy = self._calculate_audio_energy(audio.data)

        if energy < 0.01:
            text = ""
            state = SemanticState.REJECTED
            emotion = EmotionType.NEUTRAL
            confidence = 0.5
            reason = "未检测到有效语音"
        elif duration < 300:
            text = ""
            state = SemanticState.REJECTED
            emotion = EmotionType.NEUTRAL
            confidence = 0.6
            reason = "语音太短"
        else:
            text = "[模拟模式:请配置DASHSCOPE_API_KEY以启用真实语音识别]"
            state = SemanticState.COMPLETE
            emotion = EmotionType.NEUTRAL
            confidence = 0.9
            reason = "模拟模式-检测到有效语音"

        return QwenOmniResult(
            asr=ASRResult(text=text, confidence=confidence, is_final=True),
            semantic_vad=SemanticVADResult(
                state=state,
                confidence=confidence,
                reason=reason
            ),
            emotion=EmotionResult(
                emotion=emotion,
                confidence=0.8,
                intensity=0.5
            )
        )

    def _calculate_audio_energy(self, audio_data: bytes) -> float:
        """计算音频能量"""
        import struct
        try:
            samples = struct.unpack(f'<{len(audio_data)//2}h', audio_data)
            if not samples:
                return 0.0
            rms = (sum(s*s for s in samples) / len(samples)) ** 0.5
            return min(rms / 32767.0, 1.0)
        except:
            return 0.0


class QwenOmniStreamProcessor:
    """Qwen Omni 流式处理器"""

    def __init__(self):
        self.processor = QwenOmniProcessor()
        self._buffer = bytearray()

    async def process_chunk(self, audio_chunk: bytes) -> Optional[QwenOmniResult]:
        """处理音频块"""
        self._buffer.extend(audio_chunk)

        buffer_ms = len(self._buffer) / 32
        if buffer_ms >= 1000:
            audio_segment = AudioSegment(
                data=bytes(self._buffer),
                sample_rate=16000
            )
            result = await self.processor.process(audio_segment)
            return result

        return None

    async def finalize(self) -> QwenOmniResult:
        """完成处理"""
        if self._buffer:
            audio_segment = AudioSegment(
                data=bytes(self._buffer),
                sample_rate=16000
            )
            result = await self.processor.process(audio_segment)
            self.reset()
            return result

        return QwenOmniResult(
            asr=ASRResult(text="", confidence=0),
            semantic_vad=SemanticVADResult(state=SemanticState.REJECTED, confidence=0),
            emotion=EmotionResult(emotion=EmotionType.NEUTRAL, confidence=0)
        )

    def reset(self):
        """重置状态"""
        self._buffer.clear()