"""
全双工语音对话系统 v3.3 - 声学VAD模块

核心职责（v3.3）：
1. 语音起止与打断检测
2. 阈值 < 500ms
3. 检测到人声开始 → 打开音频流，送给ASR
4. 检测到静音 → 不立刻断，交给语义VAD判断
5. 检测到打断 → 停止播报，开始新一轮对话

v3.3 更新：
- 使用 Silero VAD 作为主要语音检测引擎
- 简单音量 VAD 作为备选和缓冲期间快速响应
- 移除 WebRTC VAD 依赖
"""
import struct
import time
import numpy as np
from typing import Optional, Callable, List
from ..core.logger import logger

from ..core.types import AudioSegment
from ..core.config import get_config


class SileroVADWrapper:
    """
    Silero VAD 包装器 v4

    解决延迟问题：
    - 缓冲期间使用简单VAD快速响应
    - 累积512样本后切换到Silero精准检测
    """

    def __init__(self):
        self._model = None
        self._sample_rate = 16000
        self._required_samples = 512  # Silero 需要512样本
        self._buffer = bytearray()  # 滑动窗口缓冲区
        self._last_result = False  # 上一次检测结果
        self._initialized = False  # 是否已初始化
        self._fallback_threshold = 600  # v3.6: 降低到600，确保能检测到正常说话
        self._init_model()

    def _init_model(self):
        """初始化 Silero VAD 模型"""
        try:
            import torch

            # 加载 Silero VAD 模型
            self._model, self._utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False,
                trust_repo=True
            )
            self._model.eval()
            self._model.reset_states()

            logger.info("Silero VAD 加载成功 (内置自适应阈值)")

        except Exception as e:
            logger.warning(f"Silero VAD 加载失败: {e}")
            self._model = None

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        """
        实时检测语音

        混合检测策略：
        1. 缓冲期间：使用简单音量检测（快速响应）
        2. 缓冲完成后：使用Silero精准检测

        Args:
            frame: 音频帧 (PCM 16-bit, 320样本)
            sample_rate: 采样率

        Returns:
            是否包含语音
        """
        if self._model is None:
            return self._simple_detect(frame)

        try:
            import torch

            # 将新帧加入缓冲区
            self._buffer.extend(frame)

            # 检查样本数
            num_samples = len(self._buffer) // 2  # 16-bit = 2 bytes

            if num_samples < self._required_samples:
                # 缓冲期间：使用简单检测（快速响应语音开头）
                return self._simple_detect(frame)

            # 维持滑动窗口：只保留最新的512样本
            if num_samples > self._required_samples:
                excess = (num_samples - self._required_samples) * 2
                self._buffer = self._buffer[excess:]

            # 取最新的512样本进行Silero检测
            audio_bytes = bytes(self._buffer[:self._required_samples * 2])
            samples = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float = samples.astype(np.float32) / 32768.0
            audio_tensor = torch.from_numpy(audio_float).unsqueeze(0)

            # 使用Silero检测
            with torch.no_grad():
                speech_prob = self._model(audio_tensor, sample_rate).item()

            # v3.6: 降低阈值到0.55，确保能检测到正常说话
            # 概率>0.55认为是语音（从0.65降低到0.55）
            self._last_result = speech_prob > 0.55
            self._initialized = True

            return self._last_result

        except Exception as e:
            logger.debug(f"Silero VAD 检测错误: {e}")
            return self._last_result

    def _simple_detect(self, frame: bytes) -> bool:
        """简单音量检测（用于缓冲期间的快速响应）"""
        try:
            samples = struct.unpack(f'<{len(frame)//2}h', frame)
            if not samples:
                return False
            rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
            # 使用较低阈值，确保不错过语音开头
            return rms > self._fallback_threshold
        except Exception:
            return False

    def get_speech_prob(self, frame: bytes, sample_rate: int) -> float:
        """
        获取语音概率值（用于调试）

        Returns:
            语音概率 (0.0 - 1.0)
        """
        if self._model is None:
            return 0.0

        try:
            import torch

            # 累积帧
            self._buffer.extend(frame)
            num_samples = len(self._buffer) // 2

            if num_samples < self._required_samples:
                return 0.0

            if num_samples > self._required_samples:
                excess = (num_samples - self._required_samples) * 2
                self._buffer = self._buffer[excess:]

            audio_bytes = bytes(self._buffer[:self._required_samples * 2])
            samples = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float = samples.astype(np.float32) / 32768.0
            audio_tensor = torch.from_numpy(audio_float).unsqueeze(0)

            with torch.no_grad():
                return self._model(audio_tensor, sample_rate).item()

        except Exception:
            return 0.0

    def reset_states(self):
        """重置模型状态"""
        if self._model is not None:
            self._model.reset_states()
        self._buffer.clear()
        self._last_result = False
        self._initialized = False

    @property
    def available(self) -> bool:
        """VAD 是否可用"""
        return self._model is not None


class SimpleVAD:
    """
    简单音量 VAD
    基于 RMS 音量检测
    """

    def __init__(self, threshold: float = 600):  # v3.6: 降低到600，确保能检测到正常说话
        self.threshold = threshold

    def is_speech(self, frame: bytes, sample_rate: int = 16000) -> bool:
        """检测语音"""
        try:
            samples = struct.unpack(f'<{len(frame)//2}h', frame)
            if not samples:
                return False
            rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
            return rms > self.threshold
        except Exception:
            return False


class AcousticVAD:
    """
    声学VAD检测器 v3.0

    核心职责：
    1. 语音起止与打断检测
    2. 阈值 < 500ms
    3. 检测到人声开始 → 打开音频流，送给ASR
    4. 检测到静音 → 不立刻断，交给语义VAD判断
    5. 检测到打断 → 停止播报，开始新一轮对话

    设计要点：
    - 检测到静音后，不立刻断开，而是通知上层交给语义VAD判断
    - 只有语义VAD确认语义完整后，才结束当前语音段
    """

    def __init__(self):
        self.config = get_config().acoustic_vad
        self._is_speech = False
        self._speech_frames: List[bytes] = []
        self._silence_frames = 0
        self._speech_callbacks: List[Callable] = []
        self._silence_callbacks: List[Callable] = []  # 静音回调（给语义VAD判断）
        self._interrupt_callbacks: List[Callable] = []  # 打断回调

        # 配置参数 - 按业界最佳实践配置
        self.frame_duration_ms = self.config.get("frame_duration_ms", 20)  # 帧长20ms
        self.silence_threshold_ms = self.config.get("silence_threshold_ms", 500)  # 静音超时500ms
        self.padding_duration_ms = self.config.get("padding_duration_ms", 150)
        self.sample_rate = 16000
        self.aggressiveness = self.config.get("aggressiveness", 3)

        # 预缓存配置 - 200ms回溯，避免丢失语音开头
        self.prebuffer_duration_ms = 200  # 预缓存200ms
        self.prebuffer_frames = self.prebuffer_duration_ms // self.frame_duration_ms  # 10帧
        self._prebuffer: List[bytes] = []  # 预缓存缓冲区

        # 打断检测增强 - 需要连续帧检测
        self._interrupt_speech_frames = 0  # 连续检测到语音的帧数
        self._interrupt_threshold_frames = 3  # 需要连续3帧(60ms)才确认打断

        # 计算帧大小
        self.frame_size = int(self.sample_rate * self.frame_duration_ms / 1000) * 2  # 16-bit
        self.silence_threshold_frames = self.silence_threshold_ms // self.frame_duration_ms
        self.padding_frames = self.padding_duration_ms // self.frame_duration_ms

        # 时间追踪
        self._speech_start_time: Optional[float] = None
        self._silence_start_time: Optional[float] = None
        self._last_silence_duration: float = 0.0

        # 初始化 VAD - Silero 优先，简单VAD作为备选
        self._silero_vad = SileroVADWrapper()
        self._simple_vad = SimpleVAD(threshold=700)  # 备选

        if self._silero_vad.available:
            logger.info(f"声学VAD 使用 Silero VAD (静音超时: {self.silence_threshold_ms}ms, 预缓存: {self.prebuffer_duration_ms}ms)")
        else:
            logger.info(f"声学VAD 使用简单音量 VAD (阈值: {self.silence_threshold_ms}ms)")

    def process_frame(self, audio_frame: bytes) -> Optional[AudioSegment]:
        """
        处理音频帧

        v3.3 逻辑：
        - 持续维护预缓存（200ms回溯）
        - 检测到人声开始 → 回溯预缓存内容，一起送给ASR
        - 检测到静音 → 不立刻断，交给语义VAD判断
        - 只有语义VAD确认后，才会调用 finalize_segment 获取完整语音段

        Returns:
            检测到的完整语音段，或None
        """
        # 维护预缓存（始终保留最新的N帧）
        self._prebuffer.append(audio_frame)
        if len(self._prebuffer) > self.prebuffer_frames:
            self._prebuffer.pop(0)

        is_speech = self._detect_speech(audio_frame)

        if is_speech:
            self._silence_frames = 0
            self._speech_frames.append(audio_frame)
            self._silence_start_time = None
            self._last_silence_duration = 0.0

            if not self._is_speech:
                # 语音开始时，回溯预缓存内容（避免丢失语音开头）
                self._is_speech = True
                self._speech_start_time = time.time()

                # 将预缓存的帧添加到语音帧列表开头
                if self._prebuffer:
                    self._speech_frames = list(self._prebuffer) + self._speech_frames
                    logger.debug(f"声学VAD: 检测到语音开始，回溯{len(self._prebuffer)}帧({len(self._prebuffer) * self.frame_duration_ms}ms)")

                # 触发语音开始回调
                self._notify_speech_start()

        else:
            if self._is_speech:
                self._silence_frames += 1
                self._speech_frames.append(audio_frame)  # 保留静音部分作为padding

                # 记录静音开始时间
                if self._silence_start_time is None:
                    self._silence_start_time = time.time()

                # 计算当前静音时长
                self._last_silence_duration = (time.time() - self._silence_start_time) * 1000

                # 检查是否达到静音阈值
                if self._silence_frames >= self.silence_threshold_frames:
                    # 触发静音回调（不结束语音段，交给语义VAD判断）
                    self._notify_silence_detected(self._last_silence_duration)

        return None

    def _detect_speech(self, audio_frame: bytes) -> bool:
        """检测单帧是否包含语音"""
        # 优先使用 Silero VAD
        if self._silero_vad.available:
            try:
                return self._silero_vad.is_speech(audio_frame, self.sample_rate)
            except Exception as e:
                logger.debug(f"Silero VAD 检测异常: {e}")

        # 回退到简单 VAD
        return self._simple_vad.is_speech(audio_frame)

    def finalize_segment(self) -> Optional[AudioSegment]:
        """
        结束当前语音段（由语义VAD调用）

        当语义VAD确认语义完整后，调用此方法获取完整语音段

        Returns:
            完整的语音段
        """
        if not self._speech_frames:
            return None

        audio_data = b''.join(self._speech_frames)
        duration_ms = len(audio_data) / self.sample_rate / 2 * 1000

        logger.debug(f"声学VAD: 语音段结束, 长度={len(audio_data)} bytes, 时长={duration_ms:.0f}ms")

        segment = AudioSegment(
            data=audio_data,
            sample_rate=self.sample_rate,
            is_speech=True,
            duration_ms=duration_ms
        )

        # 重置状态
        self._is_speech = False
        self._speech_frames.clear()
        self._silence_frames = 0
        self._speech_start_time = None
        self._silence_start_time = None
        self._last_silence_duration = 0.0

        return segment

    def check_interrupt(self, audio_frame: bytes) -> bool:
        """
        检查是否是打断信号

        v3.2 更新：需要连续检测到多帧语音才确认打断，减少误触发

        Returns:
            是否检测到打断
        """
        is_speech = self._detect_speech(audio_frame)

        if is_speech:
            self._interrupt_speech_frames += 1
            # 需要连续检测到足够的帧数才确认打断
            if self._interrupt_speech_frames >= self._interrupt_threshold_frames:
                logger.info(f"声学VAD: 确认打断信号 (连续{self._interrupt_speech_frames}帧)")
                self._interrupt_speech_frames = 0  # 重置计数
                self._notify_interrupt()
                return True
        else:
            # 非语音帧，重置计数
            self._interrupt_speech_frames = 0

        return False

    def get_silence_duration(self) -> float:
        """
        获取当前静音时长（毫秒）

        Returns:
            静音时长（ms）
        """
        return self._last_silence_duration

    def reset(self):
        """重置状态"""
        self._is_speech = False
        self._speech_frames.clear()
        self._silence_frames = 0
        self._speech_start_time = None
        self._silence_start_time = None
        self._last_silence_duration = 0.0
        self._interrupt_speech_frames = 0
        self._prebuffer.clear()  # 清空预缓存

        # 重置 Silero VAD 状态
        if self._silero_vad.available:
            self._silero_vad.reset_states()

    def add_speech_callback(self, callback: Callable):
        """添加语音开始回调"""
        self._speech_callbacks.append(callback)

    def add_silence_callback(self, callback: Callable):
        """添加静音检测回调"""
        self._silence_callbacks.append(callback)

    def add_interrupt_callback(self, callback: Callable):
        """添加打断回调"""
        self._interrupt_callbacks.append(callback)

    def _notify_speech_start(self):
        """通知语音开始"""
        for callback in self._speech_callbacks:
            try:
                callback("speech_start")
            except Exception as e:
                logger.error(f"语音开始回调错误: {e}")

    def _notify_silence_detected(self, duration_ms: float):
        """通知检测到静音"""
        for callback in self._silence_callbacks:
            try:
                callback("silence_detected", duration_ms)
            except Exception as e:
                logger.error(f"静音回调错误: {e}")

    def _notify_interrupt(self):
        """通知打断"""
        for callback in self._interrupt_callbacks:
            try:
                callback("interrupt")
            except Exception as e:
                logger.error(f"打断回调错误: {e}")

    @property
    def is_speech_active(self) -> bool:
        """当前是否有语音活动"""
        return self._is_speech

    @property
    def current_audio_buffer(self) -> bytes:
        """获取当前音频缓冲区"""
        return b''.join(self._speech_frames)


class StreamingVAD:
    """
    流式VAD处理器 v3.0
    支持异步处理音频流
    """

    def __init__(self):
        self.acoustic_vad = AcousticVAD()
        self._buffer = bytearray()
        self._frame_size = self.acoustic_vad.frame_size

    async def process_chunk(self, audio_chunk: bytes) -> dict:
        """
        处理音频块

        Returns:
            处理结果字典，包含：
            - event: "speech_start" / "silence_detected" / "none"
            - silence_duration: 静音时长（ms）
            - is_interrupt: 是否是打断
        """
        self._buffer.extend(audio_chunk)
        result = {
            "event": "none",
            "silence_duration": 0.0,
            "is_interrupt": False
        }

        while len(self._buffer) >= self._frame_size:
            frame = bytes(self._buffer[:self._frame_size])
            self._buffer = self._buffer[self._frame_size:]

            # 处理帧
            self.acoustic_vad.process_frame(frame)

            # 检查状态
            if self.acoustic_vad._is_speech and self.acoustic_vad._last_silence_duration > 0:
                result["event"] = "silence_detected"
                result["silence_duration"] = self.acoustic_vad.get_silence_duration()
            elif self.acoustic_vad._is_speech:
                result["event"] = "speech_active"

        return result

    def check_interrupt(self, audio_chunk: bytes) -> bool:
        """
        检查是否是打断

        Returns:
            是否检测到打断
        """
        # 处理缓冲区中的帧
        self._buffer.extend(audio_chunk)

        while len(self._buffer) >= self._frame_size:
            frame = bytes(self._buffer[:self._frame_size])
            self._buffer = self._buffer[self._frame_size:]

            if self.acoustic_vad.check_interrupt(frame):
                return True

        return False

    def finalize_segment(self) -> Optional[AudioSegment]:
        """结束当前语音段"""
        return self.acoustic_vad.finalize_segment()

    def get_silence_duration(self) -> float:
        """获取当前静音时长"""
        return self.acoustic_vad.get_silence_duration()

    def reset(self):
        """重置"""
        self._buffer.clear()
        self.acoustic_vad.reset()

    @property
    def is_speech_active(self) -> bool:
        """当前是否有语音活动"""
        return self.acoustic_vad.is_speech_active