"""
全双工语音对话系统 v3.3 - 核心系统

架构流程：
1. 声学VAD检测语音起止与打断
2. 检测到人声开始 → 打开音频流送给ASR
3. 检测到静音 → 不立刻断，交给语义VAD判断
4. Qwen ASR 17B 流式输出文本
5. Qwen3-omni-flash 语义VAD 边接收文本边判断
6. Qwen3-omni-flash 情绪识别 与语义VAD、ASR并行（输入音频，输出情绪标签）
7. 语义完整 → 文本+情绪 → 后端LLM
8. 打断逻辑：语义VAD判断有效人声 → 立即停止TTS → 继续接收完整句子 → LLM
9. 支持多轮对话上下文
10. LLM流式输出 + Qwen3-tts 实时转换

v3.3 更新：
- TTS模型替换为 Qwen3-tts-flash
- 情绪识别使用 Qwen3-omni-flash 音频输入能力
- 情绪识别与ASR、语义VAD并行运行
"""
import asyncio
from typing import Optional, Callable, List
from .core.logger import logger

from .core import (
    DialogState,
    SemanticState,
    EmotionType,
    AudioSegment,
    LLMInput,
    DialogResult,
    Message,
    DialogStateMachine,
    SemanticStateMachine,
    get_config,
)
from .core.latency import latency_tracker
from .modules import (
    AcousticVAD,
    StreamingVAD,
    QwenASRProcessor,
    SemanticVADProcessor,
    StreamingSemanticVAD,
    EmotionRecognizer,
    ParallelEmotionRecognizer,
    LLMTaskPlanner,
    ToolEngine,
    TTSEngine,
    StreamingTTS,
    StreamingTTSProcessor,
)
from .modules.semantic_vad import VoiceValidity


class VoiceDialogSystem:
    """
    全双工语音对话系统 v3.2

    核心变化：
    1. ASR独立模块（Qwen ASR 17B）流式输出
    2. 语义VAD流式判断（边接收边判断）
    3. 情绪识别与语义VAD并行
    4. 声学静音交给语义VAD决策
    5. 打断逻辑：语义VAD判断有效人声 → 立即停止TTS → 继续接收完整句子
    """

    # 超时配置（毫秒）
    MAX_SILENCE_WAIT_MS = 2000  # 最大静音等待时间
    MIN_SPEECH_DURATION_MS = 300  # 最小语音时长
    INTERRUPT_CONFIRM_TIMEOUT_MS = 1500  # 打断确认超时时间
    SILENCE_THRESHOLD_MS = 500  # 静音检测阈值

    def __init__(self):
        self.config = get_config()

        # 状态机
        self.dialog_state = DialogStateMachine()
        self.semantic_state = SemanticStateMachine()

        # ========== v3.0 新架构模块 ==========
        # 声学VAD（阈值400ms）
        self.acoustic_vad = StreamingVAD()

        # 流式ASR（Qwen ASR 17B）
        self.asr_processor = QwenASRProcessor()

        # 流式语义VAD
        self.semantic_vad = StreamingSemanticVAD()

        # 并行情绪识别
        self.emotion_recognizer = ParallelEmotionRecognizer()

        # LLM和工具
        self.llm_planner = LLMTaskPlanner()
        self.tool_engine = ToolEngine()

        # TTS
        self.tts_engine = TTSEngine()
        self.streaming_tts = StreamingTTS()

        # 回调
        self._on_result_callbacks: List[Callable] = []
        self._on_state_change_callbacks: List[Callable] = []
        self._on_partial_asr_callbacks: List[Callable] = []
        self._on_tool_executing_callbacks: List[Callable] = []
        self._on_llm_chunk_callbacks: List[Callable] = []  # LLM流式输出回调
        self._on_audio_chunk_callbacks: List[Callable] = []  # TTS音频块回调
        self._on_clear_audio_callbacks: List[Callable] = []  # 清空音频回调

        # ========== 流式处理状态 ==========
        self._is_streaming = False
        self._stream_task: Optional[asyncio.Task] = None
        self._asr_text_buffer = ""
        self._streaming_start_time: Optional[float] = None
        self._last_speech_time: Optional[float] = None
        self._silence_start_time: Optional[float] = None

        # ========== v3.2 打断控制 ==========
        self._is_interrupted = False
        self._current_tts_task: Optional[asyncio.Task] = None
        self._interrupt_confirm_mode = False  # 是否在打断确认模式
        self._interrupt_start_time: Optional[float] = None
        self._tts_stopped_for_interrupt = False  # TTS是否因打断而停止
        self._first_asr_received = False  # 是否已收到首个ASR结果
        self._streaming_tts: Optional[StreamingTTSProcessor] = None  # 流式TTS处理器
        self._llm_task: Optional[asyncio.Task] = None  # LLM流式输出任务
        self._should_stop_llm = False  # 是否应该停止LLM输出
        self._audio_sequence = 0  # 音频序列号，确保播放顺序
        self._pending_audio_queue: List[tuple] = []  # 待发送的音频队列 (序列号, 音频数据)

        # ========== v3.5 并发模型改进 ==========
        self._audio_queue: asyncio.Queue = asyncio.Queue()  # 音频输入队列
        self._audio_processor_task: Optional[asyncio.Task] = None  # 音频处理后台任务
        self._is_processing_audio = False  # 是否正在处理音频

        # 添加状态监听
        self.dialog_state.add_listener(self._on_state_change)

        # 注册时延追踪回调
        self._on_latency_update_callbacks: List[Callable] = []
        latency_tracker.on_update(self._on_latency_update)

        logger.info("语音对话系统v3.5初始化完成 - 流式架构 + 语义打断 + 并发音频处理")

    async def process_audio(self, audio_chunk: bytes) -> Optional[DialogResult]:
        """
        处理音频块 - v3.5 并发架构

        核心改进：
        - 音频接收与处理解耦
        - 使用队列实现非阻塞接收
        - LLM 推理作为后台任务运行
        - 支持在 THINKING 状态下实时打断

        流程：
        1. 将音频放入队列（非阻塞）
        2. 后台任务持续处理队列中的音频
        3. 支持 SPEAKING/THINKING 状态下的实时打断
        """
        # 启动后台音频处理任务（如果还没启动）
        if self._audio_processor_task is None or self._audio_processor_task.done():
            self._audio_processor_task = asyncio.create_task(self._audio_processing_loop())
            logger.debug("[并发] 启动音频处理后台任务")

        # 将音频放入队列（非阻塞）
        try:
            self._audio_queue.put_nowait(audio_chunk)
        except asyncio.QueueFull:
            logger.warning("[并发] 音频队列已满，丢弃旧音频")
            # 丢弃旧音频，放入新音频
            try:
                self._audio_queue.get_nowait()
                self._audio_queue.put_nowait(audio_chunk)
            except:
                pass

        return None  # 不再阻塞返回结果，结果通过回调发送

    async def _audio_processing_loop(self):
        """
        音频处理后台循环 - v3.5

        持续从队列中取出音频并处理，与前端接收解耦
        """
        logger.info("[并发] 音频处理循环启动")

        while True:
            try:
                # 从队列获取音频（带超时，避免永久阻塞）
                audio_chunk = await asyncio.wait_for(
                    self._audio_queue.get(),
                    timeout=1.0
                )

                # 处理音频
                await self._process_audio_internal(audio_chunk)

            except asyncio.TimeoutError:
                # 不再自动退出，持续监听
                # 音频处理循环应该持续运行，确保随时能接收音频
                continue
            except asyncio.CancelledError:
                logger.info("[并发] 音频处理循环被取消")
                break
            except Exception as e:
                logger.error(f"[并发] 音频处理错误: {e}")
                import traceback
                traceback.print_exc()

    async def _process_audio_internal(self, audio_chunk: bytes) -> Optional[DialogResult]:
        """
        内部音频处理 - 原来的 process_audio 逻辑

        v3.5: 支持在 THINKING 状态下实时打断
        """
        import time
        current_time = time.time() * 1000

        # ========== 1. 打断确认模式下的处理 ==========
        if self._interrupt_confirm_mode and self._is_streaming:
            # 继续接收音频进行ASR识别
            await self._process_audio_parallel(audio_chunk)

            # 检查是否确认打断（有效人声）
            interrupt_result = self._check_interrupt_voice_validity()

            if interrupt_result == "valid":
                # 确认是有效人声，立即停止TTS播报
                logger.info(f"[打断] 确认有效人声，停止TTS播报，文本: '{self._asr_text_buffer}'")
                await self._stop_tts_for_interrupt()

            elif interrupt_result == "complete":
                # 语义完整，结束打断确认，进入LLM处理
                logger.info(f"[打断] 语义完整，进入LLM处理: '{self._asr_text_buffer}'")
                return await self._finalize_interrupt_to_llm()

            elif interrupt_result in ["filler", "noise"]:
                # 非有效人声（咳嗽、嗯等），取消打断，继续播报
                logger.info(f"[打断] 非有效人声，取消打断，继续播报")
                self._cancel_interrupt_confirmation()

            elif interrupt_result == "timeout":
                # 超时，检查是否有有效内容
                if self._asr_text_buffer.strip() and self._tts_stopped_for_interrupt:
                    # 有内容且TTS已停止，进入LLM处理
                    logger.info(f"[打断] 超时但有内容，进入LLM: '{self._asr_text_buffer}'")
                    return await self._finalize_interrupt_to_llm()
                else:
                    # 无有效内容，取消打断，继续播报
                    logger.info("[打断] 超时无有效内容，取消打断，继续播报")
                    self._cancel_interrupt_confirmation()

            return None

        # ========== 2. SPEAKING/THINKING状态下检测打断 ==========
        # v3.4: 扩展到 THINKING 状态，允许在 LLM 推理期间打断
        if self.dialog_state.state in [DialogState.SPEAKING, DialogState.THINKING]:
            if self.acoustic_vad.check_interrupt(audio_chunk):
                # 检测到人声，启动打断确认模式
                return await self._start_interrupt_confirmation()

        # ========== 3. 声学VAD处理 ==========
        vad_result = await self.acoustic_vad.process_chunk(audio_chunk)

        # 4. 检测到人声开始
        if vad_result["event"] == "speech_active":
            if not self._is_streaming:
                # ========== 新问题开始，清空旧音频流 ==========
                await self._clear_audio_stream()

                # 开始新的句子追踪
                latency_tracker.start_sentence()
                latency_tracker.mark_end("vad_detect", {"event": "speech_start"})

                # 新的语音段开始
                is_interrupt_mode = self._is_interrupted
                await self._start_streaming(interrupt_mode=is_interrupt_mode)
                self._is_streaming = True
                self._streaming_start_time = current_time
                self._silence_start_time = None
                self._is_interrupted = False
                self._first_asr_received = False
                await self.dialog_state.transition_to(DialogState.LISTENING, "检测到语音")
                logger.info(f"语音段开始 (时间: {current_time:.0f}ms)")

            self._last_speech_time = current_time
            self._silence_start_time = None

        # 5. 检测到静音，检查是否应该结束
        elif vad_result["event"] == "silence_detected" and self._is_streaming:
            silence_duration = vad_result["silence_duration"]

            if self._silence_start_time is None:
                self._silence_start_time = current_time

            silence_elapsed = current_time - self._silence_start_time

            # 判断是否应该结束语音段
            should_finalize = False
            finalize_reason = ""

            # 条件1: 语义VAD判断完整
            if self.semantic_vad.processor.is_complete():
                should_finalize = True
                finalize_reason = "语义完整"

            # 条件2: 静音时间超过阈值且有ASR文本
            elif silence_elapsed >= self.MAX_SILENCE_WAIT_MS and self._asr_text_buffer:
                should_finalize = True
                finalize_reason = f"静音超时({silence_elapsed:.0f}ms)"

            # 条件3: 有足够文本且静音超过500ms（与静音超时一致）
            elif silence_elapsed >= self.SILENCE_THRESHOLD_MS and len(self._asr_text_buffer) >= 5:
                should_finalize = True
                finalize_reason = "静音+文本充足"

            if should_finalize:
                logger.info(f"结束语音段: {finalize_reason}, 文本: '{self._asr_text_buffer}'")
                return await self._finalize_streaming()

        # 6. 流式处理音频（发送到ASR和情绪识别）
        if self._is_streaming:
            await self._process_audio_parallel(audio_chunk)

        return None

    async def _start_interrupt_confirmation(self) -> Optional[DialogResult]:
        """
        启动打断确认模式

        v3.5: THINKING 和 SPEAKING 状态都等待语义VAD确认后才停止TTS
        - 声学VAD只是触发确认模式，不直接停止
        - 语义VAD确认有效人声后才真正停止TTS/LLM
        """
        import time

        current_state = self.dialog_state.state
        logger.info(f"[打断] 声学VAD检测到人声，启动语义VAD判断... (当前状态: {current_state.value})")

        # THINKING和SPEAKING状态都等待语义VAD确认
        # 不在这里直接停止，等语义VAD判断后再决定

        self._interrupt_confirm_mode = True
        self._interrupt_start_time = time.time() * 1000
        self._tts_stopped_for_interrupt = False
        self._asr_text_buffer = ""

        # 启动流式处理（打断模式）
        await self._start_streaming(interrupt_mode=True)
        self._is_streaming = True

        return None

    def _check_interrupt_voice_validity(self) -> str:
        """
        检查打断确认结果

        v3.3: 优化打断逻辑
        - 有效人声：停止TTS，继续接收用户输入
        - 非有效人声（咳嗽、嗯等）：取消打断，继续播报
        - 待判断：继续等待

        Returns:
            "valid" - 确认是有效人声（应停止TTS）
            "filler" - 语气助词/非有效人声（应取消打断，继续播报）
            "complete" - 语义完整（应进入LLM处理）
            "pending" - 仍在判断中
            "timeout" - 超时
        """
        import time

        text = self._asr_text_buffer.strip()

        # 超时检查
        elapsed = time.time() * 1000 - self._interrupt_start_time
        if elapsed > self.INTERRUPT_CONFIRM_TIMEOUT_MS:
            return "timeout"

        # 文本为空，检查是否超时
        if not text:
            # 如果已经等待了足够长时间且没有文本，可能是噪声（咳嗽等）
            if elapsed > 800:  # 800ms没有识别出文本，很可能是噪声
                logger.info(f"[打断] 长时间无有效文本，判断为噪声，取消打断")
                return "noise"
            return "pending"

        # 使用语义VAD判断人声有效性
        voice_validity = self.semantic_vad.processor.check_voice_validity(text)

        if voice_validity == VoiceValidity.VALID:
            # 是有效人声，检查是否语义完整
            if self.semantic_vad.processor.is_complete():
                return "complete"
            return "valid"

        elif voice_validity == VoiceValidity.FILLER:
            # 只是语气助词，应该取消打断，继续播报
            logger.info(f"[打断] 检测到非有效人声: '{text}'，取消打断，继续播报")
            return "filler"

        elif voice_validity == VoiceValidity.NOISE:
            # 噪声，取消打断
            logger.info(f"[打断] 检测到噪声，取消打断")
            return "noise"

        # 检查文本内容是否像噪声/非有效人声
        # 如果文本很短且全是重复字符或语气词，可能是噪声
        if len(text) <= 2 and elapsed > 500:
            # 检查是否是重复字符或常见语气词
            noise_patterns = ["嗯", "啊", "呃", "额", "咳", "哼", "哈"]
            if all(c in noise_patterns for c in text):
                logger.info(f"[打断] 短文本判断为噪声: '{text}'，取消打断")
                return "noise"

        return "pending"

    async def _stop_tts_for_interrupt(self):
        """
        停止TTS播报（语义VAD确认有效人声后调用）

        v3.5: 统一处理 THINKING 和 SPEAKING 状态的打断
        - 只有语义VAD确认有效人声后才调用此方法
        - 停止TTS、LLM、清空音频队列
        """
        current_state = self.dialog_state.state
        logger.info(f"[打断] 语义VAD确认有效人声，停止播报 (状态: {current_state.value})")

        if self._tts_stopped_for_interrupt:
            return

        self._tts_stopped_for_interrupt = True
        self._should_stop_llm = True  # 标记停止LLM输出

        # 停止流式TTS处理器
        if self._streaming_tts:
            self._streaming_tts.stop()
            self._streaming_tts = None
            logger.info("[打断] 流式TTS处理器已停止")

        # 停止TTS播放
        self.streaming_tts.stop()

        # 取消当前TTS任务
        if self._current_tts_task and not self._current_tts_task.done():
            self._current_tts_task.cancel()
            self._current_tts_task = None

        # 取消LLM流式输出任务
        if self._llm_task and not self._llm_task.done():
            self._llm_task.cancel()
            self._llm_task = None
            logger.info("[打断] LLM流式输出任务已取消")

        # 通知前端清空音频队列
        await self._notify_clear_audio()

        logger.info("[打断] TTS/LLM已停止，音频队列已清空，继续接收用户输入...")

    async def _finalize_interrupt_to_llm(self) -> DialogResult:
        """
        打断确认完成，将用户输入交给LLM处理

        v3.5: LLM 处理作为后台任务运行
        """
        self._interrupt_confirm_mode = False
        self._is_streaming = False
        self._tts_stopped_for_interrupt = False  # 重置TTS停止标志

        # ========== 清空旧音频流，准备新问题 ==========
        await self._clear_audio_stream()

        # 重置语义VAD的打断模式
        self.semantic_vad.processor.set_interrupt_mode(False)

        # 获取最终ASR结果
        asr_result = await self.asr_processor.stop_stream()
        recognized_text = asr_result.text

        logger.info(f"[打断] 最终识别文本: '{recognized_text}'")

        # 检查有效性
        if not recognized_text or not recognized_text.strip():
            logger.warning("[打断] 空输入，忽略")
            await self.dialog_state.force_state(DialogState.IDLE, "空输入")
            return DialogResult(
                text="",
                semantic_state=SemanticState.REJECTED,
                dialog_state=DialogState.IDLE
            )

        # 获取语义VAD结果
        semantic_result = await self.semantic_vad.stop()
        semantic_state = semantic_result.state
        semantic_confidence = semantic_result.confidence

        # 获取情绪识别结果
        emotion_result = await self.emotion_recognizer.finalize_sentence(recognized_text)

        await self.dialog_state.force_state(DialogState.PROCESSING, "打断后LLM处理")

        # ========== v3.5: LLM 作为后台任务运行 ==========
        logger.info("[并发] 创建 LLM 后台任务（打断后）")
        self._llm_task = asyncio.create_task(
            self._process_with_llm(
                recognized_text,
                asr_result.confidence,
                semantic_state,
                semantic_confidence,
                emotion_result.emotion,
                emotion_result.confidence,
                emotion_result.intensity
            )
        )

        return None  # 不阻塞，结果通过回调发送

    def _cancel_interrupt_confirmation(self):
        """
        取消打断确认模式，继续播报
        """
        self._interrupt_confirm_mode = False
        self._is_streaming = False
        self._tts_stopped_for_interrupt = False
        self._asr_text_buffer = ""  # 清空ASR文本缓冲区
        self._first_asr_received = False  # 重置ASR首字标记
        self._should_stop_llm = False  # 重置LLM停止标志

        # 停止流式处理
        try:
            asyncio.create_task(self.asr_processor.stop_stream())
        except:
            pass

        self.semantic_vad.reset()
        self.emotion_recognizer.reset()
        self.acoustic_vad.reset()

        logger.debug("[打断] 确认取消，继续播报")

    async def _start_streaming(self, interrupt_mode: bool = False):
        """启动流式处理"""
        logger.info(f"启动流式处理 (打断模式: {interrupt_mode})")

        try:
            asr_started = await self.asr_processor.start_stream(self._on_asr_result)
            if not asr_started:
                logger.warning("ASR流启动失败，将使用模拟模式")

            await self.semantic_vad.start(interrupt_mode=interrupt_mode)
            await self.emotion_recognizer.start()

            self._asr_text_buffer = ""
            self._streaming_start_time = None
            self._last_speech_time = None
            self._silence_start_time = None

            logger.info("流式处理已启动")

        except Exception as e:
            logger.error(f"启动流式处理失败: {e}")
            import traceback
            traceback.print_exc()

    async def _process_audio_parallel(self, audio_chunk: bytes):
        """并行处理音频"""
        await asyncio.gather(
            self.asr_processor.process_chunk(audio_chunk),
            self.emotion_recognizer.process_audio(audio_chunk)
        )

    async def _on_asr_result(self, text: str, is_final: bool):
        """ASR流式结果回调"""
        self._asr_text_buffer = text

        # 追踪ASR首字延迟
        if not self._first_asr_received and text:
            latency_tracker.mark_start("asr_first_text")
            latency_tracker.mark_end("asr_first_text", {"text": text[:10]})
            self._first_asr_received = True

        # 通知部分ASR结果
        await self._notify_partial_asr(text)

        # 更新时延追踪的文本
        latency_tracker.update_text(text)

        # 流式语义VAD判断
        if text:
            latency_tracker.mark_start("semantic_vad")
            semantic_result = await self.semantic_vad.process_text(text, is_final)
            latency_tracker.mark_end("semantic_vad", {
                "state": semantic_result.state.value,
                "confidence": semantic_result.confidence
            })
            logger.debug(f"语义VAD判断: {semantic_result.state.value} (置信度: {semantic_result.confidence:.2f})")

            # 更新语义状态
            self.semantic_state.update(semantic_result.state, semantic_result.confidence)

            if semantic_result.state == SemanticState.COMPLETE:
                logger.info(f"语义完整: '{text}'")

    async def _finalize_streaming(self) -> Optional[DialogResult]:
        """
        结束流式处理，进入融合阶段

        v3.5: LLM 处理作为后台任务运行，不阻塞音频接收
        """
        if not self._is_streaming:
            return None

        self._is_streaming = False
        self._is_interrupted = False

        # 重置语义VAD的打断模式
        self.semantic_vad.processor.set_interrupt_mode(False)

        try:
            await self.dialog_state.transition_to(DialogState.PROCESSING, "语音段结束")
        except Exception as e:
            logger.warning(f"状态转换失败: {e}")
            await self.dialog_state.force_state(DialogState.PROCESSING, "语音段结束")

        try:
            # 1. 获取最终ASR结果
            latency_tracker.mark_end("asr_streaming")
            asr_result = await self.asr_processor.stop_stream()
            recognized_text = asr_result.text

            logger.info(f"ASR最终结果: '{recognized_text}'")

            # 更新时延追踪的最终文本
            latency_tracker.update_text(recognized_text)

            # 2. 获取最终语义VAD结果
            semantic_result = await self.semantic_vad.stop()
            semantic_state = semantic_result.state
            semantic_confidence = semantic_result.confidence

            # 3. 获取情绪识别结果
            latency_tracker.mark_start("emotion")
            emotion_result = await self.emotion_recognizer.finalize_sentence(recognized_text)
            latency_tracker.mark_end("emotion", {"emotion": emotion_result.emotion.value})
            emotion = emotion_result.emotion
            emotion_confidence = emotion_result.confidence
            emotion_intensity = emotion_result.intensity

            logger.info(f"处理结果: 文本='{recognized_text}', 语义={semantic_state.value}, 情绪={emotion.value}")

            # 检查有效性
            if not recognized_text or not recognized_text.strip():
                logger.warning("空输入，忽略")
                await self.dialog_state.transition_to(DialogState.IDLE, "空输入")
                return DialogResult(
                    text="",
                    semantic_state=SemanticState.REJECTED,
                    dialog_state=DialogState.IDLE
                )

            if semantic_state == SemanticState.REJECTED:
                logger.warning("拒识输入")
                await self.dialog_state.transition_to(DialogState.IDLE, "拒识输入")
                return DialogResult(
                    text="",
                    semantic_state=SemanticState.REJECTED,
                    dialog_state=DialogState.IDLE
                )

            # ========== v3.5: LLM 作为后台任务运行 ==========
            logger.info("[并发] 创建 LLM 后台任务")
            self._llm_task = asyncio.create_task(
                self._process_with_llm(
                    recognized_text,
                    asr_result.confidence,
                    semantic_state,
                    semantic_confidence,
                    emotion,
                    emotion_confidence,
                    emotion_intensity
                )
            )

            # 不阻塞返回，结果通过回调发送
            return None

        except Exception as e:
            logger.error(f"流式处理失败: {e}")
            import traceback
            traceback.print_exc()
            await self.dialog_state.force_state(DialogState.IDLE, f"错误: {e}")
            return None

    async def _process_with_llm(
        self,
        text: str,
        text_confidence: float,
        semantic_state: SemanticState,
        semantic_confidence: float,
        emotion: EmotionType,
        emotion_confidence: float,
        emotion_intensity: float
    ) -> DialogResult:
        """融合阶段：文本+情绪 → LLM处理（流式显示 + 按句子TTS播报）"""

        # ========== 清空旧音频，准备新问题播报 ==========
        await self._notify_clear_audio()
        logger.info(f"[音频流] 新问题开始，已清空旧音频队列")

        # 重置停止标志
        self._should_stop_llm = False

        await self.dialog_state.transition_to(DialogState.THINKING, "开始LLM处理")

        llm_input = LLMInput(
            text=text,
            text_confidence=text_confidence,
            semantic_state=semantic_state,
            semantic_confidence=semantic_confidence,
            emotion=emotion,
            emotion_confidence=emotion_confidence,
            emotion_intensity=emotion_intensity,
        )

        logger.info(f"融合输入 LLM: 文本='{text}', 情绪={emotion.value}")

        # ========== v3.2.3: 流式显示 + 按句子TTS播报 ==========
        latency_tracker.mark_start("llm_process")

        # 重置音频序列号
        self._audio_sequence = 0
        self._pending_audio_queue = []

        # 定义音频块回调：按句子发送音频
        async def on_audio_chunk(audio_data: bytes):
            """TTS音频块回调 - 每个句子完成后发送"""
            if self._should_stop_llm:
                return
            await self._notify_audio_chunk(audio_data)

        # 创建流式TTS处理器并启动消费者任务
        self._streaming_tts = StreamingTTSProcessor(on_audio_chunk=on_audio_chunk)
        await self._streaming_tts.start()  # v3.4: 启动消费者任务

        # 收集完整响应文本
        response_text_parts = []

        # TTS任务跟踪
        tts_task = None

        # 定义LLM文本块回调：只做非阻塞的队列放入
        # v3.5: LLM回调完全非阻塞，立即返回，让TTS消费者独立处理
        def on_llm_chunk(chunk: str):
            if self._should_stop_llm:
                return
            response_text_parts.append(chunk)
            # 非阻塞放入TTS队列（使用put_nowait，不等待）
            if self._streaming_tts and not self._streaming_tts._should_stop:
                try:
                    self._streaming_tts.add_text_nowait(chunk)
                except asyncio.QueueFull:
                    logger.warning(f"[TTS] 队列已满，跳过文本块: {chunk[:20]}...")
            # 前端显示（创建任务，不阻塞）
            asyncio.create_task(self._notify_llm_chunk(chunk))

        # 定义工具检测回调
        def on_tool_detected(tool_name: str):
            if self._should_stop_llm:
                return
            logger.info(f"[工具检测] 发现工具调用: {tool_name}")
            asyncio.create_task(self._notify_tool_executing(tool_name, {}))

        # 使用流式方法调用LLM
        try:
            llm_response = await self.llm_planner.plan_stream(
                llm_input,
                on_chunk=on_llm_chunk,
                on_tool_detected=on_tool_detected
            )
        except asyncio.CancelledError:
            logger.info("[打断] LLM流式输出被取消")
            # 返回一个空结果
            await self.dialog_state.force_state(DialogState.IDLE, "被打断")
            return DialogResult(
                text=text,
                text_confidence=text_confidence,
                semantic_state=semantic_state,
                emotion=emotion,
                emotion_confidence=emotion_confidence,
                response="",
                response_audio=b"",
                is_interrupt=True,
                dialog_state=DialogState.IDLE
            )

        # 检查是否被打断
        if self._should_stop_llm:
            logger.info("[打断] LLM输出已停止")
            await self.dialog_state.force_state(DialogState.IDLE, "被打断")
            return DialogResult(
                text=text,
                text_confidence=text_confidence,
                semantic_state=semantic_state,
                emotion=emotion,
                emotion_confidence=emotion_confidence,
                response="".join(response_text_parts),
                response_audio=b"",
                is_interrupt=True,
                dialog_state=DialogState.IDLE
            )

        latency_tracker.mark_end("llm_process", {
            "response_preview": llm_response.text[:50] if llm_response.text else "",
            "has_tools": len(llm_response.tool_calls) > 0
        })

        # LLM完成后，刷新TTS缓冲区，确保所有文本都已转换和发送
        # TTS处理器内部有缓冲区，finalize会处理剩余文本
        if self._streaming_tts:
            logger.info(f"[TTS] LLM完成，刷新TTS缓冲区")
            full_audio = await self._streaming_tts.finalize()
            self._streaming_tts = None
            logger.info(f"[TTS] 处理完成，总音频: {len(full_audio)} bytes")
        else:
            full_audio = b""

        # 工具调用处理
        tool_results = []
        if llm_response.tool_calls:
            latency_tracker.mark_start("tool_execute")

            logger.info(f"执行工具调用: {[tc.name for tc in llm_response.tool_calls]}")
            tool_results = await self.tool_engine.execute_batch(llm_response.tool_calls)
            latency_tracker.mark_end("tool_execute", {"tools": [tc.name for tc in llm_response.tool_calls]})

            # 工具执行后的总结
            final_response = await self.llm_planner.summarize_tool_results(
                llm_response, tool_results
            )
            llm_response.final_response = final_response

            # 对工具总结进行TTS
            if final_response:
                await self.dialog_state.transition_to(DialogState.SPEAKING, "语音合成")

                async def on_tool_audio_chunk(audio_data: bytes):
                    await self._notify_audio_chunk(audio_data)

                tool_tts = StreamingTTSProcessor(on_audio_chunk=on_tool_audio_chunk)
                await tool_tts.start()  # v3.4: 启动消费者任务
                await tool_tts.add_text(final_response)
                tool_audio = await tool_tts.finalize()
                full_audio = full_audio + tool_audio if full_audio else tool_audio
        else:
            llm_response.final_response = llm_response.text

        # 更新对话历史
        self.llm_planner.add_to_history(Message(
            role="user",
            content=text,
            emotion=emotion
        ))
        self.llm_planner.add_to_history(Message(
            role="assistant",
            content=llm_response.final_response
        ))

        logger.info(f"对话完成 - 用户: '{text}' -> 助手: '{llm_response.final_response[:50]}...'")

        # 结束时延追踪
        latency_tracker.end_sentence()

        # 构建结果
        result = DialogResult(
            text=text,
            text_confidence=text_confidence,
            semantic_state=semantic_state,
            emotion=emotion,
            emotion_confidence=emotion_confidence,
            response=llm_response.final_response,
            response_audio=full_audio,
            is_interrupt=False,
            dialog_state=DialogState.SPEAKING,
            tool_calls=llm_response.tool_calls,
            tool_results=tool_results,
            llm_emotion=llm_response.llm_emotion,
        )

        await self._notify_result(result)
        await self.dialog_state.transition_to(DialogState.IDLE, "处理完成")

        return result

    async def process_text(self, text: str) -> DialogResult:
        """处理文本输入"""
        logger.info(f"处理文本输入: '{text}'")

        # 开始时延追踪
        latency_tracker.start_sentence()
        latency_tracker.mark_start("llm_process")

        if self.dialog_state.state == DialogState.IDLE:
            await self.dialog_state.transition_to(DialogState.LISTENING, "文本输入开始")

        await self.dialog_state.transition_to(DialogState.PROCESSING, "文本输入处理")

        latency_tracker.mark_start("emotion")
        emotion_result = await self.emotion_recognizer.finalize_sentence(text)
        latency_tracker.mark_end("emotion", {"emotion": emotion_result.emotion.value})

        latency_tracker.update_text(text)

        result = await self._process_with_llm(
            text,
            1.0,
            SemanticState.COMPLETE,
            1.0,
            emotion_result.emotion,
            emotion_result.confidence,
            emotion_result.intensity
        )

        # 结束时延追踪
        latency_tracker.end_sentence()

        return result

    async def interrupt(self):
        """手动触发打断"""
        await self._stop_tts_for_interrupt()
        await self.dialog_state.force_state(DialogState.IDLE, "手动打断")

    async def _on_state_change(self, old_state: DialogState, new_state: DialogState):
        """状态变化回调"""
        for callback in self._on_state_change_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(old_state, new_state)
                else:
                    callback(old_state, new_state)
            except Exception as e:
                logger.error(f"状态回调错误: {e}")

    async def _notify_result(self, result: DialogResult):
        """通知结果"""
        for callback in self._on_result_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(result)
                else:
                    callback(result)
            except Exception as e:
                logger.error(f"结果回调错误: {e}")

    async def _notify_partial_asr(self, text: str):
        """通知部分ASR结果"""
        for callback in self._on_partial_asr_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(text)
                else:
                    callback(text)
            except Exception as e:
                logger.error(f"ASR回调错误: {e}")

    async def _notify_tool_executing(self, tool_name: str, tool_args: dict):
        """通知工具正在执行"""
        for callback in self._on_tool_executing_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(tool_name, tool_args)
                else:
                    callback(tool_name, tool_args)
            except Exception as e:
                logger.error(f"工具执行回调错误: {e}")

    async def _notify_llm_chunk(self, chunk: str):
        """通知LLM流式输出块"""
        for callback in self._on_llm_chunk_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(chunk)
                else:
                    callback(chunk)
            except Exception as e:
                logger.error(f"LLM流式回调错误: {e}")

    async def _notify_audio_chunk(self, audio_data: bytes):
        """通知TTS音频块（实时播报）"""
        for callback in self._on_audio_chunk_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(audio_data)
                else:
                    callback(audio_data)
            except Exception as e:
                logger.error(f"音频块回调错误: {e}")

    def on_result(self, callback: Callable):
        """注册结果回调"""
        self._on_result_callbacks.append(callback)

    def on_state_change(self, callback: Callable):
        """注册状态变化回调"""
        self._on_state_change_callbacks.append(callback)

    def on_partial_asr(self, callback: Callable):
        """注册部分ASR回调"""
        self._on_partial_asr_callbacks.append(callback)

    def on_tool_executing(self, callback: Callable):
        """注册工具执行回调"""
        self._on_tool_executing_callbacks.append(callback)

    def on_llm_chunk(self, callback: Callable):
        """注册LLM流式输出回调"""
        self._on_llm_chunk_callbacks.append(callback)

    def on_audio_chunk(self, callback: Callable):
        """注册TTS音频块回调（用于实时播报）"""
        self._on_audio_chunk_callbacks.append(callback)

    def on_clear_audio(self, callback: Callable):
        """注册清空音频回调（新问题开始时调用）"""
        self._on_clear_audio_callbacks.append(callback)

    def on_latency_update(self, callback: Callable):
        """注册时延更新回调"""
        self._on_latency_update_callbacks.append(callback)

    def _on_latency_update(self, data):
        """时延数据更新回调"""
        for callback in self._on_latency_update_callbacks:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"时延更新回调错误: {e}")

    def reset(self):
        """重置系统"""
        self.dialog_state.reset()
        self.semantic_state.reset()
        self.acoustic_vad.reset()
        self.llm_planner.clear_history()
        self._is_interrupted = False
        self._is_streaming = False
        self._asr_text_buffer = ""
        self._streaming_start_time = None
        self._last_speech_time = None
        self._silence_start_time = None
        self._interrupt_confirm_mode = False
        self._tts_stopped_for_interrupt = False
        self._should_stop_llm = False  # 重置LLM停止标志

        # 取消LLM任务
        if self._llm_task and not self._llm_task.done():
            self._llm_task.cancel()
            self._llm_task = None

        # 取消音频处理任务
        if self._audio_processor_task and not self._audio_processor_task.done():
            self._audio_processor_task.cancel()
            self._audio_processor_task = None

        # 清空音频队列
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except:
                break

        self.semantic_vad.processor.set_interrupt_mode(False)

        logger.info("系统已重置（对话历史已清空）")

    async def _clear_audio_stream(self):
        """
        清空音频流 - 新问题开始时调用

        停止旧的TTS处理，通知前端清空音频队列
        """
        # 设置停止标志
        self._should_stop_llm = True

        # 停止旧的流式TTS处理器
        if self._streaming_tts:
            self._streaming_tts.stop()
            self._streaming_tts = None
            logger.info("[音频流] 停止旧的TTS处理器")

        # 停止流式TTS播放
        self.streaming_tts.stop()

        # 取消当前TTS任务
        if self._current_tts_task and not self._current_tts_task.done():
            self._current_tts_task.cancel()
            self._current_tts_task = None

        # 取消LLM流式输出任务
        if self._llm_task and not self._llm_task.done():
            self._llm_task.cancel()
            self._llm_task = None

        # 通知前端清空音频队列
        await self._notify_clear_audio()

        logger.info("[音频流] 已清空，准备处理新问题")

    async def _notify_clear_audio(self):
        """通知前端清空音频队列"""
        for callback in self._on_clear_audio_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logger.error(f"清空音频回调错误: {e}")

    def clear_context(self):
        """仅清空上下文，保持状态"""
        self.llm_planner.clear_history()
        logger.info("对话上下文已清空")

    @property
    def current_state(self) -> DialogState:
        """当前对话状态"""
        return self.dialog_state.state

    @property
    def is_busy(self) -> bool:
        """是否忙碌"""
        return self.dialog_state.is_busy()

    @property
    def conversation_history(self) -> List[Message]:
        """获取对话历史"""
        return self.llm_planner.history