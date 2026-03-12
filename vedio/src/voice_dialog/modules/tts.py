"""
全双工语音对话系统 v3.3 - TTS模块

支持：
- Qwen3 TTS（默认，v3.3更新）
- Edge TTS（备选）
- 流式TTS播放
- LLM流式输出实时转语音

v3.3 更新：
- 将TTS模型替换为 Qwen3-tts
- 使用 DashScope MultiModalConversation API
- 支持流式和非流式输出
"""
import asyncio
import io
import re
import base64
import os
from typing import Optional, Callable, AsyncIterator, List

from pydub import AudioSegment

from ..core.logger import logger

from ..core.types import TTSResult
from ..core.config import get_config

try:
    import dashscope
    from dashscope import MultiModalConversation
    HAS_DASHSCOPE = True
except ImportError:
    HAS_DASHSCOPE = False
    logger.warning("dashscope未安装，Qwen3 TTS将不可用")


def pcm_to_wav(pcm_data, sample_rate=24000, channels=1, sample_width=2):
    audio = AudioSegment(
        data=pcm_data,
        sample_width=sample_width,
        frame_rate=sample_rate,
        channels=channels
    )

    wav_buffer = io.BytesIO()
    audio.export(wav_buffer, format="wav")
    wav_data = wav_buffer.getvalue()

    return wav_data

def clean_text_for_tts(text: str) -> str:
    """
    清理文本中的Markdown格式符号和表情符号，使其适合TTS播报

    处理内容：
    - Markdown格式符号（粗体、斜体、删除线、链接等）
    - 表情符号
    - 多余的空白字符
    - 代码块标记（保留内容，去除标记）
    """
    if not text:
        return text

    original_text = text

    # 1. 处理代码块（先处理多行的）
    # 匹配 ```content``` 格式，提取内容
    def extract_code_block(m):
        content = m.group(1).strip() if m.group(1) else ""
        return content

    text = re.sub(r'```([\s\S]*?)```', extract_code_block, text)
    # 处理未闭合的代码块标记
    text = re.sub(r'```[\s\S]*$', '', text)  # 移除未闭合的代码块开始
    text = re.sub(r'```', '', text)  # 移除残留的代码块标记
    text = re.sub(r'`([^`]+?)`', r'\1', text)  # 行内代码

    # 2. 处理链接 [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    # 3. 处理图片 ![alt](url) -> alt
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', text)

    # 4. 处理Markdown格式符号
    # 粗体 **text** 和 __text__
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)

    # 斜体 *text* 和 _text_（注意避免匹配下划线变量名）
    text = re.sub(r'(?<![a-zA-Z0-9])\*([^*]+?)\*(?![*])', r'\1', text)
    text = re.sub(r'(?<![a-zA-Z0-9])_([^_]+?)_(?![a-zA-Z0-9_])', r'\1', text)

    # 删除线 ~~text~~ 和 --text--
    text = re.sub(r'~~(.+?)~~', r'\1', text)
    text = re.sub(r'--(.+?)--', r'\1', text)

    # 5. 处理标题 # ## ### 等
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)

    # 6. 处理引用 > text
    text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)

    # 7. 处理列表符号
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)  # 无序列表
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)  # 有序列表

    # 8. 处理分隔线 *** --- ___
    text = re.sub(r'^(\*{3,}|-{3,}|_{3,})\s*$', '', text, flags=re.MULTILINE)

    # 9. 处理HTML标签
    text = re.sub(r'<[^>]+>', '', text)

    # 10. 移除表情符号（使用Unicode范围）
    # 注意：范围必须精确，避免误删中文字符
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # 表情符号
        "\U0001F300-\U0001F5FF"  # 符号和象形文字
        "\U0001F680-\U0001F6FF"  # 交通和地图符号
        "\U0001F700-\U0001F77F"  # 炼金术符号
        "\U0001F780-\U0001F7FF"  # 几何图形扩展
        "\U0001F800-\U0001F8FF"  # 补充箭头-C
        "\U0001F900-\U0001F9FF"  # 补充符号和象形文字
        "\U0001FA00-\U0001FA6F"  # 国际象棋符号
        "\U0001FA70-\U0001FAFF"  # 符号和象形文字扩展-A
        "\U00002702-\U000027B0"  # 装饰符号
        "\U0001F200-\U0001F251"  # 包围字符补充（修复：原来是\U000024C2，会误删中文）
        "\U0001F004"             # 麻将牌
        "\U0001F0CF"             # 扑克牌
        "]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub('', text)

    # 11. 清理多余的空白字符
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n', text)
    text = text.strip()

    # 如果清理后为空，返回原文（防止过度清理）
    if not text and original_text:
        return original_text

    return text


class TTSEngine:
    """
    TTS引擎 v3.3
    支持多种TTS后端: Qwen3 TTS / Edge TTS
    """

    # Qwen3 TTS 支持的音色
    QWEN_VOICES = [
        "Cherry", "Ethan", "Luna", "Marcus", "Serena", "Atlas",
        "Aria", "Oliver", "Aurora", "Felix", "Nova", "Leo",
        "Stella", "Max", "Ivy", "Sam", "Zoe", "Luke", "Maya", "Jack"
    ]

    def __init__(self):
        self.config = get_config().tts
        self.provider = self.config.get("provider", "qwen3")
        self.voice = self.config.get("voice", "Cherry")
        self._edge_tts = None
        self._init_qwen_tts()

    def _init_qwen_tts(self):
        """初始化 Qwen3 TTS"""
        if not HAS_DASHSCOPE:
            logger.warning("dashscope未安装，Qwen3 TTS不可用，将使用Edge TTS")
            self.provider = "edge"
            return

        api_key = get_config().qwen_omni.get("api_key", "") or os.getenv("DASHSCOPE_API_KEY", "")
        if api_key:
            dashscope.api_key = api_key
            logger.info(f"Qwen3 TTS 初始化成功 (音色: {self.voice})")
        else:
            logger.warning("未配置 DASHSCOPE_API_KEY，将使用Edge TTS")
            self.provider = "edge"

    async def synthesize(self, text: str) -> TTSResult:
        """
        文本转语音
        """
        # 清理文本中的格式符号和表情
        cleaned_text = clean_text_for_tts(text)

        if not cleaned_text.strip():
            return TTSResult(audio_data=b"", duration_ms=0)

        try:
            if self.provider == "qwen3":
                return await self._synthesize_qwen3(cleaned_text)
            elif self.provider == "edge":
                return await self._synthesize_edge(cleaned_text)
            else:
                return await self._synthesize_qwen3(cleaned_text)

        except Exception as e:
            logger.error(f"TTS合成失败: {e}")
            # 尝试使用Edge TTS作为后备
            if self.provider == "qwen3":
                logger.info("尝试使用Edge TTS作为后备...")
                try:
                    return await self._synthesize_edge(cleaned_text)
                except Exception as e2:
                    logger.error(f"Edge TTS也失败: {e2}")
            return TTSResult(audio_data=b"", duration_ms=0)

    async def _synthesize_qwen3(self, text: str) -> TTSResult:
        """
        使用 Qwen3 TTS 进行语音合成

        使用 DashScope MultiModalConversation API
        模型: qwen3-tts-flash
        """
        if not HAS_DASHSCOPE:
            raise RuntimeError("dashscope未安装")

        try:
            # 限制文本长度（Qwen3 TTS最长512 Token）
            if len(text) > 500:
                # 分段处理长文本
                return await self._synthesize_qwen3_long(text)

            # 调用 Qwen3 TTS API
            response = MultiModalConversation.call(
                model="qwen3-tts-flash",
                text=text,
                voice=self.voice,
                stream=False
            )

            if response.status_code != 200:
                logger.error(f"Qwen3 TTS API错误: {response.code} - {response.message}")
                raise RuntimeError(f"Qwen3 TTS API错误: {response.message}")

            # 获取音频URL或Base64数据
            audio_data = b""
            if hasattr(response, 'output') and response.output:
                audio_info = getattr(response.output, 'audio', None)
                if audio_info:
                    # 如果有URL，下载音频
                    if hasattr(audio_info, 'url') and audio_info.url:
                        audio_data = await self._download_audio(audio_info.url)
                    # 如果有Base64数据（流式模式）
                    elif hasattr(audio_info, 'data') and audio_info.data:
                        audio_data = base64.b64decode(audio_info.data)

            if not audio_data:
                logger.warning("Qwen3 TTS 未返回音频数据")
                return TTSResult(audio_data=b"", duration_ms=0)

            # 估算时长 (WAV 24kHz 16-bit mono ≈ 48KB/s)
            duration_ms = len(audio_data) / 48000 * 1000

            logger.debug(f"Qwen3 TTS 合成成功: 文本长度={len(text)}, 音频大小={len(audio_data)}, 时长≈{duration_ms:.0f}ms")

            return TTSResult(
                audio_data=audio_data,
                format="wav",
                sample_rate=24000,
                duration_ms=duration_ms
            )

        except Exception as e:
            logger.error(f"Qwen3 TTS 合成失败: {e}")
            raise

    async def _synthesize_qwen3_long(self, text: str) -> TTSResult:
        """
        处理长文本 - 分段合成后合并
        """
        # 按句子分段
        segments = self._split_long_text(text)
        audio_parts = []

        for segment in segments:
            if not segment.strip():
                continue
            try:
                result = await self._synthesize_qwen3(segment)
                if result.audio_data:
                    audio_parts.append(result.audio_data)
                # 避免请求过快
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.warning(f"分段合成失败: {segment[:20]}... - {e}")

        if not audio_parts:
            return TTSResult(audio_data=b"", duration_ms=0)

        # 合并音频（简单拼接，实际可能需要重采样）
        all_audio = b"".join(audio_parts)
        duration_ms = len(all_audio) / 48000 * 1000

        return TTSResult(
            audio_data=all_audio,
            format="wav",
            sample_rate=24000,
            duration_ms=duration_ms
        )

    def _split_long_text(self, text: str, max_length: int = 400) -> List[str]:
        """
        将长文本按句子分割成小段
        """
        # 句子结束符
        endings = ['。', '！', '？', '；', '\n', '!', '?', ';', '.']
        segments = []
        current = ""

        for char in text:
            current += char
            if char in endings and len(current) >= 50:  # 至少50字符才分割
                segments.append(current.strip())
                current = ""

        if current.strip():
            segments.append(current.strip())

        # 合并过短的段落
        result = []
        temp = ""
        for seg in segments:
            if len(temp) + len(seg) <= max_length:
                temp += seg
            else:
                if temp:
                    result.append(temp)
                temp = seg
        if temp:
            result.append(temp)

        return result

    async def _download_audio(self, url: str) -> bytes:
        """
        从URL下载音频数据
        """
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.read()
                    else:
                        logger.error(f"下载音频失败: HTTP {response.status}")
                        return b""
        except Exception as e:
            logger.error(f"下载音频错误: {e}")
            return b""

    async def _synthesize_edge(self, text: str) -> TTSResult:
        """使用Edge TTS"""
        try:
            import edge_tts

            voice = self.config.get("voice", "zh-CN-XiaoxiaoNeural")
            rate = self.config.get("rate", "+0%")
            pitch = self.config.get("pitch", "+0Hz")

            communicate = edge_tts.Communicate(
                text=text,
                voice=voice,
                rate=rate,
                pitch=pitch
            )

            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]

            # 估算时长 (MP3约128kbps)
            duration_ms = len(audio_data) / 128 * 8

            return TTSResult(
                audio_data=audio_data,
                format="mp3",
                sample_rate=24000,
                duration_ms=duration_ms
            )

        except ImportError:
            logger.error("edge-tts未安装")
            return await self._mock_synthesize(text)

    async def _mock_synthesize(self, text: str) -> TTSResult:
        """模拟TTS（用于测试）"""
        await asyncio.sleep(0.1)

        # 生成静音音频作为占位
        # 实际使用时会被真实TTS替代
        duration_ms = len(text) * 150  # 估算时长

        return TTSResult(
            audio_data=b"\x00" * 1000,  # 模拟音频数据
            format="mp3",
            sample_rate=24000,
            duration_ms=duration_ms
        )


class StreamingTTS:
    """
    流式TTS
    支持边合成边播放
    """

    def __init__(self):
        self.engine = TTSEngine()
        self._is_playing = False
        self._should_stop = False

    async def stream_synthesize(self, text: str):
        """
        流式合成并返回音频块
        """
        # 清理文本中的格式符号和表情
        cleaned_text = clean_text_for_tts(text)

        if not cleaned_text.strip():
            return

        try:
            import edge_tts

            config = get_config().tts
            voice = config.get("voice", "zh-CN-XiaoxiaoNeural")

            communicate = edge_tts.Communicate(text=cleaned_text, voice=voice)

            self._is_playing = True
            self._should_stop = False

            async for chunk in communicate.stream():
                if self._should_stop:
                    break

                if chunk["type"] == "audio":
                    yield chunk["data"]

            self._is_playing = False

        except Exception as e:
            logger.error(f"流式TTS失败: {e}")
            self._is_playing = False

    def stop(self):
        """停止播放"""
        self._should_stop = True

    @property
    def is_playing(self) -> bool:
        return self._is_playing


class StreamingTTSProcessor:
    """
    流式TTS处理器 v3.4

    核心功能：
    - 接收LLM流式输出的文本块
    - 按句子分段转换
    - 每个句子完成后立即发送音频（流畅播报）
    - 使用队列实现生产者-消费者模型，解耦文本添加和TTS合成
    - 支持 Qwen3 TTS 和 Edge TTS

    分段策略：
    - 遇到句子结束符（。！？等）时，立即转换这个句子并发送
    - 确保每个音频块是完整句子，播放更自然

    v3.4 更新：
    - 使用生产者-消费者模型替代锁机制
    - 文本添加非阻塞，TTS合成在后台独立任务中执行
    - 音频合成完成后立即发送，不再等待锁
    """

    # 句子结束符
    SENTENCE_ENDINGS = ['。', '！', '？', '；', '\n']

    # 配置参数
    MIN_CHUNK_SIZE = 5       # 最小分段长度（字符）
    MAX_CHUNK_SIZE = 80      # 最大分段长度（字符）

    # 特殊标记
    _FLUSH_SIGNAL = object()  # 刷新信号
    _STOP_SIGNAL = object()   # 停止信号

    def __init__(self, on_audio_chunk: Optional[Callable[[bytes], None]] = None):
        """
        初始化流式TTS处理器

        Args:
            on_audio_chunk: 音频块回调函数
        """
        self.config = get_config().tts
        self.on_audio_chunk = on_audio_chunk
        self.provider = self.config.get("provider", "qwen3")
        self.voice = self.config.get("voice", "Cherry")

        # 文本缓冲区（消费者线程访问）
        self._text_buffer = ""
        self._emotion = ""

        # 生产者-消费者队列
        self._text_queue: asyncio.Queue = asyncio.Queue()
        self._consumer_task: Optional[asyncio.Task] = None

        # 状态
        self._should_stop = False
        self._is_running = False

        # 统计
        self._total_text = ""
        self._audio_chunks: List[bytes] = []
        self._sentence_count = 0  # 句子计数，用于调试

        logger.debug(f"StreamingTTSProcessor 初始化完成 (provider: {self.provider}, voice: {self.voice})")

    async def start(self):
        """
        启动消费者任务

        必须在使用前调用此方法
        """
        if self._consumer_task is not None and not self._consumer_task.done():
            logger.warning("[TTS] 消费者任务已在运行")
            return

        self._is_running = True
        self._consumer_task = asyncio.create_task(self._consumer_loop())
        logger.debug("[TTS] 消费者任务已启动")

    def add_text_nowait(self, text: str, emotion: str) -> bool:
        """
        非阻塞添加文本到队列（用于LLM回调）

        这是同步方法，不等待任何东西，立即返回。
        TTS消费者循环会独立处理队列中的文本。

        Args:
            text: 文本块

        Returns:
            是否成功添加到队列

        Raises:
            asyncio.QueueFull: 队列已满时抛出
        """
        if self._should_stop or not text:
            return False

        # 使用 put_nowait，完全非阻塞
        self._text_queue.put_nowait(text)
        self._total_text += text
        self._emotion = emotion
        return True

    async def add_text(self, text: str, emotion: str) -> bool:
        """
        添加LLM输出的文本块（异步版本）

        Args:
            text: 文本块

        Returns:
            是否成功添加到队列
        """
        if self._should_stop or not text:
            return False

        # 非阻塞放入队列
        await self._text_queue.put(text)
        self._total_text += text
        self._emotion = emotion
        return True

    async def _consumer_loop(self):
        """
        消费者循环 - 从队列中取出文本并处理

        这是后台任务，独立于文本添加
        """
        logger.debug("[TTS] 消费者循环启动")

        while self._is_running and not self._should_stop:
            try:
                # 从队列获取文本（带超时，避免永久阻塞）
                item = await asyncio.wait_for(
                    self._text_queue.get(),
                    timeout=0.5
                )

                # 检查是否是特殊信号
                if item is self._STOP_SIGNAL:
                    logger.debug("[TTS] 收到停止信号，退出消费者循环")
                    break

                if item is self._FLUSH_SIGNAL:
                    # 刷新信号，处理剩余缓冲区
                    logger.debug("[TTS] 收到刷新信号")
                    await self._process_buffer(is_flush=True)
                    # 刷新后检查是否需要停止
                    if self._should_stop:
                        logger.info("[TTS] 刷新后检测到停止信号，退出循环")
                        break
                    continue

                # 正常文本，添加到缓冲区
                self._text_buffer += item

                # 检查是否需要合成
                if self._should_synthesize():
                    await self._process_buffer(is_flush=False)
                    # 每个句子处理后检查是否需要停止（响应打断）
                    if self._should_stop:
                        logger.info("[TTS] 句子处理后检测到停止信号，退出循环")
                        break

            except asyncio.TimeoutError:
                # 超时，检查是否应该退出
                if self._text_queue.empty() and self._should_stop:
                    break
                continue
            except asyncio.CancelledError:
                logger.debug("[TTS] 消费者循环被取消")
                break
            except Exception as e:
                logger.error(f"[TTS] 消费者循环错误: {e}")
                import traceback
                traceback.print_exc()

        logger.debug("[TTS] 消费者循环结束")

    async def _process_buffer(self, is_flush: bool = False):
        """
        处理缓冲区中的文本

        Args:
            is_flush: 是否是刷新模式（处理所有剩余文本）
        """
        # 首先检查是否需要停止
        if self._should_stop:
            logger.info("[TTS] _process_buffer 检测到停止信号，跳过处理")
            return

        if not self._text_buffer:
            return

        # 分割文本
        if is_flush:
            # 刷新模式：处理所有剩余文本
            if len(self._text_buffer) < self.MIN_CHUNK_SIZE:
                # 剩余文本太短，跳过
                logger.debug(f"[TTS] flush跳过短文本: '{self._text_buffer[:20]}...'")
                self._text_buffer = ""
                return
            to_synthesize = self._text_buffer
            self._text_buffer = ""
        else:
            # 正常模式：按句子分割
            to_synthesize, remaining = self._split_text()
            self._text_buffer = remaining

        if not to_synthesize:
            return

        # 清理文本
        cleaned_text = clean_text_for_tts(to_synthesize)

        if not cleaned_text.strip():
            return

        self._sentence_count += 1
        sentence_num = self._sentence_count
        mode_str = "flush" if is_flush else "normal"

        logger.info(f"[TTS] 句子#{sentence_num}({mode_str}) 开始转换: '{cleaned_text[:50]}...'")

        try:
            audio_data = b""

            # 优先使用Qwen3 TTS
            if self.provider == "qwen3" and HAS_DASHSCOPE:
                audio_data = await self._synthesize_qwen3_stream(cleaned_text, sentence_num)
            else:
                # 使用Edge TTS
                audio_data = await self._synthesize_edge_stream(cleaned_text, sentence_num)

            # 音频合成完成后立即发送（不等待锁）
            if audio_data and len(audio_data) > 0 and self.on_audio_chunk and not self._should_stop:
                self._audio_chunks.append(audio_data)
                logger.info(f"[TTS] 句子#{sentence_num}({mode_str}) 转换完成，准备发送: {len(audio_data)} bytes")
                await self._call_callback(audio_data)
                logger.info(f"[TTS] 句子#{sentence_num}({mode_str}) 发送完成")
            elif audio_data and len(audio_data) == 0:
                logger.warning(f"[TTS] 句子#{sentence_num}({mode_str}) 音频为空，跳过发送")
            else:
                logger.debug(f"[TTS] 句子#{sentence_num}({mode_str}) 无音频或已停止")

        except Exception as e:
            logger.error(f"[TTS] 句子#{sentence_num} 转换失败: {e}")

    def _should_synthesize(self) -> bool:
        """判断是否需要进行TTS转换 - 按句子分段"""
        # 如果已停止，不进行合成
        if self._should_stop:
            return False

        if not self._text_buffer:
            return False

        # 检查是否有句子结束符
        for ending in self.SENTENCE_ENDINGS:
            if ending in self._text_buffer:
                return True

        # 超过最大长度也要转换
        if len(self._text_buffer) >= self.MAX_CHUNK_SIZE:
            return True

        return False

    def _split_text(self) -> tuple:
        """
        分割文本，返回可以转换的部分和剩余部分

        Returns:
            (待转换文本, 剩余文本)
        """
        if not self._text_buffer:
            return "", ""

        # 查找句子结束位置
        best_pos = -1
        for ending in self.SENTENCE_ENDINGS:
            pos = self._text_buffer.find(ending)
            if pos != -1:
                if best_pos == -1 or pos < best_pos:
                    best_pos = pos

        if best_pos != -1:
            # 找到句子结束符，在结束符后分割
            split_pos = best_pos + 1
            to_synthesize = self._text_buffer[:split_pos]
            remaining = self._text_buffer[split_pos:]
            return to_synthesize, remaining

        # 没有找到句子结束符，检查长度
        if len(self._text_buffer) >= self.MAX_CHUNK_SIZE:
            # 按最大长度分割（尽量在空格或标点处分割）
            split_pos = self.MAX_CHUNK_SIZE

            # 尝试找到更好的分割点
            for i in range(self.MAX_CHUNK_SIZE - 1, self.MIN_CHUNK_SIZE, -1):
                if i < len(self._text_buffer) and self._text_buffer[i] in ['，', ',', ' ', '、']:
                    split_pos = i + 1
                    break

            to_synthesize = self._text_buffer[:split_pos]
            remaining = self._text_buffer[split_pos:]
            return to_synthesize, remaining

        return "", self._text_buffer

    async def _synthesize_qwen3_stream(self, text: str, sentence_num: int) -> bytes:
        """使用Qwen3 TTS流式合成（在线程池中执行，不阻塞事件循环）"""
        # 定义同步合成函数
        def _sync_synthesize():
            audio_data = b""
            try:
                emotion_contexts = {
                    "positive": "用户现在心情不错，你可以用轻松愉快的语气回应，分享ta的喜悦。",
                    "negative": "用户情绪有些低落，请用温和关心的语气回应，表达理解和支持。",
                    "angry": "用户现在有些烦躁，请先安抚情绪，语气温和，再帮ta解决问题。",
                    "sad": "用户现在心情难过，请表达关心和理解，让ta感到被陪伴。",
                    "surprised": "用户感到惊喜，你可以一起感受这份惊喜，保持好奇。",
                    "neutral": "用户情绪平和，正常交流即可。",
                }
                # 调用Qwen3 TTS流式API（同步调用）
                response = MultiModalConversation.call(
                    model="qwen3-tts-instruct-flash",
                    text=text,
                    voice=self.voice,
                    language_type="Chinese",
                    instructions=emotion_contexts.get(self._emotion, ""),
                    optimize_instructions=True,
                    stream=True
                )

                # 收集流式输出的音频数据（同步迭代）
                for chunk in response:
                    if self._should_stop:
                        break

                    if chunk.output is not None:
                        audio = chunk.output.audio
                        if audio and hasattr(audio, 'data') and audio.data:
                            # Base64解码音频数据
                            wav_bytes = base64.b64decode(audio.data)
                            audio_data += wav_bytes

                return pcm_to_wav(audio_data)

            except Exception as e:
                logger.error(f"[TTS] Qwen3 TTS流式合成失败: {e}")
                return b""

        # 在线程池中执行同步合成，不阻塞事件循环
        loop = asyncio.get_running_loop()
        try:
            audio_data = await loop.run_in_executor(None, _sync_synthesize)
            return audio_data
        except Exception as e:
            logger.error(f"[TTS] Qwen3 TTS执行器错误: {e}")
            # 尝试使用Edge TTS作为后备
            return await self._synthesize_edge_stream(text, sentence_num)

    async def _synthesize_edge_stream(self, text: str, sentence_num: int) -> bytes:
        """使用Edge TTS流式合成"""
        import edge_tts

        voice = self.config.get("voice", "zh-CN-XiaoxiaoNeural")
        rate = self.config.get("rate", "+0%")

        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate=rate
        )

        # 收集完整音频数据
        audio_data = b""
        async for chunk in communicate.stream():
            if self._should_stop:
                break

            if chunk["type"] == "audio":
                audio_data += chunk["data"]

        return audio_data

    async def _call_callback(self, audio_data: bytes):
        """调用音频回调"""
        if self.on_audio_chunk:
            try:
                if asyncio.iscoroutinefunction(self.on_audio_chunk):
                    await self.on_audio_chunk(audio_data)
                else:
                    self.on_audio_chunk(audio_data)
            except Exception as e:
                logger.error(f"音频回调错误: {e}")

    async def flush(self):
        """
        刷新缓冲区，转换剩余文本

        发送刷新信号给消费者任务
        """
        if self._should_stop:
            return

        # 发送刷新信号
        await self._text_queue.put(self._FLUSH_SIGNAL)

        # 等待队列处理完成（等待缓冲区清空）
        max_wait = 30  # 最多等待30秒
        waited = 0
        while self._text_buffer and waited < max_wait:
            await asyncio.sleep(0.1)
            waited += 0.1

        if self._text_buffer:
            logger.warning(f"[TTS] flush超时，剩余文本: '{self._text_buffer[:30]}...'")

    async def finalize(self) -> bytes:
        """
        完成处理，返回所有音频数据

        Returns:
            完整的音频数据
        """
        # 发送刷新信号处理剩余缓冲区
        await self._text_queue.put(self._FLUSH_SIGNAL)

        # 等待队列清空
        max_wait = 60  # 最多等待60秒
        waited = 0
        while not self._text_queue.empty() and waited < max_wait:
            await asyncio.sleep(0.1)
            waited += 0.1

        # 等待缓冲区处理完成
        waited = 0
        while self._text_buffer and waited < max_wait:
            await asyncio.sleep(0.1)
            waited += 0.1

        # 停止消费者任务
        self._is_running = False
        await self._text_queue.put(self._STOP_SIGNAL)

        # 等待消费者任务结束
        if self._consumer_task and not self._consumer_task.done():
            try:
                await asyncio.wait_for(self._consumer_task, timeout=5)
            except asyncio.TimeoutError:
                self._consumer_task.cancel()
                try:
                    await self._consumer_task
                except asyncio.CancelledError:
                    pass

        self._consumer_task = None

        # 合并所有音频块
        all_audio = b"".join(self._audio_chunks)

        logger.info(f"[TTS] 处理完成: 总文本 {len(self._total_text)} 字符, 句子数 {self._sentence_count}, 音频 {len(all_audio)} bytes")

        return all_audio

    def stop(self):
        """停止处理"""
        logger.info("[TTS] stop() 被调用，停止TTS处理")
        self._should_stop = True
        self._is_running = False

        # 清空文本缓冲区（打断时丢弃未处理的文本）
        self._text_buffer = ""

        # 清空队列中的待处理文本
        cleared = 0
        while not self._text_queue.empty():
            try:
                self._text_queue.get_nowait()
                cleared += 1
            except asyncio.QueueEmpty:
                break
        if cleared > 0:
            logger.info(f"[TTS] 已清空队列中 {cleared} 个待处理文本块")

        # 发送停止信号
        try:
            self._text_queue.put_nowait(self._STOP_SIGNAL)
        except asyncio.QueueFull:
            pass

        # 取消消费者任务
        if self._consumer_task and not self._consumer_task.done():
            self._consumer_task.cancel()

        self._consumer_task = None

    def reset(self):
        """重置状态"""
        self._text_buffer = ""
        self._total_text = ""
        self._audio_chunks = []
        self._should_stop = False
        self._is_running = False
        self._sentence_count = 0

        # 清空队列
        while not self._text_queue.empty():
            try:
                self._text_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # 取消消费者任务
        if self._consumer_task and not self._consumer_task.done():
            self._consumer_task.cancel()
        self._consumer_task = None

    @property
    def is_processing(self) -> bool:
        """是否正在处理"""
        return self._is_running and (not self._text_queue.empty() or self._text_buffer)

    @property
    def total_text(self) -> str:
        return self._total_text
