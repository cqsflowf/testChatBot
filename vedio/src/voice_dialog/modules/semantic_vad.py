"""
全双工语音对话系统 v3.0 - 语义VAD流式判断模块

职责：
- 流式文本实时检测
- 边接收ASR文本边判断语义完整性
- 使用 Qwen Omni Flash 模型

关键设计：
- 语义VAD负责语义完整性判断（"说完"、"拒识"、"是不是真打断"）
- 满足声学静音阈值或自己判断出语义完整了，再给后端LLM模型推理
"""
import asyncio
import json
from typing import Optional, List, Dict
from ..core.logger import logger

try:
    from openai import AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

from ..core.types import SemanticState, SemanticVADResult
from ..core.config import get_config


class VoiceValidity:
    """人声有效性判断结果"""
    VALID = "valid"           # 有效人声（有实际语义内容）
    NOISE = "noise"           # 噪声/环境音
    FILLER = "filler"         # 语气助词/停顿词（嗯、啊、呃等）
    PENDING = "pending"       # 待判断（文本太短）


class SemanticVADProcessor:
    """
    语义VAD流式处理器

    使用 Qwen 模型进行流式语义判断
    边接收ASR文本边判断语义完整性

    v3.2 优化：
    - 添加打断模式，打断后使用更宽松的判断标准
    - 支持快速触发关键词
    - 有效人声判断（区分语气助词和有效内容）
    """

    # 使用 qwen-flash 进行文本语义分析（OpenAI兼容模式）
    MODEL_NAME = "qwen-flash"

    # 语义判断提示词
    SEMANTIC_PROMPT = """分析用户的输入文本，判断用户的表达状态。

语义状态说明：
- complete: 用户完整表达了一个意图或问题，可以开始处理
- continuing: 用户还在说，表达不完整，需要继续等待
- interrupted: 用户被打断或中途停止
- rejected: 无法识别或无效输入（如噪音、无意义声音）

判断要点：
1. 是否有完整的意图或问题？
2. 句子是否完整（主谓宾齐全）？
3. 是否有明显的结束标记（问号、句号）？
4. 是否还有后续内容的可能性？

请以JSON格式返回：
{
    "semantic_state": "complete/continuing/interrupted/rejected",
    "confidence": 0.95,
    "reason": "判断理由"
}"""

    # ========== v3.1 打断模式关键词 ==========
    # 打断场景下，这些关键词可以立即触发完整判断
    INTERRUPT_TRIGGER_WORDS = [
        # 停止类
        "停", "停下", "停止", "不要", "别", "闭嘴", "安静",
        # 替换类
        "换", "换一个", "下一个", "算了", "不", "取消",
        # 疑问类（打断后问问题）
        "什么", "为什么", "怎么", "哪", "谁",
        # 确认类
        "好的", "行", "可以", "嗯", "对", "是",
        # 短指令
        "帮我", "我要", "我想", "给我",
    ]

    # 打断模式下更宽松的完整判断关键词
    INTERRUPT_COMPLETE_WORDS = [
        "停", "换", "算", "好", "行", "对", "是", "不", "要", "有",
        "帮", "给", "想", "查", "找", "看", "听", "说", "问",
    ]

    # ========== v3.2 有效人声判断关键词 ==========
    # 语气助词/停顿词（这些不算有效人声）
    FILLER_WORDS = [
        "嗯", "啊", "呃", "额", "唔", "哦", "噢", "哈", "嘿", "哎",
        "那个", "就是", "这个", "然后", "所以", "但是", "其实",
        "呃呃", "啊啊", "嗯嗯",
    ]

    # 有效人声关键词（出现这些词立即确认为有效人声）
    VALID_VOICE_WORDS = [
        # 指令类
        "帮我", "我要", "我想", "给我", "请", "查", "找", "看", "听",
        "打开", "关闭", "播放", "设置", "停止", "开始", "换",
        # 疑问类
        "什么", "怎么", "为什么", "哪", "谁", "几", "多少", "吗", "呢",
        # 确认类
        "好的", "行", "可以", "对", "是", "不", "好",
        # 停止类
        "停", "算", "取消", "不要", "别",
    ]

    def __init__(self):
        self.config = get_config().semantic_vad if hasattr(get_config(), 'semantic_vad') else {}
        self.api_key = get_config().qwen_omni.get("api_key", "")
        self.model = self.config.get("model", self.MODEL_NAME)

        # 流式状态
        self._text_buffer = ""
        self._last_state = SemanticState.CONTINUING
        self._last_confidence = 0.5
        self._judgment_history: List[Dict] = []

        # v3.1 打断模式标志
        self._interrupt_mode = False

        # 配置
        self.min_text_length = self.config.get("streaming", {}).get("min_text_length", 2)
        self.max_wait_ms = self.config.get("streaming", {}).get("max_wait_ms", 5000)

        self._init_client()

    def _init_client(self):
        """初始化 API 客户端"""
        if not HAS_OPENAI:
            logger.warning("openai 库未安装，将使用规则判断")
            return

        if self.api_key:
            logger.info(f"语义VAD 客户端初始化成功 (模型: {self.model})")
        else:
            logger.warning("未配置 API 密钥，将使用规则判断")

    async def judge(self, text: str, is_final: bool = False) -> SemanticVADResult:
        """
        判断文本的语义完整性

        Args:
            text: ASR流式输出的文本
            is_final: 是否是最终文本

        Returns:
            语义VAD判断结果
        """
        # 更新缓冲区
        self._text_buffer = text

        # 文本太短，继续等待（但不等待太久）
        if len(text.strip()) < 1:
            self._last_state = SemanticState.CONTINUING
            return SemanticVADResult(
                state=SemanticState.CONTINUING,
                confidence=0.7,
                reason="文本为空，继续等待"
            )

        # v3.1: 打断模式下使用快速判断
        if self._interrupt_mode:
            result = self._judge_interrupt_mode(text)
            if result.state == SemanticState.COMPLETE:
                self._last_state = result.state
                self._last_confidence = result.confidence
                logger.info(f"[打断模式] 快速判断完整: '{text}'")
                return result

        # 如果是最终结果，使用更宽松的判断
        if is_final:
            result = self._judge_with_rules(text)
            # 如果仍然不是完整，但有内容，强制完整
            if result.state != SemanticState.COMPLETE and len(text.strip()) >= 2:
                result = SemanticVADResult(
                    state=SemanticState.COMPLETE,
                    confidence=0.7,
                    reason="用户说话结束，判断为完整"
                )
        else:
            # 使用模型判断
            if HAS_OPENAI and self.api_key:
                result = await self._judge_with_model(text)
            else:
                result = self._judge_with_rules(text)

        # 更新状态
        self._last_state = result.state
        self._last_confidence = result.confidence

        # 记录判断历史
        self._judgment_history.append({
            "text": text,
            "state": result.state.value,
            "confidence": result.confidence,
            "reason": result.reason
        })

        return result

    def _judge_interrupt_mode(self, text: str) -> SemanticVADResult:
        """
        v3.1 打断模式下的快速判断

        打断后用户说话通常很短，使用更宽松的判断标准
        """
        text = text.strip()
        clean_text = text

        # 去除噪声词
        noise_words = ["嗯", "啊", "呃", "那个", "就是"]
        for noise in noise_words:
            clean_text = clean_text.replace(noise, "")
        clean_text = clean_text.strip()

        # 1. 检查是否有打断触发词（立即完整）
        for trigger in self.INTERRUPT_TRIGGER_WORDS:
            if trigger in text:
                return SemanticVADResult(
                    state=SemanticState.COMPLETE,
                    confidence=0.95,
                    reason=f"[打断模式] 检测到触发词: {trigger}"
                )

        # 2. 打断模式下，有1个以上关键词就判断完整
        keyword_count = sum(1 for kw in self.INTERRUPT_COMPLETE_WORDS if kw in clean_text)
        if keyword_count >= 1 and len(clean_text) >= 1:
            return SemanticVADResult(
                state=SemanticState.COMPLETE,
                confidence=0.85,
                reason="[打断模式] 包含意图关键词"
            )

        # 3. 打断模式下，有效文本长度>=2就判断完整
        if len(clean_text) >= 2:
            return SemanticVADResult(
                state=SemanticState.COMPLETE,
                confidence=0.8,
                reason="[打断模式] 文本长度足够"
            )

        # 4. 有任何实际内容，可能完整
        if len(clean_text) >= 1:
            return SemanticVADResult(
                state=SemanticState.COMPLETE,
                confidence=0.7,
                reason="[打断模式] 有内容，判断完整"
            )

        # 继续等待
        return SemanticVADResult(
            state=SemanticState.CONTINUING,
            confidence=0.5,
            reason="[打断模式] 继续等待"
        )

    def set_interrupt_mode(self, enabled: bool):
        """
        设置打断模式

        打断模式下使用更宽松的判断标准，加快响应速度
        """
        self._interrupt_mode = enabled
        if enabled:
            logger.info("语义VAD 进入打断模式（快速判断）")
        else:
            logger.debug("语义VAD 退出打断模式")

    def check_voice_validity(self, text: str) -> str:
        """
        v3.2 检查是否是有效人声

        用于打断判断：区分有效人声和噪声/语气助词

        Args:
            text: ASR流式输出的文本

        Returns:
            VoiceValidity:
            - VALID: 有效人声，应立即停止播报
            - FILLER: 语气助词/停顿词，不应打断
            - NOISE: 噪声，不应打断
            - PENDING: 待判断，继续等待
        """
        text = text.strip()

        # 空文本，继续等待
        if not text:
            return VoiceValidity.PENDING

        # 检查是否有有效人声关键词
        for word in self.VALID_VOICE_WORDS:
            if word in text:
                logger.info(f"[有效人声] 检测到关键词: '{word}'")
                return VoiceValidity.VALID

        # 去除语气助词后的文本
        clean_text = text
        for filler in self.FILLER_WORDS:
            clean_text = clean_text.replace(filler, "")
        clean_text = clean_text.strip()

        # 如果去除语气助词后还有足够内容，认为是有效人声
        if len(clean_text) >= 2:
            logger.info(f"[有效人声] 有效内容: '{clean_text}'")
            return VoiceValidity.VALID

        # 如果只有1个有效字符，可能是有效人声开始
        if len(clean_text) == 1:
            # 检查是否是常见单字指令
            single_valid = ["停", "好", "行", "对", "是", "不", "换", "算"]
            if clean_text in single_valid:
                logger.info(f"[有效人声] 单字指令: '{clean_text}'")
                return VoiceValidity.VALID
            return VoiceValidity.PENDING

        # 只有语气助词
        if text and not clean_text:
            logger.debug(f"[语气助词] 忽略: '{text}'")
            return VoiceValidity.FILLER

        # 继续等待
        return VoiceValidity.PENDING

    async def _judge_with_model(self, text: str) -> SemanticVADResult:
        """使用 Qwen3-flash 模型进行语义判断"""
        try:
            client = AsyncOpenAI(
                api_key=self.api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )

            # 使用 qwen3-flash 模型
            response = await client.chat.completions.create(
                model=self.MODEL_NAME,  # qwen3-flash
                messages=[
                    {
                        "role": "system",
                        "content": self.SEMANTIC_PROMPT
                    },
                    {
                        "role": "user",
                        "content": f"判断以下文本的语义状态：\n{text}"
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=100
            )

            result_text = response.choices[0].message.content

            # 尝试解析JSON，处理多种情况
            try:
                # 先尝试直接解析
                result_json = json.loads(result_text)
            except json.JSONDecodeError:
                # 尝试提取JSON代码块
                import re
                # 匹配 ```json ... ``` 或 ``` ... ```
                json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', result_text)
                if json_match:
                    try:
                        result_json = json.loads(json_match.group(1).strip())
                    except json.JSONDecodeError:
                        # 尝试提取花括号内容
                        brace_match = re.search(r'\{[\s\S]*\}', result_text)
                        if brace_match:
                            try:
                                result_json = json.loads(brace_match.group())
                            except json.JSONDecodeError as je:
                                logger.debug(f"语义VAD JSON解析失败，使用规则判断: {je}")
                                return self._judge_with_rules(text)
                        else:
                            logger.debug(f"语义VAD JSON解析失败，使用规则判断: 无法提取JSON")
                            return self._judge_with_rules(text)
                else:
                    # 尝试直接提取花括号内容
                    brace_match = re.search(r'\{[\s\S]*\}', result_text)
                    if brace_match:
                        try:
                            result_json = json.loads(brace_match.group())
                        except json.JSONDecodeError as je:
                            logger.debug(f"语义VAD JSON解析失败，使用规则判断: {je}")
                            return self._judge_with_rules(text)
                    else:
                        logger.debug(f"语义VAD JSON解析失败，使用规则判断: 无JSON内容")
                        return self._judge_with_rules(text)

            # 解析状态
            state_str = result_json.get("semantic_state", "continuing")
            if state_str not in ["complete", "continuing", "interrupted", "rejected"]:
                state_str = "continuing"

            return SemanticVADResult(
                state=SemanticState(state_str),
                confidence=result_json.get("confidence", 0.8),
                reason=result_json.get("reason", "")
            )

        except json.JSONDecodeError as e:
            logger.debug(f"语义VAD JSON解析失败，使用规则判断: {e}")
            return self._judge_with_rules(text)
        except Exception as e:
            logger.debug(f"语义VAD模型判断失败，使用规则判断: {e}")
            return self._judge_with_rules(text)

    def _judge_with_rules(self, text: str) -> SemanticVADResult:
        """基于规则的语义判断 - 优化版，更容易判断完整"""
        text = text.strip()

        # 空文本或太短
        if not text or len(text) < 1:
            return SemanticVADResult(
                state=SemanticState.CONTINUING,
                confidence=0.6,
                reason="文本为空或太短"
            )

        # 忽略噪声词 - 这些词不应该单独触发判断
        noise_words = ["嗯", "啊", "呃", "那个", "就是", "这个", "然后", "所以", "但是", "其实", "额", "唔"]
        clean_text = text
        for noise in noise_words:
            clean_text = clean_text.replace(noise, "")
        clean_text = clean_text.strip()

        # 如果只剩下噪声，继续等待
        if not clean_text or len(clean_text) < 2:
            return SemanticVADResult(
                state=SemanticState.CONTINUING,
                confidence=0.7,
                reason="只是噪声词，继续等待"
            )

        # 结束标点（强信号）
        end_punctuation = text[-1] if text else ""
        has_end_mark = end_punctuation in ["。", "！", "？", "!", "?", "."]

        # 问句判断（问号或疑问词）
        is_question = "？" in text or "?" in text or any(
            text.endswith(q) or q in text for q in ["吗", "呢", "吧", "么"]
        )

        # 完整意图关键词（用户说这些词时通常已经完成表达）
        complete_indicators = [
            # 动作类
            "帮我", "请", "我要", "我想", "查一下", "播放", "设置", "打开", "关闭",
            "查询", "告诉我", "怎么样", "找", "搜", "看", "听", "读", "写",
            # 信息类
            "是什么", "怎么", "如何", "为什么", "哪", "谁", "几", "多少",
            # 结束语
            "好的", "好的", "可以", "行", "谢谢", "感谢", "再见", "拜拜"
        ]
        has_intent = any(ind in text for kw in complete_indicators for ind in [kw] if kw in text)

        # 长度判断 - 如果够长，更可能完整
        text_length = len(clean_text)

        # ========== 判断逻辑 ==========

        # 1. 有结束标点 + 有实际内容 → 大概率完整
        if has_end_mark and len(clean_text) >= 2:
            return SemanticVADResult(
                state=SemanticState.COMPLETE,
                confidence=0.9,
                reason="有结束标点且有实质内容"
            )

        # 2. 问句 → 大概率完整
        if is_question and len(clean_text) >= 2:
            return SemanticVADResult(
                state=SemanticState.COMPLETE,
                confidence=0.85,
                reason="是问句"
            )

        # 3. 有意图关键词 + 长度足够 → 较高概率完整
        if has_intent and text_length >= 3:
            return SemanticVADResult(
                state=SemanticState.COMPLETE,
                confidence=0.8,
                reason="包含意图关键词"
            )

        # 4. 纯文本长度足够（>= 4个有效字符）→ 可能完整
        # v3.6: 保持长度要求为4，但不检查结束特征，避免大段话中间被截断
        if text_length >= 4:
            # 检查是否像是在说话中途
            mid_sentence_hints = ["而且", "并且", "然后", "还有", "另外", "还有呢", "接着", "之后", "同时"]
            is_mid_sentence = any(hint in text for hint in mid_sentence_hints)
            if not is_mid_sentence:
                return SemanticVADResult(
                    state=SemanticState.COMPLETE,
                    confidence=0.7,
                    reason="文本长度足够，判断为完整"
                )

        # 5. 长度较短但有结束标记特征
        if text_length >= 2 and (has_end_mark or is_question):
            return SemanticVADResult(
                state=SemanticState.COMPLETE,
                confidence=0.75,
                reason="短句但符合完整特征"
            )

        # 6. 默认继续等待
        return SemanticVADResult(
            state=SemanticState.CONTINUING,
            confidence=0.5,
            reason="继续等待更多输入"
        )

    def is_complete(self) -> bool:
        """
        检查当前语义状态是否完整

        Returns:
            是否语义完整
        """
        return self._last_state == SemanticState.COMPLETE

    @property
    def current_state(self) -> SemanticState:
        """当前语义状态"""
        return self._last_state

    @property
    def current_text(self) -> str:
        """当前缓冲的文本"""
        return self._text_buffer

    def reset(self):
        """重置状态"""
        self._text_buffer = ""
        self._last_state = SemanticState.CONTINUING
        self._last_confidence = 0.5
        self._judgment_history.clear()
        self._interrupt_mode = False  # v3.1 重置打断模式
        logger.debug("语义VAD 状态已重置")

    def get_judgment_history(self) -> List[Dict]:
        """获取判断历史"""
        return self._judgment_history.copy()


class StreamingSemanticVAD:
    """
    流式语义VAD处理器
    与ASR流式输出配合使用

    v3.1 优化：
    - 支持打断模式，打断后快速判断
    """

    def __init__(self):
        self.processor = SemanticVADProcessor()
        self._is_active = False
        self._result_queue = asyncio.Queue()

    async def start(self, interrupt_mode: bool = False):
        """
        启动流式处理

        Args:
            interrupt_mode: 是否为打断模式（使用更宽松的判断标准）
        """
        self._is_active = True
        self.processor.reset()
        self.processor.set_interrupt_mode(interrupt_mode)
        logger.info(f"流式语义VAD 已启动 (打断模式: {interrupt_mode})")

    async def process_text(self, text: str, is_final: bool = False) -> SemanticVADResult:
        """
        处理ASR流式输出的文本

        Args:
            text: ASR输出的文本
            is_final: 是否是最终文本

        Returns:
            语义VAD判断结果
        """
        if not self._is_active:
            return SemanticVADResult(
                state=SemanticState.CONTINUING,
                confidence=0.5,
                reason="未启动"
            )

        result = await self.processor.judge(text, is_final)

        # 如果语义完整，放入队列
        if result.state == SemanticState.COMPLETE:
            await self._result_queue.put(result)

        return result

    async def wait_for_complete(self, timeout: float = 5.0) -> Optional[SemanticVADResult]:
        """
        等待语义完整

        Args:
            timeout: 超时时间（秒）

        Returns:
            语义完整的结果，或超时返回None
        """
        try:
            return await asyncio.wait_for(
                self._result_queue.get(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            return None

    async def stop(self) -> SemanticVADResult:
        """停止流式处理并返回最终结果"""
        self._is_active = False

        # 如果缓冲区有内容，进行最终判断
        if self.processor.current_text:
            # 进行最终判断
            final_result = await self.processor.judge(
                self.processor.current_text,
                is_final=True
            )

            # 如果仍然是 CONTINUING，检查文本长度
            # 如果有足够内容，强制设置为 COMPLETE
            if final_result.state == SemanticState.CONTINUING:
                clean_text = self.processor.current_text.strip()
                # 去除噪声词
                noise_words = ["嗯", "啊", "呃", "那个", "就是", "这个", "然后"]
                for noise in noise_words:
                    clean_text = clean_text.replace(noise, "")
                clean_text = clean_text.strip()

                # 如果有3个以上有效字符，认为完整
                if len(clean_text) >= 3:
                    final_result = SemanticVADResult(
                        state=SemanticState.COMPLETE,
                        confidence=0.7,
                        reason="用户停止说话，根据内容判断为完整"
                    )
                elif len(clean_text) >= 1:
                    # 有一些内容，可能完整
                    final_result = SemanticVADResult(
                        state=SemanticState.COMPLETE,
                        confidence=0.6,
                        reason="用户停止说话，内容较短但判断为完整"
                    )
                else:
                    # 只有噪声
                    final_result = SemanticVADResult(
                        state=SemanticState.REJECTED,
                        confidence=0.7,
                        reason="只有噪声，忽略"
                    )

            return final_result

        return SemanticVADResult(
            state=self.processor.current_state,
            confidence=self.processor._last_confidence,
            reason="流式处理结束"
        )

    @property
    def is_active(self) -> bool:
        """是否正在处理"""
        return self._is_active

    def reset(self):
        """重置"""
        self.processor.reset()
        self._result_queue = asyncio.Queue()