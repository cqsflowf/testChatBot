"""
全双工语音对话系统 v3.0 - LLM任务规划器
支持多轮对话上下文管理，智能工具匹配
使用统一的工具注册中心 ToolRegistry

v3.0 特性：
- 自动注入当前时间
- 情绪适配响应
- 多轮对话上下文
"""
import asyncio
import json
import re
from typing import Optional, List, Dict, Any
from ..core.logger import logger

try:
    from openai import AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    AsyncOpenAI = None
    HAS_OPENAI = False

from ..core.types import (
    LLMInput,
    LLMResponse,
    ToolCall,
    ToolResult,
    Message,
    EmotionType,
)
from ..core.config import get_config
from ..core.tool_registry import tool_registry


# 中国主要城市列表 - 扩展版
CHINESE_CITIES = [
    # 直辖市
    "北京", "上海", "天津", "重庆",
    # 省会城市
    "广州", "深圳", "杭州", "南京", "苏州", "成都", "武汉", "西安", "长沙", "郑州",
    "青岛", "大连", "厦门", "昆明", "福州", "合肥", "南昌", "哈尔滨", "长春", "沈阳",
    "济南", "石家庄", "太原", "南宁", "海口", "贵阳", "兰州", "西宁", "银川", "乌鲁木齐",
    "呼和浩特", "拉萨",
    # 其他重要城市
    "珠海", "东莞", "佛山", "宁波", "无锡", "温州", "常州", "徐州", "烟台", "潍坊",
    "临沂", "唐山", "保定", "大同", "桂林", "三亚", "洛阳", "开封", "绍兴", "嘉兴",
    "湖州", "金华", "台州", "舟山", "衢州", "丽水", "鞍山", "抚顺", "本溪", "丹东",
    "锦州", "营口", "阜新", "辽阳", "盘锦", "铁岭", "朝阳", "葫芦岛", "吉林", "四平",
    "辽源", "通化", "白山", "松原", "白城", "齐齐哈尔", "鸡西", "鹤岗", "双鸭山", "大庆",
    "伊春", "佳木斯", "七台河", "牡丹江", "黑河", "绥化", "大兴安岭"
]


class LLMTaskPlanner:
    """
    LLM任务规划器 v3.2
    支持多轮对话上下文管理和智能工具匹配
    使用统一的工具注册中心 ToolRegistry
    每次对话自动注入当前时间
    支持流式输出
    """

    SYSTEM_PROMPT_TEMPLATE = """你是一个智能语音助手，具备以下能力：
1. 理解用户意图并给出自然、友好的回应
2. 在需要时调用工具完成任务
3. 根据用户情绪调整回应方式
4. 记住对话上下文，支持多轮对话

当前环境信息：
- 当前时间：{current_time}
- 当前位置：{current_location}

你可以使用以下工具：
{tools_description}

重要规则：
1. 当用户请求需要工具才能完成时，调用相应工具
2. 你已经知道当前时间，不需要再调用get_current_time工具来回答时间问题
3. 记住用户之前提到的信息，支持上下文对话
4. 如果用户使用代词（"它"、"那里"等），根据上下文理解指代对象
5. 当用户询问天气、空气质量等需要位置的服务时，如果用户没有明确说明地址，请礼貌地询问用户的具体位置
6. 回答要简洁友好，适合语音播报，不要使用Markdown格式和表情符号"""

    # 情绪适配前缀
    EMOTION_PREFIXES = {
        EmotionType.POSITIVE: "很高兴能帮到您！",
        EmotionType.NEGATIVE: "我理解您的感受。",
        EmotionType.ANGRY: "抱歉给您带来不好的体验。",
        EmotionType.SAD: "我理解您的心情，",
        EmotionType.SURPRISED: "确实令人惊讶！",
        EmotionType.NEUTRAL: "",
    }

    def __init__(self):
        self.config = get_config().llm
        self.client: Optional[AsyncOpenAI] = None
        self.conversation_history: List[Message] = []
        self.context: Dict[str, Any] = {}  # 存储对话上下文信息
        self._current_time: Optional[str] = None  # 缓存当前时间
        self._init_client()

    def _init_client(self):
        """初始化LLM客户端"""
        if not HAS_OPENAI:
            logger.warning("openai库未安装，将使用模拟模式")
            return

        api_key = self.config.get("api_key", "") or get_config().get_api_key()
        base_url = self.config.get("base_url", "")

        if api_key:
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url
            )
            logger.info("LLM客户端初始化成功")
        else:
            logger.warning("未配置LLM API密钥，将使用模拟模式")

    async def _get_tools(self) -> List[Dict]:
        """从 ToolRegistry 获取工具定义"""
        tools = await tool_registry.get_all_tools()
        return [t.to_openai_tool() for t in tools]

    def _get_tools_description(self) -> str:
        """获取工具描述用于系统提示"""
        return tool_registry.get_tool_schemas_for_prompt()

    async def plan(self, llm_input: LLMInput) -> LLMResponse:
        """
        任务规划
        接收融合后的输入，返回响应和工具调用
        使用 ToolRegistry 动态获取工具
        自动注入当前时间
        """
        if self.client is None:
            return self._mock_plan(llm_input)

        try:
            # 获取当前时间
            from datetime import datetime
            now = datetime.now()
            weekday_names = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
            current_time = f"{now.strftime('%Y年%m月%d日')} {weekday_names[now.weekday()]} {now.strftime('%H:%M')}"
            current_location = "上海"  # 默认位置

            # 从 ToolRegistry 获取工具
            tools = await self._get_tools()
            tools_description = self._get_tools_description()
            system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(
                tools_description=tools_description,
                current_time=current_time,
                current_location=current_location
            )

            # 构建消息
            messages = self._build_messages(llm_input, system_prompt)

            # 调用LLM
            response = await self.client.chat.completions.create(
                model=self.config.get("model", "qwen-plus"),
                messages=messages,
                tools=tools if self.config.get("tools", {}).get("enabled", True) else None,
                temperature=self.config.get("generation", {}).get("temperature", 0.7),
                max_tokens=self.config.get("generation", {}).get("max_tokens", 1024)
            )

            choice = response.choices[0]
            message = choice.message

            # 解析工具调用
            tool_calls = []
            if message.tool_calls:
                for tc in message.tool_calls:
                    tool_calls.append(ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments)
                    ))

            # 构建响应
            response_text = message.content or ""

            # 情绪适配
            if self.config.get("emotion_context", {}).get("enabled", True):
                prefix = self.EMOTION_PREFIXES.get(llm_input.emotion, "")
                if prefix and not response_text.startswith(prefix):
                    response_text = prefix + response_text

            return LLMResponse(
                text=response_text,
                tool_calls=tool_calls,
                emotion_adapted=True
            )

        except Exception as e:
            logger.error(f"LLM规划失败: {e}")
            return self._mock_plan(llm_input)

    async def plan_stream(self, llm_input: LLMInput, on_chunk: Callable[[str], None] = None, on_tool_detected: Callable[[str], None] = None):
        """
        流式任务规划
        接收融合后的输入，流式返回响应

        Args:
            llm_input: 融合后的输入
            on_chunk: 流式输出回调函数（支持同步和异步），每次收到文本块时调用
            on_tool_detected: 工具检测回调函数（支持同步和异步），检测到工具调用时调用

        Returns:
            LLMResponse: 最终响应
        """
        if self.client is None:
            # 模拟模式也支持流式
            response = self._mock_plan(llm_input)
            # 模拟流式输出
            words = response.text
            for i in range(0, len(words), 3):
                chunk = words[i:i+3]
                if on_chunk:
                    on_chunk(chunk)
                await asyncio.sleep(0.05)
            # 模拟工具检测
            if response.tool_calls and on_tool_detected:
                for tc in response.tool_calls:
                    on_tool_detected(tc.name)
            return response

        try:
            # 获取并缓存当前时间
            await self._update_current_time()

            # 从 ToolRegistry 获取工具
            tools = await self._get_tools()
            tools_description = self._get_tools_description()
            system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(
                tools_description=tools_description,
                current_time=self._current_time,
                current_location=self.context.get("location", "上海")
            )

            # 构建消息
            messages = self._build_messages(llm_input, system_prompt)

            # 流式调用LLM
            import time
            call_time = time.time() * 1000
            logger.info(f"[LLM] 调用模型前, 时间: {call_time:.0f}ms, 输入文本: '{llm_input.text[:50]}...'")

            stream = await self.client.chat.completions.create(
                model=self.config.get("model", "qwen-plus"),
                messages=messages,
                tools=tools if self.config.get("tools", {}).get("enabled", True) else None,
                temperature=self.config.get("generation", {}).get("temperature", 0.7),
                max_tokens=self.config.get("generation", {}).get("max_tokens", 1024),
                stream=True  # 启用流式输出
            )

            response_text = ""
            tool_calls_data = []  # 存储工具调用数据
            detected_tools = set()  # 已检测到的工具名，避免重复回调

            first_chunk_received = False
            async for chunk in stream:
                if chunk.choices:
                    delta = chunk.choices[0].delta

                    # 记录收到第一个响应的时间
                    if not first_chunk_received:
                        import time
                        response_time = time.time() * 1000
                        logger.info(f"[LLM] 收到模型响应, 时间: {response_time:.0f}ms")
                        first_chunk_received = True

                    # 处理文本内容
                    if delta.content:
                        response_text += delta.content
                        if on_chunk:
                            # 回调是同步函数，直接调用，不等待
                            # 回调内部使用 put_nowait，不会阻塞
                            on_chunk(delta.content)

                    # 处理工具调用
                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            # 累积工具调用数据
                            while len(tool_calls_data) <= tc.index:
                                tool_calls_data.append({
                                    "id": "",
                                    "name": "",
                                    "arguments": ""
                                })
                            if tc.id:
                                tool_calls_data[tc.index]["id"] = tc.id
                            if tc.function:
                                if tc.function.name:
                                    tool_name = tc.function.name
                                    tool_calls_data[tc.index]["name"] = tool_name

                                    # 检测到工具名称，立即回调
                                    if tool_name and tool_name not in detected_tools:
                                        detected_tools.add(tool_name)
                                        logger.info(f"检测到工具调用: {tool_name}")
                                        if on_tool_detected:
                                            on_tool_detected(tool_name)

                                if tc.function.arguments:
                                    tool_calls_data[tc.index]["arguments"] += tc.function.arguments

            # 解析工具调用
            tool_calls = []
            for tc_data in tool_calls_data:
                if tc_data["name"]:
                    try:
                        arguments = json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
                    except json.JSONDecodeError:
                        arguments = {}
                    tool_calls.append(ToolCall(
                        id=tc_data["id"],
                        name=tc_data["name"],
                        arguments=arguments
                    ))

            # 情绪适配（流式模式下不添加前缀，因为已经开始输出了）
            # 这里我们在最终响应中添加前缀用于TTS
            final_text = response_text
            if self.config.get("emotion_context", {}).get("enabled", True):
                prefix = self.EMOTION_PREFIXES.get(llm_input.emotion, "")
                if prefix and not response_text.startswith(prefix):
                    final_text = prefix + response_text

            return LLMResponse(
                text=final_text,
                tool_calls=tool_calls,
                emotion_adapted=True
            )

        except Exception as e:
            logger.error(f"LLM流式规划失败: {e}")
            return self._mock_plan(llm_input)

    async def _update_current_time(self):
        """更新当前时间（调用时间工具获取准确时间）"""
        from datetime import datetime
        now = datetime.now()
        weekday_names = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        self._current_time = f"{now.strftime('%Y年%m月%d日')} {weekday_names[now.weekday()]} {now.strftime('%H:%M:%S')}"
        logger.debug(f"更新当前时间: {self._current_time}")

    def _build_messages(self, llm_input: LLMInput, system_prompt: str) -> List[Dict]:
        """构建消息列表"""
        messages = [{"role": "system", "content": system_prompt}]

        # 添加历史消息
        for msg in self.conversation_history[-10:]:  # 保留最近10条
            messages.append(msg.to_openai_format())

        # 构建用户消息（包含情绪上下文）
        user_content = llm_input.text
        if self.config.get("emotion_context", {}).get("enabled", True):
            emotion_info = f"\n[用户情绪: {llm_input.emotion.value}, 强度: {llm_input.emotion_intensity:.1f}]"
            user_content += emotion_info

        messages.append({"role": "user", "content": user_content})

        return messages

    async def summarize_tool_results(
        self,
        original_response: LLMResponse,
        tool_results: List[ToolResult]
    ) -> str:
        """
        总结工具调用结果
        """
        if not tool_results:
            return original_response.text

        if self.client is None:
            return self._mock_summarize(tool_results)

        try:
            # 构建总结请求
            results_text = "\n".join([
                f"工具 {tr.tool_call.name}: {tr.result}"
                for tr in tool_results
            ])

            response = await self.client.chat.completions.create(
                model=self.config.get("model", "qwen-plus"),
                messages=[
                    {
                        "role": "system",
                        "content": "请根据工具调用结果，给用户一个简洁友好的回复。"
                    },
                    {
                        "role": "user",
                        "content": f"工具调用结果：\n{results_text}"
                    }
                ],
                max_tokens=256
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"总结失败: {e}")
            return self._mock_summarize(tool_results)

    def _mock_plan(self, llm_input: LLMInput) -> LLMResponse:
        """模拟规划（用于测试）- 改进的意图识别"""
        text = llm_input.text.lower()
        tool_calls = []

        # ========== 时间查询（优先级最高）==========
        time_keywords = ["几点", "时间", "日期", "今天", "星期几", "周几", "几号", "月份", "年"]
        if any(kw in text for kw in time_keywords) and "天气" not in text:
            tool_calls.append(ToolCall(
                id="mock_time",
                name="get_current_time",
                arguments={}
            ))

        # ========== 天气查询 ==========
        # 检测城市名称
        detected_city = None
        for city in CHINESE_CITIES:
            if city in text:
                detected_city = city
                break

        # 检查上下文中的城市（用于"那上海呢"这类查询）
        if detected_city is None and self.context.get("last_city"):
            # 检测是否是延续天气查询（如"那上海呢"、"那里冷吗"）
            context_keywords = ["呢", "那里", "那儿", "那个", "呢", "冷", "热", "天气"]
            if any(kw in text for kw in context_keywords) and len(text) < 15:
                # 如果文本中有城市名，使用新城市
                for city in CHINESE_CITIES:
                    if city in text:
                        detected_city = city
                        break
                # 否则使用上下文城市
                if detected_city is None:
                    detected_city = self.context["last_city"]

        # 天气相关词汇 - 排除设备控制场景
        weather_keywords = ["天气", "气温", "热", "冷", "下雨", "下雪", "晴", "阴", "雨", "雪"]
        is_weather_query = any(kw in text for kw in weather_keywords)

        # 排除设备控制中的温度设置（如"空调温度28度"）
        device_control_keywords = ["空调", "暖气", "地暖", "调到", "调高", "调低"]
        is_device_control = any(kw in text for kw in device_control_keywords)

        # 只有在没有设备控制关键词时才触发天气查询
        if is_weather_query and not is_device_control:
            if detected_city:
                tool_calls.append(ToolCall(
                    id="mock_weather",
                    name="get_weather",
                    arguments={"city": detected_city}
                ))
            else:
                tool_calls.append(ToolCall(
                    id="mock_weather",
                    name="get_weather",
                    arguments={"city": "北京"}
                ))

        # ========== 复杂意图处理 ==========
        # 处理多个意图组合
        # 检测"提醒+天气"组合
        reminder_keywords = ["提醒", "闹钟", "记得", "别忘了"]
        weather_keywords_list = ["天气", "气温", "温度", "热", "冷", "下雨", "下雪", "晴", "阴", "雨", "雪", "度"]
        has_reminder = any(kw in text for kw in reminder_keywords)
        has_weather = any(kw in text for kw in weather_keywords_list)

        # ========== 提醒设置 ==========
        if has_reminder:
            # 提取时间信息
            time_str = "明天上午9点"  # 默认
            if "早上" in text or "上午" in text:
                time_str = "明天上午9点"
            elif "下午" in text:
                time_str = "今天下午3点"
            elif "晚上" in text or "晚上" in text:
                time_str = "今晚8点"

            # 提取内容
            content = "用户提醒"
            if "开会" in text:
                content = "开会"
            elif "吃药" in text:
                content = "吃药"
            elif "带伞" in text:
                content = "带伞"
            elif "体检" in text:
                content = "体检"
            elif "面试" in text:
                content = "面试"

            tool_calls.append(ToolCall(
                id="mock_reminder",
                name="set_reminder",
                arguments={"content": content, "time": time_str}
            ))

        # ========== 额外天气查询（用于复杂意图）==========
        if has_weather and has_reminder:
            # 复杂意图：提醒+天气，添加天气查询
            if detected_city:
                tool_calls.append(ToolCall(
                    id="mock_weather_extra",
                    name="get_weather",
                    arguments={"city": detected_city}
                ))
            else:
                tool_calls.append(ToolCall(
                    id="mock_weather_extra",
                    name="get_weather",
                    arguments={"city": "北京"}
                ))

        # ========== 音乐播放 ==========
        music_keywords = ["播放", "放", "听", "音乐", "歌曲", "歌"]
        is_music_request = any(kw in text for kw in music_keywords)

        if is_music_request and not tool_calls:
            # 提取歌曲/类型信息
            song_name = "随机歌曲"

            if "周杰伦" in text:
                song_name = "周杰伦的歌曲"
            elif "轻音乐" in text:
                song_name = "轻音乐"
            elif "流行" in text:
                song_name = "流行歌曲"
            elif "古典" in text:
                song_name = "古典音乐"
            elif "摇滚" in text:
                song_name = "摇滚音乐"
            elif "舒缓" in text:
                song_name = "舒缓音乐"
            elif "儿歌" in text:
                song_name = "儿歌"
            elif "开心" in text:
                song_name = "开心的歌曲"

            tool_calls.append(ToolCall(
                id="mock_music",
                name="play_music",
                arguments={"song_name": song_name}
            ))

        # ========== 设备控制 ==========
        device_keywords = {
            "灯": ["灯"],
            "空调": ["空调"],
            "电视": ["电视"],
            "风扇": ["风扇"],
            "窗帘": ["窗帘"],
            "热水器": ["热水器", "热水"],
            "加湿器": ["加湿器", "加湿"],
            "净化器": ["净化器", "空气净化器", "净化"],
            "暖气": ["暖气"],
            "地暖": ["地暖"],
        }

        action_keywords = {
            "打开": ["打开", "开", "启动", "开启"],
            "关闭": ["关闭", "关", "关掉", "关闭掉"],
            "调高": ["调高", "升高", "高一点", "调大", "大一点", "温度高"],
            "调低": ["调低", "降低", "低一点", "调小", "小一点", "温度低"],
        }

        detected_device = None
        detected_action = None

        # 首先检查上下文中的设备（用于多轮对话）
        if self.context.get("last_device"):
            # 检查是否是延续对话（如"把它关了"、"调到26度"）
            if any(kw in text for kw in ["它", "关", "调", "打开", "关闭"]):
                if "它" in text or "把" in text or "调" in text:
                    detected_device = self.context["last_device"]

        for device, keywords in device_keywords.items():
            if any(kw in text for kw in keywords):
                detected_device = device
                break

        for action, keywords in action_keywords.items():
            if any(kw in text for kw in keywords):
                detected_action = action
                break

        # 检测温度调节
        temp_match = re.search(r'(\d+)\s*度', text)
        if temp_match and detected_device in ["空调", "暖气", "地暖"]:
            temp_value = temp_match.group(1)
            if not detected_action:
                detected_action = "调到" + temp_value + "度"
            tool_calls.append(ToolCall(
                id="mock_device",
                name="control_device",
                arguments={"device": detected_device, "action": detected_action}
            ))
        elif detected_device and detected_action:
            tool_calls.append(ToolCall(
                id="mock_device",
                name="control_device",
                arguments={"device": detected_device, "action": detected_action}
            ))
        elif detected_device:
            # 有设备但没动作，根据上下文判断
            if "打开" in text or "开" in text:
                detected_action = "打开"
            elif "关" in text or "关闭" in text:
                detected_action = "关闭"
            else:
                detected_action = "打开"  # 默认动作

            tool_calls.append(ToolCall(
                id="mock_device",
                name="control_device",
                arguments={"device": detected_device, "action": detected_action}
            ))

        # ========== 生成响应 ==========
        prefix = self.EMOTION_PREFIXES.get(llm_input.emotion, "")

        # 检查是否是询问名字
        if "叫什么名字" in text or "我叫什么" in text or "我姓什么" in text:
            user_name = self.context.get("user_name", "")
            if user_name:
                if "姓" in text:
                    # 提取姓氏
                    surname = user_name[0] if len(user_name) > 0 else user_name
                    response_text = f"{prefix}您姓{surname}，叫{user_name}。"
                else:
                    response_text = f"{prefix}您叫{user_name}。"
            else:
                response_text = f"{prefix}抱歉，我不记得您的名字了。您能再告诉我一次吗？"
            return LLMResponse(
                text=response_text,
                tool_calls=[],
                emotion_adapted=True
            )

        if tool_calls:
            response_text = f"{prefix}好的，我来帮您处理。"
        else:
            # 处理特殊问候
            if "你能做什么" in text or "能做什么" in text:
                response_text = f"{prefix}我可以帮您查询天气、设置提醒、播放音乐、控制智能设备等。您想让我做什么？"
            elif "晚安" in text:
                response_text = f"{prefix}晚安，祝您有个好梦！"
            elif "再见" in text or "拜拜" in text:
                response_text = f"{prefix}再见，期待下次与您聊天！"
            else:
                response_text = f"{prefix}您好！我是您的智能语音助手，有什么可以帮您的？"

        return LLMResponse(
            text=response_text,
            tool_calls=tool_calls,
            emotion_adapted=True
        )

    def _mock_summarize(self, tool_results: List[ToolResult]) -> str:
        """模拟总结 - 返回自然语言响应，支持多工具调用"""
        if not tool_results:
            return "操作已完成。"

        # 处理多个工具调用
        if len(tool_results) > 1:
            responses = []
            seen_tools = set()  # 避免重复

            for result in tool_results:
                tool_name = result.tool_call.name
                tool_key = f"{tool_name}_{id(result.tool_call)}"

                if tool_name == "get_current_time" and "get_current_time" not in seen_tools:
                    seen_tools.add("get_current_time")
                    data = result.result
                    if data.get("success"):
                        responses.append(data.get("description", "已获取当前时间"))
                    else:
                        responses.append("获取时间失败")

                elif tool_name == "get_weather" and "get_weather" not in seen_tools:
                    seen_tools.add("get_weather")
                    data = result.result
                    city = data.get("city", "您查询的城市")
                    weather = data.get("weather", "未知")
                    temp = data.get("temperature", "?")
                    humidity = data.get("humidity", "?")
                    responses.append(f"{city}今天{weather}，气温{temp}度，湿度{humidity}%")

                elif tool_name == "set_reminder" and "set_reminder" not in seen_tools:
                    seen_tools.add("set_reminder")
                    data = result.result
                    time = data.get("time", "")
                    content = data.get("content", "")
                    if content:
                        responses.append(f"已为您设置提醒：{time} - {content}")
                    else:
                        responses.append(f"已为您设置提醒，时间是{time}")

                elif tool_name == "play_music" and "play_music" not in seen_tools:
                    seen_tools.add("play_music")
                    data = result.result
                    song = data.get("song", "音乐")
                    responses.append(f"正在为您播放{song}")

                elif tool_name == "control_device" and "control_device" not in seen_tools:
                    seen_tools.add("control_device")
                    data = result.result
                    device = data.get("device", "设备")
                    action = data.get("action", "操作")
                    responses.append(f"已为您{action}{device}")

            if responses:
                return "好的，" + "。".join(responses) + "。"
            return "操作已完成。"

        # 单个工具调用
        result = tool_results[0]
        tool_name = result.tool_call.name

        if tool_name == "get_current_time":
            data = result.result
            if data.get("success"):
                return data.get("description", "已获取当前时间")
            return "获取时间失败"

        elif tool_name == "get_weather":
            data = result.result
            city = data.get("city", "您查询的城市")
            weather = data.get("weather", "未知")
            temp = data.get("temperature", "?")
            humidity = data.get("humidity", "?")
            return f"{city}今天{weather}，气温{temp}度，湿度{humidity}%。"

        elif tool_name == "set_reminder":
            data = result.result
            time = data.get("time", "")
            content = data.get("content", "")
            if content:
                return f"好的，已经为您设置了提醒：{time} - {content}。"
            return f"好的，已经为您设置了提醒，时间是{time}。"

        elif tool_name == "play_music":
            data = result.result
            song = data.get("song", "音乐")
            return f"好的，正在为您播放{song}。"

        elif tool_name == "control_device":
            data = result.result
            device = data.get("device", "设备")
            action = data.get("action", "操作")
            return f"好的，已为您{action}{device}。"

        elif tool_name == "search_web":
            data = result.result
            results = data.get("results", [])
            if results:
                summary = f"关于「{data.get('query', '')}」找到{len(results)}条结果："
                for i, r in enumerate(results[:2]):
                    summary += f"\n{i+1}. {r.get('title', '')}"
                return summary
            return f"未找到关于「{data.get('query', '')}」的相关结果"

        else:
            return "好的，已完成您的请求。"

    def add_to_history(self, message: Message):
        """添加消息到历史"""
        self.conversation_history.append(message)

        # 更新上下文
        if message.role == "user":
            # 记住用户名字 - 更精确的匹配
            if "我叫" in message.content and "?" not in message.content and "什么" not in message.content and "呢" not in message.content:
                # 提取名字，排除询问语句
                import re
                match = re.search(r'我叫([^\s，。！？]+)', message.content)
                if match:
                    name = match.group(1).strip()
                    if name and len(name) <= 10:  # 限制名字长度
                        self.context["user_name"] = name

            # 记住提到的城市
            for city in CHINESE_CITIES:
                if city in message.content:
                    self.context["last_city"] = city
                    break

            # 记住提到的设备
            for device in ["灯", "空调", "电视", "风扇", "窗帘", "热水器", "加湿器", "净化器"]:
                if device in message.content:
                    self.context["last_device"] = device
                    break

        # 限制历史长度
        max_history = get_config().system.get("conversation", {}).get("max_history", 10)
        if len(self.conversation_history) > max_history:
            self.conversation_history = self.conversation_history[-max_history:]

    def clear_history(self):
        """清除历史"""
        self.conversation_history.clear()
        self.context.clear()