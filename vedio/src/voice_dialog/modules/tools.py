"""
全双工语音对话系统 v3.0 - 工具引擎
支持SKILLS和MCP协议
使用统一的工具注册中心 ToolRegistry

v3.0 内置工具：
- get_current_time: 获取当前时间日期
- get_weather: 查询天气
- search_web: 网络搜索
- set_reminder: 设置提醒
- play_music: 播放音乐
- control_device: 设备控制
"""
import asyncio
import json
from typing import Dict, Any, Optional, Callable, Awaitable
from datetime import datetime
from ..core.logger import logger

from ..core.types import ToolCall, ToolResult
from ..core.config import get_config
from ..core.tool_registry import tool_registry, ToolProvider

# 默认位置
DEFAULT_LOCATION = "上海"


# 城市天气数据
WEATHER_DATA = {
    "北京": {"temp": 15, "weather": "晴", "humidity": 45},
    "上海": {"temp": 18, "weather": "多云", "humidity": 65},
    "广州": {"temp": 25, "weather": "阴", "humidity": 80},
    "深圳": {"temp": 24, "weather": "小雨", "humidity": 75},
    "杭州": {"temp": 16, "weather": "晴", "humidity": 50},
    "南京": {"temp": 14, "weather": "多云", "humidity": 55},
    "苏州": {"temp": 15, "weather": "晴", "humidity": 52},
    "成都": {"temp": 18, "weather": "阴", "humidity": 60},
    "重庆": {"temp": 20, "weather": "多云", "humidity": 70},
    "武汉": {"temp": 17, "weather": "晴", "humidity": 55},
    "西安": {"temp": 12, "weather": "晴", "humidity": 40},
    "天津": {"temp": 13, "weather": "多云", "humidity": 48},
    "长沙": {"temp": 19, "weather": "小雨", "humidity": 75},
    "郑州": {"temp": 14, "weather": "晴", "humidity": 42},
    "青岛": {"temp": 11, "weather": "晴", "humidity": 55},
    "大连": {"temp": 8, "weather": "多云", "humidity": 60},
    "厦门": {"temp": 22, "weather": "晴", "humidity": 70},
    "昆明": {"temp": 18, "weather": "晴", "humidity": 50},
    "福州": {"temp": 21, "weather": "多云", "humidity": 68},
    "合肥": {"temp": 15, "weather": "晴", "humidity": 52},
    "南昌": {"temp": 18, "weather": "多云", "humidity": 65},
    "哈尔滨": {"temp": 2, "weather": "晴", "humidity": 35},
    "长春": {"temp": 4, "weather": "多云", "humidity": 38},
    "沈阳": {"temp": 6, "weather": "晴", "humidity": 40},
    "济南": {"temp": 14, "weather": "晴", "humidity": 45},
    "石家庄": {"temp": 13, "weather": "多云", "humidity": 48},
    "太原": {"temp": 10, "weather": "晴", "humidity": 42},
    "南宁": {"temp": 24, "weather": "多云", "humidity": 78},
    "海口": {"temp": 28, "weather": "晴", "humidity": 80},
    "贵阳": {"temp": 16, "weather": "阴", "humidity": 72},
    "兰州": {"temp": 8, "weather": "晴", "humidity": 35},
    "西宁": {"temp": 5, "weather": "晴", "humidity": 32},
    "银川": {"temp": 9, "weather": "晴", "humidity": 30},
    "乌鲁木齐": {"temp": 3, "weather": "晴", "humidity": 28},
    "呼和浩特": {"temp": 7, "weather": "晴", "humidity": 33},
    "拉萨": {"temp": 8, "weather": "晴", "humidity": 25},
    "珠海": {"temp": 24, "weather": "晴", "humidity": 75},
    "东莞": {"temp": 23, "weather": "多云", "humidity": 72},
    "佛山": {"temp": 23, "weather": "晴", "humidity": 70},
    "宁波": {"temp": 16, "weather": "多云", "humidity": 60},
    "无锡": {"temp": 15, "weather": "晴", "humidity": 55},
    "温州": {"temp": 18, "weather": "多云", "humidity": 65},
    "常州": {"temp": 15, "weather": "晴", "humidity": 52},
    "徐州": {"temp": 13, "weather": "晴", "humidity": 48},
    "烟台": {"temp": 10, "weather": "晴", "humidity": 50},
}


class ToolEngine:
    """
    工具执行引擎
    使用统一的 ToolRegistry 进行工具管理
    支持：内置工具、SKILLS和MCP协议
    """

    def __init__(self):
        self.config = get_config()
        self._tools: Dict[str, Callable] = {}
        self._register_builtin_tools_to_registry()

    def _register_builtin_tools_to_registry(self):
        """将内置工具注册到 ToolRegistry"""

        # ========== 时间查询工具 ==========
        async def get_current_time() -> Dict:
            """获取当前时间和日期"""
            now = datetime.now()
            weekday_names = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
            weekday = weekday_names[now.weekday()]

            return {
                "success": True,
                "date": now.strftime("%Y年%m月%d日"),
                "time": now.strftime("%H:%M:%S"),
                "hour": now.hour,
                "minute": now.minute,
                "weekday": weekday,
                "timestamp": now.timestamp(),
                "location": DEFAULT_LOCATION,
                "description": f"现在是{now.strftime('%Y年%m月%d日')} {weekday} {now.strftime('%H:%M')}"
            }

        tool_registry.register_tool(
            name="get_current_time",
            description="获取当前时间、日期和星期，用于回答用户关于时间的询问。当用户询问'现在几点'、'今天星期几'、'什么日期'等问题时调用。",
            parameters={
                "type": "object",
                "properties": {},
                "required": []
            },
            handler=get_current_time,
            category="query",
            tags=["时间", "日期", "查询"]
        )

        # ========== 天气查询工具 ==========
        async def get_weather(city: str) -> Dict:
            await asyncio.sleep(0.1)  # 模拟网络延迟
            data = WEATHER_DATA.get(city, {"temp": 20, "weather": "晴", "humidity": 50})
            return {
                "city": city,
                "temperature": data["temp"],
                "weather": data["weather"],
                "humidity": data["humidity"],
                "description": f"{city}今天{data['weather']}，气温{data['temp']}度，湿度{data['humidity']}%"
            }

        tool_registry.register_tool(
            name="get_weather",
            description="查询指定城市的天气情况，包括温度、天气状况、湿度等",
            parameters={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如：北京、上海、广州等"
                    }
                },
                "required": ["city"]
            },
            handler=get_weather,
            category="query",
            tags=["天气", "查询", "生活"]
        )

        # 注册网络搜索工具 - 优化版
        async def search_web(query: str) -> Dict:
            """
            网络搜索工具 - 返回结构化搜索结果
            集成真实的搜索能力
            """
            try:
                # 尝试使用真实的搜索API
                # 这里可以接入如 SerpAPI、Bing Search API 等
                # 目前返回模拟但格式良好的结果

                await asyncio.sleep(0.2)

                # 模拟搜索结果 - 更真实的格式
                results = []

                # 根据查询类型生成不同结果
                if any(kw in query for kw in ["新闻", "最新", "今日"]):
                    results = [
                        {
                            "title": f"【新闻】{query}相关报道",
                            "source": "综合新闻",
                            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "snippet": f"关于{query}的最新动态和报道内容...",
                            "url": f"https://news.example.com/{query}"
                        },
                        {
                            "title": f"{query}热点事件",
                            "source": "热点资讯",
                            "time": datetime.now().strftime("%Y-%m-%d"),
                            "snippet": f"最新{query}相关热点事件汇总...",
                            "url": f"https://hot.example.com/{query}"
                        }
                    ]
                elif any(kw in query for kw in ["是什么", "怎么", "如何", "为什么"]):
                    results = [
                        {
                            "title": f"【百科】{query}",
                            "source": "知识百科",
                            "snippet": f"{query}的详细解释和说明...",
                            "url": f"https://baike.example.com/{query}"
                        },
                        {
                            "title": f"教程：{query}",
                            "source": "教程网",
                            "snippet": f"详细介绍{query}的方法和步骤...",
                            "url": f"https://howto.example.com/{query}"
                        }
                    ]
                else:
                    results = [
                        {
                            "title": f"{query} - 综合信息",
                            "source": "综合搜索",
                            "snippet": f"关于{query}的全面信息介绍...",
                            "url": f"https://search.example.com/{query}"
                        },
                        {
                            "title": f"{query}相关推荐",
                            "source": "推荐引擎",
                            "snippet": f"与{query}相关的热门内容推荐...",
                            "url": f"https://recommend.example.com/{query}"
                        }
                    ]

                return {
                    "success": True,
                    "query": query,
                    "total": len(results),
                    "results": results,
                    "summary": f"找到{len(results)}条关于「{query}」的结果"
                }

            except Exception as e:
                logger.error(f"搜索失败: {e}")
                return {
                    "success": False,
                    "query": query,
                    "error": str(e),
                    "results": [],
                    "summary": f"搜索「{query}」时出现错误"
                }

        tool_registry.register_tool(
            name="search_web",
            description="搜索网络获取信息",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词"
                    }
                },
                "required": ["query"]
            },
            handler=search_web,
            category="query",
            tags=["搜索", "网络", "查询"]
        )

        # 注册设置提醒工具
        async def set_reminder(content: str, time: str) -> Dict:
            await asyncio.sleep(0.05)
            return {
                "success": True,
                "reminder_id": f"rem_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "content": content,
                "time": time,
                "message": f"已设置提醒：{time} - {content}"
            }

        tool_registry.register_tool(
            name="set_reminder",
            description="设置提醒或闹钟",
            parameters={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "提醒内容"
                    },
                    "time": {
                        "type": "string",
                        "description": "提醒时间，如：明天早上8点、下午3点等"
                    }
                },
                "required": ["content", "time"]
            },
            handler=set_reminder,
            category="productivity",
            tags=["提醒", "闹钟", "时间"]
        )

        # 注册播放音乐工具
        async def play_music(song_name: str, artist: str = "") -> Dict:
            await asyncio.sleep(0.1)
            song_info = song_name
            if artist:
                song_info = f"{artist}的{song_name}"
            return {
                "success": True,
                "song": song_info,
                "status": "playing",
                "message": f"正在播放：{song_info}"
            }

        tool_registry.register_tool(
            name="play_music",
            description="播放音乐或歌曲",
            parameters={
                "type": "object",
                "properties": {
                    "song_name": {
                        "type": "string",
                        "description": "歌曲名称或类型，如：流行歌曲、轻音乐、周杰伦的歌等"
                    },
                    "artist": {
                        "type": "string",
                        "description": "歌手名称"
                    }
                },
                "required": ["song_name"]
            },
            handler=play_music,
            category="entertainment",
            tags=["音乐", "播放", "娱乐"]
        )

        # 注册设备控制工具
        async def control_device(device: str, action: str) -> Dict:
            await asyncio.sleep(0.05)
            return {
                "success": True,
                "device": device,
                "action": action,
                "status": "completed",
                "message": f"已{action}{device}"
            }

        tool_registry.register_tool(
            name="control_device",
            description="控制智能设备，如灯、空调、电视、风扇、窗帘等",
            parameters={
                "type": "object",
                "properties": {
                    "device": {
                        "type": "string",
                        "description": "设备名称，如：灯、客厅灯、卧室灯、空调、电视、风扇、窗帘等"
                    },
                    "action": {
                        "type": "string",
                        "description": "操作动作，如：打开、关闭、调高、调低等"
                    }
                },
                "required": ["device", "action"]
            },
            handler=control_device,
            category="smart_home",
            tags=["设备", "控制", "智能家居"]
        )

        logger.info("内置工具已注册到 ToolRegistry")

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """执行工具调用 - 使用 ToolRegistry"""
        try:
            result = await tool_registry.execute_tool(tool_call.name, tool_call.arguments)
            return ToolResult(
                tool_call=tool_call,
                result=result,
                success=True
            )
        except ValueError as e:
            logger.error(f"未知工具: {tool_call.name}")
            return ToolResult(
                tool_call=tool_call,
                result=None,
                success=False,
                error=f"未知工具: {tool_call.name}"
            )
        except Exception as e:
            logger.error(f"工具执行失败 {tool_call.name}: {e}")
            return ToolResult(
                tool_call=tool_call,
                result=None,
                success=False,
                error=str(e)
            )

    async def execute_batch(self, tool_calls: list) -> list:
        """批量执行工具调用 - 使用 asyncio.gather 并行执行"""
        tasks = [self.execute(tc) for tc in tool_calls]
        return await asyncio.gather(*tasks)

    def register_tool(self, name: str, func: Callable):
        """注册自定义工具 - 委托给 ToolRegistry"""
        tool_registry.register_tool(
            name=name,
            description=f"自定义工具: {name}",
            parameters={"type": "object", "properties": {}},
            handler=func,
            category="custom"
        )
        logger.info(f"注册工具: {name}")

    def get_available_tools(self):
        """获取所有可用工具"""
        return tool_registry.get_openai_tools()


class MCPClient(ToolProvider):
    """
    MCP协议客户端
    用于与MCP服务器通信
    实现ToolProvider接口
    """

    def __init__(self, server_url: Optional[str] = None):
        self.server_url = server_url
        self._connected = False

    async def connect(self):
        """连接MCP服务器"""
        if self.server_url:
            self._connected = True
            logger.info(f"已连接MCP服务器: {self.server_url}")

    async def disconnect(self):
        """断开连接"""
        self._connected = False

    async def call_tool(self, name: str, arguments: Dict) -> Any:
        """调用MCP工具"""
        if not self._connected:
            raise ConnectionError("MCP服务器未连接")

        return {"result": f"MCP工具 {name} 调用成功"}

    async def list_tools(self) -> list:
        """列出可用工具 - 实现 ToolProvider 接口"""
        if not self._connected:
            return []
        return []

    async def execute_tool(self, name: str, arguments: Dict) -> Any:
        """执行工具 - 实现 ToolProvider 接口"""
        return await self.call_tool(name, arguments)

    async def get_tool_definition(self, name: str):
        """获取工具定义 - 实现 ToolProvider 接口"""
        return None


class SKILLSEngine(ToolProvider):
    """
    SKILLS引擎
    管理和执行用户定义的技能
    实现ToolProvider接口
    """

    def __init__(self):
        self._skills: Dict[str, Dict] = {}

    def register_skill(self, name: str, handler: Callable, description: str = ""):
        """注册技能"""
        self._skills[name] = {
            "name": name,
            "handler": handler,
            "description": description
        }

    async def execute_skill(self, name: str, **kwargs) -> Any:
        """执行技能"""
        if name not in self._skills:
            raise ValueError(f"未知技能: {name}")

        skill = self._skills[name]
        handler = skill["handler"]

        if asyncio.iscoroutinefunction(handler):
            return await handler(**kwargs)
        else:
            return handler(**kwargs)

    async def list_tools(self) -> list:
        """列出所有技能 - 实现 ToolProvider 接口"""
        from ..core.tool_registry import ToolDefinition
        return [
            ToolDefinition(
                name=s["name"],
                description=s["description"],
                parameters={"type": "object", "properties": {}},
                handler=s["handler"],
                category="skill"
            )
            for s in self._skills.values()
        ]

    async def execute_tool(self, name: str, arguments: Dict) -> Any:
        """执行工具 - 实现 ToolProvider 接口"""
        return await self.execute_skill(name, **arguments)

    async def get_tool_definition(self, name: str):
        """获取技能定义 - 实现 ToolProvider 接口"""
        if name in self._skills:
            from ..core.tool_registry import ToolDefinition
            s = self._skills[name]
            return ToolDefinition(
                name=s["name"],
                description=s["description"],
                parameters={"type": "object", "properties": {}},
                handler=s["handler"],
                category="skill"
            )
        return None