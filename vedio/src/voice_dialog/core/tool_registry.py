"""
全双工语音对话系统 v2.0 - 工具注册中心
支持动态注册MCP工具和SKILLS，实现开放解耦架构
"""
import asyncio
import json
from typing import Dict, List, Any, Callable, Optional, Awaitable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from ..core.logger import logger


@dataclass
class ToolDefinition:
    """工具定义"""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema格式
    handler: Callable  # 同步或异步处理函数
    category: str = "general"  # 工具分类
    tags: List[str] = field(default_factory=list)  # 标签用于搜索
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据

    def to_openai_tool(self) -> Dict:
        """转换为OpenAI工具格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }

    def to_mcp_tool(self) -> Dict:
        """转换为MCP工具格式"""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.parameters
        }


class ToolProvider(ABC):
    """工具提供者抽象基类 - 支持MCP和SKILLS"""

    @abstractmethod
    async def list_tools(self) -> List[ToolDefinition]:
        """列出可用工具"""
        pass

    @abstractmethod
    async def execute_tool(self, name: str, arguments: Dict) -> Any:
        """执行工具"""
        pass

    @abstractmethod
    async def get_tool_definition(self, name: str) -> Optional[ToolDefinition]:
        """获取工具定义"""
        pass


class ToolRegistry:
    """
    工具注册中心
    支持动态注册、MCP协议、SKILLS扩展
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools: Dict[str, ToolDefinition] = {}
            cls._instance._providers: List[ToolProvider] = []
            cls._instance._categories: Dict[str, List[str]] = {}
        return cls._instance

    def register_tool(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable,
        category: str = "general",
        tags: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> None:
        """
        注册工具

        Args:
            name: 工具名称
            description: 工具描述
            parameters: 参数定义(JSON Schema)
            handler: 处理函数(同步或异步)
            category: 分类
            tags: 标签
            metadata: 元数据
        """
        tool = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
            category=category,
            tags=tags or [],
            metadata=metadata or {}
        )

        self._tools[name] = tool

        # 更新分类索引
        if category not in self._categories:
            self._categories[category] = []
        self._categories[category].append(name)

        logger.info(f"注册工具: {name} (分类: {category})")

    def unregister_tool(self, name: str) -> bool:
        """注销工具"""
        if name in self._tools:
            tool = self._tools[name]
            # 从分类中移除
            if tool.category in self._categories:
                self._categories[tool.category].remove(name)
            del self._tools[name]
            logger.info(f"注销工具: {name}")
            return True
        return False

    def register_provider(self, provider: ToolProvider) -> None:
        """注册工具提供者(MCP/SKILLS)"""
        self._providers.append(provider)
        logger.info(f"注册工具提供者: {provider.__class__.__name__}")

    async def get_all_tools(self) -> List[ToolDefinition]:
        """获取所有工具(包括提供者的工具)"""
        tools = list(self._tools.values())

        # 从提供者获取工具
        for provider in self._providers:
            try:
                provider_tools = await provider.list_tools()
                for tool in provider_tools:
                    if tool.name not in self._tools:
                        tools.append(tool)
            except Exception as e:
                logger.error(f"获取提供者工具失败: {e}")

        return tools

    async def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """获取工具定义"""
        # 先查本地注册
        if name in self._tools:
            return self._tools[name]

        # 再查提供者
        for provider in self._providers:
            try:
                tool = await provider.get_tool_definition(name)
                if tool:
                    return tool
            except Exception as e:
                logger.error(f"获取工具定义失败: {e}")

        return None

    async def execute_tool(self, name: str, arguments: Dict) -> Any:
        """执行工具"""
        # 先查本地注册
        if name in self._tools:
            tool = self._tools[name]
            try:
                if asyncio.iscoroutinefunction(tool.handler):
                    return await tool.handler(**arguments)
                else:
                    return tool.handler(**arguments)
            except Exception as e:
                logger.error(f"工具执行失败 {name}: {e}")
                raise

        # 再查提供者
        for provider in self._providers:
            try:
                result = await provider.execute_tool(name, arguments)
                return result
            except Exception as e:
                logger.error(f"提供者执行失败 {name}: {e}")
                continue

        raise ValueError(f"未知工具: {name}")

    def get_tools_by_category(self, category: str) -> List[ToolDefinition]:
        """按分类获取工具"""
        tool_names = self._categories.get(category, [])
        return [self._tools[name] for name in tool_names if name in self._tools]

    def get_tools_by_tags(self, tags: List[str]) -> List[ToolDefinition]:
        """按标签搜索工具"""
        result = []
        for tool in self._tools.values():
            if any(tag in tool.tags for tag in tags):
                result.append(tool)
        return result

    def get_openai_tools(self) -> List[Dict]:
        """获取OpenAI格式的工具列表"""
        return [tool.to_openai_tool() for tool in self._tools.values()]

    def get_tool_schemas_for_prompt(self) -> str:
        """生成工具描述用于LLM提示词"""
        descriptions = []
        for tool in self._tools.values():
            desc = f"- {tool.name}: {tool.description}"
            if tool.parameters.get("properties"):
                params = ", ".join(tool.parameters["properties"].keys())
                desc += f" (参数: {params})"
            descriptions.append(desc)
        return "\n".join(descriptions)


# 全局工具注册中心实例
tool_registry = ToolRegistry()


class MCPToolProvider(ToolProvider):
    """MCP协议工具提供者"""

    def __init__(self, server_url: str, server_name: str = "mcp"):
        self.server_url = server_url
        self.server_name = server_name
        self._tools: Dict[str, ToolDefinition] = {}
        self._connected = False

    async def connect(self) -> bool:
        """连接MCP服务器"""
        try:
            # 这里应该实现实际的MCP连接逻辑
            # 目前使用模拟连接
            self._connected = True
            logger.info(f"MCP服务器已连接: {self.server_url}")
            return True
        except Exception as e:
            logger.error(f"MCP连接失败: {e}")
            return False

    async def disconnect(self):
        """断开连接"""
        self._connected = False

    async def list_tools(self) -> List[ToolDefinition]:
        """列出MCP工具"""
        if not self._connected:
            return []
        # 实际实现应该调用MCP协议
        return list(self._tools.values())

    async def execute_tool(self, name: str, arguments: Dict) -> Any:
        """执行MCP工具"""
        if not self._connected:
            raise ConnectionError("MCP服务器未连接")

        # 实际实现应该调用MCP协议
        logger.info(f"MCP工具执行: {name}({arguments})")
        return {"result": f"MCP工具 {name} 执行成功", "arguments": arguments}

    async def get_tool_definition(self, name: str) -> Optional[ToolDefinition]:
        """获取MCP工具定义"""
        return self._tools.get(name)

    def add_mcp_tool(self, tool_def: Dict):
        """添加MCP工具定义"""
        tool = ToolDefinition(
            name=tool_def["name"],
            description=tool_def.get("description", ""),
            parameters=tool_def.get("inputSchema", {"type": "object", "properties": {}}),
            handler=lambda **args: None,  # MCP工具通过execute_tool执行
            category="mcp",
            tags=["mcp", self.server_name],
            metadata={"mcp_server": self.server_url}
        )
        self._tools[tool.name] = tool


class SKILLSProvider(ToolProvider):
    """SKILLS工具提供者"""

    def __init__(self, name: str = "skills"):
        self.name = name
        self._skills: Dict[str, ToolDefinition] = {}

    def register_skill(
        self,
        name: str,
        description: str,
        handler: Callable,
        parameters: Dict[str, Any] = None,
        tags: List[str] = None
    ):
        """注册技能"""
        skill = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters or {"type": "object", "properties": {}},
            handler=handler,
            category="skill",
            tags=tags or ["skill"]
        )
        self._skills[name] = skill
        logger.info(f"注册技能: {name}")

    async def list_tools(self) -> List[ToolDefinition]:
        """列出所有技能"""
        return list(self._skills.values())

    async def execute_tool(self, name: str, arguments: Dict) -> Any:
        """执行技能"""
        if name not in self._skills:
            raise ValueError(f"未知技能: {name}")

        skill = self._skills[name]
        if asyncio.iscoroutinefunction(skill.handler):
            return await skill.handler(**arguments)
        else:
            return skill.handler(**arguments)

    async def get_tool_definition(self, name: str) -> Optional[ToolDefinition]:
        """获取技能定义"""
        return self._skills.get(name)


# 内置工具注册函数
def register_builtin_tools():
    """注册内置工具"""
    from ..core.config import get_config

    # 天气查询工具
    @tool_registry.register_tool(
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
        category="query",
        tags=["天气", "查询", "生活"]
    )
    async def get_weather(city: str) -> Dict:
        # 模拟天气数据
        weather_data = {
            "北京": {"temp": 15, "weather": "晴", "humidity": 45},
            "上海": {"temp": 18, "weather": "多云", "humidity": 65},
            "广州": {"temp": 25, "weather": "阴", "humidity": 80},
        }
        data = weather_data.get(city, {"temp": 20, "weather": "晴", "humidity": 50})
        return {
            "city": city,
            "temperature": data["temp"],
            "weather": data["weather"],
            "humidity": data["humidity"]
        }

    # 设置提醒工具
    @tool_registry.register_tool(
        name="set_reminder",
        description="设置提醒或闹钟，在指定时间提醒用户",
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
        category="productivity",
        tags=["提醒", "闹钟", "时间"]
    )
    async def set_reminder(content: str, time: str) -> Dict:
        from datetime import datetime
        return {
            "success": True,
            "reminder_id": f"rem_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "content": content,
            "time": time
        }

    # 播放音乐工具
    @tool_registry.register_tool(
        name="play_music",
        description="播放音乐或歌曲",
        parameters={
            "type": "object",
            "properties": {
                "song_name": {
                    "type": "string",
                    "description": "歌曲名称或类型"
                },
                "artist": {
                    "type": "string",
                    "description": "歌手名称（可选）"
                }
            },
            "required": ["song_name"]
        },
        category="entertainment",
        tags=["音乐", "播放", "娱乐"]
    )
    async def play_music(song_name: str, artist: str = "") -> Dict:
        song_info = f"{artist}的{song_name}" if artist else song_name
        return {
            "success": True,
            "song": song_info,
            "status": "playing"
        }

    # 设备控制工具
    @tool_registry.register_tool(
        name="control_device",
        description="控制智能设备，如灯、空调、电视、风扇、窗帘等",
        parameters={
            "type": "object",
            "properties": {
                "device": {
                    "type": "string",
                    "description": "设备名称，如：灯、空调、电视、风扇、窗帘等"
                },
                "action": {
                    "type": "string",
                    "description": "操作动作，如：打开、关闭、调高、调低等"
                }
            },
            "required": ["device", "action"]
        },
        category="smart_home",
        tags=["设备", "控制", "智能家居"]
    )
    async def control_device(device: str, action: str) -> Dict:
        return {
            "success": True,
            "device": device,
            "action": action,
            "status": "completed"
        }

    # 网络搜索工具
    @tool_registry.register_tool(
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
        category="query",
        tags=["搜索", "网络", "查询"]
    )
    async def search_web(query: str) -> Dict:
        return {
            "query": query,
            "results": [
                {"title": f"关于{query}的信息", "snippet": f"这是关于{query}的详细内容..."}
            ]
        }

    logger.info("内置工具注册完成")