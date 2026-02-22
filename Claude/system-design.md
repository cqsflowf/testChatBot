# 智能体应用系统设计文档

## 1. 文档信息

| 项目 | 内容 |
|------|------|
| 项目名称 | 智能对话助手 |
| 文档版本 | v1.0 |
| 创建日期 | 2026-02-22 |
| 关联文档 | architecture-design.md |

---

## 2. 项目结构

```
chatbot/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI 入口
│   ├── config.py               # 配置管理
│   │
│   ├── api/                    # API 路由层
│   │   ├── __init__.py
│   │   ├── v1/
│   │   │   ├── chat.py         # 对话接口
│   │   │   ├── session.py      # 会话管理
│   │   │   ├── user.py         # 用户接口
│   │   │   └── upload.py       # 文件上传
│   │   └── deps.py             # 依赖注入
│   │
│   ├── core/                   # 核心业务逻辑
│   │   ├── __init__.py
│   │   ├── orchestrator.py     # 对话编排器
│   │   ├── router.py           # 意图路由
│   │   ├── merger.py           # 响应融合
│   │   └── streamer.py         # 流式输出
│   │
│   ├── models/                 # 模型服务
│   │   ├── __init__.py
│   │   ├── base.py             # 模型基类
│   │   ├── talker.py           # Talker 服务
│   │   ├── thinker.py          # Thinker 服务
│   │   └── vision.py           # 视觉模型服务
│   │
│   ├── tools/                  # 工具服务
│   │   ├── __init__.py
│   │   ├── base.py             # 工具基类
│   │   ├── weather.py          # 天气查询
│   │   ├── search.py           # 联网搜索
│   │   ├── hotel.py            # 酒店查询
│   │   ├── flight.py           # 机票查询
│   │   └── attraction.py       # 景点推荐
│   │
│   ├── schemas/                # Pydantic 模型
│   │   ├── __init__.py
│   │   ├── chat.py             # 对话相关
│   │   ├── session.py          # 会话相关
│   │   └── user.py             # 用户相关
│   │
│   ├── db/                     # 数据库
│   │   ├── __init__.py
│   │   ├── database.py         # 数据库连接
│   │   ├── models.py           # ORM 模型
│   │   └── crud.py             # CRUD 操作
│   │
│   └── utils/                  # 工具函数
│       ├── __init__.py
│       ├── cache.py            # 缓存工具
│       ├── security.py         # 安全工具
│       └── logger.py           # 日志工具
│
├── tests/                      # 测试
├── scripts/                    # 脚本
├── docker/                     # Docker 配置
├── docs/                       # 文档
├── requirements.txt
├── docker-compose.yml
└── README.md
```

---

## 3. API 接口设计

### 3.1 对话接口

#### 3.1.1 发送消息（流式）

```
POST /api/v1/chat/completions
Content-Type: application/json
Authorization: Bearer {token}

Request:
{
    "session_id": "uuid",           // 会话ID，可选，不传则创建新会话
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "北京今天天气怎么样？"},
                {"type": "image_url", "image_url": {"url": "base64://..."}}  // 可选
            ]
        }
    ],
    "stream": true,                 // 是否流式输出
    "tools": ["weather", "search"]  // 可选，指定可用工具
}

Response (SSE Stream):
event: message_start
data: {"session_id": "uuid", "message_id": "uuid"}

event: content_delta
data: {"type": "talker", "delta": "让我"}

event: content_delta
data: {"type": "talker", "delta": "帮你查一下"}

event: tool_call
data: {"tool": "weather", "status": "calling", "args": {"city": "北京"}}

event: tool_result
data: {"tool": "weather", "status": "success", "result": {...}}

event: content_delta
data: {"type": "thinker", "delta": "北京今天天气晴朗..."}

event: message_end
data: {"usage": {"prompt_tokens": 100, "completion_tokens": 200}}
```

#### 3.1.2 获取对话历史

```
GET /api/v1/chat/history/{session_id}?limit=20&offset=0
Authorization: Bearer {token}

Response:
{
    "session_id": "uuid",
    "messages": [
        {
            "id": "uuid",
            "role": "user",
            "content": "...",
            "created_at": "2026-02-22T10:00:00Z"
        },
        {
            "id": "uuid",
            "role": "assistant",
            "content": "...",
            "tool_calls": [...],
            "created_at": "2026-02-22T10:00:01Z"
        }
    ],
    "total": 100,
    "has_more": true
}
```

### 3.2 会话管理接口

```
# 创建会话
POST /api/v1/sessions
Request: {"title": "新对话"}
Response: {"session_id": "uuid", "title": "新对话", "created_at": "..."}

# 获取会话列表
GET /api/v1/sessions?limit=20&offset=0
Response: {"sessions": [...], "total": 50}

# 删除会话
DELETE /api/v1/sessions/{session_id}
Response: {"success": true}

# 更新会话标题
PATCH /api/v1/sessions/{session_id}
Request: {"title": "新标题"}
Response: {"session_id": "uuid", "title": "新标题"}
```

### 3.3 文件上传接口

```
POST /api/v1/upload/image
Content-Type: multipart/form-data
Authorization: Bearer {token}

Request:
- file: (binary)
- session_id: "uuid" (optional)

Response:
{
    "file_id": "uuid",
    "url": "https://storage.example.com/images/xxx.jpg",
    "thumbnail_url": "https://storage.example.com/thumbnails/xxx.jpg",
    "mime_type": "image/jpeg",
    "size": 102400
}
```

---

## 4. 数据库设计

### 4.1 ER 图

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│     users       │       │    sessions     │       │    messages     │
├─────────────────┤       ├─────────────────┤       ├─────────────────┤
│ id (PK)         │──┐    │ id (PK)         │──┐    │ id (PK)         │
│ username        │  │    │ user_id (FK)    │◄─┘    │ session_id (FK) │◄─┐
│ email           │  │    │ title           │  │    │ role            │  │
│ password_hash   │  └───►│ created_at      │  │    │ content         │  │
│ avatar_url      │       │ updated_at      │  │    │ content_type    │  │
│ created_at      │       │ is_deleted      │  └───►│ tool_calls      │  │
│ updated_at      │       └─────────────────┘       │ created_at      │  │
└─────────────────┘                                 └─────────────────┘  │
                                                                         │
┌─────────────────┐       ┌─────────────────┐                           │
│   attachments   │       │   tool_calls    │                           │
├─────────────────┤       ├─────────────────┤                           │
│ id (PK)         │       │ id (PK)         │                           │
│ message_id (FK) │◄──────│ message_id (FK) │◄──────────────────────────┘
│ file_type       │       │ tool_name       │
│ file_url        │       │ arguments       │
│ file_size       │       │ result          │
│ created_at      │       │ status          │
└─────────────────┘       │ latency_ms      │
                          │ created_at      │
                          └─────────────────┘
```

### 4.2 表结构定义

```sql
-- 用户表
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    avatar_url VARCHAR(500),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 会话表
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(200) DEFAULT '新对话',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_deleted BOOLEAN DEFAULT FALSE
);
CREATE INDEX idx_sessions_user_id ON sessions(user_id);
CREATE INDEX idx_sessions_updated_at ON sessions(updated_at DESC);

-- 消息表
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    content_type VARCHAR(20) DEFAULT 'text',  -- text, multimodal
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX idx_messages_session_id ON messages(session_id);
CREATE INDEX idx_messages_created_at ON messages(created_at);

-- 工具调用记录表
CREATE TABLE tool_calls (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id UUID NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    tool_name VARCHAR(50) NOT NULL,
    arguments JSONB NOT NULL,
    result JSONB,
    status VARCHAR(20) DEFAULT 'pending',  -- pending, success, failed
    latency_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX idx_tool_calls_message_id ON tool_calls(message_id);

-- 附件表
CREATE TABLE attachments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id UUID NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    file_type VARCHAR(20) NOT NULL,  -- image, document
    file_url VARCHAR(500) NOT NULL,
    thumbnail_url VARCHAR(500),
    file_size INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX idx_attachments_message_id ON attachments(message_id);
```

---

## 5. 核心流程时序图

### 5.1 简单问答流程

```
┌──────┐          ┌──────────┐          ┌────────┐          ┌──────┐
│Client│          │API Server│          │ Router │          │Talker│
└──┬───┘          └────┬─────┘          └───┬────┘          └──┬───┘
   │                   │                    │                  │
   │  POST /chat       │                    │                  │
   │──────────────────►│                    │                  │
   │                   │                    │                  │
   │                   │  classify_intent   │                  │
   │                   │───────────────────►│                  │
   │                   │                    │                  │
   │                   │  intent: simple_qa │                  │
   │                   │◄───────────────────│                  │
   │                   │                    │                  │
   │                   │                    │   generate       │
   │                   │────────────────────┼─────────────────►│
   │                   │                    │                  │
   │  SSE: delta       │                    │   stream tokens  │
   │◄──────────────────│◄───────────────────┼──────────────────│
   │                   │                    │                  │
   │  SSE: delta       │                    │                  │
   │◄──────────────────│◄───────────────────┼──────────────────│
   │                   │                    │                  │
   │  SSE: end         │                    │                  │
   │◄──────────────────│                    │                  │
   │                   │                    │                  │
```

### 5.2 工具调用流程（Thinker-Talker 并行）

```
┌──────┐      ┌──────────┐      ┌────────┐      ┌──────┐      ┌───────┐      ┌─────┐
│Client│      │API Server│      │ Router │      │Talker│      │Thinker│      │Tools│
└──┬───┘      └────┬─────┘      └───┬────┘      └──┬───┘      └───┬───┘      └──┬──┘
   │               │                │              │              │             │
   │ POST /chat    │                │              │              │             │
   │──────────────►│                │              │              │             │
   │               │                │              │              │             │
   │               │ classify       │              │              │             │
   │               │───────────────►│              │              │             │
   │               │                │              │              │             │
   │               │ intent: tool   │              │              │             │
   │               │◄───────────────│              │              │             │
   │               │                │              │              │             │
   │               │ ═══════════════╪══════════════╪══════════════╪═══         │
   │               │ ║ 并行执行                                     ║           │
   │               │ ║              │              │              │ ║           │
   │               │ ║ quick_reply  │              │              │ ║           │
   │               │ ║─────────────►│              │              │ ║           │
   │               │ ║              │              │              │ ║           │
   │ SSE: talker   │ ║              │  "让我查..." │              │ ║           │
   │◄──────────────│◄║──────────────│──────────────│              │ ║           │
   │               │ ║              │              │              │ ║           │
   │               │ ║              │   plan       │              │ ║           │
   │               │ ║──────────────┼──────────────┼─────────────►│ ║           │
   │               │ ║              │              │              │ ║           │
   │               │ ║              │              │  call_tool   │ ║           │
   │               │ ║              │              │─────────────►│─║──────────►│
   │               │ ║              │              │              │ ║           │
   │ SSE: tool     │ ║              │              │              │ ║  result   │
   │◄──────────────│◄║──────────────┼──────────────┼──────────────│◄║───────────│
   │               │ ║              │              │              │ ║           │
   │               │ ═══════════════╪══════════════╪══════════════╪═══         │
   │               │                │              │              │             │
   │               │ merge_response │              │              │             │
   │               │───────────────►│              │              │             │
   │               │                │              │              │             │
   │ SSE: thinker  │                │              │              │             │
   │◄──────────────│ detailed answer│              │              │             │
   │               │                │              │              │             │
   │ SSE: end      │                │              │              │             │
   │◄──────────────│                │              │              │             │
```

### 5.3 多模态处理流程

```
┌──────┐      ┌──────────┐      ┌──────┐      ┌────────┐      ┌──────┐      ┌───────┐
│Client│      │API Server│      │Vision│      │ Router │      │Talker│      │Thinker│
└──┬───┘      └────┬─────┘      └──┬───┘      └───┬────┘      └──┬───┘      └───┬───┘
   │               │               │              │              │              │
   │ POST (img+txt)│               │              │              │              │
   │──────────────►│               │              │              │              │
   │               │               │              │              │              │
   │               │ encode_image  │              │              │              │
   │               │──────────────►│              │              │              │
   │               │               │              │              │              │
   │               │ image_features│              │              │              │
   │               │◄──────────────│              │              │              │
   │               │               │              │              │              │
   │               │ classify(txt + img_feat)     │              │              │
   │               │─────────────────────────────►│              │              │
   │               │               │              │              │              │
   │               │ intent: multimodal_search    │              │              │
   │               │◄─────────────────────────────│              │              │
   │               │               │              │              │              │
   │               │ ══════════════╪══════════════╪══════════════╪══════        │
   │               │ ║ 并行: Talker 先行 + Thinker 深度处理        ║            │
   │               │ ══════════════╪══════════════╪══════════════╪══════        │
   │               │               │              │              │              │
   │ SSE: stream   │               │              │              │              │
   │◄──────────────│ merged response              │              │              │
   │               │               │              │              │              │
```

---

## 6. 核心模块设计

### 6.1 意图路由器 (Intent Router)

```python
from enum import Enum
from pydantic import BaseModel

class IntentType(Enum):
    SIMPLE_QA = "simple_qa"           # 简单问答，Talker 独立处理
    TOOL_CALL = "tool_call"           # 需要工具调用
    COMPLEX_REASONING = "complex"     # 复杂推理
    MULTIMODAL = "multimodal"         # 多模态处理

class IntentResult(BaseModel):
    intent: IntentType
    confidence: float
    required_tools: list[str] = []
    requires_thinker: bool = False

class IntentRouter:
    """意图路由器：决定使用哪种处理策略"""

    def __init__(self, classifier_model):
        self.classifier = classifier_model

    async def classify(
        self,
        message: str,
        images: list[bytes] | None = None,
        context: list[dict] | None = None
    ) -> IntentResult:
        """
        分类用户意图

        路由规则：
        1. 简单问答（闲聊、简单知识）→ Talker Only
        2. 天气/搜索/旅行查询 → Talker + Thinker 并行
        3. 复杂规划/多步推理 → Thinker 主导
        4. 图片+文本 → Vision + 意图分类
        """
        # 检测是否包含图片
        if images:
            return await self._classify_multimodal(message, images)

        # 检测是否需要工具
        tool_keywords = {
            "weather": ["天气", "气温", "下雨", "空气质量"],
            "search": ["搜索", "查一下", "最新", "新闻"],
            "hotel": ["酒店", "住宿", "宾馆"],
            "flight": ["机票", "航班", "飞机"],
            "attraction": ["景点", "旅游", "玩的地方"],
        }

        required_tools = []
        for tool, keywords in tool_keywords.items():
            if any(kw in message for kw in keywords):
                required_tools.append(tool)

        if required_tools:
            return IntentResult(
                intent=IntentType.TOOL_CALL,
                confidence=0.9,
                required_tools=required_tools,
                requires_thinker=True
            )

        # 默认简单问答
        return IntentResult(
            intent=IntentType.SIMPLE_QA,
            confidence=0.85,
            requires_thinker=False
        )
```

### 6.2 响应融合器 (Response Merger)

```python
import asyncio
from dataclasses import dataclass
from typing import AsyncIterator

@dataclass
class ResponseChunk:
    source: str          # "talker" | "thinker" | "tool"
    content: str
    is_final: bool = False
    tool_info: dict | None = None

class ResponseMerger:
    """响应融合器：合并 Talker 和 Thinker 的输出"""

    def __init__(self):
        self.talker_buffer = []
        self.thinker_buffer = []
        self.tool_results = []

    async def merge_streams(
        self,
        talker_stream: AsyncIterator[str],
        thinker_stream: AsyncIterator[str] | None = None,
        tool_stream: AsyncIterator[dict] | None = None
    ) -> AsyncIterator[ResponseChunk]:
        """
        融合策略：
        1. Talker 输出立即透传（保证 < 1s 首字响应）
        2. 工具调用结果实时推送
        3. Thinker 输出在 Talker 完成后追加
        4. 如果 Thinker 修正 Talker，使用温和过渡语
        """

        # 阶段1：Talker 快速响应
        async for token in talker_stream:
            self.talker_buffer.append(token)
            yield ResponseChunk(source="talker", content=token)

        # 阶段2：工具结果（如果有）
        if tool_stream:
            async for tool_result in tool_stream:
                self.tool_results.append(tool_result)
                yield ResponseChunk(
                    source="tool",
                    content="",
                    tool_info=tool_result
                )

        # 阶段3：Thinker 补充（如果有）
        if thinker_stream:
            # 添加过渡语
            yield ResponseChunk(source="thinker", content="\n\n")

            async for token in thinker_stream:
                self.thinker_buffer.append(token)
                yield ResponseChunk(source="thinker", content=token)

        # 结束标记
        yield ResponseChunk(source="system", content="", is_final=True)
```

### 6.3 对话编排器 (Orchestrator)

```python
class ChatOrchestrator:
    """对话编排器：协调整个对话流程"""

    def __init__(
        self,
        router: IntentRouter,
        talker: TalkerService,
        thinker: ThinkerService,
        tool_executor: ToolExecutor,
        merger: ResponseMerger
    ):
        self.router = router
        self.talker = talker
        self.thinker = thinker
        self.tool_executor = tool_executor
        self.merger = merger

    async def process(
        self,
        message: str,
        images: list[bytes] | None = None,
        context: list[dict] | None = None
    ) -> AsyncIterator[ResponseChunk]:
        """处理用户消息"""

        # 1. 意图识别
        intent = await self.router.classify(message, images, context)

        # 2. 根据意图选择处理策略
        if intent.intent == IntentType.SIMPLE_QA:
            # 简单问答：仅 Talker
            async for chunk in self._handle_simple_qa(message, context):
                yield chunk

        elif intent.intent == IntentType.TOOL_CALL:
            # 工具调用：Talker + Thinker 并行
            async for chunk in self._handle_tool_call(
                message, context, intent.required_tools
            ):
                yield chunk

        elif intent.intent == IntentType.MULTIMODAL:
            # 多模态：Vision + Talker + Thinker
            async for chunk in self._handle_multimodal(
                message, images, context
            ):
                yield chunk

    async def _handle_tool_call(
        self,
        message: str,
        context: list[dict],
        tools: list[str]
    ) -> AsyncIterator[ResponseChunk]:
        """处理需要工具调用的请求"""

        # 并行启动 Talker 和 Thinker
        talker_task = asyncio.create_task(
            self.talker.quick_response(message, context)
        )
        thinker_task = asyncio.create_task(
            self.thinker.plan_and_execute(message, context, tools)
        )

        # Talker 先行响应
        talker_stream = await talker_task
        async for token in talker_stream:
            yield ResponseChunk(source="talker", content=token)

        # 等待 Thinker 完成工具调用
        thinker_result = await thinker_task

        # 输出工具调用结果
        for tool_call in thinker_result.tool_calls:
            yield ResponseChunk(
                source="tool",
                content="",
                tool_info=tool_call
            )

        # Thinker 详细回答
        async for token in thinker_result.response_stream:
            yield ResponseChunk(source="thinker", content=token)
```

---

## 7. 工具服务设计

### 7.1 工具基类

```python
from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Any

class ToolResult(BaseModel):
    success: bool
    data: Any | None = None
    error: str | None = None
    latency_ms: int = 0

class BaseTool(ABC):
    """工具基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述，用于 LLM 理解"""
        pass

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """参数 JSON Schema"""
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """执行工具"""
        pass

    def to_function_schema(self) -> dict:
        """转换为 OpenAI Function Calling 格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }
```

### 7.2 天气查询工具

```python
import httpx
from datetime import datetime

class WeatherTool(BaseTool):
    """天气查询工具"""

    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=5.0)

    @property
    def name(self) -> str:
        return "weather"

    @property
    def description(self) -> str:
        return "查询指定城市的天气信息，包括实时天气、温度、空气质量等"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称，如：北京、上海"
                },
                "days": {
                    "type": "integer",
                    "description": "预报天数，1-7天",
                    "default": 1
                }
            },
            "required": ["city"]
        }

    async def execute(self, city: str, days: int = 1) -> ToolResult:
        start_time = datetime.now()
        try:
            response = await self.client.get(
                f"{self.base_url}/weather",
                params={"city": city, "days": days, "key": self.api_key}
            )
            response.raise_for_status()
            data = response.json()

            latency = int((datetime.now() - start_time).total_seconds() * 1000)
            return ToolResult(success=True, data=data, latency_ms=latency)

        except Exception as e:
            latency = int((datetime.now() - start_time).total_seconds() * 1000)
            return ToolResult(success=False, error=str(e), latency_ms=latency)
```

### 7.3 联网搜索工具

```python
class SearchTool(BaseTool):
    """联网搜索工具"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=10.0)

    @property
    def name(self) -> str:
        return "search"

    @property
    def description(self) -> str:
        return "搜索互联网获取最新信息，包括新闻、知识、实时资讯等"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词"
                },
                "num_results": {
                    "type": "integer",
                    "description": "返回结果数量",
                    "default": 5
                }
            },
            "required": ["query"]
        }

    async def execute(self, query: str, num_results: int = 5) -> ToolResult:
        start_time = datetime.now()
        try:
            # 调用搜索 API（示例使用 SerpAPI 或自建搜索服务）
            response = await self.client.get(
                "https://api.search.example.com/search",
                params={
                    "q": query,
                    "num": num_results,
                    "api_key": self.api_key
                }
            )
            response.raise_for_status()

            results = response.json().get("results", [])
            # 格式化搜索结果
            formatted = [
                {
                    "title": r["title"],
                    "snippet": r["snippet"],
                    "url": r["url"],
                    "date": r.get("date")
                }
                for r in results
            ]

            latency = int((datetime.now() - start_time).total_seconds() * 1000)
            return ToolResult(success=True, data=formatted, latency_ms=latency)

        except Exception as e:
            latency = int((datetime.now() - start_time).total_seconds() * 1000)
            return ToolResult(success=False, error=str(e), latency_ms=latency)
```

### 7.4 工具执行器

```python
class ToolExecutor:
    """工具执行器：管理和执行所有工具"""

    def __init__(self):
        self.tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        """注册工具"""
        self.tools[tool.name] = tool

    def get_tool_schemas(self, tool_names: list[str] | None = None) -> list[dict]:
        """获取工具 Schema 列表"""
        if tool_names:
            return [
                self.tools[name].to_function_schema()
                for name in tool_names
                if name in self.tools
            ]
        return [tool.to_function_schema() for tool in self.tools.values()]

    async def execute(self, tool_name: str, arguments: dict) -> ToolResult:
        """执行指定工具"""
        if tool_name not in self.tools:
            return ToolResult(success=False, error=f"Unknown tool: {tool_name}")

        tool = self.tools[tool_name]
        return await tool.execute(**arguments)

    async def execute_parallel(
        self,
        calls: list[tuple[str, dict]]
    ) -> list[ToolResult]:
        """并行执行多个工具"""
        tasks = [
            self.execute(tool_name, args)
            for tool_name, args in calls
        ]
        return await asyncio.gather(*tasks)
```

---

## 8. 配置管理

### 8.1 配置文件结构

```yaml
# config/config.yaml
app:
  name: "智能对话助手"
  version: "1.0.0"
  debug: false

server:
  host: "0.0.0.0"
  port: 8000
  workers: 4

database:
  url: "postgresql://user:pass@localhost:5432/chatbot"
  pool_size: 20
  max_overflow: 10

redis:
  url: "redis://localhost:6379/0"
  max_connections: 100

models:
  talker:
    name: "Qwen2-7B-Chat"
    endpoint: "http://talker-service:8001"
    timeout: 30
    max_tokens: 2048

  thinker:
    name: "Qwen2-72B-Chat"
    endpoint: "http://thinker-service:8002"
    timeout: 60
    max_tokens: 4096

  vision:
    name: "Qwen2-VL-7B"
    endpoint: "http://vision-service:8003"
    timeout: 30

tools:
  weather:
    api_key: "${WEATHER_API_KEY}"
    base_url: "https://api.weather.example.com"

  search:
    api_key: "${SEARCH_API_KEY}"
    base_url: "https://api.search.example.com"

storage:
  type: "minio"
  endpoint: "http://minio:9000"
  access_key: "${MINIO_ACCESS_KEY}"
  secret_key: "${MINIO_SECRET_KEY}"
  bucket: "chatbot-files"

logging:
  level: "INFO"
  format: "json"

security:
  jwt_secret: "${JWT_SECRET}"
  jwt_expire_hours: 24
  rate_limit:
    requests_per_minute: 60
    requests_per_day: 1000
```

### 8.2 配置加载

```python
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # App
    app_name: str = "智能对话助手"
    debug: bool = False

    # Database
    database_url: str
    redis_url: str

    # Models
    talker_endpoint: str
    thinker_endpoint: str
    vision_endpoint: str

    # Tools
    weather_api_key: str
    search_api_key: str

    # Security
    jwt_secret: str
    jwt_expire_hours: int = 24

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    return Settings()
```

---

## 9. 错误处理

### 9.1 错误码定义

| 错误码 | 说明 | HTTP 状态码 |
|--------|------|-------------|
| 1001 | 参数错误 | 400 |
| 1002 | 认证失败 | 401 |
| 1003 | 权限不足 | 403 |
| 1004 | 资源不存在 | 404 |
| 1005 | 请求频率超限 | 429 |
| 2001 | 模型服务不可用 | 503 |
| 2002 | 模型响应超时 | 504 |
| 2003 | 工具调用失败 | 500 |
| 3001 | 内容安全拦截 | 451 |

### 9.2 统一错误响应

```python
from fastapi import HTTPException
from pydantic import BaseModel

class ErrorResponse(BaseModel):
    code: int
    message: str
    detail: str | None = None

class AppException(HTTPException):
    def __init__(self, code: int, message: str, detail: str = None):
        self.code = code
        self.message = message
        self.detail = detail
        super().__init__(
            status_code=self._get_http_status(code),
            detail={"code": code, "message": message, "detail": detail}
        )

    @staticmethod
    def _get_http_status(code: int) -> int:
        mapping = {
            1001: 400, 1002: 401, 1003: 403, 1004: 404, 1005: 429,
            2001: 503, 2002: 504, 2003: 500, 3001: 451
        }
        return mapping.get(code, 500)
```

### 9.3 降级策略

```python
class FallbackHandler:
    """降级处理器"""

    async def handle_model_failure(self, error: Exception) -> str:
        """模型服务故障降级"""
        return "抱歉，服务暂时繁忙，请稍后再试。"

    async def handle_tool_failure(
        self,
        tool_name: str,
        error: Exception
    ) -> ToolResult:
        """工具调用失败降级"""
        fallback_messages = {
            "weather": "暂时无法获取天气信息，请稍后再试。",
            "search": "搜索服务暂时不可用，请稍后再试。",
            "hotel": "酒店查询服务暂时不可用。",
            "flight": "机票查询服务暂时不可用。",
        }
        return ToolResult(
            success=False,
            error=fallback_messages.get(tool_name, "服务暂时不可用")
        )
```

---

## 10. 部署配置

### 10.1 Docker Compose

```yaml
version: '3.8'

services:
  # API 网关
  api:
    build:
      context: .
      dockerfile: docker/Dockerfile.api
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@db:5432/chatbot
      - REDIS_URL=redis://redis:6379/0
      - TALKER_ENDPOINT=http://talker:8001
      - THINKER_ENDPOINT=http://thinker:8002
    depends_on:
      - db
      - redis
    deploy:
      replicas: 3

  # Talker 模型服务
  talker:
    build:
      context: .
      dockerfile: docker/Dockerfile.model
    command: ["python", "-m", "vllm.entrypoints.openai.api_server",
              "--model", "/models/Qwen2-7B-Chat",
              "--port", "8001"]
    volumes:
      - ./models:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # Thinker 模型服务
  thinker:
    build:
      context: .
      dockerfile: docker/Dockerfile.model
    command: ["python", "-m", "vllm.entrypoints.openai.api_server",
              "--model", "/models/Qwen2-72B-Chat",
              "--port", "8002",
              "--tensor-parallel-size", "4"]
    volumes:
      - ./models:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 4
              capabilities: [gpu]

  # PostgreSQL
  db:
    image: postgres:15
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=chatbot
    volumes:
      - postgres_data:/var/lib/postgresql/data

  # Redis
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

  # MinIO
  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    environment:
      - MINIO_ROOT_USER=minioadmin
      - MINIO_ROOT_PASSWORD=minioadmin
    volumes:
      - minio_data:/data

volumes:
  postgres_data:
  redis_data:
  minio_data:
```

### 10.2 Nginx 配置

```nginx
upstream api_servers {
    least_conn;
    server api:8000 weight=1;
}

server {
    listen 80;
    listen 443 ssl http2;
    server_name api.example.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;

    # 限流配置
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;

    location /api/ {
        limit_req zone=api_limit burst=20 nodelay;

        proxy_pass http://api_servers;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;

        # SSE 支持
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
    }

    location /health {
        access_log off;
        return 200 "OK";
    }
}
```

---

## 11. 监控与告警

### 11.1 关键指标

| 指标 | 说明 | 告警阈值 |
|------|------|----------|
| request_latency_p95 | P95 响应延迟 | > 2s |
| request_latency_p99 | P99 响应延迟 | > 5s |
| error_rate | 错误率 | > 1% |
| model_latency | 模型推理延迟 | > 3s |
| tool_success_rate | 工具调用成功率 | < 95% |
| gpu_utilization | GPU 利用率 | > 90% |
| memory_usage | 内存使用率 | > 85% |

### 11.2 Prometheus 指标

```python
from prometheus_client import Counter, Histogram, Gauge

# 请求计数
REQUEST_COUNT = Counter(
    'chat_requests_total',
    'Total chat requests',
    ['endpoint', 'status']
)

# 响应延迟
REQUEST_LATENCY = Histogram(
    'chat_request_latency_seconds',
    'Request latency in seconds',
    ['endpoint'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

# 模型推理延迟
MODEL_LATENCY = Histogram(
    'model_inference_latency_seconds',
    'Model inference latency',
    ['model_name'],
    buckets=[0.5, 1.0, 2.0, 3.0, 5.0]
)

# 工具调用
TOOL_CALLS = Counter(
    'tool_calls_total',
    'Total tool calls',
    ['tool_name', 'status']
)

# 活跃会话数
ACTIVE_SESSIONS = Gauge(
    'active_sessions',
    'Number of active sessions'
)
```

---

## 12. 附录

### 12.1 技术栈版本

| 组件 | 版本 |
|------|------|
| Python | 3.11+ |
| FastAPI | 0.109+ |
| vLLM | 0.3+ |
| PostgreSQL | 15+ |
| Redis | 7+ |
| Docker | 24+ |

### 12.2 参考资料

- FastAPI 官方文档: https://fastapi.tiangolo.com/
- vLLM 文档: https://docs.vllm.ai/
- Qwen 模型: https://github.com/QwenLM/Qwen
- OpenAI Function Calling: https://platform.openai.com/docs/guides/function-calling
