# 全双工语音对话系统 v3.0

一个基于 Qwen 模型的全双工语音对话系统，支持流式语音识别、流式语义VAD、并行情绪识别和多轮对话。

## ✨ 核心特性

### v3.0 新特性

- **流式ASR识别**：使用 Qwen ASR 17B (Paraformer) 进行实时语音转文字，帧长20ms
- **流式语义VAD**：使用 Qwen3-Omni-Flash 边说边判断用户是否说完
- **并行情绪识别**：与ASR并行执行，实时分析用户情绪
- **声学VAD优化**：WebRTC VAD 阈值 < 500ms
- **全双工交互**：支持用户随时打断AI回复
- **多轮对话**：支持上下文连贯的多轮对话
- **工具调用**：支持天气查询、时间查询、设备控制等

### 技术架构

```
用户语音 → 声学VAD → ASR流式识别 → 语义VAD流式判断
                ↓
           情绪识别（并行）
                ↓
           LLM任务规划 → 工具调用 → TTS语音合成 → 播放
```

## 🚀 快速开始

### 1. 环境要求

- Python 3.8+
- 现代浏览器 (Chrome/Firefox/Edge)
- 网络访问 (DashScope API)

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置API密钥

```bash
# 方式一：设置环境变量
export DASHSCOPE_API_KEY="your-api-key"

# 方式二：创建 .env 文件
echo "DASHSCOPE_API_KEY=your-api-key" > .env
```

### 4. 启动服务

```bash
python start.py
```

### 5. 访问界面

打开浏览器访问 http://localhost:8765

## 📁 项目结构

```
vedio/
├── config/
│   └── model_config.yaml      # 模型配置文件
├── src/
│   └── voice_dialog/
│       ├── core/               # 核心模块
│       │   ├── types.py        # 数据类型定义
│       │   ├── config.py       # 配置管理
│       │   ├── state_machine.py # 状态机
│       │   ├── logger.py       # 日志工具
│       │   └── tool_registry.py # 工具注册中心
│       ├── modules/            # 功能模块
│       │   ├── acoustic_vad.py # 声学VAD
│       │   ├── qwen_asr.py     # 流式ASR
│       │   ├── semantic_vad.py # 语义VAD
│       │   ├── emotion.py      # 情绪识别
│       │   ├── llm_planner.py  # LLM任务规划
│       │   ├── tools.py        # 工具引擎
│       │   ├── tts.py          # 语音合成
│       │   └── qwen_omni.py    # Qwen Omni处理器
│       ├── system.py           # 核心系统
│       └── websocket_server.py # WebSocket服务器
├── web/
│   └── index.html              # Web前端
├── tests/                      # 测试用例
├── start.py                    # 启动脚本
└── requirements.txt            # 依赖列表
```

## 🔧 配置说明

编辑 `config/model_config.yaml` 自定义配置：

```yaml
# Qwen ASR 17B 配置
QWEN_ASR:
  model: "paraformer-realtime-v2"
  frame_duration_ms: 20
  sample_rate: 16000

# Qwen Omni Flash 配置
QWEN_OMNI:
  model: "Qwen3-Omni-Flash"

# 语义VAD配置
SEMANTIC_VAD:
  enabled: true
  model: "Qwen3-Omni-Flash"

# 情绪识别配置
EMOTION:
  mode: "parallel"
  model: "Qwen3-Omni-Flash"

# LLM配置
LLM:
  model: "qwen-plus"

# TTS配置
TTS:
  provider: "edge"
  voice: "zh-CN-XiaoxiaoNeural"

# 声学VAD配置
ACOUSTIC_VAD:
  engine: "webrtc"
  aggressiveness: 3
  frame_duration_ms: 20
  silence_threshold_ms: 400

# 服务器配置
SERVER:
  host: "0.0.0.0"
  port: 8765
```

## 🛠️ 支持的工具

| 工具 | 功能 | 示例 |
|------|------|------|
| get_current_time | 获取当前时间日期 | "现在几点了" |
| get_weather | 查询天气 | "北京今天天气怎么样" |
| search_web | 网络搜索 | "搜索一下Python教程" |
| set_reminder | 设置提醒 | "帮我设一个明天9点的提醒" |
| play_music | 播放音乐 | "播放一首轻松的歌" |
| control_device | 设备控制 | "打开客厅的灯" |

## 🧪 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_duplex_dialog.py -v
```

## 📊 性能指标

| 指标 | 数值 | 说明 |
|------|------|------|
| ASR帧长 | 20ms | 平衡延迟与准确率 |
| 声学VAD阈值 | <500ms | 快速响应 |
| 最大静音等待 | 2000ms | 超时保护 |
| 最小语音时长 | 300ms | 过滤短噪音 |

## 📖 文档

- [项目设计说明.md](项目设计说明.md) - 详细技术实现说明
- [架构设计文档.md](架构设计文档.md) - 系统架构设计

## 📝 API接口

### WebSocket 端点

```
ws://localhost:8765/ws/{client_id}
```

### 消息类型

| 消息类型 | 方向 | 说明 |
|----------|------|------|
| audio | 前端→后端 | 音频数据 (PCM 16kHz 16-bit mono) |
| text | 前端→后端 | 文本输入 |
| interrupt | 前端→后端 | 打断请求 |
| result | 后端→前端 | 对话结果 |
| state_change | 后端→前端 | 状态变化 |
| partial_asr | 后端→前端 | ASR部分结果 |

### HTTP 端点

- `GET /` - Web界面
- `GET /health` - 健康检查

## 🔑 技术栈

| 类别 | 技术选型 |
|------|----------|
| 编程语言 | Python 3.8+ |
| Web框架 | FastAPI + Uvicorn |
| 实时通信 | WebSocket |
| ASR模型 | Qwen ASR 17B (Paraformer) |
| 语义VAD | Qwen3-Omni-Flash |
| 情绪识别 | Qwen3-Omni-Flash |
| LLM | Qwen-Plus |
| TTS | Edge TTS |
| 声学VAD | WebRTC VAD |

## 📄 许可证

MIT License