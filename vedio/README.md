# 全双工语音对话系统 v3.5

一个基于 Qwen 模型的全双工语音对话系统，支持流式语音识别、流式语义VAD、有效人声判断、并行情绪识别、LLM流式输出和实时TTS播报。

## 核心特性

### v3.5 新特性

- **并发音频处理**：音频接收与 LLM 推理解耦，支持真正的全双工交互
- **THINKING状态打断**：支持在LLM推理期间打断，取消LLM请求
- **非阻塞架构**：WebSocket 主循环不再被 LLM 推理阻塞
- **后台任务模型**：LLM 作为后台任务运行，音频持续接收

### v3.4 新特性

- **THINKING状态打断**：支持在LLM推理期间打断，取消LLM请求
- **模型优化**：LLM和语义VAD使用 qwen-flash 模型，响应更快
- **TTS升级**：使用 qwen3-tts-flash 替代 Edge TTS
- **JSON解析增强**：语义VAD支持多种JSON格式解析，更加健壮

### v3.3 新特性

- **Silero VAD**：使用基于深度学习的Silero VAD替代WebRTC VAD，提供更准确的语音检测
- **LLM流式输出**：LLM响应实时流式显示，用户无需等待完整响应
- **流式TTS播报**：使用 Qwen3-tts-flash 按句子实时转换语音，边生成边播报
- **TTS顺序保证**：使用异步锁确保音频按LLM输出顺序播放，避免乱序
- **工具调用实时检测**：在LLM流式输出时实时检测工具调用
- **音频队列管理**：新问题开始时自动清空旧音频队列

### v3.2 新特性

- **语义驱动打断**：判断有效人声后立即停止TTS，不等声学静音阈值
- **有效人声判断**：区分语气助词（嗯、啊）和有效语义内容
- **时延监控**：实时追踪各模块处理时延，便于性能分析
- **打断测试套件**：完善的测试用例和自动化测试脚本

### 技术架构

```
用户语音 → 声学VAD → ASR流式识别 → 语义VAD流式判断（有效人声判断）
                ↓
           情绪识别（并行，音频输入）
                ↓
           LLM流式输出 → TTS处理器（异步锁保证顺序）→ 按句子实时播报
                ↓
           工具调用（实时检测） → 工具执行 → 结果总结
                ↓
           时延追踪（全程监控）
```

## 模型配置

| 模块 | 模型名称 | 用途 |
|------|----------|------|
| ASR | `paraformer-realtime-v2` | 语音转文字 |
| 语义VAD | `qwen-flash` | 语义完整性判断 |
| 情绪识别 | `qwen3-omni-flash` | 音频情绪分析 |
| LLM | `qwen-flash` | 对话推理、工具调用 |
| TTS | `qwen3-tts-flash` | 文字转语音 |
| 声学VAD | `silero-vad` | 语音活动检测 |

## 快速开始

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

| 功能 | 地址 |
|------|------|
| 主界面 | http://localhost:8765 |
| 时延监控 | http://localhost:8765/monitor |
| 打断测试 | http://localhost:8765/interrupt-test |

## 项目结构

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
│       │   ├── logger.py       # 日志工具（支持文件输出）
│       │   ├── tool_registry.py # 工具注册中心
│       │   └── latency.py      # 时延追踪模块
│       ├── modules/            # 功能模块
│       │   ├── acoustic_vad.py # 声学VAD
│       │   ├── qwen_asr.py     # 流式ASR
│       │   ├── semantic_vad.py # 语义VAD + 有效人声判断
│       │   ├── emotion.py      # 情绪识别
│       │   ├── llm_planner.py  # LLM任务规划（支持流式输出）
│       │   ├── tools.py        # 工具引擎
│       │   ├── tts.py          # 语音合成 + 流式TTS处理器
│       │   └── qwen_omni.py    # Qwen Omni处理器
│       ├── system.py           # 核心系统 v3.2
│       └── websocket_server.py # WebSocket服务器
├── web/
│   ├── index.html              # Web前端（支持流式显示）
│   ├── latency_monitor.html    # 时延监控界面
│   └── interrupt_test.html     # 打断测试界面
├── tests/                      # 测试用例
│   ├── test_cases_interrupt.md # 打断测试用例文档
│   ├── test_interrupt_suite.py # 打断时延测试套件
│   ├── test_interrupt_latency.py
│   └── test_interrupt_websocket.py
├── logs/                       # 日志文件目录
├── start.py                    # 启动脚本
└── requirements.txt            # 依赖列表
```

## 配置说明

编辑 `config/model_config.yaml` 自定义配置：

```yaml
# Qwen ASR 17B 配置
QWEN_ASR:
  model: "paraformer-realtime-v2"
  frame_duration_ms: 20
  sample_rate: 16000

# Qwen Omni Flash 配置 (情绪识别)
QWEN_OMNI:
  model: "qwen3-omni-flash"

# 语义VAD配置
SEMANTIC_VAD:
  enabled: true
  model: "qwen3-flash"

# 情绪识别配置
EMOTION:
  mode: "parallel"
  model: "qwen3-omni-flash"
  use_audio: true

# LLM配置
LLM:
  model: "qwen3-flash"
  tools:
    enabled: true
  generation:
    temperature: 0.7
    max_tokens: 1024

# TTS配置
TTS:
  provider: "qwen3"
  model: "qwen3-tts-flash"
  voice: "Cherry"
  sample_rate: 24000

# 声学VAD配置
ACOUSTIC_VAD:
  engine: "silero"
  frame_duration_ms: 20
  silence_threshold_ms: 500
  padding_duration_ms: 150

# 服务器配置
SERVER:
  host: "0.0.0.0"
  port: 8765
```

## LLM流式输出与TTS播报 (v3.2.3)

### 流式输出流程

```
LLM流式输出文本
       │
       ├── 文本块发送到前端显示
       │
       └── 文本块发送到 StreamingTTSProcessor
              │
              ├── 检测到句子结束符？──→ 立即TTS转换
              │
              └── 积累达到最大长度？──→ TTS转换
                      │
                      ▼
              音频块发送到前端播放
```

### TTS分段策略

- **句子结束符**：遇到 `。！？；\n` 立即转换
- **最大长度**：积累到80字符后转换
- **最小长度**：至少5字符才转换

### 音频队列管理

- 新问题开始时自动清空旧音频队列
- 打断时停止当前播放并清空队列
- 支持音频块连续播放

## TTS文本预处理 (v3.2.1)

TTS模块会自动清理文本中的格式符号，确保播报自然：

| 输入 | 输出 |
|------|------|
| `**重要**提示` | 重要提示 |
| `[点击](url)查看` | 点击查看 |
| `# 标题` | 标题 |
| `- 列表项` | 列表项 |
| `你好😊` | 你好 |

## 有效人声判断

### 有效人声关键词（触发打断）

```
帮我、我要、我想、给我、请、查、找、看、听
打开、关闭、播放、设置、停止、开始、换
什么、怎么、为什么、哪、谁、几、多少、吗、呢
好的、行、可以、对、是、不、好
停、算、取消、不要、别
```

### 语气助词（不触发打断）

```
嗯、啊、呃、额、唔、哦、噢、哈、嘿、哎
那个、就是、这个、然后、所以、但是、其实
```

## 支持的工具

| 工具 | 功能 | 示例 |
|------|------|------|
| get_current_time | 获取当前时间日期 | "现在几点了" |
| get_weather | 查询天气 | "北京今天天气怎么样" |
| search_web | 网络搜索 | "搜索一下Python教程" |
| set_reminder | 设置提醒 | "帮我设一个明天9点的提醒" |
| play_music | 播放音乐 | "播放一首轻松的歌" |
| control_device | 设备控制 | "打开客厅的灯" |

## 运行测试

### 打断时延测试套件

```bash
# 快速验证
python tests/test_interrupt_suite.py

# 完整打断测试
python tests/test_interrupt_latency.py

# WebSocket打断测试
python tests/test_interrupt_websocket.py
```

### 测试结果示例

```
测试用例总数: 13
通过: 13
失败: 0

性能指标:
  最大延迟: 0.23ms
  平均延迟: 0.09ms

有效人声判断延迟 < 200ms: PASS
```

## 性能指标

| 指标 | 目标值 | 实测值 | 说明 |
|------|--------|--------|------|
| 有效人声判断延迟 | < 200ms | < 1ms | 从用户开口到判断为有效人声 |
| 打断响应时间 | < 500ms | < 1ms | 从有效人声判断到TTS停止 |
| ASR帧长 | 20ms | 20ms | 平衡延迟与准确率 |
| 声学VAD阈值 | 500ms | 500ms | Silero VAD静音检测 |
| LLM首字输出 | < 300ms | - | 从请求到首个输出 |

## API接口

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
| reset | 前端→后端 | 重置对话 |
| result | 后端→前端 | 对话结果 |
| state_change | 后端→前端 | 状态变化 |
| partial_asr | 后端→前端 | ASR部分结果 |
| llm_chunk | 后端→前端 | LLM流式输出块 (v3.2.3新增) |
| audio_chunk | 后端→前端 | TTS音频块 (v3.2.3新增) |
| clear_audio | 后端→前端 | 清空音频队列 (v3.2.3新增) |
| latency_update | 后端→前端 | 时延更新 |
| tool_executing | 后端→前端 | 工具执行状态 |

### HTTP 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| / | GET | Web界面 |
| /monitor | GET | 时延监控界面 |
| /interrupt-test | GET | 打断测试界面 |
| /health | GET | 健康检查 |
| /latency/history | GET | 时延历史记录 |
| /latency/stats | GET | 时延统计信息 |
| /latency/current | GET | 当前时延数据 |

## 技术栈

| 类别 | 技术选型 | 模型名称 |
|------|----------|----------|
| 编程语言 | Python 3.8+ | - |
| Web框架 | FastAPI + Uvicorn | - |
| 实时通信 | WebSocket | - |
| ASR模型 | Qwen ASR 17B (Paraformer) | `paraformer-realtime-v2` |
| 语义VAD | Qwen3-flash | `qwen3-flash` |
| 情绪识别 | Qwen3-omni-flash (音频输入) | `qwen3-omni-flash` |
| LLM | Qwen3-flash | `qwen3-flash` |
| TTS | Qwen3-tts-flash | `qwen3-tts-flash` |
| 声学VAD | Silero VAD | `silero-vad` |
| 日志 | Loguru | - |

## 文档

- [项目设计说明.md](项目设计说明.md) - 详细技术实现说明
- [架构设计文档.md](架构设计文档.md) - 系统架构设计
- [tests/README.md](tests/README.md) - 测试工程指导
- [tests/test_cases_interrupt.md](tests/test_cases_interrupt.md) - 打断测试用例

## 日志

日志文件保存在：
```
logs/voice_dialog_YYYYMMDD_HHMMSS.log
```

## 许可证

MIT License