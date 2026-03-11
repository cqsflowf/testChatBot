# 测试工程指导

## 概述

本文档描述全双工语音对话系统的测试工程结构和运行方法。

## 测试目录结构

```
tests/
├── test_cases_interrupt.md    # 打断测试用例文档
├── test_interrupt_suite.py    # 打断时延测试套件
├── test_interrupt_latency.py  # 打断时延测试（带系统初始化）
├── test_interrupt_websocket.py # WebSocket打断测试
├── test_comprehensive.py      # 综合测试
└── test_e2e.py               # 端到端测试
```

## 运行测试

### 1. 打断时延测试套件（推荐）

```bash
cd E:\Claude\vedio
python tests/test_interrupt_suite.py
```

**测试内容**：
- TC-001 ~ TC-006: 有效人声关键词检测
- TC-007 ~ TC-010: 语气助词过滤测试
- TC-011 ~ TC-013: 混合内容测试
- 时延基准测试

### 2. 完整打断测试（需API密钥）

```bash
cd E:\Claude\vedio
python tests/test_interrupt_latency.py
```

**测试内容**：
- 有效人声检测单元测试
- 正常对话流程测试
- 打断场景测试

### 3. WebSocket打断测试（需启动服务器）

```bash
# 终端1：启动服务器
python start.py

# 终端2：运行测试
python tests/test_interrupt_websocket.py
```

### 4. Web界面测试

1. 启动服务器: `python start.py`
2. 访问打断测试页面: http://localhost:8765/interrupt-test
3. 点击"开始测试"按钮执行自动化测试
4. 查看时延监控: http://localhost:8765/monitor

## 性能指标要求

| 指标 | 目标值 | 说明 |
|-----|-------|------|
| 有效人声判断延迟 | < 200ms | 从用户开口到判断为有效人声 |
| 打断响应时间 | < 500ms | 从有效人声判断到TTS停止 |
| 语气助词不触发打断 | 正确过滤 | "嗯"、"啊"等不触发打断 |
| 关键词立即触发 | < 10ms | "帮我"、"什么"等立即触发 |

## 测试用例

详见 `tests/test_cases_interrupt.md`

### 核心测试场景

**场景**：用户语音输入 "帮我介绍下三国演义"，系统执行并播报，播报约5秒后用户再输入 "帮我介绍下西游记，里面的孙悟空很厉害，请介绍下"

**验证点**：
1. 系统在检测到"帮我"时判断为有效人声
2. 立即停止TTS播报
3. 接收完整的用户新输入
4. 正常处理新的请求

## 日志文件

测试日志保存在：
```
logs/voice_dialog_YYYYMMDD_HHMMSS.log
```

## 持续集成

### 添加新测试用例

1. 在 `tests/test_cases_interrupt.md` 中添加用例描述
2. 在 `tests/test_interrupt_suite.py` 中添加测试代码
3. 运行测试验证

### 回归测试

每次代码改动后执行：

```bash
# 快速验证
python tests/test_interrupt_suite.py

# 完整验证
python tests/test_interrupt_latency.py
python tests/test_interrupt_websocket.py
```

## 常见问题

### Q: 测试失败怎么办？

1. 检查日志文件 `logs/voice_dialog_*.log`
2. 确认API密钥配置正确
3. 确认网络连接正常

### Q: 如何查看实时时延？

访问 http://localhost:8765/monitor

### Q: 如何调试打断逻辑？

1. 查看日志中 `[打断]` 关键词
2. 查看日志中 `[有效人声]` 和 `[语气助词]` 关键词
3. 使用时延监控界面观察各阶段耗时