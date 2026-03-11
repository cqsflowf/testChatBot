#!/usr/bin/env python
"""
全双工语音对话系统 v3.0 - 启动脚本

v3.0 核心特性：
- Qwen ASR 17B 流式识别（帧长20ms）
- Qwen Omni Flash 流式语义VAD
- Qwen Omni Flash 并行情绪识别
- 声学VAD阈值<500ms
- 支持打断的全双工交互
"""
import os
import sys
import asyncio
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

from voice_dialog.websocket_server import run_server


if __name__ == "__main__":
    print("=" * 50)
    print("全双工语音对话系统 v3.0")
    print("=" * 50)
    print()
    print("功能特性:")
    print("  - Qwen ASR 17B 流式语音识别 (帧长20ms)")
    print("  - Qwen Omni Flash 流式语义VAD")
    print("  - Qwen Omni Flash 并行情绪识别")
    print("  - 声学VAD阈值<500ms")
    print("  - 多轮对话上下文管理")
    print("  - 工具调用 (天气、时间、音乐、设备控制等)")
    print("  - 支持打断的全双工交互")
    print()
    print("启动信息:")
    # 从环境变量读取 API Key
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    if api_key:
        print(f"  API Key: {api_key[:10]}...{api_key[-5:]}")
    else:
        print("  警告: 未配置 DASHSCOPE_API_KEY")
        print("  请在 .env 文件中配置:")
        print("  DASHSCOPE_API_KEY=your-api-key-here")
    print()

    print("启动服务器...")
    print("  WebSocket: ws://0.0.0.0:8765")
    print("  Web界面: http://localhost:8765")
    print()
    print("按 Ctrl+C 停止服务器")
    print()

    run_server()