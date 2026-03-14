"""
全双工语音对话系统 v3.2 - WebSocket服务器
支持全双工语音交互

v3.2 特性：
- 流式ASR + 流式语义VAD
- 并行情绪识别
- 打断支持
- 实时时延监控
- LLM流式输出
"""
import asyncio
import json
import base64
from typing import Dict, Set, Callable
from .core.logger import logger

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

from .system import VoiceDialogSystem
from .core import DialogState, DialogResult, latency_tracker


app = FastAPI(title="全双工语音对话系统 v3.2")


class ConnectionManager:
    """WebSocket连接管理器"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.dialog_systems: Dict[str, VoiceDialogSystem] = {}
        # 追踪发送失败次数，用于判断连接是否真正断开
        self._send_failures: Dict[str, int] = {}
        self._max_send_failures = 5  # 连续失败5次认为连接断开（提高容错）
        # 发送锁，防止并发写入冲突
        self._send_locks: Dict[str, asyncio.Lock] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        # 为每个连接创建发送锁
        self._send_locks[client_id] = asyncio.Lock()

        # 为每个连接创建独立的对话系统
        system = VoiceDialogSystem()
        self.dialog_systems[client_id] = system

        # 注册回调 - 简单直接的方式
        system.on_result(lambda r: asyncio.create_task(
            self.send_result(client_id, r)
        ))
        system.on_state_change(lambda old, new: asyncio.create_task(
            self.send_state_change(client_id, old, new)
        ))
        system.on_partial_asr(lambda text: asyncio.create_task(
            self.send_partial_asr(client_id, text)
        ))
        system.on_tool_executing(lambda tool_name, tool_args: asyncio.create_task(
            self.send_tool_executing(client_id, tool_name, tool_args)
        ))
        system.on_latency_update(lambda data: asyncio.create_task(
            self.send_latency_update(client_id, data)
        ))
        # 注册LLM流式输出回调
        system.on_llm_chunk(lambda chunk: asyncio.create_task(
            self.send_llm_chunk(client_id, chunk)
        ))
        # 注册TTS音频块回调（实时播报）
        system.on_audio_chunk(lambda audio: asyncio.create_task(
            self.send_audio_chunk(client_id, audio)
        ))
        # 注册清空音频回调（新问题开始时）
        system.on_clear_audio(lambda: asyncio.create_task(
            self.send_clear_audio(client_id)
        ))

        logger.info(f"客户端连接: {client_id}")

    def disconnect(self, client_id: str):
        """断开客户端连接，清理资源"""
        # 先标记需要断开
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"WebSocket连接已移除: {client_id}")

        # 清理发送失败计数
        if client_id in self._send_failures:
            del self._send_failures[client_id]

        # 清理发送锁
        if client_id in self._send_locks:
            del self._send_locks[client_id]

        # 停止并清理对话系统
        if client_id in self.dialog_systems:
            system = self.dialog_systems[client_id]
            # 重置系统，停止所有后台任务
            try:
                system.reset()
            except Exception as e:
                logger.warning(f"重置系统失败: {e}")
            del self.dialog_systems[client_id]
            logger.info(f"对话系统已清理: {client_id}")

    async def _check_and_disconnect(self, client_id: str):
        """检查是否需要断开连接（内部方法）"""
        failures = self._send_failures.get(client_id, 0)
        if failures >= self._max_send_failures:
            logger.warning(f"[连接] 客户端 {client_id} 连续发送失败 {failures} 次，主动断开")
            # 停止后台任务（使用 asyncio.create_task 避免阻塞）
            if client_id in self.dialog_systems:
                try:
                    asyncio.create_task(self.dialog_systems[client_id].interrupt())
                except:
                    pass
            # 清理连接（不调用 websocket.close()，因为连接已经不可用）
            self.disconnect(client_id)
            return True
        return False

    async def _send_json(self, client_id: str, data: dict) -> bool:
        """
        发送JSON消息（线程安全，使用锁防止并发冲突）

        Returns:
            bool: 发送是否成功
        """
        if client_id not in self.active_connections:
            return False

        # 检查数据有效性
        if not data:
            logger.debug(f"[发送] 跳过空数据: {client_id}")
            return True  # 空数据视为成功，不计数

        # 获取发送锁
        lock = self._send_locks.get(client_id)
        if not lock:
            return False

        async with lock:
            # 再次检查连接是否仍然有效
            if client_id not in self.active_connections:
                return False

            try:
                await self.active_connections[client_id].send_json(data)
                # 发送成功，重置失败计数
                self._send_failures[client_id] = 0
                return True
            except Exception as e:
                # 发送失败，增加计数
                self._send_failures[client_id] = self._send_failures.get(client_id, 0) + 1
                # 记录详细的异常信息
                import traceback
                error_type = type(e).__name__
                error_msg = str(e) if str(e) else "(无错误消息)"

                # AssertionError 是临时并发冲突，不记录堆栈，只记录警告
                if error_type == "AssertionError":
                    logger.debug(f"发送消息临时失败 (AssertionError): {client_id}")
                else:
                    logger.warning(f"发送消息失败 ({self._send_failures[client_id]}次): {client_id}, "
                                  f"类型: {error_type}, 错误: {error_msg}")
                    logger.debug(f"异常堆栈:\n{traceback.format_exc()}")

                # 检查是否需要断开
                await self._check_and_disconnect(client_id)
                return False

    async def send_json(self, client_id: str, data: dict) -> bool:
        """
        发送JSON消息（带重试机制）

        Args:
            client_id: 客户端ID
            data: 要发送的数据

        Returns:
            bool: 发送是否成功
        """
        max_retries = 3
        retry_delay = 0.1  # 重试延迟（秒）

        for attempt in range(max_retries):
            success = await self._send_json(client_id, data)
            if success:
                return True

            # 如果连接已断开，不再重试
            if client_id not in self.active_connections:
                return False

            # 最后一次尝试失败后不再等待
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)

        return False

    async def send_result(self, client_id: str, result: DialogResult):
        """发送对话结果 - 只发送LLM总结后的响应，不发送音频（音频已通过audio_chunk实时发送）"""
        data = {
            "type": "result",
            "data": {
                "text": result.text,
                "text_confidence": result.text_confidence,
                "semantic_state": result.semantic_state.value,
                "emotion": result.emotion.value,
                "emotion_confidence": result.emotion_confidence,
                "response": result.response,  # 这是LLM总结后的响应
                "is_interrupt": result.is_interrupt,
                "dialog_state": result.dialog_state.value,
                # 只显示工具名称，不返回执行结果
                "tool_calls": [{"name": tc.name} for tc in result.tool_calls],
                # 不发送音频，因为已通过audio_chunk实时发送
                "has_audio": False,
                # 大模型情绪
                "llm_emotion": result.llm_emotion.value
            }
        }

        # 流式TTS模式下不发送完整音频，避免重复播放
        # 音频已通过 send_audio_chunk 实时发送

        await self.send_json(client_id, data)

    async def send_llm_chunk(self, client_id: str, chunk: str):
        """发送LLM流式输出块"""
        # 检查数据有效性（提前过滤空消息）
        if not chunk or not chunk.strip():
            return

        # 检查连接是否存在
        if client_id not in self.active_connections:
            return

        await self.send_json(client_id, {
            "type": "llm_chunk",
            "data": {"text": chunk}
        })

    async def send_audio_chunk(self, client_id: str, audio_data: bytes):
        """发送TTS音频块（实时播报）"""
        # 检查音频数据有效性（提前过滤空消息）
        if not audio_data or len(audio_data) == 0:
            logger.debug(f"[音频流] 收到空音频数据，跳过发送: {client_id}")
            return

        # 检查连接是否存在
        if client_id not in self.active_connections:
            return

        await self.send_json(client_id, {
            "type": "audio_chunk",
            "data": {"audio": base64.b64encode(audio_data).decode()}
        })

    async def send_clear_audio(self, client_id: str):
        """发送清空音频队列消息"""
        await self.send_json(client_id, {
            "type": "clear_audio"
        })
        logger.info(f"[音频流] 已通知前端清空音频队列: {client_id}")

    async def send_tool_executing(self, client_id: str, tool_name: str, tool_args: dict):
        """发送工具执行状态"""
        await self.send_json(client_id, {
            "type": "tool_executing",
            "data": {
                "tool_name": tool_name,
                "tool_args": tool_args
            }
        })

    async def send_state_change(self, client_id: str, old_state: DialogState, new_state: DialogState):
        """发送状态变化"""
        await self.send_json(client_id, {
            "type": "state_change",
            "data": {
                "old_state": old_state.value,
                "new_state": new_state.value
            }
        })

    async def send_partial_asr(self, client_id: str, text: str):
        """发送部分ASR结果"""
        # 检查数据有效性
        if not text or not text.strip():
            return
        await self.send_json(client_id, {
            "type": "partial_asr",
            "data": {"text": text}
        })

    async def send_latency_update(self, client_id: str, data):
        """发送时延更新"""
        if data is None:
            return
        await self.send_json(client_id, {
            "type": "latency_update",
            "data": data.to_dict() if hasattr(data, 'to_dict') else data
        })

    def get_system(self, client_id: str) -> VoiceDialogSystem:
        return self.dialog_systems.get(client_id)


manager = ConnectionManager()


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)

    try:
        while True:
            data = await websocket.receive()

            if "text" in data:
                # JSON消息
                message = json.loads(data["text"])
                await handle_message(client_id, message)

            elif "bytes" in data:
                # 音频数据 (PCM 16kHz 16bit mono)
                audio_chunk = data["bytes"]
                await handle_audio(client_id, audio_chunk)

    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
        manager.disconnect(client_id)


async def handle_message(client_id: str, message: dict):
    """处理JSON消息"""
    msg_type = message.get("type", "")
    system = manager.get_system(client_id)

    if not system:
        return

    if msg_type == "text":
        # 文本输入
        text = message.get("text", "")
        if text:
            result = await system.process_text(text)
            # 结果会通过回调发送

    elif msg_type == "interrupt":
        # 打断
        await system.interrupt()

    elif msg_type == "reset":
        # 重置
        system.reset()
        await manager.send_json(client_id, {"type": "reset", "data": {"success": True}})

    elif msg_type == "ping":
        # 心跳
        await manager.send_json(client_id, {"type": "pong"})


async def handle_audio(client_id: str, audio_chunk: bytes):
    """
    处理音频数据 - v3.5 全双工非阻塞模式

    核心改进：
    - 音频接收与处理解耦
    - WebSocket 主循环不再被 LLM 推理阻塞
    - 支持在 THINKING 状态下实时打断
    """
    system = manager.get_system(client_id)

    if not system:
        return

    # 将音频放入队列（非阻塞），后台任务持续处理
    await system.process_audio(audio_chunk)
    # 结果通过回调异步发送，不阻塞 WebSocket 主循环


@app.get("/")
async def get_index():
    """返回主页"""
    return FileResponse("web/index.html")


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "version": "3.5"}


@app.get("/latency/history")
async def get_latency_history(limit: int = 10):
    """获取时延历史记录"""
    history = latency_tracker.get_history(limit)
    return {
        "history": [h.to_dict() for h in history],
        "total": len(history)
    }


@app.get("/latency/stats")
async def get_latency_stats():
    """获取时延统计信息"""
    return latency_tracker.get_stats()


@app.get("/latency/current")
async def get_current_latency():
    """获取当前时延数据"""
    current = latency_tracker.get_current()
    if current:
        return current.to_dict()
    return {"status": "no_active_sentence"}


@app.get("/monitor")
async def get_monitor():
    """返回时延监控页面"""
    return FileResponse("web/latency_monitor.html")


@app.get("/interrupt-test")
async def get_interrupt_test():
    """返回打断测试页面"""
    return FileResponse("web/interrupt_test.html")


def run_server(host: str = "0.0.0.0", port: int = 8765):
    """启动服务器"""
    from .core.config import get_config
    config = get_config()

    server_config = config.server
    host = server_config.get("host", host)
    port = server_config.get("port", port)

    logger.info(f"启动WebSocket服务器: ws://{host}:{port}")

    # 增加 WebSocket ping 间隔和超时，避免 TTS 处理期间连接断开
    # ping_interval: 每 30 秒发送一次 ping
    # ping_timeout: 等待 pong 响应的超时时间为 60 秒
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        ws_ping_interval=30.0,  # WebSocket ping 间隔
        ws_ping_timeout=60.0    # WebSocket pong 超时
    )


if __name__ == "__main__":
    run_server()