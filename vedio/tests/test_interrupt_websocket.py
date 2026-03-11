"""
自动化打断时延测试 - 通过WebSocket模拟真实用户交互

测试场景：
1. 发送第一个请求 "帮我介绍下三国演义"
2. 等待TTS开始播报
3. 在播报过程中发送打断请求
4. 发送第二个请求 "帮我介绍下西游记"
5. 记录时延数据并分析结果
"""

import asyncio
import websockets
import json
import time
from datetime import datetime


class InterruptTest:
    """打断测试客户端"""

    def __init__(self):
        self.ws_url = "ws://localhost:8765/ws/test_client"
        self.results = []
        self.state_changes = []
        self.latency_updates = []
        self.first_response_received = False
        self.interrupt_sent = False

    async def run(self):
        """运行测试"""
        print("=" * 60)
        print("打断时延自动化测试")
        print("=" * 60)
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        async with websockets.connect(self.ws_url) as ws:
            # 注册消息处理器
            asyncio.create_task(self.message_handler(ws))

            # 等待连接建立
            await asyncio.sleep(0.5)

            # ========== 测试步骤 ==========
            print("\n[步骤1] 发送第一个请求: '帮我介绍下三国演义'")
            start_time = time.time() * 1000

            await ws.send(json.dumps({
                "type": "text",
                "text": "帮我介绍下三国演义"
            }))

            # 等待LLM响应开始
            print("[等待] 等待LLM处理和TTS播报...")
            await asyncio.sleep(5)  # 等待TTS开始播报

            # ========== 打断测试 ==========
            print("\n[步骤2] 发送打断信号")
            interrupt_time = time.time() * 1000

            await ws.send(json.dumps({"type": "interrupt"}))
            self.interrupt_sent = True

            # 等待打断生效
            await asyncio.sleep(0.5)

            # ========== 发送新请求 ==========
            print("\n[步骤3] 发送打断后的新请求: '帮我介绍下西游记'")

            await ws.send(json.dumps({
                "type": "text",
                "text": "帮我介绍下西游记，里面的孙悟空很厉害，请介绍下"
            }))

            # 等待第二个响应
            await asyncio.sleep(10)

            end_time = time.time() * 1000

            # ========== 输出结果 ==========
            print("\n" + "=" * 60)
            print("测试结果")
            print("=" * 60)

            print(f"\n总测试时间: {(end_time - start_time):.0f}ms")
            print(f"收到结果数: {len(self.results)}")
            print(f"状态变化数: {len(self.state_changes)}")
            print(f"时延更新数: {len(self.latency_updates)}")

            # 显示对话结果
            print("\n对话记录:")
            for i, result in enumerate(self.results):
                print(f"  {i+1}. 用户: '{result.get('text', '')[:30]}...'")
                print(f"     助手: '{result.get('response', '')[:50]}...'")
                print(f"     打断: {result.get('is_interrupt', False)}")

            # 显示时延数据
            if self.latency_updates:
                print("\n时延数据:")
                for data in self.latency_updates[-3:]:  # 最近3条
                    if data.get('metrics'):
                        metrics = data['metrics']
                        print(f"  - 总耗时: {metrics.get('total_latency', 0):.0f}ms")
                        print(f"  - ASR延迟: {metrics.get('first_asr_latency', 0):.0f}ms")
                        print(f"  - LLM耗时: {metrics.get('llm_latency', 0):.0f}ms")
                        print(f"  - TTS耗时: {metrics.get('tts_latency', 0):.0f}ms")

            # 显示状态变化
            print("\n状态变化流程:")
            for change in self.state_changes:
                print(f"  {change['old']} -> {change['new']}")

            # 重置
            await ws.send(json.dumps({"type": "reset"}))
            print("\n[完成] 系统已重置")

    async def message_handler(self, ws):
        """处理服务器消息"""
        async for message in ws:
            try:
                data = json.loads(message)
                msg_type = data.get("type", "")

                if msg_type == "result":
                    self.results.append(data.get("data", {}))
                    text = data.get("data", {}).get("text", "")[:30]
                    response = data.get("data", {}).get("response", "")[:30]
                    print(f"[收到结果] 用户: '{text}...' -> 助手: '{response}...'")

                elif msg_type == "state_change":
                    change = data.get("data", {})
                    self.state_changes.append(change)
                    print(f"[状态] {change.get('old_state', '')} -> {change.get('new_state', '')}")

                elif msg_type == "latency_update":
                    self.latency_updates.append(data.get("data", {}))

                elif msg_type == "partial_asr":
                    text = data.get("data", {}).get("text", "")
                    print(f"[ASR] '{text}'")

            except Exception as e:
                print(f"[错误] 处理消息失败: {e}")


async def test_voice_validity():
    """测试有效人声检测"""
    print("\n" + "=" * 60)
    print("有效人声检测测试")
    print("=" * 60)

    # 通过HTTP API获取时延统计
    import aiohttp

    async with aiohttp.ClientSession() as session:
        # 获取当前时延
        async with session.get("http://localhost:8765/latency/current") as resp:
            current = await resp.json()
            print(f"\n当前时延数据: {current}")

        # 获取时延统计
        async with session.get("http://localhost:8765/latency/stats") as resp:
            stats = await resp.json()
            print(f"\n时延统计:")
            print(f"  总句数: {stats.get('total_sentences', 0)}")
            print(f"  平均总耗时: {stats.get('avg_total_latency', 0):.2f}ms")
            print(f"  平均ASR延迟: {stats.get('avg_asr_latency', 0):.2f}ms")
            print(f"  平均LLM耗时: {stats.get('avg_llm_latency', 0):.2f}ms")
            print(f"  平均TTS耗时: {stats.get('avg_tts_latency', 0):.2f}ms")


async def main():
    """主函数"""
    test = InterruptTest()
    await test.run()

    # 测试有效人声检测
    await test_voice_validity()

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
    print("\n请查看日志文件: logs/voice_dialog_*.log")
    print("访问监控界面: http://localhost:8765/monitor")
    print("访问打断测试: http://localhost:8765/interrupt-test")


if __name__ == "__main__":
    asyncio.run(main())