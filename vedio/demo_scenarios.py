"""
全双工语音对话系统 v2.0 - 场景测试演示
模拟各种全双工对话场景
"""
import asyncio
import sys
import os

# 设置UTF-8编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from voice_dialog.core.logger import logger, HAS_LOGURU
from voice_dialog import VoiceDialogSystem, DialogState, EmotionType


# 配置日志
if HAS_LOGURU:
    logger.remove()
    logger.add(sys.stdout, format="<level>{message}</level>", level="INFO")


class ScenarioDemo:
    """场景演示类"""

    def __init__(self):
        self.system = VoiceDialogSystem()

    async def run_scenario(self, name: str, inputs: list):
        """运行单个场景"""
        print(f"\n{'='*60}")
        print(f"[场景] {name}")
        print('='*60)

        for i, text in enumerate(inputs, 1):
            print(f"\n[对话 {i}]")
            print(f"用户: {text}")

            result = await self.system.process_text(text)

            print(f"情绪: {result.emotion.value}")
            if result.tool_calls:
                print(f"工具: {[tc.name for tc in result.tool_calls]}")
            print(f"助手: {result.response}")

        # 重置系统
        self.system.reset()

    async def scenario_weather_query(self):
        """场景1: 天气查询"""
        await self.run_scenario(
            "天气查询",
            [
                "北京今天天气怎么样？",
                "那上海呢？",
                "谢谢！"
            ]
        )

    async def scenario_device_control(self):
        """场景2: 设备控制"""
        await self.run_scenario(
            "设备控制",
            [
                "帮我打开客厅的灯",
                "再把空调打开",
                "温度调到26度"
            ]
        )

    async def scenario_reminder(self):
        """场景3: 设置提醒"""
        await self.run_scenario(
            "设置提醒",
            [
                "帮我设置一个明天早上9点的提醒",
                "提醒内容是开会",
                "好的，谢谢"
            ]
        )

    async def scenario_emotion_response(self):
        """场景4: 情绪化对话"""
        await self.run_scenario(
            "情绪化对话",
            [
                "太好了！终于完成了！",
                "这个功能太糟糕了，根本不能用",
                "烦死了，怎么又出问题了",
                "好吧，我知道了"
            ]
        )

    async def scenario_multi_intent(self):
        """场景5: 多意图对话"""
        await self.run_scenario(
            "多意图对话",
            [
                "查一下北京天气，然后帮我设个提醒明天带伞",
                "顺便播放一首轻松的音乐",
                "把客厅灯关掉"
            ]
        )

    async def scenario_interrupt_simulation(self):
        """场景6: 打断模拟"""
        print(f"\n{'='*60}")
        print("[场景] 打断模拟")
        print('='*60)

        print("\n[模拟AI说话时被打断]")

        # 模拟进入SPEAKING状态
        await self.system.dialog_state.transition_to(DialogState.LISTENING)
        await self.system.dialog_state.transition_to(DialogState.PROCESSING)
        await self.system.dialog_state.transition_to(DialogState.THINKING)
        await self.system.dialog_state.transition_to(DialogState.SPEAKING)

        print(f"当前状态: {self.system.current_state.value}")
        print("AI正在说话...")

        # 触发打断
        self.system.interrupt()
        await asyncio.sleep(0.1)

        print(f"打断后状态: {self.system.current_state.value}")
        print("[OK] 打断成功，可以继续对话")

        # 继续对话
        result = await self.system.process_text("我想问个问题")
        print(f"助手: {result.response}")

        self.system.reset()

    async def scenario_context_continuation(self):
        """场景7: 上下文延续"""
        await self.run_scenario(
            "上下文延续对话",
            [
                "我叫小明",
                "我的名字是什么？",
                "帮我查一下深圳天气",
                "那里冷不冷？"
            ]
        )

    async def run_all(self):
        """运行所有场景"""
        print("""
    ============================================================
         全双工语音对话系统 v2.0 - 场景测试演示
    ============================================================
        """)

        await self.scenario_weather_query()
        await self.scenario_device_control()
        await self.scenario_reminder()
        await self.scenario_emotion_response()
        await self.scenario_multi_intent()
        await self.scenario_interrupt_simulation()
        await self.scenario_context_continuation()

        print(f"\n{'='*60}")
        print("[OK] 所有场景测试完成！")
        print('='*60)


async def main():
    demo = ScenarioDemo()
    await demo.run_all()


if __name__ == "__main__":
    asyncio.run(main())