"""
全双工语音对话系统 - 打断时延测试

测试场景：
1. 用户语音输入 "帮我介绍下三国演义"
2. 系统开始语音播报
3. 播放约5秒后，用户再次输入 "帮我介绍下西游记，里面的孙悟空很厉害，请介绍下"
4. 验证：用户输入前几个字时，系统应立即识别有效VAD并打断TTS

预期结果：
- 有效人声判断延迟 < 200ms
- 打断响应时间 < 500ms
"""

import asyncio
import time
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.voice_dialog.core.logger import logger
from src.voice_dialog.core.latency import latency_tracker
from src.voice_dialog.system import VoiceDialogSystem
from src.voice_dialog.core import DialogState


class InterruptLatencyTest:
    """打断时延测试"""

    def __init__(self):
        self.system = VoiceDialogSystem()
        self.results = []
        self.current_result = None
        self.tts_start_time = None
        self.interrupt_detected_time = None
        self.interrupt_text = ""

        # 注册回调
        self.system.on_result(self._on_result)
        self.system.on_state_change(self._on_state_change)
        self.system.on_latency_update(self._on_latency_update)

    def _on_result(self, result):
        """结果回调"""
        self.results.append({
            "text": result.text,
            "response": result.response[:100] if result.response else "",
            "is_interrupt": result.is_interrupt
        })
        logger.info(f"[测试] 收到结果: '{result.text[:30]}...' -> '{result.response[:30] if result.response else ''}...'")

    def _on_state_change(self, old_state, new_state):
        """状态变化回调"""
        logger.info(f"[测试] 状态变化: {old_state.value} -> {new_state.value}")

        if new_state == DialogState.SPEAKING:
            self.tts_start_time = time.time() * 1000
            logger.info(f"[测试] TTS开始播报，时间: {self.tts_start_time:.0f}ms")

    def _on_latency_update(self, data):
        """时延更新回调"""
        if data:
            logger.debug(f"[测试] 时延更新: {data.text[:20] if data.text else ''}...")

    async def run_test(self):
        """运行测试"""
        logger.info("=" * 60)
        logger.info("开始打断时延测试")
        logger.info("=" * 60)

        # 测试场景1：正常对话流程
        logger.info("\n--- 测试场景1：正常对话 ---")
        await self._test_normal_dialog()

        # 等待系统空闲
        await asyncio.sleep(2)

        # 测试场景2：打断测试
        logger.info("\n--- 测试场景2：打断测试 ---")
        await self._test_interrupt()

        # 输出测试结果
        self._print_results()

    async def _test_normal_dialog(self):
        """测试正常对话流程"""
        logger.info("测试文本输入: '帮我介绍下三国演义'")

        result = await self.system.process_text("帮我介绍下三国演义")

        if result:
            logger.info(f"收到响应: '{result.response[:100]}...'")
            logger.info(f"是否有工具调用: {len(result.tool_calls) > 0}")

        # 等待TTS播报一段时间
        await asyncio.sleep(2)

    async def _test_interrupt(self):
        """测试打断功能"""
        logger.info("模拟打断场景...")

        # 先发起一个长回复请求
        logger.info("发起第一个请求: '帮我介绍下三国演义'")
        task1 = asyncio.create_task(self.system.process_text("帮我介绍下三国演义"))

        # 等待TTS开始播报
        await asyncio.sleep(1)

        # 模拟打断场景
        # 在播报过程中，模拟用户输入新的语音
        logger.info("TTS正在播报，模拟用户打断输入...")

        # 记录打断开始时间
        interrupt_start_time = time.time() * 1000

        # 模拟用户输入 "帮我介绍下西游记"
        # 这里我们测试有效VAD判断的延迟
        test_text = "帮我介绍下西游记，里面的孙悟空很厉害，请介绍下"

        # 分块模拟流式输入
        chunk_size = 3  # 每次输入3个字符
        for i in range(0, len(test_text), chunk_size):
            chunk = test_text[i:i+chunk_size]
            logger.info(f"[模拟ASR] 流式输出: '{chunk}'")

            # 检查是否触发有效VAD
            from src.voice_dialog.modules.semantic_vad import VoiceValidity, SemanticVADProcessor
            processor = SemanticVADProcessor()
            validity = processor.check_voice_validity(test_text[:i+chunk_size])

            if validity == VoiceValidity.VALID:
                interrupt_detected_time = time.time() * 1000
                delay = interrupt_detected_time - interrupt_start_time
                logger.info(f"[打断检测] 有效人声判断! 延迟: {delay:.0f}ms, 文本: '{test_text[:i+chunk_size]}'")

                # 触发打断
                self.system.interrupt()
                break

            await asyncio.sleep(0.1)  # 模拟ASR输出间隔

        # 等待第一个任务完成
        try:
            await asyncio.wait_for(task1, timeout=5)
        except asyncio.TimeoutError:
            logger.warning("第一个任务超时")

        # 发起新的请求
        await asyncio.sleep(0.5)
        logger.info("发起打断后的新请求: '帮我介绍下西游记'")
        result = await self.system.process_text("帮我介绍下西游记，里面的孙悟空很厉害，请介绍下")

        if result:
            logger.info(f"打断后响应: '{result.response[:100]}...'")

    def _print_results(self):
        """打印测试结果"""
        logger.info("\n" + "=" * 60)
        logger.info("测试结果汇总")
        logger.info("=" * 60)

        # 获取时延统计
        stats = latency_tracker.get_stats()
        logger.info(f"总句数: {stats['total_sentences']}")
        logger.info(f"平均总耗时: {stats['avg_total_latency']:.2f}ms")
        logger.info(f"平均ASR延迟: {stats['avg_asr_latency']:.2f}ms")
        logger.info(f"平均LLM耗时: {stats['avg_llm_latency']:.2f}ms")
        logger.info(f"平均TTS耗时: {stats['avg_tts_latency']:.2f}ms")

        # 获取历史记录
        history = latency_tracker.get_history(10)
        logger.info(f"\n最近{len(history)}条记录:")
        for i, record in enumerate(history):
            logger.info(f"  {i+1}. '{record.text[:30]}...' - 总耗时: {record.total_latency:.0f}ms")

        # 打印对话结果
        logger.info(f"\n对话结果数量: {len(self.results)}")
        for i, result in enumerate(self.results):
            logger.info(f"  {i+1}. 用户: '{result['text'][:30]}...'")
            logger.info(f"      助手: '{result['response'][:30]}...'")
            logger.info(f"      打断: {result['is_interrupt']}")

        logger.info("\n" + "=" * 60)
        logger.info("测试完成")
        logger.info("=" * 60)


async def test_voice_validity_detection():
    """测试有效人声检测"""
    from src.voice_dialog.modules.semantic_vad import VoiceValidity, SemanticVADProcessor

    processor = SemanticVADProcessor()

    test_cases = [
        ("嗯", VoiceValidity.FILLER),
        ("啊", VoiceValidity.FILLER),
        ("帮我", VoiceValidity.VALID),
        ("帮我介绍", VoiceValidity.VALID),
        ("什么", VoiceValidity.VALID),
        ("停", VoiceValidity.VALID),
        ("换一个", VoiceValidity.VALID),
        ("嗯啊", VoiceValidity.FILLER),
        ("那个", VoiceValidity.FILLER),
    ]

    logger.info("\n有效人声检测测试:")
    logger.info("-" * 40)

    all_passed = True
    for text, expected in test_cases:
        result = processor.check_voice_validity(text)
        passed = result == expected
        status = "PASS" if passed else "FAIL"
        logger.info(f"  '{text}' -> {result} (期望: {expected}) [{status}]")
        if not passed:
            all_passed = False

    logger.info("-" * 40)
    if all_passed:
        logger.info("所有测试用例通过!")
    else:
        logger.warning("部分测试用例失败!")

    return all_passed


async def main():
    """主函数"""
    # 先运行有效人声检测测试
    logger.info("运行有效人声检测测试...")
    validity_passed = await test_voice_validity_detection()

    logger.info("\n" + "=" * 60)

    # 运行打断时延测试
    test = InterruptLatencyTest()
    await test.run_test()


if __name__ == "__main__":
    asyncio.run(main())