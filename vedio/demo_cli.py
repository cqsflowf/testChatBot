"""
全双工语音对话系统 v2.0 - 命令行演示
无需浏览器，直接在命令行中体验
"""
import asyncio
import sys
import os

# 设置UTF-8编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from voice_dialog.core.logger import logger, HAS_LOGURU

# 配置日志
if HAS_LOGURU:
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level} | {message}", level="INFO")


async def main():
    """命令行演示"""
    from voice_dialog import VoiceDialogSystem, DialogState

    print("""
    ============================================================
         全双工语音对话系统 v2.0 - 命令行演示
    ============================================================
      命令:
      - 直接输入文字进行对话
      - 输入 'quit' 或 'exit' 退出
      - 输入 'reset' 重置对话
      - 输入 'status' 查看状态
    ============================================================
    """)

    # 初始化系统
    system = VoiceDialogSystem()
    print("系统初始化完成!\n")

    while True:
        try:
            # 获取用户输入
            user_input = input("用户: ").strip()

            if not user_input:
                continue

            # 命令处理
            if user_input.lower() in ['quit', 'exit']:
                print("\n再见!")
                break

            if user_input.lower() == 'reset':
                system.reset()
                print("[OK] 对话已重置\n")
                continue

            if user_input.lower() == 'status':
                print(f"状态: {system.current_state.value}")
                print(f"历史: {len(system.llm_planner.conversation_history)} 条\n")
                continue

            # 处理对话
            print("处理中...")
            result = await system.process_text(user_input)

            # 显示结果
            print(f"\n识别: {result.text}")
            print(f"情绪: {result.emotion.value} (置信度: {result.emotion_confidence:.2f})")

            if result.tool_calls:
                print(f"工具: {[tc.name for tc in result.tool_calls]}")
                for tr in result.tool_results:
                    print(f"   - {tr.tool_call.name}: {tr.result}")

            print(f"\n助手: {result.response}\n")

        except KeyboardInterrupt:
            print("\n\n再见!")
            break
        except Exception as e:
            print(f"错误: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())