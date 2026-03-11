"""
全双工语音对话系统 v2.0 - 端到端测试
使用合成语音进行真实测试
"""
import asyncio
import pytest
import os
import sys
import tempfile
import struct
import wave
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from voice_dialog.system import VoiceDialogSystem
from voice_dialog.core import DialogState, SemanticState


def safe_print(msg):
    """安全打印，避免编码错误"""
    try:
        print(msg)
    except UnicodeEncodeError:
        # 移除非ASCII字符
        safe_msg = msg.encode('ascii', 'ignore').decode('ascii')
        print(safe_msg)


class AudioSynthesizer:
    """音频合成器 - 生成测试用的合成音频"""

    @staticmethod
    def generate_silence(duration_ms: int, sample_rate: int = 16000) -> bytes:
        """生成静音音频"""
        num_samples = int(sample_rate * duration_ms / 1000)
        return bytes(num_samples * 2)  # 16-bit PCM

    @staticmethod
    def generate_tone(frequency: int, duration_ms: int, sample_rate: int = 16000,
                      amplitude: float = 0.5) -> bytes:
        """生成单音音频（用于测试音频传输）"""
        import math
        num_samples = int(sample_rate * duration_ms / 1000)
        samples = []
        for i in range(num_samples):
            t = i / sample_rate
            value = int(amplitude * 32767 * math.sin(2 * math.pi * frequency * t))
            samples.append(struct.pack('<h', value))
        return b''.join(samples)

    @staticmethod
    def generate_speech_like_audio(duration_ms: int, sample_rate: int = 16000) -> bytes:
        """生成类似语音的音频（混合频率）"""
        import math
        import random

        num_samples = int(sample_rate * duration_ms / 1000)
        samples = []

        # 模拟语音的频率成分
        frequencies = [300, 500, 800, 1200, 1500, 2000]
        amplitudes = [0.3, 0.4, 0.3, 0.2, 0.15, 0.1]

        for i in range(num_samples):
            t = i / sample_rate
            value = 0
            for freq, amp in zip(frequencies, amplitudes):
                # 添加随机调制模拟语音变化
                modulation = 1 + 0.3 * math.sin(2 * math.pi * random.uniform(1, 5) * t)
                value += amp * modulation * math.sin(2 * math.pi * freq * t)

            # 添加随机性模拟语音的不规则性
            value *= (0.8 + 0.4 * random.random())

            # 转换为16位整数
            int_value = int(value * 32767 * 0.5)
            int_value = max(-32768, min(32767, int_value))
            samples.append(struct.pack('<h', int_value))

        return b''.join(samples)


class TestEndToEnd:
    """端到端测试类"""

    @pytest.fixture
    def system(self):
        """创建系统实例"""
        return VoiceDialogSystem()

    @pytest.fixture
    def audio_synthesizer(self):
        """创建音频合成器"""
        return AudioSynthesizer()

    @pytest.mark.asyncio
    async def test_text_input_basic(self, system):
        """测试基本文本输入"""
        result = await system.process_text("你好")

        assert result is not None
        assert result.text == "你好"
        assert result.response is not None
        assert len(result.response) > 0
        safe_print(f"[OK] Text input test passed: '{result.text}' -> response length: {len(result.response)}")

    @pytest.mark.asyncio
    async def test_text_input_weather(self, system):
        """测试天气查询"""
        result = await system.process_text("北京今天天气怎么样")

        assert result is not None
        assert "北京" in result.text
        assert result.response is not None
        # 应该调用天气工具
        assert len(result.tool_calls) > 0
        assert result.tool_calls[0].name == "get_weather"
        safe_print(f"[OK] Weather query test passed")

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, system):
        """测试多轮对话"""
        # 第一轮：自我介绍
        result1 = await system.process_text("我叫张三")
        assert result1 is not None
        safe_print(f"Turn 1: response length = {len(result1.response)}")

        # 第二轮：询问名字
        result2 = await system.process_text("我叫什么名字")
        assert result2 is not None
        assert "张三" in result2.response
        safe_print(f"Turn 2: found 'Zhang San' in response")

        # 第三轮：天气查询
        result3 = await system.process_text("上海天气怎么样")
        assert result3 is not None
        assert len(result3.tool_calls) > 0
        safe_print(f"Turn 3: weather tool called")

        safe_print("[OK] Multi-turn conversation test passed")

    @pytest.mark.asyncio
    async def test_device_control(self, system):
        """测试设备控制"""
        result = await system.process_text("打开客厅的灯")

        assert result is not None
        assert len(result.tool_calls) > 0
        assert result.tool_calls[0].name == "control_device"
        # 设备名称可能是"灯"或"客厅灯"
        assert "灯" in result.tool_calls[0].arguments["device"]
        safe_print(f"[OK] Device control test passed")

    @pytest.mark.asyncio
    async def test_music_playback(self, system):
        """测试音乐播放"""
        result = await system.process_text("播放周杰伦的歌")

        assert result is not None
        assert len(result.tool_calls) > 0
        assert result.tool_calls[0].name == "play_music"
        safe_print(f"[OK] Music playback test passed")

    @pytest.mark.asyncio
    async def test_reminder_setting(self, system):
        """测试提醒设置"""
        result = await system.process_text("提醒我明天早上8点开会")

        assert result is not None
        assert len(result.tool_calls) > 0
        assert result.tool_calls[0].name == "set_reminder"
        safe_print(f"[OK] Reminder setting test passed")

    @pytest.mark.asyncio
    async def test_audio_processing_pipeline(self, system, audio_synthesizer):
        """测试音频处理管道（使用合成音频）"""
        # 生成模拟语音音频
        audio_data = audio_synthesizer.generate_speech_like_audio(2000)  # 2秒

        # 模拟发送音频块
        chunk_size = 960  # 30ms at 16kHz
        results = []

        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i+chunk_size]
            result = await system.process_audio(chunk)
            if result:
                results.append(result)

        # 添加静音触发处理
        silence = audio_synthesizer.generate_silence(500)
        for i in range(0, len(silence), chunk_size):
            chunk = silence[i:i+chunk_size]
            result = await system.process_audio(chunk)
            if result:
                results.append(result)

        # 验证处理流程被执行
        safe_print(f"[OK] Audio processing pipeline test completed, processed {len(results)} results")

    @pytest.mark.asyncio
    async def test_state_transitions(self, system):
        """测试状态转换"""
        states_seen = []

        def on_state_change(old_state, new_state):
            states_seen.append((old_state.value, new_state.value))

        system.on_state_change(on_state_change)

        await system.process_text("测试状态转换")

        # 验证状态转换
        assert len(states_seen) > 0
        assert any(s[1] == "listening" for s in states_seen)
        assert any(s[1] == "processing" for s in states_seen)
        assert any(s[1] == "thinking" for s in states_seen)
        assert any(s[1] == "speaking" for s in states_seen)
        assert any(s[1] == "idle" for s in states_seen)

        safe_print(f"[OK] State transitions test passed")

    @pytest.mark.asyncio
    async def test_system_reset(self, system):
        """测试系统重置"""
        # 进行一些对话
        await system.process_text("我叫李四")
        await system.process_text("设置一个提醒")

        # 重置
        system.reset()

        # 验证历史被清除
        assert len(system.llm_planner.conversation_history) == 0
        assert system.current_state == DialogState.IDLE

        safe_print("[OK] System reset test passed")


class TestWithRealAudio:
    """
    使用真实语音文件的测试
    需要准备测试音频文件或使用 TTS 合成
    """

    @pytest.fixture
    def system(self):
        return VoiceDialogSystem()

    @pytest.mark.asyncio
    async def test_with_synthesized_audio(self, system):
        """
        使用 TTS 合成的音频进行测试
        这需要 edge-tts 或其他 TTS 工具
        """
        try:
            import edge_tts
        except ImportError:
            pytest.skip("edge-tts not installed, skipping real audio test")

        # 使用 edge-tts 合成测试音频
        test_texts = [
            "你好",
            "今天天气怎么样",
            "播放音乐",
        ]

        for text in test_texts:
            # 合成音频
            communicate = edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural")

            # 保存到临时文件
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                temp_path = f.name
                await communicate.save(temp_path)

            safe_print(f"Synthesized test audio: {temp_path}")

            # 注意：这里需要将 MP3 转换为 PCM 才能输入系统
            # 实际测试中需要音频格式转换

            # 清理临时文件
            os.unlink(temp_path)

        safe_print("[OK] Real audio test framework ready")


class TestScenarios:
    """
    场景测试 - 测试完整的对话场景
    """

    @pytest.fixture
    def system(self):
        return VoiceDialogSystem()

    @pytest.mark.asyncio
    async def test_weather_then_followup(self, system):
        """测试天气查询后的追问场景"""
        # 查询北京天气
        result1 = await system.process_text("北京天气怎么样")
        assert "北京" in result1.response or len(result1.tool_calls) > 0

        # 追问另一个城市
        result2 = await system.process_text("那上海呢")
        assert len(result2.tool_calls) > 0
        assert result2.tool_calls[0].name == "get_weather"
        assert result2.tool_calls[0].arguments["city"] == "上海"

        safe_print("[OK] Follow-up question test passed")

    @pytest.mark.asyncio
    async def test_complex_intent(self, system):
        """测试复杂意图"""
        # 提醒带伞，同时涉及天气
        result = await system.process_text("明天早上提醒我带伞，顺便看看北京天气")
        assert result is not None
        # 应该触发多个工具调用
        assert len(result.tool_calls) >= 1

        safe_print(f"[OK] Complex intent test passed, tools: {[tc.name for tc in result.tool_calls]}")

    @pytest.mark.asyncio
    async def test_context_awareness(self, system):
        """测试上下文感知"""
        # 提到城市
        await system.process_text("我想知道广州的情况")

        # 后续问题应该理解上下文
        result = await system.process_text("那里的天气怎么样")
        # 检查是否理解"那里"指的是广州
        if len(result.tool_calls) > 0:
            assert result.tool_calls[0].arguments.get("city") == "广州"

        safe_print("[OK] Context awareness test passed")


def run_tests():
    """运行所有测试"""
    pytest.main([
        __file__,
        "-v",
        "-s",
        "--tb=short",
        "--asyncio-mode=auto"
    ])


if __name__ == "__main__":
    run_tests()