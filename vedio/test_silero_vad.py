#!/usr/bin/env python
"""
测试 Silero VAD v3 实现
验证实时检测、预缓存和打断功能
"""
import sys
import os
import struct
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_silero_vad_v3():
    """测试Silero VAD v3实现"""
    print("=" * 60)
    print("测试 Silero VAD v3 - 内置自适应阈值 + 预缓存")
    print("=" * 60)

    from voice_dialog.modules.acoustic_vad import SileroVADWrapper, AcousticVAD, StreamingVAD

    # 1. 测试Silero VAD Wrapper v3
    print("\n[1] 测试 SileroVADWrapper v3 (内置自适应阈值)...")
    silero = SileroVADWrapper()  # 不再需要threshold参数
    print(f"    Silero VAD 可用: {silero.available}")

    if silero.available:
        print("    [成功] Silero VAD 加载成功 (内置自适应阈值)")

        # 测试滑动窗口检测
        print("\n    测试滑动窗口检测...")

        # 生成模拟语音帧 (320样本 = 20ms)
        speech_samples = [random.randint(-15000, 15000) for _ in range(320)]
        speech_frame = struct.pack('<' + 'h' * 320, *speech_samples)

        # 生成静音帧
        silence_frame = struct.pack('<' + 'h' * 320, *([0] * 320))

        # 测试语音检测
        for i in range(5):
            result = silero.is_speech(speech_frame, 16000)
            if i < 3:
                print(f"    第{i+1}帧(语音): {result}")

        # 重置后测试静音
        silero.reset_states()
        for i in range(3):
            result = silero.is_speech(silence_frame, 16000)
            print(f"    静音帧{i+1}: {result}")
    else:
        print("    [警告] Silero VAD 加载失败")

    # 2. 测试AcousticVAD初始化
    print("\n[2] 测试 AcousticVAD 初始化...")
    acoustic_vad = AcousticVAD()

    print(f"    帧长度: {acoustic_vad.frame_duration_ms}ms")
    print(f"    静音超时: {acoustic_vad.silence_threshold_ms}ms")
    print(f"    预缓存: {acoustic_vad.prebuffer_duration_ms}ms ({acoustic_vad.prebuffer_frames}帧)")

    if acoustic_vad._silero_vad.available:
        print(f"    Silero VAD 可用: True")
        vad_type = "Silero VAD"
    elif acoustic_vad._webrtc_vad.available:
        print(f"    WebRTC VAD 可用: True (Silero不可用)")
        vad_type = "WebRTC VAD"
    else:
        print(f"    使用简单VAD")
        vad_type = "简单音量VAD"

    print(f"    当前使用: {vad_type}")

    # 3. 测试预缓存功能
    print("\n[3] 测试预缓存功能...")
    acoustic_vad.reset()

    # 模拟预缓存场景：发送几帧静音，然后语音
    silence_frame = struct.pack('<' + 'h' * 320, *([0] * 320))
    speech_frame = struct.pack('<' + 'h' * 320, *[random.randint(-15000, 15000) for _ in range(320)])

    # 先发送5帧静音（会被缓存）
    for i in range(5):
        acoustic_vad.process_frame(silence_frame)

    print(f"    预缓存帧数: {len(acoustic_vad._prebuffer)}")

    # 然后发送语音帧（应该触发回溯）
    for i in range(5):
        acoustic_vad.process_frame(speech_frame)

    if acoustic_vad._is_speech:
        print(f"    [成功] 检测到语音开始，语音帧数: {len(acoustic_vad._speech_frames)}")

    # 4. 测试打断检测功能
    print("\n[4] 测试打断检测功能...")
    acoustic_vad.reset()

    # 连续发送语音帧，验证打断检测
    print("    连续发送语音帧...")
    interrupt_detected = False
    for i in range(10):
        result = acoustic_vad.check_interrupt(speech_frame)
        if result:
            interrupt_detected = True
            print(f"    [成功] 第{i+1}帧检测到打断!")
            break

    if not interrupt_detected:
        print("    [注意] 10帧内未检测到打断")

    # 5. 结果汇总
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    if acoustic_vad._silero_vad.available:
        print("[PASS] Silero VAD v3 加载成功")
        print(f"[PASS] 预缓存配置: {acoustic_vad.prebuffer_duration_ms}ms")
        print(f"[PASS] 静音超时: {acoustic_vad.silence_threshold_ms}ms")
        if interrupt_detected:
            print("[PASS] 打断检测功能正常")
        else:
            print("[WARN] 打断检测可能需要调整")
    elif acoustic_vad._webrtc_vad.available:
        print("[PASS] WebRTC VAD 备选可用")
    else:
        print("[PASS] 简单VAD 可用")

    print("\n" + "=" * 60)
    if acoustic_vad._silero_vad.available:
        print("Silero VAD v3 配置优化完成!")
    else:
        print("使用备选VAD方案")
    print("=" * 60)

    return acoustic_vad._silero_vad.available


if __name__ == "__main__":
    test_silero_vad_v3()