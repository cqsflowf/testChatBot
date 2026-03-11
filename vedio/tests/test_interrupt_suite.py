"""
打断时延测试套件

测试场景（来自测试用例文档 tests/test_cases_interrupt.md）：
1. 用户语音输入 "帮我介绍下三国演义"
2. 系统执行并进行语音播报
3. 播报约5秒后，用户再次输入 "帮我介绍下西游记，里面的孙悟空很厉害，请介绍下"
4. 系统应在用户输入前几个字时判断为有效VAD并立即打断TTS

性能指标要求：
- 有效人声判断延迟 < 200ms
- 打断响应时间 < 500ms
- 语气助词不触发打断
- 关键词立即触发有效判断

运行方式：
    python tests/test_interrupt_suite.py
"""

import asyncio
import sys
import os
import time
from datetime import datetime
from typing import List, Dict, Tuple

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.voice_dialog.core.logger import logger
from src.voice_dialog.core.latency import latency_tracker
from src.voice_dialog.modules.semantic_vad import VoiceValidity, SemanticVADProcessor


class TestResult:
    """测试结果"""
    def __init__(self, case_id: str, name: str):
        self.case_id = case_id
        self.name = name
        self.passed = False
        self.actual = None
        self.expected = None
        self.duration_ms = 0
        self.error = None

    def to_dict(self) -> Dict:
        return {
            "case_id": self.case_id,
            "name": self.name,
            "passed": self.passed,
            "actual": self.actual,
            "expected": self.expected,
            "duration_ms": self.duration_ms,
            "error": self.error
        }


class InterruptTestSuite:
    """打断时延测试套件"""

    # 性能指标
    VALID_VOICE_LATENCY_LIMIT_MS = 200  # 有效人声判断延迟上限
    INTERRUPT_LATENCY_LIMIT_MS = 500    # 打断响应延迟上限

    def __init__(self):
        self.results: List[TestResult] = []
        self.processor = SemanticVADProcessor()

    def run_all(self):
        """运行所有测试"""
        print("=" * 70)
        print("打断时延测试套件")
        print("=" * 70)
        print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # TC-001 ~ TC-010: 有效人声检测测试
        self._test_valid_voice_keywords()
        self._test_filler_words()
        self._test_mixed_content()

        # 输出结果
        self._print_summary()

        return self._all_passed()

    def _test_valid_voice_keywords(self):
        """测试有效人声关键词"""
        test_cases = [
            ("TC-001", "帮我", VoiceValidity.VALID),
            ("TC-002", "我要", VoiceValidity.VALID),
            ("TC-003", "什么", VoiceValidity.VALID),
            ("TC-004", "停", VoiceValidity.VALID),
            ("TC-005", "帮我介绍", VoiceValidity.VALID),
            ("TC-006", "换一个", VoiceValidity.VALID),
        ]

        print("\n[测试组1] 有效人声关键词检测")
        print("-" * 50)

        for case_id, text, expected in test_cases:
            result = self._run_validity_test(case_id, text, expected)
            self.results.append(result)

    def _test_filler_words(self):
        """测试语气助词"""
        test_cases = [
            ("TC-007", "嗯", VoiceValidity.FILLER),
            ("TC-008", "啊", VoiceValidity.FILLER),
            ("TC-009", "嗯啊", VoiceValidity.FILLER),
            ("TC-010", "那个", VoiceValidity.FILLER),
        ]

        print("\n[测试组2] 语气助词过滤测试")
        print("-" * 50)

        for case_id, text, expected in test_cases:
            result = self._run_validity_test(case_id, text, expected)
            self.results.append(result)

    def _test_mixed_content(self):
        """测试混合内容"""
        test_cases = [
            ("TC-011", "嗯帮我", VoiceValidity.VALID),  # 包含有效关键词
            ("TC-012", "那个什么", VoiceValidity.VALID),  # 包含有效关键词
            ("TC-013", "x", VoiceValidity.PENDING),  # 单个非关键词字符
        ]

        print("\n[测试组3] 混合内容测试")
        print("-" * 50)

        for case_id, text, expected in test_cases:
            result = self._run_validity_test(case_id, text, expected)
            self.results.append(result)

    def _run_validity_test(self, case_id: str, text: str, expected: str) -> TestResult:
        """运行有效性测试"""
        result = TestResult(case_id, f"有效人声检测: '{text}'")

        start_time = time.time() * 1000
        actual = self.processor.check_voice_validity(text)
        end_time = time.time() * 1000

        result.actual = actual
        result.expected = expected
        result.duration_ms = end_time - start_time
        result.passed = (actual == expected)

        status = "PASS" if result.passed else "FAIL"
        print(f"  {case_id}: '{text}' -> {actual} (期望: {expected}) [{status}] ({result.duration_ms:.2f}ms)")

        return result

    def _print_summary(self):
        """输出测试汇总"""
        print("\n" + "=" * 70)
        print("测试结果汇总")
        print("=" * 70)

        passed_count = sum(1 for r in self.results if r.passed)
        total_count = len(self.results)

        print(f"\n测试用例总数: {total_count}")
        print(f"通过: {passed_count}")
        print(f"失败: {total_count - passed_count}")

        # 性能指标
        max_duration = max(r.duration_ms for r in self.results)
        avg_duration = sum(r.duration_ms for r in self.results) / total_count

        print(f"\n性能指标:")
        print(f"  最大延迟: {max_duration:.2f}ms")
        print(f"  平均延迟: {avg_duration:.2f}ms")

        # 检查是否满足要求
        print(f"\n性能要求检查:")
        print(f"  有效人声判断延迟 < {self.VALID_VOICE_LATENCY_LIMIT_MS}ms: {'PASS' if max_duration < self.VALID_VOICE_LATENCY_LIMIT_MS else 'FAIL'}")

        # 失败用例
        failed = [r for r in self.results if not r.passed]
        if failed:
            print(f"\n失败用例:")
            for r in failed:
                print(f"  {r.case_id}: {r.name}")
                print(f"    实际: {r.actual}, 期望: {r.expected}")

        print("\n" + "=" * 70)

    def _all_passed(self) -> bool:
        """是否全部通过"""
        return all(r.passed for r in self.results)


class LatencyBenchmark:
    """时延基准测试"""

    def __init__(self):
        self.processor = SemanticVADProcessor()

    def run_benchmark(self, iterations: int = 100):
        """运行基准测试"""
        print("\n" + "=" * 70)
        print("时延基准测试")
        print("=" * 70)
        print(f"迭代次数: {iterations}")

        test_texts = [
            "帮我",
            "帮我介绍下三国演义",
            "什么",
            "嗯",
            "帮我介绍下西游记，里面的孙悟空很厉害，请介绍下",
        ]

        results = {}

        for text in test_texts:
            latencies = []
            for _ in range(iterations):
                start = time.time() * 1000
                self.processor.check_voice_validity(text)
                end = time.time() * 1000
                latencies.append(end - start)

            avg = sum(latencies) / len(latencies)
            max_l = max(latencies)
            min_l = min(latencies)

            results[text[:20]] = {
                "avg": avg,
                "max": max_l,
                "min": min_l
            }

            print(f"\n'{text[:20]}...'")
            print(f"  平均: {avg:.3f}ms")
            print(f"  最大: {max_l:.3f}ms")
            print(f"  最小: {min_l:.3f}ms")

        # 总体统计
        all_avgs = [r["avg"] for r in results.values()]
        print(f"\n总体统计:")
        print(f"  所有文本平均延迟: {sum(all_avgs) / len(all_avgs):.3f}ms")
        print(f"  最大平均延迟: {max(all_avgs):.3f}ms")

        print("\n" + "=" * 70)


def main():
    """主函数"""
    # 运行测试套件
    suite = InterruptTestSuite()
    all_passed = suite.run_all()

    # 运行基准测试
    benchmark = LatencyBenchmark()
    benchmark.run_benchmark(100)

    # 返回退出码
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()