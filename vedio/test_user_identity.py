"""
用户身份识别功能测试脚本
测试成人/儿童识别和差异化回复功能
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.voice_dialog.modules.user_profile import UserIdentityRecognizer, UserType


def test_user_identity_recognition():
    """测试用户身份识别"""
    recognizer = UserIdentityRecognizer()
    
    print("=" * 60)
    print("用户身份识别功能测试")
    print("=" * 60)
    
    # 测试用例
    test_cases = [
        # 儿童特征明显的句子
        ("妈妈，我今天在学校好开心呀！", "儿童"),
        ("爸爸，为什么天是蓝色的呢？", "儿童"),
        ("老师，我可以去上厕所吗？", "儿童"),
        ("哇！这个玩具好棒呀！", "儿童"),
        ("我想看小猪佩奇动画片", "儿童"),
        ("奶奶，给我讲个故事吧", "儿童"),
        ("好想玩积木呀", "儿童"),
        ("嘿嘿，今天真好玩", "儿童"),
        
        # 成人特征明显的句子
        ("今天公司开会，项目进度有点慢", "成人"),
        ("房贷压力太大了，每个月都要还很多钱", "成人"),
        ("领导让我写个报告，明天要交", "成人"),
        ("最近工作压力很大，经常加班", "成人"),
        ("孩子的教育问题让我很焦虑", "成人"),
        ("我在考虑要不要换个理财产品", "成人"),
        ("这个方案需要再评估一下", "成人"),
        ("说实话，这个政策对我们影响挺大的", "成人"),
        
        # 中性句子
        ("今天天气怎么样？", "未知"),
        ("帮我查一下时间", "未知"),
        ("播放一首歌", "未知"),
    ]
    
    print("\n测试结果：\n")
    correct_count = 0
    total_count = len(test_cases)
    
    for text, expected_type in test_cases:
        profile = recognizer.recognize(text)
        user_type = profile.user_type
        confidence = profile.confidence
        evidence = profile.evidence
        
        # 判断是否正确
        is_correct = False
        if expected_type == "儿童" and user_type == UserType.CHILD:
            is_correct = True
        elif expected_type == "成人" and user_type == UserType.ADULT:
            is_correct = True
        elif expected_type == "未知" and user_type == UserType.UNKNOWN:
            is_correct = True
        
        if is_correct:
            correct_count += 1
        
        # 打印结果
        status = "✓" if is_correct else "✗"
        print(f"{status} 输入: {text}")
        print(f"  预期: {expected_type}, 实际: {user_type.value}, 置信度: {confidence:.2f}")
        if evidence:
            print(f"  依据: {', '.join(evidence[:3])}")
        print()
    
    # 打印统计
    accuracy = correct_count / total_count * 100
    print("=" * 60)
    print(f"测试完成！准确率: {accuracy:.1f}% ({correct_count}/{total_count})")
    print("=" * 60)
    
    # 测试持续学习
    print("\n测试持续学习能力：")
    print("-" * 60)
    
    # 模拟多轮对话
    child_dialogs = [
        "妈妈，我想吃冰淇淋",
        "好呀好呀，我最喜欢了",
        "嘿嘿，谢谢妈妈",
    ]
    
    for text in child_dialogs:
        profile = recognizer.recognize(text)
        print(f"输入: {text}")
        print(f"识别结果: {profile.user_type.value}, 置信度: {profile.confidence:.2f}")
        print()
    
    print("最终用户画像：")
    final_profile = recognizer.get_profile()
    print(f"用户类型: {final_profile.user_type.value}")
    print(f"置信度: {final_profile.confidence:.2f}")
    print(f"历史记录数: {len(final_profile.history)}")


def test_user_identity_context():
    """测试用户身份上下文构建"""
    from src.voice_dialog.modules.llm_planner import LLMTaskPlanner
    
    planner = LLMTaskPlanner()
    
    print("\n" + "=" * 60)
    print("用户身份上下文构建测试")
    print("=" * 60)
    
    # 测试不同用户类型的上下文
    test_cases = [
        (UserType.CHILD, 0.85),
        (UserType.ADULT, 0.90),
        (UserType.UNKNOWN, 0.0),
    ]
    
    for user_type, confidence in test_cases:
        context = planner._build_user_identity_context(user_type, confidence)
        print(f"\n用户类型: {user_type.value}, 置信度: {confidence:.2f}")
        print("-" * 60)
        print(context)
        print()


if __name__ == "__main__":
    test_user_identity_recognition()
    test_user_identity_context()
