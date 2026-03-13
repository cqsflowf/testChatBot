"""
简单的用户身份识别测试
"""
import sys
import os

# 添加项目路径
project_root = r"c:\Users\admin\Desktop\c00520203\testChatBot\vedio"
sys.path.insert(0, project_root)

# 导入模块
from src.voice_dialog.modules.user_profile import UserIdentityRecognizer, UserType

def main():
    print("=" * 60)
    print("用户身份识别功能测试")
    print("=" * 60)
    
    # 创建识别器
    recognizer = UserIdentityRecognizer()
    
    # 测试儿童特征
    print("\n【测试儿童特征】")
    child_texts = [
        "妈妈，我今天好开心呀！",
        "爸爸，为什么天是蓝色的呢？",
        "哇！这个玩具好棒呀！",
        "我想看小猪佩奇",
    ]
    
    for text in child_texts:
        profile = recognizer.recognize(text)
        print(f"输入: {text}")
        print(f"结果: {profile.user_type.value}, 置信度: {profile.confidence:.2f}")
        if profile.evidence:
            print(f"依据: {profile.evidence[:2]}")
        print()
    
    # 重置识别器
    recognizer.reset()
    
    # 测试成人特征
    print("\n【测试成人特征】")
    adult_texts = [
        "今天公司开会，项目进度有点慢",
        "房贷压力太大了",
        "领导让我写个报告",
        "最近工作压力很大",
    ]
    
    for text in adult_texts:
        profile = recognizer.recognize(text)
        print(f"输入: {text}")
        print(f"结果: {profile.user_type.value}, 置信度: {profile.confidence:.2f}")
        if profile.evidence:
            print(f"依据: {profile.evidence[:2]}")
        print()
    
    print("=" * 60)
    print("测试完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
