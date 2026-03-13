"""
用户身份识别测试 - 输出到文件
"""
import sys
import os

# 添加项目路径
project_root = r"c:\Users\admin\Desktop\c00520203\testChatBot\vedio"
sys.path.insert(0, project_root)

# 导入模块
from src.voice_dialog.modules.user_profile import UserIdentityRecognizer, UserType

# 创建输出文件
output_file = os.path.join(project_root, "test_result.txt")

with open(output_file, "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write("用户身份识别功能测试\n")
    f.write("=" * 60 + "\n\n")
    
    # 创建识别器
    recognizer = UserIdentityRecognizer()
    
    # 测试儿童特征
    f.write("【测试儿童特征】\n")
    child_texts = [
        "妈妈，我今天好开心呀！",
        "爸爸，为什么天是蓝色的呢？",
        "哇！这个玩具好棒呀！",
        "我想看小猪佩奇",
    ]
    
    for text in child_texts:
        profile = recognizer.recognize(text)
        f.write(f"输入: {text}\n")
        f.write(f"结果: {profile.user_type.value}, 置信度: {profile.confidence:.2f}\n")
        if profile.evidence:
            f.write(f"依据: {profile.evidence[:2]}\n")
        f.write("\n")
    
    # 重置识别器
    recognizer.reset()
    
    # 测试成人特征
    f.write("\n【测试成人特征】\n")
    adult_texts = [
        "今天公司开会，项目进度有点慢",
        "房贷压力太大了",
        "领导让我写个报告",
        "最近工作压力很大",
    ]
    
    for text in adult_texts:
        profile = recognizer.recognize(text)
        f.write(f"输入: {text}\n")
        f.write(f"结果: {profile.user_type.value}, 置信度: {profile.confidence:.2f}\n")
        if profile.evidence:
            f.write(f"依据: {profile.evidence[:2]}\n")
        f.write("\n")
    
    f.write("=" * 60 + "\n")
    f.write("测试完成！\n")
    f.write("=" * 60 + "\n")

print(f"测试结果已保存到: {output_file}")
