"""
用户身份识别模块
根据用户的语言特征识别用户是成人还是儿童
"""
import re
from enum import Enum
from typing import Dict, List, Any, Optional
from ..core.logger import logger


class UserType(Enum):
    """用户类型枚举"""
    ADULT = "adult"      # 成人
    CHILD = "child"      # 儿童
    UNKNOWN = "unknown"  # 未知


class UserProfile:
    """用户画像类"""
    
    def __init__(self):
        self.user_type: UserType = UserType.UNKNOWN
        self.confidence: float = 0.0  # 识别置信度
        self.evidence: List[str] = []  # 识别依据
        self.history: List[Dict[str, Any]] = []  # 历史对话记录（用于持续学习）
    
    def update(self, user_type: UserType, confidence: float, evidence: List[str]):
        """更新用户画像"""
        self.user_type = user_type
        self.confidence = confidence
        self.evidence = evidence
        logger.info(f"[用户画像] 更新: {user_type.value}, 置信度: {confidence:.2f}, 依据: {evidence}")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "user_type": self.user_type.value,
            "confidence": self.confidence,
            "evidence": self.evidence
        }


class UserIdentityRecognizer:
    """
    用户身份识别器
    基于语言特征、词汇选择、句式结构等判断用户是成人还是儿童
    """
    
    # 儿童特征关键词
    CHILD_KEYWORDS = [
        # 称呼类
        "妈妈", "爸爸", "爷爷", "奶奶", "姥姥", "姥爷", "叔叔", "阿姨",
        "老师", "小朋友", "小伙伴", "同学",
        
        # 童趣表达
        "呀", "呢", "啦", "哇", "哦", "耶", "嘿嘿", "嘻嘻", "哈哈",
        "好棒", "好厉害", "好开心", "好喜欢", "好想", "好想玩",
        "真好玩", "真有趣", "真好看", "真可爱",
        
        # 儿童常见话题
        "玩具", "动画片", "卡通", "奥特曼", "小猪佩奇", "汪汪队",
        "幼儿园", "小学", "作业", "考试", "成绩", "老师",
        "游戏", "玩耍", "捉迷藏", "积木", "乐高",
        "糖果", "零食", "冰淇淋", "巧克力",
        "宠物", "小狗", "小猫", "小兔子",
        
        # 简单句式标记
        "为什么", "是什么", "怎么", "哪里", "谁",
        "我想", "我要", "我可以", "能不能",
        
        # 儿童语气词
        "嘛", "呗", "啦", "喽", "哩", "咯",
        "好不好", "行不行", "可以吗", "对不对",
    ]
    
    # 成人特征关键词
    ADULT_KEYWORDS = [
        # 工作相关
        "工作", "公司", "会议", "项目", "报告", "方案", "计划",
        "客户", "领导", "同事", "团队", "部门", "经理", "老板",
        "加班", "出差", "任务", "进度", "效率", "业绩",
        "写报告", "开会", "汇报", "总结", "规划",
        
        # 生活相关
        "房贷", "车贷", "理财", "投资", "保险", "税务",
        "孩子", "教育", "学区", "培训班", "辅导班",
        "健康", "体检", "医院", "医生", "药品",
        "购物", "网购", "快递", "外卖",
        
        # 成人表达
        "麻烦", "困扰", "压力", "焦虑", "疲惫", "无奈",
        "考虑", "思考", "分析", "评估", "决策",
        "建议", "意见", "看法", "观点", "想法",
        "其实", "实际上", "说实话", "坦白说",
        
        # 专业词汇
        "系统", "平台", "功能", "模块", "接口", "数据",
        "政策", "法规", "条例", "规定", "制度",
        "市场", "行业", "趋势", "竞争", "发展",
    ]
    
    # 儿童句式模式
    CHILD_PATTERNS = [
        r'.*[吗呢吧呀啦哇哦耶].{0,2}$',  # 句尾语气词
        r'^(为什么|怎么|什么|哪里|谁).+',  # 疑问句
        r'.*(好不好|行不行|可以吗|对不对).*',  # 征求意见
        r'^(我想|我要|我可以).+',  # 表达意愿
        r'.*(好棒|好厉害|好开心|好喜欢).*',  # 情感表达
        r'^.{1,10}$',  # 短句（儿童倾向说短句）
    ]
    
    # 成人句式模式
    ADULT_PATTERNS = [
        r'.*(其实|实际上|说实话|坦白说).*',  # 成人表达习惯
        r'.*(考虑|思考|分析|评估).*',  # 理性表达
        r'.*(建议|意见|看法|观点).*',  # 表达观点
        r'.*(工作|公司|会议|项目).*',  # 工作话题
        r'.*(房贷|车贷|理财|投资).*',  # 成人话题
        r'^.{20,}$',  # 长句（成人倾向说长句）
    ]
    
    def __init__(self):
        self.profile = UserProfile()
    
    def recognize(self, text: str) -> UserProfile:
        """
        识别用户身份
        
        Args:
            text: 用户输入文本
            
        Returns:
            UserProfile: 用户画像
        """
        if not text or len(text.strip()) == 0:
            return self.profile
        
        text = text.strip()
        
        # 计算儿童特征得分
        child_score, child_evidence = self._calculate_child_score(text)
        
        # 计算成人特征得分
        adult_score, adult_evidence = self._calculate_adult_score(text)
        
        # 综合判断
        total_score = child_score + adult_score
        if total_score == 0:
            # 没有明确特征，保持当前状态或设为未知
            if self.profile.user_type == UserType.UNKNOWN:
                self.profile.update(UserType.UNKNOWN, 0.0, ["无明显特征"])
            return self.profile
        
        # 计算置信度
        if child_score > adult_score:
            confidence = child_score / total_score
            user_type = UserType.CHILD
            evidence = child_evidence
        elif adult_score > child_score:
            confidence = adult_score / total_score
            user_type = UserType.ADULT
            evidence = adult_evidence
        else:
            # 得分相同，保持当前状态
            return self.profile
        
        # 更新用户画像
        self.profile.update(user_type, confidence, evidence)
        
        # 记录到历史（用于持续学习）
        self.profile.history.append({
            "text": text,
            "user_type": user_type.value,
            "confidence": confidence,
            "evidence": evidence
        })
        
        # 限制历史长度
        if len(self.profile.history) > 20:
            self.profile.history = self.profile.history[-20:]
        
        return self.profile
    
    def _calculate_child_score(self, text: str) -> tuple:
        """计算儿童特征得分"""
        score = 0.0
        evidence = []
        
        # 关键词匹配
        child_keyword_count = 0
        for keyword in self.CHILD_KEYWORDS:
            if keyword in text:
                child_keyword_count += 1
                if len(evidence) < 5:  # 限制证据数量
                    evidence.append(f"儿童词汇: {keyword}")
        
        # 关键词得分（最多3分）
        score += min(child_keyword_count * 0.3, 3.0)
        
        # 句式模式匹配
        child_pattern_count = 0
        for pattern in self.CHILD_PATTERNS:
            if re.match(pattern, text):
                child_pattern_count += 1
                if len(evidence) < 5:
                    evidence.append(f"儿童句式: {pattern}")
        
        # 句式得分（最多2分）
        score += min(child_pattern_count * 0.5, 2.0)
        
        # 句子长度特征
        if len(text) <= 10:
            score += 0.5
            if len(evidence) < 5:
                evidence.append("短句特征")
        
        return score, evidence
    
    def _calculate_adult_score(self, text: str) -> tuple:
        """计算成人特征得分"""
        score = 0.0
        evidence = []
        
        # 关键词匹配
        adult_keyword_count = 0
        for keyword in self.ADULT_KEYWORDS:
            if keyword in text:
                adult_keyword_count += 1
                if len(evidence) < 5:
                    evidence.append(f"成人词汇: {keyword}")
        
        # 关键词得分（最多4分，提高权重）
        score += min(adult_keyword_count * 0.5, 4.0)
        
        # 句式模式匹配
        adult_pattern_count = 0
        for pattern in self.ADULT_PATTERNS:
            if re.match(pattern, text):
                adult_pattern_count += 1
                if len(evidence) < 5:
                    evidence.append(f"成人句式: {pattern}")
        
        # 句式得分（最多2分）
        score += min(adult_pattern_count * 0.5, 2.0)
        
        # 句子长度特征
        if len(text) >= 20:
            score += 0.5
            if len(evidence) < 5:
                evidence.append("长句特征")
        
        return score, evidence
    
    def get_user_type(self) -> UserType:
        """获取当前用户类型"""
        return self.profile.user_type
    
    def get_confidence(self) -> float:
        """获取识别置信度"""
        return self.profile.confidence
    
    def get_profile(self) -> UserProfile:
        """获取用户画像"""
        return self.profile
    
    def reset(self):
        """重置用户画像"""
        self.profile = UserProfile()
        logger.info("[用户画像] 已重置")
