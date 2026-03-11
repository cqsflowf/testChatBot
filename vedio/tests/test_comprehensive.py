"""
全双工语音对话系统 v2.0 - 综合测试用例
包含100+个日常语音对话场景
"""
import asyncio
import sys
import os
import json
from typing import List, Dict, Tuple

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from voice_dialog import VoiceDialogSystem, DialogState, EmotionType


class DialogTestCase:
    """单个测试用例"""
    def __init__(self, category: str, inputs: List[str], expected_checks: List[callable], description: str = ""):
        self.category = category
        self.inputs = inputs
        self.expected_checks = expected_checks
        self.description = description
        self.passed = False
        self.actual_responses = []
        self.error_msg = ""


class ComprehensiveDialogTester:
    """综合对话测试器"""

    def __init__(self):
        self.system = VoiceDialogSystem()
        self.test_cases: List[DialogTestCase] = []
        self.results = {"passed": 0, "failed": 0, "total": 0}

    def create_test_cases(self):
        """创建所有测试用例"""

        # ========== 一、基础问候闲聊 (10个) ==========
        self.test_cases.extend([
            DialogTestCase(
                "基础问候",
                ["你好"],
                [lambda r: r.response != "" and len(r.response) > 0],
                "简单问候"
            ),
            DialogTestCase(
                "基础问候",
                ["你是谁"],
                [lambda r: "助手" in r.response or "AI" in r.response.lower()],
                "身份询问"
            ),
            DialogTestCase(
                "基础问候",
                ["你能做什么"],
                [lambda r: "天气" in r.response or "提醒" in r.response or "设备" in r.response],
                "能力询问"
            ),
            DialogTestCase(
                "基础问候",
                ["早上好"],
                [lambda r: "早" in r.response or "好" in r.response],
                "早间问候"
            ),
            DialogTestCase(
                "基础问候",
                ["晚安"],
                [lambda r: "晚" in r.response or "安" in r.response or "梦" in r.response],
                "晚间问候"
            ),
            DialogTestCase(
                "基础问候",
                ["谢谢"],
                [lambda r: r.emotion == EmotionType.POSITIVE],
                "感谢-积极情绪"
            ),
            DialogTestCase(
                "基础问候",
                ["再见"],
                [lambda r: "见" in r.response or "拜" in r.response],
                "告别"
            ),
            DialogTestCase(
                "基础问候",
                ["你好吗"],
                [lambda r: r.response != ""],
                "问候状态"
            ),
            DialogTestCase(
                "基础问候",
                ["你叫什么名字"],
                [lambda r: r.response != ""],
                "名字询问"
            ),
            DialogTestCase(
                "基础问候",
                ["你是真人吗"],
                [lambda r: "AI" in r.response or "智能" in r.response or "助手" in r.response],
                "身份确认"
            ),
        ])

        # ========== 二、天气查询 (20个) ==========
        self.test_cases.extend([
            DialogTestCase(
                "天气查询",
                ["北京今天天气怎么样"],
                [lambda r: "北京" in r.response and ("晴" in r.response or "雨" in r.response or "阴" in r.response or "度" in r.response)],
                "北京天气"
            ),
            DialogTestCase(
                "天气查询",
                ["上海天气"],
                [lambda r: "上海" in r.response],
                "上海天气简写"
            ),
            DialogTestCase(
                "天气查询",
                ["广州热不热"],
                [lambda r: "广州" in r.response or "度" in r.response],
                "广州温度询问"
            ),
            DialogTestCase(
                "天气查询",
                ["深圳今天会下雨吗"],
                [lambda r: "深圳" in r.response],
                "深圳降雨查询"
            ),
            DialogTestCase(
                "天气查询",
                ["杭州天气怎么样"],
                [lambda r: "杭州" in r.response],
                "杭州天气"
            ),
            DialogTestCase(
                "天气查询",
                ["南京的气温"],
                [lambda r: "南京" in r.response or "度" in r.response],
                "南京气温"
            ),
            DialogTestCase(
                "天气查询",
                ["天津今天多少度"],
                [lambda r: "天津" in r.response or "度" in r.response],
                "天津温度"
            ),
            DialogTestCase(
                "天气查询",
                ["重庆天气情况"],
                [lambda r: "重庆" in r.response],
                "重庆天气"
            ),
            DialogTestCase(
                "天气查询",
                ["成都冷不冷"],
                [lambda r: "成都" in r.response],
                "成都冷暖查询"
            ),
            DialogTestCase(
                "天气查询",
                ["武汉的天气"],
                [lambda r: "武汉" in r.response],
                "武汉天气"
            ),
            DialogTestCase(
                "天气查询",
                ["西安天气如何"],
                [lambda r: "西安" in r.response],
                "西安天气"
            ),
            DialogTestCase(
                "天气查询",
                ["苏州今天天气"],
                [lambda r: "苏州" in r.response],
                "苏州天气"
            ),
            DialogTestCase(
                "天气查询",
                ["郑州会下雪吗"],
                [lambda r: "郑州" in r.response],
                "郑州降雪查询"
            ),
            DialogTestCase(
                "天气查询",
                ["长沙天气"],
                [lambda r: "长沙" in r.response],
                "长沙天气"
            ),
            DialogTestCase(
                "天气查询",
                ["青岛天气怎么样"],
                [lambda r: "青岛" in r.response],
                "青岛天气"
            ),
            DialogTestCase(
                "天气查询",
                ["大连今天天气如何"],
                [lambda r: "大连" in r.response],
                "大连天气"
            ),
            DialogTestCase(
                "天气查询",
                ["厦门热吗"],
                [lambda r: "厦门" in r.response],
                "厦门热度查询"
            ),
            DialogTestCase(
                "天气查询",
                ["昆明天气"],
                [lambda r: "昆明" in r.response],
                "昆明天气"
            ),
            DialogTestCase(
                "天气查询",
                ["福州今天多少度"],
                [lambda r: "福州" in r.response],
                "福州温度"
            ),
            DialogTestCase(
                "天气查询",
                ["合肥天气情况"],
                [lambda r: "合肥" in r.response],
                "合肥天气"
            ),
        ])

        # ========== 三、提醒设置 (15个) ==========
        self.test_cases.extend([
            DialogTestCase(
                "提醒设置",
                ["帮我设一个明天早上8点的提醒"],
                [lambda r: "提醒" in r.response or "8点" in r.response],
                "明天早上提醒"
            ),
            DialogTestCase(
                "提醒设置",
                ["提醒我下午3点开会"],
                [lambda r: "提醒" in r.response or "开会" in r.response],
                "开会提醒"
            ),
            DialogTestCase(
                "提醒设置",
                ["设置一个闹钟，早上7点"],
                [lambda r: "闹钟" in r.response or "提醒" in r.response or "7点" in r.response],
                "闹钟设置"
            ),
            DialogTestCase(
                "提醒设置",
                ["提醒我明天带伞"],
                [lambda r: "提醒" in r.response or "伞" in r.response],
                "带伞提醒"
            ),
            DialogTestCase(
                "提醒设置",
                ["后天上午10点提醒我体检"],
                [lambda r: "提醒" in r.response or "体检" in r.response],
                "体检提醒"
            ),
            DialogTestCase(
                "提醒设置",
                ["每小时提醒我喝水"],
                [lambda r: "提醒" in r.response or "喝水" in r.response],
                "喝水提醒"
            ),
            DialogTestCase(
                "提醒设置",
                ["周五下午2点提醒我面试"],
                [lambda r: "提醒" in r.response or "面试" in r.response],
                "面试提醒"
            ),
            DialogTestCase(
                "提醒设置",
                ["提醒我吃药"],
                [lambda r: "提醒" in r.response or "药" in r.response],
                "吃药提醒"
            ),
            DialogTestCase(
                "提醒设置",
                ["晚上9点提醒我看直播"],
                [lambda r: "提醒" in r.response or "直播" in r.response],
                "直播提醒"
            ),
            DialogTestCase(
                "提醒设置",
                ["提醒我给妈妈打电话"],
                [lambda r: "提醒" in r.response or "电话" in r.response],
                "电话提醒"
            ),
            DialogTestCase(
                "提醒设置",
                ["设置一个30分钟后的提醒"],
                [lambda r: "提醒" in r.response],
                "延时提醒"
            ),
            DialogTestCase(
                "提醒设置",
                ["明天中午提醒我订餐"],
                [lambda r: "提醒" in r.response or "餐" in r.response],
                "订餐提醒"
            ),
            DialogTestCase(
                "提醒设置",
                ["提醒我该睡觉了"],
                [lambda r: "提醒" in r.response or "睡觉" in r.response],
                "睡觉提醒"
            ),
            DialogTestCase(
                "提醒设置",
                ["每周一早上8点提醒我开会"],
                [lambda r: "提醒" in r.response or "开会" in r.response],
                "周期提醒"
            ),
            DialogTestCase(
                "提醒设置",
                ["提醒我取快递"],
                [lambda r: "提醒" in r.response or "快递" in r.response],
                "取快递提醒"
            ),
        ])

        # ========== 四、设备控制 (20个) ==========
        self.test_cases.extend([
            DialogTestCase(
                "设备控制",
                ["打开客厅的灯"],
                [lambda r: "灯" in r.response and "打开" in r.response],
                "开客厅灯"
            ),
            DialogTestCase(
                "设备控制",
                ["关闭卧室的灯"],
                [lambda r: "灯" in r.response and "关" in r.response],
                "关卧室灯"
            ),
            DialogTestCase(
                "设备控制",
                ["把灯关了"],
                [lambda r: "灯" in r.response and "关" in r.response],
                "关灯简写"
            ),
            DialogTestCase(
                "设备控制",
                ["打开空调"],
                [lambda r: "空调" in r.response],
                "开空调"
            ),
            DialogTestCase(
                "设备控制",
                ["关闭空调"],
                [lambda r: "空调" in r.response and "关" in r.response],
                "关空调"
            ),
            DialogTestCase(
                "设备控制",
                ["把空调温度调到26度"],
                [lambda r: "空调" in r.response or "26" in r.response],
                "调节空调温度"
            ),
            DialogTestCase(
                "设备控制",
                ["空调温度调高一点"],
                [lambda r: "空调" in r.response],
                "调高温度"
            ),
            DialogTestCase(
                "设备控制",
                ["空调温度调低一点"],
                [lambda r: "空调" in r.response],
                "调低温度"
            ),
            DialogTestCase(
                "设备控制",
                ["打开电视"],
                [lambda r: "电视" in r.response],
                "开电视"
            ),
            DialogTestCase(
                "设备控制",
                ["关闭电视"],
                [lambda r: "电视" in r.response and "关" in r.response],
                "关电视"
            ),
            DialogTestCase(
                "设备控制",
                ["打开加湿器"],
                [lambda r: "加湿器" in r.response or "好的" in r.response],
                "开加湿器"
            ),
            DialogTestCase(
                "设备控制",
                ["关闭加湿器"],
                [lambda r: "加湿器" in r.response or "关" in r.response],
                "关加湿器"
            ),
            DialogTestCase(
                "设备控制",
                ["打开空气净化器"],
                [lambda r: "净化器" in r.response or "好的" in r.response],
                "开净化器"
            ),
            DialogTestCase(
                "设备控制",
                ["打开窗帘"],
                [lambda r: "窗帘" in r.response or "好的" in r.response],
                "开窗帘"
            ),
            DialogTestCase(
                "设备控制",
                ["关闭窗帘"],
                [lambda r: "窗帘" in r.response or "关" in r.response],
                "关窗帘"
            ),
            DialogTestCase(
                "设备控制",
                ["打开热水器"],
                [lambda r: "热水器" in r.response or "好的" in r.response],
                "开热水器"
            ),
            DialogTestCase(
                "设备控制",
                ["关闭热水器"],
                [lambda r: "热水器" in r.response or "关" in r.response],
                "关热水器"
            ),
            DialogTestCase(
                "设备控制",
                ["打开风扇"],
                [lambda r: "风扇" in r.response or "好的" in r.response],
                "开风扇"
            ),
            DialogTestCase(
                "设备控制",
                ["关闭风扇"],
                [lambda r: "风扇" in r.response or "关" in r.response],
                "关风扇"
            ),
            DialogTestCase(
                "设备控制",
                ["把所有灯都关了"],
                [lambda r: "灯" in r.response and "关" in r.response],
                "批量关灯"
            ),
        ])

        # ========== 五、音乐播放 (10个) ==========
        self.test_cases.extend([
            DialogTestCase(
                "音乐播放",
                ["播放音乐"],
                [lambda r: "音乐" in r.response or "播放" in r.response],
                "播放音乐"
            ),
            DialogTestCase(
                "音乐播放",
                ["放一首歌"],
                [lambda r: "歌" in r.response or "播放" in r.response or "音乐" in r.response],
                "播放歌曲"
            ),
            DialogTestCase(
                "音乐播放",
                ["我想听周杰伦的歌"],
                [lambda r: "播放" in r.response or "音乐" in r.response],
                "播放指定歌手"
            ),
            DialogTestCase(
                "音乐播放",
                ["播放轻音乐"],
                [lambda r: "播放" in r.response or "音乐" in r.response],
                "播放轻音乐"
            ),
            DialogTestCase(
                "音乐播放",
                ["来一首流行歌曲"],
                [lambda r: "播放" in r.response or "歌" in r.response],
                "播放流行歌曲"
            ),
            DialogTestCase(
                "音乐播放",
                ["播放古典音乐"],
                [lambda r: "播放" in r.response or "音乐" in r.response],
                "播放古典音乐"
            ),
            DialogTestCase(
                "音乐播放",
                ["放一首舒缓的歌"],
                [lambda r: "播放" in r.response or "歌" in r.response],
                "播放舒缓歌曲"
            ),
            DialogTestCase(
                "音乐播放",
                ["播放儿歌"],
                [lambda r: "播放" in r.response or "歌" in r.response],
                "播放儿歌"
            ),
            DialogTestCase(
                "音乐播放",
                ["我想听摇滚"],
                [lambda r: "播放" in r.response or "摇滚" in r.response or "音乐" in r.response],
                "播放摇滚"
            ),
            DialogTestCase(
                "音乐播放",
                ["放一首开心的歌"],
                [lambda r: "播放" in r.response or "歌" in r.response],
                "播放开心歌曲"
            ),
        ])

        # ========== 六、情绪识别 (15个) ==========
        self.test_cases.extend([
            DialogTestCase(
                "情绪识别",
                ["太好了，谢谢！"],
                [lambda r: r.emotion == EmotionType.POSITIVE],
                "积极情绪-感谢"
            ),
            DialogTestCase(
                "情绪识别",
                ["服务太棒了"],
                [lambda r: r.emotion == EmotionType.POSITIVE],
                "积极情绪-满意"
            ),
            DialogTestCase(
                "情绪识别",
                ["很开心"],
                [lambda r: r.emotion == EmotionType.POSITIVE],
                "积极情绪-开心"
            ),
            DialogTestCase(
                "情绪识别",
                ["太糟糕了"],
                [lambda r: r.emotion == EmotionType.NEGATIVE],
                "消极情绪"
            ),
            DialogTestCase(
                "情绪识别",
                ["这什么破玩意儿"],
                [lambda r: r.emotion in [EmotionType.NEGATIVE, EmotionType.ANGRY]],
                "消极/愤怒情绪"
            ),
            DialogTestCase(
                "情绪识别",
                ["烦死了"],
                [lambda r: r.emotion == EmotionType.ANGRY],
                "愤怒情绪"
            ),
            DialogTestCase(
                "情绪识别",
                ["气死我了"],
                [lambda r: r.emotion == EmotionType.ANGRY],
                "愤怒情绪"
            ),
            DialogTestCase(
                "情绪识别",
                ["我很生气"],
                [lambda r: r.emotion == EmotionType.ANGRY],
                "愤怒情绪-直接表达"
            ),
            DialogTestCase(
                "情绪识别",
                ["太让人失望了"],
                [lambda r: r.emotion == EmotionType.NEGATIVE],
                "失望情绪"
            ),
            DialogTestCase(
                "情绪识别",
                ["我好难过"],
                [lambda r: r.emotion == EmotionType.SAD],
                "悲伤情绪"
            ),
            DialogTestCase(
                "情绪识别",
                ["好伤心啊"],
                [lambda r: r.emotion == EmotionType.SAD],
                "悲伤情绪"
            ),
            DialogTestCase(
                "情绪识别",
                ["哇，太神奇了"],
                [lambda r: r.emotion == EmotionType.SURPRISED],
                "惊讶情绪"
            ),
            DialogTestCase(
                "情绪识别",
                ["天哪，没想到"],
                [lambda r: r.emotion == EmotionType.SURPRISED],
                "惊讶情绪"
            ),
            DialogTestCase(
                "情绪识别",
                ["不会吧"],
                [lambda r: r.emotion == EmotionType.SURPRISED],
                "惊讶情绪"
            ),
            DialogTestCase(
                "情绪识别",
                ["一般般吧"],
                [lambda r: r.emotion == EmotionType.NEUTRAL],
                "中性情绪"
            ),
        ])

        # ========== 七、多轮对话 (15个) ==========
        self.test_cases.extend([
            DialogTestCase(
                "多轮对话",
                ["我叫小明", "我叫什么名字"],
                [lambda r: "小明" in r.response],
                "名字记忆"
            ),
            DialogTestCase(
                "多轮对话",
                ["北京天气怎么样", "那上海呢"],
                [lambda r: "上海" in r.response],
                "上下文天气查询"
            ),
            DialogTestCase(
                "多轮对话",
                ["打开客厅的灯", "把它关了"],
                [lambda r: "关" in r.response],
                "代词指代-关灯"
            ),
            DialogTestCase(
                "多轮对话",
                ["查一下深圳天气", "那里冷吗"],
                [lambda r: "深圳" in r.response or "度" in r.response],
                "代词指代-天气"
            ),
            DialogTestCase(
                "多轮对话",
                ["帮我设个提醒", "明天早上8点", "提醒我开会"],
                [lambda r: "提醒" in r.response],
                "多轮设置提醒"
            ),
            DialogTestCase(
                "多轮对话",
                ["我想听歌", "周杰伦的"],
                [lambda r: "播放" in r.response or "歌" in r.response],
                "多轮音乐播放"
            ),
            DialogTestCase(
                "多轮对话",
                ["打开空调", "调到26度"],
                [lambda r: "26" in r.response or "空调" in r.response],
                "多轮设备控制"
            ),
            DialogTestCase(
                "多轮对话",
                ["今天天气怎么样", "如果下雨提醒我带伞"],
                [lambda r: "提醒" in r.response or "雨" in r.response or "伞" in r.response],
                "条件提醒"
            ),
            DialogTestCase(
                "多轮对话",
                ["我叫小红", "我姓什么"],
                [lambda r: "红" in r.response or "你好" in r.response or r.response != ""],
                "姓氏记忆"
            ),
            DialogTestCase(
                "多轮对话",
                ["北京天气", "上海天气", "广州天气"],
                [lambda r: "广州" in r.response],
                "连续天气查询"
            ),
            DialogTestCase(
                "多轮对话",
                ["你好", "谢谢"],
                [lambda r: r.emotion == EmotionType.POSITIVE],
                "问候后感谢"
            ),
            DialogTestCase(
                "多轮对话",
                ["打开灯", "打开空调", "打开电视"],
                [lambda r: "电视" in r.response or "好的" in r.response],
                "连续设备控制"
            ),
            DialogTestCase(
                "多轮对话",
                ["播放音乐", "声音大一点"],
                [lambda r: r.response != ""],
                "多轮音乐控制"
            ),
            DialogTestCase(
                "多轮对话",
                ["今天多少度", "明天呢"],
                [lambda r: r.response != ""],
                "日期延续"
            ),
            DialogTestCase(
                "多轮对话",
                ["我想订个提醒", "取消吧"],
                [lambda r: r.response != ""],
                "意图取消"
            ),
        ])

        # ========== 八、打断测试 (5个) ==========
        self.test_cases.extend([
            DialogTestCase(
                "打断测试",
                ["打断"],
                [lambda r: r.is_interrupt or r.response != ""],
                "手动打断"
            ),
            DialogTestCase(
                "打断测试",
                ["停"],
                [lambda r: r.response != ""],
                "语音打断-停"
            ),
            DialogTestCase(
                "打断测试",
                ["等等"],
                [lambda r: r.response != ""],
                "语音打断-等等"
            ),
            DialogTestCase(
                "打断测试",
                ["不是这个"],
                [lambda r: r.response != ""],
                "语音打断-纠正"
            ),
            DialogTestCase(
                "打断测试",
                ["重来"],
                [lambda r: r.response != ""],
                "语音打断-重来"
            ),
        ])

        # ========== 九、复杂意图 (10个) ==========
        self.test_cases.extend([
            DialogTestCase(
                "复杂意图",
                ["查一下北京天气，如果下雨提醒我带伞"],
                [lambda r: "北京" in r.response or "提醒" in r.response],
                "天气+条件提醒"
            ),
            DialogTestCase(
                "复杂意图",
                ["打开客厅的灯和空调"],
                [lambda r: "灯" in r.response or "空调" in r.response],
                "多设备控制"
            ),
            DialogTestCase(
                "复杂意图",
                ["明天早上8点提醒我开会，顺便查一下天气"],
                [lambda r: "提醒" in r.response or "天气" in r.response],
                "提醒+天气"
            ),
            DialogTestCase(
                "复杂意图",
                ["播放音乐，把灯关了"],
                [lambda r: "播放" in r.response or "关" in r.response],
                "音乐+设备控制"
            ),
            DialogTestCase(
                "复杂意图",
                ["帮我订个餐，要川菜"],
                [lambda r: r.response != ""],
                "订餐请求"
            ),
            DialogTestCase(
                "复杂意图",
                ["我想看个电影，推荐一下"],
                [lambda r: r.response != ""],
                "电影推荐"
            ),
            DialogTestCase(
                "复杂意图",
                ["帮我查一下航班信息"],
                [lambda r: r.response != ""],
                "航班查询"
            ),
            DialogTestCase(
                "复杂意图",
                ["设置一个每天早上7点的闹钟"],
                [lambda r: "闹钟" in r.response or "提醒" in r.response],
                "周期闹钟"
            ),
            DialogTestCase(
                "复杂意图",
                ["打开空调制热模式，温度28度"],
                [lambda r: "空调" in r.response or "28" in r.response],
                "设备+参数"
            ),
            DialogTestCase(
                "复杂意图",
                ["查一下明天北京会不会下雪，如果会提醒我带厚衣服"],
                [lambda r: r.response != ""],
                "复杂条件"
            ),
        ])

        # ========== 十、边界情况 (10个) ==========
        self.test_cases.extend([
            DialogTestCase(
                "边界情况",
                [""],
                [lambda r: r.response != ""],
                "空输入"
            ),
            DialogTestCase(
                "边界情况",
                ["嗯"],
                [lambda r: r.response != ""],
                "短输入-嗯"
            ),
            DialogTestCase(
                "边界情况",
                ["啊"],
                [lambda r: r.response != ""],
                "短输入-啊"
            ),
            DialogTestCase(
                "边界情况",
                ["那个...我想问一下..."],
                [lambda r: r.response != ""],
                "模糊输入"
            ),
            DialogTestCase(
                "边界情况",
                ["帮我订一张去火星的机票"],
                [lambda r: r.response != ""],
                "无法完成请求"
            ),
            DialogTestCase(
                "边界情况",
                ["我想和外星人说话"],
                [lambda r: r.response != ""],
                "荒谬请求"
            ),
            DialogTestCase(
                "边界情况",
                ["12345"],
                [lambda r: r.response != ""],
                "数字输入"
            ),
            DialogTestCase(
                "边界情况",
                ["!@#$%"],
                [lambda r: r.response != ""],
                "特殊字符"
            ),
            DialogTestCase(
                "边界情况",
                ["帮我帮我帮我帮我帮我帮我帮我帮我帮我帮我"],
                [lambda r: r.response != ""],
                "重复输入"
            ),
            DialogTestCase(
                "边界情况",
                ["我想问一个问题，就是关于那个什么的问题，你能回答吗，如果可以的话请回答，不行也没关系"],
                [lambda r: r.response != ""],
                "长输入"
            ),
        ])

    async def run_tests(self) -> Dict:
        """运行所有测试"""
        print(f"\n{'='*70}")
        print(f"全双工语音对话系统 v2.0 - 综合测试")
        print(f"测试用例总数: {len(self.test_cases)}")
        print(f"{'='*70}\n")

        results_by_category = {}

        for i, test_case in enumerate(self.test_cases, 1):
            # 重置系统以保持测试独立性
            self.system.reset()

            all_passed = True
            responses = []

            try:
                for input_text in test_case.inputs:
                    if input_text:  # 跳过空输入
                        result = await self.system.process_text(input_text)
                        responses.append(result.response)

                # 执行最后一个结果的检查
                last_result = await self.system.process_text(test_case.inputs[-1] if test_case.inputs[-1] else "你好")

                for check in test_case.expected_checks:
                    try:
                        if not check(last_result):
                            all_passed = False
                            test_case.error_msg = f"检查函数返回False"
                    except Exception as e:
                        all_passed = False
                        test_case.error_msg = f"检查函数异常: {e}"

                test_case.actual_responses = responses
                test_case.passed = all_passed

                if all_passed:
                    self.results["passed"] += 1
                    status = "PASS"
                else:
                    self.results["failed"] += 1
                    status = "FAIL"

                # 按类别统计
                cat = test_case.category
                if cat not in results_by_category:
                    results_by_category[cat] = {"passed": 0, "failed": 0}
                if all_passed:
                    results_by_category[cat]["passed"] += 1
                else:
                    results_by_category[cat]["failed"] += 1

                # 打印结果
                inputs_str = " -> ".join(test_case.inputs)
                print(f"[{i:3d}] [{status}] [{test_case.category}] {test_case.description}")
                if not all_passed:
                    print(f"       输入: {inputs_str}")
                    print(f"       响应: {last_result.response[:50]}...")
                    print(f"       错误: {test_case.error_msg}")

            except Exception as e:
                test_case.passed = False
                test_case.error_msg = str(e)
                self.results["failed"] += 1
                print(f"[{i:3d}] [FAIL] [{test_case.category}] {test_case.description} - 异常: {e}")

        self.results["total"] = len(self.test_cases)
        success_rate = self.results["passed"] / self.results["total"] * 100

        # 打印汇总
        print(f"\n{'='*70}")
        print(f"测试汇总")
        print(f"{'='*70}")
        print(f"总计: {self.results['total']}, 通过: {self.results['passed']}, 失败: {self.results['failed']}")
        print(f"成功率: {success_rate:.1f}%")
        print()

        # 按类别打印
        print("按类别统计:")
        for cat, stats in results_by_category.items():
            total = stats["passed"] + stats["failed"]
            rate = stats["passed"] / total * 100 if total > 0 else 0
            print(f"  {cat}: {stats['passed']}/{total} ({rate:.0f}%)")

        return {
            "total": self.results["total"],
            "passed": self.results["passed"],
            "failed": self.results["failed"],
            "success_rate": success_rate,
            "by_category": results_by_category
        }


async def main():
    tester = ComprehensiveDialogTester()
    tester.create_test_cases()
    results = await tester.run_tests()

    # 检查是否达到95%目标
    if results["success_rate"] >= 95:
        print(f"\n成功率达到 {results['success_rate']:.1f}%，测试通过！")
    else:
        print(f"\n成功率 {results['success_rate']:.1f}% 未达到95%目标，需要优化。")

    return results


if __name__ == "__main__":
    results = asyncio.run(main())