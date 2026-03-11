"""
全双工语音对话系统 v3.0 - 状态机
"""
import asyncio
from typing import Callable, Optional, Dict, List

from .logger import logger
from .types import DialogState, SemanticState


class DialogStateMachine:
    """对话状态机"""

    # 状态转换规则
    TRANSITIONS = {
        DialogState.IDLE: [DialogState.LISTENING],
        DialogState.LISTENING: [DialogState.PROCESSING, DialogState.IDLE],
        DialogState.PROCESSING: [DialogState.THINKING, DialogState.LISTENING, DialogState.IDLE],
        DialogState.THINKING: [DialogState.SPEAKING, DialogState.LISTENING, DialogState.IDLE],
        DialogState.SPEAKING: [DialogState.LISTENING, DialogState.IDLE],
    }

    def __init__(self):
        self._state = DialogState.IDLE
        self._lock = asyncio.Lock()
        self._listeners: List[Callable] = []
        self._history: List[Dict] = []

    @property
    def state(self) -> DialogState:
        return self._state

    async def transition_to(self, new_state: DialogState, reason: str = "") -> bool:
        """状态转换"""
        async with self._lock:
            if new_state in self.TRANSITIONS.get(self._state, []):
                old_state = self._state
                self._state = new_state

                # 记录历史
                self._history.append({
                    "from": old_state.value,
                    "to": new_state.value,
                    "reason": reason
                })

                logger.debug(f"状态转换: {old_state.value} -> {new_state.value} ({reason})")

                # 通知监听器
                await self._notify_listeners(old_state, new_state)
                return True
            else:
                logger.warning(f"无效状态转换: {self._state.value} -> {new_state.value}")
                return False

    async def force_state(self, new_state: DialogState, reason: str = ""):
        """强制设置状态（用于打断等场景）"""
        async with self._lock:
            old_state = self._state
            self._state = new_state

            self._history.append({
                "from": old_state.value,
                "to": new_state.value,
                "reason": f"FORCED: {reason}"
            })

            logger.info(f"强制状态转换: {old_state.value} -> {new_state.value} ({reason})")
            await self._notify_listeners(old_state, new_state)

    def add_listener(self, callback: Callable):
        """添加状态变化监听器"""
        self._listeners.append(callback)

    def remove_listener(self, callback: Callable):
        """移除监听器"""
        if callback in self._listeners:
            self._listeners.remove(callback)

    async def _notify_listeners(self, old_state: DialogState, new_state: DialogState):
        """通知所有监听器"""
        for listener in self._listeners:
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener(old_state, new_state)
                else:
                    listener(old_state, new_state)
            except Exception as e:
                logger.error(f"状态监听器错误: {e}")

    def can_interrupt(self) -> bool:
        """是否可以打断

        v3.4 更新：扩展到 THINKING 状态，允许在 LLM 推理期间打断
        """
        return self._state in [DialogState.SPEAKING, DialogState.THINKING]

    def is_busy(self) -> bool:
        """是否忙碌"""
        return self._state in [DialogState.PROCESSING, DialogState.THINKING, DialogState.SPEAKING]

    def get_interruptible_states(self) -> list:
        """获取可被打断的状态列表"""
        return [DialogState.SPEAKING, DialogState.THINKING]

    def reset(self):
        """重置状态机"""
        self._state = DialogState.IDLE
        self._history.clear()
        logger.info("状态机已重置")

    def get_history(self, limit: int = 10) -> List[Dict]:
        """获取状态历史"""
        return self._history[-limit:]


class SemanticStateMachine:
    """语义状态机"""

    def __init__(self):
        self._state = SemanticState.UNKNOWN
        self._confidence = 0.0

    @property
    def state(self) -> SemanticState:
        return self._state

    @property
    def confidence(self) -> float:
        return self._confidence

    def update(self, new_state: SemanticState, confidence: float = 1.0):
        """更新语义状态"""
        self._state = new_state
        self._confidence = confidence

    def should_process(self) -> bool:
        """是否应该处理（用户说完）"""
        return self._state == SemanticState.COMPLETE

    def is_continuing(self) -> bool:
        """用户是否继续说"""
        return self._state == SemanticState.CONTINUING

    def is_interrupted(self) -> bool:
        """是否被打断"""
        return self._state == SemanticState.INTERRUPTED

    def reset(self):
        """重置"""
        self._state = SemanticState.UNKNOWN
        self._confidence = 0.0
