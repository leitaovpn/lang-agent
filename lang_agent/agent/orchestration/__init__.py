"""agent.orchestration：会话级编排（repair/compress/retry/结果塑形/事件分类）。

未来多 agent/多会话编排在此包新增模块。
"""
from .session import ChatSession

__all__ = ["ChatSession"]
