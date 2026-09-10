"""插件协议：独立于具体 loop 的状态与上下文类型。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig

NODE_HOOKS = (
    "before_agent",
    "after_agent",
    "before_model",
    "after_model",
    "before_tool",
    "after_tool",
)
WRAP_HOOKS = ("wrap_model_hook", "wrap_tool_hook")
HOOKS = NODE_HOOKS + WRAP_HOOKS


@dataclass(frozen=True, slots=True)
class HookPause:
    """由外层固定节点持久化的审批请求。"""

    key: str
    payload: Any


@dataclass(frozen=True, slots=True)
class HookResult:
    """hook 的状态增量与受限控制结果。"""

    update: dict[str, Any] = field(default_factory=dict)
    pause: HookPause | None = None
    route: Literal["retry", "end"] | None = None


@dataclass(frozen=True, slots=True)
class HookRuntime:
    """插件获得的运行信息；答案以当前插件的暂停 key 索引。"""

    agent_id: str
    context: Any
    config: RunnableConfig
    hook_name: str
    plugin_name: str
    revision: str
    run_id: str
    answers: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """一次模型调用的发送视图，修改时创建副本。"""

    llm: Any
    tools: list[Any]
    messages: list[BaseMessage]
    config: RunnableConfig
    runtime: Any

    def override(self, **changes: Any) -> ModelRequest:
        return replace(self, **changes)


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """写入 checkpoint 前的普通 AIMessage。"""

    message: AIMessage | None

    def override(self, **changes: Any) -> ModelResponse:
        return replace(self, **changes)


type AsyncModelHandler = Callable[[ModelRequest], Awaitable[ModelResponse]]
type SyncModelHandler = Callable[[ModelRequest], ModelResponse]


class PluginError(ValueError):
    """插件定义或运行契约不满足。"""


class PluginAgentMismatch(PluginError):
    """插件或上下文属于另一个 agent。"""


class PluginRevisionUnavailable(PluginError):
    """恢复所需的历史版本尚未加载。"""
