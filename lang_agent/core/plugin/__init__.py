"""按 agent 隔离的插件注册、生命周期和调用拦截协议。"""

from .base import PluginBase
from .registry import PluginRegistry, PluginSpecSnapshot
from .types import (
    HookPause,
    HookResult,
    HookRuntime,
    ModelRequest,
    ModelResponse,
    PluginAgentMismatch,
    PluginError,
    PluginRevisionUnavailable,
)

__all__ = [
    "HookPause",
    "HookResult",
    "HookRuntime",
    "ModelRequest",
    "ModelResponse",
    "PluginAgentMismatch",
    "PluginBase",
    "PluginError",
    "PluginRegistry",
    "PluginRevisionUnavailable",
    "PluginSpecSnapshot",
]
