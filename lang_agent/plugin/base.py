"""插件基类；未覆盖的方法不参与注册。"""

import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from .types import HookResult, HookRuntime


class PluginBase:
    """配置与版本必须保持稳定，请求数据放入 plugin_state。"""

    name: str = ""
    version: str = "1"
    parent_plugins: tuple[str, ...] = ()
    config: Mapping[str, Any] = MappingProxyType({})
    requires_buffered_output: bool = False

    def validate_answer(self, key: str, payload: Any, answer: Any) -> None:
        """纯校验：在 Command 写入 checkpoint 前拒绝无效答案。"""
        json.dumps(answer, allow_nan=False)

    def before_agent(self, state: Any, runtime: HookRuntime) -> HookResult | None:
        raise NotImplementedError

    def after_agent(self, state: Any, runtime: HookRuntime) -> HookResult | None:
        raise NotImplementedError

    def before_model(self, state: Any, runtime: HookRuntime) -> HookResult | None:
        raise NotImplementedError

    def after_model(self, state: Any, runtime: HookRuntime) -> HookResult | None:
        raise NotImplementedError

    def before_tool(self, state: Any, runtime: HookRuntime) -> HookResult | None:
        raise NotImplementedError

    def after_tool(self, state: Any, runtime: HookRuntime) -> HookResult | None:
        raise NotImplementedError

    async def abefore_agent(
        self, state: Any, runtime: HookRuntime
    ) -> HookResult | None:
        raise NotImplementedError

    async def aafter_agent(self, state: Any, runtime: HookRuntime) -> HookResult | None:
        raise NotImplementedError

    async def abefore_model(
        self, state: Any, runtime: HookRuntime
    ) -> HookResult | None:
        raise NotImplementedError

    async def aafter_model(self, state: Any, runtime: HookRuntime) -> HookResult | None:
        raise NotImplementedError

    async def abefore_tool(self, state: Any, runtime: HookRuntime) -> HookResult | None:
        raise NotImplementedError

    async def aafter_tool(self, state: Any, runtime: HookRuntime) -> HookResult | None:
        raise NotImplementedError

    def wrap_model_hook(self, request: Any, handler: Any) -> Any:
        raise NotImplementedError

    async def awrap_model_hook(self, request: Any, handler: Any) -> Any:
        raise NotImplementedError

    def wrap_tool_hook(self, request: Any, handler: Any) -> Any:
        raise NotImplementedError

    async def awrap_tool_hook(self, request: Any, handler: Any) -> Any:
        raise NotImplementedError
