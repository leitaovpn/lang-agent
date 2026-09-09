"""core.tool：工具实现与注册表。

内置演示工具（tools）与注册表（tool_registry）；
未来工具鉴权在此包新增模块。
"""
from .tool_registry import (
    ToolSpec,
    get_tool,
    instantiate_tools,
    register_tool,
    unregister_tool,
)

__all__ = ["ToolSpec", "get_tool", "instantiate_tools", "register_tool", "unregister_tool"]
